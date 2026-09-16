"""Model-free, bounded WLA handoff; the existing session remains the only env owner.

This module deliberately has no CUDA/model imports and no independent episode loop.
The predictor receives copies of native camera/proprioceptive observations, never an
environment. Production predictor IPC and agent scheduling are separate boundaries.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading
import time

import numpy as np

from .budget import BudgetExhausted


RAW_KEYS = (
    "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
    "robot0_eef_quat", "robot0_gripper_qpos",
)


class HandoffError(Exception):
    """A strict control-boundary failure, not recoverable as a guessed token."""


def copy_native_observation(obs):
    """Keep native pixels unchanged; exclude object truth even on the worker wire."""
    return {key: np.array(obs[key], copy=True) for key in RAW_KEYS}


class ControlOwnership:
    """Non-reentrant lease held during prediction AND the whole control operation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ticket = None
        self._thread = None
        self.owner = None

    @contextmanager
    def claim(self, owner):
        if not self._lock.acquire(blocking=False):
            raise HandoffError(f"Environment is already owned by {self.owner}.")
        ticket = object()
        self._ticket, self._thread, self.owner = ticket, threading.get_ident(), owner
        try:
            yield ticket
        finally:
            self._ticket = self._thread = self.owner = None
            self._lock.release()

    def validate(self, ticket):
        if (ticket is None or ticket is not self._ticket
                or self._thread != threading.get_ident()):
            raise HandoffError("Missing, expired, or cross-thread control lease.")


@dataclass(frozen=True)
class PredictionInput:
    sequence: int
    task_step: int
    history: dict | None
    current: dict


class ExecutedTrajectory:
    """History of successful pre-step observations, never decision-level frames.

    Native WLA appends before env.step. We retain that same pre-step value after a
    successful return. A failed/uncertain step forbids further execution, so there
    is no subsequent prediction for which the timing difference could matter.
    """

    def __init__(self, history_size=8):
        if history_size != 8:
            raise ValueError("Pinned WLA requires history_obs_step=8.")
        self.history = deque(maxlen=history_size)
        self.sequence = 0
        self.last_gripper_command = -1.0

    def completed(self, before, action, *, task):
        if task:
            self.history.append(copy_native_observation(before))
        self.last_gripper_command = float(action[-1])
        self.sequence += 1

    def prediction_input(self, current, task_step):
        return PredictionInput(
            sequence=self.sequence, task_step=task_step,
            history=copy_native_observation(self.history[0]) if self.history else None,
            current=copy_native_observation(current),
        )


@dataclass(frozen=True)
class ChunkResult:
    call_id: int
    input_sequence: int
    observation_sequence: int
    executed_env_steps: int
    attempted_env_steps: int
    residual_actions_discarded: int
    stop_reason: str
    official_success: bool
    last_gripper_command: float


class NativeChunkExecutor:
    """Predict once, execute at most 8 native actions, then return control.

    predictor.predict(input, original_instruction, timeout_seconds=...) must be
    synchronous and enforce its timeout; it must not retain or drive an env. This
    boundary additionally rejects late results and changed observation sequences.
    No recovery/empty-grasp heuristics run inside the chunk.
    """

    def __init__(self, session, predictor, original_instruction):
        if not isinstance(original_instruction, str) or not original_instruction.strip():
            raise ValueError("WLA needs the original complete task instruction.")
        self.session = session
        self.predictor = predictor
        self.original_instruction = original_instruction
        self.calls = 0
        self.executed_steps = 0
        self.last_result = None
        self.faulted = False

    def execute(self):
        session = self.session
        with session.control.claim("WLA_CHUNK") as lease:
            session.check_budget()
            if self.faulted:
                raise HandoffError("Previous WLA call failed; no automatic retry.")
            request = session.trajectory.prediction_input(session.obs, session.budget.steps)
            call_id = self.calls
            self.calls += 1
            attempted = 0
            predicted = 0
            stop_reason = "prediction_failed"
            try:
                session.emit(dict(event="wla_started", call_id=call_id,
                                  input_sequence=request.sequence,
                                  task_step=request.task_step))
                remaining = session.budget.timeout - (time.monotonic() - session.budget.started)
                raw = self.predictor.predict(
                    request, self.original_instruction, timeout_seconds=max(0., remaining))
                # Use the same native gripper transform, not the agent's action scaling.
                from native_contract import env_actions
                actions = env_actions(raw)
                if actions.shape != (8, 7):
                    raise HandoffError(f"Expected one native [8, 7] chunk, got {actions.shape}.")
                # Validate the ENTIRE chunk before the first action; never clip it.
                if np.max(np.abs(actions)) > 1:
                    raise HandoffError("Native action exceeds the session's normalized range.")
                predicted = len(actions)
                session.check_budget()
                if (session.trajectory.sequence != request.sequence
                        or session.budget.steps != request.task_step):
                    raise HandoffError("Stale prediction; environment changed during inference.")
                session.emit(dict(event="wla_prediction", call_id=call_id,
                                  input_sequence=request.sequence,
                                  environment_actions=actions.tolist()))
                stop_reason = "chunk_completed"
                for action in actions:
                    session.check_budget()
                    attempted += 1
                    session._step(action, "task", "WLA_CHUNK", lease=lease)
                    if session.success():
                        stop_reason = "official_success"
                        break
                    if session.returned_done:
                        stop_reason = "environment_terminated"
                        break
                    if session.budget.steps >= session.budget.max_steps:
                        stop_reason = "environment_step_budget"
                        break
            except BudgetExhausted as exc:
                stop_reason = str(exc)
                # Late prediction or mid-chunk time exhaustion must stop the episode.
                raise
            except BaseException as exc:
                self.faulted = True
                stop_reason = f"runtime_error:{type(exc).__name__}"
                raise
            finally:
                # No queue is retained on the object, including on interruption.
                executed = session.budget.steps - request.task_step
                self.executed_steps += executed
                self.last_result = ChunkResult(
                    call_id=call_id, input_sequence=request.sequence,
                    observation_sequence=session.trajectory.sequence,
                    executed_env_steps=executed, attempted_env_steps=attempted,
                    residual_actions_discarded=max(0, predicted - attempted),
                    stop_reason=stop_reason, official_success=session.success(),
                    last_gripper_command=session.trajectory.last_gripper_command,
                )
                session.emit(dict(event="wla_returned", **asdict(self.last_result)))
            return self.last_result
