"""Observation and atomic-action boundary. The planner never receives object state."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from core.record.images import prepare_view, rotate_and_flip
from interpreters.real_atomic_controller import AtomicStepResult
from .budget import Budget, BudgetExhausted
from .handoff import ControlOwnership, ExecutedTrajectory, HandoffError, copy_native_observation


def pose(obs):
    return np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_quat"])).copy()


def width(obs):
    return float(np.abs(obs["robot0_gripper_qpos"]).sum())


class LiberoSession:
    name = "libero_official_scoring"

    def __init__(self, env, initial_state, cfg, pilot, budget: Budget, directory: Path):
        self.env, self.cfg, self.budget = env, cfg, budget
        self.directory = directory
        self.obs = None
        self.initialization_steps = 0
        self.attempts = 0
        self.official_success = False
        self.returned_done = False
        self.errors = []
        self.uncertain_step = None
        self.control = ControlOwnership()
        self.trajectory = ExecutedTrajectory(pilot.get("history_obs_step", 8))
        self.last_control_mode = "initialization"
        self.gripper_authority = "initialization"
        self.last_handoff = None
        self.video_frames = 0
        self.trace = (directory / "environment-steps.jsonl").open("w")
        self.writers = {
            name: imageio.get_writer(directory / f"{name}-control.mp4", fps=20,
                                     codec="libx264", ffmpeg_params=["-threads", "1",
                                     "-movflags", "frag_keyframe+empty_moov+default_base_moof"])
            for name in ("front", "wrist")
        }
        try:
            budget.check_time()
            env.reset()
            self.obs = env.set_init_state(initial_state)
            np.save(directory / "initial_state.npy", initial_state)
            self.capture()
            for _ in range(pilot["settle_steps"]):
                self._step([0, 0, 0, 0, 0, 0, -1], "initialization", "NATIVE_SETTLE")
                if self.official_success or self.returned_done:
                    raise ValueError("Unexpected terminal state during initialization.")
        except BaseException:
            self.close()
            raise

    def success(self):
        return self.official_success

    def check_budget(self):
        if self.uncertain_step is not None:
            raise HandoffError("Previous environment step is uncertain; do not resume.")
        self.budget.before_step()
        if self.returned_done or self.official_success:
            raise BudgetExhausted("official_success" if self.official_success else "environment_terminated")

    def get_observation(self):
        # Explicit whitelist: never forward raw LIBERO observations (object poses).
        front = prepare_view(self.obs["agentview_image"], flip="both")
        wrist = prepare_view(self.obs["robot0_eye_in_hand_image"], flip="both")
        wrist = rotate_and_flip(wrist, flip=self.cfg.get("wrist_extra_flip", "none"))
        return dict(agentview=front, wrist=wrist, agentview_hd=front, wrist_hd=wrist,
                    ee_pose=pose(self.obs), gripper_width=width(self.obs))

    def capture(self):
        obs = self.get_observation()
        self.writers["front"].append_data(obs["agentview"])
        self.writers["wrist"].append_data(obs["wrist"])
        self.video_frames += 1

    def emit(self, record):
        record.update(elapsed_seconds=time.monotonic() - self.budget.started,
                      eef_pose=pose(self.obs).tolist(), gripper_width_m=width(self.obs))
        self.trace.write(json.dumps(record, allow_nan=False) + "\n")
        self.trace.flush()

    def _step(self, action, phase, token, *, lease=None):
        if phase == "task":
            self.control.validate(lease)
            self.check_budget()
        else:
            self.budget.check_time()
        array = np.asarray(action, dtype=float)
        if array.shape != (7,) or not np.isfinite(array).all() or np.max(np.abs(array)) > 1:
            raise ValueError("Invalid normalized environment action.")
        attempt = self.attempts
        self.attempts += 1
        self.uncertain_step = dict(attempt=attempt, token=token, action=array.tolist())
        self.emit(dict(event="step_started", phase=phase, **self.uncertain_step))
        # Snapshot BEFORE stepping, including when an environment reuses its arrays.
        before = copy_native_observation(self.obs)
        obs, reward, done, _ = self.env.step(array.tolist())
        self.obs = obs
        self.trajectory.completed(before, array, task=phase == "task")
        if phase == "task":
            self.last_control_mode = "wla" if token == "WLA_CHUNK" else "atomic"
            if token == "WLA_CHUNK":
                self.gripper_authority = "wla"
            elif token in ("GRASP", "RELEASE", "EMPTY_REOPEN"):
                self.gripper_authority = "atomic"
        self.uncertain_step = None
        if phase == "task":
            self.budget.steps += 1
        else:
            self.initialization_steps += 1
        self.official_success = bool(self.env.check_success())
        self.returned_done = bool(done)
        self.emit(dict(event="step_completed", phase=phase, token=token, attempt=attempt,
                       action=array.tolist(), reward=float(reward), done=bool(done),
                       official_success=self.official_success, task_steps=self.budget.steps,
                       initialization_steps=self.initialization_steps))
        self.capture()

    def execute(self, action, repetitions, token, *, lease=None):
        if lease is None:
            with self.control.claim(token) as claimed:
                return self.execute(action, repetitions, token, lease=claimed)
        self.control.validate(lease)
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError("Repetitions must be a positive integer.")
        executed = 0
        for _ in range(repetitions):
            self._step(action, "task", token, lease=lease)
            executed += 1
            if self.official_success or self.returned_done:
                break
        return executed

    def record_error(self, exc):
        self.errors.append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        return str(exc) if isinstance(exc, BudgetExhausted) else f"runtime_error:{type(exc).__name__}"

    def close(self):
        errors = []
        for name, writer in self.writers.items():
            try:
                writer.close()
            except Exception as exc:
                errors.append(f"{name}:{type(exc).__name__}")
        self.trace.close()
        return errors


class LiberoAtomicController:
    """The upstream vocabulary, calibrated normalized OSC_POSE implementation."""
    def __init__(self, session: LiberoSession, cfg, *, wla_executor=None):
        if wla_executor is not None and wla_executor.session is not session:
            raise ValueError("WLA and atomic actions must share one session.")
        self.session, self.cfg = session, cfg
        self.wla_executor = wla_executor
        self.gripper_close_threshold_m = cfg["open_width_m"]
        # No hardware Z clamp is silently imposed on the benchmark physics.
        self.z_floor_m = None

    @property
    def gripper_closed(self):
        # Commanded close is not a claim that an object is held. Do not infer it
        # from measured width or cache it separately from WLA's executed actions.
        return self.session.trajectory.last_gripper_command > 0

    def step(self, token, *, target_in_wrist=None, continuous=False):
        valid = set(self.cfg["token_vectors"]) | {"GRASP", "RELEASE", "STOP", "DONE", "WLA_CHUNK"}
        if token not in valid or continuous:
            raise ValueError("Invalid/queued action; execute no action.")
        if token == "WLA_CHUNK":
            if self.wla_executor is None or continuous:
                raise ValueError("WLA must be explicitly enabled, with no queued agent chunk.")
            before = pose(self.session.obs)
            returned = self.wla_executor.execute()
            return AtomicStepResult(
                token=token, kind="delegated", pre_pose=before, post_pose=pose(self.session.obs),
                gripper_closed=self.gripper_closed, done=False, grasp_empty=False,
                note=json.dumps(vars(returned), sort_keys=True),
            )
        with self.session.control.claim("atomic:" + str(token)) as lease:
            if (self.cfg.get("controller_revision") == "front_view_handoff_v2"
                    and self.session.last_control_mode == "wla" and token != "DONE"):
                from .coordination import HOLD_STEPS
                before = pose(self.session.obs)
                steps = self.session.execute(
                    [0, 0, 0, 0, 0, 0, 1 if self.gripper_closed else -1],
                    HOLD_STEPS, "HANDOFF_HOLD", lease=lease)
                self.session.last_handoff = dict(
                    requested_token=token, executed_steps=steps,
                    observation_sequence=self.session.trajectory.sequence)
                self.session.emit(dict(event="handoff_completed", **self.session.last_handoff))
                return AtomicStepResult(
                    token="HANDOFF_HOLD", kind="handoff", pre_pose=before,
                    post_pose=pose(self.session.obs), gripper_closed=self.gripper_closed,
                    done=False, grasp_empty=False,
                    note="Counted hold only. Proposed atomic action discarded; observe and decide again.",
                )
            return self._step_owned(token, lease=lease, target_in_wrist=target_in_wrist,
                                    continuous=continuous)

    def _step_owned(self, token, *, lease, target_in_wrist=None, continuous=False):
        if continuous:
            raise ValueError("Action chunks are disabled for this baseline.")
        before = pose(self.session.obs)
        steps_before = self.session.budget.steps
        delta = np.zeros(3)
        empty = False
        note = ""
        if token == "DONE":
            kind = "done"
        elif token in self.cfg["token_vectors"]:
            kind = "move"
            vector = np.asarray(self.cfg["token_vectors"][token], dtype=float)
            delta = vector * self.cfg["nominal_step_m"]
            action = [*(vector * self.cfg["move_amplitude"]).tolist(), 0, 0, 0,
                      1 if self.gripper_closed else -1]
            self.session.execute(action, self.cfg["move_env_steps"], token, lease=lease)
        elif token in ("GRASP", "RELEASE"):
            kind = "gripper"
            action = [0, 0, 0, 0, 0, 0, 1 if token == "GRASP" else -1]
            self.session.execute(action, self.cfg["gripper_env_steps"], token, lease=lease)
            empty = token == "GRASP" and width(self.session.obs) <= self.cfg["empty_width_m"]
            if empty and not self.session.success() and not self.session.returned_done:
                # Match upstream empty-grasp semantics: reopen, count every step.
                self.session.execute([0, 0, 0, 0, 0, 0, -1], self.cfg["gripper_env_steps"],
                                     "EMPTY_REOPEN", lease=lease)
                note = "empty close -> reopened"
        elif token == "STOP":
            kind = "stop"
            self.session.execute([0, 0, 0, 0, 0, 0, 1 if self.gripper_closed else -1],
                                 1, token, lease=lease)
        else:
            raise ValueError(f"Unrecognized atomic token {token!r}; execute no action.")
        after = pose(self.session.obs)
        result = AtomicStepResult(
            token=token, kind=kind, intended_delta_m=delta, pre_pose=before, post_pose=after,
            gripper_closed=self.gripper_closed, done=token == "DONE", grasp_empty=empty, note=note,
            step_kind="fixed" if kind == "move" else "",
            step_m=self.cfg["nominal_step_m"] if kind == "move" else 0,
        )
        self.session.emit(dict(event="atomic_completed", token=token, kind=kind,
                               executed_steps=self.session.budget.steps - steps_before,
                               official_success=self.session.success(), grasp_empty=empty))
        return result
