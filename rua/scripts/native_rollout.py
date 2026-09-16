"""Instrument the upstream WLA queue/history loop without changing the policy."""
from __future__ import annotations

import time
import traceback
from collections import deque

from native_contract import env_actions


def pilot_cases(cfg, debug=False):
    return [(0, cfg["debug_init_state_id"])] if debug else [
        (task, state) for task in cfg["task_ids"] for state in cfg["eval_init_state_ids"]
    ]


def run_episode(env, initial_state, predict, cfg, emit, capture, clock=time.monotonic):
    """predict(history, current, step) returns raw, unnormalized native actions.

    No retries, substituted actions, oracle policy inputs, or residual queue reuse.
    emit/capture must not consume model/environment randomness.
    """
    started = clock()
    result = dict(
        status="running", official_success=False, stop_reason=None,
        initialization_steps=0, task_control_steps=0, step_attempts=0,
        wla_calls=0, returned_done=False, residual_actions_discarded=0,
        agent_requests=0, agent_tokens=0,
    )
    history = deque(maxlen=cfg["history_obs_step"])
    queue = deque()
    inflight_step = None

    def check_time():
        if clock() - started >= cfg["episode_timeout_seconds"]:
            raise TimeoutError("wall_clock_budget")

    def step(action, phase, obs):
        nonlocal inflight_step
        check_time()
        inflight_step = {"phase": phase, "attempt": result["step_attempts"]}
        emit(dict(event="step_started", action=action, **inflight_step), obs)
        result["step_attempts"] += 1
        new_obs, reward, done, _ = env.step(action)
        result["initialization_steps" if phase == "settle" else "task_control_steps"] += 1
        inflight_step = None
        success = bool(env.check_success())
        result.update(official_success=success, returned_done=bool(done))
        emit(dict(event="step_completed", phase=phase, action=action,
                  reward=float(reward), done=bool(done), official_success=success,
                  initialization_steps=result["initialization_steps"],
                  task_control_steps=result["task_control_steps"]), new_obs)
        capture(new_obs)
        return new_obs, bool(done), success

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        capture(obs)
        emit(dict(event="initialized"), obs)
        # Waiting observations never enter WLA's history deque.
        for _ in range(cfg["settle_steps"]):
            obs, done, success = step([0, 0, 0, 0, 0, 0, -1], "settle", obs)
            if done or success:
                raise RuntimeError("Unexpected terminal/success state during initialization.")
        while result["task_control_steps"] < cfg["max_env_steps"]:
            check_time()
            if not queue:
                result["wla_calls"] += 1
                raw = predict(history[0] if history else None, obs, result["task_control_steps"])
                # Equivalent to the upstream torch.where; reject invalid arrays before stepping.
                actions = env_actions(raw)
                if actions.shape != (8, 7):
                    raise ValueError(f"Locked checkpoint must return [8, 7], got {actions.shape}.")
                queue.extend(actions.tolist())
                check_time()  # A slow call may return after the episode deadline.
            action = queue.popleft()
            history.append(obs)  # Upstream stores the observation BEFORE env.step.
            obs, done, success = step(action, "task", obs)
            if success or done:
                result["stop_reason"] = "official_success" if success else "terminated_without_success"
                break
        else:
            result["stop_reason"] = "environment_step_budget"
        result["status"] = "completed"
    except BaseException as exc:
        result.update(
            status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else
                   "budget_exhausted" if isinstance(exc, TimeoutError) else "error",
            stop_reason="wall_clock_budget" if isinstance(exc, TimeoutError) else type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(),
            # If env.step raised, the physics side effect is uncertain; never guess/retry.
            uncertain_step=inflight_step,
        )
    finally:
        result["residual_actions_discarded"] = len(queue)
        queue.clear()
        history.clear()
        result["elapsed_seconds"] = clock() - started
    return result


def aggregate(cases, results):
    complete = len(results) == len(cases)
    successes = sum(bool(r["official_success"]) for r in results)
    errors = sum(r["status"] not in ("completed", "budget_exhausted") or
                 bool(r.get("artifact_errors")) for r in results)
    return dict(
        planned_episodes=len(cases), finished_episodes=len(results), successes=successes,
        success_rate=successes / len(cases), runtime_errors=errors,
        stage1_gate_passed=complete and errors == 0 and successes > 0,
        per_task={
            str(task): dict(
                episodes=sum(r["task_id"] == task for r in results),
                successes=sum(r["task_id"] == task and r["official_success"] for r in results),
            ) for task in sorted({task for task, _ in cases})
        },
    )
