from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_rollout import aggregate, pilot_cases, run_episode


CFG = dict(history_obs_step=8, settle_steps=10, max_env_steps=19, episode_timeout_seconds=900)


class Env:
    def __init__(self, success_at=None, done_at=None, error_at=None):
        self.actions = []
        self.success_at, self.done_at, self.error_at = success_at, done_at, error_at

    def reset(self):
        self.actions = []

    def set_init_state(self, state):
        return {"index": 0}

    def check_success(self):
        return self.success_at is not None and len(self.actions) >= self.success_at

    def step(self, action):
        if self.error_at == len(self.actions):
            raise RuntimeError("physics error")
        self.actions.append(action)
        return {"index": len(self.actions)}, 0, len(self.actions) == self.done_at, {}


def rollout(env=None, predict=None, cfg=CFG, clock=None):
    env = env or Env()
    calls, events, frames = [], [], []

    def default_predict(history, obs, step):
        calls.append((None if history is None else history["index"], obs["index"], step))
        raw = np.zeros((8, 7))
        raw[:, 0] = np.arange(8)
        raw[:, -1] = 1
        return raw

    kwargs = {} if clock is None else {"clock": clock}
    result = run_episode(env, None, predict or default_predict, cfg,
                         lambda event, obs: events.append(event), frames.append, **kwargs)
    return result, env, calls, events, frames


def test_native_queue_history_and_actual_budget():
    result, env, calls, events, frames = rollout()
    assert result["task_control_steps"] == 19
    assert result["initialization_steps"] == 10
    assert len(env.actions) == 29
    assert calls == [(None, 10, 0), (10, 18, 8), (18, 26, 16)]
    assert [a[0] for a in env.actions[10:]] == list(range(8)) * 2 + [0, 1, 2]
    assert all(a[-1] == -1 for a in env.actions)
    assert result["residual_actions_discarded"] == 5
    assert len(frames) == 30
    assert len([e for e in events if e["event"] == "step_completed"]) == 29
    assert result["stop_reason"] == "environment_step_budget"
    assert not result["official_success"]


def test_done_without_official_success_is_not_success():
    result, env, *_ = rollout(Env(done_at=12))
    assert result["returned_done"] and not result["official_success"]
    assert result["stop_reason"] == "terminated_without_success"
    assert len(env.actions) == 12


def test_official_success_stops_immediately_and_discards_queue():
    result, env, *_ = rollout(Env(success_at=11))
    assert result["official_success"] and result["task_control_steps"] == 1
    assert result["residual_actions_discarded"] == 7
    assert result["stop_reason"] == "official_success"


@pytest.mark.parametrize("raw", [np.full((8, 7), np.nan), np.zeros((7, 7)), np.zeros(7)])
def test_invalid_prediction_executes_no_task_action(raw):
    result, env, *_ = rollout(predict=lambda *args: raw)
    assert result["status"] == "error"
    assert len(env.actions) == 10
    assert result["task_control_steps"] == 0


def test_inference_exception_no_fallback():
    def predict(*args):
        raise RuntimeError("model failure")
    result, env, *_ = rollout(predict=predict)
    assert result["status"] == "error" and len(env.actions) == 10
    assert result["wla_calls"] == 1


def test_step_exception_records_uncertain_attempt_and_no_retry():
    result, env, *_ = rollout(Env(error_at=11))
    assert result["task_control_steps"] == 1
    assert result["step_attempts"] == 12
    assert result["uncertain_step"] == {"phase": "task", "attempt": 11}
    assert len(env.actions) == 11


def test_slow_model_cannot_step_after_deadline():
    now = [0]
    def predict(*args):
        now[0] = 901
        return np.zeros((8, 7))
    result, env, *_ = rollout(predict=predict, clock=lambda: now[0])
    assert result["status"] == "budget_exhausted"
    assert result["stop_reason"] == "wall_clock_budget"
    assert len(env.actions) == 10


def test_interrupt_returns_partial_result():
    def predict(*args):
        raise InterruptedError("stop")
    result, env, *_ = rollout(predict=predict)
    assert result["status"] == "interrupted" and len(env.actions) == 10
    assert result["elapsed_seconds"] >= 0


def test_initialization_success_is_an_error_not_a_valid_episode():
    result, *_ = rollout(Env(success_at=1))
    assert result["status"] == "error"
    summary = aggregate([(0, 1)], [dict(result, task_id=0)])
    assert not summary["stage1_gate_passed"]


def test_exact_cases_and_no_incomplete_gate():
    cfg = dict(task_ids=[0, 1, 2], debug_init_state_id=0, eval_init_state_ids=[1, 2, 3, 4, 5])
    cases = pilot_cases(cfg)
    assert cases == [(task, state) for task in range(3) for state in range(1, 6)]
    assert pilot_cases(cfg, True) == [(0, 0)]
    results = [dict(status="completed", official_success=True, task_id=0)]
    assert not aggregate(cases, results)["stage1_gate_passed"]
    assert aggregate(cases, results)["success_rate"] == 1 / 15


def test_history_and_queue_never_carry_between_episodes():
    env = Env()
    first = rollout(env)
    second = rollout(env)
    assert first[2] == second[2]
    assert first[0]["residual_actions_discarded"] == second[0]["residual_actions_discarded"]
