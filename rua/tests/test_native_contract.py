from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_contract import Budget, env_actions, load_pilot


def test_native_gripper_and_motion_preserved():
    actions = np.arange(28, dtype=np.float32).reshape(4, 7) / 10
    actions[:, -1] = [0.0, 0.4999, 0.5, 1.0]
    original = actions.copy()
    result = env_actions(actions)
    np.testing.assert_array_equal(result[:, :-1], actions[:, :-1])
    np.testing.assert_array_equal(result[:, -1], [1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("actions", [np.zeros(7), np.zeros((0, 7)), np.zeros((4, 8)),
                                     np.full((4, 7), np.nan), np.full((4, 7), np.inf)])
def test_invalid_actions_never_become_fallback_motion(actions):
    with pytest.raises(ValueError):
        env_actions(actions)


def test_environment_budget_counts_actual_steps():
    budget = Budget(max_steps=2)
    budget.start()
    for _ in range(2):
        budget.check()
        budget.record_step()
    with pytest.raises(TimeoutError, match="environment_step_budget"):
        budget.check()


def test_timeout():
    budget = Budget(timeout_seconds=1, started=time.monotonic() - 2)
    with pytest.raises(TimeoutError, match="wall_clock_budget"):
        budget.check()


def test_pilot_is_paired_and_debug_is_disjoint():
    cfg = load_pilot(Path(__file__).resolve().parents[1] / "configs/pilot.json")
    assert len(cfg["task_ids"]) * len(cfg["eval_init_state_ids"]) == 15
    assert cfg["debug_init_state_id"] == 0
    assert cfg["max_env_steps"] == 220
    assert cfg["settle_steps"] == 10
