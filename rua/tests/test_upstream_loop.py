"""Compare queue/history/step semantics against the actual pinned upstream loop."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bootstrap import configure
from native_rollout import run_episode

configure()
import run_libero_eval as upstream


class FakeEnv:
    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.actions = []

    def observation(self):
        return {
            "step_id": len(self.actions),
            "robot0_eef_pos": np.array([len(self.actions), 0, 0], dtype=float),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=float),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def reset(self):
        self.actions = []

    def set_init_state(self, _):
        return self.observation()

    def step(self, action):
        self.actions.append(action)
        return self.observation(), 0, self.check_success(), {}

    def check_success(self):
        return len(self.actions) >= self.stop_after


class FakePolicy:
    def __init__(self):
        self.calls = []

    def inference(self, observation, instruction):
        self.calls.append((observation["full_image"], observation["state"].copy(), instruction))
        raw = torch.arange(56, dtype=torch.float32).reshape(8, 7) / 100
        raw[:, -1] = torch.tensor([0, 0.49, 0.5, 1, 0, 1, 0.5, 0.499])
        return raw


@pytest.mark.parametrize("stop_after", [11, 27, 999])
def test_actual_upstream_loop_matches_instrumented_loop(monkeypatch, tmp_path, stop_after):
    native_env, instrumented_env = FakeEnv(stop_after), FakeEnv(stop_after)
    native_policy, instrumented_policy = FakePolicy(), FakePolicy()
    suite = SimpleNamespace(
        n_tasks=1, get_task=lambda _: SimpleNamespace(language="original full task"),
        get_task_init_states=lambda _: [np.zeros(1)],
    )
    monkeypatch.setattr(upstream, "set_seed_everywhere", lambda _: None)
    monkeypatch.setattr(upstream, "WLA0", lambda *args: native_policy)
    monkeypatch.setattr(upstream.benchmark, "get_benchmark_dict", lambda: {"libero_spatial": lambda: suite})
    monkeypatch.setattr(upstream, "get_libero_env", lambda *args, **kwargs: (native_env, "original full task"))

    def images(history, obs, size):
        return (None if history is None else history["step_id"], obs["step_id"])

    monkeypatch.setattr(upstream, "get_libero_image", images)
    cfg = upstream.GenerateConfig(
        task_suite_name="libero_spatial", num_trials_per_task=1,
        local_log_dir=str(tmp_path), save_video=False,
    )
    # draccus decorates only CLI parsing; execute the actual function body.
    upstream.eval_libero.__wrapped__(cfg)

    def predict(history, obs, step):
        observation = dict(
            full_image=images(history, obs, (256, 256)),
            state=np.concatenate((obs["robot0_eef_pos"],
                                  upstream.quat2axisangle(obs["robot0_eef_quat"]),
                                  obs["robot0_gripper_qpos"])),
        )
        return instrumented_policy.inference(observation, "original full task")

    result = run_episode(
        instrumented_env, np.zeros(1), predict,
        dict(history_obs_step=8, settle_steps=10, max_env_steps=220, episode_timeout_seconds=900),
        lambda *args: None, lambda *args: None,
    )
    np.testing.assert_array_equal(native_env.actions, instrumented_env.actions)
    assert len(native_policy.calls) == len(instrumented_policy.calls)
    for native, wrapped in zip(native_policy.calls, instrumented_policy.calls):
        assert native[0] == wrapped[0]
        np.testing.assert_array_equal(native[1], wrapped[1])
        assert native[2] == wrapped[2]
    assert result["official_success"] == native_env.check_success()
