import json
import math
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from data_dump.training_dump import FRONT, WRIST, export, trace_records, training_action, wla_state


def make_episode(root, name, *, success=True, pre_state=True, task_steps=9):
    directory = root / name
    directory.mkdir()
    settle = 2
    events = []
    for attempt in range(settle + task_steps):
        phase = "initialization" if attempt < settle else "task"
        action = [0, 0, 0, 0, 0, 0, -1]
        started = {"event": "step_started", "phase": phase, "attempt": attempt,
                   "token": "NATIVE_SETTLE" if phase == "initialization" else "MV_UP",
                   "action": action, "decision_index": None if phase == "initialization" else 0}
        if pre_state:
            started["proprioception_before"] = {
                "robot0_eef_pos": [float(attempt), 0, 1],
                "robot0_eef_quat": [0, 0, 0, 1],
                "robot0_gripper_qpos": [0.04, -0.04],
            }
        events.append(started)
        events.append({"event": "step_completed", "phase": phase, "attempt": attempt,
                       "token": started["token"], "action": action,
                       "official_success": success and attempt == settle + task_steps - 1,
                       "decision_index": started["decision_index"]})
    (directory / "environment-steps.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))
    result = {
        "status": "completed", "official_success": success, "uncertain_step": None,
        "cleanup_errors": [], "mode": "rua_only", "backend": "claude",
        "task_instruction": "pick up the black bowl", "task_id": 1, "initial_state_id": 3,
        "pilot": {"suite": "libero_spatial", "settle_steps": settle, "max_env_steps": 220,
                  "max_agent_requests": 5},
        "task_steps": task_steps, "initialization_steps": settle,
        "torch_cuda_initialized": False, "wla_calls": 0, "qwen_calls": 0,
        "model_requests": 1,
        "request_records": [{"call_id": 1,
                             "image_manifest": [{"camera": "front"}, {"camera": "wrist"}]}],
    }
    (directory / "result.json").write_text(json.dumps(result))
    request = directory / "requests/request-001"
    request.mkdir(parents=True)
    (request / "request.json").write_text(json.dumps({"call_id": 1, "status": "received"}))
    (request / "prompt.txt").write_text("What should I do?")
    (request / "response.txt").write_text('{"decision":"MV_UP"}')
    native = directory / "native/session"
    native.mkdir(parents=True)
    (native / "steps.json").write_text(json.dumps(
        [{"i": 0, "act": "MV_UP", "model_request_id": 1}]))
    (native / "subgoals.json").write_text(json.dumps([{"id": "reach_bowl"}]))
    for camera in ("front", "wrist"):
        with imageio.get_writer(directory / f"{camera}-control.mp4", fps=20,
                                codec="libx264", macro_block_size=1) as writer:
            for frame in range(settle + task_steps + 1):
                writer.append_data(np.full((16, 16, 3), frame * 10 % 256, dtype=np.uint8))
    return directory


def make_norm_stats(root):
    path = root / "norm_stats.json"
    path.write_text(json.dumps({"libero_all": {
        "observation.state": {"min": [-1.0] * 8, "max": [1.0] * 8},
        "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
    }}))
    return path


class FakeLeRobotDataset:
    created = []

    @classmethod
    def create(cls, **kwargs):
        obj = cls()
        Path(kwargs["root"]).mkdir(parents=True)
        obj.kwargs = kwargs
        obj.episodes = []
        obj.frames = []
        obj.finalized = False
        cls.created.append(obj)
        return obj

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def finalize(self):
        self.finalized = True


def install_fake_lerobot(monkeypatch):
    root = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", root)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset)
    FakeLeRobotDataset.created.clear()


def test_native_state_and_gripper_mapping():
    state = wla_state({"robot0_eef_pos": [0, 0, 1],
                       "robot0_eef_quat": [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
                       "robot0_gripper_qpos": [0.04, -0.04]})
    assert state.shape == (8,)
    np.testing.assert_allclose(state[3:6], [0, 0, math.pi / 2], atol=1e-6)
    assert training_action([0, 0, 0, 0, 0, 0, -1])[-1] == 1
    assert training_action([0, 0, 0, 0, 0, 0, 1])[-1] == 0


def test_environment_action_kind_does_not_hide_trace_source(tmp_path):
    records = list(trace_records(tmp_path, [
        {"event": "atomic_completed", "kind": "move", "token": "MV_DOWN"},
    ]))
    assert records == [{"source": str(tmp_path), "kind": "environment",
                        "event": "atomic_completed", "environment_kind": "move",
                        "token": "MV_DOWN"}]


def test_dump_aligns_frames_and_keeps_failure_feedback(tmp_path, monkeypatch):
    install_fake_lerobot(monkeypatch)
    source = tmp_path / "rollouts"
    source.mkdir()
    successful = make_episode(source, "success")
    make_episode(source, "failure", success=False)
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot")

    assert len(manifest["episodes"]) == 1
    assert manifest["episodes"][0]["task_step_range"] == [0, 9]
    dataset = FakeLeRobotDataset.created[0]
    assert dataset.finalized and dataset.kwargs["fps"] == 20
    assert list(dataset.kwargs["features"]) == [FRONT, WRIST, "observation.state", "action"]
    frames = dataset.episodes[0]
    assert len(frames) == 9
    assert abs(float(frames[0][FRONT].mean()) - 20) < 2
    assert abs(float(frames[-1][WRIST].mean()) - 100) < 2
    assert frames[0]["observation.state"][0] == 2
    assert frames[0]["action"].shape == (7,)
    assert frames[0]["action"][-1] == 1
    assert all(frame["task"] == "pick up the black bowl" for frame in frames)

    feedback = [json.loads(line) for line in (output / "feedback.jsonl").read_text().splitlines()]
    assert len(feedback) == 2
    selected = next(row for row in feedback if row["source"] == str(successful))
    rejected = next(row for row in feedback if row["source"] != str(successful))
    assert selected["wla_candidate"] and selected["wla_training_starts"] == 1
    assert not rejected["wla_candidate"]
    assert rejected["rejection_code"] == "task_not_successful"
    trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert {row["kind"] for row in trace} == {"environment", "agent_decision",
                                             "agent_artifact", "model_request"}
    assert any(row.get("record", {}).get("model_request_id") == 1 for row in trace)
    assert any(row.get("artifact_type") == "subgoals" for row in trace)


def test_missing_pre_action_state_is_diagnostic_only(tmp_path):
    source = make_episode(tmp_path, "old", pre_state=False)
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot")
    assert manifest["episodes"] == []
    feedback = json.loads((output / "feedback.jsonl").read_text().strip())
    assert feedback["audit_pass"] is True
    assert "predates the dump collector" in feedback["rejection_reason"]


def test_review_keeps_rejected_success_in_feedback(tmp_path):
    source = make_episode(tmp_path, "successful")
    review = tmp_path / "review-input.jsonl"
    review.write_text(json.dumps({"source": str(source), "approved_for_wla": False}) + "\n")
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot", review)
    assert manifest["dataset"] is None
    assert manifest["episodes"] == []
    assert not (output / "rua_lerobot").exists()
    assert json.loads((output / "review.jsonl").read_text())["approved_for_wla"] is False
    feedback = json.loads((output / "feedback.jsonl").read_text())
    assert feedback["review_approved"] is False
    assert feedback["official_success"] is True
    assert feedback["wla_candidate"] is False
    assert feedback["rejection_code"] == "review_rejected"
    assert feedback["wla_training_starts"] == 0


def test_review_selects_contiguous_task_step_ranges(tmp_path, monkeypatch):
    install_fake_lerobot(monkeypatch)
    source = make_episode(tmp_path, "successful", task_steps=20)
    review = tmp_path / "review-input.jsonl"
    review.write_text(json.dumps({
        "source": str(source), "approved_for_wla": True,
        "task_step_ranges": [[0, 9], [11, 20]],
        "reason": "Keep two reviewed action sequences",
    }) + "\n")
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot", review)

    assert manifest["version"] == 2
    assert [row["task_step_range"] for row in manifest["episodes"]] == [[0, 9], [11, 20]]
    assert [row["wla_training_starts"] for row in manifest["episodes"]] == [1, 1]
    dataset = FakeLeRobotDataset.created[0]
    assert [len(episode) for episode in dataset.episodes] == [9, 9]
    assert dataset.episodes[0][0]["observation.state"][0] == 2
    assert dataset.episodes[1][0]["observation.state"][0] == 13
    feedback = json.loads((output / "feedback.jsonl").read_text())
    assert feedback["wla_candidate"] is True
    assert feedback["wla_training_starts"] == 2
    assert feedback["control_tokens"] == ["MV_UP"]
    saved_review = json.loads((output / "review.jsonl").read_text())
    assert saved_review["task_step_ranges"] == [[0, 9], [11, 20]]
    assert saved_review["reason"] == "Keep two reviewed action sequences"


@pytest.mark.parametrize("ranges", [
    [[0, 8]], [[0, 10], [9, 20]], [[0, 9], [20, 29]],
])
def test_invalid_review_ranges_keep_trace_and_feedback(tmp_path, ranges):
    source = make_episode(tmp_path, "successful", task_steps=20)
    review = tmp_path / "review-input.jsonl"
    review.write_text(json.dumps({"source": str(source), "approved_for_wla": True,
                                  "task_step_ranges": ranges}) + "\n")
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot", review)

    assert manifest["episodes"] == []
    assert (output / "trace.jsonl").stat().st_size > 0
    feedback = json.loads((output / "feedback.jsonl").read_text())
    assert feedback["audit_pass"] is True
    assert feedback["rejection_code"] == "invalid_review_range"
    assert feedback["wla_training_starts"] == 0


def test_session_records_pre_action_state_and_decision_id(tmp_path, monkeypatch):
    from libero_harness.budget import Budget
    from libero_harness.environment import LiberoSession

    class VideoWriter:
        def append_data(self, image):
            pass

        def close(self):
            pass

    class Environment:
        def __init__(self):
            self.position = np.array([0.0, 0.0, 1.0])

        def observation(self):
            return {"agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "robot0_eef_pos": self.position.copy(),
                    "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                    "robot0_gripper_qpos": np.array([0.04, -0.04])}

        def reset(self):
            pass

        def set_init_state(self, state):
            return self.observation()

        def step(self, action):
            self.position[0] += 0.1
            return self.observation(), 0.0, False, {}

        def check_success(self):
            return False

    monkeypatch.setattr("libero_harness.environment.imageio.get_writer",
                        lambda *args, **kwargs: VideoWriter())
    session = LiberoSession(Environment(), np.zeros(3), {"wrist_extra_flip": "none"},
                            {"settle_steps": 0, "history_obs_step": 8}, Budget(), tmp_path)
    try:
        session.agent_decision_index = 5
        with session.control.claim("test") as lease:
            session._step([0, 0, 0, 0, 0, 0, -1], "task", "MV_UP", lease=lease)
    finally:
        session.close()
    started = json.loads((tmp_path / "environment-steps.jsonl").read_text().splitlines()[0])
    assert started["decision_index"] == 5
    assert started["proprioception_before"]["robot0_eef_pos"] == [0.0, 0.0, 1.0]
    assert started["observation_sequence"] == 0


def test_real_lerobot_roundtrip(tmp_path):
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    source = make_episode(tmp_path, "episode")
    output = tmp_path / "dump"
    export([source], output, "rua_lerobot")
    dataset_root = output / "rua_lerobot"
    assert (dataset_root / "meta/info.json").is_file()
    assert list((dataset_root / "data").rglob("*.parquet"))
    dataset = lerobot.LeRobotDataset(
        "rua_lerobot", root=dataset_root, video_backend="pyav",
        delta_timestamps={
            FRONT: [0, 8 / 20, -8 / 20],
            "action": [i / 20 for i in range(8)],
        },
    )
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.camera_keys == [FRONT, WRIST]
    assert len(dataset) == 9
    sample = dataset[0]
    assert tuple(sample["observation.state"].shape) == (8,)
    assert tuple(sample["action"].shape) == (8, 7)
    assert tuple(sample[FRONT].shape) == (3, 3, 16, 16)


def test_wla_train_loader_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    wla_dataset = pytest.importorskip("dataset")
    source = make_episode(tmp_path, "episode", task_steps=20)
    review = tmp_path / "review-input.jsonl"
    review.write_text(json.dumps({
        "source": str(source), "approved_for_wla": True,
        "task_step_ranges": [[0, 9], [11, 20]],
    }) + "\n")
    output = tmp_path / "dump"
    export([source], output, "rua_lerobot", review)
    data_args = SimpleNamespace(dataset_root_dir=str(output),
                                norm_stats_path=str(make_norm_stats(tmp_path)),
                                unnorm_key="libero_all")
    model_args = SimpleNamespace(chunk_size=8, sample_num=1, use_history_obs=True,
                                 history_obs_step=8, action_condition_type="no_action_condition",
                                 max_state_dim=8, max_action_dim=7, auxiliary_drop_thresh=0.0,
                                 training_mode="image_action")
    base, norm, front, wrists = wla_dataset.load_libero_dataset(data_args, model_args, None)
    assert len(base) == 2  # Each nine-frame episode gives one real t + 8 target.
    assert not base[0][f"{front}_is_pad"][1]
    assert int(base[1]["episode_index"]) == 1
    assert not base[1][f"{front}_is_pad"][1]
    monkeypatch.setattr(wla_dataset.random, "random", lambda: 1.0)
    sample = wla_dataset.LeRobotTrainDataset(
        base_dataset=base, target_transform=lambda image: image,
        primary_image_size=16, auxiliary_image_size=16,
        primary_image_key=front, auxiliary_image_key=wrists,
        norm_stats=norm, model_args=model_args,
    )[0]
    assert sample["caption"] == "pick up the black bowl"
    assert len(sample["input_images"]) == 3
    assert len(sample["target_images"]) == 1
    assert tuple(sample["states"].shape) == (8,)
    assert tuple(sample["actions"].shape) == (8, 7)
