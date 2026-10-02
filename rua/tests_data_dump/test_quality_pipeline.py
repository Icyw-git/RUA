import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_dump.quality_labels import _carried, classify
from data_dump.quality_pipeline import annotate_efficiency
from data_dump.training_dump import export, quality_starts
from test_training_dump import make_episode


def write_oracle(source: Path, root: Path, rows: list[dict]):
    root.mkdir()
    (root / "oracle-feedback.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    trace_hash = hashlib.sha256((source / "environment-steps.jsonl").read_bytes()).hexdigest()
    (root / "oracle-manifest.json").write_text(json.dumps({
        "version": 2, "source": str(source), "task_steps": len(rows),
        "source_trace_sha256": trace_hash,
        "target_objects": ["bowl"], "goal_states": [["in", "bowl", "basket_region"]],
        "feedback_file": "oracle-feedback.jsonl",
    }))


def test_physical_miss_then_recovery_is_separate_from_error(tmp_path):
    source = make_episode(tmp_path, "source", task_steps=35)
    rows = []
    for i in range(35):
        if i < 4:
            hand = [0.09 - 0.01 * i, 0, 0.1]
        elif i <= 8:
            hand = [0.05 + 0.025 * (i - 4), 0, 0.1]
        elif i < 20:
            hand = [0.15 - 0.008 * (i - 8), 0, 0.1]
        else:
            hand = [0.054 + 0.01 * (i - 20), 0, 0.1]
        object_pos = [0, 0, 0.04] if i < 20 else [hand[0] - 0.05, 0, 0.07]
        command = 1 if 4 <= i < 8 or 20 <= i < 33 else -1
        rows.append({
            "task_step": i, "goal_predicates": [{"satisfied": i >= 34}],
            "eef_position_m": hand, "gripper_command": command,
            "grasped_objects": [], "object_positions_m": {"bowl": object_pos},
            "all_object_positions_m": {"bowl": object_pos},
        })
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    labels = classify(source, oracle)
    events = [event for label in labels for event in label["events"]]
    assert {event["event"] for event in events} >= {"grasp_missed", "target_grasped",
                                                    "goal_gained"}
    miss = next(event for event in events if event["event"] == "grasp_missed")
    assert miss["verification"] == "verified"
    assert any(label["role"] == "error" and label["task_step_range"][0] <= 4
               and label["task_step_range"][1] >= 8 for label in labels)
    assert any(label["role"] == "recovery" and label["verification"] == "verified"
               for label in labels)


def test_quality_dump_preserves_full_history_and_filters_eight_actions(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    wla_dataset = pytest.importorskip("dataset")
    source = make_episode(tmp_path, "source", task_steps=35)
    trace_hash = hashlib.sha256((source / "environment-steps.jsonl").read_bytes()).hexdigest()
    labels = []
    for start, end, role in [(0, 10, "nominal"), (10, 12, "error"),
                              (12, 30, "recovery"), (30, 35, "nominal")]:
        labels.append({"source": str(source), "source_trace_sha256": trace_hash,
                       "oracle_feedback_sha256": "oracle-v2", "labeler_version": "rules-001",
                       "task_step_range": [start, end], "role": role,
                       "verification": "verified", "coverage": "complete"})
    assert [row["frame_index"] for row in quality_starts(source, labels, 35, [(0, 35)])
            if row["set"] == "recovery"] == list(range(12, 23))
    label_file = tmp_path / "quality-labels.jsonl"
    label_file.write_text("".join(json.dumps(row) + "\n" for row in labels))
    output = tmp_path / "dump"
    manifest = export([source], output, "rua_lerobot", quality_labels=label_file)
    assert manifest["selection_mode"] == "quality"
    assert manifest["quality_recovery_starts"] == 11
    assert manifest["episodes"][0]["frames"] == 35

    data_args = SimpleNamespace(dataset_root_dir=str(output),
                                norm_stats_path="/data1/wcz/WLA/configs/norm_stats.json",
                                unnorm_key="libero_all", quality_set="recovery")
    model_args = SimpleNamespace(chunk_size=8, sample_num=1, use_history_obs=True,
                                 history_obs_step=8, action_condition_type="no_action_condition")
    base, _, front, _ = wla_dataset.load_libero_dataset(data_args, model_args, None)
    assert len(base) == 11
    sample = base[0]
    assert int(sample["frame_index"]) == 12
    assert not bool(sample[f"{front}_is_pad"][1])
    # At selected t=12, the history is the original t=4 frame, not the first
    # frame of a newly cut recovery episode.
    assert abs(float(sample[front][2].mean() * 255) - 60) < 4
    assert tuple(sample["action"].shape) == (8, 7)
    quality_file = output / "rua_lerobot/meta/quality-starts.jsonl"
    quality_file.rename(quality_file.with_suffix(".bak"))
    with pytest.raises(ValueError, match="Missing quality-starts index"):
        wla_dataset.load_libero_dataset(data_args, model_args, None)


def test_quality_starts_refuses_stale_labels(tmp_path):
    source = make_episode(tmp_path, "source", task_steps=20)
    label = {"source_trace_sha256": "old", "oracle_feedback_sha256": "oracle",
             "labeler_version": "rules-001", "task_step_range": [0, 20],
             "role": "nominal", "verification": "verified", "coverage": "complete"}
    with pytest.raises(ValueError, match="different source trace"):
        quality_starts(source, [label], 20, [(0, 20)])


def test_quality_starts_keep_future_frame_inside_reviewed_range(tmp_path):
    source = make_episode(tmp_path, "source", task_steps=35)
    trace_hash = hashlib.sha256((source / "environment-steps.jsonl").read_bytes()).hexdigest()
    label = {"source_trace_sha256": trace_hash, "oracle_feedback_sha256": "oracle",
             "labeler_version": "rules-008", "task_step_range": [0, 35],
             "role": "nominal", "verification": "verified", "coverage": "complete"}
    starts = quality_starts(source, [label], 35, [(10, 30)])
    assert [row["frame_index"] for row in starts] == list(range(10, 22))
    assert all(row["selection_version"] == "positive-002" for row in starts)
    assert [row["frame_index"] for row in quality_starts(source, [label], 35, [(10, 19)])] == [10]


def test_wrong_drawer_left_open_is_not_verified_recovery(tmp_path):
    source = make_episode(tmp_path, "drawer", task_steps=25)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"satisfied": i == 24}],
        "eef_position_m": [0, 0, 0.2],
        "fixture_joint_positions": {
            "wooden_cabinet_1_top_level": 0.04 if i >= 5 else 0.0,
            "wooden_cabinet_1_middle_level": 0.05 if i == 24 else 0.0,
        },
    } for i in range(25)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)
    manifest_path = oracle / "oracle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target_objects"] = []
    manifest["goal_states"] = [["open", "wooden_cabinet_1_middle_region"]]
    manifest_path.write_text(json.dumps(manifest))

    labels = classify(source, oracle)
    assert any(event["event"] == "wrong_drawer_moved"
               for label in labels for event in label["events"])
    assert not any(label["role"] == "recovery" for label in labels)
    assert all(label["role"] not in {"nominal", "recovery"}
               for label in labels if label["task_step_range"][0] >= 5)


def test_normal_two_object_switch_is_not_a_drop(tmp_path):
    source = make_episode(tmp_path, "two_objects", task_steps=30)
    rows = []
    for i in range(30):
        held = ["cup"] if 3 <= i <= 10 else ["bowl"] if 17 <= i <= 24 else []
        rows.append({
            "task_step": i, "goal_predicates": [
                {"objects": ["cup", "basket_region"], "satisfied": i >= 12},
                {"objects": ["bowl", "basket_region"], "satisfied": i >= 28},
            ],
            "eef_position_m": [0, 0, 0.2],
            "gripper_command": 1 if held else -1,
            "grasped_objects": held,
            "object_positions_m": {"cup": [0, 0, 0], "bowl": [0.2, 0, 0]},
            "all_object_positions_m": {"cup": [0, 0, 0], "bowl": [0.2, 0, 0]},
        })
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)
    manifest_path = oracle / "oracle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target_objects"] = ["cup", "bowl"]
    manifest["goal_states"] = [["in", "cup", "basket_region"],
                               ["in", "bowl", "basket_region"]]
    manifest_path.write_text(json.dumps(manifest))

    labels = classify(source, oracle)
    assert not any(event["event"] == "target_dropped"
                   for label in labels for event in label["events"])
    assert not any(label["role"] == "error" for label in labels)
    assert any(label["task_phase"] == "approach" and
               label["task_step_range"][0] <= 13 < label["task_step_range"][1]
               for label in labels)


def test_efficiency_ratios_describe_only_same_cohort():
    metrics = [{"source": str(i), "success": True, "cohort": ("pro", "goal", 0, 3),
                "steps": 10 * i, "eef_path_m": float(i)} for i in range(1, 5)]
    labels = [{"source": "4", "role": "nominal",
               "efficiency": {"speed_ratio": None, "path_ratio": None,
                              "reference_count": 0}}]
    annotate_efficiency(labels, metrics)
    assert labels[0]["efficiency"] == {"speed_ratio": 2.0, "path_ratio": 2.0,
                                       "reference_count": 3}
    assert labels[0]["role"] == "nominal"


def test_slow_lift_counts_as_carry_but_horizontal_push_does_not():
    def motion(lifted):
        rows = []
        for i in range(25):
            x = max(0, i - 3) * 0.004
            rows.append({
                "eef_position_m": [x + 0.04, 0, 0.1],
                "all_object_positions_m": {"bowl": [x, 0, 0.07 if lifted and i >= 4 else 0.04]},
                "gripper_command": 1 if i >= 3 else -1,
            })
        return rows

    assert any(_carried(motion(True), "bowl", 0.04))
    assert not any(_carried(motion(False), "bowl", 0.04))


def test_carry_does_not_start_during_push_before_lift():
    rows = []
    for i in range(20):
        x = max(0, i - 3) * 0.005
        rows.append({
            "eef_position_m": [x + 0.04, 0, 0.1],
            "all_object_positions_m": {"bowl": [x, 0, 0.08 if i >= 9 else 0.04]},
            "gripper_command": 1 if i >= 3 else -1,
        })

    carried = _carried(rows, "bowl", 0.04)
    assert not any(carried[:9])
    assert any(carried[9:])


def test_distant_gripper_cycle_is_not_called_a_missed_grasp(tmp_path):
    source = make_episode(tmp_path, "distant_cycle", task_steps=12)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"objects": ["bowl", "basket_region"],
                             "satisfied": False}],
        "eef_position_m": [0.3, 0, 0.1],
        "gripper_command": 1 if 3 <= i < 7 else -1,
        "grasped_objects": [],
        "object_positions_m": {"bowl": [0, 0, 0.04]},
        "all_object_positions_m": {"bowl": [0, 0, 0.04]},
    } for i in range(12)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    events = [event for label in classify(source, oracle) for event in label["events"]]
    assert any(event["event"] == "gripper_cycle_unresolved" for event in events)
    assert not any(event["event"] == "grasp_missed" for event in events)


def test_unconfirmed_cycle_does_not_make_retry_gap_nominal(tmp_path):
    source = make_episode(tmp_path, "retry_gap", task_steps=30)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"objects": ["bowl", "basket_region"],
                             "satisfied": i == 29}],
        "eef_position_m": [0.3 if i < 16 else 0.04, 0, 0.1],
        "gripper_command": 1 if 4 <= i < 8 or 16 <= i < 29 else -1,
        "grasped_objects": ["bowl"] if 16 <= i < 29 else [],
        "object_positions_m": {"bowl": [0, 0, 0.04]},
        "all_object_positions_m": {"bowl": [0, 0, 0.04]},
    } for i in range(30)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    labels = classify(source, oracle)
    retry_gap = next(label for label in labels
                     if label["task_step_range"][0] <= 10 < label["task_step_range"][1])
    assert retry_gap["role"] == "uncertain"
    assert retry_gap["verification"] == "candidate"
    assert any(label["role"] == "nominal" and
               label["task_step_range"][0] <= 16 < label["task_step_range"][1]
               for label in labels)


def test_wrong_object_release_is_not_a_verified_target_miss(tmp_path):
    source = make_episode(tmp_path, "wrong_release", task_steps=12)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"objects": ["bowl", "basket_region"],
                             "satisfied": False}],
        "eef_position_m": [0.08 if i < 9 else 0.2, 0, 0.1],
        "gripper_command": 1 if 4 <= i < 9 else -1,
        "grasped_objects": ["cup"] if 6 <= i < 9 else [],
        "object_positions_m": {"bowl": [0, 0, 0.04]},
        "all_object_positions_m": {"bowl": [0, 0, 0.04],
                                   "cup": [0.08, 0, 0.04]},
    } for i in range(12)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    events = [event for label in classify(source, oracle) for event in label["events"]]
    assert any(event["event"] == "wrong_object_grasped" for event in events)
    assert not any(event["event"] == "grasp_missed" for event in events)


def test_holding_wrong_object_and_unresolved_retry_are_not_recovery(tmp_path):
    source = make_episode(tmp_path, "wrong_object_retry", task_steps=30)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"objects": ["bowl", "basket_region"],
                             "satisfied": False}],
        "eef_position_m": [0.2, 0, 0.1],
        "gripper_command": 1 if 4 <= i < 16 or i >= 24 else -1,
        "grasped_objects": ["cup"] if 5 <= i < 16 else
                           ["bowl"] if i >= 25 else [],
        "object_positions_m": {"bowl": [0, 0, 0.04]},
        "all_object_positions_m": {"bowl": [0, 0, 0.04],
                                   "cup": [0.2, 0, 0.04]},
    } for i in range(30)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    labels = classify(source, oracle)
    assert any(event["event"] == "wrong_object_grasped"
               for label in labels for event in label["events"])
    assert not any(label["role"] == "recovery" and
                   label["task_step_range"][0] < 25
                   for label in labels)


def test_unsupported_goal_stays_unknown(tmp_path):
    source = make_episode(tmp_path, "unsupported", task_steps=12)
    rows = [{"task_step": i, "goal_predicates": [{"satisfied": i == 11}]}
            for i in range(12)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)
    path = oracle / "oracle-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["goal_states"] = [["next_to", "bowl", "plate"]]
    path.write_text(json.dumps(manifest))

    labels = classify(source, oracle)
    assert len(labels) == 1
    assert labels[0]["task_family"] == "unsupported"
    assert labels[0]["role"] == "uncertain"
    assert labels[0]["verification"] == "unknown"


def test_unresolved_release_is_candidate_not_proven_drop(tmp_path):
    source = make_episode(tmp_path, "release", task_steps=20)
    rows = [{
        "task_step": i,
        "goal_predicates": [{"objects": ["bowl", "basket_region"],
                             "satisfied": i == 19}],
        "eef_position_m": [0, 0, 0.1],
        "gripper_command": 1 if 4 <= i <= 8 else -1,
        "grasped_objects": ["bowl"] if 4 <= i <= 8 else [],
        "object_positions_m": {"bowl": [0, 0, 0.07]},
        "all_object_positions_m": {"bowl": [0, 0, 0.07]},
    } for i in range(20)]
    oracle = tmp_path / "oracle"
    write_oracle(source, oracle, rows)

    labels = classify(source, oracle)
    events = [event for label in labels for event in label["events"]]
    assert any(event["event"] == "target_release_unresolved" and
               event["verification"] == "candidate" for event in events)
    assert not any(event["event"] == "target_dropped" for event in events)
    assert any(label["role"] == "uncertain" and
               label["task_step_range"][0] <= 9 < label["task_step_range"][1]
               for label in labels)


@pytest.mark.parametrize('second_error', [False, True])
def test_wrong_object_hold_never_enters_positive_windows(tmp_path, second_error):
    source = make_episode(tmp_path, 'held_wrong', task_steps=48)
    rows = []
    for i in range(48):
        wrong = 5 <= i < 18 or (second_error and 23 <= i < 32)
        rows.append({
            'task_step': i,
            'goal_predicates': [{'objects': ['bowl', 'basket_region'], 'satisfied': i == 47}],
            'eef_position_m': [0.2, 0, 0.1], 'gripper_command': 1.0,
            'grasped_objects': ['cup'] if wrong else ['bowl'] if 40 <= i < 47 else [],
            'object_positions_m': {'bowl': [0, 0, 0.04]},
            'all_object_positions_m': {'bowl': [0, 0, 0.04], 'cup': [0.2, 0, 0.04]},
        })
    oracle = tmp_path / 'oracle'
    write_oracle(source, oracle, rows)
    labels = classify(source, oracle)
    roles = {i: label['role'] for label in labels
             for i in range(*label['task_step_range'])}
    assert roles[5] == 'error'
    assert all(roles[i] == 'uncertain' for i in range(6, 18))
    if second_error:
        assert roles[23] == 'error'
        assert all(roles[i] == 'uncertain' for i in range(18, 23))
        assert all(roles[i] == 'uncertain' for i in range(24, 32))
    assert roles[39] == 'recovery'
    starts = quality_starts(source, labels, 48, [(0, 48)])
    assert starts
    wrong_steps = {row['task_step'] for row in rows if row['grasped_objects'] == ['cup']}
    assert all(not wrong_steps.intersection(range(row['frame_index'], row['frame_index'] + 8))
               for row in starts)


def test_fixing_second_wrong_drawer_does_not_clear_first_error(tmp_path):
    source = make_episode(tmp_path, 'two_wrong_drawers', task_steps=40)
    rows = [{
        'task_step': i, 'goal_predicates': [{'satisfied': i == 39}],
        'eef_position_m': [0, 0, 0.2],
        'fixture_joint_positions': {
            'wooden_cabinet_1_top_level': 0.04 if i >= 5 else 0.0,
            'wooden_cabinet_1_bottom_level': 0.04 if 10 <= i < 25 else 0.0,
            'wooden_cabinet_1_middle_level': 0.05 if i == 39 else 0.0,
        },
    } for i in range(40)]
    oracle = tmp_path / 'oracle'
    write_oracle(source, oracle, rows)
    path = oracle / 'oracle-manifest.json'
    report = json.loads(path.read_text())
    report['target_objects'] = []
    report['goal_states'] = [['open', 'wooden_cabinet_1_middle_region']]
    path.write_text(json.dumps(report))
    labels = classify(source, oracle)
    assert not any(label['role'] in {'nominal', 'recovery'} and label['task_step_range'][1] > 5
                   for label in labels)
