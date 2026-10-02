from copy import deepcopy
from math import cos, radians, sin

import pytest

from data_dump.quality_signals import (
    approach_motion, behavior_signals, low_progress_signals, review_starts,
)


def test_approach_motion_distinguishes_progress_from_backtracking():
    def rows(xs):
        return [{"eef_position_m": [x, 0, 0],
                 "object_positions_m": {"bowl": [0, 0, 0]}} for x in xs]

    direct = approach_motion(rows([0.3, 0.2, 0.1, 0.02]), {"bowl"}, 0.04)
    detour = approach_motion(rows([0.3, 0.18, 0.29, 0.17, 0.02]), {"bowl"}, 0.04)
    assert direct["backtrack_m"] == 0
    assert detour["backtrack_m"] > 0.1
    assert detour["path_over_progress"] > direct["path_over_progress"]


def test_approach_tracks_one_target_in_multi_object_task():
    rows = [{"eef_position_m": [x, 0, 0],
             "object_positions_m": {"cup": [0, 0, 0], "bowl": [0.4, 0, 0]}}
            for x in (0.32, 0.28, 0.1, 0.02)]
    motion = approach_motion(rows, {"cup", "bowl"}, 0.04)
    assert motion["target_object"] == "cup"
    assert motion["target_progress_m"] == 0.3


def test_behavior_signals_are_candidates_and_only_review_exported_starts():
    source = "/episode"
    labels = [{
        "source": source, "task_step_range": [0, 16], "role": "nominal",
        "events": [{"event": "gripper_cycle_unresolved", "task_step": 3},
                   {"event": "gripper_cycle_unresolved", "task_step": 7}],
        "efficiency": {"eef_path_m": 0.4, "reference_count": 5,
                       "speed_ratio": 1.8,
                       "approach_motion": {"backtrack_m": 0.11,
                                            "path_over_progress": 4.5}},
    }]
    signals = behavior_signals(labels, {"bowl": 0.04}, {"bowl"})
    assert {row["kind"] for row in signals} == {
        "inefficient_motion", "repeated_gripper_cycle", "slow_relative"}
    assert all(row["status"] == "candidate" for row in signals)
    cycle = next(row for row in signals if row["kind"] == "repeated_gripper_cycle")
    assert cycle["version"] == "signals-004"
    assert cycle["evidence"] == {"event_steps": [3, 7], "cycle_count": 2}
    starts = [{"frame_index": 0, "set": "nominal"},
              {"frame_index": 8, "set": "nominal"}]
    review = review_starts(starts, signals, source)
    assert len(review) == 2
    assert "repeated_gripper_cycle" in review[0]["reasons"]
    assert "repeated_gripper_cycle" not in review[1]["reasons"]
    assert labels[0]["role"] == "nominal"


def test_cycle_counter_resets_after_verified_progress():
    label = {"source": "/episode", "task_step_range": [0, 20],
             "role": "uncertain", "efficiency": {"reference_count": 0},
             "events": [
                 {"event": "gripper_cycle_unresolved", "task_step": 2},
                 {"event": "target_grasped", "task_step": 5},
                 {"event": "grasp_missed", "task_step": 11},
             ]}
    assert not behavior_signals([label], {}, {"bowl"})


def test_uncertain_motion_is_not_called_inefficient():
    label = {"source": "/episode", "task_step_range": [0, 50],
             "role": "uncertain", "events": [],
             "efficiency": {"eef_path_m": 1.0, "reference_count": 0,
                            "approach_motion": {"backtrack_m": 0.5,
                                                "path_over_progress": 10.0}}}
    assert not behavior_signals([label], {"bowl": 0.04}, {"bowl"})


def test_distant_cycles_do_not_make_one_long_review_interval():
    label = {"source": "/episode", "task_step_range": [0, 200],
             "role": "uncertain", "efficiency": {"reference_count": 0},
             "events": [{"event": "gripper_cycle_unresolved", "task_step": step}
                        for step in (10, 15, 150, 160)]}
    ranges = [signal["task_step_range"]
              for signal in behavior_signals([label], {}, {"bowl"})]
    assert ranges == [[10, 16], [150, 161]]


def quiet_episode(length=50):
    rows = [{"eef_position_m": [0.1, 0, 0],
             "eef_quaternion_xyzw": [0, 0, 0, 1],
             "gripper_qpos_m": [0.04, -0.04], "gripper_command": -1,
             "grasped_objects": [], "goal_predicates": [{"satisfied": False}],
             "fixture_joint_positions": {},
             "all_object_positions_m": {"bowl": [0, 0, 0], "cup": [1, 0, 0]},
             "object_quaternions_wxyz": {"bowl": [1, 0, 0, 0], "cup": [1, 0, 0, 0]}}
            for _ in range(length)]
    labels = [{"source": "/episode", "task_step_range": [0, length],
               "task_family": "placement", "task_phase": "approach",
               "role": "nominal", "verification": "verified", "coverage": "complete"}]
    return labels, rows


def test_quiet_interval_is_only_a_candidate_without_mutating_inputs():
    labels, rows = quiet_episode()
    before = deepcopy((labels, rows))
    signals = low_progress_signals(labels, rows, {"bowl"})
    assert signals[0]["task_step_range"] == [1, 50]
    assert signals[0]["evidence"]["steps"] == 49
    assert signals[0]["evidence"]["reason"] == "low_progress"
    assert signals[0]["evidence"]["action_role"] == "nominal"
    assert signals[0]["status"] == "candidate"
    assert (labels, rows) == before


def test_uncertain_approach_can_propose_review_without_changing_its_role():
    labels, rows = quiet_episode()
    labels[0].update(role="uncertain", verification="unknown")
    signals = low_progress_signals(labels, rows, {"bowl"})
    assert [signal["task_step_range"] for signal in signals] == [[1, 50]]
    assert signals[0]["evidence"]["action_role"] == "uncertain"
    assert labels[0]["role"] == "uncertain"


@pytest.mark.parametrize("role", ["nominal", "uncertain"])
@pytest.mark.parametrize("motion", ["hand", "wrist", "fingers", "object", "other_object", "object_rotation"])
def test_useful_continuous_adjustments_are_not_low_progress(motion, role):
    labels, rows = quiet_episode()
    labels[0].update(role=role, verification="unknown" if role == "uncertain" else "verified")
    for i, row in enumerate(rows):
        if motion == "hand":
            row["eef_position_m"][0] += i * 0.0005
        elif motion == "wrist":
            angle = radians(i) / 2
            row["eef_quaternion_xyzw"] = [0, 0, sin(angle), cos(angle)]
        elif motion == "fingers":
            row["gripper_qpos_m"][0] += i * 0.0001
        elif motion in {"object", "other_object"}:
            name = "bowl" if motion == "object" else "cup"
            row["all_object_positions_m"][name][0] += i * 0.0002
        else:
            angle = radians(i) / 2
            row["object_quaternions_wxyz"]["bowl"] = [cos(angle), 0, 0, sin(angle)]
    assert not low_progress_signals(labels, rows, {"bowl"})


@pytest.mark.parametrize("field,value", [("task_phase", "transport"), ("task_phase", "place"),
    ("task_family", "articulation"), ("role", "error"),
    ("verification", "candidate"), ("coverage", "partial")])
def test_low_progress_does_not_expand_into_other_roles_or_phases(field, value):
    labels, rows = quiet_episode()
    labels[0][field] = value
    assert not low_progress_signals(labels, rows, {"bowl"})


@pytest.mark.parametrize("change", ["held", "closed", "goal", "fixture"])
def test_interaction_evidence_protects_window(change):
    labels, rows = quiet_episode(25)
    if change == "held":
        rows[12]["grasped_objects"] = ["bowl"]
    elif change == "closed":
        rows[12]["gripper_command"] = 1
    elif change == "goal":
        rows[12]["goal_predicates"][0]["satisfied"] = True
    else:
        rows[12]["fixture_joint_positions"] = {"joint": 0.01}
    assert not low_progress_signals(labels, rows, {"bowl"})


def test_short_pause_missing_evidence_and_ambiguous_target_are_skipped():
    labels, rows = quiet_episode(24)
    assert not low_progress_signals(labels, rows, {"bowl"})
    labels, rows = quiet_episode()
    assert not low_progress_signals(labels, rows, {"bowl", "cup"})
    del rows[0]["eef_quaternion_xyzw"]
    assert not low_progress_signals(labels, rows, {"bowl"})


def test_following_grasp_resolves_multi_object_target():
    labels, rows = quiet_episode(51)
    labels[0]["task_step_range"][1] = 50
    rows[50]["grasped_objects"] = ["cup"]
    signals = low_progress_signals(labels, rows, {"bowl", "cup"})
    assert signals[0]["evidence"]["target_object"] == "cup"
    assert signals[0]["task_step_range"] == [1, 50]


def test_motion_into_a_pause_is_excluded_and_drift_is_not_merged():
    labels, rows = quiet_episode(100)
    for i, row in enumerate(rows):
        row["eef_position_m"][0] += min(max(i - 30, 0), 10) * 0.01
    signals = low_progress_signals(labels, rows, {"bowl"})
    assert [s["task_step_range"] for s in signals] == [[1, 31], [41, 100]]
    for i, row in enumerate(rows):
        row["eef_position_m"][0] = 0.1 + i * 0.0001
    signals = low_progress_signals(labels, rows, {"bowl"})
    assert signals
    assert all(s["evidence"]["eef_excursion_m"] <= 0.005 for s in signals)
    assert all(s["task_step_range"] != [1, 100] for s in signals)


def test_quaternion_sign_changes_are_not_rotation():
    labels, rows = quiet_episode()
    for row in rows[::2]:
        row["eef_quaternion_xyzw"] = [0, 0, 0, -1]
    assert low_progress_signals(labels, rows, {"bowl"})
