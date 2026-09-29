from data_dump.quality_signals import approach_motion, behavior_signals, review_starts


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
        "inefficient_motion", "repeated_attempt", "slow_relative"}
    assert all(row["status"] == "candidate" for row in signals)
    starts = [{"frame_index": 0, "set": "nominal"},
              {"frame_index": 8, "set": "nominal"}]
    review = review_starts(starts, signals, source)
    assert len(review) == 2
    assert "repeated_attempt" in review[0]["reasons"]
    assert "repeated_attempt" not in review[1]["reasons"]
    assert labels[0]["role"] == "nominal"


def test_attempt_counter_resets_after_verified_progress():
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


def test_distant_attempts_do_not_make_one_long_review_interval():
    label = {"source": "/episode", "task_step_range": [0, 200],
             "role": "uncertain", "efficiency": {"reference_count": 0},
             "events": [{"event": "gripper_cycle_unresolved", "task_step": step}
                        for step in (10, 15, 150, 160)]}
    ranges = [signal["task_step_range"]
              for signal in behavior_signals([label], {}, {"bowl"})]
    assert ranges == [[10, 16], [150, 161]]
