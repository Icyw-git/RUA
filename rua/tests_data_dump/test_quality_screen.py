import json

from data_dump.quality_screen import screen
from test_training_dump import make_episode


def set_gripper_commands(source, transitions):
    path = source / "environment-steps.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    command = -1
    task_step = 0
    for event in events:
        if event["phase"] != "task":
            continue
        if event["event"] == "step_started":
            command = transitions.get(task_step, command)
        event["action"][-1] = command
        if event["event"] == "step_completed":
            task_step += 1
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_screen_flags_regrasp_without_discarding_successful_actions(tmp_path):
    source = make_episode(tmp_path, "recovery", task_steps=30)
    set_gripper_commands(source, {4: 1, 8: -1, 12: 1, 16: -1})
    output = tmp_path / "quality-proposals.jsonl"

    rows = screen([source], output)

    assert rows == [json.loads(output.read_text())]
    row = rows[0]
    assert row["eligible_for_wla_review"] is True
    assert row["speed_comparison"] is None
    assert row["speed_reference_count"] == 0
    assert row["signals"] == [{
        "kind": "gripper_close_open_close", "transition_steps": [4, 8, 12],
        "evidence": [{"task_step": step, "video_frame": step + 2,
                      "front_video": str(source / "front-control.mp4"),
                      "wrist_video": str(source / "wrist-control.mp4")}
                     for step in (4, 8, 12)],
    }]
    assert row["review_task_step_range"] == [0, 30]
    assert "approved_for_wla" not in row


def test_speed_uses_successful_same_task_and_scope_only(tmp_path):
    sources = [make_episode(tmp_path, f"success-{steps}", task_steps=steps)
               for steps in (20, 30, 40, 50)]
    failure = make_episode(tmp_path, "failure", success=False, task_steps=12)
    other_task = make_episode(tmp_path, "other-task", task_steps=14)
    result_path = other_task / "result.json"
    result = json.loads(result_path.read_text())
    result["task_id"] = 2
    result_path.write_text(json.dumps(result))
    output = tmp_path / "quality-proposals.jsonl"

    rows = screen([tmp_path], output)

    by_source = {row["source"]: row for row in rows}
    slow = by_source[str(sources[-1])]
    assert slow["speed_reference_count"] == 3
    assert slow["speed_comparison"] == {
        "reference_median_task_steps": 30,
        "task_steps_over_reference_median": 1.667,
        "shorter_than_references": 0, "longer_than_references": 3,
    }
    assert slow["review_task_step_range"] == [0, 50]
    assert by_source[str(failure)]["eligible_for_wla_review"] is False
    assert by_source[str(failure)]["review_task_step_range"] is None
    assert by_source[str(other_task)]["speed_comparison"] is None
