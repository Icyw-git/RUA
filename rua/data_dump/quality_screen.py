"""Suggest WLA review ranges from saved episodes; never approve training data.

Only observed task actions and the official episode result are used. A gripper
cycle is a review cue, not proof that a grasp failed or that a step was bad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from libero_harness.audit import audit_episode
from .training_dump import discover, episode_steps, image_shape, write_jsonl


MIN_SPEED_REFERENCES = 3


def task_actions(events: list[dict]) -> list[dict]:
    """Return completed task actions in the same order as the WLA exporter."""
    return [event for event in events
            if event.get("event") == "step_completed" and event.get("phase") == "task"]


def evidence(action: dict, task_step: int, source: Path) -> dict:
    # attempt is the physical action index, including initialization. Video
    # frame at that index is the observation immediately before this action.
    return {"task_step": task_step, "video_frame": action["attempt"],
            "front_video": str(source / "front-control.mp4"),
            "wrist_video": str(source / "wrist-control.mp4")}


def gripper_signals(actions: list[dict], source: Path) -> list[dict]:
    """Flag close -> open -> close commands for visual review."""
    changes = []
    previous = -1.0  # Native LIBERO begins with an open gripper.
    for step, action in enumerate(actions):
        command = float(action["action"][-1])
        if command != previous:
            changes.append((step, command))
        previous = command
    signals = []
    for first, middle, last in zip(changes, changes[1:], changes[2:]):
        if [first[1], middle[1], last[1]] != [1.0, -1.0, 1.0]:
            continue
        steps = [first[0], middle[0], last[0]]
        signals.append({
            "kind": "gripper_close_open_close",
            "transition_steps": steps,
            "evidence": [evidence(actions[step], step, source) for step in steps],
        })
    return signals


def screen(sources: list[Path], output: Path) -> list[dict]:
    """Write one quality proposal per source; review.jsonl remains human-owned."""
    directories = discover(sources)
    if not directories:
        raise ValueError("No episode with result.json and environment-steps.jsonl found")
    rows = []
    for source in directories:
        result = json.loads((source / "result.json").read_text())
        trace_file = source / "environment-steps.jsonl"
        events = [json.loads(line) for line in
                  trace_file.read_text().splitlines() if line]
        actions = task_actions(events)
        signals = gripper_signals(actions, source)
        suite = result.get("pilot", {}).get("suite", result.get("suite"))
        eligible = False
        reason = None
        try:
            if "pilot" not in result:
                raise ValueError("Legacy rollout schema")
            audit_episode(source)
            if result.get("status") != "completed" or result.get("official_success") is not True:
                raise ValueError("Task did not complete successfully")
            episode_steps(source, result, events)
            if image_shape(source) != image_shape(source, "wrist"):
                raise ValueError("Front/wrist image shapes differ")
            eligible = True
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            reason = str(exc)
        rows.append({
            "screen_version": 2, "source": str(source), "scope": result.get("scope"),
            "source_trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
            "suite": suite, "task_id": result.get("task_id"),
            "task_instruction": result.get("task_instruction"),
            "official_success": result.get("official_success"),
            "task_steps": len(actions),
            "eligible_for_wla_review": eligible, "ineligible_reason": reason,
            "speed_reference_count": 0, "speed_comparison": None,
            "signals": signals,
            # Successful recovery may contain the very gripper cycles we flag.
            # Keep it available for review; only a human may trim the range.
            "review_task_step_range": [0, len(actions)] if eligible else None,
        })

    cohorts = {}
    for row in rows:
        if row["eligible_for_wla_review"] and row["suite"] is not None and row["task_id"] is not None:
            key = (row["scope"], row["suite"], row["task_id"])
            cohorts.setdefault(key, []).append(row)
    for cohort in cohorts.values():
        for row in cohort:
            references = [other["task_steps"] for other in cohort if other is not row]
            row["speed_reference_count"] = len(references)
            if len(references) >= MIN_SPEED_REFERENCES:
                baseline = median(references)
                row["speed_comparison"] = {
                    "reference_median_task_steps": baseline,
                    "task_steps_over_reference_median": round(row["task_steps"] / baseline, 3),
                    "shorter_than_references": sum(row["task_steps"] < count for count in references),
                    "longer_than_references": sum(row["task_steps"] > count for count in references),
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Episode or run directories")
    parser.add_argument("--output", type=Path, required=True, help="quality-proposals.jsonl path")
    args = parser.parse_args()
    rows = screen(args.sources, args.output)
    print(json.dumps({"output": str(args.output), "episodes": len(rows),
                      "eligible_for_review": sum(row["eligible_for_wla_review"] for row in rows),
                      "with_signals": sum(bool(row["signals"]) for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
