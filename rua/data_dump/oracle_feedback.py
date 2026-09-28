"""Replay saved LIBERO actions to collect simulator-only step feedback.

The replay never calls a policy. It verifies each resulting robot pose and
official success against the saved execution before writing any feedback.
Simulator truth stays in a separate artifact and is never a planner input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from libero_harness.paths import save_json
from .training_dump import write_jsonl


def replay(source: Path, output: Path) -> dict:
    source = source.resolve()
    result = json.loads((source / "result.json").read_text())
    trace_file = source / "environment-steps.jsonl"
    events = [json.loads(line) for line in trace_file.read_text().splitlines() if line]
    completed = [event for event in events if event.get("event") == "step_completed"]
    if result.get("status") != "completed" or result.get("uncertain_step"):
        raise ValueError("Only completed episodes with certain actions can be replayed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    from libero.envs import OffScreenRenderEnv

    resolution = result["pilot"]["render_resolution"]
    env = OffScreenRenderEnv(bddl_file_name=str(source / "task.bddl"),
                             camera_heights=resolution, camera_widths=resolution)
    rows = []
    max_pose_error_m = 0.0
    task_step = 0
    try:
        env.seed(0)  # Match WLA's get_libero_env constructor.
        env.reset()
        env.set_init_state(np.load(source / "initial_state.npy"))
        goal_states = env.env.parsed_problem["goal_state"]
        names = sorted({name for state in goal_states for name in state[1:]
                        if name in env.env.obj_body_id})
        for event in completed:
            observation, _, _, _ = env.step(event["action"])
            pose_error = float(np.linalg.norm(
                np.asarray(observation["robot0_eef_pos"]) - event["eef_pose"][:3]))
            max_pose_error_m = max(max_pose_error_m, pose_error)
            if pose_error > 1e-4 or bool(env.check_success()) != event["official_success"]:
                raise ValueError(f"Replay diverged at attempt {event['attempt']}")
            if event["phase"] != "task":
                continue
            rows.append({
                "source": str(source), "task_step": task_step,
                "attempt": event["attempt"],
                "goal_predicates": [
                    {"predicate": state[0], "objects": state[1:],
                     "satisfied": bool(env.env._eval_predicate(state))}
                    for state in goal_states
                ],
                "object_positions_m": {
                    name: np.asarray(env.sim.data.body_xpos[env.env.obj_body_id[name]],
                                     dtype=float).tolist() for name in names
                },
                "official_success": bool(event["official_success"]),
            })
            task_step += 1
    finally:
        env.close()
    if task_step != result["task_steps"]:
        raise ValueError("Replayed task step count differs from result.json")
    report = {
        "version": 1, "source": str(source), "scope": result.get("scope"),
        "task_steps": task_step, "verified_max_eef_position_error_m": max_pose_error_m,
        "source_trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
        "feedback_file": "oracle-feedback.jsonl",
        "observation_time": "after each completed task action",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "oracle-feedback.jsonl", rows)
    save_json(output / "oracle-manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Recorded episode directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
