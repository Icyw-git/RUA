"""Label replayed LIBERO actions using simulator events, without policy input.

Only verified nominal and recovery actions are eligible for WLA imitation.
Gripper commands without contact are review cues, not proven mistakes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .training_dump import write_jsonl


VERSION = "rules-006"


def _carried(rows: list[dict], name: str, radius: float) -> list[bool]:
    """Detect lifted, sustained object motion coupled to the closed gripper."""
    positions = [np.asarray(row["all_object_positions_m"][name], dtype=float) for row in rows]
    eef = [np.asarray(row["eef_position_m"], dtype=float) for row in rows]
    relative = [position - hand for position, hand in zip(positions, eef)]
    open_motion = [np.linalg.norm(b - a) for i, (a, b) in
                   enumerate(zip(positions, positions[1:]))
                   if rows[i]["gripper_command"] < 0 and rows[i + 1]["gripper_command"] < 0]
    motion_floor = max(0.012, 6 * float(np.median(open_motion)) if open_motion else 0.0)
    lift_floor = max(0.015, 0.35 * radius)
    distance_cap = max(0.12, 2.5 * radius + 0.06)
    anchor_cap = max(0.03, 0.6 * radius)
    active = [False] * len(rows)
    for width in (4, 8, 12):
        for start in range(len(rows) - width):
            end = start + width
            object_move = np.linalg.norm(positions[end] - positions[start])
            if (not all(row["gripper_command"] > 0 for row in rows[start:end + 1])
                    or object_move < motion_floor
                    or np.linalg.norm(eef[end] - eef[start]) < motion_floor
                    or np.linalg.norm(relative[start]) > distance_cap
                    or max(np.linalg.norm(relative[i] - relative[start])
                           for i in range(start, end + 1)) > max(0.018, 0.25 * object_move)
                    or max(position[2] for position in positions[start:end + 1])
                    - positions[0][2] < lift_floor):
                continue
            # The window verifies sustained carrying, but its early frames may
            # only show the object being pushed before it is lifted.
            for i in range(start, end + 1):
                if positions[i][2] - positions[0][2] >= lift_floor:
                    active[i] = True
    # A held object can pause for many steps. Extend through those pauses only
    # while the gripper remains closed and the relative pose stays stable.
    anchor = None
    for i, is_active in enumerate(active):
        if is_active:
            anchor = relative[i]
        elif anchor is not None and rows[i]["gripper_command"] > 0 and (
                np.linalg.norm(relative[i] - anchor) < anchor_cap):
            active[i] = True
        else:
            anchor = None
    return active


def _predicates_met(row: dict) -> bool:
    return all(item["satisfied"] for item in row["goal_predicates"])


def _segments(steps: list[dict]) -> list[dict]:
    segments = []
    for step in steps:
        key = (step["phase"], step["role"], step["verification"])
        if not segments or segments[-1]["_key"] != key:
            segments.append({"_key": key, "task_step_range": [step["task_step"],
                                                       step["task_step"] + 1],
                             "phase": step["phase"], "role": step["role"],
                             "verification": step["verification"], "events": []})
        else:
            segments[-1]["task_step_range"][1] += 1
        segments[-1]["events"].extend(step["events"])
    for segment in segments:
        del segment["_key"]
    return segments


def classify(source: Path, oracle_dir: Path) -> list[dict]:
    source = source.resolve()
    report = json.loads((oracle_dir / "oracle-manifest.json").read_text())
    oracle_file = oracle_dir / report["feedback_file"]
    rows = [json.loads(line) for line in oracle_file.read_text().splitlines() if line]
    trace_hash = hashlib.sha256((source / "environment-steps.jsonl").read_bytes()).hexdigest()
    if (report["source"] != str(source) or report["source_trace_sha256"] != trace_hash
            or len(rows) != report["task_steps"]
            or [row["task_step"] for row in rows] != list(range(len(rows)))):
        raise ValueError("Oracle feedback does not match the source episode")

    targets = set(report.get("target_objects", []))
    goal_states = report.get("goal_states", [])
    predicates = {state[0].lower() for state in goal_states}
    articulated = bool(predicates) and predicates <= {"open", "close"}
    placement = bool(predicates) and predicates <= {"in", "on"}
    family = "articulation" if articulated else "placement" if placement else "unsupported"
    required = (("goal_predicates", "eef_position_m", "fixture_joint_positions")
                if articulated else
                ("goal_predicates", "eef_position_m", "object_positions_m",
                 "all_object_positions_m", "grasped_objects", "gripper_command")
                if placement else ("goal_predicates",))
    covered = report.get("version", 1) >= 2 and all(
        all(field in row for field in required) for row in rows)
    if family == "unsupported" or (placement and not targets):
        covered = False
    target_joints = set()
    if articulated and rows:
        joint_names = rows[0].get("fixture_joint_positions", {})
        for state in goal_states:
            stem = state[1].removesuffix("_region")
            matches = {name for name in joint_names if name.startswith(stem)}
            covered &= len(matches) == 1
            target_joints.update(matches)

    steps = [{"task_step": i, "phase": "articulate" if articulated else
              "approach" if placement else "unknown",
              "role": "uncertain", "verification": "unknown", "events": []}
             for i in range(len(rows))]
    if not covered:
        return _finish(source, trace_hash, oracle_file, rows, steps, required, covered, family)

    goal = [_predicates_met(row) for row in rows]
    target_grasp = [False] * len(rows)
    target_by_name = {}
    wrong_grasp = [False] * len(rows)
    if not articulated:
        radii = report.get("object_horizontal_radius_m", {})
        carried = {name: _carried(rows, name, float(radii.get(name, 0.04)))
                   for name in rows[0]["all_object_positions_m"]}
        target_by_name = {
            name: [name in row["grasped_objects"] or carried[name][i]
                   for i, row in enumerate(rows)]
            for name in targets
        }
        target_grasp = [any(mask[i] for mask in target_by_name.values())
                        for i in range(len(rows))]
        wrong_grasp = [bool(set(row["grasped_objects"]) - targets) or
                       any(mask[i] for name, mask in carried.items() if name not in targets)
                       for i, row in enumerate(rows)]
    errors = []
    wrong_drawers = {}
    gained = []
    confirmed_grasps = []
    error_spans = []
    unresolved_releases = []
    unresolved_cycles = []
    for i, row in enumerate(rows):
        if goal[i] and (i == 0 or not goal[i - 1]):
            steps[i]["events"].append({"task_step": i, "event": "goal_gained",
                                        "evidence": "goal_predicates"})
            gained.append(i)
        if i and not goal[i] and goal[i - 1]:
            steps[i]["events"].append({"task_step": i, "event": "goal_lost",
                                        "evidence": "goal_predicates"})
            errors.append(i)
        if articulated:
            continue
        # Two adjacent post-action contact readings avoid single-frame contact noise.
        if (i + 1 < len(rows) and target_grasp[i] and target_grasp[i + 1]
                and (i == 0 or not target_grasp[i - 1])):
            steps[i]["events"].append({"task_step": i, "event": "target_grasped",
                                        "evidence": "grasped_objects_or_relative_motion"})
            confirmed_grasps.append(i)
        if (i + 1 < len(rows) and wrong_grasp[i] and wrong_grasp[i + 1]
                and (i == 0 or not wrong_grasp[i - 1])):
            steps[i]["events"].append({"task_step": i, "event": "wrong_object_grasped",
                                        "evidence": "grasped_objects_or_relative_motion"})
            errors.append(i)
        for name, held in target_by_name.items():
            if not (i >= 2 and held[i - 2] and held[i - 1] and not held[i]):
                continue
            placed = any(any(item["satisfied"] and
                                 item.get("objects", [])[:1] == [name]
                                 for item in future["goal_predicates"])
                         for future in rows[i:i + 4])
            if len(targets) == 1:
                placed |= any(goal[i:i + 4])
            if not placed:
                radius = float(radii.get(name, 0.04))
                before_z = rows[i - 1]["all_object_positions_m"][name][2]
                lowest_z = min(future["all_object_positions_m"][name][2]
                               for future in rows[i:i + 8])
                fell = before_z - lowest_z > max(0.02, 0.4 * radius)
                steps[i]["events"].append({
                    "task_step": i,
                    "event": "target_dropped" if fell else "target_release_unresolved",
                    "object": name,
                    "verification": "verified" if fell else "candidate",
                    "evidence": "grasp_state+object_height+goal_predicates",
                })
                if fell:
                    errors.append(i)
                else:
                    unresolved_releases.append((i, name))

    if articulated:
        for name in rows[0]["fixture_joint_positions"]:
            if name in target_joints:
                continue
            if not any(name.startswith(target.rsplit("_", 2)[0]) for target in target_joints):
                continue
            initial = rows[0]["fixture_joint_positions"][name]
            for i in range(1, len(rows)):
                previous = rows[i - 1]["fixture_joint_positions"][name]
                current = rows[i]["fixture_joint_positions"][name]
                if abs(current - initial) > 0.03 and abs(previous - initial) <= 0.03:
                    steps[i]["events"].append({"task_step": i, "event": "wrong_drawer_moved",
                                                "evidence": f"fixture_joint_positions:{name}"})
                    errors.append(i)
                    wrong_drawers[i] = (name, initial)
        # A drawer trajectory without a verified goal transition remains unknown.
        if gained:
            for step in steps[:gained[-1] + 1]:
                step["role"], step["verification"] = "nominal", "verified"
    else:
        # On a successful run, completed goal predicates anchor the full sequence.
        # On a failed run, only the approach to an observed grasp is a positive subgoal.
        positive_end = gained[-1] if gained else (confirmed_grasps[0] if confirmed_grasps else -1)
        for step in steps[:positive_end + 1]:
            step["role"], step["verification"] = "nominal", "verified"
        last_held = None
        for i, step in enumerate(steps):
            if target_grasp[i]:
                last_held = next(name for name, mask in target_by_name.items() if mask[i])
                step["phase"] = "transport"
            elif last_held is not None:
                placed_soon = any(any(item["satisfied"] and
                                      item.get("objects", [])[:1] == [last_held]
                                      for item in future["goal_predicates"])
                                  for future in rows[i:i + 4])
                if len(targets) == 1:
                    placed_soon |= any(goal[i:i + 4])
                step["phase"] = "place" if placed_soon else "grasp"
                placed_now = any(item["satisfied"] and
                                 item.get("objects", [])[:1] == [last_held]
                                 for item in rows[i]["goal_predicates"])
                if placed_now or (len(targets) == 1 and goal[i]):
                    last_held = None
            elif goal[i]:
                step["phase"] = "place"

        # Closing and reopening without any verified grasp is ambiguous: a
        # candidate missed grasp, never a verified error or recovery.
        close_at = None
        saw_grasp = False
        for i, row in enumerate(rows):
            command = row["gripper_command"]
            previous = rows[i - 1]["gripper_command"] if i else -1.0
            if command > 0 and previous < 0:
                close_at, saw_grasp = i, False
            if close_at is not None:
                saw_grasp |= target_grasp[i]
            if command < 0 and previous > 0 and close_at is not None:
                if not saw_grasp and not goal[i]:
                    verified = False
                    for name in targets:
                        radius = float(radii.get(name, 0.04))
                        start_pos = np.asarray(rows[close_at]["object_positions_m"][name])
                        initial_distance = np.linalg.norm(
                            start_pos - rows[close_at]["eef_position_m"])
                        retreated = (np.linalg.norm(start_pos - rows[i]["eef_position_m"])
                                     > initial_distance + max(0.04, 0.8 * radius))
                        stationary = max(np.linalg.norm(
                            np.asarray(rows[j]["object_positions_m"][name]) - start_pos)
                            for j in range(close_at, i + 1)) < max(0.01, 0.25 * radius)
                        verified |= (initial_distance < max(0.07, radius + 0.065)
                                     and retreated and stationary)
                    verified &= not any(wrong_grasp[close_at:i + 1])
                    steps[i]["events"].append({"task_step": i,
                                                "event": "grasp_missed" if verified
                                                else "gripper_cycle_unresolved",
                                                "verification": "verified" if verified else "candidate",
                                                "evidence": "gripper_command+object_motion+eef_retreat"})
                    for step in steps[close_at:i + 1]:
                        step["role"], step["verification"] = (
                            ("error", "verified") if verified else ("uncertain", "candidate"))
                    if verified:
                        error_spans.append((close_at, i))
                        errors.append(i)
                    else:
                        unresolved_cycles.append((close_at, i))
                close_at = None

    unresolved = []
    for error in sorted(set(errors)):
        steps[error]["role"], steps[error]["verification"] = "error", "verified"
        if error in wrong_drawers:
            name, initial = wrong_drawers[error]
            resume = next((i for i in range(error + 1, len(rows))
                           if abs(rows[i]["fixture_joint_positions"][name] - initial) < 0.015),
                          None)
        else:
            resume = next((i for i in sorted(set(confirmed_grasps + gained))
                           if i > error and not wrong_grasp[i]), None)
        if resume is not None:
            for step in steps[error + 1:resume + 1]:
                if (step["role"] != "error" and step["verification"] != "candidate"
                        and not wrong_grasp[step["task_step"]]):
                    step["role"], step["verification"] = "recovery", "verified"
        else:
            unresolved.append(error)
    for start, end in error_spans:
        for step in steps[start:end + 1]:
            step["role"], step["verification"] = "error", "verified"
    for release, name in unresolved_releases:
        resume = next((i for i in range(release + 1, len(rows))
                       if target_by_name[name][i] or any(
                           item["satisfied"] and item.get("objects", [])[:1] == [name]
                           for item in rows[i]["goal_predicates"])), len(rows) - 1)
        for step in steps[release:resume + 1]:
            if step["role"] == "nominal":
                step["role"], step["verification"] = "uncertain", "candidate"
    for _, end in unresolved_cycles:
        resume = next((i for i in sorted(set(confirmed_grasps + gained))
                       if i > end), len(rows))
        for step in steps[end + 1:resume]:
            if step["role"] in {"nominal", "recovery"}:
                step["role"], step["verification"] = "uncertain", "candidate"
    if unresolved:
        for step in steps[min(unresolved) + 1:]:
            if step["role"] != "error":
                step["role"], step["verification"] = "uncertain", "unknown"
    return _finish(source, trace_hash, oracle_file, rows, steps, required, covered, family)


def _finish(source: Path, trace_hash: str, oracle_file: Path, rows: list[dict],
            steps: list[dict], required: tuple[str, ...], covered: bool,
            family: str) -> list[dict]:
    oracle_hash = hashlib.sha256(oracle_file.read_bytes()).hexdigest()
    segments = _segments(steps)
    for segment in segments:
        start, end = segment["task_step_range"]
        if all("eef_position_m" in row for row in rows[start:end]):
            positions = [np.asarray(row["eef_position_m"], dtype=float)
                         for row in rows[start:end]]
            movements = [float(np.linalg.norm(b - a))
                         for a, b in zip(positions, positions[1:])]
            path_m = round(sum(movements), 4)
            stationary = sum(distance < 0.002 for distance in movements)
        else:
            path_m, stationary = None, None
        basis = {
            "nominal": "later_subgoal_reached_without_detected_error",
            "recovery": "confirmed_error_followed_by_correction",
            "error": "observed_physical_error",
            "uncertain": "insufficient_evidence",
        }[segment["role"]]
        segment.update({"version": 1, "labeler_version": VERSION,
                        "task_family": family,
                        "verification_basis": basis,
                        "source": str(source), "source_trace_sha256": trace_hash,
                        "oracle_feedback_sha256": oracle_hash,
                        "coverage": "complete" if covered else "partial",
                        "coverage_fields": {field: all(field in row for row in rows)
                                            for field in required},
                        "efficiency": {"eef_path_m": path_m,
                                       "stationary_steps": stationary,
                                       "speed_ratio": None, "path_ratio": None,
                                       "reference_count": 0}})
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = classify(args.source, args.oracle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, labels)
    print(json.dumps({"source": str(args.source.resolve()), "segments": len(labels),
                      "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
