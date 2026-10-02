"""Describe inefficient-looking motion without treating it as a physical error."""
from __future__ import annotations

from collections import Counter
from math import dist

import numpy as np


VERSION = "signals-004"

# Candidate tolerances at the existing 20 Hz action rate, not training filters.
LOW_PROGRESS_STEPS = 24
HAND_POSITION_TOLERANCE_M = 0.005
OBJECT_POSITION_TOLERANCE_M = 0.002
ROTATION_TOLERANCE_DEG = 3.0
GRIPPER_TOLERANCE_M = 0.001


def _position_excursion(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.linalg.norm(values - values[0], axis=1).max())


def _rotation_excursion(quaternions) -> float:
    quaternions = np.asarray(quaternions, dtype=float)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    cosines = np.clip(np.abs(quaternions @ quaternions[0]), 0, 1)
    return float(np.degrees(2 * np.arccos(cosines)).max())


def _low_progress_evidence(rows: list[dict]) -> dict | None:
    if any(row["grasped_objects"] or row["gripper_command"] >= 0 for row in rows):
        return None
    goals = [[item["satisfied"] for item in row["goal_predicates"]] for row in rows]
    if any(goal != goals[0] for goal in goals[1:]):
        return None
    # Any fixture motion may be useful interaction, including an intermediate goal.
    if any(row.get("fixture_joint_positions", {}) != rows[0].get("fixture_joint_positions", {})
           for row in rows[1:]):
        return None
    hand = _position_excursion([row["eef_position_m"] for row in rows])
    rotation = _rotation_excursion([row["eef_quaternion_xyzw"] for row in rows])
    fingers = _position_excursion([row["gripper_qpos_m"] for row in rows])
    objects = rows[0]["all_object_positions_m"]
    object_move = max((_position_excursion([row["all_object_positions_m"][name] for row in rows])
                       for name in objects), default=0.0)
    object_rotation = max((_rotation_excursion([row["object_quaternions_wxyz"][name] for row in rows])
                           for name in objects), default=0.0)
    if (hand > HAND_POSITION_TOLERANCE_M or rotation > ROTATION_TOLERANCE_DEG
            or fingers > GRIPPER_TOLERANCE_M or object_move > OBJECT_POSITION_TOLERANCE_M
            or object_rotation > ROTATION_TOLERANCE_DEG):
        return None
    return {"reason": "low_progress", "steps": len(rows) - 1,
            "eef_excursion_m": hand, "eef_rotation_deg": rotation,
            "gripper_excursion_m": fingers, "object_excursion_m": object_move,
            "object_rotation_deg": object_rotation}


def low_progress_signals(labels: list[dict], rows: list[dict], targets: set[str]) -> list[dict]:
    """Find sustained quiet approach intervals; never change action roles or starts."""
    required = {"eef_quaternion_xyzw", "gripper_qpos_m", "object_quaternions_wxyz"}
    if not rows or not all(required <= row.keys() for row in rows):
        return []
    signals = []
    for label in labels:
        role = label["role"]
        if (label.get("task_family") != "placement" or label.get("task_phase") != "approach"
                or role not in {"nominal", "recovery", "uncertain"}
                or (role != "uncertain" and label.get("verification") != "verified")
                or label.get("coverage") != "complete"):
            continue
        start, end = label["task_step_range"]
        # Multiple targets require an immediately following, unambiguous grasp.
        candidates = targets if len(targets) == 1 else (
            set(rows[end]["grasped_objects"]) & targets if end < len(rows) else set())
        if len(candidates) != 1:
            continue
        target = next(iter(candidates))
        # Rows are post-action states. Include the preceding state so that a
        # movement into a quiet interval is not itself called low progress.
        cursor = max(start, 1)
        while cursor + LOW_PROGRESS_STEPS <= end:
            stop = cursor + LOW_PROGRESS_STEPS
            evidence = _low_progress_evidence(rows[cursor - 1:stop])
            if evidence is None:
                cursor += 1
                continue
            while stop < end:
                extended = _low_progress_evidence(rows[cursor - 1:stop + 1])
                if extended is None:
                    break
                stop += 1
                evidence = extended
            evidence["target_object"] = target
            evidence["action_role"] = role
            signals.append({"source": label["source"], "task_step_range": [cursor, stop],
                            "kind": "inefficient_motion", "status": "candidate",
                            "evidence": evidence, "version": VERSION})
            cursor = stop
    return signals


def approach_motion(rows: list[dict], targets: set[str], radius: float) -> dict | None:
    """Measure hand travel and target-distance regressions during an approach."""
    if len(rows) < 2 or not targets:
        return None
    if any(not targets.issubset(row["object_positions_m"]) for row in rows):
        return None
    target = min(targets, key=lambda name: dist(
        rows[-1]["eef_position_m"], rows[-1]["object_positions_m"][name]))
    distances = [dist(row["eef_position_m"], row["object_positions_m"][target])
                 for row in rows]
    path = sum(dist(a["eef_position_m"], b["eef_position_m"])
               for a, b in zip(rows, rows[1:]))
    net = distances[0] - distances[-1]
    backtrack = sum(max(0.0, b - a) for a, b in zip(distances, distances[1:]))
    return {"target_object": target,
            "target_distance_start_m": round(distances[0], 4),
            "target_distance_end_m": round(distances[-1], 4),
            "target_progress_m": round(net, 4),
            "backtrack_m": round(backtrack, 4),
            "path_over_progress": round(path / max(net, radius), 3)}


def behavior_signals(labels: list[dict], radii: dict[str, float],
                     targets: set[str], rows: list[dict] | None = None) -> list[dict]:
    """Propose review intervals for one episode, without changing role labels."""
    if not labels:
        return []
    signals = low_progress_signals(labels, rows, targets) if rows is not None else []
    for label in labels:
        motion = label["efficiency"].get("approach_motion")
        if label["role"] not in {"nominal", "recovery"} or not motion:
            continue
        radius = max((float(radii.get(name, 0.04)) for name in targets),
                     default=0.04)
        if (label["efficiency"]["eef_path_m"] >= max(0.12, 4 * radius)
                and (motion["backtrack_m"] >= max(0.05, 2 * radius)
                     or motion["path_over_progress"] >= 4)):
            signals.append(_signal(label, "inefficient_motion", motion))
    efficiency = labels[0]["efficiency"]
    if (efficiency["reference_count"] >= 5
            and efficiency["speed_ratio"] is not None
            and efficiency["speed_ratio"] >= 1.5):
        signals.append({"source": labels[0]["source"],
                        "task_step_range": [0, labels[-1]["task_step_range"][1]],
                        "kind": "slow_relative", "status": "candidate",
                        "evidence": {"speed_ratio": efficiency["speed_ratio"],
                                     "reference_count": efficiency["reference_count"]},
                        "version": VERSION})
    source = labels[0]["source"]
    cycles = []
    for event in sorted((event for label in labels for event in label["events"]),
                        key=lambda event: event["task_step"]):
        if event["event"] in {"target_grasped", "goal_gained"}:
            signals.extend(_repeated_gripper_cycles(source, cycles))
            cycles = []
        elif event["event"] in {"gripper_cycle_unresolved", "grasp_missed"}:
            cycles.append(event["task_step"])
    signals.extend(_repeated_gripper_cycles(source, cycles))
    return sorted(signals, key=lambda row: (row["source"], row["task_step_range"], row["kind"]))


def review_starts(starts: list[dict], signals: list[dict], source: str) -> list[dict]:
    """List selected eight-action starts that a proposed signal overlaps."""
    source_signals = [signal for signal in signals if signal["source"] == source]
    rows = []
    for start in starts:
        step = start["frame_index"]
        reasons = sorted({signal["kind"] for signal in source_signals
                          if signal["task_step_range"][0] < step + 8
                          and step < signal["task_step_range"][1]})
        if reasons:
            rows.append({"source": source, "frame_index": step,
                         "set": start["set"], "reasons": reasons})
    return rows


def signal_counts(signals: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(signal["kind"] for signal in signals).items()))


def _signal(label: dict, kind: str, evidence: dict) -> dict:
    return {"source": label["source"],
            "task_step_range": label["task_step_range"],
            "kind": kind, "status": "candidate",
            "evidence": evidence, "version": VERSION}


def _repeated_gripper_cycles(source: str, cycles: list[int]) -> list[dict]:
    """Group cycle-end events; their span is not a verified retry or training segment."""
    groups = []
    for step in cycles:
        if not groups or step - groups[-1][-1] > 40:
            groups.append([])
        groups[-1].append(step)
    return [{"source": source, "task_step_range": [group[0], group[-1] + 1],
             "kind": "repeated_gripper_cycle", "status": "candidate",
             "evidence": {"event_steps": group, "cycle_count": len(group)},
             "version": VERSION}
            for group in groups if len(group) >= 2]
