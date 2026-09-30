"""Dump RUA trace/feedback and export verified WLA training episodes.

This is an offline conversion. It never calls a policy or steps an environment.
The source artifacts remain unchanged, including failed episodes and agent traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from libero_harness.audit import audit_episode
from libero_harness.paths import save_json


FPS = 20
FRONT = "observation.images.agentview"
WRIST = "observation.images.wrist"
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y",
               "axis_angle_z", "gripper_qpos_0", "gripper_qpos_1"]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry",
                "delta_rz", "gripper_training"]


def wla_state(proprioception: dict) -> np.ndarray:
    """Match the native LIBERO quat2axisangle convention without altering obs."""
    pos = np.asarray(proprioception["robot0_eef_pos"], dtype=np.float64)
    quat = np.asarray(proprioception["robot0_eef_quat"], dtype=np.float64)
    gripper = np.asarray(proprioception["robot0_gripper_qpos"], dtype=np.float64)
    if pos.shape != (3,) or quat.shape != (4,) or gripper.shape != (2,):
        raise ValueError("Expected native LIBERO position[3], quaternion[4], gripper[2]")
    if not all(np.isfinite(x).all() for x in (pos, quat, gripper)):
        raise ValueError("Non-finite proprioception")
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    axis_angle = (np.zeros(3) if math.isclose(den, 0.0)
                  else quat[:3] * (2.0 * math.acos(w) / den))
    state = np.concatenate((pos, axis_angle, gripper)).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Invalid WLA state")
    return state


def training_action(environment_action: list) -> np.ndarray:
    """Invert native_contract.env_actions: env -1/+1 -> WLA training 1/0."""
    action = np.asarray(environment_action, dtype=np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected finite native environment action[7]")
    if not np.isclose(abs(float(action[-1])), 1.0, atol=1e-6):
        raise ValueError("Native gripper action must be -1 or +1")
    action[-1] = 1.0 if action[-1] < 0 else 0.0
    return action


def episode_steps(directory: Path, result: dict, events: list[dict],
                  require_success: bool = True) -> list[dict]:
    if result.get("status") != "completed":
        raise ValueError("Only completed episodes have aligned WLA training frames")
    if require_success and result.get("official_success") is not True:
        raise ValueError("Only officially successful episodes enter basic WLA export")
    if result.get("uncertain_step") or result.get("cleanup_errors"):
        raise ValueError("Uncertain step or cleanup error")
    starts = {event["attempt"]: event for event in events if event["event"] == "step_started"}
    steps = []
    for event in events:
        if event["event"] != "step_completed" or event["phase"] != "task":
            continue
        before = starts[event["attempt"]]
        if "proprioception_before" not in before:
            raise ValueError("Missing pre-action proprioception; this rollout predates the dump collector")
        if event["action"] != before["action"]:
            raise ValueError("Started/completed action mismatch")
        steps.append(dict(attempt=event["attempt"], task_step=len(steps),
                          state=wla_state(before["proprioception_before"]),
                          action=training_action(event["action"]),
                          environment_action=event["action"], token=event["token"],
                          official_success=event["official_success"]))
    if len(steps) != result["task_steps"] or len(steps) < 9:
        raise ValueError("Episode has fewer than nine aligned task frames, or a step count mismatch")
    if [step["attempt"] for step in steps] != list(range(steps[0]["attempt"],
                                                        steps[0]["attempt"] + len(steps))):
        raise ValueError("Non-contiguous task steps")
    if not result.get("task_instruction"):
        raise ValueError("Missing full task instruction")
    return steps


def features(shape: tuple[int, int, int]) -> dict:
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"Expected HWC RGB video, got {shape}")
    camera = {"dtype": "video", "shape": shape,
              "names": ["height", "width", "channel"]}
    return {
        FRONT: dict(camera), WRIST: dict(camera),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def image_shape(directory: Path, camera: str = "front") -> tuple[int, int, int]:
    reader = imageio.get_reader(directory / f"{camera}-control.mp4")
    try:
        return tuple(np.asarray(reader.get_data(0)).shape)
    finally:
        reader.close()


def append_episode(dataset, directory: Path, result: dict, steps: list[dict], shape) -> None:
    front = imageio.get_reader(directory / "front-control.mp4")
    wrist = imageio.get_reader(directory / "wrist-control.mp4")
    try:
        wanted = {step["attempt"]: step for step in steps}
        seen = 0
        for frame_index, (front_frame, wrist_frame) in enumerate(zip(front, wrist)):
            step = wanted.get(frame_index)
            if step is None:
                continue
            front_frame = np.asarray(front_frame)
            wrist_frame = np.asarray(wrist_frame)
            if front_frame.shape != shape or wrist_frame.shape != shape:
                raise ValueError("Camera shape changed within episode")
            dataset.add_frame({
                FRONT: front_frame, WRIST: wrist_frame,
                "observation.state": step["state"], "action": step["action"],
                "task": result["task_instruction"],
            })
            seen += 1
        if seen != len(steps):
            raise ValueError(f"Only {seen}/{len(steps)} pre-action video frames found")
        dataset.save_episode()
    finally:
        front.close()
        wrist.close()


def discover(sources: list[Path], *, include_incomplete: bool = False) -> list[Path]:
    found = set()
    for source in sources:
        if include_incomplete:
            markers = ("result.json", "environment-steps.jsonl")
            if not source.exists() or any((source / name).is_file() for name in markers):
                found.add(source.resolve())
            else:
                found.update(path.parent.resolve() for name in markers
                             for path in source.rglob(name))
            continue
        if (source / "result.json").is_file() and (source / "environment-steps.jsonl").is_file():
            found.add(source.resolve())
        else:
            found.update(path.parent.resolve() for path in source.rglob("result.json")
                         if (path.parent / "environment-steps.jsonl").is_file())
    return sorted(found)


def read_source(directory: Path) -> tuple[dict, list[dict], str | None]:
    """Read each input independently so valid trace survives a broken result."""
    result, events, errors = {}, [], []
    try:
        value = json.loads((directory / "result.json").read_text())
        if not isinstance(value, dict) or not isinstance(value.get("pilot", {}), dict):
            raise ValueError("result.json must contain an object with an object-valued pilot")
        result = value
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"result.json: {exc}")
    try:
        lines = (directory / "environment-steps.jsonl").read_text().splitlines()
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"environment-steps.jsonl: {exc}")
        lines = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Trace row must be an object")
            events.append(value)
        except ValueError as exc:
            errors.append(f"environment-steps.jsonl line {number}: {exc}")
    return result, events, "; ".join(errors) if errors else None


def write_jsonl(path: Path, records) -> None:
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def trace_records(directory: Path, events: list[dict]):
    """Keep event streams separate by kind; IDs are captured during execution."""
    source = str(directory)
    for event in events:
        record = dict(event)
        if "kind" in record:
            record["environment_kind"] = record.pop("kind")
        yield {"source": source, "kind": "environment", **record}
    native = directory / "native"
    step_directories = (
        sorted({path.parent for pattern in ("steps.json", "steps.jsonl")
                for path in native.rglob(pattern)}) if native.is_dir() else []
    )
    for run in step_directories:
        path = run / "steps.json"
        if not path.is_file():
            path = run / "steps.jsonl"
        records = (json.loads(path.read_text()) if path.suffix == ".json" else
                   [json.loads(line) for line in path.read_text().splitlines() if line])
        for record in records:
            step = int(record["i"])
            yield {
                "source": source, "kind": "agent_decision",
                "decision_index": step, "record": record,
                "front_image": str(run / "images/agentview" / f"{step:04d}.png"),
                "wrist_image": str(run / "images/wrist" / f"{step:04d}.png"),
            }
        for name in ("metadata", "subgoals", "planner_diagnostics", "summary"):
            artifact = run / f"{name}.json"
            if artifact.is_file():
                yield {"source": source, "kind": "agent_artifact",
                       "artifact_type": name, "record": json.loads(artifact.read_text())}
    for path in sorted((directory / "requests").glob("request-*/request.json")):
        request = json.loads(path.read_text())
        parent = path.parent
        yield {
            "source": source, "kind": "model_request",
            "call_id": request["call_id"], "request": request,
            "prompt": (parent / "prompt.txt").read_text()
            if (parent / "prompt.txt").is_file() else None,
            "response": (parent / "response.txt").read_text()
            if (parent / "response.txt").is_file() else None,
            "front_image": str(parent / "front.png"),
            "wrist_image": str(parent / "wrist.png"),
        }


def load_review(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    decisions = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            approved = row["approved_for_wla"]
            if not isinstance(approved, bool):
                raise ValueError("approved_for_wla must be a JSON boolean")
            source = str(Path(row["source"]).resolve())
            if source in decisions:
                raise ValueError(f"Duplicate review source: {source}")
            ranges = row.get("task_step_ranges")
            if ranges is not None and not approved:
                raise ValueError("task_step_ranges require approved_for_wla=true")
            reason = row.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("Review reason must be a string")
            decisions[source] = {"approved_for_wla": approved,
                                 "task_step_ranges": ranges, "reason": reason}
    return decisions


def selected_ranges(review: dict | None, task_steps: int) -> list[tuple[int, int]]:
    """Half-open task-step ranges; each becomes one contiguous training episode."""
    ranges = review["task_step_ranges"] if review is not None else None
    if ranges is None:
        return [(0, task_steps)]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("task_step_ranges must be a non-empty list")
    selected = []
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise ValueError("Each task_step_range must be [start, end] with integer indices")
        start, end = pair
        if start < 0 or end > task_steps or end - start < 9:
            raise ValueError("Each task_step_range must contain at least nine in-bounds frames")
        if selected and start < selected[-1][1]:
            raise ValueError("task_step_ranges must be sorted and non-overlapping")
        selected.append((start, end))
    return selected


def load_quality_labels(path: Path) -> dict[str, list[dict]]:
    by_source = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_source.setdefault(str(Path(row["source"]).resolve()), []).append(row)
    return by_source


def quality_starts(source: Path, labels: list[dict], task_steps: int,
                   ranges: list[tuple[int, int]]) -> list[dict]:
    """Return only starts whose eight supervised actions share a verified role."""
    trace_hash = hashlib.sha256((source / "environment-steps.jsonl").read_bytes()).hexdigest()
    roles = [None] * task_steps
    versions = set()
    oracle_hashes = set()
    for label in labels:
        if label["source_trace_sha256"] != trace_hash:
            raise ValueError("Quality labels refer to a different source trace")
        versions.add(label["labeler_version"])
        oracle_hashes.add(label["oracle_feedback_sha256"])
        start, end = label["task_step_range"]
        if not (0 <= start < end <= task_steps):
            raise ValueError("Quality label range is outside the episode")
        for step in range(start, end):
            if roles[step] is not None:
                raise ValueError("Quality label ranges overlap")
            roles[step] = (label["role"] if label["verification"] == "verified"
                           and label["coverage"] == "complete" else "uncertain")
    if any(role is None for role in roles) or len(versions) != 1 or len(oracle_hashes) != 1:
        raise ValueError("Quality labels must cover the episode with one evidence version")
    starts = []
    for step in range(task_steps - 8):
        role = roles[step]
        if (role not in {"nominal", "recovery"}
                or any(other != role for other in roles[step:step + 8])
                or not any(start <= step and step + 8 <= end for start, end in ranges)):
            continue
        starts.append({"frame_index": step, "set": role,
                       "labeler_version": next(iter(versions)),
                       "selection_version": "positive-001"})
    return starts


def export(sources: list[Path], output_root: Path, dataset_name: str,
           review: Path | None = None, quality_labels: Path | None = None,
           source_failures: dict[str, dict] | None = None) -> dict:
    source_failures = source_failures or {}
    candidates = sorted(set(discover(sources)) | {Path(source) for source in source_failures})
    if not candidates:
        raise ValueError("No episode with result.json and environment-steps.jsonl found")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    approvals = load_review(review)
    labels_by_source = load_quality_labels(quality_labels) if quality_labels else None
    accepted, feedback, episode_rows = [], [], []
    for directory in candidates:
        result, events, read_error = read_source(directory)
        review_row = approvals.get(str(directory)) if approvals is not None else None
        review_approved = (None if approvals is None else
                           bool(review_row and review_row["approved_for_wla"]))
        audit_pass = False
        rejection_code = None
        rejection_reason = None
        ranges = []
        selected_starts = []
        quality_role_steps = {}
        stage = "trace_invalid"
        records = []
        try:
            records.extend(trace_records(directory, events))
            failure = source_failures.get(str(directory))
            if failure:
                stage = failure["rejection_code"]
                raise ValueError(failure["rejection_reason"])
            stage = "source_invalid"
            if read_error:
                raise ValueError(read_error)
            stage = "legacy_source"
            if "pilot" not in result:
                raise ValueError("Legacy rollout schema; no RUA pre-action frame alignment")
            stage = "audit_failed"
            audit_episode(directory)
            audit_pass = True
            stage = "task_not_successful"
            if result.get("status") != "completed" or (
                    labels_by_source is None and result.get("official_success") is not True):
                raise ValueError("Only completed, officially successful episodes enter WLA training")
            stage = "uncertain_execution"
            if result.get("uncertain_step") or result.get("cleanup_errors"):
                raise ValueError("Uncertain step or cleanup error")
            stage = "training_data_invalid"
            steps = episode_steps(directory, result, events,
                                  require_success=labels_by_source is None)
            stage = "review_rejected"
            if review_approved is False:
                raise ValueError((review_row or {}).get("reason") or
                                 "Not approved for WLA in review file")
            stage = "invalid_review_range"
            ranges = selected_ranges(review_row, len(steps))
            if labels_by_source is not None:
                stage = "quality_labels_missing"
                labels = labels_by_source.get(str(directory))
                if not labels:
                    raise ValueError("No quality labels for source")
                for label in labels:
                    quality_role_steps[label["role"]] = (quality_role_steps.get(label["role"], 0)
                                                         + label["task_step_range"][1]
                                                         - label["task_step_range"][0])
                stage = "quality_labels_invalid"
                selected_starts = quality_starts(directory, labels, len(steps), ranges)
                if not selected_starts:
                    stage = "quality_no_training_starts"
                    raise ValueError("No verified eight-action training start")
            stage = "camera_incompatible"
            shape = image_shape(directory)
            if image_shape(directory, "wrist") != shape:
                raise ValueError("Front/wrist image shapes differ")
            if accepted and shape != accepted[0][3]:
                raise ValueError("Camera shape differs from other selected episodes")
            if labels_by_source is None:
                accepted.extend((directory, result, steps[start:end], shape, (start, end), [])
                                for start, end in ranges)
            else:
                accepted.append((directory, result, steps, shape, (0, len(steps)),
                                 selected_starts))
        except (AssertionError, KeyError, FileNotFoundError, ValueError) as exc:
            rejection_code = stage
            rejection_reason = str(exc)
        episode_rows.extend(records)
        if rejection_code is not None:
            selected_starts = []
        training_starts = (len(selected_starts) if labels_by_source is not None else
                           sum(end - start - 8 for start, end in ranges)) if rejection_code is None else 0
        feedback.append({
            "source": str(directory), "status": result.get("status"),
            "official_success": result.get("official_success"),
            "end_reason": result.get("end_reason"),
            "task_instruction": result.get("task_instruction"),
            "task_id": result.get("task_id"),
            "initial_state_id": result.get("initial_state_id", result.get("init_state_id")),
            "suite": result.get("pilot", {}).get("suite", result.get("suite")),
            "mode": result.get("mode"), "backend": result.get("backend"),
            "task_steps": result.get("task_steps"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "model_requests": result.get("model_requests", result.get("model_calls")),
            "wla_calls": result.get("wla_calls"),
            "control_tokens": sorted({event["token"] for event in events
                                      if event.get("event") == "step_completed"
                                      and event.get("phase") == "task" and "token" in event}),
            "audit_pass": audit_pass,
            "review_approved": review_approved,
            "wla_candidate": rejection_code is None,
            "wla_training_starts": training_starts,
            "quality_mode": labels_by_source is not None,
            "quality_nominal_starts": sum(row["set"] == "nominal" for row in selected_starts),
            "quality_recovery_starts": sum(row["set"] == "recovery" for row in selected_starts),
            "quality_role_steps": quality_role_steps if labels_by_source is not None else None,
            "quality_unselected_starts": (max(0, len(steps) - 8) - training_starts
                                           if labels_by_source is not None and
                                           rejection_code in {None, "quality_no_training_starts"}
                                           else None),
            "rejection_code": rejection_code,
            "rejection_reason": rejection_reason,
        })
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "feedback.jsonl", feedback)
    write_jsonl(output_root / "trace.jsonl", episode_rows)
    if approvals is not None:
        write_jsonl(output_root / "review.jsonl",
                    ({"source": source, **{key: value for key, value in decision.items()
                                            if value is not None}}
                     for source, decision in sorted(approvals.items())))
    manifest = {"version": 3 if labels_by_source is not None else 2,
                "dataset": dataset_name if accepted else None, "fps": FPS,
                "state": STATE_NAMES, "action": ACTION_NAMES,
                "wla_chunk_size": 8,
                "selection": ("verified nominal/recovery eight-action starts" if labels_by_source
                              is not None else "audit passed; official success; aligned frames; optional reviewed task-step ranges"),
                "selection_mode": "quality" if labels_by_source is not None else "technical_success_only",
                "gripper_mapping": "environment -1 => training 1; environment +1 => training 0",
                "episodes": [], "feedback_file": "feedback.jsonl", "trace_file": "trace.jsonl",
                "review_file": "review.jsonl" if approvals is not None else None,
                "quality_labels_file": "quality-labels.jsonl" if labels_by_source is not None else None}
    if labels_by_source is not None:
        manifest.update(quality_starts=0, quality_nominal_starts=0, quality_recovery_starts=0)
    if accepted:
        shape = accepted[0][3]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset.create(
            repo_id=dataset_name, root=output_root / dataset_name, fps=FPS,
            robot_type="libero", features=features(shape), use_videos=True, vcodec="h264",
        )
        try:
            all_quality_starts = []
            for directory, result, steps, _, task_step_range, starts in accepted:
                append_episode(dataset, directory, result, steps, shape)
                episode_index = len(manifest["episodes"])
                all_quality_starts.extend({"episode_index": episode_index, **start}
                                          for start in starts)
                manifest["episodes"].append({
                    "episode_index": episode_index, "source": str(directory),
                    "task_step_range": list(task_step_range),
                    "suite": result.get("pilot", {}).get("suite"),
                    "task_id": result.get("task_id"), "initial_state_id": result.get("initial_state_id"),
                    "mode": result.get("mode"), "backend": result.get("backend"),
                    "frames": len(steps), "wla_training_starts": (
                        len(starts) if labels_by_source is not None else len(steps) - 8),
                    "official_success": result["official_success"],
                    "task_instruction": result["task_instruction"],
                    "control_tokens": sorted({step["token"] for step in steps}),
                })
        finally:
            dataset.finalize()
        if labels_by_source is not None:
            meta = output_root / dataset_name / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            write_jsonl(meta / "quality-starts.jsonl", all_quality_starts)
            save_json(meta / "quality-selection.json", {
                "version": 1, "selection_version": "positive-001",
                "wla_chunk_size": 8,
                "training_starts": len(all_quality_starts),
            })
            manifest["quality_starts"] = len(all_quality_starts)
            manifest["quality_nominal_starts"] = sum(
                row["set"] == "nominal" for row in all_quality_starts)
            manifest["quality_recovery_starts"] = sum(
                row["set"] == "recovery" for row in all_quality_starts)
    if labels_by_source is not None:
        write_jsonl(output_root / "quality-labels.jsonl",
                    (row for source in sorted(labels_by_source)
                     for row in labels_by_source[source]))
    save_json(output_root / "dump-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Episode or run directories")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="rua_lerobot")
    parser.add_argument("--review", type=Path,
                        help="JSONL source + approved_for_wla + optional task_step_ranges")
    parser.add_argument("--quality-labels", type=Path,
                        help="Verified replay labels; export full episodes and selected WLA starts")
    args = parser.parse_args()
    report = export(args.sources, args.output_root, args.dataset_name,
                    args.review, args.quality_labels)
    print(json.dumps({"output": str(args.output_root), "episodes": len(report["episodes"]),
                      "feedback": str(args.output_root / "feedback.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
