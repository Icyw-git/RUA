"""Dump RUA trace/feedback and export verified WLA training episodes.

This is an offline conversion. It never calls a policy or steps an environment.
The source artifacts remain unchanged, including failed episodes and agent traces.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .audit import audit_episode
from .paths import save_json


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


def episode_steps(directory: Path, result: dict, events: list[dict]) -> list[dict]:
    if result.get("status") != "completed" or result.get("official_success") is not True:
        raise ValueError("Only completed, officially successful episodes enter WLA training")
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


def discover(sources: list[Path]) -> list[Path]:
    found = set()
    for source in sources:
        if (source / "result.json").is_file() and (source / "environment-steps.jsonl").is_file():
            found.add(source.resolve())
        else:
            found.update(path.parent.resolve() for path in source.rglob("result.json")
                         if (path.parent / "environment-steps.jsonl").is_file())
    return sorted(found)


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


def load_review(path: Path | None) -> dict[str, bool] | None:
    if path is None:
        return None
    decisions = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            approved = row["approved_for_wla"]
            if not isinstance(approved, bool):
                raise ValueError("approved_for_wla must be a JSON boolean")
            decisions[str(Path(row["source"]).resolve())] = approved
    return decisions


def export(sources: list[Path], output_root: Path, dataset_name: str,
           review: Path | None = None) -> dict:
    candidates = discover(sources)
    if not candidates:
        raise ValueError("No episode with result.json and environment-steps.jsonl found")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    approvals = load_review(review)
    accepted, feedback, episode_rows = [], [], []
    for directory in candidates:
        result = json.loads((directory / "result.json").read_text())
        events = [json.loads(line) for line in
                  (directory / "environment-steps.jsonl").read_text().splitlines()]
        episode_rows.append((directory, events))
        audit_pass = False
        try:
            if "pilot" not in result:
                raise ValueError("Legacy rollout schema; no RUA pre-action frame alignment")
            audit_episode(directory)
            audit_pass = True
            steps = episode_steps(directory, result, events)
            if approvals is not None and not approvals.get(str(directory), False):
                raise ValueError("Not approved for WLA in review file")
            shape = image_shape(directory)
            if image_shape(directory, "wrist") != shape:
                raise ValueError("Front/wrist image shapes differ")
            if accepted and shape != accepted[0][3]:
                raise ValueError("Camera shape differs from other selected episodes")
            accepted.append((directory, result, steps, shape))
            rejection_reason = None
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            rejection_reason = str(exc)
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
            "model_requests": result.get("model_requests", result.get("model_calls")),
            "audit_pass": audit_pass,
            "review_approved": approvals.get(str(directory), False) if approvals is not None else None,
            "wla_candidate": rejection_reason is None,
            "rejection_reason": rejection_reason,
        })
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "feedback.jsonl", feedback)
    write_jsonl(output_root / "trace.jsonl",
                (record for directory, events in episode_rows
                 for record in trace_records(directory, events)))
    if approvals is not None:
        write_jsonl(output_root / "review.jsonl",
                    ({"source": source, "approved_for_wla": approved}
                     for source, approved in sorted(approvals.items())))
    manifest = {"version": 1, "dataset": dataset_name if accepted else None, "fps": FPS,
                "state": STATE_NAMES, "action": ACTION_NAMES,
                "wla_chunk_size": 8,
                "selection": "audit passed; official success; complete aligned frames; optional review approval",
                "gripper_mapping": "environment -1 => training 1; environment +1 => training 0",
                "episodes": [], "feedback_file": "feedback.jsonl", "trace_file": "trace.jsonl",
                "review_file": "review.jsonl" if approvals is not None else None}
    if accepted:
        shape = accepted[0][3]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset.create(
            repo_id=dataset_name, root=output_root / dataset_name, fps=FPS,
            robot_type="libero", features=features(shape), use_videos=True, vcodec="h264",
        )
        try:
            for directory, result, steps, _ in accepted:
                append_episode(dataset, directory, result, steps, shape)
                manifest["episodes"].append({
                    "episode_index": len(manifest["episodes"]), "source": str(directory),
                    "suite": result.get("pilot", {}).get("suite"),
                    "task_id": result.get("task_id"), "initial_state_id": result.get("initial_state_id"),
                    "mode": result.get("mode"), "backend": result.get("backend"),
                    "frames": len(steps), "wla_training_starts": len(steps) - 8,
                    "official_success": True,
                    "task_instruction": result["task_instruction"],
                    "control_tokens": sorted({step["token"] for step in steps}),
                })
        finally:
            dataset.finalize()
    save_json(output_root / "dump-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Episode or run directories")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="rua_lerobot")
    parser.add_argument("--review", type=Path, help="JSONL source + approved_for_wla decisions")
    args = parser.parse_args()
    report = export(args.sources, args.output_root, args.dataset_name, args.review)
    print(json.dumps({"output": str(args.output_root), "episodes": len(report["episodes"]),
                      "feedback": str(args.output_root / "feedback.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
