"""Read saved native rollouts and verify accounting, conversion and video integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from bootstrap import ROOT
from native_contract import env_actions
from native_rollout import pilot_cases


def audit(directory, allow_partial=False):
    report = json.loads((directory / "report.json").read_text())
    cfg = report["config"]
    expected = pilot_cases(cfg, report["scope"] == "debug")
    assert report["cases"] == [list(case) for case in expected]
    if not allow_partial:
        assert report["status"] == "completed"
    assert [(r["task_id"], r["init_state_id"]) for r in report["results"]] == expected[:len(report["results"])]
    if not allow_partial:
        assert len(report["results"]) == len(expected)
    checked = []
    for result in report["results"]:
        episode = directory / f"task-{result['task_id']}_init-{result['init_state_id']}"
        assert json.loads((episode / "result.json").read_text()) == result
        assert result["status"] == "completed" and not result["artifact_errors"]
        events = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
        starts = [e for e in events if e["event"] == "step_started"]
        completed = [e for e in events if e["event"] == "step_completed"]
        settle = [e for e in completed if e["phase"] == "settle"]
        control = [e for e in completed if e["phase"] == "task"]
        requests = [e for e in events if e["event"] == "inference_started"]
        responses = [e for e in events if e["event"] == "inference_completed"]
        assert len(settle) == cfg["settle_steps"] == result["initialization_steps"]
        assert len(control) == result["task_control_steps"] <= cfg["max_env_steps"]
        assert len(starts) == len(completed) == result["step_attempts"]
        assert len(requests) == len(responses) == result["wla_calls"]
        assert result["official_success"] == completed[-1]["official_success"]
        assert result["returned_done"] == completed[-1]["done"]
        assert not any(e["official_success"] or e["done"] for e in completed[:-1])
        previous_current_image = None
        for request, response in zip(requests, responses):
            call = request["call_id"]
            step = call * 8
            assert request["task_step"] == response["task_step"] == step
            assert request["history_pre_step_index"] == (None if step == 0 else step - 8)
            saved = np.load(episode / f"input-{call:03d}.npz")
            assert saved["images"].shape == (3, 3, 256, 256)
            assert saved["state"].shape == (8,)
            np.testing.assert_array_equal(saved["state"], request["state"])
            if call == 0:
                np.testing.assert_array_equal(saved["images"][0], saved["images"][1])
            else:
                # At a native 8-step boundary the oldest pre-step frame is exactly
                # the current frame used by the previous inference call.
                np.testing.assert_array_equal(saved["images"][0], previous_current_image)
            previous_current_image = saved["images"][1].copy()
            raw = np.load(episode / f"actions-{call:03d}.npy")
            np.testing.assert_array_equal(raw, response["raw_actions"])
            actual = np.array([e["action"] for e in control[step:step + 8]])
            np.testing.assert_array_equal(env_actions(raw)[:len(actual)], actual)
        for before, after in zip(starts, completed):
            assert before["action"] == after["action"] and before["phase"] == after["phase"]
        frame_counts = {}
        for name in ("front", "wrist"):
            reader = imageio.get_reader(episode / f"{name}.mp4")
            count = reader.count_frames()
            assert count == len(completed) + 1 == result["video_frames"]
            Image.fromarray(reader.get_data(count - 1)).save(episode / f"{name}-final.png")
            reader.close()
            frame_counts[name] = count
        checked.append(dict(task_id=result["task_id"], init_state_id=result["init_state_id"],
                            official_success=result["official_success"], control_steps=len(control),
                            predictions=len(responses), video_frames=frame_counts))
    return dict(status="passed", scope="completed_episode_artifacts_only" if allow_partial else
                "artifact_consistency_not_independent_policy_equivalence",
                full_batch_complete=len(report["results"]) == len(expected) and report["status"] == "completed",
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                report=str(directory / "report.json"), checked_episodes=checked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "artifacts/wla-stage1").resolve()):
        raise ValueError("Only audit this task's artifact directories.")
    result = audit(directory, args.allow_partial)
    output = directory / ("audit-partial.json" if args.allow_partial else "audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "audit": str(output)}))


if __name__ == "__main__":
    main()
