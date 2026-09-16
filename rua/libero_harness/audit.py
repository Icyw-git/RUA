"""Read-only replay of the accounting records; never calls a policy/environment."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .paths import save_json


def audit_wla_records(directory, result, events):
    """Compare every applied native action with the corresponding saved chunk."""
    from native_contract import env_actions
    started = [e for e in events if e["event"] == "wla_started"]
    returned = [e for e in events if e["event"] == "wla_returned"]
    assert len(started) == len(returned) == result["wla_calls"]
    assert [e["call_id"] for e in started] == list(range(len(started)))
    assert [e["call_id"] for e in returned] == list(range(len(returned)))
    active = None
    applied = []
    for event in events:
        if event["event"] == "wla_started":
            assert active is None, "Concurrent WLA calls"
            active, applied = event, []
        elif event["event"] == "step_completed" and event["phase"] == "task":
            if event["token"] == "WLA_CHUNK":
                assert active is not None
                applied.append(event["action"])
            else:
                assert active is None, "Non-WLA action inside delegated native chunk"
                assert result.get("mode") == "rua_wla_hybrid", "Atomic action in WLA-only run"
        elif event["event"] == "wla_returned":
            assert active and event["call_id"] == active["call_id"]
            assert event["executed_env_steps"] == len(applied)
            assert event["observation_sequence"] - event["input_sequence"] == len(applied)
            path = directory / "wla" / f"actions-{event['call_id']:03d}.npy"
            if path.exists():
                expected = env_actions(np.load(path))
                assert expected.shape == (8, 7)
                if applied:
                    np.testing.assert_array_equal(np.asarray(applied), expected[:len(applied)])
                assert event["attempted_env_steps"] + event["residual_actions_discarded"] == 8
            else:
                assert not applied, "Applied actions without saved native output"
            active = None
    assert active is None, "Missing WLA return"
    wla_steps = sum(e["executed_env_steps"] for e in returned)
    assert wla_steps == (result["wla_executed_steps"] if result.get("mode") == "rua_wla_hybrid"
                         else result["task_steps"])
    assert not result.get("worker_alive", True), "Prediction worker did not shut down"


def audit_episode(directory):
    result = json.loads((directory / "result.json").read_text())
    events = [json.loads(x) for x in (directory / "environment-steps.jsonl").read_text().splitlines()]
    starts = {e["attempt"]: e for e in events if e["event"] == "step_started"}
    completed = [e for e in events if e["event"] == "step_completed"]
    assert len(starts) == sum(e["event"] == "step_started" for e in events), "Duplicate action attempts"
    for event in completed:
        assert starts[event["attempt"]]["action"] == event["action"]
        assert event["token"] != "DONE", "DONE executed an environment action"
    missing = sorted(set(starts) - {e["attempt"] for e in completed})
    assert bool(missing) == bool(result.get("uncertain_step"))
    tasks = [e for e in completed if e["phase"] == "task"]
    initialization = [e for e in completed if e["phase"] == "initialization"]
    assert len(tasks) == result["task_steps"] <= result["pilot"]["max_env_steps"]
    assert len(initialization) == result["initialization_steps"]
    assert len(initialization) == result["pilot"]["settle_steps"]
    assert result["official_success"] == any(e["official_success"] for e in completed)
    assert not result["torch_cuda_initialized"]
    if result.get("mode") in ("wla_only_unified_executor", "rua_wla_hybrid"):
        audit_wla_records(directory, result, events)
    else:
        assert result["wla_calls"] == 0
    assert result["qwen_calls"] == (result["model_requests"] if result.get("backend") == "qwen" else 0)
    records = result["request_records"]
    assert result["model_requests"] == len(records) <= result["pilot"]["max_agent_requests"]
    for index, record in enumerate(records, 1):
        assert record["call_id"] == index
        assert [x["camera"] for x in record["image_manifest"]] == ["front", "wrist"]
        assert (directory / "requests" / f"request-{index:03d}" / "request.json").exists()
    video_counts = {}
    for name in ("front", "wrist"):
        reader = imageio.get_reader(directory / f"{name}-control.mp4")
        try:
            video_counts[name] = sum(1 for _ in reader)
        finally:
            reader.close()
        assert video_counts[name] == len(completed) + 1
    return dict(status="passed", task_steps=len(tasks), initialization_steps=len(initialization),
                official_success=result["official_success"], model_requests=len(records),
                wla_calls=result["wla_calls"],
                video_frames=video_counts, uncertain_attempts=missing,
                note="Accounting integrity, not a task success or policy quality judgment.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = dict(status="passed", episodes=[])
    try:
        directories = ([args.run_root] if (args.run_root / "result.json").exists()
                       else sorted(args.run_root.glob("task-*-init-*")))
        for directory in directories:
            result["episodes"].append(dict(directory=str(directory), **audit_episode(directory)))
        assert result["episodes"], "No episode records"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    save_json(args.run_root / "audit.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
