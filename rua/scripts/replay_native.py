"""Bounded environment-only replay of a recorded interrupted episode; not evaluation."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from bootstrap import CODE, ROOT, configure

configure()
import numpy as np
import torch
from libero.libero import benchmark
from libero_utils import get_libero_env
from native_contract import load_pilot


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--isolate-env", action="store_true")
    args = parser.parse_args()
    cfg = load_pilot(CODE / "configs/pilot.json")
    episode = args.episode.resolve()
    if not episode.is_relative_to((ROOT / "artifacts/wla-stage1").resolve()):
        raise ValueError("Replay must use this task's saved artifacts.")
    metadata = json.loads((episode / "result.json").read_text())
    events = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
    starts = [event for event in events if event["event"] == "step_started"]
    completed = [event for event in events if event["event"] == "step_completed"]
    out = Path(tempfile.mkdtemp(prefix="environment-replay-", dir=ROOT / "artifacts/wla-stage1"))
    report = dict(status="running", scope="environment_only_diagnostic_not_policy_evaluation",
                  source_episode=str(episode), pid=os.getpid(), replayed_steps=0,
                  environment_process="isolated_spawn" if args.isolate_env else "same_process",
                  planned_steps=len(starts), wla_loaded=False)
    env = None
    started = time.monotonic()
    print(json.dumps({"run_directory": str(out), "pid": os.getpid()}), flush=True)

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "report.json")

    def interrupted(signum, frame):
        raise InterruptedError(f"signal={signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        report["namespace_pid"] = [line for line in Path("/proc/self/status").read_text().splitlines()
                                    if line.startswith("NSpid:")]
        report["kernel_log_before"] = subprocess.run(
            ["dmesg", "-T"], text=True, capture_output=True, timeout=10).stdout.splitlines()[-100:]
        save()
        suite = benchmark.get_benchmark_dict()[cfg["suite"]](task_order_index=cfg["task_order_index"])
        task = suite.get_task(metadata["task_id"])
        if args.isolate_env:
            from isolated_env import IsolatedLiberoEnv
            env = IsolatedLiberoEnv(cfg, metadata["task_id"])
            instruction = env.instruction
            report["worker_pid"] = env.process.pid
        else:
            env, instruction = get_libero_env(task, "wla", resolution=cfg["render_resolution"])
        env.reset()
        obs = env.set_init_state(np.load(episode / "initial_state.npy"))
        diffs = []
        with (out / "steps.jsonl").open("w") as trace:
            for index, event in enumerate(starts):
                step_started = time.monotonic()
                trace.write(json.dumps(dict(event="step_started", index=index,
                                            action=event["action"])) + "\n")
                trace.flush()
                obs, reward, done, _ = env.step(event["action"])
                success = bool(env.check_success())
                record = dict(event="step_completed", index=index, done=bool(done),
                              official_success=success, seconds=time.monotonic() - step_started)
                if index < len(completed):
                    diff = float(np.abs(obs["robot0_eef_pos"] -
                                        np.array(completed[index]["proprioception"]["robot0_eef_pos"])).max())
                    record["eef_max_abs_diff"] = diff
                    diffs.append(diff)
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                report.update(replayed_steps=index + 1, last_step=record)
                save()
                if done or success:
                    break
        report.update(status="completed", max_eef_abs_diff=max(diffs, default=None),
                      torch_cuda_initialized=torch.cuda.is_initialized())
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report.update(status="failed", close_error=str(exc))
        report["kernel_log_after"] = subprocess.run(
            ["dmesg", "-T"], text=True, capture_output=True, timeout=10).stdout.splitlines()[-100:]
        save()
        print(json.dumps({"report": str(out / "report.json"), "status": report["status"],
                          "steps": report["replayed_steps"]}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
