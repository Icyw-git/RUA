"""Preserve a stale interrupted run and record an external, evidence-based status."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from bootstrap import ROOT


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return dict(argv=args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "artifacts/wla-stage1").resolve()):
        raise ValueError("Only reconcile this task's artifacts.")
    report = json.loads((directory / "report.json").read_text())
    pid_path = Path(f"/proc/{report['pid']}")
    if pid_path.exists():
        raise RuntimeError("Recorded PID exists: do not reconcile a possibly active run.")
    current = report["current_case"]
    episode = directory / f"task-{current[0]}_init-{current[1]}"
    events = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["event"] == "step_completed"]
    record = dict(
        status="incomplete_process_exited_without_final_report",
        observed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        original_report_preserved=True, original_report_status=report["status"],
        original_pid=report["pid"], recorded_pid_absent=True,
        exit_code=None, exit_signal=None, root_cause="not_established",
        finished_episodes=len(report["results"]),
        completed_official_successes=sum(r["official_success"] for r in report["results"]),
        interrupted_case=current, interrupted_official_success=None,
        last_completed_step=completed[-1], last_trace_event=events[-1],
        unstarted_cases=report["cases"][len(report["results"]) + 1:],
        planned_batch_success_rate=None, stage1_gate_passed=False,
        observations=[
            "No final result or Python exception was written for task 0, init 5.",
            "Last trace event is step_started after 10 settling and 72 completed task steps.",
            "The side effect of the next environment step is unknown; never infer a task outcome.",
            "GPU 0 kernel logs contain repeated Xid 31 graphics MMU errors during the run.",
            "Kernel PID attribution to the container evaluation PID is not established.",
            "Container memory.oom_control reports oom_kill 0; this does not establish another cause.",
            "No automatic policy retry, sample substitution, or driver reset was performed.",
        ],
        diagnostics={
            "gpu": command(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.free",
                            "--format=csv,noheader"]),
            "kernel_log": command(["dmesg", "-T"]),
            "memory_oom_control": command(["cat", "/sys/fs/cgroup/memory/memory.oom_control"]),
            "memory_failcnt": command(["cat", "/sys/fs/cgroup/memory/memory.failcnt"]),
        },
    )
    # Keep only the incident window, not unrelated historic kernel content.
    kernel = record["diagnostics"]["kernel_log"]
    kernel["stdout"] = "\n".join(line for line in kernel["stdout"].splitlines()
                                 if "Sep 15 09:" in line) + "\n"
    output = directory / "reconciliation.json"
    if output.exists():
        raise FileExistsError("Preserve the first external reconciliation receipt.")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
