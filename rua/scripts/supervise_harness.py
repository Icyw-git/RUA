"""Bounded external supervision; does not modify the native agent/control loop."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RUA_ROOT", CODE.parent.parent)).resolve()
BASE = ROOT / "artifacts/rua-stage2"
GPU0_UUID = os.environ.get("RUA_GPU_UUID")


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot(cgroup_root="/sys/fs/cgroup"):
    sample = {"wall_time": time.time(), "monotonic": time.monotonic()}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        sample["gpus"] = []
        for row in result.stdout.strip().splitlines():
            index, uuid, free, used, utilization = [x.strip() for x in row.split(",")]
            sample["gpus"].append(dict(index=int(index), uuid=uuid, free_mib=int(free),
                                       used_mib=int(used), utilization=int(utilization)))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        sample["gpu_error"] = type(exc).__name__
    cgroup = Path(cgroup_root) / "memory"
    for name in ("usage_in_bytes", "limit_in_bytes", "max_usage_in_bytes", "failcnt"):
        path = cgroup / ("memory." + name)
        if path.exists():
            sample["cgroup_" + name] = int(path.read_text().strip())
    oom = cgroup / "memory.oom_control"
    if oom.exists():
        sample["cgroup_oom"] = {
            key: int(value) for key, value in (line.split() for line in oom.read_text().splitlines())
        }
    # Preserve memory/oom supervision on cgroup-v2 installations as well.
    unified = Path(cgroup_root)
    if not cgroup.exists() and (unified / "memory.max").exists():
        try:
            from libero_harness.resources import memory_headroom
            mem = memory_headroom(cgroup=cgroup_root)
            sample.update(cgroup_limit_in_bytes=mem["limit_bytes"],
                          cgroup_usage_in_bytes=mem["used_bytes"])
            events = dict(line.split() for line in (unified / "memory.events").read_text().splitlines())
            sample["cgroup_oom"] = {"under_oom": 0, "oom_kill": int(events.get("oom_kill", 0))}
        except (OSError, ValueError, RuntimeError) as exc:
            sample["memory_error"] = type(exc).__name__
    if not all(k in sample for k in (
            "cgroup_limit_in_bytes", "cgroup_usage_in_bytes", "cgroup_oom")):
        sample["memory_error"] = "CannotEstablishCgroup"
    return sample


def stop_reason(sample, initial, min_free_mib=8192):
    if sample.get("memory_error"):
        return "memory_telemetry_unavailable"
    if sample.get("gpu_error"):
        return "gpu_telemetry_unavailable"
    selected = os.environ.get("RUA_GPU", "0")
    def chosen(x):
        return x["uuid"] == selected if selected.startswith("GPU-") else str(x["index"]) == selected
    gpu = next((x for x in sample.get("gpus", []) if chosen(x)), None)
    original = next((x for x in initial.get("gpus", []) if chosen(x)), None)
    expected = GPU0_UUID or (original or {}).get("uuid")
    if not gpu or not original or gpu["uuid"] != expected:
        return "gpu_identity_mismatch"
    if gpu["free_mib"] < min_free_mib:
        return "gpu_free_memory_below_reserve"
    oom = sample.get("cgroup_oom", {})
    if oom.get("under_oom", 0) or oom.get("oom_kill", 0) > initial.get("cgroup_oom", {}).get("oom_kill", 0):
        return "cgroup_oom"
    limit, used = sample.get("cgroup_limit_in_bytes"), sample.get("cgroup_usage_in_bytes")
    if limit and used is not None and limit - used < 4 * 1024**3:
        return "cgroup_memory_below_reserve"
    return None


def kernel_xids():
    result = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=5)
    return dict(returncode=result.returncode,
                lines=[line for line in result.stdout.splitlines() if "NVRM: Xid" in line])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("debug", "pilot", "wla-debug", "hybrid-debug"))
    parser.add_argument("--backend", choices=("qwen", "claude"), default="qwen")
    parser.add_argument("--handoff-receipt")
    parser.add_argument("--rua-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "hybrid-debug" and (args.backend != "claude"
                                      or not (args.handoff_receipt or args.rua_only)):
        parser.error("hybrid-debug requires --backend claude and a handoff receipt (or --rua-only)")
    base = ROOT / "artifacts/rua-stage3" if args.mode in ("wla-debug", "hybrid-debug") else BASE
    base.mkdir(parents=True, exist_ok=True)
    with (base / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another supervised Show-Harness run is active.")
        return supervise(args)


def supervise(args):
    native = args.mode == "wla-debug"
    hybrid = args.mode == "hybrid-debug"
    base = ROOT / "artifacts/rua-stage3" if native or hybrid else BASE
    backend = "native_wla" if native else args.backend
    out = Path(tempfile.mkdtemp(prefix=f"supervised-{args.mode}-{backend}-", dir=base))
    report = dict(status="starting", mode=args.mode, backend=backend,
                  render_backend=os.environ.get("RUA_RENDER_BACKEND", "egl"),
                  supervisor_pid=os.getpid(), started_at=time.time(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_free_reserve_mib=8192, cgroup_free_reserve_bytes=4 * 1024**3,
                  xid_policy="record_only_no_unattributed_automatic_abort")
    save_json(out / "report.json", report)
    print(json.dumps({"supervisor_directory": str(out)}), flush=True)
    initial = snapshot()
    save_json(out / "resources-before.json", initial)
    before = kernel_xids()
    save_json(out / "kernel-before.json", before)
    reason = stop_reason(initial, initial)
    if reason:
        report.update(status="refused", stop_reason=reason)
        save_json(out / "report.json", report)
        return 2
    interrupted = []

    def interrupt(signum, frame):
        interrupted.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    command = ["bash", str(CODE / "scripts/run_show_harness.sh"), args.mode]
    if not native:
        command += ["--backend", args.backend]
    if hybrid:
        if args.handoff_receipt:
            command += ["--handoff-receipt", args.handoff_receipt]
        if args.rua_only:
            command += ["--rua-only"]
    started = time.monotonic()
    deadline = started + (1300 if native or hybrid else 1000 if args.mode == "debug" else 14030)
    child = None
    try:
        with (out / "child.log").open("w") as output, (out / "resources.jsonl").open("w", buffering=1) as trace:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            report.update(status="running", child_pid=child.pid)
            save_json(out / "report.json", report)
            term_at = None
            last_kernel = 0.0
            while child.poll() is None:
                sample = snapshot()
                trace.write(json.dumps(sample) + "\n")
                now = time.monotonic()
                if now - last_kernel >= 10:
                    current = kernel_xids()
                    new = [line for line in current["lines"] if line not in before["lines"]]
                    save_json(out / "kernel-latest.json", dict(**current, new_lines=new))
                    last_kernel = now
                reason = (reason or ("operator_signal" if interrupted else None)
                          or ("supervisor_deadline" if now >= deadline else None)
                          or stop_reason(sample, initial))
                if reason and term_at is None:
                    report["stop_reason"] = reason
                    save_json(out / "report.json", report)
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    term_at = now
                elif term_at is not None and now - term_at > 30 and child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    report["forced_kill"] = True
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        report.update(status="completed" if code == 0 and not reason else "failed",
                      returncode=code, exit_signal=-code if code < 0 else None)
    except BaseException as exc:
        report.update(status="supervisor_error", error_type=type(exc).__name__)
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            report["returncode"] = child.returncode
        raise
    finally:
        report.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started)
        save_json(out / "resources-after.json", snapshot())
        after = kernel_xids()
        save_json(out / "kernel-after.json",
                  dict(**after, new_lines=[line for line in after["lines"] if line not in before["lines"]]))
        log = out / "child.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith('{"run_root":'):
                    report["run_root"] = json.loads(line)["run_root"]
                    break
        if report["status"] == "completed" and report.get("run_root"):
            try:
                with (out / "audit.log").open("w") as audit_output:
                    audit = subprocess.run(
                        ["bash", str(CODE / "scripts/run_show_harness.sh"),
                         "audit", report["run_root"]],
                        stdout=audit_output, stderr=subprocess.STDOUT, timeout=180)
                report["audit_returncode"] = audit.returncode
                if audit.returncode:
                    report["status"] = "audit_failed"
            except subprocess.TimeoutExpired:
                report.update(status="audit_failed", audit_error="timeout")
            report["audit_ended_at"] = time.time()
        save_json(out / "report.json", report)
        print(json.dumps(report), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
