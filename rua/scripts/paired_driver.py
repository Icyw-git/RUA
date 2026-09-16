"""Serial, restart-aware supervisor and paired scoreboard. No agent policy here."""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from libero_harness.paired import ARMS, digest, load_manifest, seal_preflight
from libero_harness.paths import ROOT, save_json
from libero_harness.resources import admission
from supervise_harness import snapshot, stop_reason, kernel_xids

INTERRUPTED = []
SCRIPT = str(CODE / "scripts/run_show_harness.sh")


def on_signal(signum, frame):
    INTERRUPTED.append(signum)


def bounded(command, out, timeout):
    """Only signal the process group we created; preserve reports on every exit."""
    out.mkdir(parents=True, exist_ok=False)
    initial = snapshot()
    before = kernel_xids()
    save_json(out / "resources-before.json", initial)
    save_json(out / "kernel-before.json", before)
    reason = stop_reason(initial, initial)
    record = dict(status="refused" if reason else "starting", stop_reason=reason,
                  command=command, started_at=time.time())
    save_json(out / "report.json", record)
    if reason:
        return record
    started = time.monotonic()
    child = None
    env = dict(os.environ, RUA_RENDER_BACKEND="osmesa")
    try:
        with (out / "child.log").open("x") as output, (out / "resources.jsonl").open("x") as trace:
            child = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="running", child_pid=child.pid)
            save_json(out / "report.json", record)
            term_at = None
            while child.poll() is None:
                sample = snapshot()
                trace.write(json.dumps(sample) + "\n")
                trace.flush()
                now = time.monotonic()
                reason = (reason or ("operator_signal" if INTERRUPTED else None)
                          or ("supervisor_deadline" if now - started >= timeout else None)
                          or stop_reason(sample, initial))
                if reason and term_at is None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    term_at = now
                elif term_at is not None and now - term_at > 30 and child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    record["forced_kill"] = True
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            record.update(status="completed", returncode=child.wait(), stop_reason=reason)
    except BaseException as exc:
        record.update(status="supervisor_error", error_type=type(exc).__name__)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        after = kernel_xids()
        save_json(out / "kernel-after.json",
                  dict(**after, new_lines=[x for x in after["lines"] if x not in before["lines"]]))
        save_json(out / "resources-after.json", snapshot())
        record.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started)
        save_json(out / "report.json", record)
    return record


def classify(result, audit, supervisor):
    if supervisor.get("stop_reason"):
        return "infrastructure_error"
    if audit.get("status") != "passed":
        return "integrity_error"
    if result.get("cleanup_errors") or result.get("uncertain_step") or result.get("worker_alive"):
        return "integrity_error"
    if result.get("official_success"):
        return "success"
    errors = result.get("environment_errors") or []
    for error in errors:
        if error.get("type") == "BudgetExhausted":
            continue
        # A complete but invalid policy answer is a policy failure, not an
        # infrastructure exclusion that would inflate Opus's scored success rate.
        message = error.get("message", "")
        if error.get("type") == "ModelResponseError" and message.startswith(
                ("Invalid model JSON/schema", "Invalid JSON fence", "Model token is not exactly")):
            continue
        return "infrastructure_error"
    if result.get("status") == "error" and not errors:
        return "infrastructure_error"
    if result.get("status") in ("preflight", "initializing", "running", "interrupted"):
        return "infrastructure_error"
    return "task_failure"


def episode_row(case_id, arm, result, audit, supervisor, directory):
    classification = classify(result, audit, supervisor)
    handoffs = 0
    atomic_steps = 0
    trace = directory / "environment-steps.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "handoff_completed":
                handoffs += 1
            if (event.get("event") == "step_completed" and event.get("phase") == "task"
                    and event.get("token") != "WLA_CHUNK"):
                atomic_steps += 1
    usage = result.get("model_usage") or {}
    return dict(case_id=case_id, arm=arm, classification=classification,
                official_success=bool(result.get("official_success")),
                task_steps=result.get("task_steps"), episode_seconds=result.get("episode_seconds"),
                model_requests=result.get("model_requests", 0), input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0), wla_calls=result.get("wla_calls", 0),
                wla_steps=result.get("wla_executed_steps", result.get("task_steps", 0) if arm == "wla" else 0),
                atomic_and_hold_steps=atomic_steps, handoff_events=handoffs,
                end_reason=result.get("end_reason"), audit_status=audit.get("status"),
                directory=str(directory))


def aggregate(manifest, rows):
    expected = {(c["case_id"], a) for c in manifest["cases"] for a in ARMS}
    keys = [(r["case_id"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or out-of-manifest result")
    by_arm = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        counts = collections.Counter(r["classification"] for r in group)
        success = counts["success"]
        valid = success + counts["task_failure"]
        by_arm[arm] = dict(
            planned=20, finished=len(group), pending=20 - len(group), counts=dict(counts),
            successes=success, success_fraction_of_planned=success / 20,
            policy_scored=valid, success_fraction_policy_scored=success / valid if valid else None,
            mean_steps_all_finished=_mean(group, "task_steps"),
            mean_seconds_all_finished=_mean(group, "episode_seconds"),
            mean_steps_success_only=_mean([r for r in group if r["classification"] == "success"], "task_steps"),
            total_requests=sum(r["model_requests"] for r in group),
            total_input_tokens=sum(r["input_tokens"] for r in group),
            total_output_tokens=sum(r["output_tokens"] for r in group),
            total_wla_calls=sum(r["wla_calls"] for r in group),
            total_handoffs=sum(r["handoff_events"] for r in group),
        )
    indexed = {(r["case_id"], r["arm"]): r for r in rows}
    pairs = {}
    for left, right in (("opus", "wla"), ("hybrid", "wla"), ("hybrid", "opus")):
        counts = collections.Counter()
        for c in manifest["cases"]:
            a, b = indexed.get((c["case_id"], left)), indexed.get((c["case_id"], right))
            if not a or not b:
                counts["pending"] += 1
            elif any(r["classification"] not in ("success", "task_failure") for r in (a, b)):
                counts["infrastructure_or_integrity_excluded"] += 1
            else:
                key = ("both_success" if a["official_success"] and b["official_success"] else
                       "left_only" if a["official_success"] else
                       "right_only" if b["official_success"] else "both_failure")
                counts[key] += 1
        pairs[f"{left}_vs_{right}"] = dict(counts)
    return dict(planned_episodes=60, finished_episodes=len(rows), complete=len(rows) == 60,
                arms=by_arm, paired=pairs, note="One init per task; engineering comparison, not official benchmark.")


def _mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (float, int))]
    return sum(values) / len(values) if values else None


def publish(root, manifest, ledger):
    summary = aggregate(manifest, ledger["rows"])
    summary["batch_status"] = ledger["status"]
    save_json(root / "ledger.json", ledger)
    save_json(root / "summary.json", summary)
    columns = ["case_id", "arm", "classification", "official_success", "task_steps", "episode_seconds",
               "model_requests", "input_tokens", "output_tokens", "wla_calls", "wla_steps",
               "atomic_and_hold_steps", "handoff_events", "end_reason", "audit_status", "directory"]
    tmp = root / "results.csv.tmp"
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ledger["rows"])
    tmp.replace(root / "results.csv")
    lines = [
        "# Paired LIBERO 20-task comparison", "",
        f"Status: {ledger['status']}; finished {len(ledger['rows'])}/60 episodes.", "",
        "Spatial 0–9 + Object 0–9; init1; same frozen task/state per arm.",
        "Pending/error cases are not silently removed. Success / 20 is not a final rate until complete.",
        "Policy-scored rates exclude infrastructure/integrity errors and show their denominator.", "",
        "| Arm | Success / planned | Finished | Policy-scored | Infra/integrity | Mean steps (all finished) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, r in summary["arms"].items():
        errors = sum(r["counts"].get(x, 0) for x in ("infrastructure_error", "integrity_error"))
        lines.append(f"| {arm} | {r['successes']}/20 | {r['finished']} | {r['policy_scored']} | "
                     f"{errors} | {r['mean_steps_all_finished']} |")
    lines += ["", "Full task-level records: results.csv. Paired win/loss counts and token/time totals: summary.json.",
              "One initial state per task is a small engineering comparison, not a formal benchmark replication.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=ledger["status"], finished=len(ledger["rows"]),
                          successes={a: r["successes"] for a, r in summary["arms"].items()})), flush=True)


def drive(root, manifest_path):
    m = load_manifest(manifest_path)
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        status="preflight", manifest_sha256=digest(manifest_path), rows=[], driver_pid=os.getpid())
    if ledger["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Cannot resume with a different manifest")
    ledger["driver_pid"] = os.getpid()
    publish(root, m, ledger)
    preflight = root / "preflight"
    preflight.mkdir(exist_ok=True)
    receipt = preflight / "receipt.json"
    if not receipt.exists():
        for c in m["cases"]:
            output = preflight / c["case_id"]
            if output.exists():
                old = json.loads((output / "report.json").read_text())
                if old["status"] != "passed":
                    raise ValueError("Failed/incomplete preflight requires explicit diagnosis, not automatic rerun")
                continue
            supervision = root / "supervision" / ("preflight-" + c["case_id"])
            r = bounded(["bash", SCRIPT, "paired", "preflight", "--manifest", str(manifest_path),
                         "--case-id", c["case_id"], "--output", str(output)], supervision, 240)
            if r.get("returncode") != 0 or r.get("stop_reason"):
                ledger.update(status="preflight_failed", failed_case=c["case_id"])
                publish(root, m, ledger)
                return 2
        seal_preflight(manifest_path, preflight)
    done = {(r["case_id"], r["arm"]) for r in ledger["rows"]}
    ledger["status"] = "running"
    publish(root, m, ledger)
    for item in m["schedule"]:
        case_id, arm = item["case_id"], item["arm"]
        if (case_id, arm) in done:
            continue
        if INTERRUPTED:
            ledger["status"] = "interrupted"
            publish(root, m, ledger)
            return 2
        load_manifest(manifest_path)
        output = root / "episodes" / (case_id + "--" + arm)
        supervision = root / "supervision" / (case_id + "--" + arm)
        if output.exists() or supervision.exists():
            raise ValueError("Unreconciled existing attempt; never silently rerun an episode")
        if arm != "opus":
            try:
                admission()
            except RuntimeError as exc:
                ledger.update(status="paused_resource", resource_error=str(exc), next_case=item)
                publish(root, m, ledger)
                return 2
        ledger["active"] = item
        publish(root, m, ledger)
        sr = bounded(["bash", SCRIPT, "paired", "episode", "--paired-manifest", str(manifest_path),
                      "--preflight-receipt", str(receipt), "--case-id", case_id, "--arm", arm,
                      "--output", str(output)], supervision, 1300)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        audit = dict(status="not_available")
        if result_path.exists():
            with (supervision / "audit.log").open("x") as log:
                ar = subprocess.run(["bash", SCRIPT, "audit", str(output)],
                                    env=dict(os.environ, RUA_RENDER_BACKEND="osmesa"),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if (output / "audit.json").exists():
                audit = json.loads((output / "audit.json").read_text())
        ledger["rows"].append(episode_row(case_id, arm, result, audit, sr, output))
        ledger.pop("active", None)
        if ledger["rows"][-1]["classification"] == "integrity_error":
            ledger["status"] = "paused_integrity"
            publish(root, m, ledger)
            return 2
        if sr.get("stop_reason") or sr.get("status") != "completed":
            ledger["status"] = "paused_supervisor"
            publish(root, m, ledger)
            return 2
        publish(root, m, ledger)
    ledger["status"] = "completed"
    publish(root, m, ledger)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    # Reuse the other RUA supervisor's singleton, plus the per-episode policy lock.
    with (ROOT / "artifacts/rua-stage3/supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return drive(args.root, args.manifest)
        except BaseException as exc:
            save_json(args.root / "driver-error.json",
                      dict(error_type=type(exc).__name__, error=str(exc), time=time.time()))
            ledger_path = args.root / "ledger.json"
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                ledger.update(status="driver_error", error_type=type(exc).__name__, error=str(exc))
                save_json(ledger_path, ledger)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
