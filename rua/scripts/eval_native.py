"""Small, fixed native WLA/LIBERO pilot. No agent, API, or hybrid controller."""
from __future__ import annotations

import argparse
import faulthandler
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bootstrap import CODE, ROOT, WLA, configure

configure()

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero_utils import get_libero_env, get_libero_image, quat2axisangle
from native_contract import load_pilot
from native_rollout import aggregate, pilot_cases, run_episode
from run_libero_eval import set_seed_everywhere
from wla_utils import WLA0


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def interrupted(signum, frame):
    raise InterruptedError(f"Interrupted by signal {signum}")


def timed_out(signum, frame):
    raise TimeoutError("wall_clock_budget")


def resources():
    free, total = [int(v.strip()) for v in subprocess.check_output(
        ["nvidia-smi", "-i", os.environ.get("RUA_GPU", "0"), "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits"], text=True, timeout=15).strip().split(",")]
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(gpu0_free_mib=free, gpu0_total_mib=total,
                cpu_available_kib=int(memory["MemAvailable"].split()[0]))


def graphics_errors():
    result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10)
    return dict(returncode=result.returncode,
                recent_gpu0_xids=[line for line in result.stdout.splitlines()
                                 if "NVRM: Xid" in line][-30:],
                attribution="all GPUs; record only, not attributed to this worker")


def preflight(cfg, base, out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ.get("RUA_GPU", "0"):
        raise RuntimeError("Worker must expose exactly the explicitly selected GPU.")
    commit = subprocess.check_output(["git", "-C", str(WLA), "rev-parse", "HEAD"], text=True).strip()
    if commit != cfg["wla_commit"]:
        raise RuntimeError(f"WLA revision changed: {commit}")
    lock_path = ROOT / "models/wla-models.lock.json"
    lock = json.loads(lock_path.read_text())
    receipts = sorted(base.glob("models-offline-check-*/report.json"), key=lambda p: p.stat().st_mtime)
    if lock["status"] != "download_complete" or not receipts:
        raise RuntimeError("Verified model download and offline check are required.")
    receipt = json.loads(receipts[-1].read_text())
    if receipt["status"] != "passed" or [
        (m["repo_id"], m["revision"]) for m in receipt["models"]
    ] != [(m["repo_id"], m["revision"]) for m in lock["models"]]:
        raise RuntimeError("Latest offline check does not match the model lock.")
    # Copy exact execution inputs, not just links to mutable workspace files.
    snapshots = out / "provenance"
    snapshots.mkdir()
    sources = [
        CODE / "configs/pilot.json", lock_path, base / "requirements.lock",
        WLA / "experiments/libero/run_libero_eval.py",
        WLA / "experiments/libero/wla_utils.py",
        WLA / "experiments/libero/libero_utils.py",
        WLA / "utils/transforms.py", WLA / "configs/norm_stats.json",
        Path(__file__), CODE / "scripts/native_rollout.py", CODE / "scripts/native_contract.py",
        CODE / "scripts/run_stage1.sh", CODE / "scripts/runtime_env.sh",
        CODE / "scripts/isolated_env.py",
    ]
    hashes = {}
    for source in sources:
        shutil.copy2(source, snapshots / source.name)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    save_json(snapshots / "hashes.json", hashes)
    before = resources()
    if before["gpu0_free_mib"] < 24 * 1024 or before["cpu_available_kib"] < 64 * 1024**2:
        raise RuntimeError(f"Insufficient headroom; no model loaded: {before}")
    return dict(resources_before=before, offline_check=str(receipts[-1]),
                model_lock_sha256=hashes[str(lock_path)], source_sha256=hashes)


def evaluate_case(policy, env, instruction, initial_state, cfg, out, task_id, state_id, task):
    out.mkdir()
    started = time.monotonic()
    metadata = dict(task_id=task_id, init_state_id=state_id, task_name=task.name,
                    instruction=instruction, artifact_directory=str(out))
    save_json(out / "result.json", dict(metadata, status="running"))
    np.save(out / "initial_state.npy", initial_state)
    writers = {}
    video_frames = 0
    chunks = 0
    inference_seconds = 0.0
    result = dict(status="error", official_success=False, stop_reason="setup_error")
    trace = None
    artifact_errors = []

    def emit(event, obs):
        event["elapsed_seconds"] = time.monotonic() - started
        event["proprioception"] = {key: np.asarray(obs[key]).tolist() for key in (
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
        trace.write(json.dumps(event, allow_nan=False) + "\n")
        trace.flush()

    def capture(obs):
        nonlocal video_frames
        for name, key in (("front", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
            frame = obs[key][::-1, ::-1].copy()
            writers[name].append_data(frame)
            if video_frames == 0:
                Image.fromarray(frame).save(out / f"{name}-initial.png")
        video_frames += 1

    def predict(history, obs, step):
        nonlocal chunks, inference_seconds
        call_started = time.monotonic()
        images = get_libero_image(history, obs, tuple(cfg["image_size"]))
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Invalid native proprioceptive state.")
        call_id = chunks
        chunks += 1
        # Saved tensors are exactly what native WLA receives, not resized video frames.
        np.savez_compressed(out / f"input-{call_id:03d}.npz", state=state,
                            images=np.stack([image.numpy() for image in images[0]]))
        emit(dict(event="inference_started", call_id=call_id, task_step=step,
                  history_pre_step_index=None if history is None else max(0, step - cfg["history_obs_step"]),
                  input_image_shapes=[list(image.shape) for image in images[0]], state=state.tolist()), obs)
        raw = policy.inference({"full_image": images, "state": state}, instruction)
        torch.cuda.synchronize()
        seconds = time.monotonic() - call_started
        inference_seconds += seconds
        if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (8, 7) or not torch.isfinite(raw).all():
            raise ValueError("Invalid native WLA output: no actions executed.")
        raw_cpu = raw.detach().cpu().numpy()
        np.save(out / f"actions-{call_id:03d}.npy", raw_cpu)
        emit(dict(event="inference_completed", call_id=call_id, task_step=step,
                  raw_actions=raw_cpu.tolist(), elapsed_inference_seconds=seconds), obs)
        return raw_cpu

    try:
        trace = (out / "steps.jsonl").open("w")
        for name in ("front", "wrist"):
            writers[name] = imageio.get_writer(out / f"{name}.mp4", fps=20, codec="libx264",
                                              quality=8, ffmpeg_params=["-threads", "1"])
        signal.setitimer(signal.ITIMER_REAL, cfg["episode_timeout_seconds"])
        result = run_episode(env, initial_state, predict, cfg, emit, capture)
    except BaseException as exc:
        result.update(error=str(exc), traceback=traceback.format_exc(), stop_reason=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for name, writer in writers.items():
            try:
                writer.close()
            except Exception as exc:
                artifact_errors.append(f"{name}: {exc}")
        if trace is not None:
            trace.close()
        result.update(metadata, video_frames=video_frames, video_fps=20,
                      video_includes_initial_and_settling=True,
                      inference_seconds=inference_seconds, artifact_errors=artifact_errors,
                      total_case_seconds=time.monotonic() - started,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(out / "result.json", result)
    return result


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--isolate-env", action="store_true",
                        help="Keep native LIBERO in a separate spawn process from CUDA inference.")
    args = parser.parse_args()
    cfg = load_pilot(CODE / "configs/pilot.json")
    cases = pilot_cases(cfg, args.debug)
    base = ROOT / "artifacts/wla-stage1"
    out = Path(tempfile.mkdtemp(prefix="native-debug-" if args.debug else "native-pilot-", dir=base))
    report = dict(status="preflight", scope="debug" if args.debug else "fixed_15_episode_pilot",
                  config=cfg, cases=cases, results=[], pid=os.getpid(),
                  environment_process="isolated_spawn" if args.isolate_env else "same_process",
                  environment_worker_pids=[],
                  started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    started = time.monotonic()
    policy_lock = None
    env = None
    print(json.dumps({"run_directory": str(out), "pid": os.getpid()}), flush=True)

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["summary"] = aggregate(cases, report["results"])
        if args.debug or report["status"] != "completed":
            report["summary"]["stage1_gate_passed"] = False
        save_json(out / "report.json", report)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, timed_out)
    try:
        policy_lock = (base / ".wla-policy.lock").open("a+")
        fcntl.flock(policy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report.update(preflight(cfg, base, out))
        report["graphics_errors_before"] = graphics_errors()
        set_seed_everywhere(cfg["model_seed"])  # Exactly once before loading, as upstream.
        torch.cuda.set_per_process_memory_fraction(
            (12 * 1024) / report["resources_before"]["gpu0_total_mib"], device=0)
        torch.cuda.reset_peak_memory_stats()
        report["status"] = "loading_native_wla"
        save()
        load_started = time.monotonic()
        policy = WLA0(cfg["model_id"], None, str(WLA / "configs/norm_stats.json"), cfg["unnorm_key"])
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.monotonic() - load_started
        suite = benchmark.get_benchmark_dict()[cfg["suite"]](task_order_index=cfg["task_order_index"])
        current_task = None
        for task_id, state_id in cases:
            if current_task != task_id:
                if env is not None:
                    env.close()
                    env = None
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                if args.isolate_env:
                    from isolated_env import IsolatedLiberoEnv
                    env = IsolatedLiberoEnv(cfg, task_id)
                    instruction = env.instruction
                    report["environment_worker_pids"].append(env.process.pid)
                else:
                    env, instruction = get_libero_env(task, "wla", resolution=cfg["render_resolution"])
                current_task = task_id
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                shutil.copy2(bddl, out / "provenance" / bddl.name)
            report.update(status="running", current_case=[task_id, state_id])
            save()
            result = evaluate_case(policy, env, instruction, states[state_id], cfg,
                                   out / f"task-{task_id}_init-{state_id}", task_id, state_id, task)
            report["results"].append(result)
            save()
            print(json.dumps({k: result.get(k) for k in (
                "task_id", "init_state_id", "status", "official_success", "stop_reason",
                "task_control_steps", "wla_calls", "total_case_seconds")}), flush=True)
            # A runtime or artifact error is not permission to silently retry/change cases.
            if result["status"] != "completed" or result["artifact_errors"]:
                raise RuntimeError(f"Pilot stopped safely after {result['stop_reason']}.")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                      error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report.update(close_error=str(exc), status="failed")
        if torch.cuda.is_initialized():
            report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["graphics_errors_after"] = graphics_errors()
        report["graphics_log_unchanged"] = report.get("graphics_errors_before") == report["graphics_errors_after"]
        save()
        if policy_lock is not None:
            policy_lock.close()
        print(json.dumps({"status": report["status"], "report": str(out / "report.json"),
                          "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
