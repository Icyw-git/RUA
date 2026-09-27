"""Independent Codex supervision smoke test over the existing WLA executor.

Codex decides whether to accept each WLA chunk. It never generates robot actions;
the native WLA executor remains the only task-step controller. This deliberately
does not use the Claude/Show-Harness model entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .budget import Budget, BudgetExhausted
from .claude import parse_json
from .environment import LiberoSession
from .handoff import NativeChunkExecutor
from .paths import CODE, ROOT, execution_lock, save_json
from .service_wla_client import ServiceWLAClient


MODEL = "gpt-6-astra"
CODEX = Path("/home/wcz/.local/bin/codex")
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["USE_WLA", "STOP"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}


def ask_codex(directory: Path, budget: Budget, instruction: str, observation: dict,
              decision_index: int) -> tuple[dict, dict]:
    budget.before_request()
    call_id = budget.requests
    request_dir = directory / "requests" / f"request-{call_id:03d}"
    request_dir.mkdir(parents=True)
    images = []
    for camera, key in (("front", "agentview"), ("wrist", "wrist")):
        pixels = np.asarray(observation[key], dtype=np.uint8)
        image_path = request_dir / f"{camera}.png"
        Image.fromarray(pixels).save(image_path)
        images.append({"camera": camera, "shape": list(pixels.shape),
                       "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                       "png_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    prompt = (
        "You supervise a simulated robot. The two attached images are the current "
        "front and wrist views. The task is: " + instruction + "\n\n"
        "A WLA policy can execute the next eight low-level actions, after which you "
        "will see new images. Decide whether to accept that next WLA chunk. Choose "
        "STOP only if the task is already complete or the scene clearly makes "
        "further WLA execution inappropriate. You cannot propose robot actions. "
        "Use only the supplied images and task text; do not call tools or read files. "
        "Return only JSON with exactly two keys: decision (USE_WLA or STOP) and "
        "reason (one brief sentence)."
    )
    (request_dir / "prompt.txt").write_text(prompt)
    response_path = request_dir / "response.txt"
    record = {"call_id": call_id, "decision_index": decision_index,
              "requested_model": MODEL, "image_manifest": images,
              "transport": "codex_exec_read_only", "status": "started"}
    started = time.monotonic()
    try:
        remaining = min(90, budget.timeout - (started - budget.started))
        if remaining <= 0:
            raise BudgetExhausted("wall_clock_budget")
        completed = subprocess.run(
            [str(CODEX), "exec", "--ephemeral", "--ignore-user-config",
             "--skip-git-repo-check", "--sandbox", "read-only", "--model", MODEL,
             "--cd", str(request_dir), "--image", str(request_dir / "front.png"),
             str(request_dir / "wrist.png"), "--output-last-message", str(response_path),
             prompt], cwd=request_dir, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=remaining,
        )
        (request_dir / "codex-stdout.txt").write_text(completed.stdout)
        (request_dir / "codex-stderr.txt").write_text(completed.stderr)
        if completed.returncode != 0 or not response_path.is_file():
            raise RuntimeError(f"Codex request failed with exit code {completed.returncode}")
        decision = parse_json(response_path.read_text(), DECISION_SCHEMA)
        record.update(status="received", returned_model=MODEL)
        return decision, record
    except BaseException as exc:
        record.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        record["latency_s"] = time.monotonic() - started
        save_json(request_dir / "request.json", record)


def run(port: int) -> tuple[Path, dict]:
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use the CPU OSMesa launcher with CUDA hidden from the parent")
    if not CODEX.is_file():
        raise FileNotFoundError(CODEX)
    pilot = json.loads((CODE / "configs/pilot.json").read_text())
    cfg = json.loads((CODE / "configs/show_harness_libero.json").read_text())
    cfg.update(backend="codex", model=MODEL)
    base = ROOT / "artifacts/codex-wla-trial"
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="task0-init0-", dir=base))
    result = {"status": "initializing", "mode": "rua_wla_hybrid", "backend": "codex",
              "experiment": "codex_accepts_wla_chunks_only", "task_id": 0,
              "initial_state_id": 0, "pilot": pilot, "official_success": False,
              "task_steps": 0, "initialization_steps": 0, "wla_calls": 0,
              "wla_executed_steps": 0, "qwen_calls": 0, "model_requests": 0,
              "request_records": [], "worker_alive": False,
              "torch_cuda_initialized": False, "source_sha256": {
                  "libero_harness/codex_wla_trial.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
              }}
    save_json(directory / "result.json", result)
    print(json.dumps({"run_root": str(directory)}), flush=True)
    worker = env = session = executor = None
    budget = Budget(pilot["max_env_steps"], min(20, pilot["max_agent_requests"]),
                    pilot["episode_timeout_seconds"])
    records, decisions = [], []
    started = time.monotonic()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero_utils import get_libero_env

        worker = ServiceWLAClient(directory / "wla", pilot, port=port)
        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=pilot["task_order_index"])
        task = suite.get_task(0)
        states = suite.get_task_init_states(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        shutil.copy2(bddl, directory / "task.bddl")
        result.update(task_name=task.name, task_instruction=task.language,
                      task_definition_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                      wla_backend="shared_service", wla_service_port=port,
                      worker_startup=worker.startup)
        env, instruction = get_libero_env(task, "wla", resolution=pilot["render_resolution"])
        if instruction != task.language:
            raise ValueError("Task instruction changed")
        session = LiberoSession(env, states[0], cfg, pilot, budget, directory)
        executor = NativeChunkExecutor(session, worker, instruction)
        native = directory / "native" / "codex-wla-gate"
        for camera in ("agentview", "wrist"):
            (native / "images" / camera).mkdir(parents=True)
        result["status"] = "running"
        save_json(directory / "result.json", result)

        while not session.success() and not session.returned_done and budget.steps < budget.max_steps:
            budget.check_time()
            index = len(decisions)
            observation = session.get_observation()
            decision, record = ask_codex(directory, budget, instruction, observation, index)
            records.append(record)
            for camera in ("agentview", "wrist"):
                Image.fromarray(observation[camera]).save(
                    native / "images" / camera / f"{index:04d}.png")
            decisions.append({"i": index, "act": decision["decision"],
                              "reasoning": decision["reason"],
                              "model_request_id": record["call_id"],
                              "task_step": budget.steps})
            save_json(native / "steps.json", decisions)
            if decision["decision"] == "STOP":
                result["end_reason"] = "codex_stop"
                break
            session.agent_decision_index = index
            executor.execute()
            result.update(task_steps=budget.steps, wla_calls=executor.calls,
                          wla_executed_steps=executor.executed_steps,
                          model_requests=budget.requests, request_records=records)
            save_json(directory / "result.json", result)

        result.update(status="completed", end_reason=result.get("end_reason") or
                      ("official_success" if session.success() else
                       "environment_terminated" if session.returned_done else
                       "environment_step_budget"))
    except BaseException as exc:
        result.update(status="budget_exhausted" if isinstance(exc, BudgetExhausted) else "error",
                      end_reason=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        if session is not None:
            result.update(official_success=session.success(), task_steps=budget.steps,
                          initialization_steps=session.initialization_steps,
                          uncertain_step=session.uncertain_step,
                          control_video_frames=session.video_frames)
            try:
                cleanup_errors.extend(session.close())
            except BaseException as exc:
                cleanup_errors.append(f"session:{type(exc).__name__}")
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                cleanup_errors.append(f"environment:{type(exc).__name__}")
        if executor is not None:
            result.update(wla_calls=executor.calls, wla_executed_steps=executor.executed_steps)
        if worker is not None:
            result["wla_predictions"] = worker.records
            worker.close()
            result["worker_exitcode"] = worker.exitcode
        request_records = [json.loads(path.read_text()) for path in sorted(
            (directory / "requests").glob("request-*/request.json"))]
        result.update(model_requests=budget.requests, request_records=request_records,
                      returned_models=[MODEL] if request_records else [],
                      elapsed_seconds=time.monotonic() - started,
                      worker_alive=False, torch_cuda_initialized=torch.cuda.is_initialized(),
                      cleanup_errors=cleanup_errors)
        if cleanup_errors or result["torch_cuda_initialized"]:
            result.update(status="error", end_reason="cleanup_or_parent_cuda_error")
        save_json(directory / "result.json", result)
    return directory, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wla-service-port", type=int, choices=(8120, 8121, 8122), default=8120)
    args = parser.parse_args()
    with execution_lock():
        directory, result = run(args.wla_service_port)
    print(json.dumps({"run_root": str(directory), "status": result["status"],
                      "official_success": result["official_success"],
                      "task_steps": result["task_steps"],
                      "model_requests": result["model_requests"],
                      "end_reason": result["end_reason"]}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
