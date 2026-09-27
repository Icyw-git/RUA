"""Independent Codex smoke run using the existing Show-Harness controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .budget import Budget
from .codex_cli_client import CodexCLIClient
from .environment import LiberoSession
from .paths import CODE, ROOT, execution_lock, save_json
from .runner import build_runner, validate_calibration


CALIBRATION_FIELDS = {
    "move_amplitude": "selected_amplitude",
    "nominal_step_m": "nominal_step_m",
    "token_vectors": "suggested_token_vectors",
    "wrist_extra_flip": "wrist_extra_flip",
    "move_env_steps": "move_env_steps",
    "gripper_env_steps": "gripper_env_steps",
    "empty_width_m": "empty_width_m",
    "open_width_m": "open_width_m",
    "effective_table_height_m": "effective_table_height_m",
}


def run(calibration: Path, max_decisions: int, timeout: int) -> tuple[Path, dict]:
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use the CPU OSMesa launcher with CUDA hidden from the parent")
    measured = json.loads(calibration.read_text())
    if measured.get("status") != "passed":
        raise ValueError("Calibration has not passed")
    pilot = json.loads((CODE / "configs/pilot.json").read_text())
    pilot.update(max_agent_requests=max(8, max_decisions + 3),
                 episode_timeout_seconds=timeout)
    cfg = json.loads((CODE / "configs/show_harness_libero.json").read_text())
    cfg.update(backend="codex", model="gpt-6-astra", max_decisions=max_decisions,
               calibration_status="passed", calibration_receipt=str(calibration.resolve()),
               calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
    cfg.update({key: measured[value] for key, value in CALIBRATION_FIELDS.items()})
    validate_calibration(cfg)

    base = ROOT / "artifacts/codex-control-smoke"
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="task0-init0-", dir=base))
    result = {"status": "initializing", "mode": "codex_show_harness_smoke",
              "backend": "codex", "task_id": 0, "initial_state_id": 0,
              "pilot": pilot, "configuration": cfg, "official_success": False,
              "task_steps": 0, "initialization_steps": 0, "wla_calls": 0,
              "qwen_calls": 0, "model_requests": 0, "request_records": [],
              "torch_cuda_initialized": False, "source_sha256": {
                  "libero_harness/codex_control_smoke.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "libero_harness/codex_cli_client.py": hashlib.sha256(
                      (CODE / "libero_harness/codex_cli_client.py").read_bytes()).hexdigest(),
              }}
    save_json(directory / "result.json", result)
    print(json.dumps({"run_root": str(directory)}), flush=True)
    env = session = logger = client = native_result = None
    budget = Budget(pilot["max_env_steps"], pilot["max_agent_requests"], timeout)
    started = time.monotonic()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero_utils import get_libero_env
        from core.record.episode_logger import EpisodeLogger

        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=pilot["task_order_index"])
        task = suite.get_task(0)
        states = suite.get_task_init_states(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        shutil.copy2(bddl, directory / "task.bddl")
        result.update(task_name=task.name, task_instruction=task.language,
                      task_definition_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest())
        env, instruction = get_libero_env(task, "wla", resolution=pilot["render_resolution"])
        if instruction != task.language:
            raise ValueError("Task instruction changed")
        session = LiberoSession(env, states[0], cfg, pilot, budget, directory)
        logger = EpisodeLogger(directory / "native", 0,
                               variant="codex-show-harness-smoke", video_fps=2)
        logger.write_metadata(result)
        client = CodexCLIClient(cfg, budget, directory / "requests")
        result["status"] = "running"
        save_json(directory / "result.json", result)
        native_result = build_runner(session, logger, client, instruction, pilot, cfg).run()
        logger = None  # The Show-Harness runner closes its logger.
        result.update(status="completed", end_reason=native_result.end_reason)
    except BaseException as exc:
        result.update(status="error", end_reason=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        if logger is not None:
            try:
                logger.close(success=bool(session and session.success()), fps=2)
            except BaseException as exc:
                cleanup_errors.append(f"logger:{type(exc).__name__}")
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
        records = [json.loads(path.read_text()) for path in sorted(
            (directory / "requests").glob("request-*/request.json"))]
        result.update(model_requests=budget.requests, request_records=records,
                      returned_models=sorted({r["returned_model"] for r in records
                                              if r.get("returned_model")}),
                      native_result=vars(native_result) if native_result else None,
                      elapsed_seconds=time.monotonic() - started,
                      torch_cuda_initialized=torch.cuda.is_initialized(),
                      cleanup_errors=cleanup_errors)
        if cleanup_errors or result["torch_cuda_initialized"]:
            result.update(status="error", end_reason="cleanup_or_parent_cuda_error")
        save_json(directory / "result.json", result)
    return directory, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    if args.max_decisions < 1 or args.timeout < 30:
        parser.error("max-decisions must be positive and timeout at least 30 seconds")
    with execution_lock():
        directory, result = run(args.calibration, args.max_decisions, args.timeout)
    print(json.dumps({"run_root": str(directory), "status": result["status"],
                      "task_steps": result["task_steps"],
                      "model_requests": result["model_requests"],
                      "official_success": result["official_success"],
                      "end_reason": result["end_reason"]}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
