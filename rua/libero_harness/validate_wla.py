"""Debug native WLA through the shared executor; no alternate agent planning loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import traceback

import numpy as np
import torch

from .budget import Budget, BudgetExhausted
from .environment import LiberoSession
from .handoff import NativeChunkExecutor
from .paths import CODE, ROOT, configs, execution_lock, save_json
from .wla_client import WLAWorkerClient


def interrupted(signum, frame):
    if signum == signal.SIGALRM:
        raise BudgetExhausted("wall_clock_budget")
    raise KeyboardInterrupt("operator_signal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(0, 1, 2), default=0)
    # Deliberately debug-only. A new fixed pilot needs explicit baseline acceptance.
    parser.add_argument("--init-id", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Parent requires CPU OSMesa and hidden CUDA.")
    pilot, cfg = configs("claude")
    with execution_lock():
        return validate(args, pilot, cfg)


def validate(args, pilot, cfg):
    evaluation = None
    if getattr(args, "paired_manifest", None):
        from .paired import prepare_episode
        pilot, cfg, evaluation = prepare_episode(args)
    agent = getattr(args, "agent", False)
    rua_only = getattr(args, "rua_only", False)
    if agent:
        from .coordination import REVISION
        from .runner import validate_calibration, validate_handoff_admission
        cfg = dict(cfg, controller_revision=REVISION)
        validate_calibration(cfg)
        if not rua_only:
            receipt = Path(args.handoff_receipt)
            cfg.update(hybrid_handoff_receipt=str(receipt),
                       hybrid_handoff_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            validate_handoff_admission(cfg)
        if evaluation is None and (args.task_id != 0 or args.init_id != 0):
            raise ValueError("Current coordination receipt is scoped to task0/init0 debug only.")
    base = ROOT / "artifacts/rua-stage3"
    base.mkdir(parents=True, exist_ok=True)
    mode = "rua_v2_only" if rua_only else "rua_wla_hybrid" if agent else "wla_only_unified_executor"
    if evaluation is None:
        directory = Path(tempfile.mkdtemp(prefix=mode + "-debug-", dir=base))
    else:
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=False)
    result = dict(status="preflight", mode=mode, pid=os.getpid(),
                  task_id=args.task_id, initial_state_id=args.init_id, pilot=pilot,
                  official_success=False, task_steps=0, initialization_steps=0,
                  wla_calls=0, qwen_calls=0, model_requests=0, request_records=[],
                  benchmark_acceptance=False, scope="debug_not_paired_pilot",
                  artifact_directory=str(directory), render_backend="osmesa")
    if agent:
        result.update(backend=cfg["backend"], configuration=cfg, controller_revision=cfg["controller_revision"])
    if evaluation is not None:
        result.update(scope="paired20_engineering_comparison", evaluation=evaluation)
    sources = [*sorted((CODE / "libero_harness").glob("*.py")),
               CODE / "scripts/native_contract.py", CODE / "scripts/native_rollout.py",
               CODE / "scripts/run_show_harness.sh", CODE / "configs/pilot.json",
               CODE / "configs/show_harness_libero.json"]
    if agent:
        sources += [CODE / "vendor/show_harness/core/runners/real.py",
                    CODE / "vendor/show_harness/core/vlm/roles.py",
                    CODE / "vendor/show_harness/plugins/recovery/plugin.py",
                    CODE / "vendor/show-harness-manifest.json"]
    snapshot = directory / "source"
    snapshot.mkdir()
    result["source_sha256"] = {}
    for source in sources:
        relative = source.relative_to(CODE)
        result["source_sha256"][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    save_json(directory / "result.json", result)
    print(json.dumps(dict(run_root=str(directory), status=result["status"])), flush=True)
    worker = env = session = executor = budget = logger = client = native_result = None
    start = time.monotonic()
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        if not rua_only:
            service_port = getattr(args, "wla_service_port", None)
            if service_port:
                from .service_wla_client import ServiceWLAClient
                worker = ServiceWLAClient(directory / "wla", pilot, port=service_port)
                result.update(wla_backend="shared_service", wla_service_port=service_port,
                              worker_startup=worker.startup)
            else:
                worker = WLAWorkerClient(directory / "wla", pilot)
                result.update(wla_backend="local_worker", worker_pid=worker.process.pid,
                              worker_startup=worker.startup)
        else:
            import random
            random.seed(pilot["model_seed"])
            np.random.seed(pilot["model_seed"])
            torch.manual_seed(pilot["model_seed"])
        result["status"] = "initializing"
        save_json(directory / "result.json", result)
        if torch.cuda.is_initialized():
            raise RuntimeError("Parent must never initialize CUDA.")
        from libero.libero import benchmark, get_libero_path
        from libero_utils import get_libero_env
        # Native order: seed+load (worker), suite, task+states, environment, rollout.
        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=pilot["task_order_index"])
        task = suite.get_task(args.task_id)
        states = suite.get_task_init_states(args.task_id)
        definition = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        shutil.copy2(definition, directory / "task.bddl")
        result.update(task_name=task.name, task_instruction=task.language,
                      task_definition_sha256=hashlib.sha256(definition.read_bytes()).hexdigest())
        env, instruction = get_libero_env(task, "wla", resolution=pilot["render_resolution"])
        if instruction != task.language:
            raise ValueError("Task instruction changed.")
        budget = Budget(pilot["max_env_steps"], pilot["max_agent_requests"],
                        pilot["episode_timeout_seconds"])
        signal.setitimer(signal.ITIMER_REAL, budget.timeout)
        session = LiberoSession(env, states[args.init_id], cfg, pilot, budget, directory)
        if evaluation is not None:
            from .paired import validate_episode_state
            validate_episode_state(evaluation, task, states[args.init_id], definition, session)
        executor = NativeChunkExecutor(session, worker, instruction) if worker is not None else None
        result["status"] = "running"
        save_json(directory / "result.json", result)
        if agent:
            from core.record.episode_logger import EpisodeLogger
            from .clients import make_client
            from .runner import build_runner
            logger = EpisodeLogger(directory / "native", args.task_id,
                                   variant=mode + "-" + cfg["backend"], video_fps=2)
            logger.write_metadata(result)
            client = make_client(cfg, budget, directory / "requests")
            native_result = build_runner(session, logger, client, instruction, pilot, cfg,
                                         wla_executor=executor).run()
            logger = None  # Original runner closed it.
        else:
            while budget.steps < budget.max_steps and not (session.success() or session.returned_done):
                returned = executor.execute()
                result.update(task_steps=budget.steps, wla_calls=executor.calls,
                              last_wla_return=vars(returned))
                save_json(directory / "result.json", result)
                print(json.dumps(dict(task_steps=budget.steps, wla_calls=executor.calls,
                                      official_success=session.success(), stop_reason=returned.stop_reason)),
                      flush=True)
        result.update(status="completed", end_reason=(
            native_result.end_reason if native_result is not None else
            "official_success" if session.success() else "environment_terminated"
            if session.returned_done else "environment_step_budget"))
        if agent and session.errors:
            result.update(status="error", environment_errors=session.errors)
    except BaseException as exc:
        result.update(status="budget_exhausted" if isinstance(exc, BudgetExhausted) else
                      "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      end_reason=str(exc) if isinstance(exc, BudgetExhausted) else type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_errors = []
        if logger is not None:
            try:
                logger.close(success=bool(session and session.success()), fps=2)
            except BaseException as exc:
                cleanup_errors.append(f"logger:{type(exc).__name__}")
        if session is not None:
            result.update(official_success=session.success(), task_steps=session.budget.steps,
                          initialization_steps=session.initialization_steps,
                          environment_done=session.returned_done, uncertain_step=session.uncertain_step,
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
            result.update(worker_exitcode=worker.exitcode,
                          worker_alive=(worker.process.poll() is None)
                          if isinstance(worker, WLAWorkerClient) else False)
        if agent:
            records = client.records if client else []
            usage = {}
            for record in records:
                for key, value in record.get("usage", {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
            result.update(model_requests=budget.requests if budget else 0,
                          request_records=records, model_usage=usage,
                          returned_models=sorted({r["returned_model"] for r in records if r.get("returned_model")}),
                          native_result=vars(native_result) if native_result else None)
        result.update(elapsed_seconds=time.monotonic() - start,
                      episode_seconds=None if budget is None else time.monotonic() - budget.started,
                      torch_cuda_initialized=torch.cuda.is_initialized(), cleanup_errors=cleanup_errors)
        if cleanup_errors or result["torch_cuda_initialized"]:
            result.update(status="error", end_reason="cleanup_or_parent_cuda_error")
        save_json(directory / "result.json", result)
    print(json.dumps({k: result.get(k) for k in
                      ("status", "official_success", "task_steps", "wla_calls", "end_reason",
                       "artifact_directory", "worker_exitcode")}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
