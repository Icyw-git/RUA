"""Wire the vendored native Show-Harness loop to LIBERO. No alternate agent loop."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from core.agent.stage_control import Controller, StageControlSuite
from core.record.episode_logger import EpisodeLogger
from core.runners.real import RealEpisodeRunner
from core.v0_types import V0Config
from core.vlm.roles import ControllerAgent
from plugins.mem_text.plugin import MemTextPlugin
from plugins.proprioception.plugin import ProprioceptionPlugin
from plugins.recovery.plugin import RecoveryPlugin
from plugins.subgoal import SubgoalPlanner
from plugins.subgoal.agent import SubgoalPlannerAgent

from .budget import Budget, BudgetExhausted
from .clients import make_client
from .environment import LiberoSession, LiberoAtomicController
from .paths import ARTIFACTS, CODE, ROOT, VENDOR, configs, save_json, execution_lock


HANDOFF_SOURCE_FILES = (
    "libero_harness/environment.py", "libero_harness/handoff.py",
    "libero_harness/coordination.py", "libero_harness/wla_plugin.py",
    "libero_harness/runner.py", "libero_harness/check_handoff.py",
    "libero_harness/validate_wla.py", "libero_harness/hybrid_debug.py",
    "libero_harness/wla_client.py", "libero_harness/wla_worker.py",
    "scripts/run_show_harness.sh", "scripts/supervise_harness.py",
    "vendor/show_harness/core/runners/real.py",
    "vendor/show_harness/core/vlm/roles.py",
    "vendor/show_harness/plugins/recovery/plugin.py",
    "tests_stage3/test_coordination.py",
)


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    return dict(returncode=result.returncode, rows=result.stdout.strip().splitlines())


def build_runner(session, logger, client, task, pilot, cfg, *, wla_executor=None):
    if wla_executor is not None and wla_executor.session is not session:
        raise ValueError("WLA and the agent must share the same environment owner.")
    if wla_executor is not None:
        validate_handoff_admission(cfg)
    common = (VENDOR / "prompts/common_context.txt").read_text()
    template = (VENDOR / "prompts/controller.txt").read_text()
    recovery = RecoveryPlugin(empty_width_m=cfg["empty_width_m"],
                              open_width_m=cfg["open_width_m"])
    coordination = None
    if cfg.get("controller_revision") == "front_view_handoff_v2":
        from .coordination import controller_prompt, CoordinationContext, GripperOwnerRecovery
        template = controller_prompt(template)
        coordination = CoordinationContext(session)
        recovery = GripperOwnerRecovery(recovery, session)
    planner = SubgoalPlanner(SubgoalPlannerAgent(
        client, common, max_tokens=cfg["planner_max_tokens"]))
    if wla_executor is not None:
        from .wla_plugin import WLAActionPlugin
        wla_plugin = WLAActionPlugin(wla_executor, coordination)
    else:
        wla_plugin = coordination
    control_agent = ControllerAgent(
        client=client, prompt_template=template,
        common_context=common, cot_mode=False, gripper_color="black",
        proprio_plugin=ProprioceptionPlugin(
            fine_step_m=cfg["nominal_step_m"], coarse_step_m=None),
        mem_text_plugin=MemTextPlugin(max_recent=cfg["recent_moves"]),
        table_height_m=cfg["effective_table_height_m"],
        extra_action_plugin=wla_plugin,
    )
    return RealEpisodeRunner(
        session=session, controller=LiberoAtomicController(session, cfg, wla_executor=wla_executor),
        planner=planner,
        controls=StageControlSuite(Controller(control_agent)), logger=logger,
        config=V0Config(max_subgoal_steps=cfg["max_subgoal_decisions"], max_replans=0,
                        video_fps=2, controller_prompt_log_every=1),
        task=task, gripper_color="black", max_steps=cfg["max_decisions"],
        loop_period_s=0, use_wrist_image=True, debug=False,
        recovery_plugin=recovery,
        recent_moves_max=cfg["recent_moves"], environment_boundary=session,
    )


def alarm_handler(signum, frame):
    if signum == signal.SIGALRM:
        raise BudgetExhausted("wall_clock_budget")
    raise KeyboardInterrupt("operator_signal")


def validate_calibration(cfg):
    if cfg.get("calibration_status") != "passed":
        raise ValueError("Refusing actions before successful physical calibration.")
    receipt = Path(cfg["calibration_receipt"])
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != cfg["calibration_sha256"]:
        raise ValueError("Calibration receipt hash mismatch.")
    measured = json.loads(receipt.read_text())
    required = {"move_amplitude": "selected_amplitude", "nominal_step_m": "nominal_step_m",
                "token_vectors": "suggested_token_vectors", "wrist_extra_flip": "wrist_extra_flip",
                "move_env_steps": "move_env_steps", "gripper_env_steps": "gripper_env_steps",
                "empty_width_m": "empty_width_m", "open_width_m": "open_width_m",
                "effective_table_height_m": "effective_table_height_m"}
    for setting, measurement in required.items():
        if cfg[setting] != measured[measurement]:
            raise ValueError(f"Configuration differs from calibration: {setting}")


def validate_handoff_admission(cfg):
    """Fail closed until motion and recovery handoffs have a traceable receipt."""
    path = cfg.get("hybrid_handoff_receipt")
    digest = cfg.get("hybrid_handoff_sha256")
    if not path or not digest:
        raise ValueError("Autonomous WLA handoff is not certified; WLA-only validation remains available.")
    receipt = Path(path).read_bytes()
    if hashlib.sha256(receipt).hexdigest() != digest:
        raise ValueError("Hybrid handoff receipt hash mismatch.")
    report = json.loads(receipt)
    v2 = cfg.get("controller_revision") == "front_view_handoff_v2"
    if v2:
        if (report.get("status") != "passed"
                or report.get("front_view_direction_certified") is not True
                or report.get("recovery_handoff_certified") is not True
                or report.get("controller_revision") != cfg["controller_revision"]
                or report.get("scope") != "debug_task0_init0_front_view_only"):
            raise ValueError("Front-view motion and ownership-aware recovery must be verified for v2.")
        tests = Path(report["recovery_tests"])
        if hashlib.sha256(tests.read_bytes()).hexdigest() != report["recovery_tests_sha256"]:
            raise ValueError("Recovery test evidence hash mismatch.")
    elif (report.get("status") != "passed"
            or report.get("wrist_direction_certified") is not True
            or report.get("recovery_handoff_certified") is not True):
        raise ValueError("Both directional and recovery handoff certification are required.")
    validate_calibration(cfg)
    if report.get("atomic_calibration_sha256") != cfg["calibration_sha256"]:
        raise ValueError("Hybrid handoff was certified with a different calibration.")
    required = HANDOFF_SOURCE_FILES if v2 else ("libero_harness/environment.py", "libero_harness/handoff.py",
                "vendor/show_harness/core/runners/real.py",
                "vendor/show_harness/plugins/recovery/plugin.py")
    for name in required:
        if report.get("source_sha256", {}).get(name) != hashlib.sha256((CODE / name).read_bytes()).hexdigest():
            raise ValueError(f"Hybrid handoff certification is stale: {name}")
    if cfg.get("paired_manifest"):
        from .paired import case_admission
        case_admission(cfg["paired_manifest"], cfg["paired_preflight"], cfg["paired_case_id"])


def run_episode(suite, task_id, init_id, root, pilot, cfg):
    from libero_utils import get_libero_env
    from libero.libero import get_libero_path
    task = suite.get_task(task_id)
    directory = root / f"task-{task_id}-init-{init_id}"
    directory.mkdir()
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    definition = bddl_path.read_bytes()
    (directory / "task.bddl").write_bytes(definition)
    budget = Budget(pilot["max_env_steps"], pilot["max_agent_requests"],
                    pilot["episode_timeout_seconds"])
    report = dict(status="running", mode="rua_only", task_id=task_id, initial_state_id=init_id,
                  task_name=task.name, task_instruction=task.language, official_success=False,
                  task_definition_sha256=hashlib.sha256(definition).hexdigest(),
                  wla_calls=0, qwen_calls=0, backend=cfg.get("backend", "claude"),
                  render_backend=os.environ.get("MUJOCO_GL", "unspecified"),
                  task_steps=0, initialization_steps=0,
                  artifacts=str(directory), pilot=pilot, configuration=cfg)
    save_json(directory / "result.json", report)
    env = session = logger = client = None
    result = None
    error = None
    logger_closed = False
    start = time.monotonic()
    try:
        report["gpu_before"] = gpu_snapshot()
        random.seed(pilot["model_seed"])
        np.random.seed(pilot["model_seed"])
        torch.manual_seed(pilot["model_seed"])
        if torch.cuda.is_initialized():
            raise ValueError("RUA-only process unexpectedly initialized torch CUDA.")
        signal.setitimer(signal.ITIMER_REAL, budget.timeout)
        env, instruction = get_libero_env(task, "wla", resolution=cfg["agent_image_size"])
        if instruction != task.language:
            raise ValueError("Original benchmark task text changed.")
        env.seed(pilot["environment_seed"])
        logger = EpisodeLogger(directory / "native", task_id, variant="show-harness-" + cfg.get("backend", "claude"),
                               video_fps=2)
        logger.write_metadata(report)
        session = LiberoSession(env, suite.get_task_init_states(task_id)[init_id],
                                cfg, pilot, budget, directory)
        client = make_client(cfg, budget, directory / "requests")
        result = build_runner(session, logger, client, instruction, pilot, cfg).run()
        logger_closed = True
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        success = bool(session and session.success())
        cleanup_errors = []
        if logger is not None and not logger_closed:
            try:
                logger.close(success=success, fps=2)
            except Exception as exc:
                cleanup_errors.append(f"logger:{type(exc).__name__}")
        if session is not None:
            cleanup_errors.extend(session.close())
            report.update(initialization_steps=session.initialization_steps,
                          official_success=success, environment_done=session.returned_done,
                          uncertain_step=session.uncertain_step, environment_errors=session.errors,
                          control_video_frames=session.video_frames)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                cleanup_errors.append(f"env:{type(exc).__name__}")
        records = client.records if client else []
        usage = {}
        for record in records:
            for key, value in record.get("usage", {}).items():
                if isinstance(value, (float, int)):
                    usage[key] = usage.get(key, 0) + value
        report.update(status="completed" if error is None else "error", task_steps=budget.steps,
                      qwen_calls=budget.requests if cfg.get("backend") == "qwen" else 0,
                      model_requests=budget.requests, model_usage=usage, request_records=records,
                      returned_models=sorted({r["returned_model"] for r in records if r.get("returned_model")}),
                      elapsed_seconds=time.monotonic() - start, exception=error,
                      cleanup_errors=cleanup_errors, torch_cuda_initialized=torch.cuda.is_initialized(),
                      end_reason=result.end_reason if result else
                      ("interrupted" if error and error["type"] == "KeyboardInterrupt" else
                       (error["message"] if error else "unknown")),
                      native_result=vars(result) if result else None)
        if report["torch_cuda_initialized"]:
            report.update(status="error", end_reason="unexpected_torch_cuda_initialization")
        save_json(directory / "result.json", report)
        # Keep diagnostic failures from destroying an otherwise durable episode result.
        try:
            report["gpu_after"] = gpu_snapshot()
        except Exception as exc:
            report["gpu_snapshot_error"] = type(exc).__name__
        save_json(directory / "result.json", report)
    print(json.dumps({k: report[k] for k in ("task_id", "initial_state_id", "official_success",
                                            "task_steps", "end_reason", "elapsed_seconds")}), flush=True)
    return report


def main():
    with execution_lock():
        return _main_locked()


def _main_locked():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("debug", "pilot"))
    parser.add_argument("--backend", choices=("claude", "qwen"))
    args = parser.parse_args()
    pilot, cfg = configs(args.backend)
    validate_calibration(cfg)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock = (ARTIFACTS / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A Show-Harness episode is already running; no concurrent writer.")
    root = Path(tempfile.mkdtemp(prefix=args.mode + "-" + cfg["backend"] + "-", dir=ARTIFACTS))
    cases = [(0, pilot["debug_init_state_id"])] if args.mode == "debug" else [
        (task, initial) for task in pilot["task_ids"] for initial in pilot["eval_init_state_ids"]]
    summary = dict(status="running", mode=args.mode, backend=cfg["backend"], requested_model=cfg["model"],
                   render_backend=os.environ.get("MUJOCO_GL", "unspecified"),
                   expected_episodes=len(cases),
                   cases=cases, episodes=[], pid=os.getpid(), root=str(root),
                   scope="engineering_validation_not_formal_benchmark",
                   wla_acceptance="incomplete_not_replaced_by_stage2")
    files = [*sorted((CODE / "libero_harness").glob("*.py")),
             CODE / "configs/show_harness_libero.json", CODE / "configs/pilot.json",
             CODE / "vendor/show-harness-manifest.json", CODE / "scripts/run_show_harness.sh"]
    if cfg["backend"] == "qwen":
        files.append(CODE / "configs/show_harness_qwen.json")
    summary["source_sha256"] = {
        str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save_json(root / "report.json", summary)
    print(json.dumps({"run_root": str(root), "cases": cases}), flush=True)
    for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, alarm_handler)
    try:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=pilot["task_order_index"])
        for task_id, init_id in cases:
            episode = run_episode(suite, task_id, init_id, root, pilot, cfg)
            summary["episodes"].append({k: episode[k] for k in
                ("task_id", "initial_state_id", "official_success", "task_steps",
                 "elapsed_seconds", "model_requests", "model_usage", "end_reason", "status", "artifacts")})
            save_json(root / "report.json", summary)
            if (episode["end_reason"] == "interrupted" or episode.get("uncertain_step")
                    or episode["end_reason"].startswith("runtime_error:")):
                break
        summary["status"] = "completed" if len(summary["episodes"]) == len(cases) else "interrupted"
    except BaseException as exc:
        summary.update(status="error", error=type(exc).__name__, traceback=traceback.format_exc())
    finally:
        episodes = summary["episodes"]
        summary["official_successes"] = sum(x["official_success"] for x in episodes)
        summary["official_success_rate"] = (summary["official_successes"] / len(episodes)) if episodes else None
        summary["stage2_acceptance_passed"] = (
            args.mode == "pilot" and len(episodes) == 15 and summary["official_successes"] > 0
            and all(x["status"] == "completed" and not x["end_reason"].startswith("runtime_error")
                    for x in episodes))
        save_json(root / "report.json", summary)
        print(json.dumps({"report": str(root / "report.json"), "status": summary["status"],
                          "official_successes": summary["official_successes"]}), flush=True)
        lock.close()
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
