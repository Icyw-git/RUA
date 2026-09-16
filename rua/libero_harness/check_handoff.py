"""Physical handoff diagnostics using recorded native WLA chunks, not a new policy run.

Replays fixed debug prefixes into fresh LIBERO instances, then tests all six
calibrated base-frame translations. This does NOT certify wrist-image semantics
at arbitrary orientations and must not be treated as a benchmark result.
"""
import argparse
import json
import hashlib
from pathlib import Path
import signal
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .budget import Budget
from .environment import LiberoAtomicController, LiberoSession
from .handoff import NativeChunkExecutor
from .paths import CODE, ROOT, configs, execution_lock, save_json
from .validate_wla import interrupted


class RecordedPredictor:
    def __init__(self, source):
        self.source = source
        self.index = 0

    def predict(self, request, instruction, *, timeout_seconds):
        # Compare the actual native model input, not a lossy video representation.
        from libero_utils import get_libero_image, quat2axisangle
        expected = np.load(self.source / "wla" / f"input-{self.index:03d}.npz")
        obs = request.current
        images = get_libero_image(request.history, obs, (256, 256))
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))
        np.testing.assert_allclose(state, expected["state"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.stack([img.numpy() for img in images[0]]),
                                   expected["images"], atol=1e-6, rtol=0)
        raw = np.load(self.source / "wla" / f"actions-{self.index:03d}.npy")
        self.index += 1
        return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--coordinated", action="store_true")
    parser.add_argument("--recovery-tests", type=Path)
    args = parser.parse_args()
    source_result = json.loads((args.source / "result.json").read_text())
    if source_result["mode"] != "wla_only_unified_executor" or source_result["initial_state_id"] != 0:
        raise ValueError("Need a recorded unified WLA debug episode.")
    from libero.libero import benchmark
    from libero_utils import get_libero_env
    pilot, cfg = configs("claude")
    test_report = None
    if args.coordinated:
        from .coordination import REVISION
        from .runner import HANDOFF_SOURCE_FILES
        cfg["controller_revision"] = REVISION
        if args.recovery_tests is None:
            raise ValueError("Coordinated handoff requires the recorded recovery regression tests.")
        test_report = ET.parse(args.recovery_tests)
        suites = list(test_report.iter("testsuite"))
        if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in
                             ("failures", "errors", "skipped")):
            raise ValueError("Recovery tests must pass without skips.")
        names = {t.attrib["name"] for t in test_report.iter("testcase")}
        required = {"test_v2_closed_wla_returns_to_model_without_release",
                    "test_v2_atomic_grasp_keeps_native_recovery",
                    "test_v2_handoff_discards_requested_action_and_refreshes_history",
                    "test_v2_invalid_token_never_stabilizes_or_moves"}
        if not required.issubset(names):
            raise ValueError("Missing mandatory coordination regression tests.")
    with execution_lock():
        out = Path(tempfile.mkdtemp(prefix="recorded-handoff-", dir=ROOT / "artifacts/rua-stage3"))
        report = dict(status="running", source=str(args.source), cases=[], root=str(out),
                      scope="recorded_native_action_replay_not_fresh_model_inference",
                      wrist_direction_certified=False, benchmark_acceptance=False)
        if args.coordinated:
            report.update(controller_revision=cfg["controller_revision"],
                          scope="debug_task0_init0_front_view_only",
                          atomic_calibration_sha256=cfg["calibration_sha256"],
                          recovery_tests=str(args.recovery_tests),
                          recovery_tests_sha256=hashlib.sha256(args.recovery_tests.read_bytes()).hexdigest(),
                          source_sha256={name: hashlib.sha256((CODE / name).read_bytes()).hexdigest()
                                         for name in HANDOFF_SOURCE_FILES})
        save_json(out / "report.json", report)
        print(json.dumps(dict(run_root=str(out))), flush=True)
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            for chunks in (2, 6):  # 16 steps open; 48 steps closed, before native success.
                directory = out / f"prefix-{chunks * 8}"
                directory.mkdir()
                case = dict(prefix_steps=chunks * 8, status="running", directions=[])
                report["cases"].append(case)
                env = session = None
                try:
                    suite = benchmark.get_benchmark_dict()[pilot["suite"]](
                        task_order_index=pilot["task_order_index"])
                    task = suite.get_task(source_result["task_id"])
                    initial = suite.get_task_init_states(source_result["task_id"])[0]
                    env, instruction = get_libero_env(task, "wla", resolution=pilot["render_resolution"])
                    budget = Budget(220, 0, 180)
                    signal.setitimer(signal.ITIMER_REAL, budget.timeout)
                    session = LiberoSession(env, initial, cfg, pilot, budget, directory)
                    initial_quat = session.obs["robot0_eef_quat"].copy()
                    if args.coordinated:
                        from .coordination import validate_front_camera
                        index = env.sim.model.camera_name2id("agentview")
                        front_reference = np.r_[env.sim.data.cam_xpos[index],
                                                env.sim.data.cam_xmat[index], env.sim.model.cam_fovy[index]]
                    executor = NativeChunkExecutor(session, RecordedPredictor(args.source), instruction)
                    for _ in range(chunks):
                        executor.execute()
                    atomic = LiberoAtomicController(session, cfg)
                    command = session.trajectory.last_gripper_command
                    angle = (Rotation.from_quat(initial_quat).inv()
                             * Rotation.from_quat(session.obs["robot0_eef_quat"])).magnitude()
                    case.update(rotation_degrees=float(np.rad2deg(angle)),
                                handoff_gripper_command=command,
                                native_inputs_match_saved=True)
                    # Lift first to provide clearance. Every translation is still billed.
                    for token in ("MV_UP", "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_DOWN"):
                        before = session.obs["robot0_eef_pos"].copy()
                        result = atomic.step(token)
                        if result.kind == "handoff":
                            case["counted_handoff"] = dict(
                                **session.last_handoff,
                                drift_m=(session.obs["robot0_eef_pos"] - before).tolist())
                            # Separate re-observed atomic decision, never an internally queued action.
                            before = session.obs["robot0_eef_pos"].copy()
                            result = atomic.step(token)
                        if args.coordinated:
                            validate_front_camera(session, front_reference)
                        delta = session.obs["robot0_eef_pos"] - before
                        vector = np.asarray(cfg["token_vectors"][token])
                        parallel = float(np.dot(delta, vector))
                        transverse = float(np.linalg.norm(delta - parallel * vector))
                        row = dict(token=token, actual_delta_m=delta.tolist(),
                                   intended_axis_motion_m=parallel, transverse_motion_m=transverse,
                                   gripper_command=session.trajectory.last_gripper_command)
                        case["directions"].append(row)
                        assert parallel > .005, row
                        assert transverse < .01, row
                        assert session.trajectory.last_gripper_command == command, row
                        assert result.gripper_closed == (command > 0)
                    latest = session.trajectory.prediction_input(session.obs, budget.steps)
                    counted_hold = 5 if args.coordinated else 0
                    assert latest.task_step == chunks * 8 + 18 + counted_hold
                    assert latest.sequence == chunks * 8 + 28 + counted_hold
                    case.update(status="passed", task_steps=budget.steps,
                                latest_wla_input_step=latest.task_step)
                except BaseException as exc:
                    case.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    if session is not None:
                        save_json(directory / "result.json", dict(
                            **case, initialization_steps=session.initialization_steps,
                            official_success=session.success(), uncertain_step=session.uncertain_step,
                            control_video_frames=session.video_frames))
                        errors = session.close()
                        if errors:
                            case.update(status="failed", cleanup_errors=errors)
                    if env is not None:
                        env.close()
                    save_json(out / "report.json", report)
                    if case.get("cleanup_errors"):
                        raise RuntimeError("Failed to save handoff videos.")
            report["status"] = "passed"
            if args.coordinated:
                report.update(front_view_direction_certified=True, recovery_handoff_certified=True)
        except BaseException as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
        finally:
            report["torch_cuda_initialized"] = torch.cuda.is_initialized()
            if report["torch_cuda_initialized"]:
                report["status"] = "failed"
            save_json(out / "report.json", report)
            print(json.dumps(report), flush=True)
        return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
