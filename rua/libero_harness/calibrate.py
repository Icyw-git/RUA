"""Task-0/debug-state control calibration. No policy, agent, or object oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero_utils import get_libero_env
import robosuite.macros as macros

from .paths import ARTIFACTS, CODE, configs, save_json, execution_lock


def camera_transform(sim, name, height=512, width=512):
    """Same pinhole/axis correction as robosuite 1.4 camera_utils, without h5py imports."""
    index = sim.model.camera_name2id(name)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = sim.data.cam_xmat[index].reshape(3, 3)
    camera_to_world[:3, 3] = sim.data.cam_xpos[index]
    camera_to_world = camera_to_world @ np.diag([1, -1, -1, 1])
    f = 0.5 * height / np.tan(sim.model.cam_fovy[index] * np.pi / 360)
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]
    return intrinsic @ np.linalg.inv(camera_to_world)


def calibration_interrupted(signum, frame):
    raise InterruptedError(f"Calibration interrupted by signal {signum}")


def main():
    previous = {sig: signal.signal(sig, calibration_interrupted)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with execution_lock():
            return _main_locked()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main_locked():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-selected-only", action="store_true")
    args = parser.parse_args()
    pilot, cfg = configs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="calibration-", dir=ARTIFACTS))
    report = dict(status="running", scope="robot_camera_control_calibration_not_evaluation",
                  render_backend=os.environ.get("MUJOCO_GL", "unspecified"),
                  verify_selected_only=args.verify_selected_only,
                  installation_code_root=str(CODE.resolve()),
                  calibration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(out / "report.json", report)
    print(json.dumps({"calibration_directory": str(out)}), flush=True)
    repetitions_to_test = (cfg["move_env_steps"],) if args.verify_selected_only else (2, 3)
    amplitudes_to_test = (cfg["move_amplitude"],) if args.verify_selected_only else (0.4, 0.6, 0.8, 1.0)
    if args.verify_selected_only:
        assert cfg["calibration_status"] == "passed"
    env = None
    started = time.monotonic()
    try:
        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=0)
        env, _ = get_libero_env(suite.get_task(0), "wla", resolution=512)
        initial = suite.get_task_init_states(0)[0]

        def reset():
            env.seed(0)
            env.reset()
            obs = env.set_init_state(initial)
            for _ in range(pilot["settle_steps"]):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            assert not env.check_success()
            return obs

        obs = reset()
        table_z = float(env.env.table_offset[2])
        gripper = env.env.robots[0].gripper
        geom_names = sorted(set(name for names in gripper.important_geoms.values() for name in names))
        pads = [name for name in geom_names if "pad" in name] or geom_names
        bounds = []
        for name in pads:
            index = env.sim.model.geom_name2id(name)
            rotation = env.sim.data.geom_xmat[index].reshape(3, 3)
            half_height = float(np.abs(rotation[2]) @ env.sim.model.geom_size[index])
            bounds.append(float(env.sim.data.geom_xpos[index][2]) - half_height)
        contact_offset = float(obs["robot0_eef_pos"][2]) - min(bounds)
        effective_table = table_z + contact_offset
        report.update(table_height_m=table_z, robot_pad_geometry_names=pads,
                      eef_to_lowest_pad_m=contact_offset, effective_table_height_m=effective_table,
                      image_convention=macros.IMAGE_CONVENTION)
        # Only robot position and known table plane are used for camera calibration.
        point = obs["robot0_eef_pos"].copy()
        point[2] = table_z + 0.05
        bases = {}
        for camera, key in (("agentview", "agentview_image"),
                            ("robot0_eye_in_hand", "robot0_eye_in_hand_image")):
            transform = camera_transform(env.sim, camera)

            def project(position):
                homogeneous = transform @ np.r_[position, 1.0]
                u, v = homogeneous[:2] / homogeneous[2]
                # camera_utils is OpenCV top-down. LIBERO's raw OpenGL image is
                # bottom-up; native [::-1,::-1] therefore mirrors OpenCV horizontally.
                if macros.IMAGE_CONVENTION == "opengl":
                    return np.array([511 - u, v])
                if macros.IMAGE_CONVENTION == "opencv":
                    return np.array([511 - u, 511 - v])
                raise ValueError("Unknown camera convention.")

            bases[camera] = np.stack([
                project(point + np.eye(3)[axis] * 0.01) - project(point) for axis in (0, 1)
            ], axis=1)
            Image.fromarray(obs[key][::-1, ::-1].copy()).save(out / f"{camera}.png")
        front = bases["agentview"]
        horizontal = int(np.argmax(np.abs(front[0])))
        depth = 1 - horizontal
        right = np.eye(3)[horizontal] * np.sign(front[0, horizontal])
        forward = np.eye(3)[depth] * np.sign(front[1, depth])
        wrist = bases["robot0_eye_in_hand"]
        mirror_h = float(wrist[0] @ right[:2]) < 0
        mirror_v = float(wrist[1] @ forward[:2]) < 0
        extra_flip = "both" if mirror_h and mirror_v else "horizontal" if mirror_h else "vertical" if mirror_v else "none"
        vectors = dict(MV_FWD=forward.tolist(), MV_BACK=(-forward).tolist(),
                       MV_LEFT=(-right).tolist(), MV_RIGHT=right.tolist(),
                       MV_UP=[0, 0, 1], MV_DOWN=[0, 0, -1])
        report.update(camera_basis_pixels_per_cm={k: v.tolist() for k, v in bases.items()},
                      suggested_token_vectors=vectors, wrist_extra_flip=extra_flip)
        probes = []
        report["probes"] = probes
        for repetitions in repetitions_to_test:
            for amplitude in amplitudes_to_test:
                for token, vector in vectors.items():
                    before = reset()["robot0_eef_pos"].copy()
                    action = [*(np.array(vector) * amplitude).tolist(), 0, 0, 0, -1]
                    for _ in range(repetitions):
                        moved, _, done, _ = env.step(action)
                        assert not done and not env.check_success()
                    delta = moved["robot0_eef_pos"] - before
                    signed = float(delta @ np.array(vector))
                    assert signed > 0, (token, delta)
                    probes.append(dict(amplitude=amplitude, token=token, steps=repetitions,
                                       delta_m=delta.tolist(), signed_travel_m=signed))
                    report["completed_probes"] = len(probes)
                    save_json(out / "report.json", report)
        ranked = []
        for repetitions in repetitions_to_test:
            for amplitude in amplitudes_to_test:
                distances = [p["signed_travel_m"] for p in probes
                             if p["amplitude"] == amplitude and p["steps"] == repetitions]
                ranked.append((abs(float(np.mean(distances)) - 0.02), repetitions, amplitude, distances))
        _, selected_steps, selected, distances = min(ranked)
        assert min(distances) > 0.01 and max(distances) < 0.035, distances
        reset()
        widths = []
        for grip in (-1, 1, -1):
            for _ in range(cfg["gripper_env_steps"]):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, grip])
            widths.append(float(np.abs(obs["robot0_gripper_qpos"]).sum()))
        assert widths[0] - widths[1] > 0.02 and widths[2] - widths[1] > 0.02
        report.update(status="passed", probes=probes, selected_amplitude=selected,
                      move_env_steps=selected_steps, gripper_env_steps=cfg["gripper_env_steps"],
                      selected_travel_m=distances, nominal_step_m=float(np.mean(distances)),
                      open_close_open_widths_m=widths,
                      empty_width_m=max(0.005, widths[1] + 0.003),
                      open_width_m=min(widths[0], widths[2]) * 0.875,
                      torch_cuda_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                report.update(status="failed", cleanup_error=type(exc).__name__)
        report["elapsed_seconds"] = time.monotonic() - started
        save_json(out / "report.json", report)
        print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
