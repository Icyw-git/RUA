"""Verify genuine LIBERO rendering/control; NOT a WLA task-success experiment."""
from __future__ import annotations

import hashlib
import json
import tempfile
import traceback
from pathlib import Path

from bootstrap import CODE, ROOT, configure

configure()

import numpy as np
from PIL import Image
from libero.libero import benchmark
from libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
from native_contract import load_pilot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cfg = load_pilot(CODE / "configs/pilot.json")
    base = ROOT / "artifacts/wla-stage1"
    base.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="environment-", dir=base))
    suite = benchmark.get_benchmark_dict()[cfg["suite"]](task_order_index=0)
    results = []
    for task_id in cfg["task_ids"]:
        env = None
        result = {"task_id": task_id, "status": "error"}
        folder = out / f"task_{task_id}"
        folder.mkdir()
        try:
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            result.update({
                "task": task._asdict(),
                "eval_init_state_ids": cfg["eval_init_state_ids"],
                "init_state_count": len(states),
                "bddl_sha256": sha256(Path(suite.get_task_bddl_file_path(task_id))),
                "eval_state_sha256": {
                    str(i): hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest()
                    for i in cfg["eval_init_state_ids"]
                },
            })
            env, instruction = get_libero_env(task, "WLA", resolution=512)

            def reset_debug():
                env.seed(0)
                env.reset()
                obs = env.set_init_state(states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action("WLA"))
                return obs

            obs = reset_debug()
            repeated = reset_debug()
            assert np.allclose(obs["robot0_eef_pos"], repeated["robot0_eef_pos"], atol=1e-7)
            assert not env.check_success(), "Initial state unexpectedly satisfies task."
            result["reset_repeatable"] = True
            result["initial_success"] = False
            state = np.concatenate((
                obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            ))
            assert state.shape == (8,) and np.isfinite(state).all()
            images = get_libero_image(None, obs, (256, 256))
            result["state_shape"] = list(state.shape)
            result["wla_image_shapes"] = [list(frame.shape) for frame in images[0]]
            assert result["wla_image_shapes"] == [[3, 256, 256]] * 3
            for key, name in (("agentview_image", "front"), ("robot0_eye_in_hand_image", "wrist")):
                image = obs[key][::-1, ::-1].copy()
                assert image.shape == (512, 512, 3) and image.std() > 1
                Image.fromarray(image).save(folder / f"{name}.png")
            baseline_obs = reset_debug()
            baseline_obs, _, _, _ = env.step(get_libero_dummy_action("WLA"))
            baseline_position = baseline_obs["robot0_eef_pos"].copy()
            result["axis_probes"] = []
            for axis in range(3):
                for sign in (-1, 1):
                    reset_debug()
                    action = np.array(get_libero_dummy_action("WLA"), dtype=float)
                    action[axis] = 0.2 * sign
                    moved, _, done, _ = env.step(action)
                    effect = moved["robot0_eef_pos"] - baseline_position
                    assert np.isfinite(effect).all()
                    assert effect[axis] * sign > 1e-6, (axis, sign, effect.tolist())
                    result["axis_probes"].append({
                        "axis": axis, "sign": sign, "action": action.tolist(),
                        "delta_relative_to_noop_m": effect.tolist(),
                        "done": bool(done), "env_success": bool(env.check_success()),
                    })
            reset_debug()
            gripper_widths = []
            for grip in (-1.0, 1.0, -1.0):
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, grip])
                gripper_widths.append(float(np.sum(np.abs(obs["robot0_gripper_qpos"]))))
            assert gripper_widths[0] > gripper_widths[1] + 0.005
            assert gripper_widths[2] > gripper_widths[1] + 0.005
            result["gripper_open_close_open_widths_m"] = gripper_widths
            result["status"] = "passed"
        except Exception as exc:
            result["error"] = str(exc)
            (folder / "error.txt").write_text(traceback.format_exc())
        finally:
            if env is not None:
                env.close()
            (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
    report = {
        "scope": "environment_only_no_wla_inference_no_benchmark_claim",
        "passed": all(r["status"] == "passed" for r in results),
        "tasks": results,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "artifacts": str(out)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
