"""One native WLA inference on LIBERO debug state 0; execute no predicted action."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
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

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
from native_contract import env_actions, load_pilot
from run_libero_eval import set_seed_everywhere
from wla_utils import WLA0


def interrupted(signum, frame):
    raise InterruptedError(f"Interrupted by signal {signum}")


def main() -> int:
    started = time.monotonic()
    base = ROOT / "artifacts/wla-stage1"
    out = Path(tempfile.mkdtemp(prefix="native-inference-", dir=base))
    cfg = load_pilot(CODE / "configs/pilot.json")
    assert cfg["debug_init_state_id"] == 0 and cfg["environment_seed"] == 0
    report = {
        "status": "preflight", "scope": "one_native_inference_not_an_episode",
        "config": cfg, "initialization_steps": 0, "task_control_steps": 0,
        "predicted_actions_executed": 0,
    }
    env = None
    policy_lock = None

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "report.json")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        policy_lock = (base / ".wla-policy.lock").open("a+")
        fcntl.flock(policy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("RUA_GPU", "0")
        assert subprocess.check_output(["git", "-C", str(WLA), "rev-parse", "HEAD"], text=True).strip() == cfg["wla_commit"]
        lock = json.loads((ROOT / "models/wla-models.lock.json").read_text())
        assert lock["status"] == "download_complete"
        receipts = sorted(base.glob("models-offline-check-*/report.json"), key=lambda p: p.stat().st_mtime)
        assert receipts, "Run verify-models before GPU inference."
        receipt = json.loads(receipts[-1].read_text())
        assert receipt["status"] == "passed", "Latest offline component verification did not pass."
        assert [(m["repo_id"], m["revision"]) for m in receipt["models"]] == [
            (m["repo_id"], m["revision"]) for m in lock["models"]
        ]
        report["offline_check"] = str(receipts[-1])
        report["model_lock_sha256"] = hashlib.sha256((ROOT / "models/wla-models.lock.json").read_bytes()).hexdigest()
        free_mib, total_mib = [
            int(v.strip()) for v in subprocess.check_output(
                ["nvidia-smi", "-i", os.environ.get("RUA_GPU", "0"), "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"], text=True
            ).strip().split(",")
        ]
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        cpu_available_kib = int(memory["MemAvailable"].split()[0])
        report["resources"] = {
            "gpu0_free_mib_before_load": free_mib, "gpu0_total_mib": total_mib,
            "minimum_free_gpu_mib": 24 * 1024, "cpu_available_kib": cpu_available_kib,
            "torch_allocator_cap_gib": 12,
        }
        if free_mib < 24 * 1024 or cpu_available_kib < 64 * 1024**2:
            raise RuntimeError("Insufficient headroom for a safe first load; no model loaded.")
        set_seed_everywhere(cfg["model_seed"])
        # Resource bound only: no change to WLA dtype, checkpoint, or inference settings.
        torch.cuda.set_per_process_memory_fraction((12 * 1024) / total_mib, device=0)
        torch.cuda.reset_peak_memory_stats()
        report["status"] = "loading_native_wla"
        save()
        load_started = time.monotonic()
        policy = WLA0(cfg["model_id"], None, str(WLA / "configs/norm_stats.json"), cfg["unnorm_key"])
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.monotonic() - load_started
        report["status"] = "preparing_debug_observation"
        save()
        suite = benchmark.get_benchmark_dict()[cfg["suite"]](task_order_index=cfg["task_order_index"])
        task = suite.get_task(0)
        initial_states = suite.get_task_init_states(0)
        env, instruction = get_libero_env(task, "wla", resolution=cfg["render_resolution"])
        env.reset()
        obs = env.set_init_state(initial_states[cfg["debug_init_state_id"]])
        for _ in range(cfg["settle_steps"]):
            obs, _, _, _ = env.step(get_libero_dummy_action("wla"))
            report["initialization_steps"] += 1
        assert not env.check_success()
        images = get_libero_image(None, obs, tuple(cfg["image_size"]))
        state = np.concatenate((
            obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
        ))
        assert state.shape == (8,) and np.isfinite(state).all()
        report.update(
            instruction=instruction, task=task._asdict(), init_state_id=0,
            state=state.tolist(), history_observation=None,
            input_image_shapes=[list(image.shape) for image in images[0]],
        )
        np.savez_compressed(out / "input.npz", state=state,
                            images=np.stack([image.numpy() for image in images[0]]))
        for key, name in (("agentview_image", "front"), ("robot0_eye_in_hand_image", "wrist")):
            Image.fromarray(obs[key][::-1, ::-1].copy()).save(out / f"{name}.png")
        report["status"] = "native_inference"
        save()
        infer_started = time.monotonic()
        actions = policy.inference({"full_image": images, "state": state}, instruction)
        torch.cuda.synchronize()
        report["inference_seconds"] = time.monotonic() - infer_started
        assert isinstance(actions, torch.Tensor) and tuple(actions.shape) == (8, 7)
        assert torch.isfinite(actions).all()
        converted = actions.clone()
        converted[..., -1] = torch.where(converted[..., -1] >= 0.5, -1.0, 1.0)
        raw_list, converted_list = actions.detach().cpu().tolist(), converted.detach().cpu().tolist()
        np.testing.assert_array_equal(env_actions(np.array(raw_list)), np.array(converted_list))
        report.update(status="passed", action_shape=list(actions.shape), raw_actions=raw_list,
                      environment_actions=converted_list, official_success_after_initialization=bool(env.check_success()))
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (out / "error.txt").write_text(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["close_error"] = str(exc)
                report["status"] = "failed"
        if torch.cuda.is_initialized():
            report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        save()
        if policy_lock is not None:
            policy_lock.close()
        print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
