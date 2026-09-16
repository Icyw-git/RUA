"""Native WLA inference only: no environment object and no agent credentials."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

from .worker_protocol import receive_frame, send_frame


def main():
    descriptor, parent_pid = int(sys.argv[1]), int(sys.argv[2])
    directory = Path(sys.argv[3])
    # Only terminate this worker if its parent dies. No unrelated GPU processes.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        return 1
    channel = socket.socket(fileno=descriptor)
    try:
        initial = receive_frame(channel, time.monotonic() + 30)
        if initial.get("op") != "initialize":
            raise ValueError("Expected WLA initialization.")
        pilot = initial["pilot"]
        from .resources import admission, memory_headroom
        resources = admission()
        import numpy as np
        import torch
        from .paths import ROOT, CODE, save_json
        from eval_native import preflight
        from isolated_env import get_cpu_rng, set_cpu_rng
        from libero_utils import get_libero_image, quat2axisangle
        from run_libero_eval import set_seed_everywhere
        from wla_utils import WLA0
        native_preflight = preflight(pilot, ROOT / "artifacts/wla-stage1", directory)
        save_json(directory / "preflight.json", dict(resources=resources, native=native_preflight))
        set_seed_everywhere(pilot["model_seed"])  # Exactly once, BEFORE model loading.
        torch.cuda.set_per_process_memory_fraction(
            (12 * 1024) / resources["gpu0"]["total_mib"], device=0)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        policy = WLA0(pilot["model_id"], None, str(CODE.parent / "configs/norm_stats.json"),
                      pilot["unnorm_key"])
        torch.cuda.synchronize()
        metadata = dict(pid=os.getpid(), resources=resources,
                        model_load_seconds=time.monotonic() - started,
                        native_preflight=native_preflight)
        save_json(directory / "ready.json", metadata)
        send_frame(channel, dict(status="ready", cpu_rng=get_cpu_rng(), **metadata),
                   time.monotonic() + 30)
        expected_call = 0
        while True:
            message = receive_frame(channel, time.monotonic() + 960)
            if message.get("op") != "predict" or message["call_id"] != expected_call:
                raise ValueError("Unknown or non-sequential WLA request.")
            expected_call += 1
            if memory_headroom()["hard_headroom_bytes"] < 8 * 1024**3:
                raise RuntimeError("Cgroup reserve below 8 GiB; stop our WLA worker.")
            request = message["request"]
            set_cpu_rng(message["cpu_rng"])
            obs = request.current
            started = time.monotonic()
            images = get_libero_image(request.history, obs, tuple(pilot["image_size"]))
            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]))
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Invalid native 8D WLA state.")
            call_id = message["call_id"]
            np.savez_compressed(directory / f"input-{call_id:03d}.npz", state=state,
                                images=np.stack([image.numpy() for image in images[0]]))
            raw = policy.inference({"full_image": images, "state": state}, message["instruction"])
            torch.cuda.synchronize()
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (8, 7) or not torch.isfinite(raw).all():
                raise ValueError("Invalid native WLA output.")
            raw_cpu = raw.detach().cpu().numpy()
            np.save(directory / f"actions-{call_id:03d}.npy", raw_cpu)
            record = dict(call_id=call_id, sequence=request.sequence, task_step=request.task_step,
                          instruction=message["instruction"], elapsed_seconds=time.monotonic() - started,
                          input_image_shapes=[list(img.shape) for img in images[0]],
                          state_shape=list(state.shape), raw_shape=list(raw_cpu.shape),
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
            save_json(directory / f"prediction-{call_id:03d}.json", record)
            send_frame(channel, dict(status="prediction", cpu_rng=get_cpu_rng(),
                                     raw_actions=raw_cpu, **record), time.monotonic() + 30)
    except EOFError:
        return 0
    except BaseException as exc:
        traceback.print_exc()
        try:
            send_frame(channel, dict(status="error", error=f"{type(exc).__name__}: {exc}"),
                       time.monotonic() + 5)
        except BaseException:
            pass
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
