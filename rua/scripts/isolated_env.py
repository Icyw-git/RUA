"""Own LIBERO/EGL in a spawn worker, retaining native calls and logical CPU RNG."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import random
import signal
import traceback

import numpy as np
import torch


def get_cpu_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().copy())


def set_cpu_rng(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(torch.from_numpy(states[2].copy()))


def _worker(connection, cfg, task_id, parent_pid):
    # If the evaluator is killed, do not leave an orphan graphics controller.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        return
    from bootstrap import configure
    configure()
    from libero.libero import benchmark
    from libero_utils import get_libero_env
    env = None
    try:
        while True:
            name, args, rng = connection.recv()
            set_cpu_rng(rng)
            try:
                if name == "initialize":
                    suite = benchmark.get_benchmark_dict()[cfg["suite"]](task_order_index=cfg["task_order_index"])
                    task = suite.get_task(task_id)
                    env, instruction = get_libero_env(task, "wla", resolution=cfg["render_resolution"])
                    value = instruction
                elif name in ("reset", "set_init_state", "step", "check_success", "close"):
                    value = getattr(env, name)(*args)
                    if name == "close":
                        env = None
                else:
                    raise ValueError(f"Unknown native environment operation: {name}")
                if torch.cuda.is_initialized():
                    raise RuntimeError("EGL worker must not initialize PyTorch CUDA.")
                connection.send(("ok", value, get_cpu_rng()))
                if name == "close":
                    break
            except BaseException:
                connection.send(("error", traceback.format_exc(), get_cpu_rng()))
                break
    except EOFError:
        pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class IsolatedLiberoEnv:
    """Synchronous single-owner RPC; no policy calls or retries in the worker."""
    def __init__(self, cfg, task_id):
        context = mp.get_context("spawn")  # Never fork after CUDA initialization.
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=_worker, args=(child_connection, cfg, task_id, os.getpid()))
        self.process.start()
        child_connection.close()
        self.closed = False
        try:
            self.instruction = self._call("initialize")
        except BaseException:
            self._stop()
            raise

    def _call(self, name, *args):
        if self.closed:
            raise RuntimeError("Environment worker is closed.")
        try:
            self.connection.send((name, args, get_cpu_rng()))
            if not self.connection.poll(120):
                raise RuntimeError(f"environment_worker_timeout:{name}")
            status, value, rng = self.connection.recv()
            set_cpu_rng(rng)
            if status != "ok":
                raise RuntimeError(f"Environment worker failed during {name}:\n{value}")
            return value
        except BaseException:
            # An interrupted RPC may still have a reply in flight. Terminate this
            # owned worker; never mistake that reply for a later close/step reply.
            self._stop()
            raise

    def reset(self):
        return self._call("reset")

    def set_init_state(self, state):
        return self._call("set_init_state", state)

    def step(self, action):
        return self._call("step", action)

    def check_success(self):
        return self._call("check_success")

    def _stop(self):
        self.closed = True
        self.connection.close()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)

    def close(self):
        if self.closed:
            return
        try:
            if self.process.is_alive():
                self._call("close")
        finally:
            self._stop()
