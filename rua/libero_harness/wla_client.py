"""One prediction-only subprocess. The parent keeps all simulation ownership."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from .handoff import HandoffError
from .worker_protocol import receive_frame, send_frame


class WLAWorkerClient:
    def __init__(self, directory, pilot, *, load_timeout=300):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.closed = False
        self.sequence = 0
        self.records = []
        self._lock = threading.Lock()
        self.process = None
        self._log = (self.directory / "worker.log").open("wb")
        self.socket, child = socket.socketpair()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("RUA_GPU", "0"), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "libero_harness.wla_worker", str(child.fileno()),
                 str(os.getpid()), str(self.directory)],
                pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=subprocess.STDOUT,
            )
            child.close()
            deadline = time.monotonic() + load_timeout
            send_frame(self.socket, dict(op="initialize", pilot=pilot), deadline)
            self.startup = receive_frame(self.socket, deadline)
            self._check(self.startup, "ready")
            # Continue the same logical CPU RNG stream after native model loading.
            from isolated_env import set_cpu_rng
            set_cpu_rng(self.startup.pop("cpu_rng"))
        except BaseException:
            child.close()
            self.close()
            raise

    @staticmethod
    def _check(reply, expected):
        if not isinstance(reply, dict) or reply.get("status") != expected:
            raise HandoffError(f"WLA worker failed: {reply.get('error') if isinstance(reply, dict) else 'bad reply'}")

    def predict(self, request, instruction, *, timeout_seconds):
        if not self._lock.acquire(blocking=False):
            raise HandoffError("Concurrent WLA prediction is forbidden.")
        try:
            if self.closed or self.process.poll() is not None:
                raise HandoffError("WLA worker is not alive.")
            from isolated_env import get_cpu_rng, set_cpu_rng
            call_id = self.sequence
            self.sequence += 1
            deadline = time.monotonic() + min(120, timeout_seconds)
            send_frame(self.socket, dict(
                op="predict", call_id=call_id, request=request,
                instruction=instruction, cpu_rng=get_cpu_rng()), deadline)
            reply = receive_frame(self.socket, deadline)
            self._check(reply, "prediction")
            if reply["call_id"] != call_id or reply["sequence"] != request.sequence:
                raise HandoffError("Mismatched WLA reply; no action may execute.")
            set_cpu_rng(reply.pop("cpu_rng"))
            actions = reply.pop("raw_actions")
            self.records.append(reply)
            return actions
        except BaseException:
            # A late reply is never reused for a later prediction.
            self.close()
            raise
        finally:
            self._lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
            self.exitcode = self.process.returncode
        self._log.close()
