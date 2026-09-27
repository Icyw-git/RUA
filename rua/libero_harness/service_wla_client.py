"""Prediction-only adapter for the already running loopback WLA service."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .handoff import HandoffError


RAW_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos")
MAX_BYTES = 12 * 1024 * 1024


def _encode_observation(obs):
    encoded = {}
    for key in RAW_KEYS:
        value = np.asarray(obs[key])
        if key.endswith("image"):
            if value.dtype != np.uint8:
                raise ValueError("WLA service expects native uint8 RGB images")
            encoded[key] = {"shape": list(value.shape),
                            "data": base64.b64encode(np.ascontiguousarray(value).tobytes()).decode()}
        else:
            encoded[key] = value.tolist()
    return encoded


def _rpc(port, payload, timeout):
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("WLA service request exceeds wire limit")
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(struct.pack("!I", len(data)) + data)

        def read_exact(size):
            blocks = bytearray()
            while len(blocks) < size:
                block = connection.recv(min(size - len(blocks), 65536))
                if not block:
                    raise EOFError("WLA service closed the connection")
                blocks.extend(block)
            return blocks

        size = struct.unpack("!I", read_exact(4))[0]
        if not 0 < size <= MAX_BYTES:
            raise ValueError("Invalid WLA service reply size")
        reply = json.loads(read_exact(size))
    if reply.get("status") == "error":
        raise HandoffError(reply["error"])
    return reply


class ServiceWLAClient:
    """Expose the NativeChunkExecutor predictor contract without owning the environment."""

    def __init__(self, directory, pilot, *, port):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.port = port
        self.records = []
        self.sequence = 0
        self.faulted = False
        self.closed = False
        self.exitcode = None
        self._lock = threading.Lock()
        self.session = None

        health = _rpc(port, {"op": "health"}, 5)
        norm_path = Path(__file__).resolve().parents[2] / "configs/norm_stats.json"
        revisions_path = Path(__file__).resolve().parents[1] / "configs/model-revisions.json"
        revisions = json.loads(revisions_path.read_text())
        expected_revision = next(model["revision"] for model in revisions["models"]
                                 if model["repo_id"] == pilot["model_id"])
        if (health.get("status") != "ok" or not health.get("healthy")
                or health.get("protocol") != 1
                or health.get("action_contract") != "raw_unnormalized_8x7"
                or health.get("image_size") != pilot["image_size"]
                or health.get("model_seed") != pilot["model_seed"]
                or health.get("model_revision") != expected_revision
                or health.get("norm_sha256") != hashlib.sha256(norm_path.read_bytes()).hexdigest()):
            raise HandoffError("WLA service model, normalization, or protocol differs from this run")
        reply = _rpc(port, {"op": "start_session", "seed": pilot["model_seed"]}, 5)
        if reply.get("status") != "ok" or not reply.get("session"):
            raise HandoffError("WLA service did not start a session")
        self.session = reply["session"]
        self.startup = {"status": "ready", "backend": "shared_service", "port": port,
                        "health": health}

    def predict(self, request, instruction, *, timeout_seconds):
        if not self._lock.acquire(blocking=False):
            raise HandoffError("Concurrent WLA prediction is forbidden")
        try:
            if self.closed or self.faulted or timeout_seconds <= 0:
                raise HandoffError("WLA service session unavailable")
            call_id = self.sequence
            request_id = secrets.token_hex(16)
            timeout = min(timeout_seconds, 120)
            payload = {"op": "predict", "session": self.session, "call_id": call_id,
                       "request_id": request_id, "sequence": request.sequence,
                       "task_step": request.task_step, "instruction": instruction,
                       "timeout_ms": max(1, int(timeout * 1000)),
                       "current": _encode_observation(request.current),
                       "history": None if request.history is None
                       else _encode_observation(request.history)}
            started = time.monotonic()
            reply = _rpc(self.port, payload, timeout)
            if (reply.get("status") != "prediction" or reply.get("call_id") != call_id
                    or reply.get("sequence") != request.sequence
                    or reply.get("request_id") != request_id
                    or reply.get("action_contract") != "raw_unnormalized_8x7"):
                raise HandoffError("Mismatched WLA service prediction")
            actions = np.asarray(reply.pop("raw_actions"), dtype=np.float32)
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise HandoffError("Invalid WLA service action chunk")

            from libero_utils import get_libero_image, quat2axisangle
            observation = request.current
            images = get_libero_image(request.history, observation, (256, 256))
            state = np.concatenate((observation["robot0_eef_pos"],
                                    quat2axisangle(observation["robot0_eef_quat"].copy()),
                                    observation["robot0_gripper_qpos"]))
            np.savez_compressed(self.directory / f"input-{call_id:03d}.npz", state=state,
                                images=np.stack([image.numpy() for image in images[0]]))
            np.save(self.directory / f"actions-{call_id:03d}.npy", actions)
            record = {"call_id": call_id, "sequence": request.sequence,
                      "task_step": request.task_step, "instruction": instruction,
                      "elapsed_seconds": time.monotonic() - started,
                      "state_shape": list(state.shape), "raw_shape": list(actions.shape),
                      "service_port": self.port, **reply}
            (self.directory / f"prediction-{call_id:03d}.json").write_text(
                json.dumps(record, indent=2) + "\n")
            self.records.append(record)
            self.sequence += 1
            return actions
        except BaseException:
            self.faulted = True
            raise
        finally:
            self._lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.session is not None:
                _rpc(self.port, {"op": "close_session", "session": self.session}, 5)
            self.exitcode = 0
        except (OSError, EOFError, HandoffError):
            self.exitcode = 1
        finally:
            self.session = None
