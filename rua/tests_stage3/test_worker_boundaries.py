import socket
import struct
import threading
import time

import numpy as np
import pytest

from libero_harness.resources import admission, memory_headroom
from libero_harness.worker_protocol import MAX_FRAME_BYTES, receive_frame, send_frame


def test_private_wire_preserves_numpy_and_rng_like_tuple():
    left, right = socket.socketpair()
    payload = dict(image=np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3),
                   rng=(7, np.array([1, 2, 3]), None))
    errors = []
    def send():
        try:
            send_frame(left, payload, time.monotonic() + 3)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=send)
    thread.start()
    try:
        received = receive_frame(right, time.monotonic() + 3)
        np.testing.assert_array_equal(payload["image"], received["image"])
        np.testing.assert_array_equal(payload["rng"][1], received["rng"][1])
    finally:
        left.close()
        right.close()
        thread.join(timeout=3)
    assert not thread.is_alive() and not errors


def test_wire_timeout_is_bounded():
    left, right = socket.socketpair()
    try:
        with pytest.raises(TimeoutError):
            receive_frame(right, time.monotonic() + .03)
    finally:
        left.close()
        right.close()


def test_wire_oversize_and_peer_exit():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!Q", MAX_FRAME_BYTES + 1))
        with pytest.raises(ValueError, match="64 MiB"):
            receive_frame(right, time.monotonic() + 1)
        left.close()
        with pytest.raises(EOFError):
            receive_frame(right, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_cgroup_v1_headroom_distinguishes_raw_and_reclaim_estimate(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.limit_in_bytes").write_text(str(192 * 1024**3))
    (mem / "memory.usage_in_bytes").write_text(str(150 * 1024**3))
    (mem / "memory.stat").write_text(
        f"total_inactive_file {46 * 1024**3}\nhierarchical_memory_limit {192 * 1024**3}\n")
    proc = tmp_path / "meminfo"
    proc.write_text(f"MemAvailable: {130 * 1024**2} kB\n")
    result = memory_headroom(tmp_path, proc)
    assert result["hard_headroom_bytes"] == 42 * 1024**3
    assert result["effective_headroom_bytes"] == 88 * 1024**3
    assert admission(memory=result, gpu=dict(free_mib=35000, total_mib=143073))


@pytest.mark.parametrize("hard,effective,free", [(31, 90, 35000), (40, 63, 35000), (40, 80, 24000)])
def test_admission_rejects_each_insufficient_resource(hard, effective, free):
    with pytest.raises(RuntimeError, match="no model loaded"):
        admission(memory=dict(hard_headroom_bytes=hard * 1024**3,
                              effective_headroom_bytes=effective * 1024**3),
                  gpu=dict(free_mib=free, total_mib=143073))


def test_missing_cgroup_is_not_treated_as_infinite_memory(tmp_path):
    proc = tmp_path / "meminfo"
    proc.write_text("MemAvailable: 999999999 kB\n")
    with pytest.raises(RuntimeError, match="Cannot establish cgroup"):
        memory_headroom(tmp_path, proc)
