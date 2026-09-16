"""Deadline-bounded framing for our private, inherited Unix socket only.

Pickle is deliberately limited to a socketpair whose peer we spawn, not a network
service or user-supplied endpoint. It carries numpy arrays and native RNG state.
"""
import pickle
import struct
import time

MAX_FRAME_BYTES = 64 * 1024**2


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("WLA IPC deadline expired")
    return seconds


def send_frame(sock, value, deadline):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("WLA IPC frame exceeds 64 MiB.")
    sock.settimeout(_remaining(deadline))
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def receive_frame(sock, deadline):
    def read_exact(size):
        result = bytearray()
        while len(result) < size:
            sock.settimeout(_remaining(deadline))
            data = sock.recv(min(size - len(result), 1024 * 1024))
            if not data:
                raise EOFError("WLA worker socket closed.")
            result.extend(data)
        return bytes(result)
    size, = struct.unpack("!Q", read_exact(8))
    if size > MAX_FRAME_BYTES:
        raise ValueError("WLA IPC frame exceeds 64 MiB.")
    return pickle.loads(read_exact(size))
