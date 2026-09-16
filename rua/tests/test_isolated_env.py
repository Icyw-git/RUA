from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isolated_env import IsolatedLiberoEnv, get_cpu_rng, set_cpu_rng


def test_cpu_rng_roundtrip_preserves_shared_native_sequence():
    before = get_cpu_rng()
    try:
        random.seed(7)
        np.random.seed(0)
        torch.manual_seed(7)
        checkpoint = get_cpu_rng()
        expected = (random.random(), np.random.rand(8), torch.rand(8))
        set_cpu_rng(checkpoint)
        actual = (random.random(), np.random.rand(8), torch.rand(8))
        assert expected[0] == actual[0]
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        assert not torch.cuda.is_initialized()
    finally:
        set_cpu_rng(before)


def test_closed_worker_refuses_control():
    env = IsolatedLiberoEnv.__new__(IsolatedLiberoEnv)
    env.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        env.step([0] * 7)


def test_worker_timeout_does_not_invent_an_observation():
    class Connection:
        def send(self, payload):
            self.payload = payload

        def poll(self, timeout):
            return False

    env = IsolatedLiberoEnv.__new__(IsolatedLiberoEnv)
    env.closed = False
    env.connection = Connection()
    stopped = []
    env._stop = lambda: stopped.append(True)
    with pytest.raises(RuntimeError, match="environment_worker_timeout"):
        env.step([0] * 7)
    assert env.connection.payload[0] == "step"
    assert stopped == [True]
