"""Pure resource-guard tests: no GPU, model, simulator, or process signals."""
from scripts.supervise_harness import GPU0_UUID, stop_reason


def sample(free=20000, oom_kill=0):
    return dict(gpus=[dict(index=0, uuid=GPU0_UUID or "GPU-test-0", free_mib=free)],
                cgroup_limit_in_bytes=192 * 1024**3,
                cgroup_usage_in_bytes=140 * 1024**3,
                cgroup_oom=dict(under_oom=0, oom_kill=oom_kill))


def test_safe_headroom():
    assert stop_reason(sample(), sample()) is None


def test_gpu_reserve():
    assert stop_reason(sample(8191), sample()) == "gpu_free_memory_below_reserve"
    assert stop_reason(sample(8192), sample()) is None


def test_new_oom():
    assert stop_reason(sample(oom_kill=1), sample()) == "cgroup_oom"
    assert stop_reason(sample(oom_kill=1), sample(oom_kill=1)) is None


def test_cgroup_reserve():
    current = sample()
    current["cgroup_usage_in_bytes"] = 190 * 1024**3
    assert stop_reason(current, sample()) == "cgroup_memory_below_reserve"


def test_unknown_gpu_telemetry():
    assert stop_reason(dict(gpu_error="TimeoutExpired"), sample()) == "gpu_telemetry_unavailable"
    assert stop_reason(dict(gpus=[]), sample()) == "gpu_identity_mismatch"


def test_changed_gpu_identity():
    current = sample()
    current["gpus"][0]["uuid"] = "different"
    assert stop_reason(current, sample()) == "gpu_identity_mismatch"
