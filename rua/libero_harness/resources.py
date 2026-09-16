"""Read-only GPU and effective cgroup-memory admission checks."""
from pathlib import Path
import os
import subprocess


def memory_headroom(cgroup="/sys/fs/cgroup", proc="/proc/meminfo"):
    root = Path(cgroup)
    mem = dict(line.split(":", 1) for line in Path(proc).read_text().splitlines())
    host_available = int(mem["MemAvailable"].split()[0]) * 1024
    if (root / "memory.max").exists():
        # Requires a namespaced cgroup mount rooted at the caller's cgroup.
        version = 2
        filename_limit, filename_usage = "memory.max", "memory.current"
        stats_key = "inactive_file"
        leaf = root
    elif (root / "memory/memory.limit_in_bytes").exists():
        version = 1
        filename_limit, filename_usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
        stats_key = "total_inactive_file"
        leaf = root / "memory"
    else:
        raise RuntimeError("Cannot establish cgroup memory limit; refuse model loading.")
    # Requires a container cgroup namespace rooted at the caller's memory cgroup.
    # Read hierarchical_memory_limit too for cgroup v1.
    stats = dict(line.split() for line in (leaf / "memory.stat").read_text().splitlines())
    limit_text = (leaf / filename_limit).read_text().strip()
    limit = int(limit_text) if limit_text != "max" else host_available + int(
        (leaf / filename_usage).read_text())
    if version == 1 and "hierarchical_memory_limit" in stats:
        limit = min(limit, int(stats["hierarchical_memory_limit"]))
    used = int((leaf / filename_usage).read_text())
    inactive = max(0, min(used, int(stats.get(stats_key, 0))))
    hard = max(0, limit - used)
    # This is an estimate, NOT all page cache or guaranteed allocatable memory.
    effective = min(host_available, hard + inactive)
    return dict(cgroup_version=version, limit_bytes=limit, used_bytes=used,
                hard_headroom_bytes=hard, inactive_file_bytes=inactive,
                effective_headroom_bytes=effective, host_available_bytes=host_available)


def admission(*, memory=None, gpu=None):
    memory = memory if memory is not None else memory_headroom()
    if gpu is None:
        free, total = [int(v.strip()) for v in subprocess.check_output(
            ["nvidia-smi", "-i", os.environ.get("RUA_GPU", "0"), "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(",")]
        gpu = dict(free_mib=free, total_mib=total)
    result = dict(memory=memory, gpu0=gpu,
                  minimum_gpu_free_mib=24 * 1024, minimum_hard_headroom_gib=32,
                  minimum_effective_headroom_gib=64,
                  torch_allocator_cap_gib=12)
    if (gpu["free_mib"] < 24 * 1024
            or memory["hard_headroom_bytes"] < 32 * 1024**3
            or memory["effective_headroom_bytes"] < 64 * 1024**3):
        raise RuntimeError(f"Insufficient GPU/cgroup headroom; no model loaded: {result}")
    return result
