"""Export-only boundaries: no inference, credentials, or fabricated admission."""
import hashlib
import json
import sys
import types
from contextlib import nullcontext

import pytest

from libero_harness import paths, portable_setup, portable_run, resources
from libero_harness.budget import Budget
from libero_harness.claude import ClaudeClient
from scripts import supervise_harness as supervisor


def test_private_configuration_overlay(tmp_path, monkeypatch):
    local = tmp_path / "agent.json"
    local.write_text(json.dumps(dict(model="local", base_url="https://local.invalid")))
    monkeypatch.setenv("RUA_AGENT_CONFIG", str(local))
    monkeypatch.setenv("RUA_MODEL", "override")
    monkeypatch.setenv("RUA_API_BASE", "https://override.invalid")
    monkeypatch.setenv("RUA_AUTH_FILE", str(tmp_path / "private.json"))
    _, cfg = paths.configs("claude")
    assert cfg["model"] == "override" and cfg["base_url"] == "https://override.invalid"
    assert cfg["auth_file"] == str(tmp_path / "private.json")
    assert cfg["calibration_status"] != "passed"


def test_no_credentials_fails_before_request(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    budget = Budget()
    with pytest.raises(ValueError, match="no bundled credentials"):
        ClaudeClient(dict(base_url="https://provider.invalid", model="test"), budget, tmp_path)
    assert budget.requests == 0


@pytest.mark.parametrize("scheme", ["bearer", "x-api-key"])
def test_auth_header_selection(tmp_path, monkeypatch, scheme):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-placeholder")
    client = ClaudeClient(dict(auth_scheme=scheme), Budget(), tmp_path)
    headers = client._auth_headers()
    expected = ({"Authorization": "Bearer test-placeholder"} if scheme == "bearer"
                else {"x-api-key": "test-placeholder"})
    assert headers == dict(expected, **{"anthropic-version": "2023-06-01"})
    assert client.records == []


def test_auth_scheme_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-placeholder")
    with pytest.raises(ValueError, match="RUA_AUTH_SCHEME"):
        ClaudeClient(dict(auth_scheme="unexpected"), Budget(), tmp_path)


def test_calibration_signals_are_restored_without_running_physics(monkeypatch):
    import signal
    from libero_harness import calibrate
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(calibrate, "execution_lock", nullcontext)
    monkeypatch.setattr(calibrate, "_main_locked", lambda: 0)
    assert calibrate.main() == 0
    assert {sig: signal.getsignal(sig) for sig in saved} == saved
    with pytest.raises(InterruptedError):
        calibrate.calibration_interrupted(signal.SIGTERM, None)


def test_layout_is_local_idempotent_and_preserves_foreign_config(tmp_path, monkeypatch):
    code = tmp_path / "checkout/rua"
    (code.parent / "LIBERO/libero/libero/bddl_files").mkdir(parents=True)
    monkeypatch.setattr(portable_setup, "CODE", code)
    monkeypatch.setattr(portable_setup, "ROOT", tmp_path / "data")
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    target = portable_setup.layout()
    assert target == portable_setup.layout()
    assert str(tmp_path / "checkout/LIBERO") in target.read_text()
    target.write_text("assets: elsewhere\n")
    with pytest.raises(ValueError, match="differs"):
        portable_setup.layout()


def calibration_fixture(tmp_path, monkeypatch):
    code, root = tmp_path / "checkout/rua", tmp_path / "data"
    (code / "libero_harness").mkdir(parents=True)
    (code / "libero_harness/calibrate.py").write_text("# test source\n")
    receipt = root / "artifacts/rua-stage2/calibration/report.json"
    receipt.parent.mkdir(parents=True)
    report = dict(status="passed", torch_cuda_initialized=False,
                  scope="robot_camera_control_calibration_not_evaluation",
                  installation_code_root=str(code.resolve()),
                  calibration_source_sha256=hashlib.sha256(
                      (code / "libero_harness/calibrate.py").read_bytes()).hexdigest())
    report.update({measurement: 1 for measurement in portable_setup.FIELDS.values()})
    receipt.write_text(json.dumps(report))
    monkeypatch.setattr(portable_setup, "CODE", code)
    monkeypatch.setattr(portable_setup, "ROOT", root)
    monkeypatch.delenv("RUA_AGENT_CONFIG", raising=False)
    return receipt, report


def test_calibration_receipt_is_local_and_never_overwrites(tmp_path, monkeypatch):
    receipt, report = calibration_fixture(tmp_path, monkeypatch)
    target = portable_setup.accept_calibration(receipt)
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text())["calibration_sha256"] == hashlib.sha256(
        receipt.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        portable_setup.accept_calibration(receipt)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("torch_cuda_initialized", True),
    ("installation_code_root", "/other/code"), ("calibration_source_sha256", "old"),
])
def test_calibration_rejects_failed_or_foreign_receipts(tmp_path, monkeypatch, field, value):
    receipt, report = calibration_fixture(tmp_path, monkeypatch)
    report[field] = value
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        portable_setup.accept_calibration(receipt)


@pytest.mark.parametrize("mode,agent,rua_only", [
    ("wla", False, False), ("opus", True, True), ("hybrid", True, False),
])
def test_three_modes_dispatch_existing_validator(monkeypatch, mode, agent, rua_only):
    calls = []
    fake = types.ModuleType("libero_harness.validate_wla")
    fake.validate = lambda args, pilot, cfg: calls.append(args) or 0
    monkeypatch.setitem(sys.modules, "libero_harness.validate_wla", fake)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(portable_run, "configs", lambda *_: ({}, {}))
    monkeypatch.setattr(portable_run, "execution_lock", nullcontext)
    argv = ["run", "--mode", mode]
    if mode == "hybrid":
        argv += ["--handoff-receipt", "/receipt/validated-later.json"]
    monkeypatch.setattr(sys, "argv", argv)
    assert portable_run.main() == 0
    assert calls[0].agent is agent and calls[0].rua_only is rua_only
    assert calls[0].task_id == calls[0].init_id == 0


def test_hybrid_requires_admission_and_paired_flags_are_not_partial(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for args in (["--mode", "hybrid"],
                 ["--mode", "wla", "--case-id", "unfrozen"],
                 ["--mode", "opus", "--paired-manifest", "missing-other-flags"]):
        monkeypatch.setattr(sys, "argv", ["run", *args])
        with pytest.raises(SystemExit) as e:
            portable_run.main()
        assert e.value.code == 2


def test_cgroup_v2_admission_and_oom_snapshot(tmp_path, monkeypatch):
    (tmp_path / "memory.max").write_text(str(192 * 1024**3))
    (tmp_path / "memory.current").write_text(str(120 * 1024**3))
    (tmp_path / "memory.stat").write_text(f"inactive_file {10 * 1024**3}\n")
    (tmp_path / "memory.events").write_text("oom 0\noom_kill 2\n")
    proc = tmp_path / "meminfo"
    proc.write_text(f"MemAvailable: {130 * 1024**2} kB\n")
    mem = resources.memory_headroom(tmp_path, proc)
    assert mem["hard_headroom_bytes"] == 72 * 1024**3
    assert mem["effective_headroom_bytes"] == 82 * 1024**3
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **k:
                        types.SimpleNamespace(stdout="1, GPU-test-1, 32000, 0, 0\n"))
    original = supervisor.snapshot(str(tmp_path))
    assert not original.get("memory_error") and original["cgroup_oom"]["oom_kill"] == 2
    monkeypatch.setenv("RUA_GPU", "1")
    monkeypatch.setattr(supervisor, "GPU0_UUID", None)
    assert supervisor.stop_reason(original, original) is None
    (tmp_path / "memory.events").write_text("oom 1\noom_kill 3\n")
    assert supervisor.stop_reason(supervisor.snapshot(str(tmp_path)), original) == "cgroup_oom"
    (tmp_path / "memory.events").unlink()
    assert supervisor.stop_reason(supervisor.snapshot(str(tmp_path)), original) == "memory_telemetry_unavailable"


def test_gpu_selection_and_identity_are_bound(monkeypatch):
    monkeypatch.setenv("RUA_GPU", "1")
    monkeypatch.setattr(supervisor, "GPU0_UUID", None)
    initial = dict(gpus=[dict(index=1, uuid="GPU-test-1", free_mib=32000)])
    assert supervisor.stop_reason(initial, initial) is None
    changed = dict(gpus=[dict(index=1, uuid="GPU-replaced", free_mib=32000)])
    assert supervisor.stop_reason(changed, initial) == "gpu_identity_mismatch"
    monkeypatch.setenv("RUA_GPU", "GPU-test-1")
    assert supervisor.stop_reason(initial, initial) is None
    calls = []
    monkeypatch.setattr(resources.subprocess, "check_output",
                        lambda cmd, **kw: calls.append(cmd) or "32000, 48000")
    resources.admission(memory=dict(hard_headroom_bytes=40 * 1024**3,
                                    effective_headroom_bytes=80 * 1024**3))
    assert calls[0][2] == "GPU-test-1"
