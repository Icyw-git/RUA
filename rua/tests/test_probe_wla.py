import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_import_does_not_initialize_cuda():
    import probe_wla

    assert not probe_wla.torch.cuda.is_initialized()


def test_insufficient_gpu_headroom_loads_no_model(tmp_path, monkeypatch):
    import probe_wla

    base = tmp_path / "artifacts/wla-stage1"
    receipt_dir = base / "models-offline-check-test"
    receipt_dir.mkdir(parents=True)
    (tmp_path / "models").mkdir()
    models = [{"repo_id": "SJTU-DENG-Lab/wla_libero_all_image_action", "revision": "a" * 40}]
    (tmp_path / "models/wla-models.lock.json").write_text(
        json.dumps({"status": "download_complete", "models": models})
    )
    (receipt_dir / "report.json").write_text(json.dumps({"status": "passed", "models": models}))
    monkeypatch.setattr(probe_wla, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe_wla.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        probe_wla.subprocess, "check_output",
        lambda args, **kwargs: ("5662058f3ead8a1b4fec9865acce21dd2f5efbf1\n"
                               if args[0] == "git" else "1024, 143771\n"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Low-resource preflight must not initialize CUDA or load WLA.")

    monkeypatch.setattr(probe_wla, "WLA0", forbidden)
    monkeypatch.setattr(probe_wla, "set_seed_everywhere", forbidden)
    monkeypatch.setattr(probe_wla.torch.cuda, "set_per_process_memory_fraction", forbidden)
    assert probe_wla.main() == 1
    report_path = next(base.glob("native-inference-*/report.json"))
    result = json.loads(report_path.read_text())
    assert "Insufficient headroom" in result["error"]
    assert result["predicted_actions_executed"] == 0
    assert not probe_wla.torch.cuda.is_initialized()
