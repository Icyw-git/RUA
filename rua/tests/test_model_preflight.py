import json
import hashlib
import fnmatch
from pathlib import Path
import sys
from types import SimpleNamespace

import requests
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_models


def fake_files(repo):
    config = dict(prepare_models.SOURCES["expected_dependency_ids"])
    weight = ("vae/diffusion_pytorch_model.safetensors"
              if "Sana_" in repo else "model.safetensors")
    files = {weight: b"test weights"}
    if repo == prepare_models.REPOS[0]:
        files["config.json"] = json.dumps(config).encode()
    return files


class FakeAPI:
    def __init__(self, token):
        assert token is False

    def model_info(self, repo, *args, **kwargs):
        return SimpleNamespace(
            sha=str(prepare_models.REPOS.index(repo) + 1) * 40,
            siblings=[
                SimpleNamespace(
                    rfilename=name, size=len(data),
                    lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()),
                )
                for name, data in fake_files(repo).items()
            ],
        )


def setup_root(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(prepare_models, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_models, "PINNED",
                        {repo: str(i + 1) * 40 for i, repo in enumerate(prepare_models.REPOS)})
    monkeypatch.setattr(
        prepare_models, "snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must not download weights")),
    )
    monkeypatch.setattr(sys, "argv", ["prepare_models.py", "--download"])


def report(tmp_path):
    path = next((tmp_path / "artifacts/wla-stage1").glob("model-preflight-*/report.json"))
    return json.loads(path.read_text())


@pytest.mark.parametrize("confirmed", [False, True])
def test_tls_failure_never_downloads(tmp_path, monkeypatch, confirmed):
    setup_root(tmp_path, monkeypatch)
    if confirmed:
        monkeypatch.setattr(sys, "argv", ["prepare_models.py", "--download", "--capacity-confirmed-by-user"])

    class FailingAPI:
        def __init__(self, token):
            assert token is False

        def model_info(self, *args, **kwargs):
            raise requests.exceptions.SSLError("untrusted certificate")

    monkeypatch.setattr(prepare_models, "HfApi", FailingAPI)
    assert prepare_models.main() == 1
    result = report(tmp_path)
    assert result["status"] == "blocked"
    assert result["error_type"] == "SSLError"
    assert not (tmp_path / "models/wla-models.lock.json").exists()


def test_unknown_capacity_never_downloads(tmp_path, monkeypatch):
    setup_root(tmp_path, monkeypatch)

    monkeypatch.setattr(prepare_models, "HfApi", FakeAPI)
    monkeypatch.setattr(prepare_models.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert prepare_models.main() == 1
    result = report(tmp_path)
    assert not result["capacity_sufficient"]
    assert "Capacity not confirmed" in result["error"]
    assert not (tmp_path / "models/wla-models.lock.json").exists()


def test_user_confirmed_capacity_with_zero_statvfs_downloads_pinned_revisions(tmp_path, monkeypatch):
    setup_root(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prepare_models.py", "--download", "--capacity-confirmed-by-user"])
    monkeypatch.setattr(prepare_models, "HfApi", FakeAPI)
    monkeypatch.setattr(prepare_models.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    calls = []

    def download(repo, revision, cache_dir, token, endpoint, allow_patterns, max_workers):
        assert revision == str(prepare_models.REPOS.index(repo) + 1) * 40
        assert endpoint.startswith("https://") and token is False
        calls.append(repo)
        target = cache_dir / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        for name, data in fake_files(repo).items():
            assert name in allow_patterns
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes(data)
        return str(target)

    monkeypatch.setattr(prepare_models, "snapshot_download", download)
    assert prepare_models.main() == 0
    result = report(tmp_path)
    assert calls == list(prepare_models.REPOS)
    assert result["status"] == "download_complete"
    assert result["statvfs_free_bytes"] == 0
    assert result["capacity_basis"] == "user_confirmed_sufficient_quota"
    assert "externally_verified_free_bytes" not in result
    for model in result["models"]:
        assert model["files_verified"]
        assert (Path(model["snapshot_path"]).parent.parent / "refs/main").read_text() == model["revision"]
    assert (tmp_path / "models/wla-models.lock.json").exists()


def test_corrupted_weight_is_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        prepare_models.verify_file(path, {"size": 5, "lfs_sha256": hashlib.sha256(b"right").hexdigest()})


def test_sana_selection_excludes_unused_encoder_and_duplicate_variant():
    patterns = prepare_models.allowed_patterns(prepare_models.REPOS[2])
    for path in ("text_encoder/model.safetensors", "tokenizer/tokenizer.json",
                 "vae/diffusion_pytorch_model.fp16.safetensors"):
        assert not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    assert any(fnmatch.fnmatch("vae/diffusion_pytorch_model.safetensors", pattern) for pattern in patterns)


def test_selected_dependency_map_is_accepted():
    prepare_models.validate_dependencies(dict(prepare_models.SOURCES["expected_dependency_ids"]))


def test_wrong_dependency_role_is_rejected():
    config = dict(prepare_models.SOURCES["expected_dependency_ids"])
    config["vae_id"] = config["mllm_id"]
    with pytest.raises(RuntimeError, match="vae_id"):
        prepare_models.validate_dependencies(config)


def test_missing_dependency_is_rejected():
    config = dict(prepare_models.SOURCES["expected_dependency_ids"])
    del config["scheduler_id"]
    with pytest.raises(RuntimeError, match="scheduler_id"):
        prepare_models.validate_dependencies(config)
