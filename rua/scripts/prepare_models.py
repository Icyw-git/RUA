"""Prepare exact public WLA dependencies, with TLS and capacity fail-closed."""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bootstrap import CODE, ROOT, configure

configure()

from huggingface_hub import HfApi, constants, snapshot_download

SOURCE_MANIFEST = CODE / "configs/model_sources.json"
SOURCES = json.loads(SOURCE_MANIFEST.read_text())
PINNED = {m["repo_id"]: m["revision"] for m in json.loads(
    (CODE / "configs/model-revisions.json").read_text())["models"]}
REPOS = tuple(item["repo_id"] for item in SOURCES["selected_repos"])
PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja")


def allowed_patterns(repo: str) -> tuple:
    item = next(item for item in SOURCES["selected_repos"] if item["repo_id"] == repo)
    return tuple(item.get("allow_patterns", PATTERNS))


def verify_file(path: Path, metadata: dict) -> str:
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        raise RuntimeError(f"Incomplete model file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if metadata.get("lfs_sha256") and actual != metadata["lfs_sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {path.name}")
    return actual


def validate_dependencies(config: dict) -> None:
    for key, expected in SOURCES["expected_dependency_ids"].items():
        if config.get(key) != expected:
            raise RuntimeError(f"Unexpected dependency {key}; inspect the checkpoint before inference.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Only download after TLS, metadata and capacity checks pass.")
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--verified-free-gib", type=float, default=None,
                        help="Actual quota verified externally when CPFS statvfs is unreliable.")
    capacity.add_argument("--capacity-confirmed-by-user", action="store_true",
                          help="User explicitly confirmed sufficient real quota; do not invent a free-space value.")
    args = parser.parse_args()
    base = ROOT / "artifacts/wla-stage1"
    base.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="model-preflight-", dir=base))
    model_root = ROOT / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    cache = model_root / "huggingface/hub"
    manifest_path = model_root / "wla-models.lock.json"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": str(SOURCE_MANIFEST),
        "endpoint": constants.ENDPOINT,
        "capacity_confirmed_by_user": args.capacity_confirmed_by_user,
        "tls_verification": True, "implicit_credentials": False,
        "download_requested": args.download, "status": "checking", "models": [],
    }
    download_lock = None

    def save_report():
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "report.json")

    try:
        if not constants.ENDPOINT.startswith("https://"):
            raise RuntimeError("Model endpoint must use verified HTTPS.")
        if args.download:
            download_lock = (model_root / ".wla-download.lock").open("a+")
            fcntl.flock(download_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if manifest_path.exists():
                raise RuntimeError("A model lock already exists; refuse an implicit revision update.")
        api = HfApi(token=False)
        for repo in REPOS:
            info = api.model_info(repo, revision=PINNED[repo], files_metadata=True, timeout=10)
            if info.sha != PINNED[repo]:
                raise RuntimeError(f"Resolved revision differs from the export lock: {repo}")
            files = [
                {"name": f.rfilename, "size": f.size,
                 "lfs_sha256": getattr(getattr(f, "lfs", None), "sha256", None)}
                for f in info.siblings
                if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in allowed_patterns(repo))
            ]
            if not info.sha or not files or any(f["size"] is None for f in files):
                raise RuntimeError(f"Incomplete file/revision metadata for {repo}")
            if not any(f["name"].endswith(".safetensors") for f in files):
                raise RuntimeError(f"No safetensors in {repo}; do not silently load pickle weights.")
            report["models"].append({
                "repo_id": repo, "revision": info.sha, "files": files,
                "bytes": sum(f["size"] for f in files),
            })
        report["estimated_download_bytes"] = sum(m["bytes"] for m in report["models"])
        report["required_free_bytes"] = int(report["estimated_download_bytes"] * 1.2) + 2 * 1024**3
        report["statvfs_free_bytes"] = shutil.disk_usage(model_root).free
        reported_free = report["statvfs_free_bytes"]
        if args.verified_free_gib is not None:
            if not math.isfinite(args.verified_free_gib) or args.verified_free_gib <= 0:
                raise ValueError("Verified quota must be positive.")
            reported_free = int(args.verified_free_gib * 1024**3)
            report["externally_verified_free_bytes"] = reported_free
        report["capacity_sufficient"] = (
            args.capacity_confirmed_by_user or reported_free >= report["required_free_bytes"]
        )
        report["capacity_basis"] = (
            "user_confirmed_sufficient_quota" if args.capacity_confirmed_by_user
            else "externally_verified_free_bytes" if args.verified_free_gib is not None
            else "statvfs"
        )
        report["status"] = "metadata_ready"
        save_report()
        if args.download:
            if not report["capacity_sufficient"]:
                raise RuntimeError("Capacity not confirmed; no weights downloaded. Verify actual CPFS quota.")
            for model in report["models"]:
                report.update(status="downloading", current_repo=model["repo_id"])
                save_report()
                print(json.dumps({"event": "DOWNLOADING", "repo": model["repo_id"],
                                  "revision": model["revision"], "bytes": model["bytes"]}), flush=True)
                # Pin the fetched metadata's immutable commit, not moving "main".
                path = Path(snapshot_download(
                    model["repo_id"], revision=model["revision"], cache_dir=cache, token=False,
                    endpoint=constants.ENDPOINT,
                    allow_patterns=[f["name"] for f in model["files"]], max_workers=2,
                ))
                if path.name != model["revision"]:
                    raise RuntimeError("Model revision changed during preparation; do not run inference.")
                model["snapshot_path"] = str(path)
                report["status"] = "verifying_files"
                save_report()
                for file in model["files"]:
                    target = path / file["name"]
                    file["verified_sha256"] = verify_file(target, file)
                    save_report()
                model["files_verified"] = True
                save_report()
            config = json.loads((Path(report["models"][0]["snapshot_path"]) / "config.json").read_text())
            validate_dependencies(config)
            # Native nested constructors use repo IDs without a revision argument.
            # Populate offline main refs only after every selected file validates.
            refs = [(Path(m["snapshot_path"]).parent.parent / "refs/main", m["revision"])
                    for m in report["models"]]
            for ref, revision in refs:
                if ref.exists() and ref.read_text().strip() != revision:
                    raise RuntimeError(f"Existing cache ref differs; preserve it: {ref}")
            for ref, revision in refs:
                ref.parent.mkdir(parents=True, exist_ok=True)
                if not ref.exists():
                    with ref.open("x") as target:
                        target.write(revision)
            report["status"] = "download_complete"
            report.pop("current_repo", None)
            with manifest_path.open("x") as target:
                target.write(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        report["status"] = "blocked"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        save_report()
        if download_lock is not None:
            download_lock.close()
        print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))
    return 0 if report["status"] in ("metadata_ready", "download_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
