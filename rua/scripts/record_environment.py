"""Record installation provenance without importing models or accessing GPUs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from bootstrap import ROOT, WLA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--libero-revision", required=True)
    args = parser.parse_args()
    root = ROOT
    archive = args.runtime / f"libero-{args.libero_revision}.tar.gz"
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    sha = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_base_prefix": sys.base_prefix,
        "runtime_directory": str(args.runtime),
        "uv_version": subprocess.check_output(
            [str(args.runtime / "tools/uv"), "--version"], text=True
        ).strip(),
        "wla_commit": subprocess.check_output(
            ["git", "-C", str(WLA), "rev-parse", "HEAD"], text=True
        ).strip(),
        "libero_commit": args.libero_revision,
        "libero_archive_sha256": sha.hexdigest(),
        "libero_source": (
            "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/"
            + args.libero_revision
        ),
        "tls_verification": True,
        "private_glvnd": {
            "package": "libglvnd0_1.7.0-1build1_amd64.deb",
            "sha256": "33f5e07c74f73c2bfb44086ae0f9e6da52acd6c104d30ffe8fd768d8253d5e82",
            "directory": str(args.runtime / "egl"),
            "system_libraries_modified": False,
        },
        "opencv_install_order": "robosuite dependencies, then pinned headless cv2 binary",
        "packages": packages,
    }
    output = root / "artifacts" / "wla-stage1"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="installation-", dir=output))
    (snapshot / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (snapshot / "requirements.lock").write_text("\n".join(packages) + "\n")
    (output / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "requirements.lock").write_text("\n".join(packages) + "\n")
    print(json.dumps({"event": "ENVIRONMENT_INSTALLED", "provenance": str(snapshot)}))


if __name__ == "__main__":
    main()
