"""Audit workspace-local runtime and exact dependency equivalence; no inference."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from bootstrap import CODE, ROOT, configure

# This verification may import the WLA class but must not use GPUs or the network.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
configure()


def main() -> int:
    base = ROOT / "artifacts/wla-stage1"
    base.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="runtime-check-", dir=base))
    runtime = Path(os.environ.get("RUA_RUNTIME", ROOT / "runtime/wla-libero")).resolve()
    report = {"status": "checking", "scope": "relocation_no_model_load"}
    try:
        paths = {
            "python": sys.executable,
            "base_prefix": sys.base_prefix,
            "venv": sys.prefix,
            **{key: os.environ[key] for key in (
                "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "XDG_CACHE_HOME",
                "TMPDIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME",
            )},
        }
        report["paths"] = paths
        expected_python = Path(os.environ.get("RUA_VENV_PYTHON", runtime / "venv/bin/python"))
        assert Path(sys.executable).resolve() == expected_python.resolve()
        report["shared_dependency_override"] = not Path(sys.prefix).resolve().is_relative_to(ROOT)
        normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
        previous = {normalize(line.split("==")[0]): line.split("==")[1]
                    for line in (CODE / "requirements.lock").read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        current = {normalize(d.metadata["Name"]): d.version
                   for d in importlib.metadata.distributions()}
        report["package_diff"] = {
            "missing_or_changed": {k: dict(expected=v, actual=current.get(k))
                                   for k, v in previous.items() if current.get(k) != v},
            "additional": {k: v for k, v in current.items() if k not in previous},
        }
        assert not report["package_diff"]["missing_or_changed"], report["package_diff"]
        report["package_count"] = len(current)
        import cv2
        import torch
        from wla_utils import WLA0

        assert re.search(r"GUI:\s+NONE", cv2.getBuildInformation())
        assert not torch.cuda.is_initialized()
        report["module_paths"] = {m.__name__: m.__file__ for m in (cv2, torch)}
        report["wla_class_imported"] = WLA0.__name__
        report["cuda_initialized"] = torch.cuda.is_initialized()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
