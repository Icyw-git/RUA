"""RUA-local imports/configuration; no hardware connection or global config edits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
WLA = CODE.parent
ROOT = Path(os.environ.get("RUA_ROOT", WLA.parent)).resolve()


def configure() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models/huggingface/hub"))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for path in (WLA, WLA / "LIBERO", WLA / "experiments/libero"):
        sys.path.insert(0, str(path))
