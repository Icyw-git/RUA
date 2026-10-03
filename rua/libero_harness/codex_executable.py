"""Find the Codex CLI used by optional simulator smoke runs."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_codex_binary(configured: str | None = None) -> Path:
    candidate = configured or os.environ.get("RUA_CODEX_BINARY") or "codex"
    executable = shutil.which(candidate)
    if executable is None:
        raise FileNotFoundError(
            f"Codex CLI is not executable: {candidate}. "
            "Set RUA_CODEX_BINARY or add codex to PATH."
        )
    return Path(executable).resolve()
