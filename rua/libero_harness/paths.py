import json
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RUA_ROOT", CODE.parent.parent)).resolve()
VENDOR = CODE / "vendor/show_harness"
ARTIFACTS = ROOT / "artifacts/rua-stage2"


def configs(backend=None):
    backend = backend or os.environ.get("RUA_MODEL_BACKEND", "claude")
    cfg = json.loads((CODE / "configs/show_harness_libero.json").read_text())
    local = Path(os.environ.get("RUA_AGENT_CONFIG", ROOT / "configs/agent.local.json"))
    cfg["backend"] = backend
    if backend == "qwen":
        cfg.update(json.loads((CODE / "configs/show_harness_qwen.json").read_text()))
        cfg.pop("auth_file", None)
    elif backend != "claude":
        raise ValueError(f"Unknown model backend: {backend}")
    if local.exists():
        cfg.update(json.loads(local.read_text()))
    for env, key in (("RUA_API_BASE", "base_url"), ("RUA_MODEL", "model"),
                     ("RUA_AUTH_FILE", "auth_file"), ("RUA_AUTH_SCHEME", "auth_scheme")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return json.loads((CODE / "configs/pilot.json").read_text()), cfg


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def execution_lock(path=None):
    """Share the original native-WLA lock; no simultaneous benchmark executor."""
    path = Path(path) if path else ROOT / "artifacts/wla-stage1/.wla-policy.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another LIBERO/WLA execution holds the shared experiment lock.") from None
        yield
