"""Run one LIBERO-Pro WLA episode through the existing RUA collector."""
from __future__ import annotations

import argparse
import importlib
import sys
from types import SimpleNamespace

from libero_harness.paths import configs, execution_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-id", type=int, default=24)
    parser.add_argument("--wla-service-port", type=int, choices=(8120, 8121, 8122), default=8120)
    args = parser.parse_args()

    # The Pro checkout exposes `libero`; RUA's original runner imports `libero.libero`.
    # This alias affects only this process and keeps the Pro checkout untouched.
    pro_libero = importlib.import_module("libero")
    if not hasattr(pro_libero, "get_libero_path"):
        raise RuntimeError("LIBERO-Pro package is not first on PYTHONPATH")
    sys.modules["libero.libero"] = pro_libero

    pilot, cfg = configs("claude")
    pilot = dict(pilot, suite=args.suite, max_env_steps=300)
    trial = SimpleNamespace(task_id=args.task_id, init_id=args.init_id,
                            wla_service_port=args.wla_service_port,
                            pro_bddl_language=True)
    from libero_harness.validate_wla import validate
    with execution_lock():
        return validate(trial, pilot, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
