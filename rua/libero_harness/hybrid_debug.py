"""One explicitly scoped Opus/RUA coordination debug; uses the native runner."""
import argparse
import os

from .paths import configs, execution_lock
from .validate_wla import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("claude",), default="claude")
    parser.add_argument("--handoff-receipt")
    parser.add_argument("--rua-only", action="store_true",
                        help="Same controller-v2 boundary without WLA, for a separate matched baseline.")
    args = parser.parse_args()
    if not args.rua_only and not args.handoff_receipt:
        parser.error("--handoff-receipt is required for hybrid debug")
    args.agent, args.task_id, args.init_id = True, 0, 0
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Parent requires CPU OSMesa and hidden CUDA.")
    pilot, cfg = configs(args.backend)
    with execution_lock():
        return validate(args, pilot, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
