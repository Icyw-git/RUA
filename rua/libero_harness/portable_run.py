"""Three public entrypoints; delegate to the same evaluated executor and loop."""
import argparse
import os
from pathlib import Path

from .paths import configs, execution_lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("opus", "wla", "hybrid"), required=True)
    p.add_argument("--wla-service-port", type=int, choices=(8120, 8121, 8122))
    p.add_argument("--handoff-receipt", type=Path)
    p.add_argument("--paired-manifest", type=Path)
    p.add_argument("--preflight-receipt", type=Path)
    p.add_argument("--case-id")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use the CPU-rendering launcher; parent CUDA must remain hidden")
    if args.paired_manifest:
        if args.wla_service_port:
            p.error("Shared WLA service is for the single debug episode only")
        if not all((args.preflight_receipt, args.case_id, args.output)):
            p.error("Paired mode requires --preflight-receipt, --case-id and a new --output")
        args.arm = args.mode
    else:
        if any((args.preflight_receipt, args.case_id, args.output)):
            p.error("Custom case/output flags require a frozen paired manifest")
        if args.mode == "hybrid" and not args.handoff_receipt:
            p.error("Hybrid debug requires a freshly generated --handoff-receipt")
        if args.wla_service_port and args.mode != "wla":
            p.error("--wla-service-port requires --mode wla")
        args.agent, args.rua_only = args.mode != "wla", args.mode == "opus"
        args.task_id, args.init_id = 0, 0
    from .validate_wla import validate
    with execution_lock():
        return validate(args, *configs("claude"))


if __name__ == "__main__":
    raise SystemExit(main())
