"""Bounded recorded-policy handoff measurements; no model or agent requests."""
import argparse
import json
from pathlib import Path
import signal
import tempfile
import traceback

import numpy as np

from .budget import Budget
from .check_handoff import RecordedPredictor
from .environment import LiberoSession, LiberoAtomicController
from .handoff import NativeChunkExecutor
from .paths import ROOT, configs, execution_lock, save_json
from .validate_wla import interrupted


def controller_state(session):
    controller = session.env.env.robots[0].controller
    controller.update(force=True)
    return {name: np.asarray(getattr(controller, name)).tolist() for name in
            ("ee_pos", "ee_pos_vel", "ee_ori_vel", "goal_pos", "goal_ori", "ee_ori_mat")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    pilot, cfg = configs("claude")
    from libero.libero import benchmark
    from libero_utils import get_libero_env
    with execution_lock():
        root = Path(tempfile.mkdtemp(prefix="switch-diagnostic-", dir=ROOT / "artifacts/rua-stage3"))
        report = dict(status="running", cases=[], source=str(args.source), root=str(root))
        save_json(root / "report.json", report)
        print(json.dumps({"run_root": str(root)}), flush=True)
        for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        try:
            for chunks in (2, 6):
                for hold in (0, 5):
                    directory = root / f"prefix-{chunks * 8}-hold-{hold}"
                    directory.mkdir()
                    row = dict(prefix_steps=chunks * 8, hold_steps=hold, movements=[])
                    report["cases"].append(row)
                    env = session = None
                    try:
                        signal.setitimer(signal.ITIMER_REAL, 120)
                        suite = benchmark.get_benchmark_dict()[pilot["suite"]](task_order_index=0)
                        env, instruction = get_libero_env(suite.get_task(0), "wla", resolution=512)
                        session = LiberoSession(env, suite.get_task_init_states(0)[0], cfg,
                                                pilot, Budget(220, 0, 120), directory)
                        executor = NativeChunkExecutor(session, RecordedPredictor(args.source), instruction)
                        for _ in range(chunks):
                            executor.execute()
                        row["native_inputs_matched"] = True
                        atomic = LiberoAtomicController(session, cfg)
                        for token in ("MV_UP", "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_DOWN"):
                            previous = session.obs["robot0_eef_pos"].copy()
                            state = controller_state(session)
                            for _ in range(hold):
                                atomic.step("STOP")
                            settled = session.obs["robot0_eef_pos"].copy()
                            hold_state = controller_state(session)
                            atomic.step(token)
                            delta = session.obs["robot0_eef_pos"] - settled
                            vector = np.array(cfg["token_vectors"][token])
                            signed = float(delta @ vector)
                            record = dict(token=token, hold_delta_m=(settled - previous).tolist(),
                                          delta_m=delta.tolist(), signed_m=signed,
                                          transverse_m=float(np.linalg.norm(delta - signed * vector)),
                                          before=state, after_hold=hold_state,
                                          after_move=controller_state(session))
                            row["movements"].append(record)
                            save_json(root / "report.json", report)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        if session is not None:
                            row.update(task_steps=session.budget.steps, cleanup_errors=session.close())
                        if env is not None:
                            env.close()
                        save_json(root / "report.json", report)
            report["status"] = "completed"
        except BaseException as exc:
            report.update(status="error", error=str(exc), traceback=traceback.format_exc())
        save_json(root / "report.json", report)
        print(json.dumps(report), flush=True)
        return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
