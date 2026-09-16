"""Frozen 20-task, three-arm engineering comparison; no new agent loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from .paths import CODE, configs, execution_lock, save_json

ARMS = ("opus", "wla", "hybrid")
HORIZONS = {"libero_spatial": 220, "libero_object": 280}
SCOPE = "paired20_reset_geometry_and_synthetic_handoff_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    a = np.asarray(value)
    return hashlib.sha256(
        str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()
    ).hexdigest()


def source_hashes():
    files = [
        *sorted((CODE / "libero_harness").glob("*.py")),
        CODE / "scripts/run_show_harness.sh", CODE / "scripts/runtime_env.sh",
        CODE / "scripts/paired_driver.py", CODE / "scripts/supervise_harness.py",
        CODE / "scripts/native_contract.py", CODE / "scripts/eval_native.py",
        CODE / "configs/pilot.json", CODE / "configs/show_harness_libero.json",
        CODE / "vendor/show_harness/core/runners/real.py",
        CODE / "vendor/show_harness/core/vlm/roles.py",
    ]
    return {str(p.relative_to(CODE)): digest(p) for p in files if not p.name.startswith("._")}


def check_manifest(m):
    if m.get("version") != 1 or m.get("arms") != list(ARMS):
        raise ValueError("Unknown comparison protocol")
    cases = m["cases"]
    expected = {(suite, i) for suite in HORIZONS for i in range(10)}
    actual = {(c["suite"], c["task_id"]) for c in cases}
    if len(cases) != 20 or actual != expected:
        raise ValueError("Exactly the predetermined 20 distinct tasks are required")
    if len({c["case_id"] for c in cases}) != 20:
        raise ValueError("Duplicate case identifier")
    ordered = [(suite, i) for i in range(10) for suite in HORIZONS]
    if [(c["suite"], c["task_id"]) for c in cases] != ordered:
        raise ValueError("Case order is frozen before observing outcomes")
    schedule = [dict(case_id=c["case_id"], arm=ARMS[(i + j) % 3])
                for i, c in enumerate(cases) for j in range(3)]
    if m.get("schedule") != schedule:
        raise ValueError("Each task needs exactly three counterbalanced arms")
    for c in cases:
        if (c["initial_state_id"] != 1 or c["calibration_state_id"] != 0
                or c["max_env_steps"] != HORIZONS[c["suite"]]):
            raise ValueError("Task horizons and evaluation/calibration states are frozen")
    if m["pilot"]["settle_steps"] != 10 or m["pilot"]["max_agent_requests"] != 100:
        raise ValueError("Unapproved initialization or request budget")
    if m["pilot"]["episode_timeout_seconds"] != 900:
        raise ValueError("Unapproved wall-clock budget")
    identity = {k: m["agent_config"][k] for k in ("model", "base_url")}
    if not all(identity.values()) or identity != m.get("provider_identity"):
        raise ValueError("Model and endpoint must match the explicitly frozen provider identity")
    return m


def load_manifest(path, *, verify_sources=True):
    m = check_manifest(json.loads(Path(path).read_text()))
    if verify_sources:
        for name, expected in m["source_sha256"].items():
            if digest(CODE / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        if digest(m["handoff_receipt"]) != m["handoff_sha256"]:
            raise ValueError("Base handoff evidence changed")
    return m


def find_case(m, case_id):
    found = [c for c in m["cases"] if c["case_id"] == case_id]
    if len(found) != 1:
        raise ValueError("Case is not in frozen manifest")
    return found[0]


def task_data(m, c, *, calibration=False):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[c["suite"]](
        task_order_index=m["pilot"]["task_order_index"])
    task = suite.get_task(c["task_id"])
    states = suite.get_task_init_states(c["task_id"])
    index = c["calibration_state_id"] if calibration else c["initial_state_id"]
    state = states[index]
    definition = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    key = "calibration_state_sha256" if calibration else "initial_state_sha256"
    if (task.name != c["task_name"] or task.language != c["instruction"]
            or digest(definition) != c["definition_sha256"]
            or array_digest(state) != c[key]):
        raise ValueError("Task definition or initial state drifted")
    return task, state


def generate_manifest(path, handoff_receipt):
    from libero.libero import benchmark, get_libero_path
    from .runner import validate_handoff_admission
    pilot, cfg = configs("claude")
    cfg = dict(cfg, controller_revision="front_view_handoff_v2",
               hybrid_handoff_receipt=str(handoff_receipt),
               hybrid_handoff_sha256=digest(handoff_receipt))
    validate_handoff_admission(cfg)
    cases = []
    # Interleave suites; never reorder based on outcomes.
    suites = {name: benchmark.get_benchmark_dict()[name](
        task_order_index=pilot["task_order_index"]) for name in HORIZONS}
    for task_id in range(10):
        for name, suite in suites.items():
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            definition = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            cases.append(dict(
                case_id=f"{name}-{task_id:02d}", suite=name, task_id=task_id,
                initial_state_id=1, calibration_state_id=0,
                task_name=task.name, instruction=task.language,
                definition_sha256=digest(definition),
                initial_state_sha256=array_digest(states[1]),
                calibration_state_sha256=array_digest(states[0]),
                max_env_steps=HORIZONS[name],
            ))
    manifest = dict(
        version=1, scope="engineering_paired20_not_official_benchmark",
        arms=list(ARMS), cases=cases, pilot=pilot, agent_config=cfg,
        handoff_receipt=str(handoff_receipt), handoff_sha256=digest(handoff_receipt),
        provider_identity={k: cfg[k] for k in ("model", "base_url")},
        source_sha256=source_hashes(),
        schedule=[dict(case_id=c["case_id"], arm=ARMS[(i + j) % 3])
                  for i, c in enumerate(cases) for j in range(3)],
        policy="No full-episode retries or outcome-based sample replacement.",
    )
    check_manifest(manifest)
    if path.exists():
        raise FileExistsError("Do not overwrite a frozen manifest")
    save_json(path, manifest)
    return manifest


def geometry(session):
    sim = session.env.sim
    i = sim.model.camera_name2id("agentview")
    # Robot/camera information only. Never object coordinates.
    return dict(front=np.r_[sim.data.cam_xpos[i], sim.data.cam_xmat[i],
                             sim.model.cam_fovy[i]].tolist(),
                action_dim=int(session.env.env.action_dim),
                robot_type=type(session.env.robots[0].robot_model).__name__,
                controller_type=type(session.env.robots[0].controller).__name__)


def check_geometry(actual, expected):
    for name in ("action_dim", "robot_type", "controller_type"):
        if actual[name] != expected[name]:
            raise ValueError(f"Robot/controller geometry changed: {name}")
    np.testing.assert_allclose(actual["front"], expected["front"], atol=1e-9, rtol=0)


class CalibrationPredictor:
    """Synthetic pose excitation, explicitly NOT a trained WLA prediction."""
    def __init__(self, closed):
        self.closed = closed

    def predict(self, request, instruction, *, timeout_seconds):
        raw = np.zeros((8, 7), dtype=np.float64)
        raw[:, 5] = .02  # Bounded yaw excitation before ownership handoff.
        raw[:, 6] = 0. if self.closed else 1.  # Native gripper conversion.
        return raw


def preflight_case(manifest_path, case_id, output):
    """Separate init0 probes; no policy/API calls and no benchmark scores."""
    from libero_utils import get_libero_env
    from .budget import Budget
    from .environment import LiberoSession, LiberoAtomicController
    from .handoff import NativeChunkExecutor
    from .validate_wla import interrupted
    m = load_manifest(manifest_path)
    c = find_case(m, case_id)
    task, state = task_data(m, c, calibration=True)
    pilot, cfg = dict(m["pilot"], suite=c["suite"]), m["agent_config"]
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", scope=SCOPE, case_id=case_id,
                  manifest_sha256=digest(manifest_path), calibration_state_id=0,
                  policy_inference=False, model_requests=0, probes=[])
    save_json(output / "report.json", report)
    env = session = None
    try:
        env, instruction = get_libero_env(task, "wla", resolution=pilot["render_resolution"])
        session = LiberoSession(env, state, cfg, pilot, Budget(160, 0, 150), output)
        report["geometry"] = geometry(session)
        assert report["geometry"]["action_dim"] == 7
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, 150)
        atomic = LiberoAtomicController(session, cfg)
        atomic.step("MV_UP")
        atomic.step("MV_UP")
        for closed in (False, True):
            # Uses the real chunk/ownership boundary, not direct state mutation.
            executor = NativeChunkExecutor(session, CalibrationPredictor(closed), instruction)
            returned = executor.execute()
            assert returned.executed_env_steps == 8
            command = session.trajectory.last_gripper_command
            assert command == (1. if closed else -1.)
            rows = []
            for token in ("MV_UP", "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_DOWN"):
                before = session.obs["robot0_eef_pos"].copy()
                step = atomic.step(token)
                if step.kind == "handoff":
                    assert step.token == "HANDOFF_HOLD"
                    assert session.last_handoff["executed_steps"] == 5
                    before = session.obs["robot0_eef_pos"].copy()
                    step = atomic.step(token)  # Independent, re-observed test decision.
                delta = session.obs["robot0_eef_pos"] - before
                vector = np.asarray(cfg["token_vectors"][token])
                parallel = float(delta @ vector)
                transverse = float(np.linalg.norm(delta - parallel * vector))
                assert parallel > .005 and transverse < .01, (token, parallel, transverse)
                assert session.trajectory.last_gripper_command == command
                check_geometry(geometry(session), report["geometry"])
                rows.append(dict(token=token, intended_axis_m=parallel, transverse_m=transverse))
            report["probes"].append(dict(gripper_closed=closed, directions=rows))
            save_json(output / "report.json", report)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        errors = session.close() if session is not None else []
        if env is not None:
            env.close()
        if errors:
            report.update(status="failed", cleanup_errors=errors)
        save_json(output / "report.json", report)
    return report


def seal_preflight(manifest_path, directory):
    m = load_manifest(manifest_path)
    receipts = {}
    references = {}
    for c in m["cases"]:
        p = directory / c["case_id"] / "report.json"
        r = json.loads(p.read_text())
        if (r.get("status") != "passed" or r.get("scope") != SCOPE
                or r.get("case_id") != c["case_id"]
                or r.get("manifest_sha256") != digest(manifest_path)
                or len(r.get("probes", [])) != 2):
            raise ValueError("All 20 scoped physical preflights must pass")
        # Spatial uses MountedPanda; Object uses OnTheGroundPanda with a
        # different front camera. Require consistency WITHIN each suite;
        # every scored episode still matches its own case's physical probe.
        reference = references.setdefault(c["suite"], r["geometry"])
        check_geometry(r["geometry"], reference)
        receipts[c["case_id"]] = dict(path=str(p), sha256=digest(p))
    path = directory / "receipt.json"
    if path.exists():
        raise FileExistsError("Do not overwrite sealed preflight")
    save_json(path, dict(status="passed", scope=SCOPE,
                         manifest_sha256=digest(manifest_path), geometry_by_suite=references,
                         geometry_policy="within_suite_and_per_episode_case", cases=receipts))
    return path


def reuse_preflight(source_manifest, source_directory, source_module, manifest_path, directory):
    """Derive receipts without rerunning probes, ONLY for a sealing-only fix.

    Original reports, manifest, videos and source are retained and hash-linked.
    This is explicitly evidence reuse, never a claim of another physical run.
    """
    import ast
    old = load_manifest(source_manifest, verify_sources=False)
    new = load_manifest(manifest_path)
    if {k: v for k, v in old.items() if k != "source_sha256"} != {
            k: v for k, v in new.items() if k != "source_sha256"}:
        raise ValueError("Cannot reuse preflight after protocol/task/config changes")
    module_key = "libero_harness/paired.py"
    if {k: v for k, v in old["source_sha256"].items() if k != module_key} != {
            k: v for k, v in new["source_sha256"].items() if k != module_key}:
        raise ValueError("Cannot reuse preflight after other execution source changes")
    if digest(source_module) != old["source_sha256"][module_key]:
        raise ValueError("Archived source is not the source used by physical probes")
    trees = []
    for path in (source_module, CODE / module_key):
        tree = ast.parse(Path(path).read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name in ("seal_preflight", "reuse_preflight"))]
        trees.append(ast.dump(tree, include_attributes=False))
    if trees[0] != trees[1]:
        raise ValueError("Preflight/execution logic changed; actual physical rerun required")
    reports = []
    for c in old["cases"]:
        path = source_directory / c["case_id"] / "report.json"
        r = json.loads(path.read_text())
        if (r.get("status") != "passed" or r.get("scope") != SCOPE
                or r.get("case_id") != c["case_id"]
                or r.get("manifest_sha256") != digest(source_manifest)
                or len(r.get("probes", [])) != 2 or r.get("physical_evidence_reused")):
            raise ValueError("Only original complete physical probes can be reused")
        r.update(manifest_sha256=digest(manifest_path), physical_evidence_reused=True,
                 physical_origin=dict(report=str(path), report_sha256=digest(path),
                                      manifest=str(source_manifest),
                                      manifest_sha256=digest(source_manifest),
                                      source_module=str(source_module),
                                      source_sha256=digest(source_module)),
                 reuse_reason="Only cross-suite sealing check changed; remaining module AST and sources identical")
        reports.append((c["case_id"], r))
    directory.mkdir(parents=True, exist_ok=False)
    for case_id, r in reports:
        out = directory / case_id
        out.mkdir()
        save_json(out / "report.json", r)
    return seal_preflight(manifest_path, directory)


def case_admission(manifest_path, preflight_path, case_id):
    m = load_manifest(manifest_path)
    c = find_case(m, case_id)
    receipt = json.loads(Path(preflight_path).read_text())
    if (receipt.get("status") != "passed" or receipt.get("scope") != SCOPE
            or receipt.get("manifest_sha256") != digest(manifest_path)
            or set(receipt.get("cases", {})) != {x["case_id"] for x in m["cases"]}):
        raise ValueError("Missing/mismatched paired physical preflight")
    entry = receipt["cases"][case_id]
    if digest(entry["path"]) != entry["sha256"]:
        raise ValueError("Case preflight evidence changed")
    row = json.loads(Path(entry["path"]).read_text())
    if row["status"] != "passed" or row["case_id"] != case_id:
        raise ValueError("Case preflight did not pass")
    return m, c, row


def prepare_episode(args):
    m, c, row = case_admission(args.paired_manifest, args.preflight_receipt, args.case_id)
    if args.arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    args.agent, args.rua_only = args.arm != "wla", args.arm == "opus"
    args.task_id, args.init_id = c["task_id"], c["initial_state_id"]
    args.handoff_receipt = m["handoff_receipt"]
    cfg = dict(m["agent_config"], paired_manifest=str(args.paired_manifest),
               paired_preflight=str(args.preflight_receipt), paired_case_id=args.case_id)
    pilot = dict(m["pilot"], suite=c["suite"], task_ids=[c["task_id"]],
                 max_env_steps=c["max_env_steps"])
    return pilot, cfg, dict(case=c, geometry=row["geometry"], arm=args.arm,
                            manifest_sha256=digest(args.paired_manifest),
                            preflight_sha256=digest(args.preflight_receipt))


def validate_episode_state(evaluation, task, state, definition, session):
    c = evaluation["case"]
    if (task.name != c["task_name"] or task.language != c["instruction"]
            or digest(definition) != c["definition_sha256"]
            or array_digest(state) != c["initial_state_sha256"]):
        raise ValueError("Episode is not the frozen task/state")
    check_geometry(geometry(session), evaluation["geometry"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--handoff-receipt", type=Path, required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("seal")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p = sub.add_parser("episode")
    p.add_argument("--paired-manifest", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Paired parent requires CPU OSMesa and hidden CUDA")
    if args.mode == "manifest":
        m = generate_manifest(args.output, args.handoff_receipt)
        print(json.dumps(dict(manifest=str(args.output), tasks=len(m["cases"]), episodes=60)), flush=True)
        return 0
    if args.mode == "seal":
        print(seal_preflight(args.manifest, args.directory), flush=True)
        return 0
    with execution_lock():
        if args.mode == "preflight":
            report = preflight_case(args.manifest, args.case_id, args.output)
            print(json.dumps(report), flush=True)
            return 0 if report["status"] == "passed" else 1
        from .validate_wla import validate
        return validate(args, {}, {})


if __name__ == "__main__":
    raise SystemExit(main())
