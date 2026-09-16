import copy
import json
from pathlib import Path

import numpy as np
import pytest

from libero_harness.paired import ARMS, HORIZONS, check_manifest, array_digest, check_geometry, geometry
from paired_driver import aggregate, classify, episode_row


def manifest():
    cases = [
        dict(case_id=f"{suite}-{i:02d}", suite=suite, task_id=i, initial_state_id=1,
             calibration_state_id=0, max_env_steps=HORIZONS[suite])
        for i in range(10) for suite in HORIZONS
    ]
    return dict(
        version=1, arms=list(ARMS), cases=cases,
        pilot=dict(settle_steps=10, max_agent_requests=100, episode_timeout_seconds=900),
        agent_config=dict(model="test-model", base_url="https://provider.invalid"),
        provider_identity=dict(model="test-model", base_url="https://provider.invalid"),
        schedule=[dict(case_id=c["case_id"], arm=ARMS[(i + j) % 3])
                  for i, c in enumerate(cases) for j in range(3)],
    )


def row(case_id, arm, classification="success", steps=100):
    return dict(case_id=case_id, arm=arm, classification=classification,
                official_success=classification == "success", task_steps=steps,
                episode_seconds=10, model_requests=0 if arm == "wla" else 10,
                input_tokens=20, output_tokens=10, wla_calls=0 if arm == "opus" else 3,
                handoff_events=0)


def test_fixed_twenty_distinct_tasks_and_sixty_runs():
    m = check_manifest(manifest())
    assert len(m["cases"]) == 20 and len(m["schedule"]) == 60
    assert len({(c["suite"], c["task_id"]) for c in m["cases"]}) == 20
    assert {c["max_env_steps"] for c in m["cases"]} == {220, 280}


@pytest.mark.parametrize("change", ["state", "budget", "case", "order", "arm", "model", "endpoint"])
def test_modified_protocol_is_rejected(change):
    m = manifest()
    if change == "state":
        m["cases"][0]["initial_state_id"] = 0
    elif change == "budget":
        m["cases"][10]["max_env_steps"] += 1
    elif change == "case":
        m["cases"][-1] = m["cases"][0]
    elif change == "order":
        m["cases"].reverse()
    elif change == "arm":
        m["schedule"][0]["arm"] = "wla"
    elif change == "model":
        m["agent_config"]["model"] = "fallback"
    else:
        m["agent_config"]["base_url"] = "https://elsewhere.invalid"
    with pytest.raises(ValueError):
        check_manifest(m)


def test_state_hash_includes_values_dtype_and_shape():
    a = np.arange(8, dtype=np.float64)
    assert array_digest(a) == array_digest(a.copy())
    assert len({array_digest(a), array_digest(a.astype(np.float32)),
                array_digest(a.reshape(2, 4)), array_digest(a + 1)}) == 4


def test_geometry_rejects_camera_and_robot_changes():
    g = dict(front=[0.] * 13, action_dim=7, robot_type="Panda", controller_type="OSC_POSE")
    check_geometry(g, copy.deepcopy(g))
    bad = copy.deepcopy(g)
    bad["front"][4] = .01
    with pytest.raises(AssertionError):
        check_geometry(bad, g)
    bad = dict(g, action_dim=8)
    with pytest.raises(ValueError):
        check_geometry(bad, g)


def test_geometry_reads_action_dim_from_libero_inner_environment():
    from types import SimpleNamespace as NS
    robot = NS(robot_model=object(), controller=object())
    sim = NS(model=NS(camera_name2id=lambda name: 0, cam_fovy=np.array([45.])),
             data=NS(cam_xpos=np.zeros((1, 3)), cam_xmat=np.eye(3).reshape(1, 9)))
    session = NS(env=NS(env=NS(action_dim=7), sim=sim, robots=[robot]))
    assert geometry(session)["action_dim"] == 7


@pytest.mark.parametrize("drift", [None, "front", "robot_type"])
def test_sealing_allows_different_suites_but_rejects_within_suite_drift(tmp_path, monkeypatch, drift):
    from libero_harness import paired
    m = manifest()
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(m))
    monkeypatch.setattr(paired, "load_manifest", lambda path: m)
    directory = tmp_path / "preflight"
    for c in m["cases"]:
        obj = c["suite"] == "libero_object"
        g = dict(front=[1. if obj else 0.] * 13, action_dim=7,
                 robot_type="OnTheGroundPanda" if obj else "MountedPanda",
                 controller_type="OperationalSpaceController")
        if c["case_id"] == "libero_object-09" and drift:
            g[drift] = [2.] * 13 if drift == "front" else "different_robot"
        out = directory / c["case_id"]
        out.mkdir(parents=True)
        (out / "report.json").write_text(json.dumps(dict(
            status="passed", scope=paired.SCOPE, case_id=c["case_id"],
            manifest_sha256=paired.digest(mp), probes=[{}, {}], geometry=g)))
    if drift:
        with pytest.raises((ValueError, AssertionError)):
            paired.seal_preflight(mp, directory)
    else:
        receipt = json.loads(paired.seal_preflight(mp, directory).read_text())
        assert set(receipt["geometry_by_suite"]) == set(HORIZONS)
        assert len(receipt["cases"]) == 20


def test_preflight_reuse_rejects_changed_physical_logic(tmp_path, monkeypatch):
    from libero_harness import paired
    old_source = tmp_path / "old.py"
    old_source.write_text("def preflight_case():\n    return 1\n")
    current = tmp_path / "libero_harness/paired.py"
    current.parent.mkdir()
    current.write_text("def preflight_case():\n    return 2\n")
    old, new = manifest(), manifest()
    old["source_sha256"] = {"libero_harness/paired.py": paired.digest(old_source)}
    new["source_sha256"] = {"libero_harness/paired.py": paired.digest(current)}
    monkeypatch.setattr(paired, "CODE", tmp_path)
    monkeypatch.setattr(paired, "load_manifest",
                        lambda path, **kw: old if path == "old" else new)
    with pytest.raises(ValueError, match="physical rerun required"):
        paired.reuse_preflight("old", tmp_path, old_source, "new", tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("change", [None, "protocol", "other_source", "failed_probe"])
def test_preflight_reuse_is_linked_and_strict(tmp_path, monkeypatch, change):
    from libero_harness import paired
    old_source = tmp_path / "old.py"
    old_source.write_text("def preflight_case():\n    return 1\n\ndef seal_preflight():\n    return 1\n")
    current = tmp_path / "libero_harness/paired.py"
    current.parent.mkdir()
    current.write_text("def preflight_case():\n    return 1\n\ndef seal_preflight():\n    return 2\n")
    old, new = manifest(), manifest()
    old["source_sha256"] = {"libero_harness/paired.py": paired.digest(old_source), "other": "same"}
    new["source_sha256"] = {"libero_harness/paired.py": paired.digest(current), "other": "same"}
    if change == "protocol":
        new["cases"][0]["initial_state_id"] = 2
    elif change == "other_source":
        new["source_sha256"]["other"] = "changed"
    op, np = tmp_path / "old.json", tmp_path / "new.json"
    op.write_text(json.dumps(old))
    np.write_text(json.dumps(new))
    source = tmp_path / "original"
    for c in old["cases"]:
        out = source / c["case_id"]
        out.mkdir(parents=True)
        g = dict(front=[0.] * 13, action_dim=7, robot_type=c["suite"], controller_type="OSC")
        (out / "report.json").write_text(json.dumps(dict(
            status="failed" if change == "failed_probe" else "passed", scope=paired.SCOPE,
            case_id=c["case_id"], manifest_sha256=paired.digest(op), probes=[{}, {}], geometry=g)))
    monkeypatch.setattr(paired, "CODE", tmp_path)
    monkeypatch.setattr(paired, "load_manifest", lambda path, **kw: old if path == op else new)
    dest = tmp_path / "derived"
    if change:
        with pytest.raises(ValueError):
            paired.reuse_preflight(op, source, old_source, np, dest)
        assert not dest.exists()
    else:
        sealed = paired.reuse_preflight(op, source, old_source, np, dest)
        assert len(json.loads(sealed.read_text())["cases"]) == 20
        report = json.loads((dest / old["cases"][0]["case_id"] / "report.json").read_text())
        assert report["physical_evidence_reused"]
        assert report["physical_origin"]["manifest_sha256"] == paired.digest(op)
        assert paired.digest(report["physical_origin"]["report"]) == report["physical_origin"]["report_sha256"]
        assert report["manifest_sha256"] == paired.digest(np)


def test_budget_termination_is_not_an_api_failure():
    result = dict(status="error", official_success=False,
                  environment_errors=[dict(type="BudgetExhausted")])
    assert classify(result, {"status": "passed"}, {}) == "task_failure"
    result["environment_errors"] = [dict(type="ModelResponseError")]
    assert classify(result, {"status": "passed"}, {}) == "infrastructure_error"


def test_done_cannot_score_without_official_success():
    assert classify(dict(status="completed", official_success=False),
                    {"status": "passed"}, {}) == "task_failure"


def test_invalid_policy_json_is_not_excluded_as_infrastructure():
    r = dict(status="error", official_success=False,
             environment_errors=[dict(type="ModelResponseError",
                                      message="Invalid model JSON/schema; execute no action.")])
    assert classify(r, {"status": "passed"}, {}) == "task_failure"


def test_bad_audit_or_live_worker_never_scores_success():
    r = dict(status="completed", official_success=True)
    assert classify(r, {"status": "failed"}, {}) == "integrity_error"
    assert classify(dict(r, worker_alive=True), {"status": "passed"}, {}) == "integrity_error"
    assert classify(r, {"status": "passed"}, {"stop_reason": "supervisor_deadline"}) == "infrastructure_error"


def test_pending_and_error_denominators_are_explicit():
    m = manifest()
    rows = [row(m["cases"][0]["case_id"], "opus"),
            row(m["cases"][1]["case_id"], "opus", "infrastructure_error")]
    s = aggregate(m, rows)
    a = s["arms"]["opus"]
    assert a["successes"] == 1 and a["planned"] == 20 and a["pending"] == 18
    assert a["success_fraction_of_planned"] == .05
    assert a["policy_scored"] == 1 and a["success_fraction_policy_scored"] == 1.
    assert not s["complete"]


def test_duplicate_results_or_unknown_case_rejected():
    m = manifest()
    r = row(m["cases"][0]["case_id"], "wla")
    with pytest.raises(ValueError):
        aggregate(m, [r, r])
    with pytest.raises(ValueError):
        aggregate(m, [row("not-in-manifest", "wla")])


def test_paired_counts_do_not_treat_infra_as_policy_loss():
    m = manifest()
    case = m["cases"][0]["case_id"]
    s = aggregate(m, [row(case, "wla"), row(case, "hybrid", "infrastructure_error")])
    counts = s["paired"]["hybrid_vs_wla"]
    assert counts["infrastructure_or_integrity_excluded"] == 1
    assert counts.get("right_only", 0) == 0
    assert counts["pending"] == 19


def test_all_three_arms_have_identical_case_coverage():
    m = manifest()
    rows = [row(c["case_id"], a) for c in m["cases"] for a in ARMS]
    s = aggregate(m, rows)
    assert s["complete"] and s["finished_episodes"] == 60
    assert all(a["successes"] == 20 for a in s["arms"].values())
