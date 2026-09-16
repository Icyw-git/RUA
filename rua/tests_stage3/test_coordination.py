import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_handoff import CFG, PILOT, Env, Predictor, native_raw, make_session
from test_boundaries import Logger
from core.vlm.vlm_client import VLMResponse
from plugins.recovery.plugin import RecoveryPlugin
from libero_harness.coordination import (
    REVISION, controller_prompt, GripperOwnerRecovery,
)
from libero_harness.environment import LiberoAtomicController
from libero_harness.handoff import NativeChunkExecutor
from libero_harness.runner import build_runner


def v2_cfg():
    return dict(CFG, controller_revision=REVISION, planner_max_tokens=4096,
                max_subgoal_decisions=45, max_decisions=10, recent_moves=5,
                effective_table_height_m=.905)


def test_v2_handoff_discards_requested_action_and_refreshes_history(make_session):
    session = make_session()
    executor = NativeChunkExecutor(session, Predictor(), "original task")
    controller = LiberoAtomicController(session, v2_cfg(), wla_executor=executor)
    controller.step("WLA_CHUNK")
    before = session.obs["robot0_eef_pos"].copy()
    result = controller.step("MV_UP")
    assert result.kind == "handoff" and result.token == "HANDOFF_HOLD" and not result.done
    assert session.budget.steps == 13 and session.trajectory.last_gripper_command == 1
    np.testing.assert_array_equal(before, session.obs["robot0_eef_pos"])
    assert session.trajectory.prediction_input(session.obs, 13).sequence == 23
    # No queued move: the next explicit action can instead call WLA again.
    controller.step("WLA_CHUNK")
    assert executor.calls == 2 and session.budget.steps == 21
    assert executor.predictor.requests[-1][0].task_step == 13


def test_v2_invalid_token_never_stabilizes_or_moves(make_session):
    session = make_session()
    NativeChunkExecutor(session, Predictor(), "task").execute()
    controller = LiberoAtomicController(session, v2_cfg())
    with pytest.raises(ValueError):
        controller.step("UNKNOWN")
    with pytest.raises(ValueError):
        controller.step("MV_UP", continuous=True)
    assert session.budget.steps == 8


def test_v2_atomic_grasp_keeps_native_recovery(make_session):
    session = make_session()
    controller = LiberoAtomicController(session, v2_cfg())
    result = controller.step("GRASP")
    assert result.grasp_empty and session.budget.steps == 20
    assert session.gripper_authority == "atomic"
    assert session.trajectory.last_gripper_command == -1
    original = RecoveryPlugin(empty_width_m=.005, open_width_m=.07)
    adapted = GripperOwnerRecovery(original, session)
    kwargs = dict(token="GRASP", result=result, current_index=0,
                  subgoals=[SimpleNamespace(motion="GRASP")], measured_width_m=.002,
                  subgoal_done=False, gripper_closed=False)
    assert adapted.after_step(**kwargs) == original.after_step(**kwargs)


def test_v2_closed_wla_returns_to_model_without_release(make_session, monkeypatch):
    monkeypatch.setattr("libero_harness.runner.validate_handoff_admission", lambda cfg: None)

    class Client:
        strict_outputs = True
        def __init__(self):
            self.decisions = iter(["WLA_CHUNK", "MV_UP", "MV_UP", "WLA_CHUNK", "DONE"])
            self.prompts = []

        def complete_json(self, prompt, agentview_image, **kwargs):
            self.prompts.append(prompt)
            if "subgoals" in kwargs["schema"]["properties"]:
                payload = {"subgoals": [dict(id="one", target="bowl", affordance="rim",
                            motion="MOVE", description="Reach bowl", completion="Reached bowl")]}
                token = ""
            else:
                token = next(self.decisions)
                payload = dict(decision=token, reasoning="scripted test, not real agent evidence")
            return VLMResponse(token=token, raw_text=json.dumps(payload), payload={"json": payload})

    session = make_session()
    raw = native_raw()
    raw[:, -1] = 0
    executor = NativeChunkExecutor(session, Predictor(raw), "original full task")
    client = Client()
    result = build_runner(session, Logger(session.directory), client, "original full task",
                          PILOT, v2_cfg(), wla_executor=executor).run()
    assert executor.calls == 2 and session.budget.steps == 24  # 8+5+3+8, no RELEASE
    assert session.trajectory.last_gripper_command == 1
    assert session.gripper_authority == "wla"
    assert result.end_reason == "agent_done_without_official_success"
    events = [json.loads(line) for line in
              (session.directory / "environment-steps.jsonl").read_text().splitlines()]
    tokens = [e["token"] for e in events if e["event"] == "step_completed" and e["phase"] == "task"]
    assert tokens == ["WLA_CHUNK"] * 8 + ["HANDOFF_HOLD"] * 5 + ["MV_UP"] * 3 + ["WLA_CHUNK"] * 8
    assert any("requested=MV_UP" in p and "executed=HANDOFF_HOLD only" in p for p in client.prompts)
    assert executor.predictor.requests[-1][0].task_step == 16


def test_v2_front_prompt_removes_fixed_wrist_directions():
    from libero_harness.paths import VENDOR
    prompt = controller_prompt((VENDOR / "prompts/controller.txt").read_text())
    assert "wrist is the primary guide" not in prompt
    assert "ONLY directional guide" in prompt
    assert "{output_contract}" in prompt and "{proprio}" in prompt


def test_v2_budget_exhaustion_during_hold_never_executes_original_move(make_session):
    from libero_harness.budget import Budget, BudgetExhausted
    session = make_session(budget=Budget(max_steps=11))
    NativeChunkExecutor(session, Predictor(), "task").execute()
    with pytest.raises(BudgetExhausted):
        LiberoAtomicController(session, v2_cfg()).step("MV_UP")
    assert session.budget.steps == 11
    assert all(action[:6] == [0] * 6 for action in session.env.actions[-3:])


def test_v2_cannot_admit_legacy_wrist_receipt(tmp_path):
    from libero_harness.runner import validate_handoff_admission
    import hashlib
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(dict(status="passed", wrist_direction_certified=True,
                                   recovery_handoff_certified=True)))
    with pytest.raises(ValueError, match="Front-view"):
        validate_handoff_admission(dict(
            controller_revision=REVISION, hybrid_handoff_receipt=str(path),
            hybrid_handoff_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
