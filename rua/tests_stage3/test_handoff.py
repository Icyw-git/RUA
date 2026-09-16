import json
import threading
import time

import numpy as np
import pytest

from libero_harness.budget import Budget, BudgetExhausted
from libero_harness.environment import LiberoSession, LiberoAtomicController
from libero_harness.handoff import (
    ControlOwnership, HandoffError, NativeChunkExecutor, RAW_KEYS,
)
from native_rollout import run_episode as run_native
from libero_harness.audit import audit_wla_records


CFG = dict(token_vectors={"MV_UP": [0, 0, 1], "MV_DOWN": [0, 0, -1]},
           move_amplitude=.6, move_env_steps=3, nominal_step_m=.02,
           gripper_env_steps=10, empty_width_m=.005, open_width_m=.07,
           wrist_extra_flip="both")
PILOT = dict(settle_steps=10, history_obs_step=8, max_env_steps=220,
             episode_timeout_seconds=900)


class Writer:
    def append_data(self, image):
        assert image.shape == (16, 16, 3)

    def close(self):
        pass


class Env:
    def __init__(self, succeed_at=None, done_at=None, throw_at=None):
        self.actions = []
        self.succeed_at, self.done_at, self.throw_at = succeed_at, done_at, throw_at
        self.position = np.array([0., 0., 1.2])
        self.image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        self.qpos = np.array([.04, -.04])

    def observation(self):
        return dict(robot0_eef_pos=self.position.copy(),
                    robot0_eef_quat=np.array([0., 0., 0., 1.]),
                    robot0_gripper_qpos=self.qpos.copy(),
                    agentview_image=self.image.copy(), robot0_eye_in_hand_image=self.image.copy(),
                    forbidden_object_pose=np.ones(7))

    def reset(self):
        pass

    def set_init_state(self, _):
        return self.observation()

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == self.throw_at:
            raise RuntimeError("injected uncertain step")
        self.position += np.asarray(action[:3]) * .02
        self.qpos = np.array([.001, -.001]) if action[-1] > 0 else np.array([.04, -.04])
        return self.observation(), float(self.check_success()), len(self.actions) == self.done_at, {}

    def check_success(self):
        return self.succeed_at is not None and len(self.actions) >= self.succeed_at


def native_raw():
    raw = np.arange(56, dtype=np.float32).reshape(8, 7) / 100
    raw[:, -1] = [0, .49, .5, 1, 0, 1, .5, .499]
    return raw


class Predictor:
    def __init__(self, output=None, callback=None):
        self.output = native_raw() if output is None else output
        self.requests = []
        self.callback = callback

    def predict(self, request, instruction, timeout_seconds):
        self.requests.append((request, instruction, timeout_seconds))
        if self.callback:
            self.callback(request)
        return self.output


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    monkeypatch.setattr("libero_harness.environment.imageio.get_writer", lambda *a, **k: Writer())
    sessions = []

    def make(env=None, budget=None):
        directory = tmp_path / str(len(sessions))
        directory.mkdir()
        session = LiberoSession(env or Env(), np.zeros(3), CFG, PILOT,
                                budget or Budget(), directory)
        sessions.append(session)
        return session

    yield make
    for session in sessions:
        session.close()


def test_settling_is_not_history(make_session):
    session = make_session()
    assert not session.trajectory.history
    assert session.trajectory.sequence == 10
    assert session.budget.steps == 0
    request = session.trajectory.prediction_input(session.obs, session.budget.steps)
    assert request.history is None
    assert set(request.current) == set(RAW_KEYS)


def test_actual_pre_step_history_across_repeated_and_gripper_actions(make_session):
    session = make_session()
    controller = LiberoAtomicController(session, CFG)
    before = session.obs["robot0_eef_pos"].copy()
    controller.step("MV_UP")
    assert len(session.trajectory.history) == 3
    np.testing.assert_array_equal(session.trajectory.history[0]["robot0_eef_pos"], before)
    assert session.trajectory.history[-1]["robot0_eef_pos"][2] < session.obs["robot0_eef_pos"][2]
    controller.step("RELEASE")
    assert len(session.trajectory.history) == 8
    assert session.budget.steps == 13 and session.trajectory.sequence == 23


def test_native_pixels_not_agent_display_and_no_oracle(make_session):
    session = make_session()
    predictor = Predictor()
    NativeChunkExecutor(session, predictor, "original complete task").execute()
    request, instruction, seconds = predictor.requests[0]
    assert set(request.current) == set(RAW_KEYS) and instruction == "original complete task"
    assert 0 < seconds <= 900
    np.testing.assert_array_equal(request.current["agentview_image"], session.env.image)
    assert not np.array_equal(request.current["agentview_image"],
                              session.get_observation()["agentview"])


def test_prediction_copies_cannot_mutate_environment_or_stored_history(make_session):
    session = make_session()
    session.execute([0, 0, .1, 0, 0, 0, -1], 1, "TEST")
    def mutate(request):
        request.current["agentview_image"][:] = 0
        request.history["robot0_eef_pos"][:] = 99
    NativeChunkExecutor(session, Predictor(callback=mutate), "original").execute()
    assert np.any(session.obs["agentview_image"] != 0)
    assert all(not np.all(obs["robot0_eef_pos"] == 99) for obs in session.trajectory.history)


@pytest.mark.parametrize("success,done,max_steps", [(11, None, 220), (27, None, 220),
                                                  (None, 14, 220), (None, None, 19)])
def test_unified_matches_existing_native_queue_history_actions(make_session, success, done, max_steps):
    env_native = Env(succeed_at=success, done_at=done)
    env_unified = Env(succeed_at=success, done_at=done)
    native_inputs = []

    def predict(history, current, step):
        native_inputs.append((history, current, step))
        return native_raw()

    native = run_native(env_native, np.zeros(3), predict,
                        dict(PILOT, max_env_steps=max_steps), lambda *a: None, lambda *a: None)
    session = make_session(env_unified, Budget(max_steps=max_steps))
    predictor = Predictor()
    executor = NativeChunkExecutor(session, predictor, "original")
    while session.budget.steps < max_steps and not (session.success() or session.returned_done):
        executor.execute()
    np.testing.assert_array_equal(env_native.actions, env_unified.actions)
    assert native["wla_calls"] == executor.calls
    assert native["task_control_steps"] == session.budget.steps == executor.executed_steps
    assert native["official_success"] == session.success()
    assert len(native_inputs) == len(predictor.requests)
    for (history, current, step), (request, _, _) in zip(native_inputs, predictor.requests):
        assert request.task_step == step
        assert (request.history is None) == (history is None)
        for key in RAW_KEYS:
            np.testing.assert_array_equal(request.current[key], current[key])
            if history is not None:
                np.testing.assert_array_equal(request.history[key], history[key])


def test_wla_close_then_atomic_move_never_reopens(make_session):
    session = make_session()
    raw = np.zeros((8, 7), dtype=float)
    NativeChunkExecutor(session, Predictor(raw), "original").execute()
    controller = LiberoAtomicController(session, CFG)
    assert controller.gripper_closed
    controller.step("MV_UP")
    assert all(action[-1] == 1 for action in session.env.actions[-11:])
    assert session.budget.steps == 11  # WLA does not auto-reopen its empty close.


def test_budget_failed_grasp_does_not_change_command_state(make_session):
    session = make_session(budget=Budget(max_steps=0))
    controller = LiberoAtomicController(session, CFG)
    with pytest.raises(BudgetExhausted):
        controller.step("GRASP")
    assert not controller.gripper_closed and session.budget.steps == 0


@pytest.mark.parametrize("bad", [np.zeros((7, 7)), np.zeros((8, 6)),
                               np.full((8, 7), np.nan), np.full((8, 7), np.inf)])
def test_bad_chunk_executes_nothing_and_no_retry(make_session, bad):
    session = make_session()
    executor = NativeChunkExecutor(session, Predictor(bad), "original")
    with pytest.raises((ValueError, HandoffError)):
        executor.execute()
    assert session.budget.steps == 0 and not session.trajectory.history
    with pytest.raises(HandoffError, match="Previous WLA call failed"):
        executor.execute()
    assert executor.calls == 1


def test_invalid_last_action_rejects_entire_chunk_without_clipping(make_session):
    session = make_session()
    raw = native_raw()
    raw[-1, 0] = 1.5
    executor = NativeChunkExecutor(session, Predictor(raw), "original")
    with pytest.raises(HandoffError, match="normalized range"):
        executor.execute()
    assert session.budget.steps == 0


def test_rotational_actions_are_not_zeroed_or_repeated(make_session):
    session = make_session()
    raw = native_raw()
    NativeChunkExecutor(session, Predictor(raw), "original").execute()
    np.testing.assert_array_equal(np.array(session.env.actions[10:])[:, 3:6], raw[:, 3:6])


def test_late_prediction_no_step_no_history(make_session):
    session = make_session()
    def late(_):
        session.budget.started = time.monotonic() - 901
    executor = NativeChunkExecutor(session, Predictor(callback=late), "original")
    with pytest.raises(BudgetExhausted, match="wall_clock_budget"):
        executor.execute()
    assert session.budget.steps == 0 and not session.trajectory.history
    assert executor.last_result.residual_actions_discarded == 8


def test_failed_prediction_preserves_trajectory_and_no_step(make_session):
    session = make_session()
    def fail(_):
        raise TimeoutError("prediction deadline")
    executor = NativeChunkExecutor(session, Predictor(callback=fail), "original")
    with pytest.raises(TimeoutError):
        executor.execute()
    assert session.budget.steps == 0 and session.trajectory.sequence == 10
    assert session.control.owner is None


def test_stale_prediction_fails_closed(make_session):
    session = make_session()
    def stale(_):
        session.trajectory.sequence += 1
    executor = NativeChunkExecutor(session, Predictor(callback=stale), "original")
    with pytest.raises(HandoffError, match="Stale"):
        executor.execute()
    assert session.budget.steps == 0


def test_atomic_cannot_enter_during_wla_prediction(make_session):
    session = make_session()
    controller = LiberoAtomicController(session, CFG)
    def callback(_):
        with pytest.raises(HandoffError, match="already owned"):
            controller.step("RELEASE")
    NativeChunkExecutor(session, Predictor(callback=callback), "original").execute()
    assert session.budget.steps == 8


def test_uncertain_env_action_stops_all_future_controllers(make_session):
    session = make_session(Env(throw_at=12))
    executor = NativeChunkExecutor(session, Predictor(), "original")
    with pytest.raises(RuntimeError, match="uncertain"):
        executor.execute()
    assert session.budget.steps == 1
    assert len(session.trajectory.history) == 1
    assert executor.last_result.executed_env_steps == 1
    assert executor.last_result.attempted_env_steps == 2
    assert executor.last_result.residual_actions_discarded == 6
    assert session.uncertain_step is not None
    with pytest.raises(HandoffError, match="uncertain"):
        LiberoAtomicController(session, CFG).step("MV_UP")
    assert len(session.env.actions) == 12


def test_budget_stops_inside_chunk_and_has_return_record(make_session):
    session = make_session(budget=Budget(max_steps=2))
    executor = NativeChunkExecutor(session, Predictor(), "original")
    result = executor.execute()
    assert result.executed_env_steps == 2 and result.residual_actions_discarded == 6
    assert result.observation_sequence == 12 and not result.official_success
    assert result.stop_reason == "environment_step_budget"
    events = [json.loads(line) for line in
              (session.directory / "environment-steps.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "wla_returned"
    assert events[-1]["executed_env_steps"] == 2
    with pytest.raises(BudgetExhausted):
        executor.execute()
    assert executor.calls == 1


def test_agent_done_still_does_not_score_or_move(make_session):
    session = make_session()
    result = LiberoAtomicController(session, CFG).step("DONE")
    assert result.done and not session.success() and session.budget.steps == 0


def test_lease_expiry_cross_thread_and_release():
    ownership = ControlOwnership()
    errors = []
    with ownership.claim("test") as ticket:
        ownership.validate(ticket)
        def attempt():
            try:
                ownership.validate(ticket)
            except HandoffError:
                errors.append("cross-thread rejected")
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive() and errors
    with pytest.raises(HandoffError):
        ownership.validate(ticket)
    with ownership.claim("next"):
        pass


def test_audit_checks_saved_native_chunk_not_just_step_counts(make_session):
    session = make_session(Env(succeed_at=13))
    NativeChunkExecutor(session, Predictor(), "original").execute()
    directory = session.directory / "wla"
    directory.mkdir()
    np.save(directory / "actions-000.npy", native_raw())
    events = [json.loads(line) for line in
              (session.directory / "environment-steps.jsonl").read_text().splitlines()]
    result = dict(wla_calls=1, task_steps=3, worker_alive=False)
    audit_wla_records(session.directory, result, events)
    wrong = native_raw()
    wrong[0, 0] = .77
    np.save(directory / "actions-000.npy", wrong)
    with pytest.raises(AssertionError):
        audit_wla_records(session.directory, result, events)


def test_native_action_adapter_does_not_promote_chunk_to_subgoal_success(make_session):
    session = make_session()
    executor = NativeChunkExecutor(session, Predictor(), "original complete instruction")
    controller = LiberoAtomicController(session, CFG, wla_executor=executor)
    result = controller.step("WLA_CHUNK")
    assert result.kind == "delegated" and not result.done and not result.grasp_empty
    assert json.loads(result.note)["executed_env_steps"] == 8
    assert executor.predictor.requests[0][1] == "original complete instruction"


def test_native_action_adapter_rejects_unconfigured_wla(make_session):
    session = make_session()
    with pytest.raises(ValueError, match="explicitly enabled"):
        LiberoAtomicController(session, CFG).step("WLA_CHUNK")
    assert session.budget.steps == 0


@pytest.mark.parametrize("close,expected_steps", [(False, 8), (True, 18)])
def test_original_planner_controller_loop_dispatches_wla_then_done(
        make_session, close, expected_steps, monkeypatch):
    pytest.importorskip("torch")
    from core.vlm.vlm_client import VLMResponse
    from libero_harness.runner import build_runner
    from test_boundaries import Logger
    # Synthetic env/predictor contract test only. Real agent admission stays closed.
    monkeypatch.setattr("libero_harness.runner.validate_handoff_admission", lambda cfg: None)

    class Client:
        strict_outputs = True
        prompts = []
        decisions = ["WLA_CHUNK", "DONE"]

        def complete_json(self, prompt, agentview_image, **kwargs):
            self.prompts.append(prompt)
            schema = kwargs["schema"]
            if "subgoals" in schema["properties"]:
                payload = {"subgoals": [dict(id="one", target="bowl", affordance="rim", motion="MOVE",
                                            description="Reach bowl", completion="Reached bowl")]}
                token = ""
            else:
                assert "WLA_CHUNK" in schema["properties"]["decision"]["enum"]
                token = self.decisions.pop(0)
                payload = dict(decision=token, reasoning="scripted boundary test")
            return VLMResponse(token=token, raw_text=json.dumps(payload), payload={"json": payload})

    session = make_session()
    cfg = dict(CFG, planner_max_tokens=4096, max_subgoal_decisions=45, max_decisions=10,
               recent_moves=5, effective_table_height_m=.905)
    raw = native_raw()
    raw[:, -1] = 0 if close else 1
    executor = NativeChunkExecutor(session, Predictor(raw), "original full task")
    client = Client()
    result = build_runner(session, Logger(session.directory), client, "original full task",
                          PILOT, cfg, wla_executor=executor).run()
    assert executor.calls == 1 and session.budget.steps == expected_steps
    if close:
        # Preserve upstream recovery: it runs AFTER the bounded WLA call and
        # explicitly bills a 10-step RELEASE. Do not hide this integration risk.
        assert session.trajectory.last_gripper_command == -1
        events = [json.loads(line) for line in
                  (session.directory / "environment-steps.jsonl").read_text().splitlines()]
        wla_return = next(i for i, event in enumerate(events) if event["event"] == "wla_returned")
        releases = [i for i, event in enumerate(events)
                    if event["event"] == "step_completed" and event["token"] == "RELEASE"]
        assert len(releases) == 10 and min(releases) > wla_return
    assert result.end_reason == "agent_done_without_official_success"
    assert not session.success()
    assert any("Last WLA return:" in p and '"executed_env_steps": 8' in p for p in client.prompts)


def test_cannot_wire_wla_to_a_different_environment(make_session):
    first, second = make_session(), make_session()
    executor = NativeChunkExecutor(first, Predictor(), "original")
    with pytest.raises(ValueError, match="share one session"):
        LiberoAtomicController(second, CFG, wla_executor=executor)


def test_real_agent_handoff_is_closed_without_certification():
    pytest.importorskip("torch")
    from libero_harness.runner import validate_handoff_admission
    with pytest.raises(ValueError, match="not certified"):
        validate_handoff_admission({})


@pytest.mark.parametrize("receipt", [
    dict(status="failed", wrist_direction_certified=False, recovery_handoff_certified=False),
    dict(status="passed", wrist_direction_certified=True, recovery_handoff_certified=False),
    dict(status="passed", wrist_direction_certified=False, recovery_handoff_certified=True),
])
def test_partial_handoff_checks_do_not_unlock_autonomous_agent(tmp_path, receipt):
    pytest.importorskip("torch")
    import hashlib
    from libero_harness.runner import validate_handoff_admission
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    cfg = dict(hybrid_handoff_receipt=str(path),
               hybrid_handoff_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Both directional and recovery"):
        validate_handoff_admission(cfg)
