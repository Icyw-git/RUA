import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.vlm.vlm_client import VLMResponse
from core.vlm.roles import _complete_decision_json
from libero_harness.budget import Budget, BudgetExhausted
from libero_harness.claude import ModelResponseError, parse_json, collect_stream, image_block, ClaudeClient
from libero_harness.environment import LiberoSession, LiberoAtomicController
from libero_harness.runner import build_runner
from libero_harness.paths import execution_lock


class Writer:
    def append_data(self, data):
        assert data.shape == (16, 16, 3)

    def close(self):
        pass


class Env:
    def __init__(self):
        self.actions = []
        self.pos = np.array([0., 0., 1.2])
        self.q = np.array([.04, -.04])
        self.succeed_at = None
        self.throw_at = None
        self.image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)

    def obs(self):
        return dict(robot0_eef_pos=self.pos.copy(), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                    robot0_gripper_qpos=self.q.copy(), agentview_image=self.image,
                    robot0_eye_in_hand_image=self.image, forbidden_object_pose=np.ones(7))

    def reset(self):
        pass

    def set_init_state(self, initial):
        return self.obs()

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == self.throw_at:
            raise RuntimeError("injected failed control step")
        self.pos += np.asarray(action[:3]) * .02
        self.q = np.array([.001, -.001]) if action[-1] > 0 else np.array([.04, -.04])
        return self.obs(), 0., False, {}

    def check_success(self):
        return self.succeed_at is not None and len(self.actions) >= self.succeed_at


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("libero_harness.environment.imageio.get_writer", lambda *a, **k: Writer())
    cfg = dict(token_vectors={"MV_UP": [0, 0, 1], "MV_DOWN": [0, 0, -1]},
               move_amplitude=.6, move_env_steps=3, nominal_step_m=.02,
               gripper_env_steps=10, empty_width_m=.005, open_width_m=.07,
               wrist_extra_flip="both", planner_max_tokens=4096, max_subgoal_decisions=45,
               max_decisions=10, recent_moves=5, effective_table_height_m=.905)
    env, budget = Env(), Budget()
    session = LiberoSession(env, np.zeros(3), cfg, {"settle_steps": 10}, budget, tmp_path)
    yield env, budget, session, LiberoAtomicController(session, cfg), cfg
    session.close()


def test_observation_whitelist_and_native_image_transform(setup):
    env, _, session, _, _ = setup
    observation = session.get_observation()
    assert set(observation) == {"agentview", "wrist", "agentview_hd", "wrist_hd", "ee_pose", "gripper_width"}
    np.testing.assert_array_equal(observation["agentview"], env.image[::-1, ::-1])
    np.testing.assert_array_equal(observation["wrist"], env.image)
    assert observation["ee_pose"].shape == (7,)


def test_initialization_and_repeated_control_billing(setup):
    env, budget, session, controller, _ = setup
    assert session.initialization_steps == 10 and budget.steps == 0
    controller.step("MV_UP")
    controller.step("RELEASE")
    controller.step("STOP")
    assert budget.steps == 14 and len(env.actions) == 24
    assert session.obs["robot0_eef_pos"][2] > 1.2


def test_empty_grasp_reopen_is_billed(setup):
    _, budget, _, controller, _ = setup
    result = controller.step("GRASP")
    assert result.grasp_empty and not controller.gripper_closed
    assert budget.steps == 20


def test_done_has_no_control_step_and_never_scores_success(setup):
    env, budget, session, controller, _ = setup
    assert controller.step("DONE").done
    assert len(env.actions) == 10 and budget.steps == 0 and not session.success()


def test_unknown_token_and_chunk_execute_nothing(setup):
    env, _, _, controller, _ = setup
    with pytest.raises(ValueError):
        controller.step("GO_ANYWHERE")
    with pytest.raises(ValueError):
        controller.step("MV_UP", continuous=True)
    assert len(env.actions) == 10


def test_success_stops_inside_repeated_action(setup):
    env, budget, session, controller, _ = setup
    env.succeed_at = 11
    controller.step("MV_UP")
    assert session.success() and budget.steps == 1 and len(env.actions) == 11


def test_budget_stops_inside_repeated_action_and_persists_trace(setup):
    env, budget, session, controller, _ = setup
    budget.max_steps = 2
    with pytest.raises(BudgetExhausted, match="environment_step_budget"):
        controller.step("MV_UP")
    assert budget.steps == 2 and len(env.actions) == 12
    events = [json.loads(x) for x in (session.directory / "environment-steps.jsonl").read_text().splitlines()]
    assert sum(x["event"] == "step_completed" and x["phase"] == "task" for x in events) == 2


def test_failed_step_keeps_uncertain_action(setup):
    env, budget, session, controller, _ = setup
    env.throw_at = 11
    with pytest.raises(RuntimeError):
        controller.step("MV_UP")
    assert budget.steps == 0 and session.uncertain_step["token"] == "MV_UP"


@pytest.mark.parametrize("text", [
    '{"decision":"MV_UP",oops}', '{"decision":"MV_UP","decision":"DONE"}',
    '{"x":NaN}', 'prefix {"decision":"MV_UP"}', '["MV_UP"]'])
def test_invalid_model_output_rejected(text):
    with pytest.raises(ModelResponseError):
        parse_json(text)


def test_schema_and_image_validation():
    with pytest.raises(ModelResponseError):
        parse_json('{"decision":"BAD"}', {"type": "object", "properties": {"decision": {"enum": ["DONE"]}}})
    with pytest.raises(ValueError):
        image_block(np.zeros((4, 4, 3), dtype=float))


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", None])
def test_stream_incomplete_or_refusal_rejected(reason):
    events = [dict(type="content_block_delta", delta=dict(type="text_delta", text='{"decision":"MV_UP"}')),
              dict(type="message_delta", delta=dict(stop_reason=reason)),
              dict(type="message_stop")]
    with pytest.raises(ModelResponseError):
        collect_stream([b"data:" + json.dumps(x).encode() for x in events], lambda: None)


def test_request_budget_includes_every_attempt():
    budget = Budget(max_requests=2)
    budget.before_request()
    budget.before_request()
    with pytest.raises(BudgetExhausted, match="model_request_budget"):
        budget.before_request()
    assert budget.requests == 2
    budget.started = time.monotonic() - 901
    with pytest.raises(BudgetExhausted, match="wall_clock_budget"):
        budget.before_step()


def test_exclusive_execution_lock(tmp_path):
    path = tmp_path / "experiment.lock"
    with execution_lock(path):
        with pytest.raises(RuntimeError, match="shared experiment lock"):
            with execution_lock(path):
                pytest.fail("Second executor entered")
    with execution_lock(path):
        pass


def test_strict_backend_failure_never_recovers_guessed_move():
    class Bad:
        strict_outputs = True

        def complete_json(self, *a, **k):
            raise ModelResponseError('Invalid JSON containing "decision":"MV_UP"')

        def complete_token(self, *a, **k):
            pytest.fail("Malformed response must not trigger token recovery")

    with pytest.raises(ModelResponseError):
        _complete_decision_json(Bad(), "test", ["MV_UP", "DONE"], None)


def test_transport_error_is_not_runtimeerror(tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unit-test-not-a-secret")
    def fail(*a, **k):
        raise requests.Timeout()
    monkeypatch.setattr("libero_harness.claude.requests.post", fail)
    client = ClaudeClient({"model": "test", "base_url": "https://not-contacted.invalid"},
                          Budget(), tmp_path)
    with pytest.raises(ModelResponseError):
        client.complete_text("test", np.zeros((4, 4, 3), dtype=np.uint8))
    assert client.budget.requests == 1
    assert not issubclass(ModelResponseError, RuntimeError)
    assert (tmp_path / "request-001/request.json").exists()


class Logger:
    def __init__(self, path):
        self.run_dir = path
        self.summary = None

    def __getattr__(self, name):
        return lambda *a, **k: None

    def close(self, **k):
        return self.run_dir / "fake.mp4"

    def write_summary(self, value):
        self.summary = value


class ScriptedClient:
    strict_outputs = True

    def __init__(self, token):
        self.token = token
        self.prompts = []

    def complete_json(self, prompt, agentview_image, **kwargs):
        self.prompts.append(prompt)
        assert kwargs["wrist_image"] is not None
        if "subgoals" in kwargs["schema"].get("properties", {}):
            payload = {"subgoals": [dict(id="one", target="bowl", affordance="rim", motion="MOVE",
                                        description="Reach bowl", completion="Reached bowl")]}
        elif self.token == "ERROR":
            raise ModelResponseError("invalid response")
        else:
            payload = dict(decision=self.token, reasoning="Scripted boundary test.")
        return VLMResponse(token="", raw_text=json.dumps(payload), payload={"json": payload})


@pytest.mark.parametrize("token,reason,success", [
    ("DONE", "agent_done_without_official_success", False),
    ("ERROR", "runtime_error:ModelResponseError", False),
    ("MV_UP", "official_success", True)])
def test_native_loop_scores_official_only(setup, token, reason, success):
    env, budget, session, _, cfg = setup
    if success:
        env.succeed_at = 11
    logger = Logger(session.directory)
    client = ScriptedClient(token)
    runner = build_runner(session, logger, client, "Original full task", {}, cfg)
    assert type(runner).__name__ == "RealEpisodeRunner"
    result = runner.run()
    assert result.success is success
    assert result.end_reason == reason
    assert logger.summary["success"] is success
    assert len(client.prompts) == 2
    if not success:
        assert budget.steps == 0 and len(env.actions) == 10
