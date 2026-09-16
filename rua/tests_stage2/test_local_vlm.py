import json

import numpy as np
import pytest
import requests

from core.vlm.vlm_client import VLMClient
from libero_harness.budget import Budget, BudgetExhausted
from libero_harness.claude import ModelResponseError
from libero_harness.local_vlm import LocalVLMClient
from libero_harness.paths import configs


@pytest.fixture
def client(tmp_path):
    auth = tmp_path / "test.env"
    auth.write_text("RUA_QWEN_API_KEY=test-only-not-a-secret\n")
    cfg = dict(base_url="http://127.0.0.1:18700/v1", model="qwen3.8-27b",
               auth_env_file=str(auth), auth_key_name="RUA_QWEN_API_KEY")
    return LocalVLMClient(cfg, Budget(max_requests=3), tmp_path / "requests")


def mock_response(client, monkeypatch, raw, finish_reason="stop", status=200):
    calls = []
    class Response:
        status_code = status
        ok = status < 400
        text = "test failure"

        def json(self):
            return dict(model="qwen3.8-27b", usage={"prompt_tokens": 20, "completion_tokens": 5},
                        choices=[dict(message={"content": raw}, finish_reason=finish_reason)])

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(client.session, "post", post)
    return calls


def images():
    return np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8), np.zeros((16, 16, 3), dtype=np.uint8)


def test_reuses_upstream_and_translates_current_vllm_schema(client, monkeypatch):
    assert isinstance(client, VLMClient)
    assert LocalVLMClient.complete_json is VLMClient.complete_json
    schema = {"type": "object", "properties": {"decision": {"enum": ["MV_UP", "DONE"]}},
              "required": ["decision"], "additionalProperties": False}
    calls = mock_response(client, monkeypatch, '{"decision":"MV_UP"}')
    front, wrist = images()
    result = client.complete_json("Original task", front, wrist_image=wrist, schema=schema)
    wire = calls[0][1]["json"]
    assert wire["structured_outputs"] == {"json": schema} and "guided_json" not in wire
    assert wire["seed"] == 7 and wire["temperature"] == 0
    assert wire["chat_template_kwargs"]["enable_thinking"] is False
    assert len([p for p in wire["messages"][0]["content"] if p["type"] == "image_url"]) == 2
    assert result.payload["json"] == {"decision": "MV_UP"}
    assert client.budget.requests == 1 and len(calls) == 1
    record = json.loads((client.trace_dir / "request-001/request.json").read_text())
    assert [x["camera"] for x in record["image_manifest"]] == ["front", "wrist"]
    assert record["returned_model"] == "qwen3.8-27b"
    assert "test-only-not-a-secret" not in json.dumps(record)


@pytest.mark.parametrize("raw", [
    'prefix {"decision":"MV_UP"}', '{"decision":"MV_UP",}',
    '{"decision":"MV_UP","decision":"DONE"}', '{"decision":"NOT_ALLOWED"}'])
def test_invalid_json_never_reaches_upstream_recovery(client, monkeypatch, raw):
    calls = mock_response(client, monkeypatch, raw)
    front, wrist = images()
    schema = {"type": "object", "properties": {"decision": {"enum": ["MV_UP", "DONE"]}}}
    with pytest.raises(ModelResponseError):
        client.complete_json("task", front, wrist_image=wrist, schema=schema)
    assert len(calls) == 1 and client.budget.requests == 1
    assert client.records[0]["status"] == "error"


def test_token_choice_is_strict_and_does_not_recover_suffix(client, monkeypatch):
    calls = mock_response(client, monkeypatch, "I choose DONE")
    front, wrist = images()
    with pytest.raises(ModelResponseError):
        client.complete_token("task", ["DONE", "MV_UP"], front, wrist_image=wrist)
    assert len(calls) == 1
    assert calls[0][1]["json"]["structured_outputs"] == {"choice": ["DONE", "MV_UP"]}


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_incomplete_output_rejected(client, monkeypatch, finish_reason):
    calls = mock_response(client, monkeypatch, "DONE", finish_reason=finish_reason)
    front, wrist = images()
    with pytest.raises(ModelResponseError):
        client.complete_token("task", ["DONE"], front, wrist_image=wrist)
    assert len(calls) == 1


def test_network_failure_logged_once(client, monkeypatch):
    def fail(*a, **k):
        raise requests.Timeout()
    monkeypatch.setattr(client.session, "post", fail)
    front, wrist = images()
    with pytest.raises(ModelResponseError):
        client.complete_text("task", front, wrist_image=wrist)
    assert client.records[0]["status"] == "error"
    assert client.budget.requests == 1


def test_budget_prevents_second_http_call(client, monkeypatch):
    calls = mock_response(client, monkeypatch, "red")
    client.budget.max_requests = 1
    front, wrist = images()
    client.complete_text("color?", front, wrist_image=wrist)
    with pytest.raises(BudgetExhausted):
        client.complete_text("color?", front, wrist_image=wrist)
    assert len(calls) == 1


def test_backend_overlay_preserves_control_and_claude_default(monkeypatch):
    monkeypatch.delenv("RUA_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("RUA_AGENT_CONFIG", "/nonexistent/rua-test-config.json")
    for key in ("RUA_API_BASE", "RUA_MODEL", "RUA_AUTH_FILE"):
        monkeypatch.delenv(key, raising=False)
    pilot, claude = configs()
    qwen_pilot, qwen = configs("qwen")
    assert claude["backend"] == "claude"
    assert claude["model"] == qwen["model"] == ""
    assert claude["base_url"] == ""
    assert claude["auth_file"] is None
    assert pilot == qwen_pilot
    for key in ("token_vectors", "move_amplitude", "move_env_steps", "calibration_sha256",
                "plugins", "disabled_plugins", "planner_max_tokens", "max_decisions"):
        assert claude[key] == qwen[key]
    assert "auth_file" in claude and "auth_file" not in qwen
