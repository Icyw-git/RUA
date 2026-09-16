import json

import numpy as np
import pytest

from libero_harness.budget import Budget
from libero_harness.claude import ClaudeClient, ModelResponseError


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_explicit_thinking_is_sent_and_recorded_without_larger_budget(
    thinking, tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unit-test-not-a-secret")
    calls = []

    class Response:
        status_code = 200
        ok = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            events = [
                {"type": "message_start", "message": {"model": "test"}},
                {"type": "content_block_delta",
                 "delta": {"type": "text_delta", "text": '{"decision":"MV_UP"}'}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                 "usage": {"output_tokens": 10,
                           "output_tokens_details": {"thinking_tokens": 0}}},
                {"type": "message_stop"},
            ]
            return (b"data:" + json.dumps(event).encode() for event in events)

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("libero_harness.claude.requests.post", post)
    cfg = {"model": "test", "base_url": "https://not-contacted.invalid"}
    if thinking is not None:
        cfg["thinking"] = thinking
    budget = Budget(max_steps=220, max_requests=100, timeout=900)
    client = ClaudeClient(cfg, budget, tmp_path)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    answer = client.complete_json("Choose an action", image, wrist_image=image)
    assert answer.payload["json"] == {"decision": "MV_UP"}
    assert calls[0].get("thinking") == thinking
    assert ("thinking" in calls[0]) == (thinking is not None)
    assert calls[0]["max_tokens"] == 512
    assert "temperature" not in calls[0]
    assert budget.requests == 1 and budget.steps == 0
    assert (budget.max_steps, budget.max_requests, budget.timeout) == (220, 100, 900)
    record = json.loads((tmp_path / "request-001/request.json").read_text())
    assert record["thinking"] == thinking
    assert record["usage"]["output_tokens_details"]["thinking_tokens"] == 0
    assert "unit-test-not-a-secret" not in json.dumps(record)


@pytest.mark.parametrize("thinking", ["disabled", {"type": "enabled"}, {}])
def test_invalid_thinking_setting_is_rejected_before_request(thinking, tmp_path):
    with pytest.raises(ValueError, match="disabled thinking"):
        ClaudeClient({"thinking": thinking}, Budget(), tmp_path)


def test_thinking_only_truncated_response_is_still_rejected():
    from libero_harness.claude import collect_stream

    events = [
        {"type": "message_start", "message": {"model": "test"}},
        {"type": "content_block_delta",
         "delta": {"type": "thinking_delta", "thinking": "not an action"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
         "usage": {"output_tokens": 512,
                   "output_tokens_details": {"thinking_tokens": 512}}},
        {"type": "message_stop"},
    ]
    receipt = {}
    with pytest.raises(ModelResponseError, match="max_tokens"):
        collect_stream(
            [b"data:" + json.dumps(event).encode() for event in events],
            lambda: None, receipt,
        )
    assert receipt["response_text"] == ""
    assert receipt["usage"]["output_tokens_details"]["thinking_tokens"] == 512
