import json

import numpy as np
import pytest

from libero_harness.budget import Budget, BudgetExhausted
from libero_harness.claude import (
    ClaudeClient, ModelResponseError, RateLimitError, retry_after_seconds,
)


def setup_client(tmp_path, monkeypatch, statuses, retry_after=None, budget=None):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unit-test-not-a-secret")
    calls, waits = [], []

    class Response:
        def __init__(self, status):
            self.status_code, self.ok = status, status == 200
            self.text = "rate limit"
            self.headers = {} if retry_after is None else {"Retry-After": retry_after}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            events = [
                {"type": "content_block_delta",
                 "delta": {"type": "text_delta", "text": '{"decision":"MV_UP"}'}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                {"type": "message_stop"},
            ]
            return (b"data:" + json.dumps(event).encode() for event in events)

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response(statuses[len(calls) - 1])

    monkeypatch.setattr("libero_harness.claude.requests.post", post)
    monkeypatch.setattr("libero_harness.claude.time.sleep", waits.append)
    client = ClaudeClient(
        {"model": "test", "base_url": "https://not-contacted.invalid",
         "thinking": {"type": "disabled"}, "rate_limit_retries": 2,
         "rate_limit_backoff_seconds": 10},
        budget or Budget(), tmp_path,
    )
    return client, calls, waits


def request(client):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return client.complete_json("Choose an action", image, wrist_image=image)


def test_rate_limit_retries_count_every_attempt_and_keep_observation(tmp_path, monkeypatch):
    client, calls, waits = setup_client(tmp_path, monkeypatch, [429, 429, 200])
    assert request(client).payload["json"] == {"decision": "MV_UP"}
    assert len(calls) == client.budget.requests == 3
    assert client.budget.steps == 0 and waits == [10, 20]
    assert calls[0]["messages"][0]["content"][:2] == calls[2]["messages"][0]["content"][:2]
    assert calls[0]["max_tokens"] == calls[2]["max_tokens"] == 512
    assert all((tmp_path / f"request-{i:03d}/request.json").exists() for i in (1, 2, 3))


def test_three_rate_limits_stop_without_unbounded_retry(tmp_path, monkeypatch):
    client, calls, waits = setup_client(tmp_path, monkeypatch, [429, 429, 429])
    with pytest.raises(RateLimitError):
        request(client)
    assert len(calls) == 3 and waits == [10, 20] and client.budget.steps == 0


def test_honor_retry_after(tmp_path, monkeypatch):
    client, calls, waits = setup_client(tmp_path, monkeypatch, [429, 200], "25")
    request(client)
    assert len(calls) == 2 and waits == [25]


@pytest.mark.parametrize("budget,match", [
    (Budget(max_requests=1), "model_request_budget"),
    (Budget(timeout=5), "insufficient_time_for_rate_limit_retry"),
])
def test_retry_never_expands_episode_budget(tmp_path, monkeypatch, budget, match):
    # Reset the clock: parametrized Budget objects are constructed at collection.
    import time
    budget.started = time.monotonic()
    client, calls, waits = setup_client(tmp_path, monkeypatch, [429], budget=budget)
    with pytest.raises(BudgetExhausted, match=match):
        request(client)
    assert len(calls) == 1 and waits == [] and client.budget.steps == 0


def test_long_server_wait_stops_instead_of_retrying_early(tmp_path, monkeypatch):
    client, calls, waits = setup_client(tmp_path, monkeypatch, [429], "120")
    with pytest.raises(ModelResponseError, match="60-second"):
        request(client)
    assert len(calls) == 1 and waits == []


def test_authentication_error_does_not_retry(tmp_path, monkeypatch):
    client, calls, waits = setup_client(tmp_path, monkeypatch, [403])
    with pytest.raises(ModelResponseError, match="403"):
        request(client)
    assert len(calls) == 1 and waits == []


def test_retry_after_parser(monkeypatch):
    monkeypatch.setattr("libero_harness.claude.time.time", lambda: 0)
    assert retry_after_seconds("Thu, 01 Jan 1970 00:00:20 GMT") == 20
    assert retry_after_seconds("12") == 12
    assert retry_after_seconds("NaN") is None
    assert retry_after_seconds("bad") is None
    assert retry_after_seconds(None) is None
