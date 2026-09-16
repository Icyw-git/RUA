"""Strict streaming Anthropic-protocol boundary; credentials never enter artifacts."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import jsonschema
import numpy as np
import requests
from PIL import Image

from core.vlm.vlm_client import VLMResponse
from .budget import Budget, BudgetExhausted


class ModelResponseError(ValueError):
    """Not RuntimeError: upstream must not recover a guessed action from this error."""


class RateLimitError(ModelResponseError):
    def __init__(self, retry_after=None):
        super().__init__("Gateway HTTP 429; no action returned.")
        self.retry_after = retry_after


def retry_after_seconds(value):
    """Honor numeric or HTTP-date Retry-After; never shorten the server's delay."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def image_block(image):
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 image.")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    blob = stream.getvalue()
    return (
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                    "data": base64.b64encode(blob).decode("ascii")}},
        {"shape": list(pixels.shape), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
         "png_sha256": hashlib.sha256(blob).hexdigest()},
    )


def collect_stream(lines, check_time, receipt=None):
    text, model, stop_reason = "", None, None
    usage = {}
    stopped = False
    for line in lines:
        check_time()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelResponseError("Malformed server event; execute no action.") from exc
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message", {})
            model = message.get("model")
            usage.update(message.get("usage") or {})
        elif kind == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                text += block.get("text", "")
        elif kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
        elif kind == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
            usage.update(event.get("usage") or {})
        elif kind == "message_stop":
            stopped = True
        elif kind == "error":
            raise ModelResponseError("Gateway returned an error event; execute no action.")
    if receipt is not None:
        receipt.update(returned_model=model, usage=usage, stop_reason=stop_reason,
                       response_complete=stopped)
        receipt["response_text"] = text.strip()
    if not stopped or stop_reason != "end_turn" or not text.strip():
        raise ModelResponseError(f"Incomplete model response (stop_reason={stop_reason}); execute no action.")
    return text.strip(), model, usage


def parse_json(text, schema=None):
    clean = text.strip()
    if clean.startswith("```") and clean.endswith("```"):
        lines = clean.splitlines()
        if lines[0].strip() not in ("```", "```json"):
            raise ModelResponseError("Invalid JSON fence.")
        clean = "\n".join(lines[1:-1]).strip()
    try:
        def no_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        value = json.loads(clean, object_pairs_hook=no_duplicate_keys,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        if schema is not None:
            jsonschema.validate(value, schema)
        return value
    except (ValueError, jsonschema.ValidationError) as exc:
        raise ModelResponseError("Invalid model JSON/schema; execute no action.") from exc


class ClaudeClient:
    strict_outputs = True
    max_tokens = 512

    def __init__(self, cfg, budget: Budget, trace_dir: Path):
        self.cfg, self.budget, self.trace_dir = cfg, budget, trace_dir
        # This baseline uses non-CoT control. Explicitly disabling thinking avoids
        # the gateway consuming the whole 512-token action budget internally.
        # Omission preserves the historical server-default behavior for replay.
        self.thinking = cfg.get("thinking")
        if self.thinking is not None and self.thinking != {"type": "disabled"}:
            raise ValueError("This baseline supports only explicit disabled thinking.")
        self.rate_limit_retries = int(cfg.get("rate_limit_retries", 0))
        self.rate_limit_backoff = float(cfg.get("rate_limit_backoff_seconds", 10))
        if not 0 <= self.rate_limit_retries <= 2:
            raise ValueError("At most two rate-limit retries are supported.")
        if not math.isfinite(self.rate_limit_backoff) or not 1 <= self.rate_limit_backoff <= 30:
            raise ValueError("Rate-limit backoff must be between 1 and 30 seconds.")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not self._token:
            if not cfg.get("auth_file"):
                raise ValueError("Set ANTHROPIC_AUTH_TOKEN or RUA_AUTH_FILE; no bundled credentials.")
            self._token = json.loads(Path(cfg["auth_file"]).read_text())["token"]
        if not self._token:
            raise ValueError("Claude credentials unavailable.")
        self.auth_scheme = cfg.get("auth_scheme", "bearer")
        if self.auth_scheme not in ("bearer", "x-api-key"):
            raise ValueError("RUA_AUTH_SCHEME must be bearer or x-api-key.")
        self.records = []

    def _auth_headers(self):
        credential = ({"Authorization": "Bearer " + self._token}
                      if self.auth_scheme == "bearer" else {"x-api-key": self._token})
        return {"anthropic-version": "2023-06-01", **credential}

    def _request(self, prompt, front, wrist=None, max_tokens=None, temperature=0.0):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return self._request_once(prompt, front, wrist, max_tokens, temperature)
            except RateLimitError as exc:
                if attempt == self.rate_limit_retries:
                    raise
                delay = max(self.rate_limit_backoff * (2 ** attempt), exc.retry_after or 0)
                # A long Retry-After is a stop, not permission to retry sooner.
                if delay > 60:
                    raise ModelResponseError("Rate-limit wait exceeds the bounded 60-second retry wait.") from exc
                if self.budget.requests >= self.budget.max_requests:
                    raise BudgetExhausted("model_request_budget") from exc
                self.budget.check_time()
                remaining = self.budget.timeout - (time.monotonic() - self.budget.started)
                if delay >= remaining:
                    raise BudgetExhausted("insufficient_time_for_rate_limit_retry") from exc
                record = self.records[-1]
                record.update(retry_wait_seconds=delay, retry_number=attempt + 1)
                path = self.trace_dir / f"request-{record['call_id']:03d}" / "request.json"
                path.write_text(json.dumps(record, indent=2) + "\n")
                # The environment is not stepped while waiting; the same observation
                # is retried, with a fresh request id and remaining-budget prompt.
                time.sleep(delay)
                self.budget.check_time()

    def _request_once(self, prompt, front, wrist=None, max_tokens=None, temperature=0.0):
        self.budget.before_request()
        call_id = self.budget.requests
        started = time.monotonic()
        directory = self.trace_dir / f"request-{call_id:03d}"
        directory.mkdir()
        blocks, images = [], []
        for name, image in (("front", front), ("wrist", wrist)):
            if image is None:
                continue
            if isinstance(image, (list, tuple)):
                raise ValueError("First baseline accepts exactly front and wrist, no extra media.")
            block, manifest = image_block(image)
            blocks.append(block)
            images.append(dict(camera=name, **manifest))
            Image.fromarray(np.asarray(image)).save(directory / f"{name}.png")
        full_prompt = self.budget.describe() + "\n\n" + prompt
        (directory / "prompt.txt").write_text(full_prompt)
        blocks.append({"type": "text", "text": full_prompt})
        # This gateway/model rejects temperature (HTTP 400, verified 2026-09-15).
        # Keep the upstream interface but record the server-default sampling policy.
        body = dict(model=self.cfg["model"], max_tokens=int(max_tokens or self.max_tokens),
                    stream=True,
                    messages=[{"role": "user", "content": blocks}])
        if self.thinking is not None:
            body["thinking"] = dict(self.thinking)
        record = dict(call_id=call_id, requested_model=body["model"], image_manifest=images,
                      max_tokens=body["max_tokens"], sampling="server_default_temperature_unsupported",
                      thinking=body.get("thinking"),
                      status="started")
        self.records.append(record)
        try:
            remaining = self.budget.timeout - (time.monotonic() - self.budget.started)
            with requests.post(
                self.cfg["base_url"].rstrip("/") + "/v1/messages",
                headers=self._auth_headers(),
                json=body, stream=True, timeout=(min(10, max(1, remaining)), min(90, max(1, remaining))),
            ) as response:
                record["http_status"] = response.status_code
                if not response.ok:
                    record["error_body_excerpt"] = response.text.replace(self._token, "[REDACTED]")[:1000]
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        record["retry_after"] = retry_after
                        raise RateLimitError(retry_after_seconds(retry_after))
                    raise ModelResponseError(f"Gateway HTTP {response.status_code}; no automatic retry.")
                raw, model, usage = collect_stream(response.iter_lines(), self.budget.check_time, record)
            self.budget.check_time()
            raw = raw.replace(self._token, "[REDACTED]")
            (directory / "response.txt").write_text(raw)
            record.update(status="received", returned_model=model, usage=usage)
            return VLMResponse(token="", raw_text=raw,
                               payload={"latency_s": time.monotonic() - started, "usage": usage,
                                        "returned_model": model, "request_id": call_id})
        except requests.RequestException as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise ModelResponseError(f"Gateway transport {type(exc).__name__}; execute no action.") from None
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            if "response_text" in record:
                (directory / "response.txt").write_text(
                    record.pop("response_text").replace(self._token, "[REDACTED]"))
            record["latency_s"] = time.monotonic() - started
            (directory / "request.json").write_text(json.dumps(record, indent=2) + "\n")

    def complete_json(self, prompt, agentview_image, wrist_image=None, schema=None,
                      max_tokens=None, temperature=0.0, **kwargs):
        if schema is not None:
            prompt += "\n\nOutput JSON schema:\n" + json.dumps(schema, separators=(",", ":"))
        response = self._request(prompt, agentview_image, wrist_image, max_tokens, temperature)
        payload = parse_json(response.raw_text, schema)
        response.payload["json"] = payload
        return response

    def complete_text(self, prompt, agentview_image, wrist_image=None,
                      max_tokens=None, temperature=0.0, **kwargs):
        return self._request(prompt, agentview_image, wrist_image, max_tokens, temperature)

    def complete_token(self, prompt, allowed_tokens, agentview_image, wrist_image=None, **kwargs):
        response = self._request(prompt, agentview_image, wrist_image, 64, 0.0)
        if response.raw_text not in allowed_tokens:
            raise ModelResponseError("Model token is not exactly one allowed action; execute nothing.")
        return VLMResponse(token=response.raw_text, raw_text=response.raw_text, payload=response.payload)
