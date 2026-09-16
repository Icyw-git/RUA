"""Reuse the upstream VLM client, with bounded/strict vLLM 0.23 transport."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from core.vlm.vlm_client import VLMClient
from .claude import ModelResponseError, parse_json
from .paths import save_json


class LocalVLMClient(VLMClient):
    """Upstream request construction and interfaces; no new agent/planner loop."""
    strict_outputs = True

    def __init__(self, cfg, budget, trace_dir):
        self.cfg, self.budget, self.trace_dir = cfg, budget, Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        values = {}
        for line in Path(cfg["auth_env_file"]).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.removeprefix("export ").strip()] = value.strip().strip("\"'")
        token = os.environ.get(cfg["auth_key_name"]) or values.get(cfg["auth_key_name"])
        if not token:
            raise ValueError("Local model API credential unavailable.")
        super().__init__(
            base_url=cfg["base_url"], model=cfg["model"], api_key=token,
            timeout_s=90, max_tokens=512, temperature=0,
            provider="vllm", api_dialect="vllm", max_retries=0,
            chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False},
        )
        self.records = []

    def _post_chat(self, payload):
        # All inherited complete_* paths pass here. Validate before upstream's
        # permissive token/JSON recovery and count each actual HTTP attempt.
        self.budget.before_request()
        started = time.monotonic()
        call_id = self.budget.requests
        directory = self.trace_dir / f"request-{call_id:03d}"
        directory.mkdir()
        wire = copy.deepcopy(payload)
        schema, choices = wire.pop("guided_json", None), wire.pop("guided_choice", None)
        # vLLM 0.23 uses structured_outputs, not the older guided_* fields.
        if schema is not None:
            wire["structured_outputs"] = {"json": schema}
        elif choices is not None:
            wire["structured_outputs"] = {"choice": choices}
        wire["seed"] = 7
        content = wire["messages"][0]["content"]
        content.insert(0, {"type": "text", "text": self.budget.describe()})
        images, texts, image_index = [], [], 0
        for item in content:
            if item["type"] == "text":
                texts.append(item["text"])
            elif item["type"] == "image_url":
                if image_index >= 2:
                    raise ModelResponseError("Only front and wrist images are allowed.")
                name = ("front", "wrist")[image_index]
                image_index += 1
                url = item["image_url"]["url"]
                if not url.startswith("data:image/png;base64,"):
                    raise ModelResponseError("Expected upstream PNG data URL; no remote image fetch.")
                blob = base64.b64decode(url.split(",", 1)[1], validate=True)
                pixels = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
                (directory / f"{name}.png").write_bytes(blob)
                images.append(dict(camera=name, shape=list(pixels.shape),
                                   pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                   png_sha256=hashlib.sha256(blob).hexdigest()))
        (directory / "prompt.txt").write_text("\n\n".join(texts))
        record = dict(call_id=call_id, requested_model=self.model, status="started",
                      image_manifest=images, max_tokens=wire["max_tokens"],
                      sampling="temperature_0_seed_7_thinking_disabled",
                      transport="upstream_VLMClient_with_vllm023_boundary",
                      structured_output=wire.get("structured_outputs"))
        self.records.append(record)
        try:
            remaining = max(1, self.budget.timeout - (time.monotonic() - self.budget.started))
            response = self.session.post(
                self.base_url + "/chat/completions", json=wire,
                timeout=(min(10, remaining), min(self.timeout_s, remaining)))
            record["http_status"] = response.status_code
            if not response.ok:
                record["error_body_excerpt"] = response.text.replace(self._api_key, "[REDACTED]")[:1000]
                raise ModelResponseError(f"Local VLM HTTP {response.status_code}; no automatic retry.")
            data = response.json()
            record.update(returned_model=data.get("model"), usage=data.get("usage", {}))
            answer = data["choices"][0]
            raw = answer["message"]["content"]
            if not isinstance(raw, str):
                raise ModelResponseError("Local VLM did not return text; execute no action.")
            raw = raw.strip()
            (directory / "response.txt").write_text(raw.replace(self._api_key, "[REDACTED]"))
            record["stop_reason"] = answer.get("finish_reason")
            if answer.get("finish_reason") != "stop" or not raw:
                raise ModelResponseError("Incomplete local VLM response; execute no action.")
            if schema is not None:
                parse_json(raw, schema)  # Reject malformed output before upstream recovery.
            if choices is not None and raw not in choices:
                raise ModelResponseError("Local VLM token is not exactly an allowed action.")
            self.budget.check_time()
            record["status"] = "received"
            return data
        except requests.RequestException as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise ModelResponseError(f"Local VLM transport {type(exc).__name__}; execute nothing.") from None
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise ModelResponseError("Malformed local VLM response envelope; execute nothing.") from None
        except BaseException as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            record["latency_s"] = time.monotonic() - started
            save_json(directory / "request.json", record)

