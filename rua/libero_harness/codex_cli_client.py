"""Codex CLI transport for the existing Show-Harness model interface."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.vlm.vlm_client import VLMResponse
from .claude import ModelResponseError, parse_json
from .codex_executable import resolve_codex_binary
from .paths import save_json


class CodexCLIClient:
    """One isolated, read-only Codex invocation per Show-Harness request."""

    strict_outputs = True

    def __init__(self, cfg, budget, trace_dir):
        self.model = cfg["model"]
        self.budget = budget
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True)
        self.binary = str(resolve_codex_binary(cfg.get("codex_binary")))
        self.records = []

    def _request(self, prompt, agentview_image, wrist_image=None, *, schema=None):
        self.budget.before_request()
        call_id = self.budget.requests
        directory = self.trace_dir / f"request-{call_id:03d}"
        directory.mkdir()
        images = []
        for camera, pixels in (("front", agentview_image), ("wrist", wrist_image)):
            if pixels is None:
                continue
            array = np.asarray(pixels)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError("Codex expects RGB uint8 images")
            image_path = directory / f"{camera}.png"
            Image.fromarray(array).save(image_path)
            images.append({"camera": camera, "shape": list(array.shape),
                           "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                           "png_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
        if [image["camera"] for image in images] != ["front", "wrist"]:
            raise ValueError("Show-Harness Codex smoke requires front and wrist images")
        full_prompt = self.budget.describe() + "\n\n" + prompt
        if schema is not None:
            full_prompt += "\n\nReturn only a JSON object matching this schema:\n" + json.dumps(schema)
        full_prompt += "\n\nDo not call tools, read files, or modify files. Respond directly."
        (directory / "prompt.txt").write_text(full_prompt)
        output_path = directory / "response.txt"
        record = {"call_id": call_id, "requested_model": self.model,
                  "image_manifest": images, "status": "started",
                  "transport": "codex_exec_read_only", "schema_requested": schema is not None}
        self.records.append(record)
        started = time.monotonic()
        try:
            remaining = min(90, self.budget.timeout - (started - self.budget.started))
            if remaining <= 0:
                raise ModelResponseError("No time remains for Codex request")
            completed = subprocess.run(
                [self.binary, "exec", "--ephemeral", "--ignore-user-config",
                 "--skip-git-repo-check", "--sandbox", "read-only",
                 "--model", self.model, "--cd", str(directory),
                 "--image", str(directory / "front.png"), str(directory / "wrist.png"),
                 "--output-last-message", str(output_path), full_prompt],
                cwd=directory, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, timeout=remaining,
            )
            (directory / "codex-stdout.txt").write_text(completed.stdout)
            (directory / "codex-stderr.txt").write_text(completed.stderr)
            if completed.returncode != 0 or not output_path.is_file():
                raise ModelResponseError(f"Codex exited with status {completed.returncode}")
            raw = output_path.read_text().strip()
            if not raw:
                raise ModelResponseError("Codex returned empty text")
            self.budget.check_time()
            record.update(status="received", returned_model=self.model)
            return VLMResponse(token="", raw_text=raw,
                               payload={"request_id": call_id, "returned_model": self.model,
                                        "latency_s": time.monotonic() - started})
        except subprocess.TimeoutExpired as exc:
            record.update(status="error", error_type="TimeoutExpired")
            raise ModelResponseError("Codex request timed out") from exc
        except BaseException as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            record["latency_s"] = time.monotonic() - started
            save_json(directory / "request.json", record)

    def complete_json(self, prompt, agentview_image, wrist_image=None, schema=None, **kwargs):
        response = self._request(prompt, agentview_image, wrist_image, schema=schema)
        response.payload["json"] = parse_json(response.raw_text, schema)
        return response

    def complete_text(self, prompt, agentview_image, wrist_image=None, **kwargs):
        return self._request(prompt, agentview_image, wrist_image)

    def complete_token(self, prompt, allowed_tokens, agentview_image,
                       wrist_image=None, **kwargs):
        response = self._request(prompt, agentview_image, wrist_image)
        if response.raw_text not in allowed_tokens:
            raise ModelResponseError("Codex did not return one allowed token")
        return VLMResponse(token=response.raw_text, raw_text=response.raw_text,
                           payload=response.payload)
