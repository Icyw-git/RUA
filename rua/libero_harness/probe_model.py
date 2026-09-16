"""Test selected backend image order and parsing without an environment/action."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .budget import Budget
from .clients import make_client
from .paths import ARTIFACTS, configs, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("claude", "qwen"))
    args = parser.parse_args()
    _, cfg = configs(args.backend)
    out = Path(tempfile.mkdtemp(prefix=cfg["backend"] + "-probe-", dir=ARTIFACTS))
    client = make_client(cfg, Budget(max_requests=3, timeout=150), out / "requests")
    red = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    blue = np.full((64, 64, 3), [0, 0, 255], dtype=np.uint8)
    report = dict(status="running", backend=cfg["backend"], requested_model=cfg["model"],
                  scope="synthetic_two_image_model_protocol_probe_no_robot")
    try:
        response = client.complete_json(
            'Identify the solid color of each attached image. Return {"first":"COLOR","second":"COLOR"}.',
            red, wrist_image=blue,
            schema={"type": "object", "properties": {"first": {"type": "string"}, "second": {"type": "string"}},
                    "required": ["first", "second"], "additionalProperties": False}, max_tokens=128,
        )
        assert response.payload["json"] == {"first": "red", "second": "blue"}
        text = client.complete_text("Name the solid color of the first image. Reply with one lowercase color word.",
                                    red, wrist_image=blue, max_tokens=64)
        assert text.raw_text == "red"
        token = client.complete_token(
            "This is an API output-format test, with no robot connected. "
            "If the two images have different solid colors, output DONE; otherwise output MV_UP. "
            "Output only that single label.", ("DONE", "MV_UP"), red, wrist_image=blue)
        assert token.token == "DONE"
        report.update(status="passed", colors=response.payload["json"],
                      returned_model=client.records[0].get("returned_model"),
                      request_count=client.budget.requests)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        report["requests"] = client.records
        save_json(out / "report.json", report)
        print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
