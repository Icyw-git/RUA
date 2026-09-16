"""Offline component/cache verification. Never instantiate WLA or initialize CUDA."""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bootstrap import ROOT, configure

configure()

import torch
from diffusers import SanaTransformer2DModel
from diffusers.models import AutoencoderDC
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from models.wla import WLAConfig
from prepare_models import REPOS, validate_dependencies


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="models-offline-check-", dir=ROOT / "artifacts/wla-stage1"))
    report = {"status": "checking", "scope": "offline_components_no_wla_inference", "models": []}
    try:
        lock = json.loads((ROOT / "models/wla-models.lock.json").read_text())
        assert lock["status"] == "download_complete"
        assert [m["repo_id"] for m in lock["models"]] == list(REPOS)
        for model in lock["models"]:
            snapshot = Path(model["snapshot_path"])
            assert snapshot.resolve().is_relative_to((ROOT / "models").resolve())
            assert snapshot.name == model["revision"]
            assert (snapshot.parent.parent / "refs/main").read_text().strip() == model["revision"]
            headers = {}
            for file in model["files"]:
                path = snapshot / file["name"]
                assert path.is_file() and path.stat().st_size == file["size"]
                assert file["verified_sha256"]
                if file["name"].endswith(".safetensors"):
                    with safe_open(str(path), framework="pt", device="cpu") as tensors:
                        headers[file["name"]] = set(tensors.keys())
            assert headers
            for file in model["files"]:
                if file["name"].endswith(".safetensors.index.json"):
                    index = json.loads((snapshot / file["name"]).read_text())
                    for name, shard in index["weight_map"].items():
                        assert name in headers[shard], (name, shard)
                    assert set(index["weight_map"]) == set().union(*headers.values())
            report["models"].append({
                "repo_id": model["repo_id"], "revision": model["revision"],
                "weight_files": {name: len(keys) for name, keys in headers.items()},
                "offline_ref_verified": True,
            })

        config = WLAConfig.from_pretrained(REPOS[0], local_files_only=True)
        validate_dependencies(config.to_dict())
        assert config.max_state_dim == 8 and config.max_action_dim == 7 and config.chunk_size == 8
        assert config.use_history_obs and config.history_obs_step == 8
        backbone = AutoConfig.from_pretrained(REPOS[1], local_files_only=True)
        # Keep the processor arguments used by the native WLA constructor.
        processor = AutoProcessor.from_pretrained(
            REPOS[1], local_files_only=True, min_pixels=224 * 224, max_pixels=960 * 24 * 24
        )
        vae = AutoencoderDC.load_config(REPOS[2], subfolder="vae", local_files_only=True)
        transformer = SanaTransformer2DModel.load_config(
            REPOS[2], subfolder="transformer", local_files_only=True
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPOS[2], subfolder="scheduler", local_files_only=True
        )
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            REPOS[2], subfolder="scheduler", local_files_only=True
        )
        report["components"] = {
            "wla_config": config.model_type, "backbone_config": backbone.model_type,
            "processor_class": type(processor).__name__,
            "vae_config_class": vae.get("_class_name"),
            "transformer_config_class": transformer.get("_class_name"),
            "noise_scheduler_class": type(noise_scheduler).__name__,
            "scheduler_class": type(scheduler).__name__,
        }
        assert not torch.cuda.is_initialized()
        report.update(status="passed", cuda_initialized=False)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (out / "error.txt").write_text(traceback.format_exc())
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
