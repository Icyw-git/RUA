"""Compare two completed calibrations; never executes simulator/model actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import configs, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_report", type=Path)
    args = parser.parse_args()
    _, config = configs()
    baseline_path = Path(config["calibration_receipt"])
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(args.calibration_report.read_text())
    assert baseline["status"] == candidate["status"] == "passed"
    differences = {}
    for key in ("selected_amplitude", "move_env_steps", "gripper_env_steps",
                "nominal_step_m", "open_close_open_widths_m",
                "empty_width_m", "open_width_m", "effective_table_height_m"):
        differences[key] = float(np.max(np.abs(np.asarray(candidate[key]) - np.asarray(baseline[key]))))
        assert differences[key] < 1e-9, (key, differences[key])
    assert baseline["suggested_token_vectors"] == candidate["suggested_token_vectors"]
    assert baseline["wrist_extra_flip"] == candidate["wrist_extra_flip"]
    reference_probes = baseline["probes"]
    if candidate.get("verify_selected_only"):
        reference_probes = [p for p in reference_probes
                            if p["amplitude"] == config["move_amplitude"]
                            and p["steps"] == config["move_env_steps"]]
        assert len(reference_probes) == len(candidate["probes"]) == 6
    else:
        assert len(reference_probes) == len(candidate["probes"]) == 48
    probe_delta = max(
        float(np.max(np.abs(np.asarray(a["delta_m"]) - np.asarray(b["delta_m"]))))
        for a, b in zip(reference_probes, candidate["probes"])
    )
    assert probe_delta < 1e-9, probe_delta
    images = {}
    for name in ("agentview", "robot0_eye_in_hand"):
        a_path = baseline_path.parent / (name + ".png")
        b_path = args.calibration_report.parent / (name + ".png")
        a, b = np.array(Image.open(a_path)), np.array(Image.open(b_path))
        assert a.shape == b.shape == (512, 512, 3)
        assert float(b.std()) > 5, "Candidate image is effectively blank"
        images[name] = dict(mean_absolute_pixel_difference=float(np.abs(a.astype(float) - b).mean()),
                            candidate_std=float(b.std()),
                            candidate_png_sha256=hashlib.sha256(b_path.read_bytes()).hexdigest())
    result = dict(status="passed", baseline=str(baseline_path), candidate=str(args.calibration_report),
                  numeric_differences=differences, max_probe_delta_difference_m=probe_delta,
                  compared_direction_probes=len(reference_probes),
                  images=images, pixel_equivalence_claimed=False,
                  note="Control/camera calibration equivalence, not policy or task equivalence.")
    save_json(args.calibration_report.parent / "renderer-comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
