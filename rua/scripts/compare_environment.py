"""Record relocation comparison, including non-identical rendered pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    old = json.loads((args.before / "summary.json").read_text())
    new = json.loads((args.after / "summary.json").read_text())
    report = {
        "scope": "environment_relocation_not_policy_equivalence",
        "before": str(args.before),
        "after": str(args.after),
        "environment_checks_passed": bool(old["passed"] and new["passed"]),
        "summaries_equal": old == new,
        "images": [],
    }
    for task in new["tasks"]:
        for view in ("front", "wrist"):
            relative = Path(f"task_{task['task_id']}/{view}.png")
            previous, current = args.before / relative, args.after / relative
            a = np.array(Image.open(previous)).astype(np.int16)
            b = np.array(Image.open(current)).astype(np.int16)
            assert a.shape == b.shape
            delta = np.abs(a - b)
            report["images"].append({
                "image": str(relative),
                "shape": list(a.shape),
                "before_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
                "after_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
                "changed_pixels": int(np.count_nonzero(np.any(delta, axis=2))),
                "mean_abs_channel_diff_0_255": float(delta.mean()),
                "max_abs_channel_diff_0_255": int(delta.max()),
            })
    report["pixels_identical"] = all(i["changed_pixels"] == 0 for i in report["images"])
    report["note"] = (
        "Pixel differences are measured, not explained by this comparison. "
        "No WLA inference or policy-equivalence claim."
    )
    out = Path(tempfile.mkdtemp(prefix="relocation-comparison-", dir=ROOT / "artifacts/wla-stage1"))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out / "report.json"), "summaries_equal": report["summaries_equal"],
                      "pixels_identical": report["pixels_identical"]}))


if __name__ == "__main__":
    main()
