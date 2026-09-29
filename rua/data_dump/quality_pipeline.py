"""Replay episodes, label physical events, and dump verified WLA starts."""
from __future__ import annotations

import argparse
import json
from math import dist
from pathlib import Path
from statistics import median

from libero_harness.paths import save_json
from .oracle_feedback import replay
from .quality_labels import classify
from .quality_signals import behavior_signals, review_starts, signal_counts
from .training_dump import discover, export, write_jsonl


def annotate_efficiency(labels: list[dict], episode_metrics: list[dict]) -> None:
    """Add descriptive cohort ratios; they never change training selection."""
    for current in episode_metrics:
        references = [other for other in episode_metrics
                      if other is not current and other["success"]
                      and current["success"] and other["cohort"] == current["cohort"]]
        if len(references) < 3:
            continue
        speed_ratio = current["steps"] / median(other["steps"] for other in references)
        path_baseline = median(other["eef_path_m"] for other in references)
        path_ratio = current["eef_path_m"] / path_baseline if path_baseline else None
        for label in labels:
            if label["source"] == current["source"]:
                label["efficiency"].update({"speed_ratio": round(speed_ratio, 3),
                                            "path_ratio": round(path_ratio, 3)
                                            if path_ratio is not None else None,
                                            "reference_count": len(references)})


def run(sources: list[Path], output_root: Path, dataset_name: str = "rua_lerobot") -> dict:
    episodes = discover(sources)
    if not episodes:
        raise ValueError("No saved episodes found")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    oracle_root = output_root / "oracle"
    oracle_root.mkdir()
    labels = []
    episode_metrics = []
    oracle_reports = {}
    for index, source in enumerate(episodes):
        oracle_dir = oracle_root / f"{index:04d}"
        oracle_reports[str(source)] = replay(source, oracle_dir)
        labels.extend(classify(source, oracle_dir))
        rows = [json.loads(line) for line in
                (oracle_dir / "oracle-feedback.jsonl").read_text().splitlines()]
        result = json.loads((source / "result.json").read_text())
        positions = [row["eef_position_m"] for row in rows]
        episode_metrics.append({
            "source": str(source), "success": result["official_success"],
            "cohort": (result.get("scope"), result["pilot"]["suite"],
                       result["task_id"], result.get("initial_state_id")),
            "steps": len(rows),
            "eef_path_m": sum(dist(a, b) for a, b in zip(positions, positions[1:])),
        })
    annotate_efficiency(labels, episode_metrics)
    signals = []
    for source in episodes:
        source_labels = [label for label in labels if label["source"] == str(source)]
        report = oracle_reports[str(source)]
        source_signals = behavior_signals(
            source_labels, report["object_horizontal_radius_m"],
            set(report["target_objects"]))
        signals.extend(source_signals)
    label_path = output_root / "quality-labels.jsonl"
    write_jsonl(label_path, labels)
    write_jsonl(output_root / "behavior-signals.jsonl", signals)
    manifest = export(episodes, output_root / "dump", dataset_name,
                      quality_labels=label_path)
    review = []
    if manifest["episodes"]:
        index_path = output_root / "dump" / dataset_name / "meta/quality-starts.jsonl"
        starts = [json.loads(line) for line in index_path.read_text().splitlines()]
        for episode in manifest["episodes"]:
            source = episode["source"]
            episode_starts = [start for start in starts
                              if start["episode_index"] == episode["episode_index"]]
            review.extend(review_starts(episode_starts, signals, source))
    write_jsonl(output_root / "review-starts.jsonl", review)
    summary = {"sources": len(episodes), "segments": len(labels),
               "exported_episodes": len(manifest["episodes"]),
               "training_starts": manifest.get("quality_starts", 0),
               "nominal_starts": manifest.get("quality_nominal_starts", 0),
               "recovery_starts": manifest.get("quality_recovery_starts", 0),
               "review_starts": len(review),
               "behavior_signal_counts": signal_counts(signals),
               "signal_mode": "shadow",
               "dump": str(output_root / "dump")}
    save_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-name", default="rua_lerobot")
    args = parser.parse_args()
    print(json.dumps(run(args.sources, args.output_root, args.dataset_name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
