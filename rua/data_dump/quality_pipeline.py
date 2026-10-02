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
from .training_dump import discover, episode_steps, export, read_source, write_jsonl


NORMAL_REJECTIONS = {"task_not_successful", "review_rejected", "quality_no_training_starts"}


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
    episodes = discover(sources, include_incomplete=True)
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
    source_failures = {}
    for index, source in enumerate(episodes):
        oracle_dir = oracle_root / f"{index:04d}"
        stage = "source_invalid"
        try:
            result, events, read_error = read_source(source)
            if read_error:
                raise ValueError(read_error)
            episode_steps(source, result, events, require_success=False)
            cohort = (result.get("scope"), result["pilot"]["suite"],
                      result["task_id"], result.get("initial_state_id"))
            success = result["official_success"]
            stage = "replay_failed"
            report = replay(source, oracle_dir)
            stage = "classification_failed"
            source_labels = classify(source, oracle_dir)
            rows = [json.loads(line) for line in
                    (oracle_dir / report["feedback_file"]).read_text().splitlines()]
            positions = [row["eef_position_m"] for row in rows]
            metric = {
                "source": str(source), "success": success, "cohort": cohort,
                "steps": len(rows),
                "eef_path_m": sum(dist(a, b) for a, b in zip(positions, positions[1:])),
            }
        except (FileNotFoundError, ValueError, KeyError) as exc:
            source_failures[str(source)] = {"rejection_code": stage,
                                            "rejection_reason": str(exc)}
            continue
        oracle_reports[str(source)] = report
        labels.extend(source_labels)
        episode_metrics.append(metric)
    annotate_efficiency(labels, episode_metrics)
    signals = []
    for index, source in enumerate(episodes):
        if str(source) in source_failures:
            continue
        source_labels = [label for label in labels if label["source"] == str(source)]
        report = oracle_reports[str(source)]
        rows = [json.loads(line) for line in
                (oracle_root / f"{index:04d}" / report["feedback_file"]).read_text().splitlines()]
        source_signals = behavior_signals(
            source_labels, report["object_horizontal_radius_m"],
            set(report["target_objects"]), rows)
        signals.extend(source_signals)
    label_path = output_root / "quality-labels.jsonl"
    write_jsonl(label_path, labels)
    write_jsonl(output_root / "behavior-signals.jsonl", signals)
    manifest = export(episodes, output_root / "dump", dataset_name,
                      quality_labels=label_path, source_failures=source_failures)
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
    feedback = [json.loads(line) for line in
                (output_root / "dump/feedback.jsonl").read_text().splitlines()]
    summary = {"sources": len(episodes), "segments": len(labels),
               "exported_episodes": len(manifest["episodes"]),
               "preprocessing_failed_sources": len(source_failures),
               "export_failed_sources": sum(
                   row["source"] not in source_failures
                   and row["rejection_code"] is not None
                   and row["rejection_code"] not in NORMAL_REJECTIONS for row in feedback),
               "rejected_sources": len(episodes) - len(manifest["episodes"]),
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
    summary = run(args.sources, args.output_root, args.dataset_name)
    print(json.dumps(summary, indent=2))
    return 2 if (summary["preprocessing_failed_sources"]
                 or summary.get("export_failed_sources", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
