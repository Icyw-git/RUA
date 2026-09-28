import json

import imageio.v2 as imageio
import pytest

from data_dump.quality_screen import screen
from data_dump.review_packet import build, finalize
from test_training_dump import make_episode


def test_review_packet_and_finalization(tmp_path):
    training = make_episode(tmp_path, "training", task_steps=20)
    evaluation = make_episode(tmp_path, "evaluation", task_steps=12)
    for source, scope in ((training, "train_collection"),
                          (evaluation, "libero_pro_single_smoke")):
        path = source / "result.json"
        result = json.loads(path.read_text())
        result["scope"] = scope
        path.write_text(json.dumps(result))
    proposals = tmp_path / "proposals.jsonl"
    screen([training, evaluation], proposals)

    packet = tmp_path / "packet"
    assert build(proposals, packet, []) == {
        "episodes": 2, "with_oracle": 0, "output": str(packet)}
    queue = [json.loads(line) for line in (packet / "review-queue.jsonl").read_text().splitlines()]
    assert len(queue) == 2
    assert all(len(row["evidence"]) == 3 for row in queue)
    for row in queue:
        assert row["front_video"] == str(row["source"] + "/front-control.mp4")
        last = row["evidence"][-1]
        assert last["video_frame_after_action"] == last["video_frame_before_action"] + 1
        for camera in ("front", "wrist"):
            before = imageio.imread(packet / last[f"{camera}_before_image"])
            after = imageio.imread(packet / last[f"{camera}_after_image"])
            assert after.mean() > before.mean()
    page = (packet / "index.html").read_text()
    assert "仅验证" in page
    assert "前视 · 动作前" in page and "腕视 · 动作后" in page
    assert "完整前视视频" in page and "完整腕视视频" in page
    assert "step-label" in page and "signal-label" in page

    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("\n".join(json.dumps(row) for row in [
        {"source": str(training), "decision": "keep_ranges", "task_step_ranges": [[2, 11]],
         "reason": "Reviewed segment",
         "evidence_labels": [{"task_step": 10, "label": "recovery"}]},
        {"source": str(evaluation), "decision": "validation_only"},
    ]) + "\n")
    review = tmp_path / "review.jsonl"
    rows = finalize(proposals, decisions, review)
    assert rows == [json.loads(line) for line in review.read_text().splitlines()]
    assert rows[1] == {"source": str(training), "approved_for_wla": True,
                       "task_step_ranges": [[2, 11]], "reason": "Reviewed segment"}
    assert rows[0]["approved_for_wla"] is False
    labels = [json.loads(line) for line in
              (tmp_path / "review-labels.jsonl").read_text().splitlines()]
    assert len(labels) == 2 and all(row["signal_labels"] == [] for row in labels)
    assert next(row for row in labels if row["source"] == str(training))["evidence_labels"] == [
        {"task_step": 10, "label": "recovery"}]

    decisions.write_text("\n".join(json.dumps(row) for row in [
        {"source": str(training), "decision": "keep_full"},
        {"source": str(evaluation), "decision": "keep_full"},
    ]) + "\n")
    with pytest.raises(ValueError, match="Evaluation source cannot be approved"):
        finalize(proposals, decisions, tmp_path / "unsafe-review.jsonl")
