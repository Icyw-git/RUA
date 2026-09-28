"""Build a visual review packet and turn explicit decisions into review.jsonl."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from math import dist
from pathlib import Path

import imageio.v2 as imageio

from .training_dump import selected_ranges, write_jsonl


EVAL_SCOPE = "libero_pro_single_smoke"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def oracle_by_source(roots: list[Path]) -> dict[str, dict[int, dict]]:
    found = {}
    for root in roots:
        manifests = [root] if root.name == "oracle-manifest.json" else root.rglob("oracle-manifest.json")
        for path in manifests:
            manifest = json.loads(path.read_text())
            source = manifest["source"]
            trace = Path(source) / "environment-steps.jsonl"
            if hashlib.sha256(trace.read_bytes()).hexdigest() != manifest["source_trace_sha256"]:
                raise ValueError(f"Oracle feedback no longer matches source: {source}")
            found[source] = {row["task_step"]: row for row in
                             read_jsonl(path.parent / manifest["feedback_file"])}
    return found


def evidence_steps(proposal: dict) -> list[int]:
    count = proposal["task_steps"]
    if not count:
        return []
    steps = {0, count // 2, count - 1}
    for signal in proposal["signals"]:
        steps.update(item["task_step"] for item in signal["evidence"])
    return sorted(steps)


def build(proposals_path: Path, output: Path, oracle_roots: list[Path]) -> dict:
    proposals = read_jsonl(proposals_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    oracles = oracle_by_source(oracle_roots)
    output.mkdir(parents=True, exist_ok=True)
    image_root = output / "evidence"
    image_root.mkdir()
    queue = []
    templates = []
    for index, proposal in enumerate(proposals):
        source = Path(proposal["source"])
        trace_file = source / "environment-steps.jsonl"
        if hashlib.sha256(trace_file.read_bytes()).hexdigest() != proposal["source_trace_sha256"]:
            raise ValueError(f"Screening proposal no longer matches source: {source}")
        actions = [row for row in read_jsonl(trace_file)
                   if row.get("event") == "step_completed" and row.get("phase") == "task"]
        samples = []
        readers = {camera: imageio.get_reader(source / f"{camera}-control.mp4")
                   for camera in ("front", "wrist")}
        try:
            for step in evidence_steps(proposal):
                frame = actions[step]["attempt"]
                paths = {}
                for camera, reader in readers.items():
                    for moment, video_frame in (("before", frame), ("after", frame + 1)):
                        relative = (Path("evidence") /
                                    f"episode-{index:03d}-{camera}-step-{step:03d}-{moment}.png")
                        imageio.imwrite(output / relative, reader.get_data(video_frame))
                        paths[f"{camera}_{moment}_image"] = str(relative)
                oracle = oracles.get(str(source), {}).get(step)
                samples.append({"task_step": step, "video_frame_before_action": frame,
                                "video_frame_after_action": frame + 1,
                                **paths,
                                "goal_predicates_after_action":
                                oracle["goal_predicates"] if oracle else None,
                                "object_positions_after_action_m":
                                oracle["object_positions_m"] if oracle else None})
        finally:
            for reader in readers.values():
                reader.close()
        queue.append({
            "source": str(source), "scope": proposal.get("scope"),
            "front_video": str(source / "front-control.mp4"),
            "wrist_video": str(source / "wrist-control.mp4"),
            "suite": proposal["suite"], "task_id": proposal["task_id"],
            "task_instruction": proposal["task_instruction"],
            "official_success": proposal["official_success"],
            "task_steps": proposal["task_steps"],
            "eligible_for_wla_review": proposal["eligible_for_wla_review"],
            "signals": proposal["signals"], "evidence": samples,
            "oracle_available": str(source) in oracles,
        })
        templates.append({"source": str(source),
                          "decision": "validation_only" if proposal.get("scope") == EVAL_SCOPE
                          else "pending", "reason": ""})
    write_jsonl(output / "review-queue.jsonl", queue)
    write_jsonl(output / "review-decisions.template.jsonl", templates)
    (output / "index.html").write_text(render_html(queue), encoding="utf-8")
    return {"episodes": len(queue), "with_oracle": sum(row["oracle_available"] for row in queue),
            "output": str(output)}


def render_html(rows: list[dict]) -> str:
    sections = []
    for index, row in enumerate(rows):
        title = f"{row['suite']} / task {row['task_id']} / {Path(row['source']).name}"
        cards = []
        for sample in row["evidence"]:
            goal = sample["goal_predicates_after_action"]
            goal_text = ("目标条件：" + ", ".join(
                f"{item['predicate']}({', '.join(item['objects'])})={item['satisfied']}"
                for item in goal)) if goal is not None else "未采集逐步仿真反馈"
            positions = sample["object_positions_after_action_m"] or {}
            distance_text = ""
            if goal and len(goal[0]["objects"]) == 2:
                first, second = goal[0]["objects"]
                if first in positions and second in positions:
                    distance_text = f" · 两物体中心水平距离 {dist(positions[first][:2], positions[second][:2]):.3f} m"
            views = []
            for camera, camera_label in (("front", "前视"), ("wrist", "腕视")):
                for moment, moment_label in (("before", "动作前"), ("after", "动作后")):
                    label = f"{camera_label} · {moment_label}"
                    path = html.escape(sample[f"{camera}_{moment}_image"], quote=True)
                    views.append(f"<figure><figcaption>{label}</figcaption>"
                                 f"<img src='{path}' alt='{label}'></figure>")
            cards.append(
                f"<div class='card' data-task-step='{sample['task_step']}'>"
                f"<strong>任务动作 {sample['task_step']}</strong> "
                f"<small>视频帧 {sample['video_frame_before_action']} → "
                f"{sample['video_frame_after_action']}</small>"
                f"<div class='views'>{''.join(views)}</div>"
                f"<small>{html.escape(goal_text + distance_text)}</small>"
                "<label>动作判断 <select class='step-label'><option value='unreviewed'>未判断</option>"
                "<option value='smooth'>顺利</option><option value='recovery'>纠正/恢复</option>"
                "<option value='error'>错误</option><option value='unclear'>看不清</option>"
                "</select></label></div>"
            )
        signals = ", ".join(str(item["transition_steps"]) for item in row["signals"]) or "无"
        signal_controls = "".join(
            f"<label class='signal' data-signal-index='{signal_index}'>夹爪变化 "
            f"{html.escape(str(signal['transition_steps']))} "
            "<select class='signal-label'><option value='unreviewed'>未判断</option>"
            "<option value='normal'>正常操作</option><option value='recovery'>纠正/恢复</option>"
            "<option value='error'>错误</option><option value='unclear'>看不清</option>"
            "</select></label> "
            for signal_index, signal in enumerate(row["signals"])
        )
        eval_selected = " selected" if row.get("scope") == EVAL_SCOPE else ""
        sections.append(
            f"<section data-source='{html.escape(row['source'], quote=True)}'>"
            f"<h2>{html.escape(title)}</h2><p>{html.escape(row['task_instruction'] or '')}</p>"
            f"<p>官方成功：{row['official_success']} · 动作数：{row['task_steps']} · "
            f"夹爪线索：{html.escape(signals)}</p>"
            f"<p><a href='{html.escape(Path(row['front_video']).as_uri(), quote=True)}' "
            "target='_blank'>完整前视视频</a> · "
            f"<a href='{html.escape(Path(row['wrist_video']).as_uri(), quote=True)}' "
            "target='_blank'>完整腕视视频</a></p>"
            f"<div>{signal_controls}</div>"
            "<label>决定 <select class='decision'><option value='pending'>待审核</option>"
            f"<option value='validation_only'{eval_selected}>仅验证</option>"
            "<option value='keep_full'>保留整条</option>"
            "<option value='keep_ranges'>保留指定区间</option><option value='reject'>不纳入</option>"
            "</select></label> "
            "<label>区间 <input class='ranges' placeholder='[[0,30],[45,70]]'></label> "
            "<label>理由 <input class='reason' placeholder='简要说明'></label>"
            f"<div class='cards'>{''.join(cards)}</div></section>"
        )
    script = """
    function downloadReview() {
      const lines = [...document.querySelectorAll('section[data-source]')].map(section => {
        const row = {source: section.dataset.source,
                     decision: section.querySelector('.decision').value,
                     reason: section.querySelector('.reason').value};
        const ranges = section.querySelector('.ranges').value.trim();
        if (ranges) row.task_step_ranges = JSON.parse(ranges);
        row.signal_labels = [...section.querySelectorAll('.signal')].map(item => ({
          signal_index: Number(item.dataset.signalIndex),
          label: item.querySelector('.signal-label').value
        })).filter(item => item.label !== 'unreviewed');
        row.evidence_labels = [...section.querySelectorAll('.card')].map(item => ({
          task_step: Number(item.dataset.taskStep),
          label: item.querySelector('.step-label').value
        })).filter(item => item.label !== 'unreviewed');
        return JSON.stringify(row);
      });
      const file = new Blob([lines.join('\\n') + '\\n'], {type: 'application/jsonl'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(file);
      link.download = 'review-decisions.jsonl';
      link.click();
      URL.revokeObjectURL(link.href);
    }
    """
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'>"
            "<title>RUA 轨迹审核</title><style>body{font:16px sans-serif;max-width:1200px;"
            "margin:auto;padding:24px;background:#f6f7f9;color:#20242a}section{background:white;"
            "padding:20px;margin:20px 0;border-radius:12px}input,select,button{padding:8px}"
            ".cards{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}.card{width:620px;"
            "border:1px solid #ddd;padding:8px}.views{display:grid;grid-template-columns:1fr 1fr;"
            "gap:8px}.views figure{margin:0}.views img{width:100%}"
            "small{display:block;color:#555}</style><h1>RUA 轨迹审核</h1>"
            "<p>每张卡片对照同一动作的前后画面；目标条件是该动作后的仿真结果。夹爪线索仅供复核。"
            "评测轨迹请选择“仅验证”。</p><button onclick='downloadReview()'>下载审核决定</button>"
            + "".join(sections) + f"<script>{script}</script></html>")


def finalize(proposals_path: Path, decisions_path: Path, output: Path) -> list[dict]:
    proposals = {row["source"]: row for row in read_jsonl(proposals_path)}
    decisions = read_jsonl(decisions_path)
    by_source = {row["source"]: row for row in decisions}
    if len(by_source) != len(decisions) or set(by_source) != set(proposals):
        raise ValueError("Decisions must contain each proposed source exactly once")
    rows = []
    for source, proposal in proposals.items():
        decision = by_source[source]
        choice = decision["decision"]
        reason = decision.get("reason", "")
        if choice in ("reject", "validation_only"):
            rows.append({"source": source, "approved_for_wla": False,
                         "reason": reason or choice})
        elif choice in ("keep_full", "keep_ranges"):
            if proposal.get("scope") == EVAL_SCOPE:
                raise ValueError(f"Evaluation source cannot be approved for training: {source}")
            if not proposal["eligible_for_wla_review"]:
                raise ValueError(f"Source is not eligible for WLA review: {source}")
            row = {"source": source, "approved_for_wla": True}
            if choice == "keep_ranges":
                ranges = decision["task_step_ranges"]
                selected_ranges({"task_step_ranges": ranges}, proposal["task_steps"])
                row["task_step_ranges"] = ranges
            if reason:
                row["reason"] = reason
            rows.append(row)
        else:
            raise ValueError(f"Unfinished or unknown review decision for {source}: {choice}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    write_jsonl(output.with_name("review-labels.jsonl"),
                ({"source": source,
                  "signal_labels": by_source[source].get("signal_labels", []),
                  "evidence_labels": by_source[source].get("evidence_labels", [])}
                 for source in proposals))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("proposals", type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--oracle-root", action="append", default=[], type=Path)
    final_parser = commands.add_parser("finalize")
    final_parser.add_argument("proposals", type=Path)
    final_parser.add_argument("decisions", type=Path)
    final_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.proposals, args.output, args.oracle_root), indent=2))
    else:
        print(json.dumps({"review": str(args.output),
                          "episodes": len(finalize(args.proposals, args.decisions, args.output))},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
