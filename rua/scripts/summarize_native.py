"""Generate a compact human-readable receipt from completed, audited measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(directory):
    report = json.loads((directory / "report.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    assert report["status"] == "completed" and audit["status"] == "passed"
    cfg, results, summary = report["config"], report["results"], report["summary"]
    lines = [
        "# 原生 WLA × LIBERO 小样本结果",
        "",
        f"运行目录：`{directory}`",
        f"启动时间：`{report['started_at']}`（服务器时区）",
        "",
        f"- 固定样本：`{cfg['suite']}`，任务 0、1、2 × 初始状态 1–5。",
        f"- 官方成功：**{summary['successes']}/{summary['planned_episodes']} "
        f"({summary['success_rate']:.1%})**；运行错误 {summary['runtime_errors']}。",
        f"- 阶段 1 工程验收：{'通过' if summary['stage1_gate_passed'] else '未通过'}。",
        f"- 总运行时间：{report['elapsed_seconds']:.2f} 秒；"
        f"episode 用时合计 {sum(r['total_case_seconds'] for r in results):.2f} 秒。",
        f"- 任务控制步合计：{sum(r['task_control_steps'] for r in results)}；"
        f"初始化等待步合计：{sum(r['initialization_steps'] for r in results)}。",
        f"- WLA 预测调用：{sum(r['wla_calls'] for r in results)}；外部 agent 请求/token：0/0。",
        f"- PyTorch 已分配显存峰值：{report['peak_torch_allocated_bytes'] / 1024**3:.3f} GiB；"
        f"预留峰值：{report['peak_torch_reserved_bytes'] / 1024**3:.3f} GiB。",
        "- 逐步动作、历史图像、官方评分记录和双视图视频完整性审计：通过。",
        "",
        "## 逐任务",
        "",
        "| 任务 | 原始任务描述 | 成功 | 平均任务步 | 平均秒数 |",
        "|---|---|---:|---:|---:|",
    ]
    for task in cfg["task_ids"]:
        rows = [r for r in results if r["task_id"] == task]
        lines.append(
            f"| {task} | {rows[0]['instruction']} | "
            f"{sum(r['official_success'] for r in rows)}/{len(rows)} | "
            f"{sum(r['task_control_steps'] for r in rows) / len(rows):.1f} | "
            f"{sum(r['total_case_seconds'] for r in rows) / len(rows):.2f} |"
        )
    lines += [
        "",
        "## 全部固定样本",
        "",
        "| 任务 | 初始状态 | 官方成功 | 任务步 | WLA 调用 | 秒数 | 停止原因 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['task_id']} | {r['init_state_id']} | {'是' if r['official_success'] else '否'} | "
            f"{r['task_control_steps']} | {r['wla_calls']} | {r['total_case_seconds']:.2f} | {r['stop_reason']} |"
        )
    lines += [
        "",
        "## 边界与复查",
        "",
        "这是 15 次工程验证，不是完整 LIBERO benchmark，也不是论文成绩复现。",
        "调试任务 0 / 状态 0 的成功独立保存，不计入上表。没有训练或调参，",
        "没有 Show-Harness、Claude、外部 Qwen 或混合控制参与。",
        "",
        f"- WLA 提交：`{cfg['wla_commit']}`。",
        f"- LIBERO 提交：`{cfg['libero_commit']}`。",
        f"- checkpoint：`{cfg['model_id']}`，归一化键 `{cfg['unnorm_key']}`。",
        f"- 模型锁 SHA256：`{report['model_lock_sha256']}`。",
        f"- 模型种子 {cfg['model_seed']}、环境种子 {cfg['environment_seed']}；"
        f"每次 {cfg['max_env_steps']} 个任务步＋{cfg['settle_steps']} 个等待步。",
        "- 模型使用原生完整指令、双视图＋历史前视、8 维本体状态，动作队列和夹爪转换保持原生协议。",
        "- `report.json` 保存完整配置、逐 episode 结果及资源；`audit.json` 保存完整性审计。",
        "- `provenance/` 保存源码、依赖、模型锁、归一化统计和任务定义快照。",
        "- 每个 `task-N_init-M/` 保存 `steps.jsonl`、预测输入/输出、`front.mp4`、`wrist.mp4`。",
        "- 20 fps 视频表示模拟器时间，实际墙钟时间以上表为准。",
        "",
    ]
    output = directory / "RESULTS.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory.resolve())
