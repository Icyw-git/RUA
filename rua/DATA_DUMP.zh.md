# RUA trace / feedback 数据导出

这项功能把一次完整运行的原始记录整理成两份用途不同、但可互相追溯的数据：

1. `trace.jsonl` + `feedback.jsonl`：保留成功和失败轨迹，供后续 agent 分析、筛选和训练。
2. `rua_lerobot/`：只收录通过审计、成功且有完整监督帧的轨迹，供 WLA 训练。

导出是离线操作，不调用模型、不执行环境动作。它读取 RUA 运行目录中的 `result.json`、`environment-steps.jsonl`、`front-control.mp4`、`wrist-control.mp4`、`requests/` 和 `native/`。原始目录要保留：`trace.jsonl` 中的图片路径指向原始目录，不复制 PNG。生成的 LeRobot 视频则在新数据集内，可单独搬运。

## 运行

```bash
cd /data1/wcz/WLA
PYTHONPATH=/data1/wcz/WLA/rua \
  /data1/wcz/conda-envs/rua-data/bin/python -m libero_harness.training_dump \
  /data1/wcz/WLA/rua/artifacts/YOUR_RUN \
  --output-root /data1/wcz/rua-dumps/dump-001
```

输入可以是单个 episode 目录或其上级运行目录，也可以传多个目录。输出目录须为空或尚不存在。`rua-data` 是 `/data1/wcz` 下单独的 Conda 环境；导出不会改共享服务或别人的 Python 环境。

若要人工筛选，准备 JSONL 文件，每行指定一个 episode 的绝对路径和布尔值：

```json
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-00","approved_for_wla":true}
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-01","approved_for_wla":false}
```

再加 `--review /data1/wcz/my-review.jsonl`。传入 review 后，未列出的 episode 也不会进入 WLA 数据集。规范化后的决定写入输出的 `review.jsonl`，成功但被否决的 episode 仍保留在 trace / feedback 中。不传 review 时，以自动审计和成功条件筛选。

## 输出目录与字段

```text
dump-001/
  dump-manifest.json                    导出版本、字段顺序、来源到 episode 的映射
  feedback.jsonl                        每个原始 episode 一行，含失败和拒绝原因
  trace.jsonl                           环境事件、agent 决策、模型请求各一行
  review.jsonl                          仅使用 --review 时生成
  rua_lerobot/                          有候选数据时生成，LeRobot v3 数据集
    meta/info.json                      fps、feature schema、路径模板、总数
    meta/episodes/chunk-000/file-000.parquet  episode 长度和数据/视频索引
    meta/tasks.parquet                  task_index 对应的完整任务指令
    meta/stats.json                     LeRobot 计算的统计值
    data/chunk-000/file-000.parquet     逐帧标量/向量和索引
    videos/observation.images.agentview/chunk-000/file-000.mp4
    videos/observation.images.wrist/chunk-000/file-000.mp4
```

LeRobot 可能随 episode 数增加而产生更多 chunk/file；实际文件名以 `meta/info.json` 的模板为准。不要按上述示例硬编码文件名。

| 字段 | 格式 | 内容、示例 | 所在位置 |
| --- | --- | --- | --- |
| `task` / `task_index` | 字符串 / 整数 | 完整指令，例如 `pick up the black bowl`；`task_index` 指向该字符串 | 字符串在 `meta/tasks.parquet`，引用索引在 `data/...parquet`；通过 LeRobot API 读取时得到 `item["task"]` |
| `observation.images.agentview` | H.264 RGB 视频帧，读取为 `[3,H,W]` | 执行 `action[t]` **之前**的前视图；WLA 取 `t-8`、`t`，`image_action` 还取 `t+8` | `videos/observation.images.agentview/...mp4`，时间戳/路径由 LeRobot 元数据关联 |
| `observation.images.wrist` | H.264 RGB 视频帧，读取为 `[3,H,W]` | 执行 `action[t]` 前的腕视图 `t` | `videos/observation.images.wrist/...mp4` |
| `observation.state` | `float32[8]` | `[eef_x,eef_y,eef_z,axis_angle_x,axis_angle_y,axis_angle_z,gripper_qpos_0,gripper_qpos_1]`；例如 `[0.2,0.1,0.8,0,0,0,0.04,-0.04]` | `data/...parquet` 的一列；逐帧保存，WLA 在 `t` 读它 |
| `action` | `float32[7]` | `[delta_x,delta_y,delta_z,delta_rx,delta_ry,delta_rz,gripper_training]`；例如 `[0,0,0.03,0,0,0,1]` | `data/...parquet` 的一列；逐帧保存，WLA 读取 `t:t+8` 成 `[8,7]` |
| `timestamp` | 浮点秒 | 20 fps 时第 20 帧约 `1.0` 秒 | `data/...parquet`；LeRobot `add_frame` 自动生成 |
| `frame_index`、`episode_index`、`index` | 整数 | episode 内帧号、episode 号、数据集全局帧号；如 `20`、`7`、`912` | `data/...parquet`；LeRobot 自动生成 |
| `task_index` | 整数 | 指向任务表，例如 `3` | `data/...parquet`；LeRobot 根据传入的 `task` 自动生成 |

写入接口只接收每帧的双图、8 维状态、7 维动作和完整 `task` 字符串。LeRobot 的 `add_frame` / `save_episode` 生成索引、时间戳、任务表、Parquet、视频和元数据。`action[t:t+8]` 与 `agentview[t+8]` 是**读取时**通过 delta timestamps 组成的监督，不是 Parquet 中独立的 `[8,7]` 列或未来图片列。WLA 加载器只保留同一 episode 内有真实 `t+8` 帧的起点；开头不足 8 帧的历史图由 LeRobot 补首帧，这是 WLA 原有历史策略。

动作前状态由 `step_started.proprioception_before` 的原生位置、四元数和双指关节值计算，与 WLA 的 `quat2axisangle` 约定一致。原生环境动作最后一维 `-1/+1` 分别写成训练标签 `1/0`，对应 WLA 执行时的逆映射。初始化 settle 帧不进入训练 episode；只按已完成的 task step 对齐前视图、腕视图、状态和动作。

`feedback.jsonl` 每行包含 `source`、`suite`、`task_id`、`initial_state_id`、`task_instruction`、`status`、`official_success`、`end_reason`、`mode`、`backend`、`task_steps`、`model_requests`、`audit_pass`、`review_approved`、`wla_candidate` 和 `rejection_reason`。例如：

```json
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-00","status":"completed","official_success":true,"audit_pass":true,"review_approved":true,"wla_candidate":true,"rejection_reason":null}
```

`trace.jsonl` 中每行有 `source` 和 `kind`：

- `kind="environment"`：原始 `step_started` / `step_completed` / WLA 调用事件。`attempt` 是物理动作序号，`decision_index` 是 Show-Harness 决策序号；`step_started.proprioception_before` 和 `action` 可重建 WLA 样本。
- `kind="agent_decision"`：`decision_index`、Show-Harness `record`（动作 token、阶段、位姿、模型解释等）、原始前视图/腕视图 PNG 的绝对路径；`record.model_request_id` 若存在，连到模型请求。
- `kind="agent_artifact"`：Show-Harness 的 `metadata`、`subgoals`、`planner_diagnostics`、`summary` JSON（存在时）；`artifact_type` 指明来源类型，`record` 保留原内容。
- `kind="model_request"`：`call_id`、请求元数据、完整 prompt/response、请求时双图 PNG 的绝对路径。

将同一个 `source` 下的 `agent_decision.record.model_request_id` 对上 `model_request.call_id`，再用 `environment.decision_index` 对上实际执行动作。一次决策可能执行多步，所以是“一条决策对应多条环境动作”。没有决策编号的初始化动作与某些 WLA-only 事件仍按 `attempt` / `call_id` 查看，不臆造 agent 决策。

`dump-manifest.json` 包含 `dataset`、fps、字段顺序、gripper 映射、每条录入轨迹的来源、任务、帧数和可用 `t+8` 起点数。没有合格 episode 时，`dataset` 为 `null`，不会生成 `rua_lerobot/`，trace/feedback 仍会输出。

## 进入训练前

旧运行若没有 `step_started.proprioception_before`，无法逐帧恢复精确的 8 维状态：它仍会进入 trace/feedback，但拒绝进入 WLA 数据集。建议用更新后的采集代码重新采样。自动准入还要求完整审计、官方成功、无未确认动作/清理错误、任务动作连续、至少 9 帧、双视频可对齐；人工 review 可以继续缩小训练集。

WLA 配置示例为 `configs/libero_all_image_action.yaml`。将 `dataset_root_dir` 设成 `dump-001/`，使用现有 `configs/norm_stats.json` 的 `libero_all` 统计量；两者的数据单位及夹爪编码必须保持一致。训练加载器读取 `rua_lerobot/`，从真实帧组装 `前视图[t-8]、前视图[t]、腕视图[t]、状态[t]、动作[t:t+8]、前视图[t+8]`。如果要把导出数据和旧 LeRobot 数据放在同一根目录混训，加载器会先要求 fps 和前视图/腕视图字段名一致；不一致时要先完成数据转换。新数据的域分布和现有归一化统计量是否匹配，仍需在训练实验中验证。

修改过采集代码和 Show-Harness runner；旧的 source-hash / handoff 回执不再匹配新代码。后续启动受回执约束的真实 hybrid 实验前，按 `RUN_GUIDE.zh.md` 重做对应校验和回执。
