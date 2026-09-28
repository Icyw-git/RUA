# RUA trace / feedback 数据导出

这项功能把一次完整运行的原始记录整理成两份用途不同、但可互相追溯的数据：

1. `trace.jsonl` + `feedback.jsonl`：保留成功和失败轨迹，供后续 agent 分析、筛选和训练。
2. `rua_lerobot/`：只收录通过审计、成功且有完整监督帧的轨迹，供 WLA 训练。

RUA 运行时自动保存 `result.json`、`environment-steps.jsonl`、`front-control.mp4`、`wrist-control.mp4`，以及存在时的 `requests/`、`native/`。目前**不会自动运行**本页的质量筛选、人工复核或 LeRobot 导出；这些步骤由操作者在 episode 完成后手动启动。导出是离线操作，不调用模型、不执行环境动作。原始目录要保留：`trace.jsonl` 中的图片路径指向原始目录，不复制 PNG。生成的 LeRobot 视频则在新数据集内，可单独搬运。

## 运行

```bash
cd /data1/wcz/WLA
PYTHONPATH=/data1/wcz/WLA/rua/scripts:/data1/wcz/WLA/rua/vendor/show_harness:/data1/wcz/WLA/rua \
  /data1/wcz/conda-envs/rua-data/bin/python -m data_dump.training_dump \
  /data1/wcz/WLA/rua/artifacts/YOUR_RUN \
  --output-root /data1/wcz/rua-dumps/dump-001
```

输入可以是单个 episode 目录或其上级运行目录，也可以传多个目录。输出目录须为空或尚不存在。`rua-data` 是 `/data1/wcz` 下单独的 Conda 环境；导出不会改共享服务或别人的 Python 环境。

## 质量筛选与人工复核

先对原始运行目录生成待复核建议：

```bash
cd /data1/wcz/WLA/rua
PYTHONPATH=/data1/wcz/WLA/rua/scripts:/data1/wcz/WLA/rua/vendor/show_harness:/data1/wcz/WLA/rua \
  /data1/wcz/conda-envs/rua-data/bin/python -m data_dump.quality_screen \
  /data1/wcz/WLA/rua/artifacts/YOUR_RUN \
  --output /data1/wcz/rua-dumps/quality-proposals.jsonl
```

每个来源 episode 在 `quality-proposals.jsonl` 占一行。重要字段如下：

| 字段 | 含义 |
| --- | --- |
| `source`、`scope`、`suite`、`task_id` | 原始轨迹及比较组。速度只与相同 `scope`、suite、task_id 的合格成功轨迹比较。 |
| `eligible_for_wla_review`、`ineligible_reason` | 是否通过原有审计、成功条件和动作/状态对齐，可供人工考虑纳入 WLA；失败轨迹仍有线索记录。 |
| `task_steps` | 实际任务动作数，不含初始化动作或模型等待时间。 |
| `speed_reference_count`、`speed_comparison` | 其他同组成功轨迹数；至少 3 条参考轨迹才给出参考步数中位数、当前步数/中位数和比多少条更短或更长。样本不足时 `speed_comparison` 为 `null`。只作排序依据，不按速度自动淘汰。 |
| `signals` | 夹爪“闭合→张开→再闭合”的三个 `transition_steps`，不是错误动作区间；各步有动作前证据帧。`video_frame` 包含初始化动作。它可能是正常放置后再抓取，不能直接当作失败。原生动作 `+1` 是闭合，`-1` 是张开。 |
| `review_task_step_range` | 通过初步审计的成功轨迹给出整条 `[0, task_steps]` 待审核范围；失败轨迹为 `null`。这是审核入口，不是已经确认的训练片段。人工看片后才能决定是否缩小范围。 |
| `screen_version`、`source_trace_sha256` | 建议格式版本和原始动作记录的哈希，用于发现审核时来源已变化。 |

夹爪变化和速度只用于提示人工查看，不会自动删动作。`eligible_for_wla_review` 表示轨迹通过基础技术检查，**不是训练许可**。例如 Pro 评测轨迹即使该字段为 `true`，也只能用于验证筛选规则，不能回流训练。Pro 测试集中的 swap/init1 在 167 步成功，夹爪信号覆盖了任务中段；直接剪掉这些动作会丢掉最终成功前的调整过程，因此建议范围保留整条供人工审核。

例如对现有两条 Pro 单例轨迹试跑，成功的 `libero_spatial_lan`（72 步）给出待复核范围 `[0,72]`，失败的 `libero_spatial_swap`（300 步）不给 WLA 候选范围，但标出 4 个夹爪变化线索。两条来自不同扰动组，速度比较都为空。这两条使用 Pro 评测初始状态，只用于检查筛选程序，不能直接作为生产训练数据。

如果原始运行保存了 `task.bddl`、`initial_state.npy` 和逐步动作，可以离线重放，补充每个动作后的目标条件和目标物体位置：

```bash
/data1/wcz/WLA/rua/scripts/offline/run_oracle_feedback.sh \
  /data1/wcz/artifacts/rua-stage3/YOUR_EPISODE \
  --output /data1/wcz/rua-dumps/YOUR_ORACLE
```

生成的 `oracle-feedback.jsonl` 每行对应一个任务动作，含 `task_step`、原始 `attempt`、动作后的 `goal_predicates`（如 `on(bowl,plate)` 是否成立）、`object_positions_m`（目标条件中物体的世界坐标）和该步的 `official_success`。`oracle-manifest.json` 记录来源、原始 trace 哈希和回放机械臂末端位置的最大误差。回放会逐步对照原始末端位置与官方成功值；不一致则报错。这里的物体位置只包括目标条件提到的物体，不是全部场景物体；它也不能单独证明“抓取成功”。仿真真值只写在离线分析文件，不送给当时的控制器。

把建议、动作前后的前视/腕视关键帧和可选的逐步仿真结果放到一个审核目录：

```bash
cd /data1/wcz/WLA/rua
PYTHONPATH=/data1/wcz/WLA/rua/scripts:/data1/wcz/WLA/rua/vendor/show_harness:/data1/wcz/WLA/rua \
  /data1/wcz/conda-envs/rua-data/bin/python -m data_dump.review_packet build \
  /data1/wcz/rua-dumps/quality-proposals.jsonl \
  --output /data1/wcz/rua-dumps/review-packet \
  --oracle-root /data1/wcz/rua-dumps/YOUR_ORACLE
```

`review-packet/index.html` 可直接打开。每张卡片并排显示同一动作前后的前视与腕视，共四张图；目标条件对应动作后。每条轨迹还提供完整前视、腕视视频入口。页面可给整条轨迹选“保留整条、保留区间、不纳入、仅验证”，也可给夹爪线索和关键帧分别标“正常、纠正、错误、看不清”；点击“下载审核决定”得到 `review-decisions.jsonl`。选区间前仍要查看完整视频和 trace，首、中、末三个常规动作不足以判断整个连续片段。`review-queue.jsonl` 存动作前后帧路径、对应的原始视频帧号、完整视频路径与反馈；`review-decisions.template.jsonl` 是空白决定模板。Pro 评测样本默认选“仅验证”。

人工审核后，再生成导出器使用的文件：

```bash
PYTHONPATH=/data1/wcz/WLA/rua/scripts:/data1/wcz/WLA/rua/vendor/show_harness:/data1/wcz/WLA/rua \
  /data1/wcz/conda-envs/rua-data/bin/python -m data_dump.review_packet finalize \
  /data1/wcz/rua-dumps/quality-proposals.jsonl \
  /data1/wcz/rua-dumps/review-decisions.jsonl \
  --output /data1/wcz/rua-dumps/review.jsonl
```

`review.jsonl` 只有来源、是否批准、可选连续区间和理由，可传给下面的 WLA 导出器；`review-labels.jsonl` 单独保留动作判断，供以后研究 Agent 数据。生成器会拒绝把 `scope=libero_pro_single_smoke` 的轨迹批准给 WLA。只用模板生成的“仅验证”决定不等于有人逐步审核过，不能当人工质量标签。

若要人工筛选，准备 JSONL 文件，每行指定一个原始 episode 的绝对路径和审核决定：

```json
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-00","approved_for_wla":true,"task_step_ranges":[[10,30],[42,60]],"reason":"仅保留复核过的连续动作段"}
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-01","approved_for_wla":false,"reason":"抓错物体"}
```

再加 `--review /data1/wcz/my-review.jsonl`。传入 review 后，未列出的 episode 也不会进入 WLA 数据集。`approved_for_wla: true` 且不写 `task_step_ranges` 表示保留整条任务轨迹；写了范围则只保留这些片段。范围是任务动作编号 `[起点, 终点)`，从 0 开始，包含起点、不包含终点，不计初始化动作；例如 `[10,30]` 包含编号 10 至 29 的 20 帧，能产生 `20 - 8 = 12` 个 WLA 训练起点。每段至少 9 帧，按先后顺序填写，不能重叠；每段在 LeRobot 中成为独立 episode，避免跨过被剔除的动作拼接训练样本。`reason` 是可选的人工说明。规范化后的决定写入输出的 `review.jsonl`；被否决或被剪掉的动作仍留在原始运行目录和完整的 trace / feedback 中。不传 review 时，以自动审计和成功条件筛选整条轨迹。

`task_step_ranges` 是人工复核结果，不会由“走了多少步”或“耗时多久”自动判好坏。成功轨迹中的试错、慢速完成、纠正动作可能有价值；导出器保留它们，除非审核者明确缩小范围。片段开头不足 8 帧的历史图沿用 WLA 加载器的首帧补齐规则。

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

当前 `configs/libero_all_image_action.yaml` 设置 `auxiliary_drop_thresh: 0.0`：训练样本总会包含腕视图；`dataset.py:load_libero_dataset` 也直接取第二个相机字段，并检查所有 LIBERO 数据集的前两路相机一致。因此按**现有训练配置和加载器**，腕视视频是必需字段。模型本身可处理不同数量的输入图，但前视图单路训练需要另改加载器和配置，再独立验证训练与推理输入一致；不能简单缺省腕视列。

动作前状态由 `step_started.proprioception_before` 的原生位置、四元数和双指关节值计算，与 WLA 的 `quat2axisangle` 约定一致。原生环境动作最后一维 `-1/+1` 分别写成训练标签 `1/0`，对应 WLA 执行时的逆映射。初始化 settle 帧不进入训练 episode；只按已完成的 task step 对齐前视图、腕视图、状态和动作。

`feedback.jsonl` 每个原始 episode 一行，包含 `source`、`suite`、`task_id`、`initial_state_id`、`task_instruction`、`status`、`official_success`、`end_reason`、`mode`、`backend`、`task_steps`、`elapsed_seconds`、`model_requests`、`wla_calls`、`control_tokens`、`audit_pass`、`review_approved`、`wla_candidate`、`wla_training_starts`、`rejection_code` 和 `rejection_reason`。`task_steps` 是环境任务动作数；`elapsed_seconds` 是整次运行耗时，包含模型等待时间；`control_tokens` 是执行过的动作类型；`wla_training_starts` 是最终可读取的 8 步监督样本起点数。`review_approved` 在没有 review 文件时为 `null`，它不代表自动审计结果。`wla_candidate` 为最终是否入选；失败或被拒绝时 `rejection_code` 给出可筛选的原因类别，`rejection_reason` 给出具体说明。例如：

`rejection_code` 可能是 `legacy_source`、`audit_failed`、`task_not_successful`、`uncertain_execution`、`training_data_invalid`、`review_rejected`、`invalid_review_range` 或 `camera_incompatible`。它们分别对应旧格式、账目审计失败、任务未成功、动作或清理状态不确定、训练帧不完整、人工否决、片段范围无效和相机格式不一致。

```json
{"source":"/data1/wcz/WLA/rua/artifacts/YOUR_RUN/task-00-init-00","status":"completed","official_success":true,"task_steps":75,"elapsed_seconds":420.5,"control_tokens":["MV_UP","GRASP"],"audit_pass":true,"review_approved":true,"wla_candidate":true,"wla_training_starts":57,"rejection_code":null,"rejection_reason":null}
```

`trace.jsonl` 中每行有 `source` 和 `kind`：

- `kind="environment"`：原始 `step_started` / `step_completed` / WLA 调用事件。`attempt` 是物理动作序号，`decision_index` 是 Show-Harness 决策序号；`step_started.proprioception_before` 和 `action` 可重建 WLA 样本。
- `kind="agent_decision"`：`decision_index`、Show-Harness `record`（动作 token、阶段、位姿、模型解释等）、原始前视图/腕视图 PNG 的绝对路径；`record.model_request_id` 若存在，连到模型请求。
- `kind="agent_artifact"`：Show-Harness 的 `metadata`、`subgoals`、`planner_diagnostics`、`summary` JSON（存在时）；`artifact_type` 指明来源类型，`record` 保留原内容。
- `kind="model_request"`：`call_id`、请求元数据、完整 prompt/response、请求时双图 PNG 的绝对路径。

将同一个 `source` 下的 `agent_decision.record.model_request_id` 对上 `model_request.call_id`，再用 `environment.decision_index` 对上实际执行动作。一次决策可能执行多步，所以是“一条决策对应多条环境动作”。没有决策编号的初始化动作与某些 WLA-only 事件仍按 `attempt` / `call_id` 查看，不臆造 agent 决策。

`dump-manifest.json` 包含 `dataset`、fps、字段顺序、gripper 映射、每条录入片段的来源、`task_step_range`、任务、帧数和可用 `t+8` 起点数。同一来源若选了两段，在 manifest 和 LeRobot 中各有一条 episode；`feedback.jsonl` 仍只有一行，`wla_training_starts` 是两段之和。manifest 中的 `official_success` 指原始完整任务的官方结果。没有合格 episode 时，`dataset` 为 `null`，不会生成 `rua_lerobot/`，trace/feedback 仍会输出。

## 进入训练前

旧运行若没有 `step_started.proprioception_before`，无法逐帧恢复精确的 8 维状态：它仍会进入 trace/feedback，但拒绝进入 WLA 数据集。建议用更新后的采集代码重新采样。自动准入还要求完整审计、官方成功、无未确认动作/清理错误、任务动作连续、至少 9 帧、双视频可对齐；人工 review 可以继续缩小训练集。

WLA 配置示例为 `configs/libero_all_image_action.yaml`。将 `dataset_root_dir` 设成 `dump-001/`，使用现有 `configs/norm_stats.json` 的 `libero_all` 统计量；两者的数据单位及夹爪编码必须保持一致。训练加载器读取 `rua_lerobot/`，从真实帧组装 `前视图[t-8]、前视图[t]、腕视图[t]、状态[t]、动作[t:t+8]、前视图[t+8]`。如果要把导出数据和旧 LeRobot 数据放在同一根目录混训，加载器会先要求 fps 和前视图/腕视图字段名一致；不一致时要先完成数据转换。新数据的域分布和现有归一化统计量是否匹配，仍需在训练实验中验证。

Agent 数据目前保留在 `trace.jsonl`，可按 `source`、`decision_index`、`model_request_id` 找回每次决策的输入、输出和实际执行动作。尚未自动筛成 Agent 训练集：现有 `official_success` 只评价整条任务，不能判断其中某一步决策的好坏，失败任务里也可能有正确步骤。若要训练 Agent，应先明确训练目标与样本格式，再按决策编号人工标注“可模仿、应纠正、暂不判断”等结果；不能把整条任务的成败直接复制到每一步。

修改过采集代码和 Show-Harness runner；旧的 source-hash / handoff 回执不再匹配新代码。后续启动受回执约束的真实 hybrid 实验前，按 `RUN_GUIDE.zh.md` 重做对应校验和回执。
