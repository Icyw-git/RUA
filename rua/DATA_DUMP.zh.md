# RUA trace / feedback 数据导出

## 推荐入口：自动质量分类与 WLA 正面样本导出

新数据统一使用下面的质量流程。下文基础导出和人工复核仅用于格式检查或已有审核任务，无需再串联运行。

下文 `/path/to/...` 均为占位路径，运行前替换为实际目录；`/path/to/RUA/rua` 指本仓库或独立克隆中的 `rua/`，`/path/to/artifacts` 指原始 episode 的存放目录。

已有 episode 保存后，可在独立仿真环境里一次完成回放、分类、导出。离线脚本只依赖 `rua/` 代码，可以从本仓库或独立克隆的 `nanorua/rua` 运行。先配置仿真 Python；若 LIBERO 没安装在该环境中，再将 `LIBERO_PYTHONPATH` 指向包含 `libero/` Python 包的目录。使用 LIBERO-PRO 时，还应将 `LIBERO_CONFIG_PATH` 指向对应配置目录。`RUA_ROOT` 默认是仓库目录的父目录，可按需要设置。输入可以是一个 episode，也可以是包含多个 episode 的目录；输出目录需为空：

```bash
export RUA_VENV_PYTHON=/path/to/rua-sim/bin/python
export LIBERO_PYTHONPATH=/path/to/LIBERO-PRO/libero
export LIBERO_CONFIG_PATH=/path/to/libero-pro-config
/path/to/RUA/rua/scripts/offline/run_quality_pipeline.sh \
  /path/to/YOUR_EPISODE \
  --output-root /path/to/YOUR_QUALITY_DUMP
```

程序逐步回放原动作并核对末端位置和官方任务结果；它记录物体位置、末端位置、夹爪接触、物体与夹爪的相对运动、物体尺寸、抽屉关节和目标条件。`quality-labels.jsonl` 把动作段标为 `nominal`（有后续子目标支持、未发现明确错误）、`recovery`（已确认错误后的纠正）、`error` 或 `uncertain`。`task_phase` 是辅助诊断，可能为空，不参与训练筛选。`verification` 记录证据是否确认，`coverage` 记录所需状态是否齐全。当前只自动处理任务目标为 `On`、`In`、`Open`、`Close` 的轨迹；其他目标保存为 `unsupported/uncertain`。夹爪命令变化本身不能证明抓住、抓空或掉落；只有物体水平跟着末端移动也不足以证明抓取。放手后没有明确目标达成或下落证据时，标成待定。证据不完整的段不进入正面训练起点。标签另存末端路径长度和静止步数；同一 scope、suite、task、初始状态下至少有三条其他成功轨迹时，才写相对步数和路径比值。快慢不决定入选。

`nominal + verified` 表示有后续成功或抓取等进展证据支持，且未被当前规则判为错误或待定；它不表示逐步确认了动作高效、最优或适合所有任务。`recovery + verified` 还要求先前错误和后续纠正的证据。

输出中的 `dump/rua_lerobot/` 保留获选轨迹的**完整 episode**，`dump/rua_lerobot/meta/quality-starts.jsonl` 决定 WLA 实际读取哪些起点。每条获选起点的 8 个监督动作均属于同一类已验证动作，且有真实的第 `t+8` 帧。历史前视图仍从原 episode 的 `t−8` 读取。最终失败的轨迹也会分类；只有完成了可验证局部目标的正面动作段才可能被选中。错误、待定及未选动作仍保存在原始 trace 和分类结果中。

主要文件：`oracle/` 是逐步仿真证据；`quality-labels.jsonl` 是原轨迹的逐段标签；`behavior-signals.jsonl` 是慢速、绕行或重复夹爪周期的候选区间及证据；`review-starts.jsonl` 列出这些候选区间覆盖的已导出训练起点；`dump/feedback.jsonl` 是每条轨迹的入选数量和拒绝原因；`dump/rua_lerobot/meta/quality-starts.jsonl` 是训练起点；`summary.json` 是本次汇总。行为信号目前是影子分析，不改变训练起点。导出结果使用 `selection_mode=quality`，与下文的旧式 `technical_success_only` 格式测试导出明确区分。

批量处理时，单条输入、回放、分类或导出检查失败会在 `dump/feedback.jsonl` 中记录原因，其余来源继续执行。`summary.json` 的 `preprocessing_failed_sources` 统计回放与分类前后的失败，`export_failed_sources` 统计导出检查失败，`rejected_sources` 统计最终未导出的来源总数。拒绝来源的最终训练起点及两类起点计数均为零。正常完成退出码为 `0`（包括正常筛选后全部不入选）；任一阶段有单条数据处理失败则完成剩余导出后返回 `2`。依赖、系统或程序故障仍中止。全部拒绝时保留反馈和零计数 manifest，不创建空数据集。可选 Agent 日志损坏时，反馈的 `trace_warnings` 记录文件和原因；只要机器人训练数据与回放证据完整，该轨迹仍可进入 WLA 数据集。

训练加载器在发现 `quality-starts.jsonl` 时自动按它筛选。`quality_set=auto` 读取所有获选正面样本，也可以在训练配置里设 `quality_set=nominal` 或 `quality_set=recovery` 分别读取。数据目录指向 `YOUR_QUALITY_DUMP/dump`。当前标签器只支持 LIBERO 仿真回放中的抓取/放置与抽屉目标；缺少可重放的物理证据时保持待定，不声称能判断路径是否最优。当前规则版本为 `rules-008`；此前版本的标签与导出计数不能直接当作新版结果。

夹爪候选使用 `repeated_gripper_cycle`，`evidence.cycle_count` 统计夹爪周期结束事件，不等于目标抓取尝试次数。`task_step_range` 只覆盖这些事件所在范围，可能包含中间调整动作，不能直接用作训练片段边界。当前信号版本为 `signals-004`，历史产物保留。

新增低进展候选复用 `inefficient_motion`，通过 `evidence.reason=low_progress` 区分。检查放置任务中证据完整的接近片段：已验证的 `nominal`、`recovery` 以及无法确认动作好坏的 `uncertain` 都可生成复查候选；`error` 不纳入。`evidence.action_role` 保留原动作标签，不修改 `quality-labels.jsonl` 或 WLA 训练起点。片段还需满足夹爪打开、没有持物。单目标任务直接关联目标；多目标任务必须有紧接该段的明确目标抓取，否则跳过。连续至少 24 步（当前 20 Hz 下为 1.2 秒），手的位置偏移不超过 5 mm、转动不超过 3°、夹爪实际位置变化不超过 1 mm、场景中所有物体的位置偏移不超过 2 mm 且转动不超过 3°，并且目标条件和关节状态没有变化，才生成候选。偏移比较整个区间相对于起点的最大变化，不把多段缓慢推进合并成“停滞”。这些容差是待验证的排查标准，不能证明动作无用。`review-starts.jsonl` 仅列出候选覆盖的已入选训练起点；来自未入选的 `uncertain` 片段可能只在 `behavior-signals.jsonl` 出现。

回放证据版本 3 增加 `eef_quaternion_xyzw`（末端姿态）、`gripper_qpos_m`（实际夹爪关节位置）及 `object_quaternions_wxyz`（各物体姿态）。旧证据缺少这些字段时不生成低进展候选。证据记录的是执行后的状态，所以判断时额外读取前一步作比较；没有执行前状态的第 0 步不标记。候选区间采用 `[start,end)`，证据记录实际偏移、转角、步数和目标物体。该功能**不改四类动作标签、不减少训练起点**；即使动作慢，也可能是在精细调整，后续需验证候选是否有用。

下面是原有的基础导出和人工复核流程，接口继续可用。

这项功能把一次完整运行的原始记录整理成两份用途不同、但可互相追溯的数据：

1. `trace.jsonl` + `feedback.jsonl`：保留成功和失败轨迹，供后续 agent 分析、筛选和训练。
2. `rua_lerobot/`：只收录通过审计、成功且有完整监督帧的轨迹，供 WLA 训练。

RUA 运行时自动保存 `result.json`、`environment-steps.jsonl`、`front-control.mp4`、`wrist-control.mp4`，以及存在时的 `requests/`、`native/`。目前**不会自动运行**本页的质量筛选、人工复核或 LeRobot 导出；这些步骤由操作者在 episode 完成后手动启动。导出是离线操作，不调用模型、不执行环境动作。原始目录要保留：`trace.jsonl` 中的图片路径指向原始目录，不复制 PNG。生成的 LeRobot 视频则在新数据集内，可单独搬运。

## 基础导出（兼容入口）

```bash
cd /path/to/RUA
PYTHONPATH=/path/to/RUA/rua/scripts:/path/to/RUA/rua/vendor/show_harness:/path/to/RUA/rua \
  /path/to/rua-data/bin/python -m data_dump.training_dump \
  /path/to/artifacts/YOUR_RUN \
  --output-root /path/to/rua-dumps/dump-001
```

输入可以是单个 episode 目录或其上级运行目录，也可以传多个目录。输出目录须为空或尚不存在。`rua-data` 是单独的 Conda 环境；导出不会改共享服务或别人的 Python 环境。

## 人工复核（兼容入口）

先对原始运行目录生成待复核建议：

```bash
cd /path/to/RUA/rua
PYTHONPATH=/path/to/RUA/rua/scripts:/path/to/RUA/rua/vendor/show_harness:/path/to/RUA/rua \
  /path/to/rua-data/bin/python -m data_dump.quality_screen \
  /path/to/artifacts/YOUR_RUN \
  --output /path/to/rua-dumps/quality-proposals.jsonl
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
/path/to/RUA/rua/scripts/offline/run_oracle_feedback.sh \
  /path/to/YOUR_EPISODE \
  --output /path/to/YOUR_ORACLE
```

生成的 `oracle-feedback.jsonl` 每行对应一个任务动作，含 `task_step`、原始 `attempt`、动作后的 `goal_predicates`（如 `on(bowl,plate)` 是否成立）、`object_positions_m`（目标条件中物体的世界坐标）和该步的 `official_success`。`oracle-manifest.json` 记录来源、原始 trace 哈希和回放机械臂末端位置的最大误差。回放会逐步对照原始末端位置与官方成功值；不一致则报错。这里的物体位置只包括目标条件提到的物体，不是全部场景物体；它也不能单独证明“抓取成功”。仿真真值只写在离线分析文件，不送给当时的控制器。

把建议、动作前后的前视/腕视关键帧和可选的逐步仿真结果放到一个审核目录：

```bash
cd /path/to/RUA/rua
PYTHONPATH=/path/to/RUA/rua/scripts:/path/to/RUA/rua/vendor/show_harness:/path/to/RUA/rua \
  /path/to/rua-data/bin/python -m data_dump.review_packet build \
  /path/to/rua-dumps/quality-proposals.jsonl \
  --output /path/to/rua-dumps/review-packet \
  --oracle-root /path/to/rua-dumps/YOUR_ORACLE
```

`review-packet/index.html` 可直接打开。每张卡片并排显示同一动作前后的前视与腕视，共四张图；目标条件对应动作后。每条轨迹还提供完整前视、腕视视频入口。页面可给整条轨迹选“保留整条、保留区间、不纳入、仅验证”，也可给夹爪线索和关键帧分别标“正常、纠正、错误、看不清”；点击“下载审核决定”得到 `review-decisions.jsonl`。选区间前仍要查看完整视频和 trace，首、中、末三个常规动作不足以判断整个连续片段。`review-queue.jsonl` 存动作前后帧路径、对应的原始视频帧号、完整视频路径与反馈；`review-decisions.template.jsonl` 是空白决定模板。Pro 评测样本默认选“仅验证”。

人工审核后，再生成导出器使用的文件：

```bash
PYTHONPATH=/path/to/RUA/rua/scripts:/path/to/RUA/rua/vendor/show_harness:/path/to/RUA/rua \
  /path/to/rua-data/bin/python -m data_dump.review_packet finalize \
  /path/to/rua-dumps/quality-proposals.jsonl \
  /path/to/rua-dumps/review-decisions.jsonl \
  --output /path/to/rua-dumps/review.jsonl
```

`review.jsonl` 只有来源、是否批准、可选连续区间和理由，可传给下面的 WLA 导出器；`review-labels.jsonl` 单独保留动作判断，供以后研究 Agent 数据。生成器会拒绝把 `scope=libero_pro_single_smoke` 的轨迹批准给 WLA。只用模板生成的“仅验证”决定不等于有人逐步审核过，不能当人工质量标签。

若要人工筛选，准备 JSONL 文件，每行指定一个原始 episode 的绝对路径和审核决定：

```json
{"source":"/path/to/artifacts/YOUR_RUN/task-00-init-00","approved_for_wla":true,"task_step_ranges":[[10,30],[42,60]],"reason":"仅保留复核过的连续动作段"}
{"source":"/path/to/artifacts/YOUR_RUN/task-00-init-01","approved_for_wla":false,"reason":"抓错物体"}
```

再加 `--review /path/to/my-review.jsonl`。传入 review 后，未列出的 episode 也不会进入 WLA 数据集。`approved_for_wla: true` 且不写 `task_step_ranges` 表示保留整条任务轨迹；写了范围则只保留这些片段。范围是任务动作编号 `[起点, 终点)`，从 0 开始，包含起点、不包含终点，不计初始化动作；例如 `[10,30]` 包含编号 10 至 29 的 20 帧，能产生 `20 - 8 = 12` 个 WLA 训练起点。每段至少 9 帧，按先后顺序填写，不能重叠；每段在 LeRobot 中成为独立 episode，避免跨过被剔除的动作拼接训练样本。`reason` 是可选的人工说明。规范化后的决定写入输出的 `review.jsonl`；被否决或被剪掉的动作仍留在原始运行目录和完整的 trace / feedback 中。不传 review 时，以自动审计和成功条件筛选整条轨迹。

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
{"source":"/path/to/artifacts/YOUR_RUN/task-00-init-00","status":"completed","official_success":true,"task_steps":75,"elapsed_seconds":420.5,"control_tokens":["MV_UP","GRASP"],"audit_pass":true,"review_approved":true,"wla_candidate":true,"wla_training_starts":57,"rejection_code":null,"rejection_reason":null}
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
