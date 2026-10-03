# RUA：Show-Harness / WLA / LIBERO 三模式

本目录作为完整增量集成到固定 WLA checkout 中。不是 Claude Code agent：
Opus 提供模型推理，Show-Harness 的原生 planner/controller/runner 驱动主循环。
运行总结见交付包 `EXPERIMENT_REPORT.zh.md`；直接复用和适配列表见
`SHOW_HARNESS_REUSE.md`。不要将这20任务、每任务1初态的工程比较称为官方 benchmark。
基于轨迹和反馈导出 WLA / agent 数据，先看 `DATA_DUMP_SUMMARY.zh.md` 的功能概览；操作说明见 `DATA_DUMP.zh.md`。
在 `/data1/wcz` 运行单条 LIBERO/WLA episode 的实测命令见 `RUN_LOCAL_WLA.zh.md`。

代码按用途放置：`libero_harness/` 是正式控制与采集运行时；
`data_dump/` 只离线读取已完成的轨迹，完成质量筛选、仿真回放、人工复核和训练数据导出；
`rua_experiments/` 放单条 Pro 试跑入口；`tests_data_dump/` 放离线数据工具测试，
`tests_stage2/`、`tests_stage3/` 放运行时测试。离线回放的环境脚本在 `scripts/offline/`。
原始 episode 和生成的审核包位于 `/data1/wcz/artifacts/` 与 `/data1/wcz/rua-dumps/`，不放入运行时包。

## 1. 目录与支持范围

```text
experiment-data/              # RUA_ROOT，默认是 WLA checkout 的父目录
  wla/                       # 固定 WLA 基线
    rua/                     # 本集成，含完整 Show-Harness vendor
    LIBERO/                  # 安装时准备的固定源码
  models/                    # HF 主模型 + backbone + VAE/scheduler + lock
  runtime/wla-libero/         # 独立 Python 3.11、缓存、私有渲染库
  configs/                   # 本地生成的 LIBERO 路径、标定配置
  artifacts/                 # 完整逐步轨迹、双图视频、请求、评分与审计
```

Linux x86_64、NVIDIA GPU；本次验证机器使用 Ubuntu 24.04 容器和 OSMesa CPU 渲染。
不是 macOS 原生运行包。安装脚本不修改系统库或 shell 配置。
默认保留24 GiB空闲显存、32 GiB cgroup硬余量、64 GiB估算可回收余量，
WLA allocator上限12 GiB。选择一张GPU，不要求两张都满足门槛，也不停止其他任务。
这些是本实现的保守准入门槛，不是宣称模型必需这么大。

内存监测要求容器的 `/sys/fs/cgroup` 根对应调用进程的 memory cgroup，
支持 v1/v2；非此布局先正确配置容器/监测，不能删除安全检查。
若使用别的系统/驱动/渲染库，须重做标定和交接验收。

## 2. 准备依赖、模型和本地路径

先用交付包的 `apply.sh` 应用补丁，再进入 WLA checkout。
以下命令均在 **bash** 中执行，路径自行替换。`RUA_ROOT` 必须是绝对路径。

```bash
cd /absolute/path/to/experiment-data/wla
export RUA_ROOT=/absolute/path/to/experiment-data
export RUA_GPU=0

# 需要预先具备 git、curl、uv、dpkg-deb、apt-get 和 NVIDIA 驱动。
bash rua/scripts/install_stage1.sh
bash rua/scripts/prepare_osmesa.sh
bash rua/scripts/run_show_harness.sh setup-local layout
bash rua/scripts/run_stage1.sh verify-runtime

# 公共HF HTTPS默认启用校验；按实际网络选择镜像。
export HF_ENDPOINT=https://hf-mirror.com
bash rua/scripts/run_stage1.sh models --download
bash rua/scripts/run_stage1.sh verify-models
```

`install_stage1.sh` 固定 WLA/LIBERO 提交、Python 3.11.15、完整依赖 lock；
LIBERO 压缩包和私有 GLVND 校验 SHA256 后使用。`prepare_osmesa.sh` 使用固定
Ubuntu 24.04 包版本；若包索引没有这些版本，需从可信源准备同版本并通过校验，
不能随意替换。系统还需满足 `ldd libOSMesa.so.8` 所示的依赖（例如 LLVM）。
校验/下载错误时停止，不使用 `--insecure`。

模型由 `configs/model-revisions.json` 固定三个 public repo 的 revision，
包含 WLA 主权重、RynnBrain backbone 和 Sana VAE/scheduler 等嵌套依赖；
不是只下载主 checkpoint。模型/venv/缓存不随补丁分发。
CPFS 若报告0可用空间，只有实际配额已核实足够时才加
`--capacity-confirmed-by-user`，不要用它掩盖真实空间不足。
已有 model lock 时脚本拒绝隐式更新；检查并保留旧实验，不随手删除。

测试和本机共享依赖时可显式设置 `RUA_VENV_PYTHON`、`RUA_OSMESA_LIB_DIR`、
`RUA_RUNTIME`，见 `configs/backend.env.example`。这种烟测不等于新机器全新安装验证。
如果 `rua/` 位于独立仓库，运行需要 WLA 的入口前设置 `WLA_ROOT` 为完整的
WLA checkout 路径；默认值仍是 `rua/` 的上级目录。Codex CLI 试跑默认从
`PATH` 查找 `codex`，也可设置 `RUA_CODEX_BINARY` 为可执行文件路径；
Show-Harness Codex 客户端还支持配置项 `codex_binary`，优先于环境变量。

## 3. Opus API、真实标定与交接许可

复制 `configs/backend.env.example` 到代码目录外的私有文件，权限设为600；
填写自己的 endpoint、可用模型ID、token，再 `source` 私有文件。
三个模式的 Opus 统一使用同一模型配置，不自动降级到其他模型。
历史实验网关返回的ID为 `claude-opus-5`；这个内部别名不保证外部服务可用，
必须填供应商实际支持的模型ID。更换模型后的结果必须单列。

支持 Anthropic-compatible `/v1/messages`、双图、streaming与工具所需响应协议。
本次客户端明确关闭 thinking，不发送 temperature（历史网关拒绝此参数）；
所以它不是“模型全能力/最佳prompt”的通用上限测量。
凭证仅从 `ANTHROPIC_AUTH_TOKEN` 或私有 `RUA_AUTH_FILE` 读取。
认证头由 `RUA_AUTH_SCHEME=x-api-key|bearer` 选择；默认bearer保留历史代理行为，
示例的API-key服务使用x-api-key。以自己的服务要求为准，先做协议探测。
不要把私有URL查询参数或token写进可发布配置/日志。

```bash
bash rua/scripts/run_show_harness.sh test tests tests_stage3 \
  --junitxml="$RUA_ROOT/artifacts/export-tests.xml"

# 真实执行机器人/相机标定；不请求任何模型。
bash rua/scripts/run_show_harness.sh calibrate
# 使用刚输出的 calibration_directory/report.json，不是本包历史成绩。
bash rua/scripts/run_show_harness.sh setup-local accept-calibration \
  "$RUA_ROOT/artifacts/rua-stage2/calibration-REPLACE/report.json"

# 可选：先验证自己的API双图协议，会发送真实模型请求。
bash rua/scripts/run_show_harness.sh probe-model --backend claude
```

完整标定为48组方向/幅度/重复步数探测，CPU渲染可能需要数分钟，
安装标定单独限时900秒（导出烟测确认旧300秒不足）。这不改变任何评分episode预算。
被中断时保留失败标定报告，不可将部分探测当通过。

若已有 `agent.local.json`，接收标定时拒绝覆盖：保留旧文件，为新实验设置新的
`RUA_AGENT_CONFIG`。默认标定未认证，旧机器的标定/交接回执不随包提供。
模型非法JSON、重复键、额外字段、截断、超时均不执行猜测动作；
429最多重试2次，计入请求预算；其他失败保留原样。

## 4. 三种模式独立运行

以下简单入口固定为 Spatial任务0、调试初态0，不计入20任务评分。
默认900秒、220任务控制步+10初始化步、Opus最多100次请求。
每次调用创建新的结果目录；控制和评分复用评测的统一执行器。

```bash
# Pure Opus / RUA-only：只执行原子动作，不加载或调用WLA。
bash rua/scripts/run_show_harness.sh run --mode opus

# Pure WLA：原始完整指令+原生图像/历史/状态处理，无Opus请求。
bash rua/scripts/run_show_harness.sh run --mode wla

# Hybrid还需先用上条WLA调试轨迹验收交接。SOURCE是上条输出的run_root。
bash rua/scripts/run_show_harness.sh handoff-check --coordinated \
  --source "$RUA_ROOT/artifacts/rua-stage3/wla_only_unified_executor-debug-REPLACE" \
  --recovery-tests "$RUA_ROOT/artifacts/export-tests.xml"

# 使用刚输出的recorded-handoff目录下report.json。
export RUA_HANDOFF="$RUA_ROOT/artifacts/rua-stage3/recorded-handoff-REPLACE/report.json"
bash rua/scripts/run_show_harness.sh run --mode hybrid \
  --handoff-receipt "$RUA_HANDOFF"
```

| 模式 | 决策输入与主循环 | 实际执行 |
|---|---|---|
| Opus | Show-Harness子任务规划；双图、本体、近期动作、剩余预算 | `MV_*`、GRASP、RELEASE等原子动作 |
| WLA | 完整原始任务文本；官方观察/归一化/历史 | 原生8步chunk，逐步预算与成功检查 |
| Hybrid | 同一Show-Harness循环，自主选择原子动作或`WLA_CHUNK` | 最多一个chunk后交还新观察，允许再次委托或自己控制 |

Hybrid没有强制交替；WLA调用返回不算子任务成功。任意时刻只有一个环境执行者；
交接会保持夹爪、按需进行5步稳定等待、丢弃旧请求动作并重新观察，
所有真实步都计费。原始完整任务指令始终给WLA，不改写成agent子任务。
WLA与agent共享实际执行历史，不保留停止后未执行的动作。

`DONE`仅是agent宣称子任务完成；官方成功只取LIBERO判据。
无Qwen参与这三个模式，无跨回合经验记忆、无训练、无真机动作。

## 5. 20任务 × 三组配对实验

冻结Spatial0–9和Object0–9；每任务初态1，校准用0；任务交错、arm顺序轮换。
Spatial220/Object280任务步+10初始化步；900秒/100请求；一次一个episode。
先生成本安装的新manifest，再由batch入口逐任务物理预检后串行跑60回合：

```bash
mkdir -p "$RUA_ROOT/artifacts/rua-stage3/paired20-external"
export RUA_BATCH="$RUA_ROOT/artifacts/rua-stage3/paired20-external"
bash rua/scripts/run_show_harness.sh paired manifest \
  --output "$RUA_BATCH/manifest.json" --handoff-receipt "$RUA_HANDOFF"
bash rua/scripts/run_show_harness.sh paired-batch \
  --root "$RUA_BATCH" --manifest "$RUA_BATCH/manifest.json"
```

这会进行真实API请求和WLA推理。全过程冻结源码、模型/endpoint、任务定义、初态哈希、
预算与顺序。不要用本包原实验manifest当新安装许可，也不要运行后改源码或配置。

需要单独运行某个已冻结case时：

```bash
bash rua/scripts/run_show_harness.sh run --mode opus \
  --paired-manifest "$RUA_BATCH/manifest.json" \
  --preflight-receipt "$RUA_BATCH/preflight/receipt.json" \
  --case-id libero_spatial-00 --output "$RUA_BATCH/manual-new-opus"
```

将mode改为wla/hybrid即可。此入口需要全部预检已封存；它不自动写入batch ledger，
不要与batch同时运行，也不要将手工重跑塞回原分母。

## 6. 结果、恢复和限制

`ledger.json` / `summary.json` / `SUMMARY.md` 是进度和汇总；
每episode含 `result.json`、动作/请求记录、双视图视频和 `audit.json`。
审计检查记账完整性，不代表任务成功：

```bash
bash rua/scripts/run_show_harness.sh audit /absolute/path/to/episode
```

中断先等待本任务子进程退出再恢复同一batch命令。已完成行会跳过；
未入账但已有输出、失败预检、完整性错误会停下要求诊断，不自动重跑覆盖。
资源不足不停止别人的服务，不自动扩充预算。
若更改策略、prompt、校准或模型，使用新实验目录和新回执。

本实现当前存在明显模型格式失败和混合协同收益不足，详见报告。
补丁可应用/测试通过不意味着任意外部API兼容，也不意味着新机器复现同样成功率。
