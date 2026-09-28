# 在 `/data1/wcz` 运行一条 LIBERO episode

这里的“真实 episode”指 LIBERO 仿真器实际 reset、渲染、执行 WLA 动作并按官方任务判据评分；不是回放伪造轨迹，也不是真机实验。当前单条调试入口固定为 `libero_spatial` 任务 0、初态 0，不是 LIBERO-Pro 的扰动评测。

## 已准备的环境

- Conda 仿真环境：`/data1/wcz/conda-envs/rua-sim`，使用 CPU PyTorch、LIBERO 0.1、MuJoCo 3.3.3、robosuite 1.4、无图形 OpenCV。
- LIBERO 源码和任务资源：`/data1/wcz/WLA/LIBERO`，来自提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，归档 SHA256 为 `05ffcf8349b2e7ef31b038451253d76ca757debbf88c3a0c1de569ca38a80b14`。
- OSMesa：`/data1/wcz/runtime/wla-libero/osmesa-24.0.5`；配置：`/data1/wcz/configs/libero/config.yaml`。
- 运行入口：`/data1/wcz/run-rua-sim.sh`。仅使用 `/data1/wcz` 下的环境和文件；WLA 预测通过已运行的本机 8120 服务。8121、8122 也支持同一协议。

Conda 环境中的 robosuite 文件日志已关闭，以免写入其他人拥有的 `/tmp/robosuite.log`。入口设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，用于读取上述已校验归档中的 LIBERO 初态文件；只在该仿真进程内生效。

## 跑一条并导出

```bash
bash /data1/wcz/run-rua-sim.sh run --mode wla --wla-service-port 8120
```

输出会先打印 `run_root`。它是每次新建的目录，例如 `/data1/wcz/artifacts/rua-stage3/wla_only_unified_executor-debug-2xn0oevb`。拿这条路径审计：

```bash
bash /data1/wcz/run-rua-sim.sh audit /data1/wcz/artifacts/rua-stage3/wla_only_unified_executor-debug-2xn0oevb
```

再导出 trace、feedback 和成功轨迹的 LeRobot 数据：

```bash
PYTHONPATH=/data1/wcz/WLA/rua:/data1/wcz/WLA/rua/scripts \
  /data1/wcz/conda-envs/rua-data/bin/python -m data_dump.training_dump \
  /data1/wcz/artifacts/rua-stage3/wla_only_unified_executor-debug-2xn0oevb \
  --output-root /data1/wcz/rua-dumps/wla-real-smoke-20260927
```

每次导出必须使用新的空输出目录。数据集字段和文件结构见 `DATA_DUMP.zh.md`。

## 已完成的实例

2026-09-27 的单条运行在 75 个任务动作后由 LIBERO 判定成功，调用 WLA 10 次；审计确认 10 个初始化动作、两路视频各 86 帧、无不确定动作。导出目录 `/data1/wcz/rua-dumps/wla-real-smoke-20260927` 含 1 条 75 帧 LeRobot episode、67 个有完整 `t+8` 未来帧的 WLA 训练起点，以及 trace/feedback。WLA 训练加载器已回读出状态 `[8]`、动作 `[8,7]`、三张输入图和一张未来图。

该运行是 WLA-only，因此 `model_requests=0`，不含 agent/Opus 决策 trace；它验证了环境、WLA 控制、审计和数据 dump 的完整链路。要采集 agent 决策数据，仍需给 agent 模式提供模型服务，或另接一个受支持的本地模型后单独运行。

## LIBERO-Pro 单条筛选验证

Pro 的 BDDL `(:language ...)` 可能与文件名对应的原始指令不同。单条试跑入口会读取 BDDL 中的实际指令，使用现有 WLA 控制器和双视角、动作前状态采集器；读取共享 Pro 任务资源，不修改共享环境：

```bash
bash /data1/wcz/run-pro-single-smoke.sh \
  --suite libero_spatial_lan --task-id 0 --init-id 2 --wla-service-port 8120
```

2026-09-27 试跑两条 Pro 任务。`libero_spatial_lan` 任务 0、初态 2 在 72 个任务动作后成功；审计通过，前视和腕视各 83 帧。导出到 `/data1/wcz/rua-dumps/libero-pro-lan-single-20260927` 后，WLA 训练加载器读到 64 个真实 `t+8` 起点，样本包含三张输入图、状态 `[8]`、动作 `[8,7]` 和未来前视图。`libero_spatial_swap` 任务 0、初态 24 在 300 步上限内失败；审计通过，前视和腕视各 311 帧，筛选器将其标为 `task_not_successful`，不生成训练 episode。两条记录的 `scope` 都是 `libero_pro_single_smoke`。

这些初态来自现有 Pro 评测，导出数据仅用于检查采集和筛选链路。若要在同一 Pro 评测上报告模型分数，不要用这些初态的轨迹训练，否则会混入评测数据。作为正式训练集前，还要确定独立的训练任务/初态划分并复核轨迹质量。
