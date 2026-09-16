# 代码复用、实验版本与导出边界

这是一套 Show-Harness → LIBERO 的工程适配，不是 Claude Code 控制机器人，
也不是重写的 agent 主循环。Opus 是 Anthropic-compatible 模型接口；
主循环直接实例化完整 vendor 中的 `core.runners.real.RealEpisodeRunner`。

## 来源与直接复用

- WLA：`5662058f3ead8a1b4fec9865acce21dd2f5efbf1`，集成不修改其已有 tracked 文件。
- Show-Harness：`137d5718c3b7af0150764d8f9beeb252c9f2794a`。
- LIBERO：`8f1084e3132a39270c3a13ebe37270a43ece2a01`，安装时单独准备。
- Show-Harness 完整 vendor 保留原 Apache-2.0 LICENSE。
- `vendor/show-harness-manifest.json` 逐文件记录上游与当前 SHA256。
- `vendor/patches/` 为相对上游的两个审查补丁；完整集成补丁已含其效果，不要重复应用。

直接复用 `SubgoalPlanner`、`Controller`、`ControllerAgent`、`RealEpisodeRunner`、
共享类型、planner/controller prompt、图像和上下文处理、`EpisodeLogger`。
启用 `subgoal`、`proprioception`、`mem_text`、`recovery`；
`mem_text` 只是本 episode 的近期动作，不是跨回合经验记忆。
未启用 affordance、deepplan、动作分块、可变步长等额外策略插件。

## 上游仅修改两个文件

1. `core/runners/real.py`：隔离硬件导入、加入可选环境边界，接入预算、官方评分、
   错误持久化；delegated 返回不代表子任务成功；`HANDOFF_HOLD` 后丢弃旧动作并重新观察。
2. `core/vlm/roles.py`：严格客户端不从非法响应猜动作；允许额外动作和上下文插件。
   未指定新边界的路径保留原接口，但本包没有真机回归验收。

上游 controller 的方向提示由 `coordination.py` 局部替换为前视坐标指导，
并组合复用恢复组件；不应描述为 prompt 一字未改。

## 新增适配

- `environment.py`：双图、本体白名单、原子动作、计费、官方成功判据。
- `claude.py`：三个模型接口、双图 streaming、严格 JSON/schema/token 校验、
  有界 429 重试。重复键、额外字段等仍拒绝，不因导出而“修好”旧失败。
- `local_vlm.py`：可选 OpenAI-compatible 本地视觉接口；不用于本次三组比较。
- `validate_wla.py`：统一组装三种模式，WLA-only 无 agent 请求。
- `wla_client.py` / `wla_worker.py` / `worker_protocol.py`：独立 WLA GPU worker，
  有界 IPC、资源预检、异常与清理。CPU 模拟器进程不初始化 CUDA。
- `handoff.py` / `coordination.py`：一个原生 chunk 的有界调用、共享实际轨迹历史、
  单一执行所有权与交接；交接后无残留动作。
- `paired.py` / `paired_driver.py`：冻结20任务、配对顺序、逐任务预检、监督和审计。

## 导出时新增的可迁移性处理

源代码与数据根目录分离；私有 endpoint/model/token 外置；GPU 可选；
新增 `setup-local`、`run --mode opus|wla|hybrid`；模型 revision 和依赖固定；
新安装必须重新标定并生成 handoff 回执。资源门槛不降低。
全量CPU安装标定限时从300秒改为900秒并保存中断失败；评分episode预算不变。

本包保留评测控制与严格解析语义，但这些路径/配置/监督改动是评测后导出改动，
不冒充由60回合验证。实际应用、单元测试和同机异目录烟测范围见交付验证报告。
不同机器的新装依赖、服务响应兼容性和三组完整再评测仍需当地验证。

不打包权重、缓存、运行环境、内部代理、密钥、旧校准许可或历史视频。
原机器迁移脚本和包含内部路径的旧交付计划不导出。
