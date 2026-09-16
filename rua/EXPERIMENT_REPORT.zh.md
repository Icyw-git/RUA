# LIBERO 20任务三组配对工程比较

日期：2026-09-16。完整60/60回合已结束并独立复核。原始代码、失败、轨迹和分母未改，
未替换样本、未补跑失败回合。导出可迁移改动在另外的目录进行，验证范围单列。

## 结论

| 模式 | 官方成功/计划回合 | 非基础设施失败 | API/基础设施错误 | Spatial | Object |
|---|---:|---:|---:|---:|---:|
| Pure Opus | 0/20（0%） | 19 | 1 | 0/10 | 0/10 |
| Pure WLA | 20/20（100%） | 0 | 0 | 10/10 | 10/10 |
| Opus + WLA | 9/20（45%） | 8 | 3 | 4/10 | 5/10 |

目前应以WLA-only作为可靠基线，不能宣称混合优于WLA。混合比Pure Opus多完成9个任务，
但成功的9个回合均没有WLA→原子动作交接；实际8次交接分布在4个失败回合中。
证明的是“agent能自主委托WLA”和“确实能交替接管”，不是“交错纠错有效”。

这是当前checkpoint、prompt、原子动作/恢复适配和严格解析器构成的系统成绩，
不是Opus与WLA的一般能力排名。WLA使用面向LIBERO的专门checkpoint；
模型先验、动作表示和推理开销并不对等。WLA在这20个样本上已满分，
本组样本也不足以度量混合方案可能在哪些更难场景增益。

## 固定协议与可信度

- `libero_spatial`、`libero_object`各默认顺序任务0–9；任务间交错，三arm顺序轮换。
- 每任务评分初态1；标定初态0单列。总60回合，非20回合拆成三组。
- Spatial220/Object280任务控制步，另加10初始化等待步；所有实际动作/夹爪/恢复/交接都计费。
- 单回合900秒；有agent时最多100次模型请求，包括重试；一次仅一个episode。
- 环境种子0、模型侧种子7；API采样使用服务端默认，未宣称端到端确定性。
- Opus请求/返回模型ID均记录为`claude-opus-5`；关闭thinking，未调用外部Qwen或CC。
- 模型DONE不算官方成功。成功仅依LIBERO的官方判据。
- 基础设施错误也保留在主成功率分母；辅助报告剔除其后的0/19、20/20、9/17，
  但不剔除协议失败，也不以这个选择性分母冒充主成绩。
- 独立重新解码全部双视图视频，逐步核对动作账目：60份审计通过，
  无uncertain步骤、无清理失败或残留WLA worker；60份监督报告无新增Xid。
- 上述视频检查是机器解码/记账核验，不是逐帧人工语义审核。

固定manifest SHA256：
`afa38092900c04b8672590b923518b26013c2d6a52aad1fa242da76ee32af81b`

原始证据摘要为本包`evidence/final-audit.json`，含逐回合结果/监督/任务定义/初态哈希，
真实输入输出token、动作统计、双视图帧数和评测源码hash。原始大视频与API请求全文留在
实验归档中，不随此代码补丁分发；因此接收方可审查摘要，但不能仅凭摘要独立重算原始视频。

## 开销

| 模式 | 平均任务控制步 | 平均episode秒 | 平均runner秒 | 平均监督总秒 | 模型请求总数 | WLA调用/执行步 |
|---|---:|---:|---:|---:|---:|---:|
| Opus | 98.70 | 200.19 | 219.07 | 226.82 | 563 | 0 / 0 |
| WLA | 115.80 | 55.29 | 96.97 | 104.75 | 0 | 296 / 2316 |
| Hybrid | 116.65 | 183.91 | 224.79 | 232.44 | 392 | 269 / 2104 |

episode时间不含完整模型启动等开销；runner和监督计时逐级包含更多初始化/收尾，
不要把55.29秒当外部命令的完整耗时。这里平均数包含失败回合，
Opus较少的平均步数主要受提前格式失败影响，不表示动作更高效。

Opus总输入/输出token为1,172,914 / 55,944；Hybrid为912,259 / 42,348；
API缓存输入两项在记录中为0。未获得可靠计费口径，不估算美元成本。
Hybrid有229个非WLA任务步骤，其中40个是8次交接的稳定等待。

## 失败分解

| 类别 | Opus | Hybrid |
|---|---:|---:|
| 重复JSON键 | 4 | 1 |
| schema额外字段 | 8 | 3 |
| 其他非法JSON/schema | 2 | 1 |
| 预算耗尽 | 5 | 3 |
| HTTP400 | 1 | 1 |
| HTTP502 | 0 | 2 |

Opus的14次、Hybrid的5次属于决策协议输出被拒绝；不能据此说机器人动作策略
已经充分执行后失败。严格拒绝保留是为了不在非法输出时猜动作。
HTTP400/502属于调用链错误，现有证据不能确定全部来自代理还是上游，
不能把它们计为纯策略无能，也不能因此从主分母消失。
没有把这些失败修掉后沿用旧实验成绩。

## 逐任务配对结果

表中“步”均为实际任务控制步，不含初始化10步。交接列为Hybrid的WLA→原子边界次数。

| case | Opus | WLA | Hybrid | Hybrid交接 |
|---|---|---|---|---:|
| spatial-00 | 预算耗尽 · 220步 | 成功 · 77步 | 成功 · 155步 | 0 |
| object-00 | 重复JSON键 · 33步 | 成功 · 141步 | 额外schema字段 · 0步 | 0 |
| spatial-01 | 额外schema字段 · 200步 | 成功 · 100步 | HTTP502 · 188步 | 2 |
| object-01 | 预算耗尽 · 280步 | 成功 · 120步 | 成功 · 116步 | 0 |
| spatial-02 | 预算耗尽 · 220步 | 成功 · 94步 | 成功 · 128步 | 0 |
| object-02 | 额外schema字段 · 21步 | 成功 · 117步 | 预算耗尽 · 280步 | 4 |
| spatial-03 | 预算耗尽 · 220步 | 成功 · 88步 | 额外schema字段 · 9步 | 0 |
| object-03 | 预算耗尽 · 280步 | 成功 · 119步 | 成功 · 117步 | 0 |
| spatial-04 | 额外schema字段 · 140步 | 成功 · 131步 | 成功 · 131步 | 0 |
| object-04 | 额外schema字段 · 148步 | 成功 · 133步 | 成功 · 131步 | 0 |
| spatial-05 | 额外schema字段 · 24步 | 成功 · 93步 | 预算耗尽 · 220步 | 0 |
| object-05 | 额外schema字段 · 21步 | 成功 · 114步 | 其他JSON/schema错误 · 11步 | 0 |
| spatial-06 | 重复JSON键 · 9步 | 成功 · 106步 | 预算耗尽 · 220步 | 1 |
| object-06 | 其他JSON/schema错误 · 3步 | 成功 · 143步 | 成功 · 143步 | 0 |
| spatial-07 | 重复JSON键 · 3步 | 成功 · 115步 | 额外schema字段 · 26步 | 0 |
| object-07 | HTTP400 · 68步 | 成功 · 135步 | HTTP400 · 0步 | 0 |
| spatial-08 | 重复JSON键 · 6步 | 成功 · 94步 | HTTP502 · 172步 | 1 |
| object-08 | 额外schema字段 · 0步 | 成功 · 151步 | 重复JSON键 · 51步 | 0 |
| spatial-09 | 额外schema字段 · 12步 | 成功 · 118步 | 成功 · 118步 | 0 |
| object-09 | 其他JSON/schema错误 · 66步 | 成功 · 127步 | 成功 · 117步 | 0 |

相对WLA-only：Hybrid没有独有成功；9个共同成功、8个WLA成功而Hybrid策略/协议失败，
另3个Hybrid基础设施错误不参与配对策略胜负统计。相对Opus-only：9个Hybrid独有成功，
8个共同失败，3个基础设施配对排除。三个模式的主率依然固定20分母。

## 轨迹里能确认的事

1. Spatial00：WLA77步成功；Hybrid先走18个原子步骤，随后18次WLA调用，共155步成功。
   WLA开始后没有原子接管。这是“前置动作+持续委托”，不是交错纠错成功。
2. Spatial04、Spatial09、Object06：Hybrid从第0任务步即调用WLA；
   总步数分别131、118、143，与对应WLA-only相同，没有接管。
   这里agent提供了调用选择，却未显示控制收益。
3. Object02：WLA117步成功；Hybrid29次WLA调用、227个WLA步，
   4次交接等待20步、WLA之后30个左右平移控制步，加上前置3步，耗尽280步而失败。
   可以确认来回接管及额外动作的成本，不能仅凭动作计数确定其视觉判断为何错。
4. Spatial06：WLA106步成功；Hybrid191个WLA步、5个交接等待步，
   前置18步和后置6个右移控制步，耗尽220步。没有观察到纠错挽救成功。
5. Spatial01/08虽然发生2次/1次交接，但分别以HTTP502结束。
   调用链提前中断，无法据此归因某个交接动作必然失败。
6. Object00、Object07：Hybrid在0个任务控制步就分别因schema/HTTP400退出，
   因此这两个失败不包含一次实际WLA调用。

## 三种模式怎么运行

完整可复制命令见`docs/RUN_GUIDE.zh.md`。

- Pure Opus：原生Show-Harness planner→controller循环；
  任务、前视/腕视、本体、近期动作和预算→原子动作→LIBERO新观察。无WLA加载。
- Pure WLA：原始完整指令+官方图像/历史/8维状态处理→原生8步预测chunk
  →统一执行器逐步执行、评分和维护历史。无Opus请求。
- Hybrid：同一Show-Harness循环可以输出原子动作或`WLA_CHUNK`；
  每次最多一个原生chunk，原始任务不改写。返回后重新观察；
  可继续委托、接回原子动作或结束；不强制交替、不把工具返回当成功。

共享单一环境所有权和实际执行历史；每步检查预算/终止；
交接后丢弃残留队列和旧请求动作。无跨回合经验记忆，不训练模型，不执行真机。
Show-Harness是完整vendor复用，上游仅两个文件的可审查修改。

## 下一轮建议（尚未实施）

先修模型输出协议稳定性，在独立新版本上统计“响应可解析率/拒绝率”，
保留不执行猜测动作的边界；再单独测试原子控制和相机/场景几何适配。
目前两个suite的机器人安装与相机几何不同，预检按suite及case比对；
这提示应审查，但不凭差异本身认定存在控制bug。
之后比较“持续委托WLA”与“允许接管”消融，查明agent什么时候介入才能受益。
不要先加经验记忆或更多策略插件，掩盖输出协议和基础控制的问题。
每次改动新冻结配置、用相同配对样本；另增加预先选定的更难样本/更多初态，
才能研究稳健收益，而不是拿这20样本满分作为WLA在LIBERO总是100%的证据。

## 完整任务指令

- `libero_spatial-00`：pick up the black bowl between the plate and the ramekin and place it on the plate
- `libero_object-00`：pick up the alphabet soup and place it in the basket
- `libero_spatial-01`：pick up the black bowl next to the ramekin and place it on the plate
- `libero_object-01`：pick up the cream cheese and place it in the basket
- `libero_spatial-02`：pick up the black bowl from table center and place it on the plate
- `libero_object-02`：pick up the salad dressing and place it in the basket
- `libero_spatial-03`：pick up the black bowl on the cookie box and place it on the plate
- `libero_object-03`：pick up the bbq sauce and place it in the basket
- `libero_spatial-04`：pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
- `libero_object-04`：pick up the ketchup and place it in the basket
- `libero_spatial-05`：pick up the black bowl on the ramekin and place it on the plate
- `libero_object-05`：pick up the tomato sauce and place it in the basket
- `libero_spatial-06`：pick up the black bowl next to the cookie box and place it on the plate
- `libero_object-06`：pick up the butter and place it in the basket
- `libero_spatial-07`：pick up the black bowl on the stove and place it on the plate
- `libero_object-07`：pick up the milk and place it in the basket
- `libero_spatial-08`：pick up the black bowl next to the plate and place it on the plate
- `libero_object-08`：pick up the chocolate pudding and place it in the basket
- `libero_spatial-09`：pick up the black bowl on the wooden cabinet and place it on the plate
- `libero_object-09`：pick up the orange juice and place it in the basket
