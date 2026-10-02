# 新连续 100k：MST+SLT 与 SAC-MLP-D1-ST

本记录分析 2026-09-29 写出的两个新主目录结果。统计来自已有 JSON / 训练记录，本次没有新运行训练或评估。
研究定位：SAC+MLP 是纯强化学习起点，MST+SLT 是强基线，full 是待逐模块加入并验证的完整方法。

## 1. 先区分新运行与旧续训

| 对象 | 本次使用的原始结果 | 训练证据 |
| --- | --- | --- |
| MST+SLT | [主目录 evaluation_results.json](../mst_slt__intersection_sorted_depart4p0/evaluation_results.json) | arguments 请求 100k；paper_evaluation_detailed 记录 trained / collected raw steps 均 100,000；training_diagnostics 记录 learner updates=95,001 |
| D1-ST | [主目录 evaluation_results.json](../sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json) | arguments 为单次 raw_budget=100k；training_complete 记录 raw_steps=100,000、updates=95,001；评估 identity 指向主目录 final_model |

两项都是新主目录中的连续 100k 运行，原始训练 seed=0；不能与 `__bak_50k/c100000` 的加载模型续训混为一组。
旧续训的 MST 25%、ST 33% 属于另一训练协议，后者没有恢复 replay buffer，且续训 warmup 改为 500。
本次结果不能被概括成“连续训练到 100k 后掉到 25% / 33%”。

## 2. 固定最终模型的评估结果

共同协议为 `intersection_sorted`、`depart_scale=4.0`、连续 SAC / base 环境、100,000 raw steps、训练 seed 0、final checkpoint；评估 seed 10000–10099，共 100 回合。
评估执行器的并行度不同：MST 为 6 workers，ST 为 1 worker。逐 seed 的交通变体已核对一致，但执行器配置差异应保留，不能把运行耗时直接当作算法效率对照。

### 训练交通分区核对：不能按标签声称 80/20 holdout

MST arguments 记录 `traffic_protocol=frozen_80_20`，D1 arguments 没有同名字段。进一步静态追踪发现，两者均调用 [yield_v2 的 make_env_factory](../train_intersection_yield_v2.py)：当 depart scale 生效时，factory 为环境设置 `_partitioned_traffic_paths = lambda spec: traffic_paths`，而缩放路径枚举了 source 下全部 traffic route 文件（约 108–138、216–239 行）。这个实例覆盖会绕过底层 train / evaluation 分区选择。

MST 的 `--traffic-protocol` 进入 runner 的参数和 provenance，但本次未找到它独立控制实际 traffic selector 的调用；D1 同样把 source 指向 sorted 目录后使用上述 factory。
因此，按当前代码路径，两种方法都应使用同一全量缩放交通变体池，没有证据支持“因为参数字段不同，所以两者训练 split 不同”。但也不能把 `frozen_80_20` 标签当作已执行的留出验证。

已有训练 monitor 不保存逐回合 traffic variant 序列，不能用日志独立复原历史实际抽样。当前结果应解释为该交通池内的策略评估，**不能声称对未见交通变体的泛化**。
下轮需显式保存 train/eval 的实际 route 文件列表与选择规则。若要建立真实 holdout，应另设统一协议并对所有比较方法使用，不应只改变一个新模型的划分后混入旧表。

| 指标 | MST+SLT | D1-ST |
| --- | ---: | ---: |
| 成功率 | 53% | 50% |
| 碰撞率 | 47% | 50% |
| 超时率 | 0% | 0% |
| 偏航率 | 0% | 0% |
| mean episode return（评估原始回报） | 0.060 | 0.000 |
| 平均决策步数（全部回合） | 80.83 | 81.23 |
| 平均 raw steps（全部回合） | 241.48 | 242.87 |
| 成功回合平均完成时间（秒） | 27.396 | 26.162 |
| 成功率 Wilson 95% 区间（按回合计算） | 43.3%–62.5% | 40.4%–59.6% |

完成时间只针对各自成功的回合，两组成功集合不同，不能凭该列宣称某模型在同一批成功任务上更快。
两份结果的逐回合记录包含 `episode, seed, episode_return, decision_steps, environment_steps, raw_steps, completion_time_seconds, success, collision, off_route, timeout, traffic_variant`。

## 3. 按共同评估 seed 配对

| | ST 成功 | ST 失败（碰撞） |
| --- | ---: | ---: |
| MST 成功 | 29 | 24 |
| MST 失败（碰撞） | 21 | 26 |

- 净成功差是 MST 多 3 个回合；但 45 个回合的成败不同。
- 不一致配对数为 24 对 21，双侧精确 McNemar 检验 p≈0.766。
- 这没有提供 MST 优于 ST 的有力证据；也不是等效检验，不能据此宣布两者等效。
- Wilson 区间和该配对检验只描述固定两个 checkpoint 的评估回合差异，并依赖相应的回合独立性假设；它们不是跨训练 seed 的置信区间，也不证明具有 100 个独立交通场景。

逐 seed 的 `traffic_variant` 在两个模型之间全部一致。100 个回合复用了 30 个交通变体文件：20 个文件各 3 回合，10 个文件各 4 回合。重复变体可能引入组内相关，以上按回合计算的区间和 p 值不作为独立场景的泛化保证。
按变体的成功数比较，MST 占优 10 个、ST 占优 8 个、12 个打平。例如 traffic_28 为 MST 3/3、ST 0/3；traffic_16 为 1/4 对 3/4。每个变体只有 3–4 回合，不能据此建立场景机制结论。

### 可用于最小行为诊断的已有种子

| 配对类型 | seed | traffic variant | MST 决策步 / 结果 | ST 决策步 / 结果 |
| --- | ---: | --- | --- | --- |
| MST 成功、ST 碰撞 | 10001 | traffic_11 | 90 / 成功 | 101 / 碰撞 |
| MST 成功、ST 碰撞 | 10003 | traffic_13 | 108 / 成功 | 88 / 碰撞 |
| MST 成功、ST 碰撞 | 10007 | traffic_17 | 77 / 成功 | 75 / 碰撞 |
| ST 成功、MST 碰撞 | 10004 | traffic_14 | 54 / 碰撞 | 96 / 成功 |
| ST 成功、MST 碰撞 | 10006 | traffic_16 | 74 / 碰撞 | 141 / 成功 |
| ST 成功、MST 碰撞 | 10011 | traffic_21 | 55 / 碰撞 | 105 / 成功 |
| 双方碰撞 | 10000 | traffic_10 | 42 / 碰撞 | 62 / 碰撞 |
| 双方碰撞 | 10008 | traffic_18 | 59 / 碰撞 | 74 / 碰撞 |
| 双方碰撞 | 10009 | traffic_19 | 66 / 碰撞 | 67 / 碰撞 |

当前 JSON 没有自车几何路线、位置、碰撞地点或碰撞对象，不能仅凭这些决策步数归因于抢行、制动过晚或观测缺失。

### 训练轨迹补充：没有独立的中途评估曲线

来源：[MST train_monitor.csv](../mst_slt__intersection_sorted_depart4p0/train_monitor.csv)、[ST train_monitor.csv](../sac_mlp_d1_st__intersection_sorted_depart4p0/train_monitor.csv)。
按完整训练 episode 的 `raw_simulation_steps` 累加，以约 20k raw steps 分段；表中是训练 episode 成功率，不能替代固定策略的评估成功率。

| raw-step 区间 | MST episode 数 / 成功率 | ST episode 数 / 成功率 |
| --- | ---: | ---: |
| 0–20k | 60 / 15.0% | 58 / 10.3% |
| 20–40k | 75 / 38.7% | 103 / 20.4% |
| 40–60k | 78 / 47.4% | 154 / 39.6% |
| 60–80k | 73 / 57.5% | 137 / 46.7% |
| 80–100k | 78 / 53.8% | 94 / 45.7% |

MST 前 20k timeout=16.7%，ST 为 19.0%；随后大部分分段 timeout=0，MST 最后分段为 1.3%。训练整体表现为超时减少、成功率和回报改善；末段的小幅回落不足以认定过拟合或训练崩塌。
完整 episode 累计止于 MST 99,731、ST 99,651 raw steps，尾部未完成回合不进入此表。两种方法 episode 长度不同，分段样本量和训练相关性不同，不对本表作独立样本的优越性检验。

训练滚动 last-20 最高成功率分别为 MST 75%（raw 97,681）、ST 80%（raw 68,297）。它们是训练窗口峰值，不是独立验证，也不是正式最终评估；不得以此替换 53% / 50%。
没有按 20k 检查点进行的独立 periodic eval。MST 只有优化诊断的全程汇总，D1-ST 缺少可对齐的 alpha / Q / loss 时间序列；目前无法据这些记录归因于熵变化、价值误差或表示崩塌。

## 4. 可以支持的结论与仍未知的内容

| 判断 | 证据强度与边界 |
| --- | --- |
| D1-ST 在本次 seed=0、100k、100 回合条件下接近强基线的点估计 | 支持；53% 对 50%，不支持稳定优越性或等效性结论 |
| 当前主要失败形式是碰撞，而非超时停车 | 支持于最终评估；双方超时为 0，碰撞为 47% / 50% |
| 两方法存在不同的成功/失败场景 | 支持；45 个配对回合成败相反，但机制尚未知 |
| ST 是从 SAC+MLP 出发值得保留的增量实验参照 | 可作为当前研究决策；其相对纯 MLP 的同预算收益还需要匹配对照 |
| 连续增加训练预算必然导致此前的巨大退化 | 不支持；旧续训与新连续运行不同。新旧差异也不能单独证明 replay buffer 清空是唯一原因 |
| SLT 无用、时空模块已被证明有效、full 将优于基线 | 均不能由本次两模型对比推出；同时缺少受控机制消融及多训练 seed 稳定性证据 |

MST+SLT 与 ST 不只差一个开关：编码器结构不同，MST 还启用 SLT，ST 只由 TD 更新表示。因此不能用本次比较孤立推断 SLT 的作用。

## 5. 下一步建议与判定分支

### 优先级 0：用已有模型做小规模配对行为诊断

选上表三类各 2–3 个 seed，共 6–9 个场景，固定两个 final checkpoint 和场景协议。
若需要重放，仅补充动作/实际速度、距冲突点、相关车辆相对位置与速度、最小 TTC、碰撞对象与位置；它是诊断性评估，不是新的训练或全量 benchmark。

- 若双方经常在同类位置/交互下失败：优先检查共同观测、动作语义和评估环境，避免直接给某个模块背负根因。
- 若某模型的失败稳定伴随特定交互差异：再把该行为解释变成一个可证伪的模块实验。
- 若诊断没有一致模式：保留未知，不能从少数可视化样本推出普遍机制。

### 优先级 1：补齐同预算的纯 SAC+MLP 对照

本次枚举及参数核对仅找到纯 MLP 的 50k 主运行和其 c100000 旧续训，未找到匹配场景下从零连续 100k 的纯 MLP 正式评估。
使用从零连续 100k、相同场景 / density / seed / reward / raw-update 规则与最终评估协议。不能用旧续训的 35% 代替这项对照。
本轮补基线应先保持现有交通池协议一致，并明确记录实际选择规则；真实 holdout 泛化实验作为独立协议开展，不在补基线时只对一个方法更换划分。

- 若纯 MLP 与 ST 接近：ST 的收益尚未建立，应先排查和验证 ST 模块，而非继续堆叠 full。
- 若 ST 显示值得跟进的改善：补独立训练 seeds 验证方向，再继续路线模块增量。
- 即便单 seed 出现较大差异，也只是初筛信号，不能直接写成稳定提升。

### 优先级 2：路线 token 的下一步受控实验

在 ST 参照可靠后，以相同从零 100k 协议比较 ST 与 ST-RT，保持辅助目标关闭。旧 50k 的 48% 对 31% 是路线接入值得排查的线索，不是其在所有预算下必然有害的结论。

- RT 仍下降：先结合行为证据定位路线接入问题，再只选择一个已有替代（如 ego-only 或 late）检验具体假设。
- RT 改善或保持且行为证据支持：再考虑 topology。
- 不直接跳到 full 来判断单个模块：当前 full 同时改变槽输出头、Graph-SLT 和 SBS。

稳定性阶段至少补到 3 个独立训练 seeds；先为决定研究方向的最小方法集合补种子，不必立刻将所有历史变体全量重跑。

## 6. 记录范围

这是依据当前产物的结果分析和实验建议，未执行建议队列。目标会议未指定，因此不作接收率或论文充分性判断。下一阶段仍是研究诊断与受控实验；有新真实结果后更新本记录与历史表。
