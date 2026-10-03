# 交叉口实验全量汇总（2026-10-01；2026-10-02更新）

> 2026-10-02 更新已核验的 sorted ST-RT+3slot、sorted ST-RT+Topo+routeaware_v1、sorted ST-RT+3slot-nonlinear 与 DARRL r2 Topo+3slot 四条正式评估。原始文件及旧统计保留；本报告当前口径为42条，41条为此前快照、38条为2026-10-01快照。

## 1. 纳入范围与读表规则

从已修正、三股背景车流全部生效的 `intersection_sorted` 开始。此前单流前身 `intersection` / 旧 `intersection_yield_v2` 的结果不纳入本表。修正标记为 `.depart_sorted`，文本为 “depart-sorted traffic (three flows all active)”。纳入依据是实际场景资产与协议，不能仅凭运行日期或方法目录名。

三个背景路线为 `-E3 → -E0`、`E0 → E3`、`E2 → E1`。自车路线为 `-E1 → -E0`。后续随机场景延续该交叉口与三路背景交通，但发车、车道和驾驶参数发生变化。

截至 2026-10-02 更新，范围内共有 **42 条非 smoke 正式最终评估**：sorted 27 条、早期随机中档及变体 4 条、DARRL r1 两条、DARRL r2 九条。2026-10-01的38条与随后41条均为此前快照；本次再补录sorted参数匹配非线性ST-RT+3slot。r1两条受终止重复标记及训练奖励污染影响，保留为历史记录，不能作为修复后的有效基线；42条中40条为互斥终局、另2条为r1污染历史。sorted routeaware和非线性3slot均已完成身份核验。

- 表中 S/C/T/O 分别为成功、碰撞、超时、偏离路线；数字单位为百分比。所有本表最终评估都是 100 回合，因此也等于回合数。
- 已有正式结果 O 均为 0，主表省略这一列；不代表以后可以忽略该指标。
- 从零训练使用 seed=0；续训承接原 seed=0 的运行，但没有证据证明完整随机状态被恢复，不能称为独立 seed=0 从零实验。
- 50k/100k 是 raw SUMO ticks，不是 50k/100k 次高层决策。主表使用最终 checkpoint，非中途最优 checkpoint。
- 奖励须按[奖励函数审计与分组](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_reward_audit_20261001.md>)区分：多数运行记录同一v2系数；DARRL r1有真实双标奖励污染；旧MST 50k与七条续训缺完整奖励记录，不能声称全部历史实现完全一致。评估回报schema也有版本差异：2026-10-02新增sorted 3slot与routeaware评估为shaped return并另存raw return，不能将其与旧raw `mean_return`混读。
- “—”表示没有找到对应正式最终评估，不能读作成功率为零。
- 有些旧结果保留了总体统计但没有逐回合记录，不能声称所有历史实验都已逐样本配对核验。
- 跨场景的数字用于描述观察；流量、驾驶参数、碰撞语义和历史训练协议存在差异，不能将跨场景变化直接归因为一个变量。

## 2. 正确三流 sorted：主场景 depart_scale=4.0

**场景：`intersection_sorted`；常见运行后缀：`intersection_sorted_depart4p0`。**

30 份固定交通 XML 模板，每份包含三股背景流，分别 200/150/240 辆，背景 lane 0 发车、lane 0 到达。自车初始 lane 2、minGap=2 m。背景发车时间戳乘 4，并使用原有循环补车机制。`depart4p0` 表示时间缩放 4 倍，不是车流量乘 4。

| 方法 | 从零 50k：S/C/T | 50k→100k 续训：S/C/T | 从零连续 100k：S/C/T |
|---|---:|---:|---:|
| SAC+MLP | 37 / 63 / 0 | 35 / 65 / 0 | 22 / 38 / 40 |
| MST+SLT | 51 / 49 / 0 | 25 / 75 / 0 | 53 / 47 / 0 |
| ST | 48 / 52 / 0 | 33 / 67 / 0 | 50 / 50 / 0 |
| ST-RT | 31 / 67 / 2 | — | 63 / 37 / 0 |
| ST-Attn | 43 / 56 / 1 | — | — |
| RT-Late | 46 / 54 / 0 | 29 / 71 / 0 | — |
| RT-Gate | 40 / 60 / 0 | 24 / 76 / 0 | — |
| RT-Ego | 44 / 56 / 0 | 36 / 62 / 2 | — |
| RT-Edge | 39 / 61 / 0 | 25 / 68 / 7 | — |
| ST-RT+Topo | — | — | 39 / 39 / 22 |
| ST-RT+3slot（无 Topo） | — | — | 34 / 49 / 17 |
| ST-RT+Topo+3slot | — | — | 41 / 56 / 3 |
| ST-RT+3slot（参数匹配非线性头） | — | — | 50 / 48 / 2 |

当前本节有25条最终评估；下一段“此表共24条评估”是加入本次非线性3slot前的旧计数。

新增非线性三槽结果来源：`runs/sortg3_1002/sac_mlp_d1_st_rt_3slot_nonlinear_v1__intersection_sorted_depart4p0/evaluation_results.json`，fresh seed0、100000 raw steps/95001 updates，final validation 100回合（逻辑seed 10000–10099），S/C/T/O=50/48/2/0。shaped/raw return mean/std=1.67910049/10.77594583、0.02/0.98974744；六项奖励覆盖100回合，最大对账误差3.55e−15，train/eval diagnostic errors=0。final checkpoint、training_complete与evaluation identity SHA256一致：`cb8b19bd67c982eddf0d377f54b0f7b03fcce7ed9261328ed88790228c479306`。Shadow采样为训练20状态/240行、评估200状态/2400行；active有效160/1600、NA80/800，invalid/exception=0。身份和采集收据见[完成记录](sorted_goalonly_nonlinear3slot_completion_20261002.json)。相对既有线性ST-RT+3slot（34/49/17）是+16/−1/−15个百分点，相对ST-RT（63/37/0）是−13/+11/+2个百分点；均为单训练seed固定评估池的描述性差异，不足以归因语义槽收益。
| ST-RT+Topo+routeaware_v1 | — | — | 43 / 32 / 25 |

此表共 24 条评估。新完成的 sorted ST-RT+3slot 为从零连续100k、seed=0、final validation 100回合；`environment_step_reward_v2` shaped episode return mean/std = −1.3884/9.8977，另存 raw undiscounted return mean/std = −0.15/0.8986，六项奖励分量最大对账误差为3.55e−15，train/eval diagnostic error均为0。checkpoint SHA256为`d46b5cf71cece2a39e589a631c6624eee3f9807792ef9a77443c4aa17c00dd1d`。新增routeaware同为fresh seed=0、100000 raw steps/95001 updates、final validation 100回合，S/C/T/O=43/32/25/0；shaped return mean/std=1.0396/9.9574，raw undiscounted mean/std=0.11/0.8590，六项奖励对账最大误差3.55e−15，train/eval diagnostic error均为0，final checkpoint、training-complete与evaluation identity SHA256一致：`cee4041ceb636c77ecb6a8e6c43ed5e1ce74aa4002b486786bb4d5f8104107c8`。二者新评估均采用`environment_step_reward_v2`；shaped `mean_return`不能与本表早期旧raw `mean_return`直接比较。完整逐方法身份、交通模板核验和配对矩阵见[sorted模块结果审计](sorted_module_outcomes_20261002.json)及[完成快照](sorted_pair_completion_20261002.json)；routeaware原始评估位于`runs/sort2_1001/sac_mlp_d1_st_rt_topo_routeaware_v1__intersection_sorted_depart4p0/evaluation_results.json`。同组7种策略均用训练seed 0和相同100个评估逻辑seed、30个固定交通模板；这是固定checkpoint的一次评估比较，不是多训练seed稳定性估计。**续训列不能替代从零连续 100k 列**：已核实的续训为 base 50k + extra 50k，未恢复 replay buffer，续训 warmup 从 5000 改为 500 raw；记录的累计更新次数为 94,502。连续 100k 的更新次数为 95,001。因而不能把三列直接当作同一模型一条连续学习曲线。

旧 MST+SLT 与纯 SAC 的部分训练流量划分元数据存在不确定性：旧 MST 请求标签写 frozen_80_20，而旧 SAC 档案记录使用完整 30 模板池；实际旧 MST 训练池是否与 SAC 相同尚未完全证实。旧环境源码也没有完整冻结，不能用当前碰撞实现替代旧实现作确定解释。

原始结果主要位置：

- 从零 100k MST：[evaluation_results.json](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/mst_slt__intersection_sorted_depart4p0/evaluation_results.json>)。
- 从零 100k ST：[evaluation_results.json](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json>)。
- 从零 100k SAC / ST-RT：`runs/d0929_100k_diag`。
- 从零 100k Topo / Topo+3slot：`runs/t0930_topo3_retry01`。
- 50k 与续训的逐实验源路径见 [本地结果清单](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_inventory_local_20261001.json>)；已有说明见 [历史实验记录](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/experiment_history.md>)。

## 3. 正确三流 sorted：额外密度对照

仍为 `intersection_sorted`，下表与上节 depart_scale=4.0 分开标注。均为纯 SAC+MLP、从零 50k、seed=0、最终评估 100 回合。

| 实际配置 | 成功 S | 碰撞 C | 超时 T |
|---|---:|---:|---:|
| depart_scale=2.5 | 30 | 70 | 0 |
| depart_scale=3.0 | 0 | 98 | 2 |
| depart_scale=4.0（上一表重复作参照，不重复计数） | 37 | 63 | 0 |

前两行分别来自：
- [depart2p5 原始结果](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_depart2p5__intersection_sorted/evaluation_results.json>)。
- [depart3p0 原始结果](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_depart3p0__intersection_sorted/evaluation_results.json>)。

两条的 manifest、50k 完成记录、100 回合种子及 final checkpoint SHA 已核对。时间缩放越大，模板中的请求发车间隔越长；由于仍有插入约束和循环补车，不能据此把实际在网密度简单当作严格反比例。

## 4. 早期随机场景：原生 SUMO flow 三档

这一族背景仍为 lane 0 → lane 0，使用原有车型/驾驶参数分布，不能与后面的 DARRL 覆写版本混为一谈。

| 场景名 | 三路额定请求流量（veh/h） | 已有正式结果 |
|---|---|---|
| `intersection_random_low_v1` | 200 / 150 / 240，总计 590 | 未找到正式最终评估 |
| `intersection_random_medium_v1` | 400 / 300 / 480，总计 1180 | SAC+MLP：75 / 25 / 0 |
| `intersection_random_high_v1` | 600 / 450 / 720，总计 1770 | 未找到正式最终评估 |

中档结果为从零连续 100k、seed=0、最终评估 100 回合，来源 `runs/smlp_p05_a0930`。这里的数值是原生 flow 的请求流量，不能与“每个 0.1 秒步生成一辆车的概率”直接混用，也不是保证实际进入路网的流量。

## 5. 早期随机中档：显式逐 0.1 秒概率变体

三个进口分别独立采样，每路每个 0.1 秒步按 p 请求生成一辆车；实际插入仍受 SUMO 空间/安全条件约束。背景仍为 lane 0 → lane 0，未应用 DARRL 参数覆写。所有下表实验为纯 SAC+MLP、从零连续 100k、seed=0、最终评估 100 回合。

| 场景名 | 每路每 0.1 s 的 p | 成功 S | 碰撞 C | 超时 T |
|---|---:|---:|---:|---:|
| `intersection_random_medium_p05_v1` | **0.5** | 12 | 88 | 0 |
| `intersection_random_medium_p03_v1` | **0.3** | 17 | 83 | 0 |
| `intersection_random_medium_p02_v1` | **0.2** | 5 | 25 | 70 |

特别注意：这里的 `p05` 是 **0.5**，不是 DARRL 的 **0.05**；`p03` 同理是 **0.3**，不是 **0.03**。

来源：p=.5 为 `runs/smlp_p05_b0930`；p=.2/.3 为 `runs/smlp_p03p02_0930`。不能仅凭名称中的 low/medium 或数值 p 排序，假定成功率应单调；本表只报告已观察结果。

## 6. DARRL r1：旧概率与未修复终止标记

**同名场景旧配置**：low/medium/high 的每路每 0.1 秒概率分别为 **0.03 / 0.05 / 0.07**。

背景 lane 1 发车、lane 1 到达，jmIgnoreFoeProb=0；自车 minGap=1 m、jmIgnoreFoeProb=0；碰撞车辆 remove。使用当前研究要求的碰撞事件/几何判据，不移植 DARRL legacy 的距离/车辆消失判据。r1 当时存在终止标记重叠问题。

| 场景名及旧配置 | 方法/预算 | 原始成功标志率 | 原始碰撞标志率 | 原始超时率 | 可比性 |
|---|---|---:|---:|---:|---|
| `intersection_random_darrl_low_v1`，p=.03 | SAC+MLP，100k | 30 | 73 | 0 | 3 回合同时成功与碰撞 |
| `intersection_random_darrl_medium_v1`，p=.05 | SAC+MLP，100k | 13 | 92 | 0 | 5 回合同时成功与碰撞 |
| `intersection_random_darrl_high_v1`，p=.07 | — | — | — | — | 未找到正式最终评估 |

来源：`runs/smlp_darrl_0930`。

这两条保留原始标志率，因此合计 103%/105%。**不能作为修复后互斥终局口径的有效基线**。按碰撞优先重数，评估可得到低档 27/73/0、中档 8/92/0；但这只是同一旧评估的重新计数，不是新实验，也不修复已发生的训练奖励污染。历史审计发现低/中训练分别有 29/10 条双标记录受成功奖励优先问题影响。

## 7. DARRL r2：当前概率与互斥终局

**当前配置修订：`darrl_r2_20260930`。**

| 场景名 | 每路每 0.1 s 的 p | 配置状态 |
|---|---:|---|
| `intersection_random_darrl_low_v1` | 0.015 | 当前低档 |
| `intersection_random_darrl_medium_v1` | 0.03 | 当前中档 |
| `intersection_random_darrl_high_v1` | 0.05 | 当前高档；未找到正式最终评估 |

场景名没有新增，概率是原地修改，因此结果必须附带 r1/r2 修订。lane 1→1、minGap=1 m、jmIgnoreFoeProb=0、remove 等 DARRL 场景参数保持；终止统计修为 `exclusive_terminal_v2`，优先级 collision > off_route > success > timeout。

### 当前低档：p=.015

| 方法 | 预算 | 成功 S | 碰撞 C | 超时 T |
|---|---:|---:|---:|---:|
| SAC+MLP | 从零连续 100k | 62 | 34 | 4 |

来源：`runs/smlp_dr2_0930`。

### 当前中档：p=.03

| 方法 | 预算/状态 | 成功 S | 碰撞 C | 超时 T |
|---|---|---:|---:|---:|
| SAC+MLP | 从零连续 100k，已完成 | 31 | 62 | 7 |
| MST+SLT | 从零连续 100k，已完成 | 16 | 53 | 31 |
| ST | 从零连续 100k，已完成 | 16 | 83 | 1 |
| ST-RT | 从零连续 100k，已完成 | 14 | 66 | 20 |
| ST-RT+Topo | 从零连续 100k，已完成 | 31 | 69 | 0 |
| ST-RT+Topo+routeaware_v1 | 从零连续 100k，已完成 | 31 | 45 | 24 |
| ST-RT+3slot（无 Topo） | 从零连续 100k，已完成 | 30 | 70 | 0 |
| ST-RT+Topo+3slot | 从零连续 100k，已完成 | 20 | 47 | 33 |

此处所有已完成条目为 train seed=0，外层 deterministic final evaluation=100 回合，逻辑评估 seed=10000–10099。新增Topo+3slot完成100000 raw steps/95001 updates，final checkpoint与evaluation身份一致。最新套件 MST 存在内部额外最终评估，表中只计统一外层 `evaluation_results.json`，不将其重复算作另一个实验。

SAC 来源 `runs/smlp_dr2_0930`，七方法队列来源 `runs/dm7_1001_retry02`。Topo+3slot 在2026-10-01 20:22检查时尚未完成；随后已在同一正式root完成fresh100k训练和final100评估，现补录为 **20 / 47 / 33**（S/C/T），raw `mean_return=-0.27`。checkpoint SHA256为`b89d891756dbb8f21dfc5225ecd8efe3e88b3cb70e85ba07dff97b46984dc77d`，原始结果位于`runs/dm7_1001_retry02/sac_mlp_d1_st_rt_topo_3slot__intersection_random_darrl_medium_v1_depart1p0/evaluation_results.json`。该DARRL结果为旧raw评估schema，不能与sorted新增结果的shaped `mean_return`数值直接比较；此前pending文字仅描述当时状态，现由完成结果取代。

ST-RT+3slot 是已有的线性三槽头版本；不能把它写成尚未实验的其他三槽改进方案。routeaware_v1 是为 Topo 增加明确路线可达性条件的独立新增方法。

## 8. 未纳入正式性能表的运行与缺失项

- 6 份 runs 外层 smoke 评估：
  - `d0929_smoke_diag_02`：SAC、ST-RT，各 300 raw / 8 回合评估。
  - `t0930_topo3_smoke_01`：Topo、Topo+3slot，各 300 raw / 2 回合评估。
  - `p03_sm0930`、`p02_sm0930`：各 300 raw / 1 回合评估。
- 失败启动、路径长度失败、已被替代的重试目录，以及只有训练统计而没有正式最终评估的运行，在运行清单中保留状态，不编造最终成功率。
- `_p4_lowdensity_s*` 等未找到正式最终评估的目录，不能加入 S/C/T 表。
- 原随机 low/high、DARRL r1/r2 high 均未找到正式最终评估。
- **full 完整方法**：本次纳入的正确三流及后续场景中，未找到可用正式最终评估；旧 yield_v2 下的 D1-Full 训练文件不符合本次场景纳入范围。MST+SLT 始终作为强基线，不视为 full。
- 同名场景版本、同方法不同预算、续训以及配置不同的运行均保留，不只挑最高数值。

## 9. 从汇总可直接确认的阶段性观察

1. 在旧 sorted / depart4 从零连续 100k 中，ST-RT成功率63%，Topo 39%，Topo+3slot 41%，线性ST-RT+3slot为34%，参数匹配非线性ST-RT+3slot为50%，新增Topo+routeaware为43%；均为单训练seed下该场景/checkpoint的描述性结果。
2. 在当前 DARRL r2 中档 / p=.03，已完成方法的成功率最高目前为 31%，SAC+MLP、Topo、Topo+routeaware 同为 31%；其碰撞/超时组合分别为 62/7、69/0、45/24，因此不能只按成功率判为行为等价。
3. 方法相对表现会随场景/协议变化。不能用旧 sorted 的排序替代当前中档排序，也不能将跨版本变化直接当成单变量因果证据。
4. DARRL r2中档Topo+3slot最终评估现已补入，为20/47/33；单seed结果不足以支持Topo×3slot普遍交互结论。
5. 保持既定研究顺序：先比较强/纯基线，再逐模块比较；本次更新记录此前正式run完成的结果，没有新启动训练或评估。

## 10. 可追溯文件与审计范围

- [场景配置及沿革清单](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_scenario_catalog_20261001.json>)。
- [runs 结果与运行状态清单](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_inventory_runs_20261001.json>)：这是2026-10-01旧清单快照（25个根目录、24份外层评估JSON；18份非smoke、6份smoke），不含2026-10-02后来完成的四条结果；当前42条计数以本报告及对应run的完成/评估文件为准。
- [FD 本地结果清单](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_inventory_local_20261001.json>)：正确 sorted 的本地结果、50k/续训与额外密度变体，并记录排除项。
- [当前中档 MST+SLT 与 SAC+MLP 分析](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/darrl_medium_mst_vs_sac_analysis_20261001.md>)。
- [研究上下文](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/RESEARCH_CONTEXT.md>)、[完整实现地图](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/full_mst_slt_implementation.md>)、[历史实验记录](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/experiment_history.md>)。

本报告依据现存原始结果、manifest、完成记录和场景档案，不通过补跑来填充缺失值。单训练种子结果反映本轮观察，不支持多训练种子的稳定性结论。

## 11. 奖励函数与实验组的对应关系（本次补充）

共同 v2 数值设计为：成功 +10、碰撞 −10、偏离路线 −10、超时 −5；每策略决策 −0.01；可读取前后累计行驶里程时再加 0.02 × 里程差（米）。这不是每 raw tick 奖励，也不是距目标减少量。

| 本报告实验组 | 奖励标注 |
|---|---|
| sorted depart4 从零 50k | SAC/D1 参数确认相同 v2；MST 50k 缺完整历史奖励证据 |
| sorted depart2.5 / depart3 从零 50k | 两条 SAC 参数确认相同 v2 |
| sorted 50k→100k 续训 | 未发现回报尺度异常，但续训缺 reward/action_repeat/gamma 字段，尚未完全核实 |
| sorted 从零连续 100k | 参数/奖励分量支持同一 v2；不代表所有历史终局源码已冻结核实 |
| 原随机 medium、p=.5/.3/.2 | 参数/奖励分量支持同一 v2 |
| DARRL r1 low / medium | 名义 v2 相同，实际存在成功/碰撞双标奖励污染，单列不可比 |
| DARRL r2 low / medium | 同一 v2 数值设计＋互斥终局；碰撞优先 |

截至2026-10-01的历史评估 `mean_return` 与终局success−collision均值相符（成功 +1、碰撞 −1、timeout/off-route 0），不是训练奖励。2026-10-02新增sorted ST-RT+3slot（线性与非线性）及routeaware评估采用`environment_step_reward_v2` shaped `mean_return`，并另存raw return及奖励分量；新增DARRL r2 Topo+3slot仍为旧raw评估schema。SLT等辅助损失不是环境reward。完整公式、每组证据等级和原始路径见[奖励函数审计](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_reward_audit_20261001.md>)。

## 12. 2026-10-02：前三条完成结果补记（此前计数快照）

新增已核验最终评估包括 sorted `sac_mlp_d1_st_rt_3slot` 34/49/17/0（S/C/T/O）、sorted `sac_mlp_d1_st_rt_topo_routeaware_v1` 43/32/25/0，以及 DARRL r2 medium `sac_mlp_d1_st_rt_topo_3slot` 20/47/33/0。三者各自为seed0、fresh100k/95001 updates、final100；DARRL场景与sorted评估回报schema不同，分别记录，不当成同场景配对消融。sorted新评估采用`environment_step_reward_v2`：3slot shaped mean/std=−1.3883924/9.8976625、raw mean/std=−0.15/0.8986100；routeaware shaped mean/std=1.0395563/9.9573802、raw mean/std=0.11/0.8590111。两法六分量对账最大误差均3.55e−15，诊断错误数均0；routeaware final模型SHA=`cee4041ceb636c77ecb6a8e6c43ed5e1ce74aa4002b486786bb4d5f8104107c8`。详细协议、身份、配对矩阵和单训练seed限制见[sorted模块结果审计](sorted_module_outcomes_20261002.json)及[完成快照](sorted_pair_completion_20261002.json)；DARRL原始文件路径见第7节。

此前统计为41条（sorted26、早期随机4、DARRL r1两条污染记录、DARRL r2九条；39条互斥终局、2条r1污染历史）；非线性三槽完成后为42条（sorted27；40条互斥终局、2条r1污染历史）。本次goal-only完成后当前共43条（sorted28、早期随机4、DARRL r1两条污染记录、DARRL r2九条；41条互斥终局、2条r1污染历史）。此前2026-10-02 03:35:28(+08)的routeaware运行快照已由最终模型和100回合评估取代。旧sorted/depart4阶段的七方法及两新增方法均为单训练seed=0、fresh100k raw；100回合使用相同逻辑seed与30个固定traffic模板，配对审计核验模板内容哈希一致。该配对比较描述固定策略在这批评估交通上的结果，不能估计多训练seed稳定性；新增sorted评估为shaped `environment_step_reward_v2`，不可将其shaped `mean_return`与旧raw `mean_return`直接比较。模块归因报告见[sorted_module_attribution_20261002.md](sorted_module_attribution_20261002.md)；本段只记录最终结果和协议核验，没有启动训练或评估。

后续规则：每次正式实验完成且final checkpoint、评估回合数与评估文件身份核验后，及时更新本汇总及`RESEARCH_CONTEXT.md`、`experiment_history.md`；在核验前只记运行快照，不写完成或性能结论。

## 13. 2026-10-02：sorted 增量链的归因摘要

同一 sorted/depart4、seed0、fresh100k评估中，ST-RT为63/37/0（S/C/T），是此前本增量链最佳观测；Topo为39/39/22，route-aware Topo为43/32/25，ST-RT+3slot为34/49/17，Topo+3slot为41/56/3，参数匹配非线性ST-RT+3slot为50/48/2，goal-only Topo为24/47/29。route-aware的25个超时均终止于不能接入目标下一edge的`-E1_0`，而goal合法attention mass约为1、bypass/fallback为0；这是mask有效但未约束动作选择的证据，不是mask造成超时的因果证明。旧线性3slot与新增非线性3slot还改变跨槽连接结构，下游actor/critic仍可混合槽输出，因此固定单seed结果不构成槽语义的因果结论。

归因和实现证据见[模块归因报告](sorted_module_attribution_20261002.md)、[配对结果](sorted_module_outcomes_20261002.json)及[正常流程诊断](sorted_module_diagnostics_20261002.json)。非线性3slot与goal-only Topo均已完成；对旧ST-RT、Topo、route-aware与旧线性3slot的配对矩阵见本节后续条目和[配对审计JSON](sorted_goalonly_nonlinear3slot_pairing_20261002.json)。后续机制归因仍待进行，不把固定单seed配对差异写成模块因果。现有residual初始化尺度已有1e−3，故不将简单增加小门控重复包装成新方案。解释仍限于单训练seed与复用30个固定traffic模板；新shaped评估回报不与旧raw `mean_return`混读。

面向后续 incremental train/eval，新增 actor-intent post-LN 实际 delta 与相同支持集上的 uniform 关系基准；已有 goal post-LN 指标复用，历史 source/results 不变。相关 encoder 30 项与 module diagnostics 7 项测试通过（4项目标测试包含在内），本次未启动训练、评估或 SUMO；旧 full delegate 不提供新增字段，见[诊断采集协议](module_diagnostics_protocol_20261001.md)。本次分析和诊断补充完成后，automation-3 已由应用工具确认暂停，避免重复跟进。

## 14. 2026-10-02：sorted goal-only Topo 最终评估

`runs/sortg3_1002/sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0`已核验fresh seed0、100000 raw steps/95001 updates、final validation 100回合（逻辑seed 10000–10099），S/C/T/O=24/47/29/0。Shaped/raw mean/std=−2.94594740/8.78128240与−0.23/0.81061705；六项奖励分量覆盖100回合、最大对账误差3.55e−15，train/eval诊断错误均0。Final checkpoint、training_complete与evaluation identity SHA一致：`17465d202736d4640d46459f78bc4b55cef939ab6e3aa39b9cad63a981f1c5f5`。Shadow训练20状态/240行、评估200状态/2400行（100回合各2状态，预算上限3）；每状态12个variant，active有效140/1400、NA100/1000，invalid/exception=0。

按eval seed与traffic variant，goal-only与ST-RT、Topo及route-aware各100/100配对，100个逻辑seed映射完全一致；当前与四个先前run根的30个有效route XML逐个SHA一致。矩阵方向为reference outcome行→goal-only列，类别列序S/C/T/O：ST-RT→goal-only行S=[15,33,15,0]、C=[9,14,14,0]；Topo→goal-only行S=[11,19,9,0]、C=[7,18,14,0]、T=[6,10,6,0]；route-aware→goal-only行S=[14,17,12,0]、C=[7,15,10,0]、T=[3,15,7,0]。完整配对数据与只读复现脚本见[配对JSON](sorted_goalonly_nonlinear3slot_pairing_20261002.json)和[脚本](sorted_goalonly_nonlinear3slot_pairing_20261002.py)，身份收据见[完成receipt](sorted_goalonly_nonlinear3slot_completion_20261002.json)。

本次加入后全量清单共43条正式非smoke最终评估（sorted28、早期随机4、DARRL r1污染历史2、DARRL r2 9；41条互斥终局、2条r1污染历史）。这是单训练seed下固定100回合/30模板的结果；不作为多训练seed稳定性或模块因果结论。新shaped回报与旧raw回报不可直接比较；用户已授权的额外诊断评估尚未运行，根因分析仍待进行。

## 15. 2026-10-02：冻结 goal-only 三臂诊断补记

同一goal-only final checkpoint上完成额外的冻结策略三臂诊断（不是新训练，不改变正式43条计数）：control、topology-goal-off、route-lane-veto分别100回合，S/C/T/O为24/47/29/0、45/55/0/0、44/56/0/0。Control对原正式final评估100/100逐seed复现。Shaped mean return为−2.945947/0.943300/0.703164；raw `info.undiscounted_reward`均值为−.23/−.10/−.12；behavior `return_base`均值为−.227306/−.098606/−.119301，因action-repeat decision内折扣而不等于raw return。三臂六项shaped reward最大对账误差3.55e−15，权重冻结、95001 updates及checkpoint SHA一致。

Control和route-veto shadow分别覆盖300和297个unique state，每state 12 variants；active错误为0、inactive为显式NA，goal-off按设计关闭shadow。Route-veto共记录1778次hold veto，保持原纵向动作；该动作侧干预不能保证车辆完成变道或避免碰撞。结果及精确来源见[三臂离线审计](sorted_goalonly_causal_results_20261002.json)、[probe覆盖审计](sorted_goalonly_causal_probe_audit_20261002.json)、[派生归因复核](sortg3_causal_1002_attribution.json)与[主归因报告](sorted_task_module_attribution_20261002.md)。三臂均为单训练seed下固定评估池的描述性干预，不等于重新训练消融或安全保证；本次不新增full训练。


## 16. 后续critical shadow采样预算扩展

新增cap=4采样器供后续评估使用：每回合最多4个unique state，分别保留initial、首个route-critical、首个TTC-critical、preterminal；同decision route/TTC触发合并。需同时传入 `--behavior-diagnostics --policy-shadow-probes --policy-shadow-critical-eval-sample --policy-shadow-eval-max-per-episode 4`。sample对应致事件动作的pre-action predict输入，context为action-repeat结束后事件窗口；保留global/episode raw时钟字段 `source_action_pre_obs_raw_global_step` / `_episode_step`、`context_window_start_raw_global_step` / `_episode_step`、`context_window_end_raw_global_step` / `_episode_step`。cap=3兼容且默认关闭，训练采样按5000 raw间隔触发、全训练阶段最多20个unique state。针对性pytest 10项由文档代理在main1正确工作区通过；本次既有300回合冻结诊断仍按cap=3，既有训练/eval结果不变。
## 17. 2026-10-02—10-03：sorted RouteAct / ConflictTiming 完成训练与final100

正式双worker根为 [`runs/sortct_1002`](../../../../runs/sortct_1002)，运行配置、launcher状态及源码档案见[`suite_manifest.json`](../../../../runs/sortct_1002/suite_manifest.json)、[`launcher_status.json`](../../../../runs/sortct_1002/launcher_status.json)与[`source_archive/manifest.json`](../../../../runs/sortct_1002/source_archive/manifest.json)。supervisor PID 9984。约2026-10-02 20:52(+08)的runtime快照及file audit核实RouteAct worker PID 79656、ConflictTiming worker PID 36840均从预期main1项目根启动、导入源码SHA与source archive一致且实际使用CUDA；两worker仍running，进度分别35493 raw/30493 updates与34090 raw/29090 updates。此条只记已核验的运行身份/启动进度，不增加已完成正式评估计数，也不记录性能结果。

| 方法 | 本轮改动 | 状态 |
|---|---|---|
| `sac_mlp_d1_st_rt_routeact_v1` | 保持ST-RT网络、观测、奖励、优化；仅在执行侧将已知路线不可达的横向请求映射为hold，proposal仍用于replay | 完成：47/53/0/0（S/C/T/O） |
| `sac_mlp_d1_st_rt_conflicttime_v1` | 保留ST-RT，新增每actor 20维`conflict_timing`及零初始化20→32→128 social K/V residual（4896活跃参数） | 完成：40/57/3/0（S/C/T/O） |

共同配置已核验：`intersection_sorted`/depart_scale=4.0、训练seed0、每臂fresh 100000 raw steps且不resume、learning starts 5000 raw、action repeat 3、10000 raw checkpoint；final checkpoint在validation逻辑seed10000–10099确定性评估100回合。traffic使用相同30模板完整池，无holdout。behavior/shadow诊断开启，训练采样按5000 raw间隔触发、全训练阶段最多20个unique state，eval cap4并启用critical sampling；ConflictTiming校准sidecar保留每phase最多50000 prediction rows、200000 calibration rows、pending 1024的协议预算。这些预算不增加仿真步或评估回合。

| 方法 | shaped return mean/std | raw return mean/std | checkpoint SHA256 |
|---|---:|---:|---|
| RouteAct | 0.72763084 / 10.77789482 | −0.06 / 0.99819838 | `8a36cd283d13e08b0c83cd16d988147c1e1213f7e13a610731c1a8532176bebe` |
| ConflictTiming | −0.63535305 / 10.50959593 | −0.17 / 0.97010309 | `58ce993f4cbf1068c46a0bc52d463d2a7b028a8afb7d1b128cf6c2c7e84da9a5` |

两方法均为100000 raw/95001 updates，final checkpoint、training-complete与evaluation SHA一致；六分量reward对账最大误差3.55e−15、train/eval诊断错误数均0。身份、reward与匹配历史ST-RT矩阵见[最终审计JSON](sortct_1002_routeact_conflicttime_final_audit_20261003.json)，总体任务/模块归因见[归因报告](sorted_routeact_conflicttime_attribution_20261003.md)。RouteAct veto阶段数见[训练聚合](routeact_training_diagnostic_20261003.json)，行为结果见[评估聚合](routeact_eval_behavior_20261003.json)；几何碰撞partner基于同tick重叠规则离线重建见[partner审计](routeact_collision_partner_reconstruction_20261003.json)。

RouteAct冻结ST-RT先导的9003次决策中0次veto，仅证明原策略部署不触发映射，不能代替本轮从零训练。当前结果是描述性的单seed固定checkpoint评估；veto不等于救援，ConflictTiming添加了静态地图/当前轨迹信息且增加参数，不能视为等信息量表示消融。行为和功能依赖仍在归因，不由attention、梯度、veto数直接推断性能因果。方法边界见[协议](route_action_conflict_timing_protocol_20261002.md)、[冻结ST-RT先导](st_rt_routeact_frozen_check_20261002.md)和[归因报告](sorted_routeact_conflicttime_attribution_20261003.md)。

## 18. 2026-10-03：冻结策略诊断排队状态

`runs/sortct_frozen_1003`由coordinator PID 37380管理。两个control精确复现检查已通过，另有3臂×40回合排队、2 worker、总预算上限122。该批是已完成checkpoint的诊断评估，不是新训练或新增正式训练结果；当前无可报告的诊断终局。后续状态以该根manifest/receipt核验为准。

## 19. 2026-10-03：RouteAct / ConflictTiming 冻结诊断已完成

第18节保留为排队时状态快照；该冻结评估现已完成：两项单回合control gate加三个40回合干预臂，共122回合、无训练。RouteAct前40正式评估参考15/25/0，去除横向veto后为6/26/8；ConflictTiming参考12/26/2，关闭整支后为19/20/1，仅关闭时间通道后为11/28/1（均S/C/T；off-route均0）。它们是在匹配验证seed与30模板上的固定checkpoint干预，不能当作新训练或新增final-100方法结果。冻结run和完整配对矩阵见[冻结结果说明](sortct_frozen_intervention_results_20261003.md)、[机器可读结果](sortct_frozen_intervention_results_20261003.json)及[总体归因报告](sorted_routeact_conflicttime_attribution_20261003.md)。

另有环境几何碰撞首命中对象的被动日志补充，保持原bool碰撞路径和首命中短路；snapshot只匹配既有actor/time，不加TraCI查询或仿真步。两个指定纯测试文件15项通过，`py_compile`通过；一次早前误带真实`test_sumo_env.py`的pytest执行边界与未知raw步数见[碰撞日志/误调用审计](geometric_collision_evidence_logging_20261003.md)。该额外测试不属于122回合冻结预算，不据此产生或修改正式性能结果。报告依据已验证结果，不启动新训练或评估。

## 20. 2026-10-03：ConflictTiming 校准日志补充与当前收尾状态

未来的`conflict_timing`校准候选路径记录补充了`path_deviation_evidence`：分别保留ego/foe首个偏离的raw与decision时刻、道路/车道/位置、横距/弧长、lane membership、原因、候选lane IDs及已知数据截断；每侧entry/clearance仍以实际观测为准，既有日志不回填。该补充不增加TraCI查询、仿真步或prediction row；7项wrapper纯测试与几何碰撞相关15项共22项通过。整体方法结果与限度见[最终归因报告](sorted_routeact_conflicttime_attribution_20261003.md)。正式两路100k/final100及122回合冻结诊断均已完成，正常采集补充已完成；此后没有新增完整训练，本轮跟进可暂停。

*** End of File
## 22. 2026-10-03：D1 C8/C9 contractfix pair queued

新增独立入口注册 `sac_mlp_d1_st_contractfix_v1` 与 `sac_mlp_d1_st_rt_contractfix_v1`，用于在前序ST-RT/Longres pair完成且身份核验通过后fresh检验D1 last-valid history index（C8）与SMARTS几何边速度合同（C9）的联合修复。训练入口/launcher和CPU SB3 checkpoint save/load roundtrip已验证；前序pair completion guard目前阻止启动，故新实验仍pending且没有性能结果。具体协议见[d1_contractfix_protocol_20261003.md](d1_contractfix_protocol_20261003.md)。

## 23. 2026-10-03：retry01 ST-RT 完成，Longres pending

ST-RT `sac_mlp_d1_st_rt`在`runs/sortlr_1003_retry01`已fresh seed0完成100000 raw steps/95001 updates；final validation为100回合、seeds 10000–10099，S/C/T/O=63/37/0/0、9003 decisions。Final checkpoint SHA256 `cce43cacb359ac7688858d514dce0004b64a296ef908b3818de76d01d75274ed`与training-complete和evaluation identity相符。Shaped `environment_step_reward_v2` mean/std为4.3898484078/10.3890093033；raw未折扣return独立列为0.26/0.9656086164。六项shaped奖励分量均覆盖100回合，最大reconciliation error 3.55e−15；train/eval diagnostic errors均为0。该fresh重跑与先前同seed的63/37/0、9003 decisions/回报复现一致，不构成独立seed证据。

完整证据位于[ST-RT run目录](../../../../runs/sortlr_1003_retry01/sac_mlp_d1_st_rt__intersection_sorted_depart4p0)、[pair suite manifest](../../../../runs/sortlr_1003_retry01/suite_manifest.json)、[source archive manifest](../../../../runs/sortlr_1003_retry01/source_archive_manifest.json)及[experiment history](experiment_history.md)。训练终结记录为100000/95001；稀疏`progress.json`末值99715/94715仅保留为较早快照。Longres在2026-10-03 08:16(+08)仍training于99690 raw/94690 updates且进程存活，因此有效pair尚未完成，contractfix仍pending、未启动。D1 as-run last-valid index风险继续约束模块归因，MST+SLT first-frame mask不由此定性。
## 21. 2026-10-03：ST-RT / Longres 中断与恢复运行快照

原正式根 runs/sortlr_1003 经用户确认手动中断，raw历史保留：ST-RT 9276 raw/4276 updates、Longres 8679/3679；诊断错误均0。没有周期/最终checkpoint、replay buffer或RNG/optimizer状态，不能精确续训；详见 runs/sortlr_1003/interruption_receipt.json。上述部分训练不并入fresh重跑的100000 raw预算。

当前正式比较配对为 runs/sortlr_1003_retry01 的ST-RT和 runs/sortlr_1003_retry01_longres 的Longres。两臂均为 intersection_sorted/depart_scale=4.0、seed0 fresh、no-resume、CUDA、每臂100000 raw，最终validation逻辑seed10000–10099共100回合。最后一次状态文件快照写于2026-10-03 04:23(+08)：ST-RT 18852 raw/13852 updates，Longres 12571/7571；child status仍为training，尚无final评估，不能报告性能结果。恢复身份、失败候选和每文件source/traffic哈希见 runs/sortlr_1003_pair_recovery_receipt_20261003.json。

retry01中的Longres候选因同根并发写traffic overlay触发WinError 5退出；独立Longres root避免该写入冲突。两selected roots的30个 traffic_*.rou.xml 文件名与SHA256全部一致，depart_scale同为4.0。两个归档各8736文件，已核对的16项训练关键源码一致，唯一两项source-manifest差异是未被worker导入的launcher helper和独立window诊断脚本。

两个selected root各有一个10k边界policy observation样本，trajectory shape为6×10×5。生产x!=0 proxy下所有slot均为10/10帧；count-minus-one与真实proxy最后索引均为9，样本未见idx mismatch或索引到proxy-masked帧。样本稀疏且proxy不是真实车辆presence；由于缺actor ID和history tracking/reset timestamps，late-entrant短历史频率未知。不得用这两个样本估计总体暴露率或性能影响。D1 left-pad/count-minus-one问题是运行后发现的实现风险；当前正式运行未热改或停止，原始运行指标按as-run记录，模块设计归因需重新审视。MST+SLT的first-frame mask行为不据此定性。细节见[history-index审计](last_valid_history_index_audit_20261003.md)及配对恢复收据。
*** End of File

## 25. 2026-10-03：retry01 Longres 完成，ST-RT / Longres pair身份门通过

上一节保留ST-RT完成、Longres仍在training的08:16(+08)快照；Longres现已完成并核验最终身份。其run根为[retry01_longres](../../../../runs/sortlr_1003_retry01_longres)，方法目录为[sac_mlp_d1_st_rt_longres_v1](../../../../runs/sortlr_1003_retry01_longres/sac_mlp_d1_st_rt_longres_v1__intersection_sorted_depart4p0)。两臂均为intersection_sorted/depart4.0、fresh seed0、100000 raw、no-resume、CUDA及validation 100回合/seeds 10000–10099。Longres终结值为95001 updates，S/C/T/O=39/55/6/0；shaped environment_step_reward_v2 mean/std=−0.74052338/10.49490709，raw未折扣return mean/std=−0.16/0.95624265。Final checkpoint SHA256 41c5ead07ff053a8f221bbc8d4870da5ac7c6e39ed92b7af476054a4fa103afb与训练终结记录及eval identity一致。

六项shaped分量(success/collision/off-route/timeout/step cost/progress)均值为3.9/−5.5/0/−0.3/−1.1379/2.29737662，100回合覆盖完整，最大对账误差5.33e−15。Train/eval diagnostics errors=0；shadow train 20 unique/300行，eval 300 unique/4500行且100/100回合有样本，active-invalid=0。归档8736文件，source archive SHA与manifest一致，runtime provenance列出的文件SHA匹配归档；suite runtime probe为RTX 5060 Ti/torch 2.12.0+cu132。progress.json稀疏末值99990/94990早于training_complete的100000/95001，保留为快照。更细的证据路径和provenance device字段限制见[experiment history](experiment_history.md)。

ST-RT为前节已核验的63/37/0/0，SHA cce43cacb359ac7688858d514dce0004b64a296ef908b3818de76d01d75274ed；本次check-only显示两前序臂ready、contractfix新根runs/d1_contractfix_20261003无全局receipt、guard ready。未由本次启动训练或评估。该paired结果仍只有一个训练seed，且as-run源码含D1 last-valid实现风险；在contractfix结果出现前不作修复有效性或模块因果结论。协议见[d1 contractfix](d1_contractfix_protocol_20261003.md)。

## 26. 2026-10-03：D1 contractfix 联合修复pair启动快照

ST与ST-RT两方法已在前序身份核验通过后启动，分别位于独立run-root st和st_rt，worker PID为48788/45944、supervisor PID为72064。fresh seed0/CUDA、sorted/depart4、每臂100000 raw，final evaluation仍计划使用validation seeds 10000–10099。源归档与两个30文件traffic pool的SHA审计通过；scaled pool及train/eval overlay根由短worker子root隔离。初始持久化progress各为5082 raw/0 updates，diagnostics与trajectory-history audit errors均0；这是训练初期状态，没有性能结果。实际身份和文件链接见[experiment history](experiment_history.md)与[协议](d1_contractfix_protocol_20261003.md)。

## 27. 2026-10-03 09:32(+08)：contractfix pair post-warmup进度

ST与ST-RT两臂仍training，progress分别为8976 raw/3976 updates、8664/3664；optimization日志确认两臂均已有更新，但尾记录与progress更新时间不同，保留各自文件时点。behavior diagnostics与trajectory audit errors均为0、summary写入正常；独立审计确认两臂前5个共同完成episode的raw/decision计数逐例与train_monitor相符。该进度只证明训练与被动采集正常运行，不构成性能结果。见[d1 contractfix protocol](d1_contractfix_protocol_20261003.md)。

## 28. 2026-10-03：D1 contractfix ST完成final validation，ST-RT评估待核验

ST方法`runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0`按seed0 fresh/no-resume、CUDA训练100000 raw/95001 updates，最终validation为100个唯一seeds 10000–10099。Checkpoint `final_model.zip` SHA256 `AF0D6674ADF321E148DB54D3D48B8349D558E40101C229FDDA9D13EFFE74B162`与training-complete及evaluation identity匹配。episode布尔标签统计为S/C/T/O=45/50/5/0；shaped return mean/std=0.79356204/10.54762652，raw未折扣return mean/std=−0.05/0.97339612。六项shaped reward均值(success/collision/off-route/timeout/step cost/progress)=4.5/−5.0/0/−0.25/−0.8644/2.40796204，coverage=100，最大分量对账误差3.55e−15。评估诊断errors=0；trajectory audit 8644 real decisions、25832 raw steps、errors=0、unknown=0；shadow 357 unique、100/100回合覆盖、active-invalid=0。

ST-RT已完成同预算fresh seed0训练100000 raw/95001 updates，但本次只读核验未见最终evaluation identity/results文件，故不记其最终成绩或两臂差值。结果只对应一个训练seed；ST配置同时修复C8/C9，尚不能隔离各自贡献。完整source/runtime与评估证据见[experiment history](experiment_history.md)和[contractfix协议](d1_contractfix_protocol_20261003.md)。

## 29. 2026-10-03：D1 contractfix paired final100完成

后续检查确认suite status=complete且两个worker均exit_code=0，更新此前ST-RT final待核快照。ST-RT按fresh seed0/no-resume/CUDA训练100000 raw/95001 updates，final validation 100个唯一seeds 10000–10099。Final checkpoint SHA256 `706548B4E84450ADCE1C224C122D88F0C0647B1F212A695A9558C4FE00A55D0F`与training-complete/evaluation identity及实际zip SHA一致。结果S/C/T/O=52/48/0/0；shaped mean/std=2.02868299/10.74453241，raw未折扣mean/std=0.04/0.99919968。六分量均值(success/collision/off-route/timeout/step cost/progress)=5.2/−4.8/0/0/−0.8773/2.50598299，100回合覆盖、最大对账误差3.55e−15。评估diagnostic错误0，8773 decisions/26218 raw；轨迹审计错误0、unknown raw决策0；shadow 298 unique、100/100回合有样本、active-invalid=0。

与ST（SHA256 `AF0D6674ADF321E148DB54D3D48B8349D558E40101C229FDDA9D13EFFE74B162`；S/C/T/O=45/50/5/0）相比，ST-RT在同一validation seed池上的类别比例为52/48/0/0，shaped mean为2.02868299、raw未折扣mean为0.04；ST对应mean为0.79356204及−0.05。这是同一训练seed下的描述性方法对照。训练均为单seed，不能据此声称稳定性；C8与C9共同改变，也无法由此拆分两项修复的因果贡献。完整文件路径见[experiment history](experiment_history.md)和[contractfix协议](d1_contractfix_protocol_20261003.md)。

## 30. 2026-10-03：contractfix配对结果归因和证据审阅完成

核心身份、回报、配对转换矩阵与诊断摘要已核对。综合分析见[结果归因报告](d1_contractfix_results_attribution_20261003.md)，运行诊断和C8/C9实施暴露见[诊断审计](d1_contractfix_diagnostic_audit_20261003.md)，逐seed类别迁移与公平性边界见[配对对照证据](d1_contractfix_comparison_evidence_20261003.md)。ST和ST-RT的7个成功差与终局迁移是同一seed0训练轨迹在共同验证seed池上的描述性差异；ST→ST-RT同时启用route分支、改变有效计算/梯度路径，名义encoder参数量相同不等于有效容量相等。两方法不能证明多seed稳健性或拆分C8/C9效果。bootstrap单γ与标准γ^k目标的差异属协议差异；若之后采用新协议，论文主比较baseline也应同协议重训，旧结果按原协议保留。本次无新训练。automation-3已暂停本轮自动跟进；未追加训练或仿真。
