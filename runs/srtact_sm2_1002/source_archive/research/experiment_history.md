# 历史实验与证据边界

本记录服务于以下研究：以 SAC+MLP 为纯强化学习基线，逐步加入 full 的模块；MST+SLT 为强基线。
数字来自已有文件，不是本任务重新训练或重新评估的结果。记录需要随新实验更新，但历史快照应保留。
研究定位见 [RESEARCH_CONTEXT.md](../RESEARCH_CONTEXT.md)，实现与梯度路径见 [完整实现地图](full_mst_slt_implementation.md)。

## 最新补充：新主目录的连续 100k

用户确认新实验完成后已刷新核对，详见 [连续 100k 配对分析与下一步计划](continuous_100k_mst_vs_st.md)。这一组与下文旧 `__bak_50k/c100000` 续训不同：

| 新连续运行 | 成功 / 碰撞 / 超时 | 原始结果 |
| --- | --- | --- |
| MST+SLT，100k raw steps、seed 0 | 53% / 47% / 0% | [主目录评估](../mst_slt__intersection_sorted_depart4p0/evaluation_results.json) |
| SAC-MLP-D1-ST，100k raw steps、seed 0 | 50% / 50% / 0% | [主目录评估](../sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json) |

已核对字段为 sorted、depart_scale=4.0、最终 checkpoint、100 个共同评估 seed（10000–10099）。评估并行度为 MST 6 workers、ST 1 worker。双方都成功 29 回合，MST 单独成功 24，ST 单独成功 21，双方碰撞 26。两者点估计接近，尚不能宣称等效或跨训练 seed 稳定优劣。30 个交通变体文件在 100 回合中重复使用，不能把回合数当独立场景数。
这份较早快照编写时尚缺匹配协议的从零连续 100k 纯 SAC+MLP 对照；2026-09-29 的正式运行现已补齐，见本文件第 7 节及[本轮结果分析](mlp_strt_100k_analysis.md)。旧续训的 35% 仍不能代替 fresh 100k 结果；下文旧表继续作为历史协议快照保留。
补充静态协议核对：yield_v2 的 depart 缩放 factory 覆盖 `_partitioned_traffic_paths` 并返回全部缩放 route 文件，两种方法均走此路径。MST 的 `frozen_80_20` 参数标签不能证明实际执行 holdout；训练 monitor 未保存逐 episode variant，历史抽样无法独立重建。本组按同交通池内评估解释，不作未见交通场景泛化结论。

## 1. 历史阶段与解释修正

| 阶段 | 记录中的观察 | 当时解释或行动 | 证据与限制 |
| --- | --- | --- | --- |
| 初始 hold35k / MST 失败 | 成功率很低 | 首先排查较高学习率、大 batch、短 warmup 和探索衰减 | [早期分析](analysis_hold35k_mst_low_success.md)；属于历史分析，不能直接沿用为当前根因 |
| fixed 设置 | 文档记录成功率约 1%、timeout 约 83% | 修正先前解释，转向稀疏终止奖励与停车等待行为 | [v2 根因分析](analysis_mst_slt_fix_failure_rootcause_v2.md)；其中“根因”是文档判断，本任务未独立完成因果验证 |
| P1 奖励塑形 | 文档记录 timeout 曾降至约 26%，成功 1/66；后期 timeout 又增加，车辆在路口等待 | 进度/时间成本使策略开始前进，但择机通行仍未解决 | [P1 结果](analysis_p1_reward_shaping_result.md)；不是当前 sorted 场景的正式 benchmark |
| P2 / P4 诊断 | 默认 SUMO 让行、BC 速度、手写 gap/TTC 控制和密度缩放等被分别检查 | 区分动作接口、观测、奖励和交通密度问题 | 对应 `_p2_*`、`_p4_density_verify.py`、`collect_yield_demos.py`、`bc_pretrain_actor.py` 独立脚本；不能视为当前 D1 入口自动执行的模块 |
| yield_v2 正式训练与动作头消融 | 旧日志和 manifest 记录 MST-SLT 50k、100 集成功 59%、碰撞 41%；日志另记 hold35k 100 集全 timeout | 对观测/奖励/动作头开展进一步对照 | [旧实验日志](../experiments_log.md)、[旧根 manifest](../experiment_manifest.json)；需与 sorted 实验分开，且根 manifest 有状态矛盾 |
| 2026-09-28 / 29 的 sorted D1 实验 | 见下列原始 JSON 结果表 | 从 SAC+MLP 增量添加 full 的模块，并诊断路线信息接入方式 | 原始结果文件优先于未更新的研究 Markdown；单训练种子限制仍在 |

旧 README 写“正式训练待启动”，但同日后续日志和产物已有正式训练。根 manifest 同时含 `smoke=true`、MST-SLT 的 100 集结果及失败标记，其他若干方法只是 8 集 smoke。不能直接把根 manifest 整体作为同协议正式比较。

### 1.1 本次刷新核实的 yield_v2 原始文件

以下保留为独立协议组，不与 sorted/depart×4.0 混合。每份评估为 100 回合，评估 JSON 标 `smoke=false`；这不代表其训练也一定是正式训练。

| 方法 / 运行 | 项目记录的训练预算 | 成功 / 碰撞 / 超时 | mean return | 原始结果 |
| --- | --- | --- | ---: | --- |
| MST+SLT yield_v2 | `arguments.json` 请求 50,000 raw steps，seed 0；未见该目录的 `training_complete.json`，不将请求量写成独立观测到的完成量 | 59% / 41% / 0% | 0.180 | [evaluation_results](../mst_slt__intersection_yield_v2/evaluation_results.json) |
| SAC+MLP yield_v2 | `training_complete.json` 记 raw_steps=50,000、updates=45,001；原始 seed 0 | 41% / 59% / 0% | -0.180 | [evaluation_results](../sac_mlp__intersection_yield_v2/evaluation_results.json) |
| SAC+MLP yield_v2 续训 | 完成记录记 base=50k、extra=50k、global=100k、updates=94,502 | 39% / 61% / 0% | -0.220 | [evaluation_results](../sac_mlp__intersection_yield_v2/continue_100000/evaluation_results.json) |
| 历史 hold35k yield_v2 目录 | **当前 arguments / completion 只记录 300-step smoke，completion 的 smoke=true** | 0% / 0% / 100% | 0.000 | [evaluation_results](../hold35k__intersection_yield_v2/evaluation_results.json) |

MST/SAC 三行评估 seed 起点为 10000；hold35k 行为 420000。检查点指向各自目录的 `final_model.zip`。
**hold35k 的训练与评估 smoke 标记不一致，目录名也不能证明训练了 35k 或 50k。其 100% timeout 不能当作当前 D1 full 的正式失败结果。**
旧日志关于较长训练的叙述与当前目录 metadata 需要分别保留；本次静态调查不能确认是否发生过覆盖或复用目录。

### 1.2 D1 full 的完成状态

- `sac_mlp_d1_full__intersection_yield_v2`：本次见训练模型及训练完成记录，未见正式 `evaluation_results.json` / final evaluation；没有可纳入本表的成功率。
- `hsac_mlp_base_d1_full__intersection_yield_v2`：本次见 progress/status/monitor/checkpoint，未见训练完成记录；不能宣称训练或最终评估已经完成。
- 二者均不以历史 hold35k 或 MST+SLT 的结果代填。

本轮重新发现 26 份 `evaluation_results.json`；前一轮调查曾报告 28 份。文件集合的统计是读取快照，本记录不依据数量变化推断运行完成、删除或结果优劣。

## 2. sorted 场景 50k 快照

协议：`intersection_sorted`，`depart_scale=4.0`，原始训练 seed 0，预算 50,000 raw SUMO steps；每次动作重复 3 个仿真步。
评估为 100 回合、`final_model`。父方法代码使用 seeds 10000–10099；部分 50k D1 JSON 的 `episode_records` 为空，因此这些运行的种子范围依据代码配置，不能称为逐回合记录均已保留。

| 方法 | 成功率 | 碰撞率 | 超时率 | mean return | 来源编号 |
| --- | ---: | ---: | ---: | ---: | --- |
| MST+SLT | 51% | 49% | 0% | 0.020 | R01 |
| SAC+MLP | 37% | 63% | 0% | -0.260 | R02 |
| D1-ST | 48% | 52% | 0% | -0.040 | R03 |
| D1-ST-RT | 31% | 67% | 2% | -0.360 | R04 |
| D1-ST-Attn | 43% | 56% | 1% | -0.130 | R05 |
| RT-Late | 46% | 54% | 0% | -0.080 | R06 |
| RT-Gate | 40% | 60% | 0% | -0.200 | R07 |
| RT-Ego | 44% | 56% | 0% | -0.120 | R08 |
| RT-Edge | 39% | 61% | 0% | -0.220 | R09 |

以上 return 是评估 JSON 中的记录，不能与训练时奖励塑形后的回报直接混比。

## 3. 累计 100k 的续训快照

协议是 **base_raw_steps=50,000 + extra_raw_steps=50,000 = global_raw_steps=100,000**。
`c100000` 不表示再训练 100,000 步。各结果的 `identity.checkpoint` 指向该续训目录的 `final_model.zip`，不是 `best_model`。
该批结果保留 100 个评估回合，seed 为 10000–10099。

| 方法 | 成功率 | 碰撞率 | 超时率 | mean return | 平均决策步 / raw SUMO 步 | 来源编号 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MST+SLT | 25% | 75% | 0% | -0.500 | 84.6 / 252.6 | R01-C |
| SAC+MLP | 35% | 65% | 0% | -0.300 | 63.6 / 189.8 | R02-C |
| D1-ST | 33% | 67% | 0% | -0.340 | 79.1 / 236.3 | R03-C |
| RT-Late | 29% | 71% | 0% | -0.420 | 120.8 / 361.5 | R06-C |
| RT-Gate | 24% | 76% | 0% | -0.520 | 82.8 / 247.3 | R07-C |
| RT-Ego | 36% | 62% | 2% | -0.260 | 94.5 / 282.4 | R08-C |
| RT-Edge | 25% | 68% | 7% | -0.430 | 80.8 / 241.4 | R09-C |

本快照未发现 D1-ST-RT 和 D1-ST-Attn 对应的 c100000 结果；这不等于这些方法绝不曾运行。
各 100k 行 off-route 记录均为 0%。

### 续训条件必须随结果保留

- 续训加载 50k 模型，但没有恢复 replay buffer。
- 续训 warmup 设为 500 raw steps，原始训练 warmup 为 5,000 raw steps。
- 原始运行参数保存训练 seed 0；续训 metadata 本身没有单独保存 seed 字段，不应据此声称续训的全部随机数状态已被恢复。
- 因此 50k 与 100k 差异反映的是这套具体续训流程，不能直接等同于从零连续训练 100k 的效果。

## 4. 原始结果索引

来源均相对于本文件所在的 `analysis/`。续训各目录的 `training_complete.json`、参数文件和 `evaluation_results.json` 共同提供预算与检查点证据。

| 编号 | 50k 原始结果 | 100k 原始结果 |
| --- | --- | --- |
| R01 | [MST+SLT](../mst_slt__intersection_sorted_depart4p0__bak_50k/evaluation_results.json) | [R01-C](../mst_slt__intersection_sorted_depart4p0__bak_50k/c100000/evaluation_results.json) |
| R02 | [SAC+MLP](../sac_mlp_depart4p0__intersection_sorted/evaluation_results.json) | [R02-C](../sac_mlp_depart4p0__intersection_sorted/c100000/evaluation_results.json) |
| R03 | [D1-ST](../sac_mlp_d1_st__intersection_sorted_depart4p0__bak_50k/evaluation_results.json) | [R03-C](../sac_mlp_d1_st__intersection_sorted_depart4p0__bak_50k/c100000/evaluation_results.json) |
| R04 | [D1-ST-RT](../sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json) | 本快照未发现 |
| R05 | [D1-ST-Attn](../sac_mlp_d1_st_attn__intersection_sorted_depart4p0/evaluation_results.json) | 本快照未发现 |
| R06 | [RT-Late](../sac_mlp_d1_st_rt_late__intersection_sorted_depart4p0/evaluation_results.json) | [R06-C](../sac_mlp_d1_st_rt_late__intersection_sorted_depart4p0/c100000/evaluation_results.json) |
| R07 | [RT-Gate](../sac_mlp_d1_st_rt_gate__intersection_sorted_depart4p0/evaluation_results.json) | [R07-C](../sac_mlp_d1_st_rt_gate__intersection_sorted_depart4p0/c100000/evaluation_results.json) |
| R08 | [RT-Ego](../sac_mlp_d1_st_rt_ego__intersection_sorted_depart4p0/evaluation_results.json) | [R08-C](../sac_mlp_d1_st_rt_ego__intersection_sorted_depart4p0/c100000/evaluation_results.json) |
| R09 | [RT-Edge](../sac_mlp_d1_st_rt_edge__intersection_sorted_depart4p0/evaluation_results.json) | [R09-C](../sac_mlp_d1_st_rt_edge__intersection_sorted_depart4p0/c100000/evaluation_results.json) |

## 5. 当前可以与不可以支持的判断

**实验观察：**在此单训练种子快照中，直接 RT 的 50k 成功率低于 ST；路线接入变体结果不同；已列出的 100k 续训成功率均低于其 50k 结果。

**尚未建立的因果结论：**不能据此断言路线信息本身有害、模型过拟合、Graph-SLT无效或 full 不如强基线。网络结构、辅助目标、优化器路径、续训条件以及训练随机性都需要分别核对。

**比较边界：**当前 sorted SAC、D1 与 MST+SLT 主对照对齐了场景、发车缩放、基础动作契约、观测修正、训练奖励和基本预算。MST+SLT 还使用表示辅助目标；普通 D1-ST/RT 等关闭表示辅助目标。它们不是只改变同一个编码层的单变量对照。参数量、计算量、多训练种子方差尚未在本记录中形成完整控制证据。

## 6. 更新记录时的最小字段

记录方法名及角色、代码/配置来源、场景、depart_scale、动作头、训练奖励、训练seed、raw与decision步数、是否恢复buffer/优化器/随机状态、评估seed/回合数、final/best/checkpoint选择、成功/碰撞/超时/偏航/回报及来源文件。新增结果应按协议单独追加，再讨论是否可合并比较。

## 7. 2026-09-29：fresh 连续 100k SAC+MLP 与 ST-RT

本节追加本轮新结果，不替换前述历史表。运行根目录为 [`runs/d0929_100k_diag`](../../../../runs/d0929_100k_diag)，其 [`launcher_status.json`](../../../../runs/d0929_100k_diag/launcher_status.json) 记录两个 worker 均正常完成、退出码为 0；启动时间 2026-09-29 17:47:03 +08，完成时间 21:40:56 +08，总历时 14,032.72 秒。运行入口为 `train_intersection_yield_v2_d1.py`。两方法均从新建模型开始，没有续训；训练 seed=0、100,000 raw steps、95,001 updates、warmup=5,000 raw steps，最终 checkpoint 各评估 100 回合，eval seed=10000–10099。

两项新方法使用 `intersection_sorted`、depart scale=4.0、action repeat=3、buffer=20,000、batch=32、学习率 1e-4、gamma=0.99，以及相同环境与奖励配置。结果依据各方法的 [`arguments.json`](../../../../runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/arguments.json)、[`training_complete.json`](../../../../runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/training_complete.json)、[`evaluation_results.json`](../../../../runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/evaluation_results.json) 及对应 ST-RT 文件；评估 identity 的 checkpoint SHA-256 与 `final_model.zip` 和训练完成记录一致。最终评估使用 final checkpoint，不使用 best checkpoint。

| 方法 | 训练状态 / 预算 | 成功 | 碰撞 | 超时 | 偏航 | 平均评估回报 | 平均 raw steps / 回合 | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SAC+MLP，本轮 fresh | 完成；100k raw / 95,001 updates；seed 0 | 22/100 | 38/100 | 40/100 | 0/100 | -0.16 | 367.95 | [本轮评估](../../../../runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/evaluation_results.json) |
| D1-ST，既有 fresh 100k | 既有主目录结果；seed 0 | 50/100 | 50/100 | 0/100 | 0/100 | 0.00 | 242.87 | [原始评估](../sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json) |
| MST+SLT，既有 fresh 100k | 既有主目录结果；seed 0 | 53/100 | 47/100 | 0/100 | 0/100 | 0.06 | 241.48 | [原始评估](../mst_slt__intersection_sorted_depart4p0/evaluation_results.json) |
| ST-RT，本轮 fresh | 完成；100k raw / 95,001 updates；seed 0 | 63/100 | 37/100 | 0/100 | 0/100 | 0.26 | 268.99 | [本轮评估](../../../../runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json) |

SAC+MLP 的从零连续 100k 结果现已补齐；旧 `__bak_50k/c100000` 仍属于续训，不能作为它的匹配替代。四种方法的评估记录都可按 `(seed, traffic_variant)` 100/100 配对。评估只覆盖同一完整交通池中的 30 个 route variant：20 个各重复 3 回合，10 个各重复 4 回合；训练与评估未留出 route 模板。因此这些结果是固定 checkpoint 的配对表现，不是对独立交通场景总体的泛化结论，也不估计多训练 seed 稳定性。

以下转移矩阵统一以对照方法的回合结果作行、以 ST-RT 的同 seed / traffic 结果作列；每格按“成功、碰撞、超时”顺序。所有矩阵行和列边际均与各自原始评估的 100 回合 outcome 计数一致。

| 对照方法 | 对照 outcome | ST-RT 成功 | ST-RT 碰撞 | ST-RT 超时 |
| --- | --- | ---: | ---: | ---: |
| SAC+MLP | 成功 | 15 | 7 | 0 |
| SAC+MLP | 碰撞 | 23 | 15 | 0 |
| SAC+MLP | 超时 | 25 | 15 | 0 |
| D1-ST | 成功 | 35 | 15 | 0 |
| D1-ST | 碰撞 | 28 | 22 | 0 |
| D1-ST | 超时 | 0 | 0 | 0 |
| MST+SLT | 成功 | 39 | 14 | 0 |
| MST+SLT | 碰撞 | 24 | 23 | 0 |
| MST+SLT | 超时 | 0 | 0 | 0 |

相对于 SAC+MLP，ST-RT 在这 100 个固定配对回合中成功数净增 41；40 个原超时回合均变成成功或碰撞（25 成功、15 碰撞），碰撞率仅净降 1 个百分点。相对于 D1-ST，ST-RT 独有成功 28 回合、D1-ST 独有成功 15 回合；相对于 MST+SLT，分别为 24 与 14 回合。后两个比较的点估计分别是 +13 与 +10 个百分点，不能据单一训练 seed 宣称稳定优越。

对成功率差的 30 个 traffic-variant cluster bootstrap 仅作固定 checkpoint 的场景抽样敏感性检查：50,000 次有放回簇抽样，随机数为 NumPy `default_rng`（PCG64）、seed=20260929；每次抽 30 个 variant ID，并纳入抽中 variant 的全部实际 3/4 个回合，差值方向为 ST-RT 减对照方法，采用 95% percentile CI。结果为：相对 SAC+MLP +41 pp [30, 52]；相对 D1-ST +13 pp [2.06, 23.23]；相对 MST+SLT +10 pp [-1.05, 21.43]。该区间反映这组固定 checkpoint 在观测 route 簇上的抽样敏感性，不是 training-seed CI，也不覆盖未见交通模板。逐回合 Wilson / McNemar 结果及全部 source hashes 在[可复核离线结果 JSON](analysis_100k_results.json) 中；分析脚本为 [analyze_100k_results.py](analyze_100k_results.py)。

代码归因边界：ST-RT 配置为 `use_route=true`、`use_topology=false`、`use_slots=false`、`representation_coef=0`、`slot_balance_coef=0`，没有启用 Graph-SLT/SBS。SAC+MLP→ST-RT 同时更换了特征抽取器及路线处理路径，不能当成单一 RT 开关实验；D1-ST→ST-RT 才是当前 intended route-toggle 配对，但仍只有一个训练 seed。优化文件实际名为 `diagnostics/train/optimization.jsonl`（非 gzip）；训练/评估端行为数据及 outcome 分层另见[行为分析 JSON](analysis_100k_behavior.json)和[只读脚本](analysis_100k_behavior.py)。

SAC timeout 的静态网络核验补充：该场景实际网图是 [`intersection_sorted/map.net.xml`](../../envs/sumo/original_scenarios_v1/intersection_sorted/map.net.xml)，ego 固定路线来自 [`intersection_sorted/ego.rou.xml`](../../envs/sumo/original_scenarios_v1/intersection_sorted/ego.rou.xml)，不是通用 `networks/intersection/intersection.net.xml`。ego route 为 `scene_rep_ego_route: -E1 -> -E0`；静态网图给出的唯一 `-E1 -> -E0` 连接在第 169 行，要求 fromLane=2、toLane=1。SAC 的 40 个 timeout episode snapshot 全部记录 route `[-E1,-E0]`、route_index=0、在 `-E1` 末端 lane position 70 m 停止；38 个在 `-E1_0`，2 个在 `-E1_1`，均未停在具有该转向连接的 lane2。超时组 episode-equal 平均目标速度为 8.058 m/s、实际速度为 1.167 m/s。此证据支持“在与目标路线连接不兼容的车道上停滞、请求速度没有实现”这一观察；它本身不能确定策略为何选择这些车道，亦不足以归因成环境控制 bug。详细静态字段及 episode source path 见结果 JSON 的 `pairing.SAC_timeout_route_liveness`。

当前研究优先级与运行快照见 [RESEARCH_CONTEXT.md](../RESEARCH_CONTEXT.md) 及本文件第 8 节：单训练 seed 0 下运行 ST-RT-Topo fresh 100k 和相邻 Topo+3slot；MST+SLT 是完整方法的最终对照目标，不要求每个增量单独超过。这里保留的历史数字不作改写，也不表示 full 已验证。

## 8. 2026-09-30：ST-RT-Topo 与 Topo+3slot fresh 100k 运行快照

本节是**运行中快照**，不是训练完成或性能结果。实际 run root 为 [`runs/t0930_0023_topo3_100k`](../../../../runs/t0930_0023_topo3_100k)（此前提及的 `t0930_0021_topo3_100k` 目录不存在）。启动命令和进程关系见 [launcher supervisor 记录](../../../../runs/t0930_0023_topo3_100k/launcher_supervisor.json)；启动器状态见 [`launcher_status.json`](../../../../runs/t0930_0023_topo3_100k/launcher_status.json)，参数快照见 [`experiment_manifest.prelaunch.json`](../../../../runs/t0930_0023_topo3_100k/experiment_manifest.prelaunch.json)，启动时源码哈希见 [`source_snapshot.json`](../../../../runs/t0930_0023_topo3_100k/source_snapshot.json)。

实际配置为 fresh model、未 resume；`intersection_sorted`、depart_scale=4.0、训练 seed 0、每方法 100,000 raw steps、warmup 5,000 raw steps、action repeat=3、CUDA device 0、两并行 worker、每方法计划 final evaluation 100 episode。训练与评估使用同一完整有效 traffic pool；该协议没有 traffic holdout。两个方法分别为 `sac_mlp_d1_st_rt_topo` 与 `sac_mlp_d1_st_rt_topo_3slot`。后者打开 incremental slots（32/64/32），Graph-SLT/SBS 关闭且两个辅助系数为 0。具体不可只看方法名，配置及已核对源码 hash 见 prelaunch manifest。

2026-09-30 00:32:12 +08 文件快照：supervisor PID 56072；Topo worker PID 17496，Topo+3slot worker PID 9240；两者状态均为 `training`。各自 progress 为 raw_steps=1,195、updates=0（低于 5,000 raw warmup），状态更新时间约 00:32:12 +08。worker 的 [`arguments.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo__intersection_sorted_depart4p0/arguments.json) 与 3slot 对应文件确认 `smoke=false`、budget=100,000、seed=0 和 CUDA；对应 [`status.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo__intersection_sorted_depart4p0/status.json) 与 [`progress.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo__intersection_sorted_depart4p0/progress.json) 提供此刻 worker 状态和步数。3slot worker 文件位于同一 root 下 `sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0/`。

该快照不表示 100k 训练、final checkpoint 或 100-episode evaluation 已完成；目前不作成功率或模块效果判断。后续状态更新应保留时间戳，并以新的 launcher/status/progress、`training_complete.json` 和 final evaluation 文件为依据。训练诊断 metadata 的 `source=train_replay` 是阶段配置标签，不保证共享 online encoder 的每次 activation 都来自 replay；须结合事件类型、sample index/age、batch size 和 `grad_enabled` 解释，且不把训练 activation 关联到当前 rollout episode。3slot worker 的 [`arguments.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0/arguments.json)、[`status.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0/status.json) 和 [`progress.json`](../../../../runs/t0930_0023_topo3_100k/sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0/progress.json) 保存对应 worker 快照。file_audit 随后的 warmup 进度复核为两路 raw_steps=2,392、updates=0；该快照的独立时间戳未记录于此，不覆盖上一条明确带时点的快照。

## 9. 2026-09-30：Topo/Topo+3slot 原始运行 I/O 中断

**重启验收更新：**新 root `runs/t0930_topo3_retry01` 的 Topo 已观察到 6,578 raw / 1,578 updates，3slot 为 6,577 raw / 1,577 updates；两 CUDA worker 均继续 training，诊断错误数为 0、汇总无 pending，日志无新 traceback。已跨过 5k 预热和原故障点。本条仍不是完整 100k 结果，后续以新 run 的实时文件为准。

原 root [`t0930_0023_topo3_100k`](../../../../runs/t0930_0023_topo3_100k) 的 Topo+3slot worker 在状态文件所记约 5,651 raw steps 时，因诊断汇总发布时 `summary.json.tmp.replace(summary.json)` 返回 Windows WinError 5 而退出；不是模型或优化器错误。此前最后周期进度快照为 5,382 raw steps / 382 updates，不能将 382 当作失败瞬间的精确更新数。诊断 JSONL 保留 21 个 episode；清理后 summary 可读为 21 episodes / 5,653 raw records。目标 ACL 允许修改且路径长 173 字符，具体拒绝替换的句柄来源未知。

Topology-only worker 检查时仍正常推进（曾观察到 raw 14,374 / updates 9,374），并未发生同样的模型失败；因两路共用受影响的日志代码，为应用统一修复，在核验 PID 与启动时间后主动停止旧 supervisor 树。旧目录、日志和 checkpoint 均保留，并写入 `restart_note.txt`，区分三槽的运行失败与 Topo 的主动中断。旧运行不算完整 100k 性能证据。

修复覆盖 summary 与 progress 的 PermissionError 有界重试、非关键汇总延后发布及恢复，关键状态/配置和其他 I/O 错误仍显式失败。29 项相关测试通过。新 root `runs/t0930_topo3_retry01` 于 2026-09-30 01:24:05 +08:00 启动，supervisor/Topo/3slot PID 分别为 50304/8364/62860；原协议不变：两 CUDA worker、seed 0、fresh 连续 100k raw steps、5k warmup、10k raw checkpoint、100 回合最终评估，Graph-SLT/SBS 仍关闭。两路启动后已由 raw 2,691/updates 0 推进到 raw 5,082/updates 82，再推进到 raw 5,680/updates 680，状态均 training 且无新异常栈。此为操作与训练进度证据，不作模块性能归因；后续最新核验见[summary I/O 恢复记录](topo_3slot_io_recovery.md)。

## 10. 2026-09-30：随机交叉口 Bernoulli 流量协议验证

新增场景 `intersection_random_{low,medium,high}_v1` 沿用 `intersection_sorted` 的 map/ego 任务，固定三条 background route 和 lane 0，仅改变计划流率；总量为 590/1,180/1,770 veh/h，方向比例为 200:150:240。`<flow probability=q/3600>` 的 XML probability 按秒定义；SUMO 1.25.0 对 0.1s simulation step 按步长缩放，故每步概率为 `q/36000`。背景车到达 route 终点后离网，不由 PaperEnv 重插。train、validation、test 使用不重叠 SUMO seed 域；默认开发集 logical seed 为 10000–10099。细节及版本资产 hash 见[协议](random_intersection_v1_protocol.md)、[生成器](../../envs/sumo/random_intersection.py)及 low/medium/high 的 `scenario_manifest.json`（位于 `../../envs/sumo/original_scenarios_v1/intersection_random_{low,medium,high}_v1/`）。

固定 validation logical seed 10000、SUMO 1.25.0、0.1s step、仿真到 180s 的 headless smoke：low/medium/high 分别有 19/55/69 个计划 background request，全部实际发车；pending/missing 与结束仍在网车辆均为 0。发生正 departDelay 的车辆数为 0/7/12，最大延迟为 0/1.1/2.4s。Medium seed 10000 的独立重复得到完全相同 planned schedule hash；seed 10001 得到不同计划表。相同 seed 下跨密度 arrival times 三路均按 low⊂medium⊂high 嵌套，但同一计划时刻的 vehicle type 抽样并非全相同。SUMO 命令 exit 0，每 case 记录到 `Ignoring junction logic for junction 'J1'` 警告；没有测试策略控制变化对计划流的反事实影响。完整 schedule、实际 depart/delay、warning 和 asset hashes 见 [`runs/random_intersection_v1_validation/bernoulli_smoke_summary.json`](../../runs/random_intersection_v1_validation/bernoulli_smoke_summary.json)，可复跑脚本在同目录 `validate_bernoulli_smoke.py`。

4 项随机场景协议测试通过；主代理核实三档 PaperEnv reset/5-step、同/异 seed 与 split 隔离通过。既有 `test_paper_reproduction.py` 使用独立临时目录单独运行 48 项通过。此次没有训练或新的 100k 结果；需求档未以策略表现标定为低/中/高学习难度。`intersection_sorted` 与本 Bernoulli/随机车型 seed 协议的成绩不能作为单变量模型对照。

## 11. 2026-09-30：PaperEnv 与 D1 随机流入口检查

workspace 根 [`paper_env_integration.json`](../../../../runs/random_intersection_v1_validation/paper_env_integration.json) 的单 seed=10000 PaperEnv smoke：50s ego 发车预热后执行 5 个决策步，low/medium/high 实际驶入背景车数 9/28/35、峰值在网背景车 5/15/17、pending 均为 0，平均插入延迟 0/0.0964/1.3657s、最大延迟 0/1.4/9.6s。此为短程环境压力检查，不是学习难度标定或策略结果；与项目根下的 headless SUMO `bernoulli_smoke_summary.json` 是不同产物。

D1 `sac_mlp_d1_st_rt` medium factory reset(seed=217)+1 sampled-action step 观察到 train split/SUMO seed=217，test split/SUMO seed=1500000217；两者请求流均为 400/300/480 veh/h，protocol=`intersection_bernoulli_v1`，vehicle/pedestrian scale=1，endless=false，reinsertions=0，step 未终止。train/test overlay manifests 均记录 3 个 base flows、无额外 flow/vehicle/person，原资产未改且 map/ego route 未变；显式进程退出码未捕获，因此本记录只主张观察到的 reset/step 输出与 manifest，不称完整端到端成功。

随机场景 eval-only guard 临时检查通过：同一 run 目录已有 validation 结果时请求 test 会抛 `FileExistsError`；同 split 允许，旧场景不受该 guard 影响。D1 CLI 可组合原训练 `--model-path <checkpoint>` 与全新 `--output-dir <test-dir>` 运行 `--eval-traffic-split test`，将最终 test 结果与 validation 分开。入口代码及完整配置见[随机交叉口协议](random_intersection_v1_protocol.md)；本轮没有新的模型训练或策略评估。

## 12. 2026-09-30：p05 中档流量变体与 warmup diagnostics（启动前记录）

**配置与实现：**旧 `intersection_random_medium_v1` 保持 400/300/480 veh/h，总请求 1,180；新 `intersection_random_medium_p05_v1` 对三条背景进口 lane 0 在每个 0.1 s tick 各按 `p=0.5` 生成一条请求，每路期望 18,000、合计 54,000 计划 veh/h。p05 按 actual SUMO seed + route ID 派生 SHA256 子种子，使用 Python `Random.random()` 对 `[0,130)` 预采样显式逐车到达；每 route/tick 至多一个请求，SUMO 对受阻插入保留 pending 队列。该计划需求不代表实际通行量，亦未标定为学习难度。p05 map/ego、120 个 vType、三 route 和三个 driver distributions 与旧 medium 完全一致；ego route 解析现支持顶层 route 引用并仍严格检查 `-E1 -E0`，冻结源资产 hash 没有变化。详情见[随机交叉口协议](random_intersection_v1_protocol.md)。

**诊断采集与启动前核验：**四个随机场景 low/medium/high/p05 的正常 reset 都在第 300/400/500 个 raw tick（30/40/50 s）读取 warmup traffic state，不额外推进仿真；reset info 中的 `warmup_traffic_checkpoints` 进入现有 train/eval diagnostics 的 episode `reset_info`。schema 区分在网背景车与速度 `<0.1 m/s` 的在网停驶数、入口 lane occupancy/halting 数、到时请求、累计插入及路网外 pending；native-flow 与预采样 schedule 的请求计数口径由 `request_count_source` 标明。

父代理提供的 factory smoke [`factory_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/p05fac_20260930_luna01/factory_smoke.json>) 为 `status=passed`。四场景 train reset 与 p05 validation reset 均有三份 checkpoint，并写入五份实际 `episodes.jsonl` reset_info；medium 与 p05 train 各多执行一次 sampled-action step。p05 train SUMO seed 0 计划 1,952 个请求（642/663/647 per route），validation SUMO seed 1,000,000 计划 1,929（634/656/639），两者 route×depart 重复数均为 0。50s snapshot：train due/inserted/pending=753/83/670，validation=770/79/691；当前在网背景车为 30/33、停驶车为 0/0。待插入车辆属于路网外队列。新增 p05 单测与旧协议单测合计 8 项通过。以上均为资产/环境工厂和记录路径 smoke，不包含 learning update 或策略结果。

**计划与状态更新：**原 medium 与 medium-p05 的 SAC+MLP fresh 配对条件为 seed 0、CUDA、100,000 raw steps、learning start 5,000 raw、每 10,000 raw checkpoint、final evaluation 100 episodes，其他超参沿用既有 SAC+MLP baseline；只比较两个中档，不将其称为相同到达协议或已标定难度。实际启动、初始 p05 cache failure、修复、重启与进度见本文件第 13 节。所列 progress 尚未达到 100k，无 final evaluation 或性能结论。

## 13. 2026-09-30：SAC+MLP medium/p05 配对启动、缓存修复与运行快照

### 初始启动与 p05 overlay cache failure

初始配对 root [`smlp_p05_a0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_a0930>) 的 supervisor PID 52540；medium worker PID 15900 首次读取为 raw=1,497、updates=0，file_audit 确认其继续训练且保留在 rootA。初始 p05 worker PID 19128 在 raw=144 时 exit 1，尚未接近 5,000 raw `learning_starts`；该失败是 cache/source identity I/O 问题，不是模型或方法结果。失败日志保留在 `runs/smlp_p05_a0930/launcher_logs/medium_p05.stdout.log`，该 p05 attempt 不算有效 100k 训练或性能结果。

第二个 episode 生成动态 schedule 时 source 临时文件 stem 恒为 `episode`，overlay cache 也据固定 stem 复用路径；新 source 内容的 `source_sha256` 与原 manifest 不同，因而出现 hash drift 拒绝。修复仅为动态 schedule overlay cache 名追加 schedule 内容 SHA256 前 16 位；manifest 继续校验完整 hash，静态 medium overlay 逻辑不变。

cache 修复后的实际 factory smoke [`cache_reuse_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_cache_smoke_0930/cache_reuse_smoke.json>) exit 0：同一 env 的 p05 seed 0→1 复用相同动态 source 路径，但分别产生 `__s74e00ee8bca038e0` 与 `__s7b180967f4946b83` overlay identity；不同 temporary directory 的新 env 用相同 seed 0 时 schedule 内容 hash 相同，manifest 校验通过并复用相同 overlay。三个真实 factory reset 都保留三处 warmup 快照，scale=1、额外车辆=0、endless reinsertion=false。

### p05 fresh restart 与当前 paired snapshot

p05 单独在新 root [`smlp_p05_b0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930>) fresh-only 重启，worker PID 90004 于 2026-09-30 11:27:15 +08 启动。其 arguments 核实 `method=sac_mlp`、`scenario=intersection_random_medium_p05_v1`、budget=100,000、seed=0、device CUDA、`smoke=false`、`resume=false`。较早连续 progress 快照为 11:28:26 raw=1,193 / updates=0 与 11:29:26 raw=1,790 / updates=0；当时均仍低于 5,000 raw warmup。2026-09-30 11:31:01 +08 paired progress：medium worker PID 15900 位于 rootA 的 `sac_mlp__intersection_random_medium_v1_depart1p0`，raw=23,619 / updates=18,619 / 79 完整训练 episodes；p05 worker PID 90004 位于 rootB 的 `sac_mlp__intersection_random_medium_p05_v1_depart1p0`，raw=3,886 / updates=0 / 13 完整训练 episodes，状态 training。p05 worker 的更新数为零符合 5k learning warmup；不是失败。活动映射见 [`active_pair_manifest.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930/active_pair_manifest.json>)。两个 worker 都记录 30/40/50 s checkpoint；warmup 事实与 pending 分母说明见[协议](random_intersection_v1_protocol.md)。

rootA 中原 medium PID 15900 持续运行；未停止、未从 checkpoint 重启。rootA 的 p05 PID 19128 失败数据仍保留，rootB 的 p05 是新 fresh attempt。上述是训练进行中的运行快照，不是 100k 完成或 final evaluation；本次还没有方法性能或交通难度结论。

## 14. 2026-09-30：随机中档交通的 SAC+MLP fresh 100k 完成结果

本节更新第 13 节的启动期状态；此前状态行仍作为历史快照保留。有效 medium run 在 [`smlp_p05_a0930`](<../../../../runs/smlp_p05_a0930>)，有效 p05 fresh restart 在 [`smlp_p05_b0930`](<../../../../runs/smlp_p05_b0930>)。rootA 中 raw=144 退出的 p05 attempt 不是有效训练，已排除；不能把它与 rootB 的有效 p05 restart 混为一谈。旧 `intersection_sorted` 对照来自 [`d0929_100k_diag`](<../../../../runs/d0929_100k_diag>)。每项的 `training_complete.json` 记录 raw_steps=100000、updates=95001、replay_size=20000、smoke=false；final 模型及评估 checkpoint identity 与完成记录一致。三个运行均是 `sac_mlp`、训练 seed=0、fresh/no resume、CUDA、100k raw、5k raw learning warmup、10k raw checkpoint 间隔、action_repeat=3、batch=32、lr=1e-4、buffer=20k、gamma=.99、vehicle/pedestrian scale=1，奖励配置相同。最终 checkpoint 均评估 100 回合、单 worker、deterministic policy。随机两场景 eval split 为 validation，logical seeds 为 10000–10099，实际 SUMO seeds 为 1000010000–1000010099；sorted 使用同 logical seed 编号但回放 30 个固定 traffic variant（各重复 3 或 4 次），不是同一套随机 episode。

| 场景 / 运行 | 成功 | 碰撞 | 超时 | 越界/偏航 | `mean_return` | 平均 decision / raw steps | 成功回合平均通行时长 | final 评估原始来源 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原中档 `intersection_random_medium_v1` / rootA | 75/100 | 25/100 | 0/100 | 0/100 | 0.50 | 54.06 / 161.16 | 18.3707 s | [evaluation_results.json](../../../../runs/smlp_p05_a0930/sac_mlp__intersection_random_medium_v1_depart1p0/evaluation_results.json) |
| 中档压力变体 `intersection_random_medium_p05_v1` / rootB | 12/100 | 88/100 | 0/100 | 0/100 | -0.76 | 69.31 / 206.88 | 38.9 s | [evaluation_results.json](../../../../runs/smlp_p05_b0930/sac_mlp__intersection_random_medium_p05_v1_depart1p0/evaluation_results.json) |
| 旧 `intersection_sorted`, depart_scale=4 / sorted root | 22/100 | 38/100 | 40/100 | 0/100 | -0.16 | 122.83 / 367.95 | 27.0636 s | [evaluation_results.json](../../../../runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/evaluation_results.json) |

`mean_return` 是评估的终局 outcome 回报，不是训练用塑形回报：evaluation 累加 `info.undiscounted_reward`，SUMO 环境对 success/collision 给 +1/-1，timeout/off-route 给 0，因而表中均值与终局类别计数一致。所有行都来自 final checkpoint；不能用训练期间 best rolling window 取代 final。训练中 20-episode rolling success 的 best 分别为 medium 95%（raw clock 91,725）、p05 25%（48,411）、sorted 65%（57,705）；late completed-episode 窗口的 last20/last50 分别为 medium 75%/76%、p05 5%/8%、sorted 30%/34%。medium final 75% 与其最后训练窗口一致，现有曲线不支持将它描述为偶然单点；但每种方法只有一个训练 seed，不能据此推断收敛稳定性。best 仍是训练期窗口统计，没有另做 best checkpoint 的 100 回合独立评估。

正常评估 reset 的 warmup 只读快照（N=100）进一步显示交通暴露有明显差别。medium 在 30/40/50 s 的平均在网背景车数为 5.30/5.57/5.65，累计已入场 9.82/13.22/16.70，路网外 pending 为 0.01/0.01/0.03；p05 对应在网 29.10/29.31/29.19、已入场 50.74/65.77/80.75、pending 399.01/534.35/669.41。两个随机场景的这些时点 halt 数均为 0；pending 是路网外等待插入，不是路内排队。sorted 没有同口径 30/40/50 s warmup 记录；其 reset active_vehicles 均值 14.93、source_endless=true 为 100/100，reset/end 累计 reinsertion 均值 15.38/49.79，不能与随机场景的 warmup 在网数直接等同。

基于现有证据，后续工作中档固定原 `intersection_random_medium_v1`：三路请求率 400/300/480 veh/h、总量 1,180 veh/h，对应每 0.1 s 的逐车道发车概率 1/90、1/120、1/75。p05 每路每步 p=0.5、总请求 54,000 veh/h，作为压力变体，而非重新调 SAC 的目标中档。medium 更易通行与其实际交通暴露更低、且随机场景不循环重插离网车辆相符；这只是跨协议观察下的合理解释，不是隔离交通因素后的因果估计。sorted 同时改变请求生成、流量强度/缩放、模板复用和 endless reinsertion，故不能把结果差归因于模型或某一个 traffic 因素。仍只有一个训练 seed；尚无同一随机交通协议下 MST+SLT 的 final 对照，因此不能推断强基线 ceiling。warmup 逐场景结果及结构化来源索引见[交通校准报告](scenario_medium_calibration_20260930.md)和[JSON](scenario_medium_calibration_20260930.json)。

## 15. 2026-09-30：p03 / p02 新变体的 SAC+MLP 100k 启动记录

用户明确指定每条背景进口车道每 0.1 秒的请求概率 p=0.3、p=0.2，不是 p=0.03、p=0.02。新增场景分别为 `intersection_random_medium_p03_v1` 和 `intersection_random_medium_p02_v1`，每条进口请求率分别 10,800 / 7,200 辆/小时，三路合计 32,400 / 21,600 辆/小时；请求量不是实际入场量，也不构成已校准的中等难度。

正式运行根目录：`D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p03p02_0930`。两个结果子目录分别为 `sac_mlp__intersection_random_medium_p03_v1_depart1p0` 和 `sac_mlp__intersection_random_medium_p02_v1_depart1p0`。启动来源是 `fast-developer/launch_sac_mlp_random_pair.py --pair p03-p02 --start`；`launcher_manifest.json` 保存实际命令及源码 SHA，`launcher_status.json` 保存进程状态。

共享协议：sac_mlp，seed=0，CUDA，fresh=true，resume=false，smoke=false，各 100,000 raw 控制步，learning_starts=5,000 raw，checkpoint 每 10,000 raw，depart_scale=1.0；训练完成后自动评估 final_model 的 100 个 validation 回合，沿用逻辑种子 10000–10099。预热不计入控制步预算。

启动验收快照（会过时，以运行文件为准）：supervisor PID 73684；p03 worker PID 55492，5,977 raw / 977 updates；p02 worker PID 38576，5,965 raw / 965 updates。三进程均存活，两路已开始梯度更新。此前检查每路前 17 个训练回合，17 个种子、schedule SHA 和 overlay 缓存路径均唯一；全部 30/40/50 秒快照完整，requested_due = inserted + pending，diagnostic_error_count=0，无 Traceback 或 RuntimeError。

验证：相关 12 项单元测试通过，PaperEnv 集成检查 6 个场景通过；两路隔离 D1 CUDA smoke 均完成 300 raw / 241 updates 和 1 回合最终评估，证据目录为工作区 `runs/p03_sm0930`、`runs/p02_sm0930`，这些仅验证链路，不作为 100k 实验结果。原四个场景的配置 SHA 在变更前后完全一致。正式实验仍在运行，尚无成功率结论；保留旧实验及旧失败尝试，不覆盖历史结果。

## 16. 2026-09-30：p02 / p03 正式 100k 完成与多因素诊断

来源为工作区 `runs/smlp_p03p02_0930` 两个正式场景目录。均为 seed0、fresh 100,000 raw、95,001 updates、final_model、100 个共同 validation 种子；checkpoint SHA 和评估身份已核对。p02 最终成功/碰撞/超时为 5/25/70，p03 为 17/83/0；off-route均0。评估 mean_return（raw outcome）分别 -0.20/-0.66，评估记录的塑形回报总和 return_policy 均值分别 -5.720742/-5.049319，二者口径不同，也不等同于折扣加熵目标。

同seed描述性配对表（行 p02 的 S/C/T，列 p03 的 S/C/T）为 [[1,4,0],[3,22,0],[13,57,0]]。p03 比 p02 多12次成功，但多58次碰撞、少70次超时。p02 的70个timeout中，69个终态在不具备 -E1→-E0 连接的lane0/1；唯一合法连接来自lane2。timeout组平均目标速约5.15m/s但实际速1.17m/s、停驶比例76.1%，因此不能简单归因为主动低速等待。p03 碰撞组平均实际速8.46m/s，所有83个碰撞终态在路口内部；多数横向+1命令不改变其合法初始车道。

场景证据：50s p02/p03平均在网背景29.39/29.25，累计入场80.36/80.71，网外pending219.92/368.49；控制期实际入场约5400辆/小时。更高请求p主要增加待入场积压，未形成显著路内车量梯度。训练末100完整回合p02/p03为16S84C/14S86C，均无timeout，与p02确定性验证70%超时差别明显；随机动作/确定性输出、不同seed split及训练窗口模型变化须分别考虑。

两路真实优化配置一致：lr1e-4、batch32、buffer20k decision transitions、n_steps4、gamma.99、raw warmup5000、每decision3次梯度更新、auto entropy初值.2、target entropy-2、tau.005。奖励wrapper重新构造塑形reward，不叠加底层事件reward。确认一项共同非标准实现：4-step奖励按gamma的0–3次幂累加，但bootstrap折扣仍为gamma；源码注明为保持released SAC实现。CPU离线人工探针（无环境/训练）确认discount=.99而非.96059601，timeout dones=0仍bootstrap，真终止dones=1。该共同口径的性能影响未做对照训练，不应直接宣称它独自导致两组差异。

完整分层判断、最小选道对照建议、奖励/timeout契约和场景标定建议见 `p02_p03_100k_analysis_20260930.md`；离线数据为 `p02_p03_100k_diagnostics_20260930.json`、`p02_p03_traffic_20260930.json`、`p02_p03_nstep_probe_20260930.json`。本轮未修改生产代码、奖励、超参数或重跑环境；保留旧结果与此前启动快照。

## 17. 2026-09-30：新增三档 lane1 场景（配置记录，非训练结果）

新增场景为 `intersection_random_darrl_low_v1` / `intersection_random_darrl_medium_v1` / `intersection_random_darrl_high_v1`；模板分别为现有 `intersection_random_low_v1` / `intersection_random_medium_v1` / `intersection_random_high_v1`。每路每0.1秒独立Bernoulli请求p=0.03/0.05/0.07，三路期望请求量3240/5400/7560辆/小时，实际入场受容量限制。

用户明确参数：背景departLane=arrivalLane=1；ego minGap=1 m；全部相关vType的jmIgnoreFoeProb=0；SUMO碰撞车辆移除。用户确认碰撞判据仍沿用本项目的SUMO ego事件或几何重叠，未加入DARRL legacy距离/消失启发式。ego路线、起始lane2、50秒释放、原路网、600raw步上限、训练/评估及seed域均沿用模板。

本次没有新模型性能结果。新三档不能与此前p02/p03/p05场景解释为只改变p的单变量比较，因为背景车道、ego间距、junction忽略概率和SUMO碰撞处理也改变了。原六场景配置与历史结果保留。完整规格及短时环境验证证据见 [新增场景说明](random_intersection_darrl_v1_setup.md)。

## 18. 2026-09-30：SAC+MLP × DARRL low/medium，2 CUDA worker，各从零100k（启动）

用户授权后正式发起，时间约2026-09-30 19:40:59（Asia/Shanghai）。run根：`runs/smlp_darrl_0930`（相对工作区根）；launcher pair：`darrl-low-medium`。

| 方法 | 场景 | 每路每0.1秒p | 训练种子 | 预算 |
|---|---|---:|---:|---:|
| SAC+MLP | intersection_random_darrl_low_v1 | 0.03 | 0 | 100000 raw steps |
| SAC+MLP | intersection_random_darrl_medium_v1 | 0.05 | 0 | 100000 raw steps |

两路均fresh run，无resume/checkpoint输入；2个CUDA worker共享RTX5060Ti。沿用现有SAC+MLP超参数，learning_starts=5000 raw，checkpoint每10000 raw，action repeat=3，depart_scale=1.0；正式训练结束后各自动100回合validation评估。启动命令显式包含`--behavior-diagnostics`，正常收集训练/评估行为、优化记录和reset的30/40/50秒交通快照。场景采用§17及独立scenario manifests所记配置；不改变既定奖励、n-step实现或碰撞判据。

启动dry-run已核预算、场景、评估和诊断参数。此前已有16项场景单测、5项碰撞/命令测试和三档真实factory短步preflight通过；未为本次启动重复额外训练诊断。本节仅表示已启动，完成/性能结果须以后续`training_complete.json`和100回合`evaluation_results.json`为准。

启动核验通过：supervisor PID=91980，low worker PID=33712，medium worker PID=49052。核验快照中low已到8071 raw steps/3071 updates，medium已到6567 raw steps/1567 updates，两路均越过5000步随机warmup并开始优化；GPU进程列表已确认两worker。两路训练diagnostics的diagnostic_error_count均为0，stderr为空；正常episode均已保存30/40/50秒warmup快照且due=inserted+pending。以上仅为启动及持续运行证据，不是最终100k结果。未终止launcher或worker，训练与自动评估继续运行。

## 19. 2026-09-30：R1结果失效因素审计、同名DARRL r2配置及互斥终局

旧运行`runs/smlp_darrl_0930`已完成两路各100000 raw/95001 updates与100回合final eval，保存checkpoint SHA匹配。旧概率为low=.03、medium=.05（旧high=.07未训练）。该run新增SCENARIO_REVISION_NOTE.json；原结果/日志/模型均保留。32份旧场景资产、关键源码和运行清单已归档至scenario_revisions/darrl_r1_20260930，源/副本SHA一致。

旧eval存在success/collision同回合双标：low原S30/C73/双标3，medium S13/C92/双标5，各分母100且没有其它终局。碰撞优先的离线互斥计数为low S27/C73、medium S8/C92。训练episode日志中low双标29回合，medium双标10回合，均领取成功奖励+10而碰撞奖励为0；所以不能只修百分比后继续把旧模型当作正确奖励下训练的基线。详细源文件SHA、episode/seed、训练奖励证据及局限见[darrl_r1_outcome_audit_20260930.json](darrl_r1_outcome_audit_20260930.json)。

按用户明确要求原位修改同名三场景，当前configuration_revision=darrl_r2_20260930，low/medium/high p=.015/.03/.05；3路合计请求量1620/3240/5400辆每小时。其它指定车辆/车道、路网/路线、预热与seed协议保留。终局修复在统一step入口、场景子类重算之后实施collision>off_route>success>timeout，使奖励、终止类型和计数一致，并正常记录原生SUMO双事件及raw时限证据；terminal_outcome_protocol=exclusive_terminal_v2。

场景16项回归、6项终局/奖励回归（含16组合）、2项既有真实CARLA成功/timeout回归均通过；三档实际D1 factory预热+短步preflight通过。完整当前配置与后续启动证据见[darrl_r2_configuration_and_outcome_fix_20260930.md](darrl_r2_configuration_and_outcome_fix_20260930.md)。新r2与旧r1同时改变p和终局/奖励处理，新旧差异不能归因于密度单一因素。

正式重跑已于2026-09-30约22:24（Asia/Shanghai）发起，run根为`runs/smlp_dr2_0930`。低档p=.015/中档p=.03，同名r2；两路均fresh SAC+MLP、seed0、100000 raw，warmup5000 raw、checkpoint10000 raw、2 CUDA worker、100回合validation final eval、行为/优化/预热诊断开启。启动前真实单case复现SUMO到达+碰撞同时出现，修复后只判碰撞，shaped reward=-10.01，terminated=true、truncated=false，exit0（`runs/darrl_r2_terminal_check_0930/terminal_outcome_report.json`）。新运行与旧r1输出完全隔离；这里不声称100k或评估已完成。

新r2启动核验：supervisor PID72212、low PID14512、medium PID84484，启动时间22:24:35（Asia/Shanghai），两个worker均已进入GPU进程列表并持续推进。启动进度快照低2091 raw/中1190 raw，尚在5000步warmup，updates=0属预期。随后small-summary快照记录低27/中24个已结束episode，两侧diagnostic_error_count=0且有效success&&collision交集均为0；medium已观察到raw_sumo_arrived和raw_sumo_collision同真而最终is_success=false/collision=true、raw reward=-1，验证正常训练路径也使用修复后的终局。两侧manifest revision均为darrl_r2_20260930，terminal protocol为exclusive_terminal_v2。此后训练与自动评估继续运行，未停止或重启进程。

## 20. 2026-09-30—2026-10-01：旧场景 intersection_sorted / depart_scale=4.0：Topo 与 Topo+3slot 完成结果及实现归因

同场景 seed0 fresh 100k 的 ST→ST-RT→Topo→Topo+3slot final 100回合为 50/50/0 → 63/37/0 → 39/39/22 → 41/56/3（成功/碰撞/超时）。Topo的22个timeout终点均在 `-E1` lane0、pos70、route index0、actual speed0；目标速度均值约5.02m/s。静态 `map.net.xml` 只有 `-E1` lane2 能连到ego route的下一edge `-E0`，而ego从lane2出发；SMRTS contract下负lane command降索引，观测回合实际转入无出口lane0。它解释卡死近因，但不证明Topo attention直接导致策略选错lane。

实现审计确认：Topo相对ST-RT启用lane-graph query、topology token residual、关系边特征和route-goal topology residual，且query mask只显式约束有效节点、方向cosine与稀疏top-k，不是路线lane连接合法性mask。Topo+3slot不启用Graph-SLT/SBS；它只在同一incremental前向末端将joint MLP头替换为三个固定32/64/32线性投影，活跃头参数65,792→16,512。槽输入已有ego/social/route信息重叠；critic TD更新online encoder，actor对共享特征detach，因此不是auxiliary-loss训练。详见[独立实现审计](topo_3slot_implementation_audit_20260930.md)与[同场景归因报告](topo_3slot_same_scene_attribution_20260930.md)。本轮仅只读分析，未改生产算法或启动训练；后续优先验证Topo显式路线可达性，再做参数量接近的非线性三槽读出，两个方案尚未实施、没有新结果。此记录严格属于旧场景 `intersection_sorted` / `depart_scale=4.0`，不属于当前 `intersection_random_darrl_*` 三档；不能直接把旧结果用于新场景排名。

## 21. 2026-10-01：DARRL medium p=.03 七方法实验（retry02运行中；尚无结果）

计划run根为 `runs/dm7_1001`，场景为 `intersection_random_darrl_medium_v1`、修订 `darrl_r2_20260930`。三条背景进口流各自每0.1秒以p=.03请求发车，训练split=train、评估split=validation。既有 `intersection_sorted` 与DARRL r1/r2运行及资产保持原样。

| 方法 | 本轮比较身份 |
|---|---|
| `mst_slt` | 保留原MST+SLT强基线算法；新随机场景使用train/validation seed域。旧 `frozen_80_20` provenance字段不是本场景XML固定划分的证据。 |
| `sac_mlp_d1_st` | D1时空表示。 |
| `sac_mlp_d1_st_rt` | ST上加route表示。 |
| `sac_mlp_d1_st_rt_topo` | ST-RT上加Topo。 |
| `sac_mlp_d1_st_rt_topo_routeaware_v1` | 只对Topo ego-goal候选加路线可达mask及fallback；不改social/actor路径，也不是安全保证。 |
| `sac_mlp_d1_st_rt_3slot` | ST-RT使用既有线性三槽readout、不启用Topo；属于因素拆分，不是参数量匹配新头。 |
| `sac_mlp_d1_st_rt_topo_3slot` | Topo与既有线性三槽readout的组合。 |

协议计划为单训练seed0、fresh连续100000 raw SUMO steps、warmup5000 raw steps、action repeat=3、每10000 raw steps保存checkpoint；供七方法官方比较的final validation评估为100回合。MST入口是否另有内部评估及其计数待核，不将其混入官方比较回合数。最多两个CUDA worker。D1和MST入口均按raw-step预算停止，因此100000不是policy decision数量。首轮实际启动已七路全部失败退出，不填入任何S/C/T或性能排名；错误与PID见本节末。诊断预期包含行为、episode、优化和表示数据流及manifest/summary。父侧报告的6项模型诊断、11项route-helper及CPU真实环境两方法save/load/gradient检查仅为实现验证。详见 [研究上下文第17节](../RESEARCH_CONTEXT.md) 和 [方法实现地图第12节](full_mst_slt_implementation.md)。


首轮启动失败快照（2026-10-01约01:35 Asia/Shanghai）：launcher PID=98120，suite supervisor PID=42508，输出根 `runs/dm7_1001`。最终launcher status记载七方法全部failed、exit_code=1；子PID：mst_slt 28568、st 37320、st_rt 84656、topo 50304、routeaware 44148、st_rt_3slot 19372、topo_3slot 94324。根manifest和D1 arguments核实场景为DARRL medium r2/p=.03、train/validation seed域、CUDA、seed0、fresh100000 raw steps、smoke=false、behavior diagnostics开启、官方比较eval=100、checkpoint每10000 raw。七项均未完成训练/最终评估，故无S/C/T结果。日志的首轮共同错误为 `sumo_env.py:659` 的 `info.update(behavior_telemetry_protocol=..., **route_lane_facts)` 与route_lane_facts自身同名键冲突，抛出TypeError。run和失败日志/source archive保留；进程已自然退出。后续修复后的fresh retry应使用新根，不覆盖本次失败证据；此时retry尚未启动。


Retry01续记（2026-10-01约01:45 Asia/Shanghai）：隔离根 `runs/dm7_1001_retry01`，launcher PID=67536、suite supervisor PID=68888。六个D1方法均early exit code 1，错误为 `high_density_env_v1.py:355` overlay manifest `os.replace` 的WinError 3/FileNotFoundError；MST记录到1530 raw steps后，root终止该supervisor树，避免接着其临时进度训练或混合为正式run。1530 raw不是checkpoint恢复点，本组后续正式run须fresh。日志目标路径长度由root量得ST=261、Topo=269、routeaware=283字符，source/temp约202–224；只量源临时路径不能排除路径长度问题，须同时核两端，原因仍待验证。retry01全部失败产物、日志和source archive保留；下一fresh root尚待确定；本轮无训练完成或final eval结果。


路径原因后续验证：独立Windows探针使用实际manifest basename、parent存在且source temp写入成功，DEST=240和259字符的replace成功；DEST=260、261、269、283均复现WinError3/FileNotFoundError。源数据见工作区 `tmp/manifest_path_probe_20261001_015015/results.json`。retry01实际错误DEST长度ST=261、Topo=269、routeaware=283，与该阈值复现相符；source/temp虽较短，并不能排除DST限制。由此路径长度是目前有直接复现支持的根因解释；仍需修复后在新fresh根实测验证。之前“根因待验证”是探针之前的状态快照。



## 2026-10-01 正式队列确认：dm7_1001_retry02

当前有效运行是 `runs/dm7_1001_retry02`，已从零启动七方法；前两次目录 `dm7_1001`、`dm7_1001_retry01` 保留为失败/中断尝试，不作为性能结果，也未从其中恢复训练。新增诊断字段重复合并已修复；Windows 长目标路径问题通过仅缩短 D1 车流缓存到 `run_root/_hd/<method>/ns_tr|ns_eval` 修复，不改变模型、场景参数或随机种子。真实环境五次 reset+step 验证通过，同 seed 在不同存储位置生成的路线源/overlay 哈希和车辆数一致；验证来源为 `runs/dm7_pathchk/overlay_path_smoke_results.json`。

协议：`intersection_random_darrl_medium_v1`，DARRL r2，三路各 p=0.03/0.1秒；seed 0；每方法 fresh 100000 raw SUMO steps；checkpoint 每10000 raw steps；最多两 CUDA worker；官方比较采用 validation 外层100回合。方法为 `mst_slt`、`sac_mlp_d1_st`、`sac_mlp_d1_st_rt`、`sac_mlp_d1_st_rt_topo`、`sac_mlp_d1_st_rt_topo_routeaware_v1`、`sac_mlp_d1_st_rt_3slot`、`sac_mlp_d1_st_rt_topo_3slot`。MST原入口另有内部100回合评估；只有外层 `evaluation_results.json` 用于七方法比较，新增日志没有增加评估回合。详见正式根 `evaluation_accounting.json`。

已核验运行快照（2026-10-01 02:13:33 +08:00）：launcher PID 66336，supervisor PID 68208；MST PID39120 已6472 raw/2168 decisions；ST PID14924 progress 已6274 raw/1274 updates（异步diagnostics summary为6313 raw）。两路诊断错误计数均0，活动日志无Traceback；其余五路待worker空闲依次执行。这是运行健康检查，不是最终性能结果，100k 尚未在该快照完成。

日志已实际写出 raw、decision、episode、optimization、representation 数据。训练输入快照每10000 raw一次、最多10份；外层评估首动作与终止动作输入最多200份/worker。字段包括动作/目标与实际速度、换道请求与实际转换、路线接续合法性及错道停滞、TTC/碰撞位置、Topo/三槽表示与梯度、优化指标。D1 ST首回合预热30/40/50秒抽样：due与inserted分别19/28/41，pending均0，在网背景车15/13/21，halting均0；due-inserted-pending差额均0。这只是日志与预热流程核验，不是场景总体统计。

已通过17项behavior测试、17项route/encoder定向测试（另4个subtests）、1项真实诊断环境集成测试，以及新增方法真实环境预测/梯度/保存加载验证。源码实际内容与SHA归档在正式根 `source_archive/manifest.json`，完整启动配置在 `suite_manifest.json`。后续分析应读取最终checkpoint和外层评估原始结果，不能用本段早期训练快照推断方法优劣。


## 2026-10-01：当前中档两基线与旧场景反转核验

完整报告：[darrl_medium_mst_vs_sac_analysis_20261001.md](darrl_medium_mst_vs_sac_analysis_20261001.md)；可复算数据：[darrl_medium_mst_vs_sac_20261001.json](darrl_medium_mst_vs_sac_20261001.json)。当前SAC来自runs/smlp_dr2_0930，MST来自runs/dm7_1001_retry02；共同DARRL r2 medium p=.03、seed0、100000 raw、95001 updates、外层final validation100，逐seed计划SHA匹配100/100。S/C/T/O分别31/62/7/0与16/53/31/0。SAC行→MST列(S,C,T)矩阵为[[3,21,7],[12,28,22],[1,4,2]]；成功新增13、丢失28。配对bootstrap95%成功差[-27,-3]pp、McNemar p=.027533只适用于本批固定策略/评估交通，不代表多训练seed稳定性。

已有日志定位：MST31个超时中21个终点在-E1_0/1约70m、不能接续目标路线；7个SAC成功→MST超时中6个为这种错道末端停滞。另21个SAC成功→MST碰撞中20个碰前在:J1_21、1个在-E0起段，碰前速度4.62–9.74m/s；不能将全部成功损失归为超时或保守。MST停驶样本占58.28%，SAC为0.093%。终局快照/几何碰撞不等于已确定具体冲突车或精确撞点。

历史sorted/depart4：旧SAC22/38/40、旧MST53/47/0，旧eval traffic_variant按seed匹配100/100。旧SAC声明训练/评估共享全30模板、旧MSTrequested元数据写frozen_80_20，旧实际训练池一致性未证实。旧环境源码未完整归档，旧运行时碰撞参数未知；旧`scenario.sumocfg`的静态`collision.action=warn`不能证明实际argv。对当前`sortg3_1002`两个sorted D1方法，锁定源码链构造的SUMO命令为`collision.action=none`、`collision.check-junctions=true`，但原始child argv未保存，因此这是源码推断而非运行实录，也不能再概括成“旧warn、新remove”。新方法使用`exclusive_terminal_v2`；旧终局语义仍按各自原始证据描述。相同地图/ego路线下仍同时改变背景lane0→lane1、jmIgnoreFoeProb0.02–0.89→0、minGap2→1、固定模板循环重插→Bernoulli无循环等；故跨场景反转不是单变量因果效应。

当前仅记录基线结论。七方法初始汇总为6完成/1仍训练，Topo+3slot没有final结果；没有为本分析新开训练或评估。

## 2026-10-01：交叉口实验历史总表索引（正确三流起点）

总表：[intersection_experiment_summary_20261001.md](intersection_experiment_summary_20261001.md)；配置沿革与场景族：[intersection_scenario_catalog_20261001.json](intersection_scenario_catalog_20261001.json)。纳入分界是2026-09-28生成的`intersection_sorted`及`.depart_sorted`标记；旧`intersection`的方向分组交通只实际生效单流，因此旧yield_v2/单流结果不进入本表。

本次核对38条非smoke最终评估：sorted 24条（50k、50k检查点续训、fresh100k与两个额外depart密度），早期随机原生/显式概率场景4条，DARRL r1两条，DARRL r2八条。r1两条存在success/collision双标及训练奖励污染，保留为历史记录但不作为修复后互斥终局的有效基线；其余36条按互斥S/C/T结局汇总。续训行与fresh100k分开统计；单训练seed的观察不代表跨seed稳定性。

车流单位需按配置区分：`depart4p0`缩放固定XML中车辆depart时间戳；原生随机场景以veh/h配置SUMO flow；标准p05/p03/p02及DARRL r1/r2按每路线、每0.1秒步概率预采样车辆请求。DARRL r1/r2是同名场景的不同configuration revision，概率与终局处理不同，不能按场景名合并。完整计数、结果、源路径和局限见上述汇总报告与场景目录。

截至2026-10-01 20:22（北京时间），DARRL r2 medium队列的Topo+3slot为85413 raw steps/80413 updates，仍未发现`training_complete.json`、`evaluation_results.json`或final模型；正确sorted/depart4下该方法有已完成旧场景评估。此次只汇总现存结果，未启动新训练或评估。

## 2026-10-01：38条历史结果的奖励口径

逐运行证据及限定见[奖励审计](intersection_reward_audit_20261001.md)：范围是local 20 + runs 18；归档支持共同v2奖励系数设计，但历史源码/终局行为并非对全部38条都完整可核。
11条local fresh SAC/D1配置显式记录六个系数，fresh MST两条缺reward字段（其中100k Monitor六分量匹配）；7条续训不含独立reward/action_repeat/gamma字段，不能称已完整核实。
r1 low/medium有训练成功/碰撞双标29/10、final评估双标3/5；success分支领取+10，碰撞−10被跳过。r2改为互斥终局，优先collision > off-route > success > timeout。
训练每个决策步的v2回报含终局项、−.01和有条件的+ .02×累计行驶距离差；距离差无clamp，TraCI距离是插入后的累计行驶米数。
正式评估`episode_return`读原始无折扣`success−collision`，不是shaped training return；SLT/Graph-SLT/SBS列作辅助优化目标，不列作环境奖励。此次未运行新实验。

## 2026-10-01 22:26：旧场景 sorted 主阶段启动快照（历史；其后已完成）

用户将当前主场景从前一阶段 DARRL medium p=.03 切回正确三路固定模板 `intersection_sorted`、`depart_scale=4.0`。本轮两个 fresh SAC+MLP 增量方法为 `sac_mlp_d1_st_rt_topo_routeaware_v1` 与 `sac_mlp_d1_st_rt_3slot`，训练seed0、各100000 raw SUMO steps、两独立CUDA worker、每10000 raw checkpoint、final validation 100回合；未resume、无smoke。正式根 [`runs/sort2_1001`](../../../runs/sort2_1001)，suite配置和实际交通/source快照见 `suite_manifest.json`、`source_archive/manifest.json`。

该根于2026-10-01 22:26启动；截至22:31(+08)核验routeaware PID77596为5678 raw/678 updates、3slot PID80940为5980 raw/980 updates，均CUDA training并持续推进；诊断错误数0、活动日志无traceback。当时尚无100k完成或最终评估结果；这是启动阶段的历史快照，当前完成状态见下方2026-10-02记录。DARRL实验仍保留为上一独立阶段，不能和固定sorted交通按单变量结果合并。

新增ST spatial/temporal/social、RT、Topo/route-reachability、slot/readout activation及critic梯度/参数更新诊断，启动前43项定向单测通过。route-aware 300-raw/1-episode smoke首次在 [`runs/s2sm_1001`](../../../runs/s2sm_1001) 通过；3slot smoke训练通过、首次eval因为hook缓存`components`字典误入数值指标而失败，失败attempt及日志保留。将scratch与可序列化标量分开并在Nullable转换中保留null后，3slot单方法retry在 [`runs/s2sm_r1_1001`](../../../runs/s2sm_r1_1001) 通过；smoke不作为性能证据。诊断采样、分母和解释边界见[协议](sorted_routeaware_3slot_plan_20261001.md)及[诊断字段说明](module_diagnostics_protocol_20261001.md)。

本阶段新评估`episode_return`改为逐decision的环境shaped reward和，标记`environment_step_reward_v2`；底层raw success−collision单独放入`raw_episode_return`/raw summary字段，并保留有数据时的奖励分量对账。此前[38条结果奖励审计](intersection_reward_audit_20261001.md)是修复前历史快照，内文对raw `mean_return` 的说法只适用于该批旧记录；旧数值及分母不重写、不伪作新口径。

## 2026-10-02：三条正式完成结果补记与当前计数

`runs/sort2_1001/sac_mlp_d1_st_rt_3slot__intersection_sorted_depart4p0` 已核验 fresh seed0、100000 raw steps/95001 updates、final validation 100回合（seeds 10000–10099），S/C/T/O=34/49/17/0。新评估主回报为`environment_step_reward_v2` shaped mean/std=−1.3883924/9.8976625；raw undiscounted mean/std=−0.15/0.8986100，六奖励分量最大对账误差3.55e−15，train/eval diagnostic error=0。checkpoint、完成记录和evaluation身份SHA一致：`d46b5cf71cece2a39e589a631c6624eee3f9807792ef9a77443c4aa17c00dd1d`。完整字段见[sorted完成快照](sorted_pair_completion_20261002.json)。

`runs/sort2_1001/sac_mlp_d1_st_rt_topo_routeaware_v1__intersection_sorted_depart4p0` 现已核验 fresh seed0、100000 raw steps/95001 updates、final validation 100回合（seeds 10000–10099），S/C/T/O=43/32/25/0。新评估为`environment_step_reward_v2` shaped mean/std=1.0395563/9.9573802，raw undiscounted mean/std=0.11/0.8590111；六奖励分量最大对账误差3.55e−15，train/eval diagnostic error=0。final checkpoint、training_complete及evaluation identity SHA一致：`cee4041ceb636c77ecb6a8e6c43ed5e1ce74aa4002b486786bb4d5f8104107c8`。精确来源为该run目录下的`evaluation_results.json`及模型完成/评估记录；逐方法协议与配对矩阵见[sorted模块结果审计](sorted_module_outcomes_20261002.json)。

`runs/dm7_1001_retry02/sac_mlp_d1_st_rt_topo_3slot__intersection_random_darrl_medium_v1_depart1p0` 已核验 fresh seed0、100000 raw/95001 updates、final validation 100回合，S/C/T/O=20/47/33/0，旧schema raw `mean_return=-0.27`，checkpoint SHA=`b89d891756dbb8f21dfc5225ecd8efe3e88b3cb70e85ba07dff97b46984dc77d`。该DARRL结果与sorted结果场景、评估回报schema不同，分开记账。2026-10-01 20:22记录的Topo+3slot pending已被这条完成记录取代。

更新后共41条非smoke最终评估：sorted26、早期随机4、DARRL r1两条双标/奖励污染历史、DARRL r2九条；39条为互斥终局，另2条为污染历史。此前routeaware运行中快照（2026-10-02 03:35:28(+08)，PID77596，80569 raw/75569 updates）已由本次final结果取代。新的sorted配对审计核对7方法均fresh seed0/100k raw，外层validation seed10000–10099，seed和traffic模板配对100/100一致、固定pool 30份XML内容hash一致；单训练seed与复用模板限制仍成立。新routeaware/3slot的shaped评估回报与旧raw评估回报分开记录；不同结构的配对变动只是本批固定策略观察，不是多训练seed因果估计。本次未启动训练或评估。模块归因见`sorted_module_attribution_20261002.md`及本历史记录第22节。

后续记账规则：每次正式实验完成并核验final checkpoint哈希、评估回合数/seed和评估文件身份后，及时更新全量汇总、`RESEARCH_CONTEXT.md`和本历史记录；此前只保留带时间戳的运行快照，不写完成或性能结论。automation-3最后读取状态为ACTIVE；其prompt原定在两路完成后暂停。本次没有操作automation；暂停由父任务决定。

## 22. 2026-10-02：sorted 模块归因与待验证的下一步

本轮归因报告已完成：[sorted_module_attribution_20261002.md](sorted_module_attribution_20261002.md)，配套逐回合/配对数据与诊断汇总见[结果JSON](sorted_module_outcomes_20261002.json)和[诊断JSON](sorted_module_diagnostics_20261002.json)。在该 sorted/depart4、seed0、fresh100k 固定策略评估链中，ST-RT为63/37/0（S/C/T）；Topo为39/39/22，route-aware Topo为43/32/25，ST-RT+3slot为34/49/17，Topo+3slot为41/56/3。route-aware的25个超时均落在对目标下一edge不可达的`-E1_0`；合法goal attention质量正常，但mask不限制动作。3slot改变的不仅是语义槽布局，还把65,792参数联合非线性readout换成16,512参数线性分槽readout；下游actor/critic仍可混合，故当前退化不等于语义槽思想被否定。

后续优先建议是当前冻结ST-RT对照与等参数非线性3slot；之后再用独立goal-only Topo对照拆分拓扑路径。它们尚未实现或启动，属于待验证设计。既有residual初始化尺度1e-3已经存在，不重复提出简单“小门控”作为新改进。限制仍是单训练seed及复用30个固定流量模板；新sorted评估的`environment_step_reward_v2` shaped return与旧历史raw `mean_return`分开解释，且跨日期源码等价不能仅由超参/方法名推定。此次只更新研究记录，没有新增训练或评估。

为未来 incremental train/eval，已按用户授权增加 actor-intent post-LN 实际增量与 same-support uniform 关系基准；既有 goal post-LN 指标复用，历史 source/result 文件保持原样。本次无训练、eval 或 SUMO 运行。相关 encoder 测试30项、module diagnostics测试7项通过，其中4项目标测试属于上述测试子集；旧 full delegate 路径绕过这些新增字段，详见[采集协议](module_diagnostics_protocol_20261001.md)。本次分析与诊断补充完成后，automation-3 已由应用工具确认暂停，避免重复跟进。

## 23. 2026-10-02：sorted goal-only / nonlinear 3-slot 正式训练启动快照

实现smoke在 [`runs/sortg3sm_1002`](../../../runs/sortg3sm_1002) 完成，两个worker exit0；每种方法300 raw/241 updates、1个validation episode(seed10000)，final checkpoint hash与training/evaluation身份匹配。训练shadow各3 unique state，评估各2；每个状态12个variant行，active errors与diagnostic errors均0，inactive分支明确记为NA。两个smoke episode恰为collision，不能用于性能结论；shaped/raw回报分列且奖励分量对账误差小于4e−15。机器可读receipt见[smoke验证](sorted_goalonly_nonlinear3slot_smoke_validation_20261002.json)。

正式双路于2026-10-02 09:21:58(+08)启动于 [`runs/sortg3_1002`](../../../runs/sortg3_1002)，未恢复旧checkpoint。supervisor/coordinator PID38612/40248；独立CUDA worker PID17464为`sac_mlp_d1_st_rt_3slot_nonlinear_v1`、PID85576为`sac_mlp_d1_st_rt_topo_goalonly_v1`。args为sorted depart4、seed0、fresh raw100000、warmup5000、checkpoint每10000 raw、外层validation100，behavior diagnostics和policy shadow启用（训练每5000 raw、eval每episode至多3状态）。source archive记录807源码文件和30个sorted模板hash，含当前runner/launcher/shadow helper/tests。09:26(+08)快照：3slot 5381 raw/381 updates，goal-only 5082 raw/82 updates，二者均training；训练behavior约5486/5332 raw records及1834/1783 decision records，diagnostic_error_count=0；shadow各1个unique state、12行，active有效数8/7、inactive4/5、active invalid与exception均0。此时尚无training_complete、final_model或evaluation_results；不是结果。完整启动字段见[formal启动receipt](sorted_goalonly_nonlinear3slot_formal_start_20261002.json)，理论定义/比较边界见[研究协议](sorted_goalonly_nonlinear3slot_research_protocol_20261002.md)。既有41条非smoke完成评估计数不变。automation-3已恢复ACTIVE，指向本run完成后复核和暂停；不能把当前快照写成完成状态。

## 23. 2026-10-02：goal-only Topo / nonlinear 3-slot smoke与正式运行快照

独立smoke根 [`runs/sortg3sm_1002`](../../../runs/sortg3sm_1002) 已由launcher完成，两个worker exit0。两法均为sorted/depart4、seed0、300 raw/241 updates、final checkpoint SHA与training_complete/evaluation identity一致；各做1个validation seed10000评估，碰撞结果仅作实现验证。训练shadow分别采3个unique state，eval分别采2个；每state12行，active probe错误、诊断错误均为0，不适用分支显式标记。shaped与raw return分开记录，六分量重构误差均低于4e−15。详情见[机器可读smoke receipt](sorted_goalonly_nonlinear3slot_smoke_validation_20261002.json)。

正式两路已于2026-10-02 09:21:58(+08)启动，根为 [`runs/sortg3_1002`](../../../runs/sortg3_1002)，supervisor PID38612/coordinator PID40248，两个独立CUDA worker分别PID17464 (`sac_mlp_d1_st_rt_3slot_nonlinear_v1`) 与PID85576 (`sac_mlp_d1_st_rt_topo_goalonly_v1`)。arguments确认fresh seed0、100000 raw、warmup5000、checkpoint10000、外层validation100、sorted/depart4、policy shadow interval5000与eval每回合最多3状态；没有resume参数。启动后9:22:42两路各897 raw/0 updates，9:23:17各2094 raw/0 updates，仍处warmup；behavior telemetry已写约2090 raw/698 decision记录且diagnostic_error_count=0。source archive含807个源文件及30个sorted template hash。检查时未生成checkpoint/final evaluation，不能报告性能结果；早期历史计数41条不变。方法机制与推断限制见[研究协议](sorted_goalonly_nonlinear3slot_research_protocol_20261002.md)；运行源码/scene快照见该run的`source_archive/manifest.json`。automation-3已由应用工具确认ACTIVE，后续提示指向本run完成后复核并暂停；不得把当前训练快照当最终状态。

## 24. 2026-10-02：sorted 参数匹配非线性3slot完成（当时goal-only仍运行）

`runs/sortg3_1002/sac_mlp_d1_st_rt_3slot_nonlinear_v1__intersection_sorted_depart4p0`已核验fresh seed0、100000 raw steps/95001 updates、final validation 100回合（逻辑seed 10000–10099），S/C/T/O=50/48/2/0。Shaped/raw mean/std=1.67910049/10.77594583与0.02/0.98974744；六项奖励分量覆盖100回合、最大对账误差3.55e−15，train/eval诊断错误均0。Final checkpoint、training_complete与evaluation identity SHA一致：`cb8b19bd67c982eddf0d377f54b0f7b03fcce7ed9261328ed88790228c479306`。shadow训练20状态/240行、评估200状态/2400行，active有效160/1600、NA80/800、invalid/exception=0。验收来源：[完成记录](sorted_goalonly_nonlinear3slot_completion_20261002.json)。

在本条记录写入时，全量清单从41条增至42条：sorted27、早期随机4、DARRL r1污染历史2、DARRL r2 9；其中40条为互斥终局、2条为r1污染历史。该结果与旧线性ST-RT+3slot（34/49/17）及ST-RT（63/37/0）同为sorted/depart4、seed0、fresh100k、final100的描述性比较；非线性槽为50/48/2。单训练seed与固定评估模板不支持多seed稳定性结论，也不能据此证明语义分槽的因果收益。当时goal-only尚无final模型或评估；该状态由下一节的完成验收取代。本条保留为当时快照。

## 25. 2026-10-02：sorted goal-only Topo 完成并通过身份核验

`runs/sortg3_1002/sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0`已核验fresh seed0、100000 raw steps/95001 updates、final validation 100回合（逻辑seed 10000–10099），S/C/T/O=24/47/29/0。Shaped/raw mean/std=−2.94594740/8.78128240与−0.23/0.81061705；六项奖励分量覆盖100回合、最大对账误差3.55e−15，train/eval diagnostics errors均0。Final checkpoint、training_complete与evaluation identity SHA一致：`17465d202736d4640d46459f78bc4b55cef939ab6e3aa39b9cad63a981f1c5f5`。训练诊断记录100000 raw/33504 decisions、526个完成回合及1个预算截断回合；评估诊断记录34892 raw/11657 decisions、100回合。

Policy-shadow训练采样20个unique state/240行，评估采样200个state/2400行（100个回合各2个state，预算上限3）；每state记录12个variants，适用分支有效140/1400行、非适用NA 100/1000行，active-invalid与exception均0。按(eval seed, traffic_variant)核验与旧ST-RT、Topo、route-aware Topo分别100/100配对，100个逻辑seed映射一致；30个有效route模板文件逐一SHA一致。配对矩阵方向为旧方法 outcome 行→goal-only outcome 列：ST-RT的S行15/33/15、C行9/14/14；Topo的S行11/19/9、C行7/18/14、T行6/10/6；route-aware的S行14/17/12、C行7/15/10、T行3/15/7（各行列序为成功/碰撞/超时）。与线性ST-RT+3slot及ST-RT到非线性3slot的配对矩阵也保存在机器可读文件。

此次完成后全量清单由42条增至43条非smoke正式最终评估：sorted28、早期随机4、DARRL r1污染历史2、DARRL r2 9；其中41条为互斥终局、另2条为r1污染历史。Goal-only与此前非线性3slot共享单训练seed=0、sorted/depart4、同一100个逻辑评估seed和30个模板；这是固定checkpoint评估池上的描述性比较，不估计多训练seed稳定性，也不作模块因果结论。Shaped `environment_step_reward_v2`回报与旧raw回报不可直接比较。完成验收见[receipt](sorted_goalonly_nonlinear3slot_completion_20261002.json)，配对结果及复现脚本见[配对JSON](sorted_goalonly_nonlinear3slot_pairing_20261002.json)与[只读脚本](sorted_goalonly_nonlinear3slot_pairing_20261002.py)。新增诊断/干预评估尚未运行；根因分析仍待进行。

## 26. 2026-10-02：冻结 goal-only 三臂诊断完成

本次新增的是同一已训练checkpoint上的300回合冻结策略诊断，不是新训练，因此既有43条正式训练结果（41条互斥终局及2条R1污染历史）计数不变。Control、topology-goal-off、route-lane-veto均为100回合CPU确定性评估，使用逻辑seed 10000–10099及同一sorted/depart4流量模板；S/C/T/O分别为24/47/29/0、45/55/0/0、44/56/0/0。Control逐seed复现原正式评估的终局、回报、steps、交通variant和行为汇总。三臂使用同一final checkpoint（SHA256 `17465d202736d4640d46459f78bc4b55cef939ab6e3aa39b9cad63a981f1c5f5`，95001 updates），权重指纹前后不变。

评估奖励须区分：shaped mean return为−2.945947、0.943300、0.703164；raw episode return为−0.23、−0.10、−0.12；behavior `return_base`为−0.227306、−0.098606、−0.119301，后者包含每个action-repeat decision内部0.99折扣，不能当作raw episode return。六项shaped分量覆盖每臂100回合，最大重构误差3.55e−15。Control与route-veto分别采集300与297个唯一状态、各12个variant/state，active错误为0、inactive分支显式NA；goal-off按预注册方案关闭shadow以避免嵌套toggle。Route-veto记录1778次hold拦截，保留纵向动作，但不证明实际换道或安全性。

在这个冻结checkpoint上，goal branch关闭后成功率增加21个百分点、超时减少29个百分点、碰撞增加8个百分点；route-veto相对control将29个超时改成9个成功和20个碰撞，成功增加20、碰撞增加9。它们是固定checkpoint/评估池上的描述性干预结果，不等同重新训练消融、多训练seed稳定性或安全保证。结果与配对矩阵见[离线审计JSON](sorted_goalonly_causal_results_20261002.json)，采样覆盖见[probe审计JSON](sorted_goalonly_causal_probe_audit_20261002.json)，完整判读见[归因报告](sorted_task_module_attribution_20261002.md)。启动后源码归档共195个文件；40/40具有launch hash的源文件/模板匹配，未带launch hash的map/ego副本只代表事后采集，详见正式run的`source_archive_postlaunch/source_archive_manifest.json`。本次不新增full训练。


派生行为与末次policy-input伙伴复核见[独立归因JSON](analysis/sortg3_causal_1002_attribution.json)和[只读复现脚本](analysis/sortg3_causal_1002_attribution.py)；route-veto伙伴关联按观测ID核实，sidecar的hold动作不能被解读为原策略未提出不可达请求。


### 后续诊断采样器验证（cap=4）

为后续评估新增的critical selector支持每回合最多4个unique state：initial、首个route-critical、首个TTC-critical及preterminal；相同decision的多触发合并为同一state。命令要求同时启用 `--behavior-diagnostics --policy-shadow-probes --policy-shadow-critical-eval-sample --policy-shadow-eval-max-per-episode 4`。route/TTC观测时间以pre-action预测输入和action-repeat完成后的context窗口明确标注，记录 `source_action_pre_obs_raw_global_step` / `_episode_step`、`context_window_start_raw_global_step` / `_episode_step`、`context_window_end_raw_global_step` / `_episode_step`。cap=3路径保持兼容，默认关闭；训练shadow仍按5000 raw周期采样。文档代理在main1的 `tests_sb3_sumo/test_d1_policy_shadow_critical_sampling.py` 上执行pytest，10项通过（exit 0；当前维护者未重复运行）。此前已完成的300回合冻结诊断仍为cap=3。