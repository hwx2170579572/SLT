# D1 C8/C9 contractfix 诊断审计（2026-10-03）

本审计汇总已落盘的 ST 与 ST-RT 训练和评估诊断。方法分别位于 runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0 和 runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0。按 analysis/d1_contractfix_protocol_20261003.md，两者联合修复了 C8 历史最后有效位置和 C9 几何边 Cartesian 速度；reward、SAC、replay、单 gamma、timeout bootstrap 与预算协议保持原设定。

审计使用已落盘文件。每种方法只有训练 seed 0 和固定 validation seeds 10000–10099，不构成多训练种子估计。C8 与 C9 同时修复，结果不能分解为单项修复贡献。

## 训练与评估计数

| 方法 / phase | audit 决策行 | 计入 raw steps | started episodes | completed episodes | errors / unknown raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| ST train | 33,484 | 100,000 | 464 | 463 | 0 / 0 |
| ST eval | 8,644 | 25,832 | 100 | 100 | 0 / 0 |
| ST-RT train | 33,500 | 100,000 | 462 | 461 | 0 / 0 |
| ST-RT eval | 8,773 | 26,218 | 100 | 100 | 0 / 0 |

训练诊断 summary 的 raw 与 decision 计数和 trajectory audit 完全相同。train_monitor.csv 分别有 463、461 个完成 episode。诊断 summary 的 episodes_finished 分别为 464、462，是因为它还包含训练关闭时尚未终止的最后一个 episode：ST 为 118 raw / 40 decisions，ST-RT 为 255 raw / 85 decisions，均标记为 closed_before_terminal / incomplete。trajectory audit 因此记录多一个 started episode，completed 数仍与 monitor 一致。末尾还有一条 raw_steps_executed=0 的审计决策行；它不增加 raw 总数，unknown raw 均为 0。

两路 training_complete.json 都记录 100,000 raw / 95,001 updates。最终 evaluation_results.json 各含 100 条回合记录，checkpoint SHA 与对应 training_complete.json 一致。ST 为 45 成功、50 碰撞、5 超时、0 off-route，shaped return 均值 0.793562、raw return 均值 −0.05；ST-RT 为 52/48/0/0，shaped 均值 2.028683、raw 均值 0.04。两者奖励分量 coverage 均为 100/100，最大六分量对账误差为 3.55e−15；主回报口径为 environment_step_reward_v2。

## C8 历史 mask 与旧索引反事实

Trajectory audit 每行来自真实策略决策的 pre-action observation；不统计 replay reuse 或 shadow forward。分母是该 phase 中 actor-history observation 的数目：每个决策一个 ego history 和五个 social history。所有四个 phase 的历史均非空、无内部缺帧；ego 历史未出现旧索引不匹配。social 统计如下。

| 方法 / phase | nonempty social histories | short | left padding | right padding | 旧 count−1 错位 | 旧索引选到 padding | 旧索引选到较早有效帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ST train | 167,420 | 7,073 (4.22%) | 113 | 6,960 | 113 (0.0675%) | 61 (0.0364%) | 52 |
| ST eval | 43,220 | 1,522 (3.52%) | 22 | 1,500 | 22 (0.0509%) | 12 (0.0278%) | 10 |
| ST-RT train | 167,500 | 7,018 (4.19%) | 88 | 6,930 | 88 (0.0525%) | 49 (0.0293%) | 39 |
| ST-RT eval | 43,865 | 1,533 (3.49%) | 33 | 1,500 | 33 (0.0752%) | 19 (0.0433%) | 14 |

旧索引错位率以 nonempty social histories 为分母；错位占短历史的比例分别为 1.60%、1.45%、1.25%、2.15%。这里的命名按 collector 源码定义：left_padding 表示有效帧首索引大于 0，即前缀 padding、有效帧在右侧；right_padding 表示有效帧末索引小于 H−1，即后缀 padding、有效帧在左侧。因此短历史主要是 right padding，少量 left padding 才导致 count−1 与 last-True 不同，两项计数自洽。

首条可复核 mismatch 在 ST train trajectory_history_audit.jsonl 第 99 行：episode 0 / decision 97，审计时环境 raw=291、history_timestep=292（next_append_index）。该决策的 social 组 5 个槽中有一个 2 帧短历史，left_padding=1、right_padding=0、internal_gaps=0、old_index_mismatch=1、old_selected_padding=1。对应 actor cache slot 5 的 first_seen=290、last_seen=291、cached_history_length=2、tracked_age_slots=2。十帧历史布局下，聚合字段表示前 8 格为 padding、末 2 格有效；count−1 会取 index 1 的 padding。JSONL 保存了计数和 cache 元数据，没有序列化完整 10-bit mask，因此上述具体位置由 first/last、无 gaps、缓存长度共同重建，不是直接保存的 mask 向量。

actor history cache 在每个真实决策均可用：ST train/eval 分别是 200,904/51,864 条 actor history，ST-RT train/eval 为 201,000/52,638 条，均 nonempty。tracked age 使用 tick − first_seen；collector 将 tick 定义为 next_append_index。legacy mask 仍为 x != 0 proxy。四阶段 x=0 但其他 state 字段非零的槽计数均为 0；这不消除 x=0 全零槽与真实存在状态之间的语义歧义。

上述 mismatch 是旧 count−1 实现的反事实计数，修复后的 selector 按实际 last-True 选取。它表示全量轨迹中的暴露频率，不表示修复实现仍然选错，也不能单凭较低频率否定关键状态上的影响。

## C9 几何边与 CPA 代理量

四阶段 summary 均记录 geometry active=true、velocity_contract=smarts、status=ok、20 秒 horizon。统计分母是同一历史时刻、无序有效 actor pair 在真实决策轨迹中的重复观测；同一 pair 会跨历史槽和决策重复出现，不是独立交通事件。

| 方法 / phase | pair-history observations | closing 符号反转（旧→Cartesian） | 旧 closing 为零而新值非零 | 平均绝对 closing 差（m/s） | relative speed 为零 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ST train | 4,894,715 | 2,505,431 (51.19%) | 310,345 | 11.029 | 63 |
| ST eval | 1,269,115 | 654,267 (51.55%) | 77,025 | 10.954 | 24 |
| ST-RT train | 4,898,200 | 2,503,179 (51.10%) | 308,646 | 11.074 | 60 |
| ST-RT eval | 1,288,145 | 653,821 (50.76%) | 81,144 | 10.828 | 28 |

符号变化和非零 closing 差说明把 pseudo velocity 改回与世界位置同基底的 Cartesian 速度不是无效变换；ST 与 ST-RT 几何均 active，ST 不应记为 NA。这里的时间和距离仍是常速度 closest-approach 代理量，不是真实 TTC；零相对速度时最近时刻未定义。summary 保存聚合差异和计数，没有逐 pair 的完整差值分布或可用于交通事件推断的独立样本，故本审计不把这些数解释成碰撞概率或 C9 的性能因果效应。

## Shadow probe 覆盖与同状态动作敏感性

| 方法 / phase | unique states | probe rows | applicable rows | errors / active-invalid | 预算核对 |
| --- | ---: | ---: | ---: | ---: | --- |
| ST train | 20 | 280 | 60 | 0 / 0 | 全训练上限 20 |
| ST eval | 357 | 4,998 | 1,071 | 0 / 0 | 100 episodes；最多 400 unique（每回合 ≤4） |
| ST-RT train | 20 | 280 | 100 | 0 / 0 | 全训练上限 20 |
| ST-RT eval | 298 | 4,172 | 1,490 | 0 / 0 | 100 episodes；最多 400 unique（每回合 ≤4） |

inactive rows 是对应分支未启用的预期 skip；variant rows 不是 episode 数。有效同状态 action sensitivity 按每条 JSONL 的 action_delta_l2_normalized 求均值：这是 shadow_action_normalized_mean 与 baseline_action_normalized_mean 的二维差向量 L2 范数，不是 mean absolute delta，也不是 m/s。manifest 的动作空间为两个归一化维度 [-1,1]，该无量纲 L2 范围为 0–2√2。所有下表行的样本数均等于有效且 applicable 的 unique states，零 delta 样本数为 0。

| 方法 / phase | Probe | 有效 states | mean action_delta_l2_normalized |
| --- | --- | ---: | ---: |
| ST eval | st_spatial_off | 357 | 0.6623 |
| ST eval | st_temporal_current_only | 357 | 0.8929 |
| ST eval | st_social_ego_only | 357 | 0.8219 |
| ST-RT eval | st_spatial_off | 298 | 0.9035 |
| ST-RT eval | st_temporal_current_only | 298 | 0.7559 |
| ST-RT eval | st_social_ego_only | 298 | 0.4751 |
| ST-RT eval | rt_intent_injection_off | 298 | 0.7006 |
| ST-RT eval | rt_route_readout_zero | 298 | 0.1074 |

训练阶段 20-state shadow probes 上的对应均值为：ST spatial/temporal/social 0.6339/0.8062/0.5149；ST-RT spatial/temporal/social/RT-intent/route-readout 0.6100/0.7015/0.4891/0.5781/0.0573。它们说明固定 checkpoint 的 actor 输出对这些同状态 forward intervention 有变化；probe 不增加 SUMO 步，也不是独立评估回合，更不能据此归因总体成功率。

## 梯度与真实参数更新

训练 representation.jsonl 在每路记录 96 条 critic_td_gradient_pre_clip 与 96 条 critic_td_parameter_update 样本。两路 state_encoder、spatial_vehicle_messages、temporal_attention、social_attention 均在 96/96 个 gradient 样本中有完整参数梯度覆盖；这些组的 96/96 update 样本均有非零 delta 和 changed elements，非有限梯度/参数变化为 0。下面列出 update 样本的平均 delta L2：

| 参数组 | ST | ST-RT |
| --- | ---: | ---: |
| state encoder | 0.003756 | 0.003712 |
| spatial vehicle messages | 0.007489 | 0.007535 |
| temporal attention | 0.012911 | 0.012811 |
| social attention | 0.007252 | 0.007359 |
| MLP readout | 0.007247 | 0.007374 |
| map/route encoder | 0 | 0.007669 |
| route path attention | 0 | 0.006362 |
| route goal attention | 0 | 0.006474 |

ST 的 route 参数组 delta 为 0，符合 use_route=false；ST-RT 的 route 组在所有 96 个保存更新样本中都有非零变化。encoder_all 汇总梯度覆盖率为 ST 41.15%、ST-RT 60.11%，低于各 active leaf group 是因为它还包括未启用的结构参数。两路 encoder 梯度非有限元素均为 0。

manifest 记录 actor 与 critic 共用 extractor 对象，但 actor 会 detach extractor features；extractor 参数由 critic optimizer 100% 持有，actor optimizer 持有 0%，无单独 representation optimizer。因此这里的梯度与更新证据明确属于 critic TD 路径，不代表 actor loss 直接更新 encoder。与之互补，前节 shadow probes 显示 actor 输出对 ST、ST-RT 分支干预敏感。优化样本是稀疏诊断，不是 95,001 次更新的逐步参数轨迹。

## 评估失败形态

下表来自各自 diagnostics/eval/summary.json 的逐 episode 行。耗时按 raw_steps × 0.1 秒汇总；速度是 episode 平均实际速度，stopped fraction 是速度采样中停驶的占比。route-ineligible 时间来自可观测 route/lane telemetry。

| 方法 / outcome | n | mean elapsed s / decisions | mean actual speed m/s | mean stopped fraction | 有 route-ineligible 的 episodes | mean ineligible s（停驶 s） | mean min logged CV-OBB TTC s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ST collision | 50 | 21.48 / 71.92 | 4.66 | 0.243 | 32 | 8.38 (7.77) | 0.0016 |
| ST success | 45 | 26.87 / 89.96 | 6.71 | 0.135 | 23 | 5.61 (5.11) | 1.326 |
| ST timeout | 5 | 60.00 / 200 | 1.43 | 0.775 | 5 | 47.22 (46.50) | 0.845 |
| ST-RT collision | 48 | 23.49 / 78.67 | 3.73 | 0.023 | 0 | 0 | 0.0023 |
| ST-RT success | 52 | 28.73 / 96.10 | 6.09 | 0.011 | 0 | 0 | 1.475 |

ST 的 50 个 collision 终态都在 route edge −E1 的 junction connector：:J1_21_0 有 41 个，:J1_14_0 有 9 个；终态 route reachability 均为 1。ST-RT 的 48 个 collision 同样位于这两处：:J1_21_0 有 40 个，:J1_14_0 有 8 个，终态 reachability 也均为 1。两路碰撞 episode 均有 logged CV-OBB TTC <1 秒；该行为遥测与 C9 closest-approach 代理量是不同指标。50 条 ST、48 条 ST-RT collision_evidence 记录均没有 SUMO collision_ids 或 collision_events，因此 lane/route 是最后可用的终端或碰撞前 ego 状态，不能声称是精确接触点或指定碰撞对象。

ST 的 5 个 timeout 都是 600 raw / 200 decisions，停驶占比 70.0%–81.8%，平均速度 1.16–2.15 m/s，route-ineligible 时间 42.6–49.8 秒且大部分时间停着。其中 2 个在 −E1 lane 0/1 终止，标记 known_lane_cannot_reach_next_edge；1 个在 −E0_2 终止，标记 route_has_no_next_edge；另 2 个终止于 :J1_14_0 与 :J1_21_0，route continuation 已知。ST-RT 无 timeout，且没有 episode 记录到 route-ineligible 时段。观察到的两路行为差异与路线接续和交叉口让行/通行时机都相关；仅凭固定策略终态无法确定因果。

对碰撞 episode，观测 ID 可用与 risk-evaluable coverage 均为 100%；对 success episode，coverage 约为 99.6%。diagnostic_error_count 和 trajectory audit errors 均为 0。coverage 是日志可计算性，不代表策略看见了全部真实冲突；摘要仍记录 critical_unobserved_ticks。主要失败残留集中在路口连接器上的 collision，而 ST 的 timeout 另有明显长时间低速/停滞与少量错误路线可达性状态。这使“入路口时机、冲突判断和及时清空”成为后续优先诊断假设，但不是本次 audit 直接证明的机制。

## 证据文件与边界

主要原始来源（均在两个方法目录下）：

- diagnostics/train 与 diagnostics/eval/trajectory_history_audit.summary.json、trajectory_history_audit.jsonl：每 phase 的真实决策、raw 计数、历史 mask 反事实、cache age 与几何汇总。
- diagnostics/train 与 diagnostics/eval/summary.json：诊断 episode 结果、route/lane、速度、风险覆盖、终态信息和 collision evidence。
- diagnostics/train/representation.jsonl、optimization.jsonl、manifest.json：采样梯度、critic TD 参数更新、优化器所有权及方法开关。
- diagnostics/train 与 diagnostics/eval/policy_shadow_probes_summary.json、policy_shadow_probes.jsonl：唯一状态预算、适用/无效行及同状态 action deltas。
- evaluation_results.json、training_complete.json、train_monitor.csv：最终评估身份与回报、raw/update 预算、训练完成 episode。
- 采集 schema 与阶段限制见 analysis/d1_contractfix_protocol_20261003.md；padding 分类和 age 计算见 contractfix_trajectory_audit.py。

本证据支持两路均越过 warmup、完成 95,001 次更新，ST/ST-RT spatial-temporal 分支得到 critic 梯度并发生实际参数变化；ST-RT route 参数组也被优化，shadow actor action 对相关分支敏感。它排除了“分支完全未执行/完全没有优化”的简单解释。它不证明新方法相对旧实现的性能因果，也不证明 joint repair 中 C8 或 C9 单独有效。C8 的 x!=0 presence proxy、C9 pair-history 重复采样与碰撞状态缺少 SUMO 接触对象信息，是本诊断的主要边界。

