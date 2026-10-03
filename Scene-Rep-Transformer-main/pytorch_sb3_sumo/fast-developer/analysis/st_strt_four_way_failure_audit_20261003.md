# 旧/修复 ST、ST-RT：四组失败分布

日期：2026-10-03。仅分析已保存的最终100回合评估，没有新增仿真或训练，automation-3继续暂停。

## 修复范围

“修复ST”是sac_mlp_d1_st_contractfix_v1，“修复ST-RT”是sac_mlp_d1_st_rt_contractfix_v1。两者均联合修复：

- C8：真实最后一个有效历史帧索引与全空处理，覆盖route query、temporal query、ego旋转锚点及相关重复选择逻辑。
- C9：仅显式几何边将旧速度通道恢复为与位置一致的Cartesian速度(-old_vy, old_vx)。ST本身也有几何边，故ST也包含C9修复。

旧输入通道、奖励、回放、优化器及既有单gamma/timeout bootstrap协议不因“contractfix”而改变。这里不是C8-only或C9-only实验，不能拆分两项修复的性能贡献。

## 共同口径与证据边界

四组主场景均intersection_sorted_depart4p0，训练seed0、从零100k；最终validation为10000–10099的100个seed。旧ST-RT采用sortlr_1003_retry01的正式最终模型，非失败候选或有限窗口诊断。

下文将互斥终局类型和可重叠的历史诊断分开。终局碰撞并不提供唯一的物理/决策根因；同seed但不同动作后的轨迹也不是同状态因果干预。

旧ST的evaluation_results仅保存episode、seed、return、decision/raw/environment steps、成功完成时间、终局标志和traffic variant。没有该checkpoint的final100速度、road/lane、路线可达性、TTC或接触对象诊断。缺失项不能填零或由其他方法外推。

## 1. 失败终局

| 方法 | 成功/100 | 失败/100 | 碰撞 | 超时 | off-route终局 | 失败中碰撞比例 |
|---|---:|---:|---:|---:|---:|---:|
| 旧ST | 50 | 50 | 50 | 0 | 0 | 100% |
| 修复ST | 45 | 55 | 50 | 5 | 0 | 90.9% |
| 旧ST-RT | 63 | 37 | 37 | 0 | 0 | 100% |
| 修复ST-RT | 52 | 48 | 48 | 0 | 0 | 100% |

修复ST相对旧ST新增5个超时，碰撞总数仍50；修复ST-RT相对旧ST-RT新增11个碰撞，没有超时。不能由相同碰撞总数推断是同一批案例。

## 2. 失败位置和路线状态

下表的连接器位置是终止诊断中的最后可用ego road/lane，不是确认的impact坐标。

| 方法 | 碰撞最后可用lane :J1_21_0 / :J1_14_0 | 碰撞终态路线可达 | 失败中曾有已知当前车道不可接续路线旗标 |
|---|---|---|---|
| 旧ST | 未采集 | 未采集 | 未采集 |
| 修复ST | 41 / 9（50/50覆盖） | 50/50 | 37/55：32个碰撞、5个超时 |
| 旧ST-RT | 29 / 8（37/37覆盖） | 37/37 | 0/37 |
| 修复ST-RT | 40 / 8（48/48覆盖） | 48/48 | 0/48 |

三个有诊断的方法，碰撞终态的规划路线接续均可达。可以说碰撞判定时最后可用位置集中在路口连接器，不能仅凭ego位置说都是侧向碰撞、追尾、进入过早或出口清空失败。

route_lane_ineligible只在路线上下文已知、存在planned_next_edge、可达值为布尔且ego位姿为current时计数；False才计入ineligible。未知上下文、无后继边进入unknown统计，不混入此计数。

修复ST的23/45个成功episode也出现过该旗标，因此它不是失败的充分条件；37/55表示历史现象覆盖，不是37个失败已被因果归因为路线错误。修复ST-RT的100个episode均未出现该历史旗标。

修复ST碰撞组的历史ineligible均值8.38秒（其中停驶7.77秒）；超时组47.22秒（其中停驶46.50秒）。这支持超时与长时间路线不可接续/停驶并存，但不说明每次碰撞由该状态造成。

## 3. 碰撞回合的行为统计

先在每个episode内统计，再对该方法的碰撞episode等权平均。速度包含停驶样本；停驶阈值为speed<0.1m/s。下表不是碰撞瞬间速度，也不是跨所有raw tick合并加权。

| 方法 | 碰撞回合到终止的平均控制时长 | 回合平均实际速度的均值 | 回合停驶样本占比的均值 |
|---|---:|---:|---:|
| 旧ST | 22.41s | 未采集 | 未采集 |
| 修复ST | 21.48s | 4.66m/s | 24.3% |
| 旧ST-RT | 23.78s | 3.62m/s | 14.59% |
| 修复ST-RT | 23.49s | 3.73m/s | 2.3% |

时长用raw_steps×0.1s得到，不包含reset/warmup。旧ST碰撞组原始raw均值224.12、decision均值74.94；completion_time字段对失败为null，不能将成功完成时间填入碰撞组。

旧ST-RT的35/37个碰撞episode至少有一个停驶样本，但没有“为何等待”的标签，不能将停驶自动解释为主动让行。其合并样本停驶比例1394/8800=15.84%，与表中episode等权14.59%不同，不可混写。

修复ST-RT碰撞回合的停驶比例低于旧ST-RT，但两组碰撞案例集合不同；这不是激进程度或降速导致碰撞的因果比较。

## 4. 修复ST的5个超时

均600raw/200decisions，即60秒；停驶42.0–49.1秒。

| seed | 最后road/lane | 最后路线状态 | 回合平均速度 | 停驶占比/秒数 | 历史ineligible秒数 |
|---|---|---|---:|---:|---:|
| 10020 | :J1_14 / :J1_14_0 | known_route_continuation，可达-E0 | 1.239m/s | 79.3% / 47.6s | 48.9s |
| 10031 | -E1 / -E1_1 | known_lane_cannot_reach_next_edge，不可接续-E0 | 1.194m/s | 81.8% / 49.1s | 49.8s |
| 10043 | -E1 / -E1_0 | known_lane_cannot_reach_next_edge，不可接续-E0 | 1.162m/s | 81.0% / 48.6s | 48.9s |
| 10076 | -E0 / -E0_2 | route_has_no_next_edge | 2.146m/s | 70.0% / 42.0s | 42.6s |
| 10083 | :J1_21 / :J1_21_0 | known_route_continuation，可达-E0 | 1.414m/s | 75.3% / 45.2s | 45.9s |

10076的“路线无下一边”不等于终态车道错误；其42.6秒ineligible来自更早的已知路线接续时段。另两个在连接器终止的超时，终态可达不代表此前没有阻滞。最可靠的共同描述是长期停驶后耗尽预算，而不是“五个都选错车道”。

## 5. 能够和不能够下的判断

- 旧ST：失败均为碰撞，细分机制在旧final100日志中不可识别。静态C8风险不能冒充这50次碰撞的实测原因。
- 修复ST：碰撞与长时间停驶超时并存；许多失败有路线不可接续历史，但成功中也出现。需要将路线恢复能力和路口冲突处理分开讨论。
- 旧ST-RT：这批失败没有路线不可接续旗标或超时，碰撞末次位置全在连接器；主要剩余失败表现为路口中的碰撞判定。
- 修复ST-RT：失败形态与旧ST-RT相近，但数量由37增至48；同样不能解释为超时或终态路线不可达。碰撞回合停驶更少，但不证明更激进就是原因。

正常日志的恒速/固定航向CV-OBB TTC是预测代理，不是真实碰撞TTC；包含临近终止的episode最小值也不能判定最初哪个决策错误。

三个有诊断的方法，其SUMO collision_ids/events在这些碰撞记录中为空，但并非完全没有候选对象：旧ST-RT有37/37、修复ST有50/50、修复ST-RT有48/48个geometric_obb_fallback首命中候选，候选物理状态与ego/input为同一tick。该来源记录环境几何检测器的首个OBB overlap，不是仅按最近距离猜出的车辆；也不提供完整重叠对象列表、接触法线或SUMO确认的物理接触身份。

修复两组候选车辆的road/lane来自最近一次路线/车道上下文，距事件1–3 raw ticks；位置、速度、heading则为同tick。已有same_lane=false对应的是采样lane ID，不足以据此认定事件时刻的侧碰、追尾或接触方向。旧ST-RT候选保存的road分布为:J1_16共20例、:J1_1共17例，也不作为精确impact road。没有现成的相对航向/接触类别标签，本次未另造分类器。

因此现有日志足以描述几何首命中候选和路线背景，不能给出可靠的追尾/侧碰比例。对当前项目碰撞判据的结果统计仍然有效，缺少SUMO事件不等于没有发生项目定义的碰撞。

## 来源

- 旧ST：fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json及training_complete.json。
- 旧ST-RT：runs/sortlr_1003_retry01/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json及diagnostics/eval/episodes.jsonl、summary.json。
- 修复ST：runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0对应上述评估文件。
- 修复ST-RT：runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0对应上述评估文件。
- 采集定义：[behavior_diagnostics.py](../behavior_diagnostics.py)，尤其route eligibility采集规则和速度/停驶聚合。
- [修复协议](d1_contractfix_protocol_20261003.md)
- [两修复方法诊断审计](d1_contractfix_diagnostic_audit_20261003.md)
- [四组配对比较及旧ST字段缺失](d1_contractfix_comparison_evidence_20261003.md)
- [上一轮综合归因报告](d1_contractfix_results_attribution_20261003.md)
