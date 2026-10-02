# 旧场景 intersection_sorted / depart_scale=4.0：ST → ST-RT → Topo → 3slot 结果与归因

记录日期：2026-09-30 至 2026-10-01。依据完成实验的原始结果、配置、诊断流、模型文件哈希和与运行快照匹配的核心源码；基础检索与读取由 Luna max 完成。本次只分析和更新研究记录，没有重新训练、重跑 SUMO 或改变方法实现。

**场景边界：本报告所有实验结果均属于之前的 intersection_sorted 场景、depart_scale=4.0；不是当前 intersection_random_darrl_low_v1 / medium_v1 / high_v1，也不是此前新增的随机车流 p 系列场景。不能将这里的百分比当作新场景结果，或直接用来排列新场景中的方法表现。本文优化假设来自旧场景；转到新场景后必须重新建立相同配置下的共同对照。**

## 1. 首要结论

在 intersection_sorted、depart_scale=4.0、单训练种子 0、从零连续 100,000 raw steps、95,001 updates、final_model、确定性 100 回合评估这一组记录中：

| 方法（均为旧 intersection_sorted，depart_scale=4.0） | 成功率 ↑ | 碰撞率 ↓ | 超时率 ↓ |
|---|---:|---:|---:|
| sac_mlp_d1_st | 50% | 50% | 0% |
| sac_mlp_d1_st_rt | 63% | 37% | 0% |
| sac_mlp_d1_st_rt_topo | 39% | 39% | 22% |
| sac_mlp_d1_st_rt_topo_3slot | 41% | 56% | 3% |

评估种子为 10000–10099；逐种子核对了 traffic_variant，ST-RT/Topo 的有效路线 XML 哈希 30/30 相同，训练/评估使用同一个完整车流池，没有独立 holdout。这里的单种子结论适用于这组已训练策略，不能视为跨训练种子的架构优劣定论。D0929 ST-RT 未留 source_snapshot，不能从归档证明它与后来的 Topo 运行逐字节相同的全部源码；匹配配置和车流是较强对照，但仍不是完整重建的受控运行。

- RT 相对 ST：成功增加 13 个百分点。
- Topo 相对 ST-RT：成功下降 24 个百分点，主要伴随新增 22% 超时；碰撞净增加 2 个百分点。
- 3slot 相对 Topo：成功只增加 2 个百分点，碰撞增加 17 个百分点，超时减少 19 个百分点。不能描述成“成功率下降”，也不能视为有效安全增益。
- 因而问题首先出现在 Topo，随后三槽替换读出改变了失败类型。当前不宜直接继续叠加 full 的其他模块。

原始入口：
- ST：fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json。
- ST-RT：仓库根 runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json。
- Topo 与 3slot：仓库根 runs/t0930_topo3_retry01 下各自同名方法目录。
- 配对、行为与终端标记审计：本目录 topo3_retry01_results_audit.json；完整同场景补充由同场景审计文件记录。
- 优化、激活与梯度审计：本目录 topo3_retry01_optimization_audit.json，含字段语义、采样窗口、source hash 和缺失项。

注意：fast-developer 原位置的 ST-RT 是旧 50k 记录，不能把它误当本表中的 fresh 100k。

## 2. 配对结果揭示了什么

ST-RT → Topo（行是 ST-RT，列是 Topo，单位：回合）：

| | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 成功 | 23 | 23 | 17 |
| 碰撞 | 16 | 16 | 5 |

Topo 虽把 16 次原碰撞变成成功，却让 40 次原成功变为碰撞或超时。净碰撞率近似不变，掩盖了很大的逐场景行为改变。

Topo → 3slot：

| | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 成功 | 20 | 17 | 2 |
| 碰撞 | 15 | 24 | 0 |
| 超时 | 6 | 15 | 1 |

原有 22 次超时只有 6 次变成成功，15 次变成碰撞。称其为“解决超时”是不完整的；更准确的是错误类型从卡死转向了碰撞。配对种子并不意味着两个策略经过同一条状态轨迹。

## 3. Topo 的主要失败近因：进入无法完成路线的车道

这部分有轨迹摘要与路网连接的联合证据，强于仅看注意力或梯度推测原因。

22 个 Topo 超时回合全部：
- 终点 road=-E1、lane=-E1_0、lane_position=70 m、route_index=0；
- 终点实际速度为 0，但目标速度平均 5.020 m/s，范围 4.250–6.491 m/s；
- 终点换道命令为 -1 的有 16 个，为 0 的有 6 个；
- 本次 lane_change_applied=false，共 22/22；
- 全回合约 84% 的采样处于停驶，平均速度约 1.167 m/s。

路网与控制语义：
- 场景配置的 ego route 是 -E1 → -E0，departLane=2，arrivalLane=1；paper_env 把该 ego XML 加入 SUMO route-files，动态背景车重插排除了 ego，未见覆盖该始发配置的代码。始发 lane=2 来自生效配置核验，终点 lane=0 则来自实际回合日志；两者证据来源不同；
- -E1 到 -E0 的唯一连接是 fromLane=2 → toLane=1；
- -E1 的 lane 0、lane 1 没有到该目标出口的连接，进口车道长 70 m；
- 此旧场景使用 smarts action contract，命令 -1 表示 lane index 减 1；从 lane 2 向 lane 1/0 右移；
- 在 lane 0 再请求 -1 越界，不会发出有效换道；命令 0 按定义保持。

因此，“请求前进，但停在错误出口车道末端”是这批超时的明确近因。它不是单纯的低目标速度或因 TTC 太低而主动停车。SUMO 官方文档也说明，没有连接到路线下一条 edge 的车道会导致车辆减速停车：[VehicleSpeed](https://sumo.dlr.de/docs/Simulation/VehicleSpeed.html)。

lane_change_applied 仅表示本次本地检查及 TraCI 请求调用情况，不是物理换道完成回执；不能用最后一步 false 推断整回合从未换道。源码与 XML 的精确定位、历史源码限制见实现审计。

### 为什么这与 Topo 有关，但还不能断言唯一根因

当前 Topo 的 compatibility 主要是“全局节点有效且 heading_cosine≥0”，结合路线几何偏置、top-8 和 fallback。它没有明确检查“该 lane 是否能连接到 ego 计划路线的下一条 edge”。

同向平行车道可能都通过该筛选，却具有不同的出口可达性。这与观测到的错误车道卡死形成了具体、可证伪的机制假设：几何/朝向相关的 Topo 信息可能压过了原 RT 中用于保持正确转向车道的信息。

但是，现有日志没有逐输入的干预实验，不能证明是哪条 Topo 支路导致向右动作，更不能声称所有 Topo 都有此缺陷。连通关系部分存在于图中，模型也可能隐式学习；“没有显式路线可达性条件”不等于“模型完全没有这些信息”。

## 4. Topo 确实参与计算，但存在需要验证的耦合

本轮核心编码器、SAC、policy、诊断模块 SHA 与运行 source_snapshot 匹配。Topo 前向包含三个位置：
1. 将拓扑上下文以可训练 scalar gate 加到 actor 状态特征后再进入时空处理；
2. 将 same-lane 和 conflict-or-merge 关系分数加入车辆图的边特征；
3. 对 ego route readout 增加 Topo goal residual。

前两个位置共用 topology_residual_scale。这把“状态特征的扰动强度”和“车辆关系线索的尺度”绑在一起。

Topo 训练诊断中位数（5–20k / 20–90k / 90–100k raw steps）：
- topology gate：0.04239 / 0.002230 / 0.001560；
- goal gate：0.03617 / 0.003727 / 0.002222；
- 最后 10k 门控前 conflict relation-pair score 约 0.1253（字段名 conflict_attention_pair_mean，由 lane attention 分布与 conflict adjacency 计算，不是 policy 的注意力权重）；
- 乘 gate 后的 conflict-or-merge edge feature 约 0.000199。
- 最终评估常数 gate：topology=0.002272，goal=0.002729。

这些 gate 曾明显增大，随后回缩，不能说“因为初值小所以从未学起来”。有效边特征尺度较小是事实，但没有融合前后差值范数和同输入干预时，不能把 gate 大小等同于对动作的影响。

96 条采样的 critic TD 梯度记录中，topology_core、goal_fusion、vehicle_relations 参数梯度覆盖均为 100%。可排除“采样时完全断梯度”的解释，不能由此证明它们学到了有用关系。

route_compatible_attention_mass=1 主要由硬 mask 保证，不代表模型学会了正确的交通路线。这里的 compatible 也不是路权或冲突避让正确性指标。

另一个已排除的简单解释：关闭 Topo 时，主分支仍经过相同 topology_norm；goal 分支为
`r + LN(r + g_goal*t) - LN(r)`。
当 goal gate=0 时精确回到 r，不存在仅因新增一层归一化就必然破坏该路径的问题。

## 5. 3slot 究竟改变了什么

两路 Topo 运行都走同一个 incremental forward。ego、social、route 三个上下文均为 128 维；区别主要在最终读出。

Topo：
`z = MLP_384→128→128([e,s,r])`，两层 ReLU，活跃读出参数 65,792。

Topo+3slot：
`z = [P_e e, P_s s, P_r r]`，
投影为 128→32、128→64、128→32，三个线性层，输出合计仍为 128 维，活跃读出参数 16,512。

因此这个开关同时改变了：
- 联合非线性读出 → 各分支线性压缩；
- 信息保留容量的自由分配 → 固定 32/64/32 配额；
- 活跃读出参数减少约 74.9%。
- 输出形式从末层 ReLU 的非负表示变为可有正负值的线性投影表示；这也改变了 actor/critic 接收的特征参数化，不能只解释为多了槽标签。

两个读出模块都注册在模型中，不能只看注册总参数量判断公平性。

### 有理论依据的风险，及其适用边界

对固定 64×128 的 social 投影矩阵，零空间维数至少为 64。若固定上游上下文的一次变化落在该零空间，压缩后的 social slot 无法区分这次变化，下游网络也无法恢复它。ego/route 各有至少 96 维零空间。这个结论是线性代数事实。

原来的联合非线性读出也有 128 维瓶颈；区别是它能先结合三种上下文作非线性选择，再压缩，而当前读出先按固定配额分别压缩。不能把它夸大成“Topo 没有瓶颈、3slot 必然丢掉危险信息”。上游可训练表征可能主动把重要信息移入保留子空间，实际是否丢了安全信息仍需实验验证。

三个上下文也不是严格解耦的语义域：ego 特征已经融合 route/Topo/时空信息；social attention 用 ego 作 query，且 keys 包含 ego；route readout 同样有 Topo goal 信息。因此“把三个输出命名为槽”不自动保证语义独立。这也不是具有竞争式对象绑定过程的 [Slot Attention](https://arxiv.org/abs/2006.15055)。

本轮 Graph-SLT、SBS 关闭，辅助损失系数为 0；encoder 由 critic TD loss 更新，actor 对共享特征 detach。三槽没有额外语义分离监督。三组投影在全部 96 次采样中都有梯度，因此不是没训练到。

三槽的最终评估碰撞回合平均速度约 3.803 m/s，Topo 约 7.357 m/s；成功回合约 5.971 vs 8.644 m/s。这些速度按对应 raw speed samples 加权（sum(speed_sum)/sum(speed_samples)），不是回合等权平均。这不支持“3slot 一律更快所以撞得更多”。总体低 TTC 比例增加，但两策略访问状态分布不同，不能把该统计直接当成致因。

更换读出还通过 TD 梯度影响了上游训练：3slot 最终 topology/goal gate 为 0.002761/0.000143，而 Topo 为 0.002272/0.002729。这里缩小的是额外的 Topo goal 修正门，不是整个 route 分支；原始 route readout 仍存在。这个差异说明不能把两个训练完成模型理解为“固定同一个主干，只在推理时切换头”；它本身仍不能证明负交互的因果方向。

## 6. 奖励、优化、实现错误：现有证据能排到哪里

- 终端统计：两路 Topo 的训练和评估中，已保存的 success/collision 标记交集均为 0；不能把此次对比归因于后来发现的 DARRL 双标问题。旧运行没保存 raw_sumo_arrived/raw_sumo_collision，不能反推所有底层原始事件。
- 优化：最后 10k critic loss 中位数 Topo≈6.61、3slot≈5.90，稀疏 |TD|≈1.49 vs 1.02；alpha≈0.0204 vs 0.0212。没有出现支持“3slot 更差是因为 critic loss 更大”的证据。两个策略的数据分布不同，较小 TD loss 也不等于更准确的安全价值估计。
- 训练后期：最后 100 个训练回合 Topo 成功 43%、3slot 成功 32%；最终确定性评估为 39%/41%。训练是随机策略，不可用训练窗口当作固定评估，也不能以最佳训练窗口替代 final_model。
- 奖励与超参数：两路 Topo 的 reward、lr=1e-4、batch=32、buffer=20k、gamma=.99、5k learning starts 等协议相同。没有证据说某个超参数一定错误，也不能因超时降低就断言奖励“偏爱碰撞”。SAC 优化的是含熵项的回报目标，与成功率/碰撞率不是同一个目标：[SAC 原论文](https://proceedings.mlr.press/v80/haarnoja18b.html)。
- 单训练种子、非相同状态轨迹、部分历史 trainer/environment 源码只留哈希而无冻结副本，限制架构因果归因。当前源代码不能一概冒充旧运行源码。
- 旧碰撞 evidence 主要是 ego 终点快照，不能据此断言精确碰撞对手、撞击点或所有低速碰撞的具体类型。
- 未保存可直接复用的 SAC replay buffer/完整 policy observation 数据集；本次没有伪造同输入消融结果。

## 7. 两个优先优化方案

### 方案一：优先修正 Topo 的路线条件，针对错误车道卡死

从 ST-RT 作为可靠参照，做一个明确的 Topo-route-aware 变体：

1. 用路网 lane connection（含内部连接链）与 ego 规划的下一条 route edge，计算每个候选 lane 的路线出口可达性。区分“当前可直接继续路线”与“需先换道才能继续”。
2. 先只修改 ego goal-topology 的候选筛选/偏置，让其查询计划路线可用的车道；actor 当前车道定位与 social/conflict 查询继续保留真实占用车道和冲突车流，不能把错道 ego 或横向冲突车过滤掉。
3. 不按 lane 编号硬编码正确动作，不让 SUMO 自动纠正策略，不改变 reward、出车分布、动作阈值或评估口径。保持其余 Topo 通路与联合 MLP 读出不变。
4. 在候选集合为空时显式记录并执行有定义的 fallback，不能悄悄把“最近但不通目标出口的车道”继续标为路线合法。区分“已知不可达”和“图裁剪/节点映射缺失导致未知”，不把未知一律判成不可达。

接口范围：现有 ego_topology_valid 直接取 ego 的最后一次 topology query mask，并不是独立的 route-legality mask。图中有 SUCCESSOR 关系，但 encoder 当前持有的几何/属性/边张量没有 lane-ID 到计划 next-edge 的对齐接口。因此实现该方案需要在环境/图构建侧保留节点映射，并传入与 ego 已知规划路线对齐的 mask 或特征；不是只改一行 heading 判断。只能使用静态地图和自车已知计划，不能使用未来交通轨迹或结果标签。它引入显式地图语义，应在消融中如实报告输入变化。

依据：已证实的 lane 出口不连通、正目标速度下卡死、当前 compatibility 只约束几何/朝向。理论依据是有向道路图的可达性约束和 SUMO 的路线转移规则，而不是单凭注意力图猜测。

验证时正常采集：当前 lane 是否支持下一 route edge、指向不可达 lane 的命令比例、实际 lane 转移、目标/实际速度差、进入错误车道至超时的时间、goal attention 对可达 lane 的质量、成功/碰撞/超时及逐种子转换。

预期：错误车道卡死显著减少，成功增加且碰撞不增加。若只把 22% 超时变成碰撞，仍不能判定方案成功；若路线行为没有改变，应转查哪条 Topo 支路驱动动作，而不是继续加模块。

共享 gate 对状态融合和关系 cue 的耦合是下一层候选问题。可通过将同一 gate 拆成两个、复制相同初值且保持初始前向相同，单独检验梯度耦合；本轮不建议同时改 mask、gate、reward 和学习率。

### 方案二：保留三槽划分，恢复容量接近的非线性读出

在确认可用的同一主干上，把各槽投影改为：
- ego：128→128→32；
- social：128→128→64；
- route：128→128→32；
- 两个线性层后均放 ReLU，与原 joint MLP 的激活层数及非负输出形式对齐，最后按 32/64/32 拼为 128 维；其余网络和损失保持不变。

三组头合计 66,048 个活跃参数，对照 joint MLP 的 65,792 仅多约 0.39%。它能更直接检验当前退化是否与“独立、窄、纯线性读出”有关，避免把容量削减混同于三槽思想。

这一方案没有保证彻底消除压缩瓶颈，也没有新加跨槽交互；交互仍由上游和后续 actor/critic 网络承担。它的理论动机是用输入相关的非线性特征选择替代固定线性压缩，实际收益必须观察。

验证解释：
- 若该版本恢复安全表现：支持当前读出参数化/容量是问题的一部分，尚不能单独区分容量和非线性的贡献；
- 若仍明显差于 joint MLP：再优先检验固定维度配额和联合读出必要性；
- 不用增加无效、不参与前向的参数来“配平”容量。

如更看重保留原函数而不是容量控制，可另行采用 `z=z_joint+alpha*z_slots` 的残差替代，alpha=0 初始化、slot 分支正常初始化。该方式在初始时包含原基线，依据 [ReZero](https://arxiv.org/abs/2003.04887) 与 RL 中的 [GTrXL](https://arxiv.org/abs/1910.06764) 的保留通路设计；但它增加容量，需要额外参数匹配对照，不能与上面的主方案混为一次单变量消融。不要同时把 alpha 与整个分支输出初始化为零，以免相互阻断学习。

## 8. 实验顺序与可证伪标准

保持用户指定的单训练种子阶段，不用多种子要求阻挡下一步：

1. 优先让两路 worker 比较同一冻结运行环境下的 ST-RT 与 Topo-route-aware；同场景、同从零初始化协议、同 100k raw steps、同 final checkpoint 与评估 seeds。此处只是建议，没有启动。
2. Topo 通过“成功不降、错误车道超时减少、碰撞不升”的检查后，再比较同一 Topo 主干的 joint MLP 与非线性 3slot。
3. 想严格判断 Topo×3slot 的交互，还需要 ST-RT+3slot 这一缺失格：
   `interaction=(Topo+slots−Topo)−(STRT+slots−STRT)`。
   目前两个 Topo 分支不足以证明三槽本身无效，或证明两模块必然负交互。
4. 所有新实验应继续保存正常训练/评估中可得到的 lane/route 可达性、命令执行、表示/梯度和结果指标。额外干预探针仅在这些信息仍不能区分解释时再做。
5. 不把当前其他 DARRL 场景运行的结果并入本组消融，不同时换场景又改 Topo/读出。本报告没有干预正在运行的实验。

结论等级：错误出口车道导致本组 Topo 超时，有直接数据与路网支持；Topo 路线条件不足、共享 gate 耦合和三槽读出瓶颈，是有具体实现与理论依据的待验证解释；尚无证据把它们各自的因果贡献定量分解，也没有证据否定三槽思想或 Topo 的一般有效性。

## 9. 其余同场景 ST 家族记录：按预算分开比较

以下均为 intersection_sorted/depart4；S/C/T 的单位为百分比，各 summary 为 100 回合评估。它们不是上面 fresh 100k 主表的同预算单因素对照。

| 方法后缀（sac_mlp_d1_） | 训练状态 | updates | S/C/T |
|---|---|---:|---|
| st_rt | fresh 50k | 45,001 | 31/67/2 |
| st_attn | fresh 50k | 45,001 | 43/56/1 |
| st_rt_edge | fresh 50k | 45,001 | 39/61/0 |
| st_rt_ego | fresh 50k | 45,001 | 44/56/0 |
| st_rt_gate | fresh 50k | 45,001 | 40/60/0 |
| st_rt_late | fresh 50k | 45,001 | 46/54/0 |
| st_rt_edge | 50k checkpoint 续训至 100k | 94,502 | 25/68/7 |
| st_rt_ego | 50k checkpoint 续训至 100k | 94,502 | 36/62/2 |
| st_rt_gate | 50k checkpoint 续训至 100k | 94,502 | 24/76/0 |
| st_rt_late | 50k checkpoint 续训至 100k | 94,502 | 29/71/0 |

旧 50k st_rt、st_attn、gate、late 缺逐回合 episode_records，不能执行本报告同样的逐种子配对。edge/ego 50k 及四个续训分支保留了 100 条、seed10000–10099 的记录。续训分支 updates=94,502，也不同于 fresh100k 的95,001；必须保留断点/续训条件，而不能仅凭名字中的100000把它们当连续从零训练。

背景参照：同场景 fresh100k 的纯 SAC+MLP 为22/38/40，MST+SLT为53/47/0。但 MST+SLT 使用 legacy YV2、frozen_80_20，与 D1 的 base 协议不同；它仍是完整方法的强基线目标，当前数字不能当作严格公平的 D1 单模块比较。所有数值与源路径已写入 topo3_retry01_results_audit.json。
