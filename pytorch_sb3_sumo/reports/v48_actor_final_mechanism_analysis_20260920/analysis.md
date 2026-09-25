# v4.8/lr_half：最终 actor 历史结果的机制分析

日期：2026-09-20。范围：Cross、CARLA；训练 seed 0；最终训练模型；直接 actor 确定性评估。不使用历史检查点选择、fusion、target-critic-only 或当前新调参轮次的结果。本报告没有启动新的训练或评估。

**核心判断：已有数据证明完整方法在两个场景都有收益，并证明联合修改回报长度与 bootstrap 折扣在 Cross 上有很大的条件效果；但尚未证明 Graph-SLT 的独立控制收益，也没有隔离“16 步”本身的收益。当前不应把“CARLA 表征瓶颈、Cross 信用分配瓶颈”写成已确证的机理。它是可以检验的解释。**

## 1. 已核实的历史结果

下表均为 exact-final checkpoint、直接 actor、相同的 100 个评估种子（10000–10099）。MST 的 native/native_deterministic 路径是普通 actor 的 deterministic forward，不使用额外部署机制。四类实验都只训练了 seed 0。

| 方法 | Cross 成功率 | 碰撞率 | 超时率 | CARLA 成功率 | 碰撞率 | 超时率 |
|---|---:|---:|---:|---:|---:|---:|
| MST+SLT | 0.77 | 0.23 | 0.00 | 0.83 | 0.07 | 0.10 |
| v4.8/lr_half 完整方法 | 0.87 | 0.11 | 0.02 | 0.99 | 0.01 | 0.00 |
| minus_horizon | 0.37 | 0.63 | 0.00 | 0.98 | 0.02 | 0.00 |
| minus_factorized | 0.75 | 0.23 | 0.02 | 0.91 | 0.00 | 0.09 |

来源：
- 完整方法 Cross：`r3m1/eval/d3d3bda5838620092eca/result.json`。
- 完整方法 CARLA：`r3m1/eval/8c12f2558b7839e07eeb/result.json`。
- MST Cross：`r3m1/eval/8e264cd9dd8365e40002/result.json`。
- MST CARLA：`r3m1/eval/162c56fe7ba74b94199f/result.json`。
- 消融：`results_subtractive_ablation/screen/minus_horizon__{cross,carla}__lr_half__seed0/evaluation.json` 与对应的 `minus_factorized` 文件。

full/minus_horizon 的评估 seed 与 traffic_variant 列表一致，检查点均来自 50000 个 raw steps、45001 次更新的训练。full Cross 模型 SHA-256 为 `6323b5e06d53793f5ff559b23fd6657ba134d9d58e0ebb2e1fa6e2b7668acb08`；CARLA 为 `4dc9a4907ee912039d17feee4607eb471e39cbd9a08948041fd23917570db2e2`。probe 使用的也是这两个最终模型。

配对结果比两个百分比更有说明力：

- Cross full 对 minus_horizon：53 个种子仅 full 成功，3 个仅 minus 成功，34 个都成功，10 个都失败。碰撞率从 0.11 到 0.63，为 5.73 倍。这是当前固定训练实例上很大的效果。
- CARLA：2 个种子仅 full 成功，1 个仅 minus 成功，97 个都成功。净差只是 1 个 episode，不支持“无贡献”的等价性结论，也不能精确估计一个真实的 1 个百分点收益。
- minus_factorized 在 CARLA 损失 8 个百分点，主要转成超时；在 Cross 损失 12 个百分点，主要转成碰撞。CARLA 的高成功率不能用来说明所有组件都不重要。

完整方法相对现有 MST 基线在 Cross 为 +10 个百分点，在 CARLA 为 +16 个百分点。因此，“两个机制各在一个场景有效，所以组合无法改善两个场景”既不符合已有完整方法结果，也不是成立的逻辑推论。但整套方法有多个同时变化的组件，这些差异尚不能分摊到 Graph-SLT 或 n=16。

## 2. “表征质量”的证据边界

probe 结果是固定数据上的线性可读性，不是表征全部信息的测量，更不是组件对驾驶性能的因果效应。

| 指标 | Cross MST → v4.8 | CARLA MST → v4.8 |
|---|---:|---:|
| 下一决策自车动力学 R² | 0.7675 → 0.7382 | 0.8108 → 0.8849 |
| 当前最小几何距离 R² | 0.9458 → 0.9431 | 0.8574 → 0.9477 |
| 当前恒速 TTC R² | 0.6479 → 0.6741 | 0.4685 → 0.6365 |
| route edge index 分类准确率 | 0.9965 → 0.9988 | 0.9637 → 0.9864 |

Cross 不能描述为所有维度都持平：动力学变差，TTC 变好，距离接近，route 标签接近满分。三个回归 R² 的简单平均恰好接近，不代表统计等价或所有能力相同。

数据协议：
- 行为策略为 MST+SLT 最终模型的确定性 actor；所有 encoder 使用同一批 observation。
- 每场景 40 episodes；Cross 6017 transitions，CARLA 2182 transitions。
- 按 episode 分割，seed 73，28 个训练 episode、12 个测试 episode。Cross 4297/1720 transitions，CARLA 1521/661。
- 标准化仅使用训练统计；ridge=0.001。没有重复划分或置信区间，不能将大量相关 transitions 当作独立训练种子证据。
- route 标签是 SUMO route edge index，不是路线意图、分支决策或路径规划能力。
- TTC 是恒速外推，截断/缺省上限 20 秒；该上限比例 Cross 28.09%、CARLA 8.34%。它不是实际未来碰撞概率。
- 两场景均未发现 distance=80 的缺失哨兵值，不能用“距离标签大面积缺失”解释结果。

相关原始结果：`r3m1/representation_probe/cross/probe_result.json`、`carla/probe_result.json`；提取代码 `tools/run_latent_probes.py`。

线性 probe 文献支持的是“某类信息可否从 frozen representation 中线性读出”，不能证明 actor 使用了这些信息，也不能证明线性读不出的信息不存在。[Alain & Bengio](https://arxiv.org/abs/1610.01644)

从控制理论角度，状态表征需要保留与 reward 和 action-conditioned transitions 有关的信息。bisimulation 为这种要求提供形式化框架，但当前 Graph-SLT 并未证明满足其条件。因此本报告不将其他论文的价值误差保证直接移植给 Graph-SLT。[Zhang et al., ICLR 2021](https://arxiv.org/html/2006.10742)

## 3. 代码决定了 Graph-SLT 能保证什么、不能保证什么

实际 Graph-SLT 将 online latent 分为 ego/social/route = 32/64/32。每个 slot 用残差形式 `z_slot + MLP([z_all, action])` 预测下一决策 observation 的 target encoder slot，损失是三个 slot cosine loss 的平均。

它的 target：
- 是下一决策状态的 latent，而非真实动力学数值标签；
- 由 encoder 自身产生，并停止梯度；
- 不包含明确的 reward、碰撞风险或长时 return 监督；
- 使用 one_step_next_observations，而不是把 16 步后的状态作为 SLT 目标。

因此，horizon credit 和 Graph-SLT 操作在不同的学习环节。Graph-SLT 改变状态特征的可预测结构；多步回报改变 critic 被要求拟合的目标。短时 latent 可预测性与长时动作优劣之间没有自动等价关系。

共享梯度路径同样重要：actor 使用共享 extractor 的输出，但 detach 后训练，actor optimizer 排除 extractor 参数；critic loss 与 representation loss 更新共享 encoder。由此可见，probe 中的表示也是 TD 学习和整个训练过程的产物。即使 CARLA probe 明显变好，也不能确认改善全部来自 Graph-SLT；return estimator 改变 critic 梯度，也可能间接改变表示。

场景实现存在客观差异：
- Cross 的 Graph-SLT target 来自 Polyak critic_target encoder；CARLA 来自当前 critic encoder 的 no_grad 输出。
- 非 Cross 路径启用随机 augmentation。
- Cross 为每个 actor 2 条路径、map dim 5；CARLA 为 3 条路径、map dim 2，路线构造方式不同。
- Cross 的 600 raw-step timeout 是 truncation，bootstrap 保留；CARLA 的 302 raw-step timeout 按 termination，不 bootstrap。
- Cross 存在多个 traffic 文件；CARLA 重建配置只有一个 traffic 文件，不能把当前结果视为真正跨 traffic 文件的泛化证据。

这些差异表明“同一个算法名称”不意味着完全相同的训练信号与任务接口。注意，augmentation 与 online/Polyak encoder 开关在 MST 和 v4 两套 factory 中都存在；它们不是 Graph-SLT 专属改进，也不是已确证的场景差异原因。两场景均主要使用 success/collision 终止奖励，不能解释成“Cross 奖励稀疏、CARLA 奖励稠密”。

另一个实际差异是：Cross 的 MST+SLT 配置使用独立 EMA target projector，而 Graph-SLT 使用共享 projector 的 no-grad target 分支。该差异可能影响目标移动和训练稳定性，但没有隔离结果，不能认定它造成 Cross 动力学 probe 退化。

完整方法与基线也并非只有表示损失不同：MST 使用 HierarchicalSceneExtractor、连续二维 actor 和 1e-4 学习率；v4.8 使用 TopoTemporalGraphExtractorV2、32/64/32 slots、lane categorical + 条件速度 Gaussian actor、Graph-SLT/SoftBalancedSlots 辅助目标以及 5e-5 学习率。两者虽然都是 128 维输出，参数容量不因此相同。MST 本身已有 actor、path 和 interaction attention，不能声称它完全不能表达交互。

一个有代码依据、但仍待验证的表示假设是：Cross 输入已包含 heading/ego-social 等信息，MST 的关系 attention 对当前 probe 足够强；CARLA 输入主要是二维几何坐标，显式 topology 与 route pooling 可能对这些 probe 更有利。这种归纳偏置解释需要匹配 backbone/目标对照，不能以输入维度或任务名称直接推出。

代码来源：`algos/sb3_torch/graph_representation.py:33-143`，`sac.py:168-232`，`policies.py:18-29,81-128`，`hybrid_policy_v4.py:270-310`，`envs/sumo/sumo_env.py` 与场景配置。完整代码证据记录为 `r48s1/research_evidence/code_mechanisms.md`。

## 4. 当前 horizon 消融测到的是两个变化的联合效果

完整方法继承 v4.7 candidate，minus_horizon 回到 PARENT_CONTROL：

| 条件 | 累计长度 | bootstrap 折扣 |
|---|---:|---|
| 完整方法 | 最多 16 个 stored decision transitions | γ^h |
| minus_horizon | 最多 4 个 stored decision transitions | γ |

设决策奖励为 r_dec，实际存储长度为 h，则 bootstrap 部分为：
`y = Σ(i=0..h-1) γ^i r_dec[t+i] + (1-d) D_h V_target(s[t+h])`。

环境一个 decision 最多执行 3 个 SUMO raw steps，raw step length 为 0.1 秒；其内部还对 raw rewards 做折扣。这里 h 是存储的决策转移数，代码没有用 h×3。因此只能称其为“按当前决策奖励定义修正 bootstrap horizon”，不能直接称为严格的物理时间一致折扣。终止、超时及尾部存储处理还会改变有效长度。

γ=0.99 时，4 步正确系数为 0.960596，16 步为 0.851458。旧实现始终用 0.99。不能因为 0.99 与 0.9606 的单次差值不大，就认定修正无关紧要：重复 bootstrap 可积累偏差。

一个仅用于说明的数学例子：单状态、无终止、每步 reward=r、忽略 entropy。正确固定点为 r/(1-γ)=100r；4 步累计 reward 却仅乘 γ bootstrap 时，固定点为 (1+γ+γ²+γ³)r/(1-γ)=394.0399r。这个例子不是本项目的实测偏差，只说明“单次系数差小”不等于最终 value 偏差小。

多步 target 的理论作用包括更直接传递后续 reward、减少对短期 bootstrap 的依赖，同时也增加采样方差和对历史行为策略的依赖。更长并不在所有环境、所有阶段都更好。[Munos et al., Retrace](https://arxiv.org/pdf/1606.02647)

本实现还没有累计中间每一步的 entropy reward，也没有多步重要性比率校正。它仅在 bootstrap value 中使用下一决策的 entropy。故不能宣称当前实现拥有完整 soft-policy n-step return 或 Retrace 的理论保证。

对 Cross 最强且准确的表述应为：**在当前 seed 0、最终确定性 actor 协议中，16 步加 γ^h 的联合设置相对旧 4 步加 γ 设置具有显著的实际差异。** 现有结果不能分离长 horizon 与折扣修正的贡献。

## 5. 训练曲线反驳了过于简单的“更容易收敛”解释

对历史 Monitor 数据使用与参考图一致的处理：episode20、零填充、回填到原始步后，每个 raw step 执行 EMA = 0.999×旧值 + 0.001×当前值。EMA 初始值为 0。不把不同超参数曲线平均。

| 条件 | 最后训练 EMA 成功率 | 最后训练 EMA reward | 最后 20 个训练 episodes 成功率 |
|---|---:|---:|---:|
| Cross full | 0.6768 | 0.3471 | 0.65 |
| Cross minus_horizon | 0.7379 | 0.4697 | 0.65 |
| CARLA full | 0.8712 | 0.7347 | 0.85 |
| CARLA minus_horizon | 0.9429 | 0.8757 | 0.95 |

曲线与点表位于 `r48s1/research_evidence/monitor_curve_summary.json`、`monitor_curve_points.csv` 及对应 Cross/CARLA PNG、SVG。阴影沿用原参考变换的单次训练描述性波动带，不是跨训练种子方差，也不是两方法差异的置信区间。

![Cross training monitor](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/r48s1/research_evidence/monitor_cross_full_vs_minus_horizon.png>)

![CARLA training monitor](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/r48s1/research_evidence/monitor_carla_full_vs_minus_horizon.png>)

这并不否定最终 actor 的配对评估，但说明不能说“Cross 去掉 horizon 后训练成功率崩溃”，也不能说“完整方法已呈现更稳定的后期收敛”。数据明确显示训练表现与最终确定性评估的排序不同。

代码区分：
- 训练 warm-up 后使用 Gaussian speed noise + masked categorical lane sampling。
- 直接 actor 确定性评估用 tanh(mean) speed + 可行 lane 概率 argmax。
- 两种模型训练采样均没有经过 fusion/target critic。
- 训练 monitor 还混合了持续变化的模型、训练状态分布及探索；最终评估固定模型且使用另一批种子。

由此提出的可验证假设是：return estimator 改变了 critic 的动作排序和 actor 的 lane 概率，确定性 argmax 放大了决策差异；或者主要差异来自评估状态分布。**目前没有 lane logits/Q-gap 证据，不能认定其中任何一个已被证实。**

## 6. 更深层的解释与证据等级

**已证实：** Graph-SLT 的直接目标是一步 latent prediction；多步回报作用于 critic 监督；两者没有必须在所有场景同幅度增益的理论要求。最终收益取决于状态信息、价值估计、动作输出以及任务分布的共同作用。

**有理论依据但待验证的解释：** 在当前 Cross 中，已有表示可能足以区分一部分局部状态，而动作的长期价值/确定性选择仍是主要瓶颈；在当前 CARLA 中，局部几何与动力学信息改善可以被现有 probe 看到，且该 actor 的最终成功率已接近上限，所以再改变回报估计的可见增量很小。

**限制：**
1. Cross 的 probe 标签并不充分覆盖冲突决策；缺少线性 probe 改善不等于状态表示已足够。
2. CARLA 的 99/100 不证明表示改进导致了成功率，factorized entropy 等其他改变也很重要。
3. 两场景 episode 长度和 timeout 语义不同，但“总 episode 更长”不等于“关键动作到结果的时延更长”。需要事件级 lead-time 分布。
4. 短期预测可能保留与控制无关的变化，或遗漏稀有但决定碰撞的分支。这是目标设计的风险，不是已经实测的失败。
5. 多步 replay 没有校正带来的问题在理论上可能存在，但本轮结果并未证实其造成任何一个具体失败。

因此，目前更明确的问题是**证据归因不完整**；同时存在**表示目标未显式对齐决策相关信息**以及**多步 target 理论边界有限**的设计风险。现有数据不足以裁定“结构设计失效”或“只要换实验就能显效”。

## 7. 能区分这些解释的最小验证路线

不重复已完成的配置，且先保持用户要求的单训练 seed 筛选。

**先使用现有最终模型，检查训练—确定性评估差异。** 在固定 seeds 和 traffic 上，保存两模型的 lane probabilities、速度分布、actor 动作与 critic Q-gap，比较相同状态下的动作差异。可额外评价随机 actor，但它只是诊断，不替代最终直接确定性 actor 主协议。若已有相应结果先复用。

**先补 4 步 + γ^h。** 它和现有 4 步 + γ 区分折扣修正；和现有 16 步 + γ^h 区分长度。若 4 步 + γ^h 已恢复大部分 Cross 表现，就不能把收益写成 16 步独有。16 步 + γ 是完整因子分解的可选补充，不是首轮最低必要项。

**做匹配的表示替换。** 在相同 hybrid actor、entropy、LR、训练预算、回报设置下，用 MST/SLT 表示替代 Graph-SLT。原始整套 MST baseline 不能代替这个单因素对照。必要时再区分 Graph backbone 与 slot objective，并控制容量。

**最后组成表示 G × 回报长度 H 的 2×2。** H 比较 4 与 16 时，两者都使用 γ^h；这样才是在比较长度。固定其他设置，定义：
- 表示边际贡献 ΔG|H = J11 − J01；
- horizon 边际贡献 ΔH|G = J11 − J10；
- 交互 I = J11 − J10 − J01 + J00。

现有 full 提供 J11；旧 minus_horizon 因 γ 也变化，不能直接充当这个长度因子的 J10。要确认互补、冗余或相互干扰，不能只看两个单独指标是否各有一个场景显效。

**将 probe 与真正的控制难点对齐。** 预先固定冲突状态分层，补动作条件的多时距结果、未来 collision/timeout、不同动作的相对价值等标签；仍用共享输入、episode 级划分、必要的 episode bootstrap。不将某行为策略下的未来碰撞标签直接解释为状态固有风险，不事后筛选能显示优势的子集。

**针对 CARLA 的上限。** 保持现有 100 episodes 结果，补更多预先固定种子或预先声明的困难交通条件，报告安全、超时、制动等指标。更难的测试必须对所有方法一致，不能为了出现增益而改变协议。已有训练曲线可以分析学习过程，但不能用更好的训练曲线替代最终 actor 结果。

如果匹配表示消融仍无控制收益，应弱化 Graph-SLT 的贡献声明或重新设计并验证其目标，而不是仅凭 probe 提升保留主创新点。如果修正后的 4 步与 16 步等效，则应把 γ^h 作为正确性实现，不把常规 n-step 设置包装为独立创新。多训练种子可以继续暂缓，但正式论文的泛化与稳定性结论仍需独立训练重复支持。

## 8. 可用于论文的当前保守表述

“在两个场景的 seed 0 最终确定性 actor 评估中，完整方法优于所比较的 MST+SLT 基线。冻结表征在 CARLA 的若干局部预测 probe 上改善，在 Cross 上呈现混合变化。联合改变 return length 与 bootstrap discount 在 Cross 上有较大条件效果，在当前接近饱和的 CARLA 评估中只出现很小净差。表示模块的独立控制贡献、回报长度的独立作用及其交互仍需匹配消融。”

当前不支持的表述：
- “Graph-SLT 只对 CARLA 有效/在 Cross 无效。”
- “16 步是 Cross 提升 50 个百分点的唯一原因。”
- “CARLA 已证明不需要 horizon credit。”
- “两模块必然互补或必然没有组合收益。”
- “γ^h 修正或固定 16 步选择本身已构成充分算法创新。”

本报告的判断依据是历史结果、实际代码及上述原始理论文献；没有把理论上可能的机理当成项目已验证的事实。
