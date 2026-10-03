# 交互强化学习机制近邻与反方证据（2026-10-03）

本文件是独立的文献证据备忘，不是完整系统综述。问题限定为：动作条件的事件/首次到达时间分布，能否用于SAC critic或任务关键窗口学习；以及哪些组成已被近期方法覆盖。来源以论文原文、正式会议/期刊页、作者代码为准。预印本单独标注。以下接近度分数是检索优先级（5=直接覆盖核心机制，1=背景或外围），不是论文质量或新颖性评分。

## 最接近的反例与证据

| 优先级 | 来源与状态 | 已经做了什么 | 对候选机制的限制/未覆盖部分 |
|---|---|---|---|
| 5/5 | Tiofack et al., [“SVL: Goal-Conditioned Reinforcement Learning as Survival Learning”](https://proceedings.mlr.press/v306/tiofack26a.html), ICML 2026；[完整原文](https://arxiv.org/html/2604.17551)。 | §4.1/Appendix A.3给出事件首次到达时间的survival/hazard建模及精确身份：固定首个动作、此后按π继续时，Q可写成折扣survival概率之和；用事件时间或非信息性右删失的最大似然训练。Appendix A.3不是主方法结果，但直接排除了“动作条件首次到达时间/由survival恢复Q”本身的新颖性。 | 只建模单一goal事件；§4.2明确指出灾难性/信息性终止需要competing-risks formulation，留作future work。§6还把dense reward的sparse/dense critic decomposition列为自然下一步。因而“把成功/碰撞等分事件类别”以及“survival事件值+稠密回报critic”均已被该文直接指出为近邻方向，不能宣称首次提出这些组合概念。其主算法为离线goal-conditioned survival critic，另有flat state-action Q分析；不是当前在线off-policy连续控制的多终局SAC实现。论文摘要/主实验报告按4 seeds，且任务为OGBench稀疏目标导航，不能据此推断道路安全性能。 |
| 4/5 | Mani et al., [“Safety Representations for Safer Policy Learning”](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html), ICLR 2025；[完整HTML](https://arxiv.org/html/2502.20341)。 | S2C从状态预测到cost事件的离散时间分布，以完整轨迹未来标签训练；单次轨迹若无cost，按预设最大视界标到末档。作者报告SafeMetaDrive及Safety Gym等结果；SafeMetaDrive等主结果按5 seeds报告。 | 预测是state-conditioned而非显式action-conditioned Q；标签是近策略/回放中的未来cost结果，不能视为在给定当前候选动作并继续当前π下的因果反事实分布。单一安全cost类别，没有成功/碰撞/超时竞争风险分解。§6.1的替代表示消融也说明辅助安全表征可能与已有cost critic重复，泛化的“加一个风险头”不足以构成方法贡献。公开官方代码仓库在本次核查中未确认。 |
| 4/5 | Low et al., [“TraCeS: Learning Per-Timestep Constraint-Violation Credit from Sparse Trajectory-Level Labels”](https://proceedings.mlr.press/v306/low26a.html), ICML 2026；[作者代码](https://github.com/siowmeng/TraCeS)。 | 由整条trajectory的接受/拒绝标签训练序列式violation estimator，分解逐时刻credit，并用逐时刻未违规概率的乘积分解建模trajectory存活概率；集成进约束策略优化。官方实现先训练估计器，再用其信号训练策略。 | 不是动作条件的多类别事件时间分布，也不直接用完整事件/时刻PMF组成SAC软Q；但它已覆盖“用轨迹终局标签做时间局部credit，再改变优化/更新重点”的一般想法。若贡献是决策窗口credit或额外辅助头，必须与TraCeS类逐时刻违约归因基线实质区分。 |
| 3/5 | Tiofack et al., [“Survival Reinforcement Learning: Toward Scalable Self-Supervised RL”](https://arxiv.org/abs/2605.31273), arXiv v2（2026-09-18，预印本）；[全文](https://arxiv.org/html/2605.31273)。 | 延续SVL的动作条件首次到达/生存估计和右删失似然，扩为online goal-conditioned actor-critic，加入持续停留于目标的dwell-time目标序列。 | 仍为goal-reaching单事件，不建多种安全终局，也不做通用稠密reward decomposition。预印本声称value只使用完整时间分布的折扣标量汇总、时间信息尚未充分服务policy optimization；因此“online survival RL”也不能单独作新颖点。 |
| 3/5 | Bohlinger & Peters, [“ChronoSRL: Temporal Geometry for Self-Supervised Reinforcement Learning”](https://arxiv.org/abs/2609.36238), arXiv v1（2026-09-28，预印本）；[全文](https://arxiv.org/html/2609.36238)。 | 用state-action/goal embedding距离拟合goal到达时间，并从表示预测完整goal-reaching-time分布与到达后的dwell time；摘要报告在多个导航/运动基准上比较。 | 这是“时间几何+动作条件goal time distribution”的近期反例，故不能把时间位置/到达时间分布本身称为新。但问题仍是单一goal事件、自监督goal-conditioned control，不是SAC的成功/碰撞/超时标记竞争风险，也未证明与固定历史/路口冲突动作差有关。仅预印本，当前状态不等于同行评审定论。 |
| 2/5 | Duan et al., [“Distributional Soft Actor-Critic with Three Refinements”](https://ieeexplore.ieee.org/document/10858686/), IEEE TPAMI 47(5):3935–3946 (2025), DOI [10.1109/TPAMI.2025.3537087](https://doi.org/10.1109/TPAMI.2025.3537087)；作者代码[DSAC-v2](https://github.com/Jingliang-Duan/DSAC-v2)。arXiv初稿于2023-10-09提交：[arXiv:2310.05858](https://arxiv.org/abs/2310.05858)。 | DSAC-T/DSACv2是off-policy distributional SAC；三项改进为expected-value substitution、twin value distributions、variance-based critic gradient adjustment。 | 它建模return分布而非标记事件原因/首次事件时间；因此不是候选方法的直接等价物。不过分布式critic接入SAC本身、双critic取较低均值和普通分布式值估计均不能作为新颖性依据。若只把标量Q替换成一个概率/分布头但无法证明事件结构的独特用途，容易被视作distributional critic的任务化改写。 |

## 补充的概念边界

- 竞争风险离散事件时间分布本身是成熟统计建模工具。SVL原文相关工作引用Lee et al., [“DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks”](https://ojs.aaai.org/index.php/AAAI/article/view/11842), AAAI 2018。动作条件或离散时间类别PMF的表达形式不是贡献；研究问题应落到RL策略评价/更新的特定数学和实证增益。
- 分布式奖励分解也有RL先例：distributional reward decomposition for reinforcement learning（DRDRL），NeurIPS 2019，[官方论文页](https://proceedings.neurips.cc/paper_files/paper/2019/hash/97108695bd93b6be52fa0334874c8722-Abstract.html)。因此“分事件头再线性组合奖励权重”不够作为独立创新主张。
- 另一个需留作扩展检索的近期反例是 [Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability](https://arxiv.org/abs/2605.14246)，arXiv预印本。其摘要已有有限历史下的候选动作条件短时违规预测，并把风险用于价值学习/动作门控；它未建模完整的多原因、全时域事件PMF，但削弱“加action-conditioned safety head并影响actor/critic”的泛化主张。

## 针对候选路线的严格限定

现有证据支持的事实是：动作条件首次到达时间Q身份、survival/hazard最大似然与右删失、state-conditioned安全事件时间分布、逐步违规credit、SAC分布式critic，以及把dense与sparse价值拆分的研究动机，都有直接近邻。SVL把“信息性失败的竞争风险”及“dense reward的sparse/dense critic decomposition”明确列为未来方向。因此把这些点拼接后就声称新的一般原理，证据不成立。

仍可检验、但尚不能据此称为已验证创新的窄假设是：在固定STRT骨干和同样deadline/time augmentation下，把任务内成功、碰撞、超时作为互斥竞争终局，估计给定当前动作并按目标SAC策略延续的事件-时间联合分布，能否比参数量/训练预算匹配的标量双Q、普通分布式双Q、事件分类辅助头及逐时刻credit基线更好地校准动作排序，尤其在进入时机与清空路口窗口。功能上可依赖该表示，不代表性能收益由此产生。

这条假设的关键前提尚待验证：目标π是条件分布定义的一部分；来自行为混合回放的完整episode未来标签直接估计的是行为/混合策略，不自动等于当前π。用target-π下一动作期望与时间移位递推可以另定义目标，但必须证明概率质量、有限deadline与终止语义一致，并核验偏离数据覆盖时的误差。任务timeout若是实际任务失败应作为可观测事件类别；记录截断/数据收集结束才是删失。概率归一、terminal one-hot及Q投影要避免重复计算终局与稠密奖励。事件分布的期望若只重构原SAC标量回报，理论最优策略目标未变；性能变化只能先解释为表示/监督/优化归纳偏置，不能宣称新的Bellman理论或安全保证。

最小公平比较应在统一的SAC+MLP与正确修复STRT骨干、训练数据/步数、优化器、网络容量、deadline/time augmentation、reward与eval协议下并列：scalar twin-Q；普通distributional SAC式critic；只做事件分类辅助监督但actor/critic目标仍为标量；带时间但单事件的survival Q；候选多终局事件-时间critic。首轮只验证训练目标、概率归一/校准、Q重构一致性和固定checkpoint的action-ranking/决策窗口指标；这些不替代完整多种子策略结果，也不授权当前新增训练。另需明确联合C8/C9修复的STRT与旧方法有源码及行为差异，不可把新STRT相对旧STRT当成此候选机制的公平消融。

## 搜索与评审轨迹

公开查询围绕“action-conditioned first hitting time distribution competing risks safe reinforcement learning”“survival value learning action-conditioned Q competing risks”“sparse dense critic decomposition survival RL”“state-conditioned steps-to-cost distribution safety representation”“per-timestep constraint violation credit trajectory labels”“distributional soft actor critic three refinements”及ChronoSRL/SRL后续。筛选优先考虑论文正式出版页、原始论文正文和作者实现；预印本仅作为时效性反例，明确标记其状态。候选评分按与中心机制的重叠程度：SVL 5，SRPL/TraCeS 4，SRL/ChronoSRL 3，DSAC-T 2；分数只表示需要讨论的相似度。

CCF方面，本次核对到[CCF第七版正式发布页](https://www.ccf.org.cn/Academic_Evaluation/By_category/)（2026-03-31发布、4月9日更正）和其[人工智能会议A类目录入口](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/al/)，但官网目录附件/分类内容抓取出现405或内部错误。本备忘不据此断言ICLR属于或不属于2026 CCF-A；正式写作前应以可读的第七版AI类A/B/C目录文件逐项核验。CCF官方说明目录是推荐列表，并非单篇论文影响力评价。

## 本项目材料定位

- 当前修复实验协议、源码归档与限制：[`d1_contractfix_protocol_20261003.md`](../d1_contractfix_protocol_20261003.md)
- 同场景历史公平比较及源码版本边界：[`d1_contractfix_comparison_evidence_20261003.md`](../d1_contractfix_comparison_evidence_20261003.md)
- 历史模块任务作用及归因证据：[`sorted_module_attribution_20261002.md`](../sorted_module_attribution_20261002.md)、[`sorted_task_module_attribution_20261002.md`](../sorted_task_module_attribution_20261002.md)、[`sorted_routeact_conflicttime_attribution_20261003.md`](../sorted_routeact_conflicttime_attribution_20261003.md)
- 原方法实现映射：[`full_mst_slt_implementation.md`](../full_mst_slt_implementation.md)

