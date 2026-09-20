# v4.9：Learned Collision-Value Model

状态：`locked_before_fresh_development_after_engineering_validation`  
父版本：`v4.8.3 / v4.8 promotion`  
CCFA 模式：`standard / generic CCF-A AI-ML venue family`

## 1. 失败事实

v4.8 promotion 的 12 个作业均已运行并验收，但科学门禁失败。唯一失败项是
`per_scenario_collision_noninferiority`：Cross 上候选 collision rate 为
`0.3167`，TemporalGraph 为 `0.1333`，差值 `+0.1833`；与此同时候选在
roundabout-medium 与 CARLA 上保持或改善安全性，macro success delta 为
`+0.1167`。因此问题不是全局能力不足，而是 Cross 的碰撞风险没有被当前
reward-only 动作价值充分建模。

Cross seed0/seed1 的失败模式不同：seed0 中候选 lane-change/actor override
过多；seed1 中候选虽然把部分 TemporalGraph timeout 转成 success，却也把
部分 success 转成 collision。简单把 lane change 一律压回 keep 无法统一修复
这两个 seed。

## 2. 对“运动学安全投影”的审视

外置运动学投影不进入本版本，原因不是缺少阈值调参，而是它会产生三个方法学
问题：

1. 投影在 actor 输出之后改动作，训练 replay 中的决策模型与部署动作不一致；
2. 固定 headway、lane-veto 或 Q-regret 阈值无法表达遮挡、交互车辆意图和场景
   拓扑中的不确定性；
3. 已完成的闭环诊断中，target-only、keep-tie veto 与 Q-regret veto 在 Cross
   seed0 有改善，但在 seed1 都明显退化，说明它们是 seed-dependent 规则，不能
   作为可推广机制。

这些闭环结果只用于归因，不进入任何新门禁。v4.9 不包含 headway 阈值、几何
unsafe 判定、lane-change veto、动作替换器或其他外部安全规则。

## 3. 单一科学变化

将碰撞风险建模为策略内部可学习的动作价值：

- 在 reward twin critic 之外增加完全独立的 online/target twin collision-value
  critic；
- 监督只来自 replay 中环境实际返回的 `info["collision"]` 终止事件，不使用
  future state、oracle trajectory 或手工几何标签；
- collision target 使用与父方法相同的 16-step boundary mask、
  `gamma^actual_horizon` 与 timeout 语义；
- 每个 head 经 sigmoid 得到 `[0,1]` 的 discounted collision value，双头取最大值
  作为保守估计；
- actor 优化目标增加固定 `lambda=1.0` 的期望 collision value；
- deterministic model score 使用同一学习模型：
  `min reward twin Q - lambda * max collision twin value`；
- target 与 fusion 两个既有部署候选仍保留；fusion 的 actor-confidence 外壳和
  0.90 阈值不变，其 target fallback 改用上述学习分数。

这里的风险分数是网络输出及固定代数目标，不是运动学规则或后处理投影。训练、
校准、部署使用同一个 collision-value 模型。

## 4. 严格保持不变

- 环境 reward、observation、action space、lane mask、action repeat；
- topology/temporal/Graph-SLT 表示骨干、宽度、actor encoder detach；
- reward critic、16-step reward return、entropy、optimizer、learning rate；
- 20k 开发、50k promotion、100k formal 的原始步预算；
- exact-final / highest-training-success checkpoint 候选；
- train-only tie-replicated selector、校准 episode 数和 validation 隔离；
- formal test 锁、最终场景级 noninferiority 与 positive-effect 标准。

## 5. Fresh Development

所有四个作业都必须尝试；任一作业或门禁失败不会中断余下作业。最后统一汇总，
只有四个不可变 run 均验收后才计算科学开发结论。

| Cell | Scenario | Train seed | Train raw steps | Primary/secondary train calibration | Fresh validation |
| --- | --- | ---: | ---: | --- | --- |
| M1 | cross | 10 | 20k | 131000 / 131100 | 141000--141019 |
| M2 | cross | 11 | 20k | 132000 / 132100 | 142000--142019 |
| M3 | roundabout_medium | 12 | 20k | 133000 / 133100 | 143000--143019 |
| M4 | carla | 13 | 20k | 134000 / 134100 | 144000--144019 |

每个作业必须满足模型完整性、selector/decoder 审计、无 formal access、有限数值
和 off-route=0。Cross 两个作业还必须都观察到真实碰撞监督；其 aggregate
success≥0.60、collision≤0.30，单 seed success≥0.50、collision≤0.40，并且
至少一个 Cross seed 同时达到 success≥0.60、collision≤0.25。CARLA 维持
success≥0.50、collision≤0.10；roundabout-medium 要求 success≥0.50、
collision≤0.35。

## 6. Promotion 与 Formal

仅 development 完整通过后启动新的、未被本假设使用的 promotion：2 methods ×
3 scenarios × train seeds `{20,21}`，每作业 50k raw steps、30 validation
episodes，共 12 作业。自动化必须逐一尝试全部作业，最后再汇总；不会因单作业
失败提前退出，也不会自动访问 formal。

Promotion 沿用最终目标的先验门禁：per-scenario collision delta≤+0.05、
worst paired-seed success delta≥-0.15、macro success delta≥-0.05，且至少一个
场景 success delta≥+0.10 或 collision delta≤-0.10。

只有 promotion receipt 为 `pass` 才能访问 formal test。Formal 仍为 6 场景 ×
2 methods × 10 seeds = 120 个不可变作业，并保存配对层级 bootstrap 与 Holm
校正结果。

## 7. Claim–Evidence Matrix

| Claim | 主要质疑 | 必需证据 | 状态 |
| --- | --- | --- | --- |
| 碰撞监督是真实环境事件 | reward 缩放或事件重叠会否污染标签 | replay event-label tests、runtime label diagnostics | engineering passed |
| 模型而非规则决定安全动作 | 是否暗含几何 veto | source freeze、禁止词测试、risk-score equation trace | engineering passed |
| 训练与部署一致 | selector 选 target 时风险模型是否被忽略 | 两种 decoder 均保存 collision critics；score equation match=1 | engineering passed |
| Cross 安全性改善 | 是否只解释旧 promotion | M1/M2 全新训练与 validation | planned |
| 其他场景不被过度保守化 | collision penalty 是否导致 timeout | M3/M4 hard gates | planned |
| 相对 TemporalGraph 达成目标 | 是否为偶然单 seed | fresh paired promotion 12 jobs | locked |

## 8. No-Fabrication Status

本文中的 v4.8 数值来自已封存 promotion；规则诊断明确为 post-hoc、noncausal。
v4.9 smoke 只证明工程链路，不计入科学门禁。M1--M4、fresh promotion 与 formal
尚未运行的结果均为 `TBD`，不得推断或预填。

