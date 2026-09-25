# v4.8/lr_half 后续研究路线

状态：研究计划，不是执行记录。本轮不启动或中断实验，不修改正在运行的配置。范围为 Cross/CARLA；主结果始终使用预定训练预算结束时的最终模型和直接确定性 actor。历史 checkpoint 选择、fusion、target-critic 部署不进入路线。

## 推荐顺序

完成当前有限的稳定性调参，同时开展现有模型的低成本机制诊断；随后冻结可比较的训练协议，进行匹配的表示与回报消融；只有在缺陷被定位后才修改方法；最后进行公平调参、多训练种子与冻结测试验证。

“稳定性调参”属于调参，不应另外安排一个无边界的“先追最高分、再追平滑、最后解释机制”的过程。也不必等训练曲线单调上升才做机制研究。稳定性的目标是可比较、可复核，不能把探索波动一概认定为发散。

| 阶段 | 工作 | 技术/模型 | 产出与进入下一阶段的条件 |
|---|---|---|---|
| 0 | 固定协议、审核关键实现语义 | 配置指纹、单元测试、短 smoke test、日志审计 | actor/回报/终止/步数单位清楚，协议可追溯 |
| 1 | 完成当前有限稳定性筛选 | 既定 LR、batch、tau 候选；最终 actor 与训练日志联合评价 | 冻结一套公共设置或明确两个保留候选 |
| 2，与阶段1并行 | 比较现有最终表示、行为与失败模式 | 固定共享数据、线性/MLP probe、状态分层、CKA、轨迹重放 | 形成少量可证伪的机制假设 |
| 3 | 表示包 × 正确多步回报的匹配消融 | 统一 hybrid actor 的 MST+SLT/Graph-SLT，n=4/16，均 gamma^h | 确认边际贡献和交互，区分优化收益 |
| 4，按证据触发 | 拆解表示内部机制，检验两个风险 | 图/时序/预测损失消融；n=8 等补点；SPR、DeepMDP/DBC、Retrace 的条件候选 | 单次只改变已定位的一项，不无条件叠加模型 |
| 5 | 对保留方法进行公平调参和种子确认 | 同预算 HPO、配对训练 seeds、分层统计 | 区分条件组件效果与最佳方法效果 |
| 6 | 冻结评估、鲁棒性与论文证据包 | 新的固定 test seeds、预定义交通条件、最终 actor、安全/舒适指标 | 主张与证据对应，完整保留失败与零效果 |

## 阶段 0：先固定比较的含义

建立三个不同的协议身份：
- 原始锚点：历史 v4.8/lr_half 和 MST+SLT 的最终模型，保留原始配置与结果。
- 稳定性锚点：当前调参结束后选定的训练设置，作为随后组件实验的公共设置。
- 最终确认协议：方法及超参数冻结以后才打开的测试种子/条件。

历史 10000–10099 seeds 已被用于分析；当前用于挑选调参候选的评估 seeds 也属于开发证据。不能再将这些结果称为从未参与决策的最终测试。CARLA 当前只有一个重建 traffic 文件，不同 SUMO seed 不等于不同交通文件的 OOD 泛化。

记录并核对：场景、训练 seed、raw-step 预算、更新次数、optimizer 与各分支 LR、batch、tau、辅助损失权重、回报长度与折扣、终止/truncation 语义、encoder/projector target、augmentation、最终模型 hash、actor 输出方式以及评估协议。

关键正确性检查：
- h 是实际存储的 decision transitions，raw step 与 decision step 不混用。
- 非终止与终止的 bootstrap、短尾序列、timeout、末端处理符合明确定义。
- 明确 task 的 time limit 是任务终止还是采样截断；任何协议修正要所有方法一致，另立版本并保留历史结果。
- Graph-SLT 确实使用预期的一步或多步 target；actor、critic、表示分支的梯度路径可复核。
- 诊断评估不改变训练 RNG、buffer 或 optimizer；可用独立环境/进程及独立随机源。
- 若复现或 smoke test 暴露实现错误，先修复并标记受影响结果，不把错误当超参数问题。

无需全面重构，也不要求把所有历史环境约定改成同一种。需要的是明确语义和同一场景内的公平比较。

## 阶段 1：有限稳定性筛选

让现有已授权轮次按固定预算完成；不在看见部分曲线后无限扩展候选。优先复用已经完成且配置/协议匹配的实验。

已核对当前计划：late_decay、batch64、target_slow 三个单因素候选；lr_half 与 lr_quarter 作为复用对照。Cross/CARLA 各 seed 0，50k raw steps、5000 warmup、45001 次梯度更新、action_repeat=3；开发评估种子为 420000–420099。共有 6 个新增训练和 4 个复用模型。late_decay 同时调整 actor、critic、representation 与熵系数 optimizer，效果只能先归因于这组联合调度；batch64 的计算成本不同，不应称等计算量。来源：experiments/v48_stability_v1/PLAN.md、METRICS.md、RUNBOOK.md。

这一轮固定表示、action head 与 return 定义，主要研究已准备的优化设置。tau 是特殊项：它不只改变 critic target，在 Cross 中还会影响 Graph-SLT 的 target encoder，须记录这个耦合，不能将 tau 收益单独解释为纯数值稳定性。

选择候选时联合使用：
1. 最终直接 actor 的成功、碰撞、超时。
2. 未平滑的 episode 日志、固定窗口统计及预先定义的后期回撤。
3. 单次训练中的趋势残差波动、低表现窗口频率。
4. Q/TD 误差、gradient norm、alpha/entropy、lane probability、非有限数值等优化诊断。
5. 若测量收敛速度，预先定义阈值和必须持续满足的窗口长度；未达到要记为未达到，不能直接赋值为训练终点。

成功率的 Bernoulli 方差随均值变化，所以不能只用原始方差比较“稳定性”。也不能通过更强 EMA 或更长窗口得到更好看的图，就认定算法更稳定。展示继续沿用 episode20 + 每 raw-step EMA 0.999；推断同时使用原始数据。

区分：数值发散、持续的策略性能退化、正常探索波动。单次训练的波动不能当成跨训练种子的方差。

可以在后续新实验的固定训练步做确定性 actor 开发集评估，用于辨别探索曲线与策略能力；它只记录学习动态，不选择用于部署/主结果的 checkpoint，主结果仍是最后模型。

若当前候选均不能改善可比较性，停止盲目扩大搜索，转向具体数值与机制诊断。若延长预算，视为另一实验因素，对比较方法采用一致预算；不能只延长一种方法后归因于组件。

## 阶段 2：用现有模型判别表示“优/差在哪里”

不必等待调参全部结束。优先复用已存在的 probe 数据、最终模型与轨迹；缺少逐步动作概率或风险信息时补采集诊断数据，不重复训练已完成模型。

### 固定共享诊断数据

使用 MST、原始 v4.8、之后的稳定 v4.8 轨迹组成预先规定的混合分布。每个 encoder 必须接收完全相同的 observation/history/action，不各自在自己的 rollout 上评价表示后直接横比。

按 episode、traffic seed 或更高层场景分组划分 fit/validation/test；同一 snapshot 的不同动作分支必须处于同一划分。标准化和 probe 超参数只使用训练/验证部分。

分层维度可包括邻车数量、路线几何、交互强度、到冲突事件的时间以及正常/碰撞/超时轨迹。分层在看模型胜负前定义，并同时报告自然分布结果。过采样罕见失败适用于诊断，但不能将改变类别比例的数据用于未校正的自然风险校准结论。

### 六类比较

| 问题 | 诊断 | 解释边界 |
|---|---|---|
| 局部信息是否保留 | Ridge/逻辑回归 + 固定容量小型 MLP，预测动力学、距离、TTC | 线性与非线性 probe 区分可读性；不是信息论充分性的证明 |
| 是否保留决策信息 | action-conditioned future collision/timeout、冲突进入时间、未来进展，多个 decision horizons | 结果依赖指定行为/继续策略，不能称状态固有风险 |
| 是否能区分动作优劣 | 固定状态下的动作分支 rollout，公共 continuation policy，小型动作排序头 | 标签是该 continuation policy 下的估计，不是真实最优 Q |
| 图与时间结构是否有用 | 按关系/时延分层、合法 actor 置换、历史长度与观测扰动敏感性 | 破坏合法几何或仅测试时删模块会产生分布外效应 |
| 表示是否退化或互相干扰 | 有效秩、slot covariance、target drift、梯度范数与 cos(grad_repr, grad_TD)，辅以 CKA | CKA 只描述相似性，attention 图和梯度相关性都不单独证明因果 |
| 能否转为闭环收益 | 匹配 action head 后的最终 actor 评估 | 需要阶段3训练对照，probe 不能替代闭环实验 |

probe 的解读：
- 线性更好而 MLP 接近：支持信息更容易读出，不等于包含更多信息；这种优化优势仍可能有价值。
- 两种 probe 都更好：支持当前任务标签的可预测性改善，仍需控制数据覆盖和 probe 容量。
- 局部预测更好但关键风险/动作排序更差：支持短时预测与控制目标不一致这一风险。
- 风险 probe 更好但闭环没有收益：检查 actor/critic 是否使用了信息，以及回报估计、探索或动作头是否成为瓶颈。
- 两种表示都差：增加 raw-observation 或 privileged-state 诊断参照，区分表示损失、有限历史观测与 probe 能力；这些参照不是严格理论上界。

动作分支数据可使用 SUMO/TraCI 状态保存与恢复，但必须先验证恢复一致性，同时保存 wrapper 历史、交通调度与相关 RNG。给定同一初始状态，执行一个可行动作，再用同一个 reference policy 继续，多次配对 rollout 得到条件回报/风险。诊断中的 reference policy 不进入部署。

若比较原 critic 的绝对 Q 与 Monte Carlo 目标，必须匹配其 discount、entropy 和 continuation policy。含熵 soft Q 不能直接与无熵 episode reward 相减并称为估计偏差。

## 阶段 3：先建立可以归因的四格对照

表示因素：
- G0：MST+SLT 表示接入公共 hybrid actor/critic 训练接口。
- G1：拓扑-时序图表示 + Graph-SLT。

回报因素：
- H4：4 步，实际 h 对应 gamma^h。
- H16：16 步，实际 h 对应 gamma^h。

| | H4 | H16 |
|---|---|---|
| G0 | 公共接口下 MST+SLT，4步正确折扣 | 公共接口下 MST+SLT，16步正确折扣 |
| G1 | 图表示+Graph-SLT，4步正确折扣 | 图表示+Graph-SLT，16步正确折扣 |

每个场景内固定 action head、optimizer 配置、entropy、训练预算、评估协议以及其余损失。原始连续 actor 的 MST 仍保留为方法级基线，但不冒充四格中的 G0。

在这个四格中，G 首先指明确定义的“表示方案整体”。如果某个 slot 辅助正则不能通过公共接口在两种方案中一致应用，就把它明确列入表示包的差异，再通过后续消融分离，不能宣称它已被固定，也不能把整包效果直接归为单独 Graph-SLT loss。

先补最有区分力的 G1/H4 与 G0/H16，再决定是否立即补 G0/H4。若需要讨论交互或互补，四格必须完整。

每场景有四个训练单元，两场景共八个 seed-0 单元；符合完整配置指纹的已完成单元直接复用。旧 minus_horizon 为 4步+gamma，不能替代 H4。若稳定设置改变，旧结果保留为历史锚点，不能拼进新设置的因子分析。

定义：
- 表示在 H16 下的边际效应：J(G1,H16) - J(G0,H16)。
- horizon 在 G1 下的边际效应：J(G1,H16) - J(G1,H4)。
- 交互：J11 - J01 - J10 + J00。

如果 G0/H16 已接近 G1/H16，应如实削弱 Graph-SLT 的控制贡献声明。如果 G1/H4 已恢复旧消融损失的大部分，须将折扣正确性与长 horizon 的收益拆开陈述。

## 阶段 4：仅对定位到的风险做方法实验

### 表示风险

先在同一个图 backbone 上比较：
- Graph-SLT；
- 原 SLT；
- 只关闭预测损失，其余已有项固定。

这样先区分收益来自 graph backbone 还是预测目标。涉及 loss 权重、projector、target EMA、辅助正则的变化要逐项记录，不能同时删掉多个项却称单因素消融。

只有需要完整 backbone × objective 分解时，再加入 MST+Graph-SLT。Graph-SLT 的语义 slots 与 MST 的非结构化输出并非天然兼容；必须提供有意义、容量可控的读出接口。简单把 128 维切成 32/64/32，不能保证构成公平的语义 slot 对照。

若“拓扑”与“时间”都是核心主张，再分别训练关系弱化/移除和时间聚合替代的消融；测试时删除模块只能作为敏感性诊断。可使用保持输出接口的关系无关聚合、固定池化或 GRU 等简化替代，参数变化需报告。

根据阶段2结果只选择一项修订：
- 若主要是预测时距过短：使用 SPR 类型的 action-conditioned 多步 latent prediction，保留其他设置，比较 1/4/8/16 等预定义时距。预测更远本身不保证控制相关。
- 若主要是对安全/价值信息不敏感：以 reward/transition 预测或 DeepMDP、DBC 思路建立单独目标对照。
- 若表现为表示与 TD 的优化冲突：先测试预测损失权重/更新频率/target 更新方式；梯度冲突只是线索，仍需干预后效果证明。
- 若证据提示关键状态不可观测或有限历史不足：先测试历史长度或记忆接口；不能指望更复杂图结构恢复输入中不存在的信息。

这些方法的原始理论通常有 Markov、转移建模或其他假设。当前有限历史观测未必满足，不能直接移植性能保证。也不将 SPR、DBC 和新的 credit 算法同时加入第一次修订。

### horizon 风险

先完成 gamma^h 下的 4/16 对照。若出现明确场景交互，再补 n=8；n=1 可作短回报参照，n=32 仅在事件时延诊断提示需要时增加。不要默认完整扫所有 n。

记录实际 h、从关键动作到结果的时延、nonzero-reward target 覆盖、bootstrap 部分大小、目标方差及行为策略偏离。长 episode 不等于关键决策延迟长。

用同一批 frozen transitions 可单独比较 target 的性质；在线训练再检验策略—数据反馈后的净效果。前者不代替闭环结果。

若固定 horizon 敏感性被证实，再考虑 lambda-return 或与当前 hybrid soft objective 一致的 Retrace 类型校正。Retrace 需要记录行为策略概率/密度、处理 lane mask 与连续动作变换、正确区分 entropy 项和 terminal；没有行为概率的旧 replay 不能凭空补出严格重要性比率。

每场景分别选 n 可以是一种有效调参策略，但不支持“统一 horizon 跨场景最优”的主张。若要主张统一方法，必须有公共配置或预先定义的自适应规则，并与各固定 n 在同等协议下比较。

该风险的检验目标不是证明某个 n 对所有任务普适最优，而是检验方法对合理 n 范围的敏感性、场景间是否存在稳定交互，以及预先定义的参数选择规则是否可靠。仅因不同场景偏好不同 n，不足以判定方法失效，也不自动要求新增自适应模块。

任何新增的 latent predictor、reward/risk head 或 return 校正只进入训练与诊断。最终部署仍使用预算结束时的 actor，不通过额外风险选择器、critic decoder 或历史 checkpoint 选择实现增益。

## 阶段 5：公平调参与训练种子确认

保留两个不同问题：
1. 相同公共设置下的组件效应：用于因果归因。
2. 每种方法在相同调参预算下的最佳表现：用于方法竞争力比较。

从完整方法上选出的稳定设置直接给 MST，并不证明给了 MST 最佳优化机会。因此，对最终保留的 MST+SLT 与完整方法提供预先规定、相同预算的 HPO；保留历史试验并仅复用协议相容的单元。算法特有参数可使用合理搜索范围，但记录全部 trial，不以成功数量平衡预算。

用户此前暂缓多种子，当前筛选继续 seed 0。方法和超参数收敛后，再以至少若干独立训练 seeds 确认方向；可先 3 个用于早期确认，关键论文比较争取 5 个或更多，数量由不确定性与预算决定。这不是本轮立即启动多种子的指令。

每场景报告训练种子层面的均值、离散程度与区间。评估 episodes 嵌套在模型/训练 seed 之下，不能将几百个 episode 当成几百个独立训练复现。使用配对评估和按训练 seed/episode 分层的 bootstrap；RLiable 可作为统计报告方法参考，但两个场景不宜只压成一个总分。

## 阶段 6：冻结测试与论文证据

在调参/方法选择完成后打开新的固定测试 seeds 和预定义交通条件。同一条件用于所有方法，不根据哪种方法获胜来增删场景。

主指标：成功、碰撞、超时、路线完成。
辅助指标：最小距离/TTC、适用冲突区域中的 PET、真实运动学加速度/jerk、急制动、速度限制违反、任务完成时间。使用真实采样间隔；明确 TTC/PET 的适用条件、缺失与截断。速度、超时与安全联合报告，避免低进展策略被误判为更安全或更稳定。

鲁棒性条件可覆盖合法的交通密度/速度变化、观测噪声、短时轨迹缺失与路线几何变化。当前训练效率不是淘汰标准，但训练预算公平性与计算/参数量描述仍需要记录。

最终证据包：
- 完整方法与强基线的最终 actor 主表；
- 正确折扣下的表示×horizon 四格与交互；
- 匹配 backbone/objective 的必要内部消融；
- 分层 probe、动作排序与失败轨迹；
- 原始训练日志、规定的 EMA 图及固定步 actor 诊断；
- 多训练种子不确定性与统一鲁棒性测试；
- 零收益、失败候选、全部调参记录及代码/数据/模型身份。

如果一个模块长期只能改善局部 probe、不能改善控制相关信息或闭环表现，应弱化主贡献或删除它。若收益主要来自正确折扣或优化设置，应如实归因；正确实现与优秀调参是有效研究工作，但不能自动当作新的算法机制。

## 两个 GPU worker 的阶段安排

当前两个 worker 继续完成既定稳定性任务；CPU 可做已有数据、probe 和日志分析，不额外抢占训练 GPU。阶段3开始后，一个 worker 可负责表示对照，另一个负责正确回报对照，均按单元清单去重。阶段4只运行被证据触发的修订。阶段5再集中资源给保留方法的公平调参和种子重复。

复用判据是完整配置与协议身份一致，不是方法名或 final_model 文件名相同。新的测量可以补采集，但应说明新增信息，避免把重复跑一次评估包装成独立训练证据。

## 原始文献与本路线中的用途

- [Alain & Bengio — Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)：冻结表示的标签线性可读性；不证明控制充分性。
- [Kornblith et al. — Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html)：CKA 比较相同输入上的表示几何；相似度不等于质量。
- [Schwarzer et al. — Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://arxiv.org/abs/2007.05929)：SPR 的多步 latent prediction 可作为时距错配的条件实验；Atari 结果不保证本项目有效。
- [Zhang et al. — Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742)：DBC/bisimulation 思路使表示与奖励和动作条件转移相关；需要检查理论假设。
- [Gelada et al. — DeepMDP: Learning Continuous Latent Space Models for Representation Learning](https://proceedings.mlr.press/v97/gelada19a.html)：reward 与 latent transition prediction 及其表示质量分析；不是普通观测预测的通用保证。
- [Munos et al. — Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647)：Retrace 的多步 off-policy 校正；不是直接证明本项目 n=16 的原因。
- [Agarwal et al. — Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：独立训练运行的不确定性、分层 bootstrap 等；同模型评估 episodes 不替代训练 seeds。

理论来源经原始论文页面核对。路线中的实验判据属于针对本项目的设计建议，不是上述论文报告的本项目结果。
