# 两项新增方法：研究问题与诊断判读协议（2026-10-02）

本协议针对 intersection_sorted_depart4p0 的单 seed、从零 100k raw steps 实验。用户要求两路 worker 分别运行两个新增方案；本轮不插入第三条基线或多种子训练。正式是否启动、实际采集覆盖与验证回执以启动计划和 suite manifest 为准，不以本协议当作运行结果。

## 1. 本轮要解决的证据缺口

此前 activation、attention、gradient、post-step update 可以检查模块参与计算、是否有输入以及是否进入优化；它们不能单独证明策略利用了对应信息。最终成功/碰撞/超时又不能定位是哪条表征路径改变了动作。

本轮加入在正常训练/评估中低频执行的**同状态确定性策略敏感性检查**：对于固定模型参数和同一真实决策观测，比较正常前向与一个明确的模块干预。只计算候选动作，不执行该候选动作，不新增 SUMO 回合。它填补“内部参与”和“实际策略输出”之间的证据，不替代重新训练的消融。

三层证据必须分开：
1. 计算与优化参与：输入/掩码/激活/梯度/真实参数更新；
2. 指定干预下的局部策略敏感性：相同状态下确定性动作均值的变化；
3. 性能作用：正式策略在相同评估条件下的结局及行为差异，仍有单训练 seed 与历史版本限制。

## 2. 两个新增方法

### A. sac_mlp_d1_st_rt_topo_goalonly_v1

假设：先前 Topo 的车辆表示注入、关系边注入和 goal 读出同时变化，可能抵消 goal 可达信息的潜在收益。单独保留 goal 路径可以更清楚检验该假设。

- 保持 ST-RT 的原车辆状态/空间消息路径。
- use_topology_actor_intent=false，use_topology_relations=false。
- use_topology_goal=true，use_route_reachability=true。
- 保留合法 goal 候选 mask 和 beta goal residual；不实施 action mask/shield。
- 使用原 384→128→128、两次 ReLU 的联合 readout，活跃参数 65,792。
- Graph-SLT、SBS、representation loss 与 slot-balance loss 关闭。
- 新方法独立注册，旧 Topo、旧 routeaware 等方法保留。

最低功能验收：同共享权重条件下 beta=0 的确定性输出恢复 ST-RT；全部 Topo 作用关闭后共享参数梯度一致；关闭的 actor/relation 分支不应被解释为“漏梯度故障”；合法候选与空候选旁路满足原协议。此种功能等价不声称随机训练轨迹或未来性能相同。

### B. sac_mlp_d1_st_rt_3slot_nonlinear_v1

假设：旧线性三槽除了语义分槽，还删除大量读出参数和非线性，不能据其失败单独否定分槽思想。

- ST-RT 上游保持原配置，Topo 关闭。
- ego：128→132→32；social：128→120→64；route：128→132→32。
- 每支均 Linear→ReLU→Linear→ReLU，输出布局仍为 32/64/32。
- 活跃 head 参数 21,284 + 23,224 + 21,284 = **65,792**；与联合 head 相同。
- 原线性三槽的活跃 head 参数为 16,512，保留原方法供历史对照。
- 不增加辅助学习目标或改变 SAC actor/critic 后续网络。

本对照控制参数数量、非线性深度和末端激活，但仍改变编码器 head 的跨槽连接结构；参数相同也不等于 FLOPs、函数族或优化难度相同。槽输入在更上游已发生交互，槽名字不是语义解耦证明。

## 3. 各模块应当用哪些证据判断

| 组件 | 被动采集 | 同状态功能检查 | 可支持的解释与限制 |
|---|---|---|---|
| ST 空间消息 | 有效邻车/边、消息增量、参数更新 | 旁路空间消息，保留其余条件 | 动作依赖空间消息；不说明该依赖更安全 |
| ST 时间 | 有效历史帧、历史变化、时序读出增量 | 当前帧/历史信息受控干预 | 历史有信息时策略是否依赖它；无有效历史须标不适用 |
| social 读出 | 有效社交对象、attention、读出量级 | 保留 ego、移除非 ego 社交读取 | 与空间消息路径分开检查；不是独立训练消融 |
| RT 注入 | route 输入有效性、注入量级、梯度 | 只旁路 route→intent 注入 | 区别于末端 route goal 通路 |
| RT goal/readout | route 候选/查询及表征 | 只旁路路线目标读出 | 路线信息参与动作不等于动作满足路线 |
| Topo 车辆/关系/goal | 各路径实际 active、post-LN delta、同候选 uniform 关系基准 | 分别旁路三条作用路径 | 新 goal-only 中前两条不适用，不能填成“零作用” |
| 可达性 mask | known/unknown/legal/expand/bypass、合法 attention 质量 | 比较合法 mask 与原几何候选 mask | 仅检查候选约束如何改变策略，不是安全 shield |
| ego/social/route 三槽 | 每槽 RMS/有效变化、参数更新、actor 首层贡献 | 分别移除一个槽再计算策略均值 | 是指定移除下的依赖；零化可能分布外，不能从一次结果断言槽的语义或因果信用 |

新增功能记录应包含 baseline 和 variant 的 pre-tanh actor mean（实现可取得时）、post-tanh normalized mean action、两个维度的差、饱和程度、控制阈值跨越及可复用的物理动作解码。实际执行动作单列；训练中的随机动作不能与 shadow mean 混为同一种策略输出。

不要设没有依据的“delta 超过某值就有用”阈值。首先检查 noop 一致性、字段有效分母、采样覆盖、动作映射是否变化，再结合已有碰撞/超时/路线事件。小 action delta 可能来自饱和或冗余，大 delta 可能来自分布外破坏；都不是性能收益的充分条件。

## 4. 采样与隔离要求

- 训练使用真实 rollout 决策观测，每 5,000 raw steps 至多一次，100k 中上限约 20；记录 raw/decision/update 与 warmup/action source。不能把 replay batch 记录冒充该时刻真实动作所见状态。
- 评估每回合至多三个预声明条件样本：首个有效多车状态、首次路线/冲突事件、末个 preterminal 状态；去重、条件未出现及无法获取的情况明确记录。具体已实现 selector 与触发字段见采集协议，不临时按有利结果挑状态。
- 保存 episode、seed、decision/raw、sample ID、触发上下文和终局连接，使敏感性能够按成功/碰撞/超时、可达/不可达、有/无冲突等分层。
- 保持 shadow 权重固定，不执行候选动作、不写 replay、不调用环境 step、不增加最终评估 episode 数。
- 隔离并恢复模型模式、随机数状态、诊断缓存/计数、hook 中间状态；不改变参数、梯度、optimizer。
- 不适用模块、缺输入、无候选、错误与正常小差异须使用不同状态，不以一律写零掩盖。
- 记录采样次数、跳过原因、error、耗时与预算，以便确认两个方法诊断负担和覆盖程度。
- selector 选择了特定状态，敏感性统计不是所有决策状态的无偏平均。按真实有效样本数解释，不能把最多 300 条 probe 误当 300 个独立评估回合。
- 两个策略各自访问的状态不同。即便评估 seed/交通模板相同，其 probe 状态也不能视为相同；跨方法比较敏感性需附道路阶段、有效历史、冲突/可达性和结局分层，不能仅比较两个总体 action delta 均值就宣称某方法更依赖或更善用模块。

旧 full 的 delegated forward 不是此次两个新方法的路径；缺乏支持的旧路径应明确不适用，不假定新增字段自动覆盖全部历史模型。

## 5. 正式协议与结果判读

两条正式训练采用 seed 0、100,000 raw steps、warmup 5,000、action_repeat 3、batch 32、LR 1e-4、buffer 20k、gamma .99、10k checkpoint；两路独立 CUDA worker。最终 deterministic 评估各 100 回合、seeds 10000–10099。沿用同一 sorted 三背景流、depart_scale=4.0、30 固定模板池；训练和评估共享该池，不能宣称未见模板泛化。

保存 final_model 实际 SHA、training_complete 与 evaluation identity，核验预算/种子/方法/source archive。训练一致 shaped return 和历史可比较 raw return 分列，六个奖励分量核对；不以新的 shaped return 与旧 raw mean_return 混比。

- goal-only 重点：相对旧 Topo/routeaware 是否减少原成功回合丢失、错误车道停驶及 timeout，同时保留碰撞→成功；如 goal 分支敏感性明显而行为仍无改善，说明“依赖该信息”尚未实现任务收益。
- nonlinear slots 重点：相对线性三槽是否恢复成功、减少原成功→碰撞，是否恢复不同上下文中的动作响应。改善支持旧容量/非线性混杂参与退化，但不能分开确认这两者谁是唯一原因。
- 若新三槽仍差而输入、优化、接口和功能依赖正常，分槽位置/结构限制更值得检验；不立即叠加新模块。
- 若 goal-only 改善，支持此前耦合路径值得进一步拆分；尚不能直接判定被关闭的两个分支哪一个有害。
- 历史 ST-RT 63/37/0 是目前主链最佳观察参照，但旧运行源码不完整。用户本轮明确选择两项新增方法，故保留历史可比性限制，不擅自增加一条训练。

短程 smoke 只验证模型保存、训练更新、正常评估、诊断预算/隔离/输出和回报对账，不计入正式实验汇总，也不用于判断新方法性能。

## 6. 理论来源与解释边界

[Jain & Wallace, 2019](https://aclanthology.org/N19-1357/) 与 [Wiegreffe & Pinter, 2019](https://aclanthology.org/D19-1002/) 讨论 attention 解释的不同检验条件。这些 NLP 研究不能证明本项目 attention 必然无效，但说明仅展示权重不足以完成机制解释，需要明确基线与干预。

[ERASER, ACL 2020](https://aclanthology.org/2020.acl-main.408/) 将预测随信息移除的变化用于解释忠实性检验。本项目借用同输入、固定预测器、明确干预的比较原则；不把其 NLP 结论直接当成 RL 回报因果定理。

[Interaction Networks, NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html) 支撑关系消息建模动机，不保证当前静态 lane 关系就是动作所需的动态冲突关系。[Safe RL via Shielding, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11797) 则有助于明确内部 attention mask 与动作约束属于不同层次。

本轮方案的科学约束是：记录可观测的模块输入、学习参与与动作依赖，再与预声明评估对照；没有运行过的性能实验、没有记录的轨迹反事实以及没有训练过的解释模型，均不写成已验证结果。
