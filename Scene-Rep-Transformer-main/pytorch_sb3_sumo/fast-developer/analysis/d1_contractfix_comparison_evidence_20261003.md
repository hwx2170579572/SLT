# D1 contractfix 对照证据摘录（2026-10-03）

本摘录供本轮结果归因整合使用，不替代原始研究记录。两条修复臂的final-100均已通过身份、checkpoint SHA、100个validation seeds和reward核验：新ST为 **45/50/5**，新ST-RT为 **52/48/0**（均为S/C/T）；新ST-RT共26218 raw、8773 decisions。所有正式结果均为sorted/depart4、单训练seed 0、fresh 100k raw、同一100个validation seed（10000–10099）；这不是多训练种子稳定性结论。

## 同场景结果与解释边界

| 方法 | 成功/碰撞/超时 | 当前证据能支持的判断 |
|---|---:|---|
| SAC+MLP | 22/38/40 | 纯RL基线有大量超时；原超时组常见高目标速、低实际速停滞。 |
| 旧 ST | 50/50/0 | 相对SAC超时消失、成功增加，同时碰撞上升；支持完成/行动能力改善，不证明安全让行。 |
| 联合C8/C9修复 ST | **45/50/5** | 相对旧ST成功低5点、碰撞持平、超时多5点；两项修复联合且单seed，不能归因到C8或C9任何一项。 |
| 旧 ST-RT | 63/37/0 | 这条历史增量链的最高成功率；相对ST成功+13点、碰撞−13点。路线表征没有独立真实意图监督。 |
| 联合C8/C9修复 ST-RT | **52/48/0** | 相对旧ST-RT成功低11点；与新ST比成功+7点、碰撞−2点、超时−5点。仍是联合修复、单训练seed的描述性比较。 |
| ST-RT+Topo | 39/39/22 | 增加了多条拓扑路径后成功下降、超时出现；use_topology捆绑lane编码/query、关系边和goal residual，不能归因于单一分支。 |
| Topo+route-aware | 43/32/25 | 碰撞下降伴随更多超时；goal注意力合法候选不约束连续动作。25个超时均终止在下一route edge不可接续的`-E1_0`。 |
| ST-RT+线性3slot | 34/49/17 | 当前线性槽头退化，但此实现同时减少活跃读出参数并移除非线性，不能据此否定槽语义。 |
| ST-RT+Topo+线性3slot | 41/56/3 | 较少超时不等于更多成功；同时启用Topo与槽的结果不能拆分各自效应。 |
| ST-RT+goal-only Topo | 24/47/29 | 只保留可达goal读出也未恢复ST-RT。对同一冻结checkpoint关goal分支得45/55/0、增加路线换道veto得44/56/0；两者将29个原超时转为成功或碰撞，仍未证明整体安全收益。 |
| ST-RT+参数匹配非线性3slot | 50/48/2 | 相较旧线性槽头成功+16点、超时减少；仍较ST-RT少13点成功。支持读出容量/非线性是旧头混杂，未证明槽语义分工。 |
| RouteAct | 47/53/0 | 从零训练的执行映射方法低于旧ST-RT；冻结旧ST-RT的9003个决策中veto为0，冻结部署不触发不能替代从零训练结果。 |
| ConflictTiming | 40/57/3 | 增加20维冲突信息与4896参数后没有任务提升；功能和梯度接通不能说明该特征改善通行。 |
| Longres | 39/55/6 | bounded speed residual分支确实获得梯度和参数更新，但固定评估未改善总体结局；与旧ST-RT的差异还包括新critic条件路径及联合训练。 |

结果来源：`analysis/sorted_module_attribution_20261002.md`、`analysis/sorted_task_module_attribution_20261002.md`、`analysis/sorted_routeact_conflicttime_attribution_20261003.md`、`analysis/experiment_history.md`、`analysis/intersection_experiment_summary_20261001.md`。新修复ST原始结果文件位于`runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0/evaluation_results.json`。

## 配对validation结果迁移

以下按evaluation记录中的逻辑seed 10000–10099连接；每对100/100匹配，traffic_variant均无差异，终局标记互斥且other/off-route为0。格内按来源方法结果行、目标方法列计数。相同seed和traffic模板只说明固定checkpoint评估条件相同，不表示后续动作或交通轨迹相同。

| 配对 | 来源\目标 | S | C | T | O |
|---|---|---:|---:|---:|---:|
| 旧ST→新ST | S | 22 | 25 | 3 | 0 |
|  | C | 23 | 25 | 2 | 0 |
|  | T | 0 | 0 | 0 | 0 |
|  | O | 0 | 0 | 0 | 0 |
| 旧ST-RT→新ST-RT | S | 32 | 31 | 0 | 0 |
|  | C | 20 | 17 | 0 | 0 |
|  | T | 0 | 0 | 0 | 0 |
|  | O | 0 | 0 | 0 | 0 |
| 新ST→新ST-RT | S | 20 | 25 | 0 | 0 |
|  | C | 29 | 21 | 0 | 0 |
|  | T | 3 | 2 | 0 | 0 |
|  | O | 0 | 0 | 0 | 0 |

旧ST→新ST的23个collision→success与25个success→collision并存；新ST的5个超时来自旧ST的3个成功和2个碰撞。旧ST-RT→新ST-RT有20个collision→success、31个success→collision，净成功由63降至52。新ST→新ST-RT有29个collision→success、25个success→collision；新ST原5个超时中3个成为成功、2个成为碰撞，合计净成功增加7、碰撞减少2、超时减少5。以上只是同一评估池的策略结果迁移，不能按每个配对给C8/C9或RT分配因果贡献。

原始episode记录位置：旧ST `Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json`；新ST `runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0/evaluation_results.json`；旧ST-RT `runs/sortlr_1003_retry01/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json`；新ST-RT `runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0/evaluation_results.json`。新ST-RT的总步数、identity和SHA核验收据位于该run的`training_complete.json`、`runtime_provenance.json`、`evaluation_results.json`与pair audit。

新旧ST比较的主要协议字段一致：场景`intersection_sorted`/depart scale 4.0、seed 0、从零100000 raw steps、learning starts 5000、batch 32、LR 1e-4、replay 20000、gamma .99、action repeat 3、base env及相同六项奖励配置、validation seeds 10000–10099。旧ST的`arguments.json`记checkpoint frequency 200，新ST记10000；新ST还开启被动behavior/shadow/trajectory诊断，不增加仿真步。旧ST的运行目录没有source archive/runtime provenance或同口径新式梯度/trajectory记录；新修复的设计目标只改C8/C9，但无法逐文件证明旧ST的as-run源码完全相同。因而45/50/5对50/50/0是同预算、同池、同seed的历史checkpoint描述比较，不是隔离并估计两项修复因果效应。

新协议确认D1-ST与ST-RT encoder均为1,057,348个参数，state-dict键与初始张量相同，构造后随机数状态相同；修复没有新增可学习参数。共享encoder仍按旧SAC配置由critic TD loss训练，actor侧features detach；optimizer、replay、奖励、bootstrap及timeout协议不变。新ST训练summary在96个critic梯度/更新抽样点中，state、spatial、temporal和social活跃子模块参数梯度覆盖率均为1且实际参数变化；旧ST没有对应日志可作数值梯度差比较。新source archive为20个文件，runtime列出的11个源文件SHA与归档匹配；启动receipt及方法归档见`runs/d1_contractfix_20261003/`。协议与参数核验见`analysis/d1_contractfix_protocol_20261003.md`。

## C8/C9实际暴露及其解释

新ST的审计包装器只统计正常真实decision的pre-action observation，不计replay重复或shadow forward。`diagnostics/train/trajectory_history_audit.summary.json`记录100000 raw、33484真实决策、errors=0；旧count-minus-one反事实在ego 33484个历史中没有错位，在167420个social历史中有113个错位（61选padding，52选更早有效帧，0.0675%）。Final eval为8644个决策、100个完整回合；ego仍为0/8644，social为22/43220（12选padding、10选较早有效帧，0.0509%）。掩码遵循生产`x!=0` proxy；错位数字是旧实现会错选，不表示修复后仍错，也不是性能影响率。

几何边在新ST和新ST-RT均active，故ST不可写成geometry NA。C9仅将显式几何边的速度恢复到Cartesian基底，保留原观测状态token。新ST的train pair-history审计有4,894,715个同历史时刻actor-pair观测，其中2,505,431个closing符号相对旧基底改变，310,345个由旧零变为新非零；eval对应1,269,115、654,267、77,025。重复历史pair是重复观测而非独立交通事件。该结果证明几何输入数值确有变化，不能证明成功率由C9改变。geometry指标是constant-velocity closest-approach proxy，不是真实TTC；零相对速度时closest-approach time未定义。

审计来源：`runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0/diagnostics/{train,eval}/trajectory_history_audit.summary.json`；C8/C9合同、hash与限制见`analysis/d1_contractfix_protocol_20261003.md`。旧D1 count-minus-one适用路径和掩码局限见`analysis/last_valid_history_index_audit_20261003.md`。

## 任务难点与当前归因

证据支持的任务瓶颈分两段：从SAC+MLP到ST主要是从停滞/超时转为行动，但碰撞没有同时下降；从ST到ST-RT，路线条件表示与更高成功率相伴。Topo/goal-only诊断显示，路线目标attention筛选和执行车道选择之间仍有缺口。冻结goal-only策略的route veto可消除29个观察到的超时，却产生20个碰撞；goal分支关闭产生19个碰撞。局部干预说明解除堵停足以改变这些固定checkpoint续行的结局，更多决策难点转向选择安全冲突间隙、进入时机和清空路口。它们不是总体成功率、独立事件率或训练消融估计。

Longres把branch输出限制为pre-tanh速度均值的±0.2，因此目标速度最大差约`10*tanh(0.1)=0.99668 m/s`；训练中96次actor与critic诊断均观察到梯度和参数更新，16个有效训练同状态shadow比较里横向输出差为0。独立decision-window实验以0/6m/s强制目标速度测29个事先选定窗口，共174个干预决策，其中140次目标与策略建议速度差超过0.99668m/s。这些较大的强制改变能救回部分窗口，但无法证明Longres可学出同等改变，也不能把失败说成不可避免。窗口诊断累计85 episodes/22944 controlled raw；不得恢复或扩展其预算。来源：`analysis/longres_training_diagnostic_audit_20261003.md`、`analysis/decision_window_intervention_results_20261003.md`。

## n-step/timeout协议与后续最小公平比较

当前实现使用4个decision-transition窗口：奖励累加为`r0 + γr1 + … + γ^(k−1)r(k−1)`，但bootstrap端无论实际窗口长度`k=1…4`都只乘一次`.99`；标准k-step候选应按实际k乘`γ^k`，完整4步为`.96059601`。短尾要使用各自实际k。timeout在600个0.1秒raw ticks（60秒）后作为`truncated`，时间惩罚−5仍包含在reward return并继续bootstrap。源码注释表明单gamma是为兼容既有SAC行为的有意选择；项目审计尚未独立追到特定上游release revision，故称协议差异而不称已验证的上游bug。Intermediate entropy/importance correction没有加入replay n-step回报，本身不构成此审计确认的缺陷。

先由任务定义确定60秒是内在deadline还是外部采样上限：若deadline属于任务，应作为termination处理并将remaining time输入观察；若仅为外部截断，应保留truncation及末状态bootstrap。之后如需评估折扣指数，另开一版匹配SAC/D1实验，只比较旧one-gamma与按实际k的`gamma^k`，统一任务终止语义、reward、scene/traffic pool、seed、raw预算、updates和final validation；不可热改当前ST/ST-RT pair。离线目标probe已验证了算术和边界；本任务未授权新的完整训练，因此这里是待授权的下一步设计。

来源：`analysis/bootstrap_protocol_audit_20261003.md`；官方任务时限说明及Pardo et al.见下方链接。

## 理论来源的有限论断

- [Interaction Networks (NeurIPS 2016)](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html)：支持以对象和关系图进行交互推理的设计动机；原论文物理动力学任务不证明当前lane relation适用于路口任务或改善回报。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 与 [Graph Attention Networks](https://arxiv.org/abs/1710.10903)：支持序列/邻域内以attention进行加权表征读取；attention候选mask不约束策略最终动作。
- [Safe Reinforcement Learning via Shielding (AAAI 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11797)：讨论在形式化时序逻辑约束下监视并修正动作的shield；本地route mask和路线veto不因此拥有论文中的形式化安全保证。
- [Soft Actor-Critic (PMLR 2018)](https://proceedings.mlr.press/v80/haarnoja18b.html)：支撑最大熵RL中同时优化回报和熵；本项目actor-detach/critic-TD encoder路径由本地源码决定。
- [Gymnasium time-limit handling](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) 与 [Pardo et al., Time Limits in RL](https://arxiv.org/abs/1712.00378)：支持finite-horizon deadline需观察剩余时间、外部truncation应bootstrap；它们无法替项目决定60秒在该任务中的定义。
- [SMARTS coordinate documentation](https://smarts.readthedocs.io/en/latest/api/smarts.core.coordinates.html) 与 [SUMO TraCI vehicle-state documentation](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html)：支持核对heading/坐标约定；结合项目存储伪速度`(s cos h,s sin h)`和`h=-SUMO-angle`，实际east/north Cartesian速度为`(-s sin h,s cos h)`。该换算支持C9的输入合同修复，不支持任何性能因果结论。

## 核心来源索引

- 研究总上下文：`RESEARCH_CONTEXT.md`。
- 历史结果、身份、奖励与源码限制：`analysis/experiment_history.md`、`analysis/intersection_experiment_summary_20261001.md`。
- 模型路径、参数/梯度路径和理论边界：`analysis/full_mst_slt_implementation.md`、`analysis/sorted_module_implementation_theory_20261002.md`。
- 任务结果与失败链：`analysis/sorted_module_attribution_20261002.md`、`analysis/sorted_task_module_attribution_20261002.md`、`analysis/sorted_routeact_conflicttime_attribution_20261003.md`。
- 当前修复与新旧限制：`analysis/d1_contractfix_protocol_20261003.md`、`analysis/last_valid_history_index_audit_20261003.md`。
- Longres、固定窗口和backup协议：`analysis/longres_training_diagnostic_audit_20261003.md`、`analysis/decision_window_intervention_results_20261003.md`、`analysis/bootstrap_protocol_audit_20261003.md`。

## 旧ST final-100失败字段边界

旧ST的`evaluation_results.json` identity指向`final_model.zip`，SHA256 `247e683657d224a549fbe8393189b13e6ea2db4baa53618233c9a11d6c0d389f`，与`training_complete.json`一致；记录100回合、scene `intersection_sorted`、depart_scale 4.0。原始episode种子覆盖10000–10099，结果为S/C/T/O=50/50/0/0，终局标记互斥、每回合raw return为+1或−1。旧ST的`arguments.json`记seed 0、100000 raw、CUDA、learning starts 5000、batch 32、LR 1e-4、buffer 20000、gamma .99、action repeat 3。记录没有旧代码source archive或runtime provenance，不能逐文件重现as-run源码。

`episode_records`只有`episode/seed/episode_return/decision_steps/environment_steps/raw_steps/completion_time_seconds/success/collision/off_route/timeout/traffic_variant`。50个成功回合的完成时间均值/中位数/范围为26.162/25.85/18–42.1秒；decision steps为87.52/86.5/60–141，raw steps为261.62/258.5/180–421。50个碰撞回合的记录长度为74.94/75/40–122 decisions、224.12/223.5/120–366 raw steps；其`completion_time_seconds`为null，故raw长度只代表从episode开始至终局经过的仿真步。30个traffic variants中，按每个variant内碰撞数分组为0个碰撞4种、1个7种、2个14种、3个5种；这是复用交通模板池中的结局分布，不说明路线、时序或碰撞机制。

此旧ST run没有`diagnostics`目录或逐decision评估trajectory、速度、道路/车道、route eligibility、几何关系、collision partner/contact等字段；目录保存的其他材料是训练监视/奖励/状态、权重及route overlay。没有找到同一个final checkpoint的冻结物理状态诊断记录。`train_monitor.csv`属于训练阶段，不用于补写这50个final evaluation碰撞的原因。故四方法之间当前可比的是S/C/T/O终局迁移；旧ST无法参与道路/车道/相位、停滞时长、速度、接触车辆的同字段原因分布比较。

旧ST身份与episode文件：`Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/{arguments.json,training_complete.json,experiment_manifest.json,evaluation_results.json}`。相关结果上下文：`analysis/continuous_100k_mst_vs_st.md`、`analysis/sorted_module_attribution_20261002.md`、`analysis/experiment_history.md`。
