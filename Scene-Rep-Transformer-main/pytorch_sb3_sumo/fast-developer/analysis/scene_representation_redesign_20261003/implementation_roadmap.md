# 通行事件图表征：分阶段实现路线 v1

日期：2026-10-03。用户本轮要求：把已讨论的场景表征研究方案落实为完整的阶段、工作清单、技术/模型与验证标准。本文定义研究与验收路线；当前独立入口已有协议、replay和通用SAC接入骨架，但没有已验收的场景encoder/事件模型或任务环境集成，也未据此声称任何性能效果。

执行状态更新：用户已授权按[独立实验协议](experiment_protocol.md)逐阶段实现并验收，未通过的实现检查需修复。验收状态单独记录于[stage acceptance ledger](stage_acceptance.md)；计划获授权不代表任何实现或实证门已通过。

前置材料：`technical_route.md`、`environment_evidence.md`、`papers.md`、`review_notes.md`。此前学习机制优化路线保留备选，见 `../reward_literature_technical_route_20261003.md`。

## 1. 最终系统与实现范围

目标：在 `intersection_sorted_depart4p0` 上，通过公共通行拓扑、车辆历史与进入/占用/清空事件，获得服务闭环决策的场景表示。encoder 可重构；SAC 算法、动作接口保留，reward/replay 优化不作为本路线创新。

在线输入始终是 `O_≤t + public_map + ego_navigation`，首版输出 `z_t ∈ R^128`，actor 使用 `π(a|z_t)`，双 critic 使用 `Q1(z_t,a), Q2(z_t,a)`。真实未来 route、后续动作、碰撞结果和未来发车表不得进入在线输入。相比上一轮举例的256维，本计划将128维作为更窄的初始接入点；256维仅作后续容量控制，所有同期比较方法使用同一维度。

完整系统分为：

1. 统一观测与历史缓存。
2. 公共地图的通行路径—冲突区域拓扑。
3. 车辆因果时序编码与 polyline 地图编码。
4. 路径进度/占用事件分布（先确定性，后学习）。
5. 模式内部车辆—区域注意力与沿路径消息。
6. 保留模式关系的确定性 readout。
7. SAC policy/critic、训练采集与诊断。

SAC 从基础版本就接入，不能等全部模块完成后才检查闭环。

## 2. 四个版本，而不是一次实现完整复杂模型

| 代号 | 建议新增方法名（尚未注册） | 实质变化 | 研究角色 |
|---|---|---|---|
| M0 | `sac_scene_dualgraph_v1` | 同输入的车辆图与车辆—地图 attention | 新观测合同下的通用表征对照 |
| M1 | `sac_scene_eventgraph_cv_v1` | 加入确定性路径进度与通行事件图 | 最小事件表征版本 |
| M2 | `sac_scene_eventgraph_pred_v1` | 同路径进度分布和事实事件监督 | 检验事件可预测性及辅助表征学习 |
| M3 | `sac_scene_eventgraph_joint_v1` | 场景共享模式、联合似然、模式内部事件消息 | 完整候选方法 |

方法名只是计划，不能写入实验表当作已经运行。旧 SAC+MLP、MST+SLT、修复 ST/ST-RT 的代码/结果均保留。M0 并非 GraphAD 官方复现；它只借鉴通用异构图结构。最终论文应另有同信息/同监督的近邻适配对照。

## 3. 阶段 0：冻结问题定义与 SAC 工程协议

**目的**：确保后续差异能够归因于表征。

工作：
- 固定场景/traffic SHA、ego任务路线、reward版本、动作映射与 repeat、训练raw预算、评估身份、checkpoint规则。
- 记录 raw physics step、policy decision、n-step transition 与 update 的单位，不能用100k raw冒充100k decisions。超时/提前terminal时使用实际步数。
- 用户已确认60s是ego任务截止：暖机不计入任务时间；episode控制开始时剩余时间为60s，按实际执行的raw physics ticks递减，并作为policy/replay观测的一部分。0.1s/raw、action repeat最多3（通常0.3s/decision），n-step=4按policy decision计。deadline内只剩1–2个raw tick时缩短最后一次动作重复；它仍是一个decision transition，raw tick数另行记录。100k raw约33.3k常规decision，实际计数以日志为准。
- 到达60s deadline时，保留timeout outcome/reward标签，但Gymnasium边界必须是`terminated=True, truncated=False`，不做价值bootstrap。外部采集/runner中断是`truncated=True`：n-step序列在末状态截断并从最后真实terminal observation bootstrap；不得把自动reset observation当成截断末状态，也不得将外部截断改标为timeout失败。
- n-step目标使用实际累计的policy decision数`k`（`1≤k≤4`）：环境奖励为`Σ_(j=0)^(k−1) gamma^j r_(t+j)`，bootstrap折扣为`gamma**k`，不按raw tick或动作重复数再指数折扣。按独立协议保留reward-only多步累加加末端soft bootstrap；不累计中间熵项、不做importance-sampling/Retrace修正。这是明确的近似off-policy目标，不称为精确标准soft n-step评估器。不得在场景表示实现中悄悄引入另一学习目标。
- 审计后冻结一个 SAC 协议版本。若与历史实现不同，M0及之后全部使用新协议，历史表保留“legacy protocol”标注；不把跨协议差异计作encoder贡献。
- 明确共享encoder的actor梯度合同，初版沿用当前 actor feature stop-gradient；确认target critic包含需要同步的encoder参数。

技术：现有 PyTorch SAC、数值 Bellman target 单元检查、Gymnasium terminated/truncated 合同、配置/源码SHA归档。

产物：不可变 `scene_event_sac_deadline_nstep_v1` 配置、数据字典、target数值测试、比较矩阵。阶段0只做协议、人工transition和单元检查，不运行训练或SUMO；M0通过模型检查后再执行阶段3的有界真实SUMO smoke。时间限制依据：https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ 。

进入下一阶段：协议无歧义、目标计算通过有限人工transition测试。这里没有“先训练100k看看”的必要。

## 4. 阶段 1：观测、历史和数据采集合同

**目的**：提供可靠的物理状态，使图建模不再依赖位置为零等隐式mask。

工作：
- 新建独立 `SceneObservationV1`：actor历史、有效mask、真实时间戳、当前车身尺寸/朝向/Cartesian速度、可估计加速度及可信标记、当前lane匹配；ego导航与60s任务剩余时间单独提供。剩余时间随真实raw ticks减少，并同时存在于online、序列化与replay observations。
- tracking key只用于缓存和标签join，不嵌入网络。排序/截断不能利用未来路线、发车顺序或车辆名字。
- 初始工程上限：80m、最多24 actor含ego、过去2.0s含当前共21个0.1s采样点；这些是起点，需先检查覆盖和开销。正常0.1s仿真推进中采集，不为补历史额外调用simulationStep。
- 明确左补、右补、内部缺帧、全空、首次出现、驶出范围的mask；位置x=0可合法存在。加速度从过去估计，缺历史时不编造。
- 地图、ego路线与actor候选path仅基于公共信息。未来才入场但尚不可见的车辆属于unknown覆盖范围，不读取spawn schedule。
- 新replay存原始结构化observations或无损frame引用与map hash，不能缓存训练早期的z。future标签由独立pending队列在正常后续观测后补齐，只送aux loss，不写入policy observation。
- 按episode/traffic划分预测训练与验证数据，连续帧不得随机拆入两侧造成泄漏。

技术：TraCI/sumolib只读状态、坐标转换、ring buffer、NumPy分块数组/现有数据序列化、显式布尔mask。新依赖只有确需时才引入。

模型接口：`actor_hist[B,N,T,F]`、`history_valid[B,N,T]`、`actor_valid[B,N]`、`map_polylines`、`map_relations`、`ego_route`；F在数据字典确定，不用占位通道暗藏标签。

验收：在线/序列化回放观测一致；缺帧与旋转解析检查；标签对齐到生成动作前的state；actor截断、短历史、未覆盖入口有统计。不是要求80m/24车永不漏观测，而是漏失可解释且不伪装成free。

## 5. 阶段 2：通行拓扑与冲突区域编译器

**目的**：给出“可能在哪里发生关系”的正确静态结构。

工作：
- 从 `map.net.xml` 构建lane后继、合法换道、内部连接器和exit关系；枚举有预算的局部合法movement corridors。
- 根据车辆扫掠几何求crossing/merge区域，保存每条corridor在区域上的进入/清空弧长；静态候选区域可缓存，尺寸相关映射按actor更新。
- 显式构造沿movement的区域先后边、共享区域关系；保留跟驰和局部侧向关系，避免资源图漏掉追尾/换道。
- 区域分区使用规范化几何和可复现的版本/hash，不将绝对lane/zone ID变成可学习捷径。
- 处理同一路径上的区间交叠、粗区域误报、弯道车身扫掠、离散采样跨过边界等情形。地图foe标记作线索，不代替几何验证。

技术：sumolib地图解析、有限深度图遍历、弧长/Frenet投影、polygon/swept-footprint相交；可用现有几何函数或已安装Shapely，初版不训练地图感知网络。

产物：`PublicMapCache`、`MovementZoneCompiler`、图可视化、解析几何用例。

验收：交叉、平行、错时交叉、汇入、追尾、静止和多尺寸用例符合合同；区域覆盖不足有unknown/局部几何兜底。资源重叠是冲突代理，不标成真实碰撞概率。

阶段1与2可并行开发，阶段3前必须完成共同schema。

## 6. 阶段 3：M0 通用双图 + SAC

**目的**：建立足够强、能够正常闭环的同输入对照。

工作与模型：
- `TemporalActorEncoder`：1–2层causal relative-time Transformer，mask显式；宽128、4头为起点。GRU只作为简化替代/消融，不两者一起叠加。
- `PolylineMapEncoder`：共享point MLP + masked pooling（VectorNet式局部编码思路），输入几何、切向与合法语义。
- `DualGraphEncoder`：actor↔actor和actor↔map的relative relation attention，residual + LayerNorm；lane连通/ego导航作为结构条件。
- `SceneReadout`：ego/ego-route query聚合到固定128维z；保留ego运动与任务状态。
- 接入SAC双Q及actor；target encoder纳入同步。对同一冻结权重，online与replay forward结果一致；读取下一状态不穿越episode reset。
- 三条forward单独验收：当前Q用online encoder重算replay的原始`o_t`；target action由当前actor及其online encoder在`o_(t+k)`上生成；target Q由独立target encoder重算同一`o_(t+k)`，全程no-grad。即`a'~π(·|fθ(o_next))`、`Qbar(fbarθ(o_next),a')`；target中不能误用在线z。Polyak/硬拷贝参数集合覆盖新增temporal/event/mode参数，公共地图缓存保持同一只读版本。
- 相同输入下的简单MLP/DeepSets作为信息扩展检查，旧5邻车模型只作历史参考；不要用不同观察数量证明图的作用。

工程：小图可先用PyTorch padded dense tensors + masks，无需先安装图学习框架。核验实际PyTorch版本API及不同attention API的bool-mask语义；首版encoder dropout=0，避免将encoder随机性混入online/replay一致性测试。SAC actor本身的采样随机性保留，复现检查比较分布参数/确定性mean。

产物：M0独立入口与配置，forward/backward/target同步检查，有限smoke计划（例如500–2000 raw，只验证管线，不能当性能结论）。**当前未执行smoke。**

验收：无NaN/错位/未来泄漏，actor/critic/encoder更新合同正确，episode回报六分量对账，正常采集可用，GPU时间/显存可接受。M0不必达到某成功率才算实现正确。

## 7. 阶段 4：M1 确定性通行事件图

**目的**：先检验事件语义与关系结构，无需复杂预测器。

工作与模型：
- 沿每条由当前状态/公开地图得到的合法候选，计算投影进度与速度，以恒速CV为主基准；恒加速度CA只作为比较，不同时改变控制规则。
- 对停车、lane匹配失败、horizon以外、候选不完整，输出相应valid/unknown，不用0填充成安全。
- 从同一个进度过程映射多个zone的entry、clear、occupancy和沿path关系。**“确定性/单运动假设”不表示知道邻车真实future route**；候选仍分支表达，不能当作同车同时走多条路。
- CV/CA分别作用于每条合法候选；M1不估计候选概率，不擅自赋均匀概率，也不平均路线坐标。Readout保留带actor归属的互斥分支集合及ambiguity标记，不能把分支数当成多辆车累加占用。
- `EventGraphEncoder` 使用 actor—movement—zone 的关系attention与沿path顺序消息；局部几何/跟驰分支保留。时序编码和z维度与M0相同。
- 先做合成可辨识性检查，再在获授权的首轮双worker训练中比较M0与M1。

技术：CV/CA运动外推、几何事件投影、factor/bipartite graph attention、相对时间/关系类型embedding。

产物：M1；图覆盖、时间字段有效率、冲突/非冲突关系分层、建图开销报告。

首轮比较解释：M0→M1同时引入事件语义与消息结构，因此是“事件表示整体”比较，不能声称只改了一条边。后续加 `M0 + 同事件特征但通用消息` 控制，分离派生特征与图结构。

继续条件：几何合同正确，存在与任务相关且被读取的事件信息；若性能无提升，先区别CV预测失真、覆盖缺口与事件关系本身无效。不是每个中间版本都必须单seed涨成功率，但也不能无证据直接加复杂模块。

## 8. 阶段 5：M2 学习路径进度与事件表征

**目的**：处理恒速假设解释不了的停车、起步和到达时间变化。

工作与模型：
- `ProgressTransitionHead`：小型MLP参数化沿路径的前进/停留转移；必要时把speed bin纳入状态，形成受运动范围限制的离散进度过程。先限定3s预测、0.3s栅格，只有远期校准/资源预算允许才扩为6s；这是对原6s工程起点的保守细化。
- `EventDistributionProjector`：通过确定性forward递推，从同一进度分布求各zone first-entry/clearance/occupancy；跨区域条件关系随进度过程保留，不为每zone独立回归一个ETA。
- `PassageLabelCollector`：从正常后续实际观测生成标签；区分完成、出视野、车辆消失、碰撞移除、episode终止、时域截止。保留已观测entry，即使clearance被删失。
- 未来实际路线也只能由后来观察到的lane/位置前缀确认，不能查询隐藏route列表当真值。观察前缀尚不能区分两条候选时，对仍兼容的路线集合边缘化，不强行给唯一route类别；候选外实际行为与尚未判明是不同标记。
- 训练 masked/censored progress/event likelihood；辅助occupancy BCE可用，但只能校准边际。报告NLL、Brier、entry/clear误差/区间覆盖，按policy阶段、时间跨度、进入前/区内分层。
- M2起点为每候选route单一条件运动分布，K=1；不强迫多个假想意图。

梯度合同：SAC critic可更新普通历史/图/message/readout；aux loss更新预测头及约定的表征stem。RL forward对校准用的route概率、event分布停止**直接**critic梯度。若共享stem，critic仍能间接改变预测，这是需要测量的多目标耦合，不能宣称detach保证校准。记录梯度尺度/夹角与校准漂移；私有预测encoder或阶段性冻结只能作为有依据的后续消融。

首版更新次序明确为：同一训练batch计算`L_Q + λ_pred L_pred`，对共享encoder一次反传/一次参数更新；预测头由L_pred更新，critic head由L_Q更新。共享encoder只属于一个optimizer，禁止既被critic optimizer更新、又被aux optimizer重复step。Actor使用detach后的当前特征独立更新，温度优化保持SAC原合同；target encoder/critic按冻结协议更新，验证新加入的预测/模式参数也在相应target拷贝或同步范围内。若有效future标签尚未成熟则mask/跳过该部分aux，不能补假标签；记录实际aux有效样本数和更新次数。不能直接照搬旧full中先aux再TD、共享encoder被两次优化的顺序而不声明。

具体张量边界：预测模块输出`Ξ`（进度/事件分布）、route probabilities `ρ`和后续M3的`π`，SAC分支使用`stopgrad(Ξ,ρ,π)`。可训练的`EventTokenMLP`再把这些数值与静态关系编码为event embeddings，`EventGraph/Readout`基于它们和可训练的历史/地图stem特征产生z；它们的参数可接收TD梯度。不要直接把预测头内部hidden state或shared mode query当额外z通道绕过该边界。

| 参数组 | 允许的损失梯度 |
|---|---|
| 历史/地图/公共上下文stem θ | TD与预测aux，一次合并更新 |
| 路线/进度/场景模式预测头 η | 预测aux；无直接TD路径 |
| EventTokenMLP/EventGraph/Readout φ | TD |
| critic最后双Q heads | TD |
| actor最后分布heads | actor loss；对z stop-gradient |
| target参数 θbar/ηbar/φbar/Qbar | 无反传，仅协议规定的同步 |

这是主实现合同，不证明训练一定稳定；共用stem导致的间接校准变化仍需记录。辅助头以外模块若需aux监督，作为单独改动登记。

关键公平对照：同图有/无aux；**同一预测器及同标签预算**分别接通用图与事件图；统一参数/FLOPs/在线延时。额外离线预训练若存在，须给对照同等数据/预算并单列，不再称整体模型无预训练从零。

进入M3条件：M2相对CV/CA有可靠预测改善，且联合关联确有信息价值；若单模式已足够，则可停在M2，不为复杂而复杂。

## 9. 阶段 6：M3 共享场景模式的联合事件图

**目的**：检验“同边际、不同联合通行条件”的表征问题。它是论文核心候选，尚不是已证实创新。

首版不采用指数级joint beam。用置换不变的scene encoder和K个shared scene queries：先实现同架构K=1控制，再在证据支持时试K=2、上限4。每个query观察全部当前actor/map/event上下文，产生场景权重π_k，并条件化每车的合法路线分布及route内进度转移。M2不一定包含shared-query decoder，因此M2→M3整体差异不能自动解释为模式数的作用。

近似模型：

`p(Y_all|H,G) = Σ_k π_k ∏_i [Σ_(r∈C_i) p(r|h_i,G,q_k) p(Y_i|r,h_i,G,q_k)]`。

这表示一个场景共享latent下的条件独立近似，而不是独立actor模式按编号拼接。整个scene采用一个混合似然：

`L_joint = -logsumexp_k(logπ_k + Σ_i log p(Y_i|q_k,H,G))`。

这里`p(Y_i|q_k,H,G)`就是上式的route求和。候选集合包含归一化unknown/out-of-set分支；匹配失败或真实后续不在可见候选中不能令全部路线似然为零。Y包括有效事实和删失约束，对未观察到的部分积分，使用相应survival/censor likelihood；不把出视野或尚未清空简单标为未占用。所有未来量仅作监督。

工作：
- `SharedSceneModeDecoder` 用scene-level query cross-attention；mode无预定义“激进/保守”标签，不通过熵奖励硬分裂。
- actor、route、zone token保留 `(actor, route, mode)` 分支身份；同actor不同route互斥，不能对其概率乘积计算“同时占用”。候选集合只来自当前状态/公共地图，真实后续脱离候选要记out-of-set/unknown。
- 每个mode内先编码每条route的事件序列和车辆—zone关系，随后才做route/mode集合读出；不要先对轨迹坐标、actor边际概率做均值后再构图。
- 分支关系具体定义为：先在`(actor,route,k)`内部生成同路径事件和顺序embedding；两个不同actor的zone pair特征可计算`Σ_(r_i,r_j) p(r_i|q_k)p(r_j|q_k) φ(e_i^(r_i,k),e_j^(r_j,k))`，即在给定mode的独立近似下对条件关系做确定性边缘化。不是把互斥route当同时存在的物理车辆，也不是先平均坐标。保留路径条件/跨区过程特征进入g_k；这种有限阶消息仍可能丢失高阶联合信息，不能宣称无损恢复联合未来。
- 使用确定性的条件分布/转移核表示Ξ，不在online或replay随机采未来轨迹。读出 `[mode weight, event embedding, unknown mass]` 到相同z维度。
- 测试actor/route/mode重排、同obs重复计算、padding和unknown、合成联合反例；mode-before-graph平均、跨zone结构打乱作为针对性消融。
- 检查场景联合NLL、模式覆盖及条件依赖；边际NLL变好不是联合关系已学会的证据。K1/K2/K4比较记录新增参数/计算。

重要近似：给定mode仍假设actor残差独立，因此所有有正概率的跨actor路线组合可能有质量，并未强制跨actor社会/任务相容性；单条路线合法不等于联合世界物理自洽。消息网络和有限维readout也可能丢联合信息。若数据证明剩余路线依赖重要，再研究显式joint beam/更强joint decoder；它不是本轮首个完整版本的前置条件，也不能强制背景车避碰以制造安全预测。

action边界：预测来自行为策略实际未来；不代表未执行动作后果。可另用 `D(z_t,a_t)` 监督一步事实事件，但不作为本轮必加组件，也不称反事实world model。固定三路直行场景先验证时间/起停关联，路线意图能力须在相应场景才有证据。

## 10. 阶段 7：正式闭环证据、消融和泛化

主场景继续使用sorted/depart4，开发首轮seed0从零100k raw。两worker执行配对版本，不并行运行相互依赖的未验证新模块。

建议依次：

1. M0 vs M1：事件语义/结构是否值得继续。
2. M1 vs M2：学习进度/事实监督是否改善CV局限。
3. 同架构M3(K=1) vs M3(K=2或4)：共享多模式是否提供额外价值；M2作为已有整体方法参照。必要时先比较M2与M3(K=1)以识别decoder结构变化。

上述是未来获授权后的实验顺序，**不是已启动的六路训练**。对照结果若已在同代码/协议/数据条件下有效，可复用，不机械重复训练；改了公共观测/训练协议/预测器数据时必须重新确认可比性。

逐步补齐最小证据：同输入MLP/Set、通用双图、MST+SLT适配对照、修复ST-RT；同预测器不同图、同图无aux、独立zone头替代同路径过程、mode先平均、去跨区关系、局部跟驰分支、关键预算/超参数敏感性。参数量/计算匹配是敏感性控制，不能宣称所有不同图天然等参数。

论文阶段再做多训练seed（例如3–5）、锁定的未反复调参held-out traffic、不同junction几何、背景转向组合/密度/速度与驾驶行为、有限观察噪声/范围。训练与泛化数据须定义覆盖，不能要求只见三条直行的模型在未学习路线概率上无依据地泛化。

成功率为主要任务指标，碰撞/超时互斥统计，同时报告效率、进入/清空过程、停滞、几何偏离、学习曲线、显存与建图+推理延时。预测指标改善不能代替闭环S/C/T。

原validation seeds 10000–10099已用于开发，不能再当未见测试集。100回合同seed配对可比较episode outcome；episode bootstrap不能替代训练多seed。每次结果关联checkpoint SHA、runtime源归档、场景/reward协议并更新原实验记录。

| 核心主张（尚待验证） | 关键证据/控制 | 否证或收缩条件 |
|---|---|---|
| 事件语义与图结构更贴近通行决策 | M0/M1、M0+同事件特征、几何/覆盖和真实S/C/T | 收益仅来自扩输入，或事件正确但不改善任何决策区分 |
| 同路径学习进度优于固定CV | M1/M2、CV/CA、同预测器不同图、事实标签校准 | 预测不优于简单模型，或仅预测指标提升而闭环无收益 |
| 场景共享模式保留有用联合关系 | 同decoder K1/K2/K4、联合反例、联合评分、先平均/去跨区消融 | 只有参数规模收益、模式塌缩、联合信息无可观测价值 |
| 表示不依赖固定路口记忆 | held-out geometry/traffic、ID与坐标变换测试、跨场景闭环 | 只能在反复使用的固定模板上有效 |

各表的结果与效应值均为TBD；技术路线不预设目标成功率或显著性。

## 11. 随正常训练/评估采集，不依赖事后大规模补诊断

| 证据层 | 必备记录 | 正确统计单位 |
|---|---|---|
| 观测 | 输入/截断actor、短历史、valid mask、未观测入口、route候选覆盖 | 真实pre-action decision与actor观测；明确重复 |
| 几何图 | relation类型、zone区间/覆盖、out-of-set、建图耗时 | actor-route-zone关系；不当独立交通事件 |
| 预测 | entry/clear label及删失、anchor、route/progress概率、policy阶段 | 独立passage加重复预测anchor |
| 功能 | 同状态去时间/换mode关联/去跨区边后的动作与Q差 | shadow唯一状态；不是额外episode |
| 优化 | 分支激活、梯度与真实更新、预测/TD尺度、target同步 | update与模块版本 |
| 行为 | 进入/清空时刻、目标/实际速度车道、首次碰撞双方同tick状态、等待阶段 | completed episode及可核对事件 |

沿用既有shadow预算思想：训练整个phase最多20唯一state（按5000 raw门槛触发）；eval每episode最多4唯一state。正常probe不推进SUMO。关闭分支只证明功能依赖；若扰动是OOD，不把其动作变化当性能因果。

## 12. 建议工程组件与阶段归属

新包目录例如 `scene_event/`，此处仅规划，不创建实现文件：

| 组件 | 内容 | 阶段 |
|---|---|---|
| `schema.py`, `collector.py` | 观测合同、history、frame/label join | 1 |
| `map_cache.py`, `geometry.py` | 公共拓扑、corridors、footprint/zone | 2 |
| `temporal.py`, `polyline.py`, `dual_graph.py` | M0基础表征 | 3 |
| `event_projection.py`, `event_graph.py` | CV/CA事件、actor-zone/path消息 | 4 |
| `progress.py`, `event_targets.py`, `aux_losses.py` | 进度过程、删失label与监督 | 5 |
| `shared_modes.py`, `readout.py` | 场景共享混合、确定性读出 | 6 |
| `policy.py`, `diagnostics.py`, `configs/` | SAC接口、版本配置、正常采集 | 从3持续 |

优先使用小型PyTorch模型；SUMO已提供地图/状态，不引入图像BEV感知、LLM、扩散世界模型或大型预训练模型作为默认依赖。创新证据应来自事件关系表示，而不是规模堆叠。

## 13. 文献与技术定位

- GraphAD提供双图和mode级未来交互的强近邻，M0只借鉴一般结构：https://www.ijcai.org/proceedings/2025/0270.pdf
- QCNeXt支持query-centric/集合等变的联合预测思路，是2023技术报告/挑战赛工作而非近期主会新论文：https://arxiv.org/abs/2306.10508
- BeTop是未来行为拓扑监督与交互规划的关键对照：https://proceedings.neurips.cc/paper_files/paper/2024/file/a862f5788fd09bb6843c694d8120d50c-Paper-Conference.pdf
- CfDCA说明资源区与clearance先例及其安全假设：https://transportlab.sydney.edu.au/wp-content/uploads/2025/09/AS-DL-MR-2025.pdf
- RAP支持不混淆ego目标与背景预测目标的限制：https://vipl-epp.github.io/pdf/2025RAL-RAP.pdf
- PyTorch attention实现合同需结合本地版本核验：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

这些来源支持建模依据，不证明本方案已经创新或一定提升SAC。阶段1–3是工程与公平对照；阶段4–6才逐步检验事件表征机制；阶段7决定其是否形成可投稿的证据。
