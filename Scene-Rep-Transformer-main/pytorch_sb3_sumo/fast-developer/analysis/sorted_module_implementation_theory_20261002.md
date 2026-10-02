# Sorted/depart4 模块实现与理论适用边界审计

审计日期：2026-10-02。范围是旧场景 intersection_sorted / depart_scale=4.0 的已归档实现，覆盖 SAC+MLP→ST→ST-RT→Topo→routeaware、ST-RT→3slot、Topo→Topo+3slot。这里审代码与网络定义，不重算本轮最终结果，也不启动训练、评估或仿真。

## 结论先行

方法名对应的是不同层次的前向变化。ST 将逐帧车辆交互图与时间注意力作为表征主干；RT 将逐actor的 map 路线从 MLP fallback 改为 cross-attention，并将路线意图注入所有车辆历史 token，因而它不只是“多一个 route slot”。Topo 一次打开了静态车道图编码与 vehicle→lane soft query、lane关系诱导的车辆边特征、以及 route-goal 的 topology residual，不能把 ST-RT→Topo 解释为单一lane-token消融。routeaware 仅在ego的goal-topology pooling中筛选路线可接续车道，它不约束连续动作。3slot则保留现有ego/social/route上下文，把联合非线性读出替换为三段线性投影；它缩窄了encoder末端的可学习读出，但SAC actor/critic后面仍有跨三段混合的ReLU MLP。

同一共享encoder由critic TD loss更新；actor拿共享特征时detach，actor optimizer也不持有encoder。本轮这些新增D1分支把Graph-SLT、SBS和表示辅助损失关掉，所以模块效应应按“critic学习到的表征路径变化”理解，不能归因给SLT监督。静态网图还确认，route-aware mask不是车道动作shield：对于ego路线 -E1→-E0，-E1_0 没有通往 -E0 的连接，而起始配置指定lane2；最终停在lane0时路由已无出口，但代码不能说明策略为何选到/留在lane0，也不能证明Topo attention造成了该选择。

## 运行源码与归档一致性

正式旧场景运行根是 D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sort2_1001；源快照为该目录下 source_archive\manifest.json（796个源文件，2026-10-01 22:27创建）。我对照manifest逐文件核对了下表，列出的当前文件SHA-256与运行归档完全相同，因此下文引用的代码可用于解释该run。此核查不代表旧的、更早sorted run也拥有同样归档。

| 当前文件（相对PR根） | SHA-256，与归档相同 |
|---|---|
| fast-developer/train_intersection_yield_v2_d1.py | 1650f0c127a0f8c6980504ce3489138c65e8700dd6836b3d4f286fbeede7e69b |
| algos/sb3_torch/incremental_topo_encoder.py | 080b8bdac0cdc755747e9cac49d230a5a929eb0545368a2ace5b360a77167c7c |
| algos/sb3_torch/topo_temporal_features_v2.py | 94109debb0579aa402669962dafec53814e3d4c1e0dfa0a0f4a287bf66f9b06f |
| algos/sb3_torch/topo_temporal_features.py | efd29674777a5ab273539ae9cf82bc9f1d22e82e7094fd2e4c46e3d5f7e165e1 |
| algos/sb3_torch/policies.py | d7f44e647c14cdd8d84effd7269e826679828610ed01ac434c5b5cdda3ec5d0a |
| algos/sb3_torch/sac.py | 06c33befdd86e62f78366d76f541483880835920ba7e716c5457c99ed13c180d |
| envs/sumo/route_reachability_v1.py | 45f2f18f150cd8bf4c8a7b86817f9fbafbb101734536b2cfb042998d37bdec93 |
| envs/sumo/topology_graph_v2.py | 9ffba5e3d78d6c8e10ee602c69eab1e2f74f1d58f4ada1b91fb5d0679d3170fd |

路径记号：PR = D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo；FD = PR\fast-developer。文件路径以下均相对PR根给出。

## D1配置的真实模块差异

配置来自 fast-developer/train_intersection_yield_v2_d1.py:224-282。D1各项都映射到父方法 sac_mlp，连续SAC策略；本次分支不是不同动作头或不同RL算法。

| D1臂 | use_route | use_topology | use_slots | reachability | 前向实际变化 |
|---|---:|---:|---:|---:|---|
| ST | false | false | false | false | 车辆图+时间交互；地图仍经MLP route fallback生成route component |
| ST-RT | true | false | false | false | MapPolylineEncoder与route cross-attention替代route fallback；route context注入各actor token并进入ego route readout |
| ST-RT-Topo | true | true | false | false | 增加车道图编码/查询、Topo派生车辆关系边特征、route-goal的Topo residual |
| ST-RT-Topo-routeaware-v1 | true | true | false | true | 在Topo ego-goal pooling使用路网连接及ego当前route/index产生的三态车道mask |
| ST-RT-3slot | true | false | true | false | ST-RT特征不变，末端联合MLP改成32/64/32三个线性读出 |
| ST-RT-Topo-3slot | true | true | true | false | 与Topo使用同一增量前向，末端改成三槽读出；incremental_slots=true阻止切换到旧Full父类路径 |

SAC+MLP基线通过D1的父方法提供policy、critic、优化器和环境契约；D1-ST的差异是特征提取器。use_route=false不表示观测里没有地图：两条路都读相同trajectory/map输入，差别在Map的编码、cross-attention与注入方式。ST默认地图route component来自mlp_map_route；RT启用时路由token先和当前车辆状态做masked cross-attention，然后用于历史车辆token注入与ego route readout。见 incremental_topo_encoder.py:304-367、topo_temporal_features.py:476-497。

另须与旧Full配置区分：D1的旧sac_mlp_d1_full在默认补齐后会同时开use_graph_slt、use_sbs、representation_coef=1.0、slot_balance_coef=0.01；本表新增的routeaware/3slot臂显式关闭这些开关并用普通SceneRepresentationSAC。方法名里有slot不能推出其正在训练SLT或SBS。证据在 train_intersection_yield_v2_d1.py:243-282、426-477。

## 实际前向与维度

对 intersection_sorted，SumoSceneEnv的默认观测为trajectory=(6,10,5)：ego加5个最近actor、10帧历史、每帧5维；map=(12,10,5)：6 actor各2条路径、每条10个地图点、每点5维。相关默认值位于 envs/sumo/sumo_env.py:240-329；PaperScenarioSpec默认map_paths_per_actor=2/map_feature_dim=5，sorted场景ego出发时刻50秒，见 envs/sumo/paper_scenario_registry.py:22-40、100-103。特征extractor将state hidden和最终features固定为128维。

1. State与ST。每帧每actor 5→128→128 ReLU state encoder后，vehicle graph按同一历史帧做两层消息传递；每个edge的消息由source特征与8维edge向量经过Linear→ReLU→Linear得到，再按高斯距离权重对邻居归一化聚合，并用residual LayerNorm。之后TemporalInteractionEncoder给每个actor的历史加lag embedding，以self-attention、residual/LayerNorm/FFN编码时间，再用当前帧query对历史做pool attention。最后ego向social attention提供query，读取ego及邻车actor特征。实现位于 topo_temporal_features.py:216-305；增量路径调用在 incremental_topo_encoder.py:436-457。Social上下文已经由ego查询，因此ego/social不是隔离的原始因素。
2. RT。route-attention输出route_context。普通RT分支把它广播进所有actor的history token并做LayerNorm(state+route)，然后才做Topo查询/车辆图/temporal/social；故周车表征也看到各自route context。最终route component也由ego与路径token注意力读取。若route关闭，MLP map route产生route component但不会将route context注入交互token。代码 incremental_topo_encoder.py:304-367、459-523。
3. Topo。每lane节点用点坐标/静态属性编码，经两层六关系车道图网络。之后每个actor、每个有效history query以距离、地图route几何和车辆/车道朝向构造lane attention；Topo context以trainable scalar残差加到intent token后LayerNorm，进入车辆图。最后Topology attention另被压成车辆pair间same-lane和conflict-or-merge边特征，并与位置、速度、heading、proxy TTC拼成8维edge输入。route component再用route_base query一份topology goal attention，并以delta-LN残差融合。实现分别见 topo_temporal_features_v2.py:42-45、67-160、182-318、362-535、537-662；增量主干见 incremental_topo_encoder.py:368-429、436-456、459-518。
4. Topo query的边界。query bias由相对车道点距离、route-polyline距离/方向、vehicle-lane方向组成；mask是有效lane节点和heading_cosine≥reverse_heading_cosine_min，之后取bias top-k（当前默认8）。没有兼容节点时改选最近有效lane。这里“兼容”不是network中lane-to-next-edge的可达性判定，也不是碰撞安全约束。具体式子可读为 score(q,l)=q·k_l/√d + b_distance + softplus(w_route)b_route + softplus(w_heading)cos_heading；selected=topk(score within direction mask)，empty时nearest-valid fallback。代码 topo_temporal_features_v2.py:427-535。
5. routeaware。D1只给routeaware臂套RouteReachabilityObservationWrapper，按静态SUMO lane connections（含via internal corridor）和当前ego route/routeIndex构造lane节点三态labels：eligible=1、ineligible=0、unknown=-1。encoder只对最后ego goal-topology query做mask：原几何query与legal相交；交集空但有legal时扩为全legal；无known legal时以base mask保证有限attention，并将topology-goal delta置零。周车Topo attention、social、actor输入和动作空间不变。见 train_intersection_yield_v2_d1.py:298-355、route_reachability_v1.py:27-133、141-212、incremental_topo_encoder.py:473-518、977-1035。

## Topo和route-aware不是控制约束

静态场景证据：envs/sumo/original_scenarios_v1/intersection_sorted/ego.rou.xml:4-5指定route=-E1 -E0及departLane=2；map.net.xml:99-102给出三个70m的-E1_0/1/2 lane，而第169行唯一的 -E1→-E0连接为fromLane=2→toLane=1（via :J1_14_0），internal corridor继续到-E0_1见201-202。因此，停在-E1_0且route index仍为0时，lane0本身没有可接续该计划route的连接；车需要在到junction前进入lane2才可走该route。起始XML则指定了有该左转连接的lane2。

这可以解释“已经落到lane0末端后的路线阻塞”，不能解释策略为什么向低索引换道/为什么未保持lane2。Topo普通query使用方向兼容和route geometry bias，而非next-edge可达mask。routeaware的mask把合法lane信息用到了ego-goal表示，但没有lane_action_mask、动作投影、控制屏蔽或对actor输出作修正；其输出仍是普通连续policy特征。训练runner在factory后加observation wrapper，见 train_intersection_yield_v2_d1.py:620-625、870-879；策略是continuous SceneRepSACPolicy，见 algos/sb3_torch/policies.py:81-128。运行时terminal lane不能反推Topological attention是根因，也不能由一次lane mask证明其提高了控制安全。

现有正常流程遥测已经可以按raw tick定位“首次进入route不可达lane”：`sumo_env.py:536-554`将本次action、请求目标lane、静态route continuation标签和本地request status/reason记入decision control；`565-614`每个simulation raw step都写行为快照；`1200-1245`在快照记录当前ego lane/route可达性与仍生效的lane_control；`806-826`另记相邻决策边界的实际lane迁移。因此不需重复增加first-illegal-lane采集。注意`request_sent`/`lane_change_applied`仅表明TraCI `changeLaneRelative`命令成功发出且未抛异常，不代表SUMO接受或物理换道已完成。若以后要区分请求为何未执行，可考虑在现有step后只读记录TraCI `getLaneChangeState`（当前运行版本需先做API可用性验证）；它给出lane-change model状态及合并TraCI请求后的状态，能反映阻塞bit，但不是未来成功保证。[SUMO TraCI vehicle API](https://sumo.dlr.de/daily/pydoc/traci/_vehicle.html)

本轮routeaware运行诊断补充了一个重要边界：报告给出的25个timeout终点均为 -E1_0，且route-reachability=false；goal合法attention mass=1、unknown/bypass/fallback=0，expand均值为0.764。它说明goal attention的合法节点筛选在这些记录中实际启用，仍未阻止车辆终止在错误接续lane。静态net证实lane0没有通往计划next edge -E0的connection，因此“终点车道不可按当前route继续”可证实；但仅凭这些聚合量不能分辨车辆是否被lane-change可行性/执行、队列或right-of-way阻挡，还是策略动作/时机导致，也不能把route-aware mask认作动作约束。逐步lane command、实际lane变化、速度与SUMO插入/碰撞事件仍是区分这些机制所需证据。

这与形式化shield的作用层不同：Shielding方法监控并纠正导致违反规范的agent action；本routeaware实现筛选的是内部attention的候选键，并不改动作集合。因此只可称路线可达性条件化表示，不能称安全shield或动作可达约束。[Safe Reinforcement Learning via Shielding, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11797)

## 3slot读出：参数、非线性和槽间融合

无slot增量头把三个128维上下文拼成384维，经Linear(384,128)→ReLU→Linear(128,128)→ReLU；头部活跃参数为65,792。slot头则各自对ego/social/route的128维上下文作Linear→32/64/32，无slot头路径此时不执行；projection合计16,512参数，少49,280、约少75%。来源 incremental_topo_encoder.py:201-215、525-536；父类投影定义 topo_temporal_features.py:430-434。

这不等价于“端到端完全无槽间交互”。前层vehicle graph让邻车互相传消息；social attention的query来自ego；route component也受ego query，并在Topo臂受topology goal读取影响。StructuredLatent随后按ego/social/route拼接成128维；SB3 actor和双critic继续用配置的[128,32] ReLU MLP；critic还会把128维scene feature与64维action embedding拼接。见 topo_temporal_features.py:38-56、incremental_topo_encoder.py:453-457、policies.py:18-30、44-70、D1 builder:408-417。因而可证实的是：槽投影阶段是分块线性降维，没有384→128头里那层联合非线性；其余上下文已有融合，后续策略/Q网络还能重新组合各槽。

参数数需区分active和registered。IncrementalTopoEncoder总是构建MLP fallback与三个slot projections；父Extractor也总是构建ego/social/route projection，随后D1开关只在forward里挑一条路径。对本表这些连续SAC臂，代码结构因此让注册参数名/形状顺序和总registered参数量相同，但未使用头/Topo分支在该方法上不会得到有效梯度；“总参数相同”不等于有效容量相同。拓扑关闭的ST-RT-3slot与Topo-3slot只有projection读出路径相同，Topo-3slot多出实际启用的Topo主干；两个分支的活跃网络也不能按head参数相等看待。

理论上，分块线性投影可以在进入SAC MLP前限制跨上下文的直接融合并丢失未保留的方向，但当前ego/social/route已不是互斥原始语义块，而SAC MLP仍可对压缩后的128维做非线性组合。现有前向结构支持“更窄的encoder读出/更晚的cross-component mixing”假设，不支持“槽完全独立”或“槽信息必然丢失”的结论。若要检验3slot本身，应加容量/非线性匹配头或保留joint-head残差作为控制；要检验Topo则须把lane-token residual、goal pooling和relation-edge特征分开，不应再把一次use_topology总开关视为单机制。

一个可检验、尚未实现的等参数非线性槽头是：ego、route各用Linear(128,132)→ReLU→Linear(132,32)→ReLU；social用Linear(128,120)→ReLU→Linear(120,64)→ReLU。三支含bias分别为21,284、23,224、21,284，总计65,792，与joint head相同；输入都是各自128维component，输出仍32/64/32且两个Linear后都ReLU。这控制active head参数量与激活深度，却仍改变encoder读出连接图：各支在concat之前不能跨component交互，故可检验“块结构/早期融合缺失”而非单独证明槽语义优劣；后续actor/critic仍跨槽混合。它是容量匹配对照，不应写作已知修复。

## 优化器、辅助loss和能排除的混杂

SceneRepSACPolicy创建一个online extractor并共享给actor和online critic；DetachedSceneActor在使用features前执行detach，且actor optimizer过滤encoder参数；critic optimizer包含critic参数及共享extractor。target critic使用独立extractor，从online critic拷贝后按tau做Polyak更新。SAC actor更新时暂时冻结critic参数，但保留Q对action的导数。源码见 policies.py:18-30、81-128及 sac.py:492-584。

本表新增D1臂的representation_coef=0、Graph-SLT/SBS=false，因此sac.py:_setup_model不创建representation objective/optimizer；共享encoder仍由critic TD loss backward/optimizer更新，不会因coef=0而冻结。该规则也适用于槽projection：在线active slots由critic TD梯度学习，actor policy loss不直接更新encoder。sac.py:207-251、261-325、375-461、518-584；若辅助objective启用，梯度路径会多一次单独representation backward+optimizer，不适用于本表的这些新臂。

据此可排除或界定：

- 新routeaware/3slot方法不是因使用HSAC混合动作头或额外SLT/SBS监督而不同；D1父方法是sac_mlp，以上辅助开关在新臂为零。
- route-aware有效goal mask不是“actor被强制选合法lane”；它只改变representation readout。
- 3slot有效head宽度/参数量与MLP head不匹配，即使总registered参数一样，也不能把差异称为等容量结构控制。
- use_topology一个布尔开关同时变化lane编码/query/residual、Topo派生relation edge features、topology-goal pooling；若结果变好或变差，不能从本次开关比较指出其中哪个分支负责。
- critic-only encoder训练是所有这些SceneRepSAC D1臂共有的更新规则，不能只用“actor梯度为零”推断模块没有被训练；判断模块是否获得训练信号应看TD梯度与optimizer update。

## 理论联系与适用边界

关系消息传递的动机与Interaction Networks把实体与关系作为图上的显式计算对象相符：本实现让vehicle pair消息显式读取相对状态/软拓扑关系，再聚合到actor。但原论文研究的是可学习物理动力学任务，不证明当前路口任务的Topo关系特征会提升回报，也不证明attention relation score是因果解释。[Interaction Networks for Learning about Objects, Relations and Physics, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html)

时间self-attention与vehicle→lane/ego→route cross-attention的数学作用是学习对候选位置的加权读取；加lag embedding让模型可区分历史位置。Transformer与Graph Attention Networks提供这种内容依赖加权机制的基础，但不提供lane successor可达性、规划正确性或policy安全保证。本实现的静态route-aware三态mask才表达已知route-continuation，且只表达在goal attention上。[Attention Is All You Need, NeurIPS 2017](https://arxiv.org/abs/1706.03762)；[Graph Attention Networks, ICLR 2018](https://openreview.net/forum?id=rJXMpikCZ)

SAC的经典目标同时最大化期望回报与熵；但此处encoder的actor支路显式detach，并由critic TD loss训练，这是项目实现选择，不能把通用SAC理论默认的梯度路径套到此模型。[Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor, ICML 2018](https://proceedings.mlr.press/v80/haarnoja18b.html)

## 对根因判断的支持、限制与最小可证伪方向

| 代码/静态事实 | 可以支持的候选解释 | 不能单独证明 |
|---|---|---|
| -E1→-E0仅从-E1 lane2连到-E0 lane1；route-aware label按route/index和静态via链标legal | terminal -E1_0是计划route的不合法接续车道；若终点状态确认路网位置/route index一致，可作为错道停滞近因 | actor为何让车去lane0；是否Topo attention导致；车道变换请求是否被执行 |
| Topo residual和edge relation共享可训练scale；普通compat mask不查route successor | Topo启用多路相互依赖信号，低scale/低relation质量可形成“模块存在但影响弱”的待测假设 | 小scale意味着Topo必然没用；注意力质量相关指标本身是性能因果 |
| routeaware legal attention mass高但终点仍错道（由同轮诊断汇总提供） | 表示goal query可确实筛选合法lane，而行为瓶颈仍可能在特征到动作映射或执行/时机 | mask完全无效；或者需要用更强Topo注意力 |
| 三个projection头只对各自component线性压缩；下游policy/Q网络仍有ReLU融合 | 窄头/延后联合融合是可证伪的容量与交互假设 | 端到端没有槽间交互；性能差必然由压缩造成 |

最小后续若要区分这些假设：首先在现有episode/raw诊断中把route edge label、actor lane output、实际lane和lane-change执行按同一step对齐，以分开“路线mask编码”与“动作执行/状态约束”；其后把Topo拆成单独goal-query、relation-edge、lane-token residual臂，而不是改掉整个use_topology；最后用容量/激活匹配的三槽MLP与joint readout比较slot结构。任何route动作shield都会是独立的控制变量与安全协议，不能冒充本routeaware实现。旧场景结果为单训练seed，所有这些只能形成假设检验，不是当前性能结论。

## 关键源码定位索引

| 实现位置 | 主要证据 |
|---|---|
| fast-developer/train_intersection_yield_v2_d1.py:224-282, 358-477 | D1方法开关、父方法、encoder/policy与SAC/SACV2构建、loss系数 |
| algos/sb3_torch/incremental_topo_encoder.py:266-285, 286-368, 368-456, 459-536 | full-vs-incremental分支、RT注入、Topo主干、goal融合和两种输出头 |
| algos/sb3_torch/incremental_topo_encoder.py:977-1035 | route-aware mask及无合法候选安全回退 |
| algos/sb3_torch/topo_temporal_features.py:38-56, 126-171, 216-305, 408-434, 476-539 | structured latent、lane relation层、vehicle message passing、temporal attention、投影和route readout |
| algos/sb3_torch/topo_temporal_features_v2.py:42-45, 67-160, 182-318, 362-535, 537-662 | 六类lane relation、query attention/LayerScale、几何兼容和Topo关系诱导vehicle-edge特征 |
| algos/sb3_torch/policies.py:18-30, 33-79, 81-128 | actor detach、64维action branch、shared online extractor和target网络 |
| algos/sb3_torch/sac.py:207-251, 261-325, 375-461, 492-584 | representation optimizer创建条件、辅助目标、Bellman/TD梯度、actor/critic optimizer顺序和Polyak target |
| envs/sumo/route_reachability_v1.py:27-133, 141-212 | 从route当前/下一edge及静态lane via连接生成三态labels；wrapper仅增加观测信息 |
| envs/sumo/original_scenarios_v1/intersection_sorted/ego.rou.xml:4-5 | ego的 -E1 -E0 route和初始departLane=2 |
| envs/sumo/original_scenarios_v1/intersection_sorted/map.net.xml:99-102, 165-169, 201-202 | -E1 lane几何与route连接/内部via |
| runs/sort2_1001/source_archive/manifest.json | 本表审计所用的运行源码和traffic输入冻结清单 |

本报告没有声称重新读取所有raw rollout，也不判定本轮性能显著性；所有机制性结论限于上述D1方法配置和source-verified旧场景实现。
