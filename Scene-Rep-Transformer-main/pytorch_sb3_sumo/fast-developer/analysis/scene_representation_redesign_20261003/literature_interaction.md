# 车辆交互结构与联合未来表示：近期文献核验

检索日：2026-10-03。时间窗：2024-10-03 至 2026-10-03，正式发表时间与预印本时间分别记录。范围聚焦于车辆交互图、群组关系、多模态/联合未来以及能否服务闭环决策。它与同目录的 `literature_topology.md`、`root_source_notes.md` 配套；后两者已覆盖 lane/map graph、GraphAD、冲突区资源图 CfDCA 与 HGRL 等。本文件只补充车辆交互与联合未来部分，不重复深评地图感知和冲突区控制论文。

公开检索只用了通用关键词，没有向搜索服务发送项目路径、未发表方法名、实验数字或代码。查询包括：

- `autonomous driving vehicle interaction graph joint prediction planning NeurIPS 2024 2025`
- `vehicle interaction hypergraph trajectory prediction AAAI 2025`
- `future-aware interaction network motion forecasting ICCV 2025`
- `small-world interaction flow-aware trajectory prediction TPAMI 2026`
- `multi-agent behavior topology braid interactive autonomous driving NeurIPS 2024`
- `causal interaction representations sim-to-real motion forecasting CVPR 2025`
- `joint prediction planning role-aware autonomous driving RAL 2026`
- `sequential mode modeling sparse multimodal motion prediction CVPR 2025`
- `causal logic trajectory prediction autonomous driving IJCAI 2025`
- `joint multi-agent trajectory prediction QCNeXt`

优先核查了 NeurIPS、CVPR、ICCV、AAAI、IJCAI 官方论文页/PDF、TPAMI bibliographic record，以及作者代码库。此处不猜测任何期刊 CAS 分区，也不把检索排序或预印本日期当作会议发表时间。评分是检索优先级，不代表论文质量排名：Insight（研究机制与本题相关度）、Completeness（原文、代码和设置核验程度）、Numeric evidence（可直接追溯的量化结果），每项 0–5。

## 六篇核心近邻

| 论文、作者与正式出处 | 结构机制与监督 | 可核对的量化证据 | 对本题的迁移价值和边界 |
|---|---|---|---|
| **Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving (BeTop)** — Haochen Liu, Li Chen, Yu Qiao, Chen Lv, Hongyang Li；NeurIPS 2024 Main Track。 [NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/a862f5788fd09bb6843c694d8120d50c-Paper-Conference.pdf) · [官方代码](https://github.com/OpenDriveLab/BeTop) | 从多个主体的未来轨迹构造 braid/intertwine 拓扑：节点是主体未来轨迹，边表征轨迹间的编织关系，并区分表示 assertive/诱发让行与 passive/自身让行。BeTopNet 联合做拓扑推理、未来预测与规划，使用拓扑引导局部 attention 和 contingency branches。关系标签来自未来联合轨迹；推理时应由网络预测拓扑。 | 作者在 nuPlan Test14-Inter（筛出 1,340 个交互场景，每个场景 15s）做 reactive closed-loop 仿真：PDMScore 0.894，PlanTF 0.871，PDM-Closed 0.833；Test14-Random 的闭环平均分 0.878，PlanTF 0.847。消融显示不使用 local attention 后 Test14 的闭环分数下降（论文 Table 8）。训练使用 4 张 A100；预测和规划模型分别 30/25 epochs。 | 这是“多主体未来关系 + attention + 闭环规划”的强先例，不能声称首次建模联合未来行为拓扑、让行关系或拓扑引导注意力。代码明确完整公开 WOMD prediction pipeline，而 nuPlan planning pipeline 在 README 中仍列为 TODO；论文结果与开源规划复现能力要区分。PDMScore/规划网络也不是 SAC critic 的证据。 |
| **SWIFT: A Small-World Interaction Framework for Flow-Aware Trajectory Prediction in Autonomous Driving** — Chengyue Wang, Bin Rao, Haicheng Liao, Bonan Wang, Chengzhong Xu, Zhenning Li；arXiv v1 2026-07-03；TPAMI 48(10), 2026 Oct, pp.13389–13405，DOI 10.1109/TPAMI.2026.3709672。 [arXiv 全文](https://arxiv.org/html/2607.09741) · [PubMed 书目记录](https://pubmed.ncbi.nlm.nih.gov/42391083/) | 有向交互图中节点是 agent，边 `e_ij` 表示 i 受 j 影响。小世界先验以结构优化构造边集，再由可学习图学习模块细化；flow regime encoder 根据密度、平均速度、间距等场景统计调节本地/全局权重和交互阈值。关系推理明确区分 direct、co-influenced、co-influencing 三种邻接通道，用图卷积分路传播。后续再与地图静态编码、动态交互编码融合以生成每个目标主体的多模态轨迹。 | nuScenes：minADE5 1.15 对 WAKE 1.24；minADE1 2.89 对 3.03；minFDE1 6.73 对 7.02。MoCAD 5s RMSE 2.36 对 BAT 2.88（18.1%）；NGSIM 3s RMSE 1.23 对 GaVa 1.52。去除高阶交互推理模块后 nuScenes minFDE1 从 6.73 增至 7.37（+9.5%）。四张 A40 训练。 | 这是最直接的“结构化车辆交互图 + 动态场景状态 + 多关系聚合”近邻；通用小世界图、动态邻接与高阶关系都不能单独支撑新颖性。论文只做轨迹预测，无 SAC/闭环决策。作者论文/公开 arXiv 中未找到明确代码发布链接。书目数据库把正式期刊日期记作 2026 年 10 月，但精确到日的正式上线日期未核实；arXiv 预印本日期明确在时间窗内，TPAMI 期刊归属可确认，是否早于 10-03 的边界日期保持未决。 |
| **Future-Aware Interaction Network for Motion Forecasting (FINet)** — Shijie Li, Chunyu Liu, Xun Xu, Si Yong Yeo, Xulei Yang；ICCV 2025，pp.7505–7515。 [CVF 论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Future-Aware_Interaction_Network_For_Motion_Forecasting_ICCV_2025_paper.html) · [论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Future-Aware_Interaction_Network_For_Motion_Forecasting_ICCV_2025_paper.pdf) · [作者代码](https://github.com/sj-li/FINet) | 场景编码包含历史 agent token 和 lane token，并生成潜在 future trajectory tokens，让这些候选未来参与场景空间交互，然后用 Mamba 做空间/时间处理。Adaptive Reorder Strategy 预测可学习参考点，再按场景元素至参考点距离排序，以便对无序 scene token 使用序列模型；Temporal Enhanced Decoder 对未来轨迹 token 做时间细化。训练中使用 GT 未来终点对齐损失；未来真值是训练标签，不能作为在线 RL 状态输入。 | Argoverse 2 test：minADE6 0.66、minFDE6 1.27、MR6 0.15；QCNet 分别 0.65、1.29、0.16。Argoverse 1 val：minADE6 0.59 对次优 0.66。单 RTX A5000、batch=1 的效率表：1.47G FLOPs、17.72ms、3.7M params、0.55G GPU memory；QCNet 对应 28.0G、54.55ms、7.7M、2.92G。 | “潜在未来轨迹参与当前场景表示”和未来候选之间交互已经是明确先例。它仍是有监督 forecasting，不提供 policy-conditioned action value 或闭环 SAC 结果。相对 QCNet 的速度与精度来自整套 Mamba/FIM/decoder，不应归因到单个交互图机制。 |
| **NEST: A Neuromodulated Small-world Hypergraph Trajectory Prediction Model for Autonomous Driving** — Chengyue Wang, Haicheng Liao, Bonan Wang, Yanchen Guan, Bin Rao, Ziyuan Pu, Zhiyong Cui, Cheng-Zhong Xu, Zhenning Li；AAAI 2025，39(1):808–816。 [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/32064) · [官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32064/34219) | agent 为顶点，hyperedge 为一组参与共同交互的主体；Newman–Watts 小世界结构扩展局部交互，Neuromodulator 从 agent 特征/密度产生连接阈值 α 与附加连接概率 β，并以 vertex→hyperedge→vertex pooling 聚合群组信息。每个 target agent 有 K 个意图模式和相应轨迹不确定性参数。 | nuScenes：minADE5 1.18（论文称相对最优旧法约 14.5%）；MoCAD 5s RMSE 2.42 对 BAT 2.88（约 16%）；HighD 4s/5s 相对 BAT 改善 27.3%/22.6%。论文报告 NEST 对 12 agents 平均 11.6ms，测试硬件 RTX 3090；对照时间来自 RTX 3090 Ti 的先前工作，不是同机控制比较。 | 多车 hyperedge、密度驱动的动态图和不确定边连接已有先例。它未报告闭环或 RL 指标；多模态是 target-centered 预测，不能等同于同一个场景级 joint mode。官方论文未找到作者代码链接，本轮未能确认代码是否公开。 |
| **ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling** — Zikang Zhou, Hengjian Zhou, Haibo Hu, Zihao Wen, Jianping Wang, Yung-Hui Li, Yu-Kai Huang；CVPR 2025，pp.1612–1621。 [CVF 论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html) · [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.pdf) · [arXiv 正文](https://arxiv.org/html/2411.11911) | 把 K 个轨迹模式按序列因子化，后续 mode query 条件于已解码的前序 mode；Memory Transformer 与 context transformer 迭代细化预测。Early-Match-Take-All (EMTA) 让最早匹配 GT 的轨迹承担正样本，其他重叠 mode 退为负例，缓解重复模式与分数塌缩。场景编码可以沿用 QCNet，含 agent-agent、agent-map attention；主要贡献是模式之间的关系，不是 scene-level agent joint mode。 | WOMD val：ModeSeq 的 Soft mAP6/mAP6/MR6/minADE6/minFDE6 为 0.4562/0.4507/0.1206/0.5237/1.0681，QCNet 为 0.4508/0.4452/0.1254/0.5122/1.0225。AV2 单体 forecasting：ModeSeq minADE6 0.63、minFDE6 1.26、MR6 0.14；QCNet 0.65、1.29、0.16。WOMD validation 的模式数 3 时，其 Soft mAP6 0.4509，QCNet 0.4214；延迟 86±9ms 对 63±11ms。 | 提醒“多个 mode”必须有覆盖、置信度和联合一致性的定义；mode 排序/训练会改变统计目标。此文按主体预测多模式，不证明各主体模式索引组成一个具有校准概率的场景级 joint distribution；没有闭环决策结果。公开论文页未找到官方实现链接。 |
| **RAP: Role-Aware Joint Prediction and Planning in Autonomous Driving** — Xiaolong Tang, Meina Kan, Shiguang Shan, Xilin Chen；IEEE RAL 11(2), 2026-02，pp.1786–1793。 [作者团队论文 PDF](https://vipl-epp.github.io/pdf/2025RAL-RAP.pdf) | 基于 GNN 的 joint prediction 结构：AV、周车、车道元素为节点，边携带相对时空位置。加入 route–identity token pairing：导航 route 只对 ego AV 起作用，防止周车错误跟随 ego route；on-road/obstacle collision/agent collision 等向量辅助损失只作用于 ego；对 ego 历史和运动状态 dropout/perturbation 缓解 closed-loop feedback shortcut。 | nuPlan open-loop 和 closed-loop 均评估；论文消融中移除 ego history/kinematic dropout 与 perturbation 后 open-loop OS +1.45，但 closed-loop NRS −15.35；三项组件组合 FS=79.75。该结果说明训练目标和 ego/其他主体分工会改变闭环效果。 | 这是 prediction/planning 角色不对称与闭环反馈的近邻。它没有 conflict-zone occupancy event graph，也不是 SAC critic 的表征对照；不要把“共享预测 + ego 规划损失”写成新方法。文件名含 2025，但正式卷期为 RAL 11(2), Feb 2026。 |

## 其他筛查项

以下共计 14 篇候选（含上表六篇）。其余八篇只用来标明边界或交叉学科，不把摘要级证据冒充完整论文审查。

| 候选 | 正式出处/时间 | 判断与处置 | 一手来源 |
|---|---|---|---|
| GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving | IJCAI 2025 Main | 与两图组合、future-mode 节点、同步未来几何邻接、actor-to-map 关系直接重叠；由同目录 `literature_topology.md` 深评，不能重复包装为新文献缺口。 | [IJCAI PDF](https://www.ijcai.org/proceedings/2025/0270.pdf) · [官方代码](https://github.com/zhangyp15/GraphAD) |
| Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations | CVPR 2025，pp.17271–17281 | 以模拟 causal annotation 做 contrastive/ranking regularization，偏行人/群体轨迹；对“真正 causal interaction representation”有意义，但不是路口车辆闭环或 scene-level joint mode。25% ETH-UCY 实数据时 ranking transfer 平均 ADE/FDE 0.516/1.063，AutoBots baseline 0.539/1.113。作为辅助监督先例，放在补充层。 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Rahimi_Sim-to-Real_Causal_Transfer_A_Metric_Learning_Approach_to_Causally-Aware_Interaction_CVPR_2025_paper.pdf) · [作者代码](https://github.com/vita-epfl/CausalSim2Real) |
| Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction | IJCAI 2025，pp.9918–9926 | 把空间 map 与时间 agent factors 区分，backdoor adjustment + counterfactual decoder；是 causal/spatial-temporal 预测邻近工作但无闭环 RL。ApolloScape WSADE 1.0681，nuScenes minADE10/minADE5/FDE 0.93/1.17/6.71。非主要交互图工作。 | [IJCAI 页面](https://www.ijcai.org/proceedings/2025/1102) · [PDF](https://www.ijcai.org/proceedings/2025/1102.pdf) |
| Curb Your Attention: Causal Attention Gating for Robust Trajectory Prediction (CRiTIC) | ICRA 2025；预印本 2024-09-23，正式会议时间在窗口内 | 以因果发现估计 agent 间关系并门控 transformer 注意力，偏预测鲁棒性；GitHub 链接只是学术项目页模板仓库，未确认算法训练代码公开，且无闭环决策证据；仅补充候选。 | [arXiv](https://arxiv.org/abs/2410.07191) · [作者项目页](https://ehsan-ami.github.io/critic/) |
| CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-Scale Reinforcement Learning in Autonomous Driving | CVPR 2025，pp.17239–17248 | 有强化学习和 reactive/non-reactive closed-loop planning，但核心是 trajectory token autoregressive generation/selection 与 PPO，不是交互图 critic；训练数据与环境规模差异大。论文 γ=0.1，不能直接当作 SAC 参数依据。 | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.html) · [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.pdf) |
| T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving | CVPR 2025，pp.17197–17206 | lane/lane、lane/signal 图与 counterfactual lane topology；只覆盖静态交通地图拓扑，没有动态主体交互。OpenLane-V2 OLS 46.3 不能证明已知地图上的 SAC 表征收益。 | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2025/html/Lv_T2SG_Traffic_Topology_Scene_Graph_for_Topology_Reasoning_in_Autonomous_CVPR_2025_paper.html) · [官方代码](https://github.com/MICLAB-BUPT/T2SG) |
| Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling (Flow Planner) | NeurIPS 2025 Main | 互动行为引导的 flow-matching planner；闭环系统级近邻但不是交互事件图/SAC latent 证据。 | [NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36d1e8aa9ceec3b781682bf5e63c31bf-Abstract-Conference.html) |
| QCNeXt: A Next-Generation Framework for Joint Multi-Agent Trajectory Prediction | arXiv 技术报告 2023；CVPR 2023 Workshop challenge | 超出近两年窗，但必须作为历史 novelty anchor：query-centric 坐标系、置换等变与空间 roto-translation/time-translation invariance；DETR-like decoder 在未来时刻 joint model agent 互动；AV2 Multi-Agent Challenge 第一。不得把相对坐标不变、未来联合预测或 agent 交互解码称作新颖点。 | [arXiv](https://arxiv.org/abs/2306.10508) · [CVPR 2023 Workshop 记录](https://openaccess.thecvf.com/content/CVPR2023W/WAD/html/Zhou_QCNeXt_A_Next-Generation_Framework_For_Joint_Multi-Agent_Trajectory_Prediction_CVPRW_2023_paper.html) |

## 给候选表示路线的审查意见

候选问题被理解为：共享场景 mode 下，每个 actor 有路径与路径进度；同一 mode 里的多个 actor–conflict-zone 关系表达 entry/clearance 时间区间及顺序；保留这些 correlated events 直到 SAC critic 读取 `z_t,a_t`。这比“增加交互图再做 attention”具体，但仍不能据此宣称 novelty。

当前一手证据已覆盖其多数宽泛组成：GraphAD 用每 actor 的不同未来 mode 建 dynamic node，以同步未来几何距离连 agent–agent 和 future agent–map；BeTop 以 braid/intertwine 从 joint future 监督主体间拓扑，并用拓扑引导 attention；QCNeXt 已做 joint future prediction 与未来 agent self-consistency（虽是 2023）；SWIFT 把 direct/co-influenced/co-influencing 高阶关系纳入车辆图；NEST 用动态小世界 hyperedge 表达群组交互；ModeSeq 已专门建模不同未来 mode 的序列依赖。冲突区作为资源、路径上的 zone sequence、entry/clearance 也已有 CfDCA 等工作，见配套 `root_source_notes.md`。因此“显式 zone event”本身或“joint mode + graph”本身都可能是已有结构的重命名。

尚可检验、但未证明的窄差异是：**场景级**共享 mode 是否给出单一联合概率对象；这个 mode 内 actor 路径在多个**具名资源区**上的有序占用区间如何联合建模；该联合结构作为在线可用状态/动作条件的 critic 输入，是否改变 SAC 的 action ranking 与闭环行为。它必须明确区分 BeTop 的轨迹 braid 关系、GraphAD 的同步最小距离邻接、ModeSeq 的单体输出模式序列，以及 CfDCA 的固定路径/协议控制资源占用。不能把后者说成因果图，也不能仅在 edge 上加一个 conflict flag 就暗示全新理论。

### 实际后续占用的辅助监督

只用真实已观测的 future occupancy 做辅助目标，原则上可以训练状态表示；例如从 replay 的 `(z_t, a_t)` 预测下一步每个冲突资源是否被占用，并用 Bernoulli log loss/BCE 监督边际概率。标签要从实际 replay 后继观测生成，不能把它喂给当前 decision 的 actor/critic 输入；不要把未执行的反事实动作结果伪装成 observed occupancy。

但边际 occupancy BCE 只对每个 zone 的 Bernoulli 边际是 proper scoring rule，不能保证多个 zone 的联合依赖、共享 mode 一致性、先后次序或 clearance interval 的联合概率校准。要声称模式/事件“联合一致”，需要明确 scene-level joint event distribution，并用联合 proper score 或经过校准的结构化 likelihood 检验。较长 horizon 的 replay label 还受到未来策略轨迹和后续行为 action 混合影响；仅以当前 `a_t` 为条件不能把这些 outcome 当成当前 action 的反事实后果。辅助损失还可能与 SAC value gradient 冲突，建议保留只预测真实 transition 的最小参照与 loss ablation。

### 100k 训练预算的判断

按小场景、有限主体与少量 conflict resource 设计时，生成 event slots、做 actor–zone 消息传递和加一步 occupancy head 的计算成本可能可以控制；这是架构成本推断，不是实验事实。100k 环境步数不自动等于 100k 独立交互样本：episode 中相邻状态高度相关，稀有碰撞/抢行/多车同步占区的有效标签可能很少。不能从总 raw steps 推断 mode 或冲突事件被充分覆盖，应记录训练集里各类 event counts、同时区间占用、类别不平衡与按 seed 分布；报告不能只看 ADE。

当前没有运行代码、训练、仿真、离线项目数据分析或新增评估。结构建议是文献推导，不是已验证结果。要区分信息量与因果收益，最小的后续方法比较应控制同一 SAC、训练预算、seed、reward、动作接口与模型容量，比较普通交互图、mode-averaged feature、actor–zone marginal heads 与 mode-preserving joint event representation；除 task return 外审查 action-ranking consistency、占用概率/联合校准、计算量和稀有事件覆盖。此次没有授权或启动这些实验。

## 关键来源

- [BeTop, NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/a862f5788fd09bb6843c694d8120d50c-Paper-Conference.pdf)；[官方 WOMD 代码及 planning TODO](https://github.com/OpenDriveLab/BeTop)
- [SWIFT arXiv full text (v1 2026-07-03)](https://arxiv.org/html/2607.09741)；[TPAMI bibliographic record](https://pubmed.ncbi.nlm.nih.gov/42391083/)
- [FINet, ICCV 2025 official page and PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Future-Aware_Interaction_Network_For_Motion_Forecasting_ICCV_2025_paper.html)；[official author repo](https://github.com/sj-li/FINet)
- [NEST, AAAI 2025 page and PDF](https://ojs.aaai.org/index.php/AAAI/article/view/32064)
- [ModeSeq, CVPR 2025 official page and PDF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html)
- [RAP author PDF](https://vipl-epp.github.io/pdf/2025RAL-RAP.pdf), issue/volume metadata checked against the paper title page: RAL 11(2), Feb 2026.
- Companion road-topology and conflict-resource review: [literature_topology.md](literature_topology.md) and [root_source_notes.md](root_source_notes.md).
