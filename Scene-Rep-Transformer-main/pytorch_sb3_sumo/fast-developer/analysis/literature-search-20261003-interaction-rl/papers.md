# 近期交互决策与强化学习：14篇精选/近邻

日期：2026-10-03。时间窗：2024-10-03至2026-10-03；10篇正式发表（含2篇benchmark）与4篇明确标记的预印本分列。DSAC-T按2025期刊发表纳入，但初稿为2023。本文是筛选证据表，不是venue接受概率或完整复现。

主报告：[奖励判断与技术路线](../reward_literature_technical_route_20261003.md)。原始来源核对：[驾驶论文笔记](driving_sources.md)、[机制近邻笔记](mechanism_sources.md)。CSV为可继续维护的主表。

评分为根代理在本次阅读范围内的 I/C/N：机制洞见 / 论证与实验完整性 / 已核数值证据充分性，1–5；NA表示不适用或未读到足够量化证据，绝不是零效果。分数均为筛选判断而非客观论文质量。详细笔记原有fit/相似度等不同轴不与本表混算。Confidence还受只读摘要、PDF访问失败、预印本状态限制。Benchmark不评分方法增益。

| ID | 论文/来源 | 年份及状态 | 类型 | I/C/N | 证据置信 |
|---|---|---|---|---|---|
| P01 | [Safety Representations for Safer Policy Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html) | 2025 ICLR; published | method | 4/4/4 | high |
| P02 | [SVL: Goal-Conditioned Reinforcement Learning as Survival Learning](https://proceedings.mlr.press/v306/tiofack26a.html) | 2026 ICML; published | method | 5/4/4 | high |
| P03 | [TraCeS: Learning Per-Timestep Constraint-Violation Credit from Sparse Trajectory-Level Labels](https://proceedings.mlr.press/v306/low26a.html) | 2026 ICML; published | method | 4/3/NA | medium |
| P04 | [Distributional Soft Actor-Critic With Three Refinements](https://doi.org/10.1109/TPAMI.2025.3537087) | 2025 IEEE TPAMI 47(5):3935-3946; published; first arXiv 2023 | method | 4/4/3 | high |
| P05 | [CaRL: Learning Scalable Planning Policies with Simple Rewards](https://proceedings.mlr.press/v305/jaeger25a.html) | 2025 CoRL / PMLR 305:5301-5338; published | method | 5/5/5 | high |
| P06 | [Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)](https://papers.neurips.cc/paper_files/paper/2025/hash/c2915bc5961edb04e209a524ec167522-Abstract-Conference.html) | 2025 NeurIPS Main; published | method | 4/4/4 | high |
| P07 | [SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_SafeDrive_Fine-Grained_Safety_Reasoning_for_End-to-End_Driving_in_a_Sparse_CVPR_2026_paper.html) | 2026 CVPR pp.24854-24864; published | method | 4/4/5 | high |
| P08 | [AdaWM: Adaptive World Model based Planning for Autonomous Driving](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4c745dcbaf8ef0d7e145754e31b1516-Abstract-Conference.html) | 2025 ICLR; published | method | 4/4/4 | medium-high |
| P09 | [Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://proceedings.neurips.cc/paper_files/paper/2024/hash/017761f94a1cd66d01c041aff85492c4-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 NeurIPS Datasets and Benchmarks; published | benchmark | NA/5/NA | high |
| P10 | [NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf) | 2024 NeurIPS Datasets and Benchmarks; published | benchmark | NA/4/NA | high |
| P11 | [Learning Safe Autonomous Driving Policies Using Predictive Safety Representations](https://arxiv.org/abs/2512.17586) | 2025 arXiv 2512.17586; preprint; venue not verified | method extension | 3/3/3 | medium |
| P12 | [Survival Reinforcement Learning: Toward Scalable Self-Supervised RL](https://arxiv.org/abs/2605.31273) | 2026 arXiv 2605.31273; preprint v2 2026-09-18; first 2026-05-29 | method | 4/3/3 | medium |
| P13 | [ChronoSRL: Temporal Geometry for Self-Supervised Reinforcement Learning](https://arxiv.org/abs/2609.36238) | 2026 arXiv 2609.36238; preprint 2026-09-28 | method | 4/3/NA | low-medium |
| P14 | [Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability](https://arxiv.org/abs/2605.14246) | 2026 arXiv 2605.14246; preprint 2026-05-14; venue not verified | method | 4/3/NA | medium for mechanism; low for quantitative effects |

## P01 Safety Representations for Safer Policy Learning

作者：Kaustubh Mani et al.。

机制与关联：State-conditioned steps-to-cost distribution as safety representation。

评分证据：Table 1, SafeMetaDrive CRPO success 0.22±0.24 -> SR-CRPO 0.53±0.18; five training seeds。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Not action-conditioned target-policy event probabilities; early episode endings need censoring scrutiny。

主要来源：[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html)。代码核查状态：official author repository not verified in this search。

## P02 SVL: Goal-Conditioned Reinforcement Learning as Survival Learning

作者：Franki Nguimatsia Tiofack et al.。

机制与关联：First-hitting-time survival likelihood and goal value; offline hierarchical and flat actors。

评分证据：Table 2, four seeds, same DDPG+BC actor: HumanoidMaze-large CRL re-eval42±1% vs SVL66±2%; giant7±0% vs45±2%。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Explicitly anticipates competing failures and sparse/dense decomposition; those concepts alone are not new。

主要来源：[论文](https://proceedings.mlr.press/v306/tiofack26a.html)。代码核查状态：Software entry present on proceedings; repository contents not verified。

## P03 TraCeS: Learning Per-Timestep Constraint-Violation Credit from Sparse Trajectory-Level Labels

作者：Siow Meng Low; Ze Gong; Akshat Kumar。

机制与关联：Sequential violation credits factorize trajectory survival probability。

评分证据：Primary abstract and author code verified; PDF access failed, so no quantitative effect-size claim。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Unknown cost/threshold and coarse trajectory supervision differ from known simulator collision terminal。

主要来源：[论文](https://proceedings.mlr.press/v306/low26a.html)。代码核查状态：https://github.com/siowmeng/TraCeS。

## P04 Distributional Soft Actor-Critic With Three Refinements

作者：Jingliang Duan et al.。

机制与关联：Expected-value substitution, twin return distributions, variance-based gradient adjustment。

评分证据：Sections IV-A/B/C refinements; Section V-C/Figures4-5 five-run ablations and reward-scale tests; no exact improvement value quoted。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Continuous Gaussian return distribution does not expose competing terminal event types。

主要来源：[论文](https://doi.org/10.1109/TPAMI.2025.3537087)。代码核查状态：https://github.com/Jingliang-Duan/DSAC-v2。

## P05 CaRL: Learning Scalable Planning Policies with Simple Rewards

作者：Bernhard Jaeger et al.。

机制与关联：Simple route-completion reward and PPO minibatch scaling。

评分证据：CARLA 300M and nuPlan 500M samples; Longest6 v2 DS 64; Val14 nonreactive/reactive 91.3/90.6。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：8-GPU large-budget PPO privileged planning is not evidence that 100k SUMO SAC reward is the main problem。

主要来源：[论文](https://proceedings.mlr.press/v305/jaeger25a.html)。代码核查状态：https://github.com/autonomousvision/CaRL。

## P06 Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)

作者：Zhenjie Yang; Xiaosong Jia; Qifeng Li; Xue Yang; Maoqing Yao; Junchi Yan。

机制与关联：Privileged/raw-sensor world-model alignment。

评分证据：Bench2Drive raw-sensor RL DS 71.36 and SR 50.24%; Dev10 joint alignment ablation DS 0.0 -> 83.5。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Different information budgets; public repository described as inference code, not a confirmed full training release。

主要来源：[论文](https://papers.neurips.cc/paper_files/paper/2025/hash/c2915bc5961edb04e209a524ec167522-Abstract-Conference.html)。代码核查状态：https://github.com/Thinklab-SJTU/Raw2Drive。

## P07 SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World

作者：Kim; Oh; Yu; Shin; Kwak; Choi。

机制与关联：Trajectory-conditioned agent/time safety and drivable-area reasoning。

评分证据：NAVSIM PDMS/EPDMS 91.6/87.5; 61/12146 collisions; Bench2Drive DS 66.8。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Planning safety supervision, not online SAC; collision counts are paper-reported benchmark counts。

主要来源：[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_SafeDrive_Fine-Grained_Safety_Reasoning_for_End-to-End_Driving_in_a_Sparse_CVPR_2026_paper.html)。代码核查状态：https://github.com/SPA-junghokim/SafeDrive。

## P08 AdaWM: Adaptive World Model based Planning for Autonomous Driving

作者：Hang Wang et al.。

机制与关联：Identify policy/dynamics mismatch and selectively adapt world model or policy。

评分证据：CARLA ROM03 Table 2 SR .82 vs DreamerV3 .40; 12h pretrain +1h adaptation on V100。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Pretrained model adaptation differs from fresh 100k structured-state control。

主要来源：[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d4c745dcbaf8ef0d7e145754e31b1516-Abstract-Conference.html)。代码核查状态：not located in inspected primary records。

## P09 Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving

作者：Xiaosong Jia; Zhenjie Yang; Qifeng Li; Zhiyuan Zhang; Junchi Yan。

机制与关联：Multi-ability closed-loop driving benchmark。

评分证据：2M annotated frames;44 interaction scenarios;23 weather;12 towns;220 evaluation routes. Clip-count versions differ; not merged.。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Data/coverage evidence rather than method improvement; CARLA sensors differ from SUMO state input。

主要来源：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/017761f94a1cd66d01c041aff85492c4-Abstract-Datasets_and_Benchmarks_Track.html)。代码核查状态：https://github.com/Thinklab-SJTU/Bench2Drive。

## P10 NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking

作者：Daniel Dauner et al.。

机制与关联：Low-cost simulation-based sensor planning evaluation。

评分证据：Four-second nonreactive rollout; navtest TransFuser PDMS84.0, human94.8。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Nonreactive background does not establish interactive traffic performance。

主要来源：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf)。代码核查状态：https://github.com/autonomousvision/navsim。

## P11 Learning Safe Autonomous Driving Policies Using Predictive Safety Representations

作者：Mahesh Keswani; Raunak Bhattacharyya。

机制与关联：SRPL-style predictive safety in WOMD/nuPlan replay with reactive agents。

评分证据：Table I 1000 scenarios: WOMD PPOLag success .81 -> .90; NuPlan .72 -> .66。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Effects depend on optimizer/dataset; peer-reviewed acceptance not verified。

主要来源：[论文](https://arxiv.org/abs/2512.17586)。代码核查状态：not verified。

## P12 Survival Reinforcement Learning: Toward Scalable Self-Supervised RL

作者：Franki Nguimatsia Tiofack et al.。

机制与关联：Online survival actor-critic and goal dwell-time objective。

评分证据：Primary abstract reports 2x–8x stable long-horizon locomotion; authors acknowledge replay approximation in Sec4.2。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：No confirmed venue; online/action conditioning already covered, behavior replay is approximate。

主要来源：[论文](https://arxiv.org/abs/2605.31273)。代码核查状态：official link in arXiv HTML; see mechanism note。

## P13 ChronoSRL: Temporal Geometry for Self-Supervised Reinforcement Learning

作者：Nico Bohlinger; Jan Peters。

机制与关联：Temporal state-action/goal geometry plus goal-time distributions and dwell time。

评分证据：Abstract/metadata screen: seven locomotion/navigation benchmarks; exact effect sizes not extracted。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Very recent preprint; novelty boundary only, no acceptance or safety claim。

主要来源：[论文](https://arxiv.org/abs/2609.36238)。代码核查状态：not verified in this screening。

## P14 Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability

作者：Yushen Liu; Yin-Jen Chen; Ziyi Chen; Tao Wang; Heng Huang; Xugui Zhou; Yanfu Zhang。

机制与关联：Finite-history proxy, candidate-action near-term risk used in value penalty and optimistic/conservative ensemble gating。

评分证据：Primary abstract verified; glucose control and Safety-Gym, no numeric gains extracted。这些数字属于该论文自身协议，不与本项目SUMO成功率直接横比。

适用限制：Action-conditioned risk under partial observability is already prior art; not a full competing-terminal/time law。

主要来源：[论文](https://arxiv.org/abs/2605.14246)。代码核查状态：not verified。

## 分级与新颖性边界

ICML、NeurIPS、CVPR和TPAMI的正式出处已用primary sources核对；ICLR和CoRL作为重要相关来源保留。CCF官网类别页与2026第七版附件的可访问性不一致，本轮没有核准ICLR在最新目录中的具体类别，因此不强行标A/B。未通过年度官方分区表核查CAS Top状态，不把JCR Q1、期刊影响因子或venue声誉代替CAS分区。预印本不等于已录用。

SRPL覆盖state风险时间表示，SVL覆盖action-conditioned survival-Q并明确提示竞争失败/稀疏稠密分解，SRL覆盖在线survival，ChronoSRL覆盖时间几何，P14覆盖partial-history动作风险，TraCeS覆盖终局标签到时序安全credit，DSAC-T覆盖成熟分布价值SAC。因此 proposed route 只能作为需与上述近邻对照的研究假设，不能称已确认首创。

