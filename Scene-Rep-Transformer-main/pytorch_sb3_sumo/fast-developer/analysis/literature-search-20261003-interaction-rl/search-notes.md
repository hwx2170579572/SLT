# 检索记录与范围

日期：2026-10-03。用户授权范围：奖励诊断、近两年文献检索与可行技术路线设计。文件阅读由Luna max完成，根代理整合、核查primary来源和方法逻辑。未增加训练/仿真，自动跟进保持原暂停状态。

## 检索策略

时间窗2024-10-03—2026-10-03。正式发表/first-online与arXiv初稿分开；原始MST+SLT、SAC、PBRS等作为更早基础，不冒充近两年新作。优先PMLR、NeurIPS/ICLR/CVF proceedings、arXiv全文、作者仓库/实验表。搜索摘要用于发现；核心SVL/SRPL/SRL与奖励相关理论进入正文核对。

使用的公开查询族：autonomous driving reinforcement learning world model 2025 2026；safety representations steps-to-cost；reinforcement learning first hitting time distribution critic；survival learning goal conditioned reinforcement learning；reinforcement learning competing risks；distributional soft actor critic refinements；trajectory-level constraint credit；continuous action masking。未将本项目私有方法名、实验值、源码或路径作为外部搜索词。

工具实际返回的MDPI结果未打开/引用；与LLM文本生成、维护调度等明显无关结果未纳入。本文不是系统综述或穷尽新颖性证明。访问失败与年份不一致均保留，不以第三方榜单替代正式metadata。

## 登记的28项候选

P01–P14详见主表papers.md/csv，均保留。下列14项解释未纳入精选或仅作基础锚点的原因：

| 候选 | 来源状态 | 处理原因 |
|---|---|---|
| Think2Drive | ECCV2024；Springer first-online2024-11-24 | 会议/在线日期边界不同；作为原始世界模型驾驶锚点，不称2025新方向 |
| RAD | NeurIPS2025 | 3DGS/log-replay背景不同；主表用Raw2Drive覆盖世界模型线路 |
| Excluding the Irrelevant: Continuous Action Masking | NeurIPS2024 | 有关动作映射理论；当前首版不引入mask，保留方法边界 |
| FREA | CoRL2024 | 主要为可行性约束的对抗场景生成，不是同预算ego策略 |
| Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling | CoRL2025 | 不确定性/约束已知近邻，行人导航分布与车辆任务不同 |
| Solving Parameter-Robust Avoid Problems with Unknown Feasibility using RL | ICLR2026 | 纯avoid/可行域和robust参数问题不直接匹配任务成功目标 |
| Robust Transfer of Safety-Constrained RL Agents | ICLR2025 | 迁移/预训练设定，非当前fresh control核心 |
| Safe Planning and Policy Optimization via World Model Learning | arXiv2506.04828；ECAI书目状态未完全复核 | 已有world-model reward+cost+planning，说明大组合非自动创新；不据未核年份作正式表 |
| Verified Safe RL for Neural Network Dynamic Models | NeurIPS2024 | 形式化模型假设与当前model-free finite-history任务不匹配 |
| Meta-learning how to Share Credit among Macro-Actions | NeurIPS2025 | Atari/StreetFighter宏动作问题，不能直接移植为交通credit结论 |
| Safety-Aware RL for Control via Risk-Sensitive Action-Value Iteration and Quantile Regression | arXiv2506.06954 | 预印本、QR/CVaR近邻；已有DSAC-T作更直接成熟控制基线 |
| Back to Base: Safe Resets with Reach-Avoid Safety Filters | L4DC2025 | 主要是安全重置问题，关联较弱 |
| Safe Learning in the Real World via Adaptive Shielding with Hamilton-Jacobi Reachability | L4DC2025 | 属模型/屏蔽路线，首版不采用 |
| Distributional Reward Decomposition for RL | NeurIPS2019 | 超出窗口，但必须承认奖励/回报多头分解已有先例 |

部分登记项只读primary摘要，没有资格支撑细节或定量结论。最终主表为10篇正式出版+4篇预印本，并非14篇全部满足CCF-A/CAS Top。

## 理论基础（不计近期精选）

- SAC：[ICML2018](https://proceedings.mlr.press/v80/haarnoja18b.html)。
- Distributional RL：[ICML2017](https://proceedings.mlr.press/v70/bellemare17a.html)。
- Time limits：[Pardo等](https://arxiv.org/abs/1712.00378)。
- Policy-invariant shaping：[Ng等，ICML1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)。
- Reward decomposition：[NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/97108695bd93b6be52fa0334874c8722-Abstract.html)。
- 当前强基线原论文：[T-IV2024，arXiv初稿2022](https://arxiv.org/abs/2208.12263)。

## 访问和证据限制

- ICLR SRPL PDF体积导致工具失败，改用作者arXiv全文及官方proceedings对照。
- TraCeS PDF/OpenReview读取失败；只核官方abstract和作者仓库，不引用未经核对的效果数字。
- SafeDrive官方CVF页面间歇读取失败；子代理已核CVF PDF与作者仓库。其方法属规划安全监督，不是在线SAC。
- Bench2Drive abstract与PDF/版本中的clips数量不同；主表保留一致的44 scenarios/220 routes等，不合并clip数。
- CCF总目录声明第七版（2026），AI分类页与附件的读取状态不足以核准ICLR最新等级。使用真实venue名，不擅自给其标A；CAS年度Top未核，不以JCR区替代。

## 收敛后的研究判断

奖励排序审计不支持把奖励系数视为当前首要根因；已知bootstrap与task-time定义必须先对齐。初始“加风险/时间表征”的路线被SRPL/SVL/SRL/P14进一步收紧；事件价值分解公式本身不是新贡献。最后留下的是需验证的policy-aware竞争终局学习路线，必须对照SVL可对齐扩展、SRPL、DSAC-T、TraCeS相关机制及简单多头。研究价值取决于可校准的动作后果和真实成功率/泛化收益，而非命名。

