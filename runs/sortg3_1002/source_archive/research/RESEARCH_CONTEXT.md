# 当前研究上下文

记录依据：用户在本任务中的明确说明、项目源码，以及已有实验产物。
本文件是后续任务的阅读入口；代码细节与实验数值分别保存在下列记录中。

## 当前完成/运行状态（2026-10-02）

`runs/sort2_1001` 中 sorted `sac_mlp_d1_st_rt_3slot` 和 `sac_mlp_d1_st_rt_topo_routeaware_v1` 均已完成 fresh seed0、100000 raw SUMO steps/95001 updates、final 100回合 validation（逻辑seed 10000–10099）。3slot的S/C/T/O=34/49/17/0，shaped/raw mean/std=−1.3883924/9.8976625、−0.15/0.8986100；routeaware为43/32/25/0，shaped/raw mean/std=1.0395563/9.9573802、0.11/0.8590111。两者采用`environment_step_reward_v2`，六分量最大对账误差各为3.55e−15，train/eval diagnostics error均0；checkpoint、训练完成记录和evaluation identity SHA均匹配。逐方法与配对字段见[sorted模块结果审计](analysis/sorted_module_outcomes_20261002.json)和[完成快照](analysis/sorted_pair_completion_20261002.json)。

DARRL r2 medium `sac_mlp_d1_st_rt_topo_3slot`已完成fresh seed0、100000 raw/95001 updates、100回合，S/C/T/O=20/47/33/0，旧raw `mean_return=-0.27`，checkpoint SHA256=`b89d891756dbb8f21dfc5225ecd8efe3e88b3cb70e85ba07dff97b46984dc77d`。它属于独立DARRL r2场景与旧评估schema，不能和sorted shaped return直接比较。见[更新后的全量汇总](analysis/intersection_experiment_summary_20261001.md)。

当前范围共41条非smoke最终评估：sorted26、早期随机4、DARRL r1两条双标/奖励污染历史、DARRL r2九条；39条互斥终局、2条r1污染历史。2026-10-01的38条清单/奖励审计仍是历史快照，不回写原始数值。sorted/depart4七策略均为单训练seed0、fresh100k raw；其100次validation回合使用相同逻辑seed和30个复用的固定traffic模板，配对审计确认每对seed与模板文件名100/100匹配，且30个XML模板bundle哈希一致。这是固定checkpoint在该评估池上的描述性比较，不提供多训练seed稳定性估计。新sorted结果的shaped `mean_return`不得与旧raw `mean_return`混读。

本次结果、身份与配对协议已由模块归因报告进一步审阅：[sorted 模块归因](analysis/sorted_module_attribution_20261002.md)。报告将结果视为单训练seed下的固定策略观察，不当作多seed因果结论。记录规则：正式结果只有在final checkpoint哈希、评估回合数/seed和评估文件身份核验后写入汇总；核验前只记带时间的运行快照。

## 2026-10-02：sorted 模块归因与后续优先级

在本条单seed、fresh 100k 的 sorted/depart4 链中，ST-RT 为 63/37/0（成功/碰撞/超时），是当前增量方法中的最佳观测；ST-RT+Topo 为39/39/22，加入 route-aware goal mask 后为43/32/25，ST-RT+3slot为34/49/17，Topo+3slot为41/56/3。route-aware 的25个超时均终止在无法接到目标下一edge的 `-E1_0`；其合法候选attention mass约为1、bypass/fallback为0，表明goal mask有效但不约束动作选择。当前3slot将活跃readout从65,792参数的联合非线性头改为16,512参数的分槽线性头，下游actor/critic仍可混合槽输出；该实验不能否定语义分槽本身。

下一步建议先比较当前冻结代码的ST-RT与参数量匹配的非线性3slot，再单独评估goal-only Topo。两者都只是可证伪实验建议，尚未实现或启动；现有拓扑/goal residual初始化尺度已为1e-3，不能把再加小残差门控当作新发现。结论限于一个训练seed和重复使用的30个交通模板；新评估shaped return与旧raw return不可直接横比。完整配对、诊断和实现边界见上述归因报告及 `analysis/sorted_module_outcomes_20261002.json`、`analysis/sorted_module_diagnostics_20261002.json`。

为后续 incremental 训练/评估，已授权并补入 actor-intent post-LN 实际 delta 与同支持集 uniform 关系基准；已有 goal post-LN 指标复用，不重复实现。历史 source/results 未改写；这次没有启动训练、评估或 SUMO。相关 encoder 30 项与 module diagnostics 7 项测试通过，4项目标测试是子集；旧 full delegate 不覆盖新指标，采集边界见 [module diagnostics protocol](analysis/module_diagnostics_protocol_20261001.md)。本次分析与诊断补充完成后，automation-3 已由应用工具确认为 PAUSED，避免重复跟进。

## 2026-10-01 22:26 启动快照（历史）：旧场景 intersection_sorted / depart_scale=4.0

该阶段主场景从上一阶段 DARRL medium p=.03 切回正确三路固定模板 `intersection_sorted`、`depart_scale=4.0`。DARRL结果保留为独立历史阶段，不与本轮场景直接排名。阶段初始fresh队列为 `sac_mlp_d1_st_rt_topo_routeaware_v1` 和 `sac_mlp_d1_st_rt_3slot`，seed=0、每路100,000 raw SUMO steps、两个独立 CUDA worker、每10,000 raw steps checkpoint、final validation 100回合，不 resume。正式根为 [`runs/sort2_1001`](../../../runs/sort2_1001)；其初始22:31运行快照见下一段和更上方的2026-10-02完成更新。启动参数、有效交通池及源码快照见该目录 `suite_manifest.json` 与 `source_archive/manifest.json`。

启动前，新增/更新的ST、RT、Topo、route-reachability、三槽激活/梯度诊断通过43项定向测试；两个方法分别通过300-raw-step/1-episode smoke。route-aware首次 smoke 在 [`runs/s2sm_1001`](../../../runs/s2sm_1001) 成功；3slot首次 smoke 的训练成功、evaluation 因hook临时`components`字典被错误当作数值指标而失败，原失败日志保留，修复后在 [`runs/s2sm_r1_1001`](../../../runs/s2sm_r1_1001) 单独重试通过。smoke只证明路径和日志可运行，不是算法结果。当前诊断采样/分母/归因限制见[诊断协议](analysis/module_diagnostics_protocol_20261001.md)与[本阶段计划](analysis/sorted_routeaware_3slot_plan_20261001.md)。

该阶段评估主回报口径为`environment_step_reward_v2`（逐decision实际环境shaped reward之和），同时单独保存`raw_episode_return`/`raw_mean_return`供历史对照，并保存可用的奖励分量对账；先前38条结果奖励审计是修复前历史快照，其raw评估口径不代表本轮新schema。2026-10-02结果及当前未完成队列以本文件开头更新为准；DARRL阶段与旧sorted统计仍按各自原始协议保留。

- [完整模型实现与模块消融地图](analysis/full_mst_slt_implementation.md)
- [历史实验与证据边界](analysis/experiment_history.md)
- [本轮 SAC+MLP 与 D1-ST-RT 100k 结果分析](analysis/mlp_strt_100k_analysis.md)
- [本轮逐episode行为诊断数据](analysis/analysis_100k_behavior.json)（由 [只读分析脚本](analysis/analysis_100k_behavior.py) 生成）
- [较早 MST+SLT 与 D1-ST 连续 100k 配对分析](analysis/continuous_100k_mst_vs_st.md)
- [Topo 与三槽增量实验协议](analysis/topo_3slot_protocol.md)
- [Full、MST+SLT 与 SAC+MLP 的实现地图](analysis/full_mst_slt_implementation.md)

## 1. 用户确认的研究定位

| 对象 | 研究角色 |
| --- | --- |
| **full** | 当前待研究的完整方法，包含需要拆解验证的表示模块和辅助目标 |
| **MST+SLT** | 强基线，用于衡量完整方法与逐步构建的变体 |
| **SAC+MLP** | 纯强化学习基线，作为增量模块实验的起点 |

当前工作是：**在 SAC+MLP 的基础上，将 full 拆成模块，逐个加入，观察效果并定位问题。**
不要把 full 误写成基线，也不要把 MST+SLT 误写成正在提出的新方法。
这里的“强基线”是用户指定的研究角色，不意味着它在所有现有训练阶段都取得最高分。

## 2. 研究对象与现有工作

任务是 SUMO 无信号交叉口中的自车通行：减速让行、选择可通过的时机，并控制碰撞和超时。
历史工作包括优化设置排查、奖励塑形、路线观测修正、SUMO 默认让行示范、行为克隆和密度诊断。
较新的实验转向 `intersection_sorted` 场景、发车时间缩放 `depart_scale=4.0`，开展 D1 表示增量和路线接入方式的消融。

当前主要 SAC 分支为：

`SAC+MLP -> D1-ST -> D1-ST-RT -> D1-ST-RT-Topo -> D1-Full`

这表示实验的模块递进意图，不代表各相邻实现都只改变一个变量：

- `D1-ST` 关闭路线 token、拓扑和三槽分支，但仍存在 MLP map fallback，不能称为“不读取地图”。
- `RT` 研究路线 token 接入；`gate/late/ego/edge` 是不同接入方式的诊断分支。
- 历史 `Topo -> Full` 同时改变三槽投影、Graph-SLT 辅助目标、slot balance 和算法类；不能单独归因于 slots。新增的 `Topo -> Topo+3slot` 配置将结构投影与两个辅助目标拆开，尚待正式实验检验。
- `sac_mlp_d1_full` 继承 `sac_mlp` 的父设置；`hsac_mlp_base_d1_full` 继承 `hsac_mlp_base`。混合动作分支与 SAC 主分支需要分别记录。

完整网络、父类实现和实际梯度路径以实现地图为准。

较早的新主目录连续 100k 对比中，MST+SLT 成功 53%、D1-ST 50%；它不同于旧备份目录 c100000 续训的 25% / 33%。本轮另完成了从零训练的 SAC+MLP 与 D1-ST-RT 各 100k raw steps：100 个共同评估回合中，SAC+MLP 为成功/碰撞/超时 22%/38%/40%，D1-ST-RT 为 63%/37%/0%。配对结果含 23 个 SAC 碰撞→ST-RT 成功、7 个 SAC 成功→ST-RT 碰撞；40 个 SAC 超时中 25 个对应 ST-RT 成功、15 个对应 ST-RT 碰撞。每种方法只有一个训练 seed（0），不能据此声称多种子稳定优势。SAC 超时组平均目标速度约 8.06 m/s、实际速度约 1.17 m/s，停驶比例约 74%；不支持“策略主动选择低速”的解释。静态路网 [map.net.xml](../envs/sumo/original_scenarios_v1/intersection_sorted/map.net.xml) 显示，40 个 SAC 超时回合均停在 route `[-E1,-E0]` 的 `-E1` 末端 lane 0/1（38/40 在 lane 0、2/40 在 lane 1），但该路网从 `-E1` 到下一 route edge `-E0` 仅有 `fromLane=2 -> toLane=1` 的直接连接。因此，终端停滞与当前 lane 不连通有直接证据；策略为何进入这些 lane、能否/为何未能提前换入可连接车道仍待验证，不据此称为控制实现 bug。主分析与可复核诊断数据见上方入口。

纯 SAC 基线已完成必要的有效性与公平性检查；保留 22%/38%/40% 及高目标速但低实际速的停滞观察，不把后续优先级放在优化 SAC 基线或针对超时另做干预实验。当前观察到 D1-ST→D1-ST-RT 成功率为 50%→63%、碰撞率为 50%→37%，两者均无超时；这是单 seed 结果，不要求每个增量都超过 MST+SLT。MST+SLT 的 53% 是完整方法的最终对照目标。

当前实验约束为**单训练 seed**。下一阶段的两个方法均为 fresh 100k raw steps、seed 0、depart_scale=4.0：`sac_mlp_d1_st_rt_topo` 与 `sac_mlp_d1_st_rt_topo_3slot`。后者现在已加入 D1 runner：在同一 ST-RT+Topo 增量前向上启用 32/64/32 三槽输出，关闭 Graph-SLT/SBS，且 `representation_coef=0`、`slot_balance_coef=0`。两者分别与已完成的 fresh 100k ST-RT 对照；3slot 还与 Topo 对照，以区分拓扑和三槽结构增量。MST+SLT 仍是完整方法的最终对照目标。配置及研究问题见[实验协议](analysis/topo_3slot_protocol.md)。此前的 `experiment_history.md` 和常见 sorted topology 结果目录中未发现 Topo 的评估/完成文件，不能据此断言它从未运行。当前不安排 seed 1/2。

结构配置、增量 forward 的一致性检查和常规诊断接口已完成静态核对与单元测试；这不表示任何正式训练已开始或方法有效。两种方法的成功/碰撞/超时仍须作为联合结果报告，不以是否超过 MST+SLT 决定是否保留模块。新增的训练/eval 激活与梯度诊断及指标边界见第 7 节和[实现地图第 10 节](analysis/full_mst_slt_implementation.md)。
交通划分注意：当前 yield_v2 depart 缩放 factory 会覆盖底层 train/evaluation 分区选择并返回全量缩放路线池。MST 参数中的 `frozen_80_20` 不能作为 holdout 已生效的证据；两方法走同一覆盖路径。现有结果不能直接支持未见交通变体泛化。后续需保存实际 train/eval 路线清单；改变划分时另建共同协议，避免与旧结果混比。

## 3. 证据使用原则

1. 用户说明确定研究目的；源码和运行配置确定实际实现；原始结果文件确定已记录的实验观察。
2. README、根 manifest 和旧实验日志不是最新状态的唯一来源。它们没有覆盖全部后续 D1 实验，部分状态相互不一致。
3. 不混合不同场景、车流密度、动作头、训练奖励、训练预算或检查点选择的数字。
4. 区分训练随机种子和评估回合种子；100 个评估回合不等于 100 个独立训练种子。
5. 既有 50k -> 100k 续训加载模型但没有恢复 replay buffer，warmup 也改变。应作为特定续训协议记录，不能视作原训练的无缝延长。
6. 结果下降是观察；过拟合、路线干扰、优化不稳定等解释在得到区分性实验前均是待验证假设。
7. 研究结果与代码调查均为快照；有新运行或代码变化时，刷新对应证据而非沿用旧表。

## 4. 协作与维护

- 用户要求文件搜索、查看等基础操作交给 **Luna（max）子智能体**。
- 主代理负责综合证据、理解研究问题、检查归因边界和维护这些记录。
- 前一阶段以理解实现和保存研究上下文为主；之后用户授权为常规训练/评估加入只读诊断，并安排本轮两方法实验。本节记录该后续要求与实现验证。
- 后续记录新增实验时，应至少保存：方法与具体开关、来源文件、场景/密度、训练种子、raw steps、评估种子和回合数、检查点、成功/碰撞/超时、训练或续训条件，以及结论的证据等级。

## 5. 运行时行为诊断：实现与短程验证（2026-09-29）

用户要求能由常规训练和评估取得的信息随原流程直接保存，避免之后为此重复大量环境回放。若确实需要额外回放、干预或对照实验才能回答研究问题，应在本轮训练与评估结束后，依据当时已有证据安排；这不是永久禁止额外诊断。新增的常规采集只读 TraCI 事实，不改变 policy observation、reward、控制指令、随机数调用或优化步骤；碰撞事件与常速度 OBB/CPA 预测必须分开报告。坐标约定为车辆中心世界坐标（m）、物理 heading（rad，世界 +x 逆时针）、物理速度（m/s）；raw snapshot 带真实 SUMO `step_seconds`、模拟时间、episode/raw-step 计数和状态时间。被 SUMO 移除的 ego 若只能回退到移除前状态，标记 `pre_step_removed`，不与移除后的邻车计算同刻 TTC。

基类 `SumoSceneEnv` 与覆盖 `reset()` 的 `PaperSumoSceneEnv` 均挂接 reset/raw-step/decision/close hook；Paper 环境重置时 `close()` 只刷新 recorder 缓冲，输出流仍由外层 wrapper 最终关闭。`PaperSumoSceneEnvV4.step()` 委托给基类。reset provenance 保存本次真实 `simulation_seed`、requested seed，以及 `_info_dict` 已提供的 traffic variant/seed/cycle/roll。静态车长/宽与路线每车每 episode 缓存；动态 road/lane context 在决策边界采样；当前 actor state 复用环境已读取的历史状态。SMARTS/CARLA heading 仅为 telemetry 转换为物理坐标，不更改策略状态。

`sac.py` 在第 1 次、随后每 1,000 次 gradient update，从当前已计算的 critic 与 target 张量 detach 后记录 `diagnostic/q1_mean_sampled`、`diagnostic/q2_mean_sampled`、`diagnostic/target_q_mean_sampled` 和两路 `diagnostic/q*_abs_td_mean_sampled`。这些稀疏 scalar 不引入额外 forward/backward 或 RNG 调用；下述短路径 pipeline smoke 已确认五项字段被采集。

真实 SUMO 透明性 smoke 使用 `intersection_sorted`、reset seeds 20260929/20260930 和相同固定动作序列，对比诊断开/关环境：132 个 raw tick、44 个决策的观测、reward、终止标志、raw-step 数及关键事件 info 完全一致；两次 reset 后仍成功写出 2 个 episode、全部 raw/decision 记录。TraCI deltaT 为 0.1 秒，记录的 traffic provenance 与当前选择一致，诊断读取错误为 0；4,244 对相邻 tick 移动 actor 的速度方向与中心位移方向中位 cosine 为 0.999999999999999。可复现脚本和输出位于 `pytorch_sb3_sumo/_tmp_verify/behavior_env_smoke.py` 与 `pytorch_sb3_sumo/_tmp_verify/behavior_smoke_20260929/result.json`。该短 smoke 只验证透明性、坐标方向和流生命周期，不是训练结果，也不支持策略性能结论。

随后短路径 pipeline smoke 已由主代理核实 exit 0：显式方法为 `sac_mlp` 与 `sac_mlp_d1_st_rt`；每个方法各训练 300 raw steps、101 个 decision、241 次更新，新增 5 个 Q/TD logger 字段均已被采集。每个方法均完成 8/8 smoke 评估（每个评估阶段 4,800 raw steps、1,600 decisions），诊断 errors 为 0，风险与速度汇总有有效覆盖。较早 `diag_smoke_20260929_01` 在 Windows 长路径触发失败；该目录是已解决的旧失败，不是当前阻塞。短路径 smoke 验证的是采集管线，不等价于正式 100k 训练，也不支持方法优劣结论。

2026-09-29 的 SAC+MLP/ST-RT 运行显式指定 `--methods sac_mlp,sac_mlp_d1_st_rt --max-steps 100000 --depart-scale 4.0 --behavior-diagnostics` 并移除了 `--smoke`。当时 launcher 默认方法为 `mst_slt` 与 `sac_mlp_d1_st_rt`、默认 raw budget 为 50k；不能依赖默认值重建该实验。本节记录的是该历史运行，2026-09-30 当前 Topo/Topo+3slot 启动参数和快照见第 7 节及[实验历史第 8 节](analysis/experiment_history.md)。任何 smoke 均不替代正式结果，也不能据此提前宣称方法性能结论。

## 6. 本轮正式运行（2026-09-29，已完成）

SAC+MLP 与 D1-ST-RT 两个 CUDA worker 均以 seed 0 从零训练 100,000 raw steps，未 resume；warmup 为 5,000 raw steps。两者 exit code 均为 0，并完成每方法 100 个共同评估回合。运行根目录：`D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d0929_100k_diag`。评估结果依次为成功/碰撞/超时 22%/38%/40% 与 63%/37%/0%；这些是单训练 seed 结果，不是跨训练种子的稳定性结论。可复核的配置、汇总和局限见[本轮主分析](analysis/mlp_strt_100k_analysis.md)、[运行记录](analysis/run_20260929_100k_diagnostics.md)及[行为诊断数据](analysis/analysis_100k_behavior.json)。离线诊断数据质量：SAC 36,795 条 raw 记录中 36,773 条风险可评估，ST-RT 26,899 条中 26,836 条风险可评估；ego 缺失为 0，诊断错误行为 0。无效风险行主要是终止时 ego 已被 SUMO 移除、只能使用标记过时的移除前状态；不可与当前帧邻车状态混算。

## 7. D1-Topo 与三槽结构消融：实现审查（2026-09-30）

当前队列为单 seed 0 下的 `sac_mlp_d1_st_rt_topo` 与 `sac_mlp_d1_st_rt_topo_3slot`，各自从零训练 100k raw steps。正式运行已启动，根目录为 [`runs/t0930_0023_topo3_100k`](../../../../runs/t0930_0023_topo3_100k)；[supervisor 快照](../../../../runs/t0930_0023_topo3_100k/launcher_supervisor.json)记录 PID 56072。2026-09-30 00:32:12 +08 的文件快照显示 Topo PID 17496 和 Topo+3slot PID 9240 均为 training，各完成 1,195 raw steps、updates=0，仍在 5,000 raw warmup 内；随后 file_audit 的进度复核为两路 raw=2,392、updates=0，仍处于 warmup。前一个数字有明确快照时间，后一个是之后复核的阶段性快照；两者都不能当作完整预算或结果。本节只记录代码、测试和运行状态，不记录新方法的性能结论。用户确认的阶段含义、匹配比较与解释边界见[Topo/3slot 实验协议](analysis/topo_3slot_protocol.md)。

`Topo_3slot` 显式设置 `use_route=true`、`use_topology=true`、`use_slots=true`、`use_incremental_slots=true`，并设置 `use_graph_slt=false`、`use_sbs=false`、两个辅助系数为 0。编码器仍走 `IncrementalTopoEncoder._forward_incremental`；它与无 slot Topo 使用相同的 route→topology→vehicle interaction→temporal/social/goal 上下文构建，输出端由 384→128→128 的 MLP 改为 ego/social/route 三个输出槽（32/64/32），拼接后仍为 128 维。`use_incremental_slots` 避免无 slot Topo 基线和 3slot 方法因默认 `use_slots=true` 转入父 V2 全开路径。旧 Full 默认仍走父 V2 路径，并保留 Graph-SLT 与 SBS 配置。相同上下文进入输出头的精确等值测试通过；槽结构会改变输出头参数化/容量，当前没有声称参数量匹配。

诊断只采集正常 forward/backward 已计算的张量：训练启用后首次 forward 采样，随后每 256 次 encoder forward 采样；评估在每次现有 policy prediction 后、环境 step 前采样。梯度在首次及随后每 1,000 次 critic 更新的 critic TD backward 后、梯度裁剪前读取。它不增加 forward/backward、优化步骤或随机数调用。训练阶段的 `source=train_replay` 是配置标签；online extractor 也由 rollout actor 共享，因此不能据此断言每条训练激活样本必然来自 replay batch。解释具体快照时结合事件类型及 `diagnostic_sample_index`、`diagnostic_sample_forward_index`、sample age、batch size、`grad_enabled`；训练激活不绑定为当前环境 episode。评估激活来自现有预测。单元测试覆盖增量路径、默认 Full 父类委托、采样元信息、无 RNG 前进、槽投影梯度和 V2 回归；对应小范围测试共 12 项通过，相关文件 `py_compile` 通过。正式结果与当前性能均未知。

拓扑、关系和三槽字段的分母与有效性边界见[实现地图第 10 节](analysis/full_mst_slt_implementation.md)。lane 查询指标按有效 actor-time query 统计；关系值按实际有效 actor pair（排除对角线）统计并同时记录 relation 边数；batch=1 的跨样本方差与跨样本能量相关系数标记无效。梯度组包括 topology encoder/attention、topology/goal fusion、vehicle relation layers 和 ego/social/route projections；None gradient、零 gradient 和非零 gradient 分开保存。

### 2026-09-30：原始 Topo/Topo+3slot run 的 I/O 中断

原 root `runs/t0930_0023_topo3_100k` 中，Topo+3slot worker 在约 5,651 raw steps 因诊断 `summary.json.tmp.replace(summary.json)` 返回 WinError 5 而退出；这不是模型 forward、梯度或 checkpoint 错误。Topo 检查时仍正常训练，但两路共用受影响的写入代码，因此修复后停止了经过 PID/启动时间核对的旧进程树，保留原产物；旧三槽为运行失败，旧 Topo 为主动中断，均不作为完整 100k 科学结果。

修复版已于 2026-09-30 01:24:05（Asia/Shanghai）在 `runs/t0930_topo3_retry01` 从零启动：supervisor PID 50304，Topo PID 8364，Topo+3slot PID 62860。两路均 CUDA、seed 0、连续 100k raw steps、5k warmup、最终各 100 回合评估；不恢复旧 checkpoint、不使用 smoke 设置、不改变模型和损失。相关 I/O 故障注入及既有测试共 29 项通过。重启后的连续核验已观察到 Topo raw 6,578 / updates 1,578、3slot raw 6,577 / updates 1,577，均越过原故障点、状态 training、诊断错误数 0，且无新异常栈或待发布汇总；这是运行快照，不是 100k 完成或性能结论。详情见[summary I/O 恢复记录](analysis/topo_3slot_io_recovery.md)。

## 8. 2026-09-30：随机交叉口 Bernoulli 交通协议与 headless 验证

新增 `intersection_random_{low,medium,high}_v1`，不替换 `intersection_sorted`。每档只改变三条已存在的 background lane-0 直行流率，保持 `-E3→-E0 : E0→E3 : E2→E1 = 200:150:240`，总计划到达率分别为 590/1,180/1,770 veh/h。XML 写 `probability=q/3600`（每秒），SUMO 在 0.1s 步长内部按步长缩放为每步 `q/36000`；背景车驶完路线后离开，不循环再入网。单局 SUMO traffic seed 与训练 seed 分开，train/validation/test 实际 seed 域不相交；默认开发评估用 validation logical seeds 10000–10099。配置、seed 映射、车型权重和资产哈希见[随机流量协议](analysis/random_intersection_v1_protocol.md)、[生成器](../envs/sumo/random_intersection.py)及各场景 `scenario_manifest.json`。

SUMO 1.25.0 headless smoke（step 0.1s，跑到 180s）固定 validation logical seed 10000：low 19/19、medium 55/55、high 69/69 planned/actual background departures；三档均 0 pending/missing、0 辆仍在网内。正 departDelay 数量/最大延迟依次为 0/0s、7/1.1s、12/2.4s。观测数是单次 Bernoulli 实现，不能当作档位标定。Medium logical seed 10000 完全复跑的计划表哈希一致；logical seed 10001 的计划表不同。同一 seed 下三路 arrival-time 集合随 low⊂medium⊂high，但共享计划时刻的车型抽样不全相同，故跨密度不是逐车类型匹配。每个 case exit 0；保留警告 `Ignoring junction logic for junction 'J1'`。该 smoke 未改变 ego 控制来实测 policy-independence。

协议 4 项测试通过；主代理另核三档 PaperEnv reset/5-step、seed 与 split 检查通过。既有 `test_paper_reproduction.py` 在独立临时目录单独运行 48 项通过；此前合并批跑中的临时目录 ACL/模块导入顺序问题不作为全套通过证据。smoke 命令脚本和可复核 summary 位于 [`runs/random_intersection_v1_validation`](../runs/random_intersection_v1_validation/bernoulli_smoke_summary.json)。这只验证 SUMO 流量语法、固定 seed 重现和实际插入，不是训练或策略结果；低中高还未标定为学习难度。没有新的 100k 训练。新旧协议的发车过程、模板/随机流和 seed 隔离不同，不能将跨协议成绩差归因于模型模块。

入口补充验证见[协议记录](analysis/random_intersection_v1_protocol.md)：workspace 根 PaperEnv smoke 在同一 validation logical seed=10000 下，50s ego 发车预热后各执行 5 个决策步，低/中/高驶入背景车数为 9/28/35、峰值在网数为 5/15/17，pending 为 0。D1 `sac_mlp_d1_st_rt` medium factory 的 reset+1 step 检查观察到 train seed 217 与 test seed 1500000217 分别进入 train/test split，流率和无重插入配置一致，overlay manifest 显示原资产与 map/ego route 未修改；该检查没有 learning update，命令的显式退出码未捕获。eval-only 可从原训练 checkpoint 读取并将结果写入独立 `--output-dir`；随机场景 guard 会拒绝在已有 validation 结果目录写 test 结果。没有新的训练、策略表现或难度标定结论。

## 9. 2026-09-30：p05 中档变体、warmup 采集与 matched baseline 运行

保留原 `intersection_random_medium_v1`（三路请求 400/300/480 veh/h，总 1,180）；另增 `intersection_random_medium_p05_v1`，三条背景进口 lane 0 每个 0.1 s 仿真步各以 `p=0.5` 生成请求，即每路期望 18,000、合计 54,000 计划 veh/h。该数值是**计划需求**，不等于实际进入网络的流量，也未据学习难度标定。p05 使用 actual SUMO seed 与 route ID 派生独立 SHA256 种子，再用 `Random.random()` 预采样 `[0,130)` 的逐车时刻；SUMO 保留因插入受阻而 pending 的车辆。冻结 medium 与 p05 的 map、ego、120 个车型、3 routes 和 3 个 driver distributions 完全一致；旧 medium 的原始 400/300/480 native-flow 配置不变。生成器修正为解析车辆对顶层 route 的引用，同时严格检查 `-E1 -E0` edges；源资产 hash 无变化。协议细节见[随机交叉口协议增补](analysis/random_intersection_v1_protocol.md)。

新的只读 warmup checkpoint 适用于 low、原 medium、high、medium-p05 四个随机场景：在正常 reset 的第 300/400/500 个 raw tick（30/40/50 s，0.1 s/tick）读取现有仿真状态，不增加仿真步。`info.warmup_traffic_checkpoints` 随已有 train/eval diagnostics episode `reset_info` 写入 `episodes.jsonl`。字段分别记录在网与按 route 车辆数、当前在网背景车中速度 `<0.1 m/s` 的停驶数、入口 lane occupancy/halting 数、到时请求、累计插入、pending 和插入延迟。pending 是**路网外**等待插入量，不并入在网车辆或停驶/排队分母；p05 的到时请求来自预采样时间表，native-flow 的请求量以“已插入+pending”估算，并用 `request_count_source` 标明口径。

matched baseline 配对仍只使用原 medium 与 medium-p05 两个中档：目标条件是各自 fresh 从零训练 100k raw steps、seed 0、CUDA、`learning_starts=5000`、每 10k raw steps checkpoint、final evaluation 100 episodes，其他设置保持已有 SAC+MLP baseline；四场景 30/40/50 s warmup 采集不改变训练场景范围。初始 root [`smlp_p05_a0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_a0930>) 的 supervisor PID 52540 下，medium worker PID 15900 首次观察为 raw=1,497/updates=0，之后仍保留继续运行。初始 p05 worker PID 19128 在 raw=144 退出，远低于 5,000 learning start；原因是动态 source 文件固定 stem `episode` 导致跨 episode overlay cache 名冲突、hash drift，不是方法学习失败。该 attempt 日志保留在 rootA，不能作为有效训练结果。

修复只给动态 schedule overlay cache 文件名附加完整内容 hash 的前 16 位，manifest 仍核完整 hash，静态 medium 路径保持不变。实际 cache factory smoke [`cache_reuse_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_cache_smoke_0930/cache_reuse_smoke.json>) exit 0：同一 env 不同 p05 seed 得到不同 overlay identity；不同临时 env 的同 seed schedule hash 一致且 manifest 有效；三个 reset 均有 3 个 warmup checkpoints、scale=1、无额外车辆、endless=false。此后 p05 fresh-only worker PID 90004 于 2026-09-30 11:27:15 +08 在新 root [`smlp_p05_b0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930>) 启动；arguments 核实为 `sac_mlp`、p05、100k、seed0、CUDA、`smoke=false`。11:28:26 raw=1,193/updates=0 到 11:29:26 raw=1,790/updates=0，仍在预期 5k warmup；9 个训练 episodes 均已有三处 checkpoint。此为短期运行快照，updates=0 不等于失败；medium 未停止，且其此处仅保留首次进度观察。该配对没有 100k 完成或最终 evaluation 证据，不作方法性能判断。细节与 source-backed 配置见[随机交叉口协议](analysis/random_intersection_v1_protocol.md)。

启动前 factory smoke 的 [`factory_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/p05fac_20260930_luna01/factory_smoke.json>) 为 `status=passed`：low/medium/high/p05 train reset 都产生 30/40/50 s 快照；p05 validation seed=1,000,000,000 也产生三点，五份 train/eval `episodes.jsonl` 均带 warmup reset_info。中档 train 与 p05 train 各执行一次 sampled-action step；没有 learning update。p05 seed 0 的 `[30,40,50]s` 到时请求/实际插入/pending 为 `[438,602,753]/[53,68,83]/[385,534,670]`，validation seed 1,000,000,000 为 `[454,611,770]/[48,62,79]/[406,549,691]`；每路每计划时刻最多一车，计划数分别 1,952/1,929。50s 时两例在网背景车为 30/33，停驶数为 0/0；pending 是网外插入队列，不是路网占用或停驶车辆。它是单次环境/日志路径检查，不是流量长期估计、难度标定或策略证据。逐 checkpoint 数据见协议增补。

最新 paired progress 快照（2026-09-30 11:31:01 +08）确认 medium worker PID 15900 在 rootA [`smlp_p05_a0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_a0930>) 为 raw=23,619 / updates=18,619 / 79 完整训练 episodes；p05 fresh worker PID 90004 在 rootB [`smlp_p05_b0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930>) 为 raw=3,886 / updates=0 / 13 episodes、training 状态。p05 argument 明确 100k、seed0、CUDA、no-smoke、no-resume；0 updates 仍是 5k warmup 期。初始 rootA p05 PID19128 在 raw144 因动态 schedule overlay cache key 复用导致 source hash drift 退出；不属算法失败。只对动态 schedule overlay 名增加 SHA256 前缀后，cache reuse smoke exit 0，换 seed 得独立 overlay，同 seed跨新 env可复用。medium继续在rootA运行，p05在rootB从零重启；原失败attempt保留。活动配对映射与完整失败/修复时间线见[实验历史第13节](analysis/experiment_history.md)及[协议](analysis/random_intersection_v1_protocol.md)。这是进度快照，无100k完成或final evaluation，不作性能结论。

## 10. 2026-09-30：随机中档 traffic 的 final 100k 结果

上面的 11:31 状态是启动期快照，现已被 completion/evaluation 文件更新：medium rootA final 为成功/碰撞/超时 75/25/0，p05 rootB 为 12/88/0；旧 `intersection_sorted`、depart scale 4 的有效对照为 22/38/40。三个 `sac_mlp` 运行均为 seed 0、fresh 100,000 raw steps、95,001 updates，且评估使用 final checkpoint 与每场景 100 回合；rootA raw=144 的 p05 失败尝试排除。当前工作中档固定为原 `intersection_random_medium_v1`（400/300/480 veh/h，总 1,180；每 0.1 s 每路概率 1/90、1/120、1/75）；p05 的 54,000 requests/h 作为压力变体，不据此重新调 SAC。跨 `intersection_sorted` 的结果差同时改变 traffic 发车协议、需求、模板复用及 endless reinsertion，不能作模型因果归因。仍只有一个训练 seed，也没有同 traffic 下的 MST+SLT ceiling。详见[中档交通校准报告](analysis/scenario_medium_calibration_20260930.md)、[结构化结果](analysis/scenario_medium_calibration_20260930.json)和[实验历史第14节](analysis/experiment_history.md)。

## 11. 2026-09-30：用户指定 p=0.3 / p=0.2 变体已启动

本轮实际新增 `intersection_random_medium_p03_v1`（每条背景进口每 0.1 秒 p=0.3）和 `intersection_random_medium_p02_v1`（p=0.2）；三路总请求量分别为 32,400 / 21,600 辆/小时。沿用 p05 的逐步 Bernoulli 生成器及其余交通配置。此前建议的 p=0.015/0.03/0.05 三档未实施，不要混同本轮数值。

两路 SAC+MLP 已用两个 CUDA worker 从零运行，各 100,000 raw 控制步、训练 seed=0，训练结束后自动进行 100 回合 validation 评估。正式运行目录为工作区 `runs/smlp_p03p02_0930`，状态入口为该目录的 `launcher_status.json`，配置入口为 `launcher_manifest.json`。启动验收时两路均已超过 5,000 raw 步并开始梯度更新；这不是完成状态，也没有新的 100k 性能结论。正常训练和评估继续保存 30/40/50 秒预热及行为记录。详情见 `analysis/experiment_history.md` 第 15 节及 `analysis/random_intersection_v1_protocol.md` 的新增变体节。

## 12. 2026-09-30：p02 / p03 100k 已完成，确定性选道失败已定位

两路正式完成各 100,000 raw / 95,001 updates，final_model 的 100 回合 validation 为 p02 成功/碰撞/超时 5/25/70，p03 为 17/83/0。当前最直接证据：p02 的 70 个 timeout 中 69 个终态位于 -E1 的 lane0/1，而 -E1→-E0 的唯一连接来自 lane2；策略请求目标速度并非零，但反复向低索引车道换道。p03 多数横向命令在左侧边界无效，保留合法车道后高速进入路口碰撞。50 秒在网车辆数 p02/p03 为 29.39/29.25，实际入场 80.36/80.71，主要差别为网外 pending；不能由请求概率推断实际难度顺序。

共同训练目标需注意：wrapper 覆盖底层 reward，实际每 decision 为 terminal_bonus -0.01 +0.02×进度米数；本地 4-step 回放累计奖励后 bootstrap 只用 gamma 而非 gamma^4，是源码明确保留的 released 实现口径，已经 CPU 人工探针复核。timeout 仍作为截断 bootstrap。当前未改实现、奖励或超参数，未额外运行 SUMO 评估/训练；策略为何形成该选道模式仍需与随机训练/确定性评估及目标定义分开归因。详见 `analysis/p02_p03_100k_analysis_20260930.md` 与配套三个 JSON；历史记录第 16 节已更新，上一节启动快照保留。

## 13. 2026-09-30：按现有三档模板新增 DARRL 参数参照场景

用户要求以现有 low/medium/high 三档为模板，分别新增 `intersection_random_darrl_low_v1`、`intersection_random_darrl_medium_v1`、`intersection_random_darrl_high_v1`，每条背景进口车道每 0.1 秒发车请求概率分别为 0.03/0.05/0.07。背景 departLane/arrivalLane 均为 1；ego minGap=1 m；所有背景车型及ego的 jmIgnoreFoeProb=0；SUMO collision.action=remove。

用户随后明确保留当前 SUMO ego 碰撞事件或几何重叠判据，只改场景参数，不移植 DARRL legacy 距离/自车消失启发式。保留既有路网、路线、ego lane2/50秒释放、SMARTS观察动作契约、奖励和训练评估流程。新三档为独立场景，不覆盖原六个随机场景；三档名称尚未用成功率标定难度。

来源：用户提供 `D:\Program Files (x86)\paper\DARRL-main`；不是整体替换为该仓库的环境。配置、参数语义与验证记录入口见 [新增场景说明](analysis/random_intersection_darrl_v1_setup.md)。新场景沿用正常预热30/40/50秒统计，进口统计采用配置中的lane1。本次仅做配置与环境验证，不启动100k模型训练。

## 14. 2026-09-30：DARRL low / medium 的 SAC+MLP 100k 首跑

用户已授权启动刚新增DARRL组的低、中两档（用户消息中的darr1按本组DARRL理解），2个CUDA worker并行、单训练种子0、各从零连续100000 raw steps。2026-09-30约19:40:59（Asia/Shanghai）已发起正式运行，输出根为工作区`runs/smlp_darrl_0930`。

启动器`launch_sac_mlp_random_pair.py`新增独立pair `darrl-low-medium`，选择`intersection_random_darrl_low_v1`和`intersection_random_darrl_medium_v1`；使用既有D1 `train-method --method sac_mlp`流程，显式传入`--max-steps 100000 --checkpoint-frequency 10000 --behavior-diagnostics --depart-scale 1.0 --eval-traffic-split validation`，无resume或输入checkpoint。warmup=5000 raw，训练后自动100回合validation评估。两路共享一张RTX5060Ti，保留正常训练/评估过程诊断及30/40/50秒warmup快照。

这是运行启动记录，不是完成结果。后续分析先读该run根的launcher manifest/status、各方法目录`training_complete.json`、`evaluation_results.json`和diagnostics，确认完整预算、评估口径与实际完成状态；不要将此前的短时环境preflight当作这两路训练结果。

## 15. 2026-09-30：当前 DARRL 同名场景 r2 与终局双标修复

用户要求不新增场景名，原位将DARRL low/medium/high每路每0.1秒p改为0.015/0.03/0.05。当前configuration_revision=darrl_r2_20260930；前序r1为0.03/0.05/0.07。旧运行`runs/smlp_darrl_0930`已完成，必须按其保存的traffic_config解释；新增SCENARIO_REVISION_NOTE.json明确标记旧配置。r1的三档资产、关键源码、旧运行清单共32文件已归档并通过源/副本SHA校验，入口为analysis/scenario_revisions/darrl_r1_20260930/archive_manifest.json。

旧final eval低/中success+collision分别为30%+73%、13%+92%，源于3/5个回合同步双标，不是分母问题。训练中也有29/10个双标回合，错误领取reward_success=+10而reward_collision=0；旧模型受训练奖励错误影响。原始结果保留；离线碰撞优先计数为27%/73%、8%/92%，不能称为修复后模型表现。

当前在SumoSceneEnv.step完成场景特定事件检测之后统一解析：collision > off_route > success > timeout，奖励、terminated/truncated、replay与指标使用同一互斥结果。原生SUMO到达/碰撞及原始时限条件随正常训练/评估记录，terminal_outcome_protocol=exclusive_terminal_v2；仍使用原SUMO ego事件或几何重叠碰撞检测，不加DARRL legacy启发式，不改n-step算法和超参数。

配置、证据、测试和新训练入口见[当前r2配置与修复](analysis/darrl_r2_configuration_and_outcome_fix_20260930.md)与[旧轮审计JSON](analysis/darrl_r1_outcome_audit_20260930.json)。用户要求修复后再启动r2低/中SAC+MLP各fresh100k、2 CUDA worker、seed0、正常100回合validation评估与过程诊断；新run根与r1隔离，后续启动状态以本次运行记录为准。

R2正式新run已于约22:24（Asia/Shanghai）发起：`runs/smlp_dr2_0930`，低/中场景p=.015/.03，各fresh SAC+MLP seed0/100k raw、2 CUDA worker及100回合最终validation评估。启动前真实同tick碰撞+到达复现通过：collision-only、reward=-10.01、真实terminated且非truncated；详见r2修复记录。训练完成与表现仍须以该新run的training_complete/evaluation_results为准。

## 16. 2026-09-30—2026-10-01：旧场景 intersection_sorted / depart_scale=4.0：Topo / 3slot 归因审计

已完成的 seed0 fresh-100k 链为 ST 50/50/0 → ST-RT 63/37/0 → Topo 39/39/22 → Topo+3slot 41/56/3（成功/碰撞/超时，final 100 回合）。22个Topo超时终点均在 `-E1_0` 的70m末端、route index 0、实际速度0，而请求目标速度均值约5.02m/s。`intersection_sorted/ego.rou.xml`指定 `-E1→-E0` 且从lane2出发；`map.net.xml`唯一的 `-E1→-E0` connection 是fromLane2→toLane1。SMARTS动作契约下lane命令-1降低lane index；对应这些回合反复向低索引/right移动，最终到没有路线出口的lane0。这是超时的直接路网近因；为何策略选择右移、是否由Topo导致，仍未证实。Topo query compatibility是几何/朝向筛选和top-8，不是route-successor合法性mask；拓扑attention与选道因果不可仅凭本结果定论。

Topo与3slot共享同一 incremental forward 和 critic-only encoder更新；Graph-SLT/SBS/表示损失均关闭。差异在最终头：joint MLP 384→128→128（65,792活跃参数）换为三个128→32/64/32线性头（16,512活跃参数）；槽上下文并非语义互斥，仍由下游actor/critic非线性混合。优先建议先针对Topo显式加入路线可达lane对应（须新建 route-edge↔topology-node 映射，不能只改cosine阈值），再评估参数量接近的非线性三槽头。方案均未实施或训练；本次只读，没有生产代码变更。该证据只描述旧场景 `intersection_sorted`，不属于当前 `intersection_random_darrl_*` 低/中/高场景；不能直接用于新场景排名。详见[实现审计](analysis/topo_3slot_implementation_audit_20260930.md)及[结果归因报告](analysis/topo_3slot_same_scene_attribution_20260930.md)。

## 17. DARRL medium p=.03 七方法扩展（retry02运行中；尚无结果）

本轮拟在独立场景 `intersection_random_darrl_medium_v1`（`configuration_revision=darrl_r2_20260930`）比较七种方法：`mst_slt`、`sac_mlp_d1_st`、`sac_mlp_d1_st_rt`、`sac_mlp_d1_st_rt_topo`、`sac_mlp_d1_st_rt_topo_routeaware_v1`、`sac_mlp_d1_st_rt_3slot`、`sac_mlp_d1_st_rt_topo_3slot`。场景为三条背景进口流各自每0.1秒以p=.03请求发车；训练与validation使用不同seed域。计划训练seed=0、fresh连续100000 raw SUMO steps、用于七方法官方比较的final validation评估100回合；MST入口是否另有内部评估及其计数待核，不将其与官方比较回合数混写。最多两个CUDA worker；run根为 `runs/dm7_1001`。本计划已实际尝试启动，但七路均因环境step诊断字段重复而退出，当前无训练或性能结果；本节末记录失败快照。

`routeaware_v1`仅在Topo ego-goal候选上应用路线可达mask：先与已知合法候选求交；交集为空但存在已知合法候选时扩展至全部已知合法候选；若没有已知合法候选，则绕过mask并精确保留base route residual。它不修改social分支或actor路径，也不是安全保证。`sac_mlp_d1_st_rt_3slot`关闭Topo、保留ST-RT并复用既有线性三槽readout，是用于因素拆分的变体，不是参数量匹配的新头；`topo_3slot`保留为两者组合。

正常运行预期保存train/eval分开的行为raw/decision流、episode摘要、优化及表示事件、manifest与summary；表示记录含路线候选/expand/bypass相关量，SAC梯度记录复用既有TD梯度路径。父侧报告的启动前验证为6项模型诊断测试、11项route-helper测试和CPU真实环境两种方法save/load/gradient检查通过；这些是实现/接口证据，不是CUDA正式实验结果。启动后以launcher状态、源码和场景资产内容副本及SHA、每方法arguments/completion/final-evaluation核实执行。旧 `intersection_sorted` 和DARRL r1/r2结果均保持独立，不覆盖或混作本轮结果。详见 [方法实现地图](analysis/full_mst_slt_implementation.md) 和 [实验历史](analysis/experiment_history.md)。


首轮实际启动快照（2026-10-01约01:35 Asia/Shanghai）：launcher PID=98120、suite supervisor PID=42508，run根为工作区 `runs/dm7_1001`。launcher_status最终为7/7方法failed、每个exit_code=1；子进程PID依次为mst_slt=28568、st=37320、st_rt=84656、topo=50304、routeaware=44148、st_rt_3slot=19372、topo_3slot=94324。D1方法arguments确认CUDA、seed0、100000 raw budget、smoke=false、behavior_diagnostics=true；suite manifest确认DARRL medium p=.03、r2、validation官方比较100回合及exclusive_terminal_v2。启动中各方法在首轮环境step报错，未形成训练完成或final eval结果。失败日志显示 `sumo_env.py:659` 调用 `info.update(behavior_telemetry_protocol=..., **route_lane_facts)`，但 `_behavior_route_lane_facts` 返回值自身含同名键，引发重复关键字TypeError；错误定位属于实现集成失败，不是实验结果。run目录、日志和source archive保留；父侧确认进程均已自然退出，后续需修复并用独立fresh retry root重新启动，retry截至此快照尚未启动。


Retry01失败更新（2026-10-01约01:45 Asia/Shanghai）：新根 `runs/dm7_1001_retry01`，launcher PID=67536、suite supervisor PID=68888。六个D1方法均在启动早期失败，错误位于 `high_density_env_v1.py:355` overlay manifest 的 `os.replace`，报WinError 3/FileNotFoundError；MST在诊断下推进至1530 raw steps后由root终止整个supervisor进程树，避免本轮混入续训。1530 raw不是正式checkpoint起点，后续fresh训练不得续用。root按失败日志计算ST/Topo/routeaware的目标路径长度分别约261/269/283字符，而source/temp路径约202–224字符；仅测source/temp不足以排除Windows路径问题，需同时审计source和destination，当前根因待验证。失败run、logs及source archive保留；下一fresh root待定，retry01无完成训练/评估结果。


路径长度根因后续验证：root在独立目录对本run真实manifest basename、已存在parent目录和成功写入的source temp执行最小Windows os.replace探针。DEST长度240/259时成功，260/261/269/283时均复现WinError3/FileNotFoundError；结果保存在工作区 `tmp/manifest_path_probe_20261001_015015/results.json`。此前实际失败manifest DEST分别约261（ST）、269（Topo）、283（routeaware），与探针失败边界一致；此前只测到202–224字符的source/temp长度不足以排除它。当前证据强支持超长DEST路径是retry01 overlay失败原因；代码/路径修复验证及下一fresh root仍待完成。



Retry02启动核验（2026-10-01 02:13:33 Asia/Shanghai）：正式七方法队列位于 runs/dm7_1001_retry02，launcher PID 66336、suite supervisor PID 68208；当前先启动的 MST=39120 与 D1-ST=14924 持续推进，其他5种方法仍排队。MST已到6472 raw steps/2168 decisions/27回合；ST已到6274 raw steps/1274 updates/23回合。两者diagnostic_error_count=0，活动日志未见traceback，CUDA参数/日志已核实；正式评估尚未完成，因此没有本轮性能结果。首个ST训练回合30/40/50秒暖场分别为due/inserted/pending=19/19/0、28/28/0、41/41/0；在网背景车15/13/21，均无停驶车且守恒残差为0。这些只是采集链运行证据，不是评估指标。启动与评估计数账本见 runs/dm7_1001_retry02/evaluation_accounting.json；稳定运行快照和失败重试史见 analysis/experiment_history.md 第21节。


## 2026-10-01 正式队列确认：dm7_1001_retry02

当前有效运行是 `runs/dm7_1001_retry02`，已从零启动七方法；前两次目录 `dm7_1001`、`dm7_1001_retry01` 保留为失败/中断尝试，不作为性能结果，也未从其中恢复训练。新增诊断字段重复合并已修复；Windows 长目标路径问题通过仅缩短 D1 车流缓存到 `run_root/_hd/<method>/ns_tr|ns_eval` 修复，不改变模型、场景参数或随机种子。真实环境五次 reset+step 验证通过，同 seed 在不同存储位置生成的路线源/overlay 哈希和车辆数一致；验证来源为 `runs/dm7_pathchk/overlay_path_smoke_results.json`。

协议：`intersection_random_darrl_medium_v1`，DARRL r2，三路各 p=0.03/0.1秒；seed 0；每方法 fresh 100000 raw SUMO steps；checkpoint 每10000 raw steps；最多两 CUDA worker；官方比较采用 validation 外层100回合。方法为 `mst_slt`、`sac_mlp_d1_st`、`sac_mlp_d1_st_rt`、`sac_mlp_d1_st_rt_topo`、`sac_mlp_d1_st_rt_topo_routeaware_v1`、`sac_mlp_d1_st_rt_3slot`、`sac_mlp_d1_st_rt_topo_3slot`。MST原入口另有内部100回合评估；只有外层 `evaluation_results.json` 用于七方法比较，新增日志没有增加评估回合。详见正式根 `evaluation_accounting.json`。

已核验运行快照（2026-10-01 02:13:33 +08:00）：launcher PID 66336，supervisor PID 68208；MST PID39120 已6472 raw/2168 decisions；ST PID14924 progress 已6274 raw/1274 updates（异步diagnostics summary为6313 raw）。两路诊断错误计数均0，活动日志无Traceback；其余五路待worker空闲依次执行。这是运行健康检查，不是最终性能结果，100k 尚未在该快照完成。

日志已实际写出 raw、decision、episode、optimization、representation 数据。训练输入快照每10000 raw一次、最多10份；外层评估首动作与终止动作输入最多200份/worker。字段包括动作/目标与实际速度、换道请求与实际转换、路线接续合法性及错道停滞、TTC/碰撞位置、Topo/三槽表示与梯度、优化指标。D1 ST首回合预热30/40/50秒抽样：due与inserted分别19/28/41，pending均0，在网背景车15/13/21，halting均0；due-inserted-pending差额均0。这只是日志与预热流程核验，不是场景总体统计。

已通过17项behavior测试、17项route/encoder定向测试（另4个subtests）、1项真实诊断环境集成测试，以及新增方法真实环境预测/梯度/保存加载验证。源码实际内容与SHA归档在正式根 `source_archive/manifest.json`，完整启动配置在 `suite_manifest.json`。后续分析应读取最终checkpoint和外层评估原始结果，不能用本段早期训练快照推断方法优劣。


## 2026-10-01：按用户要求先比较两条基线

用户将结果分析顺序明确为：先MST+SLT与SAC+MLP，再SAC→ST→ST-RT→Topo→路线约束/三槽逐项讨论。当前DARRL r2 medium p=.03、seed0、fresh100k、final validation100：SAC+MLP=31成功/62碰撞/7超时，MST+SLT=16/53/31。100/100实际SUMO种子和traffic schedule SHA匹配；成功差-15pp属于本批固定checkpoint/单训练seed结论。MST既有错道停滞，也有路口较高速碰撞，不能概括为统一保守或直接认定SLT无效。旧sorted两基线22/38/40与53/47/0只作历史记录：交通车道、驾驶交互、循环补车、部分碰撞配置均变，旧训练池口径有未解决差异。详情见[基线分析](analysis/darrl_medium_mst_vs_sac_analysis_20261001.md)与对应JSON。

本次初始核验时dm7_1001_retry02已有六路最终评估；Topo+3slot仍训练约63k，没有把其阶段指标记为final，也没有新增训练/评估。后续先分析现有模块结果，不因基线排序变化立即改场景或叠模块。

## 2026-10-01：交叉口历史实验汇总入口

本次全量目录从修正后的三流 `intersection_sorted` 纳入，不纳入此前背景流只实际生效一股的 `intersection` 前身。核验到38条非smoke历史最终评估：sorted 24条、早期随机场景及显式概率变体4条、DARRL r1两条、DARRL r2八条；其中36条按互斥终局汇总，另两条DARRL r1保留作双标/训练奖励污染历史，不作为修复后有效基线。跨场景配置和终止语义有变化，结果仅作对应场景描述，不把跨版本变化解释为单变量因果效应。

完整主表与场景定义见[交叉口实验汇总](analysis/intersection_experiment_summary_20261001.md)及[场景配置目录](analysis/intersection_scenario_catalog_20261001.json)。`depart4p0`是固定模板中车辆depart时间戳乘4，不是流量乘4；原生SUMO flow以veh/h配置，标准p05/p03/p02和DARRL r1/r2则是每路线每0.1秒步的请求概率，不能混读。

截至2026-10-01 20:22（北京时间），DARRL r2中档队列的Topo+3slot为85413 raw steps/80413 updates，尚无`training_complete.json`、`evaluation_results.json`或final模型；旧sorted场景的Topo+3slot已有最终评估。本次只核对并索引现存结果，未启动新训练或评估。

## 2026-10-01：38条交叉口历史结果的奖励口径审计

奖励证据与逐组限制见[奖励审计](analysis/intersection_reward_audit_20261001.md)：38条由本地20条和runs目录18条组成；有充分依据写作采用同一v2系数设计，但不能概括成全部历史实现/终局完全一致。
本地11条fresh SAC/D1参数文件明确记录六个v2系数；两条fresh MST未记reward字段，其中100k MST的Monitor六分量支持同一设计；13条local fresh记录gamma=.99/action_repeat=3，7条续训缺reward、gamma及repeat字段。
runs下18条参数/奖励分量支持v2数值；DARRL r1 low/medium例外有训练双标29/10、评估双标3/5，训练wrapper选择成功+10而碰撞−10未执行。
DARRL r2统一collision > off-route > success > timeout，使用互斥终局；r1结果保留为污染历史，不与修复后互斥结果合并。
训练每decision reward与评估`episode_return`须分开：评估使用raw `success−collision`；训练还有每decision −.01及+ .02乘累计行驶距离差。
`D`是SUMO `getDistance`的插入后累计行驶米数，不是剩余距离变化，进度差没有正负裁剪；SLT/Graph-SLT/SBS是辅助损失而非环境reward。未改代码或启动实验。
