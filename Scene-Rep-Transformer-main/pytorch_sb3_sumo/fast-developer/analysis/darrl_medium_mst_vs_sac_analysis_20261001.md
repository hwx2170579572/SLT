# DARRL 中档：MST+SLT 与 SAC+MLP 的基线比较

日期：2026-10-01。按用户要求，当前只分析强基线与纯强化学习基线，后续再逐模块比较。本文没有新增训练、评估或干预实验，证据来自已有运行文件与正常保存的诊断。

**核心判断：本次 DARRL 中档、seed 0、100k raw steps 下，MST+SLT 的成功率确实低于 SAC+MLP。已观察到的失败包括错误车道末端停滞，以及较高速进入路口冲突区后碰撞。不能将差异概括为“更保守”，也尚不能归因为 SLT 本身失效。旧、新场景还同时改变了交通车道、到达过程、驾驶交互参数和部分运行语义，历史反转不是严格的单变量场景实验。**

## 当前两条基线的公平性

| 项目 | 两条基线的实际设置 |
|---|---|
| 场景 | intersection_random_darrl_medium_v1，darrl_r2_20260930 |
| 车流 | 三路各 p=0.03/0.1 秒，episode-seeded Bernoulli |
| 训练 | seed 0，从零 100000 raw SUMO steps；95001 learner updates |
| 公共优化设置 | learning_starts=5000，batch=32，lr=1e-4，buffer=20000，gamma=0.99，action_repeat=3 |
| 最终评估 | 各100回合，validation，final_model.zip 与评估记录的 checkpoint SHA 一致 |
| 逻辑评估种子 | 10000–10099 |
| 实际 SUMO 种子 | 1000010000–1000010099 |
| 车流配对 | 100/100 回合实际种子及 traffic schedule SHA-256 相同 |
| 终局口径 | exclusive_terminal_v2，成功/碰撞/超时/越界互斥 |

MST 原入口还执行一轮内部100回合评估。本文只用其外层 `evaluation_results.json`，没有将两轮合并为200样本。MST 旧 requested 参数仍出现 `frozen_80_20` 文本；本次运行 manifest 和逐回合记录证实实际使用上述 Bernoulli r2 协议，不应以该旧标签推断实际车流不同。

这是不同模型/表示学习方法的基线比较，不是参数量相等的单变量消融。100个评估交通种子也不等于100个训练随机种子。

## 最终结果与配对证据

| 方法 | 成功 | 碰撞 | 超时 | 越界 | 平均评估终局分 |
|---|---:|---:|---:|---:|---:|
| SAC+MLP | 31/100 | 62/100 | 7/100 | 0 | -0.31 |
| MST+SLT | 16/100 | 53/100 | 31/100 | 0 | -0.37 |
| MST−SAC | -15 pp | -9 pp | +24 pp | 0 | -0.06 |

同一车流逐回合配对如下，行为是 SAC 的结果，列为 MST 的结果：

| SAC → MST | 成功 | 碰撞 | 超时 | 合计 |
|---|---:|---:|---:|---:|
| SAC 成功 | 3 | 21 | 7 | 31 |
| SAC 碰撞 | 12 | 28 | 22 | 62 |
| SAC 超时 | 1 | 4 | 2 | 7 |
| 合计 | 16 | 53 | 31 | 100 |

MST 新增13个成功、失去28个成功；其中21个原本成功的回合变成碰撞，7个变成超时。碰撞总数下降9个不表示对每种车流都更安全：它同时新增25个碰撞（21个原成功、4个原超时），减少34个碰撞（12个转成功、22个转超时）。

固定这两个训练完成的策略，在100组配对交通上，成功率差为 -15 pp，20000次配对 bootstrap 的95%区间为 [-27,-3] pp，重采样seed=20261001；成功二分类 McNemar 精确双侧 p=0.027533。该统计只描述这两个 checkpoint 对验证交通的差异，不覆盖重新训练的随机种子稳定性。

## 可确认的行为失效

| 正常评估中保存的量 | SAC+MLP | MST+SLT |
|---|---:|---:|
| raw-tick 加权实际速度 | 2.689 m/s | 2.564 m/s |
| 有效速度样本中 v<0.1 m/s | 39/41984（0.093%） | 22021/37785（58.28%） |
| 成功回合平均完成时间 | 43.27 s，n=31 | 34.28 s，n=16 |
| 全体回合平均 raw steps | 420.21 | 378.02 |

成功条件下的时长来自不同的成功回合子集，不能据此认定 MST 整体效率更高。更短的全体回合时长也包含提前碰撞。相近的总体平均速度掩盖了 MST 长时间停滞与部分较快运动并存的状态。

**错道停滞有明确证据。** 对既定路线的下一跳 `-E1→-E0`，同一张地图中可直接接续的进口车道为lane2；lane0/1需要提前换回合适车道，不能把当前车道不可直连理解为经过任何动作也永远不可达。MST 的31个超时中，14个终点在 `-E1_0`、7个在 `-E1_1`，均停在进口路段约70m末端、route_index=0，日志明确标记不能接续目标路线。另有8个终点在路口内部且最后路线接续标记为合法，2个已经到 `-E0_2`；不能将31个超时全部归为同一原因。28/31终端实际速度≤0.5m/s；其中14个终端目标速度仍>3m/s。目标速度与实际速度脱节，支持相当一部分回合受路线/交通约束停滞，而非单纯主动给出零速。

尤其在7个“SAC成功→MST超时”回合中，6个最终停在上述错误进口车道末端，1个已到出口道路但未完成。该链条可以支持“错误车道及未能及时恢复造成一部分成功损失”，不能证明神经网络为何学出错误换道选择。

**另有冲突区通行失败，不能忽略。** 21个“SAC成功→MST碰撞”回合中，20个碰前ego快照在 `:J1_21`，1个在 `-E0` 起段；碰前实际速度约4.62–9.74m/s，目标速度约5.14–9.74m/s。终末动作中11个换道请求因 internal_lane 未发出、9个hold、1个request_sent。现有证据支持这组回合表现为较高速通过冲突区时未能安全完成，不支持把它们统一解释为执行换道导致碰撞，也没有证明简单限速必定修复。

该组碰撞的episode最小TTC接近0，但该统计包含事故时刻，本身不能充当独立的风险预测因果证据。相关几何碰撞记录没有足够的 SUMO 对方对象ID，不能确定全部具体冲突车辆；碰前/终端ego快照也不是精确重建的物理碰撞点。

## 为什么不能把历史反转称为“只改了发车随机性”

历史记录如下，均为对应运行的100k结果、100回合终评：

| 历史场景 | SAC+MLP S/C/T | MST+SLT S/C/T |
|---|---|---|
| intersection_sorted，depart_scale=4 | 22/38/40 | 53/47/0 |
| DARRL r2 medium，p=0.03 | 31/62/7 | 16/53/31 |

旧场景MST比纯基线高31pp，新场景低15pp。但 SAC 自身由22%升到31%，所以用“新场景让所有方法都更难”解释反转不充分。

经文件核对：

| 条件 | 旧 sorted | 当前 DARRL medium r2 |
|---|---|---|
| 地图 | 与新场景map.net.xml SHA相同 | 相同 |
| 自车路线/发车 | -E1→-E0；50s；lane2；speed0；arrival lane1 | 相同 |
| 背景出发/到达车道 | 所有30个模板、17700辆均 lane0→lane0 | 三路统一 lane1→lane1 |
| 背景 jmIgnoreFoeProb | 120个车型，0.02–0.89 | 全部显式覆盖为0 |
| 自车 minGap | 2m | 1m |
| 自车 jmIgnoreFoeProb | 未显式覆盖，不能假定其旧有效值非0 | 显式0 |
| 到达过程 | 30个固定XML发车模板，depart×4，并启用循环重插 | 独立Bernoulli计划，flow 0–130s，无循环补车 |
| 碰撞运行配置 | sumocfg写warn；旧运行环境源码未完整归档，命令行覆盖未知 | 明确remove，SUMO事件/ego几何，exclusive_terminal_v2 |

因此，改变的包括车道占用、让行/冲突规则、车辆到达与暴露过程，不只是随机种子或地图名称。地图相同并不意味着策略面临同样的控制问题。本轮MST是在新场景从零训练；不是把旧权重直接放进新场景的迁移评测。

旧对照也有证据边界：两条旧基线的100个评估seed及traffic_variant匹配，但旧SAC manifest明确训练/评估同一完整30模板池、无holdout，旧MST requested metadata写frozen_80_20；旧训练交通池实际是否一致尚未证明。旧final episode标签均互斥，却不能倒推出旧原始碰撞解析与r2完全同口径。故旧53%与22%应保留为历史观察，不能把全部优势归因于模型架构，更不能把跨场景差值当成严格单因素因果效应。

## 奖励与优化：能说什么、不能说什么

评估的 episode_return 累加原始 `info.undiscounted_reward`，所以这里等于 `P(success)-P(collision)`。训练层实际使用事件塑形奖励：

`r_train = 10*S - 10*C - 10*off_route - 5*timeout - 0.01 + 0.02*Δdistance_m`

过程项按decision计算，两条当前基线设置一致。因此不能说“MST超时多是因为训练对超时没有惩罚”，也不能把评估分差直接等同于训练目标差。

SUMO时间上限被记为truncated，replay target对timeout继续bootstrap。这是两条基线共享的实现，不能单独解释本轮15pp模型差异。是否应将600 raw时间上限视为任务本身的终止，取决于任务定义；不能仅因结果不好就改mask。Gymnasium官方说明区分任务内有限时域终止与外部训练截断：[Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。本轮未修改该语义。

在固定100k预算下，表示架构、辅助表示优化和策略学习可能对新交互分布有不同适应程度；这是合理的待验证解释。目前证据不足以指认SLT梯度冲突、错误的超参数、收敛不足或随机发车破坏未来预测中的任意一项为确定根因。也不能把单训练seed的结果推成“MST+SLT普遍不如MLP”。

## 当前判断与下一步

1. 保留这两条基线的原始结果与角色；本批当前成功率标准为纯基线31%、强基线16%，以后方法不能仅超过16%便声称优于两条基线。
2. 当前中档的纯基线31%落在用户此前设想的30–60%区间内；现有结果没有要求为帮助某方法获胜而重新调整场景。
3. 按用户指定的顺序，下一步使用已经完成的SAC+MLP→ST比较，检查成功损失、错误车道停滞与冲突区速度是否已出现，再继续ST→ST-RT。先区分哪些失效在加入哪些结构后出现，不凭本轮基线差异直接叠模块、调SLT权重或启动额外训练。
4. 单训练seed结论仍需限域。最终投稿若要主张稳定优于强基线，再补训练随机性与预算敏感性证据；当前不强行扩成多种子。

## 可复算来源

- 当前配对汇总与统计：[darrl_medium_mst_vs_sac_20261001.json](darrl_medium_mst_vs_sac_20261001.json)，复算脚本 `darrl_medium_sac_vs_mst_20261001.py`。
- 正常日志中的行为诊断汇总：[darrl_medium_mst_vs_sac_diagnostic_summary_20261001.json](darrl_medium_mst_vs_sac_diagnostic_summary_20261001.json)，保留来源、样本分母与归因限制。
- 逐例行为诊断：[darrl_medium_mst_vs_sac_diagnostics_20261001.json](darrl_medium_mst_vs_sac_diagnostics_20261001.json)，含21个错道超时、7个成功转超时和21个成功转碰撞的具体记录。
- 当前纯基线：`runs/smlp_dr2_0930/sac_mlp__intersection_random_darrl_medium_v1_depart1p0/evaluation_results.json` 及该目录diagnostics。
- 当前MST：`runs/dm7_1001_retry02/mst_slt__intersection_random_darrl_medium_v1_depart1p0/evaluation_results.json` 及该目录diagnostics。
- 旧纯基线：`runs/d0929_100k_diag/sac_mlp__intersection_sorted_depart4p0/evaluation_results.json`。
- 旧MST：`pytorch_sb3_sumo/fast-developer/mst_slt__intersection_sorted_depart4p0/evaluation_results.json`。
- 运行时源码：`runs/dm7_1001_retry02/source_archive/project/` 下 `envs/sumo/sumo_env.py`、`fast-developer/reward_shaping_v2.py`、`algos/sb3_torch/evaluation.py`、`replay_buffer.py`、`sac.py`。

本文未把仍在训练中的Topo+3slot记为最终结果，也未在本轮展开其他已完成模块的结论。
