# Sorted/depart4p0 奖励目标审计（2026-10-03）

本审计只读既有训练、评估和诊断文件；没有载入模型、启动 SUMO、训练或评估。可复算程序为 [audit_reward_objective_20261003.py](audit_reward_objective_20261003.py)，运行命令：`python -B analysis/audit_reward_objective_20261003.py`。程序使用 Python 标准库，只向 stdout 输出 JSON。

## 当前证据

奖励实现按每个 policy decision 计算：

\[
r_d=10\mathbf{1}_S-10\mathbf{1}_C-10\mathbf{1}_O-5\mathbf{1}_T-0.01+0.02(D_d-D_{d-1}).
\]

`S/C/O/T` 由互斥 `if/elif` 选择，优先级为 success、collision、off-route、timeout。终局项与 step cost、progress 同时加在终止 decision 上。`D` 是 ego 自插入以来的 route 累积行驶距离（米），不是距目标的剩余距离；实现没有把 progress 项裁剪到固定范围。[reward_shaping_v2.py](../reward_shaping_v2.py#L55) 的系数默认值在55–68行，终局优先级与过程项在86–106行，距离定义从121行开始。

评估按 `evaluation_results.json` 的100条 `episode_records` 分组；仅接受每条记录均标记 `environment_step_reward_v2` 且含六个数值分量的结果。按 success/collision/off-route/timeout 中恰好一个为真的记录互斥分层；return 交叉数是每个失败 episode 与每个成功 episode 的笛卡尔积，统计严格满足 `failure_return > success_return` 的对数。三组可比结果的六分量都覆盖100回合，最大重构误差为 `3.55e-15`。

| 方法和run相对路径 | 验证结果（S/C/T/O） | shaped成功回报：均值 [最小, 最大] | shaped失败回报：均值 [最小, 最大] |
|---|---:|---:|---:|
| 旧 ST-RT：`runs/sortlr_1003_retry01/sac_mlp_d1_st_rt__intersection_sorted_depart4p0` | 63/37/0/0 | 12.3502 [11.6460, 12.6607] | 碰撞 −9.1643 [−9.6181, −8.7762] |
| 修复 ST：`runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0` | 45/50/5/0 | 12.4140 [11.6152, 12.6620] | 碰撞 −9.0566 [−10.2201, −8.6095]；超时 −5.2887 [−5.6000, −4.4402] |
| 修复 ST-RT：`runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0` | 52/48/0/0 | 12.3496 [11.6524, 12.6852] | 碰撞 −9.1523 [−9.7481, −8.7825] |

三个分层中失败回报高于成功回报的 episode pair 分别为 `0/2331`、`0/2475`、`0/2496`。修复 ST 的按结果分层分量均值可说明抵消尺度：成功为终局 `+10`、progress `+3.3136`、step cost `−0.8996`，合计约 `12.4140`；碰撞为 `−10`、`+1.6626`、`−0.7192`，合计约 `−9.0566`；超时为 `−5`、`+1.7113`、`−2.0000`，合计约 `−5.2887`。因此已有评估中 process reward 会改变回报幅度，也会让失败 episode 获得正 progress，但不足以把任何失败排到成功之上。此结果只排除了这三批已评估轨迹中的“失败回报反而普遍高于成功”这一简单错配，不证明 reward 是正确的，也不证明 SAC 学到了正确排序。

脚本还从三组的 `diagnostics/eval/decisions.jsonl.gz` 独立计算了 `sum_d 0.99^(d−1) * reward_policy_d`。它把 telemetry 的1-based episode 映射到评估记录的0-based episode，验证每回合决策序列与 `decision_steps` 相符，并将未折扣和与 `episode_return` 逐回合对账；三组均覆盖100回合，最大对账差为0。结果如下：

| 方法 | 折扣成功均值 [范围] | 折扣碰撞均值 [范围] | 折扣超时均值 [范围] |
|---|---:|---:|---:|
| 旧 ST-RT | 5.2802 [2.6499, 6.9429] | −3.9405 [−5.3339, −2.2110] | — |
| 修复 ST | 5.7791 [2.6164, 7.0994] | −4.3789 [−5.8545, −1.0185] | −0.3356 [−0.3823, −0.2245] |
| 修复 ST-RT | 5.3550 [2.6480, 7.2432] | −3.9317 [−5.4569, −2.2610] | — |

这个离线数值只是**按 decision 每步直接折扣的已执行环境奖励和**。它没有 SAC 的 bootstrap Q、熵项或 replay 的 n-step 采样语义，不能称为 SAC return/target，也不能据它断言 soft policy 下的排序正确。旧 ST 的正式路径为 `Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0`：其100回合评估为 S/C/T/O=50/50/0/0，保存的 `mean_return=0` 是 raw 的 `+1/-1` 口径，没有 `environment_step_reward_v2` 六分量或逐 decision shaped reward 日志，故不纳入上表。旧 ST-RT 与两修复 run 有可比 v2 日志；不要把旧 ST raw均值与 shaped 均值横比。

三个有优化遥测的 v2 run 都设置 `ent_coef="auto_0.2"`、`target_entropy="auto"`。从训练 `diagnostics/train/optimization.jsonl` 中只取带 `metrics["train/ent_coef"]` 的稀疏行，各有96个样本：旧 ST-RT 从 raw 5,007 / update 7 的0.199888采样到 raw 100,000 / update 95,001 的0.016695，采样 min/max 为0.007497/0.199888，样本均值0.032948；修复 ST 为0.199889→0.008026，min/max 0.006372/0.199889，均值0.028056；修复 ST-RT 为0.199889→0.014219，min/max 0.009492/0.199889，均值0.030690。优化日志声明记录的是 callback 时“最新可用 logger 值”，未必是该 update 同步值；这些数是稀疏采样范围，不是逐更新 alpha 轨迹。训练奖励分量的 episode 均值在各 run `reward_branches.json`，口径是已完成训练 episode 汇总，不应当作评估 outcome 分层结果。

## 目标与实现边界

当前主训练配置 `DISCOUNT=0.99`、`ACTION_REPEAT=3`，D1 用 `n_steps=4`；SAC 使用自动熵系数。[train_intersection_yield_v2.py](../train_intersection_yield_v2.py#L71) 的 gamma/action-repeat 在71–72行，SAC 参数在848、855–856行；[train_intersection_yield_v2_d1.py](../train_intersection_yield_v2_d1.py#L1165) 与1172–1173行确认D1参数。一个 policy decision 通常推进3个0.1秒 raw tick。

已审过的 [bootstrap_protocol_audit_20261003.md](bootstrap_protocol_audit_20261003.md#L7) 和源码 `pytorch_sb3_sumo/algos/sb3_torch/replay_buffer.py:183–212,239–242`、`pytorch_sb3_sumo/algos/sb3_torch/sac.py:382,426–461` 表明：replay先累加至多4个 decision 的环境奖励 `R_k=Σ_{i=0}^{k−1}γ^i r_i`，然后对 endpoint soft value 只乘一次 `γ=.99`，不是 `γ^k`；SAC endpoint 再减 `α log π`。四步 reward aggregation 本身不含各中间步 entropy。timeout 的 `−5` 进入奖励序列，但 timeout 是可 bootstrap 的 truncation；当前观测没有剩余时间字段。因而上表离线逐步折扣和既不复现该 n-step target，也不复现完整 soft return。

progress 当前是 `0.02*(D_d-D_{d−1})`。它不是对 gamma=.99 有 policy-invariance 保证的 potential-based shaping 形式 `F(s,s')=γΦ(s')−Φ(s)`；不能仅因它叫 progress/shaping 就说最优策略不变。由此可见存在可研究的 reward/target语义问题，但日志没有隔离其对成功率的因果效应。

结果支持的判断是有限的：同一套 v2 reward 在旧 ST-RT、修复 ST、修复 ST-RT 的单 seed结果上对应63%、45%、52%成功率；终局成功回报在三组都高于所有失败回报。这说明“失败得到更高已实现回报”不是这三组的直接观察，也使“reward 是唯一主因”缺少支持。它不能证明 reward 无影响：策略架构、修复、随机种子、策略学习和 n-step target 仍影响优化。每方法只有一个训练 seed，旧 ST 又只有 raw 回报日志，所以目前不能归因 reward 是主要问题。

## 事件分布 critic 构想的反方审查

“用真实一步 transition 对 target policy bootstrap”在数学上能避免把整条行为策略轨迹的最终标签直接当成当前策略风险标签，但前提是模型真正学习 `P^π(event,time | s,a)`，每次非终止 transition 都按目标 policy 的后续动作继续 Bellman 递推，且数据覆盖 target policy 会访问的状态—动作区域。policy 在训练中不断变化，目标分布也会变；target actor/critic 的延迟更新只能缓和漂移，不能补出日志中没有的状态—动作覆盖。离线估计超出 behavior support 的风险因此不可识别，单步 Bellman 本身不会解决它。

当前任务有一个更尖锐的事件定义问题：timeout 在实现中是可 bootstrap 的截断，不是已证实的吸收终局；如果它是外部采集限制，应作为 censoring/survival 边界，而不是与 collision、success 并列的真实竞争终局。如果60秒本来就是任务目标，则应先把有限时域写进任务定义，并将剩余时间纳入状态，否则同一观测的 timeout 风险随 episode 已耗时不同，事件过程非 Markov。当前终局还含 off-route `−10`，事件类别需覆盖它；有限时域的事件分布也需要无事件/存活质量。first-event-time 单位要明确是 decision 还是 raw tick，因为动作通常重复3 tick，终止决策可以不足3 tick。

若把 `Q_event=E[γ^(T−1)R_terminal(Y)]` 与 dense progress/step-cost Q 及 entropy Q 相加，必须给每项独立且一致的 Bellman target；只监督它们的总和时，终局/过程分量的分配不可识别。若 actor 继续使用总和的期望并保持同一 SAC 目标，事件分布并未改变策略效用，本质更接近带语义辅助输出的多头或 distributional critic；潜在增益应具体落在预测校准、表征学习或有限风险目标，并与“普通 scalar/distributional critic + 同预算辅助 event head”比较。若是 twin Q，必须先把同一支 critic 的事件与 dense/entropy 项重构相加，再在两支总 Q 间取 min；逐分量各自取 min 一般不等价。把 distributional output 用于风险敏感 action selection 会改变原 SAC 策略目标，需要单独声明。

当前正式 run 不保存完整 replay buffer。两修复 run 的 checkpoint 是模型 zip；train `decisions.jsonl.gz` 保存 action/reward/终局和行为遥测，不含逐条 pre-action observation 与 next observation；`policy_observations.jsonl.gz` 是稀疏采样的 observation/probe，不是每个 transition 的数据集。对 runs 下 replay/buffer/transition 文件名的有限只读检查未找到可重建完整 replay 的文件。因此现有资料足以重算回报、统计 outcome 和部分行为日志，不能据此离线训练事件 critic，也不能做 target-policy 条件校准或有效 action-ranking 检验。

下一步的最低成本验证应先固定事件/删失定义、时间刻度和 Bellman 方程，并用手算有限 MDP 检查事件概率质量、首事件时间移位、timeout 边界以及终局期望与原 scalar terminal-Q 的恒等关系；再验证 target policy 改变时不能复用 behavior 标签。根代理的 [check_event_value_identity_20261003.py](check_event_value_identity_20261003.py) 已做有限表格动态规划 sanity check：soft-Q分解最大误差约 `2.44e−15`、event mass误差约 `4.44e−16`；同一初始 `(s,a)` 但 continuation policy 不同时事件概率 L1差约 `0.345`。我静态检查了脚本的递归、终止、熵起点和质量递推，未发现明显算术错误；它使用有限时域、完全可观测合成 MDP，类别仅 success/collision/timeout，且不是神经网络收敛、覆盖或驾驶有效性证据。由于现有 runs 没有完整 transition observations，此前不能在不采新 transition 的条件下用当前数据做真实策略条件的离线 pilot；无需把稀疏 observation probe 当 replay 替代品。

## 可复现数据源

- 旧 ST：`Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/sac_mlp_d1_st__intersection_sorted_depart4p0/evaluation_results.json`、`reward_branches.json`。
- 旧 ST-RT：`runs/sortlr_1003_retry01/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json`、`reward_branches.json`、`diagnostics/eval/decisions.jsonl.gz`、`diagnostics/train/optimization.jsonl`。
- 修复 ST：`runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0/` 下同名评估、奖励汇总和诊断文件。
- 修复 ST-RT：`runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0/` 下同名评估、奖励汇总和诊断文件。
- 奖励定义：[reward_shaping_v2.py](../reward_shaping_v2.py#L55)；训练参数：[train_intersection_yield_v2.py](../train_intersection_yield_v2.py#L71)、[train_intersection_yield_v2_d1.py](../train_intersection_yield_v2_d1.py#L1165)；n-step与timeout协议：[bootstrap_protocol_audit_20261003.md](bootstrap_protocol_audit_20261003.md#L7)。
