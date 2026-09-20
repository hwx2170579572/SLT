# v4.2-r1：Route-Event Gate（协议测量修订）

状态：`preregistered_after_pilot / confirmation_pending`  
科学方法：与 `v4_2_factorized_lane_entropy` 完全相同  
pilot 归因：`results_topo_v4_2_dev/attribution/f1_event_gate/`

## 修订边界

不修改算法、模型、reward、encoder、loss、entropy、warm-up、训练预算或评估动作。只替换已被 F1 反证的逐决策机制门禁。

旧门禁要求 6 个连续 route-intent 标签的大部分决策都重复发送换道，并把所有 6 个概率边际平均。F1 表明实际闭环只需在首个可执行时刻成功换道一次；后续 keep 与 12/12 成功相容。

## 新事件定义

对每个至少包含一个 valid non-keep route intent 的 evaluation episode：

1. `event_match`：episode 内至少一次 `lane_command == route_intent`；
2. `event_applied_match`：上述匹配动作至少一次被环境实际执行；
3. `first_window_margin`：该 episode 第一个 non-keep route-window 状态上的 `p(required)-p(keep)`。

CARLA 硬门禁：

- 原 success/collision/off-route/timeout 门禁不变；
- `event_applied_match_episode_rate >= 0.80`；
- `event_positive_margin_episode_rate >= 0.80`；
- 所有 episode 的 `first_window_margin > 0.10`。

不再使用 keep rate、按决策 applied rate、逐窗口 match rate或全窗口平均/最小 margin。

## 确认性开发顺序

F1 仅作为 pilot，不直接计为新合同通过。

1. C1：CARLA seed 0，20k，validation seeds 44000–44011；
2. C2：C1 通过后 Cross seed 0，20k，同一方法配对 block；
3. C3：C2 通过后 Cross seed 1，20k，validation seeds 45000–45011。

新合同最终通过同时要求：F1 pilot 的事件级重算通过、C1 新鲜 CARLA 结果/事件门禁通过、C2/C3 Cross 守卫通过。任一失败即停止。promotion 与 formal 的方法矩阵、阈值和未访问 seed block 均保持不变。
