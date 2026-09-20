# v4.10：共享风险表征与数据支持的多速度提案模型

状态：方法与实验在任何 v4.10 训练前预注册  
模式：CCFA standard，generic AI/ML CCF-A evidence package  
Formal test：继续锁定

## 1. 失败归因

v4.9.2 的完整 promotion 在 12/12 作业完成后失败。候选在 CARLA 上明显提升，
但 Cross 的 success/collision 相对 TemporalGraph 分别退化 `-0.20/+0.2167`；
Cross/seed20 的配对退化为 `-0.4333/+0.4667`。所有模型完整性检查均通过，故失败
不是安全规则缺失或执行协议错误。

v4.10 stage-0 对六个冻结候选 checkpoint 做了 72 个 validation 回合、7,238 个
决策的只读 probe。诊断动作从未发送给 SUMO。主要事实是：

- actor 的单速度提案在 96.63% 的访问状态上不是其冻结 target critics 的同车道
  评分驻点；
- 三个含碰撞的配对中，碰撞前 10 步有 42.50%--55.00% 的状态存在“更低 learned
  collision value 且更高联合评分”的同车道速度；
- 但这些碰撞窗口的诊断最优速度有 63.93%--90.00% 落在查询网格边界，成功窗口
  同样常见边界最优。因此直接扩大搜索、固定速度网格或 critic 最大化会利用外推误差，
  不能作为部署方法；
- v4.9 的 collision critic 使用独立场景编码器，actor 读取 reward critic 的共享编码器
  并 stop-gradient；此外 reward 编码器同时被 critic optimizer 和 representation optimizer
  持有。该结构能产生种子相关的风险表征错位与重复优化器状态。

## 2. 唯一整合假设

v4.10 不是运动学安全投影，而是一个端到端学习的
**Shared Risk-Supported Mixture (SRSM)** 模型：

1. reward critic、collision critic 与 actor 读取同一个 online scene encoder；collision
   Bellman loss 因而直接塑造 actor 使用的表征；
2. 每个可行车道由 actor 学习 3 个 squashed-Gaussian 速度成分及其混合概率，不使用固定
   速度网格；
3. 回放中实际执行的 lane/speed 通过混合似然训练 proposal support，使确定性候选来自模型
   学到的动作分布；
4. reward twin 与 collision twin 的分歧形成连续、学习得到的不确定性项；没有阈值、否决、
   shield 或动作覆写；
5. scene encoder 只有一个 optimizer owner。reward heads、collision heads、actor heads、
   representation heads 与 entropy coefficient 也各自只有一个互斥 owner。

## 3. 学习与确定性决策

对车道 `l`、速度成分 `k`，actor 学习
`pi_lane(l|s) pi_component(k|s,l) pi_speed(a|s,l,k)`。训练期 SAC 目标和 actor 目标对
`l,k` 的联合概率做精确求和。collision value 继续只使用环境实际观测到的 collision 事件和
16-step、实际 horizon 折扣的 Bellman target。

确定性候选仅为 3×3 个 learned component means。每个候选的连续模型分数冻结为：

`min(Q1,Q2) - max(C1,C2) - 0.25*(|Q1-Q2| + |C1-C2|) + 0.05*log pi_component`

其中 `C` 是 sigmoid 后的 learned collision value。lane feasibility mask 仍来自原 observation
合同；它不是新增安全规则。最终在所有可行 learned proposals 上取最大分。不存在后处理。

训练期对 replay action 的 latent-speed mixture negative log likelihood 乘 `0.05`；component
entropy 在自动 speed entropy 中的系数固定为 `0.25`。这些值在任何 v4.10 训练结果出现前
冻结。

## 4. 优化器所有权

- encoder optimizer：唯一持有 shared online scene encoder；接收 reward critic、collision
  critic 与 Graph-SLT/SoftBalancedSlots 表征损失；
- reward critic optimizer：只持有 reward Q/action-embedding heads；
- collision critic optimizer：只持有 collision/action-embedding heads；
- actor optimizer：只持有 latent、lane、component、speed heads；actor 输入 encoder 输出
  stop-gradient，以免策略损失改写 critic 表征；
- representation optimizer：只持有 Graph-SLT prediction/projector heads；
- entropy optimizer：只持有 log-alpha。

任意参数出现在两个 optimizer 中均为工程门禁失败。

## 5. 开发实验与消融

先通过单元、梯度、序列化、optimizer-ownership、mask、分数方程及真实 SUMO smoke。
随后完整尝试 D1--D4；单作业失败不得取消后续作业，全部尝试后统一汇总：

| cell | scenario | seed | raw steps | validation episodes | role |
|---|---|---:|---:|---:|---|
| D1 | cross | 10 | 20,000 | 20 | primary safety |
| D2 | cross | 11 | 20,000 | 20 | seed replication |
| D3 | roundabout_medium | 12 | 20,000 | 20 | regression guard |
| D4 | carla | 13 | 20,000 | 20 | retained-gain guard |

主模型通过后、promotion 前完成两个机制消融（同预算、同 partition，不参与主模型选择）：

- A1 shared-single：共享风险表征与互斥 optimizer，但每车道 1 个速度成分；
- A2 mixture-no-support：3 个 learned components，但移除 replay-support NLL、component prior
  与 twin-disagreement 项。

主开发门槛沿用 v4.9：每个 Cross cell success≥0.50、collision≤0.40；RAM
success≥0.50、collision≤0.35；CARLA success≥0.50、collision≤0.10；全部 off-route=0。
Cross 两种子聚合 success≥0.60、collision≤0.30，且至少一个 Cross cell
success≥0.60、collision≤0.25。任何失败都触发新版本归因，不得访问 formal test。

## 6. Promotion 与 formal 边界

开发主模型和规定消融全部完成且主门禁通过后，运行 12 个 fresh promotion 作业：
TemporalGraph/SRSM × Cross/RAM/CARLA × seeds 20/21；每个 50,000 raw steps、30 个
validation episodes。沿用 v4.9.2 的非劣与正效应门槛。promotion 的任一作业失败仍继续其余
作业，最后汇总。

只有完整 promotion gate 通过才解锁未触碰的 120-job formal test。formal 仍为 6 场景 ×
2 方法 × 10 seeds、100,000 raw steps、50 test episodes，并沿用配对统计、hierarchical
bootstrap 与 Holm 校正。

## 7. 明确禁止

- 固定速度网格进入部署；
- kinematic projection、TTC/headway 阈值、几何 unsafe 标签；
- lane-change veto、confidence gate、shield、规则 fallback；
- decoder 后动作修改；
- validation/formal 参与 checkpoint 选择；
- 不完整矩阵或失败进程被记为通过。

