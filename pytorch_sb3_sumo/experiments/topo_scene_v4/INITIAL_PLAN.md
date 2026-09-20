# Full+Decision-Aligned BalancedSlots（Full+DA-BS）初始方案

状态：`v4-initial / needs-evidence`  
创建日期：2026-08-17  
来源方案 SHA-256：`0d3124be9743105c04289700f3444351c956f9a4d860820228b3b4fccfbad9ff`

本文件冻结用户提供的优化方案为 v4 初始版。v1–v3 的代码、协议和结果保持原样；后续每次方法变化都必须新增版本文件，并在 `iteration_ledger.json` 中登记单一变化、证据、决定和回退条件。

## 1. 已有证据与问题定义

- v2 候选的路线兼容注意力、fallback、有效车道数和残差尺度均通过表示门禁，但 promotion 的宏观成功率相对 TemporalGraph 为 `-0.161111`，CARLA 两个种子均为 0 成功。
- CARLA 候选 keep 为 `0.985479`、实际接受的换道命令为 0、平均速度约 `2.51 m/s`、timeout 为 1。
- v3 将横向输出放大 5 倍后产生 177 个非 keep 命令，但它们全部发生在前 0–31 个决策且没有一次被环境接受；因此“幅值不足”被否证，当前最具体的瓶颈是状态条件化的动作时机与可执行性错位。
- CARLA 路网中 `north_upper_1` 仅连接 `wrong_out`，`north_upper_2` 才连接任务路线的 `goal_out`。CARLA 动作符号约定下，从 lane 1 切到 lane 2 对应离散命令 `-1`。

## 2. 中心假设

保留已验证的 v2 拓扑主干，停止继续堆叠拓扑模块。方法因果链改为：

`路线拓扑 → 不塌缩且包含路线意图的 route slot → 可行性感知的离散 lane intent → 正确窗口中的实际换道 → 闭环成功`。

每一段必须有独立观测量与反证条件；表示代理不得补偿任何关键场景的完全闭环失败。

## 3. 阶段 0：零训练判别

使用成功的 TemporalGraph CARLA validation 轨迹作为唯一行为数据，同一 observation 同时送入 TemporalGraph 和失败的 v2 候选，按 episode 划分 probe 训练/测试集。运行时标签只使用当前车道、当前任务路线下一条 edge 与 SUMO lane connection；不使用未来成功、test partition 或隐藏 oracle。

必须报告：

1. route slot 与完整 latent 对三分类路线意图 `{-1,0,+1}` 的 episode-split probe；
2. 对“当前是否存在非 keep 可执行动作”的 probe；
3. 路线换道窗口中的 lane-intent 匹配率与动作 mask 合法率；
4. route-slot 均值消融后的速度/换道变化；
5. 两模型在完全相同 observation 上的速度、lane command 和正确窗口行为差异。

诊断路由：

- route slot 已能预测意图而候选 actor 不执行：第一实现只修改混合动作头；
- 完整 latent 可预测但 route slot 不可预测：加入 route-slot 意图监督并修改动作头；
- 完整 latent 也不可预测：加入信息保持型表示目标与混合动作头；
- 标签类别或成功 episode 不足：停止，不把不可辨识结果解释为机制结论。

## 4. 候选方法组件

### 4.1 冻结 v2 拓扑

保留 MERGE、路线/方向约束 top-k、近零拓扑残差、路线/拓扑分池以及 TemporalGraph 回退路径，不再扫描这些参数。

### 4.2 信息保持型 BalancedSlots（仅在阶段 0 指向表示问题时启用）

取消逐样本无仿射 LayerNorm。候选表示目标为：

`L_repr = L_GraphSLT + λ_b L_scale + λ_v L_var + λ_d L_intent`。

- `L_scale` 仅作为弱 batch-statistic 尺度正则；
- `L_var` 对每个 slot 的跨 batch 标准差设置下限，阈值来自冻结 TemporalGraph 开发轨迹的对应分布；
- `L_intent` 从 route slot 预测运行时路线意图；内部道路、歧义或无法判断的样本不计入；
- 第一轮保持 actor 对共享 encoder 的 stop-gradient，不同时修改奖励。

### 4.3 真正的混合动作策略

策略分解为 `π(k,v|z,m)=π_lane(k|z,m)π_speed(v|z,k)`：

- `k∈{-1,0,+1}` 为 masked categorical lane intent；
- `v` 为按 lane intent 条件化的连续目标速度；
- critic 接收 lane one-hot 与连续速度；
- actor loss 和 target value 精确枚举三个 lane intent；
- 环境仍保留二维 Box 外部兼容接口，但 actor 只输出精确的 `-1/0/+1` 横向编码，不依赖 `±1/3` 阈值学习。

## 5. 开发、promotion 与正式测试

- 开发最多 5 个新训练任务：CARLA seed-0 20k 单项修改；通过后 Cross seed-0 20k；一个关键消融；CARLA seed-1；Cross seed-1。
- CARLA 硬门槛：success ≥ 0.30、collision ≤ 0.10、timeout ≤ 0.70、keep ≤ 0.90、accepted lane change ≥ 0.02、off-route = 0。
- 任一关键场景硬门槛失败立即停止本版本并归因，表示指标不得补偿。
- promotion：`2 methods × 3 scenarios × 2 seeds = 12 jobs`，50k raw steps，validation only。
- promotion 通过后才解锁正式测试：`2 methods × 6 scenarios × 10 paired seeds = 120 jobs`，100k raw steps，50 test episodes/cell。
- 最终目标：每个场景相对 TemporalGraph 的成功率下降不超过 0.05、碰撞率上升不超过 0.05，并且至少一个场景成功率提高 ≥ 0.10 或碰撞率降低 ≥ 0.10；同时报告 paired bootstrap CI、Holm 校正、效率和失败案例。

## 6. 禁止项

- 不访问 formal test，直到 promotion 的哈希匹配硬门禁通过；
- 不把 validation、历史中断 run 或不同协议结果混入正式估计；
- 不修改奖励、解除 actor detach、加入表示损失和换动作头于同一不可分辨实验，除非阶段 0 明确要求“表示+动作”且关键消融能拆分贡献；
- 不虚构结果；未运行或不兼容单元格保持 `TBD`。

