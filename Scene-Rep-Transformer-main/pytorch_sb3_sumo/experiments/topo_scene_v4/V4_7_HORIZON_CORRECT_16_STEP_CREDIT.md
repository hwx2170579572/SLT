# v4.7：Horizon-Correct 16-Step Terminal Credit

状态：`preregistered_before_implementation_and_fresh_development`  
父版本：`v4_6_joint_checkpoint_decoder_selector`  
CCFA 模式：`standard / generic CCF-A AI-ML venue family`（用户未指定具体会议）  
失败证据：`results_topo_v4_6_dev/attribution/f1_failure/`

## 1. 失败归因与研究问题

v4.6 F1 的实现、四组合选择器和选中解码器均通过完整性门禁，但
Cross seed 6 的独立 validation 只有 `3/12` 成功、`6/12` 碰撞。
同一验证交通块上，冻结的 v4.5 Cross seed 4 权重达到 `10/12` 成功、
`0/12` 碰撞；交换验证块后，seed 4 仍为 `11/12`，seed 6 仅恢复至
`7/12`。因此主要因果因素是训练所得权重，而不是交通块、检查点选择或
固定解码器。

四个可比的 Cross exact-final target-critic 运行中，训练末 critic loss
与验证成功率呈完全反向排序（Spearman `-1.0`，`n=4`，仅作诊断线索）。
seed 6 的训练末 critic loss 是 seed 4 的约 `7.59x`；两者训练成功轨迹
数量相近，不能归因为“没有采到成功行为”。当前 4-step replay 下，依据
真实训练 episode 长度估计，只有约 `3.22%` 的可采样起点在 return 窗口
内直接包含终局奖励；batch 32 完全没有直接终局信用样本的概率约
`35.1%`。

研究问题是：在保持 reward、网络、动作头和部署选择器不变的前提下，
延长且正确折扣的多步回报能否降低 sparse terminal credit 引起的 critic
训练方差，并恢复跨训练 seed 的闭环稳定性？

## 2. 单一科学变化

仅替换候选方法的 return estimator：

- `n_step: 4 -> 16`（每个决策保持 3 个 raw SUMO steps，因此最多覆盖
  48 个 raw steps）；
- reward accumulation 仍为
  `sum_{i=0}^{h-1} gamma^i r_{t+i}`；
- bootstrap 折扣从父实现的单个 `gamma` 改为 `gamma^h`，其中 `h` 是
  16-step 窗口或首个 episode boundary 截断后的实际时域；
- terminal transition 的 done mask 仍阻止 bootstrap；source timeout
  语义、one-step next observation（供 Graph-SLT 使用）、episode-end
  duplication 和全局 replay 容量保持不变。

这被视为一个不可拆分的“horizon-correct long-return estimator”组件；它
同时定义回报窗口和与该窗口一致的 bootstrap。父控制保留原 4-step、
单 `gamma` bootstrap 语义。

## 3. 方法机制

输入、拓扑表示、actor、critic、训练数据和终局 reward 全部不变。唯一
机制链为：

`更长且时域一致的 return -> 更多状态直接收到终局成功/碰撞信用 ->`
`更稳定的 twin-Q lane 排序 -> 更稳定的 checkpoint/decoder 校准 ->`
`独立交通上的成功与安全性`。

按 F1 的真实 episode 长度，16-step 的终局信用起点覆盖率估计为
`10.94%`，约为 4-step 的 `3.40x`；batch 32 没有任何直接终局信用样本
的概率降至约 `2.45%`。这些是设计计算，不是实验结果。

## 4. 严格保持不变

- v4.6 拓扑编码器、Graph-SLT、SoftBalancedSlots 和全部网络宽度；
- actor encoder detach、lane entropy scale、speed entropy 和优化器；
- observation、lane mask、hybrid action、conditional speed 和 critic 输入；
- sparse terminal reward、discount `0.99`、batch 32、buffer 20,000；
- 5,000 raw-step warm-up、20,000 raw-step 开发预算、action repeat 3；
- exact-final / highest-training-success 两个 checkpoint；
- target-critic / fusion-0.90 两个 deterministic decoder；
- v4.6 四组合 train-only 联合选择顺序、12-episode 校准和先收据后验证；
- traffic split、episode limit、outcome/selector/mechanism gates；
- promotion 和 formal 协议及正式 test 锁。

TemporalGraph 始终使用其原冻结 4-step 实现，不接受候选方法的 return
改动。

## 5. Claim–Evidence Matrix

| Claim | Reviewer question | Required evidence | Workload / control | Metric | Status |
| --- | --- | --- | --- | --- | --- |
| 长时域正确折扣修复 seed6 | 是否只是换了容易 block？ | 新鲜 Cross seed6 validation | v4.7 G1 | success/collision/timeout + critic diagnostics | planned |
| 改善来自 return estimator | 父方法在同一 block 是否也会通过？ | 相同 seed、calibration、validation 的 v4.6 parent control | G1 vs G2 | paired outcome delta；候选至少保持成功且有两例净改善 | planned |
| 不是单 seed 偶然 | 是否跨训练 seed？ | 独立 Cross seed8 | G3 | non-CARLA hard gate | planned |
| 不破坏必要换道场景 | 长 return 是否让 CARLA 退化？ | CARLA seed5 | G4 | CARLA hard gate + decoder integrity | planned |
| 不只适配 Cross | 是否跨场景？ | roundabout-medium seed3 | G5 | non-CARLA hard gate | planned |
| 可复现且无泄漏 | return 与选择实现是否精确？ | 单元测试、真实 SUMO smoke、冻结哈希、收据时序 | engineering | exact return/discount; 302+ parent regressions | planned |

## 6. 新鲜开发顺序与停止规则

所有单元从头训练，严格串行；historical、post-hoc 和 formal/test 结果均不
作为门禁输入。

1. `G1`：candidate，Cross seed 6，20k；train calibration
   `76000--76011`；validation `67000--67011`。
2. `G2`：v4.6 parent control，Cross seed 6，完全相同的 calibration 与
   validation；仅在 G1 outcome/selector/mechanism 全通过后执行。
3. `G3`：candidate，Cross seed 8，20k；calibration `77000--77011`；
   validation `68000--68011`；要求 G1-vs-G2 因果门禁通过。
4. `G4`：candidate，CARLA seed 5，20k；calibration `78000--78011`；
   validation `69000--69011`。
5. `G5`：candidate，roundabout-medium seed 3，20k；calibration
   `79000--79011`；validation `70000--70011`。

候选 outcome gate 沿用 v4.6：CARLA success `>=0.50`、collision `<=0.10`、
timeout `<=0.50`；其余场景 success `>=0.50`、collision `<=0.45`、
timeout `<=0.30`；全部 off-route 必须为 0。

G1-vs-G2 因果门禁要求：候选 success count 不低于父控制，且满足以下
至少一项：success 净改善 `>=2/12`，或在 success 不下降时 collision 净
减少 `>=2/12`。若父控制本身产物无效则停止为工程问题，不能把缺失控制
解释为科学增益。

任一必需门禁失败立即停止 v4.7，生成新版本归因；不得跳过 G2、补跑
validation seed、改变阈值或进入 promotion。

## 7. Promotion、Formal 与最终目标

只有 G1–G5 全部通过才执行原协议：promotion 为
`2 methods x 3 scenarios x 2 seeds = 12 jobs`、每格 50k；通过哈希门禁
后才解锁 formal 的 `120 jobs`、每格 100k、50 test episodes。最终接受
条件保持：每场景 success delta `>=-0.05`、collision delta `<=+0.05`，且
至少一个场景 success `>=+0.10` 或 collision `<=-0.10`。

## 8. Reviewer-Risk Register

| Risk | Type | Evidence / action |
| --- | --- | --- |
| 16 是事后调参 | design-fixable | 仅由 4-step 终局覆盖不足推导并在实现前冻结；不扫描 8/16/32 |
| horizon 与 bootstrap 同时变化 | evidence-fixable | 将二者定义为同一数学一致 estimator；matched parent control 测整个组件 |
| 训练块选择偏差 | evidence-fixable | 四组合仍只看 train，validation 在收据之后构造 |
| 只改善已知 seed6 | requires-new-result | G3 独立 seed8 为不可补偿门禁 |
| 更多直接信用导致碰撞偏置 | requires-new-result | collision 硬门禁、CARLA 与第三场景 guard |
| 计算或内存增加 | evidence-fixable | 保存 wall time、GPU memory、参数量；buffer 全局容量不变 |
| 新颖性 | needs-search | 当前任务未请求文献搜索；不宣称 prior-art novelty |

## 9. No-Fabrication Status

上述数值仅来自已封存的真实运行或显式设计计算。G1–G5、promotion 和
formal 的所有未运行结果均为 `TBD`，不得预填或推断。

