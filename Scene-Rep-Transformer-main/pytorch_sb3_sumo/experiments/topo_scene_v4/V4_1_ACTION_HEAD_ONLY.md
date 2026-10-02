# v4.1：Decision-Aligned Hybrid Action（动作头单项迭代）

状态：`implementation / needs-evidence`  
父方案：`INITIAL_PLAN.md`  
阶段 0 结果：`results_topo_v4_dev/stage_0_offline_diagnosis/probe_results.json`

## 优化后的 idea card

- 任务：在保持 v2 拓扑—时序表示与训练协议不变的前提下，把可预测的路线意图转化为正确时机的可执行 SUMO 动作。
- 缺口：失败候选的 route slot 可高精度线性分离路线意图，但连续 actor 在所有 held-out 换道窗口均输出 keep。
- 根因：二维连续 SAC 把本质离散且受可行性约束的 lane action 当作平滑连续变量；critic 学习到的是经过 `±1/3` 阈值后的分段常数转移，actor 又没有显式的可行动作集合。
- 核心洞察：无需继续增加表示监督；应让策略分布与执行器语义同构，使路线信息通过 categorical lane intent 直接作用于闭环动作。
- 贡献类型：方法/动作推断过程；辅以“表示可读但策略不用”的诊断性经验发现。新颖性未经文献检索，标记 `needs-search`。
- 当前边界：只对本仓库冻结的 SUMO/SB3 迁移协议与 TemporalGraph 比较负责，不提前声称通用自动驾驶优势。

## 证据约束

阶段 0 使用 40/40 个成功 TemporalGraph CARLA validation episode，保留 1483 个相同 observation；episode-split 测试包含 448 个样本，其中 66 个为路线要求换道的窗口。

| 诊断 | TemporalGraph | v2 失败候选 |
| --- | ---: | ---: |
| route-slot 意图 balanced accuracy | 1.000000 | 0.991228 |
| full-latent 意图 balanced accuracy | 1.000000 | 0.992982 |
| 换道窗口动作匹配 | 1.000000 | 0.000000 |
| 全测试集 keep rate | 0.408482 | 1.000000 |
| route-slot 均值消融导致的横向变化 | 0.055954 | 0.009002 |
| route-slot 均值消融导致的 lane-command flip | 0.073661 | 0.000000 |

因此 v4.1 不加入 `L_var` 或 `L_intent`。较低的候选 latent std 是已知现象，但不能在 probe 仍为 0.991 时作为首要因果解释。

## 方法蓝图

输入仍为原 `trajectory`、`map`，另增加当前物理可执行的三类动作 mask；mask 不包含任务路线目标。共享 encoder、Graph-SLT、SoftBalancedSlots `λ=0.01` 和 actor stop-gradient 全部保留。

策略为：

`π(k,v|z,m) = π_lane(k|z,m) π_speed(v|z,k)`。

- `k∈{-1,0,+1}`：masked categorical；不可执行类别的概率严格为 0；
- `v`：每个 lane intent 各自拥有 squashed-Gaussian 均值与方差；
- actor/target value 对三个 lane intent 精确求期望，只对条件速度进行重参数采样；
- critic 将 replay 中的二维兼容动作转换为 `[normalised_speed, one_hot(k)]`；
- rollout 与 warm-up 的横向动作只可能是精确的 `-1/0/+1`；外部 Gymnasium action space 仍为二维 `[-1,1]`，所以环境、replay 与评估接口保持兼容。

## 单一变化边界

保持不变：

- v2 soft encoder、MERGE、route/direction top-k、残差与分池；
- `λ_balance=0.01`、Graph-SLT、representation optimizer；
- reward、N-step replay、raw-step clock、动作保持、学习率、batch、buffer；
- actor 对共享 encoder 的 detach；
- 训练/validation/test 的 60/20/20 协议。

变化项：

1. 连续二维 actor → masked categorical lane + conditional speed；
2. 连续 lateral critic 输入 → lane one-hot；
3. Monte-Carlo 单 lane actor/target → 三 lane 精确枚举；
4. warm-up 从连续随机 lateral → mask 内均匀离散 lane。

这些是同一个“动作语义对齐”机制的必要组成，不作为四个独立贡献宣称。

## 反证与下一步

- 工程门禁：mask、概率、梯度、one-hot、warm-up、save/load、real SUMO 与完整回归必须全部通过。
- D1 CARLA seed-0 20k 若 success < 0.30、collision > 0.10、timeout > 0.70、keep > 0.90 或 accepted lane change < 0.02，则 v4.1 立即失败。
- D3 固定为同一 `topo_v2_soft` encoder、`λ_balance=0.01`、reward、detach 与训练预算下恢复原连续 lateral 接口；它只移除完整 hybrid action interface，作为不可补偿硬门槛的关键消融。
- 若 lane action 已在正确窗口执行但仍低速 timeout，下一版本只能单独研究速度/进度问题；不得在 v4.1 中顺带改奖励。
- 若 v4.1 失败且 trace 表明策略概率已经指向正确类别但执行仍错位，回到动作时序/动作保持归因；只有新的证据否证阶段 0 时，才启用表示监督分支。

## 冻结后的评估与访问控制

- Development 使用 `40000 + 1000×training_seed` 的 validation episode seed block；D1 与 D3 严格共享同一 block。
- Promotion 使用 `50000 + 1000×training_seed`，formal 使用 `100000 + 1000×training_seed`；同一 scenario/seed 的两种方法严格配对，不同训练 seed 的 episode block 不重叠。
- 训练过程不做中间 validation 选模（`eval_freq=0`），仅保存最终模型并执行一次合同规定的最终评估，避免重复窥视开发集。
- 正式 test 入口必须同时验证 promotion pass receipt、实验合同 SHA256 与 implementation-freeze SHA256；仅提供任意字符串哈希不能解锁。

## 风险登记

- `requires-new-result`：混合 SAC 是否比 TemporalGraph 稳定尚未知；
- `design-fixable`：单一 joint entropy temperature 可能使 categorical/continuous 熵权衡失衡；首轮冻结 target entropy `-2.0`，不预先扫描；
- `evidence-fixable`：CARLA 只有一个流量资产，必须依赖种子、场景和失败轨迹披露边界；
- `needs-search`：混合动作 SAC 与 action masking 的相关工作尚未检索，当前不作新颖性声明；
- `efficiency`：精确枚举增加 critic 计算，promotion 必须报告参数、更新耗时、推理耗时和显存。
