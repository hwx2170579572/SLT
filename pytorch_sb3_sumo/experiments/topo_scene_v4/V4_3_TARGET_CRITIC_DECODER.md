# v4.3：Target-Critic Lane Decoder

状态：`preregistered_after_v4.2-r1_failure / implementation_complete / engineering_gate_passed / confirmation_not_started`  
父方法：`v4_2_factorized_lane_entropy`  
归因：`results_topo_v4_2_r1_dev/attribution/c1_failure/`

## 单一改动

训练期间不变；只在 `deterministic=True` 的评估/部署路径替换 lane decoder。

1. actor 对三个 lane code 产生各自的 deterministic conditional speed；
2. target twin critics 对三个完整 action 做精确枚举；
3. 对不可行动作 mask 后，选择 `argmax_a min(Q1_target,Q2_target)`；
4. 输出所选 lane 的 actor conditional speed 与精确 lane code。

`deterministic=False` 仍由 masked categorical actor 采样。reward、encoder、replay、loss、entropy、warm-up、训练步数和 final-checkpoint 选择全部不变。

## 为什么不是其他改动

- seed0 final actor 0/12，换 validation block 后仍 0/12；
- seed0 final online-critic decoder 仅 1/12；
- seed0 final target-critic decoder 8/12；
- seed1 final actor与 target-critic decoder 均 12/12；
- best-training checkpoint 会把 seed1 actor 降为 0/12，因此不采用 checkpoint cherry-pick。

这表明 Polyak target critic 保留了比 actor argmax 与 online critic 更稳定的 lane-value 信号。

## 不计作确认的 pilot

43000/44000 block 上所有 actor/online/target replay 都用于提出假设，不计作 v4.3 通过。

## 确认顺序

1. R1：seed0 parent final checkpoint，用 v4.3 实现回放未访问 validation 46000–46011；
2. R2：R1 通过后，seed1 parent final checkpoint，同一新 block；
3. T1：R2 通过后，新训练 CARLA seed2，20k，validation 47000–47011；
4. T2：T1 通过后，Cross seed0，20k，validation 48000–48011；
5. T3：T2 通过后，Cross seed1，20k，validation 49000–49011。

CARLA 每 cell 必须同时满足 outcome、安全、route-event 与 target-Q decoder 门槛。任一失败停止。全部通过才允许 12-cell paired promotion；formal/test 在 promotion 通过前保持锁定。

## 工程门记录

- 专项测试：13 passed；
- 完整回归：256 passed，5 条既有 SB3 checker warning；
- 原迁移保护：111/111 unchanged；
- 真实 SUMO preflight：候选 CARLA、候选 Cross、TemporalGraph CARLA 全部通过；
- 60 raw-step 训练 smoke：31 次 learner update，checkpoint CRC 正常；
- 训练 smoke 的 target decoder/action 一致率：202/202；
- 冻结 checkpoint 重放 smoke：源模型 SHA-256 匹配，迁移后 1/1 episode 完成且 decoder/action 一致率为 1.0。

以上均为工程证据，不计作 R1/R2/T1 效果确认，也不解锁 promotion 或 formal/test。
