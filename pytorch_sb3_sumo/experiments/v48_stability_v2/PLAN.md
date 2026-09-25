# v4.8/lr_half 第二轮调参：tau 慢 + lr 衰减 / batch 的组合筛选

> 承接 `experiments/v48_stability_v1`（r48s1，第一轮 5 候选已完成）。本目录为第二轮（r48s2）。
> 约束不变：只用训练 seed 0；两个 GPU worker；部署统一 `exact_final_actor_deterministic`（无选择器、无 critic 解码）；
> 只评估 cross + carla；评估 100 回合确定性；训练 50000 raw steps / 45001 次梯度更新。

## 第一轮结论（决定第二轮方向）

第一轮 5 候选（对比基准 lr_half = LR 5e-5, tau .005, batch 32）在 cross 上的 actor 评估成功率：

| 候选 | 变什么 | cross 评估成功率 | 碰撞 | 关键曲线特征 |
|---|---|---|---|---|
| lr_half（参考） | — | 0.84 | 0.15 | 回撤 0.24，尾段 0.708 |
| **target_slow** | tau .0025 | **0.97** | 0.03 | 上限最高，但尾段 std 0.085、回撤 0.305、收敛晚 |
| late_decay | lr 5e-5→2.5e-5（20k→50k） | 0.91 | 0.09 | 回撤 0.414（最差），受低地板 + scheduled 分支影响 |
| batch64 | batch 64 | 0.80 | 0.19 | 曲线最平滑（std 0.016），但上限不升 |
| lr_quarter | lr 恒 2.5e-5 | 0.63 | 0.37 | 曲线"最稳"（回撤 0.207）但**欠拟合** |

carla 全部饱和（评估成功率均 1.0），不作为区分场景；cross 是唯一有区分度的场景。

**三个可复用的机制发现：**
1. **慢 target（tau .005→.0025）是唯一大幅提上限的单旋钮**（+0.13），但代价是尾段震荡与回撤变大。
2. **lr 衰减有益上限**（+0.07），但 (a) 终点 2.5e-5 太低——lr_quarter 证明 2.5e-5 会欠拟合；(b) 衰减触发
   `scheduled` 分支（每原始步 1 次梯度更新）带来更大回撤。
3. **batch64 只降梯度方差、让曲线平滑，不提上限**——可作为"平滑器"与慢 tau 组合。

## 第二轮假设与候选

核心假设：**慢 target 提上限，lr 衰减收紧尾段，batch64 压尾段震荡——三者按机制组合应得到"稳步上升 + 后期收敛"的曲线。**
同时沿 tau 轴继续外推，确认 .0025 是否已是该方向的最优。

| 候选 | 变什么（相对 BASE=LR 5e-5/tau .005/batch 32） | 逻辑依据 |
|---|---|---|
| lr_half | 复用（参考） | 固定基准 |
| **tau_slow_decay** | tau .0025 + lr 5e-5→2.5e-5 | 组合两个提上限机制（旗舰） |
| **tau_slow_decay_highfloor** | tau .0025 + lr 5e-5→3.75e-5 | 检验"late_decay 被 2.5e-5 地板拖累"——抬高地板保上限 |
| **tau_slower** | tau .001 | tau 轴外推，确认 .0025 是否最优 |
| **tau_slow_batch64** | tau .0025 + batch 64 | 用 batch64 平滑 target_slow 的尾段震荡 |

所有新候选 LR 起点都保持 5e-5（第一轮已排除"恒 2.5e-5 欠拟合"）。

## 判读标准

- 主要看 **cross 的 actor 评估成功率 + 碰撞率**（100 回合确定性，最可靠信号），
  其次看 **训练曲线稳定性**（5k 后最大回撤、尾段 std、excess total variation）。
- 期望出现"上限 ≥ target_slow(0.97) 且回撤/震荡低于 target_slow"的候选；若 tau_slow_decay_highfloor
  超过 tau_slow_decay，则坐实"2.5e-5 地板过低"；若 tau_slower 回落，则 tau 轴在 .0025 附近达峰。
- 单种子（seed 0）；方差/阴影仅描述单次训练内波动，不代表跨种子稳定性。

## 运行

`run_v48_stability_v2.ps1`（check/run/report）。产物在 `r48s2/`。
