# v4.8/lr_half 第三轮调参：tau_slow_decay 邻域的"保持期 + 衰减起点/地板"微调

> 承接 `experiments/v48_stability_v2`（r48s2，第二轮 4 候选已完成）。本目录为第三轮（r48s3）。
> 约束不变：只用训练 seed 0；两个 GPU worker；部署统一 `exact_final_actor_deterministic`（无选择器、无 critic 解码）；
> 只评估 cross + carla；评估 100 回合确定性；训练 50000 raw steps / 45001 次梯度更新。

## 第二轮结论 + 复核纠正

第二轮（r48s2）cross 上的 actor 评估成功率与关键曲线指标：

| 候选 | 变什么 | cross 评估 | 碰撞 | 尾段std | 5k回撤 | jerk |
|---|---|---|---|---|---|---|
| lr_half（参考） | — | 0.84 | 0.15 | 0.070 | 0.240 | 62.3 |
| **tau_slow_decay** | tau .0025 + lr 5e-5→2.5e-5(20k→50k) | 0.87 | 0.12 | **0.014** | **0.128** | 59.8 |
| tau_slow_decay_highfloor | 同上但地板 3.75e-5 | 0.72 | 0.28 | 0.028 | 0.184 | 76.2 |
| tau_slower | tau .001 | 0.84 | 0.16 | 0.076 | 0.129 | — | carla 崩塌 0.62（超时0.39）|
| tau_slow_batch64 | tau .0025 + batch64 | 0.69 | 0.31 | 0.023 | 0.248 | — |

**第二轮复核（对抗性）带来的三点关键纠正：**

1. **"tau_slow_decay 评估 0.87 优于 lr_half 0.84" 是噪声**（Δ0.03，二项 z≈0.6；jerk Δ2.6 亦不显著）。真正可区分的
   只有**曲线尾段指标**（尾段 std 0.014 vs 0.070、回撤 0.128 vs 0.240），因为它们动态范围大、样本量大。
   因此本轮**按曲线尾段 std/回撤/jerk 选型，不按 eval success**（100 回合二项噪声 ±0.03~0.04）。
2. **"地板越低越好"只有一对点**（2.5e-5 vs 3.75e-5，Δ0.15 显著但外推过度）；lr_quarter（恒 2.5e-5）=0.63 证明
   2.5e-5 的价值依赖"先 5e-5 后退火"的路径。地板方向可继续向下探，但不能走极端。
3. **tau 轴 .0025 未必是峰**（只有 .005/.0025/.001 三点，缺 .002/.003），且加了衰减后排序反转（late_decay tau .005=0.91
   > tau_slow_decay .0025=0.87），故 .003+decay 可能是 success-stability 的膝点，值得测。

## 核心机制发现（决定第三轮方向）

`learning_rate()` 的 raw_step = 5000 + n_updates，`raw_budget=50000`。**第二轮所有候选都固定 `decay_end=50000`，
即训练终点——地板只在最后一步才被触达，训练全程不存在"到达地板后保持（settle）"阶段。** 因此"延迟 decay_start"
不只是"多跑一会儿高 lr"，而是同时把稳定化窗口压到 0。`decay_end < 50000` 能把"衰减斜率"与"地板保持时长"解耦，
是第二轮完全没触及的、定性不同的 regime：

- target_slow（无衰减）尾段 std 0.085 / 回撤 0.305 —— 晚期仍在高 lr 下持续移动；
- tau_slow_decay（20k→50k）尾段 std 0.014 —— 有退火但**零保持期**（退火到终点即停）；
- 假说：**5e-5 跑到 20k~30k 建顶 → 短 ramp 退火 → 地板保持 10k~15k 定形**，既保住 target_slow 的天花板，又比
  tau_slow_decay 多一段"低 lr 收敛期"进一步压低回撤。与 lr_quarter（全程低 lr 欠拟合）不矛盾。

## 第三轮假设与候选

所有候选固定 tau ∈ {.0025,.003}（避开 .001 的 carla 崩塌）、batch 32、buffer 20000、起始 lr 5e-5（避免恒低 lr 欠拟合）。
每个候选相对锚点 tau_slow_decay（tau .0025 + 20k→50k + 地板 2.5e-5）**只改一个轴**，外加一个组合：

| 候选 | 变什么（相对 tau_slow_decay） | 逻辑依据 |
|---|---|---|
| lr_half | 复用（参考） | 固定基准 |
| **tau0025_hold35k** | decay_end 50k→35k（地板保持 15k） | 打破"零保持期"，退火后给 15k 步定形期 |
| **tau0025_start30k** | decay_start 20k→30k | 满 lr 多跑 10k 更新，抢回一部分 0.97 上限 |
| **tau0025_start30k_hold40k** | start 20k→30k + end 50k→40k（保持 10k） | 延后建顶 + 短保持，最可能的"高顶+稳"合成 |
| **tau0025_floor2e5** | 地板 2.5e-5→2e-5 | 地板轴向下探，更彻底淬灭尾段扰动 |
| **tau003_decay20k** | tau .0025→.003 | tau 轴补点，检验 .003+decay 是否落在 success-stability 膝点 |

干净 1D 对比对（均以 tau_slow_decay 为原点）：
- 保持轴 @start20k：锚点(0k) / hold35k(15k)；@start30k：start30k(0k) / start30k_hold40k(10k)
- 起点轴 @hold0：锚点(20k) / start30k(30k)；地板轴：2.5e-5 / 2e-5；tau 轴：.0025 / .003

## 判读标准

- **主指标 = cross 训练曲线尾段**：5k 后最大回撤、尾段 std、excess total variation、jerk（动态范围大、可区分）。
- 次指标 = cross actor 评估成功率/碰撞（仅当 Δ≥0.05 才当作信号；Δ≈0.03 视为噪声）。
- carla 只作防塌缩保险（预期除异常外全 1.0）；任何 carla 评估 <1.0 即判为 tau 方向失控。
- 期望：某候选 cross 曲线回撤/尾段 std ≤ tau_slow_decay(0.128/0.014) 且 eval 不显著低于 0.87，即"稳+不降"；
  若 hold35k/start30k_hold40k 同时把 eval 推到 0.9+ 且尾段仍稳，则坐实"延后建顶 + 短 ramp + 保持"方向。
- 单种子（seed 0）；方差/阴影仅描述单次训练内波动，不代表跨种子稳定性。

## 运行

`run_v48_stability_v3.ps1`（check/run/report）。产物在 `r48s3/`。
