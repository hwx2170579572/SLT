# v4.8/lr_half 第四轮调参：深地板 + 保持期组合、settle 扫描与地板深度探针

> 承接 `experiments/v48_stability_v3`（r48s3，第三轮 5 候选已完成）。本目录为第四轮（r48s4）。
> 约束不变：只用训练 seed 0；两个 GPU worker；部署统一 `exact_final_actor_deterministic`（无选择器、无 critic 解码）；
> 只评估 cross + carla；评估 100 回合确定性；训练 50000 raw steps / 45001 次梯度更新。

## 第三轮结论（r48s3）与第四轮目标

第三轮 cross 场景关键结果（基准 lr_half=0.84）：

| 候选 | floor | decay | cross 评估 | 碰撞 | 尾段std | 回撤 |
|---|---|---|---|---|---|---|
| tau0025_hold35k | 2.5e-5 | 20k→35k (settle 15k) | **0.97** | 0.03 | 0.065 | 0.128 |
| tau0025_floor2e5 | 2e-5 | 20k→50k (无 settle) | **0.97** | 0.03 | **0.030** | 0.128 |

两个杠杆（更深地板 2.5→2e-5；settle 期 50k→35k）各自把 cross 评估从 0.87（tau_slow_decay）→0.97，且回撤守住 0.128。

**但存在跨场景分歧**（r48s3 REPORT.md 的 carla 行）：

| carla | 后10k | EMA终点 | 尾段std | 回撤 |
|---|---|---|---|---|
| lr_half | 0.828 | 0.871 | 0.056 | 0.281 |
| hold35k (2.5e-5 + settle) | 0.914 | 0.910 | **0.034** | 0.127 |
| floor2e5 (2e-5 无 settle) | 0.826 | 0.726 | **0.093** | 0.247 |

即：**深地板（2e-5）在 cross 上最平滑（尾段 std 0.030），但在 carla 上最粗糙（尾段 std 0.093，反劣于 lr_half 0.056）；
settle（hold35k）在 carla 上最平滑（尾段 std 0.034）。** 这正是第三轮"无候选双场景同时达标"（recommendation=None）的根因：
hold35k 过 cross 但 cross 尾段 std 改进不足 10%；floor2e5 过 cross 但 carla 尾段 std 反劣。

**第四轮核心假说**：把两个杠杆**组合**（深地板 2e-5 + settle），期望同时拿到 cross 的平滑（来自深地板）与 carla 的平滑
（来自 settle），成为第一个双场景同时达标、且 cross 评估 0.97 的候选。settle 扫描（{0,5,10,15,20k}）直接检验
"深地板在 carla 上的粗糙是否被 settle 修复、需要多长 settle"。

## 第四轮候选

所有候选固定 tau=.0025（第三轮确认最优：.003 有害、.001 carla 崩塌）、decay_start=20k（第三轮确认延后起点有害）、
batch 32、buffer 20000、起始 lr 5e-5（避免恒低 lr 欠拟合）。

| 候选 | floor | decay_end | ramp | settle | 逻辑依据 |
|---|---|---|---|---|---|
| lr_half | 5e-5 | — | — | — | 复用参考 |
| tau0025_floor2e5_hold35k | 2e-5 | 35k | 15k | 15k | **组合**：深地板 + settle，核心假说 |
| tau0025_floor2e5_hold40k | 2e-5 | 40k | 20k | 10k | settle 扫描 |
| tau0025_floor2e5_hold30k | 2e-5 | 30k | 10k | 20k | settle 扫描 |
| tau0025_floor2e5_hold45k | 2e-5 | 45k | 25k | 5k | settle 扫描（最接近 floor2e5 的首个 settle 点） |
| tau0025_floor1p5e5_hold35k | 1.5e-5 | 35k | 15k | 15k | 地板深度探针（固定 settle 15k） |

对照关系：
- **settle 扫描 @ floor 2e-5**：{0k=floor2e5 已知 0.97/尾段std0.030, 5k, 10k, 15k, 20k}，读"深地板需要多长 settle 才能压住 carla 粗糙"。
- **地板深度 @ settle 15k**：{2.5e-5=hold35k 已知 0.97, 2e-5, 1.5e-5}，读"地板是否继续向下探"。

## 已知混淆（判读时必须显式处理）

1. **settle 时长与 ramp 斜率反相关**：decay_start=20k 与 raw_budget=50k 固定，故 ramp 时长=decay_end−20k 与 settle=50k−decay_end
   之和恒为 30k。settle 越长的候选（hold30k）ramp 越短越陡。hold30k 的 10k ramp 斜率（3e-9）已超过第三轮崩塌点
   start30k_hold40k（2.5e-9），若 hold30k 崩塌须按"ramp 过陡"而非"settle 过长"解读。
   **判读按"ramp 斜率 × settle 时长"二维，不单独宣称最优 settle 时长。**
2. **地板深度与 ramp 斜率共变**：同一 decay_end 下 5e-5→1.5e-5 的斜率（2.33e-9）比 5e-5→2e-5（2.0e-9）陡 16%，
   floor1p5e5_hold35k 与 floor2e5_hold35k 的差异不能只归因于地板深度。1.5e-5 探针只用于判断"是否继续向下探"的方向，
   不作精确量化。lr_quarter（恒 2.5e-5）=0.63 的欠拟合警告是中等风险而非灾难：floor 候选前 20k 原始步仍以 5e-5 学习，
   地板只影响末段。
3. **carla eval 饱和**（所有候选 eval=1.0），carla 侧只能靠 train 尾段 std/drawdown 判别；cross 侧 eval 0.97 已到天花板
   （100ep 噪声 ±0.017），排序靠曲线指标。推荐由**最差场景**（max 尾段 std 相对 lr_half 的比值）决定，不按单场景单数字。

## 判读标准

- 主指标 = cross + carla 训练曲线尾段 std / 5k 回撤 / jerk（动态范围大、可区分）。
- 期望：floor2e5_hold35k（组合）在 cross 上维持 0.97 且尾段 std ≤ 0.065（hold35k），同时在 carla 上尾段 std ≤ 0.056（lr_half）
  且回撤 < 0.128，即第一个双场景同时达标的候选。
- 1.5e-5 探针：若 carla 尾段 std 相对 2e-5 进一步恶化，则地板已越过最优深度，停止向下探。
- 单种子（seed 0）；方差/阴影仅描述单次训练内波动，不代表跨种子稳定性。

## 运行

`run_v48_stability_v4.ps1`（check/run/report）。产物在 `r48s4/`。
