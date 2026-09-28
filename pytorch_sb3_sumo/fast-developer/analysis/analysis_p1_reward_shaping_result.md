# P1 reward shaping 实验结果：改变了行为，但没学会让行

生成时间：2026-09-27（mst_slt + RewardShapingWrapper 训练 66 episode / 24k raw 步后的中段结论）

## 一句话结论

**P1 的 reward shaping 打破了「停车等超时」吸收盆地（timeout 从修复版 83% 降到整体 26%），ego 学会了「前进到路口」，但成功率仍是 ~1.5%（66 集仅 ep19 一次），且 timeout 正重新回升——策略滑入「前进到路口停下等超时」的第二个次优盆地。**

## 训练配置（单一变量，仅 reward 不同）

- 方法：mst_slt（连续动作头），lr=1e-4 / batch=32 / learning_starts=5000 / buffer=20000 / discount=0.99 / action_repeat=3 —— 与修复版完全一致。
- 唯一变量：环境层包裹 `RewardShapingWrapper`（见 `reward_shaping_wrapper.py`）：
  - (a) timeout -> -1（与 collision 同罚）；
  - (b) 进度奖励 `0.01 * ΔgetDistance`（沿 route 累计行驶距离增量）；
  - (c) 生活成本 `-0.005`/决策步。
- 评估复用 fixed 脚本的 6-worker eval（无 wrapper，读 info），与修复版同源可比。

## 关键证据（train_monitor.csv，66 episode）

| 分桶（10 集） | timeout | collision | success |
|---|---|---|---|
| ep1-10 | 1/10 | 9 | 0 |
| ep11-20 | 0/10 | 9 | **1（ep19）** |
| ep21-30 | 0/10 | 10 | 0 |
| ep31-40 | 2/10 | 8 | 0 |
| ep41-50 | **6/10** | 4 | 0 |
| ep51-60 | **9/10** | 1 | 0 |
| ep61-66 | **5/6** | 1 | 0 |

- 成功仅 ep19 一次，return=+2.007（success +1 + 进度 ~1.0 - 生活成本），之后 **47 集零成功**。
- `ent_coef`：0.152（ep36）→ 0.036（ep64），持续单调衰减（探索关闭，逼近修复版的 0.0035）。
- timeout 的 return≈-1.3 = -1(timeout) + 0.7(进度，ego 已前进 ~70m) - 1.0(生活成本 200 步) —— 说明 timeout 时 ego **不是停在起点，而是前进到路口后停下等超时**。

## 与修复版（无 shaping）对比

| 指标 | 修复版 | P1 shaping |
|---|---|---|
| success_rate | 1%（102 集 1 次） | 1.5%（66 集 1 次） |
| timeout 占比 | 83% | 26%（但 ep51-66 回升到 ~90%） |
| collision 占比 | 16% | 62% |
| 坍缩盆地 | 停在起点等超时 | 前进到路口停下等超时 |

## 根因（为什么 P1 不够）

1. **进度奖励只引导「前进」，不引导「择机通过」**：`getDistance` 增量在 ego 到达路口停下后归零，不再产生任何梯度。ego 学会了「前进到冲突点」，但「何时加速穿行」这一时序决策仍然只有稀疏的 `success=+1` 能提供信号。
2. **reward 量级失衡**：`抢行碰撞 return≈-0.6` > `路口等待 timeout return≈-1.3`，策略在两个次优盆地间摇摆，最终 critic 学会「碰撞危险」后退回「路口等待」——而「路口等待」仍非 success。
3. **时序决策 + 稀疏 success 的结构性难题未解**：唯一 +1 需跨 ~55 决策步（成功让行 161-168 raw 步）的「观察间隙→择机穿行」精确时序回传，被 discount=0.99 稀释到近零，critic 无法据此估计「穿行时机」的价值。
4. 熵坍缩（0.152→0.036）是结果而非原因：策略在两个次优盆地间摇摆无路可走，熵自然耗尽。

## 结论

P1 的 `timeout=-1 + 进度 + 生活成本` 三项 shaping **全部按预期生效**（都体现在行为轨迹上），但**不足以让 mst_slt 学会让行**——它把问题从「停在起点」推进到「停在路口」，却无法跨越最后一步「择机穿行」。这是稀疏 `success=+1` 无法支撑精确时序决策的体现，需要示范（P2）或课程（P4）提供更早的正向信号。

## 下一步（按预期有效性排序）

- **P2 示范暖启动（最推荐）**：用 SUMO 默认让行轨迹（实测 30/30 成功、161-168 步）做 BC 预训练 actor 或填充 replay buffer，直接教会「让行」时序，再叠加 P1 shaping 微调。
- **P4 课程学习**：先放宽冲突车流间隙（发车间隔 2.5/3/4s → 6s），在高间隙下让 success 不再稀疏、学会让行，再逐步加密回原密度。
- **P1 增强（gap-aware shaping）**：在 wrapper 里加「安全等待」信号（接近冲突点减速 + 检测到 gap 时正向奖励），但需访问交通状态、实现更复杂、可能引入 bias。
- **P3 熵地板**：给 ent_coef 设下限阻止探索关闭，但单独用无法解决「择机通过」的稀疏梯度问题。

## 产物

- `reward_shaping_wrapper.py`、`train_intersection_mst_slt_rs.py`（均可复用于 P2 微调阶段）。
- 训练产物 `mst_slt__intersection_rs/`（66 episode 的 train_monitor.csv，保留作 P1 失败证据）。
