# mst_slt 修复超参后仍失败（1% 成功）的根因重定位

生成时间：2026-09-27 03:14（基于修复版 mst_slt 训练+评估完成后的完整结果）

## 一句话结论

**根因是环境的稀疏三元终止奖励结构，不是超参，也不是动作头。** 修复超参（lr=1e-4/batch=32/warmup=5000）没有改变结局——三套异构配置（旧加速版、修复版、hold35k 混合头）全部坍缩为「停车等超时」的零回报吸收盆地，成功率 0~9%。

## 关键证据（推翻上一版"超参是主因"的判断）

| 证据 | 数据 |
|---|---|
| 修复版评估 | success_rate=0.01、timeout=0.83、collision=0.16、mean_return=-0.15 |
| 修复版训练 | 102 episode 仅 1 次成功（ep19），ep33 起几乎全 timeout |
| 旧加速版训练 | 102 episode 9 次成功（8.8%），ep27 见顶 last20=0.25，随后退化全 timeout |
| hold35k 训练 | ep12 起 21/21 全 timeout，0% 成功 |
| **修复版 ent_coef** | **0.2 → 0.0035 单调衰减（57×）**，ent_coef_loss 同步衰减到 0.0073 |
| 修复版 critic_loss | last=0.0033（critic 学会预测 0，无法估计罕见 +1） |

**核心反转**：ent_coef 从 0.2 衰减到 0.0035 是 **SAC 自动温度调节的正常收敛形态**（策略变确定 → 按熵目标自动降 α），**与 lr/batch/warmup 无关**。上一版报告把「ent_coef 23× 衰减」归因于加速超参是判断错误。

## 机理（结构性陷阱，非优化器可救）

1. `sumo_env.py:470`：`raw_reward = float(success) - float(collision)`，中间步全 0，timeout/off_route=0。
2. `sumo_env.py:645-646`：`setSpeedMode(ego,0)`/`setLaneChangeMode(ego,0)` 关闭 SUMO 让行保护，ego 可自由停车。
3. 由此「停车等待直到超时」成为**价值恒为 0 的零风险吸收策略**，严格支配早期 `p_success < p_collision` 时价值为负 `(p_success − p_collision < 0)` 的抢行策略。
4. 策略滑入该盆地 → policy entropy→0 → SAC 自动把 ent_coef 压到 0.0035 → 回放缓冲全为 timeout、无信息梯度 → 正反馈锁死。
5. 唯一 +1 需跨 ~177 个零奖励决策步回传，被 discount=0.99 稀释到近零 → critic 坍缩为预测 0。

三种 lr/batch/warmup/ent_coef/动作头配置全败，证明吸引子来自 reward 定义，而非优化器。

## 方案（按优先级）

### P1（必做）reward shaping —— 新建 wrapper，不改 sumo_env.py
在 fast-developer 新建 reward-shaping 包装器（`gym.Wrapper` 或 `SumoEnv` 子类）：
- **(a) timeout 改为负值**（-1，与 collision 同罚）——消除「0 > -1」的支配关系；
- **(b) 稠密路由进度奖励**（沿 -E1→-E0 的剩余距离减少量 / route-aligned 速度）；
- **(c) 小步长生活成本**（每决策步 -0.005~-0.02）抑制原地停留。
- 预期：打破 timeout 零回报盆地，让让行策略价值曲线为正、梯度可回传；成功率从 0-9% 显著提升（上界参考 SUMO 默认让行 30/30）。
- 约束：以新文件形式放 fast-developer，不触碰 sumo_env.py。

### P2 示范暖启动（不改环境）
用已知可解的 SUMO 默认让行轨迹（实测 30/30、161-168 步）做离线 BC 预训练 actor 或初始化 replay buffer，绕开冷启动坍缩。

### P3 熵系数下限（不改环境）
给 ent_coef 设地板（固定 ~0.05-0.1 或 target_entropy 下限），阻止 alpha 衰减到 0 关闭探索。纯算法超参改动。

### P4 课程学习（需改 rou 配置，新文件放 fast-developer）
放宽冲突车流（加大发车间隔使间隙变宽）→ 收敛后逐步加密回 2.5/3/4s 密度。

### P5 加大预算（可选增强）
raw_steps 5w→15w+，评估 300+ 集，加密 checkpoint。单独加预算无法解决 timeout 吸引子，须在 reward 修复后放大收益。
