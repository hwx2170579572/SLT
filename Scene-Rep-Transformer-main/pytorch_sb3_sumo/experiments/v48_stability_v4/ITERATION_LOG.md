# 迭代记录

## 2026-09-22 / v48_stability_v4 / 第四轮设计

- 承接 r48s3：两个杠杆（更深地板 2.5→2e-5；settle 期 50k→35k）各自把 cross 评估 0.87→0.97，回撤守住 0.128。
- 第三轮复核发现**跨场景分歧**：深地板（floor2e5）在 cross 最平滑（尾段 std 0.030）但在 carla 最粗糙（0.093，反劣 lr_half 0.056）；
  settle（hold35k）在 carla 最平滑（0.034）。第三轮因此无候选双场景同时达标。
- 第四轮假说：组合深地板 + settle，同时拿 cross 与 carla 的平滑；settle 扫描 {0,5,10,15,20k} 读"深地板需要多长 settle 压住 carla 粗糙"。
- 候选：5 新 + lr_half 复用（floor2e5_hold{35k,40k,30k,45k} + floor1p5e5_hold35k）。定稿前经 3-agent 对抗评审：
  - 删去冗余的 floor1p5e5_hold50k（1.5e-5 无 settle，与 floor2e5 几乎重复），改为 settle 5k 点补全扫描。
  - 明确两个混淆：settle 时长与 ramp 斜率反相关（ramp+settle=30k 恒定）；地板深度与 ramp 斜率共变（1.5e-5 斜率比 2e-5 陡 16%）。
    判读按"ramp 斜率 × settle 时长"二维，不单独宣称最优 settle 时长。
  - carla eval 饱和为 1.0，carla 侧只能靠 train 尾段 std 判别；推荐由最差场景决定。
- 约束沿用：seed 0、2 GPU worker、exact_final_actor_deterministic、只 cross+carla、100 回合确定性、50000 raw steps。
- 正式运行等待用户手动启动。具体已通过检查以 r48s4/preflight.json 及其绑定的 tests/smoke 证据为准。
