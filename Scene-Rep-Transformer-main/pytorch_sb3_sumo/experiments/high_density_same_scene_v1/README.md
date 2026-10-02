# 高车流同场景六方法对比 v1

本实验只提高社会交通需求，不改变 `carla`、`cross`、`roundabout` 的地图、ego
路线、观测、动作、奖励和 episode 上限。原始 route 文件保持只读；新增车辆和行人
写入 profile 下按“方法/seed/场景”隔离的 `high_density_overlays_v1/`，并由每个 run
收据中的 SHA-256 manifest 绑定。因此该实验
与已有普通密度结果、场景和脚本在命名及产物目录上完全分开。

## 六种方法

三项指定基线是 `MST+SLT`（`scene_rep`）、`TemporalGraph` 和原始完整方法
`Initial Full`（`topo_scene`）。另外三项迭代按已有真实闭环结果筛选：候选至少要同时
覆盖 CARLA 与 Cross；先比较已覆盖目标场景的等权成功率，再比较碰撞率和覆盖度。

| 角色 | 方法 | 现有筛选证据 | 边界 |
| --- | --- | --- | --- |
| Baseline | MST+SLT | 系统矩阵 | 指定基线 |
| Baseline | TemporalGraph | 系统矩阵 | 指定基线 |
| Baseline | Initial Full | v1 开发结果 | 指定初版完整方法 |
| Iteration #1 | v4.8 Tie-only Replicated Calibration | 三场景 promotion，宏成功率 0.8333 | Cross 碰撞非劣门禁失败，不代表已接收方法 |
| Iteration #2 | Full+BalancedSlots | 三 seed、六场景系统矩阵；目标三场景宏成功率 0.7578 | 历史预算不同，数字只用于透明筛选 |
| Iteration #3 | v4.1 Decision-Aligned Hybrid | CARLA/Cross 宏成功率 0.6667、宏碰撞率 0.0417 | 因 seed 不稳定被开发门禁拒绝 |

v4.3 与 v4.1 的已覆盖场景宏成功率并列，但宏碰撞率更高（0.125），因此排在其后。
v4.9 在本协议冻结时尚无完整开发汇总，未被当作“已验证最好版本”。筛选数字来自
`protocol.json` 指向的已有真实 artifact；它们不是跨协议显著性结论。

## 高密度设置

| 场景 | 机动车倍率 | 行人倍率 | 设置理由 |
| --- | ---: | ---: | --- |
| CARLA | 1.35× | 1.35× | 原流量已含 840 veh/h 和两条行人流；提高到 1134 veh/h，避免短任务起步即拥堵锁死 |
| Cross | 1.50× | 1.00× | 双汇入走廊增加 50%；复制车延后 8--10 秒发车，先保留 ego 在第 5 秒的原始插入条件，再施加并线压力 |
| Roundabout-C | 1.25× | 1.00× | 原场景本身已是高流量环岛且初始车辆多，控制增幅以免只测到入口插入拥堵 |

显式车辆按原列表均匀抽样复制，并使用协议冻结的场景化确定性发车偏移，避免同位置
同时间重叠。CARLA/环岛使用 0.8--3.0 秒抖动；Cross 使用 8--10 秒偏移，避免新增车
先于第 5 秒出发的 ego 抢占插入空间。`flow`/`personFlow` 只在 overlay 中追加差额
需求；原需求仍由原文件提供。Cross 与 Roundabout 的 endless traffic 逻辑保持不变。

## 运行

在 `pytorch_sb3_sumo` 目录下使用已验证的 `llm_pipeline` 环境：

```powershell
conda run -n llm_pipeline python tools/run_high_density_same_scene_v1.py plan `
  --profile comparison

conda run -n llm_pipeline python tools/run_high_density_same_scene_v1.py preflight `
  --device cuda --workers 1

conda run -n llm_pipeline python tools/run_high_density_same_scene_v1.py run `
  --profile comparison --device cuda --workers 1

conda run -n llm_pipeline python tools/run_high_density_same_scene_v1.py status `
  --profile comparison

conda run -n llm_pipeline python tools/run_high_density_same_scene_v1.py summarize `
  --profile comparison --require-complete
```

`comparison` 固定为 6 方法 × 3 场景 × 3 seed，共 54 次 50k raw-step 新训练和
1,620 个最终确定性评估 episode。`smoke` 仅用于工程检查，不能写入结果表。默认单
worker，避免多个 GPU 训练任务互相争抢显存；已有完整 run 会被跳过，已有不完整目录
不会被覆盖。

单个 job 也可独立运行：

```powershell
conda run -n llm_pipeline python tools/train_high_density_same_scene_v1.py `
  --method mst_slt --scenario carla --seed 0 --profile smoke `
  --output-dir results_high_density_same_scene_v1/manual `
  --model-name hdv1_manual_mst_slt_carla_seed0 --device cpu
```

## 结果合同

主指标是 Success ↑ 与 Collision ↓；Return ↑、Off-route ↓、Timeout ↓、成功完成
时间 ↓ 为次指标。每个场景先汇报三个训练 seed 的均值与总体标准差，再做三场景等权
宏平均。任何缺失或失败 job 都保持 `TBD/null`，不得用普通密度历史结果补格。

| Method | CARLA Success/Collision | Cross Success/Collision | Roundabout Success/Collision | Macro Success/Collision |
| --- | --- | --- | --- | --- |
| MST+SLT | TBD | TBD | TBD | TBD |
| TemporalGraph | TBD | TBD | TBD | TBD |
| Initial Full | TBD | TBD | TBD | TBD |
| Full+BalancedSlots | TBD | TBD | TBD | TBD |
| v4.1 | TBD | TBD | TBD | TBD |
| v4.8 | TBD | TBD | TBD | TBD |

这里尚未生成任何高密度实验数字。表中所有 `TBD` 只能由本协议的真实 run artifact
填写。
