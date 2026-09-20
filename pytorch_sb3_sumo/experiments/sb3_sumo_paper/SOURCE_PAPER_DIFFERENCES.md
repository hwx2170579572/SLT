# 源码、论文与 SUMO 迁移差异审计

本项目按用户要求以发布源码为实现依据。论文用于确定需要复现的实验规模、场景名称和
报告指标；当论文描述与源码实际执行路径冲突时，不把论文描述臆造成另一套算法。

## 已确认的论文—源码冲突

| 项目 | 论文描述 | 发布源码实际值/状态 | SUMO 迁移选择 |
|---|---|---|---|
| 训练步数 | 每个城市任务 100,000 步 | `configs/init_configs.py` 默认 1,000,000 步 | `paper` profile 用 100,000 步；另保留 `source_release` profile |
| PPO 预算终点 | 100,000 步 | 两个 on-policy runner 在外层检查预算、内层固定采满 512-step horizon，因而请求 100,000 时实际到 100,352 | 命令仍记录论文请求 100,000；按源码完成到 100,352，并在参数/结果中同时记录实际值 |
| 重复实验 | 5 个不同随机种子 | `n_experiments=1` | 正式论文 profile 使用种子 0--4 |
| 全局未来预测长度 | 3 | `future_steps=1` | 按源码使用 1 |
| 初始熵权重 | Table VII 为 $\alpha=1$ | 发布的自定义 `algos/sac.py` 默认 `alpha=0.2`，`auto_alpha=True` 时从 `log(0.2)` 开始 | 按源码使用 `auto_0.2` |
| 优化器 | 实现细节称神经网络统一使用 Adam | 发布的自定义 SAC 对 actor、critic、representation 使用 NAdam；仅温度参数使用 Adam（`beta_1=0.5`） | 按源码分别使用 NAdam 与 Adam |
| Double Merging 上限 | 400 步 | `tools/test.py` 使用 600 步 | 按源码使用 600 |
| Roundabout-C 上限 | 800 步 | `tools/test.py` 使用 1,000 步 | 按源码使用 1,000 |
| ego 初始位置 | 起始路线随机位置 | 发布的 `scenario.py` mission 固定 lane、offset 和延迟出发时间 | 按发布 mission 固定注入 |
| PPO 学习率 | 所有方法统一 1e-4 | `set_on_policy_configs()` 写死 5e-4 | 按源码使用 5e-4 |
| PPO 图像历史 | 与 DrQ 相同的连续时刻 stacked raster | SMARTS PPO runner 每步只读取当前一帧 `(80,80,3)` | 按源码使用单帧 |
| CARLA PPO 编码器 | 所有基线直接使用 MST | PPO 配置只启用不带候选路线输入的 `Ego_Neighbours_Encoder(6,10,5)` | 按源码训练路径使用 GRU/relation-attention，并明确不是完整 MST |
| CARLA 候选路线数 | Table VII 为 2 | `carla_env.py` 为每个 actor 生成 3 条 | Proposed/MST 按源码使用 3 条，并保留后续 `i*2` 索引异常 |
| 测试策略选择 | 论文称使用训练期成功率最高的 policy | 仓库明确没有 one-shot 训练入口；on-policy 没有 best saver，off-policy 保存语句被注释且条件实际是 mean return；`tools/test.py` 加载 latest checkpoint | 保存并测试每个 seed 的源码 latest checkpoint，报告跨 seed 统计；不冒充未发布的 best-policy 选择器 |
| PPO 测试调用 | 应进入测试阶段 | tf2rl 位置参数把 `test` 绑定到 `mask`，SMARTS 仍随机采样；CARLA 又把 actor-critic 的 3 个返回值解包为 2 个，按发布路径会报错 | 采用最小可运行意图修复：确定性均值动作、关闭评估 dropout，并将此列为非字面等价项 |
| 完成时间 | 只统计成功回合的均值和标准差 | 各 runner 返回的 step 计数口径不一致，已收集的 `full_step` 成功列表没有被使用 | 按论文明确指标从逐回合证据计算 `raw_steps*0.1` |
| DT 统一入口 | 作为论文基线报告 | `configs/init_configs.py` / `tools/test.py` 明确 `NotImplementedError`，数据收集脚本另行存在 | 不用名称相近的 SB3 实现冒充原 DT |

前两项属于实验统计口径，因而正式 `paper` profile 按论文执行；其余会改变模型或环境
实际行为的冲突均按源码执行。所有机器可读值同时保存在 `protocol.json`。

## 不能无歧义迁移的论文基线

- plain SAC：论文描述 LSTM 状态编码器；发布通用配置仍默认启用 hierarchical
  representation，没有唯一的一键配置可以证明就是表格中的 plain SAC。
- DrQ：论文描述堆叠 raster image；发布测试入口却把向量轨迹送入 CNN，并使用输入与
  crop 同尺寸的 random crop，源码路径不能形成无歧义、可运行的论文实现。
- DT：发布仓库含通用 Decision Transformer 文件和 500 条成功轨迹收集脚本，但主配置
  与主测试入口明确未集成，且没有发布训练数据或对应 checkpoint。
- RDM：发布实现把驾驶行为委托给 SMARTS rule-based controller；禁止安装或运行
  SMARTS 后，没有独立策略契约可直接移植到 TraCI。

因此 v9 冻结时进入正式运行且可审计的方法只有 `proposed`（MST + SLT + SAC）。v9
协议中的 `mst`/`ppo` 名称只是当时的待迁移矩阵项；该版本下的旧 `sac`/`ppo` 结果仍不
得进入正式论文汇总：

- MST 不只是“关掉 SLT 的原生 SB3 SAC”。发布源码仍使用相同的 raw-step 更新顺序、
  `alpha` 目标、NAdam、逐梯度张量裁剪和 held-action replay 时钟；现有原生 SB3
  `sac` 路径尚未完整保留这些行为。
- SMARTS PPO 在 `set_on_policy_configs()` 中声明 `(80,80,3)`，runner 每个 0.1 s
  原始步读取当前 `top_down_rgb.data/255` 并重新决策；CARLA PPO 切到向量
  `Ego_Neighbours_Encoder`，但并没有论文所称完整候选路线 MST 输入。
  因而统一向量编码且 `action_repeat=3` 的 SB3 PPO 只能作为迁移 smoke 对照，不能
  冒充论文 Table I/III 的 PPO。

未支持或尚未修正的项目会被报告为未忠实集成，不会生成伪造的对照结果。

## v12 的源码契约修正

以下修正已实现并通过 82 项全量测试；协议 v12 的最终 SHA-256 为
`9AC3560E286A5E9EB4943EBED643E6223ACE1998DFE5AB008EEA37CA5398982A`。v12 使用新的
`paper_source_audited_runs_v12_frozen` 结果根并在运行前逐字节冻结
`protocol.snapshot.json`；
已有不同快照时拒绝混写。正在运行的 v9 Proposed 与 v10 MST 任务不改参数、不重启，
但后续源码审计确认它们在 100k 非终止 partial held-action 的 replay 插入上多出一条
源码不会写入的 transition。v11 首批三个 20k checkpoint 的内部 raw 时钟又暴露为
20,002、20,000、20,002；原因是跨过阈值后延迟到下一 SB3 rollout 才保存。v9/v10/v11
因此都只列入 `diagnostic_predecessor_results`，不导入 v12 正式聚合：

- `mst` 现在复用 Proposed 的自定义源码时钟 SAC，只关闭 SLT；仍保留 raw-step
  更新顺序、`auto_0.2`、NAdam/Adam 分工、逐张量 clipnorm、四步 cpprb 聚合与
  critic-owned encoder。SLT 关闭时不会实例化辅助网络或消耗额外随机数，episode-end
  重复插入也随源码 `make_predictions=0` 关闭。
- 五个 SMARTS 场景的 PPO 使用 `RGB(80,80,32/80)`：80×80 单帧、32 m 正交视野、
  ego 对齐，并使用 v0.4.17 的道路/ego/社会车颜色。策略采用发布的四层 valid CNN
  (`16/64/128/256`, stride `3/2/2/2`) 与 global average pooling，而非向量 MST。
- PPO 的状态相关高斯方差、tanh 采样、采样时 Jacobian 修正、更新时却把已 squashed
  action 直接作为 Normal 样本且不做 Jacobian 修正的源码非对称行为均被保留；优势在
  完整 512 步 horizon 上一次性归一化，Adam `eps=1e-7`，不额外加入源码没有的梯度
  裁剪。SMARTS PPO 每个 0.1 s 原始步重新决策。
- PPO 更新路径中的 `compute_log_probs()`/`compute_entropy()` 对共享编码特征执行
  `stop_gradient`；因此 CNN/GRU 编码器只接受 value loss，actor loss 只更新 actor
  MLP。迁移版使用显式 `detach()` 保留该梯度边界。
- SMARTS on-policy runner 在第一个 episode 之后漏掉 reset frame 的 `/255`，但后续
  step frame 仍归一化；训练环境保留这个一次/回合的尺度 bug，独立测试环境始终按
  源码测试路径归一化。
- CARLA PPO 不使用 RGB。它读取 `(6,10,5)` 轨迹，经过共享 Keras 语义 GRU、6-head
  relation attention、共享两次调用的 LayerNorm、FFN 与 ego feature 拼接。源码设置
  `make_rotation=False`，因此实际把 `[x,y,heading,vx,vy]` 解释为
  `[x,y,heading,vx,wrap(vy)]`，该异常字段语义不作“修复”。环境每 0.1 s 保存一条
  transition，但 action/value/log-prob 连续复用 3 步；SUMO 包装器与 SB3 rollout
  分别保留 raw-step 和三步缓存，而不是压成一个三步 transition。
- 源码在 512 步非终止 horizon 末尾用最后一个 current-state（CARLA 还可能是缓存的）
  value 引导，而不是 next-state value；max-time episode 则以 0 结束。自定义 rollout
  同样保留这两个行为。
- 两个发布 on-policy runner 的预算判断都在固定 512-step horizon 外层，内层没有
  `max_steps` break；因此论文 profile 请求 100,000 时 PPO 实际采集 100,352 个原始步。
  迁移版沿用 SB3 完整 rollout 的同一越界语义，并在 `arguments.json` 中写明期望实际值。
- `tools/test.py` 会恢复最新周期 checkpoint。100k 位于最终 512-step horizon 内且在
  该 horizon 更新前保存，所以正式 `final_model` 是尚未包含最后 10 个 epoch 更新的
  100k checkpoint；采满 100,352 后的 learner 另存为 `post_horizon_model`。
- 发布 PPO 测试 API 自身不可同时按字面运行两个分支：SMARTS 的 `test=True` 错绑到
  `mask`，CARLA 的两返回值解包会报错。迁移版统一采用该参数显然意图表达的
  deterministic mean/eval-mode dropout。这是明确披露的最小修复，不宣称是字面执行结果。
- CARLA PPO 的训练结束评估与独立详细评估都按同一三原始步动作保持执行；PPO 的训练
  原始步数由 `num_timesteps` 验收（其环境每次 transition 恰为一个 SUMO 原始步），
  不再误用仅由源码 SAC 暴露的 `_raw_steps_seen`。
- Table III 没有 MST 测试行，但 Fig. 9 图注明确包含 CARLA 的 f 面板及橙色
  `Trans`（仅 MST）训练曲线，正文也说明 CARLA 基线采用 MST 编码。因此机器可读训练
  矩阵为 Proposed/MST/PPO 各 6 场景，共 18 个组合；MST-CARLA 没有虚构论文测试参考。
- v10 曾按 Table III 过窄地排除 MST-CARLA。发现 Fig. 9f 证据时 v10 尚无完成的正式
  模型，唯一正在运行的 MST-left_turn 继续训练作诊断；不删除或重启，但因上述末端
  replay 差异同样不导入 v12。
- 发布 off-policy runner 每个 raw step 检查预算；若 100k 落在三步动作中间，它仍做
  本 raw-step 更新，却只在完整 held boundary 或 episode event 时写 replay。v12 的
  replay buffer 因而丢弃“预算截断且非终止”的 partial transition，同时保留相应更新。
- 发布 runner 在每个 raw step 的更新之后立即检查 checkpoint 模数。v12 若发现 20k
  阈值落在三步 held action 内，会把旧 replay 上的中间更新精确分段，在阈值更新完成后
  立刻保存；若阈值正好位于 held boundary，则在写 replay 和该步更新后保存。每个正式
  任务结束时还会从所有 SB3 ZIP 的 `data` 元数据复核时钟，并把 CRC/SHA-256 写入
  `checkpoint_audit.json`；缺失或超调的任务不能通过编排器验收。

RGB 是依据作者发布 SUMO lane geometry 与 SMARTS 颜色/相机参数直接光栅化，未安装或
导入 SMARTS/Panda3D。道路资产、视野、车辆中心/尺寸与朝向有明确对应，但 Panda3D 的
GLB 三角形光栅边缘和本地 PIL 像素覆盖不可能逐像素相同，属于跨渲染器限制而非隐藏的
“完全一致”声明。

## SUMO 等价性的边界

1. 五个 SMARTS 任务直接加载作者发布的原始 `map.net.xml` 和全部
   `traffic/*.rou.xml`；ego route、lane、offset、depart time 按发布 `scenario.py`
   转录。替换的是 SMARTS 的 Bullet/车辆控制器、传感器与 provider 调度，而不是地图或
   背景交通资产。
   原 SMARTS ego 在 SUMO 端是 route-less 外部车辆，并由 Bullet/LaneFollowingController
   每帧 `moveToXY`；迁移版为了让 SUMO 承担车辆运动，保留 mission route 并通过 TraCI
   目标速度/换道控制。背景车是否能预知 ego 后续 route 因而可能不同，这是跨控制器
   迁移仍无法逐帧等同的一项明确限制。
2. SMARTS v0.4.17 首次场景遍历使用 seed 42 对词典序 traffic route 列表做
   `np.roll`，迁移版精确复现首轮顺序。原引擎在以后每轮重新滚动，并让同一个全局
   Python 随机流同时供 SUMO seed、控制器和 endless-traffic 选车道使用；不运行
   SMARTS 无法逐调用复刻该随机流。迁移版采用覆盖全部 route 的确定性重复循环，保证
   分布和可复现性，但后续 episode 的 route 顺序不宣称逐条相同。
3. 原资产由旧版 SUMO 工具链生成，正式迁移运行时为 SUMO 1.25.0。网络和 route 文件
   可直接加载，但车辆动力学与换道器的版本差异意味着轨迹不能逐帧相同。
4. CARLA 没有发布可直接转换的 Town-10 SUMO 地图/物理资产。迁移版复用
   `carla_env.py`、`wp.npy`、`wp2.npy` 中的路线、成功区域、观测顺序、目标车道、车辆
   和行人交互要素重建任务；不能声称复现 CARLA 物理或传感器时序。
5. TensorFlow/Keras 与 PyTorch/SB3 的计算图、CUDA kernel 和随机数流不同。迁移版复现
   源码网络结构、有限值 attention mask、初始化 fan 规则、优化器顺序、NAdam、梯度
   裁剪、replay 与 raw-step 更新时钟，但目标是行为和统计等价，不是权重逐 bit 相同。
6. SMARTS 把 ego 作为外部 provider 控制的车辆，并在 SUMO 同步端明确使用
   `speedMode=0b00000`。迁移版同样关闭 SUMO 安全速度、路权和显式换道安全覆盖；否则
   会凭空增加一个源码中不存在的规则安全控制器。早于该修正启动的 v3 运行仅保留作
   诊断。
7. SMARTS 碰撞事件来自 Bullet chassis contact points，并为漏检额外查询 AABB 与
   0.05 m 邻近接触；SUMO 在路口几何重叠时仍可能不给出 TraCI collision。迁移版因此
   按 SMARTS 车型尺寸把前保险杠坐标还原为中心坐标，再执行二维有向包围盒接触检测。
   仅依赖 SUMO collision 的 v4 运行同样只作诊断；正式结果从
   后续新目录重新开始。
8. SMARTS 创建外部 ego 时还设置 `tau=4`、`decel=6` 和 passenger chassis
   `3.68×1.47×1.4 m`。只修正几何事件但未把这些值同步给 SUMO 的 v5 运行不进入
   汇总。
9. 发布适配器取 `waypoint_paths[:2]`；MissionPlanner 从所选 edge 的全部车道生成
   路径并按起始 `lane_index` 稳定排序。进一步只读核对 v0.4.17 的
   `LanePoints.paths_starting_at_lanepoint()` 后确认：固定 route 过滤会拒绝不在 route
   列表中的 immediate internal edge，源码旁也留有 `What about internal lanes?`
   TODO，因此 ego 的合法转弯候选实际会在路口截断。动态附着到社会车辆的传感器
   使用 `mission=None`/`EndlessMission`，在普通 edge 上不受 route 约束并会沿真实
   internal lane 分支。v6 的虚构直线连接与 v7 的“修复后完整转弯曲线”都不是作者
   实际执行语义，均以 `NOT_FORMAL_SOURCE_FIDELITY.md` 排除；该路口过滤语义在 v8
   完成修正。
10. 发布适配器仅在 `len(res)==1` 时复制唯一候选；已有两条候选时，即使第二条在
    去掉当前 waypoint 后只剩全零 padding，也不会复制第一条。v8 错把“第二张量
    全零”等同于“只有一条候选”，因此被排除；按候选列表长度判断的正式运行从 v9
    开始。

## 正式结果的验收口径

每个 `paper` profile 正式任务必须同时满足：Proposed/MST 恰好 100,000 个 SUMO 原始
步；PPO 请求 100,000、按源码采到 100,352 且测试 100,000 checkpoint；种子记录、每
20,000 原始步 checkpoint、最终 `final_model.zip`、50-episode
`final_evaluation.json`，以及原始 111 个文件的 SHA-256 不变。中断目录保留为
`partial`，诊断/短跑目录不计入论文汇总。

最终论文级汇总还必须区分两种步数口径：`mean_raw_steps` 是全部测试回合的诊断均值，
不能冒充 Table II/III 的完成时间。后者只在成功回合上统计，并按源码/场景的 0.1 s
仿真间隔计算 `completion_time_s = raw_steps * 0.1` 的均值与标准差；需要保留 50 个
逐回合事件明细以便复算。跨种子时间汇总同时报告“seed 均值的离散度”和按每个 seed
成功数/均值/总体方差通过总方差公式合并的全部成功回合均值与总体标准差；论文没有
说明其标准差 ddof，因而不假称两者统计定义完全相同。Table II/III 的机器可读参考值
独立保存在 `paper_reference_tables.json`，避免为纯报告数据改写冻结训练协议。

Fig. 9 训练曲线按源码的 `sum(success_log[-20:]) / 20` 计算；不足 20 个已完成回合时
仍固定除以 20，因此早期缺失历史等价为 0，再重采样到每 200 raw steps。
发布源码没有绘图脚本、论文也没有说明 EMA 与跨种子聚合的先后，因此迁移版同时保留
未平滑值，并明确采用：每个 seed 先以前向保持方式重采样，再执行
`ema_t=0.99*ema_(t-1)+0.01*x_t`，最后跨 seed 计算均值与总体标准差；首个 episode
完成前的网格点不虚构数值。

论文表格的 2% 粒度和文字更像是从五次训练中选一份 policy 后仅测试 50 回合，而不是
把 5×50 回合合并。发布仓库没有实现或保存这份“训练成功率最高 policy”，因此正式
SUMO 输出保留五个 seed 各自的 50 回合结果，并同时报告 seed 均值与 pooled 指标；这些
是可复算的稳健统计，但不声称复原了论文未发布的挑选过程。
