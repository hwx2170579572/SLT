# Full、MST+SLT 与 SAC+MLP 的实现地图

这是一份基于实际代码的研究记录。研究角色由用户确认：full 是完整研究方法，MST+SLT 是强基线，SAC+MLP 是纯强化学习基线；当前从 SAC+MLP 逐项加入 full 的模块，定位效果及问题。

入口：[研究上下文](../RESEARCH_CONTEXT.md)。结果与运行协议：[历史实验](experiment_history.md)。
本记录区分代码支持的配置、实际训练入口启用的配置，以及已经留下正式评估的运行。类名、目录名和注释均不能单独证明后两者。

## 1. 实际代码边界

`fast-developer` 保存实验入口、环境包装器和产物；完整模型在上一级 `pytorch_sb3_sumo/algos/` 中。

| 职责 | 源文件 |
| --- | --- |
| D1 方法矩阵、父方法、模型构建、实际 loss 系数 | [train_intersection_yield_v2_d1.py](../train_intersection_yield_v2_d1.py)，重点约 82–113、151–254 行 |
| yield_v2 基线工厂和训练参数 | [train_intersection_yield_v2.py](../train_intersection_yield_v2.py) |
| MST+SLT 算法注册与模型配置 | [sb3_configs.py](../../configs/sb3_configs.py) |
| 公共 runner 与 raw budget 转接 | [train_sb3.py](../../tools/train_sb3.py)、[callbacks.py](../../algos/sb3_torch/callbacks.py) |
| 修正后的历史训练入口 | [train_intersection_hold35k_mst_fixed.py](../train_intersection_hold35k_mst_fixed.py) |
| 续训协议 | [train_intersection_yield_v2_d1_continue.py](../train_intersection_yield_v2_d1_continue.py) |
| 纯 MLP 特征基线 | [simple_encoder.py](../../algos/hybrid_action/hsac/simple_encoder.py) |
| MST 及公共 attention / polyline 模块 | [features.py](../../algos/sb3_torch/features.py) |
| D1 增量开关和路线变体 | [incremental_topo_encoder.py](../../algos/sb3_torch/incremental_topo_encoder.py) |
| D1 可学习交互边权 | [incremental_topo_encoder_attn.py](../../algos/sb3_torch/incremental_topo_encoder_attn.py) |
| full 的实际父类实现 | [topo_temporal_features_v2.py](../../algos/sb3_torch/topo_temporal_features_v2.py)、[topo_temporal_features.py](../../algos/sb3_torch/topo_temporal_features.py) |
| 拓扑图数据构造 | [topology_graph_v2.py](../../envs/sumo/topology_graph_v2.py) |
| 拓扑公共数据与场景容量 | [topology_graph.py](../../envs/sumo/topology_graph.py)、[paper_scenario_registry.py](../../envs/sumo/paper_scenario_registry.py) |
| SAC 训练循环和表示目标调度 | [sac.py](../../algos/sb3_torch/sac.py) |
| MST 的单向量 SLT 预测目标 | [representation.py](../../algos/sb3_torch/representation.py) |
| n-step 回报与 one-step 表示目标采样 | [replay_buffer.py](../../algos/sb3_torch/replay_buffer.py) |
| Graph-SLT 表示目标 | [graph_representation.py](../../algos/sb3_torch/graph_representation.py) |
| SBS 扩展 | [sac_v2.py](../../algos/sb3_torch/sac_v2.py) |
| 连续动作策略、共享 encoder 和 target critic | [policies.py](../../algos/sb3_torch/policies.py) |
| 混合动作策略分支 | [hybrid_policy_v4.py](../../algos/sb3_torch/hybrid_policy_v4.py) |

行号只是本次阅读的定位提示；文件变更后应重新查符号。

## 2. 方法身份与 D1 模块矩阵

- `sac_mlp_d1_full` 的 parent 是 `sac_mlp`，使用连续 SAC 父设置。
- `hsac_mlp_base_d1_full` 的 parent 是 `hsac_mlp_base`，属于混合动作分支。
- `hold35k` 是历史方法标识，使用 TASAC v4_8 混合动作通道，并调用 `tools.v48_stability_v4.model.StabilitySAC`；不能直接替代上述两个 D1 full 的运行身份。
- `IncrementalTopoEncoder` 继承 `TopoTemporalGraphExtractorV2`；全开配置实际进入父类完整路径，不能只读增量外壳就认为已经掌握 full。

增量类注释中的“full configuration == hold35k encoder”限定在编码器路径。它不意味着相同 policy、算法、训练目标或实验身份；D1 full 新建的是 `SceneRepresentationSACV2`。

| D1 配置 | route token | topology | structured slots | Graph-SLT / SBS | 关键含义 |
| --- | --- | --- | --- | --- | --- |
| ST | 关 | 关 | 关 | 关 / 关 | 时空主体；仍有 MLP map fallback |
| ST-Attn | 关 | 关 | 关 | 关 / 关 | 无拓扑车辆图的边权改为可学习权重 |
| ST-RT | 开 | 关 | 关 | 关 / 关 | 路线信息加入表示交互 |
| ST-RT-Topo | 开 | 开 | 关 | 关 / 关 | 增加拓扑信息 |
| ST-RT-Topo-3slot | 开 | 开 | 开 | 关 / 关 | 同一增量主干只切换到 32/64/32 三槽输出；Graph-SLT/SBS 均禁用 |
| Full | 开 | 开 | 开 | 开 / 开 | 同时增加三槽投影和辅助优化目标 |

full 实际构造使用 `SceneRepresentationSACV2`、`structured_representation=True`、`representation_coef=1.0`、`slot_balance_coef=0.01`。
无 slots 的 D1 变体将 `representation_coef=0`。因此 `Topo -> Full` 不是只改变 slots 的单变量实验。

新增的 `sac_mlp_d1_st_rt_topo_3slot` 配置用 `use_slots=True` 保留槽输出，但显式设置 `use_incremental_slots=True`、`use_graph_slt=False`、`use_sbs=False`、`representation_coef=0`、`slot_balance_coef=0`，由普通 `SceneRepresentationSAC` 训练。此开关使其继续执行与无 slot Topo 相同的增量 route/topology/interaction 上下文路径，而不是默认 full 的父 V2 前向。它将 Topo 的 `mlp_output` 拼接头换成三个投影头，输出宽度 32/64/32，总特征仍为 128。故它隔离了**输出结构**相对辅助学习目标的影响，但不等于参数量/有效容量匹配；当前尚无可报告的正式训练结果。

路线接入诊断分支：

- `gate`：由状态池化和 route context 计算 sigmoid 门，控制路线残差注入。
- `late`：路线不注入车辆交互节点，在后端保留路线表示。
- `ego_only`：只向 ego 节点注入路线。
- `cond_edge`：不修改节点表示，用 ego 路线和边特征修正 ego 目标行的交互权重。

这些分支的默认与优先级以 `IncrementalTopoEncoder` 的实际分支判断为准。不能仅根据后缀认为它们增加了不同数量的独立模块。

## 3. 观测与表示尺寸

在本次核对的 intersection 默认设置下：

- 车辆数为 ego + 5 辆邻车，共 6 辆；历史长度 10；每个状态 5 维。
- `trajectory` 为 `[6, 10, 5]`，运行时增加 batch 维。
- 每车 2 条 map 候选路径，因此 `map` 为 `[12, 10, 5]`，运行时增加 batch 维。
- 连续 SAC 父分支 action 为 2 维；混合动作分支须按其编码契约单独解释。
- 拓扑环境配置上限为 64 个节点、512 条边；每个 lane 包含 10 个二维采样点和 8 个属性。这是容量上限，不是每帧都有这么多有效节点/边。
- full / D1 编码隐藏宽为 128、attention heads 为 2；结构化表示为 ego/social/route 三槽，宽度分别 32/64/32，总宽 128。

地图候选数和环境配置相关，不能把以上尺寸当成所有场景的全局常量。当前主要 SAC/D1 比较使用 base 环境契约；TTC 选邻车属于另一历史分支，不能自动归入本组结果。

### 3.1 SAC+MLP：实际是 masked-mean MLP，不使用循环网络

类名虽为 `SimpleMlpLstmExtractor`，实际控制参数是 `backbone`。当前调用显式设置 `backbone="mlp"`；仅 `backbone="lstm"` 分支才构建 GRU，本组配置不进入该分支。

前向顺序（`simple_encoder.py` 约 49–99、115–160 行）：

1. 复用 ego-frame 坐标预处理；状态 5 维经 `5 -> 128 -> 128` 两层 ReLU MLP。
2. 每个 actor 的历史在有效时间位置上 masked mean，得到每车 128 维特征。
3. map 点经独立的 `5 -> 128 -> 128` MLP；对有效 polyline 点 masked mean，再对每车候选路径求均值。
4. 取 actor 0 为 ego；邻车特征 masked mean 得 social；取 ego 的地图特征为 route。
5. 拼接 `[ego, social, route]` 为 384 维，经 `384 -> 128 -> 128` ReLU MLP 输出。

因此“纯强化学习基线”指没有 SLT/Graph-SLT/SBS 辅助目标；它仍读取历史和地图，也不是把全部输入简单 flatten 后接线性层。它不执行 MST 的时序、邻车、地图或 goal attention。

### 3.2 MST：HierarchicalSceneExtractor 的完整层级路径

来源：`features.py` 中 `HierarchicalSceneExtractor`（约 193–535 行）。

1. 对轨迹做 ego-relative 预处理，使用项目坐标旋转 helper；每车历史经 2-head temporal self-attention 与 max pooling。
2. 每条候选 map polyline 经点级 self-attention 与 max pooling；首点的 flags 另经 64 维 MLP，再组合成 128 维路径表示。
3. 每辆邻车以自身为 query，与其自身及候选路线表示进行 cross-attention，形成带道路上下文的邻车特征。
4. ego 与 ego / 相关邻车特征做 cross-attention，形成 interaction representation。
5. 以 ego interaction 表示查询 ego 候选路线，形成 goal representation。
6. goal 与 interaction 经过输出头，得到 128 维 scene embedding。

这条 MST 路径不是 full 的 V2 拓扑图消息传递路径。MST+SLT 对此 embedding 使用下一决策步的表示预测辅助目标，详见第 5 节。

### 3.3 Full：TopoTemporalGraphExtractorV2 的完整路径

`IncrementalTopoEncoder` 固定继承 `TopoTemporalGraphExtractorV2(variant="soft")`。
当 route / topology / slots 均为 True 时，`forward_tokens` 直接调用父 V2 路径（增量文件约 207–211 行），核心前向位于 `topo_temporal_features_v2.py` 约 612–733 行：

1. 编码车辆状态，并把 route context 注入车辆节点。
2. lane 的二维点经 `2 -> 64 -> 128` 编码与 max pooling，并融合 8 维 lane 属性。静态 lane 图经两层、六种关系的 relational graph 层；V2 的第六种关系为 MERGE。
3. 每个 actor/时刻查询 lane tokens。soft 分支组合空间距离、路线距离、方向一致性形成 attention bias，选方向兼容的 top-8 lane；无兼容 lane 时回退到最近有效 lane。拓扑 context 经残差融合进入 route-conditioned 车辆表示。
4. 根据拓扑注意力构造车辆边：相对位置 2 维、相对速度 2 维、航向余弦、TTC、same-lane、conflict-or-merge，共 8 维；以距离高斯权重执行两层车辆图消息传递。
5. 对图交互后的历史序列做带 lag embedding 的时间 self-attention，再以当前时刻 query 池化。
6. 以 ego 为 query，对有效 actor 特征（包含 ego 本身及有效邻车）做 social attention；另对 ego 候选路线做池化。
7. 拓扑 goal 分支形成补充上下文，以 delta / LayerNorm 方式与路线表示融合。
8. 分别输出 ego / social / route 三槽，维度 32 / 64 / 32；拼接后提供给策略和价值网络。

`ST-RT-Topo` 与 full 的主干顺序并无已核实的结构性换序。前者 slots=False，走增量前向，将三个 128 维上下文拼为 384 维，再用 `mlp_output: 384 -> 128 -> 128` ReLU 输出；full 进入父 V2 路径，改为结构化的 32/64/32 输出头，并启用辅助目标。
所以这里的差异是**策略可见表示的输出头和辅助训练目标**，不是仅给同一个向量重命名，也不能无证据地称为“模块顺序发生变化”。

`ST-RT-Topo-3slot` 是后续加入的中间控制：`use_incremental_slots=True` 阻止全开开关默认触发父 V2 路径；前面的增量 route、topology、车辆交互、temporal/social 与 goal 上下文构建保持不变，只把最终 384→128→128 MLP 头替换成 ego/social/route 三槽投影，拼接后仍为 128 维。一个同权重/同输入测试确认两个路径在输出头之前的上下文完全相等。它的 Graph-SLT/SBS 系数为零，algorithm class 为普通 `SceneRepresentationSAC`；旧 Full 配置和全开父类路径仍保留原行为。源码位置见 `IncrementalTopoEncoder.forward_tokens` / `_forward_incremental` 与 D1 builder。

`IncrementalTopoEncoderAttn` 只替换无拓扑分支的固定距离边权：相对位置、速度、航向、TTC、同车道/冲突等 8 维边特征进入 MLP，再沿 source 轴 masked softmax；它不替换有拓扑的 V2 边特征路径。

## 4. 共享参数与实际梯度路径

连续 SAC 路径由 `SceneRepSACPolicy` 构造：

1. 创建一个 online extractor，并同时注入 actor 与 online critic。因此二者共享 encoder 参数对象。
2. `DetachedSceneActor` 对抽取特征执行 `.detach()`，actor optimizer 还显式排除 extractor 参数。
3. critic optimizer 包含 critic 的全部参数，其中包括共享 extractor。
4. target critic 使用独立 extractor，从 online critic 初始化，并在 target update 时随整个 critic 做 Polyak 更新。
5. actor 更新阶段临时冻结 critic 参数，但仍允许 Q 对动作的导数训练 actor；actor loss 不直接更新共享 encoder。

来源：`policies.py` 约 24、81–128 行，`sac.py` 约 116–154、376–378 行及训练循环。

| 方法 | 编码器的更新来源 | actor loss 直传 encoder | target encoder |
| --- | --- | --- | --- |
| SAC+MLP / 普通 D1 | critic TD loss | 否 | 独立副本，Polyak 更新 |
| D1 ST-RT-Topo / ST-RT-Topo-3slot | critic TD loss | 否 | 独立副本，Polyak 更新 |
| MST+SLT | 表示辅助 optimizer，随后 critic TD optimizer | 否 | 独立副本，Polyak 更新 |
| SAC 分支 D1 full | Graph-SLT/SBS 表示 optimizer，随后 critic TD optimizer | 否 | 独立副本，Polyak 更新 |

`representation_coef=0` 只关闭表示辅助 optimizer，不会冻结 encoder。辅助目标启用时，representation optimizer 持有 critic extractor 与表示预测模块参数；critic optimizer 也持有同一 online extractor。它们是针对同一 encoder 的两次顺序更新，不应简写成“一个统一 loss 的一次反传”。这本身是实现事实，不构成错误判定。

MST+SLT 与 SAC+MLP 的公共 SAC 实现使用双 Q、自动熵系数、NAdam actor/critic optimizer，以及 tau=0.005 的 target 更新。当前入口的基本设置为学习率 1e-4、batch 32、buffer 20k、gamma 0.99、训练 seed 0、action_repeat=3、raw warmup=5k；n-step replay 的 `n_steps=4`。
四决策步的 SAC 回报目标与一步的表示预测目标分开维护。具体运行仍应读其 arguments，不能把这些值视为所有旧实验的默认事实。

## 5. SLT、Graph-SLT、SBS 与目标编码器

### 5.1 MST+SLT 的 FutureRepresentationObjective

MST+SLT 使用 `FutureRepresentationObjective`，不是 full 的 Graph-SLT 三槽目标。其配置由 `configs/sb3_configs.py` 的 `scene_rep` 分支指定，`representation_coef=1`。

- 当前 128 维 scene embedding 与 replay action 的嵌入一起送入预测模块；连续 2 维动作先经 `2 -> 64` MLP。
- 预测路径含长度为 1 的 token 的 MHA / transition 模块，然后是 `128 -> 256 -> 128 -> ReLU` projector 和线性 predictor。
- 目标是下一次决策的 scene embedding 经过同一 projector，目标路径 `no_grad`。
- 当前交叉口配置使用 online critic extractor 作为目标特征编码器，目标侧停止梯度；projector 共享。
- loss 是乘样本有效 mask 后的 batch 平均负余弦相似度：`L_SLT = mean_b[-m_b * cos(pred_b, target_b)]`。

实际目标优先取 replay 的 `one_step_next_observations`；不是把 RL 的 n-step 聚合终点直接当成 SLT 目标。这里一步是一个 held-action 决策，通常最多跨 3 个 raw SUMO ticks。
来源：`representation.py` 的 `FutureRepresentationObjective`（约 14 行起）、`sac.py` 约 168–232 行、`replay_buffer.py` 约 214–258 行、`configs/sb3_configs.py` 约 95–143 行。

### 5.2 Full 的 Graph-SLT

令 `z = concat(z_ego, z_social, z_route)`，每个槽分别学习动作条件残差预测器 `F_i` 与 projector `P_i`：

```text
z_hat_i = z_i + F_i(concat(z, E_action(a)))
L_i = mean_b[ m_b * (1 - cosine(P_i(z_hat_i), stopgrad(P_i(z_next_i)))) ]
L_GraphSLT = (L_ego + L_social + L_route) / 3
```

对应实现为 `graph_representation.py` 约 86–143 行。每个 slot 有自己的预测路径，但 target 侧复用该 slot 的 projector 并停止梯度。

当前 D1 intersection 配置明确：

- `representation_online_target_encoder = (SCENARIO != "cross")`，因此当前取 True。
- `representation_separate_target_projector = False`。
- online 当前状态和 target 下一状态均调用 online critic extractor 的 `forward_tokens`；下一状态前向在 `no_grad` 内。

**这里的辅助 target 不是 SAC 的 target critic。** SAC Bellman target 仍使用独立的 `critic_target`；Graph-SLT 当前配置使用同一个 online encoder 的 stop-gradient 下一状态表示。迁移到 `cross` 时入口条件不同，不能照抄 intersection 的 target 设置。

目标优先使用 `one_step_next_observations`，否则回退到 `next_observations`。动作是 replay 中按父动作契约存储的动作；连续与混合父方法需要分别解释其动作编码。

### 5.3 mask 的实际定义

`m_b = _nonzero_mask(next_observation["trajectory"][:, 0, 0])`，其中 `_nonzero_mask(values)` 的规则是 `values[..., 0] != 0`（`features.py` 约 44–49 行）。

它检查该状态首坐标 x 是否非零，**不读取 dones 或 timeouts，也不显式剔除所有 terminal transition**。若终止 next observation 仍非零，它仍可能进入表示 loss；反之，有效状态若 x 恰为 0，也可能被 mask。
无效样本的 loss 乘零后仍按整个 batch 求均值，没有改成按有效样本数归一化。这些是静态实现事实；是否影响性能需要实验，不在本记录中直接认定为失败原因。

### 5.4 Full 的 SBS

`soft_slot_balance_loss` 位于 `topo_temporal_features_v2.py` 约 48–64 行：

```text
r_i = sqrt(mean_over_entire_batch_and_slot_dimensions(z_i ** 2) + eps)
L_SBS = population_variance_over_three_slots(log(r_i + eps))
L_rep_full = 1.0 * (L_GraphSLT + 0.01 * L_SBS)
```

SBS 的 `mean()` 不指定维度，先对整个 batch 和该槽全部元素求一个 RMS，再对三个 log-RMS 求总体方差；不是每样本分别计算后再平均。SBS 作用于当前 online 三槽，不约束 target 槽。

`SceneRepresentationSACV2` 默认 `slot_balance_coef=0`，但 D1 full 调用处实际传 0.01。表示 optimizer 独立更新 online encoder 与 Graph-SLT 模块，然后训练循环继续 critic TD 更新；不能把这里的两次 encoder 更新描述成 actor、critic、辅助 loss 一次性共同反传。

MST 的 `-cos` 与 Graph-SLT 的 `1-cos` 数值原点不同，且一个预测整体 embedding、另一个平均三槽；日志中的 loss 数值不能直接横向当成任务性能比较。

## 6. raw steps、决策步和更新频率

当前训练启用 `source_raw_step_control`。`tools/train_sb3.py` 虽将 raw budget 传给 `learn(total_timesteps=...)`，真正的 raw 步统计与停止由 `callbacks.py` 根据每次环境返回的 `raw_steps_executed` 控制，且训练开始时向环境传入 raw budget。

`sac.py` 的 raw-step interval 逻辑按每个达到 warmup 的 raw tick 安排 gradient update；held action 末尾前的更新使用已有 replay，当前 transition 入库后再进行 pending 的末尾更新。因此：

- 50,000 raw steps / action_repeat=3 对应约 16,667 次环境决策，而不是 50,000 次 held-action 决策。
- warmup=5,000 后记录 updates=45,001，与从 raw tick 5,000 到 50,000 的更新计数相符。
- 不能根据 updates 字段把预算误判为 45,001 个环境决策。
- 100k 续训没有恢复 replay buffer，warmup 改为 500 raw steps；详见实验历史，不能视作原训练无缝延长。

定位：`tools/train_sb3.py` 约 387–402、627–629 行，`algos/sb3_torch/callbacks.py` 约 50–124 行，`sac.py` 约 234–237、439–506 行。

## 7. 代码支持与实验完成状态

`sac_mlp_d1_full__intersection_yield_v2` 存在模型和记录 50k raw steps 的 `training_complete.json`，但本次刷新未发现该运行的 `evaluation_results.json` 或正式 final evaluation。
`hsac_mlp_base_d1_full__intersection_yield_v2` 本次只见进度、monitor 和检查点，未见 `training_complete`。
不能将历史 `hold35k` 的评估数字转记为这两个 D1 full 的结果，也不能因完整网络可实例化便称其性能已验证。

## 8. 归因前需要保留的边界

- ST 仍读取 map fallback；它验证的是一种时空表示及其地图接入方式，不是“完全无地图”。
- Full 的 slots、Graph-SLT、SBS 历史上一起开启，不能从旧的 Topo/Full 相邻比较中分离三者贡献；新增 ST-RT-Topo-3slot 只提供输出结构与辅助目标的拆分，不消除输出头参数量差异。
- 强基线 MST+SLT 同时具有编码结构和自监督表示目标，不能把它与纯 TD 更新变体的差异全归因于结构。
- SAC 主分支与混合动作分支应分别比较，不能仅凭共享部分 encoder 合并结果。
- 表示目标的 target、detach、预测步长和 replay 数据选择属于方法的一部分，后续修改时应和网络模块一起记录。
- 本轮进行静态实现核对和历史结果归档，不把未运行的模块组合、稳定性或性能写成已验证结论。

## 9. 本轮 SAC+MLP 与 ST-RT 对比中的 RT 和梯度路径

本节对应 2026-09-29 的 `sac_mlp__intersection_sorted_depart4p0` 与 `sac_mlp_d1_st_rt__intersection_sorted_depart4p0`，主结果见 [`mlp_strt_100k_analysis.md`](mlp_strt_100k_analysis.md)。

- `RT` 在此配置中指 route-token 路线表示接入，不是 route prediction/route auxiliary loss。SAC+MLP 与 ST-RT 都收到 `map` 输入：SAC+MLP 用 map-point MLP 和 masked pooling 得到路线特征；ST-RT 的 D1 encoder 打开 route-token 编码及其与车辆状态表示的交互。故不能把这组对比描述为“无路线输入 vs 有路线输入”。
- 本轮 ST-RT 的 `use_route=true`、`use_topology=false`、`use_slots=false`、`representation_coef=0`。因此它不运行 MST-SLT、Graph-SLT 或 SBS，也没有 RT 专属辅助目标。可训练的 route-token 分支由 SAC critic TD 目标端到端训练。
- 两种方法共享本轮环境、观测契约、动作头、奖励和训练预算；主要方法差异在特征提取器。策略与在线 critic 共享 online extractor，但 `DetachedSceneActor` 对 extractor 输出 detach，actor optimizer 也排除 extractor 参数；critic optimizer 包含该共享 extractor。因此本轮 online encoder（包括 RT 分支）只从 critic TD loss 获得梯度，不从 actor loss 直接更新。Bellman target 使用独立的 target critic/extractor，在目标计算时无梯度并按 Polyak 规则更新。
- 本记录不声称两种特征提取器参数量或有效容量匹配；该比较改变了 backbone，参数量对照在本轮文档中记为**未知**，不能据名称把结果归因为单独的 route-token 变量。

## 10. D1 Topo / Topo+3slot 激活与梯度诊断（2026-09-30）

本节记录已实现的常规采样字段和统计口径；不代表两种新方法的训练已完成。配置/阶段目标见 [`topo_3slot_protocol.md`](topo_3slot_protocol.md)。配置项位于 `fast-developer/train_intersection_yield_v2_d1.py`；编码器诊断位于 [`incremental_topo_encoder.py`](../../algos/sb3_torch/incremental_topo_encoder.py)，实际 lane-relation edge 特征诊断位于 [`topo_temporal_features_v2.py`](../../algos/sb3_torch/topo_temporal_features_v2.py)，critic 梯度采样位于 [`sac.py`](../../algos/sb3_torch/sac.py)。

启用诊断不增加前向、反向、优化器步骤或 RNG 调用。训练阶段的 `source=train_replay` 是采样配置标签，并不证明每条激活快照来自 replay batch：online extractor 也由 rollout actor 共享。训练启用后首个 forward 采样，随后每 256 次 encoder forward 采样；event type、`diagnostic_sample_index`、`diagnostic_sample_forward_index`、sample age、样本 batch 与 `grad_enabled` 用来判断快照新鲜度和实际上下文，训练激活不绑定为当前环境 episode。评估在现有 policy prediction 之后、环境 step 之前取样，每次 prediction 都采。SAC 依据单调 sample index 去重，因此读取到较旧快照时仍保留其年龄信息。`diagnostic_sampled` 仅表示最近一次 forward 是否采样，而不是激活事件的新鲜度标志。配置来源分别标为 `train_replay` 和 `eval_policy`。

| 诊断字段 | 统计对象与分母 | 有效性边界 |
| --- | --- | --- |
| `topology_attention_entropy`, `topology_effective_lanes` | lane attention 熵及其指数；对 `trajectory_valid` 的 actor×time query 求平均 | 无有效 query 时 count 为 0；和同批的 `diagnostic_valid_query_count` 一起解释 |
| `route_compatible_attention_mass`, `route_compatible_lane_count_mean` | 兼容 lane attention 质量、兼容 lane 数；以所有有效 actor×time query 平均 | fallback/无兼容 lane 的有效 query 仍在分母内，兼容质量可为 0；`topology_compatible_query_count` 单独给出存在兼容 lane 的 query 数 |
| `topology_fallback_rate`, `topology_fallback_query_count`, `topology_valid_lane_count` | fallback query / 有效 query；静态有效 lane 数 | fallback 率的分母为 `diagnostic_valid_query_count`；有效 lane 数是图配置计数，不是 per-frame 被选 lane 数 |
| `topology_conflict_directed_edge_count`, `topology_merge_directed_edge_count`, `*_relation_present` | 静态 lane graph 中 conflict/merge 有向边计数及存在标记 | 标记关系结构是否可用，不单独证明策略利用了关系 |
| `relation_pair_valid_count`, `relation_pair_candidate_count`, `relation_pair_valid_fraction` | 两 actor 均有效且 actor 索引不同的 pair；候选数为 `B×H×N×(N−1)`，有效数由现有 `pair_valid` mask 求和 | 同步保存 pair mask 是否有有效项；零有效 pair 时 pair mean 只可结合该标记解释 |
| `same_lane_attention_pair_mean`, `conflict_attention_pair_mean`, `merge_attention_pair_mean`, `conflict_or_merge_attention_pair_mean` | 每个有效 actor pair 的 lane attention 分布诱导关系分数，`alpha_i^T A alpha_j`；按上述 valid-pair 数平均 | 这是拓扑关系分数，不是车辆 attention 权重；same-lane/conflict/merge 关系通过连续 edge feature 进入 VehicleGraphLayer |
| `same_lane_edge_feature_mean`, `conflict_or_merge_edge_feature_mean` | 上述关系分数乘 `topology_residual_scale` 后的有效 pair 均值 | 反映实际注入车辆边特征通道的尺度；不是单独的因果贡献量 |
| `topology_residual_scale`, `goal_residual_scale`, `route_bias_weight`, `heading_bias_weight` | 当前 forward 使用的可训练标量 | 说明残差/偏置量级，不单独说明性能收益 |
| `ego/social/route_slot_rms`, `*_slot_energy_share` | 每槽输出 RMS；energy share 先按每样本三个槽的均方能量归一化，再在 batch 上平均 | 32/64/32 槽宽不同；share 是各自均方能量占比，不是向量范数的直接跨槽比较 |
| `*_slot_batch_std_mean`, `*_slot_within_sample_std_mean`, `slot_batch_variance_valid` | 前者跨 batch 对每维 std 后平均；后者每样本沿槽通道求 std 再平均 | 跨样本方差只有 batch≥2 才有效；eval batch=1 的 batch std 不表示塌缩。within-sample std 不是跨样本塌缩判据 |
| `slot_sample_energy_*_corr`, `slot_sample_energy_correlation_valid` | 不同槽的每样本标量能量在样本间的 Pearson 相关 | 仅 batch≥2 且所有参与能量都非零方差时写入相关值；字段缺失且 valid=0 表示无效。它不是不同维数槽向量的 cosine |
| `*_projection_weight_rms`, `*_projection_parameter_count`, `*_projection_active`, `slot_output_total_dim`, `mlp_output_head_active` | 投影权重、结构输出头启用状态和输出维度 | 是投影存在/幅度检查；参数量不代表有效梯度或容量匹配 |

encoder 参数梯度按 `diagnostic_parameter_groups()` 提供的前缀组记录：`encoder_all`、`topology_core`、`topology_fusion_goal`、`topology_vehicle_relations`、`slot_ego`、`slot_social`、`slot_route` 与 `route`。组可重叠，不能把范数相加。SAC 在首次和之后每 1,000 次 critic 更新，读取已有 critic TD backward 后、clip 前的梯度；记录 gradient tensor/element 覆盖、None 缺失、非零/非有限计数、L2、参数 L2 及 gradient-to-parameter 比。训练 activation 的 relation/slot 指标和该次 TD gradient 采样更新频率不同，不应误作严格同一 forward 的梯度归因。新 Topo/3slot 配置辅助系数均为 0，因此其 encoder 训练梯度来源为 critic TD loss；旧 Full 即使启用表示目标，TD 梯度采样仍在 TD backward 前清空梯度、之后立即读取，不能解释成 Graph-SLT/SBS 梯度。

结构与诊断的 focused 单元测试通过，且非 slot 上下文等值测试通过；正式 two-worker 训练和最终评估仍须另行记录。模型激活、关系分数与梯度覆盖是实现运行证据，不单独构成性能或因果结论。

## 11. 旧场景 intersection_sorted / depart_scale=4.0：已完成 Topo / Topo+3slot 运行的逐路径审计（2026-09-30—2026-10-01）

同场景完整100k增量结果为 ST 50/50/0、ST-RT 63/37/0、Topo 39/39/22、Topo+3slot 41/56/3（S/C/T，final100，单训练seed0）。详情见[同场景归因](topo_3slot_same_scene_attribution_20260930.md)和[独立实现审计](topo_3slot_implementation_audit_20260930.md)。

Topo不是简单多一枚lane token：`IncrementalTopoEncoder._forward_incremental`在状态route融合后增加静态车道图编码与soft query，`LN(intent+α_topo·context)`再进入vehicle graph/temporal模块；same-lane和conflict-or-merge attention诱导分数成为8维车辆边特征的末两通道，且共用`topology_residual_scale`；route目标readout另走`r+LN(r+α_goal·t)-LN(r)`。当goal gate=0，route readout正好回到r；关闭Topo时主干仍调用同一个topology_norm，所以LayerNorm差异本身不是解释。query compatibility是有效lane节点加方向cosine阈值、路线几何偏置与top-8/fallback，不显式查询lane→route-next-edge连接。拓扑图虽编码SUCCESSOR等关系，但目前query-mask没有显式route-successor合法性条件。

22个Topo timeout的终点与静态网络共同指向明确路网近因：route `-E1→-E0`仅从`-E1` lane2连接至`-E0` lane1；全部所述终态在`-E1_0`, 70m, route index0, speed0，且目标速度非零。旧场景ego从lane2出发；SMARTS动作-1降低lane index，把车向route-invalid的lane1/0移动，处于lane0时-1请求越界。证据支持“错误出口lane造成这些timeout”，不证明Topological attention是策略右移的原因。若新增route-feasibility条件，需把计划route edge与topology node建立可验证映射；当前observation/query不是直接提供lane ID或next-edge身份的合法性mask。

Topo→3slot比较仍共享相同的incremental forward和SAC梯度路径，只换输出读出：joint 384→128→128 ReLU MLP为65,792活跃参数；线性128→32/64/32槽头为16,512，输出总维数同为128。两个输出头及三个槽投影都在同一extractor构造中无条件注册，开关只选forward分支；参数注册名称/形状顺序相同、总量一致。相同种子和同构造顺序下没有代码层面的理由预期共同权重初始化不同，但运行没有保存初始化tensor，不能声称做过bitwise对照。三种上下文已共享ego/route/topology/时空信息，不是互斥的对象槽；下游actor/critic会混合128维输出。辅助Graph-SLT、SBS和representation loss本轮全关；`DetachedSceneActor`对共享encoder输出detach、actor optimizer排除encoder，在线encoder由critic TD loss更新，target extractor在Bellman目标中no-grad并Polyak更新。因此观察到的3slot效果不能归因到SLT/SBS，也不能单独证明槽语义或结构无效。

运行冻结源核对：核心encoder、Topo V2、SAC/policy、SUMO环境和拓扑图文件哈希与run `source_snapshot.json`一致；D1 trainer当前SHA与运行时SHA不同且未找到运行冻结副本，故配置身份以每方法`arguments.json`及`experiment_manifest.json`为准。当前检查的`map.net.xml` SHA256为`FA6BCFAD…C85F0ED`、`ego.rou.xml`为`E9B0D9…C8CB57`；run manifest没有单独记录两者哈希，不能声称独立验证了当时资产逐字节哈希。

后续仅建议两步：先为Topo ego goal branch显式补充route-next-edge可达lane信息并记录它，而不是直接调gate/reward；再在稳定的Topo主干上测试参数量接近的非线性32/64/32头，三槽各用`Linear(128,128)→ReLU→Linear(128,d)→ReLU`，66,048活跃参数（joint head 65,792，差约0.39%）。前者需要route-edge↔lane-node接口/匹配，不是只改cosine阈值；后者是容量/非线性控制，仍不是语义可分解性的证明。这两项未实施、未训练；本节是既有旧场景 `intersection_sorted` / `depart_scale=4.0` 的代码和结果审计，不属于当前 `intersection_random_darrl_*` 低/中/高场景，也不能用于跨场景直接排名；不新增性能结论。

## 12. DARRL medium p=.03 七方法身份与retry02运行快照（尚无结果）

计划在独立 `intersection_random_darrl_medium_v1`（`darrl_r2_20260930`）上比较 `mst_slt`、`sac_mlp_d1_st`、`sac_mlp_d1_st_rt`、`sac_mlp_d1_st_rt_topo`、`sac_mlp_d1_st_rt_topo_routeaware_v1`、`sac_mlp_d1_st_rt_3slot` 和 `sac_mlp_d1_st_rt_topo_3slot`。背景流为三条进口流各自每0.1秒p=.03请求；训练/验证按不同SUMO seed域。目标协议为seed0、fresh100000 raw SUMO ticks、warmup5000 raw、action-repeat=3、每10000 raw保存checkpoint；七方法官方比较使用100回合final validation评估。MST入口是否另有内部评估及其计数待核，不与官方比较回合数混写。最多两个CUDA worker。两入口的预算均按raw ticks，而非决策步数。MST+SLT保留既有算法和SLT，但旧 `frozen_80_20` 来源字段不证明该动态场景使用固定XML holdout。

新增 `routeaware_v1` 仅约束Topo ego-goal候选：与known route-eligible候选求交；若交集为空但存在已知合法候选，则扩展至已知合法集合；若没有已知合法候选，则绕过mask并精确保留base route residual。它不修改social上下文、actor路径、环境动作或奖励，也不是安全保证。机制验证应查看候选交集、expand、bypass和fallback诊断，不能只从成功率推断mask生效。

新增 `st_rt_3slot` 复用原线性三槽readout并关闭Topo，作为相对ST-RT的因素拆分；它不是参数量匹配头，也不新增SLT预测目标。Graph-SLT、SBS及表示辅助loss在新变体中关闭；`topo_3slot`是Topo与同一三槽head的组合。三槽不应描述为互斥或天然语义可分解。

正常训练/评估预期保存 `raw_steps.jsonl.gz`、`decisions.jsonl.gz`、`episodes.jsonl`、`optimization.jsonl`、`representation.jsonl`、`policy_observations.jsonl.gz`及phase manifest/summary；表示流含Topo/slot指标与route-mask计数，SAC记录已有TD梯度采样，不增加额外forward/backward。父侧报告的6项模型诊断、11项route-helper和CPU真实环境两方法save/load/gradient检查通过仅属实现验证。run根 `runs/dm7_1001` 首轮已经启动但七worker全部early exit，因此仍无训练结果；实际启动证据和失败栈见本节末。


首轮启动失败记录（2026-10-01约01:35 Asia/Shanghai）：`runs/dm7_1001`的launcher PID=98120、suite supervisor PID=42508，七个方法worker均以exit_code=1结束（PID：28568、37320、84656、50304、44148、19372、94324，顺序为MST、ST、ST-RT、Topo、routeaware、ST-RT-3slot、Topo-3slot）。实际arguments/manifest核验了seed0、CUDA、fresh raw100000、DARRL medium r2 p=.03、smoke=false、诊断开启、validation比较eval=100、10k raw checkpoint；这些配置证明启动意图，不证明训练完成。worker在环境step因 `sumo_env.py:659` 重复传入 `behavior_telemetry_protocol` 与 `route_lane_facts` 键而退出；没有本轮完成checkpoint/final evaluation或可报告的性能结果。旧run根及源快照/失败日志保留，进程已自然退出；修复后的fresh retry应使用独立目录，当前尚未启动。


Retry01状态更新（2026-10-01约01:45 Asia/Shanghai）：新隔离根 `runs/dm7_1001_retry01`、launcher PID=67536、suite supervisor PID=68888。D1六方法全部early exit code 1，栈停在 `high_density_env_v1.py:355` 的overlay manifest `os.replace`，错误为WinError 3/FileNotFoundError。MST虽记录到1530 raw steps，随后由root终止整个supervisor树，明确不将此短进度用于续训或正式结果；下一正式尝试须fresh。失败日志中目标路径长度由root量得ST=261、Topo=269、routeaware=283字符，source/temp约202–224字符。根因仍未确定，需核source及destination路径/目标目录，不能仅凭较短temp路径排除路径问题。全部失败run/source/log保留；新fresh root待定，没有本轮性能结果。


路径根因的后续最小验证：独立Windows文件探针在parent已存在、source temp写入成功的条件下，使用实际overlay manifest basename执行replace；DEST长度240/259成功，260/261/269/283均得到WinError3/FileNotFoundError。探针记录位于工作区 `tmp/manifest_path_probe_20261001_015015/results.json`。retry01实际ST/Topo/routeaware失败DEST分别约261/269/283，和测试边界一致，故长目标manifest路径是有复现依据的主要根因；此前只量source/temp长度的排除结论不成立。最终以修复代码和新fresh运行验证为准。



Retry02运行核验快照（2026-10-01 02:13:33 Asia/Shanghai）：正式root为 runs/dm7_1001_retry02，launcher/suite PID=66336/68208。MST worker39120为6472 raw/2168 decisions/27 episodes；D1-ST worker14924为6274 raw/1274 updates/23 episodes（summary raw records=6313、decisions=2112）。MST日志已报告CUDA，D1 arguments明确device=cuda；两路诊断错误计数均0且活动日志无traceback。五个增量方法仍在队列，当前没有完成训练或最终评估数据。第一个D1-ST训练episode的reset_info记录30/40/50秒暖场：due/inserted/pending分别19/19/0、28/28/0、41/41/0；background在网数15/13/21，halting数均0，due-inserted-pending残差均0。以上仅是在线采集链和队列启动的工程核验。外部比较每方法100个validation episodes；MST现有内部final eval还会重复100个相同seed/同final policy，评估账本位于 runs/dm7_1001_retry02/evaluation_accounting.json。


## 2026-10-01 正式队列确认：dm7_1001_retry02

当前有效运行是 `runs/dm7_1001_retry02`，已从零启动七方法；前两次目录 `dm7_1001`、`dm7_1001_retry01` 保留为失败/中断尝试，不作为性能结果，也未从其中恢复训练。新增诊断字段重复合并已修复；Windows 长目标路径问题通过仅缩短 D1 车流缓存到 `run_root/_hd/<method>/ns_tr|ns_eval` 修复，不改变模型、场景参数或随机种子。真实环境五次 reset+step 验证通过，同 seed 在不同存储位置生成的路线源/overlay 哈希和车辆数一致；验证来源为 `runs/dm7_pathchk/overlay_path_smoke_results.json`。

协议：`intersection_random_darrl_medium_v1`，DARRL r2，三路各 p=0.03/0.1秒；seed 0；每方法 fresh 100000 raw SUMO steps；checkpoint 每10000 raw steps；最多两 CUDA worker；官方比较采用 validation 外层100回合。方法为 `mst_slt`、`sac_mlp_d1_st`、`sac_mlp_d1_st_rt`、`sac_mlp_d1_st_rt_topo`、`sac_mlp_d1_st_rt_topo_routeaware_v1`、`sac_mlp_d1_st_rt_3slot`、`sac_mlp_d1_st_rt_topo_3slot`。MST原入口另有内部100回合评估；只有外层 `evaluation_results.json` 用于七方法比较，新增日志没有增加评估回合。详见正式根 `evaluation_accounting.json`。

已核验运行快照（2026-10-01 02:13:33 +08:00）：launcher PID 66336，supervisor PID 68208；MST PID39120 已6472 raw/2168 decisions；ST PID14924 progress 已6274 raw/1274 updates（异步diagnostics summary为6313 raw）。两路诊断错误计数均0，活动日志无Traceback；其余五路待worker空闲依次执行。这是运行健康检查，不是最终性能结果，100k 尚未在该快照完成。

## 奖励与评估口径索引（2026-10-01）

38条历史结果的逐组核验和运行证据见[奖励审计](intersection_reward_audit_20261001.md)：可支持共同v2系数设计，不等同于全部历史环境终局实现均一致。
v2训练reward按policy decision应用终局项、−.01 step cost及可读时的.02×距离差；该距离是ego插入后SUMO累计`getDistance`（米），非到目标距离变化，代码不裁剪。
评估`mean_return`来自raw undiscounted `success−collision`，与训练shaped reward不同；当前DARRL r2事件互斥，collision优先，r1双标曾使训练分支奖励与raw评估含义分离。
MST+SLT的SLT、以及Full的Graph-SLT/SBS属于表征辅助优化目标，不是环境每步reward；方法间奖励字段应与辅助损失分别记录。

日志已实际写出 raw、decision、episode、optimization、representation 数据。训练输入快照每10000 raw一次、最多10份；外层评估首动作与终止动作输入最多200份/worker。字段包括动作/目标与实际速度、换道请求与实际转换、路线接续合法性及错道停滞、TTC/碰撞位置、Topo/三槽表示与梯度、优化指标。D1 ST首回合预热30/40/50秒抽样：due与inserted分别19/28/41，pending均0，在网背景车15/13/21，halting均0；due-inserted-pending差额均0。这只是日志与预热流程核验，不是场景总体统计。

已通过17项behavior测试、17项route/encoder定向测试（另4个subtests）、1项真实诊断环境集成测试，以及新增方法真实环境预测/梯度/保存加载验证。源码实际内容与SHA归档在正式根 `source_archive/manifest.json`，完整启动配置在 `suite_manifest.json`。后续分析应读取最终checkpoint和外层评估原始结果，不能用本段早期训练快照推断方法优劣。


## 2026-10-01：基线分析中的奖励与终止口径补充

按dm7_1001_retry02的source_archive核验：evaluation.py累加info.undiscounted_reward，sumo_env.py原始项为success−collision，因此评估mean_return等于成功率减碰撞率，timeout/offroute在该标尺为0。learner接收reward_shaping_v2事件奖励+10S−10C−10offroute−5timeout，并按decision叠加−.01+.02×距离增量。不能把评估分理解为训练塑形回报，也不能说训练未惩罚timeout。

SUMO600raw上限作为truncated，replay排除timeout的terminal mask，SAC在timeout仍bootstrap；这是当前SAC与MST共享的协议事实，不足以单独解释模型性能差异。若未来调整有限时域任务语义，需作为单独版本/实验，保留现有基线。当前未改实现。

当前DARRL中档MST16/53/31、SAC31/62/7（S/C/T）。原始证据、旧场景对照及归因边界见[基线报告](darrl_medium_mst_vs_sac_analysis_20261001.md)。尚无证据把错道停滞和较高速路口碰撞直接归因于SLT梯度冲突、特定超参或表示层失效；用户指定下一步先比较SAC→ST，再逐模块定位。

## 13. 2026-10-01：回到旧场景 sorted 主阶段与诊断/奖励口径修复

当前主队列从前一阶段 DARRL medium p=.03 切回正确三路固定模板 `intersection_sorted`、depart_scale=4.0；DARRL结果仍保留为独立历史阶段。正式run根[`runs/sort2_1001`](../../../runs/sort2_1001)于2026-10-01 22:26启动，supervisor PID 92912；截至22:31(+08)，routeaware worker PID77596为5678 raw/678 updates，3slot worker PID80940为5980 raw/980 updates，均CUDA training且持续推进；诊断错误数0、活动日志无traceback。两路尚未完成100k或最终验证。新配置为两个fresh SAC+MLP seed0/100k raw、双CUDA、10k raw checkpoint、外层final validation 100：`sac_mlp_d1_st_rt_topo_routeaware_v1`与`sac_mlp_d1_st_rt_3slot`。各自已有父方法可支持递进解释，但二者相互对比并非单变量。固定流量/source快照见run根`suite_manifest.json`及`source_archive/manifest.json`。

本阶段启用ST spatial/temporal/social、RT/map-route、Topo及route-aware候选、三槽、梯度与实际critic参数更新诊断；参数所有权/None-vs-zero、每256-forward与TD update 1/1000的采样、有效分母及因果限制见[诊断协议](module_diagnostics_protocol_20261001.md)。启动前43项定向单测通过。routeaware单方法300 raw/1 episode smoke在`runs/s2sm_1001`成功。3slot首次smoke在`runs/s2sm_1001`的训练阶段完成、eval因slot-hook scratch字典误入数值指标失败；失败attempt及日志保留。隔离scratch并加nullable scalar合并后，3slot单方法300raw/1episode重试在`runs/s2sm_r1_1001`成功。smoke仅作实现链验收，不是方法结果。

当前评估协议切为`environment_step_reward_v2`：主`episode_return`累加实际返回训练器的shaped per-decision reward；底层raw `success−collision`另存为`raw_episode_return`/raw summary，并输出适用的六分量均值、coverage及reconciliation。此前38条结果奖励审计是旧文件/旧评估schema快照，不能用其raw `mean_return`口径解释本阶段新评估；历史数值和分母不被回写。该报告开头已注明范围。
