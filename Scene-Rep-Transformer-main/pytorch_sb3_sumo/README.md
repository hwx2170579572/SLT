# Scene-Rep-Transformer：PyTorch + SB3 + SUMO 迁移版

## Topology-Temporal Graph-SLT L1 扩展

本目录现包含可执行而非空壳的 L1 扩展：确定性 SUMO 五关系车道拓扑、逐时刻车辆图、拓扑查询、时序编码、`32/64/32` 结构化潜变量，以及保持原 SAC、奖励、动作空间与一步 SLT replay 的 Graph-SLT。原 `mst` 与 `scene_rep` 基线受 `artifacts/ccfa/baseline_freeze.json` 哈希合同保护。

- `topo_scene`：原始完整拓扑—时序方法。
- `temporal_graph`：去掉拓扑查询的机制消融。
- `topo_scene_balanced`：仅在三个结构化槽输出后加入无仿射 LayerNorm 的单变量稳定性迭代。
- `ccfa.yaml`：S0–S9 机器可检查阶段合同。
- `experiments/topo_scene/experiment_contract.yaml`：冻结的流量、raw-step、seed、指标、门禁与停止规则。
- `results_topo_scene/final/FINAL_REPORT.md`：实验完成后由真实 run artifact 编译的最终结论。

开发期实测结论是负向且有边界的：原始 `topo_scene` 在 50k 的 20 个配对
`left_turn` episode 上为成功率 0.60、碰撞率 0.40；单变量
`topo_scene_balanced` 修复到 1.00/0.00，但没有超过同为 1.00/0.00 且完成更快的
`scene_rep`。因此没有绕过门禁启动 1M-step 多场景确认，最终保留
`scene_rep` 作为经验部署参考，并将 `temporal_graph` 作为图方法内部的安全消融。
完整数字、深层归因、学习曲线和证据边界见
`results_topo_scene/final/FINAL_REPORT.md`。当前扩展回归为 `140 passed`，六图拓扑
审计和 111/111 迁移源文件哈希检查均通过。

```powershell
conda run -n llm_pipeline python tools/check_ccfa_contract.py
conda run -n llm_pipeline python tools/run_topo_experiments.py validate
conda run -n llm_pipeline python -m pytest tests_sb3_sumo -q
```


本目录中的迁移实现完全通过新增文件提供。原 TensorFlow/TF2RL、SMARTS 和
CARLA 文件没有被修改。SMARTS/CARLA 只用于核对原任务、观测、动作和奖励契约；
新版本运行时不安装、不导入、也不启动 SMARTS 或 CARLA。

本目录是迁移版唯一的维护与运行入口；其父目录仍是原始 TensorFlow 工程。迁移前曾
混放在父目录各包中的副本已经移到工作区同级的可恢复归档
`_pre_separation_interleaved_sb3_sumo_backup_20260804`，不应从归档运行。已有的
`results_sb3_sumo_paper` 和 `tmp` 是历史运行产物，不属于源代码，因此未做数 GB 的
搬运。逐项核对结果见 [SOURCE_COMPATIBILITY_AUDIT.md](SOURCE_COMPATIBILITY_AUDIT.md)。

## 运行栈

已在 conda 环境 `llm_pipeline` 验证：Python 3.10.20、PyTorch
2.10.0+cu128、Stable-Baselines3 2.9.0、Gymnasium 1.2.3、NumPy 2.2.6，
以及本机 SUMO 1.25.0。Python 依赖清单见 `requirements.txt`；SUMO
及其 TraCI Python 工具由 SUMO 安装目录提供。

## 场景对应关系

| 新场景名 | 原参照场景 | SUMO 中的任务 | 原始步数上限 |
|---|---|---|---:|
| `left_turn` | SMARTS left turn | 无信号十字路口左转并穿越横向车流 | 400 |
| `cross` | SMARTS double merging | 主路连续通过两个匝道汇入口 | 600 |
| `roundabout` | SMARTS roundabout | 高交通流环岛通行 | 1000 |
| `roundabout_easy` | SMARTS easy roundabout | 低交通流环岛通行 | 400 |
| `roundabout_medium` | SMARTS medium roundabout | 中交通流环岛通行 | 600 |
| `carla` | CARLA urban unsignalized left turn | 无信号城市路口左转，背景车辆与行人参与 | 302 |

这里的“等价”指任务语义、接口张量、控制频率、奖励/终止条件和算法训练流程
等价。SUMO 与 SMARTS/CARLA 的车辆动力学和地图格式不同，因此无法做到跨仿真器
逐帧数值相同；新增网络保留了原任务中的冲突关系、交通参与者类型与难度层次。

## 接口等价性

- 动态轨迹：6 个 actor（ego + 最近 5 个参与者），10 步历史，特征顺序为
  `[x, y, heading, vx, vy]`。
- SMARTS 类场景地图：每个 actor 两条候选路径，每条 10 点，形状
  `(12, 10, 5)`，特征为 `[x, y, heading, is_ego, is_neighbor]`。
  路径按源码的起始车道索引排序，并保留 SMARTS v0.4.17 的实际 route-filter
  行为：固定路线 ego 会因 internal edge 不在 route 列表而在路口截断；通过
  `mission=None` 附加传感器的邻车则在普通 edge 上无路线约束，可沿真实 lane
  link/internal lane 分支。两者都只扩展源码的 10 点 lookahead，再取前两条；仅当
  候选列表确实只有一条时复制，已有但全零 padding 的第二候选会原样保留。
- CARLA 类场景地图：每个 actor 三条候选路径，每条 10 点，仅保留 x/y，形状
  `(18, 10, 2)`；最近参与者同时包含车辆和行人。
- 动作：二维 `[-1, 1]`。第 0 维线性映射到 0--10 m/s；第 1 维按
  `-1/3` 和 `1/3` 离散为 `-1/0/+1`。SMARTS 契约中 `-1` 为右换道、
  `+1` 为左换道；CARLA 源码的符号相反，SUMO 适配器按场景分别处理。
- 奖励：到达 `+1`、碰撞 `-1`，其余为 0；`legacy_info` 保留
  `(finish, collision, off_route, max_time)`。
- Proposed/MST 的一个策略动作保持 3 个 0.1 秒 SUMO 步，保持期间的奖励按 gamma
  累积；历史仍逐个原始仿真步记录。SMARTS PPO 每个原始步重新决策；CARLA PPO
  每个原始步保存 transition，但 action/value/log-prob 连续复用 3 步。
- ego 按 SMARTS 外部控制车辆的源码契约使用 `speedMode=0b00000`、`tau=4`、
  `decel=6` 和 `3.68×1.47×1.4 m` passenger 尺寸，并关闭 SUMO 对显式换道请求的
  安全覆盖，避免额外的规则安全控制器替代策略动作。
- TraCI 的车辆前保险杠坐标按 SMARTS v0.4.17 provider 的车型尺寸转换为中心坐标；
  碰撞事件同时使用 SUMO 报告和带 0.05 m 余量的二维有向包围盒接触检测，对应源码
  Bullet collision sensor 的实际判定，而不是只依赖 SUMO 可能漏报的路口碰撞列表。
- 交给 replay 的 reward 保留上述保持期折扣；`info["undiscounted_reward"]`
  同时保存逐原始步奖励和，评估器用后者复现原 runner 的 episode return 统计。
- 五个 SMARTS 类场景按 v0.4.17 的词典序和 seed-42 起始滚动顺序循环原交通
  route 文件，并复现默认 `endless_traffic`：驶出非环形路线的社会车辆会沿原路线
  重新注入，避免 ego 延迟出发后交通场被错误稀释。

## 算法迁移

默认 `scene_rep` 不是普通 SB3 SAC 的别名，而是新增的完整迁移算法：

- critic 持有 PyTorch MST 分层场景编码器；actor 使用其 stop-gradient 特征；
- 时间、地图、邻车—路径、actor—actor、目标路径注意力均保留，注意力内部投影
  维度按原 Keras `key_dim` 语义实现；
- 双 Q 网络分别使用 64 维动作嵌入，MLP 为 128/32；
- 保留 SLT 的动作条件未来表征、投影器、预测器和负余弦相似度目标；
- 表征优化器与 critic 优化器分离，但共享在线编码器参数；
- `cross` 关闭随机旋转增强，并使用目标 critic 编码器和 Polyak 更新的独立目标
  投影器；其他场景按原实现使用共享在线编码器目标分支；
- `roundabout` 按原配置跳过邻车—未来路径融合；`cross` 和 `carla` 的最终目标
  注意力为 2 头，其余为 1 头；
- SAC 保留双 Q、自动熵系数（初始 0.2）、gamma 0.99、tau 0.005、4-step
  replay、NAdam、batch 32、20,000 条 replay 容量与 5,000 原始步 warm-up。
- 自动熵更新按原式直接对 `alpha=exp(log_alpha)` 求导，并使用
  Adam `beta1=0.5`，而不是 SB3 的默认 log-alpha 代理损失。
- replay 同时返回 one-step next observation 和 4-step bootstrap observation：前者
  只用于默认 `future_steps=1` 的 SLT，后者只用于 SAC TD 目标。
- Proposed 保留发布 trainer 中 `make_predictions=1` 的实际队列行为：每个 episode
  的最后一条 transition 会在常规写入后随 episode 尾队列再写入一次；MST 不重复。

MST 使用同一个源码时钟 SAC，但在构造前关闭 SLT，因而不会实例化辅助网络、消耗
额外随机数或执行 Proposed 专属的 episode-end 重复插入。PPO 不是统一向量近似：五个
SMARTS 场景使用源码的单帧 80×80 RGB CNN，CARLA 使用 `(6,10,5)` 共享 GRU 与
6-head relation attention；其 512 步 horizon、PPO 概率非对称、优势归一化和源码
bootstrap 行为均由 `SourcePPO` 保留。发布 PPO 还会在更新 actor 时切断共享编码器
梯度，因此 CNN/GRU 只由 value loss 更新；迁移版同样保留这条梯度边界。

PPO 请求 100,000 步时按源码采满 crossing horizon 到 100,352，但原 `tools/test.py`
恢复的是最终 horizon 更新前保存的 100,000 checkpoint。迁移版将后者作为
`final_model`，将前者另存为 `post_horizon_model`。原 PPO 测试 API 本身存在位置参数
错绑和 CARLA 三返回值解包错误，不能按字面运行；正式评估采用明确披露的最小意图修复：
确定性均值动作和 evaluation-mode dropout。

论文的 plain SAC 基线现作为明确标注的源码重建提供。论文称其为 LSTM，但发布
`RLEncoder` 实际实例化 `GRU(256, return_sequences=True)`，再接
`Dense(256, relu)`；迁移版按 `STATE_LSTM` 恢复 ego
`[x,y,speed,0,heading]`、邻车 `[x,y,speed,distance_to_ego,heading]`、缺帧逐分量
插值、显式时间 mask 和 `[time, actors*5]` 排列，只读取这组历史状态，完全忽略
Proposed 轨迹与候选地图，也不构造层级 Transformer。它继续复用源码的
critic-owned 共享编码器、actor stop-gradient、双 Q、64 维动作分支、NAdam、4-step
replay、三步动作保持和 raw-step 更新顺序。由于发布仓库没有可选择的 `sac` CLI 或完整
基线配置，该实现标记为 `paper_sac_lstm_reconstruction_v2`，不宣称是未发布脚本的
逐位复刻。

SB3 原生以策略决策计时，而原 runner 以仿真原始步计时。Proposed/MST 训练入口通过
raw-step callback 精确限制 `--max-steps`、warm-up 与 checkpoint 周期；一个 held
action 内的中间更新先使用旧 replay，边界 transition 写入后再执行最后一次更新，
对应发布 runner 的语句顺序。`arguments.json` 同时记录请求的原始步数和实际 SB3
决策步数。

## 训练与评估

在本文件所在目录运行：

```powershell
conda activate llm_pipeline
python tools/train_paper_sb3_sumo.py --scenario left_turn --algo scene_rep --max-steps 100000 --eval-episodes 50 --device cuda
```

动作重复默认值按源码算法自动选择：Scene-Rep/MST/SAC 为 3，PPO 为 1；CARLA PPO
的三步动作保持由 `SourcePPO` 内部实现。显式给 PPO 传入其他环境重复值会立即报错，
避免把同一动作错误地保持两次。训练产物默认写入本目录下的
`results_sb3_sumo_paper`。

其他等价场景只需替换 `--scenario`。先做环境检查而不训练：

```powershell
python tools/train_paper_sb3_sumo.py --scenario cross --algo scene_rep --check-only --model-name cross_check
```

加载并评估 checkpoint：

```powershell
python tools/eval_paper_sb3_sumo.py --model results_sb3_sumo_paper/<run>/final_model.zip --scenario left_turn --algo scene_rep --episodes 50 --seed 10000 --expected-training-raw-steps 100000 --output results_sb3_sumo_paper/<run>/paper_evaluation_detailed.json
```

源码直控迁移矩阵使用 `scene_rep`、`mst`、`ppo` 的冻结 v12 协议。优先基线矩阵使用
`sac`、`ppo` 的 v14 协议，并显式启用 `--ego-control-profile
smarts_ackermann_proxy`、`--traffic-protocol frozen_80_20` 和
`--episode-limit-profile paper`。完整协议、场景矩阵和一键命令见
`experiments/sb3_sumo_paper/README.md`。

论文称从五次训练中测试成功率最高的 policy，但发布仓库没有可用的 best-policy 保存
流程。新增训练会同时保留源码可恢复的 latest checkpoint，以及按源码日志固定分母 20
前瞻捕获的 `best_training_success_model.zip` 和选择元数据；后者明确标记为重建，并会与
latest 结果分开报告，不把自创规则伪装成论文未发布的测试过程。

## 验证

```powershell
python -m pytest tests_sb3_sumo -q --basetemp tests_sb3_sumo/.pytest_tmp
python -m compileall -q algos configs envs tools tests_sb3_sumo
python tools/verify_migration_invariant.py
```

最后一条命令根据 `MIGRATION_ORIGINAL_SHA256.json` 检查迁移开始前已存在的全部
111 个文件；新增迁移文件不在该基线清单中。

2026-08-04 本轮验收结果：`104 passed`；Scene-Rep 与 PPO 在不显式指定
`--action-repeat` 时均通过真实 SUMO `--check-only`；`compileall` 通过；原源码
哈希为 111/111 不变。测试中的 5 条 warning 均来自 SB3 通用检查器把三维 float
轨迹/地图误判为图像，属于已知静态提示。

SB3 的通用环境检查器可能把三维 float 张量误判为图像并给出提示；本实现明确使用
`HierarchicalSceneExtractor` 且 `normalize_images=False`，因此该提示不影响训练。
