# v4.9.2 运动学安全链路与模型缺口深度复核

状态：promotion 运行中；仅有 1/6 个完整配对，模型归因不得据此定版  
复核日期：2026-08-27（Asia/Shanghai）  
Formal test：未访问，仍锁定  
约束：不新增 TTC/headway/距离阈值、shield、veto、动作投影或动作重写；后续只允许完善可学习模型

## 1. 结论

当前 v4.9.2 候选没有候选侧的运动学安全投影。部署动作由三个物理可执行的
lane-conditioned actor 速度提议，经学习分数
`min(Q_reward_target) - lambda * max(V_collision_target)` 直接选出；除不存在的
相邻车道外，没有状态相关规则删除动作，也没有模型输出后的安全覆盖。

需要改进的不是缺少一条安全规则，而是学习系统本身存在表征与动作提议错位：

1. actor 与 reward critic 共享 TopoTemporalGraph 编码器，但 actor 在策略头前对
   编码特征执行 `detach`；风险梯度只能塑造 lane/speed 头，不能通过 actor 路径
   塑造共享运动表征；
2. collision critic 另建完整且独立的 TopoTemporalGraph 编码器，由稀疏碰撞 TD
   目标单独训练；它与 actor/reward 表征没有共享的风险语义坐标系；
3. deterministic decoder 每条 lane 只有一个 actor 均值速度候选。即使 collision
   critic 能正确给三个候选排序，也无法选择未被 actor 提议的更合适速度；
4. 当前唯一完整配对中，碰撞前 10 步风险排序 AUC 为 0.93，且碰撞窗口 94% 的
   选择已经是三个候选中的最低预测风险动作。这一局部证据更符合“候选集/未来
   策略与风险表征错位”，不支持再加外部安全投影。由于只完成 1/6 配对，该结论
   仍是待完整矩阵检验的假设。

## 2. “投影”一词的逐类排查

| 类型 | 实际用途 | 是否改变本轮候选动作 | 判定 |
| --- | --- | --- | --- |
| 神经网络 `query/key/value/output_projection` 与 slot projection | 线性层命名 | 否 | 普通可学习层 |
| polyline projection | 计算道路折线上的最近点/弧长 | 本轮 direct 控制不使用其曲率限速分支 | 仿真几何工具，不是候选安全策略 |
| SMARTS Ackermann proxy | 按曲率限制速度并限制加减速度 | 否；promotion 参数为 `ego_control_profile=direct` | 冻结的可选环境动力学代理，本轮未启用 |
| `lane_action_mask` | 仅检查当前非内部道路是否存在相邻 passenger driving lane | 只排除不可执行车道 | 动作空间可执行性，不读取交通风险 |
| TTC/相对位置/相对速度 | 作为 vehicle-graph 的 8 维边特征 | 只经网络学习后间接影响价值/策略 | 模型输入，不触发阈值或动作覆盖 |
| topology heading compatibility/top-k | 选择静态车道图 attention key | 影响表示，但不读取碰撞风险、不重写动作 | 继承的表示稀疏化结构，不是安全投影；后续不得再增加手工门槛 |
| target-critic argmax | 在物理可执行动作中最大化 learned score | 是最终模型决策 | 学习式价值决策，不是安全规则 |
| exact keep tie-break | 只有 learned score 精确并列时选 keep | 仅数值并列 | 与安全状态无关的确定性约定 |

## 3. 代码证据

- `envs/sumo/decision_alignment_v4.py:104-150`：mask 初始为
  `[0, 1, 0]`，只因目标 lane rank 存在而打开左右动作；没有车辆距离、TTC、
  headway 或碰撞查询。
- `envs/sumo/sumo_env.py:526-554`：换道只验证道路、driving lane 和 lane rank；
  `setLaneChangeMode(..., 0)` 关闭 SUMO 的安全换道 veto。
- `envs/sumo/sumo_env.py:607-629`：只有非 direct 控制才进入曲率限速代理；本轮已
  完成 promotion 作业的 `arguments.json` 均记录 `ego_control_profile=direct`。
- `algos/sb3_torch/topo_temporal_features_v2.py:536-603`：TTC 由相对位置/速度计算，
  作为 edge feature 拼入 vehicle graph；不存在 TTC 分支控制动作。
- `algos/sb3_torch/hybrid_policy_v4.py:99-120`：actor 特征在 lane/speed 头前
  `detach`；mask 只用于将不可执行 lane logit 置为极小值。
- `algos/sb3_torch/hybrid_policy_v4.py:273-310`：actor/reward critic 共享在线
  extractor，actor optimizer 明确排除该 extractor 参数。
- `algos/sb3_torch/hybrid_policy_v4_9_model.py:39-64`：collision online/target
  critics 各自新建 extractor，与 reward/actor extractor 独立。
- `algos/sb3_torch/sac_v4_9_model.py:228-320`：actor loss 对 reward/collision
  critic 参数冻结，但仍可通过 action 对 lane/speed 头反传；collision loss 只更新
  独立 collision critic。
- `algos/sb3_torch/hybrid_policy_v4_9_model.py:211-279`：最终部署只做 learned
  risk-adjusted score 的 masked argmax，没有 confidence gate、veto 或动作重写。
- `algos/sb3_torch/replay_buffer_v4_9.py:50-109`：collision 监督只来自已经发生的
  `info["collision"]`，未用几何阈值生成伪标签。

## 4. 当前 promotion 的不可取消事实与证据边界

Cross TemporalGraph 两个种子的 collision 均为 6/30，合计 12/60=0.20；候选
seed20 已为 20/30。即使尚未完成的候选 seed21 达到理论下界 0/30，Cross 候选
collision 仍至少为 20/60=0.3333，最小 delta 仍为 +0.1333，大于冻结门槛 +0.05。
因此当前版本的 promotion gate 已在数学上不可能通过，但这不取消剩余作业：仍按
用户要求全部尝试、统一汇总，并用完整 6 对轨迹确定下一版模型，而不是让一个局部
场景决定结构。

## 5. 下一版只允许的模型改进方向（待完整矩阵选择）

以下是候选假设集合，不是已经选定的 v4.10，也不据此提前启动新训练：

1. **共享风险表征、单一参数所有权**：reward 与 collision heads 使用同一在线/
   target 场景编码器；共享参数只归一个 optimizer 所有，联合接收 reward、真实碰撞
   与既有 Graph-SLT 梯度，两个价值任务保留独立 twin heads。
2. **风险条件化 actor**：让 lane/speed 头读取同一共享风险 latent，使碰撞梯度能
   改善动作提议，而不在 actor 输出后覆写动作。
3. **学习式多速度提议或直接策略部署**：若完整归因继续显示“最低风险候选仍
   碰撞”，扩大每 lane 的可学习速度候选并用同一 learned score 训练/选择，或直接
   部署端到端风险条件化 actor；不使用手工刹车、TTC 门槛或 shield。
4. **仅真实事件监督**：collision target 继续只由环境闭环碰撞事件构造；可使用
   观测到的相邻转移做自监督动力学表征，但不得用 oracle future、手工几何 unsafe
   标签或反事实动作规则。

完整 promotion 后必须用所有 6 个精确配对回答：风险区分是否跨场景稳定、碰撞时
是否仍选最低风险候选、actor-target 分歧是否为主因、速度提议是否覆盖安全区域、
以及共享/独立表征的尺度漂移。只有那时才能冻结单一新版本改动。

## 6. 执行约束

- 当前及未来 promotion 继续使用自动 registry；任何单作业失败都不取消余下作业。
- 中断/失败目录只读保留，恢复作业写入版本化新目录。
- promotion 完整失败后才创建 v4.10 scientific files；旧冻结文件不修改。
- 新版先过定向单测、梯度所有权测试、序列化、真实 SUMO smoke 和完整 development；
  development 通过后才允许新 promotion。
- promotion 完整通过前，120-job formal test 永不启动。

## 7. 可复跑审计

- 工具：`tools/audit_v4_9_2_kinematic_safety_model.py`
  （SHA-256 `5553872fb924bf1904201f2eea484f7f2b4444295da3ccabd83ff25324faf9ce`）；
- 定向测试：`tests_sb3_sumo/test_audit_v4_9_2_kinematic_safety_model.py`
  （SHA-256 `f36e0897bfd8a3c7bdee419c7100e1902394de0ccac04444cde88906957e61a9`），
  结果为 3 passed；
- 审计结果：
  `results_topo_v4_9_2_dev/engineering/kinematic_safety_model_reaudit.json`
  （SHA-256 `0830c83976d63b7d4e3d4b25fa6514958ca6a6176060b80fcae8a114eaa54318`）。

动态梯度探针实测：actor 输出和 learned collision objective 都能更新 lane/speed
heads，但二者都不能经 actor 路径更新共享 extractor；collision optimizer 则拥有
另一套独立 extractor。该结果将静态源码判断提升为可执行、可回归的模型证据。
