# v4.10 运动学安全投影与规则机制审计

## 1. 结论

v4.10 的可部署候选没有运动学安全投影，也没有新增 TTC/headway 阈值、固定速度
网格、置信度门、换道否决、shield、规则回退、平局偏好或解码后动作改写。候选的
改动只发生在可学习模型内部：共享风险场景编码器、每车道三个可学习速度分量、
replay 支持似然、learned twin-value uncertainty，以及对可执行提议的纯模型分数
`argmax`。

完整可复跑证据位于
`results_topo_v4_10_dev/engineering/no_kinematic_projection_audit.json`。该审计基于
当前源码、可执行梯度探针和新鲜 E0c 真实 SUMO 轨迹；没有访问 formal test。

## 2. 必须区分的三层语义

| 层次 | v4.10 行为 | 分类 |
|---|---|---|
| 模型决策 | actor 产生 3×3 learned lane/speed proposals；target twin reward、collision、disagreement 和 learned component mass 共同评分 | 可学习模型 |
| 物理动作可执行性 | 仅屏蔽当前道路上不存在相邻 passenger driving lane 的 lateral command | 既有动作空间结构约束，不查询交通风险 |
| 环境动作参数化 | `[-1,1]` 速度线性映射到 `[0,10] m/s`，精确 `{-1,0,+1}` lateral code 映射到车道命令 | 接口参数化，不是风险投影 |
| 运动学/交通风险投影 | 不存在 | 禁止项 |

物理 lane mask 不读取 leader、follower、碰撞、TTC、headway 或 unsafe 标记；它只与
`SumoSceneEnv._apply_control` 对相邻 driving lane 是否存在的检查保持一致。该约束未在
v4.10 新增，也不会根据交通风险覆写模型已经选择的速度。

## 3. 从策略输出到 SUMO 的调用链

1. `SupportedMixtureHybridActor` 的网络头学习每个 lane 的 component logits、speed
   mean 和 speed log-std；推理候选不是固定动作网格。
2. `SharedRiskSupportedMixtureSACPolicyV410._risk_adjusted_proposal_values` 对这些提议
   计算 learned reward、learned collision value、twin disagreement 与 component
   log-probability。
3. `_predict` 只用物理 lane mask 将不可执行 lane 的分数置为 `-inf`，随后对扁平
   model score 执行 `torch.argmax`，并直接返回对应 actor proposal。
4. `PaperSumoSceneEnvV4.step` 将 action 原样交给父环境；父环境只做声明动作域
   `[-1,1]` 的防御性裁剪与固定参数化。
5. 实验命令显式固定 `--ego-control-profile direct`；该分支令
   `effective_speed=requested_speed`。曲率速度限制和逐 tick 加减速代理只存在于未启用
   的 `smarts_ackermann_proxy` 分支。
6. SUMO 的 safe-speed 与 lane-change safety controller 分别通过 speed mode 0 和
   lane-change mode 0 关闭，因而不会在模型后追加隐式安全否决。

## 4. 删除的非运动学规则

审计中额外发现，早期 v4.10 实现曾在 learned score 精确同分时优先 keep lane。它
不是运动学投影，但仍属于人为语义偏好，与“只完善模型”约束不一致，因此在任何科学
development 训练前删除。当前实现对完全同分探针选择第一个扁平可执行 proposal，
与 `torch.argmax` 本身完全一致，并记录
`semantic_tie_override_used=false`。旧 E0/E0b 工程目录只读保留，不进入冻结或科学
结果；E0c 是删除该机制后的新鲜端到端证据。

## 5. 模型完善而非规则补丁

v4.9.2 的完整 promotion 表明，单一 actor speed proposal 经常没有覆盖 learned
critic 认为更优的速度区域；独立 collision encoder 还会导致风险尺度与 actor 表征
脱节。v4.10 针对这两个模型瓶颈作结构性修复：

- reward/collision/Graph-SLT 共同塑造同一 online scene encoder；
- encoder 参数仅由一个 optimizer 拥有，actor/reward/collision heads 与之严格不重叠；
- actor 仍对 encoder 特征 stop-gradient，避免多个 optimizer 竞争共享参数；
- replay 中真实执行速度通过 mixture NLL 约束 learned proposals 的支持域；
- collision 标签只来自环境已经发生的 `info["collision"]`，不构造几何 unsafe
  伪标签；
- 部署动作始终是 actor learned proposal，不进行反事实网格搜索或后处理。

可执行梯度探针确认 reward 与 collision objective 都能到达共享 encoder，三个 actor
heads 均有有效梯度，且 optimizer overlap 为 0。

## 6. E0c 真实运行证据

- 96/96 raw training steps 完成，并产生有限的 reward、collision、support 与
  uncertainty 训练诊断；
- 选择器只比较两个 checkpoint 与同一个 learned target decoder，没有 fusion 或
  confidence gate；序列化前后 policy parameter-state SHA-256 一致；
- 7/7 validation 决策同时满足：纯模型 argmax、分数方程、精确 learned proposal、
  精确部署 action、物理 mask 可执行、无 rewrite、无 semantic tie override；
- 7/7 归一化模型动作已经处于声明动作域内，环境侧裁剪逐元素为恒等映射；
- `ego_control_profile=direct`、validation split、formal lock 均由运行产物封存。

E0c 的单回合 outcome 仅用于证明完整链路可运行，绝不用于判断 v4.10 科学效果。
效果判断只能使用冻结后的 D1–D4 development，再依次通过 ablation 与完整 12-job
promotion；进入任一阶段后，单作业失败不会取消该阶段剩余作业，统一在全部尝试后
汇总。

## 7. 冻结门槛

只有以下证据同时成立才允许冻结并启动 development：定向测试与完整回归通过；E0c
候选和 E1 TemporalGraph 控制均完成真实 SUMO 闭环；无投影审计通过；动作诊断所有
精确率为 1；共享 encoder identity 与 optimizer ownership 通过；自动流水线验证
attempt-all-then-summarize；formal test 保持锁定。
