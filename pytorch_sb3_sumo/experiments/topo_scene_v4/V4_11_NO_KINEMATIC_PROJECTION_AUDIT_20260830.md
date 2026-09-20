# v4.11 运动学安全投影审计

## 结论

v4.11 PRCR 候选端不存在运动学/交通风险安全投影，也没有 TTC 或车头时距阈值、换道否决、置信门、shield、规则回退、语义平局偏置或解码后动作重写。v4.11 的改动仅发生在学习到的碰撞风险模型训练目标：soft-target Bernoulli NLL、连续目标差排序和 detached EMA 一致性。

相邻 driving lane 不存在时的车道 mask 被保留。它只定义物理上存在的离散动作，不查询前车、后车、碰撞、TTC、headway 或其他交通风险，因此不属于安全策略。

## 审计边界

机器可验证审计文件为 `results_topo_v4_11_dev/engineering/no_kinematic_projection_audit.json`，SHA-256 为 `14ce7099aa427f20ccb20ce129a890ba4bd94af122bdbceef7a80e64014bc458`。

审计绑定了 18 个传递源码文件，包括：

- v4.11 薄策略身份类及其实际继承的 v4.10/v4.9/v4 动作选择实现；
- v4.11 风险学习器、replay buffer、配置与动作诊断；
- paper 训练入口、复用的冻结环境工厂和 checkpoint/decoder 选择器；
- policy action 到 Paper SUMO、动作适配、直接速度控制和物理车道 mask 的环境调用链。

解析后的 `_predict` 和风险打分方法均由冻结的 `SharedRiskSupportedMixtureSACPolicyV410` 提供。v4.11 没有在子类中插入动作后处理。

## 可执行模型检查

可执行探针确认：

- actor 为每个车道产生 3 个学习到的速度分量；
- learned reward、collision 和 twin disagreement 共同形成模型分数；
- 只对不存在的相邻车道填充 `-inf`，之后执行普通 flattened `torch.argmax`；
- 返回值与被选中的 actor proposal 逐元素相等；
- 精确平局采用张量的首个可行 argmax，没有 keep-lane 语义覆盖；
- reward 与 collision 目标都向共享 encoder 提供梯度，actor head 接收梯度但不取得 encoder 优化器所有权；
- encoder、actor、reward critic 和 collision critic 的参数所有权交集为 0；
- direct 控制分支按请求速度原值调用 TraCI，动态探针若访问曲率/动力学代理会立即失败；
- SUMO `speedMode` 和 `laneChangeMode` 均为 0，未引入 SUMO 内置安全速度或换道否决。

## 真实 SUMO 轨迹检查

工程候选 `E0c__prcr_full__left_turn__s1` 完成 96/96 raw training steps，并在 validation 上产生 8 条真实决策记录。逐条检查结果为：

- 模型分数 argmax 与保存记录匹配率：1.0；
- 最终动作与 actor proposal 精确匹配率：1.0；
- 学习 proposal 来源率：1.0；
- 无动作重写率：1.0；
- 无语义平局覆盖率：1.0；
- 归一化纵向动作到目标速度的线性映射匹配率：1.0；
- 目标速度与 direct SUMO 有效目标速度匹配率：1.0；
- 精确车道码到环境车道命令匹配率：1.0。

该 1-episode smoke 的成功/碰撞结果仅用于工程链路验证，不作为方法效果证据。开发效果只由预注册的 D1–D4 fresh validation 实验判断。

## 可反驳条件

任一以下情况都会使审计失败并阻止冻结：继承调用链或哈希漂移、排序目标出现硬阈值/固定 margin、EMA teacher 获得梯度、模型动作被 domain clip 改变、目标速度与有效速度不等、车道码被重写、选择器改变 policy 参数、出现固定速度网格、规则元数据为真，或真实轨迹缺失。
