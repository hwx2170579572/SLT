# v4.12 Augmented Joint-Support PRCR

## 单一方法思想

v4.12 将 v4.11 的“speed-only replay support”补全为联合动作支持模型。actor 对 replay 中实际 lane 计算 masked categorical NLL，并继续计算该 lane 下的速度-mixture NLL；部署时每个物理可行 learned proposal 使用同一个连续分数：

`min Q - collision value - 0.25 * twin disagreement + 0.05 * log p(component|lane,state) + 0.05 * log p(lane|state)`。

没有置信阈值。低概率 lane 不会被屏蔽，所有物理存在的 lane 仍参与普通 `torch.argmax`。lane probability、reward Q、collision value、uncertainty、component probability 和速度 proposal 都由网络产生。

Cross 训练启用现有的随机旋转表征增强，evaluation 时沿用既有上下文管理器关闭增强。RAM/CARLA 原本已经启用该增强，因此不改变它们的增强协议。Cross 的 Graph-SLT target encoder 选择保持 v4.11 不变，以隔离增强因素。

## 明确禁止

- 不读取 TTC、headway、前后车距离或几何 unsafe 标签；
- 不增加 traffic-risk lane mask、lane-change veto、置信门或 safety shield；
- 不设置 keep-lane tie 优先级；
- 不使用固定速度网格；
- 不在 decoder 后 clip、替换或重写动作；
- 不依据 scenario 名称在推理时改变分数或动作；
- 不使用 validation/formal outcome 训练或选择模型。

保留的 lane mask 只编码相邻 driving lane 是否物理存在。

## 开发与消融

Primary 在 Cross seed10/11、RAM seed12、CARLA seed13 上 fresh train。两项 Cross seed10 消融为：

1. `no_cross_augmentation`：保留 joint lane support，关闭 Cross 旋转增强；
2. `augmentation_only`：启用 Cross 增强，但移除 lane categorical NLL 和 lane log-prior，退回 v4.11 的 speed-only support。

开发先完整运行并汇总四个 primary 作业；只有 primary development 通过才运行两个消融。只有 development 和 ablation 都完成且通过协议检查才允许 fresh 12-job promotion。formal 仍锁定。

