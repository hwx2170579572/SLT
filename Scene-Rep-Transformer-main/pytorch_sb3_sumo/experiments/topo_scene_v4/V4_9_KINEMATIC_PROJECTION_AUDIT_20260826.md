# v4.9 运动学安全投影与模型边界复核

状态：`interim_during_fresh_development_M3`  
复核日期：2026-08-26（Asia/Shanghai）  
适用约束：不得增加规则机制；通过完善可学习模型提升安全性  
Formal test：未访问，仍锁定

## 1. 结论先行

当前 v4.9 的碰撞监督、actor 风险目标和纯 `target_critic` 推理路径没有运动学
安全投影：

- 不根据 TTC、headway、相对速度或几何距离生成 unsafe 标签；
- 不在模型动作之后执行 lane veto、刹车覆盖或动作替换；
- 碰撞监督只来自实际闭环事件 `info["collision"]`；
- `target_critic` 只在合法动作 mask 内最大化学习分数
  `min(Q_reward) - lambda * max(V_collision)`。

但是，冻结候选集合还允许继承的 `fusion_0_90` 解码器。它不读取运动学量，
却通过固定 actor 置信度 0.90 决定是否覆盖 learned target 动作。旧审计将其记为
“继承的非安全行为”；在“不要添加规则机制”的最新约束下，这个分类不够严格。
因此：

1. 当前 M1--M4 继续完整运行，以保存未删减的开发和归因证据；
2. `fusion_0_90` 不进入下一版可晋级方法；
3. 下一版动作只能来自端到端学习策略或学习价值分数，不允许状态相关阈值、
   veto、投影或 post-decoder override。

## 2. 允许与禁止的边界

### 允许

- 将位置、速度、相对速度、TTC 等物理量作为网络观测特征；
- 使用真实环境碰撞终局作为监督；
- 使用神经网络输出的 reward value、collision value、策略分布和不确定性；
- 在环境提供的动作合法性 mask 内做 learned-score argmax；
- 仅为数值确定性使用与安全状态无关的精确并列 tie-break。

### 禁止

- TTC/headway/距离阈值触发的动作覆盖；
- lane-change veto、Q-regret veto、kinematic projection 或 shield；
- 用未来轨迹、oracle 轨迹或手工几何公式产生安全标签；
- actor 输出之后的规则式刹车、换道替换或 fallback；
- actor confidence 等固定阈值控制最终动作来源。

动作合法性 mask 是父方法动作空间的一部分，只排除物理上不存在的车道动作，
不根据碰撞风险作安全判断；它不属于安全投影。

## 3. 源码路径审计

| 路径 | SHA-256 | 观察 |
| --- | --- | --- |
| `algos/sb3_torch/replay_buffer_v4_9.py` | `c44dd06f5ecd3454f10886dede6a21b73028ae39196e1fb0422bfd7630e60660` | 标签只读 `info["collision"]`；16-step 折扣传播，无几何伪标签 |
| `algos/sb3_torch/sac_v4_9_model.py` | `3e0788d9f191fef05c2ac80e74113e21305de7159a6c2949b66959f77e729292` | actor 目标加入 learned collision value；没有动作覆盖 |
| `algos/sb3_torch/hybrid_policy_v4_9_model.py` | `cba4377d9da1a65c5ee45d61a9273c592fe261db284f4d97910e8a8d208198a9` | target 路径纯模型；fusion 路径仍含固定 0.90 来源门控 |
| `configs/sb3_configs_v4_9.py` | `4fa1b16d583359af64cf9865fdc2d2da3c39dfd6704ed019f3c3c57675b5bc34` | 候选显式包含 `target_critic` 与 `fusion_0_90` |

M1 和 M2 的 selector receipt 最终都选择
`CollisionConstrainedTargetCriticSACPolicyV49`，所以这两个已完成 validation
没有使用 0.90 fusion 门控。M3、M4 必须等待规范运行完成后再判断。

## 4. 模型层深层归因假设

代码检查显示：

1. actor 与 reward critic 共享状态编码器，该编码器由 reward critic 和 Graph-SLT
   目标训练；
2. collision critic 使用独立状态编码器，主要依赖稀疏碰撞 TD 监督；
3. actor 的状态特征在分布头之前 detach，风险损失只能更新 actor 动作头，不能
   让 actor 状态编码主动学习碰撞判别特征；
4. target decoder 可以用独立 collision encoder 排序离散车道动作，但连续速度
   候选仍由 actor 产生。

这形成一个可证伪的模型瓶颈：collision critic 可能能在碰撞临近时给出高风险，
但 actor 的速度提议没有稳定的碰撞表征；同时独立稀疏编码器会造成跨种子风险
尺度漂移。此时增加外部投影只能掩盖 actor/representation mismatch，不能修复
模型。

下一版优先验证“稳定共享表征 + 风险条件化动作提议”：

- collision heads 复用由密集 reward/Graph-SLT 学到的共享状态表征，并保留独立
  twin action-value heads；
- collision 目标仍只使用真实碰撞事件；
- actor 通过同一碰撞表征和可微 learned collision objective 学习速度与车道提议；
- 部署只使用 learned model score 或 actor 本身，不保留 fusion 阈值。

实现前必须把确切结构、优化器所有权、梯度路径和开发门禁写入新版本 contract，
避免两个优化器以独立状态重复更新同一参数。

## 5. M1/M2 中间证据

可复跑工具：
`tools/attribute_v4_9_learned_model.py`
（SHA-256
`0fab6a1f6a5004670712eab64859f5fc4f92af9ef4c943d702c2a873143c502e`）。
其 3 项定向单测已通过。

中间报告：
`results_topo_v4_9_dev/development/attribution/interim_m1_m2_learned_model.json`
（SHA-256
`4e233429847f89aa7148f2f103d9c36f2c79e47cfee056620e7627c83b4a3e69`）。

当前仅能支持以下有限观察：

- 两个 Cross 种子的训练期 policy expected collision value 均值分别约为
  0.1245 与 0.2011，尺度比约 1.62；
- sampled positive collision label rate 分别约为 3.28% 与 5.39%，标签频率
  差异不足以单独解释全部价值尺度漂移；
- 碰撞前 5 步，M1/M2 的 selected collision value 排序 AUC 均为 1.0，但 M1
  只有 1 个碰撞 episode，统计证据很弱；
- 碰撞前 5 步所选动作相对最低预测碰撞动作的平均 risk regret 很小（约
  0.0016 与 0.0009），说明失败更可能来自所有候选动作/速度提议都不够安全，
  而不是 learned decoder 忽略了一个显然更低风险的车道动作。

这些都是 observational attribution，不能证明替换动作即可避免碰撞，也不用于
新增 post-hoc 门禁。M3/M4 完成后必须重建四作业报告，再决定新版具体损失。

## 6. 后续阶段门控

1. v4.9 M1--M4 全部尝试并统一汇总；
2. 生成四作业完整轨迹归因和哈希清单；
3. 新建版本化模型 contract、实现、测试和冻结，不覆盖 v4.9 文件；
4. 新版 fresh development 全部通过后，自动尝试 12 个 promotion 作业；
5. promotion 通过前不得访问 120 个 formal 作业。

任何单作业失败都不取消同阶段剩余作业；阶段结束后统一汇总。
