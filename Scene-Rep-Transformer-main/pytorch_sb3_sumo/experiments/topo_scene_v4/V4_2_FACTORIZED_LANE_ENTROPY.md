# v4.2：Factorized Lane Entropy（确定性 lane 边际稳定化）

状态：`preregistered / implementation_pending`  
父版本：`v4_1_action_head_only`  
失败证据：`results_topo_v4_dev/attribution/v4_1_seed_instability/`

## 单一科学变化

v4.1 使用同一自动温度同时奖励 categorical lane 熵和 conditional-speed differential entropy：

`E_k[alpha * (log pi_lane(k|s) + log pi_speed(v|s,k)) - Q(s,k,v)]`。

v4.2 改为：

`E_k[alpha_speed * log pi_speed(v|s,k) - Q(s,k,v)]`。

- `alpha_speed` 仍自动学习，目标熵从二维联合接口的 `-2` 改为一维速度接口的 `-1`；
- lane categorical 熵系数固定为 `0`；
- lane 行为探索仍由前 5,000 原始仿真步的 mask 内均匀离散 warm-up 提供；
- actor 和 target value 仍精确枚举三个 lane intent。

目标不是让策略更激进，而是让有两个可行动作的状态形成由 Q 排序决定的确定性正边际，消除随机训练成功与确定性评估失败的错配。

## 严格保持不变

- v4.1 的 observation、物理动作 mask、hybrid actor、lane one-hot critic；
- v2 soft topology encoder、Graph-SLT、`lambda_balance=0.01`；
- actor 对共享 encoder 的 detach；
- reward、4-step replay、buffer、batch、学习率、discount、action repeat；
- traffic split、episode limit、raw-step clock；
- 无 route-intent oracle 输入，无 `L_intent`、`L_var` 或奖励塑形。

## 开发阶段与停止规则

使用未在 v4.1 门禁中访问的新 validation episode block：`42000 + 1000*training_seed`。

1. F1：CARLA seed 1，20k，直接复核已失败 seed；
2. F2：F1 通过后 CARLA seed 0，20k，检查是否破坏原成功 seed；
3. F3：F2 通过后 Cross seed 0，20k；
4. F4：F3 通过后 Cross seed 1，20k。

CARLA 沿用原结果硬门禁，并新增不可补偿机制门禁：

- route-window match rate >= 0.80；
- mean `(p_required_nonkeep - p_keep)` >= 0.05；
- 每个观察到的 route window 的 margin 必须为正。

任一 CARLA 结果门禁或机制门禁失败即停止 v4.2。Cross 沿用原回归守卫。v4.1 D1/D4 是 joint-entropy 父实现消融，不追加重复训练来补偿。

## 反证解释

- 若 critic 的 required-lane Q 仍转为负边际，说明问题主要是稀疏信用分配/分支样本不平衡，而非 categorical 熵；下一版本应只研究 decision-window replay/credit assignment。
- 若 Q 为正但 actor margin 仍不稳定，检查实现公式、梯度或温度更新，不引入 route 标签。
- 若 CARLA 稳定但 Cross 碰撞/超时退化，说明去除 lane 熵导致探索或安全工作点改变，不能进入 promotion。
