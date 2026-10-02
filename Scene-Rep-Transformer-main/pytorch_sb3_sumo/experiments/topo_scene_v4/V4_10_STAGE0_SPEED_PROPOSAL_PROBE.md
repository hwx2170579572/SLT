# v4.10 阶段 0：learned-score 速度提议覆盖 probe

状态：预注册并实现，尚未读取结果  
训练任务：0  
Formal test：不访问

## 决策问题

v4.9.2 的 6/6 配对归因显示，碰撞 episode 中部署动作经常已经是三个 actor
车道提议里的最低预测碰撞值。该事实不能区分：

1. collision critic 不能表征真正的风险；
2. critic 能识别风险，但 actor 每车道只提供一个连续速度，安全候选不在集合中。

因此在冻结 v4.10 前，对六个冻结候选 checkpoint 运行只读 probe。原策略仍在相同
validation seeds 上闭环执行；每个访问状态额外查询固定速度网格
`[-0.95,-0.75,-0.50,-0.25,0,0.25,0.50,0.75,0.95]`，对应目标速度
`[0.25,1.25,2.50,3.75,5.00,6.25,7.50,8.75,9.75] m/s`。

## 不可越过的边界

- 网格动作只查询冻结 learned reward/collision target critics，绝不发送给 SUMO；
- 不修改 checkpoint、源 run tree、奖励、标签、动作 mask 或控制器；
- 网格不是拟部署模块，不是速度阈值、投影、shield、veto 或动作重写；
- counterfactual 分数只证明 critic 自洽性，不能证明闭环安全；
- 六个配对都尝试，任一失败不取消后续配对，最后统一汇总。

## 主要观测量

对原部署选中车道，比较 actor 速度与该车道诊断候选中的最高 learned
`min_reward_q - max_collision_value`：

- 更高分速度候选的出现率与分数增益；
- 同时降低 learned collision value 且提高总分的比例；
- 候选速度变化；
- collision episode 最后 1/5/10 个决策的对应量；
- 全局车道×速度候选是否改变 learned argmax。

## 后继方法选择原则

- 若退化配对的碰撞终端窗口普遍存在同车道、更高总分且更低 learned risk 的速度，
  优先完善 learned risk-conditioned speed proposals；
- 若退化配对主要表现为 collision value 分离失败、而速度候选没有可利用余量，优先
  让碰撞监督进入共享表征并消除独立 encoder 的随机失配；
- 若两种证据同时成立，v4.10 可以采用一个统一的 learned risk-conditioned proposal
  模型，但必须通过消融分别验证共享风险表征与动作提议覆盖；
- 任何分支都不得添加手工安全规则。
