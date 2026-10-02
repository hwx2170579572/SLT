# v4.8：Tie-Only Replicated Train Calibration

状态：`preregistered_before_implementation_and_fresh_development`  
父版本：`v4.7_horizon_correct_16_step_terminal_credit`  
CCFA 模式：`standard / generic CCF-A AI-ML venue family`

## 1. 失败事实与直接归因

v4.7 G1 在 Cross seed 6 上相对同 seed 的 v4.6 父控制取得 `+2/12`
成功数并通过配对因果门禁，但独立训练 seed 8 的 G3 只有 `5/12` 成功、
`6/12` 碰撞、`1/12` timeout，因而按预注册规则停止。

G3 的四个 checkpoint×decoder 组合在首个 12-episode train 校准块上均为
`8/12` 成功、`4/12` 碰撞，旧选择器遂使用确定性偏好选择
`exact_final + target_critic`。在完全相同的 validation episodes 上，合法但
未选中的 `exact_final + fusion_0_90` 达到 `8/12` 成功、`4/12` 碰撞：相对
旧选择三例失败转为成功、零例成功退化。

两套权重×两个 validation seed blocks 的交叉回放进一步显示：G1 权重为
`8/12`、`9/12`，G3 权重为 `6/12`、`5/12`；两个 block 的平均成功数相同。
因此 validation block 难度不是主因。G3 的训练 seed 权重态仍有上游风险，
但本 cell 的直接必要失败原因是有限样本校准完全平局后使用了继承的 target
偏好，而不是“没有任何合格部署组合”。

## 2. 单一科学变化

只改变候选方法的 train-only 选择器在“数据项完全平局”时的证据采集：

1. 首轮仍对 4 个组合运行完全相同的 12 个 train episodes；
2. 排序的数据项仍依次为 success count、collision count、off-route count、
   timeout count、mean return；
3. 若首位仅一个组合，立即选择，绝不构造第二校准环境；
4. 若两个或更多组合在上述五个数据项上完全并列首位，仅对这些并列组合
   使用预注册的第二个 12-episode train seed block；
5. 对并列组合合并为 24 episodes 后按相同五个数据项重排；若仍完全平局，
   才使用旧的 `prefer exact-final`、再 `prefer target-critic`；
6. 初轮非首位组合不能通过额外样本复活；所有参与同一轮的组合保持相同
   simulator seeds 与 traffic variants。

第二轮复制相同的冻结 train-variant 支撑，但使用不重叠的 SUMO seeds；它是
随机性复制而不是新增 traffic-variant 覆盖，报告中必须如实区分。

## 3. 严格保持不变

- v4.7 的 16-step return、`gamma^actual_horizon`、terminal/done 语义；
- reward、observation、lane mask、hybrid action 与 action repeat；
- 拓扑编码器、Graph-SLT、网络宽度、actor detach 和表示损失；
- 学习率、batch、buffer、warm-up、20k 开发训练预算；
- exact-final / highest-training-success 两个 checkpoint；
- target-critic / fusion-0.90 两个 decoder 的定义和完整性门禁；
- 首轮 12-episode train 校准与 validation/formal 分区；
- outcome 硬门禁、promotion、formal test 锁与最终接受标准。

TemporalGraph 继续使用自己的冻结 checkpoint-only selector，不接受候选的
tie-only 第二校准机制。

## 4. Claim–Evidence Matrix

| Claim | 主要质疑 | 必需证据 | 单元 | 状态 |
| --- | --- | --- | --- | --- |
| 平局复制可修复已失败权重 | 是否只在旧 validation 上事后有效？ | seed8 从头训练；已知 train 校准块；全新 validation block | H1 | planned |
| 不是仅记住 seed8 | 新 seed 是否仍可选出合格组合？ | Cross seed9 全新训练/校准/validation | H2 | planned |
| 不破坏安全场景 | fusion 候选会否增加 CARLA 碰撞？ | CARLA seed5，collision≤0.10 | H3 | planned |
| 不只适配 Cross | 第三场景是否退化？ | roundabout-medium seed3 | H4 | planned |
| 无 validation 泄漏 | 第二环境是否只在平局且先于 validation？ | 收据时序、seed/variant 签名、未触发时环境不存在 | engineering | planned |

## 5. 开发顺序与停止规则

所有科学训练串行执行；任一必需门禁失败立即停止并新建版本归因。

1. `H1`：Cross seed 8，20k；primary calibration `77000--77011`；
   tie-only replication `77100--77111`；全新 validation `68100--68111`。
   这是直接救援确认；校准块来自假设生成，validation 从未访问。
2. `H2`：Cross seed 9，20k；primary `78000--78011`；secondary
   `78100--78111`；validation `69100--69111`。
3. `H3`：CARLA seed 5，20k；primary `79000--79011`；secondary
   `79100--79111`；validation `70100--70111`。
4. `H4`：roundabout-medium seed 3，20k；primary `80000--80011`；
   secondary `80100--80111`；validation `71100--71111`。

非 CARLA：success≥0.50、collision≤0.45、timeout≤0.30；CARLA：
success≥0.50、collision≤0.10、timeout≤0.50；全部 off-route=0。decoder、
mask、selector receipt 与时序完整性不可补偿。

H1 必须额外证明：若首轮完全平局，则第二轮确实执行、只包含并列首位组合、
合并排序与收据一致；若因训练非确定性导致首轮不再平局，则按规则不强制特定
decoder，但必须通过 outcome gate。H2 是对事后设计风险的不可补偿新 seed
确认。

## 6. Promotion、Formal 与最终目标

仅 H1–H4 全通过才进入原 promotion：`2 methods × 3 scenarios × 2 seeds`
的 50k validation 比较。promotion 全部门禁通过后才解锁 formal：6 场景、
10 seeds、每格 100k、50 test episodes。

最终接受标准不变：每场景 success delta≥-0.05、collision delta≤+0.05，且
至少一个场景 success delta≥+0.10 或 collision delta≤-0.10。

## 7. Reviewer-Risk Register

| 风险 | 类型 | 控制 |
| --- | --- | --- |
| 看过 G3 validation 后设计 v4.8 | requires-new-result | H1 使用全新 validation；H2 使用全新训练 seed；旧回放不进 gate |
| 第二轮重复相同 train variants | evidence-fixable | 明确称 simulator-seed replication；保存全部签名；要求跨场景新结果 |
| 自适应样本量带来选择偏差 | design-fixable | 仅完全并列首位可进入第二轮；非首位不可复活 |
| fusion 可能在 route window 外不安全 | requires-new-result | outcome 排序优先；CARLA collision 硬门禁；保留完整动作 trace |
| 额外评估成本 | evidence-fixable | 只在平局触发；报告触发率、额外 episodes 与 wall time |

## 8. No-Fabrication Status

本文已报告数值来自封存的 v4.7 开发或明确标记的 post-hoc 真实闭环回放。
H1–H4、promotion、formal 的未运行结果均为 `TBD`，不得推断或预填。

