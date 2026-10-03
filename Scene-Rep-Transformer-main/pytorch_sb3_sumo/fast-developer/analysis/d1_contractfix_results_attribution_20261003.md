# D1 C8/C9 修复实验：结果、任务作用与下一步

日期：2026-10-03。主场景：**intersection_sorted_depart4p0**。不混入 DARRL。

## 1. 结论

两路均完成 seed0、从零 100000 raw steps、95001 次更新和最终 100 回合 validation（seeds 10000–10099）。checkpoint 实算 SHA、训练终结记录和评估身份一致，奖励六分量对账通过。

**修复版 ST 为45成功/50碰撞/5超时；修复版 ST-RT 为52/48/0。相对各自旧实现的50/50/0、63/37/0，本次联合修复没有提高成功率。**

修复后的 RT 仍有任务作用：相对修复 ST，成功净增7个回合，5个超时消失，正常评估中没有记录到路线不可达 episode。但48个碰撞 episode 的最后可用自车位置均在内部连接器，终态路线可达；安全通过路口仍未解决。这个 ST→ST-RT 对照同时启用了 route 分支并改变有效计算与梯度路径；两版 encoder 名义注册参数量相同（1,057,348）不能证明启用 route 分支后的有效容量相等，也不能把观察到的净收益唯一归因于某个 route query。这里的“任务作用”仅指同场景、同验证 seed 池上的描述性结果。

ST 和 RT 均有正常 critic TD 梯度、真实参数更新和同状态动作敏感性，不支持“分支根本没接通”这一解释。功能依赖不等于闭环性能收益。当前更合理的下一步是先保证强化学习目标一致性，暂缓叠加表征模块；源码与离线 probe 确认当前 replay n-step target 对 k=1…4 均只乘一次 γ，与标准按实际 k 乘 γ^k 的 Bellman 目标存在已确认的协议差异。项目注释称单 γ 是为兼容 released SAC behavior；本地审计未独立追溯该 upstream release，因此不把它定性为已证实的上游 bug。是否调整协议及调整后是否提高成功率仍需单独判断和实验。本文没有启动任何额外训练或仿真。

## 2. 最终结果及公平性

| 方法 | 成功 | 碰撞 | 超时 | off-route终局 | shaped回报均值 | raw未折扣回报均值 |
|---|---:|---:|---:|---:|---:|---:|
| 旧ST | 50 | 50 | 0 | 0 | 不混列旧日志回报 | — |
| ST contractfix | 45 | 50 | 5 | 0 | 0.79356204 | -0.05 |
| 旧ST-RT（sortlr_1003_retry01） | 63 | 37 | 0 | 0 | 4.3898484 | 0.26 |
| ST-RT contractfix | 52 | 48 | 0 | 0 | 2.02868299 | 0.04 |
| 旧ST-RT + Longres | 39 | 55 | 6 | 0 | -0.74052338 | -0.16 |

新版 shaped 回报为 environment_step_reward_v2，raw 回报另列。ST 的 shaped/raw 标准差分别10.54762652/0.97339612，ST-RT分别10.74453241/0.99919968；这是评估episode间离散程度，不是训练seed方差。

| 平均奖励分量 | ST contractfix | ST-RT contractfix |
|---|---:|---:|
| success | 4.5 | 5.2 |
| collision | -5.0 | -4.8 |
| off-route | 0 | 0 |
| timeout | -0.25 | 0 |
| step-cost | -0.8644 | -0.8773 |
| progress | 2.40796204125 | 2.5059829937 |
| 总和 | 0.79356204125 | 2.0286829937 |

两路六分量均覆盖100回合，最大对账误差均3.55e-15。ST的5个超时不能误记为5个off-route。

工作区：D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1。两个新增运行目录：

- runs/d1_contractfix_20261003/st/sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0
- runs/d1_contractfix_20261003/st_rt/sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0

ST最终checkpoint SHA-256：AF0D6674ADF321E148DB54D3D48B8349D558E40101C229FDDA9D13EFFE74B162。

ST-RT最终checkpoint SHA-256：706548B4E84450ADCE1C224C122D88F0C0647B1F212A695A9558C4FE00A55D0F。

两路CUDA worker正常退出，suite完成；20文件归档ZIP SHA为b86378d2230f339757c463d4bbae98259fcf075c9919189203ac931714ec9f50，runtime来源已匹配归档。30份交通模板SHA一致，输出目录隔离。

旧ST与新ST的配置记录可对齐场景、seed0、fresh100k、5000 warmup、batch32、LR1e-4、buffer20k、gamma0.99、action-repeat3、训练奖励及评估seed池。旧保存频率200，新10000；新版增加只读诊断。旧ST没有完整源码/runtime归档，因此不能逐字节排除所有历史差异。对应编码器参数量同为1,057,348，参数初始化及初始化后RNG一致的测试已通过。

联合C8/C9修改不能单独识别每项性能贡献，一个训练seed也不足以证明稳定优势。旧ST-RT的63/37/0重跑复现是同一seed，不应冒充独立训练seed。

## 3. 逐seed结果转换：有救回，也有损失

三对均按10000–10099连接，100/100匹配，traffic variant一致，无多终局标记冲突。动作不同后轨迹不同；下表是同场景seed的描述，不是同状态干预因果。

| 来源→目标 | 碰撞→成功 | 成功→碰撞 | 成功→超时 | 碰撞→超时 | 来源超时去向 |
|---|---:|---:|---:|---:|---|
| 旧ST→修复ST | 23 | 25 | 3 | 2 | 无 |
| 旧ST-RT→修复ST-RT | 20 | 31 | 0 | 0 | 无 |
| 修复ST→修复ST-RT | 29 | 25 | 0 | 0 | 3成功、2碰撞 |

联合修复没有达到“更多碰撞转成功，同时不增加超时”的整体目标。修复后的RT仍有净收益，但损失25个修复ST原成功案例，不能称为对所有场景都更安全。

## 4. 修复是否真实生效

### C8：错误存在，出现频率与性能影响必须分开

| 阶段 | social历史记录数 | 旧索引会错位 | 比例 | 旧选padding | 旧选更早有效帧 |
|---|---:|---:|---:|---:|---:|
| ST train | 167420 | 113 | 0.0675% | 61 | 52 |
| ST eval | 43220 | 22 | 0.0509% | 12 | 10 |
| ST-RT train | 167500 | 88 | 0.0525% | 49 | 39 |
| ST-RT eval | 43865 | 33 | 0.0752% | 19 | 14 |

分母是正常decision中social槽的重复观测，不是独立邻车数。“旧索引会错位”是对已记录mask的反事实计算，不代表修复后仍选错。ego在这些汇总中未出现旧索引错位。

短历史不能一概称为左补。collector的left_padding表示左侧padding、有效suffix；right_padding表示有效prefix之后有padding。ST train左补113、右补6960，ST-RT train左补88、右补6930，均无内部缺帧。多数短历史为右补，count-1在连续有效prefix上本来就正确；少量左补才产生本次错位。

具体样本：ST train episode0 decision97，preaction cache tick292，一个social actor在step290首次出现，cache length/age均2；汇总显示左补、旧选padding、index lag=8。它与左侧8个padding后接2个有效帧一致：count-1选index1，真实末位index9。日志未保存完整10-bit mask，这是由摘要与缓存证据重建。

mask仍为x!=0代理，本批没有x=0而其他字段非零记录，不证明该代理普遍等价于真实presence。错误确实存在，修复应保留；低频也不证明影响小，因为事件可能发生在关键时刻。现有证据不能把性能下降归因于C8，也不能将历史退化全归咎于它。旧query选错不意味着attention无法继续汇聚有效K/V。

### C9：广泛改变了几何信号，尚不能拆分性能归因

SMARTS Heading 0朝北、逆时针；旧速度通道(s cos h, s sin h)不能直接作为east/north速度。几何边转换为(-old_vy, old_vx)，学习输入的旧通道保持不变；解析案例和布局测试已通过。

ST train的4,894,715个pair-history观测中，closing符号反转2,505,431个，旧零变新非零310,345个；eval对应1,269,115、654,267、77,025。它们是重复几何观测，不是独立冲突事件。

相对稀少C8错位，C9带来更广泛的直接数值变化。这支持“几何输入变化广泛”的判断，**不证明性能下降由C9导致**：两项修复联合启用，训练策略、采集数据和优化轨迹也随之改变。

恒速最近接近时间t*=-r·v/||v||²，零相对速度为NA；还需最近距离、车身占用及未来时域才能讨论碰撞风险，不能把t*直接称为真实TTC。CPA和另一项CV-OBB TTC诊断也不是同一个量。

正确合同是实现要求，提升成功率是实验命题。不能因旧版本分数高就恢复错误基底，也不能因公式修正就宣布性能提升。

## 5. ST、RT确实在训练和影响动作，但解决了什么

### 学习与功能证据

正常representation日志中，两路各96个critic TD梯度采样和96个参数更新采样。state/spatial/temporal/social的梯度覆盖完整、无非有限梯度，96/96有非零参数更新。ST-RT的map_route/route_path/route_goal也96/96更新；ST关闭路线模块，其相应更新为零符合配置。

这是critic TD路径的证据，不是actor loss直接更新编码器的证据。具体actor feature detach和共享提取器路径应按实现地图解释，不能仅由通用SAC论文推断。

eval shadow对同一状态计算两个归一化动作向量之差的L2均值，字段为action_delta_l2_normalized。动作Box为[-1,1]二维，差值无量纲，理论范围0到2√2；不是m/s。

| 同状态probe | ST均值（357状态） | ST-RT均值（298状态） |
|---|---:|---:|
| spatial_off | 0.6623 | 0.9035 |
| temporal_current_only | 0.8929 | 0.7559 |
| social_ego_only | 0.8219 | 0.4751 |
| rt_intent_injection_off | NA | 0.7006 |
| rt_route_readout_zero | NA | 0.1074 |

上述适用样本动作差均非零。大幅关分支可能使表示离开训练分布；这些数值验证功能依赖，不直接证明模块的性能贡献。

train shadow各20唯一状态/280 variant行，无active-invalid或错误；eval ST为357状态/4998行，ST-RT为298/4172，均覆盖100回合、每回合不超过4状态，没有增加仿真回合。

### ST：仍有路线停滞与随后冲突

ST的5个超时均600raw/200decisions，停驶比例70%–81.8%，路线不可达telemetry累计42.6–49.8秒且大多停着。终态并不相同：2个在-E1 lane0/1，known_lane_cannot_reach_next_edge；1个在-E0_2，route_has_no_next_edge；2个仍在内部连接器。不能都称为选错车道。

50个碰撞episode中32个曾记录路线不可达；但最后可用位置均在可达内部连接器：:J1_21_0有41个、:J1_14_0有9个。路线停滞和碰撞可出现在同一episode，并非两个互斥根因。

### RT：路线一致性改善，安全穿越未解决

修复ST-RT没有路线不可达episode或超时；48个碰撞末次可用位置全在上述连接器（40/8），终态可达、规划edge为-E1。该现象与路线条件有助于避免路线停滞、维持路线一致性相一致；不能称RT预测了社会车真实未来意图。ST→ST-RT还启用了route参数组，改变了有效前向计算和critic梯度路径；名义注册参数相同不能说明有效容量相等，也不能把结果唯一归因于某个route query。该对照只是同场景验证seed池上的描述性结果。

碰撞episode条件均值：ST耗时21.48秒、速度4.66m/s、停驶比例24.3%；ST-RT分别23.49秒、3.73m/s、2.3%。两组碰撞案例不同，不能据此断言降速或停驶改变导致碰撞；也不支持“整体再慢一点就能解决”的简单结论。

ST的50条collision evidence没有SUMO collision_ids/events。上述位置是最后可用自车/终端状态，不是确认的impact坐标；具体碰撞对象与接触姿态仍有限制。风险诊断覆盖不等于接触事件身份已确认。

### 计数边界

train审计：ST33484行/100000raw、463 completed；ST-RT33500行/100000raw、461 completed。behavior汇总还包含关闭时未终止的partial episode，不能将episodes_finished直接当completed；ST最后一条审计行raw_delta=0，属于预算边界调用，不增加仿真步。eval各100 completed：ST8644decision/25832raw，ST-RT8773/26218；错误及unknown raw均0。完整口径见诊断审计。

## 6. 放回逐模块研究链条

以下保留同sorted/depart4历史，均为单训练seed的描述性参照。旧源码、奖励日志及实现局限见原汇总。

| 方法 | S/C/T | 对任务的有限判断 |
|---|---|---|
| SAC+MLP | 22/38/40 | 超时是纯基线的重要问题 |
| MST+SLT | 53/47/0 | 强基线参照；其first-frame mask问题不能混作D1 C8 |
| 旧ST | 50/50/0 | 完成能力提高，同时碰撞也增加，不能只称更安全 |
| 旧ST-RT | 63/37/0 | 历史中较强的简单配置，保留旧合同局限 |
| ST-RT+Topo | 39/39/22 | 超时明显；多条前向/梯度路径耦合，非单一关系消融 |
| Topo+路线可达性 | 43/32/25 | 碰撞降低伴随超时增加，成功率未改善 |
| 线性3slot | 34/49/17 | 槽划分同时改变头部容量、非线性 |
| Topo+3slot | 41/56/3 | 超时减少不能抵消碰撞增加 |
| goal-only Topo | 24/47/29 | 关分支能解除停滞，但很多案例转为碰撞 |
| 等参数非线性3slot | 50/48/2 | 部分恢复，未证三槽语义收益 |
| RouteAct | 47/53/0 | 路线执行约束不等于冲突安全约束 |
| ConflictTiming | 40/57/3 | 信息/分支有使用证据，成功率未提升 |
| Longres | 39/55/6 | 速度残差和Q分支真正在学习，本轮没有收益 |
| 修复ST | 45/50/5 | 合同正确后仍有停滞、碰撞 |
| 修复ST-RT | 52/48/0 | 避免停滞后，路口冲突仍是瓶颈 |

已有有限窗口干预表明决策时机值得研究：选定案例中，入口前或进入时强制速度会改变结局，同一固定动作也会损害原成功案例。累计85回合/22944控制raw的预算已结束，不再启动。它不证明信用分配就是根因，也不证明有界Longres能学会那些多数超过约0.997m/s差值范围的干预。

## 7. 下一步：学习目标优先，避免继续无依据叠模块

### 优先验证实际k步bootstrap折扣

保留修复ST-RT、参数量、奖励、预算与seed池。若之后用户授权完整实验，最小候选只改replay窗口末端折扣：

**y_k = Σ(i=0..k−1) gamma^i r_i + gamma^k (1−terminal) V_soft(s_next)。**

当前实现末项统一用单个gamma。gamma=0.99、k=4时应为0.96059601，而不是0.99；短尾必须用实际k。目标差为(gamma−gamma^k)V_soft，方向取决于soft value符号，不能无条件称为Q过估计。

先做不调用环境的target/replay验证，覆盖k=1..4、真实terminal、truncation、短尾。性能比较暂保持timeout语义不变，避免混入第二变量。若后续采纳新RL训练协议，论文主比较中的SAC+MLP、MST+SLT等方法也必须按同一协议对齐；旧历史表仍按既有协议保留，不能视作已与新协议公平对齐。本轮未授权或启动这类重训。公式修正不保证成功率提高；**本报告没有执行该新训练**。

未来正常训练可记录实际k分布、bootstrap生效比例、奖励和末端soft value对target的贡献，以及临近路口/碰撞/超时状态在replay和更新中的覆盖。只有证据显示关键状态缺少更新或目标失真，再提出有针对性的回放/奖励改动；不能从“表示被使用但仍撞”直接推断回放失效。

### 再明确60秒的任务定义

若60秒是任务内在期限，应将剩余时间作为状态信息，期限结束作为terminal不再bootstrap；若只是采样外部截断、任务之后仍继续，则truncation继续bootstrap合理。字段名timeout本身不能决定语义。当前超时惩罚、剩余时间缺失与bootstrap组合需按研究任务统一。

这一调整会改变观测/任务合同，应另建协议，不与k步修正、奖励重塑一次捆绑。

### 当前不优先做

- 不再仅增加Topo、槽头或另一种冲突输入，现有多轮实验尚未证明收益。
- 不直接施加统一减速倾向；入口等待与进入后清空的需求不同，已有干预显示救回与伤害并存。
- 不把性能下降归结为一个未经识别的唯一根因。联合修复、策略改变后的采样分布、单seed与旧源码归档不足均限制归因。
- 若目标是分别量化C8/C9的训练影响，需要另行授权单因素实验；冻结同状态probe不能代替重训消融。

## 8. 理论依据与来源边界

- [SMARTS坐标官方文档](https://smarts.readthedocs.io/en/latest/api/smarts.core.coordinates.html)：支持Heading朝向/角度合同；项目通道换算还依据状态构建源码及解析测试。
- [Interaction Networks](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html)、[GAT](https://arxiv.org/abs/1710.10903)：支持对象关系/邻域加权聚合动机，不证明本项目安全通过收益。
- [SAC原论文](https://proceedings.mlr.press/v80/haarnoja18b.html)：支持奖励与熵的目标；本项目编码器梯度路径必须由源码确定。
- [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)、[Gymnasium时间限制文档](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：支持有限时域与外部截断的区分，不能替研究者决定60秒语义。
- [Shielding原论文](https://ojs.aaai.org/index.php/AAAI/article/view/11797)：形式化规格下的动作监视/修正；路线key mask或简单veto不能据此获得同等安全保证。

## 9. 原始证据入口

本轮各方法目录的training_complete、evaluation_results、experiment_manifest、runtime provenance、最终模型及diagnostics/train、diagnostics/eval为直接来源。实际嵌套路径和字段详见：

- [修复协议](d1_contractfix_protocol_20261003.md)
- [诊断审计](d1_contractfix_diagnostic_audit_20261003.md)
- [公平比较与逐seed转换](d1_contractfix_comparison_evidence_20261003.md)
- [原分场景交叉口汇总](intersection_experiment_summary_20261001.md)
- [实验历史](experiment_history.md)
- [Longres诊断](longres_training_diagnostic_audit_20261003.md)
- [bootstrap审计](bootstrap_protocol_audit_20261003.md)
- [有限窗口结果](decision_window_intervention_results_20261003.md)

两条获授权新训练及其最终评估已完成；没有追加训练、窗口预算或仿真。完成本报告及记录核验后暂停automation-3，避免重复处理。
