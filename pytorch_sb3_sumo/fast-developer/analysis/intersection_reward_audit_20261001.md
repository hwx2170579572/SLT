# 交叉口历史实验奖励审计（2026-10-01）

> **口径范围说明：**本审计对应修复前的38条历史评估文件及其当时schema。文中`evaluation mean_return`为该历史快照的raw undiscounted `success−collision`，不代表后续当前sorted主队列的评估字段。自2026-10-01 sorted主阶段起，评估主`episode_return/mean_return`累加环境实际返回的shaped decision reward并标记`environment_step_reward_v2`；raw return另存为`raw_episode_return`、`raw_mean_return`。历史文件、数字和原有分析均保留，不做追溯改写。

## 结论和范围

对应 [38 条历史评估汇总](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_experiment_summary_20261001.md>) 的固定快照，不增加新训练、评估或新的结果行。

**可核实的奖励参数采用同一套 v2 设计；没有发现为不同方法或密度有意换用另一套奖励系数。不能将全部 38 条直接写成“实际奖励实现完全相同”。** 必须区分：

1. 参数文件/奖励分量支持同一套 v2 的实验；
2. DARRL r1 两条：系数相同，但成功/碰撞双标导致真实训练奖励污染；
3. 旧 MST 50k 与 7 条续训：缺少充分历史奖励配置/源码，未发现尺度异常，但尚不能完整核实；
4. 环境训练奖励、评估 raw return、辅助表征损失三个不同概念。

“历史资料不足”不是发现奖励不同；“参数相同”也不是已证明旧环境终止实现与当前完全相同。

## 1. 已核实的 v2 训练奖励设计

每次策略决策（一次高层 env.step）的奖励为：

```text
r_train(k) = b(k) - 0.01 + progress(k)

progress(k) = 0.02 * (D(k) - D(k-1))    前后两次距离都可读取时
            = 0                       否则

b(k) = +10   成功
       -10   碰撞
       -10   偏离路线
        -5   超时
         0   没有终止事件
```

终局表按一个已选定的终局事件解释，不应对 r1 的原始重叠布尔标志直接套用线性求和。

- D 是 TraCI `vehicle.getDistance(ego_id)`：自车插入后累计行驶距离，单位米；不是到目标剩余距离的减少量，也不是路线可达性得分。
- 当前代码对距离差没有正负裁剪或上下限；通常累计里程非减，不等于代码做了 max(0, delta)。
- 每决策步固定扣 0.01，终止决策也扣。若终止后自车已被移除、当前距离不可读，该决策进度项为 0，终局项和时间成本仍执行。
- 当前底层 action_repeat=3，故一次决策最多推进三个 raw tick；终止可能提前打断。不能把 −0.01 写成每 raw tick 成本。
- 已核实的 v2 配置没有单独 TTC、换道、Topo 或 3slot 奖励项。
- 当前源码的 wrapper 仍按 success、collision、off_route、max_time 的 if/elif 分支顺序选择终局项；r2 环境先产生互斥标志，因此实际终局优先级由环境的 collision > off_route > success > timeout 决定。

当前源码依据：
- [reward_shaping_v2.py：参数](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/reward_shaping_v2.py:55>)、[每决策计算](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/reward_shaping_v2.py:79>)、[距离读取](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/reward_shaping_v2.py:121>)。
- [基础入口的奖励配置](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/train_intersection_yield_v2.py:158>)。
- [D1 入口及环境工厂](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/train_intersection_yield_v2_d1.py:489>)。
- [SUMO 决策重复与原始奖励](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/../envs/sumo/sumo_env.py:563>)、[r2 终局解析](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/../envs/sumo/sumo_env.py:1461>)。

这些源码链接解释当前设计；历史执行结论必须结合下表的运行证据，不能仅靠当前默认值倒推。

## 2. 按原实验组区分奖励证据

| 原汇总中的实验组 | 条数 | 奖励参数/执行结论 | 必须保留的限定 |
|---|---:|---|---|
| sorted depart4，从零 50k | 9 | SAC/D1 八条有显式相同 v2 参数；MST 50k 仅有总训练回报支持相同尺度 | MST 50k 缺奖励字段和匹配历史源码，不能完整核实；本组未发现训练终局重叠 |
| sorted depart2.5 / depart3，从零 50k | 2 | 两条 SAC 显式记录相同 v2 参数 | 场景密度变化，不是已记录奖励系数变化；历史源码未冻结 |
| sorted depart4，50k→100k 续训 | 7 | 监控回报与来源阶段形态相符，未发现明显尺度变化 | 续训记录缺 reward、action_repeat、gamma；不能声称续训奖励协议已完全核实 |
| sorted depart4，从零连续 100k | 6 | SAC/D1 有相同 v2 参数；MST 100k 六项奖励分量支持同一设计 | FD 本地 MST/ST 没有匹配历史源码；相同参数不代表全部历史终止细节已证明相同 |
| 原随机 medium 与 p=.5/.3/.2 三个变体，从零 100k | 4 | 参数/训练分量支持同一 v2；未发现训练成功与碰撞双标 | 发车机制/交通状态变化；不应解释为给不同 p 设置了不同奖励系数 |
| DARRL r1 low / medium，从零 100k | 2 | **v2 系数相同，但成功/碰撞双标时训练走成功 +10，漏掉碰撞 −10 分支** | **真实执行污染，不能作为修复后有效基线；评估重新计数无法修复训练** |
| DARRL r2 low / medium，从零 100k | 8 | v2 参数与奖励分量一致；当前七方法队列源码归档支持相同 wrapper；使用互斥终局 | collision > off_route > success > timeout；只包括原快照已完成的八条，未填充当时未完成的 Topo+3slot |

计数：9+2+7+6+4+2+8=38。

本地 20 条中：11 个 fresh SAC/D1 参数文件明确记录六个 v2 系数；两个 fresh MST 未记录 reward 字段，其中 100k MST 有六分量 Monitor 证据，分量和与总训练回报最大误差不超过 5e-7。13 个 fresh 本地运行均记录 action_repeat=3、discount=0.99；7 条续训未独立记录这些字段。20 份本地 Monitor 未发现已完成训练回合的成功/碰撞/超时重叠。

runs 下 18 条正式结果的参数、奖励分量和归档支持同一套 v2 数值设计；除 DARRL r1 外未发现训练成功与碰撞重叠。现存当前 wrapper、DARRL r1 归档和 dm7 retry02 归档的 wrapper 文件字节相同；这不意味着 18 条运行都各自拥有完整冻结源码，源文件相同也不等于环境给它的终局标志相同。

## 3. DARRL r1 是真实执行差异

r1 仍可同时给出 success=true 和 collision=true。由于奖励 wrapper 先判断 success，该决策的终局奖励取 +10，碰撞的 −10 分支不执行。相对于按碰撞处理，终局项相差 20。

| 旧场景 | 训练已完成回合的成功/碰撞双标数 | 最终评估双标数 |
|---|---:|---:|
| r1 low，p=.03 | 29 | 3 |
| r1 medium，p=.05 | 10 | 5 |

这是同一名义奖励配置在错误终局输入下的有效奖励差异。r2 通过环境层互斥终局处理修复，不是简单把奖励权重换了一套。

旧 sorted/原随机阶段没有观察到同样的已完成训练回合双标，不能将 r1 的已观察污染数量泛化给所有旧实验；也不能因为未观察到双标就证明旧环境已有 r2 resolver。

## 4. 训练回报与评估回报不同

评估器优先累加 info[`undiscounted_reward`]，即底层原始 success − collision；只有缺少该字段才 fallback 到 env 返回 reward。本项目所审计正式评估使用 raw return。

在互斥终局下：

| 最终结局 | 训练终局项 | 评估 episode return |
|---|---:|---:|
| 成功 | +10 | +1 |
| 碰撞 | −10 | −1 |
| 偏离路线 | −10 | 0 |
| 超时 | −5 | 0 |

训练还有每决策成本和行驶距离项；右列没有它们。因此评估 `mean_return = success_rate - collision_rate`（比例按 0–1 计），不是训练 shaped reward 的平均值。

例如当前 DARRL r2 中档：SAC+MLP 的评估 mean_return=0.31−0.62=−0.31；Topo+routeaware 为 0.31−0.45=−0.14。后者超时 24% 对该评估得分贡献为 0，不能因此说训练没有惩罚超时，或仅据这个分数认定总体任务完成更好。

r1 双标在 raw 计分中是 +1−1=0，而训练 wrapper 却选择成功 +10；其原始评估 mean_return 为 low −0.43、medium −0.79。此时 S/C 是重叠标志率，不能当作互斥终局类别。

依据：[evaluation.py](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/../algos/sb3_torch/evaluation.py:195>)、[训练入口的评估口径说明](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/train_intersection_yield_v2.py:1638>)。

## 5. 奖励相同不等于全部优化目标相同

MST+SLT 的 SLT 是辅助表征优化目标，full 中的其他辅助损失也应单列。它们不是环境每步 reward，也不应与成功 +10 等权重混称。ST/RT/Topo/3slot 的结构改变，同样不能自动称为新增奖励项。

当前 r2 中档强基线与纯基线使用同一 v2 环境奖励数值设计；现有证据不支持用“给 MST 和 SAC 设置了不同成功/碰撞奖励”解释其成绩差异。网络、辅助优化、策略行为和训练动态仍是不同因素。本次不修改奖励或启动归因实验。

## 6. 可追溯清单

- [FD 本地 20 条奖励证据](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_reward_inventory_local_20261001.json>)。
- [runs 18 条奖励证据](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_reward_inventory_runs_20261001.json>)。
- [原始实验汇总](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/intersection_experiment_summary_20261001.md>)。
- [实现地图](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/full_mst_slt_implementation.md>)。

本审计补充奖励参数、实际终局行为及证据完整性的区别，保留所有历史成绩与限制。
