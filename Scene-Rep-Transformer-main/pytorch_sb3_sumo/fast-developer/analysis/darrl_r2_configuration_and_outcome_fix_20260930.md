# 2026-09-30：DARRL 同名场景 r2 与终局互斥修复

## 当前配置与历史配置

用户明确要求原位修改三个现有场景，不新增场景名称。当前配置增加
`configuration_revision=darrl_r2_20260930`，并记录前序 `darrl_r1_20260930`。
仅凭场景名称不足以区分新旧实验，分析时必须读取实验保存的 `traffic_config`。

| 场景 | r1：每路每 0.1 秒 p | r2：每路每 0.1 秒 p | r2 每路请求量（辆/小时） | r2 三路合计 |
|---|---:|---:|---:|---:|
| `intersection_random_darrl_low_v1` | 0.03 | **0.015** | 540 | 1620 |
| `intersection_random_darrl_medium_v1` | 0.05 | **0.03** | 1080 | 3240 |
| `intersection_random_darrl_high_v1` | 0.07 | **0.05** | 1800 | 5400 |

继续采用现有逐 tick、逐路线独立 Bernoulli 请求日程；不是 SUMO 原生每秒
`flow probability`。保留背景 departLane=arrivalLane=1、自车 minGap=1 m、
背景与自车型 jmIgnoreFoeProb=0、SUMO collision.action=remove、原路网/路线、
ego lane2/50秒释放、600 raw步上限及训练/评估 seed 域。仍使用 SUMO ego 碰撞事件
或 ego 几何重叠检测，不加入 DARRL legacy 距离/消失判据。

R1 归档为 [archive_manifest.json](scenario_revisions/darrl_r1_20260930/archive_manifest.json)。
32 个文件包含三个原场景全套资产、placeholder、关键环境/训练/奖励源码和原运行清单；
源与副本 SHA256 全部一致。原运行 `runs/smlp_darrl_0930` 已完整结束，保留其模型、
原始日志和原始结果；它只训练了 r1 低/中档，高档 r1 配置仅做过环境验证。
本次显式只重建三个 DARRL 场景资产，其余六个随机场景的配置哈希与资产保留。

## 问题证据与修复

原 final eval 两路各 100 回合，成功与碰撞使用相同分母：

| r1 评估 | 原始成功数 | 原始碰撞数 | 同回合双标数 | 碰撞优先的离线互斥计数 |
|---|---:|---:|---:|---|
| low | 30 | 73 | 3 | 成功 27 / 碰撞 73 |
| medium | 13 | 92 | 5 | 成功 8 / 碰撞 92 |

这不是绘图或分母问题。环境把 SUMO 到达与碰撞事件分别转成终局布尔值，
没有保证互斥。双标时基础回报 `success - collision` 为 0；训练奖励封装采用
`if success ... elif collision ...`，因此错误进入成功奖励分支。

训练日志也证实该问题发生过：low 有 29 个双标回合，medium 有 10 个；这些回合
均记录 `reward_success=+10`、`reward_collision=0`。其整回合 shaped return
分别约 10.63–11.59、10.34–11.85。**因此离线修正评估计数不能修复已学到的模型。**
上表仅为历史事件重新分类，不代表修复后重新训练的表现。

修复统一放在 `SumoSceneEnv.step()`：先完成全部场景特定事件检测（包括子类的
CARLA 终点重算），再按 **碰撞 > off-route > 成功 > 超时** 解析唯一终局。
非终止步四类均为 false。raw reward、shaped reward、terminated/truncated、
replay 终止标志、训练与评估指标均使用同一解析结果。到达或碰撞恰好发生在时限步时，
不再同时标成 timeout；碰撞的真实终止不会因 timeout 标志被当作可 bootstrap 的截断。

正常训练、评估的 info 和逐 raw tick snapshot 额外保存：

- `raw_sumo_arrived`、`raw_sumo_collision`：现有 TraCI 查询得到的原始 ego 事件。
- `raw_max_time`：原始时间上限条件。
- `terminal_outcome_protocol=exclusive_terminal_v2`：最终终局解释协议。

原始事件允许重叠，最终终局分类保持互斥；没有增加每步 TraCI 查询或额外诊断 rollout。
SUMO API 将到达与碰撞分别暴露为列表，接口说明参见
[Simulation Value Retrieval](https://eclipse.dev/sumo/docs/TraCI/Simulation_Value_Retrieval.html)。
本项目观察到的双标及奖励错误由原始实验记录和回归测试支持，不将接口说明当作发生频率证据。

## 验证与新训练协议

- 场景回归 16 项通过；九个随机场景资产校验通过，三个 DARRL 场景均为 r2。
- 终局相关测试 6 项通过，包含 16 种布尔组合及实际 `step → reward wrapper` 路径。
  同步成功/碰撞/时间上限的测试结果为 collision=true、其它终局=false、
  terminated=true、truncated=false、reward_success=0、reward_collision=-10，
  含单步成本的 shaped reward=-10.01。
- 三档真实 D1 SAC+MLP 环境工厂预热与短步验证通过：
  `runs/darrl_r2_preflight_0930/validation_report.json`，status=passed、cases=3。

用户要求本次低/中档使用 r2 和修复后的终局协议，从零各训练 100000 raw steps，
2 个 CUDA worker、训练 seed=0、warmup=5000 raw、每10000 raw保存 checkpoint，
训练后各自动做100回合 validation 评估，保留正常过程诊断。新运行使用独立输出根，
不续训或覆盖 r1 模型。r1 与 r2 同时改变了请求概率和终局/奖励处理，不能把新旧差异
单独归因于交通密度。

### 真实碰撞复现与正式运行

独立单例验证 `verify_terminal_outcome_sumo.py` 在r2中档实际D1/base工厂内，通过明确标记的TraCI测试干预制造碰撞，不计入训练或策略评估。原生SUMO同tick确实返回ego碰撞与arrived；正常`env.step`返回collision=true、success=false、terminated=true、truncated=false，shaped reward=-10.01。报告status=`passed_same_tick_collision_over_arrival`，exit0；证据保存在工作区 `runs/darrl_r2_terminal_check_0930/terminal_outcome_report.json`。另两项既有真实CARLA成功和timeout回归通过，确认场景子类重算仍经过统一终局处理。

上轮完整审计见[darrl_r1_outcome_audit_20260930.json](darrl_r1_outcome_audit_20260930.json)，旧run的`SCENARIO_REVISION_NOTE.json`保留配置和局限说明，原结果不覆写。

全部上述验证通过后，已于2026-09-30约22:24（Asia/Shanghai）发起新正式运行：`runs/smlp_dr2_0930`，launcher pair仍为`darrl-low-medium`，场景名不变、配置为r2（low=.015，medium=.03）。2 CUDA worker分别fresh SAC+MLP、seed0、100000 raw预算，5000 raw后开始优化，每10000 raw保存，正常过程诊断开启，各训练完成后自动100回合validation评估。

这只是启动记录，不是训练完成结果。未来读取同名模型/场景时，应同时附带运行根、配置revision和terminal_outcome_protocol，避免把r1含错误奖励的旧模型与r2新模型混淆。

正式运行启动核验：supervisor72212，low14512、medium84484，两个worker均使用CUDA并推进；manifest均为r2。正常训练摘要的低27/中24个已结束episode中，有效success&&collision交集均为0，diagnostic_error_count均为0。中档已实际记录raw SUMO到达与碰撞同时为真，但最终判为collision-only，raw reward=-1。以上为启动快照，不是100k最终成绩；任务结束后进程继续训练及自动评估。
