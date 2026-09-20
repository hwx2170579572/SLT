# 六方法 × 六场景系统实验矩阵

该扩展矩阵固定比较 `SAC`、`PPO`、`MST`、`MST+SLT`、
`TemporalGraph` 与 `Full+BalancedSlots`。六个场景均运行训练种子
`0/1/2`，每个最终策略在互斥的 evaluation traffic 上执行 50 个确定性回合，
因此每个预算档共有 108 次训练和 5,400 个最终测试回合。

唯一规范源是 [protocol.json](protocol.json)。`ccfa.yaml` 的 S10–S14 将合同冻结、
36 个算法—场景接口门禁、108 次训练、聚合和深层归因拆成机器可检查阶段。所有未知
结果保持 `TBD`；不得跨预算、跨 seed 或跨协议填补。

## 方法语义

| 合同方法 | CLI | 角色 |
|---|---|---|
| `sac` | `sac` | 审计后的 recurrent SAC 重建基线 |
| `ppo` | `ppo` | 发布源码数据流的 on-policy 基线 |
| `mst` | `mst` | MST+SAC，不使用 SLT |
| `mst_slt` | `scene_rep` | MST+SLT+SAC，最近发布基线 |
| `temporal_graph` | `temporal_graph` | 去拓扑查询的机制消融 |
| `full_balanced` | `topo_scene_balanced` | TopoQuery + TemporalGraph + Graph-SLT + BalancedSlots |

论文没有报告 CARLA 上的 SAC 和 MST；这两格可运行，但合同明确标为统一环境下的
controlled extension，不作为论文表格的复现声明。`TemporalGraph → Full+BalancedSlots`
同时改变拓扑查询和槽位归一化，不能凭这一条对比单独归因于拓扑。

环境锚点是已完成的
`results_topo_scene/iteration/runs/iteration__topo_scene_balanced__left_turn__seed0/arguments.json`
（SHA-256 `B7E2C939...F75014F`）：`direct + frozen_80_20 + source`，并锁定相同
reward/action space、discount、history、neighbors 与 path contract。PPO 只保留其已审计
的 raw-tick rollout 时钟例外；强行设为三步环境 repeat 会产生另一个 PPO 方法。

## 预算档

| Profile | Raw steps/训练 | Checkpoint | 用途 |
|---|---:|---:|---|
| `development_50k` | 50,000 | 10,000 | 完整矩阵开发筛选 |
| `paper` | 100,000 | 10,000 | 用户确认的论文步数预算、三 seed、10k 周期保存 |
| `source_release_1m` | 1,000,000 | 100,000 | 源码默认预算的高成本确认 |

三个 profile 的方法、场景、seed 和最终 50 回合测试完全相同，但结果根必须分开。
1M 档把周期 checkpoint 调整到 100k 以控制存储；这不改变训练更新或最终 checkpoint。

已完成的 50k 图方法实测每次约 2.1–3.2 小时。按更新数近似线性外推，1M 的 108 次
训练即使并行也属于多周任务，且 20k checkpoint 会接近当前 D 盘剩余容量。因此真正
启动前必须在 `artifacts/contracts/systematic_matrix_budget_decision.json` 冻结所选
profile 和 worker 数，不能默默把 1M 改成 50k。

## 验证与运行

从项目根目录、使用 `llm_pipeline`：

```powershell
conda run -n llm_pipeline python tools/validate_systematic_matrix.py

conda run -n llm_pipeline python tools/reproduce_paper_sb3_sumo.py plan `
  --protocol-path experiments/systematic_matrix/protocol.json `
  --profile paper --methods all --scenarios all --seeds all `
  --output-dir results_systematic_matrix/selected

conda run -n llm_pipeline python tools/reproduce_paper_sb3_sumo.py preflight `
  --protocol-path experiments/systematic_matrix/protocol.json `
  --profile paper --methods all --scenarios all --seeds all --device cuda `
  --workers 4 --output-dir results_systematic_matrix/selected

conda run -n llm_pipeline python tools/reproduce_paper_sb3_sumo.py run `
  --protocol-path experiments/systematic_matrix/protocol.json `
  --profile paper --methods all --scenarios all --seeds all --device cuda `
  --workers 4 --schedule diversified --continue-on-error `
  --output-dir results_systematic_matrix/selected

conda run -n llm_pipeline python tools/reproduce_paper_sb3_sumo.py status `
  --protocol-path experiments/systematic_matrix/protocol.json `
  --profile paper --methods all --scenarios all --seeds all `
  --output-dir results_systematic_matrix/selected

conda run -n llm_pipeline python tools/reproduce_paper_sb3_sumo.py summarize `
  --protocol-path experiments/systematic_matrix/protocol.json `
  --profile paper --methods all --scenarios all --seeds all `
  --output-dir results_systematic_matrix/selected

conda run -n llm_pipeline python tools/analyze_systematic_matrix.py `
  --protocol experiments/systematic_matrix/protocol.json `
  --profile paper --result-root results_systematic_matrix/selected `
  --require-complete
```

编排器按 job 级恢复：已通过完整 artifact contract 的目录会跳过，失败或不匹配目录
原样保留供审计。若训练已经完成、只在最终测试阶段中断，可使用
`tools/recover_interrupted_paper_evaluation.py` 恢复测试；周期模型未保存 replay、SUMO
状态与完整 RNG 状态，因此不能把中途 checkpoint 伪称为 bitwise 可恢复训练。

## 预声明统计与归因

每个场景先报告三个 seed 的均值与总体标准差，再报告配对层级 bootstrap 95% CI。
success/collision 是主指标；return、off-route、timeout、成功完成时间和效率指标是
次指标。预声明链条为：

1. `SAC → MST`：层级场景编码；
2. `MST → MST+SLT`：一步 SLT；
3. `MST+SLT → TemporalGraph`：逐帧交互与结构化 Graph-SLT 的联合替换；
4. `TemporalGraph → Full+BalancedSlots`：拓扑查询与稳定槽的联合增量；
5. `MST+SLT → Full+BalancedSlots`：最终端到端效果；
6. `SAC ↔ PPO`：跨算法家族的描述性基线，不作单组件因果解释。

任何“深层归因”必须同时满足：对比被预声明、协议匹配、三 seed 完整、主安全指标
没有相反变化，并与训练诊断/效率证据一致。否则只报告相关性或待验证机制。
