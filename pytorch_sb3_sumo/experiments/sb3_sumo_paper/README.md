# PyTorch + SB3 + SUMO 论文复现实验

本目录只描述和编排新增的 SUMO 复现实验，不修改 TensorFlow/TF2RL、SMARTS 或 CARLA 原文件，也不要求安装 SMARTS/CARLA。

五个原 SMARTS 任务并非凭示意图重画：SUMO 直接加载作者 `v1.0.0`
Release 中的原始 `map.net.xml` 和全部交通 `*.rou.xml`；仅按原
`scenario.py` 的 mission 注入 ego。CARLA 任务没有发布 SUMO 资产，因而根据
`carla_env.py` 与 `wp.npy/wp2.npy` 重建 L 形路线、强制目标车道、交互车辆和行人。

## 协议口径

`protocol.json` 同时保存两套明确区分的规模：

- `paper`：论文 Section IV-E 的 100,000 个原始仿真步、5 个随机种子，以及测试阶段每个场景 50 个 episode；
- `source_release`：`configs/init_configs.py` 暴露的 1,000,000 步、1 次实验默认值。

算法、网络和优化器以发布源码为主；实验次数和最终表格以论文实验段落为主。所有冲突均记录在 `protocol.json`，不会悄悄选择其中一项。
便于人工核查的逐项说明见 `SOURCE_PAPER_DIFFERENCES.md`。

源码中几个容易被“标准实现”改掉的行为也被保留：四步 N-step 回报之后只乘一次
`gamma` 做 bootstrap；Proposed/MST 一个动作保持 3 个原始仿真步但每个原始步更新；
SMARTS 的步数上限可 bootstrap，而 CARLA 的 `max_time` 不可 bootstrap；随机旋转在
训练动作采集和更新时启用、正式测试时关闭；Proposed 的 episode 最后一条 transition
按发布 trainer 的队列逻辑重复写入；社会车辆保持 SMARTS 默认的 endless traffic。
它们都在协议文件中形成可审计契约。

场景映射为：

| SUMO 名称 | 论文名称 |
|---|---|
| `left_turn` | Unprotected Left Turn |
| `cross` | Double Merging |
| `roundabout_easy` | Roundabout-A |
| `roundabout_medium` | Roundabout-B |
| `roundabout` | Roundabout-C |
| `carla` | CARLA Town-10 任务的 SUMO 等价场景 |

## 可忠实运行的方法

- `proposed`：MST + SLT + SAC，对应论文 Proposed；
- `mst`：复用相同源码时钟 SAC、在构造前关闭 SLT，对应论文消融 MST；五个 SMARTS
  场景用于表格/曲线，CARLA 训练用于复现 Fig. 9f 的 `Trans` 曲线；
- `ppo`：五个 SMARTS 场景使用源码 80×80 RGB CNN，CARLA 使用源码向量
  GRU/relation-attention，并保留各自的原始步动作时钟与 PPO 数据流异常。
- `sac`：论文基线的透明重建。发布源码的所谓 LSTM 分支实际是
  `GRU(256)+Dense(256)`；v14 精确恢复 `STATE_LSTM` 的 ego
  `[x,y,speed,0,heading]`、邻车 `[x,y,speed,distance,heading]`、缺帧插值和显式时间
  mask，不读取 Proposed 轨迹/地图，也不调用 MST/层级 Transformer。由于发布配置未
  暴露可运行的 plain SAC，该方法始终带 `paper_sac_lstm_reconstruction_v2` 标识。

论文 profile 对 PPO 的请求预算仍是 100,000，但发布 runner 会完成跨过该阈值的整个
512-step horizon，实际为 100,352；而 `tools/test.py` 读取的是最后一次周期保存的
100,000 checkpoint（保存发生在最终 horizon 更新前）。迁移版将该 checkpoint 作为
`final_model`，并把 100,352 learner 另存为 `post_horizon_model`。

发布 PPO 测试 API 还有一个无法字面执行的错误：SMARTS 的 `test` 位置参数绑定成
`mask`，CARLA 分支则把三个输出解包成两个。迁移版采用最小且一致的意图修复——测试
使用确定性均值动作并关闭 dropout；该限制已写入协议，不伪称逐语句等价。

DrQ、DT、RDM 在发布源码中没有无歧义且可直接迁移的统一训练入口。plain SAC 只按
上一段可恢复组件重建，编排器通过 implementation id 强制审计，绝不会把仍使用 MST 的
分支当成 SAC 基线；具体限制记录在 v14 协议文件中。

需要明确的是，“SUMO 等价”不等于跨仿真器逐帧相同。五个 SMARTS 任务直接复用作者
发布的 SUMO 网络与交通文件，但车辆控制器由直接 TraCI 实现替代；CARLA 没有发布可
直接转换的地图/物理资产，因此只能复用源码路线、`wp.npy/wp2.npy`、成功判定、观测
张量和交互要素，在 SUMO 中重建任务。最终结果应视为同一方法在 SUMO 等价任务上的
复现实验，而不是 CARLA 物理结果的数值复刻。

## 命令

从项目根目录、使用 `llm_pipeline` 环境运行：

```powershell
python tools/reproduce_paper_sb3_sumo.py plan --profile paper --methods all
python tools/reproduce_paper_sb3_sumo.py preflight --profile paper --methods all
python tools/reproduce_paper_sb3_sumo.py run --profile paper --methods all --device cuda
python tools/reproduce_paper_sb3_sumo.py status --profile paper --methods all
python tools/reproduce_paper_sb3_sumo.py summarize --profile paper --methods all
```

v12 默认使用 `paper_source_audited_runs_v12_frozen`，共 18 个方法—场景组合、每组 5
个种子。运行前会逐字节冻结 `protocol.snapshot.json`；已有不同协议的目录会被拒绝。
冻结协议 SHA-256 为
`9AC3560E286A5E9EB4943EBED643E6223ACE1998DFE5AB008EEA37CA5398982A`。
训练结束的同一批 50 回合同时生成 `final_evaluation.json` 和含逐回合事件/成功完成
时间的 `paper_evaluation_detailed.json`。每次训练还必须生成
`checkpoint_audit.json`，逐个记录 ZIP CRC、SHA-256 和内部 raw-step 时钟。
v9/v10 结果早于 100k partial held-action replay 修正；v11 又在周期 checkpoint 落到
三步 held action 内时最多多包含两次更新。三者均列在
`diagnostic_predecessor_results`，不进入 v12 正式聚合；v12 的 90 条训练任务全部写入
新根。

SAC/PPO 优先基线使用独立的 `protocol_baselines_v14.json` 和结果根
`paper_baselines_sac_ppo_v14_frozen`，不会混入 v12。它使用论文 Table VI 步数上限、
曲率/加减速受限的 Ackermann 代理，以及固定且互斥的 released-traffic 80/20
训练/测试划分；由于论文未发布真实 held-out traffic 和 Bullet 物理状态，该协议是
“更可比的显式重建”，不是跨仿真器逐帧等价。v13 的未完成 SAC 输入仍采用 Proposed
字段语义，已停止并以 `NOT_FORMAL_SOURCE_FIDELITY.md` 排除。冻结 v14 SHA-256 为
`189D1D504D1BD15C01557C2145C0F91941DDBFF97C35252F85D6FF6B864148A3`。命令为：

```powershell
python tools/reproduce_paper_sb3_sumo.py run `
  --protocol-path experiments/sb3_sumo_paper/protocol_baselines_v14.json `
  --methods sac,ppo --scenarios all --seeds all --device cuda `
  --output-dir results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen
```

完整 Proposed 与 MST 的同环境对照使用
`protocol_proposed_mst_v15.json`。它逐字复用 v14 的 Ackermann 代理、互斥 80/20
traffic 划分和论文 episode 上限，禁止导入 direct-control 的 v12 结果。矩阵包含 30 个
Proposed 任务（六个场景×五种子）和 25 个 MST 任务（论文报告 MST 的五个 SMARTS
场景×五种子）；论文未报告 CARLA MST，因此不把额外消融混入正式复现。v15 协议
SHA-256 为
`68CAF4F814189C8FC1E223C4E4245019478D5DEDB74D3A6612A1C48F4F57308F`。命令为：

```powershell
python tools/reproduce_paper_sb3_sumo.py run `
  --protocol-path experiments/sb3_sumo_paper/protocol_proposed_mst_v15.json `
  --methods proposed,mst --scenarios all --seeds all --device cuda `
  --output-dir results_sb3_sumo_paper/paper_proposed_mst_v15_frozen
```

论文称测试“训练成功率最高”的 policy，但发布仓库没有 one-shot trainer 或可用的
best-policy 保存路径。新增运行在不改变 latest 主结果的前提下，前瞻保存固定 20 回合
窗口成功率严格提升时的 `best_training_success_model.zip`，并记录窗口、episode、模型
时钟和 tie 规则；最终报告会分别评估 latest 与该显式重建选择，避免混淆口径。

`run` 使用确定性的目录名；已存在 `final_evaluation.json` 的任务会跳过，失败或中断的目录会被保留并报告，不会自动删除实验数据。含
`NOT_FORMAL_SOURCE_FIDELITY.md` 的旧审计目录状态为 `excluded`，编排与汇总都会强制忽略。

`summarize` 还会生成逐 episode 曲线、每 200 raw-step 的逐 seed 曲线和跨 seed
聚合曲线。EMA 的处理顺序与公式写入 `summary.json`；论文 Table II/III 的时间参考值
来自独立的 `paper_reference_tables.json`，汇总表同时显示 SUMO 值、论文值和差值。
