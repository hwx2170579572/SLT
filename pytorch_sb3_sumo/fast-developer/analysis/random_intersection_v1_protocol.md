# 新增随机路口场景：intersection_bernoulli_v1

本协议新增三档场景，不替换 `intersection` 或 `intersection_sorted`。它们用于同一路口的密度对比：建议在中档进行主训练与消融，将中档训练得到的同一 checkpoint 用于低、高档的跨密度评估。

## 明确配置

流量单位为每小时计划进入的背景车辆数，不是道路上实际同时存在的车辆数，也不是保证实现的通行量。

| 新场景名 | `-E3 → -E0` | `E0 → E3` | `E2 → E1` | 总需求（辆/小时） |
|---|---:|---:|---:|---:|
| `intersection_random_low_v1` | 200 | 150 | 240 | 590 |
| `intersection_random_medium_v1` | 400 | 300 | 480 | 1180 |
| `intersection_random_high_v1` | 600 | 450 | 720 | 1770 |

| 档位 | 三路各自的平均计划发车间隔（秒，顺序同上） | 60 秒控制窗口的期望新发车数（全部入口） |
|---|---|---:|
| 低 | 18 / 24 / 15 | 9.83 |
| 中 | 9 / 12 / 7.5 | 19.67 |
| 高 | 6 / 8 / 5 | 29.50 |

这些是分布的期望值，每个回合的实际车辆数和间隔会变化。车辆插入可能因交通状态而延迟，因此必须区分计划需求、实际发车和车辆占用。

按用户提出的“每条进口车道以概率 p 发车”，采用每个 0.1 秒仿真步进行一次 Bernoulli 抽样。这里的车道是上述三条背景路线各自的入口 lane 0，不额外启用 ego 入口或其他转向车道。

| 档位 | `-E3` 每步概率 | `E0` 每步概率 | `E2` 每步概率 |
|---|---:|---:|---:|
| 低 | 0.00555556 | 0.00416667 | 0.00666667 |
| 中 | 0.01111111 | 0.00833333 | 0.01333333 |
| 高 | 0.01666667 | 0.01250000 | 0.02000000 |

换算公式为 `p_step = q_veh_per_hour × 0.1 / 3600`。SUMO XML 的 `probability` 参数是按秒定义，由引擎乘以 step-length，因此 XML 必须填写 `q/3600`，而不是上表的 `q/36000`。同一仿真步可在不同入口同时产生请求，每个入口每步最多一个请求；不是固定间隔发车，也不是仅在 RL 每次决策时抽样。

共同设置：

- 保留 `intersection_sorted` 的路网与 ego 文件，按文件 SHA256 校验一致。
- ego 路线 `-E1 → -E0`，发车请求为第 50 秒，lane=2、pos=0、speed=0。
- 每回合从仿真时刻 0 开始，ego 插入后最多 600 个 raw step；每步 0.1 秒，即最多约 60 秒受控时间。原有 action repeat 默认仍为 3。
- 三路背景车流均使用 SUMO 原生 `<flow probability="q/3600">`。配置发车时间范围为 `[0, 130)` 秒，覆盖预热、允许的 ego 插入延迟和完整控制窗口。
- 背景车均保留 lane=0、pos=0、speed=max；三条路线不变，不增加新转向。
- 新场景中背景车走完路线后离开，不启用旧 Paper 环境的 endless 重插入。否则发车需求会额外叠加循环车辆，无法直接解释三档流量。旧场景的 endless 行为保持原样。
- 收集原 30 份模板的全部 120 个车型参数实例，赋予唯一 ID；按各入口中实际出现的车辆计数建立条件车型分布。保留已有车长、跟驰、速度分布、换道等参数。三档共享同一套分布。
- 随机性由每回合 SUMO seed 驱动；相同场景、相同 seed 和相同 SUMO 版本可以复现实验输入。实际插入时刻和后续轨迹仍可受 ego 策略影响。
- 新场景的 `depart_scale=1.0`；三档流量已写入 flow，禁止再施加旧场景的 4 倍缩放。

## 设置依据与证据边界

原 30 份模板中，每份三个方向分别有 200、150、240 辆背景车，因此保留 `200:150:240` 的入流比例。以当前旧协议通常的 50–110 秒窗口计算，30 模板的平均计划需求约 868 辆/小时；用该数量级作为选择三档的需求锚点。旧协议还有 endless 重插入，这个数字不是其全部实际入流或实测通行量，也不能据此声称新中档的实际在场密度比旧场景高多少。

低、中、高是预先定义的需求档位。没有根据 Full 胜率选择档位，也尚未通过 100k 训练将其标定为低、中、高学习难度。三档内部只有入流率不同；新旧协议之间还存在发车过程、模板组合和种子划分差异，不能把跨协议的成绩差归因于模型模块。

## 训练、开发评估、最终测试隔离

随机流定义可以共用同一个 XML；隔离的是由 seed 实例化的交通过程，不能用 XML 文件数衡量新协议的场景样本数。

| split | 实际 SUMO seed 范围 | 映射公式 |
|---|---|---|
| `train` | 0–999999999 | `logical_seed % 1000000000` |
| `validation` | 1000000000–1499999999 | `1000000000 + logical_seed % 500000000` |
| `test` | 1500000000–1999999999 | `1500000000 + logical_seed % 500000000` |

默认开发评估采用逻辑 seed `10000–10099`，对应 validation seed `1000010000–1000010099`。同一组逻辑 seed 在 test 分区对应 `1500010000–1500010099`，与训练及开发评估没有交集。

模型训练仍可只用 seed=0。模型训练种子与每个回合的车流种子应分别记录；增加车流样本不是增加模型训练重复次数。最终测试分区应在方法确定后使用，不参与模块选择。

## 代码与资产入口

- `envs/sumo/random_intersection.py`：三档唯一配置来源、seed 映射、资产构建与 SHA256 清单。
- `envs/sumo/original_scenarios_v1/intersection_random_{low,medium,high}_v1/`：每档独立的 `map.net.xml`、`ego.rou.xml`、`traffic/traffic_00000.rou.xml` 和 `scenario_manifest.json`。
- `envs/sumo/paper_scenario_registry.py`、`scenario_registry.py`：独立场景注册。新场景使用 Paper 环境，不允许由旧 base 环境直接运行。
- `envs/sumo/paper_env.py`：每回合 seed 分区及正常训练/评估的信息输出。
- `fast-developer/train_intersection_yield_v2_d1.py`：用 `--scenario` 选择新场景，用 `--eval-traffic-split validation|test` 指定评估用途；默认旧场景保持不变。

在原训练命令中选择中档时，增加 `--scenario intersection_random_medium_v1 --eval-traffic-split validation`，删除旧的 `--depart-scale 4.0` 参数。低、高档仅替换场景名。构建资产本身不启动训练。

## 验证记录

2026-09-30，使用项目 Python 3.10.16 / SUMO 1.25.0：

- 新协议 4 项单测通过：原路网/ego 保持、三档仅入流率不同、seed 分区隔离、独立注册与资产哈希。
- `validate_random_intersection_env.py` 通过：三档各 reset 并执行 5 次动作，验证 info 内分区与流量记录；中档同 seed 初始车流复现、不同 seed 改变；train/validation/test 实际种子分离；旧场景 seed 不被重映射。
- `test_paper_reproduction.py` 在项目内独立 pytest 临时目录中完整 48 项通过。默认共享临时目录存在 Windows 权限问题；跨测试文件合并执行还出现过全进程模块导入检查失败，该项及整个 reproduction 文件分别运行均通过。没有修改测试断言。
- SUMO 无训练流量验收使用逻辑 seed=10000、validation 实际 seed=1000010000：发车定义止于 130 秒，运行至 180 秒让已产生车辆驶出；结果如下。单 seed 计数用于执行检查，不用于估计长期平均流量或难度。

| 档位 | 计划请求数 | 实际插入数 | 验收结束时待插入/仍在网内 | 有插入延迟的车辆数 | 最大插入延迟 |
|---|---:|---:|---|---:|---:|
| 低 | 19 | 19 | 0 / 0 | 0 | 0 s |
| 中 | 55 | 55 | 0 / 0 | 7 | 1.1 s |
| 高 | 69 | 69 | 0 / 0 | 12 | 2.4 s |

中档同 seed 重跑的车辆 ID、计划发车时间、类型与路线逐项一致；换逻辑 seed=10001 后得到另一份计划（42 个请求）。这不是不同 ego 控制策略间的反事实验证，不据此宣称已经实测策略无关性。

同一 seed 的跨密度验收中，各入口的低档计划发车时刻是中档的子集，中档是高档的子集；但匹配时刻的车型并非始终相同。因此三档共享驾驶参数分布、只改变需求参数，不等于跨密度逐车状态完全配对。方法比较应首先在相同密度与相同 seed 内进行。

验证原始记录：[PaperEnv 集成结果](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/random_intersection_v1_validation/paper_env_integration.json>)、[SUMO 车流验收结果](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/runs/random_intersection_v1_validation/bernoulli_smoke_summary.json>)。本次没有新启动 100k 训练，也没有完成学习难度或跨密度泛化评估。

### PaperEnv 与 D1 入口补充验证

workspace 根 `runs/random_intersection_v1_validation/paper_env_integration.json` 记录 PaperEnv 每档一次 reset、约 50 秒 ego 发车预热后执行 5 个决策步的 terminal info；三个 case 均使用 validation logical seed 10000。低/中/高实际驶入背景车数为 9/28/35，峰值同时在网背景车数为 5/15/17，待插入数均为 0；平均插入延迟为 0/0.0964/1.3657 秒，最大延迟为 0/1.4/9.6 秒。这只能说明该单 seed 短控制片段中流量档形成不同背景占用与插入压力，不是学习难度标定、策略表现或长期流量估计。PaperEnv 文件位于 workspace 根 `main1/runs`；上面的 headless SUMO 明细位于项目根 `main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/runs`，两者是不同验证记录。

D1 base adapter `sac_mlp_d1_st_rt` 的 medium factory 检查使用逻辑 seed 217，reset 后执行一次 sampled-action step，未终止。训练环境实际 split/SUMO seed 为 `train/217`；测试环境为 `test/1500000217`。两者均报告 `intersection_bernoulli_v1`、每入口流率 `{neg_E3_to_neg_E0:400, E0_to_E3:300, E2_to_E1:480}` veh/h、vehicle/pedestrian scale=1、endless reinsertion=false、reinsertions=0。对应 train/test overlay manifests 位于 `fast-developer/tmp/factory_reset_check_45928/`，两者均记录 3 个基础 vehicle flows、无额外 flow/vehicle/person、原始资产未修改且 map/ego route 未变。factory 检查输出可核验，但命令显式退出码未捕获；没有执行 learning update。

随机场景 eval-only 的输出保护已用临时结果验证：若目标目录已有不同 split 或未标记的随机场景 `evaluation_results.json`，会拒绝覆盖并要求独立 `--output-dir`；同场景、同 split 可通过。实现见 `fast-developer/train_intersection_yield_v2.py` 的 `_guard_random_evaluation_output()`（第 344–369 行），D1 eval-only 在评估前调用该检查。用户可从原训练 checkpoint 读模型、将 test 评估写入新目录：`python fast-developer/train_intersection_yield_v2_d1.py --eval-only --method sac_mlp_d1_st_rt --model-path <training-run>\final_model.zip --output-dir <new-test-eval-dir> --scenario intersection_random_medium_v1 --eval-traffic-split test`。该入口检查不构成训练或策略评估结果。

官方机制说明：

- [SUMO 车辆、路线与随机 flow 定义](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html)
- [SUMO 随机性与随机数流](https://sumo.dlr.de/docs/Simulation/Randomness.html)
- [SUMO 车辆插入与延迟](https://sumo.dlr.de/docs/Simulation/VehicleInsertion.html)

## 增补：每步 p=0.5 的中档变体与预热交通快照（2026-09-30）

保留 `intersection_random_medium_v1` 原设置（`-E3→-E0` / `E0→E3` / `E2→E1` 分别 400/300/480 veh/h，总请求 1,180 veh/h），另增 `intersection_random_medium_p05_v1`。p05 只在三条背景进口路线各自的 lane 0 上，以每个 0.1 s 仿真步概率 `p=0.5` 请求一辆车；每路期望请求率为 18,000 veh/h，合计 54,000 veh/h。这里描述的是计划请求，不是保证成功插入的实际流量，也不是经过学习表现标定的“中等难度”。

为满足每条路线每步至多一个请求，且避免 SUMO 原生 flow probability 超过 1，p05 在 SUMO 启动前按 actual SUMO seed 与路线 ID 派生独立 SHA256 种子，再用 Python `Random.random()` 对 `[0,130)` 的 0.1 s tick 逐步采样并生成有序的逐车 departure schedule。该计划与 policy 动作分离；若 SUMO 当时无法插车，请求仍留在插入队列中，实际插入时间和车辆占用仍可能受到仿真状态影响。p05 静态 traffic XML 只提供车型、分布和路线定义。它与冻结 medium 的 map、ego 文件、120 个车型参数实例、3 条 route 及 3 个 driver distribution 内容逐项相同；只有 medium 静态 XML 中的 native `<flow>` 改为每回合预生成的 `<vehicle>` 请求。旧 medium 配置没有改变。

`ensure_random_intersection_assets()` 生成并校验版本化静态资产；每回合调用 `write_seeded_episode_traffic(scenario, actual_seed, traffic_path, output_path)` 生成 episode 文件和 schedule 元数据。生成器现可解析 ego XML 中车辆引用顶层 `<route>` 的形式，并继续严格校验 route edges `-E1 -E0`；这修复了解析兼容性，没有改动冻结源 map/ego 资产。p05 的专门单测及既有随机场景协议测试共 8 项通过：覆盖同 seed 文件字节复现、异 seed 计划变化、每路线每 tick 不重复、0.1 s 网格与 `[0,130)` 范围、各路线 1,300 次试验的频率 sanity bound、旧 medium 配置保持不变，以及 p05/medium map、ego 和车型/route/distribution 定义一致。该结果是代码/协议测试，不是 SUMO 训练或策略证据。

### 随机交通场景的预热采样

用户要求对四个随机场景 `intersection_random_low_v1`、`intersection_random_medium_v1`、`intersection_random_high_v1` 和 `intersection_random_medium_p05_v1` 均记录 30/40/50 s 快照。PaperEnv 在正常 reset 预热中完成第 300/400/500 个 0.1 s `simulationStep` 后，只读取 TraCI 状态；这三个采样点不额外推进仿真。reset 的 `info.warmup_traffic_checkpoints` 会随已有训练/评估 diagnostics 的 episode `reset_info` 写入 `episodes.jsonl`。

每个 checkpoint 记录目标秒数与实际 simulation time、全网/背景车数与按 route 计数、ego 是否已插入；停驶指标定义为**当前已在网背景车**速度 `<0.1 m/s`，并同时记录该阈值。`inlet_lanes` 按三个入口 lane 记录 lane occupancy、SUMO halting 数、到时请求数、累计实际插入数和待插入数。额外字段包含按路线拆分的 `requested_background_due`、`actually_inserted_background_cumulative`、`pending_background_insertions`，请求计数来源、请求平衡项、已驶离车辆数和插入延迟均值/最大值。**pending 是尚未进入路网的请求，不能算作在网车辆或排队车辆；在网停驶统计也不包含 pending。**p05 请求到时按预生成计划截至 snapshot time 统计，native-flow 场景按“累计插入+待插入”估计，并通过 `request_count_source` 明示来源；不同来源的请求计数应结合该字段解释。

### matched baseline 运行与重启快照（2026-09-30）

本组仍只比较两个中档交通协议下的纯 `sac_mlp`：`intersection_random_medium_v1` 对 `intersection_random_medium_p05_v1`。每项目标配置为 fresh 从零、训练 seed 0、CUDA、100,000 raw steps、`learning_starts=5,000` raw steps、每 10,000 raw steps 保存 checkpoint、final evaluation 100 episodes；其余 SAC+MLP 设置沿用既有 baseline。30/40/50 s warmup 采集适用于四个随机场景，这组训练只用两个中档。

初始配对 root 为 [`smlp_p05_a0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_a0930>)，supervisor PID 52540。首次进度快照中，medium worker PID 15900 仍在训练，raw=1,497、updates=0；该 worker 保留在 rootA，未停止。原 p05 worker PID 19128 在 raw=144 时退出，尚未到 `learning_starts=5,000`，因此不是方法训练失败或有效 100k 结果。失败日志保留于 `runs/smlp_p05_a0930/launcher_logs/medium_p05.stdout.log`。

**根因及修复：**第二个 episode 使用动态 schedule 时，临时 source 文件固定使用 stem `episode`；overlay 缓存因此复用了同一文件名，新的 `source_sha256` 与缓存 manifest 不符而拒绝使用。修复仅对动态 schedule overlay 名追加内容 SHA256 前 16 位，manifest 仍以完整 hash 验证；原静态 medium 的缓存命名逻辑未改。启动前真实 factory cache smoke [`cache_reuse_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_cache_smoke_0930/cache_reuse_smoke.json>) exit 0：同一 env seed 0→1 保持动态 source 路径不变，但 overlay 文件名分别包含 `__s74e00ee8bca038e0`、`__s7b180967f4946b83`；新 env 不同临时目录、同 seed 0 复现相同 schedule 内容 hash，manifest 全量校验通过并复用相同 overlay。三次 reset 均保留 3 个 warmup checkpoint，vehicle scale=1、额外车辆=0、endless=false。

p05 随后在新 root [`smlp_p05_b0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930>) 以 fresh-only 方式重启，worker PID 90004，启动时间 2026-09-30 11:27:15 +08。其 `arguments.json` 确认 method=`sac_mlp`、scenario=`intersection_random_medium_p05_v1`、budget=100,000、seed=0、CUDA、`smoke=false`。连续 progress 快照为 11:28:26 raw=1,193 / updates=0，11:29:26 raw=1,790 / updates=0；此时仍低于 5,000 warmup，0 updates 是预期探索阶段状态，不是训练失败。worker 已完成 9 个训练 episodes，各记录完整 30/40/50 s warmup checkpoints。原 rootA 的 medium PID 15900 仍保留；本记录不表示 medium 已停止。两路均没有 100k 完成或 final evaluation 结果，不能据此作策略性能或场景难度结论。

以上 launch、缓存失败/修复和 restart 是运行及数据路径证据；相同 seed、预算或同一算法身份本身不消除两个 traffic protocol 在计划请求强度与插入拥堵上的差异。后续结果应报告每个 root 实际完成 raw steps、checkpoint 与 100 episode evaluation，且不要把 pending 请求写成在网交通量。

### Factory 预启动环境检查（非训练结果）

主代理提供的 [`factory_smoke.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/p05fac_20260930_luna01/factory_smoke.json>) 状态为 `passed`，scope 是 base SAC+MLP 环境 factory reset；原 medium 和 p05 train 环境各执行一次 sampled-action step，没有 learning update。四个随机场景的 train seed=0 均记录 30/40/50 s 快照；p05 validation logical seed=0 映射到 SUMO seed 1,000,000,000，也记录三点；实际 `episodes.jsonl` 均含 reset_info warmup snapshots。train/validation traffic overlay 配置为 vehicle/pedestrian scale=1、无额外 flows、endless reinsertion=false，源 map/ego 未改。该检查只证明工厂/记录路径能运行，不是完整训练或策略评估。

单次 reset 观测如下（每个三元组按 30/40/50 s 顺序；`net` 是当前在网背景车，`due/inserted/pending` 是累计数，`halt` 为在网背景车速度 `<0.1 m/s`）：

| 场景/phase（SUMO seed） | 在网背景车 `net` | 停驶 `halt` | 请求到时 `due` | 实际插入 `inserted` | 路网外待插入 `pending` |
| --- | --- | --- | --- | --- | --- |
| low train (0) | 1/2/5 | 0/0/0 | 3/6/8 | 3/5/8 | 0/1/0 |
| medium train (0) | 4/4/8 | 0/0/0 | 7/12/15 | 7/11/15 | 0/1/0 |
| high train (0) | 10/8/10 | 0/0/0 | 16/22/27 | 16/21/27 | 0/1/0 |
| medium-p05 train (0) | 29/30/30 | 0/0/0 | 438/602/753 | 53/68/83 | 385/534/670 |
| medium-p05 validation (1,000,000,000) | 29/28/33 | 0/0/0 | 454/611/770 | 48/62/79 | 406/549/691 |

在这两个 p05 单 seed reset 中，每 route/depart slot 无重复；train 计划 1,952 个请求（按三路为 642/663/647），validation 计划 1,929 个（634/656/639）。50 s 时已到期请求分别为 753、770，实际插入为 83、79，其余 670、691 仍 pending；pending 明确在路网外。三个采样时点 ego 均未出现；halting 背景车数均为 0。该短 reset 显示 p05 的计划请求远高于当次可插入数，但它只是一个 seed/协议冒烟检查，不能据此估计长期平均流量、学习难度或策略成功率。

### 最新 matched-run 状态（2026-09-30 11:31:01 +08）

活动配对映射为 [`active_pair_manifest.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930/active_pair_manifest.json>)，p05 restart 参数、命令与故障链接见 [`restart_manifest.json`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930/restart_manifest.json>)。medium worker PID 15900 在 rootA [`smlp_p05_a0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_a0930>)，状态 training，raw=23,619、updates=18,619、79 完整训练 episodes；p05 fresh worker PID 90004 在 rootB [`smlp_p05_b0930`](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/smlp_p05_b0930>)，状态 training，raw=3,886、updates=0、13 完整训练 episodes。p05 的 `arguments.json` 确认 100,000 raw budget、seed=0、CUDA、`smoke=false`、`resume=false`；updates=0 是仍未达到 `learning_starts=5,000` 的正常状态。rootB 中 p05 从零重启，rootA 中 medium 仍继续；原 p05 raw=144 的失败 attempt 保留于 rootA。两路均无 100k 完成或 final evaluation 结果。本条仅为运行快照，不是策略性能结论。

## 2026-09-30 新增：p03 / p02 逐步发车变体

| 场景 | 每条背景进口每 0.1 秒 p | 每进口请求量（辆/小时） | 三路总请求量（辆/小时） |
|---|---:|---:|---:|
| intersection_random_medium_p03_v1 | 0.3 | 10800 | 32400 |
| intersection_random_medium_p02_v1 | 0.2 | 7200 | 21600 |

两个场景直接沿用 p05 的 `intersection_bernoulli_step_schedule_v1` 协议，通过 `STEP_PROBABILITIES` 注册，动态预生成按时间排序的显式车辆请求；不是 SUMO 原生 probability flow。三条进口均采用同一档位 p，各路线独立采样。种子派生仍为协议常量、实际 SUMO seed、flow ID、arrival，不加入场景名或 p；同 seed、同路线下，p02 发车时刻集合包含于 p03，p03 包含于 p05。

固定项：路网、自车路线 -E1→-E0、现有 120 种车辆及三组按路线划分的驾驶员分布；dt=0.1 秒，背景请求时段 0–130 秒，自车计划 50 秒入场，控制上限 600 raw 步，action_repeat=3，depart_scale=1，背景车驶离后退出，不循环补车。沿用 train/validation/test 种子分区；所有随机场景正常预热都会记录 30/40/50 秒的在网车辆、停驶、实际入场和网外 pending。动态 overlay 缓存继续使用 schedule SHA 区分各回合。

本轮正式 SAC+MLP pair 位于工作区 `runs/smlp_p03p02_0930`，各从零 100k raw、seed0、CUDA、最终 100 回合 validation。仅新增上述两个 profile，原四个 profile 配置哈希保持不变；此前提出的 v2 三档建议未创建。本节参数是用户选择的实验设置，不表示已达到 30%–60% 成功率目标。启动与验证证据见 `experiment_history.md` 第 15 节。

## 2026-09-30 新增：基于现有三档模板的 DARRL 参数参照组

独立新场景：`intersection_random_darrl_low_v1`、`intersection_random_darrl_medium_v1`、`intersection_random_darrl_high_v1`。对应现有low/medium/high模板，使用每路每0.1秒p=0.03/0.05/0.07的显式seeded Bernoulli日程，背景出发与抵达均lane1，ego minGap=1 m，全体相关vType jmIgnoreFoeProb=0，SUMO collision.action=remove。

按用户确认，保留当前SUMO ego碰撞事件或几何重叠判据；不采用DARRL legacy距离/自车消失规则。其他路网/路线、ego起始lane2、50秒释放、时限、观察动作与训练评估协议不变。旧六场景继续使用原协议与资产。新组完整配置、来源与验证入口见 [新增场景说明](random_intersection_darrl_v1_setup.md)。本组难度档位暂为请求量标签，尚无训练成功率标定。

## 2026-09-30 当前更新：同名 DARRL 场景 r2

用户要求原位调整：`intersection_random_darrl_low_v1` / `intersection_random_darrl_medium_v1` / `intersection_random_darrl_high_v1` 的每路每0.1秒发车概率现为 **0.015 / 0.03 / 0.05**。configuration_revision=darrl_r2_20260930；旧r1=.03/.05/.07已连同资产和源码归档。同名场景必须结合实验traffic_config/revision解释，不能用当前配置回填旧结果。

同时修复旧运行的success/collision双标及成功奖励误发：在所有场景特定事件重算后统一采用collision>off_route>success>timeout，terminal_outcome_protocol=exclusive_terminal_v2，原生事件仍保留在正常过程日志。其余车道/车型/地图/seed协议保留。详见[r2配置与修复记录](darrl_r2_configuration_and_outcome_fix_20260930.md)；旧说明页标为历史r1，原数值不覆盖。
