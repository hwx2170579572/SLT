# 2026-09-30：基于现有三档模板新增 lane 1 场景

> **历史 r1 配置记录。** 本页的 p=0.03/0.05/0.07 对应原运行 `runs/smlp_darrl_0930`。
> 用户后来要求原位调整相同场景名；当前 r2 为 p=0.015/0.03/0.05，并修复终局双标。
> 当前配置与旧实验限制见 [r2 配置和修复记录](darrl_r2_configuration_and_outcome_fix_20260930.md)。

## 用户确认的范围

以现有 `intersection_random_low_v1`、`intersection_random_medium_v1`、
`intersection_random_high_v1` 分别为模板新增独立场景。DARRL 仅作为指定参数的
参照，不替换当前路网、观察动作接口、奖励、训练与评估协议。

用户随后明确：**保留当前碰撞判据，只改场景参数**。因此不移植 DARRL 的
legacy 距离/自车消失启发式，也不将 `source_observation_contract` 改成 legacy。

## 三档配置

| 新场景 | 原模板 | 每路每 0.1 秒请求概率 p | 每路期望请求量（辆/小时） | 三路合计（辆/小时） |
|---|---|---:|---:|---:|
| `intersection_random_darrl_low_v1` | `intersection_random_low_v1` | 0.03 | 1080 | 3240 |
| `intersection_random_darrl_medium_v1` | `intersection_random_medium_v1` | 0.05 | 1800 | 5400 |
| `intersection_random_darrl_high_v1` | `intersection_random_high_v1` | 0.07 | 2520 | 7560 |

概率使用现有逐步、逐路线独立的 seeded Bernoulli 显式车辆日程生成器，
不是将 0.03/0.05/0.07 直接写成 SUMO 原生 `<flow probability>` 的每秒参数。
每路请求量为 `p / 0.1 * 3600`，实际入场由 SUMO 插入条件与容量决定。
同一 SUMO seed 的三档请求时间采用相同均匀随机数流，低档是中档子集，中档是高档子集；
这不保证不同档位的实际轨迹、车型抽样顺序或入场时间完全配对。

共同设置：

- 背景路线仍为 `-E3 → -E0`、`E0 → E3`、`E2 → E1`。
- 背景车辆 `departLane="1"`、`arrivalLane="1"`；`departPos="0"`、`departSpeed="max"` 保留。
- 保留 30 个源模板合成的 120 个背景车型及路线条件分布，仅将所有背景车型的 `jmIgnoreFoeProb` 显式设为 `0`。
- 自车原路线 `-E1 → -E0`、起始 lane 2、初速 0、计划 50 秒出发保持；自车车型 `minGap="1"`、`jmIgnoreFoeProb="0"`。
- 运行时 SUMO 命令使用 `--collision.action remove`，保留 `--collision.check-junctions true`。
- episode 碰撞判据仍为涉及 ego 的 SUMO 碰撞事件或 ego 几何重叠；纯背景碰撞不直接结束 episode。
- 背景发车请求时段 `[0,130)` 秒；dt=0.1 秒；episode 上限 600 raw steps；默认 action repeat=3；不循环背景交通；depart_scale=1。
- 保留原训练/验证/测试 seed 域，以及正常预热 30/40/50 秒记录；新场景的进口统计跟随 lane 1。

`arrivalLane=1` 是离网车道要求，不是全程禁止背景车换道。`minGap=1 m` 是 SUMO
车辆参数，不是将本项目所有碰撞检测都替换成 1 m 距离阈值。
SUMO 的 remove 行为与车辆参数定义参见
[SUMO Safety](https://eclipse.dev/sumo/docs/Simulation/Safety.html) 和
[Vehicles and routes](https://eclipse.dev/sumo/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html)。

## 来源与可复现性

- 主实现：`../envs/sumo/random_intersection.py`（相对 `fast-developer` 的路径）；
  `get_random_intersection_config` 记录每个新场景的 `template_scenario`、覆盖字段与碰撞口径。
- 资产位于 `pytorch_sb3_sumo/envs/sumo/original_scenarios_v1/<新场景>/`，
  每个目录的 `scenario_manifest.json` 保存完整配置、源资产与生成资产 SHA256。
- 实际 SUMO 启动参数由 `paper_env.py` 按新场景配置读取；旧场景缺省仍为 `collision.action=none`。
- DARRL 参照路径为用户提供的 `D:\Program Files (x86)\paper\DARRL-main`。
  其中 `Environment/environment/envs/traffic_env.py` 与 `legacy_compat.py` 的 legacy
  判据是六方向中心距离阈值（前 2 m、后 1.5 m、侧 1 m）或自车缺失，
  `Data/Intersection_1.sumocfg` 使用碰撞移除。这些判据没有移植到本次新场景。

三档名称表示本组的请求量档位，**尚未用学习策略的成功率标定难度**；
本次不启动 100k 训练，不将短时环境验证当作模型性能实验。

## 验证结果

- 原六场景的规范化配置SHA256均未改变；原资产仍由已有manifest校验。新三场景通过manifest记录独立生成的资产。
- 场景回归：`test_random_intersection_darrl`、`test_random_intersection_step_profiles`、`test_random_intersection_p05`、`test_random_intersection_scenarios`共16项通过。
- 碰撞与命令回归：新增4项及原启动参数回归1项共5项通过。覆盖旧场景none/新场景remove、自车已移除但SUMO碰撞事件仍在、正常抵达、纯背景碰撞不误结束ego。
- 真实运行：`validate_random_intersection_darrl.py`采用D1补丁后的`base.make_env_factory("base", ...)`，即SAC+MLP的实际环境工厂，三档均完成reset（含30/40/50秒预热记录）和最多5个决策步；不创建模型、不训练。validation split，logical seed=10000。结果status=passed、cases=3、进程退出码0。
- 通过的原始证据：工作区 `runs/darrl_profiles_preflight_0930b/validation_report.json`；该目录保留各档实际加载的episode route日程和配置。
- 首次验证脚本曾错误要求运行时ego type ID与XML声明逐字相同，实际返回`scene_rep_ego_type@ego`。已改为分别记录两者并核对真实TraCI minGap=1及已加载车型参数定义；首次失败报告保留在`runs/darrl_profiles_preflight_0930/validation_report.json`。这不是模型训练失败，也不是三档难度比较结果。

以上是实现与接口验证，不是成功率标定。后续模型运行仍需使用正常固定训练/评估协议判断场景难度。

### 本次单个 validation seed 的预热观测

50秒时ego尚未插入。下表仅为环境smoke的一个seed，不用于给出三档难度或模型表现结论：

| 档位 | 背景在网 | 累计请求due | 累计实际入场 | 待插入pending | 在网停驶队列 |
|---|---:|---:|---:|---:|---:|
| low（p=0.03） | 20 | 43 | 43 | 0 | 0 |
| medium（p=0.05） | 31 | 76 | 65 | 11 | 0 |
| high（p=0.07） | 31 | 99 | 68 | 31 | 0 |

三档的30/40/50秒每个检查点均满足`due=inserted+pending`。`pending`表示尚未成功进入路网的请求，不能与在网停驶队列混为一谈。实际factory的vehicle/ped scale均1.0，overlay新增显式车辆为0；三档各5个决策步均未终止。

TraCI确认ego运行时minGap=1.0；实际加载的ego车型定义与全部120个背景车型定义均为jmIgnoreFoeProb=0，车辆日程均为departLane=arrivalLane=1。本次TraCI通用getParameter对jmIgnoreFoeProb返回空字符串，因此该参数的验证证据是已加载XML定义，不应写成“TraCI已读回0”。
