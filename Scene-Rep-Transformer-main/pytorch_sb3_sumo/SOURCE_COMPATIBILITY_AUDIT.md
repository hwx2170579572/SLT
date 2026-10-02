# PyTorch + Stable-Baselines3 + SUMO 源码一致性审计

审计日期：2026-08-04

## 结论

在用户允许“仿真环境底层细节可以不一致”的边界内，`scene_rep`、`mst` 和 `ppo`
已经达到发布源码可恢复契约的一致：算法模块、梯度边界、张量接口、动作/奖励、原始步
时钟、回放语义和默认场景设置均有对应实现与测试。它们不是 TensorFlow 与 PyTorch
之间的逐位数值复刻，SUMO 车辆动力学也不是 SMARTS/Bullet 或 CARLA 的逐帧复刻。

`sac` 只能判定为“有依据的重建”，不能判定为发布源码逐实现一致。论文列出了 LSTM
SAC 基线，仓库也提供了 `STATE_LSTM` 适配器和 `RLEncoder`，但没有发布一个可选择、
可运行的 plain-SAC 配置。迁移版明确标记为
`paper_sac_lstm_reconstruction_v2`，没有把缺失实现冒充成源码。

## 代码边界与原源码保护

- 原源码根目录：`../`
- 独立迁移源码根目录：当前目录 `pytorch_sb3_sumo/`
- 旧混放副本归档：工作区同级
  `_pre_separation_interleaved_sb3_sumo_backup_20260804/`
- 迁移版脚本把当前目录插入 `sys.path` 首位，训练、评估、测试和协议文件均从当前目录
  解析，不依赖归档副本。
- `MIGRATION_ORIGINAL_SHA256.json` 记录迁移前 111 个文件。修正后的
  `tools/verify_migration_invariant.py` 从父目录核验原源码；结果为 111/111 未改变，
  0 缺失、0 变化。
- 旧副本采用移动归档而非删除，操作可恢复。数 GB 历史结果与临时目录未移动，也不被
  任何迁移模块作为源码导入。

## 算法结构核对

| 项目 | 发布源码契约 | PyTorch/SB3 实现 | 判定 |
|---|---|---|---|
| Scene-Rep critic encoder | critic 持有分层场景编码器 | `HierarchicalSceneExtractor` 仅由 critic 持有 | 一致 |
| actor/encoder 梯度 | actor 使用 `stop_gradient` 编码特征 | actor 前向对 critic 特征 `detach` | 一致 |
| 层级关系 | 时间、地图、邻车—地图、ego—actors、ego—目标路径注意力 | 五级关系均保留；Keras `key_dim` 投影语义单独实现 | 一致 |
| Q 网络 | twin Q；动作先嵌入 64，再接 128/32/1 | 两个独立 Q，64 维动作分支与 128/32/1 | 一致 |
| SLT | 动作条件未来表征；projector/predictor；负余弦损失 | 独立表征优化器，共享在线 encoder，one-step future 输入 | 一致 |
| `cross` 特例 | 无随机旋转；target critic；独立 target projector | 三项均按场景开启，projector 做 Polyak 更新 | 一致 |
| `roundabout` 特例 | 跳过邻车未来路径融合 | `no_neighbor_future=True` | 一致 |
| Scene-Rep/MST 区别 | MST 不构造/训练 SLT | MST 在模型构造前关闭 SLT，不消耗其随机数 | 一致 |
| SAC 更新 | twin Q、自动 alpha=0.2、tau=.005、gamma=.99、NAdam、逐张量 clipnorm=5 | 均保留；alpha 使用源码直接对 `exp(log_alpha)` 求导 | 一致 |
| replay | 4-step；策略动作保持 3 个 raw step | 4-step bootstrap；保持期 reward 折扣；每 raw step 更新 | 一致 |
| Proposed 尾队列 | `future_steps=1` 时终止 transition 再写一次 | 仅 Scene-Rep 开启尾 transition 重复，MST 不开启 | 一致 |
| PPO 网络 | SMARTS 为 80×80×3 CNN；CARLA 为 6×10×5 GRU + 6-head relation attention | 两种 extractor 分开实现 | 一致 |
| PPO 梯度 | actor loss 不更新共享 encoder，value loss 更新 | actor latent `detach`，value latent保留梯度 | 一致 |
| PPO 概率语义 | collection 使用 tanh Jacobian；update 直接把已压缩 action 当 Normal 样本 | 保留源码中的非对称行为 | 一致（含源码行为） |
| PPO 时钟 | horizon=512、epoch=10、lr=5e-4；CARLA action/value/logp 保持 3 raw step | 同值；环境仍逐 raw step产生 transition | 一致 |
| plain SAC | 发布仓库无完整可运行选择 | 从 `STATE_LSTM` + `RLEncoder(GRU)` 重建 | 部分一致，非源码复刻 |

框架替换后仍不可声称逐位相同的项目包括：TensorFlow/PyTorch 随机数流、Keras 与
PyTorch NAdam 的实现细节、浮点归约顺序，以及不同仿真器产生的状态轨迹。

## 内部输入输出核对

| 接口 | 形状/语义 | 判定 |
|---|---|---|
| Scene-Rep 动态轨迹 | `(6,10,5)`；ego + 最近 5 个 actor；`[x,y,heading,vx,vy]` | 一致 |
| SMARTS 类地图 | `(12,10,5)`；每 actor 两条路径；`[x,y,heading,is_ego,is_neighbor]` | 一致 |
| CARLA 类地图 | `(18,10,2)`；每 actor 三条路径；`[x,y]` | 一致 |
| plain-SAC 状态 | `(10,30)` + `(10,)` 时间 mask；ego/social 字段语义分别恢复 | 与已发布适配器一致 |
| SMARTS PPO | 单帧 `(80,80,3)` float32 RGB；保留源码的 `/255` 归一化及训练 reset 首帧尺度 bug | 一致 |
| CARLA PPO | `(6,10,5)`，保持源码 `make_rotation=False` 的字段重解释 | 一致 |
| 动作 | Box `(2,)`，范围 `[-1,1]`；速度映射 0--10 m/s；横向阈值严格为 ±1/3 | 一致 |
| 横向符号 | SMARTS：-1 右、+1 左；CARLA：-1 左、+1 右 | 按场景转换，一致 |
| reward | 成功 +1、碰撞 -1、其他 0 | 一致 |
| info/终止 | `(finish, collision, off_route, max_time)`；SMARTS timeout 可 bootstrap，CARLA timeout terminal | 一致 |
| 评估 return | 按 raw step 的未折扣奖励求和 | 通过 `undiscounted_reward` 保留，一致 |

padding mask 严格使用源码的 `tf.not_equal(value, 0)[..., 0]`：只看第一坐标，精确比较
零。审计时发现迁移版曾使用 `abs(x) > 1e-8`，会把极小非零值误判为 padding；现已在
层级 encoder、SLT sample mask 和 CARLA PPO actor mask 三处统一修正，并覆盖 tiny
nonzero/NaN 回归用例。

## 场景设置核对

| 场景 | source 上限 | ego 发车 | 资源/任务判定 |
|---|---:|---:|---|
| `left_turn` | 400 | 15 s | 发布版 `map.net.xml` 与全部 traffic routes；mission 一致 |
| `cross` | 600 | 5 s | 发布版 double-merging 网络与 traffic routes；mission 一致 |
| `roundabout_easy` | 400 | 30 s | 发布版低流量环岛资源；mission 一致 |
| `roundabout_medium` | 600 | 30 s | 发布版中流量环岛资源；mission 一致 |
| `roundabout` | 1000 | 30 s | 发布版高流量环岛资源；mission 一致 |
| `carla` | 302 | 0 s | 任务、输入输出、目标框、奖励/终止一致；SUMO 网络与交通为重建 |

SUMO reset 会先推进到 ego 的原发车时刻，但 ego 插入后把回合 raw-step 计数重置为
0，因此等待 5/15/30 秒不会占用源码回合步数上限。五个 SMARTS 场景默认按源码的
词典序、seed-42 滚动顺序使用全部交通文件，并复现 endless social traffic 的再注入。

默认 `--episode-limit-profile source --traffic-protocol source_all` 才是源码设置。
以下选项是显式实验协议，不应被描述为源码完全一致：

- `episode-limit-profile=paper` 把 `cross` 从 600 改为 400、`roundabout` 从 1000
  改为 800，其余场景不变；
- `traffic-protocol=frozen_80_20` 保留每第五个 traffic XML 做评估；
- `ego-control-profile=smarts_ackermann_proxy` 是 SUMO 中的动力学代理。

这些差异都会写入运行参数/`info`，不会静默混入 source profile。

## 本轮发现并修正的问题

1. 将三处 `abs(x)>1e-8` 改为源码精确的 `x!=0`。
2. CLI 根据算法自动选择动作重复：off-policy=3、PPO=1；PPO 显式传 3 会报错，
   CARLA 的三步 hold 只在 `SourcePPO` 内执行一次。
3. 新增 `SourceEvaluationCallback`，SB3 周期评估期间关闭随机旋转，结束后恢复训练
   状态，对齐 `set_configs(..., test=True)`。
4. 修正独立目录中的原源码哈希核验路径，使其核对父目录而不是错误核对迁移目录。
5. 训练后端改为显式注入 paper env factory，不再修改模块全局函数；paper 入口的默认
   输出固定在独立目录的 `results_sb3_sumo_paper/`。
6. 通用 vector-SUMO 评估入口不再声称支持需要 RGB/CARLA wrapper 的 PPO；PPO 统一
   使用 `train_paper_sb3_sumo.py` / `eval_paper_sb3_sumo.py`。

## 验证标准

审计通过以下层次验证：

- 源码不变量：111/111 SHA-256 不变；
- 单元/协议测试：encoder、attention、GRU、SAC/SLT、PPO、replay、raw-step callback、
  checkpoint、评估与协议矩阵；
- 真 SUMO 场景测试：六个场景 reset/step、目标可达、交通参与者、地图路径、终止条件；
- Python 编译检查：`algos`、`configs`、`envs`、`tools`、`tests_sb3_sumo`。

本轮实测为 104/104 通过；此外分别以默认动作重复运行 Scene-Rep 与 PPO 的真实 SUMO
`--check-only`，两条入口均返回 `status=ok`。5 条 SB3 warning 是通用检查器把三维
float 轨迹/地图当作图像，不是契约失败；自定义 extractor 明确关闭图像归一化。

最终验收命令与最新通过数量记录在根 README 的“验证”一节；若未来修改任一契约，
应同时更新测试和本审计，不应只改文档结论。
