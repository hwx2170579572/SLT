# fast-developer 目录说明

这是 **intersection「让行」任务**的快速开发/实验区。所有文件围绕一个目标：让
`mst_slt`（Scene-Rep，连续动作头）与 `hold35k`（v4_8 混合动作头）在无信号四臂交叉口
学会「减速让行、择机通过」。约定：**不改动项目既有文件**，所有新代码/产物都放这里。

> 整理时间：2026-09-27。分类原则：静态产物物理分目录；`.py` 脚本因 import 强耦合
> 有意保持扁平（见下方「为什么 `.py` 不物理分目录」）。

## 目录结构

| 目录 | 内容 |
|---|---|
| `analysis/` | 根因分析与实验结果文档（`.md`） |
| `logs/` | 训练/评估运行日志（`.log`） |
| `artifacts/` | 大产物：BC 预训练模型（`.zip`）、演示数据（`.pkl`） |
| `_scratch/` | 临时验证副产物（overlay 场景、speed probe），验证脚本每次重建，可安全删除 |
| 根目录 `*__*` | 各方法训练产物（checkpoint / train_monitor.csv / evaluation） |
| 根目录 `.py` | 实验脚本（见下） |

## 核心产物（当前活跃 —— yield_v2 任务）

这是最新一轮「obs 修复 + reward shaping」的实现，是后续训练/泛化的起点：

| 文件 | 用途 |
|---|---|
| `yield_obs_env.py` | **改法 1+2**：`_YieldObsMixin` 让 ego map 穿过左转 internal 弧线、ego 车道排最前；`YieldObsIndependentV2EnvV1`（base=mst_slt）、`YieldObsIndependentV2EnvV4V1`（v4=hold35k） |
| `yield_conflict_env.py` | **改法 3**：`_ConflictAwareNeighborMixin` 用「接近时间 TTC」冲突相关性选 neighbor（替代纯欧氏距离）；`YieldConflictIndependentV2EnvV4V1`（仅 hold35k） |
| `reward_shaping_v2.py` | **可泛化 reward shaping v2**：success+10 / collision-10 / off_route-10 / timeout-5 + progress 0.02/米 + step_cost 0.01 |
| `train_intersection_yield_v2.py` | **yield_v2 训练脚本**：hold35k=改法1+2+3、mst_slt=改法1+2，两者都包 `GeneralizedRewardShapingWrapper` |
| `reward_shaping_wrapper.py` | P1 旧版 reward shaping（timeout=-1 + 进度 0.01 + 生活成本 0.005），已被 `reward_shaping_v2.py` 取代，仅作对比保留 |

## 验证脚本（下划线 `_` 前缀 = 一次性验证/诊断，非训练入口）

| 文件 | 用途 |
|---|---|
| `_yield_obs_verify.py` | 改法 1+2（ego map）验证 |
| `_yield_conflict_verify.py` | 改法 3（neighbor TTC）单元 + MRO smoke 验证 |
| `_reward_shaping_verify.py` | reward v2 单元（6 断言）+ 停车 smoke 验证 |
| `_p2_nointervention_verify.py` | 复现「SUMO 默认让行 30/30」 |
| `_p2_gap_diagnose.py` | ego 接近路口时冲突车流 ETA/gap 分布诊断 |
| `_p2_ttc_gap_controller.py` | 规则基线：TTC gap 检测让行控制器（验证让行本质可达性） |
| `_p2_speedmode_verify.py` / `_p2_yield_verify.py` / `_p2_quick_verify.py` | speed mode / 让行 / 快速验证 |
| `_p2_diag_bc_speed.py` / `_p2_eval_bc.py` | BC 速度诊断 / BC 评估 |
| `_p4_density_verify.py` | 密度-成功率映射验证（scale 4/3/2/1.5/1 → 70/63/23/20/22-30%） |

## 历史训练脚本（各阶段入口）

| 文件 | 用途 |
|---|---|
| `train_intersection_hold35k_mst.py` | 原始 hold35k + mst_slt 训练（加速超参版，已证明失败） |
| `train_intersection_hold35k_mst_fixed.py` | **修复超参版**（lr/batch/warmup 恢复）——被几乎所有其它脚本 `import ... as base` 复用 |
| `train_intersection_hsac_mlp.py` | HSAC-MLP 基线训练 |
| `train_intersection_mst_slt_rs.py` | P1：mst_slt + RewardShapingWrapper 训练 |
| `train_intersection_mst_slt_bc_rs.py` | P2：加载 BC 预训练 actor + reward shaping 微调 |
| `train_intersection_mst_slt_curriculum.py` | P4：低密度 traffic 起步课程学习 |

## BC / 演示采集（P2 链路）

| 文件 | 用途 |
|---|---|
| `collect_yield_demos.py` | 采集「SUMO 默认让行」示范轨迹 → `artifacts/yield_demos*.pkl` |
| `bc_pretrain_actor.py` | 用示范 BC 预训练 mst_slt actor → `artifacts/mst_slt_bc_pretrained.zip` |

## 训练产物目录

| 目录 | 对应训练 |
|---|---|
| `hold35k__intersection_1/` | hold35k 原始训练 |
| `mst_slt__intersection/` | mst_slt 原始训练 |
| `mst_slt__intersection_fixed/` | mst_slt 修复超参版 |
| `mst_slt__intersection_rs/` | mst_slt + P1 reward shaping |
| `mst_slt__intersection_curriculum/` | mst_slt P4 课程学习 |
| `res_isxn_hsac/` | HSAC-MLP 训练 |
| `tb/` | TensorBoard 事件 |
| `_p4_lowdensity_s{1p5,2p0,3p0,4p0}/` | P4 低密度 traffic 变体（depart×scale） |

## 为什么 `.py` 不物理分目录

几乎所有脚本通过同目录 import 互相引用，物理移动会破坏运行：

- 几乎所有训练/诊断脚本：`import train_intersection_hold35k_mst_fixed as base`
- `yield_conflict_env.py` → `from yield_obs_env import ...`
- `train_intersection_yield_v2.py` → `from reward_shaping_v2 / yield_conflict_env / yield_obs_env import ...`
- 各验证脚本 → `from train_intersection_yield_v2 / yield_obs_env / yield_conflict_env import ...`

它们依赖「脚本目录在 `sys.path` 里」这一前提（各脚本内部 `sys.path.insert` 同目录）。
因此**逻辑分类靠本 README 索引**，物理位置保持扁平。

## 实验脉络速览

1. 初始训练（加速超参）→ 成功率极低 → `analysis_hold35k_mst_low_success.md`
2. 修复超参 → 仍 1% → 根因是稀疏三元 reward → `analysis_mst_slt_fix_failure_rootcause_v2.md`
3. P1 reward shaping → 打破起点盆地但滑入「路口等超时」→ `analysis_p1_reward_shaping_result.md`
4. P2 BC 示范 + P4 密度课程 → 定位 obs 盲区 + 密度上界
5. **yield_v2（当前）**：obs 改法 1+2+3 + reward v2，实现+验证完成，待正式训练

完整时间线与命令见 [`experiments_log.md`](experiments_log.md)。
