# intersection 让行任务 —— 实验日志

> 记录从初始训练到当前 yield_v2 的完整实验脉络。每条含：日期 / 阶段 / 脚本 / 配置 /
> 结果 / 结论。关联分析文档在 `analysis/`，代码产物在根目录。

---

## 阶段 1：初始训练（加速超参）—— 失败

- **日期**：2026-09-26 ~ 09-27
- **入口**：`train_intersection_hold35k_mst.py`（hold35k + mst_slt）、`train_intersection_hsac_mlp.py`（HSAC-MLP）
- **配置（加速超参）**：lr=3e-4（无衰减）、batch=256、learning_starts=2000、action_repeat=3、5w raw 步
- **结果**：
  - hold35k：ep1-8 全碰撞 → ep17-32 全 timeout（0 撞 16 超时），30% 预算即坍缩
  - mst_slt：前 10 ep 100% 碰撞 → ep27 见顶 success_last20=0.25 → 最后 24 ep 全 timeout
  - 两者均无 `final_model.zip`（训练未跑完）
- **产物**：`hold35k__intersection_1/`、`mst_slt__intersection/`、`res_isxn_hsac/`
- **结论**：超参加速破坏训练稳定性（ent_coef 0.2→0.0086 单调衰减 23 倍，探索提前耗尽）
- **分析**：`analysis/analysis_hold35k_mst_low_success.md`

## 阶段 2：修复超参（fixed）—— 仍失败，根因重定位

- **日期**：2026-09-27
- **入口**：`train_intersection_hold35k_mst_fixed.py`
- **配置**：lr 5e-5（hold35k）/ 1e-4（mst_slt）+ 深地板衰减 2e-5、batch=32、learning_starts=5000
- **结果**：mst_slt 评估 success_rate=0.01、timeout=0.83、collision=0.16；102 ep 仅 1 次成功
- **产物**：`mst_slt__intersection_fixed/`
- **结论（关键反转）**：根因是环境的**稀疏三元奖励** `raw_reward=success-collision`（中间步全 0、timeout=0），
  不是超参。「停车等超时」是价值恒 0 的零风险吸收盆地，严格支配早期价值为负的抢行策略。
  提出 P1（reward shaping）/ P2（BC 示范）/ P3（熵地板）/ P4（课程学习）/ P5（加预算）
- **分析**：`analysis/analysis_mst_slt_fix_failure_rootcause_v2.md`

## 阶段 3：P1 reward shaping —— 打破起点盆地，但滑入第二盆地

- **日期**：2026-09-27
- **入口**：`train_intersection_mst_slt_rs.py` + `reward_shaping_wrapper.py`
- **配置**：mst_slt（lr=1e-4/batch=32/warmup=5000）+ timeout→-1、进度 0.01/米、生活成本 0.005
- **结果**：
  - timeout 从 83% 降到 26%（打破起点吸收盆地）
  - 但 success 仍 1.5%（66 ep 仅 1 次），timeout 后期回升（ep51-66 ~90%）
  - 新盆地：「前进到路口后停下等超时」（timeout return≈-1.3，ego 已前进 ~70m）
- **产物**：`mst_slt__intersection_rs/`、`logs/mst_slt_rs__train.log`
- **结论**：进度奖励只引导「前进」不引导「择机通过」，稀疏 success 无法支撑时序决策
- **分析**：`analysis/analysis_p1_reward_shaping_result.md`

## 阶段 4：P2 BC 示范 + P4 密度课程 —— 定位 obs 盲区与密度上界

- **日期**：2026-09-27
- **P2 入口**：`collect_yield_demos.py`（采集 SUMO 默认让行示范）、`bc_pretrain_actor.py`（BC 预训练 actor）、`train_intersection_mst_slt_bc_rs.py`（BC + shaping 微调）
- **P4 入口**：`_p4_density_verify.py`、`train_intersection_mst_slt_curriculum.py`
- **关键发现**：
  1. **SUMO 默认让行可达**：不干预 ego（保持 speed mode 31）时 30/30 success（161-168 步），
     证明「让行」本身可解，问题在 RL 探索/obs/reward
  2. **密度瓶颈**：scale 4.0/3.0/2.0/1.5/1.0 → 70%/63%/23%/20%/22-30%，临界密度 ~2.5
     （gap ≈ ego 穿过时间 2.7s）
  3. **obs 盲区**：neighbor 纯欧氏距离把 75m 外迎面冲突车挤出 top-5；map 前视 10m 且
     不含 internal junction 边，ego 左转弧线不在自己 map 里 → 引出改法 1+2+3
- **产物**：`artifacts/yield_demos.pkl`、`artifacts/yield_demos_ms10.pkl`、
  `artifacts/mst_slt_bc_pretrained.zip`、`mst_slt__intersection_curriculum/`、
  `_p4_lowdensity_s{1p5,2p0,3p0,4p0}/`
- **诊断脚本**：`_p2_*`、`_p4_density_verify.py`

## 阶段 5：yield_v2（当前）—— obs 改法 1+2+3 + 可泛化 reward v2

- **日期**：2026-09-27
- **消融设计（拍板）**：改法 3（neighbor TTC）只对 hold35k 生效，mst_slt 保持改法 1+2
  - hold35k → `YieldConflictIndependentV2EnvV4V1`（改法 1+2 + 改法 3）
  - mst_slt → `YieldObsIndependentV2EnvV1`（仅改法 1+2）
  - 两者都包 `GeneralizedRewardShapingWrapper`
- **改法 1+2（obs map）**：`yield_obs_env.py` —— ego map 走无约束车道图穿过 left-turn internal 弧线、ego 车道排最前
- **改法 3（neighbor）**：`yield_conflict_env.py` —— 用「接近时间 TTC」冲突相关性选车（替代纯欧氏距离）；
  速度用 TraCI `getSpeed`+`getAngle` 读真实笛卡尔速度（规避 smarts 契约下被旋转 -90° 的 `state[3:5]`）
- **reward v2**：`reward_shaping_v2.py` —— success+10 / collision-10 / off_route-10 / timeout-5
  + progress 0.02/米 + step_cost 0.01（终局压倒性、过程只给梯度、所有信号跨场景通用）
- **入口**：`train_intersection_yield_v2.py`（RUN_SUFFIX=`_yield_v2`；raw 5w 步、batch 32、
  lr 5e-5/1e-4、warmup 5000、action_repeat 3；先 `--smoke`；方法可选 `--method`）
- **密度（depart×2.0，用户拍板）**：不用 `vehicle_scale` 加密（保持 1.0，no-op overlay），
  改用 `DEPART_SCALE=2.0` 把原始 traffic 每个 vehicle 的 `depart` 时间 ×2.0（发车间隔放大
  = 密度减半），经 `_partitioned_traffic_paths` monkey-patch 注入（train/eval 同密度）。
  对应密度-成功率映射 scale 2.0→23%（在临界密度 ~2.5 之上，让行可学的目标区间）
- **验证（全部通过）**：
  - `_reward_shaping_verify.py`：单元 6 断言 + 停车 smoke（return=-7 打破吸收盆地）
  - `_yield_conflict_verify.py`：TTC 单元（冲突车 A 从欧氏第3→冲突第1）+ v4 环境 MRO smoke
  - `_yield_obs_verify.py`：改法 1+2 验证
  - 对抗审查（4 维度 finder + 2 票 verify）：0 个 finding，确认 TTC 坐标系 / isinstance / traffic 索引 / 评估交互均正确
- **结果（mst_slt depart×2.0，正式训练完成，2026-09-27）**：
  - final_model 评估 100 集：**success_rate 0.59**（59/100）、collision 0.41、off_route 0、timeout 0
  - best_training_success_model：训练中 success_last_20 峰值 **0.70**（@31734 raw 步 / 第 115 集）
  - 对比：同密度（scale 2.0）Krauss 默认让行上界仅 **23%** → mst_slt 学到**优于 Krauss 的让行策略**
  - 产物：`mst_slt__intersection_yield_v2/`（final_model.zip 17.7MB、best_training_success_model.zip、
    checkpoints/、training_curve.png、train_monitor.csv、evaluation_results.json）
- **结果（hold35k 改法 1+2+3，正式训练完成，2026-09-27）**：
  - 训练 5w 步完成（wall 3.5h），但**坍缩到「停车等超时」timeout 盆地**：ep20 起 68+ 集
    全 timeout（return≈-5.6，零碰撞，0% success）——不是「抢行撞车」，是「停止」
  - 对比 mst_slt 的「让行/抢行误判撞车」（41% 撞 / 0% 超时），两者质的不同：hold35k 学「停止」
  - 主因：混合头（离散车道 + 车道条件化速度）为多车道换道设计，单车道让行里车道维度冗余、
    速度时序学习能力弱 + timeout 安全盆地坍缩不可逆；头本身在多车道 cross/carla 达标（反证）
  - **lr 校验 bug**：训练完成后 `verify_optimizer_settings` 用初始 lr 5e-5 校验，但
    StabilitySAC 已衰减到 2e-5，抛 mismatch 导致 final_model.zip 未保存（已修复，用
    `learning_rate(config, raw_budget)`；`checkpoints/ckpt_raw_50000_steps.zip` 与 final 等价）
  - **完整评估（final_model，2026-09-27，100 集完整重评估）**：**success 0/100、timeout 100%、
    collision 0%、mean_raw_steps=600**（每集跑满 timeout），确证坍缩到 timeout 盆地。
    注：评估过程中误触发一次 `run_full --smoke`，把前 33 集 worker 数据与 `train_monitor.csv`
    覆盖；final_model.zip 已从 `ckpt_raw_50000_steps.zip` 恢复（sha256 6b007d14…），已重跑补齐 100 集
- **状态**：**mst_slt 成功 59%（best 70%）达突破**；hold35k 失败（坍缩 timeout），
  已加 hsac_mlp/sac_mlp 做 2×2 消融分离「动作头 × 编码器」（见阶段 6）

## 阶段 6：MLP 消融（当前）—— 分离「动作头 × 编码器」

- **日期**：2026-09-27
- **动机**：hold35k 失败后需干净归因「动作头（混合 vs 连续）」和「编码器（Scene-Rep vs MLP）」
  两个维度。原 hold35k vs mst_slt 有四因素混淆（动作头/lr/模型/neighbor 选车）。
- **新增两个方法**（`train_intersection_yield_v2.py`，METHODS 扩为 4 个）：
  - `hsac_mlp`：混合头（`DecisionAlignedSACPolicy`）+ `SimpleMlpLstmExtractor(backbone="mlp")`
    + 关 `representation_coef`，v4_8 契约（改法 1+2+3），lr 1e-4
  - `sac_mlp`：连续头（`SceneRepSACPolicy`）+ 同 MLP 编码器 + 关 `representation_coef`，
    base 契约（改法 1+2），lr 1e-4
  - 都包 `GeneralizedRewardShapingWrapper`（reward v2），与既有方法同环境/同奖励
- **消融矩阵**：
  - 编码器维度：mst_slt vs sac_mlp（同连续头）/ hold35k vs hsac_mlp（同混合头）
  - 动作头维度：hsac_mlp vs sac_mlp（同 MLP）
- **修复**：hold35k 的 lr 校验 bug（`verify_optimizer_settings` 用 `learning_rate(config, raw_budget)`）；
  `plot_training_curves` 空 CSV 优雅跳过（smoke 混合头 warmup 300 raw 步内无 episode 结束，
  原会抛 `RuntimeError: train_monitor.csv is empty`，现返回 None 跳过绘图）
- **结果（hsac_mlp / sac_mlp 正式训练完成，2026-09-27）**：
  | 方法 | 动作头 | 编码器 | 环境契约 | success | collision | timeout | off_route |
  |---|---|---|---|---|---|---|---|
  | hold35k | 混合 | Scene-Rep | v4_8 | 0% | 0% | **100%** | 0% |
  | mst_slt | 连续 | Scene-Rep | base | **59%** | 41% | 0% | 0% |
  | hsac_mlp | 混合 | MLP | v4_8 | 32% | 50% | 18% | 0% |
  | sac_mlp | 连续 | MLP | base | 41% | 59% | 0% | 0% |
  | sac_mlp_v48 | 连续 | MLP | v4_8 | 36% | 64% | 0% | 0% |
  | hsac_mlp_base | 混合 | MLP | base | 32% | 68% | 0% | 0% |
  - **归因反转（关键）**：hsac_mlp / hsac_mlp_base（混合头 + MLP）success 均 32%、未坍缩到
    timeout 盆地，**反驳了阶段 5「混合头在单车道让行不适配」的强结论**。hold35k 的 100%
    timeout 坍缩根因更可能是调参（lr 5e-5 + 深地板衰减 2e-5 + tau 0.0025，探索提前耗尽）或
    Scene-Rep 表示与混合头的交互，而非「混合头本身在让行任务不适配」。
  - **动作头维度（两对，干净，2026-09-27）**：
    - base 契约：sac_mlp（连续）41% vs hsac_mlp_base（混合）32% → 连续头 +9pp
    - v4_8 契约：sac_mlp_v48（连续）36% vs hsac_mlp（混合）32% → 连续头 +4pp
    - **一致结论：连续头略胜混合头**（+4~9pp），差距不大、量级相同。
    - **timeout 分量**：连续头全 0% timeout；混合头里 hold35k 100%（调参坍缩极端）、
      hsac_mlp 18%（单 seed 波动，因同混合头的 hsac_mlp_base 是 0%）、hsac_mlp_base 0%。
      故「混合头滑向 timeout」不是普遍规律，是 hold35k 调参/Scene-Rep×混合头的特定失败。
- **追加 sac_mlp_v48（用户拍板，2026-09-27）**：为消除「动作头维度」消融里的环境契约
  混杂（hsac_mlp 用 v4_8 而 sac_mlp 用 base），新增 `sac_mlp_v48` = 连续头
  `SceneRepSACPolicy` + `SimpleMlpLstmExtractor(mlp)` + **v4_8 契约** + lr 1e-4 +
  SEED_START=420000（对齐 hsac_mlp）。这样干净的「动作头维度」= hsac_mlp vs sac_mlp_v48
  （都 v4_8 + MLP + lr 1e-4 + 同 seed 序列）；编码器维度（hold35k vs hsac_mlp）接受 lr/tau
  混杂，仅作参考。
  - 技术确认：`SimpleMlpLstmExtractor` 只读 observation 的 `trajectory`/`map` 两个 key、
    忽略 v4_8 多暴露的 `lane_action_mask`；action_space 全契约通用 `Box(2)`，故连续头在
    v4_8 环境下可正常构建（sac_mlp_v48 smoke 全绿已证）。
- **追加 hsac_mlp_base（用户拍板，2026-09-27）**：为让 base 契约下也有干净的「动作头维度」
  消融（sac_mlp 用 base 而 hsac_mlp 用 v4_8，不能直接比），新增 `hsac_mlp_base` = 混合头
  `DecisionAlignedSACPolicy`（`make_hsac_model`）+ `SimpleMlpLstmExtractor(mlp)` + **base 契约
  语义**（改法 1+2 欧氏选车）+ lr 1e-4 + SEED_START=10000（对齐 sac_mlp）。
  - 技术实现：混合头硬性要求 `observation['lane_action_mask']`（`hybrid_policy_v4.py`
    `_mask_from_observation` 抛 KeyError），而 base 契约环境 `YieldObsIndependentV2EnvV1` 无此
    字段。故复用现成类 `YieldObsIndependentV2EnvV4V1`（改法 1+2 + 欧氏选车 + v4 观察契约含
    `lane_action_mask`），并让 `make_env_factory` 新增 `"v4_base"` 分支：走 baseline 分支拿
    evaluation split（与 sac_mlp 同 traffic 变体），但环境类用 v4 观察契约。smoke 全绿已证
    混合头在该环境下正常构建/评估。
- **干净的动作头维度消融（两对，对称）**：
  - base 契约：sac_mlp（连续）vs hsac_mlp_base（混合），都改法 1+2 欧氏 + lr 1e-4 + seed 10000
  - v4_8 契约：sac_mlp_v48（连续）vs hsac_mlp（混合），都改法 1+2+3 TTC + lr 1e-4 + seed 420000
- **状态**：sac_mlp_v48 与 hsac_mlp_base 正式训练均完成，两对「动作头维度」结论已出
  （连续头略胜混合头 +4~9pp）；6 方法消融矩阵完整

## 下一步

1. 读出完整 2×2 消融结论（6 方法已全出）：
   - 动作头维度（两对）：连续头略胜混合头（base +9pp，v4_8 +4pp）
   - 编码器维度（连续头）：mst_slt 59% vs sac_mlp 41%（Scene-Rep 胜 MLP ~18pp）
3. 若仍受密度瓶颈：叠加 P4 课程学习（低密度起步逐步加密）
4. 泛化验证：把 yield_v2 的 obs/reward 迁移到其它场景（cross/carla/cross_left/merge）
