# -*- coding: utf-8 -*-
"""Read-only extraction of results_topo_v4_3_dev into a structured JSON summary."""
import json, os

B = "results_topo_v4_3_dev"
PHASE = "topo v4.3 开发线"
COMMON = {"batch": "results_topo_v4_3_dev", "phase": PHASE, "variant": "v4_3", "candidate": None}

FIELDS = ["batch", "phase", "variant", "candidate", "scenario", "training_seed",
          "model_kind", "checkpoint_step", "episodes", "eval_seed_start", "decoder",
          "success_rate", "collision_rate", "timeout_rate", "mean_return",
          "status", "evidence_kind", "artifact", "source_keys", "notes"]


def row(**kw):
    r = dict(COMMON)
    r.update(kw)
    return {k: r.get(k) for k in FIELDS}


def pe_keys(seeds_text):
    return {
        "success_rate": "summary.success_rate",
        "collision_rate": "summary.collision_rate",
        "timeout_rate": "summary.timeout_rate",
        "mean_return": "summary.mean_return",
        "episodes": "evaluation_provenance.episodes（= summary.episodes = len(episode_records)）",
        "eval_seed_start": "evaluation_seed_start（episode_records 的 seed 为 %s）" % seeds_text,
        "decoder": "arguments.json:effective_method_hyperparameters.deterministic_lane_decoder = argmax_feasible_min_target_twin_q",
        "training_seed": "arguments.json:requested_raw_steps.seed",
        "checkpoint_step": "checkpoint_audit.json:checkpoints[0].raw_step = 20000（= requested_raw_steps）",
    }


DUP_NOTE = ("重复来源警告：同一组数值在 final_evaluation.json（顶层同名字段）、"
            "paper_evaluation_detailed.json:summary、action_diagnostics.json:outcomes、"
            "development/per_run.csv（同名 CSV 列）、development/summary.json:per_run[i]、"
            "development/development_decision.json:jobs.<id>.row 六处逐字段完全相等（已用脚本比对确认）；"
            "本行只取 paper_evaluation_detailed.json（含 evaluation_provenance 与 12 条 episode_records 的主产物）为唯一权威来源，未另立行。")

rows = []

rows.append(row(
    scenario="carla", training_seed=0,
    model_kind="other:外部批次 results_topo_v4_2_r1_dev/C1 final_model 冻结复现",
    checkpoint_step=20000, episodes=12, eval_seed_start=46000, decoder="target_critic",
    success_rate=1.0, collision_rate=0.0, timeout_rate=0.0, mean_return=1.0,
    status="completed", evidence_kind="development",
    artifact=B + "/development/replays/R1__cand__carla__s0__pda6b283b/paper_evaluation_detailed.json",
    source_keys={
        "success_rate": "summary.success_rate",
        "collision_rate": "summary.collision_rate",
        "timeout_rate": "summary.timeout_rate",
        "mean_return": "summary.mean_return",
        "episodes": "evaluation_provenance.episodes（= summary.episodes = len(episode_records) = 12）",
        "eval_seed_start": "evaluation_seed_start（episode_records 的 seed 为 46000–46011）",
        "training_seed": "replay_arguments.json:source_training_seed = 0",
        "checkpoint_step": "source_checkpoint_receipt.json 指向的 results_topo_v4_2_r1_dev/development/runs/C1__cand__carla__s0__p331cfbcd/arguments.json:requested_raw_steps.max_steps = 20000",
        "decoder": "replay_arguments.json + method_metadata.json:deterministic_lane_decoder = argmax_feasible_min_target_twin_q",
        "model_kind": "source_checkpoint_receipt.json:source_model = .../C1__cand__carla__s0__p331cfbcd/final_model.zip，sha256 0fbb5acf…c44a（已用磁盘文件重算 sha256 校验一致）",
    },
    notes=("开发阶段第 1 个预注册 cell（contract ordered_cells[0]，kind=frozen_checkpoint_replay，"
           "counted_as_fresh_traffic_confirmation=true、counted_as_fresh_training_confirmation=false）。"
           "被评估对象不是本批次训练出来的模型：是 results_topo_v4_2_r1_dev 批次 C1 运行（v4.2 实现，训练种子 0）的 final_model，"
           "权重冻结，只用 v4.3 的确定性 target-critic 车道解码器在 validation 分区**新流量**上闭环复现；"
           "源运行自己的评估种子是 44000，本 cell 从 46000 起，所以算新流量确认而非新训练确认。"
           "gate passed=true（carla 门禁 success≥0.50、collision≤0.10、off_route=0、timeout≤0.50）。"
           "辅助量（未单列字段）：mean_decision_steps 63.75、mean_raw_steps 191.0、lane_change_applied_rate 0.0667、"
           "exact_target_critic_argmax_rate 1.0、mean_speed 4.0668 m/s、evaluation_wall_seconds 87.73。" + DUP_NOTE),
))

rows.append(row(
    scenario="carla", training_seed=1,
    model_kind="other:外部批次 results_topo_v4_2_dev/F1 final_model 冻结复现",
    checkpoint_step=20000, episodes=12, eval_seed_start=46000, decoder="target_critic",
    success_rate=1.0, collision_rate=0.0, timeout_rate=0.0, mean_return=1.0,
    status="completed", evidence_kind="development",
    artifact=B + "/development/replays/R2__cand__carla__s1__pda6b283b/paper_evaluation_detailed.json",
    source_keys={
        "success_rate": "summary.success_rate",
        "collision_rate": "summary.collision_rate",
        "timeout_rate": "summary.timeout_rate",
        "mean_return": "summary.mean_return",
        "episodes": "evaluation_provenance.episodes（= summary.episodes = len(episode_records) = 12）",
        "eval_seed_start": "evaluation_seed_start（episode_records 的 seed 为 46000–46011）",
        "training_seed": "replay_arguments.json:source_training_seed = 1",
        "checkpoint_step": "source_checkpoint_receipt.json 指向的 results_topo_v4_2_dev/development/runs/F1__cand__carla__s1__p91f2cf40/arguments.json:requested_raw_steps.max_steps = 20000",
        "decoder": "replay_arguments.json + method_metadata.json:deterministic_lane_decoder = argmax_feasible_min_target_twin_q",
        "model_kind": "source_checkpoint_receipt.json:source_model = .../F1__cand__carla__s1__p91f2cf40/final_model.zip，sha256 8ccf0e2f…fb7a（已用磁盘文件重算 sha256 校验一致）",
    },
    notes=("开发阶段第 2 个预注册 cell（ordered_cells[1]，frozen_checkpoint_replay，requires=R1_pass）。"
           "源模型是 results_topo_v4_2_dev 批次 F1 运行（v4.2 实现，训练种子 1）的 final_model（源评估种子 43000），"
           "同样是「冻结权重 + 换 v4.3 解码器 + validation 新流量」。gate passed=true。"
           "注意 R1 与 R2 使用**相同**的评估种子区间 46000–46011，两者差异只来自被复现的模型（训练种子 0 vs 1），"
           "因此两行不可相加、也不能当作两个独立重复。辅助量：mean_decision_steps 56.5、mean_raw_steps 168.5、"
           "lane_change_applied_rate 0.0398、mean_speed 4.6028 m/s、evaluation_wall_seconds 76.20。" + DUP_NOTE),
))

rows.append(row(
    scenario="carla", training_seed=2, model_kind="exact_final",
    checkpoint_step=20000, episodes=12, eval_seed_start=47000, decoder="target_critic",
    success_rate=1.0, collision_rate=0.0, timeout_rate=0.0, mean_return=1.0,
    status="completed", evidence_kind="development",
    artifact=B + "/development/runs/T1__cand__carla__s2__pda6b283b/paper_evaluation_detailed.json",
    source_keys=pe_keys("47000–47011"),
    notes=("开发阶段第 3 个 cell（ordered_cells[2]，fresh_training，role=fresh_training_seed_confirmation，requires=R2_pass）："
           "本批次自己从头训练 20000 raw steps（训练种子 2，carla），评估的是**恰好等于请求步数**的 final_model.zip"
           "（checkpoint_audit：raw_step=recorded_clock=20000、learner_updates 15001、sha256 360f84fe…2bfa，已用磁盘文件重算校验一致）。"
           "gate passed=true，是本批次唯一的 carla 新训练确认。"
           "训练阶段指标（与评估阶段严格分开，故不写入本行 metric 列）：training_wall_seconds 5684.64、learner_updates 15001；"
           "best_training_success 规则（最近 20 回合成功率、固定分母 20）在 clock_value=12767 处选中 best 检查点"
           "（success_rate_last_20=1.0，共完成 85 个训练回合，sha256 8abbf3ea…360b），该 best 检查点另行在 attribution 中评估"
           "（见本表 model_kind=checkpoint、carla/种子 2 的行）。"
           "辅助量：mean_decision_steps 52.83、mean_raw_steps 157.42、mean_speed 4.7469 m/s、mean_inference_ms 46.93、"
           "peak_gpu_memory_mb 333.51、model+representation 参数量 2214229。" + DUP_NOTE),
))

rows.append(row(
    scenario="cross", training_seed=0, model_kind="exact_final",
    checkpoint_step=20000, episodes=12, eval_seed_start=48000, decoder="target_critic",
    success_rate=0.6666666666666666, collision_rate=0.3333333333333333, timeout_rate=0.0,
    mean_return=0.3333333333333333,
    status="completed", evidence_kind="development",
    artifact=B + "/development/runs/T2__cand__cross__s0__pda6b283b/paper_evaluation_detailed.json",
    source_keys=pe_keys("48000–48011"),
    notes=("开发阶段第 4 个 cell（ordered_cells[3]，fresh_training，role=regression_guard，requires=T1_pass）：cross 训练种子 0，"
           "评估 final_model.zip（20000 raw steps，sha256 63fdd8f7…8d05，已重算校验一致）。gate passed=true"
           "（cross 门禁 success≥0.50、collision≤0.45、off_route=0、timeout≤0.30；本行 4/12 碰撞 = 0.3333 仍在容差内）。"
           "12 个回合各用不同的交通变体文件（traffic_41/276/299/320/346/372/460/521/574/601/621/656.rou.xml），"
           "而 carla 三个 cell 全程只用 traffic_0.rou.xml —— 跨场景比较时这是重要的流量口径差异。"
           "训练阶段：best_training_success 在 clock_value=19106 选中 best 检查点（success_rate_last_20=0.75，完成 40 回合，sha256 0c291eb9…cd75），"
           "该 best 另行在 attribution 的 train 校准中评估（见本表 model_kind=checkpoint、cross/种子 0 的行）；"
           "T2 的 best 没有做 validation 闭环比对（见 missing 列表）。"
           "辅助量：mean_decision_steps 180.83、mean_raw_steps 541.5、mean_speed 4.5688 m/s。" + DUP_NOTE),
))

rows.append(row(
    scenario="cross", training_seed=1, model_kind="exact_final",
    checkpoint_step=20000, episodes=12, eval_seed_start=49000, decoder="target_critic",
    success_rate=0.0, collision_rate=0.16666666666666666, timeout_rate=0.8333333333333334,
    mean_return=-0.16666666666666666,
    status="completed", evidence_kind="development",
    artifact=B + "/development/runs/T3__cand__cross__s1__pda6b283b/paper_evaluation_detailed.json",
    source_keys=pe_keys("49000–49011"),
    notes=("开发阶段第 5 个也是最后一个 cell（ordered_cells[4]，fresh_training，role=stability_guard，requires=T2_pass）：cross 训练种子 1，"
           "评估 final_model.zip（20000 raw steps，sha256 60b7b568…9718，已重算校验一致）。"
           "gate passed=**false**，失败项是 success_rate（0.0 < 0.50）与 timeout_rate（0.8333 > 0.30）；"
           "collision 0.1667 ≤ 0.45、off_route 0.0 通过 → 开发门禁整体 decision=fail，按契约 on_fail=stop_and_create_new_versioned_attribution 终止，"
           "promotion 与 formal_test 均未解锁。**本 cell 自身是完整跑完 12/12 回合的「已完成但失败」，不是中止、也不是部分完成**；"
           "中止的是同一 job 的 infra attempt 1（见本表 status=aborted 行），且契约禁止把该 attempt 的部分检查点用于本行"
           "（partial_artifacts_used_for_acceptance=false），本行全部取自重跑成功后的最终产物。"
           "训练阶段：best_training_success 在 clock_value=15715 选中 best 检查点（success_rate_last_20=0.9，完成 32 回合，sha256 7d15cab9…f75f）；"
           "final 与 best 两个检查点都另行在 attribution 中评估（见本表两条 model_kind=checkpoint、cross/种子 1 的行）。"
           "辅助量：mean_decision_steps 172.5、mean_raw_steps 517.17、lane_change_applied_rate 0.0184、mean_speed 4.1912 m/s、"
           "exact_target_critic_argmax_rate 1.0。" + DUP_NOTE),
))

rows.append(row(
    scenario="carla", training_seed=2, model_kind="checkpoint",
    checkpoint_step=12767, episodes=12, eval_seed_start=47000, decoder="target_critic",
    success_rate=0.0, collision_rate=0.0, timeout_rate=1.0, mean_return=0.0,
    status="completed", evidence_kind="development",
    artifact=B + "/attribution/t3_failure/closed_loop/t1_best_target/attribution_result.json",
    source_keys={
        "success_rate": "outcomes.success_rate",
        "collision_rate": "outcomes.collision_rate",
        "timeout_rate": "outcomes.timeout_rate",
        "mean_return": "outcomes.mean_return",
        "episodes": "outcomes.episodes（= attribution_metadata.evaluation_episodes = 12）",
        "eval_seed_start": "attribution_metadata.evaluation_seed_start = 47000",
        "training_seed": "attribution_metadata.training_seed = 2",
        "checkpoint_step": "development/runs/T1__cand__carla__s2__pda6b283b/best_training_success.json:clock_value = 12767（clock_field=_raw_steps_seen）",
        "decoder": "attribution_metadata.decoder = target",
        "model_kind": "attribution_metadata.checkpoint = .../T1__cand__carla__s2__pda6b283b/best_training_success_model.zip，sha256 8abbf3ea…360b（已重算校验一致）",
    },
    notes=("post-hoc 归因诊断，不是预注册 cell：把 T1 的 best_training_success_model.zip（训练期最近 20 回合成功率规则在 raw step 12767 处选出的检查点）"
           "放到**与 T1 完全相同的 validation 种子 47000–47011** 上闭环复跑，用来区分「模型本身不行」和「final 检查点退化」。"
           "标签：post_hoc_diagnostic_only=true、counted_for_development_gate=false、formal_test_accessed=false。"
           "与 T1 行严格配对（同场景、同训练种子、同评估种子），可直接对照：final 12/12 成功 vs best 0/12 成功且 12/12 超时，"
           "这是 attribution_summary 判定「T1 选 final」的依据。"
           "同一检查点另有 **train 分区** 12 回合配对校准（种子 61000–61011，两半各 6 回合：t1_best_train6 = 0 成功/1.0 超时、t1_best_train6b 完全相同），"
           "结论一致（0 成功 12 超时）；因属于同一 (variant, scenario, training_seed, model_kind) 单元，本表不另立行，数值记录于此。"
           "重复来源：final_evaluation.json 与 attribution_result.json:outcomes 逐字段相等，本行取后者（带 schema 与来源元数据）。"),
))

rows.append(row(
    scenario="cross", training_seed=0, model_kind="checkpoint",
    checkpoint_step=19106, episodes=12, eval_seed_start=62000, decoder="target_critic",
    success_rate=0.75, collision_rate=0.25, timeout_rate=0.0, mean_return=0.5,
    status="completed", evidence_kind="development",
    artifact=B + "/attribution/t3_failure/attribution_summary.json",
    source_keys={
        "success_rate": "historical_train_only_checkpoint_calibration.T2.candidate_outcomes.best.success_rate",
        "collision_rate": "historical_train_only_checkpoint_calibration.T2.candidate_outcomes.best.collision_rate",
        "timeout_rate": "historical_train_only_checkpoint_calibration.T2.candidate_outcomes.best.timeout_rate",
        "mean_return": "historical_train_only_checkpoint_calibration.T2.candidate_outcomes.best.mean_return",
        "episodes": "historical_train_only_checkpoint_calibration.T2.calibration_episodes_per_checkpoint = 12（= 6 + 6 两半）",
        "eval_seed_start": "calibration/t2_best_train6/attribution_metadata.json:evaluation_seed_start = 62000，后半 t2_best_train6b = 62006（配对种子 62000–62011）",
        "training_seed": "calibration/t2_best_train6/attribution_metadata.json:training_seed = 0",
        "checkpoint_step": "development/runs/T2__cand__cross__s0__pda6b283b/best_training_success.json:clock_value = 19106",
        "decoder": "calibration/t2_best_train6/attribution_metadata.json:decoder = target",
        "model_kind": "calibration/t2_best_train6/attribution_metadata.json:checkpoint = .../T2__cand__cross__s0__pda6b283b/best_training_success_model.zip，sha256 0c291eb9…cd75（已重算校验一致）",
    },
    notes=("post-hoc 的 **train 分区**检查点校准（不是 validation，更不是 formal test）：traffic_partition=train，"
           "给 T2 的 best 与 final 各跑 12 个配对训练回合（种子 62000–62011 成对），标签 counted_for_development_gate=false、post_hoc_diagnostic_only=true。"
           "数值取 attribution_summary.json 的 12 回合汇总口径：best 9 成功/3 碰撞 → 0.75/0.25，与 final 的 0.75/0.25 完全相同 → 规则判为 complete tie 后选 final。"
           "**切勿与 T2 的 validation 结果混用**：同 (variant, scenario, training_seed=0, model_kind=checkpoint) 的 validation 闭环比对在本批次**没有做**，"
           "因此本行只能给 train 校准值，不是 T2 best 在 validation 上的成绩。"
           "重复来源：raw 两半产物（calibration/t2_best_train6/ 与 t2_best_train6b/，各 6 回合，success 0.6667 / 0.8333，collision 0.3333 / 0.1667）"
           "是同 12 回合单元的拆分，6 回合口径不单独成行，只保留 attribution_summary.json 的 12 回合汇总为权威。"),
))

rows.append(row(
    scenario="cross", training_seed=1, model_kind="checkpoint",
    checkpoint_step=15715, episodes=12, eval_seed_start=49000, decoder="target_critic",
    success_rate=0.9166666666666666, collision_rate=0.08333333333333333, timeout_rate=0.0,
    mean_return=0.8333333333333334,
    status="completed", evidence_kind="development",
    artifact=B + "/attribution/t3_failure/closed_loop/best_target/attribution_result.json",
    source_keys={
        "success_rate": "outcomes.success_rate",
        "collision_rate": "outcomes.collision_rate",
        "timeout_rate": "outcomes.timeout_rate",
        "mean_return": "outcomes.mean_return",
        "episodes": "outcomes.episodes（= attribution_metadata.evaluation_episodes = 12）",
        "eval_seed_start": "attribution_metadata.evaluation_seed_start = 49000",
        "training_seed": "attribution_metadata.training_seed = 1",
        "checkpoint_step": "development/runs/T3__cand__cross__s1__pda6b283b/best_training_success.json:clock_value = 15715",
        "decoder": "attribution_metadata.decoder = target",
        "model_kind": "attribution_metadata.checkpoint = .../T3__cand__cross__s1__pda6b283b/best_training_success_model.zip，sha256 7d15cab9…f75f（已重算校验一致）",
    },
    notes=("T3 失败归因的核心证据行：把 T3 的 best_training_success_model.zip 放在与 T3 失败时**完全相同**的 validation 种子 49000–49011 上闭环比对，"
           "得 11/12 成功、1/12 碰撞、0 超时（对比 T3 final 的 0 成功、2/12 碰撞、10/12 超时，平均速度 4.911 vs 4.191 m/s），"
           "attribution_summary 据此把主因判为 late_training_checkpoint_drift_in_longitudinal_behavior，"
           "并否决「变道根本没完成」假设（11/12 回合观察到目标车道、10/12 到达下一条路线边、10 次超时全部已在终点边 gneE23）。"
           "标签：post_hoc_diagnostic_only=true、counted_for_development_gate=false、formal_test_accessed=false —— 这是事后诊断，"
           "既不能当 T3 的正式成绩，也不能当 v4.4 的确认性证据（契约要求 v4.4 重做预注册开发确认）。"
           "同一检查点另有 train 分区 12 回合配对校准（种子 63000–63011，两半 6 回合：t3_best_train6 = 1.0 成功、"
           "t3_best_train6b = 0.6667 成功/0.3333 碰撞 → 合计 10/12 = 0.8333 成功、2/12 = 0.1667 碰撞），"
           "因同属一个 (variant, scenario, seed, model_kind) 单元，不另立行，数值记录于此（train 0.8333 与 validation 0.9167 的差异属不同分区口径）。"
           "重复来源：final_evaluation.json 与 attribution_result.json:outcomes 相等，取后者。"),
))

rows.append(row(
    scenario="carla", training_seed=0, model_kind="exact_final",
    checkpoint_step=60, episodes=2, eval_seed_start=35000, decoder="target_critic",
    success_rate=0.0, collision_rate=0.0, timeout_rate=1.0, mean_return=0.0,
    status="engineering_only", evidence_kind="smoke",
    artifact=B + "/engineering/smoke/candidate_carla_e2e/paper_evaluation_detailed.json",
    source_keys={
        "success_rate": "summary.success_rate",
        "collision_rate": "summary.collision_rate",
        "timeout_rate": "summary.timeout_rate",
        "mean_return": "summary.mean_return",
        "episodes": "summary.episodes（= len(episode_records) = 2）",
        "eval_seed_start": "evaluation_seed_start = 35000（episode_records 的 seed 为 35000、35001）",
        "training_seed": "arguments.json:requested_raw_steps.seed = 0",
        "checkpoint_step": "checkpoint_audit.json:checkpoints[0].raw_step = 60（recorded_clock = 60）",
        "decoder": "arguments.json:effective_method_hyperparameters.deterministic_lane_decoder",
        "model_kind": "paper_evaluation_detailed.json:model = .../smoke/candidate_carla_e2e/final_model.zip，sha256 91a95680…cb81（已重算校验一致）",
    },
    notes=("端到端训练冒烟（engineering_receipt.checks.end_to_end_training_smoke=passed；arguments.json 的 implementation_freeze_sha256 为 null，"
           "说明是 freeze 之前的工程产物）：只训练 60 raw steps（learner_updates 31）、评估 2 个回合，用途是验证流程跑得通，**没有科学意义**。"
           "0.0 成功率 / 1.0 超时率只反映模型欠训练，不可与正式 cell 的成功率一起平均或比较；本行 status=engineering_only。"
           "另外 action_diagnostics 显示这两回合 route_action_window_samples=0（无路线动作窗口样本），所以 route_action_window_match_rate 为 null。"),
))

rows.append(row(
    scenario="carla", training_seed=0,
    model_kind="other:外部批次 v4.2_r1 final_model 冻结复现（pre-freeze 单回合工程冒烟）",
    checkpoint_step=20000, episodes=1, eval_seed_start=35100, decoder="target_critic",
    success_rate=1.0, collision_rate=0.0, timeout_rate=0.0, mean_return=1.0,
    status="engineering_only", evidence_kind="engineering",
    artifact=B + "/engineering/replay_smoke/seed0_35100/paper_evaluation_detailed.json",
    source_keys={
        "success_rate": "summary.success_rate",
        "collision_rate": "summary.collision_rate",
        "timeout_rate": "summary.timeout_rate",
        "mean_return": "summary.mean_return",
        "episodes": "summary.episodes（= len(episode_records) = 1）",
        "eval_seed_start": "evaluation_seed_start = 35100（唯一的 episode_records[0].seed = 35100）",
        "training_seed": "source_training_seed = 0",
        "checkpoint_step": "source_checkpoint_receipt.json 指向的 results_topo_v4_2_r1_dev/.../C1__cand__carla__s0__p331cfbcd/arguments.json:requested_raw_steps.max_steps = 20000",
        "decoder": "method_metadata.json:deterministic_lane_decoder",
        "model_kind": "source_checkpoint_receipt.json:source_model = .../C1__cand__carla__s0__p331cfbcd/final_model.zip，sha256 0fbb5acf…c44a",
    },
    notes=("检查点复现链路的工程冒烟（engineering_receipt.checks.checkpoint_replay_smoke=passed，implementation_freeze_sha256='prefreeze_engineering_only'）："
           "只跑 1 个回合、种子 35100，用于验证 replay_v4_3_checkpoint.py 流程，不构成任何确认性证据；本行 status=engineering_only。"
           "**与 R1 行同源**：复现的是同一个模型文件（results_topo_v4_2_r1_dev/C1 final_model，sha256 0fbb5acf…c44a，与 R1 一致），"
           "只是回合数（1 vs 12）、评估种子区间（35100 vs 46000–46011）和证据等级不同；为满足「一个 (variant, scenario, seed, model_kind) 只一行」，"
           "本行用带冒烟说明的 model_kind 与 R1 区分开。分析时请把它当作 R1 的工程前身，切勿与 R1 一起统计。"),
))

rows.append(row(
    scenario="cross", training_seed=1,
    model_kind="other:部分训练检查点（被判定不可用，未做任何评估）",
    checkpoint_step=None, episodes=None, eval_seed_start=None, decoder="target_critic",
    success_rate=None, collision_rate=None, timeout_rate=None, mean_return=None,
    status="aborted", evidence_kind="development",
    artifact=B + "/development/failed_attempts/T3__cand__cross__s1__pda6b283b__infra_attempt1/infrastructure_failure_receipt.json",
    source_keys={
        "success_rate": "（无源：该 attempt 未做任何评估，产物里不存在成功/碰撞/超时/回报数值）",
        "episodes": "（无源）",
        "eval_seed_start": "（无源）",
        "checkpoint_step": "（无源：运行在 15715/20000 raw steps 处被 TraCI 连接重置中断，落盘的 best_training_success_model.zip 不是契约要求的 exact-final 检查点，故留 null）",
        "training_seed": "infrastructure_failure_receipt.json:training_seed = 1",
        "decoder": "run/method_metadata.json:deterministic_lane_decoder",
        "model_kind": "infrastructure_failure_receipt.json:partial_best_training_model_sha256 = f229d623…a29b（partial_artifacts_used_for_acceptance = false）",
    },
    notes=("开发阶段 T3 的**中止**尝试（同一 job 的 attempt 1），必须与已完成但失败的 T3 正式 cell 区分开。"
           "classification=infrastructure_interruption、scientific_gate_evaluated=false、cell_status=TBD；"
           "reason：SUMO/TraCI 连接在预注册的 20000 raw steps 跑完前被重置（ConnectionResetError WinError 10054），"
           "最后记录到的 raw simulation steps = 15715（该数字只是 train_monitor.csv 里的进度行，不是检查点步数，故 checkpoint_step 留 null）。"
           "restart_policy 要求归档该 attempt 并**从初始化重跑**，禁止加载或评估其部分检查点（partial_artifacts_used_for_acceptance=false）；"
           "重跑后得到了完整的 T3 结果（本表 cross/种子 1/exact_final 行）。本行所有指标字段一律 null，不做任何推算或插补。"),
))

document = {
    "directory": "results_topo_v4_3_dev",
    "phase": PHASE,
    "purpose": ("v4.3「确定性 target-critic 车道解码器」开发线批次：相对父实现 v4.2 只改一处 —— 部署时车道动作不再由 actor 采样，"
                "而是对可行车道枚举取 target twin-Q 最小的 argmax（argmax_feasible_min_target_twin_q）；训练策略、奖励、编码器、检查点选择规则全部不变。"
                "按 experiments/topo_scene_v4/experiment_contract_v4_3.yaml 顺序执行 5 个预注册 cell：R1/R2 把 v4.2 冻结检查点换上 v4.3 解码器在新 validation 流量上复现，"
                "T1（carla）做新训练确认，T2/T3（cross）做回归与稳定守卫，全部在 validation 分区各评 12 回合。"
                "T3 门禁失败（成功率 0.0、超时率 0.833）→ 开发结论 decision=fail，按契约 on_fail 停止并生成本批次的 T3 失败归因；"
                "promotion 与 formal_test 从未解锁、从未运行，本批次不含任何正式测试证据。"),
    "status": "completed",
    "protocol_files": [
        "experiments/topo_scene_v4/experiment_contract_v4_3.yaml",
        "experiments/topo_scene_v4/V4_3_TARGET_CRITIC_DECODER.md",
        "results_topo_v4_3_dev/development/stage_plan.json",
        "results_topo_v4_3_dev/development/summary.json",
        "results_topo_v4_3_dev/development/per_run.csv",
        "results_topo_v4_3_dev/development/development_decision.json",
        "results_topo_v4_3_dev/development/last_execution.json",
        "results_topo_v4_3_dev/engineering/implementation_freeze.json",
        "results_topo_v4_3_dev/engineering/engineering_receipt.json",
        "results_topo_v4_3_dev/attribution/t3_failure/attribution_summary.json",
    ],
    "rows": rows,
    "missing": [
        {"unit": "promotion 阶段全部 12 个作业（temporal_graph 与 selected_v4_candidate；场景 cross/roundabout_medium/carla；种子 0/1；50000 raw steps；30 回合）",
         "reason": "开发门禁 decision=fail（T3 成功率 0.0 < 0.50、超时率 0.833 > 0.30），契约 on_fail=stop_and_create_new_versioned_attribution 规定停止；promotion 从未启动，本批次目录下不存在任何 promotion 产物。"},
        {"unit": "formal_test 阶段全部 120 个作业（6 场景 × 2 方法 × 10 种子；100000 raw steps；50 回合；test 分区）",
         "reason": "契约 formal_test.locked=true，且所有产物里的 formal_test_accessed 一律为 false（engineering_receipt、attribution_summary、各 attribution_metadata 都是 false），正式测试证据为零。"},
        {"unit": "temporal_graph 对照方法在本批次内的任何评估数值",
         "reason": "本批次只跑了 engineering/preflights/temporal_graph_carla 的 check_only 预检（algorithm=temporal_graph_v1_control），check_result.json 只记录一个动作向量与动作合法性，不含任何 success/collision/timeout/return 指标，故不建行、也不填 0。"},
        {"unit": "T2 的 best_training_success 检查点在 validation 分区上的闭环比对（对应 T3 的 closed_loop/best_target 与 T1 的 closed_loop/t1_best_target）",
         "reason": "attribution/t3_failure/closed_loop/ 下只有 best_target（T3）与 t1_best_target（T1）两个目录，没有 T2 best 的 validation 复跑；T2 best 只有 train 分区校准值，因此本表该行口径是 train 而非 validation。"},
        {"unit": "engineering/preflights 的 candidate_carla、candidate_cross、temporal_graph_carla 三个检查结果",
         "reason": "三者都是 check_only 的动作合法性预检（status=ok、action_finite、exact_hybrid_lane_code），没有任何评估回合与结果指标，全部指标字段无处可取，故不建行。"},
        {"unit": "T3 中止尝试（T3__cand__cross__s1__pda6b283b__infra_attempt1）的评估数值",
         "reason": "运行在 15715/20000 raw steps 时被 TraCI 连接重置中断，scientific_gate_evaluated=false、cell_status=TBD，契约 restart_policy 禁止评估其部分检查点；该单元只有一条 status=aborted、指标全 null 的占位行。"},
        {"unit": "left_turn / roundabout / roundabout_easy / roundabout_medium 场景在本批次内的任何数值",
         "reason": "这几个场景只出现在契约的 promotion 与 formal_test 阶段；开发阶段 ordered_cells 只含 carla 与 cross，而这两个后续阶段都未运行。"},
    ],
    "notes": ("口径、坑与可信度要点（按重要性）：\n"
              "1) 阶段划分：本目录是纯 **development** 批次（另有 attribution 事后归因与 engineering 工程证据），所有 12 回合评估都在 **validation** 分区"
              "（traffic_partition=validation，traffic_protocol=frozen_60_20_20）。formal test（test 分区）**从未被访问** —— 每个产物的 "
              "formal_test_accessed / formal_test_locked 字段都这么写，因此本表不存在 formal_test 证据，也请不要把 validation 成绩当成正式测试成绩。\n"
              "2) 「carla」不是真的 CARLA 仿真器：evaluation_provenance 显示 environment_class=envs.sumo.paper_env_v4.PaperSumoSceneEnvV4、"
              "uses_released_assets=false、scenario_asset_source=carla_source_waypoint_reconstruction、scenario_evidence_class=controlled_extension_not_reported_in_paper —— "
              "它是用 CARLA 源航点重建出来的 SUMO 场景，论文原始场景里没有它，证据等级低于 cross 等原场景。\n"
              "3) 已完成 vs 中止：开发阶段 5 个 cell 全部跑完并被 accepted（complete=true、stopped_after=T3、failures=[]），"
              "T3 属于「完整跑完但门禁失败」，不是中止；真正中止的只有它的 infra attempt 1（已单列 status=aborted 行）。"
              "顶层 status=completed 指「本批次按契约把自己的开发阶段执行完整并给出 fail 结论」，不代表 promotion 通过或正式测试完成 —— 那两阶段一行结果都没有。\n"
              "4) 一个单元一行：本表严格按 (variant, scenario, training_seed, model_kind) 去重。同一数值通常有 6 份副本（final_evaluation.json、"
              "paper_evaluation_detailed.json:summary、action_diagnostics.json:outcomes、per_run.csv、summary.json:per_run、"
              "development_decision.json:jobs.<id>.row），已逐字段比对确认完全相等，只取主产物一份；train 分区校准（calibration/*_train6 与 *_train6b，各 6 回合两半）"
              "与 validation 闭环比对（closed_loop）评的是同一个检查点，已并入 model_kind=checkpoint 的行并在 notes 里记录另一分区的数值，"
              "未拆成多行，以免把 6 回合口径混进 12 回合口径。\n"
              "5) 数值可信度：论文评估记录 evaluation_provenance.validated=true；本次审计另用磁盘文件重算了 9 个模型 zip 的 sha256"
              "（T1/T2/T3 的 best 与 final、R1/R2 的外部源模型、engineering smoke 的 final_model），全部与元数据声明值一致，未发现伪造或错配。"
              "attribution 产物都带 fabricated_values=false、computed_from_real_closed_loop_rollout 类标记，但它们是 **post-hoc** 诊断"
              "（counted_for_development_gate=false、post_hoc_diagnostic_only=true），只能解释失败，不能当作确认性证据。\n"
              "6) 已知坑：R1 与 R2 用**同一段** validation 种子 46000–46011，两者差异只来自被复现的模型（v4.2 批次训练种子 0 vs 1）；"
              "carla 三个 cell 全程只用 traffic_0.rou.xml 一个交通变体，而 cross 两个 cell 各用 12 个不同变体文件，跨场景比较时流量口径不同；"
              "engineering 的 smoke（60 raw steps、2 回合）与 replay_smoke（1 回合）一个欠训练、一个与 R1 同源，凡带 engineering_only/smoke 标记的行都不可参与成功率平均或方法比较。\n"
              "7) 未发现在跑作业：last_execution.json 的时间戳（2026-08-18T14:49:33Z）与目录内所有 mtime 一致，批次已静止；"
              "development_decision.json 的 next_job=null 也确认没有待执行 cell。"),
}

out = os.path.abspath("reports/all_experiments_summary_20260919/raw/results_topo_v4_3_dev.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(document, fh, ensure_ascii=False, indent=2)

# self-check
keys = set()
for r in document["rows"]:
    t = (r["variant"], r["scenario"], r["training_seed"], r["model_kind"])
    assert t not in keys, "duplicate tuple: %r" % (t,)
    keys.add(t)
    assert set(r.keys()) == set(FIELDS), set(r.keys()) ^ set(FIELDS)
print(json.dumps({"file": out, "rows": len(document["rows"]),
                  "missing": len(document["missing"]),
                  "status": document["status"]}, ensure_ascii=False))
