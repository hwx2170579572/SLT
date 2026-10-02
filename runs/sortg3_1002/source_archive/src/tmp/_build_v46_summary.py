# -*- coding: utf-8 -*-
"""Read-only extractor: results_topo_v4_6_dev -> raw JSON. No source file is modified."""
import json, os
from collections import Counter

ROOT = r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo"
B    = os.path.join(ROOT, "results_topo_v4_6_dev")
REL  = "results_topo_v4_6_dev"

def L(*p):  return os.path.join(B, *p)
def J(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def f1(*p): return L("development", "runs", "F1__cand__cross__s6__p814d53b7", *p)

SUM   = J(L("development", "summary.json"))
DEC   = J(L("development", "development_decision.json"))
STAGE = J(L("development", "stage_plan.json"))
PRE   = J(L("preregistration_receipt.json"))
ENG   = J(L("engineering", "engineering_receipt.json"))
ATT   = J(L("attribution", "f1_failure", "attribution_summary.json"))
SELREC= J(f1("selector", "receipt.json"))
BEST  = J(f1("best_training_success.json"))
PAPER = J(f1("paper_evaluation_detailed.json"))
FEVAL = J(f1("final_evaluation.json"))

BATCH = "results_topo_v4_6_dev"
PHASE_V46 = "topo v4.6 开发线（联合 checkpoint×decoder 选择器）"
PHASE_V45 = "topo v4.5 线（作为 v4.6 批次内交叉干预对照）"

def rate_keys(prefix, extra=None):
    d = {
        "success_rate":    prefix + ".success_rate",
        "collision_rate":  prefix + ".collision_rate",
        "timeout_rate":    prefix + ".timeout_rate",
        "mean_return":     prefix + ".mean_return",
        "episodes":        prefix + ".episodes",
    }
    if extra: d.update(extra)
    return d

rows = []

# ---------------------------------------------------------------- F1 selected
rows.append({
    "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
    "candidate": "joint_checkpoint_decoder_selector",
    "scenario": "cross", "training_seed": 6,
    "model_kind": "selected", "checkpoint_step": 20000,
    "episodes": PAPER["summary"]["episodes"],
    "eval_seed_start": PAPER["evaluation_seed_start"],
    "decoder": "target_critic",
    "success_rate":   PAPER["summary"]["success_rate"],
    "collision_rate": PAPER["summary"]["collision_rate"],
    "timeout_rate":   PAPER["summary"]["timeout_rate"],
    "mean_return":    PAPER["summary"]["mean_return"],
    "status": "completed", "evidence_kind": "development",
    "artifact": REL + "/development/runs/F1__cand__cross__s6__p814d53b7/paper_evaluation_detailed.json",
    "source_keys": rate_keys("summary", {
        "eval_seed_start": "evaluation_seed_start (= episode_records[0].seed 63000)",
        "training_seed":   "development/runs/F1__cand__cross__s6__p814d53b7/arguments.json:requested_raw_steps.seed (=6)",
        "checkpoint_step": "development/runs/F1__cand__cross__s6__p814d53b7/checkpoint_audit.json:checkpoints[0].raw_step (=20000)",
        "model_kind":      "selector/receipt.json:selected_checkpoint_kind + selected_deployment_decoder",
        "decoder":         "selector/receipt.json:selected_deployment_decoder",
    }),
    "notes": ("v4.6 唯一被接受的开发单元 F1（cross / 训练种子 6）在 validation 分区（种子 63000-63011，12 回合）上的评估结果。"
              "该 selected_model.zip 是对 exact_final 检查点（raw 20000 步，sha256 16c18a01...）按 target_critic 解码器做的策略类重序列化"
              "（sha256 effba327...，策略参数状态 sha 9ec4bb15... 与源检查点一致）。"
              "同一组数值在 final_evaluation.json / development/summary.json / development/per_run.json / development_decision.json:jobs.F1.row / "
              "attribution_summary.json:validation_pair_matrix.exact_final__target_critic 中重复出现且一致，此处只保留 paper_evaluation_detailed.json 这一份最权威来源。"
              "结果：outcome gate 未通过（success 0.25 < 0.50；collision 0.50 > 0.45；timeout 0.25 <= 0.30 通过；off_route 0 通过），"
              "而 mechanism / selector / decoder-integrity 检查全部通过。训练阶段另记录：highest_training_success_rate_last_20 = 0.65（完成 32 回合），低于合约中 threshold_must_equal=0.90。"),
})

# ------------------------------------------------------- F1 calibration (train)
cal_meta = {
    ("highest_training_success", "target_critic"): ("selector/cal/best/target_critic", 14314),
    ("highest_training_success", "fusion_0_90"):   ("selector/cal/best/fusion_0_90",   14314),
    ("exact_final", "target_critic"):              ("selector/cal/final/target_critic", 20000),
    ("exact_final", "fusion_0_90"):                ("selector/cal/final/fusion_0_90",   20000),
}
for (ck, dcd), (rel, step) in cal_meta.items():
    det = J(f1(*rel.split("/"), "detailed.json"))
    s = det["summary"]
    rows.append({
        "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
        "candidate": "joint_checkpoint_decoder_selector",
        "scenario": "cross", "training_seed": 6,
        "model_kind": ck, "checkpoint_step": step,
        "episodes": s["episodes"],
        "eval_seed_start": SELREC["calibration_seed_start"],
        "decoder": dcd,
        "success_rate": s["success_rate"], "collision_rate": s["collision_rate"],
        "timeout_rate": s["timeout_rate"], "mean_return": s["mean_return"],
        "status": "completed", "evidence_kind": "development",
        "artifact": REL + "/development/runs/F1__cand__cross__s6__p814d53b7/" + rel + "/detailed.json",
        "source_keys": rate_keys("summary", {
            "eval_seed_start": "selector/receipt.json:calibration_seed_start (=72000，train 分区标定)",
            "checkpoint_step": ("best_training_success.json:clock_value (=14314，clock_field=_raw_steps_seen)"
                                if ck == "highest_training_success" else
                                "checkpoint_audit.json:checkpoints[0].raw_step (=20000)"),
            "model_kind": "selector/receipt.json:candidates[].checkpoint_kind",
            "decoder":    "selector/receipt.json:candidates[].deployment_decoder",
        }),
        "notes": ("F1 选择器标定数值：train 交通分区（标定种子 72000-72011，12 回合），属于训练阶段的选择环节输入，不是 outcome gate 依据。"
                  "这是 4 个 (checkpoint × decoder) 候选对被冻结后在同一 traffic block 上成对回放的结果；"
                  "数值同时出现在 selector/receipt.json:candidates[].summary 与 selector/cal/.../evaluation.json，三者一致，此处取 detailed.json（其 sha256 即 receipt 中的 calibration_result_sha256）。"
                  + ("该候选对在标定中胜出（8/12 成功，选择键 [8,-3,0,-1,0.416667,1,1]），被部署为 selected。"
                     if (ck, dcd) == ("exact_final", "target_critic") else
                     "该候选对在标定中未被选中。")),
    })

# -------------------------------------------------- post-hoc attribution replay
att_meta = {
    "best_target": ("closed_loop/best_target_validation",  "highest_training_success", "target_critic", 14314),
    "best_fusion": ("closed_loop/best_fusion_validation",  "highest_training_success", "fusion_0_90",   14314),
    "final_fusion":("closed_loop/final_fusion_validation", "exact_final", "fusion_0_90", 20000),
}
for key, (rel, ck, dcd, step) in att_meta.items():
    ar = J(L("attribution", "f1_failure", *rel.split("/"), "attribution_result.json"))
    o = ar["outcomes"]
    rows.append({
        "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
        "candidate": "joint_checkpoint_decoder_selector",
        "scenario": ar["scenario"], "training_seed": ar["training_seed"],
        "model_kind": ck, "checkpoint_step": step,
        "episodes": o["episodes"], "eval_seed_start": ar["evaluation_seed_start"],
        "decoder": dcd,
        "success_rate": o["success_rate"], "collision_rate": o["collision_rate"],
        "timeout_rate": o["timeout_rate"], "mean_return": o["mean_return"],
        "status": "completed", "evidence_kind": "development",
        "artifact": REL + "/attribution/f1_failure/" + rel + "/attribution_result.json",
        "source_keys": rate_keys("outcomes", {
            "eval_seed_start": "evaluation_seed_start (=63000)",
            "model_kind": "checkpoint_kind",
            "decoder": "decoder",
        }),
        "notes": ("F1 失败归因的事后（post-hoc）重放：把未被选中的候选对在同一 validation block（种子 63000）上重跑。"
                  "attribution_result.json 明确标注 post_hoc_diagnostic_only=true、counted_for_development_gate=false、counted_for_promotion_gate=false、formal_test_accessed=false、"
                  "mutated_checkpoint=false，因此不计入任何 gate，仅作诊断证据。"
                  "同 block 上 exact_final × target_critic 的数值与 F1 选中模型行完全一致（同一份 paper_evaluation_detailed.json 复算），故未重复成行。"),
    })

# ------------------------------------------------------ cross-block intervention
cb = ATT["crossed_checkpoint_block_intervention"]
sr, cr, tr = cb["success_rate_matrix"], cb["collision_rate_matrix"], cb["timeout_rate_matrix"]

rows.append({
    "batch": BATCH, "phase": PHASE_V45, "variant": "v4_5",
    "candidate": "parent_exact_final_target_critic",
    "scenario": "cross", "training_seed": 4, "model_kind": "exact_final",
    "checkpoint_step": 20000, "episodes": 12, "eval_seed_start": 63000,
    "decoder": "target_critic",
    "success_rate": sr["v45_seed4"]["block_63000"], "collision_rate": cr["v45_seed4"]["block_63000"],
    "timeout_rate": tr["v45_seed4"]["block_63000"], "mean_return": 0.8333333333333334,
    "status": "completed", "evidence_kind": "development",
    "artifact": REL + "/attribution/f1_failure/cross_block/v45_s4_final_target_on_v46_block/attribution_result.json",
    "source_keys": rate_keys("outcomes", {
        "eval_seed_start": "evaluation_seed_start (=63000)",
        "training_seed": "training_seed (=4)",
        "model_kind": "checkpoint_kind + checkpoint（results_topo_v4_5_dev 的 final_model.zip）",
    }),
    "notes": ("交叉干预对照：把 v4.5 cross seed 4 的 exact_final 权重放到 v4.6 的 validation block（63000）上评估，10/12 成功、0 碰撞。"
              "该检查点属于上一批次 results_topo_v4_5_dev（source_run_directory=E1__cand__cross__s4__p3c1f7f8b），本批次只记录其在本 block 上的重放结果；"
              "post_hoc_diagnostic_only=true，不计入 gate。"),
})

rows.append({
    "batch": BATCH, "phase": PHASE_V45, "variant": "v4_5",
    "candidate": "parent_exact_final_target_critic",
    "scenario": "cross", "training_seed": 4, "model_kind": "exact_final",
    "checkpoint_step": 20000, "episodes": 12, "eval_seed_start": 59000,
    "decoder": "target_critic",
    "success_rate": sr["v45_seed4"]["block_59000"], "collision_rate": cr["v45_seed4"]["block_59000"],
    "timeout_rate": tr["v45_seed4"]["block_59000"], "mean_return": None,
    "status": "completed", "evidence_kind": "development",
    "artifact": REL + "/attribution/f1_failure/attribution_summary.json",
    "source_keys": {
        "success_rate":   "crossed_checkpoint_block_intervention.success_rate_matrix.v45_seed4.block_59000",
        "collision_rate": "crossed_checkpoint_block_intervention.collision_rate_matrix.v45_seed4.block_59000",
        "timeout_rate":   "crossed_checkpoint_block_intervention.timeout_rate_matrix.v45_seed4.block_59000",
        "mean_return":    None,
        "episodes":       "crossed_checkpoint_block_intervention.paired_seed4_vs_seed6_on_block_59000.paired_episodes (=12)",
        "eval_seed_start": "crossed_checkpoint_block_intervention.paired_seed4_vs_seed6_on_block_59000.per_episode[0].seed (=59000)",
        "training_seed":  "crossed_checkpoint_block_intervention.sources.v45_seed4_block59000（v4.5 seed4 权重）",
    },
    "notes": ("交叉干预 2x2 表的另一格。数值记录在本批次 attribution_summary.json 内，但其底层源文件在批次之外"
              "（results_topo_v4_5_dev/attribution/e1_failure/closed_loop/final_target_validation/attribution_result.json，sha256 89bbc30a...），"
              "因此 mean_return 在本批次内无源可取，置 null 并计入 missing。"),
})

ar46 = J(L("attribution", "f1_failure", "cross_block", "v46_s6_final_target_on_v45_block", "attribution_result.json"))
o46 = ar46["outcomes"]
rows.append({
    "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
    "candidate": "joint_checkpoint_decoder_selector",
    "scenario": "cross", "training_seed": 6, "model_kind": "exact_final",
    "checkpoint_step": 20000, "episodes": o46["episodes"],
    "eval_seed_start": ar46["evaluation_seed_start"], "decoder": "target_critic",
    "success_rate": o46["success_rate"], "collision_rate": o46["collision_rate"],
    "timeout_rate": o46["timeout_rate"], "mean_return": o46["mean_return"],
    "status": "completed", "evidence_kind": "development",
    "artifact": REL + "/attribution/f1_failure/cross_block/v46_s6_final_target_on_v45_block/attribution_result.json",
    "source_keys": rate_keys("outcomes", {"eval_seed_start": "evaluation_seed_start (=59000)"}),
    "notes": ("交叉干预：v4.6 cross seed 6 的 exact_final+target_critic 权重在 v4.5 的 validation block（59000）上重放，7/12 成功。"
              "与同一权重在 63000 block 上的 3/12 对照，说明训练权重效应为主、权重×交通块交互为辅；post-hoc，不计入 gate。"
              "同权重在 63000 block 的数值即 F1 选中模型行，未重复成行。"),
})

# ------------------------------------------------------------------- smoke
SM = "engineering/smoke/s1"
sm_paper = J(L(*SM.split("/"), "paper_evaluation_detailed.json"))
rows.append({
    "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
    "candidate": "joint_checkpoint_decoder_selector",
    "scenario": sm_paper["scenario"], "training_seed": 42, "model_kind": "selected",
    "checkpoint_step": 60, "episodes": sm_paper["summary"]["episodes"],
    "eval_seed_start": sm_paper["evaluation_seed_start"], "decoder": "target_critic",
    "success_rate": sm_paper["summary"]["success_rate"],
    "collision_rate": sm_paper["summary"]["collision_rate"],
    "timeout_rate": sm_paper["summary"]["timeout_rate"],
    "mean_return": sm_paper["summary"]["mean_return"],
    "status": "engineering_only", "evidence_kind": "smoke",
    "artifact": REL + "/" + SM + "/paper_evaluation_detailed.json",
    "source_keys": rate_keys("summary", {
        "training_seed":   SM + "/arguments.json:requested_raw_steps.seed (=42)",
        "checkpoint_step": SM + "/checkpoint_audit.json:checkpoints[0].raw_step (=60)",
        "eval_seed_start": "evaluation_seed_start (=99000)",
    }),
    "notes": ("冒烟测试（60 raw steps 训练、单回合评估），仅验证 v4.6 管线可跑通，不是科学证据（evidence_kind=smoke，status=engineering_only）。"
              "唯一一回合 200 决策步全部超时，success=0、collision=0、timeout=1.0。"),
})

sm_rec = J(L(*SM.split("/"), "selector", "receipt.json"))
for c in sm_rec["candidates"]:
    if c["checkpoint_kind"] != "exact_final":
        continue
    s = c["summary"]
    rows.append({
        "batch": BATCH, "phase": PHASE_V46, "variant": "v4_6",
        "candidate": "joint_checkpoint_decoder_selector",
        "scenario": "cross", "training_seed": 42, "model_kind": "exact_final",
        "checkpoint_step": 60, "episodes": s["episodes"],
        "eval_seed_start": sm_rec["calibration_seed_start"],
        "decoder": c["deployment_decoder"],
        "success_rate": s["success_rate"], "collision_rate": s["collision_rate"],
        "timeout_rate": s["timeout_rate"], "mean_return": s["mean_return"],
        "status": "engineering_only", "evidence_kind": "smoke",
        "artifact": REL + "/" + SM + "/selector/receipt.json",
        "source_keys": rate_keys("candidates[checkpoint_kind=exact_final, deployment_decoder=%s].summary" % c["deployment_decoder"], {
            "eval_seed_start": "calibration_seed_start (=98000，单回合标定)",
            "checkpoint_step": SM + "/checkpoint_audit.json:checkpoints[0].raw_step (=60)",
        }),
        "notes": ("冒烟测试的选择器标定回合（1 回合），全部候选对均为 0 成功 / 1 碰撞，除检查点与解码器身份外无区分度，不构成选择器有效性证据。"
                  "注意：本次冒烟中 best_training_success_model.zip 不存在（selector/receipt.json:training_best_missing_fallback_used=true 且 fallback_reuses_exact_final_calibration=true），"
                  "两个 highest_training_success 候选对经 fallback 指向同一个 final_model.zip（sha256 9c311277...），数值与 exact_final 完全相同，为避免把回退值冒充独立测量，未单列成行。"),
    })

# ------------------------------------------------------------------ preflight
pf_meta = [
    ("baseline_cross",              "temporal_graph", "control", "cross"),
    ("candidate_cross",             "v4_6",           "joint_checkpoint_decoder_selector", "cross"),
    ("candidate_carla",             "v4_6",           "joint_checkpoint_decoder_selector", "carla"),
    ("candidate_roundabout_medium", "v4_6",           "joint_checkpoint_decoder_selector", "roundabout_medium"),
]
for name, variant, cand, scen in pf_meta:
    cr_ = J(L("engineering", "preflight", name, "check_result.json"))
    arg = J(L("engineering", "preflight", name, "arguments.json"))["requested_raw_steps"]
    rows.append({
        "batch": BATCH, "phase": PHASE_V46, "variant": variant,
        "candidate": cand, "scenario": cr_["scenario"], "training_seed": arg.get("seed"),
        "model_kind": "other:preflight_action_check", "checkpoint_step": None,
        "episodes": None, "eval_seed_start": None,
        "decoder": "native",
        "success_rate": None, "collision_rate": None, "timeout_rate": None, "mean_return": None,
        "status": "engineering_only", "evidence_kind": "engineering",
        "artifact": REL + "/engineering/preflight/" + name + "/check_result.json",
        "source_keys": {
            "success_rate": None, "collision_rate": None, "timeout_rate": None, "mean_return": None,
            "scenario": "scenario", "training_seed": "arguments.json:requested_raw_steps.seed",
            "decoder": "check_result.json:method_metadata（仅单次前向动作检查，无解码器评估）",
        },
        "notes": ("工程预检（check_only 单步动作检查），只验证动作有限、混合车道码合法、方法元数据绑定正确；未运行任何评估回合，故全部指标为 null。"
                  "check_result.json 的 status=ok、action_finite=true、exact_hybrid_lane_code=%s、checkpoint_selector_constructed_validation_env=%s。"
                  "arguments.json 中 eval_episodes=2、evaluation_seed_start=9000 仅是参数占位，未实际执行评估，故未填入。"
                  % (str(cr_.get("exact_hybrid_lane_code")), str(cr_.get("checkpoint_selector_constructed_validation_env")))),
    })

# ------------------------------------------------------------------- assemble
missing = [
    {"unit": "v4_6 / carla / training_seed 5 / selected（单元 F2__cand__carla__s5__p814d53b7）",
     "reason": "development_decision.json:jobs.F2.accepted=false、reason=\"run directory is absent\"、prerequisites.F2=false；development/runs 下无该目录，F2 从未启动。carla 场景在本批次只有 engineering 级预检，无任何训练或评估数值。"},
    {"unit": "v4_6 / cross / training_seed 7 / selected（单元 F3__cand__cross__s7__p814d53b7）",
     "reason": "development_decision.json:jobs.F3.accepted=false、reason=\"run directory is absent\"；F1 outcome gate 失败后按 on_fail 规则中止（stopped_after=F1），F3 从未执行，故本批次没有独立训练种子的稳定性证据。"},
    {"unit": "v4_6 / roundabout_medium / training_seed 2 / selected（单元 F4__cand__ram__s2__p814d53b7）",
     "reason": "development_decision.json:jobs.F4.accepted=false、reason=\"run directory is absent\"；roundabout_medium 在本批次只有 engineering 级预检，无训练或评估数值。"},
    {"unit": "formal_test 证据（任意变体 / 任意场景）",
     "reason": "preregistration_receipt.json:formal_test_unlocked=false、formal_test_accessed=false；development_decision.json:formal_test_unlocked=false；engineering_receipt.json:formal_test_accessed=false。正式测试分区全程锁定且从未访问，本批次只存在 development + post-hoc attribution + smoke + engineering 四类证据。"},
    {"unit": "v4_5 / cross / training_seed 4 / exact_final / target_critic / block 59000 的 mean_return",
     "reason": "本批次 attribution_summary.json 的 crossed_checkpoint_block_intervention 只存了 success/collision/timeout 三个矩阵，没有 return；真正的源文件在批次之外（results_topo_v4_5_dev/attribution/e1_failure/closed_loop/final_target_validation/attribution_result.json），本批次内取不到，置 null。"},
    {"unit": "v4_6 冒烟中 highest_training_success × {target_critic, fusion_0_90} 两格的独立数值",
     "reason": "engineering/smoke/s1 下不存在 best_training_success_model.zip；selector/receipt.json:training_best_missing_fallback_used=true 且 fallback_reuses_exact_final_calibration=true，两个 highest_training_success 候选对的 checkpoint_sha256 与 exact_final 相同（9c3112776b23...），数值是 fallback 复用而非独立测量，故未单列成行。"},
    {"unit": "训练过程中的周期性评估曲线（任意单元）",
     "reason": "本批次所有 run 的 arguments.json 均设 eval_freq=0（F1、preflight、smoke 一致），训练期间不产生评估记录；只有训练结束后的单次评估（paper/final evaluation）与标定回合。训练阶段可比数值仅有 training_diagnostics.json 的损失/诊断统计与 best_training_success.json 的 success_rate_last_20。"},
    {"unit": "v4_6 的跨场景泛化结论（carla / roundabout_medium / left_turn 等）",
     "reason": "本批次仅 cross 场景进入 development 阶段且已失败中止；carla 与 roundabout_medium 停留在单步预检，left_turn/roundabout 从未出现。任何跨场景结论在本批次内无证据支撑。"},
]

out = {
    "directory": BATCH,
    "phase": PHASE_V46,
    "purpose": ("v4.6 开发线只做一处科学改动：把部署选择从「只选 checkpoint」扩展为「checkpoint × decoder 联合选择」"
                "（候选 2 个检查点 × 2 个解码器 = 4 对），且标定只在 train 交通分区进行，验证与正式测试不得用于选择。"
                "批次计划 4 个 fresh 开发单元（F1 cross/s6、F2 carla/s5、F3 cross/s7、F4 roundabout_medium/s2）；"
                "F1 通过 artifact/selector/机制检查但在 outcome gate 失败（success 0.25 < 0.50），按 on_fail 规则中止，F2-F4 从未执行。"
                "随后在本批次内做失败归因（4 候选对事后重放 + v4.5/v4.6 权重 × 59000/63000 交通块交叉干预），并据此提出 v4.7 的 16 步信用分配修复假设。"),
    "status": "partial",
    "protocol_files": [
        REL + "/preregistration_receipt.json",
        REL + "/development/stage_plan.json",
        REL + "/development/development_decision.json",
        REL + "/development/summary.json",
        REL + "/development/per_run.json",
        REL + "/development/runs/F1__cand__cross__s6__p814d53b7/selector/receipt.json",
        REL + "/attribution/f1_failure/attribution_summary.json",
        REL + "/attribution/f1_failure/deep_attribution.md",
        REL + "/engineering/engineering_receipt.json",
        REL + "/engineering/implementation_freeze.json",
        "experiments/topo_scene_v4/experiment_contract_v4_6.yaml",
        "experiments/topo_scene_v4/V4_6_JOINT_CHECKPOINT_DECODER_SELECTOR.md",
    ],
    "rows": rows,
    "missing": missing,
    "notes": (
        "口径与判读要点（全部数值直接读自源文件，未做任何推算或插补）：\n"
        "1) 训练 vs 评估：本批次训练阶段不发评估曲线（eval_freq=0）。每个单元只有两类闭环数值——(a) train 交通分区上的标定回合（F1 种子 72000-72011，用于选择器排序），"
        "(b) validation 交通分区上的最终评估（F1 种子 63000-63011）。前者是选择环节输入，后者才是 outcome gate 依据，二者不可混用，表中以 eval_seed_start 与 notes 区分。\n"
        "2) 开发证据 vs 正式测试：formal_test_unlocked / formal_test_accessed 全为 false，本批次不存在任何正式测试证据；"
        "最高证据等级是 development（仅 F1），其余为 post-hoc 归因（文件内明确 declared 不计入任何 gate）、smoke 与 engineering。\n"
        "3) 完成 vs 中止：F1 本身跑完（12/12 回合成败均完整记录），但其 job gate outcome_passed=false，development_decision 的 decision=fail、complete=false、stopped_after=F1；"
        "F2/F3/F4 的 run 目录不存在（accepted=false），属于「从未执行」而非「执行后失败」。因此本批次整体只能记为 partial，绝不能当作 completed。\n"
        "4) 去重：F1 选中模型的 4 个指标在 final_evaluation.json、development/summary.json、development/per_run.json、development_decision.json:jobs.F1.row、"
        "attribution_summary.json:validation_pair_matrix.exact_final__target_critic 中重复出现且数值一致，只保留 paper_evaluation_detailed.json 一份（源 sha256 db7872b7...）。"
        "4 个标定候选对的数值同样在 selector/receipt.json:candidates[].summary、selector/cal/*/evaluation.json、selector/cal/*/detailed.json 与 development/summary.json:calibration_candidate_outcomes 中重复，"
        "本表以 selector/cal/*/detailed.json 为权威来源（其 sha256 即 receipt 中的 calibration_result_sha256，如 exact_final×target_critic = 5d9a5f27...）。\n"
        "5) 检查点身份：F1 的 highest_training_success 检查点以 _raw_steps_seen=14314 标度（不是 20000），exact_final 为 20000，已在 checkpoint_step 字段区分；"
        "selected_model.zip 是对 exact_final 的策略类重序列化（policy_parameter_state_preserved=true，策略参数状态 sha 9ec4bb15... 两侧一致），未改变权重。\n"
        "6) 关键坑 / 异常：(a) 选择器标定 8/12 成功，但同一权重在 validation 只有 3/12（0.25），标定与验证存在明显分布差，是本批次失败的核心现象；"
        "(b) 4 个候选对在 validation 上的最好成绩也只有 3/12，事后逐回合 oracle 才凑到 7/12，说明是「互补失败」而非可用选择规则（misselection 只改变 collision/timeout 构成，success 计数无 regret）；"
        "(c) 交叉干预显示训练权重效应为主（v4.5 seed4 在 63000 block 仍有 10/12 成功、0 碰撞），交通块难度单独不足以解释；"
        "(d) 冒烟运行因 best_training_success 模型缺失走了 fallback，4 个候选对塌缩成同一检查点，(checkpoint×decoder) 联合选择在冒烟中实际未被检验；"
        "(e) 归因中可比训练运行仅 n=4，Spearman=-1.0 只是诊断协变，文件自身声明 statistical_role=diagnostic_covariation_only_not_a_preregistered_test，不可当结论；"
        "(f) terminal_credit_coverage 的 0.032 / 0.351 / 3.40x 等数字是 trace 推导的设计算例（role=trace_derived_design_calculation_not_an_experimental_result），不是实验结果，故未进入 rows。\n"
        "7) 可信度：本批次数值可信度高——文件带 sha256 契约绑定、implementation_freeze、engineering_receipt（targeted 14 项与 full regression 302 项测试全通过、0 失败）与 no_fabrication 声明；"
        "F1 的 12 条 episode 记录与聚合指标自洽（成功 3/12、碰撞 6/12、超时 3/12、off-route 0，且与 completion_time 记录一致）。"
        "可信度折损点在于样本量极小（每单元 12 回合、仅 1 个训练种子），且 F2-F4 缺失使本批次无法给出跨场景/跨种子的泛化结论。"
    ),
}

dest_dir = os.path.join(ROOT, "reports", "all_experiments_summary_20260919", "raw")
os.makedirs(dest_dir, exist_ok=True)
dest = os.path.join(dest_dir, "results_topo_v4_6_dev.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("WROTE", dest)
print("rows:", len(rows))
print("status:", dict(Counter(r["status"] for r in rows)))
print("evidence_kind:", dict(Counter(r["evidence_kind"] for r in rows)))
print("missing:", len(missing))
keys = set()
for r in rows: keys.add(tuple(sorted(r.keys())))
print("row key-sets:", len(keys))
canon = {"batch","phase","variant","candidate","scenario","training_seed","model_kind","checkpoint_step",
         "episodes","eval_seed_start","decoder","success_rate","collision_rate","timeout_rate","mean_return",
         "status","evidence_kind","artifact","source_keys","notes"}
print("exact canonical field set:", all(k == canon for k in keys))
