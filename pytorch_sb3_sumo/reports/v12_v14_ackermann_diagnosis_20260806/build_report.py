"""Build the v12/v14 controller-attribution technical report artifact."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPORT_ROOT = Path(__file__).resolve().parent


def table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "spacious",
        "layout": "full",
        "defaultSort": {"field": "order", "direction": "asc"},
        "columns": columns,
    }


def build_artifact() -> dict[str, object]:
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    title = "v12 与 v14：SMARTS Ackermann 代理是否限制了基线速度？"

    sources = [
        {
            "id": "formal_results",
            "label": "v12/v14 五 seed 正式评测与论文参考表",
            "path": "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/formal_results_source.sql",
            "query": {
                "engine": "local JSON/CSV audit",
                "id": "formal_results",
                "description": "从每个 run 的 paper_evaluation_detailed.json 逐回合重算成功、碰撞、超时和成功完成时间。",
                "executed_at": generated_at,
                "language": "json",
                "tables_used": [
                    "results_sb3_sumo_paper/paper_source_audited_runs_v12_frozen/paper__proposed__{left_turn,cross}__seed{0..4}/paper_evaluation_detailed.json",
                    "results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/paper__{sac,ppo}__{left_turn,cross}__seed{0..4}/paper_evaluation_detailed.json",
                    "pytorch_sb3_sumo/experiments/sb3_sumo_paper/paper_reference_tables.json",
                ],
                "metric_definitions": [
                    "success rate = reached-goal episodes / all evaluation episodes",
                    "completion time = population mean over successful episodes only",
                    "v12/v14 aggregate = pooled successful episodes across five training seeds",
                ],
            },
        },
        {
            "id": "controller_counterfactuals",
            "label": "同权重 direct/proxy 反事实评测",
            "path": "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/controller_counterfactuals_source.sql",
            "query": {
                "engine": "SB3 deterministic evaluation",
                "id": "controller_counterfactuals",
                "description": "不更新模型权重，只切换 ego_control_profile；固定 checkpoint、交通协议、回合种子和 episode limit。",
                "executed_at": generated_at,
                "language": "json",
                "tables_used": [
                    "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/evidence/sac_left_turn_seed{0,1,3}_direct_10ep.json",
                    "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/evidence/sac_cross_seed2_direct_10ep.json",
                    "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/evidence/v12_proposed_left_seed0_proxy_10ep.json",
                    "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/evidence/v12_proposed_cross_seed0_proxy_3ep.json",
                ],
                "filters": [
                    "deterministic policy",
                    "same evaluation seed range within each pair",
                    "weights_updated = false",
                ],
            },
        },
        {
            "id": "speed_traces",
            "label": "双汇入请求/生效/实际速度轨迹",
            "path": "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/speed_traces_source.sql",
            "query": {
                "engine": "TraCI policy trace",
                "id": "speed_traces",
                "description": "每个决策记录策略请求速度、代理生效目标速度、实际 ego 速度和路线进度。",
                "executed_at": generated_at,
                "language": "json",
                "tables_used": [
                    "pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/evidence/ppo_cross_seed0_trace_h600.json",
                    "pytorch_sb3_sumo/reports/cross_roundabout_anomaly_diagnosis_20260805/evidence/sac_cross_seed0_trace_h600.json",
                    "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/evidence/sac_cross_seed2_trace_paper.json",
                ],
            },
        },
        {
            "id": "controller_contract",
            "label": "v12/v14 协议与控制器实现",
            "path": "pytorch_sb3_sumo/reports/v12_v14_ackermann_diagnosis_20260806/controller_contract_source.sql",
            "query": {
                "engine": "source audit",
                "id": "controller_contract",
                "description": "v12 默认 direct；v14 使用 smarts_ackermann_proxy、frozen_80_20 和 paper episode limits。",
                "executed_at": generated_at,
                "language": "python/json",
                "tables_used": [
                    "pytorch_sb3_sumo/envs/sumo/sumo_env.py",
                    "pytorch_sb3_sumo/experiments/sb3_sumo_paper/protocol.json",
                    "pytorch_sb3_sumo/experiments/sb3_sumo_paper/protocol_baselines_v14.json",
                ],
            },
        },
    ]

    formal_rows = [
        {"order": 1, "scenario": "左转", "result": "论文 Proposed", "method": "Proposed", "success": 0.94, "collision": 0.04, "timeout": 0.00, "time_s": 12.5, "note": "论文参考"},
        {"order": 2, "scenario": "左转", "result": "v12 Proposed", "method": "Proposed", "success": 0.98, "collision": 0.016, "timeout": 0.004, "time_s": 11.866, "note": "5 seeds / 245 个成功回合"},
        {"order": 3, "scenario": "左转", "result": "论文 SAC", "method": "SAC", "success": 0.68, "collision": 0.28, "timeout": 0.00, "time_s": 19.2, "note": "论文参考"},
        {"order": 4, "scenario": "左转", "result": "v14 SAC", "method": "SAC", "success": 0.92, "collision": 0.068, "timeout": 0.012, "time_s": 19.443, "note": "5 seeds / 230 个成功回合"},
        {"order": 5, "scenario": "左转", "result": "论文 PPO", "method": "PPO", "success": 0.36, "collision": 0.50, "timeout": 0.10, "time_s": 36.4, "note": "论文参考"},
        {"order": 6, "scenario": "左转", "result": "v14 PPO", "method": "PPO", "success": 0.528, "collision": 0.468, "timeout": 0.004, "time_s": 24.460, "note": "5 seeds / 132 个成功回合"},
        {"order": 7, "scenario": "双汇入", "result": "论文 Proposed", "method": "Proposed", "success": 0.96, "collision": 0.02, "timeout": 0.00, "time_s": 28.6, "note": "论文参考"},
        {"order": 8, "scenario": "双汇入", "result": "v12 Proposed", "method": "Proposed", "success": 0.984, "collision": 0.016, "timeout": 0.00, "time_s": 45.757, "note": "5 seeds / 246 个成功回合"},
        {"order": 9, "scenario": "双汇入", "result": "论文 SAC", "method": "SAC", "success": 0.62, "collision": 0.22, "timeout": 0.00, "time_s": 34.9, "note": "论文参考"},
        {"order": 10, "scenario": "双汇入", "result": "v14 SAC", "method": "SAC", "success": 0.152, "collision": 0.108, "timeout": 0.740, "time_s": 29.808, "note": "完成时间仅来自 seed2 的 38 个成功回合"},
        {"order": 11, "scenario": "双汇入", "result": "论文 PPO", "method": "PPO", "success": 0.36, "collision": 0.64, "timeout": 0.00, "time_s": 36.3, "note": "论文参考"},
        {"order": 12, "scenario": "双汇入", "result": "v14 PPO", "method": "PPO", "success": 0.00, "collision": 0.176, "timeout": 0.824, "time_s": None, "note": "5 seeds / 0 个成功回合"},
    ]

    left_ab = [
        {"checkpoint": "v12 Proposed s0", "profile": "direct", "seconds": 11.25, "success": 1.00, "episodes": 10},
        {"checkpoint": "v12 Proposed s0", "profile": "proxy", "seconds": 14.13, "success": 1.00, "episodes": 10},
        {"checkpoint": "v14 SAC s0", "profile": "direct", "seconds": 14.744, "success": 0.90, "episodes": 10},
        {"checkpoint": "v14 SAC s0", "profile": "proxy", "seconds": 17.970, "success": 1.00, "episodes": 10},
        {"checkpoint": "v14 SAC s1", "profile": "direct", "seconds": 32.940, "success": 1.00, "episodes": 10},
        {"checkpoint": "v14 SAC s1", "profile": "proxy", "seconds": 35.200, "success": 1.00, "episodes": 10},
        {"checkpoint": "v14 SAC s3", "profile": "direct", "seconds": 11.940, "success": 1.00, "episodes": 10},
        {"checkpoint": "v14 SAC s3", "profile": "proxy", "seconds": 13.675, "success": 0.80, "episodes": 10},
    ]

    counterfactual_rows = [
        {"order": 1, "scenario": "左转", "checkpoint": "v12 Proposed seed0", "sample": "10 回合", "direct_success": 1.00, "proxy_success": 1.00, "direct_time_s": 11.250, "proxy_time_s": 14.130, "delta_s": 2.880, "interpretation": "代理减速，但强策略不失效"},
        {"order": 2, "scenario": "左转", "checkpoint": "v14 SAC seed0", "sample": "10 回合", "direct_success": 0.90, "proxy_success": 1.00, "direct_time_s": 14.744, "proxy_time_s": 17.970, "delta_s": 3.226, "interpretation": "代理更慢但碰撞更少"},
        {"order": 3, "scenario": "左转", "checkpoint": "v14 SAC seed1", "sample": "10 回合", "direct_success": 1.00, "proxy_success": 1.00, "direct_time_s": 32.940, "proxy_time_s": 35.200, "delta_s": 2.260, "interpretation": "切回 direct 仍很慢，主要是策略差异"},
        {"order": 4, "scenario": "左转", "checkpoint": "v14 SAC seed3", "sample": "10 回合", "direct_success": 1.00, "proxy_success": 0.80, "direct_time_s": 11.940, "proxy_time_s": 13.675, "delta_s": 1.735, "interpretation": "代理减速且本样本碰撞增加"},
        {"order": 5, "scenario": "双汇入", "checkpoint": "v12 Proposed seed0", "sample": "3 回合", "direct_success": 1.00, "proxy_success": 1.00, "direct_time_s": 40.633, "proxy_time_s": 45.167, "delta_s": 4.533, "interpretation": "600步下均成功；400步下三回合都会被截断或临界"},
        {"order": 6, "scenario": "双汇入", "checkpoint": "v14 SAC seed2", "sample": "10 回合", "direct_success": 0.30, "proxy_success": 0.90, "direct_time_s": 27.733, "proxy_time_s": 29.878, "delta_s": 2.144, "interpretation": "direct 更快但碰撞显著增加；成功样本集合不同"},
    ]

    speed_rows = [
        {"order": 1, "checkpoint": "v14 PPO cross seed0", "episodes": 3, "requested_mps": 0.046, "effective_mps": 0.046, "actual_mps": 0.046, "result": "600步全超时", "diagnosis": "策略主动近乎停车；代理未截速"},
        {"order": 2, "checkpoint": "v14 SAC cross seed0", "episodes": 3, "requested_mps": 0.121, "effective_mps": 0.121, "actual_mps": 0.121, "result": "600步全超时", "diagnosis": "策略主动近乎停车；代理未截速"},
        {"order": 3, "checkpoint": "v14 SAC cross seed2", "episodes": 3, "requested_mps": 9.706, "effective_mps": 9.010, "actual_mps": 9.000, "result": "3/3成功，29.4–30.9秒", "diagnosis": "主要是起步加速钳制；代理不构成40秒硬瓶颈"},
    ]

    causes = [
        {"order": 1, "factor": "Ackermann 代理", "judgment": "部分成立，不是主因", "evidence": "左转同权重增加1.7–3.2秒；双汇入样本增加约2.1–4.5秒", "confidence": "高", "next_test": "记录 curve-cap 与 accel-clamp 的逐tick命中率"},
        {"order": 2, "factor": "双汇入 400 vs 600 tick", "judgment": "主放大器", "evidence": "v12的246个成功回合只有14个（5.7%）在40秒内完成", "confidence": "高", "next_test": "控制器×时限 2×2 配对重训/重评"},
        {"order": 3, "factor": "策略/训练 seed 崩溃", "judgment": "v14双汇入主因", "evidence": "失败seed请求速度仅0.046/0.121m/s；SAC seed2同代理可达76%成功", "confidence": "高", "next_test": "比较训练期首次成功、请求速度分布与best checkpoint"},
        {"order": 4, "factor": "方法差异", "judgment": "正式v12/v14不可直接归因", "evidence": "v12是Proposed，v14是重建SAC与PPO；网络、观测和优化器不同", "confidence": "确定", "next_test": "完成v15 Proposed/MST同环境五seed"},
        {"order": 5, "factor": "交通划分", "judgment": "影响碰撞/成功，非统一速度解释", "evidence": "同checkpoint frozen_80_20与source_all在左转可相差明显", "confidence": "中", "next_test": "所有A/B固定同一traffic_variant序列"},
        {"order": 6, "factor": "固定起点与发布路线", "judgment": "可能使40秒口径偏紧", "evidence": "论文称随机起点，发布场景与迁移使用固定任务起点", "confidence": "中低", "next_test": "核对论文有效行驶距离并报告route-normalized speed"},
    ]

    headline = [{
        "proxy_left_added_seconds_mean": 2.407,
        "v12_cross_within_40s": 0.0569,
        "v14_cross_stall_speed_sac": 0.121,
        "v14_sac_left_time_gap_vs_paper": 0.243,
    }]

    cards = [
        {"id": "left-proxy-delta", "description": "v14 SAC seeds 0/1/3、同checkpoint各10回合；proxy减direct。", "dataset": "headline", "sourceId": "controller_counterfactuals", "metrics": [{"label": "左转代理平均增加", "field": "proxy_left_added_seconds_mean", "format": "number", "suffix": " s"}]},
        {"id": "cross-horizon", "description": "v12 Proposed五seed的246个成功回合，按v14的40秒阈值回放。", "dataset": "headline", "sourceId": "formal_results", "metrics": [{"label": "v12双汇入≤40秒", "field": "v12_cross_within_40s", "format": "percent"}]},
        {"id": "cross-stall", "description": "v14 SAC cross seed0在600步诊断中的平均请求速度。", "dataset": "headline", "sourceId": "speed_traces", "metrics": [{"label": "失败策略请求速度", "field": "v14_cross_stall_speed_sac", "format": "number", "suffix": " m/s"}]},
        {"id": "sac-paper-time", "description": "v14 SAC左转19.443秒减论文SAC 19.2秒。", "dataset": "headline", "sourceId": "formal_results", "metrics": [{"label": "SAC左转时间差", "field": "v14_sac_left_time_gap_vs_paper", "format": "number", "suffix": " s"}]},
    ]

    charts = [{
        "id": "left-controller-ab",
        "title": "左转：同一权重切换 direct / smarts_ackermann_proxy",
        "subtitle": "代理稳定增加约1.7–3.2秒，但seed1切回direct后仍需32.94秒，说明剩余慢速来自策略。",
        "intent": "comparison",
        "question": "控制代理能解释多少左转完成时间差？",
        "rationale": "同checkpoint、同交通与同种子的成对柱状比较最接近单因素反事实。",
        "comparisonContext": {
            "denominator": "成功回合完成时间；每对10个相同评测种子",
            "grain": "checkpoint × controller profile",
            "normalization": "原始SUMO步×0.1秒",
            "semanticFamily": "completion time",
            "unit": "s",
        },
        "type": "bar",
        "dataset": "left_ab",
        "sourceId": "controller_counterfactuals",
        "encodings": {
            "x": {"field": "checkpoint", "type": "ordinal", "label": "检查点"},
            "y": {"field": "seconds", "type": "quantitative", "format": "number", "label": "成功完成时间"},
            "color": {"field": "profile", "type": "nominal", "label": "控制方式"},
            "tooltip": [
                {"field": "profile", "type": "nominal", "label": "控制方式"},
                {"field": "seconds", "type": "quantitative", "format": "number", "label": "秒"},
                {"field": "success", "type": "quantitative", "format": "percent", "label": "成功率"},
            ],
        },
        "xAxisTitle": "检查点",
        "yAxisTitle": "成功完成时间（秒）",
        "valueFormat": "number",
        "layout": "full",
        "maxRows": 12,
        "palette": {"kind": "categorical", "name": "blue-gold"},
        "labels": {"values": "all"},
        "settings": {"showValues": True, "sort": "custom", "categoryLabelPolicy": "wrap"},
    }]

    tables = [
        table("formal-table", "正式结果：必须按同方法与论文参考比较", "v12/v14均为当前五seed primary结果；完成时间仅统计成功回合。", "formal_rows", "formal_results", [
            {"field": "order", "label": "序", "format": "number"},
            {"field": "scenario", "label": "场景", "type": "text"},
            {"field": "result", "label": "结果", "type": "text"},
            {"field": "method", "label": "方法", "type": "text"},
            {"field": "success", "label": "成功率", "format": "percent"},
            {"field": "collision", "label": "碰撞率", "format": "percent"},
            {"field": "timeout", "label": "超时率", "format": "percent"},
            {"field": "time_s", "label": "成功时间(s)", "format": "number"},
            {"field": "note", "label": "口径", "type": "text"},
        ]),
        table("counterfactual-table", "同权重控制器反事实", "仅切换direct/proxy；这是评估期干预，不等价于重新训练。", "counterfactual_rows", "controller_counterfactuals", [
            {"field": "order", "label": "序", "format": "number"},
            {"field": "scenario", "label": "场景", "type": "text"},
            {"field": "checkpoint", "label": "检查点", "type": "text"},
            {"field": "sample", "label": "样本", "type": "text"},
            {"field": "direct_success", "label": "direct成功", "format": "percent"},
            {"field": "proxy_success", "label": "proxy成功", "format": "percent"},
            {"field": "direct_time_s", "label": "direct时间(s)", "format": "number"},
            {"field": "proxy_time_s", "label": "proxy时间(s)", "format": "number"},
            {"field": "delta_s", "label": "proxy增量(s)", "format": "number"},
            {"field": "interpretation", "label": "解释", "type": "text"},
        ]),
        table("speed-table", "双汇入速度链：请求 → 生效 → 实际", "失败seed与成功seed都在同一smarts_ackermann_proxy下。", "speed_rows", "speed_traces", [
            {"field": "order", "label": "序", "format": "number"},
            {"field": "checkpoint", "label": "检查点", "type": "text"},
            {"field": "episodes", "label": "回合", "format": "number"},
            {"field": "requested_mps", "label": "请求(m/s)", "format": "number"},
            {"field": "effective_mps", "label": "生效(m/s)", "format": "number"},
            {"field": "actual_mps", "label": "实际(m/s)", "format": "number"},
            {"field": "result", "label": "结果", "type": "text"},
            {"field": "diagnosis", "label": "诊断", "type": "text"},
        ]),
        table("cause-table", "原因归因与证据强度", "把可证实的机械影响与尚未隔离的训练/协议影响分开。", "causes", "controller_contract", [
            {"field": "order", "label": "序", "format": "number"},
            {"field": "factor", "label": "因素", "type": "text"},
            {"field": "judgment", "label": "判断", "type": "text"},
            {"field": "evidence", "label": "现有证据", "type": "text"},
            {"field": "confidence", "label": "置信度", "type": "text"},
            {"field": "next_test", "label": "下一步", "type": "text"},
        ]),
    ]

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
        {"id": "summary", "type": "markdown", "layout": "full", "body": (
            "## 技术摘要\n\n"
            "**判断：部分是，但不是主因。** `smarts_ackermann_proxy` 确实把左转成功时间增加约1.7–3.2秒，"
            "并在双汇入样本中增加约2.1–4.5秒；因此它会限制峰值速度，特别是起步和弯道阶段。"
            "但v14双汇入的大面积失败主要不是代理截速：失败策略请求速度本身只有0.046/0.121 m/s，"
            "请求、生效与实际速度相等；同一代理下SAC seed2却能达到76%成功并在29.8秒完成。\n\n"
            "按同方法与论文比较，v14基线速度也没有系统性偏低：SAC左转19.443秒几乎等于论文19.2秒，"
            "PPO左转24.46秒快于论文36.4秒，SAC双汇入的成功回合29.81秒快于论文34.9秒。"
            "真正的问题是成功率/seed稳定性。v12双汇入只是在成功率上接近论文；其45.76秒比论文28.6秒慢约60%。"
        )},
        {"id": "cards", "type": "metric-strip", "layout": "full", "cardIds": [card["id"] for card in cards]},
        {"id": "formal-intro", "type": "markdown", "layout": "full", "body": (
            "## 先纠正比较口径\n\n"
            "v12正式结果是Proposed，v14正式结果是SAC/PPO；二者不是同算法。"
            "因此不能把v12 Proposed比v14 SAC/PPO更快直接归因给控制器。正确比较是v12 Proposed对论文Proposed、"
            "v14 SAC/PPO分别对论文SAC/PPO。这个口径下，v14的完成速度并不偏慢，偏差集中在双汇入成功率。"
        )},
        {"id": "formal-block", "type": "table", "layout": "full", "tableId": "formal-table"},
        {"id": "ab-intro", "type": "markdown", "layout": "full", "body": (
            "## 控制器的可测量影响\n\n"
            "同权重反事实证明代理不是无影响：左转四个检查点全部变慢，v14 SAC三个seed平均增加2.41秒。"
            "但seed1在direct下仍需32.94秒，说明约20秒的额外慢速属于策略本身，而不是统一的机械限速。"
            "成功率影响没有固定方向：代理有时减少碰撞，有时因慢速进入冲突区而增加碰撞。"
        )},
        {"id": "ab-chart", "type": "chart", "layout": "full", "chartId": "left-controller-ab"},
        {"id": "ab-table-block", "type": "table", "layout": "full", "tableId": "counterfactual-table"},
        {"id": "cross-intro", "type": "markdown", "layout": "full", "body": (
            "## 双汇入：代理不是失败seed的直接瓶颈\n\n"
            "PPO/SAC seed0在600步诊断中长期请求近乎0 m/s，且请求速度=代理生效速度=实际速度；"
            "车辆60秒仍停在第一条边。因此这些失败不是5.56/6.94 m/s弯道上限或逐tick加减速钳制。"
            "相反，SAC seed2请求约9.7 m/s，代理后实际约9.0 m/s，三回合全部在31秒内完成。\n\n"
            "episode limit是更强的放大器：v12的246个双汇入成功回合仅14个在40秒内完成。"
            "即使保持direct，94.3%的v12成功回合在v14的400 tick口径下也会超时；代理再增加数秒会进一步恶化。"
        )},
        {"id": "speed-table-block", "type": "table", "layout": "full", "tableId": "speed-table"},
        {"id": "cause-block", "type": "table", "layout": "full", "tableId": "cause-table"},
        {"id": "scope", "type": "markdown", "layout": "full", "body": (
            "## 范围、方法与稳健性\n\n"
            "正式汇总使用当前已完成的左转/双汇入五个训练seed，每seed 50个确定性回合；"
            "成功完成时间从逐回合记录重新汇总，失败回合不进入时间均值。控制器A/B使用相同checkpoint、"
            "相同交通协议、相同评测seed和相同episode limit，只改变`ego_control_profile`，且验证模型/环境空间一致。"
            "双汇入速度诊断直接读取策略动作、`effective_target_speed`与TraCI实际速度。"
        )},
        {"id": "limitations", "type": "markdown", "layout": "full", "body": (
            "## 限制\n\n"
            "A/B是评估期反事实：模型原本在各自控制器下训练，切换控制器会产生分布偏移，不能替代成对重训。"
            "左转A/B每个checkpoint仅10回合，v12双汇入仅3回合；它们用于机制定位，不替代正式五seed结果。"
            "左转没有逐tick curve-cap命中日志，所以当前只能估计代理总效应，不能把弯道限速与加减速约束进一步拆开。"
            "v14 SAC双汇入完成时间只来自成功的seed2，存在成功条件选择偏差。"
        )},
        {"id": "recommendations", "type": "markdown", "layout": "full", "body": (
            "## 建议\n\n"
            "1. 保留`smarts_ackermann_proxy`作为正式SMARTS保真协议，不建议仅为提高速度而删除。\n"
            "2. 完成v15 Proposed/MST在v14同环境下的五seed结果，这是消除方法混杂的首要实验。\n"
            "3. 对双汇入做 controller（direct/proxy）× horizon（400/600）的2×2同seed配对重训；"
            "同时记录请求速度、有效速度、弯道限速命中率、加/减速钳制率和route-normalized progress。\n"
            "4. 把双汇入结论拆成成功率与效率：v12成功率复现良好，但完成时间并未复现论文，不能据此称场景整体已复现。"
        )},
        {"id": "questions", "type": "markdown", "layout": "full", "body": (
            "## 待确认问题\n\n"
            "正式口径是否优先追求论文400 tick，还是发布`tools/test.py`的600 tick？若目标是论文数值，"
            "还需确认论文随机起点对应的有效行驶距离；否则固定起点+400 tick可能把路线差异误计为策略失败。"
        )},
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "v12/v14正式结果、速度轨迹与同权重控制器反事实的原因归因。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "formal_rows": formal_rows,
                "left_ab": left_ab,
                "counterfactual_rows": counterfactual_rows,
                "speed_rows": speed_rows,
                "causes": causes,
            },
        },
        "sources": sources,
    }


def main() -> int:
    output = REPORT_ROOT / "artifact.json"
    output.write_text(
        json.dumps(build_artifact(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
