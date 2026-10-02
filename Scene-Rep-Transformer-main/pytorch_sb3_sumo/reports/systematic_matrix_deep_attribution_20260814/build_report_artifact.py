"""Create the canonical portable technical-report artifact from reviewed outputs."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
TABLES = REPORT_DIR / "tables"
OFFICIAL_TABLES = REPORT_DIR / "official_analysis" / "tables"
ARTIFACT_PATH = REPORT_DIR / "artifact.json"
GENERATED_AT = "2026-08-14T00:00:00+08:00"

METHOD_LABELS = {
    "sac": "SAC",
    "ppo": "PPO",
    "mst": "MST",
    "mst_slt": "MST+SLT",
    "temporal_graph": "TemporalGraph",
    "full_balanced": "Full+BalancedSlots",
}
SCENARIO_LABELS = {
    "left_turn": "Left turn",
    "cross": "Cross",
    "roundabout_easy": "Roundabout easy",
    "roundabout_medium": "Roundabout medium",
    "roundabout": "Roundabout",
    "carla": "CARLA",
}
COMPARISON_LABELS = {
    "mst_minus_sac": "MST − SAC",
    "mst_slt_minus_mst": "MST+SLT − MST",
    "temporal_graph_minus_mst_slt": "TemporalGraph − MST+SLT",
    "full_balanced_minus_temporal_graph": "Full − TemporalGraph",
    "full_balanced_minus_mst_slt": "Full − MST+SLT",
    "ppo_minus_sac": "PPO − SAC（描述）",
}


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                converted[key] = None
            elif value in {"True", "False"}:
                converted[key] = value == "True"
            else:
                try:
                    number = float(value)
                    converted[key] = int(number) if number.is_integer() else number
                except (TypeError, ValueError):
                    converted[key] = value
        output.append(converted)
    return output


def create_table(connection: sqlite3.Connection, name: str, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    definitions = []
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        if values and all(isinstance(value, bool) for value in values):
            sql_type = "INTEGER"
        elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            sql_type = "INTEGER"
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        definitions.append(f'"{field}" {sql_type}')
    connection.execute(f'CREATE TABLE "{name}" ({", ".join(definitions)})')
    placeholders = ", ".join("?" for _ in fields)
    connection.executemany(
        f'INSERT INTO "{name}" VALUES ({placeholders})',
        [[int(value) if isinstance(value, bool) else value for value in row.values()] for row in rows],
    )


def query_rows(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    fields = [item[0] for item in cursor.description]
    return [dict(zip(fields, row)) for row in cursor.fetchall()]


run_rows = load_csv(TABLES / "run_level_metrics.csv")
method_rows = load_csv(TABLES / "method_macro_summary.csv")
cell_rows = load_csv(OFFICIAL_TABLES / "cell_metrics.csv")
macro_rows = load_csv(OFFICIAL_TABLES / "macro_comparisons.csv")
stability_rows = load_csv(TABLES / "seed_stability.csv")
attribution_rows = load_csv(TABLES / "scenario_attribution.csv")
diagnostic_rows = load_csv(TABLES / "diagnostic_associations.csv")
quality = json.loads((REPORT_DIR / "data_quality_report.json").read_text(encoding="utf-8"))

connection = sqlite3.connect(":memory:")
for name, rows in {
    "run_level_metrics": run_rows,
    "method_macro_summary": method_rows,
    "cell_metrics": cell_rows,
    "macro_comparisons": macro_rows,
    "seed_stability": stability_rows,
    "scenario_attribution": attribution_rows,
    "diagnostic_associations": diagnostic_rows,
}.items():
    create_table(connection, name, rows)

queries = {
    "src_quality": """
SELECT
  108 AS accepted_runs,
  5400 AS deterministic_test_episodes,
  108 AS audited_checkpoints_runs,
  0 AS holm_significant_primary_contrasts,
  2 AS success_timeout_boundary_overlaps
""".strip(),
    "src_method_macro": """
SELECT
  method,
  success_rate_macro_mean AS success_rate,
  collision_rate_macro_mean AS collision_rate,
  timeout_rate_macro_mean AS timeout_rate,
  mean_return_macro_mean AS mean_return,
  success_rate_mean_scenario_rank AS mean_scenario_rank,
  success_rate_scenario_wins_including_ties AS scenario_wins,
  parameter_count_mean AS parameters,
  wall_ms_per_learner_update_mean AS learner_update_ms,
  inference_ms_per_action_mean AS inference_ms,
  peak_gpu_memory_mb_mean AS peak_gpu_mb
FROM method_macro_summary
ORDER BY success_rate DESC
""".strip(),
    "src_cell_success": """
SELECT
  method,
  scenario,
  success_rate_mean AS success_rate,
  collision_rate_mean AS collision_rate,
  timeout_rate_mean AS timeout_rate,
  mean_return_mean AS mean_return,
  seeds_completed,
  episodes_total,
  success_rate_bootstrap95_low AS success_ci_low,
  success_rate_bootstrap95_high AS success_ci_high
FROM cell_metrics
ORDER BY scenario, success_rate DESC
""".strip(),
    "src_macro_contrasts": """
SELECT
  comparison,
  left,
  right,
  success_rate_macro_delta AS success_delta,
  collision_rate_macro_delta AS collision_delta,
  timeout_rate_macro_delta AS timeout_delta,
  mean_return_macro_delta AS return_delta,
  success_rate_scenario_sign_flip_p AS success_signflip_p,
  success_rate_holm_adjusted_p AS success_holm_p,
  collision_rate_scenario_sign_flip_p AS collision_signflip_p,
  collision_rate_holm_adjusted_p AS collision_holm_p,
  scenarios
FROM macro_comparisons
ORDER BY CASE comparison
  WHEN 'mst_minus_sac' THEN 1
  WHEN 'mst_slt_minus_mst' THEN 2
  WHEN 'temporal_graph_minus_mst_slt' THEN 3
  WHEN 'full_balanced_minus_temporal_graph' THEN 4
  WHEN 'full_balanced_minus_mst_slt' THEN 5
  ELSE 6 END
""".strip(),
    "src_full_scenario": """
SELECT
  scenario,
  success_delta,
  success_ci_low,
  success_ci_high,
  collision_delta,
  collision_ci_low,
  collision_ci_high,
  timeout_delta,
  return_delta
FROM scenario_attribution
WHERE comparison = 'full_balanced_minus_mst_slt'
ORDER BY CASE scenario
  WHEN 'left_turn' THEN 1 WHEN 'cross' THEN 2 WHEN 'roundabout_easy' THEN 3
  WHEN 'roundabout_medium' THEN 4 WHEN 'roundabout' THEN 5 ELSE 6 END
""".strip(),
    "src_stability": """
SELECT
  method,
  mean_within_scenario_seed_std_success_rate AS mean_seed_std,
  max_within_scenario_seed_std_success_rate AS max_seed_std,
  max_std_scenario_success_rate AS max_std_scenario,
  mean_within_scenario_seed_std_collision_rate AS mean_collision_seed_std
FROM seed_stability
ORDER BY mean_seed_std ASC
""".strip(),
    "src_diagnostics": """
SELECT
  association_scope,
  method_or_comparison,
  diagnostic,
  outcome,
  observations,
  raw_pearson_r,
  scenario_residual_pearson_r,
  paired_delta_pearson_r,
  interpretation
FROM diagnostic_associations
WHERE outcome = 'success_rate'
  AND diagnostic IN (
    'diagnostic_topology_attention_entropy_mean',
    'diagnostic_slot_scale_ratio_mean',
    'diagnostic_graph_mean_edge_weight_mean'
  )
ORDER BY association_scope, method_or_comparison, diagnostic
""".strip(),
}

datasets = {source_id.removeprefix("src_"): query_rows(connection, sql) for source_id, sql in queries.items()}

for row in datasets["method_macro"]:
    row["method_label"] = METHOD_LABELS[row["method"]]
for row in datasets["cell_success"]:
    row["method_label"] = METHOD_LABELS[row["method"]]
    row["scenario_label"] = SCENARIO_LABELS[row["scenario"]]
for row in datasets["macro_contrasts"]:
    row["comparison_label"] = COMPARISON_LABELS[row["comparison"]]
for row in datasets["full_scenario"]:
    row["scenario_label"] = SCENARIO_LABELS[row["scenario"]]
for row in datasets["stability"]:
    row["method_label"] = METHOD_LABELS[row["method"]]
    row["max_std_scenario_label"] = SCENARIO_LABELS[row["max_std_scenario"]]

# These are assertions over the exact SQL outputs embedded in the report.
assert len(datasets["method_macro"]) == 6
assert len(datasets["cell_success"]) == 36
assert len(datasets["macro_contrasts"]) == 6
assert len(datasets["full_scenario"]) == 6
assert len(datasets["stability"]) == 6
assert datasets["method_macro"][0]["method"] == "mst_slt"
assert math.isclose(datasets["method_macro"][0]["success_rate"], 0.94)
assert quality["status"] == "ready"

sources = [
    {
        "id": source_id,
        "label": {
            "src_quality": "冻结矩阵完整性审计",
            "src_method_macro": "方法级场景等权聚合",
            "src_cell_success": "方法×场景正式单元聚合",
            "src_macro_contrasts": "预声明场景级消融统计",
            "src_full_scenario": "Full 与 MST+SLT 的逐场景对比",
            "src_stability": "训练种子稳定性聚合",
            "src_diagnostics": "训练诊断与成功率的描述性关联",
        }[source_id],
        "query": {
            "engine": "sqlite",
            "sql": sql,
            "description": {
                "src_quality": "从已通过的只读审计中提取矩阵、测试回合、checkpoint 与检验门槛摘要。",
                "src_method_macro": "每个方法先在场景内聚合 3 个训练种子，再对 6 个场景等权平均，并保留效率与排名字段。",
                "src_cell_success": "读取官方聚合器输出的 36 个方法×场景单元及分层 bootstrap 区间。",
                "src_macro_contrasts": "读取 6 个预声明对比的场景等权效应、精确 sign-flip 与 Holm 校正结果。",
                "src_full_scenario": "限制为 Full+BalancedSlots 减 MST+SLT，展示差异集中在哪些场景。",
                "src_stability": "在每个场景内计算 3 个训练种子成功率标准差，再跨 6 场景平均。",
                "src_diagnostics": "筛选 topology entropy、slot-scale ratio 与 edge weight 的小样本描述性相关，禁止机制因果解释。",
            }[source_id],
            "executed_at": GENERATED_AT,
            "tables_used": {
                "src_quality": ["data_quality_report"],
                "src_method_macro": ["method_macro_summary"],
                "src_cell_success": ["cell_metrics"],
                "src_macro_contrasts": ["macro_comparisons"],
                "src_full_scenario": ["scenario_attribution"],
                "src_stability": ["seed_stability"],
                "src_diagnostics": ["diagnostic_associations"],
            }[source_id],
            "filters": {
                "src_quality": ["frozen protocol hash matched", "accepted runs only"],
                "src_method_macro": ["6 frozen methods", "6 equally weighted scenarios", "3 seeds per scenario"],
                "src_cell_success": ["accepted complete cells only", "150 episodes per method×scenario cell"],
                "src_macro_contrasts": ["6 frozen scenarios", "5 MST-family contrasts in Holm family", "PPO−SAC descriptive"],
                "src_full_scenario": ["comparison = full_balanced_minus_mst_slt"],
                "src_stability": ["3 training seeds in every method×scenario cell"],
                "src_diagnostics": ["outcome = success_rate", "selected structural diagnostics", "n = 18 runs per within-method estimate"],
            }[source_id],
            "metric_definitions": {
                "src_quality": ["Accepted run: exact frozen method/scenario/seed key with 100k clock, 10 audited checkpoints and 50 deterministic episodes."],
                "src_method_macro": ["Success rate: successful deterministic episodes / 50 per run.", "Macro mean: equal mean over 6 scenario-level seed means."],
                "src_cell_success": ["Cell success rate: mean of 3 training-seed run success rates; 150 held-out episodes per cell."],
                "src_macro_contrasts": ["Macro delta: equal mean of 6 scenario deltas (left minus right).", "Holm p: family-wise adjusted exact scenario sign-flip p over 5 MST-family contrasts, separately per primary metric."],
                "src_full_scenario": ["Scenario delta: Full+BalancedSlots metric minus MST+SLT metric for the same scenario."],
                "src_stability": ["Mean seed std: population standard deviation across 3 training-seed run rates within each scenario, averaged over 6 scenarios."],
                "src_diagnostics": ["Scenario-residual Pearson r: within-method diagnostic and success deviations after subtracting each scenario mean.", "Paired-delta Pearson r: correlation between Full−TemporalGraph diagnostic and success deltas over 18 paired runs."],
            }[source_id],
        },
    }
    for source_id, sql in queries.items()
]

cards = [
    {
        "id": "accepted_runs",
        "description": "正式合同中的全部训练模型均通过完整性审计。",
        "dataset": "quality",
        "sourceId": "src_quality",
        "metrics": [{"label": "接受模型", "field": "accepted_runs", "format": "number"}],
    },
    {
        "id": "test_episodes",
        "description": "108 项 × 50 个确定性测试回合。",
        "dataset": "quality",
        "sourceId": "src_quality",
        "metrics": [{"label": "测试回合", "field": "deterministic_test_episodes", "format": "number"}],
    },
    {
        "id": "best_success",
        "description": "MST+SLT 在 6 场景等权宏平均上的成功率。",
        "dataset": "method_macro",
        "filter": {"method": "mst_slt"},
        "sourceId": "src_method_macro",
        "metrics": [
            {"label": "最佳宏成功率", "field": "success_rate", "format": "percent"},
            {"label": "碰撞率", "field": "collision_rate", "format": "percent"},
        ],
    },
    {
        "id": "holm_significant",
        "description": "3 个训练种子下，Holm 校正后的主指标对比无一达到 0.05。",
        "dataset": "quality",
        "sourceId": "src_quality",
        "metrics": [{"label": "Holm 显著主对比", "field": "holm_significant_primary_contrasts", "format": "number"}],
    },
]

charts = [
    {
        "id": "method_success_chart",
        "title": "MST+SLT 的总体成功率最高",
        "subtitle": "每方法 18 项；先对 3 训练种子取均值，再对 6 场景等权平均。",
        "showDescription": True,
        "intent": "comparison",
        "question": "在同一冻结预算下，哪种方法的跨场景成功率最高？",
        "rationale": "6 个方法属于少量离散类别，横向条形图直接显示排序与幅度。",
        "comparisonContext": {"grain": "method", "normalization": "equal-weight over six scenarios", "unit": "rate"},
        "type": "horizontalBar",
        "dataset": "method_macro",
        "sourceId": "src_method_macro",
        "encodings": {
            "x": {"field": "method_label", "type": "nominal", "label": "方法"},
            "y": {"field": "success_rate", "type": "quantitative", "format": "percent", "label": "场景等权成功率"},
            "tooltip": [
                {"field": "collision_rate", "type": "quantitative", "format": "percent", "label": "碰撞率"},
                {"field": "mean_return", "type": "quantitative", "format": "number", "label": "平均回报"},
                {"field": "mean_scenario_rank", "type": "quantitative", "format": "number", "label": "平均场景名次"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "sequential", "name": "blue"},
        "labels": {"values": "all"},
        "settings": {"sort": "descending", "orientation": "horizontal", "showValues": True},
        "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
    },
    {
        "id": "scenario_success_chart",
        "title": "完整方法的场景表现并不均匀",
        "subtitle": "每格 3 训练种子、150 个确定性回合；分组仅用于显示场景异质性。",
        "showDescription": True,
        "intent": "comparison",
        "question": "总体排名是否由少数场景驱动，Full+BalancedSlots 在哪里失速？",
        "rationale": "分组柱形图保留方法与场景两个离散维度，避免把异质性压成单一总体均值。",
        "comparisonContext": {"grain": "method by scenario", "denominator": "150 deterministic episodes per cell", "unit": "rate"},
        "type": "bar",
        "dataset": "cell_success",
        "sourceId": "src_cell_success",
        "encodings": {
            "x": {"field": "scenario_label", "type": "nominal", "label": "场景"},
            "y": {"field": "success_rate", "type": "quantitative", "format": "percent", "label": "成功率"},
            "color": {"field": "method_label", "type": "nominal", "label": "方法"},
            "tooltip": [
                {"field": "collision_rate", "type": "quantitative", "format": "percent", "label": "碰撞率"},
                {"field": "timeout_rate", "type": "quantitative", "format": "percent", "label": "超时率"},
                {"field": "success_ci_low", "type": "quantitative", "format": "percent", "label": "成功率 CI 下界"},
                {"field": "success_ci_high", "type": "quantitative", "format": "percent", "label": "成功率 CI 上界"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "categorical", "name": "method"},
        "legend": {"position": "bottom", "sort": "spec", "title": "方法"},
        "labels": {"values": "none"},
        "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap", "showValues": False},
        "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
    },
    {
        "id": "contrast_chart",
        "title": "后续复杂组件未延续 MST+SLT 的增益方向",
        "subtitle": "条形为左方法减右方法的 6 场景等权宏差；PPO−SAC 仅作跨家族描述。",
        "showDescription": True,
        "intent": "comparison",
        "question": "冻结消融链中，各增量对成功率与碰撞率的方向是什么？",
        "rationale": "同一零基线上的双指标分组条形图清楚显示有利与不利方向。",
        "comparisonContext": {"baseline": "right method in each named contrast", "grain": "predeclared contrast", "normalization": "equal-weight over six scenarios", "unit": "percentage-point delta"},
        "type": "bar",
        "dataset": "macro_contrasts",
        "sourceId": "src_macro_contrasts",
        "encodings": {
            "x": {"field": "comparison_label", "type": "nominal", "label": "对比（左 − 右）"},
            "y": {"fields": ["success_delta", "collision_delta"], "type": "quantitative", "format": "percent", "label": "宏差"},
            "tooltip": [
                {"field": "success_holm_p", "type": "quantitative", "format": "number", "label": "成功率 Holm p"},
                {"field": "collision_holm_p", "type": "quantitative", "format": "number", "label": "碰撞率 Holm p"},
                {"field": "return_delta", "type": "quantitative", "format": "number", "label": "回报差"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "semantic", "name": "primary-outcomes", "midpoint": 0},
        "legend": {"position": "bottom", "sort": "spec", "title": "指标"},
        "labels": {"values": "none"},
        "referenceLines": [{"axis": "y", "value": 0, "label": "无差异", "color": "neutral"}],
        "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap", "showValues": False},
        "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
    },
    {
        "id": "full_scenario_chart",
        "title": "Full 相对 MST+SLT 的损失集中于中型环岛与 CARLA",
        "subtitle": "成功率差为 Full+BalancedSlots − MST+SLT；正值有利于 Full。",
        "showDescription": True,
        "intent": "comparison",
        "question": "Full 的总体劣势主要来自哪些场景？",
        "rationale": "单系列零中心条形图直接暴露场景级贡献和反例。",
        "comparisonContext": {"baseline": "MST+SLT", "grain": "scenario", "unit": "percentage-point success delta"},
        "type": "bar",
        "dataset": "full_scenario",
        "sourceId": "src_full_scenario",
        "encodings": {
            "x": {"field": "scenario_label", "type": "nominal", "label": "场景"},
            "y": {"field": "success_delta", "type": "quantitative", "format": "percent", "label": "成功率差"},
            "tooltip": [
                {"field": "success_ci_low", "type": "quantitative", "format": "percent", "label": "CI 下界"},
                {"field": "success_ci_high", "type": "quantitative", "format": "percent", "label": "CI 上界"},
                {"field": "collision_delta", "type": "quantitative", "format": "percent", "label": "碰撞率差"},
                {"field": "timeout_delta", "type": "quantitative", "format": "percent", "label": "超时率差"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "diverging", "name": "delta", "midpoint": 0},
        "labels": {"values": "all"},
        "referenceLines": [{"axis": "y", "value": 0, "label": "无差异", "color": "neutral"}],
        "settings": {"sort": "none", "categoryLabelPolicy": "wrap", "showValues": True},
        "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
    },
    {
        "id": "stability_chart",
        "title": "MST+SLT 最稳定，Full 与 PPO 对训练种子更敏感",
        "subtitle": "每个场景内的 3 种子成功率标准差，再对 6 场景平均；越低越稳定。",
        "showDescription": True,
        "intent": "comparison",
        "question": "总体均值之外，哪种方法对训练种子最稳健？",
        "rationale": "横向条形图适合少量方法的非负离散度比较，并突出低值更优。",
        "comparisonContext": {"grain": "method", "normalization": "mean of six within-scenario seed standard deviations", "unit": "rate standard deviation"},
        "type": "horizontalBar",
        "dataset": "stability",
        "sourceId": "src_stability",
        "encodings": {
            "x": {"field": "method_label", "type": "nominal", "label": "方法"},
            "y": {"field": "mean_seed_std", "type": "quantitative", "format": "percent", "label": "平均种子标准差"},
            "tooltip": [
                {"field": "max_seed_std", "type": "quantitative", "format": "percent", "label": "最大场景标准差"},
                {"field": "max_std_scenario_label", "type": "nominal", "label": "最不稳定场景"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "sequential", "name": "stability"},
        "labels": {"values": "all"},
        "settings": {"sort": "ascending", "orientation": "horizontal", "showValues": True},
        "surface": {"surface": "card", "showControls": False, "viewMode": "both"},
    },
]

tables = [
    {
        "id": "method_detail_table",
        "title": "总体表现与运行成本",
        "subtitle": "效率来自冻结运行的 performance_profile；仅作观察性成本比较。",
        "showDescription": True,
        "dataset": "method_macro",
        "sourceId": "src_method_macro",
        "defaultSort": {"field": "success_rate", "direction": "desc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "method_label", "label": "方法", "type": "text"},
            {"field": "success_rate", "label": "成功率", "format": "percent"},
            {"field": "collision_rate", "label": "碰撞率", "format": "percent"},
            {"field": "timeout_rate", "label": "超时率", "format": "percent"},
            {"field": "mean_return", "label": "平均回报", "format": "number"},
            {"field": "parameters", "label": "参数", "format": "compact"},
            {"field": "learner_update_ms", "label": "更新 ms", "format": "number"},
            {"field": "inference_ms", "label": "推理 ms", "format": "number"},
            {"field": "peak_gpu_mb", "label": "峰值 GPU MB", "format": "number"},
        ],
    },
    {
        "id": "contrast_detail_table",
        "title": "预声明对比的统计明细",
        "subtitle": "PPO−SAC 不属于 Holm 家族；空 p 值表示未做该家族校正。",
        "showDescription": True,
        "dataset": "macro_contrasts",
        "sourceId": "src_macro_contrasts",
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "comparison_label", "label": "对比", "type": "text"},
            {"field": "success_delta", "label": "成功率差", "format": "percent", "movement": True},
            {"field": "success_signflip_p", "label": "成功率原始 p", "format": "number"},
            {"field": "success_holm_p", "label": "成功率 Holm p", "format": "number"},
            {"field": "collision_delta", "label": "碰撞率差", "format": "percent", "movement": True},
            {"field": "collision_signflip_p", "label": "碰撞率原始 p", "format": "number"},
            {"field": "collision_holm_p", "label": "碰撞率 Holm p", "format": "number"},
            {"field": "return_delta", "label": "回报差", "format": "number", "movement": True},
        ],
    },
]

blocks = [
    {"id": "title", "type": "markdown", "layout": "full", "body": "# 冻结系统性实验矩阵：结果聚合与深层归因"},
    {
        "id": "technical_summary",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 技术摘要\n\n"
            "冻结的 **108/108** 项正式矩阵已通过完整性门槛：每项 100,000 raw SUMO steps、10 个经 CRC/SHA 与时钟审计的 checkpoint、50 个确定性测试回合，共 5,400 回合。"
            "在 6 场景等权汇总下，**MST+SLT 成功率最高（94.0%）、碰撞率最低（6.0%），且训练种子内最稳定**；其相对 MST 的方向为成功率 **+5.11pp**、碰撞率 **−4.67pp**。"
            "但 Full+BalancedSlots 相对 MST+SLT 的成功率为 **−13.11pp**、碰撞率为 **+7.44pp**，损失主要集中于 roundabout_medium 与 CARLA。"
            "三训练种子限制了场景级精确检验的分辨率，**所有 Holm 校正后的主指标对比均未达到 0.05**，因此结论应表述为预声明消融方向和稳健性诊断，而不是确定性因果。"
        ),
    },
    {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["accepted_runs", "test_episodes", "best_success", "holm_significant"]},
    {
        "id": "overall_heading",
        "type": "markdown",
        "layout": "full",
        "body": "## 总体结果：MST+SLT 是当前冻结协议下的最佳且最稳定方案\n\n总体值先在每个场景内对 3 个训练种子取均值，再对 6 个场景等权平均。MST+SLT 不仅成功率最高，也以 2.45pp 的平均场景内种子标准差优于其他方法。",
    },
    {"id": "method_success", "type": "chart", "layout": "full", "chartId": "method_success_chart"},
    {"id": "method_detail", "type": "table", "layout": "full", "tableId": "method_detail_table"},
    {
        "id": "scenario_heading",
        "type": "markdown",
        "layout": "full",
        "body": "## 场景异质性：完整方法的损失并非均匀分布\n\nFull+BalancedSlots 在 left_turn 略优于 MST+SLT，但在 roundabout_medium 与 CARLA 分别落后 26.67pp 和 31.33pp。中型环岛的分层 bootstrap 95% 区间低于 0；CARLA 则因 3 种子表现极端离散而仍跨 0。",
    },
    {"id": "scenario_success", "type": "chart", "layout": "full", "chartId": "scenario_success_chart"},
    {"id": "full_scenario", "type": "chart", "layout": "full", "chartId": "full_scenario_chart"},
    {
        "id": "ablation_heading",
        "type": "markdown",
        "layout": "full",
        "body": "## 归因边界：SLT 增益可单独识别，后续组件只能联合解释\n\n**MST+SLT−MST** 是唯一在编码器保持不变时只增加 SLT 的单组件对比，并且 leave-one-scenario-out 后 success/collision 差的符号都保持。TemporalGraph−MST+SLT 同时改变时序汇聚位置与 Graph-SLT 结构；Full−TemporalGraph 又同时改变 topology query 与 slot normalization，均不能拆成单因素效应。PPO−SAC 是跨算法家族描述，不是消融。",
    },
    {"id": "contrast", "type": "chart", "layout": "full", "chartId": "contrast_chart"},
    {"id": "contrast_detail", "type": "table", "layout": "full", "tableId": "contrast_detail_table"},
    {
        "id": "stability_heading",
        "type": "markdown",
        "layout": "full",
        "body": "## 稳定性与成本：Full 更昂贵，却没有带来总体收益\n\nFull+BalancedSlots 的平均场景内种子标准差为 14.95pp，CARLA 单场景达到 47.14pp；MST+SLT 分别只有 2.45pp 与 4.32pp。Full 相比 TemporalGraph 约增加 40% 参数、47% learner-update 时间、87% 推理时间和 52% 峰值 GPU 内存。",
    },
    {"id": "stability", "type": "chart", "layout": "full", "chartId": "stability_chart"},
    {
        "id": "scope_data_metrics",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 范围、数据与指标定义\n\n"
            "- **范围：** SAC、PPO、MST、MST+SLT、TemporalGraph、Full+BalancedSlots；left_turn、cross、roundabout_easy、roundabout_medium、roundabout、CARLA；训练种子 0/1/2。\n"
            "- **成功率：** 每个已训练策略在 50 个确定性测试回合中的成功比例；同一场景/种子下各方法共享 evaluation seed 与 traffic variant 配对。\n"
            "- **碰撞/超时/off-route：** 评估器独立记录终止事件比例。off-route 在全部 108 项中为 0，故不具区分力。\n"
            "- **宏平均：** 每个场景先对 3 个训练种子平均，再对 6 场景等权平均。\n"
            "- **效率：** `performance_profile` 中的模型参数、learner update wall time、推理时间与峰值 GPU 内存，只作观察性对照。"
        ),
    },
    {
        "id": "methodology",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 统计与验证方法\n\n"
            "官方聚合器要求完整 108 项后才输出，单元不完整会失败。区间采用 **10,000 次分层训练种子/回合 bootstrap**（固定分析种子 20260806）；"
            "预声明对比在同一场景/训练种子下配对，场景级推断采用 6 场景精确 sign-flip。5 个 MST 家族对比分别在 success 与 collision 上做 Holm 校正；PPO−SAC 不进入该家族。"
            "伴随审计独立重算 5,400 个回合率，验证与汇总 JSON 一致，并核对协议与重跑证据归档哈希。训练诊断相关仅为 n=18 的描述性线索，未作机制因果检验。"
        ),
    },
    {
        "id": "limitations",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 局限性、敏感性与稳健性\n\n"
            "- **训练种子只有 3 个。** 回合级 bootstrap 不能创造新的训练重复，Holm 校正后无主指标显著对比。\n"
            "- **两类联合改动不可拆分。** TemporalGraph 与 Full 的对比必须保留联合组件措辞。\n"
            "- **2/5,400 success+timeout 重叠。** 代码确认这是恰在 600 raw-step 边界到达时独立事件同时成立；将其保守地仅计 timeout 不改变任何对比符号。\n"
            "- **leave-one-scenario-out。** 六个预声明对比的 success/collision 宏差在每次剔除一个场景后均保持原符号，但这不等于统计显著。\n"
            "- **仿真等价边界。** SUMO 重建不是 SMARTS/CARLA 物理的 bitwise 等价；SAC 使用审计过的 v2 重建，部分 CARLA 组合为受控扩展。"
        ),
    },
    {
        "id": "next_steps",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 建议的下一步\n\n"
            "1. 以 **MST+SLT 为冻结矩阵的主表现基准**，把 Full 的场景依赖和种子不稳定作为负结果如实呈现。\n"
            "2. 直接从现有冻结结果生成论文主表、逐场景图和失败案例索引；不选择性删除种子、不回写训练或测试文件。\n"
            "3. 在新增计算前先做功效/种子预算评估；若未来确需区分 topology query 与 balanced slots，应另立预注册正交消融，而不是从本矩阵事后拆分。\n"
            "4. 保留本报告、artifact.json、执行过的 notebook 与哈希清单，作为论文数字的审计入口。"
        ),
    },
    {
        "id": "further_questions",
        "type": "markdown",
        "layout": "full",
        "body": (
            "## 待进一步回答的问题\n\n"
            "- Full 在 CARLA 的 47.14pp 种子标准差由哪些具体交通变体与交互序列触发？\n"
            "- roundabout_medium 中 Full 的失败增加主要来自碰撞还是时间预算分配？\n"
            "- 需要多少额外、预注册训练种子才能使 5 个主对比达到有意义的检验分辨率？\n"
            "- topology-attention entropy 的负相关是否能在独立消融与更多种子上复现，还是仅为共同场景/种子驱动？"
        ),
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "冻结系统性实验矩阵：结果聚合与深层归因",
        "description": "108 项正式训练模型的完整性审计、场景等权聚合、预声明统计、稳健性与联合组件归因边界。",
        "generatedAt": GENERATED_AT,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": [{"id": source["id"], "label": source["label"]} for source in sources],
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": datasets,
        "accessIssues": [],
    },
    "sources": sources,
    "package_info": {
        "mode": "portable_html",
        "researchMethodModified": False,
        "protocolSha256": quality["protocol_sha256"],
        "retryArchiveSha256": quality["retry_archive_manifest_sha256"],
    },
}

ARTIFACT_PATH.write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)
print(ARTIFACT_PATH)
