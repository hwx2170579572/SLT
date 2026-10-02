"""Build the canonical artifact for the repaired SAC/PPO reproduction report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCENARIO_LABELS = {
    "left_turn": "未保护左转",
    "cross": "双汇入",
    "roundabout_easy": "环岛 A",
    "roundabout_medium": "环岛 B",
    "roundabout": "环岛 C",
}
METHOD_LABELS = {"sac": "RLEncoder(GRU)-SAC", "ppo": "PPO"}
PROTOCOL_LABELS = {
    "frozen_80_20": "frozen_80_20",
    "source_all": "source_all",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(
    source_id: str,
    label: str,
    *,
    description: str,
    tables_used: list[str],
    generated_at: str,
    metric_definitions: list[str] | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "engine": "local validated artifact audit",
        "id": source_id,
        "description": description,
        "executed_at": generated_at,
        "tables_used": tables_used,
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    source: dict[str, Any] = {"id": source_id, "label": label, "query": query}
    if path is not None:
        source["path"] = path
    return source


def _card(card_id: str, description: str, field: str, label: str) -> dict[str, Any]:
    return {
        "id": card_id,
        "description": description,
        "dataset": "matrix_summary",
        "sourceId": "dual_summary_source",
        "metrics": [{"label": label, "field": field, "format": "number"}],
    }


def _active_jobs(result_root: Path) -> list[str]:
    state_path = result_root / "orchestrator_state.json"
    if not state_path.is_file():
        return []
    state = _read_json(state_path)
    if state.get("status") != "running":
        return []
    return [str(item) for item in state.get("running_jobs", [])]


def build_artifact(
    *,
    summary_path: Path,
    protocol_path: Path,
    result_root: Path,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    protocol = _read_json(protocol_path)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    quality = list(summary["quality"])
    runs = list(summary["runs"])
    aggregate = list(summary["aggregate"])

    expected_matrix = summary["expected_matrix"]
    expected_runs = (
        len(expected_matrix["methods"])
        * len(expected_matrix["scenarios"])
        * len(expected_matrix["seeds"])
    )
    expected_protocol_evaluations = expected_runs * len(summary["protocols"])
    status_counts = Counter(str(row["status"]) for row in quality)
    valid_evaluations = status_counts["valid"]
    invalid_evaluations = sum(
        count
        for status, count in status_counts.items()
        if status in {"invalid", "identity_mismatch", "missing_arguments"}
    )
    valid_protocols_by_run: dict[str, set[str]] = defaultdict(set)
    for row in quality:
        if row["status"] == "valid":
            valid_protocols_by_run[str(row["run"])].add(str(row["protocol"]))
    dual_valid_runs = sum(
        protocols == set(summary["protocols"])
        for protocols in valid_protocols_by_run.values()
    )
    active_jobs = _active_jobs(result_root)
    remaining_runs = expected_runs - dual_valid_runs
    artifact_status = (
        "ready"
        if dual_valid_runs == expected_runs
        and valid_evaluations == expected_protocol_evaluations
        and invalid_evaluations == 0
        else "partial"
    )

    matrix_summary = [
        {
            "expected_seed_runs": expected_runs,
            "dual_protocol_valid_runs": dual_valid_runs,
            "valid_protocol_evaluations": valid_evaluations,
            "invalid_protocol_evaluations": invalid_evaluations,
            "remaining_seed_runs": remaining_runs,
            "active_training_jobs": len(active_jobs),
        }
    ]

    aggregate_rows: list[dict[str, Any]] = []
    for order, row in enumerate(aggregate, start=1):
        enriched = dict(row)
        enriched.update(
            {
                "order": order,
                "method_label": METHOD_LABELS[row["method"]],
                "scenario_label": SCENARIO_LABELS[row["scenario"]],
                "protocol_label": PROTOCOL_LABELS[row["traffic_protocol"]],
                "case": (
                    f"{METHOD_LABELS[row['method']]} · "
                    f"{SCENARIO_LABELS[row['scenario']]}"
                ),
                "seed_coverage": (
                    f"{row['seeds_completed']}/{len(expected_matrix['seeds'])}"
                ),
            }
        )
        aggregate_rows.append(enriched)

    per_seed_rows: list[dict[str, Any]] = []
    for order, row in enumerate(
        sorted(
            runs,
            key=lambda item: (
                item["traffic_protocol"],
                item["method"],
                item["scenario"],
                item["seed"],
            ),
        ),
        start=1,
    ):
        enriched = dict(row)
        enriched.update(
            {
                "order": order,
                "method_label": METHOD_LABELS[row["method"]],
                "scenario_label": SCENARIO_LABELS[row["scenario"]],
                "protocol_label": PROTOCOL_LABELS[row["traffic_protocol"]],
            }
        )
        per_seed_rows.append(enriched)

    quality_rows = [
        {
            "order": index,
            "status": status,
            "protocol_slots": count,
            "usable": "是" if status == "valid" else "否",
            "interpretation": {
                "valid": "通过场景、交通分区、逐回合 traffic_variant 与模型/环境空间契约",
                "missing": "训练目录存在，但该协议评估尚未生成",
                "missing_run": "该算法×场景×seed 训练任务尚未形成目录",
                "invalid": "评估文件存在，但未通过严格 provenance/space 验收",
                "identity_mismatch": "目录标识与 arguments.json 不一致",
                "missing_arguments": "训练目录缺少 arguments.json",
            }.get(status, "需要人工核对"),
        }
        for index, (status, count) in enumerate(sorted(status_counts.items()), start=1)
    ]

    contract_rows = [
        {
            "order": 1,
            "check": "最终环境工厂",
            "acceptance": "周期评估与最终评估均由注入的 env_factory(evaluation=True) 构造",
            "enforcement": "统一工厂调用 + 回归测试",
        },
        {
            "order": 2,
            "check": "环境类",
            "acceptance": "SAC=PaperSumoSceneEnv；SMARTS PPO=PaperPpoRgbEnv",
            "enforcement": "expected_environment_class 精确匹配",
        },
        {
            "order": 3,
            "check": "observation/action space",
            "acceptance": "SB3 模型空间与环境空间对象及签名完全相同",
            "enforcement": "推理前 fail-fast + 签名持久化",
        },
        {
            "order": 4,
            "check": "交通协议",
            "acceptance": "frozen=evaluation 分区；source-all=all 分区",
            "enforcement": "provenance.traffic_partition 精确匹配",
        },
        {
            "order": 5,
            "check": "发布资产",
            "acceptance": "uses_released_assets=true，且每回合 traffic_variant 非空",
            "enforcement": "逐回合记录与集合重算",
        },
        {
            "order": 6,
            "check": "确定性 seed",
            "acceptance": "评估 seed=训练 seed+10000，并连续覆盖 50 回合",
            "enforcement": "逐回合 seed 序列精确核对",
        },
        {
            "order": 7,
            "check": "成功率",
            "acceptance": "success_rate=成功回合数/50，且由逐回合记录重算一致",
            "enforcement": "绝对误差≤1e-12",
        },
        {
            "order": 8,
            "check": "不重训重评",
            "acceptance": "模型/检查点哈希通过训练完整性审计，收据声明未恢复训练",
            "enforcement": "SHA-256 + checkpoint clock/CRC 审计",
        },
    ]

    active_rows = [
        {"order": index, "run": run, "state": "running"}
        for index, run in enumerate(active_jobs, start=1)
    ] or [{"order": 1, "run": "无", "state": "idle"}]

    sources = [
        _source(
            "dual_summary_source",
            "双协议严格验收汇总",
            description=(
                "仅聚合通过 saved_evaluation_contract_errors 的 frozen_80_20 与 "
                "source_all 逐 seed 评估；缺失 seed 不以零填充。"
            ),
            tables_used=[
                "results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/summary_dual_protocol/summary.json",
                "results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/summary_dual_protocol/quality.csv",
            ],
            generated_at=generated_at,
            metric_definitions=[
                "per-seed success_rate = successful episodes / 50",
                "scenario mean = arithmetic mean of available seed success rates",
                "std = population standard deviation across available seed point estimates (ddof=0)",
                "success/collision/off-route/timeout are independently recomputed event flags; "
                "a boundary episode may be both successful and time-limited, so the four rates "
                "must not be interpreted as a mutually exclusive partition",
            ],
            path="pytorch_sb3_sumo/reports/baseline_reproduction_fixed_20260804/dual_summary_source.sql",
        ),
        _source(
            "contract_source",
            "最终环境工厂与 provenance/space 验收代码",
            description=(
                "审计统一环境构造、模型/环境空间精确匹配、交通分区、发布资产、"
                "逐回合 seed/traffic_variant 与汇总率可重算性。"
            ),
            tables_used=[
                "pytorch_sb3_sumo/tools/train_sb3.py",
                "pytorch_sb3_sumo/tools/paper_evaluation_contract.py",
                "pytorch_sb3_sumo/tools/reevaluate_paper_run.py",
                "pytorch_sb3_sumo/tests_sb3_sumo/test_paper_evaluation_contract.py",
            ],
            generated_at=generated_at,
            path="pytorch_sb3_sumo/reports/baseline_reproduction_fixed_20260804/contract_source.sql",
        ),
        _source(
            "paper_protocol_source",
            "论文参考值与 v14 复现实验协议",
            description=(
                "协议快照记录论文 100k 训练步、5 seeds、每策略 50 测试回合及 Table I "
                "成功率；同时显式记录源码与论文配置冲突。"
            ),
            tables_used=[
                "pytorch_sb3_sumo/experiments/sb3_sumo_paper/protocol_baselines_v14.json"
            ],
            generated_at=generated_at,
        ),
    ]

    missing_notice = (
        f"当前仅 {dual_valid_runs}/{expected_runs} 个训练 seed 同时具备两套有效评估；"
        f"仍缺 {remaining_runs} 个 seed 任务。所有均值±标准差只覆盖表中列出的已完成 seeds。"
    )
    title = "修复后 SAC/PPO 五场景复现与双协议验收报告"
    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "最终环境工厂修复、不重训重评、provenance/observation-space 验收，"
            "以及 frozen/source-all 分协议的逐 seed 与均值±标准差。"
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            _card(
                "expected-runs",
                "2 算法 × 5 场景 × 5 训练 seeds",
                "expected_seed_runs",
                "计划训练 seed",
            ),
            _card(
                "dual-valid-runs",
                "frozen 与 source-all 均通过严格验收的训练 seed",
                "dual_protocol_valid_runs",
                "双协议有效 seed",
            ),
            _card(
                "valid-evaluations",
                "通过 provenance、traffic_variant 与空间契约的协议评估",
                "valid_protocol_evaluations",
                "有效协议评估",
            ),
            _card(
                "invalid-evaluations",
                "文件存在但违反硬验收契约的协议评估",
                "invalid_protocol_evaluations",
                "无效协议评估",
            ),
            _card(
                "remaining-runs",
                "尚未同时形成两套有效评估的训练 seed",
                "remaining_seed_runs",
                "待补齐 seed",
            ),
            _card(
                "active-jobs",
                "报告生成时编排器仍在执行的训练任务",
                "active_training_jobs",
                "正在训练",
            ),
        ],
        "charts": [
            {
                "id": "success-by-protocol",
                "title": "各算法与场景的平均成功率",
                "subtitle": (
                    "frozen_80_20 与 source_all 分组；每条为当前已完成 seed 的等权均值，"
                    "seed 数见表格，缺失 seed 未补零。"
                ),
                "intent": "comparison",
                "question": "交通协议、算法与场景之间的成功率差异有多大？",
                "rationale": "长场景标签用横向分组条形图便于精确比较两套协议。",
                "comparisonContext": {
                    "denominator": "每 seed 50 个确定性评估回合",
                    "grain": "协议 × 算法 × 场景",
                    "normalization": "seed 成功率的算术平均",
                    "unit": "%",
                },
                "type": "horizontalBar",
                "dataset": "aggregate_results",
                "sourceId": "dual_summary_source",
                "encodings": {
                    "x": {"field": "case", "type": "ordinal", "label": "算法与场景"},
                    "y": {
                        "field": "success_rate_mean",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "成功率均值",
                    },
                    "color": {
                        "field": "protocol_label",
                        "type": "nominal",
                        "label": "交通协议",
                    },
                    "tooltip": [
                        {"field": "protocol_label", "type": "nominal", "label": "协议"},
                        {"field": "success_rate_mean", "type": "quantitative", "format": "percent", "label": "均值"},
                        {"field": "success_rate_std", "type": "quantitative", "format": "percent", "label": "标准差"},
                        {"field": "seeds_completed", "type": "quantitative", "format": "number", "label": "已完成 seed"},
                        {"field": "paper_success_rate", "type": "quantitative", "format": "percent", "label": "论文参考"},
                    ],
                },
                "xAxisTitle": "算法与场景",
                "yAxisTitle": "成功率均值",
                "valueFormat": "percent",
                "layout": "full",
                "maxRows": 40,
                "palette": {"kind": "categorical", "name": "blue-gold"},
                "labels": {"values": "all"},
                "settings": {
                    "orientation": "horizontal",
                    "showValues": True,
                    "categoryLabelPolicy": "wrap",
                },
            },
            {
                "id": "quality-slots",
                "title": "双协议评估槽位完整性",
                "subtitle": (
                    f"预期 {expected_protocol_evaluations} 个协议评估槽位；"
                    "valid 才进入结果汇总。"
                ),
                "intent": "comparison",
                "question": "当前矩阵中有效、缺失或无效的协议评估各有多少？",
                "rationale": "少量互斥质量状态用横向条形图最易核对。",
                "type": "horizontalBar",
                "dataset": "quality_counts",
                "sourceId": "dual_summary_source",
                "encodings": {
                    "x": {"field": "status", "type": "ordinal", "label": "验收状态"},
                    "y": {"field": "protocol_slots", "type": "quantitative", "format": "number", "label": "协议槽位"},
                    "tooltip": [
                        {"field": "protocol_slots", "type": "quantitative", "format": "number", "label": "槽位"},
                        {"field": "interpretation", "type": "text", "label": "判定"},
                    ],
                },
                "xAxisTitle": "验收状态",
                "yAxisTitle": "协议槽位",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 10,
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
            },
        ],
        "tables": [
            {
                "id": "aggregate-table",
                "title": "frozen/source-all：场景均值±标准差",
                "subtitle": "标准差为已完成训练 seeds 点估计的总体标准差（ddof=0）；论文值仅作不同选择口径下的参考。",
                "dataset": "aggregate_results",
                "sourceId": "dual_summary_source",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "序", "format": "number"},
                    {"field": "protocol_label", "label": "协议", "type": "text"},
                    {"field": "method_label", "label": "算法", "type": "text"},
                    {"field": "scenario_label", "label": "场景", "type": "text"},
                    {"field": "seeds", "label": "已完成 seeds", "type": "text"},
                    {"field": "seed_coverage", "label": "覆盖", "type": "text"},
                    {"field": "success_rate_mean", "label": "成功率均值", "format": "percent"},
                    {"field": "success_rate_std", "label": "成功率标准差", "format": "percent"},
                    {"field": "collision_rate_mean", "label": "碰撞率均值", "format": "percent"},
                    {"field": "timeout_rate_mean", "label": "超时率均值", "format": "percent"},
                    {"field": "paper_success_rate", "label": "论文成功率", "format": "percent"},
                    {"field": "success_rate_delta_from_paper", "label": "相对论文差值", "format": "percent", "movement": True},
                ],
            },
            {
                "id": "per-seed-table",
                "title": "frozen/source-all：逐 seed 结果与运行时 provenance",
                "subtitle": "每行 50 回合；只有通过环境类、交通分区、逐回合变体和 observation/action-space 硬验收的记录才出现。",
                "dataset": "per_seed_results",
                "sourceId": "dual_summary_source",
                "density": "compact",
                "layout": "full",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "序", "format": "number"},
                    {"field": "protocol_label", "label": "协议", "type": "text"},
                    {"field": "method_label", "label": "算法", "type": "text"},
                    {"field": "scenario_label", "label": "场景", "type": "text"},
                    {"field": "seed", "label": "seed", "format": "number"},
                    {"field": "success_rate", "label": "成功率", "format": "percent"},
                    {"field": "collision_rate", "label": "碰撞率", "format": "percent"},
                    {"field": "off_route_rate", "label": "驶离率", "format": "percent"},
                    {"field": "timeout_rate", "label": "超时率", "format": "percent"},
                    {"field": "environment_class", "label": "环境类", "type": "text"},
                    {"field": "observation_space", "label": "observation space", "type": "text"},
                    {"field": "traffic_variants_observed", "label": "交通变体数", "format": "number"},
                    {"field": "run", "label": "运行 ID", "type": "text"},
                ],
            },
            {
                "id": "quality-table",
                "title": "预期矩阵的协议槽位质量",
                "subtitle": "缺失与无效均不进入均值；missing_run 表示整个训练 seed 尚未形成。",
                "dataset": "quality_counts",
                "sourceId": "dual_summary_source",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "序", "format": "number"},
                    {"field": "status", "label": "状态", "type": "text"},
                    {"field": "protocol_slots", "label": "协议槽位", "format": "number"},
                    {"field": "usable", "label": "进入汇总", "type": "text"},
                    {"field": "interpretation", "label": "判定口径", "type": "text"},
                ],
            },
            {
                "id": "contract-table",
                "title": "最终评估 provenance/space 硬验收契约",
                "subtitle": "任一硬条件失败，编排器不得把该运行标记为可报告 complete。",
                "dataset": "contract_checks",
                "sourceId": "contract_source",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "序", "format": "number"},
                    {"field": "check", "label": "检查项", "type": "text"},
                    {"field": "acceptance", "label": "验收条件", "type": "text"},
                    {"field": "enforcement", "label": "执行方式", "type": "text"},
                ],
            },
            {
                "id": "active-table",
                "title": "报告生成时的训练编排状态",
                "subtitle": "不同场景、不同算法优先；同一 seed 波次使用 diversified 顺序。",
                "dataset": "active_jobs",
                "sourceId": "dual_summary_source",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "序", "format": "number"},
                    {"field": "run", "label": "运行 ID", "type": "text"},
                    {"field": "state", "label": "状态", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "layout": "full",
                "body": f"# {title}",
            },
            {
                "id": "summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": "dual_summary_source",
                "body": (
                    "## 结论先行\n\n"
                    "最终评估已改为统一调用场景专用环境工厂，并在推理前执行模型/环境 observation 与 action space "
                    "精确匹配。先前受错误最终环境影响的 5 个任务已用原模型不重训重评；旧模型与检查点哈希保留在评估收据中。"
                    f"当前严格账本包含 **{valid_evaluations}/{expected_protocol_evaluations}** 个有效协议评估、"
                    f"**{invalid_evaluations}** 个无效评估；{missing_notice}"
                ),
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "layout": "full",
                "cardIds": [
                    "expected-runs",
                    "dual-valid-runs",
                    "valid-evaluations",
                    "invalid-evaluations",
                    "remaining-runs",
                    "active-jobs",
                ],
            },
            {
                "id": "coverage",
                "type": "markdown",
                "layout": "full",
                "sourceId": "dual_summary_source",
                "body": (
                    "## 结果覆盖与可解释边界\n\n"
                    f"{missing_notice} frozen_80_20 是对未公开 held-out 流量的确定性 80/20 重建，"
                    "source_all 则遍历发布源码可用的全部流量；两者必须分栏报告，且都不能冒充论文未发布的精确测试集。"
                ),
            },
            {"id": "success-chart", "type": "chart", "layout": "full", "chartId": "success-by-protocol"},
            {"id": "aggregate-results", "type": "table", "layout": "full", "tableId": "aggregate-table"},
            {
                "id": "seed-results-intro",
                "type": "markdown",
                "layout": "full",
                "sourceId": "dual_summary_source",
                "body": (
                    "## 逐 seed 结果\n\n"
                    "下表保留精确 seed、四类事件率、环境类、observation-space 摘要、交通变体数与运行 ID。"
                    "成功/碰撞/驶离/超时按事件标志分别重算；恰在步数上限到达目标的边界回合可同时标记成功与超时，"
                    "因此四项不能强制相加为 100%。"
                    "均值表中的 seeds 字段是唯一参与聚合的成员清单，可直接反查，缺失 seed 不参与分母。"
                ),
            },
            {"id": "seed-results", "type": "table", "layout": "full", "tableId": "per-seed-table"},
            {
                "id": "quality-intro",
                "type": "markdown",
                "layout": "full",
                "sourceId": "dual_summary_source",
                "body": (
                    "## 证据完整性\n\n"
                    "质量账本从预期 50 个训练 seed、每个 2 个协议出发，因此尚未创建的任务也会显示为 missing_run。"
                    "这避免了只扫描现有目录时把部分矩阵误报为完整矩阵。"
                ),
            },
            {"id": "quality-chart", "type": "chart", "layout": "full", "chartId": "quality-slots"},
            {"id": "quality-detail", "type": "table", "layout": "full", "tableId": "quality-table"},
            {
                "id": "contract-intro",
                "type": "markdown",
                "layout": "full",
                "sourceId": "contract_source",
                "body": (
                    "## 最终环境与 provenance/space 验收\n\n"
                    "验收不是检查 JSON 是否存在，而是逐项验证环境类、交通分区、发布资产、确定性 seed、"
                    "逐回合 traffic_variant、模型/环境空间签名及汇总率可重算性。任何一项失败均从正式结果排除。"
                ),
            },
            {"id": "contract-detail", "type": "table", "layout": "full", "tableId": "contract-table"},
            {
                "id": "paper-comparison",
                "type": "markdown",
                "layout": "full",
                "sourceId": "paper_protocol_source",
                "body": (
                    "## 与论文数值比较时的口径\n\n"
                    "论文 Table I 是从训练成功率最高策略得到的 50 回合点估计；本报告主口径是每个训练 seed 的 "
                    "source-recoverable 最终/最新策略，再对已完成 seeds 等权求均值。因此论文列是参考线，不是同分母显著性检验。"
                    "此外，论文称 LSTM，但发布 RLEncoder 实际为 GRU(256, return_sequences=True)+Dense(256,relu)；"
                    "本报告按源码结构明确标为 RLEncoder(GRU)-SAC。"
                ),
            },
            {
                "id": "active-intro",
                "type": "markdown",
                "layout": "full",
                "sourceId": "dual_summary_source",
                "body": (
                    "## seed 补齐顺序\n\n"
                    "先补每个场景×算法的 seed0 空洞，使五个场景尽早都有 SAC 与 PPO 证据；再按 seed1→4 分波次，"
                    "每波使用 diversified 调度交错不同场景和算法，避免先把单一场景跑满。"
                ),
            },
            {"id": "active-detail", "type": "table", "layout": "full", "tableId": "active-table"},
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "sourceId": "paper_protocol_source",
                "body": (
                    "## 限制\n\n"
                    "论文未发布精确 held-out traffic 与可执行的 best-policy 选择器；SUMO 动力学代理也不等同于 "
                    "SMARTS Bullet/Ackermann。PPO 论文描述与发布配置在学习率和图像帧堆叠上存在冲突。"
                    "因此即使 50 个训练 seed 全部完成，也应把结果称为可审计迁移复现，而不是位级同构复现。"
                ),
            },
        ],
    }

    access_issues: list[dict[str, Any]] = []
    if artifact_status == "partial":
        access_issues.append(
            {
                "id": "incomplete_seed_matrix",
                "label": "seed 矩阵尚未完整",
                "description": missing_notice,
            }
        )

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": artifact_status,
            "accessIssues": access_issues,
            "datasets": {
                "matrix_summary": matrix_summary,
                "aggregate_results": aggregate_rows,
                "per_seed_results": per_seed_rows,
                "quality_counts": quality_rows,
                "contract_checks": contract_rows,
                "active_jobs": active_rows,
            },
        },
        "sources": sources,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    artifact = build_artifact(
        summary_path=args.summary.resolve(),
        protocol_path=args.protocol.resolve(),
        result_root=args.result_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
