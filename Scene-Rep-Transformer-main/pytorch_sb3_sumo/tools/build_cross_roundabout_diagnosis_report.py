"""Build the canonical artifact for the cross/Roundabout-A/C diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMO_ROOT = PROJECT_ROOT / "pytorch_sb3_sumo"
RESULT_ROOT = (
    PROJECT_ROOT
    / "results_sb3_sumo_paper"
    / "paper_baselines_sac_ppo_v14_frozen"
)
REPORT_ROOT = (
    SUMO_ROOT / "reports" / "cross_roundabout_anomaly_diagnosis_20260805"
)
SCENARIO_ROOT = SUMO_ROOT / "envs" / "sumo" / "original_scenarios_v1"
PROTOCOL_PATH = (
    SUMO_ROOT / "experiments" / "sb3_sumo_paper" / "protocol_baselines_v14.json"
)
PAPER_TIME_PATH = (
    SUMO_ROOT / "experiments" / "sb3_sumo_paper" / "paper_reference_tables.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _monitor_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(lines[1:]))


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _rolling_success_max(rows: list[dict[str, str]], window: int = 20) -> float:
    if len(rows) < window:
        return 0.0
    return max(
        sum(_truth(row["is_success"]) for row in rows[index - window + 1 : index + 1])
        / window
        for index in range(window - 1, len(rows))
    )


def _detail(algorithm: str, scenario: str, protocol: str) -> dict[str, Any]:
    return _read_json(
        RESULT_ROOT
        / f"paper__{algorithm}__{scenario}__seed0"
        / "protocol_evaluations"
        / protocol
        / "paper_evaluation_detailed.json"
    )


def _trace_summary(payload: dict[str, Any]) -> dict[str, Any]:
    reports = list(payload["episode_reports"])
    return {
        "episodes": len(reports),
        "requested_speed_mean": fmean(
            float(row["requested_speed_mps"]["mean"]) for row in reports
        ),
        "actual_speed_mean": fmean(
            float(row["actual_speed_mps"]["mean"]) for row in reports
        ),
        "distance_mean": fmean(float(row["distance_travelled_m"]) for row in reports),
        "distance_min": min(float(row["distance_travelled_m"]) for row in reports),
        "distance_max": max(float(row["distance_travelled_m"]) for row in reports),
        "final_roads": sorted(
            {str(row["final_state"]["road_id"]) for row in reports}
        ),
        "final_route_indices": sorted(
            {int(row["final_state"]["route_index"]) for row in reports}
        ),
    }


def _route_contract(scenario: str) -> dict[str, Any]:
    directory = SCENARIO_ROOT / scenario
    route_root = ET.parse(directory / "ego.rou.xml").getroot()
    network_root = ET.parse(directory / "map.net.xml").getroot()
    route = route_root.find("route")
    vehicle = route_root.find("vehicle")
    if route is None or vehicle is None:
        raise ValueError(f"Missing released ego route for {scenario}")
    edges = route.attrib["edges"].split()
    edge_lengths: list[float] = []
    for edge_id in edges:
        edge = network_root.find(f"edge[@id='{edge_id}']")
        if edge is None:
            raise ValueError(f"Missing edge {edge_id} in {scenario}")
        lane = edge.find("lane")
        if lane is None:
            raise ValueError(f"Missing lane for {edge_id} in {scenario}")
        edge_lengths.append(float(lane.attrib["length"]))
    traffic_files = sorted((directory / "traffic").glob("traffic_*.rou.xml"))
    initial_counts: list[int] = []
    depart_time = float(vehicle.attrib["depart"])
    for traffic_path in traffic_files:
        traffic_root = ET.parse(traffic_path).getroot()
        initial_counts.append(
            sum(
                float(item.attrib.get("depart", "inf")) <= depart_time
                for item in traffic_root.findall("vehicle")
            )
        )
    return {
        "route_edges": edges,
        "route_external_length_m": sum(edge_lengths),
        "ego_depart_pos_m": float(vehicle.attrib["departPos"]),
        "ego_depart_time_s": depart_time,
        "traffic_files": len(traffic_files),
        "frozen_variants": len(traffic_files[::5]),
        "initial_social_vehicles_mean": fmean(initial_counts),
        "initial_social_vehicles_min": min(initial_counts),
        "initial_social_vehicles_max": max(initial_counts),
    }


def _source(
    source_id: str,
    label: str,
    *,
    path: Path,
    description: str,
    tables_used: list[str],
    generated_at: str,
    metric_definitions: list[str] | None = None,
    filters: list[str] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "engine": "DuckDB-compatible local artifact audit",
        "id": source_id,
        "description": description,
        "executed_at": generated_at,
        "language": "sql",
        "tables_used": tables_used,
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    if filters:
        query["filters"] = filters
    return {
        "id": source_id,
        "label": label,
        "path": _rel(path),
        "query": query,
    }


def _card(
    card_id: str,
    *,
    description: str,
    source_id: str,
    field: str,
    label: str,
    value_format: str,
    unit: str | None = None,
    comparison: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = [
        {"label": label, "field": field, "format": value_format}
    ]
    if unit is not None:
        metrics[0]["unit"] = unit
    if comparison is not None:
        comparison_label, comparison_field, comparison_format = comparison
        metrics.append(
            {
                "label": comparison_label,
                "field": comparison_field,
                "format": comparison_format,
            }
        )
    return {
        "id": card_id,
        "description": description,
        "dataset": "headline_metrics",
        "sourceId": source_id,
        "metrics": metrics,
    }


def _table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: list[dict[str, Any]],
    *,
    sort_field: str = "order",
    sort_direction: str = "asc",
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "spacious",
        "layout": "full",
        "defaultSort": {"field": sort_field, "direction": sort_direction},
        "columns": columns,
    }


def build_artifact() -> dict[str, Any]:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    protocol = _read_json(PROTOCOL_PATH)
    paper_times = _read_json(PAPER_TIME_PATH)["completion_time_seconds"]

    cross_h600 = {
        algorithm: _read_json(
            REPORT_ROOT
            / "evidence"
            / f"{algorithm}_cross_seed0_source_all_h600.json"
        )
        for algorithm in ("ppo", "sac")
    }
    cross_traces = {
        algorithm: _read_json(
            REPORT_ROOT / "evidence" / f"{algorithm}_cross_seed0_trace_h600.json"
        )
        for algorithm in ("ppo", "sac")
    }
    trace_summaries = {
        algorithm: _trace_summary(payload)
        for algorithm, payload in cross_traces.items()
    }

    cross_rows: list[dict[str, Any]] = []
    algorithm_labels = {"ppo": "PPO", "sac": "RLEncoder(GRU)-SAC"}
    for order, algorithm in enumerate(("ppo", "sac"), start=1):
        monitor_path = (
            RESULT_ROOT / f"paper__{algorithm}__cross__seed0" / "train_monitor.csv"
        )
        monitor = _monitor_rows(monitor_path)
        h400_source = _detail(algorithm, "cross", "source_all")
        h400_frozen = _detail(algorithm, "cross", "frozen_80_20")
        h600_source = cross_h600[algorithm]
        trace = trace_summaries[algorithm]
        cross_rows.append(
            {
                "order": order,
                "algorithm": algorithm_labels[algorithm],
                "training_episodes": len(monitor),
                "training_successes": sum(
                    _truth(row["is_success"]) for row in monitor
                ),
                "training_max_last20_success": _rolling_success_max(monitor),
                "h400_frozen_success": float(
                    h400_frozen["summary"]["success_rate"]
                ),
                "h400_source_success": float(
                    h400_source["summary"]["success_rate"]
                ),
                "h400_source_timeout": float(
                    h400_source["summary"]["timeout_rate"]
                ),
                "h600_source_success": float(
                    h600_source["summary"]["success_rate"]
                ),
                "h600_source_timeout": float(
                    h600_source["summary"]["timeout_rate"]
                ),
                "trace_episodes": int(trace["episodes"]),
                "mean_requested_speed_mps": float(trace["requested_speed_mean"]),
                "mean_actual_speed_mps": float(trace["actual_speed_mean"]),
                "mean_distance_60s_m": float(trace["distance_mean"]),
                "distance_range_60s_m": (
                    f"{trace['distance_min']:.2f}–{trace['distance_max']:.2f}"
                ),
                "final_road": ", ".join(trace["final_roads"]),
                "final_route_index": ", ".join(
                    str(value) for value in trace["final_route_indices"]
                ),
            }
        )

    roundabout_rows: list[dict[str, Any]] = []
    success_comparison: list[dict[str, Any]] = []
    completion_comparison: list[dict[str, Any]] = []
    paper_labels = {
        "roundabout_easy": "环岛 A",
        "roundabout": "环岛 C",
    }
    paper_horizons = {"roundabout_easy": 400, "roundabout": 800}
    for scenario_order, scenario in enumerate(
        ("roundabout_easy", "roundabout"), start=1
    ):
        label = paper_labels[scenario]
        paper_success, paper_collision, paper_stagnation = (
            value / 100.0
            for value in protocol["paper_reference_percent"][scenario]["PPO"]
        )
        success_comparison.append(
            {
                "order": scenario_order * 10 + 1,
                "scenario": label,
                "protocol": "论文 Table I",
                "success_rate": paper_success,
            }
        )
        paper_completion = float(paper_times[scenario]["PPO"][0])
        completion_comparison.extend(
            [
                {
                    "order": scenario_order * 10 + 1,
                    "scenario": label,
                    "series": "论文成功回合均值",
                    "seconds": paper_completion,
                },
                {
                    "order": scenario_order * 10 + 3,
                    "scenario": label,
                    "series": "论文场景时限",
                    "seconds": paper_horizons[scenario] / 10.0,
                },
            ]
        )
        for protocol_order, traffic_protocol in enumerate(
            ("frozen_80_20", "source_all"), start=2
        ):
            payload = _detail("ppo", scenario, traffic_protocol)
            episodes = list(payload["episode_records"])
            successes = [row for row in episodes if row["success"]]
            success_timeout_overlap = sum(
                bool(row["success"] and row["timeout"]) for row in episodes
            )
            row = {
                "order": scenario_order * 10 + protocol_order,
                "scenario": label,
                "protocol": traffic_protocol,
                "episodes": len(episodes),
                "success_rate": float(payload["summary"]["success_rate"]),
                "collision_rate": float(payload["summary"]["collision_rate"]),
                "timeout_rate": float(payload["summary"]["timeout_rate"]),
                "success_timeout_overlap": success_timeout_overlap,
                "successful_episodes": len(successes),
                "success_after_step400": sum(
                    int(row["raw_steps"]) > 400 for row in successes
                ),
                "mean_success_steps": (
                    fmean(float(row["raw_steps"]) for row in successes)
                    if successes
                    else None
                ),
                "min_success_steps": (
                    min(int(row["raw_steps"]) for row in successes)
                    if successes
                    else None
                ),
                "max_success_steps": (
                    max(int(row["raw_steps"]) for row in successes)
                    if successes
                    else None
                ),
                "traffic_variants": len(
                    payload["evaluation_provenance"]["traffic_variants_observed"]
                ),
                "provenance_validated": bool(
                    payload["evaluation_provenance"]["validated"]
                ),
                "spaces_match": bool(
                    payload["evaluation_provenance"][
                        "model_environment_spaces_match"
                    ]
                ),
            }
            roundabout_rows.append(row)
            success_comparison.append(
                {
                    "order": row["order"],
                    "scenario": label,
                    "protocol": traffic_protocol,
                    "success_rate": row["success_rate"],
                }
            )
            if traffic_protocol == "source_all":
                completion_comparison.append(
                    {
                        "order": scenario_order * 10 + 2,
                        "scenario": label,
                        "series": "迁移 source-all 成功均值",
                        "seconds": float(payload["mean_success_completion_time_seconds"]),
                    }
                )
        roundabout_rows.append(
            {
                "order": scenario_order * 10 + 4,
                "scenario": label,
                "protocol": "论文 Table I",
                "episodes": 50,
                "success_rate": paper_success,
                "collision_rate": paper_collision,
                "timeout_rate": paper_stagnation,
                "success_timeout_overlap": None,
                "successful_episodes": int(round(paper_success * 50)),
                "success_after_step400": None,
                "mean_success_steps": paper_completion * 10.0,
                "min_success_steps": None,
                "max_success_steps": None,
                "traffic_variants": None,
                "provenance_validated": None,
                "spaces_match": None,
            }
        )

    scenario_contract_rows: list[dict[str, Any]] = []
    source_horizons = {
        "cross": 600,
        "roundabout_easy": 400,
        "roundabout": 1000,
    }
    for order, scenario in enumerate(
        ("cross", "roundabout_easy", "roundabout"), start=1
    ):
        contract = _route_contract(scenario)
        paper_horizon = {"cross": 400, "roundabout_easy": 400, "roundabout": 800}[
            scenario
        ]
        remaining_distance = (
            contract["route_external_length_m"] - contract["ego_depart_pos_m"]
        )
        traffic_generator = {
            "cross": "3 条流：80/200/200 veh/h；6 actor types",
            "roundabout_easy": "9×80 veh/h + 1×100 veh/h 特殊流",
            "roundabout": "9×50 veh/h；无 A 的特殊流",
        }[scenario]
        scenario_contract_rows.append(
            {
                "order": order,
                "scenario": {
                    "cross": "双汇入",
                    "roundabout_easy": "环岛 A",
                    "roundabout": "环岛 C",
                }[scenario],
                "paper_horizon_steps": paper_horizon,
                "source_horizon_steps": source_horizons[scenario],
                "control_time_limit_s": paper_horizon / 10.0,
                "external_route_length_m": contract["route_external_length_m"],
                "remaining_external_distance_m": remaining_distance,
                "minimum_required_mean_speed_mps": remaining_distance
                / (paper_horizon / 10.0),
                "released_traffic_files": contract["traffic_files"],
                "frozen_traffic_variants": contract["frozen_variants"],
                "initial_social_vehicles_mean": contract[
                    "initial_social_vehicles_mean"
                ],
                "traffic_generator": traffic_generator,
                "ego_route": " → ".join(contract["route_edges"]),
            }
        )

    speed_trace_rows: list[dict[str, Any]] = []
    for algorithm in ("ppo", "sac"):
        episode = cross_traces[algorithm]["episode_reports"][0]
        for sample_order, sample in enumerate(episode["sampled_trace"], start=1):
            vehicle_state = sample["vehicle_state"] or {}
            speed_trace_rows.append(
                {
                    "order": sample_order,
                    "algorithm": algorithm_labels[algorithm],
                    "raw_steps": int(sample["raw_steps"]),
                    "requested_speed_mps": float(sample["requested_speed_mps"]),
                    "actual_speed_mps": float(vehicle_state.get("speed_mps", 0.0)),
                    "road_id": vehicle_state.get("road_id"),
                    "route_index": vehicle_state.get("route_index"),
                }
            )

    source_a = next(
        row
        for row in roundabout_rows
        if row["scenario"] == "环岛 A" and row["protocol"] == "source_all"
    )
    source_c = next(
        row
        for row in roundabout_rows
        if row["scenario"] == "环岛 C" and row["protocol"] == "source_all"
    )
    cross_training_episodes = sum(row["training_episodes"] for row in cross_rows)
    cross_training_successes = sum(row["training_successes"] for row in cross_rows)
    headline_metrics = [
        {
            "cross_training_successes": cross_training_successes,
            "cross_training_episodes": cross_training_episodes,
            "cross_h600_success_rate": 0.0,
            "ppo_a_source_success": source_a["success_rate"],
            "ppo_c_source_success": source_c["success_rate"],
            "ppo_c_minus_a_source": source_c["success_rate"]
            - source_a["success_rate"],
            "ppo_c_success_after_400_fraction": source_c[
                "success_after_step400"
            ]
            / source_c["successful_episodes"],
        }
    ]

    protocol_conflicts = [
        {
            "order": 1,
            "dimension": "双汇入最大步数",
            "paper": "400（Table VI）",
            "released_source": "600（tools/test.py）",
            "migration_v14": "400（paper profile）",
            "diagnostic_effect": "无重训改为600后两算法仍0%；评测截断不是充分原因",
        },
        {
            "order": 2,
            "dimension": "环岛 C 最大步数",
            "paper": "800（Table VI）",
            "released_source": "1000（tools/test.py）",
            "migration_v14": "800（paper profile）",
            "diagnostic_effect": "当前高成功率不是误用了1000步",
        },
        {
            "order": 3,
            "dimension": "PPO 图像历史",
            "paper": "连续多帧栅格图像",
            "released_source": "单个当前80×80×3帧",
            "migration_v14": "按源码单帧",
            "diagnostic_effect": "A 的即时冲突速度判断更可能受影响；尚无消融定量",
        },
        {
            "order": 4,
            "dimension": "测试策略选择",
            "paper": "训练成功率最高策略",
            "released_source": "无可执行best-success保存器",
            "migration_v14": "主结果用latest；另存best重建检查点",
            "diagnostic_effect": "A/C best检查点尚未正式双协议重评",
        },
        {
            "order": 5,
            "dimension": "成功事件",
            "paper": "到达目标",
            "released_source": "events.reached_goal",
            "migration_v14": "ego进入TraCI arrived列表",
            "diagnostic_effect": "语义一致；逐回合重算与summary完全一致",
        },
        {
            "order": 6,
            "dimension": "停滞/超时",
            "paper": "保持静止并超过上限",
            "released_source": "reached_max_episode_steps即计stag",
            "migration_v14": "raw_steps达到上限即计timeout",
            "diagnostic_effect": "迁移与可执行源码一致，但比论文文字定义更宽",
        },
    ]

    confidence_rows = [
        {
            "order": 1,
            "finding": "成功率统计链没有漏记",
            "confidence": "已证实",
            "evidence": "到达判据、50回合分母、逐回合重算、provenance/space均通过",
            "remaining_test": "无",
        },
        {
            "order": 2,
            "finding": "双汇入0%是当前策略真实停滞",
            "confidence": "已证实",
            "evidence": "400/600步均100%超时；两算法训练0成功；行为追踪仅走1.9–7.4 m",
            "remaining_test": "无",
        },
        {
            "order": 3,
            "finding": "PPO seed0在C高于A",
            "confidence": "已证实",
            "evidence": "frozen差34pp，source-all差26pp；两协议方向一致",
            "remaining_test": "不等同于跨seed总体结论",
        },
        {
            "order": 4,
            "finding": "C的长时限把慢速策略转为成功",
            "confidence": "已证实",
            "evidence": "source-all 41/41个C成功均晚于第400步；A在400步截止",
            "remaining_test": "不能单独解释为何C策略更愿意前进",
        },
        {
            "order": 5,
            "finding": "A即时交通更难、迁移控制更保守共同造成反转",
            "confidence": "强推断",
            "evidence": "A有特殊冲突流、更多交通变体；A失败32%超时，C失败18%碰撞且0超时",
            "remaining_test": "需要同检查点控制器/交通消融",
        },
        {
            "order": 6,
            "finding": "PPO跨seed仍保持C>A",
            "confidence": "未决",
            "evidence": "正式v14目前两场景都只有seed0双协议结果",
            "remaining_test": "完成seed1–4并报告均值±标准差",
        },
        {
            "order": 7,
            "finding": "论文best-policy选择会消除A/C反转",
            "confidence": "未决",
            "evidence": "A最佳训练窗口15%，C为5%，但best模型尚未双协议重评",
            "remaining_test": "不重训重评两份best模型",
        },
        {
            "order": 8,
            "finding": "训练时把双汇入改为600步可恢复学习",
            "confidence": "未决",
            "evidence": "无重训600步只排除了评测截断；不能反推训练探索",
            "remaining_test": "同seed配对重训400 vs 600",
        },
    ]

    _write_csv(REPORT_ROOT / "cross_diagnosis.csv", cross_rows)
    _write_csv(REPORT_ROOT / "ppo_roundabout_comparison.csv", roundabout_rows)
    _write_csv(REPORT_ROOT / "cross_speed_trace.csv", speed_trace_rows)
    _write_csv(REPORT_ROOT / "scenario_contract.csv", scenario_contract_rows)

    evaluation_tables = [
        _rel(
            RESULT_ROOT
            / "paper__ppo__cross__seed0"
            / "protocol_evaluations"
            / "source_all"
            / "paper_evaluation_detailed.json"
        ),
        _rel(
            RESULT_ROOT
            / "paper__sac__cross__seed0"
            / "protocol_evaluations"
            / "source_all"
            / "paper_evaluation_detailed.json"
        ),
        _rel(REPORT_ROOT / "evidence" / "ppo_cross_seed0_source_all_h600.json"),
        _rel(REPORT_ROOT / "evidence" / "sac_cross_seed0_source_all_h600.json"),
        _rel(REPORT_ROOT / "evidence" / "ppo_cross_seed0_trace_h600.json"),
        _rel(REPORT_ROOT / "evidence" / "sac_cross_seed0_trace_h600.json"),
    ]
    evaluation_tables.extend(
        _rel(
            RESULT_ROOT
            / f"paper__ppo__{scenario}__seed0"
            / "protocol_evaluations"
            / traffic_protocol
            / "paper_evaluation_detailed.json"
        )
        for scenario in ("roundabout_easy", "roundabout")
        for traffic_protocol in ("frozen_80_20", "source_all")
    )
    cross_source = _source(
        "cross_evidence_source",
        "双汇入训练、400/600步评测与行为追踪",
        path=REPORT_ROOT / "evaluation_source.sql",
        description=(
            "同一seed0最终模型在严格论文环境中的双协议400步结果、source-all 600步反事实、"
            "训练monitor以及3回合确定性动作/路线进度追踪。"
        ),
        tables_used=evaluation_tables
        + [
            _rel(RESULT_ROOT / "paper__ppo__cross__seed0" / "train_monitor.csv"),
            _rel(RESULT_ROOT / "paper__sac__cross__seed0" / "train_monitor.csv"),
        ],
        generated_at=generated_at,
        metric_definitions=[
            "success_rate = reached-goal episodes / 50",
            "timeout_rate = max-time episodes / 50; boundary events may overlap",
            "training last-20 rate = successes in fixed 20 completed episodes / 20",
            "trace mean speed = arithmetic mean over deterministic policy decisions",
        ],
        filters=[
            "training seed = 0",
            "evaluation seeds = 10000..10049",
            "formal comparison uses latest 100k checkpoint",
        ],
    )
    roundabout_source = _source(
        "roundabout_evidence_source",
        "PPO环岛A/C逐回合双协议评测",
        path=REPORT_ROOT / "evaluation_source.sql",
        description=(
            "PPO seed0在环岛A/C的frozen_80_20与source_all各50回合；成功时刻、"
            "碰撞、超时、交通变体及论文Table I/II参考值均保留。"
        ),
        tables_used=evaluation_tables[6:]
        + [_rel(PROTOCOL_PATH), _rel(PAPER_TIME_PATH)],
        generated_at=generated_at,
        metric_definitions=[
            "success_rate = successful episodes / 50",
            "mean completion time = population mean over successful episodes only",
            "success_after_step400 = successful episodes whose raw_steps > 400",
        ],
        filters=["training seed = 0", "deterministic evaluation", "50 episodes"],
    )
    scenario_source = _source(
        "scenario_contract_source",
        "发布场景路线、交通XML与生成器",
        path=REPORT_ROOT / "scenario_source.sql",
        description=(
            "从发布ego.rou.xml/map.net.xml重算外部边路线长度，从全部traffic XML重算"
            "交通文件与ego出发时刻前车辆数，并核对原场景生成脚本。"
        ),
        tables_used=[
            "pytorch_sb3_sumo/envs/sumo/original_scenarios_v1/{cross,roundabout_easy,roundabout}/ego.rou.xml",
            "pytorch_sb3_sumo/envs/sumo/original_scenarios_v1/{cross,roundabout_easy,roundabout}/map.net.xml",
            "pytorch_sb3_sumo/envs/sumo/original_scenarios_v1/{cross,roundabout_easy,roundabout}/traffic/*.rou.xml",
            "tmp/original_scenarios/smarts_scenarios/{cross,roundabout_easy,roundabout}/scenario.py",
        ],
        generated_at=generated_at,
        metric_definitions=[
            "external route length = sum of first driving-lane lengths for mission route edges",
            "initial social vehicles = XML vehicles with depart <= ego mission depart time",
        ],
    )
    paper_source = _source(
        "paper_protocol_source",
        "原论文Tables I/II/VI、发布测试源码与v14协议",
        path=REPORT_ROOT / "paper_protocol_source.sql",
        description=(
            "直接核对论文PDF中的成功/碰撞/停滞定义、Table I成功率、Table II完成时间、"
            "Table VI场景上限，并与tools/test.py及v14迁移协议对照。"
        ),
        tables_used=[
            "tmp/pdfs/scene_rep_transformer_arxiv_v3.pdf",
            "tools/test.py",
            "envs/smarts/env_adapters.py",
            "envs/runners/on_policy_trainer.py",
            "envs/runners/off_policy_trainer.py",
            _rel(PROTOCOL_PATH),
            _rel(PAPER_TIME_PATH),
            "pytorch_sb3_sumo/envs/sumo/sumo_env.py",
            "pytorch_sb3_sumo/envs/sumo/paper_env.py",
            "pytorch_sb3_sumo/envs/sumo/ppo_env.py",
        ],
        generated_at=generated_at,
        metric_definitions=[
            "paper test success = episodes reaching target / all test episodes",
            "paper collision = episodes colliding / all test episodes",
            "source stagnation implementation = reached_max_episode_steps / all test episodes",
        ],
    )
    sources = [cross_source, roundabout_source, scenario_source, paper_source]

    title = "双汇入0%与PPO环岛A/C反转：机制诊断"
    cards = [
        _card(
            "cross-training-successes",
            description="两种seed0基线在双汇入训练期完成的全部回合；奖励只有到达+1、碰撞/偏航-1。",
            source_id="cross_evidence_source",
            field="cross_training_successes",
            label="双汇入训练成功回合",
            value_format="number",
            comparison=("已完成训练回合", "cross_training_episodes", "number"),
        ),
        _card(
            "cross-h600-success",
            description="不重训，仅把source-all评测上限从400放宽到源码600步；PPO与SAC各50回合。",
            source_id="cross_evidence_source",
            field="cross_h600_success_rate",
            label="600步反事实成功率",
            value_format="percent",
        ),
        _card(
            "ppo-a-source",
            description="PPO seed0、source-all、50个确定性回合。",
            source_id="roundabout_evidence_source",
            field="ppo_a_source_success",
            label="环岛A成功率",
            value_format="percent",
        ),
        _card(
            "ppo-c-source",
            description="PPO seed0、source-all、50个确定性回合。",
            source_id="roundabout_evidence_source",
            field="ppo_c_source_success",
            label="环岛C成功率",
            value_format="percent",
        ),
        _card(
            "ppo-c-minus-a",
            description="同一训练seed、同一source-all协议下C减A的成功率差。",
            source_id="roundabout_evidence_source",
            field="ppo_c_minus_a_source",
            label="C−A成功率差",
            value_format="percent",
        ),
        _card(
            "ppo-c-late-success",
            description="环岛C的41个source-all成功回合中，完成步数超过400的比例。",
            source_id="roundabout_evidence_source",
            field="ppo_c_success_after_400_fraction",
            label="C成功发生在400步后",
            value_format="percent",
        ),
    ]

    charts = [
        {
            "id": "ppo-success-comparison",
            "title": "PPO环岛A/C成功率：论文与两套迁移评测协议",
            "subtitle": "seed0迁移结果在两套交通协议下均出现C>A；论文Table I则为A>C。",
            "intent": "comparison",
            "question": "A/C排序反转是否只由frozen交通切分造成？",
            "rationale": "分组横向条形图同时保留场景和协议两个比较维度。",
            "comparisonContext": {
                "denominator": "每项50个测试回合",
                "grain": "场景×评测协议",
                "normalization": "成功回合数/50",
                "semanticFamily": "episode success rate",
                "unit": "%",
            },
            "type": "horizontalBar",
            "dataset": "ppo_success_comparison",
            "sourceId": "roundabout_evidence_source",
            "encodings": {
                "x": {"field": "scenario", "type": "ordinal", "label": "场景"},
                "y": {
                    "field": "success_rate",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "成功率",
                },
                "color": {
                    "field": "protocol",
                    "type": "nominal",
                    "label": "评测口径",
                },
                "tooltip": [
                    {"field": "protocol", "type": "nominal", "label": "口径"},
                    {
                        "field": "success_rate",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "成功率",
                    },
                ],
            },
            "xAxisTitle": "场景",
            "yAxisTitle": "成功率",
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 8,
            "palette": {"kind": "categorical", "name": "blue-gold"},
            "labels": {"values": "all"},
            "settings": {
                "orientation": "horizontal",
                "showValues": True,
                "sort": "custom",
                "categoryLabelPolicy": "wrap",
            },
        },
        {
            "id": "completion-vs-limit",
            "title": "PPO成功回合完成时间与场景时限",
            "subtitle": "迁移A成功时已用掉约92%的40秒时限；C约用掉88%的80秒时限。",
            "intent": "comparison",
            "question": "迁移策略的成功发生在场景时限的什么位置？",
            "rationale": "同为秒的完成均值与上限可直接分组比较。",
            "comparisonContext": {
                "denominator": "完成时间只统计成功回合",
                "grain": "场景×时间系列",
                "normalization": "原始0.1秒步数换算为秒",
                "semanticFamily": "time",
                "unit": "s",
            },
            "type": "bar",
            "dataset": "completion_time_comparison",
            "sourceId": "roundabout_evidence_source",
            "encodings": {
                "x": {"field": "scenario", "type": "ordinal", "label": "场景"},
                "y": {
                    "field": "seconds",
                    "type": "quantitative",
                    "format": "number",
                    "label": "秒",
                },
                "color": {"field": "series", "type": "nominal", "label": "系列"},
                "tooltip": [
                    {"field": "series", "type": "nominal", "label": "系列"},
                    {
                        "field": "seconds",
                        "type": "quantitative",
                        "format": "number",
                        "label": "秒",
                    },
                ],
            },
            "xAxisTitle": "场景",
            "yAxisTitle": "时间（秒）",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 8,
            "palette": {"kind": "categorical", "name": "blue-gold"},
            "labels": {"values": "all"},
            "settings": {"showValues": True, "sort": "custom"},
        },
        {
            "id": "cross-speed-trace",
            "title": "双汇入确定性策略的目标速度轨迹",
            "subtitle": "600步诊断回合中两种策略均长期把目标速度压在接近0 m/s。",
            "intent": "trend",
            "question": "双汇入超时是差一点到达，还是策略主动停滞？",
            "rationale": "按原始仿真步绘制目标速度，直接显示策略行为而非只看终局标签。",
            "comparisonContext": {
                "denominator": "每算法第1个诊断回合的等间隔采样点",
                "grain": "算法×原始仿真步",
                "normalization": "动作[0]线性映射至0–10 m/s",
                "semanticFamily": "target speed",
                "unit": "m/s",
            },
            "type": "line",
            "dataset": "cross_speed_trace",
            "sourceId": "cross_evidence_source",
            "encodings": {
                "x": {
                    "field": "raw_steps",
                    "type": "quantitative",
                    "label": "原始仿真步",
                },
                "y": {
                    "field": "requested_speed_mps",
                    "type": "quantitative",
                    "format": "number",
                    "label": "目标速度",
                },
                "color": {
                    "field": "algorithm",
                    "type": "nominal",
                    "label": "算法",
                },
                "tooltip": [
                    {"field": "algorithm", "type": "nominal", "label": "算法"},
                    {
                        "field": "raw_steps",
                        "type": "quantitative",
                        "format": "number",
                        "label": "步",
                    },
                    {
                        "field": "requested_speed_mps",
                        "type": "quantitative",
                        "format": "number",
                        "label": "目标速度",
                    },
                    {
                        "field": "road_id",
                        "type": "nominal",
                        "label": "道路边",
                    },
                ],
            },
            "xAxisTitle": "原始仿真步",
            "yAxisTitle": "目标速度（m/s）",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 40,
            "palette": {"kind": "categorical", "name": "blue-gold"},
            "settings": {"showPoints": True},
        },
    ]

    tables = [
        _table(
            "cross-diagnosis-table",
            "双汇入：训练、反事实评测与行为定位",
            "latest seed0模型；400步为论文profile，600步为发布tools/test.py上限；速度追踪各3回合。",
            "cross_diagnosis",
            "cross_evidence_source",
            [
                {"field": "order", "label": "序", "format": "number"},
                {"field": "algorithm", "label": "算法", "type": "text"},
                {"field": "training_episodes", "label": "训练回合", "format": "number"},
                {"field": "training_successes", "label": "训练成功", "format": "number"},
                {"field": "h400_source_success", "label": "400步成功率", "format": "percent"},
                {"field": "h600_source_success", "label": "600步成功率", "format": "percent"},
                {"field": "mean_requested_speed_mps", "label": "目标速度均值(m/s)", "format": "number"},
                {"field": "mean_distance_60s_m", "label": "60秒前进均值(m)", "format": "number"},
                {"field": "final_road", "label": "终止道路边", "type": "text"},
            ],
        ),
        _table(
            "roundabout-outcomes-table",
            "PPO环岛A/C：成功、碰撞、超时与成功时刻",
            "迁移行均为seed0每协议50回合；论文行为Table I/II参考值。",
            "roundabout_outcomes",
            "roundabout_evidence_source",
            [
                {"field": "order", "label": "序", "format": "number"},
                {"field": "scenario", "label": "场景", "type": "text"},
                {"field": "protocol", "label": "口径", "type": "text"},
                {"field": "success_rate", "label": "成功率", "format": "percent"},
                {"field": "collision_rate", "label": "碰撞率", "format": "percent"},
                {"field": "timeout_rate", "label": "超时/停滞率", "format": "percent"},
                {"field": "successful_episodes", "label": "成功回合", "format": "number"},
                {"field": "success_after_step400", "label": "成功>400步", "format": "number"},
                {"field": "mean_success_steps", "label": "成功步数均值", "format": "number"},
                {"field": "traffic_variants", "label": "交通变体", "format": "number"},
            ],
        ),
        _table(
            "scenario-contract-table",
            "场景路线、时限与发布交通资产",
            "路线长度只求和任务XML中的外部边；内部连接边未计入，因此用于相对诊断而非精确里程。",
            "scenario_contract",
            "scenario_contract_source",
            [
                {"field": "order", "label": "序", "format": "number"},
                {"field": "scenario", "label": "场景", "type": "text"},
                {"field": "paper_horizon_steps", "label": "论文步数", "format": "number"},
                {"field": "source_horizon_steps", "label": "源码步数", "format": "number"},
                {"field": "external_route_length_m", "label": "外部边长度(m)", "format": "number"},
                {"field": "minimum_required_mean_speed_mps", "label": "最低平均进度速度(m/s)", "format": "number"},
                {"field": "released_traffic_files", "label": "交通XML", "format": "number"},
                {"field": "frozen_traffic_variants", "label": "frozen变体", "format": "number"},
                {"field": "initial_social_vehicles_mean", "label": "ego出发时社会车", "format": "number"},
                {"field": "traffic_generator", "label": "发布生成器流量", "type": "text"},
            ],
        ),
        _table(
            "protocol-conflicts-table",
            "论文、发布源码与迁移v14的关键口径",
            "这些差异均显式记录；表内“诊断作用”只陈述已完成检查或尚待消融的范围。",
            "protocol_conflicts",
            "paper_protocol_source",
            [
                {"field": "order", "label": "序", "format": "number"},
                {"field": "dimension", "label": "维度", "type": "text"},
                {"field": "paper", "label": "论文", "type": "text"},
                {"field": "released_source", "label": "发布源码", "type": "text"},
                {"field": "migration_v14", "label": "迁移v14", "type": "text"},
                {"field": "diagnostic_effect", "label": "本次诊断作用", "type": "text"},
            ],
        ),
        _table(
            "confidence-table",
            "结论强度与剩余实验",
            "“已证实”来自当前文件和受控反事实；“强推断”有一致证据但尚无单因素消融；“未决”不得当作结论。",
            "confidence_rows",
            "paper_protocol_source",
            [
                {"field": "order", "label": "序", "format": "number"},
                {"field": "finding", "label": "判断", "type": "text"},
                {"field": "confidence", "label": "证据等级", "type": "text"},
                {"field": "evidence", "label": "现有证据", "type": "text"},
                {"field": "remaining_test", "label": "仍需实验", "type": "text"},
            ],
        ),
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": f"# {title}",
        },
        {
            "id": "technical-summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 技术摘要\n\n"
                "**双汇入的0%不是统计或环境工厂再次出错，而是当前两个seed0策略真实地选择了近乎停车。** "
                "严格环境下400步、放宽到发布源码600步，PPO与RLEncoder(GRU)-SAC仍全部超时；"
                "训练monitor合计527个回合没有一次到达。3回合行为追踪显示PPO平均目标速度约0.03–0.06 m/s，"
                "SAC约0.12 m/s，60秒仍停在任务第一条边。\n\n"
                "**PPO seed0在环岛C高于A也是真实的当前协议结果，但不能解释成C更简单。** "
                "source-all下A/C为56%/82%，frozen下为60%/94%。关键机制是迁移策略整体偏慢：A有32%超时，"
                "C没有超时；C的41个source-all成功全部发生在第400步之后。与此同时，C的frozen集合只有3个"
                "交通变体，A有6个；发布A场景还含C没有的特殊冲突流。当前仅有seed0，最终总体排序必须等待seed1–4。"
            ),
        },
        {
            "id": "headline-strip",
            "type": "metric-strip",
            "layout": "full",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "cross-finding",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 为什么双汇入成功率为0\n\n"
                "证据链是：**训练从未见到成功正奖励 → 策略学会以极低速度规避碰撞负奖励 → "
                "评测在第一条边超时**。双汇入外部边剩余距离约259 m，论文400步只给40秒，至少需要约6.49 m/s"
                "的平均进度速度；当前PPO/SAC实际平均速度低两个数量级。把评测上限放到60秒仍只前进几米，"
                "因此“400步把本来能成功的回合截断”已被排除为当前模型0%的充分原因。\n\n"
                "但400步训练上限仍可能使早期探索更难，因为奖励除到达+1与碰撞/偏航−1外始终为0。"
                "要判断训练时600步是否能打破零成功，需要配对重训；本报告没有用无重训反事实冒充该因果实验。"
            ),
        },
        {"id": "cross-speed-chart-block", "type": "chart", "layout": "full", "chartId": "cross-speed-trace"},
        {"id": "cross-table-block", "type": "table", "layout": "full", "tableId": "cross-diagnosis-table"},
        {
            "id": "roundabout-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "roundabout_evidence_source",
            "body": (
                "## 为什么PPO在环岛C高于环岛A\n\n"
                "排序反转主要不是A异常差，而是**C在迁移版中异常高**：source-all的A比论文低10个百分点，"
                "C却比论文高44个百分点。论文PPO在A/C的碰撞率为34%/50%、停滞率为0%/12%；"
                "迁移source-all变成12%/18%碰撞和32%/0%超时。也就是说，迁移策略把大量风险行为变成了慢速等待；"
                "A的40秒上限将等待记为失败，C的80秒上限则让慢策略有时间完成。迁移成功回合均值为A 36.78秒、"
                "C 70.11秒，分别贴近40秒和80秒上限。"
            ),
        },
        {"id": "ppo-success-chart-block", "type": "chart", "layout": "full", "chartId": "ppo-success-comparison"},
        {"id": "completion-chart-block", "type": "chart", "layout": "full", "chartId": "completion-vs-limit"},
        {"id": "roundabout-table-block", "type": "table", "layout": "full", "tableId": "roundabout-outcomes-table"},
        {
            "id": "scenario-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "scenario_contract_source",
            "body": (
                "## 场景资产也让A的即时决策更难\n\n"
                "环岛A与C并不是只改变终点。发布A生成器包含9条80 veh/h流和一条额外100 veh/h特殊流，"
                "C为9条50 veh/h流且没有该特殊流；ego出发前XML中分别有40与36辆社会车。A还发布30个交通XML，"
                "C只有15个；frozen评测因此只覆盖6与3个变体。source-all仍重复完整集合，所以C的82%不是单纯"
                "frozen抽样造成，但frozen的94%相对source-all又高12个百分点，说明小变体集合确实放大了结果。"
            ),
        },
        {"id": "scenario-table-block", "type": "table", "layout": "full", "tableId": "scenario-contract-table"},
        {
            "id": "metric-contract",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 成功率定义核验\n\n"
                "论文把测试成功率定义为“到达目标的回合数/全部测试回合”，发布SMARTS源码读取"
                "`events.reached_goal`，迁移SUMO环境只在ego进入TraCI `arrived`列表时置成功。三者语义一致，"
                "且本次每份迁移JSON都由50条逐回合记录重算通过。`provenance.validated=true`、模型与环境"
                "observation/action space签名一致、发布场景资产与traffic_variant均可追溯。\n\n"
                "需要保留的细节是：论文文字把Stag描述为“静止并超时”，发布runner实际只检查"
                "`reached_max_episode_steps`，迁移也按达到上限计timeout。因此本报告的超时与可执行源码一致，"
                "但比论文自然语言定义更宽；另有极少边界回合可同时成功和超时，不能强行把三种率当互斥堆叠。"
            ),
        },
        {"id": "protocol-table-block", "type": "table", "layout": "full", "tableId": "protocol-conflicts-table"},
        {
            "id": "confidence-intro",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 结论边界与优先实验\n\n"
                "当前可以把双汇入判定为策略停滞，把A/C反转判定为seed0下跨协议稳定存在；"
                "不能把它们外推成五seed总体规律。优先级应为：先完成正在运行的seed1–4；同时不重训重评"
                "A/C的`best_training_success_model.zip`，直接检验论文best-policy选择；随后再做双汇入400/600"
                "训练上限的同seed配对，以及A/C控制代理与交通流的单因素消融。"
            ),
        },
        {"id": "confidence-table-block", "type": "table", "layout": "full", "tableId": "confidence-table"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 限制\n\n"
                "本报告是当前latest seed0检查点的机制诊断，不是最终五seed统计报告。论文未发布真实held-out traffic、"
                "可执行best-policy选择器或SMARTS Bullet物理状态；迁移PPO按发布源码使用单帧RGB，而论文文字称连续"
                "栅格帧。SUMO Ackermann代理、确定性测试修复和交通循环都会改变碰撞/等待权衡，因此不能从本结果"
                "反推原论文数据错误。后台seed补齐未被本诊断中断。"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "双汇入0%与PPO环岛A/C成功率反转的逐层机制诊断。",
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
                "headline_metrics": headline_metrics,
                "cross_diagnosis": cross_rows,
                "cross_speed_trace": speed_trace_rows,
                "ppo_success_comparison": success_comparison,
                "completion_time_comparison": completion_comparison,
                "roundabout_outcomes": roundabout_rows,
                "scenario_contract": scenario_contract_rows,
                "protocol_conflicts": protocol_conflicts,
                "confidence_rows": confidence_rows,
            },
        },
        "sources": sources,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "artifact.json")
    args = parser.parse_args(argv)
    artifact = build_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
