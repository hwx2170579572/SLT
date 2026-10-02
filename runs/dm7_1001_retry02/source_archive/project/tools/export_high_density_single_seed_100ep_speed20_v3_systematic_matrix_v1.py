"""Export a validated full-metric matrix for the completed speed20-v3 study.

The exporter is deliberately additive: it reads frozen run artifacts and creates
one new output directory.  It never edits a run, protocol, environment, model,
or an existing summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_single_seed_100ep_speed20_v3"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_hd_ss100_s20_v3"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "comparison" / "matrix_full_v1"
PROFILE = "comparison"
Z_95 = 1.959963984540054


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, *, atol: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=atol)


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(float(value) for value in values)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    proportion = successes / total
    z2 = Z_95 * Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _normal_mean_interval(
    mean: float, population_std: float, total: int
) -> tuple[float, float, float]:
    standard_error = population_std / math.sqrt(total)
    half_width = Z_95 * standard_error
    return standard_error, mean - half_width, mean + half_width


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _validate_and_extract(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    result_root: Path,
    method: str,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    run_prefix = str(protocol["run_id_prefix"])
    run_id = f"{run_prefix}__{PROFILE}__{method}__{scenario}__seed{seed}"
    run_dir = result_root / PROFILE / "runs" / run_id
    required = (
        "arguments.json",
        "final_model.zip",
        "paper_evaluation_detailed.json",
        "high_density_job_receipt.json",
        "performance_profile.json",
        "training_diagnostics.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    _require(not missing, f"{run_id}: missing artifacts {missing}")

    receipt = _read_json(run_dir / "high_density_job_receipt.json")
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    detailed = _read_json(detailed_path)
    performance = _read_json(run_dir / "performance_profile.json")
    training = _read_json(run_dir / "training_diagnostics.json")

    _require(receipt.get("status") == "completed", f"{run_id}: bad status")
    _require(
        receipt.get("computed_from_real_run") is True,
        f"{run_id}: not marked as a real run",
    )
    _require(
        receipt.get("fabricated_values") is False,
        f"{run_id}: no-fabrication field mismatch",
    )
    for key, expected in (
        ("method", method),
        ("scenario", scenario),
        ("seed", seed),
        ("profile", PROFILE),
        ("protocol_sha256", protocol_sha256),
    ):
        _require(receipt.get(key) == expected, f"{run_id}: receipt {key} mismatch")

    artifact_hashes = receipt.get("artifact_sha256", {})
    for name in ("arguments.json", "final_model.zip", "paper_evaluation_detailed.json"):
        observed = _sha256(run_dir / name)
        _require(
            artifact_hashes.get(name) == observed,
            f"{run_id}: SHA-256 mismatch for {name}",
        )

    method_spec = protocol["methods"][method]
    density = protocol["density"][scenario]
    comparison = protocol["profiles"][PROFILE]
    summary = detailed.get("summary")
    records = detailed.get("episode_records")
    _require(isinstance(summary, dict), f"{run_id}: summary is missing")
    _require(isinstance(records, list), f"{run_id}: episode_records is missing")

    expected_episodes = int(comparison["evaluation_episodes"])
    expected_seed_start = int(comparison["evaluation_seed_start"])
    expected_seeds = list(range(expected_seed_start, expected_seed_start + expected_episodes))
    _require(int(summary["episodes"]) == expected_episodes, f"{run_id}: episode count")
    _require(len(records) == expected_episodes, f"{run_id}: record count")
    _require(
        [int(record["seed"]) for record in records] == expected_seeds,
        f"{run_id}: evaluation seed block mismatch",
    )

    outcome_keys = ("success", "collision", "off_route", "timeout")
    for index, record in enumerate(records):
        outcome_count = sum(bool(record[key]) for key in outcome_keys)
        _require(
            outcome_count == 1,
            f"{run_id}: episode {index} has {outcome_count} terminal outcomes",
        )
    counts = {
        key: sum(bool(record[key]) for record in records) for key in outcome_keys
    }
    rates = {key: counts[key] / expected_episodes for key in outcome_keys}
    summary_rate_names = {
        "success": "success_rate",
        "collision": "collision_rate",
        "off_route": "off_route_rate",
        "timeout": "timeout_rate",
    }
    for key, summary_name in summary_rate_names.items():
        _require(
            _close(rates[key], float(summary[summary_name])),
            f"{run_id}: {summary_name} does not match episode records",
        )

    returns = [float(record["episode_return"]) for record in records]
    decisions = [float(record["decision_steps"]) for record in records]
    raw_steps = [float(record["raw_steps"]) for record in records]
    return_mean = _mean(returns)
    return_std = statistics.pstdev(returns)
    decision_mean = _mean(decisions)
    raw_step_mean = _mean(raw_steps)
    for observed, expected, label in (
        (return_mean, summary["mean_return"], "mean_return"),
        (return_std, summary["std_return"], "std_return"),
        (decision_mean, summary["mean_decision_steps"], "mean_decision_steps"),
        (raw_step_mean, summary["mean_raw_steps"], "mean_raw_steps"),
    ):
        _require(_close(observed, float(expected)), f"{run_id}: {label} mismatch")

    completion_times = [
        float(record["completion_time_seconds"])
        for record in records
        if bool(record["success"])
    ]
    completion_mean = _mean(completion_times) if completion_times else None
    completion_std = statistics.pstdev(completion_times) if completion_times else None
    _require(
        int(detailed["successful_episodes"]) == counts["success"],
        f"{run_id}: successful_episodes mismatch",
    )
    if completion_times:
        _require(
            _close(completion_mean, detailed["mean_success_completion_time_seconds"]),
            f"{run_id}: successful completion mean mismatch",
        )
        _require(
            _close(completion_std, detailed["std_success_completion_time_seconds"]),
            f"{run_id}: successful completion std mismatch",
        )
    else:
        _require(
            detailed.get("mean_success_completion_time_seconds") is None,
            f"{run_id}: completion mean must be null with zero successes",
        )
        _require(
            detailed.get("std_success_completion_time_seconds") is None,
            f"{run_id}: completion std must be null with zero successes",
        )

    provenance = detailed.get("evaluation_provenance", {})
    _require(provenance.get("validated") is True, f"{run_id}: provenance invalid")
    _require(
        provenance.get("model_environment_spaces_match") is True,
        f"{run_id}: model/environment spaces mismatch",
    )
    _require(
        int(provenance["evaluation_seed_start"]) == expected_seed_start,
        f"{run_id}: provenance evaluation seed mismatch",
    )

    requested_raw_steps = int(comparison["raw_training_steps"])
    collected_raw_steps = int(detailed["collected_training_raw_steps"])
    _require(collected_raw_steps == requested_raw_steps, f"{run_id}: training budget")
    _require(int(performance["raw_steps"]) == requested_raw_steps, f"{run_id}: profile raw steps")
    _require(int(training["raw_steps"]) == requested_raw_steps, f"{run_id}: diagnostics raw steps")

    sumo_step_seconds = float(detailed["sumo_step_seconds"])
    return_sem, return_ci_low, return_ci_high = _normal_mean_interval(
        return_mean, return_std, expected_episodes
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "status": "complete_validated",
        "method": method,
        "display_label": method_spec["display_label"],
        "role": method_spec["role"],
        "trainer_adapter": method_spec["trainer_adapter"],
        "algorithm": method_spec["cli_algorithm"],
        "implementation_id": method_spec["implementation_id"],
        "scenario": scenario,
        "scenario_id": density["scenario_id"],
        "training_seed": seed,
        "vehicle_scale": float(density["vehicle_scale"]),
        "pedestrian_scale": float(density["pedestrian_scale"]),
        "clone_depart_jitter_seconds": json.dumps(
            density["clone_depart_jitter_seconds"], ensure_ascii=False
        ),
        "ego_speed_min_mps": float(
            protocol["environment_contract"]["ego_speed_control_interval_mps"][0]
        ),
        "ego_speed_max_mps": float(
            protocol["environment_contract"]["ego_speed_control_interval_mps"][1]
        ),
        "requested_training_raw_steps": requested_raw_steps,
        "collected_training_raw_steps": collected_raw_steps,
        "learner_updates": int(performance["learner_updates"]),
        "evaluation_seed_start": expected_seed_start,
        "evaluation_seed_end": expected_seeds[-1],
        "evaluation_episodes": expected_episodes,
        "success_count": counts["success"],
        "success_rate": rates["success"],
        "collision_count": counts["collision"],
        "collision_rate": rates["collision"],
        "off_route_count": counts["off_route"],
        "off_route_rate": rates["off_route"],
        "timeout_count": counts["timeout"],
        "timeout_rate": rates["timeout"],
        "mean_return": return_mean,
        "std_return_episode_population": return_std,
        "return_episode_sem": return_sem,
        "return_episode_normal_ci95_low": return_ci_low,
        "return_episode_normal_ci95_high": return_ci_high,
        "mean_decision_steps": decision_mean,
        "mean_raw_steps": raw_step_mean,
        "sumo_step_seconds": sumo_step_seconds,
        "mean_episode_simulation_seconds": raw_step_mean * sumo_step_seconds,
        "successful_episodes": counts["success"],
        "mean_success_completion_time_seconds": completion_mean,
        "std_success_completion_time_seconds_episode_population": completion_std,
        "training_wall_seconds": float(performance["end_to_end_training_wall_seconds"]),
        "training_wall_hours": float(performance["end_to_end_training_wall_seconds"]) / 3600.0,
        "wall_ms_per_learner_update": float(
            performance["end_to_end_wall_ms_per_learner_update"]
        ),
        "peak_gpu_memory_mb": float(performance["peak_gpu_memory_mb"]),
        "policy_total_parameters": int(performance["parameters"]["policy_total"]),
        "policy_trainable_parameters": int(
            performance["parameters"]["policy_trainable"]
        ),
        "critic_feature_extractor_parameters": int(
            performance["parameters"]["critic_feature_extractor"]
        ),
        "representation_objective_parameters": int(
            performance["parameters"]["representation_objective"]
        ),
        "model_plus_representation_parameters": int(
            performance["parameters"]["model_plus_representation"]
        ),
        "inference_device": performance["inference"]["device"],
        "inference_measured_calls": int(performance["inference"]["measured_calls"]),
        "mean_inference_ms_per_action": float(
            performance["inference"]["mean_milliseconds_per_action"]
        ),
        "inference_actions_per_second": float(
            performance["inference"]["actions_per_second"]
        ),
        "selected_model_learner_timesteps": detailed.get(
            "selected_model_learner_timesteps"
        ),
        "post_training_learner_timesteps": detailed.get(
            "post_training_learner_timesteps"
        ),
        "source_test_checkpoint_step": detailed.get("source_test_checkpoint_step"),
        "selected_checkpoint_kind": detailed.get("selected_checkpoint_kind"),
        "selected_deployment_decoder": detailed.get("selected_deployment_decoder"),
        "protocol_sha256": protocol_sha256,
        "arguments_sha256": artifact_hashes["arguments.json"],
        "final_model_sha256": artifact_hashes["final_model.zip"],
        "detailed_evaluation_sha256": artifact_hashes[
            "paper_evaluation_detailed.json"
        ],
        "run_directory": str(run_dir.resolve()),
        "protocol_path": str(protocol_path.resolve()),
    }
    for key in outcome_keys:
        low, high = _wilson_interval(counts[key], expected_episodes)
        row[f"{key}_rate_wilson95_low"] = low
        row[f"{key}_rate_wilson95_high"] = high
    return row


def _macro_rows(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "mean_decision_steps",
        "mean_raw_steps",
        "mean_episode_simulation_seconds",
        "mean_inference_ms_per_action",
        "inference_actions_per_second",
    )
    for method in protocol["matrix"]["method_order"]:
        selected = [row for row in rows if row["method"] == method]
        expected_count = len(protocol["matrix"]["scenario_order"])
        _require(len(selected) == expected_count, f"macro: incomplete method {method}")
        macro: dict[str, Any] = {
            "method": method,
            "display_label": protocol["methods"][method]["display_label"],
            "scenario_count": expected_count,
            "evaluation_episodes_total": sum(
                int(row["evaluation_episodes"]) for row in selected
            ),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            macro[f"macro_{metric}"] = _mean(values)
            macro[f"scenario_population_std_{metric}"] = statistics.pstdev(values)
        completion_values = [
            row["mean_success_completion_time_seconds"] for row in selected
        ]
        macro["macro_mean_success_completion_time_seconds"] = (
            _mean(completion_values)
            if all(value is not None for value in completion_values)
            else None
        )
        macro["training_wall_hours_total"] = sum(
            float(row["training_wall_hours"]) for row in selected
        )
        macro["training_wall_hours_mean_per_scenario"] = _mean(
            row["training_wall_hours"] for row in selected
        )
        macro["peak_gpu_memory_mb_max"] = max(
            float(row["peak_gpu_memory_mb"]) for row in selected
        )
        parameter_values = [
            int(row["model_plus_representation_parameters"]) for row in selected
        ]
        macro["model_plus_representation_parameters_mean"] = _mean(
            parameter_values
        )
        macro["model_plus_representation_parameters_min"] = min(parameter_values)
        macro["model_plus_representation_parameters_max"] = max(parameter_values)
        output.append(macro)

    ranked = sorted(
        output,
        key=lambda row: (
            -float(row["macro_success_rate"]),
            float(row["macro_collision_rate"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["primary_rank_success_then_collision"] = rank
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _num(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _pm(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _markdown(
    protocol: dict[str, Any],
    protocol_sha256: str,
    rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Speed20-v3 系统性实验矩阵（完整指标 v1）",
        "",
        "## 实验口径",
        "",
        f"- 完成度：18/18 正式任务；共 {sum(int(row['evaluation_episodes']) for row in rows)} 个评估回合。",
        f"- 协议 SHA-256：`{protocol_sha256}`。",
        "- 训练种子：0；每个方法×场景使用评估种子 10000–10099。",
        "- 每任务训练预算：50,000 原始 SUMO 步；自车受控速度区间：[0, 20] m/s。",
        "- 率指标的 95% Wilson 区间与回报均值的 episode-level 正态近似区间保存在 CSV/JSON；它们不是跨训练种子置信区间。",
        "- `±` 为 100 个评估回合的总体标准差（ddof=0）；单训练种子无法估计跨种子方差或显著性。",
        "",
        "## 方法矩阵",
        "",
        "| 方法 | 角色 | Adapter | 算法 |",
        "| --- | --- | --- | --- |",
    ]
    for method in protocol["matrix"]["method_order"]:
        spec = protocol["methods"][method]
        lines.append(
            f"| {spec['display_label']} | {spec['role']} | {spec['trainer_adapter']} | `{spec['cli_algorithm']}` |"
        )
    lines.extend(
        [
            "",
            "## 场景与压力设置",
            "",
            "| 场景 | 车辆倍率 | 行人倍率 | 克隆发车抖动(s) | 地图/自车路线 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for scenario in protocol["matrix"]["scenario_order"]:
        density = protocol["density"][scenario]
        jitter = density["clone_depart_jitter_seconds"]
        lines.append(
            f"| {scenario} | {density['vehicle_scale']:.2f}× | {density['pedestrian_scale']:.2f}× | {jitter[0]:.1f}–{jitter[1]:.1f} | 不变 |"
        )
    lines.extend(
        [
            "",
            "## 主结果：安全与任务完成",
            "",
            "| 方法 | 场景 | 成功 n/% ↑ | 碰撞 n/% ↓ | 偏航 n/% ↓ | 超时 n/% ↓ | 回报 μ ± σ ↑ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {display_label} | {scenario} | {success_count}/{success_pct}% | "
            "{collision_count}/{collision_pct}% | {off_route_count}/{off_route_pct}% | "
            "{timeout_count}/{timeout_pct}% | {return_pm} |".format(
                **row,
                success_pct=_pct(row["success_rate"]),
                collision_pct=_pct(row["collision_rate"]),
                off_route_pct=_pct(row["off_route_rate"]),
                timeout_pct=_pct(row["timeout_rate"]),
                return_pm=_pm(
                    row["mean_return"], row["std_return_episode_population"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 完整公共指标：轨迹、时长与计算成本",
            "",
            "| 方法 | 场景 | 决策步 μ ↓ | 原始步 μ ↓ | 仿真时长 μ(s) ↓ | 成功完成时间 μ ± σ(s) ↓ | 训练(h) ↓ | 峰值GPU(MB) ↓ | 模型+表征参数(M) ↓ | 推理(ms/action) ↓ | action/s ↑ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['display_label']} | {row['scenario']} | "
            f"{row['mean_decision_steps']:.2f} | {row['mean_raw_steps']:.2f} | "
            f"{row['mean_episode_simulation_seconds']:.2f} | "
            f"{_pm(row['mean_success_completion_time_seconds'], row['std_success_completion_time_seconds_episode_population'])} | "
            f"{row['training_wall_hours']:.2f} | {row['peak_gpu_memory_mb']:.1f} | "
            f"{row['model_plus_representation_parameters'] / 1_000_000:.3f} | "
            f"{row['mean_inference_ms_per_action']:.3f} | {row['inference_actions_per_second']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 三场景等权宏平均",
            "",
            "| 排名* | 方法 | 成功率 ↑ | 碰撞率 ↓ | 偏航率 ↓ | 超时率 ↓ | 回报 ↑ | 原始步 μ ↓ | 三场景训练总时长(h) ↓ | 推理(ms/action) ↓ |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        macro_rows, key=lambda item: item["primary_rank_success_then_collision"]
    ):
        lines.append(
            f"| {row['primary_rank_success_then_collision']} | {row['display_label']} | "
            f"{_pct(row['macro_success_rate'])}% | {_pct(row['macro_collision_rate'])}% | "
            f"{_pct(row['macro_off_route_rate'])}% | {_pct(row['macro_timeout_rate'])}% | "
            f"{row['macro_mean_return']:.3f} | {row['macro_mean_raw_steps']:.2f} | "
            f"{row['training_wall_hours_total']:.2f} | {row['macro_mean_inference_ms_per_action']:.3f} |"
        )
    lines.extend(
        [
            "",
            "\* 排名严格按协议主指标顺序：三场景宏成功率降序，若相同再按宏碰撞率升序；不表示统计显著性。",
            "",
            "## 完整性与边界",
            "",
            "- 18 个正式 run 的必需工件、回执字段和 SHA-256 均已复核；每个 run 的四类终局计数之和均为 100。",
            "- 成功完成时间只在成功回合上定义；若某场景成功数为 0，则记为 `N/A`，不是实验缺失。",
            "- 实际车速没有在六种方法的统一 100 回合评估记录中共同落盘；因此公平主矩阵只报告受控区间 `[0,20] m/s`，不拿 v4 专属动作诊断替代基线速度指标。",
            "- v4 专属机制/动作诊断、训练损失与模型选择信息仍保留在各 run 原始工件中，未混入跨方法主排名。",
            "- 本实验只有一个训练种子；不能据此给出跨种子标准差、显著性检验或稳健性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    result_root = args.result_root.resolve()
    output_dir = args.output_dir.resolve()
    _require(protocol_path.is_file(), f"protocol not found: {protocol_path}")
    _require(not output_dir.exists(), f"refusing to overwrite: {output_dir}")

    protocol = _read_json(protocol_path)
    protocol_sha256 = _sha256(protocol_path)
    _require(protocol.get("no_fabrication") is True, "protocol no_fabrication mismatch")
    profile = protocol["profiles"][PROFILE]
    seeds = [int(seed) for seed in profile["seeds"]]
    _require(seeds == [0], f"expected the frozen single seed [0], got {seeds}")

    rows = [
        _validate_and_extract(
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_sha256=protocol_sha256,
            result_root=result_root,
            method=method,
            scenario=scenario,
            seed=seed,
        )
        for method in protocol["matrix"]["method_order"]
        for scenario in protocol["matrix"]["scenario_order"]
        for seed in seeds
    ]
    expected_jobs = (
        len(protocol["matrix"]["method_order"])
        * len(protocol["matrix"]["scenario_order"])
        * len(seeds)
    )
    _require(len(rows) == expected_jobs == 18, "matrix size is not 18")
    macro_rows = _macro_rows(protocol, rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    detailed_csv = output_dir / "matrix_full_v1.csv"
    macro_csv = output_dir / "macro_full_v1.csv"
    json_path = output_dir / "matrix_full_v1.json"
    markdown_path = output_dir / "matrix_full_v1.md"
    _write_csv(detailed_csv, rows)
    _write_csv(macro_csv, macro_rows)

    payload = {
        "schema_version": "speed20-v3-systematic-complete-matrix/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "complete": True,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "profile": PROFILE,
        "training_seeds": seeds,
        "expected_jobs": expected_jobs,
        "accepted_jobs": len(rows),
        "evaluation_episodes_per_job": int(profile["evaluation_episodes"]),
        "total_evaluation_episodes": sum(
            int(row["evaluation_episodes"]) for row in rows
        ),
        "cross_training_seed_variance_estimable": False,
        "episode_interval_scope": (
            "Wilson/normal intervals describe the fixed model's 100 evaluation "
            "episodes only; they are not cross-training-seed uncertainty."
        ),
        "uniform_actual_speed_metric_available": False,
        "actual_speed_metric_exclusion_reason": (
            "Actual speed was not recorded uniformly in the common 100-episode "
            "evaluation artifact for all six methods."
        ),
        "rows": rows,
        "macro": macro_rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(
        _markdown(protocol, protocol_sha256, rows, macro_rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "accepted_jobs": len(rows),
                "total_evaluation_episodes": payload["total_evaluation_episodes"],
                "protocol_sha256": protocol_sha256,
                "output_directory": str(output_dir),
                "files": [
                    str(detailed_csv),
                    str(macro_csv),
                    str(json_path),
                    str(markdown_path),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
