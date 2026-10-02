"""Audit, aggregate, visualize, and attribute the systematic SUMO matrix.

The tool consumes only accepted outputs emitted by
``tools/reproduce_paper_sb3_sumo.py summarize``.  Missing or incompatible cells
remain explicitly ``TBD``; no metric is copied or imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "experiments" / "systematic_matrix" / "protocol.json"
)

RATE_METRICS = {
    "success_rate": "success",
    "collision_rate": "collision",
    "off_route_rate": "off_route",
    "timeout_rate": "timeout",
}
SUMMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "timeout_rate",
    "mean_return",
    "mean_success_completion_time_seconds",
)
EFFICIENCY_PATHS = {
    "parameter_count": ("parameters", "model_plus_representation"),
    "wall_ms_per_learner_update": ("end_to_end_wall_ms_per_learner_update",),
    "inference_ms_per_action": ("inference", "mean_milliseconds_per_action"),
    "peak_gpu_memory_mb": ("peak_gpu_memory_mb",),
}
DIAGNOSTIC_KEYS = (
    "train/representation_loss",
    "train/graph_slt_loss",
    "train/graph_slt_ego_loss",
    "train/graph_slt_social_loss",
    "train/graph_slt_route_loss",
    "diagnostic/topology_attention_entropy",
    "diagnostic/latent_std",
    "diagnostic/graph_mean_edge_weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nested(payload: dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    return mean(finite), pstdev(finite) if len(finite) > 1 else 0.0


def _bootstrap_mean_interval(
    arrays: list[np.ndarray], *, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Hierarchical seed/episode bootstrap for one episode-level metric."""

    data = np.stack(arrays).astype(np.float64, copy=False)
    seed_count, episode_count = data.shape
    seed_indices = rng.integers(0, seed_count, size=(resamples, seed_count))
    episode_indices = rng.integers(
        0, episode_count, size=(resamples, seed_count, episode_count)
    )
    sampled = data[seed_indices[:, :, None], episode_indices]
    values = sampled.mean(axis=(1, 2))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_paired_delta_interval(
    left: list[np.ndarray],
    right: list[np.ndarray],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Paired hierarchical bootstrap preserving seed and episode pairing."""

    left_data = np.stack(left).astype(np.float64, copy=False)
    right_data = np.stack(right).astype(np.float64, copy=False)
    if left_data.shape != right_data.shape:
        raise ValueError("Paired bootstrap arrays must have identical shapes")
    seed_count, episode_count = left_data.shape
    seed_indices = rng.integers(0, seed_count, size=(resamples, seed_count))
    episode_indices = rng.integers(
        0, episode_count, size=(resamples, seed_count, episode_count)
    )
    sampled_left = left_data[seed_indices[:, :, None], episode_indices]
    sampled_right = right_data[seed_indices[:, :, None], episode_indices]
    deltas = (sampled_left - sampled_right).mean(axis=(1, 2))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _exact_sign_flip_pvalue(values: list[float]) -> float:
    """Two-sided exact sign-flip p-value over fixed scenario blocks."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.allclose(array, 0.0):
        return 1.0
    observed = abs(float(array.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=array.size):
        statistics.append(abs(float((array * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(pvalues[key]))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _episode_array(run: dict[str, Any], field: str) -> np.ndarray:
    return np.asarray(
        [float(bool(row[field])) for row in run["episode_records"]],
        dtype=np.float64,
    )


def _run_metric(run: dict[str, Any], metric: str) -> float | None:
    if metric == "mean_success_completion_time_seconds":
        value = run.get(metric)
    else:
        value = run.get("summary", {}).get(metric)
    return None if value is None else float(value)


def _load_run(
    record: dict[str, Any],
    *,
    method_config: dict[str, Any],
    profile: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    run_dir = Path(str(record.get("result_directory", ""))).resolve()
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    arguments_path = run_dir / "arguments.json"
    performance_path = run_dir / "performance_profile.json"
    diagnostics_path = run_dir / "training_diagnostics.json"
    missing_paths = [
        path for path in (detailed_path, arguments_path) if not path.is_file()
    ]
    if missing_paths:
        return None, [f"missing {path}" for path in missing_paths]
    detailed = _read_json(detailed_path)
    arguments_payload = _read_json(arguments_path)
    arguments = arguments_payload.get("requested_raw_steps", {})
    expected_algorithm = method_config["train_cli_algorithm"]
    if detailed.get("algorithm") != expected_algorithm:
        errors.append("algorithm mismatch")
    if detailed.get("scenario") != record.get("scenario"):
        errors.append("scenario mismatch")
    if int(detailed.get("evaluation_seed_start", -1)) != int(record["seed"]) + 10_000:
        errors.append("evaluation seed start mismatch")
    if int(detailed.get("summary", {}).get("episodes", -1)) != int(
        profile["test_episodes"]
    ):
        errors.append("episode count mismatch")
    if int(detailed.get("trained_raw_steps", -1)) != int(profile["raw_training_steps"]):
        errors.append("trained raw-step mismatch")
    if int(arguments.get("max_steps", -1)) != int(profile["raw_training_steps"]):
        errors.append("requested raw-step mismatch")
    if arguments.get("ego_control_profile") != environment["ego_control_profile"]:
        errors.append("ego control profile mismatch")
    if arguments.get("traffic_protocol") != environment["traffic_protocol"]:
        errors.append("training traffic protocol mismatch")
    if arguments.get("episode_limit_profile") != environment["episode_limit_profile"]:
        errors.append("training episode limit profile mismatch")
    if (
        arguments_payload.get("implementation_fidelity", {}).get("implementation_id")
        != method_config["implementation_id"]
    ):
        errors.append("implementation id mismatch")
    provenance = detailed.get("evaluation_provenance", {})
    if provenance.get("validated") is not True:
        errors.append("evaluation provenance is not validated")
    for field in (
        "traffic_protocol",
        "episode_limit_profile",
    ):
        if provenance.get(field) != environment[field]:
            errors.append(f"{field} mismatch")
    records = detailed.get("episode_records", [])
    if len(records) != int(profile["test_episodes"]):
        errors.append("episode record count mismatch")
    expected_episode_seeds = list(
        range(int(record["seed"]) + 10_000, int(record["seed"]) + 10_000 + len(records))
    )
    if [int(row.get("seed", -1)) for row in records] != expected_episode_seeds:
        errors.append("episode seed sequence mismatch")
    if not _finite_tree(detailed):
        errors.append("non-finite detailed evaluation")

    performance = _read_json(performance_path) if performance_path.is_file() else {}
    diagnostics = _read_json(diagnostics_path) if diagnostics_path.is_file() else {}
    if not performance:
        errors.append("missing performance profile")
    if not diagnostics:
        errors.append("missing training diagnostics")
    if performance and not _finite_tree(performance):
        errors.append("non-finite performance profile")
    if diagnostics and not _finite_tree(diagnostics):
        errors.append("non-finite training diagnostics")
    if errors:
        return None, errors
    return {
        **record,
        "run_dir": str(run_dir),
        "summary": detailed["summary"],
        "episode_records": records,
        "mean_success_completion_time_seconds": detailed.get(
            "mean_success_completion_time_seconds"
        ),
        "performance": performance,
        "diagnostics": diagnostics,
    }, []


def _validate_and_load(
    protocol: dict[str, Any],
    protocol_path: Path,
    profile_name: str,
    result_root: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    profile = protocol["profiles"][profile_name]
    summary_path = result_root / "summary" / "summary.json"
    errors: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "summary_path": str(summary_path),
        "snapshot_path": str(result_root / "protocol.snapshot.json"),
    }
    if not summary_path.is_file():
        return {}, [{"scope": "matrix", "error": "summary/summary.json is missing"}], metadata
    snapshot = result_root / "protocol.snapshot.json"
    if not snapshot.is_file():
        errors.append({"scope": "matrix", "error": "protocol snapshot is missing"})
    else:
        metadata["protocol_snapshot_sha256"] = _sha256(snapshot)
        if snapshot.read_bytes() != protocol_path.read_bytes():
            errors.append(
                {"scope": "matrix", "error": "protocol snapshot bytes do not match the systematic protocol"}
            )
    generic = _read_json(summary_path)
    metadata["generic_runs_completed"] = generic.get("runs_completed")
    metadata["generic_jobs_requested"] = generic.get("jobs_requested")
    expected_runs = int(protocol["matrix_contract"]["expected_training_runs"])
    if generic.get("profile") != profile_name:
        errors.append({"scope": "matrix", "error": "generic summary profile mismatch"})
    if str(generic.get("protocol_sha256", "")).upper() != _sha256(
        protocol_path
    ):
        errors.append({"scope": "matrix", "error": "generic summary protocol hash mismatch"})
    if int(generic.get("jobs_requested", -1)) != expected_runs:
        errors.append({"scope": "matrix", "error": "generic summary job count mismatch"})
    if int(generic.get("runs_completed", -1)) != expected_runs:
        errors.append(
            {"scope": "matrix", "error": "generic summary completion count mismatch"}
        )
    records = generic.get("runs", [])
    methods = protocol["supported_methods"]
    environment = protocol["environment_protocol"]
    loaded: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("method")),
            str(record.get("scenario")),
            int(record.get("seed", -1)),
        )
        if key in loaded:
            errors.append({"scope": list(key), "error": "duplicate run record"})
            continue
        method = methods.get(key[0])
        if method is None:
            errors.append({"scope": list(key), "error": "unknown method"})
            continue
        run, run_errors = _load_run(
            record,
            method_config=method,
            profile=profile,
            environment=environment,
        )
        if run_errors:
            errors.extend({"scope": list(key), "error": error} for error in run_errors)
        elif run is not None:
            loaded[key] = run
    return loaded, errors, metadata


def _pairing_errors(
    loaded: dict[tuple[str, str, int], dict[str, Any]],
    *,
    methods: list[str],
    scenarios: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            reference: list[tuple[int, Any]] | None = None
            for method in methods:
                run = loaded.get((method, scenario, seed))
                if run is None:
                    continue
                episode_contract = [
                    (int(row["seed"]), row.get("traffic_variant"))
                    for row in run["episode_records"]
                ]
                if reference is None:
                    reference = episode_contract
                elif episode_contract != reference:
                    errors.append(
                        {
                            "scope": [method, scenario, seed],
                            "error": "paired evaluation seed/traffic sequence mismatch",
                        }
                    )
    return errors


def _aggregate_cells(
    loaded: dict[tuple[str, str, int], dict[str, Any]],
    *,
    methods: list[str],
    scenarios: list[str],
    seeds: list[int],
    resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for method in methods:
            runs = [loaded[(method, scenario, seed)] for seed in seeds]
            row: dict[str, Any] = {
                "method": method,
                "scenario": scenario,
                "seeds_completed": len(runs),
                "episodes_total": sum(len(run["episode_records"]) for run in runs),
            }
            for metric in SUMMARY_METRICS:
                values = [value for run in runs if (value := _run_metric(run, metric)) is not None]
                metric_mean, metric_std = _mean_std(values)
                row[f"{metric}_mean"] = metric_mean
                row[f"{metric}_std"] = metric_std
                if metric in RATE_METRICS:
                    arrays = [
                        _episode_array(run, RATE_METRICS[metric]) for run in runs
                    ]
                    low, high = _bootstrap_mean_interval(
                        arrays, resamples=resamples, rng=rng
                    )
                    row[f"{metric}_bootstrap95_low"] = low
                    row[f"{metric}_bootstrap95_high"] = high
            for metric, path in EFFICIENCY_PATHS.items():
                values = [
                    float(value)
                    for run in runs
                    if (value := _nested(run["performance"], path)) is not None
                ]
                metric_mean, metric_std = _mean_std(values)
                row[f"{metric}_mean"] = metric_mean
                row[f"{metric}_std"] = metric_std
            for key in DIAGNOSTIC_KEYS:
                values = [
                    float(value)
                    for run in runs
                    if (
                        value := run["diagnostics"]
                        .get("statistics", {})
                        .get(key, {})
                        .get("last")
                    )
                    is not None
                ]
                if values:
                    value_mean, value_std = _mean_std(values)
                    safe_key = key.replace("/", "_")
                    row[f"{safe_key}_last_mean"] = value_mean
                    row[f"{safe_key}_last_std"] = value_std
            rows.append(row)
    return rows


def _aggregate_comparisons(
    protocol: dict[str, Any],
    loaded: dict[tuple[str, str, int], dict[str, Any]],
    *,
    scenarios: list[str],
    seeds: list[int],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    comparison_specs = protocol["matrix_contract"]["predeclared_comparisons"]
    by_id_scenario: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in comparison_specs:
        left_method, right_method = spec["left"], spec["right"]
        for scenario in scenarios:
            left_runs = [loaded[(left_method, scenario, seed)] for seed in seeds]
            right_runs = [loaded[(right_method, scenario, seed)] for seed in seeds]
            row: dict[str, Any] = {
                "comparison": spec["id"],
                "left": left_method,
                "right": right_method,
                "scenario": scenario,
            }
            for metric in SUMMARY_METRICS:
                seed_deltas = []
                for left_run, right_run in zip(left_runs, right_runs):
                    left_value = _run_metric(left_run, metric)
                    right_value = _run_metric(right_run, metric)
                    if left_value is not None and right_value is not None:
                        seed_deltas.append(left_value - right_value)
                delta_mean, delta_std = _mean_std(seed_deltas)
                row[f"{metric}_delta_mean"] = delta_mean
                row[f"{metric}_delta_std"] = delta_std
                if metric in RATE_METRICS:
                    left_arrays = [
                        _episode_array(run, RATE_METRICS[metric]) for run in left_runs
                    ]
                    right_arrays = [
                        _episode_array(run, RATE_METRICS[metric]) for run in right_runs
                    ]
                    low, high = _bootstrap_paired_delta_interval(
                        left_arrays,
                        right_arrays,
                        resamples=resamples,
                        rng=rng,
                    )
                    row[f"{metric}_delta_bootstrap95_low"] = low
                    row[f"{metric}_delta_bootstrap95_high"] = high
            rows.append(row)
            by_id_scenario[(spec["id"], scenario)] = row

    for spec in comparison_specs:
        scenario_rows = [by_id_scenario[(spec["id"], scenario)] for scenario in scenarios]
        macro: dict[str, Any] = {
            "comparison": spec["id"],
            "left": spec["left"],
            "right": spec["right"],
            "interpretation": spec["interpretation"],
            "scenarios": len(scenarios),
        }
        for metric in SUMMARY_METRICS:
            scenario_values = [
                float(row[f"{metric}_delta_mean"])
                for row in scenario_rows
                if row.get(f"{metric}_delta_mean") is not None
            ]
            metric_mean, metric_std = _mean_std(scenario_values)
            macro[f"{metric}_macro_delta"] = metric_mean
            macro[f"{metric}_scenario_std"] = metric_std
            if metric in RATE_METRICS and scenario_values:
                macro[f"{metric}_scenario_sign_flip_p"] = _exact_sign_flip_pvalue(
                    scenario_values
                )
        macro_rows.append(macro)

    corrected_specs = [
        spec for spec in comparison_specs if spec["id"] != "ppo_minus_sac"
    ]
    for metric in ("success_rate", "collision_rate"):
        pvalues = {
            spec["id"]: next(
                row[f"{metric}_scenario_sign_flip_p"]
                for row in macro_rows
                if row["comparison"] == spec["id"]
            )
            for spec in corrected_specs
        }
        adjusted = _holm_adjust(pvalues)
        for row in macro_rows:
            if row["comparison"] in adjusted:
                row[f"{metric}_holm_adjusted_p"] = adjusted[row["comparison"]]
    return rows, macro_rows


def _effect_classification(row: dict[str, Any]) -> str:
    success = float(row.get("success_rate_macro_delta") or 0.0)
    collision = float(row.get("collision_rate_macro_delta") or 0.0)
    if collision > 0 and success <= 0:
        return "directionally_worse_with_safety_regression"
    if success > 0 and collision <= 0:
        return "directionally_positive_without_safety_tradeoff"
    if success < 0 and collision >= 0:
        return "directionally_negative"
    if success > 0 and collision > 0:
        return "success_safety_tradeoff"
    if success < 0 and collision < 0:
        return "lower_success_lower_collision_tradeoff"
    return "no_directional_difference"


def _build_attribution(
    protocol: dict[str, Any], macro_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    interpretations = {
        spec["id"]: spec["interpretation"]
        for spec in protocol["matrix_contract"]["predeclared_comparisons"]
    }
    findings = []
    for row in macro_rows:
        findings.append(
            {
                "comparison": row["comparison"],
                "classification": _effect_classification(row),
                "success_rate_macro_delta": row.get("success_rate_macro_delta"),
                "collision_rate_macro_delta": row.get("collision_rate_macro_delta"),
                "mean_return_macro_delta": row.get("mean_return_macro_delta"),
                "success_rate_holm_adjusted_p": row.get(
                    "success_rate_holm_adjusted_p"
                ),
                "collision_rate_holm_adjusted_p": row.get(
                    "collision_rate_holm_adjusted_p"
                ),
                "identifiable_scope": interpretations[row["comparison"]],
            }
        )
    return {
        "evidence_bound": True,
        "matrix_complete": True,
        "topology_and_balancing_confound_recorded": True,
        "causal_language_rule": "Only the predeclared single-variable MST-to-MST+SLT contrast is directly component-identifying. Other contrasts are bounded to their explicitly listed joint changes.",
        "findings": findings,
        "deep_attribution_limits": [
            "TemporalGraph versus MST+SLT jointly changes interaction timing and the auxiliary objective structure.",
            "Full+BalancedSlots versus TemporalGraph jointly changes topology query and slot normalization.",
            "PPO versus SAC changes algorithm family, observation encoder, update rule, and action clock.",
            "Three seeds yield weak tail-resolution; Holm-adjusted sign-flip tests across six scenarios are conservative.",
        ],
    }


def _plot_primary_matrix(
    cell_rows: list[dict[str, Any]],
    *,
    methods: list[str],
    scenarios: list[str],
    labels: dict[str, str],
    output: Path,
) -> None:
    lookup = {(row["method"], row["scenario"]): row for row in cell_rows}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, metric, title, cmap in (
        (axes[0], "success_rate_mean", "Success rate ↑", "YlGn"),
        (axes[1], "collision_rate_mean", "Collision rate ↓", "YlOrRd"),
    ):
        matrix = np.asarray(
            [[lookup[(method, scenario)][metric] for scenario in scenarios] for method in methods],
            dtype=np.float64,
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(scenarios)), [labels[item] for item in scenarios], rotation=35, ha="right")
        axis.set_yticks(range(len(methods)), methods)
        for method_index in range(len(methods)):
            for scenario_index in range(len(scenarios)):
                value = matrix[method_index, scenario_index]
                axis.text(scenario_index, method_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_balanced_delta(
    comparison_rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    labels: dict[str, str],
    output: Path,
) -> None:
    selected = {
        row["scenario"]: row
        for row in comparison_rows
        if row["comparison"] == "full_balanced_minus_mst_slt"
    }
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.36
    success = np.asarray([selected[item]["success_rate_delta_mean"] for item in scenarios])
    collision = np.asarray([selected[item]["collision_rate_delta_mean"] for item in scenarios])
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.bar(x - width / 2, success, width, label="Δ success", color="#0072B2")
    axis.bar(x + width / 2, collision, width, label="Δ collision", color="#D55E00")
    axis.set_xticks(x, [labels[item] for item in scenarios], rotation=30, ha="right")
    axis.set_ylabel("Full+BalancedSlots − MST+SLT")
    axis.set_title("Paired three-seed primary-metric deltas")
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _format_metric(value: Any) -> str:
    return "TBD" if value is None else f"{float(value):.3f}"


def _write_report(
    path: Path,
    *,
    profile_name: str,
    matrix_complete: bool,
    completion: dict[str, Any],
    cell_rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
    attribution: dict[str, Any],
) -> None:
    lines = [
        "# 六方法 × 六场景系统实验报告",
        "",
        f"Profile: `{profile_name}`",
        f"Matrix complete: `{str(matrix_complete).lower()}`",
        f"Accepted runs: {completion['accepted_runs']}/{completion['expected_runs']}",
        "",
    ]
    if not matrix_complete:
        lines.extend(
            [
                "结果矩阵尚未完整。所有缺失单元保持 `TBD`，当前文件不形成方法优劣或机制归因结论。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 主结果",
                "",
                "| 场景 | 方法 | Success ↑ | Collision ↓ | Return ↑ |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in cell_rows:
            lines.append(
                "| {scenario} | {method} | {success} | {collision} | {ret} |".format(
                    scenario=row["scenario"],
                    method=row["method"],
                    success=_format_metric(row.get("success_rate_mean")),
                    collision=_format_metric(row.get("collision_rate_mean")),
                    ret=_format_metric(row.get("mean_return_mean")),
                )
            )
        lines.extend(
            [
                "",
                "## 预声明宏观对比",
                "",
                "| 对比（左−右） | ΔSuccess | ΔCollision | ΔReturn | 归因分类 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        classifications = {
            row["comparison"]: row["classification"]
            for row in attribution["findings"]
        }
        for row in macro_rows:
            lines.append(
                "| {comparison} | {success} | {collision} | {ret} | {classification} |".format(
                    comparison=row["comparison"],
                    success=_format_metric(row.get("success_rate_macro_delta")),
                    collision=_format_metric(row.get("collision_rate_macro_delta")),
                    ret=_format_metric(row.get("mean_return_macro_delta")),
                    classification=classifications[row["comparison"]],
                )
            )
        lines.extend(
            [
                "",
                "## 归因边界",
                "",
                "`TemporalGraph → Full+BalancedSlots` 同时改变拓扑查询与槽位归一化，不能单独识别拓扑贡献。PPO/SAC 对比也不是单组件消融。统计、诊断和效率证据不一致时，以不支持因果解释处理。",
                "",
                "![Primary matrix](figures/primary_metrics_matrix.png)",
                "",
                "![Balanced deltas](figures/full_balanced_vs_mst_slt_deltas.png)",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    *,
    protocol_path: Path,
    profile_name: str,
    result_root: Path,
    output_dir: Path,
    require_complete: bool,
) -> tuple[dict[str, Any], int]:
    protocol = _read_json(protocol_path)
    if profile_name not in protocol["profiles"]:
        raise ValueError(f"Unknown profile {profile_name!r}")
    methods = list(protocol["matrix_contract"]["method_order"])
    scenarios = list(protocol["matrix_contract"]["scenario_order"])
    seeds = list(protocol["matrix_contract"]["training_seeds"])
    expected_keys = {
        (method, scenario, seed)
        for method in methods
        for scenario in scenarios
        for seed in seeds
    }
    loaded, errors, metadata = _validate_and_load(
        protocol, protocol_path, profile_name, result_root
    )
    errors.extend(
        _pairing_errors(
            loaded, methods=methods, scenarios=scenarios, seeds=seeds
        )
    )
    missing = sorted(expected_keys - set(loaded))
    unexpected = sorted(set(loaded) - expected_keys)
    if unexpected:
        errors.extend(
            {"scope": list(key), "error": "unexpected matrix run"}
            for key in unexpected
        )
    matrix_complete = not errors and not missing and len(loaded) == len(expected_keys)
    cell_status = []
    for scenario in scenarios:
        for method in methods:
            present = sorted(
                seed for seed in seeds if (method, scenario, seed) in loaded
            )
            cell_status.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "status": "complete" if present == seeds else "TBD",
                    "seeds_present": present,
                    "seeds_expected": seeds,
                }
            )
    completion = {
        "expected_runs": len(expected_keys),
        "accepted_runs": len(loaded),
        "missing_runs": [list(key) for key in missing],
        "errors": errors,
        "cells": cell_status,
    }
    base_summary: dict[str, Any] = {
        "schema_version": "ccfa.systematic-matrix-analysis/v1",
        "profile": profile_name,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "result_root": str(result_root),
        "matrix_complete": matrix_complete,
        "fabricated_values": False,
        "completion": completion,
        "source_metadata": metadata,
    }
    if not matrix_complete:
        base_summary["cell_metrics"] = [
            {**row, "metrics": "TBD"} for row in cell_status
        ]
        attribution = {
            "evidence_bound": False,
            "matrix_complete": False,
            "topology_and_balancing_confound_recorded": True,
            "reason": "The 108-run artifact contract is incomplete or incompatible; attribution is withheld.",
        }
        _write_json(output_dir / "systematic_summary.json", base_summary)
        _write_json(output_dir / "attribution.json", attribution)
        _write_report(
            output_dir / "FINAL_REPORT.md",
            profile_name=profile_name,
            matrix_complete=False,
            completion=completion,
            cell_rows=[],
            macro_rows=[],
            attribution=attribution,
        )
        return base_summary, 2 if require_complete else 0

    statistics = protocol["statistics"]
    resamples = int(statistics["bootstrap_resamples"])
    rng = np.random.default_rng(int(statistics["bootstrap_seed"]))
    cell_rows = _aggregate_cells(
        loaded,
        methods=methods,
        scenarios=scenarios,
        seeds=seeds,
        resamples=resamples,
        rng=rng,
    )
    comparison_rows, macro_rows = _aggregate_comparisons(
        protocol,
        loaded,
        scenarios=scenarios,
        seeds=seeds,
        resamples=resamples,
        rng=rng,
    )
    attribution = _build_attribution(protocol, macro_rows)
    base_summary.update(
        {
            "statistics": statistics,
            "cell_metrics": cell_rows,
            "comparison_metrics": comparison_rows,
            "macro_comparisons": macro_rows,
        }
    )
    _write_json(output_dir / "systematic_summary.json", base_summary)
    _write_json(output_dir / "attribution.json", attribution)
    _write_csv(output_dir / "tables" / "cell_metrics.csv", cell_rows)
    _write_csv(output_dir / "tables" / "comparison_metrics.csv", comparison_rows)
    _write_csv(output_dir / "tables" / "macro_comparisons.csv", macro_rows)
    figures = output_dir / "figures"
    _plot_primary_matrix(
        cell_rows,
        methods=methods,
        scenarios=scenarios,
        labels=protocol["scenario_labels"],
        output=figures / "primary_metrics_matrix.png",
    )
    _plot_balanced_delta(
        comparison_rows,
        scenarios=scenarios,
        labels=protocol["scenario_labels"],
        output=figures / "full_balanced_vs_mst_slt_deltas.png",
    )
    _write_report(
        output_dir / "FINAL_REPORT.md",
        profile_name=profile_name,
        matrix_complete=True,
        completion=completion,
        cell_rows=cell_rows,
        macro_rows=macro_rows,
        attribution=attribution,
    )
    return base_summary, 0


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result_root = args.result_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_root / "systematic_analysis"
    )
    summary, returncode = analyze(
        protocol_path=args.protocol.resolve(),
        profile_name=args.profile,
        result_root=result_root,
        output_dir=output_dir,
        require_complete=args.require_complete,
    )
    print(
        json.dumps(
            {
                "matrix_complete": summary["matrix_complete"],
                "accepted_runs": summary["completion"]["accepted_runs"],
                "expected_runs": summary["completion"]["expected_runs"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
