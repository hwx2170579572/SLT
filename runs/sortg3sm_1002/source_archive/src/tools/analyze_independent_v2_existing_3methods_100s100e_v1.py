"""Build validated per-seed matrices and descriptive v4.8 case selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (  # noqa: E402
    evaluate_independent_v2_existing_models_100s100e_v1 as study,
)
from tools import (  # noqa: E402
    run_independent_v2_existing_models_100s100e_v1 as runner,
)


DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_v2_existing_3methods_100seeds_100episodes_v1"
    / "protocol.json"
)
DEFAULT_SELECTION_PROTOCOL_PATH = (
    DEFAULT_PROTOCOL_PATH.parent / "seed_selection_protocol_v1.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_iv2_eval_3m_100s100e_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "comparison" / "seed_analysis_v1"
PRIMARY_METRICS = {"success_rate": "higher", "collision_rate": "lower"}
SECONDARY_METRICS = {
    "mean_return": "higher",
    "off_route_rate": "lower",
    "timeout_rate": "lower",
    "mean_success_completion_time_seconds": "lower",
}
DESCRIPTIVE_METRICS = (
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "timeout_rate",
    "mean_return",
    "std_return",
    "mean_decision_steps",
    "mean_raw_steps",
    "mean_success_completion_time_seconds",
    "std_success_completion_time_seconds",
)
TOLERANCE = 1e-12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".analysis-{os.getpid()}.tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _metric(block: dict[str, Any], name: str) -> float | None:
    if name in {
        "mean_success_completion_time_seconds",
        "std_success_completion_time_seconds",
    }:
        value = block.get(name)
    else:
        value = block["summary"].get(name)
    return None if value is None else float(value)


def _directional_advantage(
    candidate: float | None, baseline: float | None, direction: str
) -> float | None:
    if candidate is None or baseline is None:
        return None
    if direction == "higher":
        return float(candidate - baseline)
    if direction == "lower":
        return float(baseline - candidate)
    raise ValueError(f"unsupported metric direction {direction!r}")


def _comparison_outcome(advantage: float | None) -> str:
    if advantage is None:
        return "not_comparable"
    if advantage > TOLERANCE:
        return "win"
    if advantage < -TOLERANCE:
        return "loss"
    return "tie"


def _paired_episode_counts(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, int]:
    candidate_records = candidate["episode_records"]
    baseline_records = baseline["episode_records"]
    candidate_seeds = [int(record["seed"]) for record in candidate_records]
    baseline_seeds = [int(record["seed"]) for record in baseline_records]
    if candidate_seeds != baseline_seeds:
        raise ValueError("paired episode seeds differ between v4.8 and baseline")

    counts = {
        "paired_episodes": len(candidate_records),
        "success_wins": 0,
        "success_losses": 0,
        "success_ties": 0,
        "collision_improvements": 0,
        "collision_regressions": 0,
        "collision_ties": 0,
        "return_wins": 0,
        "return_losses": 0,
        "return_ties": 0,
    }
    for candidate_record, baseline_record in zip(
        candidate_records, baseline_records, strict=True
    ):
        candidate_success = bool(candidate_record["success"])
        baseline_success = bool(baseline_record["success"])
        if candidate_success and not baseline_success:
            counts["success_wins"] += 1
        elif baseline_success and not candidate_success:
            counts["success_losses"] += 1
        else:
            counts["success_ties"] += 1

        candidate_collision = bool(candidate_record["collision"])
        baseline_collision = bool(baseline_record["collision"])
        if baseline_collision and not candidate_collision:
            counts["collision_improvements"] += 1
        elif candidate_collision and not baseline_collision:
            counts["collision_regressions"] += 1
        else:
            counts["collision_ties"] += 1

        candidate_return = float(candidate_record["episode_return"])
        baseline_return = float(baseline_record["episode_return"])
        if candidate_return > baseline_return + TOLERANCE:
            counts["return_wins"] += 1
        elif candidate_return < baseline_return - TOLERANCE:
            counts["return_losses"] += 1
        else:
            counts["return_ties"] += 1
    return counts


def _pairwise_row(
    *,
    logical_index: int,
    scenario: str,
    baseline_method: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    seed_block = candidate["test_seed_block"]
    row: dict[str, Any] = {
        "logical_test_seed_index": logical_index,
        "episode_seed_start": int(seed_block["episode_seed_start"]),
        "episode_seed_end": int(seed_block["episode_seed_end"]),
        "scenario": scenario,
        "candidate_method": "v4_8",
        "candidate_label": protocol["methods"]["v4_8"]["display_label"],
        "baseline_method": baseline_method,
        "baseline_label": protocol["methods"][baseline_method]["display_label"],
    }
    advantages: dict[str, float | None] = {}
    for metric, direction in {**PRIMARY_METRICS, **SECONDARY_METRICS}.items():
        candidate_value = _metric(candidate, metric)
        baseline_value = _metric(baseline, metric)
        advantage = _directional_advantage(candidate_value, baseline_value, direction)
        row[f"v4_8_{metric}"] = candidate_value
        row[f"baseline_{metric}"] = baseline_value
        row[f"advantage_{metric}"] = advantage
        row[f"outcome_{metric}"] = _comparison_outcome(advantage)
        advantages[metric] = advantage

    success_advantage = advantages["success_rate"]
    collision_reduction = advantages["collision_rate"]
    assert success_advantage is not None and collision_reduction is not None
    row["primary_margin"] = (success_advantage + collision_reduction) / 2.0
    row["primary_dominance"] = (
        success_advantage >= -TOLERANCE
        and collision_reduction >= -TOLERANCE
        and (
            success_advantage > TOLERANCE
            or collision_reduction > TOLERANCE
        )
    )
    row["primary_regression"] = (
        success_advantage < -TOLERANCE
        or collision_reduction < -TOLERANCE
    )

    secondary_outcomes = [
        _comparison_outcome(advantages[metric]) for metric in SECONDARY_METRICS
    ]
    row["secondary_wins"] = secondary_outcomes.count("win")
    row["secondary_losses"] = secondary_outcomes.count("loss")
    row["secondary_ties"] = secondary_outcomes.count("tie")
    row["secondary_not_comparable"] = secondary_outcomes.count("not_comparable")
    row["secondary_comparable"] = 4 - row["secondary_not_comparable"]
    row.update(_paired_episode_counts(candidate, baseline))
    return row


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _mean_optional(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return _mean([float(value) for value in values if value is not None])


def _selection_tier(row: dict[str, Any]) -> str:
    if (
        int(row["balanced_primary_dominance_count"]) == 3
        and int(row["total_primary_regression_count"]) == 0
    ):
        return "A"
    if (
        int(row["balanced_primary_dominance_count"]) >= 2
        and int(row["total_primary_regression_count"]) <= 1
    ):
        return "B"
    return "C"


def _rank_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        secondary_rate = row["secondary_net_win_rate"]
        return (
            -int(row["balanced_primary_dominance_count"]),
            int(row["total_primary_regression_count"]),
            -int(row["total_primary_dominance_count"]),
            -float(row["worst_baseline_macro_primary_margin"]),
            -float(row["mean_primary_margin"]),
            -float(secondary_rate) if secondary_rate is not None else math.inf,
            -float(row["worst_baseline_macro_return_advantage"]),
            int(row["logical_test_seed_index"]),
        )

    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted(rows, key=key), start=1):
        ranked.append(
            {
                "selection_rank": rank,
                "selected_top10": rank <= 10,
                "selection_tier": _selection_tier(row),
                **row,
            }
        )
    return ranked


def _block_row(
    *,
    block: dict[str, Any],
    path: Path,
    block_sha256: str,
) -> dict[str, Any]:
    seed_block = block["test_seed_block"]
    return {
        "logical_test_seed_index": int(seed_block["logical_index"]),
        "episode_seed_start": int(seed_block["episode_seed_start"]),
        "episode_seed_end": int(seed_block["episode_seed_end"]),
        "episodes": int(seed_block["episode_count"]),
        "method": block["method"],
        "display_label": block["display_label"],
        "scenario": block["scenario"],
        "training_seed": int(block["training_seed"]),
        "model_basename": block["model_basename"],
        "model_sha256": block["model_sha256"],
        "successful_episodes": int(block["successful_episodes"]),
        **{metric: _metric(block, metric) for metric in DESCRIPTIVE_METRICS},
        "sumo_step_seconds": float(block["sumo_step_seconds"]),
        "wall_seconds": (
            None if block.get("wall_seconds") is None else float(block["wall_seconds"])
        ),
        "completed_at_utc": block.get("completed_at_utc"),
        "block_sha256": block_sha256,
        "source_block_relpath": _relative_path(path),
        "computed_from_real_run": block.get("computed_from_real_run") is True,
        "fabricated_values": block.get("fabricated_values") is True,
        "training_performed": block.get("training_performed") is True,
    }


def _seed_summary_row(
    *,
    logical_index: int,
    blocks: dict[tuple[int, str, str], dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    scenarios = list(protocol["matrix"]["scenario_order"])
    methods = list(protocol["matrix"]["method_order"])
    seed_pairwise = [
        row
        for row in pairwise_rows
        if int(row["logical_test_seed_index"]) == logical_index
    ]
    if len(seed_pairwise) != len(scenarios) * 2:
        raise ValueError(f"seed {logical_index} does not have six pairwise rows")

    first = blocks[(logical_index, methods[0], scenarios[0])]["test_seed_block"]
    row: dict[str, Any] = {
        "logical_test_seed_index": logical_index,
        "episode_seed_start": int(first["episode_seed_start"]),
        "episode_seed_end": int(first["episode_seed_end"]),
    }
    for method in methods:
        for metric in DESCRIPTIVE_METRICS:
            row[f"{method}_macro_{metric}"] = _mean_optional(
                [_metric(blocks[(logical_index, method, scenario)], metric) for scenario in scenarios]
            )

    dominance_by_baseline: dict[str, int] = {}
    regression_by_baseline: dict[str, int] = {}
    macro_margin_by_baseline: dict[str, float] = {}
    macro_return_by_baseline: dict[str, float] = {}
    for baseline_method in ("mst_slt", "temporal_graph"):
        baseline_rows = [
            pair
            for pair in seed_pairwise
            if pair["baseline_method"] == baseline_method
        ]
        dominance_by_baseline[baseline_method] = sum(
            bool(pair["primary_dominance"]) for pair in baseline_rows
        )
        regression_by_baseline[baseline_method] = sum(
            bool(pair["primary_regression"]) for pair in baseline_rows
        )
        macro_margin_by_baseline[baseline_method] = _mean(
            [float(pair["primary_margin"]) for pair in baseline_rows]
        )
        macro_return_by_baseline[baseline_method] = _mean(
            [float(pair["advantage_mean_return"]) for pair in baseline_rows]
        )
        row[f"primary_dominance_count_vs_{baseline_method}"] = dominance_by_baseline[
            baseline_method
        ]
        row[f"primary_regression_count_vs_{baseline_method}"] = regression_by_baseline[
            baseline_method
        ]
        row[f"macro_primary_margin_vs_{baseline_method}"] = macro_margin_by_baseline[
            baseline_method
        ]
        row[f"macro_return_advantage_vs_{baseline_method}"] = macro_return_by_baseline[
            baseline_method
        ]
        for count_name in (
            "success_wins",
            "success_losses",
            "success_ties",
            "collision_improvements",
            "collision_regressions",
            "collision_ties",
            "return_wins",
            "return_losses",
            "return_ties",
        ):
            row[f"paired_{count_name}_vs_{baseline_method}"] = sum(
                int(pair[count_name]) for pair in baseline_rows
            )

    secondary_wins = sum(int(pair["secondary_wins"]) for pair in seed_pairwise)
    secondary_losses = sum(int(pair["secondary_losses"]) for pair in seed_pairwise)
    secondary_ties = sum(int(pair["secondary_ties"]) for pair in seed_pairwise)
    secondary_comparable = sum(
        int(pair["secondary_comparable"]) for pair in seed_pairwise
    )
    row.update(
        {
            "balanced_primary_dominance_count": min(
                dominance_by_baseline.values()
            ),
            "total_primary_dominance_count": sum(
                dominance_by_baseline.values()
            ),
            "total_primary_regression_count": sum(
                regression_by_baseline.values()
            ),
            "universal_primary_dominance": (
                min(dominance_by_baseline.values()) == len(scenarios)
                and sum(regression_by_baseline.values()) == 0
            ),
            "worst_baseline_macro_primary_margin": min(
                macro_margin_by_baseline.values()
            ),
            "mean_primary_margin": _mean(
                [float(pair["primary_margin"]) for pair in seed_pairwise]
            ),
            "secondary_wins": secondary_wins,
            "secondary_losses": secondary_losses,
            "secondary_ties": secondary_ties,
            "secondary_comparable": secondary_comparable,
            "secondary_net_win_rate": (
                (secondary_wins - secondary_losses) / secondary_comparable
                if secondary_comparable
                else None
            ),
            "worst_baseline_macro_return_advantage": min(
                macro_return_by_baseline.values()
            ),
        }
    )
    return row


def build_analysis(
    *,
    protocol_path: Path,
    selection_protocol_path: Path,
    result_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = study.load_protocol(protocol_path)
    selection_protocol = _load_json(selection_protocol_path)
    if selection_protocol["experiment_protocol_sha256"] != protocol["_sha256"]:
        raise ValueError("selection protocol is bound to a different experiment hash")
    if tuple(protocol["matrix"]["method_order"]) != (
        "mst_slt",
        "temporal_graph",
        "v4_8",
    ):
        raise ValueError("analysis requires the frozen three-method protocol")

    status = runner._status_payload(protocol, result_root.resolve(), "comparison")
    if (
        int(status["completed_blocks"]) != int(status["expected_blocks"])
        or int(status["invalid_jobs"]) != 0
    ):
        raise RuntimeError(
            "formal evaluation is incomplete: "
            f"{status['completed_blocks']}/{status['expected_blocks']} valid blocks, "
            f"{status['invalid_jobs']} invalid jobs"
        )

    profile = study.profile_setting(protocol, "comparison")
    logical_indices = list(study.logical_test_seed_indices(profile))
    expected_episodes = int(profile["episodes_per_test_seed"])
    blocks: dict[tuple[int, str, str], dict[str, Any]] = {}
    block_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str, str]] = set()
    model_hashes: set[str] = set()
    block_hashes: set[str] = set()
    for job in study.build_jobs(protocol, "comparison"):
        source = study.resolve_source_model(protocol, job.method, job.scenario)
        model_hashes.add(source["model_sha256"])
        for logical_index in logical_indices:
            path = study.block_path(result_root.resolve(), job, logical_index)
            block = study.read_valid_block(
                path,
                protocol_sha256=protocol["_sha256"],
                job=job,
                source_model_sha256=source["model_sha256"],
                logical_index=logical_index,
                expected_seed_start=study.episode_seed_start(profile, logical_index),
                expected_episodes=expected_episodes,
            )
            key = (logical_index, job.method, job.scenario)
            if key in seen_keys:
                raise ValueError(f"duplicate block key: {key}")
            seen_keys.add(key)
            block_hash = study.sha256(path)
            block_hashes.add(block_hash)
            blocks[key] = block
            block_rows.append(
                _block_row(block=block, path=path, block_sha256=block_hash)
            )

    expected_block_count = (
        len(logical_indices)
        * len(protocol["matrix"]["method_order"])
        * len(protocol["matrix"]["scenario_order"])
    )
    if len(block_rows) != expected_block_count:
        raise ValueError(
            f"expected {expected_block_count} block rows, got {len(block_rows)}"
        )
    if any(
        (not row["computed_from_real_run"])
        or row["fabricated_values"]
        or row["training_performed"]
        for row in block_rows
    ):
        raise ValueError("one or more blocks violate real-run/no-training provenance")

    for logical_index in logical_indices:
        seed_ranges = {
            (
                int(block["test_seed_block"]["episode_seed_start"]),
                int(block["test_seed_block"]["episode_seed_end"]),
            )
            for (index, _, _), block in blocks.items()
            if index == logical_index
        }
        if len(seed_ranges) != 1:
            raise ValueError(f"seed block {logical_index} is not paired across all cells")

    pairwise_rows: list[dict[str, Any]] = []
    for logical_index in logical_indices:
        for scenario in protocol["matrix"]["scenario_order"]:
            candidate = blocks[(logical_index, "v4_8", scenario)]
            for baseline_method in ("mst_slt", "temporal_graph"):
                pairwise_rows.append(
                    _pairwise_row(
                        logical_index=logical_index,
                        scenario=scenario,
                        baseline_method=baseline_method,
                        candidate=candidate,
                        baseline=blocks[(logical_index, baseline_method, scenario)],
                        protocol=protocol,
                    )
                )

    seed_rows = [
        _seed_summary_row(
            logical_index=logical_index,
            blocks=blocks,
            pairwise_rows=pairwise_rows,
            protocol=protocol,
        )
        for logical_index in logical_indices
    ]
    ranked_rows = _rank_seed_rows(seed_rows)
    top10 = ranked_rows[:10]
    top10_indices = {int(row["logical_test_seed_index"]) for row in top10}
    rank_by_index = {
        int(row["logical_test_seed_index"]): int(row["selection_rank"])
        for row in top10
    }
    tier_by_index = {
        int(row["logical_test_seed_index"]): row["selection_tier"] for row in top10
    }
    top10_detail = [
        {
            "selection_rank": rank_by_index[int(row["logical_test_seed_index"])],
            "selection_tier": tier_by_index[int(row["logical_test_seed_index"])],
            **row,
        }
        for row in block_rows
        if int(row["logical_test_seed_index"]) in top10_indices
    ]
    method_order = {
        method: index for index, method in enumerate(protocol["matrix"]["method_order"])
    }
    scenario_order = {
        scenario: index
        for index, scenario in enumerate(protocol["matrix"]["scenario_order"])
    }
    top10_detail.sort(
        key=lambda row: (
            int(row["selection_rank"]),
            scenario_order[row["scenario"]],
            method_order[row["method"]],
        )
    )

    overall_summary = runner._summarize(
        protocol=protocol,
        result_root=result_root.resolve(),
        profile_name="comparison",
        require_complete=True,
    )
    selection_hash = study.sha256(selection_protocol_path)
    data_quality = {
        "status": "ready_to_share",
        "expected_block_rows": 900,
        "validated_block_rows": len(block_rows),
        "unique_block_keys": len(seen_keys),
        "unique_block_hashes": len(block_hashes),
        "expected_total_episode_records": 90000,
        "validated_total_episode_records": sum(
            int(row["episodes"]) for row in block_rows
        ),
        "logical_seed_count": len(logical_indices),
        "method_count": len(protocol["matrix"]["method_order"]),
        "scenario_count": len(protocol["matrix"]["scenario_order"]),
        "pairwise_row_count": len(pairwise_rows),
        "seed_ranges_aligned_across_all_nine_cells": True,
        "all_blocks_computed_from_real_runs": True,
        "fabricated_values_found": 0,
        "training_performed_blocks": 0,
        "validated_source_model_hashes": len(model_hashes),
        "invalid_blocks": 0,
        "missing_blocks": 0,
    }
    generated_at = _utc_now()
    payloads = {
        "block_matrix.json": {
            "schema_version": "independent-v2-three-method-block-matrix/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            "row_grain": "logical_test_seed_index x method x scenario",
            "rows": sorted(
                block_rows,
                key=lambda row: (
                    int(row["logical_test_seed_index"]),
                    scenario_order[row["scenario"]],
                    method_order[row["method"]],
                ),
            ),
        },
        "pairwise_matrix.json": {
            "schema_version": "independent-v2-three-method-pairwise-matrix/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            "row_grain": "logical_test_seed_index x scenario x baseline",
            "positive_advantage_means": "v4.8 is better under the frozen metric direction",
            "rows": pairwise_rows,
        },
        "seed_ranking.json": {
            "schema_version": "independent-v2-three-method-seed-ranking/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            "selection_protocol_sha256": selection_hash,
            "row_grain": "logical_test_seed_index aggregated equally across scenarios",
            "rows": ranked_rows,
        },
        "top10_seed_matrix.json": {
            "schema_version": "independent-v2-three-method-top10-matrix/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            "selection_protocol_sha256": selection_hash,
            "confirmatory_use_forbidden": True,
            "top10": top10,
            "detail_rows": top10_detail,
        },
        "overall_results.json": {
            "schema_version": "independent-v2-three-method-overall-results/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            "method_scenario": overall_summary["method_scenario"],
            "method_macro": overall_summary["method_macro"],
        },
        "data_quality_report.json": {
            "schema_version": "independent-v2-three-method-data-quality/v1",
            "generated_at_utc": generated_at,
            "protocol_sha256": protocol["_sha256"],
            **data_quality,
        },
    }
    output_dir = output_dir.resolve()
    for filename, payload in payloads.items():
        _write_json_atomic(output_dir / filename, payload)
    manifest = {
        "schema_version": "independent-v2-three-method-seed-analysis-manifest/v1",
        "status": "complete",
        "generated_at_utc": generated_at,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "training_performed": False,
        "protocol_path": _relative_path(protocol_path),
        "protocol_sha256": protocol["_sha256"],
        "selection_protocol_path": _relative_path(selection_protocol_path),
        "selection_protocol_sha256": selection_hash,
        "result_root": _relative_path(result_root),
        "data_quality": data_quality,
        "selected_logical_test_seed_indices": [
            int(row["logical_test_seed_index"]) for row in top10
        ],
        "selection_tier_counts_in_top10": {
            tier: sum(row["selection_tier"] == tier for row in top10)
            for tier in ("A", "B", "C")
        },
        "outputs": {
            filename: _relative_path(output_dir / filename) for filename in payloads
        },
        "publication_guardrail": selection_protocol["publication_guardrail"],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest["analysis_manifest"] = _relative_path(manifest_path)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--selection-protocol", type=Path, default=DEFAULT_SELECTION_PROTOCOL_PATH
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    try:
        manifest = build_analysis(
            protocol_path=args.protocol,
            selection_protocol_path=args.selection_protocol,
            result_root=args.result_root,
            output_dir=args.output_dir,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(
            _json_text(
                {
                    "status": "blocked",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1
    print(_json_text(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_directional_advantage",
    "_pairwise_row",
    "_rank_seed_rows",
    "_selection_tier",
    "build_analysis",
    "main",
]
