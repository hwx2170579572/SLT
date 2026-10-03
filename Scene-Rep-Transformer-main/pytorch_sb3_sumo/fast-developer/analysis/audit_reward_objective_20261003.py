"""Read-only audit of saved sorted_depart4p0 reward/evaluation records.

This script reads JSON/JSONL/GZIP telemetry only. It does not import the
training code, load a checkpoint, start SUMO, or write output files. Run it from
any working directory with the project's Python interpreter; results print to
stdout as JSON.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[4]
GAMMA = 0.99

RUNS: dict[str, Path] = {
    "old_st": Path(
        "Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/"
        "sac_mlp_d1_st__intersection_sorted_depart4p0"
    ),
    "old_st_rt": Path(
        "runs/sortlr_1003_retry01/"
        "sac_mlp_d1_st_rt__intersection_sorted_depart4p0"
    ),
    "repair_st": Path(
        "runs/d1_contractfix_20261003/st/"
        "sac_mlp_d1_st_contractfix_v1__intersection_sorted_depart4p0"
    ),
    "repair_st_rt": Path(
        "runs/d1_contractfix_20261003/st_rt/"
        "sac_mlp_d1_st_rt_contractfix_v1__intersection_sorted_depart4p0"
    ),
}

OUTCOMES = ("success", "collision", "off_route", "timeout")
COMPONENTS = (
    "reward_success",
    "reward_collision",
    "reward_off_route",
    "reward_timeout",
    "reward_step_cost",
    "reward_progress",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def classify(record: dict[str, Any]) -> str:
    active = [name for name in OUTCOMES if bool(record.get(name, False))]
    if len(active) != 1:
        return "unclassified_or_nonexclusive"
    return active[0]


def summarize_outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[classify(record)].append(record)

    result: dict[str, Any] = {}
    for name in (*OUTCOMES, "unclassified_or_nonexclusive"):
        group = buckets.get(name, [])
        if not group:
            result[name] = {"n": 0}
            continue
        returns = [float(row["episode_return"]) for row in group if finite_number(row.get("episode_return"))]
        component_means: dict[str, float | None] = {}
        for key in COMPONENTS:
            values = [float(row[key]) for row in group if finite_number(row.get(key))]
            component_means[key] = fmean(values) if values else None
        result[name] = {
            "n": len(group),
            "episode_return_mean": fmean(returns) if returns else None,
            "episode_return_min": min(returns) if returns else None,
            "episode_return_max": max(returns) if returns else None,
            "component_means_per_episode": component_means,
        }

    successes = buckets.get("success", [])
    failures = [
        record
        for label in (*OUTCOMES[1:], "unclassified_or_nonexclusive")
        for record in buckets.get(label, [])
    ]
    n_pairs = len(successes) * len(failures)
    greater_pairs = sum(
        float(failure["episode_return"]) > float(success["episode_return"])
        for success in successes
        for failure in failures
        if finite_number(success.get("episode_return"))
        and finite_number(failure.get("episode_return"))
    )
    result["failure_vs_success_return_pairs"] = {
        "strict_failure_greater_than_success": greater_pairs,
        "all_pairs": n_pairs,
        "definition": "all outcome-labeled failure episode returns compared pairwise with success returns; strict >",
    }
    return result


def discounted_decision_returns(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "diagnostics/eval/decisions.jsonl.gz"
    accum: dict[int, dict[str, Any]] = defaultdict(lambda: {"n": 0, "discounted": 0.0, "undiscounted": 0.0, "decisions": []})
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = int(row["episode"]) - 1  # telemetry is 1-based; evaluation_results is 0-based
            decision = int(row["decision"])
            reward = float(row["reward_policy"])
            bucket = accum[episode_id]
            bucket["n"] += 1
            bucket["undiscounted"] += reward
            bucket["discounted"] += (GAMMA ** (decision - 1)) * reward
            bucket["decisions"].append(decision)

    by_outcome: dict[str, list[float]] = defaultdict(list)
    by_outcome_undiscounted: dict[str, list[float]] = defaultdict(list)
    reconciliation_errors: list[float] = []
    decision_count_mismatches: list[int] = []
    log_episode_ids = set(accum)
    result_episode_ids = {int(record["episode"]) for record in records}
    for record in records:
        episode_id = int(record["episode"])
        bucket = accum.get(episode_id)
        if bucket is None:
            continue
        if bucket["n"] != int(record.get("decision_steps", -1)):
            decision_count_mismatches.append(episode_id)
        if sorted(bucket["decisions"]) != list(range(1, bucket["n"] + 1)):
            decision_count_mismatches.append(episode_id)
        if finite_number(record.get("episode_return")):
            reconciliation_errors.append(abs(bucket["undiscounted"] - float(record["episode_return"])))
        label = classify(record)
        by_outcome[label].append(bucket["discounted"])
        by_outcome_undiscounted[label].append(bucket["undiscounted"])

    summaries: dict[str, Any] = {}
    for name in (*OUTCOMES, "unclassified_or_nonexclusive"):
        values = by_outcome.get(name, [])
        raw = by_outcome_undiscounted.get(name, [])
        summaries[name] = {
            "n": len(values),
            "discounted_return_mean": fmean(values) if values else None,
            "discounted_return_min": min(values) if values else None,
            "discounted_return_max": max(values) if values else None,
            "undiscounted_log_return_mean": fmean(raw) if raw else None,
        }
    successes = by_outcome.get("success", [])
    failures = [value for name in (*OUTCOMES[1:], "unclassified_or_nonexclusive") for value in by_outcome.get(name, [])]
    result_pairs = len(successes) * len(failures)
    greater_pairs = sum(failure > success for success in successes for failure in failures)
    return {
        "source": path.relative_to(WORKSPACE).as_posix(),
        "gamma_per_decision_for_offline_metric": GAMMA,
        "definition": "sum_t gamma^(decision_index-1) * diagnostics/eval/decisions.jsonl.gz reward_policy; reward-only realized trajectory return",
        "episodes_in_eval_results": len(records),
        "episodes_in_decision_log": len(log_episode_ids),
        "missing_eval_episode_ids_in_decision_log": sorted(result_episode_ids - log_episode_ids),
        "extra_episode_ids_in_decision_log": sorted(log_episode_ids - result_episode_ids),
        "decision_count_or_sequence_mismatch_episode_ids": sorted(set(decision_count_mismatches)),
        "max_abs_undiscounted_reconciliation_error": max(reconciliation_errors, default=None),
        "outcomes": summaries,
        "failure_vs_success_return_pairs": {
            "strict_failure_greater_than_success": greater_pairs,
            "all_pairs": result_pairs,
        },
        "limitation": "not SAC's n-step bootstrapped soft target: it excludes entropy and endpoint bootstrap and discounts each recorded decision directly",
    }


def alpha_samples(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "diagnostics/train/optimization.jsonl"
    values: list[tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            alpha = row.get("metrics", {}).get("train/ent_coef")
            if finite_number(alpha):
                values.append((int(row.get("raw_steps", -1)), int(row.get("updates", -1)), float(alpha)))
    values.sort(key=lambda item: (item[0], item[1]))
    if not values:
        return {"n_samples": 0, "source": path.relative_to(WORKSPACE).as_posix()}
    return {
        "n_samples": len(values),
        "first_sample_raw_steps_updates_alpha": values[0],
        "last_sample_raw_steps_updates_alpha": values[-1],
        "min": min(item[2] for item in values),
        "max": max(item[2] for item in values),
        "mean_sparse_callback_sample": fmean(item[2] for item in values),
        "source": path.relative_to(WORKSPACE).as_posix(),
        "timing_note": "callback rows label latest available logger metrics; alpha is sparse sampled telemetry, not a per-update trace",
    }


def train_reward_branches(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "reward_branches.json"
    if not path.exists():
        return None
    data = read_json(path)
    return {
        "source": path.relative_to(WORKSPACE).as_posix(),
        "completed_training_episodes": data.get("episodes"),
        "mean_branches_per_completed_training_episode": data.get("mean_branches"),
    }


def method_eval(name: str, run_dir: Path) -> dict[str, Any]:
    path = run_dir / "evaluation_results.json"
    data = read_json(path)
    summary = data.get("summary", {})
    records = data.get("episode_records", [])
    protocol_versions = sorted(
        {str(row.get("evaluation_return_protocol_version")) for row in records if row.get("evaluation_return_protocol_version") is not None}
    )
    result: dict[str, Any] = {
        "run": run_dir.relative_to(WORKSPACE).as_posix(),
        "evaluation_results": path.relative_to(WORKSPACE).as_posix(),
        "summary": summary,
        "episode_records": len(records),
        "evaluation_return_protocol_versions": protocol_versions,
        "train_reward_branches": train_reward_branches(run_dir),
    }
    has_v2_episode_components = bool(records) and all(
        row.get("evaluation_return_protocol_version") == "environment_step_reward_v2"
        and all(finite_number(row.get(component)) for component in COMPONENTS)
        for row in records
    )
    if has_v2_episode_components:
        result["outcome_stratification"] = summarize_outcomes(records)
        decision_path = run_dir / "diagnostics/eval/decisions.jsonl.gz"
        if decision_path.exists():
            result["discounted_decision_log_audit"] = discounted_decision_returns(run_dir, records)
    else:
        result["shaped_outcome_stratification"] = {
            "available": False,
            "reason": "saved evaluation rows do not contain environment_step_reward_v2 components",
        }
    if (run_dir / "diagnostics/train/optimization.jsonl").exists():
        result["sampled_alpha"] = alpha_samples(run_dir)
    return result


def main() -> None:
    output = {
        "scope": "read_only_saved_log_audit_no_simulation_no_checkpoint_load",
        "workspace": str(WORKSPACE),
        "gamma_for_offline_discounted_metric": GAMMA,
        "methods": {},
        "old_st_comparability_note": "legacy old ST evaluation is reported as saved; it has no v2 per-episode reward components or decision-level shaped reward log, so no shaped-return stratification is computed",
    }
    for name, relative_path in RUNS.items():
        output["methods"][name] = method_eval(name, WORKSPACE / relative_path)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
