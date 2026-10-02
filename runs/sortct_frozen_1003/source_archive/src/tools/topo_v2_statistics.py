"""Predeclared paired hierarchical bootstrap for topology-temporal v2."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


PRIMARY_RECORD_FIELDS = {
    "success_rate": "success",
    "collision_rate": "collision",
}


def _aligned_episode_delta(
    candidate: dict[str, Any], baseline: dict[str, Any], field: str
) -> np.ndarray:
    candidate_records = candidate["episode_records"]
    baseline_records = baseline["episode_records"]
    if len(candidate_records) != len(baseline_records) or not candidate_records:
        raise ValueError("Paired runs must contain the same non-zero episode count")
    deltas: list[float] = []
    for candidate_record, baseline_record in zip(candidate_records, baseline_records):
        for key in ("episode", "seed", "traffic_variant"):
            if candidate_record.get(key) != baseline_record.get(key):
                raise ValueError(f"Paired episode alignment failed for {key!r}")
        deltas.append(
            float(bool(candidate_record[field])) - float(bool(baseline_record[field]))
        )
    return np.asarray(deltas, dtype=np.float64)


def paired_hierarchical_bootstrap(
    rows: list[dict[str, Any]],
    *,
    candidate_method: str,
    baseline_method: str,
    scenarios: Iterable[str],
    metrics: Iterable[str] = ("success_rate", "collision_rate"),
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Resample paired training seeds, then paired evaluation episodes."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    by_key = {
        (str(row["method"]), str(row["scenario"]), int(row["seed"])): row
        for row in rows
    }
    tests: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        candidate_seeds = {
            int(row["seed"])
            for row in rows
            if row["method"] == candidate_method and row["scenario"] == scenario
        }
        baseline_seeds = {
            int(row["seed"])
            for row in rows
            if row["method"] == baseline_method and row["scenario"] == scenario
        }
        paired_seeds = sorted(candidate_seeds & baseline_seeds)
        if not paired_seeds:
            raise ValueError(f"No paired seeds for scenario {scenario!r}")
        for metric_index, metric in enumerate(metrics):
            if metric not in PRIMARY_RECORD_FIELDS:
                raise ValueError(f"Unsupported episode metric {metric!r}")
            field = PRIMARY_RECORD_FIELDS[metric]
            seed_deltas = [
                _aligned_episode_delta(
                    by_key[(candidate_method, scenario, training_seed)],
                    by_key[(baseline_method, scenario, training_seed)],
                    field,
                )
                for training_seed in paired_seeds
            ]
            episode_counts = {len(values) for values in seed_deltas}
            if len(episode_counts) != 1:
                raise ValueError("Every paired training seed must use the same episode count")
            delta_matrix = np.stack(seed_deltas, axis=0)
            training_seed_count, episode_count = delta_matrix.shape
            rng = np.random.default_rng(
                seed + scenario_index * 1009 + metric_index * 97
            )
            sampled_seed_indices = rng.integers(
                0,
                training_seed_count,
                size=(resamples, training_seed_count),
            )
            bootstrap_seed_means = np.empty(
                (resamples, training_seed_count), dtype=np.float64
            )
            for position in range(training_seed_count):
                chosen = delta_matrix[sampled_seed_indices[:, position], :]
                sampled_episodes = rng.integers(
                    0, episode_count, size=(resamples, episode_count)
                )
                bootstrap_seed_means[:, position] = np.take_along_axis(
                    chosen, sampled_episodes, axis=1
                ).mean(axis=1)
            bootstrap_deltas = bootstrap_seed_means.mean(axis=1)
            observed = float(delta_matrix.mean(axis=1).mean())
            lower, upper = np.quantile(bootstrap_deltas, (0.025, 0.975))
            lower_tail = (np.count_nonzero(bootstrap_deltas <= 0.0) + 1.0) / (
                resamples + 1.0
            )
            upper_tail = (np.count_nonzero(bootstrap_deltas >= 0.0) + 1.0) / (
                resamples + 1.0
            )
            p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
            tests.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "candidate_minus_baseline": observed,
                    "ci_95": {"lower": float(lower), "upper": float(upper)},
                    "two_sided_bootstrap_p_value": float(p_value),
                    "paired_training_seeds": paired_seeds,
                    "training_seed_count": training_seed_count,
                    "paired_episodes_per_seed": episode_count,
                    "resamples": resamples,
                }
            )
    adjusted = holm_adjust(
        [float(test["two_sided_bootstrap_p_value"]) for test in tests]
    )
    for test, adjusted_p in zip(tests, adjusted):
        test["holm_adjusted_p_value"] = adjusted_p
    return {
        "schema_version": "topology-temporal-v2-paired-hierarchical-bootstrap/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "candidate_method": candidate_method,
        "baseline_method": baseline_method,
        "seed_pairing": True,
        "episode_pairing": True,
        "bootstrap_seed": seed,
        "multiplicity_family": "all scenario x primary-metric tests in this receipt",
        "tests": tests,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return step-down Holm adjusted p-values in original order."""

    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be finite and within [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


__all__ = ["holm_adjust", "paired_hierarchical_bootstrap"]
