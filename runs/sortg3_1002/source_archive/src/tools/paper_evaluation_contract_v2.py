"""Evaluation provenance for the v2 validation/formal-test split."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from algos.sb3_torch.evaluation import DetailedEvaluationReport
from envs.sumo.paper_env import PaperSumoSceneEnv
from tools.paper_evaluation_contract import (
    callable_name,
    validate_environment_asset_provenance,
    validate_model_environment_spaces,
)


CONTRACT_VERSION = "topology-temporal-v2-evaluation/v1"


def expected_v2_partition(traffic_protocol: str, evaluation_split: str) -> str:
    if traffic_protocol == "source_all":
        return "all"
    if traffic_protocol != "frozen_60_20_20":
        raise ValueError(f"Unsupported v2 traffic protocol {traffic_protocol!r}")
    if evaluation_split not in ("validation", "test"):
        raise ValueError("evaluation_split must be validation or test")
    return evaluation_split


def _assert_rate(label: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}={actual} cannot be recomputed as {expected}")


def build_evaluation_provenance_v2(
    *,
    model: Any,
    env: gym.Env,
    report: DetailedEvaluationReport,
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    evaluation_split: str,
    episode_limit_profile: str,
    evaluation_seed_start: int,
    environment_factory: str,
    space_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(env, PaperSumoSceneEnv):
        raise TypeError(
            "v2 paper evaluation requires a PaperSumoSceneEnv-compatible instance"
        )
    partition = expected_v2_partition(traffic_protocol, evaluation_split)
    actual_partition = getattr(env, "_traffic_partition", None)
    if actual_partition != partition:
        raise ValueError(
            f"Traffic partition is {actual_partition!r}, expected {partition!r}"
        )
    actual_limit = getattr(env, "_episode_limit_profile", None)
    if actual_limit != episode_limit_profile:
        raise ValueError(
            f"Episode-limit profile is {actual_limit!r}, expected "
            f"{episode_limit_profile!r}"
        )
    asset_provenance = validate_environment_asset_provenance(env, scenario)
    spaces = space_contract or validate_model_environment_spaces(model, env)
    records = list(report.episode_records)
    episodes = int(report.summary.episodes)
    if len(records) != episodes:
        raise ValueError(f"Expected {episodes} records, found {len(records)}")
    if [record.episode for record in records] != list(range(episodes)):
        raise ValueError("Episode indices are not contiguous from zero")
    expected_seeds = list(range(evaluation_seed_start, evaluation_seed_start + episodes))
    if [record.seed for record in records] != expected_seeds:
        raise ValueError("Evaluation seeds do not match the deterministic sequence")
    variants = [record.traffic_variant for record in records]
    if any(not isinstance(variant, str) or not variant for variant in variants):
        raise ValueError("Every episode must record traffic_variant")
    for rate_name, record_name in (
        ("success_rate", "success"),
        ("collision_rate", "collision"),
        ("off_route_rate", "off_route"),
        ("timeout_rate", "timeout"),
    ):
        actual = float(getattr(report.summary, rate_name))
        expected = sum(bool(getattr(record, record_name)) for record in records) / episodes
        _assert_rate(rate_name, actual, expected)
    return {
        "contract_version": CONTRACT_VERSION,
        "validated": True,
        "environment_factory": environment_factory,
        "environment_class": f"{type(env).__module__}.{type(env).__qualname__}",
        "algorithm": algorithm,
        "scenario": scenario,
        "traffic_protocol": traffic_protocol,
        "traffic_partition": partition,
        "evaluation_split": evaluation_split,
        "episode_limit_profile": episode_limit_profile,
        **asset_provenance,
        "evaluation_seed_start": evaluation_seed_start,
        "episodes": episodes,
        "traffic_variant_episode_count": len(variants),
        "traffic_variants_observed": sorted(set(variants)),
        **spaces,
    }


__all__ = [
    "CONTRACT_VERSION",
    "build_evaluation_provenance_v2",
    "callable_name",
    "expected_v2_partition",
]
