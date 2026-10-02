"""Simulator-agnostic evaluation helpers for the new SB3 entry points."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .features import HierarchicalSceneExtractor
from .topo_temporal_features import TopoTemporalGraphExtractor


EVALUATION_RETURN_PROTOCOL_VERSION = "environment_step_reward_v2"
RAW_RETURN_PROTOCOL_VERSION = "info_undiscounted_reward_or_step_reward_fallback_v1"
REWARD_COMPONENT_PROTOCOL_VERSION = "yield_v2_cumulative_reward_branches_v1"
REWARD_COMPONENT_KEYS = (
    "reward_success",
    "reward_collision",
    "reward_off_route",
    "reward_timeout",
    "reward_step_cost",
    "reward_progress",
)


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics; return fields follow the versioned protocols below.

    ``mean_return`` and ``std_return`` summarize rewards returned by ``env.step``.
    ``raw_mean_return`` and ``raw_std_return`` summarize
    ``info['undiscounted_reward']`` when present; fallback sources are labeled.
    """

    episodes: int
    mean_return: float
    std_return: float
    mean_decision_steps: float
    mean_raw_steps: float
    success_rate: float
    collision_rate: float
    off_route_rate: float
    timeout_rate: float
    raw_mean_return: float | None = None
    raw_std_return: float | None = None
    reward_component_means: dict[str, float | None] | None = None
    reward_component_coverage: int = 0
    reward_component_reconciliation_max_abs_error: float | None = None
    evaluation_return_protocol_version: str = EVALUATION_RETURN_PROTOCOL_VERSION
    raw_return_protocol_version: str = RAW_RETURN_PROTOCOL_VERSION
    reward_component_protocol_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """One deterministic test episode on the source raw-step clock."""

    episode: int
    seed: int
    episode_return: float
    decision_steps: int
    environment_steps: int
    raw_steps: int
    completion_time_seconds: float | None
    success: bool
    collision: bool
    off_route: bool
    timeout: bool
    traffic_variant: str | None
    raw_episode_return: float | None = None
    raw_return_source: str = "not_available"
    reward_success: float | None = None
    reward_collision: float | None = None
    reward_off_route: float | None = None
    reward_timeout: float | None = None
    reward_step_cost: float | None = None
    reward_progress: float | None = None
    evaluation_return_protocol_version: str = EVALUATION_RETURN_PROTOCOL_VERSION
    raw_return_protocol_version: str = RAW_RETURN_PROTOCOL_VERSION
    reward_component_protocol_version: str | None = None
    reward_component_reconciliation_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetailedEvaluationReport:
    """Paper-facing statistics plus the evidence needed to recompute them."""

    summary: EvaluationSummary
    successful_episodes: int
    mean_success_completion_time_seconds: float | None
    std_success_completion_time_seconds: float | None
    sumo_step_seconds: float
    episode_records: tuple[EpisodeEvaluationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_return_protocol_version": (
                self.summary.evaluation_return_protocol_version
            ),
            "raw_return_protocol_version": (
                self.summary.raw_return_protocol_version
            ),
            "reward_component_protocol_version": (
                self.summary.reward_component_protocol_version
            ),
            "summary": self.summary.to_dict(),
            "successful_episodes": self.successful_episodes,
            "mean_success_completion_time_seconds": (
                self.mean_success_completion_time_seconds
            ),
            "std_success_completion_time_seconds": (
                self.std_success_completion_time_seconds
            ),
            "sumo_step_seconds": self.sumo_step_seconds,
            "episode_records": [record.to_dict() for record in self.episode_records],
        }


def _raw_step_reward(info: dict[str, Any], reward: float) -> tuple[float, str]:
    """Return the underlying raw reward when exposed, otherwise mark fallback."""

    value = info.get("undiscounted_reward")
    if value is None:
        return float(reward), "environment_reward_fallback"
    return float(value), "info.undiscounted_reward"


def _episode_reward_components(info: dict[str, Any]) -> dict[str, float | None]:
    """Read the cumulative shaping branches exported by the reward wrapper."""

    components: dict[str, float | None] = {}
    for key in REWARD_COMPONENT_KEYS:
        value = info.get(key)
        try:
            numeric = float(value) if value is not None else None
        except (TypeError, ValueError):
            numeric = None
        components[key] = numeric if numeric is not None and np.isfinite(numeric) else None
    return components


def _component_protocol(
    components: dict[str, float | None],
) -> str | None:
    return (
        REWARD_COMPONENT_PROTOCOL_VERSION
        if all(components.get(key) is not None for key in REWARD_COMPONENT_KEYS)
        else None
    )


def _return_component_summary(
    returns: list[float],
    component_rows: list[dict[str, float | None]],
) -> tuple[dict[str, float | None], int, float | None, str | None]:
    means: dict[str, float | None] = {}
    for key in REWARD_COMPONENT_KEYS:
        values = [row[key] for row in component_rows if row.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None

    complete = [
        (index, row)
        for index, row in enumerate(component_rows)
        if all(row.get(key) is not None for key in REWARD_COMPONENT_KEYS)
    ]
    errors = [
        abs(
            float(returns[index])
            - sum(float(row[key]) for key in REWARD_COMPONENT_KEYS)
        )
        for index, row in complete
    ]
    protocol = (
        REWARD_COMPONENT_PROTOCOL_VERSION
        if component_rows and len(complete) == len(component_rows)
        else None
    )
    return means, len(complete), max(errors) if errors else None, protocol


@contextmanager
def source_evaluation_augmentation(model: Any):
    """Disable rotation augmentation as ``set_configs(..., test=True)`` does."""

    policy = getattr(model, "policy", None)
    modules = (
        [
            module
            for module in policy.modules()
            if isinstance(
                module, (HierarchicalSceneExtractor, TopoTemporalGraphExtractor)
            )
        ]
        if policy is not None and hasattr(policy, "modules")
        else []
    )
    previous = [module.source_augmentation_enabled for module in modules]
    for module in modules:
        module.set_source_augmentation(False)
    try:
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.set_source_augmentation(enabled)


def evaluate_model(
    model: Any,
    env: Any,
    episodes: int = 20,
    *,
    deterministic: bool = True,
    seed: int = 0,
    policy_action_hold: int = 1,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy_action_hold <= 0:
        raise ValueError("policy_action_hold must be positive")
    returns: list[float] = []
    raw_returns: list[float] = []
    component_rows: list[dict[str, float | None]] = []
    raw_return_sources: list[set[str]] = []
    lengths: list[int] = []
    raw_lengths: list[int] = []
    successes = collisions = off_routes = timeouts = 0
    with source_evaluation_augmentation(model):
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            raw_episode_return = 0.0
            episode_raw_sources: set[str] = set()
            environment_steps = 0
            policy_decisions = 0
            held_action: Any | None = None
            while True:
                if environment_steps % policy_action_hold == 0:
                    held_action, _ = model.predict(
                        observation, deterministic=deterministic
                    )
                    policy_decisions += 1
                if held_action is None:
                    raise RuntimeError("No policy action was selected")
                observation, reward, terminated, truncated, info = env.step(
                    held_action
                )
                episode_return += float(reward)
                raw_step_reward, raw_source = _raw_step_reward(info, reward)
                raw_episode_return += raw_step_reward
                episode_raw_sources.add(raw_source)
                environment_steps += 1
                if terminated or truncated:
                    successes += int(info.get("is_success", False))
                    collisions += int(info.get("collision", False))
                    off_routes += int(info.get("off_route", False))
                    timeouts += int(info.get("max_time", False))
                    raw_lengths.append(
                        int(info.get("raw_simulation_steps", environment_steps))
                    )
                    raw_returns.append(raw_episode_return)
                    raw_return_sources.append(episode_raw_sources)
                    component_rows.append(_episode_reward_components(info))
                    break
            returns.append(episode_return)
            lengths.append(policy_decisions)
    component_means, component_coverage, reconciliation_error, component_protocol = (
        _return_component_summary(returns, component_rows)
    )
    raw_sources = set().union(*raw_return_sources) if raw_return_sources else set()
    raw_source_protocol = (
        "mixed_info_and_environment_reward_fallback"
        if len(raw_sources) > 1
        else next(iter(raw_sources), "not_available")
    )
    return EvaluationSummary(
        episodes=episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_decision_steps=float(np.mean(lengths)),
        mean_raw_steps=float(np.mean(raw_lengths)),
        success_rate=successes / episodes,
        collision_rate=collisions / episodes,
        off_route_rate=off_routes / episodes,
        timeout_rate=timeouts / episodes,
        raw_mean_return=float(np.mean(raw_returns)),
        raw_std_return=float(np.std(raw_returns)),
        reward_component_means=component_means,
        reward_component_coverage=component_coverage,
        reward_component_reconciliation_max_abs_error=reconciliation_error,
        raw_return_protocol_version=(
            f"{RAW_RETURN_PROTOCOL_VERSION};source={raw_source_protocol}"
        ),
        reward_component_protocol_version=component_protocol,
    )


def evaluate_model_detailed(
    model: Any,
    env: Any,
    episodes: int = 50,
    *,
    deterministic: bool = True,
    seed: int = 0,
    sumo_step_seconds: float = 0.1,
    policy_action_hold: int = 1,
) -> DetailedEvaluationReport:
    """Evaluate and retain the per-episode evidence required by Tables II/III.

    The primary episode and summary returns accumulate the reward returned by
    ``env.step``. If an environment exposes ``info['undiscounted_reward']``, its
    sum is preserved separately as the raw/outcome return. The six cumulative
    yield-v2 reward branches are copied from terminal info when available.

    The released test runners count simulator steps, and the paper reports
    completion time only for successful episodes.  SUMO is configured with a
    0.1-second step, so completion time is ``raw_steps * 0.1``.  Population
    standard deviation (NumPy's source-default ``ddof=0``) is reported.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if sumo_step_seconds <= 0.0:
        raise ValueError("sumo_step_seconds must be positive")
    if policy_action_hold <= 0:
        raise ValueError("policy_action_hold must be positive")

    records: list[EpisodeEvaluationRecord] = []
    with source_evaluation_augmentation(model):
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(seed=episode_seed)
            episode_return = 0.0
            raw_episode_return = 0.0
            raw_return_sources: set[str] = set()
            environment_steps = 0
            policy_decisions = 0
            held_action: Any | None = None
            while True:
                if environment_steps % policy_action_hold == 0:
                    held_action, _ = model.predict(
                        observation, deterministic=deterministic
                    )
                    policy_decisions += 1
                if held_action is None:
                    raise RuntimeError("No policy action was selected")
                observation, reward, terminated, truncated, info = env.step(
                    held_action
                )
                episode_return += float(reward)
                raw_step_reward, raw_source = _raw_step_reward(info, reward)
                raw_episode_return += raw_step_reward
                raw_return_sources.add(raw_source)
                environment_steps += 1
                if terminated or truncated:
                    raw_steps = int(
                        info.get("raw_simulation_steps", environment_steps)
                    )
                    success = bool(info.get("is_success", False))
                    reward_components = _episode_reward_components(info)
                    component_protocol = _component_protocol(reward_components)
                    component_reconciliation_error = (
                        abs(
                            episode_return
                            - sum(
                                float(reward_components[key])
                                for key in REWARD_COMPONENT_KEYS
                            )
                        )
                        if component_protocol is not None
                        else None
                    )
                    raw_return_source = (
                        "mixed_info_and_environment_reward_fallback"
                        if len(raw_return_sources) > 1
                        else next(iter(raw_return_sources), "not_available")
                    )
                    records.append(
                        EpisodeEvaluationRecord(
                            episode=episode,
                            seed=episode_seed,
                            episode_return=episode_return,
                            decision_steps=policy_decisions,
                            environment_steps=environment_steps,
                            raw_steps=raw_steps,
                            completion_time_seconds=(
                                raw_steps * sumo_step_seconds if success else None
                            ),
                            success=success,
                            collision=bool(info.get("collision", False)),
                            off_route=bool(info.get("off_route", False)),
                            timeout=bool(info.get("max_time", False)),
                            traffic_variant=(
                                str(info["traffic_variant"])
                                if info.get("traffic_variant") is not None
                                else None
                            ),
                            raw_episode_return=raw_episode_return,
                            raw_return_source=raw_return_source,
                            **reward_components,
                            reward_component_protocol_version=component_protocol,
                            reward_component_reconciliation_error=(
                                component_reconciliation_error
                            ),
                        )
                    )
                    break

    returns = np.asarray([record.episode_return for record in records], dtype=float)
    raw_returns = np.asarray(
        [record.raw_episode_return for record in records], dtype=float
    )
    decision_lengths = np.asarray(
        [record.decision_steps for record in records], dtype=float
    )
    raw_lengths = np.asarray([record.raw_steps for record in records], dtype=float)
    completion_times = np.asarray(
        [
            record.completion_time_seconds
            for record in records
            if record.completion_time_seconds is not None
        ],
        dtype=float,
    )
    successful_episodes = int(sum(record.success for record in records))
    component_rows = [
        {key: getattr(record, key) for key in REWARD_COMPONENT_KEYS}
        for record in records
    ]
    component_means, component_coverage, reconciliation_error, component_protocol = (
        _return_component_summary(returns.tolist(), component_rows)
    )
    raw_sources = {record.raw_return_source for record in records}
    raw_source_protocol = (
        "mixed_info_and_environment_reward_fallback"
        if len(raw_sources) > 1
        else next(iter(raw_sources), "not_available")
    )
    summary = EvaluationSummary(
        episodes=episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_decision_steps=float(np.mean(decision_lengths)),
        mean_raw_steps=float(np.mean(raw_lengths)),
        success_rate=successful_episodes / episodes,
        collision_rate=sum(record.collision for record in records) / episodes,
        off_route_rate=sum(record.off_route for record in records) / episodes,
        timeout_rate=sum(record.timeout for record in records) / episodes,
        raw_mean_return=float(np.mean(raw_returns)),
        raw_std_return=float(np.std(raw_returns)),
        reward_component_means=component_means,
        reward_component_coverage=component_coverage,
        reward_component_reconciliation_max_abs_error=reconciliation_error,
        raw_return_protocol_version=(
            f"{RAW_RETURN_PROTOCOL_VERSION};source={raw_source_protocol}"
        ),
        reward_component_protocol_version=component_protocol,
    )
    return DetailedEvaluationReport(
        summary=summary,
        successful_episodes=successful_episodes,
        mean_success_completion_time_seconds=(
            float(np.mean(completion_times)) if completion_times.size else None
        ),
        std_success_completion_time_seconds=(
            float(np.std(completion_times)) if completion_times.size else None
        ),
        sumo_step_seconds=float(sumo_step_seconds),
        episode_records=tuple(records),
    )
