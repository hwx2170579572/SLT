"""Simulator-agnostic evaluation helpers for the new SB3 entry points."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .features import HierarchicalSceneExtractor
from .topo_temporal_features import TopoTemporalGraphExtractor


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    mean_return: float
    std_return: float
    mean_decision_steps: float
    mean_raw_steps: float
    success_rate: float
    collision_rate: float
    off_route_rate: float
    timeout_rate: float

    def to_dict(self) -> dict[str, int | float]:
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
    lengths: list[int] = []
    raw_lengths: list[int] = []
    successes = collisions = off_routes = timeouts = 0
    with source_evaluation_augmentation(model):
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
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
                episode_return += float(info.get("undiscounted_reward", reward))
                environment_steps += 1
                if terminated or truncated:
                    successes += int(info.get("is_success", False))
                    collisions += int(info.get("collision", False))
                    off_routes += int(info.get("off_route", False))
                    timeouts += int(info.get("max_time", False))
                    raw_lengths.append(
                        int(info.get("raw_simulation_steps", environment_steps))
                    )
                    break
            returns.append(episode_return)
            lengths.append(policy_decisions)
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
                episode_return += float(info.get("undiscounted_reward", reward))
                environment_steps += 1
                if terminated or truncated:
                    raw_steps = int(
                        info.get("raw_simulation_steps", environment_steps)
                    )
                    success = bool(info.get("is_success", False))
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
                        )
                    )
                    break

    returns = np.asarray([record.episode_return for record in records], dtype=float)
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
