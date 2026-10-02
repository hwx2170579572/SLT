"""Action-level diagnostics aligned with the v2 paper evaluation rollout.

The collector deliberately wraps the environment passed to the existing
``evaluate_model_detailed`` helper.  Consequently outcome metrics and action
diagnostics come from the same deterministic episodes rather than from a
second, merely similar evaluation run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algos.sb3_torch.evaluation import DetailedEvaluationReport, evaluate_model_detailed


ACTION_THRESHOLD = 1.0 / 3.0
NEAR_THRESHOLD_MARGIN = 0.05


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ego_speed_mps(env: Any, observation: dict[str, np.ndarray]) -> tuple[float, str]:
    """Read actual ego speed, preferring the simulator state over observation."""

    raw_env = getattr(env, "unwrapped", env)
    specification = getattr(raw_env, "specification", None)
    ego_id = getattr(specification, "ego_id", None)
    state_fn = getattr(raw_env, "_state", None)
    connection = getattr(raw_env, "_connection", None)
    ego_is_active = False
    if ego_id is not None and connection is not None:
        try:
            ego_is_active = ego_id in connection.vehicle.getIDList()
        except Exception:
            ego_is_active = False
    if ego_is_active and callable(state_fn):
        state = state_fn(f"vehicle:{ego_id}")
        if state is not None:
            vector = np.asarray(state, dtype=np.float64).reshape(-1)
            if vector.size >= 5 and np.all(np.isfinite(vector[3:5])):
                return float(np.linalg.norm(vector[3:5])), "simulator_ego_state"

    trajectory = np.asarray(observation.get("trajectory", ()), dtype=np.float64)
    if trajectory.ndim == 3 and trajectory.shape[0] and trajectory.shape[-1] >= 5:
        history = trajectory[0]
        contract = getattr(specification, "source_observation_contract", "")
        if contract == "carla":
            current = history[0]
        else:
            populated = np.flatnonzero(np.any(history != 0.0, axis=1))
            current = history[populated[-1]] if populated.size else history[0]
        if np.all(np.isfinite(current[3:5])):
            return float(np.linalg.norm(current[3:5])), "terminal_observation_fallback"
    return 0.0, "unavailable_zero_fallback"


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "population_std": None,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("Action diagnostic distribution contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array)),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


EVALUATION_MECHANISM_NAMES = (
    "topology_attention_entropy",
    "topology_effective_lanes",
    "route_compatible_attention_mass",
    "merge_attention_mass",
    "merge_pair_score",
    "reverse_lane_mask_rate",
    "topology_fallback_rate",
    "topology_residual_scale",
    "goal_residual_scale",
    "route_bias_weight",
    "heading_bias_weight",
    "soft_slot_balance_loss",
    "ego_slot_rms",
    "social_slot_rms",
    "route_slot_rms",
    "slot_rms_ratio",
)


def _model_mechanism_diagnostics(model: Any) -> dict[str, float]:
    actor = getattr(model, "actor", None)
    extractor = getattr(actor, "features_extractor", None)
    diagnostic_fn = getattr(extractor, "diagnostic_values", None)
    if not callable(diagnostic_fn):
        return {}
    raw = diagnostic_fn()
    output: dict[str, float] = {}
    for name in EVALUATION_MECHANISM_NAMES:
        value = raw.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            number = float(value.detach().mean().cpu())
        else:
            number = float(value)
        if math.isfinite(number):
            output[name] = number
    return output


class _ActionTracingEnvironment:
    """Minimal transparent wrapper used by ``evaluate_model_detailed``."""

    def __init__(self, env: Any, model: Any) -> None:
        self.env = env
        self.model = model
        self.records: list[dict[str, Any]] = []
        self._episode = -1
        self._decision = 0
        self._seed: int | None = None

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._episode += 1
        self._decision = 0
        seed = kwargs.get("seed")
        self._seed = int(seed) if seed is not None else None
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.shape != (2,) or not np.all(np.isfinite(action_array)):
            raise ValueError(f"Expected a finite two-dimensional action, got {action!r}")
        mechanism = _model_mechanism_diagnostics(self.model)
        observation, reward, terminated, truncated, info = self.env.step(action)
        lateral = float(action_array[1])
        expected_lane_command = (
            -1 if lateral < -ACTION_THRESHOLD else 1 if lateral > ACTION_THRESHOLD else 0
        )
        lane_command = int(info.get("lane_command", expected_lane_command))
        if lane_command != expected_lane_command:
            raise ValueError(
                "Environment lane discretisation disagrees with the frozen +/-1/3 contract"
            )
        actual_speed, speed_source = _ego_speed_mps(self.env, observation)
        threshold_margin = min(
            abs(lateral - ACTION_THRESHOLD), abs(lateral + ACTION_THRESHOLD)
        )
        self.records.append(
            {
                "episode": self._episode,
                "seed": self._seed,
                "decision": self._decision,
                "raw_simulation_steps": int(info.get("raw_simulation_steps", 0)),
                "action_longitudinal": float(action_array[0]),
                "action_lateral": lateral,
                "lane_command": lane_command,
                "lane_change_applied": bool(info.get("lane_change_applied", False)),
                "lane_command_threshold_margin": float(threshold_margin),
                "near_lane_command_threshold": bool(
                    threshold_margin <= NEAR_THRESHOLD_MARGIN
                ),
                "target_speed_mps": _finite_float(info.get("target_speed")),
                "effective_target_speed_mps": _finite_float(
                    info.get("effective_target_speed")
                ),
                "actual_speed_mps": actual_speed,
                "actual_speed_source": speed_source,
                "longitudinal_saturated": bool(abs(float(action_array[0])) >= 0.95),
                "traffic_variant": (
                    str(info["traffic_variant"])
                    if info.get("traffic_variant") is not None
                    else None
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "is_success": bool(info.get("is_success", False)),
                "collision": bool(info.get("collision", False)),
                "off_route": bool(info.get("off_route", False)),
                "timeout": bool(info.get("max_time", False)),
                "mechanism": mechanism,
            }
        )
        self._decision += 1
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def _episode_action_rows(
    records: list[dict[str, Any]], report: DetailedEvaluationReport
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["episode"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for outcome in report.episode_records:
        decisions = grouped.get(outcome.episode, [])
        if not decisions:
            raise ValueError(f"Evaluation episode {outcome.episode} has no action records")
        terminal = decisions[-1]
        if (
            terminal["seed"] != outcome.seed
            or terminal["is_success"] != outcome.success
            or terminal["collision"] != outcome.collision
            or terminal["off_route"] != outcome.off_route
            or terminal["timeout"] != outcome.timeout
            or terminal["traffic_variant"] != outcome.traffic_variant
        ):
            raise ValueError("Action trace is not aligned with detailed evaluation outcomes")
        counts = {
            command: sum(int(row["lane_command"] == value) for row in decisions)
            for command, value in (("negative", -1), ("keep", 0), ("positive", 1))
        }
        rows.append(
            {
                "episode": outcome.episode,
                "seed": outcome.seed,
                "traffic_variant": outcome.traffic_variant,
                "decisions": len(decisions),
                "lane_command_counts": counts,
                "lane_change_applied_count": sum(
                    int(row["lane_change_applied"]) for row in decisions
                ),
                "mean_speed_mps": float(
                    np.mean([row["actual_speed_mps"] for row in decisions])
                ),
                "success": outcome.success,
                "collision": outcome.collision,
                "off_route": outcome.off_route,
                "timeout": outcome.timeout,
            }
        )
    return rows


def evaluate_with_action_diagnostics_v2(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
) -> tuple[DetailedEvaluationReport, dict[str, Any]]:
    """Run one exact detailed evaluation and derive auditable action statistics."""

    tracing_env = _ActionTracingEnvironment(env, model)
    report = evaluate_model_detailed(
        model,
        tracing_env,
        episodes=episodes,
        seed=seed,
        deterministic=True,
        policy_action_hold=1,
    )
    records = tracing_env.records
    if len(report.episode_records) != episodes:
        raise ValueError("Detailed evaluation returned an unexpected episode count")
    if not records:
        raise ValueError("No action diagnostic records were collected")

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()

    lateral = [float(row["action_lateral"]) for row in records]
    longitudinal = [float(row["action_longitudinal"]) for row in records]
    margins = [float(row["lane_command_threshold_margin"]) for row in records]
    speeds = [float(row["actual_speed_mps"]) for row in records]
    target_speeds = [
        float(row["target_speed_mps"])
        for row in records
        if row["target_speed_mps"] is not None
    ]
    command_counts = {
        command: sum(int(row["lane_command"] == value) for row in records)
        for command, value in (("negative", -1), ("keep", 0), ("positive", 1))
    }
    decision_count = len(records)
    non_keep_count = command_counts["negative"] + command_counts["positive"]
    applied_count = sum(int(row["lane_change_applied"]) for row in records)
    applied_non_keep = sum(
        int(row["lane_change_applied"] and row["lane_command"] != 0)
        for row in records
    )
    speed_sources = sorted({str(row["actual_speed_source"]) for row in records})
    mechanism_names = sorted(
        {name for row in records for name in row.get("mechanism", {})}
    )
    diagnostics = {
        "schema_version": "topology-temporal-v2-action-diagnostics/v1",
        "computed_from_real_rollout": True,
        "aligned_with_paper_evaluation": True,
        "deterministic": True,
        "evaluation_seed_start": seed,
        "episodes": episodes,
        "decision_records": decision_count,
        "threshold_contract": {
            "negative_if_lateral_below": -ACTION_THRESHOLD,
            "positive_if_lateral_above": ACTION_THRESHOLD,
            "near_threshold_margin": NEAR_THRESHOLD_MARGIN,
        },
        "lane_command_counts": command_counts,
        "lane_command_rates": {
            key: value / decision_count for key, value in command_counts.items()
        },
        "lane_change_applied_rate": applied_count / decision_count,
        "lane_change_applied_given_non_keep_rate": (
            applied_non_keep / non_keep_count if non_keep_count else None
        ),
        "near_lane_command_threshold_rate": sum(
            int(row["near_lane_command_threshold"]) for row in records
        )
        / decision_count,
        "longitudinal_saturation_rate": sum(
            int(row["longitudinal_saturated"]) for row in records
        )
        / decision_count,
        "action_longitudinal": _distribution(longitudinal),
        "action_lateral": _distribution(lateral),
        "lane_command_threshold_margin": _distribution(margins),
        "actual_speed_mps": _distribution(speeds),
        "target_speed_mps": _distribution(target_speeds),
        "actual_speed_sources": speed_sources,
        "evaluation_mechanism": {
            name: _distribution(
                [
                    float(row["mechanism"][name])
                    for row in records
                    if name in row.get("mechanism", {})
                ]
            )
            for name in mechanism_names
        },
        "outcomes": report.summary.to_dict(),
        "per_episode": _episode_action_rows(records, report),
        "trace": {
            "path": str(trace_path.resolve()),
            "format": "json-lines",
            "records": decision_count,
            "sha256": trace_sha256,
        },
    }
    return report, diagnostics


__all__ = [
    "ACTION_THRESHOLD",
    "NEAR_THRESHOLD_MARGIN",
    "evaluate_with_action_diagnostics_v2",
]
