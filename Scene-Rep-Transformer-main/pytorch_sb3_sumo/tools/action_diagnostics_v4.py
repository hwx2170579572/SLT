"""Decision-window and hybrid-policy diagnostics aligned with one evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algos.sb3_torch.evaluation import DetailedEvaluationReport, evaluate_model_detailed
from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedHybridActor
from envs.sumo.decision_alignment_v4 import (
    decision_alignment_context,
    lane_command_index,
)
from tools.action_diagnostics_v2 import (
    ACTION_THRESHOLD,
    NEAR_THRESHOLD_MARGIN,
    _ActionTracingEnvironment,
    _distribution,
    _episode_action_rows,
)


def _hybrid_snapshot(model: Any, observation: dict[str, np.ndarray]) -> dict[str, Any] | None:
    actor = getattr(model, "actor", None)
    if not isinstance(actor, DecisionAlignedHybridActor):
        return None
    tensor_observation, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        batch = actor.all_action_samples(
            tensor_observation, deterministic_speed=True
        )
    return {
        "lane_probabilities": batch.lane_probabilities[0].cpu().tolist(),
        "lane_log_probabilities": batch.lane_log_probabilities[0].cpu().tolist(),
        "conditional_target_speed_mps": (
            (torch.tanh(batch.speed_means[0]) + 1.0) * 5.0
        ).cpu().tolist(),
        "valid_lane_actions": int(batch.action_mask[0].sum().cpu()),
    }


class _DecisionAlignedTracingEnvironment(_ActionTracingEnvironment):
    def __init__(self, env: Any, model: Any) -> None:
        super().__init__(env, model)
        self._current_observation: dict[str, np.ndarray] | None = None

    def reset(self, *args: Any, **kwargs: Any):
        observation, info = super().reset(*args, **kwargs)
        self._current_observation = observation
        return observation, info

    def step(self, action: Any):
        if self._current_observation is None:
            raise RuntimeError("decision-aligned tracing step before reset")
        raw_env = getattr(self.env, "unwrapped", self.env)
        context = decision_alignment_context(raw_env)
        hybrid = _hybrid_snapshot(self.model, self._current_observation)
        output = super().step(action)
        observation, reward, terminated, truncated, info = output
        record = self.records[-1]
        command = int(record["lane_command"])
        command_index = lane_command_index(command)
        feasible = bool(context.lane_action_mask[command_index] > 0.5)
        intent_match = bool(
            context.route_intent_valid and command == context.route_intent
        )
        correct_intent_probability = (
            float(hybrid["lane_probabilities"][lane_command_index(context.route_intent)])
            if hybrid is not None and context.route_intent_valid
            else None
        )
        record.update(
            {
                "pre_action_lane_action_mask": context.lane_action_mask.tolist(),
                "pre_action_non_keep_feasible": context.non_keep_feasible,
                "pre_action_route_intent": int(context.route_intent),
                "pre_action_route_intent_valid": bool(context.route_intent_valid),
                "pre_action_route_intent_reason": context.reason,
                "pre_action_current_edge": context.current_edge,
                "pre_action_current_lane_index": context.current_lane_index,
                "pre_action_next_route_edge": context.next_route_edge,
                "predicted_lane_action_feasible": feasible,
                "route_intent_match": intent_match,
                "correct_route_intent_probability": correct_intent_probability,
                "hybrid_policy": hybrid,
            }
        )
        self._current_observation = observation
        return observation, reward, terminated, truncated, info


def evaluate_with_action_diagnostics_v4(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
) -> tuple[DetailedEvaluationReport, dict[str, Any]]:
    tracing_env = _DecisionAlignedTracingEnvironment(env, model)
    report = evaluate_model_detailed(
        model,
        tracing_env,
        episodes=episodes,
        seed=seed,
        deterministic=True,
        policy_action_hold=1,
    )
    records = tracing_env.records
    if len(report.episode_records) != episodes or not records:
        raise ValueError("v4 evaluation/action trace is incomplete")

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()

    decision_count = len(records)
    command_counts = {
        name: sum(int(row["lane_command"] == command) for row in records)
        for name, command in (("negative", -1), ("keep", 0), ("positive", 1))
    }
    valid_intent = [row for row in records if row["pre_action_route_intent_valid"]]
    route_window = [
        row
        for row in valid_intent
        if int(row["pre_action_route_intent"]) != 0
    ]
    hybrid_rows = [row for row in records if row["hybrid_policy"] is not None]
    correct_probabilities = [
        float(row["correct_route_intent_probability"])
        for row in valid_intent
        if row["correct_route_intent_probability"] is not None
    ]
    diagnostics = {
        "schema_version": "topo-scene-v4.action-diagnostics/v1",
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
            "hybrid_lateral_codes": [-1.0, 0.0, 1.0],
        },
        "lane_command_counts": command_counts,
        "lane_command_rates": {
            key: value / decision_count for key, value in command_counts.items()
        },
        "lane_change_applied_rate": sum(
            int(row["lane_change_applied"]) for row in records
        )
        / decision_count,
        "predicted_lane_action_feasible_rate": sum(
            int(row["predicted_lane_action_feasible"]) for row in records
        )
        / decision_count,
        "valid_route_intent_samples": len(valid_intent),
        "valid_route_intent_match_rate": (
            sum(int(row["route_intent_match"]) for row in valid_intent)
            / len(valid_intent)
            if valid_intent
            else None
        ),
        "route_action_window_samples": len(route_window),
        "route_action_window_match_rate": (
            sum(int(row["route_intent_match"]) for row in route_window)
            / len(route_window)
            if route_window
            else None
        ),
        "route_action_window_applied_match_rate": (
            sum(
                int(row["route_intent_match"] and row["lane_change_applied"])
                for row in route_window
            )
            / len(route_window)
            if route_window
            else None
        ),
        "correct_route_intent_probability": _distribution(correct_probabilities),
        "hybrid_policy_records": len(hybrid_rows),
        "hybrid_exact_lateral_code_rate": (
            sum(float(row["action_lateral"]) in (-1.0, 0.0, 1.0) for row in hybrid_rows)
            / len(hybrid_rows)
            if hybrid_rows
            else None
        ),
        "action_longitudinal": _distribution(
            [float(row["action_longitudinal"]) for row in records]
        ),
        "action_lateral": _distribution(
            [float(row["action_lateral"]) for row in records]
        ),
        "actual_speed_mps": _distribution(
            [float(row["actual_speed_mps"]) for row in records]
        ),
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


__all__ = ["evaluate_with_action_diagnostics_v4"]

