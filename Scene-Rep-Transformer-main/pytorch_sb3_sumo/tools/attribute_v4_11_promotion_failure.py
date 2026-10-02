"""Attribute the complete v4.11 promotion failure from immutable rollouts.

This analysis is observational and model-diagnostic only.  It reads the twelve
accepted promotion jobs through the v4.11.2 source map, verifies that formal
testing and forbidden inference mechanisms were absent, and measures paired
outcomes, learned-risk discrimination, proposal regret, lane-command dynamics,
and an offline continuous learned lane-log-prior sensitivity.

The offline sensitivity never rewrites a stored action and cannot claim a
closed-loop outcome.  It only asks how the already-recorded learned candidate
scores would rank if the actor's continuous lane probability were included as
a model-support term, analogous to the existing component log prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import attribute_v4_10_development_failure as shared


PROMOTION_ROOT = ROOT / "results_topo_v4_11_promotion"
PROMOTION_SUMMARY = PROMOTION_ROOT / "summary.json"
PROMOTION_GATE = PROMOTION_ROOT / "promotion_gate.json"
RECOVERY_SUMMARY = (
    PROMOTION_ROOT / "engineering_recovery_v4_11_2" / "summary.json"
)
DEFAULT_OUTPUT_DIR = PROMOTION_ROOT / "attribution" / "complete_failure_v4_11"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "attribution.json"
LANE_CODES = (-1, 0, 1)
LANE_PRIOR_COEFFICIENTS = (0.01, 0.025, 0.05, 0.10)
COLLISION_RISK_COEFFICIENTS = (1.25, 1.50, 2.00)
EXPECTED_PAIRS = 6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return shared._load_jsonl(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _mean(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return statistics.fmean(items) if items else None


def _source_map(recovery: Mapping[str, Any]) -> dict[str, Path]:
    _require(recovery.get("complete") is True, "recovery summary is incomplete")
    _require(int(recovery.get("accepted_runs", 0)) == 12, "not all jobs accepted")
    selected: dict[str, Path] = {}
    for row in recovery["source_selection"]:
        job = str(row["job"])
        root = row.get("selected_source_directory")
        _require(isinstance(root, str) and bool(root), f"unresolved source: {job}")
        selected[job] = Path(root)
    _require(len(selected) == 12, "source map does not contain twelve jobs")
    return selected


def _run_paths(root: Path, *, candidate: bool) -> dict[str, Path]:
    values = {
        "paper": root / "paper_evaluation_detailed.json",
        "actions": root / "action_diagnostics.json",
        "trace": root / "action_diagnostics_decisions.jsonl",
    }
    if candidate:
        values.update(
            {
                "training": root / "training_diagnostics.json",
                "selector": root / "selector" / "receipt.json",
                "labels": root / "collision_label_diagnostics.json",
            }
        )
    for name, path in values.items():
        _require(path.is_file(), f"missing {name}: {path}")
    return values


def _outcome(record: Mapping[str, Any]) -> str:
    if bool(record.get("collision")):
        return "collision"
    if bool(record.get("off_route")):
        return "off_route"
    if bool(record.get("success")):
        return "success"
    if bool(record.get("timeout")):
        return "timeout"
    raise ValueError(f"episode has no terminal outcome: {record}")


def _paper_episode_index(paper: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    for record in paper["episode_records"]:
        seed = int(record["seed"])
        _require(seed not in values, f"duplicate evaluation seed: {seed}")
        values[seed] = {
            "episode": int(record["episode"]),
            "seed": seed,
            "traffic_variant": str(record["traffic_variant"]),
            "outcome": _outcome(record),
            "decision_steps": int(record["decision_steps"]),
        }
    return values


def _behavior(actions: Mapping[str, Any]) -> dict[str, Any]:
    def value(path: Sequence[str]) -> Any:
        current: Any = actions
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    return {
        "decision_records": int(actions["decision_records"]),
        "lane_change_applied_rate": actions.get("lane_change_applied_rate"),
        "valid_route_intent_match_rate": actions.get(
            "valid_route_intent_match_rate"
        ),
        "route_action_window_match_rate": actions.get(
            "route_action_window_match_rate"
        ),
        "non_keep_command_rate": (
            float(actions["lane_command_rates"]["negative"])
            + float(actions["lane_command_rates"]["positive"])
        ),
        "mean_action_longitudinal": value(("action_longitudinal", "mean")),
        "mean_actual_speed_mps": value(("actual_speed_mps", "mean")),
        "p90_actual_speed_mps": value(("actual_speed_mps", "p90")),
    }


def _behavior_delta(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "lane_change_applied_rate",
        "valid_route_intent_match_rate",
        "route_action_window_match_rate",
        "non_keep_command_rate",
        "mean_action_longitudinal",
        "mean_actual_speed_mps",
        "p90_actual_speed_mps",
    )
    output: dict[str, Any] = {}
    for key in keys:
        left = candidate.get(key)
        right = control.get(key)
        output[key] = (
            float(left) - float(right)
            if left is not None and right is not None
            else None
        )
    return output


def _training_evidence(training: Mapping[str, Any]) -> dict[str, Any]:
    wanted = (
        "train/collision_soft_bce_loss",
        "train/collision_pairwise_rank_loss",
        "train/collision_temporal_consistency_loss",
        "risk/collision_probability_absolute_error",
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/collision_cost_replay_prediction",
        "risk/policy_expected_collision_cost",
        "risk/collision_twin_disagreement",
        "risk/reward_twin_disagreement",
        "risk/learned_uncertainty",
        "support/component_entropy",
        "support/component_speed_spread",
        "hybrid/lane_entropy",
    )
    statistics_payload = training["statistics"]
    return {
        "statistics": {key: statistics_payload[key] for key in wanted},
        "collision_labels": training["collision_labels"],
        "optimizer_ownership": training["optimizer_ownership"],
        "scientific_version": training["scientific_version"],
        "inference_safety_rule_added": training[
            "inference_safety_rule_added"
        ],
        "external_kinematic_projection": training[
            "external_kinematic_projection"
        ],
        "kinematic_safety_projection": training[
            "kinematic_safety_projection"
        ],
        "traffic_risk_in_lane_mask": training["traffic_risk_in_lane_mask"],
        "action_postprocessing_override": training[
            "action_postprocessing_override"
        ],
    }


def _selector_evidence(
    receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    train_calibration_dynamics: Mapping[str, Any],
    validation_dynamics: Mapping[str, Any],
) -> dict[str, Any]:
    _require(receipt["selection_partition"] == "train", "selector partition drift")
    _require(receipt["validation_used_for_selection"] is False, "validation leakage")
    _require(receipt["formal_test_used_for_selection"] is False, "formal leakage")
    candidates = [
        {
            "checkpoint_kind": row["checkpoint_kind"],
            "selected": row["checkpoint_kind"]
            == receipt["selected_checkpoint_kind"],
            "train_calibration_summary": row["summary"],
        }
        for row in receipt["candidates"]
    ]
    selected = next(row for row in candidates if row["selected"])
    return {
        "selected_checkpoint_kind": receipt["selected_checkpoint_kind"],
        "selection_partition": receipt["selection_partition"],
        "validation_used_for_selection": False,
        "formal_test_used_for_selection": False,
        "candidates": candidates,
        "selected_train_minus_validation_success_rate": (
            float(selected["train_calibration_summary"]["success_rate"])
            - float(validation["success_rate"])
        ),
        "selected_train_minus_validation_collision_rate": (
            float(selected["train_calibration_summary"]["collision_rate"])
            - float(validation["collision_rate"])
        ),
        "selected_train_calibration_lane_policy_dynamics": dict(
            train_calibration_dynamics
        ),
        "selected_validation_lane_policy_dynamics": dict(
            validation_dynamics
        ),
        "selected_train_minus_validation_lane_dynamics": {
            key: (
                float(train_calibration_dynamics[key])
                - float(validation_dynamics[key])
            )
            for key in (
                "actor_mode_switch_rate",
                "selected_lane_switch_rate",
                "actor_mode_non_keep_rate",
                "selected_lane_non_keep_rate",
                "selected_actor_mode_match_rate",
            )
            if train_calibration_dynamics.get(key) is not None
            and validation_dynamics.get(key) is not None
        },
    }


def _candidate_index(
    decoder: Mapping[str, Any], lane_index: int, component_index: int
) -> dict[str, float | int]:
    proposals = shared._flatten_decoder(decoder)
    return next(
        row
        for row in proposals
        if int(row["lane_index"]) == lane_index
        and int(row["component_index"]) == component_index
    )


def _lane_prior_choice(row: Mapping[str, Any], coefficient: float) -> dict[str, Any]:
    decoder = row["target_critic_decoder"]
    probabilities = [float(value) for value in decoder["lane_probabilities"]]
    proposals = shared._flatten_decoder(decoder)
    rescored: list[tuple[float, dict[str, float | int]]] = []
    for proposal in proposals:
        lane_index = int(proposal["lane_index"])
        lane_probability = probabilities[lane_index]
        score = float(proposal["score"]) + coefficient * math.log(
            max(lane_probability, 1e-45)
        )
        rescored.append((score, proposal))
    score, selected = max(rescored, key=lambda item: item[0])
    lane_index = int(selected["lane_index"])
    component_index = int(selected["component_index"])
    return {
        "lane_index": lane_index,
        "lane_code": LANE_CODES[lane_index],
        "component_index": component_index,
        "score": score,
        "collision": float(selected["collision"]),
        "q": float(selected["q"]),
        "speed": float(selected["speed"]),
        "lane_probability": probabilities[lane_index],
    }


def _collision_risk_choice(
    row: Mapping[str, Any], coefficient: float
) -> dict[str, Any]:
    decoder = row["target_critic_decoder"]
    original_coefficient = float(decoder["collision_risk_coef"])
    _require(coefficient >= original_coefficient, "risk coefficient decreased")
    probabilities = [float(value) for value in decoder["lane_probabilities"]]
    proposals = shared._flatten_decoder(decoder)
    rescored: list[tuple[float, dict[str, float | int]]] = []
    for proposal in proposals:
        score = float(proposal["score"]) - (
            coefficient - original_coefficient
        ) * float(proposal["collision"])
        rescored.append((score, proposal))
    score, selected = max(rescored, key=lambda item: item[0])
    lane_index = int(selected["lane_index"])
    component_index = int(selected["component_index"])
    return {
        "lane_index": lane_index,
        "lane_code": LANE_CODES[lane_index],
        "component_index": component_index,
        "score": score,
        "collision": float(selected["collision"]),
        "q": float(selected["q"]),
        "speed": float(selected["speed"]),
        "lane_probability": probabilities[lane_index],
    }


def _switch_rate_by_episode(values: Sequence[tuple[int, int, int]]) -> float | None:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for episode, decision, lane_code in values:
        grouped[int(episode)].append((int(decision), int(lane_code)))
    switches = comparisons = 0
    for rows in grouped.values():
        lanes = [lane for _, lane in sorted(rows)]
        for previous, current in zip(lanes, lanes[1:]):
            comparisons += 1
            switches += int(previous != current)
    return switches / comparisons if comparisons else None


def lane_policy_dynamics(
    raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate actor intent dynamics from target-critic lane selection.

    This is a read-only diagnostic over learned outputs.  Actor modes are the
    argmax of the masked lane distribution and selected lanes are the exact
    target-critic argmax already stored in the trace.  No action is changed.
    """

    actor_lanes: list[tuple[int, int, int]] = []
    selected_lanes: list[tuple[int, int, int]] = []
    multi_actor_lanes: list[tuple[int, int, int]] = []
    multi_selected_lanes: list[tuple[int, int, int]] = []
    actor_non_keep = selected_non_keep = actor_selected_matches = 0
    multi_actor_non_keep = multi_selected_non_keep = multi_matches = 0
    multi_count = 0
    selected_lane_probabilities: list[float] = []
    multi_selected_lane_probabilities: list[float] = []
    for row in raw_rows:
        decoder = row["target_critic_decoder"]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
        _require(bool(valid_indices), "trace row has no feasible lane action")
        actor_index = max(valid_indices, key=lambda index: probabilities[index])
        selected_index = int(decoder["selected_lane_index"])
        _require(valid[selected_index], "stored selected lane is infeasible")
        actor_lane = int(LANE_CODES[actor_index])
        selected_lane = int(LANE_CODES[selected_index])
        episode = int(row["episode"])
        decision = int(row["decision"])
        actor_lanes.append((episode, decision, actor_lane))
        selected_lanes.append((episode, decision, selected_lane))
        actor_non_keep += int(actor_lane != 0)
        selected_non_keep += int(selected_lane != 0)
        actor_selected_matches += int(actor_index == selected_index)
        selected_lane_probabilities.append(probabilities[selected_index])
        if len(valid_indices) > 1:
            multi_count += 1
            multi_actor_lanes.append((episode, decision, actor_lane))
            multi_selected_lanes.append((episode, decision, selected_lane))
            multi_actor_non_keep += int(actor_lane != 0)
            multi_selected_non_keep += int(selected_lane != 0)
            multi_matches += int(actor_index == selected_index)
            multi_selected_lane_probabilities.append(probabilities[selected_index])
    count = len(raw_rows)
    _require(count > 0, "lane policy dynamics requires at least one row")
    return {
        "decision_count": count,
        "offline_observation_only": True,
        "action_rewritten": False,
        "actor_mode_switch_rate": _switch_rate_by_episode(actor_lanes),
        "selected_lane_switch_rate": _switch_rate_by_episode(selected_lanes),
        "actor_mode_non_keep_rate": actor_non_keep / count,
        "selected_lane_non_keep_rate": selected_non_keep / count,
        "selected_actor_mode_match_rate": actor_selected_matches / count,
        "selected_lane_probability": shared.numeric_summary(
            selected_lane_probabilities
        ),
        "multi_feasible": {
            "decision_count": multi_count,
            "decision_rate": multi_count / count,
            "actor_mode_switch_rate": _switch_rate_by_episode(
                multi_actor_lanes
            ),
            "selected_lane_switch_rate": _switch_rate_by_episode(
                multi_selected_lanes
            ),
            "actor_mode_non_keep_rate": (
                multi_actor_non_keep / multi_count if multi_count else None
            ),
            "selected_lane_non_keep_rate": (
                multi_selected_non_keep / multi_count if multi_count else None
            ),
            "selected_actor_mode_match_rate": (
                multi_matches / multi_count if multi_count else None
            ),
            "selected_lane_probability": shared.numeric_summary(
                multi_selected_lane_probabilities
            ),
        },
    }


def proposal_value_separation(
    raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure within-state action separation learned by reward/risk critics."""

    collision_spreads: list[float] = []
    reward_spreads: list[float] = []
    score_spreads: list[float] = []
    lane_min_collision_spreads: list[float] = []
    same_lane_collision_spreads: list[float] = []
    for row in raw_rows:
        decoder = row["target_critic_decoder"]
        proposals = shared._flatten_decoder(decoder)
        collision = [float(item["collision"]) for item in proposals]
        reward = [float(item["q"]) for item in proposals]
        score = [float(item["score"]) for item in proposals]
        collision_spreads.append(max(collision) - min(collision))
        reward_spreads.append(max(reward) - min(reward))
        score_spreads.append(max(score) - min(score))
        lane_minima = [
            min(
                float(item["collision"])
                for item in proposals
                if int(item["lane_index"]) == lane_index
            )
            for lane_index, valid in enumerate(decoder["valid_lane_actions"])
            if bool(valid)
        ]
        lane_min_collision_spreads.append(max(lane_minima) - min(lane_minima))
        selected_lane = int(decoder["selected_lane_index"])
        same_lane_collision = [
            float(item["collision"])
            for item in proposals
            if int(item["lane_index"]) == selected_lane
        ]
        same_lane_collision_spreads.append(
            max(same_lane_collision) - min(same_lane_collision)
        )
    _require(bool(collision_spreads), "proposal separation requires trace rows")
    return {
        "decision_count": len(collision_spreads),
        "learned_values_only": True,
        "feasible_collision_value_spread": shared.numeric_summary(
            collision_spreads
        ),
        "feasible_reward_q_spread": shared.numeric_summary(reward_spreads),
        "feasible_model_score_spread": shared.numeric_summary(score_spreads),
        "lane_minimum_collision_value_spread": shared.numeric_summary(
            lane_min_collision_spreads
        ),
        "selected_lane_collision_value_spread": shared.numeric_summary(
            same_lane_collision_spreads
        ),
        "mean_collision_to_reward_spread_ratio": (
            statistics.fmean(collision_spreads)
            / max(statistics.fmean(reward_spreads), 1e-12)
        ),
    }


def collision_risk_sensitivity(
    raw_rows: Sequence[Mapping[str, Any]], coefficient: float
) -> dict[str, Any]:
    """Offline reranking with only the learned collision-value coefficient."""

    choices: list[dict[str, Any]] = []
    original_lanes: list[tuple[int, int, int]] = []
    changed_proposal = changed_lane = route_matches = route_samples = 0
    actor_mode_matches = 0
    for row in raw_rows:
        decoder = row["target_critic_decoder"]
        choice = _collision_risk_choice(row, coefficient)
        original_lane_index = int(decoder["selected_lane_index"])
        original_component = int(decoder["selected_component_index"])
        changed_proposal += int(
            choice["lane_index"] != original_lane_index
            or choice["component_index"] != original_component
        )
        changed_lane += int(choice["lane_index"] != original_lane_index)
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        actor_mode = max(
            (index for index, is_valid in enumerate(valid) if is_valid),
            key=lambda index: probabilities[index],
        )
        actor_mode_matches += int(choice["lane_index"] == actor_mode)
        if bool(row.get("pre_action_route_intent_valid")):
            route_samples += 1
            route_matches += int(
                int(choice["lane_code"]) == int(row["pre_action_route_intent"])
            )
        choices.append(choice)
        original_lanes.append(
            (
                int(row["episode"]),
                int(row["decision"]),
                int(decoder["selected_lane"]),
            )
        )
    count = len(raw_rows)
    _require(count > 0, "collision-risk sensitivity requires trace rows")
    counterfactual_lanes = [
        (
            int(row["episode"]),
            int(row["decision"]),
            int(choice["lane_code"]),
        )
        for row, choice in zip(raw_rows, choices)
    ]
    return {
        "coefficient": coefficient,
        "decision_count": count,
        "offline_only_no_action_rewrite": True,
        "closed_loop_outcome_claim_allowed": False,
        "changed_proposal_rate": changed_proposal / count,
        "changed_lane_rate": changed_lane / count,
        "original_lane_command_switch_rate": _switch_rate_by_episode(
            original_lanes
        ),
        "counterfactual_lane_command_switch_rate": _switch_rate_by_episode(
            counterfactual_lanes
        ),
        "counterfactual_non_keep_rate": sum(
            int(choice["lane_code"] != 0) for choice in choices
        )
        / count,
        "counterfactual_actor_lane_mode_match_rate": actor_mode_matches / count,
        "counterfactual_route_intent_match_rate": (
            route_matches / route_samples if route_samples else None
        ),
        "counterfactual_selected_collision_value": shared.numeric_summary(
            float(choice["collision"]) for choice in choices
        ),
        "counterfactual_selected_min_reward_q": shared.numeric_summary(
            float(choice["q"]) for choice in choices
        ),
        "counterfactual_selected_speed_normalized": shared.numeric_summary(
            float(choice["speed"]) for choice in choices
        ),
    }


def lane_prior_sensitivity(
    raw_rows: Sequence[Mapping[str, Any]], coefficient: float
) -> dict[str, Any]:
    _require(coefficient > 0.0, "lane-prior coefficient must be positive")
    choices: list[dict[str, Any]] = []
    original_lanes: list[tuple[int, int, int]] = []
    changed_lane_by_outcome: Counter[str] = Counter()
    count_by_outcome: Counter[str] = Counter()
    terminal_flags: dict[int, dict[str, bool]] = defaultdict(
        lambda: {
            "collision": False,
            "off_route": False,
            "success": False,
            "timeout": False,
        }
    )
    for row in raw_rows:
        flags = terminal_flags[int(row["episode"])]
        flags["collision"] |= bool(row.get("collision"))
        flags["off_route"] |= bool(row.get("off_route"))
        flags["success"] |= bool(row.get("is_success"))
        flags["timeout"] |= bool(row.get("timeout"))
    eventual = {
        episode: (
            "collision"
            if flags["collision"]
            else "off_route"
            if flags["off_route"]
            else "success"
            if flags["success"]
            else "timeout"
        )
        for episode, flags in terminal_flags.items()
    }
    changed_proposal = changed_lane = route_matches = route_samples = 0
    actor_mode_matches = actor_mode_samples = 0
    for row in raw_rows:
        decoder = row["target_critic_decoder"]
        choice = _lane_prior_choice(row, coefficient)
        original_lane_index = int(decoder["selected_lane_index"])
        original_component = int(decoder["selected_component_index"])
        changed_proposal += int(
            choice["lane_index"] != original_lane_index
            or choice["component_index"] != original_component
        )
        lane_changed = choice["lane_index"] != original_lane_index
        changed_lane += int(lane_changed)
        outcome = eventual[int(row["episode"])]
        count_by_outcome[outcome] += 1
        changed_lane_by_outcome[outcome] += int(lane_changed)
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        actor_mode = max(
            (index for index, is_valid in enumerate(valid) if is_valid),
            key=lambda index: probabilities[index],
        )
        actor_mode_samples += 1
        actor_mode_matches += int(choice["lane_index"] == actor_mode)
        if bool(row.get("pre_action_route_intent_valid")):
            route_samples += 1
            route_matches += int(
                int(choice["lane_code"]) == int(row["pre_action_route_intent"])
            )
        choices.append(choice)
        original_lanes.append(
            (
                int(row["episode"]),
                int(row["decision"]),
                int(decoder["selected_lane"]),
            )
        )
    count = len(raw_rows)
    counterfactual_lanes = [
        (
            int(row["episode"]),
            int(row["decision"]),
            int(choice["lane_code"]),
        )
        for row, choice in zip(raw_rows, choices)
    ]
    return {
        "coefficient": coefficient,
        "decision_count": count,
        "offline_only_no_action_rewrite": True,
        "closed_loop_outcome_claim_allowed": False,
        "changed_proposal_rate": changed_proposal / count,
        "changed_lane_rate": changed_lane / count,
        "changed_lane_rate_by_eventual_outcome": {
            outcome: changed_lane_by_outcome[outcome] / count_by_outcome[outcome]
            for outcome in sorted(count_by_outcome)
        },
        "original_lane_command_switch_rate": _switch_rate_by_episode(
            original_lanes
        ),
        "counterfactual_lane_command_switch_rate": _switch_rate_by_episode(
            counterfactual_lanes
        ),
        "counterfactual_non_keep_rate": sum(
            int(choice["lane_code"] != 0) for choice in choices
        )
        / count,
        "counterfactual_actor_lane_mode_match_rate": (
            actor_mode_matches / actor_mode_samples
        ),
        "counterfactual_route_intent_match_rate": (
            route_matches / route_samples if route_samples else None
        ),
        "counterfactual_selected_collision_value": shared.numeric_summary(
            float(choice["collision"]) for choice in choices
        ),
        "counterfactual_selected_min_reward_q": shared.numeric_summary(
            float(choice["q"]) for choice in choices
        ),
        "counterfactual_selected_speed_normalized": shared.numeric_summary(
            float(choice["speed"]) for choice in choices
        ),
    }


def analyze_candidate(root: Path) -> dict[str, Any]:
    paths = _run_paths(root, candidate=True)
    paper = _load_json(paths["paper"])
    actions = _load_json(paths["actions"])
    training = _load_json(paths["training"])
    selector = _load_json(paths["selector"])
    raw_rows = _load_jsonl(paths["trace"])
    checkpoint_directory = {
        "highest_training_success": "best",
        "exact_final": "final",
    }.get(str(selector["selected_checkpoint_kind"]))
    _require(
        checkpoint_directory is not None,
        "unknown selected checkpoint kind for calibration trace",
    )
    train_calibration_trace = (
        root
        / "selector"
        / "cal"
        / checkpoint_directory
        / "target_critic"
        / "decisions.jsonl"
    )
    _require(
        train_calibration_trace.is_file(),
        f"missing selected train calibration trace: {train_calibration_trace}",
    )
    train_calibration_rows = _load_jsonl(train_calibration_trace)
    _require(paper["formal_test_accessed"] is False, "formal test was accessed")
    for key in (
        "inference_safety_rule_added",
        "external_kinematic_projection",
        "kinematic_safety_projection",
        "traffic_risk_in_lane_mask",
        "action_postprocessing_override",
    ):
        _require(actions[key] is False, f"forbidden mechanism observed: {key}")
    _require(
        _sha256(paths["trace"]) == actions["trace"]["sha256"],
        "candidate trace hash drifted",
    )
    metrics = [shared.decision_metrics(row) for row in raw_rows]
    episodes = shared.group_episodes(metrics)
    _require(len(episodes) == int(paper["summary"]["episodes"]), "episode drift")
    validation_lane_dynamics = lane_policy_dynamics(raw_rows)
    train_calibration_lane_dynamics = lane_policy_dynamics(
        train_calibration_rows
    )
    return {
        "run_directory": _relative(root),
        "summary": paper["summary"],
        "behavior": _behavior(actions),
        "all_decisions": shared.subset_summary(metrics),
        "by_outcome": {
            outcome: shared.subset_summary(
                [row for row in metrics if row["eventual_outcome"] == outcome]
            )
            for outcome in ("success", "collision", "off_route", "timeout")
        },
        "terminal_windows": {
            outcome: {
                str(window): shared.subset_summary(
                    shared.terminal_rows(
                        episodes, outcome=outcome, window=window
                    )
                )
                for window in (1, 5, 10, 20)
            }
            for outcome in ("success", "collision")
        },
        "imminent_collision_ranking": {
            str(horizon): shared.imminent_collision_ranking(
                episodes, horizon=horizon
            )
            for horizon in (1, 5, 10, 20)
        },
        "lane_command_switch_rate": shared._lane_command_switch_rate(episodes),
        "lane_policy_dynamics": validation_lane_dynamics,
        "proposal_value_separation": proposal_value_separation(raw_rows),
        "collision_risk_sensitivity": {
            str(coefficient): collision_risk_sensitivity(raw_rows, coefficient)
            for coefficient in COLLISION_RISK_COEFFICIENTS
        },
        "lane_prior_sensitivity": {
            str(coefficient): lane_prior_sensitivity(raw_rows, coefficient)
            for coefficient in LANE_PRIOR_COEFFICIENTS
        },
        "training": _training_evidence(training),
        "selector": _selector_evidence(
            selector,
            paper["summary"],
            train_calibration_dynamics=train_calibration_lane_dynamics,
            validation_dynamics=validation_lane_dynamics,
        ),
        "episode_index": _paper_episode_index(paper),
        "source_artifacts": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        }
        | {
            "selected_train_calibration_trace": {
                "path": _relative(train_calibration_trace),
                "sha256": _sha256(train_calibration_trace),
            }
        },
    }


def analyze_control(root: Path) -> dict[str, Any]:
    paths = _run_paths(root, candidate=False)
    paper = _load_json(paths["paper"])
    actions = _load_json(paths["actions"])
    _require(paper["formal_test_accessed"] is False, "formal test was accessed")
    for key in (
        "external_kinematic_projection",
        "kinematic_safety_projection",
        "traffic_risk_in_lane_mask",
        "action_postprocessing_override",
    ):
        _require(paper[key] is False, f"control mechanism drift: {key}")
    return {
        "run_directory": _relative(root),
        "summary": paper["summary"],
        "behavior": _behavior(actions),
        "episode_index": _paper_episode_index(paper),
        "source_artifacts": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def paired_evidence(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    left = control["episode_index"]
    right = candidate["episode_index"]
    _require(set(left) == set(right), "paired evaluation seeds differ")
    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for seed in sorted(left):
        _require(
            left[seed]["traffic_variant"] == right[seed]["traffic_variant"],
            f"traffic pairing drifted at seed {seed}",
        )
        transition = f"{left[seed]['outcome']}__to__{right[seed]['outcome']}"
        transitions[transition] += 1
        rows.append(
            {
                "seed": seed,
                "traffic_variant": left[seed]["traffic_variant"],
                "control_outcome": left[seed]["outcome"],
                "candidate_outcome": right[seed]["outcome"],
            }
        )
    count = len(rows)
    return {
        "paired": True,
        "episodes": count,
        "outcome_transition_counts": dict(sorted(transitions.items())),
        "same_outcome_rate": sum(
            row["control_outcome"] == row["candidate_outcome"] for row in rows
        )
        / count,
        "control_success_lost_count": sum(
            row["control_outcome"] == "success"
            and row["candidate_outcome"] != "success"
            for row in rows
        ),
        "candidate_success_gained_count": sum(
            row["control_outcome"] != "success"
            and row["candidate_outcome"] == "success"
            for row in rows
        ),
        "behavior_delta_candidate_minus_control": _behavior_delta(
            candidate["behavior"], control["behavior"]
        ),
        "episodes_detail": rows,
    }


def _compact_candidate(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": run["summary"],
        "behavior": run["behavior"],
        "imminent_collision_ranking": run["imminent_collision_ranking"],
        "lane_command_switch_rate": run["lane_command_switch_rate"],
        "lane_policy_dynamics": run["lane_policy_dynamics"],
        "proposal_value_separation": run["proposal_value_separation"],
        "collision_risk_sensitivity": run["collision_risk_sensitivity"],
        "lane_prior_sensitivity": run["lane_prior_sensitivity"],
        "selector": run["selector"],
        "training": run["training"],
    }


def build_attribution() -> dict[str, Any]:
    gate = _load_json(PROMOTION_GATE)
    summary = _load_json(PROMOTION_SUMMARY)
    recovery = _load_json(RECOVERY_SUMMARY)
    _require(gate["decision"] == "fail", "promotion gate did not fail")
    _require(summary["complete"] is True, "promotion matrix incomplete")
    _require(int(summary["accepted_runs"]) == 12, "promotion not 12/12")
    _require(gate["formal_test_accessed"] is False, "formal test was accessed")
    selected = _source_map(recovery)
    row_by_key = {
        (str(row["method"]), str(row["scenario"]), int(row["seed"])): row
        for row in summary["per_run"]
    }
    pairs: list[dict[str, Any]] = []
    for scenario in ("cross", "roundabout_medium", "carla"):
        for seed in (20, 21):
            control_row = row_by_key[("temporal_graph", scenario, seed)]
            candidate_row = row_by_key[("prcr_full", scenario, seed)]
            control = analyze_control(selected[str(control_row["job"])])
            candidate = analyze_candidate(selected[str(candidate_row["job"])])
            pair = paired_evidence(control, candidate)
            pairs.append(
                {
                    "scenario": scenario,
                    "train_seed": seed,
                    "control_job": control_row["job"],
                    "candidate_job": candidate_row["job"],
                    "success_rate_delta": float(
                        candidate["summary"]["success_rate"]
                    )
                    - float(control["summary"]["success_rate"]),
                    "collision_rate_delta": float(
                        candidate["summary"]["collision_rate"]
                    )
                    - float(control["summary"]["collision_rate"]),
                    "timeout_rate_delta": float(
                        candidate["summary"]["timeout_rate"]
                    )
                    - float(control["summary"]["timeout_rate"]),
                    "paired_episode_evidence": pair,
                    "control": control,
                    "candidate": candidate,
                }
            )
    _require(len(pairs) == EXPECTED_PAIRS, "expected six paired runs")
    scenario_summary: dict[str, Any] = {}
    for scenario in ("cross", "roundabout_medium", "carla"):
        values = [pair for pair in pairs if pair["scenario"] == scenario]
        scenario_summary[scenario] = {
            "success_rate_delta_mean": _mean(
                pair["success_rate_delta"] for pair in values
            ),
            "collision_rate_delta_mean": _mean(
                pair["collision_rate_delta"] for pair in values
            ),
            "timeout_rate_delta_mean": _mean(
                pair["timeout_rate_delta"] for pair in values
            ),
            "candidate_collision_value_auc_h10_mean": _mean(
                pair["candidate"]["imminent_collision_ranking"]["10"][
                    "selected_collision_value_auc"
                ]
                for pair in values
            ),
            "candidate_lane_change_applied_rate_mean": _mean(
                pair["candidate"]["behavior"]["lane_change_applied_rate"]
                for pair in values
            ),
            "candidate_lane_switch_rate_mean": _mean(
                pair["candidate"]["lane_command_switch_rate"]
                for pair in values
            ),
            "candidate_actor_mode_switch_rate_mean": _mean(
                pair["candidate"]["lane_policy_dynamics"][
                    "actor_mode_switch_rate"
                ]
                for pair in values
            ),
            "candidate_selected_actor_mode_match_rate_mean": _mean(
                pair["candidate"]["lane_policy_dynamics"][
                    "selected_actor_mode_match_rate"
                ]
                for pair in values
            ),
            "candidate_actor_mode_non_keep_rate_mean": _mean(
                pair["candidate"]["lane_policy_dynamics"][
                    "actor_mode_non_keep_rate"
                ]
                for pair in values
            ),
            "candidate_train_minus_validation_actor_switch_mean": _mean(
                pair["candidate"]["selector"][
                    "selected_train_minus_validation_lane_dynamics"
                ].get("actor_mode_switch_rate")
                for pair in values
            ),
            "candidate_train_minus_validation_actor_non_keep_mean": _mean(
                pair["candidate"]["selector"][
                    "selected_train_minus_validation_lane_dynamics"
                ].get("actor_mode_non_keep_rate")
                for pair in values
            ),
            "candidate_feasible_collision_spread_mean": _mean(
                pair["candidate"]["proposal_value_separation"][
                    "feasible_collision_value_spread"
                ]["mean"]
                for pair in values
            ),
            "candidate_lane_min_collision_spread_mean": _mean(
                pair["candidate"]["proposal_value_separation"][
                    "lane_minimum_collision_value_spread"
                ]["mean"]
                for pair in values
            ),
            "candidate_feasible_reward_q_spread_mean": _mean(
                pair["candidate"]["proposal_value_separation"][
                    "feasible_reward_q_spread"
                ]["mean"]
                for pair in values
            ),
            "risk_coef_1_50_changed_lane_rate_mean": _mean(
                pair["candidate"]["collision_risk_sensitivity"]["1.5"][
                    "changed_lane_rate"
                ]
                for pair in values
            ),
            "risk_coef_1_50_selected_collision_mean": _mean(
                pair["candidate"]["collision_risk_sensitivity"]["1.5"][
                    "counterfactual_selected_collision_value"
                ]["mean"]
                for pair in values
            ),
            "risk_coef_2_00_selected_collision_mean": _mean(
                pair["candidate"]["collision_risk_sensitivity"]["2.0"][
                    "counterfactual_selected_collision_value"
                ]["mean"]
                for pair in values
            ),
            "lane_prior_0_05_changed_lane_rate_mean": _mean(
                pair["candidate"]["lane_prior_sensitivity"]["0.05"][
                    "changed_lane_rate"
                ]
                for pair in values
            ),
            "lane_prior_0_05_switch_rate_mean": _mean(
                pair["candidate"]["lane_prior_sensitivity"]["0.05"][
                    "counterfactual_lane_command_switch_rate"
                ]
                for pair in values
            ),
            "lane_prior_0_05_route_match_mean": _mean(
                pair["candidate"]["lane_prior_sensitivity"]["0.05"][
                    "counterfactual_route_intent_match_rate"
                ]
                for pair in values
            ),
        }
    forbidden = {
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "ttc_or_headway_threshold": False,
        "geometry_unsafe_label": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "safety_shield_or_rule_fallback": False,
        "semantic_tie_override": False,
        "post_decoder_action_rewrite": False,
    }
    return {
        "schema_version": "topo-scene-v4.11.complete-promotion-attribution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "complete_paired_observational_model_trace_attribution",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "promotion_complete": True,
        "promotion_gate_decision": "fail",
        "decision_allowed": True,
        "pair_count": len(pairs),
        "run_count": 12,
        "formal_test_accessed": False,
        "formal_test_unlocked": False,
        "forbidden_mechanisms_used": forbidden,
        "offline_counterfactual_scope": (
            "learned_score_reranking_only_no_closed_loop_outcome_claim"
        ),
        "source_artifacts": {
            "promotion_gate": {
                "path": _relative(PROMOTION_GATE),
                "sha256": _sha256(PROMOTION_GATE),
            },
            "promotion_summary": {
                "path": _relative(PROMOTION_SUMMARY),
                "sha256": _sha256(PROMOTION_SUMMARY),
            },
            "recovery_summary": {
                "path": _relative(RECOVERY_SUMMARY),
                "sha256": _sha256(RECOVERY_SUMMARY),
            },
        },
        "failed_gate_checks": [
            key for key, passed in gate["checks"].items() if not passed
        ],
        "gate_effects": {
            "per_scenario_success_delta": gate[
                "per_scenario_success_delta"
            ],
            "per_scenario_collision_delta": gate[
                "per_scenario_collision_delta"
            ],
            "worst_paired_seed_success_delta": gate[
                "worst_paired_seed_success_delta"
            ],
            "macro_success_delta": gate["macro_success_delta"],
        },
        "scenario_summary": scenario_summary,
        "pairs": pairs,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_attribution()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "pair_count": payload["pair_count"],
                "failed_gate_checks": payload["failed_gate_checks"],
                "scenario_summary": payload["scenario_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLISION_RISK_COEFFICIENTS",
    "EXPECTED_PAIRS",
    "LANE_PRIOR_COEFFICIENTS",
    "analyze_candidate",
    "analyze_control",
    "build_attribution",
    "collision_risk_sensitivity",
    "lane_prior_sensitivity",
    "paired_evidence",
    "proposal_value_separation",
]
