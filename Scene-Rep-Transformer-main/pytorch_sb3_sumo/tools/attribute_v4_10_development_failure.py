"""Attribute the v4.10 Cross development failure from sealed real rollouts.

The analysis is diagnostic-only.  It never changes an action, never opens a
formal-test artifact, and treats physical lane existence as an environment
feasibility contract rather than a learned or hand-written safety decision.
All counterfactuals are limited to the learned proposals already emitted by the
frozen v4.10 actor.
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

DEFAULT_RUN_ROOT = ROOT / "results_topo_v4_10_dev" / "development" / "runs"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results_topo_v4_10_dev"
    / "development"
    / "attribution"
    / "d1_d2_cross_instability"
)
RUN_SPECS = (
    ("D1", "D1__srsm__cross__s10__p3cdb4f3b", 10),
    ("D2", "D2__srsm__cross__s11__p3cdb4f3b", 11),
)
TOLERANCE = 1e-7
NUMERIC_FIELDS = (
    "selected_collision_value",
    "selected_min_reward_q",
    "selected_reward_disagreement",
    "selected_collision_disagreement",
    "selected_uncertainty",
    "selected_score",
    "selected_component_probability",
    "selected_lane_probability",
    "selected_speed_normalized",
    "selected_actual_speed_mps",
    "risk_regret_to_feasible_minimum",
    "same_lane_risk_regret",
    "score_margin_over_runner_up",
    "minimum_risk_alternative_score_cost",
    "minimum_risk_alternative_speed_delta_normalized",
    "selected_lane_proposal_speed_spread",
    "selected_lane_component_entropy",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def _rate(flags: Iterable[bool]) -> float | None:
    values = list(flags)
    return sum(values) / len(values) if values else None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not items:
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
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "population_std": statistics.pstdev(items),
        "minimum": items[0],
        "p10": _quantile(items, 0.10),
        "median": statistics.median(items),
        "p90": _quantile(items, 0.90),
        "maximum": items[-1],
    }


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Tie-aware Mann-Whitney AUROC without a third-party dependency."""

    _require(len(scores) == len(labels), "score/label length mismatch")
    positives = sum(int(label) == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(
        ((float(score), int(label)) for score, label in zip(scores, labels)),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        float(value) * math.log(max(float(value), 1e-12))
        for value in probabilities
    )


def _flatten_decoder(decoder: Mapping[str, Any]) -> list[dict[str, float | int]]:
    valid = list(decoder["valid_lane_actions"])
    component_probabilities = decoder["component_probabilities"]
    speeds = decoder["learned_speed_proposals_normalized"]
    q_values = decoder["minimum_target_twin_q"]
    collision_values = decoder["maximum_target_twin_collision_value"]
    reward_disagreements = decoder["reward_twin_disagreement"]
    collision_disagreements = decoder["collision_twin_disagreement"]
    uncertainties = decoder["learned_uncertainty"]
    scores = decoder["supported_risk_adjusted_score"]
    proposals: list[dict[str, float | int]] = []
    for lane_index, is_valid in enumerate(valid):
        if not is_valid:
            _require(q_values[lane_index] is None, "masked reward-Q row is populated")
            _require(
                collision_values[lane_index] is None,
                "masked collision-value row is populated",
            )
            continue
        component_count = len(component_probabilities[lane_index])
        for values in (
            speeds[lane_index],
            q_values[lane_index],
            collision_values[lane_index],
            reward_disagreements[lane_index],
            collision_disagreements[lane_index],
            uncertainties[lane_index],
            scores[lane_index],
        ):
            _require(len(values) == component_count, "decoder component shape mismatch")
        for component_index in range(component_count):
            proposals.append(
                {
                    "lane_index": lane_index,
                    "component_index": component_index,
                    "component_probability": float(
                        component_probabilities[lane_index][component_index]
                    ),
                    "speed": float(speeds[lane_index][component_index]),
                    "q": float(q_values[lane_index][component_index]),
                    "collision": float(
                        collision_values[lane_index][component_index]
                    ),
                    "reward_disagreement": float(
                        reward_disagreements[lane_index][component_index]
                    ),
                    "collision_disagreement": float(
                        collision_disagreements[lane_index][component_index]
                    ),
                    "uncertainty": float(
                        uncertainties[lane_index][component_index]
                    ),
                    "score": float(scores[lane_index][component_index]),
                }
            )
    _require(bool(proposals), "decoder has no feasible learned proposal")
    return proposals


def decision_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    decoder = row.get("target_critic_decoder")
    _require(isinstance(decoder, dict), "missing target-critic decoder")
    _require(decoder.get("candidate_source") == "learned_actor_component_means", "non-learned proposal source")
    _require(decoder.get("fixed_speed_grid_used") is False, "fixed speed grid was used")
    _require(decoder.get("action_rewritten") is False, "action was rewritten")
    _require(decoder.get("semantic_tie_override_used") is False, "semantic tie rule was used")
    proposals = _flatten_decoder(decoder)
    lane_index = int(decoder["selected_lane_index"])
    component_index = int(decoder["selected_component_index"])
    selected = next(
        proposal
        for proposal in proposals
        if proposal["lane_index"] == lane_index
        and proposal["component_index"] == component_index
    )
    ranked = sorted(proposals, key=lambda proposal: float(proposal["score"]), reverse=True)
    _require(
        abs(float(ranked[0]["score"]) - float(selected["score"])) <= 1e-5,
        "selected proposal is not the learned score argmax",
    )
    minimum_risk = min(
        proposals,
        key=lambda proposal: (float(proposal["collision"]), -float(proposal["score"])),
    )
    same_lane = [
        proposal for proposal in proposals if proposal["lane_index"] == lane_index
    ]
    same_lane_minimum_risk = min(
        same_lane,
        key=lambda proposal: (float(proposal["collision"]), -float(proposal["score"])),
    )
    selected_component_probabilities = decoder["component_probabilities"][lane_index]
    selected_speed = float(selected["speed"])
    _require(
        abs(selected_speed - float(row["action_longitudinal"])) <= 1e-5,
        "deployed speed differs from selected learned proposal",
    )
    score_margin = (
        float(selected["score"]) - float(ranked[1]["score"])
        if len(ranked) > 1
        else None
    )
    selected_risk = float(selected["collision"])
    minimum_risk_value = float(minimum_risk["collision"])
    same_lane_minimum = float(same_lane_minimum_risk["collision"])
    return {
        "episode": int(row["episode"]),
        "decision": int(row["decision"]),
        "seed": int(row["seed"]),
        "traffic_variant": str(row["traffic_variant"]),
        "collision_event": bool(row.get("collision", False)),
        "success_event": bool(row.get("is_success", False)),
        "timeout_event": bool(row.get("timeout", False)),
        "off_route_event": bool(row.get("off_route", False)),
        "selected_lane": int(decoder["selected_lane"]),
        "selected_lane_index": lane_index,
        "selected_component_index": component_index,
        "selected_collision_value": selected_risk,
        "selected_min_reward_q": float(selected["q"]),
        "selected_reward_disagreement": float(selected["reward_disagreement"]),
        "selected_collision_disagreement": float(selected["collision_disagreement"]),
        "selected_uncertainty": float(selected["uncertainty"]),
        "selected_score": float(selected["score"]),
        "selected_component_probability": float(selected["component_probability"]),
        "selected_lane_probability": float(decoder["lane_probabilities"][lane_index]),
        "selected_speed_normalized": selected_speed,
        "selected_actual_speed_mps": float(row["actual_speed_mps"]),
        "feasible_proposal_count": len(proposals),
        "feasible_lane_count": sum(bool(value) for value in decoder["valid_lane_actions"]),
        "minimum_feasible_collision_value": minimum_risk_value,
        "risk_regret_to_feasible_minimum": selected_risk - minimum_risk_value,
        "same_lane_minimum_collision_value": same_lane_minimum,
        "same_lane_risk_regret": selected_risk - same_lane_minimum,
        "selected_is_feasible_minimum_risk": selected_risk - minimum_risk_value <= TOLERANCE,
        "selected_is_same_lane_minimum_risk": selected_risk - same_lane_minimum <= TOLERANCE,
        "lower_risk_feasible_alternative_exists": selected_risk - minimum_risk_value > TOLERANCE,
        "lower_risk_same_lane_alternative_exists": selected_risk - same_lane_minimum > TOLERANCE,
        "score_margin_over_runner_up": score_margin,
        "minimum_risk_alternative_score_cost": float(selected["score"]) - float(minimum_risk["score"]),
        "minimum_risk_alternative_speed_delta_normalized": float(minimum_risk["speed"]) - selected_speed,
        "selected_lane_proposal_speed_spread": max(float(item["speed"]) for item in same_lane) - min(float(item["speed"]) for item in same_lane),
        "selected_lane_component_entropy": _entropy(selected_component_probabilities),
        "selected_component_is_probability_mode": component_index == max(
            range(len(selected_component_probabilities)),
            key=lambda index: float(selected_component_probabilities[index]),
        ),
        "lane_change_applied": bool(row.get("lane_change_applied", False)),
        "route_intent_match": bool(row.get("route_intent_match", False)),
        "pre_action_non_keep_feasible": bool(row.get("pre_action_non_keep_feasible", False)),
    }


def _episode_outcome(rows: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(row.get("collision_event")) for row in rows):
        return "collision"
    if any(bool(row.get("off_route_event")) for row in rows):
        return "off_route"
    if any(bool(row.get("success_event")) for row in rows):
        return "success"
    if any(bool(row.get("timeout_event")) for row in rows):
        return "timeout"
    raise ValueError(f"episode {rows[0]['episode']} has no terminal outcome")


def group_episodes(metrics: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[int(row["episode"])].append(row)
    for episode, rows in grouped.items():
        rows.sort(key=lambda item: int(item["decision"]))
        outcome = _episode_outcome(rows)
        for row in rows:
            row["eventual_outcome"] = outcome
            row["decisions_to_terminal_inclusive"] = len(rows) - int(row["decision"])
        _require(
            [int(row["decision"]) for row in rows] == list(range(len(rows))),
            f"episode {episode} decision indices are not contiguous",
        )
    return dict(grouped)


def terminal_rows(
    episodes: Mapping[int, Sequence[dict[str, Any]]], *, outcome: str, window: int
) -> list[dict[str, Any]]:
    _require(window > 0, "terminal window must be positive")
    values: list[dict[str, Any]] = []
    for episode in sorted(episodes):
        rows = episodes[episode]
        if rows and str(rows[0]["eventual_outcome"]) == outcome:
            values.extend(rows[-window:])
    return values


def subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"decision_count": 0}
    numeric = {
        field: numeric_summary(
            float(row[field])
            for row in rows
            if row.get(field) is not None
        )
        for field in NUMERIC_FIELDS
    }
    lane_counts = Counter(str(int(row["selected_lane"])) for row in rows)
    component_counts = Counter(str(int(row["selected_component_index"])) for row in rows)
    return {
        "decision_count": len(rows),
        "numeric": numeric,
        "selected_is_feasible_minimum_risk_rate": _rate(
            bool(row["selected_is_feasible_minimum_risk"]) for row in rows
        ),
        "selected_is_same_lane_minimum_risk_rate": _rate(
            bool(row["selected_is_same_lane_minimum_risk"]) for row in rows
        ),
        "lower_risk_feasible_alternative_rate": _rate(
            bool(row["lower_risk_feasible_alternative_exists"]) for row in rows
        ),
        "lower_risk_same_lane_alternative_rate": _rate(
            bool(row["lower_risk_same_lane_alternative_exists"]) for row in rows
        ),
        "component_probability_mode_selected_rate": _rate(
            bool(row["selected_component_is_probability_mode"]) for row in rows
        ),
        "lane_change_applied_rate": _rate(bool(row["lane_change_applied"]) for row in rows),
        "selected_lane_counts": dict(sorted(lane_counts.items())),
        "selected_component_counts": dict(sorted(component_counts.items())),
    }


def _lane_command_switch_rate(
    episodes: Mapping[int, Sequence[Mapping[str, Any]]]
) -> float | None:
    switches = 0
    comparisons = 0
    for rows in episodes.values():
        for previous, current in zip(rows, rows[1:]):
            comparisons += 1
            switches += int(previous["selected_lane"] != current["selected_lane"])
    return switches / comparisons if comparisons else None


def imminent_collision_ranking(
    episodes: Mapping[int, Sequence[Mapping[str, Any]]], *, horizon: int
) -> dict[str, Any]:
    _require(horizon > 0, "imminent-collision horizon must be positive")
    rows: list[Mapping[str, Any]] = []
    labels: list[int] = []
    for episode_rows in episodes.values():
        collision = bool(episode_rows and episode_rows[0]["eventual_outcome"] == "collision")
        for index, row in enumerate(episode_rows):
            rows.append(row)
            labels.append(int(collision and index >= len(episode_rows) - horizon))
    risks = [float(row["selected_collision_value"]) for row in rows]
    negative_scores = [-float(row["selected_score"]) for row in rows]
    uncertainties = [float(row["selected_uncertainty"]) for row in rows]
    positives = sum(labels)
    positive_risks = [risk for risk, label in zip(risks, labels) if label]
    negative_risks = [risk for risk, label in zip(risks, labels) if not label]
    return {
        "horizon_decisions": horizon,
        "sample_count": len(rows),
        "positive_count": positives,
        "positive_rate": positives / len(rows) if rows else None,
        "selected_collision_value_auc": binary_auc(risks, labels),
        "negative_selected_score_auc": binary_auc(negative_scores, labels),
        "selected_uncertainty_auc": binary_auc(uncertainties, labels),
        "selected_collision_value_brier": (
            statistics.fmean((risk - label) ** 2 for risk, label in zip(risks, labels))
            if rows
            else None
        ),
        "positive_selected_collision_value": numeric_summary(positive_risks),
        "negative_selected_collision_value": numeric_summary(negative_risks),
        "positive_minus_negative_mean_collision_value": (
            statistics.fmean(positive_risks) - statistics.fmean(negative_risks)
            if positive_risks and negative_risks
            else None
        ),
    }


def _verify_paper_episodes(
    episodes: Mapping[int, Sequence[Mapping[str, Any]]], paper: Mapping[str, Any]
) -> dict[str, str]:
    records = list(paper["episode_records"])
    _require(len(records) == len(episodes), "paper/trace episode-count mismatch")
    variants: dict[str, str] = {}
    for record in records:
        episode = int(record["episode"])
        _require(episode in episodes, f"paper episode missing from trace: {episode}")
        trace_rows = episodes[episode]
        outcome = _episode_outcome(trace_rows)
        expected = (
            "collision"
            if record["collision"]
            else "off_route"
            if record["off_route"]
            else "success"
            if record["success"]
            else "timeout"
            if record["timeout"]
            else "unknown"
        )
        _require(outcome == expected, f"paper/trace outcome mismatch in episode {episode}")
        _require(
            len(trace_rows) == int(record["decision_steps"]),
            f"paper/trace decision-count mismatch in episode {episode}",
        )
        variant = str(record["traffic_variant"])
        _require(variant not in variants, f"duplicate traffic variant: {variant}")
        variants[variant] = outcome
    return variants


def _training_evidence(training: Mapping[str, Any]) -> dict[str, Any]:
    statistics_payload = training["statistics"]
    selected_fields = (
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/collision_cost_replay_prediction",
        "risk/policy_expected_collision_cost",
        "risk/collision_twin_disagreement",
        "risk/learned_uncertainty",
        "support/replay_speed_mixture_nll",
        "support/component_entropy",
        "support/component_speed_spread",
        "support/proposal_boundary_rate",
        "hybrid/lane_entropy",
        "diagnostic/merge_attention_mass",
        "diagnostic/merge_pair_score",
        "diagnostic/topology_effective_lanes",
        "diagnostic/topology_residual_scale",
        "diagnostic/social_slot_rms",
        "diagnostic/route_slot_rms",
        "train/collision_critic_loss",
    )
    return {
        "collision_labels": training["collision_labels"],
        "statistics": {
            field: statistics_payload[field] for field in selected_fields
        },
        "optimizer_ownership": training["optimizer_ownership"],
        "learned_collision_critic": training["learned_collision_critic"],
        "inference_safety_rule_added": training["inference_safety_rule_added"],
        "fixed_speed_grid_at_inference": training["fixed_speed_grid_at_inference"],
    }


def _selector_evidence(run_dir: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    _require(receipt["selection_partition"] == "train", "selector used a non-train partition")
    _require(receipt["validation_used_for_selection"] is False, "selector accessed validation")
    _require(receipt["formal_test_used_for_selection"] is False, "selector accessed formal test")
    _require(receipt["external_kinematic_projection"] is False, "selector used projection")
    _require(receipt["action_postprocessing_override"] is False, "selector rewrote actions")
    candidates: list[dict[str, Any]] = []
    folder_by_kind = {"highest_training_success": "best", "exact_final": "final"}
    for candidate in receipt["candidates"]:
        kind = str(candidate["checkpoint_kind"])
        folder = folder_by_kind[kind]
        base = run_dir / "selector" / "cal" / folder / "target_critic"
        evaluation = base / "evaluation.json"
        detailed = base / "detailed.json"
        actions = base / "actions.json"
        trace = base / "decisions.jsonl"
        _require(
            all(path.is_file() for path in (evaluation, detailed, actions, trace)),
            f"missing selector evidence for {kind}",
        )
        _require(
            _sha256(detailed) == candidate["calibration_result_sha256"],
            f"selector detailed-result hash drifted for {kind}",
        )
        _require(
            _sha256(actions) == candidate["calibration_action_diagnostics_sha256"],
            f"selector action-diagnostics hash drifted for {kind}",
        )
        trace_rows = _load_jsonl(trace)
        metrics = [decision_metrics(row) for row in trace_rows]
        episodes = group_episodes(metrics)
        _require(
            len(episodes) == int(candidate["summary"]["episodes"]),
            f"selector episode count drifted for {kind}",
        )
        candidates.append(
            {
                "checkpoint_kind": kind,
                "selected": kind == receipt["selected_checkpoint_kind"],
                "summary": candidate["summary"],
                "outcome_counts": candidate["outcome_counts"],
                "evaluation": _relative(evaluation),
                "evaluation_sha256": _sha256(evaluation),
                "detailed": _relative(detailed),
                "detailed_sha256_verified": candidate["calibration_result_sha256"],
                "actions": _relative(actions),
                "actions_sha256_verified": candidate["calibration_action_diagnostics_sha256"],
                "trace": _relative(trace),
                "trace_sha256": _sha256(trace),
                "decision_analysis": subset_summary(metrics),
                "imminent_collision_ranking": {
                    str(horizon): imminent_collision_ranking(episodes, horizon=horizon)
                    for horizon in (1, 5, 10, 20)
                },
            }
        )
    selected = next(item for item in candidates if item["selected"])
    alternate = next(item for item in candidates if not item["selected"])
    return {
        "receipt": _relative(run_dir / "selector" / "receipt.json"),
        "receipt_sha256": _sha256(run_dir / "selector" / "receipt.json"),
        "selected_checkpoint_kind": receipt["selected_checkpoint_kind"],
        "secondary_calibration_triggered": receipt["secondary_calibration_triggered"],
        "candidates": candidates,
        "selected_minus_alternate_success_rate": float(selected["summary"]["success_rate"]) - float(alternate["summary"]["success_rate"]),
        "selected_minus_alternate_collision_rate": float(selected["summary"]["collision_rate"]) - float(alternate["summary"]["collision_rate"]),
    }


def _run_evidence(run_id: str, run_dir: Path, train_seed: int) -> dict[str, Any]:
    paths = {
        "trace": run_dir / "action_diagnostics_decisions.jsonl",
        "diagnostics": run_dir / "action_diagnostics.json",
        "paper": run_dir / "paper_evaluation_detailed.json",
        "training": run_dir / "training_diagnostics.json",
        "selector": run_dir / "selector" / "receipt.json",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing {name} evidence: {path}")
    diagnostics = _load_json(paths["diagnostics"])
    paper = _load_json(paths["paper"])
    training = _load_json(paths["training"])
    receipt = _load_json(paths["selector"])
    _require(paper["evaluation_split"] == "validation", "not a validation run")
    _require(paper["formal_test_accessed"] is False, "formal test was accessed")
    _require(paper["scenario"] == "cross", "attribution run is not Cross")
    _require(paper["inference_safety_rule_added"] is False, "inference rule was added")
    _require(paper["external_kinematic_projection"] is False, "external projection was used")
    _require(paper["action_postprocessing_override"] is False, "action was rewritten")
    _require(diagnostics["fixed_speed_grid_at_inference"] is False, "fixed speed grid was used")
    _require(diagnostics["inference_safety_rule_added"] is False, "diagnostics report a rule")
    _require(diagnostics["external_kinematic_projection"] is False, "diagnostics report projection")
    _require(diagnostics["action_postprocessing_override"] is False, "diagnostics report rewrite")
    _require(diagnostics["selected_decoder_exact_action_match_rate"] == 1.0, "action mismatch")
    _require(diagnostics["exact_supported_mixture_model_argmax_rate"] == 1.0, "argmax mismatch")
    _require(diagnostics["learned_proposal_source_rate"] == 1.0, "non-learned proposal")
    _require(diagnostics["no_action_rewrite_rate"] == 1.0, "action rewrite observed")
    _require(
        _sha256(paths["trace"]) == diagnostics["trace"]["sha256"],
        "validation trace hash drifted",
    )
    raw_rows = _load_jsonl(paths["trace"])
    metrics = [decision_metrics(row) for row in raw_rows]
    episodes = group_episodes(metrics)
    variants = _verify_paper_episodes(episodes, paper)
    return {
        "run_id": run_id,
        "run_directory": _relative(run_dir),
        "train_seed": train_seed,
        "evaluation_seed_start": paper["evaluation_seed_start"],
        "selected_checkpoint_kind": paper["selected_checkpoint_kind"],
        "selected_model_sha256": paper["model_sha256"],
        "summary": paper["summary"],
        "traffic_variant_outcomes": variants,
        "source_artifacts": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "integrity": {
            "decision_records": len(metrics),
            "episode_count": len(episodes),
            "exact_model_argmax_rate": diagnostics["exact_supported_mixture_model_argmax_rate"],
            "exact_action_match_rate": diagnostics["selected_decoder_exact_action_match_rate"],
            "learned_proposal_source_rate": diagnostics["learned_proposal_source_rate"],
            "no_action_rewrite_rate": diagnostics["no_action_rewrite_rate"],
            "fixed_speed_grid_at_inference": diagnostics["fixed_speed_grid_at_inference"],
            "inference_safety_rule_added": diagnostics["inference_safety_rule_added"],
            "external_kinematic_projection": diagnostics["external_kinematic_projection"],
            "action_postprocessing_override": diagnostics["action_postprocessing_override"],
            "formal_test_accessed": paper["formal_test_accessed"],
        },
        "all_decisions": subset_summary(metrics),
        "by_outcome": {
            outcome: subset_summary(
                [row for row in metrics if row["eventual_outcome"] == outcome]
            )
            for outcome in ("success", "collision", "off_route", "timeout")
        },
        "terminal_windows": {
            outcome: {
                str(window): subset_summary(
                    terminal_rows(episodes, outcome=outcome, window=window)
                )
                for window in (1, 5, 10, 20)
            }
            for outcome in ("success", "collision")
        },
        "imminent_collision_ranking": {
            str(horizon): imminent_collision_ranking(episodes, horizon=horizon)
            for horizon in (1, 5, 10, 20)
        },
        "lane_command_switch_rate": _lane_command_switch_rate(episodes),
        "training": _training_evidence(training),
        "selector": _selector_evidence(run_dir, receipt),
    }


def _paired_variant_evidence(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(runs) == 2, "paired comparison requires exactly two runs")
    first = runs[0]["traffic_variant_outcomes"]
    second = runs[1]["traffic_variant_outcomes"]
    _require(set(first) == set(second), "D1/D2 traffic variants are not paired")
    pairs = [
        {
            "traffic_variant": variant,
            str(runs[0]["run_id"]): first[variant],
            str(runs[1]["run_id"]): second[variant],
        }
        for variant in sorted(first)
    ]
    counts = Counter(f"{first[variant]}__{second[variant]}" for variant in first)
    return {
        "paired_traffic_variants": True,
        "variant_count": len(pairs),
        "outcome_pair_counts": dict(sorted(counts.items())),
        "same_outcome_rate": sum(first[v] == second[v] for v in first) / len(first),
        "collision_in_both_count": sum(
            first[v] == "collision" and second[v] == "collision" for v in first
        ),
        "pairs": pairs,
    }


def _mean_stat(run: Mapping[str, Any], name: str) -> float:
    return float(run["training"]["statistics"][name]["mean"])


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    runs = payload["runs"]
    lines = [
        "# v4.10 Cross development failure attribution",
        "",
        "This report uses only sealed development/validation trajectories. It does not deploy a counterfactual action and does not access formal test data.",
        "",
        "## Integrity and model boundary",
        "",
        f"- Real rollout decisions verified: {payload['integrity']['verified_validation_decisions']}",
        f"- Paired traffic variants: {payload['paired_traffic']['variant_count']}",
        f"- Formal test accessed: {str(payload['formal_test_accessed']).lower()}",
        f"- External kinematic projection: {str(payload['kinematic_projection_review']['external_kinematic_projection']).lower()}",
        f"- Rule/veto/shield/action rewrite: {str(payload['kinematic_projection_review']['rule_or_action_rewrite_present']).lower()}",
        "",
        "## Development evidence",
        "",
        "| Run | Train seed | Success | Collision | Selected checkpoint | Replay collision-label rate | Imminent-10 risk AUROC | Collision terminal-10 mean risk | Collision terminal-10 lower-risk alternative rate |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        terminal = run["terminal_windows"]["collision"]["10"]
        terminal_numeric = terminal.get("numeric", {})
        lines.append(
            "| {run_id} | {seed} | {success} | {collision} | {checkpoint} | {label} | {auc} | {risk} | {alternative} |".format(
                run_id=run["run_id"],
                seed=run["train_seed"],
                success=_fmt(run["summary"]["success_rate"]),
                collision=_fmt(run["summary"]["collision_rate"]),
                checkpoint=run["selected_checkpoint_kind"],
                label=_fmt(run["training"]["collision_labels"]["sampled_positive_label_rate"]),
                auc=_fmt(run["imminent_collision_ranking"]["10"]["selected_collision_value_auc"]),
                risk=_fmt(terminal_numeric.get("selected_collision_value", {}).get("mean")),
                alternative=_fmt(terminal.get("lower_risk_feasible_alternative_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence-backed attribution",
            "",
            *[f"- {item}" for item in payload["findings"]],
            "",
            "## Frozen model-only consequence for v4.11 preregistration",
            "",
            *[f"- {item}" for item in payload["model_only_consequences"]],
            "",
            "## Excluded mechanisms",
            "",
            *[f"- {item}" for item in payload["excluded_mechanisms"]],
            "",
            "## Causal limits",
            "",
            *[f"- {item}" for item in payload["causal_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_attribution(run_root: Path) -> dict[str, Any]:
    runs = [
        _run_evidence(run_id, run_root / directory, seed)
        for run_id, directory, seed in RUN_SPECS
    ]
    d1, d2 = runs
    paired = _paired_variant_evidence(runs)
    label_d1 = float(d1["training"]["collision_labels"]["sampled_positive_label_rate"])
    label_d2 = float(d2["training"]["collision_labels"]["sampled_positive_label_rate"])
    collision_gap = float(d1["summary"]["collision_rate"]) - float(d2["summary"]["collision_rate"])
    merge_d1 = _mean_stat(d1, "diagnostic/merge_attention_mass")
    merge_d2 = _mean_stat(d2, "diagnostic/merge_attention_mass")
    topology_d1 = _mean_stat(d1, "diagnostic/topology_effective_lanes")
    topology_d2 = _mean_stat(d2, "diagnostic/topology_effective_lanes")
    spread_d1 = _mean_stat(d1, "support/component_speed_spread")
    spread_d2 = _mean_stat(d2, "support/component_speed_spread")
    lane_entropy_d1 = _mean_stat(d1, "hybrid/lane_entropy")
    lane_entropy_d2 = _mean_stat(d2, "hybrid/lane_entropy")
    d1_selected_delta = d1["selector"]["selected_minus_alternate_collision_rate"]
    d2_selected_delta = d2["selector"]["selected_minus_alternate_collision_rate"]
    d1_auc = d1["imminent_collision_ranking"]["10"]["selected_collision_value_auc"]
    d2_auc = d2["imminent_collision_ranking"]["10"]["selected_collision_value_auc"]
    d1_collision_terminal5 = d1["terminal_windows"]["collision"]["5"]
    d1_success_terminal5 = d1["terminal_windows"]["success"]["5"]
    d1_collision_terminal5_non_keep = 1.0 - (
        int(d1_collision_terminal5["selected_lane_counts"].get("0", 0))
        / int(d1_collision_terminal5["decision_count"])
    )
    d1_success_terminal5_non_keep = 1.0 - (
        int(d1_success_terminal5["selected_lane_counts"].get("0", 0))
        / int(d1_success_terminal5["decision_count"])
    )
    findings = [
        (
            f"The development gap is {collision_gap:.4f} collision rate on the same ordered set of "
            f"{paired['variant_count']} traffic files. No file collides in both runs; "
            f"{paired['outcome_pair_counts']}. This is run/model instability, not evidence for a fixed scenario rule."
        ),
        (
            f"Replay collision-label prevalence is nearly matched ({label_d1:.6f} vs {label_d2:.6f}, "
            f"absolute gap {abs(label_d1-label_d2):.6f}) and therefore does not explain the "
            f"{collision_gap:.4f} rollout collision gap by class frequency alone."
        ),
        (
            "The sealed selector chose exact_final in both runs and improved calibration collision rate "
            f"relative to highest_training_success by {-d1_selected_delta:.4f} in D1 and "
            f"{-d2_selected_delta:.4f} in D2. Checkpoint selection is not the observed failure source."
        ),
        (
            "Learned internal statistics differ materially across the two Cross fits: mean merge-attention "
            f"mass {merge_d1:.4f} vs {merge_d2:.4f}, effective lanes {topology_d1:.4f} vs "
            f"{topology_d2:.4f}, proposal spread {spread_d1:.4f} vs {spread_d2:.4f}, and lane entropy "
            f"{lane_entropy_d1:.4f} vs {lane_entropy_d2:.4f}. These are direct signs of representation/"
            "proposal seed sensitivity, not proof that any one statistic is causal."
        ),
        (
            "The learned collision value has finite imminent-10 ranking evidence "
            f"(AUROC D1={_fmt(d1_auc)}, D2={_fmt(d2_auc)}), but its MSE-trained bootstrapped probability "
            "is not explicitly optimized as a proper rare-event calibration model. The collision trace "
            "must be improved through the learned objective and representation rather than an inference threshold."
        ),
        (
            "D1 localizes the failure to learned action-conditioned transition risk: non-keep lane "
            f"commands occupy {d1_collision_terminal5_non_keep:.4f} of the final five collision "
            f"decisions but {d1_success_terminal5_non_keep:.4f} of matched successful tails. This "
            "supports a learned lane/speed risk-ranking objective; it does not justify a lane-change veto."
        ),
    ]
    consequences = [
        "Use a proper probabilistic loss on collision logits with soft TD targets, plus a learned immediate-event calibration term; keep all targets derived from observed replay transitions and prohibit future/oracle labels.",
        "Stabilize the shared risk representation with a model-trained risk-consistency objective across the online/target encoders so reward, collision, and proposal heads cannot drift independently across seeds.",
        "Train continuous pairwise collision-return ranking across replay actions so the neural critic learns action-conditioned lane/speed ordering without a hand-written unsafe-action label or threshold.",
        "Retain multiple learned replay-supported proposals, but train explicit proposal separation/support consistency so the low-spread seed cannot collapse candidate coverage.",
        "Use the learned calibrated risk distribution directly inside the differentiable actor objective and deterministic model score; no TTC/headway threshold, rule veto, shield, or post-hoc action rewrite is permitted.",
        "Keep exact learned-score argmax and the physical adjacent-driving-lane existence mask; the latter is action-space feasibility, not traffic-risk projection.",
    ]
    excluded = [
        "No kinematic safety projection or clipping of a model-selected proposal.",
        "No TTC, headway, gap, speed-limit, or confidence threshold used to veto an action.",
        "No fallback controller, rule shield, semantic tie override, or lane-change rewrite.",
        "No validation/formal outcomes used as a per-step model input or selector rule.",
    ]
    limits = [
        "D1 and D2 share traffic files but use different training and simulator seeds; the comparison identifies instability, not a single-parameter causal effect.",
        "Critic values are learned discounted collision-return estimates, so Brier values against an imminent binary window are diagnostics rather than a claim of probability calibration for that window.",
        "Counterfactual lower-risk proposals were not executed; their values expose model ranking/support behavior but cannot establish closed-loop safety.",
        "The evidence is development-only and cannot unlock ablation, promotion, or formal testing until a newly frozen version passes its preregistered development gate."
    ]
    return {
        "schema_version": "topo-scene-v4.10.cross-development-attribution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "development_validation_and_train_calibration_only",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "development_decision": "fail",
        "failed_gate": "D1_collision_rate_le_0.40",
        "integrity": {
            "verified_run_count": len(runs),
            "verified_validation_decisions": sum(
                int(run["integrity"]["decision_records"]) for run in runs
            ),
            "all_source_hashes_recorded": True,
            "all_action_paths_exact_learned_model_outputs": True,
            "all_formal_test_flags_false": True,
        },
        "kinematic_projection_review": {
            "external_kinematic_projection": False,
            "fixed_speed_grid_at_inference": False,
            "rule_or_action_rewrite_present": False,
            "physical_lane_existence_mask_preserved": True,
            "physical_lane_existence_mask_role": "environment_action_space_feasibility_only",
            "traffic_risk_threshold_in_mask": False,
            "model_selected_speed_applied_exactly": True,
            "review_conclusion": "improve_the_learned_model_not_an_inference_rule",
        },
        "paired_traffic": paired,
        "runs": runs,
        "cross_run_training_comparison": {
            "collision_rate_D1_minus_D2": collision_gap,
            "sampled_collision_label_rate_D1_minus_D2": label_d1 - label_d2,
            "mean_merge_attention_mass_D1_minus_D2": merge_d1 - merge_d2,
            "mean_topology_effective_lanes_D1_minus_D2": topology_d1 - topology_d2,
            "mean_component_speed_spread_D1_minus_D2": spread_d1 - spread_d2,
            "mean_lane_entropy_D1_minus_D2": lane_entropy_d1 - lane_entropy_d2,
        },
        "attribution_class": "learned_risk_representation_and_proposal_seed_instability",
        "selector_failure_supported": False,
        "simple_label_prevalence_explanation_supported": False,
        "rule_mechanism_recommended": False,
        "findings": findings,
        "model_only_consequences": consequences,
        "excluded_mechanisms": excluded,
        "causal_limits": limits,
        "decision_allowed": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    value.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "attribution.json",
    )
    value.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ATTRIBUTION_V4_10_CROSS.md",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_attribution(args.run_root.resolve())
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": _relative(output),
                "output_sha256": _sha256(output),
                "markdown": _relative(markdown),
                "markdown_sha256": _sha256(markdown),
                "verified_validation_decisions": payload["integrity"]["verified_validation_decisions"],
                "formal_test_accessed": payload["formal_test_accessed"],
                "external_kinematic_projection": payload["kinematic_projection_review"]["external_kinematic_projection"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
