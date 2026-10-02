"""Attribute the v4.12 roundabout-medium development regression.

This tool is deliberately read-only with respect to trained runs.  It compares
the immutable v4.11 parent and v4.12 candidate artifacts, verifies the absence
of rule/projection mechanisms, and removes only the continuous learned lane-log
prior from already-recorded v4.12 model scores.  The latter is an offline
ranking counterfactual and must not be interpreted as a closed-loop outcome.
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


PARENT_RUN = (
    ROOT
    / "results_topo_v4_11_dev"
    / "development"
    / "runs"
    / "D3__prcr__ram__s12__pc71e98e7"
)
CANDIDATE_RUN = ROOT / "r412" / "d" / "D3__ajs__ram__s12__p3f978ef9"
DEVELOPMENT_DECISION = (
    ROOT / "results_topo_v4_12_dev" / "development" / "development_decision.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "attribution.json"
LANE_CODES = (-1, 0, 1)
LANE_PRIOR_SWEEP = (0.0, 0.025, 0.05, 0.075, 0.10)
TRAINING_STATISTICS = (
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
    "diagnostic/topology_effective_lanes",
    "diagnostic/topology_residual_scale",
)
FORBIDDEN_FIELDS = (
    "inference_safety_rule_added",
    "external_kinematic_projection",
    "kinematic_safety_projection",
    "traffic_risk_in_lane_mask",
    "actor_confidence_gate",
    "action_postprocessing_override",
)


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
    return path.resolve().relative_to(ROOT).as_posix()


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


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


def _episode_index(paper: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in paper["episode_records"]:
        episode = int(row["episode"])
        _require(episode not in output, f"duplicate episode index {episode}")
        output[episode] = {
            "episode": episode,
            "seed": int(row["seed"]),
            "traffic_variant": str(row["traffic_variant"]),
            "outcome": _outcome(row),
            "decision_steps": int(row["decision_steps"]),
        }
    return output


def episode_alignment(
    parent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Align by rollout position while explicitly detecting seed drift."""

    left = _episode_index(parent)
    right = _episode_index(candidate)
    _require(set(left) == set(right), "episode-index sets differ")
    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for episode in sorted(left):
        _require(
            left[episode]["traffic_variant"] == right[episode]["traffic_variant"],
            f"traffic variant differs at episode {episode}",
        )
        transition = f"{left[episode]['outcome']}__to__{right[episode]['outcome']}"
        transitions[transition] += 1
        rows.append(
            {
                "episode": episode,
                "traffic_variant": left[episode]["traffic_variant"],
                "parent_seed": left[episode]["seed"],
                "candidate_seed": right[episode]["seed"],
                "same_seed": left[episode]["seed"] == right[episode]["seed"],
                "parent_outcome": left[episode]["outcome"],
                "candidate_outcome": right[episode]["outcome"],
            }
        )
    exact_seed_pairing = all(row["same_seed"] for row in rows)
    return {
        "episode_count": len(rows),
        "aligned_by_episode_position_and_traffic_variant": True,
        "exact_seed_paired": exact_seed_pairing,
        "causal_paired_outcome_claim_allowed": exact_seed_pairing,
        "seed_start_parent": min(row["parent_seed"] for row in rows),
        "seed_start_candidate": min(row["candidate_seed"] for row in rows),
        "outcome_transition_counts_descriptive_only": dict(
            sorted(transitions.items())
        ),
        "episodes": rows,
    }


def _behavior(actions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_records": int(actions["decision_records"]),
        "negative_command_rate": float(actions["lane_command_rates"]["negative"]),
        "keep_command_rate": float(actions["lane_command_rates"]["keep"]),
        "positive_command_rate": float(actions["lane_command_rates"]["positive"]),
        "non_keep_command_rate": (
            float(actions["lane_command_rates"]["negative"])
            + float(actions["lane_command_rates"]["positive"])
        ),
        "lane_change_applied_rate": float(actions["lane_change_applied_rate"]),
        "valid_route_intent_match_rate": actions.get(
            "valid_route_intent_match_rate"
        ),
        "mean_actual_speed_mps": actions["actual_speed_mps"]["mean"],
        "p90_actual_speed_mps": actions["actual_speed_mps"]["p90"],
        "mean_action_longitudinal": actions["action_longitudinal"]["mean"],
    }


def _numeric_delta(
    candidate: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in candidate:
        left = candidate.get(key)
        right = parent.get(key)
        output[key] = (
            float(left) - float(right)
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else None
        )
    return output


def _training_comparison(
    parent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = parent["statistics"]
    right = candidate["statistics"]
    values: dict[str, Any] = {}
    for key in TRAINING_STATISTICS:
        if key not in left or key not in right:
            continue
        values[key] = {
            "parent": left[key],
            "candidate": right[key],
            "candidate_minus_parent": {
                moment: float(right[key][moment]) - float(left[key][moment])
                for moment in ("mean", "last")
            },
        }
    if "support/replay_speed_mixture_nll" in left:
        values["support/speed_nll_cross_version"] = {
            "parent_metric": "support/replay_speed_mixture_nll",
            "candidate_metric": "support/replay_conditional_speed_mixture_nll",
            "parent": left["support/replay_speed_mixture_nll"],
            "candidate": right["support/replay_conditional_speed_mixture_nll"],
            "candidate_minus_parent": {
                moment: float(
                    right["support/replay_conditional_speed_mixture_nll"][moment]
                )
                - float(left["support/replay_speed_mixture_nll"][moment])
                for moment in ("mean", "last")
            },
        }
    values["candidate_only_joint_support"] = {
        key: right[key]
        for key in (
            "support/replay_lane_categorical_nll",
            "support/replay_conditional_speed_mixture_nll",
            "support/replay_joint_action_nll_recomputed",
        )
    }
    return values


def _label_comparison(
    parent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    common = (
        "stored_transition_count_including_source_duplicates",
        "stored_collision_event_count_including_source_duplicates",
        "sampled_label_count",
        "sampled_positive_label_rate",
        "sampled_mean_discounted_collision_cost",
    )
    return {
        "same_observed_label_definition": all(
            parent[key] == candidate[key]
            for key in (
                "source",
                "future_or_oracle_labels_used",
                "reward_return_changed",
                "n_step",
                "gamma",
            )
        ),
        "parent": {key: parent[key] for key in common},
        "candidate": {key: candidate[key] for key in common},
        "candidate_minus_parent": {
            key: float(candidate[key]) - float(parent[key]) for key in common
        },
    }


def lane_prior_choice(
    row: Mapping[str, Any], coefficient: float
) -> dict[str, Any]:
    decoder = row["target_critic_decoder"]
    stored_coefficient = float(decoder["lane_prior_coef"])
    _require(stored_coefficient > 0.0, "stored lane-prior coefficient is not positive")
    _require(coefficient >= 0.0, "counterfactual coefficient must be non-negative")
    probabilities = [float(value) for value in decoder["lane_probabilities"]]
    proposals = shared._flatten_decoder(decoder)
    original_lane = int(decoder["selected_lane_index"])
    original_component = int(decoder["selected_component_index"])
    stored_argmax = max(proposals, key=lambda item: float(item["score"]))
    _require(
        int(stored_argmax["lane_index"]) == original_lane
        and int(stored_argmax["component_index"]) == original_component,
        "stored proposal is not score argmax",
    )
    rescored: list[tuple[float, dict[str, float | int]]] = []
    for proposal in proposals:
        lane_index = int(proposal["lane_index"])
        log_lane = math.log(max(probabilities[lane_index], 1e-45))
        score = (
            float(proposal["score"])
            - stored_coefficient * log_lane
            + float(coefficient) * log_lane
        )
        rescored.append((score, proposal))
    score, selected = max(rescored, key=lambda item: item[0])
    lane_index = int(selected["lane_index"])
    component_index = int(selected["component_index"])
    return {
        "lane_index": lane_index,
        "lane_code": LANE_CODES[lane_index],
        "component_index": component_index,
        "counterfactual_lane_prior_coefficient": float(coefficient),
        "counterfactual_score": score,
        "collision": float(selected["collision"]),
        "q": float(selected["q"]),
        "uncertainty": float(selected["uncertainty"]),
        "speed": float(selected["speed"]),
        "lane_probability": probabilities[lane_index],
        "changed_lane": lane_index != original_lane,
        "changed_proposal": (
            lane_index != original_lane or component_index != original_component
        ),
    }


def remove_lane_prior_choice(row: Mapping[str, Any]) -> dict[str, Any]:
    choice = lane_prior_choice(row, 0.0)
    choice["score_without_lane_prior"] = choice["counterfactual_score"]
    return choice


def _switch_rate(values: Sequence[tuple[int, int, int]]) -> float | None:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for episode, decision, lane in values:
        grouped[episode].append((decision, lane))
    switches = comparisons = 0
    for rows in grouped.values():
        lanes = [lane for _, lane in sorted(rows)]
        for previous, current in zip(lanes, lanes[1:]):
            comparisons += 1
            switches += int(previous != current)
    return switches / comparisons if comparisons else None


def lane_prior_removal_sensitivity(
    raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = [shared.decision_metrics(row) for row in raw_rows]
    episodes = shared.group_episodes(metrics)
    outcome_by_episode = {
        episode: str(rows[0]["eventual_outcome"])
        for episode, rows in episodes.items()
    }
    choices = [remove_lane_prior_choice(row) for row in raw_rows]
    _require(len(choices) > 0, "lane-prior sensitivity requires trace rows")
    original_lanes: list[tuple[int, int, int]] = []
    new_lanes: list[tuple[int, int, int]] = []
    by_outcome: dict[str, list[int]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    actor_matches = route_matches = route_samples = 0
    for row, choice in zip(raw_rows, choices):
        decoder = row["target_critic_decoder"]
        episode = int(row["episode"])
        decision = int(row["decision"])
        original = int(decoder["selected_lane"])
        new = int(choice["lane_code"])
        original_lanes.append((episode, decision, original))
        new_lanes.append((episode, decision, new))
        by_outcome[outcome_by_episode[episode]].append(int(choice["changed_lane"]))
        transitions[f"{original}__to__{new}"] += 1
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
        actor_index = max(valid_indices, key=lambda index: probabilities[index])
        actor_matches += int(actor_index == int(choice["lane_index"]))
        if bool(row.get("pre_action_route_intent_valid")):
            route_samples += 1
            route_matches += int(new == int(row["pre_action_route_intent"]))
    terminal: dict[str, Any] = {}
    choice_by_key = {
        (int(row["episode"]), int(row["decision"])): choice
        for row, choice in zip(raw_rows, choices)
    }
    for outcome in ("success", "collision"):
        terminal[outcome] = {}
        for window in (1, 5, 10, 20):
            rows = shared.terminal_rows(episodes, outcome=outcome, window=window)
            changes = [
                int(
                    choice_by_key[(int(row["episode"]), int(row["decision"]))][
                        "changed_lane"
                    ]
                )
                for row in rows
            ]
            terminal[outcome][str(window)] = {
                "decision_count": len(changes),
                "changed_lane_rate": _mean(changes),
            }
    count = len(choices)
    return {
        "counterfactual": "remove_continuous_learned_lane_log_prior_only",
        "offline_only_no_environment_interaction": True,
        "action_rewritten": False,
        "closed_loop_outcome_claim_allowed": False,
        "decision_count": count,
        "stored_lane_prior_coefficient": float(
            raw_rows[0]["target_critic_decoder"]["lane_prior_coef"]
        ),
        "changed_proposal_rate": sum(
            int(choice["changed_proposal"]) for choice in choices
        )
        / count,
        "changed_lane_rate": sum(int(choice["changed_lane"]) for choice in choices)
        / count,
        "changed_lane_rate_by_eventual_outcome": {
            outcome: _mean(changes) for outcome, changes in sorted(by_outcome.items())
        },
        "terminal_window_changed_lane_rate": terminal,
        "lane_transition_counts": dict(sorted(transitions.items())),
        "original_lane_switch_rate": _switch_rate(original_lanes),
        "without_prior_lane_switch_rate": _switch_rate(new_lanes),
        "without_prior_non_keep_rate": sum(
            int(choice["lane_code"] != 0) for choice in choices
        )
        / count,
        "without_prior_actor_mode_match_rate": actor_matches / count,
        "without_prior_route_intent_match_rate": (
            route_matches / route_samples if route_samples else None
        ),
        "without_prior_selected_collision_value": shared.numeric_summary(
            float(choice["collision"]) for choice in choices
        ),
        "without_prior_selected_min_reward_q": shared.numeric_summary(
            float(choice["q"]) for choice in choices
        ),
        "without_prior_selected_uncertainty": shared.numeric_summary(
            float(choice["uncertainty"]) for choice in choices
        ),
        "without_prior_selected_speed_normalized": shared.numeric_summary(
            float(choice["speed"]) for choice in choices
        ),
    }


def lane_prior_score_sweep(
    raw_rows: Sequence[Mapping[str, Any]],
    coefficients: Sequence[float] = LANE_PRIOR_SWEEP,
) -> dict[str, Any]:
    metrics = [shared.decision_metrics(row) for row in raw_rows]
    episodes = shared.group_episodes(metrics)
    outcome_by_episode = {
        episode: str(rows[0]["eventual_outcome"])
        for episode, rows in episodes.items()
    }
    output: dict[str, Any] = {}
    for coefficient in coefficients:
        choices = [lane_prior_choice(row, float(coefficient)) for row in raw_rows]
        count = len(choices)
        by_outcome: dict[str, list[int]] = defaultdict(list)
        lanes: list[tuple[int, int, int]] = []
        route_matches = route_samples = actor_matches = 0
        for row, choice in zip(raw_rows, choices):
            episode = int(row["episode"])
            lanes.append(
                (episode, int(row["decision"]), int(choice["lane_code"]))
            )
            by_outcome[outcome_by_episode[episode]].append(
                int(choice["changed_lane"])
            )
            decoder = row["target_critic_decoder"]
            probabilities = [
                float(value) for value in decoder["lane_probabilities"]
            ]
            valid = [bool(value) for value in decoder["valid_lane_actions"]]
            valid_indices = [
                index for index, is_valid in enumerate(valid) if is_valid
            ]
            actor_index = max(
                valid_indices, key=lambda index: probabilities[index]
            )
            actor_matches += int(actor_index == int(choice["lane_index"]))
            if bool(row.get("pre_action_route_intent_valid")):
                route_samples += 1
                route_matches += int(
                    int(choice["lane_code"])
                    == int(row["pre_action_route_intent"])
                )
        output[str(float(coefficient))] = {
            "coefficient": float(coefficient),
            "offline_only_no_environment_interaction": True,
            "closed_loop_outcome_claim_allowed": False,
            "changed_lane_rate_vs_stored_0_05": sum(
                int(choice["changed_lane"]) for choice in choices
            )
            / count,
            "changed_lane_rate_by_stored_eventual_outcome": {
                outcome: _mean(changes)
                for outcome, changes in sorted(by_outcome.items())
            },
            "lane_switch_rate": _switch_rate(lanes),
            "non_keep_rate": sum(
                int(choice["lane_code"] != 0) for choice in choices
            )
            / count,
            "actor_mode_match_rate": actor_matches / count,
            "route_intent_match_rate": (
                route_matches / route_samples if route_samples else None
            ),
            "selected_collision_value": shared.numeric_summary(
                float(choice["collision"]) for choice in choices
            ),
            "selected_min_reward_q": shared.numeric_summary(
                float(choice["q"]) for choice in choices
            ),
            "selected_speed_normalized": shared.numeric_summary(
                float(choice["speed"]) for choice in choices
            ),
        }
    return output


def _source_paths(root: Path) -> dict[str, Path]:
    paths = {
        "paper": root / "paper_evaluation_detailed.json",
        "actions": root / "action_diagnostics.json",
        "trace": root / "action_diagnostics_decisions.jsonl",
        "training": root / "training_diagnostics.json",
        "labels": root / "collision_label_diagnostics.json",
        "arguments": root / "arguments.json",
        "method": root / "method_metadata.json",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing {name}: {path}")
    return paths


def _verify_no_forbidden_mechanism(
    paper: Mapping[str, Any],
    actions: Mapping[str, Any],
    training: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for key in FORBIDDEN_FIELDS:
        observed = [
            source.get(key) for source in (paper, actions, training, method)
        ]
        values = [value for value in observed if value is not None]
        _require(values and all(value is False for value in values), key)
        result[key] = False
    return result


def build_attribution() -> dict[str, Any]:
    decision = _load_json(DEVELOPMENT_DECISION)
    _require(decision["decision"] == "fail", "v4.12 development did not fail")
    _require(decision["formal_test_accessed"] is False, "formal test was accessed")
    failed_runs = [
        key for key, value in decision["per_run_gates"].items() if not value["passed"]
    ]
    _require(failed_runs == ["D3__ajs__ram__s12__p3f978ef9"], "failure is not isolated to D3")

    parent_paths = _source_paths(PARENT_RUN)
    candidate_paths = _source_paths(CANDIDATE_RUN)
    parent = {name: _load_json(path) for name, path in parent_paths.items() if name != "trace"}
    candidate = {
        name: _load_json(path)
        for name, path in candidate_paths.items()
        if name != "trace"
    }
    rows = _load_jsonl(candidate_paths["trace"])
    _require(
        _sha256(candidate_paths["trace"]) == candidate["actions"]["trace"]["sha256"],
        "candidate trace hash drifted",
    )
    _require(
        len(rows) == int(candidate["actions"]["decision_records"]),
        "candidate trace record count drifted",
    )
    forbidden = {
        "parent": _verify_no_forbidden_mechanism(
            parent["paper"],
            parent["actions"],
            parent["training"],
            parent["method"],
        ),
        "candidate": _verify_no_forbidden_mechanism(
            candidate["paper"],
            candidate["actions"],
            candidate["training"],
            candidate["method"],
        ),
    }
    parent_behavior = _behavior(parent["actions"])
    candidate_behavior = _behavior(candidate["actions"])
    parent_args = parent["arguments"]["requested_raw_steps"]
    candidate_args = candidate["arguments"]["requested_raw_steps"]
    _require(parent_args["scenario"] == "roundabout_medium", "parent scenario drift")
    _require(candidate_args["scenario"] == "roundabout_medium", "candidate scenario drift")
    _require(int(parent_args["seed"]) == int(candidate_args["seed"]) == 12, "train seed drift")
    return {
        "schema_version": "topo-scene-v4.12.development-failure-attribution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "immutable_run_observation_and_offline_score_counterfactual",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "development_gate_decision": "fail",
        "failed_runs": failed_runs,
        "formal_test_accessed": False,
        "formal_test_unlocked": False,
        "forbidden_mechanisms_used": forbidden,
        "parent_summary": parent["paper"]["summary"],
        "candidate_summary": candidate["paper"]["summary"],
        "candidate_minus_parent_summary": _numeric_delta(
            candidate["paper"]["summary"], parent["paper"]["summary"]
        ),
        "episode_alignment": episode_alignment(parent["paper"], candidate["paper"]),
        "behavior": {
            "parent": parent_behavior,
            "candidate": candidate_behavior,
            "candidate_minus_parent": _numeric_delta(
                candidate_behavior, parent_behavior
            ),
            "comparison_is_aggregate_not_exact_seed_paired": True,
        },
        "training_diagnostics": _training_comparison(
            parent["training"], candidate["training"]
        ),
        "collision_label_diagnostics": _label_comparison(
            parent["labels"], candidate["labels"]
        ),
        "augmentation_scope": {
            "roundabout_medium_parent_config": "scenario != cross -> enabled",
            "roundabout_medium_candidate_runtime_recorded": candidate["training"][
                "effective_random_rotation_augmentation"
            ],
            "cross_only_change_cannot_explain_roundabout_regression": True,
            "parent_config_path": _relative(ROOT / "configs" / "sb3_configs_v4_11.py"),
            "parent_config_sha256": _sha256(ROOT / "configs" / "sb3_configs_v4_11.py"),
            "candidate_config_path": _relative(ROOT / "configs" / "sb3_configs_v4_12.py"),
            "candidate_config_sha256": _sha256(ROOT / "configs" / "sb3_configs_v4_12.py"),
        },
        "lane_prior_removal_offline_sensitivity": lane_prior_removal_sensitivity(rows),
        "lane_prior_coefficient_offline_sweep": lane_prior_score_sweep(rows),
        "source_artifacts": {
            "development_decision": {
                "path": _relative(DEVELOPMENT_DECISION),
                "sha256": _sha256(DEVELOPMENT_DECISION),
            },
            "parent": {
                name: {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in parent_paths.items()
            },
            "candidate": {
                name: {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in candidate_paths.items()
            },
        },
        "next_required_evidence": {
            "kind": "closed_loop_frozen_checkpoint_lane_prior_ablation",
            "scenario": "roundabout_medium",
            "evaluation_seed_start": int(candidate_args["evaluation_seed_start"]),
            "episodes": int(candidate_args["eval_episodes"]),
            "lane_prior_coef": 0.0,
            "rule_or_projection_allowed": False,
        },
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    delta = payload["candidate_minus_parent_summary"]
    behavior = payload["behavior"]["candidate_minus_parent"]
    sensitivity = payload["lane_prior_removal_offline_sensitivity"]
    alignment = payload["episode_alignment"]
    return "\n".join(
        [
            "# v4.12 D3 roundabout-medium failure attribution",
            "",
            "All values below are computed from immutable real-run artifacts. No formal-test data was accessed.",
            "",
            "## Gate-localized outcome",
            "",
            f"- Success delta (candidate - parent): `{delta['success_rate']:+.6f}`",
            f"- Collision delta (candidate - parent): `{delta['collision_rate']:+.6f}`",
            f"- Exact evaluation-seed pairing: `{alignment['exact_seed_paired']}`; the cross-version episode transition table is descriptive, not causal.",
            "",
            "## Behavioral shift",
            "",
            f"- Non-keep command-rate delta: `{behavior['non_keep_command_rate']:+.6f}`",
            f"- Lane-change applied-rate delta: `{behavior['lane_change_applied_rate']:+.6f}`",
            f"- Valid route-intent match-rate delta: `{behavior['valid_route_intent_match_rate']:+.6f}`",
            f"- Mean actual-speed delta (m/s): `{behavior['mean_actual_speed_mps']:+.6f}`",
            "",
            "## Offline learned-score counterfactual",
            "",
            "This diagnostic subtracts the continuous learned lane-log-prior term from every feasible candidate score; it does not rewrite recorded actions and cannot establish a closed-loop outcome.",
            "",
            f"- Changed proposal rate: `{sensitivity['changed_proposal_rate']:.6f}`",
            f"- Changed lane rate: `{sensitivity['changed_lane_rate']:.6f}`",
            f"- Original lane-switch rate: `{sensitivity['original_lane_switch_rate']:.6f}`",
            f"- Without-prior lane-switch rate: `{sensitivity['without_prior_lane_switch_rate']:.6f}`",
            "",
            "## Mechanism boundary",
            "",
            "No kinematic projection, traffic-risk lane mask, threshold, veto, confidence gate, shield, or post-decoder rewrite was present. The next diagnostic is a frozen-model continuous-parameter ablation (`lane_prior_coef=0`) on the same v4.12 validation seeds.",
            "",
        ]
    )


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
    markdown = output.with_suffix(".md")
    markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": str(markdown),
                "failed_runs": payload["failed_runs"],
                "exact_seed_paired": payload["episode_alignment"][
                    "exact_seed_paired"
                ],
                "lane_prior_removal": payload[
                    "lane_prior_removal_offline_sensitivity"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_attribution",
    "episode_alignment",
    "lane_prior_choice",
    "lane_prior_score_sweep",
    "lane_prior_removal_sensitivity",
    "remove_lane_prior_choice",
]
