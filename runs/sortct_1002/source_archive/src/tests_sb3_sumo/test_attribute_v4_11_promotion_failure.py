from __future__ import annotations

from copy import deepcopy

import pytest

from tools import attribute_v4_11_promotion_failure as attribution


def _decoder_row() -> dict:
    decoder = {
        "valid_lane_actions": [True, True, False],
        "component_probabilities": [[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]],
        "learned_speed_proposals_normalized": [[-0.2, 0.2], [-0.1, 0.1], [0, 0]],
        "minimum_target_twin_q": [[0.5, 0.4], [0.51, 0.4], None],
        "maximum_target_twin_collision_value": [[0.1, 0.1], [0.1, 0.1], None],
        "reward_twin_disagreement": [[0.0, 0.0], [0.0, 0.0], None],
        "collision_twin_disagreement": [[0.0, 0.0], [0.0, 0.0], None],
        "learned_uncertainty": [[0.0, 0.0], [0.0, 0.0], None],
        "supported_risk_adjusted_score": [[0.39, 0.2], [0.40, 0.2], None],
        "lane_probabilities": [0.9, 0.1, 0.0],
        "selected_lane_index": 1,
        "selected_lane": 0,
        "selected_component_index": 0,
    }
    return {
        "episode": 0,
        "decision": 0,
        "pre_action_route_intent_valid": True,
        "pre_action_route_intent": -1,
        "target_critic_decoder": decoder,
    }


def test_lane_log_prior_is_continuous_model_rescoring() -> None:
    row = _decoder_row()
    weak = attribution._lane_prior_choice(row, 0.001)
    strong = attribution._lane_prior_choice(row, 0.05)
    assert weak["lane_code"] == 0
    assert strong["lane_code"] == -1
    assert strong["lane_probability"] == pytest.approx(0.9)


def test_lane_prior_sensitivity_never_claims_closed_loop_outcome() -> None:
    row = _decoder_row()
    row.update({"collision": False, "off_route": False, "is_success": True})
    value = attribution.lane_prior_sensitivity([row], 0.05)
    assert value["offline_only_no_action_rewrite"] is True
    assert value["closed_loop_outcome_claim_allowed"] is False
    assert value["changed_lane_rate"] == 1.0
    assert value["counterfactual_route_intent_match_rate"] == 1.0


def test_lane_policy_dynamics_separates_actor_and_selector_switches() -> None:
    first = _decoder_row()
    first["target_critic_decoder"]["selected_lane_index"] = 0
    first["target_critic_decoder"]["selected_lane"] = -1
    second = deepcopy(first)
    second["decision"] = 1
    second["target_critic_decoder"]["selected_lane_index"] = 1
    second["target_critic_decoder"]["selected_lane"] = 0
    # Actor mode stays on lane -1 while the selector changes lanes.
    value = attribution.lane_policy_dynamics([first, second])
    assert value["offline_observation_only"] is True
    assert value["action_rewritten"] is False
    assert value["actor_mode_switch_rate"] == 0.0
    assert value["selected_lane_switch_rate"] == 1.0
    assert value["selected_actor_mode_match_rate"] == 0.5
    assert value["multi_feasible"]["decision_count"] == 2


def test_collision_risk_rescoring_uses_learned_value_continuously() -> None:
    row = _decoder_row()
    decoder = row["target_critic_decoder"]
    decoder["collision_risk_coef"] = 1.0
    decoder["maximum_target_twin_collision_value"][0][0] = 0.01
    decoder["maximum_target_twin_collision_value"][1][0] = 0.30
    # Original scores choose lane 0; a larger learned-risk coefficient moves
    # the continuous model argmax to lane -1 without a threshold or veto.
    choice = attribution._collision_risk_choice(row, 2.0)
    assert choice["lane_code"] == -1
    row.update({"collision": False, "off_route": False, "is_success": True})
    value = attribution.collision_risk_sensitivity([row], 2.0)
    assert value["offline_only_no_action_rewrite"] is True
    assert value["closed_loop_outcome_claim_allowed"] is False


def test_proposal_value_separation_reports_within_state_spreads() -> None:
    row = _decoder_row()
    value = attribution.proposal_value_separation([row])
    assert value["learned_values_only"] is True
    assert value["feasible_collision_value_spread"]["mean"] == pytest.approx(0.0)
    assert value["feasible_reward_q_spread"]["mean"] == pytest.approx(0.11)
    assert value["lane_minimum_collision_value_spread"]["mean"] == pytest.approx(0.0)


def test_paired_evidence_requires_exact_seed_pairing() -> None:
    behavior = {
        "lane_change_applied_rate": 0.1,
        "valid_route_intent_match_rate": 0.8,
        "route_action_window_match_rate": None,
        "non_keep_command_rate": 0.2,
        "mean_action_longitudinal": 0.0,
        "mean_actual_speed_mps": 4.0,
        "p90_actual_speed_mps": 6.0,
    }
    control = {
        "behavior": behavior,
        "episode_index": {
            10: {"traffic_variant": "a", "outcome": "success"}
        },
    }
    candidate = deepcopy(control)
    candidate["episode_index"][10]["outcome"] = "collision"
    value = attribution.paired_evidence(control, candidate)
    assert value["control_success_lost_count"] == 1
    assert value["candidate_success_gained_count"] == 0
    candidate["episode_index"][11] = candidate["episode_index"].pop(10)
    with pytest.raises(ValueError, match="seeds differ"):
        attribution.paired_evidence(control, candidate)


def test_behavior_delta_preserves_missing_route_window() -> None:
    left = {
        "lane_change_applied_rate": 0.4,
        "valid_route_intent_match_rate": 0.6,
        "route_action_window_match_rate": None,
        "non_keep_command_rate": 0.5,
        "mean_action_longitudinal": 0.1,
        "mean_actual_speed_mps": 5.0,
        "p90_actual_speed_mps": 7.0,
    }
    right = dict(left)
    right["lane_change_applied_rate"] = 0.1
    value = attribution._behavior_delta(left, right)
    assert value["lane_change_applied_rate"] == pytest.approx(0.3)
    assert value["route_action_window_match_rate"] is None
