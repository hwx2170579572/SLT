from __future__ import annotations

import copy
import math

from tools import action_diagnostics_v4_12_model as diagnostics


def _decoder_row() -> dict:
    component_probabilities = [
        [0.2, 0.3, 0.5],
        [0.2, 0.5, 0.3],
        [0.4, 0.4, 0.2],
    ]
    lane_probabilities = [0.2, 0.6, 0.2]
    reward = [
        [0.1, 0.2, 0.3],
        [0.4, 0.8, 0.5],
        [0.2, 0.3, 0.4],
    ]
    collision = [[0.1, 0.1, 0.1] for _ in range(3)]
    reward_gap = [[0.02, 0.02, 0.02] for _ in range(3)]
    collision_gap = [[0.01, 0.01, 0.01] for _ in range(3)]
    uncertainty = [[0.03, 0.03, 0.03] for _ in range(3)]
    score = [
        [
            reward[lane][component]
            - collision[lane][component]
            - 0.25 * uncertainty[lane][component]
            + 0.05 * math.log(component_probabilities[lane][component])
            + 0.05 * math.log(lane_probabilities[lane])
            for component in range(3)
        ]
        for lane in range(3)
    ]
    return {
        "valid_lane_actions": [True, True, True],
        "lane_probabilities": lane_probabilities,
        "component_probabilities": component_probabilities,
        "learned_speed_proposals_normalized": [
            [-0.8, -0.2, 0.4],
            [-0.7, 0.1, 0.7],
            [-0.6, 0.2, 0.8],
        ],
        "minimum_target_twin_q": reward,
        "maximum_target_twin_collision_value": collision,
        "reward_twin_disagreement": reward_gap,
        "collision_twin_disagreement": collision_gap,
        "learned_uncertainty": uncertainty,
        "supported_risk_adjusted_score": score,
        "collision_risk_coef": 1.0,
        "twin_uncertainty_coef": 0.25,
        "component_prior_coef": 0.05,
        "lane_prior_coef": 0.05,
        "selected_lane_index": 1,
        "selected_lane": 0,
        "selected_component_index": 1,
        "selection_operator": "torch_argmax_flattened_feasible_model_scores",
        "semantic_tie_override_used": False,
        "actor_confidence_threshold_present": False,
        "candidate_source": "learned_actor_component_means",
        "fixed_speed_grid_used": False,
        "action_rewritten": False,
    }


def _trace_row() -> dict:
    decoder = _decoder_row()
    return {
        "lane_command": 0,
        "action_longitudinal": decoder["learned_speed_proposals_normalized"][1][1],
        "target_critic_decoder": decoder,
    }


def test_integrity_recomputes_joint_lane_component_score_and_action() -> None:
    value = diagnostics._proposal_integrity([_trace_row()])
    assert value["selected_decoder_exact_model_match_rate"] == 1.0
    assert value["exact_joint_support_model_argmax_rate"] == 1.0
    assert value["joint_support_score_equation_match_rate"] == 1.0
    assert value["selected_decoder_exact_action_match_rate"] == 1.0
    assert value["selected_speed_exact_proposal_match_rate"] == 1.0
    assert value["no_semantic_tie_override_rate"] == 1.0
    assert value["no_actor_confidence_threshold_rate"] == 1.0
    assert value["no_action_rewrite_rate"] == 1.0


def test_integrity_detects_score_action_and_lane_probability_tampering() -> None:
    score_row = copy.deepcopy(_trace_row())
    score_row["target_critic_decoder"]["supported_risk_adjusted_score"][0][0] += 1.0
    value = diagnostics._proposal_integrity([score_row])
    assert value["joint_support_score_equation_match_rate"] == 0.0
    assert value["selected_decoder_exact_model_match_rate"] == 0.0

    action_row = copy.deepcopy(_trace_row())
    action_row["action_longitudinal"] += 0.1
    assert diagnostics._proposal_integrity([action_row])[
        "selected_speed_exact_proposal_match_rate"
    ] == 0.0

    probability_row = copy.deepcopy(_trace_row())
    probability_row["target_critic_decoder"]["lane_probabilities"][0] += 0.1
    assert diagnostics._proposal_integrity([probability_row])[
        "joint_support_score_equation_match_rate"
    ] == 0.0


def test_integrity_rejects_forbidden_rewrite_or_threshold_metadata() -> None:
    rewrite = copy.deepcopy(_trace_row())
    rewrite["target_critic_decoder"]["action_rewritten"] = True
    assert diagnostics._proposal_integrity([rewrite])[
        "no_action_rewrite_rate"
    ] == 0.0

    threshold = copy.deepcopy(_trace_row())
    threshold["target_critic_decoder"]["actor_confidence_threshold_present"] = True
    assert diagnostics._proposal_integrity([threshold])[
        "no_actor_confidence_threshold_rate"
    ] == 0.0


def test_joint_model_integrity_requires_model_only_boundary() -> None:
    value = {
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "shared_risk_encoder_present": True,
        "joint_replay_action_support_present": True,
        "optimizer_ownership": {
            "overlap_count": 0,
            "policy_structure": {
                "online_encoder_shared_by_identity": True,
                "target_encoder_shared_by_identity": True,
            },
        },
        "fixed_speed_grid_at_inference": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
    }
    assert diagnostics.joint_support_model_integrity_passed(value) is True
    value["kinematic_safety_projection"] = True
    assert diagnostics.joint_support_model_integrity_passed(value) is False
