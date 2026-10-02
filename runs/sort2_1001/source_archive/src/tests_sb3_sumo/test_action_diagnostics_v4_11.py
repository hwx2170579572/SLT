from __future__ import annotations

import copy
import math

from tools import action_diagnostics_v4_11_model as diagnostics


def _decoder_row() -> dict:
    probabilities = [
        [0.2, 0.3, 0.5],
        [0.2, 0.5, 0.3],
        [0.4, 0.4, 0.2],
    ]
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
            + 0.05 * math.log(probabilities[lane][component])
            for component in range(3)
        ]
        for lane in range(3)
    ]
    return {
        "valid_lane_actions": [True, True, True],
        "component_probabilities": probabilities,
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
        "selected_lane_index": 1,
        "selected_lane": 0,
        "selected_component_index": 1,
        "selection_operator": "torch_argmax_flattened_feasible_model_scores",
        "semantic_tie_override_used": False,
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


def test_integrity_recomputes_full_learned_score_and_action() -> None:
    value = diagnostics._proposal_integrity([_trace_row()])
    assert value["selected_decoder_exact_model_match_rate"] == 1.0
    assert value["no_semantic_tie_override_rate"] == 1.0
    assert value["selected_decoder_exact_action_match_rate"] == 1.0
    assert value["selected_speed_exact_proposal_match_rate"] == 1.0
    assert value["supported_mixture_score_equation_match_rate"] == 1.0
    assert value["collision_value_bounded_rate"] == 1.0
    assert value["learned_proposal_source_rate"] == 1.0
    assert value["no_action_rewrite_rate"] == 1.0
    assert value["learned_speed_components_per_lane"] == 3


def test_integrity_detects_score_or_action_tampering() -> None:
    row = _trace_row()
    row["target_critic_decoder"]["supported_risk_adjusted_score"][0][0] += 1.0
    value = diagnostics._proposal_integrity([row])
    assert value["supported_mixture_score_equation_match_rate"] == 0.0
    assert value["selected_decoder_exact_model_match_rate"] == 0.0

    action_row = _trace_row()
    action_row["action_longitudinal"] += 0.1
    action_value = diagnostics._proposal_integrity([action_row])
    assert action_value["selected_speed_exact_proposal_match_rate"] == 0.0


def test_integrity_detects_forbidden_rewrite_metadata() -> None:
    row = copy.deepcopy(_trace_row())
    row["target_critic_decoder"]["action_rewritten"] = True
    value = diagnostics._proposal_integrity([row])
    assert value["no_action_rewrite_rate"] == 0.0


def test_exact_score_tie_uses_plain_flattened_argmax_without_keep_override() -> None:
    row = _trace_row()
    decoder = row["target_critic_decoder"]
    best = decoder["supported_risk_adjusted_score"][1][1]
    decoder["supported_risk_adjusted_score"][0][0] = best
    decoder["minimum_target_twin_q"][0][0] = (
        best
        + decoder["maximum_target_twin_collision_value"][0][0]
        + 0.25 * decoder["learned_uncertainty"][0][0]
        - 0.05 * math.log(decoder["component_probabilities"][0][0])
    )
    decoder["selected_lane_index"] = 0
    decoder["selected_lane"] = -1
    decoder["selected_component_index"] = 0
    row["lane_command"] = -1
    row["action_longitudinal"] = decoder["learned_speed_proposals_normalized"][0][0]
    value = diagnostics._proposal_integrity([row])
    assert value["exact_supported_mixture_model_argmax_rate"] == 1.0
    assert value["no_semantic_tie_override_rate"] == 1.0



def test_model_integrity_requires_model_only_risk_metadata() -> None:
    value = {
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "shared_risk_encoder_present": True,
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
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_hard_threshold": False,
        "collision_pairwise_fixed_margin": False,
        "action_postprocessing_override": False,
    }
    assert diagnostics.learned_supported_model_integrity_passed(value) is True
    value["kinematic_safety_projection"] = True
    assert diagnostics.learned_supported_model_integrity_passed(value) is False
