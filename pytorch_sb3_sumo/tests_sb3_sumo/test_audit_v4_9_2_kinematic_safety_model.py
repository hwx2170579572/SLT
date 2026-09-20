from __future__ import annotations

from tools.audit_v4_9_2_kinematic_safety_model import (
    build_audit,
    parameter_and_gradient_audit,
    source_boundary_audit,
)


def test_source_boundary_has_no_candidate_side_safety_projection() -> None:
    audit = source_boundary_audit()
    assert audit["lane_action_mask_has_no_traffic_risk_query"] is True
    assert audit["lane_action_mask_checks_adjacent_driving_lane"] is True
    assert audit["apply_control_has_no_traffic_risk_query"] is True
    assert audit["sumo_safe_speed_and_lane_change_checks_disabled"] is True
    assert audit["target_decoder_has_no_safety_rule"] is True
    assert audit["target_decoder_is_learned_score_argmax"] is True
    assert audit["ttc_is_vehicle_graph_feature"] is True
    assert audit["ttc_source_does_not_select_or_rewrite_action"] is True
    assert audit["collision_label_comes_from_observed_info"] is True


def test_gradient_ownership_exposes_the_model_bottleneck() -> None:
    audit = parameter_and_gradient_audit()
    assert audit["actor_and_reward_critic_share_exact_online_extractor"] is True
    assert audit["collision_online_extractor_is_independent"] is True
    assert audit["actor_optimizer_excludes_shared_extractor"] is True
    assert audit["reward_critic_optimizer_owns_shared_extractor"] is True
    assert audit["collision_optimizer_owns_independent_extractor"] is True
    assert audit["actor_head_gradient_from_actor_outputs"] == {
        "speed_head": True,
        "lane_head": True,
    }
    assert audit["actor_head_gradient_from_learned_collision_objective"] == {
        "speed_head": True,
        "lane_head": True,
    }
    assert audit["shared_extractor_gradient_from_actor_outputs"] is False
    assert (
        audit["shared_extractor_gradient_from_actor_collision_objective"]
        is False
    )


def test_full_audit_keeps_partial_evidence_non_decisional() -> None:
    audit = build_audit()
    assert audit["integrity_passed"] is True
    assert audit["candidate_side_kinematic_projection_present"] is False
    assert audit["inference_safety_rule_added"] is False
    assert audit["decision_allowed_from_partial_matrix"] is False
    partial = audit["partial_promotion_evidence"]
    assert partial["pair_count"] == 1
    assert partial["expected_pair_count"] == 6
    assert partial["decision_allowed"] is False
    assert partial["cross_collision_gate_lower_bound"]["gate_can_still_pass"] is False
