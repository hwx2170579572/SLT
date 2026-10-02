from __future__ import annotations

from tools import audit_v4_13_no_kinematic_projection as subject


def test_source_boundary_has_no_rule_or_projection_call() -> None:
    value = subject.source_boundary_audit()
    assert value["inherited_boundary_integrity"] is True
    assert value["same_deployed_lane_head_reused_for_support"] is True
    assert value["lane_support_input_is_stop_gradient_latent"] is True
    assert value["v4_13_does_not_override_inference_score"] is True
    assert value["v4_13_predict_delegates_action_to_frozen_model_argmax"] is True
    assert value["new_scientific_modules_have_no_projection_or_rule_calls"] is True
    assert all(not calls for calls in value["suspicious_runtime_calls_by_file"].values())


def test_execution_audit_proves_gradient_and_exact_action_boundary() -> None:
    value = subject.model_execution_audit()
    assert value["lane_nll_updates_same_deployed_lane_head"] is True
    assert value["lane_nll_zero_gradient_actor_latent_trunk"] is True
    assert value["lane_nll_zero_gradient_speed_heads"] is True
    assert value["lane_nll_zero_gradient_scene_encoder"] is True
    assert value["speed_nll_updates_latent_and_speed_heads"] is True
    assert value["joint_nll_equals_tempered_sum"] is True
    assert value["selected_action_equals_exact_actor_proposal"] is True
    assert value["semantic_tie_override_absent"] is True
    assert value["actor_confidence_threshold_absent"] is True
    assert value["post_decoder_action_rewrite_absent"] is True


def test_static_audit_passes_without_claiming_runtime() -> None:
    value = subject.build_audit(None, require_runtime=False)
    assert value["integrity_passed"] is True
    assert value[
        "candidate_side_kinematic_or_traffic_risk_projection_present"
    ] is False
    assert value["inference_rule_mechanism_present"] is False
    assert value["physical_lane_availability_mask_present"] is True
    assert value["physical_lane_availability_mask_is_traffic_risk_rule"] is False
