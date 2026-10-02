from __future__ import annotations

import json

from tools.audit_v4_11_no_kinematic_projection import (
    build_audit,
    model_execution_audit,
    runtime_run_audit,
    source_boundary_audit,
)


def test_source_call_chain_has_no_rule_or_kinematic_projection() -> None:
    audit = source_boundary_audit()
    assert audit["transitive_inherited_policy_sources_hashed"] is True
    assert audit["resolved_decoder_owner_is_frozen_v4_10_model"] is True
    assert audit["resolved_score_owner_is_frozen_v4_10_model"] is True
    assert audit["learned_proposals_are_actor_head_outputs"] is True
    assert audit["decoder_uses_learned_reward_collision_uncertainty_score"] is True
    assert audit["decoder_masks_only_physical_lane_feasibility"] is True
    assert audit["decoder_is_plain_flattened_argmax"] is True
    assert audit["decoder_returns_exact_selected_actor_proposal"] is True
    assert audit["decoder_has_no_keep_tie_semantic_override"] is True
    assert audit["decoder_has_no_rule_projection_call"] is True
    assert audit["decoder_has_no_fixed_speed_grid"] is True
    assert audit["collision_loss_is_proper_soft_bernoulli_nll"] is True
    assert audit["collision_ranking_is_continuous_without_fixed_margin"] is True
    assert audit["ema_risk_teacher_is_detached"] is True
    assert audit["paper_environment_passes_policy_action_unchanged"] is True
    assert audit["direct_speed_profile_applies_requested_speed_exactly"] is True
    assert audit["lane_mask_has_no_traffic_risk_query"] is True
    assert audit["lane_application_has_no_traffic_risk_query"] is True
    assert audit["sumo_safe_speed_and_lane_veto_disabled"] is True


def test_dynamic_model_path_is_learned_and_has_disjoint_ownership() -> None:
    audit = model_execution_audit()
    assert audit["learned_component_tensor_shape"] == [1, 3, 3, 2]
    assert audit["actual_score_argmax_matches_record"] is True
    assert audit["selected_action_equals_exact_actor_proposal"] is True
    assert audit["environment_domain_clip_is_identity_for_model_action"] is True
    assert audit["environment_lane_parameterization_preserves_exact_model_code"] is True
    assert audit["environment_speed_parameterization_matches_linear_contract"] is True
    assert audit["direct_control_execution_preserves_requested_speed_exactly"] is True
    assert audit["direct_control_execution_never_queries_dynamics_proxy"] is True
    assert audit["sumo_internal_safe_modes_disabled_by_execution"] is True
    assert audit["exact_tie_uses_first_flattened_feasible_argmax"] is True
    assert audit["semantic_tie_override_used"] is False
    assert audit["reward_objective_reaches_shared_encoder"] is True
    assert audit["collision_objective_reaches_shared_encoder"] is True
    assert audit["actor_heads_receive_actor_gradient"] == {
        "speed_mean": True,
        "component_logits": True,
        "lane_logits": True,
    }
    assert audit["actor_path_updates_shared_encoder"] is False
    assert audit["encoder_has_single_optimizer_owner"] is True
    assert audit["optimizer_ownership"]["overlap_count"] == 0


def test_source_only_build_passes_without_claiming_runtime_evidence() -> None:
    audit = build_audit(None, require_runtime=False)
    assert audit["integrity_passed"] is True
    assert audit["engineering_runtime"] == {"available": False, "required": False}
    assert audit["candidate_side_kinematic_or_traffic_risk_projection_present"] is False
    assert audit["inference_rule_mechanism_present"] is False


def test_runtime_audit_rejects_legacy_rule_metadata(tmp_path) -> None:
    run = tmp_path / "run"
    (run / "selector").mkdir(parents=True)
    values = {
        "arguments.json": {
            "requested_raw_steps": {
                "ego_control_profile": "direct",
                "evaluation_split": "validation",
            },
            "implementation_fidelity": {
                "inference_safety_rule_added": False,
                "external_kinematic_projection": False,
                "action_postprocessing_override": False,
                "fixed_speed_grid_at_inference": False,
                "actor_non_keep_confidence_threshold": 0.9,
            },
            "formal_unlock": None,
        },
        "method_metadata.json": {
            "inference_safety_rule_added": False,
            "external_kinematic_projection": False,
            "action_postprocessing_override": False,
            "fixed_speed_grid_at_inference": False,
        },
        "action_diagnostics.json": {
            "inference_safety_rule_added": False,
            "external_kinematic_projection": False,
            "action_postprocessing_override": False,
            "fixed_speed_grid_at_inference": False,
        },
        "paper_evaluation_detailed.json": {"formal_test_accessed": False},
    }
    for name, value in values.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    (run / "action_diagnostics_decisions.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "selector" / "receipt.json").write_text("{}", encoding="utf-8")
    audit = runtime_run_audit(run)
    assert audit["available"] is True
    assert audit["legacy_confidence_rule_metadata_absent"] is False
