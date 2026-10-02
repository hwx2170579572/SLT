from __future__ import annotations

from pathlib import Path

from tools import audit_v4_12_no_kinematic_projection as subject


def test_source_and_dynamic_model_audit_passes_without_runtime() -> None:
    value = subject.build_audit(None, require_runtime=False)
    assert value["integrity_passed"] is True
    assert value["candidate_side_kinematic_or_traffic_risk_projection_present"] is False
    assert value["inference_rule_mechanism_present"] is False
    assert value["physical_lane_availability_mask_present"] is True
    assert value["physical_lane_availability_mask_is_traffic_risk_rule"] is False
    model = value["model_execution"]
    assert model["lane_support_gradient_reaches_lane_head"] is True
    assert model["lane_prior_score_equation_exact"] is True
    assert model["selected_action_equals_exact_actor_proposal"] is True


def test_real_sumo_runtime_trace_passes_model_only_boundary() -> None:
    value = subject.runtime_run_audit(subject.DEFAULT_RUN)
    assert value["available"] is True
    assert value["all_rule_projection_fields_false"] is True
    assert value["all_exact_model_action_rates_are_one"] is True
    assert value["independent_trace_recomputation_matches"] is True
    assert value["all_joint_support_training_statistics_finite"] is True


def test_required_runtime_does_not_silently_pass_missing_directory(
    tmp_path: Path,
) -> None:
    value = subject.build_audit(tmp_path, require_runtime=True)
    assert value["integrity_passed"] is False
    assert value["engineering_runtime"]["available"] is False
