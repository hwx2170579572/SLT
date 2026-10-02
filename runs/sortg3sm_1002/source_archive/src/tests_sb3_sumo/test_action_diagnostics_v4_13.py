from __future__ import annotations

from tools.action_diagnostics_v4_13_model import (
    gradient_isolated_model_integrity_passed,
)


def _diagnostics() -> dict:
    return {
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
        "same_deployed_lane_head_for_support": True,
        "new_inference_head": False,
        "inference_score_equation_changed_from_v4_12": False,
        "scenario_conditioned_inference_rule": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
    }


def test_v413_integrity_requires_model_only_boundary() -> None:
    value = _diagnostics()
    assert gradient_isolated_model_integrity_passed(value) is True
    for key in (
        "new_inference_head",
        "inference_score_equation_changed_from_v4_12",
        "scenario_conditioned_inference_rule",
        "ttc_or_headway_threshold",
        "lane_change_veto",
    ):
        tampered = dict(value)
        tampered[key] = True
        assert gradient_isolated_model_integrity_passed(tampered) is False
