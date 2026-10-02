from __future__ import annotations

from tools import attribute_v4_8_promotion_failure as attribution


def test_numeric_summary_is_evidence_preserving() -> None:
    value = attribution.numeric_summary([1.0, 2.0, 3.0, 4.0])
    assert value["count"] == 4
    assert value["mean"] == 2.5
    assert value["minimum"] == 1.0
    assert value["maximum"] == 4.0
    assert value["median"] == 2.5


def test_paired_outcomes_uses_identical_seeds() -> None:
    control = {
        "episode_records": [
            {"seed": 1, "success": True, "traffic_variant": "a", "decision_steps": 4},
            {"seed": 2, "collision": True, "traffic_variant": "b", "decision_steps": 2},
        ]
    }
    candidate = {
        "episode_records": [
            {"seed": 1, "collision": True, "traffic_variant": "a", "decision_steps": 3},
            {"seed": 2, "success": True, "traffic_variant": "b", "decision_steps": 5},
        ]
    }
    value = attribution.paired_outcomes(control, candidate)
    assert value["transition_counts"] == {"collision->success": 1, "success->collision": 1}
    assert value["control_success_candidate_collision"] == 1
    assert value["control_non_success_candidate_success"] == 1


def test_guard_sweep_marks_fixed_observation_as_noncausal() -> None:
    rows = [
        {
            "episode": 0,
            "decision": 0,
            "pre_action_route_intent_valid": True,
            "pre_action_route_intent": 0,
            "fusion_decoder": {
                "actor_selected_lane_index": 0,
                "actor_selected_lane": -1,
                "actor_selected_confidence": 0.99,
                "actor_override": True,
                "target_selected_lane_index": 1,
                "target_selected_lane": 0,
                "keep_was_exact_tied_target_maximum": True,
                "minimum_target_twin_q": [0.0, 1.0, 0.5],
            },
        }
    ]
    values = attribution.guard_sweep(rows, {0: "collision"})
    route_zero = next(
        row for row in values if row["rule"] == "route_and_critic_tolerance_0.00"
    )
    assert route_zero["counterfactual_actor_override_records"] == 0
    assert route_zero["terminal_collision_decisions_intervened"] == 1
    assert route_zero["closed_loop_outcome_inference_forbidden"] is True
    keep_veto = next(row for row in values if row["rule"] == "keep_tie_veto")
    assert keep_veto["counterfactual_actor_override_records"] == 0
    assert keep_veto["terminal_collision_decisions_intervened"] == 1


def test_report_labels_posthoc_and_fresh_evidence() -> None:
    payload = {
        "promotion_gate": {
            "decision": "fail",
            "failed_checks": ["per_scenario_collision_noninferiority"],
            "per_scenario_success_delta": {"cross": -0.05},
            "per_scenario_collision_delta": {"cross": 0.18},
            "macro_success_delta": 0.11,
        },
        "cross_candidate_mechanism": {
            f"seed{seed}": {
                "selected_pair": {"checkpoint": "exact_final", "decoder": "fusion_0_90"},
                "candidate_all_episodes": {"lane_change_applied_rate": 0.4, "actor_override_rate": 0.3},
                "temporal_graph_all_episodes": {"lane_change_applied_rate": 0.04},
                "candidate_collision_episodes": {"actor_override_rate": 0.35},
            }
            for seed in (0, 1)
        },
    }
    report = attribution.render_report(payload)
    assert "post-hoc diagnostic only" in report
    assert "not causal closed-loop results" in report
    assert "fresh development seeds" in report
