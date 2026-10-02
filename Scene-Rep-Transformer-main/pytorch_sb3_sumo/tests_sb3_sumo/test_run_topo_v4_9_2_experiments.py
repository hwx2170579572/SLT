from __future__ import annotations

from pathlib import Path

from tools import run_all_pending_promotions as registry
from tools import run_topo_v4_9_2_experiments as runner
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER


def test_contract_and_promotion_matrix_are_strict_target_only():
    contract = runner.validate_contract(runner.load_contract())
    jobs = runner.jobs_for_stage(contract, runner.CONTRACT_SHA256, "promotion")
    assert len(jobs) == 12
    assert len({job.name for job in jobs}) == 12
    assert {job.scenario for job in jobs} == {"cross", "roundabout_medium", "carla"}
    assert {job.seed for job in jobs} == {20, 21}
    assert contract["single_protocol_change"]["deployment_decoder_candidates"] == [TARGET_DECODER]
    assert contract["single_protocol_change"]["fusion_0_90_candidate_present"] is False
    assert contract["single_protocol_change"]["external_kinematic_projection"] is False


def test_promotion_command_uses_new_trainer_and_hash_bound_output():
    contract = runner.validate_contract(runner.load_contract())
    job = runner.jobs_for_stage(contract, runner.CONTRACT_SHA256, "promotion")[6]
    hashes = {
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "stage0_results_sha256": runner.STAGE0_RESULTS_SHA256,
        "failure_attribution_sha256": runner.FAILURE_ATTRIBUTION_SHA256,
        "implementation_freeze_sha256": "f" * 64,
    }
    command = runner.command_for(job, hashes, device="cuda")
    assert command[1].endswith("train_paper_sb3_sumo_v4_9_2.py")
    assert "fusion_0_90" not in command
    assert "--formal-unlock-receipt" not in command
    assert job.protocol_tag == runner.CONTRACT_SHA256[:8]


def test_candidate_gate_adds_noncompensatory_no_rule_checks():
    parent_contract = runner.parent.validate_contract(
        runner.parent.load_contract(runner.PARENT_CONTRACT)
    )
    row = {
        "scenario": "cross",
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "off_route_rate": 0.0,
        "timeout_rate": 0.0,
        "learned_collision_critic_present": True,
        "risk_adjusted_score_equation_match_rate": 1.0,
        "inference_safety_rule_added": False,
        "selected_action_mask_feasible_rate": 1.0,
        "stored_collision_event_count": 1,
        "sampled_positive_collision_label_rate": 0.1,
        "selected_deployment_decoder": TARGET_DECODER,
        "strict_target_only_protocol": True,
        "fusion_candidate_present": False,
        "actor_confidence_threshold_present": False,
        "external_kinematic_projection": False,
        "action_postprocessing_override": False,
    }
    gate = runner._candidate_gate(row, parent_contract)
    assert gate["passed"] is True
    row["fusion_candidate_present"] = True
    gate = runner._candidate_gate(row, parent_contract)
    assert gate["passed"] is False
    assert gate["checks"]["fusion_candidate_absent"] is False


def test_registered_promotion_failures_do_not_cancel_later_versions(
    monkeypatch, tmp_path: Path
):
    calls = []

    def failed(*, device):
        calls.append(("failed", device))
        return 2, {
            "status": "promotion_gate_failed",
            "accepted_jobs": 12,
            "expected_jobs": 12,
            "promotion_gate_decision": "fail",
        }

    def passed(*, device):
        calls.append(("passed", device))
        return 0, {
            "status": "promotion_gate_passed",
            "accepted_jobs": 12,
            "expected_jobs": 12,
            "promotion_gate_decision": "pass",
        }

    monkeypatch.setattr(registry, "REGISTRY", (("failed", failed), ("passed", passed)))
    code, payload = registry.run_registered(
        device="cpu", report_path=tmp_path / "registry.json"
    )
    assert code == 1
    assert calls == [("failed", "cpu"), ("passed", "cpu")]
    assert len(payload["versions"]) == 2
    assert payload["failure_does_not_cancel_remaining_versions_or_jobs"] is True
