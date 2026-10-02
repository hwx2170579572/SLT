from __future__ import annotations

from pathlib import Path

from tools import run_topo_v4_6_experiments as v46


def _context():
    contract = v46.validate_contract(v46.load_contract(v46.DEFAULT_CONTRACT))
    return contract, v46.contract_sha256(v46.DEFAULT_CONTRACT)


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "failure_attribution_sha256": "d" * 64,
        "deep_attribution_sha256": "e" * 64,
        "preregistration_receipt_sha256": "f" * 64,
        "implementation_freeze_sha256": "1" * 64,
    }


def _passing_row(decoder: str = "target_critic") -> dict:
    row = {
        "scenario": "cross",
        "success_rate": 0.50,
        "collision_rate": 0.40,
        "off_route_rate": 0.0,
        "timeout_rate": 0.25,
        "calibration_episodes": 12,
        "selector_candidate_count": 4,
        "selector_mode": "joint_checkpoint_decoder",
        "selector_receipt_precedes_validation": True,
        "selected_deployment_decoder": decoder,
        "selected_decoder_records": 10,
        "selected_decoder_exact_rule_match_rate": 1.0,
        "selected_decoder_exact_action_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "hybrid_exact_lateral_code_rate": 1.0,
    }
    if decoder == "target_critic":
        row.update({
            "exact_target_critic_argmax_rate": 1.0,
            "target_keep_tie_rule_match_rate": 1.0,
        })
    else:
        row.update({
            "actor_non_keep_confidence_threshold": 0.90,
            "actor_override_predicate_valid_rate": 1.0,
            "target_fallback_rule_match_rate": 1.0,
        })
    return row


def test_v4_6_contract_and_stage_cardinalities() -> None:
    contract, digest = _context()
    development = v46.jobs_for_stage(contract, digest, "development")
    promotion = v46.jobs_for_stage(contract, digest, "promotion")
    formal = v46.jobs_for_stage(contract, digest, "formal")
    assert [job.job_id for job in development] == ["F1", "F2", "F3", "F4"]
    assert [job.scenario for job in development] == [
        "cross", "carla", "cross", "roundabout_medium"
    ]
    assert [job.seed for job in development] == [6, 5, 7, 2]
    assert [job.calibration_seed_start for job in development] == [
        72_000, 73_000, 74_000, 75_000
    ]
    assert [job.evaluation_seed_start for job in development] == [
        63_000, 64_000, 65_000, 66_000
    ]
    assert len(promotion) == 12
    assert len(formal) == 120


def test_v4_6_commands_use_new_trainer_and_keep_formal_locked() -> None:
    contract, digest = _context()
    development = v46.jobs_for_stage(contract, digest, "development")[0]
    command = v46.command_for(development, _hashes(), device="cuda")
    assert Path(command[1]).name == "train_paper_sb3_sumo_v4_6.py"
    assert command[command.index("--algo") + 1] == (
        "topo_v4_6_joint_checkpoint_decoder_selector"
    )
    assert command[command.index("--calibration-seed-start") + 1] == "72000"
    assert "--formal-unlock-receipt" not in command
    formal = v46.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v46.command_for(formal, _hashes(), device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_6_target_and_fusion_gates_are_decoder_aware_and_noncompensatory() -> None:
    contract, _ = _context()
    target = v46._candidate_gate(_passing_row("target_critic"), contract)
    fusion = v46._candidate_gate(_passing_row("fusion_0_90"), contract)
    assert target["passed"] and fusion["passed"]

    target_bad = _passing_row("target_critic")
    target_bad["target_keep_tie_rule_match_rate"] = 0.99
    result = v46._candidate_gate(target_bad, contract)
    assert result["passed"] is False
    assert result["outcome_passed"] is True
    assert result["mechanism_passed"] is False

    fusion_bad = _passing_row("fusion_0_90")
    fusion_bad["actor_non_keep_confidence_threshold"] = 0.95
    assert v46._candidate_gate(fusion_bad, contract)["passed"] is False

    unsafe = _passing_row("target_critic")
    unsafe["collision_rate"] = 0.50
    result = v46._candidate_gate(unsafe, contract)
    assert result["outcome_passed"] is False
    assert result["mechanism_passed"] is True


def test_v4_6_development_state_machine_advances_and_stops(monkeypatch) -> None:
    contract, digest = _context()
    accepted_ids = {"F1"}
    row = _passing_row()

    def accepted(job, _hashes_value):
        del _hashes_value
        return job.job_id in accepted_ids, "accepted" if job.job_id in accepted_ids else "absent"

    monkeypatch.setattr(v46, "accepted_run_reason", accepted)
    monkeypatch.setattr(v46, "_run_row", lambda _job: dict(row))
    status = v46.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "F2"

    row["success_rate"] = 0.0
    row["timeout_rate"] = 1.0
    stopped = v46.development_status(contract, digest, _hashes())
    assert stopped["decision"] == "fail"
    assert stopped["next_job"] is None
    assert stopped["stopped_after"] == "F1"


def test_v4_6_real_attribution_and_smoke_selector_are_revalidated() -> None:
    hashes = v46._validate_failure_attribution(
        v46.DEFAULT_FAILURE_ATTRIBUTION, v46.DEFAULT_DEEP_ATTRIBUTION
    )
    assert hashes["failure_attribution_sha256"] == v46.LINEAGE_HASHES[
        "failure_attribution_sha256"
    ]
    smoke = v46.DEFAULT_ENGINEERING_ROOT / "smoke" / "s1"
    if not smoke.is_dir():
        return
    job = v46.Job(
        stage="development", job_id="S1", kind="engineering_smoke",
        method="selected_v4_candidate",
        algorithm="topo_v4_6_joint_checkpoint_decoder_selector",
        implementation_id=v46.V46_IMPLEMENTATION_IDS[
            "topo_v4_6_joint_checkpoint_decoder_selector"
        ],
        scenario="cross", seed=42, raw_steps=60, calibration_episodes=1,
        calibration_seed_start=98_000, evaluation_episodes=1,
        evaluation_split="validation", evaluation_seed_start=99_000,
        role="engineering", protocol_tag="smoke",
    )
    original = v46.run_directory
    try:
        v46.run_directory = lambda _job: smoke
        receipt = v46._validate_selector_evidence(smoke, job)
    finally:
        v46.run_directory = original
    assert receipt["selected_deployment_decoder"] == "target_critic"
    assert receipt["policy_parameter_state_preserved"] is True

