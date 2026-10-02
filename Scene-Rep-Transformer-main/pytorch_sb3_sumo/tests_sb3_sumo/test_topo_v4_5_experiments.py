from __future__ import annotations

from tools import run_topo_v4_5_experiments as v45


def _context():
    contract = v45.validate_contract(v45.load_contract(v45.DEFAULT_CONTRACT))
    digest = v45.contract_sha256(v45.DEFAULT_CONTRACT)
    return contract, digest


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "failure_attribution_sha256": "d" * 64,
        "deep_attribution_sha256": "e" * 64,
        "implementation_freeze_sha256": "f" * 64,
    }


def _passing_selector_row() -> dict:
    return {
        "scenario": "carla",
        "success_rate": 0.50,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.30,
        "calibration_episodes": 12,
        "calibration_uses_method_own_frozen_decoder": True,
        "calibration_deterministic_lane_decoder": (
            "actor_confident_non_keep_else_target_critic"
        ),
        "fusion_decoder_records": 10,
        "actor_non_keep_confidence_threshold": 0.90,
        "exact_fusion_rule_match_rate": 1.0,
        "exact_fusion_action_match_rate": 1.0,
        "actor_override_predicate_valid_rate": 1.0,
        "target_fallback_rule_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "hybrid_exact_lateral_code_rate": 1.0,
        "actor_override_episode_count": 1,
        "target_fallback_episode_count": 1,
    }


def test_v4_5_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = v45.jobs_for_stage(contract, digest, "development")
    promotion = v45.jobs_for_stage(contract, digest, "promotion")
    formal = v45.jobs_for_stage(contract, digest, "formal")

    assert [job.job_id for job in development] == ["E1", "E2", "E3", "E4"]
    assert {job.kind for job in development} == {
        "fresh_training_with_train_only_selection"
    }
    assert [job.scenario for job in development] == [
        "cross",
        "carla",
        "cross",
        "roundabout_medium",
    ]
    assert [job.seed for job in development] == [4, 4, 5, 1]
    assert [job.raw_steps for job in development] == [20_000] * 4
    assert [job.calibration_seed_start for job in development] == [
        68_000,
        69_000,
        70_000,
        71_000,
    ]
    assert [job.evaluation_seed_start for job in development] == [
        59_000,
        60_000,
        61_000,
        62_000,
    ]
    assert len(promotion) == 12
    assert len(formal) == 120
    assert {job.calibration_seed_start for job in promotion} == {80_000, 81_000}
    assert {job.evaluation_seed_start for job in promotion} == {90_000, 91_000}
    assert {job.calibration_seed_start for job in formal} == {
        100_000 + 1_000 * seed for seed in range(10)
    }
    assert {job.evaluation_seed_start for job in formal} == {
        200_000 + 1_000 * seed for seed in range(10)
    }


def test_v4_5_commands_apply_selector_symmetrically_and_lock_formal() -> None:
    contract, digest = _context()
    development = v45.jobs_for_stage(contract, digest, "development")[0]
    command = v45.command_for(development, _hashes(), device="cuda")
    assert command[1].endswith("train_paper_sb3_sumo_v4_5.py")
    assert command[command.index("--algo") + 1] == "topo_v4_5_confident_actor_fusion"
    assert command[command.index("--calibration-seed-start") + 1] == "68000"
    assert command[command.index("--calibration-episodes") + 1] == "12"
    assert command[command.index("--evaluation-seed-start") + 1] == "59000"
    assert "--formal-unlock-receipt" not in command

    promotion = v45.jobs_for_stage(contract, digest, "promotion")
    assert {job.algorithm for job in promotion} == set(v45.V45_ALGORITHMS)
    for job in promotion:
        paired_command = v45.command_for(job, _hashes(), device="cuda")
        assert "--calibration-seed-start" in paired_command
        assert "--calibration-episodes" in paired_command

    formal = v45.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v45.command_for(formal, _hashes(), device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_5_fusion_and_selector_gates_are_noncompensatory() -> None:
    contract, _ = _context()
    passing = _passing_selector_row()
    result = v45._candidate_gate(passing, contract)
    assert result["passed"]
    assert result["outcome_passed"]
    assert result["mechanism_passed"]
    assert result["selector_passed"]

    decoder_mismatch = dict(passing, exact_fusion_rule_match_rate=0.999)
    mismatch = v45._candidate_gate(decoder_mismatch, contract)
    assert mismatch["passed"] is False
    assert mismatch["outcome_passed"] is True
    assert mismatch["mechanism_passed"] is False

    unsafe = dict(passing, collision_rate=0.11)
    unsafe_result = v45._candidate_gate(unsafe, contract)
    assert unsafe_result["passed"] is False
    assert unsafe_result["outcome_passed"] is False
    assert unsafe_result["mechanism_passed"] is True

    selector_drift = dict(passing, calibration_episodes=11)
    selector_result = v45._candidate_gate(selector_drift, contract)
    assert selector_result["passed"] is False
    assert selector_result["selector_passed"] is False


def test_v4_5_development_state_machine_advances_and_stops_strictly(
    monkeypatch,
) -> None:
    contract, digest = _context()
    accepted_ids = {"E1"}
    row = _passing_selector_row()
    row["scenario"] = "cross"

    def accepted(job, _hashes_value):
        del _hashes_value
        return (
            job.job_id in accepted_ids,
            "accepted" if job.job_id in accepted_ids else "absent",
        )

    monkeypatch.setattr(v45, "accepted_run_reason", accepted)
    monkeypatch.setattr(v45, "_run_row", lambda _job: dict(row))
    status = v45.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "E2"
    assert status["stopped_after"] is None

    row["success_rate"] = 0.0
    row["timeout_rate"] = 1.0
    stopped = v45.development_status(contract, digest, _hashes())
    assert stopped["decision"] == "fail"
    assert stopped["next_job"] is None
    assert stopped["stopped_after"] == "E1"


def test_v4_5_real_attribution_is_hash_bound() -> None:
    result = v45._validate_failure_attribution(
        v45.DEFAULT_FAILURE_ATTRIBUTION, v45.DEFAULT_DEEP_ATTRIBUTION
    )
    assert result["failure_attribution_sha256"] == v45.PARENT_HASHES[
        "failure_attribution"
    ]
    assert result["deep_attribution_sha256"] == v45.PARENT_HASHES[
        "deep_attribution"
    ]


def test_v4_5_stage_option_is_not_confused_with_global_options() -> None:
    arguments = v45.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
