from __future__ import annotations

from tools import run_topo_v4_4_experiments as v44


def _context():
    contract = v44.validate_contract(v44.load_contract(v44.DEFAULT_CONTRACT))
    digest = v44.contract_sha256(v44.DEFAULT_CONTRACT)
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
        "success_rate": 0.50,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.30,
        "calibration_episodes": 12,
        "target_route_event_episode_count": 1,
        "target_route_event_applied_match_episode_rate": 0.50,
        "applied_match_positive_target_q_margin_episode_rate": 0.50,
        "exact_target_critic_argmax_rate": 1.0,
    }


def test_v4_4_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = v44.jobs_for_stage(contract, digest, "development")
    promotion = v44.jobs_for_stage(contract, digest, "promotion")
    formal = v44.jobs_for_stage(contract, digest, "formal")

    assert [job.job_id for job in development] == ["D1", "D2", "D3", "D4"]
    assert {job.kind for job in development} == {
        "fresh_training_with_train_only_selection"
    }
    assert [job.scenario for job in development] == [
        "carla",
        "cross",
        "cross",
        "roundabout_medium",
    ]
    assert [job.seed for job in development] == [3, 2, 3, 0]
    assert [job.raw_steps for job in development] == [20_000] * 4
    assert [job.calibration_seed_start for job in development] == [
        64_000,
        65_000,
        66_000,
        67_000,
    ]
    assert [job.evaluation_seed_start for job in development] == [
        55_000,
        56_000,
        57_000,
        58_000,
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


def test_v4_4_commands_apply_selector_symmetrically_and_lock_formal() -> None:
    contract, digest = _context()
    development = v44.jobs_for_stage(contract, digest, "development")[0]
    command = v44.command_for(development, _hashes(), device="cuda")
    assert command[1].endswith("train_paper_sb3_sumo_v4_4.py")
    assert command[command.index("--algo") + 1] == "topo_v4_4_train_only_selector"
    assert command[command.index("--calibration-seed-start") + 1] == "64000"
    assert command[command.index("--calibration-episodes") + 1] == "12"
    assert command[command.index("--evaluation-seed-start") + 1] == "55000"
    assert "--formal-unlock-receipt" not in command

    promotion = v44.jobs_for_stage(contract, digest, "promotion")
    assert {job.algorithm for job in promotion} == set(v44.V44_ALGORITHMS)
    for job in promotion:
        paired_command = v44.command_for(job, _hashes(), device="cuda")
        assert "--calibration-seed-start" in paired_command
        assert "--calibration-episodes" in paired_command

    formal = v44.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v44.command_for(formal, _hashes(), device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_4_target_and_selector_gates_are_noncompensatory() -> None:
    contract, _ = _context()
    passing = _passing_selector_row()
    result = v44._carla_target_gate(passing, contract["development"])
    assert result["passed"]
    assert result["outcome_passed"]
    assert result["mechanism_passed"]
    assert result["selector_passed"]

    decoder_mismatch = dict(passing, exact_target_critic_argmax_rate=0.999)
    mismatch = v44._carla_target_gate(decoder_mismatch, contract["development"])
    assert mismatch["passed"] is False
    assert mismatch["outcome_passed"] is True
    assert mismatch["mechanism_passed"] is False

    unsafe = dict(passing, collision_rate=0.11)
    unsafe_result = v44._carla_target_gate(unsafe, contract["development"])
    assert unsafe_result["passed"] is False
    assert unsafe_result["outcome_passed"] is False
    assert unsafe_result["mechanism_passed"] is True

    selector_drift = dict(passing, calibration_episodes=11)
    selector_result = v44._carla_target_gate(selector_drift, contract["development"])
    assert selector_result["passed"] is False
    assert selector_result["selector_passed"] is False


def test_v4_4_development_state_machine_advances_and_stops_strictly(
    monkeypatch,
) -> None:
    contract, digest = _context()
    accepted_ids = {"D1"}
    row = _passing_selector_row()

    def accepted(job, _hashes_value):
        del _hashes_value
        return (
            job.job_id in accepted_ids,
            "accepted" if job.job_id in accepted_ids else "absent",
        )

    monkeypatch.setattr(v44, "accepted_run_reason", accepted)
    monkeypatch.setattr(v44, "_run_row", lambda _job: dict(row))
    status = v44.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "D2"
    assert status["stopped_after"] is None

    row["success_rate"] = 0.0
    row["timeout_rate"] = 1.0
    stopped = v44.development_status(contract, digest, _hashes())
    assert stopped["decision"] == "fail"
    assert stopped["next_job"] is None
    assert stopped["stopped_after"] == "D1"


def test_v4_4_real_attribution_is_hash_bound() -> None:
    result = v44._validate_failure_attribution(
        v44.DEFAULT_FAILURE_ATTRIBUTION, v44.DEFAULT_DEEP_ATTRIBUTION
    )
    assert result["failure_attribution_sha256"] == v44.PARENT_HASHES[
        "failure_attribution"
    ]
    assert result["deep_attribution_sha256"] == v44.PARENT_HASHES[
        "deep_attribution"
    ]


def test_v4_4_stage_option_is_not_confused_with_global_options() -> None:
    arguments = v44.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
