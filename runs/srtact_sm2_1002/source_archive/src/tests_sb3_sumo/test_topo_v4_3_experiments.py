from __future__ import annotations

from tools import run_topo_v4_3_experiments as v43


def _context():
    contract = v43.validate_contract(v43.load_contract(v43.DEFAULT_CONTRACT))
    digest = v43.contract_sha256(v43.DEFAULT_CONTRACT)
    return contract, digest


def _hashes() -> dict[str, str]:
    return {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "failure_attribution_sha256": "d" * 64,
        "implementation_freeze_sha256": "e" * 64,
    }


def _passing_carla_row() -> dict:
    return {
        "success_rate": 0.50,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.50,
        "target_route_event_episode_count": 1,
        "target_route_event_applied_match_episode_rate": 0.50,
        "applied_match_positive_target_q_margin_episode_rate": 0.50,
        "exact_target_critic_argmax_rate": 1.0,
    }


def test_v4_3_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = v43.jobs_for_stage(contract, digest, "development")
    promotion = v43.jobs_for_stage(contract, digest, "promotion")
    formal = v43.jobs_for_stage(contract, digest, "formal")

    assert [job.job_id for job in development] == ["R1", "R2", "T1", "T2", "T3"]
    assert [job.kind for job in development] == [
        "frozen_checkpoint_replay",
        "frozen_checkpoint_replay",
        "fresh_training",
        "fresh_training",
        "fresh_training",
    ]
    assert [job.scenario for job in development] == [
        "carla",
        "carla",
        "carla",
        "cross",
        "cross",
    ]
    assert [job.seed for job in development] == [0, 1, 2, 0, 1]
    assert [job.raw_steps for job in development] == [0, 0, 20_000, 20_000, 20_000]
    assert [job.evaluation_seed_start for job in development] == [
        46_000,
        46_000,
        47_000,
        48_000,
        49_000,
    ]
    assert len(promotion) == 12
    assert len(formal) == 120
    assert {job.raw_steps for job in promotion} == {50_000}
    assert {job.raw_steps for job in formal} == {100_000}


def test_v4_3_replay_and_training_commands_bind_distinct_protocols() -> None:
    contract, digest = _context()
    jobs = v43.jobs_for_stage(contract, digest, "development")
    replay_command = v43.command_for(jobs[0], _hashes(), device="cuda")
    assert replay_command[1].endswith("replay_v4_3_checkpoint.py")
    assert replay_command[replay_command.index("--source-model-sha256") + 1] == (
        v43.DEFAULT_REPLAY_MODEL_SHA256["R1"]
    )
    assert replay_command[replay_command.index("--evaluation-seed-start") + 1] == "46000"
    assert replay_command[replay_command.index("--attribution-sha256") + 1] == "d" * 64
    assert "--max-steps" not in replay_command
    assert "--formal-unlock-receipt" not in replay_command

    training_command = v43.command_for(jobs[2], _hashes(), device="cuda")
    assert training_command[1].endswith("train_paper_sb3_sumo_v4_3.py")
    assert training_command[training_command.index("--algo") + 1] == (
        "topo_v4_3_target_critic_decoder"
    )
    assert training_command[training_command.index("--max-steps") + 1] == "20000"
    assert training_command[training_command.index("--seed") + 1] == "2"
    assert training_command[training_command.index("--evaluation-seed-start") + 1] == "47000"

    formal = v43.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v43.command_for(formal, _hashes(), device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_3_target_gate_is_exact_and_noncompensatory() -> None:
    contract, _ = _context()
    passing = _passing_carla_row()
    result = v43._carla_target_gate(passing, contract["development"])
    assert result["passed"]
    assert result["outcome_passed"]
    assert result["mechanism_passed"]

    decoder_mismatch = dict(passing, exact_target_critic_argmax_rate=0.999)
    mismatch_result = v43._carla_target_gate(
        decoder_mismatch, contract["development"]
    )
    assert mismatch_result["passed"] is False
    assert mismatch_result["outcome_passed"] is True
    assert mismatch_result["mechanism_passed"] is False

    unsafe = dict(passing, collision_rate=0.11)
    unsafe_result = v43._carla_target_gate(unsafe, contract["development"])
    assert unsafe_result["passed"] is False
    assert unsafe_result["outcome_passed"] is False
    assert unsafe_result["mechanism_passed"] is True


def test_v4_3_development_state_machine_stops_or_advances_strictly(
    monkeypatch,
) -> None:
    contract, digest = _context()
    accepted_ids = {"R1"}
    row = _passing_carla_row()

    def accepted(job, _hashes):
        return (job.job_id in accepted_ids, "accepted" if job.job_id in accepted_ids else "absent")

    monkeypatch.setattr(v43, "accepted_run_reason", accepted)
    monkeypatch.setattr(v43, "_run_row", lambda _job: dict(row))
    status = v43.development_status(contract, digest, _hashes())
    assert status["decision"] == "incomplete"
    assert status["next_job"] == "R2"
    assert status["stopped_after"] is None

    row["success_rate"] = 0.0
    row["timeout_rate"] = 1.0
    stopped = v43.development_status(contract, digest, _hashes())
    assert stopped["decision"] == "fail"
    assert stopped["next_job"] is None
    assert stopped["stopped_after"] == "R1"


def test_v4_3_real_failure_attribution_is_hash_bound() -> None:
    result = v43._validate_failure_attribution(v43.DEFAULT_FAILURE_ATTRIBUTION)
    assert result["failure_attribution_sha256"] == (
        "d1b84cad36af100c6e446c4730e8893fe09c06f1333bbea2020aa0cfe0a62e5e"
    )


def test_v4_3_stage_option_is_not_confused_with_global_options() -> None:
    arguments = v43.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
