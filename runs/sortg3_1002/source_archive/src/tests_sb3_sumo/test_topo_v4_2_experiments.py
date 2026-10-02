from __future__ import annotations

from tools import run_topo_v4_2_experiments as v42


def _context():
    contract = v42.validate_contract(v42.load_contract(v42.DEFAULT_CONTRACT))
    digest = v42.contract_sha256(v42.DEFAULT_CONTRACT)
    return contract, digest


def test_v4_2_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = v42.jobs_for_stage(contract, digest, "development")
    promotion = v42.jobs_for_stage(contract, digest, "promotion")
    formal = v42.jobs_for_stage(contract, digest, "formal")

    assert [job.job_id for job in development] == ["F1", "F2", "F3", "F4"]
    assert [job.scenario for job in development] == ["carla", "carla", "cross", "cross"]
    assert [job.seed for job in development] == [1, 0, 0, 1]
    assert [job.evaluation_seed_start for job in development] == [43_000, 42_000, 42_000, 43_000]
    assert {job.raw_steps for job in development} == {20_000}
    assert len(promotion) == 12
    assert len(formal) == 120
    assert {job.raw_steps for job in promotion} == {50_000}
    assert {job.raw_steps for job in formal} == {100_000}


def test_v4_2_evaluation_blocks_pair_methods_without_seed_overlap() -> None:
    contract, digest = _context()
    for stage in ("promotion", "formal"):
        jobs = v42.jobs_for_stage(contract, digest, stage)
        by_cell = {(job.method, job.scenario, job.seed): job for job in jobs}
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        for scenario in settings["scenarios"]:
            starts = []
            for seed in settings["seeds"]:
                baseline = by_cell[("temporal_graph", scenario, seed)]
                candidate = by_cell[("selected_v4_candidate", scenario, seed)]
                assert baseline.evaluation_seed_start == candidate.evaluation_seed_start
                starts.append(candidate.evaluation_seed_start)
            assert len(starts) == len(set(starts))
            blocks = [
                set(range(start, start + int(settings["evaluation_episodes"])))
                for start in starts
            ]
            assert all(
                blocks[i].isdisjoint(blocks[j])
                for i in range(len(blocks))
                for j in range(i + 1, len(blocks))
            )


def test_v4_2_command_binds_attribution_and_raw_clock() -> None:
    contract, digest = _context()
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "v4_1_attribution_sha256": "d" * 64,
        "implementation_freeze_sha256": "e" * 64,
    }
    development = v42.jobs_for_stage(contract, digest, "development")[0]
    command = v42.command_for(development, hashes, device="cuda")
    assert command[command.index("--algo") + 1] == "topo_v4_2_factorized_entropy"
    assert command[command.index("--max-steps") + 1] == "20000"
    assert command[command.index("--learning-starts") + 1] == "5000"
    assert command[command.index("--eval-freq") + 1] == "0"
    assert command[command.index("--evaluation-seed-start") + 1] == "43000"
    assert command[command.index("--attribution-sha256") + 1] == "d" * 64
    assert "--formal-unlock-receipt" not in command

    formal = v42.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v42.command_for(formal, hashes, device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_v4_2_carla_gate_is_noncompensatory_and_margin_bound() -> None:
    contract, _ = _context()
    passing = {
        "success_rate": 0.30,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.70,
        "lane_command_keep_rate": 0.90,
        "lane_change_applied_rate": 0.02,
        "route_action_window_samples": 1,
        "route_action_window_match_rate": 0.80,
        "route_minus_keep_probability_margin_mean": 0.05,
        "route_minus_keep_probability_margin_minimum": 1e-8,
    }
    result = v42._carla_v42_gate(passing, contract["development"])
    assert result["passed"]
    assert result["outcome_passed"]
    assert result["mechanism_passed"]

    thin = dict(passing, route_minus_keep_probability_margin_minimum=0.0)
    thin_result = v42._carla_v42_gate(thin, contract["development"])
    assert thin_result["passed"] is False
    assert thin_result["outcome_passed"] is True
    assert thin_result["mechanism_passed"] is False

    unsafe = dict(passing, collision_rate=0.11)
    unsafe_result = v42._carla_v42_gate(unsafe, contract["development"])
    assert unsafe_result["passed"] is False
    assert unsafe_result["outcome_passed"] is False
    assert unsafe_result["mechanism_passed"] is True


def test_v4_2_stage_option_is_not_confused_with_global_options() -> None:
    arguments = v42.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
