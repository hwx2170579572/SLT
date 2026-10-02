from __future__ import annotations

from tools import run_topo_v4_experiments as v4


def _context():
    contract = v4.validate_contract(v4.load_contract(v4.DEFAULT_CONTRACT))
    digest = v4.contract_sha256(v4.DEFAULT_CONTRACT)
    return contract, digest


def test_v4_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = v4.jobs_for_stage(contract, digest, "development")
    promotion = v4.jobs_for_stage(contract, digest, "promotion")
    formal = v4.jobs_for_stage(contract, digest, "formal")

    assert len(development) == 5
    assert len(promotion) == 12
    assert len(formal) == 120
    assert [job.job_id for job in development] == ["D1", "D2", "D3", "D4", "D5"]
    assert [job.raw_steps for job in development] == [20_000] * 5
    assert {job.raw_steps for job in promotion} == {50_000}
    assert {job.raw_steps for job in formal} == {100_000}
    assert {job.evaluation_split for job in development + promotion} == {"validation"}
    assert {job.evaluation_split for job in formal} == {"test"}


def test_key_ablation_removes_only_hybrid_action_interface() -> None:
    contract, digest = _context()
    jobs = v4.jobs_for_stage(contract, digest, "development")
    by_id = {job.job_id: job for job in jobs}

    assert by_id["D1"].algorithm == "topo_v4_da_hybrid"
    assert by_id["D3"].algorithm == "topo_v4_continuous_ablation"
    assert by_id["D1"].scenario == by_id["D3"].scenario == "carla"
    assert by_id["D1"].seed == by_id["D3"].seed == 0
    assert by_id["D1"].evaluation_seed_start == by_id["D3"].evaluation_seed_start


def test_evaluation_seed_blocks_pair_methods_without_cross_seed_overlap() -> None:
    contract, digest = _context()
    for stage in ("promotion", "formal"):
        jobs = v4.jobs_for_stage(contract, digest, stage)
        by_cell = {
            (job.method, job.scenario, job.seed): job for job in jobs
        }
        scenarios = contract["promotion" if stage == "promotion" else "formal_test"]["scenarios"]
        seeds = contract["promotion" if stage == "promotion" else "formal_test"]["seeds"]
        for scenario in scenarios:
            starts = []
            for seed in seeds:
                baseline = by_cell[("temporal_graph", scenario, seed)]
                candidate = by_cell[("selected_v4_candidate", scenario, seed)]
                assert baseline.evaluation_seed_start == candidate.evaluation_seed_start
                starts.append(candidate.evaluation_seed_start)
            assert len(starts) == len(set(starts))
            episode_count = jobs[0].evaluation_episodes
            blocks = [set(range(start, start + episode_count)) for start in starts]
            assert all(blocks[i].isdisjoint(blocks[j]) for i in range(len(blocks)) for j in range(i + 1, len(blocks)))


def test_scientific_commands_freeze_raw_clock_and_formal_receipt() -> None:
    contract, digest = _context()
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "implementation_freeze_sha256": "c" * 64,
    }
    development = v4.jobs_for_stage(contract, digest, "development")[0]
    command = v4.command_for(development, hashes, device="cuda")
    assert command[command.index("--max-steps") + 1] == "20000"
    assert command[command.index("--learning-starts") + 1] == "5000"
    assert command[command.index("--checkpoint-freq") + 1] == "20000"
    assert command[command.index("--eval-freq") + 1] == "0"
    assert command[command.index("--evaluation-split") + 1] == "validation"
    assert "--formal-unlock-receipt" not in command

    formal = v4.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = v4.command_for(formal, hashes, device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_non_compensatory_development_gates() -> None:
    contract, _ = _context()
    passing_carla = {
        "success_rate": 0.30,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.70,
        "lane_command_keep_rate": 0.90,
        "lane_change_applied_rate": 0.02,
    }
    assert v4._metric_gate(
        passing_carla, contract["development"]["carla_hard_gate"], carla=True
    )["passed"]
    failing = dict(passing_carla, success_rate=0.29)
    result = v4._metric_gate(
        failing, contract["development"]["carla_hard_gate"], carla=True
    )
    assert result["passed"] is False
    assert result["checks"]["success_rate"] is False


def test_stage_option_is_not_confused_with_stage0_global_options() -> None:
    arguments = v4.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
