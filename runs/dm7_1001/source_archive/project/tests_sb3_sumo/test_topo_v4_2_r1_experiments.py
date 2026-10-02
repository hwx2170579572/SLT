from __future__ import annotations

from tools import run_topo_v4_2_r1_experiments as r1


def _context():
    contract = r1.validate_contract(r1.load_contract(r1.DEFAULT_CONTRACT))
    digest = r1.contract_sha256(r1.DEFAULT_CONTRACT)
    return contract, digest


def _route_row(
    *,
    episode: int,
    decision: int,
    intent: int,
    lane_command: int,
    applied: bool,
    probabilities: list[float],
) -> dict:
    return {
        "episode": episode,
        "decision": decision,
        "pre_action_route_intent_valid": True,
        "pre_action_route_intent": intent,
        "lane_command": lane_command,
        "lane_change_applied": applied,
        "hybrid_policy": {"lane_probabilities": probabilities},
    }


def test_r1_contract_and_stage_cardinalities_are_frozen() -> None:
    contract, digest = _context()
    development = r1.jobs_for_stage(contract, digest, "development")
    promotion = r1.jobs_for_stage(contract, digest, "promotion")
    formal = r1.jobs_for_stage(contract, digest, "formal")

    assert [job.job_id for job in development] == ["C1", "C2", "C3"]
    assert [job.scenario for job in development] == ["carla", "cross", "cross"]
    assert [job.seed for job in development] == [0, 0, 1]
    assert [job.evaluation_seed_start for job in development] == [44_000, 44_000, 45_000]
    assert {job.raw_steps for job in development} == {20_000}
    assert len(promotion) == 12
    assert len(formal) == 120
    assert {job.raw_steps for job in promotion} == {50_000}
    assert {job.raw_steps for job in formal} == {100_000}


def test_r1_route_event_reducer_uses_episode_and_first_window_denominators() -> None:
    records = [
        _route_row(
            episode=0,
            decision=5,
            intent=-1,
            lane_command=-1,
            applied=True,
            probabilities=[0.9, 0.1, 0.0],
        ),
        _route_row(
            episode=0,
            decision=6,
            intent=-1,
            lane_command=0,
            applied=False,
            probabilities=[0.01, 0.99, 0.0],
        ),
        _route_row(
            episode=1,
            decision=9,
            intent=-1,
            lane_command=0,
            applied=False,
            probabilities=[0.4, 0.6, 0.0],
        ),
        _route_row(
            episode=1,
            decision=10,
            intent=-1,
            lane_command=-1,
            applied=True,
            probabilities=[0.8, 0.2, 0.0],
        ),
        {
            **_route_row(
                episode=2,
                decision=1,
                intent=-1,
                lane_command=-1,
                applied=True,
                probabilities=[1.0, 0.0, 0.0],
            ),
            "pre_action_route_intent_valid": False,
        },
    ]
    metrics = r1.route_event_metrics(records)
    assert metrics["route_event_episode_count"] == 2
    assert metrics["route_event_match_episode_rate"] == 1.0
    assert metrics["route_event_applied_match_episode_rate"] == 1.0
    assert metrics["route_event_positive_margin_episode_rate"] == 0.5
    assert abs(metrics["first_route_window_margin_mean"] - 0.3) < 1e-12
    assert abs(metrics["first_route_window_margin_minimum"] - (-0.2)) < 1e-12
    assert [row["route_window_rows"] for row in metrics["route_event_episode_records"]] == [2, 2]


def test_r1_carla_event_gate_is_strict_and_noncompensatory() -> None:
    contract, _ = _context()
    passing = {
        "success_rate": 0.30,
        "collision_rate": 0.10,
        "off_route_rate": 0.0,
        "timeout_rate": 0.70,
        "route_event_episode_count": 1,
        "route_event_applied_match_episode_rate": 0.80,
        "route_event_positive_margin_episode_rate": 0.80,
        "first_route_window_margin_minimum": 0.1000001,
    }
    result = r1._carla_event_gate(passing, contract["development"])
    assert result["passed"]
    assert result["outcome_passed"]
    assert result["mechanism_passed"]

    boundary = dict(passing, first_route_window_margin_minimum=0.10)
    boundary_result = r1._carla_event_gate(boundary, contract["development"])
    assert boundary_result["passed"] is False
    assert boundary_result["outcome_passed"] is True
    assert boundary_result["mechanism_passed"] is False

    unsafe = dict(passing, collision_rate=0.11)
    unsafe_result = r1._carla_event_gate(unsafe, contract["development"])
    assert unsafe_result["passed"] is False
    assert unsafe_result["outcome_passed"] is False
    assert unsafe_result["mechanism_passed"] is True


def test_r1_pilot_is_recomputed_from_hash_bound_trace_but_not_confirmation() -> None:
    contract, _ = _context()
    pilot = r1.pilot_event_status(contract)
    assert pilot["computed_from_immutable_trace"] is True
    assert pilot["counted_as_confirmation_job"] is False
    assert pilot["row"]["route_event_episode_count"] == 12
    assert pilot["row"]["route_event_applied_match_episode_rate"] == 1.0
    assert pilot["row"]["route_event_positive_margin_episode_rate"] == 1.0
    assert pilot["row"]["first_route_window_margin_minimum"] > 0.96
    assert pilot["gate"]["passed"] is True


def test_r1_command_binds_protocol_attribution_and_keeps_formal_locked() -> None:
    contract, digest = _context()
    hashes = {
        "experiment_contract_sha256": "a" * 64,
        "stage0_results_sha256": "b" * 64,
        "stage0_attribution_sha256": "c" * 64,
        "protocol_attribution_sha256": "d" * 64,
        "implementation_freeze_sha256": "e" * 64,
    }
    development = r1.jobs_for_stage(contract, digest, "development")[0]
    command = r1.command_for(development, hashes, device="cuda")
    assert command[command.index("--algo") + 1] == "topo_v4_2_factorized_entropy"
    assert command[command.index("--max-steps") + 1] == "20000"
    assert command[command.index("--evaluation-seed-start") + 1] == "44000"
    assert command[command.index("--attribution-sha256") + 1] == "d" * 64
    assert "--formal-unlock-receipt" not in command

    formal = r1.jobs_for_stage(contract, digest, "formal")[0]
    formal_command = r1.command_for(formal, hashes, device="cuda")
    assert formal_command[formal_command.index("--evaluation-split") + 1] == "test"
    assert "--formal-unlock-receipt" in formal_command


def test_r1_stage_option_is_not_confused_with_global_options() -> None:
    arguments = r1.parser().parse_args(["plan", "--stage", "development"])
    assert arguments.command == "plan"
    assert arguments.stage == "development"
