from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.ppo_env import PaperPpoCarlaEnv, PaperPpoRgbEnv
from tools.paper_evaluation_contract import (
    expected_environment_class,
    make_injected_evaluation_env,
)
from tools.reproduce_paper_sb3_sumo import (
    Job,
    _accepted_run,
    _aggregate_resampled_curves,
    _best_evaluation_command,
    _diversified_job_order,
    _resample_training_curve,
    _training_curve,
    command_for,
    ensure_protocol_snapshot,
    load_protocol,
    locate_run,
    make_jobs,
)


EXPECTED_MISSIONS = {
    "left_turn": (
        "edge-south-SN edge-west-EW",
        {"depart": "15", "departLane": "1", "departPos": "40"},
    ),
    "cross": (
        "gneE17 gneE5 gneE23",
        {"depart": "5", "departLane": "0", "departPos": "10"},
    ),
    "roundabout_easy": (
        "edge-east-EW edge-east-EN edge-north-SN",
        {"depart": "30", "departLane": "0", "departPos": "1"},
    ),
    "roundabout_medium": (
        "edge-east-EW edge-east-EN edge-north-NW edge-west-EW",
        {"depart": "30", "departLane": "0", "departPos": "1"},
    ),
    "roundabout": (
        "edge-east-EW edge-east-EN edge-north-NW edge-west-WS edge-south-NS",
        {"depart": "30", "departLane": "0", "departPos": "1"},
    ),
}


def _valid_episode_records(episodes: int, seed_start: int) -> list[dict[str, object]]:
    return [
        {
            "episode": episode,
            "seed": seed_start + episode,
            "success": False,
            "collision": False,
            "off_route": False,
            "timeout": True,
            "traffic_variant": f"traffic_{episode % 3}.rou.xml",
        }
        for episode in range(episodes)
    ]


def _valid_evaluation_provenance(
    *,
    algorithm: str,
    scenario: str,
    traffic_protocol: str,
    episode_limit_profile: str,
    episodes: int,
    seed_start: int,
) -> dict[str, object]:
    env_class = expected_environment_class(algorithm, scenario)
    env_class_name = f"{env_class.__module__}.{env_class.__qualname__}"
    observation_signature = {"type": "test-observation-space"}
    action_signature = {"type": "test-action-space"}
    variants = sorted(
        {record["traffic_variant"] for record in _valid_episode_records(episodes, seed_start)}
    )
    return {
        "contract_version": 1,
        "validated": True,
        "environment_factory": "tests.paper_factory",
        "environment_class": env_class_name,
        "expected_environment_class": env_class_name,
        "algorithm": algorithm,
        "scenario": scenario,
        "traffic_protocol": traffic_protocol,
        "traffic_partition": "all" if traffic_protocol == "source_all" else "evaluation",
        "episode_limit_profile": episode_limit_profile,
        "uses_released_assets": True,
        "evaluation_seed_start": seed_start,
        "episodes": episodes,
        "traffic_variant_episode_count": episodes,
        "traffic_variants_observed": variants,
        "model_environment_spaces_match": True,
        "environment_observation_space": observation_signature,
        "model_observation_space": observation_signature,
        "environment_action_space": action_signature,
        "model_action_space": action_signature,
    }


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_all_evaluation_phases_use_the_injected_environment_factory() -> None:
    calls: list[bool] = []
    sentinel = object()

    def factory(_args: argparse.Namespace, *, evaluation: bool = False) -> object:
        calls.append(evaluation)
        return sentinel

    assert make_injected_evaluation_env(factory, argparse.Namespace()) is sentinel
    assert calls == [True]


def test_v12_orchestrator_uses_method_specific_source_clocks(tmp_path: Path) -> None:
    protocol = load_protocol()
    assert protocol["version"] == 12
    args = argparse.Namespace(device="cpu")

    def job(method: str, algorithm: str, scenario: str) -> Job:
        return Job(
            profile="paper",
            method=method,
            cli_algorithm=algorithm,
            scenario=scenario,
            seed=0,
            raw_steps=100_000,
            test_episodes=50,
            checkpoint_frequency=20_000,
            evaluation_frequency=0,
        )

    mst = command_for(args, tmp_path, job("mst", "mst", "left_turn"))
    assert _command_value(mst, "--algo") == "mst"
    assert _command_value(mst, "--action-repeat") == "3"
    assert _command_value(mst, "--learning-rate") == "0.0001"
    assert _command_value(mst, "--learning-starts") == "5000"

    for scenario in ("left_turn", "carla"):
        ppo = command_for(args, tmp_path, job("ppo", "ppo", scenario))
        assert _command_value(ppo, "--action-repeat") == "1"
        assert _command_value(ppo, "--learning-rate") == "0.0005"
        assert _command_value(ppo, "--learning-starts") == "0"


def test_v14_baselines_are_interleaved_and_use_corrected_environment() -> None:
    protocol = json.loads(
        Path("experiments/sb3_sumo_paper/protocol_baselines_v14.json").read_text(
            encoding="utf-8"
        )
    )
    args = argparse.Namespace(
        profile="paper", methods="sac,ppo", scenarios="left_turn", seeds="0"
    )
    jobs = make_jobs(args, protocol)
    assert [job.method for job in jobs] == ["sac", "ppo"]
    assert all(
        job.ego_control_profile == "smarts_ackermann_proxy"
        and job.traffic_protocol == "frozen_80_20"
        and job.episode_limit_profile == "paper"
        for job in jobs
    )
    assert jobs[0].implementation_id == "paper_sac_lstm_reconstruction_v2"
    assert jobs[1].implementation_id == "released_source_ppo_pytorch_port_v1"


def test_diversified_schedule_starts_different_scenarios_and_algorithms() -> None:
    protocol = json.loads(
        Path("experiments/sb3_sumo_paper/protocol_baselines_v14.json").read_text(
            encoding="utf-8"
        )
    )
    scenarios = (
        "left_turn,cross,roundabout_easy,roundabout_medium,roundabout"
    )
    args = argparse.Namespace(
        profile="paper", methods="sac,ppo", scenarios=scenarios, seeds="all"
    )
    jobs = make_jobs(args, protocol)
    ordered = _diversified_job_order(jobs)
    first_wave = ordered[:5]
    assert [job.scenario for job in first_wave] == scenarios.split(",")
    assert [job.method for job in first_wave] == [
        "sac",
        "ppo",
        "sac",
        "ppo",
        "sac",
    ]
    assert all(job.seed == 0 for job in first_wave)
    assert {job.name for job in ordered} == {job.name for job in jobs}


def test_v15_proposed_mst_use_the_same_corrected_environment() -> None:
    protocol = json.loads(
        Path("experiments/sb3_sumo_paper/protocol_proposed_mst_v15.json").read_text(
            encoding="utf-8"
        )
    )
    args = argparse.Namespace(
        profile="paper", methods="proposed,mst", scenarios="left_turn", seeds="0"
    )
    jobs = make_jobs(args, protocol)
    assert [job.method for job in jobs] == ["proposed", "mst"]
    assert all(
        job.ego_control_profile == "smarts_ackermann_proxy"
        and job.traffic_protocol == "frozen_80_20"
        and job.episode_limit_profile == "paper"
        for job in jobs
    )
    assert jobs[0].implementation_id == "released_scene_rep_pytorch_port_v1"
    assert jobs[1].implementation_id == "released_mst_pytorch_port_v1"

    carla_args = argparse.Namespace(
        profile="paper", methods="proposed,mst", scenarios="carla", seeds="0"
    )
    assert [job.method for job in make_jobs(carla_args, protocol)] == ["proposed"]


def test_best_checkpoint_evaluation_reuses_job_environment_contract(
    tmp_path: Path,
) -> None:
    protocol = json.loads(
        Path("experiments/sb3_sumo_paper/protocol_baselines_v14.json").read_text(
            encoding="utf-8"
        )
    )
    job = Job(
        profile="paper",
        method="sac",
        cli_algorithm="sac",
        scenario="cross",
        seed=2,
        raw_steps=100_000,
        test_episodes=50,
        checkpoint_frequency=20_000,
        evaluation_frequency=0,
        ego_control_profile="smarts_ackermann_proxy",
        traffic_protocol="frozen_80_20",
        episode_limit_profile="paper",
    )
    command = _best_evaluation_command(
        argparse.Namespace(device="cpu"),
        tmp_path / "evaluation.json",
        tmp_path / "model.zip",
        job,
        protocol,
    )
    assert _command_value(command, "--seed") == "10002"
    assert _command_value(command, "--ego-control-profile") == (
        "smarts_ackermann_proxy"
    )
    assert _command_value(command, "--traffic-protocol") == "frozen_80_20"
    assert _command_value(command, "--episode-limit-profile") == "paper"


def test_protocol_snapshot_rejects_result_root_mixing(tmp_path: Path) -> None:
    snapshot = ensure_protocol_snapshot(tmp_path)
    assert snapshot.read_bytes() == Path(
        "experiments/sb3_sumo_paper/protocol.json"
    ).read_bytes()
    snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Protocol mismatch"):
        ensure_protocol_snapshot(tmp_path)


def test_paper_training_matrix_includes_figure9_mst_carla_run() -> None:
    protocol = load_protocol()
    args = argparse.Namespace(
        profile="paper", methods="all", scenarios="all", seeds="0"
    )
    jobs = make_jobs(args, protocol)
    assert len(jobs) == 18
    assert any(job.method == "mst" and job.scenario == "carla" for job in jobs)
    assert sum(job.method == "proposed" for job in jobs) == 6
    assert sum(job.method == "mst" for job in jobs) == 6
    assert sum(job.method == "ppo" for job in jobs) == 6


def test_compatible_predecessor_requires_hash_and_detailed_evidence(
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "predecessor"
    run = predecessor / "paper__proposed__left_turn__seed0"
    run.mkdir(parents=True)
    snapshot_bytes = b'{"version": 9}\n'
    (predecessor / "protocol.snapshot.json").write_bytes(snapshot_bytes)
    (run / "arguments.json").write_text(
        json.dumps(
            {
                "requested_raw_steps": {
                    "max_steps": 100_000,
                    "seed": 0,
                    "scenario": "left_turn",
                    "algo": "scene_rep",
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "final_model.zip").write_bytes(b"model")
    summary = {
        "episodes": 50,
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "off_route_rate": 0.0,
        "timeout_rate": 1.0,
    }
    (run / "final_evaluation.json").write_text(json.dumps(summary), encoding="utf-8")
    detailed = {
        "algorithm": "scene_rep",
        "scenario": "left_turn",
        "evaluation_seed_start": 10_000,
        "trained_raw_steps": 100_000,
        "collected_training_raw_steps": 100_000,
        "source_test_checkpoint_step": None,
        "evaluation_provenance": _valid_evaluation_provenance(
            algorithm="scene_rep",
            scenario="left_turn",
            traffic_protocol="source_all",
            episode_limit_profile="source",
            episodes=50,
            seed_start=10_000,
        ),
        "episode_records": _valid_episode_records(50, 10_000),
        "summary": summary,
    }
    (run / "paper_evaluation_detailed.json").write_text(
        json.dumps(detailed), encoding="utf-8"
    )
    checkpoint_rows = []
    for raw_step in range(20_000, 100_001, 20_000):
        relative = Path("checkpoints") / f"scene_rep_raw_{raw_step}_steps.zip"
        checkpoint = run / relative
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_rows.append(
            {
                "raw_step": raw_step,
                "path": str(relative),
                "clock_field": "_raw_steps_seen",
                "expected_clock": raw_step,
                "recorded_clock": raw_step,
                "bytes": checkpoint.stat().st_size,
                "sha256": "A" * 64,
                "zip_crc_ok": True,
            }
        )
    (run / "checkpoint_audit.json").write_text(
        json.dumps(
            {
                "algorithm": "scene_rep",
                "scenario": "left_turn",
                "requested_raw_steps": 100_000,
                "checkpoint_frequency_raw_steps": 20_000,
                "clock_field": "_raw_steps_seen",
                "expected_checkpoint_count": 5,
                "checkpoints": checkpoint_rows,
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "compatible_predecessor_results": [
            {
                "root": str(predecessor),
                "protocol_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "methods": ["proposed"],
            }
        ]
    }
    job = Job(
        profile="paper",
        method="proposed",
        cli_algorithm="scene_rep",
        scenario="left_turn",
        seed=0,
        raw_steps=100_000,
        test_episodes=50,
        checkpoint_frequency=20_000,
        evaluation_frequency=0,
    )
    status, directory, origin = locate_run(tmp_path / "primary", job, protocol)
    assert status == "imported"
    assert directory == run
    assert origin == "compatible_predecessor"


def test_ppo_acceptance_checks_horizon_and_checkpoint_clocks(tmp_path: Path) -> None:
    run = tmp_path / "paper__ppo__left_turn__seed0"
    run.mkdir()
    (run / "arguments.json").write_text(
        json.dumps(
            {
                "requested_raw_steps": {
                    "max_steps": 100_000,
                    "seed": 0,
                    "scenario": "left_turn",
                    "algo": "ppo",
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "final_model.zip").write_bytes(b"model")
    summary = {
        "episodes": 50,
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "off_route_rate": 0.0,
        "timeout_rate": 1.0,
    }
    (run / "final_evaluation.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (run / "paper_evaluation_detailed.json").write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "scenario": "left_turn",
                "evaluation_seed_start": 10_000,
                "trained_raw_steps": 100_000,
                "collected_training_raw_steps": 100_352,
                "source_test_checkpoint_step": 100_000,
                "evaluation_provenance": _valid_evaluation_provenance(
                    algorithm="ppo",
                    scenario="left_turn",
                    traffic_protocol="source_all",
                    episode_limit_profile="source",
                    episodes=50,
                    seed_start=10_000,
                ),
                "episode_records": _valid_episode_records(50, 10_000),
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_rows = []
    for raw_step in range(20_000, 100_001, 20_000):
        relative = Path("checkpoints") / f"ppo_{raw_step}_steps.zip"
        checkpoint = run / relative
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_rows.append(
            {
                "raw_step": raw_step,
                "path": str(relative),
                "clock_field": "num_timesteps",
                "expected_clock": raw_step,
                "recorded_clock": raw_step,
                "bytes": checkpoint.stat().st_size,
                "sha256": "B" * 64,
                "zip_crc_ok": True,
            }
        )
    audit = {
        "algorithm": "ppo",
        "scenario": "left_turn",
        "requested_raw_steps": 100_000,
        "checkpoint_frequency_raw_steps": 20_000,
        "clock_field": "num_timesteps",
        "expected_checkpoint_count": 5,
        "checkpoints": checkpoint_rows,
    }
    audit_path = run / "checkpoint_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    job = Job(
        profile="paper",
        method="ppo",
        cli_algorithm="ppo",
        scenario="left_turn",
        seed=0,
        raw_steps=100_000,
        test_episodes=50,
        checkpoint_frequency=20_000,
        evaluation_frequency=0,
    )
    assert _accepted_run(run, job)

    detailed_path = run / "paper_evaluation_detailed.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    detailed["episode_records"][0]["traffic_variant"] = None
    detailed_path.write_text(json.dumps(detailed), encoding="utf-8")
    assert not _accepted_run(run, job)
    detailed["episode_records"][0]["traffic_variant"] = "traffic_0.rou.xml"
    detailed["evaluation_provenance"]["model_observation_space"] = {
        "type": "wrong-space"
    }
    detailed_path.write_text(json.dumps(detailed), encoding="utf-8")
    assert not _accepted_run(run, job)
    detailed["evaluation_provenance"] = _valid_evaluation_provenance(
        algorithm="ppo",
        scenario="left_turn",
        traffic_protocol="source_all",
        episode_limit_profile="source",
        episodes=50,
        seed_start=10_000,
    )
    detailed_path.write_text(json.dumps(detailed), encoding="utf-8")
    assert _accepted_run(run, job)

    audit["checkpoints"][-1]["recorded_clock"] = 100_002
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert not _accepted_run(run, job)


def test_figure9_resampling_applies_ema_before_seed_aggregation() -> None:
    episodes = [
        {"raw_steps": 100, "success_rate_last_20": 0.0},
        {"raw_steps": 250, "success_rate_last_20": 0.5},
        {"raw_steps": 450, "success_rate_last_20": 1.0},
    ]
    first = _resample_training_curve(episodes, 600)
    assert [row["raw_steps"] for row in first] == [200, 400, 600]
    assert [row["success_rate_last_20"] for row in first] == [0.0, 0.5, 1.0]
    assert first[0]["success_rate_ema_0_99"] == pytest.approx(0.0)
    assert first[1]["success_rate_ema_0_99"] == pytest.approx(0.005)
    assert first[2]["success_rate_ema_0_99"] == pytest.approx(0.01495)

    tagged = []
    for seed, scale in ((0, 1.0), (1, 0.0)):
        for row in first:
            tagged.append(
                {
                    "profile": "paper",
                    "method": "proposed",
                    "scenario": "left_turn",
                    "seed": seed,
                    "raw_steps": row["raw_steps"],
                    "success_rate_last_20": row["success_rate_last_20"] * scale,
                    "success_rate_ema_0_99": row["success_rate_ema_0_99"] * scale,
                }
            )
    aggregate = _aggregate_resampled_curves(tagged)
    assert aggregate[-1]["seeds_available"] == 2
    assert aggregate[-1]["success_rate_ema_0_99_mean"] == pytest.approx(
        0.01495 / 2.0
    )


def test_figure9_early_success_rate_keeps_source_denominator_20(
    tmp_path: Path,
) -> None:
    (tmp_path / "train_monitor.csv").write_text(
        '#{}\nr,l,raw_simulation_steps\n1,10,30\n-1,5,15\n',
        encoding="utf-8",
    )
    curve = _training_curve(tmp_path, action_repeat=3)
    assert curve[0]["success_rate_last_20"] == pytest.approx(0.05)
    assert curve[1]["success_rate_last_20"] == pytest.approx(0.05)
    assert curve[1]["raw_steps"] == 45


def test_source_ppo_rgb_contract_and_training_reset_scale_bug() -> None:
    env = PaperPpoRgbEnv(scenario="left_turn", action_repeat=1, training=True)
    try:
        first, first_info = env.reset(seed=123)
        second, second_info = env.reset(seed=124)
        assert first.shape == (80, 80, 3)
        assert first.dtype == np.float32
        assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
        assert float(second.max()) > 1.0
        assert first_info["source_top_down_rgb"] == "RGB(80,80,32/80)"
        assert not first_info["source_reset_scale_bug_applied"]
        assert second_info["source_reset_scale_bug_applied"]
        road = np.asarray([80, 80, 80], dtype=np.float32)
        ego = np.asarray([210, 30, 30], dtype=np.float32)
        assert np.any(np.all(np.isclose(second, road), axis=-1))
        assert np.any(np.all(np.isclose(second, ego), axis=-1))
    finally:
        env.close()


def test_carla_ppo_exposes_every_raw_step_and_vector_state() -> None:
    env = PaperPpoCarlaEnv(scenario="carla", action_repeat=1, training=False)
    try:
        observation, info = env.reset(seed=55)
        assert observation.shape == (6, 10, 5)
        assert info["source_ppo_action_hold"] == 3
        next_observation, _, terminated, truncated, step_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert next_observation.shape == (6, 10, 5)
        assert step_info["raw_steps_executed"] == 1
        assert step_info["raw_simulation_steps"] == 1
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_ego_missions_are_transcribed_from_released_scenario_sources() -> None:
    for scenario, (route_edges, vehicle_attributes) in EXPECTED_MISSIONS.items():
        root = ET.parse(PAPER_SCENARIOS[scenario].ego_route_path).getroot()
        route = root.find("route")
        vehicle = root.find("vehicle")
        assert route is not None and route.attrib["edges"] == route_edges
        assert vehicle is not None
        for name, value in vehicle_attributes.items():
            assert vehicle.attrib[name] == value


def test_release_contains_at_least_the_source_generated_traffic_variants() -> None:
    minimum_counts = {
        "left_turn": 20,
        "cross": 60,
        "roundabout_easy": 15,
        "roundabout_medium": 20,
        "roundabout": 15,
    }
    for scenario, minimum in minimum_counts.items():
        assert len(PAPER_SCENARIOS[scenario].traffic_paths) >= minimum


def test_released_traffic_uses_smarts_lexicographic_rolled_cycle() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        paths = PAPER_SCENARIOS["left_turn"].traffic_paths
        assert [path.name for path in paths] == sorted(path.name for path in paths)
        roll = random.Random(42).randint(0, len(paths))
        observed = []
        for seed in (100, 200, 300):
            env._sumo_command(seed)
            assert env._selected_traffic_path is not None
            observed.append(env._selected_traffic_path.name)
        expected = [
            paths[(-roll + episode) % len(paths)].name for episode in range(3)
        ]
        assert observed == expected
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "source_limit", "paper_limit"),
    (("cross", 600, 400), ("roundabout", 1000, 800)),
)
def test_episode_limit_profiles_preserve_source_and_paper_contracts(
    scenario: str, source_limit: int, paper_limit: int
) -> None:
    source_env = PaperSumoSceneEnv(
        scenario=scenario, episode_limit_profile="source"
    )
    paper_env = PaperSumoSceneEnv(
        scenario=scenario, episode_limit_profile="paper"
    )
    try:
        assert source_env.max_episode_steps == source_limit
        assert paper_env.max_episode_steps == paper_limit
    finally:
        source_env.close()
        paper_env.close()


@pytest.mark.parametrize("scenario", tuple(EXPECTED_MISSIONS))
def test_frozen_traffic_partitions_are_deterministic_and_disjoint(
    scenario: str,
) -> None:
    specification = PAPER_SCENARIOS[scenario]
    train_env = PaperSumoSceneEnv(scenario=scenario, traffic_partition="train")
    eval_env = PaperSumoSceneEnv(
        scenario=scenario, traffic_partition="evaluation"
    )
    try:
        training = train_env._partitioned_traffic_paths(specification)
        evaluation = eval_env._partitioned_traffic_paths(specification)
        assert set(training).isdisjoint(evaluation)
        assert set(training) | set(evaluation) == set(specification.traffic_paths)
        assert evaluation == specification.traffic_paths[::5]
    finally:
        train_env.close()
        eval_env.close()


def test_ackermann_proxy_bounds_acceleration_and_reports_profile() -> None:
    env = PaperSumoSceneEnv(
        scenario="roundabout_easy",
        action_repeat=1,
        ego_control_profile="smarts_ackermann_proxy",
    )
    try:
        env.reset(seed=0)
        previous_speed = float(
            env._connection.vehicle.getSpeed(env.specification.ego_id)
        )
        _, _, _, _, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))
        assert info["ego_control_profile"] == "smarts_ackermann_proxy"
        assert info["effective_target_speed"] <= previous_speed + 0.26 + 1e-6
        assert info["effective_target_speed"] <= 10.0
    finally:
        env.close()


def test_sac_state_lstm_observation_matches_released_adapter_fields() -> None:
    env = PaperSumoSceneEnv(
        scenario="left_turn", action_repeat=3, include_state_lstm=True
    )
    try:
        observation, _ = env.reset(seed=0)
        state_lstm = observation["state_lstm"]
        mask = observation["state_lstm_mask"]
        assert state_lstm.shape == (10, 30)
        np.testing.assert_array_equal(mask, [1.0] + [0.0] * 9)

        trajectory = observation["trajectory"]
        ego = trajectory[0, 0]
        np.testing.assert_allclose(
            state_lstm[0, :5],
            [
                ego[0],
                ego[1],
                np.linalg.norm(ego[3:5]),
                0.0,
                ego[2],
            ],
            atol=1e-6,
        )
        for actor_index in range(1, 6):
            actor = trajectory[actor_index, 0]
            actual = state_lstm[0, actor_index * 5 : (actor_index + 1) * 5]
            if not np.any(actor):
                np.testing.assert_array_equal(actual, np.zeros(5))
                continue
            np.testing.assert_allclose(
                actual,
                [
                    actor[0],
                    actor[1],
                    np.linalg.norm(actor[3:5]),
                    np.linalg.norm(actor[:2] - ego[:2]),
                    actor[2],
                ],
                atol=1e-5,
            )

        next_observation, _, terminated, truncated, _ = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        np.testing.assert_array_equal(
            next_observation["state_lstm_mask"], [1.0] * 4 + [0.0] * 6
        )
    finally:
        env.close()


def test_sac_state_lstm_only_observation_excludes_proposed_inputs() -> None:
    env = PaperSumoSceneEnv(
        scenario="left_turn",
        action_repeat=3,
        include_state_lstm=True,
        state_lstm_only=True,
    )
    try:
        observation, _ = env.reset(seed=0)
        assert set(observation) == {"state_lstm", "state_lstm_mask"}
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_paper_sumo_command_preserves_source_traffic_runtime_flags() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        command = env._sumo_command(0)
        options = dict(zip(command[1::2], command[2::2]))
        assert options["--collision.action"] == "none"
        assert options["--lanechange.duration"] == "3.0"
        assert options["--default.action-step-length"] == "0.1"
        assert options["--end"] == "31536000"
    finally:
        env.close()


def test_ego_uses_the_source_external_controller_safety_modes() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        ego_id = env.specification.ego_id
        # SMARTS v0.4.17 SumoTrafficSimulation.sync() sets no_checks=0b00000
        # for vehicles whose state is controlled by another provider.
        assert env._connection.vehicle.getSpeedMode(ego_id) == 0
        assert env._connection.vehicle.getLaneChangeMode(ego_id) == 0
        assert env._connection.vehicle.getTau(ego_id) == pytest.approx(4.0)
        assert env._connection.vehicle.getDecel(ego_id) == pytest.approx(6.0)
        assert env._connection.vehicle.getLength(ego_id) == pytest.approx(3.68)
        assert env._connection.vehicle.getWidth(ego_id) == pytest.approx(1.47)
        assert env._connection.vehicle.getHeight(ego_id) == pytest.approx(1.4)
    finally:
        env.close()


def test_smarts_state_uses_provider_center_not_sumo_front_bumper() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        ego_id = env.specification.ego_id
        front = np.asarray(env._connection.vehicle.getPosition(ego_id))
        angle = math.radians(float(env._connection.vehicle.getAngle(ego_id)))
        expected_center = front - 0.5 * 3.68 * np.asarray(
            (math.sin(angle), math.cos(angle))
        )
        state = env._state(f"vehicle:{ego_id}")
        assert state is not None
        np.testing.assert_allclose(state[:2], expected_center, atol=1e-5)
    finally:
        env.close()


def test_source_waypoint_and_neighborhood_contracts() -> None:
    for scenario in EXPECTED_MISSIONS:
        env = PaperSumoSceneEnv(scenario=scenario)
        try:
            assert env.path_spacing == pytest.approx(1.0)
            assert math.isinf(env.neighbor_radius)
            assert env.specification.source_observation_contract == "smarts"
            assert env._sumo_lane_offset(-1) == -1
            assert env._sumo_lane_offset(1) == 1
        finally:
            env.close()
    carla_env = PaperSumoSceneEnv(scenario="carla")
    try:
        assert carla_env.path_spacing == pytest.approx(2.5)
        assert math.isinf(carla_env.neighbor_radius)
        assert carla_env.specification.source_observation_contract == "carla"
        assert carla_env._sumo_lane_offset(-1) == 1
        assert carla_env._sumo_lane_offset(1) == -1
    finally:
        carla_env.close()


def test_smarts_fixed_route_paths_preserve_v0417_internal_filter_bug() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        lane_one = np.asarray(
            env._connection.lane.getShape("edge-south-SN_1"), dtype=np.float32
        )
        direction = lane_one[-1] - lane_one[-2]
        point = lane_one[-1] - 3.0 * direction / np.linalg.norm(direction)
        paths = env._route_polylines(
            ["edge-south-SN", "edge-west-EW"], point=point, limit=2
        )
        assert len(paths) == 2
        # Source ordering starts with lane 0 and then lane 1.
        assert paths[0][0, 0] > paths[1][0, 0]
        # Even lane 1's valid left turn is truncated: v0.4.17 filters the
        # immediate internal edge because it is absent from the fixed route.
        assert np.array_equal(paths[1], lane_one)
    finally:
        env.close()


def test_smarts_endless_neighbor_paths_are_unconstrained_near_junction() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        lane_one = np.asarray(
            env._connection.lane.getShape("edge-south-SN_1"), dtype=np.float32
        )
        direction = lane_one[-1] - lane_one[-2]
        point = lane_one[-1] - 3.0 * direction / np.linalg.norm(direction)
        paths = env._smarts_waypoint_polylines(
            route=["edge-south-SN", "edge-west-EW"],
            road_id="edge-south-SN",
            position=point,
            is_ego=False,
            limit=8,
        )
        internal_lane_ids = [
            lane_id
            for lane_id in env._connection.lane.getIDList()
            if lane_id.startswith(":gneJ2")
        ]
        internal_points = [
            internal_point
            for lane_id in internal_lane_ids
            for internal_point in np.asarray(
                env._connection.lane.getShape(lane_id), dtype=np.float32
            )[1:]
        ]
        assert any(
            np.allclose(path_point, internal_point)
            for path in paths
            for path_point in path
            for internal_point in internal_points
        )
    finally:
        env.close()


def test_smarts_path_duplication_uses_candidate_count(monkeypatch) -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        ego_key = f"vehicle:{env.specification.ego_id}"
        state = env._state(ego_key)
        assert state is not None
        position = state[:2]
        long_path = np.asarray(
            [position, position + np.array([2.0, 0.0], dtype=np.float32)]
        )
        zero_after_slice = np.asarray(
            [position, position + np.array([0.2, 0.0], dtype=np.float32)]
        )

        monkeypatch.setattr(
            env,
            "_smarts_waypoint_polylines",
            lambda **kwargs: [long_path, zero_after_slice],
        )
        two_candidates = env._actor_paths(ego_key, is_ego=True)
        assert np.any(two_candidates[0])
        assert not np.any(two_candidates[1])

        monkeypatch.setattr(
            env,
            "_smarts_waypoint_polylines",
            lambda **kwargs: [long_path],
        )
        one_candidate = env._actor_paths(ego_key, is_ego=True)
        assert np.array_equal(one_candidate[0], one_candidate[1])
    finally:
        env.close()


def test_smarts_heading_and_velocity_match_released_adapter() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        env.reset(seed=0)
        angle = float(env._connection.vehicle.getAngle(env.specification.ego_id))
        speed = float(env._connection.vehicle.getSpeed(env.specification.ego_id))
        state = env._state(f"vehicle:{env.specification.ego_id}")
        assert state is not None
        expected_heading = (-math.radians(angle) + math.pi) % (2 * math.pi) - math.pi
        assert state[2] == pytest.approx(expected_heading)
        assert state[3] == pytest.approx(speed * math.cos(expected_heading))
        assert state[4] == pytest.approx(speed * math.sin(expected_heading))
    finally:
        env.close()


def test_released_smarts_map_uses_one_metre_waypoints() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn")
    try:
        observation, _ = env.reset(seed=0)
        point_distances = np.linalg.norm(
            np.diff(observation["map"][0, :, :2], axis=0), axis=1
        )
        assert np.allclose(point_distances, 1.0, atol=0.05)
    finally:
        env.close()


def test_carla_ego_map_reuses_exact_released_waypoint_arrays() -> None:
    env = PaperSumoSceneEnv(scenario="carla")
    try:
        observation, _ = env.reset(seed=0)
        state = env._state(f"vehicle:{env.specification.ego_id}")
        assert state is not None
        # TraCI uses netconvert's translated coordinates, but the released
        # CARLA tensors and actor states are in the original Town-10 frame.
        assert np.linalg.norm(state[:2] - np.asarray((0.0, -64.5))) < 3.0
        assert env.specification.coordinate_offset == pytest.approx((95.0, 64.83))
        source_root = Path(__file__).resolve().parents[1] / "envs" / "carla" / "map"
        paths = [np.load(source_root / name).astype(np.float32) for name in ("wp.npy", "wp2.npy")]
        paths.sort(
            key=lambda path: float(
                np.min(np.linalg.norm(path[:, :2] - state[None, :2], axis=1))
            )
        )
        np.testing.assert_allclose(observation["map"][0], paths[0][:50:5, :2])
        np.testing.assert_allclose(observation["map"][1], paths[1][:50:5, :2])
        assert not np.any(observation["map"][2])
    finally:
        env.close()


def test_carla_history_preserves_released_query_layout() -> None:
    env = PaperSumoSceneEnv(scenario="carla", history_steps=10)
    try:
        values = [np.full(5, value, dtype=np.float32) for value in (1, 2, 3)]
        env._histories["vehicle:test"] = deque(values, maxlen=10)
        history = env._history_array("vehicle:test")
        np.testing.assert_array_equal(history[0], values[2])
        np.testing.assert_array_equal(history[8], values[0])
        np.testing.assert_array_equal(history[9], values[1])
        assert not np.any(history[1:8])
    finally:
        env.close()


def test_smarts_neighbor_history_uses_source_left_padding() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn", history_steps=10)
    try:
        ego_values = [np.full(5, value, dtype=np.float32) for value in (1, 2)]
        neighbor_value = np.full(5, 9, dtype=np.float32)
        env._history_timestep = 6
        env._histories[f"vehicle:{env.specification.ego_id}"] = deque(
            ego_values, maxlen=10
        )
        env._histories["vehicle:late"] = deque([neighbor_value], maxlen=10)
        ego_history = env._history_array(f"vehicle:{env.specification.ego_id}")
        neighbor_history = env._history_array("vehicle:late")
        np.testing.assert_array_equal(ego_history[:2], ego_values)
        assert not np.any(ego_history[2:])
        assert not np.any(neighbor_history[:5])
        np.testing.assert_array_equal(neighbor_history[5], neighbor_value)
        assert not np.any(neighbor_history[6:])
    finally:
        env.close()


def test_smarts_neighbor_history_interpolates_source_timestamp_gaps() -> None:
    env = PaperSumoSceneEnv(scenario="left_turn", history_steps=10)
    try:
        actor_key = "vehicle:reappearing"
        first = np.asarray([1.0, 2.0, 0.1, 3.0, 4.0], dtype=np.float32)
        last = np.asarray([4.0, 8.0, 0.4, 9.0, 10.0], dtype=np.float32)
        env._history_timestep = 2
        env._append_history(actor_key, first)
        env._history_timestep = 5
        env._append_history(actor_key, last)
        values = np.asarray(env._histories[actor_key])
        expected = np.stack(
            [
                first,
                first + (last - first) / 3.0,
                first + 2.0 * (last - first) / 3.0,
                last,
            ]
        )
        np.testing.assert_allclose(values, expected)
        assert env._last_seen_steps[actor_key] == 5
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "action"),
    [
        ("left_turn", (0.6, 0.0)),
        ("cross", (0.6, -0.8)),
        ("roundabout_easy", (0.4, 0.0)),
        ("roundabout_medium", (-0.2, 0.0)),
        ("roundabout", (0.4, 0.0)),
        ("carla", (0.6, -0.8)),
    ],
)
def test_paper_scenario_goal_is_reachable_in_real_sumo(
    scenario: str, action: tuple[float, float]
) -> None:
    env = PaperSumoSceneEnv(scenario=scenario, action_repeat=3)
    try:
        _, reset_info = env.reset(seed=0)
        assert reset_info["scenario_asset_source"] in {
            "authors_release_v1.0.0",
            "carla_source_waypoint_reconstruction",
        }
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
        assert info["is_success"]
        assert not info["collision"]
        assert not info["off_route"]
        assert info["raw_simulation_steps"] <= env.max_episode_steps
    finally:
        env.close()


def test_carla_equivalent_requires_the_source_target_lane_change() -> None:
    env = PaperSumoSceneEnv(scenario="carla", action_repeat=3)
    try:
        env.reset(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.asarray((0.6, 0.0), dtype=np.float32)
            )
        assert info["max_time"]
        assert not info["is_success"]
        # carla_env.py exposes max_time as done, and its runner does not apply
        # the SMARTS-only time-limit bootstrap rewrite.
        assert terminated
        assert not truncated
        assert info["source_terminal_for_bootstrap"]
    finally:
        env.close()


def test_paper_sumo_runtime_does_not_load_replaced_simulators_or_tensorflow() -> None:
    forbidden_roots = {"carla", "smarts", "tensorflow", "tf2rl"}
    loaded_roots = {name.split(".", 1)[0] for name in sys.modules}
    assert forbidden_roots.isdisjoint(loaded_roots)
