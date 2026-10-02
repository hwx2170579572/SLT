from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from envs.sumo.independent_v2_existing_models_100seeds_100episodes_v1 import (
    INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID,
    IndependentV2ExistingModelsEvaluationEnvV1,
    IndependentV2ExistingModelsEvaluationEnvV4V1,
)
from tools import evaluate_independent_v2_existing_models_100s100e_v1 as study
from tools import run_independent_v2_existing_models_100s100e_v1 as runner


def test_protocol_freezes_six_by_three_by_one_hundred_by_one_hundred() -> None:
    protocol = study.load_protocol()
    plan = runner._plan_payload(protocol, "comparison")

    assert protocol["experiment_id"] == (
        INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID
    )
    assert plan["method_count"] == 6
    assert plan["scenario_count"] == 3
    assert plan["method_scenario_jobs"] == 18
    assert plan["test_seed_blocks_per_job"] == 100
    assert plan["episodes_per_test_seed"] == 100
    assert plan["evaluation_blocks"] == 1_800
    assert plan["total_evaluation_episodes"] == 180_000
    assert plan["training_jobs"] == 0
    assert plan["parent_models_reused"] == 18


def test_formal_seed_blocks_are_paired_contiguous_and_non_overlapping() -> None:
    protocol = study.load_protocol()
    profile = study.profile_setting(protocol, "comparison")
    indices = study.logical_test_seed_indices(profile)
    starts = [study.episode_seed_start(profile, index) for index in indices]

    assert indices == tuple(range(100))
    assert starts[0] == 200_000
    assert starts[-1] == 209_900
    assert all(right - left == 100 for left, right in zip(starts, starts[1:]))
    all_episode_seeds = {
        seed
        for start in starts
        for seed in range(start, start + profile["episodes_per_test_seed"])
    }
    assert len(all_episode_seeds) == 10_000


@pytest.mark.parametrize(
    "environment_class",
    (
        IndependentV2ExistingModelsEvaluationEnvV1,
        IndependentV2ExistingModelsEvaluationEnvV4V1,
    ),
)
def test_environment_overlay_suffix_is_distinct_from_parent_v2(
    tmp_path: Path, environment_class: type
) -> None:
    environment = environment_class.__new__(environment_class)
    environment._connection = None
    environment._high_density_overlay_root = tmp_path
    environment._high_density_vehicle_scale = 1.5
    environment.scenario = "cross"

    path = environment._overlay_path(Path("traffic_261.rou.xml"))

    assert path.parent == tmp_path / "cross"
    assert path.name.endswith("_iv2e100x100_v1.rou.xml")
    assert "_ss100_v2.rou.xml" not in path.name


@pytest.mark.parametrize(
    "environment_class",
    (
        IndependentV2ExistingModelsEvaluationEnvV1,
        IndependentV2ExistingModelsEvaluationEnvV4V1,
    ),
)
def test_logical_block_positions_route_cycle_for_exact_resume(
    environment_class: type,
) -> None:
    environment = environment_class.__new__(environment_class)
    environment._connection = None
    environment._traffic_episode_index = 999
    environment._traffic_roll = 17

    schedule = environment.set_logical_test_seed_block(
        logical_index=37,
        profile_index_start=0,
        episodes_per_test_seed=100,
    )

    assert schedule["source_traffic_episode_start"] == 3_700
    assert environment._traffic_episode_index == 3_700
    assert environment._traffic_roll is None
    assert environment._independent_v2_logical_test_seed_index == 37


def _valid_two_episode_block() -> tuple[dict, study.EvaluationJob]:
    job = study.EvaluationJob(
        profile="smoke",
        method="mst_slt",
        scenario="carla",
        run_id="iv2e100x100_v1__smoke__mst_slt__carla",
    )
    records = [
        {
            "episode": 0,
            "seed": 900_000,
            "episode_return": 1.0,
            "decision_steps": 10,
            "environment_steps": 10,
            "raw_steps": 30,
            "completion_time_seconds": 3.0,
            "success": True,
            "collision": False,
            "off_route": False,
            "timeout": False,
            "traffic_variant": "traffic_0.rou.xml",
        },
        {
            "episode": 1,
            "seed": 900_001,
            "episode_return": -1.0,
            "decision_steps": 20,
            "environment_steps": 20,
            "raw_steps": 60,
            "completion_time_seconds": None,
            "success": False,
            "collision": True,
            "off_route": False,
            "timeout": False,
            "traffic_variant": "traffic_0.rou.xml",
        },
    ]
    payload = {
        "schema_version": study.BLOCK_SCHEMA_VERSION,
        "status": "completed",
        "computed_from_real_run": True,
        "fabricated_values": False,
        "training_performed": False,
        "protocol_sha256": "protocol-sha",
        "profile": "smoke",
        "method": "mst_slt",
        "scenario": "carla",
        "job_id": job.run_id,
        "model_sha256": "model-sha",
        "test_seed_block": {
            "logical_index": 0,
            "episode_seed_start": 900_000,
            "episode_seed_end": 900_001,
            "episode_count": 2,
        },
        "evaluation_provenance": {
            "validated": True,
            "evaluation_seed_start": 900_000,
            "episodes": 2,
            "model_environment_spaces_match": True,
        },
        "summary": {
            "episodes": 2,
            "mean_return": 0.0,
            "std_return": 1.0,
            "mean_decision_steps": 15.0,
            "mean_raw_steps": 45.0,
            "success_rate": 0.5,
            "collision_rate": 0.5,
            "off_route_rate": 0.0,
            "timeout_rate": 0.0,
        },
        "successful_episodes": 1,
        "mean_success_completion_time_seconds": 3.0,
        "std_success_completion_time_seconds": 0.0,
        "episode_records": records,
    }
    return payload, job


def test_block_validator_recomputes_seed_and_metric_evidence() -> None:
    payload, job = _valid_two_episode_block()

    study.validate_block_payload(
        payload,
        protocol_sha256="protocol-sha",
        job=job,
        source_model_sha256="model-sha",
        logical_index=0,
        expected_seed_start=900_000,
        expected_episodes=2,
    )

    tampered = copy.deepcopy(payload)
    tampered["summary"]["success_rate"] = 1.0
    with pytest.raises(ValueError, match="success_rate"):
        study.validate_block_payload(
            tampered,
            protocol_sha256="protocol-sha",
            job=job,
            source_model_sha256="model-sha",
            logical_index=0,
            expected_seed_start=900_000,
            expected_episodes=2,
        )


def test_protocol_model_selection_preserves_v4_8_selected_deployment() -> None:
    protocol = study.load_protocol()

    assert protocol["parent_v2"]["model_selection_rule"].startswith(
        "Use the model basename recorded"
    )
    assert protocol["methods"]["v4_8"]["model_class"].endswith(
        "ConfidentActorFusionSACV45"
    )
    assert protocol["environment_contract"]["artifact_namespace_only_change"] is True


def test_queue_process_identity_binds_pid_and_creation_time() -> None:
    identity = runner._process_identity(os.getpid())

    assert identity["pid"] == os.getpid()
    assert identity["create_time"] > 0
    assert identity["name"]
    assert Path(identity["executable"]).is_file()
