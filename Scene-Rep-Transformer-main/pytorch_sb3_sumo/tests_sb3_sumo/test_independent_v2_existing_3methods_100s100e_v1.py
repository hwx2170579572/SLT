from __future__ import annotations

from pathlib import Path

import pytest

from envs.sumo.independent_v2_existing_3methods_100seeds_100episodes_v1 import (
    INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID,
    IndependentV2ExistingThreeMethodEvaluationEnvV1,
    IndependentV2ExistingThreeMethodEvaluationEnvV4V1,
)
from tools import evaluate_independent_v2_existing_models_100s100e_v1 as study
from tools import run_independent_v2_existing_3methods_100s100e_v1 as entrypoint
from tools import run_independent_v2_existing_models_100s100e_v1 as runner


METHODS = ("mst_slt", "temporal_graph", "v4_8")


def _protocol() -> dict:
    return study.load_protocol(entrypoint.DEFAULT_PROTOCOL_PATH)


def test_protocol_freezes_three_by_three_by_one_hundred_by_one_hundred() -> None:
    protocol = _protocol()
    plan = runner._plan_payload(protocol, "comparison")

    assert protocol["experiment_id"] == (
        INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID
    )
    assert tuple(protocol["matrix"]["method_order"]) == METHODS
    assert plan["method_count"] == 3
    assert plan["scenario_count"] == 3
    assert plan["method_scenario_jobs"] == 9
    assert plan["evaluation_blocks"] == 900
    assert plan["total_evaluation_episodes"] == 90_000
    assert plan["training_jobs"] == 0
    assert plan["parent_models_reused"] == 9


def test_removed_methods_are_absent_from_protocol_and_plan() -> None:
    protocol = _protocol()
    planned = {job.method for job in study.build_jobs(protocol, "comparison")}

    assert planned == set(METHODS)
    assert not ({"initial_full", "full_balanced", "v4_1"} & planned)
    assert protocol["methods"]["temporal_graph"]["display_label"] == (
        "TemporalGraph 时序车辆图基线"
    )


def test_formal_seed_blocks_remain_paired_and_non_overlapping() -> None:
    protocol = _protocol()
    profile = study.profile_setting(protocol, "comparison")
    starts = [
        study.episode_seed_start(profile, index)
        for index in study.logical_test_seed_indices(profile)
    ]

    assert starts == list(range(200_000, 210_000, 100))
    assert len(
        {
            seed
            for start in starts
            for seed in range(start, start + profile["episodes_per_test_seed"])
        }
    ) == 10_000


@pytest.mark.parametrize(
    "environment_class",
    (
        IndependentV2ExistingThreeMethodEvaluationEnvV1,
        IndependentV2ExistingThreeMethodEvaluationEnvV4V1,
    ),
)
def test_environment_overlay_suffix_is_unique_to_three_method_study(
    tmp_path: Path, environment_class: type
) -> None:
    environment = environment_class.__new__(environment_class)
    environment._connection = None
    environment._high_density_overlay_root = tmp_path
    environment._high_density_vehicle_scale = 1.5
    environment.scenario = "cross"

    path = environment._overlay_path(Path("traffic_261.rou.xml"))

    assert path.parent == tmp_path / "cross"
    assert path.name.endswith("_iv2e3m100x100_v1.rou.xml")
    assert "_iv2e100x100_v1.rou.xml" not in path.name


def test_three_method_entrypoint_injects_isolated_defaults() -> None:
    values = entrypoint._with_three_method_defaults(["plan"])

    assert str(entrypoint.DEFAULT_PROTOCOL_PATH) in values
    assert str(entrypoint.DEFAULT_RESULT_ROOT) in values


def test_mixed_worker_plan_has_one_cpu_and_two_cuda_slots() -> None:
    worker_plan = runner._device_worker_plan(
        device="auto",
        workers=1,
        cpu_workers=1,
        gpu_workers=2,
    )

    assert worker_plan == {"cpu": 1, "cuda": 2}
    assert runner._device_worker_slots(worker_plan) == [
        ("cpu", 0),
        ("cuda", 0),
        ("cuda", 1),
    ]


def test_mixed_worker_plan_rejects_ambiguous_legacy_options() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        runner._device_worker_plan(
            device="cpu",
            workers=1,
            cpu_workers=1,
            gpu_workers=2,
        )
