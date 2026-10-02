from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from envs.sumo.high_density_single_seed_100ep_v2 import (
    HighDensitySingleSeed100EpisodeEnvV2,
    HighDensitySingleSeed100EpisodeEnvV4V2,
)
from tools import run_high_density_same_scene_v1 as shared_runner
from tools import run_high_density_single_seed_100ep_v2 as v2_runner
from tools import train_high_density_single_seed_100ep_v2 as v2_trainer
from tools import train_sb3
from tools.high_density_same_scene_v1_common import (
    build_jobs,
    load_protocol,
    profile_setting,
)


def test_v2_protocol_freezes_one_seed_and_one_hundred_episodes() -> None:
    protocol = load_protocol(v2_runner.DEFAULT_PROTOCOL_PATH)
    profile = profile_setting(protocol, "comparison")
    jobs = build_jobs(protocol, "comparison")

    assert protocol["run_id_prefix"] == "hd_ss100_v2"
    assert profile["seeds"] == [0]
    assert profile["evaluation_episodes"] == 100
    assert profile["raw_training_steps"] == 50_000
    assert len(jobs) == 18
    assert len({job.run_id for job in jobs}) == 18
    assert all(job.run_id.startswith("hd_ss100_v2__") for job in jobs)
    assert {job.seed for job in jobs} == {0}

    plan = shared_runner._plan_payload(protocol, "comparison")
    assert plan["seed_count"] == 1
    assert plan["training_jobs"] == 18
    assert plan["evaluation_episodes"] == 1_800


def test_v2_scenario_and_artifact_names_are_distinct_from_v1() -> None:
    protocol = load_protocol(v2_runner.DEFAULT_PROTOCOL_PATH)
    assert all(
        setting["scenario_id"].endswith("single_seed_100ep_v2")
        for setting in protocol["density"].values()
    )
    assert protocol["artifact_contract"]["result_root"] == "results_hd_ss100_v2"
    assert v2_trainer.OVERLAY_DIRECTORY_NAME == "overlays_hd_ss100_v2"
    assert v2_runner.DEFAULT_RESULT_ROOT.name == "results_hd_ss100_v2"


def test_v2_environment_uses_a_distinct_overlay_suffix(tmp_path: Path) -> None:
    for environment_class in (
        HighDensitySingleSeed100EpisodeEnvV2,
        HighDensitySingleSeed100EpisodeEnvV4V2,
    ):
        environment = environment_class.__new__(environment_class)
        environment._connection = None
        environment._high_density_overlay_root = tmp_path
        environment._high_density_vehicle_scale = 1.5
        environment.scenario = "cross"
        path = environment._overlay_path(Path("traffic_261.rou.xml"))
        assert path.parent == tmp_path / "cross"
        assert path.name.endswith("_ss100_v2.rou.xml")
        assert "_v1.rou.xml" not in path.name


def test_v2_wrappers_inject_only_v2_defaults() -> None:
    run_args = v2_runner._with_v2_defaults(["plan", "--profile", "comparison"])
    assert run_args[run_args.index("--protocol") + 1] == str(
        v2_runner.DEFAULT_PROTOCOL_PATH
    )
    assert run_args[run_args.index("--result-root") + 1] == str(
        v2_runner.DEFAULT_RESULT_ROOT
    )

    train_args = v2_trainer._with_v2_protocol(
        ["--method", "mst_slt", "--scenario", "carla"]
    )
    assert train_args[train_args.index("--protocol") + 1] == str(
        v2_trainer.DEFAULT_PROTOCOL_PATH
    )


def test_single_seed_summary_does_not_invent_cross_seed_variance() -> None:
    assert shared_runner._mean_std([]) == (None, None)
    assert shared_runner._mean_std([0.75]) == (0.75, None)


class _CapturedEnvironment:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _factory_args(*, evaluation_split: str = "validation") -> Namespace:
    return Namespace(
        scenario="cross",
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split=evaluation_split,
    )


def test_v2_v4_factory_keeps_calibration_on_train_partition(tmp_path: Path) -> None:
    factory = v2_trainer._make_v2_environment_factory(
        adapter="v4_8",
        density={
            "vehicle_scale": 1.5,
            "pedestrian_scale": 1.0,
            "clone_depart_jitter_seconds": [8.0, 10.0],
        },
        overlay_root=tmp_path,
        v4_environment_class=_CapturedEnvironment,
    )
    environment = factory(_factory_args(evaluation_split="train"), evaluation=True)
    assert environment.kwargs["high_density_partition"] == "train"
    assert environment.kwargs["high_density_contract_partition"] == "train"


def test_v2_v4_factory_labels_matched_holdout_as_validation(tmp_path: Path) -> None:
    factory = v2_trainer._make_v2_environment_factory(
        adapter="v4_1",
        density={
            "vehicle_scale": 1.5,
            "pedestrian_scale": 1.0,
            "clone_depart_jitter_seconds": [8.0, 10.0],
        },
        overlay_root=tmp_path,
        v4_environment_class=_CapturedEnvironment,
    )
    environment = factory(_factory_args(), evaluation=True)
    assert environment.kwargs["high_density_partition"] == "evaluation"
    assert environment.kwargs["high_density_contract_partition"] == "validation"


def test_v2_base_factory_keeps_evaluation_contract_label(tmp_path: Path) -> None:
    factory = v2_trainer._make_v2_environment_factory(
        adapter="base",
        density={
            "vehicle_scale": 1.5,
            "pedestrian_scale": 1.0,
            "clone_depart_jitter_seconds": [8.0, 10.0],
        },
        overlay_root=tmp_path,
        baseline_environment_class=_CapturedEnvironment,
    )
    environment = factory(_factory_args(), evaluation=True)
    assert environment.kwargs["high_density_partition"] == "evaluation"
    assert environment.kwargs["high_density_contract_partition"] == "evaluation"


def test_v2_tensorboard_override_uses_short_stable_token(tmp_path: Path) -> None:
    run_dir = tmp_path / ("very_long_run_name_" * 8)
    root = tmp_path / v2_trainer.TENSORBOARD_DIRECTORY_NAME
    first = train_sb3._tensorboard_log_directory(run_dir, root)
    second = train_sb3._tensorboard_log_directory(run_dir, root)
    assert first == second
    assert first.parent == root.resolve()
    assert len(first.name) == 16
    assert "very_long_run_name" not in str(first)
