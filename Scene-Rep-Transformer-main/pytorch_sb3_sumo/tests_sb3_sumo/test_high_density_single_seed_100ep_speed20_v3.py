from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs.sumo.high_density_single_seed_100ep_speed20_v3 import (
    EGO_SPEED_CONTROL_INTERVAL_MPS,
    HighDensitySingleSeed100EpisodeSpeed20EnvV3,
    HighDensitySingleSeed100EpisodeSpeed20EnvV4V3,
    SPEED20_V3_EXPERIMENT_ID,
)
from envs.sumo.high_density_single_seed_100ep_v2 import (
    HighDensitySingleSeed100EpisodeEnvV2,
)
from tools import run_high_density_same_scene_v1 as shared_runner
from tools import run_high_density_single_seed_100ep_speed20_v3 as speed20_runner
from tools import train_high_density_single_seed_100ep_speed20_v3 as speed20_trainer
from tools.high_density_same_scene_v1_common import (
    build_jobs,
    load_protocol,
    profile_setting,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_V2_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_single_seed_100ep_v2"
    / "protocol.json"
)


def test_speed20_protocol_freezes_the_requested_matrix() -> None:
    protocol = load_protocol(speed20_runner.DEFAULT_PROTOCOL_PATH)
    profile = profile_setting(protocol, "comparison")
    jobs = build_jobs(protocol, "comparison")

    assert protocol["run_id_prefix"] == "hd_ss100_s20_v3"
    assert profile["seeds"] == [0]
    assert profile["evaluation_episodes"] == 100
    assert profile["raw_training_steps"] == 50_000
    assert len(jobs) == 18
    assert len({job.run_id for job in jobs}) == 18
    assert all(job.run_id.startswith("hd_ss100_s20_v3__") for job in jobs)

    plan = shared_runner._plan_payload(protocol, "comparison")
    assert plan["method_count"] == 6
    assert plan["scenario_count"] == 3
    assert plan["seed_count"] == 1
    assert plan["training_jobs"] == 18
    assert plan["evaluation_episodes"] == 1_800


def test_speed20_protocol_keeps_v2_method_density_and_budget_values() -> None:
    parent = load_protocol(PARENT_V2_PROTOCOL_PATH)
    speed20 = load_protocol(speed20_runner.DEFAULT_PROTOCOL_PATH)

    assert speed20["matrix"]["method_order"] == parent["matrix"]["method_order"]
    assert speed20["matrix"]["scenario_order"] == parent["matrix"]["scenario_order"]
    assert speed20["methods"] == parent["methods"]
    assert speed20["iteration_selection"] == parent["iteration_selection"]
    for scenario in parent["matrix"]["scenario_order"]:
        for key in (
            "vehicle_scale",
            "pedestrian_scale",
            "clone_depart_jitter_seconds",
        ):
            assert speed20["density"][scenario][key] == parent["density"][scenario][key]
    for profile in ("smoke", "comparison"):
        for key in (
            "raw_training_steps",
            "learning_starts_raw_steps",
            "checkpoint_frequency_raw_steps",
            "evaluation_episodes",
            "calibration_episodes",
            "seeds",
        ):
            assert speed20["profiles"][profile][key] == parent["profiles"][profile][key]
    assert (
        speed20["profiles"]["comparison"]["evaluation_seed_start"]
        == parent["profiles"]["comparison"]["evaluation_seed_start"]
    )


def test_speed20_protocol_records_the_only_physical_change() -> None:
    protocol = load_protocol(speed20_runner.DEFAULT_PROTOCOL_PATH)
    contract = protocol["environment_contract"]

    assert contract["only_change_from_parent_v2"] == (
        "ego_speed_control_interval_mps"
    )
    assert contract["parent_ego_speed_control_interval_mps"] == [0.0, 10.0]
    assert contract["ego_speed_control_interval_mps"] == [0.0, 20.0]
    assert contract["normalized_longitudinal_action_interval"] == [-1.0, 1.0]
    assert contract["runtime_ego_max_speed_mps"] == 20.0
    assert contract["map_changed"] is False
    assert contract["ego_route_changed"] is False
    assert contract["traffic_density_changed_from_parent_v2"] is False
    assert contract["traffic_partition_changed_from_parent_v2"] is False
    assert contract["observation_changed"] is False
    assert contract["action_space_changed"] is False
    assert contract["lateral_action_changed"] is False
    assert contract["reward_changed"] is False
    assert contract["source_ego_xml_changed"] is False


@pytest.mark.parametrize(
    "environment_class",
    (
        HighDensitySingleSeed100EpisodeSpeed20EnvV3,
        HighDensitySingleSeed100EpisodeSpeed20EnvV4V3,
    ),
)
def test_speed20_action_mapping_changes_only_longitudinal_scale(
    environment_class: type,
) -> None:
    assert EGO_SPEED_CONTROL_INTERVAL_MPS == (0.0, 20.0)
    for action, expected_speed in (
        (np.array([-1.0, -1.0]), 0.0),
        (np.array([0.0, 0.0]), 10.0),
        (np.array([1.0, 1.0]), 20.0),
        (np.array([2.0, 0.5]), 20.0),
    ):
        speed, lane = environment_class.adapt_action(action)
        _, parent_lane = HighDensitySingleSeed100EpisodeEnvV2.adapt_action(action)
        assert speed == expected_speed
        assert lane == parent_lane


class _FakeVehicleAPI:
    def __init__(self) -> None:
        self.speed_mode: tuple[str, int] | None = None
        self.lane_mode: tuple[str, int] | None = None
        self.max_speed: tuple[str, float] | None = None
        self.commanded_speed: tuple[str, float] | None = None

    def setSpeedMode(self, ego_id: str, mode: int) -> None:
        self.speed_mode = (ego_id, mode)

    def setLaneChangeMode(self, ego_id: str, mode: int) -> None:
        self.lane_mode = (ego_id, mode)

    def setMaxSpeed(self, ego_id: str, speed: float) -> None:
        self.max_speed = (ego_id, float(speed))

    def setSpeed(self, ego_id: str, speed: float) -> None:
        self.commanded_speed = (ego_id, float(speed))


def _bare_speed20_environment() -> tuple[object, _FakeVehicleAPI]:
    environment = HighDensitySingleSeed100EpisodeSpeed20EnvV3.__new__(
        HighDensitySingleSeed100EpisodeSpeed20EnvV3
    )
    vehicle = _FakeVehicleAPI()
    environment._connection = SimpleNamespace(vehicle=vehicle)
    environment.specification = SimpleNamespace(
        ego_id="ego", source_observation_contract="carla"
    )
    environment.ego_control_profile = "direct"
    return environment, vehicle


def test_speed20_runtime_configuration_and_clip_are_both_twenty() -> None:
    environment, vehicle = _bare_speed20_environment()
    environment._configure_policy_controlled_ego("ego")
    effective = environment._apply_speed_control(99.0)

    assert vehicle.speed_mode == ("ego", 0)
    assert vehicle.lane_mode == ("ego", 0)
    assert vehicle.max_speed == ("ego", 20.0)
    assert vehicle.commanded_speed == ("ego", 20.0)
    assert effective == 20.0


def test_speed20_overlay_and_artifact_names_are_isolated(tmp_path: Path) -> None:
    for environment_class in (
        HighDensitySingleSeed100EpisodeSpeed20EnvV3,
        HighDensitySingleSeed100EpisodeSpeed20EnvV4V3,
    ):
        environment = environment_class.__new__(environment_class)
        environment._connection = None
        environment._high_density_overlay_root = tmp_path
        environment._high_density_vehicle_scale = 1.5
        environment.scenario = "cross"
        path = environment._overlay_path(Path("traffic_261.rou.xml"))
        assert path.parent == tmp_path / "cross"
        assert path.name.endswith("_ss100_s20_v3.rou.xml")
        assert "_ss100_v2.rou.xml" not in path.name

    protocol = load_protocol(speed20_runner.DEFAULT_PROTOCOL_PATH)
    assert all(
        setting["scenario_id"].endswith("speed20_v3")
        for setting in protocol["density"].values()
    )
    assert protocol["artifact_contract"]["result_root"] == (
        "results_hd_ss100_s20_v3"
    )
    assert speed20_trainer.OVERLAY_DIRECTORY_NAME == (
        "overlays_hd_ss100_s20_v3"
    )
    assert speed20_runner.DEFAULT_RESULT_ROOT.name == "results_hd_ss100_s20_v3"
    assert SPEED20_V3_EXPERIMENT_ID.endswith("speed20-v3")


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


def test_speed20_factory_preserves_v2_partition_mapping(tmp_path: Path) -> None:
    train_factory = speed20_trainer._make_speed20_v3_environment_factory(
        adapter="v4_8",
        density={
            "vehicle_scale": 1.5,
            "pedestrian_scale": 1.0,
            "clone_depart_jitter_seconds": [8.0, 10.0],
        },
        overlay_root=tmp_path,
        v4_environment_class=_CapturedEnvironment,
    )
    calibration = train_factory(
        _factory_args(evaluation_split="train"), evaluation=True
    )
    assert calibration.kwargs["high_density_partition"] == "train"
    assert calibration.kwargs["high_density_contract_partition"] == "train"

    validation = train_factory(_factory_args(), evaluation=True)
    assert validation.kwargs["high_density_partition"] == "evaluation"
    assert validation.kwargs["high_density_contract_partition"] == "validation"


def test_speed20_wrappers_inject_only_new_defaults() -> None:
    run_args = speed20_runner._with_speed20_v3_defaults(
        ["plan", "--profile", "comparison"]
    )
    assert run_args[run_args.index("--protocol") + 1] == str(
        speed20_runner.DEFAULT_PROTOCOL_PATH
    )
    assert run_args[run_args.index("--result-root") + 1] == str(
        speed20_runner.DEFAULT_RESULT_ROOT
    )

    train_args = speed20_trainer._with_speed20_v3_protocol(
        ["--method", "mst_slt", "--scenario", "carla"]
    )
    assert train_args[train_args.index("--protocol") + 1] == str(
        speed20_trainer.DEFAULT_PROTOCOL_PATH
    )
