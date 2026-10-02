from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from tools.paper_evaluation_contract import (
    validate_environment_asset_provenance,
    validate_model_environment_spaces,
)


class _SpaceOnlyEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, observation_shape: tuple[int, ...]) -> None:
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=observation_shape, dtype=np.float32
        )


def test_model_environment_space_contract_accepts_exact_match() -> None:
    env = _SpaceOnlyEnv((80, 80, 3))
    model = SimpleNamespace(
        observation_space=gym.spaces.Box(
            0.0, 1.0, shape=(80, 80, 3), dtype=np.float32
        ),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    contract = validate_model_environment_spaces(model, env)
    assert contract["model_environment_spaces_match"] is True
    assert (
        contract["model_observation_space"]
        == contract["environment_observation_space"]
    )


def test_model_environment_space_contract_rejects_dict_box_mismatch() -> None:
    env = _SpaceOnlyEnv((80, 80, 3))
    model = SimpleNamespace(
        observation_space=gym.spaces.Dict(
            {
                "trajectory": gym.spaces.Box(
                    -1.0, 1.0, shape=(6, 10, 5), dtype=np.float32
                )
            }
        ),
        action_space=env.action_space,
    )
    with pytest.raises(ValueError, match="Observation spaces do not match"):
        validate_model_environment_spaces(model, env)


def test_released_sumo_asset_contract_remains_strict() -> None:
    env = SimpleNamespace(
        uses_released_assets=True,
        _paper_specification=get_paper_scenario_spec("left_turn"),
    )
    provenance = validate_environment_asset_provenance(env, "left_turn")
    assert provenance["uses_released_assets"] is True
    assert provenance["scenario_asset_source"] == "authors_release_v1.0.0"
    assert provenance["scenario_evidence_class"] == "released_sumo_source_scenario"


def test_carla_asset_contract_accepts_only_explicit_reconstruction() -> None:
    env = SimpleNamespace(
        uses_released_assets=False,
        _paper_specification=get_paper_scenario_spec("carla"),
    )
    provenance = validate_environment_asset_provenance(env, "carla")
    assert provenance == {
        "uses_released_assets": False,
        "scenario_asset_source": "carla_source_waypoint_reconstruction",
        "source_observation_contract": "carla",
        "scenario_evidence_class": "controlled_extension_not_reported_in_paper",
    }

    env.uses_released_assets = True
    with pytest.raises(ValueError, match="uses_released_assets=True"):
        validate_environment_asset_provenance(env, "carla")
