from __future__ import annotations

import math

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.sumo.legacy import LegacySumoAdapter
from envs.sumo.scenario_registry import (
    base_runnable_scenarios,
    get_scenario_spec,
)
from envs.sumo.sumo_env import (
    SumoSceneEnv,
    _oriented_boxes_overlap,
    _smarts_curve_speed_limit_from_headings,
    _source_state_lstm_actor_state,
)


def test_source_contact_geometry_uses_oriented_boxes() -> None:
    origin = np.zeros(2, dtype=np.float64)
    assert _oriented_boxes_overlap(
        origin, 0.0, 3.68, 1.47,
        np.array([0.0, 1.0]), 90.0, 3.68, 1.47,
        leeway=0.05,
    )
    assert not _oriented_boxes_overlap(
        origin, 0.0, 3.68, 1.47,
        np.array([5.0, 5.0]), 90.0, 3.68, 1.47,
        leeway=0.05,
    )


def test_action_contract() -> None:
    assert SumoSceneEnv.adapt_action(np.array([-1.0, -1.0])) == (0.0, -1)
    assert SumoSceneEnv.adapt_action(np.array([1.0, 1.0])) == (10.0, 1)
    assert SumoSceneEnv.adapt_action(np.array([0.0, 0.0])) == (5.0, 0)


def test_smarts_curve_speed_limit_branches() -> None:
    straight = np.zeros(16, dtype=np.float32)
    moderate = np.arange(16, dtype=np.float32) * np.deg2rad(5.0)
    sharp = np.arange(16, dtype=np.float32) * np.deg2rad(20.0)
    assert math.isinf(_smarts_curve_speed_limit_from_headings(straight))
    assert _smarts_curve_speed_limit_from_headings(moderate) == pytest.approx(6.94)
    assert _smarts_curve_speed_limit_from_headings(sharp) == pytest.approx(5.56)


def test_source_state_lstm_uses_distinct_ego_and_social_fields() -> None:
    ego = np.asarray([10.0, 20.0, 0.25, 3.0, 4.0], dtype=np.float32)
    social = np.asarray([13.0, 24.0, -0.5, 0.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(
        _source_state_lstm_actor_state(ego, ego, is_ego=True),
        [10.0, 20.0, 5.0, 0.0, 0.25],
    )
    np.testing.assert_allclose(
        _source_state_lstm_actor_state(social, ego, is_ego=False),
        [13.0, 24.0, 2.0, 5.0, -0.5],
    )


@pytest.mark.parametrize("scenario", base_runnable_scenarios())
def test_scenario_files_and_single_transition(scenario: str) -> None:
    spec = get_scenario_spec(scenario)
    assert spec.config_path.is_file()
    assert spec.network_path.is_file()
    env = SumoSceneEnv(scenario=scenario)
    try:
        observation, info = env.reset(seed=17)
        assert env.observation_space.contains(observation)
        assert observation["trajectory"].shape == (6, 10, 5)
        expected_map_dim = 2 if scenario == "carla" else 5
        expected_paths = 18 if scenario == "carla" else 12
        assert observation["map"].shape == (expected_paths, 10, expected_map_dim)
        assert np.count_nonzero(observation["trajectory"][0]) > 0
        assert np.count_nonzero(observation["map"][: spec.map_paths_per_actor]) > 0
        next_observation, reward, terminated, truncated, next_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert "undiscounted_reward" in next_info
        assert not (terminated and truncated)
        assert len(next_info["legacy_info"]) == 4
        assert info["scenario"] == scenario
    finally:
        env.close()


@pytest.mark.parametrize("scenario", base_runnable_scenarios())
def test_full_episode_reaches_a_declared_terminal_event(scenario: str) -> None:
    env = SumoSceneEnv(scenario=scenario)
    try:
        observation, reset_info = env.reset(seed=31)
        action = np.array([1.0, 0.0], dtype=np.float32)
        max_decisions = int(np.ceil(env.max_episode_steps / env.action_repeat)) + 2
        saw_neighbor = bool(np.count_nonzero(observation["trajectory"][1:]))
        saw_person = reset_info["active_persons"] > 0
        for _ in range(max_decisions):
            observation, _, terminated, truncated, info = env.step(action)
            saw_neighbor = saw_neighbor or bool(
                np.count_nonzero(observation["trajectory"][1:])
            )
            saw_person = saw_person or info["active_persons"] > 0
            if terminated or truncated:
                break
        else:
            pytest.fail(f"{scenario} did not terminate within its declared step limit")
        assert any(info["legacy_info"])
        assert terminated != truncated
        assert info["raw_simulation_steps"] <= env.max_episode_steps
        assert saw_neighbor, f"{scenario} never exposed background traffic"
        if scenario == "carla":
            assert saw_person, "CARLA-equivalent SUMO scenario never spawned pedestrians"
    finally:
        env.close()


def test_gymnasium_checker() -> None:
    env = SumoSceneEnv(scenario="left_turn")
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def test_cross_starts_both_merge_streams() -> None:
    env = SumoSceneEnv(scenario="cross", action_repeat=1)
    try:
        env.reset(seed=7)
        active_ids = set(env._connection.vehicle.getIDList())
        assert any(actor.startswith("first_merge_flow.") for actor in active_ids)
        assert any(actor.startswith("second_merge_flow.") for actor in active_ids)
        ramp_actor = next(
            actor for actor in active_ids if actor.startswith("first_merge_flow.")
        )
        ramp_paths = env._actor_paths(f"vehicle:{ramp_actor}", is_ego=False)
        assert np.count_nonzero(ramp_paths[0]) > 0
        assert np.array_equal(ramp_paths[0], ramp_paths[1])
    finally:
        env.close()


def test_legacy_adapter_contract() -> None:
    adapter = LegacySumoAdapter(scenario="left_turn")
    try:
        trajectory, ego, map_state = adapter.reset(seed=23)
        assert trajectory.shape == (6, 10, 5)
        assert ego.shape == (5,)
        assert map_state.shape == (12, 10, 5)
        observation, reward, done, info = adapter.step(np.zeros(2, dtype=np.float32))
        assert observation[0].shape == trajectory.shape
        assert np.isfinite(reward)
        assert isinstance(done, bool)
        assert len(info) == 4
    finally:
        adapter.close()
