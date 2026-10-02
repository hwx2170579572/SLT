from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from envs.sumo.random_intersection import get_random_intersection_config
from envs.sumo.sumo_env import SumoSceneEnv


def _load_reward_wrapper():
    path = Path(__file__).resolve().parents[1] / "fast-developer" / "reward_shaping_v2.py"
    spec = importlib.util.spec_from_file_location("reward_shaping_v2_collision_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneralizedRewardShapingWrapper


def _command_env(traffic_config: dict) -> PaperSumoSceneEnv:
    """Build a command-only env object; this never starts SUMO."""
    template = get_paper_scenario_spec("intersection_random_medium_p03_v1")
    env = object.__new__(PaperSumoSceneEnv)
    env._paper_specification = SimpleNamespace(
        network_path=template.network_path,
        ego_route_path=template.ego_route_path,
        traffic_paths=template.traffic_paths,
    )
    env._random_traffic_config = deepcopy(traffic_config)
    # The command policy is independent of which traffic file is used. Avoid
    # materializing an episode schedule in this command-only test.
    env._random_traffic_config["dynamic_episode_traffic"] = False
    env._traffic_episode_index = 0
    env._selected_traffic_path = None
    env._selected_traffic_seed = None
    env._random_episode_schedule = None
    env._sumo_binary = "sumo"
    env._extra_sumo_args = []
    env._connection = None
    env._behavior_diagnostics = None
    env._episode_done = True
    return env


def _sumo_options(command: list[str]) -> dict[str, str]:
    return dict(zip(command[1::2], command[2::2]))


def test_collision_action_is_profile_scoped() -> None:
    old_config = get_random_intersection_config("intersection_random_medium_p03_v1")
    new_config = get_random_intersection_config("intersection_random_darrl_low_v1")
    old_env = _command_env(old_config)
    new_env = _command_env(new_config)
    assert _sumo_options(old_env._sumo_command(0))["--collision.action"] == "none"
    assert _sumo_options(new_env._sumo_command(0))["--collision.action"] == "remove"


def _events_env(
    *,
    active_vehicles: tuple[str, ...],
    arrived: tuple[str, ...] = (),
    colliding_ids: tuple[str, ...] = (),
    collision_events: tuple[tuple[str, str], ...] = (),
) -> SumoSceneEnv:
    env = object.__new__(SumoSceneEnv)
    env.specification = SimpleNamespace(ego_id="ego", max_episode_steps=600)
    env._connection = SimpleNamespace(
        simulation=SimpleNamespace(
            getArrivedIDList=lambda: list(arrived),
            getCollidingVehiclesIDList=lambda: list(colliding_ids),
            getCollisions=lambda: [
                SimpleNamespace(collider=collider, victim=victim)
                for collider, victim in collision_events
            ],
            getStartingTeleportIDList=lambda: [],
        ),
        vehicle=SimpleNamespace(getIDList=lambda: list(active_vehicles)),
    )
    env._behavior_diagnostics = None
    env._episode_done = True
    env._raw_steps = 1
    # Isolate SUMO event semantics from the separate geometric fallback.
    env._geometric_collision = lambda: False
    return env


def test_removed_ego_with_collision_event_is_collision_not_off_route() -> None:
    env = _events_env(
        active_vehicles=("background",),
        collision_events=(("ego", "background"),),
    )
    success, collision, off_route, timeout = env._events_after_step()
    assert (success, collision, off_route, timeout) == (False, True, False, False)


def test_arrived_ego_is_success_after_sumo_removes_it() -> None:
    env = _events_env(active_vehicles=("background",), arrived=("ego",))
    success, collision, off_route, timeout = env._events_after_step()
    assert (success, collision, off_route, timeout) == (True, False, False, False)


def test_background_only_collision_does_not_flag_ego_terminal_event() -> None:
    env = _events_env(
        active_vehicles=("ego", "background_a", "background_b"),
        colliding_ids=("background_a", "background_b"),
        collision_events=(("background_a", "background_b"),),
    )
    success, collision, off_route, timeout = env._events_after_step()
    assert (success, collision, off_route, timeout) == (False, False, False, False)


def test_terminal_resolver_is_mutually_exclusive_for_every_flag_combination() -> None:
    for success in (False, True):
        for collision in (False, True):
            for off_route in (False, True):
                for timeout in (False, True):
                    if collision:
                        expected = (False, True, False, False)
                    elif off_route:
                        expected = (False, False, True, False)
                    elif success:
                        expected = (True, False, False, False)
                    elif timeout:
                        expected = (False, False, False, True)
                    else:
                        expected = (False, False, False, False)
                    assert SumoSceneEnv._resolve_terminal_events(
                        success, collision, off_route, timeout
                    ) == expected


def _step_stub_with_raw_collision_and_arrival() -> SumoSceneEnv:
    """Create a one-step offline env whose scenario override reports conflicts."""
    from gymnasium import spaces

    env = object.__new__(SumoSceneEnv)
    env.specification = SimpleNamespace(
        ego_id="ego", source_observation_contract="smarts", max_episode_steps=600
    )
    env.scenario = "mock_collision_arrival"
    env.action_repeat = 1
    env.reward_discount = 0.99
    env._connection = SimpleNamespace(
        simulationStep=lambda: None,
        vehicle=SimpleNamespace(
            getIDList=lambda: ["ego"], getDistance=lambda _ego_id: 0.0
        ),
        person=SimpleNamespace(getIDList=lambda: []),
    )
    env.observation_space = spaces.Dict(
        {"ego": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)}
    )
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    env._episode_done = False
    env._raw_steps = 599
    env._lifetime_raw_steps = 0
    env._lifetime_raw_step_budget = None
    env._decision_steps = 0
    env._last_events = (False, False, False, False)
    env._last_geometric_collision = False
    env._last_raw_sumo_arrived = True
    env._last_raw_sumo_collision = True
    env._last_observation = None
    env._behavior_diagnostics = None
    # Match SumoSceneEnv.__init__ for this __new__-constructed, offline step fixture.
    env._behavior_lane_apply_diagnostics = {}
    env._pre_simulation_ego_state = None
    env._last_effective_target_speed = 0.0
    env._last_curve_speed_limit = math.inf
    env.ego_control_profile = "mock"
    env.include_state_lstm = False
    env._apply_control = lambda _target_speed, _lane_command: False
    env._apply_speed_control = lambda _target_speed: None
    env._state = lambda _actor_key: np.zeros(5, dtype=np.float32)
    env._after_simulation_step = lambda: None
    env._record_histories = lambda: None
    # This represents the tuple returned by a scenario-specific override
    # (including PaperSumoSceneEnv's CARLA success-box override): collision,
    # success, and the raw time cap all coincide before central resolution.
    env._events_after_step = lambda: (True, True, False, True)
    env._make_observation = lambda: {"ego": np.zeros(1, dtype=np.float32)}
    return env


def test_step_resolves_override_flags_before_wrapper_rewards_and_logs_raw_evidence() -> None:
    wrapper_cls = _load_reward_wrapper()
    env = _step_stub_with_raw_collision_and_arrival()
    wrapped = wrapper_cls(env)

    _, shaped_reward, terminated, truncated, info = wrapped.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert info["collision"] is True
    assert info["off_route"] is False
    assert info["max_time"] is False
    assert info["raw_sumo_arrived"] is True
    assert info["raw_sumo_collision"] is True
    assert info["raw_max_time"] is True
    assert info["terminal_outcome_protocol"] == "exclusive_terminal_v2"
    assert info["reward_collision"] == -10.0
    assert info["reward_success"] == 0.0
    assert info["reward_timeout"] == 0.0
    assert shaped_reward == -10.01
