"""Gymnasium SUMO environment for Scene-Rep-Transformer.

The environment deliberately does not import SMARTS, CARLA, TensorFlow, or
TF2RL.  It reproduces the released vector/map observation contract directly
from TraCI and exposes the same two-dimensional continuous action:

``action[0]``
    Target speed, linearly mapped from ``[-1, 1]`` to ``[0, 10]`` m/s.
``action[1]``
    Lateral command. The released SMARTS controller interprets ``+1`` as left
    and ``-1`` as right; the released CARLA adapter uses the opposite sign.
    The scenario observation contract selects the corresponding convention.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import sys
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario_registry import SumoScenarioSpec, get_scenario_spec


_INSTANCE_COUNTER = itertools.count()


# SMARTS v0.4.17 converts SUMO vehicle classes to these Bullet chassis
# dimensions before collision sensing and observation publication.  Keep the
# table local so the SUMO runtime never imports SMARTS.
_SMARTS_VEHICLE_DIMENSIONS: dict[str, tuple[float, float]] = {
    "passenger": (3.68, 1.47),
    "bus": (7.0, 2.25),
    "coach": (8.0, 2.4),
    "truck": (5.0, 1.91),
    "trailer": (10.0, 2.5),
    "motorcycle": (2.5, 1.0),
    "pedestrian": (0.5, 0.5),
}

EGO_CONTROL_PROFILES = ("direct", "smarts_ackermann_proxy")


def _smarts_curve_speed_limit_from_headings(headings: np.ndarray) -> float:
    """Reproduce the curve-speed branches in SMARTS v0.4.17.

    ``LaneFollowingController`` computes a reversed EWMA of the absolute
    relative heading between roughly one-metre waypoints.  It leaves the
    requested speed untouched on straight roads, caps it at 6.94 m/s on
    moderately curved roads, and at 5.56 m/s on strongly curved roads.
    Returning infinity for the straight branch lets callers apply the result
    with a simple ``min(requested_speed, limit)``.
    """

    values = np.asarray(headings, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.inf
    relative = np.diff(values)
    relative = (relative + math.pi) % (2.0 * math.pi) - math.pi
    ewma_road_curviness = 0.0
    for delta in reversed(relative):
        degrees = math.degrees(abs(float(delta)))
        ewma_road_curviness += 0.03 * (degrees - ewma_road_curviness)
    road_curviness = float(np.clip(ewma_road_curviness / 2.5, 0.0, 1.0))
    if road_curviness < 0.3:
        return math.inf
    if road_curviness < 0.8:
        return 6.94
    return 5.56


def _forward_vector(sumo_angle_degrees: float) -> np.ndarray:
    """Return SUMO's clockwise-from-north forward unit vector."""

    angle = math.radians(float(sumo_angle_degrees))
    return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float64)


def _front_bumper_to_center(
    position: tuple[float, float] | list[float] | np.ndarray,
    sumo_angle_degrees: float,
    length: float,
) -> np.ndarray:
    """Match SMARTS ``Pose.from_front_bumper`` for a TraCI position."""

    return np.asarray(position, dtype=np.float64) - 0.5 * float(length) * _forward_vector(
        sumo_angle_degrees
    )


def _oriented_boxes_overlap(
    center_a: np.ndarray,
    angle_a_degrees: float,
    length_a: float,
    width_a: float,
    center_b: np.ndarray,
    angle_b_degrees: float,
    length_b: float,
    width_b: float,
    *,
    leeway: float = 0.0,
) -> bool:
    """Two-dimensional separating-axis test for vehicle contact."""

    forward_a = _forward_vector(angle_a_degrees)
    side_a = np.asarray((forward_a[1], -forward_a[0]), dtype=np.float64)
    forward_b = _forward_vector(angle_b_degrees)
    side_b = np.asarray((forward_b[1], -forward_b[0]), dtype=np.float64)
    delta = np.asarray(center_b, dtype=np.float64) - np.asarray(
        center_a, dtype=np.float64
    )
    half_a = (0.5 * float(length_a), 0.5 * float(width_a))
    half_b = (0.5 * float(length_b), 0.5 * float(width_b))
    margin = max(0.0, float(leeway))
    for axis in (forward_a, side_a, forward_b, side_b):
        radius_a = half_a[0] * abs(float(np.dot(forward_a, axis))) + half_a[
            1
        ] * abs(float(np.dot(side_a, axis)))
        radius_b = half_b[0] * abs(float(np.dot(forward_b, axis))) + half_b[
            1
        ] * abs(float(np.dot(side_b, axis)))
        if abs(float(np.dot(delta, axis))) > radius_a + radius_b + margin:
            return False
    return True


def _known_sumo_root() -> Path:
    return Path(r"D:\Program Files (x86)\Eclipse\Sumo")


def resolve_sumo_binary(gui: bool = False, explicit: str | os.PathLike[str] | None = None) -> str:
    """Resolve a SUMO executable without requiring a global ``SUMO_HOME``."""

    executable = "sumo-gui" if gui else "sumo"
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SUMO executable does not exist: {path}")
        return str(path)
    found = shutil.which(executable)
    if found:
        return found
    suffix = ".exe" if os.name == "nt" else ""
    candidates: list[Path] = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]) / "bin" / f"{executable}{suffix}")
    candidates.append(_known_sumo_root() / "bin" / f"{executable}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find {executable}. Set SUMO_HOME, add SUMO/bin to PATH, "
        "or pass sumo_binary explicitly."
    )


def _import_traci(sumo_binary: str):
    """Import TraCI, adding only SUMO's own tools directory when necessary."""

    try:
        import traci  # type: ignore

        return traci
    except ImportError as first_error:
        candidates: list[Path] = []
        if os.environ.get("SUMO_HOME"):
            candidates.append(Path(os.environ["SUMO_HOME"]) / "tools")
        binary_path = Path(sumo_binary).resolve()
        candidates.append(binary_path.parent.parent / "tools")
        candidates.append(_known_sumo_root() / "tools")
        for tools_dir in candidates:
            if tools_dir.is_dir() and str(tools_dir) not in sys.path:
                sys.path.append(str(tools_dir))
                try:
                    import traci  # type: ignore

                    return traci
                except ImportError:
                    continue
        raise ImportError(
            "TraCI is unavailable. Use the Python tools shipped with the installed SUMO distribution."
        ) from first_error


def _math_heading(sumo_angle_degrees: float) -> float:
    """Convert SUMO's clockwise-from-north angle to radians from +x."""

    heading = math.radians(90.0 - float(sumo_angle_degrees))
    return float((heading + math.pi) % (2.0 * math.pi) - math.pi)


def _north_zero_heading(sumo_angle_degrees: float) -> float:
    """Match SMARTS Heading.from_sumo: north=0, counter-clockwise positive."""

    heading = -math.radians(float(sumo_angle_degrees))
    return float((heading + math.pi) % (2.0 * math.pi) - math.pi)


def _source_state_lstm_actor_state(
    state: np.ndarray,
    ego_state: np.ndarray,
    *,
    is_ego: bool,
) -> np.ndarray:
    """Build one actor row from the released ``STATE_LSTM`` adapter.

    The paper calls this an LSTM baseline, but the release uses a GRU over
    rows with different field semantics from the Proposed trajectory tensor.
    """

    actor = np.asarray(state, dtype=np.float32).reshape(5)
    ego = np.asarray(ego_state, dtype=np.float32).reshape(5)
    speed = float(np.linalg.norm(actor[3:5]))
    fourth = 0.0 if is_ego else float(np.linalg.norm(actor[:2] - ego[:2]))
    return np.asarray(
        [actor[0], actor[1], speed, fourth, actor[2]], dtype=np.float32
    )


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in observation.items()}


class SumoSceneEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Scene-Rep-compatible single-agent SUMO environment."""

    metadata = {"render_modes": [None, "human"], "render_fps": 10}

    def __init__(
        self,
        scenario: str = "left_turn",
        *,
        history_steps: int = 10,
        neighbors: int = 5,
        state_dim: int = 5,
        path_length: int = 10,
        path_spacing: float = 5.0,
        neighbor_radius: float = 80.0,
        action_repeat: int = 3,
        reward_discount: float = 0.99,
        ego_control_profile: str = "direct",
        include_state_lstm: bool = False,
        state_lstm_only: bool = False,
        render_mode: str | None = None,
        sumo_binary: str | os.PathLike[str] | None = None,
        sumo_args: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        if history_steps <= 0 or neighbors < 0 or path_length <= 0:
            raise ValueError("history_steps/path_length must be positive and neighbors non-negative")
        if state_dim != 5:
            raise ValueError("The released Scene-Rep trajectory contract requires state_dim=5")
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if ego_control_profile not in EGO_CONTROL_PROFILES:
            raise ValueError(
                f"ego_control_profile must be one of {EGO_CONTROL_PROFILES}, "
                f"got {ego_control_profile!r}"
            )
        if state_lstm_only and not include_state_lstm:
            raise ValueError("state_lstm_only requires include_state_lstm=True")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.specification: SumoScenarioSpec = get_scenario_spec(scenario)
        self.scenario = scenario
        self.history_steps = int(history_steps)
        self.neighbors = int(neighbors)
        self.state_dim = int(state_dim)
        self.path_length = int(path_length)
        self.path_spacing = float(path_spacing)
        self.neighbor_radius = float(neighbor_radius)
        self.action_repeat = int(action_repeat)
        self.reward_discount = float(reward_discount)
        self.ego_control_profile = str(ego_control_profile)
        self.include_state_lstm = bool(include_state_lstm)
        self.state_lstm_only = bool(state_lstm_only)
        self.render_mode = render_mode
        self._sumo_binary = resolve_sumo_binary(render_mode == "human", sumo_binary)
        self._traci = _import_traci(self._sumo_binary)
        self._extra_sumo_args = list(sumo_args or ())

        actor_count = self.neighbors + 1
        path_count = actor_count * self.specification.map_paths_per_actor
        map_dim = self.specification.map_feature_dim
        observation_spaces: dict[str, spaces.Space] = {}
        if not self.state_lstm_only:
            observation_spaces.update(
                {
                    "trajectory": spaces.Box(
                    low=-1000.0,
                    high=1000.0,
                    shape=(actor_count, self.history_steps, self.state_dim),
                    dtype=np.float32,
                    ),
                    "map": spaces.Box(
                    low=-1000.0,
                    high=1000.0,
                    shape=(path_count, self.path_length, map_dim),
                    dtype=np.float32,
                    ),
                }
            )
        if self.include_state_lstm:
            observation_spaces["state_lstm"] = spaces.Box(
                low=-1000.0,
                high=1000.0,
                shape=(self.history_steps, actor_count * self.state_dim),
                dtype=np.float32,
            )
            observation_spaces["state_lstm_mask"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.history_steps,),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._label = (
            f"scene_rep_{os.getpid()}_{next(_INSTANCE_COUNTER)}_{uuid.uuid4().hex[:8]}"
        )
        self._connection: Any | None = None
        self._histories: dict[str, deque[np.ndarray]] = {}
        self._state_lstm_histories: dict[str, deque[np.ndarray]] = {}
        self._state_lstm_last_seen_steps: dict[str, int] = {}
        self._history_timestep = 0
        self._first_seen_steps: dict[str, int] = {}
        self._last_seen_steps: dict[str, int] = {}
        self._driving_lane_cache: dict[str, tuple[int, ...]] = {}
        self._last_observation: dict[str, np.ndarray] | None = None
        self._raw_steps = 0
        self._lifetime_raw_steps = 0
        self._lifetime_raw_step_budget: int | None = None
        self._decision_steps = 0
        self._episode_done = True
        self._last_events = (False, False, False, False)
        self._last_geometric_collision = False
        self._pre_simulation_ego_state: np.ndarray | None = None
        self._actors_hidden_this_observation: set[str] = set()
        self._last_effective_target_speed = 0.0
        self._last_curve_speed_limit = math.inf

    @property
    def max_episode_steps(self) -> int:
        """Maximum number of raw 0.1-second simulator steps."""

        return self.specification.max_episode_steps

    def _sumo_command(self, seed: int) -> list[str]:
        command = [
            self._sumo_binary,
            "-c",
            str(self.specification.config_path),
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--quit-on-end",
            "true",
            "--time-to-teleport",
            "-1",
        ]
        command.extend(self._extra_sumo_args)
        return command

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.close()
        simulation_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**31 - 1)
        )
        self._traci.start(self._sumo_command(simulation_seed), label=self._label)
        self._connection = self._traci.getConnection(self._label)
        self._histories.clear()
        self._state_lstm_histories.clear()
        self._state_lstm_last_seen_steps.clear()
        self._history_timestep = 0
        self._first_seen_steps.clear()
        self._last_seen_steps.clear()
        self._driving_lane_cache.clear()
        self._last_observation = None
        self._raw_steps = 0
        self._decision_steps = 0
        self._episode_done = False
        self._last_events = (False, False, False, False)
        self._last_geometric_collision = False
        self._pre_simulation_ego_state = None
        self._actors_hidden_this_observation.clear()
        self._last_effective_target_speed = 0.0
        self._last_curve_speed_limit = math.inf

        # Vehicles are inserted on the first simulation step.  A short bounded
        # loop also handles a temporarily occupied departure lane.
        for _ in range(50):
            self._connection.simulationStep()
            self._after_simulation_step()
            self._raw_steps += 1
            if self.specification.ego_id in self._connection.vehicle.getIDList():
                break
        else:
            self.close()
            raise RuntimeError(
                f"Ego vehicle {self.specification.ego_id!r} was not inserted in scenario {self.scenario!r}"
            )

        ego_id = self.specification.ego_id
        self._configure_policy_controlled_ego(ego_id)
        # Episode limits count control-time steps after reset, not the insertion
        # step used to materialize the ego vehicle.
        self._raw_steps = 0
        self._record_histories()
        observation = self._make_observation()
        self._last_observation = _copy_observation(observation)
        info = self._info_dict(simulation_seed=simulation_seed)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._connection is None or self._episode_done:
            raise RuntimeError("step() called before reset() or after the episode ended")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (2,) or not np.all(np.isfinite(action_array)):
            raise ValueError(f"Expected a finite action with shape (2,), got {action!r}")
        action_array = np.clip(action_array, -1.0, 1.0)
        target_speed, lane_command = self.adapt_action(action_array)
        lane_change_applied = self._apply_control(target_speed, lane_command)

        discounted_reward = 0.0
        undiscounted_reward = 0.0
        success = collision = off_route = max_time = False
        raw_steps_executed = 0
        for repeat_index in range(self.action_repeat):
            if (
                self._lifetime_raw_step_budget is not None
                and self._lifetime_raw_steps >= self._lifetime_raw_step_budget
            ):
                break
            if self.specification.ego_id in self._connection.vehicle.getIDList():
                self._apply_speed_control(target_speed)
            self._pre_simulation_ego_state = self._state(
                f"vehicle:{self.specification.ego_id}"
            )
            self._connection.simulationStep()
            self._after_simulation_step()
            self._raw_steps += 1
            self._lifetime_raw_steps += 1
            raw_steps_executed += 1
            self._record_histories()
            success, collision, off_route, max_time = self._events_after_step()
            raw_reward = float(success) - float(collision)
            discounted_reward += (self.reward_discount**repeat_index) * raw_reward
            undiscounted_reward += raw_reward
            if success or collision or off_route or max_time:
                break

        self._decision_steps += 1
        # The released SMARTS runner explicitly rewrites a max-step ``done``
        # to False before storing it, while the CARLA runner stores the native
        # carla_env.py max-time ``done`` unchanged.  Gymnasium's split lets us
        # preserve both bootstrap contracts without modifying SB3.
        source_contract = getattr(
            self.specification, "source_observation_contract", ""
        )
        max_time_is_terminal = source_contract == "carla"
        terminated = bool(
            success or collision or off_route or (max_time and max_time_is_terminal)
        )
        truncated = bool(max_time and not terminated)
        self._episode_done = terminated or truncated
        self._last_events = (success, collision, off_route, max_time)

        if self.specification.ego_id in self._connection.vehicle.getIDList():
            observation = self._make_observation()
            self._last_observation = _copy_observation(observation)
        elif self._last_observation is not None:
            observation = _copy_observation(self._last_observation)
        else:
            observation = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
        info = self._info_dict(
            target_speed=target_speed,
            lane_command=lane_command,
            lane_change_applied=lane_change_applied,
            undiscounted_reward=float(undiscounted_reward),
            raw_steps_executed=raw_steps_executed,
            source_terminal_for_bootstrap=terminated,
        )
        return observation, float(discounted_reward), terminated, truncated, info

    @staticmethod
    def adapt_action(action: np.ndarray) -> tuple[float, int]:
        """Apply the exact scaling/discretization used by both legacy adapters."""

        speed = float(np.clip((float(action[0]) + 1.0) * 5.0, 0.0, 10.0))
        lateral = float(np.clip(action[1], -1.0, 1.0))
        if lateral < -1.0 / 3.0:
            lane = -1
        elif lateral > 1.0 / 3.0:
            lane = 1
        else:
            lane = 0
        return speed, lane

    def _apply_control(self, target_speed: float, lane_command: int) -> bool:
        ego_id = self.specification.ego_id
        if ego_id not in self._connection.vehicle.getIDList():
            return False
        if lane_command == 0:
            return False
        # SUMO lane indices increase to the left in right-hand traffic.
        # SMARTS' LaneFollowingController uses +1 for left, while carla_env.py
        # explicitly maps -1 to get_left_lane().
        sumo_offset = self._sumo_lane_offset(lane_command)
        try:
            road_id = str(self._connection.vehicle.getRoadID(ego_id))
            lane_index = int(self._connection.vehicle.getLaneIndex(ego_id))
            if road_id.startswith(":"):
                return False
            driving_lanes = self._driving_lanes(road_id)
            if lane_index not in driving_lanes:
                return False
            target_rank = driving_lanes.index(lane_index) + sumo_offset
            if target_rank < 0 or target_rank >= len(driving_lanes):
                return False
            self._connection.vehicle.changeLaneRelative(
                ego_id,
                sumo_offset,
                self.action_repeat * 0.1,
            )
            return True
        except self._traci.TraCIException:
            return False

    def _control_path_headings(self, lookahead: int = 16) -> np.ndarray:
        """Return a route-aligned heading lookahead for the dynamics proxy."""

        ego_id = self.specification.ego_id
        if ego_id not in self._connection.vehicle.getIDList() or lookahead < 2:
            return np.empty(0, dtype=np.float32)
        state = self._state(f"vehicle:{ego_id}")
        if state is None:
            return np.empty(0, dtype=np.float32)
        try:
            route = list(self._connection.vehicle.getRoute(ego_id))
            route_index = max(
                0, int(self._connection.vehicle.getRouteIndex(ego_id))
            )
            road_id = str(self._connection.vehicle.getRoadID(ego_id))
            lane_index = int(self._connection.vehicle.getLaneIndex(ego_id))
        except self._traci.TraCIException:
            return np.empty(0, dtype=np.float32)
        route_edges = route[route_index:]
        if road_id and not road_id.startswith(":") and road_id in route:
            route_edges = route[route.index(road_id) :]
        if not route_edges:
            return np.empty(0, dtype=np.float32)
        lanes = self._driving_lanes(route_edges[0])
        if not lanes:
            return np.empty(0, dtype=np.float32)
        if lane_index not in lanes:
            lane_index = min(lanes, key=lambda value: abs(value - lane_index))
        polyline = self._route_polyline(route_edges, lanes.index(lane_index))
        if polyline.shape[0] < 2:
            return np.empty(0, dtype=np.float32)

        segments = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        usable = lengths > 1e-6
        if not np.any(usable):
            return np.empty(0, dtype=np.float32)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        _, start_s, _ = self._projection_on_polyline(polyline, state[:2])
        targets = start_s + np.arange(lookahead, dtype=np.float64)
        valid = targets <= cumulative[-1] + 1e-5
        if np.count_nonzero(valid) < 2:
            return np.empty(0, dtype=np.float32)
        sampled = np.zeros((int(np.count_nonzero(valid)), 2), dtype=np.float64)
        sampled[:, 0] = np.interp(targets[valid], cumulative, polyline[:, 0])
        sampled[:, 1] = np.interp(targets[valid], cumulative, polyline[:, 1])
        deltas = np.diff(sampled, axis=0)
        # SMARTS headings use north=0 and counter-clockwise positive.
        headings = -np.arctan2(deltas[:, 0], deltas[:, 1])
        return np.concatenate([headings[:1], headings]).astype(np.float32)

    def _apply_speed_control(self, requested_speed: float) -> float:
        """Apply either direct TraCI speed or the SMARTS dynamics proxy."""

        ego_id = self.specification.ego_id
        requested = float(np.clip(requested_speed, 0.0, 10.0))
        if self.ego_control_profile == "direct":
            effective = requested
            curve_limit = math.inf
        else:
            headings = self._control_path_headings()
            curve_limit = _smarts_curve_speed_limit_from_headings(headings)
            desired = min(requested, curve_limit)
            current = max(0.0, float(self._connection.vehicle.getSpeed(ego_id)))
            # The released scenario vehicle defaults are accel=2.6 and
            # decel=4.5 m/s^2.  Bounding every 0.1 s tick is a transparent
            # SUMO proxy for the omitted Bullet/Ackermann longitudinal plant.
            effective = float(
                np.clip(desired, current - 4.5 * 0.1, current + 2.6 * 0.1)
            )
        self._connection.vehicle.setSpeed(ego_id, effective)
        self._last_effective_target_speed = effective
        self._last_curve_speed_limit = curve_limit
        return effective

    def _sumo_lane_offset(self, lane_command: int) -> int:
        contract = getattr(
            self.specification, "source_observation_contract", "carla"
        )
        return lane_command if contract == "smarts" else -lane_command

    def _configure_policy_controlled_ego(self, ego_id: str) -> None:
        """Apply the released external-provider vehicle contract."""

        # SMARTS v0.4.17 marks an externally controlled ego with
        # speedMode=0b00000 before synchronizing it through moveToXY. Keeping
        # SUMO's safe-speed or right-of-way checks would add an unrequested
        # rule controller. Its lateral pose is external as well, so explicit
        # lane commands must not be vetoed by SUMO safety checks.
        self._connection.vehicle.setSpeedMode(ego_id, 0)
        self._connection.vehicle.setLaneChangeMode(ego_id, 0)
        if (
            getattr(self.specification, "source_observation_contract", "")
            == "smarts"
        ):
            # SumoTrafficSimulation._create_vehicle() applies these exact
            # secure-gap and passenger chassis values to a non-SUMO vehicle.
            length, width = _SMARTS_VEHICLE_DIMENSIONS["passenger"]
            self._connection.vehicle.setTau(ego_id, 4.0)
            self._connection.vehicle.setDecel(ego_id, 6.0)
            self._connection.vehicle.setLength(ego_id, length)
            self._connection.vehicle.setWidth(ego_id, width)
            self._connection.vehicle.setHeight(ego_id, 1.4)

    def _events_after_step(self) -> tuple[bool, bool, bool, bool]:
        ego_id = self.specification.ego_id
        arrived = set(self._connection.simulation.getArrivedIDList())
        colliding = set(self._connection.simulation.getCollidingVehiclesIDList())
        collision_objects = self._connection.simulation.getCollisions()
        teleporting = set(self._connection.simulation.getStartingTeleportIDList())
        active = ego_id in self._connection.vehicle.getIDList()
        success = ego_id in arrived
        self._last_geometric_collision = self._geometric_collision()
        collision = ego_id in colliding or any(
            ego_id in (item.collider, item.victim) for item in collision_objects
        ) or self._last_geometric_collision
        off_route = ego_id in teleporting or (not active and not success and not collision)
        max_time = self._raw_steps >= self.max_episode_steps
        return bool(success), bool(collision), bool(off_route), bool(max_time)

    def _vehicle_dimensions(self, vehicle_id: str) -> tuple[float, float]:
        """Return source-provider length/width for one TraCI vehicle."""

        contract = getattr(
            self.specification, "source_observation_contract", "cartesian"
        )
        if contract == "smarts":
            try:
                vehicle_class = str(self._connection.vehicle.getVehicleClass(vehicle_id))
            except self._traci.TraCIException:
                vehicle_class = "passenger"
            return _SMARTS_VEHICLE_DIMENSIONS.get(
                vehicle_class, _SMARTS_VEHICLE_DIMENSIONS["passenger"]
            )
        return (
            float(self._connection.vehicle.getLength(vehicle_id)),
            float(self._connection.vehicle.getWidth(vehicle_id)),
        )

    def _vehicle_collision_box(
        self, vehicle_id: str
    ) -> tuple[np.ndarray, float, float, float] | None:
        try:
            front_position = self._connection.vehicle.getPosition(vehicle_id)
            angle = float(self._connection.vehicle.getAngle(vehicle_id))
            length, width = self._vehicle_dimensions(vehicle_id)
        except self._traci.TraCIException:
            return None
        center = _front_bumper_to_center(front_position, angle, length)
        return center, angle, length, width

    def _geometric_collision(self) -> bool:
        """Reproduce source physics-contact events absent from SUMO's list.

        SMARTS obtains agent collisions from Bullet chassis contact points and
        deliberately adds 0.05 m leeway.  SUMO may report no collision when a
        policy-controlled vehicle geometrically overlaps at a junction, so a
        source-dimension OBB test is required in the direct-TraCI migration.
        """

        ego_id = self.specification.ego_id
        if ego_id not in self._connection.vehicle.getIDList():
            return False
        ego_box = self._vehicle_collision_box(ego_id)
        if ego_box is None:
            return False
        for vehicle_id in self._connection.vehicle.getIDList():
            if (
                vehicle_id == ego_id
                or vehicle_id in self._actors_hidden_this_observation
            ):
                continue
            other_box = self._vehicle_collision_box(vehicle_id)
            if other_box is not None and _oriented_boxes_overlap(
                *ego_box, *other_box, leeway=0.05
            ):
                return True
        if self.specification.include_pedestrians:
            for person_id in self._connection.person.getIDList():
                try:
                    position = np.asarray(
                        self._connection.person.getPosition(person_id),
                        dtype=np.float64,
                    )
                    angle = float(self._connection.person.getAngle(person_id))
                except self._traci.TraCIException:
                    continue
                if _oriented_boxes_overlap(
                    *ego_box,
                    position,
                    angle,
                    0.5,
                    0.5,
                    leeway=0.05,
                ):
                    return True
        return False

    def _after_simulation_step(self) -> None:
        """Hook for source-simulator behavior that runs after each SUMO tick."""

        self._actors_hidden_this_observation.clear()

    def _state(self, actor_key: str) -> np.ndarray | None:
        domain, actor_id = actor_key.split(":", 1)
        try:
            if domain == "vehicle":
                position = self._connection.vehicle.getPosition(actor_id)
                speed = float(self._connection.vehicle.getSpeed(actor_id))
                sumo_angle = self._connection.vehicle.getAngle(actor_id)
                length, _ = self._vehicle_dimensions(actor_id)
                position = _front_bumper_to_center(position, sumo_angle, length)
            else:
                position = self._connection.person.getPosition(actor_id)
                speed = float(self._connection.person.getSpeed(actor_id))
                sumo_angle = self._connection.person.getAngle(actor_id)
        except self._traci.TraCIException:
            return None
        contract = getattr(
            self.specification, "source_observation_contract", "cartesian"
        )
        if contract == "carla":
            offset_x, offset_y = getattr(
                self.specification, "coordinate_offset", (0.0, 0.0)
            )
            position = (
                float(position[0]) - float(offset_x),
                float(position[1]) - float(offset_y),
            )
        if contract in ("smarts", "carla"):
            heading = _north_zero_heading(sumo_angle)
        else:
            heading = _math_heading(sumo_angle)
        if contract == "carla":
            # CARLA records world-frame velocity components while expressing
            # heading as yaw-90 degrees (north=0).
            velocity_x = -speed * math.sin(heading)
            velocity_y = speed * math.cos(heading)
        else:
            # The SMARTS adapter in the release computes these components
            # directly from its north-zero Heading. Preserve that source
            # behavior even though it is not a Cartesian direction vector.
            velocity_x = speed * math.cos(heading)
            velocity_y = speed * math.sin(heading)
        return np.asarray(
            [
                float(position[0]),
                float(position[1]),
                heading,
                velocity_x,
                velocity_y,
            ],
            dtype=np.float32,
        )

    def _active_actor_keys(self) -> list[str]:
        ego_id = self.specification.ego_id
        actors = [
            f"vehicle:{vehicle_id}"
            for vehicle_id in self._connection.vehicle.getIDList()
            if vehicle_id != ego_id
            and vehicle_id not in self._actors_hidden_this_observation
        ]
        if self.specification.include_pedestrians:
            actors.extend(f"person:{person_id}" for person_id in self._connection.person.getIDList())
        return actors

    def _record_histories(self) -> None:
        ego_id = self.specification.ego_id
        ego_key = f"vehicle:{ego_id}"
        ego_keys = (
            [ego_key]
            if ego_id in self._connection.vehicle.getIDList()
            else []
        )
        # An arrived SUMO vehicle is removed before TraCI returns from the
        # tick. Avoid issuing an invalid getPosition request for the terminal
        # ego; the last valid observation is retained as SB3's next_obs.
        keys = [*ego_keys, *self._active_actor_keys()]
        ego_state = self._state(ego_key) if ego_keys else None
        for actor_key in keys:
            state = self._state(actor_key)
            if state is None:
                continue
            self._append_history(actor_key, state)
            if self.include_state_lstm and ego_state is not None:
                self._append_state_lstm_history(
                    actor_key,
                    _source_state_lstm_actor_state(
                        state, ego_state, is_ego=actor_key == ego_key
                    ),
                )
        self._history_timestep += 1

    def _append_history(self, actor_key: str, state: np.ndarray) -> None:
        """Append one source-contract actor sample at the current timestep."""

        self._first_seen_steps.setdefault(actor_key, self._history_timestep)
        history = self._histories.setdefault(
            actor_key, deque(maxlen=self.history_steps)
        )
        previous_timestep = self._last_seen_steps.get(actor_key)
        if (
            getattr(self.specification, "source_observation_contract", "")
            == "smarts"
            and actor_key != f"vehicle:{self.specification.ego_id}"
            and previous_timestep is not None
            and self._history_timestep > previous_timestep + 1
            and history
        ):
            # NeighbourAgentBuffer.add() in the release linearly interpolates
            # every missing timestamp before appending a reappearing actor.
            # np.interp is component-wise linear, including the heading slot.
            previous_state = np.asarray(history[-1], dtype=np.float32)
            span = self._history_timestep - previous_timestep
            for offset in range(1, span):
                fraction = np.float32(offset / span)
                history.append(
                    previous_state + (np.asarray(state, dtype=np.float32) - previous_state) * fraction
                )
        history.append(np.asarray(state, dtype=np.float32))
        self._last_seen_steps[actor_key] = self._history_timestep

    def _append_state_lstm_history(
        self, actor_key: str, state: np.ndarray
    ) -> None:
        """Append the exact five STATE_LSTM fields, including distance.

        Distance is stored before interpolation because the released
        ``NeighbourAgentBuffer`` interpolates that scalar component directly;
        recomputing Euclidean distance from interpolated x/y would differ.
        """

        history = self._state_lstm_histories.setdefault(
            actor_key, deque(maxlen=self.history_steps)
        )
        previous_timestep = self._state_lstm_last_seen_steps.get(actor_key)
        if (
            getattr(self.specification, "source_observation_contract", "")
            == "smarts"
            and actor_key != f"vehicle:{self.specification.ego_id}"
            and previous_timestep is not None
            and self._history_timestep > previous_timestep + 1
            and history
        ):
            previous_state = np.asarray(history[-1], dtype=np.float32)
            current_state = np.asarray(state, dtype=np.float32)
            span = self._history_timestep - previous_timestep
            for offset in range(1, span):
                fraction = np.float32(offset / span)
                history.append(
                    previous_state
                    + (current_state - previous_state) * fraction
                )
        history.append(np.asarray(state, dtype=np.float32))
        self._state_lstm_last_seen_steps[actor_key] = self._history_timestep

    def _nearest_actor_keys(self, ego_state: np.ndarray) -> list[str]:
        if getattr(self.specification, "source_observation_contract", "") == "carla":
            # Reproduce select_top_actors() from carla_env.py. Its integer-key
            # dictionary lets walkers overwrite vehicles with the same list
            # index before the nearest five actors are selected.
            indexed: dict[int, tuple[float, str]] = {}
            ego_id = self.specification.ego_id
            vehicles = [
                vehicle_id
                for vehicle_id in self._connection.vehicle.getIDList()
                if vehicle_id != ego_id
            ]
            persons = list(self._connection.person.getIDList())
            for index, actor_id in enumerate(vehicles):
                actor_key = f"vehicle:{actor_id}"
                state = self._state(actor_key)
                if state is not None:
                    indexed[index] = (
                        float(np.linalg.norm(state[:2] - ego_state[:2])),
                        actor_key,
                    )
            for index, actor_id in enumerate(persons):
                actor_key = f"person:{actor_id}"
                state = self._state(actor_key)
                if state is not None:
                    indexed[index] = (
                        float(np.linalg.norm(state[:2] - ego_state[:2])),
                        actor_key,
                    )
            return [
                actor_key
                for _, actor_key in sorted(indexed.values(), key=lambda item: item[0])[
                    : self.neighbors
                ]
            ]

        distances: list[tuple[float, str]] = []
        for actor_key in self._active_actor_keys():
            state = self._state(actor_key)
            if state is None:
                continue
            distance = float(np.linalg.norm(state[:2] - ego_state[:2]))
            if distance <= self.neighbor_radius:
                distances.append((distance, actor_key))
        distances.sort(key=lambda item: (item[0], item[1]))
        return [actor_key for _, actor_key in distances[: self.neighbors]]

    def _history_array(self, actor_key: str) -> np.ndarray:
        output = np.zeros((self.history_steps, self.state_dim), dtype=np.float32)
        values = list(self._histories.get(actor_key, ()))
        if values:
            values = values[-self.history_steps :]
            values_array = np.asarray(values, dtype=np.float32)
            if getattr(
                self.specification, "source_observation_contract", ""
            ) == "carla":
                # query_single_trajs() writes current at index -0 (index 0),
                # then prior frames from the end of the tensor backwards.
                output[0] = values_array[-1]
                if len(values_array) > 1:
                    output[-(len(values_array) - 1) :] = values_array[:-1]
            elif (
                getattr(self.specification, "source_observation_contract", "")
                == "smarts"
                and actor_key != f"vehicle:{self.specification.ego_id}"
            ):
                # NeighbourAgentBuffer.pad_hist() left-pads an actor that was
                # first observed after the episode began, then the outer
                # adapter right-pads the still-short global history window.
                current_timestep = max(0, self._history_timestep - 1)
                pad_length = min(current_timestep + 1, self.history_steps)
                left_padding = max(0, pad_length - len(values_array))
                output[
                    left_padding : left_padding + len(values_array)
                ] = values_array
            else:
                output[: len(values_array)] = values_array
        return output

    def _state_lstm_history_array(self, actor_key: str) -> np.ndarray:
        output = np.zeros((self.history_steps, 5), dtype=np.float32)
        values = list(self._state_lstm_histories.get(actor_key, ()))
        if not values:
            return output
        values_array = np.asarray(values[-self.history_steps :], dtype=np.float32)
        if (
            getattr(self.specification, "source_observation_contract", "")
            == "smarts"
            and actor_key != f"vehicle:{self.specification.ego_id}"
        ):
            current_timestep = max(0, self._history_timestep - 1)
            pad_length = min(current_timestep + 1, self.history_steps)
            left_padding = max(0, pad_length - len(values_array))
            output[
                left_padding : left_padding + len(values_array)
            ] = values_array
        else:
            output[: len(values_array)] = values_array
        return output

    def _make_observation(self) -> dict[str, np.ndarray]:
        ego_key = f"vehicle:{self.specification.ego_id}"
        ego_state = self._state(ego_key)
        if ego_state is None:
            if self._last_observation is not None:
                return _copy_observation(self._last_observation)
            raise RuntimeError("Cannot construct an observation without an active ego vehicle")
        actor_keys = [ego_key, *self._nearest_actor_keys(ego_state)]

        observation: dict[str, np.ndarray] = {}
        if not self.state_lstm_only:
            trajectories = np.zeros(
                (self.neighbors + 1, self.history_steps, self.state_dim),
                dtype=np.float32,
            )
            for index, actor_key in enumerate(actor_keys):
                trajectories[index] = self._history_array(actor_key)

            paths_per_actor = self.specification.map_paths_per_actor
            map_state = np.zeros(
                (
                    (self.neighbors + 1) * paths_per_actor,
                    self.path_length,
                    self.specification.map_feature_dim,
                ),
                dtype=np.float32,
            )
            for actor_index, actor_key in enumerate(actor_keys):
                actor_paths = self._actor_paths(
                    actor_key, is_ego=actor_index == 0
                )
                start = actor_index * paths_per_actor
                map_state[start : start + paths_per_actor] = actor_paths
            observation.update({"trajectory": trajectories, "map": map_state})
        if self.include_state_lstm:
            state_lstm = np.zeros(
                (
                    self.history_steps,
                    (self.neighbors + 1) * self.state_dim,
                ),
                dtype=np.float32,
            )
            for actor_index, actor_key in enumerate(actor_keys):
                start = actor_index * self.state_dim
                state_lstm[:, start : start + self.state_dim] = (
                    self._state_lstm_history_array(actor_key)
                )
            state_lstm_mask = np.zeros(self.history_steps, dtype=np.float32)
            state_lstm_mask[: min(self._history_timestep, self.history_steps)] = 1.0
            observation["state_lstm"] = state_lstm
            observation["state_lstm_mask"] = state_lstm_mask
        return observation

    def _actor_paths(self, actor_key: str, is_ego: bool) -> np.ndarray:
        domain, actor_id = actor_key.split(":", 1)
        count = self.specification.map_paths_per_actor
        dim = self.specification.map_feature_dim
        output = np.zeros((count, self.path_length, dim), dtype=np.float32)
        state = self._state(actor_key)
        if state is None:
            return output
        if domain == "person":
            distances = self.path_spacing * np.arange(1, self.path_length + 1, dtype=np.float32)
            x = state[0] + np.cos(state[2]) * distances
            y = state[1] + np.sin(state[2]) * distances
            line = np.stack([x, y], axis=-1).astype(np.float32)
            output[:, :, :2] = line[None, :, :]
            return output

        try:
            route = list(self._connection.vehicle.getRoute(actor_id))
            route_index = max(0, int(self._connection.vehicle.getRouteIndex(actor_id)))
            road_id = str(self._connection.vehicle.getRoadID(actor_id))
            lane_index = int(self._connection.vehicle.getLaneIndex(actor_id))
        except self._traci.TraCIException:
            return output
        if not route:
            return output
        route_edges = route[route_index:]
        if road_id and not road_id.startswith(":") and road_id in route:
            route_edges = route[route.index(road_id) :]
        if not route_edges:
            return output

        current_driving = self._driving_lanes(route_edges[0])
        if not current_driving:
            return output
        if lane_index not in current_driving:
            lane_index = min(current_driving, key=lambda value: abs(value - lane_index))
        rank = current_driving.index(lane_index)
        contract = getattr(
            self.specification, "source_observation_contract", "cartesian"
        )
        if contract == "smarts":
            # The ego has a fixed mission, whereas sensors attached to social
            # vehicles are planned with mission=None (a random EndlessMission).
            # SMARTS v0.4.17 also contains a consequential route-filter bug:
            # fixed-route traversal rejects internal edges because their IDs
            # are absent from the route. Preserve that behavior rather than
            # silently repairing it with SUMO's valid turn curve.
            polylines = self._smarts_waypoint_polylines(
                route=list(route),
                road_id=road_id,
                position=state[:2],
                is_ego=is_ego,
                limit=count,
            )
            for path_index, polyline in enumerate(polylines[:count]):
                sampled, valid = self._sample_polyline(polyline, state[:2])
                if not np.any(valid):
                    continue
                output[path_index, :, :2] = sampled
                headings = self._polyline_headings(sampled, valid)
                output[path_index, :, 2] = headings
                output[path_index, valid, 3] = 1.0 if is_ego else 0.0
                output[path_index, valid, 4] = 0.0 if is_ego else 1.0
            if len(polylines) == 1:
                output[1] = output[0]
            return output

        if count == 3:
            # CARLA contract: left, current, right.
            ranks: list[int | None] = [
                rank + 1 if rank + 1 < len(current_driving) else None,
                rank,
                rank - 1 if rank - 1 >= 0 else None,
            ]
        else:
            alternative = rank + 1 if rank + 1 < len(current_driving) else rank - 1
            ranks = [rank, alternative if alternative >= 0 else None]

        for path_index, lane_rank in enumerate(ranks):
            if lane_rank is None:
                continue
            polyline = self._route_polyline(route_edges, lane_rank)
            sampled, valid = self._sample_polyline(polyline, state[:2])
            if not np.any(valid):
                continue
            output[path_index, :, :2] = sampled
            if dim == 5:
                headings = self._polyline_headings(sampled, valid)
                output[path_index, :, 2] = headings
                output[path_index, valid, 3] = 1.0 if is_ego else 0.0
                output[path_index, valid, 4] = 0.0 if is_ego else 1.0
        if count == 2 and np.any(output[0]) and not np.any(output[1]):
            # SMARTS' waypoint adapter duplicates its sole candidate so the
            # fixed two-path contract contains two valid, identical paths.
            output[1] = output[0]
        return output

    def _lane_shape(self, lane_id: str) -> np.ndarray:
        try:
            shape = np.asarray(
                self._connection.lane.getShape(lane_id), dtype=np.float32
            )
        except self._traci.TraCIException:
            return np.empty((0, 2), dtype=np.float32)
        if shape.ndim != 2 or shape.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float32)
        return shape

    @staticmethod
    def _join_polyline_pieces(pieces: Iterable[np.ndarray]) -> np.ndarray:
        joined: list[np.ndarray] = []
        for piece in pieces:
            if piece.shape[0] < 2:
                continue
            if joined and np.allclose(joined[-1][-1], piece[0]):
                piece = piece[1:]
            if piece.shape[0] > 0:
                joined.append(piece)
        if not joined:
            return np.empty((0, 2), dtype=np.float32)
        polyline = np.concatenate(joined, axis=0)
        if polyline.shape[0] <= 1:
            return polyline
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-5]
        )
        return polyline[keep]

    def _edge_lane_ids(self, edge_id: str) -> list[str]:
        try:
            count = int(self._connection.edge.getLaneNumber(edge_id))
        except self._traci.TraCIException:
            return []
        return [f"{edge_id}_{index}" for index in range(count)]

    def _lane_edge_id(self, lane_id: str) -> str:
        try:
            return str(self._connection.lane.getEdgeID(lane_id))
        except (self._traci.TraCIException, AttributeError):
            return lane_id.rsplit("_", 1)[0]

    @staticmethod
    def _projection_on_polyline(
        polyline: np.ndarray, position: np.ndarray
    ) -> tuple[float, float, float]:
        """Return (distance, arclength, total length) for the closest projection."""

        if polyline.shape[0] < 2:
            return math.inf, 0.0, 0.0
        segments = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        best_distance = math.inf
        best_s = 0.0
        for index, (start, segment, length) in enumerate(
            zip(polyline[:-1], segments, lengths)
        ):
            if length <= 1e-6:
                continue
            fraction = float(
                np.clip(
                    np.dot(position - start, segment) / (length * length), 0.0, 1.0
                )
            )
            projection = start + fraction * segment
            distance = float(np.linalg.norm(position - projection))
            if distance < best_distance:
                best_distance = distance
                best_s = float(cumulative[index] + fraction * length)
        return best_distance, best_s, float(cumulative[-1])

    def _closest_edge(self, edge_ids: Iterable[str], position: np.ndarray) -> str:
        candidates = list(dict.fromkeys(str(edge_id) for edge_id in edge_ids))
        if not candidates:
            return ""
        best_edge = candidates[0]
        best_distance = math.inf
        for edge_id in candidates:
            for lane_id in self._edge_lane_ids(edge_id):
                distance, _, _ = self._projection_on_polyline(
                    self._lane_shape(lane_id), position
                )
                if distance < best_distance:
                    best_distance = distance
                    best_edge = edge_id
        return best_edge

    def _next_lane_ids(self, lane_id: str) -> list[str]:
        try:
            links = self._connection.lane.getLinks(lane_id, extended=True)
        except self._traci.TraCIException:
            return []
        result: list[str] = []
        for link in links:
            target_lane = str(link[0])
            via_lane = str(link[4]) if link[4] else ""
            # LanePoints links the incoming lane to the via/internal lane
            # first. The target lane is immediate only without a via lane.
            next_lane = via_lane or target_lane
            if next_lane and next_lane not in result:
                result.append(next_lane)
        return result

    def _lane_graph_polylines(
        self,
        start_edge: str,
        position: np.ndarray,
        *,
        allowed_edges: set[str] | None,
        limit: int,
    ) -> list[np.ndarray]:
        """Reproduce LanePoints lookahead, branch order, and edge filtering."""

        if not start_edge or limit <= 0:
            return []
        horizon = float(self.path_spacing * self.path_length)
        results: list[np.ndarray] = []

        def continuations(
            lane_id: str, remaining: float, depth: int, branch_limit: int
        ) -> list[list[np.ndarray]]:
            if remaining <= 1e-5 or depth >= 16 or branch_limit <= 0:
                return [[]]
            eligible: list[str] = []
            for next_lane in self._next_lane_ids(lane_id):
                next_edge = self._lane_edge_id(next_lane)
                if allowed_edges is not None and next_edge not in allowed_edges:
                    continue
                eligible.append(next_lane)
            if not eligible:
                # When no branch survives, the source retains the shorter path
                # for every remaining lookahead iteration.
                return [[]]

            branches: list[list[np.ndarray]] = []
            for next_lane in eligible:
                next_shape = self._lane_shape(next_lane)
                if next_shape.shape[0] < 2:
                    continue
                _, _, next_length = self._projection_on_polyline(
                    next_shape, next_shape[0]
                )
                if remaining <= next_length + 1e-5:
                    suffixes = [[]]
                else:
                    suffixes = continuations(
                        next_lane,
                        remaining - next_length,
                        depth + 1,
                        branch_limit - len(branches),
                    )
                for suffix in suffixes:
                    branches.append([next_shape, *suffix])
                    if len(branches) >= branch_limit:
                        return branches
            return branches or [[]]

        # getLanes() returns ascending lane indexes; Python's source-side sort
        # by path[0].lane_index is stable, so this is the same final order.
        for starting_lane in self._edge_lane_ids(start_edge):
            shape = self._lane_shape(starting_lane)
            if shape.shape[0] < 2:
                continue
            _, start_s, total_length = self._projection_on_polyline(shape, position)
            distance_to_end = max(0.0, total_length - start_s)
            if horizon <= distance_to_end + 1e-5:
                branches = [[]]
            else:
                branches = continuations(
                    starting_lane,
                    horizon - distance_to_end,
                    0,
                    limit - len(results),
                )
            for suffix in branches:
                polyline = self._join_polyline_pieces([shape, *suffix])
                if polyline.shape[0] >= 2:
                    results.append(polyline)
                    if len(results) >= limit:
                        return results
        return results

    def _route_polylines(
        self,
        route_edges: list[str],
        *,
        point: np.ndarray | None = None,
        limit: int = 2,
    ) -> list[np.ndarray]:
        """Build fixed-mission SMARTS v0.4.17 waypoint paths.

        LanePoints checks the immediate next edge against non-internal route
        edge IDs. A valid SUMO turn via an internal edge is thus truncated.
        """

        if not route_edges or limit <= 0:
            return []
        if point is None:
            first_lanes = self._edge_lane_ids(route_edges[0])
            first_shape = (
                self._lane_shape(first_lanes[0])
                if first_lanes
                else np.empty((0, 2), dtype=np.float32)
            )
            point = (
                first_shape[0]
                if first_shape.shape[0]
                else np.zeros(2, dtype=np.float32)
            )
        start_edge = self._closest_edge(route_edges, point)
        return self._lane_graph_polylines(
            start_edge,
            point,
            allowed_edges=set(route_edges),
            limit=limit,
        )

    def _smarts_waypoint_polylines(
        self,
        *,
        route: list[str],
        road_id: str,
        position: np.ndarray,
        is_ego: bool,
        limit: int,
    ) -> list[np.ndarray]:
        if is_ego:
            return self._route_polylines(route, point=position, limit=limit)

        # Sensors attached to observed social vehicles use mission=None and
        # therefore an EndlessMission. On a normal edge paths are unconstrained.
        if road_id and not road_id.startswith(":"):
            return self._lane_graph_polylines(
                road_id, position, allowed_edges=None, limit=limit
            )

        # Inside a junction, EndlessMission constrains paths to the current
        # internal edge and its sole outgoing edge.
        candidate_edges = [road_id] if road_id else []
        for lane_id in self._edge_lane_ids(road_id):
            for next_lane in self._next_lane_ids(lane_id):
                edge_id = self._lane_edge_id(next_lane)
                if edge_id not in candidate_edges:
                    candidate_edges.append(edge_id)
        if not candidate_edges:
            candidate_edges = list(route)
        start_edge = self._closest_edge(candidate_edges, position)
        return self._lane_graph_polylines(
            start_edge,
            position,
            allowed_edges=set(candidate_edges),
            limit=limit,
        )

    def _driving_lanes(self, edge_id: str) -> tuple[int, ...]:
        if edge_id in self._driving_lane_cache:
            return self._driving_lane_cache[edge_id]
        result: list[int] = []
        try:
            lane_count = int(self._connection.edge.getLaneNumber(edge_id))
        except self._traci.TraCIException:
            return ()
        for index in range(lane_count):
            lane_id = f"{edge_id}_{index}"
            try:
                allowed = set(self._connection.lane.getAllowed(lane_id))
                disallowed = set(self._connection.lane.getDisallowed(lane_id))
            except self._traci.TraCIException:
                continue
            if allowed and "passenger" not in allowed:
                continue
            if "passenger" in disallowed:
                continue
            result.append(index)
        value = tuple(result)
        self._driving_lane_cache[edge_id] = value
        return value

    def _route_polyline(self, route_edges: list[str], lane_rank: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for edge_id in route_edges:
            if edge_id.startswith(":"):
                continue
            lanes = self._driving_lanes(edge_id)
            if not lanes:
                continue
            chosen_index = lanes[min(lane_rank, len(lanes) - 1)]
            lane_id = f"{edge_id}_{chosen_index}"
            try:
                shape = np.asarray(self._connection.lane.getShape(lane_id), dtype=np.float32)
            except self._traci.TraCIException:
                continue
            if shape.ndim != 2 or shape.shape[0] < 2:
                continue
            if pieces and np.allclose(pieces[-1][-1], shape[0]):
                shape = shape[1:]
            pieces.append(shape)
        if not pieces:
            return np.empty((0, 2), dtype=np.float32)
        polyline = np.concatenate(pieces, axis=0)
        if getattr(self.specification, "source_observation_contract", "") == "carla":
            offset = np.asarray(
                getattr(self.specification, "coordinate_offset", (0.0, 0.0)),
                dtype=np.float32,
            )
            polyline = polyline - offset[None, :]
        if polyline.shape[0] <= 1:
            return polyline
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-5]
        )
        return polyline[keep]

    def _sample_polyline(
        self, polyline: np.ndarray, position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        output = np.zeros((self.path_length, 2), dtype=np.float32)
        valid = np.zeros(self.path_length, dtype=bool)
        if polyline.shape[0] < 2:
            return output, valid
        segments = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        usable = lengths > 1e-6
        if not np.any(usable):
            return output, valid
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

        best_distance = math.inf
        start_s = 0.0
        for index, (start, segment, length) in enumerate(zip(polyline[:-1], segments, lengths)):
            if length <= 1e-6:
                continue
            fraction = float(np.clip(np.dot(position - start, segment) / (length * length), 0.0, 1.0))
            projection = start + fraction * segment
            distance = float(np.linalg.norm(position - projection))
            if distance < best_distance:
                best_distance = distance
                start_s = float(cumulative[index] + fraction * length)

        include_current = (
            getattr(self.specification, "source_observation_contract", "")
            == "carla"
        )
        start_index = 0 if include_current else 1
        targets = start_s + self.path_spacing * np.arange(
            start_index, start_index + self.path_length
        )
        valid = targets <= cumulative[-1] + 1e-5
        if np.any(valid):
            output[valid, 0] = np.interp(targets[valid], cumulative, polyline[:, 0])
            output[valid, 1] = np.interp(targets[valid], cumulative, polyline[:, 1])
        return output, valid

    def _polyline_headings(
        self, points: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        headings = np.zeros(points.shape[0], dtype=np.float32)
        count = int(np.sum(valid))
        if count < 2:
            return headings
        differences = np.diff(points[:count], axis=0)
        values = np.arctan2(differences[:, 1], differences[:, 0])
        if getattr(self.specification, "source_observation_contract", "") == "smarts":
            values = (values - math.pi * 0.5 + math.pi) % (2.0 * math.pi) - math.pi
        values = values.astype(np.float32)
        headings[: count - 1] = values
        headings[count - 1] = values[-1]
        return headings

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        success, collision, off_route, max_time = self._last_events
        active_vehicles = active_persons = 0
        if self._connection is not None:
            active_vehicles = len(self._connection.vehicle.getIDList())
            active_persons = len(self._connection.person.getIDList())
        info: dict[str, Any] = {
            "scenario": self.scenario,
            "is_success": bool(success),
            "collision": bool(collision),
            "geometric_collision": bool(self._last_geometric_collision),
            "off_route": bool(off_route),
            "max_time": bool(max_time),
            "legacy_info": (bool(success), bool(collision), bool(off_route), bool(max_time)),
            "raw_simulation_steps": self._raw_steps,
            "lifetime_raw_simulation_steps": self._lifetime_raw_steps,
            "decision_steps": self._decision_steps,
            "active_vehicles": active_vehicles,
            "active_persons": active_persons,
            "ego_control_profile": self.ego_control_profile,
            "state_lstm_contract": (
                "ego[x,y,speed,0,heading];social[x,y,speed,distance,heading]"
                if self.include_state_lstm
                else None
            ),
            "effective_target_speed": self._last_effective_target_speed,
            "curve_speed_limit": (
                self._last_curve_speed_limit
                if math.isfinite(self._last_curve_speed_limit)
                else None
            ),
        }
        info.update(extra)
        return info

    def set_lifetime_raw_step_budget(self, raw_steps: int | None) -> None:
        """Set an exact training-only raw-step ceiling across episode resets."""

        if raw_steps is not None and raw_steps <= 0:
            raise ValueError("raw_steps must be positive or None")
        self._lifetime_raw_steps = 0
        self._lifetime_raw_step_budget = raw_steps

    def render(self) -> None:
        # SUMO-GUI renders in its own process when render_mode="human".
        return None

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close(False)
            except Exception:
                # Closing must be idempotent, including after a SUMO-side error.
                pass
        self._episode_done = True

    def __enter__(self) -> "SumoSceneEnv":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
