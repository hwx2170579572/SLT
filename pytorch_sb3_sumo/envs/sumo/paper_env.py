"""SUMO environment backed directly by the authors' released scenario assets."""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .paper_scenario_registry import (
    PaperScenarioSpec,
    get_paper_scenario_spec,
    traffic_seed,
)
from .sumo_env import SumoSceneEnv, _copy_observation


TRAFFIC_PARTITIONS = ("all", "train", "evaluation")
EPISODE_LIMIT_PROFILES = ("source", "paper")
_PAPER_MAX_EPISODE_STEPS = {
    "left_turn": 400,
    "cross": 400,
    "cross_left": 600,
    "cross_left_unreg": 600,
    "roundabout_easy": 400,
    "roundabout_medium": 600,
    "roundabout": 800,
    # The paper does not publish a separate CARLA step count.  Retain the
    # only executable contract recoverable from carla_env.py.
    "carla": 302,
    # New SUMO scenarios (this project).  Urban merge and unsignalised
    # intersection use the same 600-decision-step cap as cross/cross_left.
    "merge": 600,
    "intersection": 600,
}


class PaperSumoSceneEnv(SumoSceneEnv):
    """Run the released SUMO networks/traffic without importing SMARTS.

    Five scenarios use the authors' released SUMO assets.  The CARLA task uses
    a SUMO network reconstructed from the released source/waypoint arrays.
    """

    def __init__(
        self,
        scenario: str = "left_turn",
        *,
        traffic_partition: str = "all",
        episode_limit_profile: str = "source",
        **kwargs: Any,
    ) -> None:
        if traffic_partition not in TRAFFIC_PARTITIONS:
            raise ValueError(
                f"traffic_partition must be one of {TRAFFIC_PARTITIONS}, "
                f"got {traffic_partition!r}"
            )
        if episode_limit_profile not in EPISODE_LIMIT_PROFILES:
            raise ValueError(
                f"episode_limit_profile must be one of {EPISODE_LIMIT_PROFILES}, "
                f"got {episode_limit_profile!r}"
            )
        paper_specification = get_paper_scenario_spec(scenario)
        self._paper_specification: PaperScenarioSpec | None = paper_specification
        self._traffic_partition = str(traffic_partition)
        self._episode_limit_profile = str(episode_limit_profile)
        self._selected_traffic_path: Path | None = None
        self._selected_traffic_seed: int | None = None
        self._traffic_episode_index = 0
        self._traffic_roll: int | None = None
        self._endless_routes: dict[str, tuple[tuple[str, ...], str]] = {}
        self._endless_route_counter = 0
        self._endless_reinsertions = 0
        self._direct_edge_connections: set[tuple[str, str]] = set()
        self._carla_source_paths: tuple[np.ndarray, ...] = ()
        kwargs.setdefault("path_spacing", paper_specification.waypoint_spacing)
        # SMARTS v0.4.17 NeighborhoodVehicles(radius=None) performs no radius
        # filtering; the released adapter then chooses the nearest five.
        kwargs.setdefault("neighbor_radius", math.inf)
        if paper_specification.source_observation_contract == "carla":
            source_map = Path(__file__).resolve().parents[1] / "carla" / "map"
            self._carla_source_paths = tuple(
                np.load(source_map / filename).astype(np.float32)
                for filename in ("wp.npy", "wp2.npy")
            )
        super().__init__(scenario=scenario, **kwargs)
        self._paper_specification = paper_specification
        # The paper specs intentionally satisfy the public attributes used by
        # SumoSceneEnv while redirecting the network to experiment assets.
        self.specification = self._paper_specification  # type: ignore[assignment]
        if paper_specification.source_observation_contract == "smarts":
            network_root = ET.parse(paper_specification.network_path).getroot()
            self._direct_edge_connections = {
                (connection.attrib["from"], connection.attrib["to"])
                for connection in network_root.findall("connection")
                if "from" in connection.attrib and "to" in connection.attrib
            }

    @property
    def max_episode_steps(self) -> int:
        if self._episode_limit_profile == "paper":
            return int(_PAPER_MAX_EPISODE_STEPS[self.scenario])
        return super().max_episode_steps

    def _partitioned_traffic_paths(
        self, specification: PaperScenarioSpec
    ) -> tuple[Path, ...]:
        paths = specification.traffic_paths
        # CARLA has one reconstructed SUMO traffic file, so a disjoint split
        # is impossible.  Preserve it in both partitions and expose that fact
        # through the info payload.
        if self._traffic_partition == "all" or len(paths) < 2:
            return paths
        evaluation = tuple(
            path for index, path in enumerate(paths) if index % 5 == 0
        )
        training = tuple(path for path in paths if path not in set(evaluation))
        selected = evaluation if self._traffic_partition == "evaluation" else training
        if not selected:
            raise RuntimeError(
                f"Traffic partition {self._traffic_partition!r} is empty for "
                f"scenario {specification.name!r}"
            )
        return selected

    @property
    def uses_released_assets(self) -> bool:
        return bool(
            self._paper_specification is not None
            and self._paper_specification.released_assets
        )

    def _sumo_command(self, seed: int) -> list[str]:
        specification = self._paper_specification
        if specification is None:
            return super()._sumo_command(seed)
        traffic_paths = self._partitioned_traffic_paths(specification)
        if self._traffic_roll is None:
            # HiWayEnv defaults to seed=42 and Scenario uses an inclusive
            # randint followed by np.roll(routes, roll_routes).  Reproduce
            # that fixed rolled cycle locally without importing SMARTS or
            # mutating Python's process-global random state.
            self._traffic_roll = random.Random(42).randint(0, len(traffic_paths))
        traffic_index = (
            -self._traffic_roll + self._traffic_episode_index
        ) % len(traffic_paths)
        traffic_path = traffic_paths[traffic_index]
        self._traffic_episode_index += 1
        self._selected_traffic_path = traffic_path
        self._selected_traffic_seed = traffic_seed(traffic_path)
        route_files = f"{traffic_path},{specification.ego_route_path}"
        command = [
            self._sumo_binary,
            "--net-file",
            str(specification.network_path),
            "--route-files",
            route_files,
            "--seed",
            str(seed),
            "--step-length",
            "0.1",
            "--collision.action",
            "none",
            "--collision.check-junctions",
            "true",
            "--lanechange.duration",
            "3.0",
            "--default.action-step-length",
            "0.1",
            "--begin",
            "0",
            "--end",
            "31536000",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--quit-on-end",
            "true",
            "--time-to-teleport",
            "-1",
            "--no-warnings",
            "true",
        ]
        command.extend(self._extra_sumo_args)
        return command

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._paper_specification is None:
            return super().reset(seed=seed, options=options)
        gym.Env.reset(self, seed=seed)
        del options
        self.close()
        simulation_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        self._traci.start(self._sumo_command(simulation_seed), label=self._label)
        self._connection = self._traci.getConnection(self._label)
        self._histories.clear()
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
        self._endless_routes.clear()
        self._endless_route_counter = 0
        self._endless_reinsertions = 0

        # The released missions inject ego after 5/15/30 seconds so that the
        # stochastic traffic field is already populated. SUMO advances at 0.1 s.
        insertion_limit = int(math.ceil(self._paper_specification.ego_depart_time / 0.1)) + 100
        for _ in range(insertion_limit):
            self._connection.simulationStep()
            self._after_simulation_step()
            self._raw_steps += 1
            if self.specification.ego_id in self._connection.vehicle.getIDList():
                break
        else:
            selected = self._selected_traffic_path
            self.close()
            raise RuntimeError(
                f"Ego was not inserted for {self.scenario!r}; traffic={selected}"
            )

        ego_id = self.specification.ego_id
        self._configure_policy_controlled_ego(ego_id)
        self._raw_steps = 0
        self._record_histories()
        observation = self._make_observation()
        self._last_observation = _copy_observation(observation)
        info = self._info_dict(simulation_seed=simulation_seed)
        return observation, info

    def _after_simulation_step(self) -> None:
        """Reproduce SMARTS' default endless social-traffic provider."""

        self._actors_hidden_this_observation.clear()
        specification = self._paper_specification
        if (
            specification is None
            or specification.source_observation_contract != "smarts"
        ):
            return

        ego_id = specification.ego_id
        active_ids = tuple(self._connection.vehicle.getIDList())
        for vehicle_id in active_ids:
            if vehicle_id == ego_id:
                continue
            try:
                route = tuple(self._connection.vehicle.getRoute(vehicle_id))
                route_index = int(
                    self._connection.vehicle.getRouteIndex(vehicle_id)
                )
                type_id = str(self._connection.vehicle.getTypeID(vehicle_id))
            except self._traci.TraCIException:
                continue
            if not route or route_index != len(route) - 1:
                continue
            if (route[-1], route[0]) in self._direct_edge_connections:
                try:
                    self._connection.vehicle.setRoute(
                        vehicle_id, [route[-1], *route]
                    )
                except self._traci.TraCIException:
                    pass
            else:
                self._endless_routes[vehicle_id] = (route, type_id)

        arrived = tuple(self._connection.simulation.getArrivedIDList())
        for vehicle_id in arrived:
            if vehicle_id == ego_id or vehicle_id not in self._endless_routes:
                continue
            route, type_id = self._endless_routes[vehicle_id]
            driving_lanes = self._driving_lanes(route[0])
            if not driving_lanes:
                continue
            lane_index = int(
                driving_lanes[
                    int(self.np_random.integers(0, len(driving_lanes)))
                ]
            )
            route_id = (
                f"scene_rep_endless_{self._endless_route_counter}_{vehicle_id}"
            )
            self._endless_route_counter += 1
            try:
                self._connection.route.add(route_id, list(route))
                self._connection.vehicle.add(
                    vehicle_id,
                    route_id,
                    typeID=type_id,
                    depart="now",
                    departLane=str(lane_index),
                    departPos="0",
                )
            except self._traci.TraCIException:
                continue
            self._actors_hidden_this_observation.add(vehicle_id)
            self._endless_reinsertions += 1

    def _actor_paths(self, actor_key: str, is_ego: bool) -> np.ndarray:
        specification = self._paper_specification
        if (
            specification is None
            or specification.source_observation_contract != "carla"
            or not is_ego
        ):
            return super()._actor_paths(actor_key, is_ego)

        # carla_env.py ignores progress along the ego route here: it orders the
        # two released global arrays by distance, takes their first 50 points,
        # pads a third mode, and finally keeps every fifth point.
        output = np.zeros(
            (
                specification.map_paths_per_actor,
                self.path_length,
                specification.map_feature_dim,
            ),
            dtype=np.float32,
        )
        state = self._state(actor_key)
        if state is None:
            return output
        ordered_paths = sorted(
            self._carla_source_paths,
            key=lambda path: float(
                np.min(np.linalg.norm(path[:, :2] - state[None, :2], axis=1))
            ),
        )
        for index, path in enumerate(ordered_paths[: output.shape[0]]):
            points = path[: self.path_length * 5 : 5, :2]
            output[index, : len(points)] = points
        return output

    def _events_after_step(self) -> tuple[bool, bool, bool, bool]:
        success, collision, off_route, max_time = super()._events_after_step()
        specification = self._paper_specification
        if (
            specification is not None
            and specification.source_observation_contract == "carla"
        ):
            # carla_env.py computes finish from the ego x/y captured before
            # world.tick(), while collision and off-route are observed after
            # the tick. Its exact success box is y > -32 and -54 < x < -50.5.
            previous = self._pre_simulation_ego_state
            success = bool(
                previous is not None
                and float(previous[1]) > -32.0
                and -54.0 < float(previous[0]) < -50.5
            )
            if success:
                off_route = False
        return bool(success), bool(collision), bool(off_route), bool(max_time)

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info["released_scenario_assets"] = self.uses_released_assets
        if self._paper_specification is not None:
            info["scenario_asset_source"] = (
                "authors_release_v1.0.0"
                if self._paper_specification.released_assets
                else "carla_source_waypoint_reconstruction"
            )
            info["source_observation_contract"] = (
                self._paper_specification.source_observation_contract
            )
            info["waypoint_spacing_m"] = self._paper_specification.waypoint_spacing
            info["source_coordinate_offset"] = list(
                self._paper_specification.coordinate_offset
            )
            info["episode_limit_profile"] = self._episode_limit_profile
            info["effective_max_episode_steps"] = self.max_episode_steps
            info["traffic_partition"] = self._traffic_partition
            info["traffic_partition_size"] = len(
                self._partitioned_traffic_paths(self._paper_specification)
            )
            info["traffic_partition_is_disjoint"] = bool(
                len(self._paper_specification.traffic_paths) >= 2
                and self._traffic_partition != "all"
            )
        if self._selected_traffic_path is not None:
            info["traffic_variant"] = self._selected_traffic_path.name
            info["traffic_variant_seed"] = self._selected_traffic_seed
            info["source_traffic_cycle_episode"] = self._traffic_episode_index - 1
            info["source_traffic_roll"] = self._traffic_roll
        info["source_endless_traffic"] = bool(
            self._paper_specification is not None
            and self._paper_specification.source_observation_contract == "smarts"
        )
        info["endless_traffic_reinsertions"] = self._endless_reinsertions
        return info
