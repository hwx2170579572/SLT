"""SUMO environment backed directly by the authors' released scenario assets."""

from __future__ import annotations

import math
import random
import tempfile
from bisect import bisect_left
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
from .random_intersection import (
    RANDOM_INTERSECTION_SCENARIOS,
    SEED_DOMAINS,
    get_random_intersection_config,
    is_random_intersection_scenario,
    random_intersection_seed,
    write_seeded_episode_traffic,
)


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


_PAPER_MAX_EPISODE_STEPS.update({name: 600 for name in RANDOM_INTERSECTION_SCENARIOS})


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
        traffic_split: str | None = None,
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
        self._random_traffic_config = (
            get_random_intersection_config(scenario)
            if is_random_intersection_scenario(scenario) else None
        )
        if self._random_traffic_config is not None:
            self._traffic_split = traffic_split or (
                "validation" if traffic_partition == "evaluation" else "train"
            )
            if self._traffic_split not in SEED_DOMAINS:
                raise ValueError(f"traffic_split must be one of {tuple(SEED_DOMAINS)}")
        else:
            if traffic_split is not None:
                raise ValueError("traffic_split applies only to the versioned random intersection scenarios")
            self._traffic_split = None
        self._traffic_logical_episode_seed: int | None = None
        self._traffic_sumo_seed: int | None = None
        self._random_episode_traffic_directory = None
        self._random_episode_schedule: dict | None = None
        self._random_warmup_checkpoints: list[dict] = []
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

    def set_traffic_split(self, split: str) -> None:
        """Select the random-traffic seed domain before starting an episode."""
        if self._random_traffic_config is None:
            raise ValueError("traffic_split is only defined for random intersection scenarios")
        if split not in SEED_DOMAINS:
            raise ValueError(f"traffic_split must be one of {tuple(SEED_DOMAINS)}")
        if getattr(self, "_connection", None) is not None:
            raise RuntimeError("Close the environment before changing its traffic split")
        self._traffic_split = split

    def _partitioned_traffic_paths(
        self, specification: PaperScenarioSpec
    ) -> tuple[Path, ...]:
        paths = specification.traffic_paths
        if self._random_traffic_config is not None:
            # A shared flow definition is not a shared arrival realization:
            # the split-specific SUMO seed instantiates each episode's traffic.
            if len(paths) != 1:
                raise ValueError("Random intersection scenarios require one frozen flow definition")
            return paths
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
        if self._random_traffic_config is not None:
            traffic_index = 0
        elif self._traffic_roll is None:
            # HiWayEnv defaults to seed=42 and Scenario uses an inclusive
            # randint followed by np.roll(routes, roll_routes).  Reproduce
            # that fixed rolled cycle locally without importing SMARTS or
            # mutating Python's process-global random state.
            self._traffic_roll = random.Random(42).randint(0, len(traffic_paths))
        if self._random_traffic_config is None:
            traffic_index = (
                -self._traffic_roll + self._traffic_episode_index
            ) % len(traffic_paths)
        traffic_path = traffic_paths[traffic_index]
        self._random_episode_schedule = None
        if self._random_traffic_config is not None and self._random_traffic_config.get("dynamic_episode_traffic"):
            if self._random_episode_traffic_directory is None:
                self._random_episode_traffic_directory = tempfile.TemporaryDirectory(prefix="paper_bernoulli_")
            episode_path = Path(self._random_episode_traffic_directory.name) / "episode.rou.xml"
            self._random_episode_schedule = write_seeded_episode_traffic(
                self._random_traffic_config["scenario"], int(seed), traffic_path, episode_path,
            )
            traffic_path = episode_path
        self._traffic_episode_index += 1
        self._selected_traffic_path = traffic_path
        self._selected_traffic_seed = (
            int(seed) if self._random_traffic_config is not None else traffic_seed(traffic_path)
        )
        route_files = f"{traffic_path},{specification.ego_route_path}"
        collision_action = (
            str(self._random_traffic_config.get("collision_action", "none"))
            if self._random_traffic_config is not None
            else "none"
        )
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
            collision_action,
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
        if self._random_traffic_config is not None:
            self._traffic_logical_episode_seed = simulation_seed
            simulation_seed = random_intersection_seed(self._traffic_split, simulation_seed)
            self._traffic_sumo_seed = simulation_seed
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
        self._random_departures = {}
        self._random_departures_at_control_start = {}
        self._random_arrivals = 0
        self._random_peak_active = 0
        self._random_delay_count = 0
        self._random_delay_sum = 0.0
        self._random_delay_max = 0.0
        self._random_control_start_time = None
        self._random_warmup_checkpoints = []
        if self._behavior_diagnostics is not None:
            self._behavior_actor_states.clear()
            self._behavior_observed_neighbor_ids = None
            self._behavior_observation_raw_step = None
            self._behavior_previous_actor_payloads.clear()
            self._behavior_static_actor_metadata.clear()
            self._behavior_dynamic_actor_context.clear()
            self._behavior_last_sim_time = None
            self._behavior_step_seconds = None
            self._behavior_pending_errors.clear()
            self._behavior_collision_events = []
            self._behavior_collision_ids = []

        # The released missions inject ego after 5/15/30 seconds so that the
        # stochastic traffic field is already populated. SUMO advances at 0.1 s.
        insertion_limit = int(math.ceil(self._paper_specification.ego_depart_time / 0.1)) + 100
        for _ in range(insertion_limit):
            self._connection.simulationStep()
            self._after_simulation_step()
            self._raw_steps += 1
            if self._random_traffic_config is not None and self._raw_steps in (300, 400, 500):
                self._capture_random_warmup_checkpoint(self._raw_steps * 0.1)
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
        if self._random_traffic_config is not None:
            self._random_control_start_time = float(self._connection.simulation.getTime())
            self._random_departures_at_control_start = dict(self._random_departures)
        self._record_histories()
        observation = self._make_observation()
        self._last_observation = _copy_observation(observation)
        info = self._info_dict(simulation_seed=simulation_seed)
        if self._random_traffic_config is not None:
            # Persist once per episode through the existing diagnostics reset
            # record, without duplicating all snapshots in every control step.
            info["warmup_traffic_checkpoints"] = list(self._random_warmup_checkpoints)
        if self._behavior_diagnostics is not None:
            info["requested_reset_seed"] = seed
            info["ego_speed_mode"] = 0
            info["ego_lane_change_mode"] = 0
            self._behavior_refresh_dynamic_context()
            snapshot = self._behavior_snapshot(self._last_events)
            self._behavior_previous_actor_payloads = self._snapshot_actor_payloads(
                snapshot
            )
            self._behavior_diagnostics.on_reset(
                snapshot, info=info, seed=simulation_seed
            )
        return observation, info

    def _capture_random_warmup_checkpoint(self, target_seconds: float) -> None:
        """Observe the normal warmup at 30/40/50 s; never advance simulation."""
        connection = self._connection
        now = float(connection.simulation.getTime())
        ego_id = self.specification.ego_id
        vehicle_ids = tuple(connection.vehicle.getIDList())
        background_ids = tuple(vehicle_id for vehicle_id in vehicle_ids if vehicle_id != ego_id)
        pending_ids = tuple(
            vehicle_id for vehicle_id in connection.simulation.getPendingVehicles()
            if vehicle_id != ego_id
        )
        flows = self._random_traffic_config["flows"]

        def counts_by_route(ids):
            counts = {flow["id"]: 0 for flow in flows}
            for vehicle_id in ids:
                key = next((
                    flow["id"] for flow in flows
                    if vehicle_id.startswith(f"background_{flow['id']}.")
                ), "other")
                counts[key] = counts.get(key, 0) + 1
            return counts

        pending_by_route = counts_by_route(pending_ids)
        inserted_by_route = {flow["id"]: self._random_departures.get(flow["id"], 0) for flow in flows}
        if self._random_episode_schedule is not None:
            # simulationStep reports time after the processed tick.  A
            # scheduled departure at exactly 'now' is due on the next tick.
            due_by_route = {
                route_id: bisect_left(times, now - 1e-8)
                for route_id, times in self._random_episode_schedule["departure_times_by_route"].items()
            }
            request_count_source = "seeded_schedule_depart_time_less_than_snapshot_time"
        else:
            # Native flow requests have no Python schedule.  With SUMO's
            # unlimited departure delay, generated demand is entered+pending.
            due_by_route = {
                route_id: count + pending_by_route.get(route_id, 0)
                for route_id, count in inserted_by_route.items()
            }
            request_count_source = "native_flow_inserted_plus_pending"
        inlet_lanes = {}
        for flow in flows:
            lane_id = f"{flow['edges'][0]}_{flow['depart_lane']}"
            inlet_lanes[lane_id] = {
                "vehicles_on_lane": int(connection.lane.getLastStepVehicleNumber(lane_id)),
                "halting_vehicles_on_lane": int(connection.lane.getLastStepHaltingNumber(lane_id)),
                "requested_due": due_by_route[flow["id"]],
                "actually_inserted_cumulative": inserted_by_route[flow["id"]],
                "pending_insertion": pending_by_route[flow["id"]],
            }
        inserted = sum(self._random_departures.values())
        due = sum(due_by_route.values())
        self._random_warmup_checkpoints.append({
            "target_seconds": float(target_seconds),
            "simulation_time_seconds": now,
            "in_network_vehicles": len(vehicle_ids),
            "in_network_background_vehicles": len(background_ids),
            "in_network_background_by_route": counts_by_route(background_ids),
            "ego_present": ego_id in vehicle_ids,
            "queue_definition": "halting vehicles with speed below 0.1 m/s; pending insertion is counted separately",
            "halting_speed_threshold_mps": 0.1,
            "halting_background_vehicles": sum(float(connection.vehicle.getSpeed(vehicle_id)) < 0.1 for vehicle_id in background_ids),
            "inlet_lanes": inlet_lanes,
            "request_count_source": request_count_source,
            "requested_background_due": due,
            "requested_background_due_by_route": due_by_route,
            "actually_inserted_background_cumulative": inserted,
            "actually_inserted_background_by_route": inserted_by_route,
            "actual_insertion_rate_vph_since_begin": inserted * 3600.0 / now if now > 0 else None,
            "pending_background_insertions": len(pending_ids),
            "pending_background_by_route": pending_by_route,
            "request_balance_due_minus_inserted_minus_pending": due - inserted - len(pending_ids),
            "arrived_background_cumulative": self._random_arrivals,
            "mean_departure_delay_seconds": self._random_delay_sum / self._random_delay_count if self._random_delay_count else None,
            "max_departure_delay_seconds": self._random_delay_max if self._random_delay_count else None,
        })

    def _after_simulation_step(self) -> None:
        """Reproduce SMARTS' default endless social-traffic provider."""

        self._actors_hidden_this_observation.clear()
        if self._random_traffic_config is not None:
            # Configured stochastic requests are the sole source of background
            # arrivals. Recycling completed vehicles would add unconfigured
            # demand and make the requested rates uninterpretable.
            ego_id = self.specification.ego_id
            for vehicle_id in self._connection.simulation.getDepartedIDList():
                if vehicle_id == ego_id:
                    continue
                route_id = next((
                    flow["id"] for flow in self._random_traffic_config["flows"]
                    if vehicle_id.startswith(f"background_{flow['id']}.")
                ), "other")
                self._random_departures[route_id] = self._random_departures.get(route_id, 0) + 1
                if hasattr(self._connection.vehicle, "getDepartDelay"):
                    delay = float(self._connection.vehicle.getDepartDelay(vehicle_id))
                    self._random_delay_count += 1
                    self._random_delay_sum += delay
                    self._random_delay_max = max(self._random_delay_max, delay)
            self._random_arrivals += sum(
                vehicle_id != ego_id
                for vehicle_id in self._connection.simulation.getArrivedIDList()
            )
            active = sum(vehicle_id != ego_id for vehicle_id in self._connection.vehicle.getIDList())
            self._random_peak_active = max(self._random_peak_active, active)
            return
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
            and self._random_traffic_config is None
        )
        info["endless_traffic_reinsertions"] = self._endless_reinsertions
        if self._random_traffic_config is not None:
            offset, width = SEED_DOMAINS[self._traffic_split]
            info["scenario_asset_source"] = "project_random_intersection_v1"
            info["traffic_partition"] = self._traffic_split
            info["traffic_partition_unit"] = "simulation_seed"
            info["traffic_partition_size"] = width
            info["traffic_partition_is_disjoint"] = True
            info["traffic_definition_file_count"] = 1
            info["traffic_protocol"] = self._random_traffic_config["protocol"]
            info["traffic_split"] = self._traffic_split
            info["traffic_logical_episode_seed"] = self._traffic_logical_episode_seed
            info["traffic_sumo_seed"] = self._traffic_sumo_seed
            info["source_traffic_cycle_episode"] = None
            info["source_traffic_roll"] = None
            control_departures = {
                flow["id"]: self._random_departures.get(flow["id"], 0)
                - self._random_departures_at_control_start.get(flow["id"], 0)
                for flow in self._random_traffic_config["flows"]
            }
            elapsed = (
                float(self._connection.simulation.getTime()) - self._random_control_start_time
                if self._connection is not None and self._random_control_start_time is not None
                else 0.0
            )
            pending = None
            if self._connection is not None and hasattr(self._connection.simulation, "getPendingVehicles"):
                pending = sum(
                    vehicle_id != self.specification.ego_id
                    for vehicle_id in self._connection.simulation.getPendingVehicles()
                )
            info["random_traffic"] = {
                "protocol": self._random_traffic_config["protocol"],
                "split": self._traffic_split,
                "logical_episode_seed": self._traffic_logical_episode_seed,
                "sumo_seed": self._traffic_sumo_seed,
                "seed_domain": [offset, offset + width - 1],
                "requested_vehicles_per_hour": self._random_traffic_config["total_vehicles_per_hour"],
                "requested_vehicles_per_hour_by_route": self._random_traffic_config["vehicles_per_hour_by_route"],
                "departed_background_total": sum(self._random_departures.values()),
                "departed_background_by_route_total": dict(self._random_departures),
                "departed_background_control": sum(control_departures.values()),
                "departed_background_by_route_control": control_departures,
                "control_elapsed_seconds": elapsed,
                "realized_departure_rate_vph_control": sum(control_departures.values()) * 3600.0 / elapsed if elapsed > 0 else None,
                "arrived_background_total": self._random_arrivals,
                "pending_background_insertions": pending,
                "peak_background_vehicles": self._random_peak_active,
                "departure_delay_samples": self._random_delay_count,
                "mean_departure_delay_seconds": self._random_delay_sum / self._random_delay_count if self._random_delay_count else None,
                "max_departure_delay_seconds": self._random_delay_max if self._random_delay_count else None,
                "endless_reinsertions_enabled": False,
            }
            if self._random_episode_schedule is not None:
                info["random_traffic"]["episode_schedule"] = {
                    key: value for key, value in self._random_episode_schedule.items()
                    if key != "departure_times_by_route"
                }
        return info
