"""Raw-tick factual actor history and M0/M1 observation construction."""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from .geometry import (
    cv_zone_event, sumo_front_to_actor_state, sumo_front_lane_position_to_center,
    swept_footprint_zone_interval,
)
from .map_cache import PublicMapCache
from .schema import (
    COMPLETE_LEGAL, INCOMPLETE, UNKNOWN, ZONE_CV_VALID, ZONE_NO_EVENT,
    PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY, PAIR_TOPOLOGY_FOE_WITHOUT_ZONE,
    empty_observation,
)


class SceneEventCollector:
    def __init__(self, config: Any, map_cache: PublicMapCache):
        self.config = config
        self.map_cache = map_cache
        self.max_actors = int(config.max_actors)
        self.history_samples = int(config.history_samples)
        self.max_candidates = 4
        self.max_path_lanes = 16
        self.radius_m = float(config.observation_radius)
        self.raw_dt = float(config.raw_dt)
        self.horizon_s = float(config.prediction_seconds)
        self._future_capacity = max(self.history_samples, int(math.ceil(self.horizon_s / self.raw_dt)) + 1)
        self._lane_index = {lane_id: index for index, lane_id in enumerate(map_cache.lane_ids)}
        self._unknown_topology_pairs: tuple[tuple[int, int, int], ...] = self._compile_unknown_topology_pairs()
        self._unknown_topology_static_counts = Counter()
        for _left, _right, status in self._unknown_topology_pairs:
            if status & PAIR_TOPOLOGY_FOE_WITHOUT_ZONE:
                self._unknown_topology_static_counts["foe_without_zone"] += 1
            if status & PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY:
                self._unknown_topology_static_counts["adjacent_exit_topology_only"] += 1
        self._path_cache: dict[tuple[int, bool], tuple[list[tuple[tuple[int, ...], bool]], bool]] = {}
        edge_index = map_cache.public_map["edge_index"]
        edge_type = map_cache.public_map["edge_type"]
        self._lateral_sources = {
            int(source) for (source, _target), relation in zip(edge_index.T, edge_type)
            if int(relation) in (2, 3)
        }
        map_lanes = len(map_cache.lane_ids)
        map_zones = len(map_cache.public_map["zone_valid"])
        self._max_path_event_cache_entries = max(256, map_lanes * 64)
        self._max_swept_interval_cache_entries = max(1024, map_lanes * max(1, map_zones) * 8)
        self._path_event_cache: OrderedDict[
            tuple[tuple[int, ...], int, int], dict[int, tuple[float, float]]
        ] = OrderedDict()
        self._swept_interval_cache: OrderedDict[
            tuple[int, int, int, int], tuple[float, float] | None
        ] = OrderedDict()
        self._geometry_cache_metrics: Counter[str] = Counter()
        self._geometry_cache_metrics["swept_interval_build_seconds"] = 0.0
        self.episode_index = 0
        self.episode_id = "uninitialized"
        self._ego_id = "ego"
        self.seed: int | None = None
        self.raw_tick = -1
        self._histories: dict[str, deque[tuple[int, np.ndarray, bool]]] = {}
        self._factual: dict[str, deque[tuple[int, np.ndarray, bool, float, int, float]]] = {}
        self._first_visible_tick: dict[str, int] = {}
        self._current_states: dict[str, tuple[np.ndarray, float, int, float]] = {}
        self._last_visible_ego: tuple[np.ndarray, float, int, float] | None = None
        self._current_lane_ids: dict[str, str | None] = {}
        self._widths: dict[str, float] = {}
        self._last_collection: dict[str, Any] | None = None
        self.audit = Counter()
        self._topology_unknown_lifetime = Counter()

    def _compile_unknown_topology_pairs(self) -> tuple[tuple[int, int, int], ...]:
        """Build endpoint-indexed unknown relation flags from the static audit."""
        lane_index = self._lane_index
        metadata = getattr(self.map_cache, "build_metadata", {})
        zone_metadata = metadata.get("zone_geometry", {}) if isinstance(metadata, dict) else {}
        pair_audit = zone_metadata.get("pair_relation_audit", ())
        codes: dict[tuple[int, int], int] = {}
        classification_code = {
            "evidence_backed_foe_relation_without_corridor_overlap": PAIR_TOPOLOGY_FOE_WITHOUT_ZONE,
            "evidence_backed_adjacent_exit_topology_only": PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY,
        }
        pair_ids: dict[str, list[str]] = {
            "foe_without_zone": [], "adjacent_exit_topology_only": [],
        }
        for record in pair_audit:
            code = classification_code.get(str(record.get("classification", "")), 0)
            if not code or record.get("event_zone_created") or not record.get("candidate_relevant"):
                continue
            pair = record.get("lane_pair", ())
            if len(pair) != 2 or pair[0] not in lane_index or pair[1] not in lane_index:
                continue
            left, right = sorted((int(lane_index[pair[0]]), int(lane_index[pair[1]])))
            if left == right:
                continue
            codes[(left, right)] = codes.get((left, right), 0) | code
            if code & PAIR_TOPOLOGY_FOE_WITHOUT_ZONE:
                pair_ids["foe_without_zone"].append(str(record.get("pair_id", "")))
            if code & PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY:
                pair_ids["adjacent_exit_topology_only"].append(str(record.get("pair_id", "")))
        self._unknown_topology_pair_ids = {
            key: sorted(value for value in values if value)
            for key, values in pair_ids.items()
        }
        return tuple((left, right, code) for (left, right), code in sorted(codes.items()))

    def _topology_status_contract(self) -> dict[str, Any]:
        actor_capacity = int(self.max_actors)
        candidate_capacity = int(self.max_candidates)
        return {
            "key": "candidate_pair_topology_status",
            "dtype": "uint8",
            "axes": ["actor_i", "candidate_i", "actor_j", "candidate_j"],
            "shape": [actor_capacity, candidate_capacity, actor_capacity, candidate_capacity],
            "bit_codes": {
                "1": "foe_without_zone",
                "2": "adjacent_exit_topology_only",
            },
            "bit_code_semantics": {
                "1": "public topology relation only; event and vehicle footprint clearance are unknown",
                "2": "adjacent exit topology only; event and vehicle footprint clearance are unknown",
            },
            "map_fingerprint": str(getattr(self.map_cache, "fingerprint", "unavailable")),
            "pair_audit_sha256": hashlib.sha256(json.dumps(
                getattr(self.map_cache, "build_metadata", {}).get("zone_geometry", {}).get(
                    "pair_relation_audit", []
                ), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest(),
            "static_pair_counts": dict(self._unknown_topology_static_counts),
            "foe_without_zone_pair_ids": list(self._unknown_topology_pair_ids["foe_without_zone"]),
            "adjacent_exit_pair_ids": list(self._unknown_topology_pair_ids["adjacent_exit_topology_only"]),
            "dynamic_route_pair_hits": int(self._topology_unknown_lifetime["topology_unknown_candidate_pairs"]),
            "dynamic_foe_route_pair_hits": int(self._topology_unknown_lifetime["topology_unknown_foe_candidate_pairs"]),
            "dynamic_adjacent_exit_route_pair_hits": int(
                self._topology_unknown_lifetime["topology_unknown_adjacent_exit_candidate_pairs"]
            ),
            "episode_dynamic_route_pair_hits": int(self.audit["topology_unknown_candidate_pairs"]),
            "unknown_relation_is_safety_label": False,
            "physical_clearance_known": False,
            "zero_semantics": "no audited topology-only pair matched these currently valid candidate routes; does not mean safe or interaction-free",
        }

    def begin_episode(self, seed: int | None) -> None:
        self.episode_index += 1
        self.seed = None if seed is None else int(seed)
        self.episode_id = f"scene_event_ep{self.episode_index:06d}_seed{self.seed if self.seed is not None else 'na'}"
        self.raw_tick = -1
        self._histories.clear()
        self._factual.clear()
        self._first_visible_tick.clear()
        self._current_states.clear()
        self._current_lane_ids.clear()
        self._widths.clear()
        self._last_visible_ego = None
        self._last_collection = None
        self.audit = Counter()

    def capture_raw_tick(self, env: Any) -> None:
        """Capture the already-completed TraCI tick; never advances SUMO."""
        if env._connection is None:
            return
        self.raw_tick += 1
        self.audit["raw_ticks"] += 1
        ego_id = str(env.specification.ego_id)
        ego_key = f"vehicle:{ego_id}"
        vehicle_ids = sorted(str(value) for value in env._connection.vehicle.getIDList())
        active_keys = {f"vehicle:{vehicle_id}" for vehicle_id in vehicle_ids}
        current: dict[str, tuple[np.ndarray, float, int, float]] = {}
        lane_names: dict[str, str | None] = {}
        for vehicle_id in vehicle_ids:
            key = f"vehicle:{vehicle_id}"
            try:
                front_xy = env._connection.vehicle.getPosition(vehicle_id)
                angle = float(env._connection.vehicle.getAngle(vehicle_id))
                speed = float(env._connection.vehicle.getSpeed(vehicle_id))
                length, width = env._vehicle_dimensions(vehicle_id)
                state = sumo_front_to_actor_state(front_xy, angle, speed, length)
                lane_name = str(env._connection.vehicle.getLaneID(vehicle_id))
                # TraCI lanePosition measures the front bumper from lane start;
                # our x/y state is the vehicle center, so subtract L/2 exactly.
                lane_s = sumo_front_lane_position_to_center(
                    float(env._connection.vehicle.getLanePosition(vehicle_id)), float(length)
                )
            except Exception:
                self.audit["state_read_errors"] += 1
                previous = self._factual.get(key)
                previous_length = float(previous[-1][3]) if previous else 0.0
                self._append_raw_record(key, np.zeros((6,), dtype=np.float32), False,
                                        previous_length, -1, 0.0)
                continue
            lane_ptr = self._lane_index.get(lane_name, -1)
            self._widths[key] = float(width)
            distance = 0.0
            if key != ego_key:
                ego_state = current.get(ego_key)
                if ego_state is not None:
                    distance = float(np.linalg.norm(state[:2] - ego_state[0][:2]))
                else:
                    distance = float("inf")
            current[key] = (state, float(length), lane_ptr, lane_s)
            lane_names[key] = lane_name

        ego = current.get(ego_key)
        if ego is not None:
            self._last_visible_ego = ego
        ego_position = (ego[0][:2] if ego is not None else
                        self._last_visible_ego[0][:2] if self._last_visible_ego is not None else None)
        visible_current: dict[str, tuple[np.ndarray, float, int, float]] = {}
        active_ids = set(vehicle_ids)
        for vehicle_id in vehicle_ids:
            key = f"vehicle:{vehicle_id}"
            values = current.get(key)
            if values is None:
                continue
            state, length, lane_ptr, lane_s = values
            visible = key == ego_key or (ego_position is not None and
                       float(np.linalg.norm(state[:2] - ego_position)) <= self.radius_m)
            if visible:
                visible_current[key] = values
                self._first_visible_tick.setdefault(key, self.raw_tick)
            self._append_raw_record(
                key, state if visible else np.zeros((6,), dtype=np.float32), bool(visible),
                float(length), int(lane_ptr), float(lane_s),
            )
        # Actors tracked on earlier ticks but absent now still occupy an
        # explicit invalid frame. This preserves a contiguous raw-tick axis
        # through departure and read failures instead of compressing history.
        for key in tuple(self._histories):
            if key not in active_keys:
                previous = self._factual.get(key)
                previous_length = float(previous[-1][3]) if previous else 0.0
                self._append_raw_record(key, np.zeros((6,), dtype=np.float32), False,
                                        previous_length, -1, 0.0)
        # Drop long-departed actors from per-episode active maps while retaining
        # their bounded future windows only for ids still within the window.
        for key in list(self._current_states):
            if key.split(":", 1)[1] not in active_ids:
                self._current_states.pop(key, None)
                self._current_lane_ids.pop(key, None)
        self._current_states = visible_current
        self._current_lane_ids = {key: lane_names.get(key) for key in visible_current}
        self.audit["visible_actor_samples"] += len(visible_current)

    def observation(self, remaining_time_s: float) -> dict[str, np.ndarray]:
        a, h, c, p = self.max_actors, self.history_samples, self.max_candidates, self.max_path_lanes
        zones = len(self.map_cache.public_map["zone_valid"])
        out = empty_observation(a, h, c, p, zones, float(self.config.deadline_seconds))
        frame_raw_tick = np.full((a, h), -1, dtype=np.int32)
        current = self._current_states
        ego_key = f"vehicle:{self._ego_id}"
        if ego_key not in current and self._last_visible_ego is None:
            self.audit["observation_without_ego"] += 1
            out["remaining_time_s"][0] = np.float32(max(0.0, remaining_time_s))
            self._last_collection = {
                "episode_id": self.episode_id, "raw_tick": self.raw_tick,
                "actor_keys": tuple([None] * a), "actor_lane_ptr": out["actor_lane_ptr"].copy(),
                "actor_valid": out["actor_valid"].copy(), "frame_raw_tick": frame_raw_tick,
                "observation": out,
            }
            return out
        all_visible = list(current.items())
        ego_values = current.get(ego_key, self._last_visible_ego)
        assert ego_values is not None
        others = [(key, values) for key, values in all_visible if key != ego_key]
        others.sort(key=lambda item: (float(np.linalg.norm(item[1][0][:2] - ego_values[0][:2])), item[0]))
        selected = [(ego_key, ego_values), *others[:a - 1]]
        actor_keys: list[str | None] = [None] * a
        for slot, (key, (state, length, lane_ptr, lane_s)) in enumerate(selected):
            actor_keys[slot] = key
            out["actor_valid"][slot] = True
            out["actor_size_lw"][slot] = (length, self._current_width(key))
            first_tick = self._first_visible_tick.get(key, self.raw_tick)
            out["actor_age_s"][slot] = np.float32(max(0.0, (self.raw_tick - first_tick) * self.raw_dt))
            out["actor_lane_ptr"][slot] = np.int32(lane_ptr)
            out["actor_lane_s_m"][slot] = np.float32(lane_s)
            out["lane_match_conf"][slot] = np.float32(1.0 if lane_ptr >= 0 else 0.0)
            history = list(self._histories.get(key, ()))
            for tick, row, valid in history[-h:]:
                target = h - 1 - (self.raw_tick - int(tick))
                if target < 0 or target >= h:
                    continue
                frame_raw_tick[slot, target] = np.int32(tick)
                if valid:
                    out["actor_history"][slot, target] = row
                    out["history_valid"][slot, target] = True
            self._fill_candidates(out, slot, key, lane_ptr, lane_s, state, length,
                                  float(out["actor_size_lw"][slot, 1]))
        self._fill_unknown_topology_pairs(out)
        out["remaining_time_s"][0] = np.float32(np.clip(remaining_time_s, 0.0, self.config.deadline_seconds))
        self.audit["policy_decisions"] += 1
        self.audit["history_slots_valid"] += int(out["history_valid"].sum())
        self.audit["actor_rows"] += len(selected)
        self.audit["candidate_valid"] += int(out["candidate_valid"].sum())
        self.audit["candidate_coverage_unknown"] += int(np.count_nonzero(out["candidate_unknown_count"] < 0))
        self.audit["zone_event_valid"] += int(out["candidate_zone_event_mask"].sum())
        pair_hits = int(np.count_nonzero(out["candidate_pair_topology_status"] & 3) // 2)
        foe_pair_hits = int(np.count_nonzero(
            out["candidate_pair_topology_status"] & PAIR_TOPOLOGY_FOE_WITHOUT_ZONE
        ) // 2)
        adjacent_pair_hits = int(np.count_nonzero(
            out["candidate_pair_topology_status"] & PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY
        ) // 2)
        self.audit["topology_unknown_candidate_pairs"] += pair_hits
        self.audit["topology_unknown_foe_candidate_pairs"] += foe_pair_hits
        self.audit["topology_unknown_adjacent_exit_candidate_pairs"] += adjacent_pair_hits
        self._topology_unknown_lifetime["topology_unknown_candidate_pairs"] += pair_hits
        self._topology_unknown_lifetime["topology_unknown_foe_candidate_pairs"] += foe_pair_hits
        self._topology_unknown_lifetime["topology_unknown_adjacent_exit_candidate_pairs"] += adjacent_pair_hits
        self._last_collection = {
            "episode_id": self.episode_id,
            "raw_tick": int(self.raw_tick),
            "actor_keys": tuple(actor_keys),
            "actor_lane_ptr": out["actor_lane_ptr"].copy(),
            "actor_valid": out["actor_valid"].copy(),
            "frame_raw_tick": frame_raw_tick,
            "observation": out,
        }
        return out

    def _fill_unknown_topology_pairs(self, out: dict[str, np.ndarray]) -> None:
        """Attach only audited no-zone relations to matching candidate paths.

        A set bit means the public topology audit found a typed relation for
        those candidate routes but no event zone models it. It is deliberately
        not converted into a time interval, collision label, or safe-negative.
        """
        if not self._unknown_topology_pairs:
            return
        actor_count, candidate_count = out["candidate_valid"].shape
        candidates_by_lane: dict[int, list[tuple[int, int]]] = {}
        for actor in range(actor_count):
            if not bool(out["actor_valid"][actor]):
                continue
            for candidate in range(candidate_count):
                if not bool(out["candidate_valid"][actor, candidate]):
                    continue
                lanes = out["candidate_lane_ptr"][actor, candidate][
                    out["candidate_lane_mask"][actor, candidate]
                ]
                for lane_ptr in np.unique(lanes):
                    candidates_by_lane.setdefault(int(lane_ptr), []).append((actor, candidate))
        for left_lane, right_lane, status in self._unknown_topology_pairs:
            left_candidates = candidates_by_lane.get(left_lane, ())
            right_candidates = candidates_by_lane.get(right_lane, ())
            for left_actor, left_route in left_candidates:
                for right_actor, right_route in right_candidates:
                    if left_actor == right_actor:
                        continue
                    out["candidate_pair_topology_status"][left_actor, left_route, right_actor, right_route] |= np.uint8(status)
                    out["candidate_pair_topology_status"][right_actor, right_route, left_actor, left_route] |= np.uint8(status)

    @property
    def config_scenario_ego_id(self) -> str:
        return self._ego_id

    def _current_width(self, key: str) -> float:
        history = self._factual.get(key)
        if not history:
            return 0.0
        # The provider length/width contract is recorded at capture time. Width
        # is not in the compact fact tuple yet; retrieve it from current TraCI
        # metadata cache populated alongside the state.
        return float(self._widths.get(key, 0.0))

    def set_ego_id(self, ego_id: str) -> None:
        self._ego_id = str(ego_id)

    def reset_widths(self) -> None:
        self._widths: dict[str, float] = {}

    def _append_raw_record(self, key: str, state: np.ndarray, valid: bool, length_m: float,
                           lane_ptr: int, lane_s_m: float) -> None:
        history = self._histories.setdefault(key, deque(maxlen=self.history_samples))
        if history and int(history[-1][0]) >= self.raw_tick:
            raise RuntimeError(f"raw history for {key} already contains tick {self.raw_tick}")
        history.append((self.raw_tick, np.asarray(state, dtype=np.float32).copy(), bool(valid)))
        factual = self._factual.setdefault(key, deque(maxlen=self._future_capacity))
        factual.append((self.raw_tick, np.asarray(state, dtype=np.float32).copy(), bool(valid),
                        float(length_m), int(lane_ptr), float(lane_s_m)))

    def _fill_candidates(self, out: dict[str, np.ndarray], actor_slot: int, actor_key: str,
                         lane_ptr: int, lane_s: float, state: np.ndarray, actor_length_m: float,
                         actor_width_m: float) -> None:
        if lane_ptr < 0 or lane_ptr >= len(self.map_cache.lane_ids):
            out["candidate_unknown_count"][actor_slot] = -1
            return
        ego = actor_key == f"vehicle:{self.config_scenario_ego_id}"
        route_mask = self.map_cache.public_map["ego_route_lane_mask"]
        if ego and not bool(route_mask[lane_ptr]):
            out["candidate_unknown_count"][actor_slot] = -1
            self.audit["ego_route_unknown"] += 1
            return
        cache_key = (int(lane_ptr), bool(ego))
        if cache_key not in self._path_cache:
            self._path_cache[cache_key] = _enumerate_successor_paths(
                lane_ptr, self.map_cache.lane_successors, self.map_cache.lane_ids,
                max_path_lanes=self.max_path_lanes, max_enumerated=8192,
                route_mask=route_mask if ego else None,
            )
        paths, overflow_unknown = self._path_cache[cache_key]
        if not paths:
            out["candidate_unknown_count"][actor_slot] = -1 if overflow_unknown else 0
            return
        path_lanes = {lane for path, _complete in paths for lane in path}
        has_lateral_alternative = bool(path_lanes & self._lateral_sources)
        for candidate_index, (path, complete) in enumerate(paths[:self.max_candidates]):
            out["candidate_valid"][actor_slot, candidate_index] = True
            out["candidate_status"][actor_slot, candidate_index] = np.int32(COMPLETE_LEGAL if complete else INCOMPLETE)
            path_size = min(len(path), self.max_path_lanes)
            out["candidate_lane_ptr"][actor_slot, candidate_index, :path_size] = np.asarray(path[:path_size], dtype=np.int32)
            out["candidate_lane_mask"][actor_slot, candidate_index, :path_size] = True
            out["candidate_progress_m"][actor_slot, candidate_index] = np.float32(lane_s)
            self._fill_zone_events(out, actor_slot, candidate_index, path, complete, lane_s,
                                   actor_length_m, actor_width_m, state)
        has_path_truncation = any(not complete for _path, complete in paths)
        if overflow_unknown or len(paths) > self.max_candidates or has_path_truncation or has_lateral_alternative:
            # -1 explicitly means full legal continuation coverage is unknown:
            # either graph enumeration was capped, a capacity truncated paths,
            # or M1 does not model a dynamically positioned lane-change path.
            out["candidate_unknown_count"][actor_slot] = -1
            self.audit["candidate_coverage_incomplete"] += 1
        else:
            out["candidate_unknown_count"][actor_slot] = 0

    def _fill_zone_events(self, out: dict[str, np.ndarray], actor_slot: int, candidate_index: int,
                          path: tuple[int, ...], complete: bool, current_lane_s: float,
                          actor_length_m: float, actor_width_m: float,
                          actor_state: np.ndarray) -> None:
        zones = len(self.map_cache.public_map["zone_valid"])
        if zones == 0:
            return
        speed = float(math.hypot(float(actor_state[4]), float(actor_state[5])))
        length_key = int(round(float(actor_length_m) * 1000.0))
        width_key = int(round(float(actor_width_m) * 1000.0))
        cache_key = (path, length_key, width_key)
        intervals = self._path_event_cache.get(cache_key)
        if intervals is None:
            self._geometry_cache_metrics["path_event_cache_misses"] += 1
            lane_zone_distance = self.map_cache.public_map["lane_zone_distance_m"]
            pieces: dict[int, list[tuple[float, float]]] = {}
            route_offset = 0.0
            for lane_idx in path:
                line = self.map_cache.lane_lines[lane_idx]
                lane_length = float(self.map_cache.lane_lengths_m[lane_idx])
                if line.length > 1e-9:
                    lane_to_geometry_scale = float(line.length) / max(lane_length, 1e-9)
                    actor_reach = math.hypot(0.5 * float(actor_length_m), 0.5 * float(actor_width_m))
                    candidate_zones = np.flatnonzero(lane_zone_distance[lane_idx] <= actor_reach + 1e-6)
                    for zone_idx_value in candidate_zones:
                        zone_idx = int(zone_idx_value)
                        interval_key = (int(lane_idx), zone_idx, length_key, width_key)
                        if interval_key in self._swept_interval_cache:
                            self._geometry_cache_metrics["swept_interval_cache_hits"] += 1
                            self._swept_interval_cache.move_to_end(interval_key)
                        else:
                            self._geometry_cache_metrics["swept_interval_cache_misses"] += 1
                            started = time.perf_counter()
                            self._swept_interval_cache[interval_key] = swept_footprint_zone_interval(
                                line, self.map_cache.zone_geometries[zone_idx],
                                float(actor_length_m), float(actor_width_m), sample_step_m=0.20,
                            )
                            self._geometry_cache_metrics["swept_interval_build_seconds"] += (
                                time.perf_counter() - started
                            )
                            self._swept_interval_cache.move_to_end(interval_key)
                            while len(self._swept_interval_cache) > self._max_swept_interval_cache_entries:
                                self._swept_interval_cache.popitem(last=False)
                        interval = self._swept_interval_cache[interval_key]
                        if interval is None:
                            continue
                        entry_s, clear_s = interval
                        # Geometry is measured along the parsed centerline;
                        # TraCI lane_s uses the lane's declared arclength.
                        entry_s /= lane_to_geometry_scale
                        clear_s /= lane_to_geometry_scale
                        pieces.setdefault(zone_idx, []).append(
                            (route_offset + float(entry_s), route_offset + float(clear_s))
                        )
                route_offset += lane_length
            intervals = {
                zone_idx: (min(entry for entry, _ in values), max(clear for _, clear in values))
                for zone_idx, values in pieces.items()
            }
            self._path_event_cache[cache_key] = intervals
            self._path_event_cache.move_to_end(cache_key)
            while len(self._path_event_cache) > self._max_path_event_cache_entries:
                self._path_event_cache.popitem(last=False)
        else:
            self._geometry_cache_metrics["path_event_cache_hits"] += 1
            self._path_event_cache.move_to_end(cache_key)
        for zone_idx in range(zones):
            if zone_idx not in intervals:
                if complete:
                    out["candidate_zone_status"][actor_slot, candidate_index, zone_idx] = np.int32(ZONE_NO_EVENT)
                continue
            center_entry, center_clear = intervals[zone_idx]
            # These are center positions of the first/last tangent-aligned
            # vehicle OBB touching the zone, so dimensions were already
            # applied in swept_footprint_zone_interval exactly once.
            distance_entry = center_entry - float(current_lane_s)
            distance_clear = center_clear - float(current_lane_s)
            distance_clear = max(distance_entry, distance_clear)
            out["candidate_zone_s_m"][actor_slot, candidate_index, zone_idx] = (distance_entry, distance_clear)
            out["candidate_zone_path_mask"][actor_slot, candidate_index, zone_idx] = True
            if distance_clear < 0.0:
                out["candidate_zone_status"][actor_slot, candidate_index, zone_idx] = np.int32(ZONE_NO_EVENT)
                continue
            event_entry = max(0.0, distance_entry)
            event_clear = max(event_entry, distance_clear)
            status, t_entry, t_clear = cv_zone_event(event_entry, event_clear, speed, self.horizon_s)
            out["candidate_zone_status"][actor_slot, candidate_index, zone_idx] = np.int32(status)
            if status == ZONE_CV_VALID:
                out["candidate_zone_event_s"][actor_slot, candidate_index, zone_idx] = (t_entry, t_clear)
                out["candidate_zone_event_mask"][actor_slot, candidate_index, zone_idx] = True

    def future_history(self, actor_key: str, from_tick: int, through_tick: int) -> list[dict[str, Any]]:
        """Return bounded factual states for future-label joins; ids stay sidecar-only."""
        if through_tick < from_tick:
            raise ValueError("through_tick must be >= from_tick")
        return [
            {"raw_tick": tick, "state_world": state.copy(), "valid": bool(valid),
             "length_m": float(length), "lane_ptr": int(lane_ptr), "lane_s_m": float(lane_s)}
            for tick, state, valid, length, lane_ptr, lane_s in self._factual.get(actor_key, ())
            if int(from_tick) <= tick <= int(through_tick)
        ]

    @property
    def last_collection(self) -> dict[str, Any] | None:
        return self._last_collection

    def summary(self) -> dict[str, Any]:
        return {
            "protocol": "scene_event_raw_history_v1",
            "episode_id": self.episode_id,
            "raw_tick": int(self.raw_tick),
            "history_samples": self.history_samples,
            "history_valid_rows_total": int(self.audit["history_slots_valid"]),
            "policy_decisions": int(self.audit["policy_decisions"]),
            "actor_rows": int(self.audit["actor_rows"]),
            "candidate_valid": int(self.audit["candidate_valid"]),
            "candidate_coverage_unknown": int(self.audit["candidate_coverage_incomplete"]),
            "ego_route_unknown": int(self.audit["ego_route_unknown"]),
            "zone_events_valid": int(self.audit["zone_event_valid"]),
            "topology_unknown_relation_static_counts": dict(self._unknown_topology_static_counts),
            "topology_unknown_candidate_pairs": int(self.audit["topology_unknown_candidate_pairs"]),
            "topology_unknown_foe_candidate_pairs": int(self.audit["topology_unknown_foe_candidate_pairs"]),
            "topology_unknown_adjacent_exit_candidate_pairs": int(self.audit["topology_unknown_adjacent_exit_candidate_pairs"]),
            "topology_unknown_relation_semantics": "known typed map relation without a mapped event zone; not a collision label or evidence of safety",
            "candidate_pair_topology_status_contract": self._topology_status_contract(),
            "raw_state_read_errors": int(self.audit["state_read_errors"]),
            "observation_without_ego": int(self.audit["observation_without_ego"]),
            "additional_simulator_steps": 0,
            "collector_social_future_routes_read": False,
            "legacy_make_observation_route_path": "disabled_by_scene_event_subclass",
            "unmapped_candidates_are_unknown": True,
            "geometry_cache": {
                "path_event_hits": int(self._geometry_cache_metrics["path_event_cache_hits"]),
                "path_event_misses": int(self._geometry_cache_metrics["path_event_cache_misses"]),
                "path_event_entries": len(self._path_event_cache),
                "path_event_capacity": self._max_path_event_cache_entries,
                "swept_interval_hits": int(self._geometry_cache_metrics["swept_interval_cache_hits"]),
                "swept_interval_misses": int(self._geometry_cache_metrics["swept_interval_cache_misses"]),
                "swept_interval_entries": len(self._swept_interval_cache),
                "swept_interval_capacity": self._max_swept_interval_cache_entries,
                "swept_interval_build_seconds": float(
                    self._geometry_cache_metrics["swept_interval_build_seconds"]
                ),
            },
        }


def _enumerate_successor_paths(start: int, successors: tuple[tuple[int, ...], ...],
                               lane_ids: tuple[str, ...], *, max_path_lanes: int,
                               max_enumerated: int, route_mask: np.ndarray | None) -> tuple[list[tuple[tuple[int, ...], bool]], bool]:
    completed: list[tuple[tuple[int, ...], bool]] = []
    expansions = 0
    capped = False
    stack: list[tuple[int, ...]] = [(int(start),)]
    while stack:
        path = stack.pop()
        expansions += 1
        if expansions > max_enumerated:
            capped = True
            break
        next_lanes = [n for n in successors[path[-1]] if n not in path and
                      (route_mask is None or bool(route_mask[n]))]
        if not next_lanes:
            cyclic = any(n in path for n in successors[path[-1]])
            completed.append((path, not cyclic))
            continue
        if len(path) >= max_path_lanes:
            completed.append((path, False))
            continue
        for nxt in reversed(sorted(next_lanes, key=lambda i: lane_ids[i])):
            stack.append((*path, int(nxt)))
    completed.sort(key=lambda item: tuple(lane_ids[index] for index in item[0]))
    return completed, capped
