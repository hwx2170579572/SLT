"""Route-conditioned conflict-timing observation extension for selected methods.

This wrapper adds a supplemental static-network feature. It leaves the released
trajectory/map arrays and environment transition protocol untouched. Only ego's
task route and currently observed actors' lane identities are read from TraCI;
background future routes are never queried.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from .route_reachability_v1 import RouteLaneReachability
from .route_conflict_timing import (
    FEATURE_DIM,
    FEATURE_NAMES,
    compute_route_conflict_timing,
    polyline_arclength,
    project_point_to_polyline,
)


PROTOCOL = "route_conflict_timing_observation_v1"
DEFAULT_HORIZON_S = 10.0
DEFAULT_MAX_DISTANCE_M = 60.0
DEFAULT_PROXY_LENGTH_M = 4.7
DEFAULT_PROXY_WIDTH_M = 1.8
DEFAULT_MAX_SOCIAL_PATHS = 4
DEFAULT_MAX_EGO_PATHS = 8
_MIN_JOIN_GAP_M = 0.75
_POSITION_PATH_TOLERANCE_M = 5.0
_MAP_POINT_TOLERANCE_M = 2.5


@dataclass(frozen=True)
class _StaticNetwork:
    reachability: RouteLaneReachability
    lane_shapes: Mapping[str, np.ndarray]
    network_sha256: str


@lru_cache(maxsize=8)
def _load_static_network(
    network_path: str,
    offset_x: float,
    offset_y: float,
    vehicle_class: str = "passenger",
) -> _StaticNetwork:
    path = Path(network_path).resolve()
    reachability = RouteLaneReachability.from_net(path, vehicle_class=vehicle_class)
    root = ET.parse(path).getroot()
    lane_shapes: dict[str, np.ndarray] = {}
    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            shape_text = lane.get("shape")
            if not lane_id or not shape_text:
                continue
            try:
                points = np.asarray(
                    [[float(value) for value in pair.split(",")] for pair in shape_text.split()],
                    dtype=np.float64,
                )
            except (ValueError, TypeError):
                continue
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                continue
            if not np.isfinite(points).all():
                continue
            if offset_x or offset_y:
                points = points - np.asarray([offset_x, offset_y], dtype=np.float64)
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
            points = points[keep]
            if len(points) >= 2:
                lane_shapes[str(lane_id)] = points
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _StaticNetwork(reachability, lane_shapes, digest)


def _as_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _join_lane_shapes(
    lane_ids: Sequence[str], lane_shapes: Mapping[str, np.ndarray]
) -> tuple[np.ndarray | None, str]:
    pieces: list[np.ndarray] = []
    for lane_id in lane_ids:
        shape = lane_shapes.get(lane_id)
        if shape is None or len(shape) < 2:
            return None, "lane_shape_missing"
        piece = np.asarray(shape, dtype=np.float64)
        if pieces:
            gap = float(np.linalg.norm(pieces[-1][-1] - piece[0]))
            if gap > _MIN_JOIN_GAP_M:
                return None, "connected_lane_shape_gap"
            if gap <= 1e-5:
                piece = piece[1:]
        if len(piece):
            pieces.append(piece)
    if not pieces:
        return None, "lane_shape_missing"
    points = np.concatenate(pieces, axis=0)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) < 2:
        return None, "degenerate_lane_path"
    return points, "known"


def _trim_path(points: np.ndarray, fixed_limit_s: float) -> np.ndarray:
    clean, cumulative = polyline_arclength(points)
    limit_s = min(float(cumulative[-1]), float(fixed_limit_s))
    if limit_s >= cumulative[-1] - 1e-8:
        return clean
    keep_count = int(np.searchsorted(cumulative, limit_s, side="right"))
    keep_count = max(1, min(keep_count, len(clean) - 1))
    result = clean[:keep_count].copy()
    segment = keep_count - 1
    span = float(cumulative[segment + 1] - cumulative[segment])
    if span <= 1e-9:
        return result
    fraction = (limit_s - float(cumulative[segment])) / span
    endpoint = clean[segment] + fraction * (clean[segment + 1] - clean[segment])
    result = np.vstack([result, endpoint])
    return result


def _current_observation_row(raw_env: Any, trajectory: np.ndarray, index: int) -> np.ndarray | None:
    if trajectory.ndim != 3 or index >= trajectory.shape[0] or trajectory.shape[1] == 0:
        return None
    contract = getattr(getattr(raw_env, "specification", None), "source_observation_contract", "")
    if contract == "carla":
        current_index = 0
    else:
        try:
            timestep = max(1, int(getattr(raw_env, "_history_timestep")))
            current_index = min(timestep, trajectory.shape[1]) - 1
        except (AttributeError, TypeError, ValueError, OverflowError):
            current_index = trajectory.shape[1] - 1
    state = np.asarray(trajectory[index, current_index], dtype=np.float64)
    if state.ndim != 1 or len(state) < 5 or not np.isfinite(state[:5]).all():
        return None
    return state


class _CompressedJsonl:
    def __init__(self, path: Path, schema: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = gzip.open(path, "xt", encoding="utf-8")
        self.rows = 0
        self.schema = schema

    def write(self, row: Mapping[str, Any]) -> bool:
        try:
            self._handle.write(json.dumps(_json_safe(row), separators=(",", ":"), allow_nan=False) + "\n")
        except (TypeError, ValueError, OSError):
            return False
        self.rows += 1
        return True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class RouteConflictTimingObservationWrapper(gym.Wrapper):
    """Add fixed-shape route-conditioned crossing-timing features per actor.

    Social candidate paths are enumerated from only the currently observed lane
    and public static lane connections. The ego's public task route constrains
    its candidate paths. This is a new static-geometry input source; it does not
    alter existing observation arrays or environment transitions.
    """

    KEY = "conflict_timing"

    def __init__(
        self,
        env: gym.Env,
        *,
        diagnostics_directory: str | Path | None = None,
        phase: str = "train",
        horizon_s: float = DEFAULT_HORIZON_S,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
        max_social_paths: int = DEFAULT_MAX_SOCIAL_PATHS,
        max_prediction_rows: int = 50000,
        max_calibration_rows: int = 200000,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Conflict timing requires a Dict observation space")
        if self.KEY in env.observation_space.spaces:
            raise ValueError("Conflict timing observation is already present")
        if phase not in {"train", "eval", "validation", "smoke"}:
            raise ValueError("phase must identify a known train/evaluation stage")
        if not math.isfinite(float(horizon_s)) or float(horizon_s) <= 0:
            raise ValueError("horizon_s must be finite and positive")
        if not math.isfinite(float(max_distance_m)) or float(max_distance_m) <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        if int(max_social_paths) <= 0 or int(max_prediction_rows) < 0 or int(max_calibration_rows) < 0:
            raise ValueError("candidate and diagnostic budgets must be non-negative; paths must be positive")
        raw_env = self.unwrapped
        specification = getattr(raw_env, "specification", None)
        network_path = getattr(specification, "network_path", None)
        if network_path is None:
            raise TypeError("Conflict timing requires specification.network_path")
        offset_x, offset_y = (0.0, 0.0)
        if getattr(specification, "source_observation_contract", "") == "carla":
            offset_x, offset_y = getattr(specification, "coordinate_offset", (0.0, 0.0))
        self.static = _load_static_network(
            str(Path(network_path).resolve()), float(offset_x), float(offset_y), "passenger"
        )
        self.phase = str(phase)
        self.horizon_s = float(horizon_s)
        self.max_distance_m = float(max_distance_m)
        self.max_social_paths = int(max_social_paths)
        self.max_prediction_rows = int(max_prediction_rows)
        self.max_calibration_rows = int(max_calibration_rows)
        self._episode_index = -1
        self._episode_seed: Any = None
        self._decision_index = 0
        self._last_frame: dict[str, Any] | None = None
        self._pending: list[dict[str, Any]] = []
        self._closed = False
        self._prediction_rows_written = 0
        self._prediction_rows_dropped = 0
        self._calibration_rows_written = 0
        self._calibration_rows_dropped = 0
        self._pending_pairs_dropped = 0
        self._pending_pairs_right_censored = 0
        self._prediction_writer: _CompressedJsonl | None = None
        self._calibration_writer: _CompressedJsonl | None = None
        self._network_path = str(Path(network_path).resolve())
        spaces = dict(env.observation_space.spaces)
        actor_count = int(spaces["trajectory"].shape[0]) if "trajectory" in spaces else None
        if actor_count is None:
            raise TypeError("Conflict timing requires the public trajectory observation")
        spaces[self.KEY] = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(actor_count, FEATURE_DIM), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        if diagnostics_directory is not None:
            directory = Path(diagnostics_directory)
            self._prediction_writer = _CompressedJsonl(
                directory / "task_conflict_predictions.jsonl.gz", "conflict_timing_prediction_v1"
            )
            self._calibration_writer = _CompressedJsonl(
                directory / "task_conflict_calibration.jsonl.gz", "conflict_timing_calibration_v1"
            )
            self._write_manifest(directory)

    def _write_manifest(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "protocol": PROTOCOL,
            "kernel_schema": "route_crossing_timing_v1",
            "feature_names": list(FEATURE_NAMES),
            "phase": self.phase,
            "network_path": self._network_path,
            "network_sha256": self.static.network_sha256,
            "geometry_source": "static_sumo_lane_shapes_plus_current_ego_task_route",
            "social_future_routes_read": False,
            "original_trajectory_and_map_preserved": True,
            "supplemental_lookahead_m": self.max_distance_m,
            "horizon_s": self.horizon_s,
            "proxy_length_m": DEFAULT_PROXY_LENGTH_M,
            "proxy_width_m": DEFAULT_PROXY_WIDTH_M,
            "minimum_supported_crossing_angle_degrees": 20.0,
            "supported_relation": "nonparallel_crossing_corridor_only",
            "unsupported_relations_are_not_safe": True,
            "social_candidate_limit": self.max_social_paths,
            "ego_candidate_limit": DEFAULT_MAX_EGO_PATHS,
            "prediction_row_budget": self.max_prediction_rows,
            "calibration_row_budget": self.max_calibration_rows,
            "calibration_pending_pair_capacity": 1024,
            "calibration_resolution": "normal policy decision observations; entry/clearance are bracketed raw-step intervals or censored",
            "raw_step_duration_s": 0.1,
            "trajectory_position_speed_source": "current row of public trajectory observation",
            "lane_identity_source": "live TraCI lane IDs for actors in current public observation only",
        }
        path = directory / "task_conflict_timing_manifest.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    def _raw_step(self) -> int | None:
        value = getattr(self.unwrapped, "_raw_steps", None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _live_actor_lane(self, actor_key: str) -> tuple[str | None, str | None, str]:
        if not actor_key.startswith("vehicle:"):
            return None, None, "unsupported_actor_kind"
        actor_id = actor_key.split(":", 1)[1]
        try:
            vehicle = self.unwrapped._connection.vehicle
            road_id = str(vehicle.getRoadID(actor_id))
            lane_id = str(vehicle.getLaneID(actor_id))
        except Exception as exc:
            return None, None, "lane_query_unavailable:" + type(exc).__name__
        if lane_id not in self.static.reachability.lane_edges:
            return road_id, lane_id, "lane_absent_from_static_network"
        if lane_id not in self.static.reachability.permitted_lanes:
            return road_id, lane_id, "lane_not_passenger_permitted"
        return road_id, lane_id, "known"

    def _compose_path(self, lanes: Sequence[str], position: np.ndarray) -> tuple[dict[str, Any] | None, str]:
        points, join_reason = _join_lane_shapes(lanes, self.static.lane_shapes)
        if points is None:
            return None, join_reason
        try:
            current_s, lateral = project_point_to_polyline(position, self.static.lane_shapes[lanes[0]])
        except (ValueError, KeyError):
            return None, "current_lane_projection_failed"
        if lateral > _POSITION_PATH_TOLERANCE_M:
            return None, "current_position_far_from_lane_geometry"
        try:
            _, current_lane_arc = polyline_arclength(self.static.lane_shapes[lanes[0]])
            # Keep the lane geometry static as the actor moves. A dynamic
            # endpoint would defeat the kernel's fixed-path crossing cache.
            points = _trim_path(points, float(current_lane_arc[-1]) + self.max_distance_m)
        except ValueError:
            return None, "path_trimming_failed"
        if len(points) < 2:
            return None, "path_too_short"
        used_lanes: list[str] = []
        cumulative = 0.0
        limit = float(current_lane_arc[-1]) + self.max_distance_m
        for lane_id in lanes:
            shape = self.static.lane_shapes.get(lane_id)
            if shape is None:
                break
            _, arc = polyline_arclength(shape)
            if cumulative <= limit + 1e-6:
                used_lanes.append(lane_id)
            cumulative += float(arc[-1])
        return {
            "points": points,
            "s_current": float(current_s),
            "path_id": ">".join(used_lanes),
            "lane_ids": used_lanes,
        }, "known"

    def _enumerate_paths(
        self,
        start_lane: str,
        position: np.ndarray,
        *,
        route: tuple[str, ...] | None,
        route_cursor: int | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool, str]:
        graph = self.static.reachability
        if start_lane not in graph.permitted_lanes:
            return [], False, "current_lane_not_passenger_permitted"
        if start_lane not in self.static.lane_shapes:
            return [], False, "current_lane_shape_missing"
        candidates: list[dict[str, Any]] = []
        candidate_lane_sequences: set[tuple[str, ...]] = set()
        sequence_endings: dict[tuple[str, ...], str] = {}
        candidate_reasons: list[str] = []
        search_truncated = False
        max_search = max(limit + 1, limit * 3)

        def route_step(next_lane: str, cursor: int) -> int | None:
            edge_id = graph.lane_edges.get(next_lane, "")
            if next_lane in graph.internal_lanes or edge_id.startswith(":"):
                return cursor
            if route is None:
                return cursor
            expected_index = cursor + 1
            if expected_index < len(route) and edge_id == route[expected_index]:
                return expected_index
            return None

        def walk(lanes: tuple[str, ...], cursor: int, depth: int) -> None:
            nonlocal search_truncated
            if len(candidate_lane_sequences) >= max_search:
                search_truncated = True
                return
            points, reason = _join_lane_shapes(lanes, self.static.lane_shapes)
            if points is None:
                candidate_reasons.append(reason)
                return
            try:
                current_s, lateral = project_point_to_polyline(position, self.static.lane_shapes[start_lane])
                _, cumulative = polyline_arclength(points)
            except ValueError:
                candidate_reasons.append("geometry_projection_failed")
                return
            if lateral > _POSITION_PATH_TOLERANCE_M:
                candidate_reasons.append("current_position_far_from_lane_geometry")
                return
            lane_start_shape = self.static.lane_shapes[start_lane]
            _, start_lane_arc = polyline_arclength(lane_start_shape)
            fixed_horizon_from_start = float(start_lane_arc[-1]) + self.max_distance_m
            reached_lookahead = float(cumulative[-1]) >= fixed_horizon_from_start
            successors = sorted(graph.successors.get(lanes[-1], ()))
            static_successors = [
                nxt for nxt in successors
                if nxt in graph.permitted_lanes and nxt not in lanes
            ]
            allowed: list[tuple[str, int]] = []
            if depth < 24 and not reached_lookahead:
                for nxt in successors:
                    if nxt not in graph.permitted_lanes or nxt in lanes:
                        continue
                    next_cursor = route_step(nxt, cursor)
                    if next_cursor is None:
                        continue
                    allowed.append((nxt, next_cursor))
            if reached_lookahead or not allowed or depth >= 24:
                candidate_lane_sequences.add(lanes)
                if reached_lookahead:
                    ending = "max_distance_reached"
                elif depth >= 24:
                    ending = "depth_limit_truncated"
                    search_truncated = True
                elif route is not None and cursor >= len(route) - 1:
                    ending = "task_route_end"
                elif not static_successors:
                    ending = "static_dead_end_or_cycle"
                elif route is not None:
                    ending = "route_filtered_successors"
                else:
                    ending = "candidate_graph_dead_end"
                sequence_endings[lanes] = ending
                return
            for nxt, next_cursor in allowed:
                if len(candidate_lane_sequences) >= max_search:
                    search_truncated = True
                    break
                walk((*lanes, nxt), next_cursor, depth + 1)

        initial_cursor = int(route_cursor or 0)
        walk((start_lane,), initial_cursor, 0)
        sequences = sorted(candidate_lane_sequences)
        if len(sequences) > limit:
            search_truncated = True
            sequences = sequences[:limit]
        for lane_ids in sequences:
            path, reason = self._compose_path(lane_ids, position)
            if path is None:
                candidate_reasons.append(reason)
                continue
            path["termination_reason"] = sequence_endings.get(lane_ids, "candidate_search_limit")
            candidates.append(path)
        if not candidates:
            reason = candidate_reasons[0] if candidate_reasons else "no_static_continuation"
            return [], search_truncated, reason
        return candidates, search_truncated, "known"

    def _ego_paths(self, road_id: str, lane_id: str, position: np.ndarray) -> tuple[list, dict]:
        vehicle = self.unwrapped._connection.vehicle
        ego_id = str(self.unwrapped.specification.ego_id)
        try:
            route = tuple(str(edge) for edge in vehicle.getRoute(ego_id))
            route_index = int(vehicle.getRouteIndex(ego_id))
        except Exception as exc:
            return [], {"status": "ego_route_unavailable:" + type(exc).__name__, "route": None,
                        "route_index": None, "candidate_count": 0, "truncated": False}
        if not route or route_index < 0 or route_index >= len(route):
            return [], {"status": "ego_route_context_unknown", "route": list(route),
                        "route_index": route_index, "candidate_count": 0, "truncated": False}
        lane_edge = self.static.reachability.lane_edges.get(lane_id)
        if not road_id or lane_edge is None or lane_edge != road_id:
            return [], {"status": "ego_road_lane_mismatch", "route": list(route),
                        "route_index": route_index, "current_road_id": road_id,
                        "current_lane_id": lane_id, "candidate_count": 0, "truncated": False}
        is_internal = lane_id in self.static.reachability.internal_lanes or (lane_edge or "").startswith(":")
        cursor = route_index
        if not is_internal:
            if road_id != route[cursor] or lane_edge != road_id:
                return [], {"status": "ego_route_lane_mismatch", "route": list(route),
                            "route_index": route_index, "candidate_count": 0, "truncated": False}
        paths, truncated, reason = self._enumerate_paths(
            lane_id, position, route=route, route_cursor=cursor, limit=DEFAULT_MAX_EGO_PATHS
        )
        status = "known_route_geometry" if reason == "known" else "ego_" + reason
        endings = [str(path.get("termination_reason", "unknown")) for path in paths]
        return paths, {
            "status": status,
            "route": list(route),
            "route_index": route_index,
            "current_road_id": road_id,
            "current_lane_id": lane_id,
            "candidate_count": len(paths),
            "truncated": bool(truncated),
            "candidate_termination_reasons": endings,
            "candidate_termination_counts": {key: endings.count(key) for key in sorted(set(endings))},
            "candidate_enumeration_complete": not bool(truncated),
            "known_prefix_not_full_route_reachability": any(
                reason not in {"max_distance_reached", "task_route_end"} for reason in endings
            ) or bool(truncated),
        }

    def _social_paths(self, road_id: str, lane_id: str, position: np.ndarray) -> tuple[list, dict]:
        lane_edge = self.static.reachability.lane_edges.get(lane_id)
        if lane_edge != road_id:
            return [], {"status": "social_road_lane_mismatch", "current_road_id": road_id,
                        "current_lane_id": lane_id, "candidate_count": 0, "truncated": False}
        paths, truncated, reason = self._enumerate_paths(
            lane_id, position, route=None, route_cursor=0, limit=self.max_social_paths
        )
        status = "known_static_lane_candidates" if reason == "known" else "social_" + reason
        endings = [str(path.get("termination_reason", "unknown")) for path in paths]
        return paths, {
            "status": status,
            "current_road_id": road_id,
            "current_lane_id": lane_id,
            "candidate_count": len(paths),
            "truncated": bool(truncated),
            "candidate_termination_reasons": endings,
            "candidate_termination_counts": {key: endings.count(key) for key in sorted(set(endings))},
            "candidate_enumeration_complete": not bool(truncated),
            "known_prefix_not_full_route_reachability": any(
                reason != "max_distance_reached" for reason in endings
            ) or bool(truncated),
        }

    def _map_coverage(self, observation: Mapping[str, Any], actor_index: int, candidates: Sequence[Mapping[str, Any]]) -> dict:
        map_state = observation.get("map")
        if map_state is None:
            return {"valid_point_count": None, "candidate_geometry_point_count": None,
                    "candidate_geometry_coverage_fraction": None, "status": "map_absent"}
        array = np.asarray(map_state)
        spec = getattr(self.unwrapped, "specification", None)
        paths_per_actor = int(getattr(spec, "map_paths_per_actor", 0))
        start = actor_index * paths_per_actor
        block = array[start:start + paths_per_actor]
        if block.ndim != 3 or block.size == 0 or block.shape[-1] < 2:
            return {"valid_point_count": None, "candidate_geometry_point_count": None,
                    "candidate_geometry_coverage_fraction": None, "status": "map_shape_unavailable"}
        xy = np.asarray(block[:, :, :2], dtype=np.float64).reshape(-1, 2)
        if block.shape[-1] >= 5:
            flags = np.asarray(block[:, :, 3:5], dtype=np.float64).reshape(-1, 2)
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(flags).all(axis=1) & np.any(flags > 0.5, axis=1)
        else:
            valid = np.isfinite(xy).all(axis=1) & np.any(np.abs(xy) > 1e-8, axis=1)
        points = xy[valid]
        if not len(points):
            return {"valid_point_count": 0, "candidate_geometry_point_count": 0,
                    "candidate_geometry_coverage_fraction": None, "status": "no_valid_original_map_points"}
        covered = 0
        candidate_lines = [np.asarray(item["points"], dtype=np.float64) for item in candidates]
        for point in points:
            nearest = math.inf
            for line in candidate_lines:
                try:
                    _, lateral = project_point_to_polyline(point, line)
                except ValueError:
                    continue
                nearest = min(nearest, lateral)
            if nearest <= _MAP_POINT_TOLERANCE_M:
                covered += 1
        return {
            "valid_point_count": int(len(points)),
            "candidate_geometry_point_count": int(covered),
            "candidate_geometry_coverage_fraction": float(covered / len(points)) if candidate_lines else None,
            "status": "candidate_only_not_full_network_coverage",
            "original_map_coverage_is_complete": False,
        }

    def _capture(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        output = dict(observation)
        trajectory = np.asarray(observation["trajectory"])
        actor_count = int(trajectory.shape[0])
        actor_keys_value = getattr(self.unwrapped, "_last_observation_actor_keys", None)
        actor_keys = tuple(str(value) for value in actor_keys_value) if actor_keys_value is not None else ()
        matrix: list[dict[str, Any] | None] = [None] * actor_count
        actor_rows: list[dict[str, Any]] = []
        ego_meta: dict[str, Any] = {"status": "ego_row_missing", "candidate_count": 0, "truncated": False}
        if not actor_keys or len(actor_keys) > actor_count:
            for index in range(actor_count):
                actor_rows.append({"actor_index": index, "key": actor_keys[index] if index < len(actor_keys) else None,
                                   "geometry_status": "observation_actor_order_unavailable", "paths": []})
        else:
            for actor_index in range(actor_count):
                actor_key = actor_keys[actor_index] if actor_index < len(actor_keys) else None
                if actor_key is None:
                    actor_rows.append({"actor_index": actor_index, "key": None,
                                       "geometry_status": "padding_not_observed", "paths": []})
                    continue
                expected_ego_key = f"vehicle:{getattr(self.unwrapped.specification, 'ego_id', '')}"
                if actor_index == 0 and actor_key != expected_ego_key:
                    actor_rows.append({"actor_index": actor_index, "key": actor_key,
                                       "geometry_status": "ego_actor_order_mismatch", "paths": []})
                    continue
                state = _current_observation_row(self.unwrapped, trajectory, actor_index)
                if state is None:
                    actor_rows.append({"actor_index": actor_index, "key": actor_key,
                                       "geometry_status": "current_trajectory_state_unavailable", "paths": []})
                    continue
                position = state[:2].copy()
                speed = float(np.linalg.norm(state[3:5]))
                road_id, lane_id, lane_status = self._live_actor_lane(actor_key)
                if lane_status != "known":
                    row = {"actor_index": actor_index, "key": actor_key, "speed_mps": speed,
                           "position_xy": position.tolist(), "road_id": road_id, "lane_id": lane_id,
                           "geometry_status": lane_status, "paths": []}
                    actor_rows.append(row)
                    matrix[actor_index] = {"key": actor_key, "speed_mps": speed, "paths": [],
                                           "geometry_status": lane_status}
                    continue
                if actor_index == 0:
                    paths, path_meta = self._ego_paths(road_id or "", lane_id or "", position)
                    ego_paths, ego_meta = paths, path_meta
                else:
                    paths, path_meta = self._social_paths(road_id or "", lane_id or "", position)
                geometry_status = str(path_meta["status"])
                row = {
                    "actor_index": actor_index,
                    "key": actor_key,
                    "raw_step": self._raw_step(),
                    "speed_mps": speed,
                    "position_xy": position.tolist(),
                    "road_id": road_id,
                    "lane_id": lane_id,
                    "geometry_status": geometry_status,
                    "path_metadata": path_meta,
                    "paths": paths,
                    "map_coverage": self._map_coverage(observation, actor_index, paths),
                }
                actor_rows.append(row)
                matrix[actor_index] = {
                    "key": actor_key,
                    "speed_mps": speed,
                    "paths": paths if actor_index == 0 else paths,
                    "geometry_status": geometry_status,
                }
        features, detail = compute_route_conflict_timing(
            matrix,
            horizon_s=self.horizon_s,
            max_distance_m=self.max_distance_m,
            length_m=DEFAULT_PROXY_LENGTH_M,
            width_m=DEFAULT_PROXY_WIDTH_M,
        )
        output[self.KEY] = features
        raw_step = self._raw_step()
        frame = {
            "protocol": PROTOCOL,
            "phase": self.phase,
            "episode_index": self._episode_index,
            "decision_index": self._decision_index,
            "raw_step": raw_step,
            "episode_seed": self._episode_seed,
            "actor_keys": list(actor_keys) + [None] * max(0, actor_count - len(actor_keys)),
            "actors": actor_rows,
            "features": features,
            "detail": detail,
            "ego_path_metadata": ego_meta,
            "source_observation_contract": getattr(
                getattr(self.unwrapped, "specification", None), "source_observation_contract", "unknown"
            ),
        }
        return output, frame

    @staticmethod
    def _compact_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
        compact = json.loads(json.dumps(_json_safe(detail), allow_nan=False))
        for row in compact.get("actors", []):
            selected = row.get("selected")
            if isinstance(selected, dict):
                selected.pop("ego_path_points", None)
                selected.pop("foe_path_points", None)
        return compact

    @staticmethod
    def _compact_actor_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for row in rows:
            path_meta = row.get("path_metadata") or {}
            compact.append(
                {
                    key: row.get(key)
                    for key in (
                        "actor_index", "key", "raw_step", "speed_mps", "position_xy",
                        "road_id", "lane_id", "geometry_status", "map_coverage",
                    )
                }
                | {
                    "candidate_paths": [
                        {
                            "path_id": item.get("path_id"),
                            "lane_ids": item.get("lane_ids"),
                            "s_current": item.get("s_current"),
                            "static_point_count": int(len(item.get("points", ()))),
                            "termination_reason": item.get("termination_reason"),
                        }
                        for item in row.get("paths", ())
                    ],
                    "path_candidate_count": path_meta.get("candidate_count"),
                    "path_candidates_truncated": bool(path_meta.get("truncated", False)),
                    "candidate_termination_reasons": path_meta.get("candidate_termination_reasons", []),
                    "candidate_termination_counts": path_meta.get("candidate_termination_counts", {}),
                    "candidate_enumeration_complete": path_meta.get("candidate_enumeration_complete"),
                    "known_prefix_not_full_route_reachability": path_meta.get(
                        "known_prefix_not_full_route_reachability"
                    ),
                }
            )
        return compact

    @staticmethod
    def _action_array(action: Any) -> Any:
        try:
            return _json_safe(np.asarray(action).tolist())
        except Exception:
            return _json_safe(action)

    def _enqueue_calibration(self, frame: Mapping[str, Any], action: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        by_index = {int(row["actor_index"]): row for row in frame.get("actors", [])}
        detail = frame.get("detail", {})
        for relation in detail.get("actors", []):
            if not relation.get("relation_valid") or not isinstance(relation.get("selected"), Mapping):
                continue
            selected = relation["selected"]
            actor_index = int(relation["actor_index"])
            social_row = by_index.get(actor_index, {})
            ego_row = by_index.get(0, {})
            if not social_row.get("key") or not ego_row.get("key"):
                continue
            prediction = {
                "phase": self.phase,
                "episode_index": frame["episode_index"],
                "decision_index": frame["decision_index"],
                "raw_step_pre_action": frame.get("raw_step"),
                "ego_key": ego_row["key"],
                "foe_key": social_row["key"],
                "foe_actor_index": actor_index,
                "ego_actor_index": 0,
                "episode_seed": frame.get("episode_seed"),
                "prediction_join_key": {
                    "phase": self.phase,
                    "episode_index": frame["episode_index"],
                    "episode_seed": frame.get("episode_seed"),
                    "decision_index": frame["decision_index"],
                    "raw_step_pre_action": frame.get("raw_step"),
                    "ego_actor_index": 0,
                    "ego_key": ego_row["key"],
                    "foe_actor_index": actor_index,
                    "foe_key": social_row["key"],
                },
                "action_policy": self._action_array(action),
                "predicted": {key: value for key, value in selected.items()
                              if key not in {"ego_path_points", "foe_path_points"}},
                "geometry_status": social_row.get("geometry_status"),
                "ego_geometry_status": ego_row.get("geometry_status"),
                "ego_candidate_truncated": bool(ego_row.get("path_metadata", {}).get("truncated", False)),
                "foe_candidate_truncated": bool(social_row.get("path_metadata", {}).get("truncated", False)),
                "candidate_count_ego": ego_row.get("path_metadata", {}).get("candidate_count"),
                "candidate_count_foe": social_row.get("path_metadata", {}).get("candidate_count"),
                "candidate_support_fraction_presented_pairs_only": relation.get("candidate_support_fraction"),
            }
            predictions.append(prediction)
            if len(self._pending) < 1024:
                self._pending.append({
                    "prediction": prediction,
                    "selected": dict(selected),
                    "last": {
                        "ego": self._initial_progress(ego_row, selected, "ego"),
                        "foe": self._initial_progress(social_row, selected, "foe"),
                    },
                    "entry": {"ego": None, "foe": None},
                    "clearance": {"ego": None, "foe": None},
                    "status": {"ego": "tracking", "foe": "tracking"},
                    "deadline_raw_step": int(frame.get("raw_step") or 0) + int(math.ceil(self.horizon_s / 0.1)),
                    "dropout_samples": {"ego": 0, "foe": 0},
                })
            else:
                self._pending_pairs_dropped += 1
        return predictions

    @staticmethod
    def _initial_progress(actor_row: Mapping[str, Any], selected: Mapping[str, Any], side: str) -> dict[str, Any]:
        path_points = selected.get(side + "_path_points")
        arc = _as_finite_float(selected.get(side + "_current_s"))
        entry = _as_finite_float(selected.get(side + "_entry_arc_m"))
        exit_value = _as_finite_float(selected.get(side + "_exit_arc_m"))
        if path_points is not None and actor_row.get("position_xy") is not None:
            try:
                projected_s, lateral = project_point_to_polyline(actor_row["position_xy"], path_points)
                if arc is None or lateral <= 5.0:
                    arc = projected_s
            except (ValueError, TypeError):
                pass
        return {
            "raw_step": actor_row.get("raw_step"),
            "s": arc,
            "entry_arc_m": entry,
            "exit_arc_m": exit_value,
            "entry_left_censored": bool(arc is not None and entry is not None and arc >= entry),
            "clearance_left_censored": bool(arc is not None and exit_value is not None and arc >= exit_value),
        }

    def _advance_calibration(self, frame: Mapping[str, Any], *, terminal: bool) -> None:
        current_raw = int(frame.get("raw_step") or 0)
        rows = {str(row.get("key")): row for row in frame.get("actors", []) if row.get("key")}
        remaining: list[dict[str, Any]] = []
        for item in self._pending:
            prediction = item["prediction"]
            selected = item["selected"]
            finish_reason = None
            for side, key_field in (("ego", "ego_key"), ("foe", "foe_key")):
                actor_row = rows.get(str(prediction[key_field]))
                if actor_row is None or actor_row.get("position_xy") is None:
                    item["dropout_samples"][side] += 1
                    continue
                path_points = selected.get(side + "_path_points")
                if path_points is None:
                    item["status"][side] = "path_geometry_unavailable"
                    continue
                try:
                    s_value, lateral = project_point_to_polyline(actor_row["position_xy"], path_points)
                except (ValueError, TypeError):
                    item["status"][side] = "projection_unavailable"
                    continue
                lane_ids = set(str(value) for value in selected.get(side + "_lane_ids", []))
                lane_id = actor_row.get("lane_id")
                lane_mismatch = bool(lane_id and lane_ids and str(lane_id) not in lane_ids)
                if lateral > 5.0 or lane_mismatch:
                    item["status"][side] = "route_or_candidate_path_deviation"
                    finish_reason = "candidate_path_deviation"
                    continue
                previous = item["last"].get(side, {})
                previous_s = _as_finite_float(previous.get("s"))
                previous_raw = previous.get("raw_step")
                entry_arc = _as_finite_float(selected.get(side + "_entry_arc_m"))
                exit_arc = _as_finite_float(selected.get(side + "_exit_arc_m"))
                for boundary, boundary_arc, target in (
                    ("entry", entry_arc, item["entry"]),
                    ("clearance", exit_arc, item["clearance"]),
                ):
                    if target.get(side) is not None or boundary_arc is None or s_value < boundary_arc:
                        continue
                    left_censored = bool(previous.get(boundary + "_left_censored", False))
                    if previous_s is None or previous_raw is None or left_censored:
                        interval = {"left_censored_at_prediction": True,
                                    "lower_raw_step": prediction.get("raw_step_pre_action"),
                                    "upper_raw_step": current_raw}
                    else:
                        interval = {"left_censored_at_prediction": False,
                                    "lower_raw_step": int(previous_raw),
                                    "upper_raw_step": current_raw}
                    target[side] = interval
                item["last"][side] = {
                    "raw_step": current_raw,
                    "s": float(s_value),
                    "entry_left_censored": bool(entry_arc is not None and s_value >= entry_arc),
                    "clearance_left_censored": bool(exit_arc is not None and s_value >= exit_arc),
                }
            done_both = all(item["clearance"].get(side) is not None for side in ("ego", "foe"))
            expired = current_raw >= int(item["deadline_raw_step"])
            if done_both:
                finish_reason = finish_reason or "both_corridors_cleared"
            elif terminal:
                finish_reason = finish_reason or "episode_terminal_right_censored"
            elif expired:
                finish_reason = finish_reason or "prediction_horizon_right_censored"
            if finish_reason is not None:
                self._write_calibration({
                    "schema": "conflict_timing_calibration_v1",
                    "protocol": PROTOCOL,
                    "prediction": prediction,
                    "outcome_status": finish_reason,
                    "entry_intervals": item["entry"],
                    "clearance_intervals": item["clearance"],
                    "actor_path_status": item["status"],
                    "unobserved_decision_samples": item["dropout_samples"],
                    "calibration_resolution": "decision samples bracket raw-step intervals; no exact subdecision crossing time is claimed",
                    "constant_speed_prediction_is_action_conditioned": True,
                    "absence_or_na_is_not_safety": True,
                })
            else:
                remaining.append(item)
        self._pending = remaining

    def _write_prediction(self, row: Mapping[str, Any]) -> None:
        if self._prediction_writer is None:
            return
        if self._prediction_rows_written >= self.max_prediction_rows:
            self._prediction_rows_dropped += 1
            return
        if self._prediction_writer.write(row):
            self._prediction_rows_written += 1
        else:
            self._prediction_rows_dropped += 1

    def _write_calibration(self, row: Mapping[str, Any]) -> None:
        if self._calibration_writer is None:
            return
        if self._calibration_rows_written >= self.max_calibration_rows:
            self._calibration_rows_dropped += 1
            return
        if self._calibration_writer.write(row):
            self._calibration_rows_written += 1
        else:
            self._calibration_rows_dropped += 1

    def reset(self, **kwargs):
        if self._pending:
            for item in self._pending:
                self._write_calibration({
                    "schema": "conflict_timing_calibration_v1",
                    "protocol": PROTOCOL,
                    "prediction": item["prediction"],
                "outcome_status": "episode_reset_right_censored",
                    "entry_intervals": item["entry"],
                    "clearance_intervals": item["clearance"],
                    "actor_path_status": item["status"],
                    "unobserved_decision_samples": item["dropout_samples"],
                    "absence_or_na_is_not_safety": True,
                })
                self._pending_pairs_right_censored += 1
            self._pending.clear()
        observation, info = self.env.reset(**kwargs)
        self._episode_index += 1
        self._episode_seed = kwargs.get("seed")
        if self._episode_seed is None:
            for key in ("episode_seed", "seed", "evaluation_seed"):
                if info.get(key) is not None:
                    self._episode_seed = info[key]
                    break
        self._decision_index = 0
        observation, frame = self._capture(observation)
        self._last_frame = frame
        info = dict(info)
        info.update({
            "conflict_timing_protocol": PROTOCOL,
            "conflict_timing_valid_relation_count": int(frame["detail"].get("valid_relation_count", 0)),
            "conflict_timing_valid_joint_timing_count": int(frame["detail"].get("valid_joint_timing_count", 0)),
            "conflict_timing_social_future_routes_read": False,
        })
        return observation, info

    def step(self, action):
        pre_frame = self._last_frame
        raw_step_before = self._raw_step()
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, next_frame = self._capture(observation)
        info = dict(info)
        next_frame["terminal"] = bool(terminated or truncated)
        if pre_frame is not None:
            predictions = self._enqueue_calibration(pre_frame, action)
            self._advance_calibration(next_frame, terminal=bool(terminated or truncated))
            self._write_prediction({
                "schema": "conflict_timing_prediction_v1",
                "protocol": PROTOCOL,
                "phase": self.phase,
                "episode_index": pre_frame["episode_index"],
                "episode_seed": pre_frame.get("episode_seed"),
                "decision_index": pre_frame["decision_index"],
                "social_future_routes_read": False,
                "raw_step_pre_action": raw_step_before,
                "raw_step_after_action": self._raw_step(),
                "actor_keys_in_trajectory_order": pre_frame["actor_keys"],
                "ego_path_metadata": pre_frame["ego_path_metadata"],
                "actor_geometry_and_map_coverage": self._compact_actor_rows(pre_frame["actors"]),
                "pre_action_features": pre_frame["features"],
                "pre_action_predictions": self._compact_detail(pre_frame["detail"]),
                "action_policy": self._action_array(action),
                "next_observed_actor_keys": next_frame["actor_keys"],
                "next_observed_actor_states": [
                    {key: row.get(key) for key in ("actor_index", "key", "position_xy", "speed_mps", "road_id", "lane_id", "geometry_status")}
                    for row in next_frame["actors"]
                ],
                "next_observation_features": next_frame["features"],
                "terminal": bool(terminated or truncated),
                "terminal_info": {
                    key: info.get(key) for key in (
                        "success", "collision", "off_route", "timeout", "raw_sumo_arrived",
                        "raw_sumo_collision", "raw_max_time", "terminal_outcome_protocol"
                    ) if key in info
                },
                "prediction_pair_count": len(predictions),
            })
        self._decision_index += 1
        self._last_frame = next_frame
        info.update({
            "conflict_timing_protocol": PROTOCOL,
            "conflict_timing_valid_relation_count": int(next_frame["detail"].get("valid_relation_count", 0)),
            "conflict_timing_valid_joint_timing_count": int(next_frame["detail"].get("valid_joint_timing_count", 0)),
            "conflict_timing_social_future_routes_read": False,
        })
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending:
            for item in self._pending:
                self._write_calibration({
                    "schema": "conflict_timing_calibration_v1",
                    "protocol": PROTOCOL,
                    "prediction": item["prediction"],
                    "outcome_status": "recorder_closed_right_censored",
                    "entry_intervals": item["entry"],
                    "clearance_intervals": item["clearance"],
                    "actor_path_status": item["status"],
                    "unobserved_decision_samples": item["dropout_samples"],
                    "absence_or_na_is_not_safety": True,
                })
                self._pending_pairs_right_censored += 1
            self._pending.clear()
        for writer in (self._prediction_writer, self._calibration_writer):
            if writer is not None:
                writer.close()
        directory = self._prediction_writer.path.parent if self._prediction_writer is not None else None
        if directory is not None:
            summary = {
                "protocol": PROTOCOL,
                "phase": self.phase,
                "prediction_rows_written": self._prediction_rows_written,
                "prediction_rows_dropped_at_budget_or_io": self._prediction_rows_dropped,
                "calibration_rows_written": self._calibration_rows_written,
                "calibration_rows_dropped_at_budget_or_io": self._calibration_rows_dropped,
                "pending_predictions_right_censored_on_close": self._pending_pairs_right_censored,
                "calibration_pairs_dropped_at_pending_capacity": self._pending_pairs_dropped,
                "prediction_sampling": "one compact row per policy decision until budget; separate pair-level calibration outcomes",
                "prediction_budget": self.max_prediction_rows,
                "calibration_budget": self.max_calibration_rows,
                "unknown_or_unobserved_is_not_safety": True,
            }
            with (directory / "task_conflict_timing_summary.json").open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        self.env.close()
