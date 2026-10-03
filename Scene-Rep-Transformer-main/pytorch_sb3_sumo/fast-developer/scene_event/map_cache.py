"""Public SUMO lane graph and movement-zone cache for SceneEventEnv."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sumolib
from shapely.geometry import LineString, Polygon
from shapely.ops import substring, unary_union

from envs.sumo.topology_graph import (
    _all_lanes, _connection_lane_id, _connection_movement_shape, _polylines_cross,
)
from envs.sumo.topology_graph_v2 import (
    CONFLICT, LEFT, MERGE, PREDECESSOR, RIGHT, SUCCESSOR,
    build_topology_graph_v2,
)
from .geometry import polyline_zone_interval


@dataclass(frozen=True)
class PublicMapCache:
    public_map: dict[str, np.ndarray]
    zone_geometries: tuple[Any, ...]
    lane_ids: tuple[str, ...]
    lane_lengths_m: np.ndarray
    lane_successors: tuple[tuple[int, ...], ...]
    lane_lines: tuple[Any, ...]
    route_edges: tuple[str, ...]
    network_path: str
    ego_route_path: str
    network_sha256: str
    fingerprint: str
    build_metadata: dict[str, Any]

    @classmethod
    def from_spec(cls, specification: Any, *, lane_sample_points: int = 10) -> "PublicMapCache":
        network_path = Path(specification.network_path).resolve()
        route_path = Path(specification.ego_route_path).resolve()
        net = sumolib.net.readNet(str(network_path), withInternal=True)
        lanes = _all_lanes(net)
        lane_ids = tuple(str(lane.getID()) for lane in lanes)
        lane_index = {lane_id: index for index, lane_id in enumerate(lane_ids)}
        lane_count = len(lane_ids)
        if lane_count <= 0:
            raise ValueError(f"No public lanes found in {network_path}")

        # Every relation is directed and connects a lane pair. This upper bound
        # is derived from actual L, replacing the old 64/512 fixed capacities.
        edge_capacity = max(1, 6 * lane_count * lane_count)
        graph, topology_info = build_topology_graph_v2(
            network_path,
            max_nodes=lane_count,
            max_edges=edge_capacity,
            lane_sample_points=lane_sample_points,
            include_merge=True,
            return_info=True,
        )
        if topology_info.lane_ids != lane_ids:
            raise RuntimeError("topology builder lane order differs from the public map parser")
        node_mask = np.asarray(graph.node_mask, dtype=bool)
        edge_mask = np.asarray(graph.edge_mask, dtype=bool)
        if not node_mask.all():
            raise RuntimeError("actual lane array was padded or truncated unexpectedly")
        lane_points = np.asarray(graph.lane_points[node_mask], dtype=np.float32).copy()
        lane_attrs = np.asarray(graph.lane_attrs[node_mask], dtype=np.float32).copy()
        edge_index = np.asarray(graph.edge_index[:, edge_mask], dtype=np.int64).copy()
        edge_type = np.asarray(graph.edge_type[edge_mask], dtype=np.int64).copy()
        if edge_index.shape[1] != topology_info.valid_edge_count:
            raise RuntimeError("typed lane-edge capacity failed to preserve every edge")
        if edge_index.size and int(edge_index.max()) >= lane_count:
            raise RuntimeError("lane edge references a lane outside the actual map")

        lane_by_id = {str(lane.getID()): lane for lane in lanes}
        relation_provenance = _relation_provenance(
            net, lane_by_id, topology_info.merge_groups
        )
        lane_lengths = np.asarray([float(lane_by_id[key].getLength()) for key in lane_ids], dtype=np.float32)
        successors_mutable: list[set[int]] = [set() for _ in lane_ids]
        for source, target in edge_index[:, edge_type == SUCCESSOR].T:
            successors_mutable[int(source)].add(int(target))
        successors = tuple(tuple(sorted(values)) for values in successors_mutable)

        route_edges, preferred_start_lane = _ego_route(route_path, str(specification.ego_id))
        ego_route_lane_mask = _route_compatible_lanes(
            lane_ids, lane_by_id, successors, route_edges, preferred_start_lane
        )
        lane_lines = tuple(_lane_line(lane_by_id[lane_id]) for lane_id in lane_ids)
        zones, zone_types, zone_build_meta = _compile_zones(
            lane_lines, lane_by_id, edge_index, edge_type, lane_ids,
            ego_route_lane_mask=ego_route_lane_mask,
            relation_provenance=relation_provenance,
        )
        zone_count = len(zones)
        max_vertices = max((len(poly.exterior.coords) - 1 for poly in zones), default=0)
        polygons = np.zeros((zone_count, max_vertices, 2), dtype=np.float32)
        vertex_valid = np.zeros((zone_count, max_vertices), dtype=bool)
        lane_zone = np.zeros((lane_count, zone_count, 2), dtype=np.float32)
        lane_zone_valid = np.zeros((lane_count, zone_count), dtype=bool)
        lane_zone_distance = np.full((lane_count, zone_count), np.inf, dtype=np.float32)
        public_zone_geometries: list[Any] = []
        for zone_index, polygon in enumerate(zones):
            points = np.asarray(list(polygon.exterior.coords)[:-1], dtype=np.float32)
            if len(points) == 0:
                raise RuntimeError("movement zone has no exterior vertices")
            polygons[zone_index, :len(points)] = points
            vertex_valid[zone_index, :len(points)] = True
            # The numeric policy/map contract stores exterior-only polygons.
            # Use that same filled proxy for collector labels to avoid a hidden
            # hole-aware target that the policy cannot observe.
            zone_proxy = Polygon(points)
            public_zone_geometries.append(zone_proxy)
            for lane_index_value, line in enumerate(lane_lines):
                lane_zone_distance[lane_index_value, zone_index] = np.float32(line.distance(zone_proxy))
                interval = polyline_zone_interval(line, zone_proxy)
                if interval is not None:
                    lane_zone[lane_index_value, zone_index] = interval
                    lane_zone_valid[lane_index_value, zone_index] = True

        public_map = {
            "lane_points_xy": lane_points,
            "lane_attrs": lane_attrs,
            "lane_valid": np.ones((lane_count,), dtype=bool),
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_valid": np.ones((edge_index.shape[1],), dtype=bool),
            "ego_route_lane_mask": ego_route_lane_mask.astype(bool, copy=False),
            "zone_polygons_xy": polygons,
            "zone_vertex_valid": vertex_valid,
            "lane_zone_s_m": lane_zone,
            "lane_zone_valid": lane_zone_valid,
            "lane_zone_distance_m": lane_zone_distance,
            "zone_type": zone_types.astype(np.int64, copy=False),
            "zone_valid": np.ones((zone_count,), dtype=bool),
        }
        if public_map["ego_route_lane_mask"].shape != (lane_count,) or not public_map["ego_route_lane_mask"].any():
            raise RuntimeError("ego public route did not map to any legal lane")
        for name, array in public_map.items():
            if not isinstance(array, np.ndarray) or array.dtype == object:
                raise TypeError(f"public map field {name} is not a numeric ndarray")
            array.setflags(write=False)
        network_sha = _sha256(network_path)
        digest = hashlib.sha256()
        digest.update(network_sha.encode("ascii"))
        digest.update(json.dumps(route_edges).encode("utf-8"))
        for name in sorted(public_map):
            value = public_map[name]
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
        meta = {
            "lane_count": lane_count,
            "typed_edge_count": int(edge_index.shape[1]),
            "zone_count": zone_count,
            "route_edges": list(route_edges),
            "route_lane_count": int(ego_route_lane_mask.sum()),
            "topology_relation_counts": dict(topology_info.relation_counts),
            "zone_geometry": zone_build_meta,
        }
        return cls(
            public_map=public_map,
            zone_geometries=tuple(public_zone_geometries),
            lane_ids=lane_ids,
            lane_lengths_m=lane_lengths,
            lane_successors=successors,
            lane_lines=lane_lines,
            route_edges=route_edges,
            network_path=str(network_path),
            ego_route_path=str(route_path),
            network_sha256=network_sha,
            fingerprint=digest.hexdigest(),
            build_metadata=meta,
        )


def _ego_route(path: Path, ego_id: str) -> tuple[tuple[str, ...], str | None]:
    root = ET.parse(path).getroot()
    vehicle = next((item for item in root.findall("vehicle") if item.attrib.get("id") == ego_id), None)
    if vehicle is None:
        raise ValueError(f"ego vehicle {ego_id!r} is missing from {path}")
    route_ref = vehicle.attrib.get("route")
    route_elem = vehicle.find("route")
    if route_elem is None and route_ref:
        route_elem = next((item for item in root.findall("route") if item.attrib.get("id") == route_ref), None)
    if route_elem is None or not route_elem.attrib.get("edges"):
        raise ValueError(f"ego route edges are missing from {path}")
    edges = tuple(route_elem.attrib["edges"].split())
    start_lane = None
    depart_lane = vehicle.attrib.get("departLane")
    if depart_lane is not None and depart_lane.isdigit():
        start_lane = f"{edges[0]}_{int(depart_lane)}"
    return edges, start_lane


def _route_compatible_lanes(lane_ids: tuple[str, ...], lane_by_id: dict[str, Any],
                            successors: tuple[tuple[int, ...], ...], route_edges: tuple[str, ...],
                            preferred_start_lane: str | None) -> np.ndarray:
    lane_index = {lane_id: index for index, lane_id in enumerate(lane_ids)}
    starts = [i for i, lane_id in enumerate(lane_ids)
              if str(lane_by_id[lane_id].getEdge().getID()) == route_edges[0]]
    if preferred_start_lane in lane_index:
        starts = [lane_index[preferred_start_lane], *[i for i in starts if i != lane_index[preferred_start_lane]]]
    seen: set[tuple[int, int]] = set()
    accepted: set[int] = set()
    pending = [(index, 0) for index in starts]
    while pending:
        current, route_index = pending.pop()
        state = (current, route_index)
        if state in seen:
            continue
        seen.add(state)
        lane_id = lane_ids[current]
        edge_id = str(lane_by_id[lane_id].getEdge().getID())
        if edge_id.startswith(":"):
            accepted.add(current)
        elif route_index < len(route_edges) and edge_id == route_edges[route_index]:
            accepted.add(current)
        else:
            continue
        if edge_id == route_edges[-1]:
            continue
        for nxt in successors[current]:
            next_edge = str(lane_by_id[lane_ids[nxt]].getEdge().getID())
            if next_edge.startswith(":"):
                pending.append((nxt, route_index))
            elif route_index + 1 < len(route_edges) and next_edge == route_edges[route_index + 1]:
                pending.append((nxt, route_index + 1))
    mask = np.zeros((len(lane_ids),), dtype=bool)
    if accepted:
        mask[np.fromiter(sorted(accepted), dtype=np.int64)] = True
    return mask


def _lane_line(lane: Any) -> LineString:
    points = [(float(p[0]), float(p[1])) for p in lane.getShape()]
    if len(points) < 2:
        raise ValueError(f"lane {lane.getID()} has fewer than two public centerline points")
    line = LineString(points)
    if line.length <= 1e-6:
        raise ValueError(f"lane {lane.getID()} has zero-length centerline")
    return line


def _connection_record(connection: Any, *, node_id: str, node_type: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "request_index": int(connection.getJunctionIndex()),
        "from_lane": str(connection.getFromLane().getID()),
        "via_lane": str(connection.getViaLaneID() or ""),
        "to_lane": str(connection.getToLane().getID()),
    }


def _relation_provenance(net: Any, lane_by_id: dict[str, Any],
                         merge_groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Retain source evidence for graph CONFLICT and MERGE edges."""
    conflicts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in sorted(net.getNodes(), key=lambda item: str(item.getID())):
        node_id, node_type = str(node.getID()), str(node.getType())
        indexed: dict[int, Any] = {}
        for connection in node.getConnections():
            request_index = int(connection.getJunctionIndex())
            if request_index >= 0:
                indexed[request_index] = connection
        if not indexed:
            continue
        if getattr(node, "_foes", None):
            indices = sorted(indexed)
            for pos, left_index in enumerate(indices):
                for right_index in indices[pos + 1:]:
                    foe_lr = bool(node.areFoes(left_index, right_index))
                    foe_rl = bool(node.areFoes(right_index, left_index))
                    if not (foe_lr or foe_rl):
                        continue
                    left_connection, right_connection = indexed[left_index], indexed[right_index]
                    left_lane = _connection_lane_id(left_connection, lane_by_id)
                    right_lane = _connection_lane_id(right_connection, lane_by_id)
                    if left_lane == right_lane:
                        continue
                    key = tuple(sorted((left_lane, right_lane)))
                    conflicts.setdefault(key, []).append({
                        "evidence_source": "sumolib_node_areFoes",
                        "node_id": node_id,
                        "node_type": node_type,
                        "request_indices": [left_index, right_index],
                        "are_foes_forward": foe_lr,
                        "are_foes_reverse": foe_rl,
                        "connections": [
                            _connection_record(left_connection, node_id=node_id, node_type=node_type),
                            _connection_record(right_connection, node_id=node_id, node_type=node_type),
                        ],
                    })
            continue

        # Match topology_graph._conflict_pairs' unregulated fallback. A strict
        # centerline crossing is retained as source evidence, not a vehicle-
        # footprint collision claim.
        driving = [index for index in sorted(indexed)
                   if not str(indexed[index].getFromLane().getID()).startswith(":")]
        for position, left_index in enumerate(driving):
            left_connection = indexed[left_index]
            left_shape = _connection_movement_shape(left_connection, lane_by_id)
            if not left_shape:
                continue
            for right_index in driving[position + 1:]:
                right_connection = indexed[right_index]
                if str(left_connection.getFromLane().getID()) == str(right_connection.getFromLane().getID()):
                    continue
                right_shape = _connection_movement_shape(right_connection, lane_by_id)
                if not right_shape or not _polylines_cross(left_shape, right_shape):
                    continue
                left_lane = _connection_lane_id(left_connection, lane_by_id)
                right_lane = _connection_lane_id(right_connection, lane_by_id)
                if left_lane == right_lane:
                    continue
                key = tuple(sorted((left_lane, right_lane)))
                conflicts.setdefault(key, []).append({
                    "evidence_source": "unregulated_via_polyline_cross",
                    "node_id": node_id,
                    "node_type": node_type,
                    "request_indices": [left_index, right_index],
                    "are_foes_forward": None,
                    "are_foes_reverse": None,
                    "connections": [
                        _connection_record(left_connection, node_id=node_id, node_type=node_type),
                        _connection_record(right_connection, node_id=node_id, node_type=node_type),
                    ],
                })

    merges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for group_id, source_lanes in sorted(merge_groups.items()):
        if len(source_lanes) < 2:
            continue
        left_lane, right_lane = str(source_lanes[0]), str(source_lanes[1])
        key = tuple(sorted((left_lane, right_lane)))
        merges.setdefault(key, []).append({
            "evidence_source": "topology_graph_v2_merge_group",
            "merge_group_id": str(group_id),
            "source_lanes": [left_lane, right_lane],
        })
    return {"conflict": conflicts, "merge": merges}


def _compile_zones(lane_lines: tuple[Any, ...], lane_by_id: dict[str, Any],
                   edge_index: np.ndarray, edge_type: np.ndarray,
                   lane_ids: tuple[str, ...], *,
                   ego_route_lane_mask: np.ndarray | None = None,
                   relation_provenance: dict[str, Any] | None = None,
                   ) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    pairs: dict[tuple[int, int], set[int]] = {}
    relation_provenance = relation_provenance or {}
    for (source, target), relation in zip(edge_index.T, edge_type):
        if int(relation) not in (CONFLICT, MERGE) or int(source) == int(target):
            continue
        key = tuple(sorted((int(source), int(target))))
        pairs.setdefault(key, set()).add(int(relation))
    successors: list[set[int]] = [set() for _ in lane_ids]
    for (source, target), relation in zip(edge_index.T, edge_type):
        if int(relation) == SUCCESSOR:
            successors[int(source)].add(int(target))
    successors_t = tuple(tuple(sorted(items)) for items in successors)
    lane_widths = tuple(max(0.5, float(lane_by_id[lane_id].getWidth())) for lane_id in lane_ids)
    path_cache: dict[int, tuple[list[tuple[int, ...]], bool]] = {}
    candidates: list[tuple[Any, int]] = []
    pair_relation_audit: list[dict[str, Any]] = []
    unmapped_pairs = 0
    same_edge_pairs = 0
    expansion_truncated_pairs = 0
    unmapped_by_relation: Counter[str] = Counter()
    mapped_by_relation: Counter[str] = Counter()

    def relation_labels(relations: set[int]) -> tuple[str, ...]:
        labels = []
        if CONFLICT in relations:
            labels.append("conflict")
        if MERGE in relations:
            labels.append("merge")
        return tuple(labels)

    def paths_from(lane_ptr: int) -> tuple[list[tuple[int, ...]], bool]:
        if lane_ptr not in path_cache:
            path_cache[lane_ptr] = _movement_paths(lane_ptr, lane_ids, successors_t)
        return path_cache[lane_ptr]

    def lane_corridor(lane_ptrs: tuple[int, ...]) -> Any | None:
        parts = [lane_lines[index].buffer(lane_widths[index] / 2.0, cap_style="flat")
                 for index in lane_ptrs]
        return unary_union(parts) if parts else None

    def internal_lanes(path: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index for index in path if lane_ids[index].startswith(":"))

    def exit_lane(path: tuple[int, ...]) -> int | None:
        internals = [position for position, index in enumerate(path) if lane_ids[index].startswith(":")]
        if not internals:
            return None
        for index in path[max(internals) + 1:]:
            if not lane_ids[index].startswith(":"):
                return int(index)
        return None

    def merge_corridor(path: tuple[int, ...], target_exit: int) -> Any | None:
        internals = internal_lanes(path)
        # Shared successor tails are omitted. The exit contribution stops after
        # one local lane-width from its start; body sweep supplies the actor's
        # own length/width around this convergence resource.
        exit_window = lane_widths[target_exit]
        exit_piece = substring(lane_lines[target_exit], 0.0,
                               min(exit_window, float(lane_lines[target_exit].length)))
        corridor_parts = [lane_lines[index].buffer(lane_widths[index] / 2.0, cap_style="flat")
                          for index in internals]
        if not exit_piece.is_empty:
            corridor_parts.append(exit_piece.buffer(lane_widths[target_exit] / 2.0, cap_style="flat"))
        return unary_union(corridor_parts) if corridor_parts else None

    def merge_centerline(path: tuple[int, ...], target_exit: int) -> Any | None:
        internals = internal_lanes(path)
        exit_piece = substring(lane_lines[target_exit], 0.0,
                               min(lane_widths[target_exit], float(lane_lines[target_exit].length)))
        parts = [lane_lines[index] for index in internals]
        if not exit_piece.is_empty:
            parts.append(exit_piece)
        return unary_union(parts) if parts else None

    for (left_index, right_index), relations in sorted(pairs.items()):
        labels = relation_labels(relations)
        left_id, right_id = lane_ids[left_index], lane_ids[right_index]
        left_lane, right_lane = lane_by_id[left_id], lane_by_id[right_id]
        if str(left_lane.getEdge().getID()) == str(right_lane.getEdge().getID()):
            same_edge_pairs += 1
            continue
        left_paths, left_truncated = paths_from(left_index)
        right_paths, right_truncated = paths_from(right_index)
        if left_truncated or right_truncated:
            expansion_truncated_pairs += 1
        left_path_names = [[lane_ids[index] for index in path] for path in left_paths]
        right_path_names = [[lane_ids[index] for index in path] for path in right_paths]
        ego_reachable = False
        if ego_route_lane_mask is not None:
            ego_reachable = bool(ego_route_lane_mask[left_index] or ego_route_lane_mask[right_index])
        candidate_relevant = bool(
            left_paths and right_paths and len(left_paths) <= 4 and len(right_paths) <= 4
            and not left_truncated and not right_truncated
        )
        pair_key = tuple(sorted((left_id, right_id)))

        # Audit each typed relation independently. A topology CONFLICT edge is
        # not itself proof of a vehicle-body collision, and a MERGE edge can
        # encode entry/exit convergence without positive-area path overlap.
        for relation_code, label in ((CONFLICT, "conflict"), (MERGE, "merge")):
            if relation_code not in relations:
                continue
            kind = 0 if relation_code == CONFLICT else 1
            path_evidence: list[dict[str, Any]] = []
            branch_pair_attempt_count = 0
            compatible_branch_pair_count = 0
            minimum_corridor_distance = float("inf")
            maximum_intersection_area = 0.0
            left_relevant_paths = 0
            right_relevant_paths = 0
            branch_pair_diagnostics: list[dict[str, Any]] = []

            if relation_code == CONFLICT:
                # Conflict proxies use branch-specific internal movement lanes.
                left_internal_paths = [(path, lane_corridor(internal_lanes(path)))
                                       for path in left_paths]
                right_internal_paths = [(path, lane_corridor(internal_lanes(path)))
                                        for path in right_paths]
                left_internal_paths = [(path, geom) for path, geom in left_internal_paths
                                       if geom is not None]
                right_internal_paths = [(path, geom) for path, geom in right_internal_paths
                                        if geom is not None]
                left_relevant_paths = len(left_internal_paths)
                right_relevant_paths = len(right_internal_paths)
                for left_path, left_corridor in left_internal_paths:
                    for right_path, right_corridor in right_internal_paths:
                        branch_pair_attempt_count += 1
                        compatible_branch_pair_count += 1
                        distance = float(left_corridor.distance(right_corridor))
                        minimum_corridor_distance = min(minimum_corridor_distance, distance)
                        overlap = left_corridor.intersection(right_corridor)
                        parts = [part for part in _polygon_parts(overlap) if float(part.area) > 1e-3]
                        pair_area = sum(float(part.area) for part in parts)
                        left_centerline = unary_union([lane_lines[index] for index in internal_lanes(left_path)])
                        right_centerline = unary_union([lane_lines[index] for index in internal_lanes(right_path)])
                        branch_pair_diagnostics.append({
                            "left_lane_path": [lane_ids[index] for index in left_path],
                            "right_lane_path": [lane_ids[index] for index in right_path],
                            "centerline_distance_m": float(left_centerline.distance(right_centerline)),
                            "centerline_intersects": bool(left_centerline.intersects(right_centerline)),
                            "corridor_distance_m": distance,
                            "positive_area_intersection_m2": pair_area,
                        })
                        for part in parts:
                            area = float(part.area)
                            maximum_intersection_area = max(maximum_intersection_area, area)
                            candidate_index = len(candidates)
                            candidates.append((part, kind))
                            path_evidence.append({
                                "candidate_index": candidate_index,
                                "left_lane_path": [lane_ids[index] for index in left_path],
                                "right_lane_path": [lane_ids[index] for index in right_path],
                                "left_lane_widths_m": [lane_widths[index] for index in left_path],
                                "right_lane_widths_m": [lane_widths[index] for index in right_path],
                                "intersection_area_m2": area,
                            })
            else:
                # Merge resources are local to branch paths with the same or
                # adjacent exit lane. Shared downstream tails are excluded.
                left_exit_paths = [(path, exit_lane(path)) for path in left_paths]
                right_exit_paths = [(path, exit_lane(path)) for path in right_paths]
                left_exit_paths = [(path, exit_ptr) for path, exit_ptr in left_exit_paths
                                   if exit_ptr is not None]
                right_exit_paths = [(path, exit_ptr) for path, exit_ptr in right_exit_paths
                                    if exit_ptr is not None]
                left_relevant_paths = len(left_exit_paths)
                right_relevant_paths = len(right_exit_paths)
                for left_path, left_exit in left_exit_paths:
                    for right_path, right_exit in right_exit_paths:
                        branch_pair_attempt_count += 1
                        left_edge = str(lane_by_id[lane_ids[left_exit]].getEdge().getID())
                        right_edge = str(lane_by_id[lane_ids[right_exit]].getEdge().getID())
                        left_lane_number = int(lane_by_id[lane_ids[left_exit]].getIndex())
                        right_lane_number = int(lane_by_id[lane_ids[right_exit]].getIndex())
                        compatible = left_exit == right_exit or (
                            left_edge == right_edge and abs(left_lane_number - right_lane_number) == 1
                        )
                        if not compatible:
                            branch_pair_diagnostics.append({
                                "left_lane_path": [lane_ids[index] for index in left_path],
                                "right_lane_path": [lane_ids[index] for index in right_path],
                                "left_exit_lane": lane_ids[left_exit],
                                "right_exit_lane": lane_ids[right_exit],
                                "same_exit_lane": left_exit == right_exit,
                                "same_exit_edge": left_edge == right_edge,
                                "adjacent_exit_lane_indices": (
                                    left_edge == right_edge and abs(left_lane_number - right_lane_number) == 1
                                ),
                                "compatible_exit_pair": False,
                            })
                            continue
                        compatible_branch_pair_count += 1
                        left_corridor = merge_corridor(left_path, left_exit)
                        right_corridor = merge_corridor(right_path, right_exit)
                        if left_corridor is None or right_corridor is None:
                            continue
                        distance = float(left_corridor.distance(right_corridor))
                        minimum_corridor_distance = min(minimum_corridor_distance, distance)
                        overlap = left_corridor.intersection(right_corridor)
                        parts = [part for part in _polygon_parts(overlap) if float(part.area) > 1e-3]
                        pair_area = sum(float(part.area) for part in parts)
                        left_centerline = merge_centerline(left_path, left_exit)
                        right_centerline = merge_centerline(right_path, right_exit)
                        branch_pair_diagnostics.append({
                            "left_lane_path": [lane_ids[index] for index in left_path],
                            "right_lane_path": [lane_ids[index] for index in right_path],
                            "left_exit_lane": lane_ids[left_exit],
                            "right_exit_lane": lane_ids[right_exit],
                            "same_exit_lane": left_exit == right_exit,
                            "same_exit_edge": left_edge == right_edge,
                            "adjacent_exit_lane_indices": (
                                left_edge == right_edge and abs(left_lane_number - right_lane_number) == 1
                            ),
                            "compatible_exit_pair": True,
                            "left_exit_lane_indices": left_lane_number,
                            "right_exit_lane_indices": right_lane_number,
                            "centerline_distance_m": float(left_centerline.distance(right_centerline)),
                            "centerline_intersects": bool(left_centerline.intersects(right_centerline)),
                            "corridor_distance_m": distance,
                            "positive_area_intersection_m2": pair_area,
                        })
                        for part in parts:
                            area = float(part.area)
                            maximum_intersection_area = max(maximum_intersection_area, area)
                            candidate_index = len(candidates)
                            candidates.append((part, kind))
                            path_evidence.append({
                                "candidate_index": candidate_index,
                                "left_lane_path": [lane_ids[index] for index in left_path],
                                "right_lane_path": [lane_ids[index] for index in right_path],
                                "left_lane_widths_m": [lane_widths[index] for index in left_path],
                                "right_lane_widths_m": [lane_widths[index] for index in right_path],
                                "left_exit_lane": lane_ids[left_exit],
                                "right_exit_lane": lane_ids[right_exit],
                                "intersection_area_m2": area,
                            })

            if path_evidence:
                mapped_by_relation[label] += 1
                classification = "mapped_public_corridor_overlap_proxy"
                unsupported_reason = None
            else:
                unmapped_by_relation[label] += 1
                if relation_code == MERGE and compatible_branch_pair_count == 0:
                    unsupported_reason = "no_same_or_adjacent_exit_lane_path_pair"
                elif left_relevant_paths == 0 or right_relevant_paths == 0:
                    unsupported_reason = "no_relevant_internal_or_exit_path"
                else:
                    unsupported_reason = "no_positive_area_corridor_intersection"
                source_evidence = relation_provenance.get(label, {}).get(pair_key, [])
                if relation_code == CONFLICT and any(
                    item.get("evidence_source") == "sumolib_node_areFoes"
                    and (item.get("are_foes_forward") or item.get("are_foes_reverse"))
                    for item in source_evidence
                ):
                    classification = "evidence_backed_foe_relation_without_corridor_overlap"
                    unsupported_reason = "foe_request_relation_is_preserved_but_no_public_corridor_event_zone_is_created"
                elif relation_code == MERGE:
                    compatible_pairs = [item for item in branch_pair_diagnostics
                                        if item.get("compatible_exit_pair")]
                    if compatible_pairs and all(
                        not item["same_exit_lane"]
                        and item["same_exit_edge"]
                        and item["adjacent_exit_lane_indices"]
                        and item["positive_area_intersection_m2"] <= 1e-3
                        for item in compatible_pairs
                    ):
                        classification = "evidence_backed_adjacent_exit_topology_only"
                        unsupported_reason = "adjacent_exit_topology_relation_has_no_positive_area_corridor_overlap"
                    else:
                        classification = "unsupported_public_corridor_geometry"
                else:
                    classification = "unsupported_public_corridor_geometry"
            pair_relation_audit.append({
                "pair_id": f"{left_id}::{right_id}::{label}",
                "lane_pair": [left_id, right_id],
                "topology_relation": label,
                "classification": classification,
                "unsupported_reason": unsupported_reason,
                "event_zone_created": bool(path_evidence),
                "corridor_relation_status": (
                    "positive_area_overlap_proxy" if path_evidence else "no_positive_area_overlap"
                ),
                "vehicle_footprint_clearance_status": (
                    "not_a_pairwise_collision_clearance_metric" if path_evidence
                    else "unknown_static_templates_omit_length_width"
                ),
                "vehicle_dimensions_source": "actor-specific length/width are read from TraCI vehicle.getLength/getWidth at runtime; route template declarations are not assumed to supply them",
                "actor_dimensions_source": "runtime TraCI vehicle.getLength/getWidth",
                "source_topology_relation_only": True,
                "candidate_relevant": candidate_relevant,
                "candidate_relevance_basis": "both endpoint lanes have <=4 fully enumerated public successor paths; traffic-template participation was not used as a filter",
                "ego_route_reachable_endpoint": ego_reachable,
                "path_enumeration": {
                    "left_path_count": len(left_paths),
                    "right_path_count": len(right_paths),
                    "left_paths": left_path_names,
                    "right_paths": right_path_names,
                    "left_movement_relevant_path_count": left_relevant_paths,
                    "right_movement_relevant_path_count": right_relevant_paths,
                    "expansion_truncated": bool(left_truncated or right_truncated),
                },
                "branch_pair_attempt_count": branch_pair_attempt_count,
                "compatible_branch_pair_count": compatible_branch_pair_count,
                "minimum_corridor_distance_m": (
                    minimum_corridor_distance if np.isfinite(minimum_corridor_distance) else None
                ),
                "maximum_intersection_area_m2": maximum_intersection_area,
                                "path_zone_evidence": path_evidence,
                "branch_pair_diagnostics": branch_pair_diagnostics,
                "topology_source_evidence": relation_provenance.get(label, {}).get(pair_key, []),
            })
        if not labels:
            unmapped_pairs += 1
    unmapped_pairs = sum(not item["event_zone_created"] for item in pair_relation_audit)
    if not candidates:
        return [], np.zeros((0,), dtype=np.int64), {
            "candidate_relation_pairs": len(pairs), "zone_count": 0,
            "same_edge_pairs_excluded": same_edge_pairs,
            "unmapped_relation_pairs": unmapped_pairs,
            "unmapped_pairs_by_relation": dict(unmapped_by_relation),
            "mapped_pairs_by_relation": dict(mapped_by_relation),
            "movement_path_expansion_truncated_pairs": expansion_truncated_pairs,
            "pair_relation_audit_schema": "corridor_v2_pair_audit_v2",
            "sumo_foe_or_right_of_way_semantics_audited": False,
            "pair_relation_audit": pair_relation_audit,
            "pair_relation_audit_counts": {
                "mapped_public_corridor_overlap_proxy": 0,
                "evidence_backed_foe_relation_without_corridor_overlap": sum(
                    item["classification"] == "evidence_backed_foe_relation_without_corridor_overlap"
                    for item in pair_relation_audit
                ),
                "evidence_backed_adjacent_exit_topology_only": sum(
                    item["classification"] == "evidence_backed_adjacent_exit_topology_only"
                    for item in pair_relation_audit
                ),
                "unsupported_public_corridor_geometry": sum(
                    item["classification"] == "unsupported_public_corridor_geometry"
                    for item in pair_relation_audit
                ),
            },
        }
    # Keep distinct local interaction resources even when they overlap
    # spatially. A global union would turn the whole connected junction into
    # one giant event zone; only topologically equal polygons are deduplicated.
    zones: list[Any] = []
    zone_kinds: list[set[int]] = []
    candidate_to_unsorted_zone: list[int] = []
    for geometry, kind in candidates:
        match = next((index for index, zone in enumerate(zones) if zone.equals(geometry)), None)
        if match is None:
            zones.append(geometry)
            zone_kinds.append({kind})
            match = len(zones) - 1
        else:
            zone_kinds[match].add(kind)
        candidate_to_unsorted_zone.append(match)
    order = sorted(range(len(zones)), key=lambda index: (
        round(float(zones[index].centroid.x), 6),
        round(float(zones[index].centroid.y), 6),
        round(float(zones[index].area), 6),
    ))
    old_to_new = {old: new for new, old in enumerate(order)}
    zones = [zones[index] for index in order]
    kinds = np.asarray([1 if 1 in zone_kinds[index] else 0 for index in order], dtype=np.int64)
    candidate_to_zone = [old_to_new[index] for index in candidate_to_unsorted_zone]
    for audit in pair_relation_audit:
        for path_evidence in audit["path_zone_evidence"]:
            candidate_index = path_evidence.pop("candidate_index")
            path_evidence["zone_ids"] = [candidate_to_zone[candidate_index]]
    metadata = {
        "candidate_relation_pairs": len(pairs), "zone_count": len(zones),
        "same_edge_pairs_excluded": same_edge_pairs,
        "unmapped_relation_pairs": unmapped_pairs,
        "unmapped_pairs_by_relation": dict(unmapped_by_relation),
        "mapped_pairs_by_relation": dict(mapped_by_relation),
        "movement_path_expansion_truncated_pairs": expansion_truncated_pairs,
        "pair_corridor_intersection_polygons": len(candidates),
        "polygons_with_holes": sum(1 for zone in zones if len(zone.interiors)),
        "numeric_polygon_hole_policy": "public numeric arrays store exterior vertices only; holes are filled in the proxy polygon and must not be interpreted as exact collision regions",
        "zone_geometry_semantics": "branch-specific public movement corridor intersections; lane centerlines are buffered by half declared lane width with flat ends; merge corridors include only one exit-lane-width after convergence; only exactly equal polygons are deduplicated",
        "entry_clearance_semantics": "computed from actor-oriented rectangular footprint sweep along each public lane polyline; static zone remains a lane-corridor proxy, not an exact collision label",
        "shared_exit_follow_tail_in_zone": False,
        "overlapping_resource_polygons_retained": True,
        "minimum_polygon_area_m2": 1e-3,
        "pair_relation_audit_schema": "corridor_v2_pair_audit_v2",
        "sumo_foe_or_right_of_way_semantics_audited": False,
        "pair_relation_audit": pair_relation_audit,
        "pair_relation_audit_counts": {
            "mapped_public_corridor_overlap_proxy": sum(
                item["classification"] == "mapped_public_corridor_overlap_proxy"
                for item in pair_relation_audit
            ),
            "evidence_backed_foe_relation_without_corridor_overlap": sum(
                item["classification"] == "evidence_backed_foe_relation_without_corridor_overlap"
                for item in pair_relation_audit
            ),
            "evidence_backed_adjacent_exit_topology_only": sum(
                item["classification"] == "evidence_backed_adjacent_exit_topology_only"
                for item in pair_relation_audit
            ),
            "unsupported_public_corridor_geometry": sum(
                item["classification"] == "unsupported_public_corridor_geometry"
                for item in pair_relation_audit
            ),
            "candidate_relevant_pair_count": sum(item["candidate_relevant"] for item in pair_relation_audit),
            "no_event_zone_pair_count": sum(not item["event_zone_created"] for item in pair_relation_audit),
        },
    }
    return zones, np.asarray(kinds, dtype=np.int64), metadata


def _movement_paths(start: int, lane_ids: tuple[str, ...],
                    successors: tuple[tuple[int, ...], ...], *,
                    max_lanes: int = 16, max_paths: int = 128) -> tuple[list[tuple[int, ...]], bool]:
    """Enumerate public lane paths through a junction to the first exit lane."""
    stack: list[tuple[tuple[int, ...], bool]] = [((int(start),), lane_ids[start].startswith(":"))]
    complete: list[tuple[int, ...]] = []
    truncated = False
    while stack:
        path, has_internal = stack.pop()
        next_lanes = [n for n in successors[path[-1]] if n not in path]
        if has_internal:
            exits = [n for n in next_lanes if not lane_ids[n].startswith(":")]
            if exits:
                complete.extend((*path, int(n)) for n in exits)
                next_lanes = [n for n in next_lanes if lane_ids[n].startswith(":")]
                if not next_lanes:
                    continue
        if not next_lanes or len(path) >= max_lanes:
            if len(path) >= max_lanes and next_lanes:
                truncated = True
            else:
                complete.append(path)
            continue
        for nxt in reversed(sorted(next_lanes, key=lambda index: lane_ids[index])):
            stack.append(((*path, int(nxt)), has_internal or lane_ids[nxt].startswith(":")))
        if len(stack) + len(complete) > max_paths:
            truncated = True
            stack = stack[:max(0, max_paths - len(complete))]
    complete = sorted(set(complete), key=lambda path: tuple(lane_ids[index] for index in path))
    return complete, truncated


def _polygon_parts(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon" or hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _polygon_parts(child)]
    return []


def _geometry_points(geometry: Any) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type in {"LineString", "LinearRing"}:
        if geometry.length <= 1e-6:
            return []
        midpoint = geometry.interpolate(0.5, normalized=True)
        return [midpoint]
    if hasattr(geometry, "geoms"):
        return [point for item in geometry.geoms for point in _geometry_points(item)]
    return []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
