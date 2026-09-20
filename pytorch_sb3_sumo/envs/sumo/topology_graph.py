"""Deterministic padded lane-topology graphs for the SUMO scene encoder.

The graph is static for one training scenario.  It is therefore parsed once
when the SB3 model is constructed and registered as buffers by the feature
extractor, instead of being duplicated in every replay transition.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sumolib


SUCCESSOR = 0
PREDECESSOR = 1
LEFT = 2
RIGHT = 3
CONFLICT = 4

RELATION_NAMES = ("successor", "predecessor", "left", "right", "conflict")
NUM_RELATIONS = len(RELATION_NAMES)
MAX_TOPO_NODES = 64
MAX_TOPO_EDGES = 256
LANE_SAMPLE_POINTS = 10
TOPOLOGY_ATTRIBUTE_DIM = 8


class TopologyCapacityError(ValueError):
    """Raised when a network cannot be represented without truncation."""


@dataclass(frozen=True)
class PaddedTopologyGraph:
    """Fixed-capacity lane graph consumed by :class:`TopoTemporalGraphExtractor`.

    Only the six arrays below are placed in the model.  Lane identifiers and
    parser diagnostics are deliberately kept out of replay/model inputs.
    """

    lane_points: np.ndarray
    lane_attrs: np.ndarray
    node_mask: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_mask: np.ndarray

    def __post_init__(self) -> None:
        node_count = int(self.lane_points.shape[0])
        if self.lane_points.ndim != 3 or self.lane_points.shape[-1] != 2:
            raise ValueError(
                "lane_points must have shape [nodes, samples, 2], got "
                f"{self.lane_points.shape}"
            )
        if self.lane_attrs.shape != (node_count, TOPOLOGY_ATTRIBUTE_DIM):
            raise ValueError(
                "lane_attrs must have shape "
                f"[{node_count}, {TOPOLOGY_ATTRIBUTE_DIM}], got {self.lane_attrs.shape}"
            )
        if self.node_mask.shape != (node_count,):
            raise ValueError(f"node_mask must have shape [{node_count}]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, edges], got {self.edge_index.shape}"
            )
        edge_count = int(self.edge_index.shape[1])
        if self.edge_type.shape != (edge_count,):
            raise ValueError(f"edge_type must have shape [{edge_count}]")
        if self.edge_mask.shape != (edge_count,):
            raise ValueError(f"edge_mask must have shape [{edge_count}]")
        valid_nodes = int(np.asarray(self.node_mask, dtype=bool).sum())
        valid_edges = np.asarray(self.edge_mask, dtype=bool)
        if valid_edges.any():
            indices = np.asarray(self.edge_index)[:, valid_edges]
            if int(indices.min()) < 0 or int(indices.max()) >= valid_nodes:
                raise ValueError(
                    "Every valid edge endpoint must refer to a valid (unpadded) node"
                )
            relations = np.asarray(self.edge_type)[valid_edges]
            if int(relations.min()) < 0 or int(relations.max()) >= NUM_RELATIONS:
                raise ValueError(
                    f"edge_type values must lie in [0, {NUM_RELATIONS - 1}]"
                )

    @property
    def valid_node_count(self) -> int:
        return int(np.asarray(self.node_mask, dtype=bool).sum())

    @property
    def valid_edge_count(self) -> int:
        return int(np.asarray(self.edge_mask, dtype=bool).sum())


@dataclass(frozen=True)
class TopologyGraphInfo:
    """Human/audit metadata produced alongside a padded graph."""

    network_path: str
    lane_ids: tuple[str, ...]
    valid_node_count: int
    valid_edge_count: int
    relation_counts: dict[str, int]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_path": self.network_path,
            "lane_ids": list(self.lane_ids),
            "valid_node_count": self.valid_node_count,
            "valid_edge_count": self.valid_edge_count,
            "relation_counts": dict(self.relation_counts),
            "fingerprint": self.fingerprint,
        }


def _all_lanes(net: Any) -> list[Any]:
    lanes = [
        lane
        for edge in net.getEdges(withInternal=True)
        for lane in edge.getLanes()
    ]
    lanes.sort(key=lambda lane: str(lane.getID()))
    lane_ids = [str(lane.getID()) for lane in lanes]
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("SUMO network contains duplicate lane IDs")
    return lanes


def _resample_polyline(shape: Iterable[Iterable[float]], samples: int) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive")
    points = np.asarray([tuple(point)[:2] for point in shape], dtype=np.float64)
    if points.size == 0:
        raise ValueError("SUMO lane has an empty centerline")
    points = points.reshape(-1, 2)
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), samples, axis=0)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-9))
    points = points[keep]
    cumulative = cumulative[keep]
    if len(points) == 1 or cumulative[-1] <= 1e-9:
        return np.repeat(points[:1].astype(np.float32), samples, axis=0)
    targets = np.linspace(0.0, float(cumulative[-1]), samples)
    output = np.stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)],
        axis=-1,
    )
    return output.astype(np.float32)


def _polyline_curvature(points: np.ndarray) -> float:
    differences = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    valid = lengths > 1e-9
    if int(valid.sum()) < 2:
        return 0.0
    headings = np.unwrap(np.arctan2(differences[valid, 1], differences[valid, 0]))
    total_length = float(lengths[valid].sum())
    return float(np.abs(np.diff(headings)).sum() / max(total_length, 1e-9))


def _successor_pairs(lanes: list[Any], lane_by_id: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for lane in lanes:
        for connection in lane.getOutgoing():
            source = str(connection.getFromLane().getID())
            target = str(connection.getToLane().getID())
            if source not in lane_by_id or target not in lane_by_id:
                raise ValueError(
                    f"Connection {source!r} -> {target!r} references an unknown lane"
                )
            via = str(connection.getViaLaneID() or "")
            if via:
                if via not in lane_by_id:
                    raise ValueError(
                        f"Connection {source!r} -> {target!r} references missing via lane {via!r}"
                    )
                pairs.add((source, via))
                pairs.add((via, target))
            else:
                pairs.add((source, target))
    return pairs


def _connection_lane_id(connection: Any, lane_by_id: dict[str, Any]) -> str:
    via = str(connection.getViaLaneID() or "")
    if via:
        if via not in lane_by_id:
            raise ValueError(f"Conflict connection references missing via lane {via!r}")
        return via
    source = str(connection.getFromLane().getID())
    if source not in lane_by_id:
        raise ValueError(f"Conflict connection references missing source lane {source!r}")
    return source


def _conflict_pairs(net: Any, lane_by_id: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for node in sorted(net.getNodes(), key=lambda item: str(item.getID())):
        indexed: dict[int, Any] = {}
        for connection in node.getConnections():
            junction_index = int(connection.getJunctionIndex())
            if junction_index < 0:
                continue
            previous = indexed.get(junction_index)
            if previous is not None:
                previous_key = (
                    str(previous.getFromLane().getID()),
                    str(previous.getToLane().getID()),
                    str(previous.getViaLaneID() or ""),
                )
                current_key = (
                    str(connection.getFromLane().getID()),
                    str(connection.getToLane().getID()),
                    str(connection.getViaLaneID() or ""),
                )
                if current_key != previous_key:
                    raise ValueError(
                        f"Junction {node.getID()!r} reuses foe index {junction_index}"
                    )
            indexed[junction_index] = connection

        indices = sorted(indexed)
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                try:
                    foes = bool(node.areFoes(left_index, right_index)) or bool(
                        node.areFoes(right_index, left_index)
                    )
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid foe indices ({left_index}, {right_index}) at junction "
                        f"{node.getID()!r}"
                    ) from exc
                if not foes:
                    continue
                left_lane = _connection_lane_id(indexed[left_index], lane_by_id)
                right_lane = _connection_lane_id(indexed[right_index], lane_by_id)
                if left_lane != right_lane:
                    pairs.add((left_lane, right_lane))
                    pairs.add((right_lane, left_lane))
    return pairs


def topology_graph_fingerprint(graph: PaddedTopologyGraph) -> str:
    digest = hashlib.sha256()
    for array in (
        graph.lane_points,
        graph.lane_attrs,
        graph.node_mask,
        graph.edge_index,
        graph.edge_type,
        graph.edge_mask,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_topology_graph(
    network_path: str | Path,
    *,
    max_nodes: int = MAX_TOPO_NODES,
    max_edges: int = MAX_TOPO_EDGES,
    lane_sample_points: int = LANE_SAMPLE_POINTS,
    coordinate_offset: tuple[float, float] = (0.0, 0.0),
    return_info: bool = False,
) -> PaddedTopologyGraph | tuple[PaddedTopologyGraph, TopologyGraphInfo]:
    """Parse one SUMO network into a deterministic fixed-capacity graph.

    No node or edge is silently truncated.  ``coordinate_offset`` aligns a
    reconstructed SUMO network (notably CARLA) with the coordinates emitted by
    the environment observation adapter.
    """

    path = Path(network_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if max_nodes <= 0 or max_edges <= 0:
        raise ValueError("max_nodes and max_edges must be positive")
    offset = np.asarray(coordinate_offset, dtype=np.float32)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("coordinate_offset must contain two finite values")

    net = sumolib.net.readNet(str(path), withInternal=True)
    lanes = _all_lanes(net)
    lane_by_id = {str(lane.getID()): lane for lane in lanes}
    lane_ids = tuple(lane_by_id)
    if len(lanes) > max_nodes:
        raise TopologyCapacityError(
            f"Network {path} has {len(lanes)} lanes, exceeding max_nodes={max_nodes}; "
            "increase the declared capacity instead of truncating"
        )

    successor_pairs = _successor_pairs(lanes, lane_by_id)
    incoming_degree = {lane_id: 0 for lane_id in lane_ids}
    outgoing_degree = {lane_id: 0 for lane_id in lane_ids}
    for source, target in successor_pairs:
        outgoing_degree[source] += 1
        incoming_degree[target] += 1

    lane_points = np.zeros((max_nodes, lane_sample_points, 2), dtype=np.float32)
    lane_attrs = np.zeros((max_nodes, TOPOLOGY_ATTRIBUTE_DIM), dtype=np.float32)
    node_mask = np.zeros(max_nodes, dtype=bool)
    for index, lane in enumerate(lanes):
        lane_id = str(lane.getID())
        sampled = _resample_polyline(lane.getShape(), lane_sample_points) - offset
        lane_points[index] = sampled
        lane_attrs[index] = np.asarray(
            [
                math.log1p(max(0.0, float(lane.getLength()))),
                math.log1p(max(0.0, float(lane.getSpeed()))),
                float(lane.getWidth()),
                _polyline_curvature(sampled),
                float(lane_id.startswith(":")),
                float(lane.getIndex()),
                float(incoming_degree[lane_id]),
                float(outgoing_degree[lane_id]),
            ],
            dtype=np.float32,
        )
        node_mask[index] = True

    typed_edges: set[tuple[str, str, int]] = set()
    for source, target in successor_pairs:
        typed_edges.add((source, target, SUCCESSOR))
        typed_edges.add((target, source, PREDECESSOR))

    for edge in net.getEdges(withInternal=True):
        edge_lanes = sorted(edge.getLanes(), key=lambda lane: int(lane.getIndex()))
        for right_lane, left_lane in zip(edge_lanes, edge_lanes[1:]):
            right_index = int(right_lane.getIndex())
            left_index = int(left_lane.getIndex())
            if left_index - right_index != 1:
                continue
            right_id = str(right_lane.getID())
            left_id = str(left_lane.getID())
            typed_edges.add((right_id, left_id, LEFT))
            typed_edges.add((left_id, right_id, RIGHT))

    for source, target in _conflict_pairs(net, lane_by_id):
        typed_edges.add((source, target, CONFLICT))

    lane_index = {lane_id: index for index, lane_id in enumerate(lane_ids)}
    sorted_edges = sorted(
        (
            (lane_index[source], lane_index[target], relation)
            for source, target, relation in typed_edges
        ),
        key=lambda item: (item[2], item[0], item[1]),
    )
    if len(sorted_edges) > max_edges:
        raise TopologyCapacityError(
            f"Network {path} has {len(sorted_edges)} directed typed edges, "
            f"exceeding max_edges={max_edges}; increase the declared capacity "
            "instead of truncating"
        )

    edge_index = np.zeros((2, max_edges), dtype=np.int64)
    edge_type = np.zeros(max_edges, dtype=np.int64)
    edge_mask = np.zeros(max_edges, dtype=bool)
    for index, (source, target, relation) in enumerate(sorted_edges):
        edge_index[:, index] = (source, target)
        edge_type[index] = relation
        edge_mask[index] = True

    graph = PaddedTopologyGraph(
        lane_points=lane_points,
        lane_attrs=lane_attrs,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_mask=edge_mask,
    )
    relation_counts = {
        name: int(
            np.logical_and(edge_mask, edge_type == relation).sum()
        )
        for relation, name in enumerate(RELATION_NAMES)
    }
    info = TopologyGraphInfo(
        network_path=str(path),
        lane_ids=lane_ids,
        valid_node_count=graph.valid_node_count,
        valid_edge_count=graph.valid_edge_count,
        relation_counts=relation_counts,
        fingerprint=topology_graph_fingerprint(graph),
    )
    return (graph, info) if return_info else graph


__all__ = [
    "CONFLICT",
    "LANE_SAMPLE_POINTS",
    "LEFT",
    "MAX_TOPO_EDGES",
    "MAX_TOPO_NODES",
    "NUM_RELATIONS",
    "PREDECESSOR",
    "PaddedTopologyGraph",
    "RELATION_NAMES",
    "RIGHT",
    "SUCCESSOR",
    "TOPOLOGY_ATTRIBUTE_DIM",
    "TopologyCapacityError",
    "TopologyGraphInfo",
    "build_topology_graph",
    "topology_graph_fingerprint",
]
