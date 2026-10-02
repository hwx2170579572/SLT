"""Merge-aware deterministic lane graphs for the topology-temporal v2 study.

This module is deliberately separate from :mod:`topology_graph`.  The v1
parser and every checkpoint produced from it remain untouched while v2 adds a
sixth, symmetric ``MERGE`` relation.  All non-merge geometry, attributes, and
relations reuse the audited v1 parsing helpers so a v1/v2 difference can be
localized to the new relation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sumolib

from .topology_graph import (
    CONFLICT,
    LANE_SAMPLE_POINTS,
    LEFT,
    MAX_TOPO_EDGES,
    MAX_TOPO_NODES,
    PREDECESSOR,
    RELATION_NAMES as V1_RELATION_NAMES,
    RIGHT,
    SUCCESSOR,
    TOPOLOGY_ATTRIBUTE_DIM,
    TopologyCapacityError,
    _all_lanes,
    _conflict_pairs,
    _polyline_curvature,
    _resample_polyline,
    _successor_pairs,
)


MERGE = 5
RELATION_NAMES = (*V1_RELATION_NAMES, "merge")
NUM_RELATIONS = len(RELATION_NAMES)


@dataclass(frozen=True)
class PaddedTopologyGraphV2:
    """Fixed-capacity lane graph with the additional ``MERGE`` relation."""

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
                f"[{node_count}, {TOPOLOGY_ATTRIBUTE_DIM}], got "
                f"{self.lane_attrs.shape}"
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
            endpoints = np.asarray(self.edge_index)[:, valid_edges]
            if int(endpoints.min()) < 0 or int(endpoints.max()) >= valid_nodes:
                raise ValueError(
                    "Every valid edge endpoint must refer to a valid node"
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
class TopologyGraphInfoV2:
    """Auditable metadata for a v2 graph."""

    network_path: str
    lane_ids: tuple[str, ...]
    valid_node_count: int
    valid_edge_count: int
    relation_counts: dict[str, int]
    merge_groups: dict[str, tuple[str, ...]]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "topology-graph-info/v2",
            "network_path": self.network_path,
            "lane_ids": list(self.lane_ids),
            "valid_node_count": self.valid_node_count,
            "valid_edge_count": self.valid_edge_count,
            "relation_counts": dict(self.relation_counts),
            "merge_groups": {
                target: list(sources)
                for target, sources in sorted(self.merge_groups.items())
            },
            "fingerprint": self.fingerprint,
        }


def _edge_id(lane: Any) -> str:
    edge = lane.getEdge()
    return str(edge.getID())


def _first_non_internal_targets(
    source: str,
    *,
    successors: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return the first downstream non-internal lanes reachable from source.

    SUMO connections may contain one or several ``:junction`` lanes.  Search
    only through those internal lanes and stop at the first ordinary lane on
    each branch.  The traversal is deterministic and cycle-safe.
    """

    pending = list(reversed(successors.get(source, ())))
    visited_internal: set[str] = set()
    targets: set[str] = set()
    while pending:
        lane_id = pending.pop()
        if not lane_id.startswith(":"):
            targets.add(lane_id)
            continue
        if lane_id in visited_internal:
            continue
        visited_internal.add(lane_id)
        pending.extend(reversed(successors.get(lane_id, ())))
    return tuple(sorted(targets))


def _merge_groups(
    lanes: list[Any],
    successor_pairs: set[tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Group distinct entry lanes that first converge on one downstream lane.

    Ordinary adjacent lanes belonging to the same SUMO edge are excluded.  A
    merge group therefore requires at least two non-internal source lanes from
    different incoming edges and a shared first non-internal downstream lane.
    """

    lane_by_id = {str(lane.getID()): lane for lane in lanes}
    successors_mutable: dict[str, list[str]] = {lane_id: [] for lane_id in lane_by_id}
    for source, target in sorted(successor_pairs):
        successors_mutable[source].append(target)
    successors = {
        lane_id: tuple(sorted(set(targets)))
        for lane_id, targets in successors_mutable.items()
    }

    source_targets: dict[str, tuple[str, ...]] = {}
    for source in sorted(lane_by_id):
        if source.startswith(":"):
            continue
        targets = _first_non_internal_targets(
            source,
            successors=successors,
        )
        source_targets[source] = tuple(
            target for target in targets if target != source
        )

    groups: dict[str, tuple[str, ...]] = {}
    by_target: dict[str, set[str]] = {}
    for source, targets in source_targets.items():
        for target in targets:
            by_target.setdefault(target, set()).add(source)

    # Exact lane convergence is the primary rule from the v2 design.  Store
    # pair-level groups so three-way junctions cannot accidentally connect two
    # ordinary adjacent source lanes that belong to the same incoming edge.
    for target, candidates in sorted(by_target.items()):
        ordered = sorted(candidates)
        group_index = 0
        for index, left in enumerate(ordered):
            left_edge = _edge_id(lane_by_id[left])
            for right in ordered[index + 1 :]:
                right_edge = _edge_id(lane_by_id[right])
                if left_edge == right_edge:
                    continue
                groups[f"lane:{target}:{group_index}"] = (left, right)
                group_index += 1

    # Some released SUMO networks (notably the double-merge asset) route two
    # distinct entries into adjacent lanes of the same downstream edge rather
    # than literally the same lane.  They still create one merge zone and are
    # precisely the case missed by junction ``areFoes``.  Generalise only to
    # adjacent downstream lanes, while retaining the different-source-edge
    # exclusion above.  Exact-lane pairs are skipped because they were already
    # registered by the primary rule.
    by_target_edge: dict[str, dict[str, set[int]]] = {}
    for source, targets in source_targets.items():
        for target in targets:
            target_lane = lane_by_id[target]
            target_edge = _edge_id(target_lane)
            by_target_edge.setdefault(target_edge, {}).setdefault(source, set()).add(
                int(target_lane.getIndex())
            )
    for target_edge, source_lanes in sorted(by_target_edge.items()):
        ordered = sorted(source_lanes)
        group_index = 0
        for index, left in enumerate(ordered):
            left_source_edge = _edge_id(lane_by_id[left])
            for right in ordered[index + 1 :]:
                if left_source_edge == _edge_id(lane_by_id[right]):
                    continue
                if set(source_targets[left]) & set(source_targets[right]):
                    continue
                adjacent = any(
                    abs(left_lane - right_lane) == 1
                    for left_lane in source_lanes[left]
                    for right_lane in source_lanes[right]
                )
                if not adjacent:
                    continue
                groups[f"edge:{target_edge}:{group_index}"] = (left, right)
                group_index += 1
    return groups


def _merge_pairs(groups: dict[str, tuple[str, ...]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for sources in groups.values():
        for index, left in enumerate(sources):
            for right in sources[index + 1 :]:
                pairs.add((left, right))
                pairs.add((right, left))
    return pairs


def topology_graph_v2_fingerprint(graph: PaddedTopologyGraphV2) -> str:
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


def build_topology_graph_v2(
    network_path: str | Path,
    *,
    max_nodes: int = MAX_TOPO_NODES,
    max_edges: int = MAX_TOPO_EDGES,
    lane_sample_points: int = LANE_SAMPLE_POINTS,
    coordinate_offset: tuple[float, float] = (0.0, 0.0),
    include_merge: bool = True,
    return_info: bool = False,
) -> PaddedTopologyGraphV2 | tuple[PaddedTopologyGraphV2, TopologyGraphInfoV2]:
    """Build the v2 graph without silently truncating nodes or relations."""

    path = Path(network_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if max_nodes <= 0 or max_edges <= 0:
        raise ValueError("max_nodes and max_edges must be positive")
    if lane_sample_points <= 0:
        raise ValueError("lane_sample_points must be positive")
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

    merge_groups = _merge_groups(lanes, successor_pairs) if include_merge else {}
    for source, target in _merge_pairs(merge_groups):
        typed_edges.add((source, target, MERGE))

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

    graph = PaddedTopologyGraphV2(
        lane_points=lane_points,
        lane_attrs=lane_attrs,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_mask=edge_mask,
    )
    relation_counts = {
        name: int(np.logical_and(edge_mask, edge_type == relation).sum())
        for relation, name in enumerate(RELATION_NAMES)
    }
    info = TopologyGraphInfoV2(
        network_path=str(path),
        lane_ids=lane_ids,
        valid_node_count=graph.valid_node_count,
        valid_edge_count=graph.valid_edge_count,
        relation_counts=relation_counts,
        merge_groups=merge_groups,
        fingerprint=topology_graph_v2_fingerprint(graph),
    )
    return (graph, info) if return_info else graph


__all__ = [
    "CONFLICT",
    "LANE_SAMPLE_POINTS",
    "LEFT",
    "MAX_TOPO_EDGES",
    "MAX_TOPO_NODES",
    "MERGE",
    "NUM_RELATIONS",
    "PREDECESSOR",
    "PaddedTopologyGraphV2",
    "RELATION_NAMES",
    "RIGHT",
    "SUCCESSOR",
    "TOPOLOGY_ATTRIBUTE_DIM",
    "TopologyCapacityError",
    "TopologyGraphInfoV2",
    "build_topology_graph_v2",
    "topology_graph_v2_fingerprint",
]
