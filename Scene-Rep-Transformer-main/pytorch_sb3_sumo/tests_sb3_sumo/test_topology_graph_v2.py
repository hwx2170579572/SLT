from __future__ import annotations

import numpy as np
import pytest

from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph import TopologyCapacityError, build_topology_graph
from envs.sumo.topology_graph_v2 import MERGE, build_topology_graph_v2
from tools.audit_topology_graphs_v2 import audit


def _typed_edges(graph):
    valid = np.asarray(graph.edge_mask, dtype=bool)
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            graph.edge_index[0, valid],
            graph.edge_index[1, valid],
            graph.edge_type[valid],
        )
    }


@pytest.mark.parametrize("scenario", tuple(PAPER_SCENARIOS))
def test_v2_preserves_v1_graph_and_adds_only_symmetric_merge(scenario: str) -> None:
    specification = PAPER_SCENARIOS[scenario]
    v1 = build_topology_graph(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        max_nodes=specification.topology_max_nodes,
        max_edges=specification.topology_max_edges,
    )
    v2, info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        max_nodes=specification.topology_max_nodes,
        max_edges=specification.topology_max_edges,
        return_info=True,
    )
    np.testing.assert_array_equal(v1.lane_points, v2.lane_points)
    np.testing.assert_array_equal(v1.lane_attrs, v2.lane_attrs)
    np.testing.assert_array_equal(v1.node_mask, v2.node_mask)
    v1_edges = _typed_edges(v1)
    v2_edges = _typed_edges(v2)
    assert v1_edges.issubset(v2_edges)
    additions = v2_edges - v1_edges
    assert all(relation == MERGE for _, _, relation in additions)
    assert all((target, source, MERGE) in additions for source, target, _ in additions)
    assert info.valid_edge_count <= len(v2.edge_mask)


def test_cross_has_the_predeclared_merge_relation_without_dense_connectivity() -> None:
    specification = PAPER_SCENARIOS["cross"]
    graph, info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        return_info=True,
    )
    assert info.relation_counts["merge"] == 2
    assert info.merge_groups == {"edge:gneE5:0": ("gneE17_0", "gneE22_0")}
    assert graph.valid_node_count == 10
    assert graph.valid_edge_count == 20


def test_disabling_merge_is_byte_identical_to_v1_arrays() -> None:
    specification = PAPER_SCENARIOS["roundabout_medium"]
    v1 = build_topology_graph(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
    )
    v2 = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        include_merge=False,
    )
    for left, right in (
        (v1.lane_points, v2.lane_points),
        (v1.lane_attrs, v2.lane_attrs),
        (v1.node_mask, v2.node_mask),
        (v1.edge_index, v2.edge_index),
        (v1.edge_type, v2.edge_type),
        (v1.edge_mask, v2.edge_mask),
    ):
        np.testing.assert_array_equal(left, right)


def test_v2_capacity_overflow_is_explicit() -> None:
    specification = PAPER_SCENARIOS["cross"]
    with pytest.raises(TopologyCapacityError):
        build_topology_graph_v2(specification.network_path, max_nodes=9)
    with pytest.raises(TopologyCapacityError):
        build_topology_graph_v2(specification.network_path, max_edges=19)


def test_all_scenario_v2_audit_passes() -> None:
    report = audit()
    assert report["passed"] is True
    assert report["scenario_count"] == len(PAPER_SCENARIOS)
    assert report["failures"] == []
