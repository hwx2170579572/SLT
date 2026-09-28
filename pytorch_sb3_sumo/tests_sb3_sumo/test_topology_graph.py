from __future__ import annotations

import numpy as np
import pytest
import torch

from algos.sb3_torch.topo_temporal_features import RelationalTopologyEncoder
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph import (
    PREDECESSOR,
    SUCCESSOR,
    TopologyCapacityError,
    build_topology_graph,
)


def _paper_graph(name: str):
    specification = PAPER_SCENARIOS[name]
    return build_topology_graph(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        max_nodes=specification.topology_max_nodes,
        max_edges=specification.topology_max_edges,
        return_info=True,
    )


@pytest.mark.parametrize("scenario", tuple(PAPER_SCENARIOS))
def test_all_released_maps_are_deterministic_and_within_capacity(scenario: str) -> None:
    specification = PAPER_SCENARIOS[scenario]
    first, first_info = _paper_graph(scenario)
    second, second_info = _paper_graph(scenario)
    assert first.valid_node_count <= specification.topology_max_nodes
    assert first.valid_edge_count <= specification.topology_max_edges
    assert first_info.fingerprint == second_info.fingerprint
    for field in (
        "lane_points",
        "lane_attrs",
        "node_mask",
        "edge_index",
        "edge_type",
        "edge_mask",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_released_capacity_scan_matches_declared_l1_bound() -> None:
    counts = [
        (graph.valid_node_count, graph.valid_edge_count)
        for graph, _ in (_paper_graph(name) for name in PAPER_SCENARIOS)
    ]
    assert max(nodes for nodes, _ in counts) == 56
    assert max(edges for _, edges in counts) == 336


def test_valid_edges_have_inverse_predecessors_and_never_touch_padding() -> None:
    graph, _ = _paper_graph("roundabout_medium")
    valid_edges = graph.edge_mask
    endpoints = graph.edge_index[:, valid_edges]
    assert int(endpoints.min()) >= 0
    assert int(endpoints.max()) < graph.valid_node_count
    typed = {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            endpoints[0], endpoints[1], graph.edge_type[valid_edges]
        )
    }
    for source, target, relation in typed:
        if relation == SUCCESSOR:
            assert (target, source, PREDECESSOR) in typed
    assert not graph.edge_mask[graph.valid_edge_count :].any()
    assert not graph.node_mask[graph.valid_node_count :].any()


def test_capacity_overflow_is_explicit_instead_of_truncated() -> None:
    graph, _ = _paper_graph("left_turn")
    specification = PAPER_SCENARIOS["left_turn"]
    with pytest.raises(TopologyCapacityError, match="instead of truncating"):
        build_topology_graph(
            specification.network_path,
            max_nodes=graph.valid_node_count - 1,
        )
    with pytest.raises(TopologyCapacityError, match="instead of truncating"):
        build_topology_graph(
            specification.network_path,
            max_edges=graph.valid_edge_count - 1,
        )


def test_relational_encoder_is_equivariant_to_node_permutation_and_edge_remap() -> None:
    graph, _ = _paper_graph("left_turn")
    torch.manual_seed(7)
    encoder = RelationalTopologyEncoder(feature_dim=32, layers=2).eval()
    points = torch.as_tensor(graph.lane_points).unsqueeze(0)
    attrs = torch.as_tensor(graph.lane_attrs)
    node_mask = torch.as_tensor(graph.node_mask)
    edge_index = torch.as_tensor(graph.edge_index)
    edge_type = torch.as_tensor(graph.edge_type)
    edge_mask = torch.as_tensor(graph.edge_mask)
    reference = encoder(points, attrs, node_mask, edge_index, edge_type, edge_mask)

    permutation = torch.randperm(graph.lane_points.shape[0])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    remapped_edges = inverse[edge_index]
    permuted = encoder(
        points[:, permutation],
        attrs[permutation],
        node_mask[permutation],
        remapped_edges,
        edge_type,
        edge_mask,
    )
    torch.testing.assert_close(reference, permuted[:, inverse], rtol=1e-5, atol=1e-6)
