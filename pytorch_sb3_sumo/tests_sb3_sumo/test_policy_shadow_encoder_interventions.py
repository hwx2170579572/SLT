from __future__ import annotations

import torch

from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
from algos.sb3_torch.topo_temporal_features import StructuredLatent
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph_v2 import build_topology_graph_v2

from tests_sb3_sumo.test_d1_goalonly_nonlinear_slots import _batch, _encoder, _space


def _make_goalonly() -> tuple[IncrementalTopoEncoder, dict[str, torch.Tensor]]:
    spec = PAPER_SCENARIOS["left_turn"]
    graph, info = build_topology_graph_v2(
        spec.network_path,
        coordinate_offset=spec.coordinate_offset,
        return_info=True,
    )
    encoder = IncrementalTopoEncoder(
        _space(route_reachability=True, route_capacity=int(graph.node_mask.shape[0])),
        topology_graph=graph,
        topology_lane_ids=tuple(info.lane_ids),
        use_route=True,
        use_topology=True,
        use_slots=False,
        use_route_reachability=True,
        use_topology_actor_intent=False,
        use_topology_relations=False,
        use_topology_goal=True,
        random_augmentation=False,
    ).eval()
    observations = _batch()
    valid_count = int(encoder.topology_node_mask.sum())
    labels = torch.full((observations["trajectory"].shape[0], int(graph.node_mask.size)), -1.0)
    labels[:, :valid_count] = 1.0
    observations["route_reachability"] = labels
    return encoder, observations


def _make_3slot() -> tuple[IncrementalTopoEncoder, dict[str, torch.Tensor]]:
    spec = PAPER_SCENARIOS["left_turn"]
    graph, info = build_topology_graph_v2(
        spec.network_path,
        coordinate_offset=spec.coordinate_offset,
        return_info=True,
    )
    encoder = IncrementalTopoEncoder(
        _space(),
        topology_graph=graph,
        topology_lane_ids=tuple(info.lane_ids),
        use_route=True,
        use_topology=False,
        use_slots=True,
        use_incremental_slots=True,
        use_parameter_matched_nonlinear_slots=True,
        random_augmentation=False,
    ).eval()
    return encoder, _batch()


def _diag_snapshot(encoder: IncrementalTopoEncoder):
    values = encoder.diagnostic_values()
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def test_noop_shadow_matches_real_forward_and_does_not_publish_diagnostics():
    encoder, observations = _make_goalonly()
    encoder.configure_diagnostics(True, sample_every=1, source="eval_policy")
    with torch.no_grad():
        ordinary = encoder(observations)
    before_index = encoder.diagnostic_sample_index
    before_forward_index = encoder._diagnostic_forward_index
    before_site_counts = dict(encoder._diagnostic_site_forward_counts)
    before_values = _diag_snapshot(encoder)

    with encoder.policy_shadow_probe("__noop__"):
        with torch.no_grad():
            shadow = encoder(observations)

    torch.testing.assert_close(shadow, ordinary, rtol=0, atol=0)
    assert encoder.diagnostic_sample_index == before_index
    assert encoder._diagnostic_forward_index == before_forward_index
    assert encoder._diagnostic_site_forward_counts == before_site_counts
    after_values = _diag_snapshot(encoder)
    assert before_values.keys() == after_values.keys()
    for key in before_values:
        if isinstance(before_values[key], torch.Tensor):
            torch.testing.assert_close(before_values[key], after_values[key], rtol=0, atol=0, msg=key)
        else:
            assert before_values[key] == after_values[key]


def test_probe_inventory_marks_only_active_goalonly_branches_applicable():
    encoder, _ = _make_goalonly()
    specs = {item["probe_name"]: item for item in encoder.policy_shadow_probe_applicability()}
    assert specs["st_spatial_off"]["applicable"]
    assert specs["st_temporal_current_only"]["applicable"]
    assert specs["st_social_ego_only"]["applicable"]
    assert specs["rt_intent_injection_off"]["applicable"]
    assert specs["rt_route_readout_zero"]["applicable"]
    assert not specs["topology_actor_intent_off"]["applicable"]
    assert not specs["topology_relations_off"]["applicable"]
    assert specs["topology_goal_off"]["applicable"]
    assert specs["route_reachability_off"]["applicable"]
    assert all(not specs[f"slot_{name}_zero"]["applicable"] for name in ("ego", "social", "route"))


def test_active_goalonly_probes_keep_shape_and_finite_values_without_cache_mutation():
    encoder, observations = _make_goalonly()
    encoder.configure_diagnostics(True, sample_every=1, source="eval_policy")
    with torch.no_grad():
        _ = encoder(observations)
    before_index = encoder.diagnostic_sample_index
    before_forward_index = encoder._diagnostic_forward_index
    before_site_counts = dict(encoder._diagnostic_site_forward_counts)
    applicable = [
        spec["probe_name"]
        for spec in encoder.policy_shadow_probe_applicability()
        if spec["applicable"]
    ]
    for name in applicable:
        with encoder.policy_shadow_probe(name):
            with torch.no_grad():
                output = encoder(observations)
        assert output.shape == (observations["trajectory"].shape[0], encoder.features_dim), name
        assert torch.isfinite(output).all(), name
        assert encoder.diagnostic_sample_index == before_index, name
        assert encoder._diagnostic_forward_index == before_forward_index, name
        assert encoder._diagnostic_site_forward_counts == before_site_counts, name


def test_slot_zero_probe_changes_only_the_selected_postprojection_slot():
    encoder, observations = _make_3slot()
    with encoder.policy_shadow_probe(None) as baseline_capture:
        with torch.no_grad():
            baseline = encoder.forward_tokens(observations)
    assert isinstance(baseline, StructuredLatent)
    for name, key in (
        ("slot_ego_zero", "slot_ego"),
        ("slot_social_zero", "slot_social"),
        ("slot_route_zero", "slot_route"),
    ):
        with encoder.policy_shadow_probe(name) as captured:
            with torch.no_grad():
                output = encoder.forward_tokens(observations)
        assert isinstance(output, StructuredLatent)
        target = captured[key]
        assert torch.count_nonzero(target) == 0
        for other in ("ego", "social", "route"):
            out_tensor = getattr(output, f"z_{other}")
            base_tensor = getattr(baseline, f"z_{other}")
            if f"slot_{other}" == key:
                assert torch.count_nonzero(out_tensor) == 0
            else:
                torch.testing.assert_close(out_tensor, base_tensor, rtol=0, atol=0)
        assert torch.isfinite(output.tensor).all()


def test_3slot_probe_inventory_and_output_dimension_are_stable():
    encoder, observations = _make_3slot()
    specs = {item["probe_name"]: item for item in encoder.policy_shadow_probe_applicability()}
    assert all(specs[f"slot_{name}_zero"]["applicable"] for name in ("ego", "social", "route"))
    assert not specs["topology_actor_intent_off"]["applicable"]
    assert not specs["topology_relations_off"]["applicable"]
    assert not specs["topology_goal_off"]["applicable"]
    for name in ("st_spatial_off", "st_temporal_current_only", "st_social_ego_only", "rt_intent_injection_off", "rt_route_readout_zero"):
        with encoder.policy_shadow_probe(name):
            with torch.no_grad():
                output = encoder(observations)
        assert output.shape == (2, 128)
        assert torch.isfinite(output).all()
