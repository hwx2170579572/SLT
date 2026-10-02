from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
from algos.sb3_torch.incremental_topo_encoder_attn import IncrementalTopoEncoderAttn
from algos.sb3_torch.topo_temporal_features import StructuredLatent
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


def _space(
    *, route_reachability: bool = False, route_capacity: int = 16
) -> gym.spaces.Dict:
    spaces = {
        "trajectory": gym.spaces.Box(
            -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
        ),
        "map": gym.spaces.Box(
            -1000.0, 1000.0, shape=(12, 10, 5), dtype=np.float32
        ),
    }
    if route_reachability:
        spaces["route_reachability"] = gym.spaces.Box(
            -1.0, 1.0, shape=(route_capacity,), dtype=np.float32
        )
    return gym.spaces.Dict(spaces)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    trajectory = torch.zeros(batch_size, 6, 10, 5)
    timeline = torch.arange(10, dtype=torch.float32)
    for batch in range(batch_size):
        for actor in range(6):
            trajectory[batch, actor, :, 0] = (
                20.0 + batch + actor * 3.0 + timeline * 0.4
            )
            trajectory[batch, actor, :, 1] = 5.0 + actor * 2.0 + timeline * 0.1
            trajectory[batch, actor, :, 2] = 0.03 * actor
            trajectory[batch, actor, :, 3] = 4.0 + 0.2 * actor
            trajectory[batch, actor, :, 4] = 0.1 * actor
    map_state = torch.zeros(batch_size, 6, 2, 10, 5)
    for batch in range(batch_size):
        for actor in range(6):
            for path in range(2):
                map_state[batch, actor, path, :, 0] = (
                    20.0 + batch + actor * 3.0 + timeline
                )
                map_state[batch, actor, path, :, 1] = (
                    5.0 + actor * 2.0 + path * 2.0
                )
                map_state[batch, actor, path, :, 2] = 0.03 * actor
                map_state[batch, actor, path, :, 3] = float(path == 0)
                map_state[batch, actor, path, :, 4] = float(path == 1)
    return {
        "trajectory": trajectory,
        "map": map_state.reshape(batch_size, 12, 10, 5),
    }


def _graph_and_lane_ids():
    spec = PAPER_SCENARIOS["left_turn"]
    graph, graph_info = build_topology_graph_v2(
        spec.network_path,
        coordinate_offset=spec.coordinate_offset,
        return_info=True,
    )
    return graph, tuple(graph_info.lane_ids)


def _encoder(
    graph,
    lane_ids,
    *,
    use_topology: bool,
    use_slots: bool = False,
    use_incremental_slots: bool = False,
    use_route_reachability: bool = False,
    actor_intent: bool | None = None,
    relations: bool | None = None,
    goal: bool | None = None,
    nonlinear_slots: bool = False,
) -> IncrementalTopoEncoder:
    return IncrementalTopoEncoder(
        _space(
            route_reachability=use_route_reachability,
            route_capacity=int(graph.node_mask.shape[0]),
        ),
        topology_graph=graph,
        topology_lane_ids=lane_ids,
        use_route=True,
        use_topology=use_topology,
        use_topology_actor_intent=actor_intent,
        use_topology_relations=relations,
        use_topology_goal=goal,
        use_slots=use_slots,
        use_incremental_slots=use_incremental_slots,
        use_route_reachability=use_route_reachability,
        use_parameter_matched_nonlinear_slots=nonlinear_slots,
        random_augmentation=False,
    )


def test_topo_goal_only_uses_route_legal_goal_attention_without_actor_or_edge_topology():
    graph, lane_ids = _graph_and_lane_ids()
    encoder = _encoder(
        graph,
        lane_ids,
        use_topology=True,
        use_route_reachability=True,
        actor_intent=False,
        relations=False,
        goal=True,
    ).train()
    valid_count = int(encoder.topology_node_mask.sum())
    capacity = int(encoder.topology_node_mask.numel())
    assert valid_count > 0
    observations = _batch()
    labels = torch.full((2, capacity), -1.0)
    labels[:, :valid_count] = 0.0
    labels[0, 0] = 1.0
    labels[1, valid_count - 1] = 1.0
    observations["route_reachability"] = labels

    captured: dict[str, torch.Tensor] = {}
    calls = {"vehicle_topology": 0}

    def capture_goal_mask(_module, inputs) -> None:
        captured["goal_mask"] = inputs[2].detach().clone()

    def count_vehicle_topology(_module, _inputs, _output) -> None:
        calls["vehicle_topology"] += 1

    goal_hook = encoder.goal_topology_attention.register_forward_pre_hook(
        capture_goal_mask
    )
    vehicle_hook = encoder.topology_attention.register_forward_hook(
        count_vehicle_topology
    )
    try:
        latent = encoder.forward_tokens(observations)
    finally:
        goal_hook.remove()
        vehicle_hook.remove()

    assert isinstance(latent, torch.Tensor)
    assert captured["goal_mask"].shape == (2, capacity)
    assert torch.equal(captured["goal_mask"], labels > 0.5)
    assert calls["vehicle_topology"] == 0
    assert encoder.use_topology_actor_intent is False
    assert encoder.use_topology_relations is False
    assert encoder.use_topology_goal is True

    latent.square().mean().backward()
    assert encoder.goal_residual_scale.grad is not None
    assert torch.isfinite(encoder.goal_residual_scale.grad).all()
    assert encoder.topology_residual_scale.grad is None
    assert all(parameter.grad is None for parameter in encoder.topology_attention.parameters())
    assert any(parameter.grad is not None for parameter in encoder.topology_encoder.parameters())


def test_goal_only_zero_beta_and_all_topo_switches_off_match_st_rt_outputs_and_shared_grads():
    graph, lane_ids = _graph_and_lane_ids()
    torch.manual_seed(41)
    baseline = _encoder(graph, lane_ids, use_topology=False).eval()
    torch.manual_seed(42)
    goal_only = _encoder(
        graph,
        lane_ids,
        use_topology=True,
        use_route_reachability=True,
        actor_intent=False,
        relations=False,
        goal=True,
    ).eval()
    goal_only.load_state_dict(baseline.state_dict(), strict=True)
    valid_count = int(goal_only.topology_node_mask.sum())
    capacity = int(goal_only.topology_node_mask.numel())
    observations = _batch()
    labels = torch.full((2, capacity), -1.0)
    labels[:, :valid_count] = 1.0
    observations["route_reachability"] = labels

    with torch.no_grad():
        goal_only.goal_residual_scale.zero_()
        expected = baseline.forward_tokens(observations)
        actual = goal_only.forward_tokens(observations)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Turning every topology branch off must recover the ordinary ST-RT path.
    all_off = _encoder(
        graph,
        lane_ids,
        use_topology=True,
        actor_intent=False,
        relations=False,
        goal=False,
    ).eval()
    all_off.load_state_dict(baseline.state_dict(), strict=True)
    torch.testing.assert_close(
        all_off.forward_tokens(_batch()), baseline.forward_tokens(_batch()), rtol=0, atol=0
    )

    baseline.zero_grad(set_to_none=True)
    all_off.zero_grad(set_to_none=True)
    baseline.forward_tokens(_batch()).square().mean().backward()
    all_off.forward_tokens(_batch()).square().mean().backward()
    baseline_params = dict(baseline.named_parameters())
    all_off_params = dict(all_off.named_parameters())
    assert baseline_params.keys() == all_off_params.keys()
    for name in baseline_params:
        left, right = baseline_params[name].grad, all_off_params[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)


def test_route_reachability_unknown_and_empty_query_use_finite_fallback():
    graph, lane_ids = _graph_and_lane_ids()
    encoder = _encoder(
        graph,
        lane_ids,
        use_topology=True,
        use_route_reachability=True,
        actor_intent=False,
        relations=False,
        goal=True,
    )
    valid_count = int(encoder.topology_node_mask.sum())
    capacity = int(encoder.topology_node_mask.numel())
    empty_query = torch.zeros(2, capacity, dtype=torch.bool)
    labels = torch.full((2, capacity), float("nan"))
    selected, state = encoder._route_reachability_goal_mask(empty_query, labels)
    expected_valid = encoder.topology_node_mask[None, :].expand(2, -1)
    assert torch.equal(selected, expected_valid)
    assert state["unknown"][:, :valid_count].all()
    assert state["safe_fallback"].all()
    assert state["bypassed"].all()


def test_parameter_matched_nonlinear_three_slot_head_is_65792_params_and_receives_gradients():
    graph, lane_ids = _graph_and_lane_ids()
    torch.manual_seed(15)
    encoder = _encoder(
        graph,
        lane_ids,
        use_topology=False,
        use_slots=True,
        use_incremental_slots=True,
        nonlinear_slots=True,
    ).train()
    assert encoder.active_readout_type == "parameter_matched_nonlinear_3slot"
    assert encoder.active_readout_parameter_count == 65792
    assert encoder.nonlinear_slot_projections is not None
    assert isinstance(encoder.nonlinear_slot_projections["ego"][0], nn.Linear)
    assert encoder.nonlinear_slot_projections["ego"][0].in_features == 128
    assert encoder.nonlinear_slot_projections["ego"][0].out_features == 132
    assert encoder.nonlinear_slot_projections["ego"][2].out_features == 32
    assert encoder.nonlinear_slot_projections["social"][0].out_features == 120
    assert encoder.nonlinear_slot_projections["social"][2].out_features == 64
    assert encoder.nonlinear_slot_projections["route"][0].out_features == 132
    assert encoder.nonlinear_slot_projections["route"][2].out_features == 32
    groups = encoder.diagnostic_parameter_groups()
    assert groups["slot_ego"] == ("nonlinear_slot_projections.ego",)
    assert groups["slot_social"] == ("nonlinear_slot_projections.social",)
    assert groups["slot_route"] == ("nonlinear_slot_projections.route",)

    latent = encoder.forward_tokens(_batch())
    assert isinstance(latent, StructuredLatent)
    assert latent.z_ego.shape == (2, 32)
    assert latent.z_social.shape == (2, 64)
    assert latent.z_route.shape == (2, 32)
    latent.tensor.square().mean().backward()
    for slot in ("ego", "social", "route"):
        grads = [
            parameter.grad
            for parameter in encoder.nonlinear_slot_projections[slot].parameters()
        ]
        assert grads and all(grad is not None for grad in grads), slot
        assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)


def test_nonlinear_head_does_not_change_legacy_default_rng_or_state_dict_contract():
    graph, lane_ids = _graph_and_lane_ids()
    kwargs = dict(
        use_topology=False,
        use_slots=True,
        use_incremental_slots=True,
    )
    torch.manual_seed(1234)
    legacy = _encoder(graph, lane_ids, **kwargs)
    legacy_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(1234)
    nonlinear = _encoder(graph, lane_ids, nonlinear_slots=True, **kwargs)
    nonlinear_rng = torch.random.get_rng_state().clone()
    assert torch.equal(legacy_rng, nonlinear_rng)
    assert not any(key.startswith("nonlinear_slot_projections.") for key in legacy.state_dict())
    legacy.load_state_dict(legacy.state_dict(), strict=True)
    common_legacy = legacy.state_dict()
    common_nonlinear = nonlinear.state_dict()
    for key, value in common_legacy.items():
        torch.testing.assert_close(value, common_nonlinear[key], rtol=0, atol=0)


def test_legacy_learned_edge_attention_still_accepts_geometry_only_incremental_path():
    graph, lane_ids = _graph_and_lane_ids()
    encoder = IncrementalTopoEncoderAttn(
        _space(),
        topology_graph=graph,
        topology_lane_ids=lane_ids,
        use_route=False,
        use_topology=False,
        use_slots=False,
        random_augmentation=False,
    ).eval()
    with torch.no_grad():
        output = encoder.forward_tokens(_batch())
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()
