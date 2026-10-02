from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
from algos.sb3_torch.policy_shadow_probes import PROBE_NAMES
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


def _space(*, conflict: bool = False) -> gym.spaces.Dict:
    spaces = {
        "trajectory": gym.spaces.Box(
            -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
        ),
        "map": gym.spaces.Box(
            -1000.0, 1000.0, shape=(12, 10, 5), dtype=np.float32
        ),
    }
    if conflict:
        spaces["conflict_timing"] = gym.spaces.Box(
            -2.0, 2.0, shape=(6, 20), dtype=np.float32
        )
    return gym.spaces.Dict(spaces)


def _graph_and_lane_ids():
    spec = PAPER_SCENARIOS["left_turn"]
    graph, info = build_topology_graph_v2(
        spec.network_path,
        coordinate_offset=spec.coordinate_offset,
        return_info=True,
    )
    return graph, tuple(info.lane_ids)


def _observations(*, include_conflict: bool = True, single_neighbor: bool = True):
    timeline = torch.arange(10, dtype=torch.float32)
    trajectory = torch.zeros(6, 10, 5)
    map_state = torch.zeros(6, 2, 10, 5)
    for actor in range(6):
        trajectory[actor, :, 0] = 20.0 + actor * 3.0 + timeline * 0.4
        trajectory[actor, :, 1] = 5.0 + actor * 2.0 + timeline * 0.1
        trajectory[actor, :, 2] = 0.03 * actor
        trajectory[actor, :, 3] = 4.0 + 0.2 * actor
        trajectory[actor, :, 4] = 0.1 * actor
        for path in range(2):
            map_state[actor, path, :, 0] = 20.0 + actor * 3.0 + timeline
            map_state[actor, path, :, 1] = 5.0 + actor * 2.0 + path * 2.0
            map_state[actor, path, :, 2] = 0.03 * actor
            map_state[actor, path, :, 3] = float(path == 0)
            map_state[actor, path, :, 4] = float(path == 1)
    if single_neighbor:
        trajectory[2:] = 0.0
        map_state[2:] = 0.0
    result = {
        "trajectory": trajectory.unsqueeze(0),
        "map": map_state.reshape(12, 10, 5).unsqueeze(0),
    }
    if include_conflict:
        conflict = torch.zeros(6, 20)
        conflict[1, 0] = 1.0
        conflict[1, 1:7] = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.8, 0.7])
        conflict[1, 7:15] = torch.tensor(
            [0.3, 0.5, 0.6, 0.8, 1.0, 1.0, 0.2, 0.1]
        )
        conflict[1, 15:] = torch.tensor([0.2, 0.3, 0.0, 0.0, 0.5])
        result["conflict_timing"] = conflict.unsqueeze(0)
    return result


def _encoder(graph, lane_ids, *, conflict: bool):
    return IncrementalTopoEncoder(
        _space(conflict=conflict),
        topology_graph=graph,
        topology_lane_ids=lane_ids,
        use_route=True,
        use_topology=False,
        use_slots=False,
        use_route_conflict_timing=conflict,
        random_augmentation=False,
    )


def _social_inputs(encoder, observations, probe=None):
    captured = {}

    def hook(_module, inputs):
        captured["query"] = inputs[0].detach().clone()
        captured["key_values"] = inputs[1].detach().clone()

    handle = encoder.social_attention.register_forward_pre_hook(hook)
    try:
        if probe is None:
            output = encoder.forward_tokens(observations)
        else:
            with encoder.policy_shadow_probe(probe):
                output = encoder.forward_tokens(observations)
    finally:
        handle.remove()
    return output, captured


def test_conflict_branch_zero_init_preserves_shared_st_rt_forward_and_rng():
    graph, lane_ids = _graph_and_lane_ids()
    torch.manual_seed(29)
    baseline = _encoder(graph, lane_ids, conflict=False).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    conflict = _encoder(graph, lane_ids, conflict=True).eval()
    conflict_rng = torch.get_rng_state().clone()
    assert torch.equal(baseline_rng, conflict_rng)

    baseline_state = baseline.state_dict()
    conflict_state = conflict.state_dict()
    shared_keys = set(baseline_state) & set(conflict_state)
    assert shared_keys == set(baseline_state)
    for key in shared_keys:
        torch.testing.assert_close(
            baseline_state[key], conflict_state[key], rtol=0.0, atol=0.0
        )
    missing, unexpected = conflict.load_state_dict(baseline_state, strict=False)
    assert unexpected == []
    assert all(key.startswith("route_conflict_timing_encoder.") for key in missing)

    observations = _observations()
    with torch.no_grad():
        expected, baseline_inputs = _social_inputs(baseline, observations)
        actual, conflict_inputs = _social_inputs(conflict, observations)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        conflict_inputs["query"], baseline_inputs["query"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        conflict_inputs["key_values"],
        baseline_inputs["key_values"],
        rtol=0.0,
        atol=0.0,
    )
    assert not any(
        key.startswith("route_conflict_timing_encoder.") for key in baseline_state
    )


def test_conflict_residual_reaches_single_social_value_and_shadow_probes_are_exact():
    graph, lane_ids = _graph_and_lane_ids()
    encoder = _encoder(graph, lane_ids, conflict=True).eval()
    # Make the test branch depend only on ego-entry-time (column 7), so the
    # times-off probe has a deterministic, narrowly specified effect.
    with torch.no_grad():
        first, final = encoder.route_conflict_timing_encoder[0], encoder.route_conflict_timing_encoder[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 7] = 1.0
        final.weight.zero_()
        final.bias.zero_()
        final.weight[0, 0] = 1.0

    observations = _observations(single_neighbor=True)
    with torch.no_grad():
        active, active_inputs = _social_inputs(encoder, observations)
        disabled, disabled_inputs = _social_inputs(
            encoder, observations, "route_conflict_off"
        )
        times_off, times_inputs = _social_inputs(
            encoder, observations, "route_conflict_times_off"
        )
    assert not torch.equal(
        active_inputs["key_values"][:, 1], disabled_inputs["key_values"][:, 1]
    )
    # The intervention is on the one valid relation row only: ego query and all
    # other rows remain precisely unchanged.
    torch.testing.assert_close(
        active_inputs["query"], disabled_inputs["query"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        active_inputs["key_values"][:, 0],
        disabled_inputs["key_values"][:, 0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        active_inputs["key_values"][:, 2:],
        disabled_inputs["key_values"][:, 2:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(disabled, times_off, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        disabled_inputs["key_values"], times_inputs["key_values"], rtol=0.0, atol=0.0
    )
    assert not torch.equal(active, disabled)

    encoder.zero_grad(set_to_none=True)
    encoder.forward_tokens(observations).square().mean().backward()
    final_weight_grad = encoder.route_conflict_timing_encoder[2].weight.grad
    assert final_weight_grad is not None
    assert torch.count_nonzero(final_weight_grad).item() > 0


def test_conflict_diagnostics_have_explicit_valid_denominators_and_probe_na():
    graph, lane_ids = _graph_and_lane_ids()
    encoder = _encoder(graph, lane_ids, conflict=True)
    encoder.configure_diagnostics(True, sample_every=1, source="eval_policy")
    encoder.forward_tokens(_observations())
    values = encoder.diagnostic_values()
    assert values["route_conflict_timing_active"].item() == 1.0
    assert values["route_conflict_timing_valid_actor_count"].item() == 1.0
    assert values["route_conflict_timing_candidate_actor_count"].item() == 1.0
    assert values["route_conflict_timing_coverage_valid"].item() == 1.0
    assert values["route_conflict_timing_delta_valid"].item() == 1.0
    assert values["route_conflict_timing_delta_to_base_valid"].item() == 1.0
    assert values["route_conflict_timing_delta_rms"].item() == 0.0

    applicability = {
        row["probe_name"]: row
        for row in encoder.policy_shadow_probe_applicability()
    }
    assert applicability["route_conflict_off"]["applicable"] is True
    assert applicability["route_conflict_times_off"]["applicable"] is True
    assert "route_conflict_off" in PROBE_NAMES
    assert "route_conflict_times_off" in PROBE_NAMES

    inactive = _encoder(graph, lane_ids, conflict=False)
    inactive_rows = {
        row["probe_name"]: row
        for row in inactive.policy_shadow_probe_applicability()
    }
    assert inactive_rows["route_conflict_off"]["applicable"] is False
    assert inactive_rows["route_conflict_times_off"]["applicable"] is False


def test_d1_registry_and_method_wrapper_contracts_are_explicit(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "fast-developer"))
    d1 = importlib.import_module("train_intersection_yield_v2_d1")
    routeact = "sac_mlp_d1_st_rt_routeact_v1"
    conflicttime = "sac_mlp_d1_st_rt_conflicttime_v1"
    assert routeact in d1.METHODS and conflicttime in d1.METHODS
    assert d1.PARENT[routeact] == d1.PARENT[conflicttime] == "sac_mlp"
    assert d1.D1_CONFIG[routeact]["use_route_action_veto"] is True
    assert d1.D1_CONFIG[routeact]["use_route_conflict_timing"] is False
    assert d1.D1_CONFIG[conflicttime]["use_route"] is True
    assert d1.D1_CONFIG[conflicttime]["use_topology"] is False
    assert d1.D1_CONFIG[conflicttime]["use_slots"] is False
    assert d1.D1_CONFIG[conflicttime]["use_route_conflict_timing"] is True
    assert d1.D1_CONFIG["sac_mlp_d1_st_rt"]["use_route_action_veto"] is False
    assert d1.D1_CONFIG["sac_mlp_d1_st_rt"]["use_route_conflict_timing"] is False

    calls = []

    class Wrapper:
        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs
            calls.append((type(self).__name__, kwargs))

    action_module = importlib.import_module("envs.sumo.route_action_consistency")
    monkeypatch.setattr(action_module, "RouteActionConsistencyWrapper", Wrapper)
    fake_conflict = types.ModuleType("envs.sumo.conflict_timing_observation")
    fake_conflict.RouteConflictTimingObservationWrapper = Wrapper
    monkeypatch.setitem(
        sys.modules, "envs.sumo.conflict_timing_observation", fake_conflict
    )
    inner = object()
    routeact_env = d1._wrap_method_env(inner, routeact, tmp_path, "train")
    assert routeact_env.env is inner
    assert calls[-1][1] == {
        "run_dir": tmp_path,
        "phase": "train",
        "method": routeact,
    }
    conflict_env = d1._wrap_method_env(inner, conflicttime, tmp_path, "eval")
    assert conflict_env.env is inner
    assert calls[-1][1] == {
        "diagnostics_directory": tmp_path / "diagnostics" / "eval",
        "phase": "eval",
    }
