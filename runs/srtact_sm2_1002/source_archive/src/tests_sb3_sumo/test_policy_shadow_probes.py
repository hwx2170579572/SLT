from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.policy_shadow_probes import (
    NOOP_PROBE_NAME,
    PROBE_NAMES,
    capture_policy_snapshot,
    run_policy_shadow_suite,
)
from algos.sb3_torch.sac import SceneRepresentationSAC
from tests_sb3_sumo.test_d1_method_checkpoint_roundtrip import (
    _PolicyOnlyDictEnv,
    _assert_method_restored,
    _import_d1_runner,
)


METHODS = (
    "sac_mlp_d1_st_rt_topo_goalonly_v1",
    "sac_mlp_d1_st_rt_3slot_nonlinear_v1",
)


def _build_model(method: str, monkeypatch):
    d1 = _import_d1_runner(monkeypatch)
    monkeypatch.setattr(d1.base, "BUFFER_SIZE", 8)
    monkeypatch.setattr(d1.base, "BATCH_SIZE", 2)
    from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
    from envs.sumo.topology_graph import MAX_TOPO_EDGES, MAX_TOPO_NODES
    from envs.sumo.topology_graph_v2 import build_topology_graph_v2

    specification = PAPER_SCENARIOS["left_turn"]
    graph, _ = build_topology_graph_v2(
        specification.network_path,
        max_nodes=int(
            getattr(specification, "topology_max_nodes", None) or MAX_TOPO_NODES
        ),
        coordinate_offset=specification.coordinate_offset,
        return_info=True,
    )
    route_method = bool(d1.D1_CONFIG[method]["use_route_reachability"])
    env = _PolicyOnlyDictEnv(
        route_capacity=int(graph.node_mask.shape[0]) if route_method else None,
        valid_route_count=int(graph.node_mask.sum()) if route_method else 0,
    )
    model = d1._build_model_d1(method, env, learning_starts=0, device="cpu")
    assert isinstance(model, SceneRepresentationSAC)
    _assert_method_restored(method, model)
    return model, env


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    try:
        result = left == right
        return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())
    except Exception:
        return repr(left) == repr(right)


def _runtime_cache_snapshot(model):
    result = {}
    for module_name, module in model.policy.named_modules():
        for name, value in vars(module).items():
            lowered = name.lower()
            if (
                name == "_diagnostics"
                or "diagnostic" in lowered
                or "last_" in lowered
                or "_last" in lowered
                or "cache" in lowered
                or "capture" in lowered
            ):
                result[(module_name, name)] = _clone(value)
    return result


def _runtime_snapshot(model):
    policy = model.policy
    return {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state().clone(),
        "modes": [(module, module.training) for module in policy.modules()],
        "state_dict": {key: value.detach().clone() for key, value in policy.state_dict().items()},
        "grads": [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in policy.parameters()
        ],
        "caches": _runtime_cache_snapshot(model),
        "hook_counts": [
            (module, len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in policy.modules()
        ],
        "action_dist": getattr(model.actor, "action_dist", None),
        "encoder_shadow_context": (
            model.critic.features_extractor._policy_shadow_active,
            model.critic.features_extractor._policy_shadow_probe_name,
            model.critic.features_extractor._policy_shadow_capture,
        ),
    }


def _assert_runtime_equal(model, before):
    after = _runtime_snapshot(model)
    assert _equal(before["rng_python"], after["rng_python"])
    assert _equal(before["rng_numpy"], after["rng_numpy"])
    assert torch.equal(before["rng_torch"], after["rng_torch"])
    assert len(before["modes"]) == len(after["modes"])
    for (module_before, mode_before), (module_after, mode_after) in zip(
        before["modes"], after["modes"]
    ):
        assert module_before is module_after
        assert mode_before == mode_after
    assert before["state_dict"].keys() == after["state_dict"].keys()
    assert all(torch.equal(before["state_dict"][key], after["state_dict"][key]) for key in before["state_dict"])
    assert len(before["grads"]) == len(after["grads"])
    for expected, actual in zip(before["grads"], after["grads"]):
        assert (expected is None and actual is None) or (
            expected is not None and actual is not None and torch.equal(expected, actual)
        )
    assert before["caches"].keys() == after["caches"].keys()
    changed = [
        key for key in before["caches"]
        if not _equal(before["caches"][key], after["caches"][key])
    ]
    assert not changed, f"runtime diagnostic/cache state changed: {changed}"
    assert len(before["hook_counts"]) == len(after["hook_counts"])
    for (module_before, fwd_before, pre_before), (module_after, fwd_after, pre_after) in zip(
        before["hook_counts"], after["hook_counts"]
    ):
        assert module_before is module_after
        assert (fwd_before, pre_before) == (fwd_after, pre_after)
    assert before["action_dist"] is after["action_dist"]
    assert _equal(before["encoder_shadow_context"], after["encoder_shadow_context"])


@pytest.mark.parametrize("method", METHODS)
def test_shadow_capture_and_suite_preserve_policy_runtime_and_report_na(method, monkeypatch):
    model, env = _build_model(method, monkeypatch)
    # The policy intervention must target the encoder actually called by the
    # actor; the critic owns a separate SAC feature-extractor instance.
    encoder = model.actor.features_extractor
    encoder.configure_diagnostics(True, sample_every=1, source="eval_policy")
    observation, _ = env.reset(seed=101)

    # Populate normal diagnostic counters and cache fields before probing.
    with source_evaluation_augmentation(model):
        model.predict(observation, deterministic=True)
    before = _runtime_snapshot(model)

    baseline = capture_policy_snapshot(model, observation, encoder=encoder)
    assert baseline["valid"] is True
    assert baseline["mean_pre_tanh"] is not None
    assert baseline["model_action"] is not None
    rows = run_policy_shadow_suite(
        model,
        encoder,
        observation,
        baseline_snapshot=baseline,
        metadata={"method": method, "seed": 101},
    )

    assert [row["probe_name"] for row in rows] == list(PROBE_NAMES)
    assert all(row["schema_version"] == "policy_shadow_probes_v1" for row in rows)
    assert all(row["method"] == method and row["seed"] == 101 for row in rows)
    assert all(row["valid"] for row in rows if row["applicable"])
    for row in rows:
        if not row["applicable"]:
            assert row["valid"] is False
            assert row["invalid_reason"] == "branch_inactive"
    if method == "sac_mlp_d1_st_rt_topo_goalonly_v1":
        assert next(row for row in rows if row["probe_name"] == "route_reachability_off")["applicable"]
        assert not next(row for row in rows if row["probe_name"] == "slot_ego_zero")["applicable"]
    else:
        assert not next(row for row in rows if row["probe_name"] == "route_reachability_off")["applicable"]
        assert all(next(row for row in rows if row["probe_name"] == f"slot_{slot}_zero")["applicable"] for slot in ("ego", "social", "route"))

    _assert_runtime_equal(model, before)


@pytest.mark.parametrize("method", METHODS)
def test_shadow_noop_is_equivalent_and_context_is_not_serialized(method, monkeypatch):
    model, env = _build_model(method, monkeypatch)
    encoder = model.actor.features_extractor
    observation, _ = env.reset(seed=202)
    with source_evaluation_augmentation(model):
        expected, _ = model.predict(observation, deterministic=True)
        with encoder.policy_shadow_probe(NOOP_PROBE_NAME):
            actual, _ = model.predict(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    assert encoder._policy_shadow_active is False
    assert encoder._policy_shadow_probe_name is None
    assert encoder._policy_shadow_capture is None
    assert all("policy_shadow" not in key for key in model.policy.state_dict())
