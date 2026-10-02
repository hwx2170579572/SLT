from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.sac import SceneRepresentationSAC
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph import MAX_TOPO_EDGES, MAX_TOPO_NODES


class _PolicyOnlyDictEnv(gym.Env):
    """Tiny dictionary-observation env for policy construction; no SUMO calls."""

    def __init__(self, *, route_capacity: int | None = None, valid_route_count: int = 0):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        spaces = {
            "trajectory": gym.spaces.Box(
                -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
            ),
            "map": gym.spaces.Box(
                -1000.0, 1000.0, shape=(12, 10, 5), dtype=np.float32
            ),
        }
        if route_capacity is not None:
            spaces["route_reachability"] = gym.spaces.Box(
                -1.0, 1.0, shape=(route_capacity,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)
        specification = PAPER_SCENARIOS["left_turn"]
        self.specification = SimpleNamespace(
            network_path=specification.network_path,
            coordinate_offset=specification.coordinate_offset,
            topology_max_nodes=(
                getattr(specification, "topology_max_nodes", None)
                or MAX_TOPO_NODES
            ),
            topology_max_edges=(
                getattr(specification, "topology_max_edges", None)
                or MAX_TOPO_EDGES
            ),
        )
        self._observation = self._make_observation(
            route_capacity=route_capacity,
            valid_route_count=valid_route_count,
        )

    @staticmethod
    def _make_observation(
        *, route_capacity: int | None, valid_route_count: int
    ) -> dict[str, np.ndarray]:
        timeline = np.arange(10, dtype=np.float32)
        trajectory = np.zeros((6, 10, 5), dtype=np.float32)
        map_state = np.zeros((6, 2, 10, 5), dtype=np.float32)
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
        result = {
            "trajectory": trajectory,
            "map": map_state.reshape(12, 10, 5),
        }
        if route_capacity is not None:
            labels = np.full((route_capacity,), -1.0, dtype=np.float32)
            labels[:valid_route_count] = 1.0
            result["route_reachability"] = labels
        return result

    def reset(self, *, seed: int | None = None, options: Any = None):
        super().reset(seed=seed)
        del options
        return {key: value.copy() for key, value in self._observation.items()}, {}

    def step(self, action):
        del action
        return (
            {key: value.copy() for key, value in self._observation.items()},
            0.0,
            False,
            False,
            {},
        )


def _import_d1_runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    fast_developer = root / "fast-developer"
    monkeypatch.syspath_prepend(str(fast_developer))
    return importlib.import_module("train_intersection_yield_v2_d1")


def _optimizer_parameter_ids(optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _assert_method_restored(method: str, model) -> None:
    extractor = model.critic.features_extractor
    if method == "sac_mlp_d1_st_rt_topo_goalonly_v1":
        assert extractor.use_topology is True
        assert extractor.use_topology_actor_intent is False
        assert extractor.use_topology_relations is False
        assert extractor.use_topology_goal is True
        assert extractor.use_route_reachability is True
        assert extractor.use_parameter_matched_nonlinear_slots is False
        assert extractor.active_readout_type == "joint_mlp"
        assert extractor.active_readout_parameter_count == 65792
        assert model.representation is None
        assert model.representation_optimizer is None
        critic_ids = _optimizer_parameter_ids(model.critic.optimizer)
        goal_params = list(extractor.goal_topology_attention.parameters())
        goal_params += list(extractor.topology_encoder.parameters())
        goal_params += list(extractor.mlp_output.parameters())
        goal_params += [extractor.goal_residual_scale]
        assert goal_params and all(id(parameter) in critic_ids for parameter in goal_params)
    else:
        assert extractor.use_topology is False
        assert extractor.use_topology_actor_intent is False
        assert extractor.use_topology_relations is False
        assert extractor.use_topology_goal is False
        assert extractor.use_route_reachability is False
        assert extractor.use_slots is True
        assert extractor.use_incremental_slots is True
        assert extractor.use_parameter_matched_nonlinear_slots is True
        assert extractor.active_readout_type == "parameter_matched_nonlinear_3slot"
        assert extractor.active_readout_parameter_count == 65792
        legacy_slot_count = sum(
            parameter.numel()
            for module in (
                extractor.ego_projection,
                extractor.social_projection,
                extractor.route_projection,
            )
            for parameter in module.parameters()
        )
        assert legacy_slot_count == 16512
        assert extractor.active_readout_parameter_count != (
            legacy_slot_count
            + sum(
                parameter.numel()
                for parameter in extractor.nonlinear_slot_projections.parameters()
            )
        )
        groups = extractor.diagnostic_parameter_groups()
        assert groups["slot_ego"] == ("nonlinear_slot_projections.ego",)
        assert groups["slot_social"] == ("nonlinear_slot_projections.social",)
        assert groups["slot_route"] == ("nonlinear_slot_projections.route",)
        nonlinear_params = list(extractor.nonlinear_slot_projections.parameters())
        critic_ids = _optimizer_parameter_ids(model.critic.optimizer)
        actor_ids = _optimizer_parameter_ids(model.actor.optimizer)
        assert nonlinear_params and all(
            id(parameter) in critic_ids for parameter in nonlinear_params
        )
        # SceneRepSACPolicy deliberately detaches the critic-owned encoder from
        # the actor optimizer; actor gradients train only the policy head.
        assert all(id(parameter) not in actor_ids for parameter in nonlinear_params)
        assert model.representation is None
        assert model.representation_optimizer is None


@pytest.mark.parametrize(
    "method",
    [
        "sac_mlp_d1_st_rt_topo_goalonly_v1",
        "sac_mlp_d1_st_rt_3slot_nonlinear_v1",
    ],
)
def test_new_d1_method_checkpoint_roundtrip_restores_kwargs_optimizer_and_actions(
    tmp_path, monkeypatch, method: str
) -> None:
    d1 = _import_d1_runner(monkeypatch)
    assert method in d1.PARENT and method in d1.D1_CONFIG
    monkeypatch.setattr(d1.base, "BUFFER_SIZE", 8)
    monkeypatch.setattr(d1.base, "BATCH_SIZE", 2)

    specification = PAPER_SCENARIOS["left_turn"]
    from envs.sumo.topology_graph_v2 import build_topology_graph_v2

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
    model = d1._build_model_d1(
        method,
        env,
        learning_starts=0,
        device="cpu",
    )
    assert isinstance(model, SceneRepresentationSAC)
    _assert_method_restored(method, model)
    observation, _ = env.reset(seed=7)
    with source_evaluation_augmentation(model):
        expected_action, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / method
    model.save(checkpoint)
    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    _assert_method_restored(method, restored)
    with source_evaluation_augmentation(restored):
        actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)


def test_legacy_policy_kwargs_without_new_switches_still_load(tmp_path, monkeypatch):
    d1 = _import_d1_runner(monkeypatch)
    monkeypatch.setattr(d1.base, "BUFFER_SIZE", 8)
    monkeypatch.setattr(d1.base, "BATCH_SIZE", 2)
    env = _PolicyOnlyDictEnv()
    method = "sac_mlp_d1_st_rt_topo"
    model = d1._build_model_d1(method, env, learning_starts=0, device="cpu")
    observation, _ = env.reset(seed=11)
    with source_evaluation_augmentation(model):
        expected_action, _ = model.predict(observation, deterministic=True)

    # Emulate the feature-extractor kwargs stored by pre-switch D1 checkpoints.
    legacy_kwargs = dict(model.policy_kwargs)
    legacy_feature_kwargs = dict(legacy_kwargs["features_extractor_kwargs"])
    for key in (
        "use_topology_actor_intent",
        "use_topology_relations",
        "use_topology_goal",
        "use_parameter_matched_nonlinear_slots",
    ):
        legacy_feature_kwargs.pop(key, None)
    legacy_kwargs["features_extractor_kwargs"] = legacy_feature_kwargs
    model.policy_kwargs = legacy_kwargs
    checkpoint = tmp_path / "legacy_d1_topology_kwargs"
    model.save(checkpoint)
    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    extractor = restored.critic.features_extractor
    assert extractor.use_topology_actor_intent is True
    assert extractor.use_topology_relations is True
    assert extractor.use_topology_goal is True
    assert extractor.use_parameter_matched_nonlinear_slots is False
    with source_evaluation_augmentation(restored):
        actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
