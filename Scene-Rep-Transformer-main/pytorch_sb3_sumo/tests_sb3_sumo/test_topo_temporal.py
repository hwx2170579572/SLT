from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos.sb3_torch import (
    RawStepControlCallback,
    SceneRepresentationSAC,
    StructuredGraphRepresentationObjective,
    StructuredLatent,
    TopoTemporalGraphExtractor,
)
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from configs.sb3_configs import make_model, source_action_repeat
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph import PaddedTopologyGraph, build_topology_graph
from tools.diagnose_slot_action_sensitivity import _action_summary, _risk_masks
from tools.reevaluate_checkpoint_curve import _checkpoint_step, _paired_protocol, _valid_existing


def _space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "trajectory": gym.spaces.Box(
                -1000.0, 1000.0, shape=(6, 10, 5), dtype=np.float32
            ),
            "map": gym.spaces.Box(
                -1000.0, 1000.0, shape=(12, 10, 5), dtype=np.float32
            ),
        }
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    trajectory = torch.zeros(batch_size, 6, 10, 5)
    timeline = torch.arange(10, dtype=torch.float32)
    for batch in range(batch_size):
        for actor in range(6):
            trajectory[batch, actor, :, 0] = 20.0 + batch + actor * 3.0 + timeline * 0.4
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
                map_state[batch, actor, path, :, 1] = 5.0 + actor * 2.0 + path * 2.0
                map_state[batch, actor, path, :, 2] = 0.03 * actor
                map_state[batch, actor, path, :, 3] = float(path == 0)
                map_state[batch, actor, path, :, 4] = float(path == 1)
    return {"trajectory": trajectory, "map": map_state.reshape(batch_size, 12, 10, 5)}


def _graph() -> PaddedTopologyGraph:
    specification = PAPER_SCENARIOS["left_turn"]
    return build_topology_graph(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
    )


def _copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    assert source_parameters.keys() == target_parameters.keys()
    with torch.no_grad():
        for name, parameter in source_parameters.items():
            target_parameters[name].copy_(parameter)


def _extractor(
    graph: PaddedTopologyGraph,
    *,
    use_topology: bool = True,
    normalize_slots: bool = False,
):
    return TopoTemporalGraphExtractor(
        _space(),
        topology_graph=graph if use_topology else None,
        random_augmentation=False,
        use_topology=use_topology,
        normalize_slots=normalize_slots,
    )


def test_topo_extractor_shape_finiteness_and_diagnostics() -> None:
    extractor = _extractor(_graph()).eval()
    output = extractor(_batch())
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()
    diagnostics = extractor.diagnostic_values()
    assert set(diagnostics) == {
        "topology_attention_entropy",
        "latent_std",
        "ego_latent_std",
        "social_latent_std",
        "route_latent_std",
        "slot_scale_ratio",
        "graph_mean_edge_weight",
    }
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert extractor.last_topology_attention is not None


def test_balanced_slots_have_per_sample_unit_scale_without_affine_escape() -> None:
    extractor = _extractor(_graph(), normalize_slots=True).eval()
    latent = extractor.forward_tokens(_batch())
    for slot in (latent.z_ego, latent.z_social, latent.z_route):
        torch.testing.assert_close(
            slot.mean(dim=-1), torch.zeros(slot.shape[0]), atol=2e-5, rtol=0.0
        )
        torch.testing.assert_close(
            slot.std(dim=-1, unbiased=False),
            torch.ones(slot.shape[0]),
            atol=2e-4,
            rtol=0.0,
        )
    assert not any(extractor.slot_norms[name].elementwise_affine for name in extractor.slot_norms)


def test_action_diagnostic_uses_sign_neutral_lane_names() -> None:
    actions = np.asarray([[-1.0, -0.8], [0.0, 0.0], [1.0, 0.8]], dtype=np.float32)
    summary = _action_summary(actions, np.ones(3, dtype=bool))
    assert summary["lane_command_negative_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["lane_command_keep_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["lane_command_positive_rate"] == pytest.approx(1.0 / 3.0)
    assert "lane_command_left_rate" not in summary
    masks = _risk_masks(np.asarray([[2.0], [4.0], [8.0]], dtype=np.float32))
    assert [int(masks[name].sum()) for name in masks] == [3, 1, 1, 1]


def test_fixed_checkpoint_curve_receipt_is_hash_and_protocol_bound(tmp_path: Any) -> None:
    checkpoint = tmp_path / "topo_scene_raw_20000_steps.zip"
    assert _checkpoint_step(checkpoint) == 20_000
    receipt = tmp_path / "raw_20000.json"
    receipt.write_text(
        '{"contract":"topo-scene.fixed-checkpoint-curve/v1",'
        '"checkpoint_sha256":"abc","raw_steps":20000,"episodes":20,'
        '"evaluation_seed_start":10000,"traffic_protocol":"frozen_80_20",'
        '"algorithm":"topo_scene","scenario":"left_turn",'
        '"episode_limit_profile":"source","deterministic_policy":true}',
        encoding="utf-8",
    )
    assert _valid_existing(
        receipt,
        checkpoint_sha256="abc",
        raw_steps=20_000,
        episodes=20,
        seed_start=10_000,
        traffic_protocol="frozen_80_20",
        algorithm="topo_scene",
        scenario="left_turn",
        episode_limit_profile="source",
        deterministic_policy=True,
    ) is not None
    assert _valid_existing(
        receipt,
        checkpoint_sha256="changed",
        raw_steps=20_000,
        episodes=20,
        seed_start=10_000,
        traffic_protocol="frozen_80_20",
        algorithm="topo_scene",
        scenario="left_turn",
        episode_limit_profile="source",
        deterministic_policy=True,
    ) is None


def test_fixed_checkpoint_curve_requires_one_paired_protocol() -> None:
    base = {
        "algorithm": "scene_rep",
        "scenario": "left_turn",
        "episodes_per_checkpoint": 20,
        "evaluation_seed_start": 10_000,
        "evaluation_seeds": list(range(10_000, 10_020)),
        "traffic_protocol": "frozen_80_20",
        "episode_limit_profile": "source",
        "deterministic_policy": True,
    }
    paired = {**base, "algorithm": "topo_scene"}
    assert _paired_protocol([base, paired])["evaluation_seed_start"] == 10_000

    mismatched = {**paired, "evaluation_seed_start": 20_000}
    with pytest.raises(ValueError, match="not paired"):
        _paired_protocol([base, mismatched])


def test_padded_topology_nodes_and_edges_do_not_change_output() -> None:
    graph = _graph()
    lane_points = graph.lane_points.copy()
    lane_attrs = graph.lane_attrs.copy()
    edge_index = graph.edge_index.copy()
    edge_type = graph.edge_type.copy()
    lane_points[graph.valid_node_count :] = 9999.0
    lane_attrs[graph.valid_node_count :] = -777.0
    edge_index[:, graph.valid_edge_count :] = np.array([[0], [1]])
    edge_type[graph.valid_edge_count :] = 4
    changed_padding = PaddedTopologyGraph(
        lane_points=lane_points,
        lane_attrs=lane_attrs,
        node_mask=graph.node_mask.copy(),
        edge_index=edge_index,
        edge_type=edge_type,
        edge_mask=graph.edge_mask.copy(),
    )
    reference = _extractor(graph).eval()
    changed = _extractor(changed_padding).eval()
    _copy_parameters(reference, changed)
    observations = _batch()
    with torch.no_grad():
        expected = reference(observations)
        actual = changed(observations)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def test_neighbor_and_corresponding_route_permutation_is_invariant() -> None:
    extractor = _extractor(_graph()).eval()
    observations = _batch()
    permutation = torch.tensor([0, 4, 2, 5, 1, 3])
    permuted = {
        "trajectory": observations["trajectory"][:, permutation],
        "map": observations["map"]
        .reshape(2, 6, 2, 10, 5)[:, permutation]
        .reshape(2, 12, 10, 5),
    }
    with torch.no_grad():
        reference = extractor(observations)
        actual = extractor(permuted)
    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-5)


def _rotate_xy(values: torch.Tensor, angle: float) -> torch.Tensor:
    cosine = torch.cos(torch.tensor(angle, dtype=values.dtype))
    sine = torch.sin(torch.tensor(angle, dtype=values.dtype))
    return torch.stack(
        [
            cosine * values[..., 0] - sine * values[..., 1],
            sine * values[..., 0] + cosine * values[..., 1],
        ],
        dim=-1,
    )


def test_joint_global_rotation_of_vehicles_routes_and_topology_is_invariant() -> None:
    angle = 0.73
    graph = _graph()
    rotated_points = _rotate_xy(torch.as_tensor(graph.lane_points), angle).numpy()
    rotated_graph = PaddedTopologyGraph(
        lane_points=rotated_points,
        lane_attrs=graph.lane_attrs.copy(),
        node_mask=graph.node_mask.copy(),
        edge_index=graph.edge_index.copy(),
        edge_type=graph.edge_type.copy(),
        edge_mask=graph.edge_mask.copy(),
    )
    observations = _batch()
    rotated_observations = {key: value.clone() for key, value in observations.items()}
    rotated_observations["trajectory"][..., :2] = _rotate_xy(
        observations["trajectory"][..., :2], angle
    )
    rotated_observations["trajectory"][..., 2] += angle
    rotated_observations["trajectory"][..., 3:5] = _rotate_xy(
        observations["trajectory"][..., 3:5], angle
    )
    rotated_observations["map"][..., :2] = _rotate_xy(
        observations["map"][..., :2], angle
    )
    rotated_observations["map"][..., 2] += angle
    reference = _extractor(graph).eval()
    rotated = _extractor(rotated_graph).eval()
    _copy_parameters(reference, rotated)
    with torch.no_grad():
        expected = reference(observations)
        actual = rotated(rotated_observations)
    torch.testing.assert_close(expected, actual, rtol=2e-4, atol=2e-4)


def test_topology_vehicle_gcn_and_temporal_parameters_receive_gradients() -> None:
    extractor = _extractor(_graph()).train()
    loss = extractor(_batch()).square().mean()
    loss.backward()

    def gradient_sum(module: torch.nn.Module) -> float:
        return float(
            sum(
                parameter.grad.abs().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
        )

    assert extractor.topology_encoder is not None
    assert gradient_sum(extractor.topology_encoder) > 0.0
    assert gradient_sum(extractor.vehicle_layers) > 0.0
    assert gradient_sum(extractor.temporal_encoder) > 0.0


def test_graph_slt_losses_are_finite_and_actions_change_predictions() -> None:
    torch.manual_seed(3)
    objective = StructuredGraphRepresentationObjective(feature_dim=128, action_dim=2)
    online = StructuredLatent.from_tensor(torch.randn(4, 128))
    target = StructuredLatent.from_tensor(torch.randn(4, 128))
    first_action = torch.zeros(4, 2)
    second_action = torch.ones(4, 2)
    first_prediction = objective.predict(online, first_action).tensor
    second_prediction = objective.predict(online, second_action).tensor
    assert not torch.allclose(first_prediction, second_prediction)
    losses = objective(online, first_action, target)
    assert all(
        torch.isfinite(value).all()
        for value in (losses.total, losses.ego, losses.social, losses.route)
    )
    losses.total.backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in objective.parameters()
        if parameter.grad is not None
    ) > 0.0


class _GraphRawStepEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = _space()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        specification = PAPER_SCENARIOS["left_turn"]
        self.specification = SimpleNamespace(
            network_path=specification.network_path,
            coordinate_offset=specification.coordinate_offset,
        )
        self.raw_budget: int | None = None
        self.lifetime_raw_steps = 0

    @staticmethod
    def _observation() -> dict[str, np.ndarray]:
        return {
            key: value[0].numpy().astype(np.float32)
            for key, value in _batch(batch_size=1).items()
        }

    def set_lifetime_raw_step_budget(self, raw_steps: int | None) -> None:
        self.raw_budget = raw_steps
        self.lifetime_raw_steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        remaining = (
            3
            if self.raw_budget is None
            else max(0, self.raw_budget - self.lifetime_raw_steps)
        )
        executed = min(3, remaining)
        self.lifetime_raw_steps += executed
        return self._observation(), 0.0, False, False, {
            "raw_steps_executed": executed,
            "lifetime_raw_simulation_steps": self.lifetime_raw_steps,
        }


@pytest.mark.parametrize("algorithm", ["topo_scene", "topo_scene_balanced"])
def test_topo_scene_single_update_save_and_load(
    tmp_path: Any, algorithm: str
) -> None:
    assert source_action_repeat("topo_scene") == 3
    assert source_action_repeat("topo_scene_balanced") == 3
    assert source_action_repeat("temporal_graph") == 3
    env = _GraphRawStepEnv()
    model = make_model(
        algorithm,
        env,
        scenario="left_turn",
        learning_starts=16,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    assert isinstance(model.critic.features_extractor, TopoTemporalGraphExtractor)
    assert model.critic.features_extractor.normalize_slots is (
        algorithm == "topo_scene_balanced"
    )
    assert isinstance(model.representation, StructuredGraphRepresentationObjective)
    model.learn(
        total_timesteps=16,
        callback=RawStepControlCallback(raw_step_budget=16),
    )
    assert model._n_updates == 1
    diagnostics = model.training_diagnostics()
    for name in (
        "diagnostic/ego_latent_std",
        "diagnostic/social_latent_std",
        "diagnostic/route_latent_std",
        "diagnostic/slot_scale_ratio",
    ):
        assert name in diagnostics
        assert np.isfinite(diagnostics[name]["last"])
    observation, _ = env.reset(seed=4)
    with source_evaluation_augmentation(model):
        expected_action, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / f"{algorithm}_smoke"
    model.save(checkpoint)
    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    with source_evaluation_augmentation(restored):
        actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
