from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos.sb3_torch import RawStepControlCallback
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.graph_representation import StructuredGraphRepresentationObjective
from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2
from algos.sb3_torch.topo_temporal_features import (
    StructuredLatent,
    TopoTemporalGraphExtractor,
)
from algos.sb3_torch.topo_temporal_features_v2 import (
    TopoTemporalGraphExtractorV2,
    soft_slot_balance_loss,
)
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph_v2 import build_topology_graph_v2
from configs.sb3_configs_v2 import (
    V2_ALGORITHMS,
    make_model_v2,
    source_action_repeat_v2,
)


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


def _batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
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


def _graph(scenario: str = "left_turn"):
    specification = PAPER_SCENARIOS[scenario]
    return build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
    )


def _extractor(variant: str, *, gate: float = 1e-3):
    return TopoTemporalGraphExtractorV2(
        _space(),
        topology_graph=_graph(),
        variant=variant,
        random_augmentation=False,
        topology_layerscale_init=gate,
        goal_layerscale_init=gate,
    )


@pytest.mark.parametrize("variant", ["merge", "query", "gated", "soft"])
def test_v2_variants_are_finite_and_emit_required_diagnostics(variant: str) -> None:
    extractor = _extractor(variant).eval()
    output = extractor(_batch())
    assert output.shape == (3, 128)
    assert torch.isfinite(output).all()
    diagnostics = extractor.diagnostic_values()
    required = {
        "topology_attention_entropy",
        "topology_effective_lanes",
        "route_compatible_attention_mass",
        "merge_attention_mass",
        "merge_pair_score",
        "reverse_lane_mask_rate",
        "topology_fallback_rate",
        "topology_residual_scale",
        "goal_residual_scale",
        "soft_slot_balance_loss",
        "ego_slot_rms",
        "social_slot_rms",
        "route_slot_rms",
        "slot_rms_ratio",
    }
    assert required.issubset(diagnostics)
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert extractor.last_topology_attention is not None


def test_query_variant_keeps_at_most_top_k_compatible_lanes() -> None:
    extractor = _extractor("query").eval()
    extractor(_batch())
    assert extractor.last_topology_key_mask is not None
    assert int(extractor.last_topology_key_mask.sum(dim=-1).max()) <= 8
    attention = extractor.last_topology_attention
    assert attention is not None
    rejected_mass = attention.masked_select(~extractor.last_topology_key_mask).abs().max()
    assert float(rejected_mass) == pytest.approx(0.0, abs=1e-8)


def _copy_backbone(source: TopoTemporalGraphExtractor, target: TopoTemporalGraphExtractorV2) -> None:
    names = (
        "state_encoder",
        "map_encoder",
        "route_attention",
        "route_norm",
        "topology_norm",
        "vehicle_layers",
        "temporal_encoder",
        "social_attention",
        "goal_attention",
        "ego_projection",
        "social_projection",
        "route_projection",
    )
    for name in names:
        target.get_submodule(name).load_state_dict(source.get_submodule(name).state_dict())


def test_zero_residual_scales_exactly_recover_temporal_graph_backbone() -> None:
    torch.manual_seed(11)
    baseline = TopoTemporalGraphExtractor(
        _space(),
        topology_graph=None,
        random_augmentation=False,
        use_topology=False,
        normalize_slots=False,
    ).eval()
    candidate = _extractor("gated", gate=0.0).eval()
    _copy_backbone(baseline, candidate)
    observations = _batch()
    with torch.no_grad():
        expected = baseline(observations)
        actual = candidate(observations)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_small_positive_gate_propagates_gradient_into_topology_branch() -> None:
    extractor = _extractor("gated", gate=1e-3).train()
    loss = extractor(_batch()).square().mean()
    loss.backward()
    gradients = [
        parameter.grad.abs().sum()
        for parameter in extractor.topology_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert float(torch.stack(gradients).sum()) > 0.0
    assert extractor.topology_residual_scale.grad is not None
    assert float(extractor.topology_residual_scale.grad.abs()) > 0.0


def test_soft_balance_uses_batch_rms_without_per_sample_normalization() -> None:
    equal = StructuredLatent(
        z_ego=torch.ones(4, 32),
        z_social=torch.ones(4, 64),
        z_route=torch.ones(4, 32),
    )
    equal_loss, equal_rms = soft_slot_balance_loss(equal)
    assert float(equal_loss) == pytest.approx(0.0, abs=1e-10)
    torch.testing.assert_close(equal_rms, torch.ones(3), atol=2e-6, rtol=0.0)

    unequal = StructuredLatent(
        z_ego=torch.ones(4, 32),
        z_social=torch.full((4, 64), 2.0),
        z_route=torch.full((4, 32), 0.25),
    )
    unequal_loss, unequal_rms = soft_slot_balance_loss(unequal)
    assert float(unequal_loss) > 0.0
    assert unequal_rms[1] > unequal_rms[0] > unequal_rms[2]


def test_route_query_is_invariant_to_joint_global_rotation() -> None:
    extractor = _extractor("query").eval()
    observations = _batch(batch_size=2)
    angle = 0.73
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))

    rotated = {key: value.clone() for key, value in observations.items()}
    for key in ("trajectory", "map"):
        x = observations[key][..., 0]
        y = observations[key][..., 1]
        rotated[key][..., 0] = cosine * x - sine * y
        rotated[key][..., 1] = sine * x + cosine * y
    rotated["trajectory"][..., 2] += angle
    vx = observations["trajectory"][..., 3]
    vy = observations["trajectory"][..., 4]
    rotated["trajectory"][..., 3] = cosine * vx - sine * vy
    rotated["trajectory"][..., 4] = sine * vx + cosine * vy
    rotated["map"][..., 2] += angle

    # Rotate the static graph in the same world frame by constructing a copy.
    graph = _graph()
    graph_points = graph.lane_points.copy()
    x = graph_points[..., 0].copy()
    y = graph_points[..., 1].copy()
    graph_points[..., 0] = cosine * x - sine * y
    graph_points[..., 1] = sine * x + cosine * y
    rotated_graph = type(graph)(
        lane_points=graph_points,
        lane_attrs=graph.lane_attrs.copy(),
        node_mask=graph.node_mask.copy(),
        edge_index=graph.edge_index.copy(),
        edge_type=graph.edge_type.copy(),
        edge_mask=graph.edge_mask.copy(),
    )
    rotated_extractor = TopoTemporalGraphExtractorV2(
        _space(),
        topology_graph=rotated_graph,
        variant="query",
        random_augmentation=False,
    ).eval()
    rotated_extractor.load_state_dict(extractor.state_dict())
    rotated_extractor.topology_lane_points.copy_(torch.as_tensor(graph_points))
    with torch.no_grad():
        expected = extractor(observations)
        actual = rotated_extractor(rotated)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


class _V2RawStepEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
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


@pytest.mark.parametrize("algorithm", V2_ALGORITHMS)
def test_v2_registry_constructs_distinct_ablation(algorithm: str) -> None:
    env = _V2RawStepEnv()
    model = make_model_v2(
        algorithm,
        env,
        scenario="left_turn",
        learning_starts=16,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
        slot_balance_coef=0.01 if algorithm == "topo_v2_soft" else 0.0,
    )
    assert source_action_repeat_v2(algorithm) == 3
    assert isinstance(model.representation, StructuredGraphRepresentationObjective)
    metadata = model.v2_method_metadata
    assert metadata["algorithm"] == algorithm
    if algorithm in (
        "temporal_graph_v1_control",
        "topo_scene_v1_control",
        "topo_scene_balanced_v1_control",
    ):
        assert metadata["v1_implementation_unchanged"] is True
        assert isinstance(model.critic.features_extractor, TopoTemporalGraphExtractor)
    elif algorithm == "temporal_graph_balanced_v1":
        assert isinstance(model.critic.features_extractor, TopoTemporalGraphExtractor)
        assert model.critic.features_extractor.normalize_slots is True
    else:
        assert isinstance(model, SceneRepresentationSACV2)
        assert isinstance(model.critic.features_extractor, TopoTemporalGraphExtractorV2)
        assert model.topology_graph_info["relation_counts"]["merge"] > 0


def test_soft_v2_single_update_diagnostics_and_save_load(tmp_path: Any) -> None:
    env = _V2RawStepEnv()
    model = make_model_v2(
        "topo_v2_soft",
        env,
        scenario="left_turn",
        learning_starts=16,
        buffer_size=64,
        batch_size=2,
        device="cpu",
        verbose=0,
        slot_balance_coef=0.01,
    )
    assert isinstance(model, SceneRepresentationSACV2)
    model.learn(
        total_timesteps=16,
        callback=RawStepControlCallback(raw_step_budget=16),
    )
    assert model._n_updates == 1
    diagnostics = model.training_diagnostics()
    for name in (
        "train/soft_slot_balance_loss",
        "train/weighted_soft_slot_balance_loss",
        "diagnostic/topology_effective_lanes",
        "diagnostic/route_compatible_attention_mass",
        "diagnostic/merge_attention_mass",
        "diagnostic/topology_residual_scale",
        "diagnostic/ego_slot_rms",
        "diagnostic/social_slot_rms",
        "diagnostic/route_slot_rms",
    ):
        assert name in diagnostics
        assert np.isfinite(diagnostics[name]["last"])

    observation, _ = env.reset(seed=4)
    with source_evaluation_augmentation(model):
        expected_action, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / "topo_v2_soft_smoke"
    model.save(checkpoint)
    restored = SceneRepresentationSACV2.load(checkpoint, env=env, device="cpu")
    with source_evaluation_augmentation(restored):
        actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
