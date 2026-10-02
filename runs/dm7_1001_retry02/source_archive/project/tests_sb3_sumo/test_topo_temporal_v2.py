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
from algos.sb3_torch.incremental_topo_encoder import IncrementalTopoEncoder
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


def _incremental_extractor(
    *, use_slots: bool, use_incremental_slots: bool = False
) -> IncrementalTopoEncoder:
    return IncrementalTopoEncoder(
        _space(),
        topology_graph=_graph(),
        use_route=True,
        use_topology=True,
        use_slots=use_slots,
        use_incremental_slots=use_incremental_slots,
        random_augmentation=False,
    )


def _routeaware_extractor() -> IncrementalTopoEncoder:
    specification = PAPER_SCENARIOS["left_turn"]
    graph, graph_info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=specification.coordinate_offset,
        return_info=True,
    )
    spaces = dict(_space().spaces)
    spaces["route_reachability"] = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(int(graph.node_mask.shape[0]),),
        dtype=np.float32,
    )
    return IncrementalTopoEncoder(
        gym.spaces.Dict(spaces),
        topology_graph=graph,
        topology_lane_ids=tuple(graph_info.lane_ids),
        use_route=True,
        use_topology=True,
        use_slots=False,
        use_route_reachability=True,
        random_augmentation=False,
    )


def _capture_input(captured: dict[str, torch.Tensor], name: str):
    def hook(_module, inputs) -> None:
        captured[name] = inputs[0].detach().clone()

    return hook


def test_incremental_slot_head_keeps_the_topology_non_slot_path_identical() -> None:
    torch.manual_seed(19)
    topo = _incremental_extractor(use_slots=False).eval()
    topo_3slot = _incremental_extractor(
        use_slots=True, use_incremental_slots=True
    ).eval()
    topo_3slot.load_state_dict(topo.state_dict(), strict=True)

    topo_inputs: dict[str, torch.Tensor] = {}
    slot_inputs: dict[str, torch.Tensor] = {}
    handles = [
        topo.mlp_output.register_forward_pre_hook(_capture_input(topo_inputs, "all")),
        topo_3slot.ego_projection.register_forward_pre_hook(
            _capture_input(slot_inputs, "ego")
        ),
        topo_3slot.social_projection.register_forward_pre_hook(
            _capture_input(slot_inputs, "social")
        ),
        topo_3slot.route_projection.register_forward_pre_hook(
            _capture_input(slot_inputs, "route")
        ),
    ]
    observations = _batch()
    try:
        topo(observations)
        latent = topo_3slot.forward_tokens(observations)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(latent, StructuredLatent)
    assert latent.z_ego.shape[-1] == 32
    assert latent.z_social.shape[-1] == 64
    assert latent.z_route.shape[-1] == 32
    assert latent.tensor.shape == (3, 128)
    shared_context = torch.cat(
        [slot_inputs["ego"], slot_inputs["social"], slot_inputs["route"]], dim=-1
    )
    torch.testing.assert_close(topo_inputs["all"], shared_context, rtol=0, atol=0)


def test_route_reachability_mask_intersects_expands_and_bypasses_safely() -> None:
    encoder = _routeaware_extractor()
    capacity = int(encoder.topology_node_mask.numel())
    valid_count = int(encoder.topology_node_mask.sum())
    assert valid_count >= 3

    query = torch.zeros(3, capacity, dtype=torch.bool)
    query[0, 0] = True
    query[0, 1] = True
    query[1, 0] = True
    query[2, 2] = True

    labels = torch.full((3, capacity), -1.0)
    labels[:, :valid_count] = 0.0
    # Sample 0 has a legal node inside its geometric query and one outside it.
    labels[0, 0] = 1.0
    labels[0, 2] = 1.0
    # Sample 1 has legal continuation nodes, but none in its geometric query.
    labels[1, 1] = 1.0

    selected, state = encoder._route_reachability_goal_mask(query, labels)
    assert selected[0].nonzero().flatten().tolist() == [0]
    assert selected[1].nonzero().flatten().tolist() == [1]
    assert selected[2].nonzero().flatten().tolist() == [2]
    assert state["expanded"].tolist() == [False, True, False]
    assert state["bypassed"].tolist() == [False, False, True]
    assert not selected[:, valid_count:].any()


def test_route_reachability_forward_is_finite_and_trains_goal_branch() -> None:
    encoder = _routeaware_extractor().train()
    encoder.configure_diagnostics(True, sample_every=1, source="train_replay")
    valid_count = int(encoder.topology_node_mask.sum())
    capacity = int(encoder.topology_node_mask.numel())
    observations = _batch(batch_size=3)
    labels = torch.full((3, capacity), -1.0)
    labels[:, :valid_count] = 0.0
    labels[0, 0] = 1.0
    # Sample 1 deliberately has no known legal continuation: its topology-goal
    # residual must be bypassed instead of using an all-masked softmax.
    labels[2, valid_count - 1] = 1.0
    observations["route_reachability"] = labels

    captured: dict[str, torch.Tensor] = {}

    def capture_goal_mask(_module, inputs) -> None:
        captured["mask"] = inputs[2].detach().clone()

    def capture_route_base(_module, _inputs, output) -> None:
        captured["route_base"] = output[0].detach().clone()

    def capture_mlp_input(_module, inputs) -> None:
        captured["mlp_input"] = inputs[0].detach().clone()

    handle = encoder.goal_topology_attention.register_forward_pre_hook(
        capture_goal_mask
    )
    route_base_handle = encoder.goal_attention.register_forward_hook(
        capture_route_base
    )
    mlp_input_handle = encoder.mlp_output.register_forward_pre_hook(
        capture_mlp_input
    )
    try:
        output = encoder(observations)
    finally:
        handle.remove()
        route_base_handle.remove()
        mlp_input_handle.remove()

    assert output.shape == (3, 128)
    assert torch.isfinite(output).all()
    assert captured["mask"].shape == (3, capacity)
    assert captured["mask"][1].any()
    assert not captured["mask"][:, valid_count:].any()
    # The no-legal sample bypasses only the topology-goal residual; its route
    # component is exactly the existing route-base readout.
    torch.testing.assert_close(
        captured["mlp_input"][1, -128:],
        captured["route_base"][1, 0],
        rtol=0,
        atol=0,
    )
    diagnostics = encoder.diagnostic_values()
    assert diagnostics["diagnostic_sample_index"].item() == 1
    assert diagnostics["route_reachability_goal_bypass_fraction"].item() == pytest.approx(1 / 3)
    assert diagnostics["route_reachability_goal_active_sample_count"].item() == 2
    assert diagnostics["route_reachability_goal_legal_attention_valid_count"].item() == 2
    assert all(torch.isfinite(value).all() for value in diagnostics.values())

    output.square().mean().backward()
    for prefix in ("goal_topology_attention.", "goal_residual_scale"):
        gradients = [
            parameter.grad
            for name, parameter in encoder.named_parameters()
            if name == prefix or name.startswith(prefix)
        ]
        assert gradients, prefix
        assert any(gradient is not None for gradient in gradients), prefix
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        ), prefix


def test_disabled_route_reachability_ignores_extra_observation_key() -> None:
    encoder = _incremental_extractor(use_slots=False).eval()
    observations = _batch(batch_size=2)
    with torch.no_grad():
        baseline = encoder(observations)
        extended = encoder(
            {
                **observations,
                "route_reachability": torch.ones(2, 16),
            }
        )
    torch.testing.assert_close(extended, baseline, rtol=0, atol=0)


def test_route_three_slot_method_emits_structured_latent_and_gradients() -> None:
    encoder = IncrementalTopoEncoder(
        _space(),
        topology_graph=_graph(),
        use_route=True,
        use_topology=False,
        use_slots=True,
        use_incremental_slots=True,
        random_augmentation=False,
    ).train()
    output = encoder.forward_tokens(_batch(batch_size=2))
    assert isinstance(output, StructuredLatent)
    assert output.z_ego.shape == (2, 32)
    assert output.z_social.shape == (2, 64)
    assert output.z_route.shape == (2, 32)
    output.tensor.square().mean().backward()
    for prefix in ("ego_projection.", "social_projection.", "route_projection."):
        gradients = [
            parameter.grad
            for name, parameter in encoder.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients and any(gradient is not None for gradient in gradients)
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        )


def test_incremental_encoder_full_default_still_delegates_to_parent_v2() -> None:
    torch.manual_seed(23)
    parent = _extractor("soft").eval()
    full_default = _incremental_extractor(use_slots=True).eval()
    full_default.load_state_dict(parent.state_dict(), strict=False)
    parent_output = parent(_batch())
    full_output = full_default(_batch())
    torch.testing.assert_close(full_output, parent_output, rtol=0, atol=0)


def test_incremental_slot_diagnostics_are_sampled_and_batch_one_is_not_variance_evidence() -> None:
    encoder = _incremental_extractor(
        use_slots=True, use_incremental_slots=True
    ).eval()
    encoder.configure_diagnostics(True, sample_every=2, source="train_replay")
    observations = _batch()

    rng_state_before = torch.random.get_rng_state().clone()
    encoder(observations)
    assert torch.equal(rng_state_before, torch.random.get_rng_state())
    first = encoder.diagnostic_values()
    assert first["diagnostic_sample_index"].item() == 1
    assert first["diagnostic_sample_forward_index"].item() == 1
    assert first["diagnostic_sample_age_forwards"].item() == 0

    encoder(observations)
    sampled = encoder.diagnostic_values()
    assert sampled["diagnostic_sample_index"].item() == 1
    assert sampled["diagnostic_sample_forward_index"].item() == 1
    assert sampled["diagnostic_sample_source_code"].item() == 1
    assert sampled["diagnostic_sample_batch_size"].item() == 3
    assert sampled["diagnostic_sample_grad_enabled"].item() == 1
    assert sampled["diagnostic_sample_age_forwards"].item() == 1
    assert sampled["diagnostic_valid_query_count"].item() > 0
    assert sampled["relation_pair_valid_count"].item() > 0
    assert sampled["ego_slot_dim"].item() == 32
    assert sampled["social_slot_dim"].item() == 64
    assert sampled["route_slot_dim"].item() == 32
    assert sampled["slot_sample_energy_correlation_valid"].item() == 0
    assert "slot_sample_energy_ego_social_corr" not in sampled
    assert all(torch.isfinite(value).all() for value in sampled.values())

    encoder.configure_diagnostics(True, sample_every=1, source="eval_policy")
    with torch.no_grad():
        encoder({key: value[:1] for key, value in observations.items()})
    single = encoder.diagnostic_values()
    assert single["diagnostic_sample_index"].item() == 2
    assert single["diagnostic_sample_source_code"].item() == 2
    assert single["diagnostic_sample_batch_size"].item() == 1
    assert single["diagnostic_sample_grad_enabled"].item() == 0
    assert single["diagnostic_batch_variance_valid"].item() == 0
    assert single["slot_batch_variance_valid"].item() == 0
    assert single["slot_sample_energy_correlation_valid"].item() == 0
    assert all(torch.isfinite(value).all() for value in single.values())


def test_slot_sample_energy_correlation_reports_validity_separately() -> None:
    correlated, correlated_valid = IncrementalTopoEncoder._safe_correlation(
        torch.tensor([1.0, 2.0, 4.0]),
        torch.tensor([2.0, 4.0, 8.0]),
    )
    constant, constant_valid = IncrementalTopoEncoder._safe_correlation(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([2.0, 3.0, 4.0]),
    )
    assert correlated_valid.item() == 1
    torch.testing.assert_close(correlated, torch.tensor(1.0))
    assert constant_valid.item() == 0


def test_incremental_three_slot_path_preserves_topology_and_projection_gradients() -> None:
    encoder = _incremental_extractor(
        use_slots=True, use_incremental_slots=True
    ).train()
    output = encoder.forward_tokens(_batch())
    assert isinstance(output, StructuredLatent)
    output.tensor.square().mean().backward()

    for prefix in (
        "topology_encoder.",
        "topology_attention.",
        "goal_topology_attention.",
        "ego_projection.",
        "social_projection.",
        "route_projection.",
    ):
        gradients = [
            parameter.grad
            for name, parameter in encoder.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients, prefix
        assert any(gradient is not None for gradient in gradients), prefix
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        ), prefix
        assert any(
            gradient is not None and gradient.abs().sum() > 0
            for gradient in gradients
        ), prefix


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
