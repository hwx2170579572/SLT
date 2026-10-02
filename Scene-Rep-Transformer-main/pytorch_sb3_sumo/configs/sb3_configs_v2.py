"""Isolated model registry for the topology-temporal v2 experiment family."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import PPO, SAC

from algos.sb3_torch import (
    DictNStepReplayBuffer,
    SceneRepSACPolicy,
    SceneRepresentationSAC,
    TopoTemporalGraphExtractor,
)
from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs import make_model as make_model_v1
from configs.sb3_configs import source_action_repeat as source_action_repeat_v1
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V2_ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_scene_v1_control",
    "topo_scene_balanced_v1_control",
    "temporal_graph_balanced_v1",
    "topo_v2_merge",
    "topo_v2_query",
    "topo_v2_gated",
    "topo_v2_soft",
)
V2_IMPLEMENTATION_IDS = {
    "temporal_graph_v1_control": "temporal_vehicle_graph_slt_pytorch_v1",
    "topo_scene_v1_control": "topo_temporal_graph_slt_pytorch_v1",
    "topo_scene_balanced_v1_control": "topo_temporal_graph_slt_balanced_slots_v1",
    "temporal_graph_balanced_v1": "temporal_vehicle_graph_hard_balanced_slots_v1",
    "topo_v2_merge": "merge_aware_topology_temporal_graph_v2",
    "topo_v2_query": "route_direction_sparse_topology_temporal_graph_v2",
    "topo_v2_gated": "gated_split_pool_topology_temporal_graph_v2",
    "topo_v2_soft": "soft_balanced_gated_topology_temporal_graph_v2",
}
_V1_CONTROL_ALIASES = {
    "temporal_graph_v1_control": "temporal_graph",
    "topo_scene_v1_control": "topo_scene",
    "topo_scene_balanced_v1_control": "topo_scene_balanced",
}
_VARIANT_BY_ALGORITHM = {
    "topo_v2_merge": "merge",
    "topo_v2_query": "query",
    "topo_v2_gated": "gated",
    "topo_v2_soft": "soft",
}


def source_action_repeat_v2(algo: str, requested: int | None = None) -> int:
    if algo not in V2_ALGORITHMS:
        return source_action_repeat_v1(algo, requested)
    action_repeat = 3 if requested is None else int(requested)
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    return action_repeat


def _make_structured_sac(
    *,
    model_class: type[SceneRepresentationSAC],
    env: Any,
    feature_extractor_class: type[torch.nn.Module],
    feature_extractor_kwargs: dict[str, Any],
    learning_rate: float,
    batch_size: int,
    discount: float,
    learning_starts: int,
    buffer_size: int,
    action_repeat: int,
    seed: int,
    device: str,
    tensorboard_log: str | None,
    verbose: int,
    model_extra_kwargs: dict[str, Any] | None = None,
) -> SceneRepresentationSAC:
    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
        "net_arch": {"pi": [128, 32], "qf": [128, 32]},
        "optimizer_class": torch.optim.NAdam,
        "optimizer_kwargs": {"eps": 1e-7},
        "n_critics": 2,
        "action_embedding_dim": 64,
    }
    return model_class(
        SceneRepSACPolicy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=action_repeat,
        n_steps=4,
        replay_buffer_class=DictNStepReplayBuffer,
        replay_buffer_kwargs={
            "n_steps": 4,
            "gamma": discount,
            "duplicate_episode_end_transition": True,
            "source_action_repeat": action_repeat,
        },
        ent_coef="auto_0.2",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        structured_representation=True,
        representation_coef=1.0,
        representation_learning_rate=learning_rate,
        representation_heads=2,
        representation_separate_target_projector=False,
        representation_online_target_encoder=(
            getattr(env.unwrapped, "scenario", None) != "cross"
        ),
        action_embedding_dim=64,
        max_grad_norm=5.0,
        raw_learning_starts=learning_starts,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **(model_extra_kwargs or {}),
    )


def make_model_v2(
    algo: str,
    env: Any,
    *,
    scenario: str,
    learning_rate: float | None = None,
    batch_size: int = 32,
    discount: float = 0.99,
    learning_starts: int = 5000,
    buffer_size: int = 20_000,
    action_repeat: int | None = None,
    seed: int = 0,
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
    slot_balance_coef: float | None = None,
    route_sigma: float = 20.0,
    topology_top_k: int = 8,
    reverse_heading_cosine_min: float = 0.0,
    route_bias_init: float = 1.0,
    heading_bias_init: float = 1.0,
    topology_layerscale_init: float = 1e-3,
    goal_layerscale_init: float = 1e-3,
) -> SAC | PPO:
    """Build either an untouched v1 model or one isolated v2 variant."""

    if algo not in V2_ALGORITHMS:
        if slot_balance_coef not in (None, 0.0):
            raise ValueError("slot_balance_coef is only valid for topo_v2_soft")
        return make_model_v1(
            algo,
            env,
            scenario=scenario,
            learning_rate=learning_rate,
            batch_size=batch_size,
            discount=discount,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            action_repeat=action_repeat,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )

    action_repeat = source_action_repeat_v2(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    random_augmentation = scenario != "cross"
    carla_contract = scenario == "carla"

    if algo in _V1_CONTROL_ALIASES:
        if slot_balance_coef not in (None, 0.0):
            raise ValueError("v1 control aliases cannot use SoftBalancedSlots")
        core_algorithm = _V1_CONTROL_ALIASES[algo]
        model = make_model_v1(
            core_algorithm,
            env,
            scenario=scenario,
            learning_rate=learning_rate,
            batch_size=batch_size,
            discount=discount,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            action_repeat=action_repeat,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
        model.v2_method_metadata = {
            "implementation_id": V2_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "core_algorithm": core_algorithm,
            "v1_implementation_unchanged": True,
            "three_way_traffic_protocol_only": True,
        }
        return model

    if algo == "temporal_graph_balanced_v1":
        if slot_balance_coef not in (None, 0.0):
            raise ValueError(
                "The 2x2 hard-normalisation ablation cannot use SoftBalancedSlots"
            )
        feature_kwargs = {
            "topology_graph": None,
            "features_dim": 128,
            "hidden_dim": 128,
            "num_heads": 2,
            "random_augmentation": random_augmentation,
            "carla_contract": carla_contract,
            "use_topology": False,
            "normalize_slots": True,
        }
        model = _make_structured_sac(
            model_class=SceneRepresentationSAC,
            env=env,
            feature_extractor_class=TopoTemporalGraphExtractor,
            feature_extractor_kwargs=feature_kwargs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            discount=discount,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            action_repeat=action_repeat,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
        model.topology_graph_info = None
        model.graph_ablation = "temporal_vehicle_graph_with_hard_balanced_slots"
        model.v2_method_metadata = {
            "implementation_id": V2_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "topology": False,
            "hard_slot_normalization": True,
            "soft_slot_balance_coef": 0.0,
        }
        return model

    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError(
            "v2 topology methods require env.unwrapped.specification.network_path"
        )
    topology_graph, topology_info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=tuple(
            getattr(specification, "coordinate_offset", (0.0, 0.0))
        ),
        return_info=True,
    )
    variant = _VARIANT_BY_ALGORITHM[algo]
    if algo == "topo_v2_soft":
        effective_balance_coef = (
            1e-3 if slot_balance_coef is None else float(slot_balance_coef)
        )
    else:
        if slot_balance_coef not in (None, 0.0):
            raise ValueError("Only topo_v2_soft may use a positive slot balance loss")
        effective_balance_coef = 0.0

    feature_kwargs = {
        "topology_graph": topology_graph,
        "variant": variant,
        "features_dim": 128,
        "hidden_dim": 128,
        "num_heads": 2,
        "random_augmentation": random_augmentation,
        "carla_contract": carla_contract,
        "route_sigma": route_sigma,
        "topology_top_k": topology_top_k,
        "reverse_heading_cosine_min": reverse_heading_cosine_min,
        "route_bias_init": route_bias_init,
        "heading_bias_init": heading_bias_init,
        "topology_layerscale_init": topology_layerscale_init,
        "goal_layerscale_init": goal_layerscale_init,
    }
    model = _make_structured_sac(
        model_class=SceneRepresentationSACV2,
        env=env,
        feature_extractor_class=TopoTemporalGraphExtractorV2,
        feature_extractor_kwargs=feature_kwargs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        discount=discount,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        action_repeat=action_repeat,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        model_extra_kwargs={"slot_balance_coef": effective_balance_coef},
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = f"topology_temporal_v2_{variant}"
    model.v2_method_metadata = {
        "implementation_id": V2_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "variant": variant,
        "topology": True,
        "merge_relation": True,
        "route_direction_query": variant in ("query", "gated", "soft"),
        "near_zero_residual": variant in ("gated", "soft"),
        "split_goal_pool": variant in ("gated", "soft"),
        "hard_slot_normalization": False,
        "soft_slot_balance_coef": effective_balance_coef,
        "route_sigma": float(route_sigma),
        "topology_top_k": int(topology_top_k),
        "reverse_heading_cosine_min": float(reverse_heading_cosine_min),
        "route_bias_init": float(route_bias_init),
        "heading_bias_init": float(heading_bias_init),
        "topology_layerscale_init": float(topology_layerscale_init),
        "goal_layerscale_init": float(goal_layerscale_init),
    }
    return model


__all__ = [
    "V2_ALGORITHMS",
    "V2_IMPLEMENTATION_IDS",
    "make_model_v2",
    "source_action_repeat_v2",
]
