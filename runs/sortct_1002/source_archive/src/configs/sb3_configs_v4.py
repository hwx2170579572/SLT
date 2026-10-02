"""Isolated registry for the decision-aligned v4.1 method family."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedSACPolicy
from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.sac_v4 import DecisionAlignedHybridSACV4
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs_v2 import make_model_v2, source_action_repeat_v2
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V4_ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_v4_continuous_ablation",
    "topo_v4_da_hybrid",
)
V4_IMPLEMENTATION_IDS = {
    "temporal_graph_v1_control": "temporal_vehicle_graph_slt_pytorch_v1_v4_protocol_control",
    "topo_v4_continuous_ablation": "topo_v2_soft_continuous_action_v4_ablation",
    "topo_v4_da_hybrid": "full_decision_aligned_hybrid_action_v4_1",
}


def source_action_repeat_v4(algo: str, requested: int | None = None) -> int:
    if algo not in V4_ALGORITHMS:
        raise ValueError(f"v4 algorithm must be one of {V4_ALGORITHMS}")
    source_algorithm = (
        "topo_v2_soft"
        if algo == "topo_v4_continuous_ablation"
        else "temporal_graph_v1_control"
    )
    return source_action_repeat_v2(source_algorithm, requested)


def make_model_v4(
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
    slot_balance_coef: float = 0.01,
    route_sigma: float = 20.0,
    topology_top_k: int = 8,
    reverse_heading_cosine_min: float = 0.0,
    route_bias_init: float = 1.0,
    heading_bias_init: float = 1.0,
    topology_layerscale_init: float = 1e-3,
    goal_layerscale_init: float = 1e-3,
) -> SAC:
    if algo not in V4_ALGORITHMS:
        raise ValueError(f"v4 algorithm must be one of {V4_ALGORITHMS}")
    action_repeat = source_action_repeat_v4(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    if algo == "temporal_graph_v1_control":
        model = make_model_v2(
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
        model.v4_method_metadata = {
            "implementation_id": V4_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "comparator_implementation_unchanged": True,
            "v4_observation_addition_ignored": "lane_action_mask",
        }
        return model

    if algo == "topo_v4_continuous_ablation":
        if abs(float(slot_balance_coef) - 0.01) > 1e-12:
            raise ValueError(
                "the continuous-action ablation freezes SoftBalancedSlots at 0.01"
            )
        model = make_model_v2(
            "topo_v2_soft",
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
            slot_balance_coef=float(slot_balance_coef),
        )
        model.v4_method_metadata = {
            "implementation_id": V4_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "ablation_of": "topo_v4_da_hybrid",
            "single_removed_mechanism": "hybrid_action_interface",
            "encoder": "topo_v2_soft_frozen_design",
            "soft_slot_balance_coef": float(slot_balance_coef),
            "lane_action_mask_observation": False,
            "lane_distribution": "continuous_lateral_thresholded_by_environment",
            "critic_action": "continuous_speed_plus_continuous_lateral",
            "actor_encoder_detach": True,
            "reward_unchanged": True,
        }
        return model

    if abs(float(slot_balance_coef) - 0.01) > 1e-12:
        raise ValueError("v4.1 freezes the inherited SoftBalancedSlots coefficient at 0.01")
    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("v4 topology method requires specification.network_path")
    topology_graph, topology_info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=tuple(
            getattr(specification, "coordinate_offset", (0.0, 0.0))
        ),
        return_info=True,
    )
    feature_kwargs = {
        "topology_graph": topology_graph,
        "variant": "soft",
        "features_dim": 128,
        "hidden_dim": 128,
        "num_heads": 2,
        "random_augmentation": scenario != "cross",
        "carla_contract": scenario == "carla",
        "route_sigma": route_sigma,
        "topology_top_k": topology_top_k,
        "reverse_heading_cosine_min": reverse_heading_cosine_min,
        "route_bias_init": route_bias_init,
        "heading_bias_init": heading_bias_init,
        "topology_layerscale_init": topology_layerscale_init,
        "goal_layerscale_init": goal_layerscale_init,
    }
    policy_kwargs = {
        "features_extractor_class": TopoTemporalGraphExtractorV2,
        "features_extractor_kwargs": feature_kwargs,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
        "net_arch": {"pi": [128, 32], "qf": [128, 32]},
        "optimizer_class": torch.optim.NAdam,
        "optimizer_kwargs": {"eps": 1e-7},
        "n_critics": 2,
        "action_embedding_dim": 64,
    }
    model = DecisionAlignedHybridSACV4(
        DecisionAlignedSACPolicy,
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
        target_entropy=-2.0,
        policy_kwargs=policy_kwargs,
        structured_representation=True,
        representation_coef=1.0,
        representation_learning_rate=learning_rate,
        representation_heads=2,
        representation_separate_target_projector=False,
        representation_online_target_encoder=(scenario != "cross"),
        action_embedding_dim=64,
        max_grad_norm=5.0,
        raw_learning_starts=learning_starts,
        slot_balance_coef=float(slot_balance_coef),
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = "topology_v2_soft_plus_decision_aligned_hybrid_action"
    model.v4_method_metadata = {
        "implementation_id": V4_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "single_change": "hybrid_action_interface",
        "encoder": "topo_v2_soft_frozen_design",
        "soft_slot_balance_coef": float(slot_balance_coef),
        "route_intent_observation": False,
        "lane_action_mask_observation": True,
        "lane_distribution": "masked_categorical_exact_enumeration",
        "speed_distribution": "lane_conditioned_squashed_gaussian",
        "critic_action": "normalised_speed_plus_lane_one_hot",
        "target_entropy": -2.0,
        "actor_encoder_detach": True,
        "reward_unchanged": True,
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
    "V4_ALGORITHMS",
    "V4_IMPLEMENTATION_IDS",
    "make_model_v4",
    "source_action_repeat_v4",
]
