"""Registry for the v4.5 confident-actor fusion decoder iteration."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4_5 import ConfidentActorFusionSACPolicyV45
from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs_v4 import make_model_v4, source_action_repeat_v4
from envs.sumo.topology_graph import MAX_TOPO_EDGES, MAX_TOPO_NODES
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V45_ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_v4_5_confident_actor_fusion",
)
V45_IMPLEMENTATION_IDS = {
    "temporal_graph_v1_control": (
        "temporal_vehicle_graph_slt_pytorch_v1_v4_5_selector_control"
    ),
    "topo_v4_5_confident_actor_fusion": (
        "full_decision_aligned_hybrid_factorized_entropy_confident_actor_"
        "fusion_train_only_selector_v4_5"
    ),
}


def source_action_repeat_v4_5(algo: str, requested: int | None = None) -> int:
    if algo not in V45_ALGORITHMS:
        raise ValueError(f"v4.5 algorithm must be one of {V45_ALGORITHMS}")
    parent = (
        "temporal_graph_v1_control"
        if algo == "temporal_graph_v1_control"
        else "topo_v4_da_hybrid"
    )
    return source_action_repeat_v4(parent, requested)


def make_model_v4_5(
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
    lane_entropy_scale: float = 0.0,
    target_entropy: float = -1.0,
) -> SAC:
    if algo not in V45_ALGORITHMS:
        raise ValueError(f"v4.5 algorithm must be one of {V45_ALGORITHMS}")
    action_repeat = source_action_repeat_v4_5(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    if algo == "temporal_graph_v1_control":
        model = make_model_v4(
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
            "implementation_id": V45_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "comparator_implementation_unchanged": True,
            "v4_5_observation_addition_ignored": "lane_action_mask",
            "train_only_checkpoint_selector": True,
            "checkpoint_selector_changed_from_v4_4": False,
            "checkpoint_calibration_partition": "train",
            "checkpoint_calibration_episodes": 12,
            "validation_used_for_checkpoint_selection": False,
            "formal_test_used_for_checkpoint_selection": False,
        }
        return model

    if abs(float(slot_balance_coef) - 0.01) > 1e-12:
        raise ValueError("v4.5 freezes SoftBalancedSlots at 0.01")
    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("v4.5 topology method requires specification.network_path")
    # Static lane-graph padding capacity is declared per scenario; defaults
    # keep every pre-existing scenario at the original 64 / 256 capacity.
    max_nodes = int(getattr(specification, "topology_max_nodes", MAX_TOPO_NODES))
    max_edges = int(getattr(specification, "topology_max_edges", MAX_TOPO_EDGES))
    topology_graph, topology_info = build_topology_graph_v2(
        specification.network_path,
        max_nodes=max_nodes,
        max_edges=max_edges,
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
    model = ConfidentActorFusionSACV45(
        ConfidentActorFusionSACPolicyV45,
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
        target_entropy=target_entropy,
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
        lane_entropy_scale=lane_entropy_scale,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "confident_actor_fusion_decoder"
    )
    model.v4_method_metadata = {
        "implementation_id": V45_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_implementation_id": (
            "full_decision_aligned_hybrid_factorized_entropy_target_critic_"
            "train_only_selector_v4_4"
        ),
        "single_change": "actor_confident_non_keep_else_target_critic",
        "encoder": "topo_v2_soft_frozen_design",
        "soft_slot_balance_coef": float(slot_balance_coef),
        "route_intent_observation": False,
        "lane_action_mask_observation": True,
        "lane_distribution": (
            "masked_categorical_training_confident_actor_non_keep_else_"
            "target_critic_deterministic_deployment"
        ),
        "deterministic_lane_decoder": (
            "actor_confident_non_keep_else_target_critic"
        ),
        "actor_non_keep_confidence_threshold": 0.90,
        "actor_override_requires_non_keep": True,
        "actor_override_requires_action_mask_feasible": True,
        "target_critic_fallback": "keep_tie_aware_minimum_target_twin_q",
        "stochastic_lane_decoder": "masked_categorical_actor_sampling",
        "speed_distribution": "lane_conditioned_squashed_gaussian",
        "critic_action": "normalised_speed_plus_lane_one_hot",
        "speed_entropy_coefficient": "auto_0.2",
        "speed_target_entropy": target_entropy,
        "lane_entropy_scale": lane_entropy_scale,
        "lane_exploration": "uniform_feasible_action_warmup_first_5000_raw_steps",
        "actor_encoder_detach": True,
        "reward_unchanged": True,
        "training_unchanged_from_v4_4": True,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_calibration_partition": "train",
        "checkpoint_calibration_episodes": 12,
        "checkpoint_selector_changed_from_v4_4": False,
        "validation_used_for_checkpoint_selection": False,
        "formal_test_used_for_checkpoint_selection": False,
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
    "V45_ALGORITHMS",
    "V45_IMPLEMENTATION_IDS",
    "make_model_v4_5",
    "source_action_repeat_v4_5",
]
