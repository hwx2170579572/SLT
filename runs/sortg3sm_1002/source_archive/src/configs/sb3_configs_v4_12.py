"""Registry for the v4.12 augmented joint-support PRCR model."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_12_model import (
    AugmentedJointSupportPRCRSACV412,
)
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs_v4_11 import (
    TEMPORAL_GRAPH,
    V411_COLLISION_RISK_COEF,
    V411_N_STEP,
    V411_PAIRWISE_RANK_COEF,
    V411_TEMPORAL_CONSISTENCY_COEF,
    make_model_v4_11,
    source_action_repeat_v4_11,
)
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V412_FULL = "topo_v4_12_augmented_joint_support_prcr_full"
V412_NO_CROSS_AUGMENTATION = (
    "topo_v4_12_joint_support_no_cross_augmentation"
)
V412_AUGMENTATION_ONLY = "topo_v4_12_cross_augmentation_only"
V412_CANDIDATES = (
    V412_FULL,
    V412_NO_CROSS_AUGMENTATION,
    V412_AUGMENTATION_ONLY,
)
V412_ALGORITHMS = (TEMPORAL_GRAPH, *V412_CANDIDATES)
V412_FORMAL_ALGORITHMS = (TEMPORAL_GRAPH, V412_FULL)
V412_LANE_PRIOR_COEF = 0.05
V412_LANE_SUPPORT_SCALE = 1.0
V412_IMPLEMENTATION_IDS = {
    TEMPORAL_GRAPH: "temporal_vehicle_graph_slt_pytorch_v1_v4_12_control",
    V412_FULL: (
        "augmented_joint_lane_speed_replay_support_proper_calibrated_"
        "ranked_risk_v4_12"
    ),
    V412_NO_CROSS_AUGMENTATION: (
        "joint_lane_speed_replay_support_no_cross_augmentation_"
        "proper_calibrated_ranked_risk_v4_12"
    ),
    V412_AUGMENTATION_ONLY: (
        "cross_rotation_augmentation_speed_only_support_"
        "proper_calibrated_ranked_risk_v4_12"
    ),
}


def source_action_repeat_v4_12(
    algo: str, requested: int | None = None
) -> int:
    if algo not in V412_ALGORITHMS:
        raise ValueError(f"v4.12 algorithm must be one of {V412_ALGORITHMS}")
    return source_action_repeat_v4_11(
        TEMPORAL_GRAPH if algo == TEMPORAL_GRAPH else "topo_v4_11_prcr_full",
        requested,
    )


def _variant(algo: str) -> dict[str, float | bool | str]:
    if algo == V412_FULL:
        return {
            "role": "primary_augmented_joint_support",
            "lane_support_scale": V412_LANE_SUPPORT_SCALE,
            "lane_prior_coef": V412_LANE_PRIOR_COEF,
            "cross_rotation_augmentation": True,
        }
    if algo == V412_NO_CROSS_AUGMENTATION:
        return {
            "role": "ablation_no_cross_augmentation",
            "lane_support_scale": V412_LANE_SUPPORT_SCALE,
            "lane_prior_coef": V412_LANE_PRIOR_COEF,
            "cross_rotation_augmentation": False,
        }
    if algo == V412_AUGMENTATION_ONLY:
        return {
            "role": "ablation_cross_augmentation_only",
            "lane_support_scale": 0.0,
            "lane_prior_coef": 0.0,
            "cross_rotation_augmentation": True,
        }
    raise ValueError(f"unsupported v4.12 candidate {algo!r}")


def make_model_v4_12(
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
    collision_risk_coef: float = V411_COLLISION_RISK_COEF,
    **kwargs: Any,
) -> SAC:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected v4.12 model arguments: {unknown}")
    if algo not in V412_ALGORITHMS:
        raise ValueError(f"v4.12 algorithm must be one of {V412_ALGORITHMS}")
    resolved_repeat = source_action_repeat_v4_12(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    if algo == TEMPORAL_GRAPH:
        model = make_model_v4_11(
            TEMPORAL_GRAPH,
            env,
            scenario=scenario,
            learning_rate=learning_rate,
            batch_size=batch_size,
            discount=discount,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            action_repeat=resolved_repeat,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            slot_balance_coef=slot_balance_coef,
        )
        parent = dict(getattr(model, "v4_method_metadata", {}) or {})
        model.v4_method_metadata = {
            **parent,
            "implementation_id": V412_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_12_role": "unchanged_temporal_graph_control",
            "network_changed": False,
        }
        return model

    if abs(float(slot_balance_coef) - 0.01) > 1e-12:
        raise ValueError("v4.12 freezes SoftBalancedSlots at 0.01")
    if abs(float(collision_risk_coef) - V411_COLLISION_RISK_COEF) > 1e-12:
        raise ValueError("v4.12 freezes collision_risk_coef at 1.0")
    values = _variant(algo)
    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("v4.12 topology method requires specification.network_path")
    topology_graph, topology_info = build_topology_graph_v2(
        specification.network_path,
        coordinate_offset=tuple(
            getattr(specification, "coordinate_offset", (0.0, 0.0))
        ),
        return_info=True,
    )
    random_augmentation = scenario != "cross" or bool(
        values["cross_rotation_augmentation"]
    )
    feature_kwargs = {
        "topology_graph": topology_graph,
        "variant": "soft",
        "features_dim": 128,
        "hidden_dim": 128,
        "num_heads": 2,
        "random_augmentation": random_augmentation,
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
        "speed_components": 3,
        "twin_uncertainty_coef": 0.25,
        "component_prior_coef": 0.05,
        "lane_prior_coef": float(values["lane_prior_coef"]),
        "lane_support_scale": float(values["lane_support_scale"]),
    }
    model = AugmentedJointSupportPRCRSACV412(
        AugmentedJointSupportPRCRPolicyV412,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=resolved_repeat,
        n_steps=V411_N_STEP,
        replay_buffer_class=CollisionAwareHorizonReplayBufferV49,
        replay_buffer_kwargs={
            "n_steps": V411_N_STEP,
            "gamma": discount,
            "duplicate_episode_end_transition": True,
            "source_action_repeat": resolved_repeat,
        },
        ent_coef="auto_0.2",
        target_entropy=-1.0,
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
        lane_entropy_scale=0.0,
        collision_risk_coef=V411_COLLISION_RISK_COEF,
        replay_support_coef=0.05,
        component_entropy_scale=0.25,
        twin_uncertainty_coef=0.25,
        collision_pairwise_rank_coef=V411_PAIRWISE_RANK_COEF,
        collision_temporal_consistency_coef=V411_TEMPORAL_CONSISTENCY_COEF,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = str(values["role"])
    model.v4_method_metadata = {
        "implementation_id": V412_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_algorithm": "topo_v4_11_prcr_full",
        "v4_12_role": values["role"],
        "network_changed": True,
        "shared_online_scene_encoder": True,
        "actor_encoder_detach": True,
        "speed_components_per_lane": 3,
        "replay_support_objective": (
            "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll"
        ),
        "replay_support_coef": 0.05,
        "lane_support_scale": float(values["lane_support_scale"]),
        "lane_prior_coef": float(values["lane_prior_coef"]),
        "component_prior_coef": 0.05,
        "cross_rotation_augmentation": bool(
            values["cross_rotation_augmentation"]
        ),
        "effective_random_rotation_augmentation": random_augmentation,
        "collision_risk_coef": V411_COLLISION_RISK_COEF,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": V411_PAIRWISE_RANK_COEF,
        "collision_temporal_consistency_coef": V411_TEMPORAL_CONSISTENCY_COEF,
        "deterministic_model_score": (
            "min_reward_q_minus_max_collision_value_minus_twin_disagreement_"
            "plus_log_component_probability_plus_log_lane_probability"
        ),
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "reward_changed": False,
        "observation_changed": False,
        "action_space_changed": False,
        "return_n_step": V411_N_STEP,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "deployment_decoder_candidates": ["target_critic"],
        "slot_balance_coef": 0.01,
        "lane_entropy_scale": 0.0,
    }
    return model


__all__ = [
    "TEMPORAL_GRAPH",
    "V412_ALGORITHMS",
    "V412_AUGMENTATION_ONLY",
    "V412_CANDIDATES",
    "V412_FORMAL_ALGORITHMS",
    "V412_FULL",
    "V412_IMPLEMENTATION_IDS",
    "V412_LANE_PRIOR_COEF",
    "V412_LANE_SUPPORT_SCALE",
    "V412_NO_CROSS_AUGMENTATION",
    "make_model_v4_12",
    "source_action_repeat_v4_12",
]

