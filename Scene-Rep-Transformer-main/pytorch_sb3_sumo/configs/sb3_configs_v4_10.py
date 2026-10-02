"""Registry for the v4.10 shared-risk supported-mixture model."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4_10_model import (
    SharedRiskSupportedMixtureSACPolicyV410,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_10_model import (
    SharedRiskSupportedMixtureSACV410,
)
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs_v4_9 import (
    TEMPORAL_GRAPH,
    make_model_v4_9,
    source_action_repeat_v4_9,
)
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V410_FULL = "topo_v4_10_srsm_full"
V410_SHARED_SINGLE = "topo_v4_10_shared_single"
V410_MIXTURE_NO_SUPPORT = "topo_v4_10_mixture_no_support"
V410_CANDIDATES = (
    V410_FULL,
    V410_SHARED_SINGLE,
    V410_MIXTURE_NO_SUPPORT,
)
V410_ALGORITHMS = (TEMPORAL_GRAPH, *V410_CANDIDATES)
V410_FORMAL_ALGORITHMS = (TEMPORAL_GRAPH, V410_FULL)
V410_N_STEP = 16
V410_COLLISION_RISK_COEF = 1.0
V410_IMPLEMENTATION_IDS = {
    TEMPORAL_GRAPH: "temporal_vehicle_graph_slt_pytorch_v1_v4_10_control",
    V410_FULL: (
        "shared_risk_encoder_three_component_replay_supported_uncertainty_"
        "mixture_actor_v4_10"
    ),
    V410_SHARED_SINGLE: (
        "shared_risk_encoder_single_component_supported_actor_ablation_v4_10"
    ),
    V410_MIXTURE_NO_SUPPORT: (
        "shared_risk_encoder_three_component_no_support_uncertainty_ablation_v4_10"
    ),
}


def source_action_repeat_v4_10(
    algo: str, requested: int | None = None
) -> int:
    if algo not in V410_ALGORITHMS:
        raise ValueError(f"v4.10 algorithm must be one of {V410_ALGORITHMS}")
    parent = TEMPORAL_GRAPH if algo == TEMPORAL_GRAPH else "topo_v4_9_learned_collision_critic"
    return source_action_repeat_v4_9(parent, requested)


def _variant(algo: str) -> dict[str, float | int | str]:
    if algo == V410_FULL:
        return {
            "role": "primary_srsm_full",
            "speed_components": 3,
            "replay_support_coef": 0.05,
            "component_entropy_scale": 0.25,
            "twin_uncertainty_coef": 0.25,
            "component_prior_coef": 0.05,
        }
    if algo == V410_SHARED_SINGLE:
        return {
            "role": "ablation_shared_single",
            "speed_components": 1,
            "replay_support_coef": 0.05,
            "component_entropy_scale": 0.25,
            "twin_uncertainty_coef": 0.25,
            "component_prior_coef": 0.05,
        }
    if algo == V410_MIXTURE_NO_SUPPORT:
        return {
            "role": "ablation_mixture_no_support",
            "speed_components": 3,
            "replay_support_coef": 0.0,
            "component_entropy_scale": 0.25,
            "twin_uncertainty_coef": 0.0,
            "component_prior_coef": 0.0,
        }
    raise ValueError(f"unsupported v4.10 candidate {algo!r}")


def make_model_v4_10(
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
    collision_risk_coef: float = V410_COLLISION_RISK_COEF,
    **kwargs: Any,
) -> SAC:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected v4.10 model arguments: {unknown}")
    if algo not in V410_ALGORITHMS:
        raise ValueError(f"v4.10 algorithm must be one of {V410_ALGORITHMS}")
    resolved_repeat = source_action_repeat_v4_10(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    if algo == TEMPORAL_GRAPH:
        model = make_model_v4_9(
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
            "implementation_id": V410_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_10_role": "unchanged_temporal_graph_control",
            "network_changed": False,
        }
        return model

    if abs(float(slot_balance_coef) - 0.01) > 1e-12:
        raise ValueError("v4.10 freezes SoftBalancedSlots at 0.01")
    if abs(float(collision_risk_coef) - V410_COLLISION_RISK_COEF) > 1e-12:
        raise ValueError("v4.10 freezes collision_risk_coef at 1.0")
    values = _variant(algo)
    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("v4.10 topology method requires specification.network_path")
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
        "speed_components": int(values["speed_components"]),
        "twin_uncertainty_coef": float(values["twin_uncertainty_coef"]),
        "component_prior_coef": float(values["component_prior_coef"]),
    }
    model = SharedRiskSupportedMixtureSACV410(
        SharedRiskSupportedMixtureSACPolicyV410,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=resolved_repeat,
        n_steps=V410_N_STEP,
        replay_buffer_class=CollisionAwareHorizonReplayBufferV49,
        replay_buffer_kwargs={
            "n_steps": V410_N_STEP,
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
        collision_risk_coef=V410_COLLISION_RISK_COEF,
        replay_support_coef=float(values["replay_support_coef"]),
        component_entropy_scale=float(values["component_entropy_scale"]),
        twin_uncertainty_coef=float(values["twin_uncertainty_coef"]),
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = str(values["role"])
    model.v4_method_metadata = {
        "implementation_id": V410_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_algorithm": "topo_v4_9_learned_collision_critic",
        "v4_10_role": values["role"],
        "network_changed": True,
        "shared_online_scene_encoder": True,
        "collision_supervision_shapes_actor_representation": True,
        "actor_encoder_detach": True,
        "speed_components_per_lane": int(values["speed_components"]),
        "fixed_speed_grid_at_inference": False,
        "replay_support_objective": "latent_speed_mixture_negative_log_likelihood",
        "replay_support_coef": float(values["replay_support_coef"]),
        "component_entropy_scale": float(values["component_entropy_scale"]),
        "twin_uncertainty_coef": float(values["twin_uncertainty_coef"]),
        "component_prior_coef": float(values["component_prior_coef"]),
        "collision_risk_coef": V410_COLLISION_RISK_COEF,
        "deterministic_model_score": (
            "min_reward_q_minus_max_collision_value_minus_twin_disagreement_"
            "plus_log_component_probability"
        ),
        "optimizer_parameter_sets_must_be_disjoint": True,
        "encoder_optimizer_owner_count": 1,
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "reward_changed": False,
        "observation_changed": False,
        "action_space_changed": False,
        "return_n_step": V410_N_STEP,
        "bootstrap_discount": "gamma_power_actual_horizon",
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "deployment_decoder_candidates": ["target_critic"],
        "slot_balance_coef": 0.01,
        "lane_entropy_scale": 0.0,
    }
    return model


__all__ = [
    "TEMPORAL_GRAPH",
    "V410_ALGORITHMS",
    "V410_CANDIDATES",
    "V410_COLLISION_RISK_COEF",
    "V410_FORMAL_ALGORITHMS",
    "V410_FULL",
    "V410_IMPLEMENTATION_IDS",
    "V410_MIXTURE_NO_SUPPORT",
    "V410_N_STEP",
    "V410_SHARED_SINGLE",
    "make_model_v4_10",
    "source_action_repeat_v4_10",
]
