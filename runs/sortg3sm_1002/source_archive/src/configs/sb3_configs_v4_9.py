"""Registry for the v4.9 learned collision-critic model iteration."""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import SAC

from algos.sb3_torch.hybrid_policy_v4_9_model import (
    CollisionConstrainedFusionSACPolicyV49,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_9_model import CollisionConstrainedFusionSACV49
from algos.sb3_torch.topo_temporal_features_v2 import TopoTemporalGraphExtractorV2
from configs.sb3_configs_v4_8 import (
    TEMPORAL_GRAPH,
    make_model_v4_8,
    source_action_repeat_v4_8,
)
from envs.sumo.topology_graph_v2 import build_topology_graph_v2


V49_CANDIDATE = "topo_v4_9_learned_collision_critic"
V49_ALGORITHMS = (TEMPORAL_GRAPH, V49_CANDIDATE)
V49_FORMAL_ALGORITHMS = V49_ALGORITHMS
V49_N_STEP = 16
V49_COLLISION_RISK_COEF = 1.0
V49_IMPLEMENTATION_IDS = {
    TEMPORAL_GRAPH: "temporal_vehicle_graph_slt_pytorch_v1_v4_9_control",
    V49_CANDIDATE: (
        "full_decision_aligned_hybrid_factorized_entropy_horizon_correct_"
        "16_step_learned_twin_collision_critic_v4_9"
    ),
}


def source_action_repeat_v4_9(algo: str, requested: int | None = None) -> int:
    if algo not in V49_ALGORITHMS:
        raise ValueError(f"v4.9 algorithm must be one of {V49_ALGORITHMS}")
    parent = TEMPORAL_GRAPH if algo == TEMPORAL_GRAPH else "topo_v4_8_tie_only_replicated_calibration"
    return source_action_repeat_v4_8(parent, requested)


def make_model_v4_9(
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
    collision_risk_coef: float = V49_COLLISION_RISK_COEF,
    **kwargs: Any,
) -> SAC:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected v4.9 model arguments: {unknown}")
    if algo not in V49_ALGORITHMS:
        raise ValueError(f"v4.9 algorithm must be one of {V49_ALGORITHMS}")
    resolved_repeat = source_action_repeat_v4_9(algo, action_repeat)
    learning_rate = 1e-4 if learning_rate is None else float(learning_rate)
    if algo == TEMPORAL_GRAPH:
        model = make_model_v4_8(
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
            "implementation_id": V49_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_9_role": "unchanged_temporal_graph_control",
            "network_changed": False,
        }
        return model

    if abs(float(slot_balance_coef) - 0.01) > 1e-12:
        raise ValueError("v4.9 freezes SoftBalancedSlots at 0.01")
    if abs(float(collision_risk_coef) - V49_COLLISION_RISK_COEF) > 1e-12:
        raise ValueError("v4.9 freezes collision_risk_coef at 1.0")
    raw_env = env.unwrapped
    specification = getattr(raw_env, "specification", None)
    if specification is None or not hasattr(specification, "network_path"):
        raise TypeError("v4.9 topology method requires specification.network_path")
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
    model = CollisionConstrainedFusionSACV49(
        CollisionConstrainedFusionSACPolicyV49,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=0,
        batch_size=batch_size,
        tau=5e-3,
        gamma=discount,
        train_freq=(1, "step"),
        gradient_steps=resolved_repeat,
        n_steps=V49_N_STEP,
        replay_buffer_class=CollisionAwareHorizonReplayBufferV49,
        replay_buffer_kwargs={
            "n_steps": V49_N_STEP,
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
        collision_risk_coef=V49_COLLISION_RISK_COEF,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.topology_graph_info = topology_info.to_dict()
    model.graph_ablation = (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "horizon_correct_16_step_credit_plus_learned_collision_critic"
    )
    model.v4_method_metadata = {
        "implementation_id": V49_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_algorithm": "topo_v4_8_tie_only_replicated_calibration",
        "single_change": "learned_twin_collision_value_actor_constraint",
        "network_changed": True,
        "learned_collision_critic": True,
        "collision_critic_heads": 2,
        "collision_probability_link": "sigmoid",
        "collision_twin_aggregation": "maximum",
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "collision_risk_coef": V49_COLLISION_RISK_COEF,
        "collision_actor_objective": "reward_sac_objective_plus_lambda_expected_collision_value",
        "inference_safety_rule_added": False,
        "reward_changed": False,
        "observation_changed": False,
        "action_space_changed": False,
        "representation_backbone_changed": False,
        "return_n_step": V49_N_STEP,
        "bootstrap_discount": "gamma_power_actual_horizon",
        "selector_changed_from_v4_8": False,
        "decoder_changed_from_v4_8": True,
        "decoder_change_kind": "learned_reward_minus_collision_value_scoring",
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "deployment_decoder_candidates": ["target_critic", "fusion_0_90"],
        "actor_non_keep_confidence_threshold": 0.90,
        "actor_encoder_detach": True,
        "slot_balance_coef": 0.01,
    }
    return model


__all__ = [
    "TEMPORAL_GRAPH",
    "V49_ALGORITHMS",
    "V49_CANDIDATE",
    "V49_COLLISION_RISK_COEF",
    "V49_FORMAL_ALGORITHMS",
    "V49_IMPLEMENTATION_IDS",
    "V49_N_STEP",
    "make_model_v4_9",
    "source_action_repeat_v4_9",
]
