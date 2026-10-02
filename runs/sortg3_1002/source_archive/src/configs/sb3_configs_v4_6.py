"""Registry for v4.6 train-only joint checkpoint/decoder selection."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import SAC

from configs.sb3_configs_v4_5 import make_model_v4_5, source_action_repeat_v4_5


V46_ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_v4_6_joint_checkpoint_decoder_selector",
)
V46_IMPLEMENTATION_IDS = {
    "temporal_graph_v1_control": (
        "temporal_vehicle_graph_slt_pytorch_v1_v4_6_selector_control"
    ),
    "topo_v4_6_joint_checkpoint_decoder_selector": (
        "full_decision_aligned_hybrid_factorized_entropy_train_only_joint_"
        "checkpoint_decoder_selector_v4_6"
    ),
}


def source_action_repeat_v4_6(algo: str, requested: int | None = None) -> int:
    if algo not in V46_ALGORITHMS:
        raise ValueError(f"v4.6 algorithm must be one of {V46_ALGORITHMS}")
    parent = (
        "temporal_graph_v1_control"
        if algo == "temporal_graph_v1_control"
        else "topo_v4_5_confident_actor_fusion"
    )
    return source_action_repeat_v4_5(parent, requested)


def make_model_v4_6(
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
    **kwargs: Any,
) -> SAC:
    """Build the unchanged v4.5 learner and attach v4.6 selector metadata."""

    if algo not in V46_ALGORITHMS:
        raise ValueError(f"v4.6 algorithm must be one of {V46_ALGORITHMS}")
    parent = (
        "temporal_graph_v1_control"
        if algo == "temporal_graph_v1_control"
        else "topo_v4_5_confident_actor_fusion"
    )
    model = make_model_v4_5(
        parent,
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
        slot_balance_coef=slot_balance_coef,
        **kwargs,
    )
    parent_metadata = dict(getattr(model, "v4_method_metadata", {}) or {})
    if algo == "temporal_graph_v1_control":
        model.v4_method_metadata = {
            **parent_metadata,
            "implementation_id": V46_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "parent_implementation_id": parent_metadata.get("implementation_id"),
            "single_change": "none_parent_control",
            "train_only_deployment_selector": "checkpoint_only_parent_control",
            "checkpoint_decoder_candidate_pairs": 2,
            "checkpoint_calibration_partition": "train",
            "checkpoint_calibration_episodes": 12,
            "validation_used_for_checkpoint_selection": False,
            "formal_test_used_for_checkpoint_selection": False,
        }
        return model

    model.graph_ablation = (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "train_only_joint_checkpoint_decoder_selector"
    )
    model.v4_method_metadata = {
        **parent_metadata,
        "implementation_id": V46_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_implementation_id": parent_metadata.get("implementation_id"),
        "single_change": "train_only_joint_checkpoint_decoder_selector",
        "training_policy_class": type(model.policy).__name__,
        "training_unchanged_from_v4_5": True,
        "deterministic_lane_decoder": "selected_train_only_from_target_or_fusion_0_90",
        "deployment_decoder_candidates": ["target_critic", "fusion_0_90"],
        "actor_non_keep_confidence_threshold": 0.90,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_decoder_candidate_pairs": 4,
        "checkpoint_calibration_partition": "train",
        "checkpoint_calibration_episodes": 12,
        "identical_unique_traffic_block_for_all_pairs": True,
        "validation_used_for_checkpoint_selection": False,
        "formal_test_used_for_checkpoint_selection": False,
        "reward_unchanged": True,
        "encoder_unchanged": True,
        "decoder_definitions_unchanged": True,
        "fusion_threshold_unchanged": True,
    }
    return model


__all__ = [
    "V46_ALGORITHMS",
    "V46_IMPLEMENTATION_IDS",
    "make_model_v4_6",
    "source_action_repeat_v4_6",
]
