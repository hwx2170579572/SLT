"""Registry for v4.4 train-only deployment-aware checkpoint selection."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import SAC

from configs.sb3_configs_v4_3 import make_model_v4_3, source_action_repeat_v4_3


V44_ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_v4_4_train_only_selector",
)
V44_IMPLEMENTATION_IDS = {
    "temporal_graph_v1_control": (
        "temporal_vehicle_graph_slt_pytorch_v1_v4_4_train_only_selector_control"
    ),
    "topo_v4_4_train_only_selector": (
        "full_decision_aligned_hybrid_factorized_entropy_target_critic_"
        "train_only_selector_v4_4"
    ),
}


def source_action_repeat_v4_4(algo: str, requested: int | None = None) -> int:
    if algo not in V44_ALGORITHMS:
        raise ValueError(f"v4.4 algorithm must be one of {V44_ALGORITHMS}")
    parent = (
        "temporal_graph_v1_control"
        if algo == "temporal_graph_v1_control"
        else "topo_v4_3_target_critic_decoder"
    )
    return source_action_repeat_v4_3(parent, requested)


def make_model_v4_4(
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
    if algo not in V44_ALGORITHMS:
        raise ValueError(f"v4.4 algorithm must be one of {V44_ALGORITHMS}")
    parent = (
        "temporal_graph_v1_control"
        if algo == "temporal_graph_v1_control"
        else "topo_v4_3_target_critic_decoder"
    )
    model = make_model_v4_3(
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
    model.v4_method_metadata = {
        **parent_metadata,
        "implementation_id": V44_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_algorithm": parent,
        "parent_implementation_id": parent_metadata.get("implementation_id"),
        "single_change": "train_only_deployment_aware_checkpoint_selector",
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_calibration_partition": "train",
        "checkpoint_calibration_episodes": 12,
        "checkpoint_selector_applied_after_training": True,
        "training_unchanged_from_v4_3": True,
        "deterministic_decoder_unchanged_from_v4_3": True,
        "validation_used_for_checkpoint_selection": False,
        "formal_test_used_for_checkpoint_selection": False,
    }
    return model


__all__ = [
    "V44_ALGORITHMS",
    "V44_IMPLEMENTATION_IDS",
    "make_model_v4_4",
    "source_action_repeat_v4_4",
]
