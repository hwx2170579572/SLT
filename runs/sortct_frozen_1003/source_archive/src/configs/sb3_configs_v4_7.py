"""Registry for the v4.7 horizon-correct long-return iteration."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import SAC

from algos.sb3_torch.replay_buffer import DictNStepReplayBuffer
from algos.sb3_torch.replay_buffer_v4_7 import (
    HorizonCorrectDictNStepReplayBufferV47,
)
from configs.sb3_configs_v4_6 import make_model_v4_6, source_action_repeat_v4_6


TEMPORAL_GRAPH = "temporal_graph_v1_control"
PARENT_CONTROL = "topo_v4_6_joint_checkpoint_decoder_selector"
V47_CANDIDATE = "topo_v4_7_horizon_correct_credit"
V47_ALGORITHMS = (TEMPORAL_GRAPH, PARENT_CONTROL, V47_CANDIDATE)
V47_FORMAL_ALGORITHMS = (TEMPORAL_GRAPH, V47_CANDIDATE)
V47_IMPLEMENTATION_IDS = {
    TEMPORAL_GRAPH: "temporal_vehicle_graph_slt_pytorch_v1_v4_7_control",
    PARENT_CONTROL: (
        "full_decision_aligned_hybrid_factorized_entropy_train_only_joint_"
        "checkpoint_decoder_selector_v4_6_matched_control"
    ),
    V47_CANDIDATE: (
        "full_decision_aligned_hybrid_factorized_entropy_horizon_correct_"
        "16_step_credit_joint_selector_v4_7"
    ),
}
V47_N_STEP = 16


def source_action_repeat_v4_7(algo: str, requested: int | None = None) -> int:
    if algo not in V47_ALGORITHMS:
        raise ValueError(f"v4.7 algorithm must be one of {V47_ALGORITHMS}")
    parent = TEMPORAL_GRAPH if algo == TEMPORAL_GRAPH else PARENT_CONTROL
    return source_action_repeat_v4_6(parent, requested)


def _replace_empty_replay_buffer(
    model: SAC,
    *,
    discount: float,
    action_repeat: int,
) -> None:
    parent = model.replay_buffer
    if not isinstance(parent, DictNStepReplayBuffer):
        raise TypeError("v4.7 candidate expected the frozen dictionary N-step buffer")
    if parent.size() != 0 or int(parent._next_insertion_id) != 0:
        raise ValueError("v4.7 may only replace an empty parent replay buffer")
    model.n_steps = V47_N_STEP
    model.replay_buffer_class = HorizonCorrectDictNStepReplayBufferV47
    model.replay_buffer_kwargs = {
        "n_steps": V47_N_STEP,
        "gamma": float(discount),
        "duplicate_episode_end_transition": True,
        "source_action_repeat": int(action_repeat),
    }
    model.replay_buffer = HorizonCorrectDictNStepReplayBufferV47(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        **model.replay_buffer_kwargs,
    )


def make_model_v4_7(
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
    if algo not in V47_ALGORITHMS:
        raise ValueError(f"v4.7 algorithm must be one of {V47_ALGORITHMS}")
    resolved_repeat = source_action_repeat_v4_7(algo, action_repeat)
    parent_algo = TEMPORAL_GRAPH if algo == TEMPORAL_GRAPH else PARENT_CONTROL
    model = make_model_v4_6(
        parent_algo,
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
        **kwargs,
    )
    parent_metadata = dict(getattr(model, "v4_method_metadata", {}) or {})
    if algo == TEMPORAL_GRAPH:
        model.v4_method_metadata = {
            **parent_metadata,
            "implementation_id": V47_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_7_role": "unchanged_temporal_graph_control",
            "return_estimator_changed": False,
        }
        return model
    if algo == PARENT_CONTROL:
        model.v4_method_metadata = {
            **parent_metadata,
            "implementation_id": V47_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_7_role": "matched_v4_6_parent_return_estimator_control",
            "return_estimator_changed": False,
            "n_step": 4,
            "bootstrap_discount": "single_gamma_source_equivalent",
        }
        return model

    _replace_empty_replay_buffer(
        model,
        discount=discount,
        action_repeat=resolved_repeat,
    )
    model.graph_ablation = (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "horizon_correct_16_step_credit_plus_joint_selector"
    )
    model.v4_method_metadata = {
        **parent_metadata,
        "implementation_id": V47_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_implementation_id": parent_metadata.get("implementation_id"),
        "single_change": "horizon_correct_16_step_terminal_credit",
        "return_estimator_changed": True,
        "parent_n_step": 4,
        "n_step": V47_N_STEP,
        "reward_accumulation": "sum_gamma_power_i_reward",
        "actual_horizon": "min_16_or_first_episode_boundary",
        "bootstrap_discount": "gamma_power_actual_horizon",
        "source_timeout_semantics_changed": False,
        "one_step_next_observation_for_graph_slt_changed": False,
        "episode_end_transition_duplication_changed": False,
        "uniform_replay_sampling_changed": False,
        "replay_capacity_changed": False,
        "reward_changed": False,
        "network_changed": False,
        "entropy_changed": False,
        "selector_changed": False,
        "decoder_changed": False,
    }
    return model


__all__ = [
    "PARENT_CONTROL",
    "TEMPORAL_GRAPH",
    "V47_ALGORITHMS",
    "V47_CANDIDATE",
    "V47_FORMAL_ALGORITHMS",
    "V47_IMPLEMENTATION_IDS",
    "V47_N_STEP",
    "make_model_v4_7",
    "source_action_repeat_v4_7",
]
