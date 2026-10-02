"""Registry for the v4.8 tie-only replicated-calibration iteration."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import SAC

from configs.sb3_configs_v4_7 import (
    PARENT_CONTROL,
    TEMPORAL_GRAPH,
    V47_CANDIDATE,
    V47_N_STEP,
    make_model_v4_7,
    source_action_repeat_v4_7,
)


V48_CANDIDATE = "topo_v4_8_tie_only_replicated_calibration"
V48_MINUS_HORIZON = "topo_v4_8_minus_horizon_correct_credit"
V48_MINUS_FACTORIZED = "topo_v4_8_minus_factorized_entropy"
V48_ALGORITHMS = (
    TEMPORAL_GRAPH,
    V48_CANDIDATE,
    V48_MINUS_HORIZON,
    V48_MINUS_FACTORIZED,
)
V48_FORMAL_ALGORITHMS = (TEMPORAL_GRAPH, V48_CANDIDATE)
V48_IMPLEMENTATION_IDS = {
    TEMPORAL_GRAPH: "temporal_vehicle_graph_slt_pytorch_v1_v4_8_control",
    V48_CANDIDATE: (
        "full_decision_aligned_hybrid_factorized_entropy_horizon_correct_"
        "16_step_credit_tie_only_replicated_joint_selector_v4_8"
    ),
    V48_MINUS_HORIZON: (
        "full_decision_aligned_hybrid_factorized_entropy_tie_only_replicated_"
        "joint_selector_minus_horizon_correct_credit_v4_8"
    ),
    V48_MINUS_FACTORIZED: (
        "full_decision_aligned_hybrid_joint_entropy_horizon_correct_"
        "16_step_credit_tie_only_replicated_joint_selector_minus_"
        "factorized_entropy_v4_8"
    ),
}

_TIE_ONLY_SELECTOR_METADATA = {
    "selector_changed": True,
    "initial_calibration_episodes_per_pair": 12,
    "secondary_calibration_trigger": "empirical_tied_top_count_greater_than_one",
    "secondary_calibration_episodes_per_tied_pair": 12,
    "secondary_calibration_seed_offset": 100,
    "non_top_candidate_reentry_forbidden": True,
    "decoder_changed": False,
}

_GRAPH_ABLATION_BY_ALGO = {
    V48_CANDIDATE: (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "horizon_correct_16_step_credit_plus_tie_only_replicated_selector"
    ),
    V48_MINUS_HORIZON: (
        "topology_v2_soft_plus_hybrid_action_plus_factorized_entropy_plus_"
        "tie_only_replicated_selector_minus_horizon_correct_credit"
    ),
    V48_MINUS_FACTORIZED: (
        "topology_v2_soft_plus_hybrid_action_plus_joint_entropy_plus_"
        "horizon_correct_16_step_credit_plus_tie_only_replicated_selector"
    ),
}

_SINGLE_CHANGE_BY_ALGO = {
    V48_CANDIDATE: "tie_only_replicated_train_calibration",
    V48_MINUS_HORIZON: "minus_horizon_correct_credit",
    V48_MINUS_FACTORIZED: "minus_factorized_entropy",
}


def source_action_repeat_v4_8(algo: str, requested: int | None = None) -> int:
    if algo not in V48_ALGORITHMS:
        raise ValueError(f"v4.8 algorithm must be one of {V48_ALGORITHMS}")
    if algo == TEMPORAL_GRAPH:
        parent = TEMPORAL_GRAPH
    elif algo == V48_MINUS_HORIZON:
        parent = PARENT_CONTROL
    else:
        parent = V47_CANDIDATE
    return source_action_repeat_v4_7(parent, requested)


def make_model_v4_8(
    algo: str,
    env: Any,
    **kwargs: Any,
) -> SAC:
    if algo not in V48_ALGORITHMS:
        raise ValueError(f"v4.8 algorithm must be one of {V48_ALGORITHMS}")
    if algo == V48_MINUS_FACTORIZED:
        kwargs = {**kwargs, "lane_entropy_scale": 1.0, "target_entropy": -2.0}
    if algo == TEMPORAL_GRAPH:
        parent_algo = TEMPORAL_GRAPH
    elif algo == V48_MINUS_HORIZON:
        parent_algo = PARENT_CONTROL
    else:
        parent_algo = V47_CANDIDATE
    model = make_model_v4_7(parent_algo, env, **kwargs)
    parent = dict(getattr(model, "v4_method_metadata", {}) or {})
    if algo == TEMPORAL_GRAPH:
        model.v4_method_metadata = {
            **parent,
            "implementation_id": V48_IMPLEMENTATION_IDS[algo],
            "algorithm": algo,
            "v4_8_role": "unchanged_temporal_graph_control",
            "selector_changed": False,
        }
        return model
    model.graph_ablation = _GRAPH_ABLATION_BY_ALGO[algo]
    extra: dict[str, Any] = {}
    if algo == V48_MINUS_HORIZON:
        extra = {
            "n_step": 4,
            "bootstrap_discount": "single_gamma_source_equivalent",
            "reference_full_model": V48_CANDIDATE,
            "ablation_direction": "subtract_horizon_correct_16_step_credit",
        }
    elif algo == V48_MINUS_FACTORIZED:
        extra = {
            "n_step": V47_N_STEP,
            "lane_entropy_scale": 1.0,
            "speed_target_entropy": -2.0,
            "reference_full_model": V48_CANDIDATE,
            "ablation_direction": "subtract_factorized_entropy_restore_joint_entropy",
        }
    model.v4_method_metadata = {
        **parent,
        "implementation_id": V48_IMPLEMENTATION_IDS[algo],
        "algorithm": algo,
        "parent_implementation_id": parent.get("implementation_id"),
        "single_change": _SINGLE_CHANGE_BY_ALGO[algo],
        "return_estimator_changed_from_v4_7": False,
        "training_changed_from_v4_7": False,
        **_TIE_ONLY_SELECTOR_METADATA,
        **extra,
    }
    return model


__all__ = [
    "TEMPORAL_GRAPH",
    "V48_ALGORITHMS",
    "V48_CANDIDATE",
    "V48_MINUS_HORIZON",
    "V48_MINUS_FACTORIZED",
    "V48_FORMAL_ALGORITHMS",
    "V48_IMPLEMENTATION_IDS",
    "make_model_v4_8",
    "source_action_repeat_v4_8",
]

