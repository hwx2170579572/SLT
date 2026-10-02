"""Attention-aggregation ablation factory for the v4.8 hold35k method.

Builds the exact v4.8 hold35k model (``V48_CANDIDATE``) but swaps the vehicle
GCN's Gaussian distance aggregation for learned attention.  Everything else --
factorized entropy, confident-actor hybrid head, horizon-correct 16-step replay,
Graph-SLT, selector metadata -- is identical to hold35k.

The swap is done by temporarily replacing the extractor class referenced inside
``sb3_configs_v4_5`` (which hard-codes ``TopoTemporalGraphExtractorV2`` in its
policy kwargs) before ``make_model_v4_8`` runs, so the optimizers are built with
the attention extractor's parameters.  No existing file is modified.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3 import SAC

import configs.sb3_configs_v4_5 as _v45
from algos.sb3_torch.topo_temporal_features_v2_attention import (
    TopoTemporalGraphExtractorV2Attention,
)
from configs.sb3_configs_v4_8 import V48_CANDIDATE, make_model_v4_8


V48_ATTENTION = "topo_v4_8_attention_vehicle_aggregation"


def make_model_v4_8_attention(env: Any, **kwargs: Any) -> SAC:
    original = _v45.TopoTemporalGraphExtractorV2
    _v45.TopoTemporalGraphExtractorV2 = TopoTemporalGraphExtractorV2Attention
    try:
        model = make_model_v4_8(V48_CANDIDATE, env, **kwargs)
    finally:
        _v45.TopoTemporalGraphExtractorV2 = original

    model.graph_ablation = (
        "topology_v2_soft_plus_attention_vehicle_aggregation_plus_hybrid_action_"
        "plus_factorized_entropy_plus_horizon_correct_16_step_credit_plus_"
        "tie_only_replicated_selector"
    )
    metadata = dict(getattr(model, "v4_method_metadata", {}) or {})
    model.v4_method_metadata = {
        **metadata,
        "implementation_id": (
            "full_decision_aligned_hybrid_factorized_entropy_horizon_correct_"
            "16_step_credit_tie_only_replicated_joint_selector_attention_"
            "vehicle_aggregation_v4_8"
        ),
        "algorithm": V48_ATTENTION,
        "parent_implementation_id": metadata.get("implementation_id"),
        "single_change": "vehicle_gcn_distance_weight_to_learned_attention",
        "vehicle_sigma_removed": True,
        "reference_full_model": V48_CANDIDATE,
    }
    return model


__all__ = ["V48_ATTENTION", "make_model_v4_8_attention"]
