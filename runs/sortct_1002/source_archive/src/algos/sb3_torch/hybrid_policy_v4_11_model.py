"""Proper-calibrated ranked-risk policy identity for the v4.11 model."""

from __future__ import annotations

from .hybrid_policy_v4_10_model import (
    SharedRiskSupportedMixtureSACPolicyV410,
    SupportedMixtureActionBatch,
    SupportedMixtureHybridActor,
    all_proposal_q_from_features,
)


class ProperCalibratedRankedRiskSACPolicyV411(
    SharedRiskSupportedMixtureSACPolicyV410
):
    """v4.10 proposal/inference path with a separately trained v4.11 risk model."""

    scientific_version = "v4.11_proper_calibrated_ranked_risk"
    collision_return_loss = "soft_target_binary_cross_entropy_with_logits"
    collision_pairwise_ranking = True
    collision_temporal_consistency = True
    inference_safety_rule_added = False
    external_kinematic_projection = False
    action_postprocessing_override = False


__all__ = [
    "ProperCalibratedRankedRiskSACPolicyV411",
    "SupportedMixtureActionBatch",
    "SupportedMixtureHybridActor",
    "all_proposal_q_from_features",
]
