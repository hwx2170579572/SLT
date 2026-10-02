"""v4.5 confidence-gated deterministic lane-decoder SAC marker class."""

from __future__ import annotations

from .sac_v4_3 import TargetCriticLaneDecoderSACV43


class ConfidentActorFusionSACV45(TargetCriticLaneDecoderSACV43):
    """Keep v4.4 training unchanged; v4.5 changes deterministic decoding."""

    deterministic_lane_decoder = "actor_confident_non_keep_else_target_critic"
    actor_non_keep_confidence_threshold = 0.90


__all__ = ["ConfidentActorFusionSACV45"]
