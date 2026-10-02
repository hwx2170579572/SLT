"""v4.9 keep-tie-veto fusion SAC marker class."""

from __future__ import annotations

from .sac_v4_5 import ConfidentActorFusionSACV45


class KeepTieVetoActorFusionSACV49(ConfidentActorFusionSACV45):
    """Keep training unchanged and apply the v4.9 deterministic decoder."""

    deterministic_lane_decoder = (
        "actor_confident_non_keep_unless_target_keep_tied_max_else_target_critic"
    )


__all__ = ["KeepTieVetoActorFusionSACV49"]
