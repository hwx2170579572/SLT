"""v4.9a critic-regret-guarded actor fusion SAC marker class."""

from __future__ import annotations

from .hybrid_policy_v4_9_qguard import DECODER
from .sac_v4_5 import ConfidentActorFusionSACV45


class CriticRegretGuardActorFusionSACV49A(ConfidentActorFusionSACV45):
    """Keep training unchanged and apply the v4.9a Q-regret decoder."""

    deterministic_lane_decoder = DECODER
    maximum_actor_target_q_regret = 0.05


__all__ = ["CriticRegretGuardActorFusionSACV49A"]
