"""v4.3 target-critic lane-decoder SAC marker class."""

from __future__ import annotations

from .sac_v4_2 import FactorizedEntropyHybridSACV42


class TargetCriticLaneDecoderSACV43(FactorizedEntropyHybridSACV42):
    """Keep v4.2 training unchanged; behavior is supplied by the v4.3 policy."""

    deterministic_lane_decoder = "argmax_feasible_min_target_twin_q"


__all__ = ["TargetCriticLaneDecoderSACV43"]
