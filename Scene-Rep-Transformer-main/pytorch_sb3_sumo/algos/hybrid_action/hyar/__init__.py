"""HyAR baseline: latent-space policy over a conditional VAE hybrid action code."""

from .config import make_hyar_model
from .cvae import HybridActionVAE
from .hyar_algorithm import HyAR, LatentActor, LatentCritic
from .sumo_adapter import (
    LANE_COMMANDS,
    SumoHyARAdapter,
    env_action_to_hybrid,
    hybrid_to_env_action,
)

__all__ = [
    "HyAR",
    "HybridActionVAE",
    "LANE_COMMANDS",
    "LatentActor",
    "LatentCritic",
    "SumoHyARAdapter",
    "env_action_to_hybrid",
    "hybrid_to_env_action",
    "make_hyar_model",
]
