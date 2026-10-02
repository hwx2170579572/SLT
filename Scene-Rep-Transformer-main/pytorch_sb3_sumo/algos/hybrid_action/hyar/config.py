"""HyAR factory."""

from __future__ import annotations

from typing import Any

from .hyar_algorithm import HyAR


def make_hyar_model(
    state_dim: int,
    *,
    num_discrete: int = 3,
    cont_dim: int = 1,
    latent_dim: int = 6,
    hidden_dim: int = 64,
    kl_beta: float = 1.0,
    lsc_coef: float = 0.1,
    **kwargs: Any,
) -> HyAR:
    """Construct the HyAR latent-space actor/critic skeleton."""

    return HyAR(
        state_dim=state_dim,
        num_discrete=num_discrete,
        cont_dim=cont_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        kl_beta=kl_beta,
        lsc_coef=lsc_coef,
    )
