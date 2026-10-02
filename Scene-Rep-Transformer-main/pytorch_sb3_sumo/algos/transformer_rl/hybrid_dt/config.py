"""Hybrid-DT factory."""

from __future__ import annotations

from typing import Any

from .hybrid_dt_model import HybridDecisionTransformer


def make_hybrid_dt(
    state_dim: int,
    *,
    hidden_size: int = 128,
    max_length: int | None = 20,
    max_ep_len: int = 4096,
    num_lane: int = 3,
    n_layer: int = 3,
    n_head: int = 1,
    **kwargs: Any,
) -> HybridDecisionTransformer:
    """Construct a Hybrid-action Decision Transformer."""

    return HybridDecisionTransformer(
        state_dim=state_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        num_lane=num_lane,
        n_layer=n_layer,
        n_head=n_head,
        **kwargs,
    )
