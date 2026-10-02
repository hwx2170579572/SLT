"""Distance-free learned-attention vehicle aggregation for the v2 extractor.

Ablation for the hold35k collapse hypothesis.  ``VehicleGraphLayer`` aggregates
neighbour messages with a hard-coded Gaussian distance weight::

    edge_weights = exp(-distance^2 / (2 * vehicle_sigma^2)),  vehicle_sigma = 20

which attenuates a distant conflicting vehicle by ~1000x (75 m vs 9.6 m) even
though that vehicle's TTC / conflict edge features are computed.  This module
replaces that hard distance prior with a learned multi-head softmax attention
over neighbours, so a far approaching vehicle can contribute regardless of
range -- matching the free social attention that gives MST+SLT its gain.

``TopoTemporalGraphExtractorV2Attention`` is identical to
``TopoTemporalGraphExtractorV2`` except for ``self.vehicle_layers``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import keras_initialize_linear
from .topo_temporal_features import MaskedCrossAttention
from .topo_temporal_features_v2 import TopoTemporalGraphExtractorV2


class VehicleGraphAttentionLayer(nn.Module):
    """Residual learned-attention message passing over vehicle slots.

    Drop-in replacement for ``VehicleGraphLayer`` with the same forward
    signature.  The aggregation weight is a learned softmax attention
    (query = target token, key/value = sender message), not the Gaussian
    distance weight; ``edge_weights`` is accepted but ignored.
    """

    def __init__(self, feature_dim: int, edge_dim: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(feature_dim + edge_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.attention = MaskedCrossAttention(feature_dim, num_heads)
        self.norm = nn.LayerNorm(feature_dim)
        self.apply(keras_initialize_linear)

    def forward(
        self,
        features: Tensor,
        edge_features: Tensor,
        edge_weights: Tensor,
        pair_valid: Tensor,
        node_valid: Tensor,
    ) -> Tensor:
        # features: [B,H,N,D]; edge_features: [B,H,Nt,Ns,E]; pair_valid: [B,H,Nt,Ns].
        batch, history, actor_count, feature_dim = features.shape
        senders = features.unsqueeze(2).expand(-1, -1, actor_count, -1, -1)
        messages = self.message(torch.cat([senders, edge_features], dim=-1))
        # Learned attention over the actor_count sources for every target slot.
        query = features.reshape(batch * history * actor_count, 1, feature_dim)
        key_value = messages.reshape(
            batch * history * actor_count, actor_count, feature_dim
        )
        key_valid = pair_valid.reshape(batch * history * actor_count, actor_count)
        context, _ = self.attention(query, key_value, key_valid)
        context = context.reshape(batch, history, actor_count, feature_dim)
        output = self.norm(features + context)
        return output * node_valid[..., None].to(output.dtype)


class TopoTemporalGraphExtractorV2Attention(TopoTemporalGraphExtractorV2):
    """v2 extractor whose vehicle GCN aggregates neighbours with attention."""

    def __init__(
        self,
        observation_space,
        *,
        topology_graph,
        variant: str = "soft",
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            topology_graph=topology_graph,
            variant=variant,
            **kwargs,
        )
        self.vehicle_layers = nn.ModuleList(
            VehicleGraphAttentionLayer(self.feature_dim)
            for _ in range(len(self.vehicle_layers))
        )


__all__ = [
    "TopoTemporalGraphExtractorV2Attention",
    "VehicleGraphAttentionLayer",
]
