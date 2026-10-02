"""Learned-attention edge weighting for the D1-1 spatiotemporal arm.

Replaces the fixed Gaussian distance kernel that decides vehicle-graph message
weights with a small MLP over the per-pair edge features (relative position,
relative velocity, heading delta, TTC, same-lane, conflict), followed by a
masked softmax over the source axis.  This lets the *semantic* edge features
participate in deciding *who* is aggregated, instead of only feeding the
message content.

The original kernel is ``w_ij = exp(-d^2 / 2 sigma^2)`` with ``sigma = 20 m``,
which lets an irrelevant near car (d=8m -> w=0.92) out-weigh a critical
oncoming car (d=30m -> w=0.33) by ~2.8x.  A learned attention can instead learn
to up-weight small-TTC / oncoming pairs regardless of Euclidean distance.

Only *new* files are added; nothing in ``incremental_topo_encoder`` or the
hold35k modules is modified.  ``IncrementalTopoEncoderAttn`` inherits
``IncrementalTopoEncoder`` and overrides only ``_vehicle_edge_features`` (the
``use_topology=False`` path used by the D1-1 arm); ``use_topology=True`` keeps
the parent ``_vehicle_edge_features_v2`` path untouched.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import Tensor, nn

from .features import keras_initialize_linear
from .incremental_topo_encoder import IncrementalTopoEncoder


class IncrementalTopoEncoderAttn(IncrementalTopoEncoder):
    """IncrementalTopoEncoder with learned edge attention instead of the
    Gaussian distance kernel."""

    def __init__(
        self, observation_space: gym.spaces.Dict, *, topology_graph, **kwargs
    ) -> None:
        super().__init__(observation_space, topology_graph=topology_graph, **kwargs)
        # Per-pair semantic weight predictor: 8-dim edge features -> scalar logit.
        self.edge_weight_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.edge_weight_net.apply(keras_initialize_linear)

    def _vehicle_edge_features(
        self,
        trajectories: Tensor,
        valid: Tensor,
        topology_attention: Tensor | None,
        *,
        include_topology_relations: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Mirror the parent implementation verbatim, except the edge weights are
        # a learned attention over edge features rather than exp(-d^2 / 2s^2).
        states = trajectories.permute(0, 2, 1, 3)
        positions = states[..., :2]
        velocities = states[..., 3:5]
        headings = states[..., 2]
        delta_position = positions.unsqueeze(2) - positions.unsqueeze(3)
        delta_velocity = velocities.unsqueeze(2) - velocities.unsqueeze(3)
        heading_delta = headings.unsqueeze(2) - headings.unsqueeze(3)
        distance_squared = delta_position.square().sum(dim=-1)
        relative_speed_squared = delta_velocity.square().sum(dim=-1)
        closing = -(delta_position * delta_velocity).sum(dim=-1)
        ttc = torch.where(
            closing > 0.0,
            closing / relative_speed_squared.clamp_min(1e-6),
            torch.full_like(closing, 20.0),
        ).clamp(0.0, 20.0) / 20.0

        use_relations = (
            self.use_topology
            if include_topology_relations is None
            else bool(include_topology_relations)
        )
        if use_relations:
            if topology_attention is None:
                raise ValueError(
                    "topology_attention is required when topology relations are enabled"
                )
            alpha = topology_attention.permute(0, 2, 1, 3)
            same_lane = torch.einsum("bhim,bhjm->bhij", alpha, alpha)
            conflict = torch.einsum(
                "bhim,mn,bhjn->bhij",
                alpha,
                self.topology_conflict_adjacency,
                alpha,
            )
        else:
            same_lane = torch.zeros_like(distance_squared)
            conflict = torch.zeros_like(distance_squared)

        edge_features = torch.cat(
            [
                delta_position,
                delta_velocity,
                torch.cos(heading_delta)[..., None],
                ttc[..., None],
                same_lane[..., None],
                conflict[..., None],
            ],
            dim=-1,
        )

        node_valid = valid.permute(0, 2, 1).bool()
        pair_valid = node_valid.unsqueeze(3) & node_valid.unsqueeze(2)
        diagonal = torch.eye(
            self.actor_count, dtype=torch.bool, device=trajectories.device
        )[None, None]
        pair_valid = pair_valid & ~diagonal

        # Learned attention weights (replaces the Gaussian distance kernel).
        # -1e9 (not -inf) keeps softmax numerically safe on all-invalid rows.
        logits = self.edge_weight_net(edge_features).squeeze(-1)  # [B,H,N,N]
        masked_logits = logits.masked_fill(~pair_valid, -1e9)
        edge_weights = torch.softmax(masked_logits, dim=3)
        edge_weights = edge_weights * pair_valid.to(edge_weights.dtype)
        return edge_features, edge_weights, pair_valid, node_valid


__all__ = ["IncrementalTopoEncoderAttn"]
