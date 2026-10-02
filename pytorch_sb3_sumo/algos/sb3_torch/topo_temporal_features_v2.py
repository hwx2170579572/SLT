"""Route-constrained, merge-aware topology-temporal representation v2.

The v1 extractor remains unchanged.  This module exposes cumulative variants
for the pre-registered V2--V5 ablations:

* ``merge``: v1 distance query/fusion with the additional MERGE relation;
* ``query``: route/direction compatible sparse topology queries;
* ``gated``: near-zero topology residuals and separate route/topology pooling;
* ``soft``: identical forward architecture to ``gated``; its batch-statistic
  SoftBalancedSlots loss is enabled by the v2 SAC class.

At zero topology and goal residual scales, a gated extractor can reproduce the
corresponding v1 TemporalGraph backbone exactly after shared weights are copied.
"""

from __future__ import annotations

import math
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from envs.sumo.topology_graph_v2 import (
    CONFLICT,
    MERGE,
    NUM_RELATIONS,
    PaddedTopologyGraphV2,
    TOPOLOGY_ATTRIBUTE_DIM,
)

from .features import keras_initialize_linear
from .topo_temporal_features import (
    MaskedCrossAttention,
    RelationalTopologyLayer,
    StructuredLatent,
    TopoTemporalGraphExtractor,
    _masked_softmax,
)


V2_VARIANTS = ("merge", "query", "gated", "soft")


def soft_slot_balance_loss(
    latent: StructuredLatent,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return variance of log slot RMS and the three per-dimension RMS values."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    rms = torch.stack(
        [
            torch.sqrt(slot.square().mean() + epsilon)
            for slot in (latent.z_ego, latent.z_social, latent.z_route)
        ]
    )
    loss = torch.log(rms + epsilon).var(unbiased=False)
    return loss, rms


class QueryAwareMaskedCrossAttention(MaskedCrossAttention):
    """v1 attention extended with an optional mask for every query/key pair."""

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_valid: Tensor,
        *,
        additive_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query and key_value must be rank-three tensors")
        if key_valid.ndim == 2:
            if key_valid.shape != key_value.shape[:2]:
                raise ValueError(
                    f"key_valid {tuple(key_valid.shape)} does not match keys "
                    f"{tuple(key_value.shape[:2])}"
                )
            query_key_valid = key_valid[:, None, :].expand(
                -1, query.shape[1], -1
            )
        elif key_valid.ndim == 3:
            expected = (query.shape[0], query.shape[1], key_value.shape[1])
            if key_valid.shape != expected:
                raise ValueError(
                    f"query-specific key_valid must have shape {expected}, got "
                    f"{tuple(key_valid.shape)}"
                )
            query_key_valid = key_valid
        else:
            raise ValueError("key_valid must have rank two or three")

        q = self._heads(self.query(query))
        k = self._heads(self.key(key_value))
        v = self._heads(self.value(key_value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )
        if additive_bias is not None:
            expected = (query.shape[0], query.shape[1], key_value.shape[1])
            if additive_bias.shape != expected:
                raise ValueError(
                    f"additive_bias must have shape {expected}, got "
                    f"{tuple(additive_bias.shape)}"
                )
            scores = scores + additive_bias[:, None]
        mask = query_key_valid[:, None].expand_as(scores)
        weights = _masked_softmax(scores, mask)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        return self.output(attended), weights.mean(dim=1)


class RelationalTopologyEncoderV2(nn.Module):
    """Geometry/attribute encoder with six relation-specific transforms."""

    def __init__(self, feature_dim: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )
        self.attribute_fusion = nn.Sequential(
            nn.Linear(feature_dim + TOPOLOGY_ATTRIBUTE_DIM, feature_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            RelationalTopologyLayer(feature_dim, num_relations=NUM_RELATIONS)
            for _ in range(layers)
        )
        self.apply(keras_initialize_linear)

    def forward(
        self,
        lane_points: Tensor,
        lane_attrs: Tensor,
        node_mask: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        if lane_points.ndim != 4:
            raise ValueError("lane_points must have shape [batch, nodes, points, 2]")
        point_features = self.point_encoder(lane_points).amax(dim=2)
        attrs = lane_attrs
        if attrs.ndim == 2:
            attrs = attrs.unsqueeze(0).expand(lane_points.shape[0], -1, -1)
        features = self.attribute_fusion(torch.cat([point_features, attrs], dim=-1))
        features = features * node_mask[None, :, None].to(features.dtype)
        for layer in self.layers:
            features = layer(features, node_mask, edge_index, edge_type, edge_mask)
        return features


def _adjacency_from_relation(
    graph: PaddedTopologyGraphV2,
    relation: int,
) -> Tensor:
    adjacency = torch.zeros(
        graph.lane_points.shape[0],
        graph.lane_points.shape[0],
        dtype=torch.float32,
    )
    edge_mask = np.asarray(graph.edge_mask, dtype=bool)
    relation_mask = edge_mask & (np.asarray(graph.edge_type) == relation)
    if relation_mask.any():
        source = torch.as_tensor(graph.edge_index[0, relation_mask], dtype=torch.long)
        target = torch.as_tensor(graph.edge_index[1, relation_mask], dtype=torch.long)
        adjacency[source, target] = 1.0
    return adjacency


class TopoTemporalGraphExtractorV2(TopoTemporalGraphExtractor):
    """Cumulative v2 extractor with evidence-facing diagnostic state."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        topology_graph: PaddedTopologyGraphV2,
        variant: str = "soft",
        features_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 2,
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
        topology_sigma: float = 20.0,
        route_sigma: float = 20.0,
        vehicle_sigma: float = 20.0,
        topology_top_k: int = 8,
        reverse_heading_cosine_min: float = 0.0,
        route_bias_init: float = 1.0,
        heading_bias_init: float = 1.0,
        topology_layerscale_init: float = 1e-3,
        goal_layerscale_init: float = 1e-3,
    ) -> None:
        if variant not in V2_VARIANTS:
            raise ValueError(f"variant must be one of {V2_VARIANTS}, got {variant!r}")
        if topology_graph is None:
            raise ValueError("topology_graph is required for every v2 variant")
        if route_sigma <= 0.0:
            raise ValueError("route_sigma must be positive")
        if topology_top_k <= 0:
            raise ValueError("topology_top_k must be positive")
        if not -1.0 <= reverse_heading_cosine_min <= 1.0:
            raise ValueError("reverse_heading_cosine_min must lie in [-1, 1]")
        if route_bias_init <= 0.0 or heading_bias_init <= 0.0:
            raise ValueError("route and heading bias initial values must be positive")
        if topology_layerscale_init < 0.0 or goal_layerscale_init < 0.0:
            raise ValueError("LayerScale initial values cannot be negative")

        # Construct the exact v1 TemporalGraph backbone first.  Replace only
        # the static graph buffers and add v2 modules below.  This gives the
        # zero-gate equivalence test a literal shared architecture.
        super().__init__(
            observation_space,
            topology_graph=None,
            features_dim=features_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            random_augmentation=random_augmentation,
            carla_contract=carla_contract,
            use_topology=False,
            normalize_slots=False,
            topology_sigma=topology_sigma,
            vehicle_sigma=vehicle_sigma,
        )
        self.variant = variant
        self.use_topology = True
        self.use_route_query = variant in ("query", "gated", "soft")
        self.use_residual_gates = variant in ("gated", "soft")
        self.split_goal_pool = variant in ("gated", "soft")
        self.route_sigma = float(route_sigma)
        self.topology_top_k = int(topology_top_k)
        self.reverse_heading_cosine_min = float(reverse_heading_cosine_min)

        # Assigning to already registered buffer names keeps PyTorch/SB3 state
        # ownership and device movement intact.
        self.topology_lane_points = torch.as_tensor(
            np.array(topology_graph.lane_points, copy=True), dtype=torch.float32
        )
        self.topology_lane_attrs = torch.as_tensor(
            np.array(topology_graph.lane_attrs, copy=True), dtype=torch.float32
        )
        self.topology_node_mask = torch.as_tensor(
            np.array(topology_graph.node_mask, copy=True), dtype=torch.bool
        )
        self.topology_edge_index = torch.as_tensor(
            np.array(topology_graph.edge_index, copy=True), dtype=torch.long
        )
        self.topology_edge_type = torch.as_tensor(
            np.array(topology_graph.edge_type, copy=True), dtype=torch.long
        )
        self.topology_edge_mask = torch.as_tensor(
            np.array(topology_graph.edge_mask, copy=True), dtype=torch.bool
        )
        self.topology_conflict_adjacency = _adjacency_from_relation(
            topology_graph, CONFLICT
        )
        self.register_buffer(
            "topology_merge_adjacency",
            _adjacency_from_relation(topology_graph, MERGE),
        )
        merge_incident = (
            self.topology_merge_adjacency.sum(dim=0)
            + self.topology_merge_adjacency.sum(dim=1)
        ) > 0
        self.register_buffer("topology_merge_incident", merge_incident)

        self.topology_encoder = RelationalTopologyEncoderV2(hidden_dim, layers=2)
        self.topology_attention = QueryAwareMaskedCrossAttention(hidden_dim, num_heads)
        self.goal_topology_attention = (
            QueryAwareMaskedCrossAttention(hidden_dim, num_heads)
            if self.split_goal_pool
            else None
        )
        self.route_topology_norm = (
            nn.LayerNorm(hidden_dim) if self.split_goal_pool else None
        )

        if self.use_route_query:
            self.route_bias_raw = nn.Parameter(
                torch.tensor(self._inverse_softplus(route_bias_init), dtype=torch.float32)
            )
            self.heading_bias_raw = nn.Parameter(
                torch.tensor(
                    self._inverse_softplus(heading_bias_init), dtype=torch.float32
                )
            )
        else:
            self.register_parameter("route_bias_raw", None)
            self.register_parameter("heading_bias_raw", None)

        if self.use_residual_gates:
            self.topology_residual_scale = nn.Parameter(
                torch.tensor(float(topology_layerscale_init), dtype=torch.float32)
            )
            self.goal_residual_scale = nn.Parameter(
                torch.tensor(float(goal_layerscale_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "topology_residual_scale",
                torch.tensor(1.0, dtype=torch.float32),
            )
            self.register_buffer(
                "goal_residual_scale",
                torch.tensor(1.0, dtype=torch.float32),
            )

        self.last_route_squared_distance: Tensor | None = None
        self.last_route_compatible_mask: Tensor | None = None
        self.last_topology_key_mask: Tensor | None = None
        self.last_topology_fallback_mask: Tensor | None = None
        self._last_edge_diagnostics: dict[str, Tensor] = {}

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(float(value)))

    @property
    def route_bias_weight(self) -> Tensor:
        if self.route_bias_raw is None:
            return torch.zeros((), device=self.topology_lane_points.device)
        return functional.softplus(self.route_bias_raw)

    @property
    def heading_bias_weight(self) -> Tensor:
        if self.heading_bias_raw is None:
            return torch.zeros((), device=self.topology_lane_points.device)
        return functional.softplus(self.heading_bias_raw)

    @staticmethod
    def _unit_direction(delta: Tensor) -> Tensor:
        return delta / delta.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

    def _vehicle_directions(
        self,
        trajectories: Tensor,
        frame: Tensor,
        augmentation: Tensor,
    ) -> Tensor:
        heading = (
            trajectories[..., 2]
            - frame[:, None, None, 2]
            - augmentation[:, None, None]
        )
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        if self.carla_contract:
            direction = torch.stack([-direction[..., 0], direction[..., 1]], dim=-1)
        return direction

    def _route_geometry(
        self,
        rotated_map: Tensor,
        map_valid: Tensor,
        local_lane_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Actor/lane route distance, direction cosine, and availability."""

        batch = rotated_map.shape[0]
        node_count = local_lane_points.shape[1]
        route_points = rotated_map[..., :2].reshape(
            batch,
            self.actor_count,
            self.paths_per_actor,
            rotated_map.shape[2],
            2,
        )
        route_valid = map_valid.reshape(
            batch,
            self.actor_count,
            self.paths_per_actor,
            rotated_map.shape[2],
        )
        route_available = route_valid.any(dim=-1).any(dim=-1)

        # cdist avoids a much larger [B,N,M,lane_points,route_points,2]
        # broadcast while preserving the exact polyline-to-polyline minimum.
        lane_flat = local_lane_points.reshape(batch, -1, 2)
        lane_per_actor = lane_flat[:, None].expand(
            -1, self.actor_count, -1, -1
        ).reshape(batch * self.actor_count, lane_flat.shape[1], 2)
        route_flat = route_points.reshape(
            batch * self.actor_count, -1, 2
        )
        valid_flat = route_valid.reshape(batch * self.actor_count, -1)
        distances = torch.cdist(lane_per_actor.detach(), route_flat.detach())
        distances = distances.masked_fill(~valid_flat[:, None, :], float("inf"))
        minimum = distances.amin(dim=-1).square().reshape(
            batch,
            self.actor_count,
            node_count,
            local_lane_points.shape[2],
        ).amin(dim=-1)
        minimum = torch.where(
            route_available[..., None], minimum, torch.zeros_like(minimum)
        )

        lane_direction = self._unit_direction(
            local_lane_points[:, :, -1] - local_lane_points[:, :, 0]
        )
        path_valid = route_valid.any(dim=-1)
        last_index = route_valid.long().sum(dim=-1).sub(1).clamp(min=0)
        gather_index = last_index[..., None, None].expand(
            *last_index.shape, 1, 2
        )
        route_last = route_points.gather(3, gather_index).squeeze(3)
        route_first = route_points[..., 0, :]
        route_direction = self._unit_direction(route_last - route_first)
        path_cosine = torch.einsum(
            "bnpd,bmd->bnpm", route_direction, lane_direction
        )
        path_cosine = path_cosine.masked_fill(~path_valid[..., None], -1.0)
        route_cosine = path_cosine.amax(dim=2)
        return minimum, route_cosine, route_available

    def _sparse_key_mask(
        self,
        bias: Tensor,
        compatible: Tensor,
        vehicle_distance: Tensor,
        vehicle_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_count = bias.shape[-1]
        top_k = min(self.topology_top_k, node_count)
        masked_bias = bias.masked_fill(~compatible, -torch.inf)
        top_indices = torch.topk(masked_bias, k=top_k, dim=-1).indices
        selected = torch.zeros_like(compatible)
        selected.scatter_(-1, top_indices, True)
        selected &= compatible

        fallback = vehicle_valid & ~compatible.any(dim=-1)
        global_nodes = self.topology_node_mask.view(
            *((1,) * (vehicle_distance.ndim - 1)), -1
        )
        nearest = vehicle_distance.masked_fill(~global_nodes, torch.inf).argmin(dim=-1)
        nearest_mask = torch.zeros_like(compatible)
        nearest_mask.scatter_(-1, nearest[..., None], True)
        selected = torch.where(fallback[..., None], nearest_mask, selected)
        selected &= vehicle_valid[..., None]
        return selected, fallback

    def _query_topology_v2(
        self,
        vehicle_tokens: Tensor,
        vehicle_positions: Tensor,
        vehicle_directions: Tensor,
        vehicle_valid: Tensor,
        local_lane_points: Tensor,
        topology_tokens: Tensor,
        rotated_map: Tensor,
        map_valid: Tensor,
        *,
        compute_context: bool = True,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor, Tensor]:
        batch, actors, history, feature_dim = vehicle_tokens.shape
        squared_distance = (
            vehicle_positions[:, :, :, None, None, :]
            - local_lane_points[:, None, None, :, :, :]
        ).square().sum(dim=-1).amin(dim=-1)
        self.last_topology_squared_distance = squared_distance.detach()
        distance_bias = -squared_distance / (2.0 * self.topology_sigma**2)

        global_nodes = self.topology_node_mask[None, None, None, :].expand(
            batch, actors, history, -1
        )
        compatible = global_nodes
        fallback = torch.zeros_like(vehicle_valid)
        route_distance = torch.zeros(
            batch,
            actors,
            self.topology_node_mask.shape[0],
            dtype=vehicle_tokens.dtype,
            device=vehicle_tokens.device,
        )
        combined_bias = distance_bias
        if self.use_route_query:
            route_distance, route_cosine, route_available = self._route_geometry(
                rotated_map, map_valid, local_lane_points
            )
            lane_direction = self._unit_direction(
                local_lane_points[:, :, -1] - local_lane_points[:, :, 0]
            )
            vehicle_cosine = torch.einsum(
                "bnhd,bmd->bnhm", vehicle_directions, lane_direction
            )
            route_cosine = route_cosine[:, :, None, :].expand(-1, -1, history, -1)
            route_available_query = route_available[:, :, None, None]
            heading_cosine = torch.where(
                route_available_query,
                0.5 * (vehicle_cosine + route_cosine),
                vehicle_cosine,
            )
            combined_bias = (
                distance_bias
                - self.route_bias_weight
                * route_distance[:, :, None, :]
                / (2.0 * self.route_sigma**2)
                + self.heading_bias_weight * heading_cosine
            )
            compatible = global_nodes & (
                heading_cosine >= self.reverse_heading_cosine_min
            )
            key_mask, fallback = self._sparse_key_mask(
                combined_bias, compatible, squared_distance, vehicle_valid
            )
        else:
            key_mask = global_nodes & vehicle_valid[..., None]

        self.last_route_squared_distance = route_distance.detach()
        self.last_route_compatible_mask = compatible.detach()
        self.last_topology_key_mask = key_mask.detach()
        self.last_topology_fallback_mask = fallback.detach()
        if not compute_context:
            return None, None, key_mask, compatible, fallback

        assert self.topology_attention is not None
        query = vehicle_tokens.reshape(batch, actors * history, feature_dim)
        context, attention = self.topology_attention(
            query,
            topology_tokens,
            key_mask.reshape(batch, actors * history, -1),
            additive_bias=combined_bias.reshape(batch, actors * history, -1),
        )
        context = context.reshape(batch, actors, history, feature_dim)
        attention = attention.reshape(batch, actors, history, -1)
        valid = vehicle_valid[..., None].to(context.dtype)
        attention = attention * valid
        return context * valid, attention, key_mask, compatible, fallback

    def _vehicle_edge_features_v2(
        self,
        trajectories: Tensor,
        valid: Tensor,
        topology_attention: Tensor,
        *,
        topology_key_mask: Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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

        alpha = topology_attention.permute(0, 2, 1, 3)
        same_lane_score = torch.einsum("bhim,bhjm->bhij", alpha, alpha)
        risk_adjacency = torch.maximum(
            self.topology_conflict_adjacency,
            self.topology_merge_adjacency,
        )
        conflict_or_merge_score = torch.einsum(
            "bhim,mn,bhjn->bhij", alpha, risk_adjacency, alpha
        )
        merge_score = torch.einsum(
            "bhim,mn,bhjn->bhij",
            alpha,
            self.topology_merge_adjacency,
            alpha,
        )
        conflict_score = None
        if collect_diagnostics:
            conflict_score = torch.einsum(
                "bhim,mn,bhjn->bhij",
                alpha,
                self.topology_conflict_adjacency,
                alpha,
            )
        topology_scale = self.topology_residual_scale
        same_lane = same_lane_score * topology_scale
        conflict_or_merge = conflict_or_merge_score * topology_scale
        edge_features = torch.cat(
            [
                delta_position,
                delta_velocity,
                torch.cos(heading_delta)[..., None],
                ttc[..., None],
                same_lane[..., None],
                conflict_or_merge[..., None],
            ],
            dim=-1,
        )
        edge_weights = torch.exp(
            -distance_squared / (2.0 * self.vehicle_sigma**2)
        )
        node_valid = valid.permute(0, 2, 1).bool()
        pair_valid = node_valid.unsqueeze(3) & node_valid.unsqueeze(2)
        diagonal = torch.eye(
            self.actor_count, dtype=torch.bool, device=trajectories.device
        )[None, None]
        pair_valid = pair_valid & ~diagonal
        edge_weights = edge_weights * pair_valid.to(edge_weights.dtype)
        merge_values = merge_score[pair_valid]
        mean_merge_score = (
            merge_values.mean()
            if merge_values.numel()
            else torch.zeros((), device=trajectories.device)
        )
        self._last_edge_diagnostics = {}
        if collect_diagnostics:
            pair_weights = pair_valid.to(merge_score.dtype)
            valid_pair_count = pair_weights.sum()
            candidate_pair_count = torch.tensor(
                pair_valid.shape[0]
                * pair_valid.shape[1]
                * pair_valid.shape[2]
                * max(pair_valid.shape[3] - 1, 0),
                device=trajectories.device,
                dtype=merge_score.dtype,
            )

            def pair_mean(values: Tensor) -> Tensor:
                return (values * pair_weights).sum() / valid_pair_count.clamp_min(1.0)

            assert conflict_score is not None
            conflict_edge_count = self.topology_conflict_adjacency.sum()
            merge_edge_count = self.topology_merge_adjacency.sum()
            edge_diagnostics = {
                "relation_pair_valid_count": valid_pair_count.detach(),
                "relation_pair_candidate_count": candidate_pair_count,
                "relation_pair_valid_fraction": (
                    valid_pair_count / candidate_pair_count.clamp_min(1.0)
                ).detach(),
                "relation_pair_mask_has_valid": (
                    valid_pair_count > 0
                ).to(merge_score.dtype),
                "topology_conflict_directed_edge_count": conflict_edge_count.detach(),
                "topology_merge_directed_edge_count": merge_edge_count.detach(),
                "topology_conflict_relation_present": (conflict_edge_count > 0).to(
                    merge_score.dtype
                ),
                "topology_merge_relation_present": (merge_edge_count > 0).to(
                    merge_score.dtype
                ),
                "same_lane_attention_pair_mean": pair_mean(same_lane_score).detach(),
                "conflict_attention_pair_mean": pair_mean(conflict_score).detach(),
                "merge_attention_pair_mean": pair_mean(merge_score).detach(),
                "conflict_or_merge_attention_pair_mean": pair_mean(
                    conflict_or_merge_score
                ).detach(),
                "same_lane_edge_feature_mean": pair_mean(same_lane).detach(),
                "conflict_or_merge_edge_feature_mean": pair_mean(
                    conflict_or_merge
                ).detach(),
                "merge_pair_score": mean_merge_score.detach(),
            }
            # Compare learned attention-induced relation scores with a uniform
            # distribution over the exact key support used by each query. The
            # support also masks the actor pairs, so attention and null baselines
            # share both the candidate set and pair denominator.
            relation_uniform_metrics = self._uniform_relation_support_diagnostics(
                same_lane_score=same_lane_score,
                conflict_score=conflict_score,
                merge_score=merge_score,
                conflict_or_merge_score=conflict_or_merge_score,
                pair_valid=pair_valid,
                topology_key_mask=topology_key_mask,
                reference=merge_score,
            )
            edge_diagnostics.update(relation_uniform_metrics)
            self._last_edge_diagnostics = edge_diagnostics
        return edge_features, edge_weights, pair_valid, node_valid, mean_merge_score

    def _uniform_relation_support_diagnostics(
        self,
        *,
        same_lane_score: Tensor,
        conflict_score: Tensor,
        merge_score: Tensor,
        conflict_or_merge_score: Tensor,
        pair_valid: Tensor,
        topology_key_mask: Tensor | None,
        reference: Tensor,
    ) -> dict[str, Tensor]:
        """Compare relation attention mass to a uniform null on identical support.

        ``topology_key_mask`` is the post-top-k/fallback support actually passed
        to lane attention, with shape [B, actors, history, lanes]. The uniform
        baseline is formed independently per actor/history query on that mask,
        and both observed and null relation masses use the same valid actor-pair
        mask. These are descriptive selectivity statistics, not causal effects.
        """
        zero = reference.new_zeros(())
        result: dict[str, Tensor] = {
            "relation_uniform_candidate_query_count": zero,
            "relation_uniform_candidate_count_mean": zero,
            "relation_uniform_support_pair_count": zero,
            "relation_uniform_support_valid": zero,
        }
        relation_names = (
            "same_lane",
            "conflict",
            "merge",
            "conflict_or_merge",
        )
        if topology_key_mask is None:
            for name in relation_names:
                result[f"{name}_attention_matched_support_mean"] = zero
                result[f"{name}_uniform_support_mean"] = zero
                result[f"{name}_attention_minus_uniform_support"] = zero
            return result

        candidates = topology_key_mask.permute(0, 2, 1, 3).to(reference.dtype)
        candidate_counts = candidates.sum(dim=-1)
        query_valid = candidate_counts > 0
        candidate_query_count = query_valid.to(reference.dtype).sum()
        uniform = candidates / candidate_counts.clamp_min(1.0)[..., None]
        actor_has_candidates = query_valid
        supported_pairs = (
            pair_valid
            & actor_has_candidates.unsqueeze(-1)
            & actor_has_candidates.unsqueeze(-2)
        )
        pair_weights = supported_pairs.to(reference.dtype)
        support_pair_count = pair_weights.sum()
        support_valid = (support_pair_count > 0).to(reference.dtype)

        def matched_mean(values: Tensor) -> Tensor:
            return (values * pair_weights).sum() / support_pair_count.clamp_min(1.0)

        same_uniform = torch.einsum("bhim,bhjm->bhij", uniform, uniform)
        conflict_uniform = torch.einsum(
            "bhim,mn,bhjn->bhij",
            uniform,
            self.topology_conflict_adjacency,
            uniform,
        )
        merge_uniform = torch.einsum(
            "bhim,mn,bhjn->bhij",
            uniform,
            self.topology_merge_adjacency,
            uniform,
        )
        risk_adjacency = torch.maximum(
            self.topology_conflict_adjacency,
            self.topology_merge_adjacency,
        )
        conflict_or_merge_uniform = torch.einsum(
            "bhim,mn,bhjn->bhij", uniform, risk_adjacency, uniform
        )
        observed = {
            "same_lane": same_lane_score,
            "conflict": conflict_score,
            "merge": merge_score,
            "conflict_or_merge": conflict_or_merge_score,
        }
        baselines = {
            "same_lane": same_uniform,
            "conflict": conflict_uniform,
            "merge": merge_uniform,
            "conflict_or_merge": conflict_or_merge_uniform,
        }
        result.update(
            {
                "relation_uniform_candidate_query_count": candidate_query_count,
                "relation_uniform_candidate_count_mean": (
                    (candidate_counts * query_valid.to(reference.dtype)).sum()
                    / candidate_query_count.clamp_min(1.0)
                ),
                "relation_uniform_support_pair_count": support_pair_count,
                "relation_uniform_support_valid": support_valid,
            }
        )
        for name in relation_names:
            observed_mean = matched_mean(observed[name])
            uniform_mean = matched_mean(baselines[name])
            result[f"{name}_attention_matched_support_mean"] = observed_mean
            result[f"{name}_uniform_support_mean"] = uniform_mean
            result[f"{name}_attention_minus_uniform_support"] = (
                observed_mean - uniform_mean
            )
        return result

    def _last_query_mask(self, key_mask: Tensor, valid: Tensor) -> Tensor:
        last_index = valid.long().sum(dim=-1).sub(1).clamp(min=0)
        gather_index = last_index[..., None, None].expand(
            *last_index.shape, 1, key_mask.shape[-1]
        )
        return key_mask.gather(2, gather_index).squeeze(2)

    def forward_tokens(self, observations: Mapping[str, Tensor]) -> StructuredLatent:
        trajectories = observations["trajectory"].float()
        map_state = observations["map"].float()
        from .features import _nonzero_mask

        trajectory_valid = _nonzero_mask(trajectories)
        map_valid = _nonzero_mask(map_state)
        rotated, frame, augmentation = self._rotate_trajectories(
            trajectories, trajectory_valid
        )
        rotated_map = self._rotate_map(map_state, map_valid, frame, augmentation)
        state_tokens = self.state_encoder(rotated) * trajectory_valid[..., None]
        current_state = self._last_valid(state_tokens, trajectory_valid)
        route_tokens, path_valid, route_context = self._encode_routes(
            rotated_map, map_valid, current_state
        )
        intent_tokens = self.route_norm(
            state_tokens + route_context[:, :, None, :]
        ) * trajectory_valid[..., None]

        local_lane_points = self._rotate_topology(frame, augmentation)
        assert self.topology_encoder is not None
        topology_tokens = self.topology_encoder(
            local_lane_points,
            self.topology_lane_attrs,
            self.topology_node_mask,
            self.topology_edge_index,
            self.topology_edge_type,
            self.topology_edge_mask,
        )
        vehicle_directions = self._vehicle_directions(
            trajectories, frame, augmentation
        )
        (
            topology_context,
            topology_attention,
            topology_key_mask,
            compatible_mask,
            fallback_mask,
        ) = self._query_topology_v2(
            intent_tokens,
            rotated[..., :2],
            vehicle_directions,
            trajectory_valid,
            local_lane_points,
            topology_tokens,
            rotated_map,
            map_valid,
        )

        topology_scale = self.topology_residual_scale
        graph_input = self.topology_norm(
            intent_tokens + topology_scale * topology_context
        )
        graph_input = graph_input * trajectory_valid[..., None]
        (
            edge_features,
            edge_weights,
            pair_valid,
            node_valid,
            mean_merge_pair_score,
        ) = self._vehicle_edge_features_v2(
            rotated, trajectory_valid, topology_attention
        )
        graph_features = graph_input.permute(0, 2, 1, 3)
        for layer in self.vehicle_layers:
            graph_features = layer(
                graph_features,
                edge_features,
                edge_weights,
                pair_valid,
                node_valid,
            )
        graph_sequence = graph_features.permute(0, 2, 1, 3)
        actor_features = self.temporal_encoder(graph_sequence, trajectory_valid)
        actor_valid = trajectory_valid.any(dim=-1)
        actor_valid = actor_valid.clone()
        actor_valid[:, 0] = True
        ego = actor_features[:, :1]
        social_context, _ = self.social_attention(ego, actor_features, actor_valid)

        ego_route_tokens = route_tokens[:, 0]
        ego_route_valid = path_valid[:, 0]
        last_key_mask = self._last_query_mask(topology_key_mask, trajectory_valid)
        ego_topology_valid = last_key_mask[:, 0]
        if self.split_goal_pool:
            route_base, _ = self.goal_attention(
                ego, ego_route_tokens, ego_route_valid
            )
            assert self.goal_topology_attention is not None
            assert self.route_topology_norm is not None
            topology_goal, _ = self.goal_topology_attention(
                route_base,
                topology_tokens,
                ego_topology_valid,
            )
            # Delta-LN preserves the exact TemporalGraph output at a zero gate
            # while still constraining the learned topology correction.
            base_normalized = self.route_topology_norm(route_base)
            fused_normalized = self.route_topology_norm(
                route_base + self.goal_residual_scale * topology_goal
            )
            route_context_final = route_base + fused_normalized - base_normalized
        else:
            goal_tokens = torch.cat([ego_route_tokens, topology_tokens], dim=1)
            topology_goal_valid = (
                ego_topology_valid
                if self.use_route_query
                else self.topology_node_mask[None].expand(trajectories.shape[0], -1)
            )
            goal_valid = torch.cat(
                [ego_route_valid, topology_goal_valid], dim=1
            )
            route_context_final, _ = self.goal_attention(
                ego, goal_tokens, goal_valid
            )

        latent = StructuredLatent(
            z_ego=self.ego_projection(ego.squeeze(1)),
            z_social=self.social_projection(social_context.squeeze(1)),
            z_route=self.route_projection(route_context_final.squeeze(1)),
        )

        attention_entropy_per_query = -(
            topology_attention.clamp_min(1e-12).log() * topology_attention
        ).sum(dim=-1)
        query_weights = trajectory_valid.to(attention_entropy_per_query.dtype)
        query_count = query_weights.sum().clamp_min(1.0)
        attention_entropy = (
            attention_entropy_per_query * query_weights
        ).sum() / query_count
        effective_lanes = (
            attention_entropy_per_query.exp() * query_weights
        ).sum() / query_count
        compatible_mass = (
            topology_attention * compatible_mask.to(topology_attention.dtype)
        ).sum(dim=-1)
        route_compatible_mass = (compatible_mass * query_weights).sum() / query_count
        merge_mass_per_query = (
            topology_attention
            * self.topology_merge_incident[None, None, None].to(
                topology_attention.dtype
            )
        ).sum(dim=-1)
        merge_attention_mass = (
            merge_mass_per_query * query_weights
        ).sum() / query_count
        global_node_count = self.topology_node_mask.sum().clamp_min(1).to(
            topology_attention.dtype
        )
        compatible_count = compatible_mask.to(topology_attention.dtype).sum(dim=-1)
        reverse_mask_rate = (
            (1.0 - compatible_count / global_node_count) * query_weights
        ).sum() / query_count
        fallback_rate = (
            fallback_mask.to(topology_attention.dtype) * query_weights
        ).sum() / query_count
        valid_edge_weights = edge_weights[pair_valid]
        graph_mean_edge_weight = (
            valid_edge_weights.mean()
            if valid_edge_weights.numel()
            else torch.zeros((), device=trajectories.device)
        )
        balance_loss, slot_rms = soft_slot_balance_loss(latent)
        slot_stds = torch.stack(
            [
                latent.z_ego.std(dim=0, unbiased=False).mean(),
                latent.z_social.std(dim=0, unbiased=False).mean(),
                latent.z_route.std(dim=0, unbiased=False).mean(),
            ]
        )
        self.last_topology_attention = topology_attention.detach()
        self._diagnostics = {
            "topology_attention_entropy": attention_entropy,
            "topology_effective_lanes": effective_lanes,
            "route_compatible_attention_mass": route_compatible_mass,
            "merge_attention_mass": merge_attention_mass,
            "merge_pair_score": mean_merge_pair_score,
            "reverse_lane_mask_rate": reverse_mask_rate,
            "topology_fallback_rate": fallback_rate,
            "topology_residual_scale": self.topology_residual_scale,
            "goal_residual_scale": self.goal_residual_scale,
            "route_bias_weight": self.route_bias_weight,
            "heading_bias_weight": self.heading_bias_weight,
            "soft_slot_balance_loss": balance_loss,
            "ego_slot_rms": slot_rms[0],
            "social_slot_rms": slot_rms[1],
            "route_slot_rms": slot_rms[2],
            "slot_rms_ratio": slot_rms.max() / slot_rms.min().clamp_min(1e-8),
            "latent_std": latent.tensor.std(dim=0, unbiased=False).mean(),
            "ego_latent_std": slot_stds[0],
            "social_latent_std": slot_stds[1],
            "route_latent_std": slot_stds[2],
            "slot_scale_ratio": slot_stds.max() / slot_stds.min().clamp_min(1e-8),
            "graph_mean_edge_weight": graph_mean_edge_weight,
        }
        return latent


__all__ = [
    "QueryAwareMaskedCrossAttention",
    "RelationalTopologyEncoderV2",
    "TopoTemporalGraphExtractorV2",
    "V2_VARIANTS",
    "soft_slot_balance_loss",
]
