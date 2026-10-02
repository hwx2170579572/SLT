"""Topology-aware temporal vehicle graph encoder for SUMO/SB3.

The implementation keeps every vehicle history frame until after spatial
interaction, queries a static lane graph with a soft geometric prior, and
returns an auditable 32/64/32 structured latent while preserving SB3's
standard 128-dimensional feature interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.sumo.topology_graph import (
    CONFLICT,
    LANE_SAMPLE_POINTS,
    NUM_RELATIONS,
    PaddedTopologyGraph,
    TOPOLOGY_ATTRIBUTE_DIM,
)

from .features import (
    HierarchicalSceneExtractor,
    MapPolylineEncoder,
    _nonzero_mask,
    keras_initialize_linear,
)


@dataclass(frozen=True)
class StructuredLatent:
    """The three semantic slots consumed by Graph-SLT."""

    z_ego: Tensor
    z_social: Tensor
    z_route: Tensor

    @property
    def tensor(self) -> Tensor:
        return torch.cat([self.z_ego, self.z_social, self.z_route], dim=-1)

    @classmethod
    def from_tensor(cls, value: Tensor) -> "StructuredLatent":
        if value.shape[-1] != 128:
            raise ValueError(
                f"Structured latent must have 128 features, got {value.shape[-1]}"
            )
        return cls(value[..., :32], value[..., 32:96], value[..., 96:128])


def _masked_softmax(scores: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """Softmax with an exact all-invalid -> zero contract."""

    mask = valid.bool()
    masked_scores = scores.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked_scores, dim=dim) * mask.to(scores.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return weights / denominator


class MaskedCrossAttention(nn.Module):
    """Small batch-first multi-head attention with safe padding semantics."""

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.feature_dim // self.num_heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.apply(keras_initialize_linear)

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

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
        if key_valid.shape != key_value.shape[:2]:
            raise ValueError(
                f"key_valid {tuple(key_valid.shape)} does not match keys "
                f"{tuple(key_value.shape[:2])}"
            )
        q = self._heads(self.query(query))
        k = self._heads(self.key(key_value))
        v = self._heads(self.value(key_value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if additive_bias is not None:
            if additive_bias.shape != (
                query.shape[0],
                query.shape[1],
                key_value.shape[1],
            ):
                raise ValueError(
                    "additive_bias must have shape [batch, query, key], got "
                    f"{tuple(additive_bias.shape)}"
                )
            scores = scores + additive_bias[:, None]
        mask = key_valid[:, None, None, :].expand_as(scores)
        weights = _masked_softmax(scores, mask)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        return self.output(attended), weights.mean(dim=1)


class RelationalTopologyLayer(nn.Module):
    """One residual relation-specific lane GCN layer using ``index_add_``."""

    def __init__(self, feature_dim: int, num_relations: int = NUM_RELATIONS) -> None:
        super().__init__()
        self.self_transform = nn.Linear(feature_dim, feature_dim, bias=False)
        self.relation_transforms = nn.ModuleList(
            nn.Linear(feature_dim, feature_dim, bias=False)
            for _ in range(num_relations)
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.apply(keras_initialize_linear)

    def forward(
        self,
        features: Tensor,
        node_mask: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        aggregate = torch.zeros_like(features)
        valid_edges = edge_mask.bool()
        for relation, transform in enumerate(self.relation_transforms):
            relation_mask = valid_edges & (edge_type == relation)
            # Do not branch through ``bool(tensor.any())`` here.  On CUDA that
            # scalar conversion synchronizes the host for every relation in
            # every extractor forward.  Empty index tensors are valid inputs
            # to ``index_select``/``index_add_`` and produce the exact same
            # zero contribution without a device synchronization.
            source = edge_index[0, relation_mask].long()
            target = edge_index[1, relation_mask].long()
            messages = transform(features.index_select(1, source))
            relation_sum = torch.zeros_like(features)
            relation_sum.index_add_(1, target, messages)
            counts = torch.zeros(
                features.shape[1], dtype=features.dtype, device=features.device
            )
            counts.index_add_(
                0,
                target,
                torch.ones(target.shape[0], dtype=features.dtype, device=features.device),
            )
            aggregate = aggregate + relation_sum / counts.clamp_min(1.0)[None, :, None]
        output = self.norm(features + self.self_transform(features) + aggregate)
        return output * node_mask[None, :, None].to(output.dtype)


class RelationalTopologyEncoder(nn.Module):
    """Encode resampled lane geometry, attributes, and five edge relations."""

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
            RelationalTopologyLayer(feature_dim) for _ in range(layers)
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


class VehicleGraphLayer(nn.Module):
    """Residual, distance-weighted message passing over six vehicle slots."""

    def __init__(self, feature_dim: int, edge_dim: int = 8) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(feature_dim + edge_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
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
        # [B,H,N,D] -> [B,H,target,source,D]
        actor_count = features.shape[2]
        senders = features.unsqueeze(2).expand(-1, -1, actor_count, -1, -1)
        messages = self.message(torch.cat([senders, edge_features], dim=-1))
        weights = edge_weights * pair_valid.to(edge_weights.dtype)
        aggregate = (messages * weights[..., None]).sum(dim=3)
        denominator = weights.sum(dim=3, keepdim=True).clamp_min(1e-6)
        output = self.norm(features + aggregate / denominator)
        return output * node_valid[..., None].to(output.dtype)


class TemporalInteractionEncoder(nn.Module):
    """Temporal self-attention after per-frame vehicle interaction."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        history_steps: int,
    ) -> None:
        super().__init__()
        self.lag_embedding = nn.Parameter(torch.empty(history_steps, feature_dim))
        nn.init.normal_(self.lag_embedding, std=0.02)
        self.self_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.pool_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.pool_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.ffn.apply(keras_initialize_linear)

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        batch, actors, history, feature_dim = features.shape
        flat = features.reshape(batch * actors, history, feature_dim)
        flat_valid = valid.reshape(batch * actors, history).bool()
        actor_valid = flat_valid.any(dim=1)
        safe_valid = flat_valid.clone()
        safe_valid[~actor_valid, 0] = True
        embedded = (flat + self.lag_embedding[None, :history]) * flat_valid[..., None]
        attended, _ = self.self_attention(
            embedded,
            embedded,
            embedded,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        encoded = self.norm1(embedded + attended) * flat_valid[..., None]
        encoded = self.norm2(encoded + self.ffn(encoded)) * flat_valid[..., None]
        current_index = flat_valid.long().sum(dim=1).sub(1).clamp(min=0)
        batch_index = torch.arange(flat.shape[0], device=flat.device)
        query = encoded[batch_index, current_index].unsqueeze(1)
        pooled, _ = self.pool_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        output = self.pool_norm(query + pooled).squeeze(1)
        output = output * actor_valid[:, None].to(output.dtype)
        return output.reshape(batch, actors, feature_dim)


def _empty_topology_graph() -> PaddedTopologyGraph:
    return PaddedTopologyGraph(
        lane_points=np.zeros((1, LANE_SAMPLE_POINTS, 2), dtype=np.float32),
        lane_attrs=np.zeros((1, TOPOLOGY_ATTRIBUTE_DIM), dtype=np.float32),
        node_mask=np.zeros(1, dtype=bool),
        edge_index=np.zeros((2, 1), dtype=np.int64),
        edge_type=np.zeros(1, dtype=np.int64),
        edge_mask=np.zeros(1, dtype=bool),
    )


class TopoTemporalGraphExtractor(BaseFeaturesExtractor):
    """Topology query -> temporal vehicle GCN -> structured 128-D latent."""

    # Reuse the exact baseline coordinate/mask semantics without constructing
    # the baseline's otherwise-unused attention hierarchy.
    _current_ego_frame = HierarchicalSceneExtractor._current_ego_frame
    _rotation_angle = HierarchicalSceneExtractor._rotation_angle
    _rotate_trajectories = HierarchicalSceneExtractor._rotate_trajectories
    _rotate_map = HierarchicalSceneExtractor._rotate_map
    set_source_augmentation = HierarchicalSceneExtractor.set_source_augmentation

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        topology_graph: PaddedTopologyGraph | None = None,
        features_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 2,
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
        use_topology: bool = True,
        normalize_slots: bool = False,
        topology_sigma: float = 20.0,
        vehicle_sigma: float = 20.0,
    ) -> None:
        if features_dim != 128 or hidden_dim != 128:
            raise ValueError("The L1 contract fixes hidden_dim=features_dim=128")
        super().__init__(observation_space, features_dim=features_dim)
        trajectory_space = observation_space.spaces["trajectory"]
        map_space = observation_space.spaces["map"]
        if len(trajectory_space.shape) != 3 or trajectory_space.shape[-1] != 5:
            raise ValueError(f"Invalid trajectory shape: {trajectory_space.shape}")
        if len(map_space.shape) != 3 or map_space.shape[-1] not in (2, 5):
            raise ValueError(f"Invalid map shape: {map_space.shape}")
        self.actor_count = int(trajectory_space.shape[0])
        self.history_steps = int(trajectory_space.shape[1])
        self.map_dim = int(map_space.shape[-1])
        if int(map_space.shape[0]) % self.actor_count:
            raise ValueError("Map path count must be divisible by actor count")
        self.paths_per_actor = int(map_space.shape[0] // self.actor_count)
        self.feature_dim = int(features_dim)
        self.random_augmentation = bool(random_augmentation)
        self.source_augmentation_enabled = True
        self.carla_contract = self.map_dim == 2 if carla_contract is None else bool(carla_contract)
        self.use_topology = bool(use_topology)
        self.normalize_slots = bool(normalize_slots)
        self.topology_sigma = float(topology_sigma)
        self.vehicle_sigma = float(vehicle_sigma)
        if self.topology_sigma <= 0.0 or self.vehicle_sigma <= 0.0:
            raise ValueError("topology_sigma and vehicle_sigma must be positive")
        if self.use_topology and topology_graph is None:
            raise ValueError("topology_graph is required when use_topology=True")
        graph = topology_graph if topology_graph is not None else _empty_topology_graph()

        self.register_buffer(
            "topology_lane_points",
            torch.as_tensor(np.array(graph.lane_points, copy=True), dtype=torch.float32),
        )
        self.register_buffer(
            "topology_lane_attrs",
            torch.as_tensor(np.array(graph.lane_attrs, copy=True), dtype=torch.float32),
        )
        self.register_buffer(
            "topology_node_mask",
            torch.as_tensor(np.array(graph.node_mask, copy=True), dtype=torch.bool),
        )
        self.register_buffer(
            "topology_edge_index",
            torch.as_tensor(np.array(graph.edge_index, copy=True), dtype=torch.long),
        )
        self.register_buffer(
            "topology_edge_type",
            torch.as_tensor(np.array(graph.edge_type, copy=True), dtype=torch.long),
        )
        self.register_buffer(
            "topology_edge_mask",
            torch.as_tensor(np.array(graph.edge_mask, copy=True), dtype=torch.bool),
        )
        conflict_adjacency = torch.zeros(
            graph.lane_points.shape[0], graph.lane_points.shape[0], dtype=torch.float32
        )
        conflict_edges = self.topology_edge_mask & (self.topology_edge_type == CONFLICT)
        if bool(conflict_edges.any()):
            source = self.topology_edge_index[0, conflict_edges]
            target = self.topology_edge_index[1, conflict_edges]
            conflict_adjacency[source, target] = 1.0
        self.register_buffer("topology_conflict_adjacency", conflict_adjacency)

        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.map_encoder = MapPolylineEncoder(self.map_dim, hidden_dim, num_heads)
        self.route_attention = MaskedCrossAttention(hidden_dim, num_heads)
        self.route_norm = nn.LayerNorm(hidden_dim)
        self.topology_encoder = (
            RelationalTopologyEncoder(hidden_dim, layers=2) if self.use_topology else None
        )
        self.topology_attention = (
            MaskedCrossAttention(hidden_dim, num_heads) if self.use_topology else None
        )
        self.topology_norm = nn.LayerNorm(hidden_dim)
        self.vehicle_layers = nn.ModuleList(
            VehicleGraphLayer(hidden_dim) for _ in range(2)
        )
        self.temporal_encoder = TemporalInteractionEncoder(
            hidden_dim, num_heads, self.history_steps
        )
        self.social_attention = MaskedCrossAttention(hidden_dim, num_heads)
        self.goal_attention = MaskedCrossAttention(hidden_dim, num_heads)
        self.ego_projection = nn.Linear(hidden_dim, 32)
        self.social_projection = nn.Linear(hidden_dim, 64)
        self.route_projection = nn.Linear(hidden_dim, 32)
        # The diagnostic variant uses non-affine normalisation so a learned
        # scale cannot recreate the late route-slot dominance it is designed
        # to test.  The original L1 method remains byte-compatible by default.
        self.slot_norms = nn.ModuleDict(
            {
                "ego": nn.LayerNorm(32, elementwise_affine=False),
                "social": nn.LayerNorm(64, elementwise_affine=False),
                "route": nn.LayerNorm(32, elementwise_affine=False),
            }
        )
        self.state_encoder.apply(keras_initialize_linear)
        self.ego_projection.apply(keras_initialize_linear)
        self.social_projection.apply(keras_initialize_linear)
        self.route_projection.apply(keras_initialize_linear)

        self.last_topology_attention: Tensor | None = None
        self.last_topology_squared_distance: Tensor | None = None
        self._diagnostics: dict[str, Tensor] = {}

    def _rotate_topology(self, frame: Tensor, augmentation: Tensor) -> Tensor:
        yaw = frame[:, 2] + augmentation
        cosine = torch.cos(yaw)[:, None, None]
        sine = torch.sin(yaw)[:, None, None]
        points = self.topology_lane_points[None].expand(frame.shape[0], -1, -1, -1)
        x = points[..., 0] - frame[:, None, None, 0]
        y = points[..., 1] - frame[:, None, None, 1]
        rotated_x = cosine * x + sine * y
        rotated_y = -sine * x + cosine * y
        if self.carla_contract:
            rotated_x = -rotated_x
        output = torch.stack([rotated_x, rotated_y], dim=-1)
        return output * self.topology_node_mask[None, :, None, None].to(output.dtype)

    @staticmethod
    def _last_valid(features: Tensor, valid: Tensor) -> Tensor:
        last_index = valid.long().sum(dim=-1).sub(1).clamp(min=0)
        gather_index = last_index[..., None, None].expand(
            *last_index.shape, 1, features.shape[-1]
        )
        return features.gather(2, gather_index).squeeze(2)

    def _encode_routes(
        self,
        rotated_map: Tensor,
        map_valid: Tensor,
        current_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = rotated_map.shape[0]
        route_tokens = self.map_encoder(rotated_map, map_valid).reshape(
            batch, self.actor_count, self.paths_per_actor, self.feature_dim
        )
        path_valid = map_valid[:, :, 0].reshape(
            batch, self.actor_count, self.paths_per_actor
        )
        route_tokens = route_tokens * path_valid[..., None].to(route_tokens.dtype)
        query = current_state.reshape(batch * self.actor_count, 1, self.feature_dim)
        keys = route_tokens.reshape(
            batch * self.actor_count, self.paths_per_actor, self.feature_dim
        )
        key_valid = path_valid.reshape(batch * self.actor_count, self.paths_per_actor)
        context, _ = self.route_attention(query, keys, key_valid)
        context = context.reshape(batch, self.actor_count, self.feature_dim)
        return route_tokens, path_valid, context

    def _query_topology(
        self,
        vehicle_tokens: Tensor,
        vehicle_positions: Tensor,
        vehicle_valid: Tensor,
        local_lane_points: Tensor,
        topology_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, actors, history, feature_dim = vehicle_tokens.shape
        if not self.use_topology:
            self.last_topology_squared_distance = None
            return (
                torch.zeros_like(vehicle_tokens),
                torch.zeros(
                    batch,
                    actors,
                    history,
                    self.topology_node_mask.shape[0],
                    dtype=vehicle_tokens.dtype,
                    device=vehicle_tokens.device,
                ),
            )
        assert self.topology_attention is not None
        squared_distance = (
            vehicle_positions[:, :, :, None, None, :]
            - local_lane_points[:, None, None, :, :, :]
        ).square().sum(dim=-1).amin(dim=-1)
        self.last_topology_squared_distance = squared_distance.detach()
        distance_bias = -squared_distance / (2.0 * self.topology_sigma**2)
        query = vehicle_tokens.reshape(batch, actors * history, feature_dim)
        bias = distance_bias.reshape(batch, actors * history, -1)
        context, attention = self.topology_attention(
            query,
            topology_tokens,
            self.topology_node_mask[None].expand(batch, -1),
            additive_bias=bias,
        )
        context = context.reshape(batch, actors, history, feature_dim)
        attention = attention.reshape(batch, actors, history, -1)
        valid = vehicle_valid[..., None].to(context.dtype)
        return context * valid, attention * valid

    def _vehicle_edge_features(
        self,
        trajectories: Tensor,
        valid: Tensor,
        topology_attention: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Use [B,H,N,*] for explicit target/source pair construction.
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

        if self.use_topology:
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
        return edge_features, edge_weights, pair_valid, node_valid

    def diagnostic_values(self) -> dict[str, Tensor]:
        return {key: value.detach() for key, value in self._diagnostics.items()}

    def forward_tokens(self, observations: Mapping[str, Tensor]) -> StructuredLatent:
        trajectories = observations["trajectory"].float()
        map_state = observations["map"].float()
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
        if self.topology_encoder is not None:
            topology_tokens = self.topology_encoder(
                local_lane_points,
                self.topology_lane_attrs,
                self.topology_node_mask,
                self.topology_edge_index,
                self.topology_edge_type,
                self.topology_edge_mask,
            )
        else:
            topology_tokens = torch.zeros(
                trajectories.shape[0],
                self.topology_node_mask.shape[0],
                self.feature_dim,
                dtype=trajectories.dtype,
                device=trajectories.device,
            )
        topology_context, topology_attention = self._query_topology(
            intent_tokens,
            rotated[..., :2],
            trajectory_valid,
            local_lane_points,
            topology_tokens,
        )
        graph_input = self.topology_norm(intent_tokens + topology_context)
        graph_input = graph_input * trajectory_valid[..., None]
        edge_features, edge_weights, pair_valid, node_valid = self._vehicle_edge_features(
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
        if self.use_topology:
            goal_tokens = torch.cat([ego_route_tokens, topology_tokens], dim=1)
            goal_valid = torch.cat(
                [
                    ego_route_valid,
                    self.topology_node_mask[None].expand(trajectories.shape[0], -1),
                ],
                dim=1,
            )
        else:
            goal_tokens = ego_route_tokens
            goal_valid = ego_route_valid
        route_context_final, _ = self.goal_attention(ego, goal_tokens, goal_valid)
        z_ego = self.ego_projection(ego.squeeze(1))
        z_social = self.social_projection(social_context.squeeze(1))
        z_route = self.route_projection(route_context_final.squeeze(1))
        if self.normalize_slots:
            z_ego = self.slot_norms["ego"](z_ego)
            z_social = self.slot_norms["social"](z_social)
            z_route = self.slot_norms["route"](z_route)
        latent = StructuredLatent(
            z_ego=z_ego,
            z_social=z_social,
            z_route=z_route,
        )

        attention_entropy = -(
            topology_attention.clamp_min(1e-12).log() * topology_attention
        ).sum(dim=-1)
        attention_queries = trajectory_valid.to(attention_entropy.dtype)
        topology_entropy = (
            (attention_entropy * attention_queries).sum()
            / attention_queries.sum().clamp_min(1.0)
            if self.use_topology
            else torch.zeros((), device=trajectories.device)
        )
        valid_edge_weights = edge_weights[pair_valid]
        graph_mean_edge_weight = (
            valid_edge_weights.mean()
            if valid_edge_weights.numel()
            else torch.zeros((), device=trajectories.device)
        )
        self.last_topology_attention = topology_attention.detach()
        slot_stds = torch.stack(
            [
                latent.z_ego.std(dim=0, unbiased=False).mean(),
                latent.z_social.std(dim=0, unbiased=False).mean(),
                latent.z_route.std(dim=0, unbiased=False).mean(),
            ]
        )
        self._diagnostics = {
            "topology_attention_entropy": topology_entropy,
            "latent_std": latent.tensor.std(dim=0, unbiased=False).mean(),
            "ego_latent_std": slot_stds[0],
            "social_latent_std": slot_stds[1],
            "route_latent_std": slot_stds[2],
            "slot_scale_ratio": slot_stds.max() / slot_stds.min().clamp_min(1e-8),
            "graph_mean_edge_weight": graph_mean_edge_weight,
        }
        return latent

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        return self.forward_tokens(observations).tensor


__all__ = [
    "MaskedCrossAttention",
    "RelationalTopologyEncoder",
    "RelationalTopologyLayer",
    "StructuredLatent",
    "TemporalInteractionEncoder",
    "TopoTemporalGraphExtractor",
    "VehicleGraphLayer",
]
