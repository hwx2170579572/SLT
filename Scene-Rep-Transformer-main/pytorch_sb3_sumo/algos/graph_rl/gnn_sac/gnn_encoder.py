"""Pure vehicle-graph encoder for the GNN+SAC baseline (DGN-style).

This is the "graph reinforcement learning" ablation arm.  Six vehicles are
treated as graph nodes and every history frame runs distance-weighted message
passing (``VehicleGraphLayer``) over 8-D geometric edge features (relative
position/velocity, heading delta, TTC).  No lane-topology query, no transformer
temporal attention and no SLT objective are used, so the graph encoder can be
read off against the MST transformer encoder under an identical continuous
action head.

Reference: Jiang et al., "Graph Convolutional Reinforcement Learning", ICLR 2020
(DGN), https://github.com/PKU-RL/DGN.
"""

from __future__ import annotations

from typing import Mapping

import gymnasium as gym
import torch
from torch import Tensor, nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algos.sb3_torch.features import (
    HierarchicalSceneExtractor,
    _nonzero_mask,
    keras_initialize_linear,
)
from algos.sb3_torch.topo_temporal_features import VehicleGraphLayer


class GnnSceneExtractor(BaseFeaturesExtractor):
    """Distance-weighted vehicle message passing followed by temporal pooling."""

    _current_ego_frame = HierarchicalSceneExtractor._current_ego_frame
    _rotation_angle = HierarchicalSceneExtractor._rotation_angle
    _rotate_trajectories = HierarchicalSceneExtractor._rotate_trajectories
    _rotate_map = HierarchicalSceneExtractor._rotate_map
    set_source_augmentation = HierarchicalSceneExtractor.set_source_augmentation

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        vehicle_sigma: float = 20.0,
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        trajectory_space = observation_space.spaces["trajectory"]
        if len(trajectory_space.shape) != 3 or trajectory_space.shape[-1] != 5:
            raise ValueError(f"Invalid trajectory shape: {trajectory_space.shape}")
        self.actor_count = int(trajectory_space.shape[0])
        self.history_steps = int(trajectory_space.shape[1])
        self.feature_dim = int(features_dim)
        self.hidden_dim = int(hidden_dim)
        self.random_augmentation = bool(random_augmentation)
        self.source_augmentation_enabled = True
        # DGN is agent-only; the map observation is deliberately ignored.
        self.carla_contract = (
            trajectory_space.shape[-1] == 2  # never true for SUMO; kept for parity
            if carla_contract is None
            else bool(carla_contract)
        )
        self.vehicle_sigma = float(vehicle_sigma)
        if self.vehicle_sigma <= 0.0:
            raise ValueError("vehicle_sigma must be positive")

        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.vehicle_layers = nn.ModuleList(
            VehicleGraphLayer(hidden_dim) for _ in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )
        self.state_encoder.apply(keras_initialize_linear)
        self.output.apply(keras_initialize_linear)

    def _vehicle_edge_features(
        self, trajectories: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Geometric edge features/weights for ``VehicleGraphLayer`` (no topology)."""
        # [B,H,N,*] layout for explicit target/source pair construction.
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
        edge_weights = torch.exp(-distance_squared / (2.0 * self.vehicle_sigma**2))
        node_valid = valid.permute(0, 2, 1).bool()
        pair_valid = node_valid.unsqueeze(3) & node_valid.unsqueeze(2)
        diagonal = torch.eye(
            self.actor_count, dtype=torch.bool, device=trajectories.device
        )[None, None]
        pair_valid = pair_valid & ~diagonal
        edge_weights = edge_weights * pair_valid.to(edge_weights.dtype)
        return edge_features, edge_weights, pair_valid, node_valid

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        trajectories = observations["trajectory"].float()
        trajectory_valid = _nonzero_mask(trajectories)
        rotated, _frame, _augmentation = self._rotate_trajectories(
            trajectories, trajectory_valid
        )
        state_tokens = self.state_encoder(rotated) * trajectory_valid[..., None]

        edge_features, edge_weights, pair_valid, node_valid = self._vehicle_edge_features(
            rotated, trajectory_valid
        )
        graph_features = state_tokens.permute(0, 2, 1, 3)
        for layer in self.vehicle_layers:
            graph_features = layer(
                graph_features, edge_features, edge_weights, pair_valid, node_valid
            )
        graph_sequence = graph_features.permute(0, 2, 1, 3)

        denominator = trajectory_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        actor_features = (
            graph_sequence * trajectory_valid[..., None]
        ).sum(dim=2) / denominator

        ego = actor_features[:, 0]
        neighbor_valid = trajectory_valid[:, 1:].any(dim=-1)
        social = (actor_features[:, 1:] * neighbor_valid[..., None]).sum(
            dim=1
        ) / neighbor_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        combined = torch.cat([ego, social], dim=-1)
        return self.output(combined)
