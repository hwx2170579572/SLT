"""Simple MLP/LSTM scene encoder for the HSAC baseline.

HSAC replaces the Scene-Rep attention hierarchy with a lightweight encoder so
the hybrid action head (``DecisionAlignedHybridActor``) can be isolated from
the graph/transformer representation.  It consumes the same
``trajectory(6,10,5)`` and ``map(12,10,5)`` observations as the Scene-Rep
family, reuses the exact ego-frame coordinate alignment from
``HierarchicalSceneExtractor``, and returns a 128-D feature vector.

Unlike ``SourceSacLstmExtractor`` (which requires the legacy ``state_lstm``
observation), this extractor deliberately reads the modern trajectory/map
contract so it can be dropped into the exact same environment family as
MST+SLT / TASAC.
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


class SimpleMlpLstmExtractor(BaseFeaturesExtractor):
    """MLP or GRU aggregation over ego-frame trajectories and map polylines.

    The hierarchy is: coordinate alignment -> per-frame state/map encoding ->
    temporal pooling (masked mean or GRU last-state) -> ego + social + route
    concatenation -> 128-D projection.  ``backbone="mlp"`` pools each actor
    history with a masked mean; ``backbone="lstm"`` aggregates with a GRU and
    keeps the last valid hidden state.
    """

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
        backbone: str = "mlp",
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        trajectory_space = observation_space.spaces["trajectory"]
        map_space = observation_space.spaces["map"]
        if len(trajectory_space.shape) != 3 or trajectory_space.shape[-1] != 5:
            raise ValueError(f"Invalid trajectory shape: {trajectory_space.shape}")
        if len(map_space.shape) != 3 or map_space.shape[-1] not in (2, 5):
            raise ValueError(f"Invalid map shape: {map_space.shape}")
        if backbone not in ("mlp", "lstm"):
            raise ValueError(f"backbone must be 'mlp' or 'lstm', got {backbone!r}")
        self.actor_count = int(trajectory_space.shape[0])
        self.history_steps = int(trajectory_space.shape[1])
        self.map_dim = int(map_space.shape[-1])
        if int(map_space.shape[0]) % self.actor_count:
            raise ValueError("Map path count must be divisible by actor count")
        self.paths_per_actor = int(map_space.shape[0] // self.actor_count)
        self.feature_dim = int(features_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone = backbone
        self.random_augmentation = bool(random_augmentation)
        self.source_augmentation_enabled = True
        self.carla_contract = (
            self.map_dim == 2 if carla_contract is None else bool(carla_contract)
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.map_point_encoder = nn.Sequential(
            nn.Linear(self.map_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal = (
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            if backbone == "lstm"
            else None
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )
        self.state_encoder.apply(keras_initialize_linear)
        self.map_point_encoder.apply(keras_initialize_linear)
        self.output.apply(keras_initialize_linear)
        if self.temporal is not None:
            for name, parameter in self.temporal.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def _temporal_pool(self, state_tokens: Tensor, valid: Tensor) -> Tensor:
        """Pool ``[batch, slots, history, hidden]`` to ``[batch, slots, hidden]``."""
        batch, slots, history, hidden = state_tokens.shape
        if self.backbone == "lstm":
            assert self.temporal is not None
            flat = state_tokens.reshape(batch * slots, history, hidden)
            flat_valid = valid.reshape(batch * slots, history)
            sequence, _ = self.temporal(flat)
            last_index = flat_valid.long().sum(dim=1).sub(1).clamp(min=0)
            gather = last_index[:, None, None].expand(-1, 1, hidden)
            pooled = sequence.gather(1, gather).squeeze(1)
            return pooled.reshape(batch, slots, hidden)
        denominator = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        return (state_tokens * valid[..., None]).sum(dim=2) / denominator

    @staticmethod
    def _masked_mean(features: Tensor, valid: Tensor) -> Tensor:
        """Mean over the slot axis, honoring a ``[batch, slots]`` mask."""
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (features * valid[..., None]).sum(dim=1) / denominator

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        trajectories = observations["trajectory"].float()
        map_state = observations["map"].float()
        trajectory_valid = _nonzero_mask(trajectories)
        map_valid = _nonzero_mask(map_state)
        rotated, frame, augmentation = self._rotate_trajectories(
            trajectories, trajectory_valid
        )
        rotated_map = self._rotate_map(map_state, map_valid, frame, augmentation)

        state_tokens = self.state_encoder(rotated) * trajectory_valid[..., None]
        actor_features = self._temporal_pool(state_tokens, trajectory_valid)

        map_tokens = self.map_point_encoder(rotated_map) * map_valid[..., None]
        path_features = self._temporal_pool(map_tokens, map_valid)
        map_per_actor = path_features.reshape(
            trajectories.shape[0], self.actor_count, self.paths_per_actor, self.hidden_dim
        ).sum(dim=2) / self.paths_per_actor

        ego = actor_features[:, 0]
        social = self._masked_mean(
            actor_features[:, 1:], trajectory_valid[:, 1:].any(dim=-1)
        )
        route = map_per_actor[:, 0]
        combined = torch.cat([ego, social, route], dim=-1)
        return self.output(combined)
