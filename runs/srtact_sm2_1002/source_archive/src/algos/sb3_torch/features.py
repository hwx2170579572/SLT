"""PyTorch port of the hierarchical Scene-Rep encoder."""

from __future__ import annotations

import math
from typing import Mapping

import gymnasium as gym
import torch
from torch import Tensor, nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def keras_initialize_linear(module: nn.Module) -> None:
    """Apply the default Keras Dense/MHA kernel and bias initialization."""

    if isinstance(module, nn.Linear):
        # Dense kernels are two-dimensional, but Keras MultiHeadAttention is
        # implemented with EinsumDense kernels such as
        # [input_dim, num_heads, head_dim]. TensorFlow 2.4 applies the generic
        # convolution-style fan calculation to those rank-three tensors.  A
        # flattened torch Linear therefore needs the original logical kernel
        # shape in order to draw from the same Glorot interval.
        logical_shape = getattr(module, "_keras_kernel_shape", None)
        if logical_shape is None:
            nn.init.xavier_uniform_(module.weight)
        else:
            shape = tuple(int(value) for value in logical_shape)
            if len(shape) == 1:
                fan_in = fan_out = shape[0]
            elif len(shape) == 2:
                fan_in, fan_out = shape
            else:
                receptive_field_size = math.prod(shape[:-2])
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
            limit = math.sqrt(6.0 / float(fan_in + fan_out))
            nn.init.uniform_(module.weight, -limit, limit)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _nonzero_mask(values: Tensor) -> Tensor:
    # The released TensorFlow code consistently builds trajectory and map
    # masks with ``tf.not_equal(tensor, 0)[..., 0]``.  Preserve that exact
    # first-coordinate contract, including its behavior on points whose x is
    # zero, rather than silently "fixing" it during the framework port.
    return values[..., 0] != 0


def _safe_attention(
    layer: "KerasStyleMultiheadAttention",
    query: Tensor,
    key: Tensor,
    key_valid: Tensor,
    query_valid: Tensor | None = None,
) -> Tensor:
    """Apply the finite additive mask used by TensorFlow 2.4 Keras MHA.

    Keras does not special-case a query row whose complete key mask is false.
    It adds ``-1e9`` to that row and still evaluates softmax.  Preserving that
    behavior matters because the released source sends an outer-product mask
    to temporal/map attention and then max-pools every query position.
    """

    valid = key_valid.bool()
    if valid.ndim != 2:
        raise ValueError(f"Expected [batch, sequence] mask, got {tuple(valid.shape)}")
    if query_valid is None:
        attention_mask = valid[:, None, :]
    else:
        query_mask = query_valid.bool()
        if query_mask.shape != query.shape[:2]:
            raise ValueError(
                "query_valid must match the query batch/sequence dimensions, "
                f"got {tuple(query_mask.shape)} for {tuple(query.shape)}"
            )
        attention_mask = query_mask[:, :, None] & valid[:, None, :]
    output, _ = layer(
        query,
        key,
        key,
        attention_mask=attention_mask,
        need_weights=False,
    )
    return output


class KerasStyleMultiheadAttention(nn.Module):
    """PyTorch attention with TensorFlow/Keras ``key_dim`` semantics.

    ``torch.nn.MultiheadAttention`` divides ``embed_dim`` across its heads,
    whereas the released TensorFlow code gives every head the full Keras
    ``key_dim`` and then projects the concatenated heads to ``output_shape``.
    Keeping those dimensions explicit avoids silently shrinking the original
    Scene-Rep attention blocks during the framework migration.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        *,
        num_heads: int,
        head_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        inner_dim = self.num_heads * self.head_dim
        self.query_projection = nn.Linear(query_dim, inner_dim)
        self.key_projection = nn.Linear(key_dim, inner_dim)
        self.value_projection = nn.Linear(value_dim, inner_dim)
        self.output_projection = nn.Linear(inner_dim, output_dim)
        self.query_projection._keras_kernel_shape = (  # type: ignore[attr-defined]
            query_dim,
            self.num_heads,
            self.head_dim,
        )
        self.key_projection._keras_kernel_shape = (  # type: ignore[attr-defined]
            key_dim,
            self.num_heads,
            self.head_dim,
        )
        self.value_projection._keras_kernel_shape = (  # type: ignore[attr-defined]
            value_dim,
            self.num_heads,
            self.head_dim,
        )
        self.output_projection._keras_kernel_shape = (  # type: ignore[attr-defined]
            self.num_heads,
            self.head_dim,
            output_dim,
        )
        self.apply(keras_initialize_linear)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if key_padding_mask is not None and attention_mask is not None:
            raise ValueError("Pass either key_padding_mask or attention_mask, not both")
        projected_query = self._split_heads(self.query_projection(query))
        projected_key = self._split_heads(self.key_projection(key))
        projected_value = self._split_heads(self.value_projection(value))
        scores = torch.matmul(projected_query, projected_key.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.head_dim))
        if key_padding_mask is not None:
            attention_mask = ~key_padding_mask[:, None, :].bool()
        if attention_mask is not None:
            mask = attention_mask.bool()
            if mask.ndim != 3:
                raise ValueError(
                    "attention_mask must have shape [batch, query, key], "
                    f"got {tuple(mask.shape)}"
                )
            compatible_negative = (
                torch.finfo(scores.dtype).min
                if scores.dtype == torch.float16
                else -1e9
            )
            adder = (1.0 - mask[:, None, :, :].to(scores.dtype)) * compatible_negative
            scores = scores + adder
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nn.functional.dropout(
            weights, p=self.dropout, training=self.training
        )
        attended = torch.matmul(weights, projected_value)
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        output = self.output_projection(attended)
        return output, weights if need_weights else None


class TemporalActorEncoder(nn.Module):
    """Encode each actor history with temporal self-attention and max pooling."""

    def __init__(self, input_dim: int = 5, feature_dim: int = 128, num_heads: int = 2) -> None:
        super().__init__()
        self.attention = KerasStyleMultiheadAttention(
            input_dim,
            input_dim,
            input_dim,
            num_heads=num_heads,
            head_dim=feature_dim,
            output_dim=feature_dim,
        )

    def forward(self, states: Tensor, valid: Tensor) -> Tensor:
        batch, actors, timesteps, state_dim = states.shape
        flat_states = states.reshape(batch * actors, timesteps, state_dim)
        flat_valid = valid.reshape(batch * actors, timesteps)
        attended = torch.relu(
            _safe_attention(
                self.attention,
                flat_states,
                flat_states,
                flat_valid,
                query_valid=flat_valid,
            )
        )
        pooled = attended.amax(dim=1)
        return pooled.reshape(batch, actors, -1)


class MapPolylineEncoder(nn.Module):
    """Encode each candidate lane polyline, matching the original map branch."""

    def __init__(self, map_dim: int, feature_dim: int = 128, num_heads: int = 2) -> None:
        super().__init__()
        self.map_dim = map_dim
        node_dim = 2 if map_dim == 2 else 3
        self.node_attention = KerasStyleMultiheadAttention(
            node_dim,
            node_dim,
            node_dim,
            num_heads=num_heads,
            head_dim=feature_dim,
            output_dim=192,
        )
        # The original CARLA branch also embeds the final two map values.  In
        # that branch they are x/y; in SMARTS they are the two path flags.
        self.vector_feature = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(192 + 64, feature_dim), nn.ReLU())

    def forward(self, map_state: Tensor, valid: Tensor) -> Tensor:
        batch, paths, points, map_dim = map_state.shape
        flat_map = map_state.reshape(batch * paths, points, map_dim)
        flat_valid = valid.reshape(batch * paths, points)
        node_dim = 2 if self.map_dim == 2 else 3
        nodes = flat_map[:, :, :node_dim]
        nodes = torch.relu(
            _safe_attention(
                self.node_attention,
                nodes,
                nodes,
                flat_valid,
                query_valid=flat_valid,
            )
        )
        pooled = nodes.amax(dim=1)
        flags = self.vector_feature(flat_map[:, 0, -2:])
        pooled = torch.cat([pooled, flags], dim=-1)
        output = self.output(pooled)
        return output.reshape(batch, paths, -1)


class HierarchicalSceneExtractor(BaseFeaturesExtractor):
    """Scene-Rep hierarchy implemented as an SB3 multi-input extractor.

    The hierarchy follows the TensorFlow implementation:

    1. transform every trajectory and map polyline into the current ego frame;
    2. aggregate each actor through temporal self-attention;
    3. relate every neighbor to its candidate paths;
    4. relate the ego actor to all actors;
    5. attend from that interaction feature to ego candidate goals.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        num_heads: int = 2,
        goal_heads: int = 1,
        num_modes: int = 1,
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
        no_neighbor_future: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        trajectory_space = observation_space.spaces["trajectory"]
        map_space = observation_space.spaces["map"]
        if len(trajectory_space.shape) != 3 or trajectory_space.shape[-1] != 5:
            raise ValueError(f"Invalid trajectory shape: {trajectory_space.shape}")
        if len(map_space.shape) != 3 or map_space.shape[-1] not in (2, 5):
            raise ValueError(f"Invalid map shape: {map_space.shape}")
        self.actor_count = int(trajectory_space.shape[0])
        self.map_dim = int(map_space.shape[-1])
        if map_space.shape[0] % self.actor_count:
            raise ValueError("Map path count must be divisible by actor count")
        self.paths_per_actor = int(map_space.shape[0] // self.actor_count)
        self.feature_dim = int(features_dim)
        self.goal_heads = int(goal_heads)
        self.num_modes = int(num_modes)
        self.random_augmentation = bool(random_augmentation)
        # The released training encoder applies random rotation even while an
        # action is sampled for environment interaction. SB3 switches policies
        # to eval() for rollouts, so this source-semantic switch must be
        # independent of nn.Module.training. Formal evaluation disables it.
        self.source_augmentation_enabled = True
        self.carla_contract = self.map_dim == 2 if carla_contract is None else bool(carla_contract)
        self.no_neighbor_future = bool(no_neighbor_future)

        self.temporal_encoder = TemporalActorEncoder(5, features_dim, num_heads)
        self.map_encoder = MapPolylineEncoder(self.map_dim, features_dim, num_heads)
        self.neighbor_map_attention = KerasStyleMultiheadAttention(
            features_dim,
            features_dim,
            features_dim,
            num_heads=num_heads,
            head_dim=features_dim,
            output_dim=features_dim,
        )
        self.actor_attention = KerasStyleMultiheadAttention(
            features_dim,
            features_dim,
            features_dim,
            num_heads=num_heads,
            head_dim=features_dim,
            output_dim=features_dim,
        )
        self.goal_attention = nn.ModuleList(
            [
                KerasStyleMultiheadAttention(
                    features_dim,
                    features_dim,
                    features_dim,
                    num_heads=self.goal_heads,
                    head_dim=features_dim,
                    output_dim=features_dim,
                )
                for _ in range(self.num_modes)
            ]
        )
        self.apply(keras_initialize_linear)

    def _current_ego_frame(self, trajectories: Tensor, valid: Tensor) -> Tensor:
        ego_valid = valid[:, 0]
        last_index = ego_valid.long().sum(dim=1).sub(1).clamp(min=0)
        batch_index = torch.arange(trajectories.shape[0], device=trajectories.device)
        return trajectories[batch_index, 0, last_index]

    def _rotation_angle(self, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.random_augmentation and self.source_augmentation_enabled:
            return (torch.rand(batch, dtype=dtype, device=device) - 0.5) * math.pi
        return torch.zeros(batch, dtype=dtype, device=device)

    def set_source_augmentation(self, enabled: bool) -> None:
        """Enable training-time source augmentation independently of eval()."""

        self.source_augmentation_enabled = bool(enabled)

    def _rotate_trajectories(
        self, trajectories: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        frame = self._current_ego_frame(trajectories, valid)
        yaw = frame[:, 2]
        augmentation = self._rotation_angle(
            trajectories.shape[0], trajectories.dtype, trajectories.device
        )
        rotation = yaw + augmentation
        cosine = torch.cos(rotation)[:, None, None]
        sine = torch.sin(rotation)[:, None, None]

        x = trajectories[..., 0] - frame[:, None, None, 0]
        y = trajectories[..., 1] - frame[:, None, None, 1]
        vx = trajectories[..., 3] - frame[:, None, None, 3]
        vy = trajectories[..., 4] - frame[:, None, None, 4]
        rotated_x = cosine * x + sine * y
        rotated_y = -sine * x + cosine * y
        rotated_vx = cosine * vx + sine * vy
        rotated_vy = -sine * vx + cosine * vy
        # The source subtracts headings directly and does not wrap the result.
        heading = trajectories[..., 2] - yaw[:, None, None]
        if self.carla_contract:
            rotated_x = -rotated_x
            rotated_vx = -rotated_vx
        rotated = torch.stack(
            [rotated_x, rotated_y, heading, rotated_vx, rotated_vy], dim=-1
        )
        return rotated * valid[..., None], frame, augmentation

    def _rotate_map(
        self,
        map_state: Tensor,
        valid: Tensor,
        frame: Tensor,
        augmentation: Tensor,
    ) -> Tensor:
        yaw = frame[:, 2] + augmentation
        cosine = torch.cos(yaw)[:, None, None]
        sine = torch.sin(yaw)[:, None, None]
        x = map_state[..., 0] - frame[:, None, None, 0]
        y = map_state[..., 1] - frame[:, None, None, 1]
        rotated_x = cosine * x + sine * y
        rotated_y = -sine * x + cosine * y
        if self.carla_contract:
            output = torch.stack([-rotated_x, rotated_y], dim=-1)
        else:
            heading = map_state[..., 2] - frame[:, None, None, 2]
            output = torch.stack(
                [rotated_x, rotated_y, heading, map_state[..., 3], map_state[..., 4]],
                dim=-1,
            )
        return output * valid[..., None]

    def _source_neighbor_maps(
        self, map_features: Tensor, path_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Select neighbor paths with the released ``i*2`` indexing contract.

        SMARTS has two paths per actor, so this is the natural grouping.  The
        CARLA branch produces three paths per actor but the released
        Hierachial_Transformer still slices ``neighbor_map[i*2:i*2+2]``.
        Preserve that overlap/omission exactly because source behavior takes
        precedence over correcting the apparent indexing bug.
        """

        neighbor_count = self.actor_count - 1
        if not self.carla_contract:
            return map_features[:, 1:], path_valid[:, 1:]
        flat_features = map_features[:, 1:].reshape(
            map_features.shape[0], -1, self.feature_dim
        )
        flat_valid = path_valid[:, 1:].reshape(path_valid.shape[0], -1)
        indices = (
            torch.arange(neighbor_count, device=map_features.device)[:, None] * 2
            + torch.arange(2, device=map_features.device)[None, :]
        )
        return flat_features[:, indices], flat_valid[:, indices]

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        trajectories = observations["trajectory"].float()
        map_state = observations["map"].float()
        trajectory_valid = _nonzero_mask(trajectories)
        map_valid = _nonzero_mask(map_state)
        rotated, frame, augmentation = self._rotate_trajectories(
            trajectories, trajectory_valid
        )
        rotated_map = self._rotate_map(map_state, map_valid, frame, augmentation)

        actor_features = self.temporal_encoder(rotated, trajectory_valid)
        map_features = self.map_encoder(rotated_map, map_valid)
        batch = trajectories.shape[0]
        map_features = map_features.reshape(
            batch, self.actor_count, self.paths_per_actor, self.feature_dim
        )
        # Released map_traj_mask checks only the first path point.
        path_valid = map_valid[:, :, 0].reshape(
            batch, self.actor_count, self.paths_per_actor
        )
        # Released actor_mask always enables ego and checks timestep zero for
        # every neighbor rather than reducing across the history.
        actor_valid = _nonzero_mask(rotated)[:, :, 0].clone()
        actor_valid[:, 0] = True

        ego_feature = actor_features[:, :1]
        neighbor_count = self.actor_count - 1
        if neighbor_count:
            # The TensorFlow layer is shared by every neighbor. Treating the
            # neighbor axis as an additional batch axis is mathematically
            # identical to the former Python loop and avoids many tiny CUDA
            # launches during every SAC gradient step.
            neighbor_queries = actor_features[:, 1:].reshape(
                batch * neighbor_count, 1, self.feature_dim
            )
            if self.no_neighbor_future:
                related_neighbors = neighbor_queries.squeeze(1)
            else:
                neighbor_maps, neighbor_path_valid = self._source_neighbor_maps(
                    map_features, path_valid
                )
                neighbor_keys = torch.cat(
                    [actor_features[:, 1:, None, :], neighbor_maps],
                    dim=2,
                ).reshape(
                    batch * neighbor_count,
                    1 + neighbor_maps.shape[2],
                    self.feature_dim,
                )
                neighbor_key_valid = torch.cat(
                    [
                        torch.ones_like(actor_valid[:, 1:, None]),
                        neighbor_path_valid,
                    ],
                    dim=2,
                ).reshape(batch * neighbor_count, 1 + neighbor_maps.shape[2])
                related_neighbors = torch.relu(
                    _safe_attention(
                        self.neighbor_map_attention,
                        neighbor_queries,
                        neighbor_keys,
                        neighbor_key_valid,
                    )
                ).squeeze(1)
            related_neighbors = related_neighbors.reshape(
                batch, neighbor_count, self.feature_dim
            )
            actor_keys = torch.cat([ego_feature, related_neighbors], dim=1)
        else:
            actor_keys = ego_feature
        interaction = torch.relu(
            _safe_attention(self.actor_attention, ego_feature, actor_keys, actor_valid)
        )

        ego_maps = map_features[:, 0]
        ego_map_valid = path_valid[:, 0]
        modes = [
            torch.relu(_safe_attention(layer, interaction, ego_maps, ego_map_valid)).squeeze(1)
            for layer in self.goal_attention
        ]
        goals = torch.stack(modes, dim=1)
        scene_modes = goals + interaction
        return scene_modes.mean(dim=1)
