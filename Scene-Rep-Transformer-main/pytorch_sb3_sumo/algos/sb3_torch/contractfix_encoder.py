"""Isolated D1 encoder repairs for the SMARTS observation contract.

This version fixes last-valid history selection and the velocity basis used by
explicit geometric vehicle edges.  It deliberately leaves the released
encoder modules and state-token path unchanged.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from torch import Tensor

from .incremental_topo_encoder import IncrementalTopoEncoder
from .topo_temporal_features import TemporalInteractionEncoder


_CPA_FEATURE_HORIZON_S = 20.0
_RELATIVE_SPEED_SQUARED_EPS = 1e-12


class ContractFixedTemporalInteractionEncoder(TemporalInteractionEncoder):
    """Temporal pooling with an actual-last-valid query and safe empty rows."""

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        if features.ndim != 4 or valid.shape != features.shape[:-1]:
            raise ValueError(
                "Temporal features/mask must have shapes [B,A,H,D]/[B,A,H], "
                f"got {tuple(features.shape)}/{tuple(valid.shape)}"
            )
        batch, actors, history, feature_dim = features.shape
        if history <= 0 or history > self.lag_embedding.shape[0]:
            raise ValueError(
                f"history length {history} is outside [1, {self.lag_embedding.shape[0]}]"
            )

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

        positions = torch.arange(history, device=flat.device)[None, :]
        last_index = torch.where(flat_valid, positions, -1).amax(dim=1).clamp_min(0)
        batch_index = torch.arange(flat.shape[0], device=flat.device)
        query = encoded[batch_index, last_index].unsqueeze(1)
        pooled, _ = self.pool_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        output = self.pool_norm(query + pooled).squeeze(1)
        output = torch.where(actor_valid[:, None], output, torch.zeros_like(output))
        return output.reshape(batch, actors, feature_dim)


class ContractFixedIncrementalTopoEncoder(IncrementalTopoEncoder):
    """D1 encoder with scoped SMARTS C8/C9 repairs.

    All ordinary ``IncrementalTopoEncoder`` keyword arguments are forwarded
    unchanged.  ``source_observation_contract`` is required so the velocity
    basis conversion cannot silently be applied to Cartesian/CARLA data.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        source_observation_contract: str,
        **kwargs: Any,
    ) -> None:
        if source_observation_contract != "smarts":
            raise ValueError(
                "ContractFixedIncrementalTopoEncoder only supports the SMARTS "
                f"observation contract, got {source_observation_contract!r}"
            )
        spaces = getattr(observation_space, "spaces", {})
        trajectory_space = spaces.get("trajectory")
        map_space = spaces.get("map")
        if trajectory_space is None or map_space is None:
            raise ValueError("Contract-fixed D1 requires Dict keys 'trajectory' and 'map'")
        if len(trajectory_space.shape) != 3 or trajectory_space.shape[-1] != 5:
            raise ValueError(
                "SMARTS trajectory must have observation shape [actors, history, 5], "
                f"got {trajectory_space.shape}"
            )
        if len(map_space.shape) != 3 or map_space.shape[-1] != 5:
            raise ValueError(
                "The SMARTS contract repair expects the released 5-channel map, "
                f"got {map_space.shape}"
            )
        if kwargs.get("carla_contract") is True:
            raise ValueError("The SMARTS contract repair cannot use carla_contract=True")

        super().__init__(observation_space, **kwargs)
        self.source_observation_contract = source_observation_contract

        # Keep the base module's initialization/RNG stream and exact parameter
        # layout.  Construction of the replacement occurs in an isolated RNG
        # fork, then receives the original temporal weights by strict copy.
        original_temporal = self.temporal_encoder
        with torch.random.fork_rng(devices=[]):
            fixed_temporal = ContractFixedTemporalInteractionEncoder(
                feature_dim=int(original_temporal.lag_embedding.shape[-1]),
                num_heads=int(original_temporal.self_attention.num_heads),
                history_steps=int(original_temporal.lag_embedding.shape[0]),
            )
        fixed_temporal.load_state_dict(original_temporal.state_dict(), strict=True)
        self.temporal_encoder = fixed_temporal

    @staticmethod
    def _last_valid(features: Tensor, valid: Tensor) -> Tensor:
        """Gather max true history index; return zeros for an empty mask."""

        if features.ndim < 2 or valid.shape != features.shape[:-1]:
            raise ValueError(
                "Features/mask must have shapes [..., H, D]/[..., H], "
                f"got {tuple(features.shape)}/{tuple(valid.shape)}"
            )
        history = int(features.shape[-2])
        if history <= 0:
            raise ValueError("History dimension must be non-empty")
        mask = valid.bool()
        positions = torch.arange(history, device=features.device)
        view_shape = (1,) * (mask.ndim - 1) + (history,)
        last_index = torch.where(mask, positions.view(view_shape), -1).amax(dim=-1)
        has_valid = last_index >= 0
        gather_index = last_index.clamp_min(0)[..., None, None].expand(
            *last_index.shape, 1, features.shape[-1]
        )
        selected = features.gather(dim=-2, index=gather_index).squeeze(-2)
        return torch.where(has_valid[..., None], selected, torch.zeros_like(selected))

    def _current_ego_frame(self, trajectories: Tensor, valid: Tensor) -> Tensor:
        """Use the same max-valid selector for the ego rotation anchor."""

        if trajectories.ndim != 4 or valid.shape != trajectories.shape[:-1]:
            raise ValueError(
                "Ego frame expects [B,A,H,5] trajectories and [B,A,H] mask"
            )
        return self._last_valid(trajectories[:, 0], valid[:, 0])

    @staticmethod
    def _last_query_mask(key_mask: Tensor, valid: Tensor) -> Tensor:
        """Select topology keys at the actual last valid history slot.

        ``key_mask`` is ``[B,A,H,K]`` and ``valid`` is ``[B,A,H]``.  This
        selector is currently bypassed by ST/ST-RT because topology is off,
        but keeping it correct prevents a future topology-enabled reuse of the
        contract-fixed class from reintroducing count-minus-one indexing.
        """

        if key_mask.ndim != 4 or valid.shape != key_mask.shape[:-1]:
            raise ValueError(
                "Topology query masks must have shapes [B,A,H,K]/[B,A,H], "
                f"got {tuple(key_mask.shape)}/{tuple(valid.shape)}"
            )
        history = int(key_mask.shape[2])
        if history <= 0:
            raise ValueError("History dimension must be non-empty")
        mask = valid.bool()
        positions = torch.arange(history, device=key_mask.device).view(1, 1, history)
        last_index = torch.where(mask, positions, -1).amax(dim=-1)
        has_valid = last_index >= 0
        gather_index = last_index.clamp_min(0)[..., None, None].expand(
            *last_index.shape, 1, key_mask.shape[-1]
        )
        selected = key_mask.gather(dim=2, index=gather_index).squeeze(2).bool()
        return selected & has_valid[..., None]

    @staticmethod
    def _correct_smarts_geometry_edges(edge_features: Tensor) -> Tensor:
        """Convert edge Δv to Cartesian and recompute the capped CPA-time field.

        For SMARTS, the legacy per-actor velocity channels are pseudo vectors
        ``(s cos h, s sin h)`` while x/y are east/north.  Applying
        ``J(x,y)=(-y,x)`` recovers the Cartesian velocity basis.  Channel 5
        remains a 20-second capped closest-approach-time proxy; it is not TTC.
        """

        if edge_features.shape[-1] != 8:
            raise ValueError(
                f"Expected the released 8-channel vehicle edge, got {edge_features.shape[-1]}"
            )
        output = edge_features.clone()
        pseudo_delta_velocity = edge_features[..., 2:4]
        cartesian_delta_velocity = torch.stack(
            (-pseudo_delta_velocity[..., 1], pseudo_delta_velocity[..., 0]), dim=-1
        )
        delta_position = edge_features[..., :2]
        relative_speed_squared = cartesian_delta_velocity.square().sum(dim=-1)
        closing = -(delta_position * cartesian_delta_velocity).sum(dim=-1)
        approaching = (closing > 0.0) & (
            relative_speed_squared > _RELATIVE_SPEED_SQUARED_EPS
        )
        closest_approach_time = torch.where(
            approaching,
            closing / relative_speed_squared.clamp_min(_RELATIVE_SPEED_SQUARED_EPS),
            torch.full_like(closing, _CPA_FEATURE_HORIZON_S),
        ).clamp(0.0, _CPA_FEATURE_HORIZON_S)
        output[..., 2:4] = cartesian_delta_velocity
        output[..., 5] = closest_approach_time / _CPA_FEATURE_HORIZON_S
        return output

    def _vehicle_edge_features(
        self,
        trajectories: Tensor,
        valid: Tensor,
        topology_attention: Tensor | None,
        *,
        include_topology_relations: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_features, edge_weights, pair_valid, node_valid = super()._vehicle_edge_features(
            trajectories,
            valid,
            topology_attention,
            include_topology_relations=include_topology_relations,
        )
        return (
            self._correct_smarts_geometry_edges(edge_features),
            edge_weights,
            pair_valid,
            node_valid,
        )

    def _vehicle_edge_features_v2(
        self,
        trajectories: Tensor,
        valid: Tensor,
        topology_attention: Tensor,
        *,
        topology_key_mask: Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            edge_features,
            edge_weights,
            pair_valid,
            node_valid,
            mean_merge_pair_score,
        ) = super()._vehicle_edge_features_v2(
            trajectories,
            valid,
            topology_attention,
            topology_key_mask=topology_key_mask,
            collect_diagnostics=collect_diagnostics,
        )
        return (
            self._correct_smarts_geometry_edges(edge_features),
            edge_weights,
            pair_valid,
            node_valid,
            mean_merge_pair_score,
        )


__all__ = [
    "ContractFixedIncrementalTopoEncoder",
    "ContractFixedTemporalInteractionEncoder",
]
