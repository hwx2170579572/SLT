"""Incremental topology-aware encoder for D1 representation ablations.

The D1 "representation" capability leg of hold35k decomposes into four
orthogonal, independently switchable increments that build the full TopoGTE
encoder up from the plain MLP encoder one step at a time:

    D1-1  spatiotemporal interaction  (vehicle graph message passing + temporal
                                      post-hoc aggregation instead of the MLP
                                      masked-mean pooling / social average)
    D1-2  route intent injection      (MapPolylineEncoder + masked cross
                                      attention instead of the MLP map mean)
    D1-3  topology prior (M1)         (static lane graph + soft geometric
                                      topology query, and topology-derived
                                      edge features)
    D1-4  structured slots + learning objective (32/64/32 three-slot output +
                                      SLT / SBS, enabled on the SAC side)

``IncrementalTopoEncoder`` inherits ``TopoTemporalGraphExtractorV2`` (the full
hold35k encoder, ``variant="soft"``) and re-uses every one of its sub-modules.
It only overrides ``forward_tokens`` so that each of the three non-full states
can disable a sub-path cleanly.  When all three switches are on, ``forward_tokens``
defers to ``super()`` and is therefore byte-identical to the hold35k encoder.

Only *new* files are added; no existing module is modified.
"""

from __future__ import annotations

from typing import Mapping

import gymnasium as gym
import torch
from torch import Tensor, nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .features import _nonzero_mask, keras_initialize_linear
from .topo_temporal_features import StructuredLatent
from .topo_temporal_features_v2 import TopoTemporalGraphExtractorV2


class IncrementalTopoEncoder(TopoTemporalGraphExtractorV2):
    """TopoGTE with independently switchable representation increments.

    Switches (all default to the full hold35k configuration):

    * ``use_route``     -- route intent injection (MapPolylineEncoder + masked
      cross attention).  When disabled, the MLP-style map mean feeds the route
      slot and no route context is injected into the interaction tokens.
    * ``use_topology``  -- M1 static lane-graph soft query.  When disabled, no
      topology context is added and the vehicle-graph edge features fall back
      to the geometry/motion-only version (no same-lane / conflict terms).
    * ``use_slots``     -- 32/64/32 three-slot projection.  When disabled, the
      ego/social/route components are concatenated through an MLP head into a
      single 128-D vector (matching the plain MLP encoder output contract).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        topology_graph,
        use_route: bool = True,
        use_topology: bool = True,
        use_slots: bool = True,
        route_gate: bool = False,
        route_late: bool = False,
        route_ego_only: bool = False,
        route_cond_edge: bool = False,
        route_gate_bias: float = 2.0,
        features_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 2,
        random_augmentation: bool = True,
        carla_contract: bool | None = None,
        route_sigma: float = 20.0,
        topology_sigma: float = 20.0,
        vehicle_sigma: float = 20.0,
        topology_top_k: int = 8,
        reverse_heading_cosine_min: float = 0.0,
        route_bias_init: float = 1.0,
        heading_bias_init: float = 1.0,
        topology_layerscale_init: float = 1e-3,
        goal_layerscale_init: float = 1e-3,
    ) -> None:
        super().__init__(
            observation_space,
            topology_graph=topology_graph,
            variant="soft",
            features_dim=features_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            random_augmentation=random_augmentation,
            carla_contract=carla_contract,
            topology_sigma=topology_sigma,
            route_sigma=route_sigma,
            vehicle_sigma=vehicle_sigma,
            topology_top_k=topology_top_k,
            reverse_heading_cosine_min=reverse_heading_cosine_min,
            route_bias_init=route_bias_init,
            heading_bias_init=heading_bias_init,
            topology_layerscale_init=topology_layerscale_init,
            goal_layerscale_init=goal_layerscale_init,
        )
        self.use_route = bool(use_route)
        # The v2 constructor hard-codes use_topology = True; re-open it so the
        # parent class's geometry-only edge-feature branch is reachable again.
        self.use_topology = bool(use_topology)
        self.use_slots = bool(use_slots)
        # Route-conditioning variants (all require use_route=True):
        #   route_gate      -- adaptive, initially-near-zero residual gate on the route
        #                      context injected into the interaction tokens.
        #   route_late      -- route does NOT enter interaction; it only feeds the route
        #                      slot (goal-side) of the final policy head.
        #   route_ego_only  -- unconditional route injection into ego only (neighbors keep
        #                      state-only); isolates "neighbor route is harmful".
        #   route_cond_edge -- route does NOT enter node representation; it only modulates
        #                      ego->j edge relevance (route-conditioned interaction).
        self.route_gate = bool(route_gate) and self.use_route
        self.route_late = bool(route_late) and self.use_route
        self.route_ego_only = bool(route_ego_only) and self.use_route
        self.route_cond_edge = bool(route_cond_edge) and self.use_route
        if self.route_gate:
            self.route_gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            self.route_gate_mlp.apply(keras_initialize_linear)
            # gate_bias > 0 => alpha = sigmoid(z - bias) starts near 0 (~0.12 at 2.0).
            self.route_gate_bias = nn.Parameter(torch.tensor(float(route_gate_bias)))
        if self.route_cond_edge:
            # edge feature dim = 8 (delta_pos 2 + delta_vel 2 + cos 1 + ttc 1 +
            # same_lane 1 + conflict 1), matching VehicleGraphLayer(edge_dim=8).
            self.route_edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.route_edge_mlp.apply(keras_initialize_linear)

        # MLP-style fallback modules, used only when a switch is disabled.
        self.mlp_map_point_encoder = nn.Sequential(
            nn.Linear(self.map_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_map_point_encoder.apply(keras_initialize_linear)
        self.mlp_output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )
        self.mlp_output.apply(keras_initialize_linear)

    # ------------------------------------------------------------------ #
    # MLP-style fallback helpers (mirror SimpleMlpLstmExtractor)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mlp_temporal_pool(state_tokens: Tensor, valid: Tensor) -> Tensor:
        """Masked mean over the history axis: [B, N, H, D] -> [B, N, D]."""
        denominator = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        return (state_tokens * valid[..., None]).sum(dim=2) / denominator

    def _mlp_map_route(self, rotated_map: Tensor, map_valid: Tensor) -> Tensor:
        """MLP-style ego map component: [B, P, 10, F] -> [B, hidden]."""
        map_tokens = self.mlp_map_point_encoder(rotated_map) * map_valid[..., None]
        path_features = self._mlp_temporal_pool(map_tokens, map_valid)
        map_per_actor = path_features.reshape(
            rotated_map.shape[0],
            self.actor_count,
            self.paths_per_actor,
            self.feature_dim,
        ).sum(dim=2) / self.paths_per_actor
        return map_per_actor[:, 0]

    def _apply_route_edge_correction(
        self,
        edge_weights: Tensor,
        edge_features: Tensor,
        route_context: Tensor,
    ) -> Tensor:
        """Route-conditioned edge relevance（R2）：route 只调 ego->j 的边权重。

        当前 edge_weights = exp(-d²/2σ²)，shape [B, H, target, source]。对 ego(target=0)
        的边加一个可学习 correction：w[ego,j] = w_orig * exp(g(r_ego, e[ego,j]))。
        g = MLP([r_ego, e[ego,j]])，keras init 使其初始近 0 => exp(g)≈1，初始中性。
        neighbor->neighbor 的边不受影响。correction clamp 到 ±5 防数值溢出。
        """
        ego_edge = edge_features[:, :, 0]  # [B, H, N, E]（target=0 的边特征）
        batch, history, actors, edge_dim = ego_edge.shape
        r_ego = route_context[:, 0]  # [B, D]
        r = r_ego[:, None, None, :].expand(batch, history, actors, -1)  # [B, H, N, D]
        correction = self.route_edge_mlp(
            torch.cat([r, ego_edge], dim=-1)
        ).squeeze(-1)  # [B, H, N]
        correction = correction.clamp(-5.0, 5.0)
        out = edge_weights.clone()
        out[:, :, 0] = edge_weights[:, :, 0] * torch.exp(correction)
        return out

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_tokens(self, observations: Mapping[str, Tensor]):
        if self.use_route and self.use_topology and self.use_slots:
            # Full configuration == the hold35k encoder, verbatim.
            return super().forward_tokens(observations)
        return self._forward_incremental(observations)

    def _forward_incremental(self, observations: Mapping[str, Tensor]):
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

        # ---- route intent injection (D1-2) ----
        if self.use_route:
            route_tokens, path_valid, route_context = self._encode_routes(
                rotated_map, map_valid, current_state
            )
            if self.route_late or self.route_cond_edge:
                # Late / edge-conditioned：route 不进 interaction token（只进 route slot
                # 或 edge relevance），state 保持纯净。
                intent_tokens = state_tokens
            elif self.route_ego_only:
                # Ego-only route injection：只让 ego(actor 0) 注入 route，neighbor 保持
                # state-only，隔离「neighbor route 进入 message 是否有害」。
                ego_mask = torch.zeros(
                    route_context.shape[1], device=route_context.device
                )
                ego_mask[0] = 1.0
                intent_tokens = self.route_norm(
                    state_tokens
                    + route_context[:, :, None, :] * ego_mask[None, :, None, None]
                ) * trajectory_valid[..., None]
            elif self.route_gate:
                # Gated route injection：自适应、初始近零的残差门控。
                state_pool = self._mlp_temporal_pool(state_tokens, trajectory_valid)
                gate_logit = self.route_gate_mlp(
                    torch.cat([state_pool, route_context], dim=-1)
                )
                alpha = torch.sigmoid(gate_logit - self.route_gate_bias)
                intent_tokens = self.route_norm(
                    state_tokens + alpha[:, :, None, :] * route_context[:, :, None, :]
                ) * trajectory_valid[..., None]
            else:
                intent_tokens = self.route_norm(
                    state_tokens + route_context[:, :, None, :]
                ) * trajectory_valid[..., None]
            route_component = None
        else:
            route_tokens, path_valid = None, None
            intent_tokens = state_tokens
            route_component = self._mlp_map_route(rotated_map, map_valid)

        # ---- topology prior (D1-3) ----
        topology_tokens = None
        if self.use_topology:
            local_lane_points = self._rotate_topology(frame, augmentation)
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
                _compatible_mask,
                _fallback_mask,
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
            graph_input = self.topology_norm(
                intent_tokens + self.topology_residual_scale * topology_context
            ) * trajectory_valid[..., None]
            (
                edge_features,
                edge_weights,
                pair_valid,
                node_valid,
                _mean_merge_pair_score,
            ) = self._vehicle_edge_features_v2(
                rotated, trajectory_valid, topology_attention
            )
        else:
            graph_input = self.topology_norm(intent_tokens) * trajectory_valid[
                ..., None
            ]
            # Geometry/motion-only edge features (same-lane / conflict terms are
            # zeroed because self.use_topology is False).
            (
                edge_features,
                edge_weights,
                pair_valid,
                node_valid,
            ) = self._vehicle_edge_features(
                rotated, trajectory_valid, topology_attention=None
            )

        # ---- route-conditioned edge relevance (R2) ----
        if self.route_cond_edge:
            edge_weights = self._apply_route_edge_correction(
                edge_weights, edge_features, route_context
            )

        # ---- spatiotemporal interaction (D1-1) ----
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
        actor_valid = trajectory_valid.any(dim=-1).clone()
        actor_valid[:, 0] = True
        ego = actor_features[:, :1]
        social_context, _ = self.social_attention(ego, actor_features, actor_valid)
        ego_component = ego.squeeze(1)
        social_component = social_context.squeeze(1)

        # ---- route slot (goal-side pooling) ----
        if self.use_route:
            ego_route_tokens = route_tokens[:, 0]
            ego_route_valid = path_valid[:, 0]
            if self.use_topology:
                last_key_mask = self._last_query_mask(
                    topology_key_mask, trajectory_valid
                )
                ego_topology_valid = last_key_mask[:, 0]
                route_base, _ = self.goal_attention(
                    ego, ego_route_tokens, ego_route_valid
                )
                topology_goal, _ = self.goal_topology_attention(
                    route_base, topology_tokens, ego_topology_valid
                )
                base_normalized = self.route_topology_norm(route_base)
                fused_normalized = self.route_topology_norm(
                    route_base + self.goal_residual_scale * topology_goal
                )
                route_context_final = route_base + fused_normalized - base_normalized
            else:
                route_context_final, _ = self.goal_attention(
                    ego, ego_route_tokens, ego_route_valid
                )
            route_component = route_context_final.squeeze(1)

        # ---- output (D1-4 structured slots) ----
        if self.use_slots:
            return StructuredLatent(
                z_ego=self.ego_projection(ego_component),
                z_social=self.social_projection(social_component),
                z_route=self.route_projection(route_component),
            )
        combined = torch.cat([ego_component, social_component, route_component], dim=-1)
        return self.mlp_output(combined)

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        out = self.forward_tokens(observations)
        if isinstance(out, StructuredLatent):
            return out.tensor
        return out


__all__ = ["IncrementalTopoEncoder"]
