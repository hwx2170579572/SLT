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
defers to ``super()`` by default and preserves the hold35k path.  The explicit
``use_incremental_slots`` option keeps the same incremental non-slot path while
selecting the structured 32/64/32 output heads, for a structure-only ablation.
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
    * ``use_incremental_slots`` -- when True with all other switches enabled,
      bypass the default parent V2 full forward and apply only the structured
      output heads after the incremental route/topology/interaction path.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        topology_graph,
        use_route: bool = True,
        use_topology: bool = True,
        use_slots: bool = True,
        use_incremental_slots: bool = False,
        use_route_reachability: bool = False,
        topology_lane_ids: tuple[str, ...] | None = None,
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
        self.use_incremental_slots = bool(use_incremental_slots)
        self.use_route_reachability = bool(use_route_reachability)
        self.topology_lane_ids = tuple(str(item) for item in (topology_lane_ids or ()))
        if self.use_incremental_slots and not self.use_slots:
            raise ValueError("use_incremental_slots requires use_slots=True")
        if self.use_route_reachability:
            if not self.use_route or not self.use_topology:
                raise ValueError(
                    "use_route_reachability requires use_route=True and use_topology=True"
                )
            route_space = observation_space.spaces.get("route_reachability")
            if route_space is None:
                raise ValueError(
                    "route-aware Topo requires observation key 'route_reachability'"
                )
            expected_nodes = int(self.topology_node_mask.shape[0])
            if tuple(route_space.shape) != (expected_nodes,):
                raise ValueError(
                    "route_reachability observation must align with topology node capacity "
                    f"{expected_nodes}, got {tuple(route_space.shape)}"
                )
            valid_node_count = int(self.topology_node_mask.sum().item())
            if len(self.topology_lane_ids) != valid_node_count:
                raise ValueError(
                    "topology_lane_ids must identify every valid topology node: "
                    f"{len(self.topology_lane_ids)} != {valid_node_count}"
                )
            expected_mask = torch.arange(expected_nodes) < valid_node_count
            if not torch.equal(self.topology_node_mask.cpu(), expected_mask):
                raise ValueError(
                    "Topology valid nodes must be a contiguous prefix before padding"
                )
            if valid_node_count <= 0 or len(set(self.topology_lane_ids)) != valid_node_count:
                raise ValueError("Route-aware topology requires unique, nonempty lane IDs")
        self.behavior_diagnostics_enabled = False
        self.behavior_diagnostics_sample_every = 256
        self.diagnostic_source = "unspecified"
        self._diagnostic_forward_index = 0
        self._diagnostic_sampling_origin = 0
        self._diagnostic_sample_index = 0
        self._diagnostic_sample_forward_index = 0
        self._diagnostic_sample_source_code = 0
        self._diagnostic_sample_batch_size = 0
        self._diagnostic_sample_grad_enabled = False
        self._diagnostic_current_forward_sampled = False
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
        diagnostic_sampled = self._begin_diagnostic_forward()
        batch_size = int(observations["trajectory"].shape[0])
        if (
            self.use_route
            and self.use_topology
            and self.use_slots
            and not self.use_incremental_slots
        ):
            # Full configuration == the hold35k encoder, verbatim.
            output = super().forward_tokens(observations)
            self._stamp_diagnostic_metadata(
                diagnostic_sampled, batch_size=batch_size
            )
            return output
        return self._forward_incremental(
            observations,
            diagnostic_sampled=diagnostic_sampled,
        )

    def _forward_incremental(
        self,
        observations: Mapping[str, Tensor],
        *,
        diagnostic_sampled: bool = False,
    ):
        trajectories = observations["trajectory"].float()
        batch_size = int(trajectories.shape[0])
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
                rotated,
                trajectory_valid,
                topology_attention,
                collect_diagnostics=diagnostic_sampled
                and self.behavior_diagnostics_enabled,
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
        route_reachability_state: dict[str, Tensor] | None = None
        route_reachability_diagnostics: dict[str, Tensor] | None = None
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
                if self.use_route_reachability:
                    (
                        goal_topology_valid,
                        route_reachability_state,
                    ) = self._route_reachability_goal_mask(
                        ego_topology_valid,
                        observations.get("route_reachability"),
                    )
                    topology_goal, goal_topology_weights = (
                        self.goal_topology_attention(
                            route_base, topology_tokens, goal_topology_valid
                        )
                    )
                else:
                    topology_goal, _ = self.goal_topology_attention(
                        route_base, topology_tokens, ego_topology_valid
                    )
                base_normalized = self.route_topology_norm(route_base)
                fused_normalized = self.route_topology_norm(
                    route_base + self.goal_residual_scale * topology_goal
                )
                if self.use_route_reachability:
                    goal_delta = fused_normalized - base_normalized
                    route_context_final = route_base + torch.where(
                        route_reachability_state["has_legal_candidate"][:, None, None],
                        goal_delta,
                        torch.zeros_like(goal_delta),
                    )
                    if diagnostic_sampled and self.behavior_diagnostics_enabled:
                        route_reachability_diagnostics = (
                            self._route_reachability_diagnostics(
                                route_reachability_state,
                                goal_topology_weights,
                                route_context_final - route_base,
                            )
                        )
                else:
                    # Keep the legacy arithmetic and readout path unchanged.
                    route_context_final = route_base + fused_normalized - base_normalized
            else:
                route_context_final, _ = self.goal_attention(
                    ego, ego_route_tokens, ego_route_valid
                )
            route_component = route_context_final.squeeze(1)

        # ---- output (D1-4 structured slots) ----
        if self.use_slots:
            output = StructuredLatent(
                z_ego=self.ego_projection(ego_component),
                z_social=self.social_projection(social_component),
                z_route=self.route_projection(route_component),
            )
        else:
            combined = torch.cat(
                [ego_component, social_component, route_component], dim=-1
            )
            output = self.mlp_output(combined)

        if diagnostic_sampled and self.behavior_diagnostics_enabled:
            self._collect_incremental_diagnostics(
                trajectory_valid=trajectory_valid,
                topology_attention=topology_attention if self.use_topology else None,
                compatible_mask=compatible_mask if self.use_topology else None,
                fallback_mask=fallback_mask if self.use_topology else None,
                ego_component=ego_component,
                social_component=social_component,
                route_component=route_component,
                output=output if isinstance(output, StructuredLatent) else None,
                batch_size=batch_size,
                route_reachability=route_reachability_diagnostics,
            )
        self._stamp_diagnostic_metadata(
            diagnostic_sampled, batch_size=batch_size
        )
        return output

    def configure_diagnostics(
        self,
        enabled: bool,
        sample_every: int = 256,
        *,
        source: str = "unspecified",
    ) -> None:
        """Enable low-frequency diagnostics from already-computed forward tensors.

        ``source`` should identify the caller, e.g. ``train_replay`` or
        ``eval_policy``. This method does not run a forward pass or alter model
        parameters, inputs, or random-number state.
        """

        interval = int(sample_every)
        if interval <= 0:
            raise ValueError("sample_every must be a positive integer")
        self.behavior_diagnostics_enabled = bool(enabled)
        self.behavior_diagnostics_sample_every = interval
        self.diagnostic_source = str(source)
        self._diagnostic_sampling_origin = self._diagnostic_forward_index

    @property
    def behavior_diagnostics_source_code(self) -> int:
        return {
            "train_replay": 1,
            "eval_policy": 2,
        }.get(self.diagnostic_source, 0)

    @property
    def diagnostic_sample_index(self) -> int:
        """Monotone model-lifetime ID of the last published forward snapshot."""

        return int(self._diagnostic_sample_index)

    def diagnostic_parameter_groups(self) -> dict[str, tuple[str, ...]]:
        """Parameter-name prefixes for post-backward diagnostics.

        The prefixes map to this extractor's registered parameters only; a
        consumer should report missing, zero and nonzero gradients separately.
        """

        return {
            "encoder_all": ("",),
            "topology_core": ("topology_encoder", "topology_attention"),
            "topology_fusion_goal": (
                "topology_norm",
                "topology_residual_scale",
                "goal_topology_attention",
                "goal_residual_scale",
                "route_topology_norm",
            ),
            "topology_vehicle_relations": ("vehicle_layers",),
            "slot_ego": ("ego_projection",),
            "slot_social": ("social_projection",),
            "slot_route": ("route_projection",),
            "route": (
                "map_encoder",
                "route_attention",
                "route_norm",
                "goal_attention",
                "route_bias_raw",
                "heading_bias_raw",
                "route_gate_mlp",
                "route_gate_bias",
                "route_edge_mlp",
            ),
        }

    def _begin_diagnostic_forward(self) -> bool:
        self._diagnostic_forward_index += 1
        sampled = bool(
            self.behavior_diagnostics_enabled
            and (self._diagnostic_forward_index - self._diagnostic_sampling_origin - 1)
            % self.behavior_diagnostics_sample_every
            == 0
        )
        self._diagnostic_current_forward_sampled = sampled
        return sampled

    def _stamp_diagnostic_metadata(
        self,
        sampled: bool,
        *,
        batch_size: int,
    ) -> None:
        if not self.behavior_diagnostics_enabled:
            return
        if sampled:
            self._diagnostic_sample_index += 1
            self._diagnostic_sample_forward_index = self._diagnostic_forward_index
            self._diagnostic_sample_source_code = (
                self.behavior_diagnostics_source_code
            )
            self._diagnostic_sample_batch_size = int(batch_size)
            self._diagnostic_sample_grad_enabled = torch.is_grad_enabled()
        device = self.topology_lane_points.device
        self._diagnostics.update(
            {
                "diagnostic_forward_index": torch.tensor(
                    self._diagnostic_forward_index, device=device, dtype=torch.float32
                ),
                "diagnostic_sample_forward_index": torch.tensor(
                    self._diagnostic_sample_forward_index,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_index": torch.tensor(
                    self._diagnostic_sample_index, device=device, dtype=torch.float32
                ),
                "diagnostic_sample_count": torch.tensor(
                    self._diagnostic_sample_index, device=device, dtype=torch.float32
                ),
                "diagnostic_sample_age_forwards": torch.tensor(
                    self._diagnostic_forward_index
                    - self._diagnostic_sample_forward_index,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sampled": torch.tensor(
                    float(sampled), device=device, dtype=torch.float32
                ),
                "diagnostic_sample_every": torch.tensor(
                    self.behavior_diagnostics_sample_every,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_source_code": torch.tensor(
                    self.behavior_diagnostics_source_code,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_source_code": torch.tensor(
                    self._diagnostic_sample_source_code,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_batch_size": torch.tensor(
                    self._diagnostic_sample_batch_size,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_grad_enabled": torch.tensor(
                    float(self._diagnostic_sample_grad_enabled),
                    device=device,
                    dtype=torch.float32,
                ),
            }
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        weights = mask.to(values.dtype)
        count = weights.sum()
        mean = (values * weights).sum() / count.clamp_min(1.0)
        return mean, count

    def _route_reachability_goal_mask(
        self,
        query_mask: Tensor,
        route_labels: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Filter only ego goal-topology keys using tri-state route labels.

        Unknown labels never become ineligible. If no known eligible lane exists,
        the original geometric query is retained as a safe attention mask, while
        the caller bypasses its topology-goal residual for that sample.
        """
        if route_labels is None:
            raise ValueError(
                "route-aware Topo requires observation key 'route_reachability'"
            )
        labels = torch.as_tensor(
            route_labels, device=query_mask.device, dtype=torch.float32
        )
        batch_size, node_count = query_mask.shape
        expected = (batch_size, int(self.topology_node_mask.shape[0]))
        if tuple(labels.shape) != expected or node_count != expected[1]:
            raise ValueError(
                "route_reachability must have shape "
                f"{expected}, got {tuple(labels.shape)}"
            )

        valid_nodes = self.topology_node_mask.to(query_mask.device)[None, :].expand(
            batch_size, -1
        )
        finite = torch.isfinite(labels)
        known = finite & (labels >= 0.0) & valid_nodes
        unknown = (~finite | (labels < 0.0)) & valid_nodes
        legal = finite & (labels > 0.5) & valid_nodes
        base_query = query_mask.to(torch.bool) & valid_nodes
        intersection = base_query & legal
        has_intersection = intersection.any(dim=-1)
        has_legal_candidate = legal.any(dim=-1)
        expanded = (~has_intersection) & has_legal_candidate
        bypassed = ~has_legal_candidate

        selected = torch.where(has_intersection[:, None], intersection, legal)
        # No known legal route node: preserve a finite mask for attention, then
        # the caller zeros the topology-goal delta for those batch elements.
        selected = torch.where(bypassed[:, None], base_query, selected)
        safe_fallback = ~selected.any(dim=-1)
        selected = torch.where(
            safe_fallback[:, None], valid_nodes, selected
        )
        state = {
            "legal": legal,
            "known": known,
            "unknown": unknown,
            "intersection": intersection,
            "expanded": expanded,
            "bypassed": bypassed,
            "has_legal_candidate": has_legal_candidate,
            "safe_fallback": safe_fallback,
        }
        return selected, state

    def _route_reachability_diagnostics(
        self,
        state: dict[str, Tensor],
        goal_attention_weights: Tensor,
        goal_delta: Tensor,
    ) -> dict[str, Tensor]:
        """Summarize route-label coverage and its actual goal-readout effect."""
        legal = state["legal"]
        known = state["known"]
        unknown = state["unknown"]
        active = state["has_legal_candidate"]
        batch_size = max(1, int(legal.shape[0]))
        denominator = torch.tensor(
            float(batch_size), device=legal.device, dtype=torch.float32
        )
        active_float = active.to(torch.float32)
        legal_attention_mass = (
            goal_attention_weights * legal[:, None, :].to(goal_attention_weights.dtype)
        ).sum(dim=-1).mean(dim=-1)
        delta_norm = torch.linalg.vector_norm(
            goal_delta.detach(), dim=-1
        ).squeeze(1)
        legal_mass_active, legal_mass_count = self._masked_mean(
            legal_attention_mass, active
        )
        delta_norm_active, delta_active_count = self._masked_mean(
            delta_norm, active
        )
        valid_count = self.topology_node_mask.sum().to(torch.float32)
        padding_count = torch.tensor(
            float(self.topology_node_mask.numel()),
            device=legal.device,
            dtype=torch.float32,
        ) - valid_count
        return {
            "route_reachability_candidate_count_mean": legal.sum(dim=-1)
            .to(torch.float32)
            .mean(),
            "route_reachability_known_node_count_mean": known.sum(dim=-1)
            .to(torch.float32)
            .mean(),
            "route_reachability_unknown_node_count_mean": unknown.sum(dim=-1)
            .to(torch.float32)
            .mean(),
            "route_reachability_padding_node_count": padding_count,
            "route_reachability_query_intersection_count_mean": state[
                "intersection"
            ]
            .sum(dim=-1)
            .to(torch.float32)
            .mean(),
            "route_reachability_goal_expand_fraction": state["expanded"]
            .to(torch.float32)
            .sum()
            / denominator,
            "route_reachability_goal_bypass_fraction": state["bypassed"]
            .to(torch.float32)
            .sum()
            / denominator,
            "route_reachability_safe_mask_fallback_fraction": state[
                "safe_fallback"
            ]
            .to(torch.float32)
            .sum()
            / denominator,
            "route_reachability_goal_active_sample_count": active_float.sum(),
            "route_reachability_goal_legal_attention_mass_active": legal_mass_active,
            "route_reachability_goal_legal_attention_valid_count": legal_mass_count,
            "route_reachability_goal_delta_norm_mean": delta_norm.mean(),
            "route_reachability_goal_delta_norm_active": delta_norm_active,
            "route_reachability_goal_delta_active_sample_count": delta_active_count,
        }

    @staticmethod
    def _safe_correlation(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        left_var = left_centered.square().mean()
        right_var = right_centered.square().mean()
        denominator = torch.sqrt(left_var * right_var)
        valid = (denominator > 1e-8) & (left.numel() >= 2)
        corr = (left_centered * right_centered).mean() / denominator.clamp_min(1e-8)
        return torch.where(valid, corr, torch.zeros_like(corr)), valid.to(left.dtype)

    def _collect_incremental_diagnostics(
        self,
        *,
        trajectory_valid: Tensor,
        topology_attention: Tensor | None,
        compatible_mask: Tensor | None,
        fallback_mask: Tensor | None,
        ego_component: Tensor,
        social_component: Tensor,
        route_component: Tensor,
        output: StructuredLatent | None,
        batch_size: int,
        route_reachability: dict[str, Tensor] | None = None,
    ) -> None:
        device = trajectory_valid.device
        valid_queries = trajectory_valid.to(torch.float32)
        query_count = valid_queries.sum()
        diagnostics: dict[str, Tensor] = {
            "diagnostic_valid_query_count": query_count.detach(),
            "diagnostic_batch_size": torch.tensor(
                batch_size, device=device, dtype=torch.float32
            ),
            "diagnostic_batch_variance_valid": torch.tensor(
                float(batch_size >= 2), device=device, dtype=torch.float32
            ),
            "context_ego_rms": ego_component.detach().square().mean().sqrt(),
            "context_social_rms": social_component.detach().square().mean().sqrt(),
            "context_route_rms": route_component.detach().square().mean().sqrt(),
            "context_ego_batch_std_mean": ego_component.detach().std(
                dim=0, unbiased=False
            ).mean(),
            "context_social_batch_std_mean": social_component.detach().std(
                dim=0, unbiased=False
            ).mean(),
            "context_route_batch_std_mean": route_component.detach().std(
                dim=0, unbiased=False
            ).mean(),
            "topology_residual_scale": self.topology_residual_scale.detach(),
            "goal_residual_scale": self.goal_residual_scale.detach(),
            "route_bias_weight": self.route_bias_weight.detach(),
            "heading_bias_weight": self.heading_bias_weight.detach(),
        }
        if topology_attention is not None:
            assert compatible_mask is not None and fallback_mask is not None
            entropy_per_query = -(
                topology_attention.clamp_min(1e-12).log() * topology_attention
            ).sum(dim=-1)
            compatible_mass_per_query = (
                topology_attention
                * compatible_mask.to(topology_attention.dtype)
            ).sum(dim=-1)
            compatible_queries = compatible_mask.any(dim=-1) & trajectory_valid
            effective_lane_count = entropy_per_query.exp()
            mean_effective_lanes, _ = self._masked_mean(
                effective_lane_count, trajectory_valid
            )
            route_compatible_mass, _ = self._masked_mean(
                compatible_mass_per_query, trajectory_valid
            )
            compatible_lane_count_mean, _ = self._masked_mean(
                compatible_mask.to(torch.float32).sum(dim=-1), trajectory_valid
            )
            fallback_count = (fallback_mask & trajectory_valid).sum()
            diagnostics.update(
                {
                    "topology_attention_entropy": self._masked_mean(
                        entropy_per_query, trajectory_valid
                    )[0],
                    "topology_effective_lanes": mean_effective_lanes,
                    "route_compatible_attention_mass": route_compatible_mass,
                    "route_compatible_lane_count_mean": compatible_lane_count_mean,
                    "topology_fallback_rate": fallback_count.to(torch.float32)
                    / query_count.clamp_min(1.0),
                    "topology_fallback_query_count": fallback_count.to(
                        torch.float32
                    ),
                    "topology_compatible_query_count": (
                        compatible_queries & trajectory_valid
                    ).sum().to(torch.float32),
                    "topology_valid_lane_count": self.topology_node_mask.sum().to(
                        torch.float32
                    ),
                }
            )
            for key, value in self._last_edge_diagnostics.items():
                diagnostics[key] = value
        if output is not None:
            slots = (output.z_ego, output.z_social, output.z_route)
            slot_names = ("ego", "social", "route")
            slot_dims: list[int] = []
            slot_energy: list[Tensor] = []
            slot_rms: list[Tensor] = []
            for name, slot in zip(slot_names, slots):
                detached = slot.detach()
                slot_dims.append(int(detached.shape[-1]))
                per_sample_energy = detached.square().mean(dim=-1)
                slot_energy.append(per_sample_energy)
                rms = detached.square().mean().sqrt()
                # Across-batch std is a collapse indicator only when the
                # actual batch contains at least two independent examples;
                # diagnostic_batch_variance_valid records that boundary.
                std_mean = detached.std(dim=0, unbiased=False).mean()
                slot_rms.append(rms)
                diagnostics[f"{name}_slot_rms"] = rms
                diagnostics[f"{name}_slot_batch_std_mean"] = std_mean
                diagnostics[f"{name}_slot_within_sample_std_mean"] = detached.std(
                    dim=-1, unbiased=False
                ).mean()
                diagnostics[f"{name}_slot_dim"] = torch.tensor(
                    int(detached.shape[-1]), device=device, dtype=torch.float32
                )
                projection = getattr(self, f"{name}_projection")
                diagnostics[f"{name}_projection_weight_rms"] = (
                    projection.weight.detach().square().mean().sqrt()
                )
                diagnostics[f"{name}_projection_parameter_count"] = torch.tensor(
                    sum(parameter.numel() for parameter in projection.parameters()),
                    device=device,
                    dtype=torch.float32,
                )
                diagnostics[f"{name}_projection_active"] = torch.ones(
                    (), device=device, dtype=torch.float32
                )
            total_energy = torch.stack(slot_energy, dim=1).sum(dim=1).clamp_min(1e-12)
            for name, energy in zip(slot_names, slot_energy):
                diagnostics[f"{name}_slot_energy_share"] = (
                    energy / total_energy
                ).mean()
            ego_social_corr, corr_valid = self._safe_correlation(
                slot_energy[0], slot_energy[1]
            )
            social_route_corr, social_route_corr_valid = self._safe_correlation(
                slot_energy[1], slot_energy[2]
            )
            ego_route_corr, ego_route_corr_valid = self._safe_correlation(
                slot_energy[0], slot_energy[2]
            )
            all_correlations_valid = (
                (corr_valid > 0)
                & (social_route_corr_valid > 0)
                & (ego_route_corr_valid > 0)
            )
            correlations_valid_flag = (
                all_correlations_valid
                if batch_size >= 2
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            diagnostics.update(
                {
                    "slot_sample_energy_correlation_valid": correlations_valid_flag,
                    "slot_batch_variance_valid": torch.tensor(
                        float(batch_size >= 2), device=device, dtype=torch.float32
                    ),
                    "slot_output_total_dim": torch.tensor(
                        sum(slot_dims), device=device, dtype=torch.float32
                    ),
                    "mlp_output_head_active": torch.zeros(
                        (), device=device, dtype=torch.float32
                    ),
                }
            )
            if batch_size >= 2 and bool(all_correlations_valid):
                # These correlations are across-example slot energies, not
                # cosine similarities between unequal-dimensional vectors.
                diagnostics.update(
                    {
                        "slot_sample_energy_ego_social_corr": ego_social_corr,
                        "slot_sample_energy_social_route_corr": social_route_corr,
                        "slot_sample_energy_ego_route_corr": ego_route_corr,
                    }
                )
        else:
            diagnostics["mlp_output_head_active"] = torch.ones(
                (), device=device, dtype=torch.float32
            )
            for name in ("ego", "social", "route"):
                diagnostics[f"{name}_projection_active"] = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
        if route_reachability is not None:
            diagnostics.update(route_reachability)
        self._diagnostics.update(
            {key: value.detach() for key, value in diagnostics.items()}
        )

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        out = self.forward_tokens(observations)
        if isinstance(out, StructuredLatent):
            return out.tensor
        return out


__all__ = ["IncrementalTopoEncoder"]
