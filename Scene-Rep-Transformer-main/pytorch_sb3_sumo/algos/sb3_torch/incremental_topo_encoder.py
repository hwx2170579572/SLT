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

from contextlib import contextmanager
from typing import Iterator, Mapping

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
        use_route_conflict_timing: bool = False,
        use_topology_actor_intent: bool | None = None,
        use_topology_relations: bool | None = None,
        use_topology_goal: bool | None = None,
        use_parameter_matched_nonlinear_slots: bool = False,
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
        self.use_route_conflict_timing = bool(use_route_conflict_timing)
        default_topology_branch = self.use_topology
        self.use_topology_actor_intent = (
            default_topology_branch
            if use_topology_actor_intent is None
            else bool(use_topology_actor_intent)
        )
        self.use_topology_relations = (
            default_topology_branch
            if use_topology_relations is None
            else bool(use_topology_relations)
        )
        self.use_topology_goal = (
            default_topology_branch
            if use_topology_goal is None
            else bool(use_topology_goal)
        )
        self.use_parameter_matched_nonlinear_slots = bool(
            use_parameter_matched_nonlinear_slots
        )
        self.topology_lane_ids = tuple(str(item) for item in (topology_lane_ids or ()))
        if self.use_incremental_slots and not self.use_slots:
            raise ValueError("use_incremental_slots requires use_slots=True")
        if self.use_parameter_matched_nonlinear_slots:
            if not self.use_slots or not self.use_incremental_slots:
                raise ValueError(
                    "parameter-matched nonlinear slots require use_slots=True and "
                    "use_incremental_slots=True"
                )
            if hidden_dim != 128 or features_dim != 128:
                raise ValueError(
                    "parameter-matched nonlinear slots are defined for hidden_dim=128 "
                    "and features_dim=128"
                )
        if not self.use_topology and any(
            (
                self.use_topology_actor_intent,
                self.use_topology_relations,
                self.use_topology_goal,
            )
        ):
            raise ValueError(
                "Topo sub-branches cannot be enabled when use_topology=False"
            )
        if self.use_route_reachability:
            if not self.use_route or not self.use_topology_goal:
                raise ValueError(
                    "use_route_reachability requires use_route=True and "
                    "use_topology_goal=True"
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
        if self.use_route_conflict_timing:
            if not self.use_route:
                raise ValueError("route conflict timing requires use_route=True")
            conflict_space = observation_space.spaces.get("conflict_timing")
            expected_shape = (int(self.actor_count), 20)
            if conflict_space is None or tuple(conflict_space.shape) != expected_shape:
                raise ValueError(
                    "route conflict timing requires observation key 'conflict_timing' "
                    f"with shape {expected_shape}"
                )
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
        self._diagnostic_active_site = "rollout_policy"
        self._diagnostic_active_update_index: int | None = None
        self._diagnostic_force_sample = False
        self._diagnostic_site_forward_counts: dict[str, int] = {}
        self._diagnostic_sample_site_code = 0
        self._diagnostic_sample_site_forward_index = 0
        self._diagnostic_active_site_forward_index = 0
        self._diagnostic_sample_update_index = -1
        # Temporary, non-checkpointed context used only for bounded same-state
        # policy probes. It is inactive during ordinary training/evaluation.
        self._policy_shadow_active = False
        self._policy_shadow_probe_name: str | None = None
        self._policy_shadow_capture: dict[str, Tensor] | None = None
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
        self.route_conflict_timing_encoder = None
        if self.use_route_conflict_timing:
            # Isolate initialization from the process RNG stream so adding this
            # optional branch does not shift the subsequent SAC actor/critic
            # initialization relative to the matched STRT parent.
            with torch.random.fork_rng(devices=[]):
                self.route_conflict_timing_encoder = nn.Sequential(
                    nn.Linear(20, 32),
                    nn.ReLU(),
                    nn.Linear(32, hidden_dim),
                )
                self.route_conflict_timing_encoder.apply(keras_initialize_linear)
                final = self.route_conflict_timing_encoder[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        self.nonlinear_slot_projections = None
        if self.use_parameter_matched_nonlinear_slots:
            # Keep the legacy global RNG stream identical for the SAC actor and
            # critic initialized after this extractor. These new head weights
            # draw from an isolated fork and are not inserted before shared
            # encoder modules in the registration/init order.
            with torch.random.fork_rng(devices=[]):
                self.nonlinear_slot_projections = nn.ModuleDict(
                    {
                        "ego": nn.Sequential(
                            nn.Linear(128, 132),
                            nn.ReLU(),
                            nn.Linear(132, 32),
                            nn.ReLU(),
                        ),
                        "social": nn.Sequential(
                            nn.Linear(128, 120),
                            nn.ReLU(),
                            nn.Linear(120, 64),
                            nn.ReLU(),
                        ),
                        "route": nn.Sequential(
                            nn.Linear(128, 132),
                            nn.ReLU(),
                            nn.Linear(132, 32),
                            nn.ReLU(),
                        ),
                    }
                )
                self.nonlinear_slot_projections.apply(keras_initialize_linear)

    @property
    def active_readout_parameter_count(self) -> int:
        if self.use_parameter_matched_nonlinear_slots:
            assert self.nonlinear_slot_projections is not None
            return sum(parameter.numel() for parameter in self.nonlinear_slot_projections.parameters())
        if self.use_slots:
            return sum(
                parameter.numel()
                for module in (
                    self.ego_projection,
                    self.social_projection,
                    self.route_projection,
                )
                for parameter in module.parameters()
            )
        return sum(parameter.numel() for parameter in self.mlp_output.parameters())

    @property
    def active_readout_type(self) -> str:
        if self.use_parameter_matched_nonlinear_slots:
            return "parameter_matched_nonlinear_3slot"
        return "linear_3slot" if self.use_slots else "joint_mlp"

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
    _POLICY_SHADOW_PROBES = (
        "st_spatial_off",
        "st_temporal_current_only",
        "st_social_ego_only",
        "rt_intent_injection_off",
        "rt_route_readout_zero",
        "route_conflict_off",
        "route_conflict_times_off",
        "topology_actor_intent_off",
        "topology_relations_off",
        "topology_goal_off",
        "route_reachability_off",
        "slot_ego_zero",
        "slot_social_zero",
        "slot_route_zero",
    )

    def policy_shadow_probe_applicability(self) -> list[dict[str, object]]:
        """Stable, explicit applicability map for the single-branch probe suite."""
        route_injection_active = bool(
            self.use_route and not self.route_late and not self.route_cond_edge
        )
        slots_active = bool(self.use_slots)
        active = {
            "st_spatial_off": True,
            "st_temporal_current_only": True,
            "st_social_ego_only": True,
            "rt_intent_injection_off": route_injection_active,
            "rt_route_readout_zero": bool(self.use_route),
            "route_conflict_off": bool(self.use_route_conflict_timing),
            "route_conflict_times_off": bool(self.use_route_conflict_timing),
            "topology_actor_intent_off": bool(self.use_topology_actor_intent),
            "topology_relations_off": bool(self.use_topology_relations),
            "topology_goal_off": bool(self.use_topology_goal),
            "route_reachability_off": bool(self.use_route_reachability),
            "slot_ego_zero": slots_active,
            "slot_social_zero": slots_active,
            "slot_route_zero": slots_active,
        }
        return [
            {
                "probe_name": name,
                "applicable": active[name],
                "invalid_reason": None if active[name] else "branch_inactive",
            }
            for name in self._POLICY_SHADOW_PROBES
        ]

    @contextmanager
    def policy_shadow_probe(self, name: str | None) -> Iterator[dict[str, Tensor]]:
        """Temporarily select one intervention without mutating model switches.

        ``None`` captures the deterministic baseline. ``__noop__`` is reserved
        for equivalence tests. The caller must run inference under no-grad and
        restore model/runtime state around the context.
        """
        if self._policy_shadow_active:
            raise RuntimeError("policy shadow probe contexts cannot be nested")
        if name is not None and name not in self._POLICY_SHADOW_PROBES and name != "__noop__":
            raise ValueError(f"unknown policy shadow probe: {name}")
        capture: dict[str, Tensor] = {}
        previous = (
            self._policy_shadow_active,
            self._policy_shadow_probe_name,
            self._policy_shadow_capture,
        )
        self._policy_shadow_active = True
        self._policy_shadow_probe_name = name
        self._policy_shadow_capture = capture
        try:
            yield capture
        finally:
            (
                self._policy_shadow_active,
                self._policy_shadow_probe_name,
                self._policy_shadow_capture,
            ) = previous

    def _capture_policy_shadow_output(self, output) -> None:
        capture = self._policy_shadow_capture
        if capture is None:
            return
        if isinstance(output, StructuredLatent):
            capture["feature_latent"] = output.tensor.detach().clone()
            capture["slot_ego"] = output.z_ego.detach().clone()
            capture["slot_social"] = output.z_social.detach().clone()
            capture["slot_route"] = output.z_route.detach().clone()
        elif isinstance(output, Tensor):
            capture["feature_latent"] = output.detach().clone()

    def forward_tokens(self, observations: Mapping[str, Tensor]):
        if self._policy_shadow_active:
            probe = self._policy_shadow_probe_name
            batch_size = int(observations["trajectory"].shape[0])
            use_parent_forward = (
                probe is None
                and self.use_route
                and self.use_topology
                and self.use_topology_actor_intent
                and self.use_topology_relations
                and self.use_topology_goal
                and self.use_slots
                and not self.use_incremental_slots
            )
            if use_parent_forward:
                output = super().forward_tokens(observations)
            else:
                output = self._forward_incremental(
                    observations,
                    diagnostic_sampled=False,
                    diagnostic_probe=probe,
                )
            self._capture_policy_shadow_output(output)
            return output

        diagnostic_sampled = self._begin_diagnostic_forward()
        batch_size = int(observations["trajectory"].shape[0])
        if (
            self.use_route
            and self.use_topology
            and self.use_topology_actor_intent
            and self.use_topology_relations
            and self.use_topology_goal
            and self.use_slots
            and not self.use_incremental_slots
        ):
            # Full configuration == the hold35k encoder, verbatim.
            output = super().forward_tokens(observations)
            self._stamp_diagnostic_metadata(
                diagnostic_sampled, batch_size=batch_size, output=output
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
        diagnostic_probe: str | None = None,
    ):
        probe = diagnostic_probe
        use_topology_actor_intent = bool(
            self.use_topology_actor_intent and probe != "topology_actor_intent_off"
        )
        use_topology_relations = bool(
            self.use_topology_relations and probe != "topology_relations_off"
        )
        use_topology_goal = bool(
            self.use_topology_goal and probe != "topology_goal_off"
        )
        use_route_reachability = bool(
            self.use_route_reachability and probe != "route_reachability_off"
        )
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
        route_attention_weights = None
        if self.use_route:
            route_attention_capture: dict[str, Tensor] = {}
            route_attention_hook = None
            if diagnostic_sampled and self.behavior_diagnostics_enabled:
                # MaskedCrossAttention already returns these weights. A temporary
                # read-only hook captures that existing output without another
                # module call or a change to the production forward path.
                def _capture_route_weights(module, inputs, output):
                    if isinstance(output, tuple) and len(output) > 1:
                        route_attention_capture["weights"] = output[1]

                route_attention_hook = self.route_attention.register_forward_hook(
                    _capture_route_weights
                )
            try:
                route_tokens, path_valid, route_context = self._encode_routes(
                    rotated_map, map_valid, current_state
                )
            finally:
                if route_attention_hook is not None:
                    route_attention_hook.remove()
            if "weights" in route_attention_capture:
                route_attention_weights = route_attention_capture["weights"].reshape(
                    batch_size,
                    self.actor_count,
                    self.paths_per_actor,
                )
            if probe == "rt_intent_injection_off":
                # Keep map encoding and route readout intact while removing the
                # route-conditioned residual from all interaction tokens.
                intent_tokens = state_tokens
            elif self.route_late or self.route_cond_edge:
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
        topology_attention = None
        topology_key_mask = None
        compatible_mask = None
        fallback_mask = None
        topology_actor_diagnostics: dict[str, Tensor] = {}
        topology_active = self.use_topology and any(
            (
                use_topology_actor_intent,
                use_topology_relations,
                use_topology_goal,
            )
        )
        if topology_active:
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
            compute_topology_context = (
                use_topology_actor_intent or use_topology_relations
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
                compute_context=compute_topology_context,
            )
            if use_topology_actor_intent:
                assert topology_context is not None
                graph_input = self.topology_norm(
                    intent_tokens + self.topology_residual_scale * topology_context
                ) * trajectory_valid[..., None]
            else:
                # Keep the exact ST-RT actor path when only the goal branch is
                # enabled.  This is intentionally the same normalization as
                # the topology-disabled incremental baseline.
                graph_input = self.topology_norm(intent_tokens) * trajectory_valid[
                    ..., None
                ]
            if (
                use_topology_actor_intent
                and diagnostic_sampled
                and self.behavior_diagnostics_enabled
            ):
                topology_actor_diagnostics = (
                    self._topology_actor_postnorm_diagnostics(
                        intent_tokens=intent_tokens,
                        graph_input=graph_input,
                        trajectory_valid=trajectory_valid,
                    )
                )
            if use_topology_relations:
                assert topology_attention is not None
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
                    topology_key_mask=(
                        topology_key_mask
                        if diagnostic_sampled and self.behavior_diagnostics_enabled
                        else None
                    ),
                    collect_diagnostics=diagnostic_sampled
                    and self.behavior_diagnostics_enabled,
                )
            else:
                (
                    edge_features,
                    edge_weights,
                    pair_valid,
                    node_valid,
                ) = self._vehicle_edge_features(
                    rotated,
                    trajectory_valid,
                    topology_attention=None,
                    include_topology_relations=False,
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
                rotated,
                trajectory_valid,
                topology_attention=None,
                include_topology_relations=False,
            )

        # ---- route-conditioned edge relevance (R2) ----
        if self.route_cond_edge:
            edge_weights = self._apply_route_edge_correction(
                edge_weights, edge_features, route_context
            )

        # ---- spatiotemporal interaction (D1-1) ----
        graph_features = graph_input.permute(0, 2, 1, 3)
        if probe != "st_spatial_off":
            for layer in self.vehicle_layers:
                graph_features = layer(
                    graph_features,
                    edge_features,
                    edge_weights,
                    pair_valid,
                    node_valid,
                )
        # Keep the ordinary message-passing output. The optional telemetry below
        # compares these existing tensors; it does not run a counterfactual path.
        graph_sequence = graph_features.permute(0, 2, 1, 3)
        if probe == "st_temporal_current_only":
            actor_features = self._last_valid(graph_sequence, trajectory_valid)
        else:
            actor_features = self.temporal_encoder(graph_sequence, trajectory_valid)
        actor_valid = trajectory_valid.any(dim=-1).clone()
        actor_valid[:, 0] = True
        ego = actor_features[:, :1]
        social_mask = actor_valid
        if probe == "st_social_ego_only":
            social_mask = torch.zeros_like(actor_valid)
            social_mask[:, 0] = True
        social_key_values = actor_features
        route_conflict_diagnostics: dict[str, Tensor] = {}
        if self.use_route_conflict_timing:
            conflict_features = observations["conflict_timing"].float()
            if conflict_features.ndim != 3 or tuple(conflict_features.shape[1:]) != (
                int(self.actor_count), 20
            ):
                raise ValueError(
                    "conflict_timing must have shape [batch, actor_count, 20], "
                    f"got {tuple(conflict_features.shape)}"
                )
            if probe == "route_conflict_times_off":
                conflict_features = conflict_features.clone()
                conflict_features[..., 7:15] = 0.0
            conflict_valid = conflict_features[..., 0] > 0.5
            conflict_valid[:, 0] = False
            conflict_valid = conflict_valid & actor_valid
            if probe == "route_conflict_off":
                conflict_residual = torch.zeros_like(actor_features)
            else:
                assert self.route_conflict_timing_encoder is not None
                conflict_residual = self.route_conflict_timing_encoder(
                    conflict_features
                )
                conflict_residual = conflict_residual * conflict_valid[..., None].to(
                    conflict_residual.dtype
                )
            social_key_values = actor_features + conflict_residual
            if diagnostic_sampled and self.behavior_diagnostics_enabled:
                neighbor_valid = actor_valid.clone()
                neighbor_valid[:, 0] = False
                valid_count = conflict_valid.sum().to(torch.float32)
                candidate_count = neighbor_valid.sum().to(torch.float32)
                metric_mask = conflict_valid[..., None].to(actor_features.dtype)
                metric_denominator = (valid_count * actor_features.shape[-1]).clamp_min(1.0)
                residual_rms = (
                    conflict_residual.detach().square() * metric_mask
                ).sum().div(metric_denominator).clamp_min(0.0).sqrt()
                base_rms = (
                    actor_features.detach().square() * metric_mask
                ).sum().div(metric_denominator).clamp_min(0.0).sqrt()
                route_conflict_diagnostics = {
                    "route_conflict_timing_active": torch.ones(
                        (), device=actor_features.device, dtype=torch.float32
                    ),
                    "route_conflict_timing_valid_actor_count": valid_count,
                    "route_conflict_timing_candidate_actor_count": candidate_count,
                    "route_conflict_timing_coverage_valid": (
                        candidate_count > 0
                    ).to(torch.float32),
                    "route_conflict_timing_coverage_fraction": (
                        valid_count / candidate_count.clamp_min(1.0)
                    ),
                    "route_conflict_timing_delta_rms": residual_rms,
                    "route_conflict_timing_base_rms_on_valid": base_rms,
                    "route_conflict_timing_delta_to_base_rms": (
                        residual_rms / base_rms.clamp_min(1e-12)
                    ),
                    "route_conflict_timing_delta_valid": (valid_count > 0).to(
                        torch.float32
                    ),
                    "route_conflict_timing_delta_to_base_valid": (
                        (valid_count > 0) & (base_rms > 1e-12)
                    ).to(torch.float32),
                }
        social_context, social_attention_weights = self.social_attention(
            ego, social_key_values, social_mask
        )
        ego_component = ego.squeeze(1)
        social_component = social_context.squeeze(1)

        # ---- route slot (goal-side pooling) ----
        route_reachability_state: dict[str, Tensor] | None = None
        route_reachability_diagnostics: dict[str, Tensor] | None = None
        if self.use_route:
            ego_route_tokens = route_tokens[:, 0]
            ego_route_valid = path_valid[:, 0]
            if use_topology_goal:
                last_key_mask = self._last_query_mask(
                    topology_key_mask, trajectory_valid
                )
                ego_topology_valid = last_key_mask[:, 0]
                route_base, _ = self.goal_attention(
                    ego, ego_route_tokens, ego_route_valid
                )
                if use_route_reachability:
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
                    if diagnostic_sampled and self.behavior_diagnostics_enabled:
                        topology_goal, goal_topology_weights = (
                            self.goal_topology_attention(
                                route_base, topology_tokens, ego_topology_valid
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
                if use_route_reachability:
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
            if probe == "rt_route_readout_zero":
                route_component = torch.zeros_like(route_component)

        # ---- output (D1-4 structured slots) ----
        if self.use_slots and self.use_parameter_matched_nonlinear_slots:
            assert self.nonlinear_slot_projections is not None
            z_ego = self.nonlinear_slot_projections["ego"](ego_component)
            z_social = self.nonlinear_slot_projections["social"](social_component)
            z_route = self.nonlinear_slot_projections["route"](route_component)
            if probe == "slot_ego_zero":
                z_ego = torch.zeros_like(z_ego)
            elif probe == "slot_social_zero":
                z_social = torch.zeros_like(z_social)
            elif probe == "slot_route_zero":
                z_route = torch.zeros_like(z_route)
            output = StructuredLatent(
                z_ego=z_ego,
                z_social=z_social,
                z_route=z_route,
            )
        elif self.use_slots:
            z_ego = self.ego_projection(ego_component)
            z_social = self.social_projection(social_component)
            z_route = self.route_projection(route_component)
            if probe == "slot_ego_zero":
                z_ego = torch.zeros_like(z_ego)
            elif probe == "slot_social_zero":
                z_social = torch.zeros_like(z_social)
            elif probe == "slot_route_zero":
                z_route = torch.zeros_like(z_route)
            output = StructuredLatent(
                z_ego=z_ego,
                z_social=z_social,
                z_route=z_route,
            )
        else:
            combined = torch.cat(
                [ego_component, social_component, route_component], dim=-1
            )
            output = self.mlp_output(combined)

        if diagnostic_sampled and self.behavior_diagnostics_enabled:
            extra_diagnostics = self._spatiotemporal_diagnostics(
                graph_input=graph_input,
                graph_output=graph_features.permute(0, 2, 1, 3),
                trajectory_valid=trajectory_valid,
                actor_features=actor_features,
                actor_valid=actor_valid,
                social_attention_weights=social_attention_weights,
                route_valid=path_valid[:, 0] if self.use_route else None,
                route_attention_weights=route_attention_weights,
                goal_attention_weights=(
                    goal_topology_weights
                    if self.use_route and self.use_topology_goal
                    else None
                ),
                goal_topology_valid=(
                    goal_topology_valid
                    if self.use_route and self.use_topology_goal and self.use_route_reachability
                    else (
                        ego_topology_valid
                        if self.use_route and self.use_topology_goal
                        else None
                    )
                ),
                goal_delta=(
                    route_context_final - route_base
                    if self.use_route and self.use_topology_goal
                    else None
                ),
            )
            extra_diagnostics.update(topology_actor_diagnostics)
            extra_diagnostics.update(route_conflict_diagnostics)
            self._collect_incremental_diagnostics(
                trajectory_valid=trajectory_valid,
                topology_attention=topology_attention,
                compatible_mask=compatible_mask,
                fallback_mask=fallback_mask,
                ego_component=ego_component,
                social_component=social_component,
                route_component=route_component,
                output=output if isinstance(output, StructuredLatent) else None,
                batch_size=batch_size,
                route_reachability=route_reachability_diagnostics,
                extra_diagnostics=extra_diagnostics,
            )
        if not self._policy_shadow_active:
            self._stamp_diagnostic_metadata(
                diagnostic_sampled, batch_size=batch_size, output=output
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
        self._diagnostic_site_forward_counts = {}
        self._diagnostic_active_site = (
            "eval_policy" if self.diagnostic_source == "eval_policy"
            else "rollout_policy"
        )
        self._diagnostic_active_update_index = None
        self._diagnostic_force_sample = False

    def set_diagnostic_context(
        self,
        site: str,
        update_index: int | None = None,
        *,
        force_sample: bool = False,
    ) -> tuple[str, int | None, bool]:
        """Set metadata for the next synchronous forward; returns prior context."""
        previous = (
            self._diagnostic_active_site,
            self._diagnostic_active_update_index,
            self._diagnostic_force_sample,
        )
        self._diagnostic_active_site = str(site)
        self._diagnostic_active_update_index = (
            None if update_index is None else int(update_index)
        )
        self._diagnostic_force_sample = bool(force_sample)
        return previous

    def restore_diagnostic_context(
        self, context: tuple[str, int | None, bool]
    ) -> None:
        self._diagnostic_active_site, self._diagnostic_active_update_index, self._diagnostic_force_sample = context

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

        if self.use_parameter_matched_nonlinear_slots:
            slot_ego_prefix = "nonlinear_slot_projections.ego"
            slot_social_prefix = "nonlinear_slot_projections.social"
            slot_route_prefix = "nonlinear_slot_projections.route"
        else:
            slot_ego_prefix = "ego_projection"
            slot_social_prefix = "social_projection"
            slot_route_prefix = "route_projection"
        return {
            "encoder_all": ("",),
            # Leaf groups are intentionally disjoint for update-delta analysis.
            # encoder_all is a summary group and must never be added to leaves.
            "state_encoder": ("state_encoder",),
            "spatial_vehicle_messages": ("vehicle_layers",),
            "temporal_attention": ("temporal_encoder",),
            "social_attention": ("social_attention",),
            "map_route_encoder": ("map_encoder",),
            "route_path_attention": ("route_attention",),
            "route_goal_attention": ("goal_attention",),
            "route_conflict_timing": (
                "route_conflict_timing_encoder",
            ),
            "route_modulation": (
                "route_bias_raw",
                "heading_bias_raw",
                "route_gate_mlp",
                "route_gate_bias",
                "route_edge_mlp",
            ),
            "topology_lane_encoder": ("topology_encoder",),
            "topology_vehicle_query": ("topology_attention",),
            "topology_goal_query": ("goal_topology_attention",),
            "topology_fusion_norms_scales": (
                "topology_norm",
                "topology_residual_scale",
                "goal_residual_scale",
                "route_topology_norm",
            ),
            "slot_ego": (slot_ego_prefix,),
            "slot_social": (slot_social_prefix,),
            "slot_route": (slot_route_prefix,),
            "mlp_readout": ("mlp_output",),
        }

    def _begin_diagnostic_forward(self) -> bool:
        self._diagnostic_forward_index += 1
        site = self._diagnostic_active_site
        site_forward_index = self._diagnostic_site_forward_counts.get(site, 0) + 1
        self._diagnostic_site_forward_counts[site] = site_forward_index
        self._diagnostic_active_site_forward_index = site_forward_index
        sampled = bool(
            self.behavior_diagnostics_enabled
            and (
                self._diagnostic_force_sample
                or (site_forward_index - 1)
                % self.behavior_diagnostics_sample_every
                == 0
            )
        )
        self._diagnostic_current_forward_sampled = sampled
        return sampled

    def _stamp_diagnostic_metadata(
        self,
        sampled: bool,
        *,
        batch_size: int,
        output=None,
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
            self._diagnostic_sample_site_code = self.diagnostic_site_code(
                self._diagnostic_active_site
            )
            self._diagnostic_sample_site_forward_index = (
                self._diagnostic_active_site_forward_index
            )
            self._diagnostic_sample_update_index = (
                -1
                if self._diagnostic_active_update_index is None
                else int(self._diagnostic_active_update_index)
            )
            output_tensor = output.tensor if isinstance(output, StructuredLatent) else output
            self._diagnostic_sample_output_requires_grad = bool(
                isinstance(output_tensor, Tensor) and output_tensor.requires_grad
            )
            self._diagnostic_sample_parameters_require_grad = any(
                parameter.requires_grad for parameter in self.parameters()
            )
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
                "diagnostic_sample_site_code": torch.tensor(
                    self._diagnostic_sample_site_code,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_site_forward_index": torch.tensor(
                    self._diagnostic_sample_site_forward_index,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_current_site_forward_index": torch.tensor(
                    self._diagnostic_active_site_forward_index,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_update_index": torch.tensor(
                    self._diagnostic_sample_update_index,
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_output_requires_grad": torch.tensor(
                    float(getattr(self, "_diagnostic_sample_output_requires_grad", False)),
                    device=device,
                    dtype=torch.float32,
                ),
                "diagnostic_sample_parameters_require_grad": torch.tensor(
                    float(getattr(self, "_diagnostic_sample_parameters_require_grad", False)),
                    device=device,
                    dtype=torch.float32,
                ),
            }
        )

    @staticmethod
    def diagnostic_site_code(site: str) -> int:
        return {
            "rollout_policy": 1,
            "representation_objective_online": 2,
            "critic_td_current": 3,
            "actor_policy_current": 4,
            "critic_actor_value": 5,
            "eval_policy": 6,
            "actor_policy_next_action_no_grad": 7,
        }.get(str(site), 0)

    def _spatiotemporal_diagnostics(
        self,
        *,
        graph_input: Tensor,
        graph_output: Tensor,
        trajectory_valid: Tensor,
        actor_features: Tensor,
        actor_valid: Tensor,
        social_attention_weights: Tensor,
        route_valid: Tensor | None,
        route_attention_weights: Tensor | None,
        goal_attention_weights: Tensor | None,
        goal_topology_valid: Tensor | None,
        goal_delta: Tensor | None,
    ) -> dict[str, Tensor]:
        """Summarize tensors from the active forward only (no auxiliary pass)."""
        dtype = graph_input.dtype
        frame_mask = trajectory_valid.to(dtype)
        frame_count = frame_mask.sum().clamp_min(1.0)
        spatial_delta = graph_output - graph_input
        valid_actor = trajectory_valid.any(dim=-1)
        history_lengths = trajectory_valid.to(dtype).sum(dim=-1)
        actor_denominator = valid_actor.to(dtype).sum().clamp_min(1.0)
        last_frame = self._last_valid(graph_output, trajectory_valid)
        temporal_delta = actor_features - last_frame
        temporal_mask = valid_actor.to(dtype)
        temporal_count = temporal_mask.sum().clamp_min(1.0)
        social_weights = social_attention_weights.squeeze(1).clamp_min(0.0)
        social_entropy = -(
            social_weights.clamp_min(1e-12).log() * social_weights
        ).sum(dim=-1)
        social_denominator = torch.tensor(
            float(social_weights.shape[0]), device=graph_input.device, dtype=dtype
        ).clamp_min(1.0)
        metrics: dict[str, Tensor] = {
            "spatial_graph_input_rms": (
                graph_input.square().mean(dim=-1) * frame_mask
            ).sum().div(frame_count).clamp_min(0.0).sqrt(),
            "spatial_graph_output_rms": (
                graph_output.square().mean(dim=-1) * frame_mask
            ).sum().div(frame_count).clamp_min(0.0).sqrt(),
            "spatial_message_delta_rms": (
                spatial_delta.square().mean(dim=-1) * frame_mask
            ).sum().div(frame_count).clamp_min(0.0).sqrt(),
            "spatial_message_delta_relative_rms": (
                (spatial_delta.square().mean(dim=-1) * frame_mask).sum()
                / frame_count
            ).clamp_min(0.0).sqrt()
            / ((graph_input.square().mean(dim=-1) * frame_mask).sum()
               .div(frame_count).clamp_min(1e-12).sqrt()),
            "spatial_valid_frame_count": frame_count,
            "history_valid_frames_per_actor_mean": (
                history_lengths * valid_actor.to(dtype)
            ).sum() / actor_denominator,
            "history_single_valid_frame_fraction": (
                ((history_lengths == 1) & valid_actor).to(dtype).sum()
                / actor_denominator
            ),
            "history_valid_actor_count": valid_actor.to(dtype).sum(),
            "temporal_output_rms": (
                actor_features.square().mean(dim=-1) * temporal_mask
            ).sum().div(temporal_count).clamp_min(0.0).sqrt(),
            "temporal_delta_from_last_frame_rms": (
                temporal_delta.square().mean(dim=-1) * temporal_mask
            ).sum().div(temporal_count).clamp_min(0.0).sqrt(),
            "temporal_delta_relative_rms": (
                (temporal_delta.square().mean(dim=-1) * temporal_mask).sum()
                / temporal_count
            ).clamp_min(0.0).sqrt()
            / ((last_frame.square().mean(dim=-1) * temporal_mask).sum()
               .div(temporal_count).clamp_min(1e-12).sqrt()),
            "temporal_valid_actor_count": temporal_count,
            "social_attention_entropy": social_entropy.mean(),
            "social_attention_effective_actor_count": social_entropy.exp().mean(),
            "social_attention_ego_self_mass": social_weights[:, 0].mean(),
            "social_attention_non_ego_mass": social_weights[:, 1:].sum(dim=-1).mean(),
            "social_attention_valid_actor_count_mean": actor_valid.to(dtype).sum(dim=-1).mean(),
            "social_attention_query_count": social_denominator,
        }
        if route_valid is not None:
            route_counts = route_valid.to(dtype).sum(dim=-1)
            route_nonempty = route_counts > 0
            route_denominator = route_nonempty.to(dtype).sum().clamp_min(1.0)
            metrics.update({
                "route_valid_path_count_mean": route_counts.mean(),
                "route_empty_path_fraction": (~route_nonempty).to(dtype).mean(),
                "route_nonempty_actor_count": route_nonempty.to(dtype).sum(),
                "route_path_count_valid_actor_mean": (
                    route_counts * route_nonempty.to(dtype)
                ).sum() / route_denominator,
            })
            if route_attention_weights is not None:
                # Route-context diagnostics below describe the ego route slot;
                # actor dimension is retained in the shared route encoder output.
                weights = route_attention_weights[:, 0].clamp_min(0.0)
                route_entropy = -(
                    weights.clamp_min(1e-12).log() * weights
                ).sum(dim=-1)
                metrics["route_attention_entropy_nonempty"] = (
                    route_entropy * route_nonempty.to(dtype)
                ).sum() / route_denominator
                metrics["route_attention_effective_paths_nonempty"] = (
                    route_entropy.exp() * route_nonempty.to(dtype)
                ).sum() / route_denominator
                metrics["route_attention_query_count"] = route_denominator
        if goal_attention_weights is not None and goal_topology_valid is not None:
            goal_weights = goal_attention_weights.squeeze(1).clamp_min(0.0)
            goal_valid = goal_topology_valid.to(torch.bool)
            goal_active = goal_valid.any(dim=-1)
            goal_entropy = -(
                goal_weights.clamp_min(1e-12).log() * goal_weights
            ).sum(dim=-1)
            goal_count = goal_active.to(dtype).sum().clamp_min(1.0)
            metrics.update({
                "goal_topology_valid_candidate_count_mean": goal_valid.to(dtype).sum(dim=-1).mean(),
                "goal_topology_active_query_count": goal_active.to(dtype).sum(),
                "goal_topology_attention_entropy_active": (
                    goal_entropy * goal_active.to(dtype)
                ).sum() / goal_count,
                "goal_topology_effective_nodes_active": (
                    goal_entropy.exp() * goal_active.to(dtype)
                ).sum() / goal_count,
                "goal_topology_attention_valid_count": goal_count,
            })
        if goal_delta is not None:
            metrics["goal_topology_delta_rms"] = goal_delta.detach().square().mean().sqrt()
        return metrics

    def _topology_actor_postnorm_diagnostics(
        self,
        *,
        intent_tokens: Tensor,
        graph_input: Tensor,
        trajectory_valid: Tensor,
    ) -> dict[str, Tensor]:
        """Measure the actual actor-intent residual after the shared LayerNorm.

        The baseline is the same ``topology_norm(intent_tokens)`` path used when
        topology is disabled. It is evaluated only for an already-selected
        diagnostic sample, on detached tensors under no_grad; it cannot add a
        training gradient or alter the policy output.
        """
        with torch.no_grad():
            dtype = graph_input.dtype
            valid = trajectory_valid.to(dtype)
            count = valid.sum()
            valid_mask = trajectory_valid[..., None]
            baseline = self.topology_norm(intent_tokens.detach()) * valid_mask
            actual = graph_input.detach()
            delta = actual - baseline
            delta_query_mean = delta.square().mean(dim=-1)
            baseline_query_mean = baseline.square().mean(dim=-1)
            delta_rms = (
                (delta_query_mean * valid).sum() / count.clamp_min(1.0)
            ).clamp_min(0.0).sqrt()
            baseline_rms = (
                (baseline_query_mean * valid).sum() / count.clamp_min(1.0)
            ).clamp_min(0.0).sqrt()
            delta_valid = (count > 0).to(dtype)
            relative_valid = (
                (count > 0) & torch.isfinite(baseline_rms) & (baseline_rms > 1e-12)
            ).to(dtype)
            relative_delta = torch.where(
                relative_valid > 0,
                delta_rms / baseline_rms.clamp_min(1e-12),
                torch.zeros_like(delta_rms),
            )
            return {
                "topology_actor_postnorm_delta_rms": delta_rms.detach(),
                "topology_actor_postnorm_baseline_rms": baseline_rms.detach(),
                "topology_actor_postnorm_delta_relative_rms": relative_delta.detach(),
                "topology_actor_postnorm_valid_query_count": count.detach(),
                "topology_actor_postnorm_delta_valid": delta_valid.detach(),
                "topology_actor_postnorm_delta_relative_valid": relative_valid.detach(),
            }

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
        extra_diagnostics: dict[str, Tensor] | None = None,
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
        if extra_diagnostics is not None:
            diagnostics.update(extra_diagnostics)
        self._diagnostics.update(
            {key: value.detach() for key, value in diagnostics.items()}
        )

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        out = self.forward_tokens(observations)
        if isinstance(out, StructuredLatent):
            return out.tensor
        return out


__all__ = ["IncrementalTopoEncoder"]
