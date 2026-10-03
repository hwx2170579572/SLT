"""Scene-event representation stages M0 (dual graph) and M1 (CV event graph)."""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from .dual_graph import DualGraphEncoder
from .event_graph import RouteEventGraph
from .polyline import PolylineEncoder, PublicMap
from .temporal import TemporalActorEncoder, masked_softmax


class TopologyRelationGraph(nn.Module):
    """Encode typed actor-route relations that have no event-zone geometry.

    Status bits are evidence that a static topology relation exists, not a
    collision label, safety label, or event-time estimate. Candidate routes
    remain mutually exclusive: sender routes are first pooled within each
    actor, then sender actors are pooled as a set.
    """

    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relation_bias = nn.Embedding(4, 1)
        self.relation_value = nn.Sequential(nn.Linear(4, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, route_features: torch.Tensor, route_active: torch.Tensor,
                relation_status: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, actors, routes, width = route_features.shape
        expected = (bsz, actors, routes, actors, routes)
        if tuple(relation_status.shape) != expected:
            raise ValueError(f"candidate_pair_topology_status must have shape {expected}")
        status = relation_status.to(torch.long).clamp(0, 3)
        active = route_active.bool()
        active_pair = (status > 0) & active[:, :, :, None, None] & active[:, None, None, :, :]
        same_actor = torch.eye(actors, dtype=torch.bool, device=route_features.device)
        active_pair = active_pair & ~same_actor[None, :, None, :, None]

        branch_count = actors * routes
        branch = route_features.reshape(bsz, branch_count, width)
        branch_active = active.reshape(bsz, branch_count)
        q = self.query(branch)
        k = self.key(branch)
        v = self.value(branch)
        status_flat = status.reshape(bsz, branch_count, branch_count)
        rel_bias = self.relation_bias(status_flat).squeeze(-1)
        logits = torch.einsum("bkw,bjw->bkj", q, k) / (width ** 0.5) + rel_bias

        actor_index = torch.arange(actors, device=route_features.device).repeat_interleave(routes)
        different_actor = actor_index[:, None] != actor_index[None, :]
        pair_mask = active_pair.reshape(bsz, branch_count, branch_count) & different_actor[None, :, :]
        pair_mask = pair_mask & branch_active[:, :, None] & branch_active[:, None, :]

        # For each receiving branch, aggregate a sender's route alternatives
        # before assigning that sender actor attention mass.
        logits_by_actor = logits.reshape(bsz, branch_count, actors, routes)
        mask_by_actor = pair_mask.reshape(bsz, branch_count, actors, routes)
        route_attention = masked_softmax(logits_by_actor, mask_by_actor, dim=-1)
        sender_value = v.reshape(bsz, actors, routes, width)
        route_message = torch.einsum("bkar,barw->bkaw", route_attention, sender_value)
        relation_type = torch.nn.functional.one_hot(status, num_classes=4).to(route_features.dtype)
        relation_type = relation_type.reshape(bsz, branch_count, actors, routes, 4)
        relation_mix = torch.einsum("bkar,bkarf->bkaf", route_attention, relation_type)
        route_message = route_message + self.relation_value(relation_mix)
        route_score = (route_attention * logits_by_actor).sum(dim=-1)
        sender_actor_valid = mask_by_actor.any(dim=-1)
        actor_attention = masked_softmax(route_score, sender_actor_valid, dim=-1)
        message = torch.einsum("bka,bkaw->bkw", actor_attention, route_message)
        has_relation = sender_actor_valid.any(dim=-1) & branch_active
        fused = self.fuse(torch.cat((branch, message), dim=-1))
        output = branch + (fused - branch) * has_relation.to(branch.dtype)[..., None]
        output_delta = output - branch
        output = output.reshape(bsz, actors, routes, width)
        output = output * active.to(output.dtype)[..., None]

        reverse_status = status.permute(0, 3, 4, 1, 2)
        status_union = torch.bitwise_or(status, reverse_status).reshape(
            bsz, branch_count, branch_count)
        unique_actor_pair = actor_index[:, None] < actor_index[None, :]
        active_pair_flat = (active[:, :, :, None, None] & active[:, None, None, :, :]).reshape(
            bsz, branch_count, branch_count)
        unique_pair_mask = ((status_union > 0) & active_pair_flat &
                            unique_actor_pair[None, :, :])
        channel_foe = ((status_union & 1) != 0) & unique_pair_mask
        channel_merge = ((status_union & 2) != 0) & unique_pair_mask
        diagnostics = {
            "topology_relation_pairs": unique_pair_mask.sum().to(route_features.dtype).detach(),
            "topology_relation_branches": has_relation.sum().to(route_features.dtype).detach(),
            "topology_relation_foe_pairs": channel_foe.sum().to(route_features.dtype).detach(),
            "topology_relation_merge_pairs": channel_merge.sum().to(route_features.dtype).detach(),
            "topology_relation_token_norm": ((message.norm(dim=-1) * has_relation.to(message.dtype)).sum() /
                                              has_relation.sum().clamp_min(1).to(message.dtype)).detach(),
            "topology_relation_readout_delta_norm": (
                (output_delta.norm(dim=-1) * has_relation.to(output_delta.dtype)).sum() /
                has_relation.sum().clamp_min(1).to(output_delta.dtype)).detach(),
        }
        return output, diagnostics


class SceneEncoder(nn.Module):
    """Actor-invariant deterministic scene encoder for the new protocol only.

    `m0` ignores all candidate-zone-event observation keys. `m1` consumes only
    the public deterministic constant-velocity event intervals, and keeps each
    candidate path as an explicit conditional branch until event aggregation.
    """

    def __init__(self, method: Literal["m0", "m1"], width: int = 128,
                 z_dim: int = 128, public_map: dict[str, Any] | None = None,
                 history_samples: int = 21, prediction_seconds: float = 3.0):
        super().__init__()
        if method not in {"m0", "m1"}:
            raise ValueError("method must be 'm0' or 'm1'")
        if width < 8 or z_dim < 1:
            raise ValueError("width and z_dim must be positive")
        if public_map is None:
            raise ValueError("public_map is required and must contain actual map dimensions")
        self.method = method
        self.width = int(width)
        self.z_dim = int(z_dim)
        self.prediction_seconds = float(prediction_seconds)
        self.public_map = PublicMap(public_map)
        self.temporal = TemporalActorEncoder(width, history_samples)
        self.polyline = PolylineEncoder(width)
        self.dual_graph = DualGraphEncoder(width)
        self.route_position = nn.Linear(2, width, bias=False)
        self.route_score = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.route_status = nn.Embedding(3, width)
        self.route_fuse = nn.Sequential(nn.Linear(width * 3 + width + 3, width),
                                        nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.route_query = nn.Linear(width, width, bias=False)
        self.coverage_encoder = nn.Sequential(nn.Linear(2, width), nn.LayerNorm(width), nn.GELU())
        self.lane_match_encoder = nn.Sequential(nn.Linear(2, width), nn.GELU())
        self.actor_route_fuse = nn.Sequential(nn.Linear(width * 4, width), nn.LayerNorm(width), nn.GELU())
        self.topology_relation = TopologyRelationGraph(width)
        self.time_encoder = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.readout = nn.Sequential(nn.Linear(width * 5, width * 2), nn.LayerNorm(width * 2), nn.GELU(),
                                     nn.Linear(width * 2, z_dim))
        self.event_graph = RouteEventGraph(width, prediction_seconds) if method == "m1" else None
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _batch(obs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        if "actor_history" not in obs:
            raise KeyError("observation is missing actor_history")
        unbatched = obs["actor_history"].ndim == 3
        if not unbatched:
            return obs, False
        result = {}
        for key, value in obs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"observation[{key}] must be a torch.Tensor")
            result[key] = value.unsqueeze(0)
        return result, True

    @staticmethod
    def _required(obs: dict[str, torch.Tensor], keys: tuple[str, ...]) -> None:
        missing = [key for key in keys if key not in obs]
        if missing:
            raise KeyError(f"observation is missing required keys: {missing}")

    @staticmethod
    def _to_local(current: torch.Tensor, actor_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ego = current[:, 0]
        ego_ok = actor_valid[:, 0].bool()
        ego_xy = ego[:, :2]
        forward = ego[:, 2:4]
        default = torch.zeros_like(forward)
        default[:, 1] = 1.0
        forward = torch.where(ego_ok[:, None], forward, default)
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        lateral = torch.stack((forward[:, 1], -forward[:, 0]), dim=-1)
        delta = current[..., :2] - ego_xy[:, None, :]
        position = torch.stack(((delta * forward[:, None, :]).sum(-1),
                                (delta * lateral[:, None, :]).sum(-1)), dim=-1) / 80.0
        velocity = current[..., 4:6]
        velocity = torch.stack(((velocity * forward[:, None, :]).sum(-1),
                                (velocity * lateral[:, None, :]).sum(-1)), dim=-1) / 20.0
        heading = current[..., 2:4]
        heading = torch.stack(((heading * forward[:, None, :]).sum(-1),
                               (heading * lateral[:, None, :]).sum(-1)), dim=-1)
        mask = actor_valid.to(current.dtype)[..., None]
        return ego_xy, forward, position * mask, velocity * mask, heading * mask

    def _routes(self, obs: dict[str, torch.Tensor], lane_tokens: torch.Tensor,
                actor_features: torch.Tensor, actor_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = obs["candidate_lane_ptr"].long()
        lane_mask = obs["candidate_lane_mask"].bool()
        candidate_valid = obs["candidate_valid"].bool()
        status = obs["candidate_status"].long().clamp(0, 2)
        progress = obs["candidate_progress_m"].clamp(0.0, 200.0)
        bsz, actors, routes, points = ptr.shape
        lane_count = lane_tokens.shape[1]
        clamped = ptr.clamp(min=0, max=lane_count - 1)
        lane_gather = clamped.reshape(bsz, -1)
        gathered = torch.gather(lane_tokens, 1, lane_gather[..., None].expand(-1, -1, self.width))
        gathered = gathered.reshape(bsz, actors, routes, points, self.width)
        gathered_lane_valid = torch.gather(self.public_map.lane_valid[None, :].expand(bsz, -1), 1,
                                           lane_gather).reshape(bsz, actors, routes, points)
        pointer_in_bounds = (ptr >= 0) & (ptr < lane_count)
        valid_path = lane_mask & pointer_in_bounds & gathered_lane_valid & candidate_valid[..., None]
        order = torch.linspace(0.0, 1.0, points, dtype=lane_tokens.dtype, device=lane_tokens.device)
        pos = torch.stack((order, 1.0 - order), dim=-1)
        path_tokens = gathered + self.route_position(pos)[None, None, None, :, :]
        score = self.route_score(path_tokens).squeeze(-1)
        weights = masked_softmax(score, valid_path, dim=-1)
        pooled = (path_tokens * weights[..., None]).sum(dim=-2)
        first_idx = valid_path.to(torch.int64).argmax(dim=-1)
        last_idx = torch.where(valid_path, torch.arange(points, device=ptr.device), -1).amax(dim=-1).clamp_min(0)
        first = path_tokens.gather(-2, first_idx[..., None, None].expand(-1, -1, -1, 1, self.width)).squeeze(-2)
        last = path_tokens.gather(-2, last_idx[..., None, None].expand(-1, -1, -1, 1, self.width)).squeeze(-2)
        route_status = self.route_status(status)
        unknown_count = obs["candidate_unknown_count"].to(lane_tokens.dtype)
        unknown_flag = (unknown_count < 0).to(lane_tokens.dtype)
        known_count = unknown_count.clamp(0.0, 32.0) / 32.0
        route_meta = torch.stack((progress / 200.0,
                                  known_count[:, :, None].expand(-1, -1, routes),
                                  unknown_flag[:, :, None].expand(-1, -1, routes)), dim=-1)
        route_features = self.route_fuse(torch.cat((pooled, first, last, route_status, route_meta), dim=-1))
        route_active = candidate_valid & valid_path.any(dim=-1) & actor_valid[..., None].bool()
        route_features = route_features * route_active.to(route_features.dtype)[..., None]
        query = self.route_query(actor_features)[:, :, None, :]
        route_logits = (route_features * query).sum(dim=-1) / (self.width ** 0.5)
        route_weights = masked_softmax(route_logits, route_active, dim=-1)
        route_summary = (route_features * route_weights[..., None]).sum(dim=2)
        return route_features, route_active, route_summary

    def forward(self, observation: dict[str, torch.Tensor], disable_events: bool = False) -> torch.Tensor:
        obs, unbatched = self._batch(observation)
        common = ("actor_history", "history_valid", "actor_valid", "actor_size_lw", "actor_age_s",
                  "actor_lane_ptr", "actor_lane_s_m", "lane_match_conf", "remaining_time_s",
                  "candidate_lane_ptr", "candidate_lane_mask", "candidate_valid", "candidate_status",
                  "candidate_progress_m", "candidate_unknown_count")
        self._required(obs, common)
        for key, value in obs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"observation[{key}] must be a torch.Tensor")
        history = obs["actor_history"].to(torch.float32)
        history_valid = obs["history_valid"].bool()
        actor_valid = obs["actor_valid"].bool()
        temporal, actor_info = self.temporal(history, history_valid, actor_valid,
                                             obs["actor_size_lw"].to(history.dtype),
                                             obs["actor_age_s"].to(history.dtype))
        model_actor_valid = actor_valid & actor_info["has_current"]
        temporal = temporal * model_actor_valid.to(temporal.dtype)[..., None]
        ego_xy, ego_forward, actor_position, actor_velocity, actor_heading = self._to_local(
            actor_info["current"], model_actor_valid)
        lane_features, lane_centers, zone_features, _ = self.polyline(self.public_map, ego_xy, ego_forward)
        lane_features = self.dual_graph.encode_map(lane_features, lane_centers, self.public_map.lane_valid,
                                                   self.public_map.edge_index, self.public_map.edge_type,
                                                   self.public_map.edge_valid)
        route_features, route_active, route_summary = self._routes(obs, lane_features, temporal, model_actor_valid)
        event_metrics: dict[str, torch.Tensor] = {
            "event_valid_intervals": history.new_zeros(()), "event_route_coverage": history.new_zeros(()),
            "event_active": history.new_zeros(()), "event_status_coverage": history.new_zeros(()),
            "event_mean_duration_s": history.new_zeros(()), "event_branch_active_routes": history.new_zeros(()),
            "event_interval_routes": history.new_zeros(()), "event_status_unknown": history.new_zeros(()),
            "event_status_no_zone": history.new_zeros(()), "event_status_cv_valid": history.new_zeros(()),
            "event_status_stationary": history.new_zeros(()), "event_status_beyond_horizon": history.new_zeros(()),
            "event_token_norm": history.new_zeros(()), "event_branch_token_norm": history.new_zeros(()),
            "event_path_intervals": history.new_zeros(()), "event_path_routes": history.new_zeros(()),
            "event_order_links": history.new_zeros(()), "event_path_token_norm": history.new_zeros(()),
        }
        if self.method == "m1":
            event_keys = ("candidate_zone_event_s", "candidate_zone_event_mask", "candidate_zone_status",
                          "candidate_zone_s_m", "candidate_zone_path_mask")
            self._required(obs, event_keys)
            marker = obs.get("_disable_events")
            if marker is not None:
                if marker.dtype != torch.bool:
                    raise TypeError("_disable_events analysis marker must be bool")
                disable_events = disable_events or bool(marker.all().detach().cpu())
            route_features, event_metrics = self.event_graph(
                temporal, route_features, route_active, zone_features,
                obs["candidate_zone_event_s"].to(history.dtype),
                obs["candidate_zone_event_mask"].bool(), obs["candidate_zone_status"].long(),
                self.public_map.zone_valid, obs["candidate_zone_s_m"].to(history.dtype),
                obs["candidate_zone_path_mask"].bool(),
                disable_events=disable_events)
        relation_status = obs.get("candidate_pair_topology_status")
        if relation_status is None:
            relation_status = torch.zeros(
                (history.shape[0], route_features.shape[1], route_features.shape[2],
                 route_features.shape[1], route_features.shape[2]),
                dtype=torch.uint8, device=history.device)
        route_features, topology_metrics = self.topology_relation(
            route_features, route_active, relation_status)
        route_query = self.route_query(temporal)[:, :, None, :]
        weights = masked_softmax((route_features * route_query).sum(-1) / (self.width ** 0.5),
                                 route_active, dim=-1)
        route_summary = (route_features * weights[..., None]).sum(dim=2)
        # M0 never branches on, validates, or consumes the event keys.
        lane_match = torch.stack((obs["lane_match_conf"].to(history.dtype).clamp(0.0, 1.0),
                                  (obs["actor_lane_ptr"] >= 0).to(history.dtype)), dim=-1)
        lane_match = self.lane_match_encoder(lane_match) * model_actor_valid.to(history.dtype)[..., None]
        candidate_unknown_count = obs["candidate_unknown_count"].to(history.dtype)
        actor_coverage = self.coverage_encoder(torch.stack((candidate_unknown_count.clamp(0.0, 32.0) / 32.0,
                                                            (candidate_unknown_count < 0).to(history.dtype)), dim=-1))
        actor_coverage = actor_coverage * model_actor_valid.to(history.dtype)[..., None]
        actor_input = self.actor_route_fuse(torch.cat((temporal, route_summary, lane_match, actor_coverage), dim=-1))
        actors, lanes, map_context = self.dual_graph(
            actor_input, actor_position, actor_velocity, actor_heading, model_actor_valid,
            obs["actor_lane_ptr"].long(), obs["actor_lane_s_m"].to(history.dtype),
            lane_features, lane_centers, self.public_map.lane_valid,
            self.public_map.edge_index, self.public_map.edge_type, self.public_map.edge_valid,
            map_encoded=True)
        actor_weights = model_actor_valid.to(history.dtype)
        actor_pool = (actors * actor_weights[..., None]).sum(dim=1) / actor_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        ego_actor = torch.where(model_actor_valid[:, 0, None], actors[:, 0], actor_pool)
        ego_map_context = torch.where(model_actor_valid[:, 0, None], map_context[:, 0], torch.zeros_like(map_context[:, 0]))
        ego_route = route_summary[:, 0]
        remaining = obs["remaining_time_s"].to(history.dtype).reshape(history.shape[0], -1)[:, :1].clamp(0.0, 60.0) / 60.0
        time_features = torch.cat((remaining, torch.sin(torch.pi * remaining), torch.cos(torch.pi * remaining)), dim=-1)
        time_emb = self.time_encoder(time_features)
        z = self.readout(torch.cat((ego_actor, actor_pool, ego_map_context, ego_route, time_emb), dim=-1))
        self._last_diagnostics = {
            **{k: v.detach() for k, v in event_metrics.items()},
            **{k: v.detach() for k, v in topology_metrics.items()},
            "valid_actor_fraction": model_actor_valid.to(history.dtype).mean().detach(),
            "valid_lane_fraction": self.public_map.lane_valid.to(history.dtype).mean().detach(),
            "route_coverage": route_active.any(dim=-1).to(history.dtype).mean().detach(),
            "history_coverage": actor_info["history_coverage"].mean().detach(),
            "map_lanes": history.new_tensor(float(self.public_map.num_lanes)),
            "map_edges": history.new_tensor(float(self.public_map.num_edges)),
            "map_zones": history.new_tensor(float(self.public_map.num_zones)),
        }
        return z.squeeze(0) if unbatched else z

    def diagnostics(self) -> dict[str, float]:
        """Detached summaries from the last forward; no extra model pass."""
        return {key: float(value.detach().cpu()) for key, value in self._last_diagnostics.items()}

    @staticmethod
    def events_off_observation(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from .event_projection import events_off_observation
        return events_off_observation(observation)


def build_encoder(method: Literal["m0", "m1"], width: int = 128, z_dim: int = 128,
                  public_map: dict[str, Any] | None = None,
                  history_samples: int = 21, prediction_seconds: float = 3.0) -> SceneEncoder:
    return SceneEncoder(method=method, width=width, z_dim=z_dim, public_map=public_map,
                        history_samples=history_samples, prediction_seconds=prediction_seconds)
