"""Route-conditional actor-zone event graph for deterministic M1 intervals."""
from __future__ import annotations

import math

import torch
from torch import nn

from .event_projection import project_event_intervals
from .temporal import masked_softmax


class RouteEventGraph(nn.Module):
    """Build candidate-branch event tokens and same-zone pair messages.

    A candidate route is a conditional alternative, not a simultaneous actor.
    Pair messages therefore retain (actor, route) axes until after zone
    interactions are formed. Pooling over another actor's candidate branches
    is a learned set aggregation, not a route probability or joint forecast.
    """

    def __init__(self, width: int = 128, prediction_seconds: float = 3.0):
        super().__init__()
        if prediction_seconds <= 0:
            raise ValueError("prediction_seconds must be positive")
        self.prediction_seconds = float(prediction_seconds)
        self.status_embedding = nn.Embedding(5, width)
        self.interval = nn.Sequential(nn.Linear(4, width), nn.GELU())
        self.self_event = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU())
        self.status_fuse = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.path_event = nn.Sequential(nn.Linear(width * 3 + 2, width), nn.LayerNorm(width), nn.GELU())
        self.path_sequence = nn.Sequential(nn.Linear(width * 3 + 2, width), nn.LayerNorm(width), nn.GELU())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relative_bias = nn.Sequential(nn.Linear(6, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.relative_value = nn.Sequential(nn.Linear(6, width), nn.GELU())
        self.branch_fuse = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, actor_features: torch.Tensor, route_features: torch.Tensor,
                route_active: torch.Tensor, zone_features: torch.Tensor,
                event_seconds: torch.Tensor, event_mask: torch.Tensor,
                event_status: torch.Tensor, zone_valid: torch.Tensor,
                zone_path_s_m: torch.Tensor, zone_path_mask: torch.Tensor,
                disable_events: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, actors, routes, width = route_features.shape
        zones = zone_features.shape[1]
        expected = (bsz, actors, routes, zones)
        if (event_seconds.shape != expected + (2,) or event_mask.shape != expected or event_status.shape != expected or
                zone_path_s_m.shape != expected + (2,) or zone_path_mask.shape != expected):
            raise ValueError("candidate zone event/path tensors do not match route/map axes")
        numeric, valid, duration = project_event_intervals(
            event_seconds, event_mask, event_status, self.prediction_seconds)
        zone_ok = zone_valid[None, None, None, :].bool()
        if disable_events:
            valid = torch.zeros_like(valid)
            numeric = torch.zeros_like(numeric)
        valid = valid & route_active[..., None] & zone_ok
        # Status counts retain useful distinctions such as a stationary actor
        # or a route with no zone. Event-off deliberately erases these too.
        status_ids = event_status.long().clamp(0, 4)
        status_onehot = torch.nn.functional.one_hot(status_ids, num_classes=5).to(route_features.dtype)
        status_onehot = status_onehot * route_active[..., None, None].to(status_onehot.dtype)
        status_onehot = status_onehot * zone_ok[..., None].to(status_onehot.dtype)
        if disable_events:
            status_onehot = torch.zeros_like(status_onehot)
        status_fraction = status_onehot.sum(dim=-2) / zone_valid.sum().clamp_min(1).to(route_features.dtype)
        status_emb = status_fraction @ self.status_embedding.weight

        path_finite = torch.isfinite(zone_path_s_m).all(dim=-1)
        path_ordered = zone_path_s_m[..., 1] >= zone_path_s_m[..., 0]
        path_valid = zone_path_mask.bool() & route_active[..., None] & zone_ok & path_finite & path_ordered
        if disable_events:
            path_valid = torch.zeros_like(path_valid)
        path_s = torch.nan_to_num(zone_path_s_m, nan=0.0, posinf=0.0, neginf=0.0).clamp(-200.0, 200.0) / 200.0
        if disable_events:
            path_s = torch.zeros_like(path_s)
        zone = zone_features[:, None, None, :, :].expand(-1, actors, routes, -1, -1)
        base = (actor_features[:, :, None, :] + route_features)
        base_z = base[..., None, :].expand(-1, -1, -1, zones, -1)
        interval_emb = self.interval(numeric)
        event_token = self.self_event(torch.cat((base_z, zone, interval_emb), dim=-1))
        attention_score = (event_token * base_z).sum(dim=-1) / math.sqrt(width)
        zone_weights = masked_softmax(attention_score, valid, dim=-1)
        self_summary = (event_token * zone_weights[..., None]).sum(dim=-2)
        # Explicit path-ordered zone context is independent of vehicle speed.
        # Distances, rather than CV arrival times, determine predecessor and
        # successor even when status=stationary or beyond the timing horizon.
        zone_status_emb = self.status_embedding(status_ids)
        zone_status_emb = zone_status_emb * route_active[..., None, None].to(route_features.dtype)
        zone_status_emb = zone_status_emb * zone_ok[..., None].to(route_features.dtype)
        if disable_events:
            zone_status_emb = torch.zeros_like(zone_status_emb)
        path_token = self.path_event(torch.cat((base_z, zone, zone_status_emb, path_s), dim=-1))
        sort_key = path_s[..., 0].masked_fill(~path_valid, torch.finfo(path_s.dtype).max)
        order = torch.argsort(sort_key, dim=-1, stable=True)
        ordered_valid = path_valid.gather(-1, order)
        ordered_token = path_token.gather(-2, order[..., None].expand(-1, -1, -1, -1, width))
        ordered_s = path_s.gather(-2, order[..., None].expand(-1, -1, -1, -1, 2))
        zero_token = torch.zeros_like(ordered_token[..., :1, :])
        zero_value = torch.zeros_like(ordered_s[..., :1, 0])
        previous_token = torch.cat((zero_token, ordered_token[..., :-1, :]), dim=-2)
        next_token = torch.cat((ordered_token[..., 1:, :], zero_token), dim=-2)
        previous_clear = torch.cat((zero_value, ordered_s[..., :-1, 1]), dim=-1)
        next_entry = torch.cat((ordered_s[..., 1:, 0], zero_value), dim=-1)
        previous_valid = torch.cat((torch.zeros_like(ordered_valid[..., :1]), ordered_valid[..., :-1]), dim=-1)
        next_valid = torch.cat((ordered_valid[..., 1:], torch.zeros_like(ordered_valid[..., :1])), dim=-1)
        previous_gap = ((ordered_s[..., 0] - previous_clear).clamp(-1.0, 1.0) * previous_valid.to(path_s.dtype))
        next_gap = ((next_entry - ordered_s[..., 1]).clamp(-1.0, 1.0) * next_valid.to(path_s.dtype))
        sequence = self.path_sequence(torch.cat((ordered_token,
                                                 previous_token * previous_valid[..., None].to(path_token.dtype),
                                                 next_token * next_valid[..., None].to(path_token.dtype),
                                                 previous_gap[..., None], next_gap[..., None]), dim=-1))
        sequence = sequence * ordered_valid[..., None].to(sequence.dtype)
        sequence_score = (sequence * base_z).sum(dim=-1) / math.sqrt(width)
        sequence_weights = masked_softmax(sequence_score, ordered_valid, dim=-1)
        path_summary = (sequence * sequence_weights[..., None]).sum(dim=-2)
        self_summary = self.status_fuse(torch.cat((self_summary + path_summary, status_emb), dim=-1))

        # Compare only same-zone timing intervals, then pool candidate routes
        # *within each sender actor* before attention pools across the actor set.
        # This avoids giving an actor more prior attention mass just because it
        # has more mutually-exclusive route alternatives. Both are learned set
        # attentions, not route probabilities.
        branch_count = actors * routes
        branch = base.reshape(bsz, branch_count, width)
        branch_active = route_active.reshape(bsz, branch_count)
        actor_index = torch.arange(actors, device=route_features.device).repeat_interleave(routes)
        q, k, v = self.query(branch), self.key(branch), self.value(branch)
        flat_entry = torch.nan_to_num(event_seconds[..., 0], nan=0.0).reshape(bsz, branch_count, zones)
        flat_clear = torch.nan_to_num(event_seconds[..., 1], nan=0.0).reshape(bsz, branch_count, zones)
        flat_valid = valid.reshape(bsz, branch_count, zones)
        entry_i, entry_j = flat_entry[:, :, None, :], flat_entry[:, None, :, :]
        clear_i, clear_j = flat_clear[:, :, None, :], flat_clear[:, None, :, :]
        overlap = (torch.minimum(clear_i, clear_j) - torch.maximum(entry_i, entry_j)).clamp_min(0.0) / self.prediction_seconds
        gap = (torch.maximum(entry_i, entry_j) - torch.minimum(clear_i, clear_j)).clamp_min(0.0) / self.prediction_seconds
        pair_zone_valid = flat_valid[:, :, None, :] & flat_valid[:, None, :, :]
        count = pair_zone_valid.sum(dim=-1).clamp_min(1).to(route_features.dtype)
        mean_de = (((entry_i - entry_j) / self.prediction_seconds).clamp(-1.0, 1.0) * pair_zone_valid).sum(-1) / count
        mean_dc = (((clear_i - clear_j) / self.prediction_seconds).clamp(-1.0, 1.0) * pair_zone_valid).sum(-1) / count
        max_overlap = overlap.masked_fill(~pair_zone_valid, -1.0).max(dim=-1).values.clamp_min(0.0)
        min_gap = gap.masked_fill(~pair_zone_valid, torch.finfo(gap.dtype).max).min(dim=-1).values
        min_gap = torch.where(pair_zone_valid.any(dim=-1), min_gap, torch.zeros_like(min_gap)).clamp(0.0, 1.0)
        mean_sign_entry = (torch.sign(entry_i - entry_j) * pair_zone_valid).sum(-1) / count
        mean_sign_clear = (torch.sign(clear_i - clear_j) * pair_zone_valid).sum(-1) / count
        relative = torch.stack((mean_de, mean_dc, max_overlap, min_gap,
                                mean_sign_entry, mean_sign_clear), dim=-1)
        different_actor = actor_index[:, None] != actor_index[None, :]
        pair_valid = (pair_zone_valid.any(dim=-1) & different_actor[None, :, :] &
                      branch_active[:, :, None] & branch_active[:, None, :])
        logits = torch.einsum("bkw,bjw->bkj", q, k) / math.sqrt(width)
        logits = logits + self.relative_bias(relative).squeeze(-1)
        logits_by_sender = logits.reshape(bsz, branch_count, actors, routes)
        mask_by_sender = pair_valid.reshape(bsz, branch_count, actors, routes)
        route_attention = masked_softmax(logits_by_sender, mask_by_sender, dim=-1)
        route_relative = relative.reshape(bsz, branch_count, actors, routes, 6)
        sender_value = v.reshape(bsz, actors, routes, width)
        route_message = torch.einsum("bkar,barw->bkaw", route_attention, sender_value)
        route_relative_mean = torch.einsum("bkar,bkarf->bkaf", route_attention, route_relative)
        route_message = route_message + self.relative_value(route_relative_mean)
        sender_actor_score = (route_attention * logits_by_sender).sum(dim=-1)
        sender_actor_valid = mask_by_sender.any(dim=-1)
        actor_attention = masked_softmax(sender_actor_score, sender_actor_valid, dim=-1)
        pair_message = torch.einsum("bka,bkaw->bkw", actor_attention, route_message)
        pair_message = pair_message * sender_actor_valid.any(dim=-1).to(pair_message.dtype)[..., None]
        pair_message = pair_message.reshape(bsz, actors, routes, width)
        pair_message = pair_message * route_active.to(pair_message.dtype)[..., None]
        candidate_output = self.branch_fuse(torch.cat((route_features, self_summary, pair_message), dim=-1))
        has_event_information = (valid.any(dim=-1) | path_valid.any(dim=-1) |
                                 status_onehot.sum(dim=(-1, -2)).gt(0)) & route_active
        output = route_features + (candidate_output - route_features) * has_event_information.to(route_features.dtype)[..., None]
        active_event_count = valid.sum().to(route_features.dtype)
        diagnostics = {
            "event_valid_intervals": active_event_count.detach(),
            "event_route_coverage": (valid.any(dim=-1).sum().to(route_features.dtype) /
                                     route_active.sum().clamp_min(1).to(route_features.dtype)).detach(),
            "event_active": (valid.any()).to(route_features.dtype).detach(),
            "event_branch_active_routes": has_event_information.sum().to(route_features.dtype).detach(),
            "event_interval_routes": valid.any(dim=-1).sum().to(route_features.dtype).detach(),
            "event_path_intervals": path_valid.sum().to(route_features.dtype).detach(),
            "event_path_routes": path_valid.any(dim=-1).sum().to(route_features.dtype).detach(),
            "event_order_links": (ordered_valid[..., 1:] & ordered_valid[..., :-1]).sum().to(route_features.dtype).detach(),
            "event_status_coverage": (status_onehot.sum() / zone_valid.sum().clamp_min(1).to(route_features.dtype) /
                                      route_active.sum().clamp_min(1).to(route_features.dtype)).detach(),
            "event_mean_duration_s": ((duration * valid.to(duration.dtype)).sum() /
                                       valid.sum().clamp_min(1).to(duration.dtype)).detach(),
            "event_status_unknown": status_onehot[..., 0].sum().detach(),
            "event_status_no_zone": status_onehot[..., 1].sum().detach(),
            "event_status_cv_valid": status_onehot[..., 2].sum().detach(),
            "event_status_stationary": status_onehot[..., 3].sum().detach(),
            "event_status_beyond_horizon": status_onehot[..., 4].sum().detach(),
            "event_token_norm": ((event_token.norm(dim=-1) * valid.to(event_token.dtype)).sum() /
                                 valid.sum().clamp_min(1).to(event_token.dtype)).detach(),
            "event_path_token_norm": ((sequence.norm(dim=-1) * ordered_valid.to(sequence.dtype)).sum() /
                                      ordered_valid.sum().clamp_min(1).to(sequence.dtype)).detach(),
            "event_branch_token_norm": (candidate_output.norm(dim=-1) * has_event_information.to(output.dtype)).sum().detach() /
                                       has_event_information.sum().clamp_min(1).to(output.dtype),
        }
        return output, diagnostics
