"""Mask-safe, actor-shared encoding of short public observation histories."""
from __future__ import annotations

import math

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over valid entries; an all-masked row returns exact zeros."""
    mask = mask.to(torch.bool)
    safe = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(safe, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)


def last_valid(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the last valid item along time (works with internal missing frames)."""
    if values.ndim < 3 or mask.shape != values.shape[:-1]:
        raise ValueError("values/mask must be [..., time, feature] / [..., time]")
    time = values.shape[-2]
    positions = torch.arange(time, device=values.device).view(*([1] * (mask.ndim - 1)), time)
    indices = torch.where(mask, positions, -1).amax(dim=-1)
    valid = indices >= 0
    gathered = values.gather(-2, indices.clamp_min(0)[..., None, None].expand(*values.shape[:-2], 1, values.shape[-1])).squeeze(-2)
    gathered = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))
    return gathered, valid


class TemporalActorEncoder(nn.Module):
    """Encode an actor in the current ego frame, without actor-ID parameters."""

    def __init__(self, width: int = 128, history_samples: int = 21):
        super().__init__()
        if width < 8 or history_samples < 1:
            raise ValueError("width/history_samples are too small")
        self.history_samples = history_samples
        self.point = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.query = nn.Linear(width, width, bias=False)
        self.meta = nn.Sequential(nn.Linear(4, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU())

    def forward(self, history: torch.Tensor, history_valid: torch.Tensor,
                actor_valid: torch.Tensor, actor_size_lw: torch.Tensor,
                actor_age_s: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if history.ndim != 4 or history.shape[-1] != 6:
            raise ValueError("actor_history must be [B,A,T,6]")
        bsz, actors, steps, _ = history.shape
        expected = (bsz, actors, steps)
        if history_valid.shape != expected or actor_valid.shape != (bsz, actors):
            raise ValueError("history/actor masks do not match actor_history")
        if actor_size_lw.shape != (bsz, actors, 2) or actor_age_s.shape != (bsz, actors):
            raise ValueError("actor size/age shapes do not match actor_history")
        if steps != self.history_samples:
            raise ValueError(f"Expected {self.history_samples} history samples, got {steps}")

        mask = history_valid.bool() & actor_valid.bool()[..., None]
        current, has_current = last_valid(history, mask)
        # Actor zero is the ego by the observation contract. SUMO heading is
        # (sin(theta), cos(theta)); these axes make longitudinal projection
        # invariant to world translation and rotation.
        ego, ego_ok = last_valid(history[:, 0:1], mask[:, 0:1])
        ego = ego[:, 0]
        ego_ok = ego_ok[:, 0]
        ego_xy = ego[..., :2]
        ego_forward = ego[..., 2:4]
        default_forward = torch.zeros_like(ego_forward)
        default_forward[..., 1] = 1.0
        ego_forward = torch.where(ego_ok[..., None], ego_forward, default_forward)
        ego_forward = ego_forward / ego_forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ego_lateral = torch.stack((ego_forward[..., 1], -ego_forward[..., 0]), dim=-1)

        delta_xy = history[..., :2] - ego_xy[:, None, None, :]
        xy_local = torch.stack(((delta_xy * ego_forward[:, None, None, :]).sum(-1),
                                (delta_xy * ego_lateral[:, None, None, :]).sum(-1)), dim=-1) / 80.0
        heading = history[..., 2:4]
        heading_local = torch.stack(((heading * ego_forward[:, None, None, :]).sum(-1),
                                     (heading * ego_lateral[:, None, None, :]).sum(-1)), dim=-1)
        velocity = history[..., 4:6]
        velocity_local = torch.stack(((velocity * ego_forward[:, None, None, :]).sum(-1),
                                      (velocity * ego_lateral[:, None, None, :]).sum(-1)), dim=-1) / 20.0
        # The sampled index conveys temporal order; padding is still excluded
        # by the shared validity mask, so an all-empty track contributes zero.
        t = torch.linspace(-1.0, 0.0, steps, dtype=history.dtype, device=history.device)
        t = t.view(1, 1, steps, 1).expand(bsz, actors, -1, -1)
        local = torch.cat((xy_local, heading_local, velocity_local, t), dim=-1)
        encoded = self.point(local)
        # Use the latest observed embedding as a content query. This is
        # computed after gathering to avoid indexing a padded endpoint.
        latest_local = torch.cat((
            torch.stack((((current[..., :2] - ego_xy[:, None, :]) * ego_forward[:, None, :]).sum(-1),
                         ((current[..., :2] - ego_xy[:, None, :]) * ego_lateral[:, None, :]).sum(-1)), dim=-1) / 80.0,
            torch.stack(((current[..., 2:4] * ego_forward[:, None, :]).sum(-1),
                         (current[..., 2:4] * ego_lateral[:, None, :]).sum(-1)), dim=-1),
            torch.stack(((current[..., 4:6] * ego_forward[:, None, :]).sum(-1),
                         (current[..., 4:6] * ego_lateral[:, None, :]).sum(-1)), dim=-1) / 20.0,
            torch.zeros((bsz, actors, 1), dtype=history.dtype, device=history.device)), dim=-1)
        latest_encoded = self.point(latest_local)
        score = (encoded * self.query(latest_encoded)[..., None, :]).sum(-1) / math.sqrt(encoded.shape[-1])
        weights = masked_softmax(score, mask, dim=-1)
        pooled = (encoded * weights[..., None]).sum(dim=2)
        size = actor_size_lw.clamp(0.0, 10.0) / 10.0
        age = actor_age_s.clamp(0.0, 60.0)[..., None] / 60.0
        meta = self.meta(torch.cat((size, age, actor_valid.to(history.dtype)[..., None]), dim=-1))
        output = self.fuse(torch.cat((latest_encoded, pooled, meta), dim=-1))
        output = output * actor_valid.to(output.dtype)[..., None]
        current = current * actor_valid.to(current.dtype)[..., None]
        coverage = (mask.sum(dim=-1).to(history.dtype) / float(steps))
        return output, {"current": current, "has_current": has_current & actor_valid.bool(),
                        "history_coverage": coverage}
