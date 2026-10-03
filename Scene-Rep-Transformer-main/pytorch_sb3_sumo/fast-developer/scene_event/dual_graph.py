"""Permutation-equivariant actor/map interaction with explicit map topology."""
from __future__ import annotations

import math

import torch
from torch import nn

from .temporal import masked_softmax


class TopologyMessageLayer(nn.Module):
    def __init__(self, width: int = 128, edge_types: int = 6):
        super().__init__()
        self.edge_type = nn.Embedding(edge_types, width)
        self.message = nn.Sequential(nn.Linear(width * 3 + 2, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, lanes: torch.Tensor, centers: torch.Tensor,
                lane_valid: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_valid: torch.Tensor) -> torch.Tensor:
        bsz, count, width = lanes.shape
        if edge_index.shape[1] == 0:
            return lanes * lane_valid.to(lanes.dtype)[None, :, None]
        src, dst = edge_index[0], edge_index[1]
        in_bounds = (src >= 0) & (src < count) & (dst >= 0) & (dst < count)
        safe_src, safe_dst = src.clamp(0, count - 1), dst.clamp(0, count - 1)
        valid = edge_valid & in_bounds & lane_valid[safe_src] & lane_valid[safe_dst]
        source, target = lanes[:, safe_src], lanes[:, safe_dst]
        relative = (centers[:, safe_src] - centers[:, safe_dst]).clamp(-2.0, 2.0)
        type_emb = self.edge_type(edge_type.clamp(0, 5))[None].expand(bsz, -1, -1)
        messages = self.message(torch.cat((source, target, type_emb, relative), dim=-1))
        messages = messages * valid.to(messages.dtype)[None, :, None]
        aggregate = lanes.new_zeros((bsz, count, width))
        aggregate.index_add_(1, safe_dst, messages)
        degree = lanes.new_zeros((count,)).index_add_(0, safe_dst, valid.to(lanes.dtype)).clamp_min(1.0)
        aggregate = aggregate / degree[None, :, None]
        updated = self.update(torch.cat((lanes, aggregate), dim=-1))
        return updated * lane_valid.to(updated.dtype)[None, :, None]


class ActorSelfAttention(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relative_bias = nn.Sequential(nn.Linear(7, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.relative_value = nn.Sequential(nn.Linear(7, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, actors: torch.Tensor, position: torch.Tensor, velocity: torch.Tensor,
                heading: torch.Tensor, actor_valid: torch.Tensor,
                lane_ptr: torch.Tensor, lane_s: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(actors), self.key(actors), self.value(actors)
        delta_pos = position[:, None, :, :] - position[:, :, None, :]
        delta_vel = velocity[:, None, :, :] - velocity[:, :, None, :]
        align = (heading[:, :, None, :] * heading[:, None, :, :]).sum(-1, keepdim=True)
        same_lane = ((lane_ptr[:, :, None] >= 0) & (lane_ptr[:, :, None] == lane_ptr[:, None, :])).to(position.dtype)
        delta_s = ((lane_s[:, None, :] - lane_s[:, :, None]).clamp(-80.0, 80.0) / 80.0)
        delta_s = delta_s * same_lane
        rel = torch.cat((delta_pos.clamp(-2.0, 2.0), delta_vel.clamp(-2.0, 2.0),
                         align, same_lane[..., None], delta_s[..., None]), dim=-1)
        logits = torch.einsum("baw,bcw->bac", q, k) / math.sqrt(q.shape[-1])
        logits = logits + self.relative_bias(rel).squeeze(-1)
        pair_mask = actor_valid[:, None, :].bool()
        weights = masked_softmax(logits, pair_mask, dim=-1)
        values = v[:, None, :, :] + self.relative_value(rel)
        context = (weights[..., None] * values).sum(dim=2)
        output = self.fuse(torch.cat((actors, context), dim=-1))
        return output * actor_valid.to(output.dtype)[..., None]


class ActorToMapAttention(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relative_bias = nn.Sequential(nn.Linear(2, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def forward(self, actors: torch.Tensor, lanes: torch.Tensor,
                actor_pos: torch.Tensor, lane_pos: torch.Tensor,
                actor_valid: torch.Tensor, lane_valid: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(actors), self.key(lanes), self.value(lanes)
        delta = lane_pos[:, None, :, :] - actor_pos[:, :, None, :]
        logits = torch.einsum("baw,blw->bal", q, k) / math.sqrt(q.shape[-1])
        logits = logits + self.relative_bias(delta.clamp(-2.0, 2.0)).squeeze(-1)
        weights = masked_softmax(logits, lane_valid[None, None, :], dim=-1)
        context = torch.einsum("bal,blw->baw", weights, v)
        return context * actor_valid.to(context.dtype)[..., None]


class DualGraphEncoder(nn.Module):
    """One actor-set block and one typed road-topology message block."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.topology = TopologyMessageLayer(width)
        self.actor_graph = ActorSelfAttention(width)
        self.actor_map = ActorToMapAttention(width)
        self.actor_fuse = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU())
        self.map_refine = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU())

    def encode_map(self, lane_features: torch.Tensor, lane_centers: torch.Tensor,
                   lane_valid: torch.Tensor, edge_index: torch.Tensor,
                   edge_type: torch.Tensor, edge_valid: torch.Tensor) -> torch.Tensor:
        lanes = self.topology(lane_features, lane_centers, lane_valid, edge_index, edge_type, edge_valid)
        return self.map_refine(lanes) * lane_valid.to(lanes.dtype)[None, :, None]

    def forward(self, actor_features: torch.Tensor, actor_position: torch.Tensor,
                actor_velocity: torch.Tensor, actor_heading: torch.Tensor,
                actor_valid: torch.Tensor, lane_ptr: torch.Tensor, lane_s: torch.Tensor,
                lane_features: torch.Tensor, lane_centers: torch.Tensor,
                lane_valid: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_valid: torch.Tensor,
                map_encoded: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lanes = lane_features if map_encoded else self.encode_map(
            lane_features, lane_centers, lane_valid, edge_index, edge_type, edge_valid)
        actor_context = self.actor_graph(actor_features, actor_position, actor_velocity,
                                         actor_heading, actor_valid, lane_ptr, lane_s)
        map_context = self.actor_map(actor_context, lanes, actor_position, lane_centers,
                                     actor_valid, lane_valid)
        actors = self.actor_fuse(torch.cat((actor_context, map_context, actor_features), dim=-1))
        actors = actors * actor_valid.to(actors.dtype)[..., None]
        return actors, lanes, map_context
