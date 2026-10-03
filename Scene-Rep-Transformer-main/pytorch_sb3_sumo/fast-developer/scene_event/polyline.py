"""Public-map buffers and lane/zone polyline encoders."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PublicMap(nn.Module):
    """Validated, non-trainable map tensors; dimensions are never truncated."""

    REQUIRED = {
        "lane_points_xy", "lane_attrs", "lane_valid", "edge_index", "edge_type",
        "edge_valid", "ego_route_lane_mask", "zone_polygons_xy", "zone_vertex_valid",
        "lane_zone_s_m", "lane_zone_valid", "zone_type", "zone_valid",
    }
    # Collector-only broad-phase geometry. Validate the field at the boundary,
    # but do not register it as a policy buffer or expose it to model features.
    OPTIONAL_AUXILIARY = {"lane_zone_distance_m"}

    def __init__(self, values: dict[str, np.ndarray | torch.Tensor]):
        super().__init__()
        missing = self.REQUIRED - set(values)
        extra = set(values) - self.REQUIRED - self.OPTIONAL_AUXILIARY
        if missing or extra:
            raise ValueError(f"public_map keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        arrays = {key: np.asarray(value) for key, value in values.items()}
        points = arrays["lane_points_xy"]
        if points.ndim != 3 or points.shape[1:] != (10, 2):
            raise ValueError("lane_points_xy must have actual shape [L,10,2]")
        lanes = points.shape[0]
        if lanes == 0:
            raise ValueError("public map must contain at least one lane")
        shapes = {
            "lane_attrs": (lanes, 8), "lane_valid": (lanes,),
            "edge_index": (2, arrays["edge_index"].shape[1] if arrays["edge_index"].ndim == 2 else -1),
            "edge_type": (arrays["edge_index"].shape[1] if arrays["edge_index"].ndim == 2 else -1,),
            "edge_valid": (arrays["edge_index"].shape[1] if arrays["edge_index"].ndim == 2 else -1,),
            "ego_route_lane_mask": (lanes,),
        }
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"public_map[{key}] expected {shape}, got {arrays[key].shape}")
        if arrays["edge_index"].ndim != 2 or arrays["edge_index"].shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        zones = arrays["zone_polygons_xy"]
        if zones.ndim != 3 or zones.shape[0] < 1 or zones.shape[2] != 2:
            raise ValueError("zone_polygons_xy must be [Z,V,2] with Z>=1")
        zcount, vertices, _ = zones.shape
        expected = {
            "zone_vertex_valid": (zcount, vertices), "lane_zone_s_m": (lanes, zcount, 2),
            "lane_zone_valid": (lanes, zcount), "zone_type": (zcount,), "zone_valid": (zcount,),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise ValueError(f"public_map[{key}] expected {shape}, got {arrays[key].shape}")
        if "lane_zone_distance_m" in arrays:
            distance = arrays["lane_zone_distance_m"]
            if distance.shape != (lanes, zcount):
                raise ValueError(
                    "public_map[lane_zone_distance_m] must be centerline-to-zone distance [L,Z] in meters"
                )
            if distance.dtype != np.float32 or not np.isfinite(distance).all() or np.any(distance < 0):
                raise ValueError("public_map[lane_zone_distance_m] must be float32, finite, and nonnegative")
        if arrays["edge_index"].size:
            valid_edges = arrays["edge_valid"].astype(bool)
            endpoints = arrays["edge_index"][:, valid_edges]
            if endpoints.size and (endpoints.min() < 0 or endpoints.max() >= lanes):
                raise ValueError("A valid topology edge points outside the actual lane array")
        if np.any((arrays["edge_type"] < 0) | (arrays["edge_type"] > 5)):
            raise ValueError("edge_type codes must be in [0,5]")
        if np.any((arrays["zone_type"] < 0) | (arrays["zone_type"] > 1)):
            raise ValueError("zone_type codes must be crossing=0 or merge=1")
        if not np.isfinite(arrays["lane_points_xy"]).all() or not np.isfinite(arrays["lane_attrs"]).all():
            raise ValueError("static lane geometry/attributes must be finite")
        if not np.isfinite(arrays["zone_polygons_xy"]).all() or not np.isfinite(arrays["lane_zone_s_m"]).all():
            raise ValueError("static zone geometry/intervals must be finite")

        dtypes = {
            "lane_points_xy": torch.float32, "lane_attrs": torch.float32,
            "lane_valid": torch.bool, "edge_index": torch.long, "edge_type": torch.long,
            "edge_valid": torch.bool, "ego_route_lane_mask": torch.bool,
            "zone_polygons_xy": torch.float32, "zone_vertex_valid": torch.bool,
            "lane_zone_s_m": torch.float32, "lane_zone_valid": torch.bool,
            "zone_type": torch.long, "zone_valid": torch.bool,
        }
        for key, dtype in dtypes.items():
            tensor = torch.as_tensor(arrays[key], dtype=dtype).clone().detach()
            self.register_buffer(key, tensor, persistent=True)
        self.num_lanes = lanes
        self.num_edges = arrays["edge_index"].shape[1]
        self.num_zones = zcount
        self.num_zone_vertices = vertices


class PolylineEncoder(nn.Module):
    """Lane and zone tokens in the instantaneous ego-aligned frame."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.lane_attr = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, width), nn.GELU())
        self.route_bit = nn.Linear(1, width, bias=False)
        self.lane_fuse = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU())
        self.lane_zone = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.zone_type = nn.Embedding(2, width)
        self.zone_fuse = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())
        self.interval_proj = nn.Sequential(nn.Linear(2, width), nn.GELU())

    @staticmethod
    def _local(points: torch.Tensor, ego_xy: torch.Tensor, ego_forward: torch.Tensor) -> torch.Tensor:
        delta = points - ego_xy[:, None, None, :]
        lateral = torch.stack((ego_forward[:, 1], -ego_forward[:, 0]), dim=-1)
        return torch.stack(((delta * ego_forward[:, None, None, :]).sum(-1),
                            (delta * lateral[:, None, None, :]).sum(-1)), dim=-1) / 80.0

    def forward(self, public_map: PublicMap, ego_xy: torch.Tensor,
                ego_forward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = ego_xy.shape[0]
        points = public_map.lane_points_xy.unsqueeze(0).expand(bsz, -1, -1, -1)
        local = self._local(points, ego_xy, ego_forward)
        point_emb = self.point(local)
        point_mean = point_emb.mean(dim=2)
        point_max = point_emb.max(dim=2).values
        attrs = self.lane_attr(public_map.lane_attrs).unsqueeze(0).expand(bsz, -1, -1)
        route = public_map.ego_route_lane_mask.to(point_emb.dtype)[None, :, None].expand(bsz, -1, -1)
        lane = self.lane_fuse(torch.cat((point_mean, point_max, attrs + self.route_bit(route)), dim=-1))
        lane = lane * public_map.lane_valid.to(lane.dtype)[None, :, None]
        lane_center_world = public_map.lane_points_xy.mean(dim=1)
        lane_center_local = self._local(lane_center_world[None, :, None, :].expand(bsz, -1, -1, -1),
                                        ego_xy, ego_forward).squeeze(2)

        zones = public_map.zone_polygons_xy.unsqueeze(0).expand(bsz, -1, -1, -1)
        zone_local = self._local(zones, ego_xy, ego_forward)
        zone_points = self.point(zone_local)
        vmask = public_map.zone_vertex_valid[None, :, :, None].expand(bsz, -1, -1, 1)
        denom = vmask.sum(dim=2).clamp_min(1).to(zone_points.dtype)
        zone_mean = (zone_points * vmask).sum(dim=2) / denom
        zone_max = zone_points.masked_fill(~vmask, torch.finfo(zone_points.dtype).min).max(dim=2).values
        zone_has_vertices = public_map.zone_vertex_valid.any(dim=-1)[None, :, None]
        zone_max = torch.where(zone_has_vertices, zone_max, torch.zeros_like(zone_max))
        zone_type = self.zone_type(public_map.zone_type.clamp(0, 1))[None].expand(bsz, -1, -1)
        zone = self.zone_fuse(torch.cat((zone_mean, zone_max + zone_type), dim=-1))
        zone = zone * public_map.zone_valid.to(zone.dtype)[None, :, None]

        # Attach lane-zone intervals through a map-derived message. These are
        # static geometry, not per-vehicle interaction observations.
        interval = public_map.lane_zone_s_m.clamp(0.0, 200.0) / 200.0
        interval_emb = self.interval_proj(interval)[None].expand(bsz, -1, -1, -1)
        link_mask = public_map.lane_zone_valid[None, :, :, None] & public_map.zone_valid[None, None, :, None]
        link = (zone[:, None, :, :] + interval_emb) * link_mask.to(zone.dtype)
        link_mean = link.sum(dim=2) / link_mask.sum(dim=2).clamp_min(1).to(zone.dtype)
        lane = (lane + self.lane_zone(link_mean)) * public_map.lane_valid.to(lane.dtype)[None, :, None]
        zone_centers = (zone_local * public_map.zone_vertex_valid[None, :, :, None]).sum(dim=2) / denom
        return lane, lane_center_local, zone, zone_centers
