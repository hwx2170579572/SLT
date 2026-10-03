"""Numerically safe projection of deterministic CV zone intervals."""
from __future__ import annotations

import torch


def project_event_intervals(event_seconds: torch.Tensor, event_mask: torch.Tensor,
                            event_status: torch.Tensor, prediction_seconds: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized [entry, clear, duration, order] features and valid mask.

    Only status=2 (finite constant-velocity projection) and the explicit mask
    enter the interval representation. Other statuses remain available to the
    event encoder as categorical coverage information. This consumes a
    deterministic geometry-derived estimate, not a learned or observed future.
    """
    if prediction_seconds <= 0:
        raise ValueError("prediction_seconds must be positive")
    if event_seconds.shape[-1] != 2 or event_mask.shape != event_status.shape or event_mask.shape != event_seconds.shape[:-1]:
        raise ValueError("event intervals/mask/status shapes are inconsistent")
    finite = torch.isfinite(event_seconds).all(dim=-1)
    valid = event_mask.bool() & (event_status == 2) & finite
    entry = torch.nan_to_num(event_seconds[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    clear = torch.nan_to_num(event_seconds[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    ordered = clear >= entry
    valid = valid & ordered
    scale = float(prediction_seconds)
    duration = (clear - entry).clamp_min(0.0)
    order = torch.where(entry <= clear, 1.0, -1.0).to(event_seconds.dtype)
    features = torch.stack((entry.clamp(0.0, scale) / scale,
                            clear.clamp(0.0, scale) / scale,
                            duration.clamp(0.0, scale) / scale,
                            order), dim=-1)
    return features * valid.to(features.dtype)[..., None], valid, duration


def events_off_observation(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy an observation and delete M1 event information for OOD sensitivity."""
    result = dict(observation)
    if "candidate_zone_event_mask" in observation:
        result["candidate_zone_event_mask"] = torch.zeros_like(observation["candidate_zone_event_mask"], dtype=torch.bool)
    if "candidate_zone_event_s" in observation:
        result["candidate_zone_event_s"] = torch.zeros_like(observation["candidate_zone_event_s"])
    if "candidate_zone_status" in observation:
        result["candidate_zone_status"] = torch.zeros_like(observation["candidate_zone_status"])
    return result
