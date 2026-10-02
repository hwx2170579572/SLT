"""Route-conditioned crossing times from current observations and static geometry.

This is a constant-speed *crossing-corridor proxy*, not a collision oracle.
Other interaction types and missing geometry remain explicitly unsupported.
No simulator, hidden social route, learned model, or future state is accessed.
"""
from __future__ import annotations

from functools import lru_cache
from math import isfinite, radians, sin
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "relation_valid", "ego_entry_distance", "ego_exit_distance",
    "foe_entry_distance", "foe_exit_distance", "ego_speed", "foe_speed",
    "ego_entry_time", "ego_exit_time", "foe_entry_time", "foe_exit_time",
    "ego_time_valid", "foe_time_valid", "signed_time_gap", "overlap_duration",
    "tangent_cos", "abs_tangent_sin", "ego_inside_corridor", "foe_inside_corridor",
    "crossing_candidate_support_fraction",
)
FEATURE_DIM = len(FEATURE_NAMES)
SCHEMA = "route_crossing_timing_v1"
SPEED_VALID_MIN_MPS = 0.2
MIN_CROSSING_ANGLE_DEG = 20.0


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def polyline_arclength(points: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, consecutive-deduplicated points and cumulative metres."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("A route polyline must have shape (n>=2, 2)")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite route geometry")
    keep = np.r_[True, np.linalg.norm(np.diff(array, axis=0), axis=1) > 1e-8]
    array = array[keep]
    if len(array) < 2:
        raise ValueError("Degenerate route polyline")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))]
    return array, cumulative


def project_point_to_polyline(position: Any, points: Any) -> tuple[float, float]:
    """Nearest projection: (arc length in metres, lateral error in metres).

    This helper is for observational calibration, not privileged route selection.
    Callers must also check candidate lane adherence and observation coverage.
    """
    path, cumulative = polyline_arclength(points)
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Position must be finite xy")
    delta = np.diff(path, axis=0)
    length_sq = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.sum((point - path[:-1]) * delta, axis=1) / length_sq, 0, 1)
    projection = path[:-1] + fraction[:, None] * delta
    distances = np.linalg.norm(projection - point, axis=1)
    index = int(np.argmin(distances))
    return (float(cumulative[index] + fraction[index] * np.sqrt(length_sq[index])),
            float(distances[index]))


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@lru_cache(maxsize=2048)
def _static_crossings(ego_key: tuple, foe_key: tuple) -> tuple[tuple[float, ...], ...]:
    """Cache crossings of fixed paths; current positions/speeds are never cached."""
    ego, es = polyline_arclength(ego_key)
    foe, fs = polyline_arclength(foe_key)
    p, q = ego[:-1, None, :], foe[None, :-1, :]
    r, s = np.diff(ego, axis=0)[:, None, :], np.diff(foe, axis=0)[None, :, :]
    le, lf = np.linalg.norm(r, axis=-1), np.linalg.norm(s, axis=-1)
    denominator = _cross(r, s)
    abs_sin = np.abs(denominator) / (le * lf)
    nonparallel = abs_sin >= sin(radians(MIN_CROSSING_ANGLE_DEG))
    safe_denominator = np.where(nonparallel, denominator, 1.0)
    a = _cross(q - p, s) / safe_denominator
    b = _cross(q - p, r) / safe_denominator
    intersects = nonparallel & (a >= -1e-8) & (a <= 1 + 1e-8) & (b >= -1e-8) & (b <= 1 + 1e-8)
    result: list[tuple[float, ...]] = []
    seen: set[tuple[float, float]] = set()
    for i, j in np.argwhere(intersects):
        ei, fj = int(i), int(j)
        se = float(es[ei] + np.clip(a[ei, fj], 0, 1) * le[ei, 0])
        sf = float(fs[fj] + np.clip(b[ei, fj], 0, 1) * lf[0, fj])
        identity = (round(se, 6), round(sf, 6))
        if identity in seen:
            continue
        seen.add(identity)
        te = r[ei, 0] / le[ei, 0]
        tf = s[0, fj] / lf[0, fj]
        xy = ego[ei] + np.clip(a[ei, fj], 0, 1) * r[ei, 0]
        result.append((se, sf, float(np.dot(te, tf)), float(abs_sin[ei, fj]),
                       float(xy[0]), float(xy[1]), float(te[0]), float(te[1]),
                       float(tf[0]), float(tf[1])))
    return tuple(result)


def _prepare_paths(actor: Mapping[str, Any] | None) -> tuple[list[dict], int]:
    prepared: list[dict] = []
    invalid = 0
    if not actor:
        return prepared, invalid
    for index, item in enumerate(actor.get("paths") or []):
        try:
            points, arc = polyline_arclength(item["points"])
            current = _finite_float(item.get("s_current"))
            if current is None or current < -1e-6 or current > arc[-1] + 1e-6:
                raise ValueError("Current arc outside path")
            prepared.append({
                "points": points, "key": tuple(map(tuple, points.tolist())),
                "s_current": float(np.clip(current, 0, arc[-1])),
                "path_id": str(item.get("path_id", index)),
                "lane_ids": list(item.get("lane_ids") or []),
            })
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
    return prepared, invalid


def _arrival_times(entry_distance: float, exit_distance: float, speed: float | None) -> dict:
    valid = speed is not None and speed >= SPEED_VALID_MIN_MPS
    return {
        "valid": valid,
        "reason": "constant_speed_assumption" if valid else "speed_missing_or_below_0p2_mps",
        "entry_s": max(0.0, entry_distance) / speed if valid else None,
        "exit_s": max(0.0, exit_distance) / speed if valid else None,
        "entry_left_censored": bool(entry_distance <= 0),
        "inside_corridor": bool(entry_distance <= 0 < exit_distance),
    }


def compute_route_conflict_timing(
    actor_states: Sequence[Mapping[str, Any] | None], *, horizon_s: float = 10.0,
    max_distance_m: float = 60.0, length_m: float = 4.7, width_m: float = 1.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute 20 bounded scalar features per observed actor, ego row zero.

    Every actor has ``key``, ``speed_mps`` and candidate ``paths``. A path has
    fixed world-xy ``points``, projected current ``s_current``, ``path_id`` and
    ``lane_ids``. Ego paths follow its known task route. Social paths must come
    from public legal continuations, never a social vehicle's hidden plan.

    A supported path pair crosses at >=20 degrees. The crossing corridor uses
    fixed, disclosed body dimensions, not hidden per-vehicle sizes. Parallel,
    following, merging and opposing encounters are not certified safe; they
    fall back to the unmodified base representation. Timing is a current-speed
    assumption; it is not the outcome of a counterfactual action or a guarantee.
    """
    if horizon_s <= 0 or max_distance_m <= 0 or length_m <= 0 or width_m <= 0:
        raise ValueError("Timing scales and proxy dimensions must be positive")
    features = np.zeros((len(actor_states), FEATURE_DIM), dtype=np.float32)
    detail: dict[str, Any] = {
        "schema": SCHEMA, "feature_names": list(FEATURE_NAMES), "actors": [],
        "horizon_s": float(horizon_s), "max_distance_m": float(max_distance_m),
        "proxy_length_m": float(length_m), "proxy_width_m": float(width_m),
        "speed_valid_min_mps": SPEED_VALID_MIN_MPS,
        "minimum_crossing_angle_degrees": MIN_CROSSING_ANGLE_DEG,
        "scope": "candidate_route_crossing_corridor_constant_speed",
        "not_a_collision_or_safety_guarantee": True,
    }
    if not actor_states:
        detail["valid_relation_count"] = 0
        return features, detail
    ego = actor_states[0]
    ego_paths, ego_invalid = _prepare_paths(ego)
    ego_speed = _finite_float(ego.get("speed_mps")) if ego else None
    if ego_speed is not None and ego_speed < 0:
        ego_speed = None
    detail["ego_key"] = ego.get("key") if ego else None
    detail["ego_valid_path_count"] = len(ego_paths)
    detail["ego_invalid_path_count"] = ego_invalid
    for actor_index, actor in enumerate(actor_states[1:], start=1):
        row: dict[str, Any] = {
            "actor_index": actor_index, "actor_key": actor.get("key") if actor else None,
            "relation_valid": False, "absence_is_not_safety": True,
            "geometry_status": actor.get("geometry_status", "unspecified") if actor else "unobserved_or_padding",
        }
        detail["actors"].append(row)
        paths, invalid = _prepare_paths(actor)
        row.update({"valid_path_count": len(paths), "invalid_path_count": invalid})
        if not actor or not paths or not ego_paths:
            row["reason"] = "actor_or_route_geometry_unavailable"
            continue
        foe_speed = _finite_float(actor.get("speed_mps"))
        if foe_speed is not None and foe_speed < 0:
            foe_speed = None
        options: list[dict] = []
        supported_pairs: set[tuple[int, int]] = set()
        for ei, ep in enumerate(ego_paths):
            for fi, fp in enumerate(paths):
                for crossing in _static_crossings(ep["key"], fp["key"]):
                    se, sf, cosine, sine, x, y, tex, tey, tfx, tfy = crossing
                    # Vehicle centres traverse a corridor around the other road.
                    # This geometric proxy is symmetric and not an SSM-device output.
                    half_span = length_m / 2 + width_m / 2 * (1 + abs(cosine)) / sine
                    de, df = se - ep["s_current"], sf - fp["s_current"]
                    ee, ex, fe, fx = de - half_span, de + half_span, df - half_span, df + half_span
                    if ex <= 0 or fx <= 0 or ee > max_distance_m or fe > max_distance_m:
                        continue
                    supported_pairs.add((ei, fi))
                    et = _arrival_times(ee, ex, ego_speed)
                    ft = _arrival_times(fe, fx, foe_speed)
                    both = bool(et["valid"] and ft["valid"])
                    gap = max(et["entry_s"], ft["entry_s"]) - min(et["exit_s"], ft["exit_s"]) if both else None
                    overlap = max(0.0, -gap) if gap is not None else None
                    event = {
                        "ego_path_id": ep["path_id"], "foe_path_id": fp["path_id"],
                        "ego_lane_ids": ep["lane_ids"], "foe_lane_ids": fp["lane_ids"],
                        "ego_path_points": ep["points"].tolist(), "foe_path_points": fp["points"].tolist(),
                        "ego_current_s": ep["s_current"], "foe_current_s": fp["s_current"],
                        "ego_conflict_s": se, "foe_conflict_s": sf,
                        "ego_entry_arc_m": se - half_span, "ego_exit_arc_m": se + half_span,
                        "foe_entry_arc_m": sf - half_span, "foe_exit_arc_m": sf + half_span,
                        "ego_entry_distance_m": ee, "ego_exit_distance_m": ex,
                        "foe_entry_distance_m": fe, "foe_exit_distance_m": fx,
                        "ego_speed_mps": ego_speed, "foe_speed_mps": foe_speed,
                        "ego_time": et, "foe_time": ft,
                        "ego_entry_time_s": et["entry_s"], "ego_exit_time_s": et["exit_s"],
                        "foe_entry_time_s": ft["entry_s"], "foe_exit_time_s": ft["exit_s"],
                        "ego_time_valid": et["valid"], "foe_time_valid": ft["valid"],
                        "signed_time_gap_s": gap, "overlap_duration_s": overlap,
                        "tangent_cos": cosine, "abs_tangent_sin": sine,
                        "conflict_xy": [x, y], "ego_tangent": [tex, tey], "foe_tangent": [tfx, tfy],
                        "half_corridor_span_m": half_span,
                    }
                    # No candidate probability is assumed. Prefer a possible temporal
                    # overlap, then nearest time separation, then nearest geometry.
                    if both:
                        priority = (0 if gap <= 0 else 1, max(et["entry_s"], ft["entry_s"]) if gap <= 0 else gap,
                                    max(de, 0) + max(df, 0), ei, fi)
                    else:
                        priority = (2, max(de, 0) + max(df, 0), 0.0, ei, fi)
                    event["_priority"] = priority
                    options.append(event)
        pair_count = len(ego_paths) * len(paths)
        row.update({"candidate_pair_count": pair_count, "supported_candidate_pair_count": len(supported_pairs),
                    "supported_crossing_count": len(options)})
        if not options:
            row["reason"] = "no_supported_crossing_in_known_geometry_horizon"
            continue
        selected = min(options, key=lambda item: item["_priority"])
        selected.pop("_priority")
        support = len(supported_pairs) / pair_count
        row.update({"relation_valid": True, "reason": "supported_candidate_crossing",
                    "candidate_support_fraction": support, "selected": selected})
        # Also expose selected scalar fields directly for streaming recorders.
        row.update({key: value for key, value in selected.items() if not key.endswith("path_points")})
        et, ft = selected["ego_time"], selected["foe_time"]
        f = features[actor_index]
        f[0] = 1
        f[1:5] = np.clip(np.asarray([selected[k] for k in ("ego_entry_distance_m", "ego_exit_distance_m",
                          "foe_entry_distance_m", "foe_exit_distance_m")]) / max_distance_m, -2, 2)
        f[5:7] = np.clip(np.asarray([ego_speed or 0.0, foe_speed or 0.0]) / 15.0, 0, 2)
        f[7:11] = np.clip(np.asarray([et["entry_s"] or 0.0, et["exit_s"] or 0.0,
                                     ft["entry_s"] or 0.0, ft["exit_s"] or 0.0]) / horizon_s, 0, 2)
        f[11:13] = [float(et["valid"]), float(ft["valid"])]
        if selected["signed_time_gap_s"] is not None:
            f[13] = np.clip(selected["signed_time_gap_s"] / horizon_s, -2, 2)
            f[14] = np.clip(selected["overlap_duration_s"] / horizon_s, 0, 2)
        f[15:20] = [selected["tangent_cos"], selected["abs_tangent_sin"],
                    float(et["inside_corridor"]), float(ft["inside_corridor"]), support]
        row["time_values_clipped_in_policy_features"] = any(
            value is not None and value > 2 * horizon_s
            for value in (et["entry_s"], et["exit_s"], ft["entry_s"], ft["exit_s"]))
        row["beyond_calibration_horizon"] = bool(
            not (et["valid"] and ft["valid"]) or max(et["exit_s"], ft["exit_s"]) > horizon_s)
    detail["valid_relation_count"] = int(np.count_nonzero(features[:, 0]))
    detail["valid_joint_timing_count"] = int(np.count_nonzero(features[:, 11] * features[:, 12]))
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite conflict policy features")
    return features, detail
