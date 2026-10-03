"""Small deterministic geometry helpers used by scene-event collection."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from .schema import ZONE_BEYOND_HORIZON, ZONE_CV_VALID, ZONE_STATIONARY


def sumo_front_to_actor_state(front_xy: tuple[float, float] | np.ndarray, angle_degrees: float,
                              speed_mps: float, length_m: float) -> np.ndarray:
    """Convert SUMO clockwise-from-north pose to world Cartesian center state."""
    angle = math.radians(float(angle_degrees))
    fx, fy = math.sin(angle), math.cos(angle)
    front = np.asarray(front_xy, dtype=np.float64).reshape(2)
    center = front - 0.5 * float(length_m) * np.asarray((fx, fy), dtype=np.float64)
    speed = float(speed_mps)
    return np.asarray((center[0], center[1], fx, fy, speed * fx, speed * fy), dtype=np.float32)


def sumo_front_lane_position_to_center(front_lane_s_m: float, length_m: float) -> float:
    """Align TraCI front-bumper lanePosition with the center state above."""
    if not math.isfinite(float(front_lane_s_m)) or not math.isfinite(float(length_m)) or length_m < 0:
        raise ValueError("lane position and vehicle length must be finite; length nonnegative")
    return float(front_lane_s_m) - 0.5 * float(length_m)


def footprint_zone_distances(route_offset_m: float, entry_s_m: float, clear_s_m: float,
                             lane_start_s_m: float, actor_center_s_m: float,
                             actor_length_m: float) -> tuple[float, float]:
    """Distance from current vehicle center until its body enters/clears a zone.

    ``entry_s_m``/``clear_s_m`` are centerline zone boundaries. TraCI's
    front-bumper lane position must first be converted to center arclength; the
    footprint extension then occurs exactly once by half the vehicle length.
    """
    values = (route_offset_m, entry_s_m, clear_s_m, lane_start_s_m, actor_center_s_m, actor_length_m)
    if not all(math.isfinite(float(value)) for value in values) or actor_length_m < 0:
        raise ValueError("zone distances and actor dimensions must be finite")
    offset, entry, clear, lane_start, current_center, length = map(float, values)
    if clear < entry:
        raise ValueError("clear boundary must be at or after entry boundary")
    half_length = 0.5 * length
    zone_entry_route = offset + entry - lane_start - half_length
    zone_clear_route = offset + clear - lane_start + half_length
    current_route = offset + current_center - lane_start
    distance_entry = max(0.0, zone_entry_route - current_route)
    distance_clear = max(distance_entry, zone_clear_route - current_route)
    return distance_entry, distance_clear


def cv_zone_event(distance_entry_m: float, distance_clear_m: float, speed_mps: float,
                  horizon_s: float, *, stationary_threshold_mps: float = 0.1) -> tuple[int, float, float]:
    """Return deterministic constant-velocity occupancy event status and times.

    This is a geometric timing proxy, not a collision probability or learned forecast.
    """
    values = (distance_entry_m, distance_clear_m, speed_mps, horizon_s)
    if not all(math.isfinite(float(v)) for v in values):
        return 0, 0.0, 0.0
    entry, clear, speed, horizon = map(float, values)
    if entry < 0 or clear < entry or horizon < 0:
        return 0, 0.0, 0.0
    if speed < stationary_threshold_mps:
        return ZONE_STATIONARY, 0.0, 0.0
    t_entry, t_clear = entry / speed, clear / speed
    if t_entry > horizon or t_clear > horizon:
        return ZONE_BEYOND_HORIZON, 0.0, 0.0
    return ZONE_CV_VALID, t_entry, t_clear


def oriented_box_polygon(center_xy: tuple[float, float] | np.ndarray,
                         heading_xy: tuple[float, float] | np.ndarray,
                         length_m: float, width_m: float) -> Polygon:
    """Build the inclusive oriented rectangular footprint in world meters."""
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    heading = np.asarray(heading_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(heading))
    if (not np.isfinite(center).all() or not np.isfinite(heading).all() or
            norm <= 1e-12 or not math.isfinite(float(length_m)) or
            not math.isfinite(float(width_m)) or length_m <= 0 or width_m <= 0):
        raise ValueError("center/heading must be finite, with nonzero heading and positive dimensions")
    forward = heading / norm
    lateral = np.asarray((-forward[1], forward[0]), dtype=np.float64)
    half_l, half_w = float(length_m) / 2.0, float(width_m) / 2.0
    corners = (
        center + half_l * forward + half_w * lateral,
        center + half_l * forward - half_w * lateral,
        center - half_l * forward - half_w * lateral,
        center - half_l * forward + half_w * lateral,
    )
    return Polygon(corners)


def swept_footprint_zone_interval(line: Any, zone: Any, length_m: float, width_m: float,
                                  *, sample_step_m: float = 0.20) -> tuple[float, float] | None:
    """Return the center-arc interval whose tangent-aligned OBB touches ``zone``.

    The oriented box is evaluated at lane-centerline samples no farther than
    ``sample_step_m`` apart. The first and last touching positions are refined
    by bisection to 1 mm. If a curved line enters the same zone more than once,
    this function returns the conservative envelope from the first entry to
    the last clearance. This is a geometric sweep proxy, not a SUMO collision
    decision.
    """
    if line is None or zone is None or line.is_empty or zone.is_empty:
        return None
    if not math.isfinite(float(sample_step_m)) or sample_step_m <= 0:
        raise ValueError("sample_step_m must be finite and positive")
    # Validate sizes before allocating the sample grid.
    if (not math.isfinite(float(length_m)) or not math.isfinite(float(width_m)) or
            length_m <= 0 or width_m <= 0):
        raise ValueError("vehicle length and width must be finite and positive")
    line_length = float(line.length)
    if line_length <= 1e-9:
        return None
    sample_count = max(1, int(math.ceil(line_length / float(sample_step_m))))
    sample_s = np.linspace(0.0, line_length, sample_count + 1, dtype=np.float64)
    prepared_zone = prep(zone)
    tangent_probe = min(0.05, max(1e-3, line_length / 10000.0))

    def footprint_intersects(s_m: float) -> bool:
        center = line.interpolate(float(s_m))
        before = line.interpolate(max(0.0, float(s_m) - tangent_probe))
        after = line.interpolate(min(line_length, float(s_m) + tangent_probe))
        heading = np.asarray((after.x - before.x, after.y - before.y), dtype=np.float64)
        if float(np.linalg.norm(heading)) <= 1e-12:
            return False
        footprint = oriented_box_polygon((center.x, center.y), heading, length_m, width_m)
        return bool(prepared_zone.intersects(footprint))

    hits = np.fromiter((footprint_intersects(float(s)) for s in sample_s), dtype=bool,
                       count=len(sample_s))
    hit_indices = np.flatnonzero(hits)
    if not len(hit_indices):
        return None

    def refine_boundary(left: float, right: float, *, entering: bool) -> float:
        # Entering: left is outside/right inside. Leaving: left inside/right outside.
        while right - left > 1e-3:
            middle = (left + right) / 2.0
            hit = footprint_intersects(middle)
            if entering:
                if hit:
                    right = middle
                else:
                    left = middle
            else:
                if hit:
                    left = middle
                else:
                    right = middle
        return right if entering else left

    first, last = int(hit_indices[0]), int(hit_indices[-1])
    entry_s = float(sample_s[first])
    clear_s = float(sample_s[last])
    if first > 0:
        entry_s = refine_boundary(float(sample_s[first - 1]), entry_s, entering=True)
    if last + 1 < len(sample_s):
        clear_s = refine_boundary(clear_s, float(sample_s[last + 1]), entering=False)
    return entry_s, clear_s


def _line_intersection_points(intersection: Any) -> list[tuple[float, float]]:
    """Flatten a Shapely intersection into points suitable for line projection."""
    if intersection is None or getattr(intersection, "is_empty", True):
        return []
    geom_type = getattr(intersection, "geom_type", "")
    if geom_type == "Point":
        return [(float(intersection.x), float(intersection.y))]
    if geom_type in {"LineString", "LinearRing"}:
        coordinates = list(intersection.coords)
        if not coordinates:
            return []
        return [(float(x), float(y)) for x, y, *_ in coordinates]
    if hasattr(intersection, "geoms"):
        return [point for part in intersection.geoms for point in _line_intersection_points(part)]
    return []


def polyline_zone_interval(line: Any, zone: Any) -> tuple[float, float] | None:
    """Return centerline entry/clear arclengths in meters for a polygon."""
    if line is None or zone is None or line.is_empty or zone.is_empty:
        return None
    crossing = line.intersection(zone)
    points = _line_intersection_points(crossing)
    if not points:
        # A centerline may just touch the geometry boundary. Keep this as a
        # zero-length boundary contact; callers can still label it as a proxy.
        try:
            from shapely.ops import nearest_points
            p_line, _ = nearest_points(line, zone)
            if not zone.buffer(1e-6).intersects(p_line):
                return None
            points = [(float(p_line.x), float(p_line.y))]
        except Exception:
            return None
    distances = [float(line.project(Point(point))) for point in points]
    return min(distances), max(distances)
