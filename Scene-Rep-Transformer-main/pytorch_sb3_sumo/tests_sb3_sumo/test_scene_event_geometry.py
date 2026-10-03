from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, box
from pathlib import Path
import sys

FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.geometry import (
    cv_zone_event,
    footprint_zone_distances,
    oriented_box_polygon,
    polyline_zone_interval,
    sumo_front_lane_position_to_center,
    sumo_front_to_actor_state,
    swept_footprint_zone_interval,
)
from scene_event.map_cache import _movement_paths
from scene_event.schema import ZONE_BEYOND_HORIZON, ZONE_CV_VALID, ZONE_STATIONARY


def test_sumo_front_pose_and_lane_position_share_center_contract() -> None:
    # SUMO angle 90 degrees points east. Both the world x and lane arclength
    # center coordinates subtract exactly one half vehicle length.
    state = sumo_front_to_actor_state((12.0, 4.0), 90.0, 5.0, 4.0)
    center_s = sumo_front_lane_position_to_center(12.0, 4.0)
    assert np.allclose(state, (10.0, 4.0, 1.0, 0.0, 5.0, 0.0), atol=1e-6)
    assert center_s == 10.0


def test_centerline_zone_and_vehicle_footprint_expand_entry_clear_once() -> None:
    # The lane centerline zone is [20, 24]. At center s=10 and L=4, body
    # occupancy begins when its center reaches18 and ends after center reaches26.
    enter, clear = footprint_zone_distances(
        route_offset_m=0.0,
        entry_s_m=20.0,
        clear_s_m=24.0,
        lane_start_s_m=10.0,
        actor_center_s_m=10.0,
        actor_length_m=4.0,
    )
    assert enter == 8.0
    assert clear == 16.0
    status, t_enter, t_clear = cv_zone_event(enter, clear, 2.0, 10.0)
    assert status == ZONE_CV_VALID
    assert (t_enter, t_clear) == (4.0, 8.0)


def test_near_departure_negative_center_s_is_not_clipped_or_double_shifted() -> None:
    # Front bumper at lane s=0 implies center s=-2; the event distance from
    # that center includes the true upstream offset exactly once.
    center_s = sumo_front_lane_position_to_center(0.0, 4.0)
    assert center_s == -2.0
    enter, clear = footprint_zone_distances(0.0, 10.0, 12.0, center_s, center_s, 4.0)
    assert enter == 10.0
    assert clear == 16.0


def test_zone_geometry_interval_and_unknown_speed_horizon_states() -> None:
    lane = LineString([(0.0, 0.0), (30.0, 0.0)])
    interval = polyline_zone_interval(lane, box(10.0, -1.0, 15.0, 1.0))
    assert interval is not None
    assert np.allclose(interval, (10.0, 15.0), atol=1e-6)
    assert cv_zone_event(1.0, 2.0, 0.0, 3.0)[0] == ZONE_STATIONARY
    assert cv_zone_event(4.0, 8.0, 1.0, 3.0)[0] == ZONE_BEYOND_HORIZON


def test_tangent_obb_width_changes_contact_with_crossing_corridor() -> None:
    lane = LineString(((0.0, 0.0), (30.0, 0.0)))
    crossing_corridor = box(10.0, 1.2, 15.0, 2.0)
    assert swept_footprint_zone_interval(lane, crossing_corridor, 4.0, 2.0) is None
    wide_interval = swept_footprint_zone_interval(lane, crossing_corridor, 4.0, 2.6)
    assert wide_interval is not None
    assert np.allclose(wide_interval, (8.0, 17.0), atol=2e-3)


def test_parallel_offset_corridors_do_not_create_a_conflict_interval() -> None:
    lane = LineString(((0.0, 0.0), (30.0, 0.0)))
    offset_lane_corridor = box(0.0, 3.1, 30.0, 5.1)
    assert swept_footprint_zone_interval(lane, offset_lane_corridor, 4.0, 2.0) is None


def test_curved_tangent_sweep_responds_to_length_width_and_heading() -> None:
    # A cubic smooth turn from an eastbound approach to a northbound exit.
    control = np.asarray(((0.0, 0.0), (8.0, 0.0), (10.0, 2.0), (10.0, 10.0)))
    t_values = np.linspace(0.0, 1.0, 201)
    curve = np.asarray([
        (1.0 - t) ** 3 * control[0] + 3.0 * (1.0 - t) ** 2 * t * control[1]
        + 3.0 * (1.0 - t) * t ** 2 * control[2] + t ** 3 * control[3]
        for t in t_values
    ])
    curved_lane = LineString(curve)
    corner_zone = box(9.0, 0.0, 11.0, 3.0)
    compact = swept_footprint_zone_interval(curved_lane, corner_zone, 3.0, 1.5)
    longer = swept_footprint_zone_interval(curved_lane, corner_zone, 6.0, 1.5)
    wider = swept_footprint_zone_interval(curved_lane, corner_zone, 3.0, 2.5)
    assert compact is not None and longer is not None and wider is not None
    assert longer[0] < compact[0] and longer[1] > compact[1]
    assert wider[0] < compact[0] and wider[1] > compact[1]
    # At a curved path sample, the box rotates with its local tangent.
    assert not oriented_box_polygon((10.0, 0.0), (1.0, 0.0), 6.0, 1.5).equals_exact(
        oriented_box_polygon((10.0, 0.0), (0.0, 1.0), 6.0, 1.5), 1e-6
    )


def test_movement_path_stops_at_first_shared_exit_not_follow_tail() -> None:
    lanes = ("approach", ":J1_connector", "exit", "shared_follow_tail")
    successors = ((1,), (2,), (3,), ())
    paths, truncated = _movement_paths(1, lanes, successors)
    assert not truncated
    assert paths == [(1, 2)]
    assert 3 not in paths[0]
