"""Analytic contracts for actor-footprint sweeps and local movement resources.

These tests use hand-computed geometry and tiny lane graphs. They do not run
SUMO, and their expected intervals do not reuse the implementation's sampling
or polygon-expansion code.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, box

# map_cache.py imports the sibling SUMO topology package; make tests runnable
# both from fast-developer and from the pytorch_sb3_sumo project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.sumo.topology_graph_v2 import MERGE, SUCCESSOR
from scene_event.geometry import (
    cv_zone_event,
    footprint_zone_distances,
    oriented_box_polygon,
    swept_footprint_zone_interval,
)
from scene_event.map_cache import _compile_zones
from scene_event.schema import ZONE_BEYOND_HORIZON, ZONE_CV_VALID, ZONE_STATIONARY, ZONE_UNKNOWN


def assert_interval(test: unittest.TestCase, actual: tuple[float, float] | None,
                    expected: tuple[float, float], tolerance: float = 0.002) -> None:
    test.assertIsNotNone(actual)
    assert actual is not None
    test.assertAlmostEqual(actual[0], expected[0], delta=tolerance)
    test.assertAlmostEqual(actual[1], expected[1], delta=tolerance)


class _FakeEdge:
    def __init__(self, edge_id: str):
        self.edge_id = edge_id

    def getID(self) -> str:
        return self.edge_id


class _FakeLane:
    def __init__(self, lane_id: str, edge_id: str, lane_index: int = 0, width: float = 2.0):
        self.lane_id = lane_id
        self.edge = _FakeEdge(edge_id)
        self.lane_index = lane_index
        self.width = width

    def getID(self) -> str:
        return self.lane_id

    def getEdge(self) -> _FakeEdge:
        return self.edge

    def getIndex(self) -> int:
        return self.lane_index

    def getWidth(self) -> float:
        return self.width


class SweptGeometryContractTests(unittest.TestCase):
    def test_right_angle_cross_has_analytic_length_dependent_entry_and_clearance(self) -> None:
        # A horizontal vehicle crosses the vertical movement rectangle. Its
        # center enters when the front reaches x=-2 and clears when the rear
        # passes x=+2.
        lane = LineString([(-20.0, 25.0), (20.0, 25.0)])
        vertical_movement = box(-2.0, 20.0, 2.0, 30.0)

        assert_interval(self, swept_footprint_zone_interval(lane, vertical_movement, 4.0, 2.0),
                        (16.0, 24.0))
        assert_interval(self, swept_footprint_zone_interval(lane, vertical_movement, 8.0, 2.0),
                        (14.0, 26.0))

    def test_width_controls_lateral_overlap_without_changing_crossing_interval(self) -> None:
        lane = LineString([(-20.0, 31.0), (20.0, 31.0)])
        movement = box(-2.0, 20.0, 2.0, 30.0)

        # The centerline is 1 m beyond the movement. A 0.9 m half-width stays
        # clear; a 1.1 m half-width overlaps, while longitudinal limits stay
        # set by the same x boundaries and vehicle length.
        self.assertIsNone(swept_footprint_zone_interval(lane, movement, 4.0, 1.8))
        assert_interval(self, swept_footprint_zone_interval(lane, movement, 4.0, 2.2),
                        (16.0, 24.0))

    def test_parallel_laterally_separated_lane_has_no_swept_overlap(self) -> None:
        lane = LineString([(-20.0, 32.1), (20.0, 32.1)])
        movement = box(-2.0, 20.0, 2.0, 30.0)
        self.assertIsNone(swept_footprint_zone_interval(lane, movement, 8.0, 2.0))

    def test_diagonal_heading_uses_vehicle_local_length_and_width_axes(self) -> None:
        root_half = math.sqrt(0.5)
        forward = np.asarray((root_half, root_half), dtype=np.float64)
        lateral = np.asarray((-root_half, root_half), dtype=np.float64)
        lane = LineString([(0.0, 0.0), (50.0 * root_half, 50.0 * root_half)])
        # This is a 10 m long, 6 m wide rectangle oriented with the lane. A
        # length-4/width-2 vehicle fits laterally, so only the 2 m half-length
        # expands its route interval from [20, 30] to [18, 32].
        corners = [20.0 * forward - 3.0 * lateral,
                   30.0 * forward - 3.0 * lateral,
                   30.0 * forward + 3.0 * lateral,
                   20.0 * forward + 3.0 * lateral]
        movement = Polygon(corners)
        assert_interval(self, swept_footprint_zone_interval(lane, movement, 4.0, 2.0),
                        (18.0, 32.0))

    def test_curved_lane_uses_local_heading_after_the_turn(self) -> None:
        # The turn is complete well before the movement. On its northbound
        # segment, the vehicle's 2 m half-length extends entry/clearance along
        # y while its width remains lateral to the road.
        lane = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 20.0)])
        movement = box(8.0, 10.0, 12.0, 12.0)
        assert_interval(self, swept_footprint_zone_interval(lane, movement, 4.0, 2.0),
                        (18.0, 24.0))

    def test_same_zone_distance_updates_across_ticks_and_route_lane_boundary(self) -> None:
        zone = box(20.0, -2.0, 30.0, 2.0)
        before_tick = LineString([(0.0, 0.0), (50.0, 0.0)])
        after_three_meters = LineString([(3.0, 0.0), (50.0, 0.0)])
        first = swept_footprint_zone_interval(before_tick, zone, 4.0, 2.0)
        second = swept_footprint_zone_interval(after_three_meters, zone, 4.0, 2.0)
        assert_interval(self, first, (18.0, 32.0))
        assert_interval(self, second, (15.0, 29.0))

        # Equivalent route coordinates before and after moving onto the next
        # lane retain the same center-to-footprint distances.
        previous_lane = footprint_zone_distances(0.0, 15.0, 20.0, 0.0, 12.0, 4.0)
        next_lane = footprint_zone_distances(10.0, 5.0, 10.0, 0.0, 2.0, 4.0)
        np.testing.assert_allclose(previous_lane, (1.0, 10.0), atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(next_lane, previous_lane, atol=1e-7, rtol=0.0)

    def test_stationary_unknown_and_horizon_states_do_not_create_arrival_times(self) -> None:
        self.assertEqual(cv_zone_event(10.0, 18.0, 0.0, 3.0), (ZONE_STATIONARY, 0.0, 0.0))
        self.assertEqual(cv_zone_event(10.0, 18.0, 0.05, 3.0), (ZONE_STATIONARY, 0.0, 0.0))
        self.assertEqual(cv_zone_event(float("nan"), 18.0, 2.0, 3.0), (ZONE_UNKNOWN, 0.0, 0.0))
        self.assertEqual(cv_zone_event(10.0, 18.0, 2.0, 3.0),
                         (ZONE_BEYOND_HORIZON, 0.0, 0.0))
        self.assertEqual(cv_zone_event(2.0, 4.0, 2.0, 3.0), (ZONE_CV_VALID, 1.0, 2.0))

    def test_box_heading_normalization_and_invalid_dimensions(self) -> None:
        unit = oriented_box_polygon((2.0, 3.0), (0.0, 2.0), length_m=6.0, width_m=2.0)
        normalized = oriented_box_polygon((2.0, 3.0), (0.0, 1.0), length_m=6.0, width_m=2.0)
        self.assertEqual(unit.bounds, normalized.bounds)
        self.assertEqual(unit.bounds, (1.0, 0.0, 3.0, 6.0))
        with self.assertRaises(ValueError):
            oriented_box_polygon((0.0, 0.0), (0.0, 0.0), length_m=4.0, width_m=2.0)
        with self.assertRaises(ValueError):
            swept_footprint_zone_interval(LineString([(0.0, 0.0), (1.0, 0.0)]), box(0, 0, 1, 1), 0.0, 2.0)

    def test_merge_zone_captures_local_convergence_but_excludes_shared_exit_tail(self) -> None:
        lane_ids = ("A_0", ":J_0", "B_0", "C_0", ":J_1")
        lanes = {
            "A_0": _FakeLane("A_0", "A"),
            ":J_0": _FakeLane(":J_0", ":J"),
            "B_0": _FakeLane("B_0", "B"),
            "C_0": _FakeLane("C_0", "C"),
            ":J_1": _FakeLane(":J_1", ":J"),
        }
        lane_lines = (
            LineString([(-10.0, 0.0), (-4.0, 0.0)]),
            LineString([(-4.0, 0.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (25.0, 0.0)]),
            LineString([(0.0, -10.0), (0.0, -4.0)]),
            LineString([(0.0, -4.0), (0.0, 0.0)]),
        )
        edge_index = np.asarray(((0, 1, 3, 4, 0), (1, 2, 4, 2, 3)), dtype=np.int64)
        edge_type = np.asarray((SUCCESSOR, SUCCESSOR, SUCCESSOR, SUCCESSOR, MERGE), dtype=np.int64)

        zones, zone_types, metadata = _compile_zones(lane_lines, lanes, edge_index, edge_type, lane_ids)
        self.assertEqual(len(zones), 1)
        self.assertEqual(zone_types.tolist(), [1])  # merge resource
        zone = zones[0]
        self.assertGreater(zone.intersection(box(-0.5, -0.5, 0.5, 0.5)).area, 0.1)
        # One 2 m exit-lane-width may follow the convergence; the 25 m shared
        # successor tail must not become part of the interaction resource.
        self.assertAlmostEqual(zone.bounds[2], 2.0, delta=1e-6)
        self.assertFalse(metadata["shared_exit_follow_tail_in_zone"])
        self.assertGreater(metadata["mapped_pairs_by_relation"]["merge"], 0)


if __name__ == "__main__":
    unittest.main()
