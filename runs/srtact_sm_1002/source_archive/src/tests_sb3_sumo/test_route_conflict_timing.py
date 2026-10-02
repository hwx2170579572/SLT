import json
import unittest

import numpy as np

from envs.sumo.route_conflict_timing import (
    FEATURE_DIM,
    compute_route_conflict_timing,
    polyline_arclength,
    project_point_to_polyline,
)


def actor(key, points, current_s=30.0, speed=10.0, **extra):
    return {
        'key': key,
        'speed_mps': speed,
        'geometry_status': 'known',
        'paths': [{'path_id': key, 'points': points, 's_current': current_s,
                   'lane_ids': [key + '_lane']}],
        **extra,
    }


def crossing():
    return [actor('ego', [[-50., 0.], [50., 0.]]),
            actor('foe', [[0., -50.], [0., 50.]])]


class RouteConflictTimingTests(unittest.TestCase):
    def test_perpendicular_overlap_uses_body_corridor(self):
        features, detail = compute_route_conflict_timing(crossing())
        self.assertEqual(features.shape, (2, FEATURE_DIM))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(features[0] == 0))
        event = detail['actors'][0]['selected']
        self.assertAlmostEqual(event['half_corridor_span_m'], 3.25)
        self.assertAlmostEqual(event['ego_entry_time_s'], 1.675)
        self.assertAlmostEqual(event['ego_exit_time_s'], 2.325)
        self.assertAlmostEqual(event['signed_time_gap_s'], -0.65)
        self.assertAlmostEqual(event['overlap_duration_s'], 0.65)
        self.assertEqual(features[1, 0], 1)
        self.assertEqual(features[1, 19], 1)
        json.dumps(detail, allow_nan=False)

    def test_nonoverlapping_arrival_is_positive_gap(self):
        states = crossing()
        states[1]['paths'][0]['s_current'] = 0.
        states[1]['speed_mps'] = 5.
        _, detail = compute_route_conflict_timing(states)
        event = detail['actors'][0]['selected']
        self.assertAlmostEqual(event['signed_time_gap_s'], 9.35 - 2.325)
        self.assertEqual(event['overlap_duration_s'], 0.)

    def test_stationary_is_unknown_timing_not_safe(self):
        states = crossing()
        states[0]['speed_mps'] = 0.
        features, detail = compute_route_conflict_timing(states)
        row = detail['actors'][0]
        self.assertTrue(row['relation_valid'])
        self.assertFalse(row['selected']['ego_time_valid'])
        self.assertIsNone(row['selected']['ego_entry_time_s'])
        self.assertIsNone(row['selected']['signed_time_gap_s'])
        self.assertEqual(features[1, 11], 0.)
        self.assertTrue(row['absence_is_not_safety'])
        json.dumps(detail, allow_nan=False)

    def test_inside_is_left_censored_and_cleared_is_excluded(self):
        states = crossing()
        states[0]['paths'][0]['s_current'] = 51.
        features, detail = compute_route_conflict_timing(states)
        timing = detail['actors'][0]['selected']['ego_time']
        self.assertTrue(timing['inside_corridor'])
        self.assertTrue(timing['entry_left_censored'])
        self.assertEqual(timing['entry_s'], 0.)
        self.assertAlmostEqual(timing['exit_s'], .225)
        self.assertEqual(features[1, 17], 1.)
        states[0]['paths'][0]['s_current'] = 60.
        features, detail = compute_route_conflict_timing(states)
        self.assertEqual(features[1, 0], 0.)
        self.assertFalse(detail['actors'][0]['relation_valid'])

    def test_parallel_opposing_padding_and_bad_paths_are_not_crossings(self):
        for points in ([[[-50., 1.], [50., 1.]]], [[[50., 0.], [-50., 0.]]]):
            states = [crossing()[0], actor('foe', points[0]), None]
            features, detail = compute_route_conflict_timing(states)
            self.assertFalse(np.any(features))
            self.assertTrue(all(r['absence_is_not_safety'] for r in detail['actors']))
        states = [crossing()[0], actor('foe', [[0., 0.], [0., 0.]])]
        features, detail = compute_route_conflict_timing(states)
        self.assertFalse(np.any(features))
        json.dumps(detail, allow_nan=False)

    def test_rotation_translation_invariance_and_real_x_zero(self):
        states = crossing()
        expected, _ = compute_route_conflict_timing(states)
        angle = .71
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        for state in states:
            path = state['paths'][0]
            path['points'] = (np.asarray(path['points']) @ rot.T + [13., -71.]).tolist()
        actual, _ = compute_route_conflict_timing(states)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_duplicate_vertices_and_segment_endpoint_do_not_duplicate_event(self):
        states = crossing()
        states[0]['paths'][0]['points'] = [[-50., 0.], [0., 0.], [0., 0.], [50., 0.]]
        states[1]['paths'][0]['points'] = [[0., -50.], [0., 0.], [0., 50.]]
        feature, detail = compute_route_conflict_timing(states)
        plain, _ = compute_route_conflict_timing(crossing())
        np.testing.assert_allclose(feature, plain)
        self.assertEqual(detail['actors'][0]['candidate_pair_count'], 1)

    def test_multiple_paths_support_is_candidate_fraction_not_probability(self):
        states = crossing()
        states[1]['paths'].append({'path_id': 'noncrossing', 'points': [[-50., 10.], [50., 10.]],
                                   's_current': 30., 'lane_ids': ['other']})
        feature, detail = compute_route_conflict_timing(states)
        self.assertAlmostEqual(feature[1, 19], .5)
        self.assertEqual(detail['actors'][0]['selected']['foe_path_id'], 'foe')

    def test_nonfinite_speed_has_explicit_unknown_and_finite_feature(self):
        states = crossing()
        states[1]['speed_mps'] = float('nan')
        feature, detail = compute_route_conflict_timing(states)
        self.assertTrue(np.all(np.isfinite(feature)))
        self.assertFalse(detail['actors'][0]['selected']['foe_time_valid'])
        json.dumps(detail, allow_nan=False)

    def test_polyline_projection(self):
        points, arc = polyline_arclength([[0., 0.], [0., 0.], [10., 0.], [10., 10.]])
        np.testing.assert_allclose(arc, [0., 10., 20.])
        s, distance = project_point_to_polyline([8., 2.], points)
        self.assertAlmostEqual(s, 8.)
        self.assertAlmostEqual(distance, 2.)


if __name__ == '__main__':
    unittest.main()
