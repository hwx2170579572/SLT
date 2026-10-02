"""Pure decision tests for the frozen-checkpoint route-lane veto diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


FAST_DEV = Path(__file__).resolve().parent
if str(FAST_DEV) not in sys.path:
    sys.path.insert(0, str(FAST_DEV))

from diagnose_sorted_goalonly_causal_20261002 import route_veto_decision


def decide(action, **overrides):
    args = {
        "context_known": True,
        "context_fresh": True,
        "current_road": "-E1",
        "has_next_edge": True,
        "current_lane_label": 1,
        "target_lane_label": 0,
        "lane_command": 1,
        "target_lane_id": "-E1_1",
        "target_lane_reason": None,
        "target_lane_in_range": True,
    }
    args.update(overrides)
    return route_veto_decision(action, **args)


class RouteLaneVetoDecisionTests(unittest.TestCase):
    def test_known_eligible_to_ineligible_change_is_vetoed_without_speed_change(self):
        for requested_command in (-1, 1):
            with self.subTest(lane_command=requested_command):
                action = np.asarray([0.27, 0.81], dtype=np.float32)
                original = action.copy()
                output, audit = decide(action, lane_command=requested_command)

                self.assertIsNot(output, action)
                np.testing.assert_array_equal(action, original)
                self.assertEqual(np.asarray(output).shape, (2,))
                self.assertEqual(float(output[0]), float(original[0]))
                self.assertEqual(float(output[1]), 0.0)
                self.assertTrue(audit["veto_applied"])
                self.assertTrue(audit["speed_action_unchanged"])
                self.assertEqual(audit["current_lane_label"], 1)
                self.assertEqual(audit["target_lane_label"], 0)
                self.assertEqual(audit["lane_command_requested"], requested_command)
                self.assertEqual(audit["lane_command_after_veto"], 0)

    def test_unknown_stale_or_non_applicable_context_passes_through(self):
        cases = (
            ("unknown_context", {"context_known": False}),
            ("stale_context", {"context_fresh": False}),
            ("internal_road", {"current_road": ":J1_14"}),
            ("no_next_edge", {"has_next_edge": False}),
            ("current_lane_ineligible", {"current_lane_label": 0}),
            ("current_lane_unknown", {"current_lane_label": -1}),
            ("target_lane_unknown", {"target_lane_label": -1}),
            ("target_lane_eligible", {"target_lane_label": 1}),
            (
                "unclassifiable_target_lane",
                {"target_lane_id": None, "target_lane_label": None},
            ),
            ("target_lane_out_of_range", {"target_lane_in_range": False}),
            (
                "target_lane_unclassifiable",
                {
                    "target_lane_label": None,
                    "target_lane_reason": "route_context_unknown",
                },
            ),
            ("hold_command", {"lane_command": 0}),
        )
        for case_name, overrides in cases:
            with self.subTest(case=case_name):
                action = np.asarray([-0.23, -0.91], dtype=np.float32)
                original = action.copy()
                output, audit = decide(action, **overrides)
                self.assertIsNot(output, action)
                np.testing.assert_array_equal(action, original)
                np.testing.assert_array_equal(np.asarray(output), original)
                self.assertFalse(audit["veto_applied"])
                self.assertTrue(audit["speed_action_unchanged"])

    def test_audit_exposes_request_and_effective_hold_separately(self):
        action = np.asarray([0.4, -0.9], dtype=np.float32)
        _, audit = decide(
            action,
            lane_command=-1,
            target_lane_id="-E1_0",
            current_road="-E1",
        )

        required = {
            "veto_applied",
            "reason",
            "current_lane_label",
            "target_lane_label",
            "current_road",
            "has_next_edge",
            "target_lane_id",
            "lane_command_requested",
            "lane_command_after_veto",
            "speed_action_unchanged",
        }
        self.assertTrue(required.issubset(audit))
        self.assertTrue(audit["veto_applied"])
        self.assertEqual(audit["lane_command_requested"], -1)
        self.assertEqual(audit["lane_command_after_veto"], 0)


if __name__ == "__main__":
    unittest.main()
