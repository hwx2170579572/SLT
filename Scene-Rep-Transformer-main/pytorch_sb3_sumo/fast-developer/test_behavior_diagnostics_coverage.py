"""Guard against reporting missing risk/time measurements as safe zeroes."""
import json
import tempfile
import unittest
from pathlib import Path

from behavior_diagnostics import BehaviorDiagnosticsRecorder


class CoverageSemanticsTests(unittest.TestCase):
    def run_episode(self, *, known_geometry=True, step_seconds=None, ego_source="current"):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "eval", "fixture")
            ego = {"id": "ego", "kind": "vehicle", "position": [0, 0],
                   "velocity": [0, 0], "heading": 0, "speed": 0,
                   "length": 4, "width": 2}
            neighbor = dict(ego, id="neighbor", position=[10, 0])
            if not known_geometry:
                neighbor.pop("width")
            snapshot = {"ego": ego, "vehicles": [neighbor], "sim_time": 1,
                        "raw_step": 1, "observed_neighbor_ids": [],
                        "ego_state_source": ego_source, "step_seconds": step_seconds}
            recorder.on_reset(snapshot, seed=123)
            recorder.on_decision_start([0])
            recorder.on_raw_step(snapshot)
            recorder.on_decision_end(0, True, False, {"success": True})
            recorder.on_policy_step([0], 0, True, False, {"success": True})
            recorder.close()
            path = Path(directory) / "diagnostics" / "eval" / "episodes.jsonl"
            return json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    def test_unknown_geometry_has_no_valid_risk_denominator(self):
        episode = self.run_episode(known_geometry=False)
        self.assertEqual(episode["risk_evaluable_ticks"], 0)
        self.assertIsNone(episode["low_ttc_fraction_of_evaluable_ticks"])
        self.assertIsNone(episode["critical_unobserved_fraction_of_covered_ticks"])

    def test_known_safe_tick_and_missing_clock_are_distinct(self):
        episode = self.run_episode()
        self.assertEqual(episode["risk_evaluable_ticks"], 1)
        self.assertEqual(episode["low_ttc_fraction_of_evaluable_ticks"], 0)
        self.assertEqual(episode["stopped_fraction_of_speed_samples"], 1)
        self.assertIsNone(episode["stopped_seconds"])
        self.assertIsNone(episode["risk_observed_seconds"])
        self.assertEqual(episode["speed_duration_samples"], 0)

    def test_measured_interval_and_removed_pose(self):
        measured = self.run_episode(step_seconds=0.1)
        self.assertAlmostEqual(measured["stopped_seconds"], 0.1)
        self.assertAlmostEqual(measured["risk_observed_seconds"], 0.1)
        removed = self.run_episode(step_seconds=0.1, ego_source="pre_step_removed")
        self.assertEqual(removed["risk_evaluable_ticks"], 0)
        self.assertEqual(removed["speed_samples"], 0)
        self.assertIsNone(removed["mean_actual_speed_mps"])

    def test_route_ineligible_stall_requires_known_current_context_and_actual_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "eval", "fixture")
            recorder.on_reset({}, seed=22)
            base = {
                "ego": {"id": "ego", "kind": "vehicle", "speed": 0.05},
                "ego_state_source": "current",
                "step_seconds": 0.1,
                "route_lane_context_known": True,
                "planned_next_edge": "edge-next",
                "current_lane_can_reach_next_edge": False,
            }
            for raw_step in (1, 2):
                recorder.on_raw_step(dict(base, raw_step=raw_step))
            # Unknown route context is excluded and breaks the consecutive stall.
            recorder.on_raw_step(
                dict(base, raw_step=3, route_lane_context_known=False)
            )
            # A known but moving vehicle is ineligible, but is not a stopped stall.
            recorder.on_raw_step(
                dict(
                    base,
                    raw_step=4,
                    ego={"id": "ego", "kind": "vehicle", "speed": 1.0},
                )
            )
            recorder.close()

            episode_path = (
                Path(directory) / "diagnostics" / "eval" / "episodes.jsonl"
            )
            episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(episode["route_lane_known_ticks"], 3)
            self.assertEqual(episode["route_lane_unknown_ticks"], 1)
            self.assertEqual(episode["route_lane_ineligible_ticks"], 3)
            self.assertEqual(episode["route_lane_ineligible_stopped_ticks"], 2)
            self.assertAlmostEqual(episode["route_lane_ineligible_seconds"], 0.3)
            self.assertAlmostEqual(episode["route_lane_ineligible_stopped_seconds"], 0.2)
            self.assertAlmostEqual(episode["route_lane_ineligible_stall_max_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
