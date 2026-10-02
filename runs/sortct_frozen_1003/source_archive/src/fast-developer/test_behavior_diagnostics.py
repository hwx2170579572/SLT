"""Small geometry and lifecycle tests; no SUMO, GPU, or training required."""
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from behavior_diagnostics import BehaviorDiagnosticsRecorder, cv_obb_ttc, pair_metrics


def actor(identity="ego", position=(0.0, 0.0), velocity=(10.0, 0.0), heading=0.0):
    return {"id": identity, "kind": "vehicle", "position": list(position),
            "velocity": list(velocity), "heading": heading,
            "speed": math.hypot(*velocity), "length": 4.0, "width": 2.0}


class GeometryTests(unittest.TestCase):
    def test_following_ttc_includes_vehicle_dimensions(self):
        a, b = actor(), actor("lead", (20, 0), (0, 0))
        risk = pair_metrics(a, b)
        self.assertAlmostEqual(risk["cv_obb_ttc_s"], 1.6)
        self.assertAlmostEqual(risk["radial_center_ttc_s"], 2.0)

    def test_crossing_rectangles(self):
        a = actor(position=(-10, 0), velocity=(5, 0))
        b = actor("crossing", (0, -10), (0, 5), math.pi / 2)
        self.assertAlmostEqual(cv_obb_ttc(a, b), 1.4)
        self.assertAlmostEqual(pair_metrics(a, b)["time_to_cpa_within_horizon_s"], 2.0)
        self.assertAlmostEqual(pair_metrics(a, b)["cpa_center_distance_m"], 0.0)

    def test_parallel_and_diverging_are_not_predicted_collisions(self):
        self.assertIsNone(cv_obb_ttc(actor(), actor("other", (20, 0), (10, 0))))
        self.assertIsNone(cv_obb_ttc(actor(), actor("other", (20, 0), (20, 0))))
        self.assertIsNone(pair_metrics(actor(), actor("other", (20, 0), (10, 0)))["radial_center_ttc_s"])

    def test_overlap_horizon_and_unknown_geometry(self):
        self.assertEqual(cv_obb_ttc(actor(), actor("other", (1, 0))), 0.0)
        self.assertIsNone(cv_obb_ttc(actor(), actor("other", (20, 0), (0, 0)), horizon=1.0))
        missing = actor("other", (20, 0), (0, 0))
        missing["width"] = None
        self.assertIsNone(cv_obb_ttc(actor(), missing))
        self.assertFalse(pair_metrics(actor(), missing)["cv_obb_geometry_available"])

    def test_rigid_transform_does_not_change_ttc(self):
        a, b = actor(), actor("lead", (20, 0), (0, 0))
        angle = 1.137
        def transform(x):
            x = dict(x)
            for key in ("position", "velocity"):
                p, q = x[key]
                x[key] = [math.cos(angle) * p - math.sin(angle) * q,
                          math.sin(angle) * p + math.cos(angle) * q]
                if key == "position":
                    x[key][0] += 31.0
                    x[key][1] -= 14.0
            x["heading"] += angle
            return x
        self.assertAlmostEqual(cv_obb_ttc(a, b), cv_obb_ttc(transform(a), transform(b)))


class RecorderTests(unittest.TestCase):
    def test_two_resets_preserve_stream_and_distinct_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "eval", "test")
            for seed in (10000, 10001):
                recorder.on_env_close()  # reset() closes the simulator, not this writer.
                recorder.on_reset({"ego": actor()}, {"traffic_variant": "traffic_10"}, seed)
                recorder.on_decision_start([0, 1], {"target_speed": 10})
                recorder.on_raw_step({"raw_step": 1, "sim_time": 0.1, "ego": actor(),
                                      "ego_state_source": "current", "vehicles": [actor("lead", (20, 0), (0, 0))],
                                      "observed_neighbor_ids": [], "events": {"collision": True}})
                recorder.on_decision_end(-1, True, False, {"collision": True})
                recorder.on_policy_step([0, 1], -10, True, False, {"collision": True})
            recorder.record_optimization(1000, 334, 500, {"train/loss": 0.4, "ignored": [1, 2]})
            recorder.close()
            path = Path(directory) / "diagnostics" / "eval"
            episodes = [json.loads(x) for x in (path / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual([x["seed"] for x in episodes], [10000, 10001])
            self.assertEqual([x["return_base"] for x in episodes], [-1, -1])
            self.assertEqual([x["return_policy"] for x in episodes], [-10, -10])
            self.assertEqual(episodes[0]["critical_unobserved_ticks"], 1)
            with gzip.open(path / "raw_steps.jsonl.gz", "rt") as stream:
                rows = [json.loads(x) for x in stream]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["vehicles"][0]["observed_by_policy"], False)
            self.assertEqual(json.loads((path / "summary.json").read_text())["raw_records"], 2)
            with self.assertRaises(FileExistsError):
                BehaviorDiagnosticsRecorder(directory, "eval", "test")

    def test_removed_pose_has_no_misaligned_risk_and_json_is_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "test")
            recorder.on_reset({}, seed=0)
            recorder.on_decision_start([0, 0])
            recorder.on_raw_step({"raw_step": 1, "ego": actor(), "ego_state_source": "pre_step_removed",
                                  "sim_time": float("nan"), "vehicles": [actor("other", (1, 0))]})
            recorder.close()
            path = Path(directory) / "diagnostics" / "train"
            with gzip.open(path / "raw_steps.jsonl.gz", "rt") as stream:
                row = json.loads(stream.readline())
            self.assertIsNone(row["sim_time"])
            self.assertFalse(row["vehicles"][0]["risk_cv"]["available"])
            self.assertNotIn("cv_obb_ttc_s", row["vehicles"][0]["risk_cv"])
            episode = json.loads((path / "episodes.jsonl").read_text())
            self.assertEqual(episode["finish_reason"], "closed_before_terminal")

    def test_eval_policy_observation_snapshots_keep_first_and_terminal_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "eval_worker_00", "test")
            recorder.on_reset({}, seed=9001)
            recorder.on_decision_start([0.0], {"planned_next_edge": "edge-a"})
            first_observation = {"ego": [1.0, 2.0], "nodes": [[3.0, 4.0]]}
            recorder.on_policy_step(
                [0.1],
                0.0,
                False,
                False,
                {},
                policy_observation=first_observation,
                policy_raw_step=0,
                policy_decision=1,
                pre_step_info={"route_lane_status_known": True},
            )
            recorder.on_decision_start([0.0], {"planned_next_edge": "edge-b"})
            terminal_observation = {"ego": [5.0, 6.0], "nodes": [[7.0, 8.0]]}
            recorder.on_policy_step(
                [0.2],
                1.0,
                True,
                False,
                {"is_success": True},
                policy_observation=terminal_observation,
                policy_raw_step=12,
                policy_decision=2,
                pre_step_info={"route_lane_status_known": True},
            )
            recorder.close()

            path = (
                Path(directory)
                / "diagnostics"
                / "eval_worker_00"
                / "policy_observations.jsonl.gz"
            )
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["trigger"] for row in rows], [
                "eval_first_decision_input",
                "eval_terminal_action_input",
            ])
            self.assertEqual(rows[0]["seed"], 9001)
            self.assertEqual(rows[0]["raw_step_before_action"], 0)
            self.assertEqual(rows[0]["observation"], first_observation)
            self.assertEqual(rows[0]["control_context"]["planned_next_edge"], "edge-a")
            self.assertEqual(rows[1]["raw_step_before_action"], 12)
            self.assertEqual(rows[1]["observation"], terminal_observation)

    def test_train_policy_observation_snapshots_are_boundary_sampled_and_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = BehaviorDiagnosticsRecorder(directory, "train", "test")
            recorder.on_reset({}, seed=19)
            for index in range(1, 12):
                raw_step = index * 10_000
                recorder.on_decision_start([0.0], {"index": index})
                recorder.on_policy_step(
                    [0.0],
                    0.0,
                    False,
                    False,
                    {},
                    policy_observation={"index": index},
                    policy_raw_step=raw_step,
                    policy_decision=index,
                )
            recorder.close()

            path = (
                Path(directory)
                / "diagnostics"
                / "train"
                / "policy_observations.jsonl.gz"
            )
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            summary = json.loads(
                (path.parent / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["raw_step_before_action"], 10_000)
            self.assertEqual(rows[-1]["raw_step_before_action"], 100_000)
            self.assertTrue(
                all(row["trigger"] == "train_first_policy_input_at_or_after_10k_raw_boundary" for row in rows)
            )
            self.assertEqual(summary["policy_observation_snapshots"], 10)
            self.assertEqual(summary["policy_observation_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
