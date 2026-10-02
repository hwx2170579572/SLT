import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from envs.sumo.conflict_timing_observation import (
    PROTOCOL,
    RouteConflictTimingObservationWrapper,
    _load_static_network,
)


NET = """<net>
<edge id="in"><lane id="in_0" index="0" shape="-10,-15 -10,-10"/><lane id="in_1" index="1" shape="0,-15 0,-10"/></edge>
<edge id="out"><lane id="out_0" index="0" shape="0,0 0,15"/></edge>
<edge id="wrong"><lane id="wrong_0" index="0" shape="-10,-10 -15,-10"/></edge>
<edge id=":J" function="internal"><lane id=":J_0_0" index="0" shape="0,-10 0,0"/></edge>
<edge id="cross"><lane id="cross_0" index="0" shape="-10,0 10,0"/></edge>
<connection from="in" to="wrong" fromLane="0" toLane="0"/>
<connection from="in" to="out" fromLane="1" toLane="0" via=":J_0_0"/>
<connection from=":J" to="out" fromLane="0" toLane="0"/>
</net>"""


class _Vehicle:
    def __init__(self):
        self.get_route_calls = []
        self.lanes = {"ego": ("in", "in_1"), "bg": ("cross", "cross_0")}

    def getRoadID(self, actor_id):
        return self.lanes[actor_id][0]

    def getLaneID(self, actor_id):
        return self.lanes[actor_id][1]

    def getRoute(self, actor_id):
        self.get_route_calls.append(actor_id)
        if actor_id != "ego":
            raise AssertionError("social future route must never be read")
        return ("in", "out")

    def getRouteIndex(self, actor_id):
        if actor_id != "ego":
            raise AssertionError("social route index must never be read")
        return 0


class _BaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, net_path):
        self.observation_space = gym.spaces.Dict({
            "trajectory": gym.spaces.Box(-100, 100, shape=(3, 4, 5), dtype=np.float32),
            "map": gym.spaces.Box(-100, 100, shape=(6, 10, 5), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.specification = SimpleNamespace(
            network_path=net_path,
            ego_id="ego",
            source_observation_contract="smarts",
            coordinate_offset=(0.0, 0.0),
            map_paths_per_actor=2,
        )
        self._connection = SimpleNamespace(vehicle=_Vehicle())
        self._last_observation_actor_keys = ("vehicle:ego", "vehicle:bg")
        self._raw_steps = 0
        self._history_timestep = 4
        self.steps = 0
        self.observation = self._make_observation()
        self.last_action = None

    def _make_observation(self):
        trajectory = np.zeros((3, 4, 5), dtype=np.float32)
        trajectory[0, -1] = [0.0, -12.0 + self.steps * 0.3, np.pi / 2, 0.0, 5.0]
        trajectory[1, -1] = [-5.0 + self.steps * 0.3, 0.0, 0.0, 5.0, 0.0]
        map_state = np.zeros((6, 10, 5), dtype=np.float32)
        ego_y = np.linspace(-15, -6, 10, dtype=np.float32)
        map_state[0, :, 0] = 0.0
        map_state[0, :, 1] = ego_y
        map_state[0, :, 3] = 1.0
        map_state[2, :, 0] = np.linspace(-10, 10, 10, dtype=np.float32)
        map_state[2, :, 1] = 0.0
        map_state[2, :, 4] = 1.0
        return {"trajectory": trajectory, "map": map_state}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._raw_steps = 0
        self._history_timestep = 4
        self._last_observation_actor_keys = ("vehicle:ego", "vehicle:bg")
        self.observation = self._make_observation()
        return self.observation, {"base_reset": True}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self.steps += 1
        self._raw_steps += 3
        self._history_timestep += 3
        self._last_observation_actor_keys = ("vehicle:ego", "vehicle:bg")
        self.observation = self._make_observation()
        return self.observation, 3.25, False, False, {"base_step": True}


class RouteConflictTimingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.net_path = Path(self.temp.name) / "geometry.net.xml"
        self.net_path.write_text(NET, encoding="utf-8")

    def test_ego_route_includes_internal_chain_and_excludes_wrong_exit(self):
        base = _BaseEnv(self.net_path)
        wrapper = RouteConflictTimingObservationWrapper(base, phase="smoke")
        observation, _ = wrapper.reset()
        self.assertEqual(observation["conflict_timing"].shape, (3, 20))
        frame = wrapper._last_frame
        ego = frame["actors"][0]
        self.assertEqual(ego["geometry_status"], "known_route_geometry")
        self.assertEqual(len(ego["paths"]), 1)
        lane_ids = ego["paths"][0]["lane_ids"]
        self.assertEqual(lane_ids, ["in_1", ":J_0_0", "out_0"])
        self.assertNotIn("wrong_0", lane_ids)
        self.assertEqual(ego["paths"][0]["termination_reason"], "task_route_end")
        self.assertFalse(ego["path_metadata"]["known_prefix_not_full_route_reachability"])
        self.assertTrue(frame["detail"]["actors"][0]["relation_valid"])
        self.assertEqual(base._connection.vehicle.get_route_calls, ["ego"])
        wrapper.close()

    def test_wrapper_preserves_legacy_arrays_transition_and_logs_compact_rows(self):
        base = _BaseEnv(self.net_path)
        with tempfile.TemporaryDirectory() as directory:
            wrapper = RouteConflictTimingObservationWrapper(
                base, diagnostics_directory=directory, phase="eval", max_prediction_rows=5
            )
            obs0, info0 = wrapper.reset(seed=3)
            np.testing.assert_array_equal(obs0["trajectory"], base.observation["trajectory"])
            np.testing.assert_array_equal(obs0["map"], base.observation["map"])
            self.assertNotIn("conflict_timing", base.observation)
            self.assertTrue(wrapper.observation_space.contains(obs0))
            action = np.asarray([0.25, -0.5], dtype=np.float32)
            obs1, reward, terminated, truncated, info = wrapper.step(action)
            np.testing.assert_array_equal(base.last_action, action)
            np.testing.assert_array_equal(obs1["trajectory"], base.observation["trajectory"])
            np.testing.assert_array_equal(obs1["map"], base.observation["map"])
            self.assertEqual(reward, 3.25)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["base_step"])
            self.assertEqual(info["conflict_timing_protocol"], PROTOCOL)
            self.assertEqual(base.steps, 1)
            wrapper.close()
            root = Path(directory)
            with gzip.open(root / "task_conflict_predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
                row = json.loads(next(stream))
            self.assertEqual(row["schema"], "conflict_timing_prediction_v1")
            self.assertFalse(row["social_future_routes_read"] if "social_future_routes_read" in row else False)
            logged_paths = row["actor_geometry_and_map_coverage"][0]["candidate_paths"]
            self.assertEqual(logged_paths[0]["lane_ids"], ["in_1", ":J_0_0", "out_0"])
            self.assertNotIn("points", logged_paths[0])
            manifest = json.loads((root / "task_conflict_timing_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["social_future_routes_read"])
            self.assertTrue(manifest["original_trajectory_and_map_preserved"])

    def test_unknown_actor_order_and_removed_actor_are_not_encoded_as_safe(self):
        base = _BaseEnv(self.net_path)
        original_reset = base.reset

        def reset_without_actor_order(**kwargs):
            observation, info = original_reset(**kwargs)
            base._last_observation_actor_keys = None
            return observation, info

        base.reset = reset_without_actor_order
        wrapper = RouteConflictTimingObservationWrapper(base, phase="smoke")
        obs, _ = wrapper.reset()
        self.assertTrue(np.all(obs["conflict_timing"] == 0))
        self.assertEqual(wrapper._last_frame["actors"][0]["geometry_status"], "observation_actor_order_unavailable")
        self.assertEqual(wrapper._last_frame["detail"]["valid_relation_count"], 0)
        wrapper.close()

    def test_first_path_deviation_evidence_uses_existing_snapshot_and_preserves_intervals(self):
        base = _BaseEnv(self.net_path)
        wrapper = RouteConflictTimingObservationWrapper(base, phase="smoke")
        rows = []
        wrapper._write_calibration = rows.append
        points = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        selected = {
            "ego_path_id": "ego-path-0",
            "foe_path_id": "foe-path-0",
            "ego_lane_ids": ["in_1"],
            "foe_lane_ids": ["cross_0"],
            "ego_path_points": points,
            "foe_path_points": points,
            "ego_current_s": 2.0,
            "foe_current_s": 5.0,
            "ego_entry_arc_m": 20.0,
            "ego_exit_arc_m": 25.0,
            "foe_entry_arc_m": 20.0,
            "foe_exit_arc_m": 25.0,
        }
        ego_context = wrapper._selected_path_context(
            {
                "paths": [{"path_id": "ego-path-0", "termination_reason": "max_distance_reached"}],
                "path_metadata": {
                    "candidate_count": 2,
                    "truncated": True,
                    "candidate_enumeration_complete": False,
                    "known_prefix_not_full_route_reachability": True,
                    "candidate_termination_reasons": ["max_distance_reached", "depth_limit_truncated"],
                    "candidate_termination_counts": {"max_distance_reached": 1, "depth_limit_truncated": 1},
                },
            },
            selected,
            "ego",
        )
        foe_context = wrapper._selected_path_context(
            {
                "paths": [{"path_id": "foe-path-0", "termination_reason": "static_dead_end_or_cycle"}],
                "path_metadata": {"candidate_count": 1, "truncated": False},
            },
            selected,
            "foe",
        )
        preserved_foe_entry = {
            "left_censored_at_prediction": True,
            "lower_raw_step": 3,
            "upper_raw_step": 6,
        }
        preserved_foe_clearance = {
            "left_censored": False,
            "lower_raw_step": 9,
            "upper_raw_step": 12,
        }
        wrapper._pending = [{
            "prediction": {"ego_key": "vehicle:ego", "foe_key": "vehicle:bg", "raw_step_pre_action": 3},
            "selected": selected,
            "selected_path_context": {"ego": ego_context, "foe": foe_context},
            "path_deviation_evidence": {"ego": None, "foe": None},
            "last": {
                "ego": {"s": 2.0, "raw_step": 3},
                "foe": {"s": 5.0, "raw_step": 3},
            },
            "entry": {"ego": None, "foe": preserved_foe_entry.copy()},
            "clearance": {"ego": None, "foe": preserved_foe_clearance.copy()},
            "status": {"ego": "tracking", "foe": "tracking"},
            "deadline_raw_step": 100,
            "dropout_samples": {"ego": 0, "foe": 0},
        }]

        def frame(raw_step, ego_position, ego_lane):
            return {
                "raw_step": raw_step,
                "decision_index": raw_step - 3,
                "actors": [
                    {"key": "vehicle:ego", "position_xy": ego_position, "lane_id": ego_lane, "road_id": "in"},
                    {"key": "vehicle:bg", "position_xy": [5.0, 0.0], "lane_id": "cross_0", "road_id": "cross"},
                ],
            }

        # The existing rule is strict: exactly 5 m with a matching lane is still tracking.
        wrapper._advance_calibration(frame(4, [2.0, 5.0], "in_1"), terminal=False)
        self.assertEqual(rows, [])
        self.assertIsNone(wrapper._pending[0]["path_deviation_evidence"]["ego"])

        wrapper._advance_calibration(frame(5, [2.0, 5.1], "other_0"), terminal=False)
        self.assertEqual(len(rows), 1)
        evidence = rows[0]["path_deviation_evidence"]["ego"]
        self.assertEqual(evidence["first_detected_raw_step"], 5)
        self.assertEqual(evidence["first_detected_decision_index"], 2)
        self.assertEqual(evidence["observed_lane_id"], "other_0")
        self.assertEqual(evidence["observed_position_xy"], [2.0, 5.1])
        self.assertAlmostEqual(evidence["projected_lateral_distance_m"], 5.1)
        self.assertIs(evidence["lane_in_selected_path"], False)
        self.assertEqual(
            evidence["trigger_reasons"],
            ["lateral_distance_over_5m", "lane_not_in_selected_path"],
        )
        self.assertEqual(evidence["selected_path_lane_ids"], ["in_1"])
        self.assertEqual(evidence["selected_path_termination_reason"], "max_distance_reached")
        self.assertEqual(evidence["candidate_path_count"], 2)
        self.assertTrue(evidence["candidate_paths_truncated"])
        # Existing single-sided entry observations survive the other actor's deviation.
        self.assertEqual(rows[0]["entry_intervals"]["foe"], preserved_foe_entry)
        self.assertEqual(rows[0]["entry_intervals"]["ego"], None)
        self.assertEqual(rows[0]["clearance_intervals"]["foe"], preserved_foe_clearance)
        self.assertEqual(rows[0]["clearance_intervals"]["ego"], None)
        self.assertIsNone(rows[0]["path_deviation_evidence"]["foe"])
        wrapper.close()

    def test_deviation_context_marks_unmatched_path_cutoff_reason_unknown(self):
        selected = {"ego_path_id": "not-in-enumerated-paths", "ego_lane_ids": ["in_1"]}
        context = RouteConflictTimingObservationWrapper._selected_path_context(
            {"paths": [], "path_metadata": {"candidate_count": 0, "truncated": True}},
            selected,
            "ego",
        )
        self.assertEqual(context["selected_path_termination_reason"], "unknown")
        self.assertEqual(context["candidate_path_count"], 0)
        self.assertTrue(context["candidate_paths_truncated"])

    def test_static_network_geometry_cache_reuses_routes_and_shapes(self):
        first = _load_static_network(str(self.net_path.resolve()), 0.0, 0.0)
        second = _load_static_network(str(self.net_path.resolve()), 0.0, 0.0)
        self.assertIs(first, second)
        self.assertIn(":J_0_0", first.reachability.internal_lanes)
        self.assertIn("in_1", first.lane_shapes)
        self.assertEqual(first.network_sha256, second.network_sha256)

    def test_real_sorted_network_preserves_internal_task_route_geometry(self):
        project = Path(__file__).resolve().parents[1]
        net_path = project / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted" / "map.net.xml"
        if not net_path.is_file():
            self.skipTest("local sorted SUMO network is not included")
        base = _BaseEnv(net_path)
        vehicle = base._connection.vehicle
        vehicle.routes = {"ego": ("-E1", "-E0")}
        vehicle.lanes["ego"] = ("-E1", "-E1_2")
        original_get_route = vehicle.getRoute
        original_route_index = vehicle.getRouteIndex
        vehicle.getRoute = lambda actor_id: vehicle.routes[actor_id] if actor_id == "ego" else original_get_route(actor_id)
        vehicle.getRouteIndex = lambda actor_id: 0 if actor_id == "ego" else original_route_index(actor_id)
        base.specification.network_path = net_path
        wrapper = RouteConflictTimingObservationWrapper(base, phase="smoke")
        shape = wrapper.static.lane_shapes["-E1_2"]
        position = (shape[0] + shape[-1]) / 2
        paths, meta = wrapper._ego_paths("-E1", "-E1_2", position)
        self.assertEqual(meta["status"], "known_route_geometry")
        self.assertGreaterEqual(len(paths), 1)
        self.assertTrue(any("-E0_1" in path["lane_ids"] for path in paths))
        self.assertTrue(all(any(lane.startswith(":") for lane in path["lane_ids"]) for path in paths))
        self.assertTrue(all("-E0_0" not in path["lane_ids"] for path in paths))
        second_position = shape[0] + 0.75 * (shape[-1] - shape[0])
        second_paths, second_meta = wrapper._ego_paths("-E1", "-E1_2", second_position)
        self.assertEqual([p["lane_ids"] for p in paths], [p["lane_ids"] for p in second_paths])
        self.assertTrue(np.array_equal(paths[0]["points"], second_paths[0]["points"]))
        self.assertNotEqual(paths[0]["s_current"], second_paths[0]["s_current"])
        wrapper.close()


if __name__ == "__main__":
    unittest.main()
