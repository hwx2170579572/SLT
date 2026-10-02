import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from envs.sumo.route_reachability_v1 import (
    ELIGIBLE, INELIGIBLE, UNKNOWN, RouteLaneReachability,
    RouteReachabilityObservationWrapper,
)


NET = """<net>
<edge id="in"><lane id="in_0" index="0"/><lane id="in_1" index="1"/><lane id="in_2" index="2"/></edge>
<edge id="out"><lane id="out_1" index="1"/></edge>
<edge id="wrong"><lane id="wrong_0" index="0"/></edge>
<edge id="ped"><lane id="ped_0" index="0" allow="pedestrian"/></edge>
<edge id=":J_0" function="internal"><lane id=":J_0_0" index="0"/></edge>
<edge id=":J_1" function="internal"><lane id=":J_1_0" index="0"/></edge>
<connection from="in" to="wrong" fromLane="0" toLane="0"/>
<connection from="in" to="wrong" fromLane="1" toLane="0"/>
<connection from="in" to="out" fromLane="2" toLane="1" via=":J_0_0"/>
<connection from=":J_0" to="out" fromLane="0" toLane="1" via=":J_1_0"/>
<connection from=":J_1" to="out" fromLane="0" toLane="1"/>
<connection from="in" to="ped" fromLane="0" toLane="0"/>
</net>"""


class RouteLaneReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "test.net.xml"
        path.write_text(NET, encoding="utf-8")
        self.network = RouteLaneReachability.from_net(path)

    def test_only_correct_exit_lane_and_internal_corridor_are_eligible(self):
        nodes = ["in_0", "in_1", "in_2", ":J_0_0", ":J_1_0", "out_1", "wrong_0"]
        result = self.network.classify(nodes, ["in", "out"], 0, current_road="in")
        self.assertTrue(result.context_known)
        self.assertEqual(result.labels, (0, 0, 1, 1, 1, 1, 0))

    def test_route_condition_changes_eligibility_without_hardcoded_lane_number(self):
        nodes = ["in_0", "in_1", "in_2", "out_1", "wrong_0"]
        result = self.network.classify(nodes, ["in", "wrong"], 0)
        self.assertEqual(result.labels, (1, 1, 0, 0, 1))

    def test_node_order_and_unknown_nodes_are_preserved(self):
        result = self.network.classify(["unknown", "in_2", "", "in_0"], ["in", "out"], 0)
        self.assertEqual(result.labels, (UNKNOWN, ELIGIBLE, UNKNOWN, INELIGIBLE))

    def test_missing_or_inconsistent_route_context_stays_unknown(self):
        for route, index, road in [
            ([], 0, None), (["in", "out"], -1, None),
            (["in", "missing"], 0, None), (["in", "out"], 0, "wrong"),
        ]:
            with self.subTest(route=route, index=index, road=road):
                result = self.network.classify(["in_0", "in_2"], route, index, road)
                self.assertFalse(result.context_known)
                self.assertEqual(result.labels, (UNKNOWN, UNKNOWN))

    def test_internal_ego_road_keeps_current_route_hop(self):
        result = self.network.classify(["in_2", ":J_1_0"], ["in", "out"], 0, ":J_0")
        self.assertTrue(result.context_known)
        self.assertEqual(result.labels, (1, 1))

    def test_final_route_edge_uses_current_edge(self):
        result = self.network.classify(["in_2", "out_1"], ["in", "out"], 1, "out")
        self.assertEqual(result.labels, (0, 1))
        self.assertEqual(result.next_edge, "")

    def test_known_but_unusable_corridor_is_not_unknown(self):
        result = self.network.classify(["in_0", "in_2", "ped_0"], ["in", "ped"], 0)
        self.assertTrue(result.context_known)
        self.assertEqual(result.labels, (0, 0, 0))


class RouteReachabilityWrapperTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "test.net.xml"
        path.write_text(NET, encoding="utf-8")

        class TestVehicle:
            missing = False
            index = 0
            road = "in"

            def getRoute(self, ego_id):
                if self.missing:
                    raise RuntimeError("ego removed")
                return ("in", "out")

            def getRouteIndex(self, ego_id):
                return self.index

            def getRoadID(self, ego_id):
                return self.road

        class TestEnvironment(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Dict({
                    "state": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                })
                self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                self.specification = SimpleNamespace(network_path=path, ego_id="ego")
                self._connection = SimpleNamespace(vehicle=TestVehicle())
                self.original_observation = {"state": np.asarray([0.0], dtype=np.float32)}

            def reset(self, **kwargs):
                return self.original_observation, {"untouched": True}

            def step(self, action):
                return self.original_observation, -7.0, self._connection.vehicle.missing, False, {"untouched": True}

        self.base = TestEnvironment()
        self.env = RouteReachabilityObservationWrapper(self.base, ("in_0", "in_2", "out_1"), 4)

    def test_adds_ordered_mask_without_mutating_original_observation_or_space(self):
        observation, info = self.env.reset()
        self.assertEqual(observation["route_reachability"].tolist(), [0, 1, 1, -1])
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertNotIn("route_reachability", self.base.observation_space.spaces)
        self.assertNotIn("route_reachability", self.base.original_observation)
        self.assertTrue(info["untouched"])
        self.assertTrue(info["route_reachability_context_known"])

    def test_observation_tracks_actual_route_progress(self):
        self.base._connection.vehicle.index = 1
        self.base._connection.vehicle.road = "out"
        observation, reward, terminated, truncated, info = self.env.step(np.zeros(1))
        self.assertEqual(observation["route_reachability"].tolist(), [0, 0, 1, -1])
        self.assertEqual(reward, -7.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_removed_ego_is_unknown_without_changing_terminal_or_reward(self):
        self.base._connection.vehicle.missing = True
        observation, reward, terminated, truncated, info = self.env.step(np.zeros(1))
        self.assertEqual(observation["route_reachability"].tolist(), [-1] * 4)
        self.assertFalse(info["route_reachability_context_known"])
        self.assertEqual(reward, -7.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["untouched"])

    def test_mismatched_graph_mapping_fails_before_training(self):
        with self.assertRaises(ValueError):
            RouteReachabilityObservationWrapper(self.base, ("not_in_network",), 4)


if __name__ == "__main__":
    unittest.main()
