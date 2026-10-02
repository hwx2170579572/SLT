from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_DEVELOPER = PROJECT_ROOT / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from diagnose_sorted_goalonly_causal_20261002 import route_veto_decision
from envs.sumo.route_action_consistency import (
    RouteActionConsistencyWrapper,
    decide_route_action,
    resolve_target_lane,
)
from envs.sumo.route_reachability_v1 import UNKNOWN


NET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="-E1">
    <lane id="-E1_0" index="0"/>
    <lane id="-E1_1" index="1"/>
    <lane id="-E1_2" index="2"/>
  </edge>
  <edge id="-E0">
    <lane id="-E0_0" index="0"/>
    <lane id="-E0_1" index="1"/>
  </edge>
  <edge id=":J1_0" function="internal">
    <lane id=":J1_0_0" index="0"/>
  </edge>
  <connection from="-E1" to="-E0" fromLane="2" toLane="1"/>
</net>
"""


class FakeVehicleDomain:
    def __init__(self):
        self.route_queries: list[str] = []
        self.present = True

    def getIDList(self):
        return ["ego", "background-hidden-route"] if self.present else ["background-hidden-route"]

    def getRoadID(self, vehicle_id):
        self.route_queries.append(f"road:{vehicle_id}")
        if vehicle_id != "ego":
            raise AssertionError("the route mapper may query only the controlled ego")
        return "-E1"

    def getLaneID(self, vehicle_id):
        self.route_queries.append(f"lane:{vehicle_id}")
        if vehicle_id != "ego":
            raise AssertionError("the route mapper may query only the controlled ego")
        return "-E1_2"

    def getRoute(self, vehicle_id):
        self.route_queries.append(f"route:{vehicle_id}")
        if vehicle_id != "ego":
            raise AssertionError("the route mapper may query only the controlled ego")
        return ["-E1", "-E0"]

    def getRouteIndex(self, vehicle_id):
        self.route_queries.append(f"route_index:{vehicle_id}")
        if vehicle_id != "ego":
            raise AssertionError("the route mapper may query only the controlled ego")
        return 0


class FakeRawSumoEnv(gym.Env):
    metadata = {}

    def __init__(self, network_path: Path, *, contract="carla", recorder_enabled=False):
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.specification = SimpleNamespace(ego_id="ego", network_path=network_path,
                                             source_observation_contract=contract)
        self._connection = SimpleNamespace(
            vehicle=FakeVehicleDomain(),
            simulation=SimpleNamespace(getTime=lambda: 12.5),
        )
        self._raw_steps = 0
        self._behavior_diagnostics = object() if recorder_enabled else None
        self.received_actions: list[np.ndarray] = []
        self.adapted_actions: list[np.ndarray] = []
        self._episode_done = False

    @staticmethod
    def adapt_action(action):
        lateral = float(np.clip(action[1], -1.0, 1.0))
        if lateral < -1.0 / 3.0:
            lane = -1
        elif lateral > 1.0 / 3.0:
            lane = 1
        else:
            lane = 0
        return float((float(action[0]) + 1.0) * 5.0), lane

    @staticmethod
    def _driving_lanes(_road):
        return (0, 1, 2)

    def _sumo_lane_offset(self, lane_command):
        return lane_command if self.specification.source_observation_contract == "smarts" else -lane_command

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_done = False
        self._raw_steps = 0
        return np.asarray([0.0], dtype=np.float32), {"seed": seed}

    def step(self, action):
        forwarded = np.asarray(action).copy()
        self.received_actions.append(forwarded)
        self.adapted_actions.append(forwarded.copy())
        self._raw_steps += 3
        info = {
            "lane_change_applied": False,
            "lane_control_request_status": "hold" if float(forwarded[1]) == 0.0 else "issued",
            "lane_control_request_reason": "hold_command" if float(forwarded[1]) == 0.0 else "requested",
            "current_lane_id": "-E1_2",
            "actual_lane_transition_since_previous_decision": None,
        }
        return np.asarray([0.25], dtype=np.float32), 2.75, False, False, info


class RouteActionDecisionTests(unittest.TestCase):
    def test_matches_existing_veto_rule_and_does_not_mutate_input(self):
        proposed = np.asarray([0.31, 0.88], dtype=np.float32)
        original = proposed.copy()
        common = dict(
            context_known=True,
            context_fresh=True,
            current_road="-E1",
            has_next_edge=True,
            current_lane_label=1,
            target_lane_label=0,
            lane_command=1,
            target_lane_id="-E1_1",
            target_lane_reason="known_lane_cannot_reach_next_edge",
            target_lane_in_range=True,
        )
        actual, audit = decide_route_action(proposed, **common)
        legacy, legacy_audit = route_veto_decision(proposed, **common)

        np.testing.assert_array_equal(proposed, original)
        np.testing.assert_array_equal(actual, legacy)
        self.assertTrue(audit["veto_applied"])
        self.assertEqual(audit["reason"], legacy_audit["reason"])
        self.assertEqual(audit["lane_command_forwarded"], legacy_audit["lane_command_after_veto"])
        self.assertEqual(float(actual[0]), float(original[0]))
        self.assertEqual(float(actual[1]), 0.0)

    def test_uncertain_or_ineligible_cases_pass_through_with_a_reason(self):
        action = np.asarray([-0.2, -0.9], dtype=np.float32)
        defaults = dict(
            context_known=True,
            context_fresh=True,
            current_road="-E1",
            has_next_edge=True,
            current_lane_label=1,
            target_lane_label=0,
            lane_command=-1,
            target_lane_id="-E1_1",
            target_lane_in_range=True,
        )
        cases = (
            ("context_unknown", {"context_known": False}),
            ("stale_context", {"context_fresh": False}),
            ("internal_road", {"current_road": ":J1_0"}),
            ("no_next_edge", {"has_next_edge": False}),
            ("current_lane_unknown", {"current_lane_label": UNKNOWN}),
            ("current_lane_ineligible", {"current_lane_label": 0}),
            ("lane_command_unknown", {"lane_command": None}),
            ("hold_command", {"lane_command": 0}),
            ("target_lane_out_of_range", {"target_lane_in_range": False}),
            ("target_lane_unknown", {"target_lane_label": UNKNOWN}),
            ("target_lane_eligible", {"target_lane_label": 1}),
        )
        original = action.copy()
        for expected_reason, changes in cases:
            with self.subTest(reason=expected_reason):
                args = dict(defaults)
                args.update(changes)
                forwarded, audit = decide_route_action(action, **args)
                self.assertFalse(audit["veto_applied"])
                self.assertEqual(audit["reason"], expected_reason)
                np.testing.assert_array_equal(forwarded, original)
                np.testing.assert_array_equal(action, original)

    def test_target_lane_direction_uses_environment_contract(self):
        raw_base = SimpleNamespace(
            _driving_lanes=lambda _road: (0, 1, 2),
            _sumo_lane_offset=lambda command: -command,
        )
        raw_smarts = SimpleNamespace(
            _driving_lanes=lambda _road: (0, 1, 2),
            _sumo_lane_offset=lambda command: command,
        )
        base_target = resolve_target_lane(
            raw_base, current_road="edge", current_lane_id="edge_1", lane_command=1
        )
        smarts_target = resolve_target_lane(
            raw_smarts, current_road="edge", current_lane_id="edge_1", lane_command=1
        )
        self.assertEqual(base_target["target_lane_id"], "edge_0")
        self.assertEqual(smarts_target["target_lane_id"], "edge_2")
        self.assertEqual(base_target["sumo_lane_offset"], -1)
        self.assertEqual(smarts_target["sumo_lane_offset"], 1)


class RouteActionWrapperTests(unittest.TestCase):
    def test_live_route_mapping_is_identical_with_or_without_behavior_recorder(self):
        with tempfile.TemporaryDirectory(prefix="route_action_consistency_") as tmp:
            root = Path(tmp)
            network_path = root / "net.xml"
            network_path.write_text(NET_XML, encoding="utf-8")
            outputs = []

            for recorder_enabled in (False, True):
                case_root = root / f"case_{int(recorder_enabled)}"
                raw = FakeRawSumoEnv(
                    network_path,
                    recorder_enabled=recorder_enabled,
                )
                wrapped = RouteActionConsistencyWrapper(
                    raw,
                    run_dir=case_root,
                    phase="train",
                    method="test_method",
                )
                wrapped.reset(seed=0)
                proposed = np.asarray([0.2, 0.8], dtype=np.float32)
                original = proposed.copy()
                observation, reward, terminated, truncated, info = wrapped.step(proposed)
                outputs.append((raw, observation, reward, terminated, truncated, info))
                # Close the JSONL handle before any assertions can fail and
                # TemporaryDirectory attempts Windows cleanup.
                wrapped.close()

                np.testing.assert_array_equal(proposed, original)
                np.testing.assert_array_equal(
                    raw.received_actions[0], np.asarray([0.2, 0.0], dtype=np.float32)
                )
                self.assertEqual(float(reward), 2.75)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                decision = info["route_action_consistency"]
                self.assertTrue(decision["veto_applied"])
                self.assertEqual(decision["reason"], "known_unreachable_target_lane_vetoed")
                np.testing.assert_allclose(decision["policy_proposed_action"], [0.2, 0.8])
                np.testing.assert_allclose(decision["env_action_forwarded"], [0.2, 0.0])
                self.assertEqual(decision["context_source"], "live_ego_traci_route_and_lane_plus_static_net")
                self.assertTrue(decision["context_fresh"])
                self.assertTrue(decision["context_known"])
                self.assertEqual(decision["context_sample_raw_step"], 0)
                self.assertEqual(decision["context_age_raw_steps"], 0)
                self.assertEqual(decision["route_context"]["planned_next_edge"], "-E0")
                self.assertEqual(
                    decision["route_context"]["current_lane_reachability_label"], 1
                )
                self.assertEqual(decision["target_context"]["target_lane_reachability_label"], 0)
                self.assertFalse(decision["environment_reported"]["lane_change_applied_reported_by_env"])
                self.assertNotIn("lane_change_executed", decision)
                self.assertTrue(all(query.endswith(":ego") for query in raw._connection.vehicle.route_queries))

                sidecar_path = case_root / "diagnostics" / "train" / "route_action_consistency.jsonl"
                rows = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(rows[0]["record_type"], "manifest")
                self.assertEqual(rows[1]["record_type"], "decision")
                self.assertEqual(rows[1]["decision"]["reason"], decision["reason"])
                self.assertEqual(rows[1]["episode_seed"], 0)

            left, right = outputs
            np.testing.assert_array_equal(left[1], right[1])
            self.assertEqual(left[2:5], right[2:5])
            self.assertEqual(
                left[5]["route_action_consistency"]["env_action_forwarded"],
                right[5]["route_action_consistency"]["env_action_forwarded"],
            )

    def test_sidecar_disabled_does_not_disable_the_action_rule(self):
        with tempfile.TemporaryDirectory(prefix="route_action_no_sidecar_") as tmp:
            network_path = Path(tmp) / "net.xml"
            network_path.write_text(NET_XML, encoding="utf-8")
            raw = FakeRawSumoEnv(network_path)
            wrapped = RouteActionConsistencyWrapper(raw, run_dir=None, phase="eval")
            try:
                wrapped.reset(seed=7)
                proposed = np.asarray([0.4, 0.9], dtype=np.float32)
                original = proposed.copy()
                _, reward, terminated, truncated, info = wrapped.step(proposed)
                np.testing.assert_array_equal(proposed, original)
                np.testing.assert_array_equal(
                    raw.received_actions[0], np.asarray([0.4, 0.0], dtype=np.float32)
                )
                self.assertEqual(reward, 2.75)
                self.assertFalse(terminated or truncated)
                self.assertTrue(info["route_action_consistency"]["veto_applied"])
                self.assertIsNone(wrapped.sidecar_path)
            finally:
                wrapped.close()

    def test_missing_ego_context_passes_action_and_is_logged_unknown(self):
        with tempfile.TemporaryDirectory(prefix="route_action_unknown_") as tmp:
            root = Path(tmp)
            network_path = root / "net.xml"
            network_path.write_text(NET_XML, encoding="utf-8")
            raw = FakeRawSumoEnv(network_path)
            raw._connection.vehicle.present = False
            wrapped = RouteActionConsistencyWrapper(raw, run_dir=root / "run", phase="eval")
            wrapped.reset(seed=5)
            proposed = np.asarray([0.1, 0.8], dtype=np.float32)
            original = proposed.copy()
            _, _, _, _, info = wrapped.step(proposed)
            np.testing.assert_array_equal(proposed, original)
            np.testing.assert_array_equal(raw.received_actions[0], original)
            audit = info["route_action_consistency"]
            self.assertFalse(audit["context_known"])
            self.assertFalse(audit["veto_applied"])
            self.assertEqual(audit["reason"], "context_unknown")
            self.assertEqual(audit["route_context"]["context_reason"], "ego_not_active")
            wrapped.close()

    def test_known_current_context_with_out_of_range_target_is_not_mislabeled_unknown(self):
        with tempfile.TemporaryDirectory(prefix="route_action_known_out_of_range_") as tmp:
            root = Path(tmp)
            network_path = root / "net.xml"
            network_path.write_text(NET_XML, encoding="utf-8")
            raw = FakeRawSumoEnv(network_path, contract="smarts")
            wrapped = RouteActionConsistencyWrapper(raw, run_dir=root / "run", phase="eval")
            try:
                wrapped.reset(seed=11)
                proposed = np.asarray([0.15, 0.8], dtype=np.float32)
                original = proposed.copy()
                _, reward, terminated, truncated, info = wrapped.step(proposed)
                audit = info["route_action_consistency"]
                np.testing.assert_array_equal(proposed, original)
                np.testing.assert_array_equal(raw.received_actions[0], original)
                self.assertEqual(reward, 2.75)
                self.assertFalse(terminated or truncated)
                self.assertTrue(audit["context_known"])
                self.assertTrue(audit["context_fresh"])
                self.assertTrue(audit["route_context"]["context_known"])
                self.assertFalse(audit["target_context"]["target_route_context_known"])
                self.assertFalse(audit["veto_applied"])
                self.assertEqual(audit["reason"], "target_lane_out_of_range")
            finally:
                wrapped.close()


if __name__ == "__main__":
    unittest.main()
