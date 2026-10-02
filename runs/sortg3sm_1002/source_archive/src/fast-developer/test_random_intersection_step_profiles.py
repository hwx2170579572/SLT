"""Focused tests for the p=.2/.3 per-step Bernoulli traffic profiles."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.sumo.high_density_env_v1 import (
    HighDensityPaperSumoSceneEnvV1,
    SUPPORTED_HIGH_DENSITY_SCENARIOS,
)
from envs.sumo.paper_env import PaperSumoSceneEnv, _PAPER_MAX_EPISODE_STEPS
from envs.sumo.random_intersection import (
    ASSET_ROOT,
    RANDOM_INTERSECTION_SCENARIOS,
    STEP_P05_SCENARIO,
    STEP_PROBABILITIES,
    STEP_SCHEDULE_PROTOCOL,
    ensure_random_intersection_assets,
    get_random_intersection_config,
    is_random_intersection_scenario,
    write_seeded_episode_traffic,
)
from envs.sumo.scenario_registry import available_scenarios


NEW_PROFILES = {
    "intersection_random_medium_p03_v1": 0.3,
    "intersection_random_medium_p02_v1": 0.2,
}


def _xml_node_signature(element):
    """Compare XML semantics while ignoring indentation-only text/tails."""
    text = element.text if element.text and element.text.strip() else None
    tail = element.tail if element.tail and element.tail.strip() else None
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_xml_node_signature(child) for child in element),
        tail,
    )


class _FakeSimulation:
    def getTime(self):
        return 30.0

    def getPendingVehicles(self):
        return tuple(f"background_{route_id}.pending" for route_id in self.route_ids)


class _FakeVehicle:
    def __init__(self, route_ids):
        self.route_ids = tuple(route_ids)

    def getIDList(self):
        return ("ego", *(f"background_{route_id}.active" for route_id in self.route_ids))

    def getSpeed(self, vehicle_id):
        return 3.0 if vehicle_id == "ego" else 0.0


class _FakeLane:
    def getLastStepVehicleNumber(self, _lane_id):
        return 1

    def getLastStepHaltingNumber(self, _lane_id):
        return 1


class StepBernoulliProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the production asset validator/generator so the test exercises
        # the same frozen map/ego/driver templates as the environment.
        cls.manifests = ensure_random_intersection_assets()
        cls.medium_traffic = ASSET_ROOT / "intersection_random_medium_v1" / "traffic" / "traffic_00000.rou.xml"

    def test_new_profiles_register_dynamic_configs_and_keep_frozen_assets(self):
        medium_root = ET.parse(self.medium_traffic).getroot()
        for name, probability in NEW_PROFILES.items():
            with self.subTest(scenario=name):
                config = get_random_intersection_config(name)
                self.assertIn(name, RANDOM_INTERSECTION_SCENARIOS)
                self.assertIn(name, available_scenarios())
                self.assertIn(name, SUPPORTED_HIGH_DENSITY_SCENARIOS)
                self.assertTrue(is_random_intersection_scenario(name))
                self.assertEqual(_PAPER_MAX_EPISODE_STEPS[name], 600)
                self.assertTrue(config["dynamic_episode_traffic"])
                self.assertEqual(config["protocol"], STEP_SCHEDULE_PROTOCOL)
                self.assertEqual(config["step_length_seconds"], 0.1)
                self.assertEqual(config["flow_begin_seconds"], 0)
                self.assertEqual(config["flow_end_seconds"], 130)
                expected_rate = int(round(probability * 36000))
                self.assertEqual(config["total_vehicles_per_hour"], 3 * expected_rate)
                self.assertEqual(len(config["flows"]), 3)
                for flow in config["flows"]:
                    self.assertAlmostEqual(flow["probability_per_step"], probability)
                    self.assertEqual(flow["vehicles_per_hour"], expected_rate)
                    self.assertIsNone(flow["sumo_probability"])

                manifest = self.manifests[name]
                self.assertEqual(manifest["configuration"], config)
                for relative, expected_hash in manifest["asset_sha256"].items():
                    actual_hash = hashlib.sha256((ASSET_ROOT / name / relative).read_bytes()).hexdigest()
                    self.assertEqual(actual_hash, expected_hash)
                for relative in ("map.net.xml", "ego.rou.xml"):
                    self.assertEqual(
                        (ASSET_ROOT / name / relative).read_bytes(),
                        (ASSET_ROOT / "intersection_random_medium_p05_v1" / relative).read_bytes(),
                    )

                traffic_path = ASSET_ROOT / name / "traffic" / "traffic_00000.rou.xml"
                root = ET.parse(traffic_path).getroot()
                self.assertEqual(len(root.findall("vType")), 120)
                self.assertEqual(len(root.findall("route")), 3)
                self.assertEqual(len(root.findall("vTypeDistribution")), 3)
                self.assertEqual(root.findall("flow"), [])
                self.assertEqual(root.findall("vehicle"), [])
                for tag in ("vType", "vTypeDistribution", "route"):
                    self.assertEqual(
                        [_xml_node_signature(item) for item in root.findall(tag)],
                        [_xml_node_signature(item) for item in medium_root.findall(tag)],
                        msg=f"changed frozen {tag} definitions",
                    )

    def test_same_seed_route_draws_are_nested_across_p02_p03_and_p05(self):
        ordered = sorted(STEP_PROBABILITIES.items(), key=lambda item: item[1])
        self.assertEqual(
            [(name, probability) for name, probability in ordered],
            [
                ("intersection_random_medium_p02_v1", 0.2),
                ("intersection_random_medium_p03_v1", 0.3),
                (STEP_P05_SCENARIO, 0.5),
            ],
        )

        with tempfile.TemporaryDirectory(prefix="step_profile_test_") as temp_dir:
            root = Path(temp_dir)
            schedules = {}
            metadata = {}
            for name, _probability in ordered:
                template = ASSET_ROOT / name / "traffic" / "traffic_00000.rou.xml"
                output = root / f"{name}.rou.xml"
                metadata[name] = write_seeded_episode_traffic(name, 217, template, output)
                schedules[name] = metadata[name]["departure_times_by_route"]
                self.assertEqual(
                    metadata[name]["sha256"],
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                )

            low, middle, high = (name for name, _probability in ordered)
            self.assertEqual(set(schedules[low]), set(schedules[middle]))
            self.assertEqual(set(schedules[middle]), set(schedules[high]))
            for route_id in schedules[low]:
                low_ticks = set(schedules[low][route_id])
                middle_ticks = set(schedules[middle][route_id])
                high_ticks = set(schedules[high][route_id])
                self.assertLessEqual(low_ticks, middle_ticks)
                self.assertLessEqual(middle_ticks, high_ticks)
                for name in (low, middle, high):
                    config = get_random_intersection_config(name)
                    dt = config["step_length_seconds"]
                    for depart in schedules[name][route_id]:
                        self.assertGreaterEqual(depart, 0)
                        self.assertLess(depart, config["flow_end_seconds"])
                        self.assertAlmostEqual(depart / dt, round(depart / dt), places=7)

    def test_warmup_capture_and_overlay_identity_accept_new_profiles(self):
        for name in NEW_PROFILES:
            with self.subTest(scenario=name):
                config = get_random_intersection_config(name)
                route_ids = tuple(flow["id"] for flow in config["flows"])
                simulation = _FakeSimulation()
                simulation.route_ids = route_ids
                connection = SimpleNamespace(
                    simulation=simulation,
                    vehicle=_FakeVehicle(route_ids),
                    lane=_FakeLane(),
                )
                schedule = {
                    "sha256": "a" * 64,
                    "departure_times_by_route": {
                        route_id: [0.0, 10.0, 40.0] for route_id in route_ids
                    },
                }
                fake_env = SimpleNamespace(
                    _connection=connection,
                    specification=SimpleNamespace(ego_id="ego"),
                    _random_traffic_config=config,
                    _random_episode_schedule=schedule,
                    _random_departures={route_id: 1 for route_id in route_ids},
                    _random_warmup_checkpoints=[],
                    _random_arrivals=0,
                    _random_delay_sum=0.0,
                    _random_delay_count=0,
                    _random_delay_max=0.0,
                )
                PaperSumoSceneEnv._capture_random_warmup_checkpoint(fake_env, 30.0)
                self.assertEqual(len(fake_env._random_warmup_checkpoints), 1)
                checkpoint = fake_env._random_warmup_checkpoints[0]
                self.assertEqual(checkpoint["target_seconds"], 30.0)
                self.assertEqual(checkpoint["requested_background_due"], 6)
                self.assertEqual(checkpoint["actually_inserted_background_cumulative"], 3)
                self.assertEqual(checkpoint["pending_background_insertions"], 3)
                self.assertEqual(checkpoint["request_balance_due_minus_inserted_minus_pending"], 0)
                self.assertEqual(checkpoint["in_network_background_vehicles"], 3)
                self.assertEqual(checkpoint["halting_background_vehicles"], 3)

                overlay = SimpleNamespace(
                    _high_density_overlay_root=Path("unused-overlay-root"),
                    scenario=name,
                    _high_density_vehicle_scale=1.0,
                    _random_episode_schedule=schedule,
                )
                source_path = Path("episode.rou.xml")
                first = HighDensityPaperSumoSceneEnvV1._overlay_path(overlay, source_path)
                same = HighDensityPaperSumoSceneEnvV1._overlay_path(overlay, source_path)
                overlay._random_episode_schedule = {"sha256": "b" * 64}
                changed = HighDensityPaperSumoSceneEnvV1._overlay_path(overlay, source_path)
                self.assertEqual(first, same)
                self.assertNotEqual(first, changed)
                self.assertIn("__saaaaaaaaaaaaaaaa", first.name)
                self.assertEqual(first.parent.name, name)


if __name__ == "__main__":
    unittest.main()
