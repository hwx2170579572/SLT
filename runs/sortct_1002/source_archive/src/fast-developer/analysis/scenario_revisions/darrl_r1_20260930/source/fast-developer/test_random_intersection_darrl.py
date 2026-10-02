"""Tests for the lane-1 DARRL Bernoulli scene family."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.sumo.high_density_env_v1 import SUPPORTED_HIGH_DENSITY_SCENARIOS
from envs.sumo.paper_env import PaperSumoSceneEnv, _PAPER_MAX_EPISODE_STEPS
from envs.sumo.random_intersection import (
    ASSET_ROOT,
    DARRL_STEP_PROBABILITIES,
    RANDOM_INTERSECTION_SCENARIOS,
    STEP_SCHEDULE_PROTOCOL,
    _ROUTES,
    _source_distributions,
    _traffic_xml,
    ensure_random_intersection_assets,
    get_random_intersection_config,
    write_seeded_episode_traffic,
)
from envs.sumo.scenario_registry import available_scenarios


LEGACY_CONFIG_SHA256 = {
    "intersection_random_low_v1": "14b9aec5fb587913ebe297ba275e7923e60ae211c213a0d45fa486c946e8f7d8",
    "intersection_random_medium_v1": "51cb687a5d763a1a31d1d66ec9c1ffcbe247f761220e028594ae44802b31907b",
    "intersection_random_high_v1": "31b48eae5287ab4d2cba349294424eb6355cbdf0229e0bdfdfb6a93e35b2c756",
    "intersection_random_medium_p05_v1": "a1b82a5c9482bb79f781b90eccbd9abdfbc8c0b918d7964f4fd8725ea57659a9",
    "intersection_random_medium_p03_v1": "5ae754983092de6bb7e79fff0879a14ff9c30186608facc2bc224bca1dac3b88",
    "intersection_random_medium_p02_v1": "72681310bafd764b07109b6798f4da8881fbb13a7db9ee06dcd59dde818a470e",
}


def _signature(element: ET.Element):
    text = element.text if element.text and element.text.strip() else None
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_signature(child) for child in element),
    )


def _type_attributes(root: ET.Element) -> dict[str, dict[str, str]]:
    return {item.attrib["id"]: dict(item.attrib) for item in root.findall("vType")}


class DarrlRandomIntersectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = {}
        for name in DARRL_STEP_PROBABILITIES:
            cls.manifests.update(ensure_random_intersection_assets(name))

    def test_new_profiles_register_and_old_six_configs_are_unchanged(self):
        self.assertEqual(
            set(LEGACY_CONFIG_SHA256),
            set(RANDOM_INTERSECTION_SCENARIOS[:6]),
        )
        for name, expected in LEGACY_CONFIG_SHA256.items():
            payload = json.dumps(
                get_random_intersection_config(name),
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, name)

        for name, probability in DARRL_STEP_PROBABILITIES.items():
            with self.subTest(scenario=name):
                config = get_random_intersection_config(name)
                self.assertIn(name, RANDOM_INTERSECTION_SCENARIOS)
                self.assertIn(name, available_scenarios())
                self.assertIn(name, SUPPORTED_HIGH_DENSITY_SCENARIOS)
                self.assertEqual(_PAPER_MAX_EPISODE_STEPS[name], 600)
                self.assertEqual(config["scenario_family"], "darrl_lane1_v1")
                self.assertEqual(config["protocol"], STEP_SCHEDULE_PROTOCOL)
                self.assertTrue(config["dynamic_episode_traffic"])
                self.assertEqual(config["step_length_seconds"], 0.1)
                self.assertEqual(config["flow_end_seconds"], 130.0)
                self.assertEqual(config["ego_depart_seconds"], 50.0)
                self.assertFalse(config["endless_traffic"])
                self.assertEqual(config["collision_action"], "remove")
                self.assertEqual(
                    config["collision_detection_mode"],
                    "sumo_events_or_ego_geometry",
                )
                self.assertEqual(config["background_vtype_overrides"], {"jmIgnoreFoeProb": "0"})
                self.assertEqual(
                    config["ego_vtype_overrides"],
                    {"minGap": "1", "jmIgnoreFoeProb": "0"},
                )
                rate = int(round(probability * 36000))
                self.assertEqual(config["total_vehicles_per_hour"], 3 * rate)
                self.assertEqual(len(config["flows"]), 3)
                for flow in config["flows"]:
                    self.assertEqual(flow["probability_per_step"], probability)
                    self.assertEqual(flow["vehicles_per_hour"], rate)
                    self.assertEqual(flow["depart_lane"], "1")
                    self.assertEqual(flow["arrival_lane"], "1")

    def test_darrl_assets_only_apply_declared_ego_and_driver_overrides(self):
        for name in DARRL_STEP_PROBABILITIES:
            with self.subTest(scenario=name):
                config = get_random_intersection_config(name)
                template = config["template_scenario"]
                asset_dir = ASSET_ROOT / name
                template_dir = ASSET_ROOT / template
                manifest = self.manifests[name]
                self.assertEqual(manifest["configuration"], config)
                for relative, expected in manifest["asset_sha256"].items():
                    self.assertEqual(
                        hashlib.sha256((asset_dir / relative).read_bytes()).hexdigest(),
                        expected,
                    )
                self.assertEqual(
                    (asset_dir / "map.net.xml").read_bytes(),
                    (template_dir / "map.net.xml").read_bytes(),
                )

                base_ego = ET.parse(template_dir / "ego.rou.xml").getroot()
                expected_ego = copy.deepcopy(base_ego)
                ego_type = expected_ego.find("vType")
                self.assertIsNotNone(ego_type)
                for key, value in config["ego_vtype_overrides"].items():
                    ego_type.set(key, value)
                actual_ego = ET.parse(asset_dir / "ego.rou.xml").getroot()
                self.assertEqual(_signature(actual_ego), _signature(expected_ego))

                base_traffic = ET.parse(
                    template_dir / "traffic" / "traffic_00000.rou.xml"
                ).getroot()
                darrl_traffic = ET.parse(
                    asset_dir / "traffic" / "traffic_00000.rou.xml"
                ).getroot()
                base_types = _type_attributes(base_traffic)
                darrl_types = _type_attributes(darrl_traffic)
                self.assertEqual(set(darrl_types), set(base_types))
                self.assertEqual(len(darrl_types), 120)
                for type_id, attributes in darrl_types.items():
                    self.assertEqual(attributes.get("jmIgnoreFoeProb"), "0")
                    attributes.pop("jmIgnoreFoeProb", None)
                    base_attributes = dict(base_types[type_id])
                    base_attributes.pop("jmIgnoreFoeProb", None)
                    self.assertEqual(attributes, base_attributes, type_id)
                for tag in ("vTypeDistribution", "route"):
                    self.assertEqual(
                        [_signature(item) for item in darrl_traffic.findall(tag)],
                        [_signature(item) for item in base_traffic.findall(tag)],
                    )

    def test_same_seed_requests_are_nested_and_use_lane_one_at_both_ends(self):
        ordered = sorted(
            DARRL_STEP_PROBABILITIES.items(), key=lambda item: item[1]
        )
        schedules = {}
        with tempfile.TemporaryDirectory(prefix="darrl_schedule_test_") as temp_dir:
            output_dir = Path(temp_dir)
            for name, _probability in ordered:
                source = ASSET_ROOT / name / "traffic" / "traffic_00000.rou.xml"
                output = output_dir / f"{name}.rou.xml"
                metadata = write_seeded_episode_traffic(name, 217, source, output)
                schedules[name] = metadata["departure_times_by_route"]
                root = ET.parse(output).getroot()
                vehicles = root.findall("vehicle")
                self.assertEqual(len(vehicles), metadata["scheduled_vehicle_count"])
                seen = set()
                for vehicle in vehicles:
                    self.assertEqual(vehicle.get("departLane"), "1")
                    self.assertEqual(vehicle.get("arrivalLane"), "1")
                    route_id = vehicle.get("route")
                    tick = round(float(vehicle.get("depart")) / 0.1)
                    self.assertNotIn((route_id, tick), seen)
                    seen.add((route_id, tick))
                    self.assertGreaterEqual(float(vehicle.get("depart")), 0.0)
                    self.assertLess(float(vehicle.get("depart")), 130.0)
                    self.assertAlmostEqual(tick * 0.1, float(vehicle.get("depart")))

        low, middle, high = (name for name, _ in ordered)
        for route_id in schedules[low]:
            low_times = set(schedules[low][route_id])
            middle_times = set(schedules[middle][route_id])
            high_times = set(schedules[high][route_id])
            self.assertLessEqual(low_times, middle_times)
            self.assertLessEqual(middle_times, high_times)

    def test_lane_one_connections_and_generic_warmup_lane_accounting(self):
        network = ET.parse(
            ASSET_ROOT / "intersection_sorted" / "map.net.xml"
        ).getroot()
        connections = {
            (
                item.get("from"), item.get("to"),
                item.get("fromLane"), item.get("toLane"),
            )
            for item in network.findall("connection")
        }
        for _name, config in (
            (name, get_random_intersection_config(name))
            for name in DARRL_STEP_PROBABILITIES
        ):
            for flow in config["flows"]:
                self.assertIn(
                    (flow["edges"][0], flow["edges"][1], "1", "1"),
                    connections,
                )

        class Simulation:
            def getTime(self):
                return 30.0

            def getPendingVehicles(self):
                return ()

        class Vehicle:
            def getIDList(self):
                return ("ego",)

            def getSpeed(self, _vehicle_id):
                return 1.0

        class Lane:
            def __init__(self):
                self.seen = []

            def getLastStepVehicleNumber(self, lane_id):
                self.seen.append(lane_id)
                return 0

            def getLastStepHaltingNumber(self, lane_id):
                self.seen.append(lane_id)
                return 0

        config = get_random_intersection_config(
            "intersection_random_darrl_medium_v1"
        )
        lane_api = Lane()
        fake_env = type("Env", (), {})()
        fake_env._connection = type("Connection", (), {
            "simulation": Simulation(), "vehicle": Vehicle(), "lane": lane_api,
        })()
        fake_env.specification = type("Specification", (), {"ego_id": "ego"})()
        fake_env._random_traffic_config = config
        fake_env._random_episode_schedule = {
            "departure_times_by_route": {flow["id"]: [] for flow in config["flows"]}
        }
        fake_env._random_departures = {flow["id"]: 0 for flow in config["flows"]}
        fake_env._random_warmup_checkpoints = []
        fake_env._random_arrivals = 0
        fake_env._random_delay_sum = 0.0
        fake_env._random_delay_count = 0
        fake_env._random_delay_max = 0.0
        PaperSumoSceneEnv._capture_random_warmup_checkpoint(fake_env, 30.0)
        checkpoint = fake_env._random_warmup_checkpoints[0]
        expected_lanes = {
            f"{flow['edges'][0]}_1" for flow in config["flows"]
        }
        self.assertEqual(set(checkpoint["inlet_lanes"]), expected_lanes)
        self.assertEqual(set(lane_api.seen), expected_lanes)


if __name__ == "__main__":
    unittest.main()
