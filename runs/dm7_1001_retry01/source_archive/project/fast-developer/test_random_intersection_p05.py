"""Tests for the per-step Bernoulli traffic schedule variant."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.sumo.random_intersection import (
    ASSET_ROOT,
    STEP_P05_SCENARIO,
    ensure_random_intersection_assets,
    get_random_intersection_config,
    write_seeded_episode_traffic,
)


class RandomIntersectionP05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real frozen-asset validator/generator before using its
        # traffic template; tests must not synthesize or bypass these assets.
        cls.manifests = ensure_random_intersection_assets()
        cls.manifest = cls.manifests[STEP_P05_SCENARIO]
        cls.traffic_path = ASSET_ROOT / STEP_P05_SCENARIO / "traffic" / "traffic_00000.rou.xml"
        cls.medium_traffic_path = ASSET_ROOT / "intersection_random_medium_v1" / "traffic" / "traffic_00000.rou.xml"

    def test_static_assets_and_episode_template_contract(self):
        self.assertTrue(self.traffic_path.is_file())
        self.assertEqual(
            hashlib.sha256(self.traffic_path.read_bytes()).hexdigest(),
            self.manifest["asset_sha256"]["traffic/traffic_00000.rou.xml"],
        )
        root = ET.parse(self.traffic_path).getroot()
        self.assertEqual(len(root.findall("vType")), 120)
        self.assertEqual(len(root.findall("route")), 3)
        self.assertEqual(len(root.findall("vTypeDistribution")), 3)
        self.assertEqual(root.findall("flow"), [])
        self.assertEqual(root.findall("vehicle"), [])

        medium_manifest = self.manifests["intersection_random_medium_v1"]
        for relative in ("map.net.xml", "ego.rou.xml"):
            self.assertEqual(
                self.manifest["asset_sha256"][relative],
                medium_manifest["asset_sha256"][relative],
            )
            self.assertEqual(
                (ASSET_ROOT / STEP_P05_SCENARIO / relative).read_bytes(),
                (ASSET_ROOT / "intersection_random_medium_v1" / relative).read_bytes(),
            )
        medium_root = ET.parse(self.medium_traffic_path).getroot()
        for tag in ("vType", "vTypeDistribution", "route"):
            p05_definitions = [element.attrib for element in root.findall(tag)]
            medium_definitions = [element.attrib for element in medium_root.findall(tag)]
            self.assertEqual(p05_definitions, medium_definitions, msg=f"changed {tag} definitions")

        config = get_random_intersection_config(STEP_P05_SCENARIO)
        self.assertTrue(config["dynamic_episode_traffic"])
        self.assertEqual(config["total_vehicles_per_hour"], 54000)
        self.assertEqual(config["step_length_seconds"], 0.1)
        self.assertEqual(config["flow_begin_seconds"], 0)
        self.assertEqual(config["flow_end_seconds"], 130)
        for flow in config["flows"]:
            self.assertEqual(flow["vehicles_per_hour"], 18000)
            self.assertIsNone(flow["sumo_probability"])
            self.assertEqual(flow["probability_per_step"], 0.5)

    def test_seeded_schedule_is_byte_reproducible_and_seed_sensitive(self):
        with tempfile.TemporaryDirectory(prefix="p05_schedule_test_") as temp_dir:
            temp = Path(temp_dir)
            first_path = temp / "first.rou.xml"
            repeat_path = temp / "repeat.rou.xml"
            other_seed_path = temp / "other_seed.rou.xml"

            first = write_seeded_episode_traffic(
                STEP_P05_SCENARIO, 731, self.traffic_path, first_path,
            )
            repeated = write_seeded_episode_traffic(
                STEP_P05_SCENARIO, 731, self.traffic_path, repeat_path,
            )
            other = write_seeded_episode_traffic(
                STEP_P05_SCENARIO, 732, self.traffic_path, other_seed_path,
            )

            self.assertEqual(first_path.read_bytes(), repeat_path.read_bytes())
            self.assertEqual(first["sha256"], repeated["sha256"])
            self.assertEqual(first["departure_times_by_route"], repeated["departure_times_by_route"])
            self.assertEqual(first["actual_seed"], 731)
            self.assertNotEqual(first_path.read_bytes(), other_seed_path.read_bytes())
            self.assertNotEqual(first["sha256"], other["sha256"])
            self.assertEqual(first["sha256"], hashlib.sha256(first_path.read_bytes()).hexdigest())

    def test_schedule_has_one_request_per_route_tick_on_the_01_second_grid(self):
        config = get_random_intersection_config(STEP_P05_SCENARIO)
        dt = config["step_length_seconds"]
        opportunities = round((config["flow_end_seconds"] - config["flow_begin_seconds"]) / dt)
        self.assertEqual(opportunities, 1300)

        with tempfile.TemporaryDirectory(prefix="p05_schedule_test_") as temp_dir:
            output_path = Path(temp_dir) / "episode.rou.xml"
            metadata = write_seeded_episode_traffic(
                STEP_P05_SCENARIO, 123456, self.traffic_path, output_path,
            )
            departures = metadata["departure_times_by_route"]
            self.assertEqual(set(departures), {flow["id"] for flow in config["flows"]})
            self.assertEqual(
                metadata["scheduled_vehicle_count"],
                sum(metadata["scheduled_count_by_route"].values()),
            )

            for route_id, times in departures.items():
                with self.subTest(route=route_id):
                    self.assertEqual(len(times), metadata["scheduled_count_by_route"][route_id])
                    self.assertEqual(len(times), len(set(times)))
                    self.assertTrue(all(0 <= time < config["flow_end_seconds"] for time in times))
                    self.assertTrue(all(abs(time / dt - round(time / dt)) < 1e-7 for time in times))
                    # A deterministic, 1300-trial Bernoulli sample should be
                    # near p=.5; this is a protocol sanity bound, not calibration.
                    observed_rate = len(times) / opportunities
                    self.assertGreaterEqual(observed_rate, 0.45)
                    self.assertLessEqual(observed_rate, 0.55)

            root = ET.parse(output_path).getroot()
            vehicles = root.findall("vehicle")
            self.assertEqual(len(vehicles), metadata["scheduled_vehicle_count"])
            by_route = {}
            ticks_by_route = {}
            for vehicle in vehicles:
                route_id = vehicle.attrib["route"].removeprefix("route_")
                depart = float(vehicle.attrib["depart"])
                tick = round(depart / dt)
                self.assertAlmostEqual(depart, tick * dt, places=7)
                self.assertGreaterEqual(depart, 0)
                self.assertLess(depart, config["flow_end_seconds"])
                by_route[route_id] = by_route.get(route_id, 0) + 1
                ticks_by_route.setdefault(route_id, set()).add(tick)
            self.assertEqual(by_route, metadata["scheduled_count_by_route"])
            for route_id, route_ticks in ticks_by_route.items():
                self.assertEqual(len(route_ticks), by_route[route_id])

    def test_existing_medium_native_flow_configuration_is_unchanged(self):
        config = get_random_intersection_config("intersection_random_medium_v1")
        self.assertEqual(config["protocol"], "intersection_bernoulli_v1")
        self.assertEqual(config["total_vehicles_per_hour"], 1180)
        self.assertFalse(config.get("dynamic_episode_traffic", False))
        expected_rates = (400, 300, 480)
        self.assertEqual(
            [flow["vehicles_per_hour"] for flow in config["flows"]],
            list(expected_rates),
        )
        for flow, rate in zip(config["flows"], expected_rates):
            self.assertAlmostEqual(flow["sumo_probability"], rate / 3600.0)
            self.assertAlmostEqual(flow["probability_per_step"], rate / 36000.0)


if __name__ == "__main__":
    unittest.main()
