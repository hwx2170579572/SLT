"""Protocol invariants for the versioned random intersection scenarios."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from envs.sumo.scenario_registry import available_scenarios, base_runnable_scenarios
from envs.sumo.high_density_env_v1 import HighDensityPaperSumoSceneEnvV1
from envs.sumo.random_intersection import (
    ASSET_ROOT,
    DARRL_STEP_PROBABILITIES,
    RANDOM_INTERSECTION_SCENARIOS,
    ensure_random_intersection_assets,
    get_random_intersection_config,
    random_intersection_seed,
)


def _xml_signature(element: ET.Element):
    text = element.text if element.text and element.text.strip() else None
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_xml_signature(child) for child in element),
    )


class RandomIntersectionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = ensure_random_intersection_assets()

    def test_geometry_ego_and_time_contract_are_preserved(self):
        source = ASSET_ROOT / "intersection_sorted"
        for name in RANDOM_INTERSECTION_SCENARIOS:
            with self.subTest(scenario=name):
                spec = get_paper_scenario_spec(name)
                self.assertEqual(spec.max_episode_steps, 600)
                self.assertEqual(spec.ego_depart_time, 50.0)
                self.assertEqual(spec.topology_max_edges, 512)
                self.assertEqual(
                    (source / "map.net.xml").read_bytes(),
                    (spec.asset_directory / "map.net.xml").read_bytes(),
                )
                if name not in DARRL_STEP_PROBABILITIES:
                    self.assertEqual(
                        (source / "ego.rou.xml").read_bytes(),
                        (spec.asset_directory / "ego.rou.xml").read_bytes(),
                    )
                else:
                    expected_ego = ET.parse(source / "ego.rou.xml").getroot()
                    ego_type = expected_ego.find("vType")
                    self.assertIsNotNone(ego_type)
                    config = get_random_intersection_config(name)
                    for key, value in config["ego_vtype_overrides"].items():
                        ego_type.set(key, value)
                    actual_ego = ET.parse(
                        spec.asset_directory / "ego.rou.xml"
                    ).getroot()
                    self.assertEqual(
                        _xml_signature(actual_ego), _xml_signature(expected_ego)
                    )

    def test_density_is_the_only_flow_definition_difference(self):
        definitions = []
        # The three original rate profiles remain native SUMO probability
        # flows. The separately versioned p05 profile uses a seeded explicit
        # per-step vehicle schedule and is tested independently.
        rate_scenarios = RANDOM_INTERSECTION_SCENARIOS[:3]
        for index, name in enumerate(rate_scenarios, start=1):
            root = ET.parse(get_paper_scenario_spec(name).traffic_paths[0]).getroot()
            self.assertEqual(len(root.findall("vType")), 120)
            self.assertEqual(len(root.findall("vTypeDistribution")), 3)
            self.assertEqual(len(root.findall("vehicle")), 0)
            flows = root.findall("flow")
            self.assertEqual(len(flows), 3)
            expected = (200 * index, 150 * index, 240 * index)
            for flow, rate in zip(flows, expected):
                self.assertAlmostEqual(float(flow.attrib["probability"]) * 3600.0, rate)
                self.assertNotIn("period", flow.attrib)
                self.assertEqual(flow.attrib["begin"], "0")
                self.assertEqual(flow.attrib["end"], "130")
                flow.set("probability", "DENSITY_ONLY")
            definitions.append(ET.tostring(root))
        self.assertEqual(definitions[0], definitions[1])
        self.assertEqual(definitions[1], definitions[2])

    def test_splits_are_disjoint_and_reproducible(self):
        domains = {}
        for split in ("train", "validation", "test"):
            domains[split] = {random_intersection_seed(split, key) for key in range(10000, 10100)}
            self.assertEqual(len(domains[split]), 100)
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(domains[first] & domains[second])
        self.assertLess(random_intersection_seed("test", 2**31 - 1), 2**31 - 1)
        with self.assertRaises(ValueError):
            random_intersection_seed("test", -1)

    def test_assets_are_registered_separately_and_have_valid_hashes(self):
        for name in RANDOM_INTERSECTION_SCENARIOS:
            self.assertIn(name, available_scenarios())
            self.assertNotIn(name, base_runnable_scenarios())
            manifest = self.manifests[name]
            self.assertEqual(manifest["configuration"], get_random_intersection_config(name))
            for relative, expected in manifest["asset_sha256"].items():
                actual = hashlib.sha256((ASSET_ROOT / name / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
        self.assertEqual(get_paper_scenario_spec("intersection_sorted").name, "intersection_sorted")

    def test_dynamic_episode_schedule_uses_distinct_overlay_cache_keys(self):
        # Call the pure path builder on a lightweight receiver. Constructing
        # the SUMO env with __new__ would trigger its real destructor without
        # initialized TraCI fields when this test ends.
        env = SimpleNamespace(
            _high_density_overlay_root=Path("unused-overlay-root"),
            scenario="intersection_random_medium_p05_v1",
            _high_density_vehicle_scale=1.0,
        )
        source = Path("episode.rou.xml")

        env._random_episode_schedule = {"sha256": "a" * 64}
        first = HighDensityPaperSumoSceneEnvV1._overlay_path(env, source)
        same_schedule = HighDensityPaperSumoSceneEnvV1._overlay_path(env, source)
        env._random_episode_schedule = {"sha256": "b" * 64}
        second = HighDensityPaperSumoSceneEnvV1._overlay_path(env, source)

        self.assertEqual(first, same_schedule)
        self.assertNotEqual(first, second)
        self.assertIn("__saaaaaaaaaaaaaaaa", first.name)
        env._random_episode_schedule = None
        self.assertEqual(
            HighDensityPaperSumoSceneEnvV1._overlay_path(env, source).name,
            "episode.rou__hdx1_v1.rou.xml",
        )


if __name__ == "__main__":
    unittest.main()
