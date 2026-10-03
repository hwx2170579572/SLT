"""Offline integration checks against the complete compiled public-map export."""
from pathlib import Path
import unittest

import numpy as np
import torch

from scene_event.encoder import build_encoder
from scene_event.polyline import PublicMap
from scene_event.schema import empty_observation


MAP_EXPORT = (Path(__file__).parent / "analysis" / "scene_representation_redesign_20261003" /
             "implementation_validation" / "intersection_sorted_public_map_corridor_v2.npz")


class PublicMapIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MAP_EXPORT.is_file():
            raise FileNotFoundError(f"Expected the full compiled PublicMapCache export at {MAP_EXPORT}")
        with np.load(MAP_EXPORT, allow_pickle=False) as source:
            # Keep every exported array, including collector-only geometry, so
            # this catches model/collector boundary mismatches end to end.
            cls.public_map = {key: source[key].copy() for key in source.files}

    def test_complete_export_constructs_and_runs_m0_and_m1(self):
        values = self.public_map
        lanes = values["lane_points_xy"].shape[0]
        zones = values["zone_valid"].shape[0]
        self.assertEqual(values["lane_zone_distance_m"].shape, (lanes, zones))
        self.assertTrue(np.isfinite(values["lane_zone_distance_m"]).all())
        self.assertTrue((values["lane_zone_distance_m"] >= 0).all())

        observation_np = empty_observation(24, 21, 4, 16, zones, 60.0)
        observation = {key: torch.from_numpy(value).unsqueeze(0)
                       for key, value in observation_np.items()}
        for method in ("m0", "m1"):
            encoder = build_encoder(method, width=16, z_dim=16, public_map=values)
            self.assertNotIn("lane_zone_distance_m", encoder.public_map.state_dict())
            with torch.no_grad():
                latent = encoder(observation)
            self.assertEqual(tuple(latent.shape), (1, 16))
            self.assertTrue(torch.isfinite(latent).all())

    def test_only_the_named_auxiliary_key_is_accepted_and_validated(self):
        values = self.public_map
        # The existing encoder contract may omit this collector-only field.
        required_only = {key: value for key, value in values.items() if key != "lane_zone_distance_m"}
        PublicMap(required_only)

        unexpected = dict(values)
        unexpected["unreviewed_extra"] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            PublicMap(unexpected)

        malformed = dict(values)
        malformed["lane_zone_distance_m"] = np.zeros((values["lane_points_xy"].shape[0], 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"\[L,Z\]"):
            PublicMap(malformed)

        negative = dict(values)
        negative["lane_zone_distance_m"] = values["lane_zone_distance_m"].copy()
        negative["lane_zone_distance_m"][0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "float32, finite, and nonnegative"):
            PublicMap(negative)

        wrong_dtype = dict(values)
        wrong_dtype["lane_zone_distance_m"] = values["lane_zone_distance_m"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            PublicMap(wrong_dtype)


if __name__ == "__main__":
    unittest.main()
