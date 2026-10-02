from __future__ import annotations

import unittest

from tools.topo_v2_statistics import holm_adjust, paired_hierarchical_bootstrap


def _episodes(success: bool, collision: bool):
    return [
        {
            "episode": episode,
            "seed": 10_000 + episode,
            "traffic_variant": f"traffic_{episode % 2}.rou.xml",
            "success": success,
            "collision": collision,
        }
        for episode in range(4)
    ]


class TopologyV2StatisticsTest(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_paired_hierarchical_bootstrap_uses_both_levels(self):
        rows = []
        for training_seed in (0, 1, 2):
            rows.extend(
                [
                    {
                        "method": "baseline",
                        "scenario": "cross",
                        "seed": training_seed,
                        "episode_records": _episodes(False, True),
                    },
                    {
                        "method": "candidate",
                        "scenario": "cross",
                        "seed": training_seed,
                        "episode_records": _episodes(True, False),
                    },
                ]
            )
        result = paired_hierarchical_bootstrap(
            rows,
            candidate_method="candidate",
            baseline_method="baseline",
            scenarios=["cross"],
            resamples=100,
            seed=7,
        )
        success, collision = result["tests"]
        self.assertEqual(success["candidate_minus_baseline"], 1.0)
        self.assertEqual(success["ci_95"], {"lower": 1.0, "upper": 1.0})
        self.assertEqual(collision["candidate_minus_baseline"], -1.0)
        self.assertEqual(success["training_seed_count"], 3)
        self.assertEqual(success["paired_episodes_per_seed"], 4)
        self.assertTrue(result["seed_pairing"])
        self.assertTrue(result["episode_pairing"])

    def test_pairing_rejects_different_evaluation_seed(self):
        baseline = _episodes(False, False)
        candidate = _episodes(True, False)
        candidate[0] = {**candidate[0], "seed": 999}
        rows = [
            {
                "method": "baseline",
                "scenario": "cross",
                "seed": 0,
                "episode_records": baseline,
            },
            {
                "method": "candidate",
                "scenario": "cross",
                "seed": 0,
                "episode_records": candidate,
            },
        ]
        with self.assertRaises(ValueError):
            paired_hierarchical_bootstrap(
                rows,
                candidate_method="candidate",
                baseline_method="baseline",
                scenarios=["cross"],
                resamples=10,
            )


if __name__ == "__main__":
    unittest.main()
