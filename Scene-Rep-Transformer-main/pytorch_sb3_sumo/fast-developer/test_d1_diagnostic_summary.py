"""D1 evaluation summaries keep raw, shaped, and slot-readout evidence distinct."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_DEVELOPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FAST_DEVELOPER))

from algos.sb3_torch.evaluation import REWARD_COMPONENT_KEYS, REWARD_COMPONENT_PROTOCOL_VERSION
from train_intersection_yield_v2_d1 import (
    _d1_evaluation_return_summary,
    _slot_linear_contributions,
    _slot_readout_metric_scalars,
)


class D1EvaluationSummaryTests(unittest.TestCase):
    def test_shaped_and_raw_return_are_separate_and_components_reconcile(self):
        first = {
            "episode_return": 2.0,
            "raw_episode_return": 1.5,
            "reward_success": 1.0,
            "reward_collision": 0.0,
            "reward_off_route": 0.0,
            "reward_timeout": 0.0,
            "reward_step_cost": -0.1,
            "reward_progress": 1.1,
            "reward_component_protocol_version": REWARD_COMPONENT_PROTOCOL_VERSION,
        }
        second = {
            "episode_return": -1.0,
            "raw_episode_return": None,
            "reward_success": 0.0,
            "reward_collision": -1.0,
            "reward_off_route": 0.0,
            "reward_timeout": 0.0,
            "reward_step_cost": -0.2,
            "reward_progress": None,
            "reward_component_protocol_version": None,
        }
        summary = _d1_evaluation_return_summary(
            [first, second],
            evaluation_protocol_version="environment_step_reward_v2",
            raw_protocol_version="raw-v1",
            component_keys=REWARD_COMPONENT_KEYS,
        )
        self.assertEqual(summary["mean_return"], 0.5)
        self.assertEqual(summary["std_return"], 1.5)
        self.assertEqual(summary["raw_mean_return"], 1.5)
        self.assertEqual(summary["raw_return_coverage"], 1)
        self.assertEqual(summary["reward_component_coverage"], 1)
        self.assertAlmostEqual(
            summary["reward_component_reconciliation_max_abs_error"], 0.0
        )
        self.assertIsNone(summary["reward_component_protocol_version"])
        self.assertEqual(summary["reward_component_means"]["reward_progress"], 1.1)


class SlotReadoutDecompositionTests(unittest.TestCase):
    def test_existing_linear_input_decomposes_exactly_without_gradient_or_rng_change(self):
        torch.manual_seed(13)
        features = torch.randn(2, 128, requires_grad=True)
        weight = torch.randn(5, 128, requires_grad=True)
        bias = torch.randn(5)
        state_before = torch.random.get_rng_state().clone()
        parts = _slot_linear_contributions(features, weight)
        self.assertIsNotNone(parts)
        reconstructed = parts["ego"] + parts["social"] + parts["route"] + bias
        actual = torch.nn.functional.linear(features.detach(), weight.detach(), bias)
        self.assertTrue(torch.allclose(reconstructed, actual, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))
        self.assertIsNone(features.grad)
        self.assertFalse(parts["ego"].requires_grad)
        self.assertFalse(parts["social"].requires_grad)
        self.assertFalse(parts["route"].requires_grad)
        self.assertIsNone(_slot_linear_contributions(features[:, :64], weight))

    def test_predict_metric_merge_preserves_null_and_excludes_hook_scratch(self):
        merged = _slot_readout_metric_scalars(
            {
                "actor_readout_ego_preactivation_l2": 1.25,
                "actor_readout_decomposition_error_l2": None,
                "components": {"ego": torch.ones(1)},
                "label": "diagnostic metadata, not a numeric metric",
                "bad_number": float("nan"),
            }
        )
        self.assertEqual(merged["actor_readout_ego_preactivation_l2"], 1.25)
        self.assertIsNone(merged["actor_readout_decomposition_error_l2"])
        self.assertIsNone(merged["bad_number"])
        self.assertNotIn("components", merged)
        self.assertNotIn("label", merged)


if __name__ == "__main__":
    unittest.main()
