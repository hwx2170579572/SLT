"""Bounded runner-level tests for policy shadow sampling and JSONL accounting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train_intersection_yield_v2_d1 import (
    _PolicyShadowProbeSink,
    _policy_shadow_eval_context,
    _policy_shadow_next_threshold,
    _policy_shadow_train_clock,
)


class PolicyShadowRunnerTests(unittest.TestCase):
    def test_train_clock_separates_observation_and_post_step_time(self):
        clock = _policy_shadow_train_clock(post_raw=63, raw_executed=3, decision_step=21)
        self.assertEqual(
            clock,
            {
                "pre_obs_raw": 60,
                "post_raw": 63,
                "pre_obs_decision": 20,
                "decision_step": 21,
            },
        )

    def test_train_sampling_respects_warmup_interval_and_unique_state_cap(self):
        due, trigger, next_threshold = _policy_shadow_next_threshold(
            pre_obs_raw=59,
            next_threshold=60,
            learning_starts=60,
            interval=100,
            samples_written=0,
        )
        self.assertFalse(due)
        self.assertEqual((trigger, next_threshold), (60, 60))

        due, trigger, next_threshold = _policy_shadow_next_threshold(
            pre_obs_raw=60,
            next_threshold=60,
            learning_starts=60,
            interval=100,
            samples_written=0,
        )
        self.assertTrue(due)
        self.assertEqual((trigger, next_threshold), (60, 160))

        due, trigger, next_threshold = _policy_shadow_next_threshold(
            pre_obs_raw=162,
            next_threshold=160,
            learning_starts=60,
            interval=100,
            samples_written=1,
        )
        self.assertTrue(due)
        self.assertEqual((trigger, next_threshold), (160, 260))

        due, _, unchanged = _policy_shadow_next_threshold(
            pre_obs_raw=999,
            next_threshold=260,
            learning_starts=60,
            interval=100,
            samples_written=20,
        )
        self.assertFalse(due)
        self.assertEqual(unchanged, 260)

    def test_eval_selector_uses_observation_only(self):
        obs = {
            "trajectory": np.ones((1, 3, 5, 6), dtype=np.float32),
            "route_reachability": np.asarray([[1, 0, -1]], dtype=np.float32),
        }
        selected = _policy_shadow_eval_context(obs)
        self.assertTrue(selected["valid_multi_car"])
        self.assertTrue(selected["valid_route_context"])

        empty = {
            "trajectory": np.zeros((1, 3, 5, 6), dtype=np.float32),
            "route_reachability": np.asarray([[-1, -1, -1]], dtype=np.float32),
        }
        selected_empty = _policy_shadow_eval_context(empty)
        self.assertFalse(selected_empty["valid_multi_car"])
        self.assertFalse(selected_empty["valid_route_context"])

    def test_sink_counts_unique_states_and_separates_na_from_errors(self):
        with tempfile.TemporaryDirectory(prefix="shadow-runner-") as temporary:
            sink = _PolicyShadowProbeSink(Path(temporary), "eval")
            sink.write_rows(
                [
                    {
                        "sample_id": "eval-e000-s00",
                        "probe_name": "active_probe",
                        "applicable": True,
                        "valid": True,
                    },
                    {
                        "sample_id": "eval-e000-s00",
                        "probe_name": "inactive_probe",
                        "applicable": False,
                        "valid": False,
                        "invalid_reason": "branch_inactive",
                    },
                    {
                        "sample_id": "eval-e000-s01",
                        "probe_name": "active_but_unavailable",
                        "applicable": True,
                        "valid": False,
                        "invalid_reason": "shadow_action_or_mean_unavailable",
                    },
                ]
            )
            summary = sink.close(metadata={"evaluation_complete": True})
            self.assertEqual(summary["rows_written"], 3)
            self.assertEqual(summary["unique_samples"], 2)
            self.assertEqual(summary["error_rows"], 0)
            self.assertEqual(summary["inactive_rows"], 1)
            self.assertEqual(summary["applicable_rows"], 2)
            self.assertEqual(summary["active_invalid_rows"], 1)
            saved = json.loads(
                (Path(temporary) / "diagnostics" / "eval" / "policy_shadow_probes_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(saved["unique_samples"], 2)


if __name__ == "__main__":
    unittest.main()
