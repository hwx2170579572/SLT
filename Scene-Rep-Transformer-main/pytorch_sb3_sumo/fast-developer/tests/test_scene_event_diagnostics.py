"""Diagnostic accounting tests; these do not run traffic simulation."""
import unittest

import numpy as np

from scene_event.diagnostics import HistoryAudit, ShadowBudget


class DiagnosticsTests(unittest.TestCase):
    def test_training_budget_counts_states_and_thresholds(self):
        budget = ShadowBudget()
        accepted = [raw for raw in range(1, 150001) if budget.take("train", 0, raw, raw)]
        self.assertEqual(accepted, list(range(5000, 100001, 5000)))
        self.assertEqual(budget.summary()["train_unique_states"], 20)
        self.assertFalse(budget.take("train", 0, 100000, 200000))

    def test_eval_budget_is_per_episode_and_short_episodes_not_inflated(self):
        budget = ShadowBudget()
        accepted = [(episode, decision) for episode in range(2) for decision in range(100)
                    if budget.take("eval", episode, decision, decision * 3)]
        self.assertEqual(len(accepted), 8)
        self.assertEqual(budget.summary()["eval_states_per_episode"], {"0": 4, "1": 4})

    def test_actual_mask_indices_and_internal_gaps(self):
        observation = {"history_valid": np.array([
            [False, False, True, True], [True, True, False, False],
            [True, False, True, True], [False, False, False, False],
            [True, True, True, True]], dtype=bool), "actor_valid": np.ones(5, dtype=bool)}
        row = HistoryAudit().observe(observation)
        self.assertEqual(row["short_nonempty"], 3)
        self.assertEqual(row["left_padding"], 1)
        self.assertEqual(row["right_padding"], 1)
        self.assertEqual(row["internal_gaps"], 1)
        self.assertEqual(row["empty_active"], 1)
        self.assertEqual(row["old_selector_mismatch"], 2)
        self.assertEqual(row["old_selector_padding"], 1)


if __name__ == "__main__":
    unittest.main()
