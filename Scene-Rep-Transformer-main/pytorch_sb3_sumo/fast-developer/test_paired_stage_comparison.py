import json
from pathlib import Path
import tempfile
import unittest

from paired_stage_comparison import compare_stages


def episode(seed, outcome, traffic="traffic_10.rou.xml"):
    return {"seed": seed, "traffic_variant": traffic, "success": outcome == "success",
            "collision": outcome == "collision", "timeout": outcome == "timeout", "off_route": False,
            "raw_steps": 100, "episode_return": 1.0 if outcome == "success" else 0.0}


class PairedComparisonTests(unittest.TestCase):
    def compare(self, reference, candidate, expected_episodes=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in (("reference", reference), ("candidate", candidate)):
                folder = root / name
                folder.mkdir()
                (folder / "evaluation_results.json").write_text(json.dumps({
                    "identity": {"method": name}, "summary": {}, "episode_records": rows}), encoding="utf-8")
            output = root / "comparison.json"
            result = compare_stages(root / "reference", root / "candidate", output, "test", expected_episodes)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            return result

    def test_rescue_regression_and_new_timeout_all_count(self):
        reference = [episode(1, "collision"), episode(2, "success"), episode(3, "success")]
        candidate = [episode(3, "timeout"), episode(1, "success"), episode(2, "collision")]
        result = self.compare(reference, candidate)
        self.assertTrue(result["paired_outcomes_complete"])
        self.assertEqual(result["collision_to_success"], 1)
        self.assertEqual(result["success_to_collision"], 1)
        self.assertEqual(result["new_timeouts_from_non_timeout"], 1)
        self.assertEqual(result["matched_success_delta_count"], -1)
        self.assertFalse(result["paired_outcome_criterion_met"])
        self.assertEqual(len(result["changed_cases"]), 3)

    def test_collision_rescue_without_timeout_meets_outcome_criterion(self):
        result = self.compare([episode(1, "collision"), episode(2, "success")],
                              [episode(1, "success"), episode(2, "success")])
        self.assertTrue(result["paired_outcome_criterion_met"])
        self.assertEqual(result["matched_timeout_delta_count"], 0)

    def test_matching_seed_with_different_traffic_is_not_a_pair(self):
        result = self.compare([episode(1, "collision", "traffic_a")],
                              [episode(1, "success", "traffic_b")])
        self.assertFalse(result["paired_outcomes_complete"])
        self.assertEqual(result["matched_pairs"], 0)
        self.assertIsNone(result["paired_outcome_criterion_met"])

    def test_incomplete_evaluation_budget_blocks_interpretation(self):
        result = self.compare([episode(1, "collision")], [episode(1, "success")], expected_episodes=100)
        self.assertFalse(result["expected_episode_count_met"])
        self.assertFalse(result["paired_outcomes_complete"])
        self.assertIsNone(result["paired_outcome_criterion_met"])

    def test_duplicates_and_invalid_outcomes_block_interpretation(self):
        invalid = episode(2, "success")
        invalid["collision"] = True
        result = self.compare([episode(1, "collision"), episode(2, "success")],
                              [episode(1, "success"), episode(1, "success"), invalid])
        self.assertFalse(result["paired_outcomes_complete"])
        self.assertEqual(len(result["candidate_record_errors"]), 2)
        self.assertEqual(result["matched_pairs"], 0)
        self.assertIsNone(result["paired_outcome_criterion_met"])


if __name__ == "__main__":
    unittest.main()
