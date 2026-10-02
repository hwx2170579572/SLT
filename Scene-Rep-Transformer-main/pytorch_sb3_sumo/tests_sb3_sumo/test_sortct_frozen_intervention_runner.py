from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest


FD = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FD) not in sys.path:
    sys.path.insert(0, str(FD))

import diagnose_sortct_frozen_interventions_20261003 as runner


class FakeEncoder:
    def __init__(self):
        self.active = None

    @contextmanager
    def policy_shadow_probe(self, name):
        if self.active is not None:
            raise RuntimeError("nested probe context")
        old = self.active
        self.active = name
        try:
            yield {}
        finally:
            self.active = old


class FakeModel:
    def __init__(self, encoder):
        self.encoder = encoder
        self.fail = False

    def predict(self, observation, deterministic=True):
        if self.encoder.active != "route_conflict_off":
            raise AssertionError("prediction did not execute inside the probe context")
        if self.fail:
            raise ValueError("fake prediction failure")
        return {"action": observation, "deterministic": deterministic}


def episode(seed=10000, outcome="success", **extra):
    row = {
        "seed": seed,
        "traffic_variant": "traffic_00.rou.xml",
        "success": outcome == "success",
        "collision": outcome == "collision",
        "timeout": outcome == "timeout",
        "off_route": outcome == "off_route",
        "raw_steps": 12,
        "decision_steps": 4,
        "environment_steps": 12,
        "episode_return": 1.25,
        "raw_episode_return": 1.0,
        "reward_success": 10.0 if outcome == "success" else 0.0,
        "reward_collision": -10.0 if outcome == "collision" else 0.0,
        "reward_off_route": -10.0 if outcome == "off_route" else 0.0,
        "reward_timeout": -5.0 if outcome == "timeout" else 0.0,
        "reward_step_cost": -0.04,
        "reward_progress": 1.29,
    }
    row.update(extra)
    return row


class FrozenInterventionRunnerTests(unittest.TestCase):
    def test_predict_probe_preserves_result_and_restores_context(self):
        encoder = FakeEncoder()
        model = FakeModel(encoder)
        restore, calls = runner._install_predict_probe(model, encoder, "route_conflict_off")
        self.assertEqual(model.predict(17), {"action": 17, "deterministic": True})
        self.assertEqual(calls["predict_calls"], 1)
        self.assertIsNone(encoder.active)
        restore()
        self.assertNotIn("predict", model.__dict__)
        self.assertIs(model.predict.__func__, FakeModel.predict)

    def test_predict_probe_restores_context_on_exception(self):
        encoder = FakeEncoder()
        model = FakeModel(encoder)
        model.fail = True
        restore, calls = runner._install_predict_probe(model, encoder, "route_conflict_off")
        with self.assertRaisesRegex(ValueError, "fake prediction failure"):
            model.predict(1)
        self.assertEqual(calls["predict_calls"], 1)
        self.assertIsNone(encoder.active)
        restore()
        self.assertNotIn("predict", model.__dict__)
        self.assertIs(model.predict.__func__, FakeModel.predict)

    def test_control_gate_compares_returns_steps_and_terminal_fields(self):
        expected = episode()
        actual = dict(expected)
        self.assertEqual(runner._control_mismatches(expected, actual), [])
        actual["raw_steps"] += 1
        actual["episode_return"] += 1e-4
        fields = {item["field"] for item in runner._control_mismatches(expected, actual)}
        self.assertEqual(fields, {"raw_steps", "episode_return"})

    def test_pair_matrix_is_reference_rows_to_candidate_columns(self):
        ref = [episode(10000, "success"), episode(10001, "collision"), episode(10002, "timeout")]
        candidate = [episode(10000, "collision"), episode(10001, "success"), episode(10002, "timeout")]
        result = runner._pair_records(ref, candidate)
        self.assertEqual(result["paired_count"], 3)
        self.assertEqual(result["traffic_variant_mismatch_seeds"], [])
        self.assertEqual(result["matrix"]["success"]["collision"], 1)
        self.assertEqual(result["matrix"]["collision"]["success"], 1)
        self.assertEqual(result["matrix"]["timeout"]["timeout"], 1)


if __name__ == "__main__":
    unittest.main()
