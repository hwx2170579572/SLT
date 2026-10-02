from __future__ import annotations

from pathlib import Path
import sys
import unittest


FAST_DEV = Path(__file__).resolve().parent
if str(FAST_DEV) not in sys.path:
    sys.path.insert(0, str(FAST_DEV))

from diagnose_sorted_goalonly_causal_20261002 import _consume_one_step_refresh


class _FakeEnv:
    def __init__(self):
        self.refreshes = 0

    def _behavior_refresh_dynamic_context(self):
        self.refreshes += 1


class ConsumeOneStepRefreshTests(unittest.TestCase):
    def test_consumes_only_one_inner_refresh_then_restores_class_method(self):
        env = _FakeEnv()
        env._behavior_refresh_dynamic_context()
        self.assertEqual(env.refreshes, 1)

        with _consume_one_step_refresh(env) as state:
            env._behavior_refresh_dynamic_context()  # Inner env.step's duplicate.
            self.assertEqual(env.refreshes, 1)
            env._behavior_refresh_dynamic_context()  # Any later refresh remains live.
            self.assertEqual(env.refreshes, 2)
            self.assertEqual(state["calls"], 2)

        self.assertNotIn("_behavior_refresh_dynamic_context", env.__dict__)
        env._behavior_refresh_dynamic_context()
        self.assertEqual(env.refreshes, 3)

    def test_restores_class_method_when_inner_step_raises(self):
        env = _FakeEnv()
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            with _consume_one_step_refresh(env):
                env._behavior_refresh_dynamic_context()
                raise RuntimeError("step failed")

        self.assertNotIn("_behavior_refresh_dynamic_context", env.__dict__)
        env._behavior_refresh_dynamic_context()
        self.assertEqual(env.refreshes, 1)

    def test_restores_preexisting_instance_override_exactly(self):
        env = _FakeEnv()
        calls = []
        original_override = lambda: calls.append("original-instance")
        env._behavior_refresh_dynamic_context = original_override
        with _consume_one_step_refresh(env):
            env._behavior_refresh_dynamic_context()
        self.assertIs(env.__dict__["_behavior_refresh_dynamic_context"], original_override)
        env._behavior_refresh_dynamic_context()
        self.assertEqual(calls, ["original-instance"])


if __name__ == "__main__":
    unittest.main()
