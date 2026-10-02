from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.action_diagnostics_v2 import evaluate_with_action_diagnostics_v2


class _DummyModel:
    policy = None

    def predict(self, observation, deterministic=True):
        del deterministic
        step = int(observation["step"])
        lateral = (-0.5, 0.0, 0.5)[step % 3]
        return np.asarray([0.2, lateral], dtype=np.float32), None


class _DummyEnvironment:
    def __init__(self) -> None:
        self.unwrapped = self
        self.specification = SimpleNamespace(
            ego_id="ego", source_observation_contract="smarts"
        )
        self._step = 0
        self._seed = 0

    def _observation(self):
        trajectory = np.zeros((1, 4, 5), dtype=np.float32)
        trajectory[0, min(self._step, 3), 3] = float(self._step + 1)
        return {"step": self._step, "trajectory": trajectory}

    def _state(self, actor_key):
        if actor_key != "vehicle:ego":
            return None
        return np.asarray([0.0, 0.0, 0.0, self._step + 1.0, 0.0])

    def reset(self, *, seed=None):
        self._seed = int(seed)
        self._step = 0
        return self._observation(), {"traffic_variant": "dummy.rou.xml"}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        lateral = float(action[1])
        command = -1 if lateral < -1 / 3 else 1 if lateral > 1 / 3 else 0
        self._step += 1
        done = self._step == 3
        success = done and self._seed % 2 == 0
        info = {
            "undiscounted_reward": float(success),
            "raw_simulation_steps": self._step * 3,
            "lane_command": command,
            "lane_change_applied": command != 0,
            "target_speed": (float(action[0]) + 1.0) * 5.0,
            "effective_target_speed": 5.5,
            "traffic_variant": "dummy.rou.xml",
            "is_success": success,
            "collision": done and not success,
            "off_route": False,
            "max_time": False,
        }
        return self._observation(), float(success), done, False, info


class ActionDiagnosticsV2Test(unittest.TestCase):
    def test_trace_and_outcomes_share_one_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "actions.jsonl"
            report, diagnostics = evaluate_with_action_diagnostics_v2(
                _DummyModel(),
                _DummyEnvironment(),
                episodes=2,
                seed=10,
                trace_path=trace,
            )
            records = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(records), 6)
        self.assertEqual(diagnostics["decision_records"], 6)
        self.assertEqual(
            diagnostics["lane_command_counts"],
            {"negative": 2, "keep": 2, "positive": 2},
        )
        self.assertEqual(diagnostics["outcomes"], report.summary.to_dict())
        self.assertEqual(report.summary.success_rate, 0.5)
        self.assertTrue(diagnostics["aligned_with_paper_evaluation"])
        self.assertEqual(len(diagnostics["trace"]["sha256"]), 64)
        self.assertEqual(diagnostics["actual_speed_mps"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
