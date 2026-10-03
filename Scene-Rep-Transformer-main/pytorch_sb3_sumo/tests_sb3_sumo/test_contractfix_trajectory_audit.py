"""Pure tests. No SUMO and no policy training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "fast-developer" / "contractfix_trajectory_audit.py"
_SPEC = importlib.util.spec_from_file_location("contractfix_trajectory_audit", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class CountingEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({"trajectory": gym.spaces.Box(-100, 100, (2, 6, 5))})
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.step_calls = 0
        self.reset_calls = 0
        self._raw_steps = 0
        self._history_timestep = 1
        self._last_observation_actor_keys = ("ego", "neighbor")
        self._histories = {"ego": [1], "neighbor": [1]}
        self._first_seen_steps = {"ego": 0, "neighbor": 0}
        self._last_seen_steps = {"ego": 0, "neighbor": 0}

    def reset(self, **kwargs):
        self.reset_calls += 1
        self._raw_steps = 0
        self.obs = {"trajectory": np.zeros((2, 6, 5), dtype=np.float32)}
        self.obs["trajectory"][:, -1, 0] = 10
        self.info = {"identity": "original"}
        return self.obs, self.info

    def step(self, action):
        self.step_calls += 1
        self.action_received = action
        self._raw_steps += 3
        self._history_timestep += 3
        self.obs = {"trajectory": np.ones((2, 6, 5), dtype=np.float32)}
        self.info = {"raw_steps_executed": np.int64(3), "identity": "original"}
        return self.obs, 0.5, self.step_calls % 2 == 0, False, self.info


class TrajectoryAuditTests(unittest.TestCase):
    def test_history_patterns_and_no_mutation(self):
        states = np.zeros((6, 6, 5), dtype=np.float32)
        patterns = [[1]*6, [0]*5+[1], [1,1,0,0,0,0], [1,0,1,0,1,0], [0]*6, [0,1,1,1,1,1]]
        states[:, :, 0] = patterns
        before = states.copy()
        report = audit.audit_trajectory(states, geometry_active=False)
        social = report["social"]
        self.assertEqual(social["nonempty"], 4)
        self.assertEqual(social["empty"], 1)
        self.assertEqual(social["short_nonempty"], 4)
        self.assertEqual(social["old_index_mismatch"], 3)
        self.assertEqual(social["old_selected_padding"], 1)
        self.assertEqual(social["old_selected_earlier_valid"], 2)
        self.assertEqual(social["internal_gaps"], 1)
        np.testing.assert_array_equal(before, states)

    def test_north_following_and_parallel_miss(self):
        states = np.zeros((2, 1, 5), dtype=np.float64)
        # Both x != 0; north heading, legacy pseudo velocity in channel vx.
        states[0, 0] = [10, 0, 0, 10, 0]
        states[1, 0] = [10, 20, 0, 5, 0]
        g = audit.audit_trajectory(states)["geometry"]
        self.assertEqual(g["approaching"], 1)
        self.assertEqual(g["old_zero_new_nonzero_closing"], 1)
        self.assertAlmostEqual(g["closing_abs_old_new_difference_sum_mps"], 5)
        self.assertAlmostEqual(g["bounded_miss_distance_sum_m"], 0)
        states[1, 0, 0] = 20
        g = audit.audit_trajectory(states)["geometry"]
        self.assertAlmostEqual(g["bounded_miss_distance_sum_m"], 10)
        self.assertEqual(g["time_semantics"], "constant_velocity_closest_approach_not_TTC")

    def test_zero_relative_motion_is_undefined_time(self):
        states = np.array([[[10, 0, 0, 3, 0]], [[10, 20, 0, 3, 0]]], dtype=float)
        g = audit.audit_trajectory(states)["geometry"]
        self.assertEqual(g["zero_relative_speed"], 1)
        self.assertEqual(g["relative_motion_defined"], 0)
        self.assertEqual(g["closest_time_in_future_horizon"], 0)
        self.assertEqual(g["bounded_miss_distance_sum_m"], 20)

    def test_cartesian_contract_is_not_converted(self):
        states = np.array([[[10, 0, 0, 0, 10]], [[10, 20, 0, 0, 5]]], dtype=float)
        g = audit.audit_trajectory(states, velocity_contract="cartesian")["geometry"]
        self.assertEqual(g["closing_abs_old_new_difference_sum_mps"], 0)
        self.assertEqual(g["approaching"], 1)

    def test_empty_and_zero_x_ambiguity_and_weighted_aggregate(self):
        states = np.zeros((2, 6, 5))
        states[1, 3, 1] = 8
        row = audit.audit_trajectory(states)
        self.assertEqual(row["social"]["empty"], 1)
        self.assertEqual(row["social"]["x_zero_other_fields_nonzero_slots"], 1)
        total = {}
        audit.add_audit_counts(total, row)
        self.assertIsNone(audit.audit_rates(total)["social"]["short_nonempty_rate"])
        states[1, -1, 0] = 10
        audit.add_audit_counts(total, audit.audit_trajectory(states))
        self.assertEqual(audit.audit_rates(total)["social"]["short_nonempty_rate"], 1)
        self.assertEqual(total["social"]["history_observations"], 2)

    def test_unknown_contract_and_shape_fail(self):
        with self.assertRaises(ValueError):
            audit.audit_trajectory(np.zeros((2, 6, 5)), velocity_contract="guess")
        with self.assertRaises(ValueError):
            audit.audit_trajectory(np.zeros((60,)))

    def test_wrapper_preserves_flow_and_counts_only_preaction_states(self):
        with tempfile.TemporaryDirectory() as directory:
            env = CountingEnv()
            wrapper = audit.TrajectoryHistoryAuditWrapper(env, run_dir=directory, phase="train",
                                                         method="test", trajectory_shape=(2, 6, 5))
            initial, info = wrapper.reset(seed=0)
            self.assertIs(initial, env.obs)
            self.assertIs(info, env.info)
            action = np.zeros(2)
            first = wrapper.step(action)
            self.assertIs(first[0], env.obs)
            self.assertIs(first[-1], env.info)
            self.assertIs(action, env.action_received)
            wrapper.step(action)
            self.assertEqual(env.reset_calls, 1)
            self.assertEqual(env.step_calls, 2)
            wrapper.close()
            log = Path(directory) / "diagnostics/train/trajectory_history_audit.jsonl"
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            decisions = [row for row in rows if row["record_type"] == "decision"]
            self.assertEqual(len(decisions), 2)
            self.assertEqual(decisions[0]["preaction_metadata"]["actors"][0]["tracked_age_slots"], 1)
            self.assertEqual(decisions[0]["audit"]["social"]["old_selected_padding"], 1)
            self.assertEqual(decisions[1]["audit"]["social"]["full"], 1)
            summary = json.loads(log.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["real_decision_rows"], 2)
            self.assertEqual(summary["raw_steps_counted"], 6)
            self.assertEqual(summary["errors"], 0)
            self.assertEqual(summary["completed_episodes"], 1)
            self.assertEqual(summary["actor_cache_counts"]["decisions_with_actor_history_cache"], 2)
            self.assertEqual(summary["rates"]["social"]["old_index_mismatch_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
