from __future__ import annotations

from types import SimpleNamespace
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

FD = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FD) not in sys.path:
    sys.path.insert(0, str(FD))

import train_intersection_yield_v2_d1_contractfix as contractfix_entry
from algos.sb3_torch.contractfix_encoder import ContractFixedIncrementalTopoEncoder
from envs.sumo.topology_graph_v2 import PaddedTopologyGraphV2


def _empty_graph() -> PaddedTopologyGraphV2:
    return PaddedTopologyGraphV2(
        lane_points=np.zeros((1, 10, 2), dtype=np.float32),
        lane_attrs=np.zeros((1, 8), dtype=np.float32),
        node_mask=np.zeros((1,), dtype=bool),
        edge_index=np.zeros((2, 1), dtype=np.int64),
        edge_type=np.zeros((1,), dtype=np.int64),
        edge_mask=np.zeros((1,), dtype=bool),
    )


class FakeDictEnv(gym.Env):
    """Small Dict/Box environment; no simulator, filesystem traffic or steps."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "trajectory": gym.spaces.Box(
                    -100.0, 100.0, shape=(6, 10, 5), dtype=np.float32
                ),
                "map": gym.spaces.Box(
                    -100.0, 100.0, shape=(12, 3, 5), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.reset_count = 0
        self.step_count = 0

    def _observation(self):
        trajectory = np.zeros((6, 10, 5), dtype=np.float32)
        for actor in range(6):
            trajectory[actor, :, 0] = np.arange(1, 11, dtype=np.float32) + actor
            trajectory[actor, :, 1] = float(actor)
            trajectory[actor, :, 2] = 0.1 * actor
            trajectory[actor, :, 3] = 2.0
        route_map = np.ones((12, 3, 5), dtype=np.float32)
        return {"trajectory": trajectory, "map": route_map}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        return self._observation(), {}

    def step(self, action):
        self.step_count += 1
        return self._observation(), 0.0, False, False, {}


class ContractfixEntrypointRoundTripTests(unittest.TestCase):
    def test_trainer_builder_cpu_save_load_preserves_fixed_encoder_and_output(self):
        torch.set_num_threads(1)
        contractfix_entry.install()
        trainer = contractfix_entry.trainer
        old_builder = trainer._build_topology_graph
        old_buffer_size = trainer.base.BUFFER_SIZE
        trainer._build_topology_graph = lambda env, return_info=False: (
            (_empty_graph(), SimpleNamespace(lane_ids=())) if return_info else _empty_graph()
        )
        trainer.base.BUFFER_SIZE = 64

        try:
            for method in contractfix_entry.METHODS:
                with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                    env = FakeDictEnv()
                    model = trainer._build_model_d1(
                        method, env, learning_starts=5, device="cpu"
                    )
                    extractor = model.actor.features_extractor
                    self.assertIsInstance(extractor, ContractFixedIncrementalTopoEncoder)
                    self.assertEqual(extractor.source_observation_contract, "smarts")

                    obs = env._observation()
                    tensor_obs, _ = model.policy.obs_to_tensor(obs)
                    model.policy.set_training_mode(False)
                    with torch.no_grad():
                        before = model.actor.features_extractor(tensor_obs).cpu()
                    self.assertEqual(tuple(before.shape), (1, 128))
                    self.assertTrue(bool(torch.isfinite(before).all()))

                    path = Path(tmp) / "contractfix_roundtrip.zip"
                    model.save(str(path))
                    loaded = type(model).load(
                        str(path), env=env, device="cpu", buffer_size=64
                    )
                    loaded.policy.set_training_mode(False)
                    loaded_extractor = loaded.actor.features_extractor
                    self.assertIsInstance(
                        loaded_extractor, ContractFixedIncrementalTopoEncoder
                    )
                    self.assertEqual(
                        loaded_extractor.source_observation_contract, "smarts"
                    )
                    loaded_obs, _ = loaded.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        after = loaded_extractor(loaded_obs).cpu()
                    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
                    self.assertEqual(env.step_count, 0)
                    loaded.env.close()
                    model.env.close()
        finally:
            trainer._build_topology_graph = old_builder
            trainer.base.BUFFER_SIZE = old_buffer_size


if __name__ == "__main__":
    unittest.main()
