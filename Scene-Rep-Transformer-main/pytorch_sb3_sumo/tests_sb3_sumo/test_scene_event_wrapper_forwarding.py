from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.env import SceneEventRewardAuditWrapper  # noqa: E402


class _RawSceneEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._collection = {"episode_id": "fixture", "raw_tick": 4}
        self._scene_event_collector = SimpleNamespace(
            summary=lambda: {"raw_state_read_errors": 0},
            last_collection=self._collection,
        )
        self._scene_event_map_cache = SimpleNamespace(
            public_map={"lane_valid": np.ones((1,), dtype=np.bool_)},
            build_metadata={},
        )

    @property
    def public_map(self):
        return self._scene_event_map_cache.public_map

    @property
    def last_audit(self):
        return self._scene_event_collector.summary()

    @property
    def last_collection(self):
        return self._scene_event_collector.last_collection

    def scene_metadata(self):
        return {"protocol": "wrapper-forwarding-fixture"}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}


class _NonForwardingRewardWrapper(gym.Env):
    """Wrapper-like environment that exposes only its standard API and `.env`."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)


def test_reward_audit_wrapper_resolves_custom_attributes_without_wrapper_getattr() -> None:
    raw = _RawSceneEnv()
    reward_wrapper = _NonForwardingRewardWrapper(raw)
    assert not hasattr(reward_wrapper, "last_collection")
    wrapped = SceneEventRewardAuditWrapper(reward_wrapper)

    assert wrapped.public_map is raw.public_map
    assert wrapped.last_audit == {"raw_state_read_errors": 0}
    assert wrapped.last_collection is raw.last_collection
    assert wrapped.scene_metadata() == {"protocol": "wrapper-forwarding-fixture"}
