"""Compatibility adapter for the original four-return-value runners."""

from __future__ import annotations

from typing import Any

import numpy as np

from .sumo_env import SumoSceneEnv


class LegacySumoAdapter:
    """Expose ``(trajectory, ego, map)`` and the legacy Gym step API.

    This adapter is useful for parity diagnostics against the existing TF2RL
    runners.  The SB3 entry points use :class:`SumoSceneEnv` directly.
    """

    def __init__(self, env: SumoSceneEnv | None = None, **env_kwargs: Any) -> None:
        self.env = env if env is not None else SumoSceneEnv(**env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space["trajectory"]
        self._max_episode_steps = self.env.max_episode_steps

    @staticmethod
    def _convert(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trajectory = observation["trajectory"]
        ego_history = trajectory[0]
        valid = np.flatnonzero(np.any(ego_history != 0.0, axis=-1))
        ego = ego_history[valid[-1]].copy() if len(valid) else np.zeros(5, dtype=np.float32)
        return trajectory, ego, observation["map"]

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation, _ = self.env.reset(**kwargs)
        return self._convert(observation)

    def step(
        self, action: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, bool, tuple[bool, bool, bool, bool]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._convert(observation), reward, terminated or truncated, info["legacy_info"]

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
