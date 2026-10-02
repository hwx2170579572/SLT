"""Horizon-correct dictionary N-step replay for the v4.7 iteration."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from .replay_buffer import DictNStepReplayBuffer, DictNStepReplayBufferSamples


class HorizonCorrectDictNStepReplayBufferV47(DictNStepReplayBuffer):
    """Use ``gamma ** actual_horizon`` for every N-step bootstrap target.

    Storage, source-compatible episode-end duplication, finalized-slot
    bookkeeping, reward accumulation, timeout handling, and the one-step SLT
    target are inherited unchanged.  The parent buffer deliberately emits one
    factor of gamma to reproduce the released implementation; v4.7 changes
    only that return-estimator contract.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon_correct_bootstrap = True
        self.sampled_horizon_count = 0
        self.sampled_horizon_sum = 0
        self.sampled_short_horizon_count = 0
        self.last_sample_actual_horizons = np.empty(0, dtype=np.int64)

    def _actual_horizons(self, batch_inds: np.ndarray) -> np.ndarray:
        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        # The project freezes one vectorized environment for source parity.
        dones = self.dones[indices, 0]
        timeouts = self.timeouts[indices, 0]
        boundaries = np.logical_or(dones, timeouts)
        first_boundary = boundaries.argmax(axis=1)
        has_boundary = boundaries.any(axis=1)
        return np.where(has_boundary, first_boundary + 1, self.n_steps).astype(
            np.int64
        )

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictNStepReplayBufferSamples:
        samples = super()._get_samples(batch_inds, env=env)
        horizons = self._actual_horizons(batch_inds)
        discounts = np.power(self.gamma, horizons, dtype=np.float64).astype(
            np.float32
        )[:, None]
        self.last_sample_actual_horizons = horizons.copy()
        self.sampled_horizon_count += int(horizons.size)
        self.sampled_horizon_sum += int(horizons.sum())
        self.sampled_short_horizon_count += int((horizons < self.n_steps).sum())
        return samples._replace(discounts=self.to_torch(discounts))

    def return_estimator_diagnostics(self) -> dict[str, Any]:
        count = self.sampled_horizon_count
        return {
            "schema_version": "topo-scene-v4.7.return-estimator-diagnostics/v1",
            "n_step": self.n_steps,
            "gamma": self.gamma,
            "horizon_correct_bootstrap": True,
            "bootstrap_discount": "gamma_power_actual_horizon",
            "reward_accumulation": "sum_gamma_power_i_reward",
            "sampled_horizon_count": count,
            "mean_sampled_actual_horizon": (
                self.sampled_horizon_sum / count if count else None
            ),
            "short_horizon_sample_rate": (
                self.sampled_short_horizon_count / count if count else None
            ),
            "last_sample_actual_horizons": self.last_sample_actual_horizons.tolist(),
            "source_timeout_semantics_changed": False,
            "one_step_next_observation_changed": False,
            "episode_end_transition_duplication_changed": False,
        }


__all__ = ["HorizonCorrectDictNStepReplayBufferV47"]
