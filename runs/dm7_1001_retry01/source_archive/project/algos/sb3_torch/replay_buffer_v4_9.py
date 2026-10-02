"""Collision-labelled horizon-correct replay for the v4.9 model."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from torch import Tensor

from .replay_buffer_v4_7 import HorizonCorrectDictNStepReplayBufferV47


class CollisionAwareReplayBufferSamplesV49(NamedTuple):
    observations: dict[str, Tensor]
    actions: Tensor
    next_observations: dict[str, Tensor]
    dones: Tensor
    rewards: Tensor
    discounts: Tensor | None
    one_step_next_observations: dict[str, Tensor]
    collision_costs: Tensor


class CollisionAwareHorizonReplayBufferV49(
    HorizonCorrectDictNStepReplayBufferV47
):
    """Expose discounted collision labels without changing reward returns.

    Labels come from the environment's observed ``info["collision"]`` terminal
    event.  They are stored alongside each replay transition, including the
    source-equivalent duplicated episode-end transition.  This avoids
    confusing reward normalization or a simultaneous terminal event with the
    collision definition.  The collision target uses the same N-step boundary
    mask and gamma powers as the v4.7/v4.8 return estimator.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.collision_label_source = "observed_environment_info_collision"
        self.collision_events = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.stored_transition_count = 0
        self.stored_collision_event_count = 0
        self.sampled_collision_label_count = 0
        self.sampled_positive_collision_label_count = 0
        self.sampled_collision_cost_sum = 0.0

    def _add_one(self, *args: Any, **kwargs: Any) -> bool:
        infos = args[5] if len(args) > 5 else kwargs.get("infos")
        if not infos or len(infos) != self.n_envs:
            raise ValueError(
                "v4.9 collision supervision requires one info payload per environment"
            )
        missing = [index for index, info in enumerate(infos) if "collision" not in info]
        if missing:
            raise ValueError(
                "v4.9 collision supervision requires info['collision']; "
                f"missing for environment indices {missing}"
            )
        insertion_index = self.pos
        collision_values = np.asarray(
            [bool(info["collision"]) for info in infos], dtype=np.float32
        )
        is_boundary = super()._add_one(*args, **kwargs)
        self.collision_events[insertion_index] = collision_values
        self.stored_transition_count += self.n_envs
        self.stored_collision_event_count += int(collision_values.sum())
        return is_boundary

    def _collision_returns(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None,
    ) -> np.ndarray:
        if self.n_envs != 1:
            raise NotImplementedError("v4.9 collision labels require one environment")
        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        horizons = self._actual_horizons(batch_inds)
        mask = steps < horizons[:, None]
        discounts = self.gamma ** np.arange(
            self.n_steps, dtype=np.float32
        ).reshape(1, -1)
        collision_indicators = self.collision_events[indices, 0]
        return (collision_indicators * discounts * mask).sum(
            axis=1, keepdims=True
        ).astype(np.float32)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> CollisionAwareReplayBufferSamplesV49:
        samples = super()._get_samples(batch_inds, env=env)
        costs = self._collision_returns(batch_inds, env)
        self.sampled_collision_label_count += int(costs.size)
        self.sampled_positive_collision_label_count += int((costs > 0.0).sum())
        self.sampled_collision_cost_sum += float(costs.sum())
        return CollisionAwareReplayBufferSamplesV49(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            discounts=samples.discounts,
            one_step_next_observations=samples.one_step_next_observations,
            collision_costs=self.to_torch(costs),
        )

    def collision_label_diagnostics(self) -> dict[str, Any]:
        count = self.sampled_collision_label_count
        return {
            "schema_version": "topo-scene-v4.9.collision-label-diagnostics/v1",
            "source": self.collision_label_source,
            "future_or_oracle_labels_used": False,
            "reward_return_changed": False,
            "n_step": self.n_steps,
            "gamma": self.gamma,
            "stored_transition_count_including_source_duplicates": (
                self.stored_transition_count
            ),
            "stored_collision_event_count_including_source_duplicates": (
                self.stored_collision_event_count
            ),
            "sampled_label_count": count,
            "sampled_positive_label_rate": (
                self.sampled_positive_collision_label_count / count
                if count
                else None
            ),
            "sampled_mean_discounted_collision_cost": (
                self.sampled_collision_cost_sum / count if count else None
            ),
        }


__all__ = [
    "CollisionAwareHorizonReplayBufferV49",
    "CollisionAwareReplayBufferSamplesV49",
]
