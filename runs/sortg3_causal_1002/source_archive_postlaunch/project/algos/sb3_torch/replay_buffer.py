"""N-step replay support for SB3 dictionary observations."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from torch import Tensor

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class DictNStepReplayBufferSamples(NamedTuple):
    """SAC n-step sample plus the one-step target required by SLT."""

    observations: dict[str, Tensor]
    actions: Tensor
    next_observations: dict[str, Tensor]
    dones: Tensor
    rewards: Tensor
    discounts: Tensor | None
    one_step_next_observations: dict[str, Tensor]


class DictNStepReplayBuffer(DictReplayBuffer):
    """Vectorized n-step returns for dictionary observations.

    SB3 2.9 includes :class:`NStepReplayBuffer` for array observations but
    intentionally rejects Dict spaces.  This class applies the same sampling
    algorithm to every observation key and returns SB3's native
    ``DictReplayBufferSamples`` structure.
    """

    def __init__(
        self,
        buffer_size: int,
        *args: Any,
        n_steps: int = 4,
        gamma: float = 0.99,
        duplicate_episode_end_transition: bool = False,
        source_action_repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        if n_steps <= 1:
            raise ValueError("DictNStepReplayBuffer requires n_steps > 1")
        if int(kwargs.get("n_envs", 1)) != 1:
            raise NotImplementedError(
                "The source-equivalent cpprb N-step staging currently supports one env"
            )
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.duplicate_episode_end_transition = bool(
            duplicate_episode_end_transition
        )
        self.source_action_repeat = int(source_action_repeat)
        if self.source_action_repeat <= 0:
            raise ValueError("source_action_repeat must be positive")
        self.source_incomplete_actions_skipped = 0
        self.requested_buffer_size = int(buffer_size)
        # cpprb keeps N-1 recent transitions in a separate local buffer in
        # addition to the configured global capacity. Allocate those slots in
        # the single SB3 ring, then exclude them from sampling until finalized.
        super().__init__(
            self.requested_buffer_size + self.n_steps - 1,
            *args,
            **kwargs,
        )
        if self.optimize_memory_usage:
            raise NotImplementedError("DictNStepReplayBuffer does not support optimize_memory_usage")
        self._insertion_ids = np.full(self.buffer_size, -1, dtype=np.int64)
        self._next_insertion_id = 0
        # Dense set of sampleable ring slots.  Sampling and ``size()`` must be
        # O(batch_size)/O(1): scanning a 20k buffer on every raw-step update
        # would dominate the paper-profile run.
        self._finalized_slots = np.full(
            self.requested_buffer_size, -1, dtype=np.int64
        )
        self._finalized_positions = np.full(self.buffer_size, -1, dtype=np.int64)
        self._finalized_count = 0

    def _unmark_finalized(self, slot: int) -> None:
        position = int(self._finalized_positions[slot])
        if position < 0:
            return
        last_position = self._finalized_count - 1
        last_slot = int(self._finalized_slots[last_position])
        if position != last_position:
            self._finalized_slots[position] = last_slot
            self._finalized_positions[last_slot] = position
        self._finalized_slots[last_position] = -1
        self._finalized_positions[slot] = -1
        self._finalized_count = last_position

    def _evict_oldest_finalized(self) -> None:
        active_slots = self._finalized_slots[: self._finalized_count]
        active_ids = self._insertion_ids[active_slots]
        oldest_position = int(np.argmin(active_ids))
        self._unmark_finalized(int(active_slots[oldest_position]))

    def _mark_finalized(self, slot: int) -> None:
        if int(self._finalized_positions[slot]) >= 0:
            return
        # cpprb's global buffer has exactly ``requested_buffer_size`` slots;
        # its separate N-1 transition staging tail does not increase capacity.
        if self._finalized_count == self.requested_buffer_size:
            self._evict_oldest_finalized()
        position = self._finalized_count
        self._finalized_slots[position] = slot
        self._finalized_positions[slot] = position
        self._finalized_count += 1

    def _add_one(self, *args: Any, **kwargs: Any) -> bool:
        insertion_index = self.pos
        self._unmark_finalized(insertion_index)
        super().add(*args, **kwargs)
        insertion_id = self._next_insertion_id
        self._insertion_ids[insertion_index] = insertion_id
        self._next_insertion_id += 1
        is_boundary = bool(self.dones[insertion_index, 0]) or bool(
            self.timeouts[insertion_index, 0]
        )
        if is_boundary:
            # cpprb.on_episode_end() flushes its incomplete local tail.  Walk
            # only the contiguous tail so a wrapped ring can never connect two
            # unrelated episodes or an overwritten gap.
            for offset in range(self.n_steps):
                candidate = (insertion_index - offset) % self.buffer_size
                expected_id = insertion_id - offset
                # Unused slots carry -1; never confuse that sentinel with a
                # real predecessor of the first few inserted transitions.
                if expected_id < 0 or int(self._insertion_ids[candidate]) != expected_id:
                    break
                self._mark_finalized(candidate)
        else:
            candidate = (insertion_index - (self.n_steps - 1)) % self.buffer_size
            expected_id = insertion_id - (self.n_steps - 1)
            if expected_id >= 0 and int(self._insertion_ids[candidate]) == expected_id:
                self._mark_finalized(candidate)
        return is_boundary

    def add(self, *args: Any, **kwargs: Any) -> None:
        dones = args[4] if len(args) > 4 else kwargs.get("done")
        infos = args[5] if len(args) > 5 else kwargs.get("infos")
        if self.source_action_repeat > 1 and infos:
            executed = int(infos[0].get("raw_steps_executed", self.source_action_repeat))
            boundary = bool(np.asarray(dones).reshape(-1)[0])
            if 0 < executed < self.source_action_repeat and not boundary:
                # The released off-policy loop stops at max_steps between
                # held-action boundaries. It still performs that raw-step
                # update, but never appends the incomplete transition.
                self.source_incomplete_actions_skipped += 1
                return
        is_boundary = self._add_one(*args, **kwargs)
        if self.duplicate_episode_end_transition and is_boundary:
            # With make_predictions=1, both released off-policy trainers add
            # the queue head once in the regular held-action block, then add
            # the still-present queue head again while draining the episode
            # tail.  Preserve that observable released-source behavior only
            # for the Proposed/SLT profile.
            self._add_one(*args, **kwargs)

    def size(self) -> int:
        """Return cpprb's global (finalized) size, excluding its local tail."""

        return self._finalized_count

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> DictNStepReplayBufferSamples:
        if self._finalized_count == 0:
            raise ValueError(
                "Cannot sample before an N-step sequence or episode boundary is "
                f"finalized (inserted={self._next_insertion_id}, pos={self.pos}, "
                f"n_steps={self.n_steps})"
            )
        positions = np.random.randint(0, self._finalized_count, size=batch_size)
        batch_indices = self._finalized_slots[positions]
        return self._get_samples(batch_indices, env=env)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictNStepReplayBufferSamples:
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)

        steps = np.arange(self.n_steps).reshape(1, -1)
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        rewards_sequence = self._normalize_reward(
            self.rewards[indices, env_indices[:, None]], env
        )
        dones_sequence = self.dones[indices, env_indices[:, None]]
        timeout_sequence = self.timeouts[indices, env_indices[:, None]]
        boundaries = np.logical_or(dones_sequence, timeout_sequence)
        boundary_index = boundaries.argmax(axis=1)
        has_boundary = boundaries.any(axis=1)
        boundary_index = np.where(has_boundary, boundary_index, self.n_steps - 1)
        mask = np.arange(self.n_steps).reshape(1, -1) <= boundary_index[:, None]

        # Preserve the released SAC implementation exactly: cpprb aggregates
        # the N-step reward and advances next_obs, but algos/sac.py still uses
        # one factor of self.discount for bootstrap rather than gamma**N.
        bootstrap_discounts = np.full(
            (batch_inds.shape[0], 1), self.gamma, dtype=np.float32
        )
        reward_discounts = self.gamma ** np.arange(
            self.n_steps, dtype=np.float32
        ).reshape(1, -1)
        returns = (rewards_sequence * reward_discounts * mask).sum(
            axis=1, keepdims=True
        )

        last_indices = (batch_inds + boundary_index) % self.buffer_size
        next_observations = self._normalize_obs(
            {
                key: values[last_indices, env_indices]
                for key, values in self.next_observations.items()
            },
            env,
        )
        one_step_next_observations = self._normalize_obs(
            {
                key: values[batch_inds, env_indices]
                for key, values in self.next_observations.items()
            },
            env,
        )
        observations = self._normalize_obs(
            {
                key: values[batch_inds, env_indices]
                for key, values in self.observations.items()
            },
            env,
        )
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(
            np.float32
        )
        final_dones = next_dones * (1.0 - next_timeouts)
        actions = self.actions[batch_inds, env_indices]
        assert isinstance(observations, dict)
        assert isinstance(next_observations, dict)
        assert isinstance(one_step_next_observations, dict)
        return DictNStepReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in observations.items()},
            actions=self.to_torch(actions),
            next_observations={
                key: self.to_torch(value) for key, value in next_observations.items()
            },
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(returns),
            discounts=self.to_torch(bootstrap_discounts),
            one_step_next_observations={
                key: self.to_torch(value)
                for key, value in one_step_next_observations.items()
            },
        )
