"""Actual-k n-step assembly and observation replay for the independent route."""
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import discounted_return, readonly_observation


@dataclass(frozen=True)
class DecisionTransition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: dict[str, np.ndarray]
    terminated: bool
    truncated: bool
    raw_steps: int
    episode_id: int
    decision_id: int


@dataclass(frozen=True)
class NStepTransition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: dict[str, np.ndarray]
    terminated: bool
    truncated: bool
    discount: float
    decisions: int
    raw_steps: int
    episode_id: int
    decision_id: int


class NStepAssembler:
    def __init__(self, n_step: int = 4, gamma: float = 0.99):
        if n_step < 1 or not 0 < gamma <= 1:
            raise ValueError("Invalid n-step configuration")
        self.n_step, self.gamma = n_step, gamma
        self.pending: deque[DecisionTransition] = deque()

    def _take(self) -> NStepTransition:
        steps = list(self.pending)[:self.n_step]
        if not steps:
            raise RuntimeError("No pending transition")
        first, last = steps[0], steps[-1]
        total, discount = discounted_return([x.reward for x in steps], self.gamma)
        result = NStepTransition(first.observation, first.action, total, last.next_observation,
                                 last.terminated, last.truncated, discount, len(steps),
                                 sum(x.raw_steps for x in steps), first.episode_id, first.decision_id)
        self.pending.popleft()
        return result

    def append(self, transition: DecisionTransition) -> list[NStepTransition]:
        if transition.raw_steps < 1:
            raise ValueError("A decision must contain actual environment steps")
        if self.pending:
            last = self.pending[-1]
            if transition.episode_id != last.episode_id or transition.decision_id != last.decision_id + 1:
                raise ValueError("Cannot join n-step returns across reset/missing decisions")
        self.pending.append(transition)
        ready = []
        if transition.terminated or transition.truncated:
            while self.pending:
                ready.append(self._take())
        elif len(self.pending) >= self.n_step:
            ready.append(self._take())
        return ready

    def append_collected(self, transition: DecisionTransition, *, action_repeat: int,
                         collection_cutoff: bool) -> tuple[list[NStepTransition], bool]:
        """Return (ready transitions, omitted incomplete macro-action).

        A true terminal decision is retained even when its physics duration is
        shorter than action_repeat. Global budget fractions are kept only in
        collection logs, so a per-decision gamma is not misused as a per-tick
        discount. Existing pending transitions end at their own real next obs.
        """
        if action_repeat < 1 or not 1 <= transition.raw_steps <= action_repeat:
            raise ValueError("Invalid physical duration for a decision")
        omit = collection_cutoff and not transition.terminated and transition.raw_steps < action_repeat
        if omit:
            return self.flush_external_truncation(), True
        return self.append(transition), False

    def flush_external_truncation(self) -> list[NStepTransition]:
        """End collection using its last real next-observation; retain bootstrap."""
        if not self.pending:
            return []
        last = self.pending[-1]
        self.pending[-1] = DecisionTransition(last.observation, last.action, last.reward,
            last.next_observation, last.terminated, not last.terminated, last.raw_steps,
            last.episode_id, last.decision_id)
        ready = []
        while self.pending:
            ready.append(self._take())
        return ready


class ObservationReplay:
    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.transitions: list[NStepTransition] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: NStepTransition) -> None:
        # Arrays have been owned once per real observation before assembly;
        # consecutive transitions can safely share their immutable snapshots.
        obs = readonly_observation(transition.observation)
        nxt = readonly_observation(transition.next_observation)
        action = np.array(transition.action, dtype=np.float32, copy=True)
        action.setflags(write=False)
        stored = NStepTransition(obs, action, transition.reward, nxt, transition.terminated,
            transition.truncated, transition.discount, transition.decisions, transition.raw_steps,
            transition.episode_id, transition.decision_id)
        if len(self.transitions) < self.capacity:
            self.transitions.append(stored)
        else:
            self.transitions[self.position] = stored
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, Any]:
        if not self.transitions or batch_size < 1:
            raise ValueError("Cannot sample an empty replay or empty batch")
        selected = [self.transitions[int(i)] for i in self.rng.integers(len(self), size=batch_size)]
        keys = selected[0].observation.keys()
        if any(s.observation.keys() != keys or s.next_observation.keys() != keys for s in selected):
            raise ValueError("Replay observation schema changed")
        return {
            "observation": {k: np.stack([s.observation[k] for s in selected]) for k in keys},
            "next_observation": {k: np.stack([s.next_observation[k] for s in selected]) for k in keys},
            "action": np.stack([s.action for s in selected]),
            **{name: np.asarray([getattr(s, name) for s in selected], dtype=dtype).reshape(-1, 1)
               for name, dtype in (("reward", np.float32), ("discount", np.float32),
                                   ("terminated", bool), ("truncated", bool),
                                   ("decisions", np.int64), ("raw_steps", np.int64))},
            "identity": [(s.episode_id, s.decision_id) for s in selected],
        }
