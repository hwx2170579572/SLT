"""SUMO adapter for the Decision Transformer baseline.

Bridges the Dict observation (trajectory/map) and ``Box(2)`` action of the SUMO
environment to the flat state/action vectors consumed by the DT sequence model.
The adapter is inference/tokenization only; it contains no learned parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import DecisionTransformer


class SumoStateTokenizer:
    """Flatten a Dict observation into a fixed-size state vector.

    The state vector is ``[ego history | neighbor current frames | ego routes]``
    in row-major order.  Its dimensionality depends only on the observation
    space, not on any learned parameters; it differs across scenarios only when
    ``map_dim`` does (CARLA uses 2-D map points, cross/cross_left use 5-D).
    """

    def __init__(self, observation_space: Any) -> None:
        trajectory_space = observation_space.spaces["trajectory"]
        map_space = observation_space.spaces["map"]
        self.actor_count = int(trajectory_space.shape[0])
        self.history_steps = int(trajectory_space.shape[1])
        self.raw_state_dim = int(trajectory_space.shape[2])
        self.path_count = int(map_space.shape[0])
        self.path_length = int(map_space.shape[1])
        self.map_dim = int(map_space.shape[2])
        if self.path_count % self.actor_count:
            raise ValueError("Map path count must be divisible by actor count")
        self.paths_per_actor = self.path_count // self.actor_count
        self.vector_dim = (
            self.history_steps * self.raw_state_dim
            + (self.actor_count - 1) * self.raw_state_dim
            + self.paths_per_actor * self.path_length * self.map_dim
        )

    @property
    def state_dim(self) -> int:
        """Flattened state-vector size consumed by the DT sequence model."""
        return self.vector_dim

    def __call__(self, observation: Any) -> np.ndarray:
        trajectory = np.asarray(observation["trajectory"], dtype=np.float32)
        map_state = np.asarray(observation["map"], dtype=np.float32)
        ego_history = trajectory[0].reshape(-1)
        neighbor_current = trajectory[1:, -1].reshape(-1)
        ego_routes = map_state[: self.paths_per_actor].reshape(-1)
        return np.concatenate([ego_history, neighbor_current, ego_routes]).astype(
            np.float32
        )


def make_decision_transformer(
    state_dim: int,
    act_dim: int,
    *,
    hidden_size: int = 128,
    max_length: int | None = 20,
    max_ep_len: int = 4096,
    action_tanh: bool = True,
    n_layer: int = 3,
    n_head: int = 1,
    **kwargs: Any,
) -> DecisionTransformer:
    """Construct a DecisionTransformer with a flat state/action contract."""

    return DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        action_tanh=action_tanh,
        n_layer=n_layer,
        n_head=n_head,
        **kwargs,
    )
