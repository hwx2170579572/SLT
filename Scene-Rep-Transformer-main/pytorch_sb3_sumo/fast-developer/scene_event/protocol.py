"""Explicit decision-time SAC contracts shared by the new methods only."""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch

from . import PROTOCOL_ID


@dataclass(frozen=True)
class ExperimentConfig:
    protocol: str = PROTOCOL_ID
    method: str = "sac_scene_dualgraph_v1"
    scenario: str = "intersection_sorted_depart4p0"
    seed: int = 0
    raw_steps: int = 100_000
    eval_episodes: int = 100
    eval_seed_start: int = 10_000
    raw_dt: float = 0.1
    action_repeat: int = 3
    deadline_seconds: float = 60.0
    deadline_is_terminal: bool = True
    gamma: float = 0.99
    n_step: int = 4
    tau: float = 0.005
    learning_rate: float = 1e-4
    batch_size: int = 32
    learning_starts_raw: int = 5_000
    replay_capacity: int = 10_000
    updates_per_decision: int = 3
    width: int = 128
    z_dim: int = 128
    hidden_dim: int = 128
    final_hidden_dim: int = 32
    action_embedding_dim: int = 64
    initial_alpha: float = 0.2
    max_actors: int = 24
    history_samples: int = 21
    observation_radius: float = 80.0
    prediction_seconds: float = 3.0
    prediction_dt: float = 0.3
    device: str = "cuda"

    def validate(self) -> None:
        if self.protocol != PROTOCOL_ID:
            raise ValueError("Unknown protocol; do not silently mix historical targets")
        if self.method not in {"sac_scene_dualgraph_v1", "sac_scene_eventgraph_cv_v1"}:
            raise ValueError("Method is not implemented/accepted in this stage")
        if self.scenario != "intersection_sorted_depart4p0":
            raise ValueError("This first implementation protocol is scene-specific")
        if not self.deadline_is_terminal or self.deadline_seconds != 60.0:
            raise ValueError("The user confirmed a 60 s finite task deadline")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid discount/Polyak coefficient")
        for name in ("n_step", "action_repeat", "raw_steps", "batch_size", "replay_capacity", "updates_per_decision"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.eval_episodes < 0 or self.learning_starts_raw < 0:
            raise ValueError("Counts must be nonnegative")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay must accommodate one batch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def discounted_return(rewards: list[float], gamma: float) -> tuple[float, float]:
    """Reward sum and final discount for *actual* k decision transitions."""
    if not rewards:
        raise ValueError("An n-step transition must contain at least one decision")
    return float(sum((gamma ** j) * r for j, r in enumerate(rewards))), float(gamma ** len(rewards))


def soft_bellman_target(reward: torch.Tensor, discount: torch.Tensor,
                       terminated: torch.Tensor, target_q: torch.Tensor,
                       next_log_prob: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """External truncation bootstraps; a task deadline is terminated=True.

    The caller must supply the last physical observation, never a reset state.
    discount is gamma**actual_k, not a constant gamma or a raw-step discount.
    """
    if not all(t.shape == reward.shape for t in (discount, terminated, target_q, next_log_prob)):
        raise ValueError("Bellman tensors must have equal [batch, 1] shapes")
    return reward + discount * (~terminated.bool()).to(reward.dtype) * (target_q - alpha * next_log_prob)


def readonly_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Own a lossless observation snapshot, not the encoder's changing z.

    Already read-only contiguous arrays (e.g. public map cache) can be shared.
    Mutable collector buffers are copied before entering replay.
    """
    result = {}
    for key, value in observation.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Observation {key} must be a typed numpy array")
        owned = value if not value.flags.writeable and value.flags.c_contiguous and value.flags.owndata else np.array(value, copy=True, order="C")
        owned.setflags(write=False)
        result[key] = owned
    return result
