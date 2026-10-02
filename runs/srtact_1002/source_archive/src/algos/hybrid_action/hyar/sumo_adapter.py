"""SUMO adapter for HyAR.

Reuses the shared ``SumoStateTokenizer`` for the Dict observation and maps the
environment's ``Box(2)`` action ``[speed, lateral]`` to HyAR's mixed action
``(lane one-hot, speed)`` and back.  TODO: wire the online/offline data flow
into the SUMO environment (collection + training loop).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algos.transformer_rl.decision_transformer.sumo_adapter import SumoStateTokenizer

LANE_COMMANDS = (-1, 0, 1)


def env_action_to_hybrid(action: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map ``[speed, lateral]`` to ``(lane one-hot, speed)``."""
    action = np.asarray(action, dtype=np.float32)
    speed = action[0]
    lateral = float(action[1])
    if lateral < -1.0 / 3.0:
        lane = 0
    elif lateral > 1.0 / 3.0:
        lane = 2
    else:
        lane = 1
    discrete = np.eye(3, dtype=np.float32)[lane]
    continuous = np.asarray([speed], dtype=np.float32)
    return discrete, continuous


def hybrid_to_env_action(discrete_index: int, continuous: Any) -> np.ndarray:
    """Map HyAR's ``(lane index, speed)`` back to ``[speed, lane_code]``."""
    speed = float(np.asarray(continuous).reshape(-1)[0])
    lane_code = LANE_COMMANDS[int(discrete_index)]
    return np.asarray([speed, lane_code], dtype=np.float32)


class SumoHyARAdapter:
    """State tokenizer plus mixed-action mapping for HyAR."""

    def __init__(self, observation_space: Any) -> None:
        self.state_tokenizer = SumoStateTokenizer(observation_space)

    @property
    def state_dim(self) -> int:
        return self.state_tokenizer.state_dim
