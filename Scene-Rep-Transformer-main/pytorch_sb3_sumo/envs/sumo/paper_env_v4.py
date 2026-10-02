"""Decision-aligned v4 environment view on the frozen v2 SUMO protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from .decision_alignment_v4 import decision_alignment_context
from .paper_env_v2 import PaperSumoSceneEnvV2


class PaperSumoSceneEnvV4(PaperSumoSceneEnvV2):
    """Expose only the physical three-way lane-action mask to the policy.

    Route intent is deliberately absent from the observation.  It is added to
    the ``info`` payload for diagnostics, so the actor must still infer intent
    from its learned scene representation rather than receiving the label.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        observation_spaces = dict(self.observation_space.spaces)
        observation_spaces["lane_action_mask"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(observation_spaces)

    def _make_observation(self) -> dict[str, np.ndarray]:
        observation = super()._make_observation()
        context = decision_alignment_context(self)
        observation["lane_action_mask"] = np.array(
            context.lane_action_mask, copy=True, dtype=np.float32
        )
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        pre_action = decision_alignment_context(self)
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(
            {
                "pre_action_lane_action_mask": pre_action.lane_action_mask.tolist(),
                "pre_action_route_intent": int(pre_action.route_intent),
                "pre_action_route_intent_valid": bool(pre_action.route_intent_valid),
                "pre_action_route_intent_reason": pre_action.reason,
                "pre_action_non_keep_feasible": bool(pre_action.non_keep_feasible),
                "pre_action_current_edge": pre_action.current_edge,
                "pre_action_current_lane_index": pre_action.current_lane_index,
                "pre_action_next_route_edge": pre_action.next_route_edge,
            }
        )
        return observation, reward, terminated, truncated, info


__all__ = ["PaperSumoSceneEnvV4"]

