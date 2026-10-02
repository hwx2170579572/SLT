"""Isolated speed-[0, 20] variant of the independent v2 comparison.

This module deliberately leaves the released environments, source ego route
files, and the independent-v2 wrappers untouched.  The new mixin changes only
the physical longitudinal control interval: the normalized policy action still
uses ``[-1, 1]``, but it maps to ``[0, 20]`` m/s.  The corresponding TraCI
target-speed clip and the runtime ego ``maxSpeed`` are raised to the same upper
bound so the requested interval reaches the simulator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .high_density_single_seed_100ep_v2 import (
    HighDensitySingleSeed100EpisodeEnvV2,
    HighDensitySingleSeed100EpisodeEnvV4V2,
    SINGLE_SEED_100EP_EXPERIMENT_ID,
)
from .sumo_env import _smarts_curve_speed_limit_from_headings


SPEED20_V3_EXPERIMENT_ID = (
    "high-density-single-seed-100ep-speed20-v3"
)
EGO_SPEED_CONTROL_INTERVAL_MPS = (0.0, 20.0)
_PARENT_OVERLAY_SUFFIX = "_ss100_v2.rou.xml"
_SPEED20_V3_OVERLAY_SUFFIX = "_ss100_s20_v3.rou.xml"


class _Speed20V3ControlAndNamespaceMixin:
    """Apply the sole physical-variable change and isolate generated files."""

    @staticmethod
    def adapt_action(action: np.ndarray) -> tuple[float, int]:
        """Map normalized longitudinal action to [0, 20] m/s.

        Lateral thresholds and lane-command semantics are byte-for-byte
        equivalent to the parent control contract.
        """

        speed = float(np.clip((float(action[0]) + 1.0) * 10.0, 0.0, 20.0))
        lateral = float(np.clip(action[1], -1.0, 1.0))
        if lateral < -1.0 / 3.0:
            lane = -1
        elif lateral > 1.0 / 3.0:
            lane = 1
        else:
            lane = 0
        return speed, lane

    def _apply_speed_control(self, requested_speed: float) -> float:
        """Apply the parent control law with only its speed clip set to 20."""

        ego_id = self.specification.ego_id
        requested = float(np.clip(requested_speed, *EGO_SPEED_CONTROL_INTERVAL_MPS))
        if self.ego_control_profile == "direct":
            effective = requested
            curve_limit = math.inf
        else:
            headings = self._control_path_headings()
            curve_limit = _smarts_curve_speed_limit_from_headings(headings)
            desired = min(requested, curve_limit)
            current = max(0.0, float(self._connection.vehicle.getSpeed(ego_id)))
            effective = float(
                np.clip(desired, current - 4.5 * 0.1, current + 2.6 * 0.1)
            )
        self._connection.vehicle.setSpeed(ego_id, effective)
        self._last_effective_target_speed = effective
        self._last_curve_speed_limit = curve_limit
        return effective

    def _configure_policy_controlled_ego(self, ego_id: str) -> None:
        """Preserve parent control flags and expose the new bound to SUMO."""

        super()._configure_policy_controlled_ego(ego_id)
        self._connection.vehicle.setMaxSpeed(
            ego_id, EGO_SPEED_CONTROL_INTERVAL_MPS[1]
        )

    def _overlay_path(self, source_path: Path) -> Path:
        inherited = super()._overlay_path(source_path)
        if not inherited.name.endswith(_PARENT_OVERLAY_SUFFIX):
            raise RuntimeError(
                "unexpected independent-v2 overlay name: "
                f"{inherited.name}"
            )
        return inherited.with_name(
            f"{inherited.name[:-len(_PARENT_OVERLAY_SUFFIX)]}"
            f"{_SPEED20_V3_OVERLAY_SUFFIX}"
        )

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "high_density_comparison_experiment_id": (
                    SPEED20_V3_EXPERIMENT_ID
                ),
                "parent_comparison_experiment_id": (
                    SINGLE_SEED_100EP_EXPERIMENT_ID
                ),
                "ego_speed_control_interval_mps": list(
                    EGO_SPEED_CONTROL_INTERVAL_MPS
                ),
                "normalized_longitudinal_action_interval": [-1.0, 1.0],
                "longitudinal_action_mapping": (
                    "clip((action[0] + 1) * 10, 0, 20)"
                ),
                "only_change_from_parent_v2": (
                    "ego_speed_control_interval_mps"
                ),
            }
        )
        return info


class HighDensitySingleSeed100EpisodeSpeed20EnvV3(
    _Speed20V3ControlAndNamespaceMixin,
    HighDensitySingleSeed100EpisodeEnvV2,
):
    """Base-adapter environment under the isolated speed20-v3 namespace."""


class HighDensitySingleSeed100EpisodeSpeed20EnvV4V3(
    _Speed20V3ControlAndNamespaceMixin,
    HighDensitySingleSeed100EpisodeEnvV4V2,
):
    """v4-adapter environment under the isolated speed20-v3 namespace."""


__all__ = [
    "EGO_SPEED_CONTROL_INTERVAL_MPS",
    "HighDensitySingleSeed100EpisodeSpeed20EnvV3",
    "HighDensitySingleSeed100EpisodeSpeed20EnvV4V3",
    "SPEED20_V3_EXPERIMENT_ID",
]
