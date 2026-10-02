"""Namespaced high-density environment for the 5x6 independent-v2 extension.

All six scenarios use the audited additive-overlay mechanism.  The original
network, source traffic files, ego route, reward, observations, actions, and
episode limits remain unchanged.  Only generated overlay basenames and
provenance fields are new, so they cannot be mistaken for earlier v1/v2 runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .high_density_env_v1 import (
    HighDensityPaperSumoSceneEnvV1,
    HighDensityPaperSumoSceneEnvV4V1,
)


EXPERIMENT_ID = "independent-v2-five-methods-six-scenarios-100ep-v1"
ENVIRONMENT_VERSION = "independent-v2-5m6s100e-high-density/v1"
_SOURCE_SUFFIX = "_v1.rou.xml"
_NAMESPACED_SUFFIX = "_i5m6s100_v1.rou.xml"


class _IndependentV2FiveBySixNamespaceMixin:
    def __init__(
        self,
        *args: Any,
        high_density_contract_partition: str | None = None,
        **kwargs: Any,
    ) -> None:
        physical_partition = str(kwargs.get("high_density_partition", "all"))
        contract_partition = (
            physical_partition
            if high_density_contract_partition is None
            else str(high_density_contract_partition)
        )
        if physical_partition not in {"all", "train", "evaluation"}:
            raise ValueError("unsupported physical high-density partition")
        if contract_partition not in {"all", "train", "evaluation", "validation"}:
            raise ValueError(
                "formal/test is forbidden in the independent-v2 extension"
            )
        self._i5m6s100_physical_partition = physical_partition
        self._i5m6s100_contract_partition = contract_partition
        super().__init__(*args, **kwargs)
        # Parent selection continues to use _high_density_partition.  This
        # public provenance label may therefore use the v4 name "validation"
        # while both adapter families still read the identical physical 20%.
        self._traffic_partition = contract_partition

    def _overlay_path(self, source_path: Path) -> Path:
        inherited = super()._overlay_path(source_path)
        if not inherited.name.endswith(_SOURCE_SUFFIX):
            raise RuntimeError(
                f"unexpected inherited high-density overlay name: {inherited.name}"
            )
        return inherited.with_name(
            f"{inherited.name[:-len(_SOURCE_SUFFIX)]}{_NAMESPACED_SUFFIX}"
        )

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "comparison_experiment_id": EXPERIMENT_ID,
                "comparison_environment_version": ENVIRONMENT_VERSION,
                "comparison_training_seeds": 1,
                "comparison_evaluation_episodes": 100,
                "matched_physical_partition": (
                    self._i5m6s100_physical_partition
                ),
                "contract_partition": self._i5m6s100_contract_partition,
                "formal_test_accessed": False,
            }
        )
        return info


class IndependentV2FiveBySixEnvV1(
    _IndependentV2FiveBySixNamespaceMixin, HighDensityPaperSumoSceneEnvV1
):
    """Continuous-action baseline/full environment for the extension."""


class IndependentV2FiveBySixEnvV4V1(
    _IndependentV2FiveBySixNamespaceMixin, HighDensityPaperSumoSceneEnvV4V1
):
    """Decision-aligned v4 environment on the same physical 80/20 split."""


__all__ = [
    "ENVIRONMENT_VERSION",
    "EXPERIMENT_ID",
    "IndependentV2FiveBySixEnvV1",
    "IndependentV2FiveBySixEnvV4V1",
]
