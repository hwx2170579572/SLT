"""Distinct environment names for the single-seed, 100-episode study.

The traffic construction and frozen map/task contract are intentionally reused
from ``high_density_env_v1``.  Only the experiment namespace and generated
overlay basename change, so these artifacts cannot be confused with the
three-seed v1 comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .high_density_env_v1 import (
    HighDensityPaperSumoSceneEnvV1,
    HighDensityPaperSumoSceneEnvV4V1,
)


SINGLE_SEED_100EP_EXPERIMENT_ID = "high-density-single-seed-100ep-v2"
_SOURCE_SUFFIX = "_v1.rou.xml"
_V2_SUFFIX = "_ss100_v2.rou.xml"


class _SingleSeed100EpisodeNamespaceMixin:
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
        if contract_partition not in {
            "all",
            "train",
            "evaluation",
            "validation",
            "test",
        }:
            raise ValueError(
                "high_density_contract_partition must be a supported split label"
            )
        self._high_density_contract_partition = contract_partition
        super().__init__(*args, **kwargs)
        # The inherited high-density environment uses ``train/evaluation`` to
        # select one matched 80/20 route partition.  v4 trainers call that same
        # held-out 20% split ``validation`` in their provenance contract.
        self._traffic_partition = contract_partition

    def _overlay_path(self, source_path: Path) -> Path:
        inherited = super()._overlay_path(source_path)
        if not inherited.name.endswith(_SOURCE_SUFFIX):
            raise RuntimeError(
                f"unexpected inherited high-density overlay name: {inherited.name}"
            )
        return inherited.with_name(
            f"{inherited.name[:-len(_SOURCE_SUFFIX)]}{_V2_SUFFIX}"
        )

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "high_density_comparison_experiment_id": (
                    SINGLE_SEED_100EP_EXPERIMENT_ID
                ),
                "high_density_comparison_training_seeds": 1,
                "high_density_comparison_evaluation_episodes": 100,
                "high_density_contract_partition": (
                    self._high_density_contract_partition
                ),
            }
        )
        return info


class HighDensitySingleSeed100EpisodeEnvV2(
    _SingleSeed100EpisodeNamespaceMixin, HighDensityPaperSumoSceneEnvV1
):
    """Baseline/full-method environment under the v2 artifact namespace."""


class HighDensitySingleSeed100EpisodeEnvV4V2(
    _SingleSeed100EpisodeNamespaceMixin, HighDensityPaperSumoSceneEnvV4V1
):
    """Decision-aligned environment under the v2 artifact namespace."""


__all__ = [
    "HighDensitySingleSeed100EpisodeEnvV2",
    "HighDensitySingleSeed100EpisodeEnvV4V2",
    "SINGLE_SEED_100EP_EXPERIMENT_ID",
]
