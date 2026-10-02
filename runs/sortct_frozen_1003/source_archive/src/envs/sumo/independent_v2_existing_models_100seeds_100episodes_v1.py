"""Artifact-isolated environment names for the independent-v2 model audit.

The physical task is intentionally identical to
``high_density_single_seed_100ep_v2``.  This wrapper changes only the
experiment identifier and generated overlay basename so the 100 test-seed by
100 episode evaluation cannot overwrite, or be mistaken for, the original v2
training/evaluation artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .high_density_single_seed_100ep_v2 import (
    HighDensitySingleSeed100EpisodeEnvV2,
    HighDensitySingleSeed100EpisodeEnvV4V2,
)


INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID = (
    "independent-v2-existing-models-100seeds-100episodes-v1"
)
_PARENT_SUFFIX = "_ss100_v2.rou.xml"
_EVALUATION_SUFFIX = "_iv2e100x100_v1.rou.xml"


class _IndependentV2ExistingModelsEvaluationNamespaceMixin:
    def set_logical_test_seed_block(
        self,
        *,
        logical_index: int,
        profile_index_start: int,
        episodes_per_test_seed: int,
    ) -> dict[str, int]:
        """Position the inherited deterministic traffic cycle for one block.

        ``PaperSumoSceneEnv`` advances its source-route cycle once per reset.
        Positioning that counter from the logical block index makes a resumed
        block use the same ordered traffic variants as an uninterrupted run.
        """

        index = int(logical_index)
        start = int(profile_index_start)
        episodes = int(episodes_per_test_seed)
        if index < start:
            raise ValueError("logical_index precedes profile_index_start")
        if episodes <= 0:
            raise ValueError("episodes_per_test_seed must be positive")
        source_traffic_episode_start = (index - start) * episodes
        self._traffic_episode_index = source_traffic_episode_start
        # The inherited roll is deterministic (Random(42)); recomputing it at
        # every block boundary removes any dependence on earlier process state.
        self._traffic_roll = None
        self._independent_v2_logical_test_seed_index = index
        self._independent_v2_source_traffic_episode_start = (
            source_traffic_episode_start
        )
        return {
            "logical_test_seed_index": index,
            "source_traffic_episode_start": source_traffic_episode_start,
            "episodes_per_test_seed": episodes,
        }

    def _overlay_path(self, source_path: Path) -> Path:
        inherited = super()._overlay_path(source_path)
        if not inherited.name.endswith(_PARENT_SUFFIX):
            raise RuntimeError(
                "unexpected independent-v2 overlay name: " f"{inherited.name}"
            )
        return inherited.with_name(
            f"{inherited.name[:-len(_PARENT_SUFFIX)]}{_EVALUATION_SUFFIX}"
        )

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "high_density_comparison_experiment_id": (
                    INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID
                ),
                "independent_v2_existing_models_reused": True,
                "independent_v2_test_seed_blocks": 100,
                "independent_v2_episodes_per_test_seed": 100,
                "independent_v2_logical_test_seed_index": getattr(
                    self, "_independent_v2_logical_test_seed_index", None
                ),
                "independent_v2_source_traffic_episode_start": getattr(
                    self,
                    "_independent_v2_source_traffic_episode_start",
                    None,
                ),
            }
        )
        return info


class IndependentV2ExistingModelsEvaluationEnvV1(
    _IndependentV2ExistingModelsEvaluationNamespaceMixin,
    HighDensitySingleSeed100EpisodeEnvV2,
):
    """Baseline-method environment under the new evaluation namespace."""


class IndependentV2ExistingModelsEvaluationEnvV4V1(
    _IndependentV2ExistingModelsEvaluationNamespaceMixin,
    HighDensitySingleSeed100EpisodeEnvV4V2,
):
    """v4-method environment under the new evaluation namespace."""


__all__ = [
    "INDEPENDENT_V2_EXISTING_MODELS_EVALUATION_ID",
    "IndependentV2ExistingModelsEvaluationEnvV1",
    "IndependentV2ExistingModelsEvaluationEnvV4V1",
]
