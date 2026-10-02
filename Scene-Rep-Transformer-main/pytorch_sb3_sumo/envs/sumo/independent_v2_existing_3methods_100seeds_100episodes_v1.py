"""Artifact-isolated environment names for the three-method v2 evaluation.

The physical maps, routes, demand, observations, actions, rewards, and episode
limits remain identical to independent v2.  Only the experiment identifier
and generated overlay basename differ from both the parent v2 study and the
superseded six-method evaluation plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .independent_v2_existing_models_100seeds_100episodes_v1 import (
    IndependentV2ExistingModelsEvaluationEnvV1,
    IndependentV2ExistingModelsEvaluationEnvV4V1,
)


INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID = (
    "independent-v2-existing-3methods-100seeds-100episodes-v1"
)
_SIX_METHOD_SUFFIX = "_iv2e100x100_v1.rou.xml"
_THREE_METHOD_SUFFIX = "_iv2e3m100x100_v1.rou.xml"


class _IndependentV2ExistingThreeMethodNamespaceMixin:
    def _overlay_path(self, source_path: Path) -> Path:
        inherited = super()._overlay_path(source_path)
        if not inherited.name.endswith(_SIX_METHOD_SUFFIX):
            raise RuntimeError(
                "unexpected six-method overlay name: " f"{inherited.name}"
            )
        return inherited.with_name(
            f"{inherited.name[:-len(_SIX_METHOD_SUFFIX)]}{_THREE_METHOD_SUFFIX}"
        )

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "high_density_comparison_experiment_id": (
                    INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID
                ),
                "independent_v2_existing_method_count": 3,
                "independent_v2_existing_method_order": (
                    "mst_slt",
                    "temporal_graph",
                    "v4_8",
                ),
            }
        )
        return info


class IndependentV2ExistingThreeMethodEvaluationEnvV1(
    _IndependentV2ExistingThreeMethodNamespaceMixin,
    IndependentV2ExistingModelsEvaluationEnvV1,
):
    """Baseline-method environment under the three-method namespace."""


class IndependentV2ExistingThreeMethodEvaluationEnvV4V1(
    _IndependentV2ExistingThreeMethodNamespaceMixin,
    IndependentV2ExistingModelsEvaluationEnvV4V1,
):
    """v4.8 environment under the three-method namespace."""


__all__ = [
    "INDEPENDENT_V2_EXISTING_THREE_METHOD_EVALUATION_ID",
    "IndependentV2ExistingThreeMethodEvaluationEnvV1",
    "IndependentV2ExistingThreeMethodEvaluationEnvV4V1",
]
