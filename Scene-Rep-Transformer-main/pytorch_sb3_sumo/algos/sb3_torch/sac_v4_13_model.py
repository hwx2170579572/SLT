"""Gradient-isolated tempered joint-support PRCR SAC for v4.13."""

from __future__ import annotations

from typing import Any

from .hybrid_policy_v4_13_model import (
    GradientIsolatedTemperedJointSupportPolicyV413,
    GradientIsolatedTemperedMixtureActorV413,
)
from .sac_v4_12_model import AugmentedJointSupportPRCRSACV412


class GradientIsolatedTemperedJointSupportSACV413(
    AugmentedJointSupportPRCRSACV412
):
    """v4.12 learner with a gradient-isolated lane auxiliary objective."""

    scientific_version = "v4.13_gradient_isolated_tempered_joint_support_prcr"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if hasattr(self, "policy"):
            self._validate_v4_13_components()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._validate_v4_13_components()

    def _validate_v4_13_components(self) -> None:
        if not isinstance(
            self.policy, GradientIsolatedTemperedJointSupportPolicyV413
        ):
            raise TypeError("v4.13 requires its isolated policy class")
        if not isinstance(
            self.actor, GradientIsolatedTemperedMixtureActorV413
        ):
            raise TypeError("v4.13 requires its isolated actor class")


__all__ = ["GradientIsolatedTemperedJointSupportSACV413"]
