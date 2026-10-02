"""Augmented joint-support PRCR SAC for the v4.12 model iteration."""

from __future__ import annotations

import statistics
from typing import Any

from .hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
    JointReplaySupportedMixtureActorV412,
)
from .sac_v4_11_model import ProperCalibratedRankedRiskSACV411


class AugmentedJointSupportPRCRSACV412(
    ProperCalibratedRankedRiskSACV411
):
    """v4.11 learner with complete hybrid replay-action support."""

    scientific_version = "v4.12_augmented_joint_support_prcr"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Stable-Baselines3 constructs an uninitialised shell with
        # ``_init_setup_model=False`` before restoring a checkpoint.  In that
        # phase ``policy`` and ``actor`` intentionally do not exist yet; the
        # invariant is checked by ``_setup_model`` as soon as they do.
        if hasattr(self, "policy"):
            self._validate_v4_12_components()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._validate_v4_12_components()

    def _validate_v4_12_components(self) -> None:
        if not isinstance(self.policy, AugmentedJointSupportPRCRPolicyV412):
            raise TypeError("v4.12 requires its isolated policy class")
        if not isinstance(self.actor, JointReplaySupportedMixtureActorV412):
            raise TypeError("v4.12 requires its joint-support actor")

    def _accumulate_training_stat(self, name: str, value: float) -> None:
        # v4.11's internal variable is named replay_support_nll.  In v4.12 it
        # contains the complete joint NLL, so persist it under an exact name.
        if name == "support/replay_speed_mixture_nll":
            name = "support/replay_joint_action_nll"
        super()._accumulate_training_stat(name, value)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        actor = self.actor
        if not isinstance(actor, JointReplaySupportedMixtureActorV412):
            raise TypeError("v4.12 learner lost its joint-support actor")
        actor.reset_joint_support_diagnostics()
        super().train(gradient_steps, batch_size=batch_size)
        values = actor.consume_joint_support_diagnostics()
        names = {
            "lane": "support/replay_lane_categorical_nll",
            "speed": "support/replay_conditional_speed_mixture_nll",
            "joint": "support/replay_joint_action_nll_recomputed",
        }
        for key, target in names.items():
            if not values[key]:
                continue
            mean = float(statistics.fmean(values[key]))
            self.logger.record(target, mean)
            super()._accumulate_training_stat(target, mean)


__all__ = ["AugmentedJointSupportPRCRSACV412"]
