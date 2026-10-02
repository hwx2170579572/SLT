"""Gradient-isolated tempered joint-support policy for v4.13."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import Tensor
from torch.nn import functional

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .features import keras_initialize_linear
from .hybrid_policy_v4_10_model import (
    SupportedMixtureActionBatch,
    SupportedMixtureHybridActor,
)
from .hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
    JointReplaySupportedMixtureActorV412,
    JointReplaySupportTerms,
)


@dataclass(frozen=True)
class GradientIsolatedSupportActionBatch(SupportedMixtureActionBatch):
    """Action batch with auxiliary lane logits from a detached actor latent."""

    support_lane_log_probabilities: Tensor


class GradientIsolatedTemperedMixtureActorV413(
    JointReplaySupportedMixtureActorV412
):
    """Calibrate the deployed lane head without updating the speed trunk."""

    def __init__(
        self,
        *args: Any,
        lane_support_scale: float = 0.25,
        isolate_lane_support_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        self.isolate_lane_support_gradient = bool(
            isolate_lane_support_gradient
        )
        super().__init__(
            *args,
            lane_support_scale=lane_support_scale,
            **kwargs,
        )

    def all_action_proposals(
        self, obs: PyTorchObs, *, deterministic_speed: bool = False
    ) -> GradientIsolatedSupportActionBatch:
        (
            lane_probabilities,
            lane_log_probabilities,
            component_probabilities,
            component_log_probabilities,
            speed_means,
            speed_log_stds,
            action_mask,
            latent,
        ) = self.proposal_distribution_parameters(obs)
        speeds, speed_log_probabilities = self._sample_squashed_speeds(
            speed_means,
            speed_log_stds,
            deterministic=deterministic_speed,
        )
        lane_codes = self.lane_codes.to(speeds).view(1, -1, 1).expand_as(
            speeds
        )
        actions = th.stack([speeds, lane_codes], dim=-1)
        joint_log_probabilities = (
            lane_log_probabilities.unsqueeze(-1)
            + component_log_probabilities
            + speed_log_probabilities
        )
        if self.isolate_lane_support_gradient:
            support_logits = self.lane_logits(latent.detach())
            support_logits = support_logits.masked_fill(~action_mask, -1e9)
            support_lane_log_probabilities = th.log_softmax(
                support_logits, dim=1
            )
        else:
            support_lane_log_probabilities = lane_log_probabilities
        return GradientIsolatedSupportActionBatch(
            actions=actions,
            lane_probabilities=lane_probabilities,
            lane_log_probabilities=lane_log_probabilities,
            component_probabilities=component_probabilities,
            component_log_probabilities=component_log_probabilities,
            speed_log_probabilities=speed_log_probabilities,
            joint_log_probabilities=joint_log_probabilities,
            action_mask=action_mask,
            speed_means=speed_means,
            speed_log_stds=speed_log_stds,
            support_lane_log_probabilities=support_lane_log_probabilities,
        )

    def joint_replay_support_terms(
        self,
        batch: SupportedMixtureActionBatch,
        replay_actions: Tensor,
    ) -> JointReplaySupportTerms:
        if not isinstance(batch, GradientIsolatedSupportActionBatch):
            raise TypeError("v4.13 requires its gradient-isolated action batch")
        lane_indices = self.replay_lane_indices(replay_actions)
        lane_nll = functional.nll_loss(
            batch.support_lane_log_probabilities,
            lane_indices,
        )
        # Call the speed-only support objective explicitly.  Using the named
        # base makes the scientific boundary auditable: the v4.13 override
        # changes only the lane term and leaves the conditional speed mixture
        # likelihood byte-for-byte on the v4.10/v4.12 path.
        speed_nll = SupportedMixtureHybridActor.replay_support_nll(
            self, batch, replay_actions
        )
        joint_nll = speed_nll + self.lane_support_scale * lane_nll
        return JointReplaySupportTerms(
            lane_categorical_nll=lane_nll,
            conditional_speed_mixture_nll=speed_nll,
            joint_action_nll=joint_nll,
        )


class GradientIsolatedTemperedJointSupportPolicyV413(
    AugmentedJointSupportPRCRPolicyV412
):
    """v4.12 inference with an isolated train-time lane-support path."""

    scientific_version = "v4.13_gradient_isolated_tempered_joint_support_prcr"
    lane_support_gradient_isolated = True
    inference_safety_rule_added = False
    external_kinematic_projection = False
    action_postprocessing_override = False

    def __init__(
        self,
        *args: Any,
        lane_prior_coef: float = 0.05,
        lane_support_scale: float = 0.25,
        isolate_lane_support_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        self.isolate_lane_support_gradient = bool(
            isolate_lane_support_gradient
        )
        super().__init__(
            *args,
            lane_prior_coef=lane_prior_coef,
            lane_support_scale=lane_support_scale,
            **kwargs,
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        shared_online = self.critic.features_extractor
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, shared_online
        )
        self.actor = GradientIsolatedTemperedMixtureActorV413(
            **actor_kwargs,
            speed_components=self.speed_components,
            lane_support_scale=self.lane_support_scale,
            isolate_lane_support_gradient=(
                self.isolate_lane_support_gradient
            ),
        ).to(self.device)
        self.actor.latent_pi.apply(keras_initialize_linear)
        encoder_ids = {id(parameter) for parameter in shared_online.parameters()}
        actor_parameters = self._parameters_excluding(self.actor, encoder_ids)
        self.actor.optimizer = self.optimizer_class(
            actor_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        ownership = self.optimizer_parameter_ownership()
        if ownership["overlap_count"] != 0:
            raise RuntimeError("v4.13 optimizer parameter ownership overlaps")

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tensor:
        start = len(self.target_decoder_records)
        actions = super()._predict(observation, deterministic=deterministic)
        if deterministic and self._record_target_decoder:
            for record in self.target_decoder_records[start:]:
                record.update(
                    {
                        "lane_support_gradient_isolated": (
                            self.isolate_lane_support_gradient
                        ),
                        "lane_support_gradient_target": (
                            "deployed_lane_head_only"
                            if self.isolate_lane_support_gradient
                            else "deployed_lane_head_and_actor_latent_trunk"
                        ),
                    }
                )
        return actions


__all__ = [
    "GradientIsolatedSupportActionBatch",
    "GradientIsolatedTemperedJointSupportPolicyV413",
    "GradientIsolatedTemperedMixtureActorV413",
    "JointReplaySupportTerms",
    "LANE_COMMANDS",
]
