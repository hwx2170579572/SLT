"""Augmented joint replay-supported policy for the v4.12 model."""

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
from .hybrid_policy_v4_11_model import (
    ProperCalibratedRankedRiskSACPolicyV411,
)


@dataclass(frozen=True)
class JointReplaySupportTerms:
    """Differentiable support losses for the complete hybrid action."""

    lane_categorical_nll: Tensor
    conditional_speed_mixture_nll: Tensor
    joint_action_nll: Tensor


class JointReplaySupportedMixtureActorV412(SupportedMixtureHybridActor):
    """Train both the replay lane and its conditional learned speed mixture."""

    def __init__(
        self,
        *args: Any,
        lane_support_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.lane_support_scale = float(lane_support_scale)
        if self.lane_support_scale < 0.0:
            raise ValueError("lane_support_scale must be non-negative")
        self._joint_support_history: dict[str, list[float]] = {
            "lane": [],
            "speed": [],
            "joint": [],
        }
        super().__init__(*args, **kwargs)

    @staticmethod
    def replay_lane_indices(replay_actions: Tensor) -> Tensor:
        if replay_actions.ndim != 2 or replay_actions.shape[1] != 2:
            raise ValueError("replay actions must have shape [batch,2]")
        lateral = replay_actions[:, 1]
        return th.where(
            lateral < -1.0 / 3.0,
            th.zeros_like(lateral, dtype=th.long),
            th.where(
                lateral > 1.0 / 3.0,
                th.full_like(lateral, 2, dtype=th.long),
                th.ones_like(lateral, dtype=th.long),
            ),
        )

    def joint_replay_support_terms(
        self,
        batch: SupportedMixtureActionBatch,
        replay_actions: Tensor,
    ) -> JointReplaySupportTerms:
        lane_indices = self.replay_lane_indices(replay_actions)
        lane_nll = functional.nll_loss(
            batch.lane_log_probabilities,
            lane_indices,
        )
        speed_nll = super().replay_support_nll(batch, replay_actions)
        joint_nll = speed_nll + self.lane_support_scale * lane_nll
        return JointReplaySupportTerms(
            lane_categorical_nll=lane_nll,
            conditional_speed_mixture_nll=speed_nll,
            joint_action_nll=joint_nll,
        )

    def replay_support_nll(
        self,
        batch: SupportedMixtureActionBatch,
        replay_actions: Tensor,
    ) -> Tensor:
        terms = self.joint_replay_support_terms(batch, replay_actions)
        self._joint_support_history["lane"].append(
            float(terms.lane_categorical_nll.detach().cpu())
        )
        self._joint_support_history["speed"].append(
            float(terms.conditional_speed_mixture_nll.detach().cpu())
        )
        self._joint_support_history["joint"].append(
            float(terms.joint_action_nll.detach().cpu())
        )
        return terms.joint_action_nll

    def reset_joint_support_diagnostics(self) -> None:
        for values in self._joint_support_history.values():
            values.clear()

    def consume_joint_support_diagnostics(self) -> dict[str, list[float]]:
        output = {
            key: list(values)
            for key, values in self._joint_support_history.items()
        }
        self.reset_joint_support_diagnostics()
        return output


class AugmentedJointSupportPRCRPolicyV412(
    ProperCalibratedRankedRiskSACPolicyV411
):
    """Exact learned argmax with a continuous joint lane/component prior."""

    scientific_version = "v4.12_augmented_joint_support_prcr"
    deterministic_lane_decoder = (
        "argmax_feasible_learned_joint_support_risk_uncertainty_score"
    )
    joint_replay_action_support = True
    inference_safety_rule_added = False
    external_kinematic_projection = False
    action_postprocessing_override = False

    def __init__(
        self,
        *args: Any,
        lane_prior_coef: float = 0.05,
        lane_support_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.lane_prior_coef = float(lane_prior_coef)
        self.lane_support_scale = float(lane_support_scale)
        if self.lane_prior_coef < 0.0:
            raise ValueError("lane_prior_coef must be non-negative")
        if self.lane_support_scale < 0.0:
            raise ValueError("lane_support_scale must be non-negative")
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        shared_online = self.critic.features_extractor
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, shared_online
        )
        self.actor = JointReplaySupportedMixtureActorV412(
            **actor_kwargs,
            speed_components=self.speed_components,
            lane_support_scale=self.lane_support_scale,
        ).to(self.device)
        self.actor.latent_pi.apply(keras_initialize_linear)
        encoder_ids = {
            id(parameter) for parameter in shared_online.parameters()
        }
        actor_parameters = self._parameters_excluding(self.actor, encoder_ids)
        self.actor.optimizer = self.optimizer_class(
            actor_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        ownership = self.optimizer_parameter_ownership()
        if ownership["overlap_count"] != 0:
            raise RuntimeError("v4.12 optimizer parameter ownership overlaps")

    def _risk_adjusted_proposal_values(
        self,
        observation: PyTorchObs,
        batch: SupportedMixtureActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        values = super()._risk_adjusted_proposal_values(observation, batch)
        score = values[-1] + self.lane_prior_coef * (
            batch.lane_log_probabilities.unsqueeze(-1)
        )
        return (*values[:-1], score)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tensor:
        start = len(self.target_decoder_records)
        actions = super()._predict(observation, deterministic=deterministic)
        if deterministic and self._record_target_decoder:
            for record in self.target_decoder_records[start:]:
                record.update(
                    {
                        "lane_prior_coef": self.lane_prior_coef,
                        "lane_support_scale": self.lane_support_scale,
                        "joint_lane_component_support_score": True,
                        "actor_confidence_threshold_present": False,
                    }
                )
        return actions


__all__ = [
    "AugmentedJointSupportPRCRPolicyV412",
    "JointReplaySupportTerms",
    "JointReplaySupportedMixtureActorV412",
    "LANE_COMMANDS",
    "SupportedMixtureActionBatch",
]

