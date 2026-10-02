"""Learned collision-value model for the v4.9 SAC candidate."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .hybrid_policy_v4 import (
    DecisionAlignedHybridActor,
    HybridActionBatch,
    HybridActionEmbeddedCritic,
)
from .hybrid_policy_v4_5 import (
    ConfidentActorFusionSACPolicyV45,
    select_fusion_lane_indices,
)


class CollisionConstrainedFusionSACPolicyV49(
    ConfidentActorFusionSACPolicyV45
):
    """Add independent online/target twin collision critics to the model.

    No deterministic safety rule is introduced.  Deployment still uses the
    inherited target/fusion decoder, while the actor is optimized against the
    learned collision value during training.
    """

    learned_collision_critic = True
    collision_risk_coef = 1.0
    deterministic_lane_decoder = (
        "actor_confident_non_keep_else_learned_risk_adjusted_critic"
    )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        collision_kwargs = self._update_features_extractor(
            self.critic_kwargs, None
        )
        collision_kwargs["share_features_extractor"] = False
        self.collision_critic = HybridActionEmbeddedCritic(
            **collision_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)

        target_kwargs = self._update_features_extractor(self.critic_kwargs, None)
        target_kwargs["share_features_extractor"] = False
        self.collision_critic_target = HybridActionEmbeddedCritic(
            **target_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)
        self.collision_critic_target.load_state_dict(
            self.collision_critic.state_dict()
        )
        self.collision_critic.optimizer = self.optimizer_class(
            self.collision_critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.collision_critic_target.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        if hasattr(self, "collision_critic"):
            self.collision_critic.set_training_mode(mode)
        if hasattr(self, "collision_critic_target"):
            self.collision_critic_target.set_training_mode(False)

    def _risk_adjusted_target_values(
        self,
        observation: PyTorchObs,
        batch: HybridActionBatch,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        reward_target = self.critic_target
        reward_features = reward_target.extract_features(
            observation, reward_target.features_extractor
        )
        reward_heads = reward_target.all_q_from_features(
            reward_features, batch.actions
        )
        minimum_reward_q = th.stack(reward_heads, dim=-1).min(
            dim=-1
        ).values.squeeze(-1)

        collision_target = self.collision_critic_target
        collision_features = collision_target.extract_features(
            observation, collision_target.features_extractor
        )
        collision_heads = collision_target.all_q_from_features(
            collision_features, batch.actions
        )
        maximum_collision_value = th.stack(
            [th.sigmoid(head) for head in collision_heads], dim=-1
        ).max(dim=-1).values.squeeze(-1)
        risk_adjusted_score = (
            minimum_reward_q
            - self.collision_risk_coef * maximum_collision_value
        )
        return minimum_reward_q, maximum_collision_value, risk_adjusted_score

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("v4.9 requires DecisionAlignedHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)

        batch = actor.all_action_samples(observation, deterministic_speed=True)
        minimum_reward_q, maximum_collision_value, risk_adjusted_score = (
            self._risk_adjusted_target_values(observation, batch)
        )
        (
            actor_indices,
            actor_confidence,
            target_indices,
            actor_override,
            selected_indices,
            keep_is_tied_maximum,
        ) = select_fusion_lane_indices(
            batch.lane_probabilities,
            batch.action_mask,
            risk_adjusted_score,
            actor_non_keep_confidence_threshold=(
                self.actor_non_keep_confidence_threshold
            ),
        )
        selected_actions = actor._select(batch.actions, selected_indices)

        if self._record_fusion_decoder:
            for row_index in range(selected_actions.shape[0]):
                valid = batch.action_mask[row_index].detach().cpu().tolist()
                reward_values = minimum_reward_q[row_index].detach().cpu().tolist()
                collision_values = (
                    maximum_collision_value[row_index].detach().cpu().tolist()
                )
                scores = risk_adjusted_score[row_index].detach().cpu().tolist()
                actor_index = int(actor_indices[row_index])
                target_index = int(target_indices[row_index])
                selected_index = int(selected_indices[row_index])
                override = bool(actor_override[row_index])
                self.fusion_decoder_records.append(
                    {
                        "lane_probabilities": batch.lane_probabilities[row_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "valid_lane_actions": valid,
                        "minimum_target_twin_q": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(reward_values, valid)
                        ],
                        "maximum_target_twin_collision_value": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(collision_values, valid)
                        ],
                        "risk_adjusted_target_score": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(scores, valid)
                        ],
                        "collision_risk_coef": self.collision_risk_coef,
                        "actor_selected_lane_index": actor_index,
                        "actor_selected_lane": int(LANE_COMMANDS[actor_index]),
                        "actor_selected_confidence": float(
                            actor_confidence[row_index]
                        ),
                        "actor_non_keep_confidence_threshold": float(
                            self.actor_non_keep_confidence_threshold
                        ),
                        "actor_override": override,
                        "target_selected_lane_index": target_index,
                        "target_selected_lane": int(LANE_COMMANDS[target_index]),
                        "target_selected_score_minus_keep_score": float(
                            (
                                risk_adjusted_score[row_index, target_index]
                                - risk_adjusted_score[row_index, 1]
                            )
                            .detach()
                            .cpu()
                        ),
                        "keep_was_exact_tied_target_maximum": bool(
                            keep_is_tied_maximum[row_index, 0]
                        ),
                        "selected_lane_index": selected_index,
                        "selected_lane": int(LANE_COMMANDS[selected_index]),
                        "selected_source": (
                            "actor" if override else "target_critic"
                        ),
                        "target_critic_score_model": (
                            "minimum_reward_q_minus_maximum_collision_value"
                        ),
                    }
                )
        return selected_actions


class CollisionConstrainedTargetCriticSACPolicyV49(
    CollisionConstrainedFusionSACPolicyV49
):
    """Use learned reward-minus-collision value for deterministic actions."""

    deterministic_lane_decoder = (
        "argmax_feasible_min_reward_q_minus_max_collision_value"
    )

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("v4.9 requires DecisionAlignedHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)

        batch = actor.all_action_samples(observation, deterministic_speed=True)
        minimum_reward_q, maximum_collision_value, risk_adjusted_score = (
            self._risk_adjusted_target_values(observation, batch)
        )
        masked_score = risk_adjusted_score.masked_fill(~batch.action_mask, -th.inf)
        selected_indices = masked_score.argmax(dim=1)
        maxima = masked_score.max(dim=1, keepdim=True).values
        keep_is_tied_maximum = masked_score[:, 1:2] == maxima
        selected_indices = th.where(
            keep_is_tied_maximum.squeeze(1),
            th.ones_like(selected_indices),
            selected_indices,
        )
        selected_actions = actor._select(batch.actions, selected_indices)

        if self._record_target_decoder:
            for row_index in range(selected_actions.shape[0]):
                valid = batch.action_mask[row_index].detach().cpu().tolist()
                reward_values = minimum_reward_q[row_index].detach().cpu().tolist()
                collision_values = (
                    maximum_collision_value[row_index].detach().cpu().tolist()
                )
                scores = risk_adjusted_score[row_index].detach().cpu().tolist()
                selected_index = int(selected_indices[row_index])
                self.target_decoder_records.append(
                    {
                        "lane_probabilities": batch.lane_probabilities[row_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "valid_lane_actions": valid,
                        "minimum_target_twin_q": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(reward_values, valid)
                        ],
                        "maximum_target_twin_collision_value": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(collision_values, valid)
                        ],
                        "risk_adjusted_target_score": [
                            float(value) if is_valid else None
                            for value, is_valid in zip(scores, valid)
                        ],
                        "collision_risk_coef": self.collision_risk_coef,
                        "selected_lane_index": selected_index,
                        "selected_lane": int(LANE_COMMANDS[selected_index]),
                        "selected_score_minus_keep_score": float(
                            (
                                risk_adjusted_score[row_index, selected_index]
                                - risk_adjusted_score[row_index, 1]
                            )
                            .detach()
                            .cpu()
                        ),
                        "keep_was_exact_tied_maximum": bool(
                            keep_is_tied_maximum[row_index, 0]
                        ),
                    }
                )
        return selected_actions


__all__ = [
    "CollisionConstrainedFusionSACPolicyV49",
    "CollisionConstrainedTargetCriticSACPolicyV49",
]
