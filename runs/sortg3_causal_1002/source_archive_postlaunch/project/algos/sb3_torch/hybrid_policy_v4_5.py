"""Confidence-gated actor/target-critic deterministic decoder for v4.5."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .hybrid_policy_v4 import DecisionAlignedHybridActor
from .hybrid_policy_v4_3 import TargetCriticDecisionAlignedSACPolicyV43


def select_fusion_lane_indices(
    lane_probabilities: th.Tensor,
    action_mask: th.Tensor,
    minimum_target_twin_q: th.Tensor,
    *,
    actor_non_keep_confidence_threshold: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Return actor, confidence, target, override, selected, and keep-tie tensors."""

    if lane_probabilities.shape != action_mask.shape or action_mask.shape != minimum_target_twin_q.shape:
        raise ValueError("fusion decoder tensors must have identical shapes")
    if lane_probabilities.ndim != 2 or lane_probabilities.shape[1] != 3:
        raise ValueError("fusion decoder expects [batch, 3] lane tensors")
    if not 0.0 <= float(actor_non_keep_confidence_threshold) <= 1.0:
        raise ValueError("fusion confidence threshold must lie in [0, 1]")
    if not bool(action_mask.any(dim=1).all()):
        raise ValueError("every fusion decision requires a feasible lane action")

    masked_q = minimum_target_twin_q.masked_fill(~action_mask, -th.inf)
    target_indices = masked_q.argmax(dim=1)
    maxima = masked_q.max(dim=1, keepdim=True).values
    keep_is_tied_maximum = masked_q[:, 1:2] == maxima
    target_indices = th.where(
        keep_is_tied_maximum.squeeze(1),
        th.ones_like(target_indices),
        target_indices,
    )

    masked_actor_probabilities = lane_probabilities.masked_fill(~action_mask, -th.inf)
    actor_indices = masked_actor_probabilities.argmax(dim=1)
    actor_confidence = lane_probabilities.gather(
        1, actor_indices[:, None]
    ).squeeze(1)
    actor_override = (actor_indices != 1) & (
        actor_confidence >= float(actor_non_keep_confidence_threshold)
    )
    selected_indices = th.where(actor_override, actor_indices, target_indices)
    return (
        actor_indices,
        actor_confidence,
        target_indices,
        actor_override,
        selected_indices,
        keep_is_tied_maximum,
    )


class ConfidentActorFusionSACPolicyV45(TargetCriticDecisionAlignedSACPolicyV43):
    """Use confident actor non-keep actions and target-twin-Q otherwise.

    Only deterministic prediction changes. Stochastic calls are delegated to
    the inherited hybrid actor, preserving v4.4 training and exploration.
    """

    deterministic_lane_decoder = "actor_confident_non_keep_else_target_critic"
    actor_non_keep_confidence_threshold = 0.90

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self._record_fusion_decoder = False
        self.fusion_decoder_records: list[dict[str, Any]] = []

    def begin_fusion_decoder_recording(self) -> None:
        self.fusion_decoder_records = []
        self._record_fusion_decoder = True

    def end_fusion_decoder_recording(self) -> list[dict[str, Any]]:
        self._record_fusion_decoder = False
        return list(self.fusion_decoder_records)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("v4.5 requires DecisionAlignedHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)

        batch = actor.all_action_samples(observation, deterministic_speed=True)
        target = self.critic_target
        features = target.extract_features(observation, target.features_extractor)
        q_heads = target.all_q_from_features(features, batch.actions)
        minimum_q = th.stack(q_heads, dim=-1).min(dim=-1).values.squeeze(-1)
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
            minimum_q,
            actor_non_keep_confidence_threshold=(
                self.actor_non_keep_confidence_threshold
            ),
        )
        selected_actions = actor._select(batch.actions, selected_indices)

        if self._record_fusion_decoder:
            for row_index in range(selected_actions.shape[0]):
                valid = batch.action_mask[row_index].detach().cpu().tolist()
                q_values = minimum_q[row_index].detach().cpu().tolist()
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
                            for value, is_valid in zip(q_values, valid)
                        ],
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
                        "target_selected_q_minus_keep_q": float(
                            minimum_q[row_index, target_index]
                            - minimum_q[row_index, 1]
                        ),
                        "keep_was_exact_tied_target_maximum": bool(
                            keep_is_tied_maximum[row_index, 0]
                        ),
                        "selected_lane_index": selected_index,
                        "selected_lane": int(LANE_COMMANDS[selected_index]),
                        "selected_source": "actor" if override else "target_critic",
                    }
                )
        return selected_actions


__all__ = ["ConfidentActorFusionSACPolicyV45", "select_fusion_lane_indices"]
