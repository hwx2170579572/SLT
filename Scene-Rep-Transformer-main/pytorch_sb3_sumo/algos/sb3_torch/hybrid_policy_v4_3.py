"""Target-critic deterministic lane decoder used only by v4.3."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .hybrid_policy_v4 import DecisionAlignedHybridActor, DecisionAlignedSACPolicy


class TargetCriticDecisionAlignedSACPolicyV43(DecisionAlignedSACPolicy):
    """Use conservative target-twin-Q enumeration only for deterministic lanes.

    Stochastic calls are delegated byte-for-byte to the inherited hybrid actor,
    so training collection and exploration remain the v4.2 method.
    """

    deterministic_lane_decoder = "argmax_feasible_min_target_twin_q"

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self._record_target_decoder = False
        self.target_decoder_records: list[dict[str, Any]] = []

    def begin_target_decoder_recording(self) -> None:
        self.target_decoder_records = []
        self._record_target_decoder = True

    def end_target_decoder_recording(self) -> list[dict[str, Any]]:
        self._record_target_decoder = False
        return list(self.target_decoder_records)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        actor = self.actor
        if not isinstance(actor, DecisionAlignedHybridActor):
            raise TypeError("v4.3 requires DecisionAlignedHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)

        batch = actor.all_action_samples(observation, deterministic_speed=True)
        target = self.critic_target
        features = target.extract_features(observation, target.features_extractor)
        q_heads = target.all_q_from_features(features, batch.actions)
        minimum_q = th.stack(q_heads, dim=-1).min(dim=-1).values.squeeze(-1)
        masked_q = minimum_q.masked_fill(~batch.action_mask, -th.inf)
        selected_indices = masked_q.argmax(dim=1)

        # Torch argmax selects the first index on exact ties.  Prefer keep for
        # exact target-Q ties so an uninformative critic cannot cause a lane move.
        maxima = masked_q.max(dim=1, keepdim=True).values
        keep_is_tied_maximum = masked_q[:, 1:2] == maxima
        selected_indices = th.where(
            keep_is_tied_maximum.squeeze(1),
            th.ones_like(selected_indices),
            selected_indices,
        )
        selected_actions = actor._select(batch.actions, selected_indices)

        if self._record_target_decoder:
            for row_index in range(selected_actions.shape[0]):
                valid = batch.action_mask[row_index].detach().cpu().tolist()
                q_values = minimum_q[row_index].detach().cpu().tolist()
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
                            for value, is_valid in zip(q_values, valid)
                        ],
                        "selected_lane_index": selected_index,
                        "selected_lane": int(LANE_COMMANDS[selected_index]),
                        "selected_q_minus_keep_q": float(
                            minimum_q[row_index, selected_index]
                            - minimum_q[row_index, 1]
                        ),
                        "keep_was_exact_tied_maximum": bool(
                            keep_is_tied_maximum[row_index, 0]
                        ),
                    }
                )
        return selected_actions


__all__ = ["TargetCriticDecisionAlignedSACPolicyV43"]
