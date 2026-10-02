"""Proper-calibrated ranked-risk SAC for the v4.11 model iteration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as functional
from stable_baselines3.common.utils import polyak_update

from .features import _nonzero_mask
from .graph_representation import GraphSLTLosses
from .hybrid_policy_v4 import HybridActionEmbeddedCritic
from .hybrid_policy_v4_11_model import (
    ProperCalibratedRankedRiskSACPolicyV411,
    SupportedMixtureActionBatch,
    SupportedMixtureHybridActor,
    all_proposal_q_from_features,
)
from .sac import _clip_gradients_like_keras
from .sac_v2 import V2_DIAGNOSTIC_NAMES
from .sac_v4_9_model import CollisionConstrainedFusionSACV49
from .topo_temporal_features import StructuredLatent
from .topo_temporal_features_v2 import soft_slot_balance_loss


def soft_target_bernoulli_loss(
    logits: tuple[th.Tensor, ...], target: th.Tensor
) -> th.Tensor:
    """Proper Bernoulli NLL for a soft collision-return target in [0,1]."""

    if not logits:
        raise ValueError("v4.11 collision logits cannot be empty")
    if not bool(((target >= 0.0) & (target <= 1.0)).all()):
        raise ValueError("v4.11 collision target must lie in [0,1]")
    losses = [
        functional.binary_cross_entropy_with_logits(value, target)
        for value in logits
    ]
    return th.stack(losses).mean()


def continuous_pairwise_ranking_loss(
    logits: tuple[th.Tensor, ...], target: th.Tensor
) -> tuple[th.Tensor, float, float]:
    """Rank soft collision returns without a class threshold or fixed margin."""

    if not logits:
        raise ValueError("v4.11 collision logits cannot be empty")
    flattened_target = target.reshape(-1)
    count = int(flattened_target.numel())
    if count < 2:
        zero = sum(value.sum() for value in logits) * 0.0
        return zero, 0.0, 0.0
    target_difference = (
        flattened_target[:, None] - flattened_target[None, :]
    )
    upper = th.triu(
        th.ones((count, count), dtype=th.bool, device=target.device),
        diagonal=1,
    )
    weights = target_difference.abs()[upper]
    signs = target_difference.sign()[upper]
    weight_sum = weights.sum()
    pair_count = int(weights.numel())
    active_rate = float((weights > 0.0).float().mean().detach().cpu())
    mean_weight = float(weights.mean().detach().cpu())
    if not bool(weight_sum > 0.0):
        zero = sum(value.sum() for value in logits) * 0.0
        return zero, active_rate, mean_weight
    losses: list[th.Tensor] = []
    for value in logits:
        flattened = value.reshape(-1)
        if int(flattened.numel()) != count:
            raise ValueError("v4.11 ranking logit/target shape mismatch")
        logit_difference = flattened[:, None] - flattened[None, :]
        pair_losses = functional.softplus(
            -signs * logit_difference[upper]
        )
        losses.append((weights * pair_losses).sum() / weight_sum)
    return th.stack(losses).mean(), active_rate, mean_weight


def temporal_risk_consistency_loss(
    student_logits: tuple[th.Tensor, ...],
    teacher_logits: tuple[th.Tensor, ...],
) -> th.Tensor:
    """Align online risk probabilities to a detached EMA teacher."""

    if len(student_logits) != len(teacher_logits) or not student_logits:
        raise ValueError("v4.11 consistency requires paired non-empty twins")
    losses = [
        functional.mse_loss(
            th.sigmoid(student),
            th.sigmoid(teacher).detach(),
        )
        for student, teacher in zip(student_logits, teacher_logits)
    ]
    return th.stack(losses).mean()


class ProperCalibratedRankedRiskSACV411(CollisionConstrainedFusionSACV49):
    """Jointly learn shared risk features and supported speed mixtures."""

    def __init__(
        self,
        *args: Any,
        replay_support_coef: float = 0.05,
        component_entropy_scale: float = 0.25,
        twin_uncertainty_coef: float = 0.25,
        collision_pairwise_rank_coef: float = 0.25,
        collision_temporal_consistency_coef: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.replay_support_coef = float(replay_support_coef)
        self.component_entropy_scale = float(component_entropy_scale)
        self.twin_uncertainty_coef = float(twin_uncertainty_coef)
        self.collision_pairwise_rank_coef = float(collision_pairwise_rank_coef)
        self.collision_temporal_consistency_coef = float(
            collision_temporal_consistency_coef
        )
        if self.replay_support_coef < 0.0:
            raise ValueError("replay_support_coef must be non-negative")
        if not 0.0 <= self.component_entropy_scale <= 1.0:
            raise ValueError("component_entropy_scale must be in [0,1]")
        if self.twin_uncertainty_coef < 0.0:
            raise ValueError("twin_uncertainty_coef must be non-negative")
        if self.collision_pairwise_rank_coef < 0.0:
            raise ValueError("collision_pairwise_rank_coef must be non-negative")
        if self.collision_temporal_consistency_coef < 0.0:
            raise ValueError(
                "collision_temporal_consistency_coef must be non-negative"
            )
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(
            self.policy, ProperCalibratedRankedRiskSACPolicyV411
        ):
            raise TypeError("v4.11 requires its supported-mixture policy")
        if abs(
            self.policy.twin_uncertainty_coef - self.twin_uncertainty_coef
        ) > 1e-12:
            raise ValueError("policy and learner uncertainty coefficients differ")
        if self.representation is not None:
            representation_parameters = [
                parameter
                for parameter in self.representation.parameters()
                if parameter.requires_grad
            ]
            if not representation_parameters:
                raise RuntimeError("v4.11 representation has no trainable heads")
            learning_rate = (
                float(self.representation_learning_rate)
                if self.representation_learning_rate is not None
                else float(self.lr_schedule(1.0))
            )
            self._representation_parameters = representation_parameters
            self.representation_optimizer = th.optim.NAdam(
                representation_parameters,
                lr=learning_rate,
                eps=1e-7,
            )
        audit = self.optimizer_ownership_audit()
        if audit["overlap_count"] != 0:
            raise RuntimeError("v4.11 optimizer parameter ownership overlaps")

    def optimizer_ownership_audit(self) -> dict[str, Any]:
        policy = self.policy
        if not isinstance(policy, ProperCalibratedRankedRiskSACPolicyV411):
            raise TypeError("v4.11 optimizer audit requires v4.11 policy")
        groups: dict[str, list[th.nn.Parameter]] = {
            "encoder": [
                parameter
                for group in policy.encoder_optimizer.param_groups
                for parameter in group["params"]
            ],
            "actor_heads": [
                parameter
                for group in self.actor.optimizer.param_groups
                for parameter in group["params"]
            ],
            "reward_heads": [
                parameter
                for group in self.critic.optimizer.param_groups
                for parameter in group["params"]
            ],
            "collision_heads": [
                parameter
                for group in policy.collision_critic.optimizer.param_groups
                for parameter in group["params"]
            ],
        }
        if self.representation_optimizer is not None:
            groups["representation_heads"] = [
                parameter
                for group in self.representation_optimizer.param_groups
                for parameter in group["params"]
            ]
        if self.ent_coef_optimizer is not None:
            groups["entropy_coefficient"] = [
                parameter
                for group in self.ent_coef_optimizer.param_groups
                for parameter in group["params"]
            ]
        owners: dict[int, list[str]] = {}
        for name, parameters in groups.items():
            for parameter in parameters:
                owners.setdefault(id(parameter), []).append(name)
        overlaps = {
            str(identifier): names
            for identifier, names in owners.items()
            if len(names) > 1
        }
        return {
            "group_parameter_counts": {
                name: len(parameters) for name, parameters in groups.items()
            },
            "total_unique_parameters_owned": len(owners),
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
            "policy_structure": policy.optimizer_parameter_ownership(),
        }

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        for name in (
            "policy.encoder_optimizer",
            "policy.collision_critic.optimizer",
        ):
            if name not in state_dicts:
                state_dicts.append(name)
        return state_dicts, variables

    def entropy_log_probabilities_v410(
        self, batch: SupportedMixtureActionBatch
    ) -> th.Tensor:
        return (
            batch.speed_log_probabilities
            + self.component_entropy_scale * batch.component_log_probabilities
            + self.lane_entropy_scale
            * batch.lane_log_probabilities.unsqueeze(-1)
        )

    def expected_entropy_log_probability_v410(
        self, batch: SupportedMixtureActionBatch
    ) -> th.Tensor:
        return (
            batch.proposal_probabilities
            * self.entropy_log_probabilities_v410(batch)
        ).sum(dim=(1, 2), keepdim=False).unsqueeze(1)

    @staticmethod
    def _twin_reward_values(heads: tuple[th.Tensor, ...]) -> tuple[th.Tensor, th.Tensor]:
        values = th.stack(heads, dim=-1).squeeze(-2)
        if values.shape[-1] != 2:
            raise ValueError("v4.11 requires exactly two reward critics")
        return values.min(dim=-1).values, (values[..., 0] - values[..., 1]).abs()

    @staticmethod
    def _twin_collision_values(
        heads: tuple[th.Tensor, ...]
    ) -> tuple[th.Tensor, th.Tensor]:
        values = th.stack([th.sigmoid(head) for head in heads], dim=-1).squeeze(-2)
        if values.shape[-1] != 2:
            raise ValueError("v4.11 requires exactly two collision critics")
        return values.max(dim=-1).values, (values[..., 0] - values[..., 1]).abs()

    def _representation_loss_v410(self, replay_data: Any) -> th.Tensor | None:
        if self.representation is None:
            return None
        if not self.structured_representation:
            raise TypeError("v4.11 requires structured Graph-SLT representation")
        target_critic = (
            self.critic
            if self.representation_online_target_encoder
            else self.critic_target
        )
        next_observations = getattr(
            replay_data,
            "one_step_next_observations",
            replay_data.next_observations,
        )
        sample_mask = _nonzero_mask(next_observations["trajectory"][:, 0, 0])
        online_extractor = self.critic.features_extractor
        target_extractor = target_critic.features_extractor
        online_features = online_extractor.forward_tokens(  # type: ignore[attr-defined]
            replay_data.observations
        )
        with th.no_grad():
            target_features = target_extractor.forward_tokens(  # type: ignore[attr-defined]
                next_observations
            )
        if not isinstance(online_features, StructuredLatent) or not isinstance(
            target_features, StructuredLatent
        ):
            raise TypeError("v4.11 Graph-SLT requires StructuredLatent features")
        graph_losses = self.representation(
            online_features,
            replay_data.actions,
            target_features,
            sample_mask=sample_mask,
        )
        if not isinstance(graph_losses, GraphSLTLosses):
            raise TypeError("v4.11 Graph-SLT returned an invalid loss object")
        balance_loss, _ = soft_slot_balance_loss(
            online_features,
            epsilon=self.slot_balance_epsilon,
        )
        weighted_balance = self.slot_balance_coef * balance_loss
        loss = graph_losses.total + weighted_balance
        if not bool(th.isfinite(loss).all()):
            raise FloatingPointError("non-finite v4.11 representation loss")
        self._last_graph_slt_losses = {
            **graph_losses.detached(),
            "soft_slot_balance_loss": balance_loss.detach(),
            "weighted_soft_slot_balance_loss": weighted_balance.detach(),
        }
        return loss

    @staticmethod
    def _optimizer_parameters(optimizer: th.optim.Optimizer) -> list[th.nn.Parameter]:
        return [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self._pending_raw_gradient_steps is not None:
            gradient_steps = self._pending_raw_gradient_steps
            self._pending_raw_gradient_steps = None
        if gradient_steps <= 0:
            return
        actor, critic, critic_target = self._hybrid_modules()
        collision_critic, collision_critic_target = self._collision_modules()
        policy = self.policy
        if not isinstance(actor, SupportedMixtureHybridActor) or not isinstance(
            policy, ProperCalibratedRankedRiskSACPolicyV411
        ):
            raise TypeError("v4.11 learner requires supported-mixture modules")
        train_started = time.perf_counter()
        policy.set_training_mode(True)
        optimizers: list[th.optim.Optimizer] = [
            policy.encoder_optimizer,
            actor.optimizer,
            critic.optimizer,
            collision_critic.optimizer,
        ]
        if self.representation_optimizer is not None:
            optimizers.append(self.representation_optimizer)
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        diagnostics: dict[str, list[float]] = {
            "ent_coef": [],
            "actor_loss": [],
            "critic_loss": [],
            "collision_critic_loss": [],
            "collision_soft_bce_loss": [],
            "collision_pairwise_rank_loss": [],
            "collision_pairwise_active_pair_rate": [],
            "collision_pairwise_mean_target_difference": [],
            "collision_temporal_consistency_loss": [],
            "collision_probability_absolute_error": [],
            "representation_loss": [],
            "replay_support_nll": [],
            "component_entropy": [],
            "component_speed_spread": [],
            "proposal_boundary_rate": [],
            "policy_expected_collision_cost": [],
            "policy_reward_twin_disagreement": [],
            "policy_collision_twin_disagreement": [],
            "policy_learned_uncertainty": [],
            "collision_cost_label": [],
            "collision_cost_target": [],
            "collision_cost_replay_prediction": [],
            "lane_entropy": [],
            "valid_lane_actions": [],
        }
        entropy_losses: list[float] = []
        graph_slt_slot_losses: dict[str, list[float]] = {
            "graph_slt_ego_loss": [],
            "graph_slt_social_loss": [],
            "graph_slt_route_loss": [],
        }

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            if not hasattr(replay_data, "collision_costs"):
                raise TypeError("v4.11 replay sample is missing collision_costs")
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            representation_loss = self._representation_loss_v410(replay_data)
            if representation_loss is not None:
                diagnostics["representation_loss"].append(
                    float(representation_loss.detach().cpu())
                )
                for name in graph_slt_slot_losses:
                    value = self._last_graph_slt_losses.get(name)
                    if value is not None:
                        graph_slt_slot_losses[name].append(float(value.cpu()))

            shared_features = critic.extract_features(
                replay_data.observations, critic.features_extractor
            )
            current_q_values = critic.forward_from_features(
                shared_features, replay_data.actions
            )
            current_collision_logits = collision_critic.forward_from_features(
                shared_features, replay_data.actions
            )
            current_collision_probabilities = tuple(
                th.sigmoid(value) for value in current_collision_logits
            )
            policy_batch = actor.all_action_proposals(
                replay_data.observations, deterministic_speed=False
            )
            expected_entropy_log_prob = (
                self.expected_entropy_log_probability_v410(policy_batch)
            )

            entropy_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                entropy_coefficient = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                entropy_loss = -(
                    th.exp(self.log_ent_coef)
                    * (expected_entropy_log_prob + self.target_entropy).detach()
                ).mean()
                entropy_losses.append(float(entropy_loss.detach().cpu()))
            else:
                entropy_coefficient = self.ent_coef_tensor
            diagnostics["ent_coef"].append(
                float(entropy_coefficient.detach().cpu())
            )

            target_extractor = critic_target.features_extractor
            target_was_training = target_extractor.training
            target_extractor.train(True)
            with th.no_grad():
                teacher_current_features = critic_target.extract_features(
                    replay_data.observations,
                    target_extractor,
                )
                teacher_current_collision_logits = (
                    collision_critic_target.forward_from_features(
                        teacher_current_features, replay_data.actions
                    )
                )
                next_policy = actor.all_action_proposals(
                    replay_data.next_observations, deterministic_speed=False
                )
                next_features = critic_target.extract_features(
                    replay_data.next_observations,
                    target_extractor,
                )
                next_reward_heads = all_proposal_q_from_features(
                    critic_target, next_features, next_policy.actions
                )
                next_min_q, _ = self._twin_reward_values(next_reward_heads)
                next_entropy = self.entropy_log_probabilities_v410(next_policy)
                next_soft = next_min_q - entropy_coefficient * next_entropy
                next_reward_value = (
                    next_policy.proposal_probabilities * next_soft
                ).sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                target_q_values = replay_data.rewards + (
                    1.0 - replay_data.dones
                ) * discounts * next_reward_value

                next_collision_heads = all_proposal_q_from_features(
                    collision_critic_target,
                    next_features,
                    next_policy.actions,
                )
                next_collision, _ = self._twin_collision_values(
                    next_collision_heads
                )
                next_collision_value = (
                    next_policy.proposal_probabilities * next_collision
                ).sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                target_collision_values = (
                    replay_data.collision_costs
                    + (1.0 - replay_data.dones)
                    * discounts
                    * next_collision_value
                ).clamp(0.0, 1.0)
            target_extractor.train(target_was_training)

            critic_loss = 0.5 * sum(
                functional.mse_loss(current, target_q_values)
                for current in current_q_values
            )
            collision_soft_bce_loss = soft_target_bernoulli_loss(
                current_collision_logits, target_collision_values
            )
            (
                collision_pairwise_rank_loss,
                collision_pairwise_active_pair_rate,
                collision_pairwise_mean_target_difference,
            ) = continuous_pairwise_ranking_loss(
                current_collision_logits, target_collision_values
            )
            collision_temporal_consistency_loss = (
                temporal_risk_consistency_loss(
                    current_collision_logits,
                    teacher_current_collision_logits,
                )
            )
            collision_critic_loss = (
                collision_soft_bce_loss
                + self.collision_pairwise_rank_coef
                * collision_pairwise_rank_loss
                + self.collision_temporal_consistency_coef
                * collision_temporal_consistency_loss
            )

            reward_head_parameters = self._optimizer_parameters(
                critic.optimizer
            )
            collision_head_parameters = self._optimizer_parameters(
                collision_critic.optimizer
            )
            for parameter in (*reward_head_parameters, *collision_head_parameters):
                parameter.requires_grad_(False)
            actor_features = shared_features.detach()
            actor_reward_heads = all_proposal_q_from_features(
                critic, actor_features, policy_batch.actions
            )
            min_q, reward_disagreement = self._twin_reward_values(
                actor_reward_heads
            )
            actor_collision_heads = all_proposal_q_from_features(
                collision_critic, actor_features, policy_batch.actions
            )
            collision_value, collision_disagreement = (
                self._twin_collision_values(actor_collision_heads)
            )
            uncertainty = reward_disagreement + collision_disagreement
            per_proposal_objective = (
                entropy_coefficient
                * self.entropy_log_probabilities_v410(policy_batch)
                - min_q
                + self.collision_risk_coef * collision_value
                + self.twin_uncertainty_coef * uncertainty
            )
            actor_loss = (
                policy_batch.proposal_probabilities * per_proposal_objective
            ).sum(dim=(1, 2), keepdim=False).mean()
            replay_support_nll = actor.replay_support_nll(
                policy_batch, replay_data.actions
            )
            actor_total_loss = (
                actor_loss + self.replay_support_coef * replay_support_nll
            )
            for parameter in (*reward_head_parameters, *collision_head_parameters):
                parameter.requires_grad_(True)

            for optimizer in optimizers:
                optimizer.zero_grad()
            critic_total_loss = critic_loss + collision_critic_loss
            if representation_loss is not None:
                critic_total_loss = (
                    critic_total_loss
                    + self.representation_coef * representation_loss
                )
            critic_total_loss.backward()
            actor_total_loss.backward()
            if entropy_loss is not None:
                entropy_loss.backward()

            encoder_parameters = self._optimizer_parameters(
                policy.encoder_optimizer
            )
            actor_parameters = self._optimizer_parameters(actor.optimizer)
            _clip_gradients_like_keras(encoder_parameters, self.max_grad_norm)
            _clip_gradients_like_keras(
                reward_head_parameters, self.max_grad_norm
            )
            _clip_gradients_like_keras(
                collision_head_parameters, self.max_grad_norm
            )
            _clip_gradients_like_keras(actor_parameters, self.max_grad_norm)
            if self._representation_parameters:
                _clip_gradients_like_keras(
                    self._representation_parameters, self.max_grad_norm
                )
            policy.encoder_optimizer.step()
            critic.optimizer.step()
            collision_critic.optimizer.step()
            if self.representation_optimizer is not None:
                self.representation_optimizer.step()
            actor.optimizer.step()
            if entropy_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    critic.parameters(), critic_target.parameters(), self.tau
                )
                target_encoder_ids = {
                    id(parameter)
                    for parameter in critic_target.features_extractor.parameters()
                }
                collision_target_heads = [
                    parameter
                    for parameter in collision_critic_target.parameters()
                    if id(parameter) not in target_encoder_ids
                ]
                polyak_update(
                    collision_head_parameters,
                    collision_target_heads,
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )
                if self.representation is not None:
                    self.representation.update_target(self.tau)

            proposal_speeds = th.tanh(policy_batch.speed_means)
            if actor.speed_components > 1:
                pairwise = (
                    proposal_speeds.unsqueeze(-1)
                    - proposal_speeds.unsqueeze(-2)
                ).abs()
                off_diagonal = ~th.eye(
                    actor.speed_components,
                    dtype=th.bool,
                    device=pairwise.device,
                ).view(1, 1, actor.speed_components, actor.speed_components)
                component_spread = pairwise.masked_select(off_diagonal).mean()
            else:
                component_spread = th.zeros((), device=proposal_speeds.device)
            component_entropy = -(
                policy_batch.component_probabilities
                * policy_batch.component_log_probabilities
            ).sum(dim=2)
            expected_component_entropy = (
                policy_batch.lane_probabilities * component_entropy
            ).sum(dim=1).mean()
            lane_entropy = -(
                policy_batch.lane_probabilities
                * policy_batch.lane_log_probabilities
            ).sum(dim=1).mean()
            expected_collision = (
                policy_batch.proposal_probabilities * collision_value
            ).sum(dim=(1, 2), keepdim=False).mean()
            expected_reward_disagreement = (
                policy_batch.proposal_probabilities * reward_disagreement
            ).sum(dim=(1, 2), keepdim=False).mean()
            expected_collision_disagreement = (
                policy_batch.proposal_probabilities * collision_disagreement
            ).sum(dim=(1, 2), keepdim=False).mean()
            expected_uncertainty = (
                policy_batch.proposal_probabilities * uncertainty
            ).sum(dim=(1, 2), keepdim=False).mean()
            diagnostics["actor_loss"].append(float(actor_loss.detach().cpu()))
            diagnostics["critic_loss"].append(float(critic_loss.detach().cpu()))
            diagnostics["collision_critic_loss"].append(
                float(collision_critic_loss.detach().cpu())
            )
            diagnostics["collision_soft_bce_loss"].append(
                float(collision_soft_bce_loss.detach().cpu())
            )
            diagnostics["collision_pairwise_rank_loss"].append(
                float(collision_pairwise_rank_loss.detach().cpu())
            )
            diagnostics["collision_pairwise_active_pair_rate"].append(
                collision_pairwise_active_pair_rate
            )
            diagnostics["collision_pairwise_mean_target_difference"].append(
                collision_pairwise_mean_target_difference
            )
            diagnostics["collision_temporal_consistency_loss"].append(
                float(collision_temporal_consistency_loss.detach().cpu())
            )
            diagnostics["collision_probability_absolute_error"].append(
                float(
                    (
                        self._collision_probabilities(current_collision_logits)
                        - target_collision_values
                    )
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            diagnostics["replay_support_nll"].append(
                float(replay_support_nll.detach().cpu())
            )
            diagnostics["component_entropy"].append(
                float(expected_component_entropy.detach().cpu())
            )
            diagnostics["component_speed_spread"].append(
                float(component_spread.detach().cpu())
            )
            diagnostics["proposal_boundary_rate"].append(
                float((proposal_speeds.abs() >= 0.95).float().mean().detach().cpu())
            )
            diagnostics["policy_expected_collision_cost"].append(
                float(expected_collision.detach().cpu())
            )
            diagnostics["policy_reward_twin_disagreement"].append(
                float(expected_reward_disagreement.detach().cpu())
            )
            diagnostics["policy_collision_twin_disagreement"].append(
                float(expected_collision_disagreement.detach().cpu())
            )
            diagnostics["policy_learned_uncertainty"].append(
                float(expected_uncertainty.detach().cpu())
            )
            diagnostics["collision_cost_label"].append(
                float(replay_data.collision_costs.mean().detach().cpu())
            )
            diagnostics["collision_cost_target"].append(
                float(target_collision_values.mean().detach().cpu())
            )
            diagnostics["collision_cost_replay_prediction"].append(
                float(
                    self._collision_probabilities(current_collision_logits)
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            diagnostics["lane_entropy"].append(float(lane_entropy.detach().cpu()))
            diagnostics["valid_lane_actions"].append(
                float(policy_batch.action_mask.sum(dim=1).float().mean().cpu())
            )
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if entropy_losses:
            value = float(np.mean(entropy_losses))
            self.logger.record("train/ent_coef_loss", value)
            self._accumulate_training_stat("train/ent_coef_loss", value)
        namespaces = {
            "ent_coef": "train/ent_coef",
            "actor_loss": "train/actor_loss",
            "critic_loss": "train/critic_loss",
            "collision_critic_loss": "train/collision_critic_loss",
            "collision_soft_bce_loss": "train/collision_soft_bce_loss",
            "collision_pairwise_rank_loss": "train/collision_pairwise_rank_loss",
            "collision_pairwise_active_pair_rate": (
                "risk/collision_pairwise_active_pair_rate"
            ),
            "collision_pairwise_mean_target_difference": (
                "risk/collision_pairwise_mean_target_difference"
            ),
            "collision_temporal_consistency_loss": (
                "train/collision_temporal_consistency_loss"
            ),
            "collision_probability_absolute_error": (
                "risk/collision_probability_absolute_error"
            ),
            "representation_loss": "train/representation_loss",
            "replay_support_nll": "support/replay_speed_mixture_nll",
            "component_entropy": "support/component_entropy",
            "component_speed_spread": "support/component_speed_spread",
            "proposal_boundary_rate": "support/proposal_boundary_rate",
            "policy_expected_collision_cost": "risk/policy_expected_collision_cost",
            "policy_reward_twin_disagreement": "risk/reward_twin_disagreement",
            "policy_collision_twin_disagreement": "risk/collision_twin_disagreement",
            "policy_learned_uncertainty": "risk/learned_uncertainty",
            "collision_cost_label": "risk/collision_cost_label",
            "collision_cost_target": "risk/collision_cost_target",
            "collision_cost_replay_prediction": "risk/collision_cost_replay_prediction",
            "lane_entropy": "hybrid/lane_entropy",
            "valid_lane_actions": "hybrid/valid_lane_actions",
        }
        for source, target in namespaces.items():
            values = diagnostics[source]
            if values:
                value = float(np.mean(values))
                self.logger.record(target, value)
                self._accumulate_training_stat(target, value)
        if diagnostics["representation_loss"]:
            value = float(np.mean(diagnostics["representation_loss"]))
            self.logger.record("train/graph_slt_loss", value)
            self._accumulate_training_stat("train/graph_slt_loss", value)
        for name, values in graph_slt_slot_losses.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(f"train/{name}", value)
                self._accumulate_training_stat(f"train/{name}", value)
        extractor = critic.features_extractor
        if hasattr(extractor, "diagnostic_values"):
            values = extractor.diagnostic_values()
            for name in sorted(V2_DIAGNOSTIC_NAMES):
                item = values.get(name)
                if item is not None and bool(th.isfinite(item).all()):
                    scalar = float(item.detach().mean().cpu())
                    self.logger.record(f"diagnostic/{name}", scalar)
                    self._accumulate_training_stat(f"diagnostic/{name}", scalar)
        self._accumulate_training_stat(
            "performance/train_ms_per_gradient_step",
            (time.perf_counter() - train_started) * 1000.0 / gradient_steps,
        )


__all__ = [
    "ProperCalibratedRankedRiskSACV411",
    "continuous_pairwise_ranking_loss",
    "soft_target_bernoulli_loss",
    "temporal_risk_consistency_loss",
]
