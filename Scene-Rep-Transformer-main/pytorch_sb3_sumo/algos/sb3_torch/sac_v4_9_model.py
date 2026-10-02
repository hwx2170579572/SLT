"""Collision-constrained hybrid SAC for the v4.9 model iteration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as functional
from stable_baselines3.common.utils import polyak_update

from .hybrid_policy_v4 import HybridActionEmbeddedCritic
from .hybrid_policy_v4_9_model import CollisionConstrainedFusionSACPolicyV49
from .sac import _clip_gradients_like_keras
from .sac_v2 import V2_DIAGNOSTIC_NAMES
from .sac_v4_5 import ConfidentActorFusionSACV45


class CollisionConstrainedFusionSACV49(ConfidentActorFusionSACV45):
    """Optimize the actor against reward Q and a learned collision value."""

    def __init__(
        self,
        *args: Any,
        collision_risk_coef: float = 1.0,
        **kwargs: Any,
    ) -> None:
        collision_risk_coef = float(collision_risk_coef)
        if collision_risk_coef < 0.0:
            raise ValueError("collision_risk_coef must be non-negative")
        self.collision_risk_coef = collision_risk_coef
        super().__init__(*args, **kwargs)

    def _collision_modules(
        self,
    ) -> tuple[HybridActionEmbeddedCritic, HybridActionEmbeddedCritic]:
        if not isinstance(self.policy, CollisionConstrainedFusionSACPolicyV49):
            raise TypeError("v4.9 requires CollisionConstrainedFusionSACPolicyV49")
        online = self.policy.collision_critic
        target = self.policy.collision_critic_target
        if not isinstance(online, HybridActionEmbeddedCritic) or not isinstance(
            target, HybridActionEmbeddedCritic
        ):
            raise TypeError("v4.9 requires hybrid collision critics")
        return online, target

    @staticmethod
    def _collision_probabilities(
        heads: tuple[th.Tensor, ...],
    ) -> th.Tensor:
        """Use the pessimistic twin maximum for a bounded collision value."""

        return th.stack([th.sigmoid(head) for head in heads], dim=-1).max(
            dim=-1
        ).values

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self._pending_raw_gradient_steps is not None:
            gradient_steps = self._pending_raw_gradient_steps
            self._pending_raw_gradient_steps = None
        if gradient_steps <= 0:
            return
        actor, critic, critic_target = self._hybrid_modules()
        collision_critic, collision_critic_target = self._collision_modules()
        train_started = time.perf_counter()
        self.policy.set_training_mode(True)
        optimizers = [
            actor.optimizer,
            critic.optimizer,
            collision_critic.optimizer,
        ]
        if (
            self.representation_optimizer is not None
            and self.representation_learning_rate is None
        ):
            optimizers.append(self.representation_optimizer)
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        entropy_losses: list[float] = []
        entropy_coefficients: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        collision_critic_losses: list[float] = []
        representation_losses: list[float] = []
        graph_slt_slot_losses: dict[str, list[float]] = {
            "graph_slt_ego_loss": [],
            "graph_slt_social_loss": [],
            "graph_slt_route_loss": [],
        }
        hybrid_diagnostics: dict[str, list[float]] = {
            "lane_entropy": [],
            "valid_lane_actions": [],
            "lane_probability_negative": [],
            "lane_probability_keep": [],
            "lane_probability_positive": [],
            "expected_joint_log_probability": [],
            "expected_speed_log_probability": [],
            "effective_lane_entropy_coefficient": [],
        }
        risk_diagnostics: dict[str, list[float]] = {
            "collision_cost_label": [],
            "collision_cost_target": [],
            "collision_cost_replay_prediction": [],
            "policy_expected_collision_cost": [],
            "collision_risk_actor_penalty": [],
        }

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            if not hasattr(replay_data, "collision_costs"):
                raise TypeError("v4.9 replay sample is missing collision_costs")
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            if self.representation is not None:
                representation_loss = self._representation_step(replay_data)
                representation_losses.append(float(representation_loss.cpu()))
                for name in graph_slt_slot_losses:
                    value = self._last_graph_slt_losses.get(name)
                    if value is not None:
                        graph_slt_slot_losses[name].append(float(value.cpu()))

            current_q_values = critic(
                replay_data.observations, replay_data.actions
            )
            current_collision_logits = collision_critic(
                replay_data.observations, replay_data.actions
            )
            current_collision_probabilities = tuple(
                th.sigmoid(value) for value in current_collision_logits
            )
            policy_batch = actor.all_action_samples(
                replay_data.observations, deterministic_speed=False
            )
            expected_entropy_log_prob = self.expected_entropy_log_probability(
                policy_batch
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
            entropy_coefficients.append(float(entropy_coefficient.detach().cpu()))

            reward_target_extractor = critic_target.features_extractor
            reward_target_was_training = reward_target_extractor.training
            collision_target_extractor = collision_critic_target.features_extractor
            collision_target_was_training = collision_target_extractor.training
            reward_target_extractor.train(True)
            collision_target_extractor.train(True)
            with th.no_grad():
                next_policy = actor.all_action_samples(
                    replay_data.next_observations, deterministic_speed=False
                )
                next_reward_features = critic_target.extract_features(
                    replay_data.next_observations,
                    critic_target.features_extractor,
                )
                next_q_heads = critic_target.all_q_from_features(
                    next_reward_features, next_policy.actions
                )
                next_min_q = th.stack(next_q_heads, dim=-1).min(dim=-1).values
                next_entropy_log_prob = self.entropy_log_probabilities(next_policy)
                next_soft_values = (
                    next_min_q
                    - entropy_coefficient
                    * next_entropy_log_prob.unsqueeze(-1)
                )
                next_reward_value = (
                    next_policy.lane_probabilities.unsqueeze(-1)
                    * next_soft_values
                ).sum(dim=1)
                target_q_values = replay_data.rewards + (
                    1.0 - replay_data.dones
                ) * discounts * next_reward_value

                next_collision_features = collision_critic_target.extract_features(
                    replay_data.next_observations,
                    collision_critic_target.features_extractor,
                )
                next_collision_heads = collision_critic_target.all_q_from_features(
                    next_collision_features, next_policy.actions
                )
                next_collision_per_lane = self._collision_probabilities(
                    next_collision_heads
                )
                next_collision_value = (
                    next_policy.lane_probabilities.unsqueeze(-1)
                    * next_collision_per_lane
                ).sum(dim=1)
                target_collision_values = (
                    replay_data.collision_costs
                    + (1.0 - replay_data.dones)
                    * discounts
                    * next_collision_value
                ).clamp(0.0, 1.0)
            reward_target_extractor.train(reward_target_was_training)
            collision_target_extractor.train(collision_target_was_training)

            critic_loss = 0.5 * sum(
                functional.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            collision_critic_loss = 0.5 * sum(
                functional.mse_loss(current, target_collision_values)
                for current in current_collision_probabilities
            )
            critic_losses.append(float(critic_loss.detach().cpu()))
            collision_critic_losses.append(
                float(collision_critic_loss.detach().cpu())
            )

            reward_parameters = list(critic.parameters())
            collision_parameters = list(collision_critic.parameters())
            reward_requires_grad = [
                parameter.requires_grad for parameter in reward_parameters
            ]
            collision_requires_grad = [
                parameter.requires_grad for parameter in collision_parameters
            ]
            for parameter in (*reward_parameters, *collision_parameters):
                parameter.requires_grad_(False)

            actor_reward_features = critic.extract_features(
                replay_data.observations, critic.features_extractor
            )
            actor_q_heads = critic.all_q_from_features(
                actor_reward_features, policy_batch.actions
            )
            min_q = th.stack(actor_q_heads, dim=-1).min(dim=-1).values
            actor_collision_features = collision_critic.extract_features(
                replay_data.observations,
                collision_critic.features_extractor,
            )
            actor_collision_heads = collision_critic.all_q_from_features(
                actor_collision_features, policy_batch.actions
            )
            collision_per_lane = self._collision_probabilities(
                actor_collision_heads
            )
            per_lane_actor_objective = (
                entropy_coefficient
                * self.entropy_log_probabilities(policy_batch).unsqueeze(-1)
                - min_q
                + self.collision_risk_coef * collision_per_lane
            )
            actor_loss = (
                policy_batch.lane_probabilities.unsqueeze(-1)
                * per_lane_actor_objective
            ).sum(dim=1).mean()
            actor_losses.append(float(actor_loss.detach().cpu()))
            for parameter, original in zip(
                reward_parameters, reward_requires_grad
            ):
                parameter.requires_grad_(original)
            for parameter, original in zip(
                collision_parameters, collision_requires_grad
            ):
                parameter.requires_grad_(original)

            actor_parameters = [
                parameter
                for parameter in actor.parameters()
                if id(parameter)
                not in {
                    id(item)
                    for item in actor.features_extractor.parameters()
                }
            ]
            critic.optimizer.zero_grad()
            collision_critic.optimizer.zero_grad()
            actor.optimizer.zero_grad()
            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
            critic_loss.backward()
            collision_critic_loss.backward()
            actor_loss.backward()
            if entropy_loss is not None:
                entropy_loss.backward()
            _clip_gradients_like_keras(reward_parameters, self.max_grad_norm)
            _clip_gradients_like_keras(
                collision_parameters, self.max_grad_norm
            )
            _clip_gradients_like_keras(actor_parameters, self.max_grad_norm)
            critic.optimizer.step()
            collision_critic.optimizer.step()

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    critic.parameters(), critic_target.parameters(), self.tau
                )
                polyak_update(
                    collision_critic.parameters(),
                    collision_critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )
                if self.representation is not None:
                    self.representation.update_target(self.tau)

            actor.optimizer.step()
            if entropy_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.step()

            lane_entropy = -(
                policy_batch.lane_probabilities
                * policy_batch.lane_log_probabilities
            ).sum(dim=1).mean()
            expected_speed_log_prob = (
                policy_batch.lane_probabilities
                * policy_batch.speed_log_probabilities
            ).sum(dim=1).mean()
            hybrid_diagnostics["lane_entropy"].append(
                float(lane_entropy.detach().cpu())
            )
            hybrid_diagnostics["valid_lane_actions"].append(
                float(policy_batch.action_mask.sum(dim=1).float().mean().cpu())
            )
            for index, name in enumerate(
                (
                    "lane_probability_negative",
                    "lane_probability_keep",
                    "lane_probability_positive",
                )
            ):
                hybrid_diagnostics[name].append(
                    float(
                        policy_batch.lane_probabilities[:, index]
                        .mean()
                        .detach()
                        .cpu()
                    )
                )
            hybrid_diagnostics["expected_joint_log_probability"].append(
                float(policy_batch.expected_log_probability.mean().detach().cpu())
            )
            hybrid_diagnostics["expected_speed_log_probability"].append(
                float(expected_speed_log_prob.detach().cpu())
            )
            hybrid_diagnostics["effective_lane_entropy_coefficient"].append(
                float(entropy_coefficient.detach().cpu()) * self.lane_entropy_scale
            )
            expected_policy_collision = (
                policy_batch.lane_probabilities.unsqueeze(-1)
                * collision_per_lane
            ).sum(dim=1).mean()
            risk_diagnostics["collision_cost_label"].append(
                float(replay_data.collision_costs.mean().detach().cpu())
            )
            risk_diagnostics["collision_cost_target"].append(
                float(target_collision_values.mean().detach().cpu())
            )
            risk_diagnostics["collision_cost_replay_prediction"].append(
                float(
                    self._collision_probabilities(current_collision_logits)
                    .mean()
                    .detach()
                    .cpu()
                )
            )
            risk_diagnostics["policy_expected_collision_cost"].append(
                float(expected_policy_collision.detach().cpu())
            )
            risk_diagnostics["collision_risk_actor_penalty"].append(
                float(
                    (self.collision_risk_coef * expected_policy_collision)
                    .detach()
                    .cpu()
                )
            )
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        scalar_series = {
            "train/ent_coef": entropy_coefficients,
            "train/actor_loss": actor_losses,
            "train/critic_loss": critic_losses,
            "train/collision_critic_loss": collision_critic_losses,
            "train/ent_coef_loss": entropy_losses,
            "train/representation_loss": representation_losses,
        }
        for name, values in scalar_series.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(name, value)
                self._accumulate_training_stat(name, value)
        if representation_losses:
            value = float(np.mean(representation_losses))
            self.logger.record("train/graph_slt_loss", value)
            self._accumulate_training_stat("train/graph_slt_loss", value)
        for name, values in graph_slt_slot_losses.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(f"train/{name}", value)
                self._accumulate_training_stat(f"train/{name}", value)
        for name, values in hybrid_diagnostics.items():
            value = float(np.mean(values))
            self.logger.record(f"hybrid/{name}", value)
            self._accumulate_training_stat(f"hybrid/{name}", value)
        for name, values in risk_diagnostics.items():
            value = float(np.mean(values))
            self.logger.record(f"risk/{name}", value)
            self._accumulate_training_stat(f"risk/{name}", value)
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


__all__ = ["CollisionConstrainedFusionSACV49"]
