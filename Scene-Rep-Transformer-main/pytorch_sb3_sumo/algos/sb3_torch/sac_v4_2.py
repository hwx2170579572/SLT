"""Factorized-entropy hybrid SAC used only by the v4.2 iteration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as functional
from stable_baselines3.common.utils import polyak_update

from .hybrid_policy_v4 import HybridActionBatch
from .sac import _clip_gradients_like_keras
from .sac_v2 import V2_DIAGNOSTIC_NAMES
from .sac_v4 import DecisionAlignedHybridSACV4


class FactorizedEntropyHybridSACV42(DecisionAlignedHybridSACV4):
    """Apply automatic entropy only to conditional speed, not lane intent.

    The hybrid policy, critic, warm-up, replay, representation learning, and
    action semantics are inherited unchanged from v4.1.  ``lane_entropy_scale``
    is explicit for auditability and is frozen to zero by the v4.2 registry.
    """

    def __init__(
        self,
        *args: Any,
        lane_entropy_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        lane_entropy_scale = float(lane_entropy_scale)
        if not 0.0 <= lane_entropy_scale <= 1.0:
            raise ValueError("lane_entropy_scale must be in [0, 1]")
        self.lane_entropy_scale = lane_entropy_scale
        super().__init__(*args, **kwargs)

    def entropy_log_probabilities(self, batch: HybridActionBatch) -> th.Tensor:
        """Return the per-lane log-probability used by entropy regularization."""
        return (
            batch.speed_log_probabilities
            + self.lane_entropy_scale * batch.lane_log_probabilities
        )

    def expected_entropy_log_probability(
        self, batch: HybridActionBatch
    ) -> th.Tensor:
        return (
            batch.lane_probabilities * self.entropy_log_probabilities(batch)
        ).sum(dim=1, keepdim=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self._pending_raw_gradient_steps is not None:
            gradient_steps = self._pending_raw_gradient_steps
            self._pending_raw_gradient_steps = None
        if gradient_steps <= 0:
            return
        actor, critic, critic_target = self._hybrid_modules()
        train_started = time.perf_counter()
        self.policy.set_training_mode(True)
        optimizers = [actor.optimizer, critic.optimizer]
        if self.representation_optimizer is not None and self.representation_learning_rate is None:
            optimizers.append(self.representation_optimizer)
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        entropy_losses: list[float] = []
        entropy_coefficients: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
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

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.representation is not None:
                representation_loss = self._representation_step(replay_data)
                representation_losses.append(float(representation_loss.cpu()))
                for name in graph_slt_slot_losses:
                    value = self._last_graph_slt_losses.get(name)
                    if value is not None:
                        graph_slt_slot_losses[name].append(float(value.cpu()))

            current_q_values = critic(replay_data.observations, replay_data.actions)
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

            target_extractor = critic_target.features_extractor
            target_was_training = target_extractor.training
            target_extractor.train(True)
            with th.no_grad():
                next_policy = actor.all_action_samples(
                    replay_data.next_observations, deterministic_speed=False
                )
                next_features = critic_target.extract_features(
                    replay_data.next_observations,
                    critic_target.features_extractor,
                )
                next_q_heads = critic_target.all_q_from_features(
                    next_features, next_policy.actions
                )
                next_min_q = th.stack(next_q_heads, dim=-1).min(dim=-1).values
                next_entropy_log_prob = self.entropy_log_probabilities(next_policy)
                next_soft_values = (
                    next_min_q
                    - entropy_coefficient * next_entropy_log_prob.unsqueeze(-1)
                )
                next_value = (
                    next_policy.lane_probabilities.unsqueeze(-1) * next_soft_values
                ).sum(dim=1)
                target_q_values = replay_data.rewards + (
                    1.0 - replay_data.dones
                ) * discounts * next_value
            target_extractor.train(target_was_training)

            critic_loss = 0.5 * sum(
                functional.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.detach().cpu()))

            critic_parameters = list(critic.parameters())
            requires_grad = [parameter.requires_grad for parameter in critic_parameters]
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            actor_q_features = critic.extract_features(
                replay_data.observations, critic.features_extractor
            )
            actor_q_heads = critic.all_q_from_features(
                actor_q_features, policy_batch.actions
            )
            min_q = th.stack(actor_q_heads, dim=-1).min(dim=-1).values
            per_lane_actor_objective = (
                entropy_coefficient
                * self.entropy_log_probabilities(policy_batch).unsqueeze(-1)
                - min_q
            )
            actor_loss = (
                policy_batch.lane_probabilities.unsqueeze(-1)
                * per_lane_actor_objective
            ).sum(dim=1).mean()
            actor_losses.append(float(actor_loss.detach().cpu()))
            for parameter, original_value in zip(critic_parameters, requires_grad):
                parameter.requires_grad_(original_value)

            actor_parameters = [
                parameter
                for parameter in actor.parameters()
                if id(parameter)
                not in {id(item) for item in actor.features_extractor.parameters()}
            ]
            critic.optimizer.zero_grad()
            actor.optimizer.zero_grad()
            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            if entropy_loss is not None:
                entropy_loss.backward()
            _clip_gradients_like_keras(critic_parameters, self.max_grad_norm)
            _clip_gradients_like_keras(actor_parameters, self.max_grad_norm)
            critic.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(critic.parameters(), critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
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
            hybrid_diagnostics["lane_entropy"].append(float(lane_entropy.detach().cpu()))
            hybrid_diagnostics["valid_lane_actions"].append(
                float(policy_batch.action_mask.sum(dim=1).float().mean().cpu())
            )
            for index, name in enumerate(
                ("lane_probability_negative", "lane_probability_keep", "lane_probability_positive")
            ):
                hybrid_diagnostics[name].append(
                    float(policy_batch.lane_probabilities[:, index].mean().detach().cpu())
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

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        scalar_series = {
            "train/ent_coef": entropy_coefficients,
            "train/actor_loss": actor_losses,
            "train/critic_loss": critic_losses,
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


__all__ = ["FactorizedEntropyHybridSACV42"]
