"""Shared-risk, replay-supported mixture policy for the v4.10 model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch as th
from torch import Tensor, nn
from stable_baselines3.common.type_aliases import PyTorchObs

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .features import keras_initialize_linear
from .hybrid_policy_v4 import (
    DecisionAlignedHybridActor,
    HybridActionBatch,
    HybridActionEmbeddedCritic,
)
from .hybrid_policy_v4_9_model import (
    CollisionConstrainedTargetCriticSACPolicyV49,
)


@dataclass(frozen=True)
class SupportedMixtureActionBatch:
    """Every learned lane/speed component and its exact policy mass."""

    actions: Tensor
    lane_probabilities: Tensor
    lane_log_probabilities: Tensor
    component_probabilities: Tensor
    component_log_probabilities: Tensor
    speed_log_probabilities: Tensor
    joint_log_probabilities: Tensor
    action_mask: Tensor
    speed_means: Tensor
    speed_log_stds: Tensor

    @property
    def proposal_probabilities(self) -> Tensor:
        return self.lane_probabilities.unsqueeze(-1) * self.component_probabilities

    @property
    def expected_joint_log_probability(self) -> Tensor:
        return (
            self.proposal_probabilities * self.joint_log_probabilities
        ).sum(dim=(1, 2), keepdim=False).unsqueeze(1)


class SupportedMixtureHybridActor(DecisionAlignedHybridActor):
    """Learn several replay-supported Gaussian speed modes for every lane."""

    def __init__(
        self,
        *args: Any,
        speed_components: int = 3,
        **kwargs: Any,
    ) -> None:
        speed_components = int(speed_components)
        if speed_components <= 0:
            raise ValueError("speed_components must be positive")
        self.speed_components = speed_components
        super().__init__(*args, **kwargs)
        del self.speed_mean
        del self.speed_log_std
        last_dim = self.net_arch[-1] if self.net_arch else self.features_dim
        proposal_count = len(LANE_COMMANDS) * self.speed_components
        self.component_logits = nn.Linear(last_dim, proposal_count)
        self.speed_mean = nn.Linear(last_dim, proposal_count)
        self.speed_log_std = nn.Linear(last_dim, proposal_count)
        self.component_logits.apply(keras_initialize_linear)
        self.speed_mean.apply(keras_initialize_linear)
        self.speed_log_std.apply(keras_initialize_linear)

    def proposal_distribution_parameters(
        self, obs: PyTorchObs
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # The shared encoder is trained by reward, collision, and Graph-SLT
        # losses through its single owner. Actor/support heads consume its
        # risk-shaped representation without taking encoder ownership.
        features = self.extract_features(obs, self.features_extractor).detach()
        latent = self.latent_pi(features)
        action_mask = self._mask_from_observation(obs)
        raw_lane_logits = self.lane_logits(latent)
        masked_lane_logits = raw_lane_logits.masked_fill(~action_mask, -1e9)
        lane_log_probabilities = th.log_softmax(masked_lane_logits, dim=1)
        lane_probabilities = th.softmax(masked_lane_logits, dim=1)
        shape = (-1, len(LANE_COMMANDS), self.speed_components)
        raw_component_logits = self.component_logits(latent).reshape(shape)
        component_log_probabilities = th.log_softmax(
            raw_component_logits, dim=2
        )
        component_probabilities = th.softmax(raw_component_logits, dim=2)
        speed_means = self.speed_mean(latent).reshape(shape)
        speed_log_stds = th.clamp(
            self.speed_log_std(latent).reshape(shape),
            -20.0,
            2.0,
        )
        return (
            lane_probabilities,
            lane_log_probabilities,
            component_probabilities,
            component_log_probabilities,
            speed_means,
            speed_log_stds,
            action_mask,
            latent,
        )

    def all_action_proposals(
        self, obs: PyTorchObs, *, deterministic_speed: bool = False
    ) -> SupportedMixtureActionBatch:
        (
            lane_probabilities,
            lane_log_probabilities,
            component_probabilities,
            component_log_probabilities,
            speed_means,
            speed_log_stds,
            action_mask,
            _,
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
        return SupportedMixtureActionBatch(
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
        )

    @staticmethod
    def _select_proposals(
        values: Tensor, lane_indices: Tensor, component_indices: Tensor
    ) -> Tensor:
        rows = th.arange(values.shape[0], device=values.device)
        return values[rows, lane_indices, component_indices]

    def _selected_component_indices(
        self, probabilities: Tensor, *, deterministic: bool
    ) -> Tensor:
        if deterministic:
            return probabilities.argmax(dim=2)
        flat = probabilities.reshape(-1, self.speed_components)
        return th.multinomial(flat, 1).reshape(
            probabilities.shape[0], len(LANE_COMMANDS)
        )

    def all_action_samples(
        self, obs: PyTorchObs, *, deterministic_speed: bool = False
    ) -> HybridActionBatch:
        """Compatibility view containing one sampled component per lane."""

        batch = self.all_action_proposals(
            obs, deterministic_speed=deterministic_speed
        )
        indices = self._selected_component_indices(
            batch.component_probabilities,
            deterministic=deterministic_speed,
        )
        gather_action = indices[:, :, None, None].expand(-1, -1, 1, 2)
        actions = batch.actions.gather(2, gather_action).squeeze(2)
        gather_scalar = indices.unsqueeze(-1)
        component_log = batch.component_log_probabilities.gather(
            2, gather_scalar
        ).squeeze(2)
        speed_log = batch.speed_log_probabilities.gather(
            2, gather_scalar
        ).squeeze(2)
        means = batch.speed_means.gather(2, gather_scalar).squeeze(2)
        log_stds = batch.speed_log_stds.gather(2, gather_scalar).squeeze(2)
        conditional_log = component_log + speed_log
        return HybridActionBatch(
            actions=actions,
            lane_probabilities=batch.lane_probabilities,
            lane_log_probabilities=batch.lane_log_probabilities,
            speed_log_probabilities=conditional_log,
            joint_log_probabilities=(
                batch.lane_log_probabilities + conditional_log
            ),
            action_mask=batch.action_mask,
            speed_means=means,
            speed_log_stds=log_stds,
        )

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tensor:
        batch = self.all_action_proposals(
            obs, deterministic_speed=deterministic
        )
        lane_indices = (
            batch.lane_probabilities.argmax(dim=1)
            if deterministic
            else th.multinomial(batch.lane_probabilities, 1).squeeze(1)
        )
        rows = th.arange(batch.actions.shape[0], device=batch.actions.device)
        selected_component_probabilities = batch.component_probabilities[
            rows, lane_indices
        ]
        component_indices = (
            selected_component_probabilities.argmax(dim=1)
            if deterministic
            else th.multinomial(selected_component_probabilities, 1).squeeze(1)
        )
        return self._select_proposals(
            batch.actions, lane_indices, component_indices
        )

    def action_log_prob(self, obs: PyTorchObs) -> tuple[Tensor, Tensor]:
        batch = self.all_action_proposals(obs, deterministic_speed=False)
        lane_indices = th.multinomial(batch.lane_probabilities, 1).squeeze(1)
        rows = th.arange(batch.actions.shape[0], device=batch.actions.device)
        selected_component_probabilities = batch.component_probabilities[
            rows, lane_indices
        ]
        component_indices = th.multinomial(
            selected_component_probabilities, 1
        ).squeeze(1)
        actions = self._select_proposals(
            batch.actions, lane_indices, component_indices
        )
        log_probability = batch.joint_log_probabilities[
            rows, lane_indices, component_indices
        ]
        return actions, log_probability

    def replay_support_nll(
        self,
        batch: SupportedMixtureActionBatch,
        replay_actions: Tensor,
    ) -> Tensor:
        """Latent-speed mixture NLL on actions actually present in replay."""

        if replay_actions.ndim != 2 or replay_actions.shape[1] != 2:
            raise ValueError("replay actions must have shape [batch,2]")
        lateral = replay_actions[:, 1]
        lane_indices = th.where(
            lateral < -1.0 / 3.0,
            th.zeros_like(lateral, dtype=th.long),
            th.where(
                lateral > 1.0 / 3.0,
                th.full_like(lateral, 2, dtype=th.long),
                th.ones_like(lateral, dtype=th.long),
            ),
        )
        rows = th.arange(replay_actions.shape[0], device=replay_actions.device)
        means = batch.speed_means[rows, lane_indices]
        log_stds = batch.speed_log_stds[rows, lane_indices]
        component_log = batch.component_log_probabilities[rows, lane_indices]
        speed = replay_actions[:, 0].clamp(-0.999999, 0.999999)
        latent_speed = th.atanh(speed).unsqueeze(1)
        std = log_stds.exp().clamp_min(1e-12)
        normal_log = -0.5 * (
            ((latent_speed - means) / std).square()
            + 2.0 * log_stds
            + math.log(2.0 * math.pi)
        )
        mixture_log_likelihood = th.logsumexp(
            component_log + normal_log, dim=1
        )
        return -mixture_log_likelihood.mean()


def all_proposal_q_from_features(
    critic: HybridActionEmbeddedCritic,
    features: Tensor,
    actions: Tensor,
) -> tuple[Tensor, ...]:
    if actions.ndim != 4 or actions.shape[1] != len(LANE_COMMANDS) or actions.shape[-1] != 2:
        raise ValueError(
            "proposal actions must have shape [batch,3,components,2], got "
            f"{tuple(actions.shape)}"
        )
    batch, lanes, components, _ = actions.shape
    expanded = (
        features[:, None, None, :]
        .expand(-1, lanes, components, -1)
        .reshape(batch * lanes * components, -1)
    )
    flattened = actions.reshape(batch * lanes * components, 2)
    values = critic.forward_from_features(expanded, flattened)
    return tuple(value.reshape(batch, lanes, components, 1) for value in values)


class SharedRiskSupportedMixtureSACPolicyV410(
    CollisionConstrainedTargetCriticSACPolicyV49
):
    """Target-critic decoder over learned, support-weighted speed proposals."""

    deterministic_lane_decoder = (
        "argmax_feasible_learned_supported_mixture_risk_uncertainty_score"
    )
    learned_collision_critic = True
    inference_safety_rule_added = False

    def __init__(
        self,
        *args: Any,
        speed_components: int = 3,
        twin_uncertainty_coef: float = 0.25,
        component_prior_coef: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.speed_components = int(speed_components)
        self.twin_uncertainty_coef = float(twin_uncertainty_coef)
        self.component_prior_coef = float(component_prior_coef)
        if self.speed_components <= 0:
            raise ValueError("speed_components must be positive")
        if self.twin_uncertainty_coef < 0.0 or self.component_prior_coef < 0.0:
            raise ValueError("uncertainty and component-prior coefficients must be non-negative")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _parameters_excluding(
        module: nn.Module, excluded: set[int]
    ) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in excluded
        ]

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        shared_online = self.critic.features_extractor
        shared_target = self.critic_target.features_extractor

        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, shared_online
        )
        self.actor = SupportedMixtureHybridActor(
            **actor_kwargs,
            speed_components=self.speed_components,
        ).to(self.device)
        self.actor.latent_pi.apply(keras_initialize_linear)

        collision_kwargs = self._update_features_extractor(
            self.critic_kwargs, shared_online
        )
        collision_kwargs["share_features_extractor"] = False
        self.collision_critic = HybridActionEmbeddedCritic(
            **collision_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)
        collision_target_kwargs = self._update_features_extractor(
            self.critic_kwargs, shared_target
        )
        collision_target_kwargs["share_features_extractor"] = False
        self.collision_critic_target = HybridActionEmbeddedCritic(
            **collision_target_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)
        self.collision_critic_target.load_state_dict(
            self.collision_critic.state_dict()
        )
        self.collision_critic_target.set_training_mode(False)

        encoder_ids = {id(parameter) for parameter in shared_online.parameters()}
        self.encoder_optimizer = self.optimizer_class(
            list(shared_online.parameters()),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        actor_parameters = self._parameters_excluding(self.actor, encoder_ids)
        reward_head_parameters = self._parameters_excluding(
            self.critic, encoder_ids
        )
        collision_head_parameters = self._parameters_excluding(
            self.collision_critic, encoder_ids
        )
        self.actor.optimizer = self.optimizer_class(
            actor_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic.optimizer = self.optimizer_class(
            reward_head_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.collision_critic.optimizer = self.optimizer_class(
            collision_head_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.share_features_extractor = True
        ownership = self.optimizer_parameter_ownership()
        if ownership["overlap_count"] != 0:
            raise RuntimeError("v4.10 optimizer parameter ownership overlaps")

    def optimizer_parameter_ownership(self) -> dict[str, Any]:
        groups = {
            "encoder": list(self.critic.features_extractor.parameters()),
            "actor_heads": list(self.actor.optimizer.param_groups[0]["params"]),
            "reward_heads": list(self.critic.optimizer.param_groups[0]["params"]),
            "collision_heads": list(
                self.collision_critic.optimizer.param_groups[0]["params"]
            ),
        }
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
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
            "online_encoder_shared_by_identity": (
                self.actor.features_extractor
                is self.critic.features_extractor
                is self.collision_critic.features_extractor
            ),
            "target_encoder_shared_by_identity": (
                self.critic_target.features_extractor
                is self.collision_critic_target.features_extractor
            ),
        }

    def _risk_adjusted_proposal_values(
        self,
        observation: PyTorchObs,
        batch: SupportedMixtureActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        reward_target = self.critic_target
        shared_features = reward_target.extract_features(
            observation, reward_target.features_extractor
        )
        reward_heads = all_proposal_q_from_features(
            reward_target, shared_features, batch.actions
        )
        collision_heads = all_proposal_q_from_features(
            self.collision_critic_target, shared_features, batch.actions
        )
        reward_values = th.stack(reward_heads, dim=-1).squeeze(-2)
        collision_values = th.stack(
            [th.sigmoid(head) for head in collision_heads], dim=-1
        ).squeeze(-2)
        minimum_reward_q = reward_values.min(dim=-1).values
        maximum_collision_value = collision_values.max(dim=-1).values
        reward_disagreement = (reward_values[..., 0] - reward_values[..., 1]).abs()
        collision_disagreement = (
            collision_values[..., 0] - collision_values[..., 1]
        ).abs()
        learned_uncertainty = reward_disagreement + collision_disagreement
        score = (
            minimum_reward_q
            - self.collision_risk_coef * maximum_collision_value
            - self.twin_uncertainty_coef * learned_uncertainty
            + self.component_prior_coef * batch.component_log_probabilities
        )
        return (
            minimum_reward_q,
            maximum_collision_value,
            reward_disagreement,
            collision_disagreement,
            learned_uncertainty,
            score,
        )

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tensor:
        actor = self.actor
        if not isinstance(actor, SupportedMixtureHybridActor):
            raise TypeError("v4.10 requires SupportedMixtureHybridActor")
        if not deterministic:
            return actor(observation, deterministic=False)
        batch = actor.all_action_proposals(
            observation, deterministic_speed=True
        )
        (
            minimum_reward_q,
            maximum_collision_value,
            reward_disagreement,
            collision_disagreement,
            learned_uncertainty,
            score,
        ) = self._risk_adjusted_proposal_values(observation, batch)
        feasible = batch.action_mask.unsqueeze(-1).expand_as(score)
        masked = score.masked_fill(~feasible, -th.inf)
        flattened = masked.reshape(masked.shape[0], -1)
        selected_flat = flattened.argmax(dim=1)
        selected_lanes = selected_flat // self.speed_components
        selected_components = selected_flat % self.speed_components
        selected_actions = actor._select_proposals(
            batch.actions, selected_lanes, selected_components
        )

        if self._record_target_decoder:
            for row in range(selected_actions.shape[0]):
                valid = batch.action_mask[row].detach().cpu().tolist()
                lane_index = int(selected_lanes[row])
                component_index = int(selected_components[row])

                def masked_rows(values: Tensor) -> list[list[float] | None]:
                    raw = values[row].detach().cpu().tolist()
                    return [
                        [float(item) for item in lane] if is_valid else None
                        for lane, is_valid in zip(raw, valid)
                    ]

                self.target_decoder_records.append(
                    {
                        "lane_probabilities": batch.lane_probabilities[row]
                        .detach()
                        .cpu()
                        .tolist(),
                        "component_probabilities": batch.component_probabilities[row]
                        .detach()
                        .cpu()
                        .tolist(),
                        "valid_lane_actions": valid,
                        "learned_speed_proposals_normalized": batch.actions[
                            row, :, :, 0
                        ]
                        .detach()
                        .cpu()
                        .tolist(),
                        "minimum_target_twin_q": masked_rows(minimum_reward_q),
                        "maximum_target_twin_collision_value": masked_rows(
                            maximum_collision_value
                        ),
                        "reward_twin_disagreement": masked_rows(
                            reward_disagreement
                        ),
                        "collision_twin_disagreement": masked_rows(
                            collision_disagreement
                        ),
                        "learned_uncertainty": masked_rows(learned_uncertainty),
                        "supported_risk_adjusted_score": masked_rows(score),
                        "collision_risk_coef": self.collision_risk_coef,
                        "twin_uncertainty_coef": self.twin_uncertainty_coef,
                        "component_prior_coef": self.component_prior_coef,
                        "selected_lane_index": lane_index,
                        "selected_lane": int(LANE_COMMANDS[lane_index]),
                        "selected_component_index": component_index,
                        "selection_operator": (
                            "torch_argmax_flattened_feasible_model_scores"
                        ),
                        "semantic_tie_override_used": False,
                        "candidate_source": "learned_actor_component_means",
                        "fixed_speed_grid_used": False,
                        "action_rewritten": False,
                    }
                )
        return selected_actions


__all__ = [
    "SharedRiskSupportedMixtureSACPolicyV410",
    "SupportedMixtureActionBatch",
    "SupportedMixtureHybridActor",
    "all_proposal_q_from_features",
]
