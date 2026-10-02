"""Masked categorical-lane/conditional-speed policy used only by v4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch as th
from torch import Tensor, nn

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN

from envs.sumo.decision_alignment_v4 import LANE_COMMANDS

from .features import keras_initialize_linear
from .policies import SceneRepSACPolicy


@dataclass(frozen=True)
class HybridActionBatch:
    """All three lane actions and their exact mixture probabilities."""

    actions: Tensor
    lane_probabilities: Tensor
    lane_log_probabilities: Tensor
    speed_log_probabilities: Tensor
    joint_log_probabilities: Tensor
    action_mask: Tensor
    speed_means: Tensor
    speed_log_stds: Tensor

    @property
    def expected_log_probability(self) -> Tensor:
        return (
            self.lane_probabilities * self.joint_log_probabilities
        ).sum(dim=1, keepdim=True)


def hybrid_action_features(actions: Tensor) -> Tensor:
    """Convert ``[normalised speed, lateral]`` to speed plus lane one-hot.

    The threshold semantics exactly match ``SumoSceneEnv.adapt_action``.  This
    also makes historical/random warm-up actions well-defined, although the v4
    learner samples exact lane codes during warm-up.
    """

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"hybrid actions must have shape [batch,2], got {tuple(actions.shape)}")
    lateral = actions[:, 1]
    negative = lateral < -1.0 / 3.0
    positive = lateral > 1.0 / 3.0
    keep = ~(negative | positive)
    lane_one_hot = th.stack([negative, keep, positive], dim=1).to(actions.dtype)
    return th.cat([actions[:, :1], lane_one_hot], dim=1)


class DecisionAlignedHybridActor(Actor):
    """Categorical lane intent and lane-conditioned squashed-Gaussian speed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if bool(kwargs.get("use_sde", False)):
            raise ValueError("DecisionAlignedHybridActor does not support gSDE")
        super().__init__(*args, **kwargs)
        # Remove the continuous 2-D heads created by Actor.  Retaining them as
        # unused parameters would silently change optimization and checkpoints.
        del self.mu
        del self.log_std
        del self.action_dist
        last_dim = self.net_arch[-1] if self.net_arch else self.features_dim
        self.lane_logits = nn.Linear(last_dim, len(LANE_COMMANDS))
        self.speed_mean = nn.Linear(last_dim, len(LANE_COMMANDS))
        self.speed_log_std = nn.Linear(last_dim, len(LANE_COMMANDS))
        self.register_buffer(
            "lane_codes",
            th.as_tensor(LANE_COMMANDS, dtype=th.float32),
        )
        self.lane_logits.apply(keras_initialize_linear)
        self.speed_mean.apply(keras_initialize_linear)
        self.speed_log_std.apply(keras_initialize_linear)

    @staticmethod
    def _mask_from_observation(obs: PyTorchObs) -> Tensor:
        if not isinstance(obs, Mapping) or "lane_action_mask" not in obs:
            raise KeyError("v4 actor requires observation['lane_action_mask']")
        mask = obs["lane_action_mask"] > 0.5
        if mask.ndim != 2 or mask.shape[1] != len(LANE_COMMANDS):
            raise ValueError(
                "lane_action_mask must have shape [batch,3], got "
                f"{tuple(mask.shape)}"
            )
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("every sample must expose at least one feasible lane action")
        return mask

    def distribution_parameters(
        self, obs: PyTorchObs
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.extract_features(obs, self.features_extractor).detach()
        latent = self.latent_pi(features)
        raw_logits = self.lane_logits(latent)
        action_mask = self._mask_from_observation(obs)
        masked_logits = raw_logits.masked_fill(~action_mask, -1e9)
        lane_log_probabilities = th.log_softmax(masked_logits, dim=1)
        lane_probabilities = th.softmax(masked_logits, dim=1)
        speed_means = self.speed_mean(latent)
        speed_log_stds = th.clamp(
            self.speed_log_std(latent), LOG_STD_MIN, LOG_STD_MAX
        )
        return (
            lane_probabilities,
            lane_log_probabilities,
            speed_means,
            speed_log_stds,
            action_mask,
            latent,
        )

    @staticmethod
    def _sample_squashed_speeds(
        means: Tensor,
        log_stds: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        gaussian = means if deterministic else means + log_stds.exp() * th.randn_like(means)
        speeds = th.tanh(gaussian)
        normal_log_prob = -0.5 * (
            ((gaussian - means) / log_stds.exp().clamp_min(1e-12)).square()
            + 2.0 * log_stds
            + math.log(2.0 * math.pi)
        )
        squash_correction = th.log(1.0 - speeds.square() + 1e-6)
        return speeds, normal_log_prob - squash_correction

    def all_action_samples(
        self, obs: PyTorchObs, *, deterministic_speed: bool = False
    ) -> HybridActionBatch:
        (
            lane_probabilities,
            lane_log_probabilities,
            speed_means,
            speed_log_stds,
            action_mask,
            _,
        ) = self.distribution_parameters(obs)
        speeds, speed_log_probabilities = self._sample_squashed_speeds(
            speed_means,
            speed_log_stds,
            deterministic=deterministic_speed,
        )
        lane_codes = self.lane_codes.to(speeds).view(1, -1).expand_as(speeds)
        actions = th.stack([speeds, lane_codes], dim=-1)
        joint_log_probabilities = lane_log_probabilities + speed_log_probabilities
        return HybridActionBatch(
            actions=actions,
            lane_probabilities=lane_probabilities,
            lane_log_probabilities=lane_log_probabilities,
            speed_log_probabilities=speed_log_probabilities,
            joint_log_probabilities=joint_log_probabilities,
            action_mask=action_mask,
            speed_means=speed_means,
            speed_log_stds=speed_log_stds,
        )

    @staticmethod
    def _select(values: Tensor, indices: Tensor) -> Tensor:
        gather = indices[:, None, None].expand(-1, 1, values.shape[-1])
        return values.gather(1, gather).squeeze(1)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tensor:
        batch = self.all_action_samples(obs, deterministic_speed=deterministic)
        if deterministic:
            indices = batch.lane_probabilities.argmax(dim=1)
        else:
            indices = th.multinomial(batch.lane_probabilities, 1).squeeze(1)
        return self._select(batch.actions, indices)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[Tensor, Tensor]:
        batch = self.all_action_samples(obs, deterministic_speed=False)
        indices = th.multinomial(batch.lane_probabilities, 1).squeeze(1)
        actions = self._select(batch.actions, indices)
        log_prob = batch.joint_log_probabilities.gather(1, indices[:, None]).squeeze(1)
        return actions, log_prob

    def get_action_dist_params(self, obs: PyTorchObs):
        del obs
        raise NotImplementedError(
            "The v4 actor has a hybrid distribution; use distribution_parameters()"
        )


class HybridActionEmbeddedCritic(ContinuousCritic):
    """Twin Q networks over continuous speed and categorical lane one-hot."""

    HYBRID_ACTION_DIM = 4

    def __init__(self, *args: Any, action_embedding_dim: int = 64, **kwargs: Any) -> None:
        net_arch = list(kwargs["net_arch"])
        activation_fn = kwargs["activation_fn"]
        feature_dim = int(kwargs["features_dim"])
        super().__init__(*args, **kwargs)
        for index in range(self.n_critics):
            delattr(self, f"qf{index}")
        self.action_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.HYBRID_ACTION_DIM, action_embedding_dim),
                    activation_fn(),
                )
                for _ in range(self.n_critics)
            ]
        )
        self.q_networks = []
        for index in range(self.n_critics):
            network = nn.Sequential(
                *create_mlp(
                    feature_dim + action_embedding_dim,
                    1,
                    net_arch,
                    activation_fn,
                )
            )
            self.add_module(f"qf{index}", network)
            self.q_networks.append(network)
        self.action_encoders.apply(keras_initialize_linear)
        for network in self.q_networks:
            network.apply(keras_initialize_linear)

    def forward_from_features(
        self, features: Tensor, actions: Tensor
    ) -> tuple[Tensor, ...]:
        encoded = hybrid_action_features(actions)
        return tuple(
            q_network(th.cat([features, action_encoder(encoded)], dim=1))
            for q_network, action_encoder in zip(self.q_networks, self.action_encoders)
        )

    def all_q_from_features(
        self, features: Tensor, actions: Tensor
    ) -> tuple[Tensor, ...]:
        if actions.ndim != 3 or actions.shape[1:] != (len(LANE_COMMANDS), 2):
            raise ValueError(
                "enumerated actions must have shape [batch,3,2], got "
                f"{tuple(actions.shape)}"
            )
        batch = features.shape[0]
        expanded_features = (
            features[:, None, :]
            .expand(-1, len(LANE_COMMANDS), -1)
            .reshape(batch * len(LANE_COMMANDS), -1)
        )
        flattened_actions = actions.reshape(batch * len(LANE_COMMANDS), 2)
        values = self.forward_from_features(expanded_features, flattened_actions)
        return tuple(value.reshape(batch, len(LANE_COMMANDS), 1) for value in values)

    def forward(self, obs: PyTorchObs, actions: Tensor) -> tuple[Tensor, ...]:
        features = self.extract_features(obs, self.features_extractor)
        return self.forward_from_features(features, actions)

    def q1_forward(self, obs: PyTorchObs, actions: Tensor) -> Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.forward_from_features(features, actions)[0]


class DecisionAlignedSACPolicy(SceneRepSACPolicy):
    """Shared stop-gradient encoder with v4 hybrid actor and critic."""

    def _build(self, lr_schedule: Schedule) -> None:
        shared_extractor = self.make_features_extractor()
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, shared_extractor)
        self.actor = DecisionAlignedHybridActor(**actor_kwargs).to(self.device)
        self.actor.latent_pi.apply(keras_initialize_linear)
        extractor_parameters = {id(parameter) for parameter in shared_extractor.parameters()}
        actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if id(parameter) not in extractor_parameters
        ]
        self.actor.optimizer = self.optimizer_class(
            actor_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        critic_kwargs = self._update_features_extractor(self.critic_kwargs, shared_extractor)
        critic_kwargs["share_features_extractor"] = False
        self.critic = HybridActionEmbeddedCritic(
            **critic_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)

        target_kwargs = self._update_features_extractor(self.critic_kwargs, None)
        target_kwargs["share_features_extractor"] = False
        self.critic_target = HybridActionEmbeddedCritic(
            **target_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic_target.set_training_mode(False)
        self.share_features_extractor = True


__all__ = [
    "DecisionAlignedHybridActor",
    "DecisionAlignedSACPolicy",
    "HybridActionBatch",
    "HybridActionEmbeddedCritic",
    "hybrid_action_features",
]

