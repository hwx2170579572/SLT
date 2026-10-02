"""SB3 policy modules that preserve the released actor/critic data flow."""

from __future__ import annotations

from typing import Any

import torch as th
from torch import nn

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN, SACPolicy

from .features import keras_initialize_linear


class DetachedSceneActor(Actor):
    """Actor head fed by the critic encoder with stop-gradient semantics."""

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor).detach()
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return mean_actions, self.log_std, {"latent_sde": latent_pi}
        log_std = th.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class ActionEmbeddedContinuousCritic(ContinuousCritic):
    """Twin-Q critic with a separate 64-D action branch for each Q head."""

    def __init__(self, *args: Any, action_embedding_dim: int = 64, **kwargs: Any) -> None:
        net_arch = list(kwargs["net_arch"])
        activation_fn = kwargs["activation_fn"]
        feature_dim = int(kwargs["features_dim"])
        super().__init__(*args, **kwargs)
        for index in range(self.n_critics):
            delattr(self, f"qf{index}")
        action_dim = int(self.action_space.shape[0])
        self.action_encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(action_dim, action_embedding_dim), activation_fn())
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

    def forward(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        features = self.extract_features(obs, self.features_extractor)
        return tuple(
            q_network(th.cat([features, action_encoder(actions)], dim=1))
            for q_network, action_encoder in zip(self.q_networks, self.action_encoders)
        )

    def q1_forward(self, obs: PyTorchObs, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](
            th.cat([features, self.action_encoders[0](actions)], dim=1)
        )


class SceneRepSACPolicy(SACPolicy):
    """SAC policy whose actor consumes a detached, critic-owned scene encoder."""

    def __init__(self, *args: Any, action_embedding_dim: int = 64, **kwargs: Any) -> None:
        self.action_embedding_dim = int(action_embedding_dim)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        shared_extractor = self.make_features_extractor()
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, shared_extractor)
        self.actor = DetachedSceneActor(**actor_kwargs).to(self.device)
        self.actor.latent_pi.apply(keras_initialize_linear)
        self.actor.mu.apply(keras_initialize_linear)
        if isinstance(self.actor.log_std, nn.Module):
            self.actor.log_std.apply(keras_initialize_linear)
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
        self.critic = ActionEmbeddedContinuousCritic(
            **critic_kwargs,
            action_embedding_dim=self.action_embedding_dim,
        ).to(self.device)

        target_kwargs = self._update_features_extractor(self.critic_kwargs, None)
        target_kwargs["share_features_extractor"] = False
        self.critic_target = ActionEmbeddedContinuousCritic(
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

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["action_embedding_dim"] = self.action_embedding_dim
        return data
