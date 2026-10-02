"""SB3 PPO preserving the released TensorFlow PPO policy/data-flow contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .features import (
    KerasStyleMultiheadAttention,
    _nonzero_mask,
    keras_initialize_linear,
)


class SourcePpoCnnExtractor(BaseFeaturesExtractor):
    """The four valid-convolution layers in ``GaussianActorCritic``."""

    def __init__(self, observation_space: spaces.Box) -> None:
        if tuple(observation_space.shape) not in ((80, 80, 3), (3, 80, 80)):
            raise ValueError(
                "The released SMARTS PPO expects one 80x80x3 RGB frame"
            )
        super().__init__(observation_space, features_dim=256)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.ndim != 4:
            raise ValueError(f"Expected batched RGB observations, got {observations.shape}")
        if observations.shape[-1] == 3:
            observations = observations.permute(0, 3, 1, 2)
        return self.cnn(observations.float())


class KerasGru(nn.Module):
    """Keras GRU(reset_after=True) with its default initializers."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel = nn.Parameter(th.empty(self.input_dim, 3 * self.hidden_dim))
        self.recurrent_kernel = nn.Parameter(
            th.empty(self.hidden_dim, 3 * self.hidden_dim)
        )
        self.input_bias = nn.Parameter(th.zeros(3 * self.hidden_dim))
        self.recurrent_bias = nn.Parameter(th.zeros(3 * self.hidden_dim))
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)

    def forward(
        self, sequence: th.Tensor, mask: th.Tensor | None = None
    ) -> th.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch,time,{self.input_dim}], got {tuple(sequence.shape)}"
            )
        if mask is not None and tuple(mask.shape) != tuple(sequence.shape[:2]):
            raise ValueError(
                "GRU mask must match [batch,time], got "
                f"{tuple(mask.shape)} for {tuple(sequence.shape)}"
            )
        hidden = th.zeros(
            sequence.shape[0],
            self.hidden_dim,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        for timestep in range(sequence.shape[1]):
            projected = sequence[:, timestep] @ self.kernel + self.input_bias
            recurrent = hidden @ self.recurrent_kernel + self.recurrent_bias
            x_z, x_r, x_h = projected.chunk(3, dim=-1)
            recurrent_z, recurrent_r, recurrent_h = recurrent.chunk(3, dim=-1)
            z = th.sigmoid(x_z + recurrent_z)
            r = th.sigmoid(x_r + recurrent_r)
            candidate = th.tanh(x_h + r * recurrent_h)
            updated = z * hidden + (1.0 - z) * candidate
            if mask is None:
                hidden = updated
            else:
                valid = mask[:, timestep].bool().unsqueeze(-1)
                hidden = th.where(valid, updated, hidden)
        return hidden


class SourcePpoCarlaExtractor(BaseFeaturesExtractor):
    """Released CARLA PPO ``Ego_Neighbours_Encoder`` (including quirks)."""

    def __init__(self, observation_space: spaces.Box) -> None:
        if tuple(observation_space.shape) != (6, 10, 5):
            raise ValueError("The released CARLA PPO expects (6,10,5) trajectories")
        super().__init__(observation_space, features_dim=512)
        self.gru = KerasGru(5, 256)
        self.relation_attention = KerasStyleMultiheadAttention(
            256,
            256,
            256,
            num_heads=6,
            head_dim=256 // 6,
            output_dim=256,
            dropout=0.1,
        )
        # The source creates [LayerNormalization()] * 2, so both calls share
        # the same gamma/beta variables. Keras LayerNorm defaults epsilon=1e-3.
        self.shared_norm = nn.LayerNorm(256, eps=1e-3)
        self.ffn_1 = nn.Linear(256, 1024)
        self.ffn_2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.1)
        self.ego_norm = nn.LayerNorm(256, eps=1e-3)
        self.ffn_1.apply(keras_initialize_linear)
        self.ffn_2.apply(keras_initialize_linear)

    @staticmethod
    def _wrap_to_pi(values: th.Tensor) -> th.Tensor:
        return th.remainder(values + th.pi, 2.0 * th.pi) - th.pi

    def forward(self, observations: th.Tensor) -> th.Tensor:
        states = observations.float()
        if states.ndim != 4 or tuple(states.shape[1:]) != (6, 10, 5):
            raise ValueError(f"Expected [batch,6,10,5], got {tuple(states.shape)}")
        # make_rotation=False in set_on_policy_configs().  Preserve the source
        # field reinterpretation: [x,y,heading,vx,vy] becomes
        # [x,y,heading,vx,wrap(vy)].
        encoded_states = th.stack(
            (
                states[..., 0],
                states[..., 1],
                states[..., 2],
                states[..., 3],
                self._wrap_to_pi(states[..., 4]),
            ),
            dim=-1,
        )
        actor_features = th.stack(
            [self.gru(encoded_states[:, actor]) for actor in range(6)], dim=1
        )
        ego = actor_features[:, 0]
        actor_mask = th.cat(
            (
                th.ones(
                    (states.shape[0], 1), dtype=th.bool, device=states.device
                ),
                _nonzero_mask(encoded_states[:, 1:, 0]),
            ),
            dim=1,
        )
        related, _ = self.relation_attention(
            ego[:, None],
            actor_features,
            actor_features,
            attention_mask=actor_mask[:, None],
        )
        value = self.shared_norm(related[:, 0])
        value = self.dropout(th.nn.functional.elu(self.ffn_1(value)))
        value = self.dropout(self.ffn_2(value))
        value = self.shared_norm(value)
        return th.cat((value, self.ego_norm(ego)), dim=-1)


def _keras_initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        keras_initialize_linear(module)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SourcePpoPolicy(ActorCriticPolicy):
    """Shared CNN PPO with the source's state-dependent squashed Gaussian.

    The released collection path stores tanh-squashed actions and their
    change-of-variables log probability.  Its update path then evaluates those
    stored actions directly under the unsquashed Normal and omits the Jacobian.
    That asymmetry is preserved because it materially changes PPO's ratio.
    """

    log_std_min = -20.0
    log_std_max = 2.0
    squash_epsilon = 1e-6

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("net_arch", {"pi": [128, 32], "vf": [128, 32]})
        kwargs.setdefault("activation_fn", nn.ReLU)
        kwargs.setdefault("ortho_init", False)
        kwargs.setdefault("features_extractor_class", SourcePpoCnnExtractor)
        kwargs.setdefault("share_features_extractor", True)
        kwargs.setdefault("normalize_images", False)
        kwargs.setdefault("optimizer_class", th.optim.Adam)
        kwargs.setdefault("optimizer_kwargs", {"eps": 1e-7})
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if not isinstance(action_space, spaces.Box) or action_space.shape != (2,):
            raise ValueError("The released PPO uses a two-dimensional Box action")

        # ActorCriticPolicy constructs a state-independent log_std Parameter.
        # Remove it and use the released Dense(action_dim) head instead.
        if hasattr(self, "log_std"):
            delattr(self, "log_std")
        self.state_log_std_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 2)
        self.apply(_keras_initialize_module)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _latents(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if not isinstance(features, th.Tensor):
            raise TypeError("SourcePpoPolicy requires a shared feature extractor")
        # GaussianActorCritic.compute_log_probs()/compute_entropy() applies
        # tf.stop_gradient to the shared encoded state, while the value path
        # in GaussianActorCritic.call() does not.  Consequently the released
        # shared CNN/GRU encoder is trained only by the critic loss even though
        # actor and critic use the same encoded features.
        latent_pi = self.mlp_extractor.forward_actor(features.detach())
        latent_vf = self.mlp_extractor.forward_critic(features)
        return latent_pi, latent_vf

    def _normal(self, latent_pi: th.Tensor) -> tuple[Normal, th.Tensor]:
        mean = self.action_net(latent_pi)
        log_std = th.clamp(
            self.state_log_std_net(latent_pi), self.log_std_min, self.log_std_max
        )
        return Normal(mean, th.exp(log_std)), mean

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        distribution, mean = self._normal(latent_pi)
        raw_actions = mean if deterministic else distribution.sample()
        actions = th.tanh(raw_actions)
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1)
        log_prob -= th.log(1.0 - actions.square() + self.squash_epsilon).sum(
            dim=-1
        )
        return actions.reshape((-1, 2)), values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._latents(obs)
        distribution, _ = self._normal(latent_pi)
        # Deliberately do not invert tanh or subtract its Jacobian: this is the
        # released GaussianActorCritic.compute_log_probs implementation.
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        latent_pi, _ = self._latents(observation)
        distribution, mean = self._normal(latent_pi)
        raw_actions = mean if deterministic else distribution.sample()
        return th.tanh(raw_actions)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        _, latent_vf = self._latents(obs)
        return self.value_net(latent_vf)


class SourcePPO(PPO):
    """PPO with source-global advantages and source horizon bootstrapping."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.source_action_hold = int(kwargs.pop("source_action_hold", 1))
        if self.source_action_hold <= 0:
            raise ValueError("source_action_hold must be positive")
        self._source_episode_step = 0
        self._source_cached_rollout: tuple[th.Tensor, th.Tensor, th.Tensor] | None = None
        kwargs["normalize_advantage"] = False
        kwargs.setdefault("max_grad_norm", float("inf"))
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        # The source computes mean/std once over all 512 transitions, then
        # shuffles that already-normalized array for every epoch.
        advantages = self.rollout_buffer.advantages
        advantages[...] = (
            advantages - float(np.mean(advantages))
        ) / (float(np.std(advantages)) + 1e-8)
        super().train()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect like SB3 while retaining two released runner behaviors.

        Time-limit rewards are not bootstrapped, and at a nonterminal 512-step
        boundary the source passes the value of the final *current* state
        rather than evaluating the next observation.
        """

        assert self._last_obs is not None
        # Source get_action_and_val(..., test=False) keeps MHA dropout active
        # during collection. The SMARTS CNN has no training-mode layers.
        self.policy.set_training_mode(True)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            should_decide = (
                self._source_cached_rollout is None
                or self._source_episode_step % self.source_action_hold == 0
            )
            if should_decide:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                self._source_cached_rollout = (actions, values, log_probs)
            else:
                assert self._source_cached_rollout is not None
                actions, values, log_probs = self._source_cached_rollout
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._source_episode_step += 1
            if bool(np.any(dones)):
                if env.num_envs != 1 and self.source_action_hold > 1:
                    raise NotImplementedError(
                        "Source CARLA action holding currently supports one environment"
                    )
                self._source_episode_step = 0
                self._source_cached_rollout = None

        # ``values`` is intentionally the value produced before the final
        # env.step(), matching finish_horizon(last_val=val) in the release.
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
