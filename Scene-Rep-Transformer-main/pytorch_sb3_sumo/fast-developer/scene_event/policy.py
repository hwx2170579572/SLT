"""SAC with explicit online/target encoder ownership for new scene encoders."""
from copy import deepcopy
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import ExperimentConfig, soft_bellman_target


def mlp(input_dim: int, hidden_dim: int, output_dim: int, final_hidden_dim: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, final_hidden_dim), nn.ReLU(), nn.Linear(final_hidden_dim, output_dim))


class SquashedGaussianActor(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int, final_hidden_dim: int = 32):
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp(z_dim, hidden_dim, 2 * action_dim, final_hidden_dim)

    def distribution(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(z).chunk(2, dim=-1)
        return mean, log_std.clamp(-20.0, 2.0)

    def sample(self, z: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(z)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else normal.rsample()
        action = torch.tanh(pre_tanh)
        # Stable log derivative of tanh; does not underflow at saturated actions.
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int, final_hidden_dim: int = 32,
                 action_embedding_dim: int = 64):
        super().__init__()
        self.action_embedding1 = nn.Sequential(nn.Linear(action_dim, action_embedding_dim), nn.ReLU())
        self.action_embedding2 = nn.Sequential(nn.Linear(action_dim, action_embedding_dim), nn.ReLU())
        self.q1 = mlp(z_dim + action_embedding_dim, hidden_dim, 1, final_hidden_dim)
        self.q2 = mlp(z_dim + action_embedding_dim, hidden_dim, 1, final_hidden_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.q1(torch.cat((z, self.action_embedding1(action)), dim=-1)),
                self.q2(torch.cat((z, self.action_embedding2(action)), dim=-1)))


def to_tensors(observation: dict[str, np.ndarray | torch.Tensor], device: torch.device,
               add_batch: bool = False) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in observation.items():
        # replay snapshots are intentionally readonly; copy to avoid exposing
        # mutable tensor aliases to those arrays on CPU.
        tensor = value.to(device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
        result[key] = tensor.unsqueeze(0) if add_batch else tensor
    return result


class SceneSAC(nn.Module):
    """Actor heads see detached features; shared stem is trained by TD.

    The Stage-3/4 implementation has no auxiliary heads. Later auxiliary models
    must extend `auxiliary_loss` without introducing a second stem optimizer.
    """
    def __init__(self, encoder: nn.Module, config: ExperimentConfig, action_dim: int = 2):
        super().__init__()
        config.validate()
        self.config, self.action_dim = config, action_dim
        self.encoder = encoder
        # Component-specific CPU initialization keeps common heads identical
        # when an encoder variant adds parameters. Preserve training RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed + 101)
            self.actor = SquashedGaussianActor(config.z_dim, action_dim, config.hidden_dim, config.final_hidden_dim)
            torch.random.default_generator.manual_seed(config.seed + 202)
            self.critic = TwinQ(config.z_dim, action_dim, config.hidden_dim, config.final_hidden_dim, config.action_embedding_dim)
        self.target_encoder = deepcopy(encoder).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha)))
        self.target_entropy = -float(action_dim)
        self.device_obj = torch.device(config.device)
        self.to(self.device_obj)
        self.critic_optimizer = torch.optim.NAdam(
            list(self.encoder.parameters()) + list(self.critic.parameters()), lr=config.learning_rate, eps=1e-7)
        self.actor_optimizer = torch.optim.NAdam(self.actor.parameters(), lr=config.learning_rate, eps=1e-7)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate,
                                                betas=(0.5, 0.999), eps=1e-7)
        self.updates = 0
        self.assert_parameter_ownership()

    def assert_parameter_ownership(self) -> None:
        groups = [set(id(p) for group in opt.param_groups for p in group["params"])
                  for opt in (self.critic_optimizer, self.actor_optimizer, self.alpha_optimizer)]
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
            raise RuntimeError("A parameter is owned by multiple optimizers")
        if set.union(*groups) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("A trainable parameter has no optimizer owner")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        z = self.encoder(to_tensors(observation, self.device_obj, add_batch=True))
        action, _ = self.actor.sample(z, deterministic=deterministic)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Policy produced a non-finite physical action")
        return action[0].cpu().numpy()

    @torch.no_grad()
    def action_mean(self, observation: dict[str, np.ndarray], **encoder_options: Any) -> tuple[np.ndarray, np.ndarray]:
        z = self.encoder(to_tensors(observation, self.device_obj, add_batch=True), **encoder_options)
        mean, log_std = self.actor.distribution(z)
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise FloatingPointError("Policy distribution is non-finite")
        return torch.tanh(mean)[0].cpu().numpy(), log_std[0].cpu().numpy()

    def auxiliary_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        # Deliberately absent in M0/M1. Returning zero is not a claim that the
        # later learned-progress stage is implemented.
        return torch.zeros((), device=self.device_obj), {"aux_active": 0.0}

    def update(self, batch: dict[str, Any], audit: bool = False) -> dict[str, float]:
        obs = to_tensors(batch["observation"], self.device_obj)
        next_obs = to_tensors(batch["next_observation"], self.device_obj)
        def field(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(batch[name], dtype=dtype, device=self.device_obj)
        action, reward, discount = field("action"), field("reward"), field("discount")
        terminated = field("terminated", torch.bool)
        before = {name: p.detach().clone() for name, p in self.encoder.named_parameters()} if audit else {}
        with torch.no_grad():
            # Three distinct forwards: next online actor features, next target
            # Q features, and current online Q features below.
            next_online_z = self.encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_online_z)
            next_target_z = self.target_encoder(next_obs)
            target_q1, target_q2 = self.target_critic(next_target_z, next_action)
            target = soft_bellman_target(reward, discount, terminated,
                torch.minimum(target_q1, target_q2), next_log_prob, self.alpha.detach())

        self.critic_optimizer.zero_grad(set_to_none=True)
        z = self.encoder(obs)
        q1, q2 = self.critic(z, action)
        td_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        aux_loss, aux_metrics = self.auxiliary_loss(batch)
        total_loss = td_loss + aux_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite critic/auxiliary loss")
        total_loss.backward()
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            p.grad.detach().norm() for p in self.encoder.parameters() if p.grad is not None]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite encoder gradient")
        self.critic_optimizer.step()

        # Recompute current features after their update, then detach. Do not
        # perform an actor-loss update of shared representation parameters.
        with torch.no_grad():
            actor_z = self.encoder(obs)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        try:
            sampled_action, log_prob = self.actor.sample(actor_z.detach())
            actor_q1, actor_q2 = self.critic(actor_z.detach(), sampled_action)
            actor_loss = (self.alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite actor loss")
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.soft_update()
        self.updates += 1
        metrics = {"updates": float(self.updates), "td_loss": float(td_loss.detach()),
                   "actor_loss": float(actor_loss.detach()), "alpha_loss": float(alpha_loss.detach()),
                   "alpha": float(self.alpha.detach()), "encoder_gradient_norm": float(gradient_norm),
                   "q_mean": float(torch.minimum(q1, q2).detach().mean()),
                   "target_mean": float(target.mean()), **aux_metrics}
        if audit:
            delta = sum(float((p.detach() - before[name]).square().sum()) for name, p in self.encoder.named_parameters())
            metrics["encoder_parameter_delta_norm"] = math.sqrt(delta)
            metrics["target_has_gradient"] = float(any(p.grad is not None for p in self.target_encoder.parameters()))
            for module_name, module in self.encoder.named_children():
                module_delta = sum(float((p.detach() - before[f"{module_name}.{name}"]).square().sum())
                                   for name, p in module.named_parameters())
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                metrics[f"module/{module_name}/update_norm"] = math.sqrt(module_delta)
                metrics[f"module/{module_name}/gradient_norm"] = math.sqrt(sum(float(g.detach().square().sum()) for g in gradients))
        return metrics

    @torch.no_grad()
    def soft_update(self) -> None:
        for online, target in ((self.encoder, self.target_encoder), (self.critic, self.target_critic)):
            source = dict(online.named_parameters())
            targets = dict(target.named_parameters())
            if source.keys() != targets.keys():
                raise RuntimeError("Target parameter schema drift")
            for name, target_parameter in targets.items():
                target_parameter.lerp_(source[name], self.config.tau)
            buffers = dict(online.named_buffers())
            for name, target_buffer in target.named_buffers():
                target_buffer.copy_(buffers[name])

    def checkpoint(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "config_sha256": self.config.digest(),
                "model": self.state_dict(), "updates": self.updates,
                "optimizers": {"critic": self.critic_optimizer.state_dict(),
                               "actor": self.actor_optimizer.state_dict(),
                               "alpha": self.alpha_optimizer.state_dict()}}
