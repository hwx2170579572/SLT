"""HyAR-TD3/DDPG skeleton: latent-space policy + VAE decoder + LSC/RSC.

HyAR (Li et al., ICLR 2022) learns a hybrid action representation through a
conditional VAE, then runs the policy and critic in the resulting latent space.
The decoder maps the latent action back to the mixed ``(discrete lane,
continuous speed)`` action.  Two auxiliary constraints stabilise the latent
space: LSC (Lipschitz smoothness of the decoder) and RSC (representation
consistency).  This module provides the trainable skeleton and losses; the
online/offline data flow into the SUMO environment is left to a follow-up step
(see ``sumo_adapter.py`` and the integration notes).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .cvae import HybridActionVAE


class LatentActor(nn.Module):
    """Deterministic latent policy: state -> latent action ``z``."""

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return torch.tanh(self.net(state))


class LatentCritic(nn.Module):
    """Latent Q-function: ``Q(state, z)``."""

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat([state, z], dim=-1))


class HyAR(nn.Module):
    """Latent-space actor/critic over a conditional VAE hybrid-action code."""

    def __init__(
        self,
        state_dim: int,
        *,
        num_discrete: int = 3,
        cont_dim: int = 1,
        latent_dim: int = 6,
        hidden_dim: int = 64,
        kl_beta: float = 1.0,
        lsc_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_discrete = int(num_discrete)
        self.cont_dim = int(cont_dim)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.lsc_coef = float(lsc_coef)

        self.vae = HybridActionVAE(
            state_dim,
            num_discrete=num_discrete,
            cont_dim=cont_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.actor = LatentActor(state_dim, latent_dim, hidden_dim)
        self.critic = LatentCritic(state_dim, latent_dim, hidden_dim)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def act(
        self, state: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(z, discrete_index, continuous_value)`` for a state."""
        z = self.actor(state)
        discrete_logits, continuous_mean = self.vae.decode(state, z)
        if deterministic:
            discrete_index = discrete_logits.argmax(dim=-1)
        else:
            discrete_index = Categorical(logits=discrete_logits).sample()
        continuous_value = torch.tanh(continuous_mean)
        return z, discrete_index, continuous_value

    def q_value(self, state: Tensor, z: Tensor) -> Tensor:
        return self.critic(state, z)

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #
    def vae_loss(
        self,
        state: Tensor,
        discrete: Tensor,
        continuous: Tensor,
    ) -> Tensor:
        out = self.vae(state, discrete, continuous)
        discrete_loss, continuous_loss = self.vae.reconstruction_loss(
            discrete,
            continuous,
            out["discrete_logits"],
            out["continuous_mean"],
        )
        kl = self.vae.kl_divergence(out["mu"], out["logvar"])
        return (discrete_loss + continuous_loss + self.kl_beta * kl).mean()

    def lsc_loss(self, state: Tensor, z: Tensor) -> Tensor:
        """Lipschitz smoothness: penalise the decoder Jacobian w.r.t. ``z``."""
        z = z.detach().requires_grad_(True)
        _discrete_logits, continuous_mean = self.vae.decode(state, z)
        gradient = torch.autograd.grad(
            continuous_mean.sum(), z, create_graph=True
        )[0]
        return gradient.square().sum(dim=-1).mean()

    def rsc_loss(
        self,
        state: Tensor,
        discrete: Tensor,
        continuous: Tensor,
    ) -> Tensor:
        """Representation consistency: encoded latent vs. re-encoded decode."""
        with torch.no_grad():
            mu, _logvar = self.vae.encode(state, discrete, continuous)
            z = self.vae.reparameterize(mu, _logvar)
            _disc_logits, cont_mean = self.vae.decode(state, z)
            reference = self.vae.encode(
                state, discrete, cont_mean.detach()
            )[0].detach()
        mu2, _ = self.vae.encode(state, discrete, cont_mean.detach())
        return F.mse_loss(mu2, reference)

    def compute_losses(
        self,
        state: Tensor,
        discrete: Tensor,
        continuous: Tensor,
        *,
        reward: Tensor | None = None,
        next_state: Tensor | None = None,
        next_discrete: Tensor | None = None,
        next_continuous: Tensor | None = None,
        done: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Assemble the full HyAR objective for one batch.

        The RL actor/critic terms are only meaningful when transition data
        (``reward``/``next_state``/``next_discrete``/``next_continuous``/
        ``done``) is supplied; the representation terms are always computable.
        """
        z = self.actor(state)
        losses: dict[str, Tensor] = {
            "vae_loss": self.vae_loss(state, discrete, continuous),
            "lsc_loss": self.lsc_loss(state, z),
            "rsc_loss": self.rsc_loss(state, discrete, continuous),
        }
        if reward is not None:
            q = self.critic(state, z)
            losses["critic_loss"] = F.mse_loss(
                q, reward.reshape_as(q)
            )
            losses["actor_loss"] = -q.mean()
        return losses
