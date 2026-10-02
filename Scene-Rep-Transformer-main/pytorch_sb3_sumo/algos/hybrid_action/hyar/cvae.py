"""Conditional VAE for hybrid-action representation (HyAR).

Encodes a mixed action ``(discrete lane, continuous speed)`` conditioned on the
state into a low-dimensional latent ``z``, and decodes ``z`` back into a lane
categorical and a speed mean.  This is the representation backbone of HyAR
(Li et al., ICLR 2022): the policy and critic operate on ``z`` while the decoder
maps back to the original mixed action space.

Reference: https://github.com/GQYXYH/discrete-continuous (HyAR-TD3/DDPG).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HybridActionVAE(nn.Module):
    """Gaussian VAE over ``[lane one-hot | speed]`` conditioned on the state."""

    def __init__(
        self,
        state_dim: int,
        *,
        num_discrete: int = 3,
        cont_dim: int = 1,
        action_embed_dim: int = 8,
        latent_dim: int = 6,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_discrete = int(num_discrete)
        self.cont_dim = int(cont_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        # Encoder: [state, discrete one-hot, continuous] -> (mu, logvar).
        encoder_input = state_dim + num_discrete + cont_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: [state, z] -> (discrete logits, continuous mean).
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.discrete_head = nn.Linear(hidden_dim, num_discrete)
        self.continuous_head = nn.Linear(hidden_dim, cont_dim)

    def encode(self, state: Tensor, discrete: Tensor, continuous: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat([state, discrete, continuous], dim=-1)
        hidden = self.encoder(inputs)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, state: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat([state, z], dim=-1)
        hidden = self.decoder(inputs)
        return self.discrete_head(hidden), self.continuous_head(hidden)

    def forward(
        self, state: Tensor, discrete: Tensor, continuous: Tensor
    ) -> dict[str, Tensor]:
        mu, logvar = self.encode(state, discrete, continuous)
        z = self.reparameterize(mu, logvar)
        discrete_logits, continuous_mean = self.decode(state, z)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "discrete_logits": discrete_logits,
            "continuous_mean": continuous_mean,
        }

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)

    def reconstruction_loss(
        self,
        discrete: Tensor,
        continuous: Tensor,
        discrete_logits: Tensor,
        continuous_mean: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Cross-entropy for the lane categorical and MSE for the speed."""
        discrete_loss = F.cross_entropy(
            discrete_logits, discrete.argmax(dim=-1), reduction="none"
        )
        continuous_loss = F.mse_loss(
            continuous_mean, continuous, reduction="none"
        ).sum(dim=-1)
        return discrete_loss, continuous_loss
