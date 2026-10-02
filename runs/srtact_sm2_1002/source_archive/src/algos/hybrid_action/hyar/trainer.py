"""Offline training loop for HyAR: VAE representation + latent DDPG.

Trains the conditional-VAE hybrid-action representation (reconstruction + KL +
LSC + RSC) jointly with a DDPG actor/critic operating in the learned latent
space, from an offline dataset of ``(state, hybrid action, reward, next state,
done)`` transitions.  Soft target networks stabilise the latent Q-learning.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .hyar_algorithm import HyAR


def _soft_update(target, source, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _next_states(states: torch.Tensor) -> torch.Tensor:
    """Shift the flat transition table by one; terminal rows are unused (1-done=0)."""
    nxt = states.clone()
    nxt[:-1] = states[1:]
    nxt[-1] = states[-1]
    return nxt


def train_hyar(
    model: HyAR,
    dataset: dict[str, np.ndarray],
    *,
    num_steps: int = 50_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 5e-3,
    lsc_coef: float = 0.1,
    rsc_coef: float = 0.1,
    actor_delay: int = 2,
    learning_rate: float = 3e-4,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, list[float]]:
    """Run the offline HyAR objective (VAE + DDPG)."""

    rng = random.Random(seed)
    states = torch.as_tensor(dataset["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
    rewards = torch.as_tensor(dataset["rewards"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(dataset["dones"], dtype=torch.float32, device=device)

    discrete = actions[:, 1:]            # [N, num_discrete] one-hot
    continuous = actions[:, 0:1]         # [N, 1] speed
    nxt_states = _next_states(states)
    num_transitions = states.shape[0]

    model = model.to(device)
    actor_target = copy.deepcopy(model.actor)
    critic_target = copy.deepcopy(model.critic)
    for target in (actor_target, critic_target):
        target.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics: dict[str, list[float]] = {
        "loss": [],
        "vae_loss": [],
        "critic_loss": [],
        "actor_loss": [],
        "lsc_loss": [],
        "rsc_loss": [],
    }

    model.train()
    for step in range(num_steps):
        idx = torch.as_tensor(
            [rng.randrange(num_transitions) for _ in range(batch_size)],
            dtype=torch.long,
            device=device,
        )
        s = states[idx]
        disc = discrete[idx]
        cont = continuous[idx]
        r = rewards[idx][:, None]
        s_next = nxt_states[idx]
        d = dones[idx][:, None]

        # Representation losses (state-conditioned VAE + smoothness/consistency).
        vae_loss = model.vae_loss(s, disc, cont)
        z_actor = model.actor(s)
        lsc_loss = model.lsc_loss(s, z_actor)
        rsc_loss = model.rsc_loss(s, disc, cont)

        # DDPG critic: Q(s, z) -> r + gamma * (1 - done) * Q'(s', actor'(s')).
        z = model.actor(s).detach()
        q = model.critic(s, z)
        with torch.no_grad():
            z_next = actor_target(s_next)
            q_next = critic_target(s_next, z_next)
            target = r + gamma * (1.0 - d) * q_next
        critic_loss = F.mse_loss(q, target)

        # Delayed actor update maximises Q(s, actor(s)).
        if step % actor_delay == 0:
            actor_loss = -model.critic(s, model.actor(s)).mean()
        else:
            actor_loss = torch.zeros((), device=device)

        loss = (
            vae_loss
            + lsc_coef * lsc_loss
            + rsc_coef * rsc_loss
            + critic_loss
            + actor_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        _soft_update(actor_target, model.actor, tau)
        _soft_update(critic_target, model.critic, tau)

        metrics["loss"].append(float(loss.detach().cpu()))
        metrics["vae_loss"].append(float(vae_loss.detach().cpu()))
        metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
        metrics["actor_loss"].append(float(actor_loss.detach().cpu()))
        metrics["lsc_loss"].append(float(lsc_loss.detach().cpu()))
        metrics["rsc_loss"].append(float(rsc_loss.detach().cpu()))

    return metrics
