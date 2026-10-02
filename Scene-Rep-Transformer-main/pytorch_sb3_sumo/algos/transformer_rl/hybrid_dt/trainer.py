"""Offline training loop for the Hybrid-action Decision Transformer."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .hybrid_dt_model import HybridDecisionTransformer


def _episode_bounds(dones: np.ndarray) -> list[tuple[int, int]]:
    """Split a flat transition table into ``(start, end)`` episode ranges."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for index, done in enumerate(dones):
        if done:
            bounds.append((start, index + 1))
            start = index + 1
    if start < len(dones):
        bounds.append((start, len(dones)))
    return bounds


def _sample_batch(
    states: Tensor,
    actions: Tensor,
    returns_to_go: Tensor,
    timesteps: Tensor,
    episode_bounds: list[tuple[int, int]],
    context_length: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sample ``batch_size`` fixed-length windows without crossing episodes."""
    batch_states, batch_actions, batch_rtg, batch_timesteps, batch_masks = [], [], [], [], []
    device = states.device
    for _ in range(batch_size):
        start, end = rng.choice(episode_bounds)
        length = end - start
        if length >= context_length:
            begin = rng.randint(start, end - context_length)
            s = states[begin : begin + context_length]
            a = actions[begin : begin + context_length]
            r = returns_to_go[begin : begin + context_length]
            t = timesteps[begin : begin + context_length]
            mask = torch.ones(context_length, dtype=torch.long, device=device)
        else:
            s = states[start:end]
            a = actions[start:end]
            r = returns_to_go[start:end]
            t = timesteps[start:end]
            padding = context_length - length
            s = F.pad(s, (0, 0, padding, 0))
            a = F.pad(a, (0, 0, padding, 0))
            r = F.pad(r, (padding, 0))
            t = F.pad(t, (padding, 0))
            mask = torch.cat(
                [
                    torch.zeros(padding, dtype=torch.long, device=device),
                    torch.ones(length, dtype=torch.long, device=device),
                ]
            )
        batch_states.append(s)
        batch_actions.append(a)
        batch_rtg.append(r)
        batch_timesteps.append(t)
        batch_masks.append(mask)
    return (
        torch.stack(batch_states),
        torch.stack(batch_actions),
        torch.stack(batch_rtg)[..., None],
        torch.stack(batch_timesteps),
        torch.stack(batch_masks),
    )


def train_hybrid_dt(
    model: HybridDecisionTransformer,
    dataset: dict[str, np.ndarray],
    *,
    context_length: int = 20,
    batch_size: int = 64,
    num_steps: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, list[float]]:
    """Run the offline DT objective: speed MSE + lane cross-entropy."""

    rng = random.Random(seed)
    states = torch.as_tensor(dataset["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
    returns_to_go = torch.as_tensor(
        dataset["returns_to_go"], dtype=torch.float32, device=device
    )
    timesteps = torch.as_tensor(dataset["timesteps"], dtype=torch.long, device=device)
    dones = np.asarray(dataset["dones"], dtype=bool)
    episode_bounds = _episode_bounds(dones)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    metrics: dict[str, list[float]] = {"loss": [], "speed_loss": [], "lane_loss": []}

    model.train()
    for _step in range(num_steps):
        s, a, r, t, mask = _sample_batch(
            states,
            actions,
            returns_to_go,
            timesteps,
            episode_bounds,
            context_length,
            batch_size,
            rng,
        )
        speed, lane_logits = model(s, a, None, r, t, attention_mask=mask)
        valid = mask.float()

        speed_loss = F.mse_loss(speed, a[..., :1], reduction="none").squeeze(-1)
        speed_loss = (speed_loss * valid).sum() / valid.sum().clamp_min(1.0)

        lane_target = a[..., 1:].argmax(dim=-1)
        lane_loss = F.cross_entropy(
            lane_logits.reshape(-1, model.num_lane),
            lane_target.reshape(-1),
            reduction="none",
        ).reshape(batch_size, context_length)
        lane_loss = (lane_loss * valid).sum() / valid.sum().clamp_min(1.0)

        loss = speed_loss + lane_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        metrics["loss"].append(float(loss.detach().cpu()))
        metrics["speed_loss"].append(float(speed_loss.detach().cpu()))
        metrics["lane_loss"].append(float(lane_loss.detach().cpu()))

    return metrics
