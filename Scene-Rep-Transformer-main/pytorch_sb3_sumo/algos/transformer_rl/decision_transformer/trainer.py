"""Offline training loop for the continuous-action Decision Transformer.

Mirrors ``hybrid_dt.trainer`` but for a ``Box(2)`` action: the objective is the
MSE between the predicted (tanh-bounded) action and the stored continuous
action, masked to valid sequence positions (left padding is ignored).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..hybrid_dt.trainer import _episode_bounds, _sample_batch
from .models import DecisionTransformer


def train_decision_transformer(
    model: DecisionTransformer,
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
    """Run the offline DT objective: continuous-action MSE."""

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
    metrics: dict[str, list[float]] = {"loss": [], "action_loss": []}

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
        _state_preds, action_preds, _return_preds = model(
            s, a, None, r, t, attention_mask=mask
        )
        valid = mask.float()
        action_loss = F.mse_loss(action_preds, a, reduction="none").mean(dim=-1)
        action_loss = (action_loss * valid).sum() / valid.sum().clamp_min(1.0)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        metrics["loss"].append(float(action_loss.detach().cpu()))
        metrics["action_loss"].append(float(action_loss.detach().cpu()))

    return metrics
