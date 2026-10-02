"""Shared trajectory-modelling interface for the DT baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_length: int | None = None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs may be passed in as None.
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
