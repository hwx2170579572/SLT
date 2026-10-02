"""Hybrid-action Decision Transformer (lane softmax + conditional speed).

A GPT-2 sequence model over ``(return, state, action)`` triplets whose action
head emits a hybrid action: a categorical lane logit over ``{-1, 0, 1}`` plus a
tanh-bounded speed.  This mirrors the hybrid action space used by TASAC/Hold35k
so the sequence-modelling baseline can be scored on the same ``Box(2)``
environment contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..decision_transformer.models.gpt2 import GPT2Config, GPT2Model
from ..decision_transformer.models.model import TrajectoryModel

LANE_COMMANDS = (-1, 0, 1)


class HybridDecisionTransformer(TrajectoryModel):
    """Sequence model with a hybrid lane/speed action head."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 128,
        max_length: int | None = 20,
        max_ep_len: int = 4096,
        num_lane: int = 3,
        n_layer: int = 3,
        n_head: int = 1,
        **kwargs,
    ):
        act_dim = 1 + num_lane  # speed + lane one-hot
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = int(hidden_size)
        self.num_lane = int(num_lane)

        config = GPT2Config(
            vocab_size=1, n_embd=self.hidden_size, n_layer=n_layer, n_head=n_head, **kwargs
        )
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(state_dim, self.hidden_size)
        self.embed_action = nn.Linear(act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_speed = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Tanh())
        self.predict_lane = nn.Linear(self.hidden_size, num_lane)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor | None,
        returns_to_go: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, seq_length), dtype=torch.long, device=states.device
            )

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch, 3 * seq_length)

        hidden = self.transformer(
            inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask
        )["last_hidden_state"]
        hidden = hidden.reshape(batch, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        speed = self.predict_speed(hidden[:, 1])       # [B, seq, 1]
        lane_logits = self.predict_lane(hidden[:, 1])  # [B, seq, num_lane]
        return speed, lane_logits

    def get_action(
        self, states, actions, rewards, returns_to_go, timesteps, **kwargs
    ) -> Tensor:
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]
            padding = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(padding), torch.ones(states.shape[1])]
            ).to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((1, padding, self.state_dim), device=states.device), states],
                dim=1,
            ).float()
            actions = torch.cat(
                [torch.zeros((1, padding, self.act_dim), device=actions.device), actions],
                dim=1,
            ).float()
            returns_to_go = torch.cat(
                [torch.zeros((1, padding, 1), device=returns_to_go.device), returns_to_go],
                dim=1,
            ).float()
            timesteps = torch.cat(
                [torch.zeros((1, padding), device=timesteps.device), timesteps], dim=1
            ).long()
        else:
            attention_mask = None

        speed, lane_logits = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )
        speed_value = speed[0, -1, 0]
        lane_index = lane_logits[0, -1].argmax()
        lane_codes = torch.tensor(LANE_COMMANDS, dtype=speed.dtype, device=speed.device)
        return torch.stack([speed_value, lane_codes[lane_index]])
