"""Lightweight GPT-2 backbone (pure PyTorch) for the Decision-Transformer baselines.

The upstream kzl Decision Transformer ships ``trajectory_gpt2.py``, a vendored
HuggingFace GPT-2 with positional embeddings removed.  That file depends on the
``transformers`` package, which is not part of the ``llm_pipeline`` environment.
This module provides the same causal-transformer interface with no external
dependencies:

* ``GPT2Config`` mirrors the ``n_embd``/``n_layer``/``n_head`` knobs used by the
  DT constructors (``vocab_size`` is accepted and ignored, as in the original).
* ``GPT2Model.forward(inputs_embeds, attention_mask)`` returns
  ``{"last_hidden_state": ...}`` exactly like the HuggingFace call, so the DT
  heads remain byte-for-byte unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class GPT2Config:
    """Minimal equivalent of ``transformers.GPT2Config`` for the DT constructors."""

    def __init__(
        self,
        vocab_size: int = 1,
        n_embd: int = 128,
        n_layer: int = 3,
        n_head: int = 1,
        **kwargs,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.n_embd = int(n_embd)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_inner = kwargs.get("n_inner", None)
        self.activation_function = kwargs.get("activation_function", "relu")
        self.resid_pdrop = float(kwargs.get("resid_pdrop", 0.1))
        self.embd_pdrop = float(kwargs.get("embd_pdrop", 0.1))
        self.attn_pdrop = float(kwargs.get("attn_pdrop", 0.1))
        self.layer_norm_epsilon = float(kwargs.get("layer_norm_epsilon", 1e-5))
        self.initializer_range = float(kwargs.get("initializer_range", 0.02))


def _activation(name: str) -> nn.Module:
    if name in ("relu", "gelu", "gelu_new", "gelu_python"):
        return nn.GELU() if name.startswith("gelu") else nn.ReLU()
    raise ValueError(f"Unsupported activation function {name!r}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention over precomputed input embeddings."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, dim = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        query = query.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        causal = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=x.device)
        ).view(1, 1, length, length)
        scores = scores.masked_fill(~causal, -1e9)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError(
                    f"attention_mask must have shape [{batch},{length}], got "
                    f"{tuple(attention_mask.shape)}"
                )
            scores = scores.masked_fill(
                ~attention_mask.bool()[:, None, None, :], -1e9
            )
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        inner = config.n_inner or 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner)
        self.c_proj = nn.Linear(inner, config.n_embd)
        self.act = _activation(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """Causal transformer over precomputed embeddings (no positional encoding)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        hidden = self.drop(inputs_embeds)
        for block in self.h:
            hidden = block(hidden, attention_mask)
        hidden = self.ln_f(hidden)
        return {"last_hidden_state": hidden}


__all__ = ["GPT2Config", "GPT2Model"]
