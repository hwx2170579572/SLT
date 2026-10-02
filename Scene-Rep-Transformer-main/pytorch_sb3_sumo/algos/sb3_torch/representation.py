"""Future scene-representation auxiliary objective."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .features import KerasStyleMultiheadAttention, keras_initialize_linear


class FutureRepresentationObjective(nn.Module):
    """Action-conditioned one-step representation consistency objective.

    The released configuration uses ``future_steps=1``.  This module retains a
    causal temporal-attention transition so the same interface can be extended
    to longer future windows without changing the loss definition.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_embedding_dim: int = 64,
        num_heads: int = 2,
        separate_target_projector: bool = False,
    ) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embedding_dim), nn.ReLU()
        )
        transition_dim = feature_dim + action_embedding_dim
        self.temporal_attention = KerasStyleMultiheadAttention(
            transition_dim,
            transition_dim,
            transition_dim,
            num_heads=num_heads,
            head_dim=feature_dim // num_heads,
            output_dim=feature_dim,
        )
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )
        self.separate_target_projector = bool(separate_target_projector)
        self.target_projector: nn.Module | None
        if self.separate_target_projector:
            self.target_projector = copy.deepcopy(self.projector)
            self.target_projector.requires_grad_(False)
        else:
            self.target_projector = None
        self.predictor = nn.Linear(feature_dim, feature_dim)
        self.apply(keras_initialize_linear)
        if self.target_projector is not None:
            self.target_projector.load_state_dict(self.projector.state_dict())

    def forward(
        self,
        online_features: Tensor,
        actions: Tensor,
        target_features: Tensor,
        sample_mask: Tensor | None = None,
    ) -> Tensor:
        transition_input = torch.cat(
            [online_features, self.action_encoder(actions)], dim=-1
        ).unsqueeze(1)
        transition, _ = self.temporal_attention(
            transition_input,
            transition_input,
            transition_input,
            need_weights=False,
        )
        online_projection = self.predictor(self.projector(transition.squeeze(1)))
        target_projector = (
            self.target_projector
            if self.target_projector is not None
            else self.projector
        )
        with torch.no_grad():
            target_projection = target_projector(target_features)
        # Keras cosine_similarity is negative cosine similarity.
        losses = -functional.cosine_similarity(
            online_projection, target_projection, dim=-1
        )
        if sample_mask is not None:
            # The source multiplies by the one-step future ego-x mask and then
            # takes a batch mean (it does not renormalize by valid samples).
            losses = losses * sample_mask.reshape(-1).to(losses.dtype)
        return losses.mean()

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        """Polyak-update the cross-scenario target projector when present."""

        if self.target_projector is None:
            return
        for online, target in zip(
            self.projector.parameters(), self.target_projector.parameters()
        ):
            target.mul_(1.0 - tau).add_(online, alpha=tau)
