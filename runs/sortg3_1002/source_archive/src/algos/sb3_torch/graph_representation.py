"""Structured, action-conditioned one-step Graph-SLT objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .features import keras_initialize_linear
from .topo_temporal_features import StructuredLatent


@dataclass(frozen=True)
class GraphSLTLosses:
    """Total and per-slot losses retained for causal diagnosis."""

    total: Tensor
    ego: Tensor
    social: Tensor
    route: Tensor

    def detached(self) -> dict[str, Tensor]:
        return {
            "graph_slt_loss": self.total.detach(),
            "graph_slt_ego_loss": self.ego.detach(),
            "graph_slt_social_loss": self.social.detach(),
            "graph_slt_route_loss": self.route.detach(),
        }


class StructuredGraphRepresentationObjective(nn.Module):
    """Predict each semantic slot with residual action-conditioned dynamics."""

    SLOT_DIMS = {"ego": 32, "social": 64, "route": 32}

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if feature_dim != sum(self.SLOT_DIMS.values()):
            raise ValueError(
                f"Graph-SLT requires the 32/64/32 latent (128 total), got {feature_dim}"
            )
        self.feature_dim = int(feature_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embedding_dim),
            nn.ReLU(),
            nn.Linear(action_embedding_dim, action_embedding_dim),
            nn.ReLU(),
        )
        transition_input_dim = feature_dim + action_embedding_dim
        self.dynamics = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(transition_input_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, slot_dim),
                )
                for name, slot_dim in self.SLOT_DIMS.items()
            }
        )
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(slot_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, slot_dim),
                )
                for name, slot_dim in self.SLOT_DIMS.items()
            }
        )
        self.apply(keras_initialize_linear)

    @staticmethod
    def _coerce(value: StructuredLatent | Tensor) -> StructuredLatent:
        return value if isinstance(value, StructuredLatent) else StructuredLatent.from_tensor(value)

    def predict(
        self,
        online_features: StructuredLatent | Tensor,
        actions: Tensor,
    ) -> StructuredLatent:
        online = self._coerce(online_features)
        full_latent = online.tensor
        transition = torch.cat([full_latent, self.action_encoder(actions)], dim=-1)
        return StructuredLatent(
            z_ego=online.z_ego + self.dynamics["ego"](transition),
            z_social=online.z_social + self.dynamics["social"](transition),
            z_route=online.z_route + self.dynamics["route"](transition),
        )

    @staticmethod
    def _masked_mean(losses: Tensor, sample_mask: Tensor | None) -> Tensor:
        if sample_mask is not None:
            losses = losses * sample_mask.reshape(-1).to(losses.dtype)
        # Match the existing SLT/source convention: invalid samples contribute
        # zero but the denominator remains the full replay batch.
        return losses.mean()

    def _slot_loss(
        self,
        name: str,
        prediction: Tensor,
        target: Tensor,
        sample_mask: Tensor | None,
    ) -> Tensor:
        online_projection = self.projectors[name](prediction)
        with torch.no_grad():
            target_projection = self.projectors[name](target)
        losses = 1.0 - functional.cosine_similarity(
            online_projection, target_projection, dim=-1
        )
        return self._masked_mean(losses, sample_mask)

    def forward(
        self,
        online_features: StructuredLatent | Tensor,
        actions: Tensor,
        target_features: StructuredLatent | Tensor,
        sample_mask: Tensor | None = None,
    ) -> GraphSLTLosses:
        target = self._coerce(target_features)
        prediction = self.predict(online_features, actions)
        ego = self._slot_loss(
            "ego", prediction.z_ego, target.z_ego, sample_mask
        )
        social = self._slot_loss(
            "social", prediction.z_social, target.z_social, sample_mask
        )
        route = self._slot_loss(
            "route", prediction.z_route, target.z_route, sample_mask
        )
        return GraphSLTLosses(
            total=(ego + social + route) / 3.0,
            ego=ego,
            social=social,
            route=route,
        )

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        """Compatibility hook; target features are encoder-EMA stop-gradients."""

        del tau


__all__ = [
    "GraphSLTLosses",
    "StructuredGraphRepresentationObjective",
]
