"""SceneRepresentationSAC extension for batch-statistic SoftBalancedSlots."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th

from .features import _nonzero_mask
from .graph_representation import GraphSLTLosses
from .sac import SceneRepresentationSAC, _clip_gradients_like_keras
from .topo_temporal_features import StructuredLatent
from .topo_temporal_features_v2 import soft_slot_balance_loss


V2_DIAGNOSTIC_NAMES = {
    "topology_effective_lanes",
    "route_compatible_attention_mass",
    "merge_attention_mass",
    "merge_pair_score",
    "reverse_lane_mask_rate",
    "topology_fallback_rate",
    "topology_residual_scale",
    "goal_residual_scale",
    "route_bias_weight",
    "heading_bias_weight",
    "soft_slot_balance_loss",
    "ego_slot_rms",
    "social_slot_rms",
    "route_slot_rms",
    "slot_rms_ratio",
}


class SceneRepresentationSACV2(SceneRepresentationSAC):
    """Add SoftBalancedSlots to the structured representation update only.

    The SAC reward, actor, twin critics, entropy objective, replay contract,
    and Graph-SLT target remain inherited and unchanged.  A zero coefficient
    is exactly the V4 training objective; a positive coefficient is V5.
    """

    def __init__(
        self,
        *args: Any,
        slot_balance_coef: float = 0.0,
        slot_balance_epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.slot_balance_coef = float(slot_balance_coef)
        self.slot_balance_epsilon = float(slot_balance_epsilon)
        if self.slot_balance_coef < 0.0:
            raise ValueError("slot_balance_coef cannot be negative")
        if self.slot_balance_epsilon <= 0.0:
            raise ValueError("slot_balance_epsilon must be positive")
        self._last_soft_balance_loss = th.zeros(())
        super().__init__(*args, **kwargs)

    def _record_v2_value(self, name: str, value: th.Tensor | float) -> None:
        scalar = float(
            value.detach().mean().cpu()
            if isinstance(value, th.Tensor)
            else value
        )
        if not np.isfinite(scalar):
            return
        self.logger.record(name, scalar)
        self._accumulate_training_stat(name, scalar)

    def _representation_step(self, replay_data: Any) -> th.Tensor:
        if not self.structured_representation:
            return super()._representation_step(replay_data)
        if self.representation is None or self.representation_optimizer is None:
            raise RuntimeError("The representation objective is disabled")

        target_critic = (
            self.critic
            if self.representation_online_target_encoder
            else self.critic_target
        )
        representation_next_observations = getattr(
            replay_data,
            "one_step_next_observations",
            replay_data.next_observations,
        )
        sample_mask = _nonzero_mask(
            representation_next_observations["trajectory"][:, 0, 0]
        )
        online_extractor = self.critic.features_extractor
        target_extractor = target_critic.features_extractor
        online_features = online_extractor.forward_tokens(  # type: ignore[attr-defined]
            replay_data.observations
        )
        with th.no_grad():
            target_features = target_extractor.forward_tokens(  # type: ignore[attr-defined]
                representation_next_observations
            )
        if not isinstance(online_features, StructuredLatent) or not isinstance(
            target_features, StructuredLatent
        ):
            raise TypeError("forward_tokens() must return StructuredLatent")

        graph_losses = self.representation(
            online_features,
            replay_data.actions,
            target_features,
            sample_mask=sample_mask,
        )
        if not isinstance(graph_losses, GraphSLTLosses):
            raise TypeError("Structured Graph-SLT must return GraphSLTLosses")
        balance_loss, slot_rms = soft_slot_balance_loss(
            online_features,
            epsilon=self.slot_balance_epsilon,
        )
        weighted_balance = self.slot_balance_coef * balance_loss
        loss = graph_losses.total + weighted_balance
        if not bool(th.isfinite(loss).all()):
            raise FloatingPointError("Non-finite Graph-SLT + SoftBalancedSlots loss")

        self._last_soft_balance_loss = balance_loss.detach()
        self._last_graph_slt_losses = {
            **graph_losses.detached(),
            "soft_slot_balance_loss": balance_loss.detach(),
            "weighted_soft_slot_balance_loss": weighted_balance.detach(),
        }
        self._record_v2_value(
            "train/graph_slt_unregularized_loss", graph_losses.total
        )
        self._record_v2_value("train/soft_slot_balance_loss", balance_loss)
        self._record_v2_value(
            "train/weighted_soft_slot_balance_loss", weighted_balance
        )
        if hasattr(online_extractor, "diagnostic_values"):
            diagnostics = online_extractor.diagnostic_values()  # type: ignore[attr-defined]
            for name in sorted(V2_DIAGNOSTIC_NAMES):
                value = diagnostics.get(name)
                if value is not None and bool(th.isfinite(value).all()):
                    self._record_v2_value(f"diagnostic/{name}", value)

        self.representation_optimizer.zero_grad()
        (self.representation_coef * loss).backward()
        _clip_gradients_like_keras(
            self._representation_parameters, self.max_grad_norm
        )
        self.representation_optimizer.step()
        return loss.detach()


__all__ = ["SceneRepresentationSACV2", "V2_DIAGNOSTIC_NAMES"]
