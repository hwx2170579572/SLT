"""Private bounded longitudinal residual for the D1 ST-RT SAC policy.

The conflict-timing observation is consumed only by private actor/critic
branches.  It never enters the shared scene encoder or the lateral action
head.  The actor residual is applied to the pre-tanh speed mean so the native
SquashedDiagGaussian distribution and its log-probability remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Mapping

import torch as th
from torch import nn

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN

from .module_diagnostics import (
    collect_parameter_gradient_diagnostics,
    collect_parameter_update_diagnostics,
    snapshot_parameter_groups,
)
from .policies import ActionEmbeddedContinuousCritic, DetachedSceneActor, SceneRepSACPolicy


CONFLICT_TIMING_FEATURE_DIM = 20
CONFLICT_RELATION_VALID_INDEX = 0
CONFLICT_SUMMARY_DIM = 2 * CONFLICT_TIMING_FEATURE_DIM + 1
LONGRES_ACTOR_GROUPS = {
    "longres_actor": ("longitudinal_residual_encoder", "longitudinal_residual_head"),
}
LONGRES_CRITIC_GROUPS = {
    "longres_critic": ("conflict_context_encoders", "conflict_q_residuals"),
}


def summarize_conflict_timing(observation: Mapping[str, Any] | None):
    """Return (fixed-size summary, per-batch support mask, valid-row count).

    The ego row is excluded: only currently observed, relation-valid social
    rows condition the private branch.  Any nonfinite feature makes its row
    unsupported instead of being silently interpreted as a zero-risk row.
    """
    if not isinstance(observation, Mapping):
        return None, None, None
    value = observation.get("conflict_timing")
    if not isinstance(value, th.Tensor) or value.ndim not in (2, 3):
        return None, None, None
    if value.shape[-1] != CONFLICT_TIMING_FEATURE_DIM:
        return None, None, None
    unbatched = value.ndim == 2
    rows = value.unsqueeze(0) if unbatched else value
    if rows.shape[1] <= 1:
        batch = rows.shape[0]
        return (
            rows.new_zeros((batch, CONFLICT_SUMMARY_DIM)),
            th.zeros(batch, dtype=th.bool, device=rows.device),
            th.zeros(batch, dtype=th.long, device=rows.device),
        )

    social_rows = rows[:, 1:, :].float()
    finite = th.isfinite(social_rows).all(dim=-1)
    relation_valid = social_rows[..., CONFLICT_RELATION_VALID_INDEX] > 0.5
    valid_rows = finite & relation_valid
    valid_count = valid_rows.sum(dim=1)
    supported = valid_count > 0
    safe_rows = th.where(th.isfinite(social_rows), social_rows, th.zeros_like(social_rows))
    weight = valid_rows.unsqueeze(-1).to(safe_rows.dtype)
    denominator = valid_count.clamp_min(1).unsqueeze(-1).to(safe_rows.dtype)
    mean = (safe_rows * weight).sum(dim=1) / denominator
    maximum = safe_rows.masked_fill(~valid_rows.unsqueeze(-1), -th.inf).amax(dim=1)
    maximum = th.where(supported.unsqueeze(-1), maximum, th.zeros_like(maximum))
    fraction = valid_count.to(safe_rows.dtype) / float(max(1, rows.shape[1] - 1))
    summary = th.cat([mean, maximum, fraction.unsqueeze(-1)], dim=-1)
    if unbatched:
        return summary, supported, valid_count
    return summary, supported, valid_count


def _scalar_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            result[str(key)] = None
            continue
        if isinstance(value, (str, bool)):
            result[str(key)] = value
            continue
        tensor = th.as_tensor(value).detach()
        if tensor.numel() != 1:
            continue
        number = float(tensor.cpu().item())
        result[str(key)] = number if math.isfinite(number) else None
    return result


class BoundedLongitudinalResidualActor(DetachedSceneActor):
    """Detached scene actor with a private CT residual on speed mean only."""

    def __init__(self, *args: Any, longitudinal_residual_bound: float = 0.2, **kwargs: Any):
        self.longitudinal_residual_bound = float(longitudinal_residual_bound)
        if not 0 < self.longitudinal_residual_bound <= 0.2:
            raise ValueError("longitudinal_residual_bound must be in (0, 0.2]")
        super().__init__(*args, **kwargs)
        self.longitudinal_residual_encoder = nn.Sequential(
            nn.Linear(CONFLICT_SUMMARY_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.longitudinal_residual_head = nn.Linear(16, 1)
        # The residual starts exactly at zero while all added parameters remain
        # trainable and optimizer-owned.
        nn.init.zeros_(self.longitudinal_residual_head.weight)
        nn.init.zeros_(self.longitudinal_residual_head.bias)
        self._longitudinal_residual_force_off = False
        self._last_longitudinal_diagnostics: dict[str, th.Tensor | float | int | None] = {}
        self._last_policy_shadow_mean_pre_tanh: th.Tensor | None = None

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor).detach()
        latent_pi = self.latent_pi(features)
        base_mean = self.mu(latent_pi)
        if self.use_sde:
            # D1 SAC uses the standard diagonal-Gaussian actor; fail explicitly
            # rather than silently bypassing the residual for gSDE.
            raise NotImplementedError("longitudinal residual policy does not support gSDE")

        summary, supported, valid_count = summarize_conflict_timing(obs)
        if summary is None or supported is None or valid_count is None:
            residual_raw = base_mean.new_zeros((base_mean.shape[0], 1))
            residual = residual_raw
            supported_batch = th.zeros(
                base_mean.shape[0], dtype=th.bool, device=base_mean.device
            )
            valid_count_batch = th.zeros(
                base_mean.shape[0], dtype=th.long, device=base_mean.device
            )
            invalid_reason = "conflict_timing_missing_or_invalid_shape"
        else:
            if summary.ndim == 1:
                summary = summary.unsqueeze(0)
            if supported.ndim == 0:
                supported = supported.unsqueeze(0)
            if valid_count.ndim == 0:
                valid_count = valid_count.unsqueeze(0)
            summary = summary.to(device=base_mean.device, dtype=base_mean.dtype)
            supported_batch = supported.to(device=base_mean.device)
            valid_count_batch = valid_count.to(device=base_mean.device)
            if self._longitudinal_residual_force_off:
                residual_raw = base_mean.new_zeros((base_mean.shape[0], 1))
                residual = residual_raw
            else:
                residual_raw = self.longitudinal_residual_head(
                    self.longitudinal_residual_encoder(summary)
                )
                bounded = self.longitudinal_residual_bound * th.tanh(residual_raw)
                residual = th.where(
                    supported_batch.unsqueeze(-1), bounded, th.zeros_like(bounded)
                )
            invalid_reason = None

        mean_actions = base_mean.clone()
        mean_actions[:, 0:1] = mean_actions[:, 0:1] + residual
        # The generic shadow helper hooks ``mu`` which is before this residual.
        # Keep the actual distribution mean for diagnostic-only snapshot reads;
        # this detached cache never participates in loss or action generation.
        self._last_policy_shadow_mean_pre_tanh = mean_actions.detach()
        self._last_longitudinal_diagnostics = {
            "longres_actor_active": float(not self._longitudinal_residual_force_off),
            "longres_actor_context_valid_fraction": supported_batch.float().mean(),
            "longres_actor_valid_relation_count_mean": valid_count_batch.float().mean(),
            "longres_actor_pre_tanh_base_speed_mean": base_mean[:, 0].detach().mean(),
            "longres_actor_pre_tanh_residual_mean": residual.detach().mean(),
            "longres_actor_pre_tanh_residual_abs_mean": residual.detach().abs().mean(),
            "longres_actor_residual_bound": self.longitudinal_residual_bound,
            "longres_actor_saturation_fraction": th.where(
                supported_batch.any(),
                (
                    (
                        (
                            residual.detach().abs()
                            / self.longitudinal_residual_bound
                        ) >= 0.95
                    )
                    & supported_batch.unsqueeze(-1)
                ).float().sum()
                / supported_batch.float().sum().clamp_min(1.0),
                residual.new_tensor(float("nan")),
            ),
            "longres_actor_invalid_reason": invalid_reason,
        }
        log_std = th.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def longitudinal_diagnostic_values(self) -> dict[str, float | int | None]:
        values = _scalar_metrics(self._last_longitudinal_diagnostics)
        if values.get("longres_actor_context_valid_fraction") == 0.0:
            values["longres_actor_invalid_reason"] = (
                values.get("longres_actor_invalid_reason")
                or "no_valid_conflict_relations"
            )
        return values

    def policy_shadow_mean_snapshot(self) -> th.Tensor | None:
        """Return the actual final pre-tanh mean from the latest actor call."""
        return self._last_policy_shadow_mean_pre_tanh

    @contextmanager
    def longitudinal_residual_probe(self, name: str | None):
        """Temporarily disable only the speed residual for same-state probes."""
        if name not in (None, "longitudinal_residual_off"):
            raise ValueError(f"unsupported longitudinal policy probe: {name}")
        previous = self._longitudinal_residual_force_off
        previous_diagnostics = dict(self._last_longitudinal_diagnostics)
        previous_mean = self._last_policy_shadow_mean_pre_tanh
        self._longitudinal_residual_force_off = name == "longitudinal_residual_off"
        try:
            yield
        finally:
            self._longitudinal_residual_force_off = previous
            self._last_longitudinal_diagnostics.clear()
            self._last_longitudinal_diagnostics.update(previous_diagnostics)
            self._last_policy_shadow_mean_pre_tanh = previous_mean


class LongitudinalResidualContinuousCritic(ActionEmbeddedContinuousCritic):
    """Action-conditioned private CT Q residual for each twin critic."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        action_embedding_dim = int(self.action_encoders[0][0].out_features)
        self.conflict_context_encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(CONFLICT_SUMMARY_DIM, 32), nn.ReLU())
                for _ in range(self.n_critics)
            ]
        )
        self.conflict_q_residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32 + action_embedding_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(self.n_critics)
            ]
        )
        for branch in self.conflict_q_residuals:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self._last_longitudinal_critic_diagnostics: dict[str, th.Tensor | float | int | None] = {}

    def _q_values(self, obs: PyTorchObs, actions: th.Tensor, features: th.Tensor):
        summary, supported, valid_count = summarize_conflict_timing(obs)
        if summary is None or supported is None or valid_count is None:
            summary = features.new_zeros((features.shape[0], CONFLICT_SUMMARY_DIM))
            supported = th.zeros(features.shape[0], dtype=th.bool, device=features.device)
            valid_count = th.zeros(features.shape[0], dtype=th.long, device=features.device)
            invalid_reason = "conflict_timing_missing_or_invalid_shape"
        else:
            if summary.ndim == 1:
                summary = summary.unsqueeze(0)
            if supported.ndim == 0:
                supported = supported.unsqueeze(0)
            if valid_count.ndim == 0:
                valid_count = valid_count.unsqueeze(0)
            summary = summary.to(device=features.device, dtype=features.dtype)
            supported = supported.to(device=features.device)
            valid_count = valid_count.to(device=features.device)
            invalid_reason = None

        values = []
        residuals = []
        for index, (q_network, action_encoder, context_encoder, q_residual) in enumerate(
            zip(
                self.q_networks,
                self.action_encoders,
                self.conflict_context_encoders,
                self.conflict_q_residuals,
            )
        ):
            action_features = action_encoder(actions)
            base_value = q_network(th.cat([features, action_features], dim=1))
            context_features = context_encoder(summary)
            residual = q_residual(th.cat([context_features, action_features], dim=1))
            residual = th.where(supported.unsqueeze(-1), residual, th.zeros_like(residual))
            values.append(base_value + residual)
            residuals.append(residual)
        self._last_longitudinal_critic_diagnostics = {
            "longres_critic_active": 1.0,
            "longres_critic_context_valid_fraction": supported.float().mean(),
            "longres_critic_valid_relation_count_mean": valid_count.float().mean(),
            "longres_critic_q_residual_abs_mean": th.cat(residuals, dim=1).detach().abs().mean(),
            "longres_critic_invalid_reason": invalid_reason,
        }
        return tuple(values)

    def forward(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        features = self.extract_features(obs, self.features_extractor)
        return self._q_values(obs, actions, features)

    def q1_forward(self, obs: PyTorchObs, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self._q_values(obs, actions, features)[0]

    def longitudinal_diagnostic_values(self) -> dict[str, float | int | None]:
        values = _scalar_metrics(self._last_longitudinal_critic_diagnostics)
        if values.get("longres_critic_context_valid_fraction") == 0.0:
            values["longres_critic_invalid_reason"] = (
                values.get("longres_critic_invalid_reason")
                or "no_valid_conflict_relations"
            )
        return values


class LongitudinalResidualSceneRepSACPolicy(SceneRepSACPolicy):
    """ST-RT SAC policy with private, optimizer-owned CT longitudinal branches."""

    def __init__(
        self,
        *args: Any,
        longitudinal_residual_bound: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.longitudinal_residual_bound = float(longitudinal_residual_bound)
        if not 0 < self.longitudinal_residual_bound <= 0.2:
            raise ValueError("longitudinal_residual_bound must be in (0, 0.2]")
        self._longitudinal_diagnostic_events: list[dict[str, Any]] = []
        self._longitudinal_optimizer_steps = {"actor": 0, "critic": 0}
        self._longitudinal_pending_updates: dict[str, dict[str, Any]] = {}
        self._longitudinal_diagnostics_enabled = False
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        original_actor = self.actor
        original_critic = self.critic
        original_target = self.critic_target

        # Construct added modules on CPU while restoring the global RNG stream;
        # the base ST-RT initialization and subsequent SAC initialization stay
        # identical to the existing method for the same seed.
        with th.random.fork_rng(devices=[]):
            actor_kwargs = self._update_features_extractor(
                self.actor_kwargs, original_actor.features_extractor
            )
            actor = BoundedLongitudinalResidualActor(
                **actor_kwargs,
                longitudinal_residual_bound=self.longitudinal_residual_bound,
            ).to(self.device)
            actor.load_state_dict(original_actor.state_dict(), strict=False)

            critic_kwargs = self._update_features_extractor(
                self.critic_kwargs, original_critic.features_extractor
            )
            critic_kwargs["share_features_extractor"] = False
            critic = LongitudinalResidualContinuousCritic(
                **critic_kwargs,
                action_embedding_dim=self.action_embedding_dim,
            ).to(self.device)
            critic.load_state_dict(original_critic.state_dict(), strict=False)

            target_kwargs = self._update_features_extractor(self.critic_kwargs, None)
            target_kwargs["share_features_extractor"] = False
            critic_target = LongitudinalResidualContinuousCritic(
                **target_kwargs,
                action_embedding_dim=self.action_embedding_dim,
            ).to(self.device)

        # As in the base policy, the target owns a separate extractor but is
        # exactly initialized from the online critic, including its new branch.
        critic_target.load_state_dict(critic.state_dict())
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        extractor_parameter_ids = {
            id(parameter) for parameter in self.actor.features_extractor.parameters()
        }
        self.actor.optimizer = self.optimizer_class(
            [
                parameter
                for parameter in self.actor.parameters()
                if id(parameter) not in extractor_parameter_ids
            ],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic_target.set_training_mode(False)
        self.share_features_extractor = True
        self._install_longitudinal_optimizer_hooks()

    def _install_longitudinal_optimizer_hooks(self) -> None:
        for name, module, optimizer, groups in (
            ("actor", self.actor, self.actor.optimizer, LONGRES_ACTOR_GROUPS),
            ("critic", self.critic, self.critic.optimizer, LONGRES_CRITIC_GROUPS),
        ):
            state_key = f"_longres_{name}"

            def before_step(optimizer, args, kwargs, *, _name=name, _module=module, _groups=groups, _key=state_key):
                next_index = self._longitudinal_optimizer_steps[_name] + 1
                if not self._longitudinal_diagnostics_enabled or not (
                    next_index == 1 or next_index % 1000 == 0
                ):
                    self._longitudinal_pending_updates.pop(_key, None)
                    return
                gradients = collect_parameter_gradient_diagnostics(_module, _groups)
                snapshots = snapshot_parameter_groups(_module, _groups, optimizer)
                branch_values = getattr(_module, "longitudinal_diagnostic_values", lambda: {})()
                self._longitudinal_pending_updates[_key] = {
                    "step_index": next_index,
                    "gradient_metrics": gradients,
                    "snapshots": snapshots,
                    "branch_values": branch_values,
                }

            def after_step(optimizer, args, kwargs, *, _name=name, _key=state_key):
                self._longitudinal_optimizer_steps[_name] += 1
                pending = self._longitudinal_pending_updates.pop(_key, None)
                if pending is None:
                    return
                updates = pending["step_index"]
                update_metrics = collect_parameter_update_diagnostics(pending["snapshots"])
                metrics = {
                    **pending["branch_values"],
                    **pending["gradient_metrics"],
                    **update_metrics,
                    "longres_optimizer_step_index": int(updates),
                    "longres_optimizer_group": _name,
                }
                self._longitudinal_diagnostic_events.append(
                    {
                        "source": f"longitudinal_residual_{_name}_optimizer_update",
                        "updates": int(updates),
                        "raw_steps": None,
                        "decision_steps": None,
                        "metrics": metrics,
                        "timing": "optimizer pre/post hooks on an actual SAC gradient step; gradients are post-clip; update delta measured across this optimizer step",
                    }
                )

            optimizer.register_step_pre_hook(before_step)
            optimizer.register_step_post_hook(after_step)

    def configure_longitudinal_diagnostics(self, enabled: bool = True) -> None:
        self._longitudinal_diagnostics_enabled = bool(enabled)

    def pop_longitudinal_diagnostic_events(self) -> list[dict[str, Any]]:
        events = self._longitudinal_diagnostic_events
        self._longitudinal_diagnostic_events = []
        return events

    def longitudinal_diagnostic_metadata(self) -> dict[str, Any]:
        def group_metadata(module: nn.Module, groups, optimizer):
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group.get("params", ())
            }
            result = {}
            for name, prefixes in groups.items():
                selected = [
                    (parameter_name, parameter)
                    for parameter_name, parameter in module.named_parameters()
                    if any(
                        prefix == parameter_name
                        or parameter_name.startswith(prefix + ".")
                        for prefix in prefixes
                    )
                ]
                total = sum(parameter.numel() for _, parameter in selected)
                owned = sum(
                    parameter.numel()
                    for _, parameter in selected
                    if id(parameter) in optimizer_ids
                )
                result[name] = {
                    "parameter_elements": total,
                    "trainable_elements": sum(
                        parameter.numel()
                        for _, parameter in selected
                        if parameter.requires_grad
                    ),
                    "optimizer_owned_elements": owned,
                    "optimizer_coverage": owned / total if total else None,
                }
            return result

        return {
            "enabled": True,
            "actor_residual_bound_pre_tanh": self.longitudinal_residual_bound,
            "actor_residual_output_init": "exact_zero",
            "actor_residual_target_speed_supremum_mps": 5.0 * 2.0 * th.tanh(
                th.tensor(self.longitudinal_residual_bound / 2.0)
            ).item(),
            "actor_changes_pre_tanh_speed_mean_only": True,
            "actor_log_std_unchanged": True,
            "critic_uses_private_action_conditioned_ct_q_residual": True,
            "critic_residual_output_init": "exact_zero",
            "shared_scene_encoder_receives_conflict_timing": False,
            "unsupported_conflict_timing_behavior": "zero_residual_with_explicit_invalid_reason",
            "optimizer_groups": {
                "actor": group_metadata(self.actor, LONGRES_ACTOR_GROUPS, self.actor.optimizer),
                "critic": group_metadata(self.critic, LONGRES_CRITIC_GROUPS, self.critic.optimizer),
            },
        }

    @contextmanager
    def longitudinal_residual_probe(self, name: str | None):
        if name != "longitudinal_residual_off":
            raise ValueError(f"unsupported longitudinal policy probe: {name}")
        with self.actor.longitudinal_residual_probe(name):
            yield

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["longitudinal_residual_bound"] = self.longitudinal_residual_bound
        return data


__all__ = [
    "BoundedLongitudinalResidualActor",
    "CONFLICT_TIMING_FEATURE_DIM",
    "LongitudinalResidualContinuousCritic",
    "LongitudinalResidualSceneRepSACPolicy",
    "summarize_conflict_timing",
]
