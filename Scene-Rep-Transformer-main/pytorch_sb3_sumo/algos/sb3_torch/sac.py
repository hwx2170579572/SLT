"""Stable-Baselines3 SAC with the Scene-Rep auxiliary update."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import numpy as np
import torch as th
import torch.nn.functional as functional

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from .features import _nonzero_mask
from .graph_representation import (
    GraphSLTLosses,
    StructuredGraphRepresentationObjective,
)
from .representation import FutureRepresentationObjective
from .module_diagnostics import (
    collect_parameter_gradient_diagnostics,
    collect_parameter_update_diagnostics,
    snapshot_parameter_groups,
)
from .topo_temporal_features import StructuredLatent


def _clip_gradients_like_keras(parameters: list[th.nn.Parameter], max_norm: float) -> None:
    """Keras Optimizer(clipnorm=...) clips each gradient tensor separately."""

    for parameter in parameters:
        if parameter.grad is not None:
            th.nn.utils.clip_grad_norm_([parameter], max_norm)


class SceneRepresentationSAC(SAC):
    """SAC plus action-conditioned future scene-representation learning.

    The core rollout, replay buffer, checkpoint, callback, and vectorized-env
    behavior remain SB3-native.  Only the gradient update is extended, using
    the same replay sample for the auxiliary objective and SAC losses.
    """

    def __init__(
        self,
        *args: Any,
        representation_coef: float = 1.0,
        representation_learning_rate: float | None = None,
        representation_heads: int = 2,
        representation_separate_target_projector: bool = False,
        representation_online_target_encoder: bool = True,
        structured_representation: bool = False,
        action_embedding_dim: int = 64,
        max_grad_norm: float = 5.0,
        raw_learning_starts: int = 5000,
        **kwargs: Any,
    ) -> None:
        self.representation_coef = float(representation_coef)
        if self.representation_coef < 0.0:
            raise ValueError("representation_coef cannot be negative")
        self.representation_learning_rate = representation_learning_rate
        self.representation_heads = int(representation_heads)
        self.representation_separate_target_projector = bool(
            representation_separate_target_projector
        )
        self.representation_online_target_encoder = bool(
            representation_online_target_encoder
        )
        self.structured_representation = bool(structured_representation)
        self.action_embedding_dim = int(action_embedding_dim)
        self.max_grad_norm = float(max_grad_norm)
        self.raw_learning_starts = int(raw_learning_starts)
        self._raw_steps_seen = 0
        self._pending_raw_gradient_steps: int | None = None
        self.representation: (
            FutureRepresentationObjective
            | StructuredGraphRepresentationObjective
            | None
        )
        self.representation_optimizer: th.optim.Optimizer | None
        self._representation_parameters: list[th.nn.Parameter] = []
        self._last_graph_slt_losses: dict[str, th.Tensor] = {}
        self._training_diagnostic_stats: dict[str, dict[str, float | int]] = {}
        self._q_td_diagnostics_sampled = False
        super().__init__(*args, **kwargs)

    def _accumulate_training_stat(self, name: str, value: float) -> None:
        if not np.isfinite(value):
            return
        row = self._training_diagnostic_stats.setdefault(
            name,
            {
                "count": 0,
                "sum": 0.0,
                "minimum": value,
                "maximum": value,
                "last": value,
            },
        )
        row["count"] = int(row["count"]) + 1
        row["sum"] = float(row["sum"]) + value
        row["minimum"] = min(float(row["minimum"]), value)
        row["maximum"] = max(float(row["maximum"]), value)
        row["last"] = value

    def training_diagnostics(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, row in self._training_diagnostic_stats.items():
            count = int(row["count"])
            output[name] = {
                "count": count,
                "mean": float(row["sum"]) / max(count, 1),
                "minimum": float(row["minimum"]),
                "maximum": float(row["maximum"]),
                "last": float(row["last"]),
            }
        return output

    def _queue_module_diagnostic_event(self, source, update, metrics):
        """Retain sampled telemetry until the normal rollout callback saves it."""
        if not hasattr(self, "_module_diagnostic_events"):
            self._module_diagnostic_events = []
        self._module_diagnostic_events.append({
            "source": str(source),
            "updates": None if update is None else int(update),
            "raw_steps": int(getattr(self, "_raw_steps_seen", self.num_timesteps)),
            "decision_steps": int(self.num_timesteps),
            "metrics": dict(metrics),
            "timing": "collection_training_position; replay samples are not the current rollout episode",
        })

    def pop_module_diagnostic_events(self):
        """Drain detached scalar events; never retains a computation graph."""
        events = getattr(self, "_module_diagnostic_events", [])
        self._module_diagnostic_events = []
        return events

    def _record_encoder_activation_snapshot(
        self,
        extractor,
        diagnostic_values: dict[str, list[float]],
    ) -> None:
        """Queue a newly sampled forward snapshot immediately at its call site."""
        if not getattr(extractor, "behavior_diagnostics_enabled", False):
            return
        sample_id = getattr(extractor, "diagnostic_sample_index", None)
        if (
            sample_id is None
            or int(sample_id) <= 0
            or sample_id == getattr(self, "_last_encoder_diagnostic_sample", None)
        ):
            return
        self._last_encoder_diagnostic_sample = int(sample_id)
        current = extractor.diagnostic_values() if hasattr(extractor, "diagnostic_values") else {}
        metrics: dict[str, float | int | None] = {}
        scalar_items: list[tuple[str, th.Tensor]] = []
        for name, value in current.items():
            if value is None:
                metrics[str(name)] = None
                continue
            tensor = th.as_tensor(value).detach()
            if tensor.numel() != 1:
                continue
            scalar_items.append((str(name), tensor.reshape(())))
        if scalar_items:
            device = scalar_items[0][1].device
            numbers = th.stack([value.to(device) for _, value in scalar_items]).cpu().tolist()
            for (name, _), number in zip(scalar_items, numbers):
                numeric = float(number)
                metrics[name] = numeric if np.isfinite(numeric) else None
                if np.isfinite(numeric):
                    diagnostic_values.setdefault(name, []).append(numeric)

        sampled_update = metrics.get("diagnostic_sample_update_index")
        event_update = (
            int(sampled_update)
            if sampled_update is not None and np.isfinite(float(sampled_update))
            and int(sampled_update) > 0
            else None
        )
        self._queue_module_diagnostic_event(
            "train_forward_activation",
            event_update,
            metrics,
        )

    @staticmethod
    def _run_encoder_callsite(
        extractor,
        site: str,
        update_number: int | None,
        force_sample: bool,
        forward_call,
    ):
        setter = getattr(extractor, "set_diagnostic_context", None)
        if not getattr(extractor, "behavior_diagnostics_enabled", False) or not callable(setter):
            return forward_call()
        previous = setter(site, update_number, force_sample=force_sample)
        try:
            return forward_call()
        finally:
            restore = getattr(extractor, "restore_diagnostic_context", None)
            if callable(restore):
                restore(previous)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.representation = None
        self.representation_optimizer = None
        if self.representation_coef > 0.0:
            feature_dim = int(self.critic.features_extractor.features_dim)
            action_dim = int(self.action_space.shape[0])
            if self.structured_representation:
                if not hasattr(self.critic.features_extractor, "forward_tokens"):
                    raise TypeError(
                        "structured_representation requires a feature extractor "
                        "with forward_tokens()"
                    )
                self.representation = StructuredGraphRepresentationObjective(
                    feature_dim=feature_dim,
                    action_dim=action_dim,
                    action_embedding_dim=self.action_embedding_dim,
                ).to(self.device)
            else:
                self.representation = FutureRepresentationObjective(
                    feature_dim=feature_dim,
                    action_dim=action_dim,
                    action_embedding_dim=self.action_embedding_dim,
                    num_heads=self.representation_heads,
                    separate_target_projector=self.representation_separate_target_projector,
                ).to(self.device)
            learning_rate = (
                float(self.representation_learning_rate)
                if self.representation_learning_rate is not None
                else float(self.lr_schedule(1.0))
            )
            parameters = list(self.critic.features_extractor.parameters()) + [
                parameter
                for parameter in self.representation.parameters()
                if parameter.requires_grad
            ]
            unique_parameters = list(
                {id(parameter): parameter for parameter in parameters}.values()
            )
            self._representation_parameters = unique_parameters
            self.representation_optimizer = th.optim.NAdam(
                unique_parameters,
                lr=learning_rate,
                eps=1e-7,
            )
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # The released TensorFlow SAC uses Adam(beta_1=0.5) for log alpha.
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef],
                lr=float(self.lr_schedule(1.0)),
                betas=(0.5, 0.999),
                eps=1e-7,
            )

    def _representation_step(self, replay_data: Any) -> th.Tensor:
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
        self._last_graph_slt_losses = {}
        if self.structured_representation:
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
            loss = graph_losses.total
            self._last_graph_slt_losses = graph_losses.detached()
        else:
            online_features = self.critic.extract_features(
                replay_data.observations, self.critic.features_extractor
            )
            with th.no_grad():
                target_features = target_critic.extract_features(
                    representation_next_observations,
                    target_critic.features_extractor,
                )
            loss = self.representation(
                online_features,
                replay_data.actions,
                target_features,
                sample_mask=sample_mask,
            )
            if not isinstance(loss, th.Tensor):
                raise TypeError("FutureRepresentationObjective must return a Tensor")
        self.representation_optimizer.zero_grad()
        (self.representation_coef * loss).backward()
        _clip_gradients_like_keras(
            self._representation_parameters, self.max_grad_norm
        )
        self.representation_optimizer.step()
        return loss.detach()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self._pending_raw_gradient_steps is not None:
            gradient_steps = self._pending_raw_gradient_steps
            self._pending_raw_gradient_steps = None
        if gradient_steps <= 0:
            return
        train_started = time.perf_counter()
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if (
            self.representation_optimizer is not None
            and self.representation_learning_rate is None
        ):
            optimizers.append(self.representation_optimizer)
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        entropy_losses: list[float] = []
        entropy_coefficients: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        representation_losses: list[float] = []
        graph_slt_slot_losses: dict[str, list[float]] = {
            "graph_slt_ego_loss": [],
            "graph_slt_social_loss": [],
            "graph_slt_route_loss": [],
        }
        # Sparse optimizer telemetry reuses the already-computed SAC tensors.
        # It adds no forward/backward pass and only synchronizes a few scalars
        # on the first actual update and every 1,000 subsequent updates.
        q_td_diagnostics: dict[str, list[float]] = {
            "q1_mean_sampled": [],
            "q2_mean_sampled": [],
            "target_q_mean_sampled": [],
            "q1_abs_td_mean_sampled": [],
            "q2_abs_td_mean_sampled": [],
        }
        diagnostic_values: dict[str, list[float]] = {
            "topology_attention_entropy": [],
            "latent_std": [],
            "ego_latent_std": [],
            "social_latent_std": [],
            "route_latent_std": [],
            "slot_scale_ratio": [],
            "graph_mean_edge_weight": [],
        }

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            update_number = int(self._n_updates) + gradient_step + 1
            extractor = self.critic.features_extractor
            force_update_sample = update_number == 1 or update_number % 1000 == 0
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.representation is not None:
                representation_loss = self._run_encoder_callsite(
                    extractor,
                    "representation_objective_online",
                    update_number,
                    False,
                    lambda: self._representation_step(replay_data),
                )
                representation_losses.append(float(representation_loss.cpu()))
                self._record_encoder_activation_snapshot(extractor, diagnostic_values)
                for name in graph_slt_slot_losses:
                    value = self._last_graph_slt_losses.get(name)
                    if value is not None:
                        graph_slt_slot_losses[name].append(float(value.cpu()))

            if self.use_sde:
                self.actor.reset_noise()

            # Match the released persistent GradientTape forward order.  Each
            # encoder call draws its own source random rotation, so this order
            # is also part of seeded experiment behavior.
            current_q_values = self._run_encoder_callsite(
                extractor,
                "critic_td_current",
                update_number,
                force_update_sample,
                lambda: self.critic(
                    replay_data.observations, replay_data.actions
                ),
            )
            self._record_encoder_activation_snapshot(extractor, diagnostic_values)
            actions_pi, log_prob = self._run_encoder_callsite(
                extractor,
                "actor_policy_current",
                update_number,
                False,
                lambda: self.actor.action_log_prob(replay_data.observations),
            )
            self._record_encoder_activation_snapshot(extractor, diagnostic_values)
            log_prob = log_prob.reshape(-1, 1)

            entropy_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                entropy_coefficient = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                # Match the original objective, which differentiates alpha =
                # exp(log_alpha), instead of SB3's default log-alpha proxy.
                entropy_loss = -(
                    th.exp(self.log_ent_coef)
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                entropy_losses.append(float(entropy_loss.detach().cpu()))
            else:
                entropy_coefficient = self.ent_coef_tensor
            entropy_coefficients.append(float(entropy_coefficient.detach().cpu()))

            target_extractor = self.critic_target.features_extractor
            target_was_training = target_extractor.training
            target_extractor.train(True)
            with th.no_grad():
                next_actions, next_log_prob = self._run_encoder_callsite(
                    extractor,
                    "actor_policy_next_action_no_grad",
                    update_number,
                    False,
                    lambda: self.actor.action_log_prob(
                        replay_data.next_observations
                    ),
                )
                self._record_encoder_activation_snapshot(extractor, diagnostic_values)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values = th.min(next_q_values, dim=1, keepdim=True).values
                next_q_values = next_q_values - entropy_coefficient * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (
                    1.0 - replay_data.dones
                ) * discounts * next_q_values
            target_extractor.train(target_was_training)

            critic_loss = 0.5 * sum(
                functional.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.detach().cpu()))
            if (
                getattr(extractor, "behavior_diagnostics_enabled", False)
                and (
                    not self._q_td_diagnostics_sampled
                    or update_number == 1
                    or update_number % 1000 == 0
                )
            ):
                self._q_td_diagnostics_sampled = True
                detached_target = target_q_values.detach()
                for critic_index, current_q in enumerate(current_q_values, start=1):
                    q_td_diagnostics[f"q{critic_index}_mean_sampled"].append(
                        float(current_q.detach().mean().cpu())
                    )
                    q_td_diagnostics[
                        f"q{critic_index}_abs_td_mean_sampled"
                    ].append(
                        float((current_q.detach() - detached_target).abs().mean().cpu())
                    )
                q_td_diagnostics["target_q_mean_sampled"].append(
                    float(detached_target.mean().cpu())
                )

            # Q weights are constants for the policy update, while gradients
            # through Q with respect to actions remain available. Build this
            # graph before applying the critic update: TensorFlow's persistent
            # tape differentiates the pre-update forward pass even though the
            # source applies the critic optimizer before the actor optimizer.
            critic_parameters = list(self.critic.parameters())
            requires_grad = [parameter.requires_grad for parameter in critic_parameters]
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            q_values_pi = th.cat(
                self._run_encoder_callsite(
                    extractor,
                    "critic_actor_value",
                    update_number,
                    False,
                    lambda: self.critic(replay_data.observations, actions_pi),
                ),
                dim=1,
            )
            self._record_encoder_activation_snapshot(extractor, diagnostic_values)
            min_qf_pi = th.min(q_values_pi, dim=1, keepdim=True).values
            actor_loss = (entropy_coefficient * log_prob - min_qf_pi).mean()
            actor_losses.append(float(actor_loss.detach().cpu()))
            for parameter, original_value in zip(critic_parameters, requires_grad):
                parameter.requires_grad_(original_value)

            actor_parameters = [
                parameter
                for parameter in self.actor.parameters()
                if id(parameter)
                not in {id(item) for item in self.actor.features_extractor.parameters()}
            ]

            # Compute every gradient from the shared pre-update forward state,
            # then apply optimizers in the exact released order: critic,
            # targets, actor, temperature.
            self.critic.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            if entropy_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
            critic_loss.backward()
            # Inspect the existing critic TD gradients before clipping. This is
            # read-only and does not add an optimizer step or another backward.
            encoder_diagnostic_update = (
                getattr(extractor, "behavior_diagnostics_enabled", False)
                and (not getattr(self, "_encoder_grad_diagnostics_sampled", False)
                     or update_number % 1000 == 0)
            )
            update_snapshot = None
            if encoder_diagnostic_update:
                group_factory = getattr(extractor, "diagnostic_parameter_groups", None)
                groups = group_factory() if callable(group_factory) else {"encoder_all": ("",)}
                gradient_metrics = collect_parameter_gradient_diagnostics(extractor, groups)
                gradient_metrics["critic_td_update_index"] = float(update_number)
                self._encoder_grad_diagnostics_sampled = True
                self._queue_module_diagnostic_event(
                    "critic_td_gradient_pre_clip", update_number, gradient_metrics
                )
                for name, value in gradient_metrics.items():
                    if value is not None and np.isfinite(value):
                        diagnostic_values.setdefault(name, []).append(float(value))
            actor_loss.backward()
            if entropy_loss is not None:
                entropy_loss.backward()
            _clip_gradients_like_keras(critic_parameters, self.max_grad_norm)
            _clip_gradients_like_keras(actor_parameters, self.max_grad_norm)
            if encoder_diagnostic_update:
                snapshot_groups = {
                    name: prefixes
                    for name, prefixes in groups.items()
                    if name != "encoder_all"
                }
                update_snapshot = snapshot_parameter_groups(
                    extractor, snapshot_groups, self.critic.optimizer
                )
            self.critic.optimizer.step()
            if update_snapshot is not None:
                update_metrics = collect_parameter_update_diagnostics(update_snapshot)
                update_metrics["critic_td_update_index"] = float(update_number)
                self._queue_module_diagnostic_event(
                    "critic_td_parameter_update", update_number, update_metrics
                )
                for name, value in update_metrics.items():
                    if value is not None and np.isfinite(value):
                        diagnostic_values.setdefault(name, []).append(float(value))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                if self.representation is not None:
                    self.representation.update_target(self.tau)

            self.actor.optimizer.step()
            if entropy_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.step()

            if hasattr(extractor, "diagnostic_values"):
                current_diagnostics = extractor.diagnostic_values()  # type: ignore[attr-defined]
                if getattr(extractor, "behavior_diagnostics_enabled", False):
                    # The extractor publishes detached scalar snapshots only on
                    # its sampling schedule. Do not count a cached snapshot as
                    # a fresh measurement on every SAC update.
                    sample_id = getattr(extractor, "diagnostic_sample_index", None)
                    if (sample_id is not None and int(sample_id) > 0
                            and sample_id != getattr(self, "_last_encoder_diagnostic_sample", None)):
                        self._last_encoder_diagnostic_sample = sample_id
                        event_metrics = {}
                        for name, value in current_diagnostics.items():
                            if value is None:
                                event_metrics[name] = None
                                continue
                            tensor = th.as_tensor(value).detach()
                            if tensor.numel() != 1:
                                continue
                            number = float(tensor.cpu())
                            event_metrics[name] = number if np.isfinite(number) else None
                            if np.isfinite(number):
                                diagnostic_values.setdefault(name, []).append(number)
                        event_metrics["diagnostic_sample_index"] = int(sample_id)
                        self._queue_module_diagnostic_event(
                            "train_forward_activation", update_number, event_metrics
                        )
                else:
                    # Keep the legacy diagnostic behavior for existing models.
                    for name in diagnostic_values:
                        value = current_diagnostics.get(name)
                        if value is not None and bool(th.isfinite(value).all()):
                            diagnostic_values[name].append(float(value.mean().cpu()))

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        ent_coef_mean = float(np.mean(entropy_coefficients))
        actor_loss_mean = float(np.mean(actor_losses))
        critic_loss_mean = float(np.mean(critic_losses))
        self.logger.record("train/ent_coef", ent_coef_mean)
        self.logger.record("train/actor_loss", actor_loss_mean)
        self.logger.record("train/critic_loss", critic_loss_mean)
        self._accumulate_training_stat("train/ent_coef", ent_coef_mean)
        self._accumulate_training_stat("train/actor_loss", actor_loss_mean)
        self._accumulate_training_stat("train/critic_loss", critic_loss_mean)
        if entropy_losses:
            ent_coef_loss_mean = float(np.mean(entropy_losses))
            self.logger.record("train/ent_coef_loss", ent_coef_loss_mean)
            self._accumulate_training_stat(
                "train/ent_coef_loss", ent_coef_loss_mean
            )
        if representation_losses:
            representation_mean = float(np.mean(representation_losses))
            self.logger.record("train/representation_loss", representation_mean)
            self._accumulate_training_stat(
                "train/representation_loss", representation_mean
            )
        if self.structured_representation and representation_losses:
            graph_slt_mean = float(np.mean(representation_losses))
            self.logger.record("train/graph_slt_loss", graph_slt_mean)
            self._accumulate_training_stat("train/graph_slt_loss", graph_slt_mean)
        for name, values in graph_slt_slot_losses.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(f"train/{name}", value)
                self._accumulate_training_stat(f"train/{name}", value)
        for name, values in diagnostic_values.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(f"diagnostic/{name}", value)
                self._accumulate_training_stat(f"diagnostic/{name}", value)
        for name, values in q_td_diagnostics.items():
            if values:
                value = float(np.mean(values))
                self.logger.record(f"diagnostic/{name}", value)
                self._accumulate_training_stat(f"diagnostic/{name}", value)
        train_ms_per_gradient_step = (
            (time.perf_counter() - train_started) * 1000.0 / gradient_steps
        )
        self._accumulate_training_stat(
            "performance/train_ms_per_gradient_step",
            train_ms_per_gradient_step,
        )

    def set_raw_step_interval(
        self,
        previous: int,
        current: int,
        *,
        checkpoint_steps: Iterable[int] = (),
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> int:
        """Reproduce the source update ordering inside one held action.

        The legacy runner advances one raw simulator step at a time, updates
        from the existing replay buffer after every eligible intermediate
        step, appends the held-action transition at the final step, and then
        performs that final update. SUMO action repeat executes those raw steps
        in one Gymnasium call, whose callback still runs before SB3 stores the
        transition. We therefore run the intermediate updates here and leave
        at most the final update pending for SB3's normal post-store train().
        """

        if current < previous:
            raise ValueError("raw-step counters must be monotonic")

        checkpoints = sorted({int(raw_step) for raw_step in checkpoint_steps})
        if any(not previous < raw_step < current for raw_step in checkpoints):
            raise ValueError(
                "mid-interval checkpoint steps must be strictly between "
                "previous and current"
            )
        if checkpoints and checkpoint_callback is None:
            raise ValueError("checkpoint_callback is required for checkpoint_steps")

        first_train_step = int(self.raw_learning_starts)

        def eligible_updates_between(start_exclusive: int, end_inclusive: int) -> int:
            return max(
                0,
                int(end_inclusive)
                - max(int(start_exclusive), first_train_step - 1),
            )

        # Clear the post-store override while explicitly training on the old
        # replay contents. The paper/source profile always warms up for 5,000
        # raw steps, so the buffer is populated before this path is reached.
        self._pending_raw_gradient_steps = None
        intermediate_updates = 0
        cursor = int(previous)
        for raw_step in checkpoints:
            segment_updates = eligible_updates_between(cursor, raw_step)
            if segment_updates > 0:
                self.train(segment_updates, batch_size=self.batch_size)
            intermediate_updates += segment_updates
            self._raw_steps_seen = raw_step
            assert checkpoint_callback is not None
            checkpoint_callback(raw_step)
            cursor = raw_step

        # The update at ``current`` must remain pending until SB3 has stored
        # the held-action transition. Everything through ``current - 1`` uses
        # the old replay contents and can be applied here.
        segment_updates = eligible_updates_between(cursor, int(current) - 1)
        if segment_updates > 0:
            self.train(segment_updates, batch_size=self.batch_size)
        intermediate_updates += segment_updates

        self._raw_steps_seen = int(current)
        final_update = int(current > previous and current >= first_train_step)
        self._pending_raw_gradient_steps = final_update
        return intermediate_updates

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        del learning_starts
        effective_learning_starts = (
            self.num_timesteps + 1
            if self._raw_steps_seen < self.raw_learning_starts
            else 0
        )
        return super()._sample_action(
            effective_learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        if self.representation is not None:
            state_dicts.extend(["representation", "representation_optimizer"])
        return state_dicts, variables
