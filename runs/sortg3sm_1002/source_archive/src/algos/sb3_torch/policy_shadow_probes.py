"""Bounded same-state, single-branch policy probes for normal train/eval flow.

These probes estimate local functional sensitivity of the deterministic policy
to one representation intervention. They do not step an environment, train a
probe, or measure performance causality.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from .topo_temporal_features import StructuredLatent


PROBE_NAMES = (
    "st_spatial_off",
    "st_temporal_current_only",
    "st_social_ego_only",
    "rt_intent_injection_off",
    "rt_route_readout_zero",
    "topology_actor_intent_off",
    "topology_relations_off",
    "topology_goal_off",
    "route_reachability_off",
    "slot_ego_zero",
    "slot_social_zero",
    "slot_route_zero",
)
NOOP_PROBE_NAME = "__noop__"
PROBE_SCHEMA_VERSION = "policy_shadow_probes_v1"


def _json_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return []
    if array.ndim > 1:
        array = array.reshape((-1, array.shape[-1]))[0]
    array = array.reshape(-1)
    return [float(item) if np.isfinite(item) else None for item in array]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def _clone_diagnostic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@contextmanager
def _preserve_policy_runtime(model: Any, encoder: Any):
    """Restore state that an inference-only probe must not perturb."""
    policy = getattr(model, "policy", None)
    modules = list(policy.modules()) if policy is not None else []
    module_modes = [(module, bool(module.training)) for module in modules]
    buffers = [
        (buffer, buffer.detach().clone())
        for module in modules
        for buffer in module.buffers(recurse=False)
    ]
    parameters = [parameter for module in modules for parameter in module.parameters(recurse=False)]
    gradients = [
        (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in parameters
    ]

    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    encoder_state: list[tuple[Any, str, Any]] = []
    if encoder is not None:
        # Normal extractor forwards publish lightweight runtime snapshots on
        # the encoder and its nested topology modules (for example
        # ``last_topology_squared_distance``). A shadow forward must not
        # replace the snapshot that the real rollout/eval just produced.
        # Capture these ephemeral fields across the whole module tree rather
        # than only the top-level encoder's diagnostic counters.
        for module in encoder.modules() if hasattr(encoder, "modules") else (encoder,):
            for name, value in vars(module).items():
                lowered = name.lower()
                if (
                    name == "_diagnostics"
                    or "diagnostic" in lowered
                    or "last_" in lowered
                    or "_last" in lowered
                    or "cache" in lowered
                    or "capture" in lowered
                ):
                    encoder_state.append((module, name, _clone_diagnostic_value(value)))

    try:
        yield
    finally:
        for buffer, saved in buffers:
            if buffer.shape == saved.shape:
                buffer.copy_(saved)
        for parameter, saved_grad in gradients:
            parameter.grad = None if saved_grad is None else saved_grad.clone()
        for module, was_training in module_modes:
            module.training = was_training
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for module, name, value in encoder_state:
            current = getattr(module, name, None)
            if isinstance(current, dict) and isinstance(value, dict):
                current.clear()
                current.update(value)
            elif isinstance(current, list) and isinstance(value, list):
                current[:] = value
            else:
                setattr(module, name, value)


def _find_encoder(model: Any, explicit: Any = None) -> Any:
    if explicit is not None:
        return explicit
    policy = getattr(model, "policy", None)
    actor = getattr(policy, "actor", None)
    return getattr(actor, "features_extractor", None)


def _capture_predict(model: Any, observation: Any, encoder: Any) -> dict[str, Any]:
    actor = getattr(getattr(model, "policy", None), "actor", None)
    mu = getattr(actor, "mu", None)
    extractor = getattr(actor, "features_extractor", encoder)
    captured: dict[str, Any] = {}
    handles = []

    if mu is not None and hasattr(mu, "register_forward_hook"):
        def _capture_mu(_module, _inputs, output):
            captured["mean_pre_tanh"] = _json_vector(output)

        handles.append(mu.register_forward_hook(_capture_mu))
    if extractor is not None and hasattr(extractor, "register_forward_hook"):
        def _capture_features(_module, _inputs, output):
            if isinstance(output, StructuredLatent):
                captured["slot_latents"] = {
                    "ego": _json_vector(output.z_ego),
                    "social": _json_vector(output.z_social),
                    "route": _json_vector(output.z_route),
                }
                captured["feature_latent"] = _json_vector(output.tensor)
            elif isinstance(output, torch.Tensor):
                captured["feature_latent"] = _json_vector(output)

        handles.append(extractor.register_forward_hook(_capture_features))

    try:
        with torch.no_grad():
            action, _ = model.predict(observation, deterministic=True)
    finally:
        for handle in handles:
            handle.remove()

    model_action = _json_vector(action)
    mean = captured.get("mean_pre_tanh")
    normalized = None
    if mean is not None:
        normalized = [
            float(np.tanh(value)) if value is not None else None for value in mean
        ]
    return {
        "valid": model_action is not None,
        "mean_pre_tanh": mean,
        "action_normalized_mean": normalized,
        "model_action": model_action,
        "action_decoded": None,
        "feature_latent": captured.get("feature_latent"),
        "slot_latents": captured.get("slot_latents"),
    }


def capture_policy_snapshot(
    model: Any,
    observation: Any,
    *,
    encoder: Any = None,
    action_decoder: Any = None,
) -> dict[str, Any]:
    """Run one deterministic baseline prediction and return JSON-safe signals."""
    encoder = _find_encoder(model, encoder)
    started = time.perf_counter()
    try:
        with _preserve_policy_runtime(model, encoder):
            if encoder is not None and hasattr(encoder, "policy_shadow_probe"):
                with encoder.policy_shadow_probe(None) as captured_latents:
                    snapshot = _capture_predict(model, observation, encoder)
                    _add_context_latents(snapshot, captured_latents)
            else:
                snapshot = _capture_predict(model, observation, encoder)
        snapshot["valid"] = bool(snapshot.get("valid") and snapshot.get("mean_pre_tanh") is not None)
        snapshot["invalid_reason"] = None if snapshot["valid"] else "actor_mean_or_action_unavailable"
    except Exception as exc:  # optional diagnostics must not fail a real rollout
        snapshot = {
            "valid": False,
            "invalid_reason": "baseline_capture_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "mean_pre_tanh": None,
            "action_normalized_mean": None,
            "model_action": None,
            "action_decoded": None,
            "feature_latent": None,
            "slot_latents": None,
        }
    snapshot["duration_seconds"] = float(time.perf_counter() - started)
    if action_decoder is not None and snapshot.get("model_action") is not None:
        try:
            snapshot["action_decoded"] = _json_safe(
                action_decoder(np.asarray(snapshot["model_action"], dtype=np.float32))
            )
        except Exception as exc:
            snapshot["action_decode_error"] = f"{type(exc).__name__}: {exc}"
    return _json_safe(snapshot)


def get_policy_shadow_probe_specs(encoder: Any) -> list[dict[str, Any]]:
    """Return the stable probe inventory and applicability for an encoder."""
    if encoder is None or not hasattr(encoder, "policy_shadow_probe_applicability"):
        return [
            {"probe_name": name, "applicable": False, "invalid_reason": "encoder_unsupported"}
            for name in PROBE_NAMES
        ]
    return encoder.policy_shadow_probe_applicability()


def _add_context_latents(snapshot: dict[str, Any], captured: Mapping[str, Any]) -> None:
    if not captured:
        return
    slot_keys = {"slot_ego": "ego", "slot_social": "social", "slot_route": "route"}
    slots = {
        destination: _json_vector(captured[source])
        for source, destination in slot_keys.items()
        if source in captured
    }
    if slots:
        snapshot["slot_latents"] = slots
    if "feature_latent" in captured:
        snapshot["feature_latent"] = _json_vector(captured["feature_latent"])


def _action_signal_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    normalized = snapshot.get("action_normalized_mean")
    if not normalized:
        return {"tanh_saturation_margin": None, "lane_threshold_distance": None}
    margins = [
        None if value is None else float(1.0 - abs(value)) for value in normalized
    ]
    lane_distance = None
    if len(normalized) >= 2 and normalized[1] is not None:
        lane_distance = float(
            min(abs(float(normalized[1]) - 1.0 / 3.0), abs(float(normalized[1]) + 1.0 / 3.0))
        )
    return {
        "tanh_saturation_margin": margins,
        "lane_threshold_distance": lane_distance,
    }


def run_policy_shadow_suite(
    model: Any,
    encoder: Any,
    observation: Any,
    *,
    baseline_snapshot: Mapping[str, Any],
    metadata: Mapping[str, Any],
    action_decoder: Any = None,
) -> list[dict[str, Any]]:
    """Evaluate each applicable single-branch intervention at one fixed state.

    Inapplicable branches are returned as explicit NA rows. A failed shadow
    prediction becomes an invalid row and never masks the normal rollout.
    """
    base = dict(baseline_snapshot)
    specs = get_policy_shadow_probe_specs(encoder)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["probe_name"])
        row = dict(_json_safe(metadata))
        row.update(
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "probe_name": name,
                "applicable": bool(spec.get("applicable", False)),
                "valid": False,
                "invalid_reason": spec.get("invalid_reason"),
                "baseline_mean_pre_tanh": base.get("mean_pre_tanh"),
                "baseline_action_normalized_mean": base.get("action_normalized_mean"),
                "baseline_model_action": base.get("model_action"),
                "baseline_action_decoded": base.get("action_decoded"),
                "baseline_feature_latent": base.get("feature_latent"),
                "baseline_slot_latents": base.get("slot_latents"),
                "baseline_capture_duration_seconds": base.get("duration_seconds"),
                "baseline_tanh_saturation_margin": _action_signal_fields(base)[
                    "tanh_saturation_margin"
                ],
                "baseline_lane_threshold_distance": _action_signal_fields(base)[
                    "lane_threshold_distance"
                ],
                "shadow_mean_pre_tanh": None,
                "shadow_action_normalized_mean": None,
                "shadow_model_action": None,
                "shadow_action_decoded": None,
                "shadow_feature_latent": None,
                "shadow_slot_latents": None,
                "shadow_tanh_saturation_margin": None,
                "shadow_lane_threshold_distance": None,
                "action_delta_normalized": None,
                "action_delta_abs_normalized": None,
                "action_delta_l2_normalized": None,
                "action_delta_pre_tanh": None,
                "action_delta_abs_pre_tanh": None,
                "action_delta_l2_pre_tanh": None,
                "shadow_duration_seconds": 0.0,
            }
        )
        if not row["applicable"]:
            row["invalid_reason"] = row["invalid_reason"] or "branch_inactive"
            rows.append(_json_safe(row))
            continue
        if not bool(base.get("valid", False)):
            row["invalid_reason"] = "baseline_capture_failed"
            row["error_type"] = base.get("error_type")
            row["error_message"] = base.get("error_message")
            rows.append(_json_safe(row))
            continue
        if encoder is None or not hasattr(encoder, "policy_shadow_probe"):
            row["invalid_reason"] = "encoder_probe_context_unsupported"
            rows.append(_json_safe(row))
            continue

        started = time.perf_counter()
        try:
            with _preserve_policy_runtime(model, encoder):
                with encoder.policy_shadow_probe(name) as captured_latents:
                    shadow = _capture_predict(model, observation, encoder)
                    _add_context_latents(shadow, captured_latents)
            row["shadow_duration_seconds"] = float(time.perf_counter() - started)
            row["shadow_mean_pre_tanh"] = shadow.get("mean_pre_tanh")
            row["shadow_action_normalized_mean"] = shadow.get("action_normalized_mean")
            row["shadow_model_action"] = shadow.get("model_action")
            row["shadow_feature_latent"] = shadow.get("feature_latent")
            row["shadow_slot_latents"] = shadow.get("slot_latents")
            shadow_signals = _action_signal_fields(shadow)
            row["shadow_tanh_saturation_margin"] = shadow_signals[
                "tanh_saturation_margin"
            ]
            row["shadow_lane_threshold_distance"] = shadow_signals[
                "lane_threshold_distance"
            ]
            if action_decoder is not None and shadow.get("model_action") is not None:
                try:
                    row["shadow_action_decoded"] = _json_safe(
                        action_decoder(np.asarray(shadow["model_action"], dtype=np.float32))
                    )
                except Exception as exc:
                    row["action_decode_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["shadow_action_decoded"] = shadow.get("action_decoded")
            baseline_action = base.get("action_normalized_mean")
            shadow_action = shadow.get("action_normalized_mean")
            if baseline_action is not None and shadow_action is not None and len(baseline_action) == len(shadow_action):
                delta = [
                    None if left is None or right is None else float(right - left)
                    for left, right in zip(baseline_action, shadow_action)
                ]
                finite_delta = [value for value in delta if value is not None]
                row["action_delta_normalized"] = delta
                row["action_delta_abs_normalized"] = [
                    None if value is None else abs(value) for value in delta
                ]
                row["action_delta_l2_normalized"] = float(
                    np.linalg.norm(np.asarray(finite_delta, dtype=np.float64))
                ) if len(finite_delta) == len(delta) else None
            baseline_mean = base.get("mean_pre_tanh")
            shadow_mean = shadow.get("mean_pre_tanh")
            if baseline_mean is not None and shadow_mean is not None and len(baseline_mean) == len(shadow_mean):
                mean_delta = [
                    None if left is None or right is None else float(right - left)
                    for left, right in zip(baseline_mean, shadow_mean)
                ]
                finite_mean_delta = [value for value in mean_delta if value is not None]
                row["action_delta_pre_tanh"] = mean_delta
                row["action_delta_abs_pre_tanh"] = [
                    None if value is None else abs(value) for value in mean_delta
                ]
                row["action_delta_l2_pre_tanh"] = float(
                    np.linalg.norm(np.asarray(finite_mean_delta, dtype=np.float64))
                ) if len(finite_mean_delta) == len(mean_delta) else None
            row["valid"] = bool(shadow.get("valid") and shadow.get("mean_pre_tanh") is not None)
            row["invalid_reason"] = None if row["valid"] else "shadow_action_or_mean_unavailable"
        except Exception as exc:
            row["shadow_duration_seconds"] = float(time.perf_counter() - started)
            row["invalid_reason"] = "probe_exception"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        rows.append(_json_safe(row))
    return rows


__all__ = [
    "NOOP_PROBE_NAME",
    "PROBE_NAMES",
    "PROBE_SCHEMA_VERSION",
    "capture_policy_snapshot",
    "get_policy_shadow_probe_specs",
    "run_policy_shadow_suite",
]
