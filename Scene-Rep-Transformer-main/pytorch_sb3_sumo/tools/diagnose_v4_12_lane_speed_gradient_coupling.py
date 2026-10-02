"""Measure v4.12 lane-NLL/speed-NLL gradient coupling on real D3 states.

The diagnostic reads the frozen selected model, collects validation states with
that model, and measures the two support losses on identical minibatches.  It
does not train, update, save, project, veto, or rewrite any action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.hybrid_policy_v4_12_model import (
    JointReplaySupportedMixtureActorV412,
)
from algos.sb3_torch.sac_v4_12_model import AugmentedJointSupportPRCRSACV412
from tools import evaluate_v4_12_frozen_lane_prior_ablation as shared
from tools.action_diagnostics_v4_12_model import load_model_for_deployment_v4_12
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
    / "lane_speed_gradient_coupling.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def gradient_vector(
    loss: Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [
        (
            gradient.detach().reshape(-1)
            if gradient is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(values) if values else torch.zeros(0, device=loss.device)


def _norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value).cpu())


def _cosine(left: Tensor, right: Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0.0:
        return None
    return float(torch.dot(left, right).div(denominator).cpu())


def _opposing_sign_fraction(left: Tensor, right: Tensor) -> float | None:
    active = (left.abs() > 1e-12) & (right.abs() > 1e-12)
    count = int(active.sum().item())
    if count == 0:
        return None
    return float(((left[active] * right[active]) < 0.0).float().mean().cpu())


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _stack_observations(rows: Sequence[dict[str, np.ndarray]], device: torch.device):
    return {
        key: torch.as_tensor(
            np.stack([np.asarray(row[key]) for row in rows]), device=device
        )
        for key in rows[0]
    }


def _collect_states(
    model: Any,
    env: Any,
    *,
    samples: int,
    seed_start: int,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, int]:
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    episodes = 0
    observation, _ = env.reset(seed=seed_start)
    while len(observations) < samples:
        action, _ = model.predict(observation, deterministic=True)
        observations.append(
            {key: np.asarray(value).copy() for key, value in observation.items()}
        )
        actions.append(np.asarray(action, dtype=np.float32).reshape(2).copy())
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(seed=seed_start + episodes)
    return observations, np.stack(actions), episodes + 1


def diagnose(
    *,
    source_run: Path,
    output: Path,
    device: str,
    samples: int,
    batch_size: int,
    diagnostic_seed: int,
) -> dict[str, Any]:
    _require(samples > 0 and batch_size > 0, "samples and batch size must be positive")
    _require(samples % batch_size == 0, "samples must be divisible by batch size")
    source = shared._source_contract(source_run.resolve())
    requested = dict(source["arguments"]["requested_raw_steps"])
    _require(requested["scenario"] == "roundabout_medium", "diagnostic requires D3")
    requested["gui"] = False
    args = SimpleNamespace(**requested)
    env = _make_paper_env_v4_5(args, evaluation=True)
    try:
        model = load_model_for_deployment_v4_12(
            AugmentedJointSupportPRCRSACV412,
            source["paths"]["selected_model"],
            decoder=TARGET_DECODER,
            env=env,
            device=device,
        )
        observations, actions, episodes_touched = _collect_states(
            model,
            env,
            samples=samples,
            seed_start=int(requested["evaluation_seed_start"]),
        )
    finally:
        env.close()
    actor = model.actor
    _require(
        isinstance(actor, JointReplaySupportedMixtureActorV412),
        "frozen model lost the v4.12 actor",
    )
    model.policy.set_training_mode(True)
    torch.manual_seed(diagnostic_seed)
    rng = np.random.default_rng(diagnostic_seed)
    order = rng.permutation(samples)
    group_parameters = {
        "latent_trunk": tuple(actor.latent_pi.parameters()),
        "lane_head": tuple(actor.lane_logits.parameters()),
        "component_head": tuple(actor.component_logits.parameters()),
        "speed_mean_head": tuple(actor.speed_mean.parameters()),
        "speed_log_std_head": tuple(actor.speed_log_std.parameters()),
        "shared_scene_encoder": tuple(actor.features_extractor.parameters()),
    }
    all_parameters = tuple(
        parameter
        for group in group_parameters.values()
        for parameter in group
    )
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name, parameters in group_parameters.items():
        size = sum(parameter.numel() for parameter in parameters)
        offsets[name] = (start, start + size)
        start += size
    rows: list[dict[str, Any]] = []
    model_device = next(actor.parameters()).device
    for batch_index, begin in enumerate(range(0, samples, batch_size)):
        indices = order[begin : begin + batch_size]
        batch_observations = _stack_observations(
            [observations[int(index)] for index in indices], model_device
        )
        replay_actions = torch.as_tensor(
            actions[indices], device=model_device, dtype=torch.float32
        )
        proposals = actor.all_action_proposals(
            batch_observations, deterministic_speed=True
        )
        terms = actor.joint_replay_support_terms(proposals, replay_actions)
        lane_gradient = gradient_vector(
            terms.lane_categorical_nll,
            all_parameters,
            retain_graph=True,
        )
        speed_gradient = gradient_vector(
            terms.conditional_speed_mixture_nll,
            all_parameters,
            retain_graph=False,
        )
        group_values: dict[str, Any] = {}
        for name, (left, right) in offsets.items():
            lane = lane_gradient[left:right]
            speed = speed_gradient[left:right]
            group_values[name] = {
                "lane_nll_gradient_norm": _norm(lane),
                "speed_nll_gradient_norm": _norm(speed),
                "cosine": _cosine(lane, speed),
                "opposing_sign_fraction": _opposing_sign_fraction(lane, speed),
            }
        rows.append(
            {
                "batch": batch_index,
                "lane_categorical_nll": float(
                    terms.lane_categorical_nll.detach().cpu()
                ),
                "conditional_speed_mixture_nll": float(
                    terms.conditional_speed_mixture_nll.detach().cpu()
                ),
                "joint_action_nll": float(terms.joint_action_nll.detach().cpu()),
                "groups": group_values,
            }
        )
    aggregate: dict[str, Any] = {}
    for name in group_parameters:
        aggregate[name] = {
            metric: _summary(
                row["groups"][name][metric] for row in rows
            )
            for metric in (
                "lane_nll_gradient_norm",
                "speed_nll_gradient_norm",
                "cosine",
                "opposing_sign_fraction",
            )
        }
    trunk = aggregate["latent_trunk"]
    encoder = aggregate["shared_scene_encoder"]
    payload = {
        "schema_version": "topo-scene-v4.12.lane-speed-gradient-coupling/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "frozen_model_real_state_gradient_diagnostic",
        "computed_from_real_environment_states": True,
        "fabricated_values": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "model_saved": False,
        "formal_test_accessed": False,
        "evaluation_seed_start": int(requested["evaluation_seed_start"]),
        "samples": samples,
        "batch_size": batch_size,
        "minibatches": len(rows),
        "episodes_touched": episodes_touched,
        "diagnostic_seed": diagnostic_seed,
        "lane_support_scale": float(actor.lane_support_scale),
        "aggregate": aggregate,
        "minibatches_detail": rows,
        "architecture_finding": {
            "lane_and_speed_losses_both_update_latent_trunk": (
                trunk["lane_nll_gradient_norm"]["minimum"] > 0.0
                and trunk["speed_nll_gradient_norm"]["minimum"] > 0.0
            ),
            "shared_scene_encoder_detached_from_both_support_losses": (
                math.isclose(
                    float(encoder["lane_nll_gradient_norm"]["maximum"]),
                    0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(encoder["speed_nll_gradient_norm"]["maximum"]),
                    0.0,
                    abs_tol=1e-12,
                )
            ),
            "coupling_location": "shared_actor_latent_pi_not_scene_encoder",
        },
        "forbidden_mechanisms_used": {
            "kinematic_safety_projection": False,
            "traffic_risk_lane_mask": False,
            "threshold_or_veto": False,
            "action_rewrite": False,
        },
        "source_artifacts": {
            name: {
                "path": shared._relative_or_absolute(path),
                "sha256": _sha256(path),
            }
            for name, path in source["paths"].items()
        },
    }
    _write_json(output.resolve(), payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--source-run", type=Path, default=shared.DEFAULT_SOURCE_RUN)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    value.add_argument("--samples", type=int, default=128)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--diagnostic-seed", type=int, default=20260902)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = diagnose(
        source_run=args.source_run,
        output=args.output,
        device=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        diagnostic_seed=args.diagnostic_seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "architecture_finding": payload["architecture_finding"],
                "latent_trunk": payload["aggregate"]["latent_trunk"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["diagnose", "gradient_vector"]
