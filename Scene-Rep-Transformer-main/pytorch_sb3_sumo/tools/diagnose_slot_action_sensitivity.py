"""Diagnose how frozen scene latents drive SAC actions on identical observations.

The input is the observation dataset produced by ``run_latent_probes.py``.
Every checkpoint therefore sees byte-identical observations and risk labels.
For structured encoders, each slot is replaced by its dataset mean in turn;
this is less out-of-distribution than zeroing a standardized latent.  The
result is a diagnostic association, not a causal performance estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, TopoTemporalGraphExtractor
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from envs.sumo.paper_env import PaperSumoSceneEnv


CONTRACT = "topo-scene.slot-action-sensitivity/v1"
SLOT_SLICES = {
    "ego": slice(0, 32),
    "social": slice(32, 96),
    "route": slice(96, 128),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT")
    return name.strip(), Path(raw_path.strip())


def _lane_command(actions: np.ndarray) -> np.ndarray:
    """Mirror the environment's three-way lateral command discretisation."""

    return np.where(actions[:, 1] < -1.0 / 3.0, -1, np.where(actions[:, 1] > 1.0 / 3.0, 1, 0))


def _action_summary(actions: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = actions[mask]
    if selected.shape[0] == 0:
        return {"samples": 0, "identifiable": False}
    target_speed = np.clip((selected[:, 0] + 1.0) * 5.0, 0.0, 10.0)
    lane = _lane_command(selected)
    return {
        "samples": int(selected.shape[0]),
        "identifiable": True,
        "target_speed_mps_mean": float(target_speed.mean()),
        "target_speed_mps_std": float(target_speed.std()),
        "target_speed_mps_p10": float(np.quantile(target_speed, 0.1)),
        "target_speed_mps_p90": float(np.quantile(target_speed, 0.9)),
        "lateral_action_mean": float(selected[:, 1].mean()),
        # Physical left/right is scenario-contract dependent.  Keep the
        # policy-space sign here so the diagnostic cannot invert semantics.
        "lane_command_negative_rate": float((lane == -1).mean()),
        "lane_command_keep_rate": float((lane == 0).mean()),
        "lane_command_positive_rate": float((lane == 1).mean()),
        "longitudinal_saturation_rate": float((np.abs(selected[:, 0]) >= 0.99).mean()),
    }


def _risk_masks(minimum_ttc: np.ndarray) -> dict[str, np.ndarray]:
    ttc = minimum_ttc.reshape(-1)
    return {
        "all": np.ones(ttc.shape[0], dtype=bool),
        "critical_ttc_le_3s": ttc <= 3.0,
        "caution_ttc_3_to_6s": (ttc > 3.0) & (ttc <= 6.0),
        "clear_ttc_gt_6s": ttc > 6.0,
    }


def _actions_from_features(model: SceneRepresentationSAC, features: torch.Tensor) -> np.ndarray:
    actor = model.actor
    latent = actor.latent_pi(features)
    mean_actions = actor.mu(latent)
    if actor.use_sde:
        log_std = actor.log_std
        kwargs = {"latent_sde": latent}
    else:
        log_std = torch.clamp(actor.log_std(latent), LOG_STD_MIN, LOG_STD_MAX)
        kwargs = {}
    scaled = actor.action_dist.actions_from_params(
        mean_actions, log_std, deterministic=True, **kwargs
    )
    return model.policy.unscale_action(scaled.detach().cpu().numpy())


def _slot_delta_summary(
    reference: np.ndarray,
    ablated: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not mask.any():
        return {"samples": 0, "identifiable": False}
    delta = np.abs(reference[mask] - ablated[mask])
    reference_lane = _lane_command(reference[mask])
    ablated_lane = _lane_command(ablated[mask])
    return {
        "samples": int(mask.sum()),
        "identifiable": True,
        "mean_absolute_action_delta": delta.mean(axis=0).tolist(),
        "mean_absolute_target_speed_delta_mps": float(delta[:, 0].mean() * 5.0),
        "lane_command_flip_rate": float((reference_lane != ablated_lane).mean()),
    }


def _encode_and_act(
    checkpoint: Path,
    scenario: str,
    trajectory: np.ndarray,
    map_state: np.ndarray,
    risk_masks: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    env = PaperSumoSceneEnv(
        scenario=scenario,
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="evaluation",
        episode_limit_profile="source",
    )
    try:
        model = SceneRepresentationSAC.load(checkpoint.resolve(), env=env, device=device)
        extractor = model.actor.features_extractor
        feature_batches: list[torch.Tensor] = []
        with source_evaluation_augmentation(model), torch.no_grad():
            for start in range(0, trajectory.shape[0], batch_size):
                stop = min(start + batch_size, trajectory.shape[0])
                observations, _ = model.policy.obs_to_tensor(
                    {"trajectory": trajectory[start:stop], "map": map_state[start:stop]}
                )
                feature_batches.append(
                    model.actor.extract_features(observations, extractor).detach()
                )
            features = torch.cat(feature_batches, dim=0)
            actions = _actions_from_features(model, features)
            structured = isinstance(extractor, TopoTemporalGraphExtractor)
            slot_sensitivity: dict[str, Any] = {}
            if structured:
                feature_mean = features.mean(dim=0, keepdim=True)
                for slot, slot_slice in SLOT_SLICES.items():
                    ablated_features = features.clone()
                    ablated_features[:, slot_slice] = feature_mean[:, slot_slice]
                    ablated_actions = _actions_from_features(model, ablated_features)
                    slot_sensitivity[slot] = {
                        risk: _slot_delta_summary(actions, ablated_actions, mask)
                        for risk, mask in risk_masks.items()
                    }
        report = {
            "checkpoint": str(checkpoint.resolve()),
            "extractor": type(extractor).__name__,
            "structured_slots": structured,
            "action_summary_by_risk": {
                risk: _action_summary(actions, mask) for risk, mask in risk_masks.items()
            },
            "slot_mean_ablation": slot_sensitivity,
        }
        return report, actions
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    dataset_path = args.dataset.resolve()
    with np.load(dataset_path, allow_pickle=False) as dataset:
        trajectory = np.asarray(dataset["trajectory"])
        map_state = np.asarray(dataset["map"])
        minimum_ttc = np.asarray(dataset["minimum_ttc"])
        metadata = json.loads(str(dataset["metadata"]))
    risk_masks = _risk_masks(minimum_ttc)
    models: dict[str, Any] = {}
    action_arrays: dict[str, np.ndarray] = {}
    for name, checkpoint in args.model:
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        models[name], action_arrays[name] = _encode_and_act(
            checkpoint,
            str(metadata["scenario"]),
            trajectory,
            map_state,
            risk_masks,
            device=args.device,
            batch_size=args.batch_size,
        )

    pairwise: dict[str, Any] = {}
    names = list(action_arrays)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            difference = np.abs(action_arrays[left] - action_arrays[right])
            pairwise[f"{left}__vs__{right}"] = {
                risk: {
                    "samples": int(mask.sum()),
                    "mean_absolute_action_difference": difference[mask].mean(axis=0).tolist()
                    if mask.any()
                    else None,
                    "mean_absolute_target_speed_difference_mps": float(
                        difference[mask, 0].mean() * 5.0
                    )
                    if mask.any()
                    else None,
                    "lane_command_disagreement_rate": float(
                        (_lane_command(action_arrays[left][mask]) != _lane_command(action_arrays[right][mask])).mean()
                    )
                    if mask.any()
                    else None,
                }
                for risk, mask in risk_masks.items()
            }

    output = {
        "contract": CONTRACT,
        "computed_from_real_observations": True,
        "fabricated_values": False,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_metadata": metadata,
        "same_observations_for_all_models": True,
        "samples": int(trajectory.shape[0]),
        "risk_bin_counts": {name: int(mask.sum()) for name, mask in risk_masks.items()},
        "models": models,
        "pairwise_action_difference": pairwise,
        "interpretation_limit": (
            "Mean-slot ablation measures local policy dependence on an observational "
            "dataset; it is not a counterfactual closed-loop performance estimate."
        ),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
