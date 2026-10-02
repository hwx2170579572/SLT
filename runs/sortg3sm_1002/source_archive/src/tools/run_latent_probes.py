"""Run leakage-resistant linear probes on identical SUMO observations.

The dataset is collected once with a fixed behaviour checkpoint and reused by
every encoder.  Splits are made by episode, not transition, so adjacent frames
cannot leak across train/test.  Route information is operationalised as the
current SUMO route-edge index; the report names this explicitly and marks the
probe unidentifiable when fewer than two classes are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import (
    SceneRepresentationSAC,
    StructuredLatent,
    TopoTemporalGraphExtractor,
)
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DATASET_CONTRACT = "topo-scene.probe-dataset/v1"
RESULT_CONTRACT = "topo-scene.latent-probes/v1"


def _last_actor_states(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.any(np.asarray(trajectory) != 0.0, axis=-1)
    states = np.zeros((trajectory.shape[0], trajectory.shape[-1]), dtype=np.float32)
    actor_valid = valid.any(axis=1)
    for actor in np.flatnonzero(actor_valid):
        index = int(valid[actor].sum()) - 1
        states[actor] = trajectory[actor, index]
    return states, actor_valid


def _transition_targets(
    observation: dict[str, np.ndarray],
    next_observation: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, float] | None:
    states, valid = _last_actor_states(observation["trajectory"])
    next_states, next_valid = _last_actor_states(next_observation["trajectory"])
    if not valid[0] or not next_valid[0]:
        return None
    ego = states[0]
    next_ego = next_states[0]
    cosine = math.cos(float(ego[2]))
    sine = math.sin(float(ego[2]))
    delta = next_ego[:2] - ego[:2]
    local_delta = np.asarray(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
        dtype=np.float32,
    )
    next_velocity = next_ego[3:5]
    local_velocity = np.asarray(
        [
            cosine * next_velocity[0] + sine * next_velocity[1],
            -sine * next_velocity[0] + cosine * next_velocity[1],
        ],
        dtype=np.float32,
    )
    dynamics = np.concatenate([local_delta, local_velocity]).astype(np.float32)

    neighbor_states = states[1:][valid[1:]]
    if neighbor_states.size == 0:
        return dynamics, 80.0, 20.0
    relative_position = neighbor_states[:, :2] - ego[None, :2]
    distances = np.linalg.norm(relative_position, axis=1)
    relative_velocity = neighbor_states[:, 3:5] - ego[None, 3:5]
    speed_squared = np.square(relative_velocity).sum(axis=1)
    dot = (relative_position * relative_velocity).sum(axis=1)
    approaching = dot < 0.0
    ttc = np.full(distances.shape, 20.0, dtype=np.float32)
    ttc[approaching] = np.minimum(
        20.0,
        -dot[approaching] / np.maximum(speed_squared[approaching], 1e-6),
    )
    return dynamics, float(distances.min()), float(ttc.min())


def collect_dataset(args: argparse.Namespace, dataset_path: Path) -> dict[str, Any]:
    env = PaperSumoSceneEnv(
        scenario=args.scenario,
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="evaluation",
        episode_limit_profile="source",
    )
    trajectories: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    dynamics: list[np.ndarray] = []
    minimum_distance: list[float] = []
    minimum_ttc: list[float] = []
    route_progress: list[int] = []
    episode_ids: list[int] = []
    episode_lengths: list[int] = []
    try:
        behaviour = SceneRepresentationSAC.load(
            args.behavior_model.resolve(), env=env, device=args.device
        )
        with source_evaluation_augmentation(behaviour):
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                terminated = truncated = False
                kept = 0
                while not (terminated or truncated):
                    ego_id = env.specification.ego_id
                    route_index = int(env._connection.vehicle.getRouteIndex(ego_id))
                    action, _ = behaviour.predict(observation, deterministic=True)
                    next_observation, _, terminated, truncated, _ = env.step(action)
                    targets = _transition_targets(observation, next_observation)
                    if targets is not None:
                        dynamic_target, distance_target, ttc_target = targets
                        trajectories.append(np.asarray(observation["trajectory"], dtype=np.float32))
                        maps.append(np.asarray(observation["map"], dtype=np.float32))
                        dynamics.append(dynamic_target)
                        minimum_distance.append(distance_target)
                        minimum_ttc.append(ttc_target)
                        route_progress.append(route_index)
                        episode_ids.append(episode)
                        kept += 1
                    observation = next_observation
                episode_lengths.append(kept)
    finally:
        env.close()
    if not trajectories:
        raise RuntimeError("Probe collection produced no valid transitions")
    metadata = {
        "contract": DATASET_CONTRACT,
        "computed_from_real_rollout": True,
        "scenario": args.scenario,
        "behavior_model": str(args.behavior_model.resolve()),
        "episodes": args.episodes,
        "evaluation_seed_start": args.seed,
        "transitions": len(trajectories),
        "episode_kept_transitions": episode_lengths,
        "labels": {
            "ego_dynamics": "next held-action local dx,dy,vx,vy",
            "minimum_vehicle_distance_m": "current observed actor geometry",
            "minimum_ttc_seconds": "constant-velocity estimate clipped to 20 s",
            "route_progress_class": "current SUMO vehicle.getRouteIndex; not a latent branch oracle",
        },
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        trajectory=np.stack(trajectories),
        map=np.stack(maps),
        ego_dynamics=np.stack(dynamics),
        minimum_distance=np.asarray(minimum_distance, dtype=np.float32)[:, None],
        minimum_ttc=np.asarray(minimum_ttc, dtype=np.float32)[:, None],
        route_progress=np.asarray(route_progress, dtype=np.int64),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
    )
    return metadata


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT")
    return name.strip(), Path(raw_path.strip())


def _load_encoder(checkpoint: Path, name: str, env: Any, device: str):
    """Load a frozen encoder with the loader that matches each method.

    ``SceneRepresentationSAC.load`` (the base class) cannot reload the v4_* /
    v4_13 checkpoints: their optimizers are structured differently, so the
    base loader's ``set_parameters`` raises on the optimizer group mismatch.
    Each method must therefore go through its own deployment loader, exactly
    as phase-3/phase-4 evaluation does.  Only the shared critic encoder is
    consumed downstream; the decoder is irrelevant to ``forward_tokens`` but
    is pinned to each method's native value so the load matches the artefact.
    """
    if name == "mst_slt":
        from algos.sb3_torch.sac import SceneRepresentationSAC

        return SceneRepresentationSAC.load(checkpoint.resolve(), env=env, device=device)
    if name == "v4_8":
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
        from tools.action_diagnostics_v4_6 import load_model_for_deployment
        from tools.checkpoint_decoder_selector_v4_6 import FUSION_DECODER

        return load_model_for_deployment(
            ConfidentActorFusionSACV45, checkpoint.resolve(),
            decoder=FUSION_DECODER, env=env, device=device,
        )
    if name == "v4_13":
        from algos.sb3_torch.sac_v4_13_model import (
            GradientIsolatedTemperedJointSupportSACV413,
        )
        from tools.action_diagnostics_v4_13_model import (
            load_model_for_deployment_v4_13,
        )
        from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER

        return load_model_for_deployment_v4_13(
            GradientIsolatedTemperedJointSupportSACV413, checkpoint.resolve(),
            decoder=TARGET_DECODER, env=env, device=device,
        )
    raise ValueError(
        f"Unknown probe model name {name!r}; expected mst_slt, v4_8, or v4_13"
    )


def _encode(
    name: str,
    checkpoint: Path,
    scenario: str,
    trajectory: np.ndarray,
    map_state: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # mst_slt was trained against the v1 observation (no lane_action_mask); the
    # v4_* methods expose the three-way lane_action_mask (PaperSumoSceneEnvV4).
    # forward_tokens() only reads ``trajectory`` and ``map``, so the mask is
    # needed solely to satisfy the load-time observation-space check.
    env_cls = PaperSumoSceneEnv if name == "mst_slt" else PaperSumoSceneEnvV4
    env = env_cls(
        scenario=scenario,
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="evaluation" if name == "mst_slt" else "all",
        episode_limit_profile="source",
    )
    outputs: dict[str, list[np.ndarray]] = {"full": []}
    try:
        model = _load_encoder(checkpoint, name, env, device)
        extractor = model.critic.features_extractor
        structured = isinstance(extractor, TopoTemporalGraphExtractor)
        if structured:
            outputs.update({"ego": [], "social": [], "route": []})
        with source_evaluation_augmentation(model), torch.no_grad():
            for start in range(0, trajectory.shape[0], batch_size):
                stop = min(start + batch_size, trajectory.shape[0])
                observation, _ = model.policy.obs_to_tensor(
                    {"trajectory": trajectory[start:stop], "map": map_state[start:stop]}
                )
                if structured:
                    latent = extractor.forward_tokens(observation)  # type: ignore[attr-defined]
                    if not isinstance(latent, StructuredLatent):
                        raise TypeError("forward_tokens() did not return StructuredLatent")
                    outputs["full"].append(latent.tensor.cpu().numpy())
                    outputs["ego"].append(latent.z_ego.cpu().numpy())
                    outputs["social"].append(latent.z_social.cpu().numpy())
                    outputs["route"].append(latent.z_route.cpu().numpy())
                else:
                    outputs["full"].append(extractor(observation).cpu().numpy())
        arrays = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
        metadata = {
            "checkpoint": str(checkpoint.resolve()),
            "extractor": type(extractor).__name__,
            "structured_slots": structured,
            "feature_dimensions": {name: int(value.shape[1]) for name, value in arrays.items()},
        }
        return arrays, metadata
    finally:
        env.close()


def _episode_split(
    episode_id: np.ndarray, *, seed: int, train_fraction: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    episodes = np.unique(episode_id)
    if episodes.size < 2:
        raise ValueError("At least two episodes are required for a leakage-resistant split")
    shuffled = episodes.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    train_count = min(max(1, int(math.floor(train_fraction * shuffled.size))), shuffled.size - 1)
    train_episodes = np.sort(shuffled[:train_count])
    test_episodes = np.sort(shuffled[train_count:])
    train = np.isin(episode_id, train_episodes)
    test = np.isin(episode_id, test_episodes)
    return train, test, {
        "split_unit": "episode",
        "seed": seed,
        "train_fraction": train_fraction,
        "train_episode_ids": train_episodes.tolist(),
        "test_episode_ids": test_episodes.tolist(),
        "train_samples": int(train.sum()),
        "test_samples": int(test.sum()),
    }


def _standardize_features(
    features: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train].mean(axis=0, keepdims=True)
    scale = features[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    return standardized, mean, scale


def _ridge_regression(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    x, _, _ = _standardize_features(features.astype(np.float64), train)
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    y = target.astype(np.float64)
    target_mean = y[train].mean(axis=0, keepdims=True)
    target_scale = y[train].std(axis=0, keepdims=True)
    target_scale[target_scale < 1e-8] = 1.0
    y_standard = (y - target_mean) / target_scale
    regularizer = np.eye(x.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x[train].T @ x[train] + regularizer, x[train].T @ y_standard[train])
    prediction = (x[test] @ weights) * target_scale + target_mean
    residual = prediction - y[test]
    baseline_residual = target_mean - y[test]
    mse = np.square(residual).mean(axis=0)
    baseline_mse = np.square(baseline_residual).mean(axis=0)
    denominator = np.square(y[test] - y[test].mean(axis=0, keepdims=True)).sum(axis=0)
    numerator = np.square(residual).sum(axis=0)
    r2 = 1.0 - numerator / np.maximum(denominator, 1e-12)
    return {
        "identifiable": True,
        "target_dimensions": int(y.shape[1]),
        "rmse": float(np.sqrt(mse.mean())),
        "rmse_per_dimension": np.sqrt(mse).tolist(),
        "mae": float(np.abs(residual).mean()),
        "r2_mean": float(r2.mean()),
        "r2_per_dimension": r2.tolist(),
        "constant_baseline_rmse": float(np.sqrt(baseline_mse.mean())),
    }


def _ridge_classification(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    classes = np.unique(target)
    train_classes = np.unique(target[train])
    test_classes = np.unique(target[test])
    if classes.size < 2 or train_classes.size < 2:
        return {
            "identifiable": False,
            "reason": "fewer than two route-progress classes in all/train samples",
            "classes": classes.tolist(),
            "train_classes": train_classes.tolist(),
            "test_classes": test_classes.tolist(),
        }
    x, _, _ = _standardize_features(features.astype(np.float64), train)
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    class_to_index = {int(value): index for index, value in enumerate(classes)}
    indices = np.asarray([class_to_index[int(value)] for value in target], dtype=np.int64)
    one_hot = np.eye(classes.size, dtype=np.float64)[indices]
    regularizer = np.eye(x.shape[1], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x[train].T @ x[train] + regularizer, x[train].T @ one_hot[train])
    predicted = (x[test] @ weights).argmax(axis=1)
    majority = int(np.bincount(indices[train]).argmax())
    recalls: list[float] = []
    for class_index in np.unique(indices[test]):
        class_mask = indices[test] == class_index
        recalls.append(float((predicted[class_mask] == class_index).mean()))
    return {
        "identifiable": True,
        "classes": classes.tolist(),
        "train_classes": train_classes.tolist(),
        "test_classes": test_classes.tolist(),
        "accuracy": float((predicted == indices[test]).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "majority_baseline_accuracy": float((indices[test] == majority).mean()),
    }


def evaluate(args: argparse.Namespace, dataset_path: Path) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata"]))
        if metadata.get("contract") != DATASET_CONTRACT:
            raise ValueError("Unsupported probe dataset contract")
        trajectory = np.asarray(dataset["trajectory"])
        map_state = np.asarray(dataset["map"])
        targets = {
            "ego_dynamics": np.asarray(dataset["ego_dynamics"]),
            "minimum_distance": np.asarray(dataset["minimum_distance"]),
            "minimum_ttc": np.asarray(dataset["minimum_ttc"]),
        }
        route_progress = np.asarray(dataset["route_progress"])
        episode_id = np.asarray(dataset["episode_id"])
    train, test, split = _episode_split(
        episode_id, seed=args.probe_seed, train_fraction=args.train_fraction
    )
    model_results: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    for name, checkpoint in args.model:
        slots, encoding_metadata = _encode(
            name,
            checkpoint,
            metadata["scenario"],
            trajectory,
            map_state,
            device=args.device,
            batch_size=args.batch_size,
        )
        model_metadata[name] = encoding_metadata
        slot_results: dict[str, Any] = {}
        for slot_name, features in slots.items():
            slot_results[slot_name] = {
                **{
                    target_name: _ridge_regression(
                        features, target, train, test, args.ridge
                    )
                    for target_name, target in targets.items()
                },
                "route_progress_class": _ridge_classification(
                    features, route_progress, train, test, args.ridge
                ),
                "latent_std_mean": float(features.std(axis=0).mean()),
                "collapsed_dimensions_below_1e-5": int(
                    (features.std(axis=0) < 1e-5).sum()
                ),
            }
        model_results[name] = slot_results
    return {
        "contract": RESULT_CONTRACT,
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "same_observations_for_all_models": True,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset": metadata,
        "split": split,
        "ridge": args.ridge,
        "models": model_metadata,
        "results": model_results,
        "interpretation_contract": {
            "probe_better_rl_not_better": "richer latent information not converted into policy benefit",
            "probe_and_rl_better": "representation enrichment is consistent with, but does not alone prove, the policy gain",
            "route_label_limitation": "SUMO route progress is a proxy, not a counterfactual route-choice label",
        },
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--behavior-model", type=Path, required=True)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--scenario", choices=tuple(PAPER_SCENARIOS), default="left_turn")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-seed", type=int, default=73)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--recollect", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.episodes < 2 or args.batch_size <= 0:
        raise ValueError("episodes must be >=2 and batch size positive")
    if not 0.0 < args.train_fraction < 1.0 or args.ridge < 0.0:
        raise ValueError("train fraction must be in (0,1) and ridge non-negative")
    dataset_path = args.dataset.resolve()
    collection = None
    if args.recollect or not dataset_path.is_file():
        collection = collect_dataset(args, dataset_path)
    result = evaluate(args, dataset_path)
    if collection is not None:
        result["collection_this_invocation"] = collection
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
