"""Measure whether a trained topology query attends to spatially relevant lanes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, TopoTemporalGraphExtractor
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from configs.sb3_configs import source_action_repeat
from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    traffic_partition = "all" if args.traffic_protocol == "source_all" else "evaluation"
    action_repeat = source_action_repeat("topo_scene", args.action_repeat)
    env = PaperSumoSceneEnv(
        scenario=args.scenario,
        neighbors=args.neighbors,
        history_steps=args.history_steps,
        path_length=args.path_length,
        action_repeat=action_repeat,
        reward_discount=args.discount,
        traffic_partition=traffic_partition,
        episode_limit_profile=args.episode_limit_profile,
    )
    expected_distances: list[float] = []
    top1_distances: list[float] = []
    mass_within_20m: list[float] = []
    mass_within_40m: list[float] = []
    entropy: list[float] = []
    episode_steps: list[int] = []
    try:
        model = SceneRepresentationSAC.load(args.model.resolve(), env=env, device=args.device)
        extractor = model.critic.features_extractor
        if not isinstance(extractor, TopoTemporalGraphExtractor) or not extractor.use_topology:
            raise TypeError("Checkpoint does not contain the full topology-query extractor")
        node_mask = extractor.topology_node_mask.detach().cpu().numpy().astype(bool)
        with source_evaluation_augmentation(model):
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    attention_tensor = extractor.last_topology_attention
                    distance_tensor = extractor.last_topology_squared_distance
                    if attention_tensor is None or distance_tensor is None:
                        raise RuntimeError("Extractor did not retain topology-query diagnostics")
                    attention = attention_tensor[0, 0].detach().cpu().numpy()
                    distances = np.sqrt(
                        np.maximum(
                            0.0,
                            distance_tensor[0, 0].detach().cpu().numpy(),
                        )
                    )
                    ego_valid = np.asarray(observation["trajectory"])[0, :, 0] != 0
                    current_index = max(int(ego_valid.sum()) - 1, 0)
                    weights = attention[current_index, node_mask].astype(np.float64)
                    lane_distances = distances[current_index, node_mask].astype(np.float64)
                    weight_sum = float(weights.sum())
                    if weight_sum > 0.0 and np.isfinite(weights).all():
                        weights = weights / weight_sum
                        expected_distances.append(float(np.dot(weights, lane_distances)))
                        top1_distances.append(float(lane_distances[int(np.argmax(weights))]))
                        mass_within_20m.append(float(weights[lane_distances <= 20.0].sum()))
                        mass_within_40m.append(float(weights[lane_distances <= 40.0].sum()))
                        entropy.append(
                            float(-(weights * np.log(np.maximum(weights, 1e-12))).sum())
                        )
                    observation, _, terminated, truncated, _ = env.step(action)
                    steps += 1
                episode_steps.append(steps)
    finally:
        env.close()

    top1_summary = _summary(top1_distances)
    within_40_summary = _summary(mass_within_40m)
    locality_passed = bool(
        top1_summary["p90"] is not None
        and float(top1_summary["p90"]) <= args.top1_p90_max_m
        and within_40_summary["mean"] is not None
        and float(within_40_summary["mean"]) >= args.minimum_mass_within_40m
    )
    return {
        "contract_version": 1,
        "computed_from_real_rollout": True,
        "model": str(args.model.resolve()),
        "scenario": args.scenario,
        "episodes": args.episodes,
        "evaluation_seed_start": args.seed,
        "traffic_protocol": args.traffic_protocol,
        "decision_observations": len(expected_distances),
        "episode_decision_steps": episode_steps,
        "metrics": {
            "expected_attention_distance_m": _summary(expected_distances),
            "top1_attention_distance_m": top1_summary,
            "attention_mass_within_20m": _summary(mass_within_20m),
            "attention_mass_within_40m": within_40_summary,
            "attention_entropy_nats": _summary(entropy),
        },
        "locality_gate": {
            "top1_p90_max_m": args.top1_p90_max_m,
            "minimum_mean_mass_within_40m": args.minimum_mass_within_40m,
            "passed": locality_passed,
        },
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(PAPER_SCENARIOS), required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--path-length", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--traffic-protocol", choices=("source_all", "frozen_80_20"), default="frozen_80_20"
    )
    parser.add_argument(
        "--episode-limit-profile", choices=("source", "paper"), default="source"
    )
    parser.add_argument("--top1-p90-max-m", type=float, default=40.0)
    parser.add_argument("--minimum-mass-within-40m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    result = diagnose(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
