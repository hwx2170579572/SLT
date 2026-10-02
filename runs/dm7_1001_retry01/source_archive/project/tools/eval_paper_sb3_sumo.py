"""Evaluate one final SB3 checkpoint on the released-source SUMO scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SourcePPO, SceneRepresentationSAC, evaluate_model_detailed
from configs.sb3_configs import source_action_repeat
from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.ppo_env import PaperPpoCarlaEnv, PaperPpoRgbEnv
from envs.sumo.sumo_env import EGO_CONTROL_PROFILES
from tools.paper_evaluation_contract import (
    build_evaluation_provenance,
    validate_model_environment_spaces,
)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=tuple(PAPER_SCENARIOS), default="left_turn"
    )
    parser.add_argument(
        "--algo",
        choices=(
            "scene_rep",
            "mst",
            "topo_scene",
            "topo_scene_balanced",
            "temporal_graph",
            "sac",
            "ppo",
        ),
        default="scene_rep",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--path-length", type=int, default=10)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help="Source default: 3 for off-policy algorithms and 1 for PPO",
    )
    parser.add_argument(
        "--ego-control-profile", choices=EGO_CONTROL_PROFILES, default="direct"
    )
    parser.add_argument(
        "--traffic-protocol",
        choices=("source_all", "frozen_80_20"),
        default="source_all",
    )
    parser.add_argument(
        "--episode-limit-profile", choices=("source", "paper"), default="source"
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--sumo-step-seconds", type=float, default=0.1)
    parser.add_argument("--policy-action-hold", type=int, default=None)
    parser.add_argument("--expected-training-raw-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    args.action_repeat = source_action_repeat(args.algo, args.action_repeat)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    traffic_partition = (
        "all" if args.traffic_protocol == "source_all" else "evaluation"
    )
    if args.algo == "ppo":
        ppo_env_class = PaperPpoCarlaEnv if args.scenario == "carla" else PaperPpoRgbEnv
        env = ppo_env_class(
            scenario=args.scenario,
            action_repeat=args.action_repeat,
            reward_discount=args.discount,
            training=False,
            ego_control_profile=args.ego_control_profile,
            traffic_partition=traffic_partition,
            episode_limit_profile=args.episode_limit_profile,
            render_mode="human" if args.gui else None,
        )
    else:
        env = PaperSumoSceneEnv(
            scenario=args.scenario,
            neighbors=args.neighbors,
            history_steps=args.history_steps,
            path_length=args.path_length,
            action_repeat=args.action_repeat,
            reward_discount=args.discount,
            ego_control_profile=args.ego_control_profile,
            include_state_lstm=args.algo == "sac",
            state_lstm_only=args.algo == "sac",
            traffic_partition=traffic_partition,
            episode_limit_profile=args.episode_limit_profile,
            render_mode="human" if args.gui else None,
        )
    model_class = {
        "scene_rep": SceneRepresentationSAC,
        "mst": SceneRepresentationSAC,
        "topo_scene": SceneRepresentationSAC,
        "topo_scene_balanced": SceneRepresentationSAC,
        "temporal_graph": SceneRepresentationSAC,
        "sac": SceneRepresentationSAC,
        "ppo": SourcePPO,
    }[args.algo]
    model_path = args.model.resolve()
    try:
        model = model_class.load(model_path, env=env, device=args.device)
        space_contract = validate_model_environment_spaces(model, env)
        trained_raw_steps = getattr(model, "_raw_steps_seen", None)
        if trained_raw_steps is None and isinstance(model, SourcePPO):
            # PPO exposes every simulator tick as one rollout transition in
            # both the SMARTS-RGB and CARLA-vector source contracts.
            trained_raw_steps = model.num_timesteps
        if args.expected_training_raw_steps is not None:
            if trained_raw_steps is None:
                raise ValueError(
                    "The selected model does not expose the source raw-step clock"
                )
            if int(trained_raw_steps) < args.expected_training_raw_steps:
                raise ValueError(
                    f"Checkpoint has only {trained_raw_steps} raw steps; "
                    f"expected at least {args.expected_training_raw_steps}"
                )
        report = evaluate_model_detailed(
            model,
            env,
            episodes=args.episodes,
            seed=args.seed,
            sumo_step_seconds=args.sumo_step_seconds,
            policy_action_hold=(
                args.policy_action_hold
                if args.policy_action_hold is not None
                else (3 if args.algo == "ppo" and args.scenario == "carla" else 1)
            ),
        )
        evaluation_provenance = build_evaluation_provenance(
            model=model,
            env=env,
            report=report,
            algorithm=args.algo,
            scenario=args.scenario,
            traffic_protocol=args.traffic_protocol,
            episode_limit_profile=args.episode_limit_profile,
            evaluation_seed_start=args.seed,
            environment_factory="tools.eval_paper_sb3_sumo",
            space_contract=space_contract,
        )
    finally:
        env.close()

    output = {
        "model": str(model_path),
        "algorithm": args.algo,
        "scenario": args.scenario,
        "evaluation_seed_start": args.seed,
        "trained_raw_steps": (
            int(trained_raw_steps) if trained_raw_steps is not None else None
        ),
        "evaluation_provenance": evaluation_provenance,
        "completion_time_population_std_ddof": 0,
        **report.to_dict(),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
