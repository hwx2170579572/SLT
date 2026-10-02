"""Evaluate an SB3/PyTorch checkpoint in a SUMO scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, evaluate_model
from envs.sumo.scenario_registry import available_scenarios
from envs.sumo.sumo_env import SumoSceneEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", choices=available_scenarios(), default="left_turn")
    parser.add_argument(
        "--algo",
        choices=(
            "scene_rep",
            "mst",
            "topo_scene",
            "topo_scene_balanced",
            "temporal_graph",
            "sac",
        ),
        default="scene_rep",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--path-length", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    env = SumoSceneEnv(
        scenario=args.scenario,
        neighbors=args.neighbors,
        history_steps=args.history_steps,
        path_length=args.path_length,
        action_repeat=args.action_repeat,
        reward_discount=args.discount,
        include_state_lstm=args.algo == "sac",
        state_lstm_only=args.algo == "sac",
        render_mode="human" if args.gui else None,
    )
    model_class = {
        "scene_rep": SceneRepresentationSAC,
        "mst": SceneRepresentationSAC,
        "topo_scene": SceneRepresentationSAC,
        "topo_scene_balanced": SceneRepresentationSAC,
        "temporal_graph": SceneRepresentationSAC,
        "sac": SceneRepresentationSAC,
    }[args.algo]
    try:
        model = model_class.load(args.model.resolve(), env=env, device=args.device)
        summary = evaluate_model(model, env, args.episodes, seed=args.seed)
    finally:
        env.close()
    output = summary.to_dict()
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.output.resolve().open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
