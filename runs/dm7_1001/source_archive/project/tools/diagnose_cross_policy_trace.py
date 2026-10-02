"""Trace deterministic cross-scenario policies without changing their weights.

The report produced by this tool is intentionally diagnostic: it records the
requested/effective speed, route progress, road occupancy, and terminal event
for a small number of episodes under the same paper environment contract used
by formal evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC, SourcePPO
from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.ppo_env import PaperPpoRgbEnv
from tools.paper_evaluation_contract import validate_model_environment_spaces


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algo", choices=("sac", "ppo"), required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--traffic-protocol",
        choices=("source_all", "frozen_80_20"),
        default="source_all",
    )
    parser.add_argument(
        "--episode-limit-profile", choices=("source", "paper"), default="source"
    )
    parser.add_argument("--sample-every-decisions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _vehicle_state(env: Any) -> dict[str, Any] | None:
    connection = env._connection
    ego_id = env.specification.ego_id
    if connection is None or ego_id not in connection.vehicle.getIDList():
        return None
    return {
        "speed_mps": float(connection.vehicle.getSpeed(ego_id)),
        "distance_m": float(connection.vehicle.getDistance(ego_id)),
        "road_id": str(connection.vehicle.getRoadID(ego_id)),
        "route_index": int(connection.vehicle.getRouteIndex(ego_id)),
        "lane_index": int(connection.vehicle.getLaneIndex(ego_id)),
        "lane_position_m": float(connection.vehicle.getLanePosition(ego_id)),
        "position_xy": [
            float(value) for value in connection.vehicle.getPosition(ego_id)
        ],
    }


def _make_env(args: argparse.Namespace) -> Any:
    partition = "all" if args.traffic_protocol == "source_all" else "evaluation"
    common = {
        "scenario": "cross",
        "ego_control_profile": "smarts_ackermann_proxy",
        "traffic_partition": partition,
        "episode_limit_profile": args.episode_limit_profile,
    }
    if args.algo == "ppo":
        return PaperPpoRgbEnv(
            **common,
            action_repeat=1,
            reward_discount=0.99,
            training=False,
        )
    return PaperSumoSceneEnv(
        **common,
        neighbors=5,
        history_steps=10,
        path_length=10,
        action_repeat=3,
        reward_discount=0.99,
        include_state_lstm=True,
        state_lstm_only=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.sample_every_decisions <= 0:
        raise ValueError("--sample-every-decisions must be positive")

    env = _make_env(args)
    model_class = SourcePPO if args.algo == "ppo" else SceneRepresentationSAC
    model = model_class.load(args.model.resolve(), env=env, device=args.device)
    space_contract = validate_model_environment_spaces(model, env)
    episode_reports: list[dict[str, Any]] = []

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            observation, reset_info = env.reset(seed=episode_seed)
            initial_state = _vehicle_state(env)
            requested_speeds: list[float] = []
            effective_speeds: list[float] = []
            actual_speeds: list[float] = []
            distances: list[float] = []
            lane_commands: Counter[int] = Counter()
            roads: Counter[str] = Counter()
            samples: list[dict[str, Any]] = []
            terminated = truncated = False
            terminal_info: dict[str, Any] = dict(reset_info)
            decision = 0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                action_array = np.asarray(action, dtype=np.float32).reshape(2)
                requested_speed, lane_command = env.adapt_action(action_array)
                requested_speeds.append(float(requested_speed))
                lane_commands[int(lane_command)] += 1
                observation, _, terminated, truncated, terminal_info = env.step(
                    action_array
                )
                state = _vehicle_state(env)
                effective_speeds.append(
                    float(terminal_info.get("effective_target_speed", 0.0))
                )
                if state is not None:
                    actual_speeds.append(float(state["speed_mps"]))
                    distances.append(float(state["distance_m"]))
                    roads[str(state["road_id"])] += 1
                if (
                    decision % args.sample_every_decisions == 0
                    or terminated
                    or truncated
                ):
                    samples.append(
                        {
                            "decision": decision,
                            "raw_steps": int(
                                terminal_info.get("raw_simulation_steps", 0)
                            ),
                            "action": [float(value) for value in action_array],
                            "requested_speed_mps": float(requested_speed),
                            "lane_command": int(lane_command),
                            "effective_target_speed_mps": float(
                                terminal_info.get("effective_target_speed", 0.0)
                            ),
                            "vehicle_state": state,
                        }
                    )
                decision += 1

            final_state = _vehicle_state(env)
            initial_distance = (
                float(initial_state["distance_m"]) if initial_state is not None else 0.0
            )
            final_distance = (
                float(final_state["distance_m"])
                if final_state is not None
                else (max(distances) if distances else initial_distance)
            )
            episode_reports.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "traffic_variant": terminal_info.get("traffic_variant"),
                    "success": bool(terminal_info.get("is_success", False)),
                    "collision": bool(terminal_info.get("collision", False)),
                    "off_route": bool(terminal_info.get("off_route", False)),
                    "timeout": bool(terminal_info.get("max_time", False)),
                    "raw_steps": int(
                        terminal_info.get("raw_simulation_steps", 0)
                    ),
                    "decision_steps": decision,
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "distance_travelled_m": final_distance - initial_distance,
                    "requested_speed_mps": _stats(requested_speeds),
                    "effective_target_speed_mps": _stats(effective_speeds),
                    "actual_speed_mps": _stats(actual_speeds),
                    "lane_command_counts": {
                        str(key): value for key, value in sorted(lane_commands.items())
                    },
                    "road_decision_counts": dict(roads.most_common()),
                    "sampled_trace": samples,
                }
            )
    finally:
        env.close()

    output = {
        "diagnostic_only": True,
        "weights_updated": False,
        "model": str(args.model.resolve()),
        "algorithm": args.algo,
        "scenario": "cross",
        "traffic_protocol": args.traffic_protocol,
        "episode_limit_profile": args.episode_limit_profile,
        "evaluation_seed_start": args.seed,
        "episodes": args.episodes,
        "environment_class": f"{type(env).__module__}.{type(env).__qualname__}",
        "max_episode_steps": int(env.max_episode_steps),
        "model_environment_space_contract": space_contract,
        "episode_reports": episode_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
