"""Read-only deterministic rollout traces for the Full-vs-TemporalGraph diagnosis.

The script loads frozen final checkpoints and records policy commands plus the
SUMO lane/position state.  It never trains, saves, or mutates a checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import SceneRepresentationSAC
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from tools.paper_evaluation_contract import (
    make_paper_evaluation_env,
    validate_model_environment_spaces,
)


RUNS = (
    "paper__full_balanced__carla__seed0",
    "paper__full_balanced__carla__seed1",
    "paper__temporal_graph__carla__seed0",
    "paper__full_balanced__roundabout_medium__seed2",
    "paper__full_balanced__roundabout_medium__seed1",
    "paper__temporal_graph__roundabout_medium__seed2",
    "paper__full_balanced__roundabout__seed2",
    "paper__full_balanced__cross__seed1",
    "paper__temporal_graph__cross__seed1",
)


def _vehicle_state(env: Any) -> dict[str, Any]:
    raw = env.unwrapped
    connection = raw._connection
    ego_id = raw.specification.ego_id
    if ego_id not in connection.vehicle.getIDList():
        return {"active": False}
    position = connection.vehicle.getPosition(ego_id)
    return {
        "active": True,
        "road_id": str(connection.vehicle.getRoadID(ego_id)),
        "lane_index": int(connection.vehicle.getLaneIndex(ego_id)),
        "x_sumo": float(position[0]),
        "y_sumo": float(position[1]),
        "speed_mps": float(connection.vehicle.getSpeed(ego_id)),
    }


def _lane_command(value: float) -> int:
    if value < -1.0 / 3.0:
        return -1
    if value > 1.0 / 3.0:
        return 1
    return 0


def trace_run(run_name: str) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "results_systematic_matrix" / "selected" / run_name
    arguments_payload = json.loads(
        (run_dir / "arguments.json").read_text(encoding="utf-8")
    )
    arguments = arguments_payload.get("requested_raw_steps", arguments_payload)
    env = make_paper_evaluation_env(
        arguments, traffic_protocol=str(arguments["traffic_protocol"])
    )
    model = SceneRepresentationSAC.load(
        run_dir / "final_model.zip", env=env, device="cpu"
    )
    validate_model_environment_spaces(model, env)
    episode_seed = int(arguments["seed"]) + 10_000
    records: list[dict[str, Any]] = []
    try:
        with source_evaluation_augmentation(model):
            observation, reset_info = env.reset(seed=episode_seed)
            while True:
                before = _vehicle_state(env)
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                command = _lane_command(float(action[1]))
                target_speed = float(np.clip((float(action[0]) + 1.0) * 5.0, 0.0, 10.0))
                observation, reward, terminated, truncated, info = env.step(action)
                records.append(
                    {
                        "decision": len(records),
                        "action_speed_raw": float(action[0]),
                        "action_lateral_raw": float(action[1]),
                        "target_speed_mps": target_speed,
                        "source_lane_command": command,
                        "sumo_lane_offset": int(env.unwrapped._sumo_lane_offset(command)),
                        "lane_change_applied": bool(info.get("lane_change_applied", False)),
                        "before": before,
                        "after": _vehicle_state(env),
                        "reward": float(info.get("undiscounted_reward", reward)),
                        "raw_steps": int(info.get("raw_simulation_steps", 0)),
                    }
                )
                if terminated or truncated:
                    outcome = {
                        "success": bool(info.get("is_success", False)),
                        "collision": bool(info.get("collision", False)),
                        "off_route": bool(info.get("off_route", False)),
                        "timeout": bool(info.get("max_time", False)),
                        "raw_steps": int(info.get("raw_simulation_steps", 0)),
                    }
                    break
    finally:
        env.close()

    lateral = np.asarray([row["source_lane_command"] for row in records], dtype=int)
    speeds = np.asarray([row["target_speed_mps"] for row in records], dtype=float)
    applied = np.asarray([row["lane_change_applied"] for row in records], dtype=bool)
    roads = [row["after"].get("road_id") for row in records if row["after"].get("active")]
    lanes = [row["after"].get("lane_index") for row in records if row["after"].get("active")]
    sample_indices = sorted(
        set([0, 1, 2, len(records) // 4, len(records) // 2, 3 * len(records) // 4, len(records) - 1])
    )
    return {
        "run": run_name,
        "algorithm": str(arguments["algo"]),
        "scenario": str(arguments["scenario"]),
        "training_seed": int(arguments["seed"]),
        "evaluation_seed": episode_seed,
        "traffic_variant": reset_info.get("traffic_variant"),
        "outcome": outcome,
        "decisions": len(records),
        "mean_target_speed_mps": float(speeds.mean()),
        "min_target_speed_mps": float(speeds.min()),
        "max_target_speed_mps": float(speeds.max()),
        "negative_lane_command_rate": float((lateral < 0).mean()),
        "keep_lane_command_rate": float((lateral == 0).mean()),
        "positive_lane_command_rate": float((lateral > 0).mean()),
        "lane_change_applied_rate": float(applied.mean()),
        "roads_visited": list(dict.fromkeys(roads)),
        "lane_indices_visited": sorted(set(int(value) for value in lanes)),
        "trace_samples": [records[index] for index in sample_indices if 0 <= index < len(records)],
    }


def main() -> None:
    payload = {
        "contract": "full-balanced-scenario-diagnosis/rollout-trace-v1",
        "read_only_checkpoint_use": True,
        "runs": [trace_run(run_name) for run_name in RUNS],
    }
    output = REPORT_DIR / "tables" / "selected_rollout_traces.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
