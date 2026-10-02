"""Bounded integration checks for the new scenarios, without model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.random_intersection import (
    RANDOM_INTERSECTION_SCENARIOS,
    ensure_random_intersection_assets,
    random_intersection_seed,
)


def reset_signature(env: PaperSumoSceneEnv, seed: int) -> tuple[str, dict]:
    _, info = env.reset(seed=seed)
    connection = env._connection
    actors = [
        {
            "id": key,
            "route": list(connection.vehicle.getRoute(key)),
            "position": list(connection.vehicle.getPosition(key)),
            "speed": float(connection.vehicle.getSpeed(key)),
        }
        for key in sorted(connection.vehicle.getIDList())
    ]
    digest = hashlib.sha256(json.dumps(actors, sort_keys=True).encode()).hexdigest()
    return digest, info


def validate() -> dict:
    ensure_random_intersection_assets()
    cases = []
    for name in RANDOM_INTERSECTION_SCENARIOS:
        env = PaperSumoSceneEnv(scenario=name, traffic_partition="evaluation", traffic_split="validation", action_repeat=3)
        try:
            signature, reset_info = reset_signature(env, 10000)
            assert reset_info["traffic_sumo_seed"] == random_intersection_seed("validation", 10000)
            assert reset_info["traffic_partition_is_disjoint"]
            assert reset_info["traffic_partition_unit"] == "simulation_seed"
            assert not reset_info["source_endless_traffic"]
            assert reset_info["random_traffic"]["endless_reinsertions_enabled"] is False
            info = reset_info
            for _ in range(5):
                _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                if terminated or truncated:
                    break
            assert info["endless_traffic_reinsertions"] == 0
            assert info["random_traffic"]["control_elapsed_seconds"] > 0
            assert info["random_traffic"]["sumo_seed"] == reset_info["traffic_sumo_seed"]
            assert "other" not in info["random_traffic"]["departed_background_by_route_total"]
            cases.append({"scenario": name, "reset_signature": signature, "terminal_smoke_info": info["random_traffic"]})
            if name == "intersection_random_medium_v1":
                repeated, _ = reset_signature(env, 10000)
                different, _ = reset_signature(env, 10001)
                assert signature == repeated, "Same seed did not reproduce the initial traffic state"
                assert signature != different, "Different seeds did not change the initial traffic state"
        finally:
            env.close()
    domains = {}
    for split in ("train", "test"):
        env = PaperSumoSceneEnv(scenario="intersection_random_medium_v1", action_repeat=3)
        try:
            env.set_traffic_split(split)
            _, info = reset_signature(env, 10000)
            assert info["traffic_sumo_seed"] == random_intersection_seed(split, 10000)
            domains[split] = info["traffic_sumo_seed"]
        finally:
            env.close()
    assert len(set(domains.values()) | {random_intersection_seed("validation", 10000)}) == 3
    old = PaperSumoSceneEnv(scenario="intersection_sorted")
    try:
        command = old._sumo_command(42)
        assert command[command.index("--seed") + 1] == "42"
        assert old._random_traffic_config is None
    finally:
        old.close()
    return {
        "status": "passed",
        "kind": "environment integration smoke; no model training or learned-policy performance evaluation",
        "cases": cases,
        "additional_split_seeds": domains,
        "same_seed_reproducible": True,
        "different_seed_changes_traffic": True,
        "legacy_scenario_seed_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "runs" / "random_intersection_v1_validation" / "paper_env_integration.json")
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
