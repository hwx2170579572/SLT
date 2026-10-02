"""Probe the [0, 20] control endpoint in real SUMO for all three scenes."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.high_density_single_seed_100ep_speed20_v3 import (  # noqa: E402
    EGO_SPEED_CONTROL_INTERVAL_MPS,
)
from tools import train_high_density_single_seed_100ep_speed20_v3 as trainer  # noqa: E402
from tools.high_density_same_scene_v1_common import load_protocol  # noqa: E402


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results_hd_ss100_s20_v3"
    / "control_endpoint_probe.json"
)


def _arguments(scenario: str) -> Namespace:
    return Namespace(
        scenario=scenario,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split="validation",
    )


def _probe_scene(
    protocol: dict[str, Any], scenario: str, overlay_root: Path
) -> dict[str, Any]:
    density = dict(protocol["density"][scenario])
    factory = trainer._make_speed20_v3_environment_factory(
        adapter="base",
        density=density,
        overlay_root=overlay_root,
    )
    environment = factory(_arguments(scenario), evaluation=True)
    try:
        _, reset_info = environment.reset(seed=10_000)
        ego_id = environment.specification.ego_id
        traci_max_speed = float(
            environment._connection.vehicle.getMaxSpeed(ego_id)
        )
        normalized_action = np.asarray([1.0, 0.0], dtype=np.float32)
        mapped_speed, lane_command = environment.adapt_action(normalized_action)
        _, _, terminated, truncated, step_info = environment.step(
            normalized_action
        )
        active = ego_id in environment._connection.vehicle.getIDList()
        actual_speed = (
            float(environment._connection.vehicle.getSpeed(ego_id))
            if active
            else None
        )
        row = {
            "scenario": scenario,
            "seed": 10_000,
            "normalized_action": normalized_action.tolist(),
            "mapped_target_speed_mps": float(mapped_speed),
            "lane_command": int(lane_command),
            "traci_ego_max_speed_mps": traci_max_speed,
            "effective_target_speed_mps": float(
                step_info["effective_target_speed"]
            ),
            "actual_ego_speed_after_one_decision_mps": actual_speed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "environment_experiment_id": reset_info[
                "high_density_comparison_experiment_id"
            ],
            "reported_control_interval_mps": reset_info[
                "ego_speed_control_interval_mps"
            ],
        }
        expected_max = EGO_SPEED_CONTROL_INTERVAL_MPS[1]
        if row["mapped_target_speed_mps"] != expected_max:
            raise AssertionError(f"{scenario}: action mapping did not reach 20 m/s")
        if row["traci_ego_max_speed_mps"] != expected_max:
            raise AssertionError(f"{scenario}: TraCI ego maxSpeed is not 20 m/s")
        if row["effective_target_speed_mps"] != expected_max:
            raise AssertionError(f"{scenario}: effective target did not reach 20 m/s")
        if row["reported_control_interval_mps"] != [0.0, 20.0]:
            raise AssertionError(f"{scenario}: environment interval metadata drifted")
        return row
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=trainer.DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    output_path = args.output.resolve()
    overlay_root = output_path.parent / "_control_probe_overlays_speed20_v3"
    rows = [
        _probe_scene(protocol, scenario, overlay_root)
        for scenario in protocol["matrix"]["scenario_order"]
    ]
    payload = {
        "schema_version": "speed20-control-endpoint-probe/v3",
        "computed_from_real_sumo": True,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "passed": True,
        "scenes": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
