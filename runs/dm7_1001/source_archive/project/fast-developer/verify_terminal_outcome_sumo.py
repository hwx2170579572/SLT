"""One-case SUMO collision-removal check; forced intervention, never an eval.

Run from the configured PyTorch environment with a fresh --output-dir.  The
script uses the ordinary D1 SAC+MLP environment factory, changes only live
TraCI state for this single test episode, then lets the normal env.step path
observe the native SUMO event and resolve terminal flags/reward.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as base
import train_intersection_yield_v2_d1 as d1
from envs.sumo.random_intersection import get_random_intersection_config


SCENARIO = "intersection_random_darrl_medium_v1"
R1_REVISION = "darrl_r1_20260930"
R2_REVISION = "darrl_r2_20260930"


def _argument(command: list[str], key: str) -> str:
    return command[command.index(key) + 1]


def _vehicle_state(connection: Any, vehicle_id: str) -> dict[str, Any]:
    return {
        "id": vehicle_id,
        "road_id": str(connection.vehicle.getRoadID(vehicle_id)),
        "lane_id": str(connection.vehicle.getLaneID(vehicle_id)),
        "lane_position_m": float(connection.vehicle.getLanePosition(vehicle_id)),
        "speed_mps": float(connection.vehicle.getSpeed(vehicle_id)),
        "route": list(connection.vehicle.getRoute(vehicle_id)),
        "route_index": int(connection.vehicle.getRouteIndex(vehicle_id)),
        "length_m": float(connection.vehicle.getLength(vehicle_id)),
    }


def _collision_payload(connection: Any) -> dict[str, Any]:
    simulation = connection.simulation
    collisions = []
    for item in simulation.getCollisions():
        row = {}
        for output_name, candidates in (
            ("collider", ("collider",)),
            ("victim", ("victim",)),
            ("lane", ("lane",)),
            ("position", ("pos", "position")),
            ("time", ("time",)),
            ("kind", ("type", "collisionType")),
        ):
            for name in candidates:
                value = getattr(item, name, None)
                if value is not None:
                    row[output_name] = value
                    break
        collisions.append(row)
    return {
        "arrived_ids": sorted(str(value) for value in simulation.getArrivedIDList()),
        "colliding_vehicle_ids": sorted(
            str(value) for value in simulation.getCollidingVehiclesIDList()
        ),
        "collision_records": collisions,
    }


def _case(output_dir: Path, logical_seed: int) -> dict[str, Any]:
    config = get_random_intersection_config(SCENARIO)
    if (
        config.get("configuration_revision") != R2_REVISION
        or config.get("previous_configuration_revision") != R1_REVISION
        or config.get("collision_action") != "remove"
        or config.get("terminal_outcome_protocol") != "exclusive_terminal_v2"
    ):
        raise RuntimeError(f"Unexpected scenario revision/protocol: {config}")

    d1._apply_patch(
        depart_scale=1.0,
        scenario=SCENARIO,
        eval_traffic_split="validation",
    )
    route_archive = output_dir / "episode_routes"
    route_archive.mkdir(parents=True, exist_ok=False)
    overlay_root = output_dir / "overlays"
    env = base.make_env_factory("base", overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    raw = env.unwrapped
    # Preserve the exact seeded route XML as evidence until the report is read.
    raw._random_episode_traffic_directory = SimpleNamespace(name=str(route_archive))
    captured_commands: list[list[str]] = []
    original_command = raw._sumo_command

    def capture_command(seed: int) -> list[str]:
        command = original_command(seed)
        captured_commands.append(list(command))
        return command

    raw._sumo_command = capture_command
    try:
        _observation, reset_info = env.reset(seed=logical_seed)
        connection = raw._connection
        if connection is None:
            raise RuntimeError("SUMO connection was not available after reset")
        if len(captured_commands) != 1:
            raise RuntimeError(f"Expected one SUMO startup, got {len(captured_commands)}")
        command = captured_commands[0]
        if _argument(command, "--collision.action") != "remove":
            raise RuntimeError("SUMO command did not set --collision.action remove")

        ego_id = str(raw.specification.ego_id)
        if ego_id not in connection.vehicle.getIDList():
            raise RuntimeError(f"Ego {ego_id!r} is not active after reset")
        ego_route = list(connection.vehicle.getRoute(ego_id))
        if not ego_route or ego_route[-1] != "-E0":
            raise RuntimeError(f"Unexpected ego route; expected final edge -E0: {ego_route}")

        # Choose one active background vehicle already on the ego's final edge,
        # in the configured arrival lane, at least 10 m from the route exit.
        candidates = []
        for vehicle_id in connection.vehicle.getIDList():
            if vehicle_id == ego_id:
                continue
            route = list(connection.vehicle.getRoute(vehicle_id))
            if not route or route[-1] != "-E0":
                continue
            if str(connection.vehicle.getRoadID(vehicle_id)) != "-E0":
                continue
            lane_id = str(connection.vehicle.getLaneID(vehicle_id))
            if not lane_id.endswith("_1"):
                continue
            lane_position = float(connection.vehicle.getLanePosition(vehicle_id))
            lane_length = float(connection.lane.getLength(lane_id))
            remaining = lane_length - lane_position
            if remaining < 10.0 or lane_position <= 1.0:
                continue
            candidates.append((abs(remaining - lane_length / 2.0), vehicle_id, lane_id, lane_position))
        if not candidates:
            raise RuntimeError(
                "No active background vehicle on final edge -E0/lane 1 with at least 10 m to exit"
            )
        _midpoint_distance, other_id, lane_id, collision_position = min(candidates)
        other_before = _vehicle_state(connection, other_id)
        lane_length = float(connection.lane.getLength(lane_id))
        raw_events_before = _collision_payload(connection)
        if ego_id in raw_events_before["arrived_ids"] or ego_id in raw_events_before["colliding_vehicle_ids"]:
            raise RuntimeError("Ego already has a terminal/collision event before the intervention")

        # This is an explicit test intervention, not policy behavior.  Both
        # vehicles are put into the same lane position and held at zero speed;
        # the collision itself must still be emitted by a normal SUMO tick.
        connection.vehicle.moveTo(ego_id, lane_id, collision_position)
        for vehicle_id in (ego_id, other_id):
            connection.vehicle.setSpeedMode(vehicle_id, 0)
            connection.vehicle.setSpeed(vehicle_id, 0.0)
        connection.vehicle.setLaneChangeMode(ego_id, 0)
        ego_after_setup = _vehicle_state(connection, ego_id)
        other_after_setup = _vehicle_state(connection, other_id)
        if (
            ego_after_setup["lane_id"] != other_after_setup["lane_id"]
            or not math.isclose(
                ego_after_setup["lane_position_m"],
                other_after_setup["lane_position_m"],
                rel_tol=0.0,
                abs_tol=1e-3,
            )
        ):
            raise RuntimeError(
                f"TraCI intervention did not establish same lane/position: "
                f"ego={ego_after_setup}, background={other_after_setup}"
            )

        # Action [-1, 0] requests zero speed / no lane change under the normal
        # action mapper.  This call performs the real simulationStep and event
        # resolver; the test never writes collision or terminal flags itself.
        _obs, reward, terminated, truncated, info = env.step(
            np.asarray([-1.0, 0.0], dtype=np.float32)
        )
        native_events_after = _collision_payload(connection)
        native_ego_collision = (
            ego_id in native_events_after["colliding_vehicle_ids"]
            or any(
                ego_id in (row.get("collider"), row.get("victim"))
                for row in native_events_after["collision_records"]
            )
        )
        result = {
            "scenario": SCENARIO,
            "scenario_revision": config["configuration_revision"],
            "case_kind": "forced_collision_test_only",
            "excluded_from_training_and_policy_evaluation": True,
            "logical_seed": int(logical_seed),
            "reset_info": reset_info,
            "sumo_command_collision_action": _argument(command, "--collision.action"),
            "step_length_seconds": float(_argument(command, "--step-length")),
            "simulation_time_before_intervention_seconds": float(
                connection.simulation.getTime()
            ),
            "ego_route": ego_route,
            "selected_background_before_intervention": other_before,
            "target_lane_id": lane_id,
            "target_lane_length_m": lane_length,
            "target_lane_position_m": collision_position,
            "ego_after_intervention_before_tick": ego_after_setup,
            "background_after_intervention_before_tick": other_after_setup,
            "raw_events_before_intervention": raw_events_before,
            "raw_events_after_env_step": native_events_after,
            "resolved_info": {
                key: info.get(key)
                for key in (
                    "raw_sumo_collision",
                    "raw_sumo_arrived",
                    "geometric_collision",
                    "collision",
                    "is_success",
                    "off_route",
                    "max_time",
                    "raw_max_time",
                    "terminal_outcome_protocol",
                    "raw_steps_executed",
                    "undiscounted_reward",
                    "reward_collision",
                    "reward_success",
                    "reward_off_route",
                    "reward_timeout",
                    "reward_step_cost",
                    "reward_progress",
                )
            },
            "returned_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "native_ego_collision_event": bool(native_ego_collision),
            "collision_terminal_pass": bool(
                native_ego_collision
                and info.get("raw_sumo_collision")
                and info.get("collision")
                and not info.get("is_success")
                and terminated
                and not truncated
                and math.isclose(float(info.get("reward_collision", 0.0)), -10.0)
            ),
            "same_tick_arrival_collision_priority_verified": bool(
                native_ego_collision
                and ego_id in native_events_after["arrived_ids"]
                and info.get("raw_sumo_arrived")
                and info.get("collision")
                and not info.get("is_success")
            ),
        }
        if not result["collision_terminal_pass"]:
            result["status"] = "failed_native_collision_terminal_contract"
        elif not result["same_tick_arrival_collision_priority_verified"]:
            result["status"] = "passed_collision_terminal_arrival_race_not_observed"
        else:
            result["status"] = "passed_same_tick_collision_over_arrival"
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-seed", type=int, default=10000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "running",
        "purpose": "one forced SUMO collision-removal check; not training or policy evaluation",
        "output_dir": str(output_dir),
    }
    try:
        report.update(_case(output_dir, args.logical_seed))
        if report["status"] == "failed_native_collision_terminal_contract":
            return_code = 1
        else:
            return_code = 0
    except BaseException as exc:
        report.update({
            "status": "failed_before_or_during_collision_check",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        return_code = 1
    report_path = output_dir / "terminal_outcome_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "native_ego_collision_event": report.get("native_ego_collision_event"),
        "raw_arrived": report.get("resolved_info", {}).get("raw_sumo_arrived"),
        "resolved_collision": report.get("resolved_info", {}).get("collision"),
        "resolved_success": report.get("resolved_info", {}).get("is_success"),
        "returned_reward": report.get("returned_reward"),
        "terminated": report.get("terminated"),
        "truncated": report.get("truncated"),
    }, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
