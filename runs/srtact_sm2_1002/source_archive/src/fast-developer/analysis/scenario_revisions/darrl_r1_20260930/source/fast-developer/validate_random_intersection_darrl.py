"""Bounded DARRL-profile integration check through the SAC-MLP base factory.

This is an environment smoke check only: it does not construct or train a model.
Use a new --output-root for each invocation; existing roots are refused.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as base
import train_intersection_yield_v2_d1 as d1
from envs.sumo.random_intersection import (
    DARRL_STEP_PROBABILITIES,
    get_random_intersection_config,
)


EXPECTED = {
    "intersection_random_darrl_low_v1": 0.03,
    "intersection_random_darrl_medium_v1": 0.05,
    "intersection_random_darrl_high_v1": 0.07,
}


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _close_float(value: object, expected: float, *, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _vtypes_by_id(path: Path) -> dict[str, ET.Element]:
    root = ET.parse(path).getroot()
    return {node.attrib["id"]: node for node in root.findall("vType") if "id" in node.attrib}


def _validate_case(scenario: str, probability: float, output_root: Path, split: str, logical_seed: int) -> dict:
    config = get_random_intersection_config(scenario)
    if scenario not in DARRL_STEP_PROBABILITIES:
        raise AssertionError(f"Not a DARRL profile: {scenario}")
    if config.get("collision_action") != "remove":
        raise AssertionError(f"collision_action must be remove: {scenario}")
    if config.get("collision_detection_mode") != "sumo_events_or_ego_geometry":
        raise AssertionError(f"Unexpected collision detector: {scenario}")
    if config.get("ego_vtype_overrides") != {"minGap": "1", "jmIgnoreFoeProb": "0"}:
        raise AssertionError(f"Unexpected ego vType overrides: {scenario}")
    if config.get("background_vtype_overrides") != {"jmIgnoreFoeProb": "0"}:
        raise AssertionError(f"Unexpected background vType overrides: {scenario}")
    for flow in config["flows"]:
        if not _close_float(flow.get("probability_per_step"), probability):
            raise AssertionError(f"Wrong per-step p in {scenario}: {flow}")
        if flow.get("depart_lane") != "1" or flow.get("arrival_lane") != "1":
            raise AssertionError(f"Expected depart/arrival lane 1 in {scenario}: {flow}")

    # D1's SAC+MLP parent contract is the existing `base` adapter.  The D1
    # patch only selects the scenario/split; no training or model is created.
    d1._apply_patch(depart_scale=1.0, scenario=scenario, eval_traffic_split=split)
    case_root = output_root / scenario
    overlay_root = case_root / "overlays"
    route_archive = case_root / "episode_routes"
    route_archive.mkdir(parents=True, exist_ok=False)
    env = base.make_env_factory("base", overlay_root)(
        base._environment_namespace(), evaluation=(split != "train")
    )
    raw = env.unwrapped

    # Keep the generated route XML as inspectable evidence.  A persistent
    # directory avoids TemporaryDirectory cleanup racing SUMO on Windows.
    raw._random_episode_traffic_directory = SimpleNamespace(name=str(route_archive))
    captured_commands: list[list[str]] = []
    original_command = raw._sumo_command

    def capture_command(seed: int) -> list[str]:
        command = original_command(seed)
        captured_commands.append(list(command))
        return command

    raw._sumo_command = capture_command
    try:
        observation, reset_info = env.reset(seed=logical_seed)
        del observation
        if len(captured_commands) != 1:
            raise AssertionError(f"Expected one SUMO startup command, got {len(captured_commands)}")
        command = captured_commands[0]
        if _argument(command, "--collision.action") != "remove":
            raise AssertionError(f"Actual SUMO command did not use collision.action=remove: {command}")
        if _argument(command, "--step-length") != "0.1":
            raise AssertionError("Expected SUMO step length 0.1 seconds")
        overlay_manifest = getattr(raw, "_selected_high_density_manifest", {}) or {}
        if not _close_float(overlay_manifest.get("vehicle_scale"), 1.0):
            raise AssertionError(f"The SAC-MLP base factory changed vehicle scale: {overlay_manifest}")
        for key in ("additional_explicit_vehicles", "additional_vehicle_flows", "additional_explicit_persons", "additional_person_flows"):
            if int(overlay_manifest.get(key, 0)) != 0:
                raise AssertionError(f"The base overlay added traffic ({key}): {overlay_manifest}")

        checkpoints = reset_info.get("warmup_traffic_checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 3:
            raise AssertionError(f"Expected warmup snapshots at 30/40/50s, got {checkpoints!r}")
        observed_times = [float(row["target_seconds"]) for row in checkpoints]
        if observed_times != [30.0, 40.0, 50.0]:
            raise AssertionError(f"Unexpected warmup snapshot targets: {observed_times}")
        for row in checkpoints:
            if not math.isclose(float(row["simulation_time_seconds"]), float(row["target_seconds"]), abs_tol=0.11):
                raise AssertionError(f"Warmup snapshot is outside its target tick: {row}")
            if row.get("request_balance_due_minus_inserted_minus_pending") != 0:
                raise AssertionError(f"Requested/inserted/pending demand does not balance: {row}")
            for lane_name, lane_counts in row["inlet_lanes"].items():
                if not lane_name.endswith("_1"):
                    raise AssertionError(f"DARRL inlet is not lane 1: {lane_name}")
                if lane_counts["requested_due"] != lane_counts["actually_inserted_cumulative"] + lane_counts["pending_insertion"]:
                    raise AssertionError(f"Lane-level demand does not balance: {lane_name}: {lane_counts}")

        traffic_path = Path(raw._selected_traffic_path)
        traffic_root = ET.parse(traffic_path).getroot()
        scheduled_vehicles = traffic_root.findall("vehicle")
        if not scheduled_vehicles:
            raise AssertionError(f"No seeded per-step arrivals were materialized: {traffic_path}")
        for vehicle in scheduled_vehicles:
            if vehicle.get("departLane") != "1" or vehicle.get("arrivalLane") != "1":
                raise AssertionError(f"Generated background vehicle is not lane 1: {vehicle.attrib}")
        background_types = _vtypes_by_id(traffic_path)
        if not background_types or any(vtype.get("jmIgnoreFoeProb") != "0" for vtype in background_types.values()):
            raise AssertionError("Generated background vTypes must all set jmIgnoreFoeProb=0")

        ego_route_path = Path(raw.specification.ego_route_path)
        ego_root = ET.parse(ego_route_path).getroot()
        ego_id = raw.specification.ego_id
        ego_node = next((item for item in ego_root.findall("vehicle") if item.get("id") == ego_id), None)
        if ego_node is None:
            raise AssertionError(f"Ego {ego_id!r} is absent from {ego_route_path}")
        ego_type_id = str(raw._connection.vehicle.getTypeID(ego_id))
        source_ego_type_id = str(ego_node.get("type"))
        ego_types = _vtypes_by_id(ego_route_path)
        ego_type = ego_types[source_ego_type_id]
        if not _close_float(ego_type.get("minGap"), 1.0) or ego_type.get("jmIgnoreFoeProb") != "0":
            raise AssertionError(f"Ego type settings are not minGap=1/jmIgnoreFoeProb=0: {ego_type.attrib}")
        runtime_min_gap = float(raw._connection.vehicle.getMinGap(ego_id))
        if not math.isclose(runtime_min_gap, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(f"Runtime ego minGap is {runtime_min_gap}, expected 1.0")
        try:
            runtime_ego_jm_ignore = raw._connection.vehicle.getParameter(ego_id, "jmIgnoreFoeProb")
        except Exception as exc:
            runtime_ego_jm_ignore = {"unavailable": f"{type(exc).__name__}: {exc}"}

        decision_steps = []
        info = reset_info
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for index in range(5):
            _, reward, terminated, truncated, info = env.step(action)
            decision_steps.append({
                "decision": index + 1,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "ego_lane": (
                    str(raw._connection.vehicle.getLaneID(ego_id))
                    if raw._connection is not None and ego_id in raw._connection.vehicle.getIDList()
                    else None
                ),
            })
            if terminated or truncated:
                break

        # Preserve the exact generated vehicle table without changing it.
        archived_route = route_archive / f"traffic_seed_{reset_info['traffic_sumo_seed']}.rou.xml"
        if archived_route.exists():
            raise FileExistsError(f"Refusing to replace archived route: {archived_route}")
        shutil.copy2(traffic_path, archived_route)
        return {
            "scenario": scenario,
            "profile_probability_per_0_1s_step": probability,
            "total_requested_vehicles_per_hour": config["total_vehicles_per_hour"],
            "route_profiles": [
                {key: flow.get(key) for key in ("id", "edges", "probability_per_step", "depart_lane", "arrival_lane")}
                for flow in config["flows"]
            ],
            "split": reset_info.get("traffic_split"),
            "logical_seed": reset_info.get("traffic_logical_episode_seed"),
            "sumo_seed": reset_info.get("traffic_sumo_seed"),
            "collision_action_actual_command": _argument(command, "--collision.action"),
            "configured_collision_detection_mode": config["collision_detection_mode"],
            "ego": {
                "id": ego_id,
                "runtime_type_id": ego_type_id,
                "route_type_id": source_ego_type_id,
                "runtime_jmIgnoreFoeProb_traci_readback": runtime_ego_jm_ignore,
                "runtime_minGap": runtime_min_gap,
                "route_vType_minGap": ego_type.get("minGap"),
                "route_vType_jmIgnoreFoeProb": ego_type.get("jmIgnoreFoeProb"),
                "lane_after_reset": str(raw._connection.vehicle.getLaneID(ego_id))
                if raw._connection is not None and ego_id in raw._connection.vehicle.getIDList()
                else None,
            },
            "background_vtypes": {
                "count": len(background_types),
                "all_jmIgnoreFoeProb_zero": all(v.get("jmIgnoreFoeProb") == "0" for v in background_types.values()),
                "generated_vehicle_count": len(scheduled_vehicles),
                "all_generated_vehicles_depart_arrive_lane_1": True,
                "traffic_xml_sha256": reset_info["random_traffic"]["episode_schedule"]["sha256"],
                "archived_route_xml": str(archived_route.resolve()),
            },
            "warmup_traffic_checkpoints": checkpoints,
            "demand_balance_by_checkpoint": [
                {
                    "target_seconds": row["target_seconds"],
                    "requested_due": row["requested_background_due"],
                    "inserted": row["actually_inserted_background_cumulative"],
                    "pending": row["pending_background_insertions"],
                    "balance": row["request_balance_due_minus_inserted_minus_pending"],
                    "in_network_background": row["in_network_background_vehicles"],
                    "halting_background": row["halting_background_vehicles"],
                }
                for row in checkpoints
            ],
            "post_reset_decision_steps": decision_steps,
            "post_step_random_traffic": info.get("random_traffic"),
            "high_density_adapter": {
                "vehicle_scale": reset_info.get("high_density_vehicle_scale"),
                "pedestrian_scale": reset_info.get("high_density_pedestrian_scale"),
                "additional_explicit_vehicles": reset_info.get("high_density_additional_explicit_vehicles"),
                "overlay_sha256": reset_info.get("high_density_overlay_sha256"),
                "overlay_manifest": overlay_manifest,
            },
            "sumo_route_files_command": _argument(command, "--route-files"),
            "overlay_root": str(overlay_root.resolve()),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--logical-seed", type=int, default=10000)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    report = {
        "status": "running",
        "kind": "real D1 base-factory SUMO environment smoke; no model construction/training",
        "split": args.split,
        "logical_seed": args.logical_seed,
        "cases": [],
    }
    try:
        for scenario, probability in EXPECTED.items():
            report["cases"].append(
                _validate_case(scenario, probability, output_root, args.split, args.logical_seed)
            )
        report["status"] = "passed"
        return_code = 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        return_code = 1
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cases": len(report["cases"]),
        "error": report.get("error"),
        "report": str(report_path),
    }, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
