"""Offline reconstruction of RouteAct geometric-collision overlaps.

This reads existing evaluation telemetry only. It does not run SUMO or modify
any experiment output. The SAT mirrors sumo_env._oriented_boxes_overlap and
uses the already center-referenced, physical-heading actor snapshots.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RAW_STEPS = Path(
    r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortct_1002"
    r"\sac_mlp_d1_st_rt_routeact_v1__intersection_sorted_depart4p0"
    r"\diagnostics\eval\raw_steps.jsonl.gz"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "routeact_collision_partner_reconstruction_20261003.json"
)
SUMO_VEHICLE_DOC = (
    "https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html"
)


def _valid_actor(actor: dict[str, Any]) -> bool:
    try:
        values = [
            float(actor["position"][0]),
            float(actor["position"][1]),
            float(actor["heading"]),
            float(actor["length"]),
            float(actor["width"]),
        ]
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return all(math.isfinite(value) for value in values)


def _rect_axes(actor: dict[str, Any]):
    heading = float(actor["heading"])
    forward = (math.cos(heading), math.sin(heading))
    side = (-forward[1], forward[0])
    half = (float(actor["length"]) / 2.0, float(actor["width"]) / 2.0)
    return forward, side, half


def _boxes_overlap(
    ego: dict[str, Any], other: dict[str, Any], leeway_m: float = 0.05
) -> bool:
    """The source SAT in physical-heading coordinates, with identical margin."""
    ego_forward, ego_side, ego_half = _rect_axes(ego)
    other_forward, other_side, other_half = _rect_axes(other)
    delta = (
        float(other["position"][0]) - float(ego["position"][0]),
        float(other["position"][1]) - float(ego["position"][1]),
    )

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    for axis in (ego_forward, ego_side, other_forward, other_side):
        ego_radius = ego_half[0] * abs(dot(ego_forward, axis)) + ego_half[1] * abs(
            dot(ego_side, axis)
        )
        other_radius = other_half[0] * abs(dot(other_forward, axis)) + other_half[
            1
        ] * abs(dot(other_side, axis))
        if abs(dot(delta, axis)) > ego_radius + other_radius + max(0.0, leeway_m):
            return False
    return True


def _angle_difference_rad(a: float, b: float) -> float:
    difference = abs((float(a) - float(b)) % (2.0 * math.pi))
    return min(difference, 2.0 * math.pi - difference)


def reconstruct(raw_steps: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exclusions = Counter()
    with gzip.open(raw_steps, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            events = record.get("events") or {}
            if not events.get("geometric_collision"):
                continue
            exclusions["geometric_collision_rows"] += 1
            ego = record.get("ego")
            if not isinstance(ego, dict) or not _valid_actor(ego):
                exclusions["missing_ego_geometry"] += 1
                continue
            if record.get("ego_state_source") == "current":
                exclusions["ego_current_rows"] += 1
            else:
                exclusions["ego_not_current"] += 1
            ego_time = record.get("ego_state_time")
            if ego_time is None:
                exclusions["missing_ego_state_time"] += 1
            raw_collision_ids = record.get("collision_ids") or []
            raw_collision_events = record.get("collision_events") or []
            if events.get("raw_sumo_collision"):
                exclusions["sumo_reported_collision_rows"] += 1
            else:
                exclusions["geometry_only_no_sumo_collision_rows"] += 1
            if not raw_collision_ids:
                exclusions["rows_without_sumo_collision_ids"] += 1
            if not raw_collision_events:
                exclusions["rows_without_sumo_collision_events"] += 1
            omitted = record.get("neighbors_omitted")
            if omitted is None:
                exclusions["rows_missing_neighbors_omitted"] += 1
            else:
                exclusions["neighbors_omitted_total"] += int(omitted)
                if int(omitted) == 0:
                    exclusions["zero_neighbors_omitted_rows"] += 1
            in_radius = record.get("neighbors_in_radius")
            if in_radius is not None:
                exclusions["neighbors_in_radius_total"] += int(in_radius)

            candidates = []
            same_time_current_count = 0
            time_mismatch_count = 0
            valid_vehicle_count = 0
            for other in record.get("vehicles") or []:
                if not isinstance(other, dict) or not _valid_actor(other):
                    continue
                valid_vehicle_count += 1
                if (
                    other.get("state_source") == "current"
                    and ego_time is not None
                    and other.get("state_time") == ego_time
                ):
                    same_time_current_count += 1
                else:
                    time_mismatch_count += 1
                if _boxes_overlap(ego, other):
                    candidates.append(other)

            if time_mismatch_count:
                exclusions["rows_with_vehicle_time_mismatch"] += 1
            elif valid_vehicle_count > 0:
                exclusions["all_valid_vehicles_current_at_ego_time_rows"] += 1
            if not candidates:
                exclusions["no_snapshot_overlap_candidate"] += 1
                continue
            if len(candidates) == 1:
                exclusions["exactly_one_snapshot_overlap_candidate"] += 1
            else:
                exclusions["multiple_snapshot_overlap_candidates"] += 1

            for other in candidates:
                heading_delta = _angle_difference_rad(
                    ego["heading"], other["heading"]
                )
                dx = float(other["position"][0]) - float(ego["position"][0])
                dy = float(other["position"][1]) - float(ego["position"][1])
                ego_velocity = ego.get("velocity")
                other_velocity = other.get("velocity")
                relative_velocity = None
                if (
                    isinstance(ego_velocity, (list, tuple))
                    and len(ego_velocity) >= 2
                    and isinstance(other_velocity, (list, tuple))
                    and len(other_velocity) >= 2
                ):
                    relative_velocity = [
                        float(other_velocity[0]) - float(ego_velocity[0]),
                        float(other_velocity[1]) - float(ego_velocity[1]),
                    ]
                actor_key = other.get("key")
                observed_keys = set(record.get("observed_neighbor_ids") or [])
                if actor_key is None:
                    visible_to_policy = other.get("observed_by_policy")
                else:
                    visible_to_policy = (
                        actor_key in observed_keys
                        or bool(other.get("observed_by_policy"))
                    )
                rows.append(
                    {
                        "episode": record.get("episode"),
                        "decision": record.get("decision"),
                        "raw_step": record.get("raw_step"),
                        "sim_time_s": record.get("sim_time"),
                        "ego_state_source": record.get("ego_state_source"),
                        "ego_state_time_s": ego_time,
                        "other_state_source": other.get("state_source"),
                        "other_state_time_s": other.get("state_time"),
                        "same_timestamp_current_vehicle_count": same_time_current_count,
                        "valid_vehicle_count": valid_vehicle_count,
                        "vehicle_time_mismatch_count": time_mismatch_count,
                        "neighbors_in_radius": in_radius,
                        "neighbors_omitted": omitted,
                        "observed_neighbor_raw_step": record.get("observed_neighbor_raw_step"),
                        "observed_neighbor_age_at_collision_raw_steps": (
                            int(record["raw_step"]) - int(record["observed_neighbor_raw_step"])
                            if record.get("raw_step") is not None
                            and record.get("observed_neighbor_raw_step") is not None
                            else None
                        ),
                        "ego": {
                            key: ego.get(key)
                            for key in (
                                "id", "key", "position", "velocity", "heading", "speed", "length",
                                "width", "road_id", "lane_id", "route_index",
                            )
                        },
                        "overlap_actor": {
                            key: other.get(key)
                            for key in (
                                "id", "key", "kind", "position", "velocity", "heading", "speed", "length",
                                "width", "road_id", "lane_id", "route_index",
                            )
                        },
                        "overlap_actor_observed_by_policy": visible_to_policy,
                        "observed_neighbor_ids_at_last_observation": record.get("observed_neighbor_ids"),
                        "center_distance_m": math.hypot(dx, dy),
                        "relative_velocity_mps": relative_velocity,
                        "relative_speed_mps_partner_minus_ego": (
                            float(other["speed"]) - float(ego["speed"])
                            if other.get("speed") is not None and ego.get("speed") is not None
                            else None
                        ),
                        "relative_heading_difference_rad": heading_delta,
                        "relative_heading_difference_deg": math.degrees(heading_delta),
                        "same_road": ego.get("road_id") == other.get("road_id"),
                        "same_lane": ego.get("lane_id") == other.get("lane_id"),
                        "snapshot_sat_overlap": True,
                        "sat_leeway_m": 0.05,
                        "sumo_reported_collision": bool(events.get("raw_sumo_collision")),
                        "sumo_collision_ids": raw_collision_ids,
                        "sumo_collision_events": raw_collision_events,
                    }
                )

    unique_actor_ids = sorted(
        {str(row["overlap_actor"].get("id")) for row in rows}
    )
    return {
        "title": "Offline RouteAct geometric-collision partner reconstruction",
        "as_of": "2026-10-03",
        "source_raw_steps": str(raw_steps),
        "method": {
            "input": "same raw-step behavior snapshot where events.geometric_collision=true",
            "geometry": "SAT over snapshot center x/y, physical heading radians, length, width",
            "sat": "same four axes and 0.05m leeway as envs/sumo/sumo_env.py::_oriented_boxes_overlap",
            "no_simulation": True,
            "not_sumo_reported_partner": True,
        },
        "sumo_coordinate_reference": {
            "url": SUMO_VEHICLE_DOC,
            "documented_vehicle_position": "center of front bumper",
            "documented_vehicle_angle": "degrees; TraCI getAngle",
            "source_conversion": "front bumper to center via SUMO angle; physical heading is radians(90-angle)",
        },
        "counts": dict(exclusions),
        "unique_reconstructed_actor_ids": len(unique_actor_ids),
        "reconstructed_rows": rows,
        "interpretation_limits": [
            "The original telemetry stores geometric_collision as a boolean and does not store an explicit partner ID.",
            "A unique overlap is reconstructed under the same geometric rule from the saved same-tick state; it is not an independent SUMO collision event or a liability/right-of-way judgment.",
            "No SUMO raw collision participants/events were reported for these geometric-only rows.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-steps", type=Path, default=DEFAULT_RAW_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = reconstruct(args.raw_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "unique_reconstructed_actor_ids": result["unique_reconstructed_actor_ids"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
