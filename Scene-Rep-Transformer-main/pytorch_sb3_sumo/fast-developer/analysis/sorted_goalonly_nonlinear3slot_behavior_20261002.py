"""Summarize saved sorted/depart4 behavior traces; performs no environment work."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
OUT = Path(__file__).with_suffix(".json")
PROJECT = ROOT / "Scene-Rep-Transformer-main" / "pytorch_sb3_sumo"

METHODS = {
    "goalonly": Path("sortg3_1002/sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0"),
    "nonlinear_3slot": Path("sortg3_1002/sac_mlp_d1_st_rt_3slot_nonlinear_v1__intersection_sorted_depart4p0"),
    "routeaware": Path("sort2_1001/sac_mlp_d1_st_rt_topo_routeaware_v1__intersection_sorted_depart4p0"),
    "linear_3slot": Path("sort2_1001/sac_mlp_d1_st_rt_3slot__intersection_sorted_depart4p0"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def finite_values(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def stats(values):
    values = finite_values(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def outcome_of(episode):
    outcome = episode.get("outcome")
    if outcome in {"success", "collision", "timeout", "off_route"}:
        return outcome
    info = episode.get("terminal_info") or {}
    for name, key in (("collision", "collision"), ("off_route", "off_route"), ("success", "is_success"), ("timeout", "max_time")):
        if info.get(key):
            return name
    return "unknown"


def raw_speed_sample(row):
    speed = row.get("speed_control") or {}
    target = speed.get("requested_target_speed_mps")
    actual = speed.get("actual_speed_mps")
    if actual is None:
        ego = row.get("ego") or {}
        actual = ego.get("speed")
    return target, actual


def route_label(row):
    value = row.get("current_lane_reachability_label")
    if value is None:
        value = (row.get("lane_control") or {}).get("current_lane_reachability_label")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def summarize_speed_rows(rows):
    groups = defaultdict(lambda: {"target": [], "actual": [], "stopped": 0, "rows": 0})
    for row in rows:
        target, actual = raw_speed_sample(row)
        label = route_label(row)
        key = "eligible" if label == 1 else "ineligible" if label == 0 else "unknown"
        for name in ("all", key):
            entry = groups[name]
            entry["rows"] += 1
            if target is not None:
                entry["target"].append(target)
            if actual is not None:
                entry["actual"].append(actual)
                if float(actual) < 0.1:
                    entry["stopped"] += 1
        if label == 0 and actual is not None and float(actual) < 0.1:
            groups["ineligible_and_stopped"]["rows"] += 1
            groups["ineligible_and_stopped"]["target"].append(target)
            groups["ineligible_and_stopped"]["actual"].append(actual)
            groups["ineligible_and_stopped"]["stopped"] += 1
    output = {}
    for name, data in groups.items():
        output[name] = {
            "raw_tick_count": data["rows"],
            "target_speed_mps": stats(data["target"]),
            "actual_speed_mps": stats(data["actual"]),
            "stopped_ticks_below_0p1mps": data["stopped"],
            "stopped_fraction_of_actual_samples": (
                data["stopped"] / len(data["actual"]) if data["actual"] else None
            ),
        }
    return output


def summarize_decision_entry(rows):
    rows = sorted(rows, key=lambda row: (row.get("episode", -1), row.get("decision", -1)))
    per_episode = defaultdict(list)
    for row in rows:
        per_episode[row.get("episode")].append(row)
    entries = []
    for episode, decisions in sorted(per_episode.items()):
        first_bad = None
        bad_target_requests_while_eligible = []
        for row in decisions:
            control = row.get("control") or {}
            transition = control.get("actual_lane_transition_since_previous_decision") or {}
            label = control.get("current_lane_reachability_label", -1)
            if first_bad is None and label == 0:
                first_bad = {
                    "decision": row.get("decision"),
                    "lane": control.get("current_lane_id"),
                    "road": control.get("current_road_id"),
                    "next_edge": control.get("planned_next_edge"),
                    "entry_transition": {
                        key: transition.get(key)
                        for key in (
                            "from_lane_id",
                            "to_lane_id",
                            "previous_lane_command_requested",
                            "previous_request_status",
                            "previous_lane_change_applied",
                            "previous_expected_target_lane_id",
                            "previous_expected_target_reached",
                            "changed",
                        )
                    },
                    "current_action": {
                        key: control.get(key)
                        for key in (
                            "lane_command_requested",
                            "lane_control_request_status",
                            "lane_change_applied",
                            "expected_target_lane_id",
                            "expected_target_lane_can_reach_next_edge",
                        )
                    },
                }
            if (
                label == 1
                and control.get("lane_command_requested") not in (None, 0)
                and control.get("expected_target_lane_can_reach_next_edge") is False
            ):
                bad_target_requests_while_eligible.append(
                    {
                        "decision": row.get("decision"),
                        "from_lane": control.get("current_lane_id"),
                        "requested_command": control.get("lane_command_requested"),
                        "request_status": control.get("lane_control_request_status"),
                        "lane_change_applied": control.get("lane_change_applied"),
                        "target_lane": control.get("expected_target_lane_id"),
                    }
                )
        if first_bad is not None or bad_target_requests_while_eligible:
            entries.append(
                {
                    "episode": episode,
                    "first_ineligible_state": first_bad,
                    "ineligible_target_requests_while_current_lane_eligible": bad_target_requests_while_eligible,
                }
            )
    return entries


def terminal_position(episode, raw_rows):
    info = episode.get("terminal_info") or {}
    terminal = max(raw_rows, key=lambda row: row.get("raw_step", -1), default={})
    ego = terminal.get("ego") or {}
    return {
        "road_id": info.get("current_road_id") or ego.get("road_id"),
        "lane_id": info.get("current_lane_id") or ego.get("lane_id"),
        "sim_time": terminal.get("sim_time"),
        "raw_step": terminal.get("raw_step"),
    }


def reconstruct_terminal_collision_partners(episode):
    """Replay the saved terminal OBB rule; snapshot positions are already centers."""
    project = str(PROJECT)
    if project not in sys.path:
        sys.path.insert(0, project)
    from envs.sumo.sumo_env import _oriented_boxes_overlap

    snapshot = episode.get("last_snapshot") or {}
    ego = snapshot.get("ego") or {}
    required = ("position", "heading", "length", "width")
    if any(ego.get(key) is None for key in required):
        return {"status": "missing_ego_geometry", "candidate_count": 0, "partners": []}

    def sumo_angle(actor):
        # Telemetry heading is math-angle radians; SUMO is degrees clockwise from north.
        return 90.0 - math.degrees(float(actor["heading"]))

    partners = []
    for actor in snapshot.get("vehicles", []):
        if any(actor.get(key) is None for key in required):
            continue
        overlap = _oriented_boxes_overlap(
            ego["position"], sumo_angle(ego), ego["length"], ego["width"],
            actor["position"], sumo_angle(actor), actor["length"], actor["width"],
            leeway=0.05,
        )
        if not overlap:
            continue
        relative_position = [
            float(actor["position"][index]) - float(ego["position"][index])
            for index in range(2)
        ]
        relative_velocity = [
            float(actor["velocity"][index]) - float(ego["velocity"][index])
            for index in range(2)
        ]
        distance = math.hypot(*relative_position)
        closing_speed = (
            -sum(relative_position[i] * relative_velocity[i] for i in range(2)) / distance
            if distance > 1e-9
            else None
        )
        heading_delta = abs(
            (float(actor["heading"]) - float(ego["heading"]) + math.pi)
            % (2.0 * math.pi) - math.pi
        )
        acute_heading_delta = min(heading_delta, math.pi - heading_delta)
        partners.append(
            {
                "id": actor.get("id"),
                "background_flow_id_prefix": (
                    str(actor.get("id")).split("-flow-", 1)[0]
                    if actor.get("id") and "-flow-" in str(actor.get("id"))
                    else None
                ),
                "road_id": actor.get("road_id"),
                "lane_id": actor.get("lane_id"),
                "route_id": actor.get("route_id"),
                "route_edges": actor.get("route_edges"),
                "position_center_m": actor.get("position"),
                "velocity_mps": actor.get("velocity"),
                "speed_mps": actor.get("speed"),
                "heading_rad_math": actor.get("heading"),
                "ego_partner_relative_position_m": relative_position,
                "ego_partner_relative_velocity_mps": relative_velocity,
                "radial_closing_speed_mps_positive_approach": closing_speed,
                "heading_difference_deg": math.degrees(heading_delta),
                "acute_heading_difference_deg": math.degrees(acute_heading_delta),
                "motion_relation": (
                    "crossing"
                    if 45.0 <= math.degrees(heading_delta) <= 135.0
                    else "same_direction"
                    if math.degrees(heading_delta) < 45.0
                    else "opposing_direction"
                ),
                "same_step_snapshot_state_time": actor.get("state_time") == ego.get("state_time"),
            }
        )
    status = (
        "unique_replay_partner"
        if len(partners) == 1
        else "ambiguous_replay_partners"
        if partners
        else "no_replay_partner"
    )
    return {
        "status": status,
        "candidate_count": len(partners),
        "collision_tick_raw_step": snapshot.get("raw_step"),
        "snapshot_position_semantics": "vehicle box centers from _state; do not apply front-bumper offset again",
        "method": "reapply sumo_env._oriented_boxes_overlap to saved ego/vehicle terminal geometry with leeway=0.05m",
        "partners": partners,
    }


def partner_pre_collision_window(episode, rows, partner_id, collision_step):
    pre = [row for row in rows if row.get("raw_step", -1) < collision_step][-30:]
    partner_rows = []
    for row in pre:
        actor = next(
            (vehicle for vehicle in row.get("vehicles", []) if vehicle.get("id") == partner_id),
            None,
        )
        if actor is None:
            continue
        ego = row.get("ego") or {}
        target, actual = raw_speed_sample(row)
        relative_position = [
            float(actor["position"][i]) - float(ego["position"][i]) for i in range(2)
        ]
        relative_velocity = [
            float(actor["velocity"][i]) - float(ego["velocity"][i]) for i in range(2)
        ]
        distance = math.hypot(*relative_position)
        closing = (
            -sum(relative_position[i] * relative_velocity[i] for i in range(2)) / distance
            if distance > 1e-9
            else None
        )
        partner_rows.append(
            {
                "raw_step": row.get("raw_step"),
                "ego_target_speed_mps": target,
                "ego_actual_speed_mps": actual,
                "ego_road_id": ego.get("road_id"),
                "ego_lane_id": ego.get("lane_id"),
                "partner_speed_mps": actor.get("speed"),
                "partner_road_id": actor.get("road_id"),
                "partner_lane_id": actor.get("lane_id"),
                "relative_position_m": relative_position,
                "relative_velocity_mps": relative_velocity,
                "radial_closing_speed_mps_positive_approach": closing,
                "min_cv_obb_ttc_s": row.get("min_cv_obb_ttc_s"),
            }
        )
    return {
        "window": "last 30 raw samples strictly before collision tick (0.1s each; at most 3.0s)",
        "partner_observed_samples": len(partner_rows),
        "ego_target_speed_mps": stats(x["ego_target_speed_mps"] for x in partner_rows),
        "ego_actual_speed_mps": stats(x["ego_actual_speed_mps"] for x in partner_rows),
        "ego_stopped_fraction_below_0p1mps": (
            sum(x["ego_actual_speed_mps"] is not None and x["ego_actual_speed_mps"] < 0.1 for x in partner_rows)
            / sum(x["ego_actual_speed_mps"] is not None for x in partner_rows)
            if any(x["ego_actual_speed_mps"] is not None for x in partner_rows)
            else None
        ),
        "partner_speed_mps": stats(x["partner_speed_mps"] for x in partner_rows),
        "partner_moving_fraction_above_0p1mps": (
            sum(x["partner_speed_mps"] is not None and x["partner_speed_mps"] >= 0.1 for x in partner_rows)
            / sum(x["partner_speed_mps"] is not None for x in partner_rows)
            if any(x["partner_speed_mps"] is not None for x in partner_rows)
            else None
        ),
        "positive_radial_closing_speed_fraction": (
            sum(x["radial_closing_speed_mps_positive_approach"] is not None and x["radial_closing_speed_mps_positive_approach"] > 0 for x in partner_rows)
            / sum(x["radial_closing_speed_mps_positive_approach"] is not None for x in partner_rows)
            if any(x["radial_closing_speed_mps_positive_approach"] is not None for x in partner_rows)
            else None
        ),
        "min_cv_obb_ttc_s": stats(x["min_cv_obb_ttc_s"] for x in partner_rows),
        "ticks": partner_rows,
    }


def collision_telemetry(episode, rows):
    info = episode.get("terminal_info") or {}
    collisions = [
        row for row in rows
        if (row.get("events") or {}).get("collision")
        or (row.get("events") or {}).get("geometric_collision")
        or (row.get("events") or {}).get("raw_sumo_collision")
    ]
    collision_row = max(collisions, key=lambda row: row.get("raw_step", -1), default=None)
    if collision_row is None:
        collision_row = max(rows, key=lambda row: row.get("raw_step", -1), default={})
    collision_step = collision_row.get("raw_step", info.get("raw_simulation_steps", -1))
    pre = [row for row in rows if row.get("raw_step", -1) < collision_step][-30:]
    pre_speeds = [raw_speed_sample(row) for row in pre]
    risk = [row for row in pre if row.get("risk_evaluable")]
    ttc = finite_values(row.get("min_cv_obb_ttc_s") for row in risk)
    collision_target, collision_actual = raw_speed_sample(collision_row)
    collision_lane_control = collision_row.get("lane_control") or {}
    ids = []
    objects = []
    for row in [*collisions, collision_row]:
        if not row:
            continue
        ids.extend(row.get("collision_ids") or [])
        objects.extend(row.get("collision_events") or [])
    terminal_lane_control = info.get("lane_control") or {}
    replay = reconstruct_terminal_collision_partners(episode)
    unique_partner = replay["partners"][0] if replay["candidate_count"] == 1 else None
    partner_window = (
        partner_pre_collision_window(
            episode, rows, unique_partner["id"], collision_step
        )
        if unique_partner is not None
        else None
    )
    return {
        "position": terminal_position(episode, rows),
        "event_evidence": {
            "terminal_outcome_collision": bool(info.get("collision")),
            "raw_sumo_collision": bool(info.get("raw_sumo_collision")),
            "geometric_collision": bool(info.get("geometric_collision")),
            "collision_row_raw_step": collision_row.get("raw_step"),
            "collision_ids": sorted(set(map(str, ids))),
            "collision_events_count": len(objects),
            "native_sumo_collision_object_identity_available": bool(ids or objects),
            "collision_object_identity_available": bool(ids or objects),
        },
        "terminal_geometry_partner_reconstruction": replay,
        "reconstructed_partner_last30_raw_ticks_before_collision": partner_window,
        "collision_tick": {
            "target_speed_mps": collision_target,
            "actual_speed_mps": collision_actual,
            "ego_road_id": (collision_row.get("ego") or {}).get("road_id"),
            "ego_lane_id": (collision_row.get("ego") or {}).get("lane_id"),
            "lane_command_requested": collision_lane_control.get("lane_command_requested"),
            "lane_control_request_status": collision_lane_control.get("lane_control_request_status"),
            "lane_control_request_reason": collision_lane_control.get("lane_control_request_reason"),
            "lane_change_applied": collision_lane_control.get("lane_change_applied"),
            "actual_lane_transition": collision_lane_control.get("actual_lane_transition_since_previous_decision"),
            "min_cv_obb_ttc_s": collision_row.get("min_cv_obb_ttc_s"),
            "risk_evaluable": collision_row.get("risk_evaluable"),
        },
        "terminal_lane_request": {
            key: terminal_lane_control.get(key, info.get(key))
            for key in (
                "lane_command_requested",
                "lane_control_request_status",
                "lane_control_request_reason",
                "lane_change_applied",
                "expected_target_lane_id",
                "actual_lane_transition_since_previous_decision",
            )
        },
        "pre_collision_last30_raw_ticks": {
            "window": "last 30 raw samples strictly before collision tick (0.1s each; at most 3.0s)",
            "raw_tick_count": len(pre),
            "target_speed_mps": stats(pair[0] for pair in pre_speeds),
            "actual_speed_mps": stats(pair[1] for pair in pre_speeds),
            "risk_evaluable_ticks": len(risk),
            "min_cv_obb_ttc_s": stats(ttc),
            "critical_unobserved_ticks": sum(bool(row.get("critical_unobserved")) for row in pre),
        },
        "collision_tick_risk": {
            "risk_evaluable": collision_row.get("risk_evaluable"),
            "min_cv_obb_ttc_s": collision_row.get("min_cv_obb_ttc_s"),
            "critical_unobserved": collision_row.get("critical_unobserved"),
        },
    }


def main():
    result = {
        "schema_version": "sorted_goalonly_nonlinear3slot_behavior_v1",
        "created_utc_date": "2026-10-02",
        "purpose": "Offline summaries of saved final-evaluation traces only; no training, SUMO, or extra evaluation.",
        "scenario_identity": {
            "scenario": "intersection_sorted",
            "depart_speed_mps": 4.0,
            "scope_note": "This is the sorted/depart4 experiment using the released intersection_sorted map; it is not the separate DARRL medium scenario. The underlying released map is shared.",
            "map_source": str(PROJECT / "envs/sumo/original_scenarios_v1/intersection_sorted/map.net.xml"),
            "junction_J1_type": "unregulated",
            "effective_sumo_launch": {
                "configured_by": "D1 env_adapter=base -> make_env_factory -> YieldObsIndependentV2EnvV1 -> IndependentV2FiveBySixEnvV1 -> HighDensityPaperSumoSceneEnvV1 -> PaperSumoSceneEnv._sumo_command",
                "net_file_argument": str(PROJECT / "envs/sumo/original_scenarios_v1/intersection_sorted/map.net.xml"),
                "collision_action": "none",
                "collision_check_junctions": True,
                "flag_owner": str(PROJECT / "envs/sumo/paper_env.py"),
                "flag_line_refs": "219-239",
                "overlay_override": "HighDensityPaperSumoSceneEnvV1 injects only an additional --route-files entry; it does not append/override collision flags.",
                "duplicate_collision_flag_occurrences": 1,
                "exact_argv_saved_in_run_artifacts": False,
                "effective_setting_evidence": "Static resolved constructor/override chain plus run arguments.json env_contract=base and terminal high-density metadata; the exact child argv is not persisted.",
                "unused_legacy_sumocfg_note": "The legacy scenarios/intersection_sorted/scenario.sumocfg has collision.action=warn, but released PaperSumoSceneEnv starts directly with --net-file/--route-files and explicit command-line collision options; that static sumocfg value is not the effective run setting.",
            },
        },
        "speed_window_definition": "target and actual speed are paired from the same raw-step speed_control record; timeout-ineligible and terminal-tail values are reported separately.",
        "stopped_threshold_mps": 0.1,
        "methods": {},
        "method_roles_code_audit": {
            "source_files": {
                "encoder": str(ROOT / "Scene-Rep-Transformer-main/pytorch_sb3_sumo/algos/sb3_torch/incremental_topo_encoder.py"),
                "d1_runner": str(ROOT / "Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/train_intersection_yield_v2_d1.py"),
                "environment_action": str(ROOT / "Scene-Rep-Transformer-main/pytorch_sb3_sumo/envs/sumo/sumo_env.py"),
            },
            "roles": [
                {
                    "module": "ST spatial interaction",
                    "code": "IncrementalTopoEncoder.forward applies vehicle_layers to graph_features at each history slice; this message-passes geometric/motion edge features across vehicle tokens.",
                    "driving_task": "Use nearby actor geometry and motion to form local interactions; it does not itself impose a route-legal lane command.",
                    "line_refs": "incremental_topo_encoder.py:730-742",
                },
                {
                    "module": "ST temporal aggregation",
                    "code": "The temporal_encoder consumes the post-spatial graph_sequence; a probe can replace it with the last valid frame.",
                    "driving_task": "Aggregate actor history to represent changing motion/interaction over time.",
                    "line_refs": "incremental_topo_encoder.py:744-749",
                },
                {
                    "module": "social readout",
                    "code": "social_attention pools actor_features using ego as query; social_ego_only masks all non-ego keys.",
                    "driving_task": "Compress relevant surrounding actors into a social slot/context.",
                    "line_refs": "incremental_topo_encoder.py:750-761",
                },
                {
                    "module": "RT route context/readout",
                    "code": "_encode_routes creates route tokens/context; default route path adds route_context to each actor history token before ST, while ego goal_attention reads map/route tokens into route_component.",
                    "driving_task": "Expose planned path information to the representation; route-conditioned features remain learned through the critic because these runs have representation_coef=0.",
                    "line_refs": "incremental_topo_encoder.py:552-600, 767-826; train_intersection_yield_v2_d1.py:590-633",
                    "limit": "The module name or input path does not establish a supervised semantic intent label; there is no separate representation loss here.",
                },
                {
                    "module": "goal-only topology and legal attention",
                    "code": "When actor-intent and relation branches are false, graph_input uses topology_norm(intent_tokens) without adding topology_context; the goal branch queries topology from route_base and can mask candidate tokens with route_reachability.",
                    "driving_task": "Give the goal/route readout topology-conditioned lane tokens and, when enabled, mask attention candidates that lack immediate route continuation.",
                    "line_refs": "incremental_topo_encoder.py:650-663, 767-820",
                    "limit": "This mask only filters representation attention; it does not constrain action[1], adapt_action, or SUMO changeLaneRelative. A lane label of 0 means no immediate continuation, not global impossibility of recovery.",
                },
                {
                    "module": "action execution",
                    "code": "adapt_action maps normalized lateral action using +/-1/3 thresholds to lane command {-1,0,1}; _apply_control uses action-contract lane offset and issues changeLaneRelative, with actual transition recorded separately.",
                    "driving_task": "Execute a relative lane request while SUMO may still constrain physical transition.",
                    "line_refs": "sumo_env.py:1250-1294, 1431-1435",
                },
            ],
            "configuration_source": "per-run arguments.json, not current defaults",
        },
        "interpretation_limits": [
            "Single training seed and a fixed 100-seed validation set are descriptive; they do not estimate across-training-seed variation.",
            "The lane reachability label describes the current sampled lane's immediate static continuation to the next route edge; it is not a global reachability or safety proof.",
            "Collision-tick min TTC may be zero by construction at overlap and is not independent evidence of a pre-collision cause; only earlier ticks are summarized as precursors.",
            "Native SUMO collision IDs/events may be empty for geometric-only outcomes; terminal snapshot OBB replay can recover candidate partners when it yields exactly one overlap, but this is not independent SUMO collision truth.",
            "The pre-collision last-30-raw-sample window is at most 3.0 seconds (0.1 seconds per raw tick), not 30 seconds.",
            "A route request being sent/applied is distinct from a confirmed lane transition; use actual_lane_transition_since_previous_decision for realized transitions.",
        ],
    }

    for method, relative in METHODS.items():
        run = RUNS / relative
        episode_path = run / "diagnostics/eval/episodes.jsonl"
        raw_path = run / "diagnostics/eval/raw_steps.jsonl.gz"
        decisions_path = run / "diagnostics/eval/decisions.jsonl.gz"
        args_path = run / "arguments.json"
        episodes = read_jsonl(episode_path)
        episode_map = {int(row["episode"]): row for row in episodes}
        outcome_map = {episode: outcome_of(row) for episode, row in episode_map.items()}
        counts = Counter(outcome_map.values())
        selected = {ep for ep, outcome in outcome_map.items() if outcome in {"timeout", "collision"}}
        raw_by_ep = defaultdict(list)
        for row in read_jsonl_gz(raw_path):
            episode_id = row.get("episode")
            if episode_id in selected:
                raw_by_ep[int(episode_id)].append(row)
        decisions_by_ep = defaultdict(list)
        for row in read_jsonl_gz(decisions_path):
            episode_id = row.get("episode")
            if episode_id in selected:
                decisions_by_ep[int(episode_id)].append(row)

        timeout_eps = [ep for ep, outcome in outcome_map.items() if outcome == "timeout"]
        collision_eps = [ep for ep, outcome in outcome_map.items() if outcome == "collision"]
        timeout_rows = [row for ep in timeout_eps for row in raw_by_ep.get(ep, [])]
        timeout_tail_rows = []
        for ep in timeout_eps:
            rows = sorted(raw_by_ep.get(ep, []), key=lambda row: row.get("raw_step", -1))
            timeout_tail_rows.extend(rows[-30:])

        decision_entries = summarize_decision_entry(
            [row for ep in timeout_eps for row in decisions_by_ep.get(ep, [])]
        )
        collision_details = [
            collision_telemetry(episode_map[ep], raw_by_ep.get(ep, []))
            for ep in collision_eps
        ]
        replay_unique_count = sum(
            detail["terminal_geometry_partner_reconstruction"]["status"] == "unique_replay_partner"
            for detail in collision_details
        )
        replay_validation = {
            "collision_episode_count": len(collision_eps),
            "terminal_geometric_collision_count": sum(
                detail["event_evidence"]["geometric_collision"]
                for detail in collision_details
            ),
            "unique_terminal_obb_replay_count": replay_unique_count,
            "all_geometric_collision_episodes_have_one_replayed_partner": (
                replay_unique_count == len(collision_eps)
                and all(
                    detail["event_evidence"]["geometric_collision"]
                    for detail in collision_details
                )
            ),
            "replay_is_independent_sumo_ground_truth": False,
        }
        if method in {"goalonly", "nonlinear_3slot"}:
            assert replay_validation["all_geometric_collision_episodes_have_one_replayed_partner"], (
                f"Saved terminal OBB replay incomplete for {method}: {replay_validation}"
            )
        terminal_lane_counts = Counter(
            (detail["position"].get("road_id"), detail["position"].get("lane_id"))
            for detail in collision_details
        )
        collision_request_counts = Counter(
            (
                detail["terminal_lane_request"].get("lane_control_request_status"),
                detail["terminal_lane_request"].get("lane_command_requested"),
            )
            for detail in collision_details
        )
        method_args = json.loads(args_path.read_text(encoding="utf-8"))
        relevant_args = {
            key: method_args.get(key)
            for key in (
                "method", "parent", "scenario", "raw_budget", "action_repeat", "seed",
                "use_route", "use_topology", "use_topology_actor_intent", "use_topology_relations",
                "use_topology_goal", "use_slots", "use_incremental_slots", "use_route_reachability",
                "use_parameter_matched_nonlinear_slots", "representation_coef", "slot_balance_coef",
                "active_readout_type", "active_readout_parameter_count", "slot_head_widths",
            )
        }
        result["methods"][method] = {
            "run_dir": str(run),
            "episode_file": str(episode_path),
            "raw_steps_file": str(raw_path),
            "decisions_file": str(decisions_path),
            "arguments_file": str(args_path),
            "arguments": relevant_args,
            "episodes": len(episodes),
            "outcome_counts": dict(counts),
            "timeout_episode_count_with_raw_trace": sum(bool(raw_by_ep.get(ep)) for ep in timeout_eps),
            "timeouts": {
                "raw_tick_speed_pairs": summarize_speed_rows(timeout_rows),
                "last_30_raw_ticks_per_episode_speed_pairs": summarize_speed_rows(timeout_tail_rows),
                "wrong_lane_entry_decisions": decision_entries,
                "wrong_lane_entry_episode_count": sum(
                    bool(entry["first_ineligible_state"]) for entry in decision_entries
                ),
                "episode_terminal_details": [
                    {
                        "episode": ep,
                        "seed": episode_map[ep].get("seed"),
                        "terminal_position": terminal_position(episode_map[ep], raw_by_ep.get(ep, [])),
                        "episode_diagnostics": {
                            key: episode_map[ep].get(key)
                            for key in (
                                "route_lane_ineligible_seconds", "route_lane_ineligible_stopped_seconds",
                                "route_lane_ineligible_stall_max_seconds", "mean_actual_speed_mps",
                                "stopped_fraction_of_speed_samples", "min_cv_obb_ttc_s", "risk_evaluable_ticks",
                                "cv_ttc_below_3s_seconds", "critical_unobserved_ticks",
                            )
                        },
                        "terminal_info": {
                            key: (episode_map[ep].get("terminal_info") or {}).get(key)
                            for key in (
                                "current_lane_id", "current_road_id", "current_lane_reachability_label",
                                "target_speed", "actual_speed_mps", "lane_command",
                                "lane_command_requested", "lane_control_request_status",
                                "lane_control_request_reason", "lane_change_applied",
                                "actual_lane_transition_since_previous_decision",
                            )
                        },
                    }
                    for ep in timeout_eps
                ],
            },
            "collisions": {
                "terminal_obb_replay_validation": replay_validation,
                "terminal_lane_road_counts": [
                    {"road_id": road, "lane_id": lane, "count": count}
                    for (road, lane), count in terminal_lane_counts.most_common()
                ],
                "terminal_request_status_and_command_counts": [
                    {"request_status": status, "command": command, "count": count}
                    for (status, command), count in collision_request_counts.most_common()
                ],
                "geometric_count": sum(detail["event_evidence"]["geometric_collision"] for detail in collision_details),
                "raw_sumo_collision_count": sum(detail["event_evidence"]["raw_sumo_collision"] for detail in collision_details),
                "native_sumo_collision_partner_identity_available_count": sum(detail["event_evidence"]["native_sumo_collision_object_identity_available"] for detail in collision_details),
                "terminal_obb_replay_unique_partner_count": sum(
                    detail["terminal_geometry_partner_reconstruction"]["status"] == "unique_replay_partner"
                    for detail in collision_details
                ),
                "terminal_obb_replay_candidate_count_distribution": dict(
                    Counter(
                        str(detail["terminal_geometry_partner_reconstruction"]["candidate_count"])
                        for detail in collision_details
                    )
                ),
                "reconstructed_partner_road_lane_counts": [
                    {"road_id": road, "lane_id": lane, "count": count}
                    for (road, lane), count in Counter(
                        (
                            partner.get("road_id"), partner.get("lane_id")
                        )
                        for detail in collision_details
                        for partner in detail["terminal_geometry_partner_reconstruction"]["partners"]
                    ).most_common()
                ],
                "reconstructed_partner_background_flow_prefix_counts": dict(
                    Counter(
                        partner.get("background_flow_id_prefix")
                        for detail in collision_details
                        for partner in detail["terminal_geometry_partner_reconstruction"]["partners"]
                    )
                ),
                "reconstructed_partner_route_edge_sequences": [
                    {"route_edges": route_edges, "count": count}
                    for route_edges, count in Counter(
                        tuple(partner.get("route_edges") or [])
                        for detail in collision_details
                        for partner in detail["terminal_geometry_partner_reconstruction"]["partners"]
                    ).most_common()
                ],
                "reconstructed_partner_motion_relation_counts": dict(
                    Counter(
                        partner.get("motion_relation")
                        for detail in collision_details
                        for partner in detail["terminal_geometry_partner_reconstruction"]["partners"]
                    )
                ),
                "reconstructed_partner_collision_tick_speed_mps": stats(
                    partner.get("speed_mps")
                    for detail in collision_details
                    for partner in detail["terminal_geometry_partner_reconstruction"]["partners"]
                ),
                "partner_last30_raw_tick_coverage": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["partner_observed_samples"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "partner_last30_raw_tick_ego_stopped_fraction": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["ego_stopped_fraction_below_0p1mps"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "partner_last30_raw_tick_ego_target_speed_mps_episode_mean": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["ego_target_speed_mps"]["mean"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "partner_last30_raw_tick_ego_actual_speed_mps_episode_mean": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["ego_actual_speed_mps"]["mean"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "partner_last30_raw_tick_partner_speed_mps_episode_mean": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["partner_speed_mps"]["mean"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "partner_last30_raw_tick_positive_closing_fraction_episode_mean": stats(
                    detail["reconstructed_partner_last30_raw_ticks_before_collision"]["positive_radial_closing_speed_fraction"]
                    for detail in collision_details
                    if detail["reconstructed_partner_last30_raw_ticks_before_collision"] is not None
                ),
                "collision_tick_target_speed_mps": stats(
                    detail["collision_tick"]["target_speed_mps"]
                    for detail in collision_details
                ),
                "collision_tick_actual_speed_mps": stats(
                    detail["collision_tick"]["actual_speed_mps"]
                    for detail in collision_details
                ),
                "collision_tick_request_status_and_command_counts": [
                    {"request_status": status, "command": command, "count": count}
                    for (status, command), count in Counter(
                        (
                            detail["collision_tick"]["lane_control_request_status"],
                            detail["collision_tick"]["lane_command_requested"],
                        )
                        for detail in collision_details
                    ).most_common()
                ],
                "collision_tick_request_reason_counts": dict(
                    Counter(
                        detail["collision_tick"]["lane_control_request_reason"]
                        for detail in collision_details
                    )
                ),
                "collision_tick_lane_transition_changed_counts": dict(
                    Counter(
                        str(
                            (detail["collision_tick"]["actual_lane_transition"] or {}).get(
                                "changed"
                            )
                        )
                        for detail in collision_details
                    )
                ),
                "pre_collision_last30_raw_tick_summary": {
                    "target_speed_mps": stats(
                        value
                        for detail in collision_details
                        for value in [detail["pre_collision_last30_raw_ticks"]["target_speed_mps"]["mean"]]
                    ),
                    "actual_speed_mps": stats(
                        value
                        for detail in collision_details
                        for value in [detail["pre_collision_last30_raw_ticks"]["actual_speed_mps"]["mean"]]
                    ),
                    "risk_evaluable_ticks": stats(
                        detail["pre_collision_last30_raw_ticks"]["risk_evaluable_ticks"]
                        for detail in collision_details
                    ),
                    "minimum_ttc_across_pre_collision_windows_s": stats(
                        detail["pre_collision_last30_raw_ticks"]["min_cv_obb_ttc_s"]["min"]
                        for detail in collision_details
                    ),
                },
                "episode_details": [
                    {"episode": ep, "seed": episode_map[ep].get("seed"), **detail}
                    for ep, detail in zip(collision_eps, collision_details)
                ],
            },
        }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
