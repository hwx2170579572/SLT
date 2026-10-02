"""Read-only aggregation of RouteActionConsistency training telemetry.

The raw sidecars are preserved. This script derives a global training raw-step
coordinate from the ordered behavior episode records and aggregates action
mapping counts without treating a veto as a successful rescue.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[4]
RUN = WORKSPACE / "runs" / "sortct_1002" / "sac_mlp_d1_st_rt_routeact_v1__intersection_sorted_depart4p0"
DIAG = RUN / "diagnostics" / "train"
OUTPUT = Path(__file__).with_name("routeact_training_diagnostic_20261003.json")
INTERVALS = [(0, 5_000), (5_000, 20_000), (20_000, 50_000), (50_000, 100_000)]


def load_episode_index():
    episodes = {}
    cumulative_raw = 0
    with (DIAG / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_number = int(row["episode"])
            terminal = row.get("terminal_info") or {}
            raw_value = terminal.get("raw_simulation_steps")
            raw_length = int(raw_value) if raw_value is not None else None
            last_ego = (row.get("last_snapshot") or {}).get("ego") or {}
            terminal_actual_speed = terminal.get("actual_speed_mps")
            if terminal_actual_speed is None:
                terminal_actual_speed = last_ego.get("speed")
            episodes[episode_number - 1] = {
                "episode": episode_number,
                "global_raw_start": cumulative_raw,
                "raw_length": raw_length,
                "outcome": row.get("outcome"),
                "decision_steps": int(terminal.get("decision_steps") or 0),
                "route_lane_ineligible_stopped_seconds": row.get("route_lane_ineligible_stopped_seconds"),
                "route_lane_ineligible_seconds": row.get("route_lane_ineligible_seconds"),
                "route_lane_stall_max_seconds": row.get("route_lane_ineligible_stall_max_seconds"),
                "stopped_fraction": row.get("stopped_fraction_of_speed_samples"),
                "terminal_lane_id": terminal.get("current_lane_id"),
                "terminal_road_id": terminal.get("current_road_id"),
                "terminal_actual_speed_mps": terminal_actual_speed,
                "terminal_target_speed_mps": terminal.get("target_speed", terminal.get("effective_target_speed")),
                "terminal_lane_command": terminal.get("lane_command"),
                "terminal_lane_reachability_label": terminal.get("current_lane_reachability_label"),
                "terminal_route_reason": terminal.get("route_lane_status_reason"),
            }
            if raw_length is not None:
                cumulative_raw += raw_length
    return episodes, cumulative_raw


def empty_bin():
    return {
        "decisions": 0,
        "vetoes": 0,
        "veto_reasons": Counter(),
        "all_decision_reasons": Counter(),
        "veto_lateral_changed": 0,
        "veto_speed_changed": 0,
        "veto_speed_unchanged_flag_false": 0,
        "all_action_dimensions_match": 0,
        "context_known": 0,
        "context_fresh": 0,
        "current_lane_label": Counter(),
        "target_lane_label": Counter(),
        "target_route_context_known": Counter(),
        "target_lane_reason": Counter(),
        "veto_lane_command_requested": Counter(),
        "veto_target_lane_id": Counter(),
        "veto_current_lane_id": Counter(),
    }


def increment_counter(counter, value):
    counter["null" if value is None else str(value)] += 1


def main():
    episodes, completed_episode_raw = load_episode_index()
    bins = [empty_bin() for _ in INTERVALS]
    per_episode = defaultdict(lambda: {"decisions": 0, "vetoes": 0, "reasons": Counter()})
    unknown_episode_rows = 0
    manifest_rows = 0
    sidecar_decision_rows = 0
    sidecar_outside_budget_rows = 0
    sidecar_outside_budget_vetoes = 0
    total_decisions = 0
    total_vetoes = 0
    first_raw = None
    last_raw = None
    first_raw_after = None
    max_global_raw_after = None
    partial_episode = {"episode_index": None, "decisions": 0, "vetoes": 0, "reasons": Counter(), "max_local_raw_after": 0, "sidecar_rows": 0, "zero_raw_rows": 0}

    with (DIAG / "route_action_consistency.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "manifest":
                manifest_rows += 1
                continue
            if row.get("record_type") != "decision":
                continue
            sidecar_decision_rows += 1
            ep_index = int(row["episode_index"])
            ep = episodes.get(ep_index)
            if ep is not None and ep.get("raw_length") is None:
                # Behavior telemetry emits an `incomplete` episode at the raw
                # budget boundary; its start offset follows all completed episodes.
                partial_episode["episode_index"] = ep_index
                per_episode[ep_index] = partial_episode
                ep_stats = per_episode[ep_index]
            elif ep is not None:
                ep_stats = per_episode[ep_index]
            else:
                unknown_episode_rows += 1
                continue
            local_raw = int(row.get("raw_step_before_action") or 0)
            global_raw = ep["global_raw_start"] + local_raw
            local_raw_after = int(row.get("raw_step_after_action") or local_raw)
            global_raw_after = ep["global_raw_start"] + local_raw_after
            if ep.get("raw_length") is None:
                partial_episode["sidecar_rows"] += 1
                partial_episode["max_local_raw_after"] = max(partial_episode["max_local_raw_after"], local_raw_after)
                if global_raw_after == global_raw:
                    partial_episode["zero_raw_rows"] += 1
            first_raw = global_raw if first_raw is None else min(first_raw, global_raw)
            last_raw = global_raw if last_raw is None else max(last_raw, global_raw)
            first_raw_after = global_raw_after if first_raw_after is None else min(first_raw_after, global_raw_after)
            max_global_raw_after = global_raw_after if max_global_raw_after is None else max(max_global_raw_after, global_raw_after)
            bucket_index = next((i for i, (start, end) in enumerate(INTERVALS) if start <= global_raw < end), None)
            if bucket_index is None:
                sidecar_outside_budget_rows += 1
                if bool((row.get("decision") or {}).get("veto_applied")):
                    sidecar_outside_budget_vetoes += 1
                continue
            if ep.get("raw_length") is None:
                partial_episode["decision_rows_inside_budget"] = partial_episode.get("decision_rows_inside_budget", 0) + 1
            b = bins[bucket_index]
            b["decisions"] += 1
            total_decisions += 1
            ep_stats["decisions"] += 1

            decision = row.get("decision") or {}
            route_context = row.get("route_context") or {}
            target_context = row.get("target_context") or {}
            env_report = row.get("environment_reported") or {}
            reason = decision.get("reason", "unknown")
            b["all_decision_reasons"][str(reason)] += 1
            b["context_known"] += bool(decision.get("context_known"))
            b["context_fresh"] += bool(decision.get("context_fresh"))
            increment_counter(b["current_lane_label"], route_context.get("current_lane_reachability_label"))
            increment_counter(b["target_lane_label"], target_context.get("target_lane_reachability_label"))
            increment_counter(b["target_route_context_known"], target_context.get("target_route_context_known"))
            increment_counter(b["target_lane_reason"], target_context.get("target_lane_reason"))

            proposed = row.get("policy_proposed_action") or []
            forwarded = row.get("env_action_forwarded") or []
            both_dims = len(proposed) >= 2 and len(forwarded) >= 2
            dims_match = both_dims and proposed[0] == forwarded[0] and proposed[1] == forwarded[1]
            if dims_match:
                b["all_action_dimensions_match"] += 1

            veto = bool(decision.get("veto_applied"))
            if veto:
                b["vetoes"] += 1
                total_vetoes += 1
                ep_stats["vetoes"] += 1
                b["veto_reasons"][str(reason)] += 1
                ep_stats["reasons"][str(reason)] += 1
                increment_counter(b["veto_lane_command_requested"], target_context.get("lane_command"))
                increment_counter(b["veto_target_lane_id"], target_context.get("target_lane_id"))
                increment_counter(b["veto_current_lane_id"], route_context.get("current_lane_id"))
                lane_changed = len(proposed) >= 2 and len(forwarded) >= 2 and proposed[1] != forwarded[1]
                speed_changed = len(proposed) >= 1 and len(forwarded) >= 1 and proposed[0] != forwarded[0]
                if lane_changed:
                    b["veto_lateral_changed"] += 1
                if speed_changed:
                    b["veto_speed_changed"] += 1
                if decision.get("speed_action_unchanged") is False:
                    b["veto_speed_unchanged_flag_false"] += 1

    serialized_bins = []
    for (start, end), b in zip(INTERVALS, bins):
        n = b["decisions"]
        v = b["vetoes"]
        serialized_bins.append({
            "global_raw_interval": [start, end],
            "decisions": n,
            "vetoes": v,
            "veto_rate": (v / n) if n else None,
            "veto_reasons": dict(b["veto_reasons"]),
            "all_decision_reasons": dict(b["all_decision_reasons"]),
            "among_vetoes": {
                "lateral_component_changed": b["veto_lateral_changed"],
                "longitudinal_component_changed": b["veto_speed_changed"],
                "speed_action_unchanged_flag_false": b["veto_speed_unchanged_flag_false"],
            },
            "all_proposal_forwarded_action_dims_exactly_equal": b["all_action_dimensions_match"],
            "context_known_decisions": b["context_known"],
            "context_fresh_decisions": b["context_fresh"],
            "current_lane_reachability_label_counts": dict(b["current_lane_label"]),
            "target_lane_reachability_label_counts": dict(b["target_lane_label"]),
            "target_route_context_known_counts": dict(b["target_route_context_known"]),
            "target_lane_resolution_reason_counts": dict(b["target_lane_reason"]),
            "veto_requested_lane_command_counts": dict(b["veto_lane_command_requested"]),
            "veto_target_lane_id_counts": dict(b["veto_target_lane_id"]),
            "veto_current_lane_id_counts": dict(b["veto_current_lane_id"]),
        })

    outcome_groups = defaultdict(lambda: {"episodes": 0, "episodes_with_veto": 0, "episodes_without_veto": 0, "decisions": 0, "vetoes": 0, "raw_steps": [], "route_stopped_seconds": [], "stopped_fraction": [], "terminal_actual_speed": [], "terminal_target_speed": [], "terminal_roads": Counter(), "terminal_lanes": Counter(), "terminal_reachability": Counter()})
    episode_rows = []
    for ep_index, ep in sorted(episodes.items()):
        es = per_episode.get(ep_index, {"decisions": 0, "vetoes": 0, "reasons": Counter()})
        outcome = ep["outcome"] or "unknown"
        g = outcome_groups[outcome]
        g["episodes"] += 1
        g["episodes_with_veto"] += int(es["vetoes"] > 0)
        g["episodes_without_veto"] += int(es["vetoes"] == 0)
        g["decisions"] += es["decisions"]
        g["vetoes"] += es["vetoes"]
        if ep["raw_length"] is not None:
            g["raw_steps"].append(ep["raw_length"])
        if ep["terminal_actual_speed_mps"] is not None:
            g["terminal_actual_speed"].append(ep["terminal_actual_speed_mps"])
        if ep["terminal_target_speed_mps"] is not None:
            g["terminal_target_speed"].append(ep["terminal_target_speed_mps"])
        increment_counter(g["terminal_roads"], ep["terminal_road_id"])
        increment_counter(g["terminal_lanes"], ep["terminal_lane_id"])
        increment_counter(g["terminal_reachability"], ep["terminal_lane_reachability_label"])
        if ep["route_lane_ineligible_stopped_seconds"] is not None:
            g["route_stopped_seconds"].append(ep["route_lane_ineligible_stopped_seconds"])
        if ep["stopped_fraction"] is not None:
            g["stopped_fraction"].append(ep["stopped_fraction"])
        episode_rows.append({
            "episode_index": ep_index,
            "episode": ep["episode"],
            "global_raw_interval": [ep["global_raw_start"], ep["global_raw_start"] + ep["raw_length"] if ep["raw_length"] is not None else None],
            "raw_length": ep["raw_length"],
            "outcome": outcome,
            "route_action_decisions": es["decisions"],
            "route_action_vetoes": es["vetoes"],
            "veto_reasons": dict(es["reasons"]),
            "route_lane_ineligible_stopped_seconds": ep["route_lane_ineligible_stopped_seconds"],
            "route_lane_ineligible_seconds": ep["route_lane_ineligible_seconds"],
            "route_lane_stall_max_seconds": ep["route_lane_stall_max_seconds"],
            "stopped_fraction": ep["stopped_fraction"],
            "terminal_lane_id": ep["terminal_lane_id"],
            "terminal_road_id": ep["terminal_road_id"],
            "terminal_actual_speed_mps": ep["terminal_actual_speed_mps"],
            "terminal_target_speed_mps": ep["terminal_target_speed_mps"],
            "terminal_lane_command": ep["terminal_lane_command"],
            "terminal_lane_reachability_label": ep["terminal_lane_reachability_label"],
            "terminal_route_reason": ep["terminal_route_reason"],
        })
    outcome_summary = {}
    for outcome, g in outcome_groups.items():
        outcome_summary[outcome] = {
            "episodes": g["episodes"],
            "episodes_with_veto": g["episodes_with_veto"],
            "episodes_without_veto": g["episodes_without_veto"],
            "route_action_decisions": g["decisions"],
            "route_action_vetoes": g["vetoes"],
            "vetoes_per_decision": g["vetoes"] / g["decisions"] if g["decisions"] else None,
            "mean_completed_episode_raw_steps": sum(g["raw_steps"]) / len(g["raw_steps"]) if g["raw_steps"] else None,
            "mean_terminal_actual_speed_mps": sum(g["terminal_actual_speed"]) / len(g["terminal_actual_speed"]) if g["terminal_actual_speed"] else None,
            "mean_terminal_target_speed_mps": sum(g["terminal_target_speed"]) / len(g["terminal_target_speed"]) if g["terminal_target_speed"] else None,
            "terminal_road_counts": dict(g["terminal_roads"]),
            "terminal_lane_counts": dict(g["terminal_lanes"]),
            "terminal_lane_reachability_label_counts": dict(g["terminal_reachability"]),
            "mean_route_lane_ineligible_stopped_seconds": sum(g["route_stopped_seconds"]) / len(g["route_stopped_seconds"]) if g["route_stopped_seconds"] else None,
            "mean_stopped_fraction": sum(g["stopped_fraction"]) / len(g["stopped_fraction"]) if g["stopped_fraction"] else None,
        }

    args = json.loads((RUN / "arguments.json").read_text(encoding="utf-8"))
    result = {
        "title": "RouteAct training diagnostic aggregation (no final-evaluation analysis)",
        "source_run": str(RUN),
        "method": "sac_mlp_d1_st_rt_routeact_v1",
        "training_identity": {
            "scenario": args.get("scenario"),
            "seed": args.get("seed"),
            "raw_step_budget": 100000,
            "training_complete": json.loads((RUN / "training_complete.json").read_text(encoding="utf-8")),
            "final_evaluation_status": "intentionally_not_analyzed_in_this_artifact",
        },
        "time_mapping": {
            "sidecar_raw_step_before_action_is_episode_local": True,
            "episode_global_offset_source": "cumulative terminal_info.raw_simulation_steps in ordered diagnostics/train/episodes.jsonl",
            "training_episode_records": len(episodes),
            "completed_episode_raw_sum": completed_episode_raw,
            "first_decision_global_raw": first_raw,
            "last_decision_global_raw_including_endpoint": last_raw,
            "first_action_end_global_raw": first_raw_after,
            "maximum_action_end_global_raw": max_global_raw_after,
            "unfinished_tail_episode": {
                "episode_index": partial_episode["episode_index"],
                "sidecar_rows_total": partial_episode["sidecar_rows"],
                "decision_rows_inside_budget": partial_episode.get("decision_rows_inside_budget", 0),
                "veto_rows": partial_episode["vetoes"],
                "zero_raw_advance_rows": partial_episode["zero_raw_rows"],
                "max_local_raw_after": partial_episode["max_local_raw_after"],
                "global_raw_offset": episodes[partial_episode["episode_index"]]["global_raw_start"] if partial_episode["episode_index"] is not None else None,
            },
            "unmatched_sidecar_decision_rows": unknown_episode_rows,
            "manifest_rows_excluded": manifest_rows,
            "bucket_convention": "half-open [start,end), each decision binned by global raw_step_before_action",
        },
        "overall": {
            "sidecar_decision_rows_total_including_boundary_records": sidecar_decision_rows,
            "sidecar_rows_outside_half_open_training_budget": sidecar_outside_budget_rows,
            "outside_budget_veto_rows": sidecar_outside_budget_vetoes,
            "decision_rows": total_decisions,
            "veto_rows": total_vetoes,
            "veto_rate": total_vetoes / total_decisions if total_decisions else None,
            "episodes_with_veto": sum(1 for x in episode_rows if x["route_action_vetoes"]),
            "episodes_without_veto": sum(1 for x in episode_rows if not x["route_action_vetoes"]),
        },
        "raw_step_bins": serialized_bins,
        "outcome_association_not_causal": outcome_summary,
        "episodes": episode_rows,
        "interpretation_limits": [
            "A veto is an execution-side request mapping, not evidence that a collision or timeout was prevented.",
            "The policy-proposed action remains the replay action; the environment receives the forwarded action.",
            "Lane requests and environment-reported request status are not proof of a physical lane transition.",
            "Per-outcome veto rates are observational and confounded by episode duration and policy state visitation.",
            "Final 100-episode evaluation is not included in this training-only aggregation.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"episode_records={len(episodes)} completed_episode_raw_sum={completed_episode_raw} decisions={total_decisions} vetoes={total_vetoes} unmatched={unknown_episode_rows}")
    for b in serialized_bins:
        print(f"{b['global_raw_interval']} n={b['decisions']} veto={b['vetoes']} rate={b['veto_rate']}")


if __name__ == "__main__":
    main()
