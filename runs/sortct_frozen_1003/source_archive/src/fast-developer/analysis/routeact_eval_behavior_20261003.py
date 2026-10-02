"""Read-only paired RouteAct vs historical STRT final-eval behavior audit.

Reads only existing eval behavior sidecars. It does not run SUMO, training, or
evaluation. Collision windows are reconstructed from raw telemetry at 0.1 s/tick.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[4]
RUNS = WORKSPACE / "runs"
ROUTEACT_RUN = RUNS / "sortct_1002" / "sac_mlp_d1_st_rt_routeact_v1__intersection_sorted_depart4p0"
STRT_RUN = RUNS / "d0929_100k_diag" / "sac_mlp_d1_st_rt__intersection_sorted_depart4p0"
OUTPUT = Path(__file__).with_name("routeact_eval_behavior_20261003.json")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_number(value):
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


def quantile(values, q):
    vals = sorted(float(v) for v in values if finite_number(v))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def numeric_summary(values):
    vals = [float(v) for v in values if finite_number(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": quantile(vals, 0.5),
        "p10": quantile(vals, 0.1),
        "p90": quantile(vals, 0.9),
        "min": min(vals),
        "max": max(vals),
    }


def episode_key(row):
    reset = row.get("reset_info") or {}
    seed = reset.get("requested_reset_seed", reset.get("simulation_seed", row.get("seed")))
    return (int(seed) if seed is not None else None, reset.get("traffic_variant"))


def episode_record(row):
    terminal = row.get("terminal_info") or {}
    snapshot_ego = ((row.get("last_snapshot") or {}).get("ego") or {})
    actual = terminal.get("actual_speed_mps")
    if actual is None:
        actual = snapshot_ego.get("speed")
    target = terminal.get("target_speed")
    if target is None:
        target = terminal.get("effective_target_speed")
    road = terminal.get("current_road_id") or snapshot_ego.get("road_id")
    lane = terminal.get("current_lane_id") or snapshot_ego.get("lane_id")
    reach = terminal.get("current_lane_reachability_label")
    route_reason = terminal.get("route_lane_status_reason")
    reset = row.get("reset_info") or {}
    evidence = row.get("collision_evidence") or []
    return {
        "episode": int(row["episode"]),
        "seed": int(reset.get("requested_reset_seed", reset.get("simulation_seed", row.get("seed")))),
        "traffic_variant": reset.get("traffic_variant"),
        "outcome": row.get("outcome"),
        "raw_steps": int(terminal.get("raw_simulation_steps") or 0),
        "decision_steps": int(terminal.get("decision_steps") or row.get("decisions") or 0),
        "mean_actual_speed_mps": row.get("mean_actual_speed_mps"),
        "stopped_fraction": row.get("stopped_fraction_of_speed_samples"),
        "terminal_actual_speed_mps": actual,
        "terminal_target_speed_mps": target,
        "terminal_target_minus_actual_mps": target - actual if finite_number(target) and finite_number(actual) else None,
        "terminal_road_id": road,
        "terminal_lane_id": lane,
        "terminal_location_class": "internal" if isinstance(road, str) and road.startswith(":") else ("edge" if road else "unknown"),
        "route_index": snapshot_ego.get("route_index"),
        "terminal_route_reachability_label": reach,
        "terminal_route_status_reason": route_reason,
        "route_lane_ineligible_ticks": row.get("route_lane_ineligible_ticks"),
        "route_lane_ineligible_stopped_seconds": row.get("route_lane_ineligible_stopped_seconds"),
        "risk_evaluable_ticks": row.get("risk_evaluable_ticks"),
        "risk_evaluable_low_ttc_ticks": row.get("risk_evaluable_low_ttc_ticks"),
        "min_cv_obb_ttc_s": row.get("min_cv_obb_ttc_s"),
        "collision_evidence_rows_with_any_event_detail": sum(bool((x.get("collision_events") or [])) for x in evidence if isinstance(x, dict)),
        "collision_evidence_rows": len(evidence),
        "raw_sumo_collision": bool(terminal.get("raw_sumo_collision")),
        "geometric_collision": bool(terminal.get("geometric_collision")),
        "diagnostic_error_count": int(row.get("diagnostic_error_count") or 0),
    }


def load_eval(run):
    rows = read_jsonl(run / "diagnostics" / "eval" / "episodes.jsonl")
    records = [episode_record(r) for r in rows]
    by_key = {episode_key(r): rec for r, rec in zip(rows, records)}
    return rows, records, by_key


def load_routeact_veto_counts(run, episodes):
    path = run / "diagnostics" / "eval" / "route_action_consistency.jsonl"
    counts = defaultdict(lambda: {"decisions": 0, "vetoes": 0, "reasons": Counter(), "speed_unchanged": 0, "lateral_changed": 0, "current_label": Counter(), "target_label": Counter(), "last_decision_step": -1, "last_decision": None})
    total = Counter()
    manifest_count = 0
    seed_alignment_errors = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "manifest":
                manifest_count += 1
                continue
            if row.get("record_type") != "decision":
                continue
            ep_index = int(row["episode_index"])
            ep_number = ep_index + 1
            if ep_index < 0 or ep_index >= len(episodes):
                seed_alignment_errors += 1
            elif row.get("episode_seed") is not None and int(row["episode_seed"]) != int(episodes[ep_index]["seed"]):
                seed_alignment_errors += 1
            c = counts[ep_number]
            c["decisions"] += 1
            total["decisions"] += 1
            d = row.get("decision") or {}
            rc = row.get("route_context") or {}
            tc = row.get("target_context") or {}
            increment_counter(c["current_label"], rc.get("current_lane_reachability_label"))
            increment_counter(c["target_label"], tc.get("target_lane_reachability_label"))
            step = int(row.get("decision_step") or 0)
            if step >= c["last_decision_step"]:
                c["last_decision_step"] = step
                c["last_decision"] = {
                    "decision_step": step,
                    "raw_step_before_action": row.get("raw_step_before_action"),
                    "raw_step_after_action": row.get("raw_step_after_action"),
                    "reason": d.get("reason"),
                    "veto_applied": bool(d.get("veto_applied")),
                    "current_road_id": rc.get("current_road_id"),
                    "current_lane_id": rc.get("current_lane_id"),
                    "current_lane_reachability_label": rc.get("current_lane_reachability_label"),
                    "planned_next_edge": rc.get("planned_next_edge"),
                    "target_lane_id": tc.get("target_lane_id"),
                    "target_lane_reachability_label": tc.get("target_lane_reachability_label"),
                    "lane_command_requested": tc.get("lane_command"),
                    "lane_command_forwarded": d.get("lane_command_forwarded"),
                    "speed_action_unchanged": d.get("speed_action_unchanged"),
                    "policy_proposed_action": row.get("policy_proposed_action"),
                    "env_action_forwarded": row.get("env_action_forwarded"),
                }
            if d.get("veto_applied"):
                c["vetoes"] += 1
                total["vetoes"] += 1
                c["reasons"][str(d.get("reason", "unknown"))] += 1
                a = row.get("policy_proposed_action") or []
                b = row.get("env_action_forwarded") or []
                if len(a) >= 2 and len(b) >= 2 and a[1] != b[1]:
                    c["lateral_changed"] += 1
                if d.get("speed_action_unchanged") is True and len(a) and len(b) and a[0] == b[0]:
                    c["speed_unchanged"] += 1
    for ep in episodes:
        c = counts.get(ep["episode"], {"decisions": 0, "vetoes": 0, "reasons": Counter(), "speed_unchanged": 0, "lateral_changed": 0, "current_label": Counter(), "target_label": Counter(), "last_decision": None})
        ep["route_action_sidecar_decisions"] = c["decisions"]
        ep["route_action_vetoes"] = c["vetoes"]
        ep["route_action_veto_reasons"] = dict(c["reasons"])
        ep["veto_lateral_changed"] = c["lateral_changed"]
        ep["veto_speed_unchanged"] = c["speed_unchanged"]
        ep["veto_current_lane_label_counts"] = dict(c["current_label"])
        ep["veto_target_lane_label_counts"] = dict(c["target_label"])
        ep["route_action_last_decision"] = c["last_decision"]
    total["manifest_records_excluded"] = manifest_count
    total["episode_seed_alignment_errors"] = seed_alignment_errors
    return dict(total), manifest_count


def increment_counter(counter, value):
    counter["null" if value is None else str(value)] += 1


def load_decision_target_spans(run):
    """Return episode-local raw spans and requested targets from decisions log."""
    path = run / "diagnostics" / "eval" / "decisions.jsonl.gz"
    by_episode = defaultdict(list)
    cursor = defaultdict(int)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ep = int(row["episode"])
            ticks = int(row.get("raw_ticks") or 0)
            start = cursor[ep] + 1
            end = cursor[ep] + ticks
            cursor[ep] += ticks
            control = row.get("control") or {}
            target = control.get("target_speed_mps")
            if ticks > 0:
                by_episode[ep].append({"start": start, "end": end, "ticks": ticks, "target_speed_mps": target, "lane_command": control.get("lane_command")})
    return by_episode, dict(cursor)


def window_mean_target(spans, low, high):
    vals = []
    for span in spans:
        if not finite_number(span.get("target_speed_mps")):
            continue
        overlap = max(0, min(high, span["end"]) - max(low, span["start"]) + 1)
        if overlap:
            vals.extend([float(span["target_speed_mps"])] * overlap)
    return sum(vals) / len(vals) if vals else None


def collision_windows(run, records, spans_by_episode):
    collision_eps = {r["episode"]: r for r in records if r["outcome"] == "collision"}
    raw_rows = defaultdict(list)
    path = run / "diagnostics" / "eval" / "raw_steps.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ep = int(row["episode"])
            if ep not in collision_eps:
                continue
            end = collision_eps[ep]["raw_steps"]
            raw_step = int(row["raw_step"])
            if max(1, end - 19) <= raw_step <= end:
                ego = row.get("ego") or {}
                sc = row.get("speed_control") or {}
                raw_rows[ep].append({
                    "raw_step": raw_step,
                    "actual_speed_mps": ego.get("speed"),
                    "target_speed_mps": sc.get("requested_target_speed_mps"),
                    "ego_state_source": row.get("ego_state_source"),
                    "road_id": ego.get("road_id"),
                    "lane_id": ego.get("lane_id"),
                    "route_index": ego.get("route_index"),
                    "current_lane_reachability_label": row.get("current_lane_reachability_label"),
                    "route_lane_context_known": row.get("route_lane_context_known"),
                    "route_lane_status_reason": row.get("route_lane_status_reason"),
                    "raw_sumo_collision": bool((row.get("events") or {}).get("raw_sumo_collision")),
                    "collision_ids_count": len(row.get("collision_ids") or []),
                    "collision_events_count": len(row.get("collision_events") or []),
                })
    per_collision = []
    for ep, rec in collision_eps.items():
        rows = sorted(raw_rows.get(ep, []), key=lambda x: x["raw_step"])
        end = rec["raw_steps"]
        row_summary = {"episode": ep, "seed": rec["seed"], "traffic_variant": rec["traffic_variant"], "raw_terminal": end}
        for seconds, width in ((1, 10), (2, 20)):
            window = [x for x in rows if x["raw_step"] >= max(1, end - width + 1)]
            actuals = [x["actual_speed_mps"] for x in window if finite_number(x["actual_speed_mps"])]
            targets = [x["target_speed_mps"] for x in window if finite_number(x["target_speed_mps"])]
            if not targets:
                targets = [window_mean_target(spans_by_episode.get(ep, []), max(1, end - width + 1), end)]
            lane_labels = Counter("null" if x["current_lane_reachability_label"] is None else str(x["current_lane_reachability_label"]) for x in window)
            roads = Counter(x["road_id"] or "null" for x in window)
            row_summary[f"last_{seconds}s"] = {
                "raw_rows": len(window),
                "actual_speed_mps": numeric_summary(actuals),
                "target_speed_mps": numeric_summary(targets),
                "near_stop_fraction_actual_lt_0p5mps": (sum(float(x) < 0.5 for x in actuals) / len(actuals)) if actuals else None,
                "current_lane_reachability_label_counts": dict(lane_labels),
                "road_id_counts": dict(roads),
                "raw_sumo_collision_rows": sum(x["raw_sumo_collision"] for x in window),
                "collision_event_detail_rows": sum(x["collision_events_count"] > 0 for x in window),
            }
        row_summary["last_raw_state"] = rows[-1] if rows else None
        row_summary["expected_2s_raw_rows"] = min(20, end)
        per_collision.append(row_summary)
    return per_collision


def summarize_method(label, records, route_counts=None, collision_rows=None, decision_tick_sums=None):
    outcomes = Counter(r["outcome"] for r in records)
    by_outcome = {}
    for outcome in sorted(outcomes):
        subset = [r for r in records if r["outcome"] == outcome]
        by_outcome[outcome] = {
            "episodes": len(subset),
            "mean_raw_steps": numeric_summary([r["raw_steps"] for r in subset]),
            "mean_actual_speed_mps_over_episode": numeric_summary([r["mean_actual_speed_mps"] for r in subset]),
            "stopped_fraction_of_speed_samples": numeric_summary([r["stopped_fraction"] for r in subset]),
            "terminal_actual_speed_mps": numeric_summary([r["terminal_actual_speed_mps"] for r in subset]),
            "terminal_target_speed_mps": numeric_summary([r["terminal_target_speed_mps"] for r in subset]),
            "terminal_target_minus_actual_mps": numeric_summary([r["terminal_target_minus_actual_mps"] for r in subset]),
            "terminal_location_class_counts": dict(Counter(r["terminal_location_class"] for r in subset)),
            "terminal_road_counts": dict(Counter(r["terminal_road_id"] or "null" for r in subset)),
            "terminal_lane_reachability_label_counts": dict(Counter("null" if r["terminal_route_reachability_label"] is None else str(r["terminal_route_reachability_label"]) for r in subset)),
            "whole_episode_min_cv_obb_ttc_s": numeric_summary([r["min_cv_obb_ttc_s"] for r in subset]),
            "episodes_with_route_ineligible_samples": sum((r["route_lane_ineligible_ticks"] or 0) > 0 for r in subset),
            "raw_sumo_collision_episodes": sum(r["raw_sumo_collision"] for r in subset),
            "geometric_collision_episodes": sum(r["geometric_collision"] for r in subset),
            "episodes_with_any_collision_event_detail": sum(r["collision_evidence_rows_with_any_event_detail"] > 0 for r in subset),
            "last_route_action_reason_counts": dict(Counter((r.get("route_action_last_decision") or {}).get("reason", "null") for r in subset)) if route_counts is not None else None,
            "last_route_action_veto_count": sum(bool((r.get("route_action_last_decision") or {}).get("veto_applied")) for r in subset) if route_counts is not None else None,
            "last_route_action_current_lane_label_counts": dict(Counter(str((r.get("route_action_last_decision") or {}).get("current_lane_reachability_label")) for r in subset)) if route_counts is not None else None,
            "last_route_action_target_lane_label_counts": dict(Counter(str((r.get("route_action_last_decision") or {}).get("target_lane_reachability_label")) for r in subset)) if route_counts is not None else None,
        }
        if route_counts is not None:
            by_outcome[outcome]["route_action_decisions"] = sum(r["route_action_sidecar_decisions"] for r in subset)
            by_outcome[outcome]["route_action_vetoes"] = sum(r["route_action_vetoes"] for r in subset)
            by_outcome[outcome]["episodes_with_any_veto"] = sum(r["route_action_vetoes"] > 0 for r in subset)
            denom = by_outcome[outcome]["route_action_decisions"]
            by_outcome[outcome]["vetoes_per_decision"] = by_outcome[outcome]["route_action_vetoes"] / denom if denom else None
        if collision_rows is not None:
            subset_keys = {(r["seed"], r["traffic_variant"]) for r in subset}
            crows = [c for c in collision_rows if (c["seed"], c["traffic_variant"]) in subset_keys]
            for seconds in (1, 2):
                key = f"last_{seconds}s"
                vals = [c[key]["actual_speed_mps"]["mean"] for c in crows if c.get(key) and c[key]["actual_speed_mps"]["mean"] is not None]
                targets = [c[key]["target_speed_mps"]["mean"] for c in crows if c.get(key) and c[key]["target_speed_mps"]["mean"] is not None]
                by_outcome[outcome][f"collision_terminal_window_{seconds}s_episode_mean_actual_speed_mps"] = numeric_summary(vals)
                by_outcome[outcome][f"collision_terminal_window_{seconds}s_episode_mean_target_speed_mps"] = numeric_summary(targets)
    return {
        "label": label,
        "episodes": len(records),
        "outcomes": dict(outcomes),
        "mean_raw_steps": numeric_summary([r["raw_steps"] for r in records]),
        "by_outcome": by_outcome,
        "route_action_sidecar_totals": route_counts,
        "collision_window_episode_count": len(collision_rows) if collision_rows is not None else 0,
        "decision_raw_tick_sum_mismatches": decision_tick_sums,
    }


def main():
    route_rows, route_records, route_by_key = load_eval(ROUTEACT_RUN)
    base_rows, base_records, base_by_key = load_eval(STRT_RUN)
    route_counts, route_manifest_count = load_routeact_veto_counts(ROUTEACT_RUN, route_records)
    route_spans, route_tick_sums = load_decision_target_spans(ROUTEACT_RUN)
    base_spans, base_tick_sums = load_decision_target_spans(STRT_RUN)
    route_collision_windows = collision_windows(ROUTEACT_RUN, route_records, route_spans)
    base_collision_windows = collision_windows(STRT_RUN, base_records, base_spans)

    route_keyset = set(route_by_key)
    base_keyset = set(base_by_key)
    shared = route_keyset & base_keyset
    pair_matrix = defaultdict(Counter)
    for key in sorted(shared):
        pair_matrix[base_by_key[key]["outcome"]][route_by_key[key]["outcome"]] += 1
    unmatched_route = sorted(list(route_keyset - base_keyset), key=str)
    unmatched_base = sorted(list(base_keyset - route_keyset), key=str)

    for rec in route_records:
        if "route_action_sidecar_decisions" not in rec:
            rec["route_action_sidecar_decisions"] = 0
            rec["route_action_vetoes"] = 0
            rec["route_action_veto_reasons"] = {}
            rec["veto_lateral_changed"] = 0
            rec["veto_speed_unchanged"] = 0
    route_tick_mismatch = {
        "episodes_with_mismatch": sum(route_tick_sums.get(r["episode"], 0) != r["raw_steps"] for r in route_records),
        "max_abs_raw_difference": max((abs(route_tick_sums.get(r["episode"], 0) - r["raw_steps"]) for r in route_records), default=0),
    }
    base_tick_mismatch = {
        "episodes_with_mismatch": sum(base_tick_sums.get(r["episode"], 0) != r["raw_steps"] for r in base_records),
        "max_abs_raw_difference": max((abs(base_tick_sums.get(r["episode"], 0) - r["raw_steps"]) for r in base_records), default=0),
    }
    result = {
        "title": "RouteAct and historical STRT final-evaluation behavior audit",
        "as_of": "2026-10-03",
        "sources": {
            "routeact_run": str(ROUTEACT_RUN),
            "strt_reference_run": str(STRT_RUN),
            "routeact_eval_episode_file": str(ROUTEACT_RUN / "diagnostics" / "eval" / "episodes.jsonl"),
            "routeact_eval_route_action_sidecar": str(ROUTEACT_RUN / "diagnostics" / "eval" / "route_action_consistency.jsonl"),
            "routeact_eval_raw_steps": str(ROUTEACT_RUN / "diagnostics" / "eval" / "raw_steps.jsonl.gz"),
            "strt_eval_episode_file": str(STRT_RUN / "diagnostics" / "eval" / "episodes.jsonl"),
            "strt_eval_raw_steps": str(STRT_RUN / "diagnostics" / "eval" / "raw_steps.jsonl.gz"),
        },
        "pairing": {
            "key": "(reset_info.requested_reset_seed, reset_info.traffic_variant)",
            "routeact_eval_n": len(route_records),
            "strt_eval_n": len(base_records),
            "matched_seed_template_pairs": len(shared),
            "unmatched_routeact": unmatched_route,
            "unmatched_strt": unmatched_base,
            "outcome_matrix_rows_strt_columns_routeact": {k: dict(v) for k, v in pair_matrix.items()},
            "routeact_decision_raw_tick_sum_check": route_tick_mismatch,
            "strt_decision_raw_tick_sum_check": base_tick_mismatch,
        },
        "routeact_final_eval": summarize_method("RouteAct", route_records, route_counts, route_collision_windows, route_tick_mismatch),
        "strt_reference_final_eval": summarize_method("historical STRT", base_records, None, base_collision_windows, base_tick_mismatch),
        "routeact_route_action_manifest_rows_excluded": route_manifest_count,
        "routeact_route_sidecar_episode_seed_alignment_errors": route_counts.get("episode_seed_alignment_errors"),
        "routeact_veto_rate_by_episode_outcome": {
            outcome: {
                "episodes": len([r for r in route_records if r["outcome"] == outcome]),
                "episodes_with_any_veto": sum(r.get("route_action_vetoes", 0) > 0 for r in route_records if r["outcome"] == outcome),
                "route_action_decisions": sum(r.get("route_action_sidecar_decisions", 0) for r in route_records if r["outcome"] == outcome),
                "route_action_vetoes": sum(r.get("route_action_vetoes", 0) for r in route_records if r["outcome"] == outcome),
            }
            for outcome in sorted({r["outcome"] for r in route_records})
        },
        "routeact_collision_windows": route_collision_windows,
        "strt_collision_windows": base_collision_windows,
        "episode_records_routeact": route_records,
        "episode_records_strt": base_records,
        "interpretation_limits": [
            "Outcome comparisons are paired on reset seed and traffic template, but single training seeds and the historical STRT run has no frozen source archive; this is descriptive rather than a perfect source-controlled causal estimate.",
            "A target-lane reachability label is a static route-continuation check, not a conflict-safety or collision-prevention label.",
            "The RouteAct sidecar logs the proposed policy action and the action forwarded to the environment; a veto is not evidence of rescue or physical lane transition.",
            "Collision window summaries describe observed speed/location near terminal time. They do not identify collision partner or establish a collision cause.",
            "Old STRT raw telemetry lacks RouteAct reachability labels and per-raw requested target speed; its decision-side control target is used for collision-window weighting where available.",
            "Current RouteAct terminal route-continuation labels and internal-lane locations must not be interpreted as a full maneuver-feasibility or safety proof.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print("pairs", result["pairing"]["matched_seed_template_pairs"], "matrix", result["pairing"]["outcome_matrix_rows_strt_columns_routeact"])
    print("RouteAct", result["routeact_final_eval"]["outcomes"], "STRT", result["strt_reference_final_eval"]["outcomes"])
    for key in ("success", "collision"):
        print("A", key, result["routeact_final_eval"]["by_outcome"].get(key))
        print("STRT", key, result["strt_reference_final_eval"]["by_outcome"].get(key))
    print("veto", route_counts)


if __name__ == "__main__":
    main()
