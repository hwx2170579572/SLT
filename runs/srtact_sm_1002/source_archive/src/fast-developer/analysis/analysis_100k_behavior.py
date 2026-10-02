"""Read-only aggregation of the 2026-09-29 embedded diagnostics run.

This script reads already-written JSONL/GZIP artifacts. It does not load a
model, start SUMO, or modify training artifacts. Output is a compact JSON
report used to audit outcome-conditioned behavior and data coverage.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


RUN = Path(r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\d0929_100k_diag")
OUT = Path(__file__).with_name("analysis_100k_behavior.json")
METHODS = {
    "sac_mlp": ("sac_mlp__intersection_sorted_depart4p0", "eval_worker_00"),
    "sac_mlp_d1_st_rt": ("sac_mlp_d1_st_rt__intersection_sorted_depart4p0", "eval"),
}


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def gzjsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def avg(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return mean(xs) if xs else None


def episode_outcome(record):
    return record.get("outcome") or "unknown"


def load_method(method, dirname, phase):
    base = RUN / dirname
    diag = base / "diagnostics" / phase
    episodes = list(jsonl(diag / "episodes.jsonl"))
    by_episode = {int(row["episode"]): row for row in episodes}
    decisions_by_episode = defaultdict(list)
    for row in gzjsonl(diag / "decisions.jsonl.gz"):
        e = int(row["episode"])
        decisions_by_episode[e].append(row)

    raw_by_episode = defaultdict(list)
    qc = Counter()
    previous_time = {}
    event_rows = defaultdict(list)
    decision_speed = defaultdict(list)
    for row in gzjsonl(diag / "raw_steps.jsonl.gz"):
        ep = int(row["episode"])
        qc["raw_records"] += 1
        qc["ego_missing"] += row.get("ego") is None
        qc["ego_current"] += row.get("ego_state_source") == "current"
        qc["ego_pre_step_removed"] += row.get("ego_state_source") == "pre_step_removed"
        qc["risk_evaluable"] += bool(row.get("risk_evaluable"))
        qc["diagnostic_error_rows"] += bool(row.get("errors"))
        if row.get("ego_state_source") == "current" and row.get("ego_state_time") is not None:
            if abs(float(row["sim_time"]) - float(row["ego_state_time"])) > 1e-6:
                qc["current_state_time_mismatch"] += 1
        t = row.get("sim_time")
        if t is not None and ep in previous_time and float(t) + 1e-6 < previous_time[ep]:
            qc["nonmonotonic_sim_time"] += 1
        if t is not None:
            previous_time[ep] = float(t)
        if row.get("events", {}).get("collision"):
            event_rows[ep].append(row)
        if row.get("ego") is not None and row.get("decision") is not None:
            decision_speed[(ep, int(row["decision"]))].append(float(row["ego"]["speed"]))
        # Keep only values needed for collision-window checks, not the full raw
        # actor lists, so this remains a small in-memory audit.
        conflict_vehicles = [
            v for v in row.get("vehicles", [])
            if (v.get("risk_cv") or {}).get("candidate_conflict_cv")
        ]
        raw_by_episode[ep].append({
            "raw_step": row.get("raw_step"),
            "sim_time": row.get("sim_time"),
            "step_seconds": row.get("step_seconds"),
            "decision": row.get("decision"),
            "ego_state_source": row.get("ego_state_source"),
            "ego_speed": (row.get("ego") or {}).get("speed"),
            "risk_evaluable": bool(row.get("risk_evaluable")),
            "min_cv_obb_ttc_s": row.get("min_cv_obb_ttc_s"),
            "min_center_distance_m": row.get("min_center_distance_m"),
            "critical_unobserved": bool(row.get("critical_unobserved")),
            "conflict_count": len(conflict_vehicles),
            "conflict_observed_count": sum(bool(v.get("observed_by_policy")) for v in conflict_vehicles),
            "collision": bool(row.get("events", {}).get("collision")),
            "collision_events": row.get("collision_events") or [],
        })

    episode_summaries = {}
    outcome_groups = defaultdict(list)
    for episode, rec in by_episode.items():
        outcome = episode_outcome(rec)
        decisions = decisions_by_episode.get(episode, [])
        controls = [d.get("control", {}) for d in decisions]
        actions = [d.get("action_env_input") or [] for d in decisions]
        target_speeds = [c.get("target_speed_mps") for c in controls]
        speed_pairs = []
        for d in decisions:
            key = (episode, int(d.get("decision", -1)))
            speeds = decision_speed.get(key, [])
            target = (d.get("control") or {}).get("target_speed_mps")
            if speeds and target is not None:
                speed_pairs.append((float(target), mean(speeds)))
        action0 = [float(a[0]) for a in actions if len(a) > 0]
        action1 = [float(a[1]) for a in actions if len(a) > 1]
        last = raw_by_episode.get(episode, [])
        collision_times = [r["sim_time"] for r in event_rows.get(episode, []) if r.get("sim_time") is not None]
        collision_time = max(collision_times) if collision_times else None
        pre_windows = {}
        if collision_time is not None:
            for seconds in (1.0, 2.0):
                window = [
                    r for r in last
                    if r.get("sim_time") is not None
                    and collision_time - seconds <= float(r["sim_time"]) < collision_time - 1e-9
                    and r.get("risk_evaluable")
                    and r.get("ego_state_source") == "current"
                ]
                valid_ttc = [r["min_cv_obb_ttc_s"] for r in window if r.get("min_cv_obb_ttc_s") is not None]
                pre_windows[str(int(seconds)) + "s"] = {
                    "valid_raw_ticks": len(window),
                    "ttc_valid_ticks": len(valid_ttc),
                    "ticks_ttc_lt_1s": sum(float(x) < 1.0 for x in valid_ttc),
                    "ticks_ttc_lt_3s": sum(float(x) < 3.0 for x in valid_ttc),
                    "minimum_ttc_s": min(valid_ttc) if valid_ttc else None,
                    "minimum_center_distance_m": min(
                        (r["min_center_distance_m"] for r in window if r.get("min_center_distance_m") is not None),
                        default=None,
                    ),
                    "critical_unobserved_ticks": sum(r["critical_unobserved"] for r in window),
                    "conflict_candidate_ticks": sum(r["conflict_count"] > 0 for r in window),
                    "conflict_candidate_unobserved_ticks": sum(
                        r["conflict_count"] > r["conflict_observed_count"] for r in window
                    ),
                }
        collisions = rec.get("collision_evidence") or []
        terminal = rec.get("terminal_info") or {}
        ep_summary = {
            "seed": rec.get("seed"),
            "outcome": outcome,
            "raw_steps": rec.get("raw_steps"),
            "decisions": rec.get("decisions"),
            "return_base": rec.get("return_base"),
            "return_policy": rec.get("return_policy"),
            "mean_actual_speed_mps": rec.get("mean_actual_speed_mps"),
            "stopped_fraction": rec.get("stopped_fraction_of_speed_samples"),
            "mean_target_speed_mps": avg(target_speeds),
            "terminal_target_speed_mps": terminal.get("target_speed"),
            "mean_speed_at_decisions_mps": avg([p[1] for p in speed_pairs]),
            "mean_target_minus_actual_mps": avg([p[0] - p[1] for p in speed_pairs]),
            "mean_action_0": avg(action0),
            "mean_action_1": avg(action1),
            "fraction_action0_abs_ge_0p95": sum(abs(x) >= 0.95 for x in action0) / len(action0) if action0 else None,
            "min_cv_obb_ttc_s_episode": rec.get("min_cv_obb_ttc_s"),
            "cv_ttc_below_3s_ticks": rec.get("cv_ttc_below_3s_ticks"),
            "risk_evaluable_ticks": rec.get("risk_evaluable_ticks"),
            "critical_unobserved_ticks": rec.get("critical_unobserved_ticks"),
            "collision_time_sim_s": collision_time,
            "collision_evidence": collisions,
            "terminal_collision_flags": {k: terminal.get(k) for k in ("collision", "geometric_collision", "is_success", "max_time")},
            "collision_pre_windows": pre_windows,
            "raw_collision_event_rows": len(event_rows.get(episode, [])),
        }
        episode_summaries[episode] = ep_summary
        outcome_groups[outcome].append(ep_summary)

    compact_groups = {}
    for outcome, rows in sorted(outcome_groups.items()):
        fields = [
            "raw_steps", "return_base", "return_policy", "mean_actual_speed_mps",
            "stopped_fraction", "mean_target_speed_mps", "terminal_target_speed_mps",
            "mean_speed_at_decisions_mps", "mean_target_minus_actual_mps", "mean_action_0",
            "fraction_action0_abs_ge_0p95", "min_cv_obb_ttc_s_episode",
            "cv_ttc_below_3s_ticks", "risk_evaluable_ticks", "critical_unobserved_ticks",
        ]
        group = {"n": len(rows)}
        for field in fields:
            values = [r.get(field) for r in rows if r.get(field) is not None]
            group[field + "_mean_equal_episode"] = avg(values)
        group["mean_speed_pooled_from_episode_denominators"] = (
            sum(float(r.get("mean_actual_speed_mps") or 0) * int(r.get("raw_steps") or 0) for r in rows)
            / max(1, sum(int(r.get("raw_steps") or 0) for r in rows))
        )
        group["mean_collision_window_min_ttc_2s"] = avg([
            r["collision_pre_windows"].get("2s", {}).get("minimum_ttc_s") for r in rows
        ])
        group["collision_window_episodes_with_ttc_lt_3s"] = sum(
            r["collision_pre_windows"].get("2s", {}).get("ticks_ttc_lt_3s", 0) > 0 for r in rows
        )
        compact_groups[outcome] = group

    return {
        "method": method,
        "episode_count": len(episodes),
        "episodes_by_seed": {str(row.get("seed")): episode_summaries[int(row["episode"])] for row in episodes},
        "outcome_groups": compact_groups,
        "data_quality": dict(qc),
        "episode_outcomes": {str(row.get("seed")): episode_outcome(row) for row in episodes},
        "phase_files": {"episodes": str(diag / "episodes.jsonl"), "raw_steps": str(diag / "raw_steps.jsonl.gz"), "decisions": str(diag / "decisions.jsonl.gz")},
    }


def main():
    all_methods = {}
    for method, (dirname, phase) in METHODS.items():
        all_methods[method] = load_method(method, dirname, phase)
    a, b = all_methods["sac_mlp"], all_methods["sac_mlp_d1_st_rt"]
    cross = Counter()
    common = sorted(set(a["episode_outcomes"]) & set(b["episode_outcomes"]), key=int)
    for seed in common:
        cross[(a["episode_outcomes"][seed], b["episode_outcomes"][seed])] += 1
    pairs = []
    desired = [
        ("timeout", "success"), ("collision", "success"),
        ("success", "collision"), ("success", "success"),
    ]
    for oa, ob in desired:
        matches = [s for s in common if a["episode_outcomes"][s] == oa and b["episode_outcomes"][s] == ob]
        if matches:
            seed = matches[0]
            pairs.append({
                "selection_rule": f"smallest common eval seed with SAC={oa}, ST-RT={ob}",
                "seed": int(seed),
                "sac_mlp": a["episodes_by_seed"][seed],
                "sac_mlp_d1_st_rt": b["episodes_by_seed"][seed],
            })
    result = {
        "run_root": str(RUN),
        "scope": "read-only evaluation episodes, decisions, and compressed raw-step diagnostics; no SUMO/model execution",
        "pair_count": len(common),
        "paired_outcome_crosstab": {f"{x[0]}__{x[1]}": n for x, n in sorted(cross.items())},
        "representative_pairs": pairs,
        "methods": all_methods,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "pair_count": result["pair_count"],
        "paired_outcome_crosstab": result["paired_outcome_crosstab"],
        "representative_pairs": [
            {"seed": p["seed"], "selection_rule": p["selection_rule"],
             "sac": {k: p["sac_mlp"].get(k) for k in ("outcome", "raw_steps", "return_policy", "mean_actual_speed_mps", "mean_target_speed_mps", "mean_action_0", "min_cv_obb_ttc_s_episode", "collision_evidence", "collision_pre_windows")},
             "st_rt": {k: p["sac_mlp_d1_st_rt"].get(k) for k in ("outcome", "raw_steps", "return_policy", "mean_actual_speed_mps", "mean_target_speed_mps", "mean_action_0", "min_cv_obb_ttc_s_episode", "collision_evidence", "collision_pre_windows")}}
            for p in pairs
        ],
        "method_groups": {m: v["outcome_groups"] for m, v in all_methods.items()},
        "quality": {m: v["data_quality"] for m, v in all_methods.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
