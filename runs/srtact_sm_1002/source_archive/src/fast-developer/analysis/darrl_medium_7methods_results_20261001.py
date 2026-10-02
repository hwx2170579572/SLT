"""Read-only aggregation for the DARRL medium seven-method run.

Reads existing run artifacts only; it never imports the simulator or model code.
Run with the workspace Python, passing the run root and output JSON path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pvariance, stdev


METHODS = [
    "mst_slt",
    "sac_mlp_d1_st",
    "sac_mlp_d1_st_rt",
    "sac_mlp_d1_st_rt_topo",
    "sac_mlp_d1_st_rt_topo_routeaware_v1",
    "sac_mlp_d1_st_rt_3slot",
    "sac_mlp_d1_st_rt_topo_3slot",
]
PAIRS = [
    ("sac_mlp_d1_st", "sac_mlp_d1_st_rt"),
    ("sac_mlp_d1_st_rt", "sac_mlp_d1_st_rt_topo"),
    ("sac_mlp_d1_st_rt_topo", "sac_mlp_d1_st_rt_topo_routeaware_v1"),
    ("sac_mlp_d1_st_rt", "sac_mlp_d1_st_rt_3slot"),
    ("sac_mlp_d1_st_rt_topo", "sac_mlp_d1_st_rt_topo_3slot"),
    ("sac_mlp_d1_st_rt_3slot", "sac_mlp_d1_st_rt_topo_3slot"),
]
OUTCOMES = ("success", "collision", "timeout", "off_route")


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_outcome(row):
    flags = {name: bool(row.get(name, False)) for name in OUTCOMES}
    active = [name for name, value in flags.items() if value]
    return active[0] if len(active) == 1 else ("none" if not active else "multiple:" + "+".join(active))


def find_eval_diag(run_dir: Path):
    candidates = [run_dir / "diagnostics" / "eval", run_dir / "diagnostics" / "eval_worker_00"]
    for candidate in candidates:
        path = candidate / "episodes.jsonl"
        if path.is_file():
            return path
    return None


def read_episode_diag(path: Path | None):
    rows = {}
    if not path:
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            traffic = (row.get("reset_info") or {}).get("random_traffic") or {}
            logical_seed = traffic.get("logical_episode_seed")
            if logical_seed is None:
                continue
            rows[int(logical_seed)] = row
    return rows


def sum_or_none(values):
    vals = [float(x) for x in values if x is not None]
    return sum(vals) if vals else None


def summarize_behavior(diag_rows):
    rows = list(diag_rows.values())
    if not rows:
        return None
    speed_samples = sum(int(r.get("speed_samples") or 0) for r in rows)
    stopped_ticks = sum(int(r.get("stopped_ticks") or 0) for r in rows)
    risk_ticks = sum(int(r.get("risk_evaluable_ticks") or 0) for r in rows)
    low_ttc = sum(int(r.get("risk_evaluable_low_ttc_ticks") or 0) for r in rows)
    covered_ticks = sum(int(r.get("risk_and_observation_covered_ticks") or 0) for r in rows)
    critical_unobserved = sum(int(r.get("covered_critical_unobserved_ticks") or 0) for r in rows)
    speed_sum = sum(float(r.get("speed_sum_mps") or 0.0) for r in rows)
    action_dims = []
    max_dims = max((len(r.get("action_statistics") or []) for r in rows), default=0)
    for dim in range(max_dims):
        action_rows = [r["action_statistics"][dim] for r in rows if len(r.get("action_statistics") or []) > dim]
        n = sum(int(a.get("n") or 0) for a in action_rows)
        abs_count = sum(int(a.get("abs_ge_0p95_count") or 0) for a in action_rows)
        action_sum = sum(float(a.get("sum") or 0.0) for a in action_rows)
        action_dims.append({
            "dimension": dim,
            "samples": n,
            "mean": action_sum / n if n else None,
            "near_unit_bound_fraction": abs_count / n if n else None,
        })
    collision_rows = [r for r in rows if r.get("outcome") == "collision"]
    terminal_locations = Counter()
    raw_collision_episodes = 0
    geometric_collision_episodes = 0
    for r in collision_rows:
        snapshot = r.get("last_snapshot") or {}
        ego = snapshot.get("ego") or {}
        road = ego.get("road_id")
        lane = ego.get("lane_id")
        terminal_locations[f"{road or 'unknown'} | {lane or 'unknown'}"] += 1
        events = snapshot.get("events") or {}
        if events.get("raw_sumo_collision") is True:
            raw_collision_episodes += 1
        if events.get("geometric_collision") is True:
            geometric_collision_episodes += 1
    return {
        "episodes_with_diagnostics": len(rows),
        "speed": {
            "sample_count": speed_samples,
            "mean_actual_speed_mps_raw_tick_weighted": speed_sum / speed_samples if speed_samples else None,
            "stopped_ticks": stopped_ticks,
            "stopped_fraction_of_speed_samples": stopped_ticks / speed_samples if speed_samples else None,
        },
        "risk": {
            "evaluable_ticks": risk_ticks,
            "low_ttc_ticks": low_ttc,
            "low_ttc_fraction_of_evaluable_ticks": low_ttc / risk_ticks if risk_ticks else None,
            "risk_and_observation_covered_ticks": covered_ticks,
            "covered_critical_unobserved_ticks": critical_unobserved,
            "critical_unobserved_fraction_of_covered_ticks": critical_unobserved / covered_ticks if covered_ticks else None,
        },
        "actions": action_dims,
        "collision_terminal_location_counts_from_last_snapshot": dict(terminal_locations),
        "terminal_collision_episodes_with_raw_sumo_collision_flag": raw_collision_episodes,
        "terminal_collision_episodes_with_geometric_collision_flag": geometric_collision_episodes,
        "limitation": "Raw SUMO collision flag and geometric collision flag are endpoint diagnostics; they are not interchangeable with the mutually exclusive terminal collision outcome.",
    }


def load_method(run_root: Path, method: str, suite_method_status):
    folder = f"{method}__intersection_random_darrl_medium_v1_depart1p0"
    run_dir = run_root / folder
    arg_path = run_dir / "arguments.json"
    manifest_path = run_dir / "experiment_manifest.json"
    completion_path = run_dir / "training_complete.json"
    progress_path = run_dir / "progress.json"
    train_diag_path = run_dir / "training_diagnostics.json"
    status_path = run_dir / "status.json"
    evaluation_path = run_dir / "evaluation_results.json"
    internal_eval_path = run_dir / "final_evaluation.json"
    args = read_json(arg_path) if arg_path.is_file() else None
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    complete = read_json(completion_path) if completion_path.is_file() else None
    progress = read_json(progress_path) if progress_path.is_file() else None
    train_diag = read_json(train_diag_path) if train_diag_path.is_file() else None
    status = read_json(status_path) if status_path.is_file() else None
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else None
    checkpoint_path = run_dir / "final_model.zip"
    identity = evaluation.get("identity", {}) if evaluation else {}
    summary = evaluation.get("summary", {}) if evaluation else {}
    records = evaluation.get("episode_records", []) if evaluation else []
    diag_path = find_eval_diag(run_dir)
    diag_by_seed = read_episode_diag(diag_path)
    outcomes = Counter(unique_outcome(row) for row in records)
    flags = {name: sum(bool(row.get(name)) for row in records) for name in OUTCOMES}
    seeds = [int(row["seed"]) for row in records if row.get("seed") is not None]
    returns = [float(row["episode_return"]) for row in records if row.get("episode_return") is not None]
    raw_steps = [int(row["raw_steps"]) for row in records if row.get("raw_steps") is not None]
    decision_steps = [int(row["decision_steps"]) for row in records if row.get("decision_steps") is not None]
    success_records = [row for row in records if row.get("success") is True]
    success_times = [float(row["completion_time_seconds"]) for row in success_records if row.get("completion_time_seconds") is not None]
    checkpoint_hash = None
    if checkpoint_path.is_file():
        checkpoint_hash = sha256(checkpoint_path)
    diag_hashes = {
        seed: (row.get("reset_info", {}).get("random_traffic", {}).get("episode_schedule", {}) or {}).get("sha256")
        for seed, row in diag_by_seed.items()
    }
    expected = {
        "suite_status": (suite_method_status or {}).get("status"),
        "suite_exit_code": (suite_method_status or {}).get("exit_code"),
        "train_raw_steps": (complete or {}).get("raw_steps") or (train_diag or {}).get("raw_steps"),
        "train_updates": (complete or {}).get("updates") or (train_diag or {}).get("learner_updates"),
        "train_wall_seconds": (complete or {}).get("wall_seconds") or ((read_json(run_dir / "performance_profile.json") if (run_dir / "performance_profile.json").is_file() else {}).get("end_to_end_training_wall_seconds")),
        "completion_present": complete is not None or bool(train_diag and train_diag.get("raw_steps") == 100000),
        "training_status": (status or {}).get("status"),
        "training_complete": bool(((complete or {}).get("raw_steps") == 100000) or ((train_diag or {}).get("raw_steps") == 100000)) and not (complete or {}).get("smoke", False) and not (status or {}).get("smoke", False),
        "arguments": args,
        "experiment_manifest": manifest,
        "progress": progress,
        "completion": complete,
        "evaluation_identity": identity,
        "external_evaluation_summary": summary,
        "external_eval_episode_count": len(records),
        "logical_seed_min": min(seeds) if seeds else None,
        "logical_seed_max": max(seeds) if seeds else None,
        "logical_seed_unique_count": len(set(seeds)),
        "seed_sequence_is_10000_to_10099": sorted(seeds) == list(range(10000, 10100)),
        "terminal_flags_raw_counts": flags,
        "exclusive_outcome_counts": {k: outcomes[k] for k in ("success", "collision", "timeout", "off_route")},
        "outcome_flags_sum_per_episode_histogram": dict(Counter(sum(bool(r.get(k)) for k in OUTCOMES) for r in records)),
        "return": {
            "mean": mean(returns) if returns else None,
            "population_variance": pvariance(returns) if len(returns) >= 1 else None,
            "population_sd": math.sqrt(pvariance(returns)) if returns else None,
        },
        "episode_steps": {
            "raw_mean_all": mean(raw_steps) if raw_steps else None,
            "raw_median_all": sorted(raw_steps)[len(raw_steps)//2] if raw_steps else None,
            "decision_mean_all": mean(decision_steps) if decision_steps else None,
            "decision_mean_success_only": mean([int(r["decision_steps"]) for r in success_records if r.get("decision_steps") is not None]) if any(r.get("decision_steps") is not None for r in success_records) else None,
            "raw_mean_success_only": mean([int(r["raw_steps"]) for r in success_records if r.get("raw_steps") is not None]) if any(r.get("raw_steps") is not None for r in success_records) else None,
            "success_completion_time_mean_seconds": mean(success_times) if success_times else None,
            "success_completion_time_median_seconds": sorted(success_times)[len(success_times)//2] if success_times else None,
        },
        "behavior": summarize_behavior(diag_by_seed),
        "diagnostic_episode_count": len(diag_by_seed),
        "diagnostic_seed_set_matches_eval": set(diag_by_seed) == set(seeds),
        "traffic_schedule_hash_count": len(set(h for h in diag_hashes.values() if h)),
        "traffic_schedule_hashes_by_logical_seed": {str(k): v for k, v in sorted(diag_hashes.items())},
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path.is_file() else None,
            "sha256_file": checkpoint_hash,
            "sha256_evaluation_identity": identity.get("checkpoint_sha256"),
            "identity_matches_final_model": bool(checkpoint_hash and checkpoint_hash == identity.get("checkpoint_sha256")),
            "identity_checkpoint_path": identity.get("checkpoint"),
        },
        "source_files": {
            "arguments": str(arg_path),
            "experiment_manifest": str(manifest_path),
            "training_complete": str(completion_path) if completion_path.is_file() else None,
            "evaluation_results": str(evaluation_path) if evaluation_path.is_file() else None,
            "eval_diagnostics_episodes": str(diag_path) if diag_path else None,
            "internal_mst_final_evaluation": str(internal_eval_path) if method == "mst_slt" and internal_eval_path.is_file() else None,
        },
        "source_sha256": {
            "arguments": sha256(arg_path) if arg_path.is_file() else None,
            "experiment_manifest": sha256(manifest_path) if manifest_path.is_file() else None,
            "training_complete": sha256(completion_path) if completion_path.is_file() else None,
            "evaluation_results": sha256(evaluation_path) if evaluation_path.is_file() else None,
            "eval_diagnostics_episodes": sha256(diag_path) if diag_path else None,
        },
        "internal_mst_eval_summary": read_json(internal_eval_path).get("summary") if method == "mst_slt" and internal_eval_path.is_file() else None,
    }
    return expected


def pair_methods(results, left, right):
    a = results.get(left)
    b = results.get(right)
    if not a or not b or not a.get("_records") or not b.get("_records"):
        return {"reference": left, "candidate": right, "status": "not both evaluated", "matrix": None}
    left_records = {int(r["seed"]): r for r in a["_records"]}
    right_records = {int(r["seed"]): r for r in b["_records"]}
    common = sorted(set(left_records) & set(right_records))
    matrix = {x: {y: 0 for y in OUTCOMES} for x in OUTCOMES}
    hash_matches = 0
    hash_checked = 0
    wins = losses = ties = 0
    for seed in common:
        x = unique_outcome(left_records[seed])
        y = unique_outcome(right_records[seed])
        if x not in OUTCOMES or y not in OUTCOMES:
            continue
        matrix[x][y] += 1
        if x == "success" and y != "success": losses += 1
        elif x != "success" and y == "success": wins += 1
        elif x == y: ties += 1
        if seed in a["_diag_hashes"] and seed in b["_diag_hashes"]:
            hash_checked += 1
            if a["_diag_hashes"][seed] == b["_diag_hashes"][seed]:
                hash_matches += 1
    return {
        "reference": left,
        "candidate": right,
        "pair_key": "logical_validation_seed",
        "common_pairs": len(common),
        "all_common_pairs_have_unique_four_way_outcomes": sum(sum(row.values()) for row in matrix.values()) == len(common),
        "traffic_schedule_hash_checks": hash_checked,
        "traffic_schedule_hash_matches": hash_matches,
        "traffic_schedule_hash_match_rate": hash_matches / hash_checked if hash_checked else None,
        "matrix_reference_rows_candidate_columns": matrix,
        "success_wins_candidate_vs_reference": wins,
        "success_losses_candidate_vs_reference": losses,
        "same_outcome_ties": ties,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suite_status = read_json(args.run_root / "launcher_status.json")
    suite_manifest = read_json(args.run_root / "suite_manifest.json")
    result_rows = {}
    paired_data = {}
    for method in METHODS:
        row = load_method(args.run_root, method, suite_status.get("methods", {}).get(method))
        if (args.run_root / f"{method}__intersection_random_darrl_medium_v1_depart1p0" / "evaluation_results.json").is_file():
            raw_eval = read_json(args.run_root / f"{method}__intersection_random_darrl_medium_v1_depart1p0" / "evaluation_results.json")
            row["_records"] = raw_eval.get("episode_records", [])
            dpath = find_eval_diag(args.run_root / f"{method}__intersection_random_darrl_medium_v1_depart1p0")
            diag = read_episode_diag(dpath)
            row["_diag_hashes"] = {seed: ((r.get("reset_info") or {}).get("random_traffic") or {}).get("episode_schedule", {}).get("sha256") for seed, r in diag.items()}
        result_rows[method] = row
    for left, right in PAIRS:
        paired_data[f"{left}__to__{right}"] = pair_methods(result_rows, left, right)
    # Factorial outcome table for the four STRT variants; this remains one fixed training seed.
    factor_methods = {
        "00_STRT": "sac_mlp_d1_st_rt",
        "10_Topo": "sac_mlp_d1_st_rt_topo",
        "01_STRT_3slot": "sac_mlp_d1_st_rt_3slot",
        "11_Topo_3slot": "sac_mlp_d1_st_rt_topo_3slot",
    }
    factors = {}
    for label, method in factor_methods.items():
        row = result_rows[method]
        factors[label] = row["terminal_flags_raw_counts"] if row.get("external_eval_episode_count") else None
    for method in result_rows.values():
        method.pop("_records", None)
        method.pop("_diag_hashes", None)
    output = {
        "schema_version": 1,
        "run_root": str(args.run_root.resolve()),
        "launcher_status": suite_status.get("status"),
        "suite_methods": list((suite_status.get("methods") or {}).keys()),
        "suite_manifest": {
            "scenario": suite_manifest.get("scenario"),
            "scenario_revision": (suite_manifest.get("traffic_config") or {}).get("configuration_revision"),
            "traffic_protocol": suite_manifest.get("traffic_protocol"),
            "traffic_config": suite_manifest.get("traffic_config"),
            "seed": suite_manifest.get("training_seed"),
            "raw_step_budget": suite_manifest.get("raw_step_budget"),
            "eval_episodes": suite_manifest.get("eval_episodes"),
            "eval_traffic_split": suite_manifest.get("eval_traffic_split"),
            "smoke": suite_manifest.get("smoke"),
        },
        "methods": result_rows,
        "paired_comparisons": paired_data,
        "strt_four_cell_terminal_flag_counts": factors,
        "interpretation_limits": [
            "Evaluation records are fixed final checkpoints from one training seed (seed 0); evaluation-episode pairing does not estimate training-seed variability.",
            "A pair is considered traffic matched only when per-logical-seed seeded schedule SHA-256 values match; episode_records alone label traffic_variant generically.",
            "Terminal outcome categories are computed from the stored mutually exclusive success/collision/timeout/off_route flags; any zero/multiple flag rows are reported separately.",
            "Raw SUMO collision and geometric-collision endpoint flags are summarized separately from terminal collision labels.",
            "The final Topo+3slot method may still be running; missing final evaluation is emitted as null/absent, never inferred from partial progress.",
            "Return variance uses the population divisor N for the finite 100-episode evaluation sample.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
