"""Offline, reproducible comparison of the DARRL-r2 medium SAC-MLP and MST baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, pvariance

OUTCOMES = ("success", "collision", "timeout", "off_route")


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_diag(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            traffic = (row.get("reset_info") or {}).get("random_traffic") or {}
            logical_seed = traffic.get("logical_episode_seed")
            if logical_seed is not None:
                result[int(logical_seed)] = row
    return result


def terminal(row):
    active = [k for k in OUTCOMES if bool(row.get(k, False))]
    if len(active) == 1:
        return active[0]
    return "none" if not active else "multiple:" + "+".join(active)


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def exact_mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def behavior_summary(diag):
    rows = list(diag.values())
    speed_n = sum(int(r.get("speed_samples") or 0) for r in rows)
    stopped = sum(int(r.get("stopped_ticks") or 0) for r in rows)
    risk_n = sum(int(r.get("risk_evaluable_ticks") or 0) for r in rows)
    low_ttc = sum(int(r.get("risk_evaluable_low_ttc_ticks") or 0) for r in rows)
    covered = sum(int(r.get("risk_and_observation_covered_ticks") or 0) for r in rows)
    critical = sum(int(r.get("covered_critical_unobserved_ticks") or 0) for r in rows)
    speed_sum = sum(float(r.get("speed_sum_mps") or 0) for r in rows)
    action_dims = []
    n_dims = max((len(r.get("action_statistics") or []) for r in rows), default=0)
    for dim in range(n_dims):
        vals = [r["action_statistics"][dim] for r in rows if len(r.get("action_statistics") or []) > dim]
        n = sum(int(v.get("n") or 0) for v in vals)
        near = sum(int(v.get("abs_ge_0p95_count") or 0) for v in vals)
        action_sum = sum(float(v.get("sum") or 0) for v in vals)
        action_dims.append({"dimension": dim, "samples": n, "mean": action_sum / n if n else None,
                            "near_unit_bound_fraction": near / n if n else None})
    collision_rows = [r for r in rows if r.get("outcome") == "collision"]
    terminal_locations = Counter()
    raw_sumo_terminal = 0
    geometric_terminal = 0
    raw_collision_event_items = 0
    for r in collision_rows:
        snap = r.get("last_snapshot") or {}
        ego = snap.get("ego") or {}
        terminal_locations[f"{ego.get('road_id') or 'unknown'} | {ego.get('lane_id') or 'unknown'}"] += 1
        events = snap.get("events") or {}
        raw_sumo_terminal += events.get("raw_sumo_collision") is True
        geometric_terminal += events.get("geometric_collision") is True
        evidence = r.get("collision_evidence") or {}
        if isinstance(evidence, dict):
            raw_collision_event_items += len(evidence.get("collision_events") or [])
        elif isinstance(evidence, list):
            raw_collision_event_items += len(evidence)
    return {
        "episodes_with_diagnostics": len(rows),
        "speed_samples": speed_n,
        "mean_actual_speed_mps_raw_tick_weighted": speed_sum / speed_n if speed_n else None,
        "stopped_ticks": stopped,
        "stopped_fraction_of_speed_samples": stopped / speed_n if speed_n else None,
        "risk_evaluable_ticks": risk_n,
        "low_ttc_ticks": low_ttc,
        "low_ttc_fraction_of_evaluable_ticks": low_ttc / risk_n if risk_n else None,
        "risk_and_observation_covered_ticks": covered,
        "covered_critical_unobserved_ticks": critical,
        "critical_unobserved_fraction_of_covered_ticks": critical / covered if covered else None,
        "actions": action_dims,
        "terminal_collision_episodes_raw_sumo_collision_flag": raw_sumo_terminal,
        "terminal_collision_episodes_geometric_collision_flag": geometric_terminal,
        "collision_evidence_items_at_terminal": raw_collision_event_items,
        "terminal_collision_snapshot_location_counts": dict(terminal_locations),
        "location_caveat": "Location is the episode final diagnostic snapshot, not necessarily the exact impact coordinate; geometric detections can have no SUMO collision event id.",
    }


def summarize_method(label, method_dir):
    method_dir = Path(method_dir)
    args_path = method_dir / "arguments.json"
    manifest_path = method_dir / "experiment_manifest.json"
    complete_path = method_dir / "training_complete.json"
    training_diag_path = method_dir / "training_diagnostics.json"
    performance_path = method_dir / "performance_profile.json"
    eval_path = method_dir / "evaluation_results.json"
    args = read_json(args_path)
    manifest = read_json(manifest_path)
    complete = read_json(complete_path) if complete_path.is_file() else None
    training_diag = read_json(training_diag_path) if training_diag_path.is_file() else None
    performance = read_json(performance_path) if performance_path.is_file() else None
    status_path = method_dir / "status.json"
    status = read_json(status_path) if status_path.is_file() else None
    evaluation = read_json(eval_path)
    identity = evaluation.get("identity", {})
    records = evaluation.get("episode_records", [])
    rows = {int(r["seed"]): r for r in records}
    diag_path = method_dir / "diagnostics" / "eval_worker_00" / "episodes.jsonl"
    if not diag_path.is_file():
        diag_path = method_dir / "diagnostics" / "eval" / "episodes.jsonl"
    diag = read_diag(diag_path) if diag_path.is_file() else {}
    hashes = {seed: (((r.get("reset_info") or {}).get("random_traffic") or {}).get("episode_schedule") or {}).get("sha256") for seed, r in diag.items()}
    runtime_seed_map = {
        seed: {
            "logical_episode_seed": seed,
            "sumo_seed": ((r.get("reset_info") or {}).get("random_traffic") or {}).get("sumo_seed"),
            "split": ((r.get("reset_info") or {}).get("random_traffic") or {}).get("split"),
            "protocol": ((r.get("reset_info") or {}).get("random_traffic") or {}).get("protocol"),
        }
        for seed, r in diag.items()
    }
    outcomes = Counter(terminal(r) for r in records)
    flag_counts = {k: sum(bool(r.get(k)) for r in records) for k in OUTCOMES}
    values = [float(r["episode_return"]) for r in records]
    raw = [int(r["raw_steps"]) for r in records if r.get("raw_steps") is not None]
    decisions = [int(r["decision_steps"]) for r in records if r.get("decision_steps") is not None]
    success = [r for r in records if r.get("success") is True]
    succ_raw = [int(r["raw_steps"]) for r in success if r.get("raw_steps") is not None]
    succ_decisions = [int(r["decision_steps"]) for r in success if r.get("decision_steps") is not None]
    succ_time = [float(r["completion_time_seconds"]) for r in success if r.get("completion_time_seconds") is not None]
    final_model = method_dir / "final_model.zip"
    model_sha = sha256(final_model) if final_model.is_file() else None
    train_raw = (complete or {}).get("raw_steps") or (training_diag or {}).get("raw_steps")
    train_updates = (complete or {}).get("updates") or (training_diag or {}).get("learner_updates")
    training = {
        "raw_steps": train_raw,
        "updates": train_updates,
        "wall_seconds": (complete or {}).get("wall_seconds") or (performance or {}).get("end_to_end_training_wall_seconds"),
        "seed": args.get("seed") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("seed"),
        "device": args.get("device") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("device"),
        "smoke": (complete or {}).get("smoke", (status or {}).get("smoke")),
        "status": (status or {}).get("status"),
        "training_complete": train_raw == 100000 and not (complete or {}).get("smoke", False) and not (status or {}).get("smoke", False),
        "matched_hyperparameters": {
            "raw_budget": args.get("raw_budget") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("max_steps"),
            "action_repeat": args.get("action_repeat") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("action_repeat"),
            "batch_size": args.get("batch_size") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("batch_size"),
            "learning_rate": args.get("learning_rate") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("learning_rate"),
            "learning_starts_raw_steps": args.get("learning_starts_raw_steps") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("learning_starts"),
            "buffer_size": args.get("buffer_size") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("buffer_size"),
            "discount": args.get("discount") if label == "SAC+MLP" else args.get("requested_raw_steps", {}).get("discount"),
        },
    }
    return {
        "label": label,
        "method_directory": str(method_dir),
        "training": training,
        "evaluation": {
            "source_file": str(eval_path),
            "source_sha256": sha256(eval_path),
            "checkpoint_path": identity.get("checkpoint"),
            "checkpoint_sha256_identity": identity.get("checkpoint_sha256"),
            "final_model_sha256": model_sha,
            "checkpoint_hash_matches_final_model": model_sha == identity.get("checkpoint_sha256"),
            "scenario": identity.get("scenario"),
            "depart_scale": identity.get("depart_scale"),
            "traffic_split": identity.get("eval_traffic_split"),
            "traffic_protocol": manifest.get("traffic_protocol"),
            "training_traffic_split": manifest.get("training_traffic_split"),
            "configuration_revision": (manifest.get("traffic_config") or {}).get("configuration_revision"),
            "terminal_outcome_protocol": (manifest.get("traffic_config") or {}).get("terminal_outcome_protocol"),
            "traffic_probability_per_step": next((x.get("probability_per_step") for x in (manifest.get("traffic_config") or {}).get("flows", []) if x.get("probability_per_step") is not None), None),
            "episodes": identity.get("episodes"),
            "workers": identity.get("workers"),
            "smoke": identity.get("smoke"),
            "episode_count": len(records),
            "logical_seed_min": min(rows) if rows else None,
            "logical_seed_max": max(rows) if rows else None,
            "logical_seed_set_is_10000_10099": sorted(rows) == list(range(10000, 10100)),
            "exclusive_terminal_counts": {k: outcomes[k] for k in OUTCOMES},
            "raw_terminal_flag_counts": flag_counts,
            "success_rate": flag_counts["success"] / len(records) if records else None,
            "success_rate_wilson_95pct": wilson(flag_counts["success"], len(records)),
            "terminal_flag_count_histogram": dict(Counter(sum(bool(r.get(k)) for k in OUTCOMES) for r in records)),
            "mean_return": mean(values) if values else None,
            "return_population_variance": pvariance(values) if values else None,
            "return_population_sd": math.sqrt(pvariance(values)) if values else None,
            "mean_raw_steps_all": mean(raw) if raw else None,
            "median_raw_steps_all": median(raw) if raw else None,
            "mean_decision_steps_all": mean(decisions) if decisions else None,
            "mean_raw_steps_success_only": mean(succ_raw) if succ_raw else None,
            "mean_decision_steps_success_only": mean(succ_decisions) if succ_decisions else None,
            "mean_completion_time_seconds_success_only": mean(succ_time) if succ_time else None,
            "median_completion_time_seconds_success_only": median(succ_time) if succ_time else None,
            "eval_diagnostics_file": str(diag_path) if diag_path.is_file() else None,
            "diagnostic_episode_count": len(diag),
            "diag_seed_set_matches_eval_seed_set": set(diag) == set(rows),
            "distinct_schedule_hashes": len({h for h in hashes.values() if h}),
            "schedule_sha256_by_logical_seed": {str(k): v for k, v in sorted(hashes.items())},
            "behavior": behavior_summary(diag),
        },
        "arguments_file": str(args_path),
        "arguments_sha256": sha256(args_path),
        "experiment_manifest_file": str(manifest_path),
        "experiment_manifest_sha256": sha256(manifest_path),
        "training_diagnostics_file": str(training_diag_path) if training_diag_path.is_file() else None,
        "performance_profile_file": str(performance_path) if performance_path.is_file() else None,
        "raw_records_by_seed": {str(k): {"outcome": terminal(v), "return": v.get("episode_return"), "raw_steps": v.get("raw_steps"), "decision_steps": v.get("decision_steps"), "completion_time_seconds": v.get("completion_time_seconds")} for k, v in sorted(rows.items())},
        "private_schedule_hashes": hashes,
        "private_runtime_seed_map": runtime_seed_map,
        "private_records": rows,
    }


def paired(sac, mst, bootstrap_seed=20261001, bootstrap_reps=20000):
    a = sac["private_records"]
    b = mst["private_records"]
    common = sorted(set(a) & set(b))
    matrix = {x: {y: 0 for y in OUTCOMES} for x in OUTCOMES}
    schedule_match = 0
    hash_checked = 0
    diffs = []
    sac_wins_mst_success = 0
    mst_wins_sac_success = 0
    same_success_class = 0
    same_terminal = 0
    transition_seeds = {x: {y: [] for y in OUTCOMES} for x in OUTCOMES}
    actual_seed_matches = 0
    actual_seed_checked = 0
    for seed in common:
        x = terminal(a[seed])
        y = terminal(b[seed])
        if x in OUTCOMES and y in OUTCOMES:
            matrix[x][y] += 1
            seed_meta = sac["private_runtime_seed_map"].get(seed, {})
            mst_seed_meta = mst["private_runtime_seed_map"].get(seed, {})
            h1 = sac["private_schedule_hashes"].get(seed)
            h2 = mst["private_schedule_hashes"].get(seed)
            transition_seeds[x][y].append({
                "logical_episode_seed": seed,
                "sac_actual_sumo_seed": seed_meta.get("sumo_seed"),
                "mst_actual_sumo_seed": mst_seed_meta.get("sumo_seed"),
                "actual_sumo_seed_match": seed_meta.get("sumo_seed") == mst_seed_meta.get("sumo_seed"),
                "traffic_schedule_sha256_match": h1 == h2,
                "traffic_schedule_sha256": h1 if h1 == h2 else {"sac": h1, "mst": h2},
            })
        sx, sy = x == "success", y == "success"
        diffs.append(int(sy) - int(sx))
        if sx and not sy:
            sac_wins_mst_success += 1
        elif sy and not sx:
            mst_wins_sac_success += 1
        if sx == sy:
            same_success_class += 1
        if x == y:
            same_terminal += 1
        h1, h2 = sac["private_schedule_hashes"].get(seed), mst["private_schedule_hashes"].get(seed)
        if h1 and h2:
            hash_checked += 1
            schedule_match += h1 == h2
        s1 = sac["private_runtime_seed_map"].get(seed, {}).get("sumo_seed")
        s2 = mst["private_runtime_seed_map"].get(seed, {}).get("sumo_seed")
        if s1 is not None and s2 is not None:
            actual_seed_checked += 1
            actual_seed_matches += s1 == s2
    rng = random.Random(bootstrap_seed)
    boot = []
    if diffs:
        for _ in range(bootstrap_reps):
            boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
        boot.sort()
    delta = mean(diffs) if diffs else None
    return {
        "pair_key": "evaluation logical_episode_seed plus identical episode_schedule.sha256",
        "reference_rows_candidate_columns": "SAC+MLP rows -> MST+SLT columns",
        "n_common_seeds": len(common),
        "common_logical_seed_range": [min(common), max(common)] if common else None,
        "same_seeded_traffic_schedule_count": schedule_match,
        "traffic_schedule_hash_checks": hash_checked,
        "all_schedules_match": bool(common) and hash_checked == len(common) and schedule_match == len(common),
        "actual_sumo_seed_matches": actual_seed_matches,
        "actual_sumo_seed_checks": actual_seed_checked,
        "logical_seed_to_actual_sumo_seed_rule": "validation split uses the configured validation offset; both evaluations are compared on logged logical seed and confirmed actual SUMO seed plus per-episode schedule SHA-256.",
        "terminal_outcome_transition_matrix": matrix,
        "transition_seed_evidence": transition_seeds,
        "sac_success_only_count": sac_wins_mst_success,
        "mst_success_only_count": mst_wins_sac_success,
        "same_success_non_success_class": same_success_class,
        "same_four_way_terminal_outcome": same_terminal,
        "success_rate_difference_mst_minus_sac": delta,
        "sac_success_rate": sum(terminal(a[s]) == "success" for s in common) / len(common) if common else None,
        "mst_success_rate": sum(terminal(b[s]) == "success" for s in common) / len(common) if common else None,
        "sac_success_rate_wilson_95pct": wilson(sum(terminal(a[s]) == "success" for s in common), len(common)),
        "mst_success_rate_wilson_95pct": wilson(sum(terminal(b[s]) == "success" for s in common), len(common)),
        "exact_two_sided_mcnemar_p_descriptive": exact_mcnemar(sac_wins_mst_success, mst_wins_sac_success),
        "paired_episode_bootstrap_success_difference_95pct": [percentile(boot, 0.025), percentile(boot, 0.975)] if boot else None,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_reps,
        "bootstrap_resampling_unit": "matched validation traffic seed pair (fixed policies/checkpoints)",
        "paired_evaluation_rate_ci_limit": "Descriptive for these two fixed seed-0 checkpoints and the validation traffic draws; does not estimate training-seed variability or general method performance.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sac-dir", required=True, type=Path)
    parser.add_argument("--mst-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sac = summarize_method("SAC+MLP", args.sac_dir)
    mst = summarize_method("MST+SLT", args.mst_dir)
    pair = paired(sac, mst)
    # Drop per-episode private maps from the public JSON; they remain represented by paired matrix and source hashes.
    sac.pop("private_records", None)
    sac.pop("private_schedule_hashes", None)
    sac.pop("private_runtime_seed_map", None)
    mst.pop("private_records", None)
    mst.pop("private_schedule_hashes", None)
    mst.pop("private_runtime_seed_map", None)
    internal = args.mst_dir / "final_evaluation.json"
    internal_summary = read_json(internal).get("summary") if internal.is_file() else None
    output = {
        "schema_version": 1,
        "created_at_note": "Offline read-only aggregation; no model, SUMO, training, or evaluation was run.",
        "scenario_identity": {
            "scenario": "intersection_random_darrl_medium_v1",
            "configuration_revision": "darrl_r2_20260930",
            "demand_probability_per_0p1s_step_per_inlet": 0.03,
            "traffic_split": "validation",
        },
        "methods": {"SAC+MLP": sac, "MST+SLT": mst},
        "paired_comparison": pair,
        "mst_internal_evaluation_note": {
            "file": str(internal) if internal.is_file() else None,
            "summary": internal_summary,
            "used_in_primary_comparison": False,
            "reason": "The comparison uses MST's outer evaluation_results.json; the existing internal final evaluation repeats the same validation seed range/policy and is reported separately, not as extra independent samples.",
        },
        "protocol_fairness_and_limits": [
            "Both policies were trained from scratch at training seed 0 with a 100000 raw-step budget, action_repeat=3, batch_size=32, learning_starts=5000 raw steps, learning_rate=1e-4, replay buffer 20000, gamma=0.99, and CUDA; checkpoints in the outer evaluation identities hash-match each final_model.zip.",
            "Both outer evaluations contain the same logical validation seeds 10000..10099, and per-seed seeded schedule SHA-256 is checked rather than relying only on episode order.",
            "MST arguments.json contains a legacy frozen_80_20 label, but its per-method experiment_manifest and runtime train/eval diagnostics identify the DARRL Bernoulli r2 traffic protocol; the two methods' 100 evaluation schedule SHA-256 values match seed by seed. The stale argument label is not used as evidence of a different realized evaluation pool.",
            "There is only one training seed per method. Wilson/binomial or paired bootstrap summaries condition on the fixed trained policies and validation traffic draws; they do not quantify training-seed variability.",
            "MST and SAC+MLP are distinct algorithms/representations; this comparison is the strong-baseline vs pure-RL-baseline result, not a single-module ablation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
