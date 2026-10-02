"""Read-only result audit for the 2026-09-29 fresh 100k SAC/ST-RT run.

Reads completed evaluation, run metadata, and embedded-diagnostic summaries.
It does not import the model, start SUMO, or change training/evaluation files.
Run from any working directory with Python and NumPy available:
    python analyze_100k_results.py
The only file it writes is analysis_100k_results.json beside this script.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from xml.etree import ElementTree as ET

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
FAST_DEVELOPER = ANALYSIS_DIR.parent
WORKSPACE = FAST_DEVELOPER.parents[2]
RUN_ROOT = WORKSPACE / "runs" / "d0929_100k_diag"
HISTORICAL_DIRS = {
    "D1-ST": FAST_DEVELOPER / "sac_mlp_d1_st__intersection_sorted_depart4p0",
    "MST+SLT": FAST_DEVELOPER / "mst_slt__intersection_sorted_depart4p0",
}
CURRENT_DIRS = {
    "SAC+MLP": RUN_ROOT / "sac_mlp__intersection_sorted_depart4p0",
    "ST-RT": RUN_ROOT / "sac_mlp_d1_st_rt__intersection_sorted_depart4p0",
}
DIAG_DIRS = {
    "SAC+MLP": {"train": "train", "eval": "eval_worker_00"},
    "ST-RT": {"train": "train", "eval": "eval"},
}
OUTCOMES = ("success", "collision", "timeout", "off_route", "other")
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_REPLICATES = 50_000
Z_95 = 1.959963984540054


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(record: dict) -> str:
    if record.get("success"):
        return "success"
    if record.get("collision"):
        return "collision"
    if record.get("off_route"):
        return "off_route"
    if record.get("timeout"):
        return "timeout"
    return "other"


def mean_or_none(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(finite) if finite else None


def timestamp_iso(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
    return value


def wilson_interval(successes: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + Z_95 * Z_95 / n
    center = (p + Z_95 * Z_95 / (2 * n)) / denom
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95 * Z_95 / (4 * n * n)) / denom
    return [center - half, center + half]


def exact_mcnemar_p(source_only: int, target_only: int) -> float:
    """Two-sided exact binomial test for discordant binary success pairs."""
    n = source_only + target_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(source_only, target_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def eval_outcome_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(classify(row) for row in records)
    return {label: int(counts.get(label, 0)) for label in OUTCOMES}


def eval_summary(records: list[dict], identity: dict) -> dict:
    counts = eval_outcome_counts(records)
    n = len(records)
    successes = counts["success"]
    return {
        "episodes": n,
        "outcome_counts": counts,
        "success_rate": successes / n if n else None,
        "success_wilson_95_episode_iid_only": wilson_interval(successes, n),
        "mean_return": mean_or_none(row.get("episode_return") for row in records),
        "mean_raw_steps": mean_or_none(row.get("raw_steps") for row in records),
        "mean_decision_steps": mean_or_none(row.get("decision_steps") for row in records),
        "mean_success_completion_seconds": mean_or_none(
            row.get("completion_time_seconds") for row in records if row.get("success")
        ),
        "checkpoint": identity.get("checkpoint"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "eval_episodes_metadata": identity.get("episodes"),
        "eval_workers_metadata": identity.get("workers"),
        "smoke": identity.get("smoke"),
    }


def keyed_records(records: list[dict]) -> dict[tuple[int, str], dict]:
    result = {}
    for row in records:
        if row.get("seed") is None or row.get("traffic_variant") is None:
            raise ValueError("evaluation record lacks seed or traffic_variant")
        key = (int(row["seed"]), str(row["traffic_variant"]))
        if key in result:
            raise ValueError(f"duplicate seed/traffic key: {key}")
        result[key] = row
    return result


def paired_comparison(source_name: str, source_records: list[dict], target_records: list[dict]) -> dict:
    source = keyed_records(source_records)
    target = keyed_records(target_records)
    common = sorted(set(source) & set(target))
    if not common:
        raise ValueError(f"no paired evaluation keys for {source_name} -> ST-RT")

    matrix = {row: {col: 0 for col in OUTCOMES} for row in OUTCOMES}
    for key in common:
        matrix[classify(source[key])][classify(target[key])] += 1
    row_totals = {row: sum(matrix[row].values()) for row in OUTCOMES}
    col_totals = {col: sum(matrix[row][col] for row in OUTCOMES) for col in OUTCOMES}
    source_counts = eval_outcome_counts(source_records)
    target_counts = eval_outcome_counts(target_records)
    if row_totals != source_counts or col_totals != target_counts:
        raise AssertionError("paired matrix margins do not match source outcome totals")

    source_only_success = sum(matrix["success"][col] for col in OUTCOMES if col != "success")
    target_only_success = sum(matrix[row]["success"] for row in OUTCOMES if row != "success")
    neither_success = sum(
        matrix[row][col]
        for row in OUTCOMES if row != "success"
        for col in OUTCOMES if col != "success"
    )

    variants = sorted({key[1] for key in common})
    clustered_keys = {variant: [key for key in common if key[1] == variant] for variant in variants}
    cluster_sizes = np.asarray([len(clustered_keys[v]) for v in variants], dtype=np.int64)
    source_cluster_counts = np.zeros((len(variants), len(OUTCOMES)), dtype=np.int64)
    target_cluster_counts = np.zeros_like(source_cluster_counts)
    for i, variant in enumerate(variants):
        for key in clustered_keys[variant]:
            source_cluster_counts[i, OUTCOMES.index(classify(source[key]))] += 1
            target_cluster_counts[i, OUTCOMES.index(classify(target[key]))] += 1

    # Resample 30 traffic-variant clusters with replacement. Keep every actual
    # 3/4 evaluation record in each selected cluster, then pool record counts.
    # This preserves the observed cluster-size weights; it is not a training-seed CI.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(0, len(variants), size=(BOOTSTRAP_REPLICATES, len(variants)))
    denominators = cluster_sizes[selected].sum(axis=1)
    bootstrap = {}
    point = {}
    for outcome in ("success", "collision", "timeout"):
        j = OUTCOMES.index(outcome)
        source_draws = source_cluster_counts[selected, j].sum(axis=1)
        target_draws = target_cluster_counts[selected, j].sum(axis=1)
        differences = 100.0 * (target_draws - source_draws) / denominators
        source_rate = source_counts[outcome] / len(common)
        target_rate = target_counts[outcome] / len(common)
        ci = np.percentile(differences, [2.5, 97.5], method="linear")
        point[outcome] = 100.0 * (target_rate - source_rate)
        bootstrap[outcome] = {
            "target_minus_source_percentage_points": point[outcome],
            "percentile_95_ci": [float(ci[0]), float(ci[1])],
        }

    return {
        "source_method_rows": source_name,
        "target_method_columns": "ST-RT",
        "pair_key": ["evaluation_seed", "traffic_variant"],
        "matched_pairs": len(common),
        "unmatched_source": len(set(source) - set(target)),
        "unmatched_target": len(set(target) - set(source)),
        "matrix_source_rows_target_columns": matrix,
        "row_margins": row_totals,
        "column_margins": col_totals,
        "success_2x2": {
            "both_success": matrix["success"]["success"],
            "source_only_success": source_only_success,
            "target_only_success": target_only_success,
            "neither_success": neither_success,
            "mcnemar_exact_two_sided_episode_iid_p": exact_mcnemar_p(
                source_only_success, target_only_success
            ),
            "interpretation_limit": "Exploratory only: episode-IID p-value ignores repeated traffic variants and does not measure training-seed variability.",
        },
        "traffic_clusters": {
            "count": len(variants),
            "variant_episode_counts": dict(sorted(Counter(map(len, clustered_keys.values())).items())),
        },
        "cluster_bootstrap": {
            "direction": "ST-RT minus source method, percentage points",
            "resampling_unit": "traffic_variant; sample 30 variant IDs with replacement and include every matched episode for each selected ID",
            "weighting": "pooled episode counts retain observed 3/4 episode cluster sizes",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "rng": "NumPy Generator PCG64 via default_rng",
            "ci": "two-sided 95% percentile interval using linear quantiles",
            "limitation": "Fixed-checkpoint scenario-cluster sensitivity only; not a confidence interval over training seeds or over unseen route populations.",
            "outcomes": bootstrap,
        },
    }


def diagnostic_summary(path: Path) -> dict:
    return load_json(path)


def outcome_conditioned_behavior(eval_records: list[dict], diag: dict) -> dict:
    by_seed = {int(row["seed"]): row for row in eval_records}
    groups = defaultdict(list)
    for episode in diag.get("episode_summaries", []):
        seed = int(episode["seed"])
        if seed not in by_seed:
            continue
        groups[classify(by_seed[seed])].append(episode)

    result = {}
    for outcome in OUTCOMES:
        episodes = groups.get(outcome, [])
        if not episodes:
            result[outcome] = {"episodes": 0, "metrics": None}
            continue
        sum_field = lambda key: sum(float(ep.get(key, 0) or 0) for ep in episodes)
        action_n = sum(
            int(action.get("n", 0))
            for ep in episodes
            for action in ep.get("action_statistics", [])
        )
        near_bound_n = sum(
            int(action.get("abs_ge_0p95_count", 0))
            for ep in episodes
            for action in ep.get("action_statistics", [])
        )
        speed_n = sum_field("speed_samples")
        risk_n = sum_field("risk_evaluable_ticks")
        raw_n = sum_field("raw_steps")
        result[outcome] = {
            "episodes": len(episodes),
            "metrics": {
                "mean_raw_steps": sum_field("raw_steps") / len(episodes),
                "mean_decision_steps": sum_field("decisions") / len(episodes),
                "mean_simulated_seconds_with_speed_observed": sum_field("speed_observed_seconds") / len(episodes),
                "actual_speed_mps_weighted_by_speed_samples": sum_field("speed_sum_mps") / speed_n if speed_n else None,
                "speed_samples_denominator": int(speed_n),
                "stopped_tick_fraction_of_speed_samples": sum_field("stopped_ticks") / speed_n if speed_n else None,
                "cv_obb_ttc_below_3s_fraction_of_risk_evaluable_ticks": sum_field("cv_ttc_below_3s_ticks") / risk_n if risk_n else None,
                "risk_evaluable_ticks_denominator": int(risk_n),
                "near_unit_action_scalar_fraction_abs_ge_0p95": near_bound_n / action_n if action_n else None,
                "action_scalar_denominator": action_n,
                "near_unit_action_scalar_count": near_bound_n,
                "risk_observation_coverage_fraction": sum_field("risk_and_observation_covered_ticks") / raw_n if raw_n else None,
            },
        }
    return result


def paired_common_success_efficiency(sac_records: list[dict], rt_records: list[dict], sac_diag: dict, rt_diag: dict) -> dict:
    sac = keyed_records(sac_records)
    rt = keyed_records(rt_records)
    sac_diag_by_seed = {int(row["seed"]): row for row in sac_diag.get("episode_summaries", [])}
    rt_diag_by_seed = {int(row["seed"]): row for row in rt_diag.get("episode_summaries", [])}
    pairs = []
    for key in sorted(set(sac) & set(rt)):
        if sac[key].get("success") and rt[key].get("success"):
            seed = key[0]
            a, b = sac[key], rt[key]
            da, db = sac_diag_by_seed[seed], rt_diag_by_seed[seed]
            va = float(da["speed_sum_mps"]) / float(da["speed_samples"])
            vb = float(db["speed_sum_mps"]) / float(db["speed_samples"])
            pairs.append({
                "seed": seed,
                "traffic_variant": key[1],
                "sac_raw_steps": int(a["raw_steps"]),
                "strt_raw_steps": int(b["raw_steps"]),
                "strt_minus_sac_raw_steps": int(b["raw_steps"]) - int(a["raw_steps"]),
                "sac_completion_seconds": float(a["completion_time_seconds"]),
                "strt_completion_seconds": float(b["completion_time_seconds"]),
                "strt_minus_sac_completion_seconds": float(b["completion_time_seconds"]) - float(a["completion_time_seconds"]),
                "sac_episode_weighted_actual_speed_mps": va,
                "strt_episode_weighted_actual_speed_mps": vb,
                "strt_minus_sac_episode_speed_mps": vb - va,
            })
    if not pairs:
        return {"matched_common_successes": 0, "pairs": []}
    def stat(field):
        values = [float(pair[field]) for pair in pairs]
        return {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
    return {
        "direction": "ST-RT minus SAC+MLP among same seed+traffic pairs where both checkpoints succeeded",
        "matched_common_successes": len(pairs),
        "raw_steps_delta": stat("strt_minus_sac_raw_steps"),
        "completion_seconds_delta": stat("strt_minus_sac_completion_seconds"),
        "episode_speed_delta_mps": stat("strt_minus_sac_episode_speed_mps"),
        "episode_speed_mean_mps": {
            "SAC+MLP": mean_or_none(pair["sac_episode_weighted_actual_speed_mps"] for pair in pairs),
            "ST-RT": mean_or_none(pair["strt_episode_weighted_actual_speed_mps"] for pair in pairs),
        },
        "paired_records": pairs,
    }


def route_liveness_audit() -> dict:
    """Cross-check fixed ego route, network connections, and timeout snapshots."""
    sumo_root = FAST_DEVELOPER.parents[0] / "envs" / "sumo"
    scenario_root = sumo_root / "original_scenarios_v1" / "intersection_sorted"
    net_path = scenario_root / "map.net.xml"
    ego_route_path = scenario_root / "ego.rou.xml"
    snapshots_path = CURRENT_DIRS["SAC+MLP"] / "diagnostics" / "eval_worker_00" / "episodes.jsonl"
    behavior_path = ANALYSIS_DIR / "analysis_100k_behavior.json"
    net = ET.parse(net_path).getroot()
    route_xml = ET.parse(ego_route_path).getroot()

    ego_route = route_xml.find("route[@id='scene_rep_ego_route']")
    ego_vehicle = route_xml.find("vehicle[@id='ego']")
    if ego_route is None or ego_vehicle is None:
        raise AssertionError("expected released intersection_sorted ego route is missing")
    route_edges = ego_route.get("edges", "").split()
    if len(route_edges) < 2:
        raise AssertionError("ego route has no next edge")

    current_edge, next_edge = route_edges[:2]
    edge_node = net.find(f"edge[@id='{current_edge}']")
    if edge_node is None:
        raise AssertionError(f"ego current edge {current_edge} not found in scenario net")
    lane_rows = []
    for lane in edge_node.findall("lane"):
        lane_rows.append({
            "id": lane.get("id"),
            "index": int(lane.get("index", -1)),
            "length_m": float(lane.get("length", "nan")),
            "allow_attribute": lane.get("allow"),
            "disallow_attribute": lane.get("disallow"),
        })
    connections = [
        {
            "from_lane": int(connection.get("fromLane", -1)),
            "to_lane": int(connection.get("toLane", -1)),
            "direction": connection.get("dir"),
            "state": connection.get("state"),
        }
        for connection in net.findall("connection")
        if connection.get("from") == current_edge and connection.get("to") == next_edge
    ]
    legal_from_lanes = sorted({row["from_lane"] for row in connections})

    timeout_snapshots = []
    with snapshots_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            if episode.get("outcome") != "timeout":
                continue
            ego = (episode.get("last_snapshot") or {}).get("ego") or {}
            reset_info = episode.get("reset_info") or {}
            terminal = episode.get("terminal_info") or {}
            timeout_snapshots.append({
                "seed": int(episode["seed"]),
                "route_id": ego.get("route_id"),
                "route_edges": ego.get("route_edges"),
                "route_index": ego.get("route_index"),
                "road_id": ego.get("road_id"),
                "lane_id": ego.get("lane_id"),
                "lane_position_m": ego.get("lane_position"),
                "ego_speed_mps": ego.get("speed"),
                "source_observation_contract": reset_info.get("source_observation_contract"),
                "speed_mode": reset_info.get("ego_speed_mode"),
                "lane_change_mode": reset_info.get("ego_lane_change_mode"),
                "terminal_target_speed_mps": terminal.get("target_speed"),
                "terminal_lane_command": terminal.get("lane_command"),
                "terminal_lane_change_request_sent": terminal.get("lane_change_applied"),
            })

    timeout_route_edges = Counter(tuple(row.get("route_edges") or ()) for row in timeout_snapshots)
    timeout_route_indices = Counter(row.get("route_index") for row in timeout_snapshots)
    timeout_lanes = Counter(row.get("lane_id") for row in timeout_snapshots)
    timeout_positions = Counter(row.get("lane_position_m") for row in timeout_snapshots)
    timeout_speeds = Counter(row.get("ego_speed_mps") for row in timeout_snapshots)
    timeout_modes = Counter(row.get("lane_change_mode") for row in timeout_snapshots)
    if len(timeout_snapshots) != 40:
        raise AssertionError(f"expected 40 SAC timeout snapshots, found {len(timeout_snapshots)}")
    if set(timeout_route_edges) != {(tuple(route_edges))} or set(timeout_route_indices) != {0}:
        raise AssertionError("timeout snapshots do not all match the fixed ego route and route index")
    if legal_from_lanes != [2]:
        raise AssertionError(f"expected only fromLane=2 for {current_edge}->{next_edge}, got {legal_from_lanes}")

    behavior = load_json(behavior_path)
    timeout_behavior = behavior["methods"]["sac_mlp"]["outcome_groups"]["timeout"]
    seed_10002 = next((row for row in timeout_snapshots if row["seed"] == 10002), None)
    if seed_10002 is None:
        raise AssertionError("seed 10002 timeout record not found")

    return {
        "scenario": "intersection_sorted",
        "network_path": str(net_path),
        "ego_route_path": str(ego_route_path),
        "timeout_episode_snapshot_path": str(snapshots_path),
        "timeout_behavior_summary_path": str(behavior_path),
        "fixed_ego_route": {
            "route_id": ego_route.get("id"),
            "route_edges": route_edges,
            "vehicle_depart_lane": ego_vehicle.get("departLane"),
            "vehicle_arrival_lane": ego_vehicle.get("arrivalLane"),
        },
        "current_to_next_edge": {
            "current_edge": current_edge,
            "next_edge": next_edge,
            "current_edge_lanes": lane_rows,
            "explicit_allow_disallow_attributes_present": any(
                lane["allow_attribute"] is not None or lane["disallow_attribute"] is not None
                for lane in lane_rows
            ),
            "connections_for_route_transition": connections,
            "legal_from_lane_indices_in_static_connections": legal_from_lanes,
        },
        "observed_sac_timeout_terminal_states": {
            "n": len(timeout_snapshots),
            "all_match_fixed_route": True,
            "route_edges_counts": {" ".join(key): count for key, count in timeout_route_edges.items()},
            "route_index_counts": dict(timeout_route_indices),
            "road_id_counts": dict(Counter(row.get("road_id") for row in timeout_snapshots)),
            "lane_id_counts": dict(timeout_lanes),
            "lane_position_m_counts": dict(timeout_positions),
            "ego_speed_mps_counts": dict(timeout_speeds),
            "lane_change_mode_counts_from_reset_info": dict(timeout_modes),
            "lane_change_mode_configured_by_environment_code": 0,
            "mean_target_speed_mps_equal_episode": timeout_behavior.get("mean_target_speed_mps_mean_equal_episode"),
            "mean_actual_speed_mps_equal_episode": timeout_behavior.get("mean_actual_speed_mps_mean_equal_episode"),
            "stopped_fraction_mean_equal_episode": timeout_behavior.get("stopped_fraction_mean_equal_episode"),
            "seed_10002_snapshot": seed_10002,
        },
        "interpretation": "All 40 timeout snapshots show the fixed route (-E1,-E0), route_index 0, and terminal position at the 70 m end of -E1 on lane 0 or 1. The static network lists only fromLane 2 for the -E1 to -E0 connection. This supports route-incompatible-lane stagnation with requested target speed not realized; it does not establish why the policy selected those lanes or by itself prove an environment control bug.",
        "not_inferred": [
            "No claim that the complete route is unreachable: the left-turn lane can be reached by lane changes before the junction.",
            "No claim that speedMode=0 disables every SUMO constraint.",
            "No causal attribution of timeout behavior to one code setting or module from this static comparison.",
        ],
    }


def main() -> None:
    launcher_path = RUN_ROOT / "launcher_status.json"
    launcher = load_json(launcher_path)
    if launcher.get("status") != "complete":
        raise RuntimeError(f"run launcher status is not complete: {launcher.get('status')}")

    method_data = {}
    artifacts = [launcher_path, RUN_ROOT / "experiment_manifest.json"]
    launch_methods = launcher.get("methods", {})
    for name, method_dir in CURRENT_DIRS.items():
        arguments_path = method_dir / "arguments.json"
        training_path = method_dir / "training_complete.json"
        eval_path = method_dir / "evaluation_results.json"
        model_path = method_dir / "final_model.zip"
        arguments = load_json(arguments_path)
        training = load_json(training_path)
        evaluation = load_json(eval_path)
        identity = evaluation.get("identity", {})
        records = evaluation["episode_records"]
        if training.get("raw_steps") != 100_000 or training.get("updates") != 95_001:
            raise AssertionError(f"unexpected completion budget for {name}")
        if identity.get("checkpoint_sha256") != sha256(model_path):
            raise AssertionError(f"evaluation checkpoint hash does not match final_model.zip for {name}")
        if identity.get("checkpoint_sha256") != training.get("checkpoint_sha256"):
            raise AssertionError(f"training/evaluation checkpoint hash mismatch for {name}")
        if len(records) != 100 or identity.get("smoke") is not False:
            raise AssertionError(f"unexpected final evaluation size or smoke flag for {name}")

        eval_diag_path = method_dir / "diagnostics" / DIAG_DIRS[name]["eval"] / "summary.json"
        train_diag_path = method_dir / "diagnostics" / DIAG_DIRS[name]["train"] / "summary.json"
        train_opt_path = method_dir / "diagnostics" / DIAG_DIRS[name]["train"] / "optimization.jsonl"
        eval_diag = diagnostic_summary(eval_diag_path)
        train_diag = diagnostic_summary(train_diag_path)
        launch_key = arguments.get("method")
        launch = launch_methods.get(launch_key, {})
        artifacts.extend([arguments_path, training_path, eval_path, model_path, eval_diag_path, train_diag_path, train_opt_path])

        opt_keys = (
            "time/raw_simulation_steps", "train/n_updates", "train/ent_coef",
            "train/actor_loss", "train/critic_loss", "diagnostic/q1_mean_sampled",
            "diagnostic/q2_mean_sampled", "diagnostic/target_q_mean_sampled",
            "diagnostic/q1_abs_td_mean_sampled", "diagnostic/q2_abs_td_mean_sampled",
        )
        opt = train_diag.get("optimization_statistics", {})
        method_data[name] = {
            "directory": str(method_dir),
            "arguments": arguments,
            "training_complete": training,
            "evaluation": eval_summary(records, identity),
            "evaluation_diagnostic_summary": {
                "path": str(eval_diag_path),
                "episodes_finished": eval_diag.get("episodes_finished"),
                "raw_records": eval_diag.get("raw_records"),
                "decision_records": eval_diag.get("decision_records"),
                "diagnostic_error_count": eval_diag.get("diagnostic_error_count"),
                "outcome_conditioned": outcome_conditioned_behavior(records, eval_diag),
            },
            "training_diagnostics": {
                "summary_path": str(train_diag_path),
                "optimization_jsonl_path": str(train_opt_path),
                "optimization_jsonl_exists": train_opt_path.exists(),
                "outcome_counts_including_incomplete": train_diag.get("outcome_counts"),
                "diagnostic_error_count": train_diag.get("diagnostic_error_count"),
                "selected_optimization_statistics": {key: opt[key] for key in opt_keys if key in opt},
            },
            "wall_seconds_training": training.get("wall_seconds"),
            "launcher_worker": {
                "status": launch.get("status"),
                "pid": launch.get("pid"),
                "exit_code": launch.get("exit_code"),
                "started_at": timestamp_iso(launch.get("started_at")),
                "finished_at": timestamp_iso(launch.get("finished_at")),
                "wall_seconds_including_eval": (
                    float(launch["finished_at"]) - float(launch["started_at"])
                    if launch.get("finished_at") is not None and launch.get("started_at") is not None
                    else None
                ),
                "log_path": launch.get("log"),
                "command": launch.get("command"),
            },
            "final_model_path": str(model_path),
            "final_model_sha256_verified": sha256(model_path),
        }

    historical_data = {}
    for name, directory in HISTORICAL_DIRS.items():
        eval_path = directory / "evaluation_results.json"
        arguments_path = directory / "arguments.json"
        evaluation = load_json(eval_path)
        records = evaluation["episode_records"]
        identity = evaluation.get("identity", {})
        args = load_json(arguments_path) if arguments_path.exists() else None
        completion_path = directory / "training_complete.json"
        historical_data[name] = {
            "directory": str(directory),
            "arguments_path": str(arguments_path) if arguments_path.exists() else None,
            "arguments": args,
            "training_complete_path": str(completion_path) if completion_path.exists() else None,
            "evaluation_path": str(eval_path),
            "evaluation": eval_summary(records, identity),
        }
        artifacts.append(eval_path)
        if arguments_path.exists():
            artifacts.append(arguments_path)
        if completion_path.exists():
            artifacts.append(completion_path)

    eval_records = {
        name: load_json((CURRENT_DIRS[name] / "evaluation_results.json"))["episode_records"]
        for name in CURRENT_DIRS
    }
    for name, data in historical_data.items():
        eval_records[name] = load_json(Path(data["evaluation_path"]))["episode_records"]
    target = eval_records["ST-RT"]
    comparisons = {
        name: paired_comparison(name, eval_records[name], target)
        for name in ("SAC+MLP", "D1-ST", "MST+SLT")
    }

    target_keys = keyed_records(target)
    traffic_counts = Counter(key[1] for key in target_keys)
    if len(target_keys) != 100 or len(traffic_counts) != 30 or Counter(traffic_counts.values()) != Counter({3: 20, 4: 10}):
        raise AssertionError("unexpected evaluation seed/traffic variant coverage")
    common_key_counts = {
        name: len(set(keyed_records(eval_records[name])) & set(target_keys))
        for name in ("SAC+MLP", "D1-ST", "MST+SLT")
    }

    diag_sac = load_json(CURRENT_DIRS["SAC+MLP"] / "diagnostics" / "eval_worker_00" / "summary.json")
    diag_rt = load_json(CURRENT_DIRS["ST-RT"] / "diagnostics" / "eval" / "summary.json")
    common_success = paired_common_success_efficiency(
        eval_records["SAC+MLP"], eval_records["ST-RT"], diag_sac, diag_rt
    )
    route_audit = route_liveness_audit()
    artifacts.extend([
        Path(route_audit["network_path"]),
        Path(route_audit["ego_route_path"]),
        Path(route_audit["timeout_episode_snapshot_path"]),
        Path(route_audit["timeout_behavior_summary_path"]),
        FAST_DEVELOPER.parents[0] / "envs" / "sumo" / "paper_scenario_registry.py",
        FAST_DEVELOPER.parents[0] / "envs" / "sumo" / "paper_env.py",
        FAST_DEVELOPER.parents[0] / "envs" / "sumo" / "sumo_env.py",
    ])

    traffic_files = launcher.get("traffic_files", [])
    result = {
        "schema_version": 1,
        "analysis_generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "scope": "Read-only audit of completed artifacts; no model, SUMO, train, or evaluation execution.",
        "run": {
            "run_root": str(RUN_ROOT),
            "launcher_status": launcher.get("status"),
            "exit_codes": launcher.get("exit_codes"),
            "failed_methods": launcher.get("failed_methods"),
            "source_script": launcher.get("source_script"),
            "scenario": launcher.get("scenario"),
            "depart_scale": launcher.get("depart_scale"),
            "raw_step_budget": launcher.get("raw_step_budget"),
            "training_seed": launcher.get("training_seed"),
            "smoke": launcher.get("smoke"),
            "checkpoint_frequency_raw_steps": launcher.get("checkpoint_frequency_raw_steps"),
            "eval_episodes": launcher.get("eval_episodes"),
            "traffic_protocol": launcher.get("traffic_protocol"),
            "traffic_file_count": len(traffic_files),
            "traffic_files_all_exist": all(Path(path).exists() for path in traffic_files),
            "traffic_files": traffic_files,
            "start_time": timestamp_iso(launcher.get("started_at")),
            "finish_time": timestamp_iso(launcher.get("finished_at")),
            "elapsed_seconds": (
                float(launcher["finished_at"]) - float(launcher["started_at"])
                if launcher.get("finished_at") is not None and launcher.get("started_at") is not None
                else None
            ),
        },
        "current_methods": method_data,
        "historical_fresh_100k_methods": historical_data,
        "pairing": {
            "join_key": ["evaluation_seed", "traffic_variant"],
            "common_pairs_by_source": common_key_counts,
            "target_variant_count": len(traffic_counts),
            "variant_size_distribution": dict(sorted(Counter(traffic_counts.values()).items())),
            "comparisons_to_ST_RT": comparisons,
            "common_success_efficiency_SAC_vs_ST_RT": common_success,
            "SAC_timeout_route_liveness": route_audit,
        },
        "limitations": [
            "Only one training seed per method; no training-seed stability estimate.",
            "The 100 episodes reuse 30 traffic variants (20 variants with 3 episodes, 10 with 4); there is no held-out route-template test.",
            "Per-episode Wilson intervals and exact McNemar p-values assume independent episodes and are exploratory under repeated variants.",
            "Traffic-cluster bootstrap intervals resample only the observed fixed-checkpoint route variants; they do not estimate training-seed variability or performance on unseen route populations.",
            "SAC+MLP to ST-RT changes the feature extractor family and route-processing path; only ST to ST-RT is the intended route-toggle ablation, and that comparison still has one training seed.",
        ],
        "source_artifacts": [
            {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None,
             "sha256": sha256(path) if path.is_file() else None}
            for path in dict.fromkeys(artifacts)
        ],
    }

    out_path = ANALYSIS_DIR / "analysis_100k_results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "methods": {name: item["evaluation"]["outcome_counts"] for name, item in method_data.items()},
        "pairs": {name: {
            "matched": item["matched_pairs"],
            "row_margins": item["row_margins"],
            "column_margins": item["column_margins"],
            "success_cluster_bootstrap_ci": item["cluster_bootstrap"]["outcomes"]["success"]["percentile_95_ci"],
        } for name, item in comparisons.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
