"""Read-only aggregation for the sealed sorted ConflictTiming run.

This script reads normal train diagnostics and the launch source archive. It
does not open SUMO, load a policy, or modify any run result.  The optional
``--phase eval`` pass is intended only after the coordinator confirms that
the final 100-episode evaluation is complete and its files are stable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


WS = Path(__file__).resolve().parents[4]
RUN = WS / "runs" / "sortct_1002" / "sac_mlp_d1_st_rt_conflicttime_v1__intersection_sorted_depart4p0"
ARCHIVE_ROOT = WS / "runs" / "sortct_1002" / "source_archive"
DEFAULT_OUTPUT = Path(__file__).with_name("conflicttime_train_diagnostics_audit_20261003.json")
RAW_STEP_SECONDS = 0.1
RIGHT_CENSORED = {
    "prediction_horizon_right_censored",
    "episode_terminal_right_censored",
    "recorder_closed_right_censored",
}
SITE_NAMES = {
    1: "rollout_policy",
    2: "representation_objective_online",
    3: "critic_td_current",
    4: "actor_policy_current",
    5: "critic_actor_value",
    6: "eval_policy",
    7: "actor_policy_next_action_no_grad",
}


def jread(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stats(values: Iterable[Any]) -> dict[str, Any]:
    vals = [x for item in values if (x := number(item)) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(vals)
    p90 = ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
    }


def bin_name(raw: int | float | None) -> str:
    if raw is None:
        return "unmapped"
    if raw <= 33333:
        return "early_0_33333"
    if raw <= 66666:
        return "middle_33334_66666"
    return "late_66667_100000"


def side_interval(row: dict[str, Any], group: str, side: str) -> dict[str, Any] | None:
    value = row.get(group, {}).get(side)
    return value if isinstance(value, dict) else None


def prediction_times(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("prediction", {}).get("predicted", {})


def add_values(target: dict[str, list[Any]], metrics: dict[str, Any], prefix: str = "") -> None:
    for key, value in metrics.items():
        if prefix and not key.startswith(prefix):
            continue
        target[key].append(value)


def path_norm(path: str) -> str:
    return str(path).replace("/", "\\").casefold()


def source_archive_audit() -> dict[str, Any]:
    provenance = jread(RUN / "runtime_provenance.json")
    archive_manifest = jread(ARCHIVE_ROOT / "manifest.json")
    archived_by_source = {path_norm(item["source"]): item for item in archive_manifest.get("files", [])}
    record = provenance.get("records", [{}])[0]
    checked = []
    for name, item in record.get("source_files", {}).items():
        archive_item = archived_by_source.get(path_norm(item.get("path", "")))
        archive_file = ARCHIVE_ROOT / archive_item["copy"] if archive_item else None
        archive_sha = sha256(archive_file) if archive_file and archive_file.exists() else None
        expected = item.get("sha256")
        checked.append({
            "name": name,
            "runtime_path": item.get("path"),
            "runtime_sha256": expected,
            "archive_copy": archive_item.get("copy") if archive_item else None,
            "manifest_sha256": archive_item.get("sha256") if archive_item else None,
            "archive_copy_sha256": archive_sha,
            "verified": bool(archive_item and archive_sha == expected == archive_item.get("sha256")),
        })
    final_model = RUN / "final_model.zip"
    training_complete = jread(RUN / "training_complete.json")
    return {
        "runtime_provenance": {
            "method": record.get("method"),
            "phase": record.get("phase"),
            "python_executable": record.get("python_executable"),
            "cwd": record.get("cwd"),
            "argv": record.get("argv"),
        },
        "archive_manifest": {
            "source_project": archive_manifest.get("source_project"),
            "source_file_count": archive_manifest.get("source_file_count"),
            "sorted_template_count": archive_manifest.get("sorted_template_count"),
            "effective_traffic_pool_count": archive_manifest.get("effective_traffic_pool", {}).get("count"),
            "effective_traffic_scale": archive_manifest.get("effective_traffic_pool", {}).get("depart_scale"),
        },
        "runtime_code_hash_checks": checked,
        "all_runtime_code_hashes_verified_in_archive": bool(checked) and all(x["verified"] for x in checked),
        "training_complete": training_complete,
        "final_model_sha256_actual": sha256(final_model) if final_model.exists() else None,
        "final_model_hash_matches_training_complete": bool(final_model.exists()) and sha256(final_model) == training_complete.get("checkpoint_sha256"),
    }


def episode_clock(phase_dir: Path) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int], dict[str, Any]]:
    episode_rows = list(rows(phase_dir / "episodes.jsonl"))
    ordered = sorted(episode_rows, key=lambda x: int(x.get("episode", 0)))
    episode_index_to_id: dict[int, int] = {}
    episode_global_start: dict[int, int] = {}
    cumulative = 0
    for index, ep in enumerate(ordered):
        episode_id = int(ep.get("episode", index + 1))
        episode_index_to_id[index] = episode_id
        episode_global_start[episode_id] = cumulative
        cumulative += int(ep.get("raw_steps", 0) or 0)

    local_pre: dict[tuple[int, int], int] = {}
    local_clock: Counter[int] = Counter()
    decision_rows = 0
    decision_ticks_by_episode: Counter[int] = Counter()
    for decision in rows(phase_dir / "decisions.jsonl.gz"):
        episode_id = int(decision.get("episode", -1))
        decision_id = int(decision.get("decision", -1))
        local_pre[(episode_id, decision_id)] = local_clock[episode_id]
        ticks = int(decision.get("raw_ticks", 0) or 0)
        local_clock[episode_id] += ticks
        decision_ticks_by_episode[episode_id] += ticks
        decision_rows += 1
    episode_raw_by_id = {int(ep.get("episode", -1)): int(ep.get("raw_steps", 0) or 0) for ep in ordered}
    raw_sum = sum(episode_raw_by_id.values())
    raw_mismatch = {
        str(ep): {"episode_raw_steps": count, "decision_raw_ticks": decision_ticks_by_episode.get(ep, 0)}
        for ep, count in episode_raw_by_id.items()
        if count != decision_ticks_by_episode.get(ep, 0)
    }
    return episode_index_to_id, episode_global_start, local_pre, {
        "episode_rows": len(ordered),
        "decision_rows": decision_rows,
        "episode_raw_steps_sum": raw_sum,
        "decision_raw_ticks_sum": sum(decision_ticks_by_episode.values()),
        "raw_step_totals_match_per_episode": not raw_mismatch,
        "raw_step_mismatch_examples": dict(list(raw_mismatch.items())[:5]),
        "episode_id_range": [ordered[0].get("episode"), ordered[-1].get("episode")] if ordered else None,
        "partial_or_terminal_episode_rows": dict(Counter(str(ep.get("outcome", "missing")) for ep in ordered)),
    }


def resolve_global_raw(
    episode_index: Any,
    decision_index: Any,
    raw_local: Any,
    index_to_id: dict[int, int],
    global_start: dict[int, int],
    local_pre: dict[tuple[int, int], int],
) -> tuple[int | None, bool | None]:
    try:
        eidx, didx, local = int(episode_index), int(decision_index), int(raw_local)
        episode_id = index_to_id[eidx]
    except (TypeError, ValueError, KeyError):
        return None, None
    match = local_pre.get((episode_id, didx + 1)) == local
    return global_start.get(episode_id, 0) + local, match


def process_predictions(
    phase_dir: Path,
    index_to_id: dict[int, int],
    global_start: dict[int, int],
    local_pre: dict[tuple[int, int], int],
) -> dict[str, Any]:
    path = phase_dir / "task_conflict_predictions.jsonl.gz"
    row_count = 0
    actor_count = 0
    aligned_actor_count = 0
    actor_geometry_status: Counter[str] = Counter()
    map_status: Counter[str] = Counter()
    path_termination_reasons: Counter[str] = Counter()
    supported_actor_reasons: Counter[str] = Counter()
    selected_time_invalid_reasons: Counter[str] = Counter()
    bins: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    row_valid_relations: Counter[str] = Counter()
    row_joint_times: Counter[str] = Counter()
    geometry_candidate_coverage: list[Any] = []
    geometry_paths: list[Any] = []
    support_totals = Counter()
    alignment = Counter()
    row_pairs = []
    actor_relation_rows = 0
    actor_timing_rows = 0
    actor_beyond_horizon = 0
    actor_clipped = 0
    social_feature_valid = 0
    social_feature_invalid = 0
    feature_shape_mismatch = 0
    future_route_flag_true = 0
    candidate_counts: list[Any] = []
    map_coverage_status_counts = Counter()

    for row in rows(path):
        row_count += 1
        eidx = row.get("episode_index")
        didx = row.get("decision_index")
        global_raw, decision_matches = resolve_global_raw(
            eidx, didx, row.get("raw_step_pre_action"), index_to_id, global_start, local_pre
        )
        alignment["mapped_to_episode_clock"] += global_raw is not None
        if decision_matches is not None:
            alignment["decision_clock_matches"] += decision_matches
            alignment["decision_clock_checked"] += 1
        if row.get("social_future_routes_read") is True:
            future_route_flag_true += 1
        bin_key = bin_name(global_raw)
        m = row.get("pre_action_predictions", {})
        rel = int(m.get("valid_relation_count", 0) or 0)
        joint = int(m.get("valid_joint_timing_count", 0) or 0)
        row_valid_relations[bin_key] += rel
        row_joint_times[bin_key] += joint
        row_pairs.append(int(row.get("prediction_pair_count", 0) or 0))
        bm = bins[bin_key]
        bm["valid_relation_count"].append(rel)
        bm["valid_joint_timing_count"].append(joint)
        bm["prediction_pair_count"].append(row.get("prediction_pair_count"))

        keys = row.get("actor_keys_in_trajectory_order", [])
        coverage = row.get("actor_geometry_and_map_coverage", [])
        if len(keys) != len(coverage):
            alignment["key_coverage_length_mismatch"] += 1
        features = row.get("pre_action_features", [])
        if len(features) != len(keys) or any(not isinstance(x, list) or len(x) != 20 for x in features):
            feature_shape_mismatch += 1
        details = m.get("actors", [])
        detail_by_index = {int(a.get("actor_index", -1)): a for a in details}
        if len(detail_by_index) != len(details):
            alignment["duplicate_actor_detail_indices"] += 1
        for index, key in enumerate(keys):
            actor_count += 1
            cov = coverage[index] if index < len(coverage) else None
            if not isinstance(cov, dict) or cov.get("actor_index") != index or cov.get("key") != key:
                alignment["coverage_actor_key_mismatch"] += 1
            else:
                aligned_actor_count += 1
                actor_geometry_status[str(cov.get("geometry_status", "missing"))] += 1
                mapcov = cov.get("map_coverage", {})
                map_coverage_status_counts[str(mapcov.get("status", "missing"))] += 1
                geometry_candidate_coverage.append(mapcov.get("candidate_geometry_coverage_fraction"))
                geometry_paths.append(cov.get("path_candidate_count"))
                if cov.get("path_candidates_truncated"):
                    support_totals["actors_with_truncated_candidate_list"] += 1
                for reason, count in (cov.get("candidate_termination_counts") or {}).items():
                    path_termination_reasons[str(reason)] += int(count or 0)
            if index == 0:
                continue
            detail = detail_by_index.get(index)
            if detail is None or detail.get("actor_key") != key:
                alignment["prediction_actor_key_mismatch"] += 1
            else:
                alignment["prediction_actor_key_checked"] += 1
                actor_relation_rows += bool(detail.get("relation_valid"))
                support_totals["candidate_path_pairs"] += int(detail.get("candidate_pair_count", 0) or 0)
                support_totals["supported_candidate_path_pairs"] += int(detail.get("supported_candidate_pair_count", 0) or 0)
                support_totals["supported_crossing_count"] += int(detail.get("supported_crossing_count", 0) or 0)
                support_totals["observed_social_actors"] += 1
                supported_actor_reasons[str(detail.get("reason", "missing"))] += 1
                selected = detail.get("selected")
                if isinstance(selected, dict):
                    actor_timing_rows += bool(selected.get("ego_time_valid") and selected.get("foe_time_valid"))
                    actor_beyond_horizon += bool(selected.get("beyond_calibration_horizon"))
                    actor_clipped += bool(selected.get("time_values_clipped_in_policy_features"))
                    for side in ("ego_time", "foe_time"):
                        t = selected.get(side, {})
                        if isinstance(t, dict) and not t.get("valid", False):
                            selected_time_invalid_reasons[str(t.get("reason", "missing"))] += 1
            if index < len(features) and isinstance(features[index], list) and features[index]:
                if number(features[index][0]) is not None and float(features[index][0]) > 0.5:
                    social_feature_valid += 1
                else:
                    social_feature_invalid += 1
                if detail is not None and bool(detail.get("relation_valid")) != (float(features[index][0]) > 0.5):
                    alignment["feature_relation_mask_mismatch"] += 1

    return {
        "rows": row_count,
        "row_prediction_pair_count_sum": sum(row_pairs),
        "actor_rows": actor_count,
        "actor_key_coverage_aligned_rows": aligned_actor_count,
        "actor_geometry_status_counts": dict(actor_geometry_status),
        "map_coverage_status_counts": dict(map_coverage_status_counts),
        "map_candidate_geometry_coverage_fraction": stats(geometry_candidate_coverage),
        "candidate_paths_per_actor": stats(geometry_paths),
        "candidate_path_termination_reason_counts": dict(path_termination_reasons),
        "social_actor_details_aligned": alignment.get("prediction_actor_key_checked", 0),
        "social_actor_relation_valid_count": actor_relation_rows,
        "social_actor_timing_valid_count": actor_timing_rows,
        "social_actor_selected_beyond_calibration_horizon_count": actor_beyond_horizon,
        "social_actor_time_values_clipped_in_policy_features_count": actor_clipped,
        "selected_time_invalid_reason_counts": dict(selected_time_invalid_reasons),
        "social_candidate_path_pair_count": support_totals["candidate_path_pairs"],
        "social_supported_candidate_path_pair_count": support_totals["supported_candidate_path_pairs"],
        "supported_crossing_count": support_totals["supported_crossing_count"],
        "social_supported_pair_fraction": (
            support_totals["supported_candidate_path_pairs"] / support_totals["candidate_path_pairs"]
            if support_totals["candidate_path_pairs"] else None
        ),
        "top_level_valid_relation_count_by_raw_bin": dict(row_valid_relations),
        "top_level_valid_joint_timing_count_by_raw_bin": dict(row_joint_times),
        "per_decision_valid_relation_and_joint_timing": {
            b: {k: stats(v) for k, v in metrics.items()} for b, metrics in bins.items()
        },
        "social_actor_reason_counts": dict(supported_actor_reasons),
        "pre_action_conflict_feature_valid_social_actor_rows": social_feature_valid,
        "pre_action_conflict_feature_invalid_social_actor_rows": social_feature_invalid,
        "pre_action_feature_shape_mismatch_decisions": feature_shape_mismatch,
        "social_future_routes_read_true_rows": future_route_flag_true,
        "actor_alignment_checks": dict(alignment),
        "episode_local_to_lifetime_raw_mapping": dict(alignment),
        "actor_coverage_candidate_geometry_fraction_denominator": "all actor rows in per-decision prediction sidecar; this is map-point coverage, not candidate-conflict recall",
        "time_horizon_interpretation": "valid ETA can exceed the 10s prediction horizon; those times are clipped in policy features and counted separately",
    }


def interval_error(predicted: Any, interval: dict[str, Any] | None, pre_raw: Any) -> tuple[float | None, float | None, float | None]:
    p, lo, hi = number(predicted), number(pre_raw), None
    if p is None or not interval:
        return None, None, None
    try:
        lower = float(interval["lower_raw_step"])
        upper = float(interval["upper_raw_step"])
    except (TypeError, KeyError, ValueError):
        return None, None, None
    if lo is None or upper < lower:
        return None, None, None
    actual_lo = (lower - lo) * RAW_STEP_SECONDS
    actual_hi = (upper - lo) * RAW_STEP_SECONDS
    signed = p - actual_lo if p < actual_lo else p - actual_hi if p > actual_hi else 0.0
    width = actual_hi - actual_lo
    return signed, abs(signed), width


def process_calibration(
    phase_dir: Path,
    index_to_id: dict[int, int],
    global_start: dict[int, int],
    local_pre: dict[tuple[int, int], int],
) -> dict[str, Any]:
    path = phase_dir / "task_conflict_calibration.jsonl.gz"
    summary_path = phase_dir / "task_conflict_timing_summary.json"
    summary = jread(summary_path)
    count = 0
    outcomes: Counter[str] = Counter()
    path_status: dict[str, Counter[str]] = {s: Counter() for s in ("ego", "foe")}
    eta_status: dict[str, Counter[str]] = {s: Counter() for s in ("ego", "foe")}
    interval_counts: dict[str, Counter[str]] = {s: Counter() for s in ("ego", "foe")}
    error_values: dict[str, dict[str, list[float]]] = {
        s: {metric: [] for metric in ("entry", "clearance")} for s in ("ego", "foe")
    }
    cluster_values: dict[str, dict[str, dict[tuple[Any, ...], list[float]]]] = {
        s: {metric: defaultdict(list) for metric in ("entry", "clearance")} for s in ("ego", "foe")
    }
    calibration_raw_bins: Counter[str] = Counter()
    mapping_checks = Counter()
    unique_decisions: set[tuple[Any, ...]] = set()
    clusters: set[tuple[Any, ...]] = set()
    cluster_row_counts: Counter[tuple[Any, ...]] = Counter()
    any_interval_rows = 0
    both_actor_interval_rows = 0
    unobserved_sample_totals: Counter[str] = Counter()
    joint_valid = 0
    row_side_observed = Counter()
    right_censor_side = Counter()
    deviation_missing_side = Counter()
    rows_per_status_by_side: dict[str, Counter[str]] = {s: Counter() for s in ("ego", "foe")}

    for row in rows(path):
        count += 1
        outcome = str(row.get("outcome_status", "missing"))
        outcomes[outcome] += 1
        pmeta = row.get("prediction", {})
        pred = pmeta.get("predicted", {})
        eidx = pmeta.get("episode_index")
        didx = pmeta.get("decision_index")
        raw_local = pmeta.get("raw_step_pre_action")
        global_raw, decision_matches = resolve_global_raw(
            eidx, didx, raw_local, index_to_id, global_start, local_pre
        )
        mapping_checks["mapped"] += global_raw is not None
        if decision_matches is not None:
            mapping_checks["decision_clock_checked"] += 1
            mapping_checks["decision_clock_matches"] += decision_matches
        calibration_raw_bins[bin_name(global_raw)] += 1
        foe_key = pmeta.get("foe_key")
        cluster = (eidx, pmeta.get("ego_key"), foe_key)
        clusters.add(cluster)
        cluster_row_counts[cluster] += 1
        signature = (eidx, didx, raw_local, pmeta.get("ego_key"), foe_key)
        unique_decisions.add(signature)
        status_map = row.get("actor_path_status", {})
        obs_map = row.get("unobserved_decision_samples", {})
        intervals_present = False
        side_interval_presence = {}
        for side in ("ego", "foe"):
            t = pred.get(side + "_time", {}) if isinstance(pred, dict) else {}
            valid = bool(t.get("valid", False))
            eta_status[side]["valid" if valid else "invalid"] += 1
            if not valid:
                eta_status[side]["invalid_reason:" + str(t.get("reason", "missing"))] += 1
            side_status = str(status_map.get(side, "missing"))
            path_status[side][side_status] += 1
            rows_per_status_by_side[side][outcome] += 1
            unobs = int(obs_map.get(side, 0) or 0)
            unobserved_sample_totals[side] += unobs
            entry = side_interval(row, "entry_intervals", side)
            clearance = side_interval(row, "clearance_intervals", side)
            side_interval_presence[side] = bool(entry or clearance)
            intervals_present |= bool(entry or clearance)
            if entry:
                interval_counts[side]["entry_observed"] += 1
                left = bool(entry.get("left_censored_at_prediction"))
                interval_counts[side]["entry_left_censored"] += left
                if left:
                    interval_counts[side]["entry_left_censored_excluded_from_point_error"] += 1
                pred_entry = t.get("entry_s") if valid else None
                signed, abs_error, width = interval_error(pred_entry, entry, raw_local)
                if signed is not None:
                    interval_counts[side]["entry_valid_prediction_and_interval"] += 1
                    if side_status != "tracking":
                        interval_counts[side]["entry_excluded_path_deviation"] += 1
                    elif unobs:
                        interval_counts[side]["entry_excluded_unobserved_gap"] += 1
                    elif left:
                        interval_counts[side]["entry_excluded_left_censored"] += 1
                    else:
                        error_values[side]["entry"].append(abs_error)
                        cluster_values[side]["entry"][cluster].append(abs_error)
                        interval_counts[side]["entry_interval_error_used"] += 1
                        interval_counts[side]["entry_predicted_inside_interval"] += abs_error == 0.0
                        interval_counts[side]["entry_interval_width_s_sum"] += width or 0.0
                elif not valid:
                    interval_counts[side]["entry_observed_but_eta_invalid"] += 1
            else:
                interval_counts[side]["entry_not_observed"] += 1
            if clearance:
                interval_counts[side]["clearance_observed"] += 1
                pred_exit = t.get("exit_s") if valid else None
                signed, abs_error, width = interval_error(pred_exit, clearance, raw_local)
                if signed is not None:
                    interval_counts[side]["clearance_valid_prediction_and_interval"] += 1
                    if side_status != "tracking":
                        interval_counts[side]["clearance_excluded_path_deviation"] += 1
                    elif unobs:
                        interval_counts[side]["clearance_excluded_unobserved_gap"] += 1
                    else:
                        error_values[side]["clearance"].append(abs_error)
                        cluster_values[side]["clearance"][cluster].append(abs_error)
                        interval_counts[side]["clearance_interval_error_used"] += 1
                        interval_counts[side]["clearance_predicted_inside_interval"] += abs_error == 0.0
                        interval_counts[side]["clearance_interval_width_s_sum"] += width or 0.0
                elif not valid:
                    interval_counts[side]["clearance_observed_but_eta_invalid"] += 1
            else:
                interval_counts[side]["clearance_not_observed"] += 1
                if outcome in RIGHT_CENSORED:
                    right_censor_side[side] += 1
                elif outcome == "candidate_path_deviation":
                    deviation_missing_side[side] += 1
            row_side_observed[side] += bool(entry or clearance)
        joint_valid += all(bool(pred.get(side + "_time", {}).get("valid", False)) for side in ("ego", "foe"))
        any_interval_rows += intervals_present
        both_actor_interval_rows += all(side_interval_presence.values())

    side_result = {}
    for side in ("ego", "foe"):
        side_result[side] = {
            "eta_validity_and_na_counts": dict(eta_status[side]),
            "path_status_counts": dict(path_status[side]),
            "entry_clearance_interval_counts": dict(interval_counts[side]),
            "entry_absolute_distance_to_observed_interval_seconds": stats(error_values[side]["entry"]),
            "clearance_absolute_distance_to_observed_interval_seconds": stats(error_values[side]["clearance"]),
            "entry_pair_episode_cluster_mean_absolute_distance_seconds": stats(
                mean(v) for v in cluster_values[side]["entry"].values() if v
            ),
            "clearance_pair_episode_cluster_mean_absolute_distance_seconds": stats(
                mean(v) for v in cluster_values[side]["clearance"].values() if v
            ),
            "right_censored_clearance_interval_missing": right_censor_side[side],
            "path_deviation_clearance_interval_missing": deviation_missing_side[side],
            "unobserved_decision_samples_sum": unobserved_sample_totals[side],
            "rows_with_any_observed_component": row_side_observed[side],
        }
    return {
        "summary_file": str(summary_path),
        "summary": summary,
        "rows_read": count,
        "rows_read_matches_summary": count == int(summary.get("calibration_rows_written", -1)),
        "pair_outcome_counts": dict(outcomes),
        "unique_prediction_snapshots": len(unique_decisions),
        "unique_episode_actorpair_clusters": len(clusters),
        "max_rows_per_episode_actorpair_cluster": max(cluster_row_counts.values(), default=0),
        "pair_records_with_any_observed_entry_or_clearance": any_interval_rows,
        "pair_records_with_both_sides_observed_component": both_actor_interval_rows,
        "pair_records_with_joint_valid_eta": joint_valid,
        "episode_local_to_lifetime_raw_mapping": dict(mapping_checks),
        "calibration_record_count_by_lifetime_raw_bin": dict(calibration_raw_bins),
        "side_components": side_result,
        "uncertainty_note": "Interval error includes only a side with a valid ETA, observed bracket, no entry left-censor, no route/candidate path deviation, and no unobserved decision gap. Each row is a prediction snapshot; repeated snapshots within an episode/actor pair are correlated. Predicted times are constant-current-speed, action-conditioned estimates, not collision truth.",
    }


def process_representation(phase_dir: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    for row in rows(phase_dir / "representation.jsonl"):
        rows_read += 1
        grouped[str(row.get("source", "missing"))].append(row)

    activation_metrics = [
        "route_conflict_timing_valid_actor_count",
        "route_conflict_timing_candidate_actor_count",
        "route_conflict_timing_coverage_fraction",
        "route_conflict_timing_delta_rms",
        "route_conflict_timing_base_rms_on_valid",
        "route_conflict_timing_delta_to_base_rms",
        "context_social_rms",
        "diagnostic_sample_age_forwards",
        "diagnostic_batch_size",
    ]
    activation: dict[str, Any] = {}
    for source, items in grouped.items():
        if source != "train_forward_activation":
            continue
        by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            m = row.get("metrics", {})
            code = int(number(m.get("diagnostic_sample_site_code")) or 0)
            by_site[str(code)].append(row)
        for code, site_rows in by_site.items():
            by_bin: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
            for row in site_rows:
                raw = number(row.get("raw_steps"))
                m = row.get("metrics", {})
                bin_key = bin_name(raw)
                for key in activation_metrics:
                    by_bin[bin_key][key].append(m.get(key))
            activation[SITE_NAMES.get(int(code), "unknown_site_" + code)] = {
                "site_code": int(code),
                "rows": len(site_rows),
                "raw_step_range": [min(int(r.get("raw_steps", 0)) for r in site_rows), max(int(r.get("raw_steps", 0)) for r in site_rows)],
                "by_raw_bin": {
                    b: {k: stats(v) for k, v in metrics.items()} for b, metrics in by_bin.items()
                },
            }

    grad_sources = {}
    update_ids: dict[str, set[int]] = {}
    for source in ("critic_td_gradient_pre_clip", "critic_td_parameter_update"):
        items = grouped.get(source, [])
        update_ids[source] = {
            int(number(r.get("metrics", {}).get("critic_td_update_index")) or -1) for r in items
        }
        summaries = {}
        prefix = "encoder_grad/route_conflict_timing/" if source.endswith("pre_clip") else "encoder_update/route_conflict_timing/"
        metric_names = (
            [
                "parameter_elements", "requires_grad_elements", "gradient_parameter_elements",
                "missing_gradient_elements", "gradient_parameter_fraction", "gradient_l2",
                "gradient_max_abs", "gradient_nonzero_elements", "gradient_nonfinite_elements",
                "gradient_to_parameter_l2", "parameter_l2",
            ]
            if source.endswith("pre_clip")
            else [
                "parameter_elements", "trainable_elements", "optimizer_owned_elements",
                "optimizer_nonowned_elements", "delta_l2", "relative_delta_l2",
                "parameter_l2_before", "parameter_l2_after", "changed_elements",
                "delta_nonfinite_elements",
            ]
        )
        by_bin: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for row in items:
            raw = row.get("raw_steps")
            m = row.get("metrics", {})
            target = by_bin[bin_name(raw)]
            for metric in metric_names:
                target[metric].append(m.get(prefix + metric))
        summaries = {b: {k: stats(v) for k, v in metrics.items()} for b, metrics in by_bin.items()}
        grad_sources[source] = {
            "rows": len(items),
            "update_index_range": [min(update_ids[source], default=None), max(update_ids[source], default=None)],
            "update_indices_unique": len(update_ids[source]) == len(items),
            "metrics_by_raw_bin": summaries,
        }

    return {
        "rows_read": rows_read,
        "source_row_counts": {k: len(v) for k, v in grouped.items()},
        "activation_site_counts": {name: data["rows"] for name, data in activation.items()},
        "route_conflict_timing_activation": activation,
        "route_conflict_timing_gradients_and_updates": grad_sources,
        "gradient_update_indices_intersection": len(update_ids.get("critic_td_gradient_pre_clip", set()) & update_ids.get("critic_td_parameter_update", set())),
        "gradient_update_indices_equal": update_ids.get("critic_td_gradient_pre_clip", set()) == update_ids.get("critic_td_parameter_update", set()),
        "interpretation_limits": [
            "Activation rows are cached normal-forward diagnostics and are separated by the archived diagnostic site code; their denominator is sampled encoder forwards, not decisions or optimizer updates.",
            "Gradient rows are sparse critic TD samples taken pre-clip; update rows are matching sampled parameter deltas. Their denominator is sampled updates, not all 95001 optimizer steps.",
            "A nonzero gradient/update demonstrates a trainable path, not causal usefulness or sufficient policy attention to the feature.",
        ],
    }


def process_optimization(phase_dir: Path) -> dict[str, Any]:
    items = list(rows(phase_dir / "optimization.jsonl"))
    keys = sorted({k for row in items for k in row.get("metrics", {})})
    relevant = [
        "train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss",
        "diagnostic/q1_mean_sampled", "diagnostic/q2_mean_sampled", "diagnostic/target_q_mean_sampled",
        "diagnostic/q1_abs_td_mean_sampled", "diagnostic/q2_abs_td_mean_sampled",
    ]
    by_bin: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    nonfinite: Counter[str] = Counter()
    for row in items:
        raw = number(row.get("raw_steps"))
        m = row.get("metrics", {})
        for key in relevant:
            value = m.get(key)
            if value is not None:
                if number(value) is None:
                    nonfinite[key] += 1
                by_bin[bin_name(raw)][key].append(value)
    return {
        "rows": len(items),
        "raw_step_range": [min((int(x.get("raw_steps", 0)) for x in items), default=None), max((int(x.get("raw_steps", 0)) for x in items), default=None)],
        "all_metric_keys": keys,
        "selected_training_metrics_by_raw_bin": {b: {k: stats(v) for k, v in m.items()} for b, m in by_bin.items()},
        "nonfinite_selected_metric_counts": dict(nonfinite),
        "timing_semantics": "optimization.jsonl records latest available logger values at callback; each scalar need not originate from the exact raw-step boundary or same update",
    }


def process_shadow(phase_dir: Path) -> dict[str, Any]:
    path = phase_dir / "policy_shadow_probes.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return {"rows": 0, "status": "not_present_or_empty"}
    all_rows = list(rows(path))
    summary_path = phase_dir / "policy_shadow_probes_summary.json"
    summary = jread(summary_path) if summary_path.exists() else None
    result: dict[str, Any] = {"rows": len(all_rows), "runner_summary": summary, "probes": {}}
    for name in ("route_conflict_off", "route_conflict_times_off"):
        selected = [r for r in all_rows if r.get("probe_name") == name]
        action_l2 = [r.get("action_delta_l2_normalized") for r in selected if r.get("valid") and r.get("applicable")]
        pretanh_l2 = [r.get("action_delta_l2_pre_tanh") for r in selected if r.get("valid") and r.get("applicable")]
        dim_abs = [r.get("action_delta_abs_normalized") for r in selected if r.get("valid") and r.get("applicable")]
        decoded_lane_diffs = []
        decoded_speed_diffs = []
        for r in selected:
            if not (r.get("valid") and r.get("applicable")):
                continue
            base, shadow = r.get("baseline_action_decoded"), r.get("shadow_action_decoded")
            if isinstance(base, list) and isinstance(shadow, list) and len(base) >= 2 and len(shadow) >= 2:
                decoded_speed_diffs.append(abs(float(base[0]) - float(shadow[0])))
                decoded_lane_diffs.append(float(base[1]) != float(shadow[1]))
        result["probes"][name] = {
            "rows": len(selected),
            "valid_applicable_rows": sum(bool(r.get("valid") and r.get("applicable")) for r in selected),
            "invalid_or_inapplicable_counts": dict(Counter(str(r.get("invalid_reason") or ("inapplicable" if not r.get("applicable") else "invalid")) for r in selected if not (r.get("valid") and r.get("applicable")))),
            "unique_sample_ids": len({r.get("sample_id") for r in selected}),
            "action_delta_l2_normalized": stats(action_l2),
            "action_delta_l2_pre_tanh": stats(pretanh_l2),
            "per_dimension_absolute_normalized_action_delta": [stats(row[i] for row in dim_abs if isinstance(row, list) and i < len(row)) for i in range(2)],
            "decoded_target_speed_absolute_delta_mps": stats(decoded_speed_diffs),
            "decoded_lane_command_changed_count": sum(decoded_lane_diffs),
            "decoded_lane_command_changed_fraction": mean(decoded_lane_diffs) if decoded_lane_diffs else None,
            "same_state_ids": sorted({str(r.get("sample_id")) for r in selected}),
            "same_state_key_fields": ["sample_id", "pre_obs_raw_step", "pre_obs_decision_step", "pre_obs_raw", "episode_index", "updates"],
        }
    result["q_sensitivity"] = {
        "present_in_shadow_schema": any("q" in key.lower() for row in all_rows for key in row),
        "interpretation": "Q sensitivity was not logged by policy-shadow rows; action deltas cannot be relabeled as Q-value changes.",
    }
    return result


def process_behavior(phase_dir: Path, clock_summary: dict[str, Any]) -> dict[str, Any]:
    episodes = list(rows(phase_dir / "episodes.jsonl"))
    outcomes = Counter(str(r.get("outcome", "missing")) for r in episodes)
    finish_reasons = Counter(str(r.get("finish_reason", "missing")) for r in episodes)
    stats_keys = (
        "raw_steps", "decisions", "route_lane_known_ticks", "route_lane_unknown_ticks",
        "route_lane_ineligible_ticks", "route_lane_ineligible_stopped_ticks",
        "route_lane_ineligible_seconds", "route_lane_ineligible_stopped_seconds",
        "route_lane_ineligible_stall_max_seconds", "stopped_fraction_of_speed_samples",
        "mean_actual_speed_mps", "min_cv_obb_ttc_s", "risk_evaluable_ticks",
        "risk_evaluable_low_ttc_ticks", "critical_unobserved_ticks",
    )
    distributions = {key: stats(r.get(key) for r in episodes) for key in stats_keys}
    return {
        "episodes_logged": len(episodes),
        "outcome_counts": dict(outcomes),
        "finish_reason_counts": dict(finish_reasons),
        "episode_metric_distributions": distributions,
        "clock_reconciliation": clock_summary,
        "note": "These are exploratory training episodes, not a fixed-checkpoint validation result. Training episode outcomes are not an evaluation performance estimate.",
    }


def run_phase(phase: str) -> dict[str, Any]:
    phase_dir = RUN / "diagnostics" / phase
    if not (phase_dir / "summary.json").exists():
        raise FileNotFoundError(f"diagnostic phase not found: {phase_dir}")
    index_to_id, global_start, local_pre, clock_summary = episode_clock(phase_dir)
    return {
        "phase": phase,
        "diagnostic_summary": jread(phase_dir / "summary.json"),
        "clock_and_episode_mapping": clock_summary,
        "behavior": process_behavior(phase_dir, clock_summary),
        "prediction_sidecar": process_predictions(phase_dir, index_to_id, global_start, local_pre),
        "calibration_sidecar": process_calibration(phase_dir, index_to_id, global_start, local_pre),
        "representation_sidecar": process_representation(phase_dir),
        "optimization_sidecar": process_optimization(phase_dir),
        "policy_shadow_sidecar": process_shadow(phase_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_meta = {
        "analysis_schema": "conflict_timing_diagnostic_audit_v1",
        "generated_by": str(Path(__file__).resolve()),
        "run_directory": str(RUN),
        "scope": "offline aggregation of stored normal-flow diagnostics; no training/evaluation/SUMO was run",
        "result_identity": source_archive_audit(),
        "method_arguments": jread(RUN / "arguments.json"),
        "phase_evaluation_status": "final evaluation intentionally not interpreted until coordinator confirms its output is stable" if args.phase == "train" else "coordinator-confirmed complete evaluation required before running this phase",
    }
    if args.output.exists():
        report = jread(args.output)
        report["run_identity"] = run_meta["result_identity"]
        report["method_arguments"] = run_meta["method_arguments"]
        report["scope"] = run_meta["scope"]
        report["phase_evaluation_status"] = run_meta["phase_evaluation_status"]
    else:
        report = run_meta | {"phases": {}}
    report.setdefault("phases", {})[args.phase] = run_phase(args.phase)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phase": args.phase, "result_sha256": sha256(RUN / "final_model.zip")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
