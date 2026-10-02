#!/usr/bin/env python3
"""Reproducible offline audit for the frozen-policy goal-only causal run.

This script reads existing JSON/JSONL artifacts only. It does not import the
environment, load a model, launch SUMO, or run an evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


OUTCOMES = ("success", "collision", "timeout", "off_route")
ARMS = ("control", "goaloff", "routeveto")
REWARD_COMPONENTS = (
    "reward_success", "reward_collision", "reward_off_route",
    "reward_timeout", "reward_step_cost", "reward_progress",
)


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    return pstdev(values) if values else None


def _outcome(record: dict[str, Any]) -> str:
    active = [name for name in OUTCOMES if bool(record.get(name, False))]
    if len(active) == 1:
        return active[0]
    return "nonexclusive_or_missing"


def _traffic_map(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    mapping: dict[int, dict[str, Any]] = {}
    for row in _jsonl(path):
        seed = row.get("episode_seed")
        if seed is not None:
            mapping[int(seed)] = row
    return mapping


def _pair(left_rows, right_rows, left_traffic, right_traffic) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in left_rows}
    right = {int(row["seed"]): row for row in right_rows}
    seeds = sorted(set(left) & set(right))
    matrix = {a: {b: 0 for b in OUTCOMES} for a in OUTCOMES}
    traffic_match = 0
    traffic_unknown = []
    traffic_mismatch = []
    exact_returns = {"shaped": 0, "raw": 0}
    exact_steps = {"raw": 0, "decision": 0}
    left_success_only = []
    right_success_only = []
    per_seed = []
    for seed in seeds:
        a, b = left[seed], right[seed]
        oa, ob = _outcome(a), _outcome(b)
        if oa in matrix and ob in matrix[oa]:
            matrix[oa][ob] += 1
        ta, tb = left_traffic.get(seed), right_traffic.get(seed)
        if ta is None or tb is None:
            traffic_unknown.append(seed)
        else:
            keys = ("traffic_index_before_reset", "traffic_roll", "selected_traffic_sha256")
            if all(ta.get(k) == tb.get(k) for k in keys):
                traffic_match += 1
            else:
                traffic_mismatch.append({
                    "seed": seed,
                    "left": {k: ta.get(k) for k in keys},
                    "right": {k: tb.get(k) for k in keys},
                })
        for name, key in (("shaped", "episode_return"), ("raw", "raw_episode_return")):
            if a.get(key) is not None and b.get(key) is not None and math.isclose(
                float(a[key]), float(b[key]), rel_tol=0.0, abs_tol=1e-9
            ):
                exact_returns[name] += 1
        for name, key in (("raw", "raw_steps"), ("decision", "decision_steps")):
            if a.get(key) == b.get(key):
                exact_steps[name] += 1
        if oa == "success" and ob != "success":
            left_success_only.append(seed)
        if ob == "success" and oa != "success":
            right_success_only.append(seed)
        per_seed.append({
            "seed": seed,
            "left": oa,
            "right": ob,
            "left_raw_steps": a.get("raw_steps"),
            "right_raw_steps": b.get("raw_steps"),
        })
    return {
        "matched_seed_pairs": len(seeds),
        "unmatched_left_seeds": sorted(set(left) - set(right)),
        "unmatched_right_seeds": sorted(set(right) - set(left)),
        "outcome_matrix_left_rows_right_columns": matrix,
        "left_success_only_seeds": left_success_only,
        "right_success_only_seeds": right_success_only,
        "traffic_identity": {
            "matched_pairs": traffic_match,
            "unknown_seed_pairs": traffic_unknown,
            "mismatches": traffic_mismatch,
            "identity_fields": ["episode_seed", "traffic_index_before_reset", "traffic_roll", "selected_traffic_sha256"],
        },
        "exact_equal_return_pairs": exact_returns,
        "exact_equal_step_pairs": exact_steps,
        "per_seed_outcomes_and_steps": per_seed,
    }


def _shadow_audit(arm_dir: Path) -> dict[str, Any] | None:
    diag = arm_dir / "diagnostics" / "eval"
    row_path = diag / "policy_shadow_probes.jsonl"
    summary_path = diag / "policy_shadow_probes_summary.json"
    if not row_path.is_file():
        return None
    rows = list(_jsonl(row_path))
    sample_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    per_episode: Counter[int] = Counter()
    probe_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    active_valid = active_invalid = inactive = errors = 0
    invalid_reasons: Counter[str] = Counter()
    for row in rows:
        seed = row.get("episode_seed")
        sample_id = row.get("sample_id")
        if seed is not None and sample_id is not None:
            sample_groups[(int(seed), str(sample_id))].append(row)
        if seed is not None:
            per_episode[int(seed)] = max(per_episode[int(seed)], 0)
        episode_index = row.get("episode_index")
        if episode_index is not None:
            per_episode[int(episode_index)] += 0
        probe_counts[str(row.get("probe_name", "missing"))] += 1
        applicable = bool(row.get("applicable", False))
        valid = bool(row.get("valid", False))
        if applicable and valid:
            active_valid += 1
        elif applicable:
            active_invalid += 1
            invalid_reasons[str(row.get("invalid_reason") or "unspecified")] += 1
        else:
            inactive += 1
            invalid_reasons[str(row.get("invalid_reason") or "unspecified")]+=1
        if row.get("error_type") or row.get("error_message"):
            errors += 1
        trigger = row.get("trigger") or []
        if isinstance(trigger, str):
            trigger = [trigger]
        if isinstance(trigger, list):
            for tag in trigger:
                trigger_counts[str(tag)] += 1
    group_sizes = [len(group) for group in sample_groups.values()]
    sample_probe_completeness = Counter(len({str(r.get("probe_name")) for r in g}) for g in sample_groups.values())
    recorded_summary = _json(summary_path) if summary_path.is_file() else None
    return {
        "source_files": {"jsonl": str(row_path), "summary": str(summary_path) if summary_path.exists() else None},
        "jsonl_sha256": _sha256(row_path),
        "rows": len(rows),
        "unique_episode_seed_sample_states": len(sample_groups),
        "rows_per_unique_state_min_max": [min(group_sizes), max(group_sizes)] if group_sizes else None,
        "unique_probe_names": sorted(probe_counts),
        "probe_rows_by_name": dict(sorted(probe_counts.items())),
        "unique_variant_count_per_state_distribution": dict(sorted(sample_probe_completeness.items())),
        "active_valid_rows": active_valid,
        "active_invalid_rows": active_invalid,
        "inactive_na_rows": inactive,
        "error_rows": errors,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "trigger_occurrence_counts": dict(sorted(trigger_counts.items())),
        "max_unique_states_per_episode_seed": max(
            Counter(seed for seed, _ in sample_groups).values(), default=0
        ),
        "recorded_summary": recorded_summary,
    }


def _arm_audit(root: Path, arm: str) -> dict[str, Any]:
    arm_dir = root / arm
    result_path = arm_dir / "evaluation_results.json"
    eval_obj = _json(result_path)
    identity = eval_obj.get("identity", {})
    records = eval_obj.get("episode_records", [])
    summary = eval_obj.get("summary", {})
    episode_path = arm_dir / "diagnostics" / "eval" / "episodes.jsonl"
    diag_summary_path = arm_dir / "diagnostics" / "eval" / "summary.json"
    diag_episodes = list(_jsonl(episode_path)) if episode_path.is_file() else []
    traffic = _traffic_map(arm_dir / "diagnostics" / "eval" / "traffic_selection.jsonl")
    cats = Counter(_outcome(row) for row in records)
    exclusive = sum(cats[o] for o in OUTCOMES)
    flags_count = Counter(
        sum(bool(row.get(o, False)) for o in OUTCOMES) for row in records
    )
    component_sums = []
    component_rows = 0
    for row in records:
        if all(row.get(k) is not None for k in REWARD_COMPONENTS):
            component_rows += 1
            component_sums.append(abs(float(row["episode_return"]) - sum(float(row[k]) for k in REWARD_COMPONENTS)))
    eval_seed_set = sorted(int(r["seed"]) for r in records)
    base_returns = [float(r["return_base"]) for r in diag_episodes if r.get("return_base") is not None]
    policy_returns = [float(r["return_policy"]) for r in diag_episodes if r.get("return_policy") is not None]
    base_counts = Counter(round(x, 8) for x in base_returns)
    expected_base_matches = 0
    expected_base_max_error = None
    per_seed_diag = {int(r["seed"]): r for r in diag_episodes if r.get("seed") is not None}
    base_errors = []
    for row in records:
        d = per_seed_diag.get(int(row["seed"]))
        if not d:
            continue
        n = int(row.get("raw_steps") or 0)
        sign = 1.0 if bool(row.get("success")) else -1.0 if bool(row.get("collision")) else 0.0
        expected = sign * (0.99 ** ((n - 1) % 3)) if n else 0.0
        err = abs(float(d.get("return_base", 0.0)) - expected)
        base_errors.append(err)
        if err <= 1e-9:
            expected_base_matches += 1
    if base_errors:
        expected_base_max_error = max(base_errors)
    lengths = [int(r["raw_steps"]) for r in records if r.get("raw_steps") is not None]
    decisions = [int(r["decision_steps"]) for r in records if r.get("decision_steps") is not None]
    success_raw = [int(r["raw_steps"]) for r in records if r.get("success") and r.get("raw_steps") is not None]
    success_seconds = [float(r["completion_time_seconds"]) for r in records if r.get("success") and r.get("completion_time_seconds") is not None]
    diag_summary = _json(diag_summary_path) if diag_summary_path.is_file() else {}
    checkpoint_path = Path(identity["checkpoint"]) if identity.get("checkpoint") else None
    checkpoint_actual_sha = _sha256(checkpoint_path) if checkpoint_path else None
    fingerprints = identity.get("model_policy_fingerprint") or []
    fingerprint_sha = [x.get("sha256") for x in fingerprints]
    fingerprint_updates = [x.get("updates") for x in fingerprints]
    fingerprint_frozen = (
        len(fingerprints) >= 2
        and len(set(fingerprint_sha)) == 1
        and len(set(fingerprint_updates)) == 1
    )
    traffic_matches_episode = 0
    traffic_missing = []
    variant_mismatch = []
    for r in records:
        seed = int(r["seed"])
        t = traffic.get(seed)
        if t is None:
            traffic_missing.append(seed)
        elif t.get("selected_traffic_path") and Path(t["selected_traffic_path"]).name != r.get("traffic_variant"):
            variant_mismatch.append(seed)
        else:
            traffic_matches_episode += 1
    behavior = {
        "episodes_file": str(episode_path),
        "episodes_file_sha256": _sha256(episode_path),
        "episodes": len(diag_episodes),
        "diagnostic_error_count_total": sum(int(r.get("diagnostic_error_count") or 0) for r in diag_episodes),
        "diagnostic_error_count_max_episode": max((int(r.get("diagnostic_error_count") or 0) for r in diag_episodes), default=0),
        "mean_return_base_decision_discounted": _mean(base_returns),
        "mean_return_policy_shaped": _mean(policy_returns),
        "return_base_counts_rounded_8dp": dict(sorted((str(k), v) for k, v in base_counts.items())),
        "return_base_expected_from_raw_terminal_and_repeat_discount": {
            "formula": "terminal_sign * 0.99**((raw_steps-1) % 3); raw terminal reward occurs on the last repeated raw step",
            "matches": expected_base_matches,
            "paired_rows": len(base_errors),
            "max_abs_error": expected_base_max_error,
        },
        "mean_episode_mean_actual_speed_mps": _mean(_float_values(diag_episodes, "mean_actual_speed_mps")),
        "mean_episode_stopped_fraction": _mean(_float_values(diag_episodes, "stopped_fraction_of_speed_samples")),
        "mean_episode_low_ttc_fraction": _mean(_float_values(diag_episodes, "low_ttc_fraction_of_evaluable_ticks")),
        "mean_episode_critical_unobserved_fraction": _mean(_float_values(diag_episodes, "critical_unobserved_fraction_of_covered_ticks")),
        "summary_error_count": diag_summary.get("diagnostic_error_count"),
    }
    return {
        "arm": arm,
        "source_files": {
            "evaluation_results": str(result_path),
            "evaluation_results_sha256": _sha256(result_path),
            "behavior_episodes": str(episode_path) if episode_path.exists() else None,
            "behavior_summary": str(diag_summary_path) if diag_summary_path.exists() else None,
            "traffic_selection": str(arm_dir / "diagnostics" / "eval" / "traffic_selection.jsonl") if (arm_dir / "diagnostics" / "eval" / "traffic_selection.jsonl").exists() else None,
        },
        "identity": {
            "method": identity.get("method"),
            "scenario": identity.get("scenario"),
            "depart_scale": identity.get("depart_scale"),
            "eval_traffic_split": identity.get("eval_traffic_split"),
            "checkpoint": identity.get("checkpoint"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "base_checkpoint_sha256": identity.get("base_checkpoint_sha256"),
            "eval_device": identity.get("eval_device"),
            "learning_or_replay_updates": identity.get("learning_or_replay_updates"),
            "policy_shadow_probes": identity.get("policy_shadow_probes"),
            "policy_shadow_critical_eval_sample": identity.get("policy_shadow_critical_eval_sample"),
            "model_policy_fingerprint": identity.get("model_policy_fingerprint"),
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.is_file()),
            "checkpoint_actual_sha256": checkpoint_actual_sha,
            "checkpoint_identity_sha_matches_file": bool(checkpoint_actual_sha and checkpoint_actual_sha == identity.get("checkpoint_sha256")),
            "policy_fingerprint_unchanged_across_eval": fingerprint_frozen,
            "fingerprint_updates_unchanged_across_eval": fingerprint_frozen,
        },
        "summary_as_recorded": summary,
        "recomputed": {
            "episodes": len(records),
            "seed_min_max": [min(eval_seed_set), max(eval_seed_set)] if eval_seed_set else None,
            "seed_count_unique": len(set(eval_seed_set)),
            "expected_seed_sequence_10000_10099": eval_seed_set == list(range(10000, 10100)),
            "outcome_counts": {o: cats[o] for o in OUTCOMES},
            "nonexclusive_or_missing_outcomes": cats["nonexclusive_or_missing"],
            "flag_sum_distribution": dict(sorted((str(k), v) for k, v in flags_count.items())),
            "episodes_with_complete_six_components": component_rows,
            "max_component_sum_abs_error": max(component_sums) if component_sums else None,
            "shaped_return_mean": _mean([float(r["episode_return"]) for r in records]),
            "shaped_return_population_std": _pstdev([float(r["episode_return"]) for r in records]),
            "raw_return_mean": _mean([float(r["raw_episode_return"]) for r in records if r.get("raw_episode_return") is not None]),
            "raw_return_population_std": _pstdev([float(r["raw_episode_return"]) for r in records if r.get("raw_episode_return") is not None]),
            "raw_return_source_counts": dict(Counter(str(r.get("raw_return_source")) for r in records)),
            "mean_raw_steps_all": _mean([float(v) for v in lengths]),
            "mean_decision_steps_all": _mean([float(v) for v in decisions]),
            "mean_raw_steps_success_only": _mean([float(v) for v in success_raw]),
            "mean_success_completion_seconds": _mean(success_seconds),
            "traffic_selection_present_and_variant_matches_eval_record": traffic_matches_episode,
            "traffic_selection_missing_seed_list": traffic_missing,
            "traffic_variant_mismatch_seed_list": variant_mismatch,
        },
        "behavior_diagnostics": behavior,
        "shadow_diagnostics": _shadow_audit(arm_dir),
        "route_veto_sidecar": _veto_audit(arm_dir),
    }


def _veto_audit(arm_dir: Path) -> dict[str, Any] | None:
    path = arm_dir / "diagnostics" / "eval" / "route_lane_veto_actions.jsonl"
    if not path.is_file():
        return None
    total = vetoed = speed_unchanged = fresh = known = returned_unchanged = 0
    reason_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    after_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    veto_seeds: Counter[int] = Counter()
    transform_exact = 0
    rows = 0
    for row in _jsonl(path):
        rows += 1
        did_veto = bool(row.get("veto_applied"))
        vetoed += int(did_veto)
        speed_unchanged += int(bool(row.get("speed_action_unchanged")))
        fresh += int(bool(row.get("context_fresh")))
        known += int(bool(row.get("context_known")))
        returned_unchanged += int(bool(row.get("reward_returned_unchanged")) and bool(row.get("terminated_returned_unchanged")) and bool(row.get("truncated_returned_unchanged")))
        reason_counts[str(row.get("reason", "missing"))] += 1
        requested_counts[str(row.get("lane_command_requested", "missing"))] += 1
        after_counts[str(row.get("lane_command_after_veto", "missing"))] += 1
        target_label_counts[str(row.get("target_lane_label", "unknown"))] += 1
        if did_veto and row.get("episode_seed") is not None:
            veto_seeds[int(row["episode_seed"])] += 1
        a0, a1 = row.get("action_requested"), row.get("action_after_veto")
        if did_veto and isinstance(a0, list) and isinstance(a1, list) and len(a0) == len(a1) == 2:
            if math.isclose(float(a0[0]), float(a1[0]), rel_tol=0.0, abs_tol=1e-12) and math.isclose(float(a1[1]), 0.0, rel_tol=0.0, abs_tol=1e-12):
                transform_exact += 1
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": rows,
        "veto_applied_rows": vetoed,
        "unique_episodes_with_veto": len(veto_seeds),
        "vetoes_by_episode_seed": dict(sorted((str(k), v) for k, v in veto_seeds.items())),
        "veto_reason_counts": dict(sorted(reason_counts.items())),
        "requested_lane_command_counts": dict(sorted(requested_counts.items())),
        "effective_lane_command_counts": dict(sorted(after_counts.items())),
        "target_lane_label_counts": dict(sorted(target_label_counts.items())),
        "context_fresh_rows": fresh,
        "context_known_rows": known,
        "speed_action_unchanged_rows": speed_unchanged,
        "reward_terminated_truncated_returned_unchanged_rows": returned_unchanged,
        "veto_rows_exactly_hold_lateral_and_preserve_speed": transform_exact,
        "interpretation_limit": "This sidecar records a policy request and the wrapper action; it does not show that a lane change physically occurred.",
    }


def _reference_reproduction(reference_path: Path, control_path: Path, root: Path) -> dict[str, Any]:
    ref_obj, ctl_obj = _json(reference_path), _json(control_path)
    ref = {int(r["seed"]): r for r in ref_obj.get("episode_records", [])}
    ctl = {int(r["seed"]): r for r in ctl_obj.get("episode_records", [])}
    seeds = sorted(set(ref) & set(ctl))
    fields = ("success", "collision", "timeout", "off_route", "episode_return", "raw_episode_return", "raw_steps", "decision_steps", "traffic_variant")
    same = Counter()
    max_abs = {"episode_return": 0.0, "raw_episode_return": 0.0}
    for seed in seeds:
        for field in fields:
            a, b = ref[seed].get(field), ctl[seed].get(field)
            if field in max_abs and a is not None and b is not None:
                max_abs[field] = max(max_abs[field], abs(float(a) - float(b)))
            if a == b or (field in max_abs and a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)):
                same[field] += 1
    ref_diag_path = reference_path.parent / "diagnostics" / "eval" / "episodes.jsonl"
    ctl_diag_path = control_path.parent / "diagnostics" / "eval" / "episodes.jsonl"
    ref_diag = {int(r["seed"]): r for r in _jsonl(ref_diag_path)} if ref_diag_path.is_file() else {}
    ctl_diag = {int(r["seed"]): r for r in _jsonl(ctl_diag_path)} if ctl_diag_path.is_file() else {}
    traffic_fields = (
        "traffic_variant", "source_traffic_roll", "high_density_overlay_sha256",
        "high_density_vehicle_scale", "high_density_pedestrian_scale",
        "traffic_partition", "traffic_partition_size", "source_endless_traffic",
    )
    traffic_same = 0
    traffic_unknown = []
    traffic_mismatch = []
    diag_equal = Counter()
    diag_compare_fields = (
        "return_base", "return_policy", "mean_actual_speed_mps",
        "stopped_fraction_of_speed_samples", "action_statistics",
    )
    for seed in seeds:
        a, b = ref_diag.get(seed), ctl_diag.get(seed)
        if a is None or b is None:
            traffic_unknown.append(seed)
            continue
        ia = a.get("reset_info") or {}
        ib = b.get("reset_info") or {}
        if all(ia.get(k) == ib.get(k) for k in traffic_fields):
            traffic_same += 1
        else:
            traffic_mismatch.append({"seed": seed, "reference": {k: ia.get(k) for k in traffic_fields}, "control": {k: ib.get(k) for k in traffic_fields}})
        for key in diag_compare_fields:
            if a.get(key) == b.get(key):
                diag_equal[key] += 1
    return {
        "reference_eval": str(reference_path),
        "control_eval": str(control_path),
        "reference_checkpoint_sha256": ref_obj.get("identity", {}).get("checkpoint_sha256"),
        "control_checkpoint_sha256": ctl_obj.get("identity", {}).get("checkpoint_sha256"),
        "matched_seeds": len(seeds),
        "exactly_equal_per_seed_fields": dict(same),
        "max_abs_return_difference": max_abs,
        "traffic_identity": {
            "exact_pairs_by_episode_reset_metadata": traffic_same,
            "unknown_pairs": traffic_unknown,
            "mismatches": traffic_mismatch,
            "fields": list(traffic_fields),
            "note": "The old reference run has no traffic_selection.jsonl; comparison uses its saved episode reset_info (including effective overlay SHA) against control reset_info.",
        },
        "exactly_equal_behavior_episode_fields": dict(diag_equal),
        "behavior_episode_fields_compared": list(diag_compare_fields),
    }


def build_audit(workspace: Path) -> dict[str, Any]:
    run_root = workspace / "runs" / "sortg3_causal_1002"
    reference_path = workspace / "runs" / "sortg3_1002" / "sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0" / "evaluation_results.json"
    status_path = run_root / "launcher_status.json"
    manifest_path = run_root / "experiment_manifest.json"
    result_arms = {arm: _arm_audit(run_root, arm) for arm in ARMS}
    pairings = {}
    traffic = {arm: _traffic_map(run_root / arm / "diagnostics" / "eval" / "traffic_selection.jsonl") for arm in ARMS}
    eval_rows = {arm: _json(run_root / arm / "evaluation_results.json").get("episode_records", []) for arm in ARMS}
    for left in ARMS:
        for right in ARMS:
            if left < right:
                pairings[f"{left}_to_{right}"] = _pair(eval_rows[left], eval_rows[right], traffic[left], traffic[right])
    reference_pair = _reference_reproduction(
        reference_path,
        run_root / "control" / "evaluation_results.json",
        run_root,
    )
    goaloff_diag_path = run_root / "goaloff" / "diagnostics" / "eval" / "episodes.jsonl"
    goaloff_diag = list(_jsonl(goaloff_diag_path))
    goaloff_eval = _json(run_root / "goaloff" / "evaluation_results.json").get("episode_records", [])
    goaloff_base_dist = Counter(round(float(r.get("return_base", 0.0)), 8) for r in goaloff_diag)
    raw_dist = Counter(round(float(r.get("raw_episode_return", 0.0)), 8) for r in goaloff_eval)
    return {
        "schema_version": "sorted_goalonly_causal_results_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_offline": True,
        "workspace_root": str(workspace),
        "run_root": str(run_root),
        "source_files": {
            "launcher_status": str(status_path),
            "launcher_status_sha256": _sha256(status_path),
            "experiment_manifest": str(manifest_path),
            "experiment_manifest_sha256": _sha256(manifest_path),
            "reference_evaluation_results": str(reference_path),
        },
        "suite_status": _json(status_path),
        "manifest_protocol": {
            "mode": _json(manifest_path).get("mode"),
            "scenario": _json(manifest_path).get("scenario"),
            "depart_scale": _json(manifest_path).get("depart_scale"),
            "evaluation_seed_start": _json(manifest_path).get("eval_seed_start"),
            "evaluation_episodes": _json(manifest_path).get("eval_episodes"),
            "eval_device": _json(manifest_path).get("eval_device"),
            "reference_checkpoint_sha256": _json(manifest_path).get("reference_model_sha256"),
            "reference_traffic_templates": _json(manifest_path).get("reference_traffic_templates"),
        },
        "arms": result_arms,
        "pairings": pairings,
        "frozen_control_reproduction_vs_original_formal_eval": reference_pair,
        "goaloff_return_semantics_reconciliation": {
            "evaluation_raw_mean_return": _json(run_root / "goaloff" / "evaluation_results.json")["summary"].get("raw_mean_return"),
            "evaluation_raw_return_source": "sum per raw SUMO step info.undiscounted_reward; terminal ±1; see algos/sb3_torch/evaluation.py::_raw_step_reward and evaluate_model_detailed",
            "evaluation_raw_return_value_counts": dict(sorted((str(k), v) for k, v in raw_dist.items())),
            "behavior_return_base_mean": _mean([float(r.get("return_base", 0.0)) for r in goaloff_diag]),
            "behavior_return_base_value_counts": dict(sorted((str(k), v) for k, v in goaloff_base_dist.items())),
            "behavior_return_base_definition": "sumo_env.py accumulates reward_discount**repeat_index * raw_reward within each action-repeat decision and passes that decision-level discounted_reward to recorder.on_decision_end; recorder sums those decision returns. Default action_repeat=3 and reward_discount=0.99. This is not the evaluation raw_episode_return.",
            "expected_return_base_formula": "terminal_sign * 0.99**((raw_steps-1) % 3)",
            "expected_formula_matches": result_arms["goaloff"]["behavior_diagnostics"]["return_base_expected_from_raw_terminal_and_repeat_discount"],
            "difference_is_denominator_or_transient": False,
        },
        "source_semantics": {
            "evaluation_raw_return": "algos/sb3_torch/evaluation.py lines 127-133 and 335-412; raw sums info.undiscounted_reward per raw env step",
            "decision_repeat_reward": "envs/sumo/sumo_env.py lines 448-472 and on_decision_end at 682; uses reward_discount**repeat_index within one action_repeat",
            "behavior_return_base": "fast-developer/behavior_diagnostics.py lines 488-494; sums discounted base reward once per policy decision",
            "shaped_return": "evaluation_results.summary.mean_return and episode_return sum reward returned by env.step; component reconciliation is checked separately",
        },
        "limitations": [
            "All three arms reuse one trained checkpoint and differ only during frozen-policy evaluation; this is not a multi-training-seed experiment.",
            "Episode pairing uses exact seed and the saved traffic selection tuple (index, roll, selected route-file SHA); this does not make the 30-template pool 100 independent traffic layouts.",
            "Shadow probes are counterfactual policy forwards at sampled states, not closed-loop rollouts or performance estimates.",
            "Route-veto sidecar reports requested versus wrapper-adjusted actions; it does not prove a physical lane transition.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Workspace root containing runs/ and Scene-Rep-Transformer-main/.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = build_audit(args.workspace.resolve())
    out = args.output or (Path(__file__).with_suffix(".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(out.resolve()), "arms": {k: v["recomputed"]["outcome_counts"] for k, v in result["arms"].items()}, "pairings": sorted(result["pairings"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
