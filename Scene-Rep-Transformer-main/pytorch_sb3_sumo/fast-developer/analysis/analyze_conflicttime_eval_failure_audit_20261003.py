"""Offline audit of the sealed ConflictTiming evaluation artifacts.

This script does not launch SUMO, load a policy, or modify experiment outputs.
Collision partner reconstruction replays the recorded center-based OBB test
against the saved final snapshot; it is not independent SUMO ground truth.
"""

from __future__ import annotations

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
EVAL = RUN / "diagnostics" / "eval"
TRAIN = RUN / "diagnostics" / "train"
ROUTEACT_RUN = WS / "runs" / "sortct_1002" / "sac_mlp_d1_st_rt_routeact_v1__intersection_sorted_depart4p0"
ROUTEACT_EVAL = ROUTEACT_RUN / "diagnostics" / "eval"
FROZEN_INTERVENTION = WS / "runs" / "sortct_frozen_1003" / "intervention"
OUT = Path(__file__).with_name("conflicttime_eval_failure_audit_20261003.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stats(values: Iterable[Any]) -> dict[str, Any]:
    vals = [x for value in values if (x := finite(value)) is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    vals.sort()
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p90": vals[max(0, math.ceil(0.9 * len(vals)) - 1)],
        "min": vals[0],
        "max": vals[-1],
    }


def outcome_name(outcome: Any) -> str:
    if isinstance(outcome, str):
        return outcome
    if isinstance(outcome, dict):
        for key in ("collision", "success", "timeout", "off_route"):
            if outcome.get(key):
                return {"collision": "collision", "success": "success", "timeout": "timeout", "off_route": "off_route"}[key]
    return "unknown"


def heading_axes(theta: float) -> tuple[tuple[float, float], tuple[float, float]]:
    forward = (math.cos(theta), math.sin(theta))
    side = (-forward[1], forward[0])
    return forward, side


def center_obb_overlap(a: dict[str, Any], b: dict[str, Any], leeway: float = 0.05) -> bool | None:
    """Replay the saved-center OBB SAT convention without bumper conversion."""
    try:
        ca = tuple(float(x) for x in a["position"])
        cb = tuple(float(x) for x in b["position"])
        fa, sa = heading_axes(float(a["heading"]))
        fb, sb = heading_axes(float(b["heading"]))
        la, wa = float(a["length"]), float(a["width"])
        lb, wb = float(b["length"]), float(b["width"])
    except (KeyError, TypeError, ValueError):
        return None
    delta = (cb[0] - ca[0], cb[1] - ca[1])
    for axis in (fa, sa, fb, sb):
        ra = 0.5 * la * abs(fa[0] * axis[0] + fa[1] * axis[1]) + 0.5 * wa * abs(sa[0] * axis[0] + sa[1] * axis[1])
        rb = 0.5 * lb * abs(fb[0] * axis[0] + fb[1] * axis[1]) + 0.5 * wb * abs(sb[0] * axis[0] + sb[1] * axis[1])
        projected = abs(delta[0] * axis[0] + delta[1] * axis[1])
        if projected > ra + rb + max(0.0, leeway):
            return False
    return True


def activation_row_map(representation_rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Representation metadata uses eval-global raw step, unlike prediction rows."""
    result = {}
    for row in representation_rows:
        metrics = row.get("metrics") or {}
        try:
            key = (
                int(metrics["evaluation_episode_index"]),
                int(metrics["evaluation_prediction_raw_step_index"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        result[key] = metrics
    return result


def summarize_activation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "route_conflict_timing_active",
        "route_conflict_timing_candidate_actor_count",
        "route_conflict_timing_valid_actor_count",
        "route_conflict_timing_coverage_fraction",
        "route_conflict_timing_delta_rms",
        "route_conflict_timing_base_rms_on_valid",
        "route_conflict_timing_delta_to_base_rms",
        "route_conflict_timing_delta_valid",
    )
    return {name: stats(row.get(name) for row in rows) for name in names}


def summarize_shadow_sampling(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    unique_samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            unique_samples.setdefault(str(sample_id), row)
    per_episode: Counter[str] = Counter()
    raw_steps = []
    sample_ids = sorted(unique_samples)
    for row in unique_samples.values():
        per_episode[str(row.get("episode_index", "unknown"))] += 1
        raw_value = row.get("pre_obs_raw") if phase == "train" else row.get("sampled_obs_raw_global_step")
        raw_step = finite(raw_value)
        if raw_step is not None:
            raw_steps.append(int(raw_step))
    raw_steps.sort()
    raw_step_gaps = [right - left for left, right in zip(raw_steps, raw_steps[1:])]
    active = [row for row in rows if row.get("applicable") is True]
    valid_active = [row for row in active if row.get("valid") is True]
    return {
        "rows_written": len(rows),
        "unique_policy_observation_states": len(unique_samples),
        "unique_states_per_episode_min": min(per_episode.values()) if per_episode else None,
        "unique_states_per_episode_max": max(per_episode.values()) if per_episode else None,
        "episodes_over_eval_cap_4": sum(count > 4 for count in per_episode.values()) if phase == "eval" else None,
        "sampled_observation_raw_step_count": len(raw_steps),
        "sampled_observation_raw_step_min": min(raw_steps) if raw_steps else None,
        "sampled_observation_raw_step_max": max(raw_steps) if raw_steps else None,
        "sampled_observation_raw_step_gap_min": min(raw_step_gaps) if raw_step_gaps else None,
        "sampled_observation_raw_step_gap_max": max(raw_step_gaps) if raw_step_gaps else None,
        "train_sample_ids_contiguous": (sample_ids == [f"train-{index:03d}" for index in range(len(sample_ids))]) if phase == "train" else None,
        "applicable_probe_rows": len(active),
        "valid_applicable_probe_rows": len(valid_active),
        "probe_error_or_invalid_rows": sum(row.get("valid") is not True and row.get("invalid_reason") != "branch_inactive" for row in rows),
        "branch_inactive_na_rows": sum(row.get("invalid_reason") == "branch_inactive" for row in rows),
        "stated_budget": "train <=20 unique states sampled at each 5000 raw-step boundary" if phase == "train" else "eval <=4 unique states per episode",
        "budget_check_passed": (len(unique_samples) <= 20 and max(raw_steps, default=0) <= 100000) if phase == "train" else all(count <= 4 for count in per_episode.values()),
    }


def summarize_shadow(shadow_rows: list[dict[str, Any]], activation_by_sample: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"rows_read": len(shadow_rows), "variants": {}, "all_probe_variants": {}, "by_critical_trigger": {}}
    variants = ("route_conflict_off", "route_conflict_times_off")

    def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        applicable = [row for row in rows if row.get("applicable") is True]
        valid = [row for row in applicable if row.get("valid") is True]
        l1 = [sum(abs(float(x)) for x in row.get("action_delta_abs_normalized", [])) for row in valid]
        linf = [max((abs(float(x)) for x in row.get("action_delta_abs_normalized", [])), default=0.0) for row in valid]
        l2 = [row.get("action_delta_l2_normalized") for row in valid]
        speed_delta = []
        lane_flip = 0
        lane_comparable = 0
        for row in valid:
            baseline = row.get("baseline_action_decoded") or []
            shadow = row.get("shadow_action_decoded") or []
            if baseline and shadow:
                speed_delta.append(abs(float(shadow[0]) - float(baseline[0])))
            if len(baseline) > 1 and len(shadow) > 1:
                lane_comparable += 1
                lane_flip += int(shadow[1] != baseline[1])
        return {
            "rows": len(rows),
            "unique_sample_ids": len({row.get("sample_id") for row in rows}),
            "applicable_rows": len(applicable),
            "valid_rows": len(valid),
            "invalid_or_inapplicable_reasons": dict(Counter(str(row.get("invalid_reason") or "missing") for row in rows if row not in valid)),
            "action_delta_l1_normalized": stats(l1),
            "action_delta_linf_normalized": stats(linf),
            "action_delta_l2_normalized": stats(l2),
            "nonzero_action_delta_states": sum(value > 1e-9 for value in l1),
            "nonzero_action_delta_fraction_of_valid": (sum(value > 1e-9 for value in l1) / len(valid)) if valid else None,
            "decoded_target_speed_absolute_delta_mps": stats(speed_delta),
            "decoded_lane_command_changed": lane_flip,
            "decoded_lane_command_comparable_states": lane_comparable,
            "decoded_lane_command_changed_fraction": (lane_flip / lane_comparable) if lane_comparable else None,
            "q_or_value_delta_logged": any(any(key.lower().startswith(("q1", "q2", "q_", "value")) for key in row) for row in valid),
        }

    for variant in variants:
        subset = [row for row in shadow_rows if row.get("probe_name") == variant]
        result["variants"][variant] = summarize_group(subset)
        result["variants"][variant]["by_terminal_outcome"] = {
            label: summarize_group([row for row in subset if outcome_name(row.get("outcome")) == label])
            for label in ("success", "collision", "timeout")
        }

    # Keep the normal-flow ST/RT/Topo/slot probes in the same-observation
    # inventory, including explicit branch-inactive NA rows. Inactive probes
    # are not represented as zero-sensitivity results.
    for variant in sorted({str(row.get("probe_name")) for row in shadow_rows if row.get("probe_name") is not None}):
        subset = [row for row in shadow_rows if row.get("probe_name") == variant]
        result["all_probe_variants"][variant] = summarize_group(subset)

    trigger_names = sorted({str(trigger) for row in shadow_rows for trigger in (row.get("trigger") or [])})
    for trigger in trigger_names:
        result["by_critical_trigger"][trigger] = {}
        for variant in variants:
            group = [row for row in shadow_rows if row.get("probe_name") == variant and trigger in (row.get("trigger") or [])]
            summary = summarize_group(group)
            matched_metrics = []
            for row in group:
                try:
                    key = (int(row["episode_index"]), int(row["sampled_obs_raw_global_step"]))
                except (KeyError, TypeError, ValueError):
                    continue
                metric = activation_by_sample.get(key)
                if metric is not None:
                    matched_metrics.append(metric)
            summary["sampled_state_activation_join"] = {
                "joined_states": len(matched_metrics),
                "expected_valid_rows": len(group),
                "activation": summarize_activation(matched_metrics),
                "join_key": "(episode_index, sampled_obs_raw_global_step) == (evaluation_episode_index, evaluation_prediction_raw_step_index)",
            }
            result["by_critical_trigger"][trigger][variant] = summary
    result["limitations"] = [
        "Shadow rows compare deterministic actions on the exact same observation; they do not step the environment and do not identify performance causality.",
        "Q/value deltas are not emitted by the shadow schema, so value-function sensitivity is unknown.",
        "Critical-trigger groups overlap by design; a state may satisfy more than one trigger.",
    ]
    return result


def episode_metric_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "raw_steps", "decisions", "mean_actual_speed_mps", "stopped_fraction_of_speed_samples",
        "stopped_seconds", "route_lane_ineligible_seconds", "route_lane_ineligible_stopped_seconds",
        "route_lane_ineligible_stall_max_seconds", "min_cv_obb_ttc_s", "risk_evaluable_low_ttc_ticks",
        "critical_unobserved_fraction_of_covered_ticks", "cv_ttc_below_3s_seconds",
    )
    result = {}
    for label in ("success", "collision", "timeout"):
        subset = [row for row in episodes if outcome_name(row.get("outcome")) == label]
        result[label] = {
            "episode_count": len(subset),
            "metrics": {name: stats(row.get(name) for row in subset) for name in fields},
        }
    return result


def summarize_calibration_records(rows: list[dict[str, Any]], raw_step_duration_s: float = 0.1) -> dict[str, Any]:
    path_status = {actor: Counter() for actor in ("ego", "foe")}
    joint_path_status = Counter()
    components: dict[str, dict[str, Any]] = {}
    encounter_keys = set()
    for row in rows:
        prediction = row.get("prediction") or {}
        encounter_keys.add((prediction.get("episode_index"), prediction.get("foe_key")))
        statuses = row.get("actor_path_status") or {}
        joint_path_status[(str(statuses.get("ego", "missing")), str(statuses.get("foe", "missing")))] += 1
        for actor in ("ego", "foe"):
            path_status[actor][str(statuses.get(actor, "missing"))] += 1

    for actor in ("ego", "foe"):
        prefix = "ego" if actor == "ego" else "foe"
        predicted_valid_count = 0
        entry_present = entry_left_censored = 0
        clearance_present = clearance_left_censored = 0
        deviation_entry_missing = horizon_entry_missing = terminal_entry_missing = 0
        deviation_clearance_missing = horizon_clearance_missing = terminal_clearance_missing = 0
        entry_interval_errors = []
        clearance_interval_errors = []
        for row in rows:
            prediction = row.get("prediction") or {}
            predicted = (prediction.get("predicted") or {}).get(f"{prefix}_time") or {}
            if predicted.get("valid") is True:
                predicted_valid_count += 1
            entry = (row.get("entry_intervals") or {}).get(actor)
            clearance = (row.get("clearance_intervals") or {}).get(actor)
            status = str(row.get("outcome_status", "missing"))
            actor_status = str((row.get("actor_path_status") or {}).get(actor, "missing"))
            pred_raw = finite(prediction.get("raw_step_pre_action"))
            if entry is not None:
                entry_present += 1
                left = entry.get("left_censored_at_prediction") is True
                entry_left_censored += int(left)
                pred_t = finite(predicted.get("entry_s"))
                lo = finite(entry.get("lower_raw_step"))
                hi = finite(entry.get("upper_raw_step"))
                if not left and predicted.get("valid") is True and pred_t is not None and pred_raw is not None and lo is not None and hi is not None:
                    obs_lo = (lo - pred_raw) * raw_step_duration_s
                    obs_hi = (hi - pred_raw) * raw_step_duration_s
                    entry_interval_errors.append(max(obs_lo - pred_t, 0.0, pred_t - obs_hi))
            elif actor_status == "route_or_candidate_path_deviation":
                deviation_entry_missing += 1
            elif status == "prediction_horizon_right_censored":
                horizon_entry_missing += 1
            elif status == "episode_terminal_right_censored":
                terminal_entry_missing += 1

            if clearance is not None:
                clearance_present += 1
                clearance_left_censored += int(clearance.get("left_censored_at_prediction") is True)
                pred_t = finite(predicted.get("exit_s"))
                lo = finite(clearance.get("lower_raw_step"))
                hi = finite(clearance.get("upper_raw_step"))
                if clearance.get("left_censored_at_prediction") is not True and predicted.get("valid") is True and pred_t is not None and pred_raw is not None and lo is not None and hi is not None:
                    obs_lo = (lo - pred_raw) * raw_step_duration_s
                    obs_hi = (hi - pred_raw) * raw_step_duration_s
                    clearance_interval_errors.append(max(obs_lo - pred_t, 0.0, pred_t - obs_hi))
            elif actor_status == "route_or_candidate_path_deviation":
                deviation_clearance_missing += 1
            elif status == "prediction_horizon_right_censored":
                horizon_clearance_missing += 1
            elif status == "episode_terminal_right_censored":
                terminal_clearance_missing += 1

        components[actor] = {
            "prediction_time_valid_rows": predicted_valid_count,
            "entry_interval_rows_available": entry_present,
            "entry_interval_left_censored_rows": entry_left_censored,
            "entry_interval_missing_by_reason": {
                "actor_path_deviation": deviation_entry_missing,
                "prediction_horizon_right_censor": horizon_entry_missing,
                "episode_terminal_right_censor": terminal_entry_missing,
            },
            "entry_prediction_error_to_observed_interval_seconds": stats(entry_interval_errors),
            "clearance_interval_rows_available": clearance_present,
            "clearance_interval_left_censored_rows": clearance_left_censored,
            "clearance_interval_missing_by_reason": {
                "actor_path_deviation": deviation_clearance_missing,
                "prediction_horizon_right_censor": horizon_clearance_missing,
                "episode_terminal_right_censor": terminal_clearance_missing,
            },
            "clearance_prediction_error_to_observed_interval_seconds": stats(clearance_interval_errors),
        }
    return {
        "rows_are_repeated_decision_time_predictions": True,
        "calibration_rows": len(rows),
        "unique_episode_foe_encounters": len(encounter_keys),
        "actor_path_status_by_component": {actor: dict(counts) for actor, counts in path_status.items()},
        "joint_actor_path_status_counts": {f"ego={ego};foe={foe}": count for (ego, foe), count in sorted(joint_path_status.items())},
        "interval_coverage_by_component": components,
        "interval_error_definition": "absolute distance in seconds from constant-speed predicted entry/exit to the observed raw-step bracket; zero when prediction falls inside the bracket. Left-censored actual entries and absent/censored intervals are excluded, not scored as zero.",
    }


def read_episode_map(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["seed"]): row for row in read_rows(path) if row.get("seed") is not None}


def summarize_baseline_collision_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique = [row for row in rows if row.get("partner_join_status") == "unique_saved_center_obb_overlap_candidate"]
    ct = [row.get("ct_partner_row") or {} for row in unique]
    return {
        "collision_episode_records": len(rows),
        "unique_partner_reconstruction_count": len(unique),
        "ambiguous_or_missing_partner_count": len(rows) - len(unique),
        "supported_crossing_partner_rows": sum(item.get("relation_valid") is True for item in ct),
        "no_supported_crossing_partner_rows": sum(item.get("reason") == "no_supported_crossing_in_known_geometry_horizon" for item in ct),
        "ego_eta_valid": sum(item.get("ego_time_valid") is True for item in ct),
        "ego_eta_low_speed_na": sum(item.get("ego_time_reason") == "speed_missing_or_below_0p2_mps" for item in ct),
        "partner_eta_valid": sum(item.get("partner_time_valid") is True for item in ct),
        "ego_pre_action_speed_mps": stats(item.get("ego_speed_mps_pre_action") for item in ct),
        "partner_pre_action_speed_mps": stats(item.get("partner_speed_mps_pre_action") for item in ct),
    }


def summarize_frozen_outcome_pairing(routeact_base: dict[int, dict[str, Any]],
                                      conflict_base: dict[int, dict[str, Any]],
                                      branch_episodes: dict[str, dict[int, dict[str, Any]]],
                                      collision_records: list[dict[str, Any]]) -> dict[str, Any]:
    target_seeds = set(range(10000, 10040))
    routeact = {seed: row for seed, row in routeact_base.items() if seed in target_seeds}
    conflict = {seed: row for seed, row in conflict_base.items() if seed in target_seeds}
    collision_by_seed = {int(row["seed"]): row for row in collision_records if row.get("seed") is not None}

    def pair_branch(base: dict[int, dict[str, Any]], name: str, include_conflict_context: bool,
                    expected_method: str | None, expected_sha256: str | None) -> dict[str, Any]:
        episodes = branch_episodes.get(name, {})
        episodes = {seed: row for seed, row in episodes.items() if seed in target_seeds}
        common = sorted(set(base) & set(episodes))
        matrix = Counter((outcome_name(base[seed].get("outcome")), outcome_name(episodes[seed].get("outcome"))) for seed in common)
        old_collision_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if include_conflict_context:
            for seed in common:
                if outcome_name(base[seed].get("outcome")) == "collision" and seed in collision_by_seed:
                    old_collision_groups[outcome_name(episodes[seed].get("outcome"))].append(collision_by_seed[seed])
        identity_path = FROZEN_INTERVENTION / name / "evaluation_results.json"
        identity = (read_json(identity_path).get("identity") or {}) if identity_path.exists() else {}
        loaded_checkpoint_method = identity.get("checkpoint_method") or identity.get("method")
        if identity.get("checkpoint_sha256") != expected_sha256 or loaded_checkpoint_method != expected_method:
            raise ValueError(
                f"Refusing mismatched checkpoint join for {name}: "
                f"expected ({expected_method}, {expected_sha256}), got "
                f"({loaded_checkpoint_method}, {identity.get('checkpoint_sha256')})"
            )
        return {
            "paired_seed_count": len(common),
            "missing_from_branch": sorted(set(base) - set(episodes)),
            "extra_branch_seeds_in_target_range": sorted(set(episodes) - set(base)),
            "intervention_identity": {key: identity.get(key) for key in ("method", "checkpoint_sha256", "diagnostic_arm", "intervention")},
            "outcome_transition_counts": {f"{old}->{new}": count for (old, new), count in sorted(matrix.items())},
            "original_collision_context_by_branch_outcome": {
                label: summarize_baseline_collision_context(group)
                for label, group in sorted(old_collision_groups.items())
            },
            "baseline_success_to_new_collision_context": "No collision partner exists in the baseline episode for this transition; CT contact coverage is unknown from a successful base run and is not inferred.",
        }

    routeact_identity = read_json(ROUTEACT_RUN / "evaluation_results.json").get("identity") or {}
    conflict_identity = read_json(RUN / "evaluation_results.json").get("identity") or {}
    a_result = pair_branch(routeact, "a_no_veto", include_conflict_context=False,
                           expected_method=routeact_identity.get("method"), expected_sha256=routeact_identity.get("checkpoint_sha256"))
    b_results = {
        name: pair_branch(conflict, name, include_conflict_context=True,
                          expected_method=conflict_identity.get("method"), expected_sha256=conflict_identity.get("checkpoint_sha256"))
        for name in ("b_conflict_off", "b_times_off")
    }
    return {
        "seed_range": [10000, 10039],
        "A_routeact_no_veto": {
            "baseline_method": routeact_identity.get("method"),
            "baseline_checkpoint_sha256": routeact_identity.get("checkpoint_sha256"),
            "paired_result": a_result,
        },
        "B_conflict_timing": {
            "baseline_method": conflict_identity.get("method"),
            "baseline_checkpoint_sha256": conflict_identity.get("checkpoint_sha256"),
            "baseline_episodes_in_seed_range": len(conflict),
            "branches": b_results,
        },
        "interpretation_limits": [
            "The `a_no_veto` arm loads the routeact checkpoint and must be paired against A's routeact reference eval. The B branch-off arms load the ConflictTiming checkpoint and must be paired against B's full-branch eval; cross-method pairing is not used.",
            "Frozen policy interventions compare the same held-out traffic seed set but trajectories diverge after action changes; this is a finite paired evaluation, not an independent training-seed estimate.",
            "Baseline collision partner context is joined only to the terminal pre-action state of the matching unmodified B episode. It describes cues in that baseline failure, not the later altered trajectory.",
            "A rescued collision associated with an available CT row does not prove the branch caused the crash; branch-off changes all downstream state/action occupancy.",
        ],
    }


def analyze() -> dict[str, Any]:
    train_episode_rows = list(read_rows(TRAIN / "episodes.jsonl"))
    train_calibration_rows = list(read_rows(TRAIN / "task_conflict_calibration.jsonl.gz"))
    train_calibration_outcomes = Counter(str(row.get("outcome_status", "missing")) for row in train_calibration_rows)
    train_calibration_pairs = {
        (int((row.get("prediction") or {}).get("episode_index", -1)),
         str((row.get("prediction") or {}).get("foe_key", "")))
        for row in train_calibration_rows
    }
    train_capture_summary = read_json(TRAIN / "summary.json")
    train_timing_summary = read_json(TRAIN / "task_conflict_timing_summary.json")
    train_optimization = list(read_rows(TRAIN / "optimization.jsonl"))
    train_shadow_rows = list(read_rows(TRAIN / "policy_shadow_probes.jsonl"))
    episode_rows = list(read_rows(EVAL / "episodes.jsonl"))
    base_episode_by_seed = {int(row["seed"]): row for row in episode_rows if row.get("seed") is not None}
    routeact_episode_by_seed = read_episode_map(ROUTEACT_EVAL / "episodes.jsonl")
    frozen_episode_maps = {
        name: read_episode_map(FROZEN_INTERVENTION / name / "diagnostics" / "eval" / "episodes.jsonl")
        for name in ("a_no_veto", "b_conflict_off", "b_times_off")
    }
    decision_rows = list(read_rows(EVAL / "decisions.jsonl.gz"))
    prediction_rows = list(read_rows(EVAL / "task_conflict_predictions.jsonl.gz"))
    calibration_rows = list(read_rows(EVAL / "task_conflict_calibration.jsonl.gz"))
    representation_rows = list(read_rows(EVAL / "representation.jsonl"))
    shadow_rows = list(read_rows(EVAL / "policy_shadow_probes.jsonl"))
    routeact_shadow_rows = list(read_rows(ROUTEACT_EVAL / "policy_shadow_probes.jsonl"))
    routeact_representation_rows = list(read_rows(ROUTEACT_EVAL / "representation.jsonl"))
    routeact_train_shadow_rows = list(read_rows(ROUTEACT_RUN / "diagnostics" / "train" / "policy_shadow_probes.jsonl"))
    raw_collision_rows = [
        row for row in read_rows(EVAL / "raw_steps.jsonl.gz")
        if (row.get("events") or {}).get("geometric_collision") is True
    ]

    ordered_episodes = sorted(episode_rows, key=lambda row: int(row.get("episode", 0)))
    episode_id_to_index = {int(row.get("episode", -1)): index for index, row in enumerate(ordered_episodes)}
    episode_by_id = {int(row.get("episode", -1)): row for row in ordered_episodes}
    global_raw_start: dict[int, int] = {}
    cumulative_global_raw = 0
    for index, row in enumerate(ordered_episodes):
        global_raw_start[index] = cumulative_global_raw
        cumulative_global_raw += int(row.get("raw_steps", 0) or 0)

    decisions_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    decision_by_local_pre: dict[tuple[int, int], dict[str, Any]] = {}
    local_clock: Counter[int] = Counter()
    for row in decision_rows:
        episode_id = int(row.get("episode", -1))
        row = dict(row)
        row["_local_pre_raw_step"] = local_clock[episode_id]
        decision_by_local_pre[(episode_id, local_clock[episode_id])] = row
        decisions_by_episode[episode_id].append(row)
        local_clock[episode_id] += int(row.get("raw_ticks", 0) or 0)

    predictions_by_episode_raw = {}
    for row in prediction_rows:
        try:
            key = (int(row["episode_index"]), int(row["raw_step_pre_action"]))
        except (KeyError, TypeError, ValueError):
            continue
        predictions_by_episode_raw[key] = row

    calibration_by_episode_raw_foe: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    calibration_episode_foe_encounters: set[tuple[int, str]] = set()
    calibration_outcomes = Counter()
    for row in calibration_rows:
        meta = row.get("prediction") or {}
        try:
            key = (int(meta["episode_index"]), int(meta["raw_step_pre_action"]), str(meta["foe_key"]))
        except (KeyError, TypeError, ValueError):
            continue
        calibration_by_episode_raw_foe[key].append(row)
        calibration_episode_foe_encounters.add((key[0], key[2]))
        calibration_outcomes[str(row.get("outcome_status", "missing"))] += 1

    activation_by_sample = activation_row_map(representation_rows)
    representation_by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representation_rows:
        metric = row.get("metrics") or {}
        try:
            episode_index = int(metric["evaluation_episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        episode_id = ordered_episodes[episode_index].get("episode") if 0 <= episode_index < len(ordered_episodes) else None
        episode = episode_by_id.get(int(episode_id)) if episode_id is not None else None
        if episode is not None:
            representation_by_outcome[outcome_name(episode.get("outcome"))].append(metric)

    shadow = summarize_shadow(shadow_rows, activation_by_sample)
    routeact_shadow = summarize_shadow(routeact_shadow_rows, activation_row_map(routeact_representation_rows))
    shadow["per_outcome_activation_for_all_eval_policy_predictions"] = {
        label: {
            "prediction_state_count": len(rows),
            "activation": summarize_activation(rows),
        }
        for label, rows in sorted(representation_by_outcome.items())
    }

    collision_records = []
    candidate_count = Counter()
    collision_location = Counter()
    partner_seen = Counter()
    partner_input = Counter()
    partner_ct_status = Counter()
    partner_calibration_status = Counter()
    collision_state_rows = []
    terminal_behavior_by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    terminal_ct_by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_decision_pre_matches = 0
    exact_prediction_pre_matches = 0
    for episode in ordered_episodes:
        episode_id = int(episode.get("episode", -1))
        eidx = episode_id_to_index.get(episode_id, -1)
        outcome = outcome_name(episode.get("outcome"))
        decisions = decisions_by_episode.get(episode_id, [])
        last_decision = decisions[-1] if decisions else None
        snap = episode.get("last_snapshot") or {}
        ego = snap.get("ego") or {}
        control = (last_decision or {}).get("control") or {}
        pre_raw = (last_decision or {}).get("_local_pre_raw_step")
        observed_raw = snap.get("observed_neighbor_raw_step")
        if observed_raw is not None and pre_raw == int(observed_raw):
            exact_decision_pre_matches += 1
        if pre_raw is None:
            pre_raw = observed_raw
        prediction = predictions_by_episode_raw.get((eidx, int(pre_raw))) if pre_raw is not None else None
        if prediction is not None:
            exact_prediction_pre_matches += 1

        terminal_prediction = prediction
        terminal_details = (terminal_prediction or {}).get("pre_action_predictions") or {}
        terminal_metrics = activation_by_sample.get(
            (eidx, global_raw_start.get(eidx, 0) + int(pre_raw or 0)),
            {},
        )
        terminal_feature_rows = (terminal_prediction or {}).get("pre_action_features") or []
        terminal_ct_by_outcome[outcome].append({
            "prediction_available": terminal_prediction is not None,
            "actor_slots": max(0, len((terminal_prediction or {}).get("actor_keys_in_trajectory_order") or []) - 1),
            "feature_valid_actor_rows": sum(
                isinstance(row, list) and bool(row) and finite(row[0]) is not None and float(row[0]) > 0.5
                for row in terminal_feature_rows[1:]
            ),
            "valid_relation_count": terminal_details.get("valid_relation_count"),
            "valid_joint_timing_count": terminal_details.get("valid_joint_timing_count"),
            "route_conflict_timing_active": terminal_metrics.get("route_conflict_timing_active"),
            "route_conflict_timing_valid_actor_count": terminal_metrics.get("route_conflict_timing_valid_actor_count"),
            "route_conflict_timing_candidate_actor_count": terminal_metrics.get("route_conflict_timing_candidate_actor_count"),
            "route_conflict_timing_coverage_fraction": terminal_metrics.get("route_conflict_timing_coverage_fraction"),
            "route_conflict_timing_delta_rms": terminal_metrics.get("route_conflict_timing_delta_rms"),
            "route_conflict_timing_delta_to_base_rms": terminal_metrics.get("route_conflict_timing_delta_to_base_rms"),
        })

        terminal_behavior_by_outcome[outcome].append({
            "requested_target_speed_mps": control.get("target_speed_mps", (snap.get("speed_control") or {}).get("requested_target_speed_mps")),
            "terminal_actual_speed_mps": ego.get("speed"),
            "current_lane_can_reach_next_edge": control.get("current_lane_can_reach_next_edge"),
            "target_lane_can_reach_next_edge": control.get("expected_target_lane_can_reach_next_edge"),
            "lane_command_requested": control.get("lane_command_requested"),
            "lane_control_request_status": control.get("lane_control_request_status"),
            "lane_control_request_reason": control.get("lane_control_request_reason"),
            "lane_change_applied": control.get("lane_change_applied"),
        })

        if outcome != "collision":
            continue

        evidence_rows = episode.get("collision_evidence") or []
        evidence = evidence_rows[-1] if evidence_rows else {}
        terminal_raw = int(snap.get("raw_step", -1))
        exact_raw = next(
            (
                row for row in raw_collision_rows
                if int(row.get("episode", -2)) == episode_id
                and int(row.get("raw_step", -3)) == terminal_raw
            ),
            None,
        )
        collision_snapshot = exact_raw or snap
        collision_ego = collision_snapshot.get("ego") or ego
        vehicles = collision_snapshot.get("vehicles") or []
        candidates = [vehicle for vehicle in vehicles if center_obb_overlap(collision_ego, vehicle) is True]
        candidate_count[str(len(candidates))] += 1
        if len(candidates) != 1:
            collision_records.append({
                "episode": episode_id,
                "seed": episode.get("seed"),
                "terminal_raw_step": collision_snapshot.get("raw_step"),
                "same_tick_geometric_collision_raw_row_found": exact_raw is not None,
                "partner_join_status": "ambiguous_multiple_geometry_candidates" if candidates else "no_geometry_candidate_in_saved_snapshot",
                "candidate_count": len(candidates),
                "saved_vehicle_count": len(vehicles),
                "neighbors_omitted": collision_snapshot.get("neighbors_omitted"),
                "candidate_ids": [vehicle.get("id") for vehicle in candidates],
            })
            continue

        partner = candidates[0]
        partner_key = partner.get("key")
        collision_location[(collision_ego.get("road_id"), partner.get("road_id"))] += 1
        observed_ids = set(collision_snapshot.get("observed_neighbor_ids") or [])
        snapshot_observed = partner_key in observed_ids
        partner_seen["snapshot_observed_by_policy_true"] += int(partner.get("observed_by_policy") is True)
        partner_seen["in_saved_observed_neighbor_ids"] += int(snapshot_observed)

        actor_index = None
        actor_present = False
        feature = None
        detail = None
        coverage = None
        if prediction is not None:
            keys = prediction.get("actor_keys_in_trajectory_order") or []
            if partner_key in keys:
                actor_present = True
                actor_index = keys.index(partner_key)
                partner_input["partner_key_in_exact_pre_action_actor_order"] += 1
                feature_rows = prediction.get("pre_action_features") or []
                feature = feature_rows[actor_index] if actor_index < len(feature_rows) else None
                details = prediction.get("pre_action_predictions", {}).get("actors") or []
                detail = next((row for row in details if row.get("actor_key") == partner_key and int(row.get("actor_index", -1)) == actor_index), None)
                cov_rows = prediction.get("actor_geometry_and_map_coverage") or []
                coverage = next((row for row in cov_rows if row.get("key") == partner_key and int(row.get("actor_index", -1)) == actor_index), None)
            else:
                partner_input["partner_key_absent_from_exact_pre_action_actor_order"] += 1
        else:
            partner_input["no_exact_pre_action_prediction_row"] += 1

        relation_valid = bool(detail and detail.get("relation_valid"))
        if relation_valid:
            partner_ct_status["supported_crossing_row"] += 1
        elif detail is not None:
            partner_ct_status["actor_row_present_but_no_supported_crossing"] += 1
        elif actor_present:
            partner_ct_status["actor_present_but_detail_missing"] += 1
        else:
            partner_ct_status["no_partner_actor_row"] += 1

        selected = (detail or {}).get("selected") or {}
        ego_coverage = None
        if prediction is not None:
            ego_coverage = next(
                (
                    row for row in (prediction.get("actor_geometry_and_map_coverage") or [])
                    if row.get("key") == "vehicle:ego" and int(row.get("actor_index", -1)) == 0
                ),
                None,
            )
        calibration_matches = calibration_by_episode_raw_foe.get((eidx, int(pre_raw), str(partner_key)), []) if pre_raw is not None else []
        calibration_status = [row.get("outcome_status") for row in calibration_matches]
        partner_calibration_status.update(str(status) for status in calibration_status)
        selected_summary = {
            "relation_valid": relation_valid,
            "reason": (detail or {}).get("reason"),
            "geometry_status": (detail or {}).get("geometry_status"),
            "candidate_pair_count": (detail or {}).get("candidate_pair_count"),
            "supported_candidate_pair_count": (detail or {}).get("supported_candidate_pair_count"),
            "actor_index": actor_index,
            "feature_valid_bit": feature[0] if isinstance(feature, list) and feature else None,
            "feature_ego_time_valid_bit": feature[11] if isinstance(feature, list) and len(feature) > 12 else None,
            "feature_foe_time_valid_bit": feature[12] if isinstance(feature, list) and len(feature) > 12 else None,
            "ego_speed_mps_pre_action": selected.get("ego_speed_mps", (ego_coverage or {}).get("speed_mps")),
            "partner_speed_mps_pre_action": selected.get("foe_speed_mps", (coverage or {}).get("speed_mps")),
            "ego_entry_time_s": selected.get("ego_entry_time_s"),
            "ego_exit_time_s": selected.get("ego_exit_time_s"),
            "ego_time_valid": selected.get("ego_time_valid"),
            "ego_time_reason": (selected.get("ego_time") or {}).get("reason"),
            "partner_entry_time_s": selected.get("foe_entry_time_s"),
            "partner_exit_time_s": selected.get("foe_exit_time_s"),
            "partner_time_valid": selected.get("foe_time_valid"),
            "partner_time_reason": (selected.get("foe_time") or {}).get("reason"),
            "beyond_calibration_horizon": selected.get("beyond_calibration_horizon"),
            "calibration_outcome_status": calibration_status,
            "calibration_pair_row_count": len(calibration_matches),
        }
        record = {
            "episode": episode_id,
            "seed": episode.get("seed"),
            "episode_index_in_sidecars": eidx,
            "terminal_raw_step": collision_snapshot.get("raw_step"),
            "terminal_pre_action_raw_step": pre_raw,
            "observed_neighbor_raw_step": collision_snapshot.get("observed_neighbor_raw_step"),
            "same_tick_geometric_collision_raw_row_found": exact_raw is not None,
            "same_tick_snapshot_vehicle_count": len(vehicles),
            "same_tick_neighbors_omitted": collision_snapshot.get("neighbors_omitted"),
            "terminal_state_time_s": collision_snapshot.get("sim_time"),
            "partner_join_status": "unique_saved_center_obb_overlap_candidate",
            "partner_id": partner.get("id"),
            "partner_key": partner_key,
            "ego_terminal_road_lane": [collision_ego.get("road_id"), collision_ego.get("lane_id")],
            "partner_terminal_road_lane": [partner.get("road_id"), partner.get("lane_id")],
            "ego_terminal_speed_mps": collision_ego.get("speed"),
            "partner_terminal_speed_mps": partner.get("speed"),
            "partner_observed_by_policy_saved_flag": partner.get("observed_by_policy"),
            "partner_in_saved_observed_neighbor_ids": snapshot_observed,
            "partner_in_exact_pre_action_actor_order": actor_present,
            "target_speed_mps_for_final_action": control.get("target_speed_mps"),
            "final_action_speed_abs_tracking_error_mps": (
                abs(float(collision_ego["speed"]) - float(control["target_speed_mps"]))
                if finite(collision_ego.get("speed")) is not None and finite(control.get("target_speed_mps")) is not None else None
            ),
            "last_action_lane_context": {
                "planned_next_edge": control.get("planned_next_edge"),
                "current_lane_id": control.get("current_lane_id"),
                "current_lane_can_reach_next_edge": control.get("current_lane_can_reach_next_edge"),
                "target_lane_id": control.get("expected_target_lane_id"),
                "target_lane_can_reach_next_edge": control.get("expected_target_lane_can_reach_next_edge"),
                "lane_command_requested": control.get("lane_command_requested"),
                "lane_control_request_status": control.get("lane_control_request_status"),
                "lane_control_request_reason": control.get("lane_control_request_reason"),
                "lane_change_applied": control.get("lane_change_applied"),
            },
            "completed_action_window_min_cv_obb_ttc_s": last_decision.get("min_cv_obb_ttc_s") if last_decision else None,
            "ct_partner_row": selected_summary,
            "terminal_event_protocol": ((collision_snapshot.get("events") or {}).get("terminal_outcome_protocol")),
            "raw_sumo_collision": (collision_snapshot.get("events") or {}).get("raw_sumo_collision"),
            "geometric_collision": (collision_snapshot.get("events") or {}).get("geometric_collision"),
            "event_collision_ids": evidence.get("collision_ids"),
            "event_collision_ids_available": bool(evidence.get("collision_ids")),
            "event_note": evidence.get("note"),
        }
        collision_records.append(record)
        collision_state_rows.append(record)

    collision_metrics = {
        "episodes": sum(outcome_name(ep.get("outcome")) == "collision" for ep in ordered_episodes),
        "terminal_raw_sumo_collision_true": sum(bool((ep.get("last_snapshot", {}).get("events") or {}).get("raw_sumo_collision")) for ep in ordered_episodes if outcome_name(ep.get("outcome")) == "collision"),
        "terminal_geometric_collision_true": sum(bool((ep.get("last_snapshot", {}).get("events") or {}).get("geometric_collision")) for ep in ordered_episodes if outcome_name(ep.get("outcome")) == "collision"),
        "terminal_collision_event_id_records_nonempty": sum(bool((ep.get("collision_evidence") or [{}])[-1].get("collision_ids")) for ep in ordered_episodes if outcome_name(ep.get("outcome")) == "collision"),
        "same_tick_geometric_collision_raw_rows_read": len(raw_collision_rows),
        "same_tick_raw_collision_row_found_for_each_collision_episode": sum(bool(row.get("same_tick_geometric_collision_raw_row_found")) for row in collision_records),
        "replayed_saved_center_obb_candidate_count_distribution": dict(candidate_count),
        "unique_partner_episode_count": sum(row.get("partner_join_status") == "unique_saved_center_obb_overlap_candidate" for row in collision_records),
        "ambiguous_or_missing_partner_episode_count": sum(row.get("partner_join_status") != "unique_saved_center_obb_overlap_candidate" for row in collision_records),
        "unique_partner_policy_visibility": dict(partner_seen),
        "unique_partner_exact_policy_input_join": dict(partner_input),
        "unique_partner_conflict_timing_status": dict(partner_ct_status),
        "unique_partner_terminal_pair_calibration_status": dict(partner_calibration_status),
        "ego_terminal_road_by_partner_road": {f"{ego}->{foe}": count for (ego, foe), count in collision_location.items()},
        "unique_partner_last_pre_action_and_terminal_speed": {
            name: stats(row.get(name) for row in collision_state_rows)
            for name in (
                "ego_terminal_speed_mps", "partner_terminal_speed_mps", "target_speed_mps_for_final_action",
                "final_action_speed_abs_tracking_error_mps",
            )
        },
        "unique_partner_pre_action_speed": {
            "ego": stats((row.get("ct_partner_row") or {}).get("ego_speed_mps_pre_action") for row in collision_state_rows),
            "partner": stats((row.get("ct_partner_row") or {}).get("partner_speed_mps_pre_action") for row in collision_state_rows),
        },
        "unique_partner_valid_time_prediction_counts": {
            "ego_time_valid": sum((row.get("ct_partner_row") or {}).get("ego_time_valid") is True for row in collision_state_rows),
            "partner_time_valid": sum((row.get("ct_partner_row") or {}).get("partner_time_valid") is True for row in collision_state_rows),
            "both_valid": sum((row.get("ct_partner_row") or {}).get("ego_time_valid") is True and (row.get("ct_partner_row") or {}).get("partner_time_valid") is True for row in collision_state_rows),
        },
        "interpretation_limits": [
            "Collision partner candidates are recomputed from each same-tick raw_steps row whose events.geometric_collision is true, using saved center positions and headings with the same 0.05 m OBB leeway; this replays the recorded geometric predicate and is not independent SUMO collision truth.",
            "SUMO collision IDs/events are separately retained; an empty raw SUMO event does not identify the geometric-contact partner.",
            "A partner present in the policy actor tensor and a supported ConflictTiming row means the branch had a represented candidate, not that the actor used that cue or that the future collision was predictable with a valid ego ETA.",
            "Terminal calibration records are right-censored by episode termination and must not be treated as forecast errors or successful clearance observations.",
        ],
        "decision_prediction_clock_join": {
            "episodes_where_last_decision_pre_raw_equals_snapshot_observed_neighbor_raw": exact_decision_pre_matches,
            "episodes_with_exact_final_pre_action_prediction": exact_prediction_pre_matches,
            "episode_denominator": len(ordered_episodes),
            "join_note": "sidecar prediction rows use episode-local raw_step_pre_action; representation/shadow rows use eval-global raw step; decision joins use cumulative episode-local raw_ticks",
        },
        "records": collision_records,
    }

    terminal_summary = {}
    for label, rows in sorted(terminal_behavior_by_outcome.items()):
        ct_rows = terminal_ct_by_outcome.get(label, [])
        terminal_summary[label] = {
            "episode_count": len(rows),
            "requested_target_speed_mps": stats(row.get("requested_target_speed_mps") for row in rows),
            "terminal_actual_speed_mps": stats(row.get("terminal_actual_speed_mps") for row in rows),
            "current_lane_eligible_count": sum(row.get("current_lane_can_reach_next_edge") is True for row in rows),
            "current_lane_ineligible_count": sum(row.get("current_lane_can_reach_next_edge") is False for row in rows),
            "current_lane_unknown_count": sum(row.get("current_lane_can_reach_next_edge") is None for row in rows),
            "lane_command_counts": dict(Counter(str(row.get("lane_command_requested")) for row in rows)),
            "lane_request_status_counts": dict(Counter(str(row.get("lane_control_request_status")) for row in rows)),
            "lane_request_reason_counts": dict(Counter(str(row.get("lane_control_request_reason")) for row in rows)),
            "lane_change_applied_counts": dict(Counter(str(row.get("lane_change_applied")) for row in rows)),
            "conflict_timing_at_last_pre_action": {
                "episodes_with_prediction_row": sum(row.get("prediction_available") is True for row in ct_rows),
                "episodes_with_any_supported_crossing_row": sum((finite(row.get("valid_relation_count")) or 0) > 0 for row in ct_rows),
                "episodes_with_any_joint_valid_eta_pair": sum((finite(row.get("valid_joint_timing_count")) or 0) > 0 for row in ct_rows),
                "feature_valid_actor_rows": stats(row.get("feature_valid_actor_rows") for row in ct_rows),
                "route_conflict_timing_active": stats(row.get("route_conflict_timing_active") for row in ct_rows),
                "valid_actor_count": stats(row.get("route_conflict_timing_valid_actor_count") for row in ct_rows),
                "candidate_actor_count": stats(row.get("route_conflict_timing_candidate_actor_count") for row in ct_rows),
                "coverage_fraction": stats(row.get("route_conflict_timing_coverage_fraction") for row in ct_rows),
                "delta_rms": stats(row.get("route_conflict_timing_delta_rms") for row in ct_rows),
                "delta_to_base_rms": stats(row.get("route_conflict_timing_delta_to_base_rms") for row in ct_rows),
            },
        }

    files = [
        EVAL / "episodes.jsonl",
        EVAL / "decisions.jsonl.gz",
        EVAL / "task_conflict_predictions.jsonl.gz",
        EVAL / "task_conflict_calibration.jsonl.gz",
        EVAL / "representation.jsonl",
        EVAL / "policy_shadow_probes.jsonl",
        EVAL / "raw_steps.jsonl.gz",
        EVAL / "policy_shadow_probes_summary.json",
        EVAL / "task_conflict_timing_summary.json",
        TRAIN / "summary.json",
        TRAIN / "optimization.jsonl",
        TRAIN / "task_conflict_calibration.jsonl.gz",
        TRAIN / "task_conflict_timing_summary.json",
        TRAIN / "task_conflict_timing_manifest.json",
        TRAIN / "policy_shadow_probes.jsonl",
        ROUTEACT_EVAL / "policy_shadow_probes.jsonl",
        ROUTEACT_EVAL / "representation.jsonl",
        ROUTEACT_EVAL / "episodes.jsonl",
        ROUTEACT_RUN / "evaluation_results.json",
        RUN / "evaluation_results.json",
        ROUTEACT_RUN / "diagnostics" / "train" / "policy_shadow_probes.jsonl",
        *(FROZEN_INTERVENTION / name / "diagnostics" / "eval" / "episodes.jsonl" for name in frozen_episode_maps),
        *(FROZEN_INTERVENTION / name / "evaluation_results.json" for name in frozen_episode_maps),
    ]
    return {
        "analysis_schema": "conflicttime_eval_failure_audit_v1",
        "generated_by": str(Path(__file__).resolve()),
        "run_directory": str(RUN),
        "phase_directory": str(EVAL),
        "method": "sac_mlp_d1_st_rt_conflicttime_v1",
        "result_protocol": "environment_step_reward_v2; terminal_outcome_protocol=exclusive_terminal_v2; B observation uses policy-observed actors only",
        "source_files": {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "episodes": {
            "episode_count": len(ordered_episodes),
            "outcome_counts": dict(Counter(outcome_name(row.get("outcome")) for row in ordered_episodes)),
            "by_terminal_outcome": episode_metric_summary(ordered_episodes),
            "last_decision_behavior_by_terminal_outcome": terminal_summary,
        },
        "training_capture_and_updates": {
            "episodes_finished": train_capture_summary.get("episodes_finished"),
            "raw_step_records": train_capture_summary.get("raw_records"),
            "policy_decision_records": train_capture_summary.get("decision_records"),
            "diagnostic_error_count": train_capture_summary.get("diagnostic_error_count"),
            "diagnostic_serialization_seconds": train_capture_summary.get("serialization_seconds"),
            "optimization_samples": train_capture_summary.get("optimization_samples"),
            "last_recorded_update_count": next((row.get("updates") for row in reversed(train_optimization) if row.get("updates") is not None), None),
            "last_optimization_logger_sample": (lambda row: {
                "raw_steps": row.get("raw_steps"),
                "decision_steps": row.get("decision_steps"),
                "updates": row.get("updates"),
                "timing_semantics": row.get("timing"),
                "metrics": {
                    key: (row.get("metrics") or {}).get(key)
                    for key in (
                        "train/actor_loss", "train/critic_loss", "train/ent_coef", "train/n_updates",
                        "diagnostic/q1_mean_sampled", "diagnostic/q2_mean_sampled", "diagnostic/target_q_mean_sampled",
                        "diagnostic/q1_abs_td_mean_sampled", "diagnostic/q2_abs_td_mean_sampled",
                        "diagnostic/encoder_grad/route_conflict_timing/gradient_parameter_fraction",
                        "diagnostic/encoder_grad/route_conflict_timing/gradient_l2",
                        "diagnostic/encoder_update/route_conflict_timing/delta_l2",
                        "diagnostic/encoder_update/route_conflict_timing/relative_delta_l2",
                    ) if key in (row.get("metrics") or {})
                },
            })(train_optimization[-1] if train_optimization else {}),
            "outcome_counts": train_capture_summary.get("outcome_counts"),
            "conflict_timing_summary": train_timing_summary,
            "calibration_outcome_status_counts": dict(train_calibration_outcomes),
            "unique_episode_foe_pairs_in_calibration_stream": len(train_calibration_pairs),
            "calibration_rows_read": len(train_calibration_rows),
            "calibration_component_path_and_interval_detail": summarize_calibration_records(train_calibration_rows),
            "note": "Calibration status counts are repeated decision-time pair outcomes, not independent encounters; task summary records zero cap/IO drops and zero pending-at-close censors separately.",
        },
        "representation_activation": {
            "rows": len(representation_rows),
            "source_counts": dict(Counter(str(row.get("source")) for row in representation_rows)),
            "by_terminal_outcome_policy_prediction_state": {
                label: {"prediction_state_count": len(rows), "metrics": summarize_activation(rows)}
                for label, rows in sorted(representation_by_outcome.items())
            },
            "shadow_state_join_contract": "shadow sampled_obs_raw_global_step joins to representation metrics evaluation_prediction_raw_step_index; do not substitute episode-local raw step",
        },
        "policy_shadow": shadow,
        "normal_eval_shadow_sampling_budget_audit": {
            "A_routeact": {
                "train": summarize_shadow_sampling(routeact_train_shadow_rows, "train"),
                "eval": summarize_shadow_sampling(routeact_shadow_rows, "eval"),
            },
            "B_conflicttime": {
                "train": summarize_shadow_sampling(train_shadow_rows, "train"),
                "eval": summarize_shadow_sampling(shadow_rows, "eval"),
            },
        },
        "routeact_reference_policy_shadow": {
            "run_directory": str(ROUTEACT_RUN),
            "evaluation_result_identity": read_json(ROUTEACT_RUN / "evaluation_results.json").get("identity"),
            "summary": routeact_shadow,
            "note": "Same-evaluation normal-flow shadow probes for the route-action reference method. State changes are functional sensitivity only, not a performance attribution.",
        },
        "calibration": {
            "summary_file": read_json(EVAL / "task_conflict_timing_summary.json"),
            "rows_read": len(calibration_rows),
            "outcome_status_counts": dict(calibration_outcomes),
            "unique_episode_raw_step_foe_prediction_keys": len(calibration_by_episode_raw_foe),
            "unique_episode_foe_encounters": len(calibration_episode_foe_encounters),
            "component_path_and_interval_detail": summarize_calibration_records(calibration_rows),
            "note": "Rows are repeated decision-time predictions over encounters, not independent encounters. Component interval errors and censoring denominators are in the companion train/eval diagnostics audit JSON.",
        },
        "collision_failure_analysis": collision_metrics,
        "frozen_intervention_pairing": summarize_frozen_outcome_pairing(routeact_episode_by_seed, base_episode_by_seed, frozen_episode_maps, collision_records),
    }


if __name__ == "__main__":
    result = analyze()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "episodes": result["episodes"]["outcome_counts"],
        "shadow": {key: {name: value for name, value in result["policy_shadow"]["variants"][key].items() if name not in ("by_terminal_outcome",)} for key in result["policy_shadow"]["variants"]},
        "collision_partner": {key: value for key, value in result["collision_failure_analysis"].items() if key != "records"},
    }, ensure_ascii=False, indent=2))
