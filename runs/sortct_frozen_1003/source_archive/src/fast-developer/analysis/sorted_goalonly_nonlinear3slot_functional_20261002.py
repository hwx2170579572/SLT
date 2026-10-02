"""Static aggregation of saved policy-shadow and training diagnostics.

No environment, model, training, or evaluation code is imported or executed.
Rows are read from the two completed sorted/depart4 runs and grouped by the
saved unique sample_id so probe variants at one state are not treated as
independent observations.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUNS_ROOT = Path(
    r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_1002"
)
DEFAULT_OUTPUT = Path(__file__).with_suffix(".json")
METHODS = {
    "nonlinear_3slot": "sac_mlp_d1_st_rt_3slot_nonlinear_v1__intersection_sorted_depart4p0",
    "goalonly": "sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0",
}
EXPECTED_PROBES = [
    "st_spatial_off",
    "st_temporal_current_only",
    "st_social_ego_only",
    "rt_intent_injection_off",
    "rt_route_readout_zero",
    "topology_actor_intent_off",
    "topology_relations_off",
    "topology_goal_off",
    "route_reachability_off",
    "slot_ego_zero",
    "slot_social_zero",
    "slot_route_zero",
]
GROUPS = [
    "spatial_vehicle_messages",
    "temporal_attention",
    "social_attention",
    "route_path_attention",
    "route_goal_attention",
    "route_modulation",
    "map_route_encoder",
    "topology_lane_encoder",
    "topology_vehicle_query",
    "topology_goal_query",
    "topology_fusion_norms_scales",
    "slot_ego",
    "slot_social",
    "slot_route",
]
ACTIVATION_METRICS = [
    "spatial_message_delta_relative_rms",
    "spatial_valid_frame_count",
    "history_valid_frames_per_actor_mean",
    "history_single_valid_frame_fraction",
    "temporal_delta_relative_rms",
    "temporal_valid_actor_count",
    "social_attention_non_ego_mass",
    "social_attention_valid_actor_count_mean",
    "social_attention_query_count",
    "route_path_count_valid_actor_mean",
    "route_empty_path_fraction",
    "route_attention_effective_paths_nonempty",
    "route_attention_query_count",
    "route_reachability_candidate_count_mean",
    "route_reachability_known_node_count_mean",
    "route_reachability_unknown_node_count_mean",
    "route_reachability_goal_expand_fraction",
    "route_reachability_goal_bypass_fraction",
    "route_reachability_safe_mask_fallback_fraction",
    "route_reachability_goal_legal_attention_mass_active",
    "route_reachability_goal_delta_norm_active",
    "goal_topology_valid_candidate_count_mean",
    "goal_topology_active_query_count",
    "goal_topology_effective_nodes_active",
    "goal_topology_attention_entropy_active",
    "goal_topology_delta_rms",
    "ego_slot_rms",
    "social_slot_rms",
    "route_slot_rms",
    "ego_slot_within_sample_std_mean",
    "social_slot_within_sample_std_mean",
    "route_slot_within_sample_std_mean",
]


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:  # preserve line/file evidence in output
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    x = (len(xs) - 1) * q
    lo, hi = int(x), min(int(x) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (x - lo)


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    xs = [n for value in values if (n := _number(value)) is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": _percentile(xs, 0.5),
        "p90": _percentile(xs, 0.9),
        "min": min(xs),
        "max": max(xs),
    }


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        n = _number(item)
        if n is None:
            return None
        out.append(n)
    return out


def _l2_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return _stats(row.get(key) for row in rows)


def _episode_outcomes(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    by_seed: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in _records(path):
        seed = row.get("seed")
        outcome = row.get("outcome")
        if seed is not None and isinstance(outcome, str):
            by_seed[str(seed)] = outcome
            counts[outcome] += 1
    return by_seed, dict(counts)


def _decision_context(path: Path, wanted: set[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Join eval states to per-episode decisions and their action-window TTC.

    Shadow decision_step is run-global; decisions.jsonl.gz uses an episode-local
    decision counter. Callers convert via cumulative episode decision offsets.
    """
    found: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists() or not wanted:
        return found
    for row in _records(path):
        episode = row.get("episode")
        decision = row.get("decision")
        if not isinstance(episode, int) or not isinstance(decision, int):
            continue
        key = (episode - 1, decision)
        if key not in wanted:
            continue
        control = row.get("control") or {}
        info = row.get("info_base") or {}
        ttc = _number(row.get("min_cv_obb_ttc_s"))
        found[key] = {
            "route_status_known": control.get("route_lane_status_known"),
            "current_lane_can_reach_next_edge": control.get("current_lane_can_reach_next_edge"),
            "expected_target_lane_can_reach_next_edge": control.get("expected_target_lane_can_reach_next_edge"),
            "current_lane_id": control.get("current_lane_id"),
            "expected_target_lane_id": control.get("expected_target_lane_id"),
            "lane_control_request_status": control.get("lane_control_request_status"),
            "lane_control_request_reason": control.get("lane_control_request_reason"),
            "min_cv_obb_ttc_s_during_following_action_window": ttc,
            "decision_window_raw_ticks": row.get("raw_ticks"),
            "post_action_terminal": bool(row.get("terminated") or row.get("truncated")),
            "undiscounted_reward": _number(info.get("undiscounted_reward")),
        }
    return found


def _probe_stats(rows: list[dict[str, Any]], *, include_context: bool = False) -> dict[str, Any]:
    applicable = [r for r in rows if r.get("applicable") is True]
    valid = [r for r in applicable if r.get("valid") is True]
    reasons = Counter(str(r.get("invalid_reason") or "unspecified") for r in rows if r not in valid and r.get("applicable") is not True)
    active_reasons = Counter(str(r.get("invalid_reason") or "unspecified") for r in applicable if r.get("valid") is not True)
    norm_components = [[], []]
    decoded_speed_delta: list[float] = []
    lane_changes: list[float] = []
    any_decoded_changes: list[float] = []
    baseline_margin: list[float] = []
    lane_threshold_distance: list[float] = []
    feature_latent_delta: list[float] = []
    slot_delta: dict[str, list[float]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    for row in valid:
        abs_delta = _vector(row.get("action_delta_abs_normalized"))
        if abs_delta and len(abs_delta) >= 2:
            norm_components[0].append(abs_delta[0])
            norm_components[1].append(abs_delta[1])
        baseline = _vector(row.get("baseline_action_decoded"))
        shadow = _vector(row.get("shadow_action_decoded"))
        if baseline is not None and shadow is not None and len(baseline) >= 2 and len(shadow) >= 2:
            decoded_speed_delta.append(abs(shadow[0] - baseline[0]))
            lane_changes.append(float(shadow[1] != baseline[1]))
            any_decoded_changes.append(float(shadow != baseline))
        margins = _vector(row.get("baseline_tanh_saturation_margin"))
        if margins:
            baseline_margin.append(min(margins))
        lane_dist = _number(row.get("baseline_lane_threshold_distance"))
        if lane_dist is not None:
            lane_threshold_distance.append(abs(lane_dist))
        base_feature = _vector(row.get("baseline_feature_latent"))
        shadow_feature = _vector(row.get("shadow_feature_latent"))
        if base_feature is not None and shadow_feature is not None and len(base_feature) == len(shadow_feature) and base_feature:
            feature_latent_delta.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(base_feature, shadow_feature)) / len(base_feature)))
        base_slots = row.get("baseline_slot_latents") or {}
        shadow_slots = row.get("shadow_slot_latents") or {}
        if isinstance(base_slots, dict) and isinstance(shadow_slots, dict):
            for name, base_value in base_slots.items():
                bvec, svec = _vector(base_value), _vector(shadow_slots.get(name))
                if bvec is not None and svec is not None and len(bvec) == len(svec) and bvec:
                    slot_delta[name].append(math.sqrt(sum((a - b) ** 2 for a, b in zip(bvec, svec)) / len(bvec)))
        outcome_obj = row.get("outcome") or {}
        if isinstance(outcome_obj, dict):
            label = next((k for k in ("success", "collision", "timeout", "off_route") if outcome_obj.get(k) is True), "other")
            outcomes[label] += 1
        trigger = row.get("trigger")
        for value in trigger if isinstance(trigger, list) else [trigger]:
            if value is not None:
                triggers[str(value)] += 1
    decoded_n = len(lane_changes)
    result: dict[str, Any] = {
        "rows": len(rows),
        "unique_states": len({str(r.get("sample_id")) for r in rows}),
        "applicable_states": len(applicable),
        "valid_states": len(valid),
        "not_applicable_states": len(rows) - len(applicable),
        "active_invalid_states": len(applicable) - len(valid),
        "not_applicable_reasons": dict(reasons),
        "active_invalid_reasons": dict(active_reasons),
        "valid_eval_outcome_sample_counts": dict(outcomes),
        "valid_sample_trigger_label_counts_nonexclusive": dict(triggers),
        "pre_tanh_delta_l2": _l2_stats(valid, "action_delta_l2_pre_tanh"),
        "normalized_action_delta_l2": _l2_stats(valid, "action_delta_l2_normalized"),
        "normalized_abs_component_0": _stats(norm_components[0]),
        "normalized_abs_component_1": _stats(norm_components[1]),
        "normalized_delta_nonzero_fraction_eps_1e_6": (
            sum((_number(r.get("action_delta_l2_normalized")) or 0.0) > 1e-6 for r in valid) / len(valid) if valid else None
        ),
        "decoded_action": {
            "comp0_abs_delta_stats": _stats(decoded_speed_delta),
            "comp1_change_fraction": (sum(lane_changes) / decoded_n if decoded_n else None),
            "any_component_change_fraction": (sum(any_decoded_changes) / decoded_n if decoded_n else None),
            "valid_decoded_pairs": decoded_n,
            "component_semantics": "SumoSceneEnv.adapt_action: component 0 maps to target speed; component 1 maps to lane command {-1,0,1} using +/-1/3 thresholds",
        },
        "baseline_saturation_margin_min_over_action_dims": _stats(baseline_margin),
        "baseline_lane_threshold_distance": _stats(lane_threshold_distance),
        "feature_latent_delta_rms_per_dimension": _stats(feature_latent_delta),
        "slot_latent_delta_rms_per_dimension": {name: _stats(values) for name, values in sorted(slot_delta.items())},
    }
    if include_context:
        result["states_by_outcome"] = {
            label: _probe_stats_subset(valid, lambda r, lab=label: _row_outcome(r) == lab)
            for label in ("success", "collision", "timeout", "off_route", "other")
        }
        trigger_labels = sorted({
            str(t) for r in valid for t in (r.get("trigger") if isinstance(r.get("trigger"), list) else [r.get("trigger")] ) if t is not None
        })
        result["states_by_trigger_label"] = {
            label: _probe_stats_subset(valid, lambda r, lab=label: lab in (r.get("trigger") if isinstance(r.get("trigger"), list) else [r.get("trigger")]))
            for label in trigger_labels
        }
    return result


def _row_outcome(row: dict[str, Any]) -> str:
    outcome = row.get("outcome") or {}
    if isinstance(outcome, dict):
        return next((key for key in ("success", "collision", "timeout", "off_route") if outcome.get(key) is True), "other")
    return "other"


def _probe_stats_subset(rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    selected = [r for r in rows if predicate(r)]
    return {
        "n_states": len(selected),
        "pre_tanh_delta_l2": _l2_stats(selected, "action_delta_l2_pre_tanh"),
        "normalized_action_delta_l2": _l2_stats(selected, "action_delta_l2_normalized"),
        "lane_command_changed_fraction": _lane_changed_fraction(selected),
    }


def _lane_changed_fraction(rows: list[dict[str, Any]]) -> float | None:
    changes = []
    for row in rows:
        base = _vector(row.get("baseline_action_decoded"))
        shadow = _vector(row.get("shadow_action_decoded"))
        if base is not None and shadow is not None and len(base) >= 2 and len(shadow) >= 2:
            changes.append(float(base[1] != shadow[1]))
    return sum(changes) / len(changes) if changes else None


def _shadow_phase(method_dir: Path, phase: str, *, include_outcomes: bool) -> dict[str, Any]:
    phase_dir = method_dir / "diagnostics" / phase
    shadow_path = phase_dir / "policy_shadow_probes.jsonl"
    summary_path = phase_dir / "policy_shadow_probes_summary.json"
    rows = list(_records(shadow_path))
    sample_ids = {str(r.get("sample_id")) for r in rows}
    samples_by_episode: Counter[str] = Counter()
    episode_sample_keys: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        episode_sample_keys[int(row.get("episode_index", -1))].add(str(row.get("sample_id")))
    for episode, ids in episode_sample_keys.items():
        samples_by_episode[str(len(ids))] += 1
    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_probe[str(row.get("probe_name"))].append(row)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    outcomes_path = phase_dir / "episodes.jsonl"
    episode_outcome_map, episode_outcome_counts = _episode_outcomes(outcomes_path) if phase == "eval" else ({}, {})
    outcome_mismatch = 0
    if phase == "eval" and episode_outcome_map:
        for row in rows:
            seed = row.get("episode_seed")
            declared = _row_outcome(row)
            official = episode_outcome_map.get(str(seed))
            if official is not None and declared != official:
                outcome_mismatch += 1
    by_probe_summary = {
        name: _probe_stats(probe_rows, include_context=include_outcomes and phase == "eval")
        for name, probe_rows in sorted(by_probe.items())
    }
    baseline_rows = []
    seen = set()
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in seen:
            continue
        seen.add(sid)
        baseline_rows.append(row)
    baseline_action0 = []
    baseline_action1 = []
    baseline_decoded_speed = []
    baseline_decoded_lane = []
    for row in baseline_rows:
        if not row.get("valid"):
            continue
        v = _vector(row.get("baseline_action_normalized_mean"))
        if v and len(v) >= 2:
            baseline_action0.append(v[0]); baseline_action1.append(v[1])
        d = _vector(row.get("baseline_action_decoded"))
        if d and len(d) >= 2:
            baseline_decoded_speed.append(d[0]); baseline_decoded_lane.append(d[1])
    out: dict[str, Any] = {
        "input_files": {
            "shadow_jsonl": str(shadow_path),
            "shadow_summary_json": str(summary_path),
            "episodes_jsonl": str(outcomes_path) if outcomes_path.exists() else None,
        },
        "logged_rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "unique_samples_per_episode_count_distribution": dict(samples_by_episode),
        "official_episode_outcome_counts": episode_outcome_counts,
        "sample_outcome_vs_episode_file_mismatch_rows": outcome_mismatch,
        "runner_summary": summary,
        "baseline_action_normalized_component_0": _stats(baseline_action0),
        "baseline_action_normalized_component_1": _stats(baseline_action1),
        "baseline_decoded_target_speed": _stats(baseline_decoded_speed),
        "baseline_decoded_lane_command": dict(Counter(str(x) for x in baseline_decoded_lane)),
        "probes": by_probe_summary,
    }
    if phase == "eval":
        episode_offsets: dict[int, int] = {}
        previous_decisions = 0
        episode_rows = sorted(
            _records(outcomes_path), key=lambda item: int(item.get("episode", 0))
        )
        for episode_row in episode_rows:
            episode_number = episode_row.get("episode")
            if not isinstance(episode_number, int):
                continue
            episode_offsets[episode_number - 1] = previous_decisions
            previous_decisions += int(episode_row.get("decisions", 0) or 0)
        wanted = set()
        for row in baseline_rows:
            episode_index = row.get("episode_index")
            global_decision = row.get("decision_step")
            offset = episode_offsets.get(episode_index) if isinstance(episode_index, int) else None
            if isinstance(global_decision, int) and offset is not None:
                local_decision = global_decision - offset
                if local_decision > 0:
                    wanted.add((episode_index, local_decision))
        context = _decision_context(phase_dir / "decisions.jsonl.gz", wanted)
        context_counts: Counter[str] = Counter()
        ttc_counts: Counter[str] = Counter()
        by_sample: dict[str, Any] = {}
        for row in baseline_rows:
            episode_index = row.get("episode_index")
            global_decision = row.get("decision_step")
            offset = episode_offsets.get(episode_index) if isinstance(episode_index, int) else None
            local_decision = global_decision - offset if isinstance(global_decision, int) and offset is not None else None
            key = (episode_index, local_decision)
            ctx = context.get(key)
            if not ctx:
                context_counts["unmatched_decision"] += 1
                continue
            route_known = ctx.get("route_status_known")
            can_reach = ctx.get("current_lane_can_reach_next_edge")
            route_label = "unknown" if route_known is not True or not isinstance(can_reach, bool) else ("current_lane_eligible" if can_reach else "current_lane_ineligible")
            context_counts[route_label] += 1
            ttc = ctx.get("min_cv_obb_ttc_s_during_following_action_window")
            if ttc is None:
                ttc_counts["missing"] += 1
            elif ttc < 1.0:
                ttc_counts["<1s_during_action_window"] += 1
            elif ttc < 3.0:
                ttc_counts["1-3s_during_action_window"] += 1
            else:
                ttc_counts[">=3s_during_action_window"] += 1
            by_sample[str(row.get("sample_id"))] = ctx
        out["eval_decision_context_join"] = {
            "description": "Matched sample to same episode/decision control fields; TTC is the min observed over the following executed action-repeat window, not a pre-action instantaneous risk estimate.",
            "matched_unique_states": len(by_sample),
            "route_eligibility_state_counts": dict(context_counts),
            "following_action_window_ttc_counts": dict(ttc_counts),
            "sample_context_by_sample_id": by_sample,
        }
    return out


def _numeric_series(path: Path, source: str, keys: list[str], *, prefix: str = "") -> dict[str, Any]:
    buckets: dict[str, list[float]] = {key: [] for key in keys}
    row_count = 0
    fresh_rows = 0
    for row in _records(path):
        row_count += 1
        if source and row.get("source") != source:
            continue
        metrics = row.get("metrics") or {}
        if source == "train_forward_activation" and _number(metrics.get("diagnostic_sampled")) != 1.0:
            continue
        fresh_rows += 1
        for key in keys:
            value = metrics.get(prefix + key)
            if (n := _number(value)) is not None:
                buckets[key].append(n)
    return {
        "path": str(path),
        "source_filter": source or None,
        "rows_read": row_count,
        "fresh_activation_rows_used": fresh_rows if source == "train_forward_activation" else None,
        "metrics": {key: _stats(values) for key, values in buckets.items()},
    }


def _gradient_update_summary(path: Path) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {
        "critic_td_gradient_pre_clip": {"n": 0, "groups": {}},
        "critic_td_parameter_update": {"n": 0, "groups": {}},
    }
    gradient_metrics = ("gradient_parameter_fraction", "gradient_l2", "gradient_nonzero_elements", "gradient_nonfinite_elements", "optimizer_owned_elements", "optimizer_nonowned_elements")
    update_metrics = ("changed_elements", "relative_delta_l2", "delta_l2", "delta_nonfinite_elements", "optimizer_owned_elements", "optimizer_nonowned_elements")
    for row in _records(path):
        source = row.get("source")
        if source not in events:
            continue
        event = events[source]
        event["n"] += 1
        metrics = row.get("metrics") or {}
        suffixes = gradient_metrics if source == "critic_td_gradient_pre_clip" else update_metrics
        for group in GROUPS:
            prefix = f"encoder_grad/{group}/" if source == "critic_td_gradient_pre_clip" else f"encoder_update/{group}/"
            if not any(key.startswith(prefix) for key in metrics):
                continue
            data = event["groups"].setdefault(group, {"rows": 0, "values": {key: [] for key in suffixes}, "positive_changed_rows": 0, "positive_nonzero_grad_rows": 0})
            data["rows"] += 1
            for suffix in suffixes:
                n = _number(metrics.get(prefix + suffix))
                if n is not None:
                    data["values"][suffix].append(n)
            if source == "critic_td_gradient_pre_clip":
                if (_number(metrics.get(prefix + "gradient_nonzero_elements")) or 0.0) > 0:
                    data["positive_nonzero_grad_rows"] += 1
            elif (_number(metrics.get(prefix + "changed_elements")) or 0.0) > 0:
                data["positive_changed_rows"] += 1
    for event_name, event in events.items():
        for group, data in event["groups"].items():
            data["metrics"] = {key: _stats(values) for key, values in data.pop("values").items()}
    return {"path": str(path), "events": events}


def _optimization_summary(path: Path) -> dict[str, Any]:
    keys = [
        "train/critic_loss",
        "train/actor_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
        "diagnostic/target_q_mean_sampled",
    ]
    values: dict[str, list[float]] = {key: [] for key in keys}
    present: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    rows = 0
    updates = []
    for row in _records(path):
        rows += 1
        if isinstance(row.get("updates"), int):
            updates.append(row["updates"])
        metrics = row.get("metrics") or {}
        for key in keys:
            if key not in metrics:
                continue
            present[key] += 1
            n = _number(metrics.get(key))
            if n is None:
                nonfinite[key] += 1
            else:
                values[key].append(n)
    return {
        "path": str(path),
        "rows": rows,
        "update_index_range": [min(updates), max(updates)] if updates else None,
        "metrics": {
            key: {
                **_stats(values[key]),
                "present_rows": present[key],
                "missing_rows": rows - present[key],
                "present_nonfinite_rows": nonfinite[key],
            }
            for key in keys
        },
        "interpretation_limit": "target_q_mean_sampled is a periodic scalar sample; critic_loss is the logged TD objective, not a per-transition TD-error distribution. Missing values are not counted as nonfinite.",
    }


def _method_summary(runs_root: Path, short_name: str, run_name: str) -> dict[str, Any]:
    run_dir = runs_root / run_name
    diag = run_dir / "diagnostics"
    train_rep = diag / "train" / "representation.jsonl"
    eval_rep = diag / "eval" / "representation.jsonl"
    return {
        "run_dir": str(run_dir),
        "method": short_name,
        "train_shadow": _shadow_phase(run_dir, "train", include_outcomes=False),
        "eval_shadow": _shadow_phase(run_dir, "eval", include_outcomes=True),
        "train_activation": _numeric_series(
            train_rep, "train_forward_activation", ACTIVATION_METRICS,
        ),
        "eval_activation": _numeric_series(
            eval_rep, "eval_policy", ACTIVATION_METRICS,
        ),
        "train_critic_grad_and_update": _gradient_update_summary(train_rep),
        "train_optimization": _optimization_summary(diag / "train" / "optimization.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = {
        "schema_version": "sorted_module_functional_diagnostics_v1",
        "created_utc_date": "2026-10-02",
        "runs_root": str(args.runs_root),
        "methods": {
            short: _method_summary(args.runs_root, short, run_name)
            for short, run_name in METHODS.items()
        },
        "analysis_limits": [
            "A probe row is a same-state deterministic actor-mean sensitivity to a single intervention; it is not a closed-loop performance counterfactual.",
            "Eval rows are grouped by unique sample_id; the first multi-car and first route-context trigger labels can overlap on the same state. Trigger counts are nonexclusive.",
            "Rows come from a small trigger-selected subset of each episode; samples are repeated within episodes and across outcomes, so n_states is not an iid sample size.",
            "The probes zero or bypass a branch. Large changes show local dependence at the sampled state; small changes can reflect redundancy, saturation, inactive inputs, or state selection. Neither alone proves performance utility or semantic disentanglement.",
            "TTC grouping, where joined, is explicitly the minimum over the following executed action-repeat window, not an instantaneous pre-action risk estimate.",
            "Gradient and parameter-update evidence is from critic TD updates and reflects optimizer participation, not actor gradient or causal task benefit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": list(result["methods"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
