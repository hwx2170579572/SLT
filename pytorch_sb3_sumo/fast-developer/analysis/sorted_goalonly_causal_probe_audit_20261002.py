#!/usr/bin/env python3
"""Read-only audit of the existing sorted goal-only critical/shadow probe logs.

This script does not create an environment or run SUMO. It deduplicates probe
rows by (episode_index, sample_id), because each sampled state has one row per
probe variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RUN_ROOT = Path(r"D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_causal_1002")
DEFAULT_OUTPUT = Path(__file__).with_suffix(".json")
ARMS = ("control", "routeveto")
INITIAL_LABELS = {"first_valid_multi_car", "first_valid_route_context"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def list_numbers(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out = [number(x) for x in value]
    return None if any(x is None for x in out) else [float(x) for x in out]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in values if number(x) is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p90": quantile(xs, .90), "min": min(xs), "max": max(xs)}


def trigger_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value] if isinstance(value, list) else []


def stages(row: dict[str, Any]) -> list[str]:
    labels = set(trigger_list(row.get("trigger")))
    out = []
    if labels & INITIAL_LABELS:
        out.append("initial")
    if "first_critical_event" in labels:
        out.append("critical")
    if "last_preterminal_prediction" in labels:
        out.append("preterminal")
    return out


def outcome_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for name in ("success", "collision", "timeout", "off_route"):
            if value.get(name) is True:
                return name
    return "unknown"


def critical_types(context: Any) -> list[str]:
    if isinstance(context, dict) and isinstance(context.get("critical_events"), list):
        return [str(x) for x in context["critical_events"]]
    return []


def sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [r for r in rows if r.get("applicable") is True]
    valid = [r for r in applicable if r.get("valid") is True]
    lane_changed, action_changed = [], []
    for r in valid:
        a, b = list_numbers(r.get("baseline_action_decoded")), list_numbers(r.get("shadow_action_decoded"))
        if a is not None and b is not None and len(a) >= 2 and len(b) >= 2:
            lane_changed.append(a[1] != b[1])
            action_changed.append(a != b)
    def rate(flags: list[bool]) -> dict[str, Any]:
        return {"n": len(flags), "count": sum(flags), "fraction": sum(flags) / len(flags) if flags else None}
    return {
        "unique_states": len(rows), "applicable": len(applicable), "valid": len(valid),
        "na_or_inapplicable": len(rows) - len(applicable), "applicable_invalid": len(applicable) - len(valid),
        "delta_action_l2_normalized": summary(x for r in valid if (x := number(r.get("action_delta_l2_normalized"))) is not None),
        "delta_action_l2_pre_tanh": summary(x for r in valid if (x := number(r.get("action_delta_l2_pre_tanh"))) is not None),
        "decoded_action_changed": rate(action_changed), "decoded_lane_command_changed": rate(lane_changed),
        "baseline_tanh_saturation_margin": summary(x for r in valid if (x := number(r.get("baseline_tanh_saturation_margin"))) is not None),
        "baseline_lane_threshold_distance": summary(x for r in valid if (x := number(r.get("baseline_lane_threshold_distance"))) is not None),
    }


def terminal_raw(ep: dict[str, Any]) -> int | None:
    # Episode summary uses episode-local raw_steps/collision_evidence.raw_step.
    if outcome_name(ep.get("outcome")) == "collision":
        evidence = ep.get("collision_evidence")
        raw = evidence.get("raw_step") if isinstance(evidence, dict) else None
        if isinstance(raw, (int, float)):
            return int(raw)
    raw = ep.get("raw_steps")
    return int(raw) if isinstance(raw, (int, float)) else None


def episode_start_global_raw(states: dict[tuple[int, str], list[dict[str, Any]]], ep_idx: int) -> int | None:
    """Shadow pre_obs_raw_step is run-global; recover episode origin from its initial state."""
    candidates = []
    for (idx, _sid), vals in states.items():
        if idx != ep_idx:
            continue
        row = vals[0]
        if set(trigger_list(row.get("trigger"))) & INITIAL_LABELS:
            raw = row.get("pre_obs_raw_step")
            if isinstance(raw, int):
                candidates.append(raw)
    return min(candidates) if candidates else None


def sample_episode_local_raw(row: dict[str, Any], start_global_raw: int | None) -> int | None:
    raw = row.get("pre_obs_raw_step")
    if not isinstance(raw, int) or start_global_raw is None:
        return None
    return raw - start_global_raw


def analyze_arm(run_root: Path, arm: str) -> dict[str, Any]:
    diag = run_root / arm / "diagnostics" / "eval"
    ep_path, probe_path = diag / "episodes.jsonl", diag / "policy_shadow_probes.jsonl"
    sum_path, veto_path = diag / "policy_shadow_probes_summary.json", diag / "route_lane_veto_actions.jsonl"
    episodes, rows = read_jsonl(ep_path), read_jsonl(probe_path)
    stored = json.loads(sum_path.read_text(encoding="utf-8-sig")) if sum_path.exists() else None
    # episode JSONL stores one-based episode numbers; shadow rows use zero-based indices.
    episode_by_index = {int(e["episode"]) - 1: e for e in episodes if isinstance(e.get("episode"), int)}
    by_state: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if isinstance(r.get("episode_index"), int) and r.get("sample_id") is not None:
            by_state[(r["episode_index"], str(r["sample_id"]))].append(r)
    episode_origin_raw = {ep_idx: episode_start_global_raw(by_state, ep_idx) for ep_idx in episode_by_index}
    states_by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for vals in by_state.values():
        states_by_ep[vals[0]["episode_index"]].append(vals[0])
    state_counts = Counter(len(v) for v in states_by_ep.values())

    trigger_counts: Counter[str] = Counter()
    stage_state_ids: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for key, vals in by_state.items():
        row = vals[0]
        trigger_counts.update(trigger_list(row.get("trigger")))
        for stage in stages(row):
            stage_state_ids[stage].add(key)

    critical_counts: Counter[str] = Counter()
    critical_event_name_counts: Counter[str] = Counter()
    critical_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    critical_gap: dict[str, list[float]] = defaultdict(list)
    collision_critical_gap: dict[str, list[float]] = defaultdict(list)
    critical_ttc: dict[str, list[float]] = defaultdict(list)
    collision_critical = Counter()
    for key, vals in by_state.items():
        row = vals[0]
        if "critical" not in stages(row):
            continue
        context = row.get("critical_event_context") or {}
        events = critical_types(context)
        critical_event_name_counts.update(events)
        route = any(x in events for x in ("known_unreachable_target_lane_requested", "route_eligible_to_ineligible_post_transition"))
        ttc = "action_window_ttc_le_3s" in events
        cls = "route_and_ttc" if route and ttc else "route" if route else "ttc" if ttc else "unknown_or_unlabeled"
        critical_counts[cls] += 1
        out = outcome_name(row.get("outcome"))
        critical_outcomes[out][cls] += 1
        val = number(context.get("action_window_min_cv_obb_ttc_s"))
        if val is not None:
            critical_ttc[cls].append(val)
        ep = episode_by_index.get(key[0])
        end = terminal_raw(ep) if ep else None
        sample_raw = sample_episode_local_raw(row, episode_origin_raw.get(key[0]))
        if end is not None and sample_raw is not None:
            critical_gap[cls].append(float(end - sample_raw))
        if out == "collision":
            if end is not None and sample_raw is not None:
                collision_critical_gap[cls].append(float(end - sample_raw))
            low_ttc_episode = number(ep.get("cv_ttc_below_3s_ticks")) if ep else None
            if route and not ttc and low_ttc_episode is not None and low_ttc_episode > 0:
                collision_critical["route_selected_but_episode_had_ttc_le_3s_exposure"] += 1
            if ttc:
                collision_critical["selected_ttc"] += 1
            elif route:
                collision_critical["selected_route"] += 1
            else:
                collision_critical["selected_unlabeled"] += 1

    outcomes = Counter(outcome_name(e.get("outcome")) for e in episodes)
    collisions = [e for e in episodes if outcome_name(e.get("outcome")) == "collision"]
    collision_ttc_exposed = sum((number(e.get("cv_ttc_below_3s_ticks")) or 0) > 0 for e in collisions)
    collision_route_ineligible = sum((number(e.get("route_lane_ineligible_ticks")) or 0) > 0 for e in collisions)
    collision_risk_evaluable = sum((number(e.get("risk_evaluable_ticks")) or 0) > 0 for e in collisions)

    terminal_gaps: dict[str, list[float]] = defaultdict(list)
    collision_preterminal = Counter()
    for key, vals in by_state.items():
        row = vals[0]
        if "preterminal" not in stages(row):
            continue
        ep = episode_by_index.get(key[0])
        if not ep:
            continue
        end = terminal_raw(ep)
        raw = sample_episode_local_raw(row, episode_origin_raw.get(key[0]))
        if end is None or raw is None:
            continue
        gap = float(end - raw)
        out = outcome_name(ep.get("outcome"))
        terminal_gaps[out].append(gap)
        if out == "collision":
            collision_preterminal["episodes_with_state"] += 1
            collision_preterminal["within_1s_10raw"] += gap <= 10
            collision_preterminal["within_2s_20raw"] += gap <= 20
            collision_preterminal["between_1_and_2s_10to20raw"] += 10 <= gap <= 20

    # Temporal coverage of all selected unique states before collisions.
    collision_state_windows = Counter()
    collision_state_windows_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    no_critical_by_outcome = Counter()
    for ep_idx, ep in episode_by_index.items():
        out = outcome_name(ep.get("outcome"))
        ep_states = [(key, vals[0]) for key, vals in by_state.items() if key[0] == ep_idx]
        if not any("critical" in stages(row) for _, row in ep_states):
            no_critical_by_outcome[out] += 1
        if out != "collision":
            continue
        end = terminal_raw(ep)
        if end is None:
            continue
        collision_state_windows["collision_episodes"] += 1
        gaps = []
        for _key, row in ep_states:
            raw = sample_episode_local_raw(row, episode_origin_raw.get(ep_idx))
            if raw is None:
                continue
            gap = float(end - raw)
            gaps.append(gap)
            for stage in stages(row):
                collision_state_windows_by_stage[stage]["sampled_states"] += 1
                if 0 <= gap <= 10:
                    collision_state_windows_by_stage[stage]["within_1s_10raw"] += 1
                if 10 < gap <= 20:
                    collision_state_windows_by_stage[stage]["within_1to2s_10to20raw"] += 1
                if 20 < gap <= 30:
                    collision_state_windows_by_stage[stage]["within_2to3s_20to30raw"] += 1
        collision_state_windows["episodes_with_any_sample_within_1s_10raw"] += any(0 <= g <= 10 for g in gaps)
        collision_state_windows["episodes_with_any_sample_1to2s_10to20raw"] += any(10 < g <= 20 for g in gaps)
        collision_state_windows["episodes_with_any_sample_2to3s_20to30raw"] += any(20 < g <= 30 for g in gaps)

    probe_names = sorted({str(r.get("probe_name")) for r in rows})
    functional = {"by_stage": {}, "preterminal_by_outcome": {}}
    for probe in probe_names:
        row_by_state = {(int(r["episode_index"]), str(r["sample_id"])): r for r in rows if r.get("probe_name") == probe}
        for stage in ("initial", "critical", "preterminal"):
            functional["by_stage"].setdefault(stage, {})[probe] = sensitivity(
                [row_by_state[k] for k in stage_state_ids[stage] if k in row_by_state]
            )
        functional["preterminal_by_outcome"][probe] = {}
        for out in ("success", "collision", "timeout"):
            keys = [k for k in stage_state_ids["preterminal"] if k in row_by_state and k[0] in episode_by_index and outcome_name(episode_by_index[k[0]].get("outcome")) == out]
            functional["preterminal_by_outcome"][probe][out] = sensitivity([row_by_state[k] for k in keys])

    veto_summary = None
    if veto_path.exists():
        acts = read_jsonl(veto_path)
        # Shadow decision_step is run-global; align by the local observation raw step.
        action_index = {(int(a.get("episode_index", -1)), int(a.get("raw_step_before_action", -1))): a for a in acts}
        joined = matched = sampled_veto = 0
        for key, vals in by_state.items():
            row = vals[0]
            raw_before = sample_episode_local_raw(row, episode_origin_raw.get(key[0]))
            if raw_before is None:
                continue
            act = action_index.get((key[0], raw_before))
            if act is None:
                continue
            joined += 1
            req, observed = list_numbers(act.get("action_requested")), list_numbers(row.get("actual_action"))
            if req is not None and observed is not None and len(req) == len(observed) and all(abs(a-b) <= 1e-6 for a, b in zip(req, observed)):
                matched += 1
            sampled_veto += act.get("veto_applied") is True
        veto_summary = {
            "rows": len(acts),
            "veto_applied": sum(a.get("veto_applied") is True for a in acts),
            "reasons": dict(Counter(str(a.get("reason")) for a in acts)),
            "sampled_states_joined_by_episode_and_pre_observation_raw_step": joined,
            "joined_probe_actual_action_equal_to_original_action_requested": matched,
            "joined_sample_states_with_veto": sampled_veto,
            "interpretation": "Shadow baseline/variant outputs belong to the original actor. Post-veto executed action is separate in this action log; raw-step alignment empirically checks the original action match.",
        }

    stage_counts = {s: len(ids) for s, ids in stage_state_ids.items()}
    return {
        "episodes": len(episodes), "outcomes": dict(outcomes), "shadow_probe_rows": len(rows),
        "unique_sampled_states": len(by_state), "unique_states_per_episode_distribution": dict(sorted(state_counts.items())),
        "episodes_without_sample": len(set(episode_by_index) - set(states_by_ep)),
        "episodes_over_3_unique_states": sum(n > 3 for n in state_counts.elements()),
        "trigger_label_counts_not_episode_counts": dict(trigger_counts),
        "unique_states_by_stage_overlap_allowed": stage_counts,
        "stage_state_overlaps": {
            "initial_and_critical": len(stage_state_ids["initial"] & stage_state_ids["critical"]),
            "critical_and_preterminal": len(stage_state_ids["critical"] & stage_state_ids["preterminal"]),
            "initial_and_preterminal": len(stage_state_ids["initial"] & stage_state_ids["preterminal"]),
        },
        "critical_selected_event_counts": dict(critical_counts),
        "critical_selected_event_names": dict(critical_event_name_counts),
        "critical_selected_by_outcome": {k: dict(v) for k, v in critical_outcomes.items()},
        "critical_selected_ttc_post_action_window_s": {k: summary(v) for k, v in critical_ttc.items()},
        "critical_sample_to_episode_terminal_gap_raw_all_outcomes": {k: summary(v) for k, v in critical_gap.items()},
        "critical_sample_to_collision_terminal_gap_raw": {k: summary(v) for k, v in collision_critical_gap.items()},
        "episodes_without_critical_sample_by_outcome": dict(no_critical_by_outcome),
        "collision_episode_risk_coverage": {
            "collision_episodes": len(collisions), "with_any_low_ttc_le_3s_episode_ticks": collision_ttc_exposed,
            "with_any_route_ineligible_ticks": collision_route_ineligible,
            "with_any_risk_evaluable_ticks": collision_risk_evaluable,
            "critical_sample_coverage": dict(collision_critical),
            "caveat": "These episode totals establish that low-TTC exposure was recorded somewhere in the episode; they do not timestamp it. The terminal collision frame may contribute, so they cannot alone establish pre-collision prediction coverage or ordering relative to a route trigger.",
        },
        "collision_sample_temporal_coverage": {
            "episode_counts": dict(collision_state_windows),
            "unique_sample_state_counts_by_stage": {k: dict(v) for k, v in collision_state_windows_by_stage.items()},
            "basis": "For each collision episode, convert the run-global selected sample pre_obs_raw_step to episode-local raw by subtracting that episode's initial-state global raw origin, then compare with episode-local collision_evidence.raw_step (fallback episode raw_steps). One raw tick=0.1s. These windows report timing only, not TTC/risk presence.",
        },
        "preterminal_sample_to_terminal_gap_raw_by_outcome": {k: summary(v) for k, v in terminal_gaps.items()},
        "collision_preterminal_gap_coverage": {
            **dict(collision_preterminal),
            "basis": "gap = episode-local collision_evidence.raw_step (fallback episode raw_steps) minus episode-local preterminal observation raw; episode-local sample raw is derived from run-global pre_obs_raw_step minus that episode's initial-state global raw origin. One raw tick=0.1s. Temporal proximity does not prove low TTC in that exact pre-action observation.",
        },
        "functional_sensitivity": functional,
        "route_veto_action_log": veto_summary,
        "stored_shadow_summary": stored,
        "source_sha256": {str(p): sha256(p) for p in (ep_path, probe_path, sum_path, veto_path) if p.exists()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    result = {
        "schema_version": "sorted_goalonly_causal_probe_audit_v1",
        "run_root": str(args.run_root),
        "scope": "Existing control and routeveto eval logs only. No SUMO/environment/extra evaluation/training was run.",
        "deduplication_key": ["episode_index", "sample_id"],
        "critical_sampler_semantics": [
            "At most one first-qualifying critical state per episode shares the middle slot with route and TTC triggers.",
            "Route-transition and action-window TTC context is collected after the just-completed action-repeat window and attached to the shadow candidate that was the pre-action observation for that same action; it is not the next observation.",
            "Episode aggregate TTC counts show exposure but do not timestamp individual risk windows.",
            "A terminal-adjacent preterminal sample is temporal coverage, not proof of a low-TTC state.",
            "routeveto shadow outputs represent the original policy; executed post-veto actions are separate.",
        ],
        "arms": {arm: analyze_arm(args.run_root, arm) for arm in ARMS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    compact = {}
    for arm, v in result["arms"].items():
        compact[arm] = {
            "episodes_outcomes": v["outcomes"], "unique_states": v["unique_sampled_states"],
            "states_per_episode": v["unique_states_per_episode_distribution"],
            "trigger_labels": v["trigger_label_counts_not_episode_counts"],
            "critical_events": v["critical_selected_event_counts"],
            "collision_risk": v["collision_episode_risk_coverage"],
            "preterminal_collision_gap": v["collision_preterminal_gap_coverage"],
        }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
