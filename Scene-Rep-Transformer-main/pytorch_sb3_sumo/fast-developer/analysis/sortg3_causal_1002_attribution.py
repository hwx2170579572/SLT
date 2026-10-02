"""Offline, paired behavior audit for the frozen sorted/depart4 causal arms.

Reads only saved evaluation, behavior, and route-veto telemetry.  It does not
construct an environment or run SUMO.  Derived output is kept beside this
script; official result files are never rewritten.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
ANALYSIS = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "sortg3_causal_1002"
REFERENCE_EVAL = ROOT / "runs" / "sortg3_1002" / "sac_mlp_d1_st_rt_topo_goalonly_v1__intersection_sorted_depart4p0" / "evaluation_results.json"
OUT = Path(__file__).with_suffix(".json")
ARMS = ("control", "goaloff", "routeveto")
ARM_PATHS = {name: RUN_ROOT / name for name in ARMS}
ORIGINAL_METHOD = "sac_mlp_d1_st_rt_topo_goalonly_v1"

if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
import sorted_goalonly_nonlinear3slot_behavior_20261002 as prior_behavior  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_stats(values):
    xs = [x for v in values if (x := finite(v)) is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
        "min": min(xs), "max": max(xs),
    }


def outcome_letter(record: dict[str, Any]) -> str:
    if record.get("success"):
        return "S"
    if record.get("collision"):
        return "C"
    if record.get("timeout"):
        return "T"
    if record.get("off_route"):
        return "O"
    return "?"


def outcome_name(episode: dict[str, Any]) -> str:
    name = episode.get("outcome")
    if name in {"success", "collision", "timeout", "off_route"}:
        return name
    return "unknown"


def collision_terminal_step(episode: dict[str, Any]) -> int | None:
    snapshot = episode.get("last_snapshot") or {}
    value = snapshot.get("raw_step")
    return int(value) if value is not None else None


def collision_partner(episode: dict[str, Any]) -> dict[str, Any]:
    replay = prior_behavior.reconstruct_terminal_collision_partners(episode)
    if replay.get("status") == "unique_replay_partner":
        return replay["partners"][0]
    return {"replay_status": replay.get("status"), "candidate_count": replay.get("candidate_count")}


def compact_raw_row(row: dict[str, Any], partner_id: str | None) -> dict[str, Any]:
    vehicles = row.get("vehicles") or []
    actor = next((x for x in vehicles if str(x.get("id")) == partner_id), None) if partner_id else None
    return {
        "raw_step": row.get("raw_step"),
        "speed_control": row.get("speed_control") or {},
        "ego": row.get("ego") or {},
        "lane_control": row.get("lane_control") or {},
        "current_lane_reachability_label": row.get("current_lane_reachability_label"),
        "min_cv_obb_ttc_s": row.get("min_cv_obb_ttc_s"),
        "risk_evaluable": row.get("risk_evaluable"),
        "vehicles": [actor] if actor is not None else [],
    }


def summarize_terminal_window(rows: list[dict[str, Any]], terminal_step: int | None) -> dict[str, Any]:
    if terminal_step is None:
        return {"expected_ticks": None, "captured_ticks": 0}
    window = [r for r in rows if r.get("raw_step") is not None and int(r["raw_step"]) < terminal_step][-30:]
    targets, actuals = [], []
    labels = Counter()
    requests = Counter()
    ttc = []
    stopped = 0
    for row in window:
        speed = row.get("speed_control") or {}
        target = finite(speed.get("requested_target_speed_mps"))
        actual = finite(speed.get("actual_speed_mps"))
        if target is not None:
            targets.append(target)
        if actual is not None:
            actuals.append(actual)
            stopped += int(actual < 0.1)
        label = row.get("current_lane_reachability_label")
        labels[str(label) if label is not None else "unknown"] += 1
        lane = row.get("lane_control") or {}
        requests[str(lane.get("lane_control_request_status") or "unknown")] += 1
        value = finite(row.get("min_cv_obb_ttc_s"))
        if value is not None:
            ttc.append(value)
    expected = min(30, max(0, terminal_step))
    known_labels = labels.get("0", 0) + labels.get("1", 0)
    return {
        "window": "last <=30 raw samples strictly before this episode's terminal raw step (0.1s per sample)",
        "expected_ticks": expected,
        "captured_ticks": len(window),
        "target_speed_mps": numeric_stats(targets),
        "actual_speed_mps": numeric_stats(actuals),
        "stopped_fraction_below_0p1mps": stopped / len(actuals) if actuals else None,
        "route_lane_label_counts": dict(labels),
        "ineligible_fraction_of_known_labels": labels.get("0", 0) / known_labels if known_labels else None,
        "lane_request_status_counts_per_raw_sample": dict(requests),
        "minimum_logged_cv_obb_ttc_s": min(ttc) if ttc else None,
    }


def summarize_partner_window(episode: dict[str, Any], rows: list[dict[str, Any]], partner_id: str | None) -> dict[str, Any]:
    terminal = collision_terminal_step(episode)
    if terminal is None or partner_id is None:
        return {"expected_ticks": min(30, terminal) if terminal is not None else None, "partner_id": partner_id, "partner_samples": 0}
    pre = [r for r in rows if r.get("raw_step") is not None and terminal - 30 <= int(r["raw_step"]) < terminal]
    present = []
    for row in pre:
        actor = next((v for v in row.get("vehicles", []) if str(v.get("id")) == partner_id), None)
        if actor is not None:
            present.append((row, actor))
    observed = Counter(
        "true" if actor.get("observed_by_policy") is True else
        "false" if actor.get("observed_by_policy") is False else "unknown"
        for _, actor in present
    )
    partner_speed = [v.get("speed") for _, v in present]
    pair_ttc = [((v.get("risk_cv") or {}).get("cv_obb_ttc_s")) for _, v in present]
    ego_target, ego_actual, closing = [], [], []
    for row, actor in present:
        speed = row.get("speed_control") or {}
        ego_target.append(speed.get("requested_target_speed_mps"))
        ego_actual.append(speed.get("actual_speed_mps"))
        risk = actor.get("risk_cv") or {}
        closing.append(risk.get("radial_closing_speed_mps"))
    expected = min(30, max(0, terminal))
    return {
        "partner_id": partner_id,
        "window": "last 30 raw samples strictly before collision snapshot; at most 3.0s",
        "expected_ticks": expected,
        "captured_raw_ticks": len(pre),
        "partner_present_samples_in_saved_vehicle_snapshot": len(present),
        "partner_presence_fraction_of_captured_ticks": len(present) / len(pre) if pre else None,
        "observed_by_policy_counts_given_partner_present": dict(observed),
        "partner_observed_by_policy_at_least_once_in_window": observed.get("true", 0) > 0,
        "policy_observation_provenance": "raw-step observed_neighbor_ids membership materialized as actor.observed_by_policy; telemetry vehicle list is not a full SUMO population dump",
        "ego_target_speed_mps_on_partner_present_samples": numeric_stats(ego_target),
        "ego_actual_speed_mps_on_partner_present_samples": numeric_stats(ego_actual),
        "partner_speed_mps_on_partner_present_samples": numeric_stats(partner_speed),
        "partner_cv_obb_ttc_s": numeric_stats(pair_ttc),
        "partner_radial_closing_speed_mps": numeric_stats(closing),
        "partner_ticks_with_cv_obb_ttc_le_3s": sum(1 for x in pair_ttc if finite(x) is not None and float(x) <= 3.0),
    }


def episode_behavior_summary(episode: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "raw_steps", "mean_actual_speed_mps", "stopped_fraction_of_speed_samples",
        "stopped_seconds", "route_lane_ineligible_seconds",
        "route_lane_ineligible_stopped_seconds", "route_lane_ineligible_stall_max_seconds",
        "route_lane_ineligible_ticks", "route_lane_known_ticks", "route_lane_unknown_ticks",
        "risk_evaluable_ticks", "cv_ttc_below_3s_ticks", "critical_unobserved_ticks",
        "covered_critical_unobserved_ticks", "diagnostic_error_count",
    )
    result = {key: episode.get(key) for key in keys}
    terminal = episode.get("terminal_info") or {}
    result["terminal_route"] = {
        key: terminal.get(key) for key in (
            "current_road_id", "current_lane_id", "current_lane_reachability_label",
            "planned_next_edge", "lane_command_requested", "lane_control_request_status",
            "lane_control_request_reason", "target_speed", "actual_speed_mps",
            "raw_sumo_collision", "geometric_collision", "collision",
        )
    }
    return result


def outcome_metric_groups(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for outcome in ("success", "collision", "timeout", "off_route"):
        rows = [x for x in episodes if outcome_name(x) == outcome]
        if not rows:
            continue
        out[outcome] = {
            "n": len(rows),
            "raw_steps_mean": numeric_stats(x.get("raw_steps") for x in rows),
            "actual_speed_mps_episode_mean": numeric_stats(x.get("mean_actual_speed_mps") for x in rows),
            "stopped_fraction_episode_mean": numeric_stats(x.get("stopped_fraction_of_speed_samples") for x in rows),
            "route_lane_ineligible_seconds_episode_mean": numeric_stats(x.get("route_lane_ineligible_seconds") for x in rows),
            "route_lane_ineligible_stopped_seconds_episode_mean": numeric_stats(x.get("route_lane_ineligible_stopped_seconds") for x in rows),
            "route_lane_ineligible_stall_max_seconds_episode_mean": numeric_stats(x.get("route_lane_ineligible_stall_max_seconds") for x in rows),
            "minimum_ttc_episode_mean_s": numeric_stats(x.get("min_cv_obb_ttc_s") for x in rows),
            "critical_unobserved_fraction_episode_mean": numeric_stats(x.get("critical_unobserved_fraction_of_covered_ticks") for x in rows),
        }
    return out


def main() -> None:
    evals: dict[str, dict[str, Any]] = {}
    episodes: dict[str, list[dict[str, Any]]] = {}
    episodes_by_no: dict[str, dict[int, dict[str, Any]]] = {}
    records_by_seed: dict[str, dict[int, dict[str, Any]]] = {}
    partners_by_episode: dict[str, dict[int, dict[str, Any]]] = {}

    for arm, path in ARM_PATHS.items():
        evals[arm] = read_json(path / "evaluation_results.json")
        records_by_seed[arm] = {int(x["seed"]): x for x in evals[arm]["episode_records"]}
        episodes[arm] = read_jsonl(path / "diagnostics" / "eval" / "episodes.jsonl")
        episodes_by_no[arm] = {int(x["episode"]): x for x in episodes[arm]}
        partners_by_episode[arm] = {}
        for ep in episodes[arm]:
            if outcome_name(ep) == "collision":
                partners_by_episode[arm][int(ep["episode"])] = collision_partner(ep)

    seed_sets = {arm: set(records_by_seed[arm]) for arm in ARMS}
    common_seeds = sorted(set.intersection(*seed_sets.values()))
    matrices: dict[str, Any] = {}
    for arm in ("goaloff", "routeveto"):
        matrix = Counter((outcome_letter(records_by_seed["control"][s]), outcome_letter(records_by_seed[arm][s])) for s in common_seeds)
        matrices[f"control_to_{arm}"] = {
            f"{before}->{after}": count for (before, after), count in sorted(matrix.items())
        }
    original_timeout_seeds = [s for s in common_seeds if outcome_letter(records_by_seed["control"][s]) == "T"]
    original_timeout_flow = {
        arm: {
            letter: sum(outcome_letter(records_by_seed[arm][s]) == letter for s in original_timeout_seeds)
            for letter in ("S", "C", "T", "O")
        }
        for arm in ARMS
    }

    # Prepare geometry identities before streaming raw traces so only collision
    # partner actors need to be retained from the raw records.
    collision_partner_id: dict[str, dict[int, str | None]] = {}
    collision_raw_step: dict[str, dict[int, int | None]] = {}
    for arm in ARMS:
        collision_partner_id[arm], collision_raw_step[arm] = {}, {}
        for ep_no, partner in partners_by_episode[arm].items():
            collision_partner_id[arm][ep_no] = partner.get("id")
            collision_raw_step[arm][ep_no] = collision_terminal_step(episodes_by_no[arm][ep_no])

    decision_rows = {
        arm: list(iter_jsonl_gz(ARM_PATHS[arm] / "diagnostics" / "eval" / "decisions.jsonl.gz"))
        for arm in ARMS
    }
    last_policy_input: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        candidate_inputs: dict[int, tuple[int, int]] = {}
        for decision in decision_rows[arm]:
            ep_no = int(decision.get("episode", -1))
            if ep_no not in collision_partner_id[arm]:
                continue
            route_control = decision.get("control") or {}
            raw_sample = route_control.get("route_lane_context_sample_raw_step")
            if raw_sample is None:
                continue
            raw_sample = int(raw_sample)
            terminal = collision_raw_step[arm].get(ep_no)
            if terminal is None or raw_sample >= terminal:
                continue
            old = candidate_inputs.get(ep_no)
            if old is None or raw_sample > old[0]:
                candidate_inputs[ep_no] = (raw_sample, int(decision.get("decision", -1)))
        last_policy_input[arm] = {
            ep_no: {
                "raw_step": raw_sample,
                "decision": decision_no,
                "matched_raw_record": False,
                "observed_neighbor_ids_available": None,
                "partner_observed_by_policy": None,
            }
            for ep_no, (raw_sample, decision_no) in candidate_inputs.items()
        }

    terminal_windows: dict[str, dict[int, deque]] = {arm: defaultdict(lambda: deque(maxlen=31)) for arm in ARMS}
    raw_counts: dict[str, Counter] = {arm: Counter() for arm in ARMS}
    route_labels: dict[str, dict[int, Counter]] = {arm: defaultdict(Counter) for arm in ARMS}
    for arm, path in ARM_PATHS.items():
        for row in iter_jsonl_gz(path / "diagnostics" / "eval" / "raw_steps.jsonl.gz"):
            ep_no = int(row.get("episode", -1))
            raw_counts[arm][ep_no] += 1
            label = row.get("current_lane_reachability_label")
            route_labels[arm][ep_no][str(label) if label is not None else "unknown"] += 1
            terminal_windows[arm][ep_no].append(compact_raw_row(row, collision_partner_id[arm].get(ep_no)))
            sample = last_policy_input[arm].get(ep_no)
            if sample is not None and int(row.get("raw_step", -1)) == sample["raw_step"]:
                observed_ids = row.get("observed_neighbor_ids")
                sample["matched_raw_record"] = True
                sample["observed_neighbor_ids_available"] = observed_ids is not None
                if observed_ids is not None:
                    observed_ids = {str(value) for value in observed_ids}
                    partner_id = collision_partner_id[arm].get(ep_no)
                    sample["partner_observed_by_policy"] = (
                        partner_id in observed_ids or f"vehicle:{partner_id}" in observed_ids
                    ) if partner_id is not None else None
                sample["raw_record_decision_index"] = row.get("decision")

    terminal_window_by_ep: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    collision_details: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for ep in episodes[arm]:
            ep_no = int(ep["episode"])
            rows = list(terminal_windows[arm][ep_no])
            terminal = collision_terminal_step(ep) if outcome_name(ep) == "collision" else int((ep.get("last_snapshot") or {}).get("raw_step") or ep.get("raw_steps") or 0)
            terminal_window_by_ep[arm][ep_no] = summarize_terminal_window(rows, terminal)
            if outcome_name(ep) != "collision":
                continue
            partner = partners_by_episode[arm].get(ep_no, {})
            partner_id = partner.get("id")
            pre_rows = [r for r in rows if terminal is not None and r.get("raw_step") is not None and terminal - 30 <= int(r["raw_step"]) < terminal]
            terminal_info = ep.get("terminal_info") or {}
            snapshot = ep.get("last_snapshot") or {}
            ego = snapshot.get("ego") or {}
            collision_details[arm].append({
                "episode": ep_no,
                "seed": ep.get("seed"),
                "outcome": "collision",
                "from_original_control_timeout_seed": int(ep.get("seed")) in original_timeout_seeds,
                "raw_steps": ep.get("raw_steps"),
                "collision_raw_step": terminal,
                "collision_protocol": {
                    "collision": terminal_info.get("collision"),
                    "geometric_collision": terminal_info.get("geometric_collision"),
                    "raw_sumo_collision": terminal_info.get("raw_sumo_collision"),
                    "raw_sumo_arrived": terminal_info.get("raw_sumo_arrived"),
                },
                "ego_terminal_road_lane": {
                    "road_id": terminal_info.get("current_road_id") or ego.get("road_id"),
                    "lane_id": terminal_info.get("current_lane_id") or ego.get("lane_id"),
                    "position_center_m": ego.get("position"),
                    "speed_mps": ego.get("speed"),
                    "target_speed_mps": terminal_info.get("target_speed"),
                },
                "terminal_route_and_action": {
                    key: terminal_info.get(key) for key in (
                        "planned_next_edge", "current_lane_reachability_label",
                        "current_lane_can_reach_next_edge", "lane_command_requested",
                        "lane_control_request_status", "lane_control_request_reason",
                        "actual_lane_transition_since_previous_decision",
                    )
                },
                "replayed_partner": partner,
                "last_policy_input_before_collision": last_policy_input[arm].get(ep_no),
                "pre_collision_partner_window": summarize_partner_window(ep, pre_rows, partner_id),
                "pre_collision_terminal_aligned_window": terminal_window_by_ep[arm][ep_no],
            })

    # The route-veto sidecar is the only source for pre-veto requests; decisions
    # independently record the exact action delivered to the environment.
    veto_path = ARM_PATHS["routeveto"] / "diagnostics" / "eval" / "route_lane_veto_actions.jsonl"
    veto_rows = read_jsonl(veto_path)
    routeveto_decisions = decision_rows["routeveto"]
    decision_map = {(int(d["episode"]), int(d["decision"])): d for d in routeveto_decisions}
    veto_reason_counts = Counter(x.get("reason", "unknown") for x in veto_rows)
    vetoed = [x for x in veto_rows if x.get("veto_applied")]
    speed_preserved = sum(
        bool(x.get("speed_action_unchanged")) and
        len(x.get("action_requested") or []) >= 1 and len(x.get("action_after_veto") or []) >= 1 and
        abs(float(x["action_requested"][0]) - float(x["action_after_veto"][0])) <= 1e-7
        for x in veto_rows
    )
    action_join = []
    veto_transition = Counter()
    veto_status = Counter()
    veto_action_command = Counter()
    for side in veto_rows:
        key = (int(side["episode_index"]) + 1, int(side["decision_step"]))
        decision = decision_map.get(key)
        delivered = (decision or {}).get("action_env_input") or []
        after = side.get("action_after_veto") or []
        requested = side.get("action_requested") or []
        action_join.append(bool(decision is not None and len(delivered) == len(after) and all(abs(float(a) - float(b)) <= 1e-7 for a, b in zip(delivered, after))))
        if side.get("veto_applied"):
            veto_status[str(side.get("lane_control_request_status") or "unknown")] += 1
            veto_action_command[str(side.get("lane_command_requested")) + "->" + str(side.get("lane_command_after_veto"))] += 1
            next_decision = decision_map.get((key[0], key[1] + 1))
            next_control = (next_decision or {}).get("control") or {}
            transition = next_control.get("actual_lane_transition_since_previous_decision") or {}
            if transition.get("known"):
                veto_transition["known_changed" if transition.get("changed") else "known_unchanged"] += 1
            else:
                veto_transition["transition_unknown_or_unavailable"] += 1

    sidecar_comparison = {
        "sidecar_rows": len(veto_rows),
        "decision_rows": len(routeveto_decisions),
        "veto_applied": len(vetoed),
        "reason_counts_all_decisions": dict(veto_reason_counts),
        "requested_to_after_veto_command_counts": dict(veto_action_command),
        "lane_control_request_status_after_veto": dict(veto_status),
        "veto_action_following_lane_transition": dict(veto_transition),
        "speed_action_unchanged_rows": speed_preserved,
        "delivered_action_matches_sidecar_after_veto_rows": sum(action_join),
        "action_join_mismatches": len(action_join) - sum(action_join),
        "interpretation": "lane_change_request_sent/request status are control requests; only next decision's actual_lane_transition_since_previous_decision is counted as observed lane transition",
    }

    # Verify frozen control reproduces the already completed official goal-only
    # evaluation, seed by seed (the reference is read-only).
    reference = read_json(REFERENCE_EVAL)
    ref_records = {int(x["seed"]): x for x in reference["episode_records"]}
    control_records = records_by_seed["control"]
    control_replay = {
        "reference_path": str(REFERENCE_EVAL),
        "same_seed_set": set(ref_records) == set(control_records),
        "reference_episodes": len(ref_records),
        "control_episodes": len(control_records),
        "outcome_mismatches": sum(outcome_letter(ref_records[s]) != outcome_letter(control_records[s]) for s in set(ref_records) & set(control_records)),
        "traffic_variant_mismatches": sum(ref_records[s].get("traffic_variant") != control_records[s].get("traffic_variant") for s in set(ref_records) & set(control_records)),
        "raw_step_mismatches": sum(ref_records[s].get("raw_steps") != control_records[s].get("raw_steps") for s in set(ref_records) & set(control_records)),
    }

    original_timeout_details = {}
    for arm in ARMS:
        rows = []
        for seed in original_timeout_seeds:
            eval_record = records_by_seed[arm][seed]
            diagnostic = next((e for e in episodes[arm] if int(e.get("seed", -1)) == seed), None)
            ep_no = int(diagnostic["episode"]) if diagnostic else None
            rows.append({
                "seed": seed,
                "traffic_variant": eval_record.get("traffic_variant"),
                "outcome": outcome_letter(eval_record),
                "episode_behavior": episode_behavior_summary(diagnostic) if diagnostic else None,
                "terminal_aligned_last30_raw_ticks": terminal_window_by_ep[arm].get(ep_no),
            })
        original_timeout_details[arm] = rows

    all_collision_summary = {}
    for arm in ARMS:
        details = collision_details[arm]
        partners = [x["replayed_partner"] for x in details if x["replayed_partner"].get("id")]
        partner_windows = [x["pre_collision_partner_window"] for x in details if x["pre_collision_partner_window"].get("partner_id")]
        policy_input_membership = [x.get("last_policy_input_before_collision") or {} for x in details if x["replayed_partner"].get("id")]
        all_collision_summary[arm] = {
            "collision_episodes": len(details),
            "geometry_replay_unique_partner": sum(x["replayed_partner"].get("id") is not None for x in details),
            "geometry_replay_ambiguous_or_missing": sum(x["replayed_partner"].get("id") is None for x in details),
            "terminal_geometric_collision_true": sum(x["collision_protocol"].get("geometric_collision") is True for x in details),
            "terminal_raw_sumo_collision_true": sum(x["collision_protocol"].get("raw_sumo_collision") is True for x in details),
            "terminal_raw_sumo_collision_false": sum(x["collision_protocol"].get("raw_sumo_collision") is False for x in details),
            "terminal_ego_lane_counts": dict(Counter(x["ego_terminal_road_lane"].get("lane_id") or "unknown" for x in details)),
            "partner_flow_prefix_counts": dict(Counter(x.get("background_flow_id_prefix") or "unknown" for x in partners)),
            "partner_road_counts": dict(Counter(x.get("road_id") or "unknown" for x in partners)),
            "partner_lane_counts": dict(Counter(x.get("lane_id") or "unknown" for x in partners)),
            "partner_motion_relation_counts": dict(Counter(x.get("motion_relation") or "unknown" for x in partners)),
            "partner_terminal_speed_mps": numeric_stats(x.get("speed_mps") for x in partners),
            "partner_pre_collision_window_ever_policy_observed": sum(x.get("partner_observed_by_policy_at_least_once_in_window") is True for x in partner_windows),
            "partner_pre_collision_window_never_policy_observed": sum(x.get("partner_observed_by_policy_at_least_once_in_window") is False for x in partner_windows),
            "mean_partner_present_fraction_in_3s_window": numeric_stats(x.get("partner_presence_fraction_of_captured_ticks") for x in partner_windows),
            "mean_partner_observed_true_samples_when_present": numeric_stats((x.get("observed_by_policy_counts_given_partner_present") or {}).get("true", 0) for x in partner_windows),
            "mean_partner_observed_tick_fraction_when_present": numeric_stats(
                ((x.get("observed_by_policy_counts_given_partner_present") or {}).get("true", 0) /
                 max(1, x.get("partner_present_samples_in_saved_vehicle_snapshot", 0)))
                for x in partner_windows
            ),
            "partners_observed_at_least_once_within_pre_collision_window": sum(x.get("partner_observed_by_policy_at_least_once_in_window") is True for x in partner_windows),
            "partners_with_some_unobserved_present_ticks": sum(
                (x.get("observed_by_policy_counts_given_partner_present") or {}).get("false", 0) > 0
                for x in partner_windows
            ),
            "last_policy_input_before_collision_partner_membership": {
                "matched_exact_raw_sample": sum(x.get("matched_raw_record") is True for x in policy_input_membership),
                "partner_observed_true": sum(x.get("partner_observed_by_policy") is True for x in policy_input_membership),
                "partner_observed_false": sum(x.get("partner_observed_by_policy") is False for x in policy_input_membership),
                "unknown_or_unmatched": sum(x.get("partner_observed_by_policy") is None for x in policy_input_membership),
                "interpretation": "The latest logged action input before collision is joined through the decision's route_lane_context_sample_raw_step to that exact raw-step observed_neighbor_ids; no nearest-tick imputation.",
            },
            "pre_collision_ego_target_speed_mps_given_partner_present": numeric_stats(x.get("ego_target_speed_mps_on_partner_present_samples", {}).get("mean") for x in partner_windows),
            "pre_collision_ego_actual_speed_mps_given_partner_present": numeric_stats(x.get("ego_actual_speed_mps_on_partner_present_samples", {}).get("mean") for x in partner_windows),
            "pre_collision_partner_speed_mps_given_partner_present": numeric_stats(x.get("partner_speed_mps_on_partner_present_samples", {}).get("mean") for x in partner_windows),
            "pre_collision_partner_cv_ttc_min_episode_s": numeric_stats(x.get("partner_cv_obb_ttc_s", {}).get("min") for x in partner_windows),
            "collision_from_original_control_timeout_seed_count": sum(x["from_original_control_timeout_seed"] for x in details),
            "collision_from_original_control_timeout_seed_details": [x for x in details if x["from_original_control_timeout_seed"]],
        }

    route_assets = {}
    for arm in ARMS:
        reset = [e.get("reset_info") or {} for e in episodes[arm]]
        route_assets[arm] = {
            "episodes": len(reset),
            "scenario_values": sorted({str(x.get("scenario")) for x in reset}),
            "depart_scale_values": sorted({x.get("depart_scale") for x in reset if x.get("depart_scale") is not None}),
            "overlay_sha256_values": sorted({x.get("high_density_overlay_sha256") for x in reset if x.get("high_density_overlay_sha256")}),
            "traffic_variant_mismatch_vs_control": sum(
                records_by_seed[arm][s].get("traffic_variant") != records_by_seed["control"][s].get("traffic_variant")
                for s in common_seeds
            ),
        }

    result = {
        "protocol": "sortg3_causal_1002_offline_attribution_v1",
        "generated_from_existing_files_only": True,
        "no_environment_or_sumo_execution": True,
        "run_root": str(RUN_ROOT),
        "source_files": {
            arm: {
                "evaluation_results": str(ARM_PATHS[arm] / "evaluation_results.json"),
                "episodes": str(ARM_PATHS[arm] / "diagnostics" / "eval" / "episodes.jsonl"),
                "decisions": str(ARM_PATHS[arm] / "diagnostics" / "eval" / "decisions.jsonl.gz"),
                "raw_steps": str(ARM_PATHS[arm] / "diagnostics" / "eval" / "raw_steps.jsonl.gz"),
                "manifest": str(ARM_PATHS[arm] / "diagnostics" / "eval" / "manifest.json"),
                "route_veto_sidecar": str(veto_path) if arm == "routeveto" else None,
            }
            for arm in ARMS
        },
        "checkpoint_and_policy_integrity": {
            arm: {
                "checkpoint_sha256": evals[arm].get("identity", {}).get("checkpoint_sha256"),
                "base_checkpoint_sha256": evals[arm].get("identity", {}).get("base_checkpoint_sha256"),
                "causal_arm": evals[arm].get("identity", {}).get("causal_arm"),
                "eval_episodes": evals[arm].get("identity", {}).get("episodes"),
                "learning_or_replay_updates": evals[arm].get("identity", {}).get("learning_or_replay_updates"),
                "policy_fingerprint_equal_before_after": (
                    len(evals[arm].get("identity", {}).get("model_policy_fingerprint", [])) == 2 and
                    evals[arm]["identity"]["model_policy_fingerprint"][0].get("sha256") == evals[arm]["identity"]["model_policy_fingerprint"][1].get("sha256")
                ),
                "official_outcome_counts": dict(Counter(outcome_letter(x) for x in evals[arm]["episode_records"])),
                "official_summary": evals[arm].get("summary"),
                "diagnostic_outcome_counts": dict(Counter(outcome_name(x) for x in episodes[arm])),
                "diagnostic_error_count": sum(int(x.get("diagnostic_error_count") or 0) for x in episodes[arm]),
            }
            for arm in ARMS
        },
        "paired_seed_protocol": {
            "common_evaluation_seed_count": len(common_seeds),
            "same_100_seeds_all_arms": all(seed_sets[arm] == seed_sets["control"] for arm in ARMS),
            "same_traffic_variant_all_arms_for_each_seed": all(
                records_by_seed[arm][s].get("traffic_variant") == records_by_seed["control"][s].get("traffic_variant")
                for arm in ARMS for s in common_seeds
            ),
            "control_exact_replay_of_preexisting_goalonly_eval": control_replay,
            "control_to_arm_paired_matrices": matrices,
            "original_control_timeout_seed_count": len(original_timeout_seeds),
            "original_control_timeout_seed_outcomes_by_arm": original_timeout_flow,
            "original_control_timeout_seed_ids": original_timeout_seeds,
        },
        "episode_behavior_metrics_by_outcome": {arm: outcome_metric_groups(episodes[arm]) for arm in ARMS},
        "original_control_timeout_seed_behavior": original_timeout_details,
        "route_veto_intervention_integrity": sidecar_comparison,
        "route_and_asset_identity": route_assets,
        "collision_replay_summary": all_collision_summary,
        "collision_episode_details": collision_details,
        "terminal_aligned_window_definition": "Per episode, target and actual speed are calculated over the final <=30 raw telemetry samples strictly before that episode's own terminal raw step. These are event-aligned trajectory windows, not same-clock counterfactual states.",
        "collision_partner_observation_definition": "Unique saved terminal OBB replay partner is searched in saved raw telemetry vehicles during the preceding <=30 ticks. observed_by_policy is based on raw observed_neighbor_ids membership. Absence from telemetry vehicle snapshots does not prove physical absence; presence does not imply route-candidate membership.",
        "geometry_limitations": {
            "partner_replay_is_independent_sumo_ground_truth": False,
            "meaning": "Re-applies the saved terminal OBB geometry rule to the terminal snapshot; it is a reconstruction of the project collision predicate, not independent SUMO collision truth.",
            "snapshot_positions": "already box-center coordinates; no front-bumper offset is applied again",
            "raw_sumo_collision": "reported independently in episode terminal info; currently geometric collision and SUMO collision events are distinct fields",
        },
        "interpretation_limits": [
            "All three arms reuse one frozen goal-only checkpoint, one training seed, and a fixed 100-seed evaluation pool; paired differences describe this checkpoint only.",
            "Goaloff disables the goal-topology branch; routeveto is a diagnostic action intervention that preserves speed and only replaces a known invalid lateral command with hold. It is not a trained policy or safety proof.",
            "A zero route-lane ineligible time says the sampled current lane directly connects to the planned next edge during recorded samples; it does not establish global reachability or collision causation.",
            "The route veto changes requests, not guaranteed lane transitions. Request sent/status and observed between-decision actual lane transition are kept separate.",
            "Observed partner status is actor-ID membership in saved policy observation IDs, not proof that the policy attended to or used that actor; route-candidate identity is not inferred.",
            "Risk/TTC and geometry replays use the recorded constant-velocity OBB diagnostic and snapshot. A zero or low TTC around an already-labeled collision is descriptive, not an independent causal predictor.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
