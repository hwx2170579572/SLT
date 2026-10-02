"""Read-only, source-backed module audit for the 2026-10-02 sorted pair.

Streams existing diagnostics only; it does not import the simulator, construct
an environment, or invoke a policy.  Run from the workspace with the project's
PyTorch interpreter.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve()
FD = HERE.parents[1]
PROJECT = HERE.parents[2]
WORKSPACE = HERE.parents[4]
RUNS = WORKSPACE / "runs"
OUT = FD / "analysis" / "sorted_module_diagnostics_20261002.json"

RUN_SPECS = {
    "routeaware": (
        "runs/sort2_1001/sac_mlp_d1_st_rt_topo_routeaware_v1__intersection_sorted_depart4p0",
        "fresh sorted routeaware D1; seed0, 100k raw, validation100",
    ),
    "str_rt_3slot": (
        "runs/sort2_1001/sac_mlp_d1_st_rt_3slot__intersection_sorted_depart4p0",
        "fresh sorted ST-RT plus 3slot D1; seed0, 100k raw, validation100",
    ),
    "old_str_rt": (
        "runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0",
        "earlier fresh sorted ST-RT D1; legacy diagnostics schema",
    ),
    "old_topo": (
        "runs/t0930_topo3_retry01/sac_mlp_d1_st_rt_topo__intersection_sorted_depart4p0",
        "earlier fresh sorted Topo D1; legacy diagnostics schema",
    ),
    "old_topo_3slot": (
        "runs/t0930_topo3_retry01/sac_mlp_d1_st_rt_topo_3slot__intersection_sorted_depart4p0",
        "earlier fresh sorted Topo plus 3slot D1; legacy diagnostics schema",
    ),
}

STAGES = (("early", 1, 31667), ("middle", 31668, 63334), ("late", 63335, 100000))

ACTIVATION_METRICS = (
    "spatial_graph_input_rms",
    "spatial_graph_output_rms",
    "spatial_message_delta_rms",
    "spatial_message_delta_relative_rms",
    "history_valid_frames_per_actor_mean",
    "history_single_valid_frame_fraction",
    "temporal_output_rms",
    "temporal_delta_from_last_frame_rms",
    "temporal_delta_relative_rms",
    "social_attention_entropy",
    "social_attention_effective_actor_count",
    "social_attention_ego_self_mass",
    "social_attention_non_ego_mass",
    "social_attention_valid_actor_count_mean",
    "route_valid_path_count_mean",
    "route_empty_path_fraction",
    "route_attention_entropy_nonempty",
    "route_attention_effective_paths_nonempty",
    "topology_attention_entropy",
    "topology_effective_lanes",
    "route_compatible_attention_mass",
    "route_compatible_lane_count_mean",
    "topology_fallback_rate",
    "diagnostic_valid_query_count",
    "topology_valid_lane_count",
    "topology_conflict_directed_edge_count",
    "topology_merge_directed_edge_count",
    "topology_conflict_relation_present",
    "topology_merge_relation_present",
    "relation_pair_valid_count",
    "relation_pair_candidate_count",
    "relation_pair_valid_fraction",
    "relation_pair_mask_has_valid",
    "same_lane_attention_pair_mean",
    "conflict_attention_pair_mean",
    "merge_attention_pair_mean",
    "conflict_or_merge_attention_pair_mean",
    "same_lane_edge_feature_mean",
    "conflict_or_merge_edge_feature_mean",
    "topology_residual_scale",
    "goal_residual_scale",
    "goal_topology_delta_rms",
    "route_reachability_candidate_count_mean",
    "route_reachability_known_node_count_mean",
    "route_reachability_unknown_node_count_mean",
    "route_reachability_query_intersection_count_mean",
    "route_reachability_goal_expand_fraction",
    "route_reachability_goal_bypass_fraction",
    "route_reachability_safe_mask_fallback_fraction",
    "route_reachability_goal_active_sample_count",
    "route_reachability_goal_legal_attention_mass_active",
    "route_reachability_goal_legal_attention_valid_count",
    "goal_topology_valid_candidate_count_mean",
    "goal_topology_active_query_count",
    "goal_topology_attention_entropy_active",
    "goal_topology_effective_nodes_active",
    "goal_topology_delta_rms",
    "route_reachability_goal_delta_norm_mean",
    "route_reachability_goal_delta_norm_active",
    "route_reachability_goal_delta_active_sample_count",
    "ego_slot_rms",
    "social_slot_rms",
    "route_slot_rms",
    "ego_slot_batch_std_mean",
    "social_slot_batch_std_mean",
    "route_slot_batch_std_mean",
    "ego_slot_within_sample_std_mean",
    "social_slot_within_sample_std_mean",
    "route_slot_within_sample_std_mean",
    "slot_ego_rms",
    "slot_social_rms",
    "slot_route_rms",
    "slot_ego_energy_share",
    "slot_social_energy_share",
    "slot_route_energy_share",
    "ego_slot_energy_share",
    "social_slot_energy_share",
    "route_slot_energy_share",
    "slot_sample_energy_correlation_valid",
    "slot_batch_variance_valid",
    "slot_sample_energy_ego_social_corr",
    "slot_sample_energy_social_route_corr",
    "slot_sample_energy_ego_route_corr",
    "diagnostic_batch_size",
    "diagnostic_batch_variance_valid",
)

OPT_METRICS = (
    "train/n_updates",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "diagnostic/q1_mean_sampled",
    "diagnostic/q2_mean_sampled",
    "diagnostic/target_q_mean_sampled",
    "diagnostic/q1_abs_td_mean_sampled",
    "diagnostic/q2_abs_td_mean_sampled",
    "rollout/source_success_rate_last_20",
)

GRAD_GROUPS = (
    "state_encoder",
    "spatial_vehicle_messages",
    "temporal_attention",
    "social_attention",
    "map_route_encoder",
    "route_path_attention",
    "route_goal_attention",
    "route_modulation",
    "topology_lane_encoder",
    "topology_vehicle_query",
    "topology_goal_query",
    "topology_fusion_norms_scales",
    "slot_ego",
    "slot_social",
    "slot_route",
    "mlp_readout",
    # Legacy Topo names are preserved as separate reported groups.
    "topology_core",
    "topology_fusion_goal",
    "topology_vehicle_relations",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def summarize(values: Iterable[Any]) -> dict[str, Any]:
    nums = [n for value in values if (n := number(value)) is not None]
    if not nums:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(nums)
    return {
        "n": len(nums),
        "mean": statistics.fmean(nums),
        "median": statistics.median(nums),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
    }


def freq(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter("<null>" if x is None else str(x) for x in values).items()))


def outcome_of(row: dict[str, Any]) -> str:
    outcome = row.get("outcome")
    if outcome in ("success", "collision", "timeout", "off_route"):
        return outcome
    terminal = row.get("terminal_info") or {}
    for key, label in (("collision", "collision"), ("off_route", "off_route"),
                       ("is_success", "success"), ("max_time", "timeout")):
        if terminal.get(key):
            return label
    return "incomplete"


def phase_for(update: Any) -> str | None:
    value = number(update)
    if value is None:
        return None
    for name, low, high in STAGES:
        if low <= value <= high:
            return name
    return None


def select_metrics(rows_in: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    return {key: summarize((r.get("metrics") or {}).get(key) for r in rows_in) for key in keys
            if any(key in (r.get("metrics") or {}) for r in rows_in)}


def eval_episode_behavior(run_dir: Path, eval_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode_by_number = {int(r.get("episode", i + 1)): r for i, r in enumerate(eval_episodes)}
    decisions_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    decisions_path = run_dir / "diagnostics" / "eval" / "decisions.jsonl.gz"
    for row in rows(decisions_path):
        decisions_by_episode[int(row.get("episode", -1))].append(row)

    per_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lane_status = defaultdict(Counter)
    lane_reason = defaultdict(Counter)
    lane_commands = defaultdict(Counter)
    reachability_actions = defaultdict(Counter)
    target_lane_reachability = defaultdict(Counter)
    target_lane_route_reason = defaultdict(Counter)
    lane_request_target_cross = defaultdict(Counter)
    transition_target_reached = defaultdict(Counter)
    transition = defaultdict(Counter)
    decision_target: dict[str, list[float]] = defaultdict(list)
    decision_actual: dict[str, list[float]] = defaultdict(list)
    decision_speed_action: dict[str, list[float]] = defaultdict(list)
    decision_lanes: dict[str, list[str]] = defaultdict(list)
    collision_roads = defaultdict(Counter)
    collision_speeds = defaultdict(list)
    timeout_ineligible = Counter()
    terminal_wrong_lane = Counter()

    for episode_num, episode in episode_by_number.items():
        outcome = outcome_of(episode)
        terminal = episode.get("terminal_info") or {}
        last = episode.get("last_snapshot") or {}
        last_ego = last.get("ego") or {}
        lane = terminal.get("current_lane_id") or last.get("lane_id") or last_ego.get("lane_id")
        road = terminal.get("current_road_id") or last.get("road_id") or last_ego.get("road_id")
        speed = number(episode.get("mean_actual_speed_mps"))
        route_ineligible = number(episode.get("route_lane_ineligible_ticks"))
        ineligible_stopped = number(episode.get("route_lane_ineligible_stopped_ticks"))
        lane_reachable = terminal.get("current_lane_can_reach_next_edge")
        next_edge = terminal.get("planned_next_edge")
        next_edge_field_present = "planned_next_edge" in terminal
        has_next = bool(next_edge)
        row = {
            "episode": episode_num,
            "seed": episode.get("seed"),
            "outcome": outcome,
            "raw_steps": episode.get("raw_steps"),
            "decisions": episode.get("decisions"),
            "mean_actual_speed_mps": speed,
            "stopped_fraction": number(episode.get("stopped_fraction_of_speed_samples")),
            "low_ttc_fraction": number(episode.get("low_ttc_fraction_of_evaluable_ticks")),
            "min_cv_obb_ttc_s": number(episode.get("min_cv_obb_ttc_s")),
            "route_known_ticks": episode.get("route_lane_known_ticks"),
            "route_unknown_ticks": episode.get("route_lane_unknown_ticks"),
            "route_ineligible_ticks": route_ineligible,
            "route_ineligible_stopped_ticks": ineligible_stopped,
            "route_ineligible_seconds": number(episode.get("route_lane_ineligible_seconds")),
            "route_ineligible_stopped_seconds": number(episode.get("route_lane_ineligible_stopped_seconds")),
            "route_ineligible_stall_max_seconds": number(episode.get("route_lane_ineligible_stall_max_seconds")),
            "terminal_lane": lane,
            "terminal_road": road,
            "terminal_lane_can_reach_next_edge": lane_reachable,
            "terminal_next_edge": terminal.get("planned_next_edge"),
            "terminal_lane_reason": terminal.get("route_lane_status_reason"),
            "last_position": last.get("position") or last_ego.get("position"),
            "last_lane_position": last.get("lane_position", last_ego.get("lane_position")),
            "last_route_index": last.get("route_index", last_ego.get("route_index")),
            "terminal_target_speed_mps": number(terminal.get("effective_target_speed") or terminal.get("target_speed")),
            "terminal_actual_speed_mps": number(terminal.get("actual_speed_mps")),
        }
        per_outcome[outcome].append(row)
        if outcome == "timeout":
            if route_ineligible is not None and route_ineligible > 0:
                timeout_ineligible["any_ineligible"] += 1
            if ineligible_stopped is not None and ineligible_stopped > 0:
                timeout_ineligible["ineligible_and_stopped"] += 1
            if lane_reachable is False and has_next:
                terminal_wrong_lane["known_ineligible_terminal_lane"] += 1
            elif lane_reachable is True and has_next:
                terminal_wrong_lane["known_eligible_terminal_lane"] += 1
            elif next_edge_field_present and has_next:
                terminal_wrong_lane["unknown_terminal_lane"] += 1
            elif next_edge_field_present:
                terminal_wrong_lane["no_planned_next_edge"] += 1
            else:
                terminal_wrong_lane["route_status_unavailable"] += 1

        for collision in episode.get("collision_evidence") or []:
            ego = collision.get("ego") or {}
            loc = f"{ego.get('road_id')}|{ego.get('lane_id')}"
            collision_roads[outcome][loc] += 1
            collision_speeds[outcome].append(ego.get("speed"))

        for decision in decisions_by_episode.get(episode_num, []):
            control = decision.get("control") or {}
            info = decision.get("info") or {}
            lane_status[outcome][str(control.get("lane_control_request_status", "<missing>"))] += 1
            lane_reason[outcome][str(control.get("lane_control_request_reason", "<missing>"))] += 1
            lane_commands[outcome][str(control.get("lane_command_requested", "<missing>"))] += 1
            current_reachability = control.get("current_lane_can_reach_next_edge")
            target_reachability = control.get("expected_target_lane_can_reach_next_edge")
            reachability_actions[outcome][str(current_reachability if current_reachability is not None else "<unknown>")] += 1
            target_lane_reachability[outcome][str(target_reachability if target_reachability is not None else "<unknown_or_no_target>")] += 1
            target_lane_route_reason[outcome][str(control.get("expected_target_lane_route_status_reason", "<unknown_or_no_target>"))] += 1
            lane_request_target_cross[outcome][f"current={current_reachability};target={target_reachability};request={control.get('lane_control_request_status')}" ] += 1
            decision_lanes[outcome].append(str(control.get("current_lane_id", "<missing>")))
            transition_row = control.get("actual_lane_transition_since_previous_decision") or {}
            if transition_row.get("known"):
                transition[outcome]["known"] += 1
                if transition_row.get("changed"):
                    transition[outcome]["actual_lane_changed"] += 1
            transition_target_reached[outcome][str(transition_row.get("previous_expected_target_reached", "<unknown>"))] += 1
            if control.get("lane_change_applied"):
                transition[outcome]["request_reported_applied"] += 1
            target = number(control.get("target_speed_mps"))
            actual = number(info.get("actual_speed_mps"))
            action = control.get("action_clipped") or decision.get("action_env_input") or []
            if target is not None:
                decision_target[outcome].append(target)
            if actual is not None:
                decision_actual[outcome].append(actual)
            if action:
                value = number(action[0])
                if value is not None:
                    decision_speed_action[outcome].append(value)

    out = {}
    for outcome, items in sorted(per_outcome.items()):
        out[outcome] = {
            "episodes": len(items),
            "mean_raw_steps": summarize(x.get("raw_steps") for x in items),
            "mean_actual_speed_episode_mps": summarize(x.get("mean_actual_speed_mps") for x in items),
            "stopped_fraction": summarize(x.get("stopped_fraction") for x in items),
            "low_ttc_fraction": summarize(x.get("low_ttc_fraction") for x in items),
            "min_cv_obb_ttc_s": summarize(x.get("min_cv_obb_ttc_s") for x in items),
            "route_ineligible_ticks_per_episode": summarize(x.get("route_ineligible_ticks") for x in items),
            "route_ineligible_stopped_ticks_per_episode": summarize(x.get("route_ineligible_stopped_ticks") for x in items),
            "route_ineligible_seconds_per_episode": summarize(x.get("route_ineligible_seconds") for x in items),
            "route_ineligible_stopped_seconds_per_episode": summarize(x.get("route_ineligible_stopped_seconds") for x in items),
            "route_ineligible_stall_max_seconds": summarize(x.get("route_ineligible_stall_max_seconds") for x in items),
            "terminal_lane_distribution": freq(x.get("terminal_lane") for x in items),
            "terminal_road_distribution": freq(x.get("terminal_road") for x in items),
            "terminal_can_reach_next_edge": freq(x.get("terminal_lane_can_reach_next_edge") for x in items),
            "terminal_lane_reason": freq(x.get("terminal_lane_reason") for x in items),
            "terminal_position_samples": [
                {k: x.get(k) for k in ("seed", "terminal_road", "terminal_lane", "last_position", "last_lane_position", "last_route_index", "terminal_lane_can_reach_next_edge", "terminal_target_speed_mps", "terminal_actual_speed_mps")}
                for x in items if outcome == "timeout"
            ][:40],
            "decision_target_speed_mps": summarize(decision_target[outcome]),
            "decision_actual_speed_mps_at_decision_end": summarize(decision_actual[outcome]),
            "decision_speed_action": summarize(decision_speed_action[outcome]),
            "lane_command_requested": dict(sorted(lane_commands[outcome].items())),
            "lane_control_request_status": dict(sorted(lane_status[outcome].items())),
            "lane_control_request_reason": dict(sorted(lane_reason[outcome].items())),
            "pre_action_lane_can_reach_next_edge": dict(sorted(reachability_actions[outcome].items())),
            "expected_target_lane_can_reach_next_edge": dict(sorted(target_lane_reachability[outcome].items())),
            "expected_target_lane_route_status_reason": dict(sorted(target_lane_route_reason[outcome].items())),
            "current_vs_target_lane_request_cross_tab": dict(sorted(lane_request_target_cross[outcome].items())),
            "lane_transition": dict(sorted(transition[outcome].items())),
            "previous_expected_target_reached": dict(sorted(transition_target_reached[outcome].items())),
            "decision_current_lane_distribution_top20": dict(Counter(decision_lanes[outcome]).most_common(20)),
            "collision_evidence_location_count": dict(sorted(collision_roads[outcome].items())),
            "collision_evidence_ego_speed_mps": summarize(collision_speeds[outcome]),
        }

    timeout_rows = per_outcome.get("timeout", [])
    timeout_episode_ids = {int(item["episode"]) for item in timeout_rows}
    previous_raw: dict[int, dict[str, Any]] = {}
    first_ineligible_events: dict[int, dict[str, Any]] = {}
    first_ineligible_classes = Counter()
    raw_path = run_dir / "diagnostics" / "eval" / "raw_steps.jsonl.gz"
    for raw in rows(raw_path) or []:
        episode_num = int(raw.get("episode", -1))
        if episode_num not in timeout_episode_ids or episode_num in first_ineligible_events:
            continue
        previous = previous_raw.get(episode_num)
        if raw.get("current_lane_can_reach_next_edge") is False:
            previous_control = (previous or {}).get("lane_control") or {}
            previous_lane_eligible = (previous or {}).get("current_lane_can_reach_next_edge")
            previous_target_eligible = previous_control.get(
                "expected_target_lane_can_reach_next_edge"
            )
            request_status = previous_control.get("lane_control_request_status")
            if previous_lane_eligible is True and previous_target_eligible is False and request_status == "request_sent":
                event_class = "requested_known_ineligible_target_from_eligible_lane"
            elif previous_lane_eligible is True and request_status == "request_sent":
                event_class = "request_sent_from_eligible_lane_target_legality_unknown_or_eligible"
            elif previous_lane_eligible is True:
                event_class = "eligible_previous_lane_without_immediately_preceding_sent_request"
            else:
                event_class = "first_observed_ineligible_without_eligible_previous_sample"
            first_ineligible_classes[event_class] += 1
            current_ego = raw.get("ego") or {}
            previous_ego = (previous or {}).get("ego") or {}
            first_ineligible_events[episode_num] = {
                "seed": next((x.get("seed") for x in timeout_rows if int(x["episode"]) == episode_num), None),
                "event_class": event_class,
                "previous_raw_step": (previous or {}).get("raw_step"),
                "previous_decision": (previous or {}).get("decision"),
                "previous_lane_id": (previous or {}).get("current_lane_id"),
                "previous_lane_can_reach_next_edge": previous_lane_eligible,
                "previous_action_clipped": previous_control.get("action_clipped"),
                "previous_lane_command_requested": previous_control.get("lane_command_requested"),
                "previous_request_status": request_status,
                "previous_request_reason": previous_control.get("lane_control_request_reason"),
                "previous_target_lane_id": previous_control.get("expected_target_lane_id"),
                "previous_target_lane_can_reach_next_edge": previous_target_eligible,
                "first_ineligible_raw_step": raw.get("raw_step"),
                "first_ineligible_decision": raw.get("decision"),
                "first_ineligible_lane_id": raw.get("current_lane_id"),
                "first_ineligible_lane_position_m": number(current_ego.get("lane_position")),
                "first_ineligible_speed_mps": number(current_ego.get("speed")),
            }
        previous_raw[episode_num] = raw

    return {
        "episodes": sum(len(v) for v in per_outcome.values()),
        "outcome_counts": {k: len(v) for k, v in sorted(per_outcome.items())},
        "by_outcome": out,
        "timeout_lane_classification": {
            "counts": dict(timeout_ineligible),
            "terminal_route_status_counts": dict(terminal_wrong_lane),
            "timeout_episode_rows": timeout_rows[:40],
        },
        "first_route_ineligible_transition": {
            "definition": "first raw row where current lane is known not to reach the planned next edge; previous raw row/control is retained to identify the triggering lane request when available",
            "classified_timeout_episodes": len(first_ineligible_events),
            "class_counts": dict(first_ineligible_classes),
            "events": [first_ineligible_events[k] for k in sorted(first_ineligible_events)],
        },
    }


def train_representation(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "diagnostics" / "train" / "representation.jsonl"
    all_rows = list(rows(path) or [])
    by_source = Counter(row.get("source", "<missing>") for row in all_rows)
    activations = [r for r in all_rows if r.get("source") == "train_forward_activation"]
    gradients = [r for r in all_rows if r.get("source") == "critic_td_gradient_pre_clip"]
    updates = [r for r in all_rows if r.get("source") == "critic_td_parameter_update"]
    site_counts = Counter()
    for r in activations:
        m = r.get("metrics") or {}
        site_counts[str(m.get("diagnostic_sample_site_code", "<missing>"))] += 1

    # Only critic TD-current site (code 3) is treated as the comparable batch
    # activation sample. Other sites remain available in source/callsite counts.
    critic_activations = [
        r for r in activations
        if number((r.get("metrics") or {}).get("diagnostic_sample_site_code")) == 3
    ]
    stage_activations: dict[str, Any] = {}
    for stage, low, high in STAGES:
        bucket = []
        for r in critic_activations:
            m = r.get("metrics") or {}
            u = number(m.get("diagnostic_sample_update_index"))
            if u is None or u < low or u > high:
                continue
            bucket.append(r)
        stage_activations[stage] = {
            "n_critic_td_current_forward_samples": len(bucket),
            "update_min": min((number((r.get("metrics") or {}).get("diagnostic_sample_update_index")) for r in bucket), default=None),
            "update_max": max((number((r.get("metrics") or {}).get("diagnostic_sample_update_index")) for r in bucket), default=None),
            "metrics": select_metrics(bucket, ACTIVATION_METRICS),
        }

    gradient_stages = {}
    update_stages = {}
    for stage, low, high in STAGES:
        g_bucket = [r for r in gradients if low <= (number(r.get("updates")) or -1) <= high]
        u_bucket = [r for r in updates if low <= (number(r.get("updates")) or -1) <= high]
        g_summary, u_summary = {}, {}
        for group in GRAD_GROUPS:
            prefix = f"encoder_grad/{group}/"
            if any(prefix + key in (r.get("metrics") or {}) for r in g_bucket for key in ("gradient_l2", "gradient_parameter_fraction")):
                g_summary[group] = {
                    metric: summarize((r.get("metrics") or {}).get(prefix + metric) for r in g_bucket)
                    for metric in ("gradient_l2", "gradient_to_parameter_l2", "gradient_parameter_fraction", "missing_gradient_elements", "gradient_nonzero_elements", "parameter_elements", "gradient_parameter_elements")
                }
            uprefix = f"encoder_update/{group}/"
            if any(uprefix + key in (r.get("metrics") or {}) for r in u_bucket for key in ("delta_l2", "optimizer_owned_elements")):
                u_summary[group] = {
                    metric: summarize((r.get("metrics") or {}).get(uprefix + metric) for r in u_bucket)
                    for metric in ("delta_l2", "relative_delta_l2", "delta_max_abs", "changed_elements", "optimizer_owned_elements", "optimizer_nonowned_elements")
                }
        gradient_stages[stage] = {"event_count": len(g_bucket), "groups": g_summary}
        update_stages[stage] = {"event_count": len(u_bucket), "groups": u_summary}

    return {
        "rows": len(all_rows),
        "source_counts": dict(by_source),
        "activation_sample_site_code_counts": dict(site_counts),
        "critic_td_current_site_code": 3,
        "critic_td_current_activation_stages": stage_activations,
        "gradient_stages": gradient_stages,
        "parameter_update_stages": update_stages,
        "first_rows_source_and_update": [
            {"source": r.get("source"), "updates": r.get("updates"), "raw_steps": r.get("raw_steps"), "sample_update": (r.get("metrics") or {}).get("diagnostic_sample_update_index"), "site_code": (r.get("metrics") or {}).get("diagnostic_sample_site_code"), "timing": r.get("timing")}
            for r in all_rows[:8]
        ],
    }


def eval_representation(run_dir: Path, eval_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "diagnostics" / "eval" / "representation.jsonl"
    all_rows = list(rows(path) or [])
    outcome_by_zero_based = {
        int(r.get("episode", i + 1)) - 1: outcome_of(r)
        for i, r in enumerate(eval_episodes)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    site_codes = Counter()
    for row in all_rows:
        metrics = row.get("metrics") or {}
        episode_idx = metrics.get("evaluation_episode_index")
        if episode_idx is None:
            episode_idx = int(row.get("episode_index", 1)) - 1
        outcome = outcome_by_zero_based.get(int(episode_idx), "unmatched")
        grouped[outcome].append(row)
        site_codes[str(metrics.get("diagnostic_sample_site_code", "<missing>"))] += 1

    slot_readout_keys = (
        "actor_readout_ego_preactivation_l2",
        "actor_readout_social_preactivation_l2",
        "actor_readout_route_preactivation_l2",
        "actor_readout_bias_l2",
        "actor_readout_first_linear_output_l2",
        "actor_readout_decomposition_error_l2",
        "actor_readout_decomposition_relative_error",
    )
    per_outcome = {}
    for outcome, bucket in sorted(grouped.items()):
        selected_goal_candidate_counts = Counter()
        for row in bucket:
            count = number((row.get("metrics") or {}).get("goal_topology_valid_candidate_count_mean"))
            if count is not None:
                selected_goal_candidate_counts[str(int(count))] += 1
        per_outcome[outcome] = {
            "prediction_samples": len(bucket),
            "episode_count": len({(r.get("metrics") or {}).get("evaluation_episode_index", r.get("episode_index")) for r in bucket}),
            "metrics": select_metrics(bucket, ACTIVATION_METRICS + slot_readout_keys),
            "routeaware_selected_goal_candidate_count_histogram": dict(sorted(selected_goal_candidate_counts.items())),
        }
    return {
        "rows": len(all_rows),
        "source_counts": dict(Counter(r.get("source", "<missing>") for r in all_rows)),
        "sample_site_code_counts": dict(site_codes),
        "per_outcome": per_outcome,
        "slot_readout_interpretation": "exact additive slice contributions to first actor Linear preactivation; not causal credit; eval batch size 1 means batch-variance/correlation metrics are invalid",
    }


GOAL_INCREMENT_METRICS = (
    "goal_topology_delta_rms",
    "route_reachability_goal_delta_norm_mean",
    "route_reachability_goal_delta_norm_active",
    "route_reachability_goal_delta_active_sample_count",
)


def routeaware_goal_increment_audit(run_dir: Path) -> dict[str, Any]:
    """Summarize already-logged post-Delta-LN goal increments for routeaware only."""
    train_path = run_dir / "diagnostics" / "train" / "representation.jsonl"
    eval_path = run_dir / "diagnostics" / "eval" / "representation.jsonl"

    train_rows = []
    for row in rows(train_path) or []:
        metrics = row.get("metrics") or {}
        if row.get("source") == "train_forward_activation" and number(
            metrics.get("diagnostic_sample_site_code")
        ) == 3:
            train_rows.append(row)

    train_stages = {}
    for stage, low, high in STAGES:
        bucket = []
        updates = []
        for row in train_rows:
            metrics = row.get("metrics") or {}
            update = number(metrics.get("diagnostic_sample_update_index"))
            if update is not None and low <= update <= high:
                bucket.append(row)
                updates.append(update)
        metric_summaries = select_metrics(bucket, GOAL_INCREMENT_METRICS)
        active_count_values = [
            number((row.get("metrics") or {}).get("route_reachability_goal_delta_active_sample_count"))
            for row in bucket
        ]
        active_count_values = [value for value in active_count_values if value is not None]
        if "route_reachability_goal_delta_active_sample_count" in metric_summaries:
            metric_summaries["route_reachability_goal_delta_active_sample_count"]["sum"] = sum(active_count_values)
        train_stages[stage] = {
            "activation_rows": len(bucket),
            "update_min": min(updates) if updates else None,
            "update_max": max(updates) if updates else None,
            "metrics": metric_summaries,
        }

    eval_rows = []
    for row in rows(eval_path) or []:
        metrics = row.get("metrics") or {}
        if row.get("source") == "eval_policy" and number(
            metrics.get("diagnostic_sample_site_code")
        ) == 6:
            eval_rows.append(row)
    eval_metrics = select_metrics(eval_rows, GOAL_INCREMENT_METRICS)
    eval_active_count_values = [
        number((row.get("metrics") or {}).get("route_reachability_goal_delta_active_sample_count"))
        for row in eval_rows
    ]
    eval_active_count_values = [value for value in eval_active_count_values if value is not None]
    if "route_reachability_goal_delta_active_sample_count" in eval_metrics:
        eval_metrics["route_reachability_goal_delta_active_sample_count"]["sum"] = sum(eval_active_count_values)

    return {
        "semantics": {
            "goal_topology_delta_rms": "RMS of route_context_final - route_base; route_context_final includes the Delta-LN correction and route-reachability gate when enabled.",
            "route_reachability_goal_delta_norm_mean": "Mean per-sample vector norm of route_context_final - route_base over the batch.",
            "route_reachability_goal_delta_norm_active": "Mean per-sample vector norm over rows with an active legal goal candidate.",
            "route_reachability_goal_delta_active_sample_count": "Number of active legal-goal rows per sampled batch/policy prediction; sum is included across logged rows.",
        },
        "training": {
            "source_file": str(train_path),
            "source": "train_forward_activation",
            "diagnostic_sample_site_code": 3,
            "site_semantics": "critic TD-current forward; same callsite used for existing early/middle/late activation aggregates",
            "stages": train_stages,
        },
        "evaluation": {
            "source_file": str(eval_path),
            "source": "eval_policy",
            "diagnostic_sample_site_code": 6,
            "prediction_rows": len(eval_rows),
            "metrics": eval_metrics,
        },
        "historical_scope_note": "This describes the new routeaware representation stream only; it does not retroactively add post-Delta-LN goal telemetry to the older t0930 Topo logs.",
    }


def refresh_routeaware_goal_increment_only() -> None:
    """Update only routeaware goal-increment summaries from its two representation JSONL files."""
    result = load_json(OUT)
    if not result:
        raise FileNotFoundError(f"Existing derived diagnostic JSON is missing: {OUT}")
    routeaware_dir = WORKSPACE / Path(RUN_SPECS["routeaware"][0])
    result["runs"]["routeaware"]["post_ln_goal_increment_audit"] = routeaware_goal_increment_audit(routeaware_dir)
    legacy = result.get("legacy_topo_activation_excerpt") or {}
    definitions = legacy.get("definitions_and_sampling") or {}
    if definitions.get("not_logged"):
        definitions["not_logged"] = (
            "For the legacy t0930 Topo/Topo+3slot streams only: no actor parameter-gradient norm, direct policy entropy/log_std, "
            "post-Delta-LN goal-delta norm, frozen identical-input representation/Q comparison, or replay-wide Q calibration. "
            "The new routeaware representation stream records its actual route_context_final - route_base goal increment."
        )
    legacy["definitions_and_sampling"] = definitions
    result["legacy_topo_activation_excerpt"] = legacy
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    audit = result["runs"]["routeaware"]["post_ln_goal_increment_audit"]
    print(f"updated {OUT}")
    print(json.dumps({"training": audit["training"]["stages"], "evaluation": audit["evaluation"]}, ensure_ascii=False))


def optimization_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "diagnostics" / "train" / "optimization.jsonl"
    all_rows = list(rows(path) or [])
    stages = {}
    for stage, low, high in STAGES:
        bucket = [r for r in all_rows if low <= (number(r.get("raw_steps")) or -1) <= high]
        stages[stage] = {
            "samples": len(bucket),
            "raw_steps_min": min((number(r.get("raw_steps")) for r in bucket), default=None),
            "raw_steps_max": max((number(r.get("raw_steps")) for r in bucket), default=None),
            "latest_available_logger_values_not_same_update": True,
            "metrics": {
                key: summarize((r.get("metrics") or {}).get(key) for r in bucket)
                for key in OPT_METRICS
                if any(key in (r.get("metrics") or {}) for r in bucket)
            },
        }
    return {"rows": len(all_rows), "stages": stages}


def run_analysis(name: str, relative: str, description: str) -> dict[str, Any]:
    run_dir = WORKSPACE / Path(relative)
    eval_file = run_dir / "evaluation_results.json"
    evaluation = load_json(eval_file) or {}
    summary = evaluation.get("summary") or {}
    eval_episodes = list(rows(run_dir / "diagnostics" / "eval" / "episodes.jsonl") or [])
    train_episodes = list(rows(run_dir / "diagnostics" / "train" / "episodes.jsonl") or [])
    diag_summaries = {
        phase: load_json(run_dir / "diagnostics" / phase / "summary.json")
        for phase in ("train", "eval")
    }
    args = load_json(run_dir / "arguments.json") or {}
    manifest = load_json(run_dir / "experiment_manifest.json") or {}
    training = load_json(run_dir / "training_complete.json") or {}
    best_training = load_json(run_dir / "best_training_success.json") or {}
    checkpoint_files = sorted((run_dir / "checkpoints").glob("ckpt_raw_*_steps.zip"))
    checkpoint_evaluation_files = sorted(
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*.json")
        if "checkpoint" in str(p.parent).lower()
        and ("eval" in p.name.lower() or "evaluation" in p.name.lower())
    )
    return {
        "method_key": name,
        "description": description,
        "run_dir": str(run_dir),
        "files": {
            "arguments": str(run_dir / "arguments.json"),
            "experiment_manifest": str(run_dir / "experiment_manifest.json"),
            "evaluation_results": str(eval_file),
            "train_diagnostics_summary": str(run_dir / "diagnostics" / "train" / "summary.json"),
            "eval_diagnostics_summary": str(run_dir / "diagnostics" / "eval" / "summary.json"),
        },
        "configuration": {
            "method": args.get("method") or manifest.get("method"),
            "scenario": args.get("scenario") or manifest.get("scenario"),
            "seed": args.get("seed") or manifest.get("training_seed"),
            "raw_budget": args.get("raw_budget") or args.get("raw_step_budget") or manifest.get("raw_budget"),
            "updates": training.get("updates"),
            "action_repeat": args.get("action_repeat"),
            "evaluation_episodes": summary.get("episodes"),
            "evaluation_return_protocol": summary.get("evaluation_return_protocol_version"),
            "model_config": {k: args.get(k) for k in ("use_route", "use_topology", "use_slots", "use_incremental_slots", "use_route_reachability", "use_graph_slt", "use_sbs", "representation_coef", "slot_balance_coef") if k in args},
        },
        "evaluation_summary": summary,
        "saved_checkpoint_review": {
            "raw_step_checkpoint_files": [p.name for p in checkpoint_files],
            "checkpoint_evaluation_summary_files": checkpoint_evaluation_files,
            "independent_validation_curve_available": bool(checkpoint_evaluation_files),
            "best_training_success_selection": best_training,
            "interpretation": "best_training_success is a rolling training-episode selector, not an independent validation result; checkpoint archives alone do not establish a validation curve",
        },
        "evaluation_episode_rows": len(eval_episodes),
        "evaluation_seed_count": len({r.get("seed") for r in eval_episodes}),
        "evaluation_behavior": eval_episode_behavior(run_dir, eval_episodes) if eval_episodes else None,
        "eval_representation_by_outcome": eval_representation(run_dir, eval_episodes) if eval_episodes else None,
        "train_episode_diagnostic_outcomes": dict(Counter(outcome_of(r) for r in train_episodes)),
        "train_episode_rows": len(train_episodes),
        "train_summary_counts": {
            phase: {
                "diagnostic_error_count": (diag_summaries[phase] or {}).get("diagnostic_error_count"),
                "raw_records": (diag_summaries[phase] or {}).get("raw_records"),
                "decision_records": (diag_summaries[phase] or {}).get("decision_records"),
                "episodes_finished": (diag_summaries[phase] or {}).get("episodes_finished"),
                "optimization_samples": (diag_summaries[phase] or {}).get("optimization_samples"),
                "representation_samples": (diag_summaries[phase] or {}).get("representation_samples"),
                "representation_samples_by_source": (diag_summaries[phase] or {}).get("representation_samples_by_source"),
            }
            for phase in ("train", "eval")
        },
        "train_representation": train_representation(run_dir),
        "post_ln_goal_increment_audit": routeaware_goal_increment_audit(run_dir) if name == "routeaware" else None,
        "optimization": optimization_summary(run_dir),
    }


def paired_matrix(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def outcomes(section):
        rows_by_seed = {}
        eval_rows = list(rows(Path(section["files"]["evaluation_results"]).parent / "diagnostics" / "eval" / "episodes.jsonl") or [])
        for r in eval_rows:
            rows_by_seed[int(r["seed"])] = outcome_of(r)
        return rows_by_seed
    oa, ob = outcomes(a), outcomes(b)
    shared = sorted(set(oa) & set(ob))
    matrix = Counter((oa[s], ob[s]) for s in shared)
    return {
        "pair": [a["method_key"], b["method_key"]],
        "matched_eval_seed_count": len(shared),
        "seed_set_identical": len(oa) == len(ob) == len(shared),
        "matrix_rows_method_a_columns_method_b": {
            x: {y: matrix[(x, y)] for y in ("success", "collision", "timeout", "off_route", "incomplete")}
            for x in ("success", "collision", "timeout", "off_route", "incomplete")
            if any(matrix[(x, y)] for y in ("success", "collision", "timeout", "off_route", "incomplete"))
        },
    }


def legacy_topo_activation_excerpt() -> dict[str, Any] | None:
    """Retain the already-reviewed replay-sample summary without mixing its cadence with new logs."""
    audit_path = FD / "analysis" / "topo3_retry01_optimization_audit.json"
    audit = load_json(audit_path)
    if not audit:
        return None
    definitions = audit.get("definitions_and_sampling") or {}
    if definitions.get("not_logged"):
        definitions["not_logged"] = (
            "For the legacy t0930 Topo/Topo+3slot streams only: no actor parameter-gradient norm, direct policy entropy/log_std, "
            "post-Delta-LN goal-delta norm, frozen identical-input representation/Q comparison, or replay-wide Q calibration. "
            "The new routeaware representation stream records its actual route_context_final - route_base goal increment."
        )
    audit["definitions_and_sampling"] = definitions
    metric_names = (
        "topology_residual_scale",
        "goal_residual_scale",
        "topology_attention_entropy",
        "topology_effective_lanes",
        "route_compatible_attention_mass",
        "topology_fallback_rate",
        "same_lane_edge_feature_mean",
        "conflict_or_merge_edge_feature_mean",
        "ego_slot_energy_share",
        "social_slot_energy_share",
        "route_slot_energy_share",
    )
    selected = {}
    for method in ("topo", "topo_3slot"):
        run = (audit.get("runs") or {}).get(method) or {}
        selected[method] = {
            "run_path": run.get("path"),
            "representation_sampling": run.get("representation_sampling"),
            "activation_by_segment": {
                segment: {
                    "row_count": values.get("row_count"),
                    "metrics": {
                        key: (values.get("metrics") or {}).get(key)
                        for key in metric_names
                        if key in (values.get("metrics") or {})
                    },
                }
                for segment, values in (run.get("train_activation_by_segment") or {}).items()
            },
        }
    return {
        "source_file": str(audit_path),
        "definitions_and_sampling": audit.get("definitions_and_sampling"),
        "runs": selected,
        "comparability_warning": "These prior activation aggregates come from train_replay source-code 1 with mixed batch sizes (mostly 32 plus batch-1 rows), and raw-step segments; current sort2 activations use critic-TD-current site code 3 and update segments. Use within-run trends, not direct numeric cross-run deltas.",
    }


def main() -> None:
    result = {
        "artifact_type": "sorted_module_diagnostic_audit",
        "created_local_date": "2026-10-02",
        "run_root": str(RUNS / "sort2_1001"),
        "source_records": {
            "research_context": str(FD / "RESEARCH_CONTEXT.md"),
            "implementation_map": str(FD / "analysis" / "full_mst_slt_implementation.md"),
            "experiment_history": str(FD / "analysis" / "experiment_history.md"),
            "diagnostics_protocol": str(FD / "analysis" / "module_diagnostics_protocol_20261001.md"),
            "sorted_pair_plan": str(FD / "analysis" / "sorted_routeaware_3slot_plan_20261001.md"),
            "prior_topo_3slot_diagnostic_audit": str(FD / "analysis" / "topo3_retry01_optimization_audit.json"),
        },
        "analysis_scope": "existing summaries, eval/train episodes, optimization, representation, decision and raw-step logs; first timeout route-ineligibility transition is reconstructed offline; no SUMO, training, evaluation rollout, or model load",
        "stage_definition": "early updates 1-31667; middle 31668-63334; late 63335-100000; activation analyses additionally restrict to critic TD-current callsite code 3 when present",
        "runs": {
            name: run_analysis(name, *spec)
            for name, spec in RUN_SPECS.items()
        },
        "legacy_topo_activation_excerpt": legacy_topo_activation_excerpt(),
    }
    result["paired_comparisons"] = [
        paired_matrix(result["runs"]["routeaware"], result["runs"]["str_rt_3slot"]),
        paired_matrix(result["runs"]["old_topo"], result["runs"]["old_topo_3slot"]),
    ]
    result["comparability_notes"] = [
        "The current routeaware and ST-RT+3slot fresh runs share sorted depart4p0, seed0, raw budget, checkpoint selection, 100 validation seeds and shaped-return protocol.",
        "Older d0929 ST-RT and t0930 Topo/Topo+3slot are same named sorted/depart4 runs, but use prior source/diagnostic/reward-output revisions; compare outcomes and behavior descriptively, not as a byte-identical implementation reproduction.",
        "The d0929 STRT run has no representation.jsonl activation/gradient/update stream, so its 63/37/0 result cannot be explained by the saved module evidence. The prior Topo audit has within-run module traces and critic gradients but no parameter-update snapshots; its replay-source activations are not numerically interchangeable with the current critic-TD-current activation sample.",
        "All statistics are one training seed per method. Evaluation episodes reuse 30 fixed traffic templates; 100 episode seeds are not 100 training seeds.",
        "Route-aware goal legal attention mass is constrained by a hard mask by construction. It is not evidence that the policy's lane command is legal; action-control lane request, request status, and observed lane transition are reported separately.",
        "Evaluation raw-step rows also contain sampled current/next-edge lane reachability and the active decision control record, so first movement into an ineligible lane can be reconstructed offline; no additional rollout is required. The lane-command 'applied' flag means a changeLaneRelative request was issued, while actual lane transition is recorded separately.",
        "Both current runs save 10k-step checkpoint archives, but no per-checkpoint independent validation summaries were found. The saved best_training_success record is selected from a rolling window of training episodes; only final-model validation supports held-out performance claims.",
        "No extra normal-flow fields or rerun are needed to locate first entry into a route-ineligible lane: eval raw_steps retain lane position, planned next edge, current-lane reachability, active control/action/request/target-lane reachability, and per-raw-step observed lane. The explicit SUMO request disposition/rejection cause beyond request-call status and observed transition is not logged.",
        "Collision-time minimum TTC can be zero by construction; it is descriptive and not an independent causal risk predictor.",
        "Activation magnitudes and attention weights are observational. Gradients are critic TD after backward and before clipping; parameter deltas are critic-optimizer-owned updates, not direct actor credit. Batch-one eval variance fields are invalid.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for name in ("routeaware", "str_rt_3slot", "old_str_rt", "old_topo", "old_topo_3slot"):
        run = result["runs"][name]
        print(name, json.dumps({"summary": run["evaluation_summary"], "diag": run["train_summary_counts"], "outcomes": (run["evaluation_behavior"] or {}).get("outcome_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
