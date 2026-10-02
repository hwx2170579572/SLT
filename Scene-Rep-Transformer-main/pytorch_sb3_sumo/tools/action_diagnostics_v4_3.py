"""v4.3 target-critic decoder diagnostics aligned to paper evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from envs.sumo.decision_alignment_v4 import lane_command_index
from tools.action_diagnostics_v4_2 import (
    _summary,
    evaluate_with_action_diagnostics_v4_2,
)


def _target_event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    route_margins: list[float] = []
    exact = 0
    for record in records:
        decoder = record["target_critic_decoder"]
        exact += int(int(record["lane_command"]) == int(decoder["selected_lane"]))
        intent = int(record["pre_action_route_intent"])
        if not bool(record["pre_action_route_intent_valid"]) or intent == 0:
            continue
        q_values = decoder["minimum_target_twin_q"]
        route_q = q_values[lane_command_index(intent)]
        keep_q = q_values[lane_command_index(0)]
        if route_q is None or keep_q is None:
            raise ValueError("route window contains an infeasible route/keep target-Q")
        margin = float(route_q) - float(keep_q)
        record["target_route_minus_keep_q_margin"] = margin
        route_margins.append(margin)
        by_episode.setdefault(int(record["episode"]), []).append(record)

    applied: list[bool] = []
    applied_positive: list[bool] = []
    per_episode: list[dict[str, Any]] = []
    for episode, episode_rows in sorted(by_episode.items()):
        matched_applied = [
            row
            for row in episode_rows
            if bool(row["route_intent_match"])
            and bool(row["lane_change_applied"])
        ]
        has_applied = bool(matched_applied)
        has_positive = any(
            float(row["target_route_minus_keep_q_margin"]) > 0.0
            for row in matched_applied
        )
        applied.append(has_applied)
        applied_positive.append(has_positive)
        per_episode.append(
            {
                "episode": episode,
                "route_window_rows": len(episode_rows),
                "event_applied_match": has_applied,
                "event_applied_match_positive_target_q_margin": has_positive,
            }
        )
    count = len(per_episode)
    return {
        "target_decoder_records": len(records),
        "exact_target_critic_argmax_rate": exact / len(records) if records else None,
        "target_route_event_episode_count": count,
        "target_route_event_applied_match_episode_rate": (
            sum(applied) / count if count else None
        ),
        "applied_match_positive_target_q_margin_episode_rate": (
            sum(applied_positive) / count if count else None
        ),
        "target_route_minus_keep_q_margin": _summary(route_margins),
        "target_route_event_per_episode": per_episode,
    }


def evaluate_with_action_diagnostics_v4_3(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    policy = model.policy
    records_enabled = hasattr(policy, "begin_target_decoder_recording")
    if records_enabled:
        policy.begin_target_decoder_recording()
    try:
        report, diagnostics = evaluate_with_action_diagnostics_v4_2(
            model,
            env,
            episodes=episodes,
            seed=seed,
            trace_path=trace_path,
        )
    finally:
        decoder_records = (
            policy.end_target_decoder_recording() if records_enabled else []
        )

    diagnostics = dict(diagnostics)
    diagnostics["schema_version"] = "topo-scene-v4.3.action-diagnostics/v1"
    if not records_enabled:
        diagnostics["deterministic_lane_decoder"] = "parent_comparator_control"
        return report, diagnostics

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(trace_rows) != len(decoder_records):
        raise ValueError(
            "target decoder/action trace length mismatch: "
            f"{len(decoder_records)} != {len(trace_rows)}"
        )
    for trace_row, decoder_row in zip(trace_rows, decoder_records):
        trace_row["target_critic_decoder"] = decoder_row
    target_metrics = _target_event_metrics(trace_rows)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    diagnostics.update(target_metrics)
    diagnostics["deterministic_lane_decoder"] = (
        "argmax_feasible_min_target_twin_q"
    )
    diagnostics["trace"] = {
        **diagnostics["trace"],
        "records": len(trace_rows),
        "sha256": trace_sha256,
    }
    return report, diagnostics


__all__ = ["evaluate_with_action_diagnostics_v4_3"]
