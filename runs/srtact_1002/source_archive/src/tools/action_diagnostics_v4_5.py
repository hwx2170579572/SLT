"""v4.5 fusion-decoder diagnostics aligned to paper evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.action_diagnostics_v4_2 import evaluate_with_action_diagnostics_v4_2


def _argmax_first(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: float(values[index]))


def fusion_decoder_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute every branch predicate from serialized decoder evidence."""

    exact_rule = 0
    exact_action = 0
    feasible = 0
    override_valid = 0
    fallback_valid = 0
    overrides = 0
    fallbacks = 0
    override_episodes: set[int] = set()
    fallback_episodes: set[int] = set()
    route_override_episodes: set[int] = set()
    thresholds: set[float] = set()
    source_counts = {"actor": 0, "target_critic": 0}

    for record in records:
        decoder = record["fusion_decoder"]
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        q_values = decoder["minimum_target_twin_q"]
        threshold = float(decoder["actor_non_keep_confidence_threshold"])
        thresholds.add(threshold)

        actor_index = _argmax_first(probabilities)
        actor_confidence = probabilities[actor_index]
        expected_override = actor_index != 1 and actor_confidence >= threshold

        finite_q = [
            float(value) if is_valid else float("-inf")
            for value, is_valid in zip(q_values, valid)
        ]
        target_index = _argmax_first(finite_q)
        target_maximum = max(finite_q)
        if finite_q[1] == target_maximum:
            target_index = 1
        expected_index = actor_index if expected_override else target_index

        stored_actor = int(decoder["actor_selected_lane_index"])
        stored_target = int(decoder["target_selected_lane_index"])
        stored_selected = int(decoder["selected_lane_index"])
        stored_override = bool(decoder["actor_override"])
        exact_rule += int(
            stored_actor == actor_index
            and stored_target == target_index
            and stored_override == expected_override
            and stored_selected == expected_index
        )
        exact_action += int(
            int(record["lane_command"]) == int(decoder["selected_lane"])
        )
        feasible += int(valid[stored_selected])

        episode = int(record["episode"])
        source = str(decoder["selected_source"])
        if source not in source_counts:
            raise ValueError(f"unknown fusion selected_source {source!r}")
        source_counts[source] += 1
        if stored_override:
            overrides += 1
            override_episodes.add(episode)
            override_valid += int(
                stored_actor != 1
                and float(decoder["actor_selected_confidence"]) >= threshold
                and valid[stored_actor]
                and stored_selected == stored_actor
                and source == "actor"
            )
            if (
                bool(record.get("pre_action_route_intent_valid"))
                and int(record.get("pre_action_route_intent", 0)) != 0
            ):
                route_override_episodes.add(episode)
        else:
            fallbacks += 1
            fallback_episodes.add(episode)
            fallback_valid += int(
                stored_selected == stored_target and source == "target_critic"
            )

    count = len(records)
    if count == 0:
        raise ValueError("fusion decoder trace is empty")
    if len(thresholds) != 1:
        raise ValueError(f"fusion threshold changed within trace: {thresholds}")
    return {
        "fusion_decoder_records": count,
        "deterministic_lane_decoder": (
            "actor_confident_non_keep_else_target_critic"
        ),
        "actor_non_keep_confidence_threshold": next(iter(thresholds)),
        "exact_fusion_rule_match_rate": exact_rule / count,
        "exact_fusion_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "actor_override_records": overrides,
        "actor_override_rate": overrides / count,
        "actor_override_episode_count": len(override_episodes),
        "route_window_actor_override_episode_count": len(route_override_episodes),
        "actor_override_predicate_valid_rate": (
            override_valid / overrides if overrides else 1.0
        ),
        "target_fallback_records": fallbacks,
        "target_fallback_rate": fallbacks / count,
        "target_fallback_episode_count": len(fallback_episodes),
        "target_fallback_rule_match_rate": (
            fallback_valid / fallbacks if fallbacks else 1.0
        ),
        "selected_source_counts": source_counts,
    }


def evaluate_with_action_diagnostics_v4_5(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    policy = model.policy
    records_enabled = hasattr(policy, "begin_fusion_decoder_recording")
    if records_enabled:
        policy.begin_fusion_decoder_recording()
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
            policy.end_fusion_decoder_recording() if records_enabled else []
        )

    diagnostics = dict(diagnostics)
    diagnostics["schema_version"] = "topo-scene-v4.5.action-diagnostics/v1"
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
            "fusion decoder/action trace length mismatch: "
            f"{len(decoder_records)} != {len(trace_rows)}"
        )
    for trace_row, decoder_row in zip(trace_rows, decoder_records):
        trace_row["fusion_decoder"] = decoder_row
    metrics = fusion_decoder_metrics(trace_rows)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    diagnostics.update(metrics)
    diagnostics["trace"] = {
        **diagnostics["trace"],
        "records": len(trace_rows),
        "sha256": trace_sha256,
    }
    return report, diagnostics


__all__ = ["evaluate_with_action_diagnostics_v4_5", "fusion_decoder_metrics"]
