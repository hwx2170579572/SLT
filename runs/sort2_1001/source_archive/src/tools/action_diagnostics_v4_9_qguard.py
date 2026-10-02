"""Auditable action diagnostics for the v4.9a critic-regret guard."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from algos.sb3_torch.hybrid_policy_v4_9_qguard import DECODER
from tools.action_diagnostics_v4_2 import evaluate_with_action_diagnostics_v4_2


def _argmax_first(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: float(values[index]))


def q_guard_decoder_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently recompute the Q-regret guard from serialized evidence."""

    if not records:
        raise ValueError("Q-guard decoder trace is empty")
    exact_rule = 0
    exact_action = 0
    feasible = 0
    base_overrides = 0
    final_overrides = 0
    vetoes = 0
    veto_valid = 0
    override_valid = 0
    fallback_valid = 0
    thresholds: set[float] = set()
    tolerances: set[float] = set()
    source_counts = {"actor": 0, "target_critic": 0}

    for record in records:
        decoder = record["q_guard_fusion_decoder"]
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        q_values = decoder["minimum_target_twin_q"]
        threshold = float(decoder["actor_non_keep_confidence_threshold"])
        tolerance = float(decoder["maximum_actor_target_q_regret"])
        thresholds.add(threshold)
        tolerances.add(tolerance)

        actor_values = [
            probability if is_valid else float("-inf")
            for probability, is_valid in zip(probabilities, valid)
        ]
        actor_index = _argmax_first(actor_values)
        actor_confidence = probabilities[actor_index]
        expected_base_override = actor_index != 1 and actor_confidence >= threshold
        finite_q = [
            float(value) if is_valid else float("-inf")
            for value, is_valid in zip(q_values, valid)
        ]
        target_index = _argmax_first(finite_q)
        keep_tied = finite_q[1] == max(finite_q)
        if keep_tied:
            target_index = 1
        regret = finite_q[target_index] - finite_q[actor_index]
        expected_guard_passed = regret <= tolerance
        expected_override = expected_base_override and expected_guard_passed
        expected_veto = expected_base_override and not expected_guard_passed
        expected_index = actor_index if expected_override else target_index

        stored_actor = int(decoder["actor_selected_lane_index"])
        stored_target = int(decoder["target_selected_lane_index"])
        stored_selected = int(decoder["selected_lane_index"])
        stored_base = bool(decoder["base_actor_override"])
        stored_guard = bool(decoder["q_regret_guard_passed"])
        stored_veto = bool(decoder["q_regret_veto_applied"])
        stored_override = bool(decoder["actor_override"])
        source = str(decoder["selected_source"])
        if source not in source_counts:
            raise ValueError(f"unknown Q-guard selected_source {source!r}")
        source_counts[source] += 1

        exact_rule += int(
            stored_actor == actor_index
            and stored_target == target_index
            and stored_base == expected_base_override
            and stored_guard == expected_guard_passed
            and stored_veto == expected_veto
            and stored_override == expected_override
            and stored_selected == expected_index
            and bool(decoder["keep_was_exact_tied_target_maximum"]) == keep_tied
            and math.isclose(
                float(decoder["actor_target_q_regret"]),
                regret,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        exact_action += int(
            int(record["lane_command"]) == int(decoder["selected_lane"])
        )
        feasible += int(valid[stored_selected])
        base_overrides += int(stored_base)
        final_overrides += int(stored_override)
        vetoes += int(stored_veto)
        veto_valid += int(
            stored_veto
            and stored_base
            and not stored_guard
            and stored_selected == stored_target
            and source == "target_critic"
        )
        override_valid += int(
            stored_override
            and stored_base
            and stored_guard
            and stored_selected == stored_actor
            and source == "actor"
        )
        fallback_valid += int(
            not stored_override
            and stored_selected == stored_target
            and source == "target_critic"
        )

    if len(thresholds) != 1 or len(tolerances) != 1:
        raise ValueError("Q-guard parameters changed within trace")
    count = len(records)
    fallbacks = count - final_overrides
    return {
        "q_guard_fusion_decoder_records": count,
        "deterministic_lane_decoder": DECODER,
        "actor_non_keep_confidence_threshold": next(iter(thresholds)),
        "maximum_actor_target_q_regret": next(iter(tolerances)),
        "exact_q_guard_fusion_rule_match_rate": exact_rule / count,
        "exact_q_guard_fusion_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "base_actor_override_records": base_overrides,
        "base_actor_override_rate": base_overrides / count,
        "q_regret_veto_records": vetoes,
        "q_regret_veto_rate": vetoes / count,
        "q_regret_veto_valid_rate": veto_valid / vetoes if vetoes else 1.0,
        "actor_override_records": final_overrides,
        "actor_override_rate": final_overrides / count,
        "actor_override_predicate_valid_rate": (
            override_valid / final_overrides if final_overrides else 1.0
        ),
        "target_fallback_records": fallbacks,
        "target_fallback_rate": fallbacks / count,
        "target_fallback_rule_match_rate": (
            fallback_valid / fallbacks if fallbacks else 1.0
        ),
        "selected_source_counts": source_counts,
    }


def evaluate_with_action_diagnostics_v4_9_qguard(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    policy = model.policy
    if not hasattr(policy, "begin_fusion_decoder_recording"):
        raise TypeError("v4.9a evaluation requires fusion recording")
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
        decoder_records = policy.end_fusion_decoder_recording()

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(trace_rows) != len(decoder_records):
        raise ValueError(
            "Q-guard decoder/action trace length mismatch: "
            f"{len(decoder_records)} != {len(trace_rows)}"
        )
    for trace_row, decoder_row in zip(trace_rows, decoder_records):
        trace_row["q_guard_fusion_decoder"] = decoder_row
    metrics = q_guard_decoder_metrics(trace_rows)
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    diagnostics = {
        **dict(diagnostics),
        "schema_version": "topo-scene-v4.9a.action-diagnostics/v1",
        **metrics,
    }
    diagnostics["trace"] = {
        **diagnostics["trace"],
        "records": len(trace_rows),
        "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    return report, diagnostics


def q_guard_decoder_integrity_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_deployment_decoder": DECODER,
        "selected_decoder_records": int(
            diagnostics["q_guard_fusion_decoder_records"]
        ),
        "selected_decoder_exact_rule_match_rate": float(
            diagnostics["exact_q_guard_fusion_rule_match_rate"]
        ),
        "selected_decoder_exact_action_match_rate": float(
            diagnostics["exact_q_guard_fusion_action_match_rate"]
        ),
        "selected_action_mask_feasible_rate": float(
            diagnostics["selected_action_mask_feasible_rate"]
        ),
        "actor_non_keep_confidence_threshold": float(
            diagnostics["actor_non_keep_confidence_threshold"]
        ),
        "maximum_actor_target_q_regret": float(
            diagnostics["maximum_actor_target_q_regret"]
        ),
        "q_regret_veto_records": int(diagnostics["q_regret_veto_records"]),
        "q_regret_veto_valid_rate": float(
            diagnostics["q_regret_veto_valid_rate"]
        ),
    }


def q_guard_decoder_integrity_passed(diagnostics: dict[str, Any]) -> bool:
    return bool(
        diagnostics.get("q_guard_fusion_decoder_records", 0) > 0
        and diagnostics.get("deterministic_lane_decoder") == DECODER
        and diagnostics.get("actor_non_keep_confidence_threshold") == 0.90
        and diagnostics.get("maximum_actor_target_q_regret") == 0.05
        and diagnostics.get("exact_q_guard_fusion_rule_match_rate") == 1.0
        and diagnostics.get("exact_q_guard_fusion_action_match_rate") == 1.0
        and diagnostics.get("selected_action_mask_feasible_rate") == 1.0
        and diagnostics.get("q_regret_veto_valid_rate") == 1.0
        and diagnostics.get("actor_override_predicate_valid_rate") == 1.0
        and diagnostics.get("target_fallback_rule_match_rate") == 1.0
        and diagnostics.get("hybrid_exact_lateral_code_rate") == 1.0
    )


__all__ = [
    "evaluate_with_action_diagnostics_v4_9_qguard",
    "q_guard_decoder_integrity_passed",
    "q_guard_decoder_integrity_summary",
    "q_guard_decoder_metrics",
]
