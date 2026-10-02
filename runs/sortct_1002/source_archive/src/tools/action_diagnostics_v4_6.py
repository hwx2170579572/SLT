"""Decoder-aware action diagnostics for the v4.6 selected deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algos.sb3_torch.hybrid_policy_v4_3 import TargetCriticDecisionAlignedSACPolicyV43
from algos.sb3_torch.hybrid_policy_v4_5 import ConfidentActorFusionSACPolicyV45
from tools.action_diagnostics_v4_3 import evaluate_with_action_diagnostics_v4_3
from tools.action_diagnostics_v4_5 import evaluate_with_action_diagnostics_v4_5
from tools.checkpoint_decoder_selector_v4_6 import (
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
)


def policy_class_for_decoder(decoder: str):
    if decoder == TARGET_DECODER:
        return TargetCriticDecisionAlignedSACPolicyV43
    if decoder == FUSION_DECODER:
        return ConfidentActorFusionSACPolicyV45
    if decoder == PARENT_DECODER:
        return None
    raise ValueError(f"unsupported deployment decoder {decoder!r}")


def load_model_for_deployment(
    model_class: type,
    checkpoint: Path,
    *,
    decoder: str,
    env: Any | None,
    device: str,
):
    policy_class = policy_class_for_decoder(decoder)
    custom_objects = {"policy_class": policy_class} if policy_class is not None else None
    return model_class.load(
        checkpoint,
        env=env,
        device=device,
        custom_objects=custom_objects,
    )


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("v4.6 decoder trace is empty")
    return rows


def _target_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_rule = 0
    exact_action = 0
    feasible = 0
    exact_keep_tie = 0
    keep_tie_records = 0
    for row in rows:
        decoder = row["target_critic_decoder"]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        finite_q = [
            float(value) if is_valid else float("-inf")
            for value, is_valid in zip(decoder["minimum_target_twin_q"], valid)
        ]
        expected = max(range(len(finite_q)), key=lambda index: finite_q[index])
        keep_tied = finite_q[1] == max(finite_q)
        if keep_tied:
            expected = 1
            keep_tie_records += 1
        stored = int(decoder["selected_lane_index"])
        exact_rule += int(stored == expected)
        exact_action += int(int(row["lane_command"]) == int(decoder["selected_lane"]))
        feasible += int(valid[stored])
        exact_keep_tie += int(
            bool(decoder["keep_was_exact_tied_maximum"]) == keep_tied
            and (not keep_tied or stored == 1)
        )
    count = len(rows)
    return {
        "selected_deployment_decoder": TARGET_DECODER,
        "selected_decoder_records": count,
        "selected_decoder_exact_rule_match_rate": exact_rule / count,
        "selected_decoder_exact_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "target_keep_tie_records": keep_tie_records,
        "target_keep_tie_rule_match_rate": exact_keep_tie / count,
    }


def _fusion_integrity(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_deployment_decoder": FUSION_DECODER,
        "selected_decoder_records": int(diagnostics["fusion_decoder_records"]),
        "selected_decoder_exact_rule_match_rate": float(
            diagnostics["exact_fusion_rule_match_rate"]
        ),
        "selected_decoder_exact_action_match_rate": float(
            diagnostics["exact_fusion_action_match_rate"]
        ),
        "selected_action_mask_feasible_rate": float(
            diagnostics["selected_action_mask_feasible_rate"]
        ),
    }


def evaluate_with_action_diagnostics_v4_6(
    model: Any,
    env: Any,
    *,
    deployment_decoder: str,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    """Evaluate exactly the decoder sealed in the deployment receipt."""

    if deployment_decoder == TARGET_DECODER:
        report, diagnostics = evaluate_with_action_diagnostics_v4_3(
            model, env, episodes=episodes, seed=seed, trace_path=trace_path
        )
        integrity = _target_integrity(_load_trace(trace_path))
    elif deployment_decoder in (FUSION_DECODER, PARENT_DECODER):
        report, diagnostics = evaluate_with_action_diagnostics_v4_5(
            model, env, episodes=episodes, seed=seed, trace_path=trace_path
        )
        integrity = (
            _fusion_integrity(diagnostics)
            if deployment_decoder == FUSION_DECODER
            else {
                "selected_deployment_decoder": PARENT_DECODER,
                "selected_decoder_records": int(diagnostics["decision_records"]),
                "selected_decoder_exact_rule_match_rate": None,
                "selected_decoder_exact_action_match_rate": None,
                "selected_action_mask_feasible_rate": None,
            }
        )
    else:
        raise ValueError(f"unsupported deployment decoder {deployment_decoder!r}")

    diagnostics = {
        **diagnostics,
        "schema_version": "topo-scene-v4.6.action-diagnostics/v1",
        **integrity,
    }
    return report, diagnostics


def decoder_integrity_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    decoder = str(diagnostics["selected_deployment_decoder"])
    common = {
        "selected_deployment_decoder": decoder,
        "selected_decoder_records": int(diagnostics["selected_decoder_records"]),
        "selected_decoder_exact_rule_match_rate": diagnostics[
            "selected_decoder_exact_rule_match_rate"
        ],
        "selected_decoder_exact_action_match_rate": diagnostics[
            "selected_decoder_exact_action_match_rate"
        ],
        "selected_action_mask_feasible_rate": diagnostics[
            "selected_action_mask_feasible_rate"
        ],
    }
    if decoder == TARGET_DECODER:
        common.update(
            {
                "exact_target_critic_argmax_rate": diagnostics[
                    "exact_target_critic_argmax_rate"
                ],
                "target_keep_tie_rule_match_rate": diagnostics[
                    "target_keep_tie_rule_match_rate"
                ],
            }
        )
    elif decoder == FUSION_DECODER:
        common.update(
            {
                "actor_non_keep_confidence_threshold": diagnostics[
                    "actor_non_keep_confidence_threshold"
                ],
                "actor_override_predicate_valid_rate": diagnostics[
                    "actor_override_predicate_valid_rate"
                ],
                "target_fallback_rule_match_rate": diagnostics[
                    "target_fallback_rule_match_rate"
                ],
            }
        )
    return common


def decoder_integrity_passed(diagnostics: dict[str, Any]) -> bool:
    decoder = str(diagnostics["selected_deployment_decoder"])
    if decoder == PARENT_DECODER:
        return True
    common = (
        diagnostics.get("selected_decoder_records", 0) > 0
        and diagnostics.get("selected_decoder_exact_rule_match_rate") == 1.0
        and diagnostics.get("selected_decoder_exact_action_match_rate") == 1.0
        and diagnostics.get("selected_action_mask_feasible_rate") == 1.0
        and diagnostics.get("hybrid_exact_lateral_code_rate") == 1.0
    )
    if decoder == TARGET_DECODER:
        return bool(
            common
            and diagnostics.get("exact_target_critic_argmax_rate") == 1.0
            and diagnostics.get("target_keep_tie_rule_match_rate") == 1.0
        )
    if decoder == FUSION_DECODER:
        return bool(
            common
            and diagnostics.get("actor_non_keep_confidence_threshold") == 0.90
            and diagnostics.get("actor_override_predicate_valid_rate") == 1.0
            and diagnostics.get("target_fallback_rule_match_rate") == 1.0
        )
    return False


__all__ = [
    "decoder_integrity_passed",
    "decoder_integrity_summary",
    "evaluate_with_action_diagnostics_v4_6",
    "load_model_for_deployment",
    "policy_class_for_decoder",
]
