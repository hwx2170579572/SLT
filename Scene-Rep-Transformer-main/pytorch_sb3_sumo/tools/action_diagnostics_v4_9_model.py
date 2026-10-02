"""Deployment loading and model-integrity evidence for learned v4.9 risk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algos.sb3_torch.hybrid_policy_v4_9_model import (
    CollisionConstrainedFusionSACPolicyV49,
    CollisionConstrainedTargetCriticSACPolicyV49,
)
from algos.sb3_torch.sac_v4_9_model import CollisionConstrainedFusionSACV49
from tools.action_diagnostics_v4_6 import (
    decoder_integrity_passed as parent_decoder_integrity_passed,
    decoder_integrity_summary as parent_decoder_integrity_summary,
    evaluate_with_action_diagnostics_v4_6,
    load_model_for_deployment as load_parent_model_for_deployment,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
)


def policy_class_for_decoder_v4_9(decoder: str):
    if decoder == TARGET_DECODER:
        return CollisionConstrainedTargetCriticSACPolicyV49
    if decoder == FUSION_DECODER:
        return CollisionConstrainedFusionSACPolicyV49
    raise ValueError(f"unsupported v4.9 deployment decoder {decoder!r}")


def load_model_for_deployment_v4_9(
    model_class: type,
    checkpoint: Path,
    *,
    decoder: str,
    env: Any | None,
    device: str,
):
    if decoder == PARENT_DECODER:
        return load_parent_model_for_deployment(
            model_class,
            checkpoint,
            decoder=decoder,
            env=env,
            device=device,
        )
    policy_class = policy_class_for_decoder_v4_9(decoder)
    model = model_class.load(
        checkpoint,
        env=env,
        device=device,
        custom_objects={"policy_class": policy_class},
    )
    if not isinstance(model, CollisionConstrainedFusionSACV49):
        raise TypeError("loaded v4.9 deployment lost its algorithm class")
    if not hasattr(model.policy, "collision_critic") or not hasattr(
        model.policy, "collision_critic_target"
    ):
        raise TypeError("loaded v4.9 deployment lost learned collision critics")
    if float(model.collision_risk_coef) != 1.0:
        raise ValueError("loaded v4.9 collision risk coefficient drifted")
    return model


def evaluate_with_action_diagnostics_v4_9_model(
    model: Any,
    env: Any,
    *,
    deployment_decoder: str,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    report, diagnostics = evaluate_with_action_diagnostics_v4_6(
        model,
        env,
        deployment_decoder=deployment_decoder,
        episodes=episodes,
        seed=seed,
        trace_path=trace_path,
    )
    online_present = bool(hasattr(model.policy, "collision_critic"))
    target_present = bool(hasattr(model.policy, "collision_critic_target"))
    legacy_integrity = {
        key: diagnostics.get(key)
        for key in (
            "selected_decoder_exact_rule_match_rate",
            "exact_target_critic_argmax_rate",
            "target_keep_tie_rule_match_rate",
            "exact_fusion_rule_match_rate",
            "target_fallback_rule_match_rate",
        )
        if key in diagnostics
    }
    if online_present:
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if deployment_decoder == TARGET_DECODER:
            model_integrity = _risk_adjusted_target_integrity(rows)
        elif deployment_decoder == FUSION_DECODER:
            model_integrity = _risk_adjusted_fusion_integrity(rows)
        else:
            raise ValueError(
                "learned v4.9 model cannot use the parent-control decoder"
            )
        diagnostics.update(model_integrity)
    diagnostics = {
        **diagnostics,
        "schema_version": "topo-scene-v4.9.action-diagnostics/v1",
        "learned_collision_critic_present": online_present,
        "learned_collision_critic_target_present": target_present,
        "collision_risk_coef": (
            float(model.collision_risk_coef) if online_present else None
        ),
        "inference_safety_rule_added": False,
        "model_decision_equation": (
            "minimum_reward_q_minus_lambda_maximum_collision_value"
            if online_present
            else None
        ),
        "legacy_reward_only_decoder_counterfactual": legacy_integrity,
    }
    return report, diagnostics


def _risk_components(
    decoder: dict[str, Any],
) -> tuple[list[bool], list[float], list[float], list[float], float]:
    valid = [bool(value) for value in decoder["valid_lane_actions"]]
    reward = [
        float(value) if is_valid else float("-inf")
        for value, is_valid in zip(decoder["minimum_target_twin_q"], valid)
    ]
    collision = [
        float(value) if is_valid else 0.0
        for value, is_valid in zip(
            decoder["maximum_target_twin_collision_value"], valid
        )
    ]
    score = [
        float(value) if is_valid else float("-inf")
        for value, is_valid in zip(decoder["risk_adjusted_target_score"], valid)
    ]
    coefficient = float(decoder["collision_risk_coef"])
    return valid, reward, collision, score, coefficient


def _risk_argmax(
    valid: list[bool], score: list[float]
) -> tuple[int, bool]:
    if len(valid) != 3 or not any(valid):
        raise ValueError("v4.9 risk decoder requires three lanes and one feasible action")
    selected = max(range(3), key=lambda index: score[index])
    keep_tied = bool(valid[1] and score[1] == max(score))
    return (1 if keep_tied else selected), keep_tied


def _equation_matches(
    valid: list[bool],
    reward: list[float],
    collision: list[float],
    score: list[float],
    coefficient: float,
) -> bool:
    return all(
        (not is_valid)
        or (
            0.0 <= collision[index] <= 1.0
            and abs(score[index] - (reward[index] - coefficient * collision[index]))
            <= 1e-5
        )
        for index, is_valid in enumerate(valid)
    )


def _risk_adjusted_target_integrity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("v4.9 target decoder trace is empty")
    exact_model = exact_action = feasible = exact_tie = equation = 0
    tie_records = 0
    for row in rows:
        decoder = row["target_critic_decoder"]
        valid, reward, collision, score, coefficient = _risk_components(decoder)
        expected, keep_tied = _risk_argmax(valid, score)
        stored = int(decoder["selected_lane_index"])
        exact_model += int(stored == expected)
        exact_action += int(int(row["lane_command"]) == int(decoder["selected_lane"]))
        feasible += int(valid[stored])
        equation += int(
            _equation_matches(valid, reward, collision, score, coefficient)
        )
        tie_records += int(keep_tied)
        exact_tie += int(
            bool(decoder["keep_was_exact_tied_maximum"]) == keep_tied
            and (not keep_tied or stored == 1)
        )
    count = len(rows)
    return {
        "selected_deployment_decoder": TARGET_DECODER,
        "selected_decoder_records": count,
        "selected_decoder_exact_rule_match_rate": exact_model / count,
        "selected_decoder_exact_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "exact_risk_adjusted_model_argmax_rate": exact_model / count,
        "risk_adjusted_score_equation_match_rate": equation / count,
        "risk_adjusted_keep_tie_records": tie_records,
        "risk_adjusted_keep_tie_match_rate": exact_tie / count,
        "deterministic_lane_decoder": (
            "argmax_feasible_min_reward_q_minus_max_collision_value"
        ),
    }


def _risk_adjusted_fusion_integrity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("v4.9 fusion decoder trace is empty")
    exact_model = exact_action = feasible = equation = 0
    overrides = override_valid = fallbacks = fallback_valid = 0
    thresholds: set[float] = set()
    for row in rows:
        decoder = row["fusion_decoder"]
        valid, reward, collision, score, coefficient = _risk_components(decoder)
        target_index, _ = _risk_argmax(valid, score)
        probabilities = [float(value) for value in decoder["lane_probabilities"]]
        actor_index = max(range(3), key=lambda index: probabilities[index])
        threshold = float(decoder["actor_non_keep_confidence_threshold"])
        thresholds.add(threshold)
        expected_override = actor_index != 1 and probabilities[actor_index] >= threshold
        expected = actor_index if expected_override else target_index
        stored = int(decoder["selected_lane_index"])
        exact_model += int(
            int(decoder["actor_selected_lane_index"]) == actor_index
            and int(decoder["target_selected_lane_index"]) == target_index
            and bool(decoder["actor_override"]) == expected_override
            and stored == expected
        )
        exact_action += int(int(row["lane_command"]) == int(decoder["selected_lane"]))
        feasible += int(valid[stored])
        equation += int(
            _equation_matches(valid, reward, collision, score, coefficient)
        )
        if bool(decoder["actor_override"]):
            overrides += 1
            override_valid += int(
                actor_index != 1
                and probabilities[actor_index] >= threshold
                and valid[actor_index]
                and stored == actor_index
            )
        else:
            fallbacks += 1
            fallback_valid += int(stored == target_index)
    if len(thresholds) != 1:
        raise ValueError(f"v4.9 fusion threshold drifted: {thresholds}")
    count = len(rows)
    return {
        "selected_deployment_decoder": FUSION_DECODER,
        "selected_decoder_records": count,
        "selected_decoder_exact_rule_match_rate": exact_model / count,
        "selected_decoder_exact_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "exact_risk_adjusted_model_fusion_rate": exact_model / count,
        "risk_adjusted_score_equation_match_rate": equation / count,
        "actor_non_keep_confidence_threshold": next(iter(thresholds)),
        "actor_override_records": overrides,
        "actor_override_predicate_valid_rate": (
            override_valid / overrides if overrides else 1.0
        ),
        "target_fallback_records": fallbacks,
        "risk_adjusted_target_fallback_match_rate": (
            fallback_valid / fallbacks if fallbacks else 1.0
        ),
        "deterministic_lane_decoder": (
            "actor_confident_non_keep_else_learned_risk_adjusted_critic"
        ),
    }


def learned_collision_model_integrity_passed(
    diagnostics: dict[str, Any]
) -> bool:
    return bool(
        diagnostics.get("learned_collision_critic_present") is True
        and diagnostics.get("learned_collision_critic_target_present") is True
        and diagnostics.get("collision_risk_coef") == 1.0
        and diagnostics.get("inference_safety_rule_added") is False
    )


def decoder_integrity_summary_v4_9(
    diagnostics: dict[str, Any]
) -> dict[str, Any]:
    if not diagnostics.get("learned_collision_critic_present"):
        return parent_decoder_integrity_summary(diagnostics)
    decoder = str(diagnostics["selected_deployment_decoder"])
    summary = {
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
        "risk_adjusted_score_equation_match_rate": diagnostics[
            "risk_adjusted_score_equation_match_rate"
        ],
        "learned_collision_model_integrity_passed": (
            learned_collision_model_integrity_passed(diagnostics)
        ),
    }
    if decoder == TARGET_DECODER:
        summary.update(
            {
                "exact_risk_adjusted_model_argmax_rate": diagnostics[
                    "exact_risk_adjusted_model_argmax_rate"
                ],
                "risk_adjusted_keep_tie_match_rate": diagnostics[
                    "risk_adjusted_keep_tie_match_rate"
                ],
            }
        )
    elif decoder == FUSION_DECODER:
        summary.update(
            {
                "actor_non_keep_confidence_threshold": diagnostics[
                    "actor_non_keep_confidence_threshold"
                ],
                "actor_override_predicate_valid_rate": diagnostics[
                    "actor_override_predicate_valid_rate"
                ],
                "risk_adjusted_target_fallback_match_rate": diagnostics[
                    "risk_adjusted_target_fallback_match_rate"
                ],
            }
        )
    return summary


def decoder_integrity_passed_v4_9(diagnostics: dict[str, Any]) -> bool:
    if not diagnostics.get("learned_collision_critic_present"):
        return parent_decoder_integrity_passed(diagnostics)
    common = bool(
        learned_collision_model_integrity_passed(diagnostics)
        and diagnostics.get("selected_decoder_records", 0) > 0
        and diagnostics.get("selected_decoder_exact_rule_match_rate") == 1.0
        and diagnostics.get("selected_decoder_exact_action_match_rate") == 1.0
        and diagnostics.get("selected_action_mask_feasible_rate") == 1.0
        and diagnostics.get("hybrid_exact_lateral_code_rate") == 1.0
        and diagnostics.get("risk_adjusted_score_equation_match_rate") == 1.0
    )
    decoder = str(diagnostics.get("selected_deployment_decoder"))
    if decoder == TARGET_DECODER:
        return bool(
            common
            and diagnostics.get("exact_risk_adjusted_model_argmax_rate") == 1.0
            and diagnostics.get("risk_adjusted_keep_tie_match_rate") == 1.0
        )
    if decoder == FUSION_DECODER:
        return bool(
            common
            and diagnostics.get("actor_non_keep_confidence_threshold") == 0.90
            and diagnostics.get("actor_override_predicate_valid_rate") == 1.0
            and diagnostics.get("risk_adjusted_target_fallback_match_rate") == 1.0
        )
    return False


__all__ = [
    "evaluate_with_action_diagnostics_v4_9_model",
    "decoder_integrity_passed_v4_9",
    "decoder_integrity_summary_v4_9",
    "learned_collision_model_integrity_passed",
    "load_model_for_deployment_v4_9",
    "policy_class_for_decoder_v4_9",
]
