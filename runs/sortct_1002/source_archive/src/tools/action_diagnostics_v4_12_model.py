"""Exact joint-support proposal diagnostics for the v4.12 model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from algos.sb3_torch.hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
    JointReplaySupportedMixtureActorV412,
)
from algos.sb3_torch.sac_v4_12_model import (
    AugmentedJointSupportPRCRSACV412,
)
from tools.action_diagnostics_v4_11_model import (
    decoder_integrity_passed_v4_11,
    decoder_integrity_summary_v4_11,
    evaluate_with_action_diagnostics_v4_11_model,
    load_model_for_deployment_v4_11,
)
from tools.action_diagnostics_v4_2 import evaluate_with_action_diagnostics_v4_2
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)


def load_model_for_deployment_v4_12(
    model_class: type,
    checkpoint: Path,
    *,
    decoder: str,
    env: Any | None,
    device: str,
):
    if decoder == PARENT_DECODER:
        return load_model_for_deployment_v4_11(
            model_class,
            checkpoint,
            decoder=decoder,
            env=env,
            device=device,
        )
    if decoder != TARGET_DECODER:
        raise ValueError("v4.12 candidate permits only the learned target decoder")
    model = model_class.load(
        checkpoint,
        env=env,
        device=device,
        custom_objects={
            "policy_class": AugmentedJointSupportPRCRPolicyV412
        },
    )
    if not isinstance(model, AugmentedJointSupportPRCRSACV412):
        raise TypeError("loaded v4.12 deployment lost its algorithm class")
    if not isinstance(model.policy, AugmentedJointSupportPRCRPolicyV412):
        raise TypeError("loaded v4.12 deployment lost its policy class")
    if not isinstance(model.actor, JointReplaySupportedMixtureActorV412):
        raise TypeError("loaded v4.12 deployment lost joint action support")
    if model.optimizer_ownership_audit()["overlap_count"] != 0:
        raise ValueError("loaded v4.12 optimizer ownership overlaps")
    return model


def _candidate_argmax(decoder: dict[str, Any]) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []
    scores = decoder["supported_risk_adjusted_score"]
    components = len(decoder["component_probabilities"][0])
    for lane_index, valid in enumerate(decoder["valid_lane_actions"]):
        if not bool(valid):
            continue
        for component_index in range(components):
            candidates.append(
                (
                    float(scores[lane_index][component_index]),
                    lane_index,
                    component_index,
                )
            )
    if not candidates:
        raise ValueError("v4.12 decoder has no feasible learned proposal")
    _, lane, component = max(candidates, key=lambda item: item[0])
    return lane, component


def _proposal_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("v4.12 target decoder trace is empty")
    exact_model = exact_action = feasible = equation = exact_speed = 0
    bounded_collision = no_rewrite = learned_source = no_semantic_tie = 0
    no_threshold = selected_boundary = 0
    component_counts: set[int] = set()
    for row in rows:
        decoder = row["target_critic_decoder"]
        valid = [bool(value) for value in decoder["valid_lane_actions"]]
        component_probabilities = decoder["component_probabilities"]
        lane_probabilities = [
            float(value) for value in decoder["lane_probabilities"]
        ]
        components = len(component_probabilities[0])
        component_counts.add(components)
        expected_lane, expected_component = _candidate_argmax(decoder)
        stored_lane = int(decoder["selected_lane_index"])
        stored_component = int(decoder["selected_component_index"])
        exact_model += int(
            stored_lane == expected_lane
            and stored_component == expected_component
        )
        exact_action += int(
            int(row["lane_command"]) == int(decoder["selected_lane"])
        )
        feasible += int(valid[stored_lane])
        proposed_speed = float(
            decoder["learned_speed_proposals_normalized"][stored_lane][
                stored_component
            ]
        )
        exact_speed += int(
            abs(float(row["action_longitudinal"]) - proposed_speed) <= 1e-5
        )
        selected_boundary += int(abs(proposed_speed) >= 0.95)
        no_semantic_tie += int(
            decoder.get("selection_operator")
            == "torch_argmax_flattened_feasible_model_scores"
            and decoder.get("semantic_tie_override_used") is False
        )
        no_threshold += int(
            decoder.get("actor_confidence_threshold_present") is False
        )
        collision_coef = float(decoder["collision_risk_coef"])
        uncertainty_coef = float(decoder["twin_uncertainty_coef"])
        component_coef = float(decoder["component_prior_coef"])
        lane_coef = float(decoder["lane_prior_coef"])
        row_equation = abs(sum(lane_probabilities) - 1.0) <= 1e-5
        row_bounded = True
        for lane_index, is_valid in enumerate(valid):
            if not is_valid:
                continue
            probability_sum = sum(
                float(value)
                for value in component_probabilities[lane_index]
            )
            row_equation &= abs(probability_sum - 1.0) <= 1e-5
            for component_index in range(components):
                reward = float(
                    decoder["minimum_target_twin_q"][lane_index][
                        component_index
                    ]
                )
                collision = float(
                    decoder["maximum_target_twin_collision_value"][
                        lane_index
                    ][component_index]
                )
                reward_gap = float(
                    decoder["reward_twin_disagreement"][lane_index][
                        component_index
                    ]
                )
                collision_gap = float(
                    decoder["collision_twin_disagreement"][lane_index][
                        component_index
                    ]
                )
                uncertainty = float(
                    decoder["learned_uncertainty"][lane_index][component_index]
                )
                component_probability = float(
                    component_probabilities[lane_index][component_index]
                )
                score = float(
                    decoder["supported_risk_adjusted_score"][lane_index][
                        component_index
                    ]
                )
                expected = (
                    reward
                    - collision_coef * collision
                    - uncertainty_coef * (reward_gap + collision_gap)
                    + component_coef
                    * math.log(max(component_probability, 1e-45))
                    + lane_coef
                    * math.log(max(lane_probabilities[lane_index], 1e-45))
                )
                row_equation &= (
                    abs(uncertainty - (reward_gap + collision_gap)) <= 1e-5
                    and abs(score - expected) <= 1e-5
                )
                row_bounded &= 0.0 <= collision <= 1.0
        equation += int(row_equation)
        bounded_collision += int(row_bounded)
        no_rewrite += int(
            decoder.get("fixed_speed_grid_used") is False
            and decoder.get("action_rewritten") is False
        )
        learned_source += int(
            decoder.get("candidate_source")
            == "learned_actor_component_means"
        )
    count = len(rows)
    if len(component_counts) != 1:
        raise ValueError(f"v4.12 component count drifted: {component_counts}")
    return {
        "selected_deployment_decoder": TARGET_DECODER,
        "selected_decoder_records": count,
        "selected_decoder_exact_model_match_rate": exact_model / count,
        "selected_decoder_exact_action_match_rate": exact_action / count,
        "selected_action_mask_feasible_rate": feasible / count,
        "exact_joint_support_model_argmax_rate": exact_model / count,
        "joint_support_score_equation_match_rate": equation / count,
        "selected_speed_exact_proposal_match_rate": exact_speed / count,
        "collision_value_bounded_rate": bounded_collision / count,
        "learned_proposal_source_rate": learned_source / count,
        "no_action_rewrite_rate": no_rewrite / count,
        "no_semantic_tie_override_rate": no_semantic_tie / count,
        "no_actor_confidence_threshold_rate": no_threshold / count,
        "learned_speed_components_per_lane": next(iter(component_counts)),
        "selected_proposal_boundary_rate": selected_boundary / count,
        "deterministic_lane_decoder": (
            "argmax_feasible_learned_joint_support_risk_uncertainty_score"
        ),
    }


def evaluate_with_action_diagnostics_v4_12_model(
    model: Any,
    env: Any,
    *,
    deployment_decoder: str,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    if not isinstance(model, AugmentedJointSupportPRCRSACV412):
        return evaluate_with_action_diagnostics_v4_11_model(
            model,
            env,
            deployment_decoder=deployment_decoder,
            episodes=episodes,
            seed=seed,
            trace_path=trace_path,
        )
    if deployment_decoder != TARGET_DECODER:
        raise ValueError("v4.12 candidate permits only target_critic deployment")
    policy = model.policy
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
        decoder_records = policy.end_target_decoder_recording()
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(decoder_records):
        raise ValueError("v4.12 decoder/action trace length mismatch")
    for row, decoder in zip(rows, decoder_records):
        row["target_critic_decoder"] = decoder
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    diagnostics = {
        **diagnostics,
        "schema_version": "topo-scene-v4.12.action-diagnostics/v1",
        **_proposal_integrity(rows),
        "trace": {
            **diagnostics["trace"],
            "records": len(rows),
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "shared_risk_encoder_present": True,
        "joint_replay_action_support_present": True,
        "optimizer_ownership": model.optimizer_ownership_audit(),
        "collision_risk_coef": float(model.collision_risk_coef),
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": float(
            model.collision_pairwise_rank_coef
        ),
        "collision_pairwise_hard_threshold": False,
        "collision_pairwise_fixed_margin": False,
        "collision_temporal_consistency_coef": float(
            model.collision_temporal_consistency_coef
        ),
        "replay_support_coef": float(model.replay_support_coef),
        "lane_support_scale": float(model.actor.lane_support_scale),
        "lane_prior_coef": float(policy.lane_prior_coef),
        "component_prior_coef": float(policy.component_prior_coef),
        "effective_random_rotation_augmentation": bool(
            model.critic.features_extractor.random_augmentation
        ),
        "fixed_speed_grid_at_inference": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "model_decision_equation": (
            "min_reward_q_minus_max_collision_value_minus_twin_disagreement_"
            "plus_log_component_probability_plus_log_lane_probability"
        ),
    }
    return report, diagnostics


def joint_support_model_integrity_passed(
    diagnostics: dict[str, Any]
) -> bool:
    ownership = diagnostics.get("optimizer_ownership", {})
    structure = ownership.get("policy_structure", {})
    return bool(
        diagnostics.get("learned_collision_critic_present") is True
        and diagnostics.get("learned_collision_critic_target_present") is True
        and diagnostics.get("shared_risk_encoder_present") is True
        and diagnostics.get("joint_replay_action_support_present") is True
        and ownership.get("overlap_count") == 0
        and structure.get("online_encoder_shared_by_identity") is True
        and structure.get("target_encoder_shared_by_identity") is True
        and diagnostics.get("fixed_speed_grid_at_inference") is False
        and diagnostics.get("inference_safety_rule_added") is False
        and diagnostics.get("external_kinematic_projection") is False
        and diagnostics.get("kinematic_safety_projection") is False
        and diagnostics.get("traffic_risk_in_lane_mask") is False
        and diagnostics.get("actor_confidence_gate") is False
        and diagnostics.get("action_postprocessing_override") is False
    )


def decoder_integrity_summary_v4_12(
    diagnostics: dict[str, Any]
) -> dict[str, Any]:
    if not diagnostics.get("joint_replay_action_support_present"):
        return decoder_integrity_summary_v4_11(diagnostics)
    keys = (
        "selected_deployment_decoder",
        "selected_decoder_records",
        "selected_decoder_exact_model_match_rate",
        "selected_decoder_exact_action_match_rate",
        "selected_action_mask_feasible_rate",
        "exact_joint_support_model_argmax_rate",
        "joint_support_score_equation_match_rate",
        "selected_speed_exact_proposal_match_rate",
        "collision_value_bounded_rate",
        "learned_proposal_source_rate",
        "no_action_rewrite_rate",
        "no_semantic_tie_override_rate",
        "no_actor_confidence_threshold_rate",
        "learned_speed_components_per_lane",
        "selected_proposal_boundary_rate",
    )
    return {
        **{key: diagnostics[key] for key in keys},
        "joint_support_model_integrity_passed": (
            joint_support_model_integrity_passed(diagnostics)
        ),
    }


def decoder_integrity_passed_v4_12(diagnostics: dict[str, Any]) -> bool:
    if not diagnostics.get("joint_replay_action_support_present"):
        return decoder_integrity_passed_v4_11(diagnostics)
    return bool(
        joint_support_model_integrity_passed(diagnostics)
        and diagnostics.get("selected_deployment_decoder") == TARGET_DECODER
        and diagnostics.get("selected_decoder_records", 0) > 0
        and diagnostics.get("selected_decoder_exact_model_match_rate") == 1.0
        and diagnostics.get("selected_decoder_exact_action_match_rate") == 1.0
        and diagnostics.get("selected_action_mask_feasible_rate") == 1.0
        and diagnostics.get("hybrid_exact_lateral_code_rate") == 1.0
        and diagnostics.get("exact_joint_support_model_argmax_rate") == 1.0
        and diagnostics.get("joint_support_score_equation_match_rate") == 1.0
        and diagnostics.get("selected_speed_exact_proposal_match_rate") == 1.0
        and diagnostics.get("collision_value_bounded_rate") == 1.0
        and diagnostics.get("learned_proposal_source_rate") == 1.0
        and diagnostics.get("no_action_rewrite_rate") == 1.0
        and diagnostics.get("no_semantic_tie_override_rate") == 1.0
        and diagnostics.get("no_actor_confidence_threshold_rate") == 1.0
    )


__all__ = [
    "decoder_integrity_passed_v4_12",
    "decoder_integrity_summary_v4_12",
    "evaluate_with_action_diagnostics_v4_12_model",
    "joint_support_model_integrity_passed",
    "load_model_for_deployment_v4_12",
]

