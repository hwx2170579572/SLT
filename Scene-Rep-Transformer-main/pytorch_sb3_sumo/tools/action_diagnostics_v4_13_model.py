"""Exact model-only diagnostics for v4.13 gradient-isolated support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algos.sb3_torch.hybrid_policy_v4_13_model import (
    GradientIsolatedTemperedJointSupportPolicyV413,
    GradientIsolatedTemperedMixtureActorV413,
)
from algos.sb3_torch.sac_v4_13_model import (
    GradientIsolatedTemperedJointSupportSACV413,
)
from tools.action_diagnostics_v4_12_model import (
    decoder_integrity_passed_v4_12,
    decoder_integrity_summary_v4_12,
    evaluate_with_action_diagnostics_v4_12_model,
    joint_support_model_integrity_passed,
    load_model_for_deployment_v4_12,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)


def load_model_for_deployment_v4_13(
    model_class: type,
    checkpoint: Path,
    *,
    decoder: str,
    env: Any | None,
    device: str,
):
    if decoder == PARENT_DECODER:
        return load_model_for_deployment_v4_12(
            model_class,
            checkpoint,
            decoder=decoder,
            env=env,
            device=device,
        )
    if decoder != TARGET_DECODER:
        raise ValueError("v4.13 candidate permits only target_critic deployment")
    model = model_class.load(
        checkpoint,
        env=env,
        device=device,
        custom_objects={
            "policy_class": GradientIsolatedTemperedJointSupportPolicyV413
        },
    )
    if not isinstance(model, GradientIsolatedTemperedJointSupportSACV413):
        raise TypeError("loaded v4.13 deployment lost its algorithm class")
    if not isinstance(
        model.policy, GradientIsolatedTemperedJointSupportPolicyV413
    ):
        raise TypeError("loaded v4.13 deployment lost its policy class")
    if not isinstance(model.actor, GradientIsolatedTemperedMixtureActorV413):
        raise TypeError("loaded v4.13 deployment lost its actor class")
    if model.optimizer_ownership_audit()["overlap_count"] != 0:
        raise ValueError("loaded v4.13 optimizer ownership overlaps")
    return model


def evaluate_with_action_diagnostics_v4_13_model(
    model: Any,
    env: Any,
    *,
    deployment_decoder: str,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    report, diagnostics = evaluate_with_action_diagnostics_v4_12_model(
        model,
        env,
        deployment_decoder=deployment_decoder,
        episodes=episodes,
        seed=seed,
        trace_path=trace_path,
    )
    if not isinstance(model, GradientIsolatedTemperedJointSupportSACV413):
        return report, diagnostics
    actor = model.actor
    diagnostics.update(
        {
            "schema_version": "topo-scene-v4.13.action-diagnostics/v1",
            "scientific_version": (
                "v4.13_gradient_isolated_tempered_joint_support_prcr"
            ),
            "lane_support_gradient_isolated": bool(
                actor.isolate_lane_support_gradient
            ),
            "lane_support_gradient_target": (
                "deployed_lane_head_only"
                if actor.isolate_lane_support_gradient
                else "deployed_lane_head_and_actor_latent_trunk"
            ),
            "same_deployed_lane_head_for_support": True,
            "new_inference_head": False,
            "inference_score_equation_changed_from_v4_12": False,
            "scenario_conditioned_inference_rule": False,
            "ttc_or_headway_threshold": False,
            "lane_change_veto": False,
        }
    )
    return report, diagnostics


def gradient_isolated_model_integrity_passed(
    diagnostics: dict[str, Any]
) -> bool:
    return bool(
        joint_support_model_integrity_passed(diagnostics)
        and diagnostics.get("same_deployed_lane_head_for_support") is True
        and diagnostics.get("new_inference_head") is False
        and diagnostics.get("inference_score_equation_changed_from_v4_12")
        is False
        and diagnostics.get("scenario_conditioned_inference_rule") is False
        and diagnostics.get("ttc_or_headway_threshold") is False
        and diagnostics.get("lane_change_veto") is False
    )


def decoder_integrity_summary_v4_13(
    diagnostics: dict[str, Any]
) -> dict[str, Any]:
    if "lane_support_gradient_isolated" not in diagnostics:
        return decoder_integrity_summary_v4_12(diagnostics)
    return {
        **decoder_integrity_summary_v4_12(diagnostics),
        "lane_support_gradient_isolated": diagnostics[
            "lane_support_gradient_isolated"
        ],
        "lane_support_gradient_target": diagnostics[
            "lane_support_gradient_target"
        ],
        "same_deployed_lane_head_for_support": diagnostics[
            "same_deployed_lane_head_for_support"
        ],
        "gradient_isolated_model_integrity_passed": (
            gradient_isolated_model_integrity_passed(diagnostics)
        ),
    }


def decoder_integrity_passed_v4_13(diagnostics: dict[str, Any]) -> bool:
    if "lane_support_gradient_isolated" not in diagnostics:
        return decoder_integrity_passed_v4_12(diagnostics)
    return bool(
        decoder_integrity_passed_v4_12(diagnostics)
        and gradient_isolated_model_integrity_passed(diagnostics)
    )


__all__ = [
    "decoder_integrity_passed_v4_13",
    "decoder_integrity_summary_v4_13",
    "evaluate_with_action_diagnostics_v4_13_model",
    "gradient_isolated_model_integrity_passed",
    "load_model_for_deployment_v4_13",
]
