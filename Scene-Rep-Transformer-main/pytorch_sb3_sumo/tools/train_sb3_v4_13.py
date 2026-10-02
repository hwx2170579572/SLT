"""Train v4.13 gradient-isolated tempered support models."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_13 import (
    TEMPORAL_GRAPH,
    V413_ALGORITHMS,
    V413_CANDIDATES,
    V413_FORMAL_ALGORITHMS,
    V413_FULL,
    V413_IMPLEMENTATION_IDS,
    V413_ISOLATED_UNTEMPERED,
    V413_NO_CROSS_AUGMENTATION,
    V413_UNISOLATED_TEMPERED,
    make_model_v4_13,
    source_action_repeat_v4_13,
)
from tools import train_sb3_v4_12 as parent
from tools.action_diagnostics_v4_13_model import (
    decoder_integrity_passed_v4_13,
    decoder_integrity_summary_v4_13,
    evaluate_with_action_diagnostics_v4_13_model,
    load_model_for_deployment_v4_13,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE
from tools.checkpoint_selector_v4_4 import sha256


_PARENT_WRITE = parent._write_json_v4_12
_PARENT_WRITE_PLAIN = parent._write_json_plain_v4_12
_PARENT_METHOD_HYPERPARAMETERS = parent._method_hyperparameters_v4_12


def _version_payload(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        output = {key: _version_payload(item) for key, item in output.items()}
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.12."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.12.", "topo-scene-v4.13.", 1
            )
    elif isinstance(output, list):
        output = [_version_payload(item) for item in output]
    return output


def _write_direct(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _active_model() -> Any:
    return parent._active_model()


def _rewrite_v4_13(path: Path) -> None:
    if not path.is_file():
        return
    value = _version_payload(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        _write_direct(path, value)
        return
    algorithm = value.get("algorithm")
    if path.name == "arguments.json":
        algorithm = value.get("requested_raw_steps", {}).get("algo")
        candidate = algorithm in V413_CANDIDATES
        fidelity = dict(value.get("implementation_fidelity", {}))
        fidelity.pop("v4_12_isolated_files", None)
        fidelity.update(
            {
                "v4_13_isolated_files": True,
                "single_change": (
                    "gradient_isolated_tempered_joint_support_model"
                    if candidate
                    else "none_frozen_control"
                ),
                "network_changed": candidate,
                "same_deployed_lane_head_for_support": candidate,
                "new_inference_head": False,
                "inference_score_equation_changed_from_v4_12": False,
                "inference_safety_rule_added": False,
                "external_kinematic_projection": False,
                "kinematic_safety_projection": False,
                "traffic_risk_in_lane_mask": False,
                "ttc_or_headway_threshold": False,
                "lane_change_veto": False,
                "action_postprocessing_override": False,
                "scenario_conditioned_inference_rule": False,
            }
        )
        value["implementation_fidelity"] = fidelity
    elif path.name == "method_metadata.json":
        for key in (
            "external_kinematic_projection",
            "kinematic_safety_projection",
            "traffic_risk_in_lane_mask",
            "ttc_or_headway_threshold",
            "lane_change_veto",
            "action_postprocessing_override",
            "inference_safety_rule_added",
            "scenario_conditioned_inference_rule",
        ):
            value[key] = False
        if algorithm in V413_CANDIDATES:
            value.update(
                {
                    "deployment_protocol_version": "v4.13_target_only",
                    "deployment_decoder_candidates": [TARGET_DECODER],
                    "checkpoint_decoder_candidate_pairs": 2,
                    "selector_mode": SELECTOR_MODE,
                    "fusion_candidate_present": False,
                    "actor_confidence_threshold_present": False,
                }
            )
    elif path.name == "training_diagnostics.json" and algorithm in V413_CANDIDATES:
        model = _active_model()
        value.update(
            {
                "scientific_version": (
                    "v4.13_gradient_isolated_tempered_joint_support_prcr"
                ),
                "replay_support_objective": (
                    "gradient_isolated_masked_lane_categorical_nll_plus_"
                    "conditional_speed_mixture_nll"
                ),
                "lane_support_gradient_isolated": bool(
                    model.actor.isolate_lane_support_gradient
                ),
                "lane_support_gradient_target": (
                    "deployed_lane_head_only"
                    if model.actor.isolate_lane_support_gradient
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
    elif path.name == "paper_evaluation_detailed.json":
        value["formal_test_accessed"] = value.get("evaluation_split") == "test"
        if algorithm in V413_CANDIDATES:
            model = _active_model()
            value.update(
                {
                    "scientific_version": (
                        "v4.13_gradient_isolated_tempered_joint_support_prcr"
                    ),
                    "lane_support_gradient_isolated": bool(
                        model.actor.isolate_lane_support_gradient
                    ),
                    "lane_support_gradient_target": (
                        "deployed_lane_head_only"
                        if model.actor.isolate_lane_support_gradient
                        else "deployed_lane_head_and_actor_latent_trunk"
                    ),
                    "same_deployed_lane_head_for_support": True,
                    "new_inference_head": False,
                    "inference_score_equation_changed_from_v4_12": False,
                }
            )
    elif path.name == "action_diagnostics.json":
        for key in (
            "external_kinematic_projection",
            "kinematic_safety_projection",
            "traffic_risk_in_lane_mask",
            "ttc_or_headway_threshold",
            "lane_change_veto",
            "action_postprocessing_override",
            "inference_safety_rule_added",
            "scenario_conditioned_inference_rule",
        ):
            value[key] = False
    if path.name == "receipt.json" and value.get(
        "selected_deployment_decoder"
    ) == TARGET_DECODER:
        value.update(
            {
                "schema_version": (
                    "topo-scene-v4.13.target-only-tie-replicated-selector/v1"
                ),
                "deployment_protocol_version": "v4.13_target_only",
                "selector_mode": SELECTOR_MODE,
                "candidate_count": 2,
                "deployment_decoder_candidates": [TARGET_DECODER],
                "fusion_candidate_present": False,
                "actor_confidence_threshold_present": False,
                "external_kinematic_projection": False,
                "kinematic_safety_projection": False,
                "traffic_risk_in_lane_mask": False,
                "action_postprocessing_override": False,
            }
        )
    _write_direct(path, value)


def _write_json_v4_13(path: Path, payload: Any) -> None:
    _PARENT_WRITE(path, payload)
    _rewrite_v4_13(path)
    if path.name == "training_diagnostics.json":
        _rewrite_v4_13(path.parent / "return_estimator_diagnostics.json")
        _rewrite_v4_13(path.parent / "collision_label_diagnostics.json")


def _write_json_plain_v4_13(path: Path, payload: Any) -> None:
    _PARENT_WRITE_PLAIN(path, payload)
    _rewrite_v4_13(path)


def _variant(algorithm: str) -> tuple[bool, float, bool, str]:
    values = {
        V413_FULL: (True, 0.25, True, "primary_gradient_isolated_tempered"),
        V413_UNISOLATED_TEMPERED: (
            False,
            0.25,
            True,
            "ablation_unisolated_tempered",
        ),
        V413_ISOLATED_UNTEMPERED: (
            True,
            1.0,
            True,
            "ablation_isolated_untempered",
        ),
        V413_NO_CROSS_AUGMENTATION: (
            True,
            0.25,
            False,
            "ablation_no_cross_augmentation",
        ),
    }
    if algorithm not in values:
        raise ValueError(f"unsupported v4.13 algorithm {algorithm!r}")
    return values[algorithm]


def _method_hyperparameters_v4_13(algorithm: str) -> dict[str, Any]:
    if algorithm == TEMPORAL_GRAPH:
        value = _PARENT_METHOD_HYPERPARAMETERS(TEMPORAL_GRAPH)
        value.update(
            {
                "single_change": "none_temporal_graph_control",
                "network_changed": False,
                "inference_safety_rule_added": False,
                "external_kinematic_projection": False,
                "kinematic_safety_projection": False,
                "traffic_risk_in_lane_mask": False,
                "action_postprocessing_override": False,
            }
        )
        return value
    isolated, lane_scale, cross_augmentation, role = _variant(algorithm)
    return {
        "reward_unchanged": True,
        "single_change": "gradient_isolated_tempered_joint_support_model",
        "v4_13_role": role,
        "network_changed": True,
        "shared_online_scene_encoder": True,
        "actor_encoder_detach": True,
        "optimizer_parameter_sets_must_be_disjoint": True,
        "encoder_optimizer_owner_count": 1,
        "learned_speed_components_per_lane": 3,
        "fixed_speed_grid_at_inference": False,
        "replay_support_objective": (
            "gradient_isolated_masked_lane_categorical_nll_plus_"
            "conditional_speed_mixture_nll"
        ),
        "replay_support_coef": 0.05,
        "lane_support_scale": lane_scale,
        "lane_support_gradient_isolated": isolated,
        "lane_support_gradient_target": (
            "deployed_lane_head_only"
            if isolated
            else "deployed_lane_head_and_actor_latent_trunk"
        ),
        "same_deployed_lane_head_for_support": True,
        "lane_prior_coef": 0.05,
        "component_prior_coef": 0.05,
        "cross_rotation_augmentation": cross_augmentation,
        "collision_risk_coef": 1.0,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": 0.25,
        "collision_temporal_consistency_coef": 0.05,
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "inference_score_equation_changed_from_v4_12": False,
        "new_inference_head": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "scenario_conditioned_inference_rule": False,
        "n_step": 16,
        "bootstrap_discount": "gamma_power_actual_horizon",
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "deployment_decoder_candidates": [TARGET_DECODER],
        "checkpoint_decoder_candidate_pairs": 2,
        "checkpoint_calibration_episodes": 12,
        "checkpoint_calibration_partition": "train",
        "selector_mode": SELECTOR_MODE,
        "validation_used_for_checkpoint_selection": False,
        "slot_balance_coef": 0.01,
        "lane_entropy_scale": 0.0,
        "speed_target_entropy": -1.0,
    }


def _validate_formal_unlock_v4_13(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with test")
        return None
    if args.algo not in V413_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.13 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing v4.13 promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.13.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_algorithms": list(V413_FORMAL_ALGORITHMS),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal test locked: receipt {key!r} does not match")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    replacements = {
        "V412_ALGORITHMS": V413_ALGORITHMS,
        "V412_CANDIDATES": V413_CANDIDATES,
        "V412_FORMAL_ALGORITHMS": V413_FORMAL_ALGORITHMS,
        "V412_FULL": V413_FULL,
        "V412_IMPLEMENTATION_IDS": V413_IMPLEMENTATION_IDS,
        "make_model_v4_12": make_model_v4_13,
        "source_action_repeat_v4_12": source_action_repeat_v4_13,
        "_method_hyperparameters_v4_12": _method_hyperparameters_v4_13,
        "_validate_formal_unlock_v4_12": _validate_formal_unlock_v4_13,
        "_write_json_v4_12": _write_json_v4_13,
        "_write_json_plain_v4_12": _write_json_plain_v4_13,
        "load_model_for_deployment_v4_12": load_model_for_deployment_v4_13,
        "evaluate_with_action_diagnostics_v4_12_model": (
            evaluate_with_action_diagnostics_v4_13_model
        ),
        "decoder_integrity_passed_v4_12": decoder_integrity_passed_v4_13,
        "decoder_integrity_summary_v4_12": decoder_integrity_summary_v4_13,
    }
    originals = {name: getattr(parent, name) for name in replacements}
    for name, value in replacements.items():
        setattr(parent, name, value)
    try:
        return parent.main(
            argv,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        for name, value in originals.items():
            setattr(parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_method_hyperparameters_v4_13",
    "_rewrite_v4_13",
    "_validate_formal_unlock_v4_13",
    "main",
]
