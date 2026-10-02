"""Train v4.12 augmented joint-support PRCR models without safety rules."""

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

from configs.sb3_configs_v4_12 import (
    TEMPORAL_GRAPH,
    V412_ALGORITHMS,
    V412_AUGMENTATION_ONLY,
    V412_CANDIDATES,
    V412_FORMAL_ALGORITHMS,
    V412_FULL,
    V412_IMPLEMENTATION_IDS,
    V412_NO_CROSS_AUGMENTATION,
    make_model_v4_12,
    source_action_repeat_v4_12,
)
from tools import train_sb3_v4_11 as parent
from tools.action_diagnostics_v4_12_model import (
    decoder_integrity_passed_v4_12,
    decoder_integrity_summary_v4_12,
    evaluate_with_action_diagnostics_v4_12_model,
    load_model_for_deployment_v4_12,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE
from tools.checkpoint_selector_v4_4 import sha256


_PARENT_WRITE = parent._write_json_v4_11
_PARENT_WRITE_PLAIN = parent._write_json_plain_v4_11
_PARENT_METHOD_HYPERPARAMETERS = parent._method_hyperparameters_v4_11


def _version_payload(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        output = {key: _version_payload(item) for key, item in output.items()}
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.11."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.11.", "topo-scene-v4.12.", 1
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


def _rewrite_v4_12(path: Path) -> None:
    if not path.is_file():
        return
    value = _version_payload(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        _write_direct(path, value)
        return
    algorithm = value.get("algorithm")
    if path.name == "arguments.json":
        algorithm = value.get("requested_raw_steps", {}).get("algo")
        candidate = algorithm in V412_CANDIDATES
        fidelity = dict(value.get("implementation_fidelity", {}))
        fidelity.pop("v4_11_isolated_files", None)
        fidelity.update(
            {
                "v4_12_isolated_files": True,
                "single_change": (
                    "augmented_joint_replay_supported_ranked_risk_model"
                    if candidate
                    else "none_frozen_control"
                ),
                "network_changed": candidate,
                "joint_replay_action_support": candidate,
                "continuous_lane_log_prior": candidate,
                "fixed_speed_grid_at_inference": False,
                "deployment_decoder_candidates": (
                    [TARGET_DECODER] if candidate else [PARENT_DECODER]
                ),
                "actor_confidence_threshold_present": False,
                "inference_safety_rule_added": False,
                "external_kinematic_projection": False,
                "kinematic_safety_projection": False,
                "traffic_risk_in_lane_mask": False,
                "action_postprocessing_override": False,
            }
        )
        value["implementation_fidelity"] = fidelity
    elif path.name == "method_metadata.json":
        value["external_kinematic_projection"] = False
        value["kinematic_safety_projection"] = False
        value["traffic_risk_in_lane_mask"] = False
        value["action_postprocessing_override"] = False
        value["inference_safety_rule_added"] = False
        if algorithm in V412_CANDIDATES:
            value["deployment_protocol_version"] = "v4.12_target_only"
            value["deployment_decoder_candidates"] = [TARGET_DECODER]
            value["checkpoint_decoder_candidate_pairs"] = 2
            value["selector_mode"] = SELECTOR_MODE
            value["fusion_candidate_present"] = False
            value["actor_confidence_threshold_present"] = False
    elif path.name == "training_diagnostics.json" and algorithm in V412_CANDIDATES:
        model = _active_model()
        value["scientific_version"] = "v4.12_augmented_joint_support_prcr"
        value["optimizer_ownership"] = model.optimizer_ownership_audit()
        value["learned_speed_components_per_lane"] = model.actor.speed_components
        value["joint_replay_action_support_present"] = True
        value["replay_support_objective"] = (
            "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll"
        )
        value["lane_support_scale"] = float(model.actor.lane_support_scale)
        value["lane_prior_coef"] = float(model.policy.lane_prior_coef)
        value["effective_random_rotation_augmentation"] = bool(
            model.critic.features_extractor.random_augmentation
        )
        value["fixed_speed_grid_at_inference"] = False
        value["inference_safety_rule_added"] = False
        value["external_kinematic_projection"] = False
        value["kinematic_safety_projection"] = False
        value["traffic_risk_in_lane_mask"] = False
        value["actor_confidence_gate"] = False
        value["action_postprocessing_override"] = False
    elif path.name == "paper_evaluation_detailed.json":
        value["external_kinematic_projection"] = False
        value["kinematic_safety_projection"] = False
        value["traffic_risk_in_lane_mask"] = False
        value["action_postprocessing_override"] = False
        value["formal_test_accessed"] = value.get("evaluation_split") == "test"
        if algorithm in V412_CANDIDATES:
            model = _active_model()
            value["scientific_version"] = "v4.12_augmented_joint_support_prcr"
            value["joint_replay_action_support_present"] = True
            value["lane_support_scale"] = float(model.actor.lane_support_scale)
            value["lane_prior_coef"] = float(model.policy.lane_prior_coef)
            value["effective_random_rotation_augmentation"] = bool(
                model.critic.features_extractor.random_augmentation
            )
            value["actor_confidence_gate"] = False
            value["inference_safety_rule_added"] = False
    elif path.name == "action_diagnostics.json" and value.get(
        "joint_replay_action_support_present"
    ) is not True:
        value["joint_replay_action_support_present"] = False
        value["fixed_speed_grid_at_inference"] = False
        value["inference_safety_rule_added"] = False
        value["external_kinematic_projection"] = False
        value["kinematic_safety_projection"] = False
        value["traffic_risk_in_lane_mask"] = False
        value["actor_confidence_gate"] = False
        value["action_postprocessing_override"] = False
    elif path.name == "check_result.json" and algorithm in V412_CANDIDATES:
        action = value.get("action")
        lane = action[1] if isinstance(action, list) and len(action) >= 2 else None
        value["exact_hybrid_lane_code"] = lane in (-1.0, 0.0, 1.0)
    if path.name == "receipt.json" and value.get(
        "selected_deployment_decoder"
    ) == TARGET_DECODER:
        value["schema_version"] = (
            "topo-scene-v4.12.target-only-tie-replicated-selector/v1"
        )
        value["deployment_protocol_version"] = "v4.12_target_only"
        value["selector_mode"] = SELECTOR_MODE
        value["candidate_count"] = 2
        value["deployment_decoder_candidates"] = [TARGET_DECODER]
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
        value["external_kinematic_projection"] = False
        value["kinematic_safety_projection"] = False
        value["traffic_risk_in_lane_mask"] = False
        value["action_postprocessing_override"] = False
        value["calibration_deterministic_lane_decoder"] = TARGET_DECODER
        value["calibration_uses_method_own_frozen_decoder"] = True
    _write_direct(path, value)


def _write_json_v4_12(path: Path, payload: Any) -> None:
    _PARENT_WRITE(path, payload)
    _rewrite_v4_12(path)
    if path.name == "training_diagnostics.json":
        _rewrite_v4_12(path.parent / "return_estimator_diagnostics.json")
        _rewrite_v4_12(path.parent / "collision_label_diagnostics.json")


def _write_json_plain_v4_12(path: Path, payload: Any) -> None:
    _PARENT_WRITE_PLAIN(path, payload)
    _rewrite_v4_12(path)


def _method_hyperparameters_v4_12(algorithm: str) -> dict[str, Any]:
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
    variants = {
        V412_FULL: (1.0, 0.05, True, "primary_augmented_joint_support"),
        V412_NO_CROSS_AUGMENTATION: (
            1.0,
            0.05,
            False,
            "ablation_no_cross_augmentation",
        ),
        V412_AUGMENTATION_ONLY: (
            0.0,
            0.0,
            True,
            "ablation_cross_augmentation_only",
        ),
    }
    if algorithm not in variants:
        raise ValueError(f"unsupported v4.12 algorithm {algorithm!r}")
    lane_scale, lane_prior, cross_augmentation, role = variants[algorithm]
    return {
        "reward_unchanged": True,
        "single_change": "augmented_joint_replay_supported_ranked_risk_model",
        "v4_12_role": role,
        "network_changed": True,
        "shared_online_scene_encoder": True,
        "actor_encoder_detach": True,
        "optimizer_parameter_sets_must_be_disjoint": True,
        "encoder_optimizer_owner_count": 1,
        "learned_speed_components_per_lane": 3,
        "fixed_speed_grid_at_inference": False,
        "replay_support_objective": (
            "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll"
        ),
        "replay_support_coef": 0.05,
        "lane_support_scale": lane_scale,
        "lane_prior_coef": lane_prior,
        "component_prior_coef": 0.05,
        "cross_rotation_augmentation": cross_augmentation,
        "collision_risk_coef": 1.0,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": 0.25,
        "collision_pairwise_hard_threshold": False,
        "collision_pairwise_fixed_margin": False,
        "collision_temporal_consistency_coef": 0.05,
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
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


def _validate_formal_unlock_v4_12(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with test")
        return None
    if args.algo not in V412_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.12 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing v4.12 promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.12.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_algorithms": list(V412_FORMAL_ALGORITHMS),
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
        "V411_ALGORITHMS": V412_ALGORITHMS,
        "V411_CANDIDATES": V412_CANDIDATES,
        "V411_FORMAL_ALGORITHMS": V412_FORMAL_ALGORITHMS,
        "V411_FULL": V412_FULL,
        "V411_IMPLEMENTATION_IDS": V412_IMPLEMENTATION_IDS,
        "make_model_v4_11": make_model_v4_12,
        "source_action_repeat_v4_11": source_action_repeat_v4_12,
        "_method_hyperparameters_v4_11": _method_hyperparameters_v4_12,
        "_validate_formal_unlock_v4_11": _validate_formal_unlock_v4_12,
        "_write_json_v4_11": _write_json_v4_12,
        "_write_json_plain_v4_11": _write_json_plain_v4_12,
        "load_model_for_deployment_v4_11": load_model_for_deployment_v4_12,
        "evaluate_with_action_diagnostics_v4_11_model": (
            evaluate_with_action_diagnostics_v4_12_model
        ),
        "decoder_integrity_passed_v4_11": decoder_integrity_passed_v4_12,
        "decoder_integrity_summary_v4_11": decoder_integrity_summary_v4_12,
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
    "_method_hyperparameters_v4_12",
    "_rewrite_v4_12",
    "_validate_formal_unlock_v4_12",
    "main",
]
