"""Train v4.10 shared-risk supported-mixture models without safety rules."""

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

from configs.sb3_configs_v4_10 import (
    TEMPORAL_GRAPH,
    V410_ALGORITHMS,
    V410_CANDIDATES,
    V410_FORMAL_ALGORITHMS,
    V410_FULL,
    V410_IMPLEMENTATION_IDS,
    make_model_v4_10,
    source_action_repeat_v4_10,
)
from tools import train_sb3_v4_7 as selector_parent
from tools import train_sb3_v4_8 as selection_parent
from tools import train_sb3_v4_9 as parent
from tools.action_diagnostics_v4_10_model import (
    decoder_integrity_passed_v4_10,
    decoder_integrity_summary_v4_10,
    evaluate_with_action_diagnostics_v4_10_model,
    load_model_for_deployment_v4_10,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)
from tools.checkpoint_decoder_selector_v4_9_2 import (
    SELECTOR_MODE,
    TARGET_ONLY_DECODERS,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_4 import sha256


_PARENT_WRITE = parent._write_json_v4_9
_PARENT_WRITE_PLAIN = parent._write_json_plain_v4_9


def _version_payload(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        output = {key: _version_payload(item) for key, item in output.items()}
        schema = output.get("schema_version")
        if isinstance(schema, str):
            if schema.startswith("topo-scene-v4.9.2."):
                output["schema_version"] = schema.replace(
                    "topo-scene-v4.9.2.", "topo-scene-v4.10.", 1
                )
            elif schema.startswith("topo-scene-v4.9."):
                output["schema_version"] = schema.replace(
                    "topo-scene-v4.9.", "topo-scene-v4.10.", 1
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
    model = selection_parent.frozen._ACTIVE_MODEL
    if model is None:
        raise RuntimeError("v4.10 active model is unavailable")
    return model


def _rewrite_v4_10(path: Path) -> None:
    if not path.is_file():
        return
    value = _version_payload(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        _write_direct(path, value)
        return
    algorithm = value.get("algorithm")
    if path.name == "arguments.json":
        algorithm = value.get("requested_raw_steps", {}).get("algo")
        fidelity = dict(value.get("implementation_fidelity", {}))
        fidelity.pop("v4_9_isolated_files", None)
        fidelity.pop("actor_non_keep_confidence_threshold", None)
        candidate = algorithm in V410_CANDIDATES
        fidelity.update(
            {
                "v4_10_isolated_files": True,
                "single_change": (
                    "shared_risk_supported_mixture_model"
                    if candidate
                    else "none_frozen_control"
                ),
                "network_changed": candidate,
                "shared_risk_encoder": candidate,
                "learned_speed_mixture": candidate,
                "fixed_speed_grid_at_inference": False,
                "optimizer_parameter_sets_disjoint": candidate,
                "deployment_decoder_candidates": (
                    [TARGET_DECODER] if candidate else [PARENT_DECODER]
                ),
                "checkpoint_decoder_candidate_pairs": 2,
                "fusion_candidate_present": False,
                "actor_confidence_threshold_present": False,
                "inference_safety_rule_added": False,
                "external_kinematic_projection": False,
                "action_postprocessing_override": False,
            }
        )
        value["implementation_fidelity"] = fidelity
    elif path.name == "method_metadata.json":
        value.pop("actor_non_keep_confidence_threshold", None)
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
        value["inference_safety_rule_added"] = False
        if algorithm in V410_CANDIDATES:
            value["deployment_protocol_version"] = "v4.10_target_only"
            value["deployment_decoder_candidates"] = [TARGET_DECODER]
            value["checkpoint_decoder_candidate_pairs"] = 2
            value["selector_mode"] = SELECTOR_MODE
            value["fusion_candidate_present"] = False
            value["actor_confidence_threshold_present"] = False
    elif path.name == "training_diagnostics.json" and algorithm in V410_CANDIDATES:
        model = _active_model()
        value["scientific_version"] = "v4.10_shared_risk_supported_mixture"
        value["optimizer_ownership"] = model.optimizer_ownership_audit()
        value["learned_speed_components_per_lane"] = model.actor.speed_components
        value["fixed_speed_grid_at_inference"] = False
        value["inference_safety_rule_added"] = False
    elif path.name == "paper_evaluation_detailed.json":
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
        value["formal_test_accessed"] = value.get("evaluation_split") == "test"
        if algorithm in V410_CANDIDATES:
            value["scientific_version"] = "v4.10_shared_risk_supported_mixture"
            value["shared_risk_encoder"] = True
            value["learned_speed_components_per_lane"] = (
                _active_model().actor.speed_components
            )
            value["fixed_speed_grid_at_inference"] = False
            value["inference_safety_rule_added"] = False
    elif path.name == "action_diagnostics.json" and value.get(
        "learned_collision_critic_present"
    ) is not True:
        value["learned_collision_critic_present"] = False
        value["learned_collision_critic_target_present"] = False
        value["shared_risk_encoder_present"] = False
        value["fixed_speed_grid_at_inference"] = False
        value["inference_safety_rule_added"] = False
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
    elif path.name == "check_result.json" and algorithm in V410_CANDIDATES:
        action = value.get("action")
        lane = action[1] if isinstance(action, list) and len(action) >= 2 else None
        value["exact_hybrid_lane_code"] = lane in (-1.0, 0.0, 1.0)
    if path.name == "receipt.json" and value.get(
        "selected_deployment_decoder"
    ) == TARGET_DECODER:
        value["schema_version"] = (
            "topo-scene-v4.10.target-only-tie-replicated-selector/v1"
        )
        value["deployment_protocol_version"] = "v4.10_target_only"
        value["selector_mode"] = SELECTOR_MODE
        value["candidate_count"] = 2
        value["deployment_decoder_candidates"] = [TARGET_DECODER]
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
        value["calibration_deterministic_lane_decoder"] = TARGET_DECODER
        value["calibration_uses_method_own_frozen_decoder"] = True
    _write_direct(path, value)


def _write_json_v4_10(path: Path, payload: Any) -> None:
    _PARENT_WRITE(path, payload)
    _rewrite_v4_10(path)
    if path.name == "training_diagnostics.json":
        _rewrite_v4_10(path.parent / "return_estimator_diagnostics.json")
        _rewrite_v4_10(path.parent / "collision_label_diagnostics.json")


def _write_json_plain_v4_10(path: Path, payload: Any) -> None:
    _PARENT_WRITE_PLAIN(path, payload)
    _rewrite_v4_10(path)


def _method_hyperparameters_v4_10(algorithm: str) -> dict[str, Any]:
    if algorithm == TEMPORAL_GRAPH:
        value = parent._PARENT_METHOD_HYPERPARAMETERS(TEMPORAL_GRAPH)
        value.update(
            {
                "single_change": "none_temporal_graph_control",
                "network_changed": False,
                "inference_safety_rule_added": False,
            }
        )
        return value
    if algorithm not in V410_CANDIDATES:
        raise ValueError(f"unsupported v4.10 algorithm {algorithm!r}")
    variants = {
        V410_FULL: (3, 0.05, 0.25, 0.25, 0.05, "primary_srsm_full"),
        V410_CANDIDATES[1]: (1, 0.05, 0.25, 0.25, 0.05, "ablation_shared_single"),
        V410_CANDIDATES[2]: (3, 0.0, 0.25, 0.0, 0.0, "ablation_mixture_no_support"),
    }
    components, support, component_entropy, uncertainty, prior, role = variants[
        algorithm
    ]
    return {
        "reward_unchanged": True,
        "single_change": "shared_risk_supported_mixture_model",
        "v4_10_role": role,
        "network_changed": True,
        "shared_online_scene_encoder": True,
        "actor_encoder_detach": True,
        "optimizer_parameter_sets_must_be_disjoint": True,
        "encoder_optimizer_owner_count": 1,
        "learned_speed_components_per_lane": components,
        "fixed_speed_grid_at_inference": False,
        "replay_support_objective": "latent_speed_mixture_negative_log_likelihood",
        "replay_support_coef": support,
        "component_entropy_scale": component_entropy,
        "twin_uncertainty_coef": uncertainty,
        "component_prior_coef": prior,
        "collision_risk_coef": 1.0,
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
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
        "secondary_calibration_trigger": "empirical_tied_top_count_greater_than_one",
        "secondary_calibration_episodes_per_tied_pair": 12,
        "validation_used_for_checkpoint_selection": False,
        "slot_balance_coef": 0.01,
        "lane_entropy_scale": 0.0,
        "speed_target_entropy": -1.0,
    }


def _validate_formal_unlock_v4_10(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with test")
        return None
    if args.algo not in V410_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.10 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing v4.10 promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.10.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_algorithms": list(V410_FORMAL_ALGORITHMS),
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
    argv_list = list(sys.argv[1:] if argv is None else argv)
    algorithm = (
        argv_list[argv_list.index("--algo") + 1]
        if "--algo" in argv_list
        else V410_FULL
    )
    active_candidate = algorithm if algorithm in V410_CANDIDATES else V410_FULL
    replacements = {
        "V49_ALGORITHMS": V410_ALGORITHMS,
        "V49_FORMAL_ALGORITHMS": V410_FORMAL_ALGORITHMS,
        "V49_IMPLEMENTATION_IDS": V410_IMPLEMENTATION_IDS,
        "V49_CANDIDATE": active_candidate,
        "make_model_v4_9": make_model_v4_10,
        "source_action_repeat_v4_9": source_action_repeat_v4_10,
        "_method_hyperparameters_v4_9": _method_hyperparameters_v4_10,
        "_validate_formal_unlock_v4_9": _validate_formal_unlock_v4_10,
        "_write_json_v4_9": _write_json_v4_10,
        "_write_json_plain_v4_9": _write_json_plain_v4_10,
        "load_model_for_deployment_v4_9": load_model_for_deployment_v4_10,
        "evaluate_with_action_diagnostics_v4_9_model": (
            evaluate_with_action_diagnostics_v4_10_model
        ),
        "decoder_integrity_passed_v4_9": decoder_integrity_passed_v4_10,
        "decoder_integrity_summary_v4_9": decoder_integrity_summary_v4_10,
        "CANDIDATE_DECODERS": TARGET_ONLY_DECODERS,
        "SELECTOR_MODE": SELECTOR_MODE,
    }
    originals = {name: getattr(parent, name) for name in replacements}
    selector_original = selector_parent.CANDIDATE_DECODERS
    selection_originals = {
        "CANDIDATE_DECODERS": selection_parent.CANDIDATE_DECODERS,
        "SELECTOR_MODE": selection_parent.SELECTOR_MODE,
        "tied_top_pairs": selection_parent.tied_top_pairs,
        "select_deployment": selection_parent.select_deployment,
    }
    selector_parent.CANDIDATE_DECODERS = TARGET_ONLY_DECODERS
    selection_parent.CANDIDATE_DECODERS = TARGET_ONLY_DECODERS
    selection_parent.SELECTOR_MODE = SELECTOR_MODE
    selection_parent.tied_top_pairs = tied_top_pairs
    selection_parent.select_deployment = select_deployment
    for name, value in replacements.items():
        setattr(parent, name, value)
    try:
        return parent.main(
            argv_list,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        for name, value in originals.items():
            setattr(parent, name, value)
        selector_parent.CANDIDATE_DECODERS = selector_original
        for name, value in selection_originals.items():
            setattr(selection_parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
