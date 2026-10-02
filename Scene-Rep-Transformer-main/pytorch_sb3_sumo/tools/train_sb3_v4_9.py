"""Train v4.9 with a learned collision critic and no new inference rule."""

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

from configs.sb3_configs_v4_9 import (
    TEMPORAL_GRAPH,
    V49_ALGORITHMS,
    V49_CANDIDATE,
    V49_FORMAL_ALGORITHMS,
    V49_IMPLEMENTATION_IDS,
    make_model_v4_9,
    source_action_repeat_v4_9,
)
from tools import train_sb3_v4_6 as deployment_parent
from tools import train_sb3_v4_8 as parent
from tools.action_diagnostics_v4_9_model import (
    decoder_integrity_passed_v4_9,
    decoder_integrity_summary_v4_9,
    evaluate_with_action_diagnostics_v4_9_model,
    load_model_for_deployment_v4_9,
)
from tools.checkpoint_decoder_selector_v4_6 import CANDIDATE_DECODERS
from tools.checkpoint_decoder_selector_v4_8 import SELECTOR_MODE
from tools.checkpoint_selector_v4_4 import sha256


_PARENT_WRITE = parent._write_json_v4_8
_PARENT_WRITE_PLAIN = parent._write_json_plain_v4_8
_PARENT_METHOD_HYPERPARAMETERS = parent._method_hyperparameters_v4_8


def _version_payload(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        output = {key: _version_payload(item) for key, item in output.items()}
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.8."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.8.", "topo-scene-v4.9.", 1
            )
    elif isinstance(output, list):
        output = [_version_payload(item) for item in output]
    return output


def _write_direct(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _collision_runtime_diagnostics() -> dict[str, Any]:
    model = parent.frozen._ACTIVE_MODEL
    if model is None or not hasattr(model.replay_buffer, "collision_label_diagnostics"):
        raise RuntimeError("v4.9 runtime collision-label diagnostics unavailable")
    return model.replay_buffer.collision_label_diagnostics()


def _rewrite_v4_9(path: Path) -> None:
    if not path.is_file():
        return
    value = _version_payload(json.loads(path.read_text(encoding="utf-8")))
    if isinstance(value, dict):
        if value.get("schema_version") == (
            "topo-scene-v4.9.return-estimator-diagnostics/v1"
        ):
            value["return_estimator_changed_from_v4_8"] = False
        embedded_return = value.get("return_estimator")
        if isinstance(embedded_return, dict):
            embedded_return["return_estimator_changed_from_v4_8"] = False
        algorithm = value.get("algorithm")
        if path.name == "arguments.json":
            algorithm = value.get("requested_raw_steps", {}).get("algo")
            fidelity = dict(value.get("implementation_fidelity", {}))
            fidelity.pop("v4_8_isolated_files", None)
            fidelity.update(
                {
                    "v4_9_isolated_files": True,
                    "single_change": (
                        "learned_twin_collision_value_actor_constraint"
                        if algorithm == V49_CANDIDATE
                        else "none_frozen_control"
                    ),
                    "network_changed": algorithm == V49_CANDIDATE,
                    "training_changed_from_v4_7": (
                        algorithm == V49_CANDIDATE
                    ),
                    "training_changed_from_v4_8": (
                        algorithm == V49_CANDIDATE
                    ),
                    "return_estimator_changed_from_v4_8": False,
                    "learned_collision_critic": algorithm == V49_CANDIDATE,
                    "inference_safety_rule_added": False,
                    "collision_risk_coef": (
                        1.0 if algorithm == V49_CANDIDATE else None
                    ),
                    "selector_changed_from_v4_8": False,
                    "selector_changed": False,
                    "decoder_changed_from_v4_8": (
                        algorithm == V49_CANDIDATE
                    ),
                    "decoder_changed": algorithm == V49_CANDIDATE,
                    "decoder_change_kind": (
                        "learned_reward_minus_collision_value_scoring"
                        if algorithm == V49_CANDIDATE
                        else None
                    ),
                }
            )
            value["implementation_fidelity"] = fidelity
        elif path.name == "training_diagnostics.json" and algorithm == V49_CANDIDATE:
            value["collision_labels"] = _collision_runtime_diagnostics()
            value["learned_collision_critic"] = True
            value["inference_safety_rule_added"] = False
        elif path.name == "paper_evaluation_detailed.json" and algorithm == V49_CANDIDATE:
            value["scientific_version"] = "v4.9_learned_collision_critic"
            value["learned_collision_critic"] = True
            value["collision_risk_coef"] = 1.0
            value["inference_safety_rule_added"] = False
    _write_direct(path, value)


def _write_json_v4_9(path: Path, payload: Any) -> None:
    original_candidate = parent.V48_CANDIDATE
    original_ids = parent.V48_IMPLEMENTATION_IDS
    parent.V48_CANDIDATE = V49_CANDIDATE
    parent.V48_IMPLEMENTATION_IDS = V49_IMPLEMENTATION_IDS
    try:
        _PARENT_WRITE(path, payload)
    finally:
        parent.V48_CANDIDATE = original_candidate
        parent.V48_IMPLEMENTATION_IDS = original_ids
    _rewrite_v4_9(path)
    if path.name == "training_diagnostics.json":
        return_path = path.parent / "return_estimator_diagnostics.json"
        _rewrite_v4_9(return_path)
        if parent.frozen._ACTIVE_MODEL is not None and hasattr(
            parent.frozen._ACTIVE_MODEL.replay_buffer,
            "collision_label_diagnostics",
        ):
            collision_path = path.parent / "collision_label_diagnostics.json"
            _write_direct(
                collision_path,
                _version_payload(_collision_runtime_diagnostics()),
            )


def _write_json_plain_v4_9(path: Path, payload: Any) -> None:
    _PARENT_WRITE_PLAIN(path, payload)
    _rewrite_v4_9(path)


def _method_hyperparameters_v4_9(algorithm: str) -> dict[str, Any]:
    if algorithm == V49_CANDIDATE:
        return {
            "reward_unchanged": True,
            "single_change": "learned_twin_collision_value_actor_constraint",
            "network_changed": True,
            "training_changed_from_v4_8": True,
            "return_estimator_changed_from_v4_8": False,
            "learned_collision_critic": True,
            "collision_critic_heads": 2,
            "collision_probability_link": "sigmoid",
            "collision_twin_aggregation": "maximum",
            "collision_label_source": "observed_environment_info_collision",
            "collision_label_future_or_oracle_used": False,
            "collision_risk_coef": 1.0,
            "inference_safety_rule_added": False,
            "n_step": 16,
            "reward_accumulation": "sum_gamma_power_i_reward",
            "bootstrap_discount": "gamma_power_actual_horizon",
            "checkpoint_candidates": ["highest_training_success", "exact_final"],
            "deployment_decoder_candidates": list(CANDIDATE_DECODERS),
            "checkpoint_decoder_candidate_pairs": 4,
            "checkpoint_calibration_episodes": 12,
            "checkpoint_calibration_partition": "train",
            "selector_mode": SELECTOR_MODE,
            "secondary_calibration_trigger": "empirical_tied_top_count_greater_than_one",
            "secondary_calibration_episodes_per_tied_pair": 12,
            "validation_used_for_checkpoint_selection": False,
            "selector_changed_from_v4_8": False,
            "decoder_changed_from_v4_8": True,
            "decoder_change_kind": "learned_reward_minus_collision_value_scoring",
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "lane_entropy_scale": 0.0,
            "speed_target_entropy": -1.0,
            "actor_non_keep_confidence_threshold": 0.90,
        }
    if algorithm == TEMPORAL_GRAPH:
        value = _PARENT_METHOD_HYPERPARAMETERS(TEMPORAL_GRAPH)
        value.update(
            {
                "single_change": "none_temporal_graph_control",
                "network_changed": False,
                "training_changed_from_v4_8": False,
                "return_estimator_changed_from_v4_8": False,
                "inference_safety_rule_added": False,
            }
        )
        return value
    raise ValueError(f"unsupported v4.9 algorithm {algorithm!r}")


def _validate_formal_unlock_v4_9(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with test")
        return None
    if args.algo not in V49_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.9 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing v4.9 promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.9.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_algorithms": list(V49_FORMAL_ALGORITHMS),
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
        "V48_ALGORITHMS": V49_ALGORITHMS,
        "V48_FORMAL_ALGORITHMS": V49_FORMAL_ALGORITHMS,
        "V48_IMPLEMENTATION_IDS": V49_IMPLEMENTATION_IDS,
        "V48_CANDIDATE": V49_CANDIDATE,
        "make_model_v4_8": make_model_v4_9,
        "source_action_repeat_v4_8": source_action_repeat_v4_9,
        "_method_hyperparameters_v4_8": _method_hyperparameters_v4_9,
        "_validate_formal_unlock_v4_8": _validate_formal_unlock_v4_9,
        "_write_json_v4_8": _write_json_v4_9,
        "_write_json_plain_v4_8": _write_json_plain_v4_9,
        "load_model_for_deployment": load_model_for_deployment_v4_9,
        "evaluate_with_action_diagnostics_v4_6": evaluate_with_action_diagnostics_v4_9_model,
        "decoder_integrity_passed": decoder_integrity_passed_v4_9,
        "decoder_integrity_summary": decoder_integrity_summary_v4_9,
    }
    originals = {name: getattr(parent, name) for name in replacements}
    deployment_originals = {
        "load_model_for_deployment": deployment_parent.load_model_for_deployment,
        "evaluate_with_action_diagnostics_v4_6": deployment_parent.evaluate_with_action_diagnostics_v4_6,
        "decoder_integrity_passed": deployment_parent.decoder_integrity_passed,
        "decoder_integrity_summary": deployment_parent.decoder_integrity_summary,
    }
    for name, value in replacements.items():
        setattr(parent, name, value)
    deployment_parent.load_model_for_deployment = load_model_for_deployment_v4_9
    deployment_parent.evaluate_with_action_diagnostics_v4_6 = (
        evaluate_with_action_diagnostics_v4_9_model
    )
    deployment_parent.decoder_integrity_passed = decoder_integrity_passed_v4_9
    deployment_parent.decoder_integrity_summary = decoder_integrity_summary_v4_9
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
        for name, value in deployment_originals.items():
            setattr(deployment_parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
