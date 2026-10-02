"""Strict target-only deployment wrapper for the frozen v4.9 learned model.

This version does not add a safety rule and does not alter the learned model,
losses, collision labels, or training budget.  It removes the inherited
``fusion_0_90`` candidate and selects only between two learned target-critic
checkpoints using train-only calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_9 import (
    TEMPORAL_GRAPH,
    V49_CANDIDATE,
    V49_FORMAL_ALGORITHMS,
)
from tools import train_sb3_v4_7 as train_v4_7
from tools import train_sb3_v4_8 as train_v4_8
from tools import train_sb3_v4_9 as parent
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import (
    SELECTOR_MODE,
    TARGET_ONLY_DECODERS,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_4 import sha256


_PARENT_WRITE = parent._write_json_v4_9
_PARENT_WRITE_PLAIN = parent._write_json_plain_v4_9
_PARENT_HYPERPARAMETERS = parent._method_hyperparameters_v4_9


def _version_payload(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        output = {key: _version_payload(item) for key, item in output.items()}
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.9."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.9.", "topo-scene-v4.9.2.", 1
            )
    elif isinstance(output, list):
        output = [_version_payload(item) for item in output]
    return output


def _write_direct(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rewrite_v4_9_2(path: Path) -> None:
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
        fidelity.pop("actor_non_keep_confidence_threshold", None)
        fidelity.update(
            {
                "v4_9_2_isolated_files": True,
                "deployment_protocol_version": "v4.9.2_strict_target_only",
                "scientific_model_changed_from_v4_9": False,
                "training_changed_from_v4_9": False,
                "selector_changed_from_v4_9": algorithm == V49_CANDIDATE,
                "selector_changed": algorithm == V49_CANDIDATE,
                "decoder_candidate_set_changed_from_v4_9": (
                    algorithm == V49_CANDIDATE
                ),
                "deployment_decoder_candidates": (
                    [TARGET_DECODER] if algorithm == V49_CANDIDATE else ["parent_control"]
                ),
                "checkpoint_decoder_candidate_pairs": 2,
                "fusion_candidate_present": False,
                "actor_confidence_threshold_present": False,
                "external_kinematic_projection": False,
                "headway_or_ttc_threshold": False,
                "lane_change_veto": False,
                "manual_geometry_collision_label": False,
                "action_postprocessing_override": False,
            }
        )
        value["implementation_fidelity"] = fidelity
    elif path.name == "check_result.json":
        value["checkpoint_decoder_candidate_pairs"] = 2
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
    elif path.name == "method_metadata.json" and algorithm == V49_CANDIDATE:
        value.pop("actor_non_keep_confidence_threshold", None)
        value["deployment_protocol_version"] = "v4.9.2_strict_target_only"
        value["deployment_decoder_candidates"] = [TARGET_DECODER]
        value["checkpoint_decoder_candidate_pairs"] = 2
        value["selector_mode"] = SELECTOR_MODE
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
    elif path.name == "paper_evaluation_detailed.json" and algorithm == V49_CANDIDATE:
        value["scientific_version"] = "v4.9.2_strict_target_only"
        value["deployment_decoder_candidates"] = [TARGET_DECODER]
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
    elif (
        path.name == "action_diagnostics.json"
        and value.get("learned_collision_critic_present") is True
    ):
        value["deployment_protocol_version"] = "v4.9.2_strict_target_only"
        value["fusion_candidate_present"] = False
        value["actor_confidence_threshold_present"] = False
        value["external_kinematic_projection"] = False
        value["action_postprocessing_override"] = False
    if path.name == "receipt.json" and value.get("selected_deployment_decoder") == TARGET_DECODER:
        value["deployment_protocol_version"] = "v4.9.2_strict_target_only"
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


def _write_json_v4_9_2(path: Path, payload: Any) -> None:
    _PARENT_WRITE(path, payload)
    _rewrite_v4_9_2(path)
    if path.name == "training_diagnostics.json":
        for name in (
            "return_estimator_diagnostics.json",
            "collision_label_diagnostics.json",
        ):
            _rewrite_v4_9_2(path.parent / name)


def _write_json_plain_v4_9_2(path: Path, payload: Any) -> None:
    _PARENT_WRITE_PLAIN(path, payload)
    _rewrite_v4_9_2(path)


def _method_hyperparameters_v4_9_2(algorithm: str) -> dict[str, Any]:
    value = copy.deepcopy(_PARENT_HYPERPARAMETERS(algorithm))
    value["deployment_protocol_version"] = "v4.9.2_strict_target_only"
    value["scientific_model_changed_from_v4_9"] = False
    value["training_changed_from_v4_9"] = False
    value["external_kinematic_projection"] = False
    value["headway_or_ttc_threshold"] = False
    value["lane_change_veto"] = False
    value["manual_geometry_collision_label"] = False
    value["action_postprocessing_override"] = False
    value["fusion_candidate_present"] = False
    value["actor_confidence_threshold_present"] = False
    value.pop("actor_non_keep_confidence_threshold", None)
    if algorithm == V49_CANDIDATE:
        value.update(
            {
                "deployment_decoder_candidates": [TARGET_DECODER],
                "checkpoint_decoder_candidate_pairs": 2,
                "selector_mode": SELECTOR_MODE,
                "secondary_calibration_trigger": (
                    "empirical_target_checkpoint_tie_only"
                ),
                "selector_changed_from_v4_9": True,
                "selector_changed": True,
                "decoder_candidate_set_changed_from_v4_9": True,
                "validation_used_for_checkpoint_selection": False,
                "formal_test_used_for_checkpoint_selection": False,
            }
        )
    else:
        value.update(
            {
                "deployment_decoder_candidates": ["parent_control"],
                "checkpoint_decoder_candidate_pairs": 2,
                "selector_changed_from_v4_9": False,
                "selector_changed": False,
                "decoder_candidate_set_changed_from_v4_9": False,
            }
        )
    return value


def _validate_formal_unlock_v4_9_2(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with test")
        return None
    if args.algo not in V49_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.9.2 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing v4.9.2 promotion receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.9.2.promotion-gate/v1",
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
        "v47_decoders": (train_v4_7, "CANDIDATE_DECODERS", TARGET_ONLY_DECODERS),
        "v48_decoders": (train_v4_8, "CANDIDATE_DECODERS", TARGET_ONLY_DECODERS),
        "v48_selector_mode": (train_v4_8, "SELECTOR_MODE", SELECTOR_MODE),
        "v48_tied": (train_v4_8, "tied_top_pairs", tied_top_pairs),
        "v48_select": (train_v4_8, "select_deployment", select_deployment),
        "v49_decoders": (parent, "CANDIDATE_DECODERS", TARGET_ONLY_DECODERS),
        "v49_selector_mode": (parent, "SELECTOR_MODE", SELECTOR_MODE),
        "v49_write": (parent, "_write_json_v4_9", _write_json_v4_9_2),
        "v49_write_plain": (
            parent,
            "_write_json_plain_v4_9",
            _write_json_plain_v4_9_2,
        ),
        "v49_hyperparameters": (
            parent,
            "_method_hyperparameters_v4_9",
            _method_hyperparameters_v4_9_2,
        ),
        "v49_formal_unlock": (
            parent,
            "_validate_formal_unlock_v4_9",
            _validate_formal_unlock_v4_9_2,
        ),
    }
    originals = {
        key: getattr(module, name)
        for key, (module, name, _) in replacements.items()
    }
    for module, name, replacement in replacements.values():
        setattr(module, name, replacement)
    try:
        return parent.main(
            argv,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        for key, (module, name, _) in replacements.items():
            setattr(module, name, originals[key])


if __name__ == "__main__":
    raise SystemExit(main())
