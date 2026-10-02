"""Train v4.8 by adding tie-only replicated calibration to frozen v4.7."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_8 import (
    TEMPORAL_GRAPH,
    V48_ALGORITHMS,
    V48_CANDIDATE,
    V48_FORMAL_ALGORITHMS,
    V48_IMPLEMENTATION_IDS,
    make_model_v4_8,
    source_action_repeat_v4_8,
)
from tools import train_sb3_v4_6 as deployment_parent
from tools import train_sb3_v4_7 as frozen
from tools.action_diagnostics_v4_6 import (
    decoder_integrity_passed,
    decoder_integrity_summary,
    evaluate_with_action_diagnostics_v4_6,
    load_model_for_deployment,
)
from tools.checkpoint_decoder_selector_v4_6 import CANDIDATE_DECODERS
from tools.checkpoint_decoder_selector_v4_8 import (
    SELECTOR_MODE,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_4 import sha256
from tools.paper_evaluation_contract import validate_model_environment_spaces


SECONDARY_SEED_OFFSET = 100
ENGINEERING_PATCHES = (
    "v4_7_1_bind_sealed_selected_deployment_fields",
    "v4_7_2_preserve_parent_preregistration_context",
)
_FROZEN_EVALUATE = deployment_parent._evaluate_checkpoint_v4_6
_FROZEN_SELECT_ADAPTER = deployment_parent._select_deployment_adapter
_CONTEXTS: dict[str, dict[str, Any]] = {}
_ACTIVE_ALGORITHM: str | None = None


def _version_payload(payload: Any) -> Any:
    value = copy.deepcopy(payload)
    if isinstance(value, dict):
        value = {key: _version_payload(item) for key, item in value.items()}
        schema = value.get("schema_version")
        if isinstance(schema, str):
            for old in (
                "topo-scene-v4.5.",
                "topo-scene-v4.6.",
                "topo-scene-v4.7.",
            ):
                if schema.startswith(old):
                    value["schema_version"] = schema.replace(
                        old, "topo-scene-v4.8.", 1
                    )
                    break
    elif isinstance(value, list):
        value = [_version_payload(item) for item in value]
    return value


def _write_plain(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _selected_deployment_bindings() -> dict[str, str]:
    selector = deployment_parent._SELECTED_DEPLOYMENT
    if selector is None:
        raise RuntimeError("selected deployment is not sealed before detailed output")
    bindings = {
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_source_checkpoint_sha256": selector[
            "selected_checkpoint_sha256"
        ],
        "selected_model_policy_class": selector["selected_model_policy_class"],
        "selected_model_parameter_state_sha256": selector[
            "selected_model_parameter_state_sha256"
        ],
    }
    if not all(isinstance(value, str) and value for value in bindings.values()):
        raise ValueError("sealed selector contains an empty deployment binding")
    return bindings


def _write_json_v4_8(path: Path, payload: Any) -> None:
    value = _version_payload(payload)
    if isinstance(value, dict):
        if path.name == "arguments.json":
            algorithm = value.get("requested_raw_steps", {}).get("algo")
            fidelity = dict(value.get("implementation_fidelity", {}))
            for key in ("v4_5_isolated_files", "v4_6_isolated_files", "v4_7_isolated_files"):
                fidelity.pop(key, None)
            fidelity.update(
                {
                    "implementation_id": V48_IMPLEMENTATION_IDS[algorithm],
                    "v4_8_isolated_files": True,
                    "single_change": (
                        "tie_only_replicated_train_calibration"
                        if algorithm == V48_CANDIDATE
                        else "none_frozen_control"
                    ),
                    "return_estimator_changed_from_v4_7": False,
                    "training_changed_from_v4_7": False,
                    "n_step": 16 if algorithm == V48_CANDIDATE else 4,
                    "bootstrap_discount": (
                        "gamma_power_actual_horizon"
                        if algorithm == V48_CANDIDATE
                        else "single_gamma_source_equivalent"
                    ),
                    "checkpoint_decoder_candidate_pairs": (
                        4 if algorithm == V48_CANDIDATE else 2
                    ),
                    "selector_changed": algorithm == V48_CANDIDATE,
                    "selector_mode": (
                        SELECTOR_MODE if algorithm == V48_CANDIDATE else "checkpoint_only_parent_control"
                    ),
                    "secondary_calibration_seed_offset": (
                        SECONDARY_SEED_OFFSET if algorithm == V48_CANDIDATE else None
                    ),
                    "decoder_changed": False,
                }
            )
            value["implementation_fidelity"] = fidelity
            requested = value.get("requested_raw_steps", {})
            if algorithm == V48_CANDIDATE:
                requested["secondary_calibration_seed_start"] = int(
                    requested["calibration_seed_start"]
                ) + SECONDARY_SEED_OFFSET
        elif path.name == "check_result.json":
            algorithm = value.get("algorithm")
            if algorithm != TEMPORAL_GRAPH:
                action = np.asarray(value.get("action", [])).reshape(-1)
                value["exact_hybrid_lane_code"] = bool(
                    len(action) >= 2 and float(action[1]) in (-1.0, 0.0, 1.0)
                )
                value["checkpoint_decoder_candidate_pairs"] = 4
                value["selector_mode"] = SELECTOR_MODE
            value["return_estimator"] = _version_payload(
                frozen._return_estimator_diagnostics(algorithm)
            )
        elif path.name == "training_diagnostics.json":
            algorithm = value.get("algorithm")
            diagnostics = _version_payload(
                frozen._return_estimator_diagnostics(algorithm)
            )
            value["return_estimator"] = diagnostics
            _write_plain(
                path.parent / "return_estimator_diagnostics.json",
                diagnostics,
            )
        elif path.name == "paper_evaluation_detailed.json":
            bindings = _selected_deployment_bindings()
            for key, expected in bindings.items():
                if key in value and value[key] != expected:
                    raise ValueError(f"detailed evaluation conflicts with sealed {key}")
            value.update(bindings)
            value.update(
                {
                    "scientific_version": "v4.8_tie_only_replicated_train_calibration",
                    "inherited_engineering_patches": list(ENGINEERING_PATCHES),
                    "selected_deployment_bindings_source": (
                        "selector_receipt_sealed_before_validation"
                    ),
                    "scientific_protocol_changed_by_engineering_patch": False,
                }
            )
    _write_plain(path, value)


def _write_json_plain_v4_8(path: Path, payload: Any) -> None:
    value = _version_payload(payload)
    if (
        isinstance(value, dict)
        and path.name == "receipt.json"
        and value.get("selector_mode") == SELECTOR_MODE
    ):
        value["schema_version"] = (
            "topo-scene-v4.8.tie-only-replicated-selector/v1"
        )
    _write_plain(path, value)


def _method_hyperparameters_v4_8(algorithm: str) -> dict[str, Any]:
    if algorithm == V48_CANDIDATE:
        return {
            "reward_unchanged": True,
            "single_change": "tie_only_replicated_train_calibration",
            "training_changed_from_v4_7": False,
            "return_estimator_changed_from_v4_7": False,
            "n_step": 16,
            "reward_accumulation": "sum_gamma_power_i_reward",
            "actual_horizon": "min_16_or_first_episode_boundary",
            "bootstrap_discount": "gamma_power_actual_horizon",
            "checkpoint_candidates": ["highest_training_success", "exact_final"],
            "deployment_decoder_candidates": list(CANDIDATE_DECODERS),
            "checkpoint_decoder_candidate_pairs": 4,
            "checkpoint_calibration_episodes": 12,
            "checkpoint_calibration_partition": "train",
            "selector_mode": SELECTOR_MODE,
            "secondary_calibration_trigger": (
                "empirical_tied_top_count_greater_than_one"
            ),
            "secondary_calibration_episodes_per_tied_pair": 12,
            "secondary_calibration_seed_offset": SECONDARY_SEED_OFFSET,
            "non_top_candidate_reentry_forbidden": True,
            "validation_used_for_checkpoint_selection": False,
            "selector_changed": True,
            "decoder_changed": False,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": [-1, 0, 1],
            "lane_action_mask": True,
            "lane_conditioned_speed": True,
            "critic_lane_one_hot": True,
            "speed_target_entropy": -1.0,
            "lane_entropy_scale": 0.0,
            "actor_non_keep_confidence_threshold": 0.90,
        }
    if algorithm == TEMPORAL_GRAPH:
        return {
            "reward_unchanged": True,
            "single_change": "none_temporal_graph_control",
            "training_changed_from_v4_7": False,
            "return_estimator_changed_from_v4_7": False,
            "n_step": 4,
            "bootstrap_discount": "single_gamma_source_equivalent",
            "checkpoint_candidates": ["highest_training_success", "exact_final"],
            "deployment_decoder_candidates": ["parent_control"],
            "checkpoint_decoder_candidate_pairs": 2,
            "checkpoint_calibration_episodes": 12,
            "checkpoint_calibration_partition": "train",
            "selector_mode": "checkpoint_only_parent_control",
            "selector_changed": False,
            "decoder_changed": False,
            "comparator_implementation_unchanged": True,
        }
    raise ValueError(f"unsupported v4.8 algorithm {algorithm!r}")


def _validate_formal_unlock_v4_8(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with the test split")
        return None
    if args.algo not in V48_FORMAL_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.8 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.8.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal test locked: receipt {key!r} does not match")
    if receipt.get("formal_algorithms") != list(V48_FORMAL_ALGORITHMS):
        raise ValueError("formal test locked: receipt method pair drifted")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def _evaluate_checkpoint_v4_8(**kwargs: Any) -> dict[str, Any]:
    result = _FROZEN_EVALUATE(**kwargs)
    _CONTEXTS[str(kwargs["checkpoint_kind"])] = dict(kwargs)
    return result


def _evaluate_secondary_pair(
    *,
    checkpoint_kind: str,
    deployment_decoder: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(context["checkpoint_path"])
    args = context["args"]
    env_factory = context["env_factory"]
    run_dir = Path(context["run_dir"])
    model_class = context["model_class"]
    short_kind = {
        "highest_training_success": "best",
        "exact_final": "final",
    }[checkpoint_kind]
    output = run_dir / "selector" / "cal_secondary" / short_kind / deployment_decoder
    output.mkdir(parents=True, exist_ok=False)
    secondary_seed = int(args.calibration_seed_start) + SECONDARY_SEED_OFFSET
    calibration_args = deployment_parent.base._partition_args(args, "train")
    calibration_env = env_factory(calibration_args, evaluation=True)
    try:
        model = load_model_for_deployment(
            model_class,
            checkpoint_path,
            decoder=deployment_decoder,
            env=calibration_env,
            device=args.device,
        )
        space_contract = validate_model_environment_spaces(model, calibration_env)
        report, diagnostics = evaluate_with_action_diagnostics_v4_6(
            model,
            calibration_env,
            deployment_decoder=deployment_decoder,
            episodes=args.calibration_episodes,
            seed=secondary_seed,
            trace_path=output / "decisions.jsonl",
        )
    finally:
        calibration_env.close()
    if not decoder_integrity_passed(diagnostics):
        raise ValueError(
            f"secondary decoder integrity failed: {checkpoint_kind} x {deployment_decoder}"
        )
    summary = report.summary.to_dict()
    integrity = decoder_integrity_summary(diagnostics)
    detailed = {
        "schema_version": "topo-scene-v4.8.secondary-checkpoint-decoder-calibration/v1",
        "computed_from_real_run": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "tie_only_triggered": True,
        "candidate_was_empirical_tied_top": True,
        "checkpoint_kind": checkpoint_kind,
        "deployment_decoder": deployment_decoder,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "algorithm": args.algo,
        "scenario": args.scenario,
        "training_seed": args.seed,
        "traffic_partition": "train",
        "calibration_seed_start": secondary_seed,
        "calibration_episodes": args.calibration_episodes,
        "space_contract": space_contract,
        "decoder_integrity": integrity,
        "decoder_integrity_passed": True,
        **report.to_dict(),
    }
    deployment_parent._write_json_plain(output / "evaluation.json", summary)
    deployment_parent._write_json_plain(output / "actions.json", diagnostics)
    deployment_parent._write_json_plain(output / "detailed.json", detailed)
    return {
        "checkpoint_kind": checkpoint_kind,
        "deployment_decoder": deployment_decoder,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "traffic_partition": "train",
        "calibration_seed_start": secondary_seed,
        "summary": summary,
        "episode_records": detailed["episode_records"],
        "calibration_result_sha256": sha256(output / "detailed.json"),
        "calibration_action_diagnostics_sha256": sha256(output / "actions.json"),
        "decoder_integrity": integrity,
        "decoder_integrity_passed": True,
    }


def _select_deployment_adapter_v4_8(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    if _ACTIVE_ALGORITHM != V48_CANDIDATE:
        return _FROZEN_SELECT_ADAPTER(checkpoints)
    candidates = [
        pair
        for checkpoint in checkpoints
        for pair in checkpoint["deployment_candidates"]
    ]
    fallback_used = not any(
        row["checkpoint_kind"] == "highest_training_success" for row in candidates
    )
    if fallback_used:
        aliases = []
        for row in candidates:
            alias = copy.deepcopy(row)
            alias["checkpoint_kind"] = "highest_training_success"
            aliases.append(alias)
        candidates = aliases + candidates

    tied = tied_top_pairs(candidates)
    secondary: list[dict[str, Any]] | None = None
    if len(tied) > 1:
        secondary = []
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        for checkpoint_kind, decoder in tied:
            source_kind = (
                "exact_final"
                if fallback_used and checkpoint_kind == "highest_training_success"
                else checkpoint_kind
            )
            context = _CONTEXTS.get(source_kind)
            if context is None:
                raise RuntimeError(f"missing calibration context for {source_kind}")
            cache_key = (source_kind, decoder)
            if cache_key not in cache:
                cache[cache_key] = _evaluate_secondary_pair(
                    checkpoint_kind=source_kind,
                    deployment_decoder=decoder,
                    context=context,
                )
            row = copy.deepcopy(cache[cache_key])
            row["checkpoint_kind"] = checkpoint_kind
            secondary.append(row)
    receipt = select_deployment(
        candidates,
        secondary,
        secondary_seed_offset=SECONDARY_SEED_OFFSET,
    )
    receipt["training_best_missing_fallback_used"] = fallback_used
    receipt["fallback_reuses_exact_final_calibration"] = fallback_used
    return receipt


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    global _ACTIVE_ALGORITHM
    _CONTEXTS.clear()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--algo" in argv_list:
        _ACTIVE_ALGORITHM = argv_list[argv_list.index("--algo") + 1]
    else:
        _ACTIVE_ALGORITHM = V48_CANDIDATE
    originals = {
        "V47_ALGORITHMS": frozen.V47_ALGORITHMS,
        "V47_FORMAL_ALGORITHMS": frozen.V47_FORMAL_ALGORITHMS,
        "V47_IMPLEMENTATION_IDS": frozen.V47_IMPLEMENTATION_IDS,
        "V47_CANDIDATE": frozen.V47_CANDIDATE,
        "make_model_v4_7": frozen.make_model_v4_7,
        "source_action_repeat_v4_7": frozen.source_action_repeat_v4_7,
        "_method_hyperparameters_v4_7": frozen._method_hyperparameters_v4_7,
        "_validate_formal_unlock_v4_7": frozen._validate_formal_unlock_v4_7,
        "_write_json_v4_7": frozen._write_json_v4_7,
        "_write_json_plain_v4_7": frozen._write_json_plain_v4_7,
        "parent_evaluate": deployment_parent._evaluate_checkpoint_v4_6,
        "parent_select": deployment_parent._select_deployment_adapter,
    }
    frozen.V47_ALGORITHMS = V48_ALGORITHMS
    frozen.V47_FORMAL_ALGORITHMS = V48_FORMAL_ALGORITHMS
    frozen.V47_IMPLEMENTATION_IDS = V48_IMPLEMENTATION_IDS
    frozen.V47_CANDIDATE = V48_CANDIDATE
    frozen.make_model_v4_7 = make_model_v4_8
    frozen.source_action_repeat_v4_7 = source_action_repeat_v4_8
    frozen._method_hyperparameters_v4_7 = _method_hyperparameters_v4_8
    frozen._validate_formal_unlock_v4_7 = _validate_formal_unlock_v4_8
    frozen._write_json_v4_7 = _write_json_v4_8
    frozen._write_json_plain_v4_7 = _write_json_plain_v4_8
    deployment_parent._evaluate_checkpoint_v4_6 = _evaluate_checkpoint_v4_8
    deployment_parent._select_deployment_adapter = _select_deployment_adapter_v4_8
    try:
        return frozen.main(
            argv_list,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        frozen.V47_ALGORITHMS = originals["V47_ALGORITHMS"]
        frozen.V47_FORMAL_ALGORITHMS = originals["V47_FORMAL_ALGORITHMS"]
        frozen.V47_IMPLEMENTATION_IDS = originals["V47_IMPLEMENTATION_IDS"]
        frozen.V47_CANDIDATE = originals["V47_CANDIDATE"]
        frozen.make_model_v4_7 = originals["make_model_v4_7"]
        frozen.source_action_repeat_v4_7 = originals["source_action_repeat_v4_7"]
        frozen._method_hyperparameters_v4_7 = originals[
            "_method_hyperparameters_v4_7"
        ]
        frozen._validate_formal_unlock_v4_7 = originals[
            "_validate_formal_unlock_v4_7"
        ]
        frozen._write_json_v4_7 = originals["_write_json_v4_7"]
        frozen._write_json_plain_v4_7 = originals["_write_json_plain_v4_7"]
        deployment_parent._evaluate_checkpoint_v4_6 = originals["parent_evaluate"]
        deployment_parent._select_deployment_adapter = originals["parent_select"]
        _CONTEXTS.clear()
        _ACTIVE_ALGORITHM = None

if __name__ == "__main__":
    raise SystemExit(main())
