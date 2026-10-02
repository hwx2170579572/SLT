"""Train unchanged v4.5 weights and seal a v4.6 checkpoint/decoder deployment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_6 import (
    V46_ALGORITHMS,
    V46_IMPLEMENTATION_IDS,
    make_model_v4_6,
    source_action_repeat_v4_6,
)
from tools import train_sb3_v4_5 as base
from tools.action_diagnostics_v4_6 import (
    decoder_integrity_passed,
    decoder_integrity_summary,
    evaluate_with_action_diagnostics_v4_6,
    load_model_for_deployment,
    policy_class_for_decoder,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    CANDIDATE_DECODERS,
    PARENT_DECODER,
    select_deployment,
)
from tools.checkpoint_selector_v4_4 import sha256, verify_checkpoint_zip
from tools.paper_evaluation_contract import validate_model_environment_spaces


_SELECTED_DEPLOYMENT: dict[str, Any] | None = None


def _write_json_plain(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_json_v4_6(path: Path, payload: Any) -> None:
    """Version parent workflow envelopes without mutating frozen v4.5 code."""

    value = copy.deepcopy(payload)
    if isinstance(value, dict):
        schema = value.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.5."):
            value["schema_version"] = schema.replace(
                "topo-scene-v4.5.", "topo-scene-v4.6.", 1
            )
        if path.name == "arguments.json":
            fidelity = dict(value.get("implementation_fidelity", {}))
            fidelity.pop("v4_5_isolated_files", None)
            algorithm = value.get("requested_raw_steps", {}).get("algo")
            fidelity.update(
                {
                    "implementation_id": V46_IMPLEMENTATION_IDS[algorithm],
                    "v4_6_isolated_files": True,
                    "single_change": "train_only_joint_checkpoint_decoder_selector",
                    "checkpoint_decoder_candidate_pairs": (
                        4
                        if algorithm
                        == "topo_v4_6_joint_checkpoint_decoder_selector"
                        else 2
                    ),
                    "decoder_definitions_changed": False,
                    "fusion_threshold_changed": False,
                }
            )
            value["implementation_fidelity"] = fidelity
        if path.name == "check_result.json":
            algorithm = value.get("algorithm")
            if algorithm == "topo_v4_6_joint_checkpoint_decoder_selector":
                action = np.asarray(value.get("action", [])).reshape(-1)
                value["exact_hybrid_lane_code"] = bool(
                    len(action) >= 2 and float(action[1]) in (-1.0, 0.0, 1.0)
                )
                value["checkpoint_decoder_candidate_pairs"] = 4
        if path.name == "paper_evaluation_detailed.json" and _SELECTED_DEPLOYMENT:
            value.update(
                {
                    "selected_deployment_decoder": _SELECTED_DEPLOYMENT[
                        "selected_deployment_decoder"
                    ],
                    "selected_source_checkpoint_sha256": _SELECTED_DEPLOYMENT[
                        "selected_checkpoint_sha256"
                    ],
                    "selected_model_policy_class": _SELECTED_DEPLOYMENT[
                        "selected_model_policy_class"
                    ],
                    "selected_model_parameter_state_sha256": _SELECTED_DEPLOYMENT[
                        "selected_model_parameter_state_sha256"
                    ],
                }
            )
    _write_json_plain(path, value)


def _method_hyperparameters_v4_6(algorithm: str) -> dict[str, Any]:
    common = {
        "reward_unchanged": True,
        "training_unchanged_from_v4_5": True,
        "train_only_deployment_selector": True,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_calibration_episodes": 12,
        "checkpoint_calibration_partition": "train",
    }
    if algorithm == "topo_v4_6_joint_checkpoint_decoder_selector":
        return {
            **common,
            "single_change": "train_only_joint_checkpoint_decoder_selector",
            "deployment_decoder_candidates": ["target_critic", "fusion_0_90"],
            "checkpoint_decoder_candidate_pairs": 4,
            "identical_unique_traffic_block_for_all_pairs": True,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": [-1, 0, 1],
            "lane_action_mask": True,
            "lane_conditioned_speed": True,
            "critic_lane_one_hot": True,
            "speed_target_entropy": -1.0,
            "lane_entropy_scale": 0.0,
            "actor_non_keep_confidence_threshold": 0.90,
            "decoder_definitions_changed": False,
        }
    if algorithm == "temporal_graph_v1_control":
        return {
            **common,
            "single_change": "none_parent_control",
            "deployment_decoder_candidates": [PARENT_DECODER],
            "checkpoint_decoder_candidate_pairs": 2,
            "comparator_implementation_unchanged": True,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.0,
        }
    raise ValueError(f"unsupported v4.6 algorithm {algorithm!r}")


def _validate_formal_unlock_v4_6(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with the test split")
        return None
    if args.algo not in V46_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.6 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.6.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal test locked: receipt {key!r} does not match")
    if receipt.get("formal_algorithms") != list(V46_ALGORITHMS):
        raise ValueError("formal test locked: receipt method pair drifted")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def _calibration_decoders(algorithm: str) -> tuple[str, ...]:
    return (
        CANDIDATE_DECODERS
        if algorithm == "topo_v4_6_joint_checkpoint_decoder_selector"
        else (PARENT_DECODER,)
    )


def _evaluate_checkpoint_v4_6(
    *,
    checkpoint_kind: str,
    checkpoint_path: Path,
    model_class: type,
    args: argparse.Namespace,
    env_factory: Callable[..., Any],
    run_dir: Path,
) -> dict[str, Any]:
    short_kind = {
        "highest_training_success": "best",
        "exact_final": "final",
    }[checkpoint_kind]
    checkpoint_output = run_dir / "selector" / "cal" / short_kind
    checkpoint_output.mkdir(parents=True, exist_ok=False)
    pairs: list[dict[str, Any]] = []
    for decoder in _calibration_decoders(args.algo):
        output = checkpoint_output / decoder
        output.mkdir(parents=True, exist_ok=False)
        calibration_args = base._partition_args(args, "train")
        calibration_env = env_factory(calibration_args, evaluation=True)
        try:
            calibration_model = load_model_for_deployment(
                model_class,
                checkpoint_path,
                decoder=decoder,
                env=calibration_env,
                device=args.device,
            )
            space_contract = validate_model_environment_spaces(
                calibration_model, calibration_env
            )
            report, diagnostics = evaluate_with_action_diagnostics_v4_6(
                calibration_model,
                calibration_env,
                deployment_decoder=decoder,
                episodes=args.calibration_episodes,
                seed=args.calibration_seed_start,
                trace_path=output / "decisions.jsonl",
            )
        finally:
            calibration_env.close()
        summary = report.summary.to_dict()
        integrity = decoder_integrity_summary(diagnostics)
        integrity_passed = decoder_integrity_passed(diagnostics)
        if not integrity_passed:
            raise ValueError(
                f"calibration decoder integrity failed: {checkpoint_kind} x {decoder}"
            )
        detailed = {
            "schema_version": "topo-scene-v4.6.checkpoint-decoder-calibration/v1",
            "computed_from_real_run": True,
            "fabricated_values": False,
            "formal_test_accessed": False,
            "checkpoint_kind": checkpoint_kind,
            "deployment_decoder": decoder,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "algorithm": args.algo,
            "scenario": args.scenario,
            "training_seed": args.seed,
            "traffic_partition": "train",
            "calibration_seed_start": args.calibration_seed_start,
            "calibration_episodes": args.calibration_episodes,
            "space_contract": space_contract,
            "decoder_integrity": integrity,
            "decoder_integrity_passed": True,
            **report.to_dict(),
        }
        _write_json_plain(output / "evaluation.json", summary)
        _write_json_plain(output / "actions.json", diagnostics)
        _write_json_plain(output / "detailed.json", detailed)
        pairs.append(
            {
                "checkpoint_kind": checkpoint_kind,
                "deployment_decoder": decoder,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256(checkpoint_path),
                "traffic_partition": "train",
                "calibration_seed_start": args.calibration_seed_start,
                "summary": summary,
                "episode_records": detailed["episode_records"],
                "calibration_result_sha256": sha256(output / "detailed.json"),
                "calibration_action_diagnostics_sha256": sha256(
                    output / "actions.json"
                ),
                "decoder_integrity": integrity,
                "decoder_integrity_passed": True,
            }
        )
    return {
        "checkpoint_kind": checkpoint_kind,
        "deterministic_lane_decoder": (
            "joint_checkpoint_decoder_selector_v4_6"
            if len(pairs) == 2
            else PARENT_DECODER
        ),
        "deployment_candidates": pairs,
    }


def _select_deployment_adapter(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        pair
        for checkpoint in checkpoints
        for pair in checkpoint["deployment_candidates"]
    ]
    fallback_used = not any(
        row["checkpoint_kind"] == "highest_training_success" for row in candidates
    )
    if fallback_used:
        # The parent protocol defines a missing training-best snapshot as an
        # exact-final fallback. Reuse the already evaluated exact-final evidence
        # under the missing label so the Cartesian selector remains explicit.
        aliases = []
        for row in candidates:
            alias = copy.deepcopy(row)
            alias["checkpoint_kind"] = "highest_training_success"
            aliases.append(alias)
        candidates = aliases + candidates
    receipt = select_deployment(candidates)
    receipt["training_best_missing_fallback_used"] = fallback_used
    receipt["fallback_reuses_exact_final_calibration"] = fallback_used
    return receipt


def _policy_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _seal_selector_then_construct_evaluation_env_v4_6(
    *,
    selector: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    env_factory: Callable[..., Any],
):
    """Bind decoder class into selected_model before validation exists."""

    global _SELECTED_DEPLOYMENT
    selected_source = Path(selector["selected_checkpoint_path"])
    selected_decoder = str(selector["selected_deployment_decoder"])
    model_class = None
    # The selected source stores its algorithm class in the SB3 archive. Loading
    # through the training class is not needed here; SB3's classmethod is reached
    # through the concrete class captured from the most recent calibration model.
    # Store it on the selector adapter at evaluation time.
    model_class = _seal_selector_then_construct_evaluation_env_v4_6.model_class
    selected_object = load_model_for_deployment(
        model_class,
        selected_source,
        decoder=selected_decoder,
        env=None,
        device=args.device,
    )
    source_state_sha = _policy_state_sha256(selected_object)
    selected_model = run_dir / "selected_model.zip"
    selected_object.save(run_dir / "selected_model")
    verify_checkpoint_zip(selected_model)
    reloaded = model_class.load(selected_model, env=None, device=args.device)
    deployed_state_sha = _policy_state_sha256(reloaded)
    if deployed_state_sha != source_state_sha:
        raise ValueError("selected deployment reserialization changed policy tensors")
    expected_policy = policy_class_for_decoder(selected_decoder)
    if expected_policy is not None and not isinstance(reloaded.policy, expected_policy):
        raise ValueError("selected model did not persist the sealed decoder policy class")

    selector.update(
        {
            "schema_version": "topo-scene-v4.6.checkpoint-decoder-selector-receipt/v1",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selector_receipt_precedes_validation_environment": True,
            "selected_source_checkpoint_path": str(selected_source.resolve()),
            "selected_source_checkpoint_sha256": selector[
                "selected_checkpoint_sha256"
            ],
            "selected_model_path": str(selected_model.resolve()),
            "selected_model_sha256": sha256(selected_model),
            "selected_model_materialization": "policy_class_bound_reserialization",
            "selected_model_policy_class": type(reloaded.policy).__name__,
            "source_policy_parameter_state_sha256": source_state_sha,
            "selected_model_parameter_state_sha256": deployed_state_sha,
            "policy_parameter_state_preserved": True,
            "experiment_contract_sha256": args.experiment_contract_sha256,
            "attribution_sha256": args.attribution_sha256,
            "implementation_freeze_sha256": args.implementation_freeze_sha256,
        }
    )
    receipt = run_dir / "selector" / "receipt.json"
    _write_json_plain(receipt, selector)
    receipt_sha = sha256(receipt)
    _SELECTED_DEPLOYMENT = selector

    # This remains deliberately after the durable selector/deployment receipt.
    final_eval_env = env_factory(args, evaluation=True)
    constructed_at = datetime.now(timezone.utc).isoformat()
    return selected_model, receipt, receipt_sha, final_eval_env, constructed_at


_seal_selector_then_construct_evaluation_env_v4_6.model_class = None


def _evaluate_selected_v4_6(
    model: Any,
    env: Any,
    *,
    episodes: int,
    seed: int,
    trace_path: Path,
):
    if _SELECTED_DEPLOYMENT is None:
        raise RuntimeError("selected deployment receipt was not sealed")
    decoder = str(_SELECTED_DEPLOYMENT["selected_deployment_decoder"])
    expected_policy = policy_class_for_decoder(decoder)
    if expected_policy is not None and not isinstance(model.policy, expected_policy):
        raise ValueError("validation model policy class disagrees with sealed decoder")
    return evaluate_with_action_diagnostics_v4_6(
        model,
        env,
        deployment_decoder=decoder,
        episodes=episodes,
        seed=seed,
        trace_path=trace_path,
    )


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    global _SELECTED_DEPLOYMENT
    _SELECTED_DEPLOYMENT = None
    originals = {
        "V45_ALGORITHMS": base.V45_ALGORITHMS,
        "V45_IMPLEMENTATION_IDS": base.V45_IMPLEMENTATION_IDS,
        "make_model_v4_5": base.make_model_v4_5,
        "source_action_repeat_v4_5": base.source_action_repeat_v4_5,
        "_method_hyperparameters": base._method_hyperparameters,
        "_validate_formal_unlock": base._validate_formal_unlock,
        "_evaluate_checkpoint": base._evaluate_checkpoint,
        "select_checkpoint": base.select_checkpoint,
        "_seal_selector_then_construct_evaluation_env": (
            base._seal_selector_then_construct_evaluation_env
        ),
        "evaluate_with_action_diagnostics_v4_5": (
            base.evaluate_with_action_diagnostics_v4_5
        ),
        "_write_json": base._write_json,
    }
    base.V45_ALGORITHMS = V46_ALGORITHMS
    base.V45_IMPLEMENTATION_IDS = V46_IMPLEMENTATION_IDS
    base.make_model_v4_5 = make_model_v4_6
    base.source_action_repeat_v4_5 = source_action_repeat_v4_6
    base._method_hyperparameters = _method_hyperparameters_v4_6
    base._validate_formal_unlock = _validate_formal_unlock_v4_6
    base._evaluate_checkpoint = _evaluate_checkpoint_v4_6
    base.select_checkpoint = _select_deployment_adapter
    base._seal_selector_then_construct_evaluation_env = (
        _seal_selector_then_construct_evaluation_env_v4_6
    )
    base.evaluate_with_action_diagnostics_v4_5 = _evaluate_selected_v4_6
    base._write_json = _write_json_v4_6

    # Capture the concrete model class without changing the parent workflow's
    # call signature. The calibration wrapper receives it first.
    original_evaluate = base._evaluate_checkpoint

    def evaluate_and_capture(**kwargs: Any):
        _seal_selector_then_construct_evaluation_env_v4_6.model_class = kwargs[
            "model_class"
        ]
        return original_evaluate(**kwargs)

    base._evaluate_checkpoint = evaluate_and_capture
    try:
        return base.main(
            argv,
            env_factory=env_factory,
            default_output_dir=default_output_dir,
            require_paper_evaluation_contract=require_paper_evaluation_contract,
        )
    finally:
        for name, value in originals.items():
            setattr(base, name, value)
        _seal_selector_then_construct_evaluation_env_v4_6.model_class = None


if __name__ == "__main__":
    raise SystemExit(main())
