"""Train v4.5 with a sealed train-only deployment checkpoint selector."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from algos.sb3_torch import BestTrainingSuccessCallback, RawStepControlCallback
from configs.sb3_configs_v4_5 import (
    V45_ALGORITHMS,
    V45_IMPLEMENTATION_IDS,
    make_model_v4_5,
    source_action_repeat_v4_5,
)
from tools.action_diagnostics_v4_5 import evaluate_with_action_diagnostics_v4_5
from tools.checkpoint_selector_v4_4 import (
    select_checkpoint,
    sha256,
    verify_checkpoint_zip,
)
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.paper_evaluation_contract_v2 import (
    build_evaluation_provenance_v2,
    callable_name,
)
from tools.train_sb3 import (
    _audit_checkpoints,
    _check_environment,
    _decision_steps,
    _make_env,
    _model_learning_starts,
    _model_parameter_profile,
    _profile_inference,
    _tensorboard_log_directory,
    argument_parser as v1_argument_parser,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _method_hyperparameters(algorithm: str) -> dict[str, Any]:
    common = {
        "reward_unchanged": True,
        "training_unchanged_from_v4_4": True,
        "train_only_checkpoint_selector": True,
        "checkpoint_selector_changed_from_v4_4": False,
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "checkpoint_calibration_episodes": 12,
    }
    if algorithm == "topo_v4_5_confident_actor_fusion":
        return {
            **common,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": [-1, 0, 1],
            "lane_action_mask": True,
            "lane_conditioned_speed": True,
            "critic_lane_one_hot": True,
            "speed_target_entropy": -1.0,
            "lane_entropy_scale": 0.0,
            "deterministic_lane_decoder": (
                "actor_confident_non_keep_else_target_critic"
            ),
            "actor_non_keep_confidence_threshold": 0.90,
            "actor_override_requires_non_keep": True,
            "target_critic_fallback": (
                "keep_tie_aware_argmax_feasible_min_target_twin_q"
            ),
        }
    if algorithm == "temporal_graph_v1_control":
        return {
            **common,
            "actor_encoder_detach": True,
            "slot_balance_coef": 0.0,
            "hybrid_lane_actions": None,
            "lane_action_mask": False,
            "comparator_implementation_unchanged": True,
        }
    raise ValueError(f"unsupported v4.5 algorithm {algorithm!r}")


def _validate_formal_unlock(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with the test split")
        return None
    if args.algo not in V45_ALGORITHMS:
        raise ValueError("formal test permits only the frozen v4.5 method pair")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError("formal test requires experiment and implementation hashes")
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    if not receipt_path.is_file():
        raise ValueError(f"formal test locked: missing receipt {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "topo-scene-v4.5.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"formal test locked: receipt {key!r} does not match")
    if receipt.get("formal_algorithms") != list(V45_ALGORITHMS):
        raise ValueError("formal test locked: receipt method pair drifted")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = v1_argument_parser()
    parser.description = __doc__
    algorithm_action = next(action for action in parser._actions if action.dest == "algo")
    algorithm_action.choices = V45_ALGORITHMS
    parser.set_defaults(algo="topo_v4_5_confident_actor_fusion", eval_freq=0)
    traffic_action = next(
        action for action in parser._actions if action.dest == "traffic_protocol"
    )
    traffic_action.choices = ("source_all", "frozen_60_20_20")
    parser.set_defaults(traffic_protocol="frozen_60_20_20")
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--evaluation-seed-start", type=int, default=None)
    parser.add_argument("--calibration-seed-start", type=int, default=None)
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--experiment-contract-sha256", default=None)
    parser.add_argument("--stage0-results-sha256", default=None)
    parser.add_argument("--attribution-sha256", default=None)
    parser.add_argument("--implementation-freeze-sha256", default=None)
    parser.add_argument("--formal-unlock-receipt", type=Path, default=None)
    parser.add_argument("--tensorboard-log-root", type=Path, default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    return parser


def _model_kwargs(
    args: argparse.Namespace, *, tensorboard_log: str | None, verbose: int
) -> dict[str, Any]:
    return {
        "scenario": args.scenario,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "discount": args.discount,
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "action_repeat": args.action_repeat,
        "seed": args.seed,
        "device": args.device,
        "tensorboard_log": tensorboard_log,
        "verbose": verbose,
        "slot_balance_coef": 0.01,
    }


def _partition_args(args: argparse.Namespace, partition: str) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.evaluation_split = partition
    return copied


def _evaluate_checkpoint(
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
    output = run_dir / "selector" / "cal" / short_kind
    output.mkdir(parents=True, exist_ok=False)
    calibration_args = _partition_args(args, "train")
    calibration_env = env_factory(calibration_args, evaluation=True)
    try:
        calibration_model = model_class.load(
            checkpoint_path,
            env=calibration_env,
            device=args.device,
        )
        space_contract = validate_model_environment_spaces(
            calibration_model, calibration_env
        )
        report, diagnostics = evaluate_with_action_diagnostics_v4_5(
            calibration_model,
            calibration_env,
            episodes=args.calibration_episodes,
            seed=args.calibration_seed_start,
            trace_path=output / "decisions.jsonl",
        )
    finally:
        calibration_env.close()
    summary = report.summary.to_dict()
    detailed = {
        "schema_version": "topo-scene-v4.5.checkpoint-calibration/v1",
        "computed_from_real_run": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "algorithm": args.algo,
        "scenario": args.scenario,
        "training_seed": args.seed,
        "traffic_partition": "train",
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_episodes": args.calibration_episodes,
        "space_contract": space_contract,
        **report.to_dict(),
    }
    _write_json(output / "evaluation.json", summary)
    _write_json(output / "actions.json", diagnostics)
    _write_json(output / "detailed.json", detailed)
    return {
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "traffic_partition": "train",
        "calibration_seed_start": args.calibration_seed_start,
        "summary": summary,
        "episode_records": detailed["episode_records"],
        "calibration_result_sha256": sha256(output / "detailed.json"),
        "calibration_action_diagnostics_sha256": sha256(output / "actions.json"),
        "deterministic_lane_decoder": diagnostics[
            "deterministic_lane_decoder"
        ],
    }


def _seal_selector_then_construct_evaluation_env(
    *,
    selector: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    env_factory: Callable[..., Any],
) -> tuple[Path, Path, str, Any, str]:
    """Seal deployment evidence before the first validation/test env exists."""

    selected_source = Path(selector["selected_checkpoint_path"])
    selected_model = run_dir / "selected_model.zip"
    shutil.copyfile(selected_source, selected_model)
    if sha256(selected_model) != selector["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint copy hash mismatch")
    selector.update(
        {
            "schema_version": "topo-scene-v4.5.checkpoint-selector-receipt/v1",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selector_receipt_precedes_validation_environment": True,
            "selected_model_path": str(selected_model.resolve()),
            "selected_model_sha256": sha256(selected_model),
            "experiment_contract_sha256": args.experiment_contract_sha256,
            "attribution_sha256": args.attribution_sha256,
            "implementation_freeze_sha256": args.implementation_freeze_sha256,
        }
    )
    selector_receipt = run_dir / "selector" / "receipt.json"
    _write_json(selector_receipt, selector)
    selector_receipt_sha = sha256(selector_receipt)

    # Keep this call after the durable receipt write; tests instrument this edge.
    final_eval_env = env_factory(args, evaluation=True)
    constructed_at = datetime.now(timezone.utc).isoformat()
    return (
        selected_model,
        selector_receipt,
        selector_receipt_sha,
        final_eval_env,
        constructed_at,
    )


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
) -> int:
    parser = argument_parser()
    if default_output_dir is not None:
        parser.set_defaults(output_dir=default_output_dir)
    args = parser.parse_args(argv)
    args.action_repeat = source_action_repeat_v4_5(args.algo, args.action_repeat)
    args.learning_rate = 1e-4 if args.learning_rate is None else args.learning_rate
    if args.evaluation_seed_start is None:
        args.evaluation_seed_start = args.seed + 10_000
    if args.calibration_seed_start is None:
        args.calibration_seed_start = args.seed + 30_000
    if (
        args.evaluation_seed_start < 0
        or args.calibration_seed_start < 0
        or args.calibration_episodes <= 0
    ):
        raise ValueError("evaluation and calibration seed contracts are invalid")
    if args.max_steps <= 0 or args.learning_starts < 0:
        raise ValueError("max-steps must be positive and learning-starts non-negative")
    if args.eval_freq != 0:
        raise ValueError(
            "v4.5 forbids during-training validation; --eval-freq must be zero"
        )
    formal_unlock = _validate_formal_unlock(args)
    if env_factory is None:
        env_factory = _make_env

    total_decisions = _decision_steps(args.max_steps, args.action_repeat)
    warmup_decisions = _decision_steps(args.learning_starts, args.action_repeat)
    checkpoint_decisions = _decision_steps(args.checkpoint_freq, args.action_repeat)
    args.learning_starts = _model_learning_starts(
        args.learning_starts,
        args.action_repeat,
        source_raw_step_control=True,
    )
    run_name = args.model_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__"
        f"{args.algo}__{args.scenario}__seed{args.seed}"
    )
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_prefix = args.checkpoint_prefix or args.algo
    if not checkpoint_prefix or Path(checkpoint_prefix).name != checkpoint_prefix:
        raise ValueError("checkpoint-prefix must be a non-empty filename component")
    tensorboard_log_dir = _tensorboard_log_directory(
        run_dir, override_root=args.tensorboard_log_root
    )
    arguments_payload = {
        "schema_version": "topo-scene-v4.5.run-arguments/v1",
        "implementation_fidelity": {
            "implementation_id": V45_IMPLEMENTATION_IDS[args.algo],
            "v4_5_isolated_files": True,
            "single_change": "actor_confident_non_keep_else_target_critic",
            "actor_non_keep_confidence_threshold": 0.90,
            "formal_test_locked_at_creation": formal_unlock is None,
        },
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "stage0_results_sha256": args.stage0_results_sha256,
        "attribution_sha256": args.attribution_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_unlock": formal_unlock,
        "runtime_paths": {
            "tensorboard_log": str(tensorboard_log_dir),
            "tensorboard_short_root_injected": args.tensorboard_log_root is not None,
            "checkpoint_filename_prefix": checkpoint_prefix,
        },
        "requested_raw_steps": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_method_hyperparameters": _method_hyperparameters(args.algo),
        "effective_sb3_decision_steps": {
            "expected_total_without_partial_repeats": total_decisions,
            "learning_starts_decisions": warmup_decisions,
            "model_learning_starts_argument": args.learning_starts,
            "checkpoint_freq": checkpoint_decisions,
            "gradient_steps_per_decision": args.action_repeat,
        },
    }
    _write_json(run_dir / "arguments.json", arguments_payload)

    preflight_env = env_factory(args)
    try:
        _check_environment(preflight_env, args.seed)
    finally:
        preflight_env.close()

    if args.check_only:
        model_check_env = env_factory(args)
        try:
            model = make_model_v4_5(
                args.algo,
                model_check_env,
                **_model_kwargs(args, tensorboard_log=None, verbose=0),
            )
            observation, _ = model_check_env.reset(seed=args.seed)
            action, _ = model.predict(observation, deterministic=True)
            if not np.isfinite(np.asarray(action)).all():
                raise FloatingPointError("v4.5 model check produced a non-finite action")
            result = {
                "status": "ok",
                "scenario": args.scenario,
                "algorithm": args.algo,
                "model_class": type(model).__name__,
                "policy_class": type(model.policy).__name__,
                "action": np.asarray(action).tolist(),
                "action_finite": True,
                "exact_hybrid_lane_code": (
                    float(np.asarray(action).reshape(-1)[1]) in (-1.0, 0.0, 1.0)
                    if args.algo == "topo_v4_5_confident_actor_fusion"
                    else None
                ),
                "method_metadata": getattr(model, "v4_method_metadata", None),
                "checkpoint_selector_constructed_validation_env": False,
            }
            _write_json(run_dir / "check_result.json", result)
        finally:
            model_check_env.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    env = env_factory(args)
    try:
        monitored_env = Monitor(
            env,
            filename=str(run_dir / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            ),
        )
        model = make_model_v4_5(
            args.algo,
            monitored_env,
            **_model_kwargs(
                args, tensorboard_log=str(tensorboard_log_dir), verbose=1
            ),
        )
        model_class = type(model)
        topology_info = getattr(model, "topology_graph_info", None)
        metadata = getattr(model, "v4_method_metadata", None)
        if topology_info is not None:
            _write_json(run_dir / "topology_graph.json", topology_info)
        _write_json(run_dir / "method_metadata.json", metadata)
        parameter_profile = _model_parameter_profile(model)
        model_device = getattr(model, "device", torch.device("cpu"))
        cuda_profiled = getattr(model_device, "type", str(model_device)) == "cuda"
        if cuda_profiled:
            torch.cuda.reset_peak_memory_stats(model_device)

        callbacks = [
            RawStepControlCallback(
                raw_step_budget=args.max_steps,
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_path=run_dir / "checkpoints",
                checkpoint_prefix=checkpoint_prefix,
            ),
            BestTrainingSuccessCallback(run_dir / "best_training_success_model"),
        ]
        started = time.perf_counter()
        model.learn(
            total_timesteps=args.max_steps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        training_wall_seconds = time.perf_counter() - started
        peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated(model_device)) / (1024.0**2)
            if cuda_profiled
            else None
        )
        collected_training_raw_steps = int(model._raw_steps_seen)
        _write_json(
            run_dir / "training_diagnostics.json",
            {
                "schema_version": "topo-scene-v4.5.training-diagnostics/v1",
                "algorithm": args.algo,
                "scenario": args.scenario,
                "raw_steps": args.max_steps,
                "learner_updates": int(getattr(model, "_n_updates", 0)),
                "statistics": model.training_diagnostics(),
            },
        )
        checkpoint_audit = _audit_checkpoints(
            run_dir,
            algorithm=args.algo,
            checkpoint_prefix=checkpoint_prefix,
            scenario=args.scenario,
            requested_raw_steps=args.max_steps,
            checkpoint_frequency=args.checkpoint_freq,
            action_repeat=args.action_repeat,
            source_raw_step_control=True,
        )
        _write_json(run_dir / "checkpoint_audit.json", checkpoint_audit)
        final_path = run_dir / "final_model.zip"
        model.save(run_dir / "final_model")
        verify_checkpoint_zip(final_path)
        profile_observation, _ = env.reset(seed=args.seed + 20_000)
        inference_profile = _profile_inference(model, profile_observation)
        _write_json(
            run_dir / "performance_profile.json",
            {
                "schema_version": "topo-scene-v4.5.performance/v1",
                "algorithm": args.algo,
                "scenario": args.scenario,
                "raw_steps": args.max_steps,
                "learner_updates": int(getattr(model, "_n_updates", 0)),
                "end_to_end_training_wall_seconds": training_wall_seconds,
                "end_to_end_wall_ms_per_learner_update": (
                    training_wall_seconds * 1000.0
                    / max(int(getattr(model, "_n_updates", 0)), 1)
                ),
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "parameters": parameter_profile,
                "inference": inference_profile,
            },
        )
    finally:
        env.close()

    candidate_paths = [("exact_final", final_path)]
    training_best_path = run_dir / "best_training_success_model.zip"
    if training_best_path.is_file():
        candidate_paths.insert(0, ("highest_training_success", training_best_path))
    candidates = [
        _evaluate_checkpoint(
            checkpoint_kind=kind,
            checkpoint_path=path,
            model_class=model_class,
            args=args,
            env_factory=env_factory,
            run_dir=run_dir,
        )
        for kind, path in candidate_paths
    ]
    selector = select_checkpoint(candidates)
    calibration_decoders = {
        str(candidate["deterministic_lane_decoder"]) for candidate in candidates
    }
    if len(calibration_decoders) != 1:
        raise ValueError("checkpoint candidates used different deployment decoders")
    selector["calibration_deterministic_lane_decoder"] = next(
        iter(calibration_decoders)
    )
    selector["calibration_uses_method_own_frozen_decoder"] = True
    (
        selected_model,
        selector_receipt,
        selector_receipt_sha,
        final_eval_env,
        validation_environment_constructed_at_utc,
    ) = _seal_selector_then_construct_evaluation_env(
        selector=selector,
        run_dir=run_dir,
        args=args,
        env_factory=env_factory,
    )
    try:
        selected_model_object = model_class.load(
            selected_model,
            env=final_eval_env,
            device=args.device,
        )
        space_contract = validate_model_environment_spaces(
            selected_model_object, final_eval_env
        )
        report, action_diagnostics = evaluate_with_action_diagnostics_v4_5(
            selected_model_object,
            final_eval_env,
            episodes=args.eval_episodes,
            seed=args.evaluation_seed_start,
            trace_path=run_dir / "action_diagnostics_decisions.jsonl",
        )
        provenance = (
            build_evaluation_provenance_v2(
                model=selected_model_object,
                env=final_eval_env,
                report=report,
                algorithm=args.algo,
                scenario=args.scenario,
                traffic_protocol=args.traffic_protocol,
                evaluation_split=args.evaluation_split,
                episode_limit_profile=args.episode_limit_profile,
                evaluation_seed_start=args.evaluation_seed_start,
                environment_factory=callable_name(env_factory),
                space_contract=space_contract,
            )
            if require_paper_evaluation_contract
            else {
                "validated": True,
                "paper_contract": False,
                "environment_factory": callable_name(env_factory),
                **space_contract,
            }
        )
    finally:
        final_eval_env.close()

    _write_json(run_dir / "final_evaluation.json", report.summary.to_dict())
    _write_json(run_dir / "action_diagnostics.json", action_diagnostics)
    detailed_payload = {
        "schema_version": "topo-scene-v4.5.detailed-evaluation/v1",
        "model": str(selected_model.resolve()),
        "model_sha256": sha256(selected_model),
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selector_receipt": str(selector_receipt.resolve()),
        "selector_receipt_sha256": selector_receipt_sha,
        "selector_receipt_precedes_validation_environment": True,
        "selector_receipt_sealed_at_utc": selector["sealed_at_utc"],
        "validation_environment_constructed_at_utc": (
            validation_environment_constructed_at_utc
        ),
        "algorithm": args.algo,
        "implementation_id": V45_IMPLEMENTATION_IDS[args.algo],
        "method_metadata": metadata,
        "scenario": args.scenario,
        "evaluation_seed_start": args.evaluation_seed_start,
        "trained_raw_steps": collected_training_raw_steps,
        "collected_training_raw_steps": collected_training_raw_steps,
        "selected_model_learner_timesteps": int(selected_model_object.num_timesteps),
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "stage0_results_sha256": args.stage0_results_sha256,
        "attribution_sha256": args.attribution_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "evaluation_split": args.evaluation_split,
        "evaluation_provenance": provenance,
        "completion_time_population_std_ddof": 0,
        **report.to_dict(),
    }
    _write_json(run_dir / "paper_evaluation_detailed.json", detailed_payload)
    print(
        json.dumps(
            {
                "model": str(selected_model),
                "model_sha256": sha256(selected_model),
                "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
                "selector_receipt_sha256": selector_receipt_sha,
                "algorithm": args.algo,
                "requested_raw_steps": args.max_steps,
                "collected_training_raw_steps": collected_training_raw_steps,
                **report.summary.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
