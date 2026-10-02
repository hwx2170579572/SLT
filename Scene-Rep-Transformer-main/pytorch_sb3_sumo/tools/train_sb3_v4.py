"""Train v4 decision-aligned methods with immutable run receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from algos.sb3_torch import (
    BestTrainingSuccessCallback,
    RawStepControlCallback,
    SourceEvaluationCallback,
)
from configs.sb3_configs_v4 import (
    V4_ALGORITHMS,
    V4_IMPLEMENTATION_IDS,
    make_model_v4,
    source_action_repeat_v4,
)
from tools.action_diagnostics_v4 import evaluate_with_action_diagnostics_v4
from tools.paper_evaluation_contract import (
    make_injected_evaluation_env,
    validate_model_environment_spaces,
)
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
    argument_parser as v1_argument_parser,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_hyperparameters(algorithm: str) -> dict[str, Any]:
    common = {
        "reward_unchanged": True,
        "actor_encoder_detach": True,
    }
    if algorithm == "topo_v4_da_hybrid":
        return {
            **common,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": [-1, 0, 1],
            "lane_action_mask": True,
            "lane_conditioned_speed": True,
            "exact_lane_enumeration": True,
            "critic_lane_one_hot": True,
            "target_entropy": -2.0,
        }
    if algorithm == "topo_v4_continuous_ablation":
        return {
            **common,
            "slot_balance_coef": 0.01,
            "hybrid_lane_actions": None,
            "lane_action_mask": False,
            "lane_conditioned_speed": False,
            "exact_lane_enumeration": False,
            "critic_lane_one_hot": False,
            "continuous_lateral_threshold": 1.0 / 3.0,
        }
    if algorithm == "temporal_graph_v1_control":
        return {
            **common,
            "slot_balance_coef": 0.0,
            "hybrid_lane_actions": None,
            "lane_action_mask": False,
            "lane_conditioned_speed": False,
            "exact_lane_enumeration": False,
            "critic_lane_one_hot": False,
            "continuous_lateral_threshold": 1.0 / 3.0,
            "comparator_implementation_unchanged": True,
        }
    raise ValueError(f"unsupported v4 algorithm {algorithm!r}")


def _validate_formal_unlock(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.evaluation_split != "test":
        if args.formal_unlock_receipt is not None:
            raise ValueError("formal unlock receipt may only be used with the test split")
        return None
    if args.algo not in ("temporal_graph_v1_control", "topo_v4_da_hybrid"):
        raise ValueError("the formal test permits only the frozen comparator and candidate")
    if not args.experiment_contract_sha256 or not args.implementation_freeze_sha256:
        raise ValueError(
            "test evaluation requires experiment and implementation freeze hashes"
        )
    if args.formal_unlock_receipt is None:
        raise ValueError("formal test locked: missing promotion gate receipt")
    receipt_path = args.formal_unlock_receipt.resolve()
    if not receipt_path.is_file():
        raise ValueError(f"formal test locked: missing receipt {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("formal test locked: invalid promotion receipt")
    required = {
        "schema_version": "topo-scene-v4.promotion-gate/v1",
        "decision": "pass",
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"formal test locked: promotion receipt {key!r} does not match"
            )
    algorithms = receipt.get("formal_algorithms")
    if algorithms != ["temporal_graph_v1_control", "topo_v4_da_hybrid"]:
        raise ValueError("formal test locked: receipt does not freeze both algorithms")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "promotion_results_sha256": receipt.get("promotion_results_sha256"),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = v1_argument_parser()
    parser.description = __doc__
    algorithm_action = next(action for action in parser._actions if action.dest == "algo")
    algorithm_action.choices = V4_ALGORITHMS
    parser.set_defaults(algo="topo_v4_da_hybrid")
    traffic_action = next(
        action for action in parser._actions if action.dest == "traffic_protocol"
    )
    traffic_action.choices = ("source_all", "frozen_60_20_20")
    parser.set_defaults(traffic_protocol="frozen_60_20_20")
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--experiment-contract-sha256", default=None)
    parser.add_argument("--stage0-results-sha256", default=None)
    parser.add_argument("--implementation-freeze-sha256", default=None)
    parser.add_argument("--formal-unlock-receipt", type=Path, default=None)
    parser.add_argument(
        "--evaluation-seed-start",
        type=int,
        default=None,
        help=(
            "First evaluation episode seed. The v4 orchestrator assigns "
            "non-overlapping blocks while pairing methods exactly."
        ),
    )
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
    args.action_repeat = source_action_repeat_v4(args.algo, args.action_repeat)
    args.learning_rate = 1e-4 if args.learning_rate is None else args.learning_rate
    if args.evaluation_seed_start is None:
        args.evaluation_seed_start = args.seed + 10_000
    if args.evaluation_seed_start < 0:
        raise ValueError("evaluation-seed-start must be non-negative")
    if args.max_steps <= 0 or args.learning_starts < 0:
        raise ValueError("max-steps must be positive and learning-starts non-negative")
    formal_unlock = _validate_formal_unlock(args)
    if env_factory is None:
        env_factory = _make_env

    total_decisions = _decision_steps(args.max_steps, args.action_repeat)
    warmup_decisions = _decision_steps(args.learning_starts, args.action_repeat)
    checkpoint_decisions = _decision_steps(args.checkpoint_freq, args.action_repeat)
    evaluation_decisions = _decision_steps(args.eval_freq, args.action_repeat)
    model_learning_starts = _model_learning_starts(
        args.learning_starts,
        args.action_repeat,
        source_raw_step_control=True,
    )
    args.learning_starts = model_learning_starts

    run_name = args.model_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__"
        f"{args.algo}__{args.scenario}__seed{args.seed}"
    )
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    arguments_payload = {
        "schema_version": "topo-scene-v4.run-arguments/v1",
        "implementation_fidelity": {
            "implementation_id": V4_IMPLEMENTATION_IDS[args.algo],
            "research_extension": args.algo == "topo_v4_da_hybrid",
            "v4_isolated_files": True,
            "formal_test_locked_at_creation": formal_unlock is None,
        },
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "stage0_results_sha256": args.stage0_results_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "formal_unlock": formal_unlock,
        "requested_raw_steps": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_method_hyperparameters": _method_hyperparameters(args.algo),
        "effective_sb3_decision_steps": {
            "expected_total_without_partial_repeats": total_decisions,
            "learning_starts_decisions": warmup_decisions,
            "model_learning_starts_argument": model_learning_starts,
            "checkpoint_freq": checkpoint_decisions,
            "eval_freq": evaluation_decisions,
            "gradient_steps_per_decision": args.action_repeat,
        },
    }
    _write_json(run_dir / "arguments.json", arguments_payload)

    validation_env = env_factory(args)
    try:
        _check_environment(validation_env, args.seed)
    finally:
        validation_env.close()

    if args.check_only:
        model_check_env = env_factory(args)
        try:
            model = make_model_v4(
                args.algo,
                model_check_env,
                **_model_kwargs(args, tensorboard_log=None, verbose=0),
            )
            observation, _ = model_check_env.reset(seed=args.seed)
            action, _ = model.predict(observation, deterministic=True)
            if not np.isfinite(np.asarray(action)).all():
                raise FloatingPointError("v4 model check produced a non-finite action")
            if args.algo == "topo_v4_da_hybrid" and float(action[1]) not in (-1.0, 0.0, 1.0):
                raise ValueError("hybrid model did not emit an exact lane code")
            topology_info = getattr(model, "topology_graph_info", None)
            metadata = getattr(model, "v4_method_metadata", None)
            if topology_info is not None:
                _write_json(run_dir / "topology_graph.json", topology_info)
            _write_json(run_dir / "method_metadata.json", metadata)
            result = {
                "status": "ok",
                "scenario": args.scenario,
                "algorithm": args.algo,
                "model_class": type(model).__name__,
                "policy_class": type(model.policy).__name__,
                "feature_extractor_class": type(model.critic.features_extractor).__name__,
                "action": np.asarray(action).tolist(),
                "action_finite": True,
                "exact_hybrid_lane_code": (
                    float(action[1]) in (-1.0, 0.0, 1.0)
                    if args.algo == "topo_v4_da_hybrid"
                    else None
                ),
                "method_metadata": metadata,
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
        model = make_model_v4(
            args.algo,
            monitored_env,
            **_model_kwargs(
                args, tensorboard_log=str(run_dir / "tensorboard"), verbose=1
            ),
        )
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

        callbacks: list[Any] = [
            RawStepControlCallback(
                raw_step_budget=args.max_steps,
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_path=run_dir / "checkpoints",
                checkpoint_prefix=args.algo,
            ),
            BestTrainingSuccessCallback(run_dir / "best_training_success_model"),
        ]
        eval_env = None
        if evaluation_decisions > 0:
            eval_env = Monitor(make_injected_evaluation_env(env_factory, args))
            callbacks.append(
                SourceEvaluationCallback(
                    eval_env,
                    best_model_save_path=str(run_dir / "best"),
                    log_path=str(run_dir / "evaluation"),
                    eval_freq=evaluation_decisions,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                    evaluation_seed_start=args.evaluation_seed_start,
                )
            )

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
        _write_json(
            run_dir / "training_diagnostics.json",
            {
                "schema_version": "topo-scene-v4.training-diagnostics/v1",
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
            scenario=args.scenario,
            requested_raw_steps=args.max_steps,
            checkpoint_frequency=args.checkpoint_freq,
            action_repeat=args.action_repeat,
            source_raw_step_control=True,
        )
        _write_json(run_dir / "checkpoint_audit.json", checkpoint_audit)

        collected_training_raw_steps = int(model._raw_steps_seen)
        model_path = run_dir / "final_model"
        model.save(model_path)
        profile_observation, _ = env.reset(seed=args.seed + 20_000)
        inference_profile = _profile_inference(model, profile_observation)
        _write_json(
            run_dir / "performance_profile.json",
            {
                "schema_version": "topo-scene-v4.performance/v1",
                "algorithm": args.algo,
                "scenario": args.scenario,
                "raw_steps": args.max_steps,
                "learner_updates": int(getattr(model, "_n_updates", 0)),
                "end_to_end_training_wall_seconds": training_wall_seconds,
                "end_to_end_wall_ms_per_learner_update": (
                    training_wall_seconds
                    * 1000.0
                    / max(int(getattr(model, "_n_updates", 0)), 1)
                ),
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "parameters": parameter_profile,
                "inference": inference_profile,
            },
        )
        if eval_env is not None:
            eval_env.close()

        final_eval_env = make_injected_evaluation_env(env_factory, args)
        try:
            space_contract = validate_model_environment_spaces(model, final_eval_env)
            report, action_diagnostics = evaluate_with_action_diagnostics_v4(
                model,
                final_eval_env,
                episodes=args.eval_episodes,
                seed=args.evaluation_seed_start,
                trace_path=run_dir / "action_diagnostics_decisions.jsonl",
            )
            provenance = (
                build_evaluation_provenance_v2(
                    model=model,
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
            "schema_version": "topo-scene-v4.detailed-evaluation/v1",
            "model": str(model_path.resolve()) + ".zip",
            "algorithm": args.algo,
            "implementation_id": V4_IMPLEMENTATION_IDS[args.algo],
            "method_metadata": metadata,
            "scenario": args.scenario,
            "evaluation_seed_start": args.evaluation_seed_start,
            "trained_raw_steps": int(model._raw_steps_seen),
            "collected_training_raw_steps": collected_training_raw_steps,
            "selected_model_learner_timesteps": int(model.num_timesteps),
            "experiment_contract_sha256": args.experiment_contract_sha256,
            "stage0_results_sha256": args.stage0_results_sha256,
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
                    "model": str(model_path) + ".zip",
                    "algorithm": args.algo,
                    "requested_raw_steps": args.max_steps,
                    "selected_model_sb3_timesteps": int(model.num_timesteps),
                    "collected_training_raw_steps": collected_training_raw_steps,
                    **report.summary.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
