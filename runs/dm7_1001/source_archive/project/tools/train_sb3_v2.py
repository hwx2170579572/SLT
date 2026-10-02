"""Train an isolated topology-temporal v2 ablation with full run receipts."""

from __future__ import annotations

import argparse
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
from configs.sb3_configs_v2 import (
    V2_ALGORITHMS,
    V2_IMPLEMENTATION_IDS,
    make_model_v2,
    source_action_repeat_v2,
)
from tools.paper_evaluation_contract import (
    make_injected_evaluation_env,
    validate_model_environment_spaces,
)
from tools.paper_evaluation_contract_v2 import (
    build_evaluation_provenance_v2,
    callable_name,
)
from tools.action_diagnostics_v2 import evaluate_with_action_diagnostics_v2
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


def argument_parser() -> argparse.ArgumentParser:
    parser = v1_argument_parser()
    parser.description = __doc__
    algorithm_action = next(
        action for action in parser._actions if action.dest == "algo"
    )
    algorithm_action.choices = V2_ALGORITHMS
    parser.set_defaults(algo="topo_v2_soft")
    traffic_action = next(
        action for action in parser._actions if action.dest == "traffic_protocol"
    )
    traffic_action.choices = ("source_all", "frozen_60_20_20")
    parser.set_defaults(traffic_protocol="frozen_60_20_20")
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
        help="Development uses validation; formal testing is contract-gated.",
    )
    parser.add_argument(
        "--slot-balance-coef",
        type=float,
        default=None,
        help="V5 only; default 1e-3. Must be zero/omitted for V2-V4.",
    )
    parser.add_argument("--route-sigma", type=float, default=20.0)
    parser.add_argument("--topology-top-k", type=int, default=8)
    parser.add_argument(
        "--reverse-heading-cosine-min", type=float, default=0.0
    )
    parser.add_argument("--route-bias-init", type=float, default=1.0)
    parser.add_argument("--heading-bias-init", type=float, default=1.0)
    parser.add_argument("--topology-layerscale-init", type=float, default=1e-3)
    parser.add_argument("--goal-layerscale-init", type=float, default=1e-3)
    parser.add_argument(
        "--experiment-contract-sha256",
        default=None,
        help="Optional protocol hash copied into every immutable run receipt.",
    )
    parser.add_argument("--freeze-manifest-sha256", default=None)
    parser.add_argument("--topology-audit-sha256", default=None)
    parser.add_argument("--runtime-environment-sha256", default=None)
    return parser


def _effective_balance_coef(args: argparse.Namespace) -> float:
    if args.algo == "topo_v2_soft":
        return 1e-3 if args.slot_balance_coef is None else float(args.slot_balance_coef)
    if args.slot_balance_coef not in (None, 0.0):
        raise ValueError("Only topo_v2_soft may use --slot-balance-coef > 0")
    return 0.0


def _model_kwargs(args: argparse.Namespace, *, tensorboard_log: str | None, verbose: int) -> dict[str, Any]:
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
        "slot_balance_coef": _effective_balance_coef(args),
        "route_sigma": args.route_sigma,
        "topology_top_k": args.topology_top_k,
        "reverse_heading_cosine_min": args.reverse_heading_cosine_min,
        "route_bias_init": args.route_bias_init,
        "heading_bias_init": args.heading_bias_init,
        "topology_layerscale_init": args.topology_layerscale_init,
        "goal_layerscale_init": args.goal_layerscale_init,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    if args.algo not in V2_ALGORITHMS:
        raise ValueError(f"v2 trainer requires one of {V2_ALGORITHMS}")
    args.action_repeat = source_action_repeat_v2(args.algo, args.action_repeat)
    args.learning_rate = 1e-4 if args.learning_rate is None else args.learning_rate
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative")
    _effective_balance_coef(args)
    if env_factory is None:
        env_factory = _make_env

    total_decisions = _decision_steps(args.max_steps, args.action_repeat)
    warmup_decisions = _decision_steps(args.learning_starts, args.action_repeat)
    checkpoint_decisions = _decision_steps(args.checkpoint_freq, args.action_repeat)
    evaluation_decisions = _decision_steps(args.eval_freq, args.action_repeat)
    source_raw_step_control = True
    model_learning_starts = _model_learning_starts(
        args.learning_starts,
        args.action_repeat,
        source_raw_step_control=True,
    )
    # The config receives the raw threshold because its learner implements the
    # same RawStepControlCallback contract as v1.
    args.learning_starts = model_learning_starts

    run_name = args.model_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__"
        f"{args.algo}__{args.scenario}__seed{args.seed}"
    )
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    arguments_payload = {
        "schema_version": "topology-temporal-v2-run-arguments/v1",
        "implementation_fidelity": {
            "implementation_id": V2_IMPLEMENTATION_IDS[args.algo],
            "source_primary": False,
            "research_extension": True,
            "v2_isolated_files": True,
        },
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "frozen_dependencies": {
            "freeze_manifest_sha256": args.freeze_manifest_sha256,
            "topology_audit_sha256": args.topology_audit_sha256,
            "runtime_environment_sha256": args.runtime_environment_sha256,
        },
        "requested_raw_steps": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_method_hyperparameters": {
            "slot_balance_coef": _effective_balance_coef(args),
            "route_sigma": args.route_sigma,
            "topology_top_k": args.topology_top_k,
            "reverse_heading_cosine_min": args.reverse_heading_cosine_min,
            "route_bias_init": args.route_bias_init,
            "heading_bias_init": args.heading_bias_init,
            "topology_layerscale_init": args.topology_layerscale_init,
            "goal_layerscale_init": args.goal_layerscale_init,
        },
        "effective_sb3_decision_steps": {
            "expected_total_without_partial_repeats": total_decisions,
            "learn_loop_upper_bound": args.max_steps,
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
            model = make_model_v2(
                args.algo,
                model_check_env,
                **_model_kwargs(args, tensorboard_log=None, verbose=0),
            )
            observation, _ = model_check_env.reset(seed=args.seed)
            action, _ = model.predict(observation, deterministic=True)
            if not np.isfinite(np.asarray(action)).all():
                raise FloatingPointError("Model check produced a non-finite action")
            topology_info = getattr(model, "topology_graph_info", None)
            method_metadata = getattr(model, "v2_method_metadata", None)
            if topology_info is not None:
                _write_json(run_dir / "topology_graph.json", topology_info)
            _write_json(run_dir / "method_metadata.json", method_metadata)
            result = {
                "status": "ok",
                "scenario": args.scenario,
                "algorithm": args.algo,
                "model_class": type(model).__name__,
                "feature_extractor_class": type(
                    model.critic.features_extractor
                ).__name__,
                "action_shape": list(np.asarray(action).shape),
                "action_finite": True,
                "topology_graph": topology_info,
                "method_metadata": method_metadata,
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
        model = make_model_v2(
            args.algo,
            monitored_env,
            **_model_kwargs(
                args,
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=1,
            ),
        )
        topology_info = getattr(model, "topology_graph_info", None)
        method_metadata = getattr(model, "v2_method_metadata", None)
        if topology_info is not None:
            _write_json(run_dir / "topology_graph.json", topology_info)
        _write_json(run_dir / "method_metadata.json", method_metadata)

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
                    evaluation_seed_start=args.seed + 10_000,
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
        persisted_diagnostics = model.training_diagnostics()
        _write_json(
            run_dir / "training_diagnostics.json",
            {
                "schema_version": "topology-temporal-v2-training-diagnostics/v1",
                "algorithm": args.algo,
                "scenario": args.scenario,
                "raw_steps": args.max_steps,
                "learner_updates": int(getattr(model, "_n_updates", 0)),
                "statistics": persisted_diagnostics,
            },
        )
        checkpoint_audit = _audit_checkpoints(
            run_dir,
            algorithm=args.algo,
            scenario=args.scenario,
            requested_raw_steps=args.max_steps,
            checkpoint_frequency=args.checkpoint_freq,
            action_repeat=args.action_repeat,
            source_raw_step_control=source_raw_step_control,
        )
        _write_json(run_dir / "checkpoint_audit.json", checkpoint_audit)

        collected_training_raw_steps = int(model._raw_steps_seen)
        model_path = run_dir / "final_model"
        model.save(model_path)
        profile_observation, _ = env.reset(seed=args.seed + 20_000)
        inference_profile = _profile_inference(model, profile_observation)
        performance_profile = {
            "schema_version": "topology-temporal-v2-performance/v1",
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
        }
        _write_json(run_dir / "performance_profile.json", performance_profile)
        if eval_env is not None:
            eval_env.close()

        final_eval_env = make_injected_evaluation_env(env_factory, args)
        try:
            space_contract = validate_model_environment_spaces(model, final_eval_env)
            report, action_diagnostics = evaluate_with_action_diagnostics_v2(
                model,
                final_eval_env,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                trace_path=run_dir / "action_diagnostics_decisions.jsonl",
            )
            summary = report.summary
            evaluation_provenance = (
                build_evaluation_provenance_v2(
                    model=model,
                    env=final_eval_env,
                    report=report,
                    algorithm=args.algo,
                    scenario=args.scenario,
                    traffic_protocol=args.traffic_protocol,
                    evaluation_split=args.evaluation_split,
                    episode_limit_profile=args.episode_limit_profile,
                    evaluation_seed_start=args.seed + 10_000,
                    environment_factory=callable_name(env_factory),
                    space_contract=space_contract,
                )
                if require_paper_evaluation_contract
                else {
                    "contract_version": 1,
                    "validated": True,
                    "paper_contract": False,
                    "environment_factory": callable_name(env_factory),
                    **space_contract,
                }
            )
        finally:
            final_eval_env.close()

        _write_json(run_dir / "final_evaluation.json", summary.to_dict())
        _write_json(run_dir / "action_diagnostics.json", action_diagnostics)
        detailed_payload = {
            "schema_version": "topology-temporal-v2-detailed-evaluation/v1",
            "model": str(model_path.resolve()) + ".zip",
            "algorithm": args.algo,
            "implementation_id": V2_IMPLEMENTATION_IDS[args.algo],
            "method_metadata": method_metadata,
            "scenario": args.scenario,
            "evaluation_seed_start": args.seed + 10_000,
            "trained_raw_steps": int(model._raw_steps_seen),
            "collected_training_raw_steps": collected_training_raw_steps,
            "selected_model_learner_timesteps": int(model.num_timesteps),
            "experiment_contract_sha256": args.experiment_contract_sha256,
            "evaluation_split": args.evaluation_split,
            "evaluation_provenance": evaluation_provenance,
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
                    **summary.to_dict(),
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
