"""Train the PyTorch/SB3 implementation in a SUMO scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from algos.sb3_torch import (
    BestTrainingSuccessCallback,
    RawStepControlCallback,
    SourceEvaluationCallback,
    evaluate_model_detailed,
)
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from configs.sb3_configs import make_model, source_action_repeat
from envs.sumo.scenario_registry import available_scenarios
from envs.sumo.sumo_env import EGO_CONTROL_PROFILES, SumoSceneEnv
from tools.paper_evaluation_contract import (
    build_evaluation_provenance,
    callable_name,
    make_injected_evaluation_env,
    validate_model_environment_spaces,
)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=available_scenarios(), default="left_turn")
    parser.add_argument(
        "--algo",
        choices=(
            "scene_rep",
            "mst",
            "topo_scene",
            "topo_scene_balanced",
            "temporal_graph",
            "sac",
            "ppo",
        ),
        default="scene_rep",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1_000_000, help="Raw SUMO simulation steps"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Algorithm source default: 1e-4 for SAC/MST, 5e-4 for PPO",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--learning-starts", type=int, default=5000, help="Warm-up in raw SUMO steps"
    )
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--path-length", type=int, default=10)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help="Source default: 3 for off-policy algorithms and 1 for PPO",
    )
    parser.add_argument(
        "--ego-control-profile",
        choices=EGO_CONTROL_PROFILES,
        default="direct",
        help="Direct TraCI control or a curve/acceleration-limited SMARTS Ackermann proxy",
    )
    parser.add_argument(
        "--traffic-protocol",
        choices=("source_all", "frozen_80_20"),
        default="source_all",
        help="Paper wrapper only: reuse all released traffic or reserve every fifth XML for testing",
    )
    parser.add_argument(
        "--episode-limit-profile",
        choices=("source", "paper"),
        default="source",
        help="Paper wrapper only: released tools/test.py limits or paper Table VI limits",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_sb3_sumo")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run Gymnasium/SUMO validation and a short random rollout without training",
    )
    return parser


def _make_env(args: argparse.Namespace, *, evaluation: bool = False) -> SumoSceneEnv:
    return SumoSceneEnv(
        scenario=args.scenario,
        history_steps=args.history_steps,
        neighbors=args.neighbors,
        path_length=args.path_length,
        action_repeat=args.action_repeat,
        reward_discount=args.discount,
        ego_control_profile=args.ego_control_profile,
        include_state_lstm=args.algo == "sac",
        state_lstm_only=args.algo == "sac",
        render_mode="human" if args.gui and not evaluation else None,
    )


def _check_environment(env: SumoSceneEnv, seed: int) -> None:
    check_env(env, warn=True, skip_render_check=True)
    observation, _ = env.reset(seed=seed)
    for _ in range(8):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, _ = env.reset(seed=seed + 1)
    assert env.observation_space.contains(observation)


def _decision_steps(raw_steps: int, action_repeat: int) -> int:
    if raw_steps < 0:
        raise ValueError("Raw step counts must be non-negative")
    if action_repeat <= 0:
        raise ValueError("--action-repeat must be positive")
    return int(math.ceil(raw_steps / action_repeat))


def _source_ppo_collected_steps(raw_steps: int, horizon: int = 512) -> int:
    """Released on-policy runners finish the horizon that crosses max_steps."""

    if raw_steps <= 0:
        raise ValueError("Raw step count must be positive")
    if horizon <= 0:
        raise ValueError("PPO horizon must be positive")
    return int(math.ceil(raw_steps / horizon) * horizon)


def _source_ppo_test_checkpoint_step(
    requested_raw_steps: int, checkpoint_frequency: int
) -> int:
    """Latest checkpoint that the released test entry point would restore."""

    if requested_raw_steps <= 0:
        raise ValueError("Raw step count must be positive")
    if checkpoint_frequency <= 0:
        raise ValueError("PPO source-equivalent testing requires checkpoints")
    step = (requested_raw_steps // checkpoint_frequency) * checkpoint_frequency
    if step <= 0:
        raise ValueError("No PPO checkpoint exists at or below the requested budget")
    return step


def _model_learning_starts(
    raw_steps: int, action_repeat: int, *, source_raw_step_control: bool
) -> int:
    """Return the clock expected by the selected learner.

    Source-backed SAC variants receive intervals from RawStepControlCallback
    and therefore compare against the original raw-step threshold. PPO uses
    its raw-step rollout counter directly.
    """

    return (
        int(raw_steps)
        if source_raw_step_control
        else _decision_steps(raw_steps, action_repeat)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _tensorboard_log_directory(
    run_dir: Path, override_root: Path | None = None
) -> Path:
    """Return a stable TensorBoard directory without repeating long run IDs.

    The default preserves the historical layout.  Versioned orchestrators may
    inject a short root on Windows; the digest keeps jobs distinct while
    avoiding the legacy MAX_PATH failure seen in long comparison run names.
    """

    if override_root is None:
        return run_dir / "tensorboard"
    token = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    return override_root.resolve() / token


def _model_parameter_profile(model: Any) -> dict[str, int]:
    policy_parameters = list(model.policy.parameters())
    trainable = sum(
        parameter.numel() for parameter in policy_parameters if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in policy_parameters)
    critic_extractor = getattr(getattr(model, "critic", None), "features_extractor", None)
    extractor_total = (
        sum(parameter.numel() for parameter in critic_extractor.parameters())
        if critic_extractor is not None
        else 0
    )
    representation = getattr(model, "representation", None)
    representation_total = (
        sum(parameter.numel() for parameter in representation.parameters())
        if representation is not None
        else 0
    )
    return {
        "policy_total": int(total),
        "policy_trainable": int(trainable),
        "critic_feature_extractor": int(extractor_total),
        "representation_objective": int(representation_total),
        "model_plus_representation": int(total + representation_total),
    }


def _profile_inference(
    model: Any,
    observation: Any,
    *,
    warmup: int = 10,
    repetitions: int = 100,
) -> dict[str, float | int | str]:
    if repetitions <= 0 or warmup < 0:
        raise ValueError("Invalid inference profiling repetitions")
    device = getattr(model, "device", torch.device("cpu"))
    with source_evaluation_augmentation(model):
        for _ in range(warmup):
            model.predict(observation, deterministic=True)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repetitions):
            model.predict(observation, deterministic=True)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000.0 / repetitions
    return {
        "device": str(device),
        "warmup_calls": warmup,
        "measured_calls": repetitions,
        "total_seconds": elapsed,
        "mean_milliseconds_per_action": milliseconds,
        "actions_per_second": 1000.0 / max(milliseconds, 1e-12),
    }


def _audit_checkpoints(
    run_dir: Path,
    *,
    algorithm: str,
    checkpoint_prefix: str | None = None,
    scenario: str,
    requested_raw_steps: int,
    checkpoint_frequency: int,
    action_repeat: int,
    source_raw_step_control: bool,
) -> dict[str, object]:
    """Verify every periodic model from its SB3 ZIP metadata and CRC."""

    expected_raw_steps = (
        list(
            range(
                checkpoint_frequency,
                requested_raw_steps + 1,
                checkpoint_frequency,
            )
        )
        if checkpoint_frequency > 0
        else []
    )
    clock_field = "_raw_steps_seen" if source_raw_step_control else "num_timesteps"
    filename_prefix = checkpoint_prefix or algorithm
    rows: list[dict[str, object]] = []
    for raw_step in expected_raw_steps:
        expected_clock = (
            raw_step
            if source_raw_step_control
            else _decision_steps(raw_step, action_repeat)
        )
        filename = (
            f"{filename_prefix}_raw_{raw_step}_steps.zip"
            if source_raw_step_control
            else f"{filename_prefix}_{expected_clock}_steps.zip"
        )
        path = run_dir / "checkpoints" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected checkpoint was not saved: {path}")
        with ZipFile(path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(
                    f"Checkpoint CRC failure in {path}: member {bad_member}"
                )
            metadata = json.loads(archive.read("data").decode("utf-8"))
            member_count = len(archive.namelist())
        recorded_clock = int(metadata[clock_field])
        if recorded_clock != expected_clock:
            raise RuntimeError(
                f"Checkpoint {path} records {clock_field}={recorded_clock}, "
                f"expected {expected_clock}"
            )
        rows.append(
            {
                "raw_step": raw_step,
                "path": str(path.relative_to(run_dir)),
                "clock_field": clock_field,
                "expected_clock": expected_clock,
                "recorded_clock": recorded_clock,
                "learner_updates": int(metadata.get("_n_updates", 0)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "zip_crc_ok": True,
                "zip_member_count": member_count,
            }
        )
    return {
        "algorithm": algorithm,
        "checkpoint_filename_prefix": filename_prefix,
        "scenario": scenario,
        "requested_raw_steps": requested_raw_steps,
        "checkpoint_frequency_raw_steps": checkpoint_frequency,
        "source_raw_step_control": source_raw_step_control,
        "clock_field": clock_field,
        "expected_checkpoint_count": len(expected_raw_steps),
        "checkpoints": rows,
    }


def main(
    argv: list[str] | None = None,
    *,
    env_factory: Callable[..., Any] | None = None,
    default_output_dir: Path | None = None,
    require_paper_evaluation_contract: bool = False,
    tensorboard_log_root: Path | None = None,
) -> int:
    parser = argument_parser()
    if default_output_dir is not None:
        parser.set_defaults(output_dir=default_output_dir)
    args = parser.parse_args(argv)
    args.action_repeat = source_action_repeat(args.algo, args.action_repeat)
    if env_factory is None:
        if args.algo == "ppo":
            raise ValueError(
                "Source PPO requires the released RGB/CARLA observation wrapper; "
                "use tools/train_paper_sb3_sumo.py"
            )
        env_factory = _make_env
    if args.learning_rate is None:
        args.learning_rate = 5e-4 if args.algo == "ppo" else 1e-4
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative")
    total_decisions = _decision_steps(args.max_steps, args.action_repeat)
    warmup_decisions = _decision_steps(args.learning_starts, args.action_repeat)
    checkpoint_decisions = _decision_steps(args.checkpoint_freq, args.action_repeat)
    evaluation_decisions = _decision_steps(args.eval_freq, args.action_repeat)
    source_raw_step_control = args.algo in (
        "scene_rep",
        "mst",
        "topo_scene",
        "topo_scene_balanced",
        "temporal_graph",
        "sac",
    )
    model_learning_starts = _model_learning_starts(
        args.learning_starts,
        args.action_repeat,
        source_raw_step_control=source_raw_step_control,
    )
    run_name = args.model_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{args.algo}__{args.scenario}__seed{args.seed}"
    )
    run_dir = args.output_dir.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_log_dir = _tensorboard_log_directory(
        run_dir, override_root=tensorboard_log_root
    )
    with (run_dir / "arguments.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "implementation_fidelity": {
                    "implementation_id": {
                        "scene_rep": "released_scene_rep_pytorch_port_v1",
                        "mst": "released_mst_pytorch_port_v1",
                        "topo_scene": "topo_temporal_graph_slt_pytorch_v1",
                        "topo_scene_balanced": "topo_temporal_graph_slt_balanced_slots_v1",
                        "temporal_graph": "temporal_vehicle_graph_slt_pytorch_v1",
                        "sac": "paper_sac_lstm_reconstruction_v2",
                        "ppo": "released_source_ppo_pytorch_port_v1",
                    }[args.algo],
                    "source_primary": args.algo
                    in ("scene_rep", "mst", "sac", "ppo"),
                    "research_extension": args.algo
                    in ("topo_scene", "topo_scene_balanced", "temporal_graph"),
                    "reconstruction_required": args.algo == "sac",
                    "reconstruction_reason": (
                        "The release contains the GRU-based RLEncoder and "
                        "STATE_LSTM adapter but omits a selectable/runnable "
                        "plain-SAC configuration; v2 restores the adapter's "
                        "distinct ego/social field semantics and time mask."
                        if args.algo == "sac"
                        else None
                    ),
                },
                "runtime_paths": {
                    "tensorboard_log": str(tensorboard_log_dir),
                    "tensorboard_short_root_injected": tensorboard_log_root
                    is not None,
                },
                "requested_raw_steps": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "effective_sb3_decision_steps": {
                    "expected_total_without_partial_repeats": total_decisions,
                    "learn_loop_upper_bound": (
                        args.max_steps if source_raw_step_control else total_decisions
                    ),
                    "learning_starts_decisions": warmup_decisions,
                    "model_learning_starts_argument": model_learning_starts,
                    "checkpoint_freq": checkpoint_decisions,
                    "eval_freq": evaluation_decisions,
                    "gradient_steps_per_decision": args.action_repeat,
                    "source_ppo_horizon": 512 if args.algo == "ppo" else None,
                    "source_ppo_expected_collected_raw_steps": (
                        _source_ppo_collected_steps(args.max_steps)
                        if args.algo == "ppo" and not args.check_only
                        else None
                    ),
                    "source_ppo_test_checkpoint_raw_step": (
                        _source_ppo_test_checkpoint_step(
                            args.max_steps, args.checkpoint_freq
                        )
                        if args.algo == "ppo" and not args.check_only
                        else None
                    ),
                },
            },
            handle,
            indent=2,
        )

    validation_env = env_factory(args)
    try:
        _check_environment(validation_env, args.seed)
    finally:
        validation_env.close()
    if args.check_only:
        model_check_env = env_factory(args)
        try:
            check_model = make_model(
                args.algo,
                model_check_env,
                scenario=args.scenario,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                discount=args.discount,
                learning_starts=model_learning_starts,
                buffer_size=args.buffer_size,
                action_repeat=args.action_repeat,
                seed=args.seed,
                device=args.device,
                tensorboard_log=None,
                verbose=0,
            )
            check_observation, _ = model_check_env.reset(seed=args.seed)
            check_action, _ = check_model.predict(
                check_observation, deterministic=True
            )
            if not np.isfinite(np.asarray(check_action)).all():
                raise FloatingPointError("Model check produced a non-finite action")
            topology_graph_info = getattr(
                check_model, "topology_graph_info", None
            )
            if topology_graph_info is not None:
                with (run_dir / "topology_graph.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(topology_graph_info, handle, indent=2)
            check_result = {
                "status": "ok",
                "scenario": args.scenario,
                "algorithm": args.algo,
                "model_class": type(check_model).__name__,
                "feature_extractor_class": type(
                    check_model.critic.features_extractor
                    if hasattr(check_model, "critic")
                    else check_model.policy.features_extractor
                ).__name__,
                "action_shape": list(np.asarray(check_action).shape),
                "action_finite": True,
                "topology_graph": topology_graph_info,
            }
            with (run_dir / "check_result.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(check_result, handle, indent=2)
        finally:
            model_check_env.close()
        print(json.dumps(check_result, ensure_ascii=False, indent=2))
        return 0

    # Validation performs several resets and random rollouts.  Train in a
    # fresh environment so those checks cannot advance the released scenario
    # traffic cycle or alter the learner's first episode.
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
        model = make_model(
            args.algo,
            monitored_env,
            scenario=args.scenario,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            discount=args.discount,
            learning_starts=model_learning_starts,
            buffer_size=args.buffer_size,
            action_repeat=args.action_repeat,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(tensorboard_log_dir),
        )
        topology_graph_info = getattr(model, "topology_graph_info", None)
        if topology_graph_info is not None:
            with (run_dir / "topology_graph.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(topology_graph_info, handle, indent=2)
        parameter_profile = _model_parameter_profile(model)
        model_device = getattr(model, "device", torch.device("cpu"))
        cuda_profiled = getattr(model_device, "type", str(model_device)) == "cuda"
        if cuda_profiled:
            torch.cuda.reset_peak_memory_stats(model_device)
        callbacks = []
        if source_raw_step_control:
            callbacks.append(
                RawStepControlCallback(
                    raw_step_budget=args.max_steps,
                    checkpoint_frequency=args.checkpoint_freq,
                    checkpoint_path=run_dir / "checkpoints",
                    checkpoint_prefix=args.algo,
                )
            )
        elif checkpoint_decisions > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_decisions,
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix=args.algo,
                )
            )
        callbacks.append(
            BestTrainingSuccessCallback(
                run_dir / "best_training_success_model"
            )
        )
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
        training_started = time.perf_counter()
        model.learn(
            total_timesteps=(
                args.max_steps if source_raw_step_control else total_decisions
            ),
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=False,
        )
        training_wall_seconds = time.perf_counter() - training_started
        peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated(model_device)) / (1024.0**2)
            if cuda_profiled
            else None
        )
        persisted_diagnostics = (
            model.training_diagnostics()
            if hasattr(model, "training_diagnostics")
            else {}
        )
        with (run_dir / "training_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "algorithm": args.algo,
                    "scenario": args.scenario,
                    "raw_steps": args.max_steps,
                    "learner_updates": int(getattr(model, "_n_updates", 0)),
                    "statistics": persisted_diagnostics,
                },
                handle,
                indent=2,
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
        with (run_dir / "checkpoint_audit.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(checkpoint_audit, handle, indent=2)
        post_training_learner_timesteps = int(model.num_timesteps)
        if source_raw_step_control:
            collected_training_raw_steps = int(model._raw_steps_seen)
        elif args.algo == "ppo":
            collected_training_raw_steps = post_training_learner_timesteps
        else:
            collected_training_raw_steps = (
                post_training_learner_timesteps * args.action_repeat
            )
        source_test_checkpoint_step = None
        if args.algo == "ppo":
            # Both released on-policy runners finish the crossing horizon, but
            # tools/test.py restores the latest periodic checkpoint. Preserve
            # the post-horizon learner separately and make final_model the
            # source-selected test policy.
            model.save(run_dir / "post_horizon_model")
            source_test_checkpoint_step = _source_ppo_test_checkpoint_step(
                args.max_steps, args.checkpoint_freq
            )
            source_checkpoint = (
                run_dir
                / "checkpoints"
                / f"{args.algo}_{source_test_checkpoint_step}_steps.zip"
            )
            if not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Source PPO test checkpoint was not saved: {source_checkpoint}"
                )
            model = type(model).load(source_checkpoint, device=args.device)
        model_path = run_dir / "final_model"
        model.save(model_path)
        selected_model_learner_timesteps = int(model.num_timesteps)
        profile_observation, _ = env.reset(seed=args.seed + 20_000)
        inference_profile = _profile_inference(model, profile_observation)
        performance_profile = {
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
        with (run_dir / "performance_profile.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(performance_profile, handle, indent=2)
        if eval_env is not None:
            eval_env.close()

        final_eval_env = make_injected_evaluation_env(env_factory, args)
        try:
            space_contract = validate_model_environment_spaces(
                model, final_eval_env
            )
            report = evaluate_model_detailed(
                model,
                final_eval_env,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                policy_action_hold=(
                    3 if args.algo == "ppo" and args.scenario == "carla" else 1
                ),
            )
            summary = report.summary
            evaluation_provenance = (
                build_evaluation_provenance(
                    model=model,
                    env=final_eval_env,
                    report=report,
                    algorithm=args.algo,
                    scenario=args.scenario,
                    traffic_protocol=args.traffic_protocol,
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
                    "environment_class": (
                        f"{type(final_eval_env).__module__}."
                        f"{type(final_eval_env).__qualname__}"
                    ),
                    **space_contract,
                }
            )
        finally:
            final_eval_env.close()
        with (run_dir / "final_evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        trained_raw_steps = getattr(model, "_raw_steps_seen", None)
        if trained_raw_steps is None and args.algo == "ppo":
            trained_raw_steps = model.num_timesteps
        detailed_payload = {
            "model": str(model_path.resolve()) + ".zip",
            "algorithm": args.algo,
            "scenario": args.scenario,
            "evaluation_seed_start": args.seed + 10_000,
            "trained_raw_steps": (
                int(trained_raw_steps) if trained_raw_steps is not None else None
            ),
            "collected_training_raw_steps": collected_training_raw_steps,
            "post_training_learner_timesteps": post_training_learner_timesteps,
            "selected_model_learner_timesteps": selected_model_learner_timesteps,
            "source_test_checkpoint_step": source_test_checkpoint_step,
            "best_training_success_checkpoint": (
                json.loads(
                    (run_dir / "best_training_success.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (run_dir / "best_training_success.json").is_file()
                else None
            ),
            "evaluation_provenance": evaluation_provenance,
            "completion_time_population_std_ddof": 0,
            **report.to_dict(),
        }
        with (run_dir / "paper_evaluation_detailed.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(detailed_payload, handle, indent=2)
        print(
            json.dumps(
                {
                    "model": str(model_path) + ".zip",
                    "requested_raw_steps": args.max_steps,
                    "selected_model_sb3_timesteps": selected_model_learner_timesteps,
                    "post_training_sb3_timesteps": post_training_learner_timesteps,
                    "collected_training_raw_steps": collected_training_raw_steps,
                    "source_test_checkpoint_step": source_test_checkpoint_step,
                    **summary.to_dict(),
                },
                indent=2,
            )
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
