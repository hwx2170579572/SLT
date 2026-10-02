"""Training cells: hold35k (v4_8) and MST+SLT (base) for one scenario each."""
from __future__ import annotations

import os
import time

import torch

from .common import (
    CHECKPOINT_FREQUENCY,
    HOLD35K_CANDIDATE,
    LEARNING_STARTS_RAW_STEPS,
    RAW_TRAINING_STEPS,
    RESULT_ROOT,
    lock,
    make_hold35k_env,
    make_mst_env_factory,
    seal,
    sha256,
    train_dir,
    write,
)


def _already_done(cell_dir, artifacts):
    return all((cell_dir / name).is_file() for name in artifacts)


def train_hold35k(scene, device="cuda", smoke=False):
    """Train one hold35k (v4_8) cell with a checkpoint every 2000 raw steps."""
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch import RawStepControlCallback
    from tools.phase2_model_factory_v1 import verify_optimizer_settings
    from tools.v48_stability_v4.common import model_state_sha
    from tools.v48_stability_v4.model import learning_rate, load_model, make_model

    output = train_dir("hold35k", scene, smoke=smoke)
    if (output / "training_complete.json").is_file():
        return str(output / "final_model.zip")
    if output.exists():
        raise RuntimeError(f"Partial training retained, no silent restart: {output}")
    output.mkdir(parents=True, exist_ok=False)

    budget = 96 if smoke else RAW_TRAINING_STEPS
    warmup = 48 if smoke else LEARNING_STARTS_RAW_STEPS
    frequency = 48 if smoke else CHECKPOINT_FREQUENCY
    namespace = "sm" if smoke else "tr"
    started = time.time()
    env, model = None, None
    write(output / "status.json", dict(status="starting", pid=os.getpid(), smoke=smoke))
    try:
        env = Monitor(
            make_hold35k_env(scene, namespace, training=True),
            filename=str(output / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            ),
        )
        model = make_model(env, scene, HOLD35K_CANDIDATE, device=device, smoke=smoke)
        model.stability_trace_path = str(output / "optimization_trace.jsonl")
        seal(
            output / "arguments.json",
            dict(
                scene=scene,
                candidate=HOLD35K_CANDIDATE,
                settings=model.stability_config,
                seed=930100 if smoke else 0,
                raw_budget=budget,
                warmup=warmup,
                smoke=smoke,
                initial_tensor_sha256=model_state_sha(model),
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    write(
                        output / "progress.json",
                        dict(
                            raw_steps=self.model._raw_steps_seen,
                            updates=self.model._n_updates,
                            updated_at=time.time(),
                        ),
                    )
                return True

        model.learn(
            total_timesteps=budget,
            callback=CallbackList(
                [
                    RawStepControlCallback(
                        raw_step_budget=budget,
                        checkpoint_frequency=frequency,
                        checkpoint_path=output / "checkpoints",
                        checkpoint_prefix="ckpt",
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != budget or model._n_updates != budget - warmup + 1:
            raise AssertionError("Raw-step/update budget mismatch")
        rate = learning_rate(model.stability_config, budget)
        verify_optimizer_settings(model, learning_rate=rate, tau=model.stability_config["tau"])
        model.audit_parameters()
        final = output / "final_model.zip"
        model.save(final)
        restored = load_model(final, scene, env, device)
        if model_state_sha(restored) != model_state_sha(model):
            raise AssertionError("Final model roundtrip changes tensors")
        verify_optimizer_settings(restored, learning_rate=rate, tau=model.stability_config["tau"])
        write(output / "training_diagnostics.json", model.training_diagnostics())
        complete = dict(
            smoke=smoke,
            checkpoint_sha256=sha256(final),
            raw_steps=budget,
            updates=model._n_updates,
            tensor_roundtrip_equal=True,
            finite_parameter_check=True,
            numerical_divergence_detected=False,
            replay_size=model.replay_buffer.size(),
            replay_insertions=model.replay_buffer._next_insertion_id,
            optimizer_settings=verify_optimizer_settings(
                model, learning_rate=rate, tau=model.stability_config["tau"]
            ),
            train_monitor_sha256=sha256(output / "train_monitor.csv"),
            optimization_trace_sha256=sha256(output / "optimization_trace.jsonl"),
            wall_seconds=time.time() - started,
        )
        seal(output / "training_complete.json", complete)
        write(output / "status.json", dict(status="completed", **complete))
        return str(final)
    except BaseException as exc:
        write(
            output / "status.json",
            dict(
                status="failed",
                error=repr(exc),
                smoke=smoke,
                numerical_divergence_detected=isinstance(exc, FloatingPointError),
                raw_steps=getattr(model, "_raw_steps_seen", None),
                wall_seconds=time.time() - started,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


def train_mst_slt(scene, device="cuda", smoke=False):
    """Train one MST+SLT (base) cell via train_sb3 with --checkpoint-freq 2000."""
    from tools import train_sb3

    output = train_dir("mst_slt", scene, smoke=smoke)
    if _already_done(output, ("final_model.zip", "paper_evaluation_detailed.json")):
        return str(output / "final_model.zip")
    if output.exists():
        raise RuntimeError(f"Partial training retained, no silent restart: {output}")

    max_steps = 96 if smoke else RAW_TRAINING_STEPS
    learning_starts = 48 if smoke else LEARNING_STARTS_RAW_STEPS
    frequency = 48 if smoke else CHECKPOINT_FREQUENCY
    eval_episodes = 1 if smoke else 100

    argv = [
        "--algo",
        "scene_rep",
        "--scenario",
        scene,
        "--max-steps",
        str(max_steps),
        "--learning-starts",
        str(learning_starts),
        "--checkpoint-freq",
        str(frequency),
        "--eval-freq",
        "0",
        "--eval-episodes",
        str(eval_episodes),
        "--seed",
        "0",
        "--device",
        device,
        "--batch-size",
        "32",
        "--learning-rate",
        "0.0001",
        "--discount",
        "0.99",
        "--buffer-size",
        "20000",
        "--action-repeat",
        "3",
        "--output-dir",
        str(output.parent.resolve()),
        "--model-name",
        output.name,
        "--ego-control-profile",
        "direct",
        "--episode-limit-profile",
        "source",
        "--traffic-protocol",
        "frozen_80_20",
    ]
    train_sb3.main(
        argv,
        env_factory=make_mst_env_factory(scene),
        default_output_dir=output.parent,
        require_paper_evaluation_contract=True,
        tensorboard_log_root=RESULT_ROOT / "tb",
    )
    return str(output / "final_model.zip")


def run_cell(method, scene, *, device="cuda", smoke=False):
    torch.set_num_threads(1)
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This screen is configured for GPU workers only")
    if method not in ("hold35k", "mst_slt"):
        raise ValueError(f"Unknown method {method!r}")
    with lock(RESULT_ROOT / "locks" / f"{method}__{scene}.json"):
        if method == "hold35k":
            return train_hold35k(scene, device=device, smoke=smoke)
        return train_mst_slt(scene, device=device, smoke=smoke)
