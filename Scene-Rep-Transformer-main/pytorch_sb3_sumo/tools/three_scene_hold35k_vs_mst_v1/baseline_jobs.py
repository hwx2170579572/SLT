"""Training cells for the seven comparison baselines.

Four SB3-native baselines (HSAC-MLP, HSAC-LSTM, GNN+SAC, MST-no-SLT) train
online exactly like hold35k/mst_slt: ``model.learn`` with a checkpoint every
2000 raw steps.  The three sequence baselines (Hybrid-DT, Decision Transformer,
HyAR) train offline: they first imitate a trained strong-baseline policy, then
run a fixed number of gradient steps with a checkpoint every 2000 steps.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from algos.hybrid_action.hyar.trainer import train_hyar
from algos.transformer_rl import SumoStateTokenizer, train_hybrid_dt
from algos.transformer_rl.decision_transformer.trainer import train_decision_transformer
from algos.transformer_rl.offline_dataset import (
    compute_returns_to_go,
    continuous_action_to_vector,
    hybrid_action_to_vector,
)
from configs.sb3_configs_baselines import make_baseline_model, make_sequence_baseline

from .common import (
    BASELINE_CHECKPOINT_PREFIX,
    BASELINE_ENV_CONTRACT,
    CHECKPOINT_FREQUENCY,
    LEARNING_STARTS_RAW_STEPS,
    RAW_TRAINING_STEPS,
    RESULT_ROOT,
    SEQUENCE_BASELINE_METHODS,
    baseline_checkpoint,
    baseline_data_source,
    lock,
    make_baseline_env,
    seal,
    sha256,
    train_dir,
    write,
)

SEQUENCE_DATASET_EPISODES = 100


# --------------------------------------------------------------------------- #
# Offline data collection for the sequence baselines
# --------------------------------------------------------------------------- #
def _load_source_env_model(source: str, scene: str, device: str, namespace: str):
    """Load the trained strong-baseline policy and a matching environment.

    ``hsac_mlp`` (v4_8 hybrid head) seeds Hybrid-DT / HyAR; ``gnn_sac`` (base
    continuous head) seeds the Decision Transformer.  Each source is loaded
    actor-only where it has a hybrid head, exactly as its evaluation does.
    """
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.v48_stability_v4.model import use_actor_only

    checkpoint = baseline_checkpoint(source, scene, RAW_TRAINING_STEPS)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Data source {source} for {scene} is not trained yet: {checkpoint}. "
            "Train the SB3 baselines before the sequence baselines."
        )

    if source == "hsac_mlp":
        env = make_baseline_env("hsac_mlp", scene, namespace, training=False)
        model = use_actor_only(
            SceneRepresentationSAC.load(
                str(checkpoint), env=env, device=device, buffer_size=32
            )
        )
        return env, model

    if source == "gnn_sac":
        env = make_baseline_env("gnn_sac", scene, namespace, training=False)
        model = SceneRepresentationSAC.load(
            str(checkpoint), env=env, device=device, buffer_size=32
        )
        return env, model

    raise ValueError(f"Unknown sequence data source {source!r}")


def _collect_sequence_dataset(
    method: str, scene: str, *, episodes: int, device: str, namespace: str
) -> dict[str, np.ndarray]:
    """Roll out the source policy and pack a flat ``(state, action, ...)`` table."""
    source = baseline_data_source(method)
    env, model = _load_source_env_model(source, scene, device, namespace)
    tokenizer = SumoStateTokenizer(env.observation_space)
    action_mapper = (
        continuous_action_to_vector if source == "gnn_sac" else hybrid_action_to_vector
    )
    seed_start = 420000 if source == "hsac_mlp" else 10000
    transitions: list[dict] = []
    try:
        for index in range(episodes):
            episode_seed = seed_start + index
            np.random.seed(episode_seed + 600000)
            torch.manual_seed(episode_seed + 600000)
            if hasattr(env, "_traffic_episode_index"):
                env._traffic_episode_index, env._traffic_roll = index, None
            observation, _ = env.reset(seed=episode_seed)
            for timestep in range(1000):
                state = tokenizer(observation)
                action = model.predict(observation, deterministic=True)[0]
                action_vector = action_mapper(action)
                next_observation, reward, terminated, truncated, info = env.step(action)
                transitions.append(
                    {
                        "state": np.asarray(state, dtype=np.float32),
                        "action": np.asarray(action_vector, dtype=np.float32),
                        "reward": float(info.get("undiscounted_reward", reward)),
                        "done": bool(terminated or truncated),
                        "timestep": timestep,
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        env.close()

    returns_to_go = compute_returns_to_go(transitions, discount=0.99)
    return {
        "states": np.stack([t["state"] for t in transitions]).astype(np.float32),
        "actions": np.stack([t["action"] for t in transitions]).astype(np.float32),
        "rewards": np.asarray([t["reward"] for t in transitions], dtype=np.float32),
        "returns_to_go": returns_to_go.astype(np.float32),
        "timesteps": np.asarray([t["timestep"] for t in transitions], dtype=np.int64),
        "dones": np.asarray([t["done"] for t in transitions], dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# SB3-native baselines (online)
# --------------------------------------------------------------------------- #
def _train_sb3_baseline(method: str, scene: str, device: str, smoke: bool):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch import RawStepControlCallback

    output = train_dir(method, scene, smoke=smoke)
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
            make_baseline_env(method, scene, namespace, training=True),
            filename=str(output / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            ),
        )
        model = make_baseline_model(
            method,
            env,
            scenario=scene,
            device=device,
            seed=930100 if smoke else 0,
        )
        seal(
            output / "arguments.json",
            dict(
                method=method,
                scene=scene,
                contract=BASELINE_ENV_CONTRACT[method],
                raw_budget=budget,
                warmup=warmup,
                smoke=smoke,
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
                        checkpoint_prefix=BASELINE_CHECKPOINT_PREFIX[method],
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != budget:
            raise AssertionError("Raw-step budget mismatch")
        final = output / "final_model.zip"
        model.save(final)
        write(output / "training_diagnostics.json", model.training_diagnostics())
        complete = dict(
            smoke=smoke,
            checkpoint_sha256=sha256(final),
            raw_steps=budget,
            updates=model._n_updates,
            replay_size=model.replay_buffer.size(),
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
                raw_steps=getattr(model, "_raw_steps_seen", None),
                wall_seconds=time.time() - started,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


# --------------------------------------------------------------------------- #
# Sequence baselines (offline)
# --------------------------------------------------------------------------- #
def _train_sequence_with_checkpoints(
    trainer, model, dataset, *, num_steps, frequency, save_fn, device, seed=0
):
    """Run ``trainer`` in contiguous blocks, checkpointing at every frequency."""
    remaining = num_steps
    block = 0
    while remaining > 0:
        steps = min(frequency, remaining)
        trainer(model, dataset, num_steps=steps, device=device, seed=seed + block)
        remaining -= steps
        block += 1
        save_fn(num_steps - remaining)


def _train_sequence_baseline(method: str, scene: str, device: str, smoke: bool):
    output = train_dir(method, scene, smoke=smoke)
    if (output / "training_complete.json").is_file():
        return str(output / "final_model.pt")
    if output.exists():
        raise RuntimeError(f"Partial training retained, no silent restart: {output}")
    output.mkdir(parents=True, exist_ok=False)

    num_steps = 96 if smoke else RAW_TRAINING_STEPS
    frequency = 48 if smoke else CHECKPOINT_FREQUENCY
    episodes = 2 if smoke else SEQUENCE_DATASET_EPISODES
    namespace = "sm" if smoke else "dl"
    started = time.time()
    write(output / "status.json", dict(status="starting", pid=os.getpid(), smoke=smoke))
    try:
        dataset = _collect_sequence_dataset(
            method, scene, episodes=episodes, device=device, namespace=namespace
        )
        state_dim = int(dataset["states"].shape[1])
        act_dim = int(dataset["actions"].shape[1])
        target_return = float(np.percentile(dataset["returns_to_go"], 95))
        write(
            output / "dataset_info.json",
            dict(
                source=baseline_data_source(method),
                episodes=episodes,
                transitions=int(dataset["states"].shape[0]),
                state_dim=state_dim,
                act_dim=act_dim,
                target_return=target_return,
            ),
        )

        model = make_sequence_baseline(method, state_dim, act_dim=act_dim)
        if method == "hybrid_dt":
            trainer = train_hybrid_dt
        elif method == "decision_transformer":
            trainer = train_decision_transformer
        elif method == "hyar":
            trainer = train_hyar
        else:
            raise ValueError(f"Not a sequence baseline: {method!r}")

        checkpoints = output / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        prefix = BASELINE_CHECKPOINT_PREFIX[method]

        def save_fn(step: int) -> None:
            payload = {
                "method": method,
                "state_dim": state_dim,
                "act_dim": act_dim,
                "target_return": target_return,
                "state_dict": model.state_dict(),
            }
            torch.save(payload, checkpoints / f"{prefix}_raw_{step}_steps.pt")
            write(
                output / "progress.json",
                dict(gradient_steps=step, updated_at=time.time()),
            )

        _train_sequence_with_checkpoints(
            trainer,
            model,
            dataset,
            num_steps=num_steps,
            frequency=frequency,
            save_fn=save_fn,
            device=device,
            seed=0,
        )
        final = output / "final_model.pt"
        torch.save(
            {
                "method": method,
                "state_dim": state_dim,
                "act_dim": act_dim,
                "state_dict": model.state_dict(),
            },
            final,
        )
        complete = dict(
            smoke=smoke,
            gradient_steps=num_steps,
            dataset_episodes=episodes,
            dataset_transitions=int(dataset["states"].shape[0]),
            checkpoint_sha256=sha256(final),
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
                wall_seconds=time.time() - started,
            ),
        )
        raise


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def run_baseline_cell(method: str, scene: str, *, device: str = "cuda", smoke: bool = False):
    torch.set_num_threads(1)
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This screen is configured for GPU workers only")
    if method not in SEQUENCE_BASELINE_METHODS + ("hsac_mlp", "hsac_lstm", "gnn_sac", "mst"):
        raise ValueError(f"Unknown baseline method {method!r}")
    with lock(RESULT_ROOT / "locks" / f"{method}__{scene}.json"):
        if method in SEQUENCE_BASELINE_METHODS:
            return _train_sequence_baseline(method, scene, device=device, smoke=smoke)
        return _train_sb3_baseline(method, scene, device=device, smoke=smoke)
