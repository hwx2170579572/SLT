"""Post-hoc checkpoint evaluation for the three-scene hold35k vs MST+SLT study.

Evaluates every historical checkpoint (2000..50000 raw steps, 25 points) of
both methods on cross, carla and cross_left with 50 episodes each, then the
report draws standard success-rate vs raw-steps curves.

Each family is evaluated in its own environment contract:
  * hold35k (v4_8): actor-only deterministic load, 50 eps seed block 420000.
  * mst_slt (base): SceneRepresentationSAC load, 50 eps seed 10000.

Evaluation is inference-only (no gradient updates); the controller defaults to
6 concurrent GPU workers sharing one GPU, and the replay buffer is loaded at
capacity 32 because inference never touches it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.three_scene_hold35k_vs_mst_v1.common import (
    BASELINE_ENV_CONTRACT,
    BASELINE_METHODS,
    EVAL_CURVE_ROOT,
    EVAL_EPISODES,
    RAW_STEPS,
    RESULT_ROOT,
    SCENES,
    baseline_checkpoint,
    hold35k_checkpoint,
    make_baseline_env_factory,
    make_hold35k_env,
    make_mst_env_factory,
    mst_checkpoint,
    read,
    seal,
    sha256,
    write,
)

METHODS = ("hold35k", "mst_slt")
SEED_START = {
    "hold35k": 420000,
    "mst_slt": 10000,
    "hsac_mlp": 420000,
    "hsac_lstm": 420000,
    "gnn_sac": 10000,
    "hybrid_dt": 420000,
    "decision_transformer": 10000,
    "hyar": 420000,
    "mst": 10000,
}


def build_tasks():
    tasks = []
    for scene in SCENES:
        for method in METHODS:
            for raw_step in RAW_STEPS:
                checkpoint = (
                    hold35k_checkpoint(scene, raw_step)
                    if method == "hold35k"
                    else mst_checkpoint(scene, raw_step)
                )
                tasks.append(
                    dict(
                        method=method,
                        scene=scene,
                        raw_step=raw_step,
                        checkpoint=str(checkpoint),
                        output=str(
                            EVAL_CURVE_ROOT
                            / f"{method}__{scene}"
                            / f"raw_{raw_step}"
                            / "result.json"
                        ),
                    )
                )
    tasks.extend(build_baseline_tasks())
    return tasks


def task_name(task):
    return f"{task['method']}__{task['scene']}__raw{task['raw_step']}"


def summarize(records):
    import numpy as np

    n = len(records)
    returns = [r["episode_return"] for r in records]
    raw_lengths = [r["raw_steps"] for r in records]
    return dict(
        episodes=n,
        success_rate=sum(int(r["success"]) for r in records) / n,
        collision_rate=sum(int(r["collision"]) for r in records) / n,
        off_route_rate=sum(int(r["off_route"]) for r in records) / n,
        timeout_rate=sum(int(r["timeout"]) for r in records) / n,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_raw_steps=float(np.mean(raw_lengths)),
        episode_records=records,
    )


def evaluate_hold35k(task, device, episodes):
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    namespace = "ec" + sha256(checkpoint)[:10]
    env = make_hold35k_env(scene, namespace)
    model = use_actor_only(
        StabilitySAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
    )
    try:
        if model._raw_steps_seen != raw_step:
            raise ValueError(f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}")
        records = []
        for index in range(episodes):
            episode_seed = SEED_START["hold35k"] + index
            np.random.seed(episode_seed + 600000)
            torch.manual_seed(episode_seed + 600000)
            env._traffic_episode_index, env._traffic_roll = index, None
            detailed = evaluate_model_detailed(
                model,
                env,
                episodes=1,
                seed=episode_seed,
                deterministic=True,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            records.append(detailed.episode_records[0].to_dict())
    finally:
        env.close()
    return records


def evaluate_mst_slt(task, device, episodes):
    import argparse as _argparse

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.paper_evaluation_contract import make_injected_evaluation_env

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    factory = make_mst_env_factory(scene)
    args = _argparse.Namespace(
        scenario=scene,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
    )
    env = make_injected_evaluation_env(factory, args)
    model = SceneRepresentationSAC.load(
        str(checkpoint), env=env, device=device, buffer_size=32
    )
    try:
        if model._raw_steps_seen != raw_step:
            raise ValueError(f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}")
        report = evaluate_model_detailed(
            model, env, episodes=episodes, seed=SEED_START["mst_slt"], policy_action_hold=1
        )
        records = [r.to_dict() for r in report.episode_records]
    finally:
        env.close()
    return records


def run_task(task, device, smoke=False):
    import torch

    torch.set_num_threads(1)
    episodes = 1 if smoke else EVAL_EPISODES

    if not smoke:
        identity = dict(
            method=task["method"],
            scene=task["scene"],
            raw_step=task["raw_step"],
            checkpoint=task["checkpoint"],
            checkpoint_sha256=sha256(task["checkpoint"]),
            episodes=episodes,
            seed_start=SEED_START[task["method"]],
            deployment="actor_deterministic",
            device=device,
        )
        output = Path(task["output"])
        if output.exists():
            existing = read(output)
            if existing.get("identity") != identity:
                raise ValueError(f"Evaluation identity mismatch: {output}")
            return str(output)

    started = time.time()
    if task["method"] == "hold35k":
        records = evaluate_hold35k(task, device, episodes)
    elif task["method"] == "mst_slt":
        records = evaluate_mst_slt(task, device, episodes)
    else:
        records = BASELINE_EVALUATORS[task["method"]](task, device, episodes)

    if smoke:
        print(
            json.dumps(
                dict(
                    method=task["method"],
                    scene=task["scene"],
                    raw_step=task["raw_step"],
                    _raw_steps_seen_assert_passed=True,
                    episode=records[0],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return None

    result = dict(
        identity=identity, **summarize(records), wall_seconds=time.time() - started
    )
    seal(task["output"], result)
    return task["output"]


def controller(workers, smoke=False):
    import torch

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; evaluation has no CPU fallback")
    torch.empty(1, device="cuda")
    print(
        f"{workers} GPU worker processes on {torch.cuda.get_device_name(0)}; "
        f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB",
        flush=True,
    )

    tasks = build_tasks()
    log_dir = EVAL_CURVE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running, done = {}, {}
    stop_requested = threading.Event()
    start = time.time()

    def execute(task, index):
        if stop_requested.is_set():
            return dict(exit_code=-1, log=None, cancelled=True)
        name = task_name(task)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = log_dir / f"{name}_{stamp}.log"
        command = [
            sys.executable,
            "-m",
            "tools.three_scene_eval_curve",
            "run-task",
            "--index",
            str(index),
        ]
        if smoke:
            command.append("--smoke")
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            running[name] = process
            returncode = process.wait()
            running.pop(name, None)
        return dict(exit_code=returncode, log=str(log), completed_at=time.time())

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(execute, task, i): i for i, task in enumerate(tasks)}
    try:
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            name = task_name(tasks[index])
            done[name] = future.result()
            print(name, done[name]["exit_code"], flush=True)
            write(
                EVAL_CURVE_ROOT / "controller_progress.json",
                dict(done=done, total=len(tasks), updated_at=time.time()),
            )
    except BaseException:
        stop_requested.set()
        for future in futures:
            future.cancel()
        for process in list(running.values()):
            try:
                import psutil

                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
            except Exception:
                pass
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    receipt = dict(
        passed=all(v["exit_code"] == 0 for v in done.values()),
        jobs=done,
        workers=workers,
        smoke=smoke,
        wall_seconds=time.time() - start,
    )
    write(EVAL_CURVE_ROOT / "controller.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError(
            "Some jobs failed; inspect logs. Partial results were preserved."
        )
    return receipt


def evaluate_hybrid_sac_baseline(task, device, episodes):
    """Actor-only deterministic evaluation of a v4_8 hybrid-head SAC baseline (HSAC)."""

    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.v48_stability_v4.model import use_actor_only

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    namespace = "ec" + sha256(checkpoint)[:10]
    env = make_hold35k_env(scene, namespace)
    model = use_actor_only(
        SceneRepresentationSAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
    )
    try:
        if model._raw_steps_seen != raw_step:
            raise ValueError(f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}")
        records = []
        for index in range(episodes):
            episode_seed = SEED_START[task["method"]] + index
            np.random.seed(episode_seed + 600000)
            torch.manual_seed(episode_seed + 600000)
            env._traffic_episode_index, env._traffic_roll = index, None
            detailed = evaluate_model_detailed(
                model,
                env,
                episodes=1,
                seed=episode_seed,
                deterministic=True,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            records.append(detailed.episode_records[0].to_dict())
    finally:
        env.close()
    return records


def evaluate_continuous_sac_baseline(task, device, episodes):
    """Actor-only deterministic evaluation of a base-contract baseline (GNN+SAC / MST-no-SLT)."""

    import argparse as _argparse

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.paper_evaluation_contract import make_injected_evaluation_env

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    # Continuous-head baselines pair with MST+SLT: reuse its exact environment
    # factory (same overlay, same base adapter) so the ablation is isolated.
    factory = make_mst_env_factory(scene)
    args = _argparse.Namespace(
        scenario=scene,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
    )
    env = make_injected_evaluation_env(factory, args)
    model = SceneRepresentationSAC.load(
        str(checkpoint), env=env, device=device, buffer_size=32
    )
    try:
        if model._raw_steps_seen != raw_step:
            raise ValueError(f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}")
        report = evaluate_model_detailed(
            model, env, episodes=episodes, seed=SEED_START[task["method"]], policy_action_hold=1
        )
        records = [r.to_dict() for r in report.episode_records]
    finally:
        env.close()
    return records


def _rollout_sequence_episode(
    model, env, tokenizer, method, act_dim, target_return, device, episode, seed
):
    """Autoregressive rollout returning an ``EpisodeEvaluationRecord`` dict.

    Hybrid-DT / Decision Transformer use return-to-go conditioning over a
    growing ``(return, state, action, timestep)`` history; HyAR uses its
    state-conditioned latent policy.  All three emit a ``Box(2)`` action
    ``[speed, lane]`` consumed by the environment.
    """
    import numpy as np
    import torch

    from algos.hybrid_action.hyar.sumo_adapter import hybrid_to_env_action
    from algos.sb3_torch.evaluation import EpisodeEvaluationRecord
    from algos.transformer_rl.offline_dataset import hybrid_action_to_vector

    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    decision_steps = 0
    info: dict = {}

    if method == "hyar":
        for _ in range(10000):
            state = torch.as_tensor(
                np.asarray([tokenizer(observation)], dtype=np.float32), device=device
            )
            with torch.no_grad():
                _, discrete_index, continuous = model.act(state, deterministic=True)
            action = hybrid_to_env_action(
                int(discrete_index.item()), continuous.detach().cpu().numpy()
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(info.get("undiscounted_reward", reward))
            decision_steps += 1
            if terminated or truncated:
                break
    else:
        states = [tokenizer(observation).astype(np.float32)]
        # The last action slot is a placeholder that ``get_action`` consumes as
        # the "current" position; it is replaced by the predicted action before
        # the next history entry is appended.
        actions = [np.zeros(act_dim, dtype=np.float32)]
        returns_to_go = [float(target_return)]
        timesteps = [0]
        for _ in range(10000):
            with torch.no_grad():
                action_t = model.get_action(
                    torch.as_tensor(np.stack(states), device=device),
                    torch.as_tensor(np.stack(actions), device=device),
                    None,
                    torch.as_tensor(
                        np.asarray(returns_to_go, dtype=np.float32), device=device
                    ),
                    torch.as_tensor(
                        np.asarray(timesteps, dtype=np.int64), device=device
                    ),
                )
            action = action_t.detach().cpu().numpy()
            # Hybrid-DT's history carries ``[speed, lane one-hot]``; its
            # ``get_action`` returns ``[speed, lane_code]`` which we map back.
            actions[-1] = (
                hybrid_action_to_vector(action) if method == "hybrid_dt" else action
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(info.get("undiscounted_reward", reward))
            decision_steps += 1
            if terminated or truncated:
                break
            states.append(tokenizer(observation).astype(np.float32))
            actions.append(np.zeros(act_dim, dtype=np.float32))
            returns_to_go.append(
                returns_to_go[-1] - float(info.get("undiscounted_reward", reward))
            )
            timesteps.append(decision_steps)

    raw_steps = int(info.get("raw_simulation_steps", decision_steps))
    success = bool(info.get("is_success", False))
    return EpisodeEvaluationRecord(
        episode=episode,
        seed=seed,
        episode_return=episode_return,
        decision_steps=decision_steps,
        environment_steps=decision_steps,
        raw_steps=raw_steps,
        completion_time_seconds=raw_steps * 0.1 if success else None,
        success=success,
        collision=bool(info.get("collision", False)),
        off_route=bool(info.get("off_route", False)),
        timeout=bool(info.get("max_time", False)),
        traffic_variant=(
            str(info["traffic_variant"])
            if info.get("traffic_variant") is not None
            else None
        ),
    ).to_dict()


def evaluate_sequence_baseline(task, device, episodes):
    """Return-to-go-conditioned rollout for Hybrid-DT / Decision Transformer,
    and latent-policy rollout for HyAR.

    Sequence baselines are not SB3 models, so they run a dedicated
    autoregressive evaluation loop instead of ``model.predict``.  The recorded
    ``EpisodeEvaluationRecord`` fields match ``evaluate_model_detailed`` exactly
    so the curve report can consume them unchanged.
    """
    import argparse as _argparse

    import numpy as np
    import torch

    from algos.transformer_rl import SumoStateTokenizer
    from algos.transformer_rl.offline_dataset import hybrid_action_to_vector
    from configs.sb3_configs_baselines import make_sequence_baseline
    from tools.paper_evaluation_contract import make_injected_evaluation_env

    method, scene, raw_step = task["method"], task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    contract = BASELINE_ENV_CONTRACT[method]

    if contract == "v4_8":
        namespace = "ec" + sha256(checkpoint)[:10]
        env = make_hold35k_env(scene, namespace)
    else:
        factory = make_mst_env_factory(scene)
        env = make_injected_evaluation_env(
            factory,
            _argparse.Namespace(
                scenario=scene,
                history_steps=10,
                neighbors=5,
                path_length=10,
                action_repeat=3,
                discount=0.99,
                ego_control_profile="direct",
                episode_limit_profile="source",
                gui=False,
            ),
        )

    tokenizer = SumoStateTokenizer(env.observation_space)
    state_dim = tokenizer.state_dim

    payload = torch.load(checkpoint, map_location="cpu")
    model = make_sequence_baseline(method, state_dim, act_dim=payload.get("act_dim"))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    target_return = float(payload.get("target_return", 500.0))
    act_dim = int(payload.get("act_dim", 2))

    records = []
    try:
        for index in range(episodes):
            episode_seed = SEED_START[method] + index
            np.random.seed(episode_seed + 600000)
            torch.manual_seed(episode_seed + 600000)
            if contract == "v4_8":
                env._traffic_episode_index, env._traffic_roll = index, None
            records.append(
                _rollout_sequence_episode(
                    model,
                    env,
                    tokenizer,
                    method,
                    act_dim,
                    target_return,
                    device,
                    episode=index,
                    seed=episode_seed,
                )
            )
    finally:
        env.close()
    return records


BASELINE_EVALUATORS = {
    "hsac_mlp": evaluate_hybrid_sac_baseline,
    "hsac_lstm": evaluate_hybrid_sac_baseline,
    "gnn_sac": evaluate_continuous_sac_baseline,
    "mst": evaluate_continuous_sac_baseline,
    "hybrid_dt": evaluate_sequence_baseline,
    "decision_transformer": evaluate_sequence_baseline,
    "hyar": evaluate_sequence_baseline,
}


def build_baseline_tasks():
    """Build evaluation tasks for the six comparison baselines."""
    tasks = []
    for scene in SCENES:
        for method in BASELINE_METHODS:
            for raw_step in RAW_STEPS:
                tasks.append(
                    dict(
                        method=method,
                        scene=scene,
                        raw_step=raw_step,
                        checkpoint=str(baseline_checkpoint(method, scene, raw_step)),
                        output=str(
                            EVAL_CURVE_ROOT
                            / f"{method}__{scene}"
                            / f"raw_{raw_step}"
                            / "result.json"
                        ),
                    )
                )
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run-task", "run", "status"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks()
    if args.command == "list":
        for i, task in enumerate(tasks):
            print(i, task["method"], task["scene"], task["raw_step"], task["checkpoint"])
        return
    if args.command == "run-task":
        if args.index is None or not (0 <= args.index < len(tasks)):
            parser.error("run-task requires --index in range")
        result = run_task(tasks[args.index], args.device, smoke=args.smoke)
        if result:
            print(result)
        return
    if args.command == "run":
        controller(args.workers, smoke=args.smoke)
        return
    if args.command == "status":
        for name in ("controller_progress.json", "controller.json"):
            path = EVAL_CURVE_ROOT / name
            print(name, read(path) if path.exists() else "not started")
        return


if __name__ == "__main__":
    main()
