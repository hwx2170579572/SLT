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
    EVAL_CURVE_ROOT,
    EVAL_EPISODES,
    RAW_STEPS,
    RESULT_ROOT,
    SCENES,
    hold35k_checkpoint,
    make_hold35k_env,
    make_mst_env_factory,
    mst_checkpoint,
    read,
    seal,
    sha256,
    write,
)

METHODS = ("hold35k", "mst_slt")
SEED_START = {"hold35k": 420000, "mst_slt": 10000}


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
    else:
        records = evaluate_mst_slt(task, device, episodes)

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
