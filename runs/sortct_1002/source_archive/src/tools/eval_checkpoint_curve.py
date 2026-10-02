"""Post-hoc checkpoint evaluation: common-RL eval-success-rate vs raw steps.

Evaluates existing intermediate checkpoints (no retraining) for the screened
v4_8/lr_half candidates plus the MST+SLT baseline, on cross+carla, at raw steps
10k/20k/30k/40k.  The already-known 50k final point is spliced in at plot time.

Each family is evaluated in its own environment contract:
  * v4_8:    tools.v48_stability_v4 make_env + actor-only load, 100 eps seed block 420000.
  * mst_slt: high_density_single_seed_100ep_v2 base adapter env, 100 eps seed 10000.

Evaluation is inference-only (no gradient updates), so the controller defaults
to 6 concurrent GPU workers (configurable) sharing one GPU; the replay buffer is
loaded at capacity 32 because inference never touches it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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

ROOT = PROJECT_ROOT
SCENES = ("cross", "carla")
RAW_STEPS = (10000, 20000, 30000, 40000)
V48_CANDIDATES = (
    "lr_half",
    "tau0025_floor2e5_hold35k",
    "tau0025_floor2e5_hold40k",
    "tau0025_floor2e5_hold30k",
    "tau0025_floor2e5_hold45k",
)
EVAL_EPISODES = 100
EVAL_CURVE_ROOT = ROOT / "r48s4" / "eval_curve"
MST_EVAL_CURVE_ROOT = ROOT / "results_hd_ss100_v2" / "comparison" / "eval_curve"
MST_OVERLAY_ROOT = (
    ROOT
    / "results_hd_ss100_v2"
    / "comparison"
    / "overlays_hd_ss100_v2"
    / "mst_slt"
    / "seed_0"
)
V2_PROTOCOL_PATH = (
    ROOT / "experiments" / "high_density_single_seed_100ep_v2" / "protocol.json"
)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def seal(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable artifact differs: {path}")
    else:
        write(path, value)


def v48_checkpoint(scene, candidate, raw_step):
    if candidate == "lr_half":
        return (
            ROOT
            / "results_phase2_runtime_v2"
            / "screen"
            / f"v4_8__{scene}__lr_half__seed0"
            / "checkpoints"
            / f"ckpt_raw_{raw_step}_steps.zip"
        )
    return (
        ROOT
        / "r48s4"
        / "train"
        / f"{scene}__{candidate}"
        / "checkpoints"
        / f"ckpt_raw_{raw_step}_steps.zip"
    )


def mst_checkpoint(scene, raw_step):
    return (
        ROOT
        / "results_hd_ss100_v2"
        / "comparison"
        / "runs"
        / f"hd_ss100_v2__comparison__mst_slt__{scene}__seed0"
        / "checkpoints"
        / f"scene_rep_raw_{raw_step}_steps.zip"
    )


def build_tasks():
    tasks = []
    for scene in SCENES:
        for candidate in V48_CANDIDATES:
            for raw_step in RAW_STEPS:
                tasks.append(
                    dict(
                        family="v4_8",
                        scene=scene,
                        candidate=candidate,
                        raw_step=raw_step,
                        checkpoint=str(v48_checkpoint(scene, candidate, raw_step)),
                        output=str(
                            EVAL_CURVE_ROOT
                            / f"{scene}__{candidate}"
                            / f"raw_{raw_step}"
                            / "result.json"
                        ),
                    )
                )
        for raw_step in RAW_STEPS:
            tasks.append(
                dict(
                    family="mst_slt",
                    scene=scene,
                    candidate="mst_slt",
                    raw_step=raw_step,
                    checkpoint=str(mst_checkpoint(scene, raw_step)),
                    output=str(
                        MST_EVAL_CURVE_ROOT
                        / f"mst_slt__{scene}"
                        / f"raw_{raw_step}"
                        / "result.json"
                    ),
                )
            )
    return tasks


def task_name(task):
    return f"{task['family']}__{task['scene']}__{task['candidate']}__raw{task['raw_step']}"


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


def evaluate_v4_8(task, device, episodes):
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from tools.v48_stability_v4.common import make_env
    from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    namespace = "ec" + sha(checkpoint)[:10]
    env = make_env(scene, namespace)
    model = use_actor_only(
        StabilitySAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
    )
    try:
        if model._raw_steps_seen != raw_step:
            raise ValueError(
                f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}"
            )
        records = []
        for index in range(episodes):
            episode_seed = 420000 + index
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
    from tools.high_density_same_scene_v1_common import density_setting, load_protocol
    from tools.paper_evaluation_contract import make_injected_evaluation_env
    from tools.train_high_density_single_seed_100ep_v2 import (
        _make_v2_environment_factory,
    )

    scene, raw_step = task["scene"], task["raw_step"]
    checkpoint = Path(task["checkpoint"])
    protocol = load_protocol(V2_PROTOCOL_PATH)
    density = density_setting(protocol, scene)
    factory = _make_v2_environment_factory(
        adapter="base", density=density, overlay_root=MST_OVERLAY_ROOT
    )
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
            raise ValueError(
                f"_raw_steps_seen={model._raw_steps_seen} != {raw_step}"
            )
        report = evaluate_model_detailed(
            model, env, episodes=episodes, seed=10000, policy_action_hold=1
        )
        records = [r.to_dict() for r in report.episode_records]
    finally:
        env.close()
    return records


def run_task(task, device, smoke=False):
    import torch

    torch.set_num_threads(1)
    episodes = 1 if smoke else EVAL_EPISODES

    identity = None
    if not smoke:
        identity = dict(
            family=task["family"],
            scene=task["scene"],
            candidate=task["candidate"],
            raw_step=task["raw_step"],
            checkpoint=task["checkpoint"],
            checkpoint_sha256=sha(task["checkpoint"]),
            episodes=episodes,
            seed_start=(420000 if task["family"] == "v4_8" else 10000),
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
    if task["family"] == "v4_8":
        records = evaluate_v4_8(task, device, episodes)
    else:
        records = evaluate_mst_slt(task, device, episodes)

    if smoke:
        print(
            json.dumps(
                dict(
                    family=task["family"],
                    scene=task["scene"],
                    candidate=task["candidate"],
                    raw_step=task["raw_step"],
                    _raw_steps_seen_assert_passed=True,
                    episode=records[0],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return None

    result = dict(identity=identity, **summarize(records), wall_seconds=time.time() - started)
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
            "tools.eval_checkpoint_curve",
            "run-task",
            "--index",
            str(index),
        ]
        if smoke:
            command.append("--smoke")
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
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
            print(i, task["family"], task["scene"], task["candidate"], task["raw_step"], task["checkpoint"])
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
