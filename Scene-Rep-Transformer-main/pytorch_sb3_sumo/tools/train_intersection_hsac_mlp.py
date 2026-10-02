"""HSAC-MLP 在 intersection 场景上的独立 GPU 训练 + 6-worker 评估 + 绘图。

这是一个**新增**的独立脚本，不改动任何现有文件，且所有实验产物都写入
独立的 ``res_isxn_hsac/`` 目录，与既有 three-scene / v4_8 / baseline 实验
完全区分开。

HSAC-MLP 的定义（复刻项目内 ``algos.hybrid_action.hsac`` 的消融臂）：
  * ``DecisionAlignedSACPolicy`` 混合动作头（离散 3 路车道 + 连续速度），
    环境契约是 v4_8（观测额外暴露 ``lane_action_mask``）；
  * 编码器换成朴素 ``SimpleMlpLstmExtractor(backbone="mlp")``，替换掉
    Scene-Rep 图/Transformer 表示；
  * 关闭辅助 SLT 目标（``representation_coef=0.0``）。

因此"纯 SAC" = 混合动作 SAC 去掉场景表示，只保留朴素 MLP 编码器，是本项目
已有的强基线 ``hsac_mlp``。

训练规模（用户指定）：20,000 个 raw 时间步（SUMO 0.1 s 步），每 200 步保存
一个模型；训练完成后以 6 个并发 worker 评估 30 个 episode；最后从
``train_monitor.csv`` 绘制训练曲线。

加速手段（全部通过构造参数实现，不改现有代码）：
  * ``device="cuda"`` GPU 训练；
  * ``batch_size=256``（基线为 32）提升 GPU 并行度；
  * ``learning_rate=3e-4``（基线为 1e-4）加速收敛；
  * ``learning_starts=2000``（基线为 5000）更早进入学习阶段；
  * 主进程 ``torch.set_num_threads(1)`` 减少 CPU 线程竞争。

用法::

    # 完整训练 + 评估 + 绘图（正式）
    python -m tools.train_intersection_hsac_mlp

    # 快速 smoke（几十秒内验证全链路）
    python -m tools.train_intersection_hsac_mlp --smoke

    # 单独重跑评估（需要一个已训练好的 final_model.zip）
    python -m tools.train_intersection_hsac_mlp --eval-only \
        --model-path <run_dir>/final_model.zip --output-dir <run_dir>

评估 worker 通过 ``python -m tools.train_intersection_hsac_mlp eval-worker``
以独立子进程运行（subprocess + ``-m`` 模块形式，规避 Windows 上
multiprocessing spawn 依赖主模块 ``__file__`` 的问题）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 实验配置（集中在此，便于审查与复现）
# --------------------------------------------------------------------------- #
SCENARIO = "intersection"
METHOD = "hsac_mlp"

# 训练规模（用户指定：2w 时间步，每 200 步保存）。
RAW_TRAINING_STEPS = 20_000
CHECKPOINT_FREQUENCY = 200
# 加速训练：更早进入学习阶段（基线 5000 -> 2000）。
LEARNING_STARTS = 2_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
BUFFER_SIZE = 20_000
DISCOUNT = 0.99
ACTION_REPEAT = 3
SEED = 0

# 评估（用户指定：6 worker，30 次）。
EVAL_EPISODES = 30
EVAL_WORKERS = 6
EVAL_SEED_START = 420_000

# 独立输出根目录（与既有 three_scene / baseline 实验彻底分开）。
# 目录名刻意取短：项目本已位于较深的 Windows 路径下，high-density overlay 的
# manifest 文件名又很长，路径过深会触及 MAX_PATH(260) 边界导致 os.replace
# 失败，故用 "isxn" 缩写 intersection、压缩 run 目录名以留出余量。
RESULT_ROOT = PROJECT_ROOT / "res_isxn_hsac"

# intersection 原始 traffic 已经是实验密度（满速抢必撞、让行可过），因此
# 用 vehicle_scale=1.0 的空叠加层直接跑 base demand，不再额外克隆车辆。
DENSITY = {
    "vehicle_scale": 1.0,
    "pedestrian_scale": 1.0,
    "clone_depart_jitter_seconds": (0.0, 0.0),
}


# --------------------------------------------------------------------------- #
# 环境工厂（自包含：不写 three_scene 的 overlay 目录）
# --------------------------------------------------------------------------- #
def _environment_namespace() -> argparse.Namespace:
    """与 ``_make_environment_factory`` 消费的 namespace 一致。"""
    return argparse.Namespace(
        scenario=SCENARIO,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=ACTION_REPEAT,
        discount=DISCOUNT,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split="validation",
    )


def make_intersection_environment(
    *, evaluation: bool, overlay_root: Path
):
    """构造 v4_8 契约的 intersection 环境（含 ``lane_action_mask``）。

    v4_8 契约的物理流量分区沿用项目惯例：训练用 train 分区，评估用
    evaluation 分区；契约标签在评估时为 "validation"。
    """
    from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
        IndependentV2FiveBySixEnvV4V1,
    )

    if not evaluation:
        physical_partition = "train"
        contract_partition = "train"
    else:
        physical_partition = "evaluation"
        contract_partition = "validation"

    return IndependentV2FiveBySixEnvV4V1(
        scenario=SCENARIO,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=ACTION_REPEAT,
        reward_discount=DISCOUNT,
        ego_control_profile="direct",
        include_state_lstm=False,
        state_lstm_only=False,
        episode_limit_profile="source",
        render_mode=None,
        high_density_vehicle_scale=float(DENSITY["vehicle_scale"]),
        high_density_pedestrian_scale=float(DENSITY["pedestrian_scale"]),
        high_density_clone_jitter_seconds=tuple(
            float(value) for value in DENSITY["clone_depart_jitter_seconds"]
        ),
        high_density_overlay_root=overlay_root,
        high_density_partition=physical_partition,
        high_density_contract_partition=contract_partition,
    )


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training(run_dir: Path, smoke: bool) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.hybrid_action.hsac import make_hsac_model
    from algos.sb3_torch.callbacks import RawStepControlCallback

    torch.set_num_threads(1)

    raw_budget = 300 if smoke else RAW_TRAINING_STEPS
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    learning_starts = 60 if smoke else LEARNING_STARTS
    overlay_root = run_dir / "overlays"
    namespace = "sm" if smoke else "tr"

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None
    model = None

    _write_json_atomic(
        run_dir / "status.json",
        dict(status="training", pid=os.getpid(), smoke=smoke, started_at=started),
    )

    try:
        env = Monitor(
            make_intersection_environment(
                evaluation=False, overlay_root=overlay_root / f"ns_{namespace}"
            ),
            filename=str(run_dir / "train_monitor.csv"),
            info_keywords=(
                "raw_simulation_steps",
                "is_success",
                "collision",
                "off_route",
                "max_time",
            ),
        )
        model = make_hsac_model(
            METHOD,
            env,
            scenario=SCENARIO,
            device="cuda",
            seed=SEED,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            learning_starts=learning_starts,
            buffer_size=BUFFER_SIZE,
            discount=DISCOUNT,
            action_repeat=ACTION_REPEAT,
            tensorboard_log=str(run_dir / "tensorboard_log"),
            verbose=1,
        )

        _write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method=METHOD,
                scenario=SCENARIO,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                discount=DISCOUNT,
                action_repeat=ACTION_REPEAT,
                seed=SEED,
                device="cuda",
                density=DENSITY,
                smoke=smoke,
                env_contract="v4_8",
            ),
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    _write_json_atomic(
                        run_dir / "progress.json",
                        dict(
                            raw_steps=getattr(self.model, "_raw_steps_seen", None),
                            updates=self.model._n_updates,
                            updated_at=time.time(),
                        ),
                    )
                return True

        model.learn(
            total_timesteps=raw_budget,
            callback=CallbackList(
                [
                    RawStepControlCallback(
                        raw_step_budget=raw_budget,
                        checkpoint_frequency=frequency,
                        checkpoint_path=run_dir / "checkpoints",
                        checkpoint_prefix="hsac_mlp_intersection",
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != raw_budget:
            raise AssertionError(
                f"raw-step budget mismatch: {model._raw_steps_seen} != {raw_budget}"
            )

        final = run_dir / "final_model.zip"
        model.save(final)
        _write_json_atomic(run_dir / "training_diagnostics.json", model.training_diagnostics())
        _write_json_atomic(
            run_dir / "training_complete.json",
            dict(
                smoke=smoke,
                checkpoint_sha256=_sha256(final),
                raw_steps=raw_budget,
                updates=model._n_updates,
                replay_size=model.replay_buffer.size(),
                wall_seconds=time.time() - started,
            ),
        )
        _write_json_atomic(
            run_dir / "status.json",
            dict(status="trained", smoke=smoke, wall_seconds=time.time() - started),
        )
        return final
    except BaseException as exc:
        _write_json_atomic(
            run_dir / "status.json",
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
# 评估（6 worker 子进程，30 episode）
# --------------------------------------------------------------------------- #
def run_eval_worker(args: argparse.Namespace) -> None:
    """单个评估 worker：评估 ``[start_episode, end_episode)`` 这段 episode。

    以 ``python -m ... eval-worker`` 独立子进程运行；重依赖在此延迟导入，
    使子进程顶层 import 保持轻量。
    """
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.v48_stability_v4.model import use_actor_only

    torch.set_num_threads(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_root = output_dir / "overlays" / f"eval_w{args.worker_id}"
    namespace = f"ecw{args.worker_id}"

    env = make_intersection_environment(evaluation=True, overlay_root=overlay_root)
    model = use_actor_only(
        SceneRepresentationSAC.load(
            str(args.model_path), env=env, device="cpu", buffer_size=32
        )
    )
    records = []
    try:
        for index in range(args.start_episode, args.end_episode):
            episode_seed = EVAL_SEED_START + index
            np.random.seed(episode_seed + 600_000)
            torch.manual_seed(episode_seed + 600_000)
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
            rec = detailed.episode_records[0].to_dict()
            rec["episode"] = index  # 回填全局 episode 序号（evaluate_model_detailed 内部恒写 0）
            records.append(rec)
    finally:
        env.close()

    _write_json_atomic(
        output_dir / f"eval_worker_{args.worker_id:02d}.json",
        dict(
            worker_id=args.worker_id,
            start_episode=args.start_episode,
            end_episode=args.end_episode,
            model_path=str(args.model_path),
            records=records,
        ),
    )
    print(
        json.dumps(
            dict(
                worker_id=args.worker_id,
                episodes=len(records),
                start=args.start_episode,
                end=args.end_episode,
            )
        ),
        flush=True,
    )


def summarize_records(records: list[dict]) -> dict:
    import numpy as np

    n = len(records)
    returns = [r["episode_return"] for r in records]
    raw_lengths = [r["raw_steps"] for r in records]
    decision_lengths = [r["decision_steps"] for r in records]
    completion_times = [
        r["completion_time_seconds"]
        for r in records
        if r.get("completion_time_seconds") is not None
    ]
    successes = sum(int(r["success"]) for r in records)
    return dict(
        episodes=n,
        success_rate=successes / n if n else 0.0,
        collision_rate=sum(int(r["collision"]) for r in records) / n if n else 0.0,
        off_route_rate=sum(int(r["off_route"]) for r in records) / n if n else 0.0,
        timeout_rate=sum(int(r["timeout"]) for r in records) / n if n else 0.0,
        mean_return=float(np.mean(returns)) if n else None,
        std_return=float(np.std(returns)) if n else None,
        mean_decision_steps=float(np.mean(decision_lengths)) if n else None,
        mean_raw_steps=float(np.mean(raw_lengths)) if n else None,
        mean_success_completion_time_seconds=(
            float(np.mean(completion_times)) if completion_times else None
        ),
        std_success_completion_time_seconds=(
            float(np.std(completion_times)) if completion_times else None
        ),
        successful_episodes=successes,
    )


def run_evaluation(run_dir: Path, final_model: Path, smoke: bool) -> dict:
    workers = 2 if smoke else EVAL_WORKERS
    episodes = 4 if smoke else EVAL_EPISODES
    device = "cpu"

    per_worker = episodes // workers
    remainder = episodes % workers
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for worker_id in range(workers):
        count = per_worker + (1 if worker_id < remainder else 0)
        ranges.append((cursor, cursor + count))
        cursor += count

    log_dir = run_dir / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, subprocess.Popen] = {}
    stop_requested = threading.Event()
    started = time.time()

    def execute(worker_id: int, start: int, end: int):
        if stop_requested.is_set():
            return dict(worker_id=worker_id, exit_code=-1, cancelled=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = log_dir / f"worker_{worker_id:02d}_{stamp}.log"
        command = [
            sys.executable,
            "-m",
            "tools.train_intersection_hsac_mlp",
            "eval-worker",
            "--worker-id",
            str(worker_id),
            "--start-episode",
            str(start),
            "--end-episode",
            str(end),
            "--model-path",
            str(final_model),
            "--output-dir",
            str(run_dir),
        ]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            running[str(worker_id)] = process
            returncode = process.wait()
            running.pop(str(worker_id), None)
        return dict(
            worker_id=worker_id, exit_code=returncode, log=str(log), completed_at=time.time()
        )

    done: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execute, wid, start, end): wid
            for wid, (start, end) in enumerate(ranges)
        }
        try:
            for future in as_completed(futures):
                wid = futures[future]
                done[wid] = future.result()
                print(f"eval worker {wid}: exit {done[wid]['exit_code']}", flush=True)
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

    failed = [wid for wid, result in done.items() if result["exit_code"] != 0]
    if failed:
        raise RuntimeError(f"evaluation workers failed: {failed}; inspect {log_dir}")

    # 汇总所有 worker 的 episode records（按全局 episode 序号排序）。
    records: list[dict] = []
    for wid in sorted(done):
        path = run_dir / f"eval_worker_{wid:02d}.json"
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    records.sort(key=lambda r: r["episode"])
    summary = summarize_records(records)

    result = dict(
        identity=dict(
            method=METHOD,
            scenario=SCENARIO,
            checkpoint=str(final_model),
            checkpoint_sha256=_sha256(final_model),
            episodes=episodes,
            workers=workers,
            device=device,
            seed_start=EVAL_SEED_START,
            deployment="actor_deterministic",
            smoke=smoke,
        ),
        summary=summary,
        episode_records=records,
        wall_seconds=time.time() - started,
    )
    _write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_training_curves(run_dir: Path) -> Path:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    monitor_csv = run_dir / "train_monitor.csv"
    if not monitor_csv.is_file():
        raise FileNotFoundError(monitor_csv)

    rows = []
    with monitor_csv.open(newline="") as handle:
        # Monitor 在 CSV 头前写一行以 ``#`` 开头的元数据注释，需跳过。
        lines = [line for line in handle if not line.startswith("#")]
        rows = list(csv.DictReader(lines))
    if not rows:
        raise RuntimeError("train_monitor.csv is empty")

    episode = np.arange(1, len(rows) + 1)
    returns = np.asarray([float(r["r"]) for r in rows], dtype=float)
    # Monitor 把 bool 信息字段写成 Python 字符串 "True"/"False"。
    successes = np.asarray(
        [
            1.0 if str(r.get("is_success", "")).strip().lower() == "true" else 0.0
            for r in rows
        ],
        dtype=float,
    )

    def ema(values: np.ndarray, span: int) -> np.ndarray:
        span = max(1, min(span, len(values)))
        alpha = 2.0 / (span + 1)
        out = np.empty_like(values, dtype=float)
        acc = values[0]
        for i, v in enumerate(values):
            acc = alpha * v + (1.0 - alpha) * acc
            out[i] = acc
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(episode, returns, color="#1f77b4", alpha=0.35, lw=1.0, label="episode return")
    axes[0].plot(episode, ema(returns, 10), color="#1f77b4", lw=2.0, label="return (EMA 10)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("episode return")
    axes[0].set_title("HSAC-MLP on intersection — episode return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    window = 10
    if len(rows) >= window:
        success_rate = np.convolve(successes, np.ones(window) / window, mode="valid")
        axes[1].plot(
            episode[window - 1 :], success_rate, color="#ff7f0e", lw=2.0,
            label=f"success rate (window {window})",
        )
    else:
        # 训练步数太少（如 smoke）时 episode 数不足一个窗口，直接画原始值。
        axes[1].plot(
            episode, successes, color="#ff7f0e", marker="o", lw=1.5,
            label="success (per episode)",
        )
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("success rate")
    axes[1].set_title("HSAC-MLP on intersection — success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"HSAC-MLP / intersection / {RAW_TRAINING_STEPS} raw steps", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = run_dir / "training_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_full(smoke: bool) -> Path:
    if not smoke and not _cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"smoke_{stamp}" if smoke else f"seed{SEED}_{stamp}"
    run_dir = RESULT_ROOT / tag

    final = run_training(run_dir, smoke=smoke)
    result = run_evaluation(run_dir, final, smoke=smoke)
    curve = plot_training_curves(run_dir)

    manifest = dict(
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve),
        smoke=smoke,
        raw_training_steps=RAW_TRAINING_STEPS,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
    )
    _write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return run_dir


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("eval-worker",),
        default=None,
        help="留空表示训练主流程；``eval-worker`` 为内部子进程入口",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    # eval-worker 子命令参数
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    args = parser.parse_args(argv)

    if args.command == "eval-worker":
        # eval-worker 子进程入口。
        if args.worker_id is None or args.start_episode is None or args.end_episode is None:
            parser.error(
                "eval-worker requires --worker-id, --start-episode, --end-episode"
            )
        if args.model_path is None or args.output_dir is None:
            parser.error("eval-worker requires --model-path and --output-dir")
        run_eval_worker(args)
        return 0

    if args.eval_only:
        if args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --model-path and --output-dir")
        run_dir = Path(args.output_dir)
        result = run_evaluation(run_dir, Path(args.model_path), smoke=args.smoke)
        plot_training_curves(run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    run_full(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
