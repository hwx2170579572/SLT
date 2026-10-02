"""hold35k + MST+SLT 在 intersection 场景上的独立 GPU 训练 + 6-worker 评估 + 绘图。

这是一个**新增**的独立脚本，不改动任何现有文件；所有实验产物写入独立的
``fast-developer/`` 目录，与既有 three-scene / v4_8 / baseline / hsac_mlp 实验
完全区分开。

两个方法（均来自 ``tools/three_scene_hold35k_vs_mst_v1`` 的 ``METHODS``）：

  * ``hold35k``  -- TASAC：v4_8 混合动作头（离散 3 路车道 + 连续速度）+
                   Scene-Rep 表示 + SLT 辅助目标；候选配置
                   ``tau0025_floor2e5_hold35k``（``ConfidentActorFusionSACV45``）。
  * ``mst_slt``   -- MST+SLT：base 连续动作头（Box(2)）+ Scene-Rep 表示 +
                   SLT 辅助目标（``SceneRepresentationSAC`` + ``scene_rep``）。

训练规模（用户指定）：两个方法**并行**训练（2 个训练 worker，各一个独立
子进程、共享 GPU 显存），每个方法 50,000 个 raw 时间步（SUMO 0.1 s 步），
每 200 步保存一个模型；训练完成后每个方法以 6 个并发 worker **总共评估 100
回合**；最后从 ``train_monitor.csv`` 绘制训练曲线。

**修复版**（相对同目录下 ``train_intersection_hold35k_mst.py`` 的加速版）：

首次加速训练（batch=256 / lr=3e-4 常量 / learning_starts=2000）导致 SAC 过早
坍缩为「避碰不动→超时」局部最优（ent_coef 23× 衰减、episode 12 起全 timeout），
根因分析见 ``fast-developer/analysis_hold35k_mst_low_success.md``。本版恢复方法
默认超参，仅保留 GPU 与线程优化：

  * ``device="cuda"`` GPU 训练（保留）；
  * ``batch_size=32``（恢复方法默认，原加速值 256 使梯度噪声消失、锁死首个吸引子）；
  * ``learning_rate`` 按方法恢复：hold35k=5e-5（含深地板衰减 5e-5→2e-5）、
    mst_slt=1e-4（原加速值 3e-4 常量移除了深地板衰减，探索过早耗尽）；
  * ``learning_starts=5000``（恢复方法默认，原加速值 2000 时 buffer 未满即学习）；
  * 主进程 ``torch.set_num_threads(1)`` 减少 CPU 线程竞争（保留）。

用法::

    # 完整训练 + 评估 + 绘图（正式；launcher 并行启动 2 个训练 worker）
    python tools/train_intersection_hold35k_mst.py

    # 快速 smoke（几分钟内验证全链路）
    python tools/train_intersection_hold35k_mst.py --smoke

    # 单独重跑某方法的评估
    python tools/train_intersection_hold35k_mst.py --eval-only \
        --method hold35k --model-path <run_dir>/final_model.zip --output-dir <run_dir>

子进程通过 ``subprocess`` + ``__file__`` 以独立进程运行（``train-method`` 训练
流水线、``eval-worker`` 评估），脚本位置无关——从 ``tools/`` 或 ``fast-developer/``
运行等价。
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 实验配置（集中在此，便于审查与复现）
# --------------------------------------------------------------------------- #
SCENARIO = "intersection"
METHODS = ("hold35k", "mst_slt")

# 训练规模（用户指定：5w 时间步，每 200 步保存）。
RAW_TRAINING_STEPS = 50_000
CHECKPOINT_FREQUENCY = 200
# 修复：恢复方法默认超参（原加速版 LEARNING_STARTS=2000 / BATCH_SIZE=256 /
# LEARNING_RATE=3e-4 常量，导致探索过早坍缩）。
LEARNING_STARTS = 5_000
BATCH_SIZE = 32
# lr 按方法恢复默认：hold35k=5e-5（含深地板衰减 5e-5→2e-5），mst_slt=1e-4。
LEARNING_RATE = {
    "hold35k": 5e-5,
    "mst_slt": 1e-4,
}
BUFFER_SIZE = 20_000
DISCOUNT = 0.99
ACTION_REPEAT = 3
SEED = 0

# 评估（用户指定：每方法总共 100 回合，6 worker 并行分摊）。
EVAL_EPISODES_TOTAL = 100
EVAL_WORKERS = 6
# 训练并行度（用户指定：2 个训练 worker 并行，各占一个 GPU 槽位）。
TRAIN_WORKERS = 2
SEED_START = {
    "hold35k": 420_000,
    "mst_slt": 10_000,
}

# hold35k 的方法候选配置（保持 tau=0.0025 的深地板特征；加速只动 lr/batch/
# warmup，不动 tau）。
HOLD35K_CANDIDATE = "tau0025_floor2e5_hold35k"

# 独立输出根目录（用户指定：fast-developer）。
RESULT_ROOT = PROJECT_ROOT / "fast-developer"

# 修复版 run 目录后缀：与加速版（hold35k__intersection / mst_slt__intersection）
# 的结果目录区分，避免覆盖首次失败基线。
RUN_SUFFIX = "_fixed"

# intersection 原始 traffic 已经是实验密度（满速抢必撞、让行可过），因此
# 用 vehicle_scale=1.0 的空叠加层直接跑 base demand，不再额外克隆车辆。
DENSITY = {
    "vehicle_scale": 1.0,
    "pedestrian_scale": 1.0,
    "clone_depart_jitter_seconds": (0.0, 0.0),
}


# --------------------------------------------------------------------------- #
# 环境工厂（自包含：复用项目内 _make_environment_factory，overlay 写到本脚本
# 的 fast-developer 目录，不写 three_scene 的 overlay 目录）
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


def make_env_factory(adapter: str, overlay_root: Path):
    """返回 ``make_environment(args, *, evaluation=False)``。

    ``adapter`` 为 "v4_8"（hold35k，混合动作头 + lane_action_mask）或
    "base"（mst_slt，连续 Box(2) 动作头）。
    """
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory

    return _make_environment_factory(
        adapter=adapter,
        density=DENSITY,
        overlay_root=overlay_root,
    )


# --------------------------------------------------------------------------- #
# 通用文件工具
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


# --------------------------------------------------------------------------- #
# 训练：hold35k（TASAC, v4_8）
# --------------------------------------------------------------------------- #
def run_training_hold35k(run_dir: Path, smoke: bool) -> Path:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch.callbacks import RawStepControlCallback
    from tools.phase2_model_factory_v1 import make_phase2_model, verify_optimizer_settings
    from tools.v48_stability_v4.common import CANDIDATES
    from tools.v48_stability_v4.model import StabilitySAC

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
            make_env_factory("v4_8", overlay_root / f"ns_{namespace}")(
                _environment_namespace(), evaluation=False
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

        # 内联 make_model 的逻辑，覆盖 batch_size / learning_rate / learning_starts
        # 以加速（保持 tau=0.0025 与 buffer_size 不变）。
        original_config = dict(CANDIDATES[HOLD35K_CANDIDATE])
        config = dict(original_config)
        # 修复：不再覆盖 learning_rate / final_learning_rate，保留候选配置的
        # 深地板衰减（lr 5e-5 -> 2e-5，decay 20000 -> 35000）；仅 batch_size 用常量。
        config.update(batch_size=BATCH_SIZE)
        model = make_phase2_model(
            "v4_8",
            env,
            learning_rate=config["learning_rate"],
            tau=config["tau"],
            scenario=SCENARIO,
            batch_size=config["batch_size"],
            learning_starts=learning_starts,
            buffer_size=config["buffer_size"],
            action_repeat=ACTION_REPEAT,
            seed=SEED,
            device="cuda",
            verbose=0,
        )
        model.__class__ = StabilitySAC
        model.stability_config = config
        model.stability_trace_path = str(run_dir / "optimization_trace.jsonl")

        _write_json_atomic(
            run_dir / "arguments.json",
            dict(
                method="hold35k",
                scenario=SCENARIO,
                candidate=HOLD35K_CANDIDATE,
                original_candidate_config=original_config,
                accelerated_config=config,
                raw_budget=raw_budget,
                checkpoint_frequency=frequency,
                learning_starts_raw_steps=learning_starts,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                tau=config["tau"],
                buffer_size=config["buffer_size"],
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
                        checkpoint_prefix="ckpt",
                    ),
                    Progress(),
                ]
            ),
        )
        if model._raw_steps_seen != raw_budget:
            raise AssertionError(
                f"raw-step budget mismatch: {model._raw_steps_seen} != {raw_budget}"
            )
        if model._n_updates != raw_budget - learning_starts + 1:
            raise AssertionError(
                f"update-count mismatch: {model._n_updates} != "
                f"{raw_budget - learning_starts + 1}"
            )
        rate = config["learning_rate"]
        verify_optimizer_settings(model, learning_rate=rate, tau=config["tau"])
        model.audit_parameters()

        final = run_dir / "final_model.zip"
        model.save(final)
        _write_json_atomic(
            run_dir / "training_diagnostics.json", model.training_diagnostics()
        )
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
# 训练：mst_slt（MST+SLT, base）—— 复用 train_sb3 的 scene_rep 入口
# --------------------------------------------------------------------------- #
def run_training_mst_slt(run_dir: Path, smoke: bool) -> Path:
    import torch

    torch.set_num_threads(1)

    from tools import train_sb3

    max_steps = 300 if smoke else RAW_TRAINING_STEPS
    learning_starts = 60 if smoke else LEARNING_STARTS
    frequency = 100 if smoke else CHECKPOINT_FREQUENCY
    eval_episodes = 1 if smoke else 100

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    env_factory = make_env_factory("base", run_dir / "overlays" / "seed_0")

    argv = [
        "--algo", "scene_rep",
        "--scenario", SCENARIO,
        "--max-steps", str(max_steps),
        "--learning-starts", str(learning_starts),
        "--checkpoint-freq", str(frequency),
        "--eval-freq", "0",
        "--eval-episodes", str(eval_episodes),
        "--seed", str(SEED),
        "--device", "cuda",
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE["mst_slt"]),
        "--discount", str(DISCOUNT),
        "--buffer-size", str(BUFFER_SIZE),
        "--action-repeat", str(ACTION_REPEAT),
        "--output-dir", str(run_dir.parent.resolve()),
        "--model-name", run_dir.name,
        "--ego-control-profile", "direct",
        "--episode-limit-profile", "source",
        "--traffic-protocol", "frozen_80_20",
    ]
    train_sb3.main(
        argv,
        env_factory=env_factory,
        default_output_dir=run_dir.parent,
        require_paper_evaluation_contract=True,
        tensorboard_log_root=RESULT_ROOT / "tb",
    )
    final = run_dir / "final_model.zip"
    if not final.is_file():
        raise FileNotFoundError(f"train_sb3 did not produce {final}")
    _write_json_atomic(
        run_dir / "status.json",
        dict(status="trained", smoke=smoke, method="mst_slt"),
    )
    return final


# --------------------------------------------------------------------------- #
# 评估（6 worker 子进程，每 worker 100 episode）
# --------------------------------------------------------------------------- #
def run_eval_worker(args: argparse.Namespace) -> None:
    """单个评估 worker，评估 ``[start_episode, end_episode)`` 这段 episode。

    以 ``python -m ... eval-worker`` 独立子进程运行；重依赖在此延迟导入，
    使子进程顶层 import 保持轻量。按 ``--method`` 分派到 hold35k / mst_slt。
    """
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import evaluate_model_detailed

    torch.set_num_threads(1)
    method = args.method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_root = output_dir / "overlays" / f"eval_w{args.worker_id}"
    start = args.start_episode
    end = args.end_episode
    n = end - start
    model_path = str(args.model_path)

    if method == "hold35k":
        from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

        env = make_env_factory("v4_8", overlay_root)(
            _environment_namespace(), evaluation=True
        )
        model = use_actor_only(
            StabilitySAC.load(model_path, env=env, device="cpu", buffer_size=32)
        )
        records = []
        try:
            for index in range(start, end):
                episode_seed = SEED_START["hold35k"] + index
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
                rec["episode"] = index  # 回填全局 episode 序号（内部恒写 0）
                records.append(rec)
        finally:
            env.close()
    else:  # mst_slt
        from algos.sb3_torch.sac import SceneRepresentationSAC

        env = make_env_factory("base", overlay_root)(
            _environment_namespace(), evaluation=True
        )
        model = SceneRepresentationSAC.load(
            model_path, env=env, device="cpu", buffer_size=32
        )
        records = []
        try:
            # mst_slt 一次性评估整段；traffic 轮换由环境内部 _traffic_episode_index
            # 计数器驱动，故在此把起点拨到全局起始 index，避免 6 个 worker 重复
            # 评估相同的 traffic 文件。
            env._traffic_episode_index = start
            env._traffic_roll = None
            detailed = evaluate_model_detailed(
                model,
                env,
                episodes=n,
                seed=SEED_START["mst_slt"] + start,
                deterministic=True,
                sumo_step_seconds=0.1,
                policy_action_hold=1,
            )
            for i, rec in enumerate(detailed.episode_records):
                r = rec.to_dict()
                r["episode"] = start + i  # 回填全局 episode 序号
                records.append(r)
        finally:
            env.close()

    _write_json_atomic(
        output_dir / f"eval_worker_{args.worker_id:02d}.json",
        dict(
            worker_id=args.worker_id,
            method=method,
            start_episode=start,
            end_episode=end,
            model_path=model_path,
            records=records,
        ),
    )
    print(
        json.dumps(
            dict(
                worker_id=args.worker_id,
                method=method,
                episodes=len(records),
                start=start,
                end=end,
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


def run_evaluation(
    method: str, run_dir: Path, final_model: Path, smoke: bool
) -> dict:
    workers = 2 if smoke else EVAL_WORKERS
    episodes = 8 if smoke else EVAL_EPISODES_TOTAL
    device = "cpu"

    # 均匀切分 episode 区间给各 worker（100 回合 / 6 worker => 16~17 回合/worker）。
    ranges: list[tuple[int, int]] = []
    for worker_id in range(workers):
        start = worker_id * episodes // workers
        end = (worker_id + 1) * episodes // workers
        if end > start:
            ranges.append((start, end))

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
            str(Path(__file__).resolve()),
            "eval-worker",
            "--method",
            method,
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
            worker_id=worker_id,
            exit_code=returncode,
            log=str(log),
            completed_at=time.time(),
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
                print(
                    f"[{method}] eval worker {wid}: exit {done[wid]['exit_code']}",
                    flush=True,
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
            method=method,
            scenario=SCENARIO,
            checkpoint=str(final_model),
            checkpoint_sha256=_sha256(final_model),
            episodes=episodes,
            workers=workers,
            episodes_per_worker=[end - start for start, end in ranges],
            device=device,
            seed_start=SEED_START[method],
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
def plot_training_curves(method: str, run_dir: Path) -> Path:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    monitor_csv = run_dir / "train_monitor.csv"
    if not monitor_csv.is_file():
        raise FileNotFoundError(monitor_csv)

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

    title = "hold35k" if method == "hold35k" else "MST+SLT"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        episode, returns, color="#1f77b4", alpha=0.35, lw=1.0, label="episode return"
    )
    axes[0].plot(episode, ema(returns, 10), color="#1f77b4", lw=2.0, label="return (EMA 10)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("episode return")
    axes[0].set_title(f"{title} on intersection — episode return")
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
        axes[1].plot(
            episode, successes, color="#ff7f0e", marker="o", lw=1.5,
            label="success (per episode)",
        )
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("success rate")
    axes[1].set_title(f"{title} on intersection — success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{title} / intersection / {RAW_TRAINING_STEPS} raw steps", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = run_dir / "training_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_train_method(method: str, smoke: bool) -> Path:
    """单个方法的完整流水线：训练 -> 评估 -> 绘图 -> 写 manifest。

    以独立子进程运行（每个方法一个训练 worker，两方法并行时由 ``run_full``
    的 launcher 同时启动，共享同一张 GPU 的显存）。"""
    run_dir = RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}"
    # 注意：不在这里提前 mkdir(run_dir)——train_sb3.main 内部用 exist_ok=False
    # 建目录，提前建会触发 FileExistsError。各训练函数自行创建所需目录。

    if method == "hold35k":
        final = run_training_hold35k(run_dir, smoke=smoke)
    else:
        final = run_training_mst_slt(run_dir, smoke=smoke)

    result = run_evaluation(method, run_dir, final, smoke=smoke)
    curve = plot_training_curves(method, run_dir)

    manifest = dict(
        method=method,
        scenario=SCENARIO,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve),
        smoke=smoke,
        raw_training_steps=RAW_TRAINING_STEPS,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
        train_workers=TRAIN_WORKERS,
        eval_episodes_total=EVAL_EPISODES_TOTAL,
        eval_workers=EVAL_WORKERS,
        hyperparams=dict(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            learning_starts=LEARNING_STARTS,
        ),
    )
    _write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(method=method, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


def run_full(smoke: bool) -> Path:
    """launcher：以 2 个并行子进程分别训练 hold35k 与 mst_slt（各一个训练
    worker，共享 GPU 显存），等待两者完成，汇总顶层 manifest。

    每个方法最多尝试 2 次；首次失败（如 SUMO traci 偶发断连）会清理残留目录
    后重试一次，避免偶发崩溃导致整个实验作废。
    """
    if not smoke and not _cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    # 防静默重启：正式模式下目标目录已存在则拒绝，避免覆盖历史实验产物。
    if not smoke:
        existing = [
            m for m in METHODS if (RESULT_ROOT / f"{m}__{SCENARIO}{RUN_SUFFIX}").exists()
        ]
        if existing:
            raise RuntimeError(
                f"output dirs already exist (remove first): {existing}"
            )

    script = str(Path(__file__).resolve())
    max_attempts = 1 if smoke else 2
    codes: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        todo = [m for m in METHODS if m not in codes or codes[m] != 0]
        if not todo:
            break
        if attempt > 1:
            import shutil

            print(f"[launcher] attempt {attempt}: retrying failed methods {todo}")
            for m in todo:
                shutil.rmtree(RESULT_ROOT / f"{m}__{SCENARIO}{RUN_SUFFIX}", ignore_errors=True)
        procs: dict[str, subprocess.Popen] = {}
        for method in todo:
            cmd = [sys.executable, script, "train-method", "--method", method]
            if smoke:
                cmd.append("--smoke")
            log = RESULT_ROOT / f"{method}{RUN_SUFFIX}__train_launcher.log"
            with log.open("w", encoding="utf-8") as handle:
                procs[method] = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            print(f"[launcher] {method} attempt {attempt} (pid {procs[method].pid})")
        for method in todo:
            codes[method] = procs[method].wait()
            print(f"[launcher] {method} attempt {attempt} exit {codes[method]}")

    per_method: dict = {}
    for method in METHODS:
        manifest_path = (
            RESULT_ROOT / f"{method}__{SCENARIO}{RUN_SUFFIX}" / "experiment_manifest.json"
        )
        if manifest_path.is_file():
            per_method[method] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

    failed = {m: c for m, c in codes.items() if c != 0}
    _write_json_atomic(
        RESULT_ROOT / "experiment_manifest.json",
        dict(
            scenario=SCENARIO,
            methods=per_method,
            exit_codes=codes,
            failed=list(failed),
            raw_training_steps=RAW_TRAINING_STEPS,
            checkpoint_frequency=CHECKPOINT_FREQUENCY,
            train_workers=TRAIN_WORKERS,
            smoke=smoke,
        ),
    )
    print(json.dumps(per_method, ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(
            f"training worker(s) failed after {max_attempts} attempt(s): {failed}; "
            f"inspect {RESULT_ROOT}/*__train_launcher.log"
        )
    return RESULT_ROOT


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("eval-worker", "train-method"),
        default=None,
        help="留空表示 launcher（2 并行训练 worker）；``train-method`` 为单方法"
        "训练流水线子进程入口；``eval-worker`` 为评估子进程入口",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    # eval-worker 子命令参数
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    args = parser.parse_args(argv)

    if args.command == "eval-worker":
        # eval-worker 子进程入口。
        if (
            args.method is None
            or args.worker_id is None
            or args.start_episode is None
            or args.end_episode is None
        ):
            parser.error(
                "eval-worker requires --method, --worker-id, --start-episode, --end-episode"
            )
        if args.model_path is None or args.output_dir is None:
            parser.error("eval-worker requires --model-path and --output-dir")
        run_eval_worker(args)
        return 0

    if args.command == "train-method":
        # 单方法训练流水线子进程入口（由 launcher 并行启动）。
        if args.method is None:
            parser.error("train-method requires --method")
        run_train_method(args.method, smoke=args.smoke)
        return 0

    if args.eval_only:
        if args.method is None or args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --method, --model-path and --output-dir")
        run_dir = Path(args.output_dir)
        result = run_evaluation(
            args.method, run_dir, Path(args.model_path), smoke=args.smoke
        )
        plot_training_curves(args.method, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    run_full(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
