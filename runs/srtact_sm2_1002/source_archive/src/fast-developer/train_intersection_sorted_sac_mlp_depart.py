"""在新副本场景 intersection_sorted 上做 sac_mlp（SAC+MLP）的难度验证。

背景
----
原始 intersection 的 traffic 文件按流向分组、depart 时间在组间回退，SUMO 报
「Route file should be sorted by departure time, ignoring ...」后把左右手车流
（东→西 -E3 -E0、西→东 E0 E3）静默忽略，实际只有北→南（E2 E1）一股车流在跑。
intersection_sorted 副本把 traffic 按 (depart, departLane) 重排后，三股车流都
真正参与仿真——场景难度随之上升。本脚本验证新场景在两种发车间隔下 sac_mlp 的
可学习性。

三组发车间隔（depart 时间），CUDA worker 共享本机 GPU 并行训练：
  - sac_mlp_depart2p5：DEPART_SCALE=2.5，发车间隔×2.5（密度÷2.5）
  - sac_mlp_depart3p0：DEPART_SCALE=3.0，发车间隔×3.0（密度÷3）
  - sac_mlp_depart4p0：DEPART_SCALE=4.0，发车间隔×4.0（密度÷4，理论可完成档）

结果目录 ``fast-developer/{组名}__intersection_sorted``。

用法::

    python fast-developer/train_intersection_sorted_sac_mlp_depart.py --smoke
    python fast-developer/train_intersection_sorted_sac_mlp_depart.py   # 两组并行正式训练
    # 单组（子进程入口 / 手动补跑）：
    python fast-developer/train_intersection_sorted_sac_mlp_depart.py train-method \
        --depart-scale 2.5 --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as yv2

NEW_SCENARIO = "intersection_sorted"
SOURCE_TRAFFIC = (
    PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
    / "intersection_sorted" / "traffic"
)
METHOD = "sac_mlp"

# 组名 -> (DEPART_SCALE, 说明)。组名即 run 目录名的一部分。
GROUPS: list[tuple[str, float, str]] = [
    ("sac_mlp_depart2p5", 2.5, "发车间隔×2.5（密度÷2.5）"),
    ("sac_mlp_depart3p0", 3.0, "发车间隔×3.0（密度÷3）"),
    ("sac_mlp_depart4p0", 4.0, "发车间隔×4.0（密度÷4，理论可完成档）"),
]


def _name_for_scale(scale: float) -> str:
    return f"sac_mlp_depart{str(scale).replace('.', 'p')}"


def _apply_patch(scale: float) -> None:
    """把 yv2 的场景与发车间隔常量指向本实验的配置（仅当前进程内生效）。

    两个 CUDA worker 共享本机 GPU：不隔离 CUDA_VISIBLE_DEVICES，各自子进程里的
    ``device="cuda"`` 都落到默认 cuda:0（RTX 5060 Ti 16GB 跑两个 SAC+MLP 足够）。
    """
    yv2.SCENARIO = NEW_SCENARIO
    yv2.SOURCE_TRAFFIC = SOURCE_TRAFFIC
    yv2.DEPART_SCALE = scale


def _run_dir(global_name: str) -> Path:
    return yv2.RESULT_ROOT / f"{global_name}__{NEW_SCENARIO}"


def evaluate(
    global_name: str, scale: float, run_dir: Path, final_model: Path, smoke: bool
) -> dict:
    """subprocess 并行评估（每个 worker 独立进程，退出自动回收 SUMO）。

    eval-worker 子进程带 ``--depart-scale``，由其 main 分支 ``_apply_patch(scale)``
    恢复发车间隔后调用底层 ``run_eval_worker``。
    """
    workers = 2 if smoke else yv2.EVAL_WORKERS
    episodes = 8 if smoke else yv2.EVAL_EPISODES_TOTAL

    ranges: list[tuple[int, int]] = []
    for worker_id in range(workers):
        start = worker_id * episodes // workers
        end = (worker_id + 1) * episodes // workers
        if end > start:
            ranges.append((start, end))

    script = str(Path(__file__).resolve())
    procs: list[tuple[int, subprocess.Popen, object]] = []
    for wid, (start, end) in enumerate(ranges):
        cmd = [
            sys.executable, script, "eval-worker",
            "--method", METHOD,
            "--depart-scale", str(scale),
            "--worker-id", str(wid),
            "--start-episode", str(start),
            "--end-episode", str(end),
            "--model-path", str(final_model),
            "--output-dir", str(run_dir),
        ]
        handle = (run_dir / f"eval_worker_{wid:02d}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        procs.append((wid, proc, handle))
    for wid, proc, handle in procs:
        code = proc.wait()
        handle.close()
        if code != 0:
            raise RuntimeError(
                f"eval worker {wid} ({global_name}) failed: exit {code}; "
                f"see {run_dir}/eval_worker_{wid:02d}.log"
            )

    records: list[dict] = []
    for wid in range(len(ranges)):
        path = run_dir / f"eval_worker_{wid:02d}.json"
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    records.sort(key=lambda r: r["episode"])
    summary = yv2.summarize_records(records)

    result = dict(
        identity=dict(
            method=METHOD,
            group=global_name,
            depart_scale=scale,
            scenario=NEW_SCENARIO,
            checkpoint=str(final_model),
            checkpoint_sha256=yv2._sha256(final_model),
            episodes=episodes,
            workers=workers,
            smoke=smoke,
        ),
        summary=summary,
        episode_records=records,
    )
    yv2._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def train_and_eval(global_name: str, scale: float, smoke: bool) -> Path:
    run_dir = _run_dir(global_name)
    # 不预先 mkdir：run_training_mlp 自己 mkdir(exist_ok=True)。
    final = yv2.run_training_mlp(METHOD, run_dir, smoke=smoke)
    result = evaluate(global_name, scale, run_dir, final, smoke=smoke)
    curve = yv2.plot_training_curves(METHOD, run_dir)

    manifest = dict(
        method=METHOD,
        group=global_name,
        depart_scale=scale,
        scenario=NEW_SCENARIO,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve) if curve else None,
        smoke=smoke,
        note=(
            "intersection_sorted 副本（三股车流都参与仿真）难度验证：发车间隔 "
            f"×{scale}（发车间隔放大，密度÷{scale}）。"
        ),
    )
    yv2._write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(group=global_name, depart_scale=scale, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


def run_full(smoke: bool) -> Path:
    if not smoke and not yv2._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    root = yv2.RESULT_ROOT  # fast-developer
    root.mkdir(parents=True, exist_ok=True)
    if not smoke:
        existing = [name for name, _, _ in GROUPS if _run_dir(name).exists()]
        if existing:
            raise RuntimeError(f"output dirs already exist (remove first): {existing}")

    script = str(Path(__file__).resolve())
    codes: dict[str, int] = {}

    def run_one(name: str, scale: float) -> tuple[str, int]:
        cmd = [sys.executable, script, "train-method", "--depart-scale", str(scale)]
        if smoke:
            cmd.append("--smoke")
        log = root / f"{name}__train.log"
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).returncode
        return name, returncode

    # 两个 CUDA worker 共享本机 GPU 并行（不隔离 CUDA_VISIBLE_DEVICES）。
    with ThreadPoolExecutor(max_workers=len(GROUPS)) as pool:
        futures = {
            pool.submit(run_one, name, scale): name
            for name, scale, _ in GROUPS
        }
        for future in as_completed(futures):
            name, code = future.result()
            codes[name] = code
            print(f"[launcher] {name} exit {code}", flush=True)

    failed = {n: c for n, c in codes.items() if c != 0}
    if failed:
        raise RuntimeError(
            f"training worker(s) failed: {failed}; inspect {root}/*__train.log"
        )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train-method", "eval-worker"),
        default=None,
        help="留空表示 launcher（两组 CUDA worker 并行）",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument(
        "--depart-scale",
        type=float,
        choices=(2.5, 3.0, 4.0),
        help="发车间隔缩放（2.5 / 3.0 / 4.0，均为降密度）",
    )
    parser.add_argument("--method", type=str, default=METHOD)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "train-method":
        if args.depart_scale is None:
            parser.error("train-method requires --depart-scale")
        _apply_patch(args.depart_scale)
        train_and_eval(_name_for_scale(args.depart_scale), args.depart_scale,
                       smoke=args.smoke)
        return 0

    if args.command == "eval-worker":
        if (
            args.depart_scale is None
            or args.worker_id is None
            or args.start_episode is None
            or args.end_episode is None
            or args.model_path is None
            or args.output_dir is None
        ):
            parser.error(
                "eval-worker requires --depart-scale --worker-id --start-episode "
                "--end-episode --model-path --output-dir"
            )
        _apply_patch(args.depart_scale)
        args.method = METHOD
        yv2.run_eval_worker(args)
        return 0

    run_full(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
