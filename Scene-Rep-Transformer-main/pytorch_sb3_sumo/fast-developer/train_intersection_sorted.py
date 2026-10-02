"""在「depart 排序版」intersection 副本场景上训练两个脚本里的所有方法。

背景
----
原 ``intersection`` 场景的 traffic 文件（``traffic_00..29.rou.xml``）按**流向**
分组排列，组与组切换时 depart 时间大幅回退（~597s -> 0s）。SUMO 要求 route
文件按 depart 时间排序，于是报

    Warning: Route file should be sorted by departure time, ignoring ...

并把第二、三组（东→西 ``-E3 -E0``、西→东 ``E0 E3``）车流**静默忽略**，实际
仿真里只有北→南（``E2 E1``）一股车流在跑 —— 左右手方向的冲突其实一直不存在。

本脚本先（幂等地）生成 depart 排序版副本场景
``envs/sumo/original_scenarios_v1/intersection_sorted/``：把每个 traffic 文件的
``<vehicle>`` 按 ``(depart, departLane)`` 升序重排（``map.net.xml`` / ``ego.rou.xml``
原样复制），让三股车流都真正参与仿真；然后复用两个既有脚本的**训练 / 评估 /
绘图**底层函数（只把它们的 ``SCENARIO`` 常量指向新场景），在新场景上训练。

方法清单（8 个）
----------------
  yield_v2 家族（``train_intersection_yield_v2``：DEPART_SCALE=2.0 降密度 +
  reward shaping v2 + 改法 1+2/3）:
      hold35k / mst_slt / hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base
  legacy 家族（``train_intersection_hold35k_mst_fixed``：原始密度 + 纸面 reward +
  改法 1+2）:
      hold35k_legacy / mst_slt_legacy

结果目录 ``fast-developer/{方法名}__intersection_sorted``。

用法::

    python fast-developer/train_intersection_sorted.py --make-scenario-only
    python fast-developer/train_intersection_sorted.py --smoke            # 8 方法 smoke
    python fast-developer/train_intersection_sorted.py                    # 8 方法正式训练
    python fast-developer/train_intersection_sorted.py --method hold35k --smoke
    python fast-developer/train_intersection_sorted.py --method sac_mlp
    python fast-developer/train_intersection_sorted.py --eval-only \
        --method sac_mlp --model-path <run_dir>/final_model.zip --output-dir <run_dir>

注意：``intersection_sorted`` 已在 ``paper_scenario_registry.py`` 与
``scenario_registry.py`` 注册；若这两处缺失，本脚本的 ``ensure_sorted_scenario``
生成资产后仍会因 ``get_paper_scenario_spec`` 报 KeyError。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2 as yv2
import train_intersection_hold35k_mst_fixed as legacy

NEW_SCENARIO = "intersection_sorted"
NEW_SOURCE_TRAFFIC = (
    PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
    / "intersection_sorted" / "traffic"
)

# 全局唯一方法名 -> (脚本模块, 模块内部方法名)。legacy 与 yield_v2 的
# hold35k / mst_slt 重名，故 legacy 用 ``*_legacy`` 后缀区分。
DISPATCH: dict[str, tuple] = {
    "hold35k": (yv2, "hold35k"),
    "mst_slt": (yv2, "mst_slt"),
    "hsac_mlp": (yv2, "hsac_mlp"),
    "sac_mlp": (yv2, "sac_mlp"),
    "sac_mlp_v48": (yv2, "sac_mlp_v48"),
    "hsac_mlp_base": (yv2, "hsac_mlp_base"),
    "hold35k_legacy": (legacy, "hold35k"),
    "mst_slt_legacy": (legacy, "mst_slt"),
}
ALL_METHODS = tuple(DISPATCH)
TRAIN_WORKERS = 2  # run_full 并行启动的训练 worker 数（对齐原脚本）


def _apply_patch() -> None:
    """把两个脚本的 SCENARIO（及 yield_v2 的 traffic 源）指向新副本场景。

    只在当前进程内覆盖模块常量：所有复用函数运行时经模块全局查找 SCENARIO /
    SOURCE_TRAFFIC，因此覆盖即刻生效。不覆盖 ``__file__`` —— 本脚本用自己的
    subprocess 编排（train-method 子进程 + 进程内评估），不依赖原脚本的
    eval-worker 子进程，从而绕开两个脚本方法名重名的歧义。
    """
    for mod in (yv2, legacy):
        mod.SCENARIO = NEW_SCENARIO
    yv2.SOURCE_TRAFFIC = NEW_SOURCE_TRAFFIC


# --------------------------------------------------------------------------- #
# depart 排序版副本场景生成（幂等）
# --------------------------------------------------------------------------- #
def _sort_traffic_file(src: Path, dst: Path) -> None:
    """按 (depart, departLane) 升序重排一个 traffic 文件的 <vehicle>。

    vType 保持在前（SUMO 要求类型先于车辆定义），vehicle 顺序调整后写入 dst。
    """
    tree = ET.parse(src)
    root = tree.getroot()
    vtypes = [child for child in list(root) if child.tag == "vType"]
    vehicles = [child for child in list(root) if child.tag == "vehicle"]
    vehicles.sort(
        key=lambda v: (
            float(v.attrib.get("depart", "0")),
            int(v.attrib.get("departLane", "0")),
        )
    )
    new_root = ET.Element("routes")
    for vt in vtypes:
        new_root.append(vt)
    for v in vehicles:
        new_root.append(v)
    ET.ElementTree(new_root).write(dst, encoding="UTF-8", xml_declaration=True)


def ensure_sorted_scenario() -> Path:
    """生成（或复用已生成的）``intersection_sorted`` 副本场景，返回其目录。"""
    src = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection"
    dst = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted"
    marker = dst / ".depart_sorted"
    if marker.is_file():
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("map.net.xml", "ego.rou.xml"):
        shutil.copy2(src / name, dst / name)

    traffic_src = src / "traffic"
    traffic_dst = dst / "traffic"
    traffic_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(traffic_src.glob("traffic_*.rou.xml")):
        _sort_traffic_file(src_file, traffic_dst / src_file.name)

    # base SumoSceneEnv 构造时 get_scenario_spec 会检查 scenarios/<name>/
    # scenario.sumocfg 存在性；复制 intersection 的占位即可（net-file 相对路径
    # 仍指向 networks/intersection，与新增 registry 的默认 network_name 一致）。
    base_src = PROJECT_ROOT / "envs" / "sumo" / "scenarios" / "intersection"
    base_dst = PROJECT_ROOT / "envs" / "sumo" / "scenarios" / "intersection_sorted"
    base_dst.mkdir(parents=True, exist_ok=True)
    for name in ("scenario.sumocfg", "routes.rou.xml"):
        candidate = base_src / name
        if candidate.is_file():
            shutil.copy2(candidate, base_dst / name)

    marker.write_text(
        "depart-sorted traffic (three flows all active)\n", encoding="utf-8"
    )
    return dst


# --------------------------------------------------------------------------- #
# 训练 + 评估编排（进程内评估，无 eval-worker 子进程）
# --------------------------------------------------------------------------- #
def _run_dir(global_name: str) -> Path:
    mod, _ = DISPATCH[global_name]
    return mod.RESULT_ROOT / f"{global_name}__{NEW_SCENARIO}"


def _train(global_name: str, run_dir: Path, smoke: bool) -> Path:
    """复用底层训练函数（当前进程内训练，返回 final_model 路径）。"""
    mod, method = DISPATCH[global_name]
    if method == "hold35k":
        return mod.run_training_hold35k(run_dir, smoke=smoke)
    if method == "mst_slt":
        return mod.run_training_mst_slt(run_dir, smoke=smoke)
    # 其余为 yield_v2 的 MLP 方法（hsac_mlp / sac_mlp / sac_mlp_v48 / hsac_mlp_base）
    return mod.run_training_mlp(method, run_dir, smoke=smoke)


def evaluate(
    global_name: str, run_dir: Path, final_model: Path, smoke: bool
) -> dict:
    """subprocess 并行评估：每个 worker 独立进程调用底层 ``run_eval_worker``。

    不通过原脚本的 ``run_evaluation``（其用内部方法名，会与方法重名冲突）；这里
    用全局唯一名 global_name 传给 eval-worker 子进程，新脚本 main 的 eval-worker
    分支再映射回模块内部方法名。subprocess 隔离让每个 worker 退出时自动回收
    SUMO 子进程，避免进程内线程并行时 SUMO 进程残留锁住 traffic 文件。
    """
    mod, method = DISPATCH[global_name]
    workers = 2 if smoke else mod.EVAL_WORKERS
    episodes = 8 if smoke else mod.EVAL_EPISODES_TOTAL

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
            "--method", global_name,
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
    summary = mod.summarize_records(records)

    result = dict(
        identity=dict(
            method=global_name,
            source_script=mod.__name__,
            source_method=method,
            scenario=NEW_SCENARIO,
            checkpoint=str(final_model),
            checkpoint_sha256=mod._sha256(final_model),
            episodes=episodes,
            workers=workers,
            episodes_per_worker=[end - start for start, end in ranges],
            smoke=smoke,
        ),
        summary=summary,
        episode_records=records,
    )
    mod._write_json_atomic(run_dir / "evaluation_results.json", result)
    return result


def train_and_eval(global_name: str, smoke: bool) -> Path:
    mod, method = DISPATCH[global_name]
    run_dir = _run_dir(global_name)
    # 不预先 mkdir：hold35k / mlp 各自 mkdir(exist_ok=True)，而 mst_slt 经
    # train_sb3.main 要求 run_dir 尚不存在（exist_ok=False）。
    final = _train(global_name, run_dir, smoke=smoke)
    result = evaluate(global_name, run_dir, final, smoke=smoke)
    curve = mod.plot_training_curves(method, run_dir)

    manifest = dict(
        method=global_name,
        source_script=mod.__name__,
        source_method=method,
        scenario=NEW_SCENARIO,
        run_dir=str(run_dir),
        final_model=str(final),
        evaluation=result["summary"],
        training_curve=str(curve) if curve else None,
        smoke=smoke,
        note=(
            "depart 排序版 intersection 副本：三股车流（北→南 / 东→西 / 西→东）"
            "都参与仿真，修复原场景左右手车流被 SUMO 忽略的 bug。"
        ),
    )
    mod._write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            dict(method=global_name, summary=result["summary"]),
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


# --------------------------------------------------------------------------- #
# launcher / main
# --------------------------------------------------------------------------- #
def run_full(smoke: bool) -> Path:
    if not smoke and not yv2._cuda_available():
        raise RuntimeError("CUDA unavailable; training has no CPU fallback")

    root = yv2.RESULT_ROOT  # fast-developer
    root.mkdir(parents=True, exist_ok=True)
    if not smoke:
        existing = [n for n in ALL_METHODS if _run_dir(n).exists()]
        if existing:
            raise RuntimeError(f"output dirs already exist (remove first): {existing}")

    script = str(Path(__file__).resolve())
    codes: dict[str, int] = {}

    def run_one(name: str) -> tuple[str, int]:
        cmd = [sys.executable, script, "train-method", "--method", name]
        if smoke:
            cmd.append("--smoke")
        log = root / f"{name}__sorted_train.log"
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).returncode
        return name, returncode

    with ThreadPoolExecutor(max_workers=TRAIN_WORKERS) as pool:
        futures = {pool.submit(run_one, name): name for name in ALL_METHODS}
        for future in as_completed(futures):
            name, code = future.result()
            codes[name] = code
            print(f"[launcher] {name} exit {code}", flush=True)

    failed = {n: c for n, c in codes.items() if c != 0}
    if failed:
        raise RuntimeError(
            f"training worker(s) failed: {failed}; inspect "
            f"{root}/*__sorted_train.log"
        )
    return root


def main(argv: list[str] | None = None) -> int:
    _apply_patch()
    ensure_sorted_scenario()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train-method", "eval-worker"),
        default=None,
        help="留空表示 launcher（并行启动全部方法）",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小规模快速验证全链路")
    parser.add_argument("--method", choices=ALL_METHODS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--start-episode", type=int)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--make-scenario-only",
        action="store_true",
        help="只生成 depart 排序版副本场景后退出",
    )
    args = parser.parse_args(argv)

    if args.make_scenario_only:
        print(f"scenario ready: {NEW_SOURCE_TRAFFIC.parent}")
        return 0

    if args.command == "train-method":
        if args.method is None:
            parser.error("train-method requires --method")
        train_and_eval(args.method, smoke=args.smoke)
        return 0

    if args.command == "eval-worker":
        if (
            args.method is None
            or args.worker_id is None
            or args.start_episode is None
            or args.end_episode is None
            or args.model_path is None
            or args.output_dir is None
        ):
            parser.error(
                "eval-worker requires --method --worker-id --start-episode "
                "--end-episode --model-path --output-dir"
            )
        mod, m = DISPATCH[args.method]
        args.method = m  # 映射回模块内部方法名
        mod.run_eval_worker(args)
        return 0

    if args.eval_only:
        if args.method is None or args.model_path is None or args.output_dir is None:
            parser.error("--eval-only requires --method, --model-path, --output-dir")
        mod, method = DISPATCH[args.method]
        run_dir = Path(args.output_dir)
        result = evaluate(args.method, run_dir, Path(args.model_path), smoke=args.smoke)
        mod.plot_training_curves(method, run_dir)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.method is not None:
        # 便捷入口：``--method X`` 直接训练单个方法（无需 train-method 子命令）。
        train_and_eval(args.method, smoke=args.smoke)
        return 0

    run_full(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
