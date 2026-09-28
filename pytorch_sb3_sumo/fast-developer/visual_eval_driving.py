"""fast-developer 下所有已保存最终模型的 SUMO-GUI 实时驾驶回放脚本。

扫描 ``fast-developer`` 里每个方法的 run 目录，加载其已保存的最终模型
（优先 ``final_model.zip``，其次 ``best_training_success_model.zip``），用与训练/评估
**完全一致**的环境契约（env 类 + 改法 1/2/3 + depart×scale 低密度 traffic + reward
wrapper，见 ``train_intersection_yield_v2.py`` / ``train_intersection_hold35k_mst_fixed.py``）
驱动 ego 车辆在 ``intersection`` 场景上实时驾驶，打开可见的 SUMO-GUI 窗口供人工观察。

纯推理：不训练、不复制、不改任何 checkpoint。每个方法用与冻结评估一致的口径加载：
hold35k / hold35k_attn = ``StabilitySAC`` + ``use_actor_only``；hsac_mlp* = ``SceneRepresentationSAC``
+ ``use_actor_only``；sac_mlp* / mst_slt* = ``SceneRepresentationSAC``（连续头标准前向）；
D1-4（use_slots） = ``SceneRepresentationSACV2``。

用法::

    # 图形界面：勾选方法 → 开始回放（推荐）
    python fast-developer/visual_eval_driving.py --gui

    # 命令行单方法回放（N 集）
    python fast-developer/visual_eval_driving.py --method sac_mlp__intersection_yield_v2 --episodes 5
    python fast-developer/visual_eval_driving.py --method hold35k__intersection_yield_v2 --episodes 3 --speed 4
    python fast-developer/visual_eval_driving.py --list                 # 列出所有可用方法
    python fast-developer/visual_eval_driving.py --smoke                # 快速 smoke：单方法 1 集
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # pytorch_sb3_sumo
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FAST_DEV = Path(__file__).resolve().parent  # fast-developer
if str(FAST_DEV) not in sys.path:
    sys.path.insert(0, str(FAST_DEV))

import train_intersection_yield_v2 as yv2  # noqa: E402
import train_intersection_hold35k_mst_fixed as legacy  # noqa: E402

# 场景注册表：场景名 -> (原始 traffic 目录, yield_v2 家族默认 depart×scale)。
# intersection_sorted 是 depart 排序版 intersection 副本（三股车流都参与仿真，
# 修复原场景左右手车流被 SUMO 静默忽略的 bug），已在 paper_scenario_registry /
# scenario_registry 注册。legacy 家族走 base demand，不读 depart×scale。
SCENARIOS: dict[str, tuple[Path, float]] = {
    "intersection": (
        PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection" / "traffic",
        2.0,
    ),
    "intersection_sorted": (
        PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted" / "traffic",
        2.0,
    ),
}
EPISODES_DEFAULT = 5
DEFAULT_OUTPUT_ROOT = FAST_DEV / "_visual_eval_driving"

# --------------------------------------------------------------------------- #
# 方法注册表：run 目录名 -> (标签, 环境家族, env adapter, 加载口径, seed 块)
#
# 环境家族：
#   "yield_v2"  走 train_intersection_yield_v2.make_env_factory（改法 1+2/3 +
#               depart×scale 2.0 低密度 + GeneralizedRewardShapingWrapper）
#   "legacy"    走 train_intersection_hold35k_mst_fixed.make_env_factory（paper
#               标准环境，无改法，base demand）
# 加载口径：
#   stability_actor  StabilitySAC.load + use_actor_only（hold35k 混合头）
#   scene_rep_actor  SceneRepresentationSAC.load + use_actor_only（hsac 混合头）
#   scene_rep        SceneRepresentationSAC.load（连续头标准前向）
#   scene_rep_v2     SceneRepresentationSACV2.load（D1-4 三槽 + SLT/SBS）
# --------------------------------------------------------------------------- #
METHODS: list[tuple[str, str, str, str, str, int, str, float | None]] = [
    # --- yield_v2 家族（intersection_yield_v2，改法 1+2/3 + reward shaping v2） ---
    ("hold35k__intersection_yield_v2",        "hold35k (TASAC, v4_8 混合头)",     "yield_v2", "v4_8",    "stability_actor", 420_000, "intersection", None),
    ("hold35k_attn__intersection_yield_v2",   "hold35k_attn (GCN→attention)",     "yield_v2", "v4_8",    "stability_actor", 420_000, "intersection", None),
    ("hsac_mlp__intersection_yield_v2",       "HSAC-MLP (混合头+MLP)",            "yield_v2", "v4_8",    "scene_rep_actor", 420_000, "intersection", None),
    ("hsac_mlp_base__intersection_yield_v2",  "HSAC-MLP base (混合头, 改法1+2)",  "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection", None),
    ("sac_mlp__intersection_yield_v2",        "SAC-MLP (连续头+MLP)",             "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("sac_mlp_v48__intersection_yield_v2",    "SAC-MLP v4_8 (连续头, 改法1+2+3)", "yield_v2", "v4_8",    "scene_rep",       420_000, "intersection", None),
    ("mst_slt__intersection_yield_v2",        "MST+SLT (scene_rep, yield v2)",    "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("sac_mlp_hcc16__intersection_yield_v2",  "SAC-MLP HCC-16 (n=16 回报)",       "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("hsac_mlp_base_hcc16__intersection_yield_v2", "HSAC-MLP base HCC-16",        "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection", None),
    ("sac_mlp_d1_st__intersection_yield_v2",  "SAC D1-1 时空交互",                "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("sac_mlp_d1_st_rt__intersection_yield_v2", "SAC D1-2 +路由意图",             "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("sac_mlp_d1_st_rt_topo__intersection_yield_v2", "SAC D1-3 +拓扑先验",        "yield_v2", "base",    "scene_rep",       10_000, "intersection", None),
    ("sac_mlp_d1_full__intersection_yield_v2", "SAC D1-4 三槽+SLT/SBS",           "yield_v2", "base",    "scene_rep_v2",    10_000, "intersection", None),
    ("hsac_mlp_base_d1_st__intersection_yield_v2", "HSAC base D1-1 时空交互",     "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection", None),
    ("hsac_mlp_base_d1_st_rt__intersection_yield_v2", "HSAC base D1-2 +路由意图", "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection", None),
    ("hsac_mlp_base_d1_st_rt_topo__intersection_yield_v2", "HSAC base D1-3 +拓扑", "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection", None),
    # --- legacy mst_slt 家族（intersection 基础场景，paper 标准环境） ---
    ("mst_slt__intersection",                 "MST+SLT 加速基线 (batch256)",      "legacy",   "base",    "scene_rep",       10_000, "intersection", None),
    ("mst_slt__intersection_fixed",           "MST+SLT 修复基线 (lr1e-4)",        "legacy",   "base",    "scene_rep",       10_000, "intersection", None),
    ("mst_slt__intersection_rs",              "MST+SLT +reward shaping",          "legacy",   "base",    "scene_rep",       10_000, "intersection", None),
    # --- intersection_sorted 副本（depart 排序，三股车流都参与仿真） ---
    ("hold35k__intersection_sorted",          "hold35k (sorted 三车流)",          "yield_v2", "v4_8",    "stability_actor", 420_000, "intersection_sorted", None),
    ("mst_slt__intersection_sorted",          "MST+SLT (sorted 三车流)",          "yield_v2", "base",    "scene_rep",       10_000, "intersection_sorted", None),
    ("hsac_mlp__intersection_sorted",         "HSAC-MLP (sorted 三车流)",         "yield_v2", "v4_8",    "scene_rep_actor", 420_000, "intersection_sorted", None),
    ("sac_mlp__intersection_sorted",          "SAC-MLP (sorted 三车流)",          "yield_v2", "base",    "scene_rep",       10_000, "intersection_sorted", None),
    ("sac_mlp_v48__intersection_sorted",      "SAC-MLP v4_8 (sorted 三车流)",     "yield_v2", "v4_8",    "scene_rep",       420_000, "intersection_sorted", None),
    ("hsac_mlp_base__intersection_sorted",    "HSAC-MLP base (sorted 三车流)",    "yield_v2", "v4_base", "scene_rep_actor", 10_000, "intersection_sorted", None),
    ("hold35k_legacy__intersection_sorted",   "hold35k legacy (sorted 三车流)",   "legacy",   "v4_8",    "stability_actor", 420_000, "intersection_sorted", None),
    ("mst_slt_legacy__intersection_sorted",   "MST+SLT legacy (sorted 三车流)",   "legacy",   "base",    "scene_rep",       10_000, "intersection_sorted", None),
    # --- intersection_sorted 发车间隔难度验证（sac_mlp，两组 depart×scale） ---
    ("sac_mlp_depart1p0__intersection_sorted", "SAC-MLP depart×1.0 (sorted 密度不变)", "yield_v2", "base", "scene_rep", 10_000, "intersection_sorted", 1.0),
    ("sac_mlp_depart0p5__intersection_sorted", "SAC-MLP depart×0.5 (sorted 密度×2)",    "yield_v2", "base", "scene_rep", 10_000, "intersection_sorted", 0.5),
]

METHOD_BY_DIR = {entry[0]: entry for entry in METHODS}
_MODEL_FILES = ("final_model.zip", "best_training_success_model.zip")


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model(run_dir_name: str) -> Path | None:
    """返回方法的模型文件：优先 ``final_model.zip``，其次 ``best_training_success_model.zip``。"""
    run_dir = FAST_DEV / run_dir_name
    for name in _MODEL_FILES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def discover_available() -> list[tuple[str, str, str, str, str, int, str, float | None, Path]]:
    """只返回确有已保存模型的 (method_entry, model_path)。"""
    available = []
    for entry in METHODS:
        model = resolve_model(entry[0])
        if model is not None:
            available.append((*entry, model))
    return available


def _install_visible_sumo_spawn() -> None:
    """显式 SHOWNORMAL 启动 SUMO-GUI 子进程（pythonw 启动时窗口不隐藏）。"""
    if os.name != "nt":
        return
    import subprocess

    original_popen = subprocess.Popen

    def visible_popen(*popen_args: Any, **popen_kwargs: Any) -> Any:
        command = popen_kwargs.get("args")
        if command is None and popen_args:
            command = popen_args[0]
        executable = ""
        if isinstance(command, (list, tuple)) and command:
            executable = Path(str(command[0])).name.lower()
        if executable in {"sumo-gui", "sumo-gui.exe"}:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 1
            popen_kwargs["startupinfo"] = startupinfo
            popen_kwargs["creationflags"] = int(
                popen_kwargs.get("creationflags", 0)
            ) | int(subprocess.CREATE_NEW_PROCESS_GROUP)
        return original_popen(*popen_args, **popen_kwargs)

    subprocess.Popen = visible_popen


def _apply_scenario_patch(scenario: str, depart_scale: float | None) -> None:
    """把 yv2 / legacy 的场景常量指向目标场景（当前进程内生效）。

    legacy 家族不读 depart×scale（base demand），只需 ``legacy.SCENARIO``；
    yield_v2 家族还需 ``yv2.SOURCE_TRAFFIC``（depart×scale 直接从它 glob traffic）
    与 ``yv2.DEPART_SCALE``。``depart_scale=None`` 表示用该场景默认值。
    """
    if scenario not in SCENARIOS:
        raise KeyError(f"未知场景 {scenario!r}（可用：{sorted(SCENARIOS)}）")
    source_traffic, default_scale = SCENARIOS[scenario]
    for mod in (yv2, legacy):
        mod.SCENARIO = scenario
    yv2.SOURCE_TRAFFIC = source_traffic
    yv2.DEPART_SCALE = default_scale if depart_scale is None else depart_scale


def make_live_env(run_dir_name: str, adapter: str, family: str,
                  scenario: str, depart_scale: float | None) -> Any:
    """构造带 render_mode='human' 的实时环境（与各方法训练环境一致）。"""
    run_dir = FAST_DEV / run_dir_name
    overlay_root = run_dir / "overlays" / "viz"
    _apply_scenario_patch(scenario, depart_scale)
    if family == "legacy":
        ns = legacy._environment_namespace()
        ns.gui = True
        return legacy.make_env_factory(adapter, overlay_root)(ns, evaluation=False)
    ns = yv2._environment_namespace()
    ns.gui = True
    return yv2.make_env_factory(adapter, overlay_root)(ns, evaluation=False)


# --- 渲染美化 ------------------------------------------------------------------ #
# SUMO-GUI 默认把车辆画成方块（vType 无 guiShape）。在 env.reset() 后通过 traci 给
# 车辆设 passenger 形状、行人放大，纯 type 层、不改物理与源资产。
PEDESTRIAN_VISUAL_LENGTH = 1.2
PEDESTRIAN_VISUAL_WIDTH = 0.6


def _beautify_rendering(env: Any) -> None:
    connection = getattr(getattr(env, "unwrapped", env), "_connection", None)
    if connection is None:
        return
    try:
        type_ids = list(connection.vehicletype.getIDList())
    except BaseException:
        return
    for type_id in type_ids:
        if str(type_id).startswith("DEFAULT_"):
            continue
        try:
            vehicle_class = str(connection.vehicletype.getVehicleClass(type_id))
        except BaseException:
            continue
        try:
            if vehicle_class == "pedestrian":
                connection.vehicletype.setShapeClass(type_id, "pedestrian")
                connection.vehicletype.setLength(type_id, PEDESTRIAN_VISUAL_LENGTH)
                connection.vehicletype.setWidth(type_id, PEDESTRIAN_VISUAL_WIDTH)
            else:
                connection.vehicletype.setShapeClass(type_id, "passenger")
        except BaseException:
            continue


def load_model(load_spec: str, checkpoint: Path, env: Any, device: str) -> Any:
    if load_spec == "stability_actor":
        from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

        return use_actor_only(
            StabilitySAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
        )
    if load_spec == "scene_rep_actor":
        from algos.sb3_torch.sac import SceneRepresentationSAC
        from tools.v48_stability_v4.model import use_actor_only

        return use_actor_only(
            SceneRepresentationSAC.load(
                str(checkpoint), env=env, device=device, buffer_size=32
            )
        )
    if load_spec == "scene_rep_v2":
        from algos.sb3_torch.sac_v2 import SceneRepresentationSACV2

        return SceneRepresentationSACV2.load(
            str(checkpoint), env=env, device=device, buffer_size=32
        )
    from algos.sb3_torch.sac import SceneRepresentationSAC

    return SceneRepresentationSAC.load(
        str(checkpoint), env=env, device=device, buffer_size=32
    )


def _record(episode: int, seed: int, episode_return: float, decision_steps: int,
            info: dict[str, Any]) -> dict[str, Any]:
    success = bool(info.get("is_success", False))
    raw_steps = int(info.get("raw_simulation_steps", decision_steps * 3))
    return {
        "episode": episode,
        "seed": seed,
        "episode_return": episode_return,
        "decision_steps": decision_steps,
        "raw_steps": raw_steps,
        "completion_time_seconds": raw_steps * 0.1 if success else None,
        "success": success,
        "collision": bool(info.get("collision", False)),
        "off_route": bool(info.get("off_route", False)),
        "timeout": bool(info.get("max_time", False)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    n = len(records)
    returns = [float(r["episode_return"]) for r in records]
    completions = [
        float(r["completion_time_seconds"])
        for r in records
        if r["completion_time_seconds"] is not None
    ]
    return {
        "episodes": n,
        "success_rate": sum(int(r["success"]) for r in records) / n,
        "collision_rate": sum(int(r["collision"]) for r in records) / n,
        "off_route_rate": sum(int(r["off_route"]) for r in records) / n,
        "timeout_rate": sum(int(r["timeout"]) for r in records) / n,
        "mean_return": float(np.mean(returns)),
        "mean_raw_steps": float(np.mean([int(r["raw_steps"]) for r in records])),
        "mean_success_completion_time_seconds": (
            float(np.mean(completions)) if completions else None
        ),
    }


def live_episode(model: Any, env: Any, *, episode: int, seed: int, speed: float,
                 beautify: bool) -> dict[str, Any]:
    if speed <= 0:
        raise ValueError("--speed must be positive")
    action_repeat = int(getattr(getattr(env, "unwrapped", env), "action_repeat", 3))
    hold = 0.1 * action_repeat / speed
    observation, _ = env.reset(seed=seed)
    if beautify:
        _beautify_rendering(env)
    episode_return = 0.0
    decision_steps = 0
    while True:
        action, _ = model.predict(observation, deterministic=True)
        decision_steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(info.get("undiscounted_reward", reward))
        time.sleep(hold)
        if terminated or truncated:
            return _record(episode, seed, episode_return, decision_steps, info)


def run_cell(run_dir_name: str, adapter: str, family: str, load_spec: str,
             seed_start: int, scenario: str, depart_scale: float | None,
             checkpoint: Path, *, episodes: int, speed: float,
             device: str, beautify: bool, stop_event: Any = None) -> dict[str, Any]:
    import numpy as np
    import torch

    from algos.sb3_torch.evaluation import source_evaluation_augmentation

    torch.set_num_threads(1)
    env = make_live_env(run_dir_name, adapter, family, scenario, depart_scale)
    try:
        model = load_model(load_spec, checkpoint, env, device)
        records: list[dict[str, Any]] = []
        with source_evaluation_augmentation(model):
            for index in range(episodes):
                if stop_event is not None and stop_event.is_set():
                    break
                episode_seed = seed_start + index
                np.random.seed(episode_seed + 600_000)
                torch.manual_seed(episode_seed + 600_000)
                # 写到底层 env（wrapper 无 __getattr__，写到 wrapper 会吞掉）。
                env.unwrapped._traffic_episode_index = index
                env.unwrapped._traffic_roll = None
                records.append(
                    live_episode(
                        model, env, episode=index, seed=episode_seed, speed=speed,
                        beautify=beautify,
                    )
                )
        if not records:
            raise RuntimeError("no episodes recorded (stopped before the first episode)")
        return {
            "run_dir": run_dir_name,
            "label": METHOD_BY_DIR[run_dir_name][1],
            "checkpoint": str(checkpoint),
            "family": family,
            "adapter": adapter,
            "scenario": scenario,
            "depart_scale": depart_scale,
            "deployment": (
                "actor_deterministic"
                if load_spec in ("stability_actor", "scene_rep_actor")
                else "symmetric_sac"
            ),
            "seed_start": seed_start,
            "summary": _summarize(records),
            "episode_records": records,
        }
    finally:
        env.close()


def execute_run(
    methods: list[tuple[str, str, str, str, str, int, str, float | None, Path]],
    *,
    episodes: int,
    speed: float,
    device: str,
    output: Path,
    emit: Callable[[str], None] | None = None,
    stop_event: Any = None,
    beautify: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    def log(text: str) -> None:
        if emit is not None:
            emit(text)
        print(text, flush=True)

    _install_visible_sumo_spawn()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_dir_name, label, family, adapter, load_spec, seed_start, scenario, depart_scale, checkpoint in methods:
        if stop_event is not None and stop_event.is_set():
            break
        log(f"[start] {label}  ({run_dir_name}) — SUMO-GUI 应已弹出")
        try:
            row = run_cell(
                run_dir_name, adapter, family, load_spec, seed_start, scenario,
                depart_scale, checkpoint,
                episodes=episodes, speed=speed, device=device, beautify=beautify,
                stop_event=stop_event,
            )
        except BaseException as exc:
            failures.append({"method": run_dir_name, "error": repr(exc)})
            log(f"[error] {label}: {exc!r}")
            continue
        rows.append(row)
        summary = row["summary"]
        log(
            f"[done] {label}: success={summary['success_rate']:.3f}, "
            f"collision={summary['collision_rate']:.3f}, timeout={summary['timeout_rate']:.3f}"
        )
    manifest = {
        "schema_version": "fast-developer-visual-driving/v1",
        "status": "stopped" if (stop_event and stop_event.is_set()) else "completed",
        "training_performed": False,
        "live_sumogui": True,
        "deterministic": True,
        "episodes": episodes,
        "speed": speed,
        "device": device,
        "beautify_rendering": beautify,
        "methods": [m[0] for m in methods],
        "completed": [r["run_dir"] for r in rows],
        "failed": failures,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[completed] 结果目录: {output}")
    return rows, failures, output


def _new_output_dir(tag: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = DEFAULT_OUTPUT_ROOT / tag / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = DEFAULT_OUTPUT_ROOT / tag / f"{stamp}_{suffix:02d}"
    return candidate


# --------------------------------------------------------------------------- #
# 图形界面
# --------------------------------------------------------------------------- #
def run_gui(args: argparse.Namespace) -> None:
    import queue
    import threading
    import tkinter as tk
    from tkinter import messagebox, ttk

    available = discover_available()
    if not available:
        raise SystemExit("fast-developer 下没有找到任何已保存模型。")

    initial_device = resolve_device(args.device)
    root = tk.Tk()
    root.title("fast-developer 最终模型实时驾驶回放 — SUMO-GUI")
    root.geometry("920x680")

    method_vars = {m[0]: tk.BooleanVar(value=False) for m in available}
    episodes_var = tk.IntVar(value=EPISODES_DEFAULT)
    speed_var = tk.DoubleVar(value=4.0)
    device_var = tk.StringVar(value=initial_device)
    beautify_var = tk.BooleanVar(value=True)

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    methods_frame = ttk.LabelFrame(main, text="选择方法（勾选一个或多个）", padding=8)
    methods_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(methods_frame, borderwidth=0)
    scrollbar = ttk.Scrollbar(methods_frame, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for idx, (name, label, family, adapter, load_spec, seed, scenario, depart_scale, _) in enumerate(available):
        model_name = resolve_model(name).name
        text = f"{label}  —  {name}  [{family}/{adapter}, {model_name}]"
        ttk.Checkbutton(inner, text=text, variable=method_vars[name]).grid(
            row=idx, column=0, sticky="w", padx=4, pady=1
        )

    def _set_methods(value: bool) -> None:
        for var in method_vars.values():
            var.set(value)

    sel_btns = ttk.Frame(methods_frame)
    sel_btns.pack(side="bottom", fill="x", pady=(6, 0))
    ttk.Button(sel_btns, text="全选", width=10, command=lambda: _set_methods(True)).pack(
        side="left", padx=(0, 4)
    )
    ttk.Button(sel_btns, text="清空", width=10, command=lambda: _set_methods(False)).pack(
        side="left"
    )

    controls = ttk.Frame(main)
    controls.pack(fill="x", pady=(10, 6))
    ttk.Label(controls, text="每方法 episode 数:").pack(side="left")
    ttk.Spinbox(controls, from_=1, to=100, textvariable=episodes_var, width=5).pack(
        side="left", padx=(4, 16)
    )
    ttk.Label(controls, text="回放速度:").pack(side="left")
    ttk.Spinbox(
        controls, from_=0.5, to=32, increment=0.5, textvariable=speed_var, width=5
    ).pack(side="left", padx=(4, 16))
    ttk.Label(controls, text="设备:").pack(side="left")
    ttk.Combobox(
        controls, textvariable=device_var, values=("cuda", "cpu", "auto"),
        state="readonly", width=8,
    ).pack(side="left", padx=(4, 16))
    ttk.Checkbutton(controls, text="渲染美化", variable=beautify_var).pack(side="left")

    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(0, 6))
    start_btn = ttk.Button(buttons, text="开始回放")
    start_btn.pack(side="left")
    stop_btn = ttk.Button(buttons, text="停止", state="disabled")
    stop_btn.pack(side="left", padx=(8, 0))
    output_label = ttk.Label(buttons, text="", foreground="#666666")
    output_label.pack(side="right")

    status = tk.Text(main, height=12, wrap="none", state="disabled")
    status.pack(fill="both", expand=True)
    status_scroll = ttk.Scrollbar(main, orient="horizontal", command=status.xview)
    status_scroll.pack(fill="x")
    status.configure(xscrollcommand=status_scroll.set)

    q: "queue.Queue[tuple]" = queue.Queue()
    stop_event = threading.Event()

    def append_status(text: str) -> None:
        status.configure(state="normal")
        status.insert("end", text + "\n")
        status.see("end")
        status.configure(state="disabled")

    def poll_queue() -> None:
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    append_status(payload)
                elif kind == "finish":
                    rows, failures, out = payload
                    stop_event.clear()
                    start_btn.configure(state="normal")
                    stop_btn.configure(state="disabled")
                    append_status("—" * 60)
                    if failures:
                        append_status(f"完成：{len(rows)} 个方法成功，{len(failures)} 个失败。")
                        for f in failures:
                            append_status(f"  失败 {f['method']}: {f['error']}")
                    else:
                        append_status(f"全部 {len(rows)} 个方法完成。")
                    append_status(f"结果目录: {out}")
                    messagebox.showinfo("完成", f"{len(rows)} 个方法完成\n{out}")
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    def worker(selected, episodes, speed, device, output, beautify):
        try:
            rows, failures, out = execute_run(
                selected, episodes=episodes, speed=speed, device=device, output=output,
                emit=lambda text: q.put(("log", text)), stop_event=stop_event,
                beautify=beautify,
            )
        except BaseException as exc:  # pragma: no cover
            q.put(("log", f"[fatal] {exc!r}"))
            rows, failures, out = [], [], output
        q.put(("finish", (rows, failures, out)))

    def on_start() -> None:
        selected = [m for m in available if method_vars[m[0]].get()]
        if not selected:
            messagebox.showwarning("未选择", "请至少勾选一个方法。")
            return
        episodes = int(episodes_var.get())
        speed = float(speed_var.get())
        device = resolve_device(device_var.get())
        if episodes <= 0 or speed <= 0:
            messagebox.showwarning("参数错误", "episode 数和速度必须为正。")
            return
        output = _new_output_dir("live")
        output_label.configure(text=str(output))
        append_status(f"输出目录: {output}")
        append_status(f"已选 {len(selected)} 个方法，每个 {episodes} 集，速度 {speed}，设备 {device}")
        start_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        stop_event.clear()
        threading.Thread(
            target=worker,
            args=(selected, episodes, speed, device, output, bool(beautify_var.get())),
            daemon=True,
        ).start()

    def on_stop() -> None:
        stop_event.set()
        append_status("[stop] 已请求停止，将在当前 episode 结束后停下…")
        stop_btn.configure(state="disabled")

    def on_close() -> None:
        stop_event.set()
        root.destroy()

    start_btn.configure(command=on_start)
    stop_btn.configure(command=on_stop)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="图形界面选择方法（推荐）")
    parser.add_argument("--list", action="store_true", help="列出所有可用方法并退出")
    parser.add_argument("--method", type=str, default=None, help="单个方法的 run 目录名")
    parser.add_argument("--device", default="cuda", help="cuda | cpu | auto")
    parser.add_argument("--episodes", type=int, default=EPISODES_DEFAULT)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--no-beautify", action="store_true", help="关闭渲染美化")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="单方法 1 集快速验证")
    args = parser.parse_args(argv)

    if args.list:
        for name, label, family, adapter, load_spec, seed, scenario, depart_scale, model in discover_available():
            print(f"{name}\t{label}\t[{family}/{adapter}] scenario={scenario} model={model.name}")
        return 0

    if args.gui:
        run_gui(args)
        return 0

    available = discover_available()
    by_dir = {m[0]: m for m in available}
    if args.smoke:
        args.method = args.method or available[0][0]
        args.episodes = 1

    if args.method is None:
        parser.error("CLI 模式需要 --method（或 --gui / --list）")
    if args.method not in by_dir:
        parser.error(
            f"方法 {args.method!r} 不存在或无已保存模型；用 --list 查看可用方法"
        )

    selected = [by_dir[args.method]]
    output = args.output_dir or _new_output_dir("live")
    device = resolve_device(args.device)
    rows, failures, output = execute_run(
        selected, episodes=args.episodes, speed=args.speed, device=device, output=output,
        beautify=not args.no_beautify,
    )
    if failures:
        print(f"[warn] {len(failures)} 个方法失败：{failures}", file=sys.stderr)
        return 1
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - surface any startup error under pythonw
        import traceback

        log = Path(__file__).with_name("_visual_eval_error.log")
        try:
            log.write_text(traceback.format_exc(), encoding="utf-8")
        except BaseException:
            pass
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "fast-developer 驾驶回放启动失败",
                f"{exc!r}\n\n完整 traceback 已写入：\n{log}",
            )
            root.destroy()
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
