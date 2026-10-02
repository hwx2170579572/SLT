"""Visual verification of the nine three-scene methods' final models.

Loads the final saved model of every method in ``results_three_scene_v1``
(``final_model.zip`` for SB3-native methods, ``final_model.pt`` for the three
sequence/latent baselines) and drives it through the three released scenarios
``cross``, ``carla`` and ``cross_left`` with a *visible* SUMO-GUI window, so the
ego vehicle's behaviour can be inspected directly.

Two entry points share the same engine:

* ``--gui``            interactive Tkinter window: tick the methods and scenes
                       to run, set episodes / speed / device, then press start.
* (no ``--gui``)       headless CLI loop over ``--methods`` x ``--scenes``.

Both are inference-only: nothing trains, changes, or copies a checkpoint.  Each
method is loaded with the exact deployment path used by
``tools/three_scene_eval_curve.py`` (actor-only deterministic for the hybrid-head
families, native SAC for the continuous-head families, return-to-go / latent
rollout for the sequence baselines) and evaluated on the same deterministic seed
block, so the playback matches the frozen-evaluation convention.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.evaluation import source_evaluation_augmentation  # noqa: E402
from tools.three_scene_hold35k_vs_mst_v1.common import (  # noqa: E402
    ALL_METHODS,
    ALL_SCENES,
    BASELINE_ENV_CONTRACT,
    OVERLAY_ROOT,
    RESULT_ROOT,
    SCENES,
    SEQUENCE_BASELINE_METHODS,
    TRAIN_ROOT,
    checkpoint_source_scene,
    density,
    sha256,
    write,
)

DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "visual_verification"
EPISODES_DEFAULT = 10
LIVE_SUMO_ARGS = ("--window-pos", "40,40", "--window-size", "1280,800")

METHOD_LABELS = {
    "hold35k": "TASAC (hold35k)",
    "mst_slt": "MST+SLT",
    "hsac_mlp": "HSAC-MLP",
    "hsac_lstm": "HSAC-LSTM",
    "gnn_sac": "GNN+SAC",
    "hybrid_dt": "Hybrid-DT",
    "decision_transformer": "Decision Transformer",
    "hyar": "HyAR",
    "mst": "MST (no SLT)",
}
METHOD_COLORS = {
    "hold35k": "#D62728",
    "mst_slt": "#1f77b4",
    "hsac_mlp": "#ff7f0e",
    "hsac_lstm": "#2ca02c",
    "gnn_sac": "#9467bd",
    "hybrid_dt": "#8c564b",
    "decision_transformer": "#e377c2",
    "hyar": "#7f7f7f",
    "mst": "#17becf",
}
SCENE_TITLES = {
    "cross": "Cross (double merge)",
    "carla": "CARLA Town-10",
    "cross_left": "Cross-left (unprotected left)",
    # Fourth trained scene: uncontrolled (unregulated) 4-way, base density.
    "cross_left_unreg": "Cross-left unreg (uncontrolled, 0.2 veh/s/arm)",
    # New SUMO scenarios (zero-shot: load the source scene's final model).
    "merge": "Merge (highway + on-ramp, zero-shot)",
    "intersection": "Intersection (unsignalised left, zero-shot)",
}
# Evaluation seed blocks; identical to three_scene_eval_curve.SEED_START so the
# live playback reuses the first N episodes of the frozen 50-episode evaluation.
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


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def method_contract(method: str) -> str:
    """Environment contract per method (``v4_8`` hybrid head or ``base`` continuous)."""
    if method == "hold35k":
        return "v4_8"
    if method == "mst_slt":
        return "base"
    return BASELINE_ENV_CONTRACT[method]


def final_model_path(method: str, scene: str) -> Path:
    # Zero-shot: ``merge``/``intersection`` reuse the source scene's final model.
    source_scene = checkpoint_source_scene(scene)
    if method in SEQUENCE_BASELINE_METHODS:
        return TRAIN_ROOT / f"{method}__{source_scene}" / "final_model.pt"
    return TRAIN_ROOT / f"{method}__{source_scene}" / "final_model.zip"


def _install_visible_sumo_spawn() -> None:
    """Override only SUMO-GUI child creation with an explicit show state.

    ``traci.start`` delegates to ``subprocess.Popen``; when this evaluator is
    launched from ``pythonw.exe`` (as the desktop ``.cmd`` launcher does), Windows
    can propagate a hidden startup state to that child even though
    ``render_mode='human'`` selected ``sumo-gui``.  Supplying ``SW_SHOWNORMAL``
    here keeps the live window on the interactive desktop without touching the
    repository's environment implementation.  This hook is process-local.
    """
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


def make_live_env(method: str, scene: str) -> Any:
    """Construct the same evaluation env as the curve study, but with SUMO-GUI.

    ``_make_environment_factory`` forces ``render_mode=None`` whenever
    ``evaluation=True``, so the live env is built directly against the same
    classes, densities, overlay roots and partitions the factory would use.
    """
    from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
        IndependentV2FiveBySixEnvV1,
        IndependentV2FiveBySixEnvV4V1,
    )

    contract = method_contract(method)
    environment_class = (
        IndependentV2FiveBySixEnvV1
        if contract == "base"
        else IndependentV2FiveBySixEnvV4V1
    )
    den = density(scene)
    # v4_8 families share the hold35k overlay so they observe TASAC's exact
    # physical traffic; base families share the MST+SLT overlay.
    overlay_root = (
        OVERLAY_ROOT / "hold35k" / "ns_viz_final"
        if contract == "v4_8"
        else OVERLAY_ROOT / "mst_slt" / "seed_0"
    )
    return environment_class(
        scenario=scene,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        reward_discount=0.99,
        ego_control_profile="direct",
        include_state_lstm=False,
        state_lstm_only=False,
        episode_limit_profile="source",
        render_mode="human",
        high_density_vehicle_scale=float(den["vehicle_scale"]),
        high_density_pedestrian_scale=float(den["pedestrian_scale"]),
        high_density_clone_jitter_seconds=tuple(
            float(value) for value in den["clone_depart_jitter_seconds"]
        ),
        high_density_overlay_root=overlay_root,
        high_density_partition="evaluation",
        high_density_contract_partition=(
            "evaluation" if contract == "base" else "validation"
        ),
        sumo_args=LIVE_SUMO_ARGS,
    )


# --- rendering beautification -------------------------------------------------- #
# SUMO-GUI draws vehicles as plain rectangles unless each vType carries an explicit
# shape class, and the released ego/traffic vTypes omit ``guiShape``.  The fixes
# below are applied at runtime through traci *after* the environment has reset, so
# no route/network source asset is modified.  Shape classes and the pedestrian
# body enlargement are type-level attributes that SUMO inherits into every already
# spawned actor (verified against 1.25.0); none of them alter the policy's action
# space, the reward, or any success/collision/off-route outcome recorded by the
# frozen ``eval_curve`` evaluation, which this playback never rewrites.
PEDESTRIAN_VISUAL_LENGTH = 1.2
PEDESTRIAN_VISUAL_WIDTH = 0.6


def _beautify_rendering(env: Any) -> None:
    """Give SUMO-GUI real silhouettes: passenger cars and visible pedestrians.

    Called after ``env.reset()`` so the traci connection is live.  Vehicle types
    are switched to the ``passenger`` shape class (cars instead of boxes) and
    pedestrian types to ``pedestrian`` with a slightly larger body so the small
    default actors (0.5 m) are visible at whole-map zoom.  Built-in ``DEFAULT_*``
    fallback types are left untouched.
    """
    connection = getattr(env, "_connection", None)
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


def load_sb3_model(method: str, checkpoint: Path, env: Any, device: str) -> Any:
    from algos.sb3_torch.sac import SceneRepresentationSAC

    if method == "hold35k":
        from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

        return use_actor_only(
            StabilitySAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
        )
    if method in ("hsac_mlp", "hsac_lstm"):
        from tools.v48_stability_v4.model import use_actor_only

        return use_actor_only(
            SceneRepresentationSAC.load(str(checkpoint), env=env, device=device, buffer_size=32)
        )
    return SceneRepresentationSAC.load(str(checkpoint), env=env, device=device, buffer_size=32)


def load_sequence_model(
    method: str, checkpoint: Path, env: Any, device: str
) -> tuple[Any, Any, float, int]:
    import torch

    from algos.transformer_rl import SumoStateTokenizer
    from configs.sb3_configs_baselines import make_sequence_baseline

    tokenizer = SumoStateTokenizer(env.observation_space)
    state_dim = tokenizer.state_dim
    payload = torch.load(checkpoint, map_location="cpu")
    model = make_sequence_baseline(method, state_dim, act_dim=payload.get("act_dim"))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    target_return = float(payload.get("target_return", 500.0))
    act_dim = int(payload.get("act_dim", 2))
    return model, tokenizer, target_return, act_dim


def _record(episode: int, seed: int, episode_return: float, decision_steps: int,
            environment_steps: int, info: dict[str, Any]) -> dict[str, Any]:
    raw_steps = int(info.get("raw_simulation_steps", environment_steps))
    success = bool(info.get("is_success", False))
    return {
        "episode": episode,
        "seed": seed,
        "episode_return": episode_return,
        "decision_steps": decision_steps,
        "environment_steps": environment_steps,
        "raw_steps": raw_steps,
        "completion_time_seconds": raw_steps * 0.1 if success else None,
        "success": success,
        "collision": bool(info.get("collision", False)),
        "off_route": bool(info.get("off_route", False)),
        "timeout": bool(info.get("max_time", False)),
        "traffic_variant": info.get("traffic_variant"),
    }


def live_sb3_episode(
    model: Any, env: Any, *, episode: int, seed: int, speed: float,
    beautify: bool = True,
) -> dict[str, Any]:
    if speed <= 0:
        raise ValueError("--speed must be positive")
    with source_evaluation_augmentation(model):
        observation, _ = env.reset(seed=seed)
        if beautify:
            _beautify_rendering(env)
        episode_return = 0.0
        policy_decisions = 0
        environment_steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            policy_decisions += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(info.get("undiscounted_reward", reward))
            environment_steps += 1
            time.sleep(0.1 * int(getattr(env, "action_repeat", 1)) / speed)
            if terminated or truncated:
                return _record(
                    episode, seed, episode_return, policy_decisions,
                    environment_steps, info,
                )


def live_sequence_episode(
    model: Any,
    env: Any,
    tokenizer: Any,
    method: str,
    act_dim: int,
    target_return: float,
    device: str,
    *,
    episode: int,
    seed: int,
    speed: float,
    beautify: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from algos.hybrid_action.hyar.sumo_adapter import hybrid_to_env_action
    from algos.transformer_rl.offline_dataset import hybrid_action_to_vector

    observation, _ = env.reset(seed=seed)
    if beautify:
        _beautify_rendering(env)
    episode_return = 0.0
    decision_steps = 0
    info: dict[str, Any] = {}
    hold = 0.1 * 3 / speed  # action_repeat == 3 in the shared env contract

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
            time.sleep(hold)
            if terminated or truncated:
                break
    else:
        states = [tokenizer(observation).astype(np.float32)]
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
            actions[-1] = (
                hybrid_action_to_vector(action) if method == "hybrid_dt" else action
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(info.get("undiscounted_reward", reward))
            decision_steps += 1
            time.sleep(hold)
            if terminated or truncated:
                break
            states.append(tokenizer(observation).astype(np.float32))
            actions.append(np.zeros(act_dim, dtype=np.float32))
            returns_to_go.append(
                returns_to_go[-1] - float(info.get("undiscounted_reward", reward))
            )
            timesteps.append(decision_steps)

    return _record(episode, seed, episode_return, decision_steps, decision_steps, info)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    n = len(records)
    returns = [float(r["episode_return"]) for r in records]
    raw_lengths = [int(r["raw_steps"]) for r in records]
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
        "std_return": float(np.std(returns)),
        "mean_raw_steps": float(np.mean(raw_lengths)),
        "mean_success_completion_time_seconds": (
            float(np.mean(completions)) if completions else None
        ),
    }


def run_cell(
    method: str,
    scene: str,
    *,
    episodes: int,
    speed: float,
    device: str,
    stop_event: Any = None,
    beautify: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import torch

    torch.set_num_threads(1)
    checkpoint = final_model_path(method, scene)
    contract = method_contract(method)
    env = make_live_env(method, scene)
    model_meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        if method in SEQUENCE_BASELINE_METHODS:
            model, tokenizer, target_return, act_dim = load_sequence_model(
                method, checkpoint, env, device
            )
            model_meta["target_return"] = target_return
            model_meta["act_dim"] = act_dim
            for index in range(episodes):
                if stop_event is not None and stop_event.is_set():
                    break
                episode_seed = SEED_START[method] + index
                np.random.seed(episode_seed + 600000)
                torch.manual_seed(episode_seed + 600000)
                if contract == "v4_8":
                    env._traffic_episode_index, env._traffic_roll = index, None
                records.append(
                    live_sequence_episode(
                        model, env, tokenizer, method, act_dim, target_return, device,
                        episode=index, seed=episode_seed, speed=speed,
                        beautify=beautify,
                    )
                )
        else:
            model = load_sb3_model(method, checkpoint, env, device)
            model_meta["_raw_steps_seen"] = int(getattr(model, "_raw_steps_seen", -1))
            for index in range(episodes):
                if stop_event is not None and stop_event.is_set():
                    break
                episode_seed = SEED_START[method] + index
                np.random.seed(episode_seed + 600000)
                torch.manual_seed(episode_seed + 600000)
                if contract == "v4_8":
                    env._traffic_episode_index, env._traffic_roll = index, None
                records.append(
                    live_sb3_episode(
                        model, env, episode=index, seed=episode_seed, speed=speed,
                        beautify=beautify,
                    )
                )
        if not records:
            raise RuntimeError("no episodes recorded (stopped before the first episode)")
        return {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "scene": scene,
            "scene_label": SCENE_TITLES[scene],
            "checkpoint_source_scene": checkpoint_source_scene(scene),
            "contract": contract,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "deployment": (
                "return_to_go" if method in SEQUENCE_BASELINE_METHODS else "actor_deterministic"
            ),
            "seed_start": SEED_START[method],
            "episodes": len(records),
            "model_meta": model_meta,
            "summary": summarize(records),
            "episode_records": records,
        }
    finally:
        env.close()


def _plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    output.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    by_key = {(row["method"], row["scene"]): row for row in rows}
    present_scenes = [s for s in ALL_SCENES if any((m, s) in by_key for m in ALL_METHODS)]
    present_methods = [
        m for m in ALL_METHODS if any((m, s) in by_key for s in present_scenes)
    ]
    if not present_scenes or not present_methods:
        return
    metrics = (
        ("success_rate", "Success"),
        ("collision_rate", "Collision"),
        ("timeout_rate", "Timeout"),
    )
    figure, axes = plt.subplots(
        1, len(present_scenes), figsize=(6 * len(present_scenes), 6),
        sharey=True, squeeze=False,
    )
    x = np.arange(len(metrics))
    width = 0.08
    for scene_index, scene in enumerate(present_scenes):
        axis = axes.flat[scene_index]
        for method_index, method in enumerate(present_methods):
            values = [
                100.0 * float(by_key[(method, scene)]["summary"][metric])
                for metric, _ in metrics
            ]
            offset = (method_index - (len(present_methods) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(
            f"({chr(97 + scene_index)}) {SCENE_TITLES[scene]}", loc="left"
        )
        axis.set_xticks(x, [name for _, name in metrics])
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Outcome")
    axes.flat[0].set_ylabel("Episodes (%)")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    figure.suptitle(
        "Three-scene visual verification | final models | live SUMO-GUI playback",
        fontsize=14,
    )
    figure.text(
        0.5, 0.01,
        f"Deterministic evaluation | bars are per-scene percentages "
        f"(seeds {SEED_START[ALL_METHODS[0]]}…)",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=[0, 0.03, 1, 0.93])
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"three_scene_summary.{extension}", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_returns(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    output.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(row["method"], row["scene"]): row for row in rows}
    present_scenes = [s for s in ALL_SCENES if any((m, s) in by_key for m in ALL_METHODS)]
    present_methods = [
        m for m in ALL_METHODS if any((m, s) in by_key for s in present_scenes)
    ]
    if not present_scenes or not present_methods:
        return
    figure, axes = plt.subplots(
        1, len(present_scenes), figsize=(6 * len(present_scenes), 5),
        sharey=True, squeeze=False,
    )
    for scene_index, scene in enumerate(present_scenes):
        axis = axes.flat[scene_index]
        for method in present_methods:
            row = by_key[(method, scene)]
            axis.plot(
                [int(rec["episode"]) + 1 for rec in row["episode_records"]],
                [float(rec["episode_return"]) for rec in row["episode_records"]],
                marker="o", ms=3, lw=1.3,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            )
        axis.set_title(f"({chr(97 + scene_index)}) {SCENE_TITLES[scene]}", loc="left")
        axis.set_xlabel("Evaluation episode")
        axis.grid(alpha=0.25)
    axes.flat[0].set_ylabel("Episode return")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="best", ncol=2)
    figure.suptitle("Three-scene visual verification | episode returns", fontsize=14)
    figure.text(0.5, 0.01, "No smoothing or cross-seed aggregation", ha="center", fontsize=8)
    figure.tight_layout(rect=[0, 0.03, 1, 0.93])
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"three_scene_returns.{extension}", dpi=150, bbox_inches="tight")
    plt.close(figure)


def execute_run(
    methods: list[str],
    scenes: list[str],
    *,
    episodes: int,
    speed: float,
    device: str,
    output: Path,
    emit: Callable[[str], None] | None = None,
    stop_event: Any = None,
    beautify: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    """Run the selected cells sequentially, tolerating per-cell failures.

    Returns ``(rows, failures, output)`` where ``rows`` are completed cells and
    ``failures`` are ``{"method", "scene", "error"}`` records.
    """
    def log(text: str) -> None:
        if emit is not None:
            emit(text)
        print(text, flush=True)

    _install_visible_sumo_spawn()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        "schema_version": "three-scene-visual-verification-live/v1",
        "status": "running",
        "training_performed": False,
        "fabricated_values": False,
        "live_sumogui": True,
        "deterministic": True,
        "episodes_per_cell": episodes,
        "speed": speed,
        "device": device,
        "beautify_rendering": beautify,
        "methods": list(methods),
        "scenes": list(scenes),
        "output_directory": str(output),
        "completed_cells": [],
        "failed_cells": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write(output / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for method in methods:
            for scene in scenes:
                if stop_event is not None and stop_event.is_set():
                    break
                safe = f"{method}__{scene}"
                log(f"[start] {METHOD_LABELS[method]} / {scene} (SUMO-GUI should be visible)")
                try:
                    row = run_cell(
                        method, scene, episodes=episodes, speed=speed,
                        device=device, stop_event=stop_event, beautify=beautify,
                    )
                except BaseException as exc:
                    failure = {"method": method, "scene": scene, "error": repr(exc)}
                    failures.append(failure)
                    manifest["failed_cells"].append(safe)
                    write(output / "manifest.json", manifest)
                    write(
                        output / "cells" / f"{safe}.json",
                        {"method": method, "scene": scene, "status": "failed", "error": repr(exc)},
                    )
                    log(f"[error] {METHOD_LABELS[method]} / {scene}: {exc!r}")
                    continue
                rows.append(row)
                write(output / "cells" / f"{safe}.json", row)
                manifest["completed_cells"].append(safe)
                write(output / "manifest.json", manifest)
                summary = row["summary"]
                log(
                    f"[done] {METHOD_LABELS[method]} / {scene}: "
                    f"success={summary['success_rate']:.3f}, "
                    f"collision={summary['collision_rate']:.3f}, "
                    f"timeout={summary['timeout_rate']:.3f}"
                )
        if rows:
            _plot_summary(rows, output / "figures")
            _plot_returns(rows, output / "figures")
        write(output / "summary.json", {"rows": rows, "failures": failures, "episodes": episodes})
        stopped = stop_event is not None and stop_event.is_set()
        manifest.update(
            {
                "status": "stopped" if stopped else "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
                "figure_directory": str((output / "figures").resolve()),
            }
        )
        write(output / "manifest.json", manifest)
        (output / "README.md").write_text(
            "# Three-scene visual verification (live SUMO-GUI)\n\n"
            "Live playback of selected methods' final models in the selected "
            "scenarios. No checkpoints were modified.\n\n"
            "- Cells: `cells/{method}__{scene}.json`\n"
            "- Aggregate: `summary.json`\n"
            "- Figures: `figures/three_scene_summary.*` and `figures/three_scene_returns.*`\n"
            "- Provenance: `manifest.json`\n\n"
            f"Deterministic, {episodes} episodes per cell.\n",
            encoding="utf-8",
        )
        log(f"[completed] {output}")
        return rows, failures, output
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error": repr(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
            }
        )
        write(output / "manifest.json", manifest)
        raise


def _new_output_dir(root: Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / tag / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / tag / f"{stamp}_{suffix:02d}"
    return candidate


def run(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    methods = list(args.methods or ALL_METHODS)
    scenes = list(args.scenes or ALL_SCENES)
    output = args.output_dir or _new_output_dir(DEFAULT_OUTPUT_ROOT, "live")
    _, failures, output = execute_run(
        methods, scenes, episodes=args.episodes, speed=args.speed,
        device=device, output=output, beautify=not args.no_beautify,
    )
    if failures:
        print(
            f"[warn] {len(failures)}/{len(methods) * len(scenes)} cells failed; "
            f"see manifest.json",
            file=sys.stderr,
        )
    return output


def run_gui(args: argparse.Namespace) -> None:
    """Interactive Tkinter selector: tick methods/scenes, then start playback."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import messagebox, ttk

    initial_device = resolve_device(args.device)

    root = tk.Tk()
    root.title("三场景可视化验证 — SUMO-GUI 实时回放")
    root.geometry("760x640")

    method_vars = {m: tk.BooleanVar(value=True) for m in ALL_METHODS}
    scene_vars = {s: tk.BooleanVar(value=True) for s in ALL_SCENES}
    episodes_var = tk.IntVar(value=EPISODES_DEFAULT)
    speed_var = tk.DoubleVar(value=4.0)
    device_var = tk.StringVar(value=initial_device)
    beautify_var = tk.BooleanVar(value=True)

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    # ---- method / scene selection -------------------------------------------------
    select = ttk.Frame(main)
    select.pack(fill="x")
    methods_frame = ttk.LabelFrame(select, text="方法", padding=8)
    methods_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
    for idx, method in enumerate(ALL_METHODS):
        ttk.Checkbutton(
            methods_frame, text=METHOD_LABELS[method], variable=method_vars[method]
        ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=4, pady=2)

    def _set_methods(value: bool) -> None:
        for var in method_vars.values():
            var.set(value)

    ttk.Button(methods_frame, text="全选", width=8, command=lambda: _set_methods(True)).grid(
        row=5, column=0, sticky="w", pady=(6, 0)
    )
    ttk.Button(methods_frame, text="清空", width=8, command=lambda: _set_methods(False)).grid(
        row=5, column=1, sticky="w", pady=(6, 0)
    )

    scenes_frame = ttk.LabelFrame(select, text="场景", padding=8)
    scenes_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))
    for idx, scene in enumerate(ALL_SCENES):
        ttk.Checkbutton(
            scenes_frame, text=SCENE_TITLES[scene], variable=scene_vars[scene]
        ).pack(anchor="w", pady=3)

    def _set_scenes(value: bool) -> None:
        for var in scene_vars.values():
            var.set(value)

    scene_btns = ttk.Frame(scenes_frame)
    scene_btns.pack(anchor="w", pady=(6, 0))
    ttk.Button(scene_btns, text="全选", width=8, command=lambda: _set_scenes(True)).pack(
        side="left", padx=(0, 4)
    )
    ttk.Button(scene_btns, text="清空", width=8, command=lambda: _set_scenes(False)).pack(
        side="left"
    )

    # ---- controls -----------------------------------------------------------------
    controls = ttk.Frame(main)
    controls.pack(fill="x", pady=(10, 6))
    ttk.Label(controls, text="每格 episode 数:").pack(side="left")
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
    ttk.Checkbutton(
        controls, text="渲染美化", variable=beautify_var,
    ).pack(side="left", padx=(0, 8))

    # ---- buttons ------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(0, 6))
    start_btn = ttk.Button(buttons, text="开始回放")
    start_btn.pack(side="left")
    stop_btn = ttk.Button(buttons, text="停止", state="disabled")
    stop_btn.pack(side="left", padx=(8, 0))
    output_label = ttk.Label(buttons, text="", foreground="#666666")
    output_label.pack(side="right")

    # ---- status log ---------------------------------------------------------------
    status = tk.Text(main, height=16, wrap="none", state="disabled")
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
                        append_status(f"完成：{len(rows)} 个 cell 成功，{len(failures)} 个失败。")
                        for f in failures:
                            append_status(f"  失败 {f['method']}__{f['scene']}: {f['error']}")
                    else:
                        append_status(f"全部 {len(rows)} 个 cell 完成。")
                    append_status(f"结果目录: {out}")
                    messagebox.showinfo("完成", f"{len(rows)} 个 cell 完成\n{out}")
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    def worker(methods: list[str], scenes: list[str], episodes: int, speed: float,
               device: str, output: Path, beautify: bool) -> None:
        try:
            rows, failures, out = execute_run(
                methods, scenes, episodes=episodes, speed=speed, device=device,
                output=output, emit=lambda text: q.put(("log", text)),
                stop_event=stop_event, beautify=beautify,
            )
        except BaseException as exc:  # pragma: no cover - catastrophic failure
            q.put(("log", f"[fatal] {exc!r}"))
            rows, failures, out = [], [], output
        q.put(("finish", (rows, failures, out)))

    def on_start() -> None:
        methods = [m for m in ALL_METHODS if method_vars[m].get()]
        scenes = [s for s in ALL_SCENES if scene_vars[s].get()]
        if not methods or not scenes:
            messagebox.showwarning("未选择", "请至少选择一个方法和一个场景。")
            return
        episodes = int(episodes_var.get())
        speed = float(speed_var.get())
        device = resolve_device(device_var.get())
        if episodes <= 0 or speed <= 0:
            messagebox.showwarning("参数错误", "episode 数和速度必须为正。")
            return
        output = _new_output_dir(DEFAULT_OUTPUT_ROOT, "live")
        output_label.configure(text=str(output))
        append_status(f"输出目录: {output}")
        append_status(f"已选 {len(methods)} 个方法 × {len(scenes)} 个场景，"
                      f"每格 {episodes} 集，速度 {speed}，设备 {device}")
        start_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        stop_event.clear()
        threading.Thread(
            target=worker,
            args=(methods, scenes, episodes, speed, device, output, bool(beautify_var.get())),
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
    parser.add_argument("--gui", action="store_true",
                        help="interactive selector window (ignore CLI method/scene filters)")
    parser.add_argument("--device", default="cuda", help="cuda | cpu | auto")
    parser.add_argument("--episodes", type=int, default=EPISODES_DEFAULT)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument(
        "--no-beautify", action="store_true",
        help="disable the post-reset rendering beautification (car/pedestrian "
             "silhouettes); boxes and small pedestrians are drawn instead",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(ALL_METHODS), default=None,
        help="restrict to a subset of methods (CLI mode only)",
    )
    parser.add_argument(
        "--scenes", nargs="+", choices=tuple(ALL_SCENES), default=None,
        help="restrict to a subset of scenarios (CLI mode only)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="CLI mode only: single hold35k/carla cell, one episode",
    )
    args = parser.parse_args(argv)
    if args.gui:
        run_gui(args)
        return 0
    if args.smoke:
        args.methods = args.methods or ["hold35k"]
        args.scenes = args.scenes or ["carla"]
        args.episodes = 1
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_METHODS",
    "ALL_SCENES",
    "SCENES",
    "SEED_START",
    "make_live_env",
    "execute_run",
    "run",
    "run_cell",
    "run_gui",
    "main",
]
