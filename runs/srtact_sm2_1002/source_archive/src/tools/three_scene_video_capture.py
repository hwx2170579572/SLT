"""Headless trajectory capture + top-down video rendering for the three-scene methods.

The live SUMO-GUI playback (``three_scene_visual_verification.py``) crashes on the
SMARTS-contract high-density scenes (``cross`` / ``cross_left``) because sumo-gui
itself exits on startup for those overlays, whereas the *headless* ``sumo`` binary
runs the identical mission correctly in well under a second.  This module therefore
decouples the two concerns:

1. **Capture** -- drive every method's final model through the three released
   scenarios with the headless ``sumo`` binary and, at each decision step, record a
   rich trajectory snapshot (ego pose/speed/lane, every visible vehicle and
   pedestrian, the chosen action and the outcome flags) straight from the live TraCI
   connection.  This is fast and deterministic and needs no GUI.

2. **Render** -- redraw that trajectory as a top-down animation (road network from
   ``traci.lane.getShape``, ego highlighted, neighbours as boxes, pedestrians as
   dots, collision / success banners) and encode it to ``.mp4`` via imageio-ffmpeg.

The resulting videos and trajectory JSONs are the substrate for the behavioural
deep-analysis step; nothing here re-trains or mutates a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
    SEQUENCE_BASELINE_METHODS,
    checkpoint_source_scene,
    density,
    sha256,
    write,
)
from tools.three_scene_visual_verification import (  # noqa: E402
    METHOD_LABELS,
    METHOD_COLORS,
    SCENE_TITLES,
    SEED_START,
    final_model_path,
    load_sb3_model,
    load_sequence_model,
    method_contract,
)

DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "visual_verification"
VIDEO_FPS = 10
CAMERA_HALF_WIDTH = 80.0   # metres; the top-down view follows the ego
CAMERA_HALF_HEIGHT = 50.0


# --------------------------------------------------------------------------- #
# Environment / network geometry
# --------------------------------------------------------------------------- #
def make_headless_env(method: str, scene: str) -> Any:
    """Same env as ``make_live_env`` but with ``render_mode=None`` (headless sumo)."""
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
        render_mode=None,
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
        sumo_args=(),
    )


def _capture_network(env: Any) -> dict[str, Any]:
    """Collect the road-lane polyline geometry and its bounding box once per scene."""
    connection = env._connection
    lanes: list[list[list[float]]] = []
    xs: list[float] = []
    ys: list[float] = []
    try:
        lane_ids = list(connection.lane.getIDList())
    except BaseException:
        lane_ids = []
    for lane_id in lane_ids:
        try:
            shape = connection.lane.getShape(lane_id)
        except BaseException:
            continue
        polyline = [[float(x), float(y)] for x, y in shape]
        if len(polyline) < 2:
            continue
        lanes.append(polyline)
        for x, y in polyline:
            xs.append(x)
            ys.append(y)
    if not lanes:
        return {"lanes": [], "bbox": [0.0, 0.0, 1.0, 1.0]}
    return {
        "lanes": lanes,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
    }


# --------------------------------------------------------------------------- #
# Trajectory capture
# --------------------------------------------------------------------------- #
def _snapshot(
    env: Any, step: int, action: np.ndarray, info: dict[str, Any]
) -> dict[str, Any]:
    connection = env._connection
    ego_id = env.specification.ego_id
    snap: dict[str, Any] = {
        "step": step,
        "action": [float(a) for a in np.asarray(action).reshape(-1)],
        "events": {
            "collision": bool(info.get("collision", False)),
            "off_route": bool(info.get("off_route", False)),
            "success": bool(info.get("is_success", False)),
            "timeout": bool(info.get("max_time", False)),
        },
        "ego": None,
        "vehicles": [],
        "pedestrians": [],
    }
    try:
        x, y = connection.vehicle.getPosition(ego_id)
        snap["ego"] = {
            "x": float(x),
            "y": float(y),
            "angle": float(connection.vehicle.getAngle(ego_id)),
            "speed": float(connection.vehicle.getSpeed(ego_id)),
            "lane": str(connection.vehicle.getLaneID(ego_id)),
            "road": str(connection.vehicle.getRoadID(ego_id)),
        }
    except BaseException:
        pass
    try:
        vehicle_ids = list(connection.vehicle.getIDList())
    except BaseException:
        vehicle_ids = []
    for vid in vehicle_ids:
        if vid == ego_id:
            continue
        try:
            px, py = connection.vehicle.getPosition(vid)
            snap["vehicles"].append(
                {
                    "id": str(vid),
                    "x": float(px),
                    "y": float(py),
                    "angle": float(connection.vehicle.getAngle(vid)),
                    "speed": float(connection.vehicle.getSpeed(vid)),
                    "length": float(connection.vehicle.getLength(vid)),
                    "width": float(connection.vehicle.getWidth(vid)),
                }
            )
        except BaseException:
            continue
    try:
        person_ids = list(connection.person.getIDList())
    except BaseException:
        person_ids = []
    for pid in person_ids:
        try:
            px, py = connection.person.getPosition(pid)
            snap["pedestrians"].append(
                {"id": str(pid), "x": float(px), "y": float(py)}
            )
        except BaseException:
            continue
    return snap


def _record(episode: int, seed: int, decision_steps: int, info: dict[str, Any],
            frames: list[dict[str, Any]]) -> dict[str, Any]:
    raw_steps = int(info.get("raw_simulation_steps", decision_steps))
    success = bool(info.get("is_success", False))
    return {
        "episode": episode,
        "seed": seed,
        "decision_steps": decision_steps,
        "raw_steps": raw_steps,
        "completion_time_seconds": raw_steps * 0.1 if success else None,
        "success": success,
        "collision": bool(info.get("collision", False)),
        "off_route": bool(info.get("off_route", False)),
        "timeout": bool(info.get("max_time", False)),
        "traffic_variant": info.get("traffic_variant"),
        "frames": frames,
    }


def run_trajectory_episode(
    model: Any,
    env: Any,
    method: str,
    *,
    episode: int,
    seed: int,
    device: str,
    tokenizer: Any = None,
    target_return: float = 0.0,
    act_dim: int = 2,
) -> dict[str, Any]:
    """Run one episode and capture a per-decision-step trajectory snapshot."""
    import torch

    if method in SEQUENCE_BASELINE_METHODS:
        from algos.hybrid_action.hyar.sumo_adapter import hybrid_to_env_action
        from algos.transformer_rl.offline_dataset import hybrid_action_to_vector

        observation, _ = env.reset(seed=seed)
        frames: list[dict[str, Any]] = []
        decision_steps = 0
        info: dict[str, Any] = {}
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
                    torch.as_tensor(np.asarray(returns_to_go, dtype=np.float32), device=device),
                    torch.as_tensor(np.asarray(timesteps, dtype=np.int64), device=device),
                )
            action = action_t.detach().cpu().numpy()
            if method == "hyar":
                with torch.no_grad():
                    _, discrete_index, continuous = model.act(
                        torch.as_tensor(np.asarray([tokenizer(observation)], dtype=np.float32), device=device),
                        deterministic=True,
                    )
                action = hybrid_to_env_action(
                    int(discrete_index.item()), continuous.detach().cpu().numpy()
                )
            else:
                actions[-1] = (
                    hybrid_action_to_vector(action) if method == "hybrid_dt" else action
                )
            observation, reward, terminated, truncated, info = env.step(action)
            decision_steps += 1
            frames.append(_snapshot(env, decision_steps, action, info))
            if terminated or truncated:
                break
            states.append(tokenizer(observation).astype(np.float32))
            actions.append(np.zeros(act_dim, dtype=np.float32))
            returns_to_go.append(
                returns_to_go[-1] - float(info.get("undiscounted_reward", reward))
            )
            timesteps.append(decision_steps)
    else:
        with source_evaluation_augmentation(model):
            observation, _ = env.reset(seed=seed)
            frames: list[dict[str, Any]] = []
            decision_steps = 0
            info: dict[str, Any] = {}
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                decision_steps += 1
                frames.append(_snapshot(env, decision_steps, action, info))
                if terminated or truncated:
                    break

    return _record(episode, seed, decision_steps, info, frames)


# --------------------------------------------------------------------------- #
# Top-down rendering
# --------------------------------------------------------------------------- #
def _to_plot(x: float, y: float) -> tuple[float, float]:
    # SUMO y grows southward; matplotlib y grows upward.
    return float(x), -float(y)


def render_topdown(
    frames: list[dict[str, Any]],
    network: dict[str, Any],
    output_mp4: Path,
    *,
    method: str,
    scene: str,
    episode: int,
) -> Path:
    """Render a trajectory as an ego-following top-down mp4."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return output_mp4

    lanes = network.get("lanes", [])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)  # 1280x800, divisible by 16
    writer = imageio.get_writer(output_mp4, fps=VIDEO_FPS, codec="libx264", quality=7)
    try:
        for frame in frames:
            ax.clear()
            ax.set_aspect("equal")
            ax.axis("off")
            # camera follows the ego
            ego = frame.get("ego") or {}
            cx = float(ego.get("x", 0.0))
            cy = -float(ego.get("y", 0.0))
            ax.set_xlim(cx - CAMERA_HALF_WIDTH, cx + CAMERA_HALF_WIDTH)
            ax.set_ylim(cy - CAMERA_HALF_HEIGHT, cy + CAMERA_HALF_HEIGHT)

            # road network
            for polyline in lanes:
                px = [p[0] for p in polyline]
                py = [-p[1] for p in polyline]
                ax.plot(px, py, color="#c8c8c8", lw=1.0, zorder=1)

            # neighbours
            for veh in frame.get("vehicles", []):
                length = float(veh.get("length", 5.0))
                width = float(veh.get("width", 1.8))
                angle = 90.0 - float(veh.get("angle", 0.0))
                x, y = _to_plot(float(veh["x"]), float(veh["y"]))
                ax.add_patch(
                    Rectangle(
                        (x - width / 2, y - length / 2),
                        width, length, angle=angle, rotation_point="center",
                        facecolor="#7f7f7f", edgecolor="none", zorder=3,
                    )
                )

            # pedestrians
            for ped in frame.get("pedestrians", []):
                x, y = _to_plot(float(ped["x"]), float(ped["y"]))
                ax.add_patch(Circle((x, y), 0.7, facecolor="#ff9800", edgecolor="none", zorder=4))

            # ego
            if ego:
                length = 4.7
                width = 1.8
                angle = 90.0 - float(ego.get("angle", 0.0))
                x, y = _to_plot(float(ego.get("x", 0.0)), float(ego.get("y", 0.0)))
                ax.add_patch(
                    Rectangle(
                        (x - width / 2, y - length / 2),
                        width, length, angle=angle, rotation_point="center",
                        facecolor="#d62728", edgecolor="#7f0000", lw=1.0, zorder=5,
                    )
                )

            # annotation
            events = frame.get("events", {})
            status = "SUCCESS" if events.get("success") else (
                "COLLISION" if events.get("collision") else (
                    "OFF-ROUTE" if events.get("off_route") else ""
                )
            )
            if events.get("timeout"):
                status = (status + " TIMEOUT").strip()
            text = (
                f"{METHOD_LABELS[method]} | {SCENE_TITLES[scene]} | ep {episode}\n"
                f"t = {frame['step'] * 0.3:.1f}s   "
                f"speed = {float(ego.get('speed', 0.0)):.1f} m/s   "
                f"lane = {ego.get('lane', '?')}\n"
                f"action = [{frame['action'][0]:+.2f}, {frame['action'][1]:+.2f}]   "
                f"{status}"
            )
            ax.text(
                0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
            if status:
                color = "#1a9850" if "SUCCESS" in status else "#d62728"
                ax.text(
                    0.98, 0.02, status, transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=16, family="monospace", color=color, weight="bold",
                )

            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(rgb)
    finally:
        writer.close()
        plt.close(fig)
    return output_mp4


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _network_for_scene(scene: str) -> dict[str, Any]:
    """Capture the road-lane geometry for a scene once, using a throwaway env."""
    env = make_headless_env("hold35k", scene)
    try:
        env.reset(seed=SEED_START["hold35k"])
        return _capture_network(env)
    finally:
        env.close()


def run_capture(
    methods: list[str],
    scenes: list[str],
    *,
    episodes: int,
    device: str,
    output: Path,
    emit: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    import torch

    def log(text: str) -> None:
        if emit is not None:
            emit(text)
        print(text, flush=True)

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        "schema_version": "three-scene-video-capture/v1",
        "status": "running",
        "training_performed": False,
        "fabricated_values": False,
        "headless_sumo": True,
        "deterministic": True,
        "episodes_per_cell": episodes,
        "device": device,
        "methods": list(methods),
        "scenes": list(scenes),
        "output_directory": str(output),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_cells": [],
        "failed_cells": [],
    }
    write(output / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    network_cache: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        network_cache[scene] = _network_for_scene(scene)
        log(f"[network] {scene}: {len(network_cache[scene]['lanes'])} lanes")

    for method in methods:
        for scene in scenes:
            safe = f"{method}__{scene}"
            log(f"[start] {METHOD_LABELS[method]} / {scene}")
            try:
                checkpoint = final_model_path(method, scene)
                env = make_headless_env(method, scene)
                try:
                    model_meta: dict[str, Any] = {}
                    tokenizer = None
                    target_return = 0.0
                    act_dim = 2
                    if method in SEQUENCE_BASELINE_METHODS:
                        model, tokenizer, target_return, act_dim = load_sequence_model(
                            method, checkpoint, env, device
                        )
                        model_meta["target_return"] = target_return
                        model_meta["act_dim"] = act_dim
                    else:
                        model = load_sb3_model(method, checkpoint, env, device)
                        model_meta["_raw_steps_seen"] = int(getattr(model, "_raw_steps_seen", -1))

                    contract = method_contract(method)
                    episode_records: list[dict[str, Any]] = []
                    for index in range(episodes):
                        episode_seed = SEED_START[method] + index
                        np.random.seed(episode_seed + 600000)
                        torch.manual_seed(episode_seed + 600000)
                        if contract == "v4_8":
                            env._traffic_episode_index, env._traffic_roll = index, None
                        rec = run_trajectory_episode(
                            model, env, method, episode=index, seed=episode_seed,
                            device=device, tokenizer=tokenizer,
                            target_return=target_return, act_dim=act_dim,
                        )
                        episode_records.append(rec)
                        if index == 0:
                            rec["video"] = str(
                                render_topdown(
                                    rec["frames"], network_cache[scene],
                                    output / "videos" / f"{safe}_ep0.mp4",
                                    method=method, scene=scene, episode=0,
                                )
                            )

                    n = len(episode_records)
                    row = {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "scene": scene,
                        "scene_label": SCENE_TITLES[scene],
                        "checkpoint_source_scene": checkpoint_source_scene(scene),
                        "contract": contract,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "seed_start": SEED_START[method],
                        "episodes": n,
                        "model_meta": model_meta,
                        "summary": {
                            "success_rate": sum(int(r["success"]) for r in episode_records) / n,
                            "collision_rate": sum(int(r["collision"]) for r in episode_records) / n,
                            "off_route_rate": sum(int(r["off_route"]) for r in episode_records) / n,
                            "timeout_rate": sum(int(r["timeout"]) for r in episode_records) / n,
                        },
                        "episode_records": episode_records,
                    }
                finally:
                    env.close()
                rows.append(row)
                write(output / "cells" / f"{safe}.json", row)
                manifest["completed_cells"].append(safe)
                write(output / "manifest.json", manifest)
                s = row["summary"]
                log(
                    f"[done] {METHOD_LABELS[method]} / {scene}: "
                    f"success={s['success_rate']:.3f}, collision={s['collision_rate']:.3f}, "
                    f"off_route={s['off_route_rate']:.3f}"
                )
            except BaseException as exc:
                failure = {"method": method, "scene": scene, "error": repr(exc)}
                failures.append(failure)
                manifest["failed_cells"].append(safe)
                write(output / "manifest.json", manifest)
                log(f"[error] {METHOD_LABELS[method]} / {scene}: {exc!r}")

    write(output / "summary.json", {"rows": rows, "failures": failures, "episodes": episodes})
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.time() - started,
        }
    )
    write(output / "manifest.json", manifest)
    return rows, failures, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(ALL_METHODS), default=None)
    parser.add_argument("--scenes", nargs="+", choices=tuple(ALL_SCENES), default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="single hold35k/carla cell, one episode")
    args = parser.parse_args(argv)
    methods = list(args.methods or ALL_METHODS)
    scenes = list(args.scenes or ALL_SCENES)
    if args.smoke:
        methods, scenes, args.episodes = ["hold35k"], ["carla"], 1
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / "video_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_capture(methods, scenes, episodes=args.episodes, device=args.device, output=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
