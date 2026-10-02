"""Evaluate three existing policies on six scenes and export visual summaries.

This is an evaluation-only entry point.  It never trains, changes, or copies
the source checkpoints.  The three bindings are intentionally explicit:

* MST+SLT default: the phase-1 ``exact_final`` deployment recorded in the
  frozen six-scene plan;
* v4.8 ``lr_half``: the completed phase-2 screen ``final_model.zip``;
* v4.13 ``tau_half``: the completed phase-2 screen ``final_model.zip``.

Each method/scenario cell uses deterministic evaluation seeds 10000--10019
(20 episodes), the existing evaluation partition, and the decoder recorded
with that existing model.  Results are written into a new timestamped run
directory unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch import evaluate_model_detailed  # noqa: E402
from tools.independent_v2_5m6s100e_v1_common import (  # noqa: E402
    SCENARIOS,
    density_setting,
    load_protocol,
)
from tools.train_independent_v2_5m6s100e_v1 import (  # noqa: E402
    _make_environment_factory,
)


DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "experiments"
    / "independent_v2_five_methods_six_scenarios_100ep_v1"
    / "protocol.json"
)
DEFAULT_PHASE1_PLAN = (
    PROJECT_ROOT / "results_phase1_checkpoint_diagnostics_v1" / "plan.json"
)
DEFAULT_PHASE2_ROOT = PROJECT_ROOT / "results_phase2_runtime_v2" / "screen"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_visual_eval_three_methods_20ep_v1"
EVALUATION_SEED_START = 10_000
EPISODES = 20
METHOD_ORDER = ("mst_slt_default", "v4_8_lr_half", "v4_13_tau_half")
SCENE_TITLES = {
    "left_turn": "Left Turn",
    "cross": "Cross",
    "roundabout_easy": "Roundabout-A",
    "roundabout_medium": "Roundabout-B",
    "roundabout": "Roundabout-C",
    "carla": "CARLA",
}
METHOD_LABELS = {
    "mst_slt_default": "MST+SLT default",
    "v4_8_lr_half": "v4.8 lr_half",
    "v4_13_tau_half": "v4.13 tau_half",
}
METHOD_COLORS = {
    "mst_slt_default": "#666666",
    "v4_8_lr_half": "#0072B2",
    "v4_13_tau_half": "#009E73",
}
# Keep the live SUMO-GUI window on the primary desktop and large enough for
# direct inspection.  These arguments are only used by ``--live``; headless
# evaluation keeps the original SUMO command unchanged.
LIVE_SUMO_ARGS = ("--window-pos", "40,40", "--window-size", "1280,800")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _new_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stamp}_{suffix:02d}"
    candidate.mkdir(parents=True)
    return candidate


def _phase1_exact_final(plan_path: Path, scene: str) -> dict[str, Any]:
    plan = read_json(plan_path)
    # ``jobs`` contains the exact-final binding and its run metadata; the
    # separate ``sources`` section contains checkpoint-diagnostic entries.
    source_rows = plan.get("jobs", [])
    matches = [
        row
        for row in source_rows
        if row.get("method") == "mst_slt"
        and row.get("scenario") == scene
        and row.get("kind") == "exact_final"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one MST+SLT exact_final source for {scene}, got {len(matches)}"
        )
    row = dict(matches[0])
    checkpoint = (PROJECT_ROOT / str(row["checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if row.get("checkpoint_sha256") and sha256(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError(f"MST+SLT checkpoint hash drifted: {checkpoint}")
    run_dir = (PROJECT_ROOT / str(row["run"])).resolve()
    arguments_path = run_dir / "arguments.json"
    if not arguments_path.is_file():
        raise FileNotFoundError(arguments_path)
    arguments = read_json(arguments_path)
    requested = arguments.get("requested_raw_steps")
    if not isinstance(requested, dict):
        raise ValueError(f"Missing requested_raw_steps in {arguments_path}")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "run_dir": run_dir,
        "arguments_path": arguments_path,
        "requested_arguments": dict(requested),
        "decoder": str(row.get("decoder", "native_deterministic")),
        "source_kind": "phase1_plan_exact_final",
        "source_plan": plan_path,
    }


def _phase2_candidate(
    phase2_root: Path, method: str, scene: str, candidate: str
) -> dict[str, Any]:
    run_dir = (phase2_root / f"{method}__{scene}__{candidate}__seed0").resolve()
    status_path = run_dir / "status.json"
    arguments_path = run_dir / "arguments.json"
    checkpoint = run_dir / "final_model.zip"
    for path in (status_path, arguments_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise ValueError(f"Phase-2 source is not completed: {run_dir}")
    arguments = read_json(arguments_path)
    requested = arguments.get("inherited_environment_arguments")
    if not isinstance(requested, dict):
        raise ValueError(f"Missing inherited environment arguments in {arguments_path}")
    if arguments.get("method") != method or arguments.get("scenario") != scene:
        raise ValueError(f"Phase-2 source identity mismatch: {run_dir}")
    if arguments.get("candidate", {}).get("id") != candidate:
        raise ValueError(f"Phase-2 candidate mismatch: {run_dir}")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "run_dir": run_dir,
        "arguments_path": arguments_path,
        "requested_arguments": dict(requested),
        "decoder": str(arguments.get("decoder", "")),
        "source_kind": f"phase2_screen_{candidate}",
        "source_status": status_path,
    }


def resolve_sources(
    *, phase1_plan: Path, phase2_root: Path, scene: str, method: str
) -> dict[str, Any]:
    if method == "mst_slt_default":
        return _phase1_exact_final(phase1_plan, scene)
    if method == "v4_8_lr_half":
        return _phase2_candidate(phase2_root, "v4_8", scene, "lr_half")
    if method == "v4_13_tau_half":
        return _phase2_candidate(phase2_root, "v4_13", scene, "tau_half")
    raise ValueError(f"Unknown method binding {method}")


def _model_class(method: str) -> type[Any]:
    if method == "mst_slt_default":
        from algos.sb3_torch.sac import SceneRepresentationSAC

        return SceneRepresentationSAC
    if method == "v4_8_lr_half":
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45

        return ConfidentActorFusionSACV45
    if method == "v4_13_tau_half":
        from algos.sb3_torch.sac_v4_13_model import (
            GradientIsolatedTemperedJointSupportSACV413,
        )

        return GradientIsolatedTemperedJointSupportSACV413
    raise ValueError(f"Unknown method binding {method}")


def _load_model(method: str, source: dict[str, Any], env: Any, device: str) -> Any:
    model_class = _model_class(method)
    checkpoint = Path(source["checkpoint"])
    decoder = str(source["decoder"])
    if method == "v4_8_lr_half":
        from tools.action_diagnostics_v4_6 import load_model_for_deployment

        return load_model_for_deployment(
            model_class, checkpoint, decoder=decoder, env=env, device=device
        )
    if method == "v4_13_tau_half":
        from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13

        return load_model_for_deployment_v4_13(
            model_class, checkpoint, decoder=decoder, env=env, device=device
        )
    return model_class.load(checkpoint, env=env, device=device)


def _make_env(
    *, protocol: dict[str, Any], source: dict[str, Any], scene: str, method: str, output: Path
) -> Any:
    # The method binding names are deliberately different from the protocol
    # method names; use the exact adapter names expected by the factory.
    adapter = {"mst_slt_default": "base", "v4_8_lr_half": "v4_8", "v4_13_tau_half": "v4_13"}[method]
    requested = dict(source["requested_arguments"])
    requested["scenario"] = scene
    requested["gui"] = False
    requested["evaluation_split"] = "validation"
    requested.setdefault("history_steps", 10)
    requested.setdefault("neighbors", 5)
    requested.setdefault("path_length", 10)
    requested.setdefault("action_repeat", 3)
    requested.setdefault("discount", 0.99)
    requested.setdefault("ego_control_profile", "direct")
    requested.setdefault("episode_limit_profile", "source")
    # The environment appends the scenario name when constructing overlay
    # paths, so the factory root must stop at the method level.  Keep this
    # generated cache short because SUMO's legacy Windows path handling can
    # reject an otherwise valid 260-character manifest path.
    overlay_root = PROJECT_ROOT / "tmp" / "visual_eval_20ep_v1" / method
    factory = _make_environment_factory(
        adapter=adapter,
        density=density_setting(protocol, scene),
        overlay_root=overlay_root,
    )
    return factory(SimpleNamespace(**requested), evaluation=True)


def _make_live_env(
    *, protocol: dict[str, Any], source: dict[str, Any], scene: str, method: str, output: Path
) -> Any:
    """Construct the same evaluation env, but keep SUMO-GUI visible."""

    from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
        IndependentV2FiveBySixEnvV1,
        IndependentV2FiveBySixEnvV4V1,
    )

    adapter = {"mst_slt_default": "base", "v4_8_lr_half": "v4_8", "v4_13_tau_half": "v4_13"}[method]
    environment_class = (
        IndependentV2FiveBySixEnvV1 if adapter == "base" else IndependentV2FiveBySixEnvV4V1
    )
    requested = dict(source["requested_arguments"])
    density = density_setting(protocol, scene)
    # Match the existing factory's effective behavior: it passes the vehicle
    # scale to the high-density mixin, whose omitted pedestrian scale follows
    # that value.  This keeps live and headless evaluation comparable.
    overlay_root = PROJECT_ROOT / "tmp" / "visual_eval_20ep_v1_live" / method
    return environment_class(
        scenario=scene,
        history_steps=int(requested.get("history_steps", 10)),
        neighbors=int(requested.get("neighbors", 5)),
        path_length=int(requested.get("path_length", 10)),
        action_repeat=int(requested.get("action_repeat", 3)),
        reward_discount=float(requested.get("discount", 0.99)),
        ego_control_profile=str(requested.get("ego_control_profile", "direct")),
        include_state_lstm=False,
        state_lstm_only=False,
        episode_limit_profile=str(requested.get("episode_limit_profile", "source")),
        render_mode="human",
        high_density_vehicle_scale=float(density["vehicle_scale"]),
        high_density_clone_jitter_seconds=tuple(
            float(value) for value in density["clone_depart_jitter_seconds"]
        ),
        high_density_overlay_root=overlay_root,
        high_density_partition="evaluation",
        high_density_contract_partition=(
            "evaluation" if adapter == "base" else "validation"
        ),
        sumo_args=LIVE_SUMO_ARGS,
    )


def _install_visible_sumo_spawn() -> None:
    """Override only SUMO-GUI child creation with an explicit show state.

    ``traci.start`` delegates to ``subprocess.Popen``.  When this evaluator is
    launched from ``pythonw.exe``, Windows can propagate a hidden startup
    state to that child even though ``render_mode='human'`` selected
    ``sumo-gui``.  Supplying ``SW_SHOWNORMAL`` here keeps the live window on
    the interactive desktop without changing the repository's existing
    environment implementation.  The evaluator is a standalone process, so
    this process-local hook cannot affect training or headless evaluation.
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
            # ``subprocess`` does not expose SW_SHOWNORMAL on every supported
            # Python build; its documented Win32 value is 1.
            startupinfo.wShowWindow = 1
            popen_kwargs["startupinfo"] = startupinfo
            popen_kwargs["creationflags"] = int(
                popen_kwargs.get("creationflags", 0)
            ) | int(subprocess.CREATE_NEW_PROCESS_GROUP)
        return original_popen(*popen_args, **popen_kwargs)

    subprocess.Popen = visible_popen


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
    by_key = {(row["method"], row["scenario"]): row for row in rows}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    x = np.arange(3)
    width = 0.23
    metrics = (("success_rate", "Success"), ("collision_rate", "Collision"), ("timeout_rate", "Timeout"))
    for index, (scene, axis) in enumerate(zip(SCENARIOS, axes.flat)):
        for method_index, method in enumerate(METHOD_ORDER):
            values = [
                100.0 * float(by_key[(method, scene)]["summary"][metric])
                for metric, _ in metrics
            ]
            axis.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(f"({chr(97 + index)}) {SCENE_TITLES[scene]}", loc="left")
        axis.set_xticks(x, [name for _, name in metrics])
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        if index % 3 == 0:
            axis.set_ylabel("Episodes (%)")
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper left")
    figure.suptitle("Existing-model evaluation | six scenes | 20 deterministic episodes", fontsize=15)
    figure.text(
        0.5,
        0.01,
        "Evaluation seeds 10000–10019 | bars are percentages within each scene | single training seed",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.04, 1, 0.94])
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"six_scene_summary.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _live_episode(model: Any, env: Any, *, episode: int, seed: int, speed: float) -> dict[str, Any]:
    from algos.sb3_torch.evaluation import source_evaluation_augmentation

    if speed <= 0:
        raise ValueError("--speed must be positive")
    with source_evaluation_augmentation(model):
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        environment_steps = 0
        policy_decisions = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            policy_decisions += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(info.get("undiscounted_reward", reward))
            environment_steps += 1
            # One environment step advances action_repeat SUMO ticks, each of
            # which is 0.1 s.  The factor gives an approximately real-time
            # view at --speed 1 and a faster inspection at larger values.
            time.sleep(0.1 * int(getattr(env, "action_repeat", 1)) / speed)
            if terminated or truncated:
                raw_steps = int(info.get("raw_simulation_steps", environment_steps))
                success = bool(info.get("is_success", False))
                return {
                    "episode": episode,
                    "seed": seed,
                    "episode_return": episode_return,
                    "decision_steps": policy_decisions,
                    "environment_steps": environment_steps,
                    "raw_steps": raw_steps,
                    "completion_time_seconds": raw_steps * 0.1 if success else None,
                    "success": success,
                    "collision": bool(info.get("collision", False)),
                    "off_route": bool(info.get("off_route", False)),
                    "timeout": bool(info.get("max_time", False)),
                    "traffic_variant": info.get("traffic_variant"),
                }


def run_live(args: argparse.Namespace) -> Path:
    """Run the same 18 cells with visible SUMO-GUI windows."""

    import numpy as np

    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    phase1_plan = args.phase1_plan.resolve()
    phase2_root = args.phase2_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _new_output_dir(DEFAULT_OUTPUT_ROOT / "live")
    )
    if args.output_dir is not None:
        output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        "schema_version": "visual-three-method-six-scene-live-evaluation/v1",
        "status": "running",
        "training_performed": False,
        "fabricated_values": False,
        "live_sumogui": True,
        "episodes_per_cell": EPISODES,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "deterministic": True,
        "speed": args.speed,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "phase1_plan_path": str(phase1_plan),
        "phase1_plan_sha256": sha256(phase1_plan),
        "phase2_root": str(phase2_root),
        "output_directory": str(output),
        "methods": list(METHOD_ORDER),
        "scenarios": list(SCENARIOS),
        "completed_cells": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    _install_visible_sumo_spawn()
    rows: list[dict[str, Any]] = []
    try:
        for method in METHOD_ORDER:
            for scene in SCENARIOS:
                print(
                    f"[live-start] {METHOD_LABELS[method]} / {scene}; "
                    "SUMO-GUI should be visible",
                    flush=True,
                )
                source = resolve_sources(
                    phase1_plan=phase1_plan,
                    phase2_root=phase2_root,
                    scene=scene,
                    method=method,
                )
                env = None
                try:
                    env = _make_live_env(
                        protocol=protocol,
                        source=source,
                        scene=scene,
                        method=method,
                        output=output,
                    )
                    model = _load_model(method, source, env, args.device)
                    records = []
                    for episode in range(EPISODES):
                        record = _live_episode(
                            model,
                            env,
                            episode=episode,
                            seed=EVALUATION_SEED_START + episode,
                            speed=args.speed,
                        )
                        records.append(record)
                        print(
                            f"[live-episode] {METHOD_LABELS[method]} / {scene} "
                            f"{episode + 1}/{EPISODES}: "
                            f"success={record['success']}",
                            flush=True,
                        )
                finally:
                    if env is not None:
                        env.close()
                returns = np.asarray([float(row["episode_return"]) for row in records])
                decisions = np.asarray([int(row["decision_steps"]) for row in records])
                raw_steps = np.asarray([int(row["raw_steps"]) for row in records])
                successes = [bool(row["success"]) for row in records]
                completions = [
                    float(row["completion_time_seconds"])
                    for row in records
                    if row["completion_time_seconds"] is not None
                ]
                summary = {
                    "episodes": EPISODES,
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "mean_decision_steps": float(np.mean(decisions)),
                    "mean_raw_steps": float(np.mean(raw_steps)),
                    "success_rate": float(np.mean(successes)),
                    "collision_rate": float(np.mean([bool(row["collision"]) for row in records])),
                    "off_route_rate": float(np.mean([bool(row["off_route"]) for row in records])),
                    "timeout_rate": float(np.mean([bool(row["timeout"]) for row in records])),
                }
                row = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "scenario": scene,
                    "scenario_label": SCENE_TITLES[scene],
                    "source_kind": source["source_kind"],
                    "decoder": source["decoder"],
                    "checkpoint": str(source["checkpoint"]),
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "source_run_dir": str(source["run_dir"]),
                    "source_arguments": str(source["arguments_path"]),
                    "summary": summary,
                    "successful_episodes": int(sum(successes)),
                    "mean_success_completion_time_seconds": (
                        float(np.mean(completions)) if completions else None
                    ),
                    "episode_records": records,
                }
                rows.append(row)
                safe_name = f"{method}__{scene}"
                write_json(output / "cells" / f"{safe_name}.json", row)
                manifest["completed_cells"].append(safe_name)
                write_json(output / "manifest.json", manifest)
        _plot_summary(rows, output / "figures")
        write_json(output / "summary.json", {"rows": rows, "episodes": EPISODES})
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
                "figure_directory": str((output / "figures").resolve()),
            }
        )
        write_json(output / "manifest.json", manifest)
        print(f"[live-completed] {output}", flush=True)
        return output
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error": repr(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
            }
        )
        write_json(output / "manifest.json", manifest)
        raise

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for index, (scene, axis) in enumerate(zip(SCENARIOS, axes.flat)):
        for method in METHOD_ORDER:
            records = by_key[(method, scene)]["episode_records"]
            axis.plot(
                [int(record["episode"]) + 1 for record in records],
                [float(record["episode_return"]) for record in records],
                marker="o",
                ms=3,
                lw=1.4,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(f"({chr(97 + index)}) {SCENE_TITLES[scene]}", loc="left")
        axis.set_xlabel("Evaluation episode")
        axis.grid(alpha=0.25)
        if index % 3 == 0:
            axis.set_ylabel("Episode return")
    axes.flat[0].legend(frameon=False, fontsize=9, loc="best")
    figure.suptitle("Existing-model evaluation returns | six scenes", fontsize=15)
    figure.text(0.5, 0.01, "No smoothing or cross-seed aggregation", ha="center", fontsize=9)
    figure.tight_layout(rect=[0, 0.04, 1, 0.94])
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"episode_returns.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    import torch

    torch.set_num_threads(1)
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    phase1_plan = args.phase1_plan.resolve()
    phase2_root = args.phase2_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _new_output_dir(DEFAULT_OUTPUT_ROOT)
    )
    if args.output_dir is not None:
        output.mkdir(parents=True, exist_ok=False)

    started = time.time()
    manifest = {
        "schema_version": "visual-three-method-six-scene-evaluation/v1",
        "status": "running",
        "training_performed": False,
        "fabricated_values": False,
        "episodes_per_cell": EPISODES,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "deterministic": True,
        "policy_action_hold": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "phase1_plan_path": str(phase1_plan),
        "phase1_plan_sha256": sha256(phase1_plan),
        "phase2_root": str(phase2_root),
        "output_directory": str(output),
        "methods": list(METHOD_ORDER),
        "scenarios": list(SCENARIOS),
        "completed_cells": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    try:
        for method in METHOD_ORDER:
            for scene in SCENARIOS:
                print(f"[start] {METHOD_LABELS[method]} / {scene}", flush=True)
                source = resolve_sources(
                    phase1_plan=phase1_plan,
                    phase2_root=phase2_root,
                    scene=scene,
                    method=method,
                )
                env = None
                try:
                    env = _make_env(
                        protocol=protocol,
                        source=source,
                        scene=scene,
                        method=method,
                        output=output,
                    )
                    model = _load_model(method, source, env, args.device)
                    report = evaluate_model_detailed(
                        model,
                        env,
                        episodes=EPISODES,
                        deterministic=True,
                        seed=EVALUATION_SEED_START,
                        sumo_step_seconds=0.1,
                        policy_action_hold=1,
                    ).to_dict()
                finally:
                    if env is not None:
                        env.close()
                row = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "scenario": scene,
                    "scenario_label": SCENE_TITLES[scene],
                    "source_kind": source["source_kind"],
                    "decoder": source["decoder"],
                    "checkpoint": str(source["checkpoint"]),
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "source_run_dir": str(source["run_dir"]),
                    "source_arguments": str(source["arguments_path"]),
                    **report,
                }
                rows.append(row)
                safe_name = f"{method}__{scene}"
                write_json(output / "cells" / f"{safe_name}.json", row)
                manifest["completed_cells"].append(safe_name)
                write_json(output / "manifest.json", manifest)
                summary = report["summary"]
                print(
                    f"[done] {METHOD_LABELS[method]} / {scene}: "
                    f"success={summary['success_rate']:.3f}, "
                    f"collision={summary['collision_rate']:.3f}, "
                    f"timeout={summary['timeout_rate']:.3f}",
                    flush=True,
                )
        _plot_summary(rows, output / "figures")
        write_json(output / "summary.json", {"rows": rows, "episodes": EPISODES})
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
                "figure_directory": str((output / "figures").resolve()),
            }
        )
        write_json(output / "manifest.json", manifest)
        (output / "README.md").write_text(
            "# Three existing models: six-scene visual evaluation\n\n"
            "This directory contains a new 20-episode deterministic evaluation. "
            "The source checkpoints and existing result directories were not modified.\n\n"
            "- New evaluation cells: `cells/*.json`\n"
            "- New aggregate result: `summary.json`\n"
            "- New figures: `figures/six_scene_summary.*` and `figures/episode_returns.*`\n"
            "- Provenance and source hashes: `manifest.json`\n\n"
            "Seeds are 10000--10019. This is a 20-episode screen, not a replacement "
            "for the existing 100-episode or multi-seed evidence.\n",
            encoding="utf-8",
        )
        print(f"[completed] {output}", flush=True)
        return output
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error": repr(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.time() - started,
            }
        )
        write_json(output / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--phase1-plan", type=Path, default=DEFAULT_PHASE1_PLAN)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--live",
        action="store_true",
        help="show real-time SUMO-GUI driving while evaluating the same 20 episodes",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="live display speed; 1 is approximately real time, larger is faster",
    )
    args = parser.parse_args(argv)
    if args.live:
        run_live(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EPISODES", "METHOD_ORDER", "SCENARIOS", "main", "run"]
