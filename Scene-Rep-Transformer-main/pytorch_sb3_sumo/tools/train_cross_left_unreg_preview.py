"""Preview training of hold35k + mst_slt on ``cross_left_unreg`` with 200-step checkpoints.

Self-contained: it builds the same environment contracts as the three-scene /
seven-baseline pipeline (``density("cross_left_unreg")`` -> ``vehicle_scale=1.0``,
an empty additive overlay), but writes every artifact under
``results_cross_left_unreg_preview/`` and does not modify any existing file.

Two methods, one uncontrolled 4-way scenario (0.2 veh/s per approach):

  * hold35k (v4_8): ``make_model`` from ``tools.v48_stability_v4`` (StabilitySAC).
  * mst_slt (base): ``tools.train_sb3`` with ``--algo scene_rep``.

Checkpoints are saved every ``--checkpoint-freq`` raw steps (default 200) and,
after both methods finish training, every checkpoint is evaluated headlessly
(``--eval-episodes`` deterministic episodes, default 10) to draw a success /
collision rate vs raw-steps training curve for the two methods.

This is a preview: evaluation uses the natural traffic-partition cycling and a
small episode count, not the frozen 50-episode convention of
``three_scene_eval_curve.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from tools.three_scene_hold35k_vs_mst_v1.common import (  # noqa: E402
    HOLD35K_CANDIDATE,
    density,
)

RESULT_ROOT = PROJECT_ROOT / "results_cross_left_unreg_preview"
SCENE = "cross_left_unreg"
METHODS = ("hold35k", "mst_slt")
SEED_START = {"hold35k": 420000, "mst_slt": 10000}
METHOD_LABELS = {"hold35k": "TASAC (hold35k)", "mst_slt": "MST+SLT"}
METHOD_COLORS = {"hold35k": "#D62728", "mst_slt": "#1f77b4"}
LEARNING_STARTS = 5000  # shared raw-step warmup used by both method builders


def _namespace() -> argparse.Namespace:
    return argparse.Namespace(
        scenario=SCENE,
        history_steps=10,
        neighbors=5,
        path_length=10,
        action_repeat=3,
        discount=0.99,
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split="validation",
    )


def _factory(adapter: str):
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory

    return _make_environment_factory(
        adapter=adapter,
        density=density(SCENE),
        overlay_root=RESULT_ROOT / "overlays" / adapter / "seed_0",
    )


def _train_dir(method: str) -> Path:
    return RESULT_ROOT / "train" / f"{method}__{SCENE}"


def _checkpoint(method: str, raw_step: int) -> Path:
    prefix = "ckpt" if method == "hold35k" else "scene_rep"
    return _train_dir(method) / "checkpoints" / f"{prefix}_raw_{raw_step}_steps.zip"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_hold35k(max_steps: int, checkpoint_freq: int, device: str) -> Path:
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor

    from algos.sb3_torch import RawStepControlCallback
    from tools.v48_stability_v4.model import make_model

    output = _train_dir("hold35k")
    if (output / "final_model.zip").is_file():
        return output
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        _factory("v4_8")(_namespace(), evaluation=False),
        filename=str(output / "train_monitor.csv"),
        info_keywords=(
            "raw_simulation_steps", "is_success", "collision", "off_route", "max_time",
        ),
    )
    try:
        model = make_model(env, SCENE, HOLD35K_CANDIDATE, device=device)
        model.learn(
            total_timesteps=max_steps,
            callback=CallbackList(
                [
                    RawStepControlCallback(
                        raw_step_budget=max_steps,
                        checkpoint_frequency=checkpoint_freq,
                        checkpoint_path=output / "checkpoints",
                        checkpoint_prefix="ckpt",
                    )
                ]
            ),
        )
        model.save(output / "final_model.zip")
    finally:
        env.close()
    return output


def train_mst_slt(max_steps: int, checkpoint_freq: int, device: str) -> Path:
    from tools import train_sb3

    output = _train_dir("mst_slt")
    if (output / "final_model.zip").is_file():
        return output
    # train_sb3.main creates the run directory itself (exist_ok=False), so it
    # must not already exist here.
    argv = [
        "--algo", "scene_rep",
        "--scenario", SCENE,
        "--max-steps", str(max_steps),
        "--learning-starts", str(LEARNING_STARTS),
        "--checkpoint-freq", str(checkpoint_freq),
        "--eval-freq", "0",
        "--eval-episodes", "10",
        "--seed", "0",
        "--device", device,
        "--batch-size", "32",
        "--learning-rate", "0.0001",
        "--discount", "0.99",
        "--buffer-size", "20000",
        "--action-repeat", "3",
        "--output-dir", str(output.parent.resolve()),
        "--model-name", output.name,
        "--ego-control-profile", "direct",
        "--episode-limit-profile", "source",
        "--traffic-protocol", "frozen_80_20",
    ]
    train_sb3.main(
        argv,
        env_factory=_factory("base"),
        default_output_dir=output.parent,
        require_paper_evaluation_contract=True,
        tensorboard_log_root=RESULT_ROOT / "tb",
    )
    return output


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(method: str, raw_step: int, device: str, episodes: int) -> dict[str, Any]:
    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tools.v48_stability_v4.model import StabilitySAC, use_actor_only

    path = _checkpoint(method, raw_step)
    if method == "hold35k":
        env = _factory("v4_8")(_namespace(), evaluation=True)
        model = use_actor_only(
            StabilitySAC.load(str(path), env=env, device=device, buffer_size=32)
        )
    else:
        env = _factory("base")(_namespace(), evaluation=True)
        model = SceneRepresentationSAC.load(str(path), env=env, device=device, buffer_size=32)
    try:
        report = evaluate_model_detailed(
            model, env, episodes=episodes, seed=SEED_START[method], policy_action_hold=1
        )
        summary = report.summary
        return {
            "method": method,
            "raw_step": raw_step,
            "success_rate": float(summary.success_rate),
            "collision_rate": float(summary.collision_rate),
            "off_route_rate": float(summary.off_route_rate),
            "timeout_rate": float(summary.timeout_rate),
            "mean_return": float(summary.mean_return),
            "mean_raw_steps": float(summary.mean_raw_steps),
        }
    finally:
        env.close()


def evaluate_method(method: str, raw_steps: list[int], device: str, episodes: int) -> list[dict[str, Any]]:
    rows = []
    for raw_step in raw_steps:
        rows.append(evaluate_checkpoint(method, raw_step, device, episodes))
        row = rows[-1]
        print(
            f"[eval] {method:8s} raw={raw_step:6d} "
            f"success={row['success_rate']:.2f} collision={row['collision_rate']:.2f} "
            f"timeout={row['timeout_rate']:.2f}",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(rows: list[dict[str, Any]], output: Path, episodes: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    by_method = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for metric, axis, title in (
        ("success_rate", axes[0], "Success rate"),
        ("collision_rate", axes[1], "Collision rate"),
    ):
        for method in METHODS:
            data = by_method[method]
            xs = [r["raw_step"] for r in data]
            ys = [r[metric] for r in data]
            axis.plot(
                xs, ys, marker="o", ms=3, lw=1.5,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            )
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlabel("raw steps", fontsize=9)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("rate", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"cross_left_unreg preview | hold35k vs mst_slt | "
        f"{episodes} deterministic eval episodes per checkpoint",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(rows: list[dict[str, Any]], png_path: Path, output: Path) -> Path:
    import base64

    by_method = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    milestones = [r["raw_step"] for r in rows if r["method"] == "hold35k"]
    # Show a condensed table at ~every 2000 steps plus the final point.
    shown = sorted(set([0] + [s for s in milestones if s % 2000 == 0] + [milestones[-1]]))
    shown = [s for s in shown if s in milestones]

    def cell(method: str, step: int) -> str:
        row = next((r for r in by_method[method] if r["raw_step"] == step), None)
        if row is None:
            return "—"
        return f"{row['success_rate']:.2f} / {row['collision_rate']:.2f}"

    header = "| raw steps | hold35k succ/coll | mst_slt succ/coll |"
    sep = "|---|---|---|"
    table_lines = [header, sep]
    for step in shown:
        table_lines.append(f"| {step} | {cell('hold35k', step)} | {cell('mst_slt', step)} |")
    table = "\n".join(table_lines)

    with png_path.open("rb") as handle:
        png_b64 = base64.b64encode(handle.read()).decode("ascii")

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>cross_left_unreg 预览训练曲线</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; }}
h1 {{ font-size: 20px; }}
img {{ max-width: 1000px; width: 100%%; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin-top: 16px; }}
td, th {{ border: 1px solid #ccc; padding: 4px 12px; text-align: left; }}
code {{ background: #f5f5f5; padding: 1px 5px; }}
</style></head><body>
<h1>cross_left_unreg 预览训练曲线（hold35k vs mst_slt）</h1>
<p>场景：完全无管制四臂交叉口、70 m 车道、0.2 veh/s/进口道（base density，无叠加）。
两个方法各自 <code>vehicle_scale=1.0</code>（空叠加）。</p>
<img src="data:image/png;base64,{png_b64}" alt="training curve">
<h2>关键步成功率 / 碰撞率</h2>
{table}
</body></html>"""
    report_path = output / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--checkpoint-freq", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true",
                        help="skip training and only re-evaluate + re-plot existing checkpoints")
    args = parser.parse_args(argv)

    if args.max_steps <= 0 or args.checkpoint_freq <= 0:
        parser.error("--max-steps and --checkpoint-freq must be positive")
    if args.checkpoint_freq > args.max_steps:
        parser.error("--checkpoint-freq must not exceed --max-steps")

    torch.set_num_threads(1)
    output = RESULT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    raw_steps = list(range(args.checkpoint_freq, args.max_steps + 1, args.checkpoint_freq))

    if not args.skip_train:
        print(f"[train] hold35k on {SCENE} for {args.max_steps} raw steps "
              f"(checkpoint every {args.checkpoint_freq})", flush=True)
        train_hold35k(args.max_steps, args.checkpoint_freq, args.device)
        print(f"[train] mst_slt on {SCENE} for {args.max_steps} raw steps "
              f"(checkpoint every {args.checkpoint_freq})", flush=True)
        train_mst_slt(args.max_steps, args.checkpoint_freq, args.device)
    else:
        for method in METHODS:
            if not _train_dir(method).is_dir():
                parser.error(f"--skip-train but {_train_dir(method)} is missing")

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.extend(evaluate_method(method, raw_steps, args.device, args.eval_episodes))

    json_path = output / "curve_data.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    png = plot_curves(rows, output, args.eval_episodes)
    report = write_report(rows, png, output)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "scenario": SCENE,
                "methods": list(METHODS),
                "max_steps": args.max_steps,
                "checkpoint_freq": args.checkpoint_freq,
                "eval_episodes": args.eval_episodes,
                "device": args.device,
                "vehicle_scale": density(SCENE)["vehicle_scale"],
                "wall_seconds": time.time() - started,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[done] {png}", flush=True)
    print(f"[done] {report}", flush=True)
    print(f"[done] {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
