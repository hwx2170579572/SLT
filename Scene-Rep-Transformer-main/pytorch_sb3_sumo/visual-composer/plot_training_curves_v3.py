"""Plot saved-model success curves against time steps with EMA(0.999).

The source logs store success once per episode.  For a transparent time-step
view, each episode outcome is held over its logged ``raw_simulation_steps``;
the resulting descriptive binary time series is aggregated with a 2,000-step
rolling window and smoothed per time step by EMA alpha=0.999.  No model is
loaded for training and no new evaluation episodes are generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = PROJECT_ROOT / "visual-composer" / "plot_training_curves_v1.py"
BASE = runpy.run_path(str(V1_SCRIPT))
read_json = BASE["read_json"]
sha256 = BASE["sha256"]
resolve_source = BASE["resolve_source"]
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "independent_v2_existing_3methods_100seeds_100episodes_v1" / "protocol.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v3"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    cjk_font = Path("C:/Windows/Fonts/msyh.ttc")
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(cjk_font)).get_name()
except Exception:  # pragma: no cover - fallback for non-Windows hosts
    pass
plt.rcParams["axes.unicode_minus"] = False


SCENARIO_LABELS = {"carla": "CARLA", "cross": "Cross", "roundabout": "Roundabout"}
SCENARIO_PANEL_LABELS = {"carla": "a)", "cross": "b)", "roundabout": "c)"}
METHOD_SHORT_LABELS = {"mst_slt": "MST+SLT", "temporal_graph": "TemporalGraph", "v4_8": "v4.8"}
METHOD_STYLES = {
    "mst_slt": {"color": "#E69F00", "linestyle": "-"},
    "temporal_graph": {"color": "#009E73", "linestyle": "--"},
    "v4_8": {"color": "#0072B2", "linestyle": "-."},
}


def build_time_step_curve(
    rows: list[dict[str, Any]],
    *,
    window_steps: int,
    ema_alpha: float,
    sample_stride: int,
) -> list[dict[str, Any]]:
    """Expand episode outcomes over raw steps, then compute rolling+EMA curves."""

    if window_steps <= 0 or sample_stride <= 0:
        raise ValueError("window_steps and sample_stride must be positive")
    if not 0.0 < ema_alpha < 1.0:
        raise ValueError("ema_alpha must be between 0 and 1")
    success_values = np.asarray([int(row["is_success"]) for row in rows], dtype=np.float64)
    raw_steps = np.asarray([int(row["raw_simulation_steps"]) for row in rows], dtype=np.int64)
    if np.any(raw_steps <= 0):
        raise ValueError("raw_simulation_steps must be positive")

    # Episode-level success is the only available success label. Holding it
    # over its logged duration makes the time-step weighting explicit rather
    # than pretending that internal per-step labels were recorded.
    step_series = np.repeat(success_values, raw_steps)
    total_steps = int(step_series.size)
    indices = np.arange(total_steps, dtype=np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(step_series, dtype=np.float64)))
    starts = np.maximum(0, indices + 1 - int(window_steps))
    counts = indices + 1 - starts
    rolling_mean = (cumulative[indices + 1] - cumulative[starts]) / counts
    rolling_std = np.sqrt(np.clip(rolling_mean * (1.0 - rolling_mean), 0.0, 1.0))

    # Count overlapping episodes for a descriptive SE denominator.  This is
    # deliberately not presented as a formal confidence interval.
    episode_ends = np.cumsum(raw_steps, dtype=np.int64)
    episode_at_index = np.searchsorted(episode_ends, indices, side="right")
    episode_at_start = np.searchsorted(episode_ends, starts, side="right")
    overlapping_episodes = np.maximum(1, episode_at_index - episode_at_start + 1)
    rolling_se = rolling_std / np.sqrt(overlapping_episodes.astype(np.float64))

    ema_mean = np.empty(total_steps, dtype=np.float64)
    ema_se = np.empty(total_steps, dtype=np.float64)
    ema_mean[0] = rolling_mean[0]
    ema_se[0] = rolling_se[0]
    one_minus_alpha = 1.0 - float(ema_alpha)
    for index in range(1, total_steps):
        ema_mean[index] = float(ema_alpha) * ema_mean[index - 1] + one_minus_alpha * rolling_mean[index]
        ema_se[index] = float(ema_alpha) * ema_se[index - 1] + one_minus_alpha * rolling_se[index]

    sample_indices = np.unique(
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.arange(sample_stride - 1, total_steps, sample_stride, dtype=np.int64),
                np.asarray([total_steps - 1], dtype=np.int64),
            )
        )
    )
    curve: list[dict[str, Any]] = []
    for index in sample_indices.tolist():
        center = float(ema_mean[index])
        band = float(ema_se[index])
        curve.append(
            {
                "time_step": int(index),
                "rolling_success_rate": float(rolling_mean[index]),
                "rolling_std": float(rolling_std[index]),
                "rolling_se": float(rolling_se[index]),
                "ema_success_rate": center,
                "ema_band_lower": max(0.0, center - band),
                "ema_band_upper": min(1.0, center + band),
                "window_steps": int(window_steps),
                "ema_alpha": float(ema_alpha),
                "sample_stride_steps": int(sample_stride),
            }
        )
    return curve


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty data table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_axis(
    axis: Any,
    curves_by_method: dict[str, list[dict[str, Any]]],
    *,
    scenario: str,
    panel_label: str,
    x_max: int,
    window_steps: int,
    ema_alpha: float,
) -> None:
    for method in ("mst_slt", "temporal_graph", "v4_8"):
        curve = curves_by_method.get(method, [])
        if not curve:
            continue
        style = METHOD_STYLES[method]
        x = [float(row["time_step"]) for row in curve]
        y = [float(row["ema_success_rate"]) for row in curve]
        lower = [float(row["ema_band_lower"]) for row in curve]
        upper = [float(row["ema_band_upper"]) for row in curve]
        axis.fill_between(x, lower, upper, color=style["color"], alpha=0.22, linewidth=0, zorder=1)
        axis.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.9,
            label=METHOD_SHORT_LABELS[method],
            zorder=2,
        )
    axis.set_title(f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=10.5, pad=7)
    axis.text(
        -0.12,
        1.10,
        panel_label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4},
    )
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, 1.0)
    tick_step = 10_000
    axis.set_xticks(list(range(0, x_max + 1, tick_step)))
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_xlabel("Time steps", fontsize=9.5)
    axis.set_ylabel("Avg Success rate", fontsize=9.5)
    axis.tick_params(labelsize=8.5)
    axis.grid(True, color="#BFBFBF", linewidth=0.65, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper left", fontsize=8.3, frameon=True, framealpha=0.82, borderpad=0.45, handlelength=2.2)
    axis.set_facecolor("#FCFCFC")


def plot_composite(
    curves: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    scenarios: list[str],
    output_dir: Path,
    x_max: int,
    window_steps: int,
    ema_alpha: float,
) -> dict[str, str]:
    fig, axes = plt.subplots(1, len(scenarios), figsize=(15.0, 4.55), sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.18, wspace=0.22)
    for axis, scenario in zip(axes, scenarios):
        draw_axis(
            axis,
            curves[scenario],
            scenario=scenario,
            panel_label={"carla": "a)", "cross": "b)", "roundabout": "c)"}.get(scenario, ""),
            x_max=x_max,
            window_steps=window_steps,
            ema_alpha=ema_alpha,
        )
    fig.text(
        0.5,
        0.045,
        f"EMA α={ema_alpha:g} per time step · rolling window={window_steps:,} time steps · band: descriptive rolling SE (SD/√n), seed 0 · saved parent-v2 logs; no retraining",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.patch.set_facecolor("white")
    png_path = output_dir / "success_curves_time_steps_ema999.png"
    svg_path = output_dir / "success_curves_time_steps_ema999.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path.resolve()), "svg": str(svg_path.resolve())}


def plot_individual(
    curves: dict[str, list[dict[str, Any]]],
    *,
    scenario: str,
    output_dir: Path,
    x_max: int,
    window_steps: int,
    ema_alpha: float,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for extension in ("png", "svg"):
        path = output_dir / f"{scenario}_success_curve_time_steps_ema999.{extension}"
        fig, axis = plt.subplots(figsize=(5.1, 4.55))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.18)
        draw_axis(
            axis,
            curves,
            scenario=scenario,
            panel_label={"carla": "a)", "cross": "b)", "roundabout": "c)"}.get(scenario, ""),
            x_max=x_max,
            window_steps=window_steps,
            ema_alpha=ema_alpha,
        )
        fig.text(
            0.5,
            0.035,
            f"EMA α={ema_alpha:g} · {window_steps:,}-step rolling window · descriptive rolling SE (SD/√n), seed 0",
            ha="center",
            fontsize=7.4,
            color="#555555",
        )
        fig.patch.set_facecolor("white")
        fig.savefig(path, dpi=300 if extension == "png" else None, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        result[extension] = str(path.resolve())
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-steps", type=int, default=2000)
    parser.add_argument("--ema-alpha", type=float, default=0.999)
    parser.add_argument("--sample-stride", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(PROTOCOL_PATH)
    if args.window_steps <= 0 or args.sample_stride <= 0:
        raise ValueError("window and sample stride must be positive")
    if not 0.0 < args.ema_alpha < 1.0:
        raise ValueError("EMA alpha must be between 0 and 1")
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(protocol["matrix"]["method_order"])
    scenarios = list(protocol["matrix"]["scenario_order"])
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {scenario: {} for scenario in scenarios}
    data_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    max_steps = 0
    for method in methods:
        for scenario in scenarios:
            source_rows, source_audit = resolve_source(protocol, method, scenario, 20)
            curve = build_time_step_curve(
                source_rows,
                window_steps=args.window_steps,
                ema_alpha=args.ema_alpha,
                sample_stride=args.sample_stride,
            )
            curves[scenario][method] = curve
            for row in curve:
                data_rows.append(
                    {
                        "method": method,
                        "method_label": source_rows[0]["method_label"],
                        "scenario": scenario,
                        "scenario_label": source_rows[0]["scenario_label"],
                        "run_id": source_rows[0]["run_id"],
                        "training_seed": 0,
                        **row,
                    }
                )
            max_steps = max(max_steps, int(curve[-1]["time_step"]))
            source_audit["time_step_transform"] = {
                "episode_outcome_held_over_logged_raw_steps": True,
                "window_steps": args.window_steps,
                "ema_alpha": args.ema_alpha,
                "sample_stride_steps": args.sample_stride,
                "curve_sample_count": len(curve),
            }
            sources.append(source_audit)
    x_max = max(50_000, int(math.ceil(max_steps / 10_000.0) * 10_000))
    data_path = output_dir / "training_success_curve_time_steps_ema999_data.csv"
    write_csv(data_path, data_rows)
    composite = plot_composite(
        curves,
        scenarios=scenarios,
        output_dir=output_dir,
        x_max=x_max,
        window_steps=args.window_steps,
        ema_alpha=args.ema_alpha,
    )
    individual = {
        scenario: plot_individual(
            curves[scenario],
            scenario=scenario,
            output_dir=output_dir,
            x_max=x_max,
            window_steps=args.window_steps,
            ema_alpha=args.ema_alpha,
        )
        for scenario in scenarios
    }
    manifest = {
        "artifact_id": "iv2e3m100x100_training_curves_v3_time_steps_ema999",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_v2": protocol.get("parent_v2"),
        "matrix": {"methods": methods, "scenarios": scenarios, "source_run_count": len(sources), "training_seed": 0},
        "training_is_forbidden_in_current_protocol": bool(protocol.get("training_is_forbidden", False)),
        "curve_definition": {
            "metric": "is_success",
            "x_axis": "cumulative raw_simulation_steps displayed as time steps",
            "episode_to_time_step_proxy": "hold each episode's is_success outcome over its logged raw_simulation_steps",
            "rolling_window_steps": args.window_steps,
            "ema_alpha": args.ema_alpha,
            "ema_update_frequency": "every raw simulation time step",
            "render_sample_stride_steps": args.sample_stride,
            "band": "EMA-smoothed descriptive rolling SE (population SD/√number of overlapping episodes), clipped to [0,1]",
            "cross_seed_uncertainty": False,
            "no_imputation": True,
        },
        "data_table": {"path": str(data_path.resolve()), "sha256": sha256(data_path), "row_count": len(data_rows)},
        "figures": {"composite": composite, "individual": individual},
        "source_audit": sources,
        "qa": {
            "expected_source_count": 9,
            "observed_source_count": len(sources),
            "all_monitor_rows_nonempty": all(item["monitor"]["episode_rows"] > 0 for item in sources),
            "all_models_exist_and_hashed": all(Path(item["model_path"]).is_file() and len(item["model_sha256"]) == 64 for item in sources),
            "all_logs_exist_and_hashed": all(Path(item["monitor"]["monitor_path"]).is_file() and len(item["monitor"]["monitor_sha256"]) == 64 for item in sources),
            "tensorboard_event_coverage": sum(bool(item["tensorboard"]["event_paths"]) for item in sources),
            "data_row_count": len(data_rows),
        },
        "notes": [
            "The x-axis is cumulative raw simulation time steps, not episode index or wall-clock time.",
            "EMA alpha=0.999 is applied per raw time step after a 2,000-step rolling aggregation.",
            "Because monitor logs label success per episode, the time-step series holds that outcome over the episode's logged duration; this is a descriptive proxy.",
            "The band is not a cross-seed confidence interval; all parent training logs use seed 0.",
        ],
    }
    manifest_path = output_dir / "training_success_curve_time_steps_ema999_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Time-step training curves with EMA(0.999)\n\n"
        "The x-axis is cumulative raw simulation time steps. Each episode's success outcome is held over its logged duration, aggregated over a 2,000-step rolling window, and smoothed per time step with EMA alpha=0.999. Curves are rendered every 100 steps. The band is descriptive rolling SE, not a cross-seed confidence interval.\n\n"
        "See `training_success_curve_time_steps_ema999_manifest.json` for source hashes and the exact transform.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "composite": composite,
        "individual": individual,
        "manifest": str(manifest_path.resolve()),
        "data_rows": len(data_rows),
        "source_count": len(sources),
        "x_max_time_steps": x_max,
        "window_steps": args.window_steps,
        "ema_alpha": args.ema_alpha,
        "qa": manifest["qa"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
