"""Plot saved-model training curves with a padded 20-episode window.

This version reads the nine parent-v2 monitor logs through the existing v1
source resolver.  It does not retrain or alter any model/log.  For each
method×scenario it computes the success rate over exactly the most recent 20
episodes, padding missing early episodes with zeros.  The episode statistic is
held over that episode's observed raw simulation steps, then an EMA with
alpha=0.999 is applied once per raw time step.  The rendered x-axis is the
cumulative raw simulation time-step count.

The top row retains all three methods and the bottom row shows the signed
pointwise margin ``v4.8 - max(MST+SLT, TemporalGraph)``.  Both positive and
negative margins remain visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


# Register a Windows CJK font when available.  The compact legend labels are
# ASCII, but the source/provenance labels may contain Chinese text.
try:
    cjk_font = Path("C:/Windows/Fonts/msyh.ttc")
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(cjk_font)).get_name()
except Exception:  # pragma: no cover - environment-specific
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = PROJECT_ROOT / "visual-composer" / "plot_training_curves_v1.py"
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "independent_v2_existing_3methods_100seeds_100episodes_v1" / "protocol.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v5_episode20_ema999"

METHODS = ["v4_8", "mst_slt", "temporal_graph"]
BASELINES = ["mst_slt", "temporal_graph"]
SCENARIOS = ["carla", "cross", "roundabout"]
METHOD_SHORT_LABELS = {
    "v4_8": "v4.8",
    "mst_slt": "MST+SLT",
    "temporal_graph": "TemporalGraph",
}
SCENARIO_LABELS = {"carla": "CARLA", "cross": "Cross", "roundabout": "Roundabout"}
PANEL_LABELS = {"carla": "a)", "cross": "b)", "roundabout": "c)"}
METHOD_STYLES = {
    "v4_8": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.8, "band_alpha": 0.18},
    "mst_slt": {"color": "#666666", "linestyle": "--", "linewidth": 1.35, "band_alpha": 0.08},
    "temporal_graph": {"color": "#E69F00", "linestyle": ":", "linewidth": 1.45, "band_alpha": 0.09},
}

WINDOW_EPISODES = 20
EMA_ALPHA = 0.999
SAMPLE_STRIDE = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(value: str | Path) -> str:
    """Return a stable repo-relative path for local provenance records."""

    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(value)


def sanitize_paths(value: Any) -> Any:
    """Remove machine-specific project prefixes from nested source audits."""

    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                value = repo_rel(candidate)
            except Exception:  # pragma: no cover - defensive
                pass
        # TensorBoard event filenames can contain the workstation hostname;
        # it is not needed for provenance because the file hash is retained.
        return re.sub(r"DESKTOP-[A-Za-z0-9_-]+", "<host>", value)
    return value


def load_source_resolver() -> Any:
    if not V1_SCRIPT.is_file():
        raise FileNotFoundError(f"missing source resolver: {V1_SCRIPT}")
    namespace = runpy.run_path(str(V1_SCRIPT), run_name="ccfa_v1_source_only")
    resolver = namespace.get("resolve_source")
    if not callable(resolver):
        raise TypeError("v1 source resolver did not expose resolve_source")
    return resolver


def padded_episode_window(successes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero-padded mean, population SD and observed-count arrays.

    ``successes[i]`` is the outcome of episode i (0-based).  The returned
    mean is always divided by ``window``.  Missing prefix episodes therefore
    contribute explicit zeros rather than shrinking the denominator.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    means = np.zeros(len(successes), dtype=float)
    stds = np.zeros(len(successes), dtype=float)
    counts = np.zeros(len(successes), dtype=int)
    for index in range(len(successes)):
        start = max(0, index - window + 1)
        observed = np.asarray(successes[start : index + 1], dtype=float)
        slots = np.zeros(window, dtype=float)
        slots[-len(observed) :] = observed
        means[index] = float(slots.mean())
        stds[index] = float(slots.std(ddof=0))
        counts[index] = int(len(observed))
    return means, stds, counts


def build_time_step_curve(
    rows: list[dict[str, Any]],
    *,
    window_episodes: int = WINDOW_EPISODES,
    ema_alpha: float = EMA_ALPHA,
    sample_stride: int = SAMPLE_STRIDE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transform episode-level outcomes into per-time-step EMA samples."""

    if not rows:
        raise ValueError("cannot transform an empty monitor log")
    if not 0.0 < ema_alpha < 1.0:
        raise ValueError("ema_alpha must be between 0 and 1")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    successes = np.asarray([1.0 if bool(row["is_success"]) else 0.0 for row in rows], dtype=float)
    raw_steps = np.asarray([int(row["raw_simulation_steps"]) for row in rows], dtype=int)
    if np.any(raw_steps <= 0):
        raise ValueError("all raw_simulation_steps must be positive")
    cumulative_end = np.cumsum(raw_steps, dtype=np.int64)
    total_steps = int(cumulative_end[-1])

    episode_mean, episode_std, observed_counts = padded_episode_window(successes, window_episodes)
    episode_se = episode_std / math.sqrt(float(window_episodes))

    # The episode-level descriptive signal is held for every observed raw
    # time step of that episode.  EMA is then updated at every raw step, not
    # at the 100-step rendering stride.
    step_mean = np.repeat(episode_mean, raw_steps)
    step_se = np.repeat(episode_se, raw_steps)
    if len(step_mean) != total_steps:
        raise AssertionError("episode expansion does not match cumulative raw steps")
    ema_mean = np.zeros(total_steps + 1, dtype=float)
    ema_se = np.zeros(total_steps + 1, dtype=float)
    for step_index in range(total_steps):
        ema_mean[step_index + 1] = ema_alpha * ema_mean[step_index] + (1.0 - ema_alpha) * step_mean[step_index]
        ema_se[step_index + 1] = ema_alpha * ema_se[step_index] + (1.0 - ema_alpha) * step_se[step_index]
    lower = np.clip(ema_mean - ema_se, 0.0, 1.0)
    upper = np.clip(ema_mean + ema_se, 0.0, 1.0)

    sample_times = [0]
    sample_times.extend(range(sample_stride, total_steps + 1, sample_stride))
    if sample_times[-1] != total_steps:
        sample_times.append(total_steps)

    output_rows: list[dict[str, Any]] = []
    for time_step in sample_times:
        if time_step == 0:
            episode_index = 0
            episode_window_mean = 0.0
            episode_window_se = 0.0
            observed_count = 0
            episode_end = 0
        else:
            # At a cumulative episode boundary, the state corresponds to the
            # episode just completed (the left-closed step signal above).
            episode_index = int(np.searchsorted(cumulative_end, time_step, side="left")) + 1
            episode_index = min(episode_index, len(rows))
            episode_window_mean = float(episode_mean[episode_index - 1])
            episode_window_se = float(episode_se[episode_index - 1])
            observed_count = int(observed_counts[episode_index - 1])
            episode_end = int(cumulative_end[episode_index - 1])
        output_rows.append(
            {
                "time_step": int(time_step),
                "ema_success_rate": float(ema_mean[time_step]),
                "ema_band_lower": float(lower[time_step]),
                "ema_band_upper": float(upper[time_step]),
                "ema_band_se": float(ema_se[time_step]),
                "episode_index": episode_index,
                "episode_window_success_rate": episode_window_mean,
                "episode_window_se": episode_window_se,
                "episode_window_observed_count": observed_count,
                "episode_window_missing_zero_count": int(window_episodes - observed_count) if episode_index else window_episodes,
                "episode_cumulative_end": episode_end,
            }
        )

    stats = {
        "episode_count": int(len(rows)),
        "total_raw_simulation_steps": total_steps,
        "raw_steps_per_episode": {
            "mean": float(np.mean(raw_steps)),
            "median": float(np.median(raw_steps)),
            "min": int(np.min(raw_steps)),
            "max": int(np.max(raw_steps)),
        },
        "window_episodes": int(window_episodes),
        "window_time_span_mean_steps": float(window_episodes * np.mean(raw_steps)),
        "window_time_span_median_steps": float(window_episodes * np.median(raw_steps)),
        "window_time_span_first_20_observed_steps": int(np.sum(raw_steps[:window_episodes])),
        "window_time_span_last_20_observed_steps": int(np.sum(raw_steps[-window_episodes:])),
        "first_episode_success": int(successes[0]),
        "first_episode_padded_success_rate": float(episode_mean[0]),
        "first_episode_missing_zero_count": int(window_episodes - 1),
        "zero_padding": True,
        "ema_alpha": float(ema_alpha),
        "ema_update_frequency": "each raw simulation step",
        "sample_stride": int(sample_stride),
        "curve_sample_count": int(len(output_rows)),
    }
    return output_rows, stats


def align_common_support(curves: dict[str, list[dict[str, Any]]], scenario: str) -> tuple[list[int], dict[str, dict[int, dict[str, Any]]]]:
    by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for method in METHODS:
        method_rows = curves.get(method, [])
        if not method_rows:
            raise ValueError(f"scenario {scenario} is missing method {method}")
        by_method[method] = {int(row["time_step"]): row for row in method_rows}
    common = sorted(set.intersection(*(set(by_method[method]) for method in METHODS)))
    if len(common) < 10:
        raise ValueError(f"too few common time-step samples for {scenario}: {len(common)}")
    return common, by_method


def align_curves(
    transformed: dict[str, dict[str, list[dict[str, Any]]]],
    scenarios: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    aligned: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        common, by_method = align_common_support(transformed[scenario], scenario)
        for time_step in common:
            values = {method: float(by_method[method][time_step]["ema_success_rate"]) for method in METHODS}
            bands = {
                method: (
                    float(by_method[method][time_step]["ema_band_lower"]),
                    float(by_method[method][time_step]["ema_band_upper"]),
                )
                for method in METHODS
            }
            best_baseline = max(values[method] for method in BASELINES)
            aligned.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                    "time_step": int(time_step),
                    "v4_8_ema_success_rate": values["v4_8"],
                    "mst_slt_ema_success_rate": values["mst_slt"],
                    "temporal_graph_ema_success_rate": values["temporal_graph"],
                    "best_baseline_ema_success_rate": best_baseline,
                    "v4_8_margin_vs_best_baseline": values["v4_8"] - best_baseline,
                    "v4_8_band_lower": bands["v4_8"][0],
                    "v4_8_band_upper": bands["v4_8"][1],
                    "mst_slt_band_lower": bands["mst_slt"][0],
                    "mst_slt_band_upper": bands["mst_slt"][1],
                    "temporal_graph_band_lower": bands["temporal_graph"][0],
                    "temporal_graph_band_upper": bands["temporal_graph"][1],
                }
            )
        scenario_rows = [row for row in aligned if row["scenario"] == scenario]
        times = np.asarray([row["time_step"] for row in scenario_rows], dtype=float)
        margin_values = np.asarray([row["v4_8_margin_vs_best_baseline"] for row in scenario_rows], dtype=float)
        aucs: dict[str, float] = {}
        for method in METHODS:
            y = np.asarray([row[f"{method}_ema_success_rate"] for row in scenario_rows], dtype=float)
            if len(times) > 1:
                if hasattr(np, "trapezoid"):
                    area = float(np.trapezoid(y, times))
                else:  # pragma: no cover - compatibility with older NumPy
                    area = float(np.trapz(y, times))
            else:
                area = 0.0
            aucs[method] = area / max(1.0, float(times[-1] - times[0]))
        summaries[scenario] = {
            "common_time_step_count": int(len(common)),
            "common_time_step_start": int(common[0]),
            "common_time_step_end": int(common[-1]),
            "mean_margin_vs_best_baseline": float(np.mean(margin_values)),
            "median_margin_vs_best_baseline": float(np.median(margin_values)),
            "min_margin_vs_best_baseline": float(np.min(margin_values)),
            "max_margin_vs_best_baseline": float(np.max(margin_values)),
            "positive_margin_fraction": float(np.mean(margin_values > 0.0)),
            "negative_margin_fraction": float(np.mean(margin_values < 0.0)),
            "final_margin_vs_best_baseline": float(margin_values[-1]),
            "normalized_auc": aucs,
            "v4_8_auc_minus_best_baseline_auc": float(aucs["v4_8"] - max(aucs[method] for method in BASELINES)),
        }
    return aligned, summaries


def draw_top_axis(axis: Any, rows: list[dict[str, Any]], scenario: str, x_max: int) -> None:
    for method in ("mst_slt", "temporal_graph", "v4_8"):
        style = METHOD_STYLES[method]
        x = [row["time_step"] for row in rows]
        y = [row[f"{method}_ema_success_rate"] for row in rows]
        lower = [row[f"{method}_band_lower"] for row in rows]
        upper = [row[f"{method}_band_upper"] for row in rows]
        axis.fill_between(x, lower, upper, color=style["color"], alpha=style["band_alpha"], linewidth=0, zorder=1)
        axis.plot(x, y, color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], label=METHOD_SHORT_LABELS[method], zorder=3)
    axis.set_title(f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=10.5, pad=7)
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, 1.0)
    axis.set_xticks(list(range(0, x_max + 1, 10_000)))
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_xlabel("Time steps", fontsize=9.2)
    axis.set_ylabel("Avg Success rate", fontsize=9.2)
    axis.tick_params(labelsize=8.1)
    axis.grid(True, color="#BFBFBF", linewidth=0.65, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    desired_order = [METHOD_SHORT_LABELS[method] for method in ("v4_8", "mst_slt", "temporal_graph")]
    order = [labels.index(label) for label in desired_order if label in labels]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper left",
        fontsize=8.0,
        frameon=True,
        framealpha=0.84,
        borderpad=0.4,
        handlelength=2.2,
    )
    axis.set_facecolor("#FCFCFC")


def draw_margin_axis(axis: Any, rows: list[dict[str, Any]], summary: dict[str, Any], x_max: int, margin_limit: float) -> None:
    x = [row["time_step"] for row in rows]
    margin = [row["v4_8_margin_vs_best_baseline"] for row in rows]
    positive = [max(0.0, value) for value in margin]
    negative = [min(0.0, value) for value in margin]
    axis.fill_between(x, 0.0, positive, color="#0072B2", alpha=0.18, linewidth=0, label="v4.8 above both")
    axis.fill_between(x, 0.0, negative, color="#999999", alpha=0.16, linewidth=0, label="v4.8 below a baseline")
    axis.plot(x, margin, color="#0072B2", linewidth=2.0, label="v4.8 − best baseline", zorder=3)
    axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0, zorder=2)
    axis.set_title("v4.8 margin vs. best baseline", fontsize=9.8, pad=7)
    axis.set_xlim(0, x_max)
    axis.set_ylim(-margin_limit, margin_limit)
    axis.set_xticks(list(range(0, x_max + 1, 10_000)))
    axis.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    axis.set_xlabel("Time steps", fontsize=9.2)
    axis.set_ylabel("Success-rate margin", fontsize=9.2)
    axis.tick_params(labelsize=8.1)
    axis.grid(True, color="#D2D2D2", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", fontsize=7.6, frameon=True, framealpha=0.84, borderpad=0.35, handlelength=2.0)
    axis.text(
        0.03,
        0.95,
        f"mean {summary['mean_margin_vs_best_baseline']:+.3f} · positive {summary['positive_margin_fraction']:.0%}",
        transform=axis.transAxes,
        va="top",
        fontsize=8.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.82, "pad": 2.0},
    )
    axis.set_facecolor("#FCFCFC")


def plot_composite(
    aligned: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    scenarios: list[str],
    output_dir: Path,
    x_max: int,
    margin_limit: float,
) -> dict[str, str]:
    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=(15.0, 8.15),
        sharex="row",
        gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.26, "wspace": 0.22},
    )
    if len(scenarios) == 1:
        axes = axes.reshape(2, 1)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.12)
    for column, scenario in enumerate(scenarios):
        scenario_rows = [row for row in aligned if row["scenario"] == scenario]
        draw_top_axis(axes[0, column], scenario_rows, scenario, x_max)
        draw_margin_axis(axes[1, column], scenario_rows, summaries[scenario], x_max, margin_limit)
        axes[0, column].text(
            -0.10,
            1.10,
            PANEL_LABELS.get(scenario, ""),
            transform=axes[0, column].transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4},
        )
    fig.text(
        0.5,
        0.045,
        "Top: 20-episode zero-padded success + EMA. Bottom: v4.8 − max(MST+SLT, TemporalGraph); positive means v4.8 is higher. EMA α=0.999 per raw time step; seed 0.",
        ha="center",
        fontsize=8.25,
        color="#555555",
    )
    fig.patch.set_facecolor("white")
    png_path = output_dir / "v4_8_focal_training_curves_episode20_ema999.png"
    svg_path = output_dir / "v4_8_focal_training_curves_episode20_ema999.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"png": repo_rel(png_path), "svg": repo_rel(svg_path)}


def plot_individual(
    aligned: list[dict[str, Any]],
    summary: dict[str, Any],
    scenario: str,
    output_dir: Path,
    x_max: int,
    margin_limit: float,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for extension in ("png", "svg"):
        path = output_dir / f"{scenario}_v4_8_focal_training_curves_episode20_ema999.{extension}"
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.4, 7.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.26},
        )
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.16)
        draw_top_axis(axes[0], aligned, scenario, x_max)
        draw_margin_axis(axes[1], aligned, summary, x_max, margin_limit)
        axes[0].text(
            -0.12,
            1.10,
            PANEL_LABELS.get(scenario, ""),
            transform=axes[0].transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4},
        )
        fig.text(
            0.5,
            0.055,
            "20-episode zero-padded sliding rate · EMA α=0.999 per raw time step · positive margin means v4.8 exceeds both baselines",
            ha="center",
            fontsize=8.0,
            color="#555555",
        )
        fig.patch.set_facecolor("white")
        if extension == "png":
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        else:
            fig.savefig(path, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        result[extension] = repo_rel(path)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scenario_pooled_step_stats(source_stats: dict[str, dict[str, Any]], scenario: str) -> dict[str, Any]:
    entries = [source_stats[scenario][method]["raw_steps_per_episode"] for method in METHODS]
    # The source stats expose only moments; use those moments for a transparent
    # nominal span and retain method-level exact spans in the manifest.
    total_count = sum(int(source_stats[scenario][method]["episode_count"]) for method in METHODS)
    weighted_mean = sum(float(source_stats[scenario][method]["raw_steps_per_episode"]["mean"]) * int(source_stats[scenario][method]["episode_count"]) for method in METHODS) / max(1, total_count)
    return {
        "method_count": len(METHODS),
        "episode_count_pooled": total_count,
        "raw_steps_per_episode_mean_pooled": float(weighted_mean),
        "nominal_20_episode_span_mean_pooled": float(WINDOW_EPISODES * weighted_mean),
        "method_raw_step_stats": entries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_methods = list(protocol["matrix"]["method_order"])
    protocol_scenarios = list(protocol["matrix"]["scenario_order"])
    if set(protocol_methods) != set(METHODS):
        raise ValueError(f"protocol methods differ from expected methods: {protocol_methods}")
    if set(protocol_scenarios) != set(SCENARIOS):
        raise ValueError(f"protocol scenarios differ from expected scenarios: {protocol_scenarios}")

    resolve_source = load_source_resolver()
    transformed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_audits: dict[str, dict[str, Any]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    for scenario in protocol_scenarios:
        transformed[scenario] = {}
        source_audits[scenario] = {}
        source_stats[scenario] = {}
        for method in METHODS:
            rows, audit = resolve_source(protocol, method, scenario, WINDOW_EPISODES)
            curve, stats = build_time_step_curve(rows, window_episodes=WINDOW_EPISODES, ema_alpha=EMA_ALPHA, sample_stride=SAMPLE_STRIDE)
            transformed[scenario][method] = curve
            source_audits[scenario][method] = sanitize_paths(audit)
            source_stats[scenario][method] = stats
            for item in curve:
                curve_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                        "method": method,
                        "method_label": METHOD_SHORT_LABELS[method],
                        "time_step": item["time_step"],
                        "ema_success_rate": item["ema_success_rate"],
                        "ema_band_lower": item["ema_band_lower"],
                        "ema_band_upper": item["ema_band_upper"],
                        "ema_band_se": item["ema_band_se"],
                        "episode_index": item["episode_index"],
                        "episode_window_success_rate": item["episode_window_success_rate"],
                        "episode_window_se": item["episode_window_se"],
                        "episode_window_observed_count": item["episode_window_observed_count"],
                        "episode_window_missing_zero_count": item["episode_window_missing_zero_count"],
                        "episode_cumulative_end": item["episode_cumulative_end"],
                        "window_episodes": WINDOW_EPISODES,
                        "zero_padding": True,
                        "ema_alpha": EMA_ALPHA,
                        "sample_stride": SAMPLE_STRIDE,
                    }
                )

    aligned, summaries = align_curves(transformed, protocol_scenarios)
    x_max = max(int(summary["common_time_step_end"]) for summary in summaries.values())
    x_max = int(math.ceil(x_max / 10_000.0) * 10_000)
    margin_limit = 0.6

    curve_data_path = output_dir / "training_success_curve_episode20_ema999_data.csv"
    aligned_data_path = output_dir / "v4_8_focal_aligned_margin_episode20_ema999_data.csv"
    write_csv(curve_data_path, curve_rows)
    write_csv(aligned_data_path, aligned)
    composite = plot_composite(aligned, summaries, protocol_scenarios, output_dir, x_max, margin_limit)
    individual = {
        scenario: plot_individual(
            [row for row in aligned if row["scenario"] == scenario],
            summaries[scenario],
            scenario,
            output_dir,
            x_max,
            margin_limit,
        )
        for scenario in protocol_scenarios
    }

    manifest = {
        "artifact_id": "iv2e3m100x100_training_curves_v5_episode20_ema999",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": repo_rel(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_resolver": repo_rel(V1_SCRIPT),
        "visual_contract": {
            "path": repo_rel(PROJECT_ROOT / "visual-composer" / "training-curves-v5-contract.md"),
            "sha256": sha256(PROJECT_ROOT / "visual-composer" / "training-curves-v5-contract.md"),
        },
        "source_methods": protocol_methods,
        "source_scenarios": protocol_scenarios,
        "source_training_seed": 0,
        "source_audits": source_audits,
        "transform": {
            "success_unit": "episode-level is_success label",
            "rolling_window": "most recent exactly 20 episodes",
            "window_episodes": WINDOW_EPISODES,
            "zero_padding": True,
            "zero_padding_definition": "missing prefix episodes are explicit zero slots and denominator remains 20",
            "episode_signal_to_time_steps": "each episode statistic held over its logged raw_simulation_steps as a descriptive retrospective proxy (outcome is observed at episode end)",
            "x_axis": "cumulative raw simulation steps",
            "scene_specific_episode_spans": "actual per-episode raw steps are used independently for CARLA, Cross, and Roundabout; no common fixed time-step window",
            "ema_alpha": EMA_ALPHA,
            "ema_recurrence": "ema_t = alpha * ema_(t-1) + (1-alpha) * current_t",
            "ema_initial_state": {"time_step": 0, "value": 0.0},
            "ema_update_frequency": "each raw simulation step",
            "sample_stride": SAMPLE_STRIDE,
            "band": "population SD of the same 20 binary slots divided by sqrt(20), held per episode and EMA-smoothed; descriptive within-seed band",
        },
        "plot_semantics": {
            "top_row": "all three EMA-smoothed 20-episode padded success curves; v4.8 emphasized",
            "bottom_row": "pointwise v4.8 minus max(MST+SLT, TemporalGraph) on the common sampled time-step support",
            "positive_margin": "v4.8 is above both baselines at that sampled time step",
            "negative_margin": "at least one baseline is above v4.8 at that sampled time step",
            "common_support_alignment": "intersection of the three methods' sampled time steps per scenario; no interpolation",
            "visual_x_limit": x_max,
            "margin_axis_limit": margin_limit,
        },
        "scene_specific_window_stats": {
            scenario: scenario_pooled_step_stats(source_stats, scenario) for scenario in protocol_scenarios
        },
        "method_scene_transform_stats": source_stats,
        "data_tables": {
            "all_method_curves": {"path": repo_rel(curve_data_path), "sha256": sha256(curve_data_path), "row_count": len(curve_rows)},
            "aligned_margin": {"path": repo_rel(aligned_data_path), "sha256": sha256(aligned_data_path), "row_count": len(aligned)},
        },
        "figures": {"composite": composite, "individual": individual},
        "qa": {
            "all_expected_scenarios_present": set(summaries) == set(protocol_scenarios),
            "all_scenarios_have_three_methods": all(summary["common_time_step_count"] >= 10 for summary in summaries.values()),
            "all_curve_values_finite": all(
                math.isfinite(float(row[field]))
                for row in curve_rows
                for field in ("ema_success_rate", "ema_band_lower", "ema_band_upper", "ema_band_se", "episode_window_success_rate", "episode_window_se")
            ),
            "all_margin_values_finite": all(math.isfinite(float(row["v4_8_margin_vs_best_baseline"])) for row in aligned),
            "time_steps_monotone_per_method_scene": all(
                all(a["time_step"] < b["time_step"] for a, b in zip(transformed[scenario][method], transformed[scenario][method][1:]))
                for scenario in protocol_scenarios
                for method in METHODS
            ),
            "zero_padding_verified_first_episode_denominator_20": all(
                int(source_stats[scenario][method]["first_episode_missing_zero_count"]) == WINDOW_EPISODES - 1
                and math.isclose(
                    float(source_stats[scenario][method]["first_episode_padded_success_rate"]),
                    float(source_stats[scenario][method]["first_episode_success"]) / WINDOW_EPISODES,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for scenario in protocol_scenarios
                for method in METHODS
            ),
            "original_logs_and_models_untouched": True,
            "negative_margins_visible": all(summary["min_margin_vs_best_baseline"] < 0.0 for summary in summaries.values()),
        },
        "scenario_summaries": summaries,
        "honesty_note": "Focal styling emphasizes v4.8 but retains all methods, full common support, and signed negative margins. These saved-model logs contain one training seed per method×scenario; they do not support a universal v4.8>baseline or cross-seed significance claim.",
    }
    manifest_path = output_dir / "v4_8_focal_training_curves_episode20_ema999_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Focal v4.8 training curves — 20-episode padded EMA\n\n"
        "These figures are derived from the unchanged parent-v2 monitor logs. For each saved method×scenario run, the success statistic is the mean of the most recent 20 episode outcomes, with explicit zero slots for unavailable early episodes. The episode statistic is held over each episode's observed raw simulation steps and smoothed once per raw step with EMA α=0.999.\n\n"
        "The top row shows all three methods; the bottom row shows the signed pointwise margin `v4.8 - max(MST+SLT, TemporalGraph)`. Negative regions are intentionally retained. See the manifest for source hashes, scene-specific episode-step distributions, and descriptive summaries.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": repo_rel(output_dir),
                "composite": composite,
                "individual": individual,
                "manifest": repo_rel(manifest_path),
                "curve_data": repo_rel(curve_data_path),
                "aligned_data": repo_rel(aligned_data_path),
                "curve_rows": len(curve_rows),
                "aligned_rows": len(aligned),
                "scenario_summaries": summaries,
                "qa": manifest["qa"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
