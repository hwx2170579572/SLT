"""Plot scene-specific training reward and success curves.

The input is the complete independent-v2 five-method by six-scenario matrix.
The script only reads the saved ``train_monitor.csv`` files.  For every
method×scene cell it keeps the scene's actual per-episode raw simulation-step
lengths, computes a trailing 20-episode reward mean, computes a fixed-
denominator 20-episode success rate with zero-padded prefix slots, holds each
episode statistic over that episode's logged raw steps, and applies EMA(0.999)
at every raw step.  Curves are sampled every 100 steps for rendering only.

No macro-average curve is produced and no training/evaluation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "independent_v2_five_methods_six_scenarios_100ep_v1" / "protocol.json"
SUMMARY_PATH = PROJECT_ROOT / "results_iv2_5m6s100e_v1" / "comparison" / "summary" / "summary.json"
CONTRACT_PATH = PROJECT_ROOT / "visual-composer" / "independent_v2_5m6s100e_v1" / "visual-contract.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "independent_v2_5m6s100e_training_curves_v1"

WINDOW_EPISODES = 20
EMA_ALPHA = 0.999
SAMPLE_STRIDE = 100

SCENARIO_LABELS = {
    "left_turn": "Left-turn",
    "cross": "Cross",
    "roundabout_easy": "Roundabout-A",
    "roundabout_medium": "Roundabout-B",
    "roundabout": "Roundabout-C",
    "carla": "CARLA",
}
PANEL_LABELS = {
    "left_turn": "(a)",
    "cross": "(b)",
    "roundabout_easy": "(c)",
    "roundabout_medium": "(d)",
    "roundabout": "(e)",
    "carla": "(f)",
}

# Okabe-Ito-compatible colors plus distinct line styles for grayscale output.
METHOD_STYLES = {
    "mst_slt": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "temporal_graph": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "full_balanced": {"color": "#009E73", "linestyle": ":", "marker": "^"},
    "v4_8": {"color": "#D55E00", "linestyle": "-.", "marker": "D"},
    "v4_13": {"color": "#CC79A7", "linestyle": (0, (5, 1, 1, 1)), "marker": "P"},
}
METHOD_SHORT_LABELS = {
    "mst_slt": "MST+SLT",
    "temporal_graph": "TemporalGraph",
    "full_balanced": "Full+BalancedSlots",
    "v4_8": "v4.8",
    "v4_13": "v4.13",
}

REQUIRED_MONITOR_FIELDS = {
    "r",
    "raw_simulation_steps",
    "is_success",
}

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_bool(value: Any, *, path: Path, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean {value!r} at {path}:{row_number}")


def read_monitor(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one Stable-Baselines monitor file and preserve episode order."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"monitor log has no data rows: {path}")
    metadata_line = lines[0]
    reader = csv.DictReader(lines[1:])
    fields = {str(field).strip() for field in (reader.fieldnames or []) if field is not None}
    missing = REQUIRED_MONITOR_FIELDS - fields
    if missing:
        raise ValueError(f"monitor log {path} is missing fields: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    cumulative = 0
    for row_number, row in enumerate(reader, start=3):
        if not row or all(value in (None, "") for value in row.values()):
            continue
        try:
            reward = float(row["r"])
            raw_steps = int(round(float(row["raw_simulation_steps"])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric monitor row at {path}:{row_number}") from exc
        if not math.isfinite(reward) or raw_steps <= 0:
            raise ValueError(f"invalid reward/raw steps at {path}:{row_number}")
        cumulative += raw_steps
        rows.append(
            {
                "episode_index": len(rows) + 1,
                "return": reward,
                "is_success": int(parse_bool(row["is_success"], path=path, row_number=row_number)),
                "raw_simulation_steps": raw_steps,
                "cumulative_raw_simulation_steps": cumulative,
            }
        )
    if not rows:
        raise ValueError(f"monitor log has no parsed rows: {path}")
    return rows, {
        "metadata_line": metadata_line,
        "fields": list(reader.fieldnames or []),
        "episode_count": len(rows),
        "total_raw_simulation_steps": cumulative,
        "raw_steps_min": min(row["raw_simulation_steps"] for row in rows),
        "raw_steps_max": max(row["raw_simulation_steps"] for row in rows),
        "raw_steps_mean": float(np.mean([row["raw_simulation_steps"] for row in rows])),
        "return_min": min(row["return"] for row in rows),
        "return_max": max(row["return"] for row in rows),
        "monitor_sha256": sha256(path),
    }


def rolling_reward(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing reward mean; early prefix uses its available denominator."""

    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty(values.size, dtype=np.float64)
    denominators = np.empty(values.size, dtype=np.int64)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        denominator = index + 1 - start
        result[index] = (prefix[index + 1] - prefix[start]) / float(denominator)
        denominators[index] = denominator
    return result, denominators


def padded_success(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-denominator trailing success rate with explicit zero padding."""

    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty(values.size, dtype=np.float64)
    observed_counts = np.empty(values.size, dtype=np.int64)
    missing_counts = np.empty(values.size, dtype=np.int64)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        observed = index + 1 - start
        result[index] = (prefix[index + 1] - prefix[start]) / float(window)
        observed_counts[index] = observed
        missing_counts[index] = window - observed
    return result, observed_counts, missing_counts


def expand_ema(
    episode_signal: np.ndarray,
    raw_steps: np.ndarray,
    *,
    ema_alpha: float,
    sample_stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hold episode values over raw steps and smooth once at every raw step."""

    if not 0.0 < ema_alpha < 1.0:
        raise ValueError("EMA alpha must be in (0, 1)")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    step_signal = np.repeat(episode_signal.astype(np.float64), raw_steps.astype(np.int64))
    total_steps = int(step_signal.size)
    ema = np.empty(total_steps + 1, dtype=np.float64)
    ema[0] = 0.0
    one_minus_alpha = 1.0 - float(ema_alpha)
    for index, value in enumerate(step_signal):
        ema[index + 1] = float(ema_alpha) * ema[index] + one_minus_alpha * float(value)

    cumulative_end = np.cumsum(raw_steps, dtype=np.int64)
    sample_times = [0]
    sample_times.extend(range(sample_stride, total_steps + 1, sample_stride))
    if sample_times[-1] != total_steps:
        sample_times.append(total_steps)
    curve: list[dict[str, Any]] = []
    for time_step in sample_times:
        if time_step == 0:
            episode_index = 0
        else:
            episode_index = int(np.searchsorted(cumulative_end, time_step, side="left")) + 1
            episode_index = min(episode_index, len(raw_steps))
        curve.append(
            {
                "time_step": int(time_step),
                "ema_value": float(ema[time_step]),
                "episode_index": episode_index,
                "episode_cumulative_end": int(cumulative_end[episode_index - 1]) if episode_index else 0,
            }
        )
    return curve, {
        "total_raw_simulation_steps": total_steps,
        "sample_count": len(curve),
        "sample_stride": sample_stride,
        "ema_alpha": ema_alpha,
        "ema_initial_value": 0.0,
        "ema_update_frequency": "every raw simulation step",
    }


def resolve_cell_run_dir(protocol: dict[str, Any], method: str, scenario: str) -> tuple[Path, str]:
    parent = protocol["parent_v2_reuse"]
    parent_methods = set(parent["methods"])
    parent_scenarios = set(parent["scenarios"])
    if method in parent_methods and scenario in parent_scenarios:
        run_name = str(parent["run_id_template"]).format(method=method, scenario=scenario)
        run_dir = PROJECT_ROOT / str(parent["result_root"]) / str(parent["profile"]) / "runs" / run_name
        return run_dir.resolve(), "adopted_parent_v2"
    run_name = f"{protocol['run_id_prefix']}__comparison__{method}__{scenario}__seed0"
    run_dir = PROJECT_ROOT / str(protocol["artifact_contract"]["result_root"]) / "comparison" / "runs" / run_name
    return run_dir.resolve(), "fresh_i5m6s100_v1"


def read_source_cell(
    protocol: dict[str, Any],
    summary_row: dict[str, Any],
    method: str,
    scenario: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir, expected_source = resolve_cell_run_dir(protocol, method, scenario)
    if str(summary_row.get("source")) != expected_source:
        raise ValueError(f"source mismatch for {method}/{scenario}: {summary_row.get('source')} != {expected_source}")
    if summary_row.get("status") != "complete" or int(summary_row.get("evaluation_episodes", -1)) != 100:
        raise ValueError(f"summary cell is not complete for {method}/{scenario}")
    if Path(str(summary_row.get("run_directory"))).resolve() != run_dir:
        raise ValueError(f"run directory mismatch for {method}/{scenario}")
    monitor_path = run_dir / "train_monitor.csv"
    if not monitor_path.is_file():
        raise FileNotFoundError(f"missing training monitor: {monitor_path}")
    rows, audit = read_monitor(monitor_path)
    return rows, {
        "method": method,
        "scenario": scenario,
        "source": expected_source,
        "run_directory": repo_rel(run_dir),
        "monitor_path": repo_rel(monitor_path),
        "monitor_sha256": audit["monitor_sha256"],
        "evaluation_episodes": int(summary_row["evaluation_episodes"]),
        "detailed_evaluation_sha256": str(summary_row["detailed_evaluation_sha256"]),
        "monitor": audit,
    }


def fmt_k(value: float, _position: int) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def setup_axis(axis: Any, *, scenario: str, x_max: int, metric: str) -> None:
    axis.set_title(f"{PANEL_LABELS[scenario]} {SCENARIO_LABELS[scenario]}", loc="left", fontsize=11, pad=8, fontweight="bold")
    axis.set_xlim(0, x_max)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    axis.set_xlabel("Cumulative raw simulation time steps", fontsize=9)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(True, color="#D0D0D0", linewidth=0.6, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_facecolor("#FCFCFC")
    if metric == "reward":
        axis.set_ylim(-1.05, 1.05)
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.set_ylabel("EMA trailing mean reward", fontsize=9)
    else:
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_ylabel("EMA padded 20-episode success", fontsize=9)


def plot_composite(
    curves: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
    methods: list[str],
    scenarios: list[str],
    output_dir: Path,
    *,
    metric: str,
) -> dict[str, str]:
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.0), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.13, hspace=0.34, wspace=0.22)
    for index, scenario in enumerate(scenarios):
        row_index, col_index = divmod(index, 3)
        axis = axes[row_index, col_index]
        x_max = max(int(curves[(scenario, method)]["curve"][-1]["time_step"]) for method in methods)
        setup_axis(axis, scenario=scenario, x_max=x_max, metric=metric)
        y_field = "ema_reward" if metric == "reward" else "ema_success_rate"
        for method in methods:
            data = curves[(scenario, method)]["curve"]
            style = METHOD_STYLES[method]
            x = [point["time_step"] for point in data]
            y = [point[y_field] for point in data]
            axis.plot(x, y, color=style["color"], linestyle=style["linestyle"], linewidth=1.75, label=METHOD_SHORT_LABELS[method])
    handles = [
        Line2D([0], [0], color=METHOD_STYLES[method]["color"], linestyle=METHOD_STYLES[method]["linestyle"], linewidth=2.0, label=METHOD_SHORT_LABELS[method])
        for method in methods
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(methods), frameon=False, fontsize=9.2, handlelength=2.7)
    title = "Training reward by scene" if metric == "reward" else "Training success by scene"
    fig.suptitle(f"{title} — independent-v2 high-traffic matrix", fontsize=14, y=0.995)
    note = (
        "Reward: trailing mean of up to 20 episodes; success: fixed 20 slots with prefix zero padding; "
        f"EMA α={EMA_ALPHA:g} at every raw step; curves sampled every {SAMPLE_STRIDE} steps."
    )
    fig.text(0.5, 0.035, note, ha="center", fontsize=8.2, color="#555555")
    fig.patch.set_facecolor("white")
    stem = "training_reward_curves_by_scene_ema999" if metric == "reward" else "training_success_curves_by_scene_episode20_ema999"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"png": repo_rel(png_path), "svg": repo_rel(svg_path)}


def plot_individual(
    curves: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
    methods: list[str],
    scenario: str,
    output_dir: Path,
    *,
    metric: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    x_max = max(int(curves[(scenario, method)]["curve"][-1]["time_step"]) for method in methods)
    for extension in ("png", "svg"):
        stem = f"{scenario}_{'reward' if metric == 'reward' else 'success'}_curve_{'ema999' if metric == 'reward' else 'episode20_ema999'}"
        path = output_dir / f"{stem}.{extension}"
        fig, axis = plt.subplots(figsize=(7.2, 4.7))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.19)
        setup_axis(axis, scenario=scenario, x_max=x_max, metric=metric)
        y_field = "ema_reward" if metric == "reward" else "ema_success_rate"
        for method in methods:
            data = curves[(scenario, method)]["curve"]
            style = METHOD_STYLES[method]
            axis.plot([point["time_step"] for point in data], [point[y_field] for point in data], color=style["color"], linestyle=style["linestyle"], linewidth=2.0, label=METHOD_SHORT_LABELS[method])
        axis.legend(loc="best", fontsize=8.5, frameon=True, framealpha=0.9, ncol=2)
        fig.suptitle(f"{SCENARIO_LABELS[scenario]} — {'training reward' if metric == 'reward' else 'training success'}", fontsize=12.5, y=0.97)
        fig.text(0.5, 0.055, f"EMA α={EMA_ALPHA:g} per raw step · {'20-episode trailing reward mean' if metric == 'reward' else '20-episode success with zero-padded prefix'} · seed 0", ha="center", fontsize=7.8, color="#555555")
        fig.patch.set_facecolor("white")
        fig.savefig(path, dpi=300 if extension == "png" else None, facecolor="white", bbox_inches="tight")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-episodes", type=int, default=WINDOW_EPISODES)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--sample-stride", type=int, default=SAMPLE_STRIDE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.window_episodes != WINDOW_EPISODES:
        raise ValueError("this artifact is fixed to a 20-episode window")
    if not 0.0 < args.ema_alpha < 1.0:
        raise ValueError("ema-alpha must be between 0 and 1")
    if args.sample_stride <= 0:
        raise ValueError("sample-stride must be positive")
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    methods = list(protocol["matrix"]["method_order"])
    scenarios = list(protocol["matrix"]["scenario_order"])
    summary_rows = summary.get("method_scenario")
    if not isinstance(summary_rows, list) or len(summary_rows) != len(methods) * len(scenarios):
        raise ValueError("summary does not contain the complete method×scenario matrix")
    summary_by_cell = {(str(row["method"]), str(row["scenario"])): row for row in summary_rows}
    if len(summary_by_cell) != len(summary_rows):
        raise ValueError("summary contains duplicate method×scenario cells")

    curves: dict[tuple[str, str], dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    for scenario in scenarios:
        for method in methods:
            summary_key = (method, scenario)
            curve_key = (scenario, method)
            if summary_key not in summary_by_cell:
                raise ValueError(f"missing summary cell {method}/{scenario}")
            source_rows, source_audit = read_source_cell(protocol, summary_by_cell[summary_key], method, scenario)
            rewards = np.asarray([float(row["return"]) for row in source_rows], dtype=np.float64)
            successes = np.asarray([float(row["is_success"]) for row in source_rows], dtype=np.float64)
            raw_steps = np.asarray([int(row["raw_simulation_steps"]) for row in source_rows], dtype=np.int64)
            reward_mean, reward_denominator = rolling_reward(rewards, args.window_episodes)
            success_mean, success_observed, success_missing = padded_success(successes, args.window_episodes)
            reward_curve, reward_ema_audit = expand_ema(reward_mean, raw_steps, ema_alpha=args.ema_alpha, sample_stride=args.sample_stride)
            success_curve, success_ema_audit = expand_ema(success_mean, raw_steps, ema_alpha=args.ema_alpha, sample_stride=args.sample_stride)
            combined_curve = []
            success_by_time = {int(point["time_step"]): point for point in success_curve}
            for point in reward_curve:
                success_point = success_by_time[int(point["time_step"])]
                combined_curve.append({
                    "time_step": int(point["time_step"]),
                    "ema_reward": float(point["ema_value"]),
                    "ema_success_rate": float(success_point["ema_value"]),
                    "episode_index": int(point["episode_index"]),
                    "episode_cumulative_end": int(point["episode_cumulative_end"]),
                })
            curves[curve_key] = {
                "curve": combined_curve,
                "reward_curve": reward_curve,
                "success_curve": success_curve,
                "source": source_audit["source"],
                "monitor": source_audit["monitor"],
            }
            for row, reward_value, reward_denom, success_value, observed, missing in zip(source_rows, reward_mean, reward_denominator, success_mean, success_observed, success_missing):
                episode_rows.append({
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "method": method,
                    "method_label": METHOD_SHORT_LABELS[method],
                    "source": source_audit["source"],
                    "episode_index": int(row["episode_index"]),
                    "raw_simulation_steps": int(row["raw_simulation_steps"]),
                    "cumulative_raw_simulation_steps": int(row["cumulative_raw_simulation_steps"]),
                    "return": float(row["return"]),
                    "reward_trailing_mean_20": float(reward_value),
                    "reward_observed_denominator": int(reward_denom),
                    "is_success": int(row["is_success"]),
                    "success_padded_mean_20": float(success_value),
                    "success_observed_count": int(observed),
                    "success_missing_zero_count": int(missing),
                    "window_episodes": args.window_episodes,
                    "ema_alpha": args.ema_alpha,
                })
            for point in combined_curve:
                sampled_rows.append({
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "method": method,
                    "method_label": METHOD_SHORT_LABELS[method],
                    "source": source_audit["source"],
                    "time_step": int(point["time_step"]),
                    "ema_reward": float(point["ema_reward"]),
                    "ema_success_rate": float(point["ema_success_rate"]),
                    "episode_index": int(point["episode_index"]),
                    "episode_cumulative_end": int(point["episode_cumulative_end"]),
                    "window_episodes": args.window_episodes,
                    "zero_padded_success": True,
                    "ema_alpha": args.ema_alpha,
                    "sample_stride": args.sample_stride,
                })
            source_audit["reward_transform"] = reward_ema_audit
            source_audit["success_transform"] = success_ema_audit
            source_audit["first_episode_success"] = int(successes[0])
            source_audit["first_episode_success_padded_rate"] = float(success_mean[0])
            source_audit["first_episode_missing_zero_count"] = int(success_missing[0])
            source_audits.append(source_audit)

    episode_data_path = output_dir / "training_curves_episode_level.csv"
    sampled_data_path = output_dir / "training_curves_time_step_ema999.csv"
    write_csv(episode_data_path, episode_rows)
    write_csv(sampled_data_path, sampled_rows)
    figure_paths = {
        "reward_composite": plot_composite(curves, methods, scenarios, output_dir, metric="reward"),
        "success_composite": plot_composite(curves, methods, scenarios, output_dir, metric="success"),
        "reward_individual": {scenario: plot_individual(curves, methods, scenario, output_dir, metric="reward") for scenario in scenarios},
        "success_individual": {scenario: plot_individual(curves, methods, scenario, output_dir, metric="success") for scenario in scenarios},
    }

    all_time_steps = [int(row["time_step"]) for row in sampled_rows]
    all_finite = all(math.isfinite(float(row[field])) for row in sampled_rows for field in ("ema_reward", "ema_success_rate"))
    monotone = all(
        all(a["time_step"] < b["time_step"] for a, b in zip(curves[(scenario, method)]["curve"], curves[(scenario, method)]["curve"][1:]))
        for scenario in scenarios
        for method in methods
    )
    zero_padding_verified = all(
        audit["first_episode_missing_zero_count"] == args.window_episodes - 1
        and math.isclose(audit["first_episode_success_padded_rate"], audit["first_episode_success"] / args.window_episodes, rel_tol=0.0, abs_tol=1e-12)
        for audit in source_audits
    )
    source_count = {source: sum(audit["source"] == source for audit in source_audits) for source in ("adopted_parent_v2", "fresh_i5m6s100_v1")}
    qa = {
        "expected_cells": len(methods) * len(scenarios),
        "observed_cells": len(source_audits),
        "expected_scenarios": len(scenarios),
        "expected_methods": len(methods),
        "episode_level_rows": len(episode_rows),
        "sampled_curve_rows": len(sampled_rows),
        "source_count": source_count,
        "all_source_logs_nonempty": all(audit["monitor"]["episode_count"] > 0 for audit in source_audits),
        "all_curve_values_finite": all_finite,
        "time_steps_monotone_per_cell": monotone,
        "zero_padding_verified": zero_padding_verified,
        "no_macro_average_curve": True,
    }
    manifest = {
        "artifact_id": "independent_v2_5m6s100e_training_curves_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": repo_rel(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "summary_path": repo_rel(SUMMARY_PATH),
        "summary_sha256": sha256(SUMMARY_PATH),
        "visual_contract_path": repo_rel(CONTRACT_PATH),
        "visual_contract_sha256": sha256(CONTRACT_PATH),
        "matrix": {"methods": methods, "scenarios": scenarios, "cells": len(source_audits), "training_seed": 0},
        "transform": {
            "x_axis": "scene/method-specific cumulative raw_simulation_steps",
            "episode_lengths": "actual raw_simulation_steps from each scene/method monitor; no fixed episode length and no cross-scene resampling",
            "reward": "mean return over the most recent 20 episodes; early prefix denominator is the number of observed episodes",
            "success": "mean of exactly 20 binary is_success slots; unavailable prefix episodes are explicit zeros",
            "episode_to_time_step": "hold each episode statistic over its logged raw_simulation_steps",
            "ema_alpha": args.ema_alpha,
            "ema_recurrence": "ema_t = alpha*ema_(t-1) + (1-alpha)*signal_t",
            "ema_initial_value": 0.0,
            "ema_update_frequency": "every raw simulation step",
            "sample_stride": args.sample_stride,
            "sample_stride_is_render_only": True,
        },
        "source_audits": source_audits,
        "data_tables": {
            "episode_level": {"path": repo_rel(episode_data_path), "sha256": sha256(episode_data_path), "row_count": len(episode_rows)},
            "time_step_ema": {"path": repo_rel(sampled_data_path), "sha256": sha256(sampled_data_path), "row_count": len(sampled_rows)},
        },
        "figures": figure_paths,
        "qa": qa,
        "limitations": [
            "One training seed (seed 0) per method×scene; curves are descriptive and do not provide cross-seed uncertainty.",
            "Episode success is observed at episode completion, so holding it over the episode duration is an explicit retrospective time-step proxy.",
            "No macro-average curve is calculated or plotted.",
        ],
    }
    manifest_path = output_dir / "training_curves_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Independent-v2 scene-specific training curves\n\n"
        "The two composite figures contain six scene panels and all five methods. "
        "The x-axis is each method×scene log's cumulative raw simulation steps. "
        "Reward is a trailing mean over up to 20 episodes; success is a fixed 20-slot window with explicit prefix zero padding. "
        f"Both signals are held over observed episode lengths and smoothed at every raw step with EMA α={args.ema_alpha:g}; samples are rendered every {args.sample_stride} steps.\n\n"
        "No macro-average curve is generated. See `training_curves_manifest.json` for source hashes and QA.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": repo_rel(output_dir), "manifest": repo_rel(manifest_path), "figures": figure_paths, "qa": qa}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
