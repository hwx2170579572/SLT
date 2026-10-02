"""Plot training curves for the nine saved parent-v2 models.

This script intentionally reads the parent-v2 artifacts referenced by the
independent-v2 protocol.  It never loads a policy for training and never
creates synthetic observations or metrics.  The primary curves come from
the per-episode ``train_monitor.csv`` logs; TensorBoard files are audited and
their available scalar tags are recorded in the manifest for traceability.

Run with the project's plotting environment, for example::

    conda run -n llm_copy python visual-composer/plot_training_curves_v1.py

Outputs are written to ``outputs/iv2e3m100x100_training_curves_v1`` unless an
alternative output directory is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Use a non-interactive backend so the same script works on a headless worker.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


# The method label supplied by the protocol contains Chinese characters.  A
# Windows CJK font is present on the experiment host; registering it avoids
# missing-glyph warnings and keeps the labels visible in both PNG and SVG.
_CJK_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
if _CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT_PATH))
    _CJK_FONT_FAMILY = font_manager.FontProperties(fname=str(_CJK_FONT_PATH)).get_name()
    plt.rcParams["font.family"] = _CJK_FONT_FAMILY
else:  # pragma: no cover - only used on non-Windows plotting hosts
    _CJK_FONT_FAMILY = "DejaVu Sans"
    plt.rcParams["font.family"] = _CJK_FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_v2_existing_3methods_100seeds_100episodes_v1"
    / "protocol.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v1"

METHOD_FALLBACK_LABELS = {
    "mst_slt": "MST+SLT",
    "temporal_graph": "TemporalGraph 时序车辆图基线",
    "v4_8": "v4.8 Tie-only Replicated Calibration",
}
# Wrapped legend labels preserve the protocol's full method names while
# preventing the long v4.8/TemporalGraph labels from being clipped in the
# right-hand legend column.  The unwrapped labels remain in the CSV/manifest.
METHOD_LEGEND_LABELS = {
    "mst_slt": "MST+SLT",
    "temporal_graph": "TemporalGraph\n时序车辆图基线",
    "v4_8": "v4.8 Tie-only\nReplicated Calibration",
}
SCENARIO_LABELS = {
    "carla": "CARLA",
    "cross": "Cross",
    "roundabout": "Roundabout",
}

# Okabe--Ito colours, paired with line styles so grayscale output remains
# interpretable.  The vermillion accent is kept for the proposed v4.8 method
# in every scenario.
METHOD_STYLES = {
    "mst_slt": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "temporal_graph": {"color": "#56B4E9", "linestyle": "--", "marker": "s"},
    "v4_8": {"color": "#D55E00", "linestyle": "-.", "marker": "D"},
}

REQUIRED_MONITOR_FIELDS = {
    "r",
    "l",
    "t",
    "raw_simulation_steps",
    "is_success",
    "collision",
    "off_route",
    "max_time",
}
REQUIRED_TENSORBOARD_TAGS = {
    "rollout/ep_rew_mean",
    "rollout/source_success_rate_last_20",
    "rollout/success_rate",
    "time/raw_simulation_steps",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def as_bool(value: Any, *, path: Path, row_number: int, field: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean {value!r} at {path}:{row_number} field={field}")


def finite_float(value: Any, *, path: Path, row_number: int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number {value!r} at {path}:{row_number} field={field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite number at {path}:{row_number} field={field}")
    return result


def rolling_mean(values: Iterable[float], window: int) -> list[float]:
    result: list[float] = []
    buffer: deque[float] = deque(maxlen=window)
    for value in values:
        buffer.append(float(value))
        result.append(float(sum(buffer) / len(buffer)))
    return result


def load_monitor_rows(
    path: Path,
    *,
    method: str,
    method_label: str,
    scenario: str,
    scenario_label: str,
    run_id: str,
    model_path: Path,
    model_sha256: str,
    rolling_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"monitor log has no header/data rows: {path}")
    # Stable-Baselines Monitor writes a JSON metadata line before the CSV
    # header.  Keep it in the manifest but exclude it from DictReader.
    monitor_metadata = lines[0]
    reader = csv.DictReader(lines[1:])
    if reader.fieldnames is None:
        raise ValueError(f"monitor log has no CSV header: {path}")
    fields = {str(name).strip() for name in reader.fieldnames if name is not None}
    missing = REQUIRED_MONITOR_FIELDS - fields
    if missing:
        raise ValueError(f"monitor log {path} is missing fields: {sorted(missing)}")

    parsed: list[dict[str, Any]] = []
    cumulative_steps = 0
    previous_cumulative = 0
    for row_number, row in enumerate(reader, start=3):
        if not row or all(value in (None, "") for value in row.values()):
            continue
        raw_steps = int(
            round(
                finite_float(
                    row["raw_simulation_steps"],
                    path=path,
                    row_number=row_number,
                    field="raw_simulation_steps",
                )
            )
        )
        if raw_steps <= 0:
            raise ValueError(f"raw_simulation_steps must be positive at {path}:{row_number}")
        cumulative_steps += raw_steps
        if cumulative_steps <= previous_cumulative:
            raise ValueError(f"cumulative raw steps are not increasing at {path}:{row_number}")
        previous_cumulative = cumulative_steps
        parsed.append(
            {
                "episode_index": len(parsed) + 1,
                "raw_simulation_steps": raw_steps,
                "cumulative_raw_simulation_steps": cumulative_steps,
                "return": finite_float(row["r"], path=path, row_number=row_number, field="r"),
                "episode_length": finite_float(row["l"], path=path, row_number=row_number, field="l"),
                "elapsed_seconds": finite_float(row["t"], path=path, row_number=row_number, field="t"),
                "is_success": as_bool(row["is_success"], path=path, row_number=row_number, field="is_success"),
                "collision": as_bool(row["collision"], path=path, row_number=row_number, field="collision"),
                "off_route": as_bool(row["off_route"], path=path, row_number=row_number, field="off_route"),
                "max_time": as_bool(row["max_time"], path=path, row_number=row_number, field="max_time"),
            }
        )
    if not parsed:
        raise ValueError(f"monitor log has no data rows: {path}")

    return_values = [float(row["return"]) for row in parsed]
    success_values = [1.0 if row["is_success"] else 0.0 for row in parsed]
    return_means = rolling_mean(return_values, rolling_window)
    success_means = rolling_mean(success_values, rolling_window)
    long_rows: list[dict[str, Any]] = []
    for row, return_mean, success_mean in zip(parsed, return_means, success_means):
        long_rows.append(
            {
                "method": method,
                "method_label": method_label,
                "scenario": scenario,
                "scenario_label": scenario_label,
                "run_id": run_id,
                "training_seed": 0,
                "model_path": str(model_path.resolve()),
                "model_sha256": model_sha256,
                "monitor_path": str(path.resolve()),
                "episode_index": row["episode_index"],
                "raw_simulation_steps": row["raw_simulation_steps"],
                "cumulative_raw_simulation_steps": row["cumulative_raw_simulation_steps"],
                "return": row["return"],
                "return_trailing_mean_20": return_mean,
                "episode_length": row["episode_length"],
                "elapsed_seconds": row["elapsed_seconds"],
                "is_success": int(row["is_success"]),
                "success_trailing_mean_20": success_mean,
                "collision": int(row["collision"]),
                "off_route": int(row["off_route"]),
                "max_time": int(row["max_time"]),
            }
        )

    audit = {
        "monitor_path": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "monitor_metadata_line": monitor_metadata,
        "monitor_fields": list(reader.fieldnames),
        "episode_rows": len(long_rows),
        "raw_simulation_steps_sum": int(sum(row["raw_simulation_steps"] for row in parsed)),
        "raw_simulation_steps_min": int(min(row["raw_simulation_steps"] for row in parsed)),
        "raw_simulation_steps_max": int(max(row["raw_simulation_steps"] for row in parsed)),
        "return_min": float(min(return_values)),
        "return_max": float(max(return_values)),
        "return_mean": float(sum(return_values) / len(return_values)),
        "success_rate_mean": float(sum(success_values) / len(success_values)),
        "collision_rate_mean": float(sum(1.0 if row["collision"] else 0.0 for row in parsed) / len(parsed)),
        "off_route_rate_mean": float(sum(1.0 if row["off_route"] else 0.0 for row in parsed) / len(parsed)),
        "max_time_rate_mean": float(sum(1.0 if row["max_time"] else 0.0 for row in parsed) / len(parsed)),
        "rolling_window_episodes": rolling_window,
    }
    return long_rows, audit


def audit_tensorboard(event_paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": bool(event_paths),
        "event_paths": [str(path.resolve()) for path in event_paths],
        "events": [],
        "required_tags_present": False,
        "error": None,
    }
    if not event_paths:
        result["error"] = "no TensorBoard event file found"
        return result
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_summaries: list[dict[str, Any]] = []
        all_required = True
        for path in sorted(event_paths):
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = list(accumulator.Tags().get("scalars", []))
            counts = {tag: len(accumulator.Scalars(tag)) for tag in tags}
            required_present = REQUIRED_TENSORBOARD_TAGS.issubset(set(tags))
            all_required = all_required and required_present
            event_summaries.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "scalar_tags": tags,
                    "scalar_counts": counts,
                    "required_tags_present": required_present,
                }
            )
        result["events"] = event_summaries
        result["required_tags_present"] = all_required
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        result["error"] = f"TensorBoard audit failed: {type(exc).__name__}: {exc}"
    return result


def resolve_source(
    protocol: dict[str, Any], method: str, scenario: str, rolling_window: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent = protocol["parent_v2"]
    run_id = str(parent["run_id_template"]).format(method=method, scenario=scenario)
    run_dir = PROJECT_ROOT / str(parent["result_root"]) / str(parent["profile"]) / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing parent run directory: {run_dir}")
    arguments_path = run_dir / "arguments.json"
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    receipt_path = run_dir / "high_density_job_receipt.json"
    for artifact in (arguments_path, detailed_path, receipt_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"missing required parent artifact: {artifact}")
    arguments = read_json(arguments_path)
    detailed = read_json(detailed_path)
    receipt = read_json(receipt_path)
    requested = arguments.get("requested_raw_steps", arguments)
    expected = protocol["methods"][method]
    if requested.get("scenario") != scenario:
        raise ValueError(f"parent scenario drifted for {method}/{scenario}")
    if requested.get("algo") != expected["algorithm"]:
        raise ValueError(f"parent algorithm drifted for {method}/{scenario}")
    if int(requested.get("seed", -1)) != 0:
        raise ValueError(f"parent training seed is not 0 for {method}/{scenario}")
    if detailed.get("scenario") != scenario or detailed.get("algorithm") != expected["algorithm"]:
        raise ValueError(f"parent evaluation metadata drifted for {method}/{scenario}")
    if receipt.get("status") != "completed" or receipt.get("fabricated_values") is not False:
        raise ValueError(f"parent receipt is not a completed non-fabricated run for {method}/{scenario}")
    parent_protocol_sha = str(parent.get("protocol_sha256", "")).lower()
    if str(receipt.get("protocol_sha256", "")).lower() != parent_protocol_sha:
        raise ValueError(f"parent receipt protocol hash drifted for {method}/{scenario}")

    recorded_model = detailed.get("model")
    if not isinstance(recorded_model, str) or not recorded_model:
        raise ValueError(f"model is not recorded for {method}/{scenario}")
    model_name = Path(recorded_model).name
    model_path = run_dir / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"missing deployment model: {model_path}")
    if method == "v4_8" and model_name != "selected_model.zip":
        raise ValueError("v4.8 deployment model must be selected_model.zip")
    model_hash = sha256(model_path)
    recorded_hash = detailed.get("model_sha256")
    if recorded_hash is not None and str(recorded_hash).lower() != model_hash:
        raise ValueError(f"deployment model hash drifted for {method}/{scenario}")

    monitor_path = run_dir / "train_monitor.csv"
    diagnostics_path = run_dir / "training_diagnostics.json"
    if not monitor_path.is_file():
        raise FileNotFoundError(f"missing training monitor: {monitor_path}")
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"missing training diagnostics: {diagnostics_path}")
    diagnostics = read_json(diagnostics_path)
    event_paths = list(run_dir.glob("tensorboard/**/events.out.tfevents.*"))
    method_label = str(expected.get("display_label", METHOD_FALLBACK_LABELS.get(method, method)))
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    rows, monitor_audit = load_monitor_rows(
        monitor_path,
        method=method,
        method_label=method_label,
        scenario=scenario,
        scenario_label=scenario_label,
        run_id=run_id,
        model_path=model_path,
        model_sha256=model_hash,
        rolling_window=rolling_window,
    )
    source_audit = {
        "method": method,
        "method_label": method_label,
        "scenario": scenario,
        "scenario_label": scenario_label,
        "run_id": run_id,
        "run_directory": str(run_dir.resolve()),
        "training_seed": int(requested.get("seed", 0)),
        "algorithm": expected["algorithm"],
        "adapter": expected.get("adapter"),
        "model_basename": model_name,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_hash,
        "arguments_path": str(arguments_path.resolve()),
        "arguments_sha256": sha256(arguments_path),
        "parent_evaluation_path": str(detailed_path.resolve()),
        "parent_evaluation_sha256": sha256(detailed_path),
        "parent_receipt_path": str(receipt_path.resolve()),
        "parent_receipt_sha256": sha256(receipt_path),
        "training_diagnostics_path": str(diagnostics_path.resolve()),
        "training_diagnostics_sha256": sha256(diagnostics_path),
        "diagnostics_algorithm": diagnostics.get("algorithm"),
        "diagnostics_scenario": diagnostics.get("scenario"),
        "diagnostics_raw_steps": diagnostics.get("raw_steps"),
        "diagnostics_learner_updates": diagnostics.get("learner_updates"),
        "monitor": monitor_audit,
        "tensorboard": audit_tensorboard(event_paths),
    }
    return rows, source_audit


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty training-curve data table")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scenario(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    scenario_label: str,
    output_dir: Path,
    rolling_window: int,
) -> dict[str, str]:
    by_method = {
        method: sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["episode_index"]),
        )
        for method in METHOD_FALLBACK_LABELS
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.2, 8.1),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.78, top=0.83, bottom=0.12)
    ax_return, ax_success = axes
    for method in METHOD_FALLBACK_LABELS:
        data = by_method[method]
        if not data:
            continue
        style = METHOD_STYLES[method]
        x = [float(row["cumulative_raw_simulation_steps"]) / 1000.0 for row in data]
        raw_return = [float(row["return"]) for row in data]
        mean_return = [float(row["return_trailing_mean_20"]) for row in data]
        raw_success = [int(row["is_success"]) for row in data]
        mean_success = [float(row["success_trailing_mean_20"]) for row in data]
        label = METHOD_LEGEND_LABELS.get(method, str(data[0]["method_label"]))
        # Raw traces are intentionally faint; the rolling mean is the primary
        # visual signal and uses the same method encoding.
        ax_return.plot(x, raw_return, color=style["color"], alpha=0.16, linewidth=0.65)
        ax_return.plot(
            x,
            mean_return,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.25,
            label=label,
        )
        ax_success.plot(x, raw_success, color=style["color"], alpha=0.08, linewidth=0.6)
        ax_success.plot(
            x,
            mean_success,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.25,
            label=label,
        )

    ax_return.set_title(
        f"{scenario_label} — saved parent-v2 model training dynamics",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax_return.text(
        0.0,
        1.02,
        f"training seed=0 · raw monitor trace + {rolling_window}-episode trailing mean · current evaluation did not retrain",
        transform=ax_return.transAxes,
        fontsize=9.6,
        color="#444444",
        va="bottom",
    )
    ax_return.set_ylabel("Episodic return")
    ax_success.set_ylabel("Success rate")
    ax_success.set_xlabel("Cumulative raw simulation steps (×10³)")
    ax_success.set_ylim(-0.03, 1.03)
    ax_success.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_return.grid(axis="both", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax_success.grid(axis="both", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax_return.spines["top"].set_visible(False)
    ax_return.spines["right"].set_visible(False)
    ax_success.spines["top"].set_visible(False)
    ax_success.spines["right"].set_visible(False)
    ax_return.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=10,
        title="Method",
        title_fontsize=10,
    )
    # One compact source note makes the provenance visible without putting
    # nine long Windows paths into the plotting area.
    fig.text(
        0.09,
        0.035,
        "Source: results_hd_ss100_v2/comparison/runs/*/train_monitor.csv; exact paths and SHA-256 values are in training_curve_manifest.json.",
        fontsize=8.4,
        color="#555555",
        ha="left",
    )
    fig.text(
        0.09,
        0.065,
        "Shaded/faint lines show each logged episode; solid/dashed/dash-dot lines show the trailing mean. No values are imputed.",
        fontsize=8.4,
        color="#555555",
        ha="left",
    )
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("#FCFCFC")

    png_path = output_dir / f"{scenario}_training_curves.png"
    svg_path = output_dir / f"{scenario}_training_curves.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return {"png": str(png_path.resolve()), "svg": str(svg_path.resolve())}


def plot_overview(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    rolling_window: int,
) -> dict[str, str]:
    """Create a compact three-scenario overview (rows=scenarios, cols=metrics)."""

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.0, 12.0),
        sharex="col",
        gridspec_kw={"hspace": 0.22, "wspace": 0.18},
    )
    fig.subplots_adjust(left=0.085, right=0.84, top=0.91, bottom=0.08)
    for row_index, scenario in enumerate(SCENARIO_LABELS):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for column, metric in enumerate(("return", "success")):
            axis = axes[row_index, column]
            for method in METHOD_FALLBACK_LABELS:
                data = sorted(
                    [item for item in scenario_rows if item["method"] == method],
                    key=lambda item: int(item["episode_index"]),
                )
                if not data:
                    continue
                style = METHOD_STYLES[method]
                x = [float(item["cumulative_raw_simulation_steps"]) / 1000.0 for item in data]
                if metric == "return":
                    y = [float(item["return_trailing_mean_20"]) for item in data]
                else:
                    y = [float(item["success_trailing_mean_20"]) for item in data]
                axis.plot(
                    x,
                    y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.65,
                    label=METHOD_LEGEND_LABELS.get(method, str(data[0]["method_label"])),
                )
            axis.grid(color="#DDDDDD", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column == 0:
                axis.set_ylabel(f"{SCENARIO_LABELS[scenario]}\nEpisodic return")
            else:
                axis.set_ylabel("Success rate")
                axis.set_ylim(-0.03, 1.03)
                axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if row_index == 0:
                axis.set_title("Trailing return" if metric == "return" else "Trailing success rate", fontsize=12)
            if row_index == 2:
                axis.set_xlabel("Cumulative raw simulation steps (×10³)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.86, 0.90),
        frameon=False,
        fontsize=9.2,
        title="Method",
        title_fontsize=9.5,
    )
    fig.suptitle(
        "Saved parent-v2 model training curves — 3 methods × 3 scenarios",
        x=0.085,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.028,
        f"Training seed=0 · {rolling_window}-episode trailing means · source paths and hashes in training_curve_manifest.json · no retraining in current evaluation",
        fontsize=8.5,
        color="#555555",
        ha="left",
    )
    png_path = output_dir / "all_scenarios_training_curves_overview.png"
    svg_path = output_dir / "all_scenarios_training_curves_overview.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return {"png": str(png_path.resolve()), "svg": str(svg_path.resolve())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for figures, data, and manifest",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="trailing episode window for return/success curves (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive")
    protocol = read_json(PROTOCOL_PATH)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_hash = sha256(PROTOCOL_PATH)
    methods = list(protocol["matrix"]["method_order"])
    scenarios = list(protocol["matrix"]["scenario_order"])
    if methods != list(METHOD_FALLBACK_LABELS):
        # Keep the plot faithful to the protocol's ordering while still
        # validating that the expected three-method matrix is being used.
        METHOD_FALLBACK_LABELS.update({method: method for method in methods if method not in METHOD_FALLBACK_LABELS})
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for method in methods:
        for scenario in scenarios:
            rows, source = resolve_source(protocol, method, scenario, args.rolling_window)
            all_rows.extend(rows)
            sources.append(source)

    data_path = output_dir / "training_curve_data.csv"
    write_long_csv(data_path, all_rows)
    figures: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_rows = [row for row in all_rows if row["scenario"] == scenario]
        figures[scenario] = plot_scenario(
            scenario_rows,
            scenario=scenario,
            scenario_label=SCENARIO_LABELS.get(scenario, scenario),
            output_dir=output_dir,
            rolling_window=args.rolling_window,
        )
    figures["overview"] = plot_overview(all_rows, output_dir=output_dir, rolling_window=args.rolling_window)

    manifest = {
        "artifact_id": "iv2e3m100x100_training_curves_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": protocol_hash,
        "parent_v2": protocol.get("parent_v2"),
        "matrix": {
            "methods": methods,
            "scenarios": scenarios,
            "method_count": len(methods),
            "scenario_count": len(scenarios),
            "source_run_count": len(sources),
            "training_seed": 0,
        },
        "training_is_forbidden_in_current_protocol": bool(protocol.get("training_is_forbidden", False)),
        "source_audit": sources,
        "curve_definition": {
            "primary_log": "train_monitor.csv",
            "x_field": "cumulative_raw_simulation_steps",
            "x_unit": "raw simulation steps",
            "return_field": "r",
            "success_field": "is_success",
            "rolling_window_episodes": args.rolling_window,
            "rolling_prefix_behavior": "mean of all available prefix episodes before the window is full",
            "raw_trace_behavior": "raw per-episode trace drawn faintly; trailing mean is the primary line",
            "no_imputation": True,
        },
        "data_table": {
            "path": str(data_path.resolve()),
            "sha256": sha256(data_path),
            "row_count": len(all_rows),
        },
        "figures": figures,
        "qa": {
            "expected_source_count": 9,
            "observed_source_count": len(sources),
            "all_monitor_rows_nonempty": all(source["monitor"]["episode_rows"] > 0 for source in sources),
            "all_tensorboard_required_tags_present": all(source["tensorboard"]["required_tags_present"] for source in sources),
            "all_models_exist_and_hashed": all(Path(source["model_path"]).is_file() and len(source["model_sha256"]) == 64 for source in sources),
            "all_logs_exist_and_hashed": all(Path(source["monitor"]["monitor_path"]).is_file() and len(source["monitor"]["monitor_sha256"]) == 64 for source in sources),
        },
        "notes": [
            "These are training logs from the already saved parent-v2 models, all training seed 0.",
            "The independent-v2 100-seed×100-episode evaluation reused these models and did not retrain them.",
            "Training curves are descriptive single-seed evidence and are not uncertainty bands over the 100 logical test seeds.",
        ],
    }
    manifest_path = output_dir / "training_curve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "figure_paths": figures,
        "source_count": len(sources),
        "data_row_count": len(all_rows),
        "qa": manifest["qa"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
