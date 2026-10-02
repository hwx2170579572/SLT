"""Reference-style success-rate curves for the nine saved parent-v2 models.

The v2 presentation follows the supplied visual reference: one small multiple
per scenario, a mean success-rate line, and a translucent variability band.
The values are still read from the real parent-v2 ``train_monitor.csv`` logs;
the band is explicitly a within-seed rolling standard deviation (20 episodes),
not a cross-seed confidence interval.  No model is trained by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
import statistics
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the v1 loader/validator so the source binding and SHA checks remain
# identical.  ``run_path`` imports definitions only; v1's main() is guarded.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = PROJECT_ROOT / "visual-composer" / "plot_training_curves_v1.py"
BASE = runpy.run_path(str(V1_SCRIPT))
read_json = BASE["read_json"]
sha256 = BASE["sha256"]
resolve_source = BASE["resolve_source"]
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "independent_v2_existing_3methods_100seeds_100episodes_v1" / "protocol.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    # Keep CJK method labels available when a Windows host provides the font.
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
    # v4.8 is the proposed method and keeps a stable blue accent.
    "mst_slt": {"color": "#E69F00", "linestyle": "-"},
    "temporal_graph": {"color": "#009E73", "linestyle": "--"},
    "v4_8": {"color": "#0072B2", "linestyle": "-."},
}


def add_rolling_band(rows: list[dict[str, Any]], window: int) -> None:
    """Add a descriptive mean±SD/√n band from logged binary successes.

    The standard error keeps a Bernoulli rolling band readable when there is
    only one training seed.  It is descriptive smoothing, not a formal CI.
    """

    values: deque[float] = deque(maxlen=window)
    for row in rows:
        values.append(float(row["is_success"]))
        mean = float(sum(values) / len(values))
        std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        standard_error = std / math.sqrt(len(values))
        row["success_rolling_std_20"] = std
        row["success_rolling_se_20"] = standard_error
        row["success_band_lower"] = max(0.0, mean - standard_error)
        row["success_band_upper"] = min(1.0, mean + standard_error)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    output_path: Path,
    panel_label: str | None,
    rolling_window: int,
    width: float,
    height: float,
    x_max: float,
) -> None:
    methods = ["mst_slt", "temporal_graph", "v4_8"]
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.18)
    for method in methods:
        data = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["episode_index"]))
        if not data:
            continue
        style = METHOD_STYLES[method]
        x = [float(row["cumulative_raw_simulation_steps"]) / 1000.0 for row in data]
        mean = [float(row["success_trailing_mean_20"]) for row in data]
        lower = [float(row["success_band_lower"]) for row in data]
        upper = [float(row["success_band_upper"]) for row in data]
        ax.fill_between(x, lower, upper, color=style["color"], alpha=0.22, linewidth=0, zorder=1)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.9,
            label=METHOD_SHORT_LABELS[method],
            zorder=2,
        )
    ax.set_title(f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=10.5, pad=7)
    if panel_label:
        ax.text(
            -0.13,
            1.10,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4},
        )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(range(0, int(x_max) + 1, 10)))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Step", fontsize=9.5)
    ax.set_ylabel("Avg Success rate", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, color="#BFBFBF", linewidth=0.65, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.3, frameon=True, framealpha=0.82, borderpad=0.45, handlelength=2.2)
    fig.text(
        0.5,
        0.035,
        f"Mean: {rolling_window}-episode trailing success rate · band: ±1 descriptive rolling SE (SD/√n), seed 0 · saved parent-v2 logs; no retraining",
        ha="center",
        fontsize=7.4,
        color="#555555",
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_composite(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    output_dir: Path,
    rolling_window: int,
    x_max: float,
) -> dict[str, str]:
    methods = ["mst_slt", "temporal_graph", "v4_8"]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(15.0, 4.55), sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.18, wspace=0.22)
    for axis, scenario in zip(axes, scenarios):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for method in methods:
            data = sorted((row for row in scenario_rows if row["method"] == method), key=lambda row: int(row["episode_index"]))
            if not data:
                continue
            style = METHOD_STYLES[method]
            x = [float(row["cumulative_raw_simulation_steps"]) / 1000.0 for row in data]
            mean = [float(row["success_trailing_mean_20"]) for row in data]
            lower = [float(row["success_band_lower"]) for row in data]
            upper = [float(row["success_band_upper"]) for row in data]
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.22, linewidth=0, zorder=1)
            axis.plot(
                x,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=METHOD_SHORT_LABELS[method],
                zorder=2,
            )
        axis.set_title(f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=10.2, pad=7)
        axis.text(
            -0.10,
            1.10,
            SCENARIO_PANEL_LABELS.get(scenario, ""),
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4},
        )
        axis.set_xlim(0, x_max)
        axis.set_ylim(0, 1.0)
        axis.set_xticks(list(range(0, int(x_max) + 1, 10)))
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_xlabel("Step", fontsize=9.2)
        axis.set_ylabel("Avg Success rate", fontsize=9.2)
        axis.tick_params(labelsize=8.1)
        axis.grid(True, color="#BFBFBF", linewidth=0.65, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="upper left", fontsize=7.9, frameon=True, framealpha=0.82, borderpad=0.4, handlelength=2.0)
        axis.set_facecolor("#FCFCFC")
    fig.text(
        0.5,
        0.045,
        f"Mean: {rolling_window}-episode trailing success rate · band: ±1 descriptive rolling SE (SD/√n), seed 0 · saved parent-v2 logs; no retraining",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.patch.set_facecolor("white")
    png_path = output_dir / "success_curves_by_scenario.png"
    svg_path = output_dir / "success_curves_by_scenario.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path.resolve()), "svg": str(svg_path.resolve())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive")
    protocol = read_json(PROTOCOL_PATH)
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(protocol["matrix"]["method_order"])
    scenarios = list(protocol["matrix"]["scenario_order"])
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for method in methods:
        for scenario in scenarios:
            source_rows, source_audit = resolve_source(protocol, method, scenario, args.rolling_window)
            add_rolling_band(source_rows, args.rolling_window)
            rows.extend(source_rows)
            sources.append(source_audit)

    data_path = output_dir / "training_success_curve_data.csv"
    write_csv(data_path, rows)
    max_x = max(float(row["cumulative_raw_simulation_steps"]) / 1000.0 for row in rows)
    x_max = float(max(50, int(math.ceil(max_x / 10.0) * 10)))
    composite = plot_composite(rows, scenarios=scenarios, output_dir=output_dir, rolling_window=args.rolling_window, x_max=x_max)
    individual: dict[str, dict[str, str]] = {}
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        png = output_dir / f"{scenario}_success_curve.png"
        svg = output_dir / f"{scenario}_success_curve.svg"
        plot_panel(
            scenario_rows,
            scenario=scenario,
            output_path=png,
            panel_label=SCENARIO_PANEL_LABELS.get(scenario),
            rolling_window=args.rolling_window,
            width=5.1,
            height=4.55,
            x_max=x_max,
        )
        # Render the same panel as editable vector text.
        plot_panel(
            scenario_rows,
            scenario=scenario,
            output_path=svg,
            panel_label=SCENARIO_PANEL_LABELS.get(scenario),
            rolling_window=args.rolling_window,
            width=5.1,
            height=4.55,
            x_max=x_max,
        )
        individual[scenario] = {"png": str(png.resolve()), "svg": str(svg.resolve())}

    protocol_sha = sha256(PROTOCOL_PATH)
    manifest = {
        "artifact_id": "iv2e3m100x100_training_curves_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": protocol_sha,
        "parent_v2": protocol.get("parent_v2"),
        "matrix": {"methods": methods, "scenarios": scenarios, "source_run_count": len(sources), "training_seed": 0},
        "training_is_forbidden_in_current_protocol": bool(protocol.get("training_is_forbidden", False)),
        "curve_definition": {
            "metric": "is_success",
            "y_label": "Avg Success rate",
            "x_field": "cumulative_raw_simulation_steps",
            "x_unit": "raw simulation steps (×10³)",
            "mean": f"{args.rolling_window}-episode trailing mean",
            "band": f"±1 {args.rolling_window}-episode trailing descriptive standard error (population SD/√n) within training seed 0, clipped to [0,1]",
            "cross_seed_uncertainty": False,
            "no_imputation": True,
        },
        "data_table": {"path": str(data_path.resolve()), "sha256": sha256(data_path), "row_count": len(rows)},
        "figures": {"composite": composite, "individual": individual},
        "source_audit": sources,
        "qa": {
            "expected_source_count": 9,
            "observed_source_count": len(sources),
            "all_monitor_rows_nonempty": all(item["monitor"]["episode_rows"] > 0 for item in sources),
            "all_models_exist_and_hashed": all(Path(item["model_path"]).is_file() and len(item["model_sha256"]) == 64 for item in sources),
            "all_logs_exist_and_hashed": all(Path(item["monitor"]["monitor_path"]).is_file() and len(item["monitor"]["monitor_sha256"]) == 64 for item in sources),
            "tensorboard_event_coverage": sum(bool(item["tensorboard"]["event_paths"]) for item in sources),
            "data_row_count": len(rows),
        },
        "notes": [
            "The reference-style band is a descriptive rolling standard-error band (population SD/√n), not a cross-seed confidence interval.",
            "All curves come from saved parent-v2 models; the independent-v2 evaluation did not retrain them.",
            "Some parent directories do not retain TensorBoard event files, so train_monitor.csv is the consistent source for all panels.",
        ],
    }
    manifest_path = output_dir / "training_success_curve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Reference-style saved-model success curves\n\n"
        "The composite `success_curves_by_scenario.png` mirrors the supplied reference layout: one panel per scenario, mean success-rate lines, and translucent bands. The band is a 20-episode descriptive rolling standard error (population SD/√n) within training seed 0; it is not a cross-seed confidence interval.\n\n"
        "Source provenance, model/log hashes, and row-level data are in `training_success_curve_manifest.json` and `training_success_curve_data.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "composite": composite,
        "individual": individual,
        "manifest": str(manifest_path.resolve()),
        "data_rows": len(rows),
        "source_count": len(sources),
        "qa": manifest["qa"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
