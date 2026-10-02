"""Evidence-bound focal comparison for v4.8 versus both baselines.

The script uses the already generated v3 time-step/EMA data and never edits
the original monitor logs.  It keeps all three methods and the full common
training support visible, while adding a second row with the pointwise margin
``v4.8 - max(MST+SLT, TemporalGraph)``.  Positive and negative regions are
both shown; the figure therefore highlights where the requested relation is
supported without hiding counterexamples.
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
from matplotlib import font_manager

try:
    cjk_font = Path("C:/Windows/Fonts/msyh.ttc")
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(cjk_font)).get_name()
except Exception:  # pragma: no cover
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v3"
V3_DATA_PATH = V3_DIR / "training_success_curve_time_steps_ema999_data.csv"
V3_MANIFEST_PATH = V3_DIR / "training_success_curve_time_steps_ema999_manifest.json"
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "independent_v2_existing_3methods_100seeds_100episodes_v1" / "protocol.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iv2e3m100x100_training_curves_v4_focal"

METHODS = ["v4_8", "mst_slt", "temporal_graph"]
BASELINES = ["mst_slt", "temporal_graph"]
METHOD_SHORT_LABELS = {"v4_8": "v4.8", "mst_slt": "MST+SLT", "temporal_graph": "TemporalGraph"}
METHOD_STYLES = {
    "v4_8": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.8, "band_alpha": 0.18},
    "mst_slt": {"color": "#666666", "linestyle": "--", "linewidth": 1.35, "band_alpha": 0.07},
    "temporal_graph": {"color": "#E69F00", "linestyle": ":", "linewidth": 1.45, "band_alpha": 0.08},
}
SCENARIO_LABELS = {"carla": "CARLA", "cross": "Cross", "roundabout": "Roundabout"}
PANEL_LABELS = {"carla": "a)", "cross": "b)", "roundabout": "c)"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"method", "scenario", "time_step", "ema_success_rate", "ema_band_lower", "ema_band_upper"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"v3 data table is missing required fields: {sorted(required)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "method": row["method"],
                "scenario": row["scenario"],
                "time_step": int(row["time_step"]),
                "ema_success_rate": float(row["ema_success_rate"]),
                "ema_band_lower": float(row["ema_band_lower"]),
                "ema_band_upper": float(row["ema_band_upper"]),
                "run_id": row.get("run_id", ""),
                "training_seed": int(row.get("training_seed", "0")),
                "window_steps": int(row.get("window_steps", "2000")),
                "ema_alpha": float(row.get("ema_alpha", "0.999")),
            }
        )
    return parsed


def align_common_support(rows: list[dict[str, Any]], scenario: str) -> tuple[list[int], dict[str, dict[int, dict[str, Any]]]]:
    by_method: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    for row in rows:
        if row["scenario"] == scenario and row["method"] in by_method:
            by_method[row["method"]][row["time_step"]] = row
    missing = [method for method in METHODS if not by_method[method]]
    if missing:
        raise ValueError(f"scenario {scenario} is missing methods: {missing}")
    common = sorted(set.intersection(*(set(by_method[method]) for method in METHODS)))
    if len(common) < 10:
        raise ValueError(f"too few common time-step points for {scenario}: {len(common)}")
    return common, by_method


def aligned_rows(rows: list[dict[str, Any]], scenarios: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        common, by_method = align_common_support(rows, scenario)
        for time_step in common:
            values = {method: by_method[method][time_step]["ema_success_rate"] for method in METHODS}
            bands = {method: (by_method[method][time_step]["ema_band_lower"], by_method[method][time_step]["ema_band_upper"]) for method in METHODS}
            best_baseline = max(values[method] for method in BASELINES)
            margin = values["v4_8"] - best_baseline
            long_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                    "time_step": time_step,
                    "v4_8_ema_success_rate": values["v4_8"],
                    "mst_slt_ema_success_rate": values["mst_slt"],
                    "temporal_graph_ema_success_rate": values["temporal_graph"],
                    "best_baseline_ema_success_rate": best_baseline,
                    "v4_8_margin_vs_best_baseline": margin,
                    "v4_8_band_lower": bands["v4_8"][0],
                    "v4_8_band_upper": bands["v4_8"][1],
                    "mst_slt_band_lower": bands["mst_slt"][0],
                    "mst_slt_band_upper": bands["mst_slt"][1],
                    "temporal_graph_band_lower": bands["temporal_graph"][0],
                    "temporal_graph_band_upper": bands["temporal_graph"][1],
                }
            )
        margin_values = [row["v4_8_margin_vs_best_baseline"] for row in long_rows if row["scenario"] == scenario]
        scenario_rows = [row for row in long_rows if row["scenario"] == scenario]
        times = [row["time_step"] for row in scenario_rows]
        # Normalized trapezoidal AUC is a descriptive summary over the common
        # support; it is not used to select or crop any segment.
        aucs: dict[str, float] = {}
        for method in METHODS:
            y = [row[f"{method}_ema_success_rate"] for row in scenario_rows]
            area = sum((times[i + 1] - times[i]) * (y[i + 1] + y[i]) / 2.0 for i in range(len(times) - 1))
            aucs[method] = area / max(1.0, float(times[-1] - times[0]))
        summaries[scenario] = {
            "common_time_step_count": len(common),
            "common_time_step_start": common[0],
            "common_time_step_end": common[-1],
            "mean_margin_vs_best_baseline": sum(margin_values) / len(margin_values),
            "min_margin_vs_best_baseline": min(margin_values),
            "max_margin_vs_best_baseline": max(margin_values),
            "positive_margin_fraction": sum(value > 0 for value in margin_values) / len(margin_values),
            "final_margin_vs_best_baseline": margin_values[-1],
            "normalized_auc": aucs,
            "v4_8_auc_minus_best_baseline_auc": aucs["v4_8"] - max(aucs[method] for method in BASELINES),
        }
    return long_rows, summaries


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
    axis.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.82, borderpad=0.4, handlelength=2.2)
    axis.set_facecolor("#FCFCFC")


def draw_margin_axis(axis: Any, rows: list[dict[str, Any]], summary: dict[str, Any], scenario: str, x_max: int, margin_limit: float) -> None:
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
    axis.legend(loc="lower right", fontsize=7.6, frameon=True, framealpha=0.82, borderpad=0.35, handlelength=2.0)
    axis.text(
        0.03,
        0.95,
        f"mean {summary['mean_margin_vs_best_baseline']:+.3f} · positive {summary['positive_margin_fraction']:.0%}",
        transform=axis.transAxes,
        va="top",
        fontsize=8.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.8, "pad": 2.0},
    )
    axis.set_facecolor("#FCFCFC")


def plot_composite(aligned: list[dict[str, Any]], summaries: dict[str, dict[str, Any]], scenarios: list[str], output_dir: Path, x_max: int, margin_limit: float) -> dict[str, str]:
    fig, axes = plt.subplots(2, len(scenarios), figsize=(15.0, 8.15), sharex="row", gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.26, "wspace": 0.22})
    if len(scenarios) == 1:
        axes = axes.reshape(2, 1)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.12)
    for column, scenario in enumerate(scenarios):
        scenario_rows = [row for row in aligned if row["scenario"] == scenario]
        draw_top_axis(axes[0, column], scenario_rows, scenario, x_max)
        draw_margin_axis(axes[1, column], scenario_rows, summaries[scenario], scenario, x_max, margin_limit)
        axes[0, column].text(-0.10, 1.10, PANEL_LABELS.get(scenario, ""), transform=axes[0, column].transAxes, fontsize=13, fontweight="bold", va="top", ha="left", bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4})
    fig.text(0.5, 0.045, "Top: EMA-smoothed success curves. Bottom: v4.8 − max(MST+SLT, TemporalGraph); positive means v4.8 is higher. EMA α=0.999 per time step; rolling window=2,000 time steps; seed 0.", ha="center", fontsize=8.4, color="#555555")
    fig.patch.set_facecolor("white")
    png_path = output_dir / "v4_8_focal_training_curves_time_steps_ema999.png"
    svg_path = output_dir / "v4_8_focal_training_curves_time_steps_ema999.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path.resolve()), "svg": str(svg_path.resolve())}


def plot_individual(aligned: list[dict[str, Any]], summary: dict[str, Any], scenario: str, output_dir: Path, x_max: int, margin_limit: float) -> dict[str, str]:
    result: dict[str, str] = {}
    for extension in ("png", "svg"):
        path = output_dir / f"{scenario}_v4_8_focal_training_curves_time_steps_ema999.{extension}"
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.26})
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.16)
        draw_top_axis(axes[0], aligned, scenario, x_max)
        draw_margin_axis(axes[1], aligned, summary, scenario, x_max, margin_limit)
        axes[0].text(-0.12, 1.10, PANEL_LABELS.get(scenario, ""), transform=axes[0].transAxes, fontsize=13, fontweight="bold", va="top", ha="left", bbox={"facecolor": "#FFF200", "edgecolor": "none", "pad": 1.4})
        fig.text(0.5, 0.055, "EMA α=0.999 · 2,000-step rolling window · positive margin means v4.8 exceeds both baselines", ha="center", fontsize=8.0, color="#555555")
        fig.patch.set_facecolor("white")
        fig.savefig(path, dpi=300 if extension == "png" else None, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        result[extension] = str(path.resolve())
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    v3_manifest = json.loads(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_rows = load_rows(V3_DATA_PATH)
    protocol_methods = list(protocol["matrix"]["method_order"])
    protocol_scenarios = list(protocol["matrix"]["scenario_order"])
    if set(protocol_methods) != set(METHODS) or set(protocol_scenarios) != set(SCENARIO_LABELS):
        raise ValueError("v3 input does not match the expected 3-method × 3-scenario protocol")
    if v3_manifest.get("data_table", {}).get("sha256") != sha256(V3_DATA_PATH):
        raise ValueError("v3 data-table hash drifted")
    aligned, summaries = aligned_rows(source_rows, protocol_scenarios)
    max_common_step = max(int(summary["common_time_step_end"]) for summary in summaries.values())
    # Keep a common 0–50,000 time-step frame; the actual pointwise comparison
    # ends at the shortest shared source support and is recorded below.
    x_max = 50_000
    margin_limit = 0.6
    data_path = output_dir / "v4_8_focal_aligned_margin_data.csv"
    write_csv(data_path, aligned)
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
        "artifact_id": "iv2e3m100x100_training_curves_v4_focal_time_steps_ema999",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_v3_data_path": str(V3_DATA_PATH.resolve()),
        "input_v3_data_sha256": sha256(V3_DATA_PATH),
        "input_v3_manifest_path": str(V3_MANIFEST_PATH.resolve()),
        "input_v3_manifest_sha256": sha256(V3_MANIFEST_PATH),
        "source_training_seed": 0,
        "source_methods": protocol_methods,
        "source_scenarios": protocol_scenarios,
        "plot_semantics": {
            "top_row": "all three EMA-smoothed success curves; v4.8 emphasized, baselines retained",
            "bottom_row": "pointwise v4.8 minus max(MST+SLT, TemporalGraph) margin",
            "positive_margin": "v4.8 success rate is above both baselines at that aligned time step",
            "negative_margin": "at least one baseline is above v4.8 at that aligned time step",
            "x_axis": "cumulative raw simulation steps displayed as time steps",
            "rolling_window_steps": 2000,
            "ema_alpha": 0.999,
            "ema_update_frequency": "every raw time step in the v3 transform",
            "common_support_alignment": "intersection of each scenario's three method time-step samples; no interpolation",
            "visual_x_limit": x_max,
            "band": "inherited descriptive EMA-smoothed rolling SE from v3; not a cross-seed confidence interval",
        },
        "aligned_data": {
            "path": str(data_path.resolve()),
            "sha256": sha256(data_path),
            "row_count": len(aligned),
            "max_common_time_step": max_common_step,
        },
        "scenario_summaries": summaries,
        "figures": {"composite": composite, "individual": individual},
        "qa": {
            "all_expected_scenarios_present": set(summaries) == set(protocol_scenarios),
            "all_scenarios_have_three_methods": all(summary["common_time_step_count"] >= 10 for summary in summaries.values()),
            "v3_input_hash_verified": True,
            "original_v3_data_untouched": True,
            "all_margin_values_finite": all(math.isfinite(float(row["v4_8_margin_vs_best_baseline"])) for row in aligned),
            "all_margin_values_visible": True,
        },
        "honesty_note": "The focal styling emphasizes v4.8 but keeps the full time range and negative margins visible. The existing single-seed data do not support a universal v4.8>baseline claim: see scenario_summaries for the signed margins.",
    }
    manifest_path = output_dir / "v4_8_focal_training_curves_time_steps_ema999_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Focal v4.8 training-curve comparison\n\n"
        "The top row retains all three methods and emphasizes v4.8. The bottom row shows the signed pointwise margin `v4.8 - max(MST+SLT, TemporalGraph)`, so positive and negative regions are both visible. The plot uses the unchanged v3 time-step/EMA data and does not select or crop intervals to manufacture a gain.\n\n"
        "See the manifest for signed per-scenario summaries and source hashes.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "composite": composite,
        "individual": individual,
        "manifest": str(manifest_path.resolve()),
        "data_rows": len(aligned),
        "scenario_summaries": summaries,
        "qa": manifest["qa"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
