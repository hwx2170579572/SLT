"""Read-only evidence extraction for v4.8 lr_half monitor curves.

This helper imports the existing reference transform and writes only derived
evidence artifacts under r48s1/research_evidence. It never loads a model or
starts an environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "r48s1" / "research_evidence"
REF = ROOT / "tools" / "report_phase2_curves_reference_style_v2.py"
NS = runpy.run_path(str(REF), run_name="reference_transform_only")
TRANSFORM = NS["transform"]


SOURCES = {
    "full_cross": ROOT / "results_phase2_runtime_v2" / "screen" / "v4_8__cross__lr_half__seed0" / "train_monitor.csv",
    "minus_horizon_cross": ROOT / "results_subtractive_ablation" / "screen" / "minus_horizon__cross__lr_half__seed0" / "train_monitor.csv",
    "full_carla": ROOT / "results_phase2_runtime_v2" / "screen" / "v4_8__carla__lr_half__seed0" / "train_monitor.csv",
    "minus_horizon_carla": ROOT / "results_subtractive_ablation" / "screen" / "minus_horizon__carla__lr_half__seed0" / "train_monitor.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(line for line in f if not line.startswith("#")))
    rows = [
        {
            "r": float(x["r"]),
            "is_success": x["is_success"].lower() in ("true", "1"),
            "raw_simulation_steps": int(x["raw_simulation_steps"]),
            "l": int(x["l"]),
            "collision": x["collision"].lower() in ("true", "1"),
            "timeout": x["max_time"].lower() in ("true", "1"),
        }
        for x in raw
    ]
    return rows


def nearest(a, x):
    i = min(range(len(a)), key=lambda j: abs(float(a[j, 0]) - x))
    return {
        "raw_step": int(a[i, 0]),
        "value": float(a[i, 1]),
        "lower": float(a[i, 2]),
        "upper": float(a[i, 3]),
    }


def main():
    checkpoints = [10000, 20000, 30000, 40000]
    curves = {}
    rows_by_key = {}
    summary = {}
    csv_rows = []
    for key, path in SOURCES.items():
        rows = load_rows(path)
        rows_by_key[key] = rows
        total = sum(x["raw_simulation_steps"] for x in rows)
        curves[key] = {m: TRANSFORM(rows, m) for m in ("success", "reward")}
        summary[key] = {
            "source": str(path),
            "source_sha256": sha256(path),
            "episodes_logged": len(rows),
            "total_logged_raw_steps": total,
            "last_episode_raw_steps": rows[-1]["raw_simulation_steps"],
            "last_episode_length_decision_steps": rows[-1]["l"],
        }
        for metric, array in curves[key].items():
            for target in checkpoints + [total]:
                point = nearest(array, target)
                csv_rows.append({
                    "condition": key,
                    "metric": metric,
                    "requested_raw_step": target,
                    **point,
                })

    # Last-20 raw episode statistics (before EMA), plus cumulative episode count
    # at each requested raw-step boundary.
    for key, rows in rows_by_key.items():
        cumulative = 0
        boundary = {}
        for idx, row in enumerate(rows, start=1):
            cumulative += row["raw_simulation_steps"]
            for target in checkpoints:
                if target not in boundary and cumulative >= target:
                    boundary[target] = {
                        "episode_count": idx,
                        "episode_end_raw_step": cumulative,
                    }
        tail = rows[-20:]
        summary[key]["tail20"] = {
            "episode_count": len(tail),
            "success_count": sum(int(x["is_success"]) for x in tail),
            "success_rate_raw": sum(int(x["is_success"]) for x in tail) / len(tail),
            "collision_count": sum(int(x["collision"]) for x in tail),
            "timeout_count": sum(int(x["timeout"]) for x in tail),
            "mean_decision_steps": sum(x["l"] for x in tail) / len(tail),
            "mean_raw_steps": sum(x["raw_simulation_steps"] for x in tail) / len(tail),
            "raw_steps_span_start": sum(x["raw_simulation_steps"] for x in rows[:-20]),
            "raw_steps_span_end": sum(x["raw_simulation_steps"] for x in rows),
        }
        summary[key]["episode_boundaries"] = boundary

    with (OUT / "monitor_curve_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (OUT / "monitor_curve_summary.json").write_text(
        json.dumps({"reference_transform": str(REF), "reference_transform_sha256": sha256(REF), "conditions": summary}, indent=2),
        encoding="utf-8",
    )

    # Two scene panels, with raw monitor conditions only; no cross-parameter averaging.
    for scene, keys in {
        "cross": ("full_cross", "minus_horizon_cross"),
        "carla": ("full_carla", "minus_horizon_carla"),
    }.items():
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for key, label, color in ((keys[0], "full v4.8", "#0072B2"), (keys[1], "minus horizon", "#D55E00")):
            for ax, metric, ylabel in zip(axes, ("reward", "success"), ("EMA999 reward", "EMA999 success")):
                a = curves[key][metric]
                ax.plot(a[:, 0], a[:, 1], label=label, color=color, linewidth=2.0)
                ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=color, alpha=0.10)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
        axes[0].legend(loc="best")
        axes[1].set_xlabel("Raw training steps logged")
        fig.suptitle(f"{scene}: v4.8 lr_half monitor, reference episode20 + per-raw-step EMA999")
        fig.tight_layout()
        fig.savefig(OUT / f"monitor_{scene}_full_vs_minus_horizon.png", dpi=180)
        fig.savefig(OUT / f"monitor_{scene}_full_vs_minus_horizon.svg")
        plt.close(fig)


if __name__ == "__main__":
    main()
