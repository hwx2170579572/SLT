"""Render the evidence-bound topology-temporal result dashboard.

The script intentionally refuses incomplete or exploratory sources.  All
plotted values come from verified experiment summaries, explicit-seed
checkpoint re-evaluations, or diagnostics computed on one shared observation
dataset.  It also writes a source-data table and SHA256 receipt next to the
figure so that visual claims remain machine-auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("scene_rep", "temporal_graph", "topo_scene", "topo_scene_balanced")
METHOD_LABEL = {
    "scene_rep": "MST+SLT",
    "temporal_graph": "TemporalGraph",
    "topo_scene": "Full",
    "topo_scene_balanced": "Full+BalancedSlots",
}
METHOD_COLOR = {
    "scene_rep": "#666666",
    "temporal_graph": "#0072B2",
    "topo_scene": "#D55E00",
    "topo_scene_balanced": "#009E73",
}
METHOD_MARKER = {
    "scene_rep": "o",
    "temporal_graph": "^",
    "topo_scene": "s",
    "topo_scene_balanced": "D",
}
METHOD_LINESTYLE = {
    "scene_rep": "-",
    "temporal_graph": "--",
    "topo_scene": "-.",
    "topo_scene_balanced": ":",
}
SLOT_COLOR = {"ego": "#0072B2", "social": "#009E73", "route": "#D55E00"}
SLOT_MARKER = {"ego": "o", "social": "s", "route": "D"}
TIMELINE_PATTERN = re.compile(
    r"^(?P<family>full|topo|topo_scene|balanced|topo_scene_balanced)_"
    r"(?P<step>\d+)(?P<unit>k)?$"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence payload must be a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_summary(payload: dict[str, Any], phase: str) -> None:
    _require(payload.get("schema_version") == "topo-scene.results/v1", f"{phase}: wrong summary contract")
    _require(payload.get("phase") == phase, f"{phase}: phase mismatch")
    _require(payload.get("computed_from_real_runs") is True, f"{phase}: not marked as real runs")
    _require(payload.get("fabricated_values") is False, f"{phase}: fabricated-values flag is not false")
    _require(payload.get("missing_jobs") == [], f"{phase}: summary has missing jobs")
    _require(int(payload.get("complete_jobs", -1)) == int(payload.get("expected_jobs", -2)), f"{phase}: incomplete jobs")


def _one_run(summary: dict[str, Any], method: str) -> dict[str, Any]:
    matches = [row for row in summary["per_run"] if row["method"] == method]
    _require(len(matches) == 1, f"Expected exactly one completed {method} run, found {len(matches)}")
    return matches[0]


def _rate_intervals(summary: dict[str, Any], method: str) -> dict[str, dict[str, float]]:
    matches = [row for row in summary["aggregates"] if row["method"] == method]
    _require(len(matches) == 1, f"Expected exactly one {method} aggregate, found {len(matches)}")
    intervals = matches[0]["bootstrap_95_ci"]
    return {
        metric: {"lower": float(intervals[metric]["lower"]), "upper": float(intervals[metric]["upper"])}
        for metric in ("success_rate", "collision_rate")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _timeline_key(name: str) -> tuple[str, int]:
    match = TIMELINE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Timeline model name {name!r} must be full_10k/topo_10k or balanced_10k style"
        )
    raw_family = match.group("family")
    family = "full" if raw_family in {"full", "topo", "topo_scene"} else "balanced"
    step = int(match.group("step"))
    if match.group("unit") == "k":
        step *= 1000
    return family, step


def _timeline_index(models: dict[str, Any]) -> dict[str, dict[int, str]]:
    index: dict[str, dict[int, str]] = {"full": {}, "balanced": {}}
    for name in models:
        family, step = _timeline_key(name)
        _require(step not in index[family], f"Duplicate {family} checkpoint at raw step {step}")
        index[family][step] = name
    _require(index["full"] and index["balanced"], "Both Full and Balanced checkpoint timelines are required")
    _require(
        set(index["full"]) == set(index["balanced"]),
        "Full and Balanced timeline checkpoint steps must match exactly",
    )
    return index


def _style_axis(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def _build_figure(
    final_rows: dict[str, dict[str, Any]],
    curves: dict[str, dict[str, Any]],
    probes: dict[str, Any],
    actions: dict[str, Any],
    timeline: dict[str, dict[int, str]],
    source_rows: list[dict[str, Any]],
    context_note: str | None = None,
    rate_intervals: dict[str, dict[str, dict[str, float]]] | None = None,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), constrained_layout=True)
    ax_outcome, ax_curve, ax_slots, ax_action = axes.ravel()

    # (a) Closed-loop final outcomes.
    positions = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34
    success = [float(final_rows[name]["success_rate"]) for name in METHOD_ORDER]
    collision = [float(final_rows[name]["collision_rate"]) for name in METHOD_ORDER]
    bars_success = ax_outcome.bar(
        positions - width / 2,
        success,
        width,
        color=[METHOD_COLOR[name] for name in METHOD_ORDER],
        edgecolor="#222222",
        linewidth=0.6,
        label="Success",
    )
    bars_collision = ax_outcome.bar(
        positions + width / 2,
        collision,
        width,
        color="white",
        edgecolor=[METHOD_COLOR[name] for name in METHOD_ORDER],
        linewidth=1.4,
        hatch="///",
        label="Collision",
    )
    ax_outcome.bar_label(bars_success, fmt="%.2f", padding=2, fontsize=7.5)
    ax_outcome.bar_label(bars_collision, fmt="%.2f", padding=2, fontsize=7.5)
    if rate_intervals is not None:
        success_error = np.asarray(
            [
                [success[index] - rate_intervals[method]["success_rate"]["lower"] for index, method in enumerate(METHOD_ORDER)],
                [rate_intervals[method]["success_rate"]["upper"] - success[index] for index, method in enumerate(METHOD_ORDER)],
            ]
        )
        collision_error = np.asarray(
            [
                [collision[index] - rate_intervals[method]["collision_rate"]["lower"] for index, method in enumerate(METHOD_ORDER)],
                [rate_intervals[method]["collision_rate"]["upper"] - collision[index] for index, method in enumerate(METHOD_ORDER)],
            ]
        )
        ax_outcome.errorbar(
            positions - width / 2,
            success,
            yerr=success_error,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=2.5,
        )
        ax_outcome.errorbar(
            positions + width / 2,
            collision,
            yerr=collision_error,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=2.5,
        )
    ax_outcome.set_xticks(positions, [METHOD_LABEL[name] for name in METHOD_ORDER], rotation=12, ha="right")
    ax_outcome.set_ylim(0.0, 1.14)
    ax_outcome.set_ylabel("Episode rate")
    ax_outcome.set_title("(a) Final paired closed-loop outcome")
    ax_outcome.legend(frameon=False, ncols=2, loc="upper center")
    _style_axis(ax_outcome)
    for method in METHOD_ORDER:
        row = final_rows[method]
        for metric in ("success_rate", "collision_rate", "mean_return"):
            source_rows.append(
                {"panel": "a", "method": method, "raw_steps": row["raw_steps"], "metric": metric, "value": row[metric]}
            )
        if rate_intervals is not None:
            for metric in ("success_rate", "collision_rate"):
                source_rows.extend(
                    [
                        {"panel": "a", "method": method, "raw_steps": row["raw_steps"], "metric": f"{metric}_bootstrap95_lower", "value": rate_intervals[method][metric]["lower"]},
                        {"panel": "a", "method": method, "raw_steps": row["raw_steps"], "metric": f"{metric}_bootstrap95_upper", "value": rate_intervals[method][metric]["upper"]},
                    ]
                )

    # (b) Checkpoint success curves from explicit paired seeds only.
    for method in METHOD_ORDER:
        points = sorted(curves[method]["points"], key=lambda row: int(row["raw_steps"]))
        xs = [int(row["raw_steps"]) / 1000 for row in points]
        ys = [float(row["success_rate"]) for row in points]
        ax_curve.plot(
            xs,
            ys,
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            linestyle=METHOD_LINESTYLE[method],
            linewidth=2.0,
            markersize=4.5,
            label=METHOD_LABEL[method],
        )
        for row in points:
            for metric in ("success_rate", "collision_rate", "mean_return"):
                source_rows.append(
                    {"panel": "b", "method": method, "raw_steps": row["raw_steps"], "metric": metric, "value": row[metric]}
                )
    ax_curve.set_ylim(-0.04, 1.04)
    ax_curve.set_xlabel("Raw SUMO steps (thousands)")
    ax_curve.set_ylabel("Success rate")
    ax_curve.set_title("(b) Fixed-seed checkpoint trajectory")
    ax_curve.legend(frameon=False, ncols=2, loc="lower left")
    _style_axis(ax_curve)

    # (c) Slot-scale evolution on one shared observation dataset.
    family_style = {"full": "-", "balanced": "--"}
    family_label = {"full": "Full", "balanced": "Balanced"}
    for family in ("full", "balanced"):
        steps = sorted(timeline[family])
        for slot in ("ego", "social", "route"):
            values = [float(probes["results"][timeline[family][step]][slot]["latent_std_mean"]) for step in steps]
            ax_slots.plot(
                [step / 1000 for step in steps],
                values,
                color=SLOT_COLOR[slot],
                linestyle=family_style[family],
                marker=SLOT_MARKER[slot],
                markerfacecolor=(SLOT_COLOR[slot] if family == "full" else "white"),
                linewidth=1.8,
                markersize=4.0,
                label=f"{family_label[family]} / {slot}",
            )
            for step, value in zip(steps, values):
                source_rows.append(
                    {"panel": "c", "method": family, "raw_steps": step, "metric": f"{slot}_latent_std", "value": value}
                )
    ax_slots.set_xlabel("Raw SUMO steps (thousands)")
    ax_slots.set_ylabel("Latent std. on shared observations")
    ax_slots.set_title("(c) Structured-slot scale evolution")
    ax_slots.legend(frameon=False, ncols=2, loc="best")
    _style_axis(ax_slots)

    # (d) Observational policy utilization, explicitly separate from closed loop.
    action_twin = ax_action.twinx()
    for family, method in (("full", "topo_scene"), ("balanced", "topo_scene_balanced")):
        steps = sorted(timeline[family])
        speed_values: list[float] = []
        route_values: list[float] = []
        for step in steps:
            row = actions["models"][timeline[family][step]]
            speed = float(row["action_summary_by_risk"]["critical_ttc_le_3s"]["target_speed_mps_mean"])
            route = float(
                row["slot_mean_ablation"]["route"]["critical_ttc_le_3s"]
                ["mean_absolute_target_speed_delta_mps"]
            )
            speed_values.append(speed)
            route_values.append(route)
            source_rows.extend(
                [
                    {"panel": "d", "method": method, "raw_steps": step, "metric": "critical_target_speed_mps", "value": speed},
                    {"panel": "d", "method": method, "raw_steps": step, "metric": "route_slot_speed_sensitivity_mps", "value": route},
                ]
            )
        xs = [step / 1000 for step in steps]
        ax_action.plot(
            xs,
            speed_values,
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            linestyle="-" if family == "full" else "--",
            linewidth=2.0,
            label=f"{METHOD_LABEL[method]} critical speed",
        )
        action_twin.plot(
            xs,
            route_values,
            color=METHOD_COLOR[method],
            marker="x" if family == "full" else "+",
            linestyle=":" if family == "full" else "-.",
            linewidth=1.7,
            label=f"{METHOD_LABEL[method]} route sensitivity",
        )
    ax_action.set_xlabel("Raw SUMO steps (thousands)")
    ax_action.set_ylabel("Target speed on TTC<=3 s observations (m/s)")
    action_twin.set_ylabel("Route-slot mean-ablation speed delta (m/s)")
    ax_action.set_title("(d) Local policy-utilization diagnostic")
    lines = ax_action.get_lines() + action_twin.get_lines()
    ax_action.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7.2, loc="best")
    _style_axis(ax_action)
    action_twin.spines["top"].set_visible(False)
    action_twin.grid(False)

    figure.suptitle(
        "Topology-temporal Graph-SLT: outcome, trajectory, and mechanism evidence",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.012,
        context_note
        or "Closed-loop outcomes and shared-observation probes are shown under their validated source contracts; "
        "panels (c-d) are local diagnostics, not counterfactual safety estimates.",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#444444",
    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/development/summary.json")
    parser.add_argument("--diagnostic-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnostic/summary.json")
    parser.add_argument("--iteration-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/iteration/summary.json")
    parser.add_argument("--curves", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/fixed_checkpoint_curves.json")
    parser.add_argument("--checkpoint-probes", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/checkpoint_latent_probes.json")
    parser.add_argument("--checkpoint-actions", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/checkpoint_action_sensitivity.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/figures")
    args = parser.parse_args()

    source_paths = [
        args.development_summary.resolve(),
        args.diagnostic_summary.resolve(),
        args.iteration_summary.resolve(),
        args.curves.resolve(),
        args.checkpoint_probes.resolve(),
        args.checkpoint_actions.resolve(),
    ]
    development, diagnostic, iteration, curve_payload, probes, actions = [
        _read_json(path) for path in source_paths
    ]
    _validate_summary(development, "development")
    _validate_summary(diagnostic, "diagnostic")
    _validate_summary(iteration, "iteration")

    _require(curve_payload.get("contract") == "topo-scene.fixed-checkpoint-curve/v1", "Wrong checkpoint curve contract")
    _require(curve_payload.get("computed_from_real_rollouts") is True, "Checkpoint curves are not real rollouts")
    _require(curve_payload.get("fabricated_values") is False, "Checkpoint curves permit fabricated values")
    _require(curve_payload.get("fixed_checkpoint_curves_are_primary") is True, "Checkpoint curves are not marked primary")
    _require(isinstance(curve_payload.get("paired_protocol"), dict), "Checkpoint curves lack a paired protocol receipt")

    _require(probes.get("contract") == "topo-scene.latent-probes/v1", "Wrong latent-probe contract")
    _require(probes.get("computed_from_real_rollout") is True and probes.get("fabricated_values") is False, "Latent probes are not real")
    _require(probes.get("same_observations_for_all_models") is True, "Latent probes are not paired on observations")
    _require(actions.get("contract") == "topo-scene.slot-action-sensitivity/v1", "Wrong action-diagnostic contract")
    _require(actions.get("computed_from_real_observations") is True and actions.get("fabricated_values") is False, "Action diagnostics are not real")
    _require(actions.get("same_observations_for_all_models") is True, "Action diagnostics are not paired on observations")
    _require(
        probes.get("dataset_sha256") == actions.get("dataset_sha256"),
        "Probe/action dataset hashes differ",
    )
    _require(
        int(actions.get("samples", -1)) == int(probes["dataset"].get("transitions", -2)),
        "Probe/action dataset sample counts differ",
    )

    final_rows = {
        "scene_rep": _one_run(development, "scene_rep"),
        "temporal_graph": _one_run(diagnostic, "temporal_graph"),
        "topo_scene": _one_run(development, "topo_scene"),
        "topo_scene_balanced": _one_run(iteration, "topo_scene_balanced"),
    }
    interval_rows = {
        "scene_rep": _rate_intervals(development, "scene_rep"),
        "temporal_graph": _rate_intervals(diagnostic, "temporal_graph"),
        "topo_scene": _rate_intervals(development, "topo_scene"),
        "topo_scene_balanced": _rate_intervals(iteration, "topo_scene_balanced"),
    }
    curves = {str(row["algorithm"]): row for row in curve_payload["runs"]}
    _require(set(METHOD_ORDER).issubset(curves), f"Checkpoint curves lack methods: {set(METHOD_ORDER) - set(curves)}")
    timeline = _timeline_index(probes["models"])
    _require(set(actions["models"]) == set(probes["models"]), "Probe/action checkpoint model sets differ")

    paired = curve_payload["paired_protocol"]
    evaluation_seeds = paired["evaluation_seeds"]
    training_seeds = sorted({int(row["seed"]) for row in final_rows.values()})
    context_note = (
        f"Closed-loop: {paired['scenario']}, training seeds {training_seeds}, "
        f"{paired['episodes_per_checkpoint']} deterministic episodes with seeds "
        f"{evaluation_seeds[0]}-{evaluation_seeds[-1]}. Panels (c-d) use the same "
        f"{int(probes['dataset']['transitions']):,} observations and are local diagnostics, "
        "not counterfactual safety estimates. Panel (a) intervals are episode-bootstrap 95% CIs, "
        "not between-training-seed uncertainty."
    )
    source_rows: list[dict[str, Any]] = []
    figure = _build_figure(
        final_rows,
        curves,
        probes,
        actions,
        timeline,
        source_rows,
        context_note,
        interval_rows,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "final_evidence_dashboard"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    source_json = {
        "contract": "topo-scene.figure-source-data/v1",
        "computed_from_real_evidence": True,
        "fabricated_values": False,
        "rows": source_rows,
    }
    _atomic_json(output_dir / "dashboard_source_data.json", source_json)
    csv_path = output_dir / "dashboard_source_data.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "method", "raw_steps", "metric", "value"])
        writer.writeheader()
        writer.writerows(source_rows)
    temporary_csv.replace(csv_path)

    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        output_dir / "dashboard_source_data.json",
        csv_path,
    ]
    receipt = {
        "contract": "topo-scene.figure-receipt/v1",
        "computed_from_real_evidence": True,
        "fabricated_values": False,
        "sources": [{"path": _repo_path(path), "sha256": _sha256(path)} for path in source_paths],
        "outputs": [{"path": _repo_path(path), "sha256": _sha256(path)} for path in outputs],
    }
    _atomic_json(output_dir / "dashboard_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
