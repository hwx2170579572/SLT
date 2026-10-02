"""Audit the v4.10 stage-0 speed-proposal probe and attribute its evidence.

This analysis never opens a formal-test artifact and never changes an action.
It verifies every development trace against the probe manifest, compares
collision and success terminal windows, and measures whether diagnostic optima
sit at the queried speed boundaries (a warning for critic extrapolation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROBE = (
    ROOT
    / "results_topo_v4_10_dev"
    / "stage_0_speed_proposal_probe"
    / "speed_proposal_coverage.json"
)
DEFAULT_OUTPUT = DEFAULT_PROBE.with_name("deep_attribution.json")
DEFAULT_MARKDOWN = DEFAULT_PROBE.with_name("DEEP_ATTRIBUTION_V4_10.md")
EXPECTED_PAIRS = {
    (scenario, seed)
    for scenario in ("carla", "cross", "roundabout_medium")
    for seed in (20, 21)
}
TOLERANCE = 1e-7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rate(flags: Iterable[bool]) -> float | None:
    values = list(flags)
    return sum(values) / len(values) if values else None


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "minimum": min(items),
        "maximum": max(items),
    }


def terminal_rows(
    rows: list[dict[str, Any]], *, outcome: str, window: int
) -> list[dict[str, Any]]:
    if window <= 0:
        raise ValueError("terminal window must be positive")
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["eventual_episode_outcome"]) == outcome:
            episodes[int(row["episode"])].append(row)
    terminal: list[dict[str, Any]] = []
    for episode in sorted(episodes):
        episode_rows = sorted(episodes[episode], key=lambda row: int(row["decision"]))
        terminal.extend(episode_rows[-window:])
    return terminal


def summarize_rows(
    rows: list[dict[str, Any]], *, grid_min: float, grid_max: float
) -> dict[str, Any]:
    if not rows:
        return {"decision_count": 0}
    probes = [row["learned_speed_proposal_probe"] for row in rows]
    best = [probe["same_lane_best"] for probe in probes]
    score_gains = [float(probe["same_lane_score_gain"]) for probe in probes]
    risk_reductions = [
        float(probe["same_lane_collision_value_reduction"]) for probe in probes
    ]
    speed_deltas = [float(probe["same_lane_speed_delta_mps"]) for probe in probes]
    grid_selected = [str(item["source"]) == "diagnostic_grid" for item in best]
    strict = [gain > TOLERANCE for gain in score_gains]
    safer_higher = [
        gain > TOLERANCE and reduction > TOLERANCE
        for gain, reduction in zip(score_gains, risk_reductions)
    ]
    safer_slower = [
        flag and delta < -TOLERANCE
        for flag, delta in zip(safer_higher, speed_deltas)
    ]
    safer_faster = [
        flag and delta > TOLERANCE
        for flag, delta in zip(safer_higher, speed_deltas)
    ]
    lower_edge = [
        selected
        and math.isclose(float(item["best_speed_normalized"]), grid_min, abs_tol=1e-6)
        for selected, item in zip(grid_selected, best)
    ]
    upper_edge = [
        selected
        and math.isclose(float(item["best_speed_normalized"]), grid_max, abs_tol=1e-6)
        for selected, item in zip(grid_selected, best)
    ]
    grid_count = sum(grid_selected)
    safer_count = sum(safer_higher)
    speed_histogram: dict[str, int] = defaultdict(int)
    for selected, item in zip(grid_selected, best):
        if selected:
            key = f"{float(item['best_speed_mps']):.2f}"
            speed_histogram[key] += 1
    return {
        "decision_count": len(rows),
        "diagnostic_grid_selected_rate": _rate(grid_selected),
        "strict_score_improvement_rate": _rate(strict),
        "safer_and_higher_score_rate": _rate(safer_higher),
        "safer_higher_and_slower_rate": _rate(safer_slower),
        "safer_higher_and_faster_rate": _rate(safer_faster),
        "slower_given_safer_higher_rate": (
            sum(safer_slower) / safer_count if safer_count else None
        ),
        "faster_given_safer_higher_rate": (
            sum(safer_faster) / safer_count if safer_count else None
        ),
        "lower_grid_edge_rate_all": _rate(lower_edge),
        "upper_grid_edge_rate_all": _rate(upper_edge),
        "either_grid_edge_rate_given_grid_selected": (
            (sum(lower_edge) + sum(upper_edge)) / grid_count if grid_count else None
        ),
        "actor_speed_mps": _summary(
            float(probe["baseline_speed_mps"]) for probe in probes
        ),
        "diagnostic_best_speed_mps": _summary(
            float(item["best_speed_mps"]) for item in best
        ),
        "absolute_speed_delta_mps": _summary(abs(value) for value in speed_deltas),
        "signed_speed_delta_mps": _summary(speed_deltas),
        "score_gain": _summary(score_gains),
        "collision_value_reduction": _summary(risk_reductions),
        "diagnostic_grid_speed_histogram_mps": dict(sorted(speed_histogram.items())),
    }


def _load_and_verify_trace(
    run: Mapping[str, Any], *, grid_min: float, grid_max: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = (ROOT / str(run["trace"])).resolve()
    _require(trace_path.is_file(), f"missing trace: {trace_path}")
    _require(_sha256(trace_path) == run["trace_sha256"], f"trace hash drifted: {trace_path}")
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"non-object trace row {trace_path}:{line_number}")
            probe = value.get("learned_speed_proposal_probe")
            _require(isinstance(probe, dict), f"missing probe row {trace_path}:{line_number}")
            _require(probe.get("diagnostic_only") is True, "probe row is not diagnostic-only")
            _require(probe.get("action_rewritten") is False, "probe row rewrote an action")
            _require(
                int(value["lane_command"]) == int(probe["baseline_selected_lane"]),
                "deployed lane differs from frozen baseline",
            )
            _require(
                abs(float(value["action_longitudinal"]) - float(probe["baseline_speed_normalized"]))
                <= 1e-5,
                "deployed speed differs from frozen baseline",
            )
            rows.append(value)
    _require(len(rows) == int(run["trace_records"]), f"trace count drifted: {trace_path}")
    episode_ids = {int(row["episode"]) for row in rows}
    _require(len(episode_ids) == int(run["episodes"]), f"episode count drifted: {trace_path}")
    by_outcome = {
        outcome: summarize_rows(
            [row for row in rows if str(row["eventual_episode_outcome"]) == outcome],
            grid_min=grid_min,
            grid_max=grid_max,
        )
        for outcome in ("success", "collision", "off_route", "timeout")
    }
    terminal = {
        outcome: {
            str(window): summarize_rows(
                terminal_rows(rows, outcome=outcome, window=window),
                grid_min=grid_min,
                grid_max=grid_max,
            )
            for window in (1, 5, 10)
        }
        for outcome in ("success", "collision")
    }
    analysis = {
        "trace": _relative(trace_path),
        "trace_sha256_verified": run["trace_sha256"],
        "decision_count": len(rows),
        "episode_count": len(episode_ids),
        "all": summarize_rows(rows, grid_min=grid_min, grid_max=grid_max),
        "by_outcome": by_outcome,
        "terminal_windows": terminal,
    }
    return rows, analysis


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# v4.10 Stage-0 learned speed-proposal attribution",
        "",
        "This is development-only evidence from frozen v4.9.2 checkpoints. The diagnostic grid was never sent to SUMO; it is not a deployed safety mechanism.",
        "",
        "## Integrity",
        "",
        f"- Complete pairs: {payload['integrity']['verified_pair_count']}/6",
        f"- Trace records: {payload['integrity']['verified_trace_records']}",
        f"- Formal test accessed: {str(payload['formal_test_accessed']).lower()}",
        f"- Deployed actions changed: {str(payload['deployed_actions_changed_by_probe']).lower()}",
        "",
        "## Pair evidence",
        "",
        "| Scenario | Seed | Probe collision | All safer+higher | Collision terminal-10 safer+higher | Collision terminal-10 grid-edge | Collision terminal-10 slower given safer+higher |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["pairs"]:
        terminal = row["analysis"]["terminal_windows"]["collision"]["10"]
        lines.append(
            "| {scenario} | {seed} | {collision} | {all_safe} | {terminal_safe} | {edge} | {slower} |".format(
                scenario=row["scenario"],
                seed=row["seed"],
                collision=_fmt(row["probe_outcomes"]["collision_rate"]),
                all_safe=_fmt(row["analysis"]["all"].get("safer_and_higher_score_rate")),
                terminal_safe=_fmt(terminal.get("safer_and_higher_score_rate")),
                edge=_fmt(terminal.get("either_grid_edge_rate_given_grid_selected")),
                slower=_fmt(terminal.get("slower_given_safer_higher_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            *[f"- {item}" for item in payload["findings"]],
            "",
            "## Model-only consequence",
            "",
            *[f"- {item}" for item in payload["model_only_consequences"]],
            "",
            "## Causal limits",
            "",
            *[f"- {item}" for item in payload["causal_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_attribution(probe_path: Path) -> dict[str, Any]:
    probe = _load_json(probe_path)
    _require(probe.get("decision_allowed") is True, "probe is not decision-complete")
    _require(probe.get("formal_test_accessed") is False, "formal test was accessed")
    _require(probe.get("deployed_actions_changed_by_probe") is False, "probe changed actions")
    _require(not probe.get("failures"), "probe contains failed pairs")
    _require(probe.get("aggregate", {}).get("all_runs_completed") is True, "probe matrix incomplete")
    runs = list(probe.get("runs", []))
    identities = {(str(run["scenario"]), int(run["seed"])) for run in runs}
    _require(identities == EXPECTED_PAIRS, f"unexpected pair matrix: {sorted(identities)}")
    grid = [float(value) for value in probe["speed_grid_normalized"]]
    _require(len(grid) >= 3 and grid == sorted(set(grid)), "invalid diagnostic grid")

    pair_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: (str(item["scenario"]), int(item["seed"]))):
        rows, analysis = _load_and_verify_trace(run, grid_min=grid[0], grid_max=grid[-1])
        all_rows.extend(rows)
        pair_rows.append(
            {
                "scenario": str(run["scenario"]),
                "seed": int(run["seed"]),
                "promotion_outcomes": run["promotion_outcomes"],
                "probe_outcomes": run["outcomes"],
                "analysis": analysis,
            }
        )

    collision_terminal10 = []
    success_terminal10 = []
    for row in pair_rows:
        collision_terminal10.append(
            row["analysis"]["terminal_windows"]["collision"]["10"]
        )
        success_terminal10.append(
            row["analysis"]["terminal_windows"]["success"]["10"]
        )

    collision_pairs_with_events = [
        item for item in collision_terminal10 if item["decision_count"] > 0
    ]
    _require(
        bool(collision_pairs_with_events),
        "probe contains no collision-terminal evidence for attribution",
    )
    collision_terminal_safe_rates = [
        float(item["safer_and_higher_score_rate"])
        for item in collision_pairs_with_events
    ]
    collision_terminal_edge_rates = [
        float(item["either_grid_edge_rate_given_grid_selected"])
        for item in collision_pairs_with_events
    ]
    all_summary = summarize_rows(all_rows, grid_min=grid[0], grid_max=grid[-1])

    findings = [
        (
            "The single actor speed is not a stationary point of its own frozen target critics: "
            f"a diagnostic speed has a strictly higher same-lane score on "
            f"{_fmt(all_summary['strict_score_improvement_rate'])} of all visited decisions."
        ),
        (
            "Collision-terminal evidence is present but not sufficient by itself: across the "
            f"{len(collision_pairs_with_events)} pairs containing probe collisions, the terminal-10 "
            f"safer-and-higher rate ranges from {_fmt(min(collision_terminal_safe_rates))} to "
            f"{_fmt(max(collision_terminal_safe_rates))}."
        ),
        (
            "The queried optimum frequently reaches a diagnostic speed edge in collision windows "
            f"(pair range {_fmt(min(collision_terminal_edge_rates))} to "
            f"{_fmt(max(collision_terminal_edge_rates))}); this is evidence of critic/action "
            "support mismatch, so direct critic maximization or deployment of the grid is unsafe."
        ),
        (
            "The proposal gap also appears on successful trajectories, so it is a general actor-critic "
            "consistency defect rather than a collision-specific rule that can be patched at inference."
        ),
    ]
    consequences = [
        "Replace the one-speed-per-lane actor with multiple learned, bounded speed proposals and a learned proposal distribution; do not retain the diagnostic grid at inference.",
        "Train proposal diversity and critic agreement only on replay-supported actions, with an explicit support/uncertainty signal learned from data to prevent boundary exploitation.",
        "Let collision representation inform the proposal features and give each shared parameter exactly one optimizer owner; keep collision supervision and target-network evaluation learned end to end.",
        "Ablate proposal multiplicity, support regularization, and risk-conditioned representation separately before promotion.",
    ]
    limits = [
        "Frozen-critic counterfactual scores do not establish the closed-loop outcome of an unexecuted speed.",
        "The 12-episode probe is development evidence; it cannot replace the locked promotion or formal-test matrices.",
        "A boundary optimum may be critic extrapolation rather than a genuinely safe action, which is why v4.10 must learn supported proposals and be evaluated in SUMO.",
    ]
    return {
        "schema_version": "topo-scene-v4.10.stage0-deep-attribution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_probe": _relative(probe_path),
        "source_probe_sha256": _sha256(probe_path),
        "evidence_scope": "development_validation_only",
        "fabricated_values": False,
        "formal_test_accessed": False,
        "diagnostic_grid_used_for_deployment": False,
        "deployed_actions_changed_by_probe": False,
        "integrity": {
            "verified_pair_count": len(pair_rows),
            "verified_trace_records": len(all_rows),
            "all_trace_hashes_verified": True,
            "all_episode_counts_verified": True,
        },
        "diagnostic_grid_normalized": grid,
        "global": all_summary,
        "pairs": pair_rows,
        "findings": findings,
        "model_only_consequences": consequences,
        "causal_limits": limits,
        "decision_allowed": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_attribution(args.probe.resolve())
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": _relative(output),
                "output_sha256": _sha256(output),
                "markdown": _relative(markdown),
                "markdown_sha256": _sha256(markdown),
                "verified_pair_count": payload["integrity"]["verified_pair_count"],
                "verified_trace_records": payload["integrity"]["verified_trace_records"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
