"""Post-hoc trace attribution for the failed v4.3 T3 development cell.

This tool never edits a frozen run and never treats fixed-observation decoder
recomputations as closed-loop outcomes.  It quantifies lane-command
continuity, observed lane-index transitions, and candidate arbitration rules
using only already-recorded real-rollout traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T2 = (
    PROJECT_ROOT
    / "results_topo_v4_3_dev"
    / "development"
    / "runs"
    / "T2__cand__cross__s0__pda6b283b"
)
DEFAULT_T3 = (
    PROJECT_ROOT
    / "results_topo_v4_3_dev"
    / "development"
    / "runs"
    / "T3__cand__cross__s1__pda6b283b"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results_topo_v4_3_dev" / "attribution" / "t3_failure"
)
LANE_COMMANDS = (-1, 0, 1)
CONTROLLER_PERIOD_SECONDS = 0.3
SUMO_LANE_CHANGE_DURATION_SECONDS = 3.0
CONFIDENCE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"missing or invalid JSONL records in {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _max_consecutive(decisions: Iterable[int]) -> int:
    ordered = sorted(set(int(value) for value in decisions))
    maximum = current = 0
    previous: int | None = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def _selected_from_actor(row: dict[str, Any]) -> int:
    probabilities = row["hybrid_policy"]["lane_probabilities"]
    return int(LANE_COMMANDS[max(range(3), key=lambda index: float(probabilities[index]))])


def _selected_from_fusion(row: dict[str, Any], threshold: float) -> int:
    actor_lane = _selected_from_actor(row)
    probabilities = row["hybrid_policy"]["lane_probabilities"]
    actor_confidence = float(probabilities[LANE_COMMANDS.index(actor_lane)])
    target_lane = int(row["target_critic_decoder"]["selected_lane"])
    if actor_lane != 0 and actor_confidence >= threshold:
        return actor_lane
    return target_lane


def _fixed_observation_rule_summary(
    rows: list[dict[str, Any]], *, rule: str, threshold: float | None = None
) -> dict[str, Any]:
    if rule == "target":
        selected = [int(row["target_critic_decoder"]["selected_lane"]) for row in rows]
    elif rule == "actor":
        selected = [_selected_from_actor(row) for row in rows]
    elif rule == "fusion" and threshold is not None:
        selected = [_selected_from_fusion(row, threshold) for row in rows]
    else:
        raise ValueError(f"invalid fixed-observation rule {rule!r}")

    route_indices = [
        index
        for index, row in enumerate(rows)
        if bool(row["pre_action_route_intent_valid"])
        and int(row["pre_action_route_intent"]) != 0
    ]
    matches = sum(
        int(selected[index] == int(rows[index]["pre_action_route_intent"]))
        for index in route_indices
    )
    by_episode: dict[int, list[int]] = defaultdict(list)
    for index in route_indices:
        if selected[index] == int(rows[index]["pre_action_route_intent"]):
            by_episode[int(rows[index]["episode"])].append(int(rows[index]["decision"]))
    streaks = [_max_consecutive(values) for values in by_episode.values()]
    all_episodes = sorted({int(row["episode"]) for row in rows})
    return {
        "rule": rule,
        "actor_non_keep_confidence_threshold": threshold,
        "fixed_observation_only": True,
        "closed_loop_outcome_inferred": False,
        "non_keep_rate": sum(int(value != 0) for value in selected) / len(selected),
        "route_window_rows": len(route_indices),
        "route_window_match_rate": matches / len(route_indices) if route_indices else None,
        "episodes_with_route_match": len(by_episode),
        "mean_maximum_route_match_streak": (
            sum(streaks) / len(all_episodes) if all_episodes else None
        ),
        "maximum_route_match_streak": max(streaks, default=0),
        "episodes_with_estimated_full_duration_streak": sum(
            int(_max_consecutive(by_episode.get(episode, [])) * CONTROLLER_PERIOD_SECONDS
                >= SUMO_LANE_CHANGE_DURATION_SECONDS)
            for episode in all_episodes
        ),
    }


def _episode_continuity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode"])].append(row)
    output: list[dict[str, Any]] = []
    for episode, episode_rows in sorted(by_episode.items()):
        episode_rows.sort(key=lambda row: int(row["decision"]))
        route_rows = [
            row
            for row in episode_rows
            if bool(row["pre_action_route_intent_valid"])
            and int(row["pre_action_route_intent"]) != 0
        ]
        matched = [row for row in route_rows if bool(row["route_intent_match"])]
        matched_decisions = [int(row["decision"]) for row in matched]
        maximum_streak = _max_consecutive(matched_decisions)

        edge_sequence: list[str] = []
        for row in episode_rows:
            edge = row.get("pre_action_current_edge")
            if edge is not None and (not edge_sequence or edge != edge_sequence[-1]):
                edge_sequence.append(str(edge))
        approximate_distance_m = 0.0
        previous_raw_steps = 0
        for row in episode_rows:
            raw_steps = int(row["raw_simulation_steps"])
            raw_delta = max(0, raw_steps - previous_raw_steps)
            approximate_distance_m += (
                float(row["actual_speed_mps"]) * 0.1 * raw_delta
            )
            previous_raw_steps = raw_steps
        tail_rows = episode_rows[-20:]

        target_lane_observed = False
        next_route_edge_observed = False
        if route_rows:
            first = route_rows[0]
            edge = first.get("pre_action_current_edge")
            next_edge = first.get("pre_action_next_route_edge")
            initial_lane = first.get("pre_action_current_lane_index")
            intent = int(first["pre_action_route_intent"])
            if initial_lane is not None:
                target_lane = int(initial_lane) + intent
                target_lane_observed = any(
                    row.get("pre_action_current_edge") == edge
                    and row.get("pre_action_current_lane_index") == target_lane
                    for row in episode_rows
                )
            next_route_edge_observed = any(
                row.get("pre_action_current_edge") == next_edge for row in episode_rows
            )

        terminal = episode_rows[-1]
        output.append(
            {
                "episode": episode,
                "seed": int(terminal["seed"]),
                "success": bool(terminal["is_success"]),
                "collision": bool(terminal["collision"]),
                "timeout": bool(terminal["timeout"]),
                "route_window_rows": len(route_rows),
                "route_match_commands": len(matched),
                "maximum_consecutive_route_match_commands": maximum_streak,
                "maximum_contiguous_request_seconds": (
                    maximum_streak * CONTROLLER_PERIOD_SECONDS
                ),
                "estimated_full_duration_streak": (
                    maximum_streak * CONTROLLER_PERIOD_SECONDS
                    >= SUMO_LANE_CHANGE_DURATION_SECONDS
                ),
                "target_lane_index_observed_later": target_lane_observed,
                "next_route_edge_observed_later": next_route_edge_observed,
                "compressed_edge_sequence": edge_sequence,
                "terminal_edge": terminal.get("pre_action_current_edge"),
                "terminal_lane_index": terminal.get("pre_action_current_lane_index"),
                "terminal_route_intent_reason": terminal.get(
                    "pre_action_route_intent_reason"
                ),
                "mean_actual_speed_mps": sum(
                    float(row["actual_speed_mps"]) for row in episode_rows
                )
                / len(episode_rows),
                "last_20_mean_actual_speed_mps": sum(
                    float(row["actual_speed_mps"]) for row in tail_rows
                )
                / len(tail_rows),
                "approximate_integrated_distance_m": approximate_distance_m,
            }
        )
    return output


def _group_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def _run_summary(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "action_diagnostics_decisions.jsonl"
    diagnostics_path = run_dir / "action_diagnostics.json"
    evaluation_path = run_dir / "final_evaluation.json"
    best_path = run_dir / "best_training_success.json"
    for path in (trace_path, diagnostics_path, evaluation_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = _load_jsonl(trace_path)
    diagnostics = _load_json(diagnostics_path)
    evaluation = _load_json(evaluation_path)
    best = _load_json(best_path)
    per_episode = _episode_continuity(rows)
    successes = [row for row in per_episode if row["success"]]
    failures = [row for row in per_episode if not row["success"]]
    fixed_rules = [
        _fixed_observation_rule_summary(rows, rule="target"),
        _fixed_observation_rule_summary(rows, rule="actor"),
        *[
            _fixed_observation_rule_summary(rows, rule="fusion", threshold=value)
            for value in CONFIDENCE_THRESHOLDS
        ],
    ]
    return {
        "run_directory": str(run_dir.resolve()),
        "source_hashes": {
            "trace_sha256": _sha256(trace_path),
            "action_diagnostics_sha256": _sha256(diagnostics_path),
            "final_evaluation_sha256": _sha256(evaluation_path),
            "best_training_success_sha256": _sha256(best_path),
        },
        "outcomes": evaluation,
        "best_training_success": best,
        "observed_decoder": {
            "lane_command_rates": diagnostics["lane_command_rates"],
            "route_action_window_match_rate": diagnostics["route_action_window_match_rate"],
            "target_route_event_applied_match_episode_rate": diagnostics.get(
                "target_route_event_applied_match_episode_rate"
            ),
            "actual_speed_mps": diagnostics["actual_speed_mps"],
        },
        "continuity_contract": {
            "controller_period_seconds": CONTROLLER_PERIOD_SECONDS,
            "sumo_lane_change_duration_seconds": SUMO_LANE_CHANGE_DURATION_SECONDS,
            "commands_for_nominal_duration": int(
                SUMO_LANE_CHANGE_DURATION_SECONDS / CONTROLLER_PERIOD_SECONDS
            ),
            "duration_is_mechanistic_hypothesis_not_outcome_proof": True,
        },
        "continuity_by_outcome": {
            "successful_episodes": len(successes),
            "failed_episodes": len(failures),
            "success_mean_maximum_streak": _group_mean(
                successes, "maximum_consecutive_route_match_commands"
            ),
            "failure_mean_maximum_streak": _group_mean(
                failures, "maximum_consecutive_route_match_commands"
            ),
            "success_target_lane_observed_rate": _group_mean(
                successes, "target_lane_index_observed_later"
            ),
            "failure_target_lane_observed_rate": _group_mean(
                failures, "target_lane_index_observed_later"
            ),
            "success_next_route_edge_observed_rate": _group_mean(
                successes, "next_route_edge_observed_later"
            ),
            "failure_next_route_edge_observed_rate": _group_mean(
                failures, "next_route_edge_observed_later"
            ),
        },
        "per_episode_continuity": per_episode,
        "fixed_observation_decoder_sweep": fixed_rules,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v4.3 T3 Failure Trace Attribution",
        "",
        "All values below come from existing real-rollout artifacts. Fixed-observation",
        "decoder sweeps do not infer closed-loop success or collision outcomes.",
        "",
    ]
    for name in ("T2_cross_seed0", "T3_cross_seed1"):
        run = payload["runs"][name]
        outcome = run["outcomes"]
        continuity = run["continuity_by_outcome"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- success={outcome['success_rate']:.6f}, collision={outcome['collision_rate']:.6f}, timeout={outcome['timeout_rate']:.6f}",
                f"- successful episodes={continuity['successful_episodes']}, failed episodes={continuity['failed_episodes']}",
                f"- successful mean maximum route-command streak={continuity['success_mean_maximum_streak']}",
                f"- failed mean maximum route-command streak={continuity['failure_mean_maximum_streak']}",
                f"- successful target-lane observed rate={continuity['success_target_lane_observed_rate']}",
                f"- failed target-lane observed rate={continuity['failure_target_lane_observed_rate']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The trace can support or reject command-continuity and lane-transition hypotheses.",
            "Any decoder chosen for a new version still requires real closed-loop replay followed",
            "by fresh preregistered development cells.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--t2-run", type=Path, default=DEFAULT_T2)
    parser.add_argument("--t3-run", type=Path, default=DEFAULT_T3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "topo-scene-v4.3.t3-trace-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "formal_test_accessed": False,
        "runs": {
            "T2_cross_seed0": _run_summary(args.t2_run.resolve()),
            "T3_cross_seed1": _run_summary(args.t3_run.resolve()),
        },
    }
    _write_json(output / "trace_attribution.json", payload)
    (output / "trace_attribution.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "trace_attribution_sha256": _sha256(output / "trace_attribution.json"),
        "markdown_sha256": _sha256(output / "trace_attribution.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
