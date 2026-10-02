"""Paired trace attribution for the strict target-only v4.9.2 promotion.

This analysis is observational and may run on a partial matrix only when
explicitly requested.  It compares paired TemporalGraph/candidate validation
episodes, measures behavior-distribution shifts, and reuses the learned-risk
separation analysis.  It never changes an action, creates a geometric label,
or applies a kinematic threshold/projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import attribute_v4_9_learned_model as learned
from tools import run_all_v4_9_2_2_recovery as recovery
from tools import run_all_v4_9_2_3_short_recovery as short_recovery
from tools import run_topo_v4_9_2_experiments as frozen


EXPECTED_PAIRS = 6
WINDOWS = (1, 5, 10)
DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "attribution"
    / "promotion_model_attribution.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return fmean(items) if items else None


def _delta(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def _trace_path(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    diagnostics = _load_json(run_dir / "action_diagnostics.json")
    metadata = diagnostics.get("trace")
    if not isinstance(metadata, dict):
        raise TypeError(f"missing trace metadata: {run_dir}")
    path = Path(str(metadata["path"]))
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        fallback = run_dir / "action_diagnostics_decisions.jsonl"
        if not fallback.is_file():
            raise FileNotFoundError(path)
        path = fallback
    actual = _sha256(path)
    expected = str(metadata.get("sha256", ""))
    if expected and actual != expected:
        raise ValueError(f"trace hash mismatch: {path}")
    return path, diagnostics


def load_episodes(run_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    path, diagnostics = _trace_path(run_dir)
    episodes: dict[int, list[dict[str, Any]]] = {}
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"trace line {line_number} is not an object")
            episodes.setdefault(int(row["episode"]), []).append(row)
            count += 1
    if count != int(diagnostics["trace"]["records"]):
        raise ValueError(f"trace record count drifted: {path}")
    for episode, rows in episodes.items():
        rows.sort(key=lambda row: int(row["decision"]))
        if not rows or not bool(rows[-1].get("terminated") or rows[-1].get("truncated")):
            raise ValueError(f"episode {episode} lacks a terminal row")
    return episodes, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": count,
    }


def outcome(row: dict[str, Any]) -> str:
    if bool(row.get("collision")):
        return "collision"
    if bool(row.get("is_success")):
        return "success"
    if bool(row.get("off_route")):
        return "off_route"
    if bool(row.get("timeout")):
        return "timeout"
    return "other"


def _actor_command(row: dict[str, Any]) -> int | None:
    hybrid = row.get("hybrid_policy")
    if not isinstance(hybrid, dict):
        return None
    probabilities = hybrid.get("lane_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 3:
        return None
    index = max(range(3), key=lambda item: float(probabilities[item]))
    return (-1, 0, 1)[index]


def behavior_metrics(
    episodes: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    terminal = [rows[-1] for rows in episodes.values()]
    all_rows = [row for rows in episodes.values() for row in rows]
    adjacent_pairs = [
        (left, right)
        for rows in episodes.values()
        for left, right in zip(rows, rows[1:])
    ]
    valid_route = [
        row for row in all_rows if bool(row.get("pre_action_route_intent_valid"))
    ]
    feasible_nonkeep = [
        row for row in all_rows if bool(row.get("pre_action_non_keep_feasible"))
    ]
    actor_rows = [
        (row, command)
        for row in all_rows
        if (command := _actor_command(row)) is not None
    ]

    def terminal_window(window: int, key: str) -> dict[str, float | None]:
        collision_values = [
            fmean(float(row[key]) for row in rows[-window:])
            for rows in episodes.values()
            if bool(rows[-1].get("collision"))
        ]
        noncollision_values = [
            fmean(float(row[key]) for row in rows[-window:])
            for rows in episodes.values()
            if not bool(rows[-1].get("collision"))
        ]
        return {
            "collision_episode_mean": _mean(collision_values),
            "noncollision_episode_mean": _mean(noncollision_values),
            "collision_minus_noncollision": _delta(
                _mean(collision_values), _mean(noncollision_values)
            ),
        }

    grouped: dict[str, Any] = {}
    for label in ("collision", "noncollision"):
        selected = {
            episode: rows
            for episode, rows in episodes.items()
            if (outcome(rows[-1]) == "collision") == (label == "collision")
        }
        selected_rows = [row for rows in selected.values() for row in rows]
        grouped[label] = {
            "episodes": len(selected),
            "decisions": len(selected_rows),
            "non_keep_command_rate": _mean(
                float(int(row["lane_command"]) != 0) for row in selected_rows
            ),
            "lane_change_applied_rate": _mean(
                float(bool(row.get("lane_change_applied")))
                for row in selected_rows
            ),
            "lane_changes_applied_per_episode": _mean(
                sum(bool(row.get("lane_change_applied")) for row in rows)
                for rows in selected.values()
            ),
            "target_speed_mps": _mean(
                float(row["target_speed_mps"]) for row in selected_rows
            ),
            "actual_speed_mps": _mean(
                float(row["actual_speed_mps"]) for row in selected_rows
            ),
        }

    return {
        "episodes": len(episodes),
        "decisions": len(all_rows),
        "outcomes": dict(Counter(outcome(row) for row in terminal)),
        "non_keep_command_rate": _mean(
            float(int(row["lane_command"]) != 0) for row in all_rows
        ),
        "lane_change_applied_rate": _mean(
            float(bool(row.get("lane_change_applied"))) for row in all_rows
        ),
        "lane_changes_applied_per_episode": _mean(
            sum(bool(row.get("lane_change_applied")) for row in rows)
            for rows in episodes.values()
        ),
        "command_switch_rate": _mean(
            float(int(left["lane_command"]) != int(right["lane_command"]))
            for left, right in adjacent_pairs
        ),
        "opposite_direction_flip_rate": _mean(
            float(int(left["lane_command"]) * int(right["lane_command"]) == -1)
            for left, right in adjacent_pairs
        ),
        "non_keep_when_feasible_rate": _mean(
            float(int(row["lane_command"]) != 0) for row in feasible_nonkeep
        ),
        "route_intent_match_rate": _mean(
            float(bool(row.get("route_intent_match"))) for row in valid_route
        ),
        "longitudinal_saturation_rate": _mean(
            float(bool(row.get("longitudinal_saturated"))) for row in all_rows
        ),
        "target_speed_mps": _mean(float(row["target_speed_mps"]) for row in all_rows),
        "actual_speed_mps": _mean(float(row["actual_speed_mps"]) for row in all_rows),
        "actor_target_lane_match_rate": _mean(
            float(command == int(row["lane_command"]))
            for row, command in actor_rows
        ),
        "actor_non_keep_rate": _mean(
            float(command != 0) for _row, command in actor_rows
        ),
        "terminal_windows": {
            str(window): {
                "target_speed_mps": terminal_window(window, "target_speed_mps"),
                "actual_speed_mps": terminal_window(window, "actual_speed_mps"),
            }
            for window in WINDOWS
        },
        "by_collision_outcome": grouped,
    }


def analyze_source(
    *,
    method: str,
    scenario: str,
    seed: int,
    source_kind: str,
    run_dir: Path,
) -> dict[str, Any]:
    detailed = _load_json(run_dir / "paper_evaluation_detailed.json")
    episodes, trace = load_episodes(run_dir)
    metrics = behavior_metrics(episodes)
    risk: dict[str, Any] | None = None
    if method == "selected_v4_candidate":
        full = learned.analyze_run(run_dir)
        risk = {
            "selected_deployment_decoder": full[
                "selected_deployment_decoder"
            ],
            "label_and_scale": full["label_and_scale"],
            "terminal_window_separation": full[
                "terminal_window_separation"
            ],
            "evidence": full["evidence"],
        }
    return {
        "method": method,
        "scenario": scenario,
        "seed": int(seed),
        "source_kind": source_kind,
        "run_dir": str(run_dir.resolve()),
        "evaluation_summary": detailed["summary"],
        "selected_checkpoint_kind": detailed.get("selected_checkpoint_kind"),
        "selected_deployment_decoder": detailed.get(
            "selected_deployment_decoder"
        ),
        "behavior": metrics,
        "learned_risk": risk,
        "evidence": {
            "detailed_sha256": _sha256(
                run_dir / "paper_evaluation_detailed.json"
            ),
            "action_diagnostics_sha256": _sha256(
                run_dir / "action_diagnostics.json"
            ),
            "trace": trace,
        },
    }


def discover_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    short_recovery.configure_engine()
    contract = frozen.validate_contract(frozen.load_contract())
    digest = frozen._sha256(frozen.DEFAULT_CONTRACT)
    hashes = frozen.protocol_hashes()
    sources: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for job in frozen.jobs_for_stage(contract, digest, "promotion"):
        kind, root, evidence = recovery.selected_source(job, hashes)
        if root is None or kind is None:
            unresolved.append({"job": job.name, "sources": evidence})
            continue
        sources.append(
            {
                "job": job.name,
                "method": job.method,
                "scenario": job.scenario,
                "seed": job.seed,
                "source_kind": kind,
                "run_dir": root,
            }
        )
    return sources, unresolved


def pair_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(run["method"]), str(run["scenario"]), int(run["seed"])): run
        for run in runs
    }
    pairs: list[dict[str, Any]] = []
    keys = sorted({(str(run["scenario"]), int(run["seed"])) for run in runs})
    for scenario, seed in keys:
        control = indexed.get(("temporal_graph", scenario, seed))
        candidate = indexed.get(("selected_v4_candidate", scenario, seed))
        if control is None or candidate is None:
            continue
        control_episodes, _ = load_episodes(Path(control["run_dir"]))
        candidate_episodes, _ = load_episodes(Path(candidate["run_dir"]))
        control_by_seed = {int(rows[-1]["seed"]): outcome(rows[-1]) for rows in control_episodes.values()}
        candidate_by_seed = {int(rows[-1]["seed"]): outcome(rows[-1]) for rows in candidate_episodes.values()}
        if set(control_by_seed) != set(candidate_by_seed):
            raise ValueError(f"paired evaluation seeds drifted: {scenario} seed{seed}")
        transitions = Counter(
            f"{control_by_seed[item]}_to_{candidate_by_seed[item]}"
            for item in sorted(control_by_seed)
        )
        control_behavior = control["behavior"]
        candidate_behavior = candidate["behavior"]
        deltas = {
            name: _delta(candidate_behavior.get(name), control_behavior.get(name))
            for name in (
                "non_keep_command_rate",
                "lane_change_applied_rate",
                "lane_changes_applied_per_episode",
                "command_switch_rate",
                "opposite_direction_flip_rate",
                "non_keep_when_feasible_rate",
                "route_intent_match_rate",
                "longitudinal_saturation_rate",
                "target_speed_mps",
                "actual_speed_mps",
            )
        }
        pairs.append(
            {
                "scenario": scenario,
                "seed": seed,
                "evaluation_episode_seeds": sorted(control_by_seed),
                "outcome_transitions": dict(transitions),
                "behavior_delta_candidate_minus_temporal_graph": deltas,
                "control": control,
                "candidate": candidate,
            }
        )
    return pairs


def summarize(*, allow_partial: bool) -> dict[str, Any]:
    source_specs, unresolved = discover_sources()
    runs = [
        analyze_source(
            method=str(spec["method"]),
            scenario=str(spec["scenario"]),
            seed=int(spec["seed"]),
            source_kind=str(spec["source_kind"]),
            run_dir=Path(spec["run_dir"]),
        )
        for spec in source_specs
    ]
    pairs = pair_runs(runs)
    complete = len(pairs) == EXPECTED_PAIRS
    if not complete and not allow_partial:
        raise RuntimeError(
            f"promotion attribution requires {EXPECTED_PAIRS} pairs; found {len(pairs)}"
        )
    risk_rows = [
        pair["candidate"]["learned_risk"] for pair in pairs
        if pair["candidate"]["learned_risk"] is not None
    ]
    window10 = [row["terminal_window_separation"]["10"] for row in risk_rows]
    behavior_deltas = [
        pair["behavior_delta_candidate_minus_temporal_graph"] for pair in pairs
    ]
    payload = {
        "schema_version": "topo-scene-v4.9.2.promotion-model-attribution/v1",
        "analysis_kind": "paired_observational_trace_attribution",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "partial_matrix": not complete,
        "decision_allowed": complete,
        "pair_count": len(pairs),
        "expected_pair_count": EXPECTED_PAIRS,
        "run_count": len(runs),
        "unresolved_logical_jobs": unresolved,
        "forbidden_mechanisms_used": {
            "external_kinematic_projection": False,
            "headway_or_ttc_threshold": False,
            "lane_change_veto": False,
            "manual_geometry_label": False,
            "counterfactual_action_rewrite": False,
        },
        "aggregate_mechanism_measurements": {
            "candidate_collision_value_window10_rank_auc_mean": _mean(
                row.get("mean_value_rank_auc") for row in window10
                if row.get("mean_value_rank_auc") is not None
            ),
            "candidate_collision_selected_minimum_predicted_risk_rate_mean": _mean(
                1.0 - float(row["collision_minimum_value_action_disagreement_rate"])
                for row in window10
                if row.get("collision_minimum_value_action_disagreement_rate") is not None
            ),
            "candidate_minus_temporal_graph_non_keep_rate_mean": _mean(
                row["non_keep_command_rate"] for row in behavior_deltas
                if row["non_keep_command_rate"] is not None
            ),
            "candidate_minus_temporal_graph_lane_change_applied_rate_mean": _mean(
                row["lane_change_applied_rate"] for row in behavior_deltas
                if row["lane_change_applied_rate"] is not None
            ),
            "candidate_actor_target_lane_match_rate_mean": _mean(
                pair["candidate"]["behavior"]["actor_target_lane_match_rate"]
                for pair in pairs
                if pair["candidate"]["behavior"]["actor_target_lane_match_rate"] is not None
            ),
            "candidate_actor_non_keep_rate_mean": _mean(
                pair["candidate"]["behavior"]["actor_non_keep_rate"]
                for pair in pairs
                if pair["candidate"]["behavior"]["actor_non_keep_rate"] is not None
            ),
        },
        "hypothesis_evidence_status": {
            "missing_external_safety_projection": (
                "not_a_permitted_hypothesis_and_no_projection_is_proposed"
            ),
            "collision_value_discrimination_failure": (
                "measured_but_requires_complete_matrix"
                if not complete
                else "ready_for_attribution_decision"
            ),
            "limited_action_proposal_or_future_policy_mismatch": (
                "measured_but_requires_complete_matrix"
                if not complete
                else "ready_for_attribution_decision"
            ),
            "independent_collision_vs_actor_reward_representation_mismatch": (
                "measured_but_requires_complete_matrix"
                if not complete
                else "ready_for_attribution_decision"
            ),
        },
        "pairs": pairs,
    }
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--allow-partial", action="store_true")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = summarize(allow_partial=args.allow_partial)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_PAIRS",
    "analyze_source",
    "behavior_metrics",
    "discover_sources",
    "load_episodes",
    "outcome",
    "pair_runs",
    "summarize",
]
