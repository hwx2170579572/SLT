"""Build a read-only, evidence-bound attribution for the v4.8 promotion fail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "results_topo_v4_8_promotion"
RUNS = PROMOTION / "runs"
GATE = PROMOTION / "promotion_gate.json"
SUMMARY = PROMOTION / "summary.json"
RECONCILIATION = PROMOTION / "automation" / "promotion_reconciliation_v3.json"
DEFAULT_OUTPUT = PROMOTION / "attribution" / "promotion_failure_v4_8.json"
DEFAULT_REPORT = PROMOTION / "attribution" / "PROMOTION_FAILURE_V4_8.md"

CROSS_RUNS = {
    ("temporal_graph", 0): "P__tg__cross__s0__p42f54135",
    ("candidate", 0): "P__cand__cross__s0__p42f54135",
    ("temporal_graph", 1): "P__tg__cross__s1__p42f54135",
    ("candidate", 1): "P__cand__cross__s1__p42f54135",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), f"empty JSONL: {path}")
    _require(all(isinstance(row, dict) for row in rows), f"invalid JSONL: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "population_std": statistics.pstdev(finite),
        "minimum": min(finite),
        "p10": _percentile(finite, 0.10),
        "p25": _percentile(finite, 0.25),
        "median": _percentile(finite, 0.50),
        "p75": _percentile(finite, 0.75),
        "p90": _percentile(finite, 0.90),
        "maximum": max(finite),
    }


def _outcome(row: dict[str, Any]) -> str:
    for key in ("success", "collision", "off_route", "timeout"):
        if bool(row.get(key)):
            return key
    return "unknown"


def _episode_map(detailed: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = detailed.get("episode_records")
    _require(isinstance(rows, list) and len(rows) == 30, "expected 30 episode records")
    output = {int(row["episode"]): row for row in rows}
    _require(len(output) == 30, "episode ids are not unique")
    return output


def paired_outcomes(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in control["episode_records"]}
    right = {int(row["seed"]): row for row in candidate["episode_records"]}
    _require(left.keys() == right.keys(), "paired validation seeds differ")
    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for seed in sorted(left):
        control_outcome = _outcome(left[seed])
        candidate_outcome = _outcome(right[seed])
        transitions[f"{control_outcome}->{candidate_outcome}"] += 1
        rows.append(
            {
                "seed": seed,
                "traffic_variant": left[seed]["traffic_variant"],
                "control_outcome": control_outcome,
                "candidate_outcome": candidate_outcome,
                "control_decision_steps": left[seed]["decision_steps"],
                "candidate_decision_steps": right[seed]["decision_steps"],
            }
        )
    return {
        "paired_episodes": len(rows),
        "transition_counts": dict(sorted(transitions.items())),
        "control_success_candidate_collision": sum(
            row["control_outcome"] == "success"
            and row["candidate_outcome"] == "collision"
            for row in rows
        ),
        "control_non_success_candidate_success": sum(
            row["control_outcome"] != "success"
            and row["candidate_outcome"] == "success"
            for row in rows
        ),
        "per_episode": rows,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def action_group(
    rows: list[dict[str, Any]], episode_ids: set[int]
) -> dict[str, Any]:
    selected = [row for row in rows if int(row["episode"]) in episode_ids]
    valid_route = [row for row in selected if bool(row["pre_action_route_intent_valid"])]
    overrides = [
        row
        for row in selected
        if isinstance(row.get("fusion_decoder"), dict)
        and bool(row["fusion_decoder"].get("actor_override"))
    ]
    override_route = [
        row
        for row in overrides
        if bool(row["pre_action_route_intent_valid"])
        and int(row["fusion_decoder"]["actor_selected_lane"])
        == int(row["pre_action_route_intent"])
    ]
    keep_intent_override = [
        row
        for row in overrides
        if bool(row["pre_action_route_intent_valid"])
        and int(row["pre_action_route_intent"]) == 0
        and int(row["fusion_decoder"]["actor_selected_lane"]) != 0
    ]
    q_gaps_target: list[float] = []
    q_gaps_keep: list[float] = []
    for row in overrides:
        decoder = row["fusion_decoder"]
        q_values = decoder["minimum_target_twin_q"]
        actor_index = int(decoder["actor_selected_lane_index"])
        target_index = int(decoder["target_selected_lane_index"])
        actor_q = q_values[actor_index]
        target_q = q_values[target_index]
        keep_q = q_values[1]
        if actor_q is not None and target_q is not None:
            q_gaps_target.append(float(actor_q) - float(target_q))
        if actor_q is not None and keep_q is not None:
            q_gaps_keep.append(float(actor_q) - float(keep_q))
    return {
        "episodes": len(episode_ids),
        "decision_records": len(selected),
        "lane_change_applied_rate": _ratio(
            sum(bool(row["lane_change_applied"]) for row in selected), len(selected)
        ),
        "route_intent_match_rate": _ratio(
            sum(bool(row["route_intent_match"]) for row in valid_route), len(valid_route)
        ),
        "mean_speed_mps": (
            statistics.fmean(float(row["actual_speed_mps"]) for row in selected)
            if selected
            else None
        ),
        "actor_override_records": len(overrides),
        "actor_override_rate": _ratio(len(overrides), len(selected)),
        "actor_override_route_match_rate": _ratio(len(override_route), len(overrides)),
        "keep_intent_actor_non_keep_override_records": len(keep_intent_override),
        "keep_intent_actor_non_keep_override_rate": _ratio(
            len(keep_intent_override), len(overrides)
        ),
        "actor_q_minus_target_q": numeric_summary(q_gaps_target),
        "actor_q_minus_keep_q": numeric_summary(q_gaps_keep),
    }


def terminal_collision_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode"])].append(row)
    output: list[dict[str, Any]] = []
    for episode, episode_rows in grouped.items():
        terminal = max(episode_rows, key=lambda row: int(row["decision"]))
        if not bool(terminal.get("collision")):
            continue
        decoder = terminal.get("fusion_decoder")
        q_gap = None
        if isinstance(decoder, dict):
            q_values = decoder["minimum_target_twin_q"]
            actor_q = q_values[int(decoder["actor_selected_lane_index"])]
            target_q = q_values[int(decoder["target_selected_lane_index"])]
            if actor_q is not None and target_q is not None:
                q_gap = float(actor_q) - float(target_q)
        output.append(
            {
                "episode": episode,
                "seed": terminal["seed"],
                "traffic_variant": terminal["traffic_variant"],
                "decision": terminal["decision"],
                "edge": terminal["pre_action_current_edge"],
                "lane_index": terminal["pre_action_current_lane_index"],
                "speed_mps": terminal["actual_speed_mps"],
                "lane_command": terminal["lane_command"],
                "route_intent": terminal["pre_action_route_intent"],
                "route_intent_match": terminal["route_intent_match"],
                "selected_source": (
                    decoder.get("selected_source") if isinstance(decoder, dict) else None
                ),
                "actor_confidence": (
                    decoder.get("actor_selected_confidence")
                    if isinstance(decoder, dict)
                    else None
                ),
                "actor_q_minus_target_q": q_gap,
            }
        )
    return output


def _fusion_values(row: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
    decoder = row["fusion_decoder"]
    q_values = decoder["minimum_target_twin_q"]
    actor_q = q_values[int(decoder["actor_selected_lane_index"])]
    target_q = q_values[int(decoder["target_selected_lane_index"])]
    _require(actor_q is not None and target_q is not None, "selected Q is missing")
    return decoder, float(actor_q), float(target_q)


def guard_sweep(
    rows: list[dict[str, Any]], episode_outcomes: dict[int, str]
) -> list[dict[str, Any]]:
    rules: list[tuple[str, float | None, bool]] = []
    for threshold in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        rules.append((f"confidence_only_{threshold:.3f}", threshold, False))
    for tolerance in (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.25, 0.50, 1.0, 2.0):
        rules.append((f"critic_tolerance_{tolerance:.2f}", tolerance, False))
        rules.append((f"route_and_critic_tolerance_{tolerance:.2f}", tolerance, True))
        rules.append((f"keep_tie_veto_critic_tolerance_{tolerance:.3f}", tolerance, False))
    rules.append(("keep_tie_veto", None, False))
    terminal_by_episode: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode = int(row["episode"])
        if episode not in terminal_by_episode or int(row["decision"]) > int(
            terminal_by_episode[episode]["decision"]
        ):
            terminal_by_episode[episode] = row
    output: list[dict[str, Any]] = []
    for name, parameter, require_route in rules:
        kept = 0
        changed = 0
        changed_episodes: set[int] = set()
        changed_collision_episodes: set[int] = set()
        changed_success_episodes: set[int] = set()
        terminal_collision_interventions = 0
        terminal_success_interventions = 0
        counterfactual_route_matches = 0
        route_samples = 0
        for row in rows:
            decoder, actor_q, target_q = _fusion_values(row)
            actor_index = int(decoder["actor_selected_lane_index"])
            actor_lane = int(decoder["actor_selected_lane"])
            confidence = float(decoder["actor_selected_confidence"])
            original = bool(decoder["actor_override"])
            route_ok = bool(row["pre_action_route_intent_valid"]) and actor_lane == int(
                row["pre_action_route_intent"]
            )
            keep_is_tied = bool(decoder["keep_was_exact_tied_target_maximum"])
            if name.startswith("confidence_only"):
                use_actor = actor_index != 1 and confidence >= float(parameter)
            elif name == "keep_tie_veto":
                use_actor = original and not keep_is_tied
            elif name.startswith("keep_tie_veto_critic_tolerance"):
                use_actor = (
                    original
                    and not keep_is_tied
                    and actor_q >= target_q - float(parameter)
                )
            else:
                use_actor = original and actor_q >= target_q - float(parameter)
                if require_route:
                    use_actor = use_actor and route_ok
            selected_lane = (
                actor_lane if use_actor else int(decoder["target_selected_lane"])
            )
            if use_actor:
                kept += 1
            if use_actor != original:
                changed += 1
                episode = int(row["episode"])
                changed_episodes.add(episode)
                if episode_outcomes[episode] == "collision":
                    changed_collision_episodes.add(episode)
                if episode_outcomes[episode] == "success":
                    changed_success_episodes.add(episode)
                if terminal_by_episode[episode] is row:
                    if episode_outcomes[episode] == "collision":
                        terminal_collision_interventions += 1
                    if episode_outcomes[episode] == "success":
                        terminal_success_interventions += 1
            if bool(row["pre_action_route_intent_valid"]):
                route_samples += 1
                counterfactual_route_matches += selected_lane == int(
                    row["pre_action_route_intent"]
                )
        output.append(
            {
                "rule": name,
                "fixed_observation_only": True,
                "closed_loop_outcome_inference_forbidden": True,
                "counterfactual_actor_override_records": kept,
                "counterfactual_actor_override_rate": kept / len(rows),
                "decisions_changed_from_fusion_0_90": changed,
                "episodes_with_any_intervention": len(changed_episodes),
                "collision_episodes_with_any_intervention": len(
                    changed_collision_episodes
                ),
                "success_episodes_with_any_intervention": len(
                    changed_success_episodes
                ),
                "terminal_collision_decisions_intervened": terminal_collision_interventions,
                "terminal_success_decisions_intervened": terminal_success_interventions,
                "counterfactual_route_intent_match_rate": _ratio(
                    counterfactual_route_matches, route_samples
                ),
            }
        )
    return output


def _run_sources(run_name: str) -> dict[str, dict[str, str]]:
    root = RUNS / run_name
    paths = {
        "detailed": root / "paper_evaluation_detailed.json",
        "actions": root / "action_diagnostics.json",
        "decisions": root / "action_diagnostics_decisions.jsonl",
        "selector": root / "selector" / "receipt.json",
    }
    return {
        key: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for key, path in paths.items()
    }


def _calibration_matrix(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{row['checkpoint_kind']}__{row['deployment_decoder']}": row["summary"]
        for row in receipt["candidates"]
    }


def _selected_decoder_map(summary: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in summary["per_run"]:
        if row["method"] != "selected_v4_candidate":
            continue
        key = f"{row['scenario']}__seed{row['seed']}"
        output[key] = {
            "checkpoint": row["selected_checkpoint_kind"],
            "decoder": row["selected_deployment_decoder"],
            "success_rate": row["success_rate"],
            "collision_rate": row["collision_rate"],
            "timeout_rate": row["timeout_rate"],
        }
    return output


def build_attribution() -> dict[str, Any]:
    gate = _load(GATE)
    summary = _load(SUMMARY)
    reconciliation = _load(RECONCILIATION)
    _require(gate.get("decision") == "fail", "promotion gate is not failed")
    failed_checks = [key for key, value in gate["checks"].items() if not value]
    _require(
        failed_checks == ["per_scenario_collision_noninferiority"],
        f"unexpected failed checks: {failed_checks}",
    )
    _require(gate.get("formal_test_unlocked") is False, "formal is unlocked")
    _require(reconciliation.get("formal_accepted") == 0, "formal was accessed")
    _require(_sha256(SUMMARY) == gate["promotion_results_sha256"], "summary hash drifted")

    details: dict[tuple[str, int], dict[str, Any]] = {}
    traces: dict[tuple[str, int], list[dict[str, Any]]] = {}
    selectors: dict[int, dict[str, Any]] = {}
    source_hashes: dict[str, Any] = {}
    for key, run_name in CROSS_RUNS.items():
        run = RUNS / run_name
        details[key] = _load(run / "paper_evaluation_detailed.json")
        traces[key] = _load_jsonl(run / "action_diagnostics_decisions.jsonl")
        if key[0] == "candidate":
            selectors[key[1]] = _load(run / "selector" / "receipt.json")
        source_hashes[run_name] = _run_sources(run_name)

    per_seed: dict[str, Any] = {}
    candidate_mechanism: dict[str, Any] = {}
    for seed in (0, 1):
        control_detail = details[("temporal_graph", seed)]
        candidate_detail = details[("candidate", seed)]
        candidate_episodes = _episode_map(candidate_detail)
        outcomes = {episode: _outcome(row) for episode, row in candidate_episodes.items()}
        collision_ids = {episode for episode, outcome in outcomes.items() if outcome == "collision"}
        success_ids = {episode for episode, outcome in outcomes.items() if outcome == "success"}
        candidate_rows = traces[("candidate", seed)]
        control_rows = traces[("temporal_graph", seed)]
        per_seed[f"seed{seed}"] = paired_outcomes(control_detail, candidate_detail)
        candidate_mechanism[f"seed{seed}"] = {
            "selected_pair": {
                "checkpoint": selectors[seed]["selected_checkpoint_kind"],
                "decoder": selectors[seed]["selected_deployment_decoder"],
            },
            "calibration_matrix": _calibration_matrix(selectors[seed]),
            "candidate_all_episodes": action_group(
                candidate_rows, set(candidate_episodes)
            ),
            "candidate_collision_episodes": action_group(
                candidate_rows, collision_ids
            ),
            "candidate_success_episodes": action_group(candidate_rows, success_ids),
            "temporal_graph_all_episodes": action_group(
                control_rows, set(_episode_map(control_detail))
            ),
            "terminal_collision_records": terminal_collision_records(candidate_rows),
            "fixed_observation_guard_sweep": guard_sweep(candidate_rows, outcomes),
        }

    payload = {
        "schema_version": "topo-scene-v4.8.promotion-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "formal_test_accessed": False,
        "promotion_gate": {
            "decision": gate["decision"],
            "failed_checks": failed_checks,
            "per_scenario_success_delta": gate["per_scenario_success_delta"],
            "per_scenario_collision_delta": gate["per_scenario_collision_delta"],
            "macro_success_delta": gate["macro_success_delta"],
            "worst_paired_seed_success_delta": gate[
                "worst_paired_seed_success_delta"
            ],
        },
        "decoder_outcome_localization": _selected_decoder_map(summary),
        "cross_paired_outcomes": per_seed,
        "cross_candidate_mechanism": candidate_mechanism,
        "primary_attribution": {
            "mechanism": (
                "cross-only fusion deployment permits confidence-only actor "
                "non-keep overrides without critic-Q or route-consistency veto; "
                "the selected train calibration pair does not generalize its "
                "collision advantage to validation"
            ),
            "confidence": "trace-supported hypothesis requiring closed-loop intervention",
            "known": [
                "the only failed promotion check is cross collision noninferiority",
                "both cross candidate runs selected fusion_0_90",
                "roundabout_medium and carla selected target_critic and improved or held collision",
                "fusion actor overrides ignore target-critic Q in the deployed rule",
            ],
            "not_yet_causal": [
                "offline fixed-observation guard sweeps cannot infer closed-loop outcomes",
                "target-critic or guarded-fusion replay on the same checkpoint is required",
            ],
        },
        "selected_next_hypothesis": {
            "version": "v4.9_critic_route_consistent_fusion",
            "single_change": (
                "replace confidence-only actor override with a conservative "
                "actor override requiring confidence, route-intent agreement, "
                "and bounded target-critic Q regret"
            ),
            "training_changed": False,
            "return_estimator_changed": False,
            "checkpoint_candidates_changed": False,
            "selector_partition_changed": False,
            "decoder_inference_rule_changed": True,
            "fresh_development_required": True,
            "fresh_promotion_validation_required": True,
            "formal_test_eligible": False,
        },
        "source_hashes": {
            "promotion_gate": {"path": str(GATE.resolve()), "sha256": _sha256(GATE)},
            "promotion_summary": {
                "path": str(SUMMARY.resolve()),
                "sha256": _sha256(SUMMARY),
            },
            "v3_reconciliation": {
                "path": str(RECONCILIATION.resolve()),
                "sha256": _sha256(RECONCILIATION),
            },
            "cross_runs": source_hashes,
        },
    }
    return payload


def render_report(payload: dict[str, Any]) -> str:
    gate = payload["promotion_gate"]
    mechanism = payload["cross_candidate_mechanism"]
    lines = [
        "# v4.8 Promotion Failure Attribution",
        "",
        "Status: post-hoc diagnostic only; no formal-test access and no gate reuse.",
        "",
        "## Gate outcome",
        "",
        f"- Decision: `{gate['decision']}`.",
        "- Sole failed check: `per_scenario_collision_noninferiority`.",
        f"- Cross success delta: {gate['per_scenario_success_delta']['cross']:.4f}.",
        f"- Cross collision delta: {gate['per_scenario_collision_delta']['cross']:.4f}.",
        f"- Macro success delta: {gate['macro_success_delta']:.4f}.",
        "",
        "## Evidence-bound mechanism",
        "",
    ]
    for seed in (0, 1):
        row = mechanism[f"seed{seed}"]
        candidate = row["candidate_all_episodes"]
        control = row["temporal_graph_all_episodes"]
        collision = row["candidate_collision_episodes"]
        lines.extend(
            [
                f"- Seed {seed}: selected `{row['selected_pair']['checkpoint']}__{row['selected_pair']['decoder']}`; "
                f"candidate lane-change rate {candidate['lane_change_applied_rate']:.4f} versus TemporalGraph "
                f"{control['lane_change_applied_rate']:.4f}; actor override rate {candidate['actor_override_rate']:.4f}; "
                f"collision-episode override rate {collision['actor_override_rate']:.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            "Both Cross seeds selected the confidence-only fusion decoder. Roundabout-medium and CARLA selected target-critic. "
            "The localization and traces support a decoder-generalization hypothesis, but fixed-observation sweeps are not causal closed-loop results.",
            "",
            "## Next hypothesis",
            "",
            "Create `v4.9_critic_route_consistent_fusion`: keep training, return estimator, checkpoints, selector partition, rewards, and environments fixed; "
            "change only deterministic deployment so an actor non-keep override must pass confidence, route-intent agreement, and bounded critic-Q regret.",
            "First run same-checkpoint post-hoc closed-loop attribution, then preregister fresh development seeds. Promotion and formal results from v4.8 must not be reused as v4.9 gate inputs.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = build_attribution()
    output = args.output.resolve()
    report = args.report.resolve()
    _write_json(output, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256(output),
                "report": str(report),
                "report_sha256": _sha256(report),
                "failed_checks": payload["promotion_gate"]["failed_checks"],
                "next_hypothesis": payload["selected_next_hypothesis"]["version"],
                "formal_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
