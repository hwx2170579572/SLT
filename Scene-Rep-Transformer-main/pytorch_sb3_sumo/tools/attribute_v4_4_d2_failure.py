"""Seal the post-hoc causal attribution for the failed v4.4 D2 cell.

The script reads frozen v4.4 development artifacts and decoder-only closed-loop
replays.  It never edits the failed run, never counts post-hoc outcomes toward a
development gate, and never accesses the formal-test partition.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V44_ROOT = ROOT / "results_topo_v4_4_dev"
DEVELOPMENT = V44_ROOT / "development"
ATTRIBUTION = V44_ROOT / "attribution" / "d2_failure"
D1_RUN = DEVELOPMENT / "runs" / "D1__cand__carla__s3__pa80dbb24"
D2_RUN = DEVELOPMENT / "runs" / "D2__cand__cross__s2__pa80dbb24"
CLOSED_LOOP = ATTRIBUTION / "closed_loop"
CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_4.yaml"
FREEZE = V44_ROOT / "engineering" / "implementation_freeze.json"
DECISION = DEVELOPMENT / "development_decision.json"

EXPECTED_CONTRACT_SHA256 = (
    "a80dbb24e549cc7835606eac574780fab8170d4dcbbd8c68ee1be3344e5ea32d"
)
EXPECTED_FREEZE_SHA256 = (
    "8626ba1ddd5588a56287f57b032e23093a4214018bd63857d3223b2d5292510f"
)
EXPECTED_DECISION_SHA256 = (
    "0ff6ed286106edd8865ed039b88a2cd535fa02e4415a4c239884b89232564d21"
)
FUSION_THRESHOLD = 0.90


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"missing or invalid records in {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _outcome_counts(summary: dict[str, Any]) -> dict[str, int]:
    episodes = int(summary["episodes"])
    output = {"episodes": episodes}
    for name in ("success", "collision", "off_route", "timeout"):
        raw = float(summary[f"{name}_rate"]) * episodes
        count = round(raw)
        _require(abs(raw - count) <= 1e-9, f"non-integral {name} count")
        output[f"{name}_count"] = int(count)
    _require(
        sum(value for key, value in output.items() if key.endswith("_count"))
        == episodes,
        "terminal outcomes do not partition episodes",
    )
    return output


def _validate_replay(
    directory: Path,
    *,
    checkpoint_sha256: str,
    partition: str,
    seed_start: int,
    decoder: str = "fusion",
) -> dict[str, Any]:
    result_path = directory / "attribution_result.json"
    result = _load(result_path)
    expected_flags = {
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
    }
    for key, value in expected_flags.items():
        _require(result.get(key) is value, f"{directory.name}: {key} drifted")
    _require(result.get("decoder") == decoder, f"{directory.name}: decoder drifted")
    _require(
        result.get("checkpoint_sha256") == checkpoint_sha256,
        f"{directory.name}: checkpoint hash drifted",
    )
    _require(
        result.get("traffic_partition") == partition,
        f"{directory.name}: traffic partition drifted",
    )
    _require(
        int(result.get("evaluation_seed_start")) == seed_start,
        f"{directory.name}: seed block drifted",
    )
    _require(
        int(result.get("evaluation_episodes")) == 12,
        f"{directory.name}: episode count drifted",
    )
    if decoder == "fusion":
        _require(
            math.isclose(
                float(result.get("actor_non_keep_confidence_threshold")),
                FUSION_THRESHOLD,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"{directory.name}: fusion threshold drifted",
        )
    result["source"] = _source(result_path)
    return result


def _paired_success_effect(
    control_diagnostics: dict[str, Any], treatment_diagnostics: dict[str, Any]
) -> dict[str, Any]:
    control = control_diagnostics["per_episode"]
    treatment = treatment_diagnostics["per_episode"]
    _require(len(control) == len(treatment) == 12, "paired episode count drifted")
    transitions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for left, right in zip(control, treatment):
        signature_left = (
            int(left["episode"]),
            int(left["seed"]),
            left.get("traffic_variant"),
        )
        signature_right = (
            int(right["episode"]),
            int(right["seed"]),
            right.get("traffic_variant"),
        )
        _require(signature_left == signature_right, "paired traffic signature drifted")
        key = f"{int(bool(left['success']))}_to_{int(bool(right['success']))}"
        transitions[key] += 1
        records.append(
            {
                "episode": signature_left[0],
                "seed": signature_left[1],
                "traffic_variant": signature_left[2],
                "target_success": bool(left["success"]),
                "fusion_success": bool(right["success"]),
                "target_collision": bool(left["collision"]),
                "fusion_collision": bool(right["collision"]),
                "target_timeout": bool(left["timeout"]),
                "fusion_timeout": bool(right["timeout"]),
            }
        )
    improved = transitions["0_to_1"]
    regressed = transitions["1_to_0"]
    discordant = improved + regressed
    if discordant == 0:
        sign_test = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(improved, regressed) + 1)
        ) / (2**discordant)
        sign_test = min(1.0, 2.0 * tail)
    return {
        "paired_episodes": 12,
        "transition_counts": dict(sorted(transitions.items())),
        "improved_count": improved,
        "regressed_count": regressed,
        "exact_two_sided_paired_sign_p_value": sign_test,
        "statistical_role": "post_hoc_support_only_not_a_preregistered_gate",
        "per_episode": records,
    }


def _terminal_collision_edges(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode"])].append(row)
    edges: Counter[str] = Counter()
    for episode_rows in by_episode.values():
        terminal = max(episode_rows, key=lambda row: int(row["decision"]))
        if bool(terminal["collision"]):
            edges[str(terminal.get("pre_action_current_edge"))] += 1
    return dict(edges.most_common())


def _fixed_rule(
    trace_summary: dict[str, Any], rule: str, threshold: float | None = None
) -> dict[str, Any]:
    candidates = trace_summary["fixed_observation_decoder_sweep"]
    for row in candidates:
        if row["rule"] != rule:
            continue
        received = row.get("actor_non_keep_confidence_threshold")
        if threshold is None and received is None:
            return row
        if threshold is not None and received is not None and math.isclose(
            float(received), threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            return row
    raise ValueError(f"missing fixed-observation rule {rule}@{threshold}")


def _markdown(payload: dict[str, Any]) -> str:
    causal = payload["decoder_only_paired_intervention"]
    target = causal["control_target_decoder_outcomes"]
    fusion = causal["treatment_fusion_decoder_outcomes"]
    continuity = payload["open_loop_fixed_observation_continuity"]
    selection = payload["fusion_train_only_selector_counterfactual"]
    d1 = payload["carla_preservation_check"]
    return f"""# v4.4 D2 Deep Attribution

Status: post-hoc diagnosis only. No result in this report is counted toward a
v4.4 or v4.5 development gate, and the formal-test partition remains untouched.

## Failure and paired intervention

The frozen v4.4 state machine stopped after D2 (Cross, training seed 2). Its
selected best checkpoint used the target-twin-Q decoder and achieved
{target['success_rate']:.3f} success, {target['collision_rate']:.3f} collision,
and {target['timeout_rate']:.3f} timeout on validation seeds 56000--56011.

Changing only the deterministic deployment decoder on the exact same checkpoint,
traffic variants, and seeds to the 0.90-confidence fusion rule changed outcomes
to {fusion['success_rate']:.3f} success, {fusion['collision_rate']:.3f} collision,
and {fusion['timeout_rate']:.3f} timeout. Of 12 paired episodes,
{causal['paired_success_effect']['improved_count']} changed from failure to success
and {causal['paired_success_effect']['regressed_count']} changed from success to
failure (post-hoc exact paired sign p={causal['paired_success_effect']['exact_two_sided_paired_sign_p_value']:.9f}).

The actor-only and fusion replays have identical D2 best-checkpoint outcome and
decision-trace hashes. This identifies the target-critic lateral arbitration as
the active failure mechanism on these route windows, not a difference in weights,
training data, environment, traffic, or evaluation seeds.

## Mechanistic trace

On the target-decoder rollout, route-window match was
{continuity['target']['route_window_match_rate']:.3f}, and the mean maximum
contiguous matching request was
{continuity['target']['mean_maximum_route_match_streak']:.2f} decisions
({continuity['target_mean_maximum_route_match_seconds']:.3f} s) against SUMO's
3.0 s nominal lane-change duration. Only
{continuity['target']['episodes_with_estimated_full_duration_streak']}/12 episodes
reached that request duration. On fixed observations, the actor and 0.90 fusion
both raise route-window match to {continuity['fusion_0_90']['route_window_match_rate']:.3f}
and the mean maximum streak to
{continuity['fusion_0_90']['mean_maximum_route_match_streak']:.2f} decisions.
These fixed-observation values diagnose continuity only; the real closed-loop
results above provide the outcome evidence.

## Selector and preservation checks

Under fusion, both D2 best and final checkpoints score 12/12 success with zero
adverse outcomes on paired training seeds 65000--65011. The already frozen
v4.4 lexicographic selector therefore chooses `{selection['selected_checkpoint_kind']}`
by its exact-final tie preference. The selected final checkpoint then scores
12/12 success with zero adverse outcomes on validation seeds 56000--56011.

On D1 CARLA, applying fusion to the same final checkpoint and validation block
preserves {d1['fusion_outcomes']['success_rate']:.3f} success and zero adverse
outcomes; all lane-command rates equal the target-decoder run. This is a narrow
preservation check, not a fresh development result.

## v4.5 single-change recommendation

Freeze one deterministic deployment rule: use the actor's feasible non-keep
argmax only when its probability is at least 0.90; otherwise retain the existing
keep-tie-aware minimum-target-twin-Q choice. Keep training, representations,
reward, environment, entropy, candidate checkpoints, selector ordering, and all
promotion/formal protocols unchanged. The 0.90 threshold was selected post hoc,
so v4.5 must use new training and traffic seed blocks and must not count any
evidence in this report toward its gates.
"""


def main() -> int:
    from tools.attribute_v4_3_t3_failure import _run_summary
    from tools.checkpoint_selector_v4_4 import SELECTION_ORDER, selection_key

    _require(_sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256, "v4.4 contract drifted")
    _require(_sha256(FREEZE) == EXPECTED_FREEZE_SHA256, "v4.4 freeze drifted")
    _require(_sha256(DECISION) == EXPECTED_DECISION_SHA256, "v4.4 decision drifted")
    decision = _load(DECISION)
    _require(
        decision.get("decision") == "fail"
        and decision.get("stopped_after") == "D2"
        and decision.get("next_job") is None,
        "v4.4 state-machine failure receipt drifted",
    )
    _require(
        decision.get("implementation_freeze_sha256") == EXPECTED_FREEZE_SHA256,
        "development decision is not bound to the expected freeze",
    )

    d1_row = decision["jobs"]["D1"]["row"]
    d2_row = decision["jobs"]["D2"]["row"]
    _require(decision["jobs"]["D2"]["gate"]["selector_passed"] is True, "D2 selector gate did not pass")
    _require(decision["jobs"]["D2"]["gate"]["outcome_passed"] is False, "D2 outcome gate did not fail")
    d1_hash = str(d1_row["selected_model_sha256"])
    d2_best_hash = str(d2_row["selected_model_sha256"])
    d2_final_hash = str(d2_row["final_checkpoint_sha256"])

    d2_trace_summary = _run_summary(D2_RUN)
    target_rule = _fixed_rule(d2_trace_summary, "target")
    actor_rule = _fixed_rule(d2_trace_summary, "actor")
    fusion_rule = _fixed_rule(d2_trace_summary, "fusion", FUSION_THRESHOLD)

    actor_best_validation = _validate_replay(
        CLOSED_LOOP / "best_actor_validation",
        checkpoint_sha256=d2_best_hash,
        partition="validation",
        seed_start=56_000,
        decoder="actor",
    )
    d1_fusion_validation = _validate_replay(
        CLOSED_LOOP / "d1_final_fusion_validation",
        checkpoint_sha256=d1_hash,
        partition="validation",
        seed_start=55_000,
    )
    best_fusion_train = _validate_replay(
        CLOSED_LOOP / "d2_best_fusion_train",
        checkpoint_sha256=d2_best_hash,
        partition="train",
        seed_start=65_000,
    )
    best_fusion_validation = _validate_replay(
        CLOSED_LOOP / "d2_best_fusion_validation",
        checkpoint_sha256=d2_best_hash,
        partition="validation",
        seed_start=56_000,
    )
    final_fusion_train = _validate_replay(
        CLOSED_LOOP / "d2_final_fusion_train",
        checkpoint_sha256=d2_final_hash,
        partition="train",
        seed_start=65_000,
    )
    final_fusion_validation = _validate_replay(
        CLOSED_LOOP / "d2_final_fusion_validation",
        checkpoint_sha256=d2_final_hash,
        partition="validation",
        seed_start=56_000,
    )

    _require(
        actor_best_validation["outcomes"] == best_fusion_validation["outcomes"],
        "actor and fusion closed-loop outcomes diverged",
    )
    _require(
        actor_best_validation["trace_sha256"] == best_fusion_validation["trace_sha256"],
        "actor and fusion closed-loop traces diverged",
    )
    _require(
        d1_fusion_validation["outcomes"] == _load(D1_RUN / "final_evaluation.json"),
        "fusion did not preserve D1 outcomes",
    )
    d1_target_actions = _load(D1_RUN / "action_diagnostics.json")
    _require(
        d1_fusion_validation["lane_command_rates"]
        == d1_target_actions["lane_command_rates"],
        "fusion did not preserve D1 lane-command rates",
    )

    best_key = selection_key(
        best_fusion_train["outcomes"], checkpoint_kind="highest_training_success"
    )
    final_key = selection_key(
        final_fusion_train["outcomes"], checkpoint_kind="exact_final"
    )
    _require(final_key > best_key, "fusion calibration no longer selects exact final")
    _require(
        _outcome_counts(best_fusion_train["outcomes"])["success_count"] == 12
        and _outcome_counts(final_fusion_train["outcomes"])["success_count"] == 12,
        "fusion calibration was not a complete outcome tie",
    )

    target_diagnostics = _load(D2_RUN / "action_diagnostics.json")
    fusion_diagnostics = _load(
        CLOSED_LOOP / "d2_best_fusion_validation" / "action_diagnostics.json"
    )
    paired = _paired_success_effect(target_diagnostics, fusion_diagnostics)
    target_rows = _load_jsonl(D2_RUN / "action_diagnostics_decisions.jsonl")

    payload = {
        "schema_version": "topo-scene-v4.4.d2-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "formal_test_accessed": False,
        "failed_job": "D2__cand__cross__s2__pa80dbb24",
        "failure_gate": {
            "selector_passed": True,
            "outcome_passed": False,
            "development_decision": "fail",
            "stopped_after": "D2",
            "D3_and_D4_executed": False,
        },
        "primary_attribution": {
            "mechanism": "target_critic_lateral_arbitration_breaks_route_command_continuity",
            "causal_intervention": "decoder_only_same_checkpoint_same_paired_validation_traffic",
            "confidence": "strong_post_hoc_causal_evidence_requires_fresh_v4_5_confirmation",
            "rejected_as_primary": [
                "checkpoint_selector_integrity_failure",
                "unavailable_lane_change_actuation",
                "environment_or_traffic_seed_difference",
                "training_weight_difference",
            ],
        },
        "decoder_only_paired_intervention": {
            "checkpoint_kind": "highest_training_success",
            "checkpoint_sha256": d2_best_hash,
            "scenario": "cross",
            "validation_seed_start": 56_000,
            "episodes": 12,
            "control_decoder": "argmax_feasible_min_target_twin_q_keep_tie",
            "treatment_decoder": "actor_confident_non_keep_else_target_critic",
            "actor_non_keep_confidence_threshold": FUSION_THRESHOLD,
            "control_target_decoder_outcomes": d2_row,
            "treatment_fusion_decoder_outcomes": best_fusion_validation["outcomes"],
            "success_rate_delta": (
                float(best_fusion_validation["outcomes"]["success_rate"])
                - float(d2_row["success_rate"])
            ),
            "collision_rate_delta": (
                float(best_fusion_validation["outcomes"]["collision_rate"])
                - float(d2_row["collision_rate"])
            ),
            "timeout_rate_delta": (
                float(best_fusion_validation["outcomes"]["timeout_rate"])
                - float(d2_row["timeout_rate"])
            ),
            "paired_success_effect": paired,
            "actor_and_fusion_outcomes_identical": True,
            "actor_and_fusion_trace_sha256": best_fusion_validation["trace_sha256"],
        },
        "open_loop_fixed_observation_continuity": {
            "interpretation_boundary": (
                "decoder recomputation on target-rollout observations; no closed-loop "
                "outcome is inferred from this subsection"
            ),
            "controller_period_seconds": 0.3,
            "sumo_nominal_lane_change_duration_seconds": 3.0,
            "target": target_rule,
            "actor": actor_rule,
            "fusion_0_90": fusion_rule,
            "target_mean_maximum_route_match_seconds": (
                float(target_rule["mean_maximum_route_match_streak"]) * 0.3
            ),
            "fusion_mean_maximum_route_match_seconds": (
                float(fusion_rule["mean_maximum_route_match_streak"]) * 0.3
            ),
            "target_collision_terminal_edges": _terminal_collision_edges(target_rows),
            "failed_episode_target_lane_observed_rate": d2_trace_summary[
                "continuity_by_outcome"
            ]["failure_target_lane_observed_rate"],
            "failed_episode_next_route_edge_observed_rate": d2_trace_summary[
                "continuity_by_outcome"
            ]["failure_next_route_edge_observed_rate"],
        },
        "fusion_train_only_selector_counterfactual": {
            "post_hoc_only": True,
            "calibration_partition": "train",
            "calibration_seed_start": 65_000,
            "episodes_per_checkpoint": 12,
            "selection_order": list(SELECTION_ORDER),
            "highest_training_success": {
                "checkpoint_sha256": d2_best_hash,
                "outcomes": best_fusion_train["outcomes"],
                "selection_key": list(best_key),
            },
            "exact_final": {
                "checkpoint_sha256": d2_final_hash,
                "outcomes": final_fusion_train["outcomes"],
                "selection_key": list(final_key),
            },
            "selected_checkpoint_kind": "exact_final",
            "selected_checkpoint_validation_outcomes": final_fusion_validation[
                "outcomes"
            ],
        },
        "carla_preservation_check": {
            "post_hoc_only": True,
            "checkpoint_kind": "exact_final",
            "checkpoint_sha256": d1_hash,
            "validation_seed_start": 55_000,
            "target_outcomes": _load(D1_RUN / "final_evaluation.json"),
            "fusion_outcomes": d1_fusion_validation["outcomes"],
            "lane_command_rates_equal": True,
        },
        "recommended_single_change": {
            "version": "v4.5",
            "name": "actor_confident_non_keep_else_target_critic",
            "actor_non_keep_confidence_threshold": FUSION_THRESHOLD,
            "actor_override_requires_non_keep": True,
            "actor_choice_remains_action_mask_feasible": True,
            "fallback": "keep_tie_aware_argmax_feasible_min_target_twin_q",
            "training_changed": False,
            "representations_changed": False,
            "reward_changed": False,
            "environment_changed": False,
            "checkpoint_selector_changed": False,
            "post_hoc_threshold_selection_declared": True,
            "fresh_preregistered_development_required": True,
        },
        "source_hashes": {
            "experiment_contract": _source(CONTRACT),
            "implementation_freeze": _source(FREEZE),
            "development_decision": _source(DECISION),
            "d1_target_evaluation": _source(D1_RUN / "final_evaluation.json"),
            "d1_target_actions": _source(D1_RUN / "action_diagnostics.json"),
            "d2_target_trace": _source(D2_RUN / "action_diagnostics_decisions.jsonl"),
            "d2_target_actions": _source(D2_RUN / "action_diagnostics.json"),
            "d2_target_evaluation": _source(D2_RUN / "final_evaluation.json"),
            "best_actor_validation": actor_best_validation["source"],
            "d1_final_fusion_validation": d1_fusion_validation["source"],
            "d2_best_fusion_train": best_fusion_train["source"],
            "d2_best_fusion_validation": best_fusion_validation["source"],
            "d2_final_fusion_train": final_fusion_train["source"],
            "d2_final_fusion_validation": final_fusion_validation["source"],
        },
    }
    summary_path = ATTRIBUTION / "attribution_summary.json"
    markdown_path = ATTRIBUTION / "deep_attribution.md"
    _write_json(summary_path, payload)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "attribution_summary": str(summary_path.resolve()),
                "attribution_summary_sha256": _sha256(summary_path),
                "deep_attribution": str(markdown_path.resolve()),
                "deep_attribution_sha256": _sha256(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
