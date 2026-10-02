"""Seal the post-hoc attribution for the failed v4.5 E1 development cell.

This program only reads frozen v4.5/v4.5.1 artifacts and explicitly post-hoc
closed-loop decoder replays.  It writes a versioned attribution report; none of
its outcomes are eligible for a development or formal-test gate.
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

V45_ROOT = ROOT / "results_topo_v4_5_dev"
DEVELOPMENT = V45_ROOT / "development"
RUN = DEVELOPMENT / "runs" / "E1__cand__cross__s4__p3c1f7f8b"
ATTRIBUTION = V45_ROOT / "attribution" / "e1_failure"
CLOSED_LOOP = ATTRIBUTION / "closed_loop"
CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_5.yaml"
FREEZE = V45_ROOT / "engineering" / "implementation_freeze.json"
PATCH_FREEZE = V45_ROOT / "engineering" / "v4_5_1" / "implementation_freeze.json"
DECISION = DEVELOPMENT / "development_decision.json"
V44_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_4_dev"
    / "attribution"
    / "d2_failure"
    / "attribution_summary.json"
)

EXPECTED_CONTRACT_SHA256 = (
    "3c1f7f8bb2f2967f7559907386b58e67760c53dfa8e46bf581fc4824d186bb7c"
)
EXPECTED_FREEZE_SHA256 = (
    "9ab0ccc71b33110ac526f5a9f89e8f504d0f3fc0ddafd6407c6ff36dc2414666"
)
EXPECTED_PATCH_FREEZE_SHA256 = (
    "2dc362f52b80c03162d3b4875f4c667ebc976af5fd128b5c9680a0c15cea3cea"
)
EXPECTED_DECISION_SHA256 = (
    "3e6e2d2c403db0784b7a2c67a8205bce47fc8e7cd7765f265a0039584540a0c3"
)
EXPECTED_V44_ATTRIBUTION_SHA256 = (
    "87090b8c67650c6debe62a6ea69990875e1eef2f7476326baf356b0c6155136c"
)
BEST_SHA256 = "5f324d6f68a46eea527983b589d89960ba72fe40a9970521403d12665be125d6"
FINAL_SHA256 = "2d9afa0420fd78e6dec7b60f8b5fb53f5139ccdbb63affe8bd3a29fcc30b7519"
FUSION_THRESHOLD = 0.90
EVENTS = ("success", "collision", "off_route", "timeout")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"missing or invalid JSONL records in {path}")
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_replay(
    name: str,
    *,
    checkpoint_sha256: str,
    decoder: str,
    partition: str,
    seed_start: int,
) -> dict[str, Any]:
    directory = CLOSED_LOOP / name
    result_path = directory / "attribution_result.json"
    result = _load(result_path)
    expected = {
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
    }
    for key, value in expected.items():
        _require(result.get(key) is value, f"{name}: {key} drifted")
    _require(result.get("checkpoint_sha256") == checkpoint_sha256, f"{name}: checkpoint drifted")
    _require(result.get("decoder") == decoder, f"{name}: decoder drifted")
    _require(result.get("traffic_partition") == partition, f"{name}: partition drifted")
    _require(int(result.get("evaluation_seed_start")) == seed_start, f"{name}: seeds drifted")
    _require(int(result.get("evaluation_episodes")) == 12, f"{name}: episode count drifted")
    if decoder == "fusion":
        _require(
            math.isclose(
                float(result.get("actor_non_keep_confidence_threshold")),
                FUSION_THRESHOLD,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"{name}: fusion threshold drifted",
        )
    diagnostics_path = directory / "action_diagnostics.json"
    diagnostics = _load(diagnostics_path)
    _require(
        diagnostics.get("outcomes") == result.get("outcomes"),
        f"{name}: diagnostic outcomes drifted",
    )
    result["source"] = _source(result_path)
    result["diagnostics_source"] = _source(diagnostics_path)
    result["diagnostics"] = diagnostics
    return result


def _paired_effect(
    control: dict[str, Any], treatment: dict[str, Any], *, control_name: str, treatment_name: str
) -> dict[str, Any]:
    left = control["per_episode"]
    right = treatment["per_episode"]
    _require(len(left) == len(right) == 12, "paired episode count drifted")
    transitions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for control_row, treatment_row in zip(left, right):
        control_signature = (
            int(control_row["episode"]),
            int(control_row["seed"]),
            control_row.get("traffic_variant"),
        )
        treatment_signature = (
            int(treatment_row["episode"]),
            int(treatment_row["seed"]),
            treatment_row.get("traffic_variant"),
        )
        _require(control_signature == treatment_signature, "paired traffic signature drifted")
        key = f"{int(bool(control_row['success']))}_to_{int(bool(treatment_row['success']))}"
        transitions[key] += 1
        records.append(
            {
                "episode": control_signature[0],
                "seed": control_signature[1],
                "traffic_variant": control_signature[2],
                f"{control_name}_success": bool(control_row["success"]),
                f"{treatment_name}_success": bool(treatment_row["success"]),
                f"{control_name}_collision": bool(control_row["collision"]),
                f"{treatment_name}_collision": bool(treatment_row["collision"]),
                f"{control_name}_timeout": bool(control_row["timeout"]),
                f"{treatment_name}_timeout": bool(treatment_row["timeout"]),
            }
        )
    improved = transitions["0_to_1"]
    regressed = transitions["1_to_0"]
    discordant = improved + regressed
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(improved, regressed) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "paired_episodes": 12,
        "transition_counts": dict(sorted(transitions.items())),
        "improved_count": improved,
        "regressed_count": regressed,
        "exact_two_sided_paired_sign_p_value": p_value,
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
        if bool(terminal.get("collision")):
            edges[str(terminal.get("pre_action_current_edge"))] += 1
    return dict(edges.most_common())


def _threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(rows)
    for threshold in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        count = sum(
            int(
                int(row["fusion_decoder"]["actor_selected_lane"]) != 0
                and float(row["fusion_decoder"]["actor_selected_confidence"])
                >= threshold
            )
            for row in rows
        )
        output.append(
            {
                "threshold": threshold,
                "counterfactual_actor_override_records_on_fixed_observations": count,
                "counterfactual_actor_override_rate_on_fixed_observations": count / total,
            }
        )
    return output


def _joint_key(summary: dict[str, Any], checkpoint: str, decoder: str) -> tuple[Any, ...]:
    from tools.checkpoint_selector_v4_5_1 import event_flag_counts

    counts = event_flag_counts(summary)
    return (
        counts["success_count"],
        -counts["collision_count"],
        -counts["off_route_count"],
        -counts["timeout_count"],
        float(summary["mean_return"]),
        int(checkpoint == "exact_final"),
        int(decoder == "target_critic"),
    )


def _joint_matrix(
    best_fusion: dict[str, Any],
    final_fusion: dict[str, Any],
    best_target: dict[str, Any],
    final_target: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        ("highest_training_success", "fusion_0_90", best_fusion),
        ("exact_final", "fusion_0_90", final_fusion),
        ("highest_training_success", "target_critic", best_target),
        ("exact_final", "target_critic", final_target),
    ]
    candidates = [
        {
            "checkpoint_kind": checkpoint,
            "decoder": decoder,
            "outcomes": summary,
            "selection_key": list(_joint_key(summary, checkpoint, decoder)),
        }
        for checkpoint, decoder, summary in rows
    ]
    selected = max(candidates, key=lambda row: tuple(row["selection_key"]))
    return {
        "post_hoc_only": True,
        "calibration_partition": "train",
        "calibration_seed_start": 68_000,
        "paired_episodes_per_candidate": 12,
        "selection_order": [
            "maximize_success_count",
            "minimize_collision_count",
            "minimize_off_route_count",
            "minimize_timeout_count",
            "maximize_mean_return",
            "prefer_exact_final_on_complete_outcome_tie",
            "prefer_target_critic_on_complete_checkpoint_decoder_tie",
        ],
        "candidates": candidates,
        "selected_checkpoint_kind": selected["checkpoint_kind"],
        "selected_decoder": selected["decoder"],
        "selected_key": selected["selection_key"],
        "validation_used_for_selection": False,
        "formal_test_used_for_selection": False,
    }


def _markdown(payload: dict[str, Any]) -> str:
    clean = payload["decoder_only_same_checkpoint_validation_intervention"]
    fusion = clean["fusion_outcomes"]
    target = clean["target_outcomes"]
    selected = payload["joint_train_only_selector_counterfactual"]
    selected_validation = payload["joint_selector_selected_pair_validation_check"]["outcomes"]
    historical = payload["cross_seed_heterogeneity"]
    trace = payload["trace_mechanism"]
    return f"""# v4.5 E1 Deep Attribution

Status: post-hoc diagnosis only. No value in this report counts toward a v4.5
or v4.6 development gate. The formal-test partition was not accessed.

## Frozen failure

The preregistered v4.5 state machine stopped after E1 (Cross, training seed 4).
Its selector and decoder-integrity gates passed, but the selected fusion policy
achieved {fusion['success_rate']:.3f} success, {fusion['collision_rate']:.3f}
collision, and {fusion['timeout_rate']:.3f} timeout on validation seeds
59000--59011, failing both the success and collision guards.

## Decoder-only intervention

On the exact same selected checkpoint and paired validation traffic, changing
only the deterministic decoder from 0.90 fusion to target critic changed success
to {target['success_rate']:.3f}, collision to {target['collision_rate']:.3f}, and
timeout to {target['timeout_rate']:.3f}. The paired intervention produced
{clean['paired_success_effect']['improved_count']} failure-to-success changes and
{clean['paired_success_effect']['regressed_count']} regressions (post-hoc exact
paired sign p={clean['paired_success_effect']['exact_two_sided_paired_sign_p_value']:.6f}).

The fusion trace contains {trace['keep_intent_actor_non_keep_records']} decisions
where route intent was keep but the actor override issued a non-keep command;
{trace['keep_intent_actor_non_keep_on_gneE5_records']} occurred on `gneE5`.
All {trace['regular_collision_terminal_edges'].get('gneE5', 0)} regular terminal
collisions were on that edge. Raising the confidence threshold alone is weak:
even threshold 0.999 would still override
{trace['threshold_sweep'][-1]['counterfactual_actor_override_rate_on_fixed_observations']:.3f}
of these fixed observations.

## Why neither decoder can be globally frozen

Cross seed 4 favors target critic, but the frozen v4.4 Cross seed 2 evidence has
the opposite sign: target train calibration reached only
{historical['v4_4_cross_seed2']['target_best_train_success_rate']:.3f} success,
while both fusion candidates reached
{historical['v4_4_cross_seed2']['fusion_train_success_rate']:.3f}. This is
checkpoint/seed-conditioned decoder heterogeneity, not evidence that either
decoder globally dominates.

## v4.6 recommendation

Evaluate the Cartesian product of the two already eligible checkpoints and two
already frozen deterministic decoders on the same paired train-only calibration
block. Apply the unchanged lexicographic outcome ordering, then prefer
exact-final and target critic only on complete ties. On E1's training block this
post-hoc rule selects `{selected['selected_checkpoint_kind']} ×
{selected['selected_decoder']}` without validation feedback. That pair later
achieved {selected_validation['success_rate']:.3f} success,
{selected_validation['collision_rate']:.3f} collision, and
{selected_validation['timeout_rate']:.3f} timeout on the independent validation
block; this remains diagnostic and fresh v4.6 runs are required.

Keep the network, learned weights, stochastic actor, reward, training objective,
entropy, environment, candidate checkpoints, decoder definitions, threshold,
outcome gates, promotion protocol, and formal protocol unchanged. Only the
train-only deployment selector expands from two checkpoints under one global
decoder to four checkpoint-decoder pairs.
"""


def main() -> int:
    _require(_sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256, "v4.5 contract drifted")
    _require(_sha256(FREEZE) == EXPECTED_FREEZE_SHA256, "v4.5 freeze drifted")
    _require(_sha256(PATCH_FREEZE) == EXPECTED_PATCH_FREEZE_SHA256, "v4.5.1 freeze drifted")
    _require(_sha256(DECISION) == EXPECTED_DECISION_SHA256, "v4.5 decision drifted")
    _require(
        _sha256(V44_ATTRIBUTION) == EXPECTED_V44_ATTRIBUTION_SHA256,
        "v4.4 attribution drifted",
    )

    decision = _load(DECISION)
    _require(
        decision.get("decision") == "fail"
        and decision.get("stopped_after") == "E1"
        and decision.get("next_job") is None,
        "v4.5 state-machine stop receipt drifted",
    )
    e1 = decision["jobs"]["E1"]
    _require(e1.get("accepted") is True, "E1 artifact acceptance drifted")
    _require(e1["gate"].get("selector_passed") is True, "E1 selector did not pass")
    _require(e1["gate"].get("mechanism_passed") is True, "E1 mechanism did not pass")
    _require(e1["gate"].get("outcome_passed") is False, "E1 outcome did not fail")

    best_target_train = _validate_replay(
        "best_target_train",
        checkpoint_sha256=BEST_SHA256,
        decoder="target",
        partition="train",
        seed_start=68_000,
    )
    final_target_train = _validate_replay(
        "final_target_train",
        checkpoint_sha256=FINAL_SHA256,
        decoder="target",
        partition="train",
        seed_start=68_000,
    )
    best_target_validation = _validate_replay(
        "target_validation",
        checkpoint_sha256=BEST_SHA256,
        decoder="target",
        partition="validation",
        seed_start=59_000,
    )
    actor_validation = _validate_replay(
        "actor_validation",
        checkpoint_sha256=BEST_SHA256,
        decoder="actor",
        partition="validation",
        seed_start=59_000,
    )
    final_target_validation = _validate_replay(
        "final_target_validation",
        checkpoint_sha256=FINAL_SHA256,
        decoder="target",
        partition="validation",
        seed_start=59_000,
    )

    fusion_evaluation = _load(RUN / "final_evaluation.json")
    fusion_diagnostics = _load(RUN / "action_diagnostics.json")
    _require(fusion_diagnostics["outcomes"] == fusion_evaluation, "E1 fusion outcomes drifted")
    _require(
        fusion_diagnostics.get("exact_fusion_rule_match_rate") == 1.0
        and fusion_diagnostics.get("target_fallback_rule_match_rate") == 1.0
        and fusion_diagnostics.get("actor_override_predicate_valid_rate") == 1.0,
        "E1 frozen fusion mechanism integrity drifted",
    )
    paired = _paired_effect(
        fusion_diagnostics,
        best_target_validation["diagnostics"],
        control_name="fusion",
        treatment_name="target",
    )

    best_fusion_train = _load(RUN / "selector" / "cal" / "best" / "evaluation.json")
    final_fusion_train = _load(RUN / "selector" / "cal" / "final" / "evaluation.json")
    joint = _joint_matrix(
        best_fusion_train,
        final_fusion_train,
        best_target_train["outcomes"],
        final_target_train["outcomes"],
    )
    _require(
        joint["selected_checkpoint_kind"] == "exact_final"
        and joint["selected_decoder"] == "target_critic",
        "E1 train-only joint-selector counterfactual drifted",
    )

    fusion_rows = _load_jsonl(RUN / "action_diagnostics_decisions.jsonl")
    regular_rows = [row for row in fusion_rows if int(row["episode"]) != 8]
    keep_actor_non_keep = [
        row
        for row in fusion_rows
        if bool(row.get("pre_action_route_intent_valid"))
        and int(row.get("pre_action_route_intent")) == 0
        and row["fusion_decoder"].get("selected_source") == "actor"
        and int(row["fusion_decoder"].get("selected_lane")) != 0
    ]

    v44 = _load(V44_ATTRIBUTION)
    v44_target = v44["decoder_only_paired_intervention"]["control_target_decoder_outcomes"]
    v44_fusion = v44["fusion_train_only_selector_counterfactual"]
    payload = {
        "schema_version": "topo-scene-v4.5.e1-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "formal_test_accessed": False,
        "failed_job": "E1__cand__cross__s4__p3c1f7f8b",
        "failure_gate": {
            "selector_passed": True,
            "mechanism_passed": True,
            "outcome_passed": False,
            "failed_checks": ["success_rate", "collision_rate"],
            "development_decision": "fail",
            "stopped_after": "E1",
            "E2_E3_E4_executed": False,
        },
        "primary_attribution": {
            "mechanism": "global_fusion_decoder_overrides_safe_target_keep_choices_on_cross_seed4",
            "causal_intervention": "decoder_only_same_checkpoint_same_paired_validation_traffic",
            "confidence": "strong_post_hoc_causal_evidence_requires_fresh_v4_6_confirmation",
            "rejected_as_primary": [
                "selector_integrity_failure",
                "fusion_implementation_mismatch",
                "action_mask_infeasibility",
                "environment_or_traffic_seed_difference",
                "training_weight_difference_in_same_checkpoint_intervention",
                "confidence_threshold_too_low_as_sole_cause",
            ],
        },
        "decoder_only_same_checkpoint_validation_intervention": {
            "post_hoc_only": True,
            "checkpoint_kind": "highest_training_success",
            "checkpoint_sha256": BEST_SHA256,
            "scenario": "cross",
            "validation_seed_start": 59_000,
            "episodes": 12,
            "control_decoder": "actor_confident_non_keep_else_target_critic@0.90",
            "treatment_decoder": "keep_tie_aware_argmax_feasible_min_target_twin_q",
            "fusion_outcomes": fusion_evaluation,
            "target_outcomes": best_target_validation["outcomes"],
            "success_rate_delta_target_minus_fusion": (
                float(best_target_validation["outcomes"]["success_rate"])
                - float(fusion_evaluation["success_rate"])
            ),
            "collision_rate_delta_target_minus_fusion": (
                float(best_target_validation["outcomes"]["collision_rate"])
                - float(fusion_evaluation["collision_rate"])
            ),
            "timeout_rate_delta_target_minus_fusion": (
                float(best_target_validation["outcomes"]["timeout_rate"])
                - float(fusion_evaluation["timeout_rate"])
            ),
            "paired_success_effect": paired,
            "actor_only_outcomes": actor_validation["outcomes"],
        },
        "joint_train_only_selector_counterfactual": joint,
        "joint_selector_selected_pair_validation_check": {
            "post_hoc_only": True,
            "selection_did_not_use_validation": True,
            "checkpoint_kind": "exact_final",
            "checkpoint_sha256": FINAL_SHA256,
            "decoder": "target_critic",
            "validation_seed_start": 59_000,
            "outcomes": final_target_validation["outcomes"],
        },
        "cross_seed_heterogeneity": {
            "interpretation": "decoder superiority changes sign across independently trained Cross seeds",
            "v4_5_cross_seed4": {
                "target_best_train_success_rate": best_target_train["outcomes"]["success_rate"],
                "target_final_train_success_rate": final_target_train["outcomes"]["success_rate"],
                "fusion_best_train_success_rate": best_fusion_train["success_rate"],
                "fusion_final_train_success_rate": final_fusion_train["success_rate"],
            },
            "v4_4_cross_seed2": {
                "target_best_train_success_rate": v44_target["calibration_candidate_outcomes"]["highest_training_success"]["success_rate"],
                "target_final_train_success_rate": v44_target["calibration_candidate_outcomes"]["exact_final"]["success_rate"],
                "fusion_train_success_rate": v44_fusion["highest_training_success"]["outcomes"]["success_rate"],
                "fusion_selected_checkpoint_kind": v44_fusion["selected_checkpoint_kind"],
                "fusion_selected_validation_success_rate": v44_fusion["selected_checkpoint_validation_outcomes"]["success_rate"],
            },
        },
        "trace_mechanism": {
            "fusion_decision_records": len(fusion_rows),
            "actor_override_records": fusion_diagnostics["actor_override_records"],
            "actor_override_rate": fusion_diagnostics["actor_override_rate"],
            "keep_intent_actor_non_keep_records": len(keep_actor_non_keep),
            "keep_intent_actor_non_keep_rate": len(keep_actor_non_keep) / len(fusion_rows),
            "keep_intent_actor_non_keep_on_gneE5_records": sum(
                row.get("pre_action_current_edge") == "gneE5" for row in keep_actor_non_keep
            ),
            "regular_collision_terminal_edges": _terminal_collision_edges(regular_rows),
            "all_collision_terminal_edges": _terminal_collision_edges(fusion_rows),
            "threshold_sweep": _threshold_sweep(fusion_rows),
            "route_window_metric_blind_spot": (
                "route_action_window_match counts only non-keep route intent; it does not penalize "
                "actor overrides after the route intent returns to keep"
            ),
        },
        "recommended_single_change": {
            "version": "v4.6",
            "name": "train_only_joint_checkpoint_decoder_selector",
            "checkpoint_candidates": ["highest_training_success", "exact_final"],
            "decoder_candidates": ["target_critic", "fusion_0_90"],
            "candidate_pairs": 4,
            "paired_train_calibration_required": True,
            "selector_receipt_must_predate_validation": True,
            "training_changed": False,
            "learned_weights_changed_by_selector": False,
            "decoder_definitions_changed": False,
            "fusion_threshold_changed": False,
            "reward_changed": False,
            "environment_changed": False,
            "outcome_gates_changed": False,
            "fresh_preregistered_development_required": True,
        },
        "source_hashes": {
            "experiment_contract": _source(CONTRACT),
            "implementation_freeze": _source(FREEZE),
            "v4_5_1_patch_freeze": _source(PATCH_FREEZE),
            "development_decision": _source(DECISION),
            "e1_fusion_evaluation": _source(RUN / "final_evaluation.json"),
            "e1_fusion_actions": _source(RUN / "action_diagnostics.json"),
            "e1_fusion_trace": _source(RUN / "action_diagnostics_decisions.jsonl"),
            "best_fusion_train": _source(RUN / "selector" / "cal" / "best" / "evaluation.json"),
            "final_fusion_train": _source(RUN / "selector" / "cal" / "final" / "evaluation.json"),
            "best_target_train": best_target_train["source"],
            "final_target_train": final_target_train["source"],
            "best_target_validation": best_target_validation["source"],
            "actor_validation": actor_validation["source"],
            "final_target_validation": final_target_validation["source"],
            "v4_4_d2_attribution": _source(V44_ATTRIBUTION),
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
