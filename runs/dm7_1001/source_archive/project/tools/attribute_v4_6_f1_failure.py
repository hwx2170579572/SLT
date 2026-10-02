"""Seal the post-hoc attribution for the failed v4.6 F1 development cell."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V46_ROOT = ROOT / "results_topo_v4_6_dev"
DEVELOPMENT = V46_ROOT / "development"
RUN = DEVELOPMENT / "runs" / "F1__cand__cross__s6__p814d53b7"
ATTRIBUTION = V46_ROOT / "attribution" / "f1_failure"
CLOSED_LOOP = ATTRIBUTION / "closed_loop"
CROSS_BLOCK = ATTRIBUTION / "cross_block"
CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_6.yaml"
FREEZE = V46_ROOT / "engineering" / "implementation_freeze.json"
DECISION = DEVELOPMENT / "development_decision.json"
RECEIPT = RUN / "selector" / "receipt.json"
REPLAY_TOOL = ROOT / "tools" / "replay_v4_6_pair_posthoc.py"
V45_RUN = (
    ROOT
    / "results_topo_v4_5_dev"
    / "development"
    / "runs"
    / "E1__cand__cross__s4__p3c1f7f8b"
)
V45_TARGET = (
    ROOT
    / "results_topo_v4_5_dev"
    / "attribution"
    / "e1_failure"
    / "closed_loop"
    / "final_target_validation"
)
TRAFFIC = ROOT / "envs" / "sumo" / "original_scenarios_v1" / "cross" / "traffic"

EXPECTED_CONTRACT_SHA256 = "814d53b74afe6ceb5963f0cb4aca1b556c02e71bc88adfb244e5b31f67750c9d"
EXPECTED_FREEZE_SHA256 = "02169f71285b39aa87b6ed78a868d4a1f066b3a1debd409ecaa609e888aacf33"
EXPECTED_DECISION_SHA256 = "0d520b8b791a35e9bde1adcf4bcab487f1f60adf279fd12cbea1f34612f42ae3"
EXPECTED_REPLAY_TOOL_SHA256 = "cb54741603192525f582f0698bbbc2ff75e44ca17abf84290be8f3b61b1cbef9"
BEST_SHA256 = "848d54f2f2e9bf1942bc03a4ce10959e7a140c5d7e25479639fdc7c51f98a986"
FINAL_SHA256 = "16c18a0161ee73dd1cfa63df2a5de22a040bfed15edb088a11f49d0042bf4835"
EVENTS = ("success", "collision", "off_route", "timeout")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL file: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _counts(summary: dict[str, Any]) -> dict[str, int]:
    episodes = int(summary["episodes"])
    return {
        "episodes": episodes,
        "success_count": round(float(summary["success_rate"]) * episodes),
        "collision_count": round(float(summary["collision_rate"]) * episodes),
        "off_route_count": round(float(summary["off_route_rate"]) * episodes),
        "timeout_count": round(float(summary["timeout_rate"]) * episodes),
    }


def _selection_key(
    summary: dict[str, Any], checkpoint: str, decoder: str
) -> tuple[Any, ...]:
    count = _counts(summary)
    return (
        count["success_count"],
        -count["collision_count"],
        -count["off_route_count"],
        -count["timeout_count"],
        float(summary["mean_return"]),
        int(checkpoint == "exact_final"),
        int(decoder == "target_critic"),
    )


def _signature(rows: list[dict[str, Any]]) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (int(row["episode"]), int(row["seed"]), str(row["traffic_variant"]))
        for row in rows
    )


def _validate_replay(
    directory: Path,
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    decoder: str,
    seed_start: int,
) -> dict[str, Any]:
    result_path = directory / "attribution_result.json"
    diagnostics_path = directory / "action_diagnostics.json"
    trace_path = directory / "decisions.jsonl"
    result = _load(result_path)
    diagnostics = _load(diagnostics_path)
    expected = {
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "mutated_checkpoint": False,
    }
    for key, value in expected.items():
        _require(result.get(key) is value, f"{directory.name}: {key} drifted")
    _require(result.get("checkpoint_kind") == checkpoint, "checkpoint kind drifted")
    _require(result.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint hash drifted")
    _require(result.get("decoder") == decoder, "decoder drifted")
    _require(result.get("traffic_partition") == "validation", "partition drifted")
    _require(int(result.get("evaluation_seed_start")) == seed_start, "seed block drifted")
    _require(int(result.get("evaluation_episodes")) == 12, "episode count drifted")
    _require(diagnostics.get("outcomes") == result.get("outcomes"), "outcomes drifted")
    _require(_sha256(trace_path) == result.get("trace_sha256"), "trace hash drifted")
    return {
        **result,
        "source": _source(result_path),
        "diagnostics_source": _source(diagnostics_path),
        "trace_source": _source(trace_path),
        "diagnostics": diagnostics,
        "trace_rows": _load_jsonl(trace_path),
    }


def _actual_final_target() -> dict[str, Any]:
    detailed_path = RUN / "paper_evaluation_detailed.json"
    result_path = RUN / "final_evaluation.json"
    diagnostics_path = RUN / "action_diagnostics.json"
    trace_path = RUN / "action_diagnostics_decisions.jsonl"
    detailed = _load(detailed_path)
    result = _load(result_path)
    diagnostics = _load(diagnostics_path)
    _require(diagnostics.get("outcomes") == result, "actual target outcomes drifted")
    _require(detailed.get("selected_deployment_decoder") == "target_critic", "actual decoder drifted")
    _require(detailed.get("selected_source_checkpoint_sha256") == FINAL_SHA256, "actual checkpoint drifted")
    return {
        "checkpoint_kind": "exact_final",
        "checkpoint_sha256": FINAL_SHA256,
        "decoder": "target_critic",
        "traffic_partition": "validation",
        "evaluation_seed_start": 63_000,
        "evaluation_episodes": 12,
        "outcomes": result,
        "per_episode": detailed["episode_records"],
        "diagnostics": diagnostics,
        "trace_rows": _load_jsonl(trace_path),
        "source": _source(detailed_path),
        "diagnostics_source": _source(diagnostics_path),
        "trace_source": _source(trace_path),
    }


def _historical_v45_target() -> dict[str, Any]:
    result_path = V45_TARGET / "attribution_result.json"
    diagnostics_path = V45_TARGET / "action_diagnostics.json"
    trace_path = V45_TARGET / "action_diagnostics_decisions.jsonl"
    result = _load(result_path)
    diagnostics = _load(diagnostics_path)
    expected = {
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "computed_from_real_closed_loop_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
    }
    for key, value in expected.items():
        _require(result.get(key) is value, f"historical v4.5 {key} drifted")
    _require(result.get("decoder") == "target", "historical decoder drifted")
    _require(
        result.get("checkpoint_sha256")
        == "2d9afa0420fd78e6dec7b60f8b5fb53f5139ccdbb63affe8bd3a29fcc30b7519",
        "historical checkpoint drifted",
    )
    _require(int(result.get("evaluation_seed_start")) == 59_000, "historical seed block drifted")
    _require(diagnostics.get("outcomes") == result.get("outcomes"), "historical outcomes drifted")
    return {
        "checkpoint_kind": "exact_final",
        "checkpoint_sha256": result["checkpoint_sha256"],
        "decoder": "target_critic",
        "traffic_partition": "validation",
        "evaluation_seed_start": 59_000,
        "evaluation_episodes": 12,
        "outcomes": result["outcomes"],
        "per_episode": diagnostics["per_episode"],
        "diagnostics": diagnostics,
        "trace_rows": _load_jsonl(trace_path),
        "source": _source(result_path),
        "diagnostics_source": _source(diagnostics_path),
        "trace_source": _source(trace_path),
    }


def _monitor_outcomes(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(lines[1:]))
    return {
        "episodes": len(rows),
        "success_count": sum(row["is_success"] == "True" for row in rows),
        "collision_count": sum(row["collision"] == "True" for row in rows),
        "timeout_count": sum(row["max_time"] == "True" for row in rows),
        "last_20_success_rate": sum(
            row["is_success"] == "True" for row in rows[-20:]
        )
        / 20.0,
        "source": _source(path),
    }


def _paired_effect(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    control_name: str,
    treatment_name: str,
) -> dict[str, Any]:
    left = control["per_episode"]
    right = treatment["per_episode"]
    _require(_signature(left) == _signature(right), "paired episode signature drifted")
    transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for lrow, rrow in zip(left, right):
        key = f"{int(bool(lrow['success']))}_to_{int(bool(rrow['success']))}"
        transitions[key] += 1
        rows.append(
            {
                "episode": int(lrow["episode"]),
                "seed": int(lrow["seed"]),
                "traffic_variant": lrow["traffic_variant"],
                f"{control_name}_success": bool(lrow["success"]),
                f"{treatment_name}_success": bool(rrow["success"]),
                f"{control_name}_collision": bool(lrow["collision"]),
                f"{treatment_name}_collision": bool(rrow["collision"]),
                f"{control_name}_timeout": bool(lrow["timeout"]),
                f"{treatment_name}_timeout": bool(rrow["timeout"]),
            }
        )
    improved = transitions["0_to_1"]
    regressed = transitions["1_to_0"]
    discordant = improved + regressed
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(improved, regressed) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "paired_episodes": len(left),
        "transition_counts": dict(sorted(transitions.items())),
        "improved_count": improved,
        "regressed_count": regressed,
        "exact_two_sided_paired_sign_p_value": p_value,
        "post_hoc_support_only": True,
        "per_episode": rows,
    }


def _trace_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    entropies: list[float] = []
    multi_action_entropies: list[float] = []
    for row in rows:
        by_episode[int(row["episode"])].append(row)
        probabilities = [float(value) for value in row["hybrid_policy"]["lane_probabilities"]]
        entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
        entropies.append(entropy)
        if int(row["hybrid_policy"]["valid_lane_actions"]) > 1:
            multi_action_entropies.append(entropy)
    terminals = [max(group, key=lambda row: int(row["decision"])) for group in by_episode.values()]
    collision_edges = Counter(
        str(row.get("pre_action_current_edge"))
        for row in terminals
        if bool(row.get("collision"))
    )
    early_collisions = [
        {
            "episode": int(row["episode"]),
            "seed": int(row["seed"]),
            "traffic_variant": row["traffic_variant"],
            "terminal_decision": int(row["decision"]),
            "terminal_edge": row.get("pre_action_current_edge"),
        }
        for row in terminals
        if bool(row.get("collision")) and int(row["decision"]) <= 40
    ]
    return {
        "decision_records": len(rows),
        "episode_records": len(by_episode),
        "multi_action_records": len(multi_action_entropies),
        "mean_lane_entropy_all_records": sum(entropies) / len(entropies),
        "mean_lane_entropy_multi_action_records": (
            sum(multi_action_entropies) / len(multi_action_entropies)
            if multi_action_entropies
            else None
        ),
        "multi_action_entropy_below_rate": {
            str(threshold): sum(value < threshold for value in multi_action_entropies)
            / len(multi_action_entropies)
            for threshold in (0.001, 0.01, 0.02, 0.05, 0.1)
        }
        if multi_action_entropies
        else {},
        "terminal_collision_edges": dict(collision_edges.most_common()),
        "early_collision_count": len(early_collisions),
        "early_collisions": early_collisions,
    }


def _traffic_features(name: str) -> dict[str, Any]:
    path = TRAFFIC / name
    root = ET.parse(path).getroot()
    vehicles = list(root.findall("vehicle"))
    departures: list[float] = []
    route_counts: Counter[str] = Counter()
    route_before_60: Counter[str] = Counter()
    for vehicle in vehicles:
        depart = float(vehicle.attrib["depart"])
        departures.append(depart)
        route = vehicle.find("route")
        edges = " ".join((route.attrib.get("edges", "") if route is not None else "").split())
        route_counts[edges] += 1
        if depart <= 60.0:
            route_before_60[edges] += 1
    rounded = Counter(round(value, 3) for value in departures)
    return {
        "traffic_variant": name,
        "source": _source(path),
        "vehicle_count": len(vehicles),
        "depart_at_zero_count": sum(value == 0.0 for value in departures),
        "depart_by_60_seconds_count": sum(value <= 60.0 for value in departures),
        "peak_simultaneous_departures": max(rounded.values()) if rounded else 0,
        "route_counts": dict(sorted(route_counts.items())),
        "route_counts_by_60_seconds": dict(sorted(route_before_60.items())),
    }


def _mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return sum(values) / len(values)


def _traffic_group_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "variants": len(features),
        "mean_vehicle_count": _mean(float(row["vehicle_count"]) for row in features),
        "mean_depart_at_zero_count": _mean(float(row["depart_at_zero_count"]) for row in features),
        "mean_depart_by_60_seconds_count": _mean(
            float(row["depart_by_60_seconds_count"]) for row in features
        ),
        "mean_peak_simultaneous_departures": _mean(
            float(row["peak_simultaneous_departures"]) for row in features
        ),
    }


def _training_diagnostics(run: Path, target_success: float, seed: int) -> dict[str, Any]:
    diagnostics = _load(run / "training_diagnostics.json")
    stats = diagnostics["statistics"]
    return {
        "seed": seed,
        "target_decoder_validation_success_rate": target_success,
        "critic_loss_last": float(stats["train/critic_loss"]["last"]),
        "lane_entropy_last": float(stats["hybrid/lane_entropy"]["last"]),
        "representation_loss_last": float(stats["train/representation_loss"]["last"]),
        "topology_residual_scale_last": float(
            stats["diagnostic/topology_residual_scale"]["last"]
        ),
        "source": _source(run / "training_diagnostics.json"),
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    lrank, rrank = _rank(left), _rank(right)
    lmean, rmean = _mean(lrank), _mean(rrank)
    numerator = sum((x - lmean) * (y - rmean) for x, y in zip(lrank, rrank))
    denominator = math.sqrt(
        sum((x - lmean) ** 2 for x in lrank)
        * sum((y - rmean) ** 2 for y in rrank)
    )
    return numerator / denominator


def _monitor_credit_coverage(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines[1:])
    lengths = [int(row["l"]) for row in reader]
    transitions = sum(lengths) + len(lengths)  # released profile duplicates each episode end
    output: dict[str, Any] = {
        "episodes": len(lengths),
        "decision_transitions_with_duplicated_episode_ends": transitions,
        "batch_size": 32,
    }
    for horizon in (4, 16):
        eligible = sum(min(horizon, length) + 1 for length in lengths)
        fraction = eligible / transitions
        output[f"n{horizon}"] = {
            "approximate_terminal_credit_slots": eligible,
            "approximate_terminal_credit_fraction": fraction,
            "expected_terminal_credit_samples_per_uniform_batch": 32 * fraction,
            "probability_uniform_batch_contains_zero_terminal_credit_samples": (
                (1.0 - fraction) ** 32
            ),
        }
    output["coverage_multiplier_n16_over_n4"] = (
        output["n16"]["approximate_terminal_credit_fraction"]
        / output["n4"]["approximate_terminal_credit_fraction"]
    )
    output["role"] = "trace_derived_design_calculation_not_an_experimental_result"
    return output


def _markdown(payload: dict[str, Any]) -> str:
    actual = payload["validation_pair_matrix"]["exact_final__target_critic"]["outcomes"]
    fusion = payload["validation_pair_matrix"]["exact_final__fusion_0_90"]["outcomes"]
    cross = payload["crossed_checkpoint_block_intervention"]["success_rate_matrix"]
    diagnostics = payload["optimizer_instability_evidence"]
    credit = payload["terminal_credit_coverage"]
    return f"""# v4.6 F1 Deep Attribution

Status: post-hoc diagnosis only. No value in this report counts toward a
development, promotion, or formal-test gate. The formal test partition was not
accessed.

## Frozen failure

F1 (Cross, training seed 6) passed artifact, selector, and decoder-integrity
checks but failed the outcome gate: the selected exact-final target-critic pair
reached {actual['success_rate']:.3f} success, {actual['collision_rate']:.3f}
collision, and {actual['timeout_rate']:.3f} timeout on seeds 63000--63011.

## Selector is secondary, not primary

All four frozen checkpoint/decoder pairs were replayed on the same validation
traffic. The best validation success count was still only 3/12. Switching the
selected exact-final checkpoint from target critic to fusion kept success at
{fusion['success_rate']:.3f}, reduced collision to {fusion['collision_rate']:.3f},
and increased timeout to {fusion['timeout_rate']:.3f}. Thus calibration chose a
less safe failure mode, but no eligible pair could pass the success guard.
A leakage-only per-episode oracle over all four pairs would cover 7/12
successes, exposing complementary failures but not supplying a legal deployment
rule.

## Crossed weight/block intervention

The target-critic success matrix is:

| frozen weights | block 59000 | block 63000 |
| --- | ---: | ---: |
| v4.5 Cross seed 4 | {cross['v45_seed4']['block_59000']:.3f} | {cross['v45_seed4']['block_63000']:.3f} |
| v4.6 Cross seed 6 | {cross['v46_seed6']['block_59000']:.3f} | {cross['v46_seed6']['block_63000']:.3f} |

Seed-4 weights retain 10/12 success and zero collision on the exact block where
seed-6 weights reached only 3/12 with six collisions. Seed-6 improves to 7/12
on block 59000 but remains below seed-4's 11/12. This establishes a dominant
training-weight effect plus a smaller weight-by-traffic interaction; traffic
block difficulty alone is rejected as the primary cause.

## Training mechanism evidence

For four comparable Cross runs using the exact-final target decoder, final
critic loss and validation success have Spearman rho
{diagnostics['comparable_target_runs_spearman_critic_loss_vs_success']:.3f}
(n=4; diagnostic only). Seed 6 ends with critic loss
{diagnostics['seed6_vs_seed4']['critic_loss_ratio']:.2f}x seed 4 and lane entropy
{diagnostics['seed6_vs_seed4']['lane_entropy_seed4_over_seed6']:.1f}x lower.
Training episode success is similar, so missing successful rollouts is not the
main explanation. The evidence instead points to high-variance sparse terminal
credit producing unstable Q ordering and saturated per-state lane decisions.

Under the released 4-step replay semantics, only about
{credit['n4']['approximate_terminal_credit_fraction']:.3f} of stored starting
positions have a terminal reward inside their return window; a uniform batch of
32 has probability {credit['n4']['probability_uniform_batch_contains_zero_terminal_credit_samples']:.3f}
of containing none. A horizon-correct 16-step estimator raises this trace-based
coverage by {credit['coverage_multiplier_n16_over_n4']:.2f}x without changing
the reward, observation, action head, selector, or deployment decoder.

## v4.7 recommendation

Change only the return estimator: use 16-decision n-step returns and bootstrap
with gamma raised to the actual sampled horizon. Keep the v4.6 model, reward,
warm-up, raw-step budget, batch size, buffer capacity, joint checkpoint/decoder
selector, decoder definitions, thresholds, gates, and formal lock unchanged.
Run the failed Cross seed 6 first on a fresh block, followed by a matched frozen
4-step parent control if the candidate passes, then independent Cross, CARLA,
and roundabout-medium guards. This is a falsifiable credit-assignment repair;
failure sends the project back to attribution rather than promotion.
"""


def main() -> int:
    _require(_sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256, "v4.6 contract drifted")
    _require(_sha256(FREEZE) == EXPECTED_FREEZE_SHA256, "v4.6 freeze drifted")
    _require(_sha256(DECISION) == EXPECTED_DECISION_SHA256, "v4.6 decision drifted")
    _require(_sha256(REPLAY_TOOL) == EXPECTED_REPLAY_TOOL_SHA256, "replay tool drifted")

    decision = _load(DECISION)
    f1 = decision["jobs"]["F1"]
    _require(decision.get("decision") == "fail" and decision.get("stopped_after") == "F1", "state machine did not stop at F1")
    _require(f1.get("accepted") is True, "F1 artifact acceptance drifted")
    _require(f1["gate"].get("mechanism_passed") is True, "mechanism gate did not pass")
    _require(f1["gate"].get("selector_passed") is True, "selector gate did not pass")
    _require(f1["gate"].get("outcome_passed") is False, "outcome gate did not fail")
    _require(decision.get("formal_test_unlocked") is False, "formal test unexpectedly unlocked")

    actual_target = _actual_final_target()
    validation = {
        "highest_training_success__target_critic": _validate_replay(
            CLOSED_LOOP / "best_target_validation",
            checkpoint="highest_training_success",
            checkpoint_sha256=BEST_SHA256,
            decoder="target_critic",
            seed_start=63_000,
        ),
        "highest_training_success__fusion_0_90": _validate_replay(
            CLOSED_LOOP / "best_fusion_validation",
            checkpoint="highest_training_success",
            checkpoint_sha256=BEST_SHA256,
            decoder="fusion_0_90",
            seed_start=63_000,
        ),
        "exact_final__target_critic": actual_target,
        "exact_final__fusion_0_90": _validate_replay(
            CLOSED_LOOP / "final_fusion_validation",
            checkpoint="exact_final",
            checkpoint_sha256=FINAL_SHA256,
            decoder="fusion_0_90",
            seed_start=63_000,
        ),
    }
    signatures = {_signature(value["per_episode"]) for value in validation.values()}
    _require(len(signatures) == 1, "validation candidate pairs are not traffic-paired")

    receipt = _load(RECEIPT)
    calibration: dict[str, Any] = {}
    for row in receipt["candidates"]:
        kind = str(row["checkpoint_kind"])
        decoder = str(row["deployment_decoder"])
        short = "best" if kind == "highest_training_success" else "final"
        detailed_path = RUN / "selector" / "cal" / short / decoder / "detailed.json"
        detailed = _load(detailed_path)
        key = f"{kind}__{decoder}"
        _require(detailed["summary"] == row["summary"], f"calibration summary drifted: {key}")
        calibration[key] = {
            "checkpoint_kind": kind,
            "decoder": decoder,
            "outcomes": row["summary"],
            "per_episode": detailed["episode_records"],
            "selection_key": list(_selection_key(row["summary"], kind, decoder)),
            "source": _source(detailed_path),
        }
    _require(set(calibration) == set(validation), "candidate pair set drifted")
    calibration_signatures = {_signature(value["per_episode"]) for value in calibration.values()}
    _require(len(calibration_signatures) == 1, "calibration pairs are not traffic-paired")

    validation_matrix: dict[str, Any] = {}
    for key, value in validation.items():
        kind, decoder = key.split("__", 1)
        validation_matrix[key] = {
            "checkpoint_kind": kind,
            "decoder": decoder,
            "outcomes": value["outcomes"],
            "outcome_counts": _counts(value["outcomes"]),
            "selection_key": list(_selection_key(value["outcomes"], kind, decoder)),
            "source": value["source"],
            "diagnostics_source": value["diagnostics_source"],
            "trace_source": value["trace_source"],
            "trace_metrics": _trace_metrics(value["trace_rows"]),
        }
    calibration_rank = sorted(calibration, key=lambda key: tuple(calibration[key]["selection_key"]), reverse=True)
    validation_rank = sorted(validation_matrix, key=lambda key: tuple(validation_matrix[key]["selection_key"]), reverse=True)

    v45_59000 = _historical_v45_target()
    v45_63000 = _validate_replay(
        CROSS_BLOCK / "v45_s4_final_target_on_v46_block",
        checkpoint="exact_final",
        checkpoint_sha256="2d9afa0420fd78e6dec7b60f8b5fb53f5139ccdbb63affe8bd3a29fcc30b7519",
        decoder="target_critic",
        seed_start=63_000,
    )
    v46_59000 = _validate_replay(
        CROSS_BLOCK / "v46_s6_final_target_on_v45_block",
        checkpoint="exact_final",
        checkpoint_sha256=FINAL_SHA256,
        decoder="target_critic",
        seed_start=59_000,
    )
    v46_63000 = actual_target
    _require(
        _signature(v45_59000["per_episode"]) == _signature(v46_59000["per_episode"]),
        "block 59000 is not checkpoint-paired",
    )
    _require(
        _signature(v45_63000["per_episode"]) == _signature(v46_63000["per_episode"]),
        "block 63000 is not checkpoint-paired",
    )
    crossed = {
        "design": "two_frozen_weight_sets_by_two_validation_traffic_blocks",
        "decoder": "target_critic",
        "post_hoc_diagnostic_only": True,
        "formal_test_accessed": False,
        "success_rate_matrix": {
            "v45_seed4": {
                "block_59000": v45_59000["outcomes"]["success_rate"],
                "block_63000": v45_63000["outcomes"]["success_rate"],
            },
            "v46_seed6": {
                "block_59000": v46_59000["outcomes"]["success_rate"],
                "block_63000": v46_63000["outcomes"]["success_rate"],
            },
        },
        "collision_rate_matrix": {
            "v45_seed4": {
                "block_59000": v45_59000["outcomes"]["collision_rate"],
                "block_63000": v45_63000["outcomes"]["collision_rate"],
            },
            "v46_seed6": {
                "block_59000": v46_59000["outcomes"]["collision_rate"],
                "block_63000": v46_63000["outcomes"]["collision_rate"],
            },
        },
        "timeout_rate_matrix": {
            "v45_seed4": {
                "block_59000": v45_59000["outcomes"]["timeout_rate"],
                "block_63000": v45_63000["outcomes"]["timeout_rate"],
            },
            "v46_seed6": {
                "block_59000": v46_59000["outcomes"]["timeout_rate"],
                "block_63000": v46_63000["outcomes"]["timeout_rate"],
            },
        },
        "paired_seed4_vs_seed6_on_block_59000": _paired_effect(
            v46_59000,
            v45_59000,
            control_name="seed6_weights",
            treatment_name="seed4_weights",
        ),
        "paired_seed4_vs_seed6_on_block_63000": _paired_effect(
            v46_63000,
            v45_63000,
            control_name="seed6_weights",
            treatment_name="seed4_weights",
        ),
        "seed4_minus_seed6_success_rate_effect": {
            "block_59000": v45_59000["outcomes"]["success_rate"]
            - v46_59000["outcomes"]["success_rate"],
            "block_63000": v45_63000["outcomes"]["success_rate"]
            - v46_63000["outcomes"]["success_rate"],
        },
        "block_63000_minus_59000_success_rate_effect": {
            "v45_seed4": v45_63000["outcomes"]["success_rate"]
            - v45_59000["outcomes"]["success_rate"],
            "v46_seed6": v46_63000["outcomes"]["success_rate"]
            - v46_59000["outcomes"]["success_rate"],
        },
        "difference_in_differences_success_rate": (
            v45_63000["outcomes"]["success_rate"]
            - v45_59000["outcomes"]["success_rate"]
        )
        - (
            v46_63000["outcomes"]["success_rate"]
            - v46_59000["outcomes"]["success_rate"]
        ),
        "interpretation": (
            "dominant training-weight effect with a smaller weight-by-traffic interaction; "
            "traffic-block difficulty alone is not the primary cause"
        ),
        "sources": {
            "v45_seed4_block59000": v45_59000["source"],
            "v45_seed4_block63000": v45_63000["source"],
            "v46_seed6_block59000": v46_59000["source"],
            "v46_seed6_block63000": v46_63000["source"],
        },
    }

    candidate_union_rows: list[dict[str, Any]] = []
    candidate_values = list(validation.items())
    for index, signature in enumerate(next(iter(signatures))):
        successes = [name for name, value in candidate_values if value["per_episode"][index]["success"]]
        candidate_union_rows.append(
            {
                "episode": signature[0],
                "seed": signature[1],
                "traffic_variant": signature[2],
                "successful_pairs": successes,
                "any_pair_success": bool(successes),
            }
        )
    union_count = sum(row["any_pair_success"] for row in candidate_union_rows)

    t2 = ROOT / "results_topo_v4_3_dev" / "development" / "runs" / "T2__cand__cross__s0__pda6b283b"
    t3 = ROOT / "results_topo_v4_3_dev" / "development" / "runs" / "T3__cand__cross__s1__pda6b283b"
    target_runs = [
        _training_diagnostics(t2, _load(t2 / "final_evaluation.json")["success_rate"], 0),
        _training_diagnostics(t3, _load(t3 / "final_evaluation.json")["success_rate"], 1),
        _training_diagnostics(V45_RUN, v45_59000["outcomes"]["success_rate"], 4),
        _training_diagnostics(RUN, v46_63000["outcomes"]["success_rate"], 6),
    ]
    critic_success_spearman = _spearman(
        [row["critic_loss_last"] for row in target_runs],
        [row["target_decoder_validation_success_rate"] for row in target_runs],
    )
    seed4_diag = next(row for row in target_runs if row["seed"] == 4)
    seed6_diag = next(row for row in target_runs if row["seed"] == 6)
    optimizer_evidence = {
        "comparable_target_runs": target_runs,
        "comparability_scope": (
            "Cross exact-final target-critic decoder under the unchanged v4.2-v4.6 learner; "
            "validation blocks differ and n=4"
        ),
        "comparable_target_runs_spearman_critic_loss_vs_success": critic_success_spearman,
        "statistical_role": "diagnostic_covariation_only_not_a_preregistered_test",
        "seed6_vs_seed4": {
            "critic_loss_ratio": seed6_diag["critic_loss_last"] / seed4_diag["critic_loss_last"],
            "lane_entropy_seed4_over_seed6": seed4_diag["lane_entropy_last"]
            / seed6_diag["lane_entropy_last"],
            "topology_residual_scale_sign_changed": (
                math.copysign(1.0, seed4_diag["topology_residual_scale_last"])
                != math.copysign(1.0, seed6_diag["topology_residual_scale_last"])
            ),
        },
        "training_episode_outcomes": {
            "seed4": _monitor_outcomes(V45_RUN / "train_monitor.csv"),
            "seed6": _monitor_outcomes(RUN / "train_monitor.csv"),
        },
        "closed_loop_lane_entropy": {
            "seed4_good_block59000": _trace_metrics(v45_59000["trace_rows"]),
            "seed6_bad_block63000": _trace_metrics(v46_63000["trace_rows"]),
        },
    }

    calibration_variants = sorted(
        {row[2] for row in next(iter(calibration_signatures))}
    )
    validation_variants = sorted({row[2] for row in next(iter(signatures))})
    _require(not set(calibration_variants) & set(validation_variants), "traffic partitions overlap")
    calibration_features = [_traffic_features(name) for name in calibration_variants]
    validation_features = [_traffic_features(name) for name in validation_variants]
    traffic_evidence = {
        "calibration_train_partition": _traffic_group_summary(calibration_features),
        "validation_partition": _traffic_group_summary(validation_features),
        "calibration_variants": calibration_features,
        "validation_variants": validation_features,
        "interpretation": (
            "descriptive only; the crossed checkpoint/block intervention is the causal evidence "
            "against traffic difficulty as the sole explanation"
        ),
    }

    selected_key = "exact_final__target_critic"
    validation_oracle_key = validation_rank[0]
    selector_attribution = {
        "calibration_rank": calibration_rank,
        "validation_rank_under_same_lexicographic_rule": validation_rank,
        "calibration_selected_pair": selected_key,
        "validation_oracle_pair_post_hoc_only": validation_oracle_key,
        "selected_pair_validation_outcomes": validation_matrix[selected_key]["outcomes"],
        "oracle_pair_validation_outcomes": validation_matrix[validation_oracle_key]["outcomes"],
        "success_count_regret": (
            validation_matrix[validation_oracle_key]["outcome_counts"]["success_count"]
            - validation_matrix[selected_key]["outcome_counts"]["success_count"]
        ),
        "collision_count_regret": (
            validation_matrix[selected_key]["outcome_counts"]["collision_count"]
            - validation_matrix[validation_oracle_key]["outcome_counts"]["collision_count"]
        ),
        "paired_selected_target_to_final_fusion": _paired_effect(
            actual_target,
            validation["exact_final__fusion_0_90"],
            control_name="final_target",
            treatment_name="final_fusion",
        ),
        "all_pair_per_episode_success_oracle": {
            "success_count": union_count,
            "success_rate": union_count / 12.0,
            "passes_f1_success_gate": union_count / 12.0 >= 0.50,
            "rows": candidate_union_rows,
        },
        "interpretation": (
            "calibration misselection changes collision-versus-timeout composition but has zero "
            "success-count regret; no fixed eligible pair reaches the preregistered 0.50 success "
            "gate, while a leakage-only per-episode oracle over all four pairs reaches 7/12 and "
            "therefore reveals complementary failures rather than a deployable selector"
        ),
    }

    credit_coverage = _monitor_credit_coverage(RUN / "train_monitor.csv")
    payload = {
        "schema_version": "topo-scene-v4.6.f1-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "counted_for_development_gate": False,
        "counted_for_promotion_gate": False,
        "formal_test_accessed": False,
        "failed_job": "F1__cand__cross__s6__p814d53b7",
        "failure_gate": {
            "artifact_accepted": True,
            "selector_passed": True,
            "mechanism_passed": True,
            "outcome_passed": False,
            "failed_checks": ["success_rate", "collision_rate"],
            "development_decision": "fail",
            "stopped_after": "F1",
            "F2_F3_F4_executed": False,
            "formal_test_unlocked": False,
        },
        "primary_attribution": {
            "mechanism": (
                "training-seed-dependent critic/decision-weight instability under sparse terminal "
                "credit; seed6 remains worse under both validation blocks"
            ),
            "confidence": "strong_post_hoc_causal_weight_intervention_plus_diagnostic_mechanism_evidence",
            "causal_intervention": "two_frozen_weight_sets_crossed_with_two_paired_validation_blocks",
            "rejected_as_primary": [
                "artifact_or_decoder_implementation_failure",
                "checkpoint_choice_alone",
                "decoder_choice_alone",
                "validation_block_difficulty_alone",
                "absence_of_successful_training_episodes",
            ],
        },
        "calibration_pair_matrix": calibration,
        "validation_pair_matrix": validation_matrix,
        "selector_attribution": selector_attribution,
        "crossed_checkpoint_block_intervention": crossed,
        "optimizer_instability_evidence": optimizer_evidence,
        "traffic_variant_evidence": traffic_evidence,
        "terminal_credit_coverage": credit_coverage,
        "selected_next_hypothesis": {
            "version": "v4.7_horizon_correct_16_step_credit",
            "single_change": (
                "replace source-equivalent 4-step replay targets with horizon-correct 16-step "
                "returns using gamma raised to each sampled actual horizon"
            ),
            "why_this_change": (
                "increase direct terminal-credit coverage without changing reward, observations, "
                "actions, network, selector, or decoder"
            ),
            "unchanged": [
                "reward",
                "observation and action spaces",
                "v4.6 network and representation losses",
                "actor detach and entropy formulation",
                "warm-up and raw-step budget",
                "batch size and replay capacity",
                "joint checkpoint-decoder selector",
                "target and fusion decoder definitions",
                "development gates and formal lock",
            ],
            "fresh_confirmation_required": True,
            "formal_test_eligible": False,
        },
        "source_integrity": {
            "experiment_contract": _source(CONTRACT),
            "implementation_freeze": _source(FREEZE),
            "development_decision": _source(DECISION),
            "selector_receipt": _source(RECEIPT),
            "posthoc_replay_tool": _source(REPLAY_TOOL),
        },
    }
    summary_path = ATTRIBUTION / "attribution_summary.json"
    markdown_path = ATTRIBUTION / "deep_attribution.md"
    _write_json(summary_path, payload)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "markdown": str(markdown_path),
                "markdown_sha256": _sha256(markdown_path),
                "primary_attribution": payload["primary_attribution"],
                "selected_next_hypothesis": payload["selected_next_hypothesis"]["version"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
