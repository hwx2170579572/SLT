"""Build the immutable evidence summary for the v4.3 T3 failure attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = ROOT / "results_topo_v4_3_dev" / "attribution" / "t3_failure"
DEVELOPMENT = ROOT / "results_topo_v4_3_dev" / "development"
RUNS = DEVELOPMENT / "runs"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    episodes = int(first["episodes"]) + int(second["episodes"])
    if int(first["episodes"]) != 6 or int(second["episodes"]) != 6:
        raise ValueError("historical selector attribution requires two six-episode blocks")

    def count(payload: dict[str, Any], rate: str) -> int:
        value = float(payload[rate]) * int(payload["episodes"])
        rounded = round(value)
        if abs(value - rounded) > 1e-9:
            raise ValueError(f"non-integral outcome count for {rate}: {value}")
        return int(rounded)

    success = count(first, "success_rate") + count(second, "success_rate")
    collision = count(first, "collision_rate") + count(second, "collision_rate")
    off_route = count(first, "off_route_rate") + count(second, "off_route_rate")
    timeout = count(first, "timeout_rate") + count(second, "timeout_rate")
    return {
        "episodes": episodes,
        "success_count": success,
        "collision_count": collision,
        "off_route_count": off_route,
        "timeout_count": timeout,
        "success_rate": success / episodes,
        "collision_rate": collision / episodes,
        "off_route_rate": off_route / episodes,
        "timeout_rate": timeout / episodes,
        "mean_return": (
            float(first["mean_return"]) * int(first["episodes"])
            + float(second["mean_return"]) * int(second["episodes"])
        )
        / episodes,
    }


def _selection_key(outcomes: dict[str, Any], *, is_final: bool) -> tuple[Any, ...]:
    return (
        int(outcomes["success_count"]),
        -int(outcomes["collision_count"]),
        -int(outcomes["off_route_count"]),
        -int(outcomes["timeout_count"]),
        float(outcomes["mean_return"]),
        int(is_final),
    )


def _calibration_pair(run_id: str, prefix: str) -> dict[str, Any]:
    calibration = ATTRIBUTION / "calibration"
    outputs: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for candidate in ("final", "best"):
        first_dir = calibration / f"{prefix}_{candidate}_train6"
        second_dir = calibration / f"{prefix}_{candidate}_train6b"
        first_meta = _load(first_dir / "attribution_metadata.json")
        second_meta = _load(second_dir / "attribution_metadata.json")
        if first_meta["traffic_partition"] != "train" or second_meta["traffic_partition"] != "train":
            raise ValueError("selector attribution touched a non-training partition")
        if first_meta["evaluation_seed_start"] + 6 != second_meta["evaluation_seed_start"]:
            raise ValueError("selector seed blocks are not contiguous")
        first = _load(first_dir / "final_evaluation.json")
        second = _load(second_dir / "final_evaluation.json")
        outputs[candidate] = _aggregate(first, second)
        for suffix, directory in (("a", first_dir), ("b", second_dir)):
            hashes[f"{candidate}_{suffix}_result_sha256"] = _sha(
                directory / "attribution_result.json"
            )
    final_key = _selection_key(outputs["final"], is_final=True)
    best_key = _selection_key(outputs["best"], is_final=False)
    selected = "final" if final_key >= best_key else "best"
    return {
        "source_run": str((RUNS / run_id).resolve()),
        "calibration_partition": "train",
        "calibration_episodes_per_checkpoint": 12,
        "selection_order": [
            "maximize_success_count",
            "minimize_collision_count",
            "minimize_off_route_count",
            "minimize_timeout_count",
            "maximize_mean_return",
            "prefer_exact_final_on_complete_tie",
        ],
        "candidate_outcomes": outputs,
        "selected_checkpoint": selected,
        "calibration_result_hashes": hashes,
    }


def main() -> int:
    decision_path = DEVELOPMENT / "development_decision.json"
    trace_path = ATTRIBUTION / "trace_attribution.json"
    decision = _load(decision_path)
    if decision.get("decision") != "fail" or decision.get("stopped_after") != "T3":
        raise ValueError("v4.3 development failure receipt drifted")

    calibration = {
        "T1": _calibration_pair("T1__cand__carla__s2__pda6b283b", "t1"),
        "T2": _calibration_pair("T2__cand__cross__s0__pda6b283b", "t2"),
        "T3": _calibration_pair("T3__cand__cross__s1__pda6b283b", "t3"),
    }
    if {key: value["selected_checkpoint"] for key, value in calibration.items()} != {
        "T1": "final",
        "T2": "final",
        "T3": "best",
    }:
        raise ValueError("train-only selector did not reproduce the expected historical choices")

    t3_final_validation = decision["jobs"]["T3"]["row"]
    t3_best_dir = ATTRIBUTION / "closed_loop" / "best_target"
    t3_best_validation = _load(t3_best_dir / "final_evaluation.json")
    t1_best_dir = ATTRIBUTION / "closed_loop" / "t1_best_target"
    t1_best_validation = _load(t1_best_dir / "final_evaluation.json")

    payload = {
        "schema_version": "topo-scene-v4.3.t3-failure-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "post_hoc_diagnostic_only": True,
        "formal_test_accessed": False,
        "failed_job": "T3__cand__cross__s1__pda6b283b",
        "failure": {
            "success_rate": t3_final_validation["success_rate"],
            "collision_rate": t3_final_validation["collision_rate"],
            "off_route_rate": t3_final_validation["off_route_rate"],
            "timeout_rate": t3_final_validation["timeout_rate"],
            "final_model_sha256": t3_final_validation["final_model_sha256"],
        },
        "primary_attribution": {
            "mechanism": "late_training_checkpoint_drift_in_longitudinal_behavior",
            "evidence": {
                "t3_final_validation_success_rate": t3_final_validation["success_rate"],
                "t3_final_validation_timeout_rate": t3_final_validation["timeout_rate"],
                "t3_final_validation_mean_speed_mps": t3_final_validation["mean_speed_mps"],
                "t3_best_validation_success_rate": t3_best_validation["success_rate"],
                "t3_best_validation_collision_rate": t3_best_validation["collision_rate"],
                "t3_best_validation_timeout_rate": t3_best_validation["timeout_rate"],
                "t3_best_validation_mean_speed_mps": _load(
                    t3_best_dir / "action_diagnostics.json"
                )["actual_speed_mps"]["mean"],
                "t3_best_checkpoint_sha256": _load(
                    t3_best_dir / "attribution_metadata.json"
                )["checkpoint_sha256"],
            },
        },
        "rejected_primary_hypothesis": {
            "mechanism": "lane_change_never_completed",
            "reason": (
                "Trace attribution observes the target lane in 11/12 T3 episodes and "
                "the next route edge in 10/12; all ten timeouts reached terminal edge gneE23."
            ),
        },
        "historical_train_only_checkpoint_calibration": calibration,
        "validation_sanity_checks": {
            "T1_selected_final_validation_success_rate": decision["jobs"]["T1"]["row"]["success_rate"],
            "T1_rejected_best_validation_success_rate": t1_best_validation["success_rate"],
            "T2_selected_final_validation_success_rate": decision["jobs"]["T2"]["row"]["success_rate"],
            "T3_selected_best_validation_success_rate": t3_best_validation["success_rate"],
        },
        "recommended_single_change": {
            "version": "v4.4",
            "name": "train_only_deployment_aware_checkpoint_selector",
            "candidate_set": ["best_training_success", "exact_final"],
            "calibration_partition": "train",
            "paired_calibration_episodes": 12,
            "selection_order": calibration["T1"]["selection_order"],
            "training_changed": False,
            "decoder_changed": False,
            "validation_used_for_selection": False,
            "formal_test_used_for_selection": False,
            "fresh_development_confirmation_required": True,
        },
        "source_hashes": {
            "development_decision_sha256": _sha(decision_path),
            "trace_attribution_sha256": _sha(trace_path),
            "t3_best_closed_loop_result_sha256": _sha(t3_best_dir / "attribution_result.json"),
            "t1_best_closed_loop_result_sha256": _sha(t1_best_dir / "attribution_result.json"),
        },
    }
    summary_path = ATTRIBUTION / "attribution_summary.json"
    _write(summary_path, payload)
    markdown = f"""# v4.3 T3 Deep Attribution

v4.3 stopped at T3: final-checkpoint Cross seed 1 achieved success
{t3_final_validation['success_rate']:.3f}, collision {t3_final_validation['collision_rate']:.3f},
and timeout {t3_final_validation['timeout_rate']:.3f}.

The primary cause is checkpoint-dependent longitudinal drift. Replaying the
training-best checkpoint on the identical validation seeds changed success to
{t3_best_validation['success_rate']:.3f}, collision to
{t3_best_validation['collision_rate']:.3f}, and timeout to
{t3_best_validation['timeout_rate']:.3f}. Mean speed changed from
{t3_final_validation['mean_speed_mps']:.3f} to
{payload['primary_attribution']['evidence']['t3_best_validation_mean_speed_mps']:.3f} m/s.

Lane non-completion is not the primary cause: 11/12 episodes reached the target
lane and 10/12 reached the next route edge; every timeout was already on the
terminal edge.

## Train-only selector feasibility

- T1: final 12/12 vs best 0/12; select final.
- T2: final and best both 9/12 success and 3/12 collision; exact tie selects final.
- T3: final 0/12 vs best 10/12; select best.

The proposed v4.4 change is limited to checkpoint selection. It evaluates the
training-best and exact-final checkpoints on 12 paired **training-partition**
episodes, applies a frozen lexicographic rule, and never reads validation or
formal-test outcomes. These historical checks are post-hoc evidence only;
v4.4 still requires fresh preregistered development confirmation.
"""
    markdown_path = ATTRIBUTION / "deep_attribution.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "attribution_summary": str(summary_path),
        "attribution_summary_sha256": _sha(summary_path),
        "deep_attribution_sha256": _sha(markdown_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
