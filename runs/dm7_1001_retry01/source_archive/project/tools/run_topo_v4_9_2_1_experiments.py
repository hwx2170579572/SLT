"""Read-only v4.9.2.1 acceptance patch for completed v4.9.2 runs.

The frozen v4.9.2 runner required ``paper_evaluation_detailed.json`` to carry
``formal_test_accessed``.  The training writer instead records the stronger
underlying evidence: the evaluation split/provenance and the formal-unlock
receipt in ``arguments.json``.  Consequently scientifically valid promotion
runs were rejected after successful training.

This engineering-only patch does not train, edit, rename, or delete a run.  It
revalidates the immutable v4.9.2 run tree from those recorded provenance
signals and writes a separately versioned summary and gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_9 import V49_CANDIDATE, V49_FORMAL_ALGORITHMS
from tools import run_topo_v4_9_2_experiments as frozen


PATCH_VERSION = "v4.9.2.1"
PATCH_KIND = "acceptance_metadata_only"
PATCH_ROOT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_patch_v4_9_2_1"
)
DEFAULT_SUMMARY = PATCH_ROOT / "summary.json"
DEFAULT_GATE = PATCH_ROOT / "promotion_gate.json"
DEFAULT_REPORT = PATCH_ROOT / "revalidation_report.json"
FROZEN_RUNNER_SHA256 = (
    "6c3f77ecae3403f12eca4204cbed9f7d99947389674202d4424516cf61cafb25"
)


class V4921EngineeringError(ValueError):
    """Raised when immutable run evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4921EngineeringError(message)


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _recorded_formal_access(root: Path, job: frozen.Job) -> dict[str, Any]:
    """Derive formal access from the actual split and unlock provenance.

    An explicit marker, when present, is treated as a redundant assertion and
    must agree.  Its absence is not an error because all underlying evidence
    is independently sealed in the run.
    """

    detailed = _load_json(root / "paper_evaluation_detailed.json")
    arguments = _load_json(root / "arguments.json")
    requested = arguments.get("requested_raw_steps", {})
    provenance = detailed.get("evaluation_provenance", {})
    _require(isinstance(requested, dict), "requested_raw_steps is not a mapping")
    _require(isinstance(provenance, dict), "evaluation_provenance is not a mapping")

    expected_access = job.stage == "formal"
    expected_split = "test" if expected_access else "validation"
    split_evidence = {
        "job": job.evaluation_split,
        "detailed": detailed.get("evaluation_split"),
        "arguments": requested.get("evaluation_split"),
        "provenance_evaluation_split": provenance.get("evaluation_split"),
        "provenance_traffic_partition": provenance.get("traffic_partition"),
    }
    for name, value in split_evidence.items():
        _require(
            value == expected_split,
            f"formal evidence {name} split drifted: {value!r}",
        )

    unlock_receipt = requested.get("formal_unlock_receipt")
    unlock_binding = arguments.get("formal_unlock")
    if expected_access:
        _require(
            isinstance(unlock_receipt, str) and bool(unlock_receipt),
            "formal run lacks unlock receipt path",
        )
        _require(
            isinstance(unlock_binding, dict) and bool(unlock_binding),
            "formal run lacks verified unlock binding",
        )
    else:
        _require(unlock_receipt is None, "promotion run records formal unlock receipt")
        _require(unlock_binding is None, "promotion run records formal unlock binding")

    explicit_present = "formal_test_accessed" in detailed
    if explicit_present:
        _require(
            detailed["formal_test_accessed"] is expected_access,
            "explicit formal access marker contradicts sealed provenance",
        )

    return {
        "formal_test_accessed": expected_access,
        "derivation": (
            "explicit_marker_plus_split_and_unlock_provenance"
            if explicit_present
            else "split_and_unlock_provenance"
        ),
        "explicit_marker_present": explicit_present,
        "split_evidence": split_evidence,
        "formal_unlock_receipt_present": unlock_receipt is not None,
        "formal_unlock_binding_present": unlock_binding is not None,
    }


def accepted_run_reason(
    job: frozen.Job,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    """Apply every frozen validator except the redundant missing-field test."""

    root = frozen.run_directory(job)
    try:
        _require(root.is_dir(), "run directory missing")
        for relative in frozen.FULL_RUN_REQUIRED_FILES:
            _require(
                (root / relative).is_file(),
                f"required artifact missing: {relative}",
            )
        frozen._validate_arguments(root, job, hashes)
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        _require(
            detailed.get("schema_version")
            == "topo-scene-v4.9.2.detailed-evaluation/v1",
            "detailed schema drifted",
        )
        _recorded_formal_access(root, job)
        if job.algorithm == V49_CANDIDATE:
            _require(
                (root / "collision_label_diagnostics.json").is_file(),
                "collision diagnostics missing",
            )
            frozen._validate_candidate(root, job)
        else:
            frozen._validate_control(root)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        frozen.V492ProtocolError,
        V4921EngineeringError,
        json.JSONDecodeError,
    ) as exc:
        return False, str(exc)
    return True, "accepted_by_v4_9_2_1_provenance_patch"


def accepted_run(job: frozen.Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


def _run_row(job: frozen.Job) -> dict[str, Any]:
    row = frozen._run_row(job)
    evidence = _recorded_formal_access(frozen.run_directory(job), job)
    row.update(
        {
            "acceptance_patch_version": PATCH_VERSION,
            "acceptance_patch_kind": PATCH_KIND,
            "formal_access_derivation": evidence["derivation"],
            "formal_access_explicit_marker_present": evidence[
                "explicit_marker_present"
            ],
        }
    )
    return row


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def _artifact_manifest(
    jobs: list[frozen.Job],
) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for job in jobs:
        root = frozen.run_directory(job)
        if not root.is_dir():
            continue
        relatives = list(frozen.FULL_RUN_REQUIRED_FILES)
        collision = "collision_label_diagnostics.json"
        if (root / collision).is_file() and collision not in relatives:
            relatives.append(collision)
        files: dict[str, str] = {}
        for relative in sorted(set(relatives)):
            path = root / relative
            if path.is_file():
                files[relative] = frozen._sha256(path)
        manifest[job.name] = files
    return manifest


def summarize_stage(
    contract: dict[str, Any],
    digest: str,
    stage: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    jobs = frozen.jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job) for job in jobs if accepted_run(job, hashes)]
    manifest = _artifact_manifest(jobs)
    payload = {
        "schema_version": "topo-scene-v4.9.2.1.stage-results/v1",
        "patch_version": PATCH_VERSION,
        "patch_kind": PATCH_KIND,
        "stage": stage,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_tree_modified": False,
        "source_runner_sha256": frozen._sha256(Path(frozen.__file__)),
        "source_runner_matches_freeze": (
            frozen._sha256(Path(frozen.__file__)) == FROZEN_RUNNER_SHA256
        ),
        "formal_test_accessed": stage == "formal" and bool(rows),
        **hashes,
        "accepted_runs": len(rows),
        "expected_runs": len(jobs),
        "complete": len(rows) == len(jobs),
        "explicit_formal_marker_missing_runs": sum(
            not bool(row["formal_access_explicit_marker_present"])
            for row in rows
        ),
        "source_artifact_manifest": manifest,
        "source_artifact_manifest_sha256": _digest_json(manifest),
        "per_run": rows,
        "aggregate": frozen.aggregate_rows(rows),
    }
    _write_json(PATCH_ROOT / f"{stage}_summary.json", payload)
    if stage == "promotion":
        _write_json(DEFAULT_SUMMARY, payload)
    return payload


def compute_promotion_gate(
    contract: dict[str, Any],
    digest: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    parent_contract = frozen.parent.validate_contract(
        frozen.parent.load_contract(frozen.PARENT_CONTRACT)
    )
    rows = summary["per_run"]
    aggregate = {
        (row["method"], row["scenario"]): row
        for row in summary["aggregate"]
    }
    scenarios = contract["promotion"]["scenarios"]
    success = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"]
        - aggregate[("temporal_graph", scenario)]["success_rate"]
        for scenario in scenarios
    }
    collision = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"]
        - aggregate[("temporal_graph", scenario)]["collision_rate"]
        for scenario in scenarios
    }
    paired = frozen._paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {
        f"{row['scenario']}__seed{row['seed']}": frozen._candidate_gate(
            row, parent_contract
        )
        for row in rows
        if row["method"] == "selected_v4_candidate"
    }
    checks = {
        "candidate_development_guards_each_seed": all(
            gate["passed"] for gate in candidate_gates.values()
        ),
        "all_candidate_off_route_zero": all(
            aggregate[("selected_v4_candidate", scenario)]["off_route_rate"]
            <= 0.0
            for scenario in scenarios
        ),
        "per_scenario_collision_noninferiority": all(
            delta
            <= float(rules["per_scenario_collision_noninferiority_margin"])
            for delta in collision.values()
        ),
        "worst_paired_seed_success_noninferiority": min(
            row["success_rate_delta"] for row in paired
        )
        >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success.values())
        >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": max(success.values())
        >= float(rules["positive_effect_success_gain"])
        or min(collision.values())
        <= -float(rules["positive_effect_collision_reduction"]),
        "train_only_target_only_selection_applied": all(
            row["selected_checkpoint_kind"]
            in ("highest_training_success", "exact_final")
            and row["selected_deployment_decoder"] == "target_critic"
            and row.get("fusion_candidate_present") is False
            for row in rows
            if row["method"] == "selected_v4_candidate"
        ),
    }
    passed = all(checks.values())
    receipt = {
        "schema_version": "topo-scene-v4.9.2.1.promotion-gate/v1",
        "patch_version": PATCH_VERSION,
        "patch_kind": PATCH_KIND,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_tree_modified": False,
        "decision": "pass" if passed else "fail",
        **hashes,
        "promotion_results": str(DEFAULT_SUMMARY.resolve()),
        "promotion_results_sha256": frozen._sha256(DEFAULT_SUMMARY),
        "source_artifact_manifest_sha256": summary[
            "source_artifact_manifest_sha256"
        ],
        "formal_algorithms": list(V49_FORMAL_ALGORITHMS),
        "checks": checks,
        "candidate_per_seed_gates": candidate_gates,
        "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision,
        "worst_paired_seed_success_delta": min(
            row["success_rate_delta"] for row in paired
        ),
        "macro_success_delta": _mean(success.values()),
        "paired_seed_deltas": paired,
        "formal_test_unlocked": passed,
        "formal_test_accessed": False,
    }
    _write_json(DEFAULT_GATE, receipt)
    return receipt


def revalidate_promotion() -> tuple[int, dict[str, Any]]:
    _require(
        frozen._sha256(Path(frozen.__file__)) == FROZEN_RUNNER_SHA256,
        "frozen v4.9.2 runner changed; engineering patch cannot be applied",
    )
    contract = frozen.validate_contract(frozen.load_contract())
    digest = frozen._sha256(frozen.DEFAULT_CONTRACT)
    hashes = frozen.protocol_hashes()
    jobs = frozen.jobs_for_stage(contract, digest, "promotion")
    rows = []
    for job in jobs:
        accepted, reason = accepted_run_reason(job, hashes)
        rows.append(
            {
                "job": job.name,
                "method": job.method,
                "scenario": job.scenario,
                "seed": job.seed,
                "accepted": accepted,
                "acceptance_reason": reason,
                "source_run_directory": str(frozen.run_directory(job).resolve()),
            }
        )
    accepted_count = sum(row["accepted"] for row in rows)
    complete = accepted_count == len(jobs)
    summary = summarize_stage(contract, digest, "promotion", hashes)
    gate = compute_promotion_gate(contract, digest, hashes) if complete else None
    status = (
        "promotion_gate_passed"
        if gate and gate["decision"] == "pass"
        else "promotion_gate_failed"
        if gate
        else "promotion_runs_incomplete"
    )
    report = {
        "schema_version": "topo-scene-v4.9.2.1.revalidation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_version": PATCH_VERSION,
        "patch_kind": PATCH_KIND,
        "status": status,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_tree_modified": False,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        **hashes,
        "source_runner_sha256": frozen._sha256(Path(frozen.__file__)),
        "patch_module_sha256": frozen._sha256(Path(__file__)),
        "jobs": rows,
        "accepted_jobs": accepted_count,
        "expected_jobs": len(jobs),
        "promotion_matrix_complete": complete,
        "summary_sha256": frozen._sha256(DEFAULT_SUMMARY),
        "promotion_gate_decision": gate.get("decision") if gate else None,
        "promotion_gate_sha256": frozen._sha256(DEFAULT_GATE) if gate else None,
    }
    _write_json(DEFAULT_REPORT, report)
    return (0 if gate and gate["decision"] == "pass" else 2 if gate else 1), report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument(
        "action",
        nargs="?",
        choices=("status", "aggregate"),
        default="aggregate",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, report = revalidate_promotion()
    if args.action == "status" and not report["promotion_matrix_complete"]:
        code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GATE",
    "DEFAULT_REPORT",
    "DEFAULT_SUMMARY",
    "PATCH_KIND",
    "PATCH_ROOT",
    "PATCH_VERSION",
    "accepted_run",
    "accepted_run_reason",
    "compute_promotion_gate",
    "revalidate_promotion",
    "summarize_stage",
]
