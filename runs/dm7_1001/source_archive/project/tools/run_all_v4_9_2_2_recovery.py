"""Recover interrupted v4.9.2 promotion jobs without overwriting evidence.

For each of the 12 frozen logical jobs, this runner accepts the canonical run
when complete.  A missing canonical run is executed at its canonical path; an
existing but incomplete path is immutable and causes an identical fresh run in
a versioned recovery directory.  Every logical job is attempted even if an
earlier process fails.  Aggregation happens only after all attempts finish.

This is an engineering recovery layer: commands, seeds, budgets, model,
selection, partitions, hashes, and scientific gates remain frozen v4.9.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_9 import V49_CANDIDATE, V49_FORMAL_ALGORITHMS
from tools import run_topo_v4_9_2_1_experiments as acceptance
from tools import run_topo_v4_9_2_experiments as frozen


RECOVERY_VERSION = "v4.9.2.2"
RECOVERY_KIND = "immutable_interrupted_run_reexecution"
RECOVERY_ROOT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_recovery_v4_9_2_2"
)
RECOVERY_RUN_ROOT = RECOVERY_ROOT / "runs"
RECOVERY_LOG_ROOT = RECOVERY_ROOT / "launcher_logs"
DEFAULT_EXECUTION = RECOVERY_ROOT / "last_execution.json"
DEFAULT_SUMMARY = RECOVERY_ROOT / "summary.json"
DEFAULT_GATE = RECOVERY_ROOT / "promotion_gate.json"
DEFAULT_REPORT = RECOVERY_ROOT / "all_promotion_attempt_summary.json"
MAX_RECOVERY_ATTEMPTS_PER_JOB = 8


class V4922RecoveryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4922RecoveryError(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_name(job: frozen.Job, attempt: int) -> str:
    _require(1 <= attempt <= MAX_RECOVERY_ATTEMPTS_PER_JOB, "invalid attempt")
    return f"{job.name}__recovery_r{attempt}"


def recovery_directory(job: frozen.Job, attempt: int) -> Path:
    return RECOVERY_RUN_ROOT / recovery_name(job, attempt)


def candidate_directories(job: frozen.Job) -> list[tuple[str, Path]]:
    values = [("canonical", frozen.run_directory(job))]
    values.extend(
        (f"recovery_r{attempt}", recovery_directory(job, attempt))
        for attempt in range(1, MAX_RECOVERY_ATTEMPTS_PER_JOB + 1)
    )
    return values


def accepted_root_reason(
    job: frozen.Job,
    hashes: dict[str, str],
    root: Path,
) -> tuple[bool, str]:
    try:
        _require(root.is_dir(), "run directory missing")
        for relative in frozen.FULL_RUN_REQUIRED_FILES:
            _require(
                (root / relative).is_file(),
                f"required artifact missing: {relative}",
            )
        frozen._validate_arguments(root, job, hashes)
        detailed = frozen._load_json(root / "paper_evaluation_detailed.json")
        _require(
            detailed.get("schema_version")
            == "topo-scene-v4.9.2.detailed-evaluation/v1",
            "detailed schema drifted",
        )
        acceptance._recorded_formal_access(root, job)
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
        acceptance.V4921EngineeringError,
        V4922RecoveryError,
        json.JSONDecodeError,
    ) as exc:
        return False, str(exc)
    return True, "accepted_by_v4_9_2_2_recovery_layer"


def selected_source(
    job: frozen.Job,
    hashes: dict[str, str],
) -> tuple[str | None, Path | None, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    for kind, root in candidate_directories(job):
        if not root.exists():
            continue
        accepted, reason = accepted_root_reason(job, hashes, root)
        evidence.append(
            {
                "kind": kind,
                "path": str(root.resolve()),
                "accepted": accepted,
                "reason": reason,
            }
        )
        if accepted:
            return kind, root, evidence
    return None, None, evidence


def _next_target(job: frozen.Job) -> tuple[str, Path, int | None]:
    canonical = frozen.run_directory(job)
    if not canonical.exists():
        return "canonical", canonical, None
    for attempt in range(1, MAX_RECOVERY_ATTEMPTS_PER_JOB + 1):
        root = recovery_directory(job, attempt)
        if not root.exists():
            return f"recovery_r{attempt}", root, attempt
    raise V4922RecoveryError(
        f"all recovery slots exhausted for {job.name}; preserved every attempt"
    )


def command_for_target(
    job: frozen.Job,
    hashes: dict[str, str],
    *,
    device: str,
    target_root: Path,
) -> list[str]:
    command = frozen.command_for(job, hashes, device=device)
    output_index = command.index("--output-dir") + 1
    name_index = command.index("--model-name") + 1
    command[output_index] = str(target_root.parent.resolve())
    command[name_index] = target_root.name
    return command


def execute_all(
    contract: dict[str, Any],
    digest: str,
    hashes: dict[str, str],
    *,
    device: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    jobs = frozen.jobs_for_stage(contract, digest, "promotion")
    attempts: list[dict[str, Any]] = []
    RECOVERY_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        selected_kind, selected_root, sources = selected_source(job, hashes)
        if selected_root is not None:
            attempts.append(
                {
                    "job": job.name,
                    "status": "already_accepted",
                    "selected_source_kind": selected_kind,
                    "selected_source_directory": str(selected_root.resolve()),
                    "available_sources": sources,
                }
            )
            continue
        try:
            target_kind, target_root, recovery_attempt = _next_target(job)
        except V4922RecoveryError as exc:
            attempts.append(
                {
                    "job": job.name,
                    "status": "recovery_slots_exhausted",
                    "error": str(exc),
                    "available_sources": sources,
                }
            )
            continue
        command = command_for_target(
            job,
            hashes,
            device=device,
            target_root=target_root,
        )
        log_path = RECOVERY_LOG_ROOT / f"{target_root.name}.log"
        start_record = {
            "job": job.name,
            "status": "planned" if dry_run else "running",
            "target_kind": target_kind,
            "recovery_attempt": recovery_attempt,
            "target_directory": str(target_root.resolve()),
            "log": str(log_path.resolve()),
            "command": command,
            "preserved_incomplete_sources": sources,
        }
        print(json.dumps(start_record, ensure_ascii=False), flush=True)
        if dry_run:
            attempts.append(start_record)
            continue
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        accepted, reason = accepted_root_reason(job, hashes, target_root)
        result = {
            **start_record,
            "status": (
                "accepted"
                if completed.returncode == 0 and accepted
                else "failed"
            ),
            "returncode": completed.returncode,
            "accepted": accepted,
            "acceptance_reason": reason,
        }
        attempts.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return attempts


@contextmanager
def _source_view(root: Path) -> Iterator[None]:
    original = frozen.run_directory
    frozen.run_directory = lambda _job: root
    try:
        yield
    finally:
        frozen.run_directory = original


def _run_row(job: frozen.Job, root: Path, kind: str) -> dict[str, Any]:
    with _source_view(root):
        row = frozen._run_row(job)
    formal = acceptance._recorded_formal_access(root, job)
    row.update(
        {
            "recovery_layer_version": RECOVERY_VERSION,
            "logical_job_name": job.name,
            "selected_source_kind": kind,
            "selected_source_directory": str(root.resolve()),
            "source_run_recovered": kind != "canonical",
            "formal_access_derivation": formal["derivation"],
        }
    )
    return row


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def _selected_manifest(
    jobs: list[frozen.Job],
    hashes: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    for job in jobs:
        kind, root, evidence = selected_source(job, hashes)
        if root is None or kind is None:
            unresolved.append({"job": job.name, "sources": evidence})
            continue
        relatives = list(frozen.FULL_RUN_REQUIRED_FILES)
        if (root / "collision_label_diagnostics.json").is_file():
            relatives.append("collision_label_diagnostics.json")
        manifest[job.name] = {
            "kind": kind,
            "root": str(root.resolve()),
            "files": {
                relative: frozen._sha256(root / relative)
                for relative in sorted(set(relatives))
                if (root / relative).is_file()
            },
        }
    return manifest, unresolved


def summarize(
    contract: dict[str, Any],
    digest: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    jobs = frozen.jobs_for_stage(contract, digest, "promotion")
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for job in jobs:
        kind, root, evidence = selected_source(job, hashes)
        source_rows.append(
            {
                "job": job.name,
                "selected_source_kind": kind,
                "selected_source_directory": (
                    str(root.resolve()) if root is not None else None
                ),
                "available_sources": evidence,
            }
        )
        if root is not None and kind is not None:
            rows.append(_run_row(job, root, kind))
    manifest, unresolved = _selected_manifest(jobs, hashes)
    payload = {
        "schema_version": "topo-scene-v4.9.2.2.stage-results/v1",
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "stage": "promotion",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        **hashes,
        "accepted_runs": len(rows),
        "expected_runs": len(jobs),
        "complete": len(rows) == len(jobs),
        "recovered_run_count": sum(
            row["source_run_recovered"] for row in rows
        ),
        "source_selection": source_rows,
        "unresolved": unresolved,
        "selected_artifact_manifest": manifest,
        "selected_artifact_manifest_sha256": _digest_json(manifest),
        "per_run": rows,
        "aggregate": frozen.aggregate_rows(rows),
    }
    _write_json(DEFAULT_SUMMARY, payload)
    return payload


def compute_gate(
    contract: dict[str, Any],
    digest: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    summary = summarize(contract, digest, hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
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
    parent_contract = frozen.parent.validate_contract(
        frozen.parent.load_contract(frozen.PARENT_CONTRACT)
    )
    candidate_gates = {
        f"{row['scenario']}__seed{row['seed']}": frozen._candidate_gate(
            row, parent_contract
        )
        for row in rows
        if row["method"] == "selected_v4_candidate"
    }
    rules = contract["promotion"]["thresholds"]
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
        "schema_version": "topo-scene-v4.9.2.2.promotion-gate/v1",
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "decision": "pass" if passed else "fail",
        **hashes,
        "promotion_results": str(DEFAULT_SUMMARY.resolve()),
        "promotion_results_sha256": frozen._sha256(DEFAULT_SUMMARY),
        "selected_artifact_manifest_sha256": summary[
            "selected_artifact_manifest_sha256"
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


def run_all(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    contract = frozen.validate_contract(frozen.load_contract())
    digest = frozen._sha256(frozen.DEFAULT_CONTRACT)
    hashes = frozen.protocol_hashes()
    attempts = execute_all(
        contract,
        digest,
        hashes,
        device=device,
        dry_run=dry_run,
    )
    if dry_run:
        report = {
            "schema_version": "topo-scene-v4.9.2.2.recovery-plan/v1",
            "execution_policy": "attempt_all_then_summarize",
            "failure_does_not_cancel_remaining_jobs": True,
            "attempts": attempts,
            "formal_stage_launched": False,
        }
        _write_json(report_path.resolve(), report)
        return 0, report

    execution = {
        "schema_version": "topo-scene-v4.9.2.2.execution/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        **hashes,
        "attempts": attempts,
    }
    _write_json(DEFAULT_EXECUTION, execution)
    summary = summarize(contract, digest, hashes)
    gate = compute_gate(contract, digest, hashes) if summary["complete"] else None
    status = (
        "promotion_gate_passed"
        if gate and gate["decision"] == "pass"
        else "promotion_gate_failed"
        if gate
        else "promotion_jobs_incomplete_after_all_attempts"
    )
    report = {
        "schema_version": "topo-scene-v4.9.2.2.all-promotion-automation/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        **hashes,
        "attempts": attempts,
        "accepted_jobs": summary["accepted_runs"],
        "expected_jobs": summary["expected_runs"],
        "promotion_matrix_complete": summary["complete"],
        "recovered_run_count": summary["recovered_run_count"],
        "summary_sha256": frozen._sha256(DEFAULT_SUMMARY),
        "promotion_gate_decision": gate.get("decision") if gate else None,
        "promotion_gate_sha256": frozen._sha256(DEFAULT_GATE) if gate else None,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": bool(gate and gate["decision"] == "pass"),
    }
    _write_json(report_path.resolve(), report)
    return (0 if gate and gate["decision"] == "pass" else 2 if gate else 1), report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    value.add_argument("--dry-run", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, report = run_all(
        device=args.device,
        report_path=args.report,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GATE",
    "DEFAULT_REPORT",
    "DEFAULT_SUMMARY",
    "RECOVERY_ROOT",
    "RECOVERY_RUN_ROOT",
    "RECOVERY_VERSION",
    "accepted_root_reason",
    "candidate_directories",
    "command_for_target",
    "recovery_directory",
    "recovery_name",
    "run_all",
    "selected_source",
]
