"""Recover interrupted v4.11 promotion jobs without overwriting evidence.

This is an engineering-only execution recovery for the frozen v4.11 PRCR
promotion.  Canonical complete runs are reused.  Canonical incomplete runs are
kept immutable and the same logical job is restarted from scratch in a
versioned recovery directory.  Seeds, budgets, algorithms, partitions,
selection, model code, protocol hashes, and scientific gates remain v4.11.

Every unresolved logical job is attempted even when an earlier child fails.
Only after all attempts finish are the twelve logical jobs source-mapped,
summarized, and passed to the frozen v4.11 promotion gate.  Formal testing is
never launched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_11_experiments as frozen


RECOVERY_VERSION = "v4.11.1"
RECOVERY_KIND = "immutable_interrupted_run_reexecution"
RECOVERY_ROOT = (
    ROOT
    / "results_topo_v4_11_promotion"
    / "engineering_recovery_v4_11_1"
)
RECOVERY_RUN_ROOT = RECOVERY_ROOT / "runs"
RECOVERY_LOG_ROOT = RECOVERY_ROOT / "launcher_logs"
DEFAULT_EXECUTION = RECOVERY_ROOT / "last_execution.json"
DEFAULT_SUMMARY = RECOVERY_ROOT / "summary.json"
DEFAULT_REPORT = (
    ROOT / "results_promotion_automation" / "v4_11_1_pipeline_summary.json"
)
MAX_RECOVERY_ATTEMPTS_PER_JOB = 8


class V4111RecoveryError(ValueError):
    """Raised when immutable v4.11 recovery evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4111RecoveryError(message)


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


@contextmanager
def _source_view(selected: dict[str, Path]) -> Iterator[None]:
    """Temporarily map promotion logical jobs to immutable accepted sources."""

    original = frozen.run_directory

    def mapped(job: frozen.Job) -> Path:
        if job.stage == "promotion" and job.name in selected:
            return selected[job.name]
        return original(job)

    frozen.run_directory = mapped
    try:
        yield
    finally:
        frozen.run_directory = original


def accepted_root_reason(
    job: frozen.Job,
    hashes: dict[str, str],
    root: Path,
) -> tuple[bool, str]:
    with _source_view({job.name: root}):
        return frozen.accepted_run_reason(job, hashes)


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
    raise V4111RecoveryError(
        f"all recovery slots exhausted for {job.name}; every attempt is preserved"
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
            command = command_for_target(
                job,
                hashes,
                device=device,
                target_root=target_root,
            )
            log_path = RECOVERY_LOG_ROOT / f"{target_root.name}.log"
            start_record: dict[str, Any] = {
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
        except Exception as exc:
            result = {
                "job": job.name,
                "status": "launcher_exception",
                "returncode": None,
                "accepted": False,
                "acceptance_reason": f"{type(exc).__name__}: {exc}",
                "preserved_incomplete_sources": sources,
            }
        attempts.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return attempts


def _selected_sources(
    jobs: list[frozen.Job],
    hashes: dict[str, str],
) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, Path] = {}
    source_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
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
        if kind is None or root is None:
            unresolved.append({"job": job.name, "sources": evidence})
        else:
            selected[job.name] = root
    return selected, source_rows, unresolved


def _selected_manifest(
    jobs: list[frozen.Job],
    selected: dict[str, Path],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for job in jobs:
        root = selected.get(job.name)
        if root is None:
            continue
        relatives = list(frozen.FULL_RUN_REQUIRED_FILES)
        if (root / "collision_label_diagnostics.json").is_file():
            relatives.append("collision_label_diagnostics.json")
        manifest[job.name] = {
            "root": str(root.resolve()),
            "source_run_recovered": root != frozen.run_directory(job),
            "files": {
                relative: frozen._sha256(root / relative)
                for relative in sorted(set(relatives))
                if (root / relative).is_file()
            },
        }
    return manifest


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
            "schema_version": "topo-scene-v4.11.1.recovery-plan/v1",
            "recovery_version": RECOVERY_VERSION,
            "execution_policy": "attempt_all_then_summarize",
            "failure_does_not_cancel_remaining_jobs": True,
            "attempts": attempts,
            "formal_stage_launched": False,
            "formal_test_accessed": False,
        }
        _write_json(report_path.resolve(), report)
        return 0, report

    jobs = frozen.jobs_for_stage(contract, digest, "promotion")
    selected, source_rows, unresolved = _selected_sources(jobs, hashes)
    with _source_view(selected):
        base_summary = frozen.summarize_stage(
            contract, digest, "promotion", hashes
        )
    gate: dict[str, Any] | None = None
    gate_error: str | None = None
    if base_summary["complete"]:
        try:
            with _source_view(selected):
                gate = frozen.compute_promotion_gate(contract, digest, hashes)
        except Exception as exc:
            gate_error = f"{type(exc).__name__}: {exc}"

    manifest = _selected_manifest(jobs, selected)
    recovery_summary = {
        "schema_version": "topo-scene-v4.11.1.recovery-summary/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "incomplete_attempts_preserved": True,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        **hashes,
        "accepted_runs": base_summary["accepted_runs"],
        "expected_runs": base_summary["expected_runs"],
        "complete": base_summary["complete"],
        "recovered_run_count": sum(
            root != frozen.run_directory(job)
            for job in jobs
            for root in [selected.get(job.name)]
            if root is not None
        ),
        "source_selection": source_rows,
        "unresolved": unresolved,
        "selected_artifact_manifest": manifest,
        "selected_artifact_manifest_sha256": _digest_json(manifest),
        "frozen_v4_11_summary": str(
            (frozen.stage_root("promotion") / "summary.json").resolve()
        ),
        "frozen_v4_11_summary_sha256": frozen._sha256(
            frozen.stage_root("promotion") / "summary.json"
        ),
        "aggregate": base_summary["aggregate"],
        "gate_error": gate_error,
    }
    _write_json(DEFAULT_SUMMARY, recovery_summary)
    execution = {
        "schema_version": "topo-scene-v4.11.1.execution/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        **hashes,
        "attempts": attempts,
    }
    _write_json(DEFAULT_EXECUTION, execution)

    status = (
        "promotion_gate_passed"
        if gate is not None and gate.get("decision") == "pass"
        else "promotion_gate_failed_after_all_jobs"
        if gate is not None
        else "promotion_incomplete_after_all_attempts"
    )
    report = {
        "schema_version": "topo-scene-v4.11.1.automatic-recovery/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recovery_version": RECOVERY_VERSION,
        "recovery_kind": RECOVERY_KIND,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "incomplete_attempts_preserved": True,
        **hashes,
        "attempts": attempts,
        "accepted_jobs": base_summary["accepted_runs"],
        "expected_jobs": base_summary["expected_runs"],
        "promotion_matrix_complete": base_summary["complete"],
        "recovered_run_count": recovery_summary["recovered_run_count"],
        "recovery_summary": str(DEFAULT_SUMMARY.resolve()),
        "recovery_summary_sha256": frozen._sha256(DEFAULT_SUMMARY),
        "promotion_gate_decision": gate.get("decision") if gate else None,
        "promotion_gate_sha256": (
            frozen._sha256(frozen.DEFAULT_PROMOTION_GATE) if gate else None
        ),
        "gate_error": gate_error,
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": bool(
            gate is not None and gate.get("decision") == "pass"
        ),
    }
    _write_json(report_path.resolve(), report)
    return (
        0
        if status == "promotion_gate_passed"
        else 2
        if status == "promotion_gate_failed_after_all_jobs"
        else 1,
        report,
    )


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
    "DEFAULT_REPORT",
    "DEFAULT_SUMMARY",
    "RECOVERY_ROOT",
    "RECOVERY_RUN_ROOT",
    "RECOVERY_VERSION",
    "V4111RecoveryError",
    "accepted_root_reason",
    "candidate_directories",
    "command_for_target",
    "execute_all",
    "recovery_directory",
    "recovery_name",
    "run_all",
    "selected_source",
]
