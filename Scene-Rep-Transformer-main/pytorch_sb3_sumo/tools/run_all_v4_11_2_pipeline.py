"""Short-path immutable recovery for the frozen v4.11 promotion.

The v4.11.1 recovery policy correctly preserved interrupted evidence and kept
running after a child failure, but its descriptive Windows path can exceed the
legacy path budget used by TensorBoard.  This adapter reuses that recovery
engine and changes only fresh runtime directory identities.  Scientific model,
training protocol, job matrix, hashes, gates, and source evidence remain frozen.

Every unresolved logical job is attempted before aggregation.  Formal testing
is never launched by this engineering recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_all_v4_11_1_pipeline as engine


VERSION = "v4.11.2"
KIND = "windows_short_path_immutable_reexecution"
SHORT_ROOT = ROOT / "r4112"
SHORT_RUN_ROOT = SHORT_ROOT / "r"
SHORT_LOG_ROOT = SHORT_ROOT / "l"
RESULT_ROOT = (
    ROOT
    / "results_topo_v4_11_promotion"
    / "engineering_recovery_v4_11_2"
)
DEFAULT_REPORT = (
    ROOT / "results_promotion_automation" / "v4_11_2_pipeline_summary.json"
)
LEGACY_FAILED_RECOVERY_ROOT = (
    ROOT
    / "results_topo_v4_11_promotion"
    / "engineering_recovery_v4_11_1"
    / "runs"
)
_ENGINE_CONFIGURATION_FIELDS = (
    "RECOVERY_VERSION",
    "RECOVERY_KIND",
    "RECOVERY_ROOT",
    "RECOVERY_RUN_ROOT",
    "RECOVERY_LOG_ROOT",
    "DEFAULT_EXECUTION",
    "DEFAULT_SUMMARY",
    "DEFAULT_REPORT",
    "recovery_name",
    "recovery_directory",
)


def short_recovery_name(job: engine.frozen.Job, attempt: int) -> str:
    """Return a deterministic compact runtime name for a logical job."""

    engine._require(
        1 <= attempt <= engine.MAX_RECOVERY_ATTEMPTS_PER_JOB,
        "invalid attempt",
    )
    identity = hashlib.sha256(job.name.encode("utf-8")).hexdigest()[:10]
    return f"r{attempt}_{identity}"


def short_recovery_directory(job: engine.frozen.Job, attempt: int) -> Path:
    return SHORT_RUN_ROOT / short_recovery_name(job, attempt)


def configure_engine() -> None:
    """Point only fresh engineering outputs at the compact runtime roots."""

    engine.RECOVERY_VERSION = VERSION
    engine.RECOVERY_KIND = KIND
    engine.RECOVERY_ROOT = RESULT_ROOT
    engine.RECOVERY_RUN_ROOT = SHORT_RUN_ROOT
    engine.RECOVERY_LOG_ROOT = SHORT_LOG_ROOT
    engine.DEFAULT_EXECUTION = RESULT_ROOT / "last_execution.json"
    engine.DEFAULT_SUMMARY = RESULT_ROOT / "summary.json"
    engine.DEFAULT_REPORT = RESULT_ROOT / "engine_report.json"
    engine.recovery_name = short_recovery_name
    engine.recovery_directory = short_recovery_directory


@contextmanager
def configured_engine() -> Iterator[None]:
    """Apply compact paths for one call, then restore the shared engine."""

    original = {
        name: getattr(engine, name) for name in _ENGINE_CONFIGURATION_FIELDS
    }
    configure_engine()
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(engine, name, value)


def run_all(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    engine_report_path = RESULT_ROOT / "engine_report.json"
    with configured_engine():
        code, engine_report = engine.run_all(
            device=device,
            report_path=engine_report_path,
            dry_run=dry_run,
        )
    attempts = engine_report.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    expected_jobs = engine_report.get("expected_jobs")
    if not isinstance(expected_jobs, int):
        expected_jobs = len(attempts)
    accepted_jobs = engine_report.get("accepted_jobs")
    if not isinstance(accepted_jobs, int):
        accepted_jobs = sum(
            row.get("status") == "already_accepted"
            for row in attempts
            if isinstance(row, dict)
        )
    legacy_attempts = (
        sorted(
            str(path.resolve())
            for path in LEGACY_FAILED_RECOVERY_ROOT.iterdir()
            if path.is_dir()
        )
        if LEGACY_FAILED_RECOVERY_ROOT.is_dir()
        else []
    )
    report = {
        "schema_version": "topo-scene-v4.11.2.short-path-recovery/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": engine_report.get("status"),
        "dry_run": dry_run,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "recovery_version": VERSION,
        "recovery_kind": KIND,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "experiment_contract_changed": False,
        "source_run_trees_modified": False,
        "incomplete_attempts_preserved": True,
        "short_run_root": str(SHORT_RUN_ROOT.resolve()),
        "short_log_root": str(SHORT_LOG_ROOT.resolve()),
        "legacy_failed_recovery_attempts_preserved": legacy_attempts,
        "engine_report": str(engine_report_path.resolve()),
        "accepted_jobs": accepted_jobs,
        "expected_jobs": expected_jobs,
        "planned_jobs": sum(
            row.get("status") == "planned"
            for row in attempts
            if isinstance(row, dict)
        ),
        "planned_recovery_jobs": sum(
            row.get("status") == "planned"
            and row.get("recovery_attempt") is not None
            for row in attempts
            if isinstance(row, dict)
        ),
        "planned_canonical_jobs": sum(
            row.get("status") == "planned"
            and row.get("recovery_attempt") is None
            for row in attempts
            if isinstance(row, dict)
        ),
        "promotion_matrix_complete": engine_report.get(
            "promotion_matrix_complete"
        ),
        "promotion_gate_decision": engine_report.get(
            "promotion_gate_decision"
        ),
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": engine_report.get(
            "formal_test_unlocked", False
        ),
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return code, report


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
    "KIND",
    "LEGACY_FAILED_RECOVERY_ROOT",
    "RESULT_ROOT",
    "SHORT_LOG_ROOT",
    "SHORT_ROOT",
    "SHORT_RUN_ROOT",
    "VERSION",
    "configure_engine",
    "configured_engine",
    "run_all",
    "short_recovery_directory",
    "short_recovery_name",
]
