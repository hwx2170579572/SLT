"""Short-path recovery adapter for Windows-interrupted v4.9.2 jobs.

v4.9.2.2 correctly preserved an interrupted run, but its descriptive recovery
path exceeded the legacy Windows path budget used by TensorBoard.  This adapter
reuses that tested recovery engine while placing only fresh run/log directories
under the short repo-local ``r_v4923`` root.  Logical identity and full command
provenance remain in the separately versioned report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_all_v4_9_2_2_recovery as engine


VERSION = "v4.9.2.3"
KIND = "windows_short_path_immutable_reexecution"
SHORT_ROOT = ROOT / "r_v4923"
SHORT_RUN_ROOT = SHORT_ROOT / "runs"
SHORT_LOG_ROOT = SHORT_ROOT / "logs"
RESULT_ROOT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_recovery_v4_9_2_3"
)
DEFAULT_REPORT = RESULT_ROOT / "all_promotion_attempt_summary.json"
LEGACY_FAILED_RECOVERY_ROOT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_recovery_v4_9_2_2"
    / "runs"
)


def short_recovery_name(job: engine.frozen.Job, attempt: int) -> str:
    engine._require(
        1 <= attempt <= engine.MAX_RECOVERY_ATTEMPTS_PER_JOB,
        "invalid attempt",
    )
    identity = hashlib.sha256(job.name.encode("utf-8")).hexdigest()[:12]
    return f"r{attempt}_{identity}"


def short_recovery_directory(job: engine.frozen.Job, attempt: int) -> Path:
    return SHORT_RUN_ROOT / short_recovery_name(job, attempt)


def configure_engine() -> None:
    engine.RECOVERY_VERSION = VERSION
    engine.RECOVERY_KIND = KIND
    engine.RECOVERY_ROOT = RESULT_ROOT
    engine.RECOVERY_RUN_ROOT = SHORT_RUN_ROOT
    engine.RECOVERY_LOG_ROOT = SHORT_LOG_ROOT
    engine.DEFAULT_EXECUTION = RESULT_ROOT / "last_execution.json"
    engine.DEFAULT_SUMMARY = RESULT_ROOT / "summary.json"
    engine.DEFAULT_GATE = RESULT_ROOT / "promotion_gate.json"
    engine.DEFAULT_REPORT = RESULT_ROOT / "engine_report.json"
    engine.recovery_name = short_recovery_name
    engine.recovery_directory = short_recovery_directory


def run_all(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any]]:
    configure_engine()
    started = datetime.now(timezone.utc).isoformat()
    code, engine_report = engine.run_all(
        device=device,
        report_path=RESULT_ROOT / "engine_report.json",
        dry_run=dry_run,
    )
    legacy_attempts = sorted(
        str(path.resolve())
        for path in LEGACY_FAILED_RECOVERY_ROOT.glob("*")
        if path.is_dir()
    ) if LEGACY_FAILED_RECOVERY_ROOT.is_dir() else []
    report = {
        "schema_version": (
            "topo-scene-v4.9.2.3.short-path-recovery-automation/v1"
        ),
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
        "short_run_root": str(SHORT_RUN_ROOT.resolve()),
        "short_log_root": str(SHORT_LOG_ROOT.resolve()),
        "legacy_failed_recovery_attempts_preserved": legacy_attempts,
        "engine_report": str((RESULT_ROOT / "engine_report.json").resolve()),
        "accepted_jobs": engine_report.get("accepted_jobs"),
        "expected_jobs": engine_report.get("expected_jobs"),
        "promotion_matrix_complete": engine_report.get(
            "promotion_matrix_complete"
        ),
        "promotion_gate_decision": engine_report.get(
            "promotion_gate_decision"
        ),
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": engine_report.get("formal_test_unlocked", False),
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

