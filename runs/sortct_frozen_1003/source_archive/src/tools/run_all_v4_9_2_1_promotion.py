"""Canonical attempt-all automation for v4.9.2 plus its v4.9.2.1 patch.

The frozen v4.9.2 automation remains untouched and is still responsible for
training every missing job.  Irrespective of its metadata-only return code,
this wrapper then applies the read-only v4.9.2.1 provenance revalidation and
aggregates the complete matrix once.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_all_v4_9_2_promotion as frozen_automation
from tools import run_topo_v4_9_2_1_experiments as patched


DEFAULT_REPORT = patched.PATCH_ROOT / "all_promotion_attempt_summary.json"


def run_all(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    frozen_code: int | None = None
    frozen_report: dict[str, Any] | None = None
    frozen_error: str | None = None
    try:
        frozen_code, frozen_report = frozen_automation.run_all(device=device)
    except Exception as exc:  # the patch still audits every artifact available
        frozen_error = f"{type(exc).__name__}: {exc}"

    patched_code: int | None = None
    patched_report: dict[str, Any] | None = None
    patched_error: str | None = None
    try:
        patched_code, patched_report = patched.revalidate_promotion()
    except Exception as exc:
        patched_error = f"{type(exc).__name__}: {exc}"

    if patched_report is None:
        status = "revalidation_exception_after_all_attempts"
        exit_code = 1
    elif not patched_report.get("promotion_matrix_complete"):
        status = "promotion_jobs_incomplete_after_all_attempts"
        exit_code = 1
    elif patched_report.get("promotion_gate_decision") == "pass":
        status = "promotion_gate_passed"
        exit_code = 0
    else:
        status = "promotion_gate_failed"
        exit_code = 2

    report = {
        "schema_version": "topo-scene-v4.9.2.1.all-promotion-automation/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "frozen_training_automation_returncode": frozen_code,
        "frozen_training_automation_status": (
            frozen_report.get("status") if frozen_report else None
        ),
        "frozen_training_automation_error": frozen_error,
        "metadata_revalidation_returncode": patched_code,
        "metadata_revalidation_error": patched_error,
        "patch_version": patched.PATCH_VERSION,
        "patch_kind": patched.PATCH_KIND,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_tree_modified": False,
        "accepted_jobs": (
            patched_report.get("accepted_jobs") if patched_report else None
        ),
        "expected_jobs": (
            patched_report.get("expected_jobs") if patched_report else None
        ),
        "promotion_matrix_complete": bool(
            patched_report and patched_report.get("promotion_matrix_complete")
        ),
        "promotion_gate_decision": (
            patched_report.get("promotion_gate_decision")
            if patched_report
            else None
        ),
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": bool(
            patched_report
            and patched_report.get("promotion_gate_decision") == "pass"
        ),
        "revalidation_report": str(patched.DEFAULT_REPORT.resolve()),
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return exit_code, report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, report = run_all(device=args.device, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

