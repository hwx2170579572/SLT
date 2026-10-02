"""Attempt every v4.9 promotion job, then summarize and gate once.

Job or artifact failure never cancels later promotion jobs.  Formal testing is
never launched by this automation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_topo_v4_9_experiments as runner


AUTOMATION_ROOT = runner.stage_root("promotion") / "automation"
DEFAULT_REPORT = AUTOMATION_ROOT / "all_promotion_attempt_summary.json"


def run_all(*, device: str, report_path: Path = DEFAULT_REPORT) -> tuple[int, dict[str, Any]]:
    contract_path = runner.DEFAULT_CONTRACT.resolve()
    contract = runner.validate_contract(runner.load_contract(contract_path))
    digest = runner._sha256(contract_path)
    hashes = runner.protocol_hashes(contract_path=contract_path)
    jobs = runner.jobs_for_stage(contract, digest, "promotion")

    started = datetime.now(timezone.utc).isoformat()
    execution_returncode: int | None = None
    orchestration_error: str | None = None
    try:
        execution_returncode = runner.execute_jobs(
            contract,
            digest,
            "promotion",
            hashes,
            selector="all",
            device=device,
            workers=1,
            dry_run=False,
        )
    except Exception as exc:  # Persist the full-batch orchestration failure.
        orchestration_error = f"{type(exc).__name__}: {exc}"

    accepted_rows = []
    for job in jobs:
        accepted, reason = runner.accepted_run_reason(job, hashes)
        accepted_rows.append(
            {
                "job": job.name,
                "algorithm": job.algorithm,
                "scenario": job.scenario,
                "seed": job.seed,
                "accepted": accepted,
                "acceptance_reason": reason,
                "run_directory": str(runner.run_directory(job).resolve()),
            }
        )
    accepted_count = sum(row["accepted"] for row in accepted_rows)
    complete = accepted_count == len(jobs)
    summary = runner.summarize_stage(contract, digest, "promotion", hashes)
    gate = runner.compute_promotion_gate(contract, digest, hashes) if complete else None

    if orchestration_error is not None:
        status = "orchestration_failed_after_recording_available_results"
        exit_code = 1
    elif not complete:
        status = "promotion_jobs_incomplete_after_all_attempts"
        exit_code = 1
    elif gate and gate.get("decision") == "pass":
        status = "promotion_gate_passed"
        exit_code = 0
    else:
        status = "promotion_gate_failed"
        exit_code = 2
    report = {
        "schema_version": "topo-scene-v4.9.all-promotion-automation/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        **hashes,
        "execution_returncode": execution_returncode,
        "orchestration_error": orchestration_error,
        "jobs": accepted_rows,
        "accepted_jobs": accepted_count,
        "expected_jobs": len(jobs),
        "promotion_matrix_complete": complete,
        "summary_sha256": runner._sha256(
            runner.stage_root("promotion") / "summary.json"
        ),
        "promotion_gate_decision": gate.get("decision") if gate else None,
        "promotion_gate_sha256": (
            runner._sha256(runner.DEFAULT_PROMOTION_GATE) if gate else None
        ),
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": bool(gate and gate.get("decision") == "pass"),
        "summary": {
            "accepted_runs": summary["accepted_runs"],
            "expected_runs": summary["expected_runs"],
            "complete": summary["complete"],
            "aggregate": summary["aggregate"],
        },
    }
    runner._write_json(report_path.resolve(), report)
    return exit_code, report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--device", default="cuda")
    root.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    exit_code, report = run_all(device=args.device, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
