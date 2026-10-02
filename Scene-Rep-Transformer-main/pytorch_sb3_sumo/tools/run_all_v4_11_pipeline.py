"""Run every eligible v4.11 stage through promotion without per-job fail-fast.

Every job in an entered stage is attempted before that stage is summarized.
Scientific stage gates still apply: a failed development gate does not launch
ablation, and formal testing is never launched by this automation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_11_experiments as runner


DEFAULT_REPORT = (
    ROOT / "results_promotion_automation" / "v4_11_pipeline_summary.json"
)
Gate = Callable[[dict[str, Any], str, dict[str, str]], dict[str, Any]]


def _attempt_stage(
    *,
    contract: dict[str, Any],
    digest: str,
    hashes: dict[str, str],
    stage: str,
    device: str,
    gate: Gate,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    execution_returncode: int | None = None
    execution_error: str | None = None
    try:
        execution_returncode = runner.execute_jobs(
            contract,
            digest,
            stage,
            hashes,
            selector="all",
            device=device,
            workers=1,
            dry_run=False,
        )
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"

    summary_error: str | None = None
    try:
        summary = runner.summarize_stage(contract, digest, stage, hashes)
    except Exception as exc:
        summary = {
            "accepted_runs": 0,
            "expected_runs": len(
                runner.jobs_for_stage(contract, digest, stage)
            ),
            "complete": False,
            "aggregate": [],
        }
        summary_error = f"{type(exc).__name__}: {exc}"

    gate_value: dict[str, Any] | None = None
    gate_error: str | None = None
    try:
        gate_value = gate(contract, digest, hashes)
    except Exception as exc:
        gate_error = f"{type(exc).__name__}: {exc}"

    return {
        "stage": stage,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "execution_returncode": execution_returncode,
        "execution_error": execution_error,
        "summary_error": summary_error,
        "gate_error": gate_error,
        "accepted_jobs": summary.get("accepted_runs"),
        "expected_jobs": summary.get("expected_runs"),
        "matrix_complete": summary.get("complete"),
        "decision": gate_value.get("decision") if gate_value else None,
        "aggregate": summary.get("aggregate", []),
    }


def run_all(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[int, dict[str, Any]]:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner._sha256(runner.DEFAULT_CONTRACT)
    hashes = runner.protocol_hashes()
    started = datetime.now(timezone.utc).isoformat()
    stages: list[dict[str, Any]] = []

    development = _attempt_stage(
        contract=contract,
        digest=digest,
        hashes=hashes,
        stage="development",
        device=device,
        gate=runner.compute_development_gate,
    )
    stages.append(development)
    if development["decision"] != "pass":
        status = (
            "development_gate_failed_after_all_jobs"
            if development["matrix_complete"]
            else "development_incomplete_after_all_attempts"
        )
        exit_code = 2 if development["matrix_complete"] else 1
    else:
        ablation = _attempt_stage(
            contract=contract,
            digest=digest,
            hashes=hashes,
            stage="ablation",
            device=device,
            gate=runner.compute_ablation_decision,
        )
        stages.append(ablation)
        if ablation["decision"] != "complete":
            status = "required_ablation_incomplete_after_all_attempts"
            exit_code = 1
        else:
            promotion = _attempt_stage(
                contract=contract,
                digest=digest,
                hashes=hashes,
                stage="promotion",
                device=device,
                gate=runner.compute_promotion_gate,
            )
            stages.append(promotion)
            if promotion["decision"] == "pass":
                status = "promotion_gate_passed"
                exit_code = 0
            elif promotion["matrix_complete"]:
                status = "promotion_gate_failed_after_all_jobs"
                exit_code = 2
            else:
                status = "promotion_incomplete_after_all_attempts"
                exit_code = 1

    accepted_jobs = sum(
        int(stage.get("accepted_jobs") or 0) for stage in stages
    )
    expected_jobs = sum(
        int(stage.get("expected_jobs") or 0) for stage in stages
    )
    promotion_stage = next(
        (stage for stage in stages if stage["stage"] == "promotion"), None
    )
    payload = {
        "schema_version": "topo-scene-v4.11.automatic-pipeline/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execution_policy": (
            "attempt_all_jobs_in_each_entered_stage_then_summarize"
        ),
        "failure_does_not_cancel_remaining_jobs_in_entered_stage": True,
        "stage_gates_preserved": True,
        **hashes,
        "accepted_jobs": accepted_jobs,
        "expected_jobs": expected_jobs,
        "stages": stages,
        "promotion_gate_decision": (
            promotion_stage.get("decision") if promotion_stage else None
        ),
        "formal_stage_launched": False,
        "formal_test_accessed": False,
        "formal_test_unlocked": status == "promotion_gate_passed",
        "next_action": {
            "promotion_gate_passed": (
                "launch_formal_only_with_explicit_formal_runner"
            ),
            "promotion_gate_failed_after_all_jobs": (
                "complete_attribution_and_create_new_version"
            ),
            "promotion_incomplete_after_all_attempts": (
                "repair_missing_or_invalid_promotion_artifacts"
            ),
            "development_gate_failed_after_all_jobs": (
                "complete_attribution_and_create_new_version"
            ),
            "development_incomplete_after_all_attempts": (
                "repair_missing_or_invalid_development_artifacts"
            ),
            "required_ablation_incomplete_after_all_attempts": (
                "repair_required_ablation_artifacts"
            ),
        }[status],
    }
    runner._write_json(report_path.resolve(), payload)
    return exit_code, payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, value = run_all(device=args.device, report_path=args.report)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
