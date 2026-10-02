"""Reconcile the v2 promotion false negatives and compute the frozen gate.

All twelve promotion jobs have already run.  The v2 automation inspected a
completed job by calling ``accepted_run_reason_v4_8_3`` outside the patch
context.  That loses the inherited v4.8.1 runtime-diagnostic adapter and makes
every candidate look rejected even though the authoritative v4.8.3 runner
accepts it.  This script is a read-only run-artifact reconciler: it verifies
the exact false-negative pattern, evaluates all jobs inside the complete
v4.8.3 context, snapshots required run artifacts before and after, asks the
frozen runner to summarize and gate promotion, and never launches formal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_remaining_v4_8_3_promotion as v1
from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_3_experiments as runner


SCHEMA_VERSION = "topo-scene-v4.8.3.promotion-reconciliation/v3"
AUTOMATION_ROOT = v1.PROMOTION_ROOT / "automation"
DEFAULT_V2_STATE = AUTOMATION_ROOT / "remaining_promotion_v2_state.json"
DEFAULT_V2_SUMMARY = AUTOMATION_ROOT / "remaining_promotion_v2_attempt_summary.json"
DEFAULT_OUTPUT = AUTOMATION_ROOT / "promotion_reconciliation_v3.json"
DEFAULT_STATE = AUTOMATION_ROOT / "promotion_reconciliation_v3_state.json"
DEFAULT_CHILD_LOG = AUTOMATION_ROOT / "promotion_reconciliation_v3_child.log"
DEFAULT_LOCK = AUTOMATION_ROOT / "promotion_reconciliation_v3.lock.json"


class ReconciliationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconciliationError(f"JSON root is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExclusiveLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "ExclusiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "token": self.token,
            "created_at_utc": _utc_now(),
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise ReconciliationError(
                f"reconciliation lock exists; verify its owner: {self.path}"
            ) from exc
        try:
            os.write(
                descriptor,
                (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        finally:
            os.close(descriptor)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            payload = _read_json(self.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if payload.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def frozen_command(python_executable: str, command: str) -> list[str]:
    prefix = [python_executable, str(v1.RUNNER_PATH)]
    if command == "status":
        return prefix + ["status"]
    if command == "summarize_promotion":
        return prefix + ["summarize", "--stage", "promotion"]
    if command == "gate_promotion":
        return prefix + ["gate", "--stage", "promotion"]
    raise ReconciliationError(f"unsupported frozen command: {command}")


def _append_log(path: Path, command: Sequence[str], output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{_utc_now()}] {json.dumps(list(command), ensure_ascii=False)}\n")
        handle.write(output)
        if output and not output.endswith("\n"):
            handle.write("\n")


def capture_json(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    _append_log(log_path, command, completed.stdout)
    if completed.returncode != 0:
        raise ReconciliationError(
            f"frozen runner returned {completed.returncode}: {list(command)}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReconciliationError(
            f"frozen runner did not emit one JSON object: {list(command)}"
        ) from exc
    if not isinstance(value, dict):
        raise ReconciliationError("frozen runner JSON root is not an object")
    return value


def assert_formal_zero(status: dict[str, Any]) -> None:
    stages = status.get("stages")
    formal = stages.get("formal") if isinstance(stages, dict) else None
    if not isinstance(formal, dict):
        raise ReconciliationError("status lacks formal stage")
    if formal.get("accepted") != 0 or formal.get("complete") is not False:
        raise ReconciliationError(
            "formal stage was accessed: "
            + json.dumps(formal, ensure_ascii=False, sort_keys=True)
        )


def assert_promotion_complete(status: dict[str, Any]) -> None:
    stages = status.get("stages")
    promotion = stages.get("promotion") if isinstance(stages, dict) else None
    if not isinstance(promotion, dict):
        raise ReconciliationError("status lacks promotion stage")
    if promotion.get("accepted") != 12 or promotion.get("complete") is not True:
        raise ReconciliationError(
            "authoritative promotion status is incomplete: "
            + json.dumps(promotion, ensure_ascii=False, sort_keys=True)
        )


def context_bound_states(
    jobs: Sequence[v48.Job], hashes: dict[str, str]
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    with runner._patched_v4_8_3_interfaces():
        for job in jobs:
            accepted, reason = v48.accepted_run_reason(job, hashes)
            states.append(
                {
                    "job": job.name,
                    "algorithm": job.algorithm,
                    "accepted": accepted,
                    "reason": reason,
                    "run_directory": str(v48.run_directory(job).resolve()),
                }
            )
    return states


def outside_context_states(
    jobs: Sequence[v48.Job], hashes: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            **asdict(v1.job_state(job, hashes)),
            "algorithm": job.algorithm,
        }
        for job in jobs
    ]


def validate_v2_false_negative(
    state: dict[str, Any], summary: dict[str, Any], jobs: Sequence[v48.Job]
) -> None:
    if state.get("status") != "all_jobs_attempted_with_failures":
        raise ReconciliationError("v2 state is not the completed false-negative state")
    if state.get("accepted_count") != 6 or state.get("expected_count") != 12:
        raise ReconciliationError("v2 6/12 trigger count drifted")
    if state.get("scientific_gate_computed") is not False:
        raise ReconciliationError("v2 unexpectedly computed the scientific gate")
    if state.get("formal_stage_launched") is not False:
        raise ReconciliationError("v2 reports formal access")
    attempts = summary.get("attempts_this_orchestration")
    if not isinstance(attempts, list):
        raise ReconciliationError("v2 attempts are missing")
    candidate_names = {job.name for job in jobs if job.algorithm == runner.V48_CANDIDATE}
    observed_false_negatives = {
        row.get("job")
        for row in attempts
        if isinstance(row, dict)
        and row.get("status") == "failed"
        and row.get("returncode") == 0
        and row.get("acceptance_reason") == "v4.8 changed return estimator"
    }
    if observed_false_negatives != candidate_names:
        raise ReconciliationError(
            "v2 false-negative job set drifted: "
            + json.dumps(sorted(observed_false_negatives), ensure_ascii=False)
        )
    if summary.get("all_jobs_attempted_or_preaccepted") is not True:
        raise ReconciliationError("v2 did not finish attempting all jobs")
    if summary.get("formal_accepted") != 0:
        raise ReconciliationError("v2 summary reports formal access")


def required_run_hashes(jobs: Sequence[v48.Job]) -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for job in jobs:
        root = v48.run_directory(job)
        files: dict[str, str] = {}
        for relative in v48.FULL_RUN_REQUIRED_FILES:
            path = root / relative
            if not path.is_file():
                raise ReconciliationError(f"missing required run artifact: {path}")
            files[relative] = _sha256(path)
        snapshot[job.name] = files
    return snapshot


def validate_summary(summary: dict[str, Any], jobs: Sequence[v48.Job]) -> None:
    rows = summary.get("per_run")
    if not isinstance(rows, list) or len(rows) != len(jobs):
        raise ReconciliationError("promotion summary does not contain 12 run rows")
    names = {row.get("run_name") for row in rows if isinstance(row, dict)}
    expected = {job.name for job in jobs}
    if names != expected:
        raise ReconciliationError("promotion summary run identities drifted")
    if summary.get("complete") is not True:
        raise ReconciliationError("promotion summary is not marked complete")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--v2-state", type=Path, default=DEFAULT_V2_STATE)
    value.add_argument("--v2-summary", type=Path, default=DEFAULT_V2_SUMMARY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--state", type=Path, default=DEFAULT_STATE)
    value.add_argument("--child-log", type=Path, default=DEFAULT_CHILD_LOG)
    value.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_path = args.output.resolve()
    state_path = args.state.resolve()
    child_log = args.child_log.resolve()
    lock_path = args.lock.resolve()
    with ExclusiveLock(lock_path):
        try:
            jobs, hashes = v1.promotion_protocol()
            v2_state_path = args.v2_state.resolve()
            v2_summary_path = args.v2_summary.resolve()
            v2_state = _read_json(v2_state_path)
            v2_summary = _read_json(v2_summary_path)
            validate_v2_false_negative(v2_state, v2_summary, jobs)

            before = required_run_hashes(jobs)
            outside = outside_context_states(jobs, hashes)
            inside = context_bound_states(jobs, hashes)
            if not all(row["accepted"] is True for row in inside):
                raise ReconciliationError(
                    "context-bound v4.8.3 acceptance is incomplete: "
                    + json.dumps(inside, ensure_ascii=False)
                )
            false_negative_names = {
                row["name"] for row in outside if not row["accepted"]
            }
            candidate_names = {
                job.name for job in jobs if job.algorithm == runner.V48_CANDIDATE
            }
            if false_negative_names != candidate_names:
                raise ReconciliationError("outside-context false-negative set drifted")

            status_before = capture_json(
                frozen_command(sys.executable, "status"), child_log
            )
            assert_promotion_complete(status_before)
            assert_formal_zero(status_before)
            summary = capture_json(
                frozen_command(sys.executable, "summarize_promotion"), child_log
            )
            validate_summary(summary, jobs)
            after_summary = required_run_hashes(jobs)
            if after_summary != before:
                raise ReconciliationError("summarization modified frozen run artifacts")

            gate = capture_json(
                frozen_command(sys.executable, "gate_promotion"), child_log
            )
            decision = gate.get("decision")
            if decision not in {"pass", "fail"}:
                raise ReconciliationError(
                    f"unexpected promotion gate decision: {decision!r}"
                )
            status_after = capture_json(
                frozen_command(sys.executable, "status"), child_log
            )
            assert_promotion_complete(status_after)
            assert_formal_zero(status_after)
            after_gate = required_run_hashes(jobs)
            if after_gate != before:
                raise ReconciliationError("gate computation modified frozen run artifacts")

            payload = {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": _utc_now(),
                "status": "reconciled",
                "root_cause": (
                    "v2 called accepted_run_reason_v4_8_3 outside the complete "
                    "v4.8.3 patch context, dropping the inherited v4.8.1 "
                    "runtime-diagnostic acceptance adapter"
                ),
                "scientific_protocol_changed": False,
                "scientific_jobs_rerun": False,
                "run_artifacts_modified": False,
                "v2_state_sha256": _sha256(v2_state_path),
                "v2_attempt_summary_sha256": _sha256(v2_summary_path),
                "v2_outside_context_states": outside,
                "v3_context_bound_states": inside,
                "authoritative_status_before_gate": status_before,
                "promotion_summary": summary,
                "promotion_gate": gate,
                "promotion_gate_decision": decision,
                "authoritative_status_after_gate": status_after,
                "formal_accepted": 0,
                "formal_stage_launched": False,
                "run_required_artifact_hashes": before,
            }
            _write_json_atomic(output_path, payload)
            _write_json_atomic(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": (
                        "promotion_gate_passed"
                        if decision == "pass"
                        else "promotion_gate_failed"
                    ),
                    "updated_at_utc": _utc_now(),
                    "promotion_accepted": 12,
                    "promotion_expected": 12,
                    "promotion_gate_decision": decision,
                    "reconciliation": str(output_path),
                    "formal_accepted": 0,
                    "formal_stage_launched": False,
                },
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if decision == "pass" else 2
        except Exception as exc:
            _write_json_atomic(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "reconciliation_failed",
                    "updated_at_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "formal_stage_launched": False,
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
