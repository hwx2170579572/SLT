"""Run every remaining v4.8.3 promotion job before producing one summary.

This v2 orchestrator supersedes the v1 stop-on-first-job-failure policy.  It
waits for a currently running immutable job, then invokes the frozen v4.8.3
runner once per remaining job in contract order.  A completed subprocess or
acceptance failure is recorded and the next job is attempted.  After all 12
jobs have either been accepted or attempted, the orchestrator writes a joint
attempt summary.  It computes the scientific promotion gate only when all 12
jobs are accepted, and it never launches the formal stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_remaining_v4_8_3_promotion as base
from tools import run_topo_v4_8_experiments as v48


SCHEMA_VERSION = "topo-scene-v4.8.3.remaining-promotion-automation/v2"
AUTOMATION_ROOT = base.PROMOTION_ROOT / "automation"
DEFAULT_STATE = AUTOMATION_ROOT / "remaining_promotion_v2_state.json"
DEFAULT_EVENTS = AUTOMATION_ROOT / "remaining_promotion_v2_events.jsonl"
DEFAULT_CHILD_LOG = AUTOMATION_ROOT / "remaining_promotion_v2_child.log"
DEFAULT_LOCK = AUTOMATION_ROOT / "remaining_promotion_v2.lock.json"
DEFAULT_ATTEMPT_SUMMARY = (
    AUTOMATION_ROOT / "remaining_promotion_v2_attempt_summary.json"
)


class ExclusiveAutomationLockV2:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "ExclusiveAutomationLockV2":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "token": self.token,
            "created_at_utc": base._utc_now(),
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise base.PromotionAutomationError(
                f"v2 automation lock already exists; verify its owner before removing: {self.path}"
            ) from exc
        try:
            os.write(
                descriptor,
                (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )
        finally:
            os.close(descriptor)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            payload = base._read_json(self.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if payload.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def runner_command(
    python_executable: str,
    command: str,
    *,
    device: str,
    job_name: str | None = None,
) -> list[str]:
    prefix = [python_executable, str(base.RUNNER_PATH)]
    if command == "run_promotion_job":
        if not job_name or not job_name.startswith("P__"):
            raise base.PromotionAutomationError(
                "an exact frozen promotion job name is required"
            )
        return prefix + [
            "run",
            "--stage",
            "promotion",
            "--job",
            job_name,
            "--device",
            device,
            "--workers",
            "1",
        ]
    if command == "summarize_promotion":
        return prefix + ["summarize", "--stage", "promotion"]
    if command == "gate_promotion":
        return prefix + ["gate", "--stage", "promotion"]
    if command == "status":
        return prefix + ["status"]
    raise base.PromotionAutomationError(f"unsupported safe v2 command: {command}")


def assert_formal_untouched(status: dict[str, Any]) -> None:
    stages = status.get("stages")
    if not isinstance(stages, dict):
        raise base.PromotionAutomationError("status is missing stages")
    formal = stages.get("formal")
    if not isinstance(formal, dict):
        raise base.PromotionAutomationError("status is missing formal stage")
    if formal.get("accepted") != 0 or formal.get("complete") is not False:
        raise base.PromotionAutomationError(
            f"formal stage was unexpectedly accessed: {json.dumps(formal, ensure_ascii=False)}"
        )


def accepted_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "run_name",
        "method",
        "algorithm",
        "scenario",
        "seed",
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "mean_decision_steps",
        "mean_raw_steps",
        "selected_checkpoint_kind",
        "selected_deployment_decoder",
        "training_wall_seconds",
        "mean_inference_ms",
    )
    rows = summary.get("per_run", [])
    if not isinstance(rows, list):
        raise base.PromotionAutomationError("promotion summary per_run is not a list")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise base.PromotionAutomationError("promotion summary contains a non-object row")
        output.append({key: row.get(key) for key in keys})
    return output


def _failure_attempt(
    job: v48.Job,
    state: base.PromotionJobState,
    *,
    returncode: int | None,
    launcher_failure: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    return {
        "job": job.name,
        "status": "failed",
        "source": source,
        "returncode": returncode,
        "acceptance_reason": state.acceptance_reason,
        "run_directory": str(v48.run_directory(job).resolve()),
        "launcher_failure": launcher_failure,
    }


def wait_for_current_job(
    job: v48.Job,
    hashes: dict[str, str],
    *,
    poll_seconds: float,
    stall_timeout_seconds: float,
    recorder: base.EventRecorder,
    state_path: Path,
) -> dict[str, Any]:
    last_activity = base._tree_activity_ns(job)
    last_change = time.monotonic()
    recorder.emit(
        "waiting_for_current_job",
        job=job.name,
        policy="record_failure_and_continue",
        poll_seconds=poll_seconds,
        stall_timeout_seconds=stall_timeout_seconds,
    )
    while True:
        state = base.job_state(job, hashes)
        if state.accepted:
            result = {
                "job": job.name,
                "status": "accepted",
                "source": "preexisting_live_job",
                "returncode": 0,
                "acceptance_reason": "accepted",
                "run_directory": str(v48.run_directory(job).resolve()),
                "launcher_failure": None,
            }
            recorder.emit("current_job_accepted", job=job.name)
            return result
        launcher_failure = base._recorded_failure(job.name)
        if launcher_failure is not None:
            result = _failure_attempt(
                job,
                state,
                returncode=launcher_failure.get("returncode"),
                launcher_failure=launcher_failure,
                source="preexisting_live_job",
            )
            recorder.emit(
                "current_job_failed_continuing",
                job=job.name,
                acceptance_reason=state.acceptance_reason,
                launcher_failure=launcher_failure,
            )
            return result
        activity = base._tree_activity_ns(job)
        if activity > last_activity:
            last_activity = activity
            last_change = time.monotonic()
        stalled_for = time.monotonic() - last_change
        base._write_json_atomic(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "waiting_for_current_job",
                "updated_at_utc": base._utc_now(),
                "current_job": {
                    **base.asdict(state),
                    "failure_policy": "record_and_continue_after_subprocess_completion",
                },
                "stalled_for_seconds": stalled_for,
                "formal_stage_launched": False,
            },
        )
        if stalled_for >= stall_timeout_seconds:
            raise base.PromotionAutomationError(
                f"current job has unknown live/stalled state for {stalled_for:.1f}s: {job.name}; "
                "cannot safely overlap another GPU/SUMO job"
            )
        time.sleep(poll_seconds)


def attempt_one_job(
    job: v48.Job,
    hashes: dict[str, str],
    *,
    device: str,
    recorder: base.EventRecorder,
    child_log: Path,
) -> dict[str, Any]:
    recorder.emit("job_started", job=job.name, failure_policy="continue")
    returncode = base.stream_command(
        runner_command(
            sys.executable,
            "run_promotion_job",
            device=device,
            job_name=job.name,
        ),
        child_log,
    )
    state = base.job_state(job, hashes)
    launcher_failure = base._recorded_failure(job.name)
    if state.accepted:
        result = {
            "job": job.name,
            "status": "accepted",
            "source": "v2_orchestrator",
            "returncode": returncode,
            "acceptance_reason": "accepted",
            "run_directory": str(v48.run_directory(job).resolve()),
            "launcher_failure": launcher_failure,
        }
        recorder.emit("job_accepted", job=job.name, returncode=returncode)
        return result
    result = _failure_attempt(
        job,
        state,
        returncode=returncode,
        launcher_failure=launcher_failure,
        source="v2_orchestrator",
    )
    recorder.emit(
        "job_failed_continuing",
        job=job.name,
        returncode=returncode,
        acceptance_reason=state.acceptance_reason,
        launcher_failure=launcher_failure,
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--poll-seconds", type=float, default=30.0)
    value.add_argument("--stall-timeout-seconds", type=float, default=3600.0)
    value.add_argument("--state", type=Path, default=DEFAULT_STATE)
    value.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    value.add_argument("--child-log", type=Path, default=DEFAULT_CHILD_LOG)
    value.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    value.add_argument(
        "--attempt-summary",
        type=Path,
        default=DEFAULT_ATTEMPT_SUMMARY,
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise base.PromotionAutomationError("--poll-seconds must be positive")
    if args.stall_timeout_seconds <= args.poll_seconds:
        raise base.PromotionAutomationError(
            "--stall-timeout-seconds must exceed --poll-seconds"
        )
    state_path = args.state.resolve()
    events_path = args.events.resolve()
    child_log = args.child_log.resolve()
    lock_path = args.lock.resolve()
    attempt_summary_path = args.attempt_summary.resolve()
    recorder = base.EventRecorder(events_path)
    with ExclusiveAutomationLockV2(lock_path):
        try:
            jobs, hashes = base.promotion_protocol()
            initial_states = base.job_states(jobs, hashes)
            current_name = base.takeover_job_name(initial_states)
            preexisting_accepted = [
                state.name for state in initial_states if state.accepted
            ]
            attempts: list[dict[str, Any]] = []
            handled_incomplete: set[str] = set()
            recorder.emit(
                "automation_v2_started",
                ordered_jobs=[job.name for job in jobs],
                preexisting_accepted=preexisting_accepted,
                current_job=current_name,
                failure_policy="attempt_all_then_summarize",
                formal_stage_launched=False,
            )
            if current_name is not None:
                current = next(job for job in jobs if job.name == current_name)
                attempts.append(
                    wait_for_current_job(
                        current,
                        hashes,
                        poll_seconds=args.poll_seconds,
                        stall_timeout_seconds=args.stall_timeout_seconds,
                        recorder=recorder,
                        state_path=state_path,
                    )
                )
                handled_incomplete.add(current_name)

            for job in jobs:
                state = base.job_state(job, hashes)
                if state.accepted:
                    continue
                if state.directory_exists:
                    if job.name in handled_incomplete:
                        continue
                    raise base.PromotionAutomationError(
                        f"unclassified immutable incomplete run blocks safe continuation: {job.name}"
                    )
                base._write_json_atomic(
                    state_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "running_all_remaining_jobs",
                        "updated_at_utc": base._utc_now(),
                        "current_job": job.name,
                        "completed_attempts": attempts,
                        "failure_policy": "attempt_all_then_summarize",
                        "formal_stage_launched": False,
                    },
                )
                attempts.append(
                    attempt_one_job(
                        job,
                        hashes,
                        device=args.device,
                        recorder=recorder,
                        child_log=child_log,
                    )
                )

            status = base.capture_json_command(
                runner_command(sys.executable, "status", device=args.device),
                child_log,
            )
            assert_formal_untouched(status)
            summary = base.capture_json_command(
                runner_command(
                    sys.executable,
                    "summarize_promotion",
                    device=args.device,
                ),
                child_log,
            )
            final_states = base.job_states(jobs, hashes)
            failed_states = [state for state in final_states if not state.accepted]
            attempt_summary: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": base._utc_now(),
                "failure_policy": "attempt_all_then_summarize",
                "ordered_jobs": [job.name for job in jobs],
                "preexisting_accepted_jobs": preexisting_accepted,
                "attempts_this_orchestration": attempts,
                "accepted_jobs": [state.name for state in final_states if state.accepted],
                "failed_or_unaccepted_jobs": [
                    base.asdict(state) for state in failed_states
                ],
                "accepted_count": len(final_states) - len(failed_states),
                "expected_count": len(final_states),
                "all_jobs_attempted_or_preaccepted": True,
                "scientific_gate_computed": False,
                "formal_accepted": status["stages"]["formal"]["accepted"],
                "formal_stage_launched": False,
                "accepted_metrics": accepted_metric_rows(summary),
                "aggregate_available_results": summary.get("aggregate", []),
            }

            if failed_states:
                base._write_json_atomic(attempt_summary_path, attempt_summary)
                final_state = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "all_jobs_attempted_with_failures",
                    "updated_at_utc": base._utc_now(),
                    "accepted_count": attempt_summary["accepted_count"],
                    "expected_count": attempt_summary["expected_count"],
                    "failed_jobs": [state.name for state in failed_states],
                    "attempt_summary": str(attempt_summary_path),
                    "scientific_gate_computed": False,
                    "formal_stage_launched": False,
                }
                base._write_json_atomic(state_path, final_state)
                recorder.emit(
                    "automation_v2_finished_with_failures",
                    accepted_count=attempt_summary["accepted_count"],
                    failed_jobs=final_state["failed_jobs"],
                    attempt_summary=str(attempt_summary_path),
                    formal_stage_launched=False,
                )
                return 3

            base.assert_promotion_complete_without_formal(status)
            gate = base.capture_json_command(
                runner_command(
                    sys.executable,
                    "gate_promotion",
                    device=args.device,
                ),
                child_log,
            )
            final_status = base.capture_json_command(
                runner_command(sys.executable, "status", device=args.device),
                child_log,
            )
            base.assert_promotion_complete_without_formal(final_status)
            decision = gate.get("decision")
            if decision not in {"pass", "fail"}:
                raise base.PromotionAutomationError(
                    f"unexpected promotion gate decision: {decision!r}"
                )
            attempt_summary.update(
                {
                    "scientific_gate_computed": True,
                    "promotion_gate_decision": decision,
                    "gate": gate,
                }
            )
            base._write_json_atomic(attempt_summary_path, attempt_summary)
            final_state = {
                "schema_version": SCHEMA_VERSION,
                "status": (
                    "promotion_gate_passed"
                    if decision == "pass"
                    else "promotion_gate_failed"
                ),
                "updated_at_utc": base._utc_now(),
                "accepted_count": 12,
                "expected_count": 12,
                "promotion_gate_decision": decision,
                "attempt_summary": str(attempt_summary_path),
                "formal_accepted": 0,
                "formal_stage_launched": False,
                "formal_test_unlocked": final_status.get("formal_test_unlocked"),
            }
            base._write_json_atomic(state_path, final_state)
            recorder.emit(
                "automation_v2_finished",
                decision=decision,
                accepted_count=12,
                attempt_summary=str(attempt_summary_path),
                formal_stage_launched=False,
            )
            return 0 if decision == "pass" else 2
        except Exception as exc:
            base._write_json_atomic(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "automation_infrastructure_failed",
                    "updated_at_utc": base._utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "formal_stage_launched": False,
                },
            )
            recorder.emit(
                "automation_v2_infrastructure_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                formal_stage_launched=False,
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
