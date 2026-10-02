"""Resume-safe serial automation for the remaining frozen v4.8.3 promotion.

The orchestrator may be started while the first incomplete promotion job is
already running.  It waits for that immutable run to become accepted, then
delegates every remaining job to the frozen v4.8.3 runner in contract order.
It summarizes the completed matrix and computes the promotion gate, but it
never launches or reads the formal experiment stage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_3_experiments as runner


PROMOTION_ROOT = PROJECT_ROOT / "results_topo_v4_8_promotion"
AUTOMATION_ROOT = PROMOTION_ROOT / "automation"
DEFAULT_STATE = AUTOMATION_ROOT / "remaining_promotion_state.json"
DEFAULT_EVENTS = AUTOMATION_ROOT / "remaining_promotion_events.jsonl"
DEFAULT_CHILD_LOG = AUTOMATION_ROOT / "remaining_promotion_child.log"
DEFAULT_LOCK = AUTOMATION_ROOT / "remaining_promotion.lock.json"
RUNNER_PATH = PROJECT_ROOT / "tools" / "run_topo_v4_8_3_experiments.py"
LAST_EXECUTION = PROMOTION_ROOT / "last_execution.json"
SCHEMA_VERSION = "topo-scene-v4.8.3.remaining-promotion-automation/v1"


class PromotionAutomationError(RuntimeError):
    """Raised when takeover would violate the immutable promotion protocol."""


@dataclass(frozen=True)
class PromotionJobState:
    name: str
    accepted: bool
    acceptance_reason: str
    directory_exists: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionAutomationError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class EventRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        payload = {"timestamp_utc": _utc_now(), "event": event, **values}
        line = json.dumps(payload, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        return payload


class ExclusiveAutomationLock:
    """Small exclusive lock that never guesses whether an old owner is stale."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "ExclusiveAutomationLock":
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
            raise PromotionAutomationError(
                f"automation lock already exists; verify its owner before removing: {self.path}"
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
            payload = _read_json(self.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if payload.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def promotion_protocol() -> tuple[list[v48.Job], dict[str, str]]:
    runner.validate_patch_contract(runner.load_patch_contract())
    runner.validate_implementation_freeze(runner.DEFAULT_FREEZE)
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    hashes = v48.protocol_hashes(freeze_path=runner.PARENT_V48_FREEZE)
    runner._assert_stage_can_run_v4_8_3(hashes, "promotion")
    jobs = v48.jobs_for_stage(contract, digest, "promotion")
    if len(jobs) != 12:
        raise PromotionAutomationError("frozen promotion matrix is not 12 jobs")
    return jobs, hashes


def job_state(job: v48.Job, hashes: dict[str, str]) -> PromotionJobState:
    directory = v48.run_directory(job)
    if not directory.exists():
        return PromotionJobState(
            name=job.name,
            accepted=False,
            acceptance_reason="run directory missing",
            directory_exists=False,
        )
    accepted, reason = runner.accepted_run_reason_v4_8_3(job, hashes)
    return PromotionJobState(
        name=job.name,
        accepted=accepted,
        acceptance_reason=reason,
        directory_exists=True,
    )


def job_states(
    jobs: Iterable[v48.Job], hashes: dict[str, str]
) -> list[PromotionJobState]:
    return [job_state(job, hashes) for job in jobs]


def takeover_job_name(states: Sequence[PromotionJobState]) -> str | None:
    """Return the sole eligible in-progress job, rejecting unsafe layouts."""

    first_unaccepted = next((state for state in states if not state.accepted), None)
    incomplete_directories = [
        state for state in states if state.directory_exists and not state.accepted
    ]
    if len(incomplete_directories) > 1:
        names = ", ".join(state.name for state in incomplete_directories)
        raise PromotionAutomationError(
            f"multiple immutable incomplete run directories exist: {names}"
        )
    if not incomplete_directories:
        return None
    current = incomplete_directories[0]
    if first_unaccepted is None or current.name != first_unaccepted.name:
        expected = first_unaccepted.name if first_unaccepted else "none"
        raise PromotionAutomationError(
            f"incomplete run is out of frozen order: {current.name}; expected {expected}"
        )
    return current.name


def _tree_activity_ns(job: v48.Job) -> int:
    candidates: list[Path] = []
    directory = v48.run_directory(job)
    if directory.exists():
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    launcher_log = PROMOTION_ROOT / "launcher_logs" / f"{job.name}.log"
    if launcher_log.is_file():
        candidates.append(launcher_log)
    observed: list[int] = []
    for path in candidates:
        try:
            observed.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return max(observed, default=0)


def _recorded_failure(job_name: str) -> dict[str, Any] | None:
    if not LAST_EXECUTION.is_file():
        return None
    try:
        execution = _read_json(LAST_EXECUTION)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    for failure in execution.get("failures", []):
        if isinstance(failure, dict) and failure.get("job") == job_name:
            return failure
    return None


def wait_for_takeover_job(
    job: v48.Job,
    hashes: dict[str, str],
    *,
    poll_seconds: float,
    stall_timeout_seconds: float,
    recorder: EventRecorder,
    state_path: Path,
) -> None:
    last_activity = _tree_activity_ns(job)
    last_change = time.monotonic()
    recorder.emit(
        "waiting_for_current_job",
        job=job.name,
        poll_seconds=poll_seconds,
        stall_timeout_seconds=stall_timeout_seconds,
    )
    while True:
        state = job_state(job, hashes)
        if state.accepted:
            recorder.emit("current_job_accepted", job=job.name)
            return
        failure = _recorded_failure(job.name)
        if failure is not None:
            raise PromotionAutomationError(
                f"current job recorded a launcher failure: {json.dumps(failure, ensure_ascii=False)}"
            )
        activity = _tree_activity_ns(job)
        if activity > last_activity:
            last_activity = activity
            last_change = time.monotonic()
        stalled_for = time.monotonic() - last_change
        _write_json_atomic(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "waiting_for_current_job",
                "updated_at_utc": _utc_now(),
                "current_job": asdict(state),
                "stalled_for_seconds": stalled_for,
                "formal_stage_launched": False,
            },
        )
        if stalled_for >= stall_timeout_seconds:
            raise PromotionAutomationError(
                f"current job activity stalled for {stalled_for:.1f}s: {job.name}; "
                f"last acceptance reason: {state.acceptance_reason}"
            )
        time.sleep(poll_seconds)


def runner_command(
    python_executable: str,
    command: str,
    *,
    device: str,
) -> list[str]:
    base = [python_executable, str(RUNNER_PATH)]
    if command == "run_all_promotion":
        return base + [
            "run",
            "--stage",
            "promotion",
            "--job",
            "all",
            "--device",
            device,
            "--workers",
            "1",
        ]
    if command == "summarize_promotion":
        return base + ["summarize", "--stage", "promotion"]
    if command == "gate_promotion":
        return base + ["gate", "--stage", "promotion"]
    if command == "status":
        return base + ["status"]
    raise PromotionAutomationError(f"unsupported safe runner command: {command}")


def _append_child_log(path: Path, heading: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{_utc_now()}] {heading}\n")
        handle.write(content)
        if content and not content.endswith("\n"):
            handle.write("\n")


def stream_command(command: Sequence[str], child_log: Path) -> int:
    _append_child_log(child_log, "COMMAND", json.dumps(list(command), ensure_ascii=False))
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with child_log.open("a", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
    return process.wait()


def capture_json_command(command: Sequence[str], child_log: Path) -> dict[str, Any]:
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
    _append_child_log(
        child_log,
        "COMMAND " + json.dumps(list(command), ensure_ascii=False),
        completed.stdout,
    )
    if completed.returncode != 0:
        raise PromotionAutomationError(
            f"runner command failed with {completed.returncode}: {list(command)}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PromotionAutomationError(
            f"runner command did not emit JSON: {list(command)}"
        ) from exc
    if not isinstance(value, dict):
        raise PromotionAutomationError(f"runner JSON is not an object: {list(command)}")
    return value


def assert_promotion_complete_without_formal(status: dict[str, Any]) -> None:
    stages = status.get("stages")
    if not isinstance(stages, dict):
        raise PromotionAutomationError("status is missing stages")
    promotion = stages.get("promotion")
    formal = stages.get("formal")
    if not isinstance(promotion, dict) or not isinstance(formal, dict):
        raise PromotionAutomationError("status is missing promotion/formal stages")
    if promotion.get("accepted") != 12 or promotion.get("complete") is not True:
        raise PromotionAutomationError(
            f"promotion did not complete: {json.dumps(promotion, ensure_ascii=False)}"
        )
    if formal.get("accepted") != 0 or formal.get("complete") is not False:
        raise PromotionAutomationError(
            f"formal stage was unexpectedly accessed: {json.dumps(formal, ensure_ascii=False)}"
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--poll-seconds", type=float, default=30.0)
    value.add_argument("--stall-timeout-seconds", type=float, default=3600.0)
    value.add_argument("--state", type=Path, default=DEFAULT_STATE)
    value.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    value.add_argument("--child-log", type=Path, default=DEFAULT_CHILD_LOG)
    value.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise PromotionAutomationError("--poll-seconds must be positive")
    if args.stall_timeout_seconds <= args.poll_seconds:
        raise PromotionAutomationError(
            "--stall-timeout-seconds must exceed --poll-seconds"
        )
    state_path = args.state.resolve()
    events_path = args.events.resolve()
    child_log = args.child_log.resolve()
    lock_path = args.lock.resolve()
    recorder = EventRecorder(events_path)
    with ExclusiveAutomationLock(lock_path):
        try:
            jobs, hashes = promotion_protocol()
            states = job_states(jobs, hashes)
            current_name = takeover_job_name(states)
            recorder.emit(
                "automation_started",
                ordered_jobs=[job.name for job in jobs],
                accepted_jobs=[state.name for state in states if state.accepted],
                current_job=current_name,
                python_executable=sys.executable,
                formal_stage_launched=False,
            )
            if current_name is not None:
                current = next(job for job in jobs if job.name == current_name)
                wait_for_takeover_job(
                    current,
                    hashes,
                    poll_seconds=args.poll_seconds,
                    stall_timeout_seconds=args.stall_timeout_seconds,
                    recorder=recorder,
                    state_path=state_path,
                )

            refreshed = job_states(jobs, hashes)
            unsafe = [
                state.name
                for state in refreshed
                if state.directory_exists and not state.accepted
            ]
            if unsafe:
                raise PromotionAutomationError(
                    "refusing to launch over incomplete immutable runs: " + ", ".join(unsafe)
                )
            pending = [state.name for state in refreshed if not state.accepted]
            _write_json_atomic(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running_remaining_promotion",
                    "updated_at_utc": _utc_now(),
                    "ordered_jobs": [job.name for job in jobs],
                    "accepted_jobs": [state.name for state in refreshed if state.accepted],
                    "pending_jobs": pending,
                    "formal_stage_launched": False,
                },
            )
            recorder.emit("launching_remaining_jobs", pending_jobs=pending)
            if pending:
                returncode = stream_command(
                    runner_command(
                        sys.executable,
                        "run_all_promotion",
                        device=args.device,
                    ),
                    child_log,
                )
                if returncode != 0:
                    raise PromotionAutomationError(
                        f"remaining promotion runner failed with return code {returncode}"
                    )

            status = capture_json_command(
                runner_command(sys.executable, "status", device=args.device),
                child_log,
            )
            assert_promotion_complete_without_formal(status)
            summary = capture_json_command(
                runner_command(
                    sys.executable,
                    "summarize_promotion",
                    device=args.device,
                ),
                child_log,
            )
            gate = capture_json_command(
                runner_command(
                    sys.executable,
                    "gate_promotion",
                    device=args.device,
                ),
                child_log,
            )
            final_status = capture_json_command(
                runner_command(sys.executable, "status", device=args.device),
                child_log,
            )
            assert_promotion_complete_without_formal(final_status)
            decision = gate.get("decision")
            if decision not in {"pass", "fail"}:
                raise PromotionAutomationError(f"unexpected promotion gate decision: {decision!r}")
            final_state = {
                "schema_version": SCHEMA_VERSION,
                "status": "promotion_gate_passed" if decision == "pass" else "promotion_gate_failed",
                "updated_at_utc": _utc_now(),
                "promotion_gate_decision": decision,
                "promotion_accepted": final_status["stages"]["promotion"]["accepted"],
                "formal_accepted": final_status["stages"]["formal"]["accepted"],
                "formal_stage_launched": False,
                "formal_test_unlocked": final_status.get("formal_test_unlocked"),
                "summary_schema_version": summary.get("schema_version"),
                "gate": gate,
            }
            _write_json_atomic(state_path, final_state)
            recorder.emit(
                "automation_finished",
                decision=decision,
                promotion_accepted=12,
                formal_accepted=0,
                formal_stage_launched=False,
            )
            return 0 if decision == "pass" else 2
        except Exception as exc:
            failure_state = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "updated_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "formal_stage_launched": False,
            }
            _write_json_atomic(state_path, failure_state)
            recorder.emit(
                "automation_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                formal_stage_launched=False,
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
