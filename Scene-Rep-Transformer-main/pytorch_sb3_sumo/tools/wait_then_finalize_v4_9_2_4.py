"""Finalize v4.9.2 promotion evidence after the short-path watcher exits.

The finalizer is deliberately read-only with respect to scientific runs.  It
attempts every registered analysis even if an earlier analysis fails, writes a
single evidence-status report, and never launches the formal stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import psutil


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_finalizer_v4_9_2_4"
)
DEFAULT_REPORT = RESULT_ROOT / "automatic_finalization.json"
POST_PROMOTION_AUDIT = RESULT_ROOT / "kinematic_safety_model_reaudit.json"
FINAL_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "attribution"
    / "promotion_model_attribution.json"
)
V4923_ENGINE_REPORT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_recovery_v4_9_2_3"
    / "engine_report.json"
)
FORMAL_ROOT = ROOT / "results_topo_v4_9_2_formal"


def same_process_alive(pid: int, expected_create_time: float) -> bool:
    try:
        process = psutil.Process(int(pid))
        return (
            process.is_running()
            and abs(process.create_time() - float(expected_create_time)) < 1.0
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def analysis_commands() -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "post_promotion_kinematic_model_audit",
            [
                sys.executable,
                str(ROOT / "tools" / "audit_v4_9_2_kinematic_safety_model.py"),
                "--output",
                str(POST_PROMOTION_AUDIT),
            ],
        ),
        (
            "complete_six_pair_model_attribution",
            [
                sys.executable,
                str(ROOT / "tools" / "attribute_v4_9_2_promotion.py"),
                "--output",
                str(FINAL_ATTRIBUTION),
            ],
        ),
    )


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def evidence_snapshot(
    *,
    engine_report_path: Path = V4923_ENGINE_REPORT,
    attribution_path: Path = FINAL_ATTRIBUTION,
    formal_root: Path = FORMAL_ROOT,
) -> dict[str, Any]:
    engine_error: str | None = None
    attribution_error: str | None = None
    try:
        engine = _load_optional(engine_report_path)
    except Exception as exc:
        engine = None
        engine_error = f"{type(exc).__name__}: {exc}"
    try:
        attribution = _load_optional(attribution_path)
    except Exception as exc:
        attribution = None
        attribution_error = f"{type(exc).__name__}: {exc}"
    formal_files = (
        sorted(str(path.resolve()) for path in formal_root.rglob("*") if path.is_file())
        if formal_root.is_dir()
        else []
    )
    matrix_complete = bool(
        engine
        and engine.get("promotion_matrix_complete") is True
        and int(engine.get("accepted_jobs", 0)) == int(engine.get("expected_jobs", -1))
        and int(engine.get("expected_jobs", 0)) == 12
    )
    attribution_complete = bool(
        attribution
        and attribution.get("decision_allowed") is True
        and attribution.get("partial_matrix") is False
        and int(attribution.get("pair_count", 0)) == int(
            attribution.get("expected_pair_count", -1)
        )
        and int(attribution.get("expected_pair_count", 0)) == 6
    )
    ready = matrix_complete and attribution_complete
    gate_decision = engine.get("promotion_gate_decision") if engine else None
    if not ready:
        next_stage = "finish_or_recover_unresolved_promotion_evidence"
    elif gate_decision == "pass":
        next_stage = "review_promotion_unlock_then_run_untouched_formal"
    else:
        next_stage = "ccf_idea_optimizer_model_only_v4_10_iteration"
    return {
        "engine_report": str(engine_report_path.resolve()),
        "engine_report_present": engine is not None,
        "engine_report_error": engine_error,
        "accepted_jobs": engine.get("accepted_jobs") if engine else None,
        "expected_jobs": engine.get("expected_jobs") if engine else None,
        "promotion_matrix_complete": matrix_complete,
        "promotion_gate_decision": gate_decision,
        "attribution": str(attribution_path.resolve()),
        "attribution_present": attribution is not None,
        "attribution_error": attribution_error,
        "attribution_pair_count": attribution.get("pair_count") if attribution else None,
        "attribution_expected_pair_count": (
            attribution.get("expected_pair_count") if attribution else None
        ),
        "attribution_complete": attribution_complete,
        "ready_for_scientific_decision": ready,
        "formal_files": formal_files,
        "formal_test_accessed": bool(formal_files),
        "formal_stage_launched_by_finalizer": False,
        "next_stage": next_stage,
    }


def _run_all_analyses(
    commands: Iterable[tuple[str, list[str]]],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, command in commands:
        started = datetime.now(timezone.utc).isoformat()
        try:
            completed = command_runner(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            returncode = int(completed.returncode)
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
            error = None
        except Exception as exc:
            returncode = 1
            stdout = ""
            stderr = ""
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "name": name,
                "started_at_utc": started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "returncode": returncode,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "error": error,
            }
        )
    return rows


def wait_then_finalize(
    *,
    wait_pid: int,
    wait_create_time: float,
    poll_seconds: float,
    max_wait_hours: float,
    report_path: Path = DEFAULT_REPORT,
    process_probe: Callable[[int, float], bool] = same_process_alive,
    command_runner: Callable[..., Any] = subprocess.run,
    commands: Iterable[tuple[str, list[str]]] | None = None,
    snapshot_builder: Callable[[], dict[str, Any]] = evidence_snapshot,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    monotonic_start = time.monotonic()
    timed_out = False
    while process_probe(wait_pid, wait_create_time):
        if time.monotonic() - monotonic_start >= max_wait_hours * 3600.0:
            timed_out = True
            break
        time.sleep(poll_seconds)

    analyses = [] if timed_out else _run_all_analyses(
        analysis_commands() if commands is None else commands,
        command_runner=command_runner,
    )
    try:
        snapshot = snapshot_builder()
        snapshot_error = None
    except Exception as exc:
        snapshot_error = f"{type(exc).__name__}: {exc}"
        snapshot = {
            "ready_for_scientific_decision": False,
            "formal_test_accessed": False,
            "next_stage": "repair_evidence_snapshot_then_refinalize",
        }
    analyses_passed = bool(analyses) and all(
        row["returncode"] == 0 for row in analyses
    )
    passed = (
        not timed_out
        and analyses_passed
        and snapshot.get("ready_for_scientific_decision") is True
        and snapshot.get("formal_test_accessed") is False
    )
    report = {
        "schema_version": "topo-scene-v4.9.2.4.automatic-post-promotion-finalizer/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wait_pid": int(wait_pid),
        "wait_create_time": float(wait_create_time),
        "poll_seconds": float(poll_seconds),
        "max_wait_hours": float(max_wait_hours),
        "timed_out": timed_out,
        "execution_policy": "attempt_all_analyses_then_summarize",
        "failure_does_not_cancel_later_analyses": True,
        "scientific_model_changed": False,
        "training_protocol_changed": False,
        "source_run_trees_modified": False,
        "analyses": analyses,
        "analyses_passed": analyses_passed,
        "evidence": snapshot,
        "evidence_snapshot_error": snapshot_error,
        "finalization_passed": passed,
        "formal_stage_launched": False,
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (0 if passed else 1), report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--wait-pid", type=int, required=True)
    value.add_argument("--wait-create-time", type=float, required=True)
    value.add_argument("--poll-seconds", type=float, default=30.0)
    value.add_argument("--max-wait-hours", type=float, default=96.0)
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, report = wait_then_finalize(
        wait_pid=args.wait_pid,
        wait_create_time=args.wait_create_time,
        poll_seconds=args.poll_seconds,
        max_wait_hours=args.max_wait_hours,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
