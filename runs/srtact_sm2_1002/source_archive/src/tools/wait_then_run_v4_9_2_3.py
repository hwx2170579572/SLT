"""Launch short-path recovery after an active promotion process exits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "engineering_recovery_v4_9_2_3"
    / "automatic_handoff.json"
)


def same_process_alive(pid: int, expected_create_time: float) -> bool:
    try:
        process = psutil.Process(int(pid))
        return (
            process.is_running()
            and abs(process.create_time() - float(expected_create_time)) < 1.0
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def recovery_command(*, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "run_all_pending_promotions_v4_9_2_3.py"),
        "--device",
        device,
    ]


def wait_then_run(
    *,
    wait_pid: int,
    wait_create_time: float,
    device: str,
    poll_seconds: float,
    max_wait_hours: float,
    report_path: Path = DEFAULT_REPORT,
    process_probe: Callable[[int, float], bool] = same_process_alive,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    monotonic_start = time.monotonic()
    timed_out = False
    while process_probe(wait_pid, wait_create_time):
        if time.monotonic() - monotonic_start >= max_wait_hours * 3600.0:
            timed_out = True
            break
        time.sleep(poll_seconds)

    command = recovery_command(device=device)
    returncode: int | None = None
    launched = not timed_out
    if launched:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        returncode = completed.returncode
    report = {
        "schema_version": "topo-scene-v4.9.2.3.automatic-handoff/v1",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wait_pid": int(wait_pid),
        "wait_create_time": float(wait_create_time),
        "poll_seconds": float(poll_seconds),
        "max_wait_hours": float(max_wait_hours),
        "timed_out": timed_out,
        "short_path_recovery_launched": launched,
        "command": command,
        "returncode": returncode,
        "formal_stage_launched": False,
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (1 if timed_out else int(returncode or 0)), report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--wait-pid", type=int, required=True)
    value.add_argument("--wait-create-time", type=float, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--poll-seconds", type=float, default=30.0)
    value.add_argument("--max-wait-hours", type=float, default=72.0)
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, report = wait_then_run(
        wait_pid=args.wait_pid,
        wait_create_time=args.wait_create_time,
        device=args.device,
        poll_seconds=args.poll_seconds,
        max_wait_hours=args.max_wait_hours,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

