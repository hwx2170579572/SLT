"""Wait for one detached attribution process, then run the full matrix.

This tiny supervisor exists so interrupting the Codex session cannot prevent
the remaining D1--D4 diagnostic cells from being attempted.  A failure or
timeout of the prerequisite process still proceeds to the continue-all matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v4_12_frozen_lane_prior_attribution_matrix as matrix


DEFAULT_SUPERVISOR_RECEIPT = (
    matrix.ATTRIBUTION_ROOT / "closed_loop_lane_prior_supervisor.json"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def wait_for_process(pid: int, *, poll_seconds: float, max_wait_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    observed_alive = False
    while True:
        alive = psutil.pid_exists(pid)
        if alive:
            try:
                process = psutil.Process(pid)
                alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                alive = psutil.pid_exists(pid)
        observed_alive |= alive
        elapsed = time.monotonic() - started
        if not alive:
            return {
                "pid": pid,
                "observed_alive": observed_alive,
                "wait_status": "process_exited",
                "wait_seconds": elapsed,
            }
        if elapsed >= max_wait_seconds:
            return {
                "pid": pid,
                "observed_alive": observed_alive,
                "wait_status": "timeout_proceeding_to_matrix",
                "wait_seconds": elapsed,
            }
        time.sleep(poll_seconds)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--wait-pid", type=int, required=True)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    value.add_argument("--poll-seconds", type=float, default=10.0)
    value.add_argument("--max-wait-seconds", type=float, default=10_800.0)
    value.add_argument("--receipt", type=Path, default=DEFAULT_SUPERVISOR_RECEIPT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.wait_pid <= 0 or args.poll_seconds <= 0 or args.max_wait_seconds <= 0:
        raise ValueError("supervisor timing and pid arguments must be positive")
    wait = wait_for_process(
        args.wait_pid,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    payload: dict[str, Any] = {
        "schema_version": "topo-scene-v4.12.lane-prior-matrix-supervisor/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_independent": True,
        "wait": wait,
        "matrix_started_after_wait": True,
    }
    try:
        result = matrix.run_matrix(device=args.device)
        payload["matrix"] = {
            "complete": result["complete"],
            "accepted_cells": result["accepted_cells"],
            "failed_cells": result["failed_cells"],
            "summary": str(matrix.DEFAULT_SUMMARY.resolve()),
        }
    except Exception as error:
        payload["matrix"] = {
            "complete": False,
            "accepted_cells": 0,
            "failed_cells": len(matrix.CELLS),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.receipt.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["matrix"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["wait_for_process"]
