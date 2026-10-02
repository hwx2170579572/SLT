"""Run the real-state v4.12 gradient diagnostic after a detached process."""

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

from tools import diagnose_v4_12_lane_speed_gradient_coupling as diagnostic
from tools import run_v4_12_lane_prior_matrix_after_process as supervisor


DEFAULT_RECEIPT = diagnostic.DEFAULT_OUTPUT.with_name(
    "lane_speed_gradient_coupling_supervisor.json"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--wait-pid", type=int, required=True)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    value.add_argument("--poll-seconds", type=float, default=10.0)
    value.add_argument("--max-wait-seconds", type=float, default=14_400.0)
    value.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    wait = supervisor.wait_for_process(
        args.wait_pid,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    payload: dict[str, Any] = {
        "schema_version": "topo-scene-v4.12.gradient-diagnostic-supervisor/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_independent": True,
        "wait": wait,
        "formal_test_accessed": False,
    }
    try:
        result = diagnostic.diagnose(
            source_run=diagnostic.shared.DEFAULT_SOURCE_RUN,
            output=diagnostic.DEFAULT_OUTPUT,
            device=args.device,
            samples=128,
            batch_size=32,
            diagnostic_seed=20260902,
        )
        payload["diagnostic"] = {
            "complete": True,
            "output": str(diagnostic.DEFAULT_OUTPUT.resolve()),
            "architecture_finding": result["architecture_finding"],
        }
    except Exception as error:
        payload["diagnostic"] = {
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.receipt.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["diagnostic"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
