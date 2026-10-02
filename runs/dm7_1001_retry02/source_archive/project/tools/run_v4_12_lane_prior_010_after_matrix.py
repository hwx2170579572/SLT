"""Run the global 0.10 lane-prior sensitivity after the 0.00 matrix exits."""

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

from tools import evaluate_v4_12_frozen_lane_prior_ablation as evaluator
from tools import run_v4_12_lane_prior_matrix_after_process as supervisor


DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
    / "closed_loop_lane_prior_0_10_D3"
)
DEFAULT_RECEIPT = DEFAULT_OUTPUT.parent / "lane_prior_0_10_supervisor.json"


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
    value.add_argument("--max-wait-seconds", type=float, default=10_800.0)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        "schema_version": "topo-scene-v4.12.lane-prior-0.10-supervisor/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_independent": True,
        "wait": wait,
        "formal_test_accessed": False,
    }
    try:
        result = evaluator.evaluate(
            source_run=evaluator.DEFAULT_SOURCE_RUN,
            output_dir=args.output_dir,
            lane_prior_coef=0.10,
            device=args.device,
            episodes=None,
            evaluation_seed_start=None,
            allow_increase=True,
        )
        payload["evaluation"] = {
            "complete": True,
            "source_summary": result["source_summary"],
            "sensitivity_summary": result["ablation_summary"],
            "sensitivity_minus_source": result["ablation_minus_source"],
            "output_dir": str(args.output_dir.resolve()),
        }
    except Exception as error:
        payload["evaluation"] = {
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "output_dir": str(args.output_dir.resolve()),
        }
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.receipt.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["evaluation"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
