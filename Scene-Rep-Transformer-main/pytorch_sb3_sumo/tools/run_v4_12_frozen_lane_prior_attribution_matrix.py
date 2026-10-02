"""Run all v4.12 development lane-prior closed-loop attributions.

Every missing cell is attempted even if another cell fails.  Existing cells
are accepted only when their immutable receipt is complete.  The aggregate is
diagnostic evidence for the next preregistration and never unlocks formal test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import evaluate_v4_12_frozen_lane_prior_ablation as evaluator


ATTRIBUTION_ROOT = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
)
DEFAULT_SUMMARY = ATTRIBUTION_ROOT / "closed_loop_lane_prior_matrix.json"
CELLS = (
    (
        "D1",
        ROOT / "r412" / "d" / "D1__ajs__x__s10__p3f978ef9",
        ATTRIBUTION_ROOT / "closed_loop_lane_prior_0_D1",
    ),
    (
        "D2",
        ROOT / "r412" / "d" / "D2__ajs__x__s11__p3f978ef9",
        ATTRIBUTION_ROOT / "closed_loop_lane_prior_0_D2",
    ),
    (
        "D3",
        ROOT / "r412" / "d" / "D3__ajs__ram__s12__p3f978ef9",
        ATTRIBUTION_ROOT / "closed_loop_lane_prior_0",
    ),
    (
        "D4",
        ROOT / "r412" / "d" / "D4__ajs__ca__s13__p3f978ef9",
        ATTRIBUTION_ROOT / "closed_loop_lane_prior_0_D4",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _accepted_existing(output_dir: Path) -> dict[str, Any] | None:
    receipt_path = output_dir / "receipt.json"
    evaluation_path = output_dir / "closed_loop_evaluation.json"
    if not receipt_path.is_file() or not evaluation_path.is_file():
        return None
    receipt = _load_json(receipt_path)
    if receipt.get("complete") is not True:
        return None
    if receipt.get("formal_test_accessed") is not False:
        return None
    if receipt.get("evaluation_sha256") != _sha256(evaluation_path):
        return None
    return _load_json(evaluation_path)


def run_matrix(
    *,
    device: str,
    summary_path: Path = DEFAULT_SUMMARY,
    run_evaluation: Callable[..., dict[str, Any]] = evaluator.evaluate,
    cells: tuple[tuple[str, Path, Path], ...] = CELLS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell, source_run, output_dir in cells:
        row: dict[str, Any] = {
            "cell": cell,
            "source_run": str(source_run.resolve()),
            "output_dir": str(output_dir.resolve()),
            "attempted_this_invocation": False,
        }
        try:
            payload = _accepted_existing(output_dir)
            if payload is None:
                if output_dir.exists():
                    raise FileExistsError(
                        f"incomplete immutable output requires a new attempt path: {output_dir}"
                    )
                row["attempted_this_invocation"] = True
                payload = run_evaluation(
                    source_run=source_run,
                    output_dir=output_dir,
                    lane_prior_coef=0.0,
                    device=device,
                    episodes=None,
                    evaluation_seed_start=None,
                )
            row.update(
                {
                    "status": "accepted",
                    "scenario": payload.get("scenario")
                    or payload["evaluation_provenance"]["scenario"],
                    "source_summary": payload["source_summary"],
                    "ablation_summary": payload["ablation_summary"],
                    "ablation_minus_source": payload["ablation_minus_source"],
                    "evaluation_sha256": _sha256(
                        output_dir / "closed_loop_evaluation.json"
                    ),
                }
            )
        except Exception as error:  # continue-all aggregation is intentional
            row.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
    accepted = [row for row in rows if row["status"] == "accepted"]
    failed = [row for row in rows if row["status"] == "failed"]
    payload = {
        "schema_version": "topo-scene-v4.12.frozen-lane-prior-attribution-matrix/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "posthoc_closed_loop_continuous_model_ablation",
        "computed_from_real_rollouts": len(accepted) > 0,
        "fabricated_values": False,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_cells": True,
        "formal_test_accessed": False,
        "formal_test_unlocked": False,
        "complete": len(accepted) == len(cells),
        "accepted_cells": len(accepted),
        "failed_cells": len(failed),
        "total_cells": len(cells),
        "rows": rows,
        "conclusion_scope": (
            "frozen_v4_12_checkpoint_lane_prior_attribution_only_"
            "not_a_v4_13_development_gate"
        ),
    }
    _write_json(summary_path.resolve(), payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    value.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = run_matrix(device=args.device, summary_path=args.summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CELLS", "run_matrix"]
