"""Run every registered current/future promotion protocol without fail-fast.

New scientifically versioned promotion automations are appended to ``REGISTRY``.
Each automation is responsible for attempting its full matrix and aggregating
once.  A failed version is recorded and later registered versions still run;
formal testing is never launched here.
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

from tools import run_all_v4_10_pipeline


Promotion = Callable[..., tuple[int, dict[str, Any]]]
REGISTRY: tuple[tuple[str, Promotion], ...] = (
    ("v4.10", run_all_v4_10_pipeline.run_all),
)
COMPLETED_HISTORY = (
    {
        "version": "v4.9.2-engineering-recovery-v4.9.2.3",
        "status": "promotion_gate_failed_after_complete_12_job_matrix",
        "accepted_jobs": 12,
        "expected_jobs": 12,
        "summary_sha256": "c3204ac944b71899c4ea6c3175986db88092661594a112e6b9bc7bb8a155b96d",
        "promotion_gate_sha256": "080cff14f2edabaf994f8349144d21b570b977cc0d0129be2dee15090eddab56",
        "rerun": False,
    },
)
DEFAULT_REPORT = ROOT / "results_promotion_automation" / "all_registered_promotions.json"


def run_registered(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    registry: tuple[tuple[str, Promotion], ...] | None = None,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    versions = []
    active = REGISTRY if registry is None else registry
    for version, function in active:
        try:
            returncode, report = function(device=device)
            versions.append(
                {
                    "version": version,
                    "returncode": returncode,
                    "status": report.get("status"),
                    "accepted_jobs": report.get("accepted_jobs"),
                    "expected_jobs": report.get("expected_jobs"),
                    "promotion_gate_decision": report.get("promotion_gate_decision"),
                    "error": None,
                }
            )
        except Exception as exc:
            versions.append(
                {
                    "version": version,
                    "returncode": 1,
                    "status": "automation_exception",
                    "accepted_jobs": None,
                    "expected_jobs": None,
                    "promotion_gate_decision": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passed = bool(versions) and all(row["returncode"] == 0 for row in versions)
    payload = {
        "schema_version": "topo-scene.all-registered-promotions/v2",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "attempt_all_versions_and_all_jobs_then_summarize",
        "failure_does_not_cancel_remaining_versions_or_jobs": True,
        "completed_history": list(COMPLETED_HISTORY),
        "registered_versions": [version for version, _ in active],
        "versions": versions,
        "all_registered_promotions_passed": passed,
        "formal_stage_launched": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return (0 if passed else 1), payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--device", default="cuda")
    root.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, payload = run_registered(device=args.device, report_path=args.report.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
