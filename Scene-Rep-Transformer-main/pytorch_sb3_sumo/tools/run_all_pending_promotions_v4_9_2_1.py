"""Run all current and later registered promotions without fail-fast.

Append each later *scientific* promotion automation to ``REGISTRY``.  A failed
version is recorded, subsequent versions still execute, and only after all
registered versions finish is the aggregate report written.  Formal testing
is never started by this registry.
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

from tools import run_all_v4_9_2_1_promotion


Promotion = Callable[..., tuple[int, dict[str, Any]]]
REGISTRY: tuple[tuple[str, Promotion], ...] = (
    ("v4.9.2+v4.9.2.1-engineering", run_all_v4_9_2_1_promotion.run_all),
)
DEFAULT_REPORT = (
    ROOT
    / "results_promotion_automation"
    / "all_registered_promotions_v4_9_2_1.json"
)


def run_registered(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    registry: tuple[tuple[str, Promotion], ...] | None = None,
) -> tuple[int, dict[str, Any]]:
    active_registry = REGISTRY if registry is None else registry
    started = datetime.now(timezone.utc).isoformat()
    versions: list[dict[str, Any]] = []
    for version, function in active_registry:
        try:
            returncode, report = function(device=device)
            versions.append(
                {
                    "version": version,
                    "returncode": returncode,
                    "status": report.get("status"),
                    "accepted_jobs": report.get("accepted_jobs"),
                    "expected_jobs": report.get("expected_jobs"),
                    "promotion_gate_decision": report.get(
                        "promotion_gate_decision"
                    ),
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
        "schema_version": (
            "topo-scene.all-registered-promotions.v4.9.2.1/v1"
        ),
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": (
            "attempt_all_versions_and_all_jobs_then_summarize"
        ),
        "failure_does_not_cancel_remaining_versions_or_jobs": True,
        "registered_versions": [version for version, _ in active_registry],
        "versions": versions,
        "all_registered_promotions_passed": passed,
        "formal_stage_launched": False,
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (0 if passed else 1), payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, payload = run_registered(
        device=args.device,
        report_path=args.report.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

