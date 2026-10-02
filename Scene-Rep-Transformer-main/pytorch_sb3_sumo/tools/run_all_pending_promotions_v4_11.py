"""Discover and run current/future versioned promotion pipelines without fail-fast.

Pipeline modules named ``tools/run_all_v<major>_<minor>[_<patch>]_pipeline.py``
are discovered automatically. A version-level exception is recorded and does
not prevent later versions from running. Each versioned pipeline must itself
attempt every job in an entered scientific stage before aggregation.

Completed immutable scientific versions are reported but not rerun by default.
Pass ``--rerun-completed`` only for an explicit non-scientific replay request.
Formal testing is never launched here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


Promotion = Callable[..., tuple[int, dict[str, Any]]]
PIPELINE_PATTERN = re.compile(
    r"^run_all_v(?P<major>\d+)_(?P<minor>\d+)"
    r"(?:_(?P<patch>\d+))?_pipeline\.py$"
)
TERMINAL_STATUSES = {
    "promotion_gate_passed",
    "promotion_gate_failed_after_all_jobs",
    "development_gate_failed_after_all_jobs",
}
DEFAULT_REPORT = (
    ROOT / "results_promotion_automation" / "all_discovered_promotions_v4_11.json"
)


def discover_pipelines() -> tuple[tuple[str, Promotion, Path], ...]:
    discovered: list[tuple[tuple[int, int, int], str, Promotion, Path]] = []
    for path in (ROOT / "tools").glob("run_all_v*_pipeline.py"):
        match = PIPELINE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        major = int(match.group("major"))
        minor = int(match.group("minor"))
        patch = int(match.group("patch") or 0)
        version = f"v{major}.{minor}"
        if match.group("patch") is not None:
            version += f".{patch}"
        module = importlib.import_module(f"tools.{path.stem}")
        function = getattr(module, "run_all", None)
        report_path = getattr(module, "DEFAULT_REPORT", None)
        if not callable(function) or not isinstance(report_path, Path):
            raise TypeError(
                f"promotion module {path.name} must expose run_all and DEFAULT_REPORT"
            )
        discovered.append(
            ((major, minor, patch), version, function, report_path)
        )
    discovered.sort(key=lambda row: row[0])
    return tuple((version, function, report) for _, version, function, report in discovered)


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"pipeline report must be a JSON object: {path}")
    return value


def _counts(report: dict[str, Any]) -> tuple[int, int]:
    if isinstance(report.get("accepted_jobs"), int) and isinstance(
        report.get("expected_jobs"), int
    ):
        return int(report["accepted_jobs"]), int(report["expected_jobs"])
    stages = report.get("stages", [])
    if not isinstance(stages, list):
        return 0, 0
    return (
        sum(int(row.get("accepted_jobs") or 0) for row in stages),
        sum(int(row.get("expected_jobs") or 0) for row in stages),
    )


def _promotion_decision(report: dict[str, Any]) -> str | None:
    direct = report.get("promotion_gate_decision")
    if isinstance(direct, str):
        return direct
    for stage in report.get("stages", []):
        if stage.get("stage") == "promotion":
            value = stage.get("decision")
            return str(value) if value is not None else None
    return None


def run_discovered(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    pipelines: tuple[tuple[str, Promotion, Path], ...] | None = None,
    rerun_completed: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    active = discover_pipelines() if pipelines is None else pipelines
    versions: list[dict[str, Any]] = []
    for version, function, version_report_path in active:
        try:
            prior = _load_report(version_report_path)
            if (
                prior is not None
                and prior.get("status") in TERMINAL_STATUSES
                and not rerun_completed
            ):
                returncode = 0 if prior.get("status") == "promotion_gate_passed" else 2
                report = prior
                execution = "skipped_completed_immutable_version"
            else:
                returncode, report = function(
                    device=device, report_path=version_report_path
                )
                execution = "executed"
            accepted, expected = _counts(report)
            versions.append(
                {
                    "version": version,
                    "execution": execution,
                    "report": str(version_report_path.resolve()),
                    "returncode": int(returncode),
                    "status": report.get("status"),
                    "accepted_jobs": accepted,
                    "expected_jobs": expected,
                    "promotion_gate_decision": _promotion_decision(report),
                    "error": None,
                }
            )
        except Exception as exc:
            versions.append(
                {
                    "version": version,
                    "execution": "attempted_with_exception",
                    "report": str(version_report_path.resolve()),
                    "returncode": 1,
                    "status": "automation_exception",
                    "accepted_jobs": 0,
                    "expected_jobs": 0,
                    "promotion_gate_decision": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    all_passed = bool(versions) and all(
        row["returncode"] == 0 for row in versions
    )
    payload = {
        "schema_version": "topo-scene.all-discovered-promotions/v3",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": (
            "attempt_all_discovered_versions_and_all_jobs_in_each_entered_stage_"
            "then_summarize"
        ),
        "failure_does_not_cancel_remaining_versions_or_jobs": True,
        "scientific_stage_gates_preserved": True,
        "completed_versions_rerun": rerun_completed,
        "discovered_versions": [version for version, _, _ in active],
        "versions": versions,
        "aggregate": {
            "versions_total": len(versions),
            "versions_executed": sum(
                row["execution"] == "executed" for row in versions
            ),
            "versions_skipped_completed": sum(
                row["execution"] == "skipped_completed_immutable_version"
                for row in versions
            ),
            "versions_with_errors_or_failed_gates": sum(
                row["returncode"] != 0 for row in versions
            ),
            "accepted_jobs_in_entered_stages": sum(
                int(row["accepted_jobs"]) for row in versions
            ),
            "expected_jobs_in_entered_stages": sum(
                int(row["expected_jobs"]) for row in versions
            ),
        },
        "all_discovered_promotions_passed": all_passed,
        "formal_stage_launched": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (0 if all_passed else 1), payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    value.add_argument("--rerun-completed", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, payload = run_discovered(
        device=args.device,
        report_path=args.report.resolve(),
        rerun_completed=args.rerun_completed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
