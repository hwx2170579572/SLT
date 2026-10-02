"""Run the newest implementation of every versioned promotion family.

Engineering recovery patches such as v4.11.2 supersede v4.11 and v4.11.1 for
automation purposes.  Scientific version families are still all considered,
and a failed family does not cancel later families.  An already-running version
pipeline is reported rather than launched a second time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_all_pending_promotions_v4_11 as legacy


VERSION = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?$")
TERMINAL_STATUSES = set(legacy.TERMINAL_STATUSES)
DEFAULT_REPORT = (
    ROOT
    / "results_promotion_automation"
    / "all_latest_family_promotions_v4_12.json"
)


def _version_key(version: str) -> tuple[int, int, int]:
    match = VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"invalid discovered version: {version}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


def latest_family_pipelines():
    latest: dict[tuple[int, int], tuple[str, legacy.Promotion, Path]] = {}
    superseded: dict[str, str] = {}
    for version, function, report in legacy.discover_pipelines():
        major, minor, patch = _version_key(version)
        family = (major, minor)
        current = latest.get(family)
        if current is None or patch > _version_key(current[0])[2]:
            if current is not None:
                superseded[current[0]] = version
            latest[family] = (version, function, report)
        else:
            superseded[version] = current[0]
    selected = tuple(latest[key] for key in sorted(latest))
    return selected, superseded


def _pipeline_script(version: str) -> str:
    major, minor, patch = _version_key(version)
    suffix = f"_{patch}" if "." in version[version.find(".") + 1 :] else ""
    return f"run_all_v{major}_{minor}{suffix}_pipeline.py"


def running_pipeline_pid(version: str) -> int | None:
    target = _pipeline_script(version).lower()
    own_pid = psutil.Process().pid
    for process in psutil.process_iter(["name"]):
        if process.pid == own_pid:
            continue
        try:
            command = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if len(command) > 1 and Path(command[1]).name.lower() == target:
            return int(process.pid)
    return None


def run_latest(
    *,
    device: str,
    report_path: Path = DEFAULT_REPORT,
    rerun_completed: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    selected, superseded = latest_family_pipelines()
    versions = []
    for version, function, version_report_path in selected:
        prior = legacy._load_report(version_report_path)
        pid = running_pipeline_pid(version)
        try:
            if pid is not None:
                report = prior or {}
                returncode = 0
                execution = "already_running_detached"
                error = None
            elif (
                prior is not None
                and prior.get("status") in TERMINAL_STATUSES
                and not rerun_completed
            ):
                report = prior
                returncode = 0 if prior.get("status") == "promotion_gate_passed" else 2
                execution = "skipped_completed_immutable_version"
                error = None
            else:
                returncode, report = function(
                    device=device, report_path=version_report_path
                )
                execution = "executed"
                error = None
            accepted, expected = legacy._counts(report)
            versions.append(
                {
                    "version": version,
                    "execution": execution,
                    "active_pid": pid,
                    "report": str(version_report_path.resolve()),
                    "returncode": int(returncode),
                    "status": report.get("status"),
                    "accepted_jobs": accepted,
                    "expected_jobs": expected,
                    "promotion_gate_decision": legacy._promotion_decision(report),
                    "error": error,
                }
            )
        except Exception as exc:
            versions.append(
                {
                    "version": version,
                    "execution": "attempted_with_exception",
                    "active_pid": pid,
                    "report": str(version_report_path.resolve()),
                    "returncode": 1,
                    "status": "automation_exception",
                    "accepted_jobs": 0,
                    "expected_jobs": 0,
                    "promotion_gate_decision": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    payload = {
        "schema_version": "topo-scene.all-latest-family-promotions/v4",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "attempt_all_latest_version_families_then_summarize",
        "failure_does_not_cancel_remaining_versions_or_jobs": True,
        "scientific_stage_gates_preserved": True,
        "newest_engineering_patch_per_scientific_family": True,
        "superseded_pipeline_versions": superseded,
        "selected_versions": [version for version, _, _ in selected],
        "versions": versions,
        "aggregate": {
            "families_total": len(versions),
            "families_executed": sum(row["execution"] == "executed" for row in versions),
            "families_already_running": sum(row["execution"] == "already_running_detached" for row in versions),
            "families_with_errors_or_failed_gates": sum(row["returncode"] != 0 for row in versions),
            "accepted_jobs_in_entered_stages": sum(int(row["accepted_jobs"]) for row in versions),
            "expected_jobs_in_entered_stages": sum(int(row["expected_jobs"]) for row in versions),
        },
        "formal_stage_launched": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    complete = bool(versions) and all(
        row["returncode"] == 0
        or row["execution"] == "already_running_detached"
        for row in versions
    )
    return (0 if complete else 1), payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--device", default="cuda")
    value.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    value.add_argument("--rerun-completed", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, payload = run_latest(
        device=args.device,
        report_path=args.report.resolve(),
        rerun_completed=args.rerun_completed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

