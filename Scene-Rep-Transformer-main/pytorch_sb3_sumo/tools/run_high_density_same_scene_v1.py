"""Plan, run, resume, and summarize the high-density six-method matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.high_density_same_scene_v1_common import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RESULT_ROOT,
    METHOD_ADAPTERS,
    ComparisonJob,
    build_jobs,
    load_protocol,
    profile_setting,
    protocol_run_id_prefix,
)


TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_high_density_same_scene_v1.py"
PRIMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "timeout_rate",
    "mean_return",
    "mean_success_completion_time_seconds",
)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _common_subparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--profile", default="comparison")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the immutable job matrix")
    _common_subparser(plan)

    preflight = subparsers.add_parser(
        "preflight", help="Run real-SUMO check-only jobs for all method/scenario pairs"
    )
    _common_subparser(preflight)
    preflight.set_defaults(profile="smoke")
    preflight.add_argument("--device", default="auto")
    preflight.add_argument("--workers", type=int, default=1)

    run = subparsers.add_parser("run", help="Run or resume the fresh comparison")
    _common_subparser(run)
    run.add_argument("--device", default="auto")
    run.add_argument("--workers", type=int, default=1)

    status = subparsers.add_parser("status", help="Inspect run completion")
    _common_subparser(status)

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate only validated real run artifacts"
    )
    _common_subparser(summarize)
    summarize.add_argument("--require-complete", action="store_true")
    return parser


def _profile_root(result_root: Path, profile: str) -> Path:
    return result_root.resolve() / profile


def _run_directory(result_root: Path, job: ComparisonJob) -> Path:
    return _profile_root(result_root, job.profile) / "runs" / job.run_id


def _preflight_job(
    job: ComparisonJob, *, run_id_prefix: str = "hdv1"
) -> ComparisonJob:
    return ComparisonJob(
        profile="smoke",
        method=job.method,
        scenario=job.scenario,
        seed=job.seed,
        run_id=f"{run_id_prefix}__preflight__{job.method}__{job.scenario}",
    )


def _required_artifacts(check_only: bool) -> tuple[str, ...]:
    if check_only:
        return (
            "arguments.json",
            "check_result.json",
            "high_density_job_receipt.json",
        )
    return (
        "arguments.json",
        "final_model.zip",
        "paper_evaluation_detailed.json",
        "high_density_job_receipt.json",
    )


def _run_state(
    run_dir: Path,
    *,
    protocol_sha256: str,
    check_only: bool,
) -> tuple[str, str | None]:
    if not run_dir.exists():
        return "missing", None
    missing = [
        name for name in _required_artifacts(check_only) if not (run_dir / name).is_file()
    ]
    if missing:
        return "incomplete", f"missing artifacts: {missing}"
    receipt_path = run_dir / "high_density_job_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "incomplete", f"invalid receipt: {exc}"
    expected_status = "preflight_passed" if check_only else "completed"
    if receipt.get("status") != expected_status:
        return "incomplete", "receipt status mismatch"
    if receipt.get("protocol_sha256") != protocol_sha256:
        return "incomplete", "protocol hash mismatch"
    if receipt.get("fabricated_values") is not False:
        return "incomplete", "no-fabrication receipt field mismatch"
    return "complete", None


def _job_command(
    *,
    protocol_path: Path,
    result_root: Path,
    job: ComparisonJob,
    device: str,
    check_only: bool,
) -> list[str]:
    output_dir = _profile_root(result_root, job.profile) / (
        "preflight_runs" if check_only else "runs"
    )
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--method",
        job.method,
        "--scenario",
        job.scenario,
        "--seed",
        str(job.seed),
        "--profile",
        job.profile,
        "--protocol",
        str(protocol_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--model-name",
        job.run_id,
        "--device",
        device,
    ]
    if check_only:
        command.append("--check-only")
    return command


def _execution_run_dir(
    result_root: Path, job: ComparisonJob, *, check_only: bool
) -> Path:
    folder = "preflight_runs" if check_only else "runs"
    return _profile_root(result_root, job.profile) / folder / job.run_id


def _execute_job(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    result_root: Path,
    job: ComparisonJob,
    device: str,
    check_only: bool,
) -> dict[str, Any]:
    run_dir = _execution_run_dir(result_root, job, check_only=check_only)
    state, reason = _run_state(
        run_dir,
        protocol_sha256=protocol_sha256,
        check_only=check_only,
    )
    if state == "complete":
        return {"job": job.run_id, "status": "skipped_complete", "returncode": 0}
    if state == "incomplete":
        return {
            "job": job.run_id,
            "status": "refused_incomplete_existing_directory",
            "returncode": 2,
            "reason": reason,
        }

    profile_root = _profile_root(result_root, job.profile)
    log_root = profile_root / ("preflight_logs" if check_only else "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    command = _job_command(
        protocol_path=protocol_path,
        result_root=result_root,
        job=job,
        device=device,
        check_only=check_only,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (log_root / f"{job.run_id}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_root / f"{job.run_id}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    final_state, final_reason = _run_state(
        run_dir,
        protocol_sha256=protocol_sha256,
        check_only=check_only,
    )
    success = completed.returncode == 0 and final_state == "complete"
    return {
        "job": job.run_id,
        "status": "completed" if success else "failed",
        "returncode": completed.returncode if not success else 0,
        "artifact_state": final_state,
        "reason": final_reason,
        "command": command,
    }


def _execute_matrix(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    result_root: Path,
    jobs: list[ComparisonJob],
    device: str,
    workers: int,
    check_only: bool,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_job,
                protocol_path=protocol_path,
                protocol_sha256=protocol_sha256,
                result_root=result_root,
                job=job,
                device=device,
                check_only=check_only,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(_json_text(row), flush=True)
    rows.sort(key=lambda row: row["job"])
    failures = [row for row in rows if int(row["returncode"]) != 0]
    return {
        "schema_version": "same-scene-high-density-execution/v1",
        "check_only": check_only,
        "attempted_jobs": len(rows),
        "successful_or_skipped_jobs": len(rows) - len(failures),
        "failed_jobs": len(failures),
        "rows": rows,
    }


def _status_payload(
    protocol: dict[str, Any],
    result_root: Path,
    profile: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in build_jobs(protocol, profile):
        run_dir = _run_directory(result_root, job)
        state, reason = _run_state(
            run_dir,
            protocol_sha256=protocol["_sha256"],
            check_only=False,
        )
        rows.append(
            {
                **asdict(job),
                "status": state,
                "reason": reason,
                "run_directory": str(run_dir.resolve()),
            }
        )
    counts = {
        state: sum(row["status"] == state for row in rows)
        for state in ("complete", "missing", "incomplete")
    }
    return {
        "schema_version": "same-scene-high-density-status/v1",
        "protocol_sha256": protocol["_sha256"],
        "profile": profile,
        "expected_jobs": len(rows),
        "counts": counts,
        "rows": rows,
    }


def _metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    return None if value is None else float(value)


def _read_completed_result(
    protocol: dict[str, Any], result_root: Path, job: ComparisonJob
) -> dict[str, Any] | None:
    run_dir = _run_directory(result_root, job)
    state, _ = _run_state(
        run_dir,
        protocol_sha256=protocol["_sha256"],
        check_only=False,
    )
    if state != "complete":
        return None
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    summary = detailed.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"evaluation summary is missing in {detailed_path}")
    return {
        "method": job.method,
        "display_label": protocol["methods"][job.method]["display_label"],
        "scenario": job.scenario,
        "seed": job.seed,
        "run_directory": str(run_dir.resolve()),
        "evaluation_episodes": int(summary["episodes"]),
        **{metric: _metric(summary, metric) for metric in PRIMARY_METRICS},
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    return mean, statistics.pstdev(values) if len(values) >= 2 else None


def _aggregate_results(
    protocol: dict[str, Any],
    profile: str,
    per_run: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_seeds = [int(seed) for seed in profile_setting(protocol, profile)["seeds"]]
    aggregates: list[dict[str, Any]] = []
    for method in protocol["matrix"]["method_order"]:
        for scenario in protocol["matrix"]["scenario_order"]:
            rows = [
                row
                for row in per_run
                if row["method"] == method and row["scenario"] == scenario
            ]
            row: dict[str, Any] = {
                "method": method,
                "display_label": protocol["methods"][method]["display_label"],
                "scenario": scenario,
                "expected_seeds": expected_seeds,
                "completed_seeds": sorted(item["seed"] for item in rows),
                "status": "complete" if len(rows) == len(expected_seeds) else "TBD",
            }
            for metric in PRIMARY_METRICS:
                values = [
                    float(item[metric])
                    for item in rows
                    if item.get(metric) is not None
                ]
                mean, population_std = _mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_population_std"] = population_std
            aggregates.append(row)
    return aggregates


def _macro_results(
    protocol: dict[str, Any], aggregates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scenario_count = len(protocol["matrix"]["scenario_order"])
    for method in protocol["matrix"]["method_order"]:
        rows = [row for row in aggregates if row["method"] == method]
        complete = len(rows) == scenario_count and all(
            row["status"] == "complete" for row in rows
        )
        macro: dict[str, Any] = {
            "method": method,
            "display_label": protocol["methods"][method]["display_label"],
            "status": "complete" if complete else "TBD",
        }
        for metric in PRIMARY_METRICS:
            values = [row[f"{metric}_mean"] for row in rows]
            macro[f"macro_{metric}"] = (
                statistics.fmean(float(value) for value in values)
                if complete and all(value is not None for value in values)
                else None
            )
        output.append(macro)
    return output


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "display_label",
        "scenario",
        "status",
        "expected_seeds",
        "completed_seeds",
    ] + [
        key
        for metric in PRIMARY_METRICS
        for key in (f"{metric}_mean", f"{metric}_population_std")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["expected_seeds"] = json.dumps(output["expected_seeds"])
            output["completed_seeds"] = json.dumps(output["completed_seeds"])
            writer.writerow(output)


def _summarize(
    protocol: dict[str, Any],
    result_root: Path,
    profile: str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    jobs = build_jobs(protocol, profile)
    per_run = [
        row
        for job in jobs
        if (row := _read_completed_result(protocol, result_root, job)) is not None
    ]
    aggregates = _aggregate_results(protocol, profile, per_run)
    missing_jobs = [
        job.run_id
        for job in jobs
        if not any(
            row["method"] == job.method
            and row["scenario"] == job.scenario
            and row["seed"] == job.seed
            for row in per_run
        )
    ]
    if require_complete and missing_jobs:
        raise RuntimeError(
            f"comparison is incomplete; {len(missing_jobs)} jobs remain missing or invalid"
        )
    payload = {
        "schema_version": "same-scene-high-density-summary/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "profile": profile,
        "expected_jobs": len(jobs),
        "accepted_jobs": len(per_run),
        "complete": not missing_jobs,
        "missing_jobs": missing_jobs,
        "per_run": per_run,
        "aggregate": aggregates,
        "macro": _macro_results(protocol, aggregates),
    }
    summary_root = _profile_root(result_root, profile) / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "summary.json").write_text(
        _json_text(payload), encoding="utf-8"
    )
    _write_aggregate_csv(summary_root / "aggregate.csv", aggregates)
    return payload


def _plan_payload(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    jobs = build_jobs(protocol, profile)
    settings = profile_setting(protocol, profile)
    return {
        "schema_version": "same-scene-high-density-plan/v1",
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "profile": profile,
        "run_id_prefix": protocol_run_id_prefix(protocol),
        "method_count": len(METHOD_ADAPTERS),
        "scenario_count": len(protocol["matrix"]["scenario_order"]),
        "seed_count": len(settings["seeds"]),
        "training_jobs": len(jobs),
        "evaluation_episodes": len(jobs) * int(settings["evaluation_episodes"]),
        "jobs": [asdict(job) for job in jobs],
    }


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    profile_setting(protocol, args.profile)
    result_root = args.result_root.resolve()

    if args.command == "plan":
        print(_json_text(_plan_payload(protocol, args.profile)))
        return 0
    if args.command == "status":
        print(_json_text(_status_payload(protocol, result_root, args.profile)))
        return 0
    if args.command == "summarize":
        payload = _summarize(
            protocol,
            result_root,
            args.profile,
            require_complete=args.require_complete,
        )
        print(_json_text(payload))
        return 0

    jobs = build_jobs(protocol, args.profile)
    check_only = args.command == "preflight"
    if check_only:
        first_seed = int(profile_setting(protocol, args.profile)["seeds"][0])
        run_id_prefix = protocol_run_id_prefix(protocol)
        jobs = [
            _preflight_job(
                ComparisonJob(
                    profile=args.profile,
                    method=method,
                    scenario=scenario,
                    seed=first_seed,
                    run_id="unused",
                ),
                run_id_prefix=run_id_prefix,
            )
            for method in protocol["matrix"]["method_order"]
            for scenario in protocol["matrix"]["scenario_order"]
        ]
    result = _execute_matrix(
        protocol_path=Path(protocol["_path"]),
        protocol_sha256=protocol["_sha256"],
        result_root=result_root,
        jobs=jobs,
        device=args.device,
        workers=args.workers,
        check_only=check_only,
    )
    print(_json_text(result))
    return 0 if result["failed_jobs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
