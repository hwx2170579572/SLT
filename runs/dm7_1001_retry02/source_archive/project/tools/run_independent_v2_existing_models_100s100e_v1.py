"""Plan, audit, run, resume, and summarize the 6x3x100x100 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (  # noqa: E402
    evaluate_independent_v2_existing_models_100s100e_v1 as study,
)


DEFAULT_PROTOCOL_PATH = study.DEFAULT_PROTOCOL_PATH
DEFAULT_RESULT_ROOT = study.DEFAULT_RESULT_ROOT
EVALUATOR_SCRIPT = PROJECT_ROOT / "tools" / (
    "evaluate_independent_v2_existing_models_100s100e_v1.py"
)
RUNNER_SCRIPT = Path(__file__).resolve()
SUMMARY_SCHEMA_VERSION = "independent-v2-existing-models-summary/v1"
RATE_METRICS = ("success_rate", "collision_rate", "off_route_rate", "timeout_rate")
BLOCK_METRICS = RATE_METRICS + (
    "mean_return",
    "mean_decision_steps",
    "mean_raw_steps",
    "mean_success_completion_time_seconds",
)


def _json_text(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".j{os.getpid()}.tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".c{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _common_subparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)


def _execution_subparser(parser: argparse.ArgumentParser) -> None:
    _common_subparser(parser)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="Dedicated CPU worker slots; use with --gpu-workers for a mixed pool",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=0,
        help="Dedicated CUDA worker slots; use with --cpu-workers for a mixed pool",
    )


def _device_worker_plan(
    *,
    device: str,
    workers: int,
    cpu_workers: int,
    gpu_workers: int,
) -> dict[str, int]:
    """Resolve legacy homogeneous options or an explicit CPU/CUDA worker pool."""
    if cpu_workers < 0 or gpu_workers < 0:
        raise ValueError("cpu-workers and gpu-workers must be non-negative")
    if cpu_workers or gpu_workers:
        if device != "auto" or workers != 1:
            raise ValueError(
                "--cpu-workers/--gpu-workers cannot be combined with a non-default "
                "--device or --workers"
            )
        plan: dict[str, int] = {}
        if cpu_workers:
            plan["cpu"] = cpu_workers
        if gpu_workers:
            plan["cuda"] = gpu_workers
        return plan
    if workers <= 0:
        raise ValueError("workers must be positive")
    return {device: workers}


def _device_worker_slots(worker_plan: dict[str, int]) -> list[tuple[str, int]]:
    slots: list[tuple[str, int]] = []
    for device, count in worker_plan.items():
        if count <= 0:
            raise ValueError(f"worker count for {device!r} must be positive")
        slots.extend((device, slot_index) for slot_index in range(count))
    if not slots:
        raise ValueError("at least one worker slot is required")
    return slots


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the frozen evaluation matrix")
    _common_subparser(plan)
    plan.add_argument("--profile", default="comparison")

    audit = subparsers.add_parser(
        "source-audit", help="Verify every selected parent deployment and hash"
    )
    _common_subparser(audit)

    preflight = subparsers.add_parser(
        "preflight", help="Load every deployment and run the smoke seed block"
    )
    _execution_subparser(preflight)

    run = subparsers.add_parser("run", help="Run or resume all formal seed blocks")
    _execution_subparser(run)

    status = subparsers.add_parser("status", help="Inspect block-level completion")
    _common_subparser(status)
    status.add_argument("--profile", default="comparison")

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate only validated real block artifacts"
    )
    _common_subparser(summarize)
    summarize.add_argument("--profile", default="comparison")
    summarize.add_argument("--require-complete", action="store_true")

    launch = subparsers.add_parser(
        "launch", help="Start the formal resumable run in a hidden background process"
    )
    _execution_subparser(launch)
    launch.add_argument(
        "--wait-for-pid",
        type=int,
        default=None,
        help="Queue until this exact local process exits before starting",
    )

    queued_run = subparsers.add_parser("_queued-run", help=argparse.SUPPRESS)
    _execution_subparser(queued_run)
    queued_run.add_argument("--wait-for-pid", type=int, required=True)
    queued_run.add_argument("--wait-for-create-time", type=float, required=True)

    return parser


def _plan_payload(protocol: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile = study.profile_setting(protocol, profile_name)
    jobs = study.build_jobs(protocol, profile_name)
    block_count = len(jobs) * int(profile["test_seed_count"])
    total_episodes = block_count * int(profile["episodes_per_test_seed"])
    return {
        "schema_version": "independent-v2-existing-models-plan/v1",
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "profile": profile_name,
        "method_count": len(protocol["matrix"]["method_order"]),
        "scenario_count": len(protocol["matrix"]["scenario_order"]),
        "method_scenario_jobs": len(jobs),
        "test_seed_blocks_per_job": int(profile["test_seed_count"]),
        "episodes_per_test_seed": int(profile["episodes_per_test_seed"]),
        "evaluation_blocks": block_count,
        "total_evaluation_episodes": total_episodes,
        "training_jobs": 0,
        "parent_models_reused": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }


def _source_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in protocol["matrix"]["method_order"]:
        for scenario in protocol["matrix"]["scenario_order"]:
            try:
                source = study.resolve_source_model(protocol, method, scenario)
                rows.append(
                    {
                        key: value
                        for key, value in source.items()
                        if key != "requested_arguments"
                    }
                )
            except Exception as exc:
                errors.append(
                    {"method": method, "scenario": scenario, "error": str(exc)}
                )
    model_hashes = [row["model_sha256"] for row in rows]
    expected_models = len(protocol["matrix"]["method_order"]) * len(
        protocol["matrix"]["scenario_order"]
    )
    return {
        "schema_version": "independent-v2-existing-models-source-audit/v1",
        "protocol_sha256": protocol["_sha256"],
        "expected_models": expected_models,
        "validated_models": len(rows),
        "unique_model_hashes": len(set(model_hashes)),
        "complete": len(rows) == expected_models and not errors,
        "training_performed": False,
        "rows": rows,
        "errors": errors,
    }


def _job_status(
    *,
    protocol: dict[str, Any],
    result_root: Path,
    job: study.EvaluationJob,
) -> dict[str, Any]:
    profile = study.profile_setting(protocol, job.profile)
    indices = study.logical_test_seed_indices(profile)
    episodes = int(profile["episodes_per_test_seed"])
    try:
        source = study.resolve_source_model(protocol, job.method, job.scenario)
    except Exception as exc:
        return {
            **asdict(job),
            "status": "source_invalid",
            "expected_blocks": len(indices),
            "valid_blocks": 0,
            "missing_blocks": len(indices),
            "invalid_blocks": 0,
            "completed_episodes": 0,
            "source_error": str(exc),
            "invalid_details": [],
        }

    valid = 0
    missing = 0
    invalid: list[dict[str, Any]] = []
    for logical_index in indices:
        path = study.block_path(result_root, job, logical_index)
        if not path.exists():
            missing += 1
            continue
        try:
            study.read_valid_block(
                path,
                protocol_sha256=protocol["_sha256"],
                job=job,
                source_model_sha256=source["model_sha256"],
                logical_index=logical_index,
                expected_seed_start=study.episode_seed_start(profile, logical_index),
                expected_episodes=episodes,
            )
            valid += 1
        except Exception as exc:
            invalid.append(
                {
                    "logical_test_seed_index": logical_index,
                    "path": str(path.resolve()),
                    "error": str(exc),
                }
            )
    if invalid:
        state = "invalid"
    elif valid == len(indices):
        state = "complete"
    elif valid:
        state = "partial"
    else:
        state = "missing"
    return {
        **asdict(job),
        "status": state,
        "expected_blocks": len(indices),
        "valid_blocks": valid,
        "missing_blocks": missing,
        "invalid_blocks": len(invalid),
        "completed_episodes": valid * episodes,
        "expected_episodes": len(indices) * episodes,
        "model_basename": source["model_basename"],
        "model_sha256": source["model_sha256"],
        "invalid_details": invalid,
    }


def _status_payload(
    protocol: dict[str, Any], result_root: Path, profile_name: str
) -> dict[str, Any]:
    rows = [
        _job_status(protocol=protocol, result_root=result_root, job=job)
        for job in study.build_jobs(protocol, profile_name)
    ]
    return {
        "schema_version": "independent-v2-existing-models-status/v1",
        "protocol_sha256": protocol["_sha256"],
        "profile": profile_name,
        "expected_jobs": len(rows),
        "complete_jobs": sum(row["status"] == "complete" for row in rows),
        "partial_jobs": sum(row["status"] == "partial" for row in rows),
        "missing_jobs": sum(row["status"] == "missing" for row in rows),
        "invalid_jobs": sum(
            row["status"] in {"invalid", "source_invalid"} for row in rows
        ),
        "completed_blocks": sum(int(row["valid_blocks"]) for row in rows),
        "expected_blocks": sum(int(row["expected_blocks"]) for row in rows),
        "completed_episodes": sum(int(row["completed_episodes"]) for row in rows),
        "expected_episodes": sum(int(row.get("expected_episodes", 0)) for row in rows),
        "rows": rows,
    }


def _job_command(
    *,
    protocol_path: Path,
    result_root: Path,
    job: study.EvaluationJob,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR_SCRIPT),
        "--method",
        job.method,
        "--scenario",
        job.scenario,
        "--profile",
        job.profile,
        "--protocol",
        str(protocol_path.resolve()),
        "--result-root",
        str(result_root.resolve()),
        "--device",
        device,
    ]


def _execute_job(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    result_root: Path,
    job: study.EvaluationJob,
    device: str,
) -> dict[str, Any]:
    before = _job_status(protocol=protocol, result_root=result_root, job=job)
    if before["status"] == "complete":
        return {
            "job": job.run_id,
            "status": "skipped_complete",
            "returncode": 0,
            "device": device,
        }
    if before["status"] in {"invalid", "source_invalid"}:
        return {
            "job": job.run_id,
            "status": "refused_invalid_existing_state",
            "returncode": 2,
            "device": device,
            "details": before,
        }

    log_root = result_root.resolve() / job.profile / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{job.run_id}.stdout.log"
    stderr_path = log_root / f"{job.run_id}.stderr.log"
    command = _job_command(
        protocol_path=protocol_path,
        result_root=result_root,
        job=job,
        device=device,
    )
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        stdout_handle.write(
            f"\n=== resume {datetime.now(timezone.utc).isoformat()} ===\n"
        )
        stdout_handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    after = _job_status(protocol=protocol, result_root=result_root, job=job)
    success = completed.returncode == 0 and after["status"] == "complete"
    return {
        "job": job.run_id,
        "status": "completed" if success else "failed",
        "returncode": 0 if success else int(completed.returncode),
        "device": device,
        "final_state": after["status"],
        "valid_blocks": after["valid_blocks"],
        "expected_blocks": after["expected_blocks"],
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
        "command": command,
    }


def _execute_matrix(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    result_root: Path,
    profile_name: str,
    worker_plan: dict[str, int],
) -> dict[str, Any]:
    jobs = study.build_jobs(protocol, profile_name)
    slots = _device_worker_slots(worker_plan)
    if len(slots) > len(jobs):
        raise ValueError(f"workers must be between 1 and {len(jobs)}")
    job_queue: Queue[study.EvaluationJob] = Queue()
    for job in jobs:
        job_queue.put(job)

    def execute_slot(device: str, slot_index: int) -> list[dict[str, Any]]:
        slot_rows: list[dict[str, Any]] = []
        while True:
            try:
                job = job_queue.get_nowait()
            except Empty:
                break
            try:
                row = _execute_job(
                    protocol=protocol,
                    protocol_path=protocol_path,
                    result_root=result_root,
                    job=job,
                    device=device,
                )
            except Exception as exc:
                row = {
                    "job": job.run_id,
                    "status": "runner_exception",
                    "returncode": 1,
                    "device": device,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                job_queue.task_done()
            row["worker_device"] = device
            row["worker_slot"] = slot_index
            slot_rows.append(row)
            print(_json_text(row), flush=True)
        return slot_rows

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = {
            executor.submit(execute_slot, device, slot_index): (device, slot_index)
            for device, slot_index in slots
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: row["job"])
    failures = [row for row in rows if int(row["returncode"]) != 0]
    return {
        "schema_version": "independent-v2-existing-models-execution/v1",
        "profile": profile_name,
        "worker_plan": worker_plan,
        "worker_count": len(slots),
        "attempted_jobs": len(rows),
        "successful_or_skipped_jobs": len(rows) - len(failures),
        "failed_jobs": len(failures),
        "rows": rows,
    }


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _population_std(values: list[float]) -> float | None:
    return float(statistics.pstdev(values)) if values else None


def _block_metric(payload: dict[str, Any], metric: str) -> float | None:
    if metric == "mean_success_completion_time_seconds":
        value = payload.get(metric)
    else:
        value = payload["summary"].get(metric)
    return None if value is None else float(value)


def _aggregate_cell(
    *,
    protocol: dict[str, Any],
    job: study.EvaluationJob,
    source: dict[str, Any],
    blocks: list[dict[str, Any]],
    expected_blocks: int,
    episodes_per_block: int,
) -> dict[str, Any]:
    records = [record for block in blocks for record in block["episode_records"]]
    completed_episodes = len(records)
    row: dict[str, Any] = {
        "method": job.method,
        "display_label": protocol["methods"][job.method]["display_label"],
        "scenario": job.scenario,
        "status": "complete" if len(blocks) == expected_blocks else "partial",
        "training_seed": 0,
        "model_basename": source["model_basename"],
        "model_sha256": source["model_sha256"],
        "expected_test_seed_blocks": expected_blocks,
        "completed_test_seed_blocks": len(blocks),
        "missing_test_seed_blocks": expected_blocks - len(blocks),
        "episodes_per_test_seed": episodes_per_block,
        "expected_episodes": expected_blocks * episodes_per_block,
        "completed_episodes": completed_episodes,
    }
    if not records:
        for metric in BLOCK_METRICS:
            row[metric] = None
            row[f"test_seed_block_{metric}_mean"] = None
            row[f"test_seed_block_{metric}_population_std"] = None
        row["std_return"] = None
        row["std_success_completion_time_seconds"] = None
        return row

    for metric, key in (
        ("success_rate", "success"),
        ("collision_rate", "collision"),
        ("off_route_rate", "off_route"),
        ("timeout_rate", "timeout"),
    ):
        row[metric] = sum(bool(record[key]) for record in records) / completed_episodes
    returns = [float(record["episode_return"]) for record in records]
    decisions = [float(record["decision_steps"]) for record in records]
    raw_steps = [float(record["raw_steps"]) for record in records]
    completions = [
        float(record["completion_time_seconds"])
        for record in records
        if record.get("completion_time_seconds") is not None
    ]
    row["mean_return"] = _mean(returns)
    row["std_return"] = _population_std(returns)
    row["mean_decision_steps"] = _mean(decisions)
    row["mean_raw_steps"] = _mean(raw_steps)
    row["mean_success_completion_time_seconds"] = _mean(completions)
    row["std_success_completion_time_seconds"] = _population_std(completions)
    for metric in BLOCK_METRICS:
        values = [
            value
            for block in blocks
            if (value := _block_metric(block, metric)) is not None
        ]
        row[f"test_seed_block_{metric}_mean"] = _mean(values)
        row[f"test_seed_block_{metric}_population_std"] = _population_std(values)
    return row


def _macro_rows(
    protocol: dict[str, Any], method_scenario_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    expected_scenarios = len(protocol["matrix"]["scenario_order"])
    for method in protocol["matrix"]["method_order"]:
        cells = [row for row in method_scenario_rows if row["method"] == method]
        complete = len(cells) == expected_scenarios and all(
            row["status"] == "complete" for row in cells
        )
        row: dict[str, Any] = {
            "method": method,
            "display_label": protocol["methods"][method]["display_label"],
            "status": "complete" if complete else "TBD",
            "completed_scenarios": sum(cell["status"] == "complete" for cell in cells),
            "expected_scenarios": expected_scenarios,
        }
        for metric in (
            "success_rate",
            "collision_rate",
            "off_route_rate",
            "timeout_rate",
            "mean_return",
            "mean_success_completion_time_seconds",
        ):
            values = [cell.get(metric) for cell in cells]
            row[f"macro_{metric}"] = (
                _mean([float(value) for value in values if value is not None])
                if complete and all(value is not None for value in values)
                else None
            )
        output.append(row)
    return output


def _summarize(
    *,
    protocol: dict[str, Any],
    result_root: Path,
    profile_name: str,
    require_complete: bool,
) -> dict[str, Any]:
    profile = study.profile_setting(protocol, profile_name)
    indices = study.logical_test_seed_indices(profile)
    episodes = int(profile["episodes_per_test_seed"])
    block_rows: list[dict[str, Any]] = []
    method_scenario_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for job in study.build_jobs(protocol, profile_name):
        try:
            source = study.resolve_source_model(protocol, job.method, job.scenario)
        except Exception as exc:
            invalid.append(
                {"job": job.run_id, "kind": "source_model", "error": str(exc)}
            )
            source = {
                "model_basename": None,
                "model_sha256": None,
            }
            method_scenario_rows.append(
                _aggregate_cell(
                    protocol=protocol,
                    job=job,
                    source=source,
                    blocks=[],
                    expected_blocks=len(indices),
                    episodes_per_block=episodes,
                )
            )
            continue

        valid_blocks: list[dict[str, Any]] = []
        for logical_index in indices:
            path = study.block_path(result_root, job, logical_index)
            base_row = {
                "method": job.method,
                "display_label": protocol["methods"][job.method]["display_label"],
                "scenario": job.scenario,
                "logical_test_seed_index": logical_index,
                "episode_seed_start": study.episode_seed_start(profile, logical_index),
                "episodes": episodes,
                "model_sha256": source["model_sha256"],
                "block_path": str(path.resolve()),
            }
            if not path.exists():
                missing.append({**base_row, "kind": "missing_block"})
                block_rows.append({**base_row, "status": "missing"})
                continue
            try:
                block = study.read_valid_block(
                    path,
                    protocol_sha256=protocol["_sha256"],
                    job=job,
                    source_model_sha256=source["model_sha256"],
                    logical_index=logical_index,
                    expected_seed_start=study.episode_seed_start(
                        profile, logical_index
                    ),
                    expected_episodes=episodes,
                )
            except Exception as exc:
                error = {**base_row, "kind": "invalid_block", "error": str(exc)}
                invalid.append(error)
                block_rows.append({**base_row, "status": "invalid", "error": str(exc)})
                continue
            valid_blocks.append(block)
            block_rows.append(
                {
                    **base_row,
                    "status": "complete",
                    "block_sha256": study.sha256(path),
                    **{
                        metric: _block_metric(block, metric)
                        for metric in BLOCK_METRICS
                    },
                    "std_return": float(block["summary"]["std_return"]),
                    "std_success_completion_time_seconds": block[
                        "std_success_completion_time_seconds"
                    ],
                    "wall_seconds": block.get("wall_seconds"),
                }
            )
        method_scenario_rows.append(
            _aggregate_cell(
                protocol=protocol,
                job=job,
                source=source,
                blocks=valid_blocks,
                expected_blocks=len(indices),
                episodes_per_block=episodes,
            )
        )

    macro = _macro_rows(protocol, method_scenario_rows)
    expected_job_count = len(protocol["matrix"]["method_order"]) * len(
        protocol["matrix"]["scenario_order"]
    )
    complete = not missing and not invalid and all(
        row["status"] == "complete" for row in method_scenario_rows
    )
    summary_root = result_root.resolve() / profile_name / "summary"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "training_performed": False,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "profile": profile_name,
        "complete": complete,
        "expected_method_scenario_jobs": expected_job_count,
        "completed_method_scenario_jobs": sum(
            row["status"] == "complete" for row in method_scenario_rows
        ),
        "expected_test_seed_blocks": expected_job_count * len(indices),
        "completed_test_seed_blocks": sum(
            row["status"] == "complete" for row in block_rows
        ),
        "expected_evaluation_episodes": expected_job_count * len(indices) * episodes,
        "accepted_evaluation_episodes": sum(
            int(row["episodes"])
            for row in block_rows
            if row["status"] == "complete"
        ),
        "missing_blocks": missing,
        "invalid_artifacts": invalid,
        "test_seed_blocks": block_rows,
        "method_scenario": method_scenario_rows,
        "method_macro": macro,
        "aggregation_note": (
            "test_seed_block_*_population_std is variability across logical "
            "100-episode test-seed blocks; it is not cross-training-seed variability."
        ),
    }
    _write_json_atomic(summary_root / "summary.json", summary)

    block_fields = [
        "method",
        "display_label",
        "scenario",
        "logical_test_seed_index",
        "episode_seed_start",
        "episodes",
        "status",
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "std_return",
        "mean_decision_steps",
        "mean_raw_steps",
        "mean_success_completion_time_seconds",
        "std_success_completion_time_seconds",
        "wall_seconds",
        "model_sha256",
        "block_sha256",
        "block_path",
        "error",
    ]
    _write_csv_atomic(summary_root / "test_seed_blocks.csv", block_rows, block_fields)
    cell_fields = list(method_scenario_rows[0]) if method_scenario_rows else []
    _write_csv_atomic(
        summary_root / "method_scenario.csv", method_scenario_rows, cell_fields
    )
    macro_fields = list(macro[0]) if macro else []
    _write_csv_atomic(summary_root / "method_macro.csv", macro, macro_fields)

    if require_complete and not complete:
        raise RuntimeError(
            "formal evaluation is incomplete: "
            f"{len(missing)} missing blocks and {len(invalid)} invalid artifacts"
        )
    return summary


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch_background(
    *,
    protocol_path: Path,
    result_root: Path,
    worker_plan: dict[str, int],
    explicit_device_pool: bool,
    wait_for_pid: int | None,
    max_workers: int,
) -> dict[str, Any]:
    slots = _device_worker_slots(worker_plan)
    workers = len(slots)
    if workers > max_workers:
        raise ValueError(f"workers must be between 1 and {max_workers}")
    launcher_root = result_root.resolve() / "launcher"
    launcher_root.mkdir(parents=True, exist_ok=True)
    active_path = launcher_root / "active_launch.json"
    if active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        active_pid = int(active.get("pid", -1))
        if _pid_exists(active_pid):
            raise RuntimeError(
                f"an evaluation launcher is already active with PID {active_pid}"
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout_path = launcher_root / f"formal_{stamp}.stdout.log"
    stderr_path = launcher_root / f"formal_{stamp}.stderr.log"
    runner_command = "run"
    wait_identity: dict[str, Any] | None = None
    if wait_for_pid is not None:
        wait_identity = _process_identity(wait_for_pid)
        runner_command = "_queued-run"
    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        runner_command,
        "--protocol",
        str(protocol_path.resolve()),
        "--result-root",
        str(result_root.resolve()),
    ]
    if explicit_device_pool:
        command.extend(
            (
                "--cpu-workers",
                str(worker_plan.get("cpu", 0)),
                "--gpu-workers",
                str(worker_plan.get("cuda", 0)),
            )
        )
    else:
        device, homogeneous_workers = next(iter(worker_plan.items()))
        command.extend(("--device", device, "--workers", str(homogeneous_workers)))
    if wait_identity is not None:
        command.extend(
            (
                "--wait-for-pid",
                str(wait_identity["pid"]),
                "--wait-for-create-time",
                repr(wait_identity["create_time"]),
            )
        )
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            close_fds=True,
            creationflags=creationflags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    receipt = {
        "schema_version": "independent-v2-existing-models-launch/v1",
        "status": "queued" if wait_identity is not None else "started",
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path.resolve()),
        "result_root": str(result_root.resolve()),
        "device": "mixed" if len(worker_plan) > 1 else next(iter(worker_plan)),
        "workers": workers,
        "worker_plan": worker_plan,
        "wait_for_process": wait_identity,
        "command": command,
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
    }
    _write_json_atomic(active_path, receipt)
    _write_json_atomic(launcher_root / f"launch_{stamp}.json", receipt)
    return receipt


def _process_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("wait-for PID must be positive")
    import psutil

    try:
        process = psutil.Process(pid)
        return {
            "pid": pid,
            "create_time": float(process.create_time()),
            "name": process.name(),
            "executable": process.exe(),
        }
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        raise ValueError(f"wait-for process {pid} is not running") from exc


def _wait_for_exact_process(pid: int, expected_create_time: float) -> None:
    import psutil

    try:
        process = psutil.Process(pid)
        actual_create_time = float(process.create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        print(
            _json_text(
                {"queue_status": "released", "wait_for_pid": pid, "reason": "exited"}
            ),
            flush=True,
        )
        return
    if not math.isclose(
        actual_create_time,
        float(expected_create_time),
        rel_tol=0.0,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            f"PID {pid} was reused; expected create time {expected_create_time}, "
            f"observed {actual_create_time}"
        )
    while True:
        try:
            process.wait(timeout=60)
            break
        except psutil.TimeoutExpired:
            print(
                _json_text(
                    {
                        "queue_status": "waiting",
                        "wait_for_pid": pid,
                        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                flush=True,
            )
    print(
        _json_text(
            {"queue_status": "released", "wait_for_pid": pid, "reason": "exited"}
        ),
        flush=True,
    )


def _finalize_matching_launch_receipt(
    *, result_root: Path, execution: dict[str, Any]
) -> None:
    active_path = result_root.resolve() / "launcher" / "active_launch.json"
    if not active_path.is_file():
        return
    try:
        receipt = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if int(receipt.get("pid", -1)) != os.getpid():
        return
    receipt.update(
        {
            "status": (
                "completed" if int(execution["failed_jobs"]) == 0 else "failed"
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_jobs": int(execution["failed_jobs"]),
            "execution_receipt": str(
                (
                    result_root.resolve()
                    / "comparison"
                    / "execution_receipt.json"
                ).resolve()
            ),
        }
    )
    _write_json_atomic(active_path, receipt)


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    protocol = study.load_protocol(args.protocol)
    result_root = args.result_root.resolve()

    if args.command == "plan":
        print(_json_text(_plan_payload(protocol, args.profile)))
        return 0
    if args.command == "source-audit":
        payload = _source_audit(protocol)
        audit_path = result_root / "source_audit.json"
        _write_json_atomic(audit_path, payload)
        payload["audit_path"] = str(audit_path.resolve())
        print(_json_text(payload))
        return 0 if payload["complete"] else 1
    if args.command == "status":
        print(_json_text(_status_payload(protocol, result_root, args.profile)))
        return 0
    if args.command == "summarize":
        try:
            payload = _summarize(
                protocol=protocol,
                result_root=result_root,
                profile_name=args.profile,
                require_complete=args.require_complete,
            )
        except RuntimeError as exc:
            print(_json_text({"status": "incomplete", "error": str(exc)}))
            return 1
        print(
            _json_text(
                {
                    "status": "complete" if payload["complete"] else "partial",
                    "summary_path": str(
                        result_root / args.profile / "summary" / "summary.json"
                    ),
                    "completed_test_seed_blocks": payload[
                        "completed_test_seed_blocks"
                    ],
                    "expected_test_seed_blocks": payload[
                        "expected_test_seed_blocks"
                    ],
                    "accepted_evaluation_episodes": payload[
                        "accepted_evaluation_episodes"
                    ],
                    "expected_evaluation_episodes": payload[
                        "expected_evaluation_episodes"
                    ],
                }
            )
        )
        return 0
    if args.command == "launch":
        worker_plan = _device_worker_plan(
            device=args.device,
            workers=args.workers,
            cpu_workers=args.cpu_workers,
            gpu_workers=args.gpu_workers,
        )
        print(
            _json_text(
                _launch_background(
                    protocol_path=Path(protocol["_path"]),
                    result_root=result_root,
                    worker_plan=worker_plan,
                    explicit_device_pool=bool(args.cpu_workers or args.gpu_workers),
                    wait_for_pid=args.wait_for_pid,
                    max_workers=len(study.build_jobs(protocol, "comparison")),
                )
            )
        )
        return 0

    if args.command == "_queued-run":
        _wait_for_exact_process(args.wait_for_pid, args.wait_for_create_time)

    profile_name = "smoke" if args.command == "preflight" else "comparison"
    worker_plan = _device_worker_plan(
        device=args.device,
        workers=args.workers,
        cpu_workers=args.cpu_workers,
        gpu_workers=args.gpu_workers,
    )
    worker_count = len(_device_worker_slots(worker_plan))
    thread_limits: dict[str, str] = {}
    if worker_count > 1:
        thread_limits = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        os.environ.update(thread_limits)
    result = _execute_matrix(
        protocol=protocol,
        protocol_path=Path(protocol["_path"]),
        result_root=result_root,
        profile_name=profile_name,
        worker_plan=worker_plan,
    )
    result["worker_thread_limits"] = thread_limits
    receipt_path = result_root / profile_name / "execution_receipt.json"
    result["execution_receipt"] = str(receipt_path.resolve())
    if profile_name == "comparison" and result["failed_jobs"] == 0:
        summary = _summarize(
            protocol=protocol,
            result_root=result_root,
            profile_name=profile_name,
            require_complete=True,
        )
        result["summary_path"] = str(
            (result_root / profile_name / "summary" / "summary.json").resolve()
        )
        result["accepted_evaluation_episodes"] = summary[
            "accepted_evaluation_episodes"
        ]
    _write_json_atomic(receipt_path, result)
    if profile_name == "comparison":
        _finalize_matching_launch_receipt(
            result_root=result_root,
            execution=result,
        )
    print(_json_text(result))
    return 0 if result["failed_jobs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "DEFAULT_RESULT_ROOT",
    "_aggregate_cell",
    "_device_worker_plan",
    "_device_worker_slots",
    "_macro_rows",
    "_plan_payload",
    "_source_audit",
    "_status_payload",
    "_summarize",
    "main",
]
