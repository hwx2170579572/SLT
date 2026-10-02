"""Plan, adopt, preflight, resume, and summarize the independent-v2 5x6 study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.independent_v2_5m6s100e_v1_common import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RESULT_ROOT,
    METHOD_ADAPTERS,
    PARENT_METHODS,
    PARENT_SCENARIOS,
    PROJECT_ROOT,
    RECEIPT_NAME,
    RUN_ID_PREFIX,
    MatrixCell,
    assert_bound_file,
    build_cells,
    build_fresh_cells,
    load_protocol,
    profile_setting,
    run_id,
    sha256,
)


TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_independent_v2_5m6s100e_v1.py"
PARENT_RECEIPT_NAME = "high_density_job_receipt.json"
PRIMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "timeout_rate",
    "mean_return",
    "mean_success_completion_time_seconds",
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".j{os.getpid()}.tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    os.replace(temporary, path)


def _common_subparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print all 30 cells and execution source")
    _common_subparser(plan)

    adopt = subparsers.add_parser(
        "adopt", help="Hash-audit and reference the 12 completed parent-v2 cells"
    )
    _common_subparser(adopt)

    preflight = subparsers.add_parser(
        "preflight", help="Run real-SUMO check-only validation for the 18 fresh cells"
    )
    _common_subparser(preflight)
    preflight.add_argument("--device", default="cpu")
    preflight.add_argument("--workers", type=int, default=1)

    run = subparsers.add_parser(
        "run", help="Execute/resume only the 18 missing comparison cells"
    )
    _common_subparser(run)
    run.add_argument("--device", default="auto")
    run.add_argument("--workers", type=int, default=1)

    launch = subparsers.add_parser(
        "launch", help="Start the resumable comparison controller in background"
    )
    _common_subparser(launch)
    launch.add_argument("--device", default="cuda")
    launch.add_argument("--workers", type=int, default=1)

    status = subparsers.add_parser("status", help="Audit adopted and fresh cell state")
    _common_subparser(status)

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate only hash-validated real evaluation artifacts"
    )
    _common_subparser(summarize)
    summarize.add_argument("--require-complete", action="store_true")
    return parser


def _profile_root(result_root: Path, profile: str) -> Path:
    return result_root.resolve() / profile


def _fresh_run_dir(result_root: Path, cell: MatrixCell, *, check_only: bool) -> Path:
    folder = "preflight_runs" if check_only else "runs"
    return _profile_root(result_root, cell.profile) / folder / cell.run_id


def _required_fresh_artifacts(check_only: bool) -> tuple[str, ...]:
    if check_only:
        return ("arguments.json", "check_result.json", RECEIPT_NAME)
    return (
        "arguments.json",
        "final_model.zip",
        "paper_evaluation_detailed.json",
        RECEIPT_NAME,
    )


def _fresh_state(
    protocol: dict[str, Any],
    result_root: Path,
    cell: MatrixCell,
    *,
    check_only: bool,
) -> tuple[str, str | None]:
    run_dir = _fresh_run_dir(result_root, cell, check_only=check_only)
    if not run_dir.exists():
        return "missing", None
    required = _required_fresh_artifacts(check_only)
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return "incomplete", f"missing artifacts: {missing}"
    try:
        receipt = json.loads((run_dir / RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "incomplete", f"invalid receipt: {exc}"
    expected_status = "preflight_passed" if check_only else "completed"
    expected = {
        "status": expected_status,
        "fabricated_values": False,
        "method": cell.method,
        "scenario": cell.scenario,
        "seed": cell.seed,
        "profile": cell.profile,
        "protocol_sha256": protocol["_sha256"],
        "formal_test_accessed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            return "incomplete", f"receipt {key} mismatch"
    artifact_hashes = receipt.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        return "incomplete", "receipt artifact hashes are missing"
    for name in required:
        if name == RECEIPT_NAME:
            continue
        if artifact_hashes.get(name) != sha256(run_dir / name):
            return "incomplete", f"artifact SHA-256 mismatch: {name}"
    if not check_only:
        try:
            detailed = json.loads(
                (run_dir / "paper_evaluation_detailed.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            return "incomplete", f"invalid detailed evaluation: {exc}"
        records = detailed.get("episode_records")
        if not isinstance(records, list) or len(records) != 100:
            return "incomplete", "detailed evaluation must contain 100 records"
        seeds = [row.get("seed") for row in records]
        if seeds != list(range(10_000, 10_100)):
            return "incomplete", "evaluation seed sequence mismatch"
        if detailed.get("evaluation_split") == "test" or detailed.get(
            "formal_test_accessed"
        ) is True:
            return "incomplete", "formal/test evidence was accessed"
    return "complete", None


def _parent_run_dir(protocol: dict[str, Any], method: str, scenario: str) -> Path:
    parent = protocol["parent_v2_reuse"]
    run_name = str(parent["run_id_template"]).format(
        method=method, scenario=scenario
    )
    return (
        PROJECT_ROOT
        / str(parent["result_root"])
        / str(parent["profile"])
        / "runs"
        / run_name
    ).resolve()


def _audit_parent(protocol: dict[str, Any]) -> dict[str, Any]:
    parent_protocol_path = assert_bound_file(
        protocol, "protocol_path", "protocol_sha256", "parent_v2_reuse"
    )
    parent_summary_path = assert_bound_file(
        protocol, "summary_path", "summary_sha256", "parent_v2_reuse"
    )
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for method in PARENT_METHODS:
        for scenario in PARENT_SCENARIOS:
            run_dir = _parent_run_dir(protocol, method, scenario)
            files = {
                name: run_dir / name
                for name in (
                    "arguments.json",
                    "final_model.zip",
                    "paper_evaluation_detailed.json",
                    PARENT_RECEIPT_NAME,
                )
            }
            missing = [name for name, path in files.items() if not path.is_file()]
            row: dict[str, Any] = {
                "method": method,
                "scenario": scenario,
                "seed": 0,
                "source_run_directory": str(run_dir),
            }
            if missing:
                row.update(status="invalid", reason=f"missing artifacts: {missing}")
                rows.append(row)
                errors.append(f"{method}/{scenario}: missing {missing}")
                continue
            try:
                receipt = json.loads(files[PARENT_RECEIPT_NAME].read_text(encoding="utf-8"))
                detailed = json.loads(
                    files["paper_evaluation_detailed.json"].read_text(encoding="utf-8")
                )
                expected_receipt = {
                    "schema_version": "high-density-single-seed-100ep-job-receipt/v2",
                    "status": "completed",
                    "fabricated_values": False,
                    "method": method,
                    "scenario": scenario,
                    "seed": 0,
                    "profile": "comparison",
                    "protocol_sha256": protocol["parent_v2_reuse"]["protocol_sha256"],
                }
                for key, value in expected_receipt.items():
                    if receipt.get(key) != value:
                        raise ValueError(f"receipt {key} mismatch")
                density = protocol["scenarios"][scenario]
                if float(receipt["vehicle_scale"]) != float(density["vehicle_scale"]):
                    raise ValueError("vehicle scale mismatch")
                if float(receipt["pedestrian_scale"]) != float(
                    density["pedestrian_scale"]
                ):
                    raise ValueError("pedestrian scale mismatch")
                if list(receipt["clone_depart_jitter_seconds"]) != list(
                    density["clone_depart_jitter_seconds"]
                ):
                    raise ValueError("clone jitter mismatch")
                receipt_hashes = receipt.get("artifact_sha256", {})
                for name in (
                    "arguments.json",
                    "final_model.zip",
                    "paper_evaluation_detailed.json",
                ):
                    if receipt_hashes.get(name) != sha256(files[name]):
                        raise ValueError(f"artifact SHA-256 mismatch: {name}")
                records = detailed.get("episode_records")
                if not isinstance(records, list) or len(records) != 100:
                    raise ValueError("expected 100 detailed episode records")
                if [record.get("seed") for record in records] != list(
                    range(10_000, 10_100)
                ):
                    raise ValueError("evaluation seed sequence mismatch")
                if int(detailed.get("summary", {}).get("episodes", -1)) != 100:
                    raise ValueError("summary episode count mismatch")
                row.update(
                    status="validated",
                    evaluation_episodes=100,
                    artifact_sha256={name: sha256(path) for name, path in files.items()},
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                row.update(status="invalid", reason=str(exc))
                errors.append(f"{method}/{scenario}: {exc}")
            rows.append(row)
    return {
        "schema_version": "independent-v2-parent-adoption-audit/v1",
        "new_protocol_path": protocol["_path"],
        "new_protocol_sha256": protocol["_sha256"],
        "parent_protocol_path": str(parent_protocol_path),
        "parent_protocol_sha256": sha256(parent_protocol_path),
        "parent_summary_path": str(parent_summary_path),
        "parent_summary_sha256": sha256(parent_summary_path),
        "expected_cells": 12,
        "validated_cells": sum(row["status"] == "validated" for row in rows),
        "complete": not errors and len(rows) == 12,
        "errors": errors,
        "training_or_evaluation_rerun": False,
        "rows": rows,
    }


def _adoption_path(protocol: dict[str, Any], result_root: Path) -> Path:
    return result_root.resolve() / str(
        protocol["artifact_contract"]["adoption_receipt"]
    )


def _write_adoption(protocol: dict[str, Any], result_root: Path) -> dict[str, Any]:
    audit = _audit_parent(protocol)
    if not audit["complete"]:
        raise RuntimeError(f"parent adoption audit failed: {audit['errors']}")
    _write_json_atomic(_adoption_path(protocol, result_root), audit)
    return audit


def _preflight_cells(protocol: dict[str, Any]) -> list[MatrixCell]:
    comparison_fresh = build_fresh_cells(protocol, "comparison")
    return [
        MatrixCell(
            profile="smoke",
            method=cell.method,
            scenario=cell.scenario,
            seed=cell.seed,
            run_id=(
                f"{RUN_ID_PREFIX}__preflight__{cell.method}__"
                f"{cell.scenario}__seed{cell.seed}"
            ),
            execution="fresh",
        )
        for cell in comparison_fresh
    ]


def _job_command(
    protocol: dict[str, Any],
    result_root: Path,
    cell: MatrixCell,
    *,
    device: str,
    check_only: bool,
) -> list[str]:
    output_dir = _profile_root(result_root, cell.profile) / (
        "preflight_runs" if check_only else "runs"
    )
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--method",
        cell.method,
        "--scenario",
        cell.scenario,
        "--seed",
        str(cell.seed),
        "--profile",
        cell.profile,
        "--protocol",
        protocol["_path"],
        "--output-dir",
        str(output_dir.resolve()),
        "--model-name",
        cell.run_id,
        "--device",
        device,
    ]
    if check_only:
        command.append("--check-only")
    return command


def _execute_one(
    protocol: dict[str, Any],
    result_root: Path,
    cell: MatrixCell,
    *,
    device: str,
    check_only: bool,
) -> dict[str, Any]:
    run_dir = _fresh_run_dir(result_root, cell, check_only=check_only)
    state, reason = _fresh_state(
        protocol, result_root, cell, check_only=check_only
    )
    if state == "complete":
        return {"job": cell.run_id, "status": "skipped_complete", "returncode": 0}
    if state == "incomplete":
        return {
            "job": cell.run_id,
            "status": "refused_incomplete_existing_directory",
            "returncode": 2,
            "reason": reason,
        }
    log_root = _profile_root(result_root, cell.profile) / (
        "preflight_logs" if check_only else "logs"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    command = _job_command(
        protocol, result_root, cell, device=device, check_only=check_only
    )
    stdout_path = log_root / f"{cell.run_id}.stdout.log"
    stderr_path = log_root / f"{cell.run_id}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    final_state, final_reason = _fresh_state(
        protocol, result_root, cell, check_only=check_only
    )
    success = completed.returncode == 0 and final_state == "complete"
    return {
        "job": cell.run_id,
        "status": "completed" if success else "failed",
        "returncode": 0 if success else int(completed.returncode or 1),
        "artifact_state": final_state,
        "reason": final_reason,
        "run_directory": str(run_dir),
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
        "command": command,
    }


def _execute_matrix(
    protocol: dict[str, Any],
    result_root: Path,
    cells: list[MatrixCell],
    *,
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
                _execute_one,
                protocol,
                result_root,
                cell,
                device=device,
                check_only=check_only,
            ): cell
            for cell in cells
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(_json_text(row), flush=True)
    rows.sort(key=lambda row: row["job"])
    failures = [row for row in rows if int(row["returncode"]) != 0]
    payload = {
        "schema_version": "independent-v2-5m6s100e-execution/v1",
        "protocol_sha256": protocol["_sha256"],
        "check_only": check_only,
        "fresh_jobs_only": True,
        "attempted_jobs": len(rows),
        "successful_or_skipped_jobs": len(rows) - len(failures),
        "failed_jobs": len(failures),
        "rows": rows,
    }
    root = _profile_root(result_root, cells[0].profile if cells else "comparison")
    _write_json_atomic(
        root / ("last_preflight.json" if check_only else "last_execution.json"),
        payload,
    )
    return payload


def _process_is_matching_runner(pid: int) -> bool:
    try:
        import psutil

        process = psutil.Process(int(pid))
        command = " ".join(process.cmdline()).lower()
        return process.is_running() and TRAIN_SCRIPT.name.lower().replace(
            "train_", "run_"
        ) in command and " run " in f" {command} "
    except Exception:
        return False


@contextmanager
def _execution_lock(result_root: Path, name: str):
    controller_root = result_root.resolve() / "comparison" / "controller"
    controller_root.mkdir(parents=True, exist_ok=True)
    lock_path = controller_root / f"{name}.lock.json"
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            existing_pid = -1
        if existing_pid >= 0 and _process_is_matching_runner(existing_pid):
            raise RuntimeError(
                f"{name} controller is already running as PID {existing_pid}"
            )
        stale = lock_path.with_name(f"{lock_path.stem}.stale_pid{existing_pid}.json")
        os.replace(lock_path, stale)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(
            descriptor,
            _json_text(
                {
                    "schema_version": "independent-v2-controller-lock/v1",
                    "pid": os.getpid(),
                    "controller": name,
                }
            ).encode("utf-8"),
        )
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


def _launch_background(
    protocol: dict[str, Any],
    result_root: Path,
    *,
    device: str,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    _write_adoption(protocol, result_root)
    status = _status_payload(protocol, result_root)
    if status["counts"]["incomplete"]:
        raise RuntimeError("refusing launch while incomplete run directories exist")
    controller_root = result_root.resolve() / "comparison" / "controller"
    controller_root.mkdir(parents=True, exist_ok=True)
    active_path = controller_root / "active_launcher.json"
    if active_path.is_file():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            active_pid = int(active["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            active_pid = -1
        if active_pid >= 0 and _process_is_matching_runner(active_pid):
            active["already_running"] = True
            return active
    if status["counts"]["fresh_missing"] == 0:
        return {
            "schema_version": "independent-v2-background-launch/v1",
            "status": "nothing_to_launch",
            "already_complete": True,
            "protocol_sha256": protocol["_sha256"],
        }

    stdout_path = controller_root / "controller.stdout.log"
    stderr_path = controller_root / "controller.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--protocol",
        protocol["_path"],
        "--result-root",
        str(result_root.resolve()),
        "--device",
        device,
        "--workers",
        str(workers),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
            getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    receipt = {
        "schema_version": "independent-v2-background-launch/v1",
        "status": "launched",
        "already_running": False,
        "pid": int(process.pid),
        "protocol_sha256": protocol["_sha256"],
        "device": device,
        "workers": workers,
        "fresh_cells_at_launch": status["counts"]["fresh_missing"],
        "adopted_cells_at_launch": status["counts"]["adopted_complete"],
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
        "command": command,
    }
    _write_json_atomic(active_path, receipt)
    _write_json_atomic(controller_root / f"launch_pid{process.pid}.json", receipt)
    return receipt


def _status_payload(protocol: dict[str, Any], result_root: Path) -> dict[str, Any]:
    adoption = _audit_parent(protocol)
    adopted_lookup = {
        (row["method"], row["scenario"]): row for row in adoption["rows"]
    }
    active_run_ids = _active_fresh_run_ids(result_root)
    rows: list[dict[str, Any]] = []
    for cell in build_cells(protocol, "comparison"):
        if cell.execution == "adopt_parent_v2":
            source = adopted_lookup[(cell.method, cell.scenario)]
            state = "adopted_complete" if source["status"] == "validated" else "incomplete"
            rows.append(
                {
                    **asdict(cell),
                    "status": state,
                    "reason": source.get("reason"),
                    "run_directory": source["source_run_directory"],
                }
            )
        else:
            state, reason = _fresh_state(
                protocol, result_root, cell, check_only=False
            )
            if cell.run_id in active_run_ids and state in {"missing", "incomplete"}:
                display_state = "fresh_running"
                reason = None
            else:
                display_state = (
                    f"fresh_{state}" if state != "incomplete" else state
                )
            rows.append(
                {
                    **asdict(cell),
                    "status": display_state,
                    "reason": reason,
                    "run_directory": str(
                        _fresh_run_dir(result_root, cell, check_only=False)
                    ),
                }
            )
    possible = (
        "adopted_complete",
        "fresh_complete",
        "fresh_running",
        "fresh_missing",
        "incomplete",
    )
    counts = {name: sum(row["status"] == name for row in rows) for name in possible}
    return {
        "schema_version": "independent-v2-5m6s100e-status/v1",
        "protocol_sha256": protocol["_sha256"],
        "expected_cells": 30,
        "adoption_audit_complete": adoption["complete"],
        "counts": counts,
        "complete": counts["adopted_complete"] == 12
        and counts["fresh_complete"] == 18,
        "rows": rows,
    }


def _active_fresh_run_ids(result_root: Path) -> set[str]:
    active_path = (
        result_root.resolve()
        / "comparison"
        / "controller"
        / "active_launcher.json"
    )
    if not active_path.is_file():
        return set()
    try:
        payload = json.loads(active_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        if not _process_is_matching_runner(pid):
            return set()
        import psutil

        output: set[str] = set()
        for process in psutil.Process(pid).children(recursive=True):
            try:
                command = process.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if "--model-name" not in command:
                continue
            index = command.index("--model-name") + 1
            if index < len(command):
                output.add(str(command[index]))
        return output
    except Exception:
        return set()


def _metric(summary: dict[str, Any], detailed: dict[str, Any], name: str) -> float | None:
    value = (
        detailed.get(name)
        if name == "mean_success_completion_time_seconds"
        else summary.get(name)
    )
    return None if value is None else float(value)


def _result_row(
    protocol: dict[str, Any],
    cell: MatrixCell,
    run_dir: Path,
    *,
    source: str,
) -> dict[str, Any]:
    detailed_path = run_dir / "paper_evaluation_detailed.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    summary = detailed["summary"]
    return {
        "method": cell.method,
        "display_label": protocol["methods"][cell.method]["display_label"],
        "scenario": cell.scenario,
        "seed": cell.seed,
        "source": source,
        "status": "complete",
        "evaluation_episodes": int(summary["episodes"]),
        "run_directory": str(run_dir.resolve()),
        "detailed_evaluation_sha256": sha256(detailed_path),
        **{
            metric: _metric(summary, detailed, metric) for metric in PRIMARY_METRICS
        },
    }


def _summarize(
    protocol: dict[str, Any],
    result_root: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    adoption = _audit_parent(protocol)
    adopted_lookup = {
        (row["method"], row["scenario"]): row for row in adoption["rows"]
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in build_cells(protocol, "comparison"):
        if cell.execution == "adopt_parent_v2":
            adopted = adopted_lookup[(cell.method, cell.scenario)]
            if adopted["status"] == "validated":
                rows.append(
                    _result_row(
                        protocol,
                        cell,
                        Path(adopted["source_run_directory"]),
                        source="adopted_parent_v2",
                    )
                )
            else:
                missing.append(cell.run_id)
        else:
            state, _ = _fresh_state(protocol, result_root, cell, check_only=False)
            if state == "complete":
                rows.append(
                    _result_row(
                        protocol,
                        cell,
                        _fresh_run_dir(result_root, cell, check_only=False),
                        source="fresh_i5m6s100_v1",
                    )
                )
            else:
                missing.append(cell.run_id)
    if require_complete and missing:
        raise RuntimeError(f"comparison incomplete; {len(missing)} cells are unavailable")

    complete_lookup = {(row["method"], row["scenario"]): row for row in rows}
    method_scenario: list[dict[str, Any]] = []
    for cell in build_cells(protocol, "comparison"):
        row = complete_lookup.get((cell.method, cell.scenario))
        if row is None:
            method_scenario.append(
                {
                    "method": cell.method,
                    "display_label": protocol["methods"][cell.method]["display_label"],
                    "scenario": cell.scenario,
                    "seed": cell.seed,
                    "source": cell.execution,
                    "status": "TBD",
                    "evaluation_episodes": None,
                    "run_directory": None,
                    "detailed_evaluation_sha256": None,
                    **{metric: None for metric in PRIMARY_METRICS},
                }
            )
        else:
            method_scenario.append(row)

    macro: list[dict[str, Any]] = []
    for method in protocol["matrix"]["method_order"]:
        method_rows = [row for row in rows if row["method"] == method]
        complete = len(method_rows) == 6
        aggregate: dict[str, Any] = {
            "method": method,
            "display_label": protocol["methods"][method]["display_label"],
            "status": "complete" if complete else "TBD",
            "completed_scenarios": len(method_rows),
            "expected_scenarios": 6,
            "training_seed_count": 1,
            "cross_training_seed_std": None,
            "significance_claimed": False,
        }
        for metric in PRIMARY_METRICS:
            values = [row[metric] for row in method_rows]
            aggregate[f"macro_{metric}"] = (
                statistics.fmean(float(value) for value in values)
                if complete and all(value is not None for value in values)
                else None
            )
        macro.append(aggregate)

    payload = {
        "schema_version": "independent-v2-5m6s100e-summary/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "expected_method_scenario_cells": 30,
        "accepted_method_scenario_cells": len(rows),
        "expected_evaluation_episodes": 3000,
        "accepted_evaluation_episodes": sum(
            int(row["evaluation_episodes"]) for row in rows
        ),
        "complete": not missing,
        "missing_cells": missing,
        "single_training_seed_warning": (
            "Descriptive comparison only; no cross-training-seed variance or "
            "significance claim is available."
        ),
        "method_scenario": method_scenario,
        "method_macro": macro,
    }
    summary_root = result_root.resolve() / "comparison" / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(summary_root / "summary.json", payload)
    _write_csv(summary_root / "method_scenario.csv", method_scenario)
    _write_csv(summary_root / "method_macro.csv", macro)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plan(protocol: dict[str, Any]) -> dict[str, Any]:
    cells = build_cells(protocol, "comparison")
    return {
        "schema_version": "independent-v2-5m6s100e-plan/v1",
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "methods": list(METHOD_ADAPTERS),
        "scenarios": list(protocol["matrix"]["scenario_order"]),
        "density": {
            scenario: {
                "vehicle_scale": protocol["scenarios"][scenario]["vehicle_scale"],
                "pedestrian_scale": protocol["scenarios"][scenario][
                    "pedestrian_scale"
                ],
                "clone_depart_jitter_seconds": protocol["scenarios"][scenario][
                    "clone_depart_jitter_seconds"
                ],
            }
            for scenario in protocol["matrix"]["scenario_order"]
        },
        "method_scenario_cells": len(cells),
        "adopted_parent_cells": sum(
            cell.execution == "adopt_parent_v2" for cell in cells
        ),
        "fresh_cells": sum(cell.execution == "fresh" for cell in cells),
        "evaluation_episodes_per_cell": 100,
        "total_evaluation_episodes": 3000,
        "cells": [asdict(cell) for cell in cells],
    }


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    result_root = args.result_root.resolve()
    if args.command == "plan":
        print(_json_text(_plan(protocol)))
        return 0
    if args.command == "adopt":
        print(_json_text(_write_adoption(protocol, result_root)))
        return 0
    if args.command == "status":
        print(_json_text(_status_payload(protocol, result_root)))
        return 0
    if args.command == "summarize":
        print(
            _json_text(
                _summarize(
                    protocol,
                    result_root,
                    require_complete=args.require_complete,
                )
            )
        )
        return 0
    if args.command == "launch":
        print(
            _json_text(
                _launch_background(
                    protocol,
                    result_root,
                    device=args.device,
                    workers=args.workers,
                )
            )
        )
        return 0

    if args.command == "preflight":
        cells = _preflight_cells(protocol)
        check_only = True
    else:
        _write_adoption(protocol, result_root)
        cells = build_fresh_cells(protocol, "comparison")
        check_only = False
    with _execution_lock(
        result_root, "preflight" if check_only else "comparison"
    ):
        payload = _execute_matrix(
            protocol,
            result_root,
            cells,
            device=args.device,
            workers=args.workers,
            check_only=check_only,
        )
    print(_json_text(payload))
    return 0 if payload["failed_jobs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
