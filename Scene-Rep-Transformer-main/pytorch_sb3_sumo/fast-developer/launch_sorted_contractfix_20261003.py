"""Guarded fresh-run launcher for the repaired D1 ST and ST-RT methods.

Default and --check-only/--preview are read-only. --launch remains blocked
until the two valid predecessor arms have final 100-episode evaluations with
matching checkpoint identities. The old pair's aggregate launcher status is
intentionally not used because one sibling candidate had failed independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import time
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT.parent.parent
RUNS = WORKSPACE / "runs"
METHODS = (
    "sac_mlp_d1_st_contractfix_v1",
    "sac_mlp_d1_st_rt_contractfix_v1",
)
WORKER_ROOT_NAMES = {
    METHODS[0]: "st",
    METHODS[1]: "st_rt",
}
OLD_ARMS = {
    "sac_mlp_d1_st_rt": RUNS / "sortlr_1003_retry01",
    "sac_mlp_d1_st_rt_longres_v1": RUNS / "sortlr_1003_retry01_longres",
}
DEFAULT_ROOT = RUNS / "d1_contractfix_20261003"
PAIR_RECEIPT = RUNS / "d1_contractfix_20261003_pair_launch_receipt.json"
OLD_METHOD_DIRS = {
    "sac_mlp_d1_st_rt": "sac_mlp_d1_st_rt__intersection_sorted_depart4p0",
    "sac_mlp_d1_st_rt_longres_v1": (
        "sac_mlp_d1_st_rt_longres_v1__intersection_sorted_depart4p0"
    ),
}
SOURCE_FILES = (
    HERE / "train_intersection_yield_v2_d1_contractfix.py",
    HERE / "launch_sorted_contractfix_20261003.py",
    HERE / "train_intersection_yield_v2_d1.py",
    HERE / "train_intersection_yield_v2.py",
    HERE / "behavior_diagnostics.py",
    HERE / "contractfix_trajectory_audit.py",
    HERE / "reward_shaping_v2.py",
    HERE / "analysis" / "d1_contractfix_protocol_20261003.md",
    PROJECT / "algos" / "sb3_torch" / "contractfix_encoder.py",
    PROJECT / "algos" / "sb3_torch" / "incremental_topo_encoder.py",
    PROJECT / "algos" / "sb3_torch" / "topo_temporal_features.py",
    PROJECT / "algos" / "sb3_torch" / "topo_temporal_features_v2.py",
    PROJECT / "algos" / "sb3_torch" / "features.py",
    PROJECT / "algos" / "sb3_torch" / "policy_shadow_probes.py",
    PROJECT / "algos" / "sb3_torch" / "sac.py",
    PROJECT / "algos" / "sb3_torch" / "replay_buffer.py",
    PROJECT / "algos" / "sb3_torch" / "evaluation.py",
    PROJECT / "envs" / "sumo" / "sumo_env.py",
    PROJECT / "tests_sb3_sumo" / "test_contractfix_entrypoint_roundtrip.py",
    PROJECT / "tests_sb3_sumo" / "test_contractfix_launch_gate.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _runner_python() -> Path:
    manifests = [read_json(root / "suite_manifest.json") for root in OLD_ARMS.values()]
    candidates = [item.get("python") for item in manifests if isinstance(item, dict)]
    if len(candidates) == len(OLD_ARMS) and len(set(candidates)) == 1:
        candidate = Path(candidates[0]).expanduser().resolve()
        if candidate.is_file():
            return candidate
    return Path(sys.executable).resolve()


def _probe_cuda(executable: Path) -> dict:
    probe = (
        "import json, torch; "
        "print(json.dumps({'available': bool(torch.cuda.is_available()), "
        "'count': int(torch.cuda.device_count()), 'gpu': "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'torch': str(torch.__version__)}))"
    )
    completed = subprocess.run(
        [str(executable), "-c", probe],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"CUDA probe returned no result for {executable}")
    result = json.loads(lines[-1])
    if not result.get("available") or result.get("count", 0) < 1:
        raise RuntimeError(f"CUDA unavailable in verified runner Python {executable}")
    return result


def _verify_old_arm(method: str, root: Path) -> dict:
    run_dir = root / OLD_METHOD_DIRS[method]
    reasons = []
    status = read_json(run_dir / "status.json")
    complete = read_json(run_dir / "training_complete.json")
    arguments = read_json(run_dir / "arguments.json")
    evaluation = read_json(run_dir / "evaluation_results.json")
    suite = read_json(root / "suite_manifest.json")
    checkpoint = run_dir / "final_model.zip"
    checkpoint_sha = digest(checkpoint) if checkpoint.is_file() else None

    if not isinstance(status, dict) or status.get("status") != "trained":
        reasons.append(f"status_not_trained:{(status or {}).get('status')}")
    if not isinstance(complete, dict):
        reasons.append("missing_training_complete")
    else:
        if complete.get("raw_steps") != 100000 or complete.get("smoke") is not False:
            reasons.append("training_not_fresh_100k_non_smoke")
        if not checkpoint_sha or complete.get("checkpoint_sha256") != checkpoint_sha:
            reasons.append("training_checkpoint_sha_mismatch")
    if not isinstance(arguments, dict):
        reasons.append("missing_arguments")
    else:
        if arguments.get("method") != method:
            reasons.append("arguments_method_mismatch")
        if arguments.get("scenario") != "intersection_sorted":
            reasons.append("arguments_scenario_mismatch")
        if arguments.get("raw_budget") != 100000 or arguments.get("seed") != 0:
            reasons.append("arguments_budget_or_seed_mismatch")
        if arguments.get("smoke") is not False:
            reasons.append("arguments_smoke_not_false")
        for key in ("resume", "resume_from", "continue_from", "model_path", "checkpoint_path"):
            if key in arguments and arguments[key] not in (None, False, ""):
                reasons.append(f"arguments_indicate_resume_or_checkpoint_input:{key}")

    if not isinstance(suite, dict):
        reasons.append("missing_suite_launch_manifest")
    else:
        if suite.get("fresh") is not True or suite.get("resume") is not False:
            reasons.append("suite_does_not_prove_fresh_no_resume")
        if (suite.get("training_seed") != 0
                or suite.get("raw_step_budget_per_method") != 100000
                or suite.get("smoke") is not False):
            reasons.append("suite_seed_budget_or_smoke_mismatch")
        suite_evaluation = suite.get("evaluation") or {}
        if suite_evaluation != {
            "split": "validation", "episodes": 100, "seed_start": 10000,
            "seed_end": 10099, "checkpoint": "final_model.zip",
        }:
            reasons.append("suite_final_evaluation_contract_mismatch")
        suite_python = suite.get("python")
        if not suite_python or not Path(suite_python).expanduser().is_file():
            reasons.append("suite_runner_python_missing")
        matching_commands = []
        for argv in suite.get("commands", []):
            if not isinstance(argv, list):
                continue
            try:
                method_index = argv.index("--method")
            except ValueError:
                continue
            if method_index + 1 < len(argv) and argv[method_index + 1] == method:
                matching_commands.append(argv)
        if len(matching_commands) != 1:
            reasons.append("suite_method_command_missing_or_duplicated")
        else:
            argv = matching_commands[0]
            forbidden = {"--resume", "--resume-from", "--continue-from", "--model-path", "--checkpoint-path"}
            if forbidden.intersection(argv):
                reasons.append("suite_method_command_has_resume_or_checkpoint_input")
            for flag, expected in (("--max-steps", "100000"), ("--checkpoint-frequency", "10000")):
                try:
                    actual = argv[argv.index(flag) + 1]
                except (ValueError, IndexError):
                    actual = None
                if actual != expected:
                    reasons.append(f"suite_method_command_{flag[2:].replace('-', '_')}_mismatch")

    if not isinstance(evaluation, dict):
        reasons.append("missing_evaluation_results")
        identity = {}
        episode_records = []
    else:
        identity = evaluation.get("identity") or {}
        episode_records = evaluation.get("episode_records") or []
        if identity.get("method") != method:
            reasons.append("evaluation_method_mismatch")
        if identity.get("scenario") != "intersection_sorted":
            reasons.append("evaluation_scenario_mismatch")
        if identity.get("depart_scale") != 4.0:
            reasons.append("evaluation_depart_scale_mismatch")
        if identity.get("eval_traffic_split") != "validation":
            reasons.append("evaluation_split_mismatch")
        if identity.get("episodes") != 100 or identity.get("smoke") is not False:
            reasons.append("evaluation_not_final_100_non_smoke")
        if not checkpoint_sha or identity.get("checkpoint_sha256") != checkpoint_sha:
            reasons.append("evaluation_checkpoint_sha_mismatch")
        identity_checkpoint = identity.get("checkpoint")
        if not identity_checkpoint or Path(identity_checkpoint).resolve() != checkpoint.resolve():
            reasons.append("evaluation_checkpoint_path_mismatch")
        seeds = [row.get("seed") for row in episode_records if isinstance(row, dict)]
        expected_seeds = list(range(10000, 10100))
        if len(episode_records) != 100 or seeds != expected_seeds:
            reasons.append("evaluation_seed_records_not_10000_10099")
        summary = evaluation.get("summary") or {}
        if summary.get("episodes") != 100:
            reasons.append("evaluation_summary_episode_count_mismatch")

    return {
        "method": method,
        "run_dir": str(run_dir),
        "ready": not reasons,
        "reasons": reasons,
        "status": status.get("status") if isinstance(status, dict) else None,
        "suite_fresh": suite.get("fresh") if isinstance(suite, dict) else None,
        "suite_resume": suite.get("resume") if isinstance(suite, dict) else None,
        "suite_python": suite.get("python") if isinstance(suite, dict) else None,
        "raw_steps": complete.get("raw_steps") if isinstance(complete, dict) else None,
        "training_seed": arguments.get("seed") if isinstance(arguments, dict) else None,
        "evaluation_episodes": len(episode_records),
        "evaluation_seed_start": (
            episode_records[0].get("seed")
            if episode_records and isinstance(episode_records[0], dict) else None
        ),
        "checkpoint_sha256": checkpoint_sha,
    }


def check_preconditions(run_root: Path) -> dict:
    arms = [_verify_old_arm(method, root) for method, root in OLD_ARMS.items()]
    output_exists = Path(run_root).exists()
    reservation = read_json(PAIR_RECEIPT)
    blockers = []
    if any(not arm["ready"] for arm in arms):
        blockers.append("predecessor_pair_not_complete_and_identity_verified")
    if output_exists:
        blockers.append("requested_new_run_root_already_exists")
    if reservation is not None:
        blockers.append("contractfix_pair_already_reserved_or_launched")
    runner_pythons = {arm.get("suite_python") for arm in arms if arm.get("ready")}
    if len(runner_pythons) > 1:
        blockers.append("predecessor_runner_python_mismatch")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "valid_predecessor_arms": arms,
        "existing_pair_launch_receipt": reservation,
        "ignored_aggregate_status": (
            "Per-method artifacts are authoritative; the old retry01 root may show failed "
            "because its independent longres candidate exited nonzero."
        ),
    }


def worker_layout(run_root: Path, method: str) -> dict:
    if method not in WORKER_ROOT_NAMES:
        raise ValueError(f"unknown contractfix worker method: {method}")
    method_root = Path(run_root).expanduser().resolve() / WORKER_ROOT_NAMES[method]
    method_output = method_root / f"{method}__intersection_sorted_depart4p0"
    overlay_root = method_output.parent / "_hd" / method_output.name.split("__", 1)[0]
    return {
        "method": method,
        "run_root": str(method_root),
        "authorization_marker_value": str(method_root),
        "method_output_dir": str(method_output),
        "scaled_traffic_root": str(
            method_root / "_p4_lowdensity_s4p0__intersection_sorted"
        ),
        "traffic_overlay_root": str(overlay_root),
        "train_traffic_overlay_root": str(overlay_root / "ns_tr"),
        "eval_traffic_overlay_root": str(overlay_root / "ns_eval"),
    }


def worker_command(run_root: Path, method: str) -> list[str]:
    entry = HERE / "train_intersection_yield_v2_d1_contractfix.py"
    return [
        str(_runner_python()), str(entry),
        "train-method",
        "--method", method,
        "--scenario", "intersection_sorted",
        "--depart-scale", "4.0",
        "--eval-traffic-split", "validation",
        "--max-steps", "100000",
        "--run-root", worker_layout(run_root, method)["run_root"],
        "--checkpoint-frequency", "10000",
        "--behavior-diagnostics",
        "--policy-shadow-probes",
        "--policy-shadow-train-raw-interval", "5000",
        "--policy-shadow-eval-max-per-episode", "4",
        "--policy-shadow-critical-eval-sample",
    ]


def worker_specs(run_root: Path) -> list[dict]:
    specs = []
    for method in METHODS:
        layout = worker_layout(run_root, method)
        specs.append({**layout, "argv": worker_command(run_root, method)})
    if len({spec["run_root"] for spec in specs}) != len(specs):
        raise RuntimeError("contractfix workers must use distinct run roots")
    if len({spec["traffic_overlay_root"] for spec in specs}) != len(specs):
        raise RuntimeError("contractfix workers must use distinct traffic overlay roots")
    if len({spec["scaled_traffic_root"] for spec in specs}) != len(specs):
        raise RuntimeError("contractfix workers must use distinct scaled traffic roots")
    return specs


def archive_source(root: Path) -> dict:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required contractfix source missing: " + ", ".join(missing))
    archive = root / "source_archive.zip"
    entries = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted({candidate.resolve() for candidate in SOURCE_FILES}):
            relative = path.relative_to(PROJECT).as_posix()
            payload = path.read_bytes()
            zipped.writestr(relative, payload)
            entries.append({
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            })
    manifest = {
        "created_at": now(),
        "project": str(PROJECT),
        "file_count": len(entries),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "required_files": entries,
        "snapshot_contract": "explicit_runtime_sources_including_parent_temporal_selector",
    }
    write_json(root / "source_archive_manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key != "required_files"}


def supervise(run_root: Path) -> int:
    run_root = Path(run_root).expanduser().resolve()
    authorized = os.environ.get("D1_CONTRACTFIX_SUPERVISOR_RUN_ROOT")
    if not authorized or Path(authorized).expanduser().resolve() != run_root:
        raise RuntimeError("internal contractfix supervisor root authorization failed")

    suite_path = run_root / "suite_manifest.json"
    deadline = time.monotonic() + 30.0
    suite = None
    while time.monotonic() < deadline:
        suite = read_json(suite_path)
        if (
            isinstance(suite, dict)
            and suite.get("status") == "launched"
            and suite.get("supervisor_pid") == os.getpid()
        ):
            break
        time.sleep(0.05)
    if not isinstance(suite, dict) or Path(suite.get("run_root", "")).resolve() != run_root:
        raise RuntimeError("contractfix supervisor manifest does not match its run root")
    if suite.get("status") != "launched" or suite.get("supervisor_pid") != os.getpid():
        raise RuntimeError("contractfix supervisor was not started by a completed guarded launch")

    specs = suite.get("workers")
    expected = worker_specs(run_root)
    if not isinstance(specs, list) or [item.get("method") for item in specs] != list(METHODS):
        raise RuntimeError("contractfix supervisor manifest has invalid worker specifications")
    for actual, expected_item in zip(specs, expected):
        for key in ("run_root", "traffic_overlay_root", "scaled_traffic_root"):
            if Path(actual.get(key, "")).resolve() != Path(expected_item[key]).resolve():
                raise RuntimeError(f"contractfix worker {key} is not isolated")
        if Path(actual.get("authorization_marker_value", "")).resolve() != Path(
            actual["run_root"]
        ).resolve():
            raise RuntimeError("contractfix worker authorization marker does not match its run root")
        if actual.get("argv") != expected_item["argv"]:
            raise RuntimeError("contractfix supervisor command differs from the guarded worker command")

    processes = {}
    logs = {}
    try:
        # Validate all target roots before starting either process, then create
        # the short isolated roots so generated traffic and SUMO overlays cannot
        # contend on a shared traffic_*.tmp destination.
        for spec in specs:
            method_root = Path(spec["run_root"])
            if method_root.exists():
                raise FileExistsError(f"worker run root already exists: {method_root}")
        for spec in specs:
            Path(spec["run_root"]).mkdir(parents=True, exist_ok=False)

        suite["status"] = "workers_starting"
        suite["workers"] = [
            {**spec, "status": "starting"} for spec in specs
        ]
        write_json(suite_path, suite)

        for spec in specs:
            method = spec["method"]
            log_path = run_root / f"{WORKER_ROOT_NAMES[method]}.stdout.log"
            logs[method] = log_path.open("xb")
            environment = dict(os.environ)
            environment["D1_CONTRACTFIX_AUTHORIZED_RUN_ROOT"] = spec[
                "authorization_marker_value"
            ]
            process = subprocess.Popen(
                spec["argv"],
                cwd=PROJECT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=logs[method],
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes[method] = process
            for item in suite["workers"]:
                if item["method"] == method:
                    item.update(status="running", pid=process.pid, started_at=now(), log=str(log_path))
            suite["status"] = "running"
            write_json(suite_path, suite)

        for spec in specs:
            method = spec["method"]
            exit_code = processes[method].wait()
            logs[method].close()
            for item in suite["workers"]:
                if item["method"] == method:
                    item.update(
                        status="complete" if exit_code == 0 else "failed",
                        exit_code=exit_code,
                        finished_at=now(),
                    )
            write_json(suite_path, suite)

        failed = [item for item in suite["workers"] if item.get("exit_code") != 0]
        suite.update(
            status="failed" if failed else "complete",
            exit_codes={item["method"]: item["exit_code"] for item in suite["workers"]},
            finished_at=now(),
        )
        write_json(suite_path, suite)
        receipt = read_json(PAIR_RECEIPT) or {}
        receipt.update(
            status=suite["status"],
            finished_at=suite["finished_at"],
            exit_codes=suite["exit_codes"],
        )
        write_json(PAIR_RECEIPT, receipt)
        return 1 if failed else 0
    except BaseException as error:
        for handle in logs.values():
            if not handle.closed:
                handle.close()
        suite.update(status="supervisor_failed", supervisor_error=repr(error), finished_at=now())
        write_json(suite_path, suite)
        receipt = read_json(PAIR_RECEIPT) or {}
        receipt.update(status="supervisor_failed", error=repr(error), finished_at=now())
        write_json(PAIR_RECEIPT, receipt)
        raise


def launch(run_root: Path, gate: dict) -> dict:
    runner = _runner_python()
    cuda = _probe_cuda(runner)
    reservation = {
        "schema_version": "d1_contractfix_pair_launch_receipt_v1",
        "status": "reserved",
        "reserved_at": now(),
        "run_root": str(run_root.resolve()),
        "methods": list(METHODS),
        "predecessor_gate": gate,
    }
    # One global, exclusive reservation prevents a heartbeat or manual second
    # invocation from starting the same pair under another run-root name.
    with PAIR_RECEIPT.open("x", encoding="utf-8") as handle:
        json.dump(reservation, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        source_archive = archive_source(run_root)
        specs = worker_specs(run_root)
        manifest = {
            "created_at": now(),
            "status": "launching",
            "run_root": str(run_root),
            "methods": list(METHODS),
            "method_protocol": "d1_c8_c9_contractfix_joint_v1",
            "scene": "intersection_sorted_depart4p0",
            "scenario": "intersection_sorted",
            "depart_scale": 4.0,
            "training_seed": 0,
            "fresh": True,
            "resume": False,
            "raw_step_budget_per_method": 100000,
            "checkpoint_frequency_raw_steps": 10000,
            "max_cuda_workers": 2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all-visible"),
            "gpu": cuda["gpu"],
            "python": str(runner),
            "torch": cuda["torch"],
            "smoke": False,
            "evaluation": {
                "split": "validation",
                "episodes": 100,
                "seed_start": 10000,
                "seed_end": 10099,
                "checkpoint": "final_model.zip",
            },
            "reward_contract": "unchanged_environment_step_reward_v2",
            "bootstrap_protocol": "unchanged_released_source_compatible_single_gamma_n_step_timeout_bootstrap",
            "policy_shadow": {
                "train_raw_interval": 5000,
                "train_max_unique_states_entire_phase": 20,
                "eval_max_unique_states_per_episode": 4,
                "extra_environment_steps": 0,
            },
            "trajectory_history_audit": {
                "phase_scope": ["train", "eval"],
                "sampling": "every_real_preaction_decision",
                "geometry_active": True,
                "velocity_contract": "smarts",
                "extra_environment_steps": 0,
            },
            "predecessor_gate": gate,
            "worker_root_layout": "distinct_short_method_roots_st_and_st_rt",
            "workers": specs,
            "commands": [spec["argv"] for spec in specs],
            "source_archive": source_archive,
            "launcher_entrypoint": str(Path(__file__).resolve()),
        }
        write_json(run_root / "suite_manifest.json", manifest)
        stdout_path = run_root / "supervisor.stdout.log"
        stderr_path = run_root / "supervisor.stderr.log"
        environment = dict(os.environ)
        environment["D1_CONTRACTFIX_SUPERVISOR_RUN_ROOT"] = str(run_root.resolve())
        supervisor_argv = [
            str(runner), str(Path(__file__).resolve()),
            "--_supervise", "--run-root", str(run_root.resolve()),
        ]
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                supervisor_argv,
                cwd=PROJECT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
            )
        manifest.update(
            status="launched",
            supervisor_pid=process.pid,
            launched_at=now(),
            supervisor_command=supervisor_argv,
        )
        write_json(run_root / "suite_manifest.json", manifest)
        reservation.update(
            status="launched",
            launched_at=manifest["launched_at"],
            supervisor_pid=process.pid,
            source_archive_manifest=str(run_root / "source_archive_manifest.json"),
        )
        write_json(PAIR_RECEIPT, reservation)
        return {
            "status": "launched",
            "run_root": str(run_root),
            "supervisor_pid": process.pid,
            "worker_run_roots": {spec["method"]: spec["run_root"] for spec in specs},
        }
    except BaseException as error:
        if run_root.exists():
            write_json(run_root / "launch_error.json", {"time": now(), "error": repr(error)})
        reservation.update(status="launch_error", failed_at=now(), error=repr(error))
        write_json(PAIR_RECEIPT, reservation)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--_supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--check-only", "--preview", dest="check_only", action="store_true",
        help="read-only prerequisite check (default)",
    )
    parser.add_argument("--launch", action="store_true", help="launch only after the predecessor gate passes")
    args = parser.parse_args(argv)
    if args._supervise:
        return supervise(args.run_root)
    if args.check_only and args.launch:
        parser.error("choose --check-only/--preview or --launch")
    run_root = args.run_root.expanduser().resolve()
    gate = check_preconditions(run_root)
    preview = {
        "mode": "launch" if args.launch else "check_only",
        "status": "ready" if gate["ready"] else "blocked",
        "run_root": str(run_root),
        "methods": list(METHODS),
        "training": {"seed": 0, "fresh": True, "raw_steps_per_method": 100000},
        "evaluation": {"split": "validation", "episodes": 100, "seeds": [10000, 10099]},
        "gate": gate,
        "workers": worker_specs(run_root),
    }
    if not args.launch or not gate["ready"]:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0 if not args.launch else 2
    result = launch(run_root, gate)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
