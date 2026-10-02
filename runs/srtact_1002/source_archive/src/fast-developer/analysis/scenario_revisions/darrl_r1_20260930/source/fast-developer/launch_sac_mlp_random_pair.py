"""Launch matched fresh SAC+MLP runs on a selected random-traffic pair.

Dry-run is the default. Pass ``--start`` only after the scenario/factory smoke
checks pass. The two workers use the existing D1 ``train-method``
entrypoint, independent scenario-specific output directories, and CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


FAST_DEVELOPER = Path(__file__).resolve().parent
PROJECT_ROOT = FAST_DEVELOPER.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
RUNS_ROOT = WORKSPACE_ROOT / "runs"
TRAINER = FAST_DEVELOPER / "train_intersection_yield_v2_d1.py"
SCENARIOS = (
    ("medium", "intersection_random_medium_v1"),
    ("medium_p05", "intersection_random_medium_p05_v1"),
)
SCENARIO_PAIRS = {
    "medium-p05": SCENARIOS,
    "darrl-low-medium": (
        ("darrl_low", "intersection_random_darrl_low_v1"),
        ("darrl_medium", "intersection_random_darrl_medium_v1"),
    ),
    "p03-p02": (
        ("medium_p03", "intersection_random_medium_p03_v1"),
        ("medium_p02", "intersection_random_medium_p02_v1"),
    ),
}
RAW_STEPS = 100_000
LEARNING_STARTS_RAW_STEPS = 5_000
CHECKPOINT_FREQUENCY_RAW_STEPS = 10_000
EVALUATION_EPISODES = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    candidates = {
        "launcher": Path(__file__),
        "d1_trainer": TRAINER,
        "base_trainer": FAST_DEVELOPER / "train_intersection_yield_v2.py",
        "random_intersection": PROJECT_ROOT / "envs" / "sumo" / "random_intersection.py",
        "paper_env": PROJECT_ROOT / "envs" / "sumo" / "paper_env.py",
        "high_density_adapter": PROJECT_ROOT / "envs" / "sumo" / "high_density_env_v1.py",
    }
    return {
        name: _sha256(path)
        for name, path in candidates.items()
        if path.is_file()
    }


def build_worker_command(scenario: str, run_root: Path) -> list[str]:
    """Build one fresh, full-budget worker command with no resume/checkpoint input."""
    return [
        sys.executable,
        str(TRAINER),
        "train-method",
        "--method",
        "sac_mlp",
        "--scenario",
        scenario,
        "--depart-scale",
        "1.0",
        "--eval-traffic-split",
        "validation",
        "--max-steps",
        str(RAW_STEPS),
        "--run-root",
        str(run_root),
        "--checkpoint-frequency",
        str(CHECKPOINT_FREQUENCY_RAW_STEPS),
        "--behavior-diagnostics",
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_run_root() -> Path:
    stamp = datetime.now().astimezone().strftime("%y%m%d_%H%M%S")
    return RUNS_ROOT / f"smlp_pair_{stamp}"


def _validate_run_root(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    allowed = RUNS_ROOT.resolve()
    try:
        relative = root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"run root must be inside the short workspace runs directory: {allowed}") from exc
    if not relative.parts:
        raise ValueError("run root must be a new child directory of workspace runs")
    if root.exists():
        raise FileExistsError(f"refusing to reuse existing run root: {root}")
    return root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(
    root: Path, *, poll_seconds: float,
    scenarios: tuple[tuple[str, str], ...] = SCENARIOS,
) -> int:
    root.mkdir(parents=True, exist_ok=False)
    log_root = root / "launcher_logs"
    log_root.mkdir()
    worker_specs = []
    for key, scenario in scenarios:
        command = build_worker_command(scenario, root)
        worker_specs.append(
            {
                "key": key,
                "scenario": scenario,
                "method": "sac_mlp",
                "command": command,
                "command_windows": subprocess.list2cmdline(command),
                "stdout": str(log_root / f"{key}.stdout.log"),
                "stderr": str(log_root / f"{key}.stderr.log"),
                "status": "queued",
            }
        )

    manifest = {
        "schema": "sac-mlp-random-pair-launch/v1",
        "created_at_utc": _utc_now(),
        "run_root": str(root),
        "source_hashes": _source_hashes(),
        "method": "sac_mlp",
        "scenario_pair": [scenario for _, scenario in scenarios],
        "training_seed": 0,
        "training_seed_source": "base trainer constant; D1 CLI has no seed override",
        "traffic_seed_split": "train for training; validation for final evaluation",
        "raw_training_steps": RAW_STEPS,
        "learning_starts_raw_steps": LEARNING_STARTS_RAW_STEPS,
        "checkpoint_frequency_raw_steps": CHECKPOINT_FREQUENCY_RAW_STEPS,
        "evaluation_episodes": EVALUATION_EPISODES,
        "device_request": "cuda (both workers share the inherited visible CUDA device set)",
        "fresh_from_scratch": True,
        "resume": False,
        "smoke": False,
        "worker_specs": worker_specs,
    }
    status: dict[str, Any] = {
        "schema": "sac-mlp-random-pair-status/v1",
        "status": "starting",
        "supervisor_pid": os.getpid(),
        "started_at_utc": _utc_now(),
        "workers": {},
    }
    _write_json(root / "launcher_manifest.json", manifest)
    status_path = root / "launcher_status.json"
    _write_json(status_path, status)

    try:
        import torch
    except Exception as exc:
        status.update(status="preflight_failed", error=f"cannot import torch: {exc!r}")
        _write_json(status_path, status)
        raise
    if not torch.cuda.is_available():
        status.update(status="preflight_failed", error="torch.cuda.is_available() is false")
        _write_json(status_path, status)
        raise RuntimeError("CUDA is unavailable in the launcher interpreter")
    status["cuda_device_0"] = torch.cuda.get_device_name(0)
    _write_json(status_path, status)

    streams: dict[str, tuple[Any, Any]] = {}
    processes: dict[str, subprocess.Popen] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        for spec in worker_specs:
            key = spec["key"]
            stdout_stream = Path(spec["stdout"]).open("x", encoding="utf-8", buffering=1)
            stderr_stream = Path(spec["stderr"]).open("x", encoding="utf-8", buffering=1)
            streams[key] = (stdout_stream, stderr_stream)
            try:
                process = subprocess.Popen(
                    spec["command"],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    creationflags=create_no_window,
                    close_fds=False,
                )
            except BaseException as exc:
                spec.update(status="launch_failed", launch_error=repr(exc))
                status["workers"][key] = {
                    "status": "launch_failed",
                    "error": repr(exc),
                }
                status.update(status="partial_launch_failure", updated_at_utc=_utc_now())
                _write_json(status_path, status)
                return 1
            processes[key] = process
            spec.update(status="running", pid=process.pid, started_at_utc=_utc_now())
            status["workers"][key] = {
                "scenario": spec["scenario"],
                "pid": process.pid,
                "status": "running",
                "stdout": spec["stdout"],
                "stderr": spec["stderr"],
                "command": spec["command"],
            }
            status.update(status="running", updated_at_utc=_utc_now())
            _write_json(root / "launcher_manifest.json", manifest)
            _write_json(status_path, status)

        while True:
            all_done = True
            for spec in worker_specs:
                key = spec["key"]
                process = processes[key]
                code = process.poll()
                if code is None:
                    all_done = False
                    continue
                worker_status = status["workers"][key]
                if worker_status.get("exit_code") != code:
                    worker_status.update(
                        status="complete" if code == 0 else "failed",
                        exit_code=code,
                        finished_at_utc=_utc_now(),
                    )
                    spec.update(status=worker_status["status"], exit_code=code)
                    _write_json(root / "launcher_manifest.json", manifest)
                    status.update(updated_at_utc=_utc_now())
                    _write_json(status_path, status)
            if all_done:
                break
            time.sleep(poll_seconds)
    finally:
        for stdout_stream, stderr_stream in streams.values():
            stdout_stream.close()
            stderr_stream.close()

    codes = {key: proc.returncode for key, proc in processes.items()}
    status.update(
        status="complete" if all(code == 0 for code in codes.values()) else "failed",
        exit_codes=codes,
        finished_at_utc=_utc_now(),
    )
    manifest["exit_codes"] = codes
    manifest["finished_at_utc"] = status["finished_at_utc"]
    _write_json(root / "launcher_manifest.json", manifest)
    _write_json(status_path, status)
    return 0 if all(code == 0 for code in codes.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--pair", choices=tuple(SCENARIO_PAIRS), default="medium-p05")
    parser.add_argument("--start", action="store_true", help="start both full 100k workers; default is dry-run")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    scenarios = SCENARIO_PAIRS[args.pair]
    root = args.run_root or _default_run_root()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        root = _validate_run_root(root)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    plan = {
        "run_root": str(root),
        "training_seed": 0,
        "raw_training_steps": RAW_STEPS,
        "learning_starts_raw_steps": LEARNING_STARTS_RAW_STEPS,
        "checkpoint_frequency_raw_steps": CHECKPOINT_FREQUENCY_RAW_STEPS,
        "evaluation_episodes": EVALUATION_EPISODES,
        "device_request": "cuda",
        "workers": [
            {
                "scenario": scenario,
                "stdout": str(root / "launcher_logs" / f"{key}.stdout.log"),
                "stderr": str(root / "launcher_logs" / f"{key}.stderr.log"),
                "command": build_worker_command(scenario, root),
            }
            for key, scenario in scenarios
        ],
    }
    if not args.start:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("dry-run only; pass --start after the short scenario/factory smoke passes")
        return 0
    return run(root, poll_seconds=args.poll_seconds, scenarios=scenarios)


if __name__ == "__main__":
    raise SystemExit(main())
