"""Retry the real-SUMO v4.13 smoke and preserve every failed attempt."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / "results_topo_v4_13_dev" / "engineering"
SUMMARY = ENGINEERING / "smoke_supervisor_summary_breakaway.json"
PREVIOUS_SUMMARY = ENGINEERING / "smoke_supervisor_summary.json"
ATTEMPTS = ("e7", "e8", "e9", "e10")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "train_paper_sb3_sumo_v4_13.py"),
        "--scenario", "left_turn",
        "--algo", "topo_v4_13_gradient_isolated_tempered_joint_support_prcr_full",
        "--max-steps", "96",
        "--learning-starts", "48",
        "--checkpoint-freq", "96",
        "--eval-freq", "0",
        "--eval-episodes", "1",
        "--evaluation-split", "validation",
        "--evaluation-seed-start", "166003",
        "--calibration-episodes", "1",
        "--calibration-seed-start", "156003",
        "--seed", "3",
        "--device", "cpu",
        "--batch-size", "2",
        "--learning-rate", "0.0001",
        "--discount", "0.99",
        "--buffer-size", "128",
        "--action-repeat", "3",
        "--ego-control-profile", "direct",
        "--traffic-protocol", "frozen_60_20_20",
        "--episode-limit-profile", "source",
        "--output-dir", str(run_dir.parent),
        "--model-name", run_dir.name,
        "--experiment-contract-sha256",
        "64705b16ec83b62b2dd9113d9c21bc4aa70033bc9402ef7d5cca69447a24a2fd",
        "--attribution-sha256",
        "40c32791653d52756fb3e5e7c4a86d8b04be5b24bae6f32dbad71b5e15eff0b2",
    ]


def main() -> int:
    ENGINEERING.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().astimezone().isoformat()
    rows: list[dict[str, object]] = []
    selected_run: str | None = None
    for index, attempt in enumerate(ATTEMPTS):
        run_dir = ROOT / "r413s" / attempt / "m"
        stdout_path = ENGINEERING / f"smoke_{attempt}.stdout.log"
        stderr_path = ENGINEERING / f"smoke_{attempt}.stderr.log"
        if run_dir.exists():
            rows.append(
                {
                    "attempt": attempt,
                    "status": "immutable_directory_exists",
                    "run_directory": str(run_dir),
                }
            )
            continue
        command = _command(run_dir)
        row: dict[str, object] = {
            "attempt": attempt,
            "started_at_local": dt.datetime.now().astimezone().isoformat(),
            "run_directory": str(run_dir),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command": command,
        }
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            row["returncode"] = int(completed.returncode)
            required = (
                run_dir / "selected_model.zip",
                run_dir / "action_diagnostics.json",
                run_dir / "paper_evaluation_detailed.json",
                run_dir / "selector" / "receipt.json",
            )
            complete = completed.returncode == 0 and all(
                path.is_file() for path in required
            )
            row["status"] = "complete" if complete else "failed"
            row["required_artifacts_complete"] = complete
            if complete:
                selected_run = str(run_dir)
        except Exception as exc:
            row["returncode"] = None
            row["status"] = "launcher_exception"
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["completed_at_local"] = dt.datetime.now().astimezone().isoformat()
        rows.append(row)
        _write_json(
            SUMMARY,
            {
                "schema_version": "topo-scene-v4.13.engineering-smoke-supervisor/v1",
                "started_at_local": started,
                "status": "complete" if selected_run else "running",
                "attempt_all_failures_preserved": True,
                "selected_run_directory": selected_run,
                "attempts": rows,
                "formal_test_accessed": False,
            },
        )
        if selected_run:
            break
        if index + 1 < len(ATTEMPTS):
            time.sleep(10)
    payload = {
        "schema_version": "topo-scene-v4.13.engineering-smoke-supervisor/v1",
        "started_at_local": started,
        "completed_at_local": dt.datetime.now().astimezone().isoformat(),
        "status": "complete" if selected_run else "all_attempts_failed",
        "attempt_all_failures_preserved": True,
        "previous_failed_attempt_summary": str(PREVIOUS_SUMMARY),
        "selected_run_directory": selected_run,
        "attempts": rows,
        "formal_test_accessed": False,
    }
    _write_json(SUMMARY, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if selected_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
