"""Benchmark safe multi-process throughput; outputs are never scientific evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("cross", "roundabout_medium", "roundabout", "carla")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _command(root: Path, scenario: str, raw_steps: int) -> tuple[str, list[str]]:
    scenario_code = {
        "cross": "cross",
        "roundabout_medium": "rab",
        "roundabout": "rac",
        "carla": "carla",
    }[scenario]
    name = f"b__soft__{scenario_code}__s91"
    return name, [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v2.py"),
        "--scenario",
        scenario,
        "--algo",
        "topo_v2_soft",
        "--max-steps",
        str(raw_steps),
        "--learning-starts",
        str(max(3, raw_steps * 3 // 20)),
        "--buffer-size",
        "1000",
        "--checkpoint-freq",
        str(raw_steps),
        "--eval-freq",
        "0",
        "--eval-episodes",
        "1",
        "--seed",
        "91",
        "--device",
        "cuda",
        "--traffic-protocol",
        "frozen_60_20_20",
        "--evaluation-split",
        "validation",
        "--episode-limit-profile",
        "source",
        "--slot-balance-coef",
        "0.001",
        "--experiment-contract-sha256",
        "CONCURRENCY_BENCHMARK_NOT_EVIDENCE",
        "--output-dir",
        str(root / "runs"),
        "--model-name",
        name,
    ]


def benchmark(root: Path, *, workers: int, raw_steps: int) -> dict[str, Any]:
    if workers <= 0 or workers > len(SCENARIOS):
        raise ValueError(f"workers must be in [1, {len(SCENARIOS)}]")
    if raw_steps < 120:
        raise ValueError("raw_steps must be at least 120")
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"Benchmark output already exists: {root}")
    (root / "runs").mkdir(parents=True)
    (root / "logs").mkdir()
    commands = [_command(root, scenario, raw_steps) for scenario in SCENARIOS]

    def run_one(name: str, command: list[str]) -> dict[str, Any]:
        log_path = root / "logs" / f"{name}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        elapsed = time.perf_counter() - started
        run_dir = root / "runs" / name
        required = (
            "final_model.zip",
            "checkpoint_audit.json",
            "paper_evaluation_detailed.json",
            "training_diagnostics.json",
            "performance_profile.json",
            "action_diagnostics.json",
        )
        missing = [item for item in required if not (run_dir / item).is_file()]
        row: dict[str, Any] = {
            "name": name,
            "returncode": completed.returncode,
            "process_elapsed_seconds": elapsed,
            "missing_artifacts": missing,
            "log": str(log_path),
        }
        if not missing:
            performance = _load_json(run_dir / "performance_profile.json")
            detailed = _load_json(run_dir / "paper_evaluation_detailed.json")
            checkpoint = _load_json(run_dir / "checkpoint_audit.json")
            row.update(
                {
                    "training_wall_seconds": performance[
                        "end_to_end_training_wall_seconds"
                    ],
                    "train_ms_per_learner_update": performance[
                        "end_to_end_wall_ms_per_learner_update"
                    ],
                    "peak_gpu_memory_mb": performance["peak_gpu_memory_mb"],
                    "trained_raw_steps": detailed["trained_raw_steps"],
                    "final_checkpoint_raw_step": checkpoint["checkpoints"][-1][
                        "raw_step"
                    ],
                }
            )
        row["accepted"] = bool(
            completed.returncode == 0
            and not missing
            and row.get("trained_raw_steps") == raw_steps
            and row.get("final_checkpoint_raw_step") == raw_steps
        )
        return row

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, name, command): name
            for name, command in commands
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    makespan = time.perf_counter() - started
    rows.sort(key=lambda row: row["name"])
    payload = {
        "schema_version": "topology-temporal-v2-concurrency-benchmark/v1",
        "scientific_evidence": False,
        "purpose": "Select a safe experiment-runner worker count only.",
        "workers": workers,
        "logical_processors": os.cpu_count(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_memory_mb": (
            torch.cuda.get_device_properties(0).total_memory / (1024.0**2)
            if torch.cuda.is_available()
            else None
        ),
        "raw_steps_per_run": raw_steps,
        "run_count": len(rows),
        "makespan_seconds": makespan,
        "sum_process_elapsed_seconds": sum(
            float(row["process_elapsed_seconds"]) for row in rows
        ),
        "throughput_speedup_over_observed_serial": (
            sum(float(row["process_elapsed_seconds"]) for row in rows) / makespan
        ),
        "all_accepted": all(bool(row["accepted"]) for row in rows),
        "runs": rows,
    }
    _write_json(root / "benchmark.json", payload)
    return payload


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw-steps", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "topo_v2" / "concurrency_benchmark_w4",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    payload = benchmark(args.output, workers=args.workers, raw_steps=args.raw_steps)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
