"""Continue prioritized paper seed waves with dual-protocol acceptance.

The coordinator deliberately composes the existing formal training,
re-evaluation, summarization, and report CLIs.  It does not implement another
environment factory, algorithm, metric, or acceptance path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMARTS_SCENARIOS = (
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
)
NON_LEFT_SCENARIOS = SMARTS_SCENARIOS[1:]
METHODS = ("sac", "ppo")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wave_scenarios(seed: int) -> tuple[str, ...]:
    # left_turn already has both methods for seeds 0-2 and SAC for seed3.
    # Finish broad scene coverage before filling the remaining left-turn tail.
    return NON_LEFT_SCENARIOS if seed in (1, 2) else SMARTS_SCENARIOS


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


class Coordinator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.result_root = args.result_root.resolve()
        self.protocol_path = args.protocol_path.resolve()
        self.report_dir = args.report_dir.resolve()
        self.state_path = self.result_root / "seed_completion_coordinator_state.json"
        self.log_dir = self.result_root / "seed_completion_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "status": "running",
            "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seeds": args.seeds,
            "training_workers": args.training_workers,
            "evaluation_workers": args.evaluation_workers,
            "device": args.device,
            "waves": [],
            "failures": [],
        }

    def save(self) -> None:
        _atomic_json(self.state_path, self.state)

    def run_logged(self, label: str, command: list[str]) -> int:
        stdout_path = self.log_dir / f"{label}.stdout.log"
        stderr_path = self.log_dir / f"{label}.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            stdout.write(
                f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"{_command_text(command)}\n"
            )
            stdout.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return int(completed.returncode)

    def wait_for_existing_training(self) -> None:
        orchestrator_state = self.result_root / "orchestrator_state.json"
        while True:
            current = _load_json(orchestrator_state)
            if not current or current.get("status") != "running":
                return
            self.state["waiting_for_existing_orchestrator_pid"] = current.get("pid")
            self.state["existing_running_jobs"] = current.get("running_jobs", [])
            self.save()
            time.sleep(self.args.poll_seconds)

    def train_wave(self, seed: int, scenarios: tuple[str, ...]) -> int:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "reproduce_paper_sb3_sumo.py"),
            "run",
            "--protocol-path",
            str(self.protocol_path),
            "--profile",
            "paper",
            "--methods",
            ",".join(METHODS),
            "--scenarios",
            ",".join(scenarios),
            "--seeds",
            str(seed),
            "--device",
            self.args.device,
            "--output-dir",
            str(self.result_root),
            "--workers",
            str(self.args.training_workers),
            "--schedule",
            "diversified",
            "--continue-on-error",
        ]
        return self.run_logged(f"seed{seed}_training", command)

    def _reevaluate_one(self, seed: int, method: str, scenario: str) -> dict[str, Any]:
        run_name = f"paper__{method}__{scenario}__seed{seed}"
        run_dir = self.result_root / run_name
        required = (
            run_dir / "final_model.zip",
            run_dir / "checkpoint_audit.json",
            run_dir / "paper_evaluation_detailed.json",
        )
        if any(not path.is_file() for path in required):
            return {
                "run": run_name,
                "status": "training_incomplete",
                "missing": [str(path.name) for path in required if not path.is_file()],
            }
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "reevaluate_paper_run.py"),
            "--run-dir",
            str(run_dir),
            "--protocols",
            "frozen_80_20,source_all",
            "--device",
            self.args.device,
        ]
        returncode = self.run_logged(f"seed{seed}_{method}_{scenario}_evaluation", command)
        manifest = _load_json(run_dir / "reevaluation_manifest.json")
        return {
            "run": run_name,
            "status": "evaluated" if returncode == 0 else "evaluation_failed",
            "returncode": returncode,
            "evaluations": manifest.get("evaluations") if manifest else None,
        }

    def reevaluate_wave(
        self, seed: int, scenarios: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        jobs = [(method, scenario) for method in METHODS for scenario in scenarios]
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.args.evaluation_workers) as executor:
            futures = {
                executor.submit(self._reevaluate_one, seed, method, scenario): (
                    method,
                    scenario,
                )
                for method, scenario in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda row: row["run"])

    def refresh_summary_and_report(self, seed: int) -> list[dict[str, Any]]:
        summary_dir = self.result_root / "summary_dual_protocol"
        commands = [
            (
                f"seed{seed}_summary",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "summarize_dual_protocol_evaluations.py"),
                    "--result-root",
                    str(self.result_root),
                    "--protocol-path",
                    str(self.protocol_path),
                    "--output-dir",
                    str(summary_dir),
                ],
            ),
            (
                f"seed{seed}_report_artifact",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "build_baseline_reproduction_report.py"),
                    "--summary",
                    str(summary_dir / "summary.json"),
                    "--protocol",
                    str(self.protocol_path),
                    "--result-root",
                    str(self.result_root),
                    "--output",
                    str(self.report_dir / "artifact.json"),
                ],
            ),
            (
                f"seed{seed}_report_package",
                [
                    str(self.args.node_executable.resolve()),
                    str(self.args.portable_builder.resolve()),
                    "--input",
                    str(self.report_dir / "artifact.json"),
                    "--output",
                    str(self.report_dir / "baseline_reproduction_fixed.html"),
                ],
            ),
        ]
        results: list[dict[str, Any]] = []
        for label, command in commands:
            returncode = self.run_logged(label, command)
            results.append({"step": label, "returncode": returncode})
            if returncode != 0:
                break
        return results

    def run(self) -> int:
        previous = _load_json(self.state_path)
        if previous and previous.get("status") == "running" and not self.args.force:
            raise RuntimeError(
                f"Coordinator state already says running with pid={previous.get('pid')}; "
                "use --force only after verifying it is stale"
            )
        self.save()
        try:
            self.wait_for_existing_training()
            self.state.pop("waiting_for_existing_orchestrator_pid", None)
            self.state.pop("existing_running_jobs", None)
            for seed in self.args.seeds:
                scenarios = _wave_scenarios(seed)
                wave: dict[str, Any] = {
                    "seed": seed,
                    "scenarios": list(scenarios),
                    "status": "training",
                    "started_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                }
                self.state["waves"].append(wave)
                self.save()

                training_returncode = self.train_wave(seed, scenarios)
                wave["training_returncode"] = training_returncode
                wave["status"] = "evaluating"
                if training_returncode != 0:
                    self.state["failures"].append(
                        {"seed": seed, "stage": "training", "returncode": training_returncode}
                    )
                self.save()

                evaluations = self.reevaluate_wave(seed, scenarios)
                wave["evaluations"] = evaluations
                failed_evaluations = [
                    row
                    for row in evaluations
                    if row["status"] not in {"evaluated"}
                    or int(row.get("returncode", 0)) != 0
                ]
                if failed_evaluations:
                    self.state["failures"].extend(failed_evaluations)
                wave["status"] = "reporting"
                self.save()

                reporting = self.refresh_summary_and_report(seed)
                wave["reporting"] = reporting
                failed_reporting = [row for row in reporting if row["returncode"] != 0]
                if failed_reporting:
                    self.state["failures"].extend(failed_reporting)
                wave["status"] = (
                    "complete"
                    if training_returncode == 0
                    and not failed_evaluations
                    and not failed_reporting
                    else "complete_with_failures"
                )
                wave["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                self.save()

            self.state["status"] = (
                "complete" if not self.state["failures"] else "complete_with_failures"
            )
            self.state["finished_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            self.save()
            return 0 if not self.state["failures"] else 1
        except BaseException as exc:
            self.state["status"] = "crashed"
            self.state["fatal_error"] = f"{type(exc).__name__}: {exc}"
            self.state["finished_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            self.save()
            raise


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or any(seed not in (1, 2, 3, 4) for seed in seeds):
        raise ValueError("--seeds must be a subset of 1,2,3,4")
    return list(dict.fromkeys(seeds))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    parser.add_argument("--portable-builder", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1,2,3,4"))
    parser.add_argument("--training-workers", type=int, default=4)
    parser.add_argument("--evaluation-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.training_workers <= 0 or args.evaluation_workers <= 0:
        raise ValueError("worker counts must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.plan_only:
        print(
            json.dumps(
                {
                    "seeds": args.seeds,
                    "waves": [
                        {"seed": seed, "scenarios": list(_wave_scenarios(seed))}
                        for seed in args.seeds
                    ],
                },
                indent=2,
            )
        )
        return 0
    return Coordinator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
