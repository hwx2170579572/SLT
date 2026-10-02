"""Plan, execute, inspect, and summarize the SB3/SUMO paper reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper_evaluation_contract import saved_evaluation_contract_errors

PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "sb3_sumo_paper" / "protocol.json"
PAPER_REFERENCE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sb3_sumo_paper"
    / "paper_reference_tables.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results_sb3_sumo_paper"
    / "paper_source_audited_runs_v12_frozen"
)


@dataclass(frozen=True)
class Job:
    profile: str
    method: str
    cli_algorithm: str
    scenario: str
    seed: int
    raw_steps: int
    test_episodes: int
    checkpoint_frequency: int
    evaluation_frequency: int
    ego_control_profile: str = "direct"
    traffic_protocol: str = "source_all"
    episode_limit_profile: str = "source"
    implementation_id: str | None = None

    @property
    def name(self) -> str:
        return f"{self.profile}__{self.method}__{self.scenario}__seed{self.seed}"


def load_protocol() -> dict[str, Any]:
    with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_paper_references() -> dict[str, Any]:
    with PAPER_REFERENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_protocol_snapshot(output_dir: Path) -> Path:
    """Freeze the exact protocol bytes before any formal subprocess starts."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "protocol.snapshot.json"
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    if snapshot.is_file():
        if snapshot.read_bytes() != protocol_bytes:
            raise RuntimeError(
                f"Protocol mismatch in {output_dir}; use a fresh result root"
            )
    else:
        snapshot.write_bytes(protocol_bytes)
    return snapshot


def _comma_values(value: str | None, allowed: Iterable[str], label: str) -> list[str]:
    allowed_values = list(allowed)
    if value is None or value.strip().lower() == "all":
        return allowed_values
    selected = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(selected) - set(allowed_values))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; choose from {allowed_values}")
    return selected


def _seed_values(value: str | None, defaults: Iterable[int]) -> list[int]:
    if value is None or value.strip().lower() == "all":
        return list(defaults)
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def make_jobs(args: argparse.Namespace, protocol: dict[str, Any]) -> list[Job]:
    profile = protocol["profiles"][args.profile]
    environment_protocol = {
        **protocol.get("environment_protocol", {}),
        **profile.get("environment_protocol", {}),
    }
    method_names = _comma_values(
        args.methods, protocol["supported_methods"], "method"
    )
    scenario_names = _comma_values(
        args.scenarios, protocol["scenario_order"], "scenario"
    )
    seeds = _seed_values(args.seeds, profile["seeds"])
    jobs: list[Job] = []
    # Keep method comparisons close in wall-clock time: complete both methods
    # for one scenario/seed before moving to the next seed. Job identities and
    # acceptance contracts are independent of this scheduling order.
    for scenario in scenario_names:
        for seed in seeds:
            for method in method_names:
                method_config = protocol["supported_methods"][method]
                supported_scenarios = set(
                    method_config.get("scenarios", protocol["scenario_order"])
                )
                if scenario not in supported_scenarios:
                    continue
                method_environment = {
                    **environment_protocol,
                    **method_config.get("environment_protocol", {}),
                }
                cli_algorithm = method_config["train_cli_algorithm"]
                jobs.append(
                    Job(
                        profile=args.profile,
                        method=method,
                        cli_algorithm=cli_algorithm,
                        scenario=scenario,
                        seed=seed,
                        raw_steps=int(profile["raw_training_steps"]),
                        test_episodes=int(profile["test_episodes"]),
                        checkpoint_frequency=int(
                            profile["checkpoint_frequency_raw_steps"]
                        ),
                        evaluation_frequency=int(
                            profile["evaluation_frequency_raw_steps"]
                        ),
                        ego_control_profile=str(
                            method_environment.get(
                                "ego_control_profile", "direct"
                            )
                        ),
                        traffic_protocol=str(
                            method_environment.get(
                                "traffic_protocol", "source_all"
                            )
                        ),
                        episode_limit_profile=str(
                            method_environment.get(
                                "episode_limit_profile", "source"
                            )
                        ),
                        implementation_id=method_config.get(
                            "implementation_id"
                        ),
                    )
                )
    if not jobs:
        raise ValueError("No supported method/scenario combinations were selected")
    return jobs


def _diversified_job_order(jobs: Iterable[Job]) -> list[Job]:
    """Interleave scenarios and methods while retaining per-pair seed order.

    This affects scheduling only. Job identities, seeds, protocol parameters,
    output directories, and acceptance checks remain unchanged. With two
    methods and five scenarios, the first five runnable slots cover five
    different scenarios and alternate methods.
    """

    job_list = list(jobs)
    scenario_order = list(dict.fromkeys(job.scenario for job in job_list))
    method_order = list(dict.fromkeys(job.method for job in job_list))
    queues = {
        (scenario, method): deque(
            job
            for job in job_list
            if job.scenario == scenario and job.method == method
        )
        for scenario in scenario_order
        for method in method_order
    }
    ordered: list[Job] = []
    round_index = 0
    while len(ordered) < len(job_list):
        made_progress = False
        for method_pass in range(len(method_order)):
            for scenario_index, scenario in enumerate(scenario_order):
                method = method_order[
                    (scenario_index + round_index + method_pass)
                    % len(method_order)
                ]
                queue = queues[(scenario, method)]
                if queue:
                    ordered.append(queue.popleft())
                    made_progress = True
        if not made_progress:
            raise RuntimeError("Diversified scheduler could not consume all jobs")
        round_index += 1
    return ordered


def run_directory(output_dir: Path, job: Job) -> Path:
    return output_dir.resolve() / job.name


def is_excluded(output_dir: Path, job: Job) -> bool:
    marker = "NOT_FORMAL_SOURCE_FIDELITY.md"
    return (output_dir.resolve() / marker).is_file() or (
        run_directory(output_dir, job) / marker
    ).is_file()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _accepted_run(directory: Path, job: Job) -> bool:
    required = (
        directory / "arguments.json",
        directory / "final_model.zip",
        directory / "final_evaluation.json",
        directory / "paper_evaluation_detailed.json",
        directory / "checkpoint_audit.json",
    )
    if not all(path.is_file() for path in required):
        return False
    argument_payload = json.loads(required[0].read_text(encoding="utf-8"))
    arguments = argument_payload["requested_raw_steps"]
    evaluation = json.loads(required[2].read_text(encoding="utf-8"))
    detailed = json.loads(required[3].read_text(encoding="utf-8"))
    checkpoint_audit = json.loads(required[4].read_text(encoding="utf-8"))
    expected_checkpoint_steps = list(
        range(
            job.checkpoint_frequency,
            job.raw_steps + 1,
            job.checkpoint_frequency,
        )
    )
    checkpoint_rows = checkpoint_audit.get("checkpoints", [])
    expected_clock_field = (
        "num_timesteps" if job.cli_algorithm == "ppo" else "_raw_steps_seen"
    )
    checkpoint_contract = (
        str(checkpoint_audit.get("algorithm")) == job.cli_algorithm
        and str(checkpoint_audit.get("scenario")) == job.scenario
        and int(checkpoint_audit.get("requested_raw_steps", -1)) == job.raw_steps
        and int(checkpoint_audit.get("checkpoint_frequency_raw_steps", -1))
        == job.checkpoint_frequency
        and str(checkpoint_audit.get("clock_field")) == expected_clock_field
        and int(checkpoint_audit.get("expected_checkpoint_count", -1))
        == len(expected_checkpoint_steps)
        and len(checkpoint_rows) == len(expected_checkpoint_steps)
    )
    if checkpoint_contract:
        for expected_step, row in zip(expected_checkpoint_steps, checkpoint_rows):
            checkpoint_path = directory / str(row.get("path", ""))
            expected_clock = expected_step
            if not (
                int(row.get("raw_step", -1)) == expected_step
                and str(row.get("clock_field")) == expected_clock_field
                and int(row.get("expected_clock", -1)) == expected_clock
                and int(row.get("recorded_clock", -1)) == expected_clock
                and row.get("zip_crc_ok") is True
                and checkpoint_path.is_file()
                and int(row.get("bytes", -1)) == checkpoint_path.stat().st_size
                and len(str(row.get("sha256", ""))) == 64
            ):
                checkpoint_contract = False
                break
    common_contract = (
        int(arguments["max_steps"]) == job.raw_steps
        and int(arguments["seed"]) == job.seed
        and str(arguments["scenario"]) == job.scenario
        and str(arguments["algo"]) == job.cli_algorithm
        and str(arguments.get("ego_control_profile", "direct"))
        == job.ego_control_profile
        and str(arguments.get("traffic_protocol", "source_all"))
        == job.traffic_protocol
        and str(arguments.get("episode_limit_profile", "source"))
        == job.episode_limit_profile
        and str(detailed["algorithm"]) == job.cli_algorithm
        and str(detailed["scenario"]) == job.scenario
        and int(detailed["evaluation_seed_start"]) == job.seed + 10_000
        and int(detailed["summary"]["episodes"]) == job.test_episodes
        and detailed["summary"] == evaluation
        and checkpoint_contract
    )
    provenance_errors = saved_evaluation_contract_errors(
        detailed,
        algorithm=job.cli_algorithm,
        scenario=job.scenario,
        traffic_protocol=job.traffic_protocol,
        episode_limit_profile=job.episode_limit_profile,
        episodes=job.test_episodes,
        evaluation_seed_start=job.seed + 10_000,
    )
    common_contract = common_contract and not provenance_errors
    if not common_contract:
        return False
    if job.implementation_id is not None:
        fidelity = argument_payload.get("implementation_fidelity", {})
        if str(fidelity.get("implementation_id")) != job.implementation_id:
            return False
    trained_raw_steps = int(detailed["trained_raw_steps"])
    collected_raw_steps = int(detailed["collected_training_raw_steps"])
    if job.cli_algorithm == "ppo":
        expected_collected = int(math.ceil(job.raw_steps / 512) * 512)
        expected_checkpoint = (
            job.raw_steps // job.checkpoint_frequency
        ) * job.checkpoint_frequency
        return (
            expected_checkpoint > 0
            and trained_raw_steps == expected_checkpoint
            and collected_raw_steps == expected_collected
            and int(detailed["source_test_checkpoint_step"])
            == expected_checkpoint
        )
    return (
        trained_raw_steps == job.raw_steps
        and collected_raw_steps == job.raw_steps
        and detailed.get("source_test_checkpoint_step") is None
    )


def locate_run(
    output_dir: Path, job: Job, protocol: dict[str, Any] | None = None
) -> tuple[str, Path, str]:
    """Locate an accepted primary run or a hash-validated predecessor run."""

    protocol = protocol or load_protocol()
    primary = run_directory(output_dir, job)
    if is_excluded(output_dir, job):
        return "excluded", primary, "primary"
    if _accepted_run(primary, job):
        return "complete", primary, "primary"
    if primary.exists():
        return "partial", primary, "primary"

    for predecessor in protocol.get("compatible_predecessor_results", []):
        if job.method not in predecessor.get("methods", []):
            continue
        root = (PROJECT_ROOT / predecessor["root"]).resolve()
        if root == output_dir.resolve() or not root.exists():
            continue
        snapshot = root / "protocol.snapshot.json"
        expected_hash = str(predecessor["protocol_sha256"]).upper()
        if not snapshot.is_file() or _sha256(snapshot) != expected_hash:
            raise RuntimeError(
                f"Compatible predecessor protocol mismatch: {root}"
            )
        directory = root / job.name
        marker = "NOT_FORMAL_SOURCE_FIDELITY.md"
        if (root / marker).is_file() or (directory / marker).is_file():
            continue
        if _accepted_run(directory, job):
            return "imported", directory, "compatible_predecessor"
        if directory.exists():
            return "partial", directory, "compatible_predecessor"
    return "pending", primary, "primary"


def classify(
    output_dir: Path, job: Job, protocol: dict[str, Any] | None = None
) -> str:
    return locate_run(output_dir, job, protocol)[0]


def command_for(args: argparse.Namespace, output_dir: Path, job: Job) -> list[str]:
    protocol = load_protocol()
    common = protocol["common_hyperparameters"]
    method = protocol["supported_methods"][job.method]
    action_repeat = int(
        method.get(
            "environment_action_repeat",
            common.get("off_policy_action_repeat", common.get("action_repeat", 3)),
        )
    )
    learning_rate = float(method.get("learning_rate", common["learning_rate"]))
    learning_starts = int(
        method.get(
            "learning_starts_raw_steps", common["learning_starts_raw_steps"]
        )
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo.py"),
        "--scenario",
        job.scenario,
        "--algo",
        job.cli_algorithm,
        "--max-steps",
        str(job.raw_steps),
        "--seed",
        str(job.seed),
        "--device",
        args.device,
        "--batch-size",
        str(common["batch_size"]),
        "--learning-rate",
        str(learning_rate),
        "--discount",
        str(common["discount"]),
        "--learning-starts",
        str(learning_starts),
        "--buffer-size",
        str(common["replay_buffer_capacity"]),
        "--neighbors",
        str(common["neighbors"]),
        "--history-steps",
        str(common["history_steps"]),
        "--path-length",
        str(common["path_length"]),
        "--action-repeat",
        str(action_repeat),
        "--ego-control-profile",
        job.ego_control_profile,
        "--traffic-protocol",
        job.traffic_protocol,
        "--episode-limit-profile",
        job.episode_limit_profile,
        "--checkpoint-freq",
        str(job.checkpoint_frequency),
        "--eval-freq",
        str(job.evaluation_frequency),
        "--eval-episodes",
        str(job.test_episodes),
        "--output-dir",
        str(output_dir.resolve()),
        "--model-name",
        job.name,
    ]
    return command


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def plan(args: argparse.Namespace, jobs: list[Job]) -> int:
    protocol = load_protocol()
    output_dir = args.output_dir.resolve()
    rows = []
    for job in jobs:
        status, directory, origin = locate_run(output_dir, job, protocol)
        rows.append(
            {
                "job": job.name,
                "status": status,
                "result_origin": origin,
                "result_directory": str(directory),
                "raw_steps": job.raw_steps,
                "test_episodes": job.test_episodes,
                "ego_control_profile": job.ego_control_profile,
                "traffic_protocol": job.traffic_protocol,
                "episode_limit_profile": job.episode_limit_profile,
                "implementation_id": job.implementation_id,
                "command": _display_command(command_for(args, output_dir, job)),
            }
        )
    print(json.dumps({"jobs": rows, "count": len(rows)}, indent=2))
    return 0


def preflight(args: argparse.Namespace, jobs: list[Job]) -> int:
    snapshot = ensure_protocol_snapshot(args.output_dir)
    output_dir = args.output_dir.resolve() / "_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preflight_report.json"
    seen: set[tuple[str, str]] = set()
    results = []
    for job in jobs:
        key = (job.cli_algorithm, job.scenario)
        if key in seen:
            continue
        seen.add(key)
        name = f"check__{job.method}__{job.scenario}__{int(time.time_ns())}"
        command = command_for(args, output_dir, job)
        command[command.index("--max-steps") + 1] = "1"
        command[command.index("--model-name") + 1] = name
        command.append("--check-only")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        results.append(
            {
                "method": job.method,
                "scenario": job.scenario,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if completed.returncode != 0:
            report = {
                "status": "failed",
                "passed": False,
                "protocol": str(PROTOCOL_PATH),
                "protocol_sha256": _sha256(snapshot),
                "jobs_requested": len(jobs),
                "unique_algorithm_scenario_pairs_expected": len(
                    {(item.cli_algorithm, item.scenario) for item in jobs}
                ),
                "unique_algorithm_scenario_pairs_checked": len(results),
                "results": results,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return completed.returncode
    report = {
        "status": "ok",
        "passed": True,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256(snapshot),
        "jobs_requested": len(jobs),
        "unique_algorithm_scenario_pairs_expected": len(
            {(item.cli_algorithm, item.scenario) for item in jobs}
        ),
        "unique_algorithm_scenario_pairs_checked": len(results),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _best_evaluation_command(
    args: argparse.Namespace,
    output_path: Path,
    model_path: Path,
    job: Job,
    protocol: dict[str, Any] | None = None,
) -> list[str]:
    protocol = protocol or load_protocol()
    common = protocol["common_hyperparameters"]
    method = protocol["supported_methods"][job.method]
    action_repeat = int(
        method.get(
            "environment_action_repeat",
            common.get("off_policy_action_repeat", 3),
        )
    )
    action_hold = method.get("policy_action_hold", 1)
    if isinstance(action_hold, dict):
        action_hold = action_hold.get(
            job.scenario, action_hold.get("default", 1)
        )
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_paper_sb3_sumo.py"),
        "--model",
        str(model_path.resolve()),
        "--scenario",
        job.scenario,
        "--algo",
        job.cli_algorithm,
        "--episodes",
        str(job.test_episodes),
        "--seed",
        str(job.seed + 10_000),
        "--device",
        args.device,
        "--neighbors",
        str(common["neighbors"]),
        "--history-steps",
        str(common["history_steps"]),
        "--path-length",
        str(common["path_length"]),
        "--action-repeat",
        str(action_repeat),
        "--discount",
        str(common["discount"]),
        "--policy-action-hold",
        str(int(action_hold)),
        "--ego-control-profile",
        job.ego_control_profile,
        "--traffic-protocol",
        job.traffic_protocol,
        "--episode-limit-profile",
        job.episode_limit_profile,
        "--output",
        str(output_path.resolve()),
    ]


def evaluate_best(args: argparse.Namespace, jobs: list[Job]) -> int:
    """Evaluate prospectively captured best-training-success policies."""

    protocol = load_protocol()
    output_dir = args.output_dir.resolve()
    ensure_protocol_snapshot(output_dir)
    rows: list[dict[str, Any]] = []
    failures = 0
    for job in jobs:
        status, directory, origin = locate_run(output_dir, job, protocol)
        if status not in ("complete", "imported"):
            rows.append({"job": job.name, "status": f"training_{status}"})
            continue
        model_path = directory / "best_training_success_model.zip"
        metadata_path = directory / "best_training_success.json"
        detailed_path = directory / "best_training_success_evaluation_detailed.json"
        summary_path = directory / "best_training_success_evaluation.json"
        if not model_path.is_file() or not metadata_path.is_file():
            rows.append(
                {
                    "job": job.name,
                    "status": "best_checkpoint_unavailable",
                    "result_origin": origin,
                }
            )
            continue
        if not detailed_path.is_file():
            completed = subprocess.run(
                _best_evaluation_command(
                    args, detailed_path, model_path, job, protocol
                ),
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0:
                failures += 1
                rows.append(
                    {
                        "job": job.name,
                        "status": "evaluation_failed",
                        "returncode": completed.returncode,
                        "stderr": completed.stderr[-2000:],
                    }
                )
                if not args.continue_on_error:
                    break
                continue
        detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance_errors = saved_evaluation_contract_errors(
            detailed,
            algorithm=job.cli_algorithm,
            scenario=job.scenario,
            traffic_protocol=job.traffic_protocol,
            episode_limit_profile=job.episode_limit_profile,
            episodes=job.test_episodes,
            evaluation_seed_start=job.seed + 10_000,
        )
        contract = (
            str(detailed.get("algorithm")) == job.cli_algorithm
            and str(detailed.get("scenario")) == job.scenario
            and int(detailed.get("evaluation_seed_start", -1))
            == job.seed + 10_000
            and int(detailed.get("summary", {}).get("episodes", -1))
            == job.test_episodes
            and int(detailed.get("trained_raw_steps", -1))
            == int(metadata.get("clock_value", -2))
            and not provenance_errors
        )
        if not contract:
            failures += 1
            rows.append({"job": job.name, "status": "artifact_contract_failed"})
            if not args.continue_on_error:
                break
            continue
        summary_payload = {
            "selection": metadata,
            "evaluation": detailed["summary"],
            "successful_episodes": detailed["successful_episodes"],
            "mean_success_completion_time_seconds": detailed[
                "mean_success_completion_time_seconds"
            ],
            "std_success_completion_time_seconds": detailed[
                "std_success_completion_time_seconds"
            ],
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "job": job.name,
                "status": "complete",
                "selected_clock": metadata["clock_value"],
                "training_success_rate": metadata["success_rate_last_20"],
                "test_success_rate": detailed["summary"]["success_rate"],
            }
        )
    print(json.dumps({"jobs": rows, "failures": failures}, indent=2))
    return 0 if failures == 0 else 1


def execute(args: argparse.Namespace, jobs: list[Job]) -> int:
    protocol = load_protocol()
    output_dir = args.output_dir.resolve()
    ensure_protocol_snapshot(output_dir)
    state_path = output_dir / "orchestrator_state.json"
    attempted = 0
    failures: list[dict[str, Any]] = []
    runnable_jobs: list[Job] = []
    for job in jobs:
        status, located_directory, _ = locate_run(output_dir, job, protocol)
        if status == "excluded":
            print(f"SKIP excluded {job.name}", flush=True)
            continue
        if status in ("complete", "imported"):
            print(f"SKIP complete {job.name}", flush=True)
            continue
        if status == "partial":
            failure = {
                "job": job.name,
                "reason": "partial directory exists; preserved for inspection",
                "directory": str(located_directory),
            }
            failures.append(failure)
            print(f"BLOCK partial {job.name}", flush=True)
            if not args.continue_on_error:
                break
            continue
        runnable_jobs.append(job)

    if args.schedule == "diversified":
        runnable_jobs = _diversified_job_order(runnable_jobs)
    if args.max_runs is not None:
        runnable_jobs = runnable_jobs[: args.max_runs]
    pending_jobs = runnable_jobs
    attempted = len(pending_jobs)

    started_unix = time.time()
    completed_jobs: list[str] = []
    running_names: set[str] = set()

    def write_running_state() -> None:
        state = {
            "status": "running",
            "pid": os.getpid(),
            "workers": args.workers,
            "schedule": args.schedule,
            "running_jobs": sorted(running_names),
            "completed_jobs": completed_jobs,
            "failures": failures,
            "started_unix": started_unix,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def run_job(job: Job) -> int:
        command = command_for(args, output_dir, job)
        print(f"RUN {job.name}", flush=True)
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode

    job_iterator = iter(pending_jobs)
    stop_scheduling = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        running: dict[Future[int], Job] = {}

        def submit_one() -> bool:
            try:
                next_job = next(job_iterator)
            except StopIteration:
                return False
            running_names.add(next_job.name)
            running[executor.submit(run_job, next_job)] = next_job
            return True

        for _ in range(min(args.workers, len(pending_jobs))):
            submit_one()
        if running:
            write_running_state()

        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                running_names.discard(job.name)
                returncode = future.result()
                if returncode != 0:
                    failure = {"job": job.name, "returncode": returncode}
                    failures.append(failure)
                    print(f"FAIL {job.name}: {returncode}", flush=True)
                    if not args.continue_on_error:
                        stop_scheduling = True
                else:
                    completed_jobs.append(job.name)
                    print(f"DONE {job.name}", flush=True)
                if not stop_scheduling:
                    submit_one()
            write_running_state()
    counts = status_counts(output_dir, jobs, protocol)
    final_state = {
        "status": "finished" if not failures else "finished_with_failures",
        "attempted": attempted,
        "workers": args.workers,
        "schedule": args.schedule,
        "completed_jobs": completed_jobs,
        "counts": counts,
        "failures": failures,
        "finished_unix": time.time(),
    }
    state_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
    print(json.dumps(final_state, indent=2), flush=True)
    return 0 if not failures else 1


def status_counts(
    output_dir: Path,
    jobs: Iterable[Job],
    protocol: dict[str, Any] | None = None,
) -> dict[str, int]:
    protocol = protocol or load_protocol()
    counts = {
        "complete": 0,
        "imported": 0,
        "partial": 0,
        "pending": 0,
        "excluded": 0,
    }
    for job in jobs:
        counts[classify(output_dir, job, protocol)] += 1
    return counts


def show_status(args: argparse.Namespace, jobs: list[Job]) -> int:
    protocol = load_protocol()
    output_dir = args.output_dir.resolve()
    details = []
    for job in jobs:
        status, directory, origin = locate_run(output_dir, job, protocol)
        details.append(
            {
                "job": job.name,
                "status": status,
                "result_origin": origin,
                "result_directory": str(directory),
            }
        )
    state_path = output_dir / "orchestrator_state.json"
    state = None
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "counts": status_counts(output_dir, jobs, protocol),
                "state": state,
                "jobs": details,
            },
            indent=2,
        )
    )
    return 0


def _training_curve(run_dir: Path, action_repeat: int) -> list[dict[str, Any]]:
    monitor_path = run_dir / "train_monitor.csv"
    if not monitor_path.is_file():
        return []
    episodes = []
    cumulative_decisions = 0
    cumulative_raw_steps = 0
    with monitor_path.open("r", encoding="utf-8") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in rows:
            length = int(row["l"])
            cumulative_decisions += length
            episode_raw_steps = int(
                float(row.get("raw_simulation_steps") or length * action_repeat)
            )
            cumulative_raw_steps += episode_raw_steps
            episode_return = float(row["r"])
            episodes.append(
                {
                    "raw_steps": cumulative_raw_steps,
                    "decision_steps": cumulative_decisions,
                    "episode_return": episode_return,
                    "success": int(episode_return > 0.0),
                    "failure_penalty": int(episode_return < 0.0),
                }
            )
    for index, episode in enumerate(episodes):
        window = episodes[max(0, index - 19) : index + 1]
        # Released runners always divide by 20, including before 20 episodes
        # exist, so the missing prehistory contributes zeros.
        episode["success_rate_last_20"] = (
            sum(item["success"] for item in window) / 20.0
        )
    return episodes


def _resample_training_curve(
    episodes: list[dict[str, Any]],
    max_raw_steps: int,
    *,
    interval_raw_steps: int = 200,
    ema_previous_weight: float = 0.99,
) -> list[dict[str, Any]]:
    """Forward-sample completed episodes, then smooth each seed independently."""

    if interval_raw_steps <= 0:
        raise ValueError("interval_raw_steps must be positive")
    if not 0.0 <= ema_previous_weight < 1.0:
        raise ValueError("ema_previous_weight must be in [0, 1)")
    if not episodes:
        return []
    output: list[dict[str, Any]] = []
    episode_index = 0
    latest_success_rate: float | None = None
    ema: float | None = None
    for raw_steps in range(interval_raw_steps, max_raw_steps + 1, interval_raw_steps):
        while (
            episode_index < len(episodes)
            and int(episodes[episode_index]["raw_steps"]) <= raw_steps
        ):
            latest_success_rate = float(
                episodes[episode_index]["success_rate_last_20"]
            )
            episode_index += 1
        # Do not invent a value before the first episode has completed.
        if latest_success_rate is None:
            continue
        ema = (
            latest_success_rate
            if ema is None
            else ema_previous_weight * ema
            + (1.0 - ema_previous_weight) * latest_success_rate
        )
        output.append(
            {
                "raw_steps": raw_steps,
                "success_rate_last_20": latest_success_rate,
                "success_rate_ema_0_99": ema,
            }
        )
    return output


def _aggregate_resampled_curves(
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in curves:
        key = (
            str(row["profile"]),
            str(row["method"]),
            str(row["scenario"]),
            int(row["raw_steps"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (profile, method, scenario, raw_steps), members in sorted(groups.items()):
        raw_values = [float(row["success_rate_last_20"]) for row in members]
        ema_values = [float(row["success_rate_ema_0_99"]) for row in members]
        output.append(
            {
                "profile": profile,
                "method": method,
                "scenario": scenario,
                "raw_steps": raw_steps,
                "seeds_available": len(members),
                "success_rate_last_20_mean": mean(raw_values),
                "success_rate_last_20_std": (
                    pstdev(raw_values) if len(raw_values) > 1 else 0.0
                ),
                "success_rate_ema_0_99_mean": mean(ema_values),
                "success_rate_ema_0_99_std": (
                    pstdev(ema_values) if len(ema_values) > 1 else 0.0
                ),
            }
        )
    return output


def summarize(args: argparse.Namespace, jobs: list[Job]) -> int:
    protocol = load_protocol()
    paper_references = load_paper_references()
    output_dir = args.output_dir.resolve()
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    resampled_curves: list[dict[str, Any]] = []
    for job in jobs:
        status, run_dir, origin = locate_run(output_dir, job, protocol)
        if status not in ("complete", "imported"):
            continue
        evaluation_path = run_dir / "final_evaluation.json"
        if not evaluation_path.is_file():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        record = {
            "profile": job.profile,
            "method": job.method,
            "scenario": job.scenario,
            "seed": job.seed,
            "result_origin": origin,
            "result_directory": str(run_dir),
            **evaluation,
        }
        detailed_path = run_dir / "paper_evaluation_detailed.json"
        if detailed_path.is_file():
            detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
            detailed_summary = detailed["summary"]
            for metric in (
                "episodes",
                "mean_return",
                "std_return",
                "mean_decision_steps",
                "mean_raw_steps",
                "success_rate",
                "collision_rate",
                "off_route_rate",
                "timeout_rate",
            ):
                if detailed_summary[metric] != evaluation[metric]:
                    raise ValueError(
                        f"Detailed/final evaluation mismatch for {job.name}: {metric}"
                    )
            record.update(
                {
                    "successful_episodes": detailed["successful_episodes"],
                    "mean_success_completion_time_seconds": detailed[
                        "mean_success_completion_time_seconds"
                    ],
                    "std_success_completion_time_seconds": detailed[
                        "std_success_completion_time_seconds"
                    ],
                }
            )
        records.append(record)
        action_repeat = int(
            protocol["supported_methods"][job.method].get(
                "environment_action_repeat",
                protocol["common_hyperparameters"].get(
                    "off_policy_action_repeat", 3
                ),
            )
        )
        episode_curve = _training_curve(run_dir, action_repeat)
        for point in episode_curve:
            curves.append(
                {
                    "profile": job.profile,
                    "method": job.method,
                    "scenario": job.scenario,
                    "seed": job.seed,
                    **point,
                }
            )
        for point in _resample_training_curve(episode_curve, job.raw_steps):
            resampled_curves.append(
                {
                    "profile": job.profile,
                    "method": job.method,
                    "scenario": job.scenario,
                    "seed": job.seed,
                    **point,
                }
            )

    aggregate: list[dict[str, Any]] = []
    metric_names = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_raw_steps",
        "mean_return",
        "mean_success_completion_time_seconds",
    )
    groups = sorted({(row["method"], row["scenario"]) for row in records})
    for method, scenario in groups:
        members = [
            row
            for row in records
            if row["method"] == method and row["scenario"] == scenario
        ]
        item: dict[str, Any] = {
            "profile": args.profile,
            "method": method,
            "scenario": scenario,
            "seeds_completed": len(members),
        }
        for metric in metric_names:
            values = [
                float(member[metric])
                for member in members
                if member.get(metric) is not None
            ]
            if not values:
                continue
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        completion_members = [
            member
            for member in members
            if member.get("mean_success_completion_time_seconds") is not None
            and member.get("std_success_completion_time_seconds") is not None
            and int(member.get("successful_episodes", 0)) > 0
        ]
        if completion_members:
            successful_total = sum(
                int(member["successful_episodes"])
                for member in completion_members
            )
            pooled_mean = sum(
                int(member["successful_episodes"])
                * float(member["mean_success_completion_time_seconds"])
                for member in completion_members
            ) / successful_total
            pooled_variance = sum(
                int(member["successful_episodes"])
                * (
                    float(member["std_success_completion_time_seconds"]) ** 2
                    + (
                        float(member["mean_success_completion_time_seconds"])
                        - pooled_mean
                    )
                    ** 2
                )
                for member in completion_members
            ) / successful_total
            item["successful_episodes_total"] = successful_total
            item["success_completion_time_seconds_pooled_mean"] = pooled_mean
            item["success_completion_time_seconds_pooled_std"] = pooled_variance ** 0.5
        reference = protocol["paper_reference_percent"].get(scenario, {}).get(
            protocol["supported_methods"][method]["paper_label"]
        )
        if reference is not None:
            item["paper_success_rate"] = reference[0] / 100.0
            item["paper_collision_rate"] = reference[1] / 100.0
            item["paper_stagnation_rate"] = reference[2] / 100.0
            item["success_rate_delta_from_paper"] = (
                item["success_rate_mean"] - item["paper_success_rate"]
            )
        completion_reference = paper_references["completion_time_seconds"].get(
            scenario, {}
        ).get(protocol["supported_methods"][method]["paper_label"])
        if completion_reference is not None:
            item["paper_completion_time_seconds_mean"] = float(
                completion_reference[0]
            )
            item["paper_completion_time_seconds_std"] = float(
                completion_reference[1]
            )
            if item.get("success_completion_time_seconds_pooled_mean") is not None:
                item["completion_time_mean_delta_from_paper_seconds"] = (
                    item["success_completion_time_seconds_pooled_mean"]
                    - item["paper_completion_time_seconds_mean"]
                )
        aggregate.append(item)

    payload = {
        "profile": args.profile,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "paper_reference_sha256": _sha256(PAPER_REFERENCE_PATH),
        "jobs_requested": len(jobs),
        "runs_completed": len(records),
        "curve_processing": {
            "episode_metric": "sum of successes over the latest up-to-20 completed episodes divided by the fixed source denominator 20",
            "resample_interval_raw_steps": 200,
            "resample_rule": "last completed episode at or before the grid point; omit grid points before the first completed episode",
            "ema_order": "resample each seed, apply EMA per seed, then compute cross-seed population mean/std",
            "ema_formula": "ema_t = 0.99 * ema_(t-1) + 0.01 * x_t; first ema equals first available x",
        },
        "aggregate": aggregate,
        "runs": records,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csv(summary_dir / "runs.csv", records)
    _write_csv(summary_dir / "aggregate.csv", aggregate)
    _write_csv(summary_dir / "training_curves.csv", curves)
    _write_csv(summary_dir / "training_curves_resampled.csv", resampled_curves)
    _write_csv(
        summary_dir / "training_curves_aggregate.csv",
        _aggregate_resampled_curves(resampled_curves),
    )
    _write_markdown(summary_dir / "summary.md", aggregate, len(records), len(jobs))
    print(json.dumps(payload, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path, aggregate: list[dict[str, Any]], completed: int, requested: int
) -> None:
    lines = [
        "# SB3 + SUMO reproduction summary",
        "",
        f"Completed runs: {completed}/{requested}",
        "",
        "| Method | Scenario | Seeds | Success | Collision | Timeout | SUMO successful time (s) | Paper time (s) | Time delta (s) | Paper success | Success delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        paper = row.get("paper_success_rate")
        delta = row.get("success_rate_delta_from_paper")
        paper_time = row.get("paper_completion_time_seconds_mean")
        paper_time_std = row.get("paper_completion_time_seconds_std")
        time_delta = row.get("completion_time_mean_delta_from_paper_seconds")
        lines.append(
            "| {method} | {scenario} | {seeds_completed} | {success_rate_mean:.3f} | "
            "{collision_rate_mean:.3f} | {timeout_rate_mean:.3f} | {completion} | "
            "{paper_time} | {time_delta} | {paper} | {delta} |".format(
                **row,
                completion=(
                    "-"
                    if row.get("success_completion_time_seconds_pooled_mean") is None
                    else "{:.3f} ± {:.3f}".format(
                        row["success_completion_time_seconds_pooled_mean"],
                        row["success_completion_time_seconds_pooled_std"],
                    )
                ),
                paper_time=(
                    "-"
                    if paper_time is None
                    else f"{paper_time:.3f} ± {paper_time_std:.3f}"
                ),
                time_delta="-" if time_delta is None else f"{time_delta:+.3f}",
                paper="-" if paper is None else f"{paper:.3f}",
                delta="-" if delta is None else f"{delta:+.3f}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    protocol = load_protocol()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=(
            "plan",
            "preflight",
            "run",
            "status",
            "evaluate-best",
            "summarize",
        ),
    )
    result.add_argument("--protocol-path", type=Path, default=PROTOCOL_PATH)
    result.add_argument("--profile", choices=tuple(protocol["profiles"]), default="paper")
    result.add_argument(
        "--methods",
        default=next(iter(protocol["supported_methods"])),
        help="Comma-separated supported methods or 'all'",
    )
    result.add_argument(
        "--scenarios", default="all", help="Comma-separated scenarios or 'all'"
    )
    result.add_argument("--seeds", default="all", help="Comma-separated integer seeds")
    result.add_argument("--device", default="cuda")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--max-runs", type=int, default=None)
    result.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of isolated training subprocesses to run concurrently",
    )
    result.add_argument(
        "--schedule",
        choices=("protocol", "diversified"),
        default="protocol",
        help="Protocol order or scenario/method-interleaved execution order",
    )
    result.add_argument("--continue-on-error", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    global PROTOCOL_PATH
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--protocol-path", type=Path, default=PROTOCOL_PATH)
    preliminary_args, _ = preliminary.parse_known_args(argv)
    PROTOCOL_PATH = preliminary_args.protocol_path.resolve()
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Protocol does not exist: {PROTOCOL_PATH}")
    args = parser().parse_args(argv)
    protocol = load_protocol()
    jobs = make_jobs(args, protocol)
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    handlers = {
        "plan": plan,
        "preflight": preflight,
        "run": execute,
        "status": show_status,
        "evaluate-best": evaluate_best,
        "summarize": summarize,
    }
    return handlers[args.command](args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
