"""Validate, execute, and aggregate the topology-scene experiment contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "experiments" / "topo_scene" / "experiment_contract.yaml"
ALGORITHMS = {
    "scene_rep",
    "temporal_graph",
    "topo_scene",
    "topo_scene_balanced",
}


class ExperimentContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentContractError(message)


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Experiment contract must be a mapping")
    return payload


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version") == "topo-scene.experiment-contract/v1",
        "Unexpected experiment schema_version",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    for field in (
        "methods",
        "phases",
        "fixed_controls",
        "metrics",
        "statistics",
        "development_gate",
        "result_artifacts",
    ):
        _require(field in contract, f"Missing experiment field {field!r}")
    methods = contract["methods"]
    _require(isinstance(methods, dict) and methods, "methods must be a non-empty mapping")
    for name, method in methods.items():
        _require(isinstance(method, dict), f"Method {name!r} must be a mapping")
        _require(
            method.get("cli_algorithm") in ALGORITHMS,
            f"Method {name!r} has unsupported cli_algorithm",
        )
        _require(method.get("implementation_id"), f"Method {name!r} needs implementation_id")
    phases = contract["phases"]
    _require(isinstance(phases, dict) and phases, "phases must be a non-empty mapping")
    for name, phase in phases.items():
        _require(isinstance(phase, dict), f"Phase {name!r} must be a mapping")
        for list_field in ("methods", "scenarios", "seeds"):
            _require(
                isinstance(phase.get(list_field), list) and phase[list_field],
                f"Phase {name!r} needs non-empty {list_field}",
            )
        unknown = set(phase["methods"]) - set(methods)
        _require(not unknown, f"Phase {name!r} references unknown methods {sorted(unknown)}")
        for integer_field in (
            "raw_training_steps",
            "learning_starts_raw_steps",
            "checkpoint_frequency_raw_steps",
            "evaluation_frequency_raw_steps",
            "final_evaluation_episodes",
        ):
            _require(
                isinstance(phase.get(integer_field), int) and phase[integer_field] >= 0,
                f"Phase {name!r}.{integer_field} must be a non-negative integer",
            )
        _require(phase["raw_training_steps"] > 0, f"Phase {name!r} needs a positive budget")
        _require(
            phase["final_evaluation_episodes"] > 0,
            f"Phase {name!r} needs final evaluation episodes",
        )
    controls = contract["fixed_controls"]
    _require(int(controls.get("latent_dim", -1)) == 128, "latent_dim fairness must be 128")
    _require(int(controls.get("action_repeat", -1)) == 3, "action_repeat fairness must be 3")
    _require(
        controls.get("deterministic_final_policy") is True,
        "Final evaluation must be deterministic",
    )
    return contract


@dataclass(frozen=True)
class Job:
    phase: str
    method: str
    algorithm: str
    implementation_id: str
    scenario: str
    seed: int
    raw_steps: int
    learning_starts: int
    checkpoint_frequency: int
    evaluation_frequency: int
    evaluation_episodes: int
    traffic_protocol: str
    episode_limit_profile: str

    @property
    def name(self) -> str:
        return f"{self.phase}__{self.method}__{self.scenario}__seed{self.seed}"


def _selection(value: str | None, allowed: Iterable[Any]) -> list[Any]:
    allowed_values = list(allowed)
    if value is None or value == "all":
        return allowed_values
    requested: list[Any] = []
    for token in value.split(","):
        converted: Any = int(token) if allowed_values and isinstance(allowed_values[0], int) else token
        _require(converted in allowed_values, f"Value {converted!r} not in {allowed_values}")
        requested.append(converted)
    return requested


def make_jobs(
    contract: dict[str, Any],
    phase_name: str,
    *,
    methods: str | None = None,
    scenarios: str | None = None,
    seeds: str | None = None,
) -> list[Job]:
    phases = contract["phases"]
    _require(phase_name in phases, f"Unknown phase {phase_name!r}")
    phase = phases[phase_name]
    selected_methods = _selection(methods, phase["methods"])
    selected_scenarios = _selection(scenarios, phase["scenarios"])
    selected_seeds = _selection(seeds, phase["seeds"])
    jobs: list[Job] = []
    for method_name in selected_methods:
        method = contract["methods"][method_name]
        for scenario in selected_scenarios:
            for seed in selected_seeds:
                jobs.append(
                    Job(
                        phase=phase_name,
                        method=method_name,
                        algorithm=str(method["cli_algorithm"]),
                        implementation_id=str(method["implementation_id"]),
                        scenario=str(scenario),
                        seed=int(seed),
                        raw_steps=int(phase["raw_training_steps"]),
                        learning_starts=int(phase["learning_starts_raw_steps"]),
                        checkpoint_frequency=int(phase["checkpoint_frequency_raw_steps"]),
                        evaluation_frequency=int(phase["evaluation_frequency_raw_steps"]),
                        evaluation_episodes=int(phase["final_evaluation_episodes"]),
                        traffic_protocol=str(phase["traffic_protocol"]),
                        episode_limit_profile=str(phase["episode_limit_profile"]),
                    )
                )
    return jobs


def output_root(contract: dict[str, Any]) -> Path:
    path = Path(str(contract["output_root"]))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_directory(contract: dict[str, Any], job: Job) -> Path:
    return output_root(contract) / job.phase / "runs" / job.name


def command_for(contract: dict[str, Any], job: Job, device: str) -> list[str]:
    controls = contract["fixed_controls"]
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo.py"),
        "--scenario",
        job.scenario,
        "--algo",
        job.algorithm,
        "--max-steps",
        str(job.raw_steps),
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--batch-size",
        str(controls["batch_size"]),
        "--learning-rate",
        str(controls["learning_rate"]),
        "--discount",
        str(controls["discount"]),
        "--learning-starts",
        str(job.learning_starts),
        "--buffer-size",
        str(controls["buffer_size"]),
        "--action-repeat",
        str(controls["action_repeat"]),
        "--checkpoint-freq",
        str(job.checkpoint_frequency),
        "--eval-freq",
        str(job.evaluation_frequency),
        "--eval-episodes",
        str(job.evaluation_episodes),
        "--traffic-protocol",
        job.traffic_protocol,
        "--episode-limit-profile",
        job.episode_limit_profile,
        "--output-dir",
        str(output_root(contract) / job.phase / "runs"),
        "--model-name",
        job.name,
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_run(contract: dict[str, Any], job: Job) -> bool:
    directory = run_directory(contract, job)
    required = contract["result_artifacts"]["run_required"]
    if any(not (directory / name).is_file() for name in required):
        return False
    arguments = _load_json(directory / "arguments.json")
    requested = arguments["requested_raw_steps"]
    fidelity = arguments["implementation_fidelity"]
    detailed = _load_json(directory / "paper_evaluation_detailed.json")
    return bool(
        requested["algo"] == job.algorithm
        and requested["scenario"] == job.scenario
        and int(requested["seed"]) == job.seed
        and int(requested["max_steps"]) == job.raw_steps
        and fidelity["implementation_id"] == job.implementation_id
        and int(detailed["trained_raw_steps"]) == job.raw_steps
        and detailed["algorithm"] == job.algorithm
        and detailed["scenario"] == job.scenario
    )


def _attention_command(contract: dict[str, Any], job: Job, device: str) -> list[str]:
    directory = run_directory(contract, job)
    gate = contract["development_gate"]["attention_locality"]
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "diagnose_topology_attention.py"),
        "--model",
        str(directory / "final_model.zip"),
        "--scenario",
        job.scenario,
        "--episodes",
        str(min(5, job.evaluation_episodes)),
        "--seed",
        str(job.seed + 30_000),
        "--device",
        device,
        "--traffic-protocol",
        job.traffic_protocol,
        "--episode-limit-profile",
        job.episode_limit_profile,
        "--top1-p90-max-m",
        str(gate["top1_distance_p90_max_m"]),
        "--minimum-mass-within-40m",
        str(gate["minimum_mean_mass_within_40m"]),
        "--output",
        str(directory / "attention_diagnostics.json"),
    ]


def execute_jobs(
    contract: dict[str, Any],
    jobs: list[Job],
    *,
    device: str,
    dry_run: bool,
) -> int:
    if jobs and jobs[0].phase == "confirmation":
        gate_path = output_root(contract) / "development" / "gate_decision.json"
        _require(gate_path.is_file(), "Confirmation requires a completed development gate")
        gate = _load_json(gate_path)
        _require(gate.get("decision") == "pass", "Development gate did not pass")
    plan = []
    for job in jobs:
        plan.append(
            {
                "name": job.name,
                "accepted": accepted_run(contract, job),
                "run_directory": str(run_directory(contract, job)),
                "command": command_for(contract, job, device),
            }
        )
    if dry_run:
        print(json.dumps({"jobs": plan}, ensure_ascii=False, indent=2))
        return 0

    failures: list[str] = []
    for row, job in zip(plan, jobs):
        directory = run_directory(contract, job)
        if row["accepted"]:
            print(json.dumps({"job": job.name, "status": "accepted-existing"}))
        else:
            if directory.exists():
                raise FileExistsError(
                    f"Incomplete/non-matching run directory already exists: {directory}"
                )
            directory.parent.mkdir(parents=True, exist_ok=True)
            log_path = directory.parent / f"{job.name}.launcher.log"
            print(json.dumps({"job": job.name, "status": "running", "log": str(log_path)}))
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    row["command"],
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            if completed.returncode != 0 or not accepted_run(contract, job):
                failures.append(job.name)
                print(json.dumps({"job": job.name, "status": "failed", "returncode": completed.returncode}))
                continue
            print(json.dumps({"job": job.name, "status": "trained"}))

        if job.algorithm in ("topo_scene", "topo_scene_balanced"):
            attention_output = directory / "attention_diagnostics.json"
            if not attention_output.is_file():
                attention_log = directory / "attention_diagnostics.log"
                with attention_log.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        _attention_command(contract, job, device),
                        cwd=PROJECT_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                if completed.returncode != 0 or not attention_output.is_file():
                    failures.append(f"{job.name}:attention")
                    continue
            print(json.dumps({"job": job.name, "status": "attention-diagnosed"}))
    return 1 if failures else 0


def _bootstrap_interval(
    values: list[float], resamples: int, seed: int
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _run_row(contract: dict[str, Any], job: Job) -> dict[str, Any]:
    directory = run_directory(contract, job)
    detailed = _load_json(directory / "paper_evaluation_detailed.json")
    summary = detailed["summary"]
    performance = _load_json(directory / "performance_profile.json")
    diagnostics = _load_json(directory / "training_diagnostics.json")
    attention_path = directory / "attention_diagnostics.json"
    attention = _load_json(attention_path) if attention_path.is_file() else None
    completion = detailed.get("mean_success_completion_time_seconds")
    return {
        "phase": job.phase,
        "method": job.method,
        "algorithm": job.algorithm,
        "implementation_id": job.implementation_id,
        "scenario": job.scenario,
        "seed": job.seed,
        "raw_steps": job.raw_steps,
        "run_directory": str(directory),
        "success_rate": float(summary["success_rate"]),
        "collision_rate": float(summary["collision_rate"]),
        "off_route_rate": float(summary["off_route_rate"]),
        "timeout_rate": float(summary["timeout_rate"]),
        "mean_return": float(summary["mean_return"]),
        "successful_completion_time_seconds": (
            float(completion) if completion is not None else None
        ),
        "parameter_count": int(performance["parameters"]["model_plus_representation"]),
        "train_ms_per_gradient_step": diagnostics["statistics"].get(
            "performance/train_ms_per_gradient_step", {}
        ).get("mean"),
        "inference_ms_per_action": float(
            performance["inference"]["mean_milliseconds_per_action"]
        ),
        "peak_gpu_memory_mb": performance["peak_gpu_memory_mb"],
        "training_diagnostics": diagnostics["statistics"],
        "attention_diagnostics": attention,
        "episode_records": detailed["episode_records"],
    }


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "population_std": None}
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "population_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_rows(contract: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["method"]), []).append(row)
    statistics_contract = contract["statistics"]
    resamples = int(statistics_contract["bootstrap_resamples"])
    bootstrap_seed = int(statistics_contract["bootstrap_seed"])
    output: list[dict[str, Any]] = []
    metric_names = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "successful_completion_time_seconds",
        "parameter_count",
        "train_ms_per_gradient_step",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
    )
    for group_index, ((scenario, method), members) in enumerate(sorted(grouped.items())):
        metrics = {
            metric: _mean_std(
                [float(row[metric]) for row in members if row.get(metric) is not None]
            )
            for metric in metric_names
        }
        episode_rates = {
            "success_rate": [float(record["success"]) for row in members for record in row["episode_records"]],
            "collision_rate": [float(record["collision"]) for row in members for record in row["episode_records"]],
            "off_route_rate": [float(record["off_route"]) for row in members for record in row["episode_records"]],
            "timeout_rate": [float(record["timeout"]) for row in members for record in row["episode_records"]],
        }
        intervals = {}
        for metric, values in episode_rates.items():
            lower, upper = _bootstrap_interval(
                values, resamples, bootstrap_seed + group_index
            )
            intervals[metric] = {"lower": lower, "upper": upper, "episodes": len(values)}
        output.append(
            {
                "scenario": scenario,
                "method": method,
                "seeds": sorted(int(row["seed"]) for row in members),
                "metrics": metrics,
                "bootstrap_95_ci": intervals,
            }
        )
    return output


def development_gate(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_method = {row["method"]: row for row in rows if row["scenario"] == "left_turn"}
    if not {"scene_rep", "topo_scene"}.issubset(by_method):
        return {
            "computed_from_real_runs": True,
            "decision": "incomplete",
            "missing_methods": sorted({"scene_rep", "topo_scene"} - set(by_method)),
        }
    baseline = by_method["scene_rep"]
    full = by_method["topo_scene"]
    gate = contract["development_gate"]
    success_delta = full["success_rate"] - baseline["success_rate"]
    collision_delta = full["collision_rate"] - baseline["collision_rate"]
    return_delta = full["mean_return"] - baseline["mean_return"]
    finite = _finite_tree(
        {
            "baseline": {key: baseline[key] for key in ("success_rate", "collision_rate", "mean_return")},
            "full": {key: full[key] for key in ("success_rate", "collision_rate", "mean_return")},
            "diagnostics": full["training_diagnostics"],
        }
    )
    positive = bool(
        success_delta >= float(gate["positive_effect_any"]["minimum_success_rate_increase"])
        or return_delta > float(gate["positive_effect_any"]["minimum_return_increase"])
    )
    collision_ok = collision_delta <= float(gate["maximum_collision_rate_increase"])
    attention = full.get("attention_diagnostics")
    attention_ok = bool(
        attention is not None and attention["locality_gate"]["passed"] is True
    )
    if not finite:
        decision = "stop"
    elif positive and collision_ok and attention_ok:
        decision = "pass"
    else:
        decision = "diagnose"
    return {
        "computed_from_real_runs": True,
        "decision": decision,
        "comparison": {
            "topo_scene_minus_scene_rep": {
                "success_rate": success_delta,
                "collision_rate": collision_delta,
                "mean_return": return_delta,
            }
        },
        "checks": {
            "finite": finite,
            "positive_effect": positive,
            "collision_guardrail": collision_ok,
            "attention_locality": attention_ok,
        },
        "thresholds": gate,
        "claim_allowed": decision == "pass",
    }


def summarize_phase(
    contract: dict[str, Any], phase: str, jobs: list[Job]
) -> int:
    complete = [job for job in jobs if accepted_run(contract, job)]
    missing = [job.name for job in jobs if job not in complete]
    rows = [_run_row(contract, job) for job in complete]
    aggregates = aggregate_rows(contract, rows)
    phase_root = output_root(contract) / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "topo-scene.results/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "phase": phase,
        "expected_jobs": len(jobs),
        "complete_jobs": len(complete),
        "missing_jobs": missing,
        "per_run": rows,
        "aggregates": aggregates,
    }
    (phase_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "phase",
        "method",
        "scenario",
        "seed",
        "raw_steps",
        "success_rate",
        "collision_rate",
        "mean_return",
        "off_route_rate",
        "timeout_rate",
        "successful_completion_time_seconds",
        "parameter_count",
        "train_ms_per_gradient_step",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
        "run_directory",
    ]
    with (phase_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})
    if phase == "development":
        gate = development_gate(contract, rows)
        (phase_root / "gate_decision.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for name in ("plan", "run", "summarize"):
        child = subparsers.add_parser(name)
        child.add_argument(
            "--phase",
            choices=("pilot", "development", "diagnostic", "iteration", "confirmation"),
            required=True,
        )
        child.add_argument("--methods", default=None)
        child.add_argument("--scenarios", default=None)
        child.add_argument("--seeds", default=None)
        if name in ("plan", "run"):
            child.add_argument("--device", default="cuda")
        if name == "run":
            child.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = validate_contract(load_contract(contract_path))
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "contract": str(contract_path),
                    "valid": True,
                    "methods": list(contract["methods"]),
                    "phases": list(contract["phases"]),
                    "no_fabrication": contract["no_fabrication"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    jobs = make_jobs(
        contract,
        args.phase,
        methods=args.methods,
        scenarios=args.scenarios,
        seeds=args.seeds,
    )
    if args.command == "plan":
        return execute_jobs(contract, jobs, device=args.device, dry_run=True)
    if args.command == "run":
        return execute_jobs(contract, jobs, device=args.device, dry_run=args.dry_run)
    if args.command == "summarize":
        return summarize_phase(contract, args.phase, jobs)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentContractError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
