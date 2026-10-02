"""Run, validate, aggregate, and gate the topology-temporal v2 study.

This is a separate v2 orchestrator.  It never consumes the historical v1
matrix as confirmatory evidence, and it refuses to construct formal-test jobs
until a hash-bound development gate has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.topo_v2_statistics import paired_hierarchical_bootstrap


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v2" / "experiment_contract.yaml"
)
SCENARIOS = (
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
)
ALGORITHMS = (
    "temporal_graph_v1_control",
    "topo_scene_v1_control",
    "topo_scene_balanced_v1_control",
    "temporal_graph_balanced_v1",
    "topo_v2_merge",
    "topo_v2_query",
    "topo_v2_gated",
    "topo_v2_soft",
)
PHASE_ORDER = (
    "engineering",
    "causal_2x2",
    "v2_merge",
    "v3_query",
    "v4_gated",
    "v5_lambda",
    "candidate_validation",
    "formal_test",
)
PHASE_CODES = {
    "engineering": "eng",
    "causal_2x2": "f0",
    "v2_merge": "v2",
    "v3_query": "v3",
    "v4_gated": "v4",
    "v5_lambda": "v5",
    "candidate_validation": "cv",
    "formal_test": "ft",
}
METHOD_CODES = {
    "temporal_graph": "tg",
    "topology_v1": "tv1",
    "temporal_hard": "th",
    "full_hard": "fh",
    "v2_merge": "mrg",
    "v3_query": "qry",
    "v4_gated": "gate",
    "v5_soft_1e4": "s1e4",
    "v5_soft_1e3": "s1e3",
    "v5_soft_1e2": "s1e2",
    "selected_candidate": "cand",
}
SCENARIO_CODES = {
    "left_turn": "lt",
    "cross": "cross",
    "roundabout_easy": "rae",
    "roundabout_medium": "rab",
    "roundabout": "rac",
    "carla": "carla",
}
FULL_RUN_REQUIRED_EXTRA = ("action_diagnostics_decisions.jsonl",)


class ExperimentContractError(ValueError):
    """Raised when a frozen experiment rule is absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentContractError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_sha256(path: Path) -> str:
    return _sha256(path)


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Experiment contract must be a mapping")
    return payload


def _phase_integer(phase: dict[str, Any], controls: dict[str, Any], name: str) -> int:
    value = phase.get(name, controls.get(name))
    _require(
        isinstance(value, int) and value >= 0,
        f"Phase field {name!r} must be a non-negative integer",
    )
    return int(value)


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version") == "topo-scene-v2.experiment-contract/v1",
        "Unexpected v2 experiment schema_version",
    )
    _require(contract.get("mode") == "standard", "CCFA mode must be standard")
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    for field in (
        "scope",
        "freeze",
        "claims",
        "methods",
        "phases",
        "fixed_controls",
        "metrics",
        "statistics",
        "lambda_selection",
        "development_gate",
        "stop_rules",
        "result_artifacts",
    ):
        _require(field in contract, f"Missing experiment field {field!r}")

    controls = contract["fixed_controls"]
    frozen_values = {
        "latent_dim": 128,
        "hidden_dim": 128,
        "num_heads": 2,
        "batch_size": 32,
        "replay_buffer_capacity": 20_000,
        "n_step": 4,
        "action_repeat": 3,
        "neighbors": 5,
        "history_steps": 10,
        "path_length": 10,
        "ego_control_profile": "direct",
        "raw_training_steps": 100_000,
        "learning_starts_raw_steps": 5_000,
        "traffic_protocol": "frozen_60_20_20",
        "episode_limit_profile": "source",
        "training_partition": "train",
        "development_partition": "validation",
        "formal_partition": "test",
    }
    for name, expected in frozen_values.items():
        _require(
            controls.get(name) == expected,
            f"fixed_controls.{name} must equal {expected!r}",
        )
    for unchanged in (
        "reward_unchanged",
        "action_space_unchanged",
        "actor_critic_unchanged",
        "slt_target_unchanged",
        "deterministic_final_policy",
    ):
        _require(controls.get(unchanged) is True, f"{unchanged} must be true")
    freeze = contract["freeze"]
    for name in ("source_and_assets", "topology_audit", "runtime_environment"):
        _require(isinstance(freeze.get(name), str), f"freeze.{name} must be a path")
    _require(
        freeze.get("every_run_binds_all_freeze_sha256") is True,
        "Every run must bind all frozen dependency hashes",
    )

    methods = contract["methods"]
    _require(isinstance(methods, dict) and methods, "methods must be a mapping")
    for name, method in methods.items():
        _require(isinstance(method, dict), f"Method {name!r} must be a mapping")
        if "resolver" in method:
            _require(
                name == "selected_candidate",
                "Only selected_candidate may use a resolver",
            )
        else:
            _require(
                method.get("cli_algorithm") in ALGORITHMS,
                f"Method {name!r} has an unsupported cli_algorithm",
            )
            _require(
                isinstance(method.get("implementation_id"), str),
                f"Method {name!r} needs implementation_id",
            )
            coefficient = float(method.get("soft_slot_balance_coef", 0.0))
            _require(coefficient >= 0.0, f"Method {name!r} has negative lambda")
            _require(
                method.get("cli_algorithm") == "topo_v2_soft" or coefficient == 0.0,
                f"Only topo_v2_soft may have positive lambda ({name!r})",
            )

    phases = contract["phases"]
    _require(tuple(phases) == PHASE_ORDER, "Phases must follow the frozen order")
    for phase_name, phase in phases.items():
        _require(isinstance(phase, dict), f"Phase {phase_name!r} must be a mapping")
        for list_field in ("methods", "scenarios", "seeds"):
            _require(
                isinstance(phase.get(list_field), list) and phase[list_field],
                f"Phase {phase_name!r} needs non-empty {list_field}",
            )
        _require(
            not (set(phase["methods"]) - set(methods)),
            f"Phase {phase_name!r} references unknown methods",
        )
        _require(
            not (set(phase["scenarios"]) - set(SCENARIOS)),
            f"Phase {phase_name!r} references unknown scenarios",
        )
        _require(
            all(isinstance(seed, int) and seed >= 0 for seed in phase["seeds"]),
            f"Phase {phase_name!r} seeds must be non-negative integers",
        )
        if phase_name == "engineering":
            _require(phase.get("check_only") is True, "Engineering must be check-only")
        else:
            _require(
                _phase_integer(phase, controls, "raw_training_steps") == 100_000,
                f"Phase {phase_name!r} must use 100k raw steps",
            )
            _require(
                _phase_integer(phase, controls, "final_evaluation_episodes") > 0,
                f"Phase {phase_name!r} needs final evaluation episodes",
            )
        expected_split = "test" if phase_name == "formal_test" else "validation"
        _require(
            phase.get("evaluation_split") == expected_split,
            f"Phase {phase_name!r} must use {expected_split!r}",
        )

    factorial_methods = phases["causal_2x2"]["methods"]
    _require(len(factorial_methods) == 4, "causal_2x2 must contain four cells")
    factorial_cells = {
        (
            bool(methods[name].get("topology")),
            bool(methods[name].get("hard_slot_normalization")),
        )
        for name in factorial_methods
    }
    _require(
        factorial_cells == {(False, False), (True, False), (False, True), (True, True)},
        "causal_2x2 is not a complete topology x hard-normalisation factorial",
    )
    _require(
        tuple(phases["causal_2x2"]["scenarios"]) == SCENARIOS,
        "causal_2x2 must cover all six scenarios",
    )
    _require(
        phases["formal_test"]["seeds"] == list(range(10)),
        "formal_test must use seeds 0..9",
    )
    _require(
        phases["formal_test"]["methods"] == ["temporal_graph", "selected_candidate"],
        "formal_test must compare TemporalGraph with the frozen candidate",
    )
    return contract


def output_root(contract: dict[str, Any]) -> Path:
    raw = Path(str(contract["output_root"]))
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def _resolve_project_path(value: str) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def frozen_dependency_hashes(contract: dict[str, Any]) -> dict[str, str]:
    freeze = contract["freeze"]
    paths = {
        "freeze_manifest_sha256": _resolve_project_path(freeze["source_and_assets"]),
        "topology_audit_sha256": _resolve_project_path(freeze["topology_audit"]),
        "runtime_environment_sha256": _resolve_project_path(
            freeze["runtime_environment"]
        ),
    }
    for path in paths.values():
        _require(path.is_file(), f"Frozen dependency is missing: {path}")
    manifest = _load_json(paths["freeze_manifest_sha256"])
    audit = _load_json(paths["topology_audit_sha256"])
    runtime = _load_json(paths["runtime_environment_sha256"])
    _require(
        manifest.get("schema_version") == "topology-temporal-v2-freeze/v1"
        and isinstance(manifest.get("manifest_content_sha256"), str),
        "Invalid v2 freeze manifest",
    )
    _require(
        audit.get("schema_version") == "topology-graph-v2-audit/v1"
        and audit.get("passed") is True
        and int(audit.get("scenario_count", -1)) == 6,
        "Topology audit is missing or failed",
    )
    _require(
        runtime.get("schema_version")
        == "topology-temporal-v2-runtime-environment/v1"
        and runtime.get("scientific_runtime") is True,
        "Invalid v2 runtime environment receipt",
    )
    return {name: _sha256(path) for name, path in paths.items()}


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
    evaluation_split: str
    traffic_protocol: str
    episode_limit_profile: str
    slot_balance_coef: float
    check_only: bool
    protocol_tag: str

    @property
    def name(self) -> str:
        return (
            f"{PHASE_CODES[self.phase]}__{METHOD_CODES[self.method]}__"
            f"{SCENARIO_CODES[self.scenario]}__s{self.seed}__p{self.protocol_tag}"
        )


def _selected_candidate(
    contract: dict[str, Any], contract_digest: str
) -> dict[str, Any]:
    resolver = contract["methods"]["selected_candidate"]["resolver"]
    path = _resolve_project_path(str(resolver))
    _require(path.is_file(), f"Selected candidate is not frozen: {path}")
    payload = _load_json(path)
    _require(payload.get("status") == "frozen", "Candidate selection is not frozen")
    _require(
        payload.get("experiment_contract_sha256") == contract_digest,
        "Candidate selection was produced under a different contract",
    )
    _require(payload.get("cli_algorithm") in ALGORITHMS, "Invalid candidate algorithm")
    _require(
        isinstance(payload.get("implementation_id"), str),
        "Candidate selection lacks implementation_id",
    )
    return payload


def _selection(value: str | None, allowed: Iterable[Any]) -> list[Any]:
    allowed_values = list(allowed)
    if value is None or value == "all":
        return allowed_values
    selected: list[Any] = []
    for token in value.split(","):
        token = token.strip()
        converted: Any = (
            int(token)
            if allowed_values and isinstance(allowed_values[0], int)
            else token
        )
        _require(converted in allowed_values, f"Value {converted!r} not in {allowed_values}")
        _require(converted not in selected, f"Duplicate selection {converted!r}")
        selected.append(converted)
    return selected


def make_jobs(
    contract: dict[str, Any],
    phase_name: str,
    *,
    contract_digest: str,
    methods: str | None = None,
    scenarios: str | None = None,
    seeds: str | None = None,
) -> list[Job]:
    _require(phase_name in PHASE_ORDER, f"Unknown phase {phase_name!r}")
    phase = contract["phases"][phase_name]
    controls = contract["fixed_controls"]
    selected_methods = _selection(methods, phase["methods"])
    selected_scenarios = _selection(scenarios, phase["scenarios"])
    selected_seeds = _selection(seeds, phase["seeds"])
    jobs: list[Job] = []
    check_only = bool(phase.get("check_only", False))
    for method_name in selected_methods:
        method = contract["methods"][method_name]
        if method_name == "selected_candidate":
            resolved = _selected_candidate(contract, contract_digest)
            algorithm = str(resolved["cli_algorithm"])
            implementation_id = str(resolved["implementation_id"])
            coefficient = float(resolved["soft_slot_balance_coef"])
        else:
            algorithm = str(method["cli_algorithm"])
            implementation_id = str(method["implementation_id"])
            coefficient = float(method.get("soft_slot_balance_coef", 0.0))
        for scenario in selected_scenarios:
            for seed in selected_seeds:
                jobs.append(
                    Job(
                        phase=phase_name,
                        method=str(method_name),
                        algorithm=algorithm,
                        implementation_id=implementation_id,
                        scenario=str(scenario),
                        seed=int(seed),
                        raw_steps=_phase_integer(phase, controls, "raw_training_steps"),
                        learning_starts=_phase_integer(
                            phase, controls, "learning_starts_raw_steps"
                        ),
                        checkpoint_frequency=(
                            0
                            if check_only
                            else _phase_integer(
                                phase, controls, "checkpoint_frequency_raw_steps"
                            )
                        ),
                        evaluation_frequency=(
                            0
                            if check_only
                            else _phase_integer(
                                phase, controls, "evaluation_frequency_raw_steps"
                            )
                        ),
                        evaluation_episodes=(
                            1
                            if check_only
                            else _phase_integer(
                                phase, controls, "final_evaluation_episodes"
                            )
                        ),
                        evaluation_split=str(phase["evaluation_split"]),
                        traffic_protocol=str(controls["traffic_protocol"]),
                        episode_limit_profile=str(controls["episode_limit_profile"]),
                        slot_balance_coef=coefficient,
                        check_only=check_only,
                        protocol_tag=contract_digest[:8],
                    )
                )
    names = [job.name for job in jobs]
    _require(len(names) == len(set(names)), "Experiment jobs are not uniquely named")
    return jobs


def phase_root(contract: dict[str, Any], phase: str) -> Path:
    return output_root(contract) / phase


def run_directory(contract: dict[str, Any], job: Job) -> Path:
    return phase_root(contract, job.phase) / "runs" / job.name


def command_for(
    contract: dict[str, Any], job: Job, *, device: str, contract_digest: str
) -> list[str]:
    controls = contract["fixed_controls"]
    dependency_hashes = frozen_dependency_hashes(contract)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v2.py"),
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
        str(controls["replay_buffer_capacity"]),
        "--action-repeat",
        str(controls["action_repeat"]),
        "--neighbors",
        str(controls.get("neighbors", 5)),
        "--history-steps",
        str(controls.get("history_steps", 10)),
        "--path-length",
        str(controls.get("path_length", 10)),
        "--ego-control-profile",
        str(controls.get("ego_control_profile", "direct")),
        "--checkpoint-freq",
        str(job.checkpoint_frequency),
        "--eval-freq",
        str(job.evaluation_frequency),
        "--eval-episodes",
        str(job.evaluation_episodes),
        "--traffic-protocol",
        job.traffic_protocol,
        "--evaluation-split",
        job.evaluation_split,
        "--episode-limit-profile",
        job.episode_limit_profile,
        "--route-sigma",
        str(controls["route_sigma"]),
        "--topology-top-k",
        str(controls["topology_top_k"]),
        "--reverse-heading-cosine-min",
        str(controls["reverse_heading_cosine_min"]),
        "--route-bias-init",
        str(controls["route_bias_init"]),
        "--heading-bias-init",
        str(controls["heading_bias_init"]),
        "--topology-layerscale-init",
        str(controls["topology_layerscale_init"]),
        "--goal-layerscale-init",
        str(controls["goal_layerscale_init"]),
        "--experiment-contract-sha256",
        contract_digest,
        "--freeze-manifest-sha256",
        dependency_hashes["freeze_manifest_sha256"],
        "--topology-audit-sha256",
        dependency_hashes["topology_audit_sha256"],
        "--runtime-environment-sha256",
        dependency_hashes["runtime_environment_sha256"],
        "--output-dir",
        str(phase_root(contract, job.phase) / "runs"),
        "--model-name",
        job.name,
    ]
    if job.algorithm == "topo_v2_soft":
        command.extend(["--slot-balance-coef", str(job.slot_balance_coef)])
    if job.check_only:
        command.append("--check-only")
    return command


def _same_float(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def accepted_run_reason(
    contract: dict[str, Any], job: Job, contract_digest: str
) -> tuple[bool, str]:
    directory = run_directory(contract, job)
    engineering_required = (
        "arguments.json",
        "method_metadata.json",
        "check_result.json",
    )
    required = (
        engineering_required
        if job.check_only
        else tuple(contract["result_artifacts"]["run_required"])
        + FULL_RUN_REQUIRED_EXTRA
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        arguments = _load_json(directory / "arguments.json")
        requested = arguments["requested_raw_steps"]
        fidelity = arguments["implementation_fidelity"]
        effective = arguments["effective_method_hyperparameters"]
        expected_dependencies = frozen_dependency_hashes(contract)
        checks = (
            requested.get("algo") == job.algorithm,
            requested.get("scenario") == job.scenario,
            int(requested.get("seed", -1)) == job.seed,
            int(requested.get("max_steps", -1)) == job.raw_steps,
            requested.get("evaluation_split") == job.evaluation_split,
            requested.get("traffic_protocol") == job.traffic_protocol,
            requested.get("episode_limit_profile") == job.episode_limit_profile,
            fidelity.get("implementation_id") == job.implementation_id,
            arguments.get("experiment_contract_sha256") == contract_digest,
            _same_float(effective.get("slot_balance_coef"), job.slot_balance_coef),
            arguments.get("frozen_dependencies") == expected_dependencies,
        )
        if not all(checks):
            return False, "arguments or implementation receipt mismatch"
        metadata = _load_json(directory / "method_metadata.json")
        if metadata.get("implementation_id") != job.implementation_id:
            return False, "method_metadata implementation mismatch"
        if job.check_only:
            check = _load_json(directory / "check_result.json")
            if not (
                check.get("status") == "ok"
                and check.get("action_finite") is True
                and check.get("algorithm") == job.algorithm
                and check.get("scenario") == job.scenario
            ):
                return False, "engineering check_result failed"
            return True, "accepted engineering check"

        detailed = _load_json(directory / "paper_evaluation_detailed.json")
        provenance = detailed.get("evaluation_provenance", {})
        if not (
            detailed.get("algorithm") == job.algorithm
            and detailed.get("scenario") == job.scenario
            and int(detailed.get("trained_raw_steps", -1)) == job.raw_steps
            and detailed.get("experiment_contract_sha256") == contract_digest
            and detailed.get("evaluation_split") == job.evaluation_split
            and int(detailed.get("summary", {}).get("episodes", -1))
            == job.evaluation_episodes
            and provenance.get("validated") is True
            and provenance.get("traffic_partition") == job.evaluation_split
        ):
            return False, "paper evaluation receipt mismatch"
        checkpoint = _load_json(directory / "checkpoint_audit.json")
        checkpoints = checkpoint.get("checkpoints", [])
        if not checkpoints or int(checkpoints[-1].get("raw_step", -1)) != job.raw_steps:
            return False, "missing exact final raw-step checkpoint"
        if not all(item.get("zip_crc_ok") is True for item in checkpoints):
            return False, "checkpoint ZIP audit failed"
        action = _load_json(directory / "action_diagnostics.json")
        trace = action.get("trace", {})
        trace_path = directory / "action_diagnostics_decisions.jsonl"
        if not (
            action.get("computed_from_real_rollout") is True
            and action.get("aligned_with_paper_evaluation") is True
            and int(action.get("episodes", -1)) == job.evaluation_episodes
            and int(trace.get("records", -1)) == int(action.get("decision_records", -2))
            and str(trace.get("sha256", "")).lower() == _sha256(trace_path).lower()
            and action.get("outcomes") == detailed.get("summary")
        ):
            return False, "action diagnostic trace mismatch"
        if job.algorithm.startswith("topo_v2_"):
            evaluation_mechanism = action.get("evaluation_mechanism", {})
            required_mechanism = {
                "topology_effective_lanes",
                "route_compatible_attention_mass",
                "topology_fallback_rate",
            }
            if not required_mechanism.issubset(evaluation_mechanism):
                return False, "evaluation mechanism diagnostics are incomplete"
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return False, f"invalid artifact: {exc}"
    return True, "accepted complete run"


def accepted_run(contract: dict[str, Any], job: Job, contract_digest: str) -> bool:
    return accepted_run_reason(contract, job, contract_digest)[0]


def _phase_complete_receipt(contract: dict[str, Any], phase: str) -> dict[str, Any] | None:
    path = phase_root(contract, phase) / "summary.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    return payload if payload.get("complete") is True else None


def assert_phase_can_run(
    contract: dict[str, Any], phase: str, contract_digest: str
) -> None:
    index = PHASE_ORDER.index(phase)
    if index > 0:
        previous = PHASE_ORDER[index - 1]
        _require(
            _phase_complete_receipt(contract, previous) is not None,
            f"Phase {phase!r} requires complete summarized phase {previous!r}",
        )
    if phase == "v2_merge":
        _require(
            (output_root(contract) / "factorial_2x2.json").is_file(),
            "v2_merge requires the 2x2 attribution receipt",
        )
    if phase in ("candidate_validation", "formal_test"):
        _selected_candidate(contract, contract_digest)
    if phase == "formal_test":
        gate_path = _resolve_project_path(
            str(contract["phases"]["formal_test"]["gate_receipt"])
        )
        _require(gate_path.is_file(), "Formal test is locked: no development gate")
        gate = _load_json(gate_path)
        _require(gate.get("decision") == "pass", "Formal test is locked: gate did not pass")
        _require(
            gate.get("experiment_contract_sha256") == contract_digest,
            "Formal test gate belongs to another contract",
        )


def _job_dict(
    contract: dict[str, Any], job: Job, device: str, contract_digest: str
) -> dict[str, Any]:
    accepted, reason = accepted_run_reason(contract, job, contract_digest)
    return {
        **asdict(job),
        "name": job.name,
        "run_directory": str(run_directory(contract, job)),
        "accepted": accepted,
        "acceptance_reason": reason,
        "command": command_for(
            contract, job, device=device, contract_digest=contract_digest
        ),
    }


def write_phase_plan(
    contract: dict[str, Any],
    contract_path: Path,
    phase: str,
    jobs: list[Job],
    *,
    device: str,
    contract_digest: str,
) -> None:
    root = phase_root(contract, phase)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "protocol.snapshot.yaml"
    contract_bytes = contract_path.read_bytes()
    if snapshot.exists():
        _require(
            snapshot.read_bytes() == contract_bytes,
            f"Protocol snapshot drift in {snapshot}",
        )
    else:
        snapshot.write_bytes(contract_bytes)
    full_jobs = make_jobs(
        contract, phase, contract_digest=contract_digest
    )
    _write_json(
        root / "jobs.json",
        {
            "schema_version": "topology-temporal-v2-jobs/v1",
            "experiment_contract": str(contract_path),
            "experiment_contract_sha256": contract_digest,
            "phase": phase,
            "device": device,
            "jobs": [
                _job_dict(contract, job, device, contract_digest)
                for job in full_jobs
            ],
        },
    )
    _write_json(
        root / "last_selection.json",
        {
            "schema_version": "topology-temporal-v2-job-selection/v1",
            "experiment_contract_sha256": contract_digest,
            "phase": phase,
            "selected_job_names": [job.name for job in jobs],
            "full_phase_job_count": len(full_jobs),
        },
    )


def execute_jobs(
    contract: dict[str, Any],
    contract_path: Path,
    jobs: list[Job],
    *,
    device: str,
    contract_digest: str,
    dry_run: bool,
    max_jobs: int | None,
    continue_on_error: bool,
    workers: int,
) -> int:
    if not jobs:
        raise ExperimentContractError("No jobs selected")
    phase = jobs[0].phase
    _require(all(job.phase == phase for job in jobs), "Jobs span multiple phases")
    if not dry_run:
        assert_phase_can_run(contract, phase, contract_digest)
    write_phase_plan(
        contract,
        contract_path,
        phase,
        jobs,
        device=device,
        contract_digest=contract_digest,
    )
    plan = [_job_dict(contract, job, device, contract_digest) for job in jobs]
    if dry_run:
        print(json.dumps({"phase": phase, "jobs": plan}, ensure_ascii=False, indent=2))
        return 0

    _require(workers > 0, "--workers must be positive")

    pending = [job for job in jobs if not accepted_run(contract, job, contract_digest)]
    if max_jobs is not None:
        _require(max_jobs > 0, "--max-jobs must be positive")
        pending = pending[:max_jobs]
    failures: list[dict[str, Any]] = []
    for job in pending:
        directory = run_directory(contract, job)
        if directory.exists():
            accepted, reason = accepted_run_reason(contract, job, contract_digest)
            if not accepted:
                raise FileExistsError(
                    f"Refusing to overwrite incomplete or mismatched run {directory}: {reason}"
                )
    log_root = phase_root(contract, phase) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def run_one(job: Job) -> dict[str, Any] | None:
        log_path = log_root / f"{job.name}.log"
        command = command_for(
            contract, job, device=device, contract_digest=contract_digest
        )
        print(
            json.dumps(
                {"job": job.name, "status": "running", "log": str(log_path)},
                ensure_ascii=False,
            ),
            flush=True,
        )
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
        accepted, reason = accepted_run_reason(contract, job, contract_digest)
        if completed.returncode != 0 or not accepted:
            failure = {
                "job": job.name,
                "returncode": completed.returncode,
                "acceptance_reason": reason,
                "log": str(log_path),
            }
            print(json.dumps({**failure, "status": "failed"}, ensure_ascii=False), flush=True)
            return failure
        else:
            print(
                json.dumps({"job": job.name, "status": "accepted"}, ensure_ascii=False),
                flush=True,
            )
            return None

    if workers == 1:
        for job in pending:
            failure = run_one(job)
            if failure is not None:
                failures.append(failure)
                if not continue_on_error:
                    break
    else:
        # Every run owns a separate SUMO/TraCI instance, output directory,
        # and launcher log.  The main process alone validates and aggregates
        # receipts after each child exits.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, job): job for job in pending}
            for future in as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
    _write_json(
        phase_root(contract, phase) / "last_execution.json",
        {
            "schema_version": "topology-temporal-v2-execution/v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_contract_sha256": contract_digest,
            "selected_jobs": len(jobs),
            "attempted_jobs": len(pending),
            "workers": workers,
            "failures": failures,
        },
    )
    return 1 if failures else 0


def _stat_mean(diagnostics: dict[str, Any], name: str) -> float | None:
    row = diagnostics.get(name)
    if not isinstance(row, dict) or row.get("mean") is None:
        return None
    value = float(row["mean"])
    return value if math.isfinite(value) else None


def _run_row(contract: dict[str, Any], job: Job) -> dict[str, Any]:
    directory = run_directory(contract, job)
    detailed = _load_json(directory / "paper_evaluation_detailed.json")
    summary = detailed["summary"]
    performance = _load_json(directory / "performance_profile.json")
    diagnostics = _load_json(directory / "training_diagnostics.json")["statistics"]
    actions = _load_json(directory / "action_diagnostics.json")
    completion = detailed.get("mean_success_completion_time_seconds")
    mechanism_names = (
        "diagnostic/topology_attention_entropy",
        "diagnostic/topology_effective_lanes",
        "diagnostic/route_compatible_attention_mass",
        "diagnostic/merge_attention_mass",
        "diagnostic/merge_pair_score",
        "diagnostic/topology_residual_scale",
        "diagnostic/goal_residual_scale",
        "diagnostic/topology_fallback_rate",
        "diagnostic/ego_slot_rms",
        "diagnostic/social_slot_rms",
        "diagnostic/route_slot_rms",
        "diagnostic/slot_rms_ratio",
        "diagnostic/latent_std",
        "train/soft_slot_balance_loss",
    )
    return {
        "phase": job.phase,
        "method": job.method,
        "algorithm": job.algorithm,
        "implementation_id": job.implementation_id,
        "soft_slot_balance_coef": job.slot_balance_coef,
        "scenario": job.scenario,
        "seed": job.seed,
        "raw_steps": job.raw_steps,
        "evaluation_split": job.evaluation_split,
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
        "train_ms_per_gradient_step": _stat_mean(
            diagnostics, "performance/train_ms_per_gradient_step"
        ),
        "training_wall_seconds": float(
            performance["end_to_end_training_wall_seconds"]
        ),
        "inference_ms_per_action": float(
            performance["inference"]["mean_milliseconds_per_action"]
        ),
        "peak_gpu_memory_mb": (
            float(performance["peak_gpu_memory_mb"])
            if performance.get("peak_gpu_memory_mb") is not None
            else None
        ),
        "lane_command_negative_rate": float(actions["lane_command_rates"]["negative"]),
        "lane_command_keep_rate": float(actions["lane_command_rates"]["keep"]),
        "lane_command_positive_rate": float(actions["lane_command_rates"]["positive"]),
        "lane_command_threshold_margin": float(
            actions["lane_command_threshold_margin"]["mean"]
        ),
        "lane_change_applied_rate": float(actions["lane_change_applied_rate"]),
        "mean_speed_mps": float(actions["actual_speed_mps"]["mean"]),
        "mechanism": {
            name.removeprefix("diagnostic/").removeprefix("train/"): _stat_mean(
                diagnostics, name
            )
            for name in mechanism_names
        },
        "evaluation_mechanism": {
            name: (
                float(distribution["mean"])
                if distribution.get("mean") is not None
                else None
            )
            for name, distribution in actions.get("evaluation_mechanism", {}).items()
        },
        "episode_records": detailed["episode_records"],
    }


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "population_std": None, "minimum": None, "maximum": None}
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "population_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["method"]), []).append(row)
    metric_names = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "successful_completion_time_seconds",
        "parameter_count",
        "train_ms_per_gradient_step",
        "training_wall_seconds",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
        "lane_command_negative_rate",
        "lane_command_keep_rate",
        "lane_command_positive_rate",
        "lane_command_threshold_margin",
        "lane_change_applied_rate",
        "mean_speed_mps",
    )
    output: list[dict[str, Any]] = []
    for (scenario, method), members in sorted(grouped.items()):
        mechanism_names = sorted(
            {name for member in members for name in member["mechanism"]}
        )
        output.append(
            {
                "scenario": scenario,
                "method": method,
                "seeds": sorted(int(row["seed"]) for row in members),
                "metrics": {
                    name: _mean_std(
                        [
                            float(row[name])
                            for row in members
                            if row.get(name) is not None
                        ]
                    )
                    for name in metric_names
                },
                "mechanism": {
                    name: _mean_std(
                        [
                            float(row["mechanism"][name])
                            for row in members
                            if row["mechanism"].get(name) is not None
                        ]
                    )
                    for name in mechanism_names
                },
                "evaluation_mechanism": {
                    name: _mean_std(
                        [
                            float(row["evaluation_mechanism"][name])
                            for row in members
                            if row["evaluation_mechanism"].get(name) is not None
                        ]
                    )
                    for name in sorted(
                        {
                            key
                            for member in members
                            for key in member["evaluation_mechanism"]
                        }
                    )
                },
            }
        )
    return output


def _paired_contrast(
    rows: list[dict[str, Any]], method: str, baseline: str
) -> dict[str, Any]:
    by_key = {(row["scenario"], row["seed"], row["method"]): row for row in rows}
    metrics = ("success_rate", "collision_rate", "mean_return", "timeout_rate")
    pairs: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["scenario"], row["seed"])
            for row in rows
            if row["method"] == method
        }
    )
    for scenario, seed in keys:
        candidate = by_key.get((scenario, seed, method))
        control = by_key.get((scenario, seed, baseline))
        if candidate is None or control is None:
            continue
        pairs.append(
            {
                "scenario": scenario,
                "seed": seed,
                **{
                    f"delta_{metric}": float(candidate[metric] - control[metric])
                    for metric in metrics
                },
            }
        )
    return {
        "method": method,
        "baseline": baseline,
        "paired_cells": len(pairs),
        "pairs": pairs,
        "macro_delta": {
            metric: (
                float(statistics.mean([row[f"delta_{metric}"] for row in pairs]))
                if pairs
                else None
            )
            for metric in metrics
        },
    }


def phase_attribution(phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    contrast_map = {
        "v2_merge": ("v2_merge", "topology_v1"),
        "v3_query": ("v3_query", "v2_merge"),
        "v4_gated": ("v4_gated", "v3_query"),
        "v5_lambda": ("v5_soft_1e3", "v4_gated"),
        "candidate_validation": ("selected_candidate", "temporal_graph"),
        "formal_test": ("selected_candidate", "temporal_graph"),
    }
    contrasts: list[dict[str, Any]] = []
    if phase == "v5_lambda":
        contrasts.extend(
            _paired_contrast(rows, method, "v4_gated")
            for method in ("v5_soft_1e4", "v5_soft_1e3", "v5_soft_1e2")
        )
    elif phase in contrast_map:
        method, baseline = contrast_map[phase]
        contrasts.append(_paired_contrast(rows, method, baseline))
    return {
        "schema_version": "topology-temporal-v2-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "phase": phase,
        "finite": _finite_tree(rows),
        "contrasts": contrasts,
        "interpretation_rule": "Deltas are descriptive until the predeclared gate/statistical analysis is complete.",
    }


def factorial_2x2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell = {
        (False, False): "temporal_graph",
        (True, False): "topology_v1",
        (False, True): "temporal_hard",
        (True, True): "full_hard",
    }
    by_key = {(row["scenario"], row["seed"], row["method"]): row for row in rows}
    output_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row["scenario"] == scenario
            }
        )
        for metric in ("success_rate", "collision_rate", "mean_return", "timeout_rate"):
            effects = []
            for seed in seeds:
                values = {
                    key: by_key.get((scenario, seed, method), {}).get(metric)
                    for key, method in cell.items()
                }
                if any(value is None for value in values.values()):
                    continue
                f00 = float(values[(False, False)])
                f10 = float(values[(True, False)])
                f01 = float(values[(False, True)])
                f11 = float(values[(True, True)])
                effects.append(
                    {
                        "seed": seed,
                        "topology_main_effect": ((f10 - f00) + (f11 - f01)) / 2.0,
                        "hard_norm_main_effect": ((f01 - f00) + (f11 - f10)) / 2.0,
                        "interaction": f11 - f10 - f01 + f00,
                    }
                )
            output_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "seed_effects": effects,
                    "mean_effects": {
                        name: (
                            float(statistics.mean([row[name] for row in effects]))
                            if effects
                            else None
                        )
                        for name in (
                            "topology_main_effect",
                            "hard_norm_main_effect",
                            "interaction",
                        )
                    },
                }
            )
    complete = all(len(row["seed_effects"]) == 3 for row in output_rows)
    return {
        "schema_version": "topology-temporal-v2-factorial/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "complete": complete,
        "cells": {f"topology={key[0]},hard={key[1]}": value for key, value in cell.items()},
        "effects": output_rows,
    }


def _phase_context_rows(
    contract: dict[str, Any], phase: str, own_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = list(own_rows)
    dependencies = {
        "v2_merge": ("causal_2x2",),
        "v3_query": ("causal_2x2", "v2_merge"),
        "v4_gated": ("v3_query",),
        "v5_lambda": ("v4_gated",),
        "candidate_validation": ("causal_2x2",),
    }
    for dependency in dependencies.get(phase, ()):
        path = phase_root(contract, dependency) / "summary.json"
        if path.is_file():
            rows.extend(_load_json(path).get("per_run", []))
    return rows


def summarize_phase(
    contract: dict[str, Any],
    contract_path: Path,
    phase: str,
    jobs: list[Job],
    *,
    contract_digest: str,
) -> int:
    complete_jobs = [
        job for job in jobs if accepted_run(contract, job, contract_digest)
    ]
    missing = [job.name for job in jobs if job not in complete_jobs]
    rows = [] if jobs[0].check_only else [_run_row(contract, job) for job in complete_jobs]
    root = phase_root(contract, phase)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "topology-temporal-v2-results/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "experiment_contract": str(contract_path),
        "experiment_contract_sha256": contract_digest,
        "phase": phase,
        "expected_jobs": len(jobs),
        "complete_jobs": len(complete_jobs),
        "missing_jobs": missing,
        "complete": not missing,
        "per_run": rows,
        "aggregates": aggregate_rows(rows),
    }
    _write_json(root / "summary.json", payload)
    csv_fields = [
        "phase",
        "method",
        "scenario",
        "seed",
        "raw_steps",
        "evaluation_split",
        "soft_slot_balance_coef",
        "success_rate",
        "collision_rate",
        "mean_return",
        "off_route_rate",
        "timeout_rate",
        "successful_completion_time_seconds",
        "lane_command_negative_rate",
        "lane_command_keep_rate",
        "lane_command_positive_rate",
        "lane_change_applied_rate",
        "mean_speed_mps",
        "parameter_count",
        "train_ms_per_gradient_step",
        "training_wall_seconds",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
        "run_directory",
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})
    context_rows = _phase_context_rows(contract, phase, rows)
    attribution = phase_attribution(phase, context_rows)
    _write_json(root / "attribution.json", attribution)
    next_phase = PHASE_ORDER[PHASE_ORDER.index(phase) + 1] if phase != PHASE_ORDER[-1] else None
    _write_json(
        root / "iteration_receipt.json",
        {
            "schema_version": "topology-temporal-v2-iteration-receipt/v1",
            "computed_from_real_runs": True,
            "fabricated_values": False,
            "experiment_contract_sha256": contract_digest,
            "phase": phase,
            "status": "complete" if not missing else "incomplete",
            "next_phase": next_phase if not missing else None,
            "claim_status": "eligible_for_predeclared_analysis" if not missing else "TBD",
            "attribution_file": str((root / "attribution.json").resolve()),
            "stop_rules_remain_binding": True,
        },
    )
    if phase == "causal_2x2":
        factorial = factorial_2x2(rows)
        _write_json(output_root(contract) / "factorial_2x2.json", factorial)
    if phase == "formal_test":
        if not missing:
            inference = paired_hierarchical_bootstrap(
                rows,
                candidate_method="selected_candidate",
                baseline_method="temporal_graph",
                scenarios=SCENARIOS,
                resamples=int(contract["statistics"]["hierarchical_bootstrap_resamples"]),
                seed=int(contract["statistics"]["bootstrap_seed"]),
            )
            _write_json(root / "formal_inference.json", inference)
            _write_json(output_root(contract) / "formal_inference.json", inference)
        _write_json(output_root(contract) / "formal_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


def _summary_rows(contract: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    path = phase_root(contract, phase) / "summary.json"
    _require(path.is_file(), f"Missing summary for phase {phase!r}")
    payload = _load_json(path)
    _require(payload.get("complete") is True, f"Phase {phase!r} is incomplete")
    return list(payload.get("per_run", []))


def select_candidate(
    contract: dict[str, Any], contract_digest: str
) -> dict[str, Any]:
    rows = _summary_rows(contract, "v4_gated") + _summary_rows(contract, "v5_lambda")
    scenarios = ("roundabout_medium", "roundabout", "carla")
    methods = ("v4_gated", "v5_soft_1e4", "v5_soft_1e3", "v5_soft_1e2")
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((row["method"], row["scenario"]), []).append(row)
    for method in methods:
        for scenario in scenarios:
            _require(
                len(by_group.get((method, scenario), [])) == 3,
                f"Candidate selection lacks {method}/{scenario} three-seed evidence",
            )
    baseline_collision = {
        scenario: statistics.mean(
            row["collision_rate"] for row in by_group[("v4_gated", scenario)]
        )
        for scenario in scenarios
    }
    evaluations: list[dict[str, Any]] = []
    for method in methods:
        members = [row for scenario in scenarios for row in by_group[(method, scenario)]]
        scenario_collision = {
            scenario: statistics.mean(
                row["collision_rate"] for row in by_group[(method, scenario)]
            )
            for scenario in scenarios
        }
        safe = all(
            scenario_collision[scenario] <= baseline_collision[scenario] + 0.02
            for scenario in scenarios
        )
        soft_losses = [
            float(row["mechanism"]["soft_slot_balance_loss"])
            for row in members
            if row["mechanism"].get("soft_slot_balance_loss") is not None
        ]
        coefficient = float(members[0]["soft_slot_balance_coef"])
        evaluations.append(
            {
                "method": method,
                "cli_algorithm": members[0]["algorithm"],
                "implementation_id": members[0]["implementation_id"],
                "soft_slot_balance_coef": coefficient,
                "safe": safe,
                "macro_success_rate": float(
                    statistics.mean(row["success_rate"] for row in members)
                ),
                "mean_log_slot_rms_variance": (
                    float(statistics.mean(soft_losses)) if soft_losses else None
                ),
                "scenario_collision_rate": scenario_collision,
                "v4_collision_guardrail": {
                    scenario: baseline_collision[scenario] + 0.02
                    for scenario in scenarios
                },
            }
        )
    safe_rows = [row for row in evaluations if row["safe"]]
    _require(safe_rows, "Every lambda candidate violated the safety guardrail")
    best_success = max(row["macro_success_rate"] for row in safe_rows)
    success_tied = [
        row for row in safe_rows if best_success - row["macro_success_rate"] < 0.01
    ]
    balance_values = [
        row["mean_log_slot_rms_variance"]
        for row in success_tied
        if row["mean_log_slot_rms_variance"] is not None
    ]
    if balance_values:
        best_balance = min(balance_values)
        balance_tied = [
            row
            for row in success_tied
            if row["mean_log_slot_rms_variance"] is not None
            and math.isclose(
                row["mean_log_slot_rms_variance"], best_balance, rel_tol=0.0, abs_tol=1e-12
            )
        ]
    else:
        balance_tied = success_tied
    selected = min(balance_tied, key=lambda row: row["soft_slot_balance_coef"])
    payload = {
        "schema_version": "topology-temporal-v2-candidate-selection/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "status": "frozen",
        "experiment_contract_sha256": contract_digest,
        "selection_rule": contract["lambda_selection"],
        "evaluations": evaluations,
        "selected_method": selected["method"],
        "cli_algorithm": selected["cli_algorithm"],
        "implementation_id": selected["implementation_id"],
        "soft_slot_balance_coef": selected["soft_slot_balance_coef"],
        "selected_from_validation_only": True,
        "formal_test_accessed": False,
    }
    path = _resolve_project_path(str(contract["lambda_selection"]["output"]))
    if path.exists():
        _require(_load_json(path) == payload, "Refusing to change a frozen candidate")
    else:
        _write_json(path, payload)
    return payload


def compute_development_gate(
    contract: dict[str, Any], contract_digest: str
) -> dict[str, Any]:
    candidate_rows = _summary_rows(contract, "candidate_validation")
    baseline_rows = [
        row
        for row in _summary_rows(contract, "causal_2x2")
        if row["method"] == "temporal_graph"
        and row["scenario"] in contract["development_gate"]["required_scenarios"]
    ]
    rows = baseline_rows + candidate_rows
    candidate = _selected_candidate(contract, contract_digest)
    by_key = {(row["method"], row["scenario"], row["seed"]): row for row in rows}
    required = contract["development_gate"]["required_scenarios"]
    pairs: list[dict[str, Any]] = []
    missing: list[str] = []
    for scenario in required:
        for seed in (0, 1, 2):
            baseline = by_key.get(("temporal_graph", scenario, seed))
            selected = by_key.get(("selected_candidate", scenario, seed))
            if baseline is None or selected is None:
                missing.append(f"{scenario}/seed{seed}")
                continue
            pairs.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "success_delta": selected["success_rate"] - baseline["success_rate"],
                    "collision_delta": selected["collision_rate"] - baseline["collision_rate"],
                    "candidate": selected,
                    "baseline": baseline,
                }
            )
    _require(not missing, f"Development gate lacks paired evidence: {missing}")
    gate = contract["development_gate"]
    finite = _finite_tree(pairs)
    macro_success_delta = statistics.mean(pair["success_delta"] for pair in pairs)
    success_ok = macro_success_delta >= -float(gate["success_noninferiority_margin"])
    collision_by_scenario = {
        scenario: statistics.mean(
            pair["collision_delta"] for pair in pairs if pair["scenario"] == scenario
        )
        for scenario in required
    }
    collision_ok = all(
        value <= float(gate["collision_noninferiority_margin"])
        for value in collision_by_scenario.values()
    )
    worst_seed_delta = min(pair["success_delta"] for pair in pairs)
    worst_seed_ok = worst_seed_delta >= -float(gate["worst_seed_success_margin"])
    candidate_only = [pair["candidate"] for pair in pairs]
    mechanism_checks = gate["mechanism_checks"]
    effective = [
        row["evaluation_mechanism"].get("topology_effective_lanes")
        for row in candidate_only
    ]
    compatible = [
        row["evaluation_mechanism"].get("route_compatible_attention_mass")
        for row in candidate_only
    ]
    fallback = [
        row["evaluation_mechanism"].get("topology_fallback_rate")
        for row in candidate_only
    ]
    mechanism_finite = all(
        value is not None and math.isfinite(float(value))
        for value in (*effective, *compatible, *fallback)
    )
    mechanism_result = {
        "topology_effective_lanes": {
            "observed_max": max(effective) if mechanism_finite else None,
            "threshold_max": float(mechanism_checks["topology_effective_lanes_max"]),
            "passed": mechanism_finite
            and max(effective) <= float(mechanism_checks["topology_effective_lanes_max"]),
        },
        "route_compatible_attention_mass": {
            "observed_min": min(compatible) if mechanism_finite else None,
            "threshold_min": float(mechanism_checks["route_compatible_attention_mass_min"]),
            "passed": mechanism_finite
            and min(compatible) >= float(mechanism_checks["route_compatible_attention_mass_min"]),
        },
        "topology_fallback_rate": {
            "observed_max": max(fallback) if mechanism_finite else None,
            "threshold_max": float(mechanism_checks["topology_fallback_rate_max"]),
            "passed": mechanism_finite
            and max(fallback) <= float(mechanism_checks["topology_fallback_rate_max"]),
        },
    }
    positive_scenarios = gate["positive_effect_any"]["scenarios"]
    scenario_deltas = {
        scenario: {
            "success": statistics.mean(
                pair["success_delta"] for pair in pairs if pair["scenario"] == scenario
            ),
            "collision": statistics.mean(
                pair["collision_delta"] for pair in pairs if pair["scenario"] == scenario
            ),
        }
        for scenario in positive_scenarios
    }
    positive = any(
        values["success"] >= float(gate["positive_effect_any"]["minimum_success_gain"])
        or -values["collision"]
        >= float(gate["positive_effect_any"]["minimum_collision_reduction"])
        for values in scenario_deltas.values()
    )
    checks = {
        "finite_all_metrics": finite and mechanism_finite,
        "aggregate_success_noninferiority": success_ok,
        "per_scenario_collision_noninferiority": collision_ok,
        "worst_paired_seed_success": worst_seed_ok,
        "mechanism": mechanism_result,
        "positive_effect_any": positive,
    }
    passed = bool(
        checks["finite_all_metrics"]
        and success_ok
        and collision_ok
        and worst_seed_ok
        and all(row["passed"] for row in mechanism_result.values())
        and positive
    )
    payload = {
        "schema_version": "topology-temporal-v2-development-gate/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "experiment_contract_sha256": contract_digest,
        "candidate": candidate,
        "decision": "pass" if passed else "fail",
        "formal_test_unlocked": passed,
        "checks": checks,
        "observed": {
            "macro_success_delta": macro_success_delta,
            "collision_delta_by_scenario": collision_by_scenario,
            "worst_paired_seed_success_delta": worst_seed_delta,
            "positive_scenario_deltas": scenario_deltas,
        },
        "thresholds": gate,
        "paired_evidence": [
            {
                "scenario": pair["scenario"],
                "seed": pair["seed"],
                "success_delta": pair["success_delta"],
                "collision_delta": pair["collision_delta"],
            }
            for pair in pairs
        ],
    }
    path = _resolve_project_path(str(contract["phases"]["formal_test"]["gate_receipt"]))
    _write_json(path, payload)
    return payload


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for name in ("plan", "run", "summarize"):
        child = subparsers.add_parser(name)
        child.add_argument("--phase", choices=PHASE_ORDER, required=True)
        child.add_argument("--methods")
        child.add_argument("--scenarios")
        child.add_argument("--seeds")
        if name in ("plan", "run"):
            child.add_argument("--device", default="cuda")
        if name == "run":
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--max-jobs", type=int)
            child.add_argument(
                "--workers",
                type=int,
                default=1,
                help="Independent training processes; benchmark before increasing.",
            )
            child.add_argument("--continue-on-error", action="store_true")
    subparsers.add_parser("select-candidate")
    subparsers.add_parser("development-gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = validate_contract(load_contract(contract_path))
    digest = contract_sha256(contract_path)
    if args.command == "validate":
        dependency_hashes = frozen_dependency_hashes(contract)
        payload = {
            "contract": str(contract_path),
            "experiment_contract_sha256": digest,
            "valid": True,
            "no_fabrication": contract["no_fabrication"],
            "frozen_dependencies": dependency_hashes,
            "methods": list(contract["methods"]),
            "phases": list(contract["phases"]),
            "formal_test_locked": not _resolve_project_path(
                str(contract["phases"]["formal_test"]["gate_receipt"])
            ).is_file(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-candidate":
        print(json.dumps(select_candidate(contract, digest), ensure_ascii=False, indent=2))
        return 0
    if args.command == "development-gate":
        payload = compute_development_gate(contract, digest)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["decision"] == "pass" else 1
    jobs = make_jobs(
        contract,
        args.phase,
        contract_digest=digest,
        methods=args.methods,
        scenarios=args.scenarios,
        seeds=args.seeds,
    )
    if args.command == "plan":
        return execute_jobs(
            contract,
            contract_path,
            jobs,
            device=args.device,
            contract_digest=digest,
            dry_run=True,
            max_jobs=None,
            continue_on_error=False,
            workers=1,
        )
    if args.command == "run":
        return execute_jobs(
            contract,
            contract_path,
            jobs,
            device=args.device,
            contract_digest=digest,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            continue_on_error=args.continue_on_error,
            workers=args.workers,
        )
    if args.command == "summarize":
        return summarize_phase(
            contract,
            contract_path,
            args.phase,
            jobs,
            contract_digest=digest,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentContractError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
