"""Validate the frozen six-method, six-scenario experiment matrix contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs import source_action_repeat
from envs.sumo.scenario_registry import available_scenarios
from tools import reproduce_paper_sb3_sumo
from tools.train_sb3 import argument_parser as training_argument_parser


DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "experiments" / "systematic_matrix" / "protocol.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "contracts"
    / "systematic_matrix_protocol_validation.json"
)

EXPECTED_METHODS = {
    "sac": ("sac", "paper_sac_lstm_reconstruction_v2", 3),
    "ppo": ("ppo", "released_source_ppo_pytorch_port_v1", 1),
    "mst": ("mst", "released_mst_pytorch_port_v1", 3),
    "mst_slt": ("scene_rep", "released_scene_rep_pytorch_port_v1", 3),
    "temporal_graph": (
        "temporal_graph",
        "temporal_vehicle_graph_slt_pytorch_v1",
        3,
    ),
    "full_balanced": (
        "topo_scene_balanced",
        "topo_temporal_graph_slt_balanced_slots_v1",
        3,
    ),
}
EXPECTED_SCENARIOS = (
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
)
EXPECTED_PROFILES = {
    "development_50k": (50_000, 10_000),
    "paper": (100_000, 10_000),
    "source_release_1m": (1_000_000, 100_000),
}
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_EPISODES = 50
EXPECTED_RUNS = len(EXPECTED_METHODS) * len(EXPECTED_SCENARIOS) * len(EXPECTED_SEEDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class CheckLedger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, check_id: str, condition: bool, **details: Any) -> None:
        self.rows.append(
            {"id": check_id, "passed": bool(condition), **details}
        )

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row["passed"] for row in self.rows)


def _training_cli_algorithms() -> set[str]:
    parser = training_argument_parser()
    for action in parser._actions:
        if action.dest == "algo":
            return set(action.choices or ())
    raise RuntimeError("Training parser does not expose --algo")


def _make_all_jobs(protocol: dict[str, Any], profile: str) -> list[Any]:
    args = Namespace(
        profile=profile,
        methods="all",
        scenarios="all",
        seeds="all",
    )
    return reproduce_paper_sb3_sumo.make_jobs(args, protocol)


def validate(
    protocol_path: Path,
    preflight_report: Path | None = None,
) -> dict[str, Any]:
    ledger = CheckLedger()
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "passed": False,
            "protocol": str(protocol_path),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": [],
        }

    ledger.add(
        "schema_version",
        protocol.get("schema_version") == "ccfa.systematic-matrix-protocol/v1",
        actual=protocol.get("schema_version"),
    )
    ledger.add("no_fabrication", protocol.get("no_fabrication") is True)

    matrix = protocol.get("matrix_contract", {})
    ledger.add(
        "method_order",
        tuple(matrix.get("method_order", ())) == tuple(EXPECTED_METHODS),
        expected=list(EXPECTED_METHODS),
        actual=matrix.get("method_order"),
    )
    ledger.add(
        "scenario_order",
        tuple(protocol.get("scenario_order", ())) == EXPECTED_SCENARIOS
        and tuple(matrix.get("scenario_order", ())) == EXPECTED_SCENARIOS,
        expected=list(EXPECTED_SCENARIOS),
        protocol_actual=protocol.get("scenario_order"),
        matrix_actual=matrix.get("scenario_order"),
    )
    ledger.add(
        "training_seeds",
        tuple(matrix.get("training_seeds", ())) == EXPECTED_SEEDS,
        expected=list(EXPECTED_SEEDS),
        actual=matrix.get("training_seeds"),
    )
    ledger.add(
        "test_episodes",
        matrix.get("test_episodes_per_policy") == EXPECTED_EPISODES,
        expected=EXPECTED_EPISODES,
        actual=matrix.get("test_episodes_per_policy"),
    )
    ledger.add(
        "declared_matrix_size",
        matrix.get("expected_method_scenario_cells") == 36
        and matrix.get("expected_training_runs") == EXPECTED_RUNS
        and matrix.get("expected_test_episodes_per_method_scenario") == 150,
        expected_runs=EXPECTED_RUNS,
        declared_runs=matrix.get("expected_training_runs"),
    )

    registered_scenarios = set(available_scenarios())
    ledger.add(
        "scenario_registry",
        set(EXPECTED_SCENARIOS).issubset(registered_scenarios),
        registered=sorted(registered_scenarios),
    )

    cli_algorithms = _training_cli_algorithms()
    methods = protocol.get("supported_methods", {})
    ledger.add(
        "method_keys",
        tuple(methods) == tuple(EXPECTED_METHODS),
        expected=list(EXPECTED_METHODS),
        actual=list(methods),
    )
    for method_name, (algorithm, implementation_id, action_repeat) in EXPECTED_METHODS.items():
        method = methods.get(method_name, {})
        ledger.add(
            f"method_{method_name}_identity",
            method.get("train_cli_algorithm") == algorithm
            and method.get("implementation_id") == implementation_id
            and algorithm in cli_algorithms,
            expected_algorithm=algorithm,
            actual_algorithm=method.get("train_cli_algorithm"),
            expected_implementation_id=implementation_id,
            actual_implementation_id=method.get("implementation_id"),
        )
        ledger.add(
            f"method_{method_name}_scenario_coverage",
            tuple(method.get("scenarios", ())) == EXPECTED_SCENARIOS,
            expected=list(EXPECTED_SCENARIOS),
            actual=method.get("scenarios"),
        )
        resolved_repeat: int | None
        try:
            resolved_repeat = source_action_repeat(algorithm, action_repeat)
        except Exception:
            resolved_repeat = None
        ledger.add(
            f"method_{method_name}_clock",
            method.get("environment_action_repeat") == action_repeat
            and resolved_repeat == action_repeat,
            expected=action_repeat,
            actual=method.get("environment_action_repeat"),
            resolved=resolved_repeat,
        )

    environment = protocol.get("environment_protocol", {})
    ledger.add(
        "matched_environment",
        environment
        == {
            "ego_control_profile": "direct",
            "traffic_protocol": "frozen_80_20",
            "episode_limit_profile": "source",
        },
        actual=environment,
    )
    anchor = protocol.get("environment_anchor", {})
    anchor_path = PROJECT_ROOT / str(anchor.get("run_arguments_path", ""))
    anchor_payload: dict[str, Any] = {}
    anchor_hash = _sha256(anchor_path) if anchor_path.is_file() else None
    if anchor_path.is_file():
        anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_arguments = anchor_payload.get("requested_raw_steps", {})
    locked_fields = anchor.get("locked_environment_fields", {})
    anchor_fields_match = all(
        anchor_arguments.get(field) == expected
        for field, expected in locked_fields.items()
    )
    protocol_fields_match = all(
        (
            environment.get(field) == expected
            if field in environment
            else protocol.get("common_hyperparameters", {}).get(field) == expected
        )
        for field, expected in locked_fields.items()
    )
    ledger.add(
        "full_balanced_environment_anchor",
        anchor.get("method") == "full_balanced"
        and anchor_hash == anchor.get("run_arguments_sha256")
        and anchor_fields_match
        and protocol_fields_match,
        path=str(anchor_path),
        expected_sha256=anchor.get("run_arguments_sha256"),
        actual_sha256=anchor_hash,
        anchor_fields_match=anchor_fields_match,
        protocol_fields_match=protocol_fields_match,
    )

    profiles = protocol.get("profiles", {})
    ledger.add(
        "profile_keys",
        tuple(profiles) == tuple(EXPECTED_PROFILES),
        expected=list(EXPECTED_PROFILES),
        actual=list(profiles),
    )
    profile_job_counts: dict[str, int] = {}
    profile_unique_counts: dict[str, int] = {}
    for profile_name, (raw_steps, checkpoint_frequency) in EXPECTED_PROFILES.items():
        profile = profiles.get(profile_name, {})
        profile_ok = (
            profile.get("raw_training_steps") == raw_steps
            and tuple(profile.get("seeds", ())) == EXPECTED_SEEDS
            and profile.get("test_episodes") == EXPECTED_EPISODES
            and profile.get("checkpoint_frequency_raw_steps")
            == checkpoint_frequency
            and profile.get("evaluation_frequency_raw_steps") == 0
            and raw_steps % checkpoint_frequency == 0
        )
        ledger.add(
            f"profile_{profile_name}",
            profile_ok,
            raw_steps=profile.get("raw_training_steps"),
            checkpoint_frequency=profile.get("checkpoint_frequency_raw_steps"),
            seeds=profile.get("seeds"),
            test_episodes=profile.get("test_episodes"),
        )
        try:
            jobs = _make_all_jobs(protocol, profile_name)
            names = {job.name for job in jobs}
        except Exception as exc:
            jobs = []
            names = set()
            ledger.add(
                f"profile_{profile_name}_job_generation_error",
                False,
                error=f"{type(exc).__name__}: {exc}",
            )
        profile_job_counts[profile_name] = len(jobs)
        profile_unique_counts[profile_name] = len(names)
        ledger.add(
            f"profile_{profile_name}_job_matrix",
            len(jobs) == EXPECTED_RUNS and len(names) == EXPECTED_RUNS,
            generated=len(jobs),
            unique=len(names),
            expected=EXPECTED_RUNS,
        )
        if jobs:
            cells = {(job.method, job.scenario) for job in jobs}
            seed_values = {job.seed for job in jobs}
            ledger.add(
                f"profile_{profile_name}_coverage",
                len(cells) == 36 and seed_values == set(EXPECTED_SEEDS),
                method_scenario_cells=len(cells),
                seeds=sorted(seed_values),
            )

    artifact_contract = protocol.get("artifact_contract", {})
    required_run_artifacts = set(artifact_contract.get("run_required", ()))
    ledger.add(
        "run_artifact_contract",
        {
            "arguments.json",
            "final_model.zip",
            "checkpoint_audit.json",
            "final_evaluation.json",
            "paper_evaluation_detailed.json",
            "training_diagnostics.json",
            "performance_profile.json",
        }.issubset(required_run_artifacts),
        actual=sorted(required_run_artifacts),
    )
    ledger.add(
        "paper_reference_compatibility",
        isinstance(protocol.get("paper_reference_percent"), dict),
    )

    if preflight_report is not None:
        report: dict[str, Any] = {}
        report_error: str | None = None
        if preflight_report.is_file():
            try:
                report = json.loads(preflight_report.read_text(encoding="utf-8"))
            except Exception as exc:
                report_error = f"{type(exc).__name__}: {exc}"
        ledger.add(
            "preflight_report_exists",
            preflight_report.is_file() and report_error is None,
            path=str(preflight_report),
            error=report_error,
        )
        ledger.add(
            "preflight_passed",
            report.get("status") == "ok"
            and report.get("passed") is True
            and report.get("jobs_requested") == EXPECTED_RUNS,
            status=report.get("status"),
            passed=report.get("passed"),
            jobs_requested=report.get("jobs_requested"),
            expected_jobs=EXPECTED_RUNS,
        )
        ledger.add(
            "all_algorithm_scenario_pairs_checked",
            report.get("unique_algorithm_scenario_pairs_expected") == 36
            and report.get("unique_algorithm_scenario_pairs_checked") == 36
            and len(report.get("results", ())) == 36
            and all(
                row.get("returncode") == 0
                for row in report.get("results", ())
            ),
            pairs_expected=report.get(
                "unique_algorithm_scenario_pairs_expected"
            ),
            pairs_checked=report.get(
                "unique_algorithm_scenario_pairs_checked"
            ),
            result_count=len(report.get("results", ())),
        )
        ledger.add(
            "preflight_protocol_hash_matches",
            report.get("protocol_sha256") == _sha256(protocol_path),
            expected=_sha256(protocol_path),
            actual=report.get("protocol_sha256"),
        )

    return {
        "passed": ledger.passed,
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "expected_runs_per_profile": EXPECTED_RUNS,
        "expected_final_test_episodes_per_profile": EXPECTED_RUNS
        * EXPECTED_EPISODES,
        "profile_job_counts": profile_job_counts,
        "profile_unique_job_counts": profile_unique_counts,
        "checks": ledger.rows,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Optionally require a complete preflight report for this protocol.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    preflight_report = (
        args.preflight_report.resolve()
        if args.preflight_report is not None
        else None
    )
    result = validate(args.protocol.resolve(), preflight_report)
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
