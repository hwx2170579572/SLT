"""Preview or launch the sorted ST-RT route-action/conflict-time pair.

Preview is the default and writes nothing. ``--start`` creates a fresh run
root, archives the exact source and traffic inputs, then starts the existing
D1 two-worker runner. This file does not itself create environments or train.
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

import launch_sorted_routeaware_3slot as common


PROJECT = common.PROJECT
WORKSPACE = common.WORKSPACE
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
METHODS = (
    "sac_mlp_d1_st_rt_routeact_v1",
    "sac_mlp_d1_st_rt_conflicttime_v1",
)
DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "sortct_1002"
RAW_STEPS = 100_000
CHECKPOINT_RAW = 10_000
EVAL_EPISODES = 100
EVAL_SEED_START = 10_000
EVAL_SPLIT = "validation"
SHADOW_TRAIN_INTERVAL_RAW = 5_000
SHADOW_TRAIN_MAX_STATES = 20
SHADOW_EVAL_MAX_PER_EPISODE = 4
SAFE_PATH_LIMIT = 240
PROTOCOL_DOCUMENT = "route_action_conflict_timing_protocol_20261002.md"

METHOD_DESIGN = {
    METHODS[0]: {
        "change": "execution-side route/action consistency mapping",
        "network_and_observation": "unchanged from sac_mlp_d1_st_rt",
        "replay_action": "policy proposal; environment wrapper may forward a safe lane request",
    },
    METHODS[1]: {
        "change": "adds conflict_timing [actors,20] and zero-initialized 20-32-128 social K/V residual",
        "network_base": "ST-RT retained; topology and slots disabled",
        "input_comparability": "adds static route/geometry-derived information; not equal-information ablation",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_method_config(_base, d1, method: str) -> dict:
    if method not in d1.DISPATCH or method not in d1.D1_CONFIG:
        raise ValueError(f"D1 method is not registered: {method}")
    if d1.PARENT.get(method) != "sac_mlp":
        raise ValueError(f"{method} must inherit the pure SAC+MLP baseline")
    if d1._env_adapter_d1(method) != "base":
        raise ValueError(f"{method} must retain the base SAC/MLP environment contract")

    cfg = dict(d1.D1_CONFIG[method])
    common_expected = {
        "use_route": True,
        "use_topology": False,
        "use_slots": False,
        "use_graph_slt": False,
        "use_sbs": False,
        "representation_coef": 0.0,
        "slot_balance_coef": 0.0,
    }
    if method == METHODS[0]:
        expected = {
            **common_expected,
            "use_route_action_veto": True,
            "use_route_conflict_timing": False,
        }
    elif method == METHODS[1]:
        expected = {
            **common_expected,
            "use_route_action_veto": False,
            "use_route_conflict_timing": True,
        }
    else:  # pragma: no cover - METHODS is a fixed pair.
        raise ValueError(method)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Unexpected {method}.{key}: {cfg.get(key)!r}; expected {value!r}"
            )
    return cfg


def _configure_common() -> None:
    common.SCENARIO = SCENARIO
    common.DEPART_SCALE = DEPART_SCALE
    common.METHODS = METHODS
    common.DEFAULT_RUN_ROOT = DEFAULT_RUN_ROOT
    common.RAW_STEPS = RAW_STEPS
    common.CHECKPOINT_FREQUENCY_RAW = CHECKPOINT_RAW
    common.EVAL_EPISODES = EVAL_EPISODES
    common.EVAL_SEED_START = EVAL_SEED_START
    common.EVAL_SPLIT_LABEL = EVAL_SPLIT
    common.SAFE_PATH_LIMIT = SAFE_PATH_LIMIT
    common._assert_method_config = _assert_method_config


def _worker_command(run_root: Path) -> list[str]:
    expected_project = Path(__file__).resolve().parents[1]
    project = PROJECT.resolve()
    if project != expected_project:
        raise RuntimeError(
            f"launcher project mismatch: PROJECT={project}, script project={expected_project}"
        )
    entrypoint = (project / "fast-developer" / "train_intersection_yield_v2_d1.py").resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"D1 worker entrypoint is missing: {entrypoint}")
    return [
        sys.executable,
        "-u",
        str(entrypoint),
        "--scenario", SCENARIO,
        "--depart-scale", str(DEPART_SCALE),
        "--eval-traffic-split", EVAL_SPLIT,
        "--methods", ",".join(METHODS),
        "--max-steps", str(RAW_STEPS),
        "--run-root", str(run_root.resolve()),
        "--checkpoint-frequency", str(CHECKPOINT_RAW),
        "--behavior-diagnostics",
        "--policy-shadow-probes",
        "--policy-shadow-train-raw-interval", str(SHADOW_TRAIN_INTERVAL_RAW),
        "--policy-shadow-eval-max-per-episode", str(SHADOW_EVAL_MAX_PER_EPISODE),
        "--policy-shadow-critical-eval-sample",
    ]


def _run_dir(run_root: Path, method: str) -> Path:
    return run_root / f"{method}__{SCENARIO}_depart4p0"


def _validate_required_wrappers() -> dict[str, str]:
    # Import-only preflight: confirms the method wrappers exist without creating
    # an environment, starting SUMO, or stepping a simulator.
    from envs.sumo.route_action_consistency import RouteActionConsistencyWrapper
    from envs.sumo.conflict_timing_observation import RouteConflictTimingObservationWrapper

    return {
        "route_action_consistency": (
            f"{RouteActionConsistencyWrapper.__module__}.{RouteActionConsistencyWrapper.__name__}"
        ),
        "route_conflict_timing": (
            f"{RouteConflictTimingObservationWrapper.__module__}.{RouteConflictTimingObservationWrapper.__name__}"
        ),
    }


def validate_runtime_plan(run_root: Path) -> dict:
    base, d1, _, _ = common.load_runtime_modules()
    wrappers = _validate_required_wrappers()
    plan = common.validate_runtime_plan(run_root)
    if d1.TRAIN_WORKERS != 2:
        raise ValueError(f"expected two independent D1 workers, got {d1.TRAIN_WORKERS}")
    configs = {method: _assert_method_config(base, d1, method) for method in METHODS}
    plan.update(
        {
            "methods": list(METHODS),
            "method_configs": configs,
            "method_design": METHOD_DESIGN,
            "method_wrappers": wrappers,
            "raw_steps_per_method": RAW_STEPS,
            "checkpoint_frequency_raw_steps": CHECKPOINT_RAW,
            "shadow": {
                "enabled": True,
                "train_raw_interval": SHADOW_TRAIN_INTERVAL_RAW,
                "train_unique_sample_cap": SHADOW_TRAIN_MAX_STATES,
                "eval_unique_state_cap_per_episode": SHADOW_EVAL_MAX_PER_EPISODE,
                "critical_eval_sampling": True,
                "additional_environment_steps": 0,
            },
        }
    )
    return plan


def _archive_protocol_document(run_root: Path, archive: dict) -> None:
    source = PROJECT / "fast-developer" / "analysis" / PROTOCOL_DOCUMENT
    if not source.is_file():
        raise FileNotFoundError(f"required protocol document is missing: {source}")
    common._archive_file(
        source,
        run_root / "source_archive" / "research" / source.name,
        archive["files"],
    )
    archive["source_file_count"] = len(archive["files"])
    common.write_json(run_root / "source_archive" / "manifest.json", archive)


def _validate_shadow_streams(run_root: Path) -> dict:
    from algos.sb3_torch.policy_shadow_probes import PROBE_NAMES

    errors: list[str] = []
    methods: dict[str, dict] = {}
    for method in METHODS:
        run_dir = _run_dir(run_root, method)
        args_path = run_dir / "arguments.json"
        row = {"phases": {}}
        if not args_path.is_file():
            errors.append(f"{method}: missing arguments.json")
            methods[method] = {"status": "incomplete", "phases": {}}
            continue
        args = json.loads(args_path.read_text(encoding="utf-8"))
        for name, expected in (
            ("behavior_diagnostics", True),
            ("policy_shadow_probes", True),
            ("policy_shadow_critical_eval_sample", True),
            ("policy_shadow_eval_max_per_episode", SHADOW_EVAL_MAX_PER_EPISODE),
            ("policy_shadow_train_raw_interval", SHADOW_TRAIN_INTERVAL_RAW),
        ):
            if args.get(name) != expected:
                errors.append(f"{method}: unexpected arguments.{name}={args.get(name)!r}")

        for phase, sample_budget in (
            ("train", SHADOW_TRAIN_MAX_STATES),
            ("eval", EVAL_EPISODES * SHADOW_EVAL_MAX_PER_EPISODE),
        ):
            diag_dir = run_dir / "diagnostics" / phase
            stream = diag_dir / "policy_shadow_probes.jsonl"
            summary_path = diag_dir / "policy_shadow_probes_summary.json"
            if not stream.is_file() or not summary_path.is_file():
                errors.append(f"{method}: missing {phase} shadow stream or summary")
                row["phases"][phase] = {"status": "missing"}
                continue
            rows = [
                json.loads(line)
                for line in stream.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            by_sample: dict[str, set[str]] = {}
            for item in rows:
                by_sample.setdefault(str(item.get("sample_id")), set()).add(
                    str(item.get("probe_name"))
                )
            complete = all(probes == set(PROBE_NAMES) for probes in by_sample.values())
            exceptions = sum(
                item.get("invalid_reason") in {
                    "probe_exception", "baseline_capture_failed", "sample_capture_error",
                    "encoder_probe_context_unsupported",
                }
                for item in rows
            )
            phase_ok = (
                len(by_sample) > 0
                and len(by_sample) <= sample_budget
                and summary.get("unique_samples") == len(by_sample)
                and summary.get("error_rows", 0) == 0
                and exceptions == 0
                and complete
                and (
                    summary.get("training_complete") is True
                    if phase == "train"
                    else summary.get("evaluation_complete") is True
                )
            )
            if not phase_ok:
                errors.append(f"{method}: {phase} shadow budget/schema/completion failed")
            row["phases"][phase] = {
                "status": "passed" if phase_ok else "failed",
                "unique_samples": len(by_sample),
                "rows": len(rows),
                "summary_unique_samples": summary.get("unique_samples"),
                "error_rows": summary.get("error_rows"),
                "complete_probe_sets": complete,
                "applicable_rows": summary.get("applicable_rows"),
                "inactive_rows": summary.get("inactive_rows"),
                "active_invalid_rows": summary.get("active_invalid_rows"),
            }
        methods[method] = row
    return {
        "status": "passed" if not errors else "failed",
        "methods": methods,
        "errors": errors,
    }


def _worker_execution(command: list[str]) -> dict:
    source_paths = {
        "d1_entrypoint": PROJECT / "fast-developer" / "train_intersection_yield_v2_d1.py",
        "base_runner": PROJECT / "fast-developer" / "train_intersection_yield_v2.py",
        "behavior_diagnostics": PROJECT / "fast-developer" / "behavior_diagnostics.py",
        "incremental_topo_encoder": PROJECT / "algos" / "sb3_torch" / "incremental_topo_encoder.py",
        "policy_shadow_probes": PROJECT / "algos" / "sb3_torch" / "policy_shadow_probes.py",
        "sac": PROJECT / "algos" / "sb3_torch" / "sac.py",
        "policies": PROJECT / "algos" / "sb3_torch" / "policies.py",
        "replay_buffer": PROJECT / "algos" / "sb3_torch" / "replay_buffer.py",
        "route_action_wrapper": PROJECT / "envs" / "sumo" / "route_action_consistency.py",
        "conflict_timing_wrapper": PROJECT / "envs" / "sumo" / "conflict_timing_observation.py",
        "conflict_timing_kernel": PROJECT / "envs" / "sumo" / "route_conflict_timing.py",
    }
    resolved_sources = {}
    for label, path in source_paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"worker source is missing: {resolved}")
        resolved_sources[label] = {
            "path": str(resolved),
            "sha256": common.sha256(resolved),
        }
    return {
        "python_executable": str(Path(command[0]).resolve()),
        "entrypoint": str(Path(command[2]).resolve()),
        "cwd": str(PROJECT.resolve()),
        "expected_project_root": str(PROJECT.resolve()),
        "source_files": resolved_sources,
    }


def _launch_plan(run_root: Path, command: list[str], runtime_plan: dict) -> dict:
    payload = {
        "experiment_id": "sorted_st_rt_routeact_vs_conflict_timing_fresh100k",
        "scenario": SCENARIO,
        "depart_scale": DEPART_SCALE,
        "methods": list(METHODS),
        "method_configs": runtime_plan["method_configs"],
        "method_design": METHOD_DESIGN,
        "training_seed": 0,
        "raw_steps_per_method": RAW_STEPS,
        "learning_starts_raw_steps": 5_000,
        "checkpoint_frequency_raw_steps": CHECKPOINT_RAW,
        "expected_updates": 95_001,
        "action_repeat": 3,
        "evaluation": {
            "episodes": EVAL_EPISODES,
            "traffic_split": EVAL_SPLIT,
            "seed_start": EVAL_SEED_START,
            "seed_end_inclusive": EVAL_SEED_START + EVAL_EPISODES - 1,
            "deterministic": True,
            "checkpoint": "final_model.zip",
        },
        "shadow_diagnostics": {
            "enabled": True,
            "train_raw_interval": SHADOW_TRAIN_INTERVAL_RAW,
            "train_max_unique_samples": SHADOW_TRAIN_MAX_STATES,
            "eval_cap_per_episode": SHADOW_EVAL_MAX_PER_EPISODE,
            "critical_eval_sample": True,
            "extra_environment_steps": 0,
        },
        "command": command,
        "worker_execution": _worker_execution(command),
        "run_root": str(run_root.resolve()),
    }
    payload["launch_plan_sha256"] = _canonical_hash(payload)
    return payload


def launch(run_root: Path, runtime_plan: dict) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    base, d1, _, _ = common.load_runtime_modules()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = run_root
    effective_traffic = tuple(base._traffic_paths_for_scale(DEPART_SCALE))
    assets = common.inspect_sorted_assets()
    archive = common.archive_sources(run_root, effective_traffic)
    _archive_protocol_document(run_root, archive)

    command = _worker_command(run_root)
    launch_plan = _launch_plan(run_root, command, runtime_plan)
    common.write_json(run_root / "launch_plan.json", launch_plan)
    manifest = {
        **launch_plan,
        "created_at_utc": _utc_now(),
        "protocol_revision": "environment_step_reward_v2_sorted_routeact_conflicttime_v1",
        "scenario_revision": "intersection_sorted_depart_sorted_templates",
        "traffic_template_count": assets["traffic_template_count"],
        "training_evaluation_traffic_pool": "same complete effective 30-template pool; no holdout",
        "fresh_model": True,
        "resume": False,
        "smoke": False,
        "max_cuda_workers": 2,
        "worker_execution": launch_plan["worker_execution"],
        "worker_processes_independent": True,
        "cuda_device_per_worker": "cuda:0 (shared GPU device, matching the established sorted launcher)",
        "training_hyperparameters": runtime_plan["training_hyperparameters"],
        "training_reward_config": runtime_plan["training_reward_config"],
        "evaluation_protocol": runtime_plan["evaluation"],
        "terminal_outcome_protocol": assets["terminal_outcome_protocol"],
        "static_sumocfg_collision_action": assets["sumo_config_collision_action"],
        "effective_traffic_hashes": archive["effective_traffic_pool"],
        "source_archive_manifest": str(run_root / "source_archive" / "manifest.json"),
        "source_archive_file_count": archive["source_file_count"],
    }
    common.write_json(run_root / "suite_manifest.json", manifest)

    status_path = run_root / "sorted_routeact_conflicttime_status.json"
    status = {
        "status": "starting",
        "created_at_utc": _utc_now(),
        "launcher_pid": os.getpid(),
        "run_root": str(run_root.resolve()),
        "methods": list(METHODS),
        "smoke": False,
        "command": command,
        "worker_execution": launch_plan["worker_execution"],
    }
    common.write_json(status_path, status)
    stdout_path = run_root / "supervisor.stdout.log"
    stderr_path = run_root / "supervisor.stderr.log"
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT.resolve()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        status.update(status="running", supervisor_pid=process.pid, started_at_utc=_utc_now())
        common.write_json(status_path, status)
        exit_code = process.wait()
    status.update(
        status="child_complete" if exit_code == 0 else "child_failed",
        child_exit_code=exit_code,
        child_finished_at_utc=_utc_now(),
    )
    common.write_json(status_path, status)
    if exit_code != 0:
        return exit_code

    evaluation_validation = common.validate_completed_outputs(run_root)
    shadow_validation = _validate_shadow_streams(run_root)
    common.write_json(run_root / "policy_shadow_validation.json", shadow_validation)
    if evaluation_validation["status"] != "passed" or shadow_validation["status"] != "passed":
        status.update(
            status="validation_failed",
            evaluation_validation=evaluation_validation["status"],
            shadow_validation=shadow_validation["status"],
            exit_code=2,
        )
        common.write_json(status_path, status)
        return 2

    try:
        from paired_stage_comparison import compare_stages

        pairing = compare_stages(
            _run_dir(run_root, METHODS[0]) / "evaluation_results.json",
            _run_dir(run_root, METHODS[1]) / "evaluation_results.json",
            run_root / "paired_method_comparison.json",
            label="route_action_vs_conflict_timing",
            expected_episodes=EVAL_EPISODES,
        )
        status["pairing_status"] = pairing.get("status", "written")
    except Exception as error:
        status["pairing_status"] = "failed"
        status["pairing_error"] = repr(error)
    status.update(status="complete", exit_code=0, finished_at_utc=_utc_now())
    common.write_json(status_path, status)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--start", action="store_true", help="archive inputs and start two workers")
    args = parser.parse_args(argv)
    _configure_common()
    run_root = (args.run_root or DEFAULT_RUN_ROOT).expanduser().resolve()
    runs_root = (WORKSPACE / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        parser.error(f"--run-root must be a child of {runs_root}")
    if run_root.exists():
        parser.error(f"refusing existing run root: {run_root}")
    try:
        runtime_plan = validate_runtime_plan(run_root)
    except Exception as error:
        parser.error(f"preflight failed: {error}")
    command = _worker_command(run_root)
    if not args.start:
        runtime_plan["worker_execution"] = _worker_execution(command)
        print(
            json.dumps(
                {
                    "mode": "preview_only",
                    "training_started": False,
                    "run_root_exists": run_root.exists(),
                    "run_root": str(run_root),
                    "command": command,
                    "plan": runtime_plan,
                    "note": "No directory, environment, simulator, or training process was created.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return launch(run_root, runtime_plan)


if __name__ == "__main__":
    raise SystemExit(main())
