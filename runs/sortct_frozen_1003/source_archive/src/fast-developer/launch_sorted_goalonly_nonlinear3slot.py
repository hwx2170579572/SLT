"""Preview or launch the sorted-scene goal-only Topo vs nonlinear 3-slot pair.

The launcher reuses the audited sorted-scene helpers, but has its own method
pair, assertions, run root, manifest, and status files. Preview is the default;
no run directory or simulator is created unless ``--start`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_sorted_routeaware_3slot as common


PROJECT = common.PROJECT
WORKSPACE = common.WORKSPACE
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
METHODS = (
    "sac_mlp_d1_st_rt_topo_goalonly_v1",
    "sac_mlp_d1_st_rt_3slot_nonlinear_v1",
)
DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "sortg3_1002"
DEFAULT_SMOKE_ROOT = WORKSPACE / "runs" / "sortg3sm_1002"
RAW_STEPS = 100_000
CHECKPOINT_RAW = 10_000
SMOKE_RAW_STEPS = 300
SMOKE_CHECKPOINT_RAW = 100
SMOKE_WARMUP_RAW = 60
SMOKE_EVAL_EPISODES = 1
POLICY_SHADOW_TRAIN_INTERVAL_RAW = 5_000
SMOKE_POLICY_SHADOW_TRAIN_INTERVAL_RAW = 100
POLICY_SHADOW_EVAL_MAX_PER_EPISODE = 3
POLICY_SHADOW_TRAIN_MAX_UNIQUE_SAMPLES = 20
EVAL_EPISODES = 100
EVAL_SEED_START = 10_000
EVAL_SPLIT = "validation"
SAFE_PATH_LIMIT = 240
METHOD_DESIGN = {
    METHODS[0]: {
        "description": "ST-RT actor-intent and geometry-only relation path; route mask on goal-only topology readout",
        "goal_topology": True,
        "actor_intent_topology": False,
        "relation_topology_injection": False,
        "route_reachability_mask": True,
        "active_readout": "joint_mlp",
        "active_readout_parameters": 65_792,
    },
    METHODS[1]: {
        "description": "nonlinear parameter-matched 3-slot head; Topo/Graph-SLT/SBS disabled",
        "goal_topology": False,
        "actor_intent_topology": False,
        "relation_topology_injection": False,
        "route_reachability_mask": False,
        "slot_widths": {
            "ego": [128, 132, 32],
            "social": [128, 120, 64],
            "route": [128, 132, 32],
        },
        "relu_layers_per_branch": 2,
        "active_readout": "parameter_matched_nonlinear_3slot",
        "active_readout_parameters": 65_792,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_method_config(_base, d1, method: str) -> dict:
    """Fail closed unless D1 registration matches the requested mechanisms."""
    if method not in d1.DISPATCH or method not in d1.D1_CONFIG:
        raise ValueError(f"D1 method is not registered: {method}")
    if d1.PARENT.get(method) != "sac_mlp":
        raise ValueError(f"{method} must inherit the pure SAC+MLP baseline")
    if d1._env_adapter_d1(method) != "base":
        raise ValueError(f"{method} must retain the base SAC/MLP environment contract")
    cfg = dict(d1.D1_CONFIG[method])

    if method == METHODS[0]:
        expected = {
            "use_route": True,
            "use_topology": True,
            "use_topology_actor_intent": False,
            "use_topology_relations": False,
            "use_topology_goal": True,
            "use_route_reachability": True,
            "use_slots": False,
            "use_incremental_slots": False,
            "use_parameter_matched_nonlinear_slots": False,
            "use_graph_slt": False,
            "use_sbs": False,
            "representation_coef": 0.0,
            "slot_balance_coef": 0.0,
            "active_readout_type": "joint_mlp",
            "active_readout_parameter_count": 65_792,
        }
    elif method == METHODS[1]:
        expected = {
            "use_route": True,
            "use_topology": False,
            "use_topology_actor_intent": False,
            "use_topology_relations": False,
            "use_topology_goal": False,
            "use_route_reachability": False,
            "use_slots": True,
            "use_incremental_slots": True,
            "use_parameter_matched_nonlinear_slots": True,
            "use_graph_slt": False,
            "use_sbs": False,
            "representation_coef": 0.0,
            "slot_balance_coef": 0.0,
            "active_readout_type": "parameter_matched_nonlinear_3slot",
            "active_readout_parameter_count": 65_792,
        }
    else:  # pragma: no cover - METHODS is a two-entry constant.
        raise ValueError(method)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unexpected {method}.{key}: {cfg.get(key)!r}; expected {value!r}")
    return cfg


def _configure_common() -> None:
    """Reuse only the established sorted-traffic/protocol/archive helpers."""
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


def _worker_command(run_root: Path, *, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT / "fast-developer" / "train_intersection_yield_v2_d1.py"),
        "--scenario", SCENARIO,
        "--depart-scale", str(DEPART_SCALE),
        "--eval-traffic-split", EVAL_SPLIT,
        "--methods", ",".join(METHODS),
        "--max-steps", str(SMOKE_RAW_STEPS if smoke else RAW_STEPS),
        "--run-root", str(run_root),
        "--checkpoint-frequency", str(SMOKE_CHECKPOINT_RAW if smoke else CHECKPOINT_RAW),
        "--behavior-diagnostics",
        "--policy-shadow-probes",
        "--policy-shadow-train-raw-interval",
        str(SMOKE_POLICY_SHADOW_TRAIN_INTERVAL_RAW if smoke else POLICY_SHADOW_TRAIN_INTERVAL_RAW),
        "--policy-shadow-eval-max-per-episode",
        str(POLICY_SHADOW_EVAL_MAX_PER_EPISODE),
    ]
    if smoke:
        command.extend(("--smoke", "--smoke-eval-episodes", str(SMOKE_EVAL_EPISODES)))
    return command


def _run_dirs(run_root: Path) -> dict[str, Path]:
    return {
        method: run_root / f"{method}__{SCENARIO}_depart4p0"
        for method in METHODS
    }


def _validate_smoke(run_root: Path, *, smoke: bool = True) -> dict:
    rows = {}
    errors = []
    expected_steps = SMOKE_RAW_STEPS if smoke else RAW_STEPS
    expected_eval_episodes = SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES
    expected_train_interval = (
        SMOKE_POLICY_SHADOW_TRAIN_INTERVAL_RAW
        if smoke else POLICY_SHADOW_TRAIN_INTERVAL_RAW
    )
    for method, run_dir in _run_dirs(run_root).items():
        required = {
            "arguments": run_dir / "arguments.json",
            "training_complete": run_dir / "training_complete.json",
            "final_model": run_dir / "final_model.zip",
            "evaluation": run_dir / "evaluation_results.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            rows[method] = {"status": "incomplete", "missing": missing}
            errors.append(f"{method}: missing {missing}")
            continue
        args = common.json.loads(required["arguments"].read_text(encoding="utf-8"))
        training = common.json.loads(required["training_complete"].read_text(encoding="utf-8"))
        evaluation = common.json.loads(required["evaluation"].read_text(encoding="utf-8"))
        records = evaluation.get("episode_records") or []
        identity = evaluation.get("identity") or {}
        digest = common.sha256(required["final_model"])
        diagnostic_rows = {}
        diagnostic_failures = []
        shadow_rows_by_phase = {}
        for phase in ("train", "eval"):
            diagnostic_dir = run_dir / "diagnostics" / phase
            summary_path = diagnostic_dir / "summary.json"
            representation_path = diagnostic_dir / "representation.jsonl"
            optimization_path = diagnostic_dir / "optimization.jsonl"
            shadow_path = diagnostic_dir / "policy_shadow_probes.jsonl"
            shadow_summary_path = diagnostic_dir / "policy_shadow_probes_summary.json"
            behavior = (
                common.json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file() else {}
            )
            representation_first = (
                next((line for line in representation_path.read_text(encoding="utf-8").splitlines() if line.strip()), None)
                if representation_path.is_file() else None
            )
            optimization_first = (
                next((line for line in optimization_path.read_text(encoding="utf-8").splitlines() if line.strip()), None)
                if optimization_path.is_file() else None
            )
            shadow_rows = (
                [json.loads(line) for line in shadow_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if shadow_path.is_file() else []
            )
            shadow_summary = (
                json.loads(shadow_summary_path.read_text(encoding="utf-8"))
                if shadow_summary_path.is_file() else {}
            )
            if behavior.get("diagnostic_error_count") != 0:
                diagnostic_failures.append(f"{phase}.diagnostic_error_count")
            if representation_first is None:
                diagnostic_failures.append(f"{phase}.representation_missing")
            if phase == "train" and optimization_first is None:
                diagnostic_failures.append("train.optimization_missing")
            sample_groups: dict[str, list[dict]] = {}
            for row in shadow_rows:
                sample_groups.setdefault(str(row.get("sample_id")), []).append(row)
            probe_names = {
                "st_spatial_off", "st_temporal_current_only", "st_social_ego_only",
                "rt_intent_injection_off", "rt_route_readout_zero",
                "topology_actor_intent_off", "topology_relations_off", "topology_goal_off",
                "route_reachability_off", "slot_ego_zero", "slot_social_zero", "slot_route_zero",
            }
            shadow_exception_rows = [
                row for row in shadow_rows
                if row.get("invalid_reason") in {
                    "probe_exception", "baseline_capture_failed", "sample_capture_error",
                    "encoder_probe_context_unsupported",
                }
            ]
            shadow_groups_complete = bool(sample_groups) and all(
                {row.get("probe_name") for row in group} == probe_names
                for group in sample_groups.values()
            )
            shadow_applicability_consistent = all(
                (row.get("applicable") is True)
                or (
                    row.get("applicable") is False
                    and row.get("valid") is False
                    and row.get("invalid_reason") == "branch_inactive"
                )
                for row in shadow_rows
            )
            phase_sample_budget = (
                POLICY_SHADOW_TRAIN_MAX_UNIQUE_SAMPLES
                if phase == "train" else expected_eval_episodes * POLICY_SHADOW_EVAL_MAX_PER_EPISODE
            )
            if not shadow_summary_path.is_file() or not shadow_path.is_file():
                diagnostic_failures.append(f"{phase}.policy_shadow_stream_missing")
            if shadow_summary.get("unique_samples", 0) <= 0:
                diagnostic_failures.append(f"{phase}.policy_shadow_no_samples")
            if shadow_summary.get("unique_samples", 0) > phase_sample_budget:
                diagnostic_failures.append(f"{phase}.policy_shadow_budget_exceeded")
            if shadow_summary.get("unique_samples") != len(sample_groups):
                diagnostic_failures.append(f"{phase}.policy_shadow_unique_sample_mismatch")
            if shadow_summary.get("error_rows", 0) != 0 or shadow_exception_rows:
                diagnostic_failures.append(f"{phase}.policy_shadow_exceptions")
            if phase == "train" and shadow_summary.get("training_complete") is not True:
                diagnostic_failures.append("train.policy_shadow_incomplete")
            if phase == "eval" and shadow_summary.get("evaluation_complete") is not True:
                diagnostic_failures.append("eval.policy_shadow_incomplete")
            if not shadow_groups_complete:
                diagnostic_failures.append(f"{phase}.policy_shadow_probe_rows_incomplete")
            if not shadow_applicability_consistent:
                diagnostic_failures.append(f"{phase}.policy_shadow_applicability_inconsistent")
            if not any(row.get("applicable") is True for row in shadow_rows):
                diagnostic_failures.append(f"{phase}.policy_shadow_all_branches_inactive")
            if not any(
                row.get("applicable") is True and row.get("valid") is True
                for row in shadow_rows
            ):
                diagnostic_failures.append(f"{phase}.policy_shadow_no_valid_active_row")
            if not any(row.get("applicable") is False for row in shadow_rows):
                diagnostic_failures.append(f"{phase}.policy_shadow_no_na_rows")
            required_timing_fields = {
                "pre_obs_raw", "post_raw", "pre_obs_decision", "decision_step",
                "post_decision_step", "updates",
                "sample_id", "probe_id", "trigger", "outcome", "actual_action", "action_source",
            }
            if any(not required_timing_fields.issubset(row) for row in shadow_rows):
                diagnostic_failures.append(f"{phase}.policy_shadow_timing_fields_missing")
            shadow_rows_by_phase[phase] = {
                "rows": len(shadow_rows),
                "unique_samples": len(sample_groups),
                "summary_unique_samples": shadow_summary.get("unique_samples"),
                "error_rows": shadow_summary.get("error_rows"),
                "invalid_rows": shadow_summary.get("invalid_rows"),
                "applicable_rows": shadow_summary.get("applicable_rows"),
                "inactive_rows": shadow_summary.get("inactive_rows"),
                "active_invalid_rows": shadow_summary.get("active_invalid_rows"),
                "skip_reasons": shadow_summary.get("skip_reasons", {}),
                "critical_exception_rows": len(shadow_exception_rows),
                "sample_groups_complete": shadow_groups_complete,
                "applicability_consistent": shadow_applicability_consistent,
            }
            diagnostic_rows[phase] = {
                "diagnostic_error_count": behavior.get("diagnostic_error_count"),
                "representation_record_present": representation_first is not None,
                "optimization_record_present": optimization_first is not None,
                "summary_path": str(summary_path),
                "policy_shadow": shadow_rows_by_phase[phase],
            }
        record_protocol_ok = len(records) == expected_eval_episodes and all(
            record.get("evaluation_return_protocol_version") == common.EVALUATION_RETURN_PROTOCOL
            and record.get("raw_episode_return") is not None
            and record.get("raw_return_source") == "info.undiscounted_reward"
            and record.get("reward_component_protocol_version") == common.REWARD_COMPONENT_PROTOCOL
            and all(record.get(key) is not None for key in common.REWARD_COMPONENT_KEYS)
            and record.get("reward_component_reconciliation_error") is not None
            and float(record["reward_component_reconciliation_error"]) <= 1e-5
            for record in records
        )
        checks = {
            "method": args.get("method") == method,
            "scenario": args.get("scenario") == SCENARIO,
            "smoke": args.get("smoke") is smoke,
            "device_cuda": args.get("device") == "cuda",
            "seed0": args.get("seed") == 0,
            "raw_step_budget": training.get("raw_steps") == expected_steps,
            "diagnostics": args.get("behavior_diagnostics") is True
            and training.get("behavior_diagnostics") is True,
            "policy_shadow_probes": args.get("policy_shadow_probes") is True
            and training.get("policy_shadow_probes") is True
            and evaluation.get("identity", {}).get("policy_shadow_probes") is True,
            "policy_shadow_train_interval": args.get("policy_shadow_train_raw_interval")
            == expected_train_interval,
            "policy_shadow_eval_budget": args.get("policy_shadow_eval_max_per_episode")
            == POLICY_SHADOW_EVAL_MAX_PER_EPISODE,
            "eval_count": len(records) == expected_eval_episodes,
            "eval_seed_range": [int(record.get("seed", -1)) for record in records]
            == list(range(EVAL_SEED_START, EVAL_SEED_START + expected_eval_episodes)),
            "evaluation_reward_protocol_and_reconciliation": record_protocol_ok,
            "final_hash_training": training.get("checkpoint_sha256") == digest,
            "final_hash_evaluation": identity.get("checkpoint_sha256") == digest,
            "diagnostic_streams": not diagnostic_failures,
        }
        failures = [name for name, ok in checks.items() if not ok]
        if failures:
            errors.append(f"{method}: checks failed {failures}")
        rows[method] = {
            "status": "passed" if not failures else "failed",
            "checks": checks,
            "diagnostics": diagnostic_rows,
            "diagnostic_failures": diagnostic_failures,
        }
    return {"status": "passed" if not errors else "failed", "methods": rows, "errors": errors}


def launch(run_root: Path, plan: dict, *, smoke: bool) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    base, d1, _, _ = common.load_runtime_modules()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = run_root
    effective_traffic = base._traffic_paths_for_scale(DEPART_SCALE)
    assets = common.inspect_sorted_assets()
    archive = common.archive_sources(run_root, tuple(effective_traffic))
    for plan_name in (
        "sorted_goalonly_nonlinear3slot_plan_20261002.md",
        "sorted_goalonly_nonlinear3slot_research_protocol_20261002.md",
    ):
        plan_doc = Path(__file__).resolve().parent / "analysis" / plan_name
        if plan_doc.is_file():
            common._archive_file(
                plan_doc,
                run_root / "source_archive" / "research" / plan_doc.name,
                archive["files"],
            )
    archive["source_file_count"] = len(archive["files"])
    common.write_json(run_root / "source_archive" / "manifest.json", archive)
    command = _worker_command(run_root, smoke=smoke)
    budget = SMOKE_RAW_STEPS if smoke else RAW_STEPS
    eval_count = SMOKE_EVAL_EPISODES if smoke else EVAL_EPISODES
    checkpoint = SMOKE_CHECKPOINT_RAW if smoke else CHECKPOINT_RAW
    warmup = SMOKE_WARMUP_RAW if smoke else 5_000
    manifest = {
        "created_at_utc": _utc_now(),
        "experiment_id": "sorted_goalonly_topology_vs_param_matched_nonlinear_3slot",
        "protocol_revision": "environment_step_reward_v2_sorted_goal_slot_v1",
        "scenario": SCENARIO,
        "scenario_revision": "intersection_sorted_depart_sorted_templates",
        "depart_scale": DEPART_SCALE,
        "traffic_template_count": assets["traffic_template_count"],
        "train_eval_traffic_pool": "same complete effective 30-template pool; no holdout",
        "training_seed": 0,
        "raw_steps_per_method": budget,
        "action_repeat": 3,
        "learning_starts_raw_steps": warmup,
        "checkpoint_frequency_raw_steps": checkpoint,
        "expected_updates": None if smoke else 95_001,
        "fresh_model": True,
        "resume": False,
        "smoke": smoke,
        "max_cuda_workers": 2,
        "workers_are_independent_training_processes": True,
        "cuda_device": "cuda:0",
        "methods": list(METHODS),
        "method_design": METHOD_DESIGN,
        "method_configs": plan["method_configs"],
        "method_environment_contracts": plan["method_environment_contracts"],
        "training_hyperparameters": plan["training_hyperparameters"],
        "training_reward_config": plan["training_reward_config"],
        "policy_shadow_probes": {
            "enabled": True,
            "train_raw_interval": (
                SMOKE_POLICY_SHADOW_TRAIN_INTERVAL_RAW if smoke
                else POLICY_SHADOW_TRAIN_INTERVAL_RAW
            ),
            "train_max_unique_samples": POLICY_SHADOW_TRAIN_MAX_UNIQUE_SAMPLES,
            "eval_max_unique_samples_per_episode": POLICY_SHADOW_EVAL_MAX_PER_EPISODE,
            "eval_unique_sample_budget": eval_count * POLICY_SHADOW_EVAL_MAX_PER_EPISODE,
            "extra_environment_steps": 0,
            "same_state_deterministic_policy_forwards_only": True,
            "raw_clock_fields": ["pre_obs_raw", "post_raw"],
            "decision_clock_fields": ["pre_obs_decision", "decision_step"],
        },
        "terminal_outcome_protocol": "exclusive_terminal_v2",
        "evaluation": {
            "episodes": eval_count,
            "seed_start": EVAL_SEED_START,
            "seed_end_inclusive": EVAL_SEED_START + eval_count - 1,
            "traffic_split": EVAL_SPLIT,
            "deterministic": True,
            "checkpoint": "final_model.zip",
            "reward_protocol": common.EVALUATION_RETURN_PROTOCOL,
            "raw_reward_protocol": common.RAW_RETURN_PROTOCOL,
            "reward_component_protocol": common.REWARD_COMPONENT_PROTOCOL,
        },
        "source_archive_manifest": str(run_root / "source_archive" / "manifest.json"),
        "source_archive_file_count": archive["source_file_count"],
        "traffic_asset_hashes": archive["sorted_template_hashes"],
        "effective_traffic_hashes": archive["effective_traffic_pool"],
        "command": command,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": common.package_versions(),
        "run_root": str(run_root),
    }
    common.write_json(run_root / "suite_manifest.json", manifest)
    status_path = run_root / "sorted_goal_slot_status.json"
    status = {
        "status": "starting",
        "created_at_utc": _utc_now(),
        "launcher_pid": os.getpid(),
        "run_root": str(run_root),
        "methods": list(METHODS),
        "smoke": smoke,
        "policy_shadow_probes": True,
        "policy_shadow_train_raw_interval": (
            SMOKE_POLICY_SHADOW_TRAIN_INTERVAL_RAW if smoke
            else POLICY_SHADOW_TRAIN_INTERVAL_RAW
        ),
        "command": command,
    }
    common.write_json(status_path, status)
    stdout_path = run_root / "supervisor.stdout.log"
    stderr_path = run_root / "supervisor.stderr.log"
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        status.update(status="running", coordinator_pid=process.pid, started_at_utc=_utc_now())
        common.write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        exit_code = process.wait()
    status.update(
        status="child_complete" if exit_code == 0 else "child_failed",
        child_exit_code=exit_code,
        child_finished_at_utc=_utc_now(),
    )
    common.write_json(status_path, status)
    if exit_code != 0:
        return exit_code

    if smoke:
        validation = _validate_smoke(run_root)
        common.write_json(run_root / "smoke_validation.json", validation)
        if validation["status"] != "passed":
            status.update(status="validation_failed", validation=validation, exit_code=2)
            common.write_json(status_path, status)
            return 2
    else:
        validation = common.validate_completed_outputs(run_root)
        status["evaluation_reward_validation"] = validation["status"]
        if validation["status"] != "passed":
            status.update(status="validation_failed", exit_code=2)
            common.write_json(status_path, status)
            return 2
        shadow_validation = _validate_smoke(run_root, smoke=False)
        common.write_json(run_root / "policy_shadow_validation.json", shadow_validation)
        status["policy_shadow_validation"] = shadow_validation["status"]
        if shadow_validation["status"] != "passed":
            status.update(status="validation_failed", exit_code=2)
            common.write_json(status_path, status)
            return 2

    try:
        from paired_stage_comparison import compare_stages

        dirs = _run_dirs(run_root)
        pairing = compare_stages(
            dirs[METHODS[0]] / "evaluation_results.json",
            dirs[METHODS[1]] / "evaluation_results.json",
            run_root / "paired_method_comparison.json",
            label="goal-only-routeaware-vs-nonlinear-3slot",
            expected_episodes=eval_count,
        )
        status["pairing_status"] = pairing.get("status", "written")
    except Exception as error:  # preserve training/evaluation outputs if comparison fails.
        status["pairing_status"] = "failed"
        status["pairing_error"] = repr(error)
    status.update(status="complete", exit_code=0, finished_at_utc=_utc_now())
    common.write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--start", action="store_true", help="archive sources and launch two workers")
    parser.add_argument("--smoke", action="store_true", help="300-raw-step engineering check, not a result")
    args = parser.parse_args(argv)
    _configure_common()
    default_root = DEFAULT_SMOKE_ROOT if args.smoke else DEFAULT_RUN_ROOT
    run_root = (args.run_root or default_root).expanduser().resolve()
    runs_root = (WORKSPACE / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        parser.error(f"--run-root must be a child of {runs_root}")
    if run_root.exists():
        parser.error(f"refusing existing run root: {run_root}")
    plan = common.validate_runtime_plan(run_root)
    command = _worker_command(run_root, smoke=args.smoke)
    if not args.start:
        compact = dict(plan)
        if args.smoke:
            compact["raw_steps_per_method"] = SMOKE_RAW_STEPS
            compact["learning_starts_raw_steps"] = SMOKE_WARMUP_RAW
            compact["checkpoint_frequency_raw_steps"] = SMOKE_CHECKPOINT_RAW
            compact["evaluation"] = dict(plan["evaluation"])
            compact["evaluation"].update(
                episodes=SMOKE_EVAL_EPISODES,
                seed_end_inclusive=EVAL_SEED_START + SMOKE_EVAL_EPISODES - 1,
            )
        compact["traffic"] = {
            key: plan["traffic"][key]
            for key in (
                "traffic_template_count", "background_vehicle_counts_per_template",
                "ego_task", "sumo_config_collision_action", "terminal_outcome_protocol",
                "traffic_schedule_protocol", "train_eval_pool",
            )
        }
        print(json.dumps({
            "mode": "preview_only",
            "training_started": False,
            "run_root_exists": run_root.exists(),
            "smoke": args.smoke,
            "command": command,
            "plan": compact,
            "note": "Preview creates no run directory, SUMO environment, or training process.",
        }, ensure_ascii=False, indent=2))
        return 0
    return launch(run_root, plan, smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
