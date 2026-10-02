"""Successive-halving development runner for topology-temporal v2.

The fast protocol is deliberately separate from the original frozen study.  It
reuses the original command construction and strict per-run acceptance audit,
but treats 20k runs as screening evidence and 50k runs as promotion evidence.
Only a hash-bound passing promotion receipt unlocks the full formal test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_topo_v2_experiments as full
from tools.topo_v2_statistics import paired_hierarchical_bootstrap


DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "experiments" / "topo_scene_v2" / "fast_iteration_contract.yaml"
)
SCHEMA = "topo-scene-v2.fast-iteration-contract/v1"
STAGE_SCHEMA = "topology-temporal-v2-fast-stage-results/v1"
DEVELOPMENT_STAGES = (
    "factorial_anchor",
    "merge_probe",
    "query_probe",
    "gated_probe",
    "soft_center",
    "soft_bracket",
    "promotion",
)
PRIMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "mean_return",
    "off_route_rate",
    "timeout_rate",
)
ACTION_METRICS = (
    "lane_command_negative_rate",
    "lane_command_keep_rate",
    "lane_command_positive_rate",
    "lane_command_threshold_margin",
    "lane_change_applied_rate",
    "mean_speed_mps",
)
LEGACY_MECHANISM_DIAGNOSTICS = (
    "diagnostic/ego_latent_std",
    "diagnostic/social_latent_std",
    "diagnostic/route_latent_std",
    "diagnostic/slot_scale_ratio",
    "diagnostic/graph_mean_edge_weight",
)


class FastProtocolError(ValueError):
    """Raised when a fast-v2 invariant or stage gate is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FastProtocolError(message)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _protocol_output_root(protocol: dict[str, Any]) -> Path:
    return _resolve(str(protocol["output_root"]))


def _stage_root(protocol: dict[str, Any], stage: str) -> Path:
    return _protocol_output_root(protocol) / "stages" / stage


def _receipt_path(protocol: dict[str, Any], name: str) -> Path:
    return _resolve(str(protocol[name]))


def load_protocol(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
    path = path.resolve()
    _require(path.is_file(), f"Missing fast protocol: {path}")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(protocol, dict), "Fast protocol must be a YAML mapping")
    _require(protocol.get("schema_version") == SCHEMA, "Unexpected fast schema")
    _require(protocol.get("mode") == "standard", "CCFA mode must be standard")
    _require(protocol.get("no_fabrication") is True, "no_fabrication must be true")

    base_path = _resolve(str(protocol["base_contract"]))
    _require(base_path.is_file(), f"Missing base contract: {base_path}")
    observed_base_sha = _sha256(base_path)
    _require(
        observed_base_sha == str(protocol["base_contract_sha256"]),
        "Base experiment contract drifted",
    )
    base_contract = full.validate_contract(full.load_contract(base_path))
    effective = copy.deepcopy(base_contract)
    effective["output_root"] = str(protocol["output_root"])
    effective["methods"]["selected_candidate"]["resolver"] = str(
        protocol["candidate_receipt"]
    )
    effective["lambda_selection"]["output"] = str(protocol["candidate_receipt"])
    effective["phases"]["formal_test"]["gate_receipt"] = str(
        protocol["promotion_gate_receipt"]
    )

    order = protocol.get("stage_order")
    stages = protocol.get("stages")
    _require(isinstance(order, list) and order, "stage_order must be non-empty")
    _require(isinstance(stages, dict), "stages must be a mapping")
    _require(order == list(stages), "stages must follow stage_order exactly")
    _require(order[-1] == "formal_test", "formal_test must be the final stage")
    controls = protocol.get("fixed_controls", {})
    _require(controls.get("python_environment") == "llm_pipeline", "Wrong Python env")
    _require(
        controls.get("traffic_protocol") == "frozen_60_20_20",
        "Fast runs must retain frozen traffic partitioning",
    )
    _require(controls.get("development_partition") == "validation", "Bad dev split")
    _require(controls.get("formal_partition") == "test", "Bad formal split")

    seen: set[tuple[str, str, str, int]] = set()
    for stage_name in order:
        stage = stages[stage_name]
        _require(isinstance(stage, dict), f"Stage {stage_name} must be a mapping")
        phase = stage.get("phase")
        _require(phase in full.PHASE_ORDER, f"Unsupported phase for {stage_name}")
        for key in ("methods", "scenarios", "seeds"):
            _require(isinstance(stage.get(key), list) and stage[key], f"{stage_name}.{key}")
        _require(
            not (set(stage["methods"]) - set(effective["methods"])),
            f"Unknown method in {stage_name}",
        )
        _require(
            not (set(stage["scenarios"]) - set(full.SCENARIOS)),
            f"Unknown scenario in {stage_name}",
        )
        _require(
            all(isinstance(seed, int) and seed >= 0 for seed in stage["seeds"]),
            f"Invalid seed in {stage_name}",
        )
        raw_steps = int(stage.get("raw_training_steps", -1))
        episodes = int(stage.get("final_evaluation_episodes", -1))
        _require(raw_steps > int(controls["learning_starts_raw_steps"]), f"{stage_name} too short")
        _require(episodes > 0, f"{stage_name} needs evaluation episodes")
        expected_split = "test" if stage_name == "formal_test" else "validation"
        _require(stage.get("evaluation_split") == expected_split, f"Bad split in {stage_name}")
        for method in stage["methods"]:
            for scenario in stage["scenarios"]:
                for seed in stage["seeds"]:
                    key = (str(phase), str(method), str(scenario), int(seed))
                    _require(key not in seen, f"Duplicate run identity across stages: {key}")
                    seen.add(key)

    factorial = stages["factorial_anchor"]
    cells = {
        (
            bool(effective["methods"][name].get("topology")),
            bool(effective["methods"][name].get("hard_slot_normalization")),
        )
        for name in factorial["methods"]
    }
    _require(
        cells == {(False, False), (True, False), (False, True), (True, True)},
        "factorial_anchor is not a complete 2x2",
    )
    formal = stages["formal_test"]
    _require(formal["methods"] == ["temporal_graph", "selected_candidate"], "Bad formal methods")
    _require(tuple(formal["scenarios"]) == full.SCENARIOS, "Formal must cover six scenarios")
    _require(formal["seeds"] == list(range(10)), "Formal must use seeds 0..9")
    _require(int(formal["raw_training_steps"]) == 100_000, "Formal must use 100k")
    _require(int(formal["final_evaluation_episodes"]) == 50, "Formal must use 50 episodes")
    full.frozen_dependency_hashes(effective)
    return protocol, effective, _sha256(path), base_path


def _selected_metadata(effective: dict[str, Any], digest: str) -> dict[str, Any]:
    return full._selected_candidate(effective, digest)


def jobs_for_stage(
    protocol: dict[str, Any], effective: dict[str, Any], digest: str, stage_name: str
) -> list[full.Job]:
    _require(stage_name in protocol["stages"], f"Unknown stage {stage_name!r}")
    stage = protocol["stages"][stage_name]
    controls = protocol["fixed_controls"]
    jobs: list[full.Job] = []
    selected: dict[str, Any] | None = None
    if "selected_candidate" in stage["methods"]:
        selected = _selected_metadata(effective, digest)
    for method_name in stage["methods"]:
        if method_name == "selected_candidate":
            assert selected is not None
            algorithm = str(selected["cli_algorithm"])
            implementation_id = str(selected["implementation_id"])
            coefficient = float(selected["soft_slot_balance_coef"])
        else:
            method = effective["methods"][method_name]
            algorithm = str(method["cli_algorithm"])
            implementation_id = str(method["implementation_id"])
            coefficient = float(method.get("soft_slot_balance_coef", 0.0))
        for scenario in stage["scenarios"]:
            for seed in stage["seeds"]:
                jobs.append(
                    full.Job(
                        phase=str(stage["phase"]),
                        method=str(method_name),
                        algorithm=algorithm,
                        implementation_id=implementation_id,
                        scenario=str(scenario),
                        seed=int(seed),
                        raw_steps=int(stage["raw_training_steps"]),
                        learning_starts=int(controls["learning_starts_raw_steps"]),
                        checkpoint_frequency=int(
                            controls["checkpoint_frequency_raw_steps"]
                        ),
                        evaluation_frequency=int(
                            controls["evaluation_frequency_raw_steps"]
                        ),
                        evaluation_episodes=int(stage["final_evaluation_episodes"]),
                        evaluation_split=str(stage["evaluation_split"]),
                        traffic_protocol=str(controls["traffic_protocol"]),
                        episode_limit_profile=str(controls["episode_limit_profile"]),
                        slot_balance_coef=coefficient,
                        check_only=False,
                        protocol_tag=digest[:8],
                    )
                )
    names = [job.name for job in jobs]
    _require(len(names) == len(set(names)), f"Duplicate job name in {stage_name}")
    return jobs


def _job_record(
    effective: dict[str, Any], job: full.Job, device: str, digest: str
) -> dict[str, Any]:
    accepted, reason = full.accepted_run_reason(effective, job, digest)
    return {
        **asdict(job),
        "name": job.name,
        "run_directory": str(full.run_directory(effective, job)),
        "accepted": accepted,
        "acceptance_reason": reason,
        "command": full.command_for(
            effective, job, device=device, contract_digest=digest
        ),
    }


def write_stage_plan(
    protocol: dict[str, Any],
    effective: dict[str, Any],
    protocol_path: Path,
    base_path: Path,
    digest: str,
    stage_name: str,
    jobs: list[full.Job],
    device: str,
) -> dict[str, Any]:
    root = _stage_root(protocol, stage_name)
    root.mkdir(parents=True, exist_ok=True)
    snapshots = (
        (root / "fast_protocol.snapshot.yaml", protocol_path.read_bytes()),
        (root / "base_contract.snapshot.yaml", base_path.read_bytes()),
    )
    for target, content in snapshots:
        if target.exists():
            _require(target.read_bytes() == content, f"Protocol snapshot drift: {target}")
        else:
            target.write_bytes(content)
    payload = {
        "schema_version": "topology-temporal-v2-fast-stage-plan/v1",
        "fast_protocol": str(protocol_path),
        "fast_protocol_sha256": digest,
        "base_contract": str(base_path),
        "base_contract_sha256": str(protocol["base_contract_sha256"]),
        "stage": stage_name,
        "evidence_role": protocol["stages"][stage_name]["evidence_role"],
        "device": device,
        "jobs": [_job_record(effective, job, device, digest) for job in jobs],
    }
    _write_json(root / "stage_plan.json", payload)
    return payload


def _summary_path(protocol: dict[str, Any], stage: str) -> Path:
    return _stage_root(protocol, stage) / "summary.json"


def _decision_path(protocol: dict[str, Any], stage: str) -> Path:
    return _stage_root(protocol, stage) / "decision.json"


def _completed_summary(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    path = _summary_path(protocol, stage)
    _require(path.is_file(), f"Stage {stage!r} has no summary")
    payload = _load_json(path)
    _require(payload.get("complete") is True, f"Stage {stage!r} is incomplete")
    return payload


def _completed_decision(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    path = _decision_path(protocol, stage)
    _require(path.is_file(), f"Stage {stage!r} has no decision")
    payload = _load_json(path)
    _require(payload.get("status") == "complete", f"Stage {stage!r} decision incomplete")
    return payload


def assert_stage_can_run(
    protocol: dict[str, Any], effective: dict[str, Any], digest: str, stage: str
) -> None:
    prerequisites = {
        "merge_probe": "factorial_anchor",
        "query_probe": "merge_probe",
        "gated_probe": "query_probe",
        "soft_center": "gated_probe",
        "soft_bracket": "soft_center",
    }
    if stage in prerequisites:
        previous = prerequisites[stage]
        _completed_summary(protocol, previous)
        decision = _completed_decision(protocol, previous)
        if stage == "soft_center":
            _require(decision.get("continue_to_soft") is True, "V4 safety gate stopped V5")
        if stage == "soft_bracket":
            _require(decision.get("run_bracket") is True, "Soft centre did not justify bracketing")
    if stage == "promotion":
        _selected_metadata(effective, digest)
    if stage == "formal_test":
        gate_path = _receipt_path(protocol, "promotion_gate_receipt")
        _require(gate_path.is_file(), "Formal test locked: missing promotion gate")
        gate = _load_json(gate_path)
        _require(gate.get("decision") == "pass", "Formal test locked: gate did not pass")
        _require(
            gate.get("fast_protocol_sha256") == digest,
            "Formal test locked: gate belongs to another protocol",
        )


def execute_stage(
    protocol: dict[str, Any],
    effective: dict[str, Any],
    protocol_path: Path,
    base_path: Path,
    digest: str,
    stage: str,
    *,
    device: str,
    workers: int,
    dry_run: bool = False,
    max_jobs: int | None = None,
    continue_on_error: bool = False,
) -> int:
    jobs = jobs_for_stage(protocol, effective, digest, stage)
    if not dry_run:
        assert_stage_can_run(protocol, effective, digest, stage)
    plan = write_stage_plan(
        protocol,
        effective,
        protocol_path,
        base_path,
        digest,
        stage,
        jobs,
        device,
    )
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    _require(workers > 0, "--workers must be positive")
    pending = [job for job in jobs if not full.accepted_run(effective, job, digest)]
    if max_jobs is not None:
        _require(max_jobs > 0, "--max-jobs must be positive")
        pending = pending[:max_jobs]
    for job in pending:
        directory = full.run_directory(effective, job)
        if directory.exists():
            accepted, reason = full.accepted_run_reason(effective, job, digest)
            if not accepted:
                raise FileExistsError(
                    f"Refusing to overwrite incomplete fast run {directory}: {reason}"
                )

    log_root = _stage_root(protocol, stage) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []

    def run_one(job: full.Job) -> dict[str, Any] | None:
        log_path = log_root / f"{job.name}.log"
        command = full.command_for(
            effective, job, device=device, contract_digest=digest
        )
        print(
            json.dumps(
                {"stage": stage, "job": job.name, "status": "running", "log": str(log_path)},
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
        accepted, reason = full.accepted_run_reason(effective, job, digest)
        if completed.returncode != 0 or not accepted:
            failure = {
                "stage": stage,
                "job": job.name,
                "returncode": completed.returncode,
                "acceptance_reason": reason,
                "log": str(log_path),
            }
            print(json.dumps({**failure, "status": "failed"}, ensure_ascii=False), flush=True)
            return failure
        print(
            json.dumps({"stage": stage, "job": job.name, "status": "accepted"}, ensure_ascii=False),
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, job): job for job in pending}
            for future in as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
    _write_json(
        _stage_root(protocol, stage) / "last_execution.json",
        {
            "schema_version": "topology-temporal-v2-fast-execution/v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fast_protocol_sha256": digest,
            "stage": stage,
            "selected_jobs": len(jobs),
            "attempted_jobs": len(pending),
            "workers": workers,
            "failures": failures,
        },
    )
    return 1 if failures else 0


def _stage_rows(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return list(_completed_summary(protocol, stage).get("per_run", []))


def _enrich_legacy_mechanism(
    row: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    """Keep V1/hard-path diagnostics visible beside the V2 slot-RMS fields."""

    enriched = {**row, "mechanism": dict(row.get("mechanism", {}))}
    for source_name in LEGACY_MECHANISM_DIAGNOSTICS:
        output_name = source_name.removeprefix("diagnostic/")
        enriched["mechanism"][output_name] = full._stat_mean(
            diagnostics, source_name
        )
    return enriched


def _fast_run_row(contract: dict[str, Any], job: full.Job) -> dict[str, Any]:
    row = full._run_row(contract, job)
    directory = full.run_directory(contract, job)
    diagnostics = _load_json(directory / "training_diagnostics.json")["statistics"]
    return _enrich_legacy_mechanism(row, diagnostics)


def _context_rows(
    protocol: dict[str, Any], stage: str, own_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    dependencies = {
        "merge_probe": ("factorial_anchor",),
        "query_probe": ("merge_probe",),
        "gated_probe": ("query_probe",),
        "soft_center": ("gated_probe",),
        "soft_bracket": ("gated_probe", "soft_center"),
    }
    rows = list(own_rows)
    for dependency in dependencies.get(stage, ()):
        rows.extend(_stage_rows(protocol, dependency))
    return rows


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(finite)) if finite else None


def paired_contrast(
    rows: list[dict[str, Any]], candidate_method: str, baseline_method: str
) -> dict[str, Any]:
    by_key = {(row["scenario"], row["seed"], row["method"]): row for row in rows}
    keys = sorted(
        {
            (row["scenario"], int(row["seed"]))
            for row in rows
            if row["method"] == candidate_method
        }
    )
    pairs: list[dict[str, Any]] = []
    for scenario, seed in keys:
        candidate = by_key.get((scenario, seed, candidate_method))
        baseline = by_key.get((scenario, seed, baseline_method))
        if candidate is None or baseline is None:
            continue
        metrics: dict[str, Any] = {}
        for name in (*PRIMARY_METRICS, *ACTION_METRICS):
            if candidate.get(name) is None or baseline.get(name) is None:
                continue
            metrics[name] = {
                "candidate": float(candidate[name]),
                "baseline": float(baseline[name]),
                "delta": float(candidate[name] - baseline[name]),
            }
        mechanism: dict[str, Any] = {}
        candidate_mechanism = {
            **{f"train/{k}": v for k, v in candidate["mechanism"].items()},
            **{f"evaluation/{k}": v for k, v in candidate["evaluation_mechanism"].items()},
        }
        baseline_mechanism = {
            **{f"train/{k}": v for k, v in baseline["mechanism"].items()},
            **{f"evaluation/{k}": v for k, v in baseline["evaluation_mechanism"].items()},
        }
        for name in sorted(set(candidate_mechanism) | set(baseline_mechanism)):
            left = candidate_mechanism.get(name)
            right = baseline_mechanism.get(name)
            mechanism[name] = {
                "candidate": left,
                "baseline": right,
                "delta": (
                    float(left - right) if left is not None and right is not None else None
                ),
            }
        pairs.append(
            {
                "scenario": scenario,
                "seed": seed,
                "metrics": metrics,
                "mechanism": mechanism,
            }
        )
    macro_delta = {
        name: _mean(
            pair["metrics"][name]["delta"]
            for pair in pairs
            if name in pair["metrics"]
        )
        for name in (*PRIMARY_METRICS, *ACTION_METRICS)
    }
    return {
        "candidate": candidate_method,
        "baseline": baseline_method,
        "paired_cells": len(pairs),
        "pairs": pairs,
        "macro_delta": macro_delta,
    }


def _factorial_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell = {
        (False, False): "temporal_graph",
        (True, False): "topology_v1",
        (False, True): "temporal_hard",
        (True, True): "full_hard",
    }
    by_key = {(row["scenario"], row["seed"], row["method"]): row for row in rows}
    effects: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if row["scenario"] == scenario}):
            for metric in PRIMARY_METRICS:
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
                        "scenario": scenario,
                        "seed": seed,
                        "metric": metric,
                        "topology_main_effect": ((f10 - f00) + (f11 - f01)) / 2.0,
                        "hard_norm_main_effect": ((f01 - f00) + (f11 - f10)) / 2.0,
                        "interaction": f11 - f10 - f01 + f00,
                    }
                )
    return {
        "design": "topology_x_hard_normalisation_2x2",
        "screening_only": True,
        "cells": {f"topology={k[0]},hard={k[1]}": v for k, v in cell.items()},
        "effects": effects,
    }


def attribution_for_stage(
    protocol: dict[str, Any], stage: str, own_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    context = _context_rows(protocol, stage, own_rows)
    if stage == "factorial_anchor":
        analysis: Any = _factorial_attribution(context)
    else:
        pair = {
            "merge_probe": ("v2_merge", "topology_v1"),
            "query_probe": ("v3_query", "v2_merge"),
            "gated_probe": ("v4_gated", "v3_query"),
            "soft_center": ("v5_soft_1e3", "v4_gated"),
            "promotion": ("selected_candidate", "temporal_graph"),
            "formal_test": ("selected_candidate", "temporal_graph"),
        }
        if stage == "soft_bracket":
            analysis = [
                paired_contrast(context, method, "v4_gated")
                for method in ("v5_soft_1e4", "v5_soft_1e2")
            ]
        elif stage in pair:
            analysis = paired_contrast(context, *pair[stage])
        else:
            analysis = None
    return {
        "schema_version": "topology-temporal-v2-fast-attribution/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "stage": stage,
        "evidence_role": protocol["stages"][stage]["evidence_role"],
        "finite": _finite(context),
        "analysis": analysis,
        "interpretation_rule": "Screening and promotion deltas are descriptive; only formal-test inference supports final claims.",
    }


def summarize_stage(
    protocol: dict[str, Any],
    effective: dict[str, Any],
    digest: str,
    stage: str,
) -> int:
    jobs = jobs_for_stage(protocol, effective, digest, stage)
    accepted = [job for job in jobs if full.accepted_run(effective, job, digest)]
    missing = [job.name for job in jobs if job not in accepted]
    rows = [_fast_run_row(effective, job) for job in accepted]
    attribution = attribution_for_stage(protocol, stage, rows)
    payload = {
        "schema_version": STAGE_SCHEMA,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "fast_protocol_sha256": digest,
        "stage": stage,
        "evidence_role": protocol["stages"][stage]["evidence_role"],
        "expected_jobs": len(jobs),
        "accepted_jobs": len(accepted),
        "missing_jobs": missing,
        "complete": not missing,
        "per_run": rows,
        "aggregates": full.aggregate_rows(rows),
        "attribution": attribution,
    }
    root = _stage_root(protocol, stage)
    _write_json(root / "summary.json", payload)
    _write_json(root / "attribution.json", attribution)
    fields = (
        "phase",
        "method",
        "scenario",
        "seed",
        "raw_steps",
        "evaluation_split",
        "soft_slot_balance_coef",
        *PRIMARY_METRICS,
        *ACTION_METRICS,
        "parameter_count",
        "train_ms_per_gradient_step",
        "training_wall_seconds",
        "inference_ms_per_action",
        "peak_gpu_memory_mb",
        "run_directory",
    )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    if stage == "formal_test" and not missing:
        inference = paired_hierarchical_bootstrap(
            rows,
            candidate_method="selected_candidate",
            baseline_method="temporal_graph",
            scenarios=full.SCENARIOS,
            resamples=int(protocol["statistics"]["formal_hierarchical_bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
        _write_json(root / "formal_inference.json", inference)
        _write_json(_protocol_output_root(protocol) / "formal_summary.json", payload)
        _write_json(_protocol_output_root(protocol) / "formal_inference.json", inference)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


def _contrast_for(protocol: dict[str, Any], stage: str, candidate: str, baseline: str) -> dict[str, Any]:
    rows = _context_rows(protocol, stage, _stage_rows(protocol, stage))
    return paired_contrast(rows, candidate, baseline)


def _macro_delta(contrast: dict[str, Any], metric: str) -> float:
    value = contrast["macro_delta"].get(metric)
    _require(value is not None and math.isfinite(float(value)), f"Missing finite {metric} delta")
    return float(value)


def _collision_deltas(contrast: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for pair in contrast["pairs"]:
        metric = pair["metrics"].get("collision_rate")
        if metric is not None:
            values.setdefault(pair["scenario"], []).append(float(metric["delta"]))
    return {scenario: float(statistics.mean(rows)) for scenario, rows in values.items()}


def _mechanism_gate(rows: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
    names = {
        "route_compatible_attention_mass": ("minimum", "compatible_attention_mass_min"),
        "topology_fallback_rate": ("maximum", "topology_fallback_rate_max"),
        "topology_effective_lanes": ("maximum", "topology_effective_lanes_max"),
    }
    output: dict[str, Any] = {}
    for name, (direction, threshold_name) in names.items():
        values = [
            row["evaluation_mechanism"].get(name)
            for row in rows
            if row["evaluation_mechanism"].get(name) is not None
        ]
        finite = len(values) == len(rows) and all(math.isfinite(float(v)) for v in values)
        threshold = float(rules[threshold_name])
        observed = None
        passed = False
        if finite:
            observed = min(values) if direction == "minimum" else max(values)
            passed = observed >= threshold if direction == "minimum" else observed <= threshold
        output[name] = {
            "direction": direction,
            "observed": observed,
            "threshold": threshold,
            "finite": finite,
            "passed": passed,
        }
    output["passed"] = all(item["passed"] for key, item in output.items() if key != "passed")
    return output


def _balance_score(rows: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for row in rows:
        value = row["mechanism"].get("slot_rms_ratio")
        if value is None:
            value = row["evaluation_mechanism"].get("slot_rms_ratio")
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            continue
        scores.append(abs(math.log(float(value))))
    return float(statistics.mean(scores)) if scores else None


def decide_stage(
    protocol: dict[str, Any], digest: str, stage: str
) -> dict[str, Any]:
    _completed_summary(protocol, stage)
    rules = protocol["decision_rules"]
    payload: dict[str, Any] = {
        "schema_version": "topology-temporal-v2-fast-decision/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "fast_protocol_sha256": digest,
        "stage": stage,
        "status": "complete",
        "evidence_role": protocol["stages"][stage]["evidence_role"],
    }
    if stage == "factorial_anchor":
        no_topology = _contrast_for(protocol, stage, "temporal_hard", "temporal_graph")
        topology = _contrast_for(protocol, stage, "full_hard", "topology_v1")
        success = _mean(
            [_macro_delta(no_topology, "success_rate"), _macro_delta(topology, "success_rate")]
        )
        collision = _mean(
            [_macro_delta(no_topology, "collision_rate"), _macro_delta(topology, "collision_rate")]
        )
        assert success is not None and collision is not None
        rejected = (
            success <= -float(rules[stage]["gross_success_drop"])
            or collision >= float(rules[stage]["gross_collision_increase"])
        )
        payload.update(
            {
                "hard_normalisation_decision": "reject" if rejected else "not_rejected_by_screen",
                "hard_normalisation_macro_success_delta": success,
                "hard_normalisation_macro_collision_delta": collision,
                "contrasts": [no_topology, topology],
                "continue_to_merge": True,
            }
        )
    elif stage == "merge_probe":
        contrast = _contrast_for(protocol, stage, "v2_merge", "topology_v1")
        payload.update(
            {
                "contrast": contrast,
                "matched_anchor_count": contrast["paired_cells"],
                "merge_claim_status": (
                    "screen_positive"
                    if contrast["paired_cells"]
                    and (
                        _macro_delta(contrast, "success_rate")
                        >= float(rules["episode_quantum_screen"])
                        or _macro_delta(contrast, "collision_rate")
                        <= -float(rules["episode_quantum_screen"])
                    )
                    else "not_supported_by_short_screen"
                ),
                "continue_to_query": True,
            }
        )
    elif stage == "query_probe":
        contrast = _contrast_for(protocol, stage, "v3_query", "v2_merge")
        candidate_rows = _stage_rows(protocol, stage)
        mechanism = _mechanism_gate(candidate_rows, rules[stage])
        payload.update(
            {
                "contrast": contrast,
                "mechanism": mechanism,
                "route_localisation_status": "screen_positive" if mechanism["passed"] else "screen_negative",
                "continue_to_gated": True,
            }
        )
    elif stage == "gated_probe":
        contrast = _contrast_for(protocol, stage, "v4_gated", "v3_query")
        collision = _collision_deltas(contrast)
        mechanism = _mechanism_gate(_stage_rows(protocol, stage), rules[stage])
        safe = (
            _macro_delta(contrast, "success_rate")
            >= -float(rules[stage]["macro_success_drop_max"])
            and bool(collision)
            and all(
                value <= float(rules[stage]["per_scenario_collision_increase_max"])
                for value in collision.values()
            )
            and mechanism["passed"]
            and _finite(contrast)
        )
        payload.update(
            {
                "contrast": contrast,
                "collision_delta_by_scenario": collision,
                "mechanism": mechanism,
                "v4_safe": safe,
                "continue_to_soft": safe,
                "stop_reason": None if safe else "V4 failed the predeclared short-horizon safety/mechanism gate",
            }
        )
    elif stage == "soft_center":
        contrast = _contrast_for(protocol, stage, "v5_soft_1e3", "v4_gated")
        collision = _collision_deltas(contrast)
        v4_rows = [
            row
            for row in _stage_rows(protocol, "gated_probe")
            if row["scenario"] in protocol["stages"][stage]["scenarios"]
        ]
        candidate_rows = _stage_rows(protocol, stage)
        v4_balance = _balance_score(v4_rows)
        candidate_balance = _balance_score(candidate_rows)
        improvement = (
            v4_balance - candidate_balance
            if v4_balance is not None and candidate_balance is not None
            else None
        )
        safe = (
            _macro_delta(contrast, "success_rate")
            >= -float(rules[stage]["macro_success_drop_max"])
            and bool(collision)
            and all(
                value <= float(rules[stage]["per_scenario_collision_increase_max"])
                for value in collision.values()
            )
            and _finite(contrast)
        )
        run_bracket = bool(
            safe
            and improvement is not None
            and improvement >= float(rules[stage]["minimum_balance_score_improvement"])
        )
        payload.update(
            {
                "contrast": contrast,
                "collision_delta_by_scenario": collision,
                "safe": safe,
                "v4_balance_score": v4_balance,
                "candidate_balance_score": candidate_balance,
                "balance_score_improvement": improvement,
                "run_bracket": run_bracket,
                "fallback_candidate": "v4_gated" if not run_bracket else None,
            }
        )
    elif stage == "soft_bracket":
        payload.update(
            {
                "selection_ready": True,
                "contrasts": [
                    _contrast_for(protocol, stage, method, "v4_gated")
                    for method in ("v5_soft_1e4", "v5_soft_1e2")
                ],
            }
        )
    else:
        raise FastProtocolError(f"Stage {stage!r} has no standalone decision")
    _write_json(_decision_path(protocol, stage), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def select_candidate(
    protocol: dict[str, Any], effective: dict[str, Any], digest: str
) -> dict[str, Any]:
    gated = _completed_decision(protocol, "gated_probe")
    _require(gated.get("v4_safe") is True, "Cannot select candidate: V4 failed")
    centre = _completed_decision(protocol, "soft_center")
    v4_rows = [
        row
        for row in _stage_rows(protocol, "gated_probe")
        if row["scenario"] in protocol["stages"]["soft_center"]["scenarios"]
    ]
    rows_by_method: dict[str, list[dict[str, Any]]] = {"v4_gated": v4_rows}
    rows_by_method["v5_soft_1e3"] = _stage_rows(protocol, "soft_center")
    if centre.get("run_bracket") is True:
        _completed_decision(protocol, "soft_bracket")
        for row in _stage_rows(protocol, "soft_bracket"):
            rows_by_method.setdefault(row["method"], []).append(row)

    guardrail = float(protocol["decision_rules"]["candidate_selection"]["v4_collision_guardrail"])
    scenarios = list(protocol["stages"]["soft_center"]["scenarios"])
    v4_collision = {
        scenario: statistics.mean(
            row["collision_rate"] for row in v4_rows if row["scenario"] == scenario
        )
        for scenario in scenarios
    }
    evaluations: list[dict[str, Any]] = []
    for method_name, rows in sorted(rows_by_method.items()):
        _require(len(rows) == len(scenarios), f"Incomplete candidate screen for {method_name}")
        scenario_collision = {
            scenario: statistics.mean(
                row["collision_rate"] for row in rows if row["scenario"] == scenario
            )
            for scenario in scenarios
        }
        safe = all(
            scenario_collision[scenario] <= v4_collision[scenario] + guardrail
            for scenario in scenarios
        )
        evaluations.append(
            {
                "method": method_name,
                "safe": safe,
                "macro_success_rate": float(statistics.mean(row["success_rate"] for row in rows)),
                "scenario_collision_rate": scenario_collision,
                "balance_score": _balance_score(rows),
                "soft_slot_balance_coef": float(rows[0]["soft_slot_balance_coef"]),
            }
        )

    if centre.get("run_bracket") is not True:
        selected_name = "v4_gated"
        selection_reason = "lambda=1e-3 did not justify the predeclared bracket; retain V4"
    else:
        safe_rows = [row for row in evaluations if row["safe"]]
        _require(safe_rows, "No safe V4/V5 candidate")
        best_success = max(row["macro_success_rate"] for row in safe_rows)
        tie = float(protocol["decision_rules"]["candidate_selection"]["success_tie"])
        tied = [row for row in safe_rows if best_success - row["macro_success_rate"] <= tie]
        with_balance = [row for row in tied if row["balance_score"] is not None]
        if with_balance:
            best_balance = min(row["balance_score"] for row in with_balance)
            tied = [
                row
                for row in with_balance
                if math.isclose(row["balance_score"], best_balance, rel_tol=0.0, abs_tol=1e-12)
            ]
        selected_name = min(tied, key=lambda row: row["soft_slot_balance_coef"])["method"]
        selection_reason = "predeclared safety-success-balance-coefficient ordering"

    metadata = effective["methods"][selected_name]
    payload = {
        "schema_version": "topology-temporal-v2-fast-candidate-selection/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "status": "frozen",
        "fast_protocol_sha256": digest,
        "experiment_contract_sha256": digest,
        "selected_method": selected_name,
        "cli_algorithm": str(metadata["cli_algorithm"]),
        "implementation_id": str(metadata["implementation_id"]),
        "soft_slot_balance_coef": float(metadata.get("soft_slot_balance_coef", 0.0)),
        "evaluations": evaluations,
        "selection_reason": selection_reason,
        "selected_from_validation_only": True,
        "formal_test_accessed": False,
    }
    path = _receipt_path(protocol, "candidate_receipt")
    if path.exists():
        _require(_load_json(path) == payload, "Refusing to change frozen candidate")
    else:
        _write_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def compute_promotion_gate(
    protocol: dict[str, Any], digest: str
) -> dict[str, Any]:
    rows = _stage_rows(protocol, "promotion")
    contrast = paired_contrast(rows, "selected_candidate", "temporal_graph")
    expected_pairs = len(protocol["stages"]["promotion"]["scenarios"]) * len(
        protocol["stages"]["promotion"]["seeds"]
    )
    _require(contrast["paired_cells"] == expected_pairs, "Promotion lacks paired evidence")
    rules = protocol["decision_rules"]["promotion_gate"]
    collision = _collision_deltas(contrast)
    success_deltas = [
        float(pair["metrics"]["success_rate"]["delta"]) for pair in contrast["pairs"]
    ]
    macro_success = float(statistics.mean(success_deltas))
    worst_success = min(success_deltas)
    candidate_rows = [row for row in rows if row["method"] == "selected_candidate"]
    mechanism = _mechanism_gate(candidate_rows, rules)
    scenario_effects = {
        scenario: {
            "success_delta": float(
                statistics.mean(
                    pair["metrics"]["success_rate"]["delta"]
                    for pair in contrast["pairs"]
                    if pair["scenario"] == scenario
                )
            ),
            "collision_delta": float(collision[scenario]),
        }
        for scenario in rules["positive_effect_any"]["scenarios"]
    }
    positive = any(
        values["success_delta"]
        >= float(rules["positive_effect_any"]["minimum_success_gain"])
        or -values["collision_delta"]
        >= float(rules["positive_effect_any"]["minimum_collision_reduction"])
        for values in scenario_effects.values()
    )
    checks = {
        "finite": _finite(contrast) and _finite(mechanism),
        "macro_success_noninferiority": macro_success
        >= -float(rules["macro_success_noninferiority_margin"]),
        "per_scenario_collision_noninferiority": all(
            value <= float(rules["per_scenario_collision_noninferiority_margin"])
            for value in collision.values()
        ),
        "worst_paired_seed_success": worst_success
        >= -float(rules["worst_paired_seed_success_margin"]),
        "mechanism": mechanism["passed"],
        "positive_effect_any": positive,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "topology-temporal-v2-fast-promotion-gate/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "fast_protocol_sha256": digest,
        "decision": "pass" if passed else "fail",
        "formal_test_unlocked": passed,
        "checks": checks,
        "observed": {
            "macro_success_delta": macro_success,
            "collision_delta_by_scenario": collision,
            "worst_paired_success_delta": worst_success,
            "mechanism": mechanism,
            "positive_scenario_deltas": scenario_effects,
        },
        "thresholds": rules,
        "paired_evidence": contrast,
    }
    _write_json(_receipt_path(protocol, "promotion_gate_receipt"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def protocol_status(
    protocol: dict[str, Any], effective: dict[str, Any], digest: str
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for name in protocol["stage_order"]:
        declaration = protocol["stages"][name]
        declared_jobs = (
            len(declaration["methods"])
            * len(declaration["scenarios"])
            * len(declaration["seeds"])
        )
        try:
            jobs = jobs_for_stage(protocol, effective, digest, name)
        except (full.ExperimentContractError, FastProtocolError):
            jobs = []
        accepted = sum(full.accepted_run(effective, job, digest) for job in jobs)
        stages.append(
            {
                "stage": name,
                "declared_jobs": declared_jobs,
                "instantiated_jobs": len(jobs),
                "accepted_jobs": accepted,
                "summary": _summary_path(protocol, name).is_file(),
                "decision": _decision_path(protocol, name).is_file(),
            }
        )
    return {
        "schema_version": "topology-temporal-v2-fast-status/v1",
        "fast_protocol_sha256": digest,
        "stages": stages,
        "candidate_frozen": _receipt_path(protocol, "candidate_receipt").is_file(),
        "promotion_gate": (
            _load_json(_receipt_path(protocol, "promotion_gate_receipt")).get("decision")
            if _receipt_path(protocol, "promotion_gate_receipt").is_file()
            else "TBD"
        ),
    }


def run_development(
    protocol: dict[str, Any],
    effective: dict[str, Any],
    protocol_path: Path,
    base_path: Path,
    digest: str,
    *,
    device: str,
    workers: int,
) -> int:
    linear = ("factorial_anchor", "merge_probe", "query_probe", "gated_probe")
    for stage in linear:
        code = execute_stage(
            protocol,
            effective,
            protocol_path,
            base_path,
            digest,
            stage,
            device=device,
            workers=workers,
            continue_on_error=False,
        )
        if code:
            return code
        if summarize_stage(protocol, effective, digest, stage):
            return 1
        decision = decide_stage(protocol, digest, stage)
        if stage == "gated_probe" and decision.get("continue_to_soft") is not True:
            return 2

    code = execute_stage(
        protocol,
        effective,
        protocol_path,
        base_path,
        digest,
        "soft_center",
        device=device,
        workers=workers,
    )
    if code or summarize_stage(protocol, effective, digest, "soft_center"):
        return 1
    centre = decide_stage(protocol, digest, "soft_center")
    if centre.get("run_bracket") is True:
        code = execute_stage(
            protocol,
            effective,
            protocol_path,
            base_path,
            digest,
            "soft_bracket",
            device=device,
            workers=workers,
        )
        if code or summarize_stage(protocol, effective, digest, "soft_bracket"):
            return 1
        decide_stage(protocol, digest, "soft_bracket")

    select_candidate(protocol, effective, digest)
    code = execute_stage(
        protocol,
        effective,
        protocol_path,
        base_path,
        digest,
        "promotion",
        device=device,
        workers=workers,
    )
    if code or summarize_stage(protocol, effective, digest, "promotion"):
        return 1
    gate = compute_promotion_gate(protocol, digest)
    return 0 if gate["decision"] == "pass" else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("status")
    for command in ("plan", "run"):
        child = sub.add_parser(command)
        child.add_argument(
            "--stage",
            choices=(*DEVELOPMENT_STAGES, "formal_test", "candidate_validation"),
            required=True,
        )
        child.add_argument("--device", default="cuda")
        if command == "run":
            child.add_argument("--workers", type=int, default=1)
            child.add_argument("--max-jobs", type=int)
            child.add_argument("--continue-on-error", action="store_true")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--stage", choices=tuple(protocol_stage for protocol_stage in (*DEVELOPMENT_STAGES, "formal_test")), required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--stage", choices=("factorial_anchor", "merge_probe", "query_probe", "gated_probe", "soft_center", "soft_bracket"), required=True)
    sub.add_parser("select-candidate")
    sub.add_parser("promotion-gate")
    development = sub.add_parser("run-development")
    development.add_argument("--device", default="cuda")
    development.add_argument("--workers", type=int, default=4)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol, effective, digest, base_path = load_protocol(protocol_path)
    if args.command == "validate":
        payload = protocol_status(protocol, effective, digest)
        payload.update(
            {
                "valid": True,
                "fast_protocol": str(protocol_path),
                "base_contract": str(base_path),
                "base_contract_sha256": str(protocol["base_contract_sha256"]),
                "frozen_dependencies": full.frozen_dependency_hashes(effective),
                "formal_test_locked": not _receipt_path(protocol, "promotion_gate_receipt").is_file(),
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(protocol_status(protocol, effective, digest), ensure_ascii=False, indent=2))
        return 0
    if args.command in ("plan", "run"):
        stage = args.stage
        _require(stage is not None, "--stage is required")
        # Accept the original phase spelling for the two identically named stages.
        if stage == "candidate_validation":
            stage = "promotion"
        if args.command == "plan":
            return execute_stage(
                protocol,
                effective,
                protocol_path,
                base_path,
                digest,
                stage,
                device=args.device,
                workers=1,
                dry_run=True,
            )
        return execute_stage(
            protocol,
            effective,
            protocol_path,
            base_path,
            digest,
            stage,
            device=args.device,
            workers=args.workers,
            max_jobs=args.max_jobs,
            continue_on_error=args.continue_on_error,
        )
    if args.command == "summarize":
        return summarize_stage(protocol, effective, digest, args.stage)
    if args.command == "decide":
        decide_stage(protocol, digest, args.stage)
        return 0
    if args.command == "select-candidate":
        select_candidate(protocol, effective, digest)
        return 0
    if args.command == "promotion-gate":
        gate = compute_promotion_gate(protocol, digest)
        return 0 if gate["decision"] == "pass" else 2
    if args.command == "run-development":
        return run_development(
            protocol,
            effective,
            protocol_path,
            base_path,
            digest,
            device=args.device,
            workers=args.workers,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
