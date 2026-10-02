"""Frozen STRT control vs route-action consistency diagnostic.

This runner reuses D1's saved-model evaluator and performs no learning.  The
route-action arm is selected through the separately registered
``sac_mlp_d1_st_rt_routeact_v1`` environment flag; both arms load the same
frozen ST-RT checkpoint and evaluate the same sorted traffic templates.

Preview is the default.  ``--start`` creates a new root, archives code and the
effective traffic pool, then launches one CPU evaluation process per arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
WORKSPACE = SCRIPT.parents[3]
FAST_DEVELOPER = SCRIPT.parent
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
EVAL_SPLIT = "validation"
SEED_START = 10_000
EPISODES = 100
CONTROL_METHOD = "sac_mlp_d1_st_rt"
ROUTEACT_METHOD = "sac_mlp_d1_st_rt_routeact_v1"
ARMS = ("frozen_control", "routeact")
ARM_METHODS = {
    "frozen_control": CONTROL_METHOD,
    "routeact": ROUTEACT_METHOD,
}
ARM_DIRS = {"frozen_control": "control", "routeact": "routeact"}
REFERENCE_RUN = (
    WORKSPACE / "runs" / "d0929_100k_diag"
    / f"{CONTROL_METHOD}__{SCENARIO}_depart4p0"
)
REFERENCE_MODEL = REFERENCE_RUN / "final_model.zip"
REFERENCE_EVALUATION = REFERENCE_RUN / "evaluation_results.json"
REFERENCE_COMPLETE = REFERENCE_RUN / "training_complete.json"
REFERENCE_ARGUMENTS = REFERENCE_RUN / "arguments.json"
REFERENCE_EVAL_MANIFEST = REFERENCE_RUN / "diagnostics" / "eval" / "manifest.json"
DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "srtact_1002"
EXPECTED_MODEL_SHA256 = "070b05b5ad2b60ce6fc4e47ee4508f0c1d2f1dd45579a0462d50e7d6a93f6262"
EXPECTED_RAW_STEPS = 100_000
EXPECTED_UPDATES = 95_001
PATH_LIMIT = 240


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runtime():
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    if str(FAST_DEVELOPER) not in sys.path:
        sys.path.insert(0, str(FAST_DEVELOPER))
    import launch_sorted_routeaware_3slot as sorted_archive
    import train_intersection_yield_v2 as base
    import train_intersection_yield_v2_d1 as d1
    from algos.sb3_torch import evaluation

    return sorted_archive, base, d1, evaluation


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _row_seed(row: dict[str, Any]) -> int | None:
    value = row.get("episode_seed", row.get("seed"))
    return None if value is None else int(value)


def _outcome(row: dict[str, Any]) -> str:
    labels = [
        key for key in ("success", "collision", "timeout", "off_route")
        if bool(row.get(key, False))
    ]
    if len(labels) == 1:
        return labels[0]
    return "unknown" if not labels else "multi:" + "+".join(labels)


def _method_config_checks(d1) -> dict[str, dict[str, Any]]:
    for method in (CONTROL_METHOD, ROUTEACT_METHOD):
        if method not in d1.DISPATCH or method not in d1.D1_CONFIG:
            raise RuntimeError(f"D1 registry is missing {method!r}")
        if d1.DISPATCH[method][0] != "d1":
            raise RuntimeError(f"{method} is not dispatched through D1 evaluation")
        if d1.PARENT.get(method) != "sac_mlp":
            raise RuntimeError(f"{method} must use the SAC+MLP D1 parent")
        if d1._env_adapter_d1(method) != "base":
            raise RuntimeError(f"{method} must retain the base environment contract")

    control = dict(d1.D1_CONFIG[CONTROL_METHOD])
    routeact = dict(d1.D1_CONFIG[ROUTEACT_METHOD])
    shared_keys = (
        "use_route", "use_topology", "use_slots", "use_incremental_slots",
        "use_parameter_matched_nonlinear_slots", "use_graph_slt", "use_sbs",
        "representation_coef", "slot_balance_coef", "use_topology_actor_intent",
        "use_topology_relations", "use_topology_goal", "use_route_reachability",
    )
    defaults = {
        "use_route": False,
        "use_topology": False,
        "use_slots": False,
        "use_incremental_slots": False,
        "use_parameter_matched_nonlinear_slots": False,
        "use_graph_slt": False,
        "use_sbs": False,
        "representation_coef": 0.0,
        "slot_balance_coef": 0.0,
        "use_topology_actor_intent": False,
        "use_topology_relations": False,
        "use_topology_goal": False,
        "use_route_reachability": False,
    }
    mismatches = {
        key: (control.get(key, defaults[key]), routeact.get(key, defaults[key]))
        for key in shared_keys
        if control.get(key, defaults[key]) != routeact.get(key, defaults[key])
    }
    if mismatches:
        raise RuntimeError(f"route-action arm changes encoder/model config: {mismatches}")
    if control.get("use_route") is not True or control.get("use_topology") is not False:
        raise RuntimeError("frozen reference must be registered as ST-RT without Topo")
    if routeact.get("use_route_action_veto") is not True:
        raise RuntimeError("route-action method must explicitly enable use_route_action_veto")
    if control.get("use_route_action_veto", False):
        raise RuntimeError("frozen control must leave route-action veto disabled")
    return {CONTROL_METHOD: control, ROUTEACT_METHOD: routeact}


def _validate_reference() -> dict[str, Any]:
    for path in (
        REFERENCE_MODEL,
        REFERENCE_EVALUATION,
        REFERENCE_COMPLETE,
        REFERENCE_ARGUMENTS,
        REFERENCE_EVAL_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen reference artifact: {path}")
    digest = _sha256(REFERENCE_MODEL)
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"reference model SHA changed: {digest}")

    arguments = _read_json(REFERENCE_ARGUMENTS)
    complete = _read_json(REFERENCE_COMPLETE)
    evaluation = _read_json(REFERENCE_EVALUATION)
    identity = evaluation.get("identity") or {}
    records = evaluation.get("episode_records") or []
    expected_seeds = list(range(SEED_START, SEED_START + EPISODES))
    actual_seeds = [_row_seed(row) for row in records]
    if arguments.get("method") != CONTROL_METHOD or arguments.get("parent") != "sac_mlp":
        raise RuntimeError("reference arguments do not identify the frozen SAC+MLP ST-RT model")
    if arguments.get("scenario") != SCENARIO or float(identity.get("depart_scale", -1)) != DEPART_SCALE:
        raise RuntimeError("reference scene/depart scale differs from sorted depart4")
    if int(arguments.get("seed", -1)) != 0:
        raise RuntimeError("reference training seed is not 0")
    if int(arguments.get("raw_budget", -1)) != EXPECTED_RAW_STEPS:
        raise RuntimeError("reference arguments do not record a fresh 100k raw budget")
    if int(complete.get("raw_steps", -1)) != EXPECTED_RAW_STEPS:
        raise RuntimeError("reference training completion does not show 100k raw steps")
    if int(complete.get("updates", -1)) != EXPECTED_UPDATES:
        raise RuntimeError("reference completion update count differs from 95001")
    if complete.get("checkpoint_sha256") != digest or identity.get("checkpoint_sha256") != digest:
        raise RuntimeError("final model SHA does not match both completion and original evaluation")
    if identity.get("method") != CONTROL_METHOD or identity.get("scenario") != SCENARIO:
        raise RuntimeError("original evaluation identity is not the expected STRT sorted run")
    if len(records) != EPISODES or actual_seeds != expected_seeds:
        raise RuntimeError("original evaluation is not the ordered seed set 10000..10099")
    for index, row in enumerate(records):
        if _outcome(row) not in ("success", "collision", "timeout", "off_route"):
            raise RuntimeError(f"original evaluation row {index} has nonexclusive outcome")
        if not row.get("traffic_variant"):
            raise RuntimeError(f"original evaluation row {index} lacks traffic_variant")

    manifest = _read_json(REFERENCE_EVAL_MANIFEST)
    effective = manifest.get("effective_traffic_files") or []
    if len(effective) != 30:
        raise RuntimeError(f"reference evaluation manifest must identify 30 effective templates; got {len(effective)}")
    effective_by_name = {Path(item["path"]).name: item for item in effective}
    if len(effective_by_name) != 30:
        raise RuntimeError("reference manifest has duplicate or missing effective traffic filenames")
    for row in records:
        variant = Path(str(row["traffic_variant"])).name
        if variant not in effective_by_name:
            raise RuntimeError(f"reference eval traffic variant absent from manifest: {variant}")

    return {
        "model_path": str(REFERENCE_MODEL.resolve()),
        "model_sha256": digest,
        "training_complete_path": str(REFERENCE_COMPLETE.resolve()),
        "raw_steps": int(complete["raw_steps"]),
        "updates": int(complete["updates"]),
        "training_seed": int(arguments["seed"]),
        "evaluation_path": str(REFERENCE_EVALUATION.resolve()),
        "evaluation_identity": identity,
        "evaluation_episodes": len(records),
        "evaluation_seeds": {"start": expected_seeds[0], "end_inclusive": expected_seeds[-1]},
        "reference_eval_outcomes": dict(Counter(_outcome(row) for row in records)),
        "reference_traffic_manifest_path": str(REFERENCE_EVAL_MANIFEST.resolve()),
        "reference_effective_traffic": effective,
        "reference_traffic_variants_per_seed": [
            {"seed": _row_seed(row), "traffic_variant": row.get("traffic_variant")}
            for row in records
        ],
        "reference_arguments": arguments,
    }


def _preflight(run_root: Path, *, smoke: bool) -> dict[str, Any]:
    common, base, d1, evaluation = _load_runtime()
    model_configs = _method_config_checks(d1)
    reference = _validate_reference()
    root = Path(run_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse diagnostic root: {root}")

    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    if d1.NEW_SCENARIO != SCENARIO or d1.EVAL_TRAFFIC_SPLIT != EVAL_SPLIT:
        raise RuntimeError("D1 failed to select the original sorted validation scene")
    if base.SEED_START.get("sac_mlp") != SEED_START:
        raise RuntimeError(f"expected SAC+MLP evaluation seed start {SEED_START}")
    if int(base.SEED) != 0 or int(base.ACTION_REPEAT) != 3:
        raise RuntimeError("current D1/base evaluation seed or action-repeat contract changed")
    expected_reward = {
        "success_reward": 10.0,
        "collision_reward": -10.0,
        "off_route_reward": -10.0,
        "timeout_reward": -5.0,
        "progress_scale": 0.02,
        "step_cost": 0.01,
    }
    if dict(base.REWARD) != expected_reward:
        raise RuntimeError(f"unexpected reward contract in D1 evaluator: {base.REWARD}")
    reward_protocol = getattr(evaluation, "EVALUATION_RETURN_PROTOCOL_VERSION", None)
    if reward_protocol != "environment_step_reward_v2":
        raise RuntimeError(f"unexpected shaped evaluation reward protocol: {reward_protocol!r}")

    assets = common.inspect_sorted_assets()
    if int(assets.get("traffic_template_count", 0)) != 30:
        raise RuntimeError("current sorted source pool is not exactly 30 templates")
    traffic_hashes = [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in assets["traffic_templates"]
    ]
    ref_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in reference["reference_effective_traffic"]
    }
    source_names = {Path(item["path"]).name for item in assets["traffic_templates"]}
    if set(ref_hashes) != source_names:
        raise RuntimeError("reference traffic filenames differ from the current sorted 30-template pool")

    output_paths = [
        root / "experiment_manifest.json",
        root / "source_archive" / "manifest.json",
        root / "launcher_status.json",
        root / "control" / "evaluation_results.json",
        root / "routeact" / "evaluation_results.json",
        root / "routeact" / "diagnostics" / "eval" / "route_action_consistency.jsonl",
    ]
    archived_paths = [
        root / "source_archive" / "src" / path.relative_to(PROJECT)
        for path in common._source_candidates()
    ]
    maximum_path = max((len(str(path)) for path in output_paths + archived_paths), default=0)
    if maximum_path > PATH_LIMIT:
        raise OSError(f"anticipated diagnostic path length {maximum_path} exceeds {PATH_LIMIT}")

    d1_sig = str(__import__("inspect").signature(d1._evaluate_d1))
    return {
        "schema_version": "sorted_st_rt_routeact_frozen_diagnostic_v1",
        "created_at_utc": _utc_now(),
        "run_root": str(root),
        "mode": "smoke_functional_only" if smoke else "formal_frozen_diagnostic",
        "arms": list(ARMS),
        "method_by_arm": dict(ARM_METHODS),
        "scene": {"scenario": SCENARIO, "depart_scale": DEPART_SCALE, "traffic_split": EVAL_SPLIT},
        "evaluation": {
            "episodes": 1 if smoke else EPISODES,
            "seed_start": SEED_START,
            "seed_end_inclusive": SEED_START if smoke else SEED_START + EPISODES - 1,
            "deterministic": True,
            "device": "cpu (same device used by the frozen reference evaluation)",
            "behavior_diagnostics": True,
            "route_action_sidecar_expected_for_routeact": True,
            "no_learning_or_replay_updates": True,
            "checkpoint": str(REFERENCE_MODEL.resolve()),
            "checkpoint_sha256": reference["model_sha256"],
        },
        "frozen_reference": reference,
        "method_configs": model_configs,
        "reward": {
            "evaluation_protocol": reward_protocol,
            "raw_return_protocol": getattr(evaluation, "RAW_RETURN_PROTOCOL_VERSION", None),
            "component_protocol": getattr(evaluation, "REWARD_COMPONENT_PROTOCOL_VERSION", None),
            "six_component_keys": list(getattr(evaluation, "REWARD_COMPONENT_KEYS", ())),
            "training_reward_coefficients": expected_reward,
            "control_and_routeact_share_evaluator_reward_code": True,
        },
        "traffic": {
            "template_count": 30,
            "pool_protocol": "same complete effective 30-template pool; sorted departures scaled by 4.0; validation seeds 10000..10099",
            "current_source_template_hashes": traffic_hashes,
            "reference_effective_template_hashes": reference["reference_effective_traffic"],
            "reference_and_current_source_names_match": True,
        },
        "max_parallel_processes": 2,
        "worker_processes_are_independent": True,
        "cuda_used": False,
        "resume": False,
        "training": False,
        "d1_evaluator_signature": d1_sig,
        "path_check": {"limit_chars": PATH_LIMIT, "maximum_anticipated_path_chars": maximum_path},
        "source_files_to_archive": len(common._source_candidates()),
    }


def _state_fingerprint(model) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
        tensor_count += 1
    return {
        "policy_state_sha256": digest.hexdigest(),
        "state_tensor_count": tensor_count,
        "n_updates": int(getattr(model, "_n_updates", 0)),
    }


def _install_load_capture(d1, fingerprints: list[dict[str, Any]], models: list[Any]):
    from algos.sb3_torch.sac import SceneRepresentationSAC

    model_class = SceneRepresentationSAC
    had_local_load = "load" in model_class.__dict__
    old_local_load = model_class.__dict__.get("load")
    original_bound_load = model_class.load

    def load_capture(_class, *args, **kwargs):
        model = original_bound_load(*args, **kwargs)
        fingerprints.append({"phase": "before_eval", **_state_fingerprint(model)})
        models.append(model)
        return model

    model_class.load = classmethod(load_capture)

    def restore():
        if had_local_load:
            setattr(model_class, "load", old_local_load)
        else:
            try:
                delattr(model_class, "load")
            except AttributeError:
                pass

    return restore


def _apply_eval_protocol(common, base, d1, run_root: Path) -> tuple[Path, ...]:
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    # Reuse the already archived/verified effective pool. It is read-only in
    # this diagnostic and exactly matches the original final evaluation.
    base.RESULT_ROOT = REFERENCE_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT
    effective = tuple(base._traffic_paths_for_scale(DEPART_SCALE))
    if len(effective) != 30:
        raise RuntimeError(f"expected 30 effective traffic files; found {len(effective)}")
    expected = {
        Path(item["path"]).name: item["sha256"]
        for item in _read_json(REFERENCE_EVAL_MANIFEST)["effective_traffic_files"]
    }
    for path in effective:
        digest = _sha256(path)
        if expected.get(path.name) != digest:
            raise RuntimeError(f"effective traffic differs from frozen reference for {path.name}")
    return effective


def _worker_main(args) -> int:
    arm = str(args.worker_arm)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    run_root = Path(args.run_root).resolve()
    arm_dir = run_root / ARM_DIRS[arm]
    arm_dir.mkdir(parents=True, exist_ok=False)
    status_path = arm_dir / "worker_status.json"
    common, base, d1, _evaluation = _load_runtime()
    _apply_eval_protocol(common, base, d1, run_root)
    method = ARM_METHODS[arm]
    parent = d1.PARENT[method]
    base.SEED_START[parent] = int(args.seed_start)
    d1.EVAL_EPISODES_TOTAL = int(args.episodes)
    d1.SMOKE_EVAL_EPISODES = int(args.episodes)
    base.RESULT_ROOT = REFERENCE_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT

    fingerprints: list[dict[str, Any]] = []
    loaded_models: list[Any] = []
    restore_loader = _install_load_capture(d1, fingerprints, loaded_models)
    _write_json(status_path, {
        "status": "running",
        "arm": arm,
        "method": method,
        "pid": os.getpid(),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "checkpoint_sha256": _sha256(REFERENCE_MODEL),
        "device": "cpu",
        "started_at_utc": _utc_now(),
    })
    try:
        result = d1._evaluate_d1(
            method,
            arm_dir,
            REFERENCE_MODEL,
            smoke=bool(args.smoke),
            behavior_diagnostics=True,
            policy_shadow_probes=False,
            policy_shadow_eval_max_per_episode=3,
        )
        if len(loaded_models) != 1 or len(fingerprints) != 1:
            raise RuntimeError(
                f"expected exactly one loaded model; models={len(loaded_models)}, fingerprints={len(fingerprints)}"
            )
        after = {"phase": "after_eval", **_state_fingerprint(loaded_models[0])}
        fingerprints.append(after)
        before = fingerprints[0]
        if before["policy_state_sha256"] != after["policy_state_sha256"]:
            raise RuntimeError("evaluation changed frozen policy parameters/buffers")
        if before["n_updates"] != after["n_updates"]:
            raise RuntimeError("evaluation changed frozen model update count")
        model_after_eval_sha = _sha256(REFERENCE_MODEL)
        if model_after_eval_sha != EXPECTED_MODEL_SHA256:
            raise RuntimeError("frozen checkpoint file changed during evaluation")

        identity = result.get("identity") or {}
        if identity.get("checkpoint_sha256") != model_after_eval_sha:
            raise RuntimeError("D1 evaluator identity does not match the frozen checkpoint")
        if identity.get("device") not in (None, "cpu"):
            raise RuntimeError(f"D1 evaluator used an unexpected device: {identity.get('device')}")
        identity.update({
            "diagnostic_experiment": "sorted_st_rt_route_action_consistency_v1",
            "diagnostic_arm": arm,
            "base_method": CONTROL_METHOD,
            "base_checkpoint_path": str(REFERENCE_MODEL.resolve()),
            "base_checkpoint_sha256": EXPECTED_MODEL_SHA256,
            "eval_device": "cpu",
            "learning_or_replay_updates": 0,
            "resume": False,
            "intervention": (
                "none; frozen STRT control"
                if arm == "frozen_control"
                else "known direct-eligible current lane to known ineligible target lane only; lateral hold, speed unchanged"
            ),
            "model_policy_fingerprint": fingerprints,
        })
        result["identity"] = identity
        _write_json(arm_dir / "evaluation_results.json", result)

        records = result.get("episode_records") or []
        expected_seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.episodes)))
        observed_seeds = [_row_seed(row) for row in records]
        if len(records) != int(args.episodes) or observed_seeds != expected_seeds:
            raise RuntimeError(f"evaluation episode/seed mismatch: {len(records)} rows, seeds={observed_seeds}")
        invalid_outcomes = [i for i, row in enumerate(records) if _outcome(row) not in ("success", "collision", "timeout", "off_route")]
        if invalid_outcomes:
            raise RuntimeError(f"nonexclusive/unknown terminal outcomes at rows {invalid_outcomes[:8]}")

        route_sidecar = arm_dir / "diagnostics" / "eval" / "route_action_consistency.jsonl"
        if arm == "routeact" and not route_sidecar.is_file():
            raise RuntimeError("route-action method did not create its expected decision sidecar")
        route_rows = 0
        route_reasons: Counter[str] = Counter()
        if route_sidecar.is_file():
            with route_sidecar.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("record_type") == "decision":
                        route_rows += 1
                        route_reasons[str((row.get("decision") or {}).get("reason", "unknown"))] += 1

        summary_path = arm_dir / "diagnostics" / "eval" / "summary.json"
        diagnostic_summary = _read_json(summary_path) if summary_path.is_file() else {}
        errors = int(diagnostic_summary.get("diagnostic_error_count", 0))
        if errors != 0:
            raise RuntimeError(f"behavior diagnostics report {errors} errors")
        status = {
            "status": "complete",
            "arm": arm,
            "method": method,
            "pid": os.getpid(),
            "episodes": len(records),
            "seed_start": int(args.seed_start),
            "seed_end_inclusive": int(args.seed_start) + int(args.episodes) - 1,
            "checkpoint_sha256": model_after_eval_sha,
            "policy_fingerprint_unchanged": before["policy_state_sha256"] == after["policy_state_sha256"],
            "update_count_unchanged": before["n_updates"] == after["n_updates"],
            "diagnostic_error_count": errors,
            "route_action_decision_rows": route_rows,
            "route_action_reasons": dict(route_reasons),
            "completed_at_utc": _utc_now(),
        }
        _write_json(status_path, status)
        return 0
    except BaseException as exc:
        _write_json(status_path, {
            "status": "failed",
            "arm": arm,
            "method": method,
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "failed_at_utc": _utc_now(),
        })
        raise
    finally:
        restore_loader()


def _read_traffic_variant_rows(run_dir: Path) -> dict[int, str]:
    result = _read_json(run_dir / "evaluation_results.json")
    by_seed = {}
    for row in result.get("episode_records") or []:
        seed = _row_seed(row)
        if seed is None:
            continue
        if seed in by_seed:
            raise ValueError(f"duplicate eval seed {seed} in {run_dir}")
        variant = row.get("traffic_variant")
        by_seed[seed] = Path(str(variant)).name if variant else ""
    return by_seed


def _pair_results(run_root: Path, *, episodes: int, seed_start: int) -> dict[str, Any]:
    control_path = run_root / ARM_DIRS["frozen_control"] / "evaluation_results.json"
    routeact_path = run_root / ARM_DIRS["routeact"] / "evaluation_results.json"
    control = _read_json(control_path)
    routeact = _read_json(routeact_path)
    control_records = control.get("episode_records") or []
    routeact_records = routeact.get("episode_records") or []
    expected_seeds = list(range(seed_start, seed_start + episodes))
    by_arm = {}
    for arm, records in (("frozen_control", control_records), ("routeact", routeact_records)):
        indexed = {}
        for row in records:
            seed = _row_seed(row)
            if seed is None or seed in indexed:
                raise ValueError(f"missing or duplicate eval seed in {arm}")
            indexed[seed] = row
        if sorted(indexed) != expected_seeds:
            raise ValueError(f"{arm} seeds do not match expected evaluation range")
        by_arm[arm] = indexed

    matrix: dict[str, dict[str, int]] = {}
    changed = []
    traffic_variant_mismatches = []
    control_variants = _read_traffic_variant_rows(run_root / ARM_DIRS["frozen_control"])
    routeact_variants = _read_traffic_variant_rows(run_root / ARM_DIRS["routeact"])
    for seed in expected_seeds:
        left = _outcome(by_arm["frozen_control"][seed])
        right = _outcome(by_arm["routeact"][seed])
        matrix.setdefault(left, {}).setdefault(right, 0)
        matrix[left][right] += 1
        if left != right:
            changed.append({"seed": seed, "control": left, "routeact": right})
        if control_variants.get(seed) != routeact_variants.get(seed):
            traffic_variant_mismatches.append({
                "seed": seed,
                "control": control_variants.get(seed),
                "routeact": routeact_variants.get(seed),
            })

    ref_eval = _read_json(REFERENCE_EVALUATION)
    reference_variants = {
        _row_seed(row): Path(str(row.get("traffic_variant", ""))).name
        for row in ref_eval.get("episode_records", [])
    }
    reference_traffic_mismatches = [
        {
            "seed": seed,
            "reference": reference_variants.get(seed),
            "control": control_variants.get(seed),
        }
        for seed in expected_seeds
        if reference_variants.get(seed) != control_variants.get(seed)
    ]
    effective_by_name = {
        Path(item["path"]).name: item["sha256"]
        for item in _read_json(REFERENCE_EVAL_MANIFEST)["effective_traffic_files"]
    }
    selected_hashes = [
        {"traffic_variant": name, "sha256": effective_by_name.get(name)}
        for name in sorted(set(control_variants.values()) | set(routeact_variants.values()))
    ]
    for arm in ARMS:
        arm_result = _read_json(run_root / ARM_DIRS[arm] / "evaluation_results.json")
        identity = arm_result.get("identity") or {}
        if identity.get("checkpoint_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"{arm} evaluation used an unexpected checkpoint identity")
        if identity.get("evaluation_return_protocol_version") not in (None, "environment_step_reward_v2"):
            raise RuntimeError(f"{arm} evaluation reward protocol is unexpected")

    return {
        "schema_version": "sorted_st_rt_routeact_pairs_v1",
        "reference_model_sha256": EXPECTED_MODEL_SHA256,
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_end_inclusive": seed_start + episodes - 1,
        "orientation": "frozen_control outcome rows -> routeact outcome columns",
        "outcome_matrix": matrix,
        "changed_outcomes": changed,
        "traffic_variant_identity": {
            "control_vs_routeact_exact_pairs": episodes - len(traffic_variant_mismatches),
            "control_vs_routeact_mismatches": traffic_variant_mismatches,
            "reference_vs_control_exact_pairs": episodes - len(reference_traffic_mismatches),
            "reference_vs_control_mismatches": reference_traffic_mismatches,
            "selected_template_sha256": selected_hashes,
            "template_hash_source": str(REFERENCE_EVAL_MANIFEST.resolve()),
        },
        "interpretation_limit": "paired fixed-checkpoint intervention on one trained seed; not a multi-training-seed estimate",
        "source_results": {
            "control": str(control_path.resolve()),
            "routeact": str(routeact_path.resolve()),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    delays = (0.05, 0.10, 0.20)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _prepare_source_archive(run_root: Path, common, base, d1) -> dict[str, Any]:
    reference_pool = tuple(base._traffic_paths_for_scale(DEPART_SCALE))
    archive = common.archive_sources(run_root, reference_pool)
    docs = (
        FAST_DEVELOPER / "analysis" / "sorted_goalonly_task_diagnostic_protocol_20261002.md",
        FAST_DEVELOPER / "analysis" / "sorted_task_module_attribution_20261002.md",
    )
    archive_root = run_root / "source_archive"
    for source in docs:
        if source.is_file():
            destination = archive_root / "research" / source.name
            common._archive_file(source, destination, archive["files"])
    archive["source_file_count"] = len(archive["files"])
    archive["diagnostic_protocol_sources"] = [str(path.resolve()) for path in docs if path.is_file()]
    common.write_json(archive_root / "manifest.json", archive)
    return archive


def _start(args) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    plan = _preflight(run_root, smoke=bool(args.smoke))
    run_root.mkdir(parents=True, exist_ok=False)
    common, base, d1, evaluation = _load_runtime()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    # The old final evaluation's effective 30-file pool is reused read-only.
    base.RESULT_ROOT = REFERENCE_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT
    archive = _prepare_source_archive(run_root, common, base, d1)
    plan.update({
        "status": "starting",
        "started_at_utc": _utc_now(),
        "supervisor_pid": os.getpid(),
        "archive_manifest": str((run_root / "source_archive" / "manifest.json").resolve()),
        "source_archive_files": archive["source_file_count"],
        "effective_traffic_pool_reused_read_only_from": str(REFERENCE_RUN.parents[0].resolve()),
        "workers": {},
    })
    _write_json(run_root / "experiment_manifest.json", plan)
    status_lock = threading.Lock()
    live_status: dict[str, Any] = {"status": "running", "pid": os.getpid(), "workers": {}}
    _write_json(run_root / "launcher_status.json", live_status)
    script = SCRIPT
    episodes = 1 if args.smoke else EPISODES
    seed_start = int(args.smoke_seed) if args.smoke else SEED_START

    def launch(arm: str):
        arm_dir = run_root / ARM_DIRS[arm]
        stdout_path = run_root / f"{ARM_DIRS[arm]}.stdout.log"
        stderr_path = run_root / f"{ARM_DIRS[arm]}.stderr.log"
        command = [
            sys.executable, "-u", str(script),
            "--worker-arm", arm,
            "--run-root", str(run_root),
            "--episodes", str(episodes),
            "--seed-start", str(seed_start),
        ]
        if args.smoke:
            command.append("--smoke")
        _write_json(run_root / f"{ARM_DIRS[arm]}.command.json", {
            "command": command,
            "cwd": str(PROJECT),
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        })
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        with stdout_path.open("w", encoding="utf-8", newline="\n") as out, stderr_path.open("w", encoding="utf-8", newline="\n") as err:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                creationflags=flags,
            )
            with status_lock:
                live_status["workers"][arm] = {
                    "pid": process.pid,
                    "status": "running",
                    "method": ARM_METHODS[arm],
                    "command": command,
                    "stdout": str(stdout_path.resolve()),
                    "stderr": str(stderr_path.resolve()),
                }
                _write_json(run_root / "launcher_status.json", live_status)
            code = process.wait()
        return arm, process.pid, code

    outcomes: dict[str, Any] = {}
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, arm): arm for arm in ARMS}
        for future in concurrent.futures.as_completed(futures):
            arm, pid, code = future.result()
            outcomes[arm] = {
                "pid": pid,
                "exit_code": code,
                "status": "complete" if code == 0 else "failed",
            }
            with status_lock:
                live_status["workers"][arm] = dict(live_status["workers"].get(arm, {}), **outcomes[arm])
                live_status["status"] = "failed" if code != 0 else "running"
                _write_json(run_root / "launcher_status.json", live_status)

    complete = len(outcomes) == len(ARMS) and all(row["exit_code"] == 0 for row in outcomes.values())
    paired = None
    if complete:
        paired = _pair_results(run_root, episodes=episodes, seed_start=seed_start)
        _write_json(run_root / "paired_outcomes.json", paired)
    _write_json(run_root / "launcher_status.json", {
        "status": "complete" if complete else "failed",
        "pid": os.getpid(),
        "workers": outcomes,
        "completed_at_utc": _utc_now(),
        "paired_outcomes": str((run_root / "paired_outcomes.json").resolve()) if complete else None,
    })
    plan["status"] = "complete" if complete else "failed"
    plan["completed_at_utc"] = _utc_now()
    _write_json(run_root / "experiment_manifest.json", plan)
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="preflight without creating output or running SUMO")
    parser.add_argument("--start", action="store_true", help="create an isolated root and launch two frozen eval workers")
    parser.add_argument("--smoke", action="store_true", help="one original validation seed per arm; functional only")
    parser.add_argument("--smoke-seed", type=int, default=SEED_START)
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=EPISODES, help=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=SEED_START, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_arm:
        return _worker_main(args)
    if args.start and args.dry_run:
        parser.error("choose either --dry-run or --start")
    if args.smoke and not (args.start or args.dry_run):
        parser.error("--smoke requires --dry-run or --start")
    if not args.start and not args.dry_run:
        parser.error("choose --dry-run to preview or --start to execute")
    if args.smoke and not SEED_START <= args.smoke_seed < SEED_START + EPISODES:
        parser.error("--smoke-seed must be from the original validation range 10000..10099")
    try:
        plan = _preflight(args.run_root, smoke=bool(args.smoke))
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        return _start(args)
    except Exception as exc:
        print(f"preflight/runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
