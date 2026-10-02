"""Bounded frozen-policy diagnostic for sorted RouteAct and ConflictTiming.

This runner uses the existing D1 saved-model evaluator and never calls learn()
or changes replay data.  It first runs one unmodified control episode for each
checkpoint and compares it field-by-field with that checkpoint's original
official evaluation.  Only if both controls match does it run the three
pre-registered 40-seed interventions.

Default invocation is a read-only preview.  --start creates an isolated root;
it is the only option that executes SUMO episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
WORKSPACE = SCRIPT.parents[3]
FAST_DEVELOPER = SCRIPT.parent
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
EVAL_SPLIT = "validation"
SEED_START = 10_000
FORMAL_EPISODES = 40
MAX_PARALLEL = 2
PATH_LIMIT = 240

ROUTEACT_METHOD = "sac_mlp_d1_st_rt_routeact_v1"
STRT_METHOD = "sac_mlp_d1_st_rt"
CONFLICT_METHOD = "sac_mlp_d1_st_rt_conflicttime_v1"

ROUTEACT_RUN = (
    WORKSPACE / "runs" / "sortct_1002"
    / f"{ROUTEACT_METHOD}__{SCENARIO}_depart4p0"
)
CONFLICT_RUN = (
    WORKSPACE / "runs" / "sortct_1002"
    / f"{CONFLICT_METHOD}__{SCENARIO}_depart4p0"
)
ROUTEACT_SHA256 = "8a36cd283d13e08b0c83cd16d988147c1e1213f7e13a610731c1a8532176bebe"
CONFLICT_SHA256 = "58ce993f4cbf1068c46a0bc52d463d2a7b028a8afb7d1b128cf6c2c7e84da9a5"

DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "sortct_frozen_1003"

ARM_SPEC: dict[str, dict[str, Any]] = {
    "a_control_smoke": {
        "phase": "control_smoke", "method": ROUTEACT_METHOD,
        "checkpoint_method": ROUTEACT_METHOD, "probe": None,
        "reference_run": ROUTEACT_RUN, "reference_seed_start": SEED_START,
    },
    "b_control_smoke": {
        "phase": "control_smoke", "method": CONFLICT_METHOD,
        "checkpoint_method": CONFLICT_METHOD, "probe": None,
        "reference_run": CONFLICT_RUN, "reference_seed_start": SEED_START,
    },
    "a_remove_route_veto": {
        "phase": "intervention", "method": STRT_METHOD,
        "checkpoint_method": ROUTEACT_METHOD, "probe": None,
        "reference_run": ROUTEACT_RUN, "reference_seed_start": SEED_START,
    },
    "b_conflict_off": {
        "phase": "intervention", "method": CONFLICT_METHOD,
        "checkpoint_method": CONFLICT_METHOD, "probe": "route_conflict_off",
        "reference_run": CONFLICT_RUN, "reference_seed_start": SEED_START,
    },
    "b_times_off": {
        "phase": "intervention", "method": CONFLICT_METHOD,
        "checkpoint_method": CONFLICT_METHOD, "probe": "route_conflict_times_off",
        "reference_run": CONFLICT_RUN, "reference_seed_start": SEED_START,
    },
}
CONTROL_ARMS = ("a_control_smoke", "b_control_smoke")
INTERVENTION_ARMS = ("a_remove_route_veto", "b_conflict_off", "b_times_off")
ALL_ARMS = CONTROL_ARMS + INTERVENTION_ARMS
ARM_DIR = {
    "a_control_smoke": "a_control",
    "b_control_smoke": "b_control",
    "a_remove_route_veto": "a_no_veto",
    "b_conflict_off": "b_conflict_off",
    "b_times_off": "b_times_off",
}

REWARD_KEYS = (
    "reward_success", "reward_collision", "reward_off_route",
    "reward_timeout", "reward_step_cost", "reward_progress",
)
CONTROL_FIELDS = (
    "seed", "traffic_variant", "success", "collision", "timeout", "off_route",
    "raw_steps", "decision_steps", "environment_steps", "episode_return",
    "raw_episode_return", *REWARD_KEYS,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(payload, encoding="utf-8")
    last_error = None
    for delay in (0.0, 0.05, 0.1, 0.2):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temp, path)
            return
        except PermissionError as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def _load_runtime():
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    if str(FAST_DEVELOPER) not in sys.path:
        sys.path.insert(0, str(FAST_DEVELOPER))
    import launch_sorted_routeaware_3slot as common
    import train_intersection_yield_v2 as base
    import train_intersection_yield_v2_d1 as d1
    from algos.sb3_torch import evaluation
    return common, base, d1, evaluation


def _seed(row: dict[str, Any]) -> int | None:
    value = row.get("seed", row.get("episode_seed"))
    return None if value is None else int(value)


def _outcome(row: dict[str, Any]) -> str:
    labels = [key for key in ("success", "collision", "timeout", "off_route") if bool(row.get(key, False))]
    return labels[0] if len(labels) == 1 else ("unknown" if not labels else "multi:" + "+".join(labels))


def _method_config_checks(d1) -> dict[str, dict[str, Any]]:
    required = (ROUTEACT_METHOD, STRT_METHOD, CONFLICT_METHOD)
    for method in required:
        if method not in d1.DISPATCH or method not in d1.D1_CONFIG:
            raise RuntimeError(f"D1 registry is missing {method!r}")
        if d1.DISPATCH[method][0] != "d1" or d1.PARENT.get(method) != "sac_mlp":
            raise RuntimeError(f"{method} must dispatch to the SAC+MLP D1 evaluator")
        if d1._env_adapter_d1(method) != "base":
            raise RuntimeError(f"{method} must retain the base environment contract")

    route = dict(d1.D1_CONFIG[ROUTEACT_METHOD])
    no_veto = dict(d1.D1_CONFIG[STRT_METHOD])
    model_keys = (
        "use_route", "use_topology", "use_slots", "use_incremental_slots",
        "use_parameter_matched_nonlinear_slots", "use_graph_slt", "use_sbs",
        "representation_coef", "slot_balance_coef", "use_topology_actor_intent",
        "use_topology_relations", "use_topology_goal", "use_route_reachability",
        "use_route_conflict_timing",
    )
    defaults = {
        "use_route": False, "use_topology": False, "use_slots": False,
        "use_incremental_slots": False, "use_parameter_matched_nonlinear_slots": False,
        "use_graph_slt": False, "use_sbs": False, "representation_coef": 0.0,
        "slot_balance_coef": 0.0, "use_topology_actor_intent": False,
        "use_topology_relations": False, "use_topology_goal": False,
        "use_route_reachability": False, "use_route_conflict_timing": False,
    }
    mismatches = {
        key: (route.get(key, defaults[key]), no_veto.get(key, defaults[key]))
        for key in model_keys
        if route.get(key, defaults[key]) != no_veto.get(key, defaults[key])
    }
    if mismatches:
        raise RuntimeError(f"A intervention changes the checkpoint network config: {mismatches}")
    if route.get("use_route_action_veto") is not True:
        raise RuntimeError("RouteAct checkpoint method must enable the route-action wrapper")
    if no_veto.get("use_route_action_veto", False):
        raise RuntimeError("A no-veto method unexpectedly retains the route-action wrapper")
    conflict = dict(d1.D1_CONFIG[CONFLICT_METHOD])
    if conflict.get("use_route_conflict_timing") is not True:
        raise RuntimeError("ConflictTiming method must enable its conflict encoder")
    return {ROUTEACT_METHOD: route, STRT_METHOD: no_veto, CONFLICT_METHOD: conflict}


def _reference_for(method: str) -> tuple[Path, str]:
    if method == ROUTEACT_METHOD:
        return ROUTEACT_RUN, ROUTEACT_SHA256
    if method == CONFLICT_METHOD:
        return CONFLICT_RUN, CONFLICT_SHA256
    raise KeyError(method)


def _validate_reference(method: str) -> dict[str, Any]:
    run, expected_sha = _reference_for(method)
    model = run / "final_model.zip"
    evaluation_path = run / "evaluation_results.json"
    complete_path = run / "training_complete.json"
    arguments_path = run / "arguments.json"
    eval_manifest_path = run / "diagnostics" / "eval" / "manifest.json"
    for path in (model, evaluation_path, complete_path, arguments_path, eval_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen reference artifact: {path}")
    actual_sha = _sha256(model)
    if actual_sha != expected_sha:
        raise RuntimeError(f"frozen {method} checkpoint SHA mismatch: {actual_sha}")
    evaluation = _read_json(evaluation_path)
    complete = _read_json(complete_path)
    arguments = _read_json(arguments_path)
    eval_manifest = _read_json(eval_manifest_path)
    identity = evaluation.get("identity") or {}
    records = evaluation.get("episode_records") or []
    expected_seeds = list(range(SEED_START, SEED_START + 100))
    actual_seeds = [_seed(row) for row in records]
    if complete.get("raw_steps") != 100_000 or complete.get("updates") != 95_001:
        raise RuntimeError(f"{method} completion record does not certify fresh100k/95001 updates")
    if complete.get("checkpoint_sha256") != actual_sha or identity.get("checkpoint_sha256") != actual_sha:
        raise RuntimeError(f"{method} final, completion and evaluation checkpoint identities disagree")
    if identity.get("scenario") != SCENARIO or float(identity.get("depart_scale", -1)) != DEPART_SCALE:
        raise RuntimeError(f"{method} evaluation is not sorted/depart4")
    if identity.get("episodes") != 100 or len(records) != 100 or actual_seeds != expected_seeds:
        raise RuntimeError(f"{method} reference eval must contain seeds 10000..10099 exactly once")
    if identity.get("eval_traffic_split") != EVAL_SPLIT or identity.get("smoke") is not False:
        raise RuntimeError(f"{method} reference eval split/smoke identity mismatch")
    eval_traffic = eval_manifest.get("effective_traffic_files") or []
    if (
        eval_manifest.get("scenario") != SCENARIO
        or float(eval_manifest.get("depart_scale", -1)) != DEPART_SCALE
        or eval_manifest.get("training_seed") != 0
        or eval_manifest.get("action_repeat") != 3
        or eval_manifest.get("evaluation_episodes") != 100
        or len(eval_traffic) != 30
        or eval_manifest.get("traffic_partition") != "same_complete_effective_pool_for_train_and_eval; no holdout"
    ):
        raise RuntimeError(f"{method} eval manifest does not certify sorted/depart4/seed0/30-pool protocol")
    invalid = [index for index, row in enumerate(records) if _outcome(row) not in ("success", "collision", "timeout", "off_route")]
    if invalid:
        raise RuntimeError(f"{method} reference evaluation has nonexclusive outcomes: {invalid[:5]}")
    if arguments.get("seed") != 0 or arguments.get("raw_budget") != 100_000 or arguments.get("scenario") != SCENARIO:
        raise RuntimeError(f"{method} training arguments do not certify seed0/sorted/raw100k")
    if arguments.get("smoke") is not False:
        raise RuntimeError(f"{method} training arguments do not certify a non-smoke run")
    return {
        "method": method,
        "run_dir": str(run.resolve()),
        "final_model": str(model.resolve()),
        "checkpoint_sha256": actual_sha,
        "training_complete": complete,
        "arguments": arguments,
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_identity": identity,
        "evaluation_summary": evaluation.get("summary") or {},
        "episode_records": records,
        "eval_manifest_path": str(eval_manifest_path.resolve()),
        "eval_manifest": eval_manifest,
    }


def _scan_shadow_gate(method: str, reference_run: Path) -> dict[str, Any]:
    names = ("route_conflict_off", "route_conflict_times_off")
    source_files = [
        reference_run / "diagnostics" / "train" / "policy_shadow_probes.jsonl",
        reference_run / "diagnostics" / "eval" / "policy_shadow_probes.jsonl",
    ]
    result = {
        "method": method,
        "probe_streams": [],
        "probes": {name: {"rows": 0, "valid": 0, "active_invalid": 0, "nonzero_action_rows": 0, "unique_samples": 0} for name in names},
    }
    samples: dict[str, set[str]] = {name: set() for name in names}
    for path in source_files:
        if not path.is_file():
            continue
        file_entry = {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
        result["probe_streams"].append(file_entry)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("probe_name")
                if name not in names:
                    continue
                stats = result["probes"][name]
                stats["rows"] += 1
                sid = row.get("sample_id")
                if sid is not None:
                    samples[name].add(str(sid))
                if row.get("applicable") is True and row.get("valid") is True:
                    stats["valid"] += 1
                    delta = row.get("action_delta_l2_normalized")
                    if delta is None:
                        base_action = row.get("baseline_action_normalized_mean")
                        shadow_action = row.get("shadow_action_normalized_mean")
                        if isinstance(base_action, list) and isinstance(shadow_action, list) and len(base_action) == len(shadow_action):
                            delta = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(base_action, shadow_action)))
                    if delta is not None and math.isfinite(float(delta)) and float(delta) > 1e-8:
                        stats["nonzero_action_rows"] += 1
                elif row.get("applicable") is True:
                    stats["active_invalid"] += 1
    for name in names:
        result["probes"][name]["unique_samples"] = len(samples[name])
        if result["probes"][name]["valid"] == 0 or result["probes"][name]["nonzero_action_rows"] == 0:
            raise RuntimeError(
                f"B intervention gate failed for {name}: "
                f"valid={result['probes'][name]['valid']} nonzero={result['probes'][name]['nonzero_action_rows']}"
            )
    if not result["probe_streams"]:
        raise FileNotFoundError(f"no normal policy-shadow streams found for B gate in {reference_run}")
    result["passed"] = True
    result["criterion"] = "both probes must have at least one applicable valid state with nonzero normalized action delta (>1e-8) in existing normal train/eval policy-shadow records"
    return result


def _config_reward_preflight(base, d1, evaluation) -> dict[str, Any]:
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    if d1.NEW_SCENARIO != SCENARIO or float(d1.DEPART_SCALE) != DEPART_SCALE or d1.EVAL_TRAFFIC_SPLIT != EVAL_SPLIT:
        raise RuntimeError("D1 failed to select sorted/depart4/validation")
    if base.SEED_START.get("sac_mlp") != SEED_START or int(base.SEED) != 0 or int(base.ACTION_REPEAT) != 3:
        raise RuntimeError("base seed/action-repeat contract differs from the frozen evaluations")
    expected_reward = {
        "success_reward": 10.0, "collision_reward": -10.0,
        "off_route_reward": -10.0, "timeout_reward": -5.0,
        "progress_scale": 0.02, "step_cost": 0.01,
    }
    if dict(base.REWARD) != expected_reward:
        raise RuntimeError(f"unexpected evaluation reward: {base.REWARD}")
    protocol = getattr(evaluation, "EVALUATION_RETURN_PROTOCOL_VERSION", None)
    if protocol != "environment_step_reward_v2":
        raise RuntimeError(f"unexpected evaluation return protocol {protocol!r}")
    return {"reward": expected_reward, "evaluation_return_protocol": protocol}


def _preflight(run_root: Path) -> dict[str, Any]:
    common, base, d1, evaluation = _load_runtime()
    configs = _method_config_checks(d1)
    route_ref = _validate_reference(ROUTEACT_METHOD)
    conflict_ref = _validate_reference(CONFLICT_METHOD)
    reward = _config_reward_preflight(base, d1, evaluation)
    root = Path(run_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse frozen intervention root: {root}")

    # Verify the current pool is the exact pool used in each saved final eval.
    # Preview is read-only: do not invoke an asset-generation helper here.
    assets = common.inspect_sorted_assets()
    if int(assets.get("traffic_template_count", 0)) != 30:
        raise RuntimeError("current sorted source pool is not exactly 30 templates")
    base.RESULT_ROOT = ROUTEACT_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT
    current_files = tuple(base._traffic_paths_for_scale(DEPART_SCALE))
    if len(current_files) != 30:
        raise RuntimeError(f"expected 30 sorted effective traffic templates, found {len(current_files)}")
    current_hashes = {Path(path).name: _sha256(Path(path)) for path in current_files}
    ref_hashes: dict[str, dict[str, str]] = {}
    for method, reference in ((ROUTEACT_METHOD, route_ref), (CONFLICT_METHOD, conflict_ref)):
        files = reference["eval_manifest"].get("effective_traffic_files") or []
        ref = {Path(item["path"]).name: item["sha256"] for item in files}
        if set(ref) != set(current_hashes) or any(current_hashes[name] != digest for name, digest in ref.items()):
            raise RuntimeError(f"current sorted traffic pool does not match {method} original evaluation assets")
        ref_hashes[method] = ref
    for seed in range(SEED_START, SEED_START + FORMAL_EPISODES):
        route_row = route_ref["episode_records"][seed - SEED_START]
        conflict_row = conflict_ref["episode_records"][seed - SEED_START]
        if route_row.get("traffic_variant") != conflict_row.get("traffic_variant"):
            raise RuntimeError(f"reference methods do not share traffic variant at seed {seed}")

    b_gate = _scan_shadow_gate(CONFLICT_METHOD, CONFLICT_RUN)
    output_paths = [
        root / "experiment_manifest.json", root / "source_archive" / "manifest.json",
        root / "launcher_status.json", root / "control" / "a_control" / "evaluation_results.json",
        root / "control" / "b_control" / "evaluation_results.json",
        root / "intervention" / "a_no_veto" / "evaluation_results.json",
        root / "intervention" / "b_conflict_off" / "evaluation_results.json",
        root / "intervention" / "b_times_off" / "evaluation_results.json",
    ]
    archived = [root / "source_archive" / "src" / path.relative_to(PROJECT) for path in common._source_candidates()]
    max_path = max((len(str(p)) for p in output_paths + archived), default=0)
    if max_path > PATH_LIMIT:
        raise OSError(f"anticipated path length {max_path} exceeds conservative limit {PATH_LIMIT}")
    d1_signature = str(__import__("inspect").signature(d1._evaluate_d1))
    return {
        "schema_version": "sortct_frozen_interventions_v1",
        "created_at_utc": _utc_now(),
        "run_root": str(root),
        "mode": "read_only_preview",
        "design": {
            "scenario": SCENARIO, "depart_scale": DEPART_SCALE,
            "traffic_split": EVAL_SPLIT, "deterministic": True,
            "seed_start": SEED_START, "seed_end_inclusive": SEED_START + FORMAL_EPISODES - 1,
            "control_smokes": 2, "intervention_arms": list(INTERVENTION_ARMS),
            "episodes_per_intervention_arm": FORMAL_EPISODES,
            "total_max_episodes": 2 + len(INTERVENTION_ARMS) * FORMAL_EPISODES,
            "max_parallel_cpu_workers": MAX_PARALLEL,
            "no_learning_or_replay_updates": True,
        },
        "checkpoints": {
            ROUTEACT_METHOD: {"path": str((ROUTEACT_RUN / "final_model.zip").resolve()), "sha256": route_ref["checkpoint_sha256"]},
            CONFLICT_METHOD: {"path": str((CONFLICT_RUN / "final_model.zip").resolve()), "sha256": conflict_ref["checkpoint_sha256"]},
        },
        "reference_metrics": {
            ROUTEACT_METHOD: route_ref["evaluation_summary"],
            CONFLICT_METHOD: conflict_ref["evaluation_summary"],
        },
        "reference_eval_records": {
            ROUTEACT_METHOD: str((ROUTEACT_RUN / "evaluation_results.json").resolve()),
            CONFLICT_METHOD: str((CONFLICT_RUN / "evaluation_results.json").resolve()),
        },
        "b_shadow_gate": b_gate,
        "current_traffic_files": [{"path": str(Path(path).resolve()), "sha256": current_hashes[Path(path).name]} for path in current_files],
        "reference_traffic_hashes": ref_hashes,
        "method_configs": configs,
        "reward": reward,
        "evaluation_device": "cpu",
        "d1_evaluator_signature": d1_signature,
        "path_check": {"limit_chars": PATH_LIMIT, "maximum_anticipated_path_chars": max_path},
        "source_files_to_archive": len(common._source_candidates()) + 1,
        "execution_command_template": [sys.executable, str(SCRIPT), "--start", "--run-root", str(root)],
    }


def _state_fingerprint(model) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
        count += 1
    return {
        "policy_state_sha256": digest.hexdigest(),
        "state_tensor_count": count,
        "n_updates": int(getattr(model, "_n_updates", 0)),
    }


def _find_probe_encoder(model, probe_name: str):
    candidates = []
    for owner in (getattr(model, "actor", None), getattr(model, "critic", None), getattr(model, "policy", None)):
        candidate = getattr(owner, "features_extractor", None) if owner is not None else None
        if candidate is not None and all(candidate is not item for item in candidates):
            candidates.append(candidate)
    matches = [candidate for candidate in candidates if callable(getattr(candidate, "policy_shadow_probe", None))]
    if not matches:
        raise RuntimeError("loaded frozen policy has no policy_shadow_probe encoder")
    encoder = matches[0]
    applicability = getattr(encoder, "policy_shadow_probe_applicability", None)
    if callable(applicability):
        status = {item.get("probe_name"): item for item in applicability()}
        if not status.get(probe_name, {}).get("applicable", False):
            raise RuntimeError(f"probe {probe_name!r} is not applicable to loaded encoder")
    return encoder


def _install_predict_probe(model, encoder, probe_name: str) -> tuple[Callable[[], None], dict[str, int]]:
    original_predict = model.predict
    model_dict = getattr(model, "__dict__", {})
    had_instance_predict = "predict" in model_dict
    old_instance_predict = model_dict.get("predict")
    call_count = {"predict_calls": 0}

    def wrapped_predict(*args, **kwargs):
        call_count["predict_calls"] += 1
        with encoder.policy_shadow_probe(probe_name):
            return original_predict(*args, **kwargs)

    model.predict = wrapped_predict

    def restore() -> None:
        if had_instance_predict:
            model.predict = old_instance_predict
        else:
            try:
                delattr(model, "predict")
            except AttributeError:
                pass

    return restore, call_count


def _install_model_capture(d1, probe_name: str | None):
    from algos.sb3_torch.sac import SceneRepresentationSAC

    cls = SceneRepresentationSAC
    had_local_load = "load" in cls.__dict__
    old_local_load = cls.__dict__.get("load")
    original_load = cls.load
    loaded: list[Any] = []
    fingerprints: list[dict[str, Any]] = []
    probe_calls: list[dict[str, int]] = []
    restore_predictors: list[Callable[[], None]] = []

    def capture(_class, *args, **kwargs):
        model = original_load(*args, **kwargs)
        loaded.append(model)
        fingerprints.append({"phase": "before_eval", **_state_fingerprint(model)})
        if probe_name is not None:
            encoder = _find_probe_encoder(model, probe_name)
            restore_predict, calls = _install_predict_probe(model, encoder, probe_name)
            restore_predictors.append(restore_predict)
            probe_calls.append(calls)
        return model

    cls.load = classmethod(capture)

    def restore() -> None:
        for restore_predict in reversed(restore_predictors):
            restore_predict()
        if had_local_load:
            setattr(cls, "load", old_local_load)
        else:
            try:
                delattr(cls, "load")
            except AttributeError:
                pass

    return loaded, fingerprints, probe_calls, restore


def _float_equal(left: Any, right: Any, *, abs_tol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return left == right


def _control_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, Any]]:
    differences = []
    for key in CONTROL_FIELDS:
        left, right = expected.get(key), actual.get(key)
        if key in ("episode_return", "raw_episode_return", *REWARD_KEYS):
            same = _float_equal(left, right)
        else:
            same = left == right
        if not same:
            differences.append({"field": key, "expected": left, "actual": right})
    return differences


def _pair_records(reference_records: list[dict[str, Any]], candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    ref = {_seed(row): row for row in reference_records}
    candidate = {_seed(row): row for row in candidate_records}
    if None in ref or None in candidate or set(ref) != set(candidate):
        raise RuntimeError("reference and intervention eval seeds differ")
    matrix: dict[str, dict[str, int]] = {
        outcome: {target: 0 for target in ("success", "collision", "timeout", "off_route", "unknown")}
        for outcome in ("success", "collision", "timeout", "off_route", "unknown")
    }
    variant_mismatches = []
    rows = []
    for seed in sorted(ref):
        left, right = ref[seed], candidate[seed]
        old_outcome, new_outcome = _outcome(left), _outcome(right)
        matrix.setdefault(old_outcome, {}).setdefault(new_outcome, 0)
        matrix[old_outcome][new_outcome] += 1
        same_variant = left.get("traffic_variant") == right.get("traffic_variant")
        if not same_variant:
            variant_mismatches.append(seed)
        rows.append({
            "seed": seed,
            "reference_traffic_variant": left.get("traffic_variant"),
            "candidate_traffic_variant": right.get("traffic_variant"),
            "traffic_variant_match": same_variant,
            "reference_outcome": old_outcome,
            "candidate_outcome": new_outcome,
        })
    return {
        "orientation": "reference row -> intervention column",
        "paired_count": len(rows),
        "traffic_variant_mismatch_seeds": variant_mismatches,
        "matrix": matrix,
        "pairs": rows,
    }


def _archive(run_root: Path, common, base, effective_files: tuple[Path, ...]) -> dict[str, Any]:
    archive = common.archive_sources(run_root, effective_files)
    archive_root = run_root / "source_archive"
    for source, subdir in (
        (SCRIPT, "runner"),
        (FAST_DEVELOPER / "analysis" / "sortct_frozen_intervention_protocol_20261003.md", "research"),
    ):
        if source.is_file():
            common._archive_file(source, archive_root / subdir / source.name, archive["files"])
    reference_evidence = []
    for method, run in ((ROUTEACT_METHOD, ROUTEACT_RUN), (CONFLICT_METHOD, CONFLICT_RUN)):
        for name in ("arguments.json", "training_complete.json", "evaluation_results.json", "diagnostics/eval/manifest.json"):
            source = run / name
            if source.is_file():
                destination = archive_root / "reference_evidence" / method / Path(name)
                common._archive_file(source, destination, archive["files"])
                reference_evidence.append({"source": str(source.resolve()), "sha256": _sha256(source), "archived_as": str(destination.relative_to(run_root))})
    archive["source_file_count"] = len(archive["files"])
    archive["reference_evidence"] = reference_evidence
    archive["generated_at_utc"] = _utc_now()
    common.write_json(archive_root / "manifest.json", archive)
    return archive


def _phase_dir(run_root: Path, arm: str) -> Path:
    spec = ARM_SPEC[arm]
    return run_root / spec["phase"] / ARM_DIR[arm]


def _worker_main(args) -> int:
    arm = str(args.worker_arm)
    if arm not in ARM_SPEC:
        raise ValueError(f"unknown worker arm {arm!r}")
    spec = ARM_SPEC[arm]
    run_root = Path(args.run_root).resolve()
    arm_dir = _phase_dir(run_root, arm)
    arm_dir.mkdir(parents=True, exist_ok=False)
    status_path = arm_dir / "worker_status.json"
    common, base, d1, _evaluation = _load_runtime()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = ROUTEACT_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT
    effective_files = tuple(base._traffic_paths_for_scale(DEPART_SCALE))
    seed_start = int(args.seed_start)
    episode_count = int(args.episode_count)
    if not effective_files or len(effective_files) != 30:
        raise RuntimeError(f"expected 30 effective sorted templates, got {len(effective_files)}")
    # Child worker always evaluates the full original episode horizon; only
    # the number of sampled evaluation seeds changes in the control smoke.
    method = str(spec["method"])
    parent = d1.PARENT[method]
    base.SEED_START[parent] = seed_start
    d1.EVAL_EPISODES_TOTAL = episode_count
    d1.RESULT_ROOT = base.RESULT_ROOT

    checkpoint_run = spec["reference_run"]
    checkpoint_path = checkpoint_run / "final_model.zip"
    expected_checkpoint_sha = _reference_for(str(spec["checkpoint_method"]))[1]
    if _sha256(checkpoint_path) != expected_checkpoint_sha:
        raise RuntimeError(f"checkpoint changed before {arm} worker start")
    loaded, fingerprints, probe_calls, restore_capture = _install_model_capture(d1, spec["probe"])
    _write_json(status_path, {
        "status": "running", "phase": spec["phase"], "arm": arm,
        "method": method, "checkpoint_method": spec["checkpoint_method"],
        "checkpoint_sha256": expected_checkpoint_sha, "pid": os.getpid(),
        "episodes": episode_count, "seed_start": seed_start,
        "probe": spec["probe"], "device": "cpu", "started_at_utc": _utc_now(),
    })
    try:
        result = d1._evaluate_d1(
            method,
            arm_dir,
            checkpoint_path,
            smoke=False,
            behavior_diagnostics=True,
            policy_shadow_probes=False,
            policy_shadow_eval_max_per_episode=3,
            policy_shadow_critical_eval_sample=False,
        )
    except BaseException as exc:
        _write_json(status_path, {
            "status": "failed", "phase": spec["phase"], "arm": arm,
            "method": method, "pid": os.getpid(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "failed_at_utc": _utc_now(),
        })
        raise
    finally:
        restore_capture()
    if len(loaded) != 1 or len(fingerprints) != 1:
        raise RuntimeError(f"expected one loaded model/fingerprint; got {len(loaded)} / {len(fingerprints)}")
    after = {"phase": "after_eval", **_state_fingerprint(loaded[0])}
    fingerprints.append(after)
    before = fingerprints[0]
    if before["policy_state_sha256"] != after["policy_state_sha256"] or before["n_updates"] != after["n_updates"]:
        raise RuntimeError("frozen policy weights/buffers or update count changed during evaluation")
    if before["n_updates"] != 95_001:
        raise RuntimeError(f"loaded checkpoint update count is not 95001: {before['n_updates']}")
    if _sha256(checkpoint_path) != expected_checkpoint_sha:
        raise RuntimeError("frozen checkpoint file changed during evaluation")

    identity = result.get("identity") or {}
    if identity.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise RuntimeError("D1 evaluator identity does not match frozen checkpoint")
    if identity.get("device") not in (None, "cpu"):
        raise RuntimeError(f"frozen evaluation unexpectedly used device={identity.get('device')!r}")
    identity.update({
        "diagnostic_experiment": "sortct_frozen_interventions_20261003",
        "diagnostic_phase": spec["phase"],
        "diagnostic_arm": arm,
        "method": method,
        "checkpoint_method": spec["checkpoint_method"],
        "base_checkpoint_path": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": expected_checkpoint_sha,
        "eval_device": "cpu",
        "learning_or_replay_updates": 0,
        "resume": False,
        "intervention": spec["probe"] or ("route-action wrapper enabled" if arm == "a_control_smoke" else "no route-action wrapper" if arm == "a_remove_route_veto" else "unmodified conflict-time policy"),
        "policy_shadow_probe": spec["probe"],
        "policy_predict_probe_calls": probe_calls[0]["predict_calls"] if probe_calls else 0,
        "model_policy_fingerprint": fingerprints,
    })
    result["identity"] = identity
    _write_json(arm_dir / "evaluation_results.json", result)
    records = result.get("episode_records") or []
    expected_seeds = list(range(seed_start, seed_start + episode_count))
    observed_seeds = [_seed(row) for row in records]
    if len(records) != episode_count or observed_seeds != expected_seeds:
        raise RuntimeError(f"episode/seed mismatch: got {len(records)} rows; seeds={observed_seeds}")
    bad = [idx for idx, row in enumerate(records) if _outcome(row) not in ("success", "collision", "timeout", "off_route")]
    if bad:
        raise RuntimeError(f"nonexclusive outcomes at rows {bad[:10]}")
    if spec["probe"] is not None and (not probe_calls or probe_calls[0]["predict_calls"] <= 0):
        raise RuntimeError(f"intervention probe {spec['probe']} did not wrap any policy prediction")
    eval_summary_path = arm_dir / "diagnostics" / "eval" / "summary.json"
    diag_summary = _read_json(eval_summary_path) if eval_summary_path.is_file() else {}
    errors = int(diag_summary.get("diagnostic_error_count", 0))
    if errors:
        raise RuntimeError(f"behavior diagnostics reported {errors} errors")
    _write_json(status_path, {
        "status": "complete", "phase": spec["phase"], "arm": arm,
        "method": method, "checkpoint_method": spec["checkpoint_method"],
        "pid": os.getpid(), "episodes": len(records),
        "seed_start": seed_start, "seed_end_inclusive": seed_start + episode_count - 1,
        "checkpoint_sha256": expected_checkpoint_sha,
        "policy_fingerprint_unchanged": True, "updates_unchanged": True,
        "policy_predict_probe_calls": probe_calls[0]["predict_calls"] if probe_calls else 0,
        "diagnostic_error_count": errors, "completed_at_utc": _utc_now(),
    })
    return 0


def _compare_control_with_reference(control_arm_dir: Path, reference_run: Path, seed: int) -> dict[str, Any]:
    control = _read_json(control_arm_dir / "evaluation_results.json")
    reference = _read_json(reference_run / "evaluation_results.json")
    actual_records = control.get("episode_records") or []
    reference_records = reference.get("episode_records") or []
    actual = next((row for row in actual_records if _seed(row) == seed), None)
    expected = next((row for row in reference_records if _seed(row) == seed), None)
    if actual is None or expected is None:
        raise RuntimeError(f"control smoke missing seed {seed} in {reference_run}")
    mismatches = _control_mismatches(expected, actual)
    return {
        "reference_run": str(reference_run.resolve()),
        "seed": seed,
        "passed": not mismatches,
        "mismatches": mismatches,
        "expected_outcome": _outcome(expected),
        "actual_outcome": _outcome(actual),
        "expected_traffic_variant": expected.get("traffic_variant"),
        "actual_traffic_variant": actual.get("traffic_variant"),
    }


def _run_stage(run_root: Path, arms: tuple[str, ...], *, episodes: int, seed_start: int, stage: str) -> dict[str, Any]:
    import concurrent.futures
    status_lock = threading.Lock()
    status_path = run_root / "launcher_status.json"
    live = _read_json(status_path)
    live["stage"] = stage
    live["status"] = "running"
    _write_json(status_path, live)

    def launch(arm: str):
        arm_dir = _phase_dir(run_root, arm)
        arm_dir.parent.mkdir(parents=True, exist_ok=True)
        out_path = run_root / "launcher_logs" / f"{ARM_DIR[arm]}.stdout.log"
        err_path = run_root / "launcher_logs" / f"{ARM_DIR[arm]}.stderr.log"
        command = [
            sys.executable, "-u", str(SCRIPT), "--worker-arm", arm,
            "--run-root", str(run_root), "--episode-count", str(episodes),
            "--seed-start", str(seed_start),
        ]
        _write_json(run_root / "commands" / f"{ARM_DIR[arm]}.command.json", {
            "command": command, "cwd": str(PROJECT), "phase": stage,
            "stdout": str(out_path.resolve()), "stderr": str(err_path.resolve()),
        })
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with out_path.open("w", encoding="utf-8", newline="\n") as stdout, err_path.open("w", encoding="utf-8", newline="\n") as stderr:
            process = subprocess.Popen(
                command, cwd=str(PROJECT), env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=flags,
            )
            with status_lock:
                workers = live.setdefault("workers", {})
                workers[arm] = {
                    "pid": process.pid, "status": "running", "phase": stage,
                    "method": ARM_SPEC[arm]["method"],
                    "checkpoint_method": ARM_SPEC[arm]["checkpoint_method"],
                    "command": command, "stdout": str(out_path.resolve()),
                    "stderr": str(err_path.resolve()),
                }
                _write_json(status_path, live)
            code = process.wait()
        return arm, process.pid, code

    outcomes: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(launch, arm): arm for arm in arms}
        for future in concurrent.futures.as_completed(futures):
            arm, pid, code = future.result()
            outcomes[arm] = {"pid": pid, "exit_code": code, "status": "complete" if code == 0 else "failed"}
            with status_lock:
                live.setdefault("workers", {})[arm].update(outcomes[arm])
                if code != 0:
                    live["status"] = "failed"
                _write_json(status_path, live)
    return outcomes


def _pair_all_interventions(run_root: Path) -> dict[str, Any]:
    result = {
        "orientation": "official frozen evaluation row -> intervention column",
        "pairing_key": ["seed", "traffic_variant"],
        "seed_range": [SEED_START, SEED_START + FORMAL_EPISODES - 1],
        "arms": {},
    }
    for arm in INTERVENTION_ARMS:
        spec = ARM_SPEC[arm]
        reference = _read_json(spec["reference_run"] / "evaluation_results.json")
        candidate = _read_json(_phase_dir(run_root, arm) / "evaluation_results.json")
        first = SEED_START
        last = SEED_START + FORMAL_EPISODES
        reference_subset = [
            row for row in (reference.get("episode_records") or [])
            if first <= int(_seed(row) or -1) < last
        ]
        matrix = _pair_records(reference_subset, candidate.get("episode_records") or [])
        if matrix["paired_count"] != FORMAL_EPISODES or matrix["traffic_variant_mismatch_seeds"]:
            raise RuntimeError(f"{arm} failed same-seed/template pairing: {matrix['traffic_variant_mismatch_seeds']}")
        result["arms"][arm] = matrix
    return result


def _start(args) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    plan = _preflight(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    common, base, d1, _evaluation = _load_runtime()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = ROUTEACT_RUN.parents[0]
    d1.RESULT_ROOT = base.RESULT_ROOT
    effective = tuple(Path(p) for p in base._traffic_paths_for_scale(DEPART_SCALE))
    archive = _archive(run_root, common, base, effective)
    _write_json(run_root / "experiment_manifest.json", {
        **plan,
        "mode": "executed_frozen_evaluation",
        "status": "starting",
        "started_at_utc": _utc_now(),
        "supervisor_pid": os.getpid(),
        "source_archive_manifest": str((run_root / "source_archive" / "manifest.json").resolve()),
        "source_archive_file_count": archive["source_file_count"],
        "reference_checkpoint_files_not_copied": True,
        "intervention_scope": "frozen evaluations only; no learning/replay updates",
    })
    (run_root / "launcher_logs").mkdir(parents=True, exist_ok=False)
    (run_root / "commands").mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "launcher_status.json", {
        "status": "running", "stage": "control_smokes", "pid": os.getpid(),
        "started_at_utc": _utc_now(), "workers": {},
    })

    control_outcomes = _run_stage(run_root, CONTROL_ARMS, episodes=1, seed_start=SEED_START, stage="control_smokes")
    control_gate: dict[str, Any] = {"status": "passed", "checks": {}}
    for arm in CONTROL_ARMS:
        if control_outcomes[arm]["exit_code"] != 0:
            control_gate["status"] = "failed"
            control_gate["checks"][arm] = {"passed": False, "reason": "worker_failed"}
            break
        spec = ARM_SPEC[arm]
        check = _compare_control_with_reference(_phase_dir(run_root, arm), spec["reference_run"], SEED_START)
        control_gate["checks"][arm] = check
        if not check["passed"]:
            control_gate["status"] = "failed"
    _write_json(run_root / "control_gate.json", control_gate)
    status = _read_json(run_root / "launcher_status.json")
    status["control_gate"] = control_gate
    status["control_smoke_exit_codes"] = control_outcomes
    _write_json(run_root / "launcher_status.json", status)
    if control_gate["status"] != "passed":
        status.update({"status": "failed", "stage": "control_gate_failed", "completed_at_utc": _utc_now()})
        _write_json(run_root / "launcher_status.json", status)
        return 1

    intervention_outcomes = _run_stage(
        run_root, INTERVENTION_ARMS, episodes=FORMAL_EPISODES,
        seed_start=SEED_START, stage="interventions",
    )
    complete = all(item["exit_code"] == 0 for item in intervention_outcomes.values())
    paired = None
    if complete:
        paired = _pair_all_interventions(run_root)
        _write_json(run_root / "paired_outcomes.json", paired)
    status = _read_json(run_root / "launcher_status.json")
    status.update({
        "status": "complete" if complete else "failed",
        "stage": "complete" if complete else "interventions_failed",
        "intervention_exit_codes": intervention_outcomes,
        "paired_outcomes": str((run_root / "paired_outcomes.json").resolve()) if paired is not None else None,
        "completed_at_utc": _utc_now(),
    })
    _write_json(run_root / "launcher_status.json", status)
    manifest = _read_json(run_root / "experiment_manifest.json")
    manifest["status"] = status["status"]
    manifest["completed_at_utc"] = status["completed_at_utc"]
    _write_json(run_root / "experiment_manifest.json", manifest)
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="read-only preview; does not create files or run SUMO")
    parser.add_argument("--start", action="store_true", help="run two control smokes, then gated three-arm frozen eval")
    parser.add_argument("--worker-arm", choices=ALL_ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--episode-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=SEED_START, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_arm:
        if args.episode_count is None or args.episode_count <= 0:
            parser.error("internal worker requires positive --episode-count")
        return _worker_main(args)
    if args.start and args.dry_run:
        parser.error("choose either --dry-run or --start")
    if not args.start and not args.dry_run:
        parser.error("choose --dry-run to preview or --start to execute")
    try:
        plan = _preflight(args.run_root)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        return _start(args)
    except Exception as exc:
        print(f"preflight/runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
