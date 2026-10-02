"""Dry-run-first, bounded frozen ST-RT decision-window intervention runner.

This tool is intentionally separate from the training/evaluation entrypoints.
Importing it or running without ``--start`` does not construct a SUMO env.
It uses deterministic same-seed prefix replays; it does not restore simulator
state, train, alter replay data, or force background-vehicle trajectories.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
FAST_DEVELOPER = SCRIPT.parent
PROJECT = FAST_DEVELOPER.parent
WORKSPACE = PROJECT.parent.parent
METHOD = "sac_mlp_d1_st_rt"
SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
EVAL_SPLIT = "validation"
SEED_START = 10_000
REFERENCE_EVAL_RUN = WORKSPACE / "runs" / "srtact_1002" / "control"
REFERENCE_TRAIN_RUN = (
    WORKSPACE / "runs" / "d0929_100k_diag"
    / f"{METHOD}__{SCENARIO}_depart4p0"
)
CHECKPOINT_PATH = REFERENCE_TRAIN_RUN / "final_model.zip"
CHECKPOINT_SHA256 = "070b05b5ad2b60ce6fc4e47ee4508f0c1d2f1dd45579a0462d50e7d6a93f6262"
PROTOCOL_PATH = FAST_DEVELOPER / "analysis" / "decision_window_intervention_protocol_20261003.md"
DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "strt_window_1003"
MAX_COLLISION_CASES = 8
MAX_SUCCESS_CASES = 4
IDENTITY_CASES_PER_OUTCOME = 1
MAX_RAW_STEPS_PER_EPISODE = 600
MAX_CONTROL_EPISODES = 12 + 2 + 12 * 3 * 2
MAX_CONTROL_RAW_STEPS = MAX_CONTROL_EPISODES * MAX_RAW_STEPS_PER_EPISODE
INTERVENTION_TARGET_SPEEDS_MPS = (0.0, 6.0)
INTERVENTION_DECISIONS = 3


class PrefixMismatch(RuntimeError):
    """A replay no longer matches its baseline before the intervention anchor."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy-like observation objects into deterministic JSON values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return {"__float_hex__": value.hex()}
        return {"__float_special__": repr(value)}
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        data = value.tolist()
        return {
            "__array__": _jsonable(data),
            "dtype": str(getattr(value, "dtype", "unknown")),
            "shape": list(getattr(value, "shape", ())),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported value for deterministic hashing: {type(value).__name__}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def physical_snapshot_payload(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep all active actors from the raw behavior snapshot, never hidden routes."""
    if not isinstance(snapshot, dict):
        return None
    actor_fields = (
        "id", "key", "kind", "position", "velocity", "heading", "speed",
        "length", "width", "road_id", "lane_id", "lane_position",
        "state_source", "state_time",
    )

    def actor(value):
        if not isinstance(value, dict):
            return None
        return {name: value.get(name) for name in actor_fields}

    raw_vehicles = snapshot.get("vehicles")
    if not isinstance(raw_vehicles, list):
        return None
    vehicles = [actor(item) for item in raw_vehicles]
    vehicles = [item for item in vehicles if item is not None]
    vehicles.sort(key=lambda item: (str(item.get("key") or ""), str(item.get("id") or "")))
    sim_time = snapshot.get("sim_time")
    ego = actor(snapshot.get("ego"))
    errors = list(snapshot.get("errors") or [])

    def finite_scalar(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    actor_state_errors = []

    def actor_error_reasons(value, label):
        if not isinstance(value, dict):
            return [f"{label}:missing_actor_payload"]
        reasons = []
        for name in ("id", "key", "kind", "road_id", "lane_id"):
            if value.get(name) is None:
                reasons.append(f"{label}:{name}_missing")
        for name in ("position", "velocity"):
            pair = value.get(name)
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                reasons.append(f"{label}:{name}_missing_or_invalid")
            elif not (finite_scalar(pair[0]) and finite_scalar(pair[1])):
                reasons.append(f"{label}:{name}_nonfinite")
        for name in ("heading", "speed", "length", "width"):
            if value.get(name) is None:
                reasons.append(f"{label}:{name}_missing")
            elif not finite_scalar(value.get(name)):
                reasons.append(f"{label}:{name}_nonfinite")
        if value.get("lane_position") is not None and not finite_scalar(value.get("lane_position")):
            reasons.append(f"{label}:lane_position_nonfinite")
        if value.get("state_source") != "current":
            reasons.append(f"{label}:state_source_not_current")
        if sim_time is None or value.get("state_time") is None:
            reasons.append(f"{label}:state_time_missing")
        elif value.get("state_time") != sim_time:
            reasons.append(f"{label}:state_time_mismatch")
        return reasons

    actor_state_errors.extend(actor_error_reasons(ego, "ego"))
    for index, value in enumerate(vehicles):
        label = str(value.get("key") or value.get("id") or f"vehicle_index_{index}")
        actor_state_errors.extend(actor_error_reasons(value, label))
    actors_complete = not actor_state_errors
    return {
        "raw_step": snapshot.get("raw_step"),
        "sim_time": snapshot.get("sim_time"),
        "ego_state_source": snapshot.get("ego_state_source"),
        "ego_state_time": snapshot.get("ego_state_time"),
        "snapshot_scope": "pre_recorder_filter_active_actor_snapshot",
        "physical_state_complete": bool(actors_complete and not errors),
        "physical_state_incomplete_reasons": (
            (["snapshot_errors"] if errors else [])
            + actor_state_errors
        ),
        "physical_snapshot_error_details": errors,
        "physical_actor_state_errors": actor_state_errors,
        "physical_actor_count": len(vehicles) + int(ego is not None),
        "ego": ego,
        "vehicles": vehicles,
        "speed_control": snapshot.get("speed_control"),
        "lane_control": snapshot.get("lane_control"),
    }


def verify_physical_snapshot_membership(
    snapshot: dict[str, Any] | None,
    *,
    current_vehicle_ids: list[str] | None,
    current_person_ids: list[str] | None,
    current_raw_step: int | None,
    query_errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Require raw-snapshot IDs and raw time to match the live, read-only state."""
    if snapshot is None:
        return None
    result = dict(snapshot)
    reasons = list(result.get("physical_state_incomplete_reasons") or [])
    ego = result.get("ego") or {}
    vehicles = result.get("vehicles") or []
    captured_vehicle_ids = {
        str(item.get("id")) for item in vehicles
        if isinstance(item, dict) and item.get("kind") == "vehicle" and item.get("id") is not None
    }
    if ego.get("kind") == "vehicle" and ego.get("id") is not None:
        captured_vehicle_ids.add(str(ego["id"]))
    membership = {
        "vehicle_id_set_matches": None,
        "missing_vehicle_ids": None,
        "unexpected_vehicle_ids": None,
        "person_id_set_matches": None,
        "missing_person_ids": None,
        "unexpected_person_ids": None,
        "raw_step_matches": None,
        "query_errors": list(query_errors or []),
    }
    if current_vehicle_ids is None:
        reasons.append("vehicle_id_list_unavailable")
    else:
        current = {str(item) for item in current_vehicle_ids}
        missing = sorted(current - captured_vehicle_ids)
        unexpected = sorted(captured_vehicle_ids - current)
        membership.update(
            vehicle_id_set_matches=not missing and not unexpected,
            missing_vehicle_ids=missing,
            unexpected_vehicle_ids=unexpected,
        )
        if missing or unexpected:
            reasons.append("vehicle_id_set_mismatch")
    captured_person_ids = {
        str(item.get("id")) for item in vehicles
        if isinstance(item, dict) and item.get("kind") == "person" and item.get("id") is not None
    }
    if current_person_ids is None:
        if captured_person_ids:
            reasons.append("person_id_list_unavailable")
    else:
        current = {str(item) for item in current_person_ids}
        missing = sorted(current - captured_person_ids)
        unexpected = sorted(captured_person_ids - current)
        membership.update(
            person_id_set_matches=not missing and not unexpected,
            missing_person_ids=missing,
            unexpected_person_ids=unexpected,
        )
        if missing or unexpected:
            reasons.append("person_id_set_mismatch")
    snapshot_raw_step = result.get("raw_step")
    if current_raw_step is None:
        reasons.append("live_raw_step_unavailable")
    else:
        membership["raw_step_matches"] = int(current_raw_step) == snapshot_raw_step
        if not membership["raw_step_matches"]:
            reasons.append("stale_raw_snapshot")
    if membership["query_errors"]:
        reasons.extend(f"id_query_failed:{error}" for error in membership["query_errors"])
    result["physical_membership"] = membership
    result["physical_state_incomplete_reasons"] = sorted(set(reasons))
    result["physical_state_complete"] = not result["physical_state_incomplete_reasons"]
    return result


def physical_fingerprint_payload(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hash only physical identity/state, excluding controls and diagnostics."""
    if snapshot is None:
        return None
    keys = (
        "raw_step", "sim_time", "ego_state_source", "ego_state_time",
        "physical_actor_count", "ego", "vehicles",
    )
    return {key: snapshot.get(key) for key in keys}


def choose_cases(episode_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select ascending-seed outcome-stratified cases from the frozen 100-eval."""
    collisions = sorted(
        (row for row in episode_records if bool(row.get("collision"))),
        key=lambda row: int(row["seed"]),
    )[:MAX_COLLISION_CASES]
    successes = sorted(
        (row for row in episode_records if bool(row.get("success"))),
        key=lambda row: int(row["seed"]),
    )[:MAX_SUCCESS_CASES]
    if len(collisions) != MAX_COLLISION_CASES or len(successes) != MAX_SUCCESS_CASES:
        raise ValueError(
            f"reference lacks required strata: collisions={len(collisions)}, successes={len(successes)}"
        )
    chosen = [
        {"seed": int(row["seed"]), "outcome": "collision", "raw_steps": int(row["raw_steps"]),
         "decision_steps": int(row["decision_steps"])}
        for row in collisions
    ] + [
        {"seed": int(row["seed"]), "outcome": "success", "raw_steps": int(row["raw_steps"]),
         "decision_steps": int(row["decision_steps"])}
        for row in successes
    ]
    return chosen


def plan_anchors(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Use baseline pre-action roads to define and merge protocol anchors."""
    if not trace:
        return {"phases": {}, "unique": []}
    first_internal = None
    for index, row in enumerate(trace):
        road = row.get("pre_ego_road_id")
        if isinstance(road, str) and road.startswith(":"):
            first_internal = index
            break
    phase_indices: dict[str, int | None] = {
        "three_decisions_before_first_internal": (
            max(0, first_internal - 3) if first_internal is not None else None
        ),
        "first_internal_pre_action": first_internal,
        "three_decisions_before_terminal": max(0, len(trace) - 3),
    }
    by_index: dict[int, list[str]] = {}
    for phase, index in phase_indices.items():
        if index is not None and 0 <= index < len(trace):
            by_index.setdefault(index, []).append(phase)
    unique = [
        {"decision_index": index, "decision": index + 1, "phases": phases,
         "pre_obs_raw_step": trace[index].get("pre_obs_raw_step"),
         "sim_time": trace[index].get("pre_sim_time"),
         "time_to_terminal_decisions": len(trace) - index}
        for index, phases in sorted(by_index.items())
    ]
    return {"phases": phase_indices, "unique": unique}


def action_for_target_speed(target_speed_mps: float) -> float:
    if not 0.0 <= float(target_speed_mps) <= 10.0:
        raise ValueError("target speed must lie in the environment's [0,10] m/s range")
    return float(target_speed_mps) / 5.0 - 1.0


def replace_speed_action(action: Any, target_speed_mps: float):
    """Change only action coordinate 0, preserving the policy lateral channel."""
    import numpy as np

    array = np.asarray(action).copy()
    if array.size < 2:
        raise ValueError(f"expected speed+lateral action, got shape {array.shape}")
    if array.ndim == 1:
        array[0] = action_for_target_speed(target_speed_mps)
    else:
        array[..., 0] = action_for_target_speed(target_speed_mps)
    return array


def prefix_mismatch(
    baseline_trace: list[dict[str, Any]],
    replay_trace: list[dict[str, Any]],
    anchor_index: int,
) -> dict[str, Any] | None:
    """Require observation, action, and complete physical hashes through anchor."""
    if anchor_index < 0:
        return {"reason": "negative_anchor", "decision_index": anchor_index}
    for index in range(anchor_index + 1):
        if index >= len(baseline_trace) or index >= len(replay_trace):
            return {"reason": "prefix_length_mismatch", "decision_index": index}
        left, right = baseline_trace[index], replay_trace[index]
        for field in ("observation_sha256", "policy_action_sha256"):
            if left.get(field) != right.get(field):
                return {"reason": f"{field}_mismatch", "decision_index": index}
        if left.get("physical_state_complete") is not True or right.get("physical_state_complete") is not True:
            return {"reason": "prefix_full_physical_state_unavailable", "decision_index": index}
        left_hash = left.get("physical_state_sha256")
        right_hash = right.get("physical_state_sha256")
        if left_hash is None or right_hash is None:
            return {"reason": "prefix_physical_state_missing", "decision_index": index}
        if left_hash != right_hash:
            return {
                "reason": "prefix_physical_state_mismatch",
                "decision_index": index,
                "physical_state_diff": _physical_diff_summary(
                    left.get("physical_state"), right.get("physical_state")
                ),
            }
        if left.get("physical_state_complete") is not True or right.get("physical_state_complete") is not True:
            return {"reason": "prefix_full_physical_state_unavailable", "decision_index": index}
        left_hash = left.get("physical_state_sha256")
        right_hash = right.get("physical_state_sha256")
        if left_hash is None or right_hash is None:
            return {"reason": "prefix_physical_state_missing", "decision_index": index}
        if left_hash != right_hash:
            return {
                "reason": "prefix_physical_state_mismatch",
                "decision_index": index,
                "physical_state_diff": _physical_diff_summary(
                    left.get("physical_state"), right.get("physical_state")
                ),
            }
    if anchor_index >= len(baseline_trace) or anchor_index >= len(replay_trace):
        return {"reason": "anchor_unavailable", "decision_index": anchor_index}
    if len(baseline_trace) != len(replay_trace):
        return {"reason": "trace_length_mismatch", "baseline": len(baseline_trace), "replay": len(replay_trace)}
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_reference() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    actual_sha = sha256_file(CHECKPOINT_PATH)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"ST-RT checkpoint SHA mismatch: {actual_sha}")
    evaluation_path = REFERENCE_EVAL_RUN / "evaluation_results.json"
    result = _read_json(evaluation_path)
    identity = result.get("identity") or {}
    rows = result.get("episode_records") or []
    expected_seeds = list(range(SEED_START, SEED_START + 100))
    if identity.get("method") != METHOD or identity.get("scenario") != SCENARIO:
        raise RuntimeError("reference evaluation is not the expected frozen sorted ST-RT run")
    if float(identity.get("depart_scale", -1)) != DEPART_SCALE:
        raise RuntimeError("reference evaluation depart scale differs")
    if identity.get("checkpoint_sha256") != actual_sha:
        raise RuntimeError("reference evaluation checkpoint identity differs from checkpoint file")
    if len(rows) != 100 or [int(row.get("seed", -1)) for row in rows] != expected_seeds:
        raise RuntimeError("reference evaluation must contain each validation seed 10000..10099 in order")
    if identity.get("eval_traffic_split") != EVAL_SPLIT or identity.get("smoke") is not False:
        raise RuntimeError("reference evaluation split/smoke identity differs")
    decisions_path = REFERENCE_EVAL_RUN / "diagnostics" / "eval" / "decisions.jsonl.gz"
    episodes_path = REFERENCE_EVAL_RUN / "diagnostics" / "eval" / "episodes.jsonl"
    if not decisions_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("reference decision/behavior streams are missing")
    chosen = choose_cases(rows)
    identity_seeds = [
        next(item["seed"] for item in chosen if item["outcome"] == "collision"),
        next(item["seed"] for item in chosen if item["outcome"] == "success"),
    ]
    return rows, chosen, {
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_sha256": actual_sha,
        "evaluation_results": str(evaluation_path.resolve()),
        "decision_stream": str(decisions_path.resolve()),
        "episode_stream": str(episodes_path.resolve()),
        "identity_replay_seeds": identity_seeds,
        "summary": result.get("summary"),
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _physical_snapshot(recorder) -> dict[str, Any] | None:
    episode = getattr(recorder, "episode", None)
    if not isinstance(episode, dict):
        return None
    # The recorder's last_snapshot is deliberately filtered/capped to local
    # diagnostic neighbors. The on_raw_step observer below captures the raw
    # snapshot before that filtering. Reset's initial_snapshot is also raw.
    snapshot = episode.get("initial_snapshot")
    result = physical_snapshot_payload(snapshot)
    if result is not None:
        result["snapshot_capture_source"] = "reset_initial_snapshot"
    return result


def _snapshot_from_environment_cache(env) -> dict[str, Any] | None:
    """Read the reset/current physical cache without advancing SUMO.

    ``_behavior_snapshot`` only assembles already-recorded actor state plus
    read-only simulation time/delta. Preserve its diagnostic bookkeeping so a
    fallback snapshot cannot consume errors or change the policy's telemetry
    state.
    """
    try:
        base = env.unwrapped
        snapshot_fn = getattr(base, "_behavior_snapshot", None)
        if not callable(snapshot_fn):
            return None
        pending = getattr(base, "_behavior_pending_errors", None)
        error_keys = getattr(base, "_behavior_error_keys", None)
        old_pending = list(pending) if isinstance(pending, list) else None
        old_error_keys = set(error_keys) if isinstance(error_keys, set) else None
        old_step_seconds = getattr(base, "_behavior_step_seconds", None)
        events = tuple(getattr(base, "_last_events", (False, False, False, False)))
        try:
            snapshot = snapshot_fn(events)
        finally:
            if old_pending is not None:
                pending[:] = old_pending
            if old_error_keys is not None:
                error_keys.clear()
                error_keys.update(old_error_keys)
            if hasattr(base, "_behavior_step_seconds"):
                base._behavior_step_seconds = old_step_seconds
        result = physical_snapshot_payload(snapshot)
        if result is not None:
            result["snapshot_capture_source"] = "environment_behavior_snapshot_cache"
        return result
    except Exception as exc:
        return {
            "physical_state_complete": False,
            "physical_state_incomplete_reasons": [
                f"behavior_snapshot_cache_read_failed:{type(exc).__name__}"
            ],
            "snapshot_capture_source": "environment_behavior_snapshot_cache",
        }


def _physical_diff_summary(expected: dict[str, Any] | None, actual: dict[str, Any] | None) -> dict[str, Any]:
    """Compactly identify which physical fields differ at a failed prefix."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {"available": False}
    actor_fields = (
        "position", "velocity", "heading", "speed", "road_id", "lane_id",
        "lane_position", "state_time", "state_source",
    )
    expected_actors = {"ego": expected.get("ego")}
    actual_actors = {"ego": actual.get("ego")}
    for actor in expected.get("vehicles", []):
        if isinstance(actor, dict):
            expected_actors[str(actor.get("key") or actor.get("id"))] = actor
    for actor in actual.get("vehicles", []):
        if isinstance(actor, dict):
            actual_actors[str(actor.get("key") or actor.get("id"))] = actor
    expected_ids = set(expected_actors)
    actual_ids = set(actual_actors)
    changed = []
    for actor_id in sorted(expected_ids & actual_ids):
        left = expected_actors[actor_id] or {}
        right = actual_actors[actor_id] or {}
        fields = [field for field in actor_fields if left.get(field) != right.get(field)]
        if fields:
            changed.append({"actor": actor_id, "fields": fields})
    return {
        "available": True,
        "expected_raw_step": expected.get("raw_step"),
        "actual_raw_step": actual.get("raw_step"),
        "expected_sim_time": expected.get("sim_time"),
        "actual_sim_time": actual.get("sim_time"),
        "expected_complete": expected.get("physical_state_complete"),
        "actual_complete": actual.get("physical_state_complete"),
        "missing_actor_keys": sorted(expected_ids - actual_ids),
        "unexpected_actor_keys": sorted(actual_ids - expected_ids),
        "changed_actor_count": len(changed),
        "changed_actors_preview": changed[:20],
    }


def _validate_intervention_anchor(predictor, outcome: str | None = None) -> None:
    """Reject a replay that terminated before the requested intervention ran."""
    if (
        predictor.target_speed_mps is None
        or predictor.anchor_index is None
        or predictor.prefix_error is not None
    ):
        return
    if len(predictor.trace) <= int(predictor.anchor_index):
        predictor.prefix_error = {
            "reason": "anchor_not_reached",
            "anchor_index": int(predictor.anchor_index),
            "trace_decisions": len(predictor.trace),
            "replay_outcome": outcome,
        }
    elif predictor.intervention_applied_decisions < 1:
        predictor.prefix_error = {
            "reason": "intervention_not_applied",
            "anchor_index": int(predictor.anchor_index),
            "trace_decisions": len(predictor.trace),
        }


def _verified_physical_snapshot(recorder, env, raw_snapshot=None) -> dict[str, Any] | None:
    observer_snapshot_supplied = raw_snapshot is not None
    snapshot = raw_snapshot if observer_snapshot_supplied else _physical_snapshot(recorder)
    if snapshot is None:
        snapshot = {
            "raw_step": None,
            "sim_time": None,
            "ego": None,
            "vehicles": [],
            "physical_state_complete": False,
            "physical_state_incomplete_reasons": ["reset_initial_snapshot_unavailable"],
            "snapshot_scope": "unavailable",
            "snapshot_capture_source": "unavailable",
        }
    errors = []
    vehicles = persons = current_raw_step = None
    try:
        base = env.unwrapped
        connection = getattr(base, "_connection", None)
        if connection is None:
            errors.append("connection_unavailable")
        else:
            vehicles = [str(value) for value in connection.vehicle.getIDList()]
            if bool(getattr(base.specification, "include_pedestrians", False)):
                persons = [str(value) for value in connection.person.getIDList()]
            else:
                persons = []
        raw_value = getattr(base, "_raw_steps", None)
        current_raw_step = int(raw_value) if raw_value is not None else None
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    verified = verify_physical_snapshot_membership(
        snapshot,
        current_vehicle_ids=vehicles,
        current_person_ids=persons,
        current_raw_step=current_raw_step,
        query_errors=errors,
    )
    if verified.get("physical_state_complete") or observer_snapshot_supplied:
        return verified
    # Reset can precede the first raw-step observer callback. Refresh only from
    # the environment's existing current actor cache; do not step or enumerate
    # routes. Keep the original ID-list query and raw counter as the same-tick
    # membership gate.
    refreshed = _snapshot_from_environment_cache(env)
    if refreshed is None:
        return verified
    refreshed_verified = verify_physical_snapshot_membership(
        refreshed,
        current_vehicle_ids=vehicles,
        current_person_ids=persons,
        current_raw_step=current_raw_step,
        query_errors=errors,
    )
    refreshed_verified["initial_snapshot_incomplete_reasons"] = verified.get(
        "physical_state_incomplete_reasons"
    )
    return refreshed_verified


def _install_raw_snapshot_observer(recorder):
    """Capture the full raw snapshot before the recorder filters/caps neighbors."""
    original = recorder.on_raw_step
    holder: dict[str, Any] = {"predictor": None}

    def observed_on_raw_step(snapshot):
        predictor = holder.get("predictor")
        if predictor is not None:
            predictor.latest_full_physical_snapshot = physical_snapshot_payload(snapshot)
            if predictor.latest_full_physical_snapshot is not None:
                predictor.latest_full_physical_snapshot["snapshot_capture_source"] = "raw_step_observer"
        return original(snapshot)

    recorder.on_raw_step = observed_on_raw_step

    def restore():
        recorder.on_raw_step = original

    return holder, restore


class _WindowPredictor:
    def __init__(
        self,
        model,
        recorder,
        *,
        env=None,
        baseline_trace=None,
        anchor_index=None,
        target_speed_mps=None,
        intervention_decisions=INTERVENTION_DECISIONS,
        verify_entire_trace=False,
    ):
        self.model = model
        self.recorder = recorder
        self.env = env
        self.baseline_trace = baseline_trace
        self.anchor_index = anchor_index
        self.target_speed_mps = target_speed_mps
        self.intervention_decisions = int(intervention_decisions)
        self.verify_entire_trace = bool(verify_entire_trace)
        self.trace: list[dict[str, Any]] = []
        self.prefix_error: dict[str, Any] | None = None
        self.intervention_applied_decisions = 0
        self.latest_full_physical_snapshot: dict[str, Any] | None = None
        self.reset_raw_observer_callback_count: int | None = None
        self._raw_count_before_reset = int(getattr(recorder, "raw_count", 0))

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, observation, *args, **kwargs):
        import numpy as np

        policy_action, state = self.model.predict(observation, *args, **kwargs)
        index = len(self.trace)
        raw_physical = self.latest_full_physical_snapshot or _physical_snapshot(self.recorder)
        physical = _verified_physical_snapshot(self.recorder, self.env, raw_physical)
        episode = getattr(self.recorder, "episode", None)
        if self.reset_raw_observer_callback_count is None:
            self.reset_raw_observer_callback_count = max(
                0, int(getattr(self.recorder, "raw_count", 0)) - self._raw_count_before_reset
            )
        ego = (physical or {}).get("ego") or {}
        action_array = np.asarray(policy_action).copy()
        action_values = action_array.reshape(-1).tolist()
        item = {
            "decision_index": index,
            "decision": index + 1,
            "pre_obs_raw_step": (physical or {}).get("raw_step"),
            "pre_sim_time": (physical or {}).get("sim_time"),
            "pre_ego_road_id": ego.get("road_id"),
            "pre_ego_lane_id": ego.get("lane_id"),
            "observation_sha256": stable_hash(observation),
            "policy_action_sha256": stable_hash(action_array),
            "physical_state_sha256": stable_hash(physical_fingerprint_payload(physical)) if physical is not None else None,
            "physical_state_complete": (physical or {}).get("physical_state_complete") is True,
            "physical_state_scope": (physical or {}).get("snapshot_scope"),
            "physical_state_capture_source": (physical or {}).get("snapshot_capture_source"),
            "physical_actor_count": (physical or {}).get("physical_actor_count"),
            "physical_state_incomplete_reasons": (physical or {}).get("physical_state_incomplete_reasons"),
            "physical_actor_state_errors": (physical or {}).get("physical_actor_state_errors"),
            "physical_membership": (physical or {}).get("physical_membership"),
            "physical_state_available": physical is not None and (physical or {}).get("ego") is not None,
            "policy_action": action_values,
            "executed_action": action_values,
            "target_speed_intervention_mps": None,
            "action_speed_changed": False,
            "step_info": None,
        }
        if self.baseline_trace is not None and self.prefix_error is None:
            verify_to = (
                len(self.baseline_trace) - 1
                if self.verify_entire_trace
                else self.anchor_index
            )
            if verify_to is not None and index <= verify_to:
                if index >= len(self.baseline_trace):
                    self.prefix_error = {"reason": "prefix_length_mismatch", "decision_index": index}
                else:
                    expected = self.baseline_trace[index]
                    for field in ("observation_sha256", "policy_action_sha256"):
                        if expected.get(field) != item.get(field):
                            self.prefix_error = {"reason": f"{field}_mismatch", "decision_index": index}
                            break
                    check_physical = self.verify_entire_trace or (
                        self.anchor_index is not None and index <= self.anchor_index
                    )
                    if self.prefix_error is None and check_physical:
                        if expected.get("physical_state_complete") is not True or item.get("physical_state_complete") is not True:
                            self.prefix_error = {
                                "reason": "full_physical_state_unavailable",
                                "decision_index": index,
                                "expected_reasons": expected.get("physical_state_incomplete_reasons"),
                                "actual_reasons": item.get("physical_state_incomplete_reasons"),
                            }
                        elif expected.get("physical_state_sha256") != item.get("physical_state_sha256"):
                            self.prefix_error = {
                                "reason": "prefix_physical_state_mismatch",
                                "decision_index": index,
                                "expected_physical_state_sha256": expected.get("physical_state_sha256"),
                                "actual_physical_state_sha256": item.get("physical_state_sha256"),
                                "physical_state_diff": _physical_diff_summary(
                                    expected.get("physical_state"), physical
                                ),
                            }
        if (
            self.target_speed_mps is not None
            and self.anchor_index is not None
            and self.prefix_error is None
            and self.anchor_index <= index < self.anchor_index + self.intervention_decisions
        ):
            executed = replace_speed_action(action_array, self.target_speed_mps)
            item["executed_action"] = executed.reshape(-1).tolist()
            item["target_speed_intervention_mps"] = float(self.target_speed_mps)
            item["action_speed_changed"] = not np.array_equal(
                np.asarray(item["policy_action"]), np.asarray(item["executed_action"])
            )
            self.intervention_applied_decisions += 1
            action_array = executed
        self.trace.append(item)
        if self.baseline_trace is None or index == self.anchor_index:
            # Keep a complete baseline physical record and each intervention
            # anchor. On mismatch, the compact diff above preserves the fields
            # needed to identify timing, membership, or motion divergence.
            item["physical_state"] = physical
        return action_array, state

    def after_step(self, reward, terminated, truncated, info):
        if not self.trace:
            return
        info = dict(info or {})
        row = self.trace[-1]
        row["step_info"] = {
            key: info.get(key) for key in (
                "raw_simulation_steps", "raw_steps_executed", "actual_speed_mps",
                "target_speed", "effective_target_speed", "lane_command_requested",
                "lane_change_applied", "current_road_id", "current_lane_id",
                "planned_next_edge", "current_lane_can_reach_next_edge",
                "min_cv_obb_ttc_s", "min_center_distance_m",
            ) if key in info
        }
        physical = self.latest_full_physical_snapshot
        ego = (physical or {}).get("ego") or {}
        speed_control = (physical or {}).get("speed_control") or {}
        lane_control = (physical or {}).get("lane_control") or {}
        row["post_action_state"] = {
            "snapshot_raw_step": (physical or {}).get("raw_step"),
            "snapshot_sim_time": (physical or {}).get("sim_time"),
            "ego_state_source": (physical or {}).get("ego_state_source"),
            "ego_state_time": (physical or {}).get("ego_state_time"),
            "actual_speed_mps": ego.get("speed"),
            "actual_road_id": ego.get("road_id"),
            "actual_lane_id": ego.get("lane_id"),
            "actual_lane_position_m": ego.get("lane_position"),
            "requested_target_speed_mps": speed_control.get("requested_target_speed_mps"),
            "effective_target_speed_mps": speed_control.get("effective_target_speed_mps"),
            "lane_control": lane_control,
        }
        row["reward"] = float(reward)
        row["terminated"] = bool(terminated)
        row["truncated"] = bool(truncated)


class _StepCaptureEnv:
    """Transparent delegation that pairs existing step results with prediction rows."""
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        self.predictor.after_step(reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info


def _outcome(record: dict[str, Any]) -> str:
    if bool(record.get("collision")):
        return "collision"
    if bool(record.get("success")):
        return "success"
    if bool(record.get("off_route")):
        return "off_route"
    if bool(record.get("timeout")):
        return "timeout"
    return "unknown"


def _runtime_sources() -> dict[str, str]:
    sources = _runtime_source_paths()
    return {name: sha256_file(path) for name, path in sources.items()}


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "fast-developer/train_intersection_yield_v2.py": FAST_DEVELOPER / "train_intersection_yield_v2.py",
        "fast-developer/train_intersection_yield_v2_d1.py": FAST_DEVELOPER / "train_intersection_yield_v2_d1.py",
        "fast-developer/behavior_diagnostics.py": FAST_DEVELOPER / "behavior_diagnostics.py",
        "algos/sb3_torch/evaluation.py": PROJECT / "algos" / "sb3_torch" / "evaluation.py",
        "algos/sb3_torch/sac.py": PROJECT / "algos" / "sb3_torch" / "sac.py",
        "algos/sb3_torch/policies.py": PROJECT / "algos" / "sb3_torch" / "policies.py",
        "algos/sb3_torch/representation.py": PROJECT / "algos" / "sb3_torch" / "representation.py",
        "algos/sb3_torch/features.py": PROJECT / "algos" / "sb3_torch" / "features.py",
        "algos/sb3_torch/incremental_topo_encoder.py": PROJECT / "algos" / "sb3_torch" / "incremental_topo_encoder.py",
        "algos/sb3_torch/topo_temporal_features_v2.py": PROJECT / "algos" / "sb3_torch" / "topo_temporal_features_v2.py",
        "envs/sumo/sumo_env.py": PROJECT / "envs" / "sumo" / "sumo_env.py",
        "envs/sumo/paper_env.py": PROJECT / "envs" / "sumo" / "paper_env.py",
        "envs/sumo/high_density_env_v1.py": PROJECT / "envs" / "sumo" / "high_density_env_v1.py",
        "fast-developer/diagnose_strt_decision_windows_20261003.py": SCRIPT,
        "fast-developer/analysis/decision_window_intervention_protocol_20261003.md": PROTOCOL_PATH,
    }


def _archive_runtime_sources(run_root: Path) -> dict[str, Any]:
    """Copy the runner/protocol and hash all inputs used for this diagnostic."""
    snapshot_root = run_root / "runtime_source_snapshot"
    if snapshot_root.exists():
        raise FileExistsError(f"refusing to reuse source snapshot: {snapshot_root}")
    entries = []
    for relative, source in _runtime_source_paths().items():
        if not source.is_file():
            raise FileNotFoundError(f"runtime source missing: {source}")
        payload = source.read_bytes()
        destination = snapshot_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if sha256_file(destination) != digest:
            raise IOError(f"runtime source snapshot hash mismatch: {destination}")
        entries.append({"path": relative, "sha256": digest, "bytes": len(payload)})
    manifest = {
        "snapshot_root": str(snapshot_root.resolve()),
        "file_count": len(entries),
        "files": entries,
    }
    (run_root / "runtime_source_snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _load_runtime_and_model(run_root: Path):
    import numpy as np
    import torch

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    if str(FAST_DEVELOPER) not in sys.path:
        sys.path.insert(0, str(FAST_DEVELOPER))
    import train_intersection_yield_v2 as base
    import train_intersection_yield_v2_d1 as d1
    from algos.sb3_torch.evaluation import (
        evaluate_model_detailed,
        source_evaluation_augmentation,
    )
    from algos.sb3_torch.sac import SceneRepresentationSAC

    if Path(base.__file__).resolve().parent != FAST_DEVELOPER:
        raise RuntimeError(f"wrong base trainer imported: {base.__file__}")
    if Path(d1.__file__).resolve().parent != FAST_DEVELOPER:
        raise RuntimeError(f"wrong D1 trainer imported: {d1.__file__}")
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = run_root
    d1.RESULT_ROOT = run_root
    method = METHOD
    overlay_root = run_root / "_hd" / "st_rt_window" / "ns_eval"
    env = base.make_env_factory(d1._env_adapter_d1(method), overlay_root)(
        base._environment_namespace(), evaluation=True
    )
    env = d1._wrap_route_reachability_env(env, method)
    env = d1._wrap_method_env(env, method, run_root, "eval")
    env, recorder = base._wrap_behavior_diagnostics_env(
        env,
        run_root,
        "eval",
        method,
        env_contract=d1._env_adapter_d1(method),
        evaluation_seed_start=SEED_START,
        evaluation_episode_start=0,
        evaluation_episode_end=MAX_CONTROL_EPISODES,
        evaluation_episodes=MAX_CONTROL_EPISODES,
        deterministic=True,
        checkpoint=str(CHECKPOINT_PATH.resolve()),
        device="cpu",
        smoke=False,
        diagnostic_experiment="frozen_decision_window_intervention",
    )
    model = SceneRepresentationSAC.load(
        str(CHECKPOINT_PATH), env=env, device="cpu", buffer_size=32
    )
    return base, d1, evaluate_model_detailed, source_evaluation_augmentation, env, recorder, model, np, torch


def _model_fingerprint(model) -> tuple[str, int]:
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), int(getattr(model, "_n_updates", 0))


def _run_episode(
    *, env, recorder, model, evaluator, source_augmentation, np, torch,
    snapshot_observer=None,
    seed: int, traffic_episode_index: int, output_episode_index: int,
    baseline_trace=None, anchor_index=None, target_speed_mps=None,
    verify_entire_trace=False,
):
    np.random.seed(int(seed) + 600_000)
    torch.manual_seed(int(seed) + 600_000)
    env.unwrapped._traffic_episode_index = int(traffic_episode_index)
    env.unwrapped._traffic_roll = None
    predictor = _WindowPredictor(
        model,
        recorder,
        env=env,
        baseline_trace=baseline_trace,
        anchor_index=anchor_index,
        target_speed_mps=target_speed_mps,
        verify_entire_trace=verify_entire_trace,
    )
    if snapshot_observer is not None:
        snapshot_observer["predictor"] = predictor
    capture_env = _StepCaptureEnv(env, predictor)
    try:
        detailed = evaluator(
            predictor,
            capture_env,
            episodes=1,
            seed=int(seed),
            deterministic=True,
            sumo_step_seconds=0.1,
            policy_action_hold=1,
        )
    finally:
        if snapshot_observer is not None:
            snapshot_observer["predictor"] = None
    _validate_intervention_anchor(
        predictor,
        detailed.episode_records[0].to_dict().get("outcome")
        if detailed.episode_records else None,
    )
    record = detailed.episode_records[0].to_dict()
    record["episode"] = int(output_episode_index)
    record["traffic_episode_index"] = int(traffic_episode_index)
    record["trace"] = predictor.trace
    record["reset_raw_observer_callback_count"] = predictor.reset_raw_observer_callback_count
    record["reset_snapshot_sim_time_s"] = (
        predictor.trace[0].get("pre_sim_time") if predictor.trace else None
    )
    record["reset_warmup_raw_steps"] = None
    record["reset_warmup_unavailable_reason"] = (
        "SumoEnv.reset advances SUMO before resetting local raw_steps and exposes no warmup tick count; "
        "on_raw_step callback count is recorded separately."
    )
    record["prefix_match"] = predictor.prefix_error is None if baseline_trace is not None else None
    record["prefix_error"] = predictor.prefix_error
    record["intervention"] = {
        "anchor_index": anchor_index,
        "anchor_decision": anchor_index + 1 if anchor_index is not None else None,
        "requested_target_speed_mps": target_speed_mps,
        "requested_decisions": INTERVENTION_DECISIONS if target_speed_mps is not None else 0,
        "applied_decisions": predictor.intervention_applied_decisions,
        "speed_action_coordinate_formula": "clip((action[0]+1)*5,0,10); inverse action[0]=target_mps/5-1",
        "lateral_coordinate_changed": False,
    }
    return record


def _write_manifest(run_root: Path, cases, reference):
    run_root.mkdir(parents=True, exist_ok=False)
    source_snapshot = _archive_runtime_sources(run_root)
    identity_seeds = reference["identity_replay_seeds"]
    manifest = {
        "schema_version": "strt_decision_window_intervention_v1",
        "status": "prepared",
        "training": False,
        "simulator_started": False,
        "scenario": SCENARIO,
        "depart_scale": DEPART_SCALE,
        "traffic_split": EVAL_SPLIT,
        "method": METHOD,
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "reference_evaluation": reference["evaluation_results"],
        "reference_decision_stream": reference["decision_stream"],
        "case_selection": "outcome-stratified; first 8 collision and first 4 success by ascending logical validation seed; not a population-rate estimator",
        "cases": cases,
        "identity_replay_seeds": identity_seeds,
        "runtime_sources": _runtime_sources(),
        "runtime_source_snapshot": source_snapshot,
        "episode_budget": {
            "baseline_max": 12,
            "identity_max": 2,
            "intervention_max": 72,
            "total_max": MAX_CONTROL_EPISODES,
            "raw_control_steps_per_episode_max": MAX_RAW_STEPS_PER_EPISODE,
            "raw_control_steps_total_max": MAX_CONTROL_RAW_STEPS,
            "reset_warmup_raw_steps": None,
            "reset_warmup_unavailable_reason": (
                "SumoEnv.reset advances simulationStep before resetting local raw_steps and does not expose the exact count; "
                "recorder callback count and initial snapshot simulation time are separate observations."
            ),
        },
        "anchors": [
            "three decisions before first pre-action internal-road state",
            "first pre-action internal-road state",
            "three decisions before baseline termination",
        ],
        "speed_intervention": {
            "target_speeds_mps": list(INTERVENTION_TARGET_SPEEDS_MPS),
            "policy_decisions": INTERVENTION_DECISIONS,
            "action_mapping": "environment speed=clip((a0+1)*5,0,10), inverse a0=target_mps/5-1",
            "lateral_action": "retain deterministic policy coordinate 1",
            "return_to_policy_after": INTERVENTION_DECISIONS,
        },
        "prefix_match": {
            "policy_observation_and_unmodified_action_hashes": "exact for every decision up to and including the anchor",
            "physical_state": "exact at anchor; current ego and vehicle IDs/positions/velocities/heading/dimensions/road/lane fields only; hidden route fields excluded",
            "mismatch_action": "do not apply intervention; finish that allocated episode for cleanup/logging, mark invalid and stop the suite",
        },
        "runtime_source_sha256": _runtime_sources(),
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _run_start(run_root: Path, rows, cases, reference):
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing window-diagnostic root: {run_root}")
    manifest = _write_manifest(run_root, cases, reference)
    (run_root / "episodes").mkdir(parents=True, exist_ok=True)
    status_path = run_root / "launcher_status.json"
    status = {"status": "starting", "run_root": str(run_root), "completed_episodes": 0}
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    base = d1 = evaluator = source_augmentation = env = recorder = model = np = torch = None
    baseline_by_seed: dict[int, dict[str, Any]] = {}
    reference_by_seed = {int(row["seed"]): row for row in rows}
    completed_episode_count = 0
    raw_control_steps = 0
    abort_reason = None

    def persist_progress(current_phase: str, current_seed: int | None = None) -> None:
        status.update(
            status="running",
            completed_episodes=completed_episode_count,
            ego_control_raw_steps_used=raw_control_steps,
            ego_control_raw_steps_max=MAX_CONTROL_RAW_STEPS,
            current_phase=current_phase,
            current_seed=current_seed,
        )
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        base, d1, evaluator, source_augmentation, env, recorder, model, np, torch = _load_runtime_and_model(run_root)
        snapshot_observer, restore_snapshot_observer = _install_raw_snapshot_observer(recorder)
        before_fingerprint = _model_fingerprint(model)
        (run_root / "runtime_identity.json").write_text(json.dumps({
            "checkpoint": str(CHECKPOINT_PATH.resolve()),
            "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
            "policy_state_sha256_before": before_fingerprint[0],
            "n_updates_before": before_fingerprint[1],
            "python_executable": sys.executable,
            "runtime_sources": _runtime_sources(),
            "runtime_source_snapshot_manifest": str(
                (run_root / "runtime_source_snapshot_manifest.json").resolve()
            ),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Baselines first; each must reproduce archived outcome and raw/decision length.
        for output_index, case in enumerate(cases):
            seed = int(case["seed"])
            rec = _run_episode(
                env=env, recorder=recorder, model=model, evaluator=evaluator,
                source_augmentation=source_augmentation, np=np, torch=torch,
                snapshot_observer=snapshot_observer,
                seed=seed, traffic_episode_index=seed - SEED_START,
                output_episode_index=completed_episode_count,
            )
            expected = reference_by_seed[seed]
            rec["reference_match"] = {
                "outcome": _outcome(rec) == _outcome(expected),
                "raw_steps": int(rec["raw_steps"]) == int(expected["raw_steps"]),
                "decision_steps": int(rec["decision_steps"]) == int(expected["decision_steps"]),
                "traffic_variant": rec.get("traffic_variant") == expected.get("traffic_variant"),
            }
            rec["reference_record"] = {
                "outcome": _outcome(expected),
                "raw_steps": int(expected["raw_steps"]),
                "decision_steps": int(expected["decision_steps"]),
                "traffic_variant": expected.get("traffic_variant"),
            }
            anchors = plan_anchors(rec["trace"])
            rec["anchor_plan"] = anchors
            rec["anchor_physical_snapshots"] = {
                str(item["decision_index"]): rec["trace"][item["decision_index"]].get("physical_state_sha256")
                for item in anchors["unique"]
            }
            _append_jsonl(run_root / "episodes" / "baseline.jsonl", rec)
            raw_control_steps += int(rec["raw_steps"])
            completed_episode_count += 1
            persist_progress("baseline", seed)
            if int(rec["raw_steps"]) > MAX_RAW_STEPS_PER_EPISODE:
                raise RuntimeError(f"baseline episode exceeded the {MAX_RAW_STEPS_PER_EPISODE}-raw-step cap")
            if not all(rec["reference_match"].values()):
                raise PrefixMismatch(
                    f"baseline did not reproduce archived seed={seed}: "
                    f"got={_outcome(rec)}/{rec['raw_steps']}/{rec['decision_steps']} "
                    f"expected={_outcome(expected)}/{expected['raw_steps']}/{expected['decision_steps']}; "
                    f"traffic_variant={rec.get('traffic_variant')!r}/"
                    f"{expected.get('traffic_variant')!r}"
                )
            baseline_by_seed[seed] = rec
            if completed_episode_count > MAX_CONTROL_EPISODES or raw_control_steps > MAX_CONTROL_RAW_STEPS:
                raise RuntimeError("hard episode/raw-step budget exceeded")

        # Two full identity replays are a gate before any action change.
        identity_seeds = reference["identity_replay_seeds"]
        for seed in identity_seeds:
            baseline = baseline_by_seed[int(seed)]
            replay = _run_episode(
                env=env, recorder=recorder, model=model, evaluator=evaluator,
                source_augmentation=source_augmentation, np=np, torch=torch,
                snapshot_observer=snapshot_observer,
                seed=int(seed), traffic_episode_index=int(seed) - SEED_START,
                output_episode_index=completed_episode_count,
                baseline_trace=baseline["trace"], verify_entire_trace=True,
            )
            mismatch = prefix_mismatch(baseline["trace"], replay["trace"], len(baseline["trace"]) - 1)
            replay["identity_match"] = (
                mismatch is None
                and len(replay["trace"]) == len(baseline["trace"])
                and _outcome(replay) == _outcome(baseline)
                and int(replay["raw_steps"]) == int(baseline["raw_steps"])
                and int(replay["decision_steps"]) == int(baseline["decision_steps"])
            )
            replay["identity_mismatch"] = mismatch
            _append_jsonl(run_root / "episodes" / "identity.jsonl", replay)
            raw_control_steps += int(replay["raw_steps"])
            completed_episode_count += 1
            persist_progress("identity", int(seed))
            if int(replay["raw_steps"]) > MAX_RAW_STEPS_PER_EPISODE:
                raise RuntimeError(f"identity replay exceeded the {MAX_RAW_STEPS_PER_EPISODE}-raw-step cap")
            if completed_episode_count > MAX_CONTROL_EPISODES or raw_control_steps > MAX_CONTROL_RAW_STEPS:
                raise RuntimeError("hard episode/raw-step budget exceeded")
            if not replay["identity_match"]:
                raise PrefixMismatch(f"identity replay failed for seed={seed}: {mismatch}")

        # Interventions are separate deterministic prefix replays from reset.
        for case in cases:
            seed = int(case["seed"])
            baseline = baseline_by_seed[seed]
            anchors = baseline.get("anchor_plan") or plan_anchors(baseline["trace"])
            for anchor in anchors["unique"]:
                anchor_index = int(anchor["decision_index"])
                for target in INTERVENTION_TARGET_SPEEDS_MPS:
                    intervention = _run_episode(
                        env=env, recorder=recorder, model=model, evaluator=evaluator,
                        source_augmentation=source_augmentation, np=np, torch=torch,
                        snapshot_observer=snapshot_observer,
                        seed=seed, traffic_episode_index=seed - SEED_START,
                        output_episode_index=completed_episode_count,
                        baseline_trace=baseline["trace"], anchor_index=anchor_index,
                        target_speed_mps=target,
                    )
                    intervention["anchor_phases"] = anchor["phases"]
                    intervention["anchor_baseline"] = anchor
                    intervention["prefix_match"] = intervention.get("prefix_error") is None
                    _append_jsonl(run_root / "episodes" / "interventions.jsonl", intervention)
                    raw_control_steps += int(intervention["raw_steps"])
                    completed_episode_count += 1
                    persist_progress("intervention", seed)
                    if not intervention["prefix_match"]:
                        abort_reason = f"prefix mismatch before intervention seed={seed}, anchor={anchor_index}"
                        raise PrefixMismatch(abort_reason)
                    if int(intervention["raw_steps"]) > MAX_RAW_STEPS_PER_EPISODE:
                        raise RuntimeError("episode exceeded the 600 raw control-step cap")
                    if completed_episode_count > MAX_CONTROL_EPISODES or raw_control_steps > MAX_CONTROL_RAW_STEPS:
                        raise RuntimeError("hard episode/raw-step budget exceeded")

        after_fingerprint = _model_fingerprint(model)
        if before_fingerprint != after_fingerprint:
            raise RuntimeError("frozen policy state or update counter changed during intervention run")
        if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
            raise RuntimeError("frozen checkpoint bytes changed during intervention run")
        summary = {
            "status": "complete",
            "episodes_used": completed_episode_count,
            "episodes_max": MAX_CONTROL_EPISODES,
            "ego_control_raw_steps_used": raw_control_steps,
            "ego_control_raw_steps_max": MAX_CONTROL_RAW_STEPS,
            "reset_warmup_raw_steps": None,
            "reset_warmup_unavailable_reason": (
                "SumoEnv.reset advances simulationStep before resetting local raw_steps and does not expose the exact count; "
                "see per-episode reset_raw_observer_callback_count and reset_snapshot_sim_time_s."
            ),
            "checkpoint_state_fingerprint_unchanged": True,
            "n_updates_before_after": before_fingerprint[1],
            "identity_replays_passed": len(identity_seeds),
            "abort_reason": None,
        }
        (run_root / "window_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "complete"
        (run_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        status.update(status="complete", completed_episodes=completed_episode_count)
        return 0
    except Exception as exc:
        abort_reason = abort_reason or f"{type(exc).__name__}: {exc}"
        manifest["status"] = "aborted"
        manifest["abort_reason"] = abort_reason
        (run_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (run_root / "window_summary.json").write_text(json.dumps({
            "status": "aborted", "episodes_used": completed_episode_count,
            "ego_control_raw_steps_used": raw_control_steps,
            "episodes_max": MAX_CONTROL_EPISODES,
            "ego_control_raw_steps_max": MAX_CONTROL_RAW_STEPS,
            "abort_reason": abort_reason,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status.update(status="aborted", completed_episodes=completed_episode_count, error=abort_reason)
        return 2
    finally:
        if "restore_snapshot_observer" in locals():
            try:
                restore_snapshot_observer()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        status["finished_at"] = time.time()
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--start", action="store_true", help="explicitly run the bounded SUMO diagnostic")
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    runs_root = (WORKSPACE / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        parser.error(f"run root must be a child of {runs_root}")
    try:
        rows, cases, reference = _validate_reference()
        if run_root.exists():
            raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
        plan = {
            "mode": "execute" if args.start else "read_only_preview",
            "training_started": False,
            "sumo_started": bool(args.start),
            "run_root": str(run_root),
            "checkpoint": reference["checkpoint"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "reference_summary": reference["summary"],
            "cases": cases,
            "identity_replay_seeds": reference["identity_replay_seeds"],
            "budgets": {
                "episodes_max": MAX_CONTROL_EPISODES,
                "ego_control_raw_steps_max": MAX_CONTROL_RAW_STEPS,
                "per_episode_raw_steps_max": MAX_RAW_STEPS_PER_EPISODE,
                "targets_mps": list(INTERVENTION_TARGET_SPEEDS_MPS),
                "decisions_per_intervention": INTERVENTION_DECISIONS,
            },
        }
        if not args.start:
            print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        return _run_start(run_root, rows, cases, reference)
    except Exception as exc:
        print(f"window-runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
