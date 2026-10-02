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
RECOVERY_SUBDIR = "recovery01"
MAX_COLLISION_CASES = 8
MAX_SUCCESS_CASES = 4
IDENTITY_CASES_PER_OUTCOME = 1
MAX_RAW_STEPS_PER_EPISODE = 600
MAX_CONTROL_EPISODES = 12 + 2 + 12 * 3 * 2
MAX_CONTROL_RAW_STEPS = MAX_CONTROL_EPISODES * MAX_RAW_STEPS_PER_EPISODE
RECOVERY_PARENT_EPISODES_EXPECTED = 13
RECOVERY_PARENT_RAW_STEPS_EXPECTED = 3340
RECOVERY_INTERVENTION_PAIRS_MAX = 29
INTERVENTION_TARGET_SPEEDS_MPS = (0.0, 6.0)
INTERVENTION_DECISIONS = 3
ANCHOR_PHASE_ORDER = (
    "three_decisions_before_first_internal",
    "first_internal_pre_action",
    "three_decisions_before_terminal",
)


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
        "state_source", "state_time", "road_lane_identity_source",
        "road_lane_query_sim_time", "road_lane_query_raw_step",
        "road_lane_cache_sample_raw_step", "road_lane_cache_observation",
    )

    def actor(value):
        if not isinstance(value, dict):
            return None
        output = {name: value.get(name) for name in actor_fields}
        if output["road_lane_cache_sample_raw_step"] is None:
            output["road_lane_cache_sample_raw_step"] = value.get("context_sample_raw_step")
        if output["road_lane_cache_observation"] is None:
            missing = value.get("missing") or {}
            output["road_lane_cache_observation"] = {
                "cached_road_id": value.get("road_id"),
                "cached_lane_id": value.get("lane_id"),
                "sample_raw_step": output["road_lane_cache_sample_raw_step"],
                "road_id_missing_reason": missing.get("road_id"),
                "lane_id_missing_reason": missing.get("lane_id"),
            }
        return output

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


def _supplement_live_road_lane_identity(
    snapshot: dict[str, Any] | None,
    env,
    *,
    current_vehicle_ids: list[str] | None,
    current_person_ids: list[str] | None,
    current_raw_step: int | None,
    query_errors: list[str] | None = None,
) -> dict[str, Any] | None:
    """Read current road/lane IDs for every active actor without stepping SUMO."""
    if snapshot is None:
        return None
    errors = list(query_errors or [])
    membership = snapshot.get("physical_membership") or {}
    if (
        errors
        or membership.get("vehicle_id_set_matches") is not True
        or membership.get("person_id_set_matches") is not True
        or membership.get("raw_step_matches") is not True
        or current_vehicle_ids is None
        or current_person_ids is None
        or current_raw_step is None
    ):
        result = dict(snapshot)
        result["physical_identity_query"] = {
            "source": "read_only_traci_current_road_lane",
            "completed": False,
            "errors": errors or ["snapshot_membership_or_raw_step_gate_failed"],
        }
        result["physical_state_complete"] = False
        result["physical_state_incomplete_reasons"] = sorted(set(
            list(result.get("physical_state_incomplete_reasons") or [])
            + ["road_lane_query_prerequisite_failed"]
        ))
        return result

    base = connection = None
    try:
        base = env.unwrapped
        connection = getattr(base, "_connection", None)
        if connection is None:
            raise RuntimeError("connection_unavailable")
        pre_time = float(connection.simulation.getTime())
        pre_raw_step = int(getattr(base, "_raw_steps"))
        pre_vehicle_ids = sorted(str(value) for value in connection.vehicle.getIDList())
        pre_person_ids = sorted(str(value) for value in connection.person.getIDList())
    except Exception as exc:
        errors.append(f"before_query:{type(exc).__name__}:{exc}")
        pre_time = None
        pre_raw_step = None
        pre_vehicle_ids = None
        pre_person_ids = None

    captured_actors = [snapshot.get("ego"), *(snapshot.get("vehicles") or [])]
    actor_by_key = {
        (str(actor.get("kind")), str(actor.get("id"))): actor
        for actor in captured_actors
        if isinstance(actor, dict) and actor.get("kind") and actor.get("id") is not None
    }
    live_groups = (
        ("vehicle", sorted(str(value) for value in current_vehicle_ids)),
        ("person", sorted(str(value) for value in current_person_ids)),
    )
    query_count = 0
    if pre_vehicle_ids != live_groups[0][1] or pre_person_ids != live_groups[1][1]:
        errors.append("live_actor_id_set_changed_before_query")
    if pre_raw_step != current_raw_step or snapshot.get("raw_step") != current_raw_step:
        errors.append("raw_step_changed_before_query")
    if pre_time != snapshot.get("sim_time"):
        errors.append("sim_time_differs_from_snapshot_before_query")

    if not errors:
        for kind, actor_ids in live_groups:
            api = getattr(connection, kind, None)
            if api is None and actor_ids:
                errors.append(f"{kind}_api_unavailable")
                continue
            for actor_id in actor_ids:
                actor = actor_by_key.get((kind, actor_id))
                if actor is None:
                    errors.append(f"snapshot_actor_missing:{kind}:{actor_id}")
                    continue
                try:
                    cached_road_id = actor.get("road_id")
                    cached_lane_id = actor.get("lane_id")
                    cached_sample = actor.get("road_lane_cache_sample_raw_step")
                    if cached_sample is None:
                        cached_sample = actor.get("context_sample_raw_step")
                    try:
                        cache_sample_raw_step = (
                            int(cached_sample) if cached_sample is not None else None
                        )
                    except (TypeError, ValueError, OverflowError):
                        cache_sample_raw_step = None
                    road_value = api.getRoadID(actor_id)
                    lane_value = api.getLaneID(actor_id)
                    if road_value is None or lane_value is None:
                        raise ValueError("TraCI returned None for road/lane identity")
                    road_id = str(road_value)
                    lane_id = str(lane_value)
                    if not road_id or not lane_id:
                        raise ValueError("TraCI returned an empty road/lane identity")
                    for field, live_value in (("road_id", road_id), ("lane_id", lane_id)):
                        cached_value = cached_road_id if field == "road_id" else cached_lane_id
                        if cached_value is None or str(cached_value) == live_value:
                            continue
                        if cache_sample_raw_step == pre_raw_step:
                            errors.append(
                                f"same_step_cached_{field}_mismatch:{kind}:{actor_id}:"
                                f"{cached_value!r}!={live_value!r}"
                            )
                    if (
                        cache_sample_raw_step is not None
                        and cache_sample_raw_step > int(snapshot.get("raw_step"))
                    ):
                        errors.append(
                            f"future_cached_road_lane_sample:{kind}:{actor_id}:"
                            f"{cache_sample_raw_step}>{snapshot.get('raw_step')}"
                        )
                    if cache_sample_raw_step == pre_raw_step:
                        nonempty_cache_conflict = (
                            (cached_road_id is not None and str(cached_road_id) != road_id)
                            or (cached_lane_id is not None and str(cached_lane_id) != lane_id)
                        )
                        if nonempty_cache_conflict:
                            cache_relation = "same_step_conflict"
                        elif cached_road_id is None or cached_lane_id is None:
                            cache_relation = "same_step_cache_incomplete"
                        else:
                            cache_relation = "same_step_match"
                    elif (
                        cache_sample_raw_step is not None
                        and cache_sample_raw_step > int(snapshot.get("raw_step"))
                    ):
                        cache_relation = "future_sample"
                    elif (
                        cache_sample_raw_step is not None
                        and cache_sample_raw_step < int(snapshot.get("raw_step"))
                    ):
                        cache_relation = "stale_sample"
                    else:
                        cache_relation = "sample_time_unknown"
                    actor["road_lane_cache_sample_raw_step"] = cache_sample_raw_step
                    actor["road_lane_cache_observation"] = {
                        "cached_road_id": cached_road_id,
                        "cached_lane_id": cached_lane_id,
                        "sample_raw_step": cache_sample_raw_step,
                        "relation_to_live_query": cache_relation,
                    }
                    actor["road_id"] = road_id
                    actor["lane_id"] = lane_id
                    actor["road_lane_identity_source"] = "read_only_traci_live_actor"
                    actor["road_lane_query_sim_time"] = pre_time
                    actor["road_lane_query_raw_step"] = pre_raw_step
                    query_count += 1
                except Exception as exc:
                    errors.append(f"actor_query:{kind}:{actor_id}:{type(exc).__name__}:{exc}")

    try:
        post_time = float(connection.simulation.getTime())
        post_raw_step = int(getattr(base, "_raw_steps"))
        post_vehicle_ids = sorted(str(value) for value in connection.vehicle.getIDList())
        post_person_ids = sorted(str(value) for value in connection.person.getIDList())
    except Exception as exc:
        errors.append(f"after_query:{type(exc).__name__}:{exc}")
        post_time = None
        post_raw_step = None
        post_vehicle_ids = None
        post_person_ids = None

    stable_time = pre_time is not None and post_time == pre_time == snapshot.get("sim_time")
    stable_raw_step = (
        pre_raw_step is not None
        and post_raw_step == pre_raw_step == current_raw_step == snapshot.get("raw_step")
    )
    stable_ids = (
        post_vehicle_ids == pre_vehicle_ids == live_groups[0][1]
        and post_person_ids == pre_person_ids == live_groups[1][1]
    )
    if not stable_time:
        errors.append("sim_time_changed_during_road_lane_query")
    if not stable_raw_step:
        errors.append("raw_step_changed_during_road_lane_query")
    if not stable_ids:
        errors.append("live_actor_id_set_changed_during_road_lane_query")

    rebuilt = physical_snapshot_payload({
        "raw_step": snapshot.get("raw_step"),
        "sim_time": snapshot.get("sim_time"),
        "ego_state_source": snapshot.get("ego_state_source"),
        "ego_state_time": snapshot.get("ego_state_time"),
        "ego": snapshot.get("ego"),
        "vehicles": snapshot.get("vehicles") or [],
        "errors": snapshot.get("physical_snapshot_error_details") or [],
        "speed_control": snapshot.get("speed_control"),
        "lane_control": snapshot.get("lane_control"),
    })
    if rebuilt is None:
        return None
    rebuilt["snapshot_capture_source"] = snapshot.get("snapshot_capture_source")
    if snapshot.get("initial_snapshot_incomplete_reasons") is not None:
        rebuilt["initial_snapshot_incomplete_reasons"] = snapshot[
            "initial_snapshot_incomplete_reasons"
        ]
    rebuilt["physical_identity_query"] = {
        "source": "read_only_traci_current_road_lane",
        "completed": query_count == sum(len(ids) for _, ids in live_groups) and not errors,
        "queried_actor_count": query_count,
        "expected_actor_count": sum(len(ids) for _, ids in live_groups),
        "pre_sim_time": pre_time,
        "post_sim_time": post_time,
        "pre_raw_step": pre_raw_step,
        "post_raw_step": post_raw_step,
        "vehicle_id_set_stable": post_vehicle_ids == pre_vehicle_ids == live_groups[0][1],
        "person_id_set_stable": post_person_ids == pre_person_ids == live_groups[1][1],
        "errors": errors,
        "cache_observations": [
            {
                "kind": kind,
                "id": actor_id,
                **(actor_by_key.get((kind, actor_id), {}).get("road_lane_cache_observation") or {}),
            }
            for kind, actor_ids in live_groups
            for actor_id in actor_ids
            if (kind, actor_id) in actor_by_key
        ],
    }
    return verify_physical_snapshot_membership(
        rebuilt,
        current_vehicle_ids=current_vehicle_ids,
        current_person_ids=current_person_ids,
        current_raw_step=current_raw_step,
        query_errors=errors,
    )


def physical_fingerprint_payload(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hash only physical identity/state, excluding controls and diagnostics."""
    if snapshot is None:
        return None
    query_metadata = {
        "road_lane_identity_source",
        "road_lane_query_sim_time",
        "road_lane_query_raw_step",
        "road_lane_cache_sample_raw_step",
        "road_lane_cache_observation",
    }

    def physical_actor(value):
        if not isinstance(value, dict):
            return value
        return {key: item for key, item in value.items() if key not in query_metadata}

    keys = (
        "raw_step", "sim_time", "ego_state_source", "ego_state_time",
        "physical_actor_count",
    )
    result = {key: snapshot.get(key) for key in keys}
    result["ego"] = physical_actor(snapshot.get("ego"))
    result["vehicles"] = [physical_actor(actor) for actor in snapshot.get("vehicles", [])]
    return result


def _trajectory_history_index_audit(observation, np=None) -> dict[str, Any]:
    """Audit the encoder's count-minus-one index without changing its input."""
    if np is None:
        import numpy as np
    if not isinstance(observation, dict) or "trajectory" not in observation:
        return {"status": "unavailable", "reason": "trajectory_observation_missing"}
    trajectory = observation["trajectory"]
    try:
        from algos.sb3_torch.features import _nonzero_mask

        valid_mask = _nonzero_mask(trajectory)
        mask_source = "algos.sb3_torch.features._nonzero_mask"
    except Exception as exc:
        # Keep pure unit tests and preview tooling usable without importing the
        # training runtime; this is the helper's exact first-coordinate rule.
        raw = trajectory.detach().cpu().numpy() if hasattr(trajectory, "detach") else np.asarray(trajectory)
        valid_mask = raw[..., 0] != 0
        mask_source = f"exact_numpy_fallback:{type(exc).__name__}"

    raw = trajectory.detach().cpu().numpy() if hasattr(trajectory, "detach") else np.asarray(trajectory)
    if hasattr(valid_mask, "detach"):
        valid_mask = valid_mask.detach().cpu().numpy()
    else:
        valid_mask = np.asarray(valid_mask)
    if raw.ndim == 3 and valid_mask.shape == raw.shape[:-1]:
        batches = [(None, raw, valid_mask)]
    elif raw.ndim == 4 and valid_mask.shape == raw.shape[:-1]:
        batches = [(index, raw[index], valid_mask[index]) for index in range(raw.shape[0])]
    else:
        return {
            "status": "unsupported_shape",
            "trajectory_shape": list(raw.shape),
            "mask_shape": list(valid_mask.shape),
            "mask_source": mask_source,
        }

    actors = []
    for batch_index, sample, sample_mask in batches:
        for actor_index, history_mask in enumerate(sample_mask):
            mask = np.asarray(history_mask, dtype=bool).reshape(-1)
            valid_count = int(mask.sum())
            count_minus_one_index = max(0, valid_count - 1)
            valid_indices = np.flatnonzero(mask)
            true_last_valid_index = int(valid_indices[-1]) if valid_indices.size else None
            selected_is_padding = (
                valid_count == 0 or not bool(mask[count_minus_one_index])
            )
            actors.append({
                "batch_index": batch_index,
                "actor_index": int(actor_index),
                "actor_role": "ego" if actor_index == 0 else "social",
                "history_length": int(mask.size),
                "valid_count": valid_count,
                "count_minus_one_index": count_minus_one_index,
                "count_minus_one_index_is_model_padding": selected_is_padding,
                "true_last_valid_index": true_last_valid_index,
                "index_delta_true_minus_count_minus_one": (
                    true_last_valid_index - count_minus_one_index
                    if true_last_valid_index is not None else None
                ),
            })
    return {
        "status": "complete",
        "observation_key": "trajectory",
        "trajectory_shape": list(raw.shape),
        "actor_axis": "0 for unbatched; 1 for batched observations",
        "history_axis": "1 for unbatched; 2 for batched observations",
        "mask_source": mask_source,
        "mask_rule": "trajectory[..., 0] != 0; exactly matches _nonzero_mask",
        "count_minus_one_rule": "max(valid_count - 1, 0)",
        "index_unit": "history-array index; no physical-time interval inferred",
        "actors": actors,
    }


def _summarize_history_index_audit(episode_records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        role: {
            "actor_decision_slots": 0,
            "nonempty_actor_histories": 0,
            "selected_model_padding_count": 0,
            "count_minus_one_matches_true_last_count": 0,
            "index_deltas": [],
        }
        for role in ("ego", "social")
    }
    decision_count = 0
    audit_unavailable_decisions = 0
    for episode in episode_records:
        for trace_row in episode.get("trace", []):
            audit = trace_row.get("observation_history_index_audit") or {}
            if audit.get("status") != "complete":
                audit_unavailable_decisions += 1
                continue
            decision_count += 1
            for actor in audit.get("actors", []):
                group = groups.get(actor.get("actor_role"))
                if group is None:
                    continue
                group["actor_decision_slots"] += 1
                if int(actor.get("valid_count", 0)) <= 0:
                    continue
                group["nonempty_actor_histories"] += 1
                if actor.get("count_minus_one_index_is_model_padding") is True:
                    group["selected_model_padding_count"] += 1
                if actor.get("count_minus_one_index") == actor.get("true_last_valid_index"):
                    group["count_minus_one_matches_true_last_count"] += 1
                delta = actor.get("index_delta_true_minus_count_minus_one")
                if delta is not None:
                    group["index_deltas"].append(int(delta))
    for group in groups.values():
        denominator = group["nonempty_actor_histories"]
        group["selected_model_padding_rate_among_nonempty"] = (
            group["selected_model_padding_count"] / denominator if denominator else None
        )
        group["count_minus_one_matches_true_last_rate_among_nonempty"] = (
            group["count_minus_one_matches_true_last_count"] / denominator
            if denominator else None
        )
        deltas = group.pop("index_deltas")
        group["index_delta_min"] = min(deltas) if deltas else None
        group["index_delta_max"] = max(deltas) if deltas else None
        group["index_delta_mean"] = sum(deltas) / len(deltas) if deltas else None
    return {
        "scope": "supplied episode records only; selected outcome-stratified cases do not estimate the full evaluation or training distribution",
        "episode_count": len(episode_records),
        "decision_count": decision_count,
        "audit_unavailable_decision_count": audit_unavailable_decisions,
        "mask_rule": "trajectory[..., 0] != 0 via _nonzero_mask",
        "exposure_definition": "among nonempty actor histories, count_minus_one_index points to a false _nonzero_mask slot",
        "actor_groups": groups,
    }


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


def _build_intervention_schedule(
    cases: list[dict[str, Any]],
    baseline_by_seed: dict[int, dict[str, Any]],
    *,
    pair_limit: int,
) -> dict[str, Any]:
    """Reserve phase/seed slots first; data gaps never refill the schedule."""
    slots = []
    pairs = []
    skipped = []
    seen = set()
    ordered_cases = sorted(cases, key=lambda row: int(row["seed"]))
    for phase in ANCHOR_PHASE_ORDER:
        for case in ordered_cases:
            seed = int(case["seed"])
            baseline = baseline_by_seed.get(seed)
            anchor_plan = (
                baseline.get("anchor_plan") or plan_anchors(baseline.get("trace") or [])
                if baseline is not None else None
            )
            anchor = None
            if anchor_plan is not None:
                anchor = next((
                    item for item in anchor_plan.get("unique", [])
                    if phase in item.get("phases", [])
                ), None)
            if baseline is None:
                slots.append({"seed": seed, "phase": phase, "slot_reason": "baseline_missing"})
                continue
            if anchor is None:
                slots.append({"seed": seed, "phase": phase, "slot_reason": "anchor_unavailable"})
                continue
            anchor_index = int(anchor["decision_index"])
            pair_key = (seed, anchor_index)
            if pair_key in seen:
                slots.append({
                    "seed": seed,
                    "phase": phase,
                    "anchor_index": anchor_index,
                    "slot_reason": "duplicate_anchor_merged",
                })
                continue
            seen.add(pair_key)
            slots.append({
                "seed": seed,
                "phase": phase,
                "anchor_index": anchor_index,
                "anchor": anchor,
            })

    reserved_slots = slots[:max(0, int(pair_limit))]
    for slot_index, slot in enumerate(reserved_slots):
        seed = int(slot["seed"])
        phase = slot["phase"]
        if slot.get("slot_reason"):
            skipped.append({
                "schedule_slot": slot_index,
                "seed": seed,
                "phase": phase,
                "reason": slot["slot_reason"],
            })
            continue
        baseline = baseline_by_seed.get(seed)
        if baseline is None:
            skipped.append({
                "schedule_slot": slot_index,
                "seed": seed,
                "phase": phase,
                "reason": "baseline_missing",
            })
            continue
        anchor = slot["anchor"]
        anchor_index = int(slot["anchor_index"])
        trace = baseline.get("trace") or []
        prefix_problem = None
        for index in range(anchor_index + 1):
            if index >= len(trace):
                prefix_problem = {"reason": "baseline_prefix_length_missing", "decision_index": index}
                break
            row = trace[index]
            if row.get("physical_state_complete") is not True:
                prefix_problem = {
                    "reason": "baseline_physical_state_incomplete",
                    "decision_index": index,
                    "details": row.get("physical_state_incomplete_reasons"),
                }
                break
            if not row.get("physical_state_sha256"):
                prefix_problem = {"reason": "baseline_physical_state_hash_missing", "decision_index": index}
                break
            identity_query = row.get("physical_identity_query") or {}
            if identity_query.get("completed") is not True:
                prefix_problem = {
                    "reason": "baseline_live_road_lane_query_incomplete",
                    "decision_index": index,
                    "details": identity_query.get("errors"),
                }
                break
        if prefix_problem is not None:
            skipped.append({
                "schedule_slot": slot_index,
                "seed": seed,
                "phase": phase,
                "anchor_index": anchor_index,
                **prefix_problem,
            })
            continue
        pairs.append({
            "pair_index": len(pairs),
            "schedule_slot": slot_index,
            "seed": seed,
            "phase": phase,
            "merged_phases": list(anchor.get("phases", [])),
            "anchor_index": anchor_index,
            "decision": int(anchor.get("decision", anchor_index + 1)),
            "pre_obs_raw_step": anchor.get("pre_obs_raw_step"),
            "sim_time": anchor.get("sim_time"),
            "time_to_terminal_decisions": anchor.get("time_to_terminal_decisions"),
            "target_speeds_mps": list(INTERVENTION_TARGET_SPEEDS_MPS),
        })

    for slot_index, slot in enumerate(slots[len(reserved_slots):], start=len(reserved_slots)):
        skipped.append({
            "schedule_slot": slot_index,
            "seed": int(slot["seed"]),
            "phase": slot["phase"],
            "anchor_index": slot.get("anchor_index"),
            "reason": "outside_fixed_schedule_cap",
        })
    return {
        "phase_priority": list(ANCHOR_PHASE_ORDER),
        "target_speeds_mps": list(INTERVENTION_TARGET_SPEEDS_MPS),
        "pair_limit": int(pair_limit),
        "schedule_slot_count": len(slots),
        "reserved_slot_count": len(reserved_slots),
        "data_gaps_do_not_refill_slots": True,
        "reserved_slots": [
            {
                "schedule_slot": index,
                "seed": int(slot["seed"]),
                "phase": slot["phase"],
                "anchor_index": slot.get("anchor_index"),
                "reservation_reason": slot.get("slot_reason"),
            }
            for index, slot in enumerate(reserved_slots)
        ],
        "pairs": pairs,
        "skipped": skipped,
        "pairs_planned": len(pairs),
        "episodes_planned": 2 * len(pairs),
    }


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _recovery_budget_plan(parent_episodes: int, parent_raw_steps: int) -> dict[str, int]:
    parent_episodes = int(parent_episodes)
    parent_raw_steps = int(parent_raw_steps)
    if parent_episodes < 0 or parent_raw_steps < 0:
        raise ValueError("parent budget consumption cannot be negative")
    episode_remaining = MAX_CONTROL_EPISODES - parent_episodes
    raw_remaining = MAX_CONTROL_RAW_STEPS - parent_raw_steps
    required_replays = MAX_COLLISION_CASES + MAX_SUCCESS_CASES + 2 * IDENTITY_CASES_PER_OUTCOME
    if episode_remaining < required_replays:
        raise ValueError("remaining episode budget cannot cover 12 baselines and 2 identity replays")
    if raw_remaining < required_replays * MAX_RAW_STEPS_PER_EPISODE:
        raise ValueError("remaining raw-step budget cannot cover 12 baselines and 2 identity replays")
    episode_pair_capacity = (episode_remaining - required_replays) // 2
    raw_pair_capacity = (
        raw_remaining - required_replays * MAX_RAW_STEPS_PER_EPISODE
    ) // (2 * MAX_RAW_STEPS_PER_EPISODE)
    pair_cap = min(
        RECOVERY_INTERVENTION_PAIRS_MAX,
        episode_pair_capacity,
        raw_pair_capacity,
    )
    intervention_episodes = 2 * pair_cap
    local_episode_cap = required_replays + intervention_episodes
    local_raw_upper_bound = local_episode_cap * MAX_RAW_STEPS_PER_EPISODE
    planned_cumulative_episodes = parent_episodes + local_episode_cap
    planned_cumulative_raw_upper_bound = parent_raw_steps + local_raw_upper_bound
    return {
        "parent_episodes_consumed": parent_episodes,
        "parent_raw_control_steps_consumed": parent_raw_steps,
        "episodes_remaining_at_start": episode_remaining,
        "raw_control_steps_remaining_at_start": raw_remaining,
        "baseline_episodes_to_rerecord": MAX_COLLISION_CASES + MAX_SUCCESS_CASES,
        "identity_episodes_to_rerecord": 2 * IDENTITY_CASES_PER_OUTCOME,
        "intervention_pairs_max": pair_cap,
        "intervention_episodes_max": intervention_episodes,
        "recovery_episodes_max": local_episode_cap,
        "recovery_raw_control_steps_upper_bound": local_raw_upper_bound,
        "planned_cumulative_episodes_upper_bound": planned_cumulative_episodes,
        "planned_cumulative_raw_control_steps_upper_bound": planned_cumulative_raw_upper_bound,
        "unused_episode_slots_after_schedule": MAX_CONTROL_EPISODES - planned_cumulative_episodes,
        "unused_raw_control_steps_after_worst_case_schedule": (
            MAX_CONTROL_RAW_STEPS - planned_cumulative_raw_upper_bound
        ),
        "episodes_total_max": MAX_CONTROL_EPISODES,
        "raw_control_steps_total_max": MAX_CONTROL_RAW_STEPS,
    }


def _validate_recovery_parent(
    parent_manifest_path: Path,
    run_root: Path,
    reference: dict[str, Any],
    *,
    expected_parent_root: Path | None = None,
) -> dict[str, Any]:
    """Accept only the recorded aborted run and reserve its spent budget."""
    parent_manifest_path = Path(parent_manifest_path).expanduser().resolve()
    parent_root = parent_manifest_path.parent
    expected_root = Path(expected_parent_root or DEFAULT_RUN_ROOT).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    if parent_manifest_path.name != "experiment_manifest.json":
        raise ValueError("recovery parent must be an experiment_manifest.json file")
    if parent_root != expected_root:
        raise ValueError(f"recovery parent must be the known aborted run at {expected_root}")
    if run_root != parent_root / RECOVERY_SUBDIR:
        raise ValueError(f"recovery output must be a new child named {RECOVERY_SUBDIR}")
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing recovery root: {run_root}")
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    prior_linkage = _read_jsonl(parent_root / "recovery_linkage.jsonl")
    if any(
        Path(str(row.get("recovery_root", ""))).resolve() == run_root
        for row in prior_linkage
    ):
        raise ValueError("this recovery output already has a parent linkage record; refusing reuse")

    manifest = _read_json(parent_manifest_path)
    if manifest.get("schema_version") != "strt_decision_window_intervention_v1":
        raise ValueError("recovery parent manifest has an unexpected schema")
    if manifest.get("status") != "aborted" or manifest.get("training") is not False:
        raise ValueError("recovery parent must be an aborted frozen-policy diagnostic")
    if (
        manifest.get("method") != METHOD
        or manifest.get("scenario") != SCENARIO
        or float(manifest.get("depart_scale", -1)) != DEPART_SCALE
        or manifest.get("traffic_split") != EVAL_SPLIT
    ):
        raise ValueError("recovery parent experiment identity differs")
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("recovery parent checkpoint SHA differs from the frozen reference")
    if manifest.get("checkpoint_sha256") != reference.get("checkpoint_sha256"):
        raise ValueError("recovery parent and current reference checkpoint identities differ")

    summary_path = parent_root / "window_summary.json"
    launcher_path = parent_root / "launcher_status.json"
    runtime_identity_path = parent_root / "runtime_identity.json"
    source_manifest_path = parent_root / "runtime_source_snapshot_manifest.json"
    for required_path in (summary_path, launcher_path, runtime_identity_path, source_manifest_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"recovery parent evidence missing: {required_path}")
    summary = _read_json(summary_path)
    launcher = _read_json(launcher_path)
    runtime_identity = _read_json(runtime_identity_path)
    source_manifest = _read_json(source_manifest_path)
    if summary.get("status") != "aborted" or launcher.get("status") != "aborted":
        raise ValueError("recovery parent summary and launcher must both record aborted status")
    if Path(str(launcher.get("run_root", ""))).resolve() != parent_root:
        raise ValueError("recovery parent launcher path does not match its manifest directory")
    if (
        runtime_identity.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or runtime_identity.get("checkpoint_sha256") != reference.get("checkpoint_sha256")
    ):
        raise ValueError("recovery parent runtime checkpoint SHA differs")

    if source_manifest != manifest.get("runtime_source_snapshot"):
        raise ValueError("recovery parent source snapshot manifest does not match its archived manifest")
    source_rows = {row.get("path"): row for row in source_manifest.get("files", [])}
    source_hashes = manifest.get("runtime_sources") or {}
    for relative in (
        "fast-developer/diagnose_strt_decision_windows_20261003.py",
        "fast-developer/analysis/decision_window_intervention_protocol_20261003.md",
    ):
        row = source_rows.get(relative)
        copied = parent_root / "runtime_source_snapshot" / Path(relative)
        if row is None or not copied.is_file():
            raise ValueError(f"recovery parent archived source missing: {relative}")
        actual_source_sha = sha256_file(copied)
        if actual_source_sha != row.get("sha256") or actual_source_sha != source_hashes.get(relative):
            raise ValueError(f"recovery parent archived source SHA mismatch: {relative}")

    episode_root = parent_root / "episodes"
    baselines = _read_jsonl(episode_root / "baseline.jsonl")
    identities = _read_jsonl(episode_root / "identity.jsonl")
    interventions = _read_jsonl(episode_root / "interventions.jsonl")
    actual_episode_count = len(baselines) + len(identities) + len(interventions)
    actual_raw_steps = sum(
        int(row.get("raw_steps", 0)) for row in [*baselines, *identities, *interventions]
    )
    expected_identity_seed = int(reference["identity_replay_seeds"][0])
    abort_text = str(manifest.get("abort_reason", ""))
    if len(baselines) != 12 or len(identities) != 1 or interventions:
        raise ValueError("recovery parent episode logs do not match the recorded 12+1 pre-intervention failure")
    if any(not all((row.get("reference_match") or {}).values()) for row in baselines):
        raise ValueError("recovery parent includes a baseline that failed archived reference matching")
    identity = identities[0]
    if (
        int(identity.get("seed", -1)) != expected_identity_seed
        or identity.get("identity_match") is not False
        or (identity.get("prefix_error") or {}).get("reason") != "full_physical_state_unavailable"
        or "prefix_full_physical_state_unavailable" not in abort_text
    ):
        raise ValueError("recovery parent identity failure is not the expected physical-prefix gate")
    if (
        actual_episode_count != RECOVERY_PARENT_EPISODES_EXPECTED
        or actual_raw_steps != RECOVERY_PARENT_RAW_STEPS_EXPECTED
        or summary.get("episodes_used") != actual_episode_count
        or summary.get("ego_control_raw_steps_used") != actual_raw_steps
        or launcher.get("completed_episodes") != actual_episode_count
        or launcher.get("ego_control_raw_steps_used") != actual_raw_steps
    ):
        raise ValueError("recovery parent episode/raw-step budget evidence is inconsistent")
    budget = manifest.get("episode_budget") or {}
    if (
        budget.get("total_max") != MAX_CONTROL_EPISODES
        or budget.get("raw_control_steps_total_max") != MAX_CONTROL_RAW_STEPS
        or summary.get("episodes_max") != MAX_CONTROL_EPISODES
        or summary.get("ego_control_raw_steps_max") != MAX_CONTROL_RAW_STEPS
        or launcher.get("ego_control_raw_steps_max") != MAX_CONTROL_RAW_STEPS
    ):
        raise ValueError("recovery parent declared budget differs from this protocol")

    return {
        "parent_root": str(parent_root),
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_episodes_consumed": actual_episode_count,
        "parent_raw_control_steps_consumed": actual_raw_steps,
        "budget_plan": _recovery_budget_plan(actual_episode_count, actual_raw_steps),
    }


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


def _append_recovery_linkage(
    recovery_context: dict[str, Any] | None,
    run_root: Path,
    *,
    event: str,
    status: str,
    local_episodes: int | None = None,
    local_raw_steps: int | None = None,
) -> None:
    if recovery_context is None:
        return
    manifest_path = run_root / "experiment_manifest.json"
    _append_jsonl(Path(recovery_context["parent_root"]) / "recovery_linkage.jsonl", {
        "event": event,
        "timestamp": time.time(),
        "parent_manifest": recovery_context["parent_manifest"],
        "parent_manifest_sha256": recovery_context["parent_manifest_sha256"],
        "recovery_root": str(Path(run_root).resolve()),
        "recovery_manifest": str(manifest_path.resolve()),
        "recovery_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "status": status,
        "recovery_episodes_used": local_episodes,
        "recovery_raw_control_steps_used": local_raw_steps,
    })


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
            persons = [str(value) for value in connection.person.getIDList()]
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
    if not verified.get("physical_state_complete") and not observer_snapshot_supplied:
        # Reset can precede the first raw-step observer callback. Refresh only
        # from the environment's already-recorded current actor cache and keep
        # diagnostic bookkeeping unchanged. Current road/lane IDs are then
        # queried read-only for every live actor below.
        initial_reasons = verified.get("physical_state_incomplete_reasons")
        refreshed = _snapshot_from_environment_cache(env)
        if refreshed is not None:
            verified = verify_physical_snapshot_membership(
                refreshed,
                current_vehicle_ids=vehicles,
                current_person_ids=persons,
                current_raw_step=current_raw_step,
                query_errors=errors,
            )
            verified["initial_snapshot_incomplete_reasons"] = initial_reasons
    return _supplement_live_road_lane_identity(
        verified,
        env,
        current_vehicle_ids=vehicles,
        current_person_ids=persons,
        current_raw_step=current_raw_step,
        query_errors=errors,
    )


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
        history_index_audit = _trajectory_history_index_audit(observation, np=np)
        index = len(self.trace)
        # Only a raw-step observer capture suppresses reset-cache fallback.
        # Passing the recorder's reset snapshot here would incorrectly make
        # _verified_physical_snapshot treat it as an observer-supplied state.
        physical = _verified_physical_snapshot(
            self.recorder, self.env, self.latest_full_physical_snapshot
        )
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
            "observation_history_index_audit": history_index_audit,
            "policy_action_sha256": stable_hash(action_array),
            "physical_state_sha256": stable_hash(physical_fingerprint_payload(physical)) if physical is not None else None,
            "physical_state_complete": (physical or {}).get("physical_state_complete") is True,
            "physical_state_scope": (physical or {}).get("snapshot_scope"),
            "physical_state_capture_source": (physical or {}).get("snapshot_capture_source"),
            "physical_actor_count": (physical or {}).get("physical_actor_count"),
            "physical_state_incomplete_reasons": (physical or {}).get("physical_state_incomplete_reasons"),
            "physical_actor_state_errors": (physical or {}).get("physical_actor_state_errors"),
            "physical_membership": (physical or {}).get("physical_membership"),
            "physical_identity_query": (physical or {}).get("physical_identity_query"),
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
    record["history_index_audit_summary"] = _summarize_history_index_audit([record])
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


def _write_manifest(run_root: Path, cases, reference, recovery_context=None):
    run_root.mkdir(parents=True, exist_ok=False)
    source_snapshot = _archive_runtime_sources(run_root)
    identity_seeds = reference["identity_replay_seeds"]
    recovery_budget = (recovery_context or {}).get("budget_plan") or {}
    intervention_max = recovery_budget.get("intervention_episodes_max", 72)
    run_episode_max = recovery_budget.get("recovery_episodes_max", MAX_CONTROL_EPISODES)
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
        "recovery": recovery_context,
        "runtime_sources": _runtime_sources(),
        "runtime_source_snapshot": source_snapshot,
        "episode_budget": {
            "baseline_max": 12,
            "identity_max": 2,
            "intervention_max": intervention_max,
            "run_episode_max": run_episode_max,
            "total_max": MAX_CONTROL_EPISODES,
            "raw_control_steps_per_episode_max": MAX_RAW_STEPS_PER_EPISODE,
            "raw_control_steps_total_max": MAX_CONTROL_RAW_STEPS,
            "parent_episodes_consumed": recovery_budget.get("parent_episodes_consumed", 0),
            "parent_raw_control_steps_consumed": recovery_budget.get(
                "parent_raw_control_steps_consumed", 0
            ),
            "intervention_pairs_max": recovery_budget.get(
                "intervention_pairs_max", (MAX_CONTROL_EPISODES - 14) // 2
            ),
            "planned_run_episode_max": run_episode_max,
            "planned_cumulative_episode_upper_bound": recovery_budget.get(
                "planned_cumulative_episodes_upper_bound", run_episode_max
            ),
            "planned_cumulative_raw_control_steps_upper_bound": recovery_budget.get(
                "planned_cumulative_raw_control_steps_upper_bound",
                run_episode_max * MAX_RAW_STEPS_PER_EPISODE,
            ),
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
            "physical_state": "exact at every pre-action prefix decision; active actor ID sets, current finite kinematics/dimensions/timestamps, and current road/lane IDs read through read-only TraCI calls; hidden route fields excluded",
            "road_lane_query": "vehicle/person getRoadID and getLaneID for every live actor; simulator time, raw-step counter, and live ID sets must remain unchanged before/after the queries",
            "mismatch_action": "do not apply intervention; finish that allocated episode for cleanup/logging, mark invalid and stop the suite",
        },
        "history_index_audit": {
            "observation_key": "trajectory",
            "mask_rule": "trajectory[..., 0] != 0 using algos.sb3_torch.features._nonzero_mask",
            "per_actor_fields": [
                "valid_count", "count_minus_one_index", "count_minus_one_index_is_model_padding",
                "true_last_valid_index", "index_delta_true_minus_count_minus_one",
            ],
            "actor_roles": "ego slot 0 and social slots 1..N-1; slots only, IDs are not present in the policy observation",
            "audit_only": "calculated after deterministic policy prediction; does not modify policy input or action and is excluded from prefix hashes",
            "scope_limit": "newly rerecorded outcome-stratified baseline cases, not the full evaluation or training distribution",
        },
        "runtime_source_sha256": _runtime_sources(),
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _run_start(run_root: Path, rows, cases, reference, recovery_context=None):
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing window-diagnostic root: {run_root}")
    manifest = _write_manifest(run_root, cases, reference, recovery_context)
    (run_root / "episodes").mkdir(parents=True, exist_ok=True)
    budget_plan = (recovery_context or {}).get("budget_plan") or {}
    parent_episodes = int(budget_plan.get("parent_episodes_consumed", 0))
    parent_raw_steps = int(budget_plan.get("parent_raw_control_steps_consumed", 0))
    local_episode_cap = int(budget_plan.get("recovery_episodes_max", MAX_CONTROL_EPISODES))
    local_raw_cap = min(
        MAX_CONTROL_RAW_STEPS - parent_raw_steps,
        int(budget_plan.get("recovery_raw_control_steps_upper_bound", MAX_CONTROL_RAW_STEPS)),
    )
    pair_cap = int(budget_plan.get(
        "intervention_pairs_max", (MAX_CONTROL_EPISODES - 14) // 2
    ))
    status_path = run_root / "launcher_status.json"
    status = {
        "status": "starting",
        "run_root": str(run_root),
        "completed_episodes": 0,
        "ego_control_raw_steps_used": 0,
        "parent_episodes_consumed": parent_episodes,
        "parent_raw_control_steps_consumed": parent_raw_steps,
        "cumulative_episodes_used": parent_episodes,
        "cumulative_raw_control_steps_used": parent_raw_steps,
        "cumulative_episodes_max": MAX_CONTROL_EPISODES,
        "cumulative_raw_control_steps_max": MAX_CONTROL_RAW_STEPS,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    _append_recovery_linkage(
        recovery_context, run_root, event="started", status="starting",
        local_episodes=0, local_raw_steps=0,
    )
    base = d1 = evaluator = source_augmentation = env = recorder = model = np = torch = None
    baseline_by_seed: dict[int, dict[str, Any]] = {}
    reference_by_seed = {int(row["seed"]): row for row in rows}
    completed_episode_count = 0
    raw_control_steps = 0
    abort_reason = None
    baseline_history_index_audit_summary = None

    def persist_progress(current_phase: str, current_seed: int | None = None) -> None:
        status.update(
            status="running",
            completed_episodes=completed_episode_count,
            ego_control_raw_steps_used=raw_control_steps,
            ego_control_raw_steps_max=local_raw_cap,
            current_phase=current_phase,
            current_seed=current_seed,
            parent_episodes_consumed=parent_episodes,
            parent_raw_control_steps_consumed=parent_raw_steps,
            cumulative_episodes_used=parent_episodes + completed_episode_count,
            cumulative_raw_control_steps_used=parent_raw_steps + raw_control_steps,
            cumulative_episodes_max=MAX_CONTROL_EPISODES,
            cumulative_raw_control_steps_max=MAX_CONTROL_RAW_STEPS,
            recovery_episodes_max=local_episode_cap,
            recovery_raw_control_steps_max=local_raw_cap,
        )
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def require_episode_capacity(phase: str, seed: int | None) -> None:
        if completed_episode_count >= local_episode_cap:
            raise RuntimeError(f"local recovery episode budget exhausted before {phase} seed={seed}")
        if parent_episodes + completed_episode_count >= MAX_CONTROL_EPISODES:
            raise RuntimeError(f"cumulative episode budget exhausted before {phase} seed={seed}")
        if parent_raw_steps + raw_control_steps + MAX_RAW_STEPS_PER_EPISODE > MAX_CONTROL_RAW_STEPS:
            raise RuntimeError(f"cumulative raw-step reserve exhausted before {phase} seed={seed}")
        if raw_control_steps + MAX_RAW_STEPS_PER_EPISODE > local_raw_cap:
            raise RuntimeError(f"local raw-step reserve exhausted before {phase} seed={seed}")

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
            require_episode_capacity("baseline", seed)
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
            if (
                completed_episode_count > local_episode_cap
                or raw_control_steps > local_raw_cap
                or parent_episodes + completed_episode_count > MAX_CONTROL_EPISODES
                or parent_raw_steps + raw_control_steps > MAX_CONTROL_RAW_STEPS
            ):
                raise RuntimeError("hard episode/raw-step budget exceeded")

        baseline_history_index_audit_summary = _summarize_history_index_audit(
            [baseline_by_seed[int(case["seed"])] for case in cases]
        )
        (run_root / "baseline_history_index_audit_summary.json").write_text(
            json.dumps(
                baseline_history_index_audit_summary,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        manifest["baseline_history_index_audit_summary"] = baseline_history_index_audit_summary
        (run_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        # Two full identity replays are a gate before any action change.
        identity_seeds = reference["identity_replay_seeds"]
        for seed in identity_seeds:
            baseline = baseline_by_seed[int(seed)]
            require_episode_capacity("identity", int(seed))
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
            if (
                completed_episode_count > local_episode_cap
                or raw_control_steps > local_raw_cap
                or parent_episodes + completed_episode_count > MAX_CONTROL_EPISODES
                or parent_raw_steps + raw_control_steps > MAX_CONTROL_RAW_STEPS
            ):
                raise RuntimeError("hard episode/raw-step budget exceeded")
            if not replay["identity_match"]:
                raise PrefixMismatch(f"identity replay failed for seed={seed}: {mismatch}")

        schedule = _build_intervention_schedule(
            cases, baseline_by_seed, pair_limit=pair_cap
        )
        manifest["intervention_schedule"] = schedule
        manifest["status"] = "intervention_schedule_prepared"
        (run_root / "intervention_schedule.json").write_text(
            json.dumps(schedule, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (run_root / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _append_recovery_linkage(
            recovery_context, run_root, event="schedule_prepared",
            status="intervention_schedule_prepared",
            local_episodes=completed_episode_count,
            local_raw_steps=raw_control_steps,
        )
        persist_progress("intervention_schedule_prepared")

        # Replay only the precommitted phase-first pair slots; gaps do not refill the cap.
        for pair in schedule["pairs"]:
            seed = int(pair["seed"])
            baseline = baseline_by_seed[seed]
            anchor_index = int(pair["anchor_index"])
            anchor = next(
                item for item in (baseline.get("anchor_plan") or {}).get("unique", [])
                if int(item["decision_index"]) == anchor_index
            )
            for target in pair["target_speeds_mps"]:
                require_episode_capacity("intervention", seed)
                intervention = _run_episode(
                    env=env, recorder=recorder, model=model, evaluator=evaluator,
                    source_augmentation=source_augmentation, np=np, torch=torch,
                    snapshot_observer=snapshot_observer,
                    seed=seed, traffic_episode_index=seed - SEED_START,
                    output_episode_index=completed_episode_count,
                    baseline_trace=baseline["trace"], anchor_index=anchor_index,
                    target_speed_mps=float(target),
                )
                intervention["anchor_phases"] = anchor["phases"]
                intervention["anchor_baseline"] = anchor
                intervention["schedule_pair_index"] = pair["pair_index"]
                intervention["schedule_slot"] = pair["schedule_slot"]
                intervention["scheduled_phase"] = pair["phase"]
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
                if (
                    completed_episode_count > local_episode_cap
                    or raw_control_steps > local_raw_cap
                    or parent_episodes + completed_episode_count > MAX_CONTROL_EPISODES
                    or parent_raw_steps + raw_control_steps > MAX_CONTROL_RAW_STEPS
                ):
                    raise RuntimeError("hard episode/raw-step budget exceeded")

        after_fingerprint = _model_fingerprint(model)
        if before_fingerprint != after_fingerprint:
            raise RuntimeError("frozen policy state or update counter changed during intervention run")
        if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
            raise RuntimeError("frozen checkpoint bytes changed during intervention run")
        summary = {
            "status": "complete",
            "episodes_used": completed_episode_count,
            "episodes_max": local_episode_cap,
            "ego_control_raw_steps_used": raw_control_steps,
            "ego_control_raw_steps_max": local_raw_cap,
            "parent_episodes_consumed": parent_episodes,
            "parent_raw_control_steps_consumed": parent_raw_steps,
            "cumulative_episodes_used": parent_episodes + completed_episode_count,
            "cumulative_raw_control_steps_used": parent_raw_steps + raw_control_steps,
            "cumulative_episodes_max": MAX_CONTROL_EPISODES,
            "cumulative_raw_control_steps_max": MAX_CONTROL_RAW_STEPS,
            "intervention_pairs_reserved": schedule["reserved_slot_count"],
            "intervention_pairs_executed": schedule["pairs_planned"],
            "intervention_episodes_planned": schedule["episodes_planned"],
            "intervention_episodes_used": max(0, completed_episode_count - len(cases) - len(identity_seeds)),
            "baseline_history_index_audit_summary": baseline_history_index_audit_summary,
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
        _append_recovery_linkage(
            recovery_context, run_root, event="complete", status="complete",
            local_episodes=completed_episode_count, local_raw_steps=raw_control_steps,
        )
        status.update(
            status="complete",
            completed_episodes=completed_episode_count,
            ego_control_raw_steps_used=raw_control_steps,
            cumulative_episodes_used=parent_episodes + completed_episode_count,
            cumulative_raw_control_steps_used=parent_raw_steps + raw_control_steps,
        )
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
            "episodes_max": local_episode_cap,
            "ego_control_raw_steps_max": local_raw_cap,
            "parent_episodes_consumed": parent_episodes,
            "parent_raw_control_steps_consumed": parent_raw_steps,
            "cumulative_episodes_used": parent_episodes + completed_episode_count,
            "cumulative_raw_control_steps_used": parent_raw_steps + raw_control_steps,
            "cumulative_episodes_max": MAX_CONTROL_EPISODES,
            "cumulative_raw_control_steps_max": MAX_CONTROL_RAW_STEPS,
            "baseline_history_index_audit_summary": baseline_history_index_audit_summary,
            "abort_reason": abort_reason,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _append_recovery_linkage(
            recovery_context, run_root, event="aborted", status="aborted",
            local_episodes=completed_episode_count, local_raw_steps=raw_control_steps,
        )
        status.update(
            status="aborted",
            completed_episodes=completed_episode_count,
            ego_control_raw_steps_used=raw_control_steps,
            cumulative_episodes_used=parent_episodes + completed_episode_count,
            cumulative_raw_control_steps_used=parent_raw_steps + raw_control_steps,
            error=abort_reason,
        )
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
    parser.add_argument(
        "--recover-aborted-manifest",
        type=Path,
        help="validate and reserve the budget from the known aborted parent run",
    )
    parser.add_argument("--start", action="store_true", help="explicitly run the bounded SUMO diagnostic")
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    runs_root = (WORKSPACE / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        parser.error(f"run root must be a child of {runs_root}")
    try:
        rows, cases, reference = _validate_reference()
        recovery_context = None
        if args.recover_aborted_manifest is not None:
            recovery_context = _validate_recovery_parent(
                args.recover_aborted_manifest,
                run_root,
                reference,
            )
        elif run_root == DEFAULT_RUN_ROOT / RECOVERY_SUBDIR:
            raise ValueError(
                "the reserved recovery01 directory requires --recover-aborted-manifest"
            )
        if run_root.exists():
            raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
        budget_plan = (recovery_context or {}).get("budget_plan") or {
            "parent_episodes_consumed": 0,
            "parent_raw_control_steps_consumed": 0,
            "intervention_pairs_max": (MAX_CONTROL_EPISODES - 14) // 2,
            "intervention_episodes_max": 2 * ((MAX_CONTROL_EPISODES - 14) // 2),
            "recovery_episodes_max": MAX_CONTROL_EPISODES,
            "recovery_raw_control_steps_upper_bound": MAX_CONTROL_RAW_STEPS,
            "planned_cumulative_episodes_upper_bound": MAX_CONTROL_EPISODES,
            "planned_cumulative_raw_control_steps_upper_bound": MAX_CONTROL_RAW_STEPS,
        }
        plan = {
            "mode": "execute" if args.start else "read_only_preview",
            "training_started": False,
            "sumo_started": bool(args.start),
            "run_root": str(run_root),
            "recovery_parent": recovery_context,
            "checkpoint": reference["checkpoint"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "reference_summary": reference["summary"],
            "cases": cases,
            "identity_replay_seeds": reference["identity_replay_seeds"],
            "budgets": {
                **budget_plan,
                "episodes_max": budget_plan["recovery_episodes_max"],
                "ego_control_raw_steps_max": budget_plan[
                    "recovery_raw_control_steps_upper_bound"
                ],
                "cumulative_episodes_max": MAX_CONTROL_EPISODES,
                "cumulative_raw_control_steps_max": MAX_CONTROL_RAW_STEPS,
                "per_episode_raw_steps_max": MAX_RAW_STEPS_PER_EPISODE,
                "targets_mps": list(INTERVENTION_TARGET_SPEEDS_MPS),
                "decisions_per_intervention": INTERVENTION_DECISIONS,
                "schedule_policy": {
                    "phase_priority": list(ANCHOR_PHASE_ORDER),
                    "seed_order": "ascending logical validation seed",
                    "fixed_slots": True,
                    "missing_or_merged_slots_refilled": False,
                },
            },
        }
        if not args.start:
            print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        return _run_start(run_root, rows, cases, reference, recovery_context)
    except Exception as exc:
        print(f"window-runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
