"""Passive C8/C9 audit of rollout observations; never advances the simulator.

Counts describe repeated decision/history-slot observations, not unique actors or
independent traffic events. The legacy x != 0 presence mask is deliberately kept.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from collections.abc import Mapping

import gymnasium as gym
import numpy as np


def _history_group(states: np.ndarray) -> dict:
    valid = states[..., 0] != 0
    horizon = valid.shape[-1]
    count = valid.sum(axis=-1)
    present = count > 0
    index = np.arange(horizon)
    last = np.where(valid, index, -1).max(axis=-1)
    first = np.where(valid, index, horizon).min(axis=-1)
    old = np.maximum(count - 1, 0)
    old_valid = np.take_along_axis(valid, old[..., None], axis=-1)[..., 0]
    mismatch = present & (old != last)
    holes = present & (count < last - first + 1)
    nonzero_other = np.any(states[..., 1:] != 0, axis=-1)
    return {
        "history_observations": int(count.size),
        "nonempty": int(present.sum()),
        "empty": int((~present).sum()),
        "short_nonempty": int((present & (count < horizon)).sum()),
        "full": int((count == horizon).sum()),
        "valid_frame_observations": int(count.sum()),
        "frame_slots": int(valid.size),
        "left_padding": int((present & (first > 0)).sum()),
        "right_padding": int((present & (last < horizon - 1)).sum()),
        "internal_gaps": int(holes.sum()),
        "old_index_mismatch": int(mismatch.sum()),
        "old_selected_padding": int((present & ~old_valid).sum()),
        "old_selected_earlier_valid": int((mismatch & old_valid).sum()),
        "index_lag_slots_sum": int(np.where(mismatch, last - old, 0).sum()),
        "x_zero_other_fields_nonzero_slots": int(((~valid) & nonzero_other).sum()),
    }


def audit_trajectory(trajectory, *, velocity_contract: str = "smarts",
                     geometry_active: bool = True, horizon_seconds: float = 20.0) -> dict:
    """Audit a canonical [actor, history, x/y/heading/vx/vy] observation.

    Geometry denominators are unordered, valid, same-history-slot actor pairs.
    Velocity conversion is ONLY for this read-only calculation. No input writes.
    No vehicle extents are available here: no collision/TTC classification is made.
    """
    states = np.asarray(trajectory)
    if states.ndim != 3 or states.shape[-1] < 5 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError(f"Expected [actor, history, >=5], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("Non-finite rollout observation")
    if velocity_contract not in {"smarts", "cartesian"}:
        raise ValueError(f"Unsupported velocity contract: {velocity_contract}")
    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be finite and positive")
    result = {
        "ego": _history_group(states[:1]),
        "social": _history_group(states[1:]),
        "geometry": {"active": bool(geometry_active), "velocity_contract": velocity_contract,
                     "horizon_seconds": float(horizon_seconds),
                     "time_semantics": "constant_velocity_closest_approach_not_TTC"},
    }
    if not geometry_active:
        result["geometry"]["status"] = "NA_disabled_by_method"
        return result
    pairs = np.triu_indices(states.shape[0], k=1)
    pair_valid = (states[pairs[0], :, 0] != 0) & (states[pairs[1], :, 0] != 0)
    r = (states[pairs[0], :, :2] - states[pairs[1], :, :2])[pair_valid].astype(np.float64)
    old_v = (states[pairs[0], :, 3:5] - states[pairs[1], :, 3:5])[pair_valid].astype(np.float64)
    v = np.stack((-old_v[:, 1], old_v[:, 0]), axis=-1) if velocity_contract == "smarts" else old_v
    distance = np.linalg.norm(r, axis=-1)
    speed2 = np.sum(v * v, axis=-1)
    moving = speed2 > 1e-8
    noncoincident = distance > 1e-8
    dot = np.sum(r * v, axis=-1)
    old_dot = np.sum(r * old_v, axis=-1)
    closing = np.divide(-dot, distance, out=np.zeros_like(dot), where=noncoincident)
    old_closing = np.divide(-old_dot, distance, out=np.zeros_like(dot), where=noncoincident)
    tau = np.divide(-dot, speed2, out=np.zeros_like(dot), where=moving)
    bounded_time = np.clip(tau, 0, horizon_seconds)
    bounded_miss = np.linalg.norm(r + bounded_time[:, None] * v, axis=-1)
    closest_miss = np.linalg.norm(r + tau[:, None] * v, axis=-1)
    g = result["geometry"]
    g.update({
        "status": "ok",
        "pair_history_observations": int(len(r)),
        "relative_motion_defined": int(moving.sum()),
        "zero_relative_speed": int((~moving).sum()),
        "coincident_position": int((~noncoincident).sum()),
        "closing_defined": int(noncoincident.sum()),
        "approaching": int((noncoincident & (closing > 1e-8)).sum()),
        "receding": int((noncoincident & (closing < -1e-8)).sum()),
        "closing_abs_old_new_difference_sum_mps": float(np.abs(closing - old_closing)[noncoincident].sum()),
        "closing_sign_reversal": int((noncoincident & (closing * old_closing < -1e-8)).sum()),
        "old_zero_new_nonzero_closing": int((noncoincident & (np.abs(old_closing) <= 1e-8) & (np.abs(closing) > 1e-8)).sum()),
        "closest_time_in_future_horizon": int((moving & (tau >= 0) & (tau <= horizon_seconds)).sum()),
        "closest_time_past": int((moving & (tau < 0)).sum()),
        "closest_time_beyond_horizon": int((moving & (tau > horizon_seconds)).sum()),
        "bounded_miss_distance_sum_m": float(bounded_miss.sum()),
        "unconstrained_miss_distance_defined_sum_m": float(closest_miss[moving].sum()),
    })
    return result


def add_audit_counts(total: dict, sample: dict) -> None:
    """Aggregate exact counters/sums; never average per-decision percentages."""
    for group in ("ego", "social"):
        dest = total.setdefault(group, {})
        for key, value in sample[group].items():
            dest[key] = dest.get(key, 0) + value
    dest = total.setdefault("geometry", {})
    for key, value in sample["geometry"].items():
        if key in {"active", "velocity_contract", "horizon_seconds", "time_semantics", "status"}:
            dest[key] = value
        elif isinstance(value, (int, float)):
            dest[key] = dest.get(key, 0) + value


def audit_rates(total: dict) -> dict:
    result = {}
    for group in ("ego", "social"):
        counts = total.get(group, {})
        n = counts.get("nonempty", 0)
        result[group] = {
            "denominator_nonempty_history_observations": n,
            "short_nonempty_rate": counts.get("short_nonempty", 0) / n if n else None,
            "old_index_mismatch_rate": counts.get("old_index_mismatch", 0) / n if n else None,
            "old_selected_padding_rate": counts.get("old_selected_padding", 0) / n if n else None,
        }
    geometry = total.get("geometry", {})
    n = geometry.get("closing_defined", 0)
    result["geometry"] = {
        "denominator_closing_defined_pair_history_observations": n,
        "mean_abs_old_new_closing_difference_mps": (
            geometry.get("closing_abs_old_new_difference_sum_mps", 0) / n if n else None),
    }
    return result


class TrajectoryHistoryAuditWrapper(gym.Wrapper):
    """One audit row per real action decision, with pre-action observation.

    reset caches the observation; step logs that cached state and delegates exactly
    once. Post-terminal observations are never counted as policy decisions. All
    state/identity metadata are passive reads of the existing environment cache.
    """

    def __init__(self, env, *, phase, log_path=None, run_dir=None, method="unknown",
                 geometry_active=True, velocity_contract="smarts", trajectory_shape=None):
        super().__init__(env)
        if log_path is None:
            if run_dir is None:
                raise ValueError("log_path or run_dir is required")
            log_path = Path(run_dir) / "diagnostics" / phase / "trajectory_history_audit.jsonl"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = str(phase)
        self.method = str(method)
        self.geometry_active = bool(geometry_active)
        self.velocity_contract = str(velocity_contract)
        self.trajectory_shape = tuple(trajectory_shape) if trajectory_shape is not None else None
        self.session_id = f"{time.time_ns()}"
        self.episode = -1
        self.decision = 0
        self.rows = 0
        self.completed_episodes = 0
        self.errors = 0
        self.raw_steps_counted = 0
        self.raw_steps_unknown_decisions = 0
        self.total = {}
        self.metadata_total = {"decisions_with_actor_history_cache": 0,
                               "actor_histories_available": 0,
                               "actor_histories_nonempty": 0,
                               "actor_histories_short_nonempty": 0}
        self._pre_audit = None
        self._pre_meta = None
        self._file = self.log_path.open("a", encoding="utf-8", buffering=65536)
        self._closed = False
        self._write({"record_type": "session", "schema": "d1_contractfix_trajectory_audit_v1",
                     "mask_contract": "legacy_x_nonzero_proxy",
                     "sampling": "every_real_preaction_decision; no_replay_or_shadow",
                     "geometry_unit": "unordered_same_history_slot_pairs_repeated_across_decisions",
                     "geometry_is_collision_risk": False})

    def _write(self, row):
        payload = {"phase": self.phase, "method": self.method, "session_id": self.session_id, **row}
        self._file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def _snapshot_history_cache(self, trajectory):
        source = self.env.unwrapped
        keys = getattr(source, "_last_observation_actor_keys", None)
        histories = getattr(source, "_histories", None)
        first = getattr(source, "_first_seen_steps", {})
        last = getattr(source, "_last_seen_steps", {})
        tick = getattr(source, "_history_timestep", None)
        raw = getattr(source, "_raw_steps", None)
        meta = {"environment_raw_steps": int(raw) if raw is not None else None,
                "history_timestep": int(tick) if tick is not None else None,
                "history_timestep_semantics": "next_append_index",
                "actor_history_cache_available": keys is not None and histories is not None,
                "actors": []}
        if keys is not None and histories is not None:
            if len(keys) > trajectory.shape[0]:
                raise ValueError("Actor cache has more slots than observation")
            for slot, key in enumerate(keys):
                history = histories.get(key)
                n = len(history) if history is not None else None
                first_seen = first.get(key)
                last_seen = last.get(key)
                meta["actors"].append({
                    "slot": slot, "actor_key": str(key),
                    "cached_history_length": n,
                    "first_seen_history_timestep": int(first_seen) if first_seen is not None else None,
                    "last_seen_history_timestep": int(last_seen) if last_seen is not None else None,
                    "tracked_age_slots": int(tick - first_seen) if tick is not None and first_seen is not None else None,
                    "short_cached_history": bool(0 < n < trajectory.shape[1]) if n is not None else None,
                })
        return meta

    def _cache(self, observation):
        if not isinstance(observation, Mapping) or "trajectory" not in observation:
            raise ValueError("D1 contract audit requires a trajectory observation key")
        trajectory = np.asarray(observation["trajectory"])
        # This wrapper is inside vectorization; batched/flattened input is a setup error.
        if trajectory.ndim != 3:
            raise ValueError(f"Expected unbatched trajectory, got {trajectory.shape}")
        if self.trajectory_shape is not None and trajectory.shape != self.trajectory_shape:
            raise ValueError(f"Expected trajectory shape {self.trajectory_shape}, got {trajectory.shape}")
        self._pre_audit = audit_trajectory(trajectory, velocity_contract=self.velocity_contract,
                                         geometry_active=self.geometry_active)
        self._pre_meta = self._snapshot_history_cache(trajectory)

    def _summary(self):
        self._file.flush()
        data = {"schema": "d1_contractfix_trajectory_audit_v1", "phase": self.phase,
                "method": self.method, "session_id": self.session_id,
                "real_decision_rows": self.rows, "started_episodes": self.episode + 1,
                "completed_episodes": self.completed_episodes,
                "errors": self.errors, "raw_steps_counted": self.raw_steps_counted,
                "raw_steps_unknown_decisions": self.raw_steps_unknown_decisions,
                "mask_contract": "legacy_x_nonzero_proxy", "counts": self.total,
                "rates": audit_rates(self.total), "actor_cache_counts": self.metadata_total}
        path = self.log_path.with_suffix(".summary.json")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        self.episode += 1
        self.decision = 0
        try:
            self._cache(obs)
        except Exception as error:
            self.errors += 1
            self._write({"record_type": "error", "stage": "reset", "error": repr(error)})
            self._summary()
            raise
        return result

    def step(self, action):
        if self._pre_audit is None:
            raise RuntimeError("reset required before step")
        result = self.env.step(action)
        if len(result) == 5:
            obs, _, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            obs, _, done, info = result
            done = bool(done)
        else:
            raise ValueError("Unexpected environment step signature")
        raw_delta = info.get("raw_steps_executed") if isinstance(info, Mapping) else None
        if raw_delta is None:
            post_raw = getattr(self.env.unwrapped, "_raw_steps", None)
            pre_raw = self._pre_meta.get("environment_raw_steps")
            if post_raw is not None and pre_raw is not None and post_raw >= pre_raw:
                raw_delta = int(post_raw - pre_raw)
        if raw_delta is None:
            self.raw_steps_unknown_decisions += 1
        else:
            raw_delta = int(raw_delta)
            self.raw_steps_counted += int(raw_delta)
        self._write({"record_type": "decision", "episode": self.episode,
                     "decision": self.decision, "raw_steps_executed": raw_delta,
                     "terminal_after_action": done, "preaction_metadata": self._pre_meta,
                     "audit": self._pre_audit})
        add_audit_counts(self.total, self._pre_audit)
        meta = self._pre_meta
        if meta["actor_history_cache_available"]:
            self.metadata_total["decisions_with_actor_history_cache"] += 1
        for actor in meta["actors"]:
            n = actor["cached_history_length"]
            if n is not None:
                self.metadata_total["actor_histories_available"] += 1
                self.metadata_total["actor_histories_nonempty"] += int(n > 0)
                self.metadata_total["actor_histories_short_nonempty"] += int(actor["short_cached_history"])
        self.rows += 1
        self.decision += 1
        if done:
            self.completed_episodes += 1
            self._pre_audit = None
            self._pre_meta = None
        else:
            try:
                self._cache(obs)
            except Exception as error:
                self.errors += 1
                self._write({"record_type": "error", "stage": "step", "error": repr(error)})
                self._summary()
                raise
        if done or self.rows % 100 == 0:
            self._summary()
        return result

    def close(self):
        if not self._closed:
            self._summary()
            self._file.close()
            self._closed = True
        return self.env.close()
