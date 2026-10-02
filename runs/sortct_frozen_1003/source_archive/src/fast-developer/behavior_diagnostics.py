"""Read-only behavior telemetry. Nothing in this module is a policy input.

Risk estimates assume constant planar velocity and heading. In particular,
cv_obb_ttc_s is a forecast for translating rectangles, NOT a SUMO collision
event or ground-truth intersection TTC. Unknown values are JSON null.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import time
from pathlib import Path

SCHEMA_VERSION = 1
BEHAVIOR_TELEMETRY_PROTOCOL = "route_lane_behavior_v1"
ROUTE_LANE_STALL_SPEED_MPS = 0.1
EPS = 1e-9


def _clean(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if hasattr(value, "item"):
        return _clean(value.item())
    return str(value)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _vec(actor, key):
    value = actor.get(key)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    result = tuple(_number(x) for x in value)
    return None if None in result else result


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _axes(actor):
    heading = _number(actor.get("heading"))
    length, width = _number(actor.get("length")), _number(actor.get("width"))
    if heading is None or length is None or width is None or min(length, width) <= 0:
        return None
    return ((math.cos(heading), math.sin(heading)),
            (-math.sin(heading), math.cos(heading)), length / 2, width / 2)


def cv_obb_ttc(ego, other, horizon=10.0):
    """Continuous separating-axis interval intersection for fixed headings.

    Positions must be box centers, headings radians counterclockwise from +x,
    and velocities m/s. Returns None if unavailable or no overlap by horizon.
    """
    p, q, v, w = (_vec(ego, "position"), _vec(other, "position"),
                   _vec(ego, "velocity"), _vec(other, "velocity"))
    a, b = _axes(ego), _axes(other)
    if any(x is None for x in (p, q, v, w, a, b)):
        return None
    delta = (q[0] - p[0], q[1] - p[1])
    relative = (w[0] - v[0], w[1] - v[1])
    enter, leave = 0.0, float(horizon)
    for axis in (a[0], a[1], b[0], b[1]):
        radius = sum(abs(_dot(axis, x[0])) * x[2] +
                     abs(_dot(axis, x[1])) * x[3] for x in (a, b))
        distance, speed = _dot(delta, axis), _dot(relative, axis)
        if abs(speed) <= EPS:
            if abs(distance) > radius:
                return None
            continue
        t0, t1 = (-radius - distance) / speed, (radius - distance) / speed
        enter, leave = max(enter, min(t0, t1)), min(leave, max(t0, t1))
        if enter > leave:
            return None
    return max(0.0, enter) if leave >= 0.0 and enter <= horizon else None


def pair_metrics(ego, other, horizon=10.0):
    p, q, v, w = (_vec(ego, "position"), _vec(other, "position"),
                   _vec(ego, "velocity"), _vec(other, "velocity"))
    if any(x is None for x in (p, q, v, w)):
        return {"available": False, "reason": "missing_position_or_velocity"}
    r, u = (q[0] - p[0], q[1] - p[1]), (w[0] - v[0], w[1] - v[1])
    distance, vv = math.hypot(*r), _dot(u, u)
    closing = -_dot(r, u) / distance if distance > EPS else 0.0
    tcpa = max(0.0, min(float(horizon), -_dot(r, u) / vv)) if vv > EPS else 0.0
    dcpa = math.hypot(r[0] + tcpa * u[0], r[1] + tcpa * u[1])
    geometry_available = _axes(ego) is not None and _axes(other) is not None
    ttc = cv_obb_ttc(ego, other, horizon) if geometry_available else None
    lane_a, lane_b = ego.get("lane_id"), other.get("lane_id")
    return {"available": True, "center_distance_m": distance,
            "relative_position_m": list(r), "relative_velocity_mps": list(u),
            "radial_closing_speed_mps": closing,
            "radial_center_ttc_s": distance / closing if closing > EPS else None,
            "time_to_cpa_within_horizon_s": tcpa, "cpa_center_distance_m": dcpa,
            "cv_obb_geometry_available": geometry_available,
            "cv_obb_ttc_s": ttc, "forecast_horizon_s": horizon,
            "same_lane": lane_a == lane_b if lane_a and lane_b else None,
            "candidate_conflict_cv": ttc is not None}


class BehaviorDiagnosticsRecorder:
    def __init__(self, output_dir, phase, method, metadata=None,
                 radius_m=120.0, max_neighbors=32, horizon_s=10.0):
        self.path = Path(output_dir) / "diagnostics" / str(phase)
        self.path.mkdir(parents=True, exist_ok=True)
        # Accidentally reusing a diagnostic directory must not erase evidence.
        for name in ("raw_steps.jsonl.gz", "decisions.jsonl.gz", "episodes.jsonl", "optimization.jsonl", "representation.jsonl", "policy_observations.jsonl.gz"):
            if (self.path / name).exists():
                raise FileExistsError("Diagnostic output already exists: " + str(self.path / name))
        self.streams = {
            "raw": gzip.open(self.path / "raw_steps.jsonl.gz", "wt", encoding="utf-8", compresslevel=1),
            "decisions": gzip.open(self.path / "decisions.jsonl.gz", "wt", encoding="utf-8", compresslevel=1),
            "episodes": (self.path / "episodes.jsonl").open("w", encoding="utf-8"),
            "optimization": (self.path / "optimization.jsonl").open("w", encoding="utf-8"),
            "representation": (self.path / "representation.jsonl").open("w", encoding="utf-8"),
            "policy_observations": gzip.open(
                self.path / "policy_observations.jsonl.gz",
                "wt",
                encoding="utf-8",
                compresslevel=1,
            ),
        }
        self.radius, self.max_neighbors, self.horizon = radius_m, max_neighbors, horizon_s
        self.episode_index, self.episode, self.decision = 0, None, None
        self.previous_ego = None
        self.closed, self.raw_count, self.decision_count = False, 0, 0
        self.completed, self.error_count, self.error_examples = [], 0, []
        self.optimization_samples, self.optimization_stats = 0, {}
        self.representation_samples = 0
        self.representation_samples_by_source = {}
        self.representation_stats_by_source = {}
        self.representation_non_scalar_values = 0
        self.policy_observation_snapshots = 0
        self.policy_observation_dropped = 0
        self._policy_observation_keys = set()
        self._next_train_observation_raw_step = 10_000
        self._policy_observation_kind = "train" if str(phase).lower().startswith("train") else "eval"
        self._policy_observation_cap = 10 if self._policy_observation_kind == "train" else 200
        self.serialization_seconds = 0.0
        self.summary_write_pending = False
        self.summary_permission_denials = 0
        self.summary_deferred_flushes = 0
        self.summary_recovered_flushes = 0
        self.summary_last_error = None
        self.last_flush = time.monotonic()
        self.metadata = {"schema_version": SCHEMA_VERSION,
                         "behavior_telemetry_protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
                         "method": method, "phase": phase,
                         "neighbor_radius_m": radius_m, "max_logged_neighbors": max_neighbors,
                         "forecast_horizon_s": horizon_s,
                         "definitions": {
                             "position": "actor center, world coordinates, metres",
                             "heading": "radians counterclockwise from world +x",
                             "velocity": "world vx, vy, metres/second",
                             "cv_obb_ttc_s": "constant-velocity, fixed-heading rectangular collision forecast; null means unavailable or no predicted overlap within horizon; inspect cv_obb_geometry_available",
                             "radial_center_ttc_s": "center distance divided by positive radial closing speed; not intersection collision TTC",
                             "cpa_center_distance_m": "closest center distance over [0, forecast_horizon_s] assuming constant velocity",
                             "collision_events": "SUMO-reported events; forecasts are not actual collisions",
                             "removed_ego": "pre-step fallback, when present, is not a measured post-step collision position",
                             "reward_base": "reward returned by the raw environment before outer shaping",
                             "reward_policy": "reward returned to the learner/evaluator by the instrumented outer environment",
                         }}
        self.metadata["route_lane_telemetry"] = {
            "protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
            "reachability_protocol": "route_continuation_v1",
            "classification": "static SUMO lane connections plus decision-sampled ego route/current lane; not collision safety",
            "actual_lane_transition": "observed between decision-boundary lane samples; attached to the later decision row with the earlier action decision id",
            "stall": "raw actual speed below 0.1 m/s while a known current lane cannot continue to the next planned route edge; lane/route context age is recorded; no per-raw TraCI lane query",
        }
        self.metadata["policy_input_snapshots"] = {
            "stream": "policy_observations.jsonl.gz",
            "content": "complete policy observation before the action plus episode/seed/raw-step/decision/route-lane provenance",
            "training_schedule": "first policy input at or after each 10000 raw-step boundary; maximum 10 rows",
            "evaluation_schedule": "first decision input and terminal-action input per episode; maximum 200 unique rows",
            "collection": "reuse of the observation already passed to policy/environment; no additional forward, rollout, RNG, or SUMO query",
        }
        self.metadata["representation_diagnostics"] = {
            "stream": "representation.jsonl",
            "collection": "existing forward/backward only; no additional environment rollout",
            "training_position": "raw_steps and decision_steps identify collection time, not replay transition identity",
            "episode_provenance": "null for replay batches; explicit episode_index for policy prediction snapshots",
            "missing_values": "null; statistics track finite n and missing separately",
            "gradient_timing": "after existing critic backward, before clipping and optimizer step",
            "gradient_groups": "groups may overlap; their norms must not be added",
            "interpretation": "activation and attention metrics describe computation, not causal policy contribution",
        }
        self.write_run_metadata(metadata or {})

    def write_run_metadata(self, metadata):
        self.metadata.update(_clean(metadata))
        self._json_file("manifest.json", self.metadata)

    def _json_file(self, filename, data):
        target = self.path / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        if filename != "summary.json":
            # Configuration/provenance failures must still fail explicitly.
            temporary.write_text(json.dumps(_clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(target)
            return True

        # Windows readers can temporarily deny rename even when ACLs permit
        # writing. A derived summary must not abort an already-recorded step.
        # JSONL streams have been flushed before this method is called.
        was_pending = self.summary_write_pending
        retry_delays = (0.05, 0.10, 0.20)
        payload = dict(data)
        for attempt in range(len(retry_delays) + 1):
            payload["summary_write"] = {
                "pending": False,
                "permission_denials": self.summary_permission_denials,
                "deferred_flushes": self.summary_deferred_flushes,
                "recovered_flushes": self.summary_recovered_flushes + int(was_pending),
                "last_error": None,
            }
            try:
                temporary.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(target)
            except PermissionError as exc:
                self.summary_permission_denials += 1
                self.summary_last_error = str(exc)
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                self.summary_write_pending = True
                self.summary_deferred_flushes += 1
                payload["summary_write"] = {
                    "pending": True,
                    "permission_denials": self.summary_permission_denials,
                    "deferred_flushes": self.summary_deferred_flushes,
                    "recovered_flushes": self.summary_recovered_flushes,
                    "last_error": self.summary_last_error,
                }
                # Keep a complete latest snapshot alongside the last successful
                # canonical summary. The next regular flush retries publication.
                try:
                    temporary.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                except PermissionError:
                    # A lock on the temporary file can also prevent this copy;
                    # the flushed JSONL remains authoritative in that case.
                    pass
                if not was_pending:
                    logging.getLogger(__name__).warning(
                        "Diagnostic summary publication deferred after %s attempts: %s. "
                        "Flushed JSONL is preserved; inspect %s if present. "
                        "The next normal flush will retry.",
                        attempt + 1, exc, temporary,
                    )
                return False
            else:
                self.summary_recovered_flushes += int(was_pending)
                self.summary_write_pending = False
                self.summary_last_error = None
                return True

    def _write(self, stream, value):
        started = time.monotonic()
        self.streams[stream].write(json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self.serialization_seconds += time.monotonic() - started
        if time.monotonic() - self.last_flush >= 5.0:
            self.flush()

    def on_reset(self, snapshot, info=None, seed=None):
        if self.episode is not None:
            self._finish("reset_without_terminal")
        self.episode_index += 1
        self.previous_ego, self.decision = None, None
        self.episode = {"schema_version": SCHEMA_VERSION,
                        "behavior_telemetry_protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
                        "episode": self.episode_index,
                        "seed": seed, "reset_info": _clean(info or {}),
                        "initial_snapshot": _clean(snapshot), "raw_steps": 0, "decisions": 0,
                        "return_base": 0.0, "return_policy": 0.0,
                        "min_cv_obb_ttc_s": None, "min_center_distance_m": None,
                        "critical_unobserved_ticks": 0, "ego_missing_ticks": 0,
                        "collision_ticks": 0, "diagnostic_error_count": 0}
        self.episode.update(
            route_lane_known_ticks=0,
            route_lane_unknown_ticks=0,
            route_lane_ineligible_ticks=0,
            route_lane_ineligible_stopped_ticks=0,
            route_lane_ineligible_duration_samples=0,
            route_lane_ineligible_seconds=0.0,
            route_lane_ineligible_stopped_duration_samples=0,
            route_lane_ineligible_stopped_seconds=0.0,
            route_lane_ineligible_stall_current_seconds=0.0,
            route_lane_ineligible_stall_max_seconds=0.0,
        )
        self.episode.update(speed_samples=0, speed_sum_mps=0.0,
                            stopped_ticks=0, stopped_seconds=0.0,
                            speed_duration_samples=0, speed_observed_seconds=0.0,
                            harsh_braking_ticks=0, min_acceleration_mps2=None,
                            cv_ttc_below_1s_ticks=0, cv_ttc_below_3s_ticks=0,
                            cv_ttc_below_3s_seconds=0.0, observed_ids_available_ticks=0,
                            risk_evaluable_ticks=0, risk_evaluable_low_ttc_ticks=0,
                            risk_duration_samples=0, risk_observed_seconds=0.0,
                            risk_and_observation_covered_ticks=0, covered_critical_unobserved_ticks=0,
                            ego_current_ticks=0, ego_fallback_ticks=0,
                            action_statistics=[], collision_evidence=[])

    def on_decision_start(self, action, control=None):
        if self.episode is None:
            self.on_reset({}, info={"warning": "reset_hook_not_seen"})
        self.episode["decisions"] += 1
        self.decision = {"episode": self.episode_index,
                         "decision": self.episode["decisions"],
                         "behavior_telemetry_protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
                         "action_env_input": _clean(action), "control": _clean(control or {}),
                         "raw_ticks": 0, "min_cv_obb_ttc_s": None, "min_center_distance_m": None}

    @staticmethod
    def _minimum(record, field, value):
        if value is not None and (record.get(field) is None or value < record[field]):
            record[field] = value

    def on_raw_step(self, snapshot):
        if self.episode is None:
            self.on_reset({}, info={"warning": "reset_hook_not_seen"})
        row = dict(snapshot)
        row.update(episode=self.episode_index,
                   decision=self.decision["decision"] if self.decision else None,
                   behavior_telemetry_protocol=BEHAVIOR_TELEMETRY_PROTOCOL)
        self.raw_count += 1
        self.episode["raw_steps"] += 1
        errors = snapshot.get("errors") or []
        self.error_count += len(errors)
        self.episode["diagnostic_error_count"] += len(errors)
        for error in errors:
            if len(self.error_examples) < 20 and error not in self.error_examples:
                self.error_examples.append(_clean(error))
        ego = snapshot.get("ego")
        route_context_known = snapshot.get("route_lane_context_known") is True
        lane_can_continue = snapshot.get("current_lane_can_reach_next_edge")
        has_next_route_edge = bool(snapshot.get("planned_next_edge"))
        current_lane_status_known = (
            route_context_known
            and has_next_route_edge
            and isinstance(lane_can_continue, bool)
            and snapshot.get("ego_state_source", "current") == "current"
        )
        step_seconds = _number(snapshot.get("step_seconds"))
        speed = _number(ego.get("speed")) if ego else None
        if current_lane_status_known:
            self.episode["route_lane_known_ticks"] += 1
            if lane_can_continue:
                self.episode["route_lane_ineligible_stall_current_seconds"] = 0.0
            else:
                self.episode["route_lane_ineligible_ticks"] += 1
                if step_seconds is not None and step_seconds > 0:
                    self.episode["route_lane_ineligible_duration_samples"] += 1
                    self.episode["route_lane_ineligible_seconds"] += step_seconds
                if speed is not None and speed < ROUTE_LANE_STALL_SPEED_MPS:
                    self.episode["route_lane_ineligible_stopped_ticks"] += 1
                    if step_seconds is not None and step_seconds > 0:
                        self.episode["route_lane_ineligible_stopped_duration_samples"] += 1
                        self.episode["route_lane_ineligible_stopped_seconds"] += step_seconds
                        streak = self.episode["route_lane_ineligible_stall_current_seconds"] + step_seconds
                        self.episode["route_lane_ineligible_stall_current_seconds"] = streak
                        self.episode["route_lane_ineligible_stall_max_seconds"] = max(
                            self.episode["route_lane_ineligible_stall_max_seconds"], streak
                        )
                else:
                    self.episode["route_lane_ineligible_stall_current_seconds"] = 0.0
        else:
            self.episode["route_lane_unknown_ticks"] += 1
            self.episode["route_lane_ineligible_stall_current_seconds"] = 0.0
        observed = snapshot.get("observed_neighbor_ids")
        observed = set(str(x) for x in observed) if observed is not None else None
        self.episode["observed_ids_available_ticks"] += int(observed is not None)
        collision_ids = set(str(x) for x in (snapshot.get("collision_ids") or []))
        neighbors = []
        if ego:
            for actor in snapshot.get("vehicles", []):
                if str(actor.get("id")) == str(ego.get("id")) and actor.get("kind") == ego.get("kind"):
                    continue
                if snapshot.get("ego_state_source", "current") != "current":
                    metrics = {"available": False, "reason": "ego_pose_is_pre_step_fallback"}
                else:
                    metrics = pair_metrics(ego, actor, self.horizon)
                distance = metrics.get("center_distance_m")
                actor_id = str(actor.get("id"))
                actor_key = str(actor.get("key") or (str(actor.get("kind", "vehicle")) + ":" + actor_id))
                was_observed = (actor_id in observed or actor_key in observed) if observed is not None else None
                if distance is not None and distance > self.radius and actor_id not in collision_ids and actor_key not in collision_ids and not was_observed:
                    continue
                item = dict(actor)
                item["risk_cv"] = metrics
                item["observed_by_policy"] = was_observed
                neighbors.append(item)
            neighbors.sort(key=lambda x: (x["risk_cv"].get("cv_obb_ttc_s") is None,
                                           x["risk_cv"].get("cv_obb_ttc_s") if x["risk_cv"].get("cv_obb_ttc_s") is not None else float("inf"),
                                           x["risk_cv"].get("center_distance_m", float("inf"))))
            ttcs = [x["risk_cv"]["cv_obb_ttc_s"] for x in neighbors if x["risk_cv"].get("cv_obb_ttc_s") is not None]
            distances = [x["risk_cv"]["center_distance_m"] for x in neighbors if x["risk_cv"].get("center_distance_m") is not None]
            minimum_ttc, minimum_distance = min(ttcs) if ttcs else None, min(distances) if distances else None
            for target in (self.episode, self.decision):
                if target is not None:
                    self._minimum(target, "min_cv_obb_ttc_s", minimum_ttc)
                    self._minimum(target, "min_center_distance_m", minimum_distance)
            row["min_cv_obb_ttc_s"] = minimum_ttc
            row["min_center_distance_m"] = minimum_distance
            step_seconds = _number(snapshot.get("step_seconds"))
            risk_evaluable = (snapshot.get("ego_state_source", "current") == "current"
                              and not errors
                              and all(x["risk_cv"].get("cv_obb_geometry_available", False) for x in neighbors))
            row["risk_evaluable"] = risk_evaluable
            self.episode["risk_evaluable_ticks"] += int(risk_evaluable)
            self.episode["risk_evaluable_low_ttc_ticks"] += int(
                risk_evaluable and minimum_ttc is not None and minimum_ttc < 3.0)
            if risk_evaluable and step_seconds is not None and step_seconds > 0:
                self.episode["risk_duration_samples"] += 1
                self.episode["risk_observed_seconds"] += step_seconds
                if minimum_ttc is not None and minimum_ttc < 3.0:
                    self.episode["cv_ttc_below_3s_seconds"] += step_seconds
            if minimum_ttc is not None:
                self.episode["cv_ttc_below_1s_ticks"] += int(minimum_ttc < 1.0)
                self.episode["cv_ttc_below_3s_ticks"] += int(minimum_ttc < 3.0)
            row["critical_unobserved"] = any(x["observed_by_policy"] is False and
                                              x["risk_cv"].get("cv_obb_ttc_s") is not None and
                                              x["risk_cv"]["cv_obb_ttc_s"] <= 3.0 for x in neighbors)
            self.episode["critical_unobserved_ticks"] += int(row["critical_unobserved"])
            observation_covered = risk_evaluable and observed is not None
            self.episode["risk_and_observation_covered_ticks"] += int(observation_covered)
            self.episode["covered_critical_unobserved_ticks"] += int(observation_covered and row["critical_unobserved"])
            now, speed = _number(snapshot.get("sim_time")), _number(ego.get("speed"))
            current_pose = snapshot.get("ego_state_source", "current") == "current"
            self.episode["ego_current_ticks"] += int(current_pose)
            self.episode["ego_fallback_ticks"] += int(not current_pose)
            if current_pose and speed is not None:
                self.episode["speed_samples"] += 1
                self.episode["speed_sum_mps"] += speed
                self.episode["stopped_ticks"] += int(speed < 0.1)
                if step_seconds is not None and step_seconds > 0:
                    self.episode["speed_duration_samples"] += 1
                    self.episode["speed_observed_seconds"] += step_seconds
                    if speed < 0.1:
                        self.episode["stopped_seconds"] += step_seconds
            if current_pose and now is not None and speed is not None and self.previous_ego:
                previous_time, previous_speed = self.previous_ego
                row["ego_acceleration_mps2"] = (speed - previous_speed) / (now - previous_time) if now > previous_time else None
            else:
                row["ego_acceleration_mps2"] = None
            self.previous_ego = (now, speed) if current_pose and now is not None and speed is not None else None
            acceleration = row["ego_acceleration_mps2"]
            if acceleration is not None:
                self._minimum(self.episode, "min_acceleration_mps2", acceleration)
                self.episode["harsh_braking_ticks"] += int(acceleration < -3.0)
        else:
            self.episode["ego_missing_ticks"] += 1
            self.previous_ego = None
        # Aggregate risk uses all captured neighbors in the radius before truncation.
        row["neighbors_in_radius"] = len(neighbors)
        row["neighbors_omitted"] = max(0, len(neighbors) - self.max_neighbors)
        row["vehicles"] = neighbors[:self.max_neighbors]
        events = snapshot.get("events") or {}
        self.episode["collision_ticks"] += int(bool(events.get("collision")))
        if events.get("collision"):
            self.episode["collision_evidence"].append({
                "raw_step": snapshot.get("raw_step"), "sim_time": snapshot.get("sim_time"),
                "ego": _clean(ego), "ego_state_source": snapshot.get("ego_state_source"),
                "collision_events": _clean(snapshot.get("collision_events") or []),
                "collision_ids": _clean(snapshot.get("collision_ids") or []),
                "note": "ego fallback positions precede collision; use SUMO event lane/position when available"})
        self.episode["last_snapshot"] = row
        if self.decision is not None:
            self.decision["raw_ticks"] += 1
        self._write("raw", row)

    def on_decision_end(self, reward, terminated, truncated, info=None):
        if self.episode is None:
            return
        self.episode["return_base"] += float(reward)
        if self.decision is not None:
            self.decision.update(reward_base=float(reward), terminated=bool(terminated),
                                 truncated=bool(truncated), info_base=_clean(info or {}))

    def _record_policy_observation(self, observation, *, raw_step, decision, trigger, info=None):
        if self.closed or observation is None or self.episode is None:
            return
        key = (self.episode_index, int(decision))
        if key in self._policy_observation_keys:
            return
        if self.policy_observation_snapshots >= self._policy_observation_cap:
            self.policy_observation_dropped += 1
            return
        self._policy_observation_keys.add(key)
        control = (self.decision or {}).get("control", {})
        self._write(
            "policy_observations",
            {
                "behavior_telemetry_protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
                "episode": self.episode_index,
                "seed": self.episode.get("seed"),
                "raw_step_before_action": int(raw_step),
                "decision": int(decision),
                "trigger": str(trigger),
                "control_context": control,
                "pre_step_info": _clean(info or {}),
                "observation": _clean(observation),
            },
        )
        self.policy_observation_snapshots += 1

    def on_policy_step(
        self,
        action,
        reward,
        terminated,
        truncated,
        info=None,
        *,
        policy_observation=None,
        policy_raw_step=None,
        policy_decision=None,
        pre_step_info=None,
    ):
        if self.episode is None:
            return
        self.episode["return_policy"] += float(reward)
        action_values = _clean(action)
        if isinstance(action_values, list) and all(_number(x) is not None for x in action_values):
            statistics = self.episode["action_statistics"]
            while len(statistics) < len(action_values):
                statistics.append({"n": 0, "sum": 0.0, "min": None, "max": None,
                                   "abs_ge_0p95_count": 0})
            for stat, value in zip(statistics, action_values):
                value = float(value)
                stat["n"] += 1
                stat["sum"] += value
                stat["min"] = value if stat["min"] is None else min(stat["min"], value)
                stat["max"] = value if stat["max"] is None else max(stat["max"], value)
                stat["abs_ge_0p95_count"] += int(abs(value) >= 0.95)
        row = dict(self.decision or {"episode": self.episode_index, "warning": "decision_hook_not_seen"})
        row.update(action_policy=_clean(action), reward_policy=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated), info=_clean(info or {}))
        self._write("decisions", row)
        self.decision_count += 1
        decision_index = (
            int(policy_decision)
            if policy_decision is not None
            else int((self.decision or {}).get("decision", self.decision_count))
        )
        raw_start = self.raw_count if policy_raw_step is None else int(policy_raw_step)
        if self._policy_observation_kind == "train":
            if raw_start >= self._next_train_observation_raw_step:
                while self._next_train_observation_raw_step <= raw_start:
                    self._next_train_observation_raw_step += 10_000
                self._record_policy_observation(
                    policy_observation,
                    raw_step=raw_start,
                    decision=decision_index,
                    trigger="train_first_policy_input_at_or_after_10k_raw_boundary",
                    info=pre_step_info,
                )
        else:
            triggers = []
            if decision_index == 1:
                triggers.append("eval_first_decision_input")
            if terminated or truncated:
                triggers.append("eval_terminal_action_input")
            if triggers:
                self._record_policy_observation(
                    policy_observation,
                    raw_step=raw_start,
                    decision=decision_index,
                    trigger="+".join(triggers),
                    info=pre_step_info,
                )
        if terminated or truncated:
            self._finish("environment_terminal", info)

    def _finish(self, reason, info=None):
        if self.episode is None:
            return
        self.episode["finish_reason"] = reason
        self.episode["terminal_info"] = _clean(info or {})
        n = self.episode["speed_samples"]
        self.episode["mean_actual_speed_mps"] = self.episode["speed_sum_mps"] / n if n else None
        self.episode["stopped_fraction_of_speed_samples"] = self.episode["stopped_ticks"] / n if n else None
        if not self.episode["speed_duration_samples"]:
            self.episode["stopped_seconds"] = None
            self.episode["speed_observed_seconds"] = None
        if not self.episode["route_lane_ineligible_duration_samples"]:
            self.episode["route_lane_ineligible_seconds"] = None
        if not self.episode["route_lane_ineligible_stopped_duration_samples"]:
            self.episode["route_lane_ineligible_stopped_seconds"] = None
            self.episode["route_lane_ineligible_stall_max_seconds"] = None
        risk_n = self.episode["risk_evaluable_ticks"]
        observation_n = self.episode["risk_and_observation_covered_ticks"]
        self.episode["low_ttc_fraction_of_evaluable_ticks"] = self.episode["risk_evaluable_low_ttc_ticks"] / risk_n if risk_n else None
        self.episode["critical_unobserved_fraction_of_covered_ticks"] = self.episode["covered_critical_unobserved_ticks"] / observation_n if observation_n else None
        if not self.episode["risk_duration_samples"]:
            self.episode["cv_ttc_below_3s_seconds"] = None
            self.episode["risk_observed_seconds"] = None
        for stat in self.episode["action_statistics"]:
            stat["mean"] = stat["sum"] / stat["n"] if stat["n"] else None
            stat["abs_ge_0p95_fraction"] = stat["abs_ge_0p95_count"] / stat["n"] if stat["n"] else None
        final_events = (self.episode.get("last_snapshot") or {}).get("events") or {}
        facts = dict(final_events)
        facts.update(info or {})
        if reason != "environment_terminal":
            outcome = "incomplete"
        elif facts.get("is_success") or facts.get("success"):
            outcome = "success"
        elif facts.get("collision"):
            outcome = "collision"
        elif facts.get("off_route"):
            outcome = "off_route"
        elif facts.get("max_time") or facts.get("timeout"):
            outcome = "timeout"
        else:
            outcome = "other_terminal"
        self.episode["outcome"] = outcome
        self._write("episodes", self.episode)
        compact = {k: v for k, v in self.episode.items() if k not in ("initial_snapshot", "last_snapshot", "reset_info")}
        self.completed.append(compact)
        self.episode, self.decision = None, None
        self.flush()

    def record_optimization(self, raw_steps, decision_steps, updates, metrics):
        scalars = {}
        for key, value in (metrics or {}).items():
            if hasattr(value, "numel") and value.numel() != 1:
                continue
            number = _number(value)
            if number is not None:
                scalars[str(key)] = number
                stat = self.optimization_stats.setdefault(str(key), {"n": 0, "sum": 0.0,
                                                                     "min": number, "max": number, "last": number})
                stat["n"] += 1
                stat["sum"] += number
                stat["min"], stat["max"], stat["last"] = min(stat["min"], number), max(stat["max"], number), number
        self.optimization_samples += 1
        self._write("optimization", {"raw_steps": int(raw_steps), "decision_steps": int(decision_steps),
                                     "updates": int(updates), "metrics": scalars,
                                     "timing": "latest_available_logger_values_at_callback; not necessarily same-update metrics"})

    def record_representation(self, raw_steps, decision_steps, metrics, source,
                              episode=None, updates=None, timing=None):
        """Save scalars from existing forwards/backwards, with explicit provenance.

        Replay-batch telemetry is deliberately not assigned to a rollout episode.
        Callers may identify an episode only for its actual policy prediction.
        Missing/nonfinite values stay null and have separate coverage counts.
        """
        if self.closed:
            return
        source = str(source)
        statistics = self.representation_stats_by_source.setdefault(source, {})
        scalars = {}
        for key, value in (metrics or {}).items():
            if hasattr(value, "numel") and value.numel() != 1:
                self.representation_non_scalar_values += 1
                continue
            if isinstance(value, (dict, list, tuple)):
                self.representation_non_scalar_values += 1
                continue
            key = str(key)
            number = _number(value) if value is not None else None
            scalars[key] = number
            stat = statistics.setdefault(key, {"n": 0, "missing": 0, "sum": 0.0,
                                                "min": None, "max": None, "last": None})
            stat["last"] = number
            if number is None:
                stat["missing"] += 1
            else:
                stat["n"] += 1
                stat["sum"] += number
                stat["min"] = number if stat["min"] is None else min(stat["min"], number)
                stat["max"] = number if stat["max"] is None else max(stat["max"], number)
        self.representation_samples += 1
        self.representation_samples_by_source[source] = self.representation_samples_by_source.get(source, 0) + 1
        row = {"raw_steps": int(raw_steps), "decision_steps": int(decision_steps),
               "source": source, "updates": None if updates is None else int(updates),
               "episode_index": None if episode is None else int(episode),
               "metrics": scalars, "timing": timing}
        if episode is not None and int(episode) == self.episode_index and self.episode is not None:
            row["episode_raw_steps"] = self.episode.get("raw_steps")
            row["episode_decisions"] = self.episode.get("decisions")
            row["seed"] = self.episode.get("seed")
            row["reset_info"] = self.episode.get("reset_info")
        self._write("representation", row)

    def on_env_close(self):
        if not self.closed:
            # reset() closes SUMO between episodes; do not close output streams.
            self.flush()

    def flush(self):
        if self.closed:
            return
        for stream in self.streams.values():
            stream.flush()
        self.last_flush = time.monotonic()
        outcomes = {}
        for episode in self.completed:
            key = episode["outcome"]
            outcomes[key] = outcomes.get(key, 0) + 1
        self._json_file("summary.json", {"schema_version": SCHEMA_VERSION,
                                         "behavior_telemetry_protocol": BEHAVIOR_TELEMETRY_PROTOCOL,
                                         "episodes_finished": len(self.completed),
                                         "raw_records": self.raw_count, "decision_records": self.decision_count,
                                         "diagnostic_error_count": self.error_count,
                                         "diagnostic_error_examples": self.error_examples,
                                         "serialization_seconds": self.serialization_seconds,
                                         "outcome_counts": outcomes,
                                         "optimization_samples": self.optimization_samples,
                                         "optimization_statistics": self.optimization_stats,
                                         "representation_samples": self.representation_samples,
                                         "representation_samples_by_source": self.representation_samples_by_source,
                                         "representation_statistics_by_source": self.representation_stats_by_source,
                                         "representation_non_scalar_values": self.representation_non_scalar_values,
                                         "policy_observation_snapshots": self.policy_observation_snapshots,
                                         "policy_observation_dropped": self.policy_observation_dropped,
                                         "episode_summaries": self.completed})

    def close(self):
        if self.closed:
            return
        if self.episode is not None:
            self._finish("closed_before_terminal")
        self.flush()
        for stream in self.streams.values():
            stream.close()
        self.closed = True


def get_behavior_recorder(env):
    seen, pending = set(), [env]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        own = getattr(current, "__dict__", {})
        if own.get("behavior_recorder") is not None:
            return own["behavior_recorder"]
        if own.get("_behavior_diagnostics") is not None:
            return own["_behavior_diagnostics"]
        pending.extend(own.get("envs") or [])
        pending.extend([own.get("env"), own.get("venv")])
    return None


def wrap_behavior_diagnostics(env, output_dir, phase, method, metadata=None):
    import gymnasium as gym

    base = env.unwrapped
    if not hasattr(base, "set_behavior_diagnostics"):
        raise TypeError("Environment does not expose raw-step behavior diagnostic hooks")
    if get_behavior_recorder(env) is not None:
        raise RuntimeError("Behavior diagnostics are already attached to this environment")
    run_metadata = dict(metadata or {})
    run_metadata["action_space"] = {
        "type": type(env.action_space).__name__,
        "low": _clean(getattr(env.action_space, "low", None)),
        "high": _clean(getattr(env.action_space, "high", None)),
        "abs_ge_0p95_definition": "Absolute action >= 0.95; interpret as near saturation only for normalized [-1,1] channels.",
    }
    recorder = BehaviorDiagnosticsRecorder(output_dir, phase, method, run_metadata)
    base.set_behavior_diagnostics(recorder)

    class _BehaviorWrapper(gym.Wrapper):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.behavior_recorder = recorder
            self._last_observation = None
            self._last_info = {}

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            self._last_observation = observation
            self._last_info = dict(info or {})
            return observation, info

        def step(self, action):
            policy_observation = self._last_observation
            pre_step_info = self._last_info
            policy_raw_step = recorder.raw_count
            policy_decision = (
                int(recorder.episode.get("decisions", 0)) + 1
                if recorder.episode is not None
                else recorder.decision_count + 1
            )
            observation, reward, terminated, truncated, info = self.env.step(action)
            recorder.on_policy_step(
                action,
                reward,
                terminated,
                truncated,
                info,
                policy_observation=policy_observation,
                policy_raw_step=policy_raw_step,
                policy_decision=policy_decision,
                pre_step_info=pre_step_info,
            )
            self._last_observation = observation
            self._last_info = dict(info or {})
            return observation, reward, terminated, truncated, info

        def close(self):
            try:
                self.env.close()
            finally:
                recorder.close()

    return _BehaviorWrapper(env)
