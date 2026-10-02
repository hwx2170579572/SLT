"""Fresh-state route lane action mapping for route-consistent SAC commands.

This module is deliberately independent of the optional behavior recorder. It
reads only the controlled ego's live TraCI route/lane state and a static SUMO
network graph. It never inspects a background vehicle's route, changes reward,
or claims that a requested lane change was physically completed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import gymnasium as gym

from .route_reachability_v1 import (
    ELIGIBLE,
    INELIGIBLE,
    UNKNOWN,
    RouteLaneReachability,
)


PROTOCOL_VERSION = "route_action_consistency_v1"


def _json_safe(value: Any) -> Any:
    """Convert telemetry values to strict JSON without changing their meaning."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_route_label(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number in (UNKNOWN, INELIGIBLE, ELIGIBLE) else None


def decide_route_action(
    proposed_action: Any,
    *,
    context_known: bool,
    context_fresh: bool,
    current_road: str | None,
    has_next_edge: bool,
    current_lane_label: Any,
    target_lane_label: Any,
    lane_command: Any,
    target_lane_id: str | None = None,
    target_lane_reason: str | None = None,
    target_lane_in_range: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a copied env action and an auditable direct-route veto decision.

    The only transformed case is a known, fresh, eligible current lane with a
    known next route edge and an explicit non-hold request to a known
    ineligible target lane on a non-internal road. In that case only lateral
    coordinate 1 becomes 0.0 (SUMO hold); coordinate 0 is preserved exactly.
    Unknown or out-of-scope cases pass through unchanged.
    """
    try:
        action_copy = np.asarray(proposed_action).copy()
        flat = action_copy.reshape(-1)
    except Exception:
        action_copy = np.asarray([], dtype=np.float32)
        flat = action_copy

    speed_before = np.array(flat[:1], copy=True)

    try:
        command = int(lane_command)
    except (TypeError, ValueError, OverflowError):
        command = None

    current_label = _as_route_label(current_lane_label)
    target_label = _as_route_label(target_lane_label)
    try:
        finite_action = bool(flat.size == 2 and np.all(np.isfinite(flat.astype(float))))
    except (TypeError, ValueError, OverflowError):
        finite_action = False

    reason = "veto_not_applicable"
    if flat.size != 2:
        reason = "action_shape_unknown"
    elif not finite_action:
        reason = "action_nonfinite"
    elif not context_known:
        reason = "context_unknown"
    elif not context_fresh:
        reason = "stale_context"
    elif not current_road:
        reason = "current_road_unknown"
    elif str(current_road).startswith(":"):
        reason = "internal_road"
    elif not has_next_edge:
        reason = "no_next_edge"
    elif current_label is None or current_label == UNKNOWN:
        reason = "current_lane_unknown"
    elif current_label != ELIGIBLE:
        reason = "current_lane_ineligible"
    elif command is None:
        reason = "lane_command_unknown"
    elif command == 0:
        reason = "hold_command"
    elif not target_lane_in_range:
        reason = "target_lane_out_of_range"
    elif target_label is None or target_label == UNKNOWN:
        reason = "target_lane_unknown"
    elif target_label == ELIGIBLE:
        reason = "target_lane_eligible"
    elif target_label == INELIGIBLE:
        reason = "known_unreachable_target_lane_vetoed"

    veto_applied = reason == "known_unreachable_target_lane_vetoed"
    if veto_applied:
        flat[1] = 0.0

    audit = {
        "protocol": PROTOCOL_VERSION,
        "veto_applied": veto_applied,
        "reason": reason,
        "context_known": bool(context_known),
        "context_fresh": bool(context_fresh),
        "current_road": current_road,
        "has_next_edge": bool(has_next_edge),
        "current_lane_label": current_label,
        "target_lane_label": target_label,
        "target_lane_id": target_lane_id,
        "target_lane_reason": target_lane_reason,
        "target_lane_in_range": bool(target_lane_in_range),
        "lane_command_requested": command,
        "lane_command_forwarded": 0 if veto_applied else command,
        "policy_proposed_action": _json_safe(proposed_action),
        "env_action_forwarded": _json_safe(action_copy),
        "speed_action_unchanged": bool(
            flat.size == 2 and np.array_equal(flat[:1], speed_before, equal_nan=True)
        ),
        "interpretation_limit": (
            "The lateral command sent to the environment may be held; this does "
            "not establish that a requested lane change was executed or that "
            "the route will be completed."
        ),
    }
    return action_copy, audit


def resolve_target_lane(
    raw_env: Any,
    *,
    current_road: str | None,
    current_lane_id: str | None,
    lane_command: int | None,
) -> dict[str, Any]:
    """Resolve the lane a command targets using the environment's lane contract.

    This calls the same ``_driving_lanes`` and ``_sumo_lane_offset`` helpers as
    the SUMO controller. It returns a planned target only; it does not issue a
    TraCI lane-change request.
    """
    result: dict[str, Any] = {
        "target_lane_id": None,
        "target_lane_in_range": False,
        "target_lane_reason": "target_lane_context_unavailable",
        "lane_command": lane_command,
    }
    if lane_command is None:
        result["target_lane_reason"] = "lane_command_unknown"
        return result
    try:
        command = int(lane_command)
    except (TypeError, ValueError, OverflowError):
        result["target_lane_reason"] = "lane_command_unknown"
        return result
    if command == 0:
        result.update(
            target_lane_id=current_lane_id,
            target_lane_in_range=True,
            target_lane_reason="hold_command",
        )
        return result
    if not current_road or not current_lane_id:
        result["target_lane_reason"] = "current_lane_unknown"
        return result

    driving_lanes_fn = getattr(raw_env, "_driving_lanes", None)
    lane_offset_fn = getattr(raw_env, "_sumo_lane_offset", None)
    if not callable(driving_lanes_fn) or not callable(lane_offset_fn):
        result["target_lane_reason"] = "lane_mapping_helpers_unavailable"
        return result

    try:
        lanes = tuple(int(index) for index in driving_lanes_fn(str(current_road)))
    except Exception as exc:
        result["target_lane_reason"] = f"driving_lanes_error:{type(exc).__name__}"
        return result
    suffix = str(current_lane_id).rsplit("_", 1)[-1]
    current_index = int(suffix) if suffix.isdigit() else None
    if not lanes or current_index not in lanes:
        result["target_lane_reason"] = "current_lane_not_in_driving_lanes"
        return result

    try:
        offset = int(lane_offset_fn(command))
    except Exception as exc:
        result["target_lane_reason"] = f"lane_offset_error:{type(exc).__name__}"
        return result
    current_rank = lanes.index(current_index)
    target_rank = current_rank + offset
    if target_rank < 0 or target_rank >= len(lanes):
        result["target_lane_reason"] = "target_lane_out_of_range"
        return result

    target_lane_id = f"{current_road}_{lanes[target_rank]}"
    result.update(
        target_lane_id=target_lane_id,
        target_lane_in_range=True,
        target_lane_reason="target_lane_resolved",
        current_lane_index=current_index,
        lane_indices=lanes,
        target_lane_index=lanes[target_rank],
        sumo_lane_offset=offset,
    )
    return result


@lru_cache(maxsize=16)
def _load_static_network(network_path: str) -> RouteLaneReachability:
    return RouteLaneReachability.from_net(Path(network_path))


def read_live_ego_route_context(raw_env: Any, network: RouteLaneReachability) -> dict[str, Any]:
    """Read only the controlled ego's current TraCI state and classify its lane.

    The context is fresh at the wrapper decision boundary because all dynamic
    state is read directly from TraCI immediately before forwarding the action.
    No behavior-recorder cache or background-vehicle route is consulted.
    """
    captured_at = datetime.now(timezone.utc).isoformat()
    raw_step = getattr(raw_env, "_raw_steps", None)
    try:
        raw_step = int(raw_step) if raw_step is not None else None
    except (TypeError, ValueError, OverflowError):
        raw_step = None

    context: dict[str, Any] = {
        "context_source": "live_ego_traci_route_and_lane_plus_static_net",
        "context_sample_time_utc": captured_at,
        "context_sample_raw_step": raw_step,
        "context_sample_sim_time_seconds": None,
        "context_age_raw_steps": 0 if raw_step is not None else None,
        "context_fresh": False,
        "context_known": False,
        "context_reason": "context_unavailable",
        "ego_id": None,
        "ego_active": False,
        "current_road_id": None,
        "current_lane_id": None,
        "route_edges": None,
        "route_index": None,
        "planned_current_route_edge": None,
        "planned_next_edge": None,
        "current_lane_reachability_label": UNKNOWN,
        "current_lane_can_reach_next_edge": None,
    }

    specification = getattr(raw_env, "specification", None)
    ego_id = getattr(specification, "ego_id", None)
    connection = getattr(raw_env, "_connection", None)
    if not ego_id or connection is None:
        context["context_reason"] = "ego_identity_or_connection_unavailable"
        return context
    context["ego_id"] = str(ego_id)

    try:
        # Use the ego ID only; do not inspect any background vehicle routes.
        if str(ego_id) not in set(connection.vehicle.getIDList()):
            context["context_reason"] = "ego_not_active"
            return context
        context["ego_active"] = True
        road_id = str(connection.vehicle.getRoadID(str(ego_id)))
        lane_id = str(connection.vehicle.getLaneID(str(ego_id)))
        route_edges = tuple(str(edge) for edge in connection.vehicle.getRoute(str(ego_id)))
        route_index = int(connection.vehicle.getRouteIndex(str(ego_id)))
    except Exception as exc:
        context["context_reason"] = f"ego_traci_read_error:{type(exc).__name__}"
        return context

    context.update(
        current_road_id=road_id,
        current_lane_id=lane_id,
        route_edges=route_edges,
        route_index=route_index,
    )
    simulation = getattr(connection, "simulation", None)
    get_time = getattr(simulation, "getTime", None)
    if callable(get_time):
        try:
            context["context_sample_sim_time_seconds"] = float(get_time())
        except Exception:
            # A missing simulation clock does not invalidate directly sampled
            # route/lane state; its absence is explicit in the context.
            pass

    try:
        classified = network.classify([lane_id], route_edges, route_index, road_id)
        lane_label = int(classified.labels[0]) if classified.labels else UNKNOWN
        context.update(
            planned_current_route_edge=classified.current_edge or None,
            planned_next_edge=classified.next_edge or None,
            current_lane_reachability_label=lane_label,
            current_lane_can_reach_next_edge=(
                bool(lane_label == ELIGIBLE)
                if classified.context_known and lane_label != UNKNOWN and classified.next_edge
                else None
            ),
            context_known=bool(classified.context_known and lane_label != UNKNOWN),
            context_fresh=True,
            context_reason=(
                "known_live_ego_route_lane"
                if classified.context_known and lane_label != UNKNOWN
                else "live_ego_state_but_route_or_lane_unknown"
            ),
        )
    except Exception as exc:
        context["context_reason"] = f"static_route_classification_error:{type(exc).__name__}"
        context["context_fresh"] = True
    return context


class RouteActionConsistencyWrapper(gym.Wrapper):
    """Map only proven route-incompatible lateral requests to SUMO hold.

    Place this wrapper inside the normal behavior-diagnostics wrapper and
    outside the environment/reward wrappers. The recorder then receives the
    policy-proposed action and the returned info contains the action forwarded
    to the environment. SB3's off-policy collector stores its pre-wrapper
    proposal in replay; this wrapper does not alter reward or observation.
    """

    def __init__(
        self,
        env,
        *,
        run_dir: str | Path | None = None,
        phase: str = "train",
        method: str = "unspecified",
        static_network: RouteLaneReachability | None = None,
    ) -> None:
        super().__init__(env)
        if phase not in ("train", "eval") and not phase.startswith("eval_worker_"):
            raise ValueError("phase must be train, eval, or eval_worker_<id>")
        self.raw_env = getattr(env, "unwrapped", env)
        self.phase = str(phase)
        self.method = str(method)
        self.episode_index = -1
        self.episode_seed: int | None = None
        self.decision_step = 0
        self._sidecar = None
        self.sidecar_path: Path | None = None

        self.static_network = static_network
        self.static_network_error: str | None = None
        if self.static_network is None:
            specification = getattr(self.raw_env, "specification", None)
            network_path = getattr(specification, "network_path", None)
            if network_path:
                try:
                    self.static_network = _load_static_network(str(Path(network_path).resolve()))
                except Exception as exc:
                    self.static_network_error = f"{type(exc).__name__}: {exc}"
            else:
                self.static_network_error = "network_path_unavailable"

        if run_dir is not None:
            phase_dir = Path(run_dir) / "diagnostics" / self.phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            self.sidecar_path = phase_dir / "route_action_consistency.jsonl"
            self._sidecar = self.sidecar_path.open("x", encoding="utf-8", newline="\n")
            self._write_sidecar(
                {
                    "record_type": "manifest",
                    "schema_version": PROTOCOL_VERSION,
                    "phase": self.phase,
                    "method": self.method,
                    "network_path": getattr(
                        getattr(self.raw_env, "specification", None), "network_path", None
                    ),
                    "network_available": self.static_network is not None,
                    "network_error": self.static_network_error,
                    "context_source": "live_ego_traci_route_and_lane_plus_static_net",
                    "replay_action_semantics": "policy_proposal_is_stored_by_SB3; environment receives mapped action",
                    "reward_and_observation_mapping": "wrapper returns inner observation/reward/done unchanged",
                    "lane_change_outcome_semantics": "requested and forwarded lane commands are not proof of an executed lane transition",
                }
            )

    def _write_sidecar(self, row: Mapping[str, Any]) -> None:
        if self._sidecar is None:
            return
        self._sidecar.write(
            json.dumps(_json_safe(dict(row)), ensure_ascii=False, allow_nan=False) + "\n"
        )
        self._sidecar.flush()

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.episode_index += 1
        self.episode_seed = int(seed) if seed is not None else None
        self.decision_step = 0
        return observation, info

    def step(self, action):
        self.decision_step += 1
        proposed = np.asarray(action).copy()
        route_context: dict[str, Any]
        if self.static_network is None:
            route_context = {
                "context_source": "live_ego_traci_route_and_lane_plus_static_net",
                "context_sample_time_utc": datetime.now(timezone.utc).isoformat(),
                "context_sample_raw_step": getattr(self.raw_env, "_raw_steps", None),
                "context_sample_sim_time_seconds": None,
                "context_age_raw_steps": None,
                "context_fresh": False,
                "context_known": False,
                "context_reason": "static_network_unavailable",
                "current_road_id": None,
                "current_lane_id": None,
                "route_edges": None,
                "route_index": None,
                "planned_current_route_edge": None,
                "planned_next_edge": None,
                "current_lane_reachability_label": UNKNOWN,
                "current_lane_can_reach_next_edge": None,
            }
        else:
            route_context = read_live_ego_route_context(self.raw_env, self.static_network)

        lane_command = None
        target_context: dict[str, Any] = {
            "target_lane_id": None,
            "target_lane_in_range": False,
            "target_lane_reason": "lane_command_unknown",
        }
        adapt_action = getattr(self.raw_env, "adapt_action", None)
        try:
            if callable(adapt_action) and proposed.reshape(-1).size == 2:
                decoded = np.clip(proposed.reshape(-1), -1.0, 1.0)
                _target_speed, lane_command = adapt_action(decoded)
                lane_command = int(lane_command)
                target_context = resolve_target_lane(
                    self.raw_env,
                    current_road=route_context.get("current_road_id"),
                    current_lane_id=route_context.get("current_lane_id"),
                    lane_command=lane_command,
                )
                if self.static_network is not None and target_context.get("target_lane_id"):
                    target_result = self.static_network.classify(
                        [target_context["target_lane_id"]],
                        route_context.get("route_edges"),
                        route_context.get("route_index"),
                        route_context.get("current_road_id"),
                    )
                    target_context["target_lane_reachability_label"] = (
                        int(target_result.labels[0]) if target_result.labels else UNKNOWN
                    )
                    target_context["target_route_context_known"] = bool(target_result.context_known)
                    target_context["target_planned_next_edge"] = target_result.next_edge or None
                else:
                    target_context["target_lane_reachability_label"] = UNKNOWN
                    target_context["target_route_context_known"] = False
        except Exception as exc:
            target_context = {
                "target_lane_id": None,
                "target_lane_in_range": False,
                "target_lane_reason": f"action_or_target_decode_error:{type(exc).__name__}",
                "target_lane_reachability_label": UNKNOWN,
                "target_route_context_known": False,
            }

        context_known = bool(
            route_context.get("context_known")
            and target_context.get("target_route_context_known", False)
        )
        route_current_label = route_context.get("current_lane_reachability_label", UNKNOWN)
        target_label = target_context.get("target_lane_reachability_label", UNKNOWN)
        has_next_edge = bool(route_context.get("planned_next_edge"))
        forwarded, audit = decide_route_action(
            proposed,
            context_known=context_known,
            context_fresh=bool(route_context.get("context_fresh")),
            current_road=route_context.get("current_road_id"),
            has_next_edge=has_next_edge,
            current_lane_label=route_current_label,
            target_lane_label=target_label,
            lane_command=lane_command,
            target_lane_id=target_context.get("target_lane_id"),
            target_lane_reason=target_context.get("target_lane_reason"),
            target_lane_in_range=bool(target_context.get("target_lane_in_range")),
        )

        raw_step_before = getattr(self.raw_env, "_raw_steps", None)
        row = {
            "schema_version": PROTOCOL_VERSION,
            "record_type": "decision",
            "phase": self.phase,
            "method": self.method,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "decision_step": self.decision_step,
            "raw_step_before_action": raw_step_before,
            "route_context": route_context,
            "target_context": target_context,
            "policy_proposed_action": _json_safe(proposed),
            "env_action_forwarded": _json_safe(forwarded),
            "decision": audit,
        }

        try:
            observation, reward, terminated, truncated, inner_info = self.env.step(forwarded)
        except BaseException as exc:
            row["step_error"] = f"{type(exc).__name__}: {exc}"
            self._write_sidecar(row)
            raise

        new_info = dict(inner_info) if isinstance(inner_info, Mapping) else {
            "inner_info_type": type(inner_info).__name__
        }
        environment_reported = {
            "lane_change_applied_reported_by_env": new_info.get("lane_change_applied"),
            "lane_control_request_status_reported_by_env": new_info.get("lane_control_request_status"),
            "lane_control_request_reason_reported_by_env": new_info.get("lane_control_request_reason"),
            "lane_transition_reported_by_env": new_info.get("actual_lane_transition_since_previous_decision"),
            "returned_current_lane_id_reported_by_env": new_info.get("current_lane_id"),
        }
        row["environment_reported"] = environment_reported
        row["raw_step_after_action"] = getattr(self.raw_env, "_raw_steps", None)
        new_info["route_action_consistency"] = {
            **audit,
            "phase": self.phase,
            "method": self.method,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "decision_step": self.decision_step,
            "raw_step_before_action": raw_step_before,
            "raw_step_after_action": getattr(self.raw_env, "_raw_steps", None),
            "context_source": route_context.get("context_source"),
            "context_sample_time_utc": route_context.get("context_sample_time_utc"),
            "context_sample_raw_step": route_context.get("context_sample_raw_step"),
            "context_sample_sim_time_seconds": route_context.get("context_sample_sim_time_seconds"),
            "context_age_raw_steps": route_context.get("context_age_raw_steps"),
            "context_reason": route_context.get("context_reason"),
            "route_context": _json_safe(route_context),
            "target_context": _json_safe(target_context),
            "environment_reported": _json_safe(environment_reported),
        }
        self._write_sidecar(row)
        return observation, reward, terminated, truncated, new_info

    def close(self):
        sidecar_error = None
        try:
            if self._sidecar is not None and not self._sidecar.closed:
                self._sidecar.flush()
                self._sidecar.close()
        except BaseException as exc:
            sidecar_error = exc
        try:
            result = self.env.close()
        except BaseException:
            if sidecar_error is None:
                raise
            raise
        if sidecar_error is not None:
            raise sidecar_error
        return result


__all__ = [
    "PROTOCOL_VERSION",
    "RouteActionConsistencyWrapper",
    "decide_route_action",
    "read_live_ego_route_context",
    "resolve_target_lane",
]
