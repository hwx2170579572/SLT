"""Runtime-only lane feasibility and route-intent labels for v4.

The labels use only state already available to the controller: the ego lane,
the next edge in its assigned route, and current SUMO lane connections.  They
never inspect a future outcome, reward, validation result, or test artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LANE_COMMANDS = (-1, 0, 1)
KEEP_COMMAND_INDEX = 1


def lane_command_index(command: int) -> int:
    """Map the public lane command ``-1/0/+1`` to a three-way index."""

    try:
        return LANE_COMMANDS.index(int(command))
    except ValueError as exc:
        raise ValueError(f"lane command must be one of {LANE_COMMANDS}, got {command}") from exc


@dataclass(frozen=True)
class DecisionAlignmentContext:
    """One controller-time route label and executable-action mask."""

    lane_action_mask: np.ndarray
    route_intent: int
    route_intent_valid: bool
    current_edge: str | None
    current_lane_index: int | None
    next_route_edge: str | None
    reason: str

    @property
    def route_intent_one_hot(self) -> np.ndarray:
        value = np.zeros(3, dtype=np.float32)
        if self.route_intent_valid:
            value[lane_command_index(self.route_intent)] = 1.0
        return value

    @property
    def non_keep_feasible(self) -> bool:
        return bool(self.lane_action_mask[0] > 0.5 or self.lane_action_mask[2] > 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_action_mask": self.lane_action_mask.astype(float).tolist(),
            "route_intent": int(self.route_intent),
            "route_intent_valid": bool(self.route_intent_valid),
            "route_intent_one_hot": self.route_intent_one_hot.astype(float).tolist(),
            "non_keep_feasible": self.non_keep_feasible,
            "current_edge": self.current_edge,
            "current_lane_index": self.current_lane_index,
            "next_route_edge": self.next_route_edge,
            "reason": self.reason,
        }


def _invalid_context(
    reason: str,
    *,
    lane_action_mask: np.ndarray | None = None,
    current_edge: str | None = None,
    current_lane_index: int | None = None,
    next_route_edge: str | None = None,
) -> DecisionAlignmentContext:
    mask = (
        np.asarray(lane_action_mask, dtype=np.float32)
        if lane_action_mask is not None
        else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    return DecisionAlignmentContext(
        lane_action_mask=mask,
        route_intent=0,
        route_intent_valid=False,
        current_edge=current_edge,
        current_lane_index=current_lane_index,
        next_route_edge=next_route_edge,
        reason=reason,
    )


def _link_target_edges(env: Any, lane_id: str) -> set[str]:
    """Return non-internal approached edges for all current lane links."""

    edges: set[str] = set()
    links = env._connection.lane.getLinks(lane_id, extended=True)
    for link in links:
        if not link:
            continue
        approached_lane = str(link[0])
        if approached_lane:
            edges.add(str(env._lane_edge_id(approached_lane)))
    return edges


def decision_alignment_context(env: Any) -> DecisionAlignmentContext:
    """Compute a mask and deterministic route intent from current live state.

    The mask exactly mirrors :meth:`SumoSceneEnv._apply_control`: keep is
    always available, while a non-keep command is available only on a
    non-internal edge with an adjacent driving lane under the scenario's
    SMARTS/CARLA sign convention.
    """

    connection = getattr(env, "_connection", None)
    if connection is None:
        return _invalid_context("environment_not_running")
    ego_id = env.specification.ego_id
    if ego_id not in connection.vehicle.getIDList():
        return _invalid_context("ego_not_active")

    try:
        road_id = str(connection.vehicle.getRoadID(ego_id))
        lane_index = int(connection.vehicle.getLaneIndex(ego_id))
        route = [str(edge) for edge in connection.vehicle.getRoute(ego_id)]
        route_index = max(0, int(connection.vehicle.getRouteIndex(ego_id)))
    except env._traci.TraCIException:
        return _invalid_context("traci_state_error")

    if not road_id or road_id.startswith(":"):
        return _invalid_context(
            "internal_or_missing_edge",
            current_edge=road_id or None,
            current_lane_index=lane_index,
        )

    driving_lanes = tuple(int(value) for value in env._driving_lanes(road_id))
    if lane_index not in driving_lanes:
        return _invalid_context(
            "current_lane_not_driving",
            current_edge=road_id,
            current_lane_index=lane_index,
        )

    rank = driving_lanes.index(lane_index)
    mask = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    target_lane_by_command: dict[int, str] = {}
    for command in (-1, 1):
        target_rank = rank + int(env._sumo_lane_offset(command))
        if 0 <= target_rank < len(driving_lanes):
            mask[lane_command_index(command)] = 1.0
            target_lane_by_command[command] = f"{road_id}_{driving_lanes[target_rank]}"

    route_position: int | None = None
    if route_index < len(route) and route[route_index] == road_id:
        route_position = route_index
    else:
        for index in range(min(route_index, len(route)), len(route)):
            if route[index] == road_id:
                route_position = index
                break
    if route_position is None:
        return _invalid_context(
            "current_edge_not_in_remaining_route",
            lane_action_mask=mask,
            current_edge=road_id,
            current_lane_index=lane_index,
        )
    if route_position + 1 >= len(route):
        return _invalid_context(
            "route_has_no_next_edge",
            lane_action_mask=mask,
            current_edge=road_id,
            current_lane_index=lane_index,
        )

    next_edge = route[route_position + 1]
    current_lane_id = f"{road_id}_{lane_index}"
    try:
        if next_edge in _link_target_edges(env, current_lane_id):
            return DecisionAlignmentContext(
                lane_action_mask=mask,
                route_intent=0,
                route_intent_valid=True,
                current_edge=road_id,
                current_lane_index=lane_index,
                next_route_edge=next_edge,
                reason="current_lane_reaches_route",
            )
        route_commands = [
            command
            for command, target_lane in target_lane_by_command.items()
            if next_edge in _link_target_edges(env, target_lane)
        ]
    except env._traci.TraCIException:
        return _invalid_context(
            "traci_lane_link_error",
            lane_action_mask=mask,
            current_edge=road_id,
            current_lane_index=lane_index,
            next_route_edge=next_edge,
        )

    if len(route_commands) == 1:
        return DecisionAlignmentContext(
            lane_action_mask=mask,
            route_intent=route_commands[0],
            route_intent_valid=True,
            current_edge=road_id,
            current_lane_index=lane_index,
            next_route_edge=next_edge,
            reason="adjacent_lane_reaches_route",
        )
    return _invalid_context(
        "ambiguous_adjacent_route" if route_commands else "no_adjacent_route_connection",
        lane_action_mask=mask,
        current_edge=road_id,
        current_lane_index=lane_index,
        next_route_edge=next_edge,
    )


__all__ = [
    "DecisionAlignmentContext",
    "KEEP_COMMAND_INDEX",
    "LANE_COMMANDS",
    "decision_alignment_context",
    "lane_command_index",
]

