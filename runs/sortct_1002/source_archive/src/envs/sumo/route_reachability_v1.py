"""Route-conditioned lane-node eligibility for the optional reachability method.

Only static SUMO lane connections and the ego vehicle's known route are used.
Labels describe the immediate route-continuation corridor, not collision safety.
The existing encoders and environments do not import this module implicitly.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET


UNKNOWN = -1
INELIGIBLE = 0
ELIGIBLE = 1


@dataclass(frozen=True)
class RouteReachabilityResult:
    labels: tuple
    context_known: bool
    current_edge: str = ""
    next_edge: str = ""


class RouteLaneReachability:
    def __init__(self, lane_edges, successors, internal_lanes, permitted_lanes):
        self.lane_edges = dict(lane_edges)
        self.successors = {key: frozenset(value) for key, value in successors.items()}
        self.internal_lanes = frozenset(internal_lanes)
        self.permitted_lanes = frozenset(permitted_lanes)

    @classmethod
    def from_net(cls, path, vehicle_class="passenger"):
        root = ET.parse(Path(path)).getroot()
        lane_edges = {}
        lane_lookup = {}
        internal_lanes = set()
        permitted_lanes = set()
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            internal = edge.get("function") == "internal" or edge_id.startswith(":")
            for lane in edge.findall("lane"):
                lane_id = lane.get("id")
                if not lane_id:
                    continue
                lane_edges[lane_id] = edge_id
                lane_lookup[(edge_id, lane.get("index", "0"))] = lane_id
                if internal:
                    internal_lanes.add(lane_id)
                allow = set(lane.get("allow", "").split())
                disallow = set(lane.get("disallow", "").split())
                if ((not allow or "all" in allow or vehicle_class in allow)
                        and "all" not in disallow and vehicle_class not in disallow):
                    permitted_lanes.add(lane_id)
        successors = defaultdict(set)
        for connection in root.findall("connection"):
            source = lane_lookup.get((connection.get("from"), connection.get("fromLane", "0")))
            target = lane_lookup.get((connection.get("to"), connection.get("toLane", "0")))
            if source is None or target is None:
                continue
            via = connection.get("via", "").split()
            chain = [source] + via + [target]
            # The XML's from/to pair declares the endpoint connection; internal
            # connection entries additionally retain longer via chains.
            for first, second in zip(chain, chain[1:]):
                if first in permitted_lanes and second in permitted_lanes:
                    successors[first].add(second)
        return cls(lane_edges, successors, internal_lanes, permitted_lanes)

    @lru_cache(maxsize=256)
    def continuation_lanes(self, current_edge, next_edge):
        starts = {lane for lane in self.permitted_lanes
                  if self.lane_edges[lane] == current_edge}
        if not next_edge:
            return frozenset(starts)
        targets = {lane for lane in self.permitted_lanes
                   if self.lane_edges[lane] == next_edge}
        if not starts or not targets:
            return frozenset()
        # Walk backwards from the next route edge through internal lanes only.
        reverse = defaultdict(set)
        for source, destinations in self.successors.items():
            for target in destinations:
                reverse[target].add(source)
        can_finish = set(targets)
        pending = deque(targets)
        while pending:
            target = pending.popleft()
            for source in reverse.get(target, ()):
                if source in self.internal_lanes and source not in can_finish:
                    can_finish.add(source)
                    pending.append(source)
        legal_starts = {lane for lane in starts
                        if any(target in can_finish for target in self.successors.get(lane, ()))}
        result = set(legal_starts)
        pending = deque(legal_starts)
        while pending:
            source = pending.popleft()
            for target in self.successors.get(source, ()):
                if target not in can_finish or target in result:
                    continue
                result.add(target)
                if target in self.internal_lanes:
                    pending.append(target)
        return frozenset(result)

    def classify(self, node_lane_ids, route, route_index, current_road=None):
        node_lane_ids = tuple(node_lane_ids)
        route = tuple(str(edge) for edge in (route or ()))
        try:
            index = int(route_index)
        except (TypeError, ValueError):
            index = -1
        if index < 0 or index >= len(route):
            return RouteReachabilityResult((UNKNOWN,) * len(node_lane_ids), False)
        current_edge = route[index]
        next_edge = route[index + 1] if index + 1 < len(route) else ""
        known_edges = set(self.lane_edges.values())
        if current_edge not in known_edges or (next_edge and next_edge not in known_edges):
            return RouteReachabilityResult((UNKNOWN,) * len(node_lane_ids), False,
                                           current_edge, next_edge)
        if current_road and not current_road.startswith(":") and current_road != current_edge:
            return RouteReachabilityResult((UNKNOWN,) * len(node_lane_ids), False,
                                           current_edge, next_edge)
        eligible = self.continuation_lanes(current_edge, next_edge)
        labels = tuple(
            UNKNOWN if lane_id not in self.lane_edges else
            ELIGIBLE if lane_id in eligible else INELIGIBLE
            for lane_id in node_lane_ids
        )
        return RouteReachabilityResult(labels, True, current_edge, next_edge)


# Optional observation extension. Existing methods never construct this wrapper.
import gymnasium as gym
import numpy as np


class RouteReachabilityObservationWrapper(gym.ObservationWrapper):
    KEY = "route_reachability"
    PROTOCOL = "route_continuation_v1"

    def __init__(self, env, topology_lane_ids, topology_node_count):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("Route reachability requires a Dict observation space")
        if self.KEY in env.observation_space.spaces:
            raise ValueError("Route reachability observation is already present")
        self.node_count = int(topology_node_count)
        lane_ids = tuple(str(value) for value in topology_lane_ids)
        if self.node_count <= 0 or len(lane_ids) > self.node_count:
            raise ValueError("Topology lane IDs do not fit the padded node count")
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("Topology lane IDs must be unique and preserve node order")
        self.network = RouteLaneReachability.from_net(self.unwrapped.specification.network_path)
        missing = set(lane_ids) - set(self.network.lane_edges)
        if missing:
            raise ValueError("Topology lane IDs are absent from the environment network: "
                             + repr(sorted(missing)))
        self.node_lane_ids = lane_ids + (None,) * (self.node_count - len(lane_ids))
        spaces = dict(env.observation_space.spaces)
        spaces[self.KEY] = gym.spaces.Box(-1.0, 1.0, shape=(self.node_count,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.last_route_reachability_info = {}

    def observation(self, observation):
        raw_env = self.unwrapped
        reason = "known"
        try:
            vehicle = raw_env._connection.vehicle
            ego_id = raw_env.specification.ego_id
            route = vehicle.getRoute(ego_id)
            route_index = vehicle.getRouteIndex(ego_id)
            road = vehicle.getRoadID(ego_id)
            result = self.network.classify(self.node_lane_ids, route, route_index, road)
            if not result.context_known:
                reason = "route_context_unknown"
        except Exception as exc:
            # Ego removal at a terminal step is expected. Keep uncertainty
            # explicit; never turn a failed route query into illegal lanes.
            result = RouteReachabilityResult((UNKNOWN,) * self.node_count, False)
            reason = "route_query_unavailable:" + type(exc).__name__
        labels = np.asarray(result.labels, dtype=np.float32)
        self.last_route_reachability_info = {
            "route_reachability_protocol": self.PROTOCOL,
            "route_reachability_context_known": bool(result.context_known),
            "route_reachability_current_edge": result.current_edge,
            "route_reachability_next_edge": result.next_edge,
            "route_reachability_eligible_nodes": int(np.count_nonzero(labels == ELIGIBLE)),
            "route_reachability_ineligible_nodes": int(np.count_nonzero(labels == INELIGIBLE)),
            "route_reachability_unknown_nodes": int(np.count_nonzero(labels == UNKNOWN)),
            "route_reachability_reason": reason,
        }
        output = dict(observation)
        output[self.KEY] = labels
        return output

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.observation(observation)
        info = dict(info)
        info.update(self.last_route_reachability_info)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation(observation)
        info = dict(info)
        info.update(self.last_route_reachability_info)
        return observation, reward, terminated, truncated, info
