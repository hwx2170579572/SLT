from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.sumo.decision_alignment_v4 import (
    decision_alignment_context,
    lane_command_index,
)
from envs.sumo.paper_env_v2 import PaperSumoSceneEnvV2
from tools.diagnose_decision_alignment_v4 import (
    _classification_summary,
    _lane_commands,
)


class _FakeVehicle:
    def __init__(self, *, road: str, lane: int, route: list[str], route_index: int) -> None:
        self.road = road
        self.lane = lane
        self.route = route
        self.route_index = route_index

    @staticmethod
    def getIDList() -> list[str]:
        return ["ego"]

    def getRoadID(self, vehicle_id: str) -> str:
        assert vehicle_id == "ego"
        return self.road

    def getLaneIndex(self, vehicle_id: str) -> int:
        assert vehicle_id == "ego"
        return self.lane

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]:
        assert vehicle_id == "ego"
        return tuple(self.route)

    def getRouteIndex(self, vehicle_id: str) -> int:
        assert vehicle_id == "ego"
        return self.route_index


class _FakeLane:
    def __init__(self, links: dict[str, list[str]]) -> None:
        self.links = links

    def getLinks(self, lane_id: str, extended: bool = True):
        assert extended is True
        return [(target,) for target in self.links.get(lane_id, [])]


class _FakeTraci:
    TraCIException = RuntimeError


class _FakeEnv:
    def __init__(
        self,
        *,
        contract: str,
        road: str = "north_upper",
        lane: int = 1,
        driving_lanes: tuple[int, ...] = (1, 2),
        links: dict[str, list[str]] | None = None,
    ) -> None:
        self.contract = contract
        self._lanes = driving_lanes
        self.specification = SimpleNamespace(ego_id="ego")
        self._traci = _FakeTraci()
        self._connection = SimpleNamespace(
            vehicle=_FakeVehicle(
                road=road,
                lane=lane,
                route=["ego_approach", road, "goal_out"],
                route_index=1,
            ),
            lane=_FakeLane(
                links
                or {
                    f"{road}_1": ["wrong_out_1"],
                    f"{road}_2": ["goal_out_1"],
                }
            ),
        )

    def _driving_lanes(self, edge_id: str) -> tuple[int, ...]:
        assert edge_id == self._connection.vehicle.road
        return self._lanes

    def _sumo_lane_offset(self, lane_command: int) -> int:
        return lane_command if self.contract == "smarts" else -lane_command

    @staticmethod
    def _lane_edge_id(lane_id: str) -> str:
        return lane_id.rsplit("_", 1)[0]


def test_carla_route_intent_uses_negative_command_for_left_lane() -> None:
    context = decision_alignment_context(_FakeEnv(contract="carla"))
    assert context.route_intent_valid is True
    assert context.route_intent == -1
    assert context.reason == "adjacent_lane_reaches_route"
    np.testing.assert_array_equal(context.lane_action_mask, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(context.route_intent_one_hot, [1.0, 0.0, 0.0])


def test_smarts_sign_maps_the_same_sumo_left_lane_to_positive() -> None:
    context = decision_alignment_context(_FakeEnv(contract="smarts"))
    assert context.route_intent_valid is True
    assert context.route_intent == 1
    np.testing.assert_array_equal(context.lane_action_mask, [0.0, 1.0, 1.0])


def test_current_lane_route_connection_has_keep_priority() -> None:
    env = _FakeEnv(
        contract="carla",
        links={
            "north_upper_1": ["goal_out_1"],
            "north_upper_2": ["goal_out_1"],
        },
    )
    context = decision_alignment_context(env)
    assert context.route_intent_valid is True
    assert context.route_intent == 0
    assert context.reason == "current_lane_reaches_route"


def test_internal_edge_is_invalid_and_exposes_only_keep() -> None:
    env = _FakeEnv(contract="carla", road=":split_0", lane=0, driving_lanes=(0,))
    context = decision_alignment_context(env)
    assert context.route_intent_valid is False
    assert context.reason == "internal_or_missing_edge"
    np.testing.assert_array_equal(context.lane_action_mask, [0.0, 1.0, 0.0])


def test_ambiguous_two_sided_route_connection_is_not_supervised() -> None:
    env = _FakeEnv(
        contract="smarts",
        lane=2,
        driving_lanes=(1, 2, 3),
        links={
            "north_upper_1": ["goal_out_1"],
            "north_upper_2": ["wrong_out_1"],
            "north_upper_3": ["goal_out_1"],
        },
    )
    context = decision_alignment_context(env)
    assert context.route_intent_valid is False
    assert context.reason == "ambiguous_adjacent_route"
    np.testing.assert_array_equal(context.lane_action_mask, [1.0, 1.0, 1.0])


def test_lane_command_helpers_and_balanced_accuracy() -> None:
    actions = np.asarray([[0.0, -0.5], [0.0, 0.0], [0.0, 0.8]], dtype=np.float32)
    np.testing.assert_array_equal(_lane_commands(actions), [-1, 0, 1])
    assert lane_command_index(-1) == 0
    assert lane_command_index(0) == 1
    assert lane_command_index(1) == 2
    summary = _classification_summary(
        np.asarray([-1, 0, 0, 1]), np.asarray([-1, 0, 1, 1])
    )
    assert summary["accuracy"] == 0.75
    assert summary["balanced_accuracy"] == (1.0 + 1.0 + 0.5) / 3.0


def test_real_carla_context_identifies_required_negative_window() -> None:
    env = PaperSumoSceneEnvV2(
        scenario="carla",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        _, _ = env.reset(seed=30000)
        initial = decision_alignment_context(env)
        assert initial.route_intent_valid is True
        assert initial.route_intent == 0
        found = None
        for _ in range(101):
            context = decision_alignment_context(env)
            if context.current_edge == "north_upper" and context.route_intent_valid:
                found = context
                break
            _, _, terminated, truncated, _ = env.step(
                np.asarray([1.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert found is not None
        assert found.next_route_edge == "goal_out"
        assert found.current_lane_index == 1
        assert found.route_intent == -1
        assert found.non_keep_feasible is True
    finally:
        env.close()

