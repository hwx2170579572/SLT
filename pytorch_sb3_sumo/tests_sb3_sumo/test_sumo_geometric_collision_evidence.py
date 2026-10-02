from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from envs.sumo.sumo_env import SumoSceneEnv


class FakeTraCIException(Exception):
    pass


class FakeVehicleAPI:
    def __init__(self, positions: dict[str, tuple[float, float]]) -> None:
        self.positions = positions
        self.position_calls: list[str] = []

    def getIDList(self) -> list[str]:
        return list(self.positions)

    def getPosition(self, actor_id: str) -> tuple[float, float]:
        self.position_calls.append(actor_id)
        return self.positions[actor_id]

    def getAngle(self, _actor_id: str) -> float:
        return 0.0

    def getLength(self, _actor_id: str) -> float:
        return 4.0

    def getWidth(self, _actor_id: str) -> float:
        return 2.0


class FakePersonAPI:
    def __init__(self, positions: dict[str, tuple[float, float]]) -> None:
        self.positions = positions
        self.position_calls: list[str] = []

    def getIDList(self) -> list[str]:
        return list(self.positions)

    def getPosition(self, person_id: str) -> tuple[float, float]:
        self.position_calls.append(person_id)
        return self.positions[person_id]

    def getAngle(self, _person_id: str) -> float:
        return 0.0


def _collision_env(
    vehicle_positions: dict[str, tuple[float, float]],
    *,
    include_pedestrians: bool = False,
    person_positions: dict[str, tuple[float, float]] | None = None,
) -> SumoSceneEnv:
    env = SumoSceneEnv.__new__(SumoSceneEnv)
    vehicle = FakeVehicleAPI(vehicle_positions)
    person = FakePersonAPI(person_positions or {})
    env.specification = SimpleNamespace(
        ego_id="ego",
        include_pedestrians=include_pedestrians,
        source_observation_contract="cartesian",
        max_episode_steps=100,
    )
    env._connection = SimpleNamespace(vehicle=vehicle, person=person)
    env._traci = SimpleNamespace(TraCIException=FakeTraCIException)
    env._actors_hidden_this_observation = set()
    env._last_geometric_collision_partner = None
    env._last_geometric_collision = False
    env._last_raw_sumo_arrived = False
    env._last_raw_sumo_collision = False
    env._behavior_diagnostics = None
    env._raw_steps = 1
    return env


def test_vehicle_first_hit_records_id_and_preserves_short_circuit_order() -> None:
    env = _collision_env(
        {"ego": (0.0, 0.0), "first": (0.0, 0.0), "second": (0.0, 0.0)}
    )
    try:
        assert env._geometric_collision() is True
        assert env._last_geometric_collision_partner == {
            "kind": "vehicle",
            "id": "first",
        }
        # The old detector returns on the first overlapping actor. Metadata
        # must be captured in that branch rather than by a second scan.
        assert env._connection.vehicle.position_calls == ["ego", "first"]
    finally:
        env._connection = None


def test_no_collision_clears_partner_cache() -> None:
    env = _collision_env({"ego": (0.0, 0.0), "far": (100.0, 100.0)})
    env._last_geometric_collision_partner = {"kind": "vehicle", "id": "stale"}
    try:
        assert env._geometric_collision() is False
        assert env._last_geometric_collision_partner is None
    finally:
        env._connection = None


def test_close_clears_partner_cache_between_episodes() -> None:
    env = _collision_env({"ego": (0.0, 0.0)})
    env._connection = None
    env._last_geometric_collision_partner = {"kind": "vehicle", "id": "old"}
    try:
        env.close()
        assert env._last_geometric_collision_partner is None
    finally:
        env._connection = None


def test_pedestrian_first_hit_records_person_domain_and_short_circuits() -> None:
    env = _collision_env(
        {"ego": (0.0, 0.0)},
        include_pedestrians=True,
        person_positions={"person-first": (0.0, -2.0), "person-second": (0.0, -2.0)},
    )
    try:
        assert env._geometric_collision() is True
        assert env._last_geometric_collision_partner == {
            "kind": "person",
            "id": "person-first",
        }
        assert env._connection.person.position_calls == ["person-first"]
    finally:
        env._connection = None


def test_event_path_clears_stale_metadata_even_when_bool_override_is_used() -> None:
    env = _collision_env({"ego": (0.0, 0.0)})
    simulation = SimpleNamespace(
        getArrivedIDList=lambda: [],
        getCollidingVehiclesIDList=lambda: [],
        getCollisions=lambda: [],
        getStartingTeleportIDList=lambda: [],
    )
    env._connection.simulation = simulation
    env._last_geometric_collision_partner = {"kind": "vehicle", "id": "old"}
    # Preserve compatibility with tests/subclasses that override the existing
    # boolean hook and do not provide diagnostic metadata.
    env._geometric_collision = lambda: False  # type: ignore[method-assign]
    try:
        result = env._events_after_step()
        assert result == (False, False, False, False)
        assert env._last_geometric_collision_partner is None
    finally:
        env._connection = None


def test_event_path_keeps_collision_bool_and_captures_first_vehicle_hit() -> None:
    env = _collision_env({"ego": (0.0, 0.0), "partner": (0.0, 0.0)})
    env._connection.simulation = SimpleNamespace(
        getArrivedIDList=lambda: [],
        getCollidingVehiclesIDList=lambda: [],
        getCollisions=lambda: [],
        getStartingTeleportIDList=lambda: [],
    )
    try:
        result = env._events_after_step()
        assert result == (False, True, False, False)
        assert env._last_geometric_collision is True
        assert env._last_raw_sumo_collision is False
        assert env._last_geometric_collision_partner == {
            "kind": "vehicle",
            "id": "partner",
        }
    finally:
        env._connection = None


def _snapshot_env(
    *,
    include_partner: bool = True,
    partner_state_time: float | None = 12.3,
    collided: bool = True,
) -> SumoSceneEnv:
    env = SumoSceneEnv.__new__(SumoSceneEnv)
    env.specification = SimpleNamespace(ego_id="ego", max_episode_steps=100)
    env._connection = SimpleNamespace(
        simulation=SimpleNamespace(getTime=lambda: 12.3)
    )
    env._behavior_step_seconds = 0.1
    env._behavior_pending_errors = []
    env._behavior_actor_states = {"vehicle:ego": np.zeros(5, dtype=np.float32)}
    if include_partner:
        env._behavior_actor_states["vehicle:partner"] = np.zeros(5, dtype=np.float32)

    def payload(actor_key: str, _state: Any, sim_time: float, _errors: list[Any]):
        actor_id = actor_key.split(":", 1)[1]
        state_time = partner_state_time if actor_id == "partner" else sim_time
        return {
            "id": actor_id,
            "key": actor_key,
            "kind": actor_key.split(":", 1)[0],
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "heading": 0.0,
            "speed": 0.0,
            "length": 4.0,
            "width": 2.0,
            "road_id": "edge",
            "lane_id": "edge_0",
            "route_index": 0,
            "state_source": "current",
            "state_time": state_time,
        }

    env._behavior_actor_payload = payload
    env._behavior_collision_events = []
    env._behavior_collision_ids = []
    env._behavior_observed_neighbor_ids = ("vehicle:partner",)
    env._behavior_observation_raw_step = 10
    env._behavior_route_lane_facts = lambda _ego: {}
    env._behavior_active_control = None
    env._last_effective_target_speed = 0.0
    env._behavior_last_sim_time = None
    env._raw_steps = 11
    env._lifetime_raw_steps = 11
    env._last_geometric_collision = collided
    env._last_geometric_collision_partner = (
        {"kind": "vehicle", "id": "partner"} if collided else None
    )
    env._last_raw_sumo_arrived = False
    env._last_raw_sumo_collision = False
    return env


def test_snapshot_joins_partner_to_same_time_state_and_last_observation() -> None:
    env = _snapshot_env()
    try:
        snapshot = env._behavior_snapshot((False, True, False, False))
        evidence = snapshot["geometric_collision_evidence"]
        assert evidence["source"] == "geometric_obb_fallback"
        assert evidence["partner_key"] == "vehicle:partner"
        assert evidence["partner_capture_status"] == "current_same_time"
        assert evidence["partner_state_time_matches_input"] is True
        assert evidence["partner_in_last_policy_observation"] is True
        assert evidence["observed_neighbor_raw_step"] == 10
        assert evidence["input_raw_step"] == snapshot["raw_step"] == 11
    finally:
        env._connection = None


def test_snapshot_marks_missing_or_time_mismatched_partner_as_unknown() -> None:
    missing = _snapshot_env(include_partner=False)
    mismatched = _snapshot_env(partner_state_time=12.2)
    try:
        missing_evidence = missing._behavior_snapshot(
            (False, True, False, False)
        )["geometric_collision_evidence"]
        mismatched_evidence = mismatched._behavior_snapshot(
            (False, True, False, False)
        )["geometric_collision_evidence"]
        assert missing_evidence["partner_capture_status"] == (
            "partner_missing_from_current_snapshot"
        )
        assert missing_evidence["partner_state_time_matches_input"] is None
        assert mismatched_evidence["partner_capture_status"] == (
            "partner_snapshot_time_mismatch"
        )
        assert mismatched_evidence["partner_state_time_matches_input"] is False
        assert mismatched_evidence["partner_in_last_policy_observation"] is True
    finally:
        missing._connection = None
        mismatched._connection = None


def test_snapshot_uses_null_evidence_when_geometric_collision_is_false() -> None:
    env = _snapshot_env(collided=False)
    try:
        snapshot = env._behavior_snapshot((False, False, False, False))
        assert snapshot["geometric_collision_evidence"] is None
    finally:
        env._connection = None
