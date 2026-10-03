from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from shapely.geometry import LineString, box


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.collector import SceneEventCollector  # noqa: E402
from scene_event.env import SceneEventRawEnv  # noqa: E402
from scene_event.schema import empty_observation, make_observation_space, validate_observation  # noqa: E402


def _public_map() -> dict[str, np.ndarray]:
    return {
        "lane_points_xy": np.zeros((2, 10, 2), dtype=np.float32),
        "lane_attrs": np.zeros((2, 8), dtype=np.float32),
        "lane_valid": np.ones((2,), dtype=bool),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_type": np.zeros((0,), dtype=np.int64),
        "edge_valid": np.ones((0,), dtype=bool),
        "ego_route_lane_mask": np.ones((2,), dtype=bool),
        "zone_polygons_xy": np.zeros((1, 4, 2), dtype=np.float32),
        "zone_vertex_valid": np.ones((1, 4), dtype=bool),
        "lane_zone_s_m": np.asarray([[[20.0, 30.0]], [[20.0, 30.0]]], dtype=np.float32),
        "lane_zone_valid": np.ones((2, 1), dtype=bool),
        "lane_zone_distance_m": np.zeros((2, 1), dtype=np.float32),
        "zone_type": np.zeros((1,), dtype=np.int64),
        "zone_valid": np.ones((1,), dtype=bool),
    }


class _FakeVehicles:
    def __init__(self):
        self.fail_position = set()
        self.rows = {
            "ego": ((2.0, 0.0), 90.0, 0.0, "lane_ego", 12.0),
            "social": ((7.0, 0.0), 90.0, 0.0, "lane_social", 12.0),
        }

    def getIDList(self):
        return list(self.rows)

    def getPosition(self, vehicle_id):
        if vehicle_id in self.fail_position:
            raise RuntimeError("synthetic state read failure")
        return self.rows[vehicle_id][0]

    def getAngle(self, vehicle_id):
        return self.rows[vehicle_id][1]

    def getSpeed(self, vehicle_id):
        return self.rows[vehicle_id][2]

    def getLaneID(self, vehicle_id):
        return self.rows[vehicle_id][3]

    def getLanePosition(self, vehicle_id):
        return self.rows[vehicle_id][4]


class _FakeEnv:
    def __init__(self, vehicles):
        self._connection = SimpleNamespace(vehicle=vehicles)
        self.specification = SimpleNamespace(ego_id="ego")

    @staticmethod
    def _vehicle_dimensions(_vehicle_id):
        return (4.0, 2.0)


def test_scene_observation_has_explicit_history_masks_and_center_aligned_lane_s() -> None:
    config = SimpleNamespace(
        max_actors=2, history_samples=3, observation_radius=80.0,
        raw_dt=0.1, prediction_seconds=3.0, deadline_seconds=60.0,
    )
    map_cache = SimpleNamespace(
        public_map=_public_map(),
        zone_geometries=(box(20.0, -2.0, 30.0, 2.0),),
        lane_lines=(LineString(((0.0, 0.0), (100.0, 0.0))),
                    LineString(((0.0, 0.0), (100.0, 0.0)))),
        lane_ids=("lane_ego", "lane_social"),
        lane_lengths_m=np.asarray((100.0, 100.0), dtype=np.float32),
        lane_successors=((), ()),
    )
    collector = SceneEventCollector(config, map_cache)
    collector.set_ego_id("ego")
    collector.begin_episode(7)
    vehicles = _FakeVehicles()
    env = _FakeEnv(vehicles)

    collector.capture_raw_tick(env)
    observation = collector.observation(60.0)
    assert observation["actor_valid"].tolist() == [True, True]
    assert observation["actor_history"][0, -1, 0] == 0.0  # x=0 is a valid state
    assert observation["history_valid"][1].tolist() == [False, False, True]
    assert observation["actor_history"][1, -1, 0] == 5.0
    assert observation["actor_lane_s_m"].tolist() == [10.0, 10.0]
    assert collector.last_collection["frame_raw_tick"][0].tolist() == [-1, -1, 0]
    assert observation["candidate_zone_path_mask"][0, 0, 0]
    assert observation["candidate_zone_status"][0, 0, 0] == 3  # stationary, geometry is known
    assert observation["candidate_zone_s_m"][0, 0, 0].tolist() == [8.0, 22.0]

    vehicles.rows["social"] = ((102.0, 0.0), 90.0, 0.0, "lane_social", 112.0)
    collector.capture_raw_tick(env)
    assert collector.observation(59.9)["actor_valid"].tolist() == [True, False]

    vehicles.rows["social"] = ((5.0, 0.0), 90.0, 0.0, "lane_social", 12.0)
    collector.capture_raw_tick(env)
    observation = collector.observation(59.8)
    assert observation["history_valid"][1].tolist() == [True, False, True]
    assert observation["actor_age_s"][1] == np.float32(0.2)
    space = make_observation_space(2, 3, 4, 16, 1, 60.0)
    assert space.contains(observation)
    validate_observation(observation, max_actors=2, history_samples=3, max_candidates=4,
                         max_path_lanes=16, zone_count=1, deadline_seconds=60.0)
    future = collector.future_history("vehicle:social", 0, 2)
    assert [row["raw_tick"] for row in future] == [0, 1, 2]
    assert [row["valid"] for row in future] == [True, False, True]


def test_no_zone_topology_relations_are_explicit_candidate_pair_unknowns() -> None:
    config = SimpleNamespace(
        max_actors=2, history_samples=3, observation_radius=80.0,
        raw_dt=0.1, prediction_seconds=3.0, deadline_seconds=60.0,
    )
    pair_audit = [
        {"lane_pair": ["lane_ego", "lane_social"], "classification": "evidence_backed_foe_relation_without_corridor_overlap",
         "event_zone_created": False, "candidate_relevant": True},
        {"lane_pair": ["lane_ego", "lane_social"], "classification": "evidence_backed_adjacent_exit_topology_only",
         "event_zone_created": False, "candidate_relevant": True},
    ]
    map_cache = SimpleNamespace(
        public_map=_public_map(),
        zone_geometries=(box(20.0, -2.0, 30.0, 2.0),),
        lane_lines=(LineString(((0.0, 0.0), (100.0, 0.0))),
                    LineString(((0.0, 0.0), (100.0, 0.0)))),
        lane_ids=("lane_ego", "lane_social"),
        lane_lengths_m=np.asarray((100.0, 100.0), dtype=np.float32),
        lane_successors=((), ()),
        build_metadata={"zone_geometry": {"pair_relation_audit": pair_audit}},
    )
    collector = SceneEventCollector(config, map_cache)
    collector.set_ego_id("ego")
    collector.begin_episode(23)
    vehicles = _FakeVehicles()
    collector.capture_raw_tick(_FakeEnv(vehicles))
    observation = collector.observation(60.0)

    status = observation["candidate_pair_topology_status"]
    assert status.shape == (2, 4, 2, 4)
    assert status.dtype == np.uint8
    assert status[0, 0, 1, 0] == 3  # bit 1 foe, bit 2 adjacent-exit topology only
    assert status[1, 0, 0, 0] == 3  # route-pair message is symmetric
    assert np.count_nonzero(status) == 2
    summary = collector.summary()
    assert summary["topology_unknown_relation_static_counts"] == {
        "foe_without_zone": 1,
        "adjacent_exit_topology_only": 1,
    }


def test_legacy_base_observation_builder_is_not_called_for_scene_event_policy() -> None:
    # The legacy `_make_observation` route path may call `_actor_paths/getRoute`.
    # The new subclass returns a shape-safe placeholder without touching it.
    env = object.__new__(SceneEventRawEnv)
    env.state_lstm_only = False
    env.include_state_lstm = False
    env.neighbors = 23
    env.history_steps = 21
    env.state_dim = 5
    env.path_length = 16
    env.specification = SimpleNamespace(map_paths_per_actor=2, map_feature_dim=5)
    env._connection = None
    env._actor_paths = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy route builder called"))
    observation = env._make_observation()
    assert observation["trajectory"].shape == (24, 21, 5)
    assert observation["map"].shape == (48, 16, 5)


def test_candidate_unknown_minus_one_is_inside_declared_space() -> None:
    observation = empty_observation(2, 3, 4, 16, 1, 60.0)
    observation["candidate_unknown_count"][0] = -1
    space = make_observation_space(2, 3, 4, 16, 1, 60.0)
    assert space.contains(observation)


def test_failed_and_missing_raw_ticks_keep_explicit_slots() -> None:
    config = SimpleNamespace(
        max_actors=2, history_samples=4, observation_radius=80.0,
        raw_dt=0.1, prediction_seconds=3.0, deadline_seconds=60.0,
    )
    map_cache = SimpleNamespace(
        public_map=_public_map(),
        zone_geometries=(box(20.0, -2.0, 30.0, 2.0),),
        lane_lines=(LineString(((0.0, 0.0), (100.0, 0.0))),
                    LineString(((0.0, 0.0), (100.0, 0.0)))),
        lane_ids=("lane_ego", "lane_social"),
        lane_lengths_m=np.asarray((100.0, 100.0), dtype=np.float32),
        lane_successors=((), ()),
    )
    collector = SceneEventCollector(config, map_cache)
    collector.set_ego_id("ego")
    collector.begin_episode(9)
    vehicles = _FakeVehicles()
    env = _FakeEnv(vehicles)

    collector.capture_raw_tick(env)
    vehicles.fail_position.add("social")
    collector.capture_raw_tick(env)
    vehicles.fail_position.clear()
    collector.capture_raw_tick(env)
    observation = collector.observation(59.7)
    slot = collector.last_collection["actor_keys"].index("vehicle:social")
    assert collector.last_collection["frame_raw_tick"][slot].tolist() == [-1, 0, 1, 2]
    assert observation["history_valid"][slot].tolist() == [False, True, False, True]
    vehicles.rows.pop("social")
    collector.capture_raw_tick(env)
    rows = collector.future_history("vehicle:social", 0, 3)
    assert [row["raw_tick"] for row in rows] == [0, 1, 2, 3]
    assert [row["valid"] for row in rows] == [True, False, True, False]
    assert collector.summary()["raw_state_read_errors"] == 1


def test_zone_entry_clearance_survives_raw_tick_crossing() -> None:
    config = SimpleNamespace(
        max_actors=2, history_samples=3, observation_radius=80.0,
        raw_dt=0.1, prediction_seconds=3.0, deadline_seconds=60.0,
    )
    map_cache = SimpleNamespace(
        public_map=_public_map(),
        zone_geometries=(box(20.0, -2.0, 30.0, 2.0),),
        lane_lines=(LineString(((0.0, 0.0), (100.0, 0.0))),
                    LineString(((0.0, 0.0), (100.0, 0.0)))),
        lane_ids=("lane_ego", "lane_social"),
        lane_lengths_m=np.asarray((100.0, 100.0), dtype=np.float32),
        lane_successors=((), ()),
    )
    collector = SceneEventCollector(config, map_cache)
    collector.set_ego_id("ego")
    collector.begin_episode(11)
    vehicles = _FakeVehicles()
    vehicles.rows["ego"] = ((12.0, 0.0), 90.0, 2.0, "lane_ego", 12.0)
    env = _FakeEnv(vehicles)
    collector.capture_raw_tick(env)
    collector.observation(60.0)

    vehicles.rows["ego"] = ((21.0, 0.0), 90.0, 2.0, "lane_ego", 21.0)
    collector.capture_raw_tick(env)
    inside = collector.observation(59.9)
    inside_distance = inside["candidate_zone_s_m"][0, 0, 0]
    assert np.allclose(inside_distance, (-1.0, 13.0), atol=2e-3)
    assert inside["candidate_zone_path_mask"][0, 0, 0]
    assert inside["candidate_zone_status"][0, 0, 0] == 4  # clearance is beyond the 3 s horizon

    vehicles.rows["ego"] = ((35.0, 0.0), 90.0, 2.0, "lane_ego", 35.0)
    collector.capture_raw_tick(env)
    cleared = collector.observation(59.8)
    assert cleared["candidate_zone_path_mask"][0, 0, 0]
    assert cleared["candidate_zone_s_m"][0, 0, 0, 1] < 0.0
    assert cleared["candidate_zone_status"][0, 0, 0] == 1  # body has cleared the zone
    assert collector.last_collection["frame_raw_tick"][0].tolist() == [0, 1, 2]
