"""Typed observation contract for the independent scene-event environment."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


UNKNOWN = 0
COMPLETE_LEGAL = 1
INCOMPLETE = 2

ZONE_UNKNOWN = 0
ZONE_NO_EVENT = 1
ZONE_CV_VALID = 2
ZONE_STATIONARY = 3
ZONE_BEYOND_HORIZON = 4

# Bit flags for topology relations that are present in the public lane graph
# but have no mapped event zone. These are explicit unknown relation messages,
# not collision labels or evidence of safe separation.
PAIR_TOPOLOGY_FOE_WITHOUT_ZONE = 1
PAIR_TOPOLOGY_ADJACENT_EXIT_ONLY = 2

ACTOR_FEATURES = ("x_center_m", "y_center_m", "heading_x", "heading_y", "vx_mps", "vy_mps")


def observation_shapes(max_actors: int, history_samples: int, max_candidates: int,
                       max_path_lanes: int, zone_count: int) -> dict[str, tuple[int, ...]]:
    a, h, c, p, z = map(int, (max_actors, history_samples, max_candidates, max_path_lanes, zone_count))
    if min(a, h, c, p) <= 0 or z < 0:
        raise ValueError("actor/history/candidate/path capacities must be positive; zones nonnegative")
    return {
        "actor_history": (a, h, 6),
        "history_valid": (a, h),
        "actor_valid": (a,),
        "actor_size_lw": (a, 2),
        "actor_age_s": (a,),
        "actor_lane_ptr": (a,),
        "actor_lane_s_m": (a,),
        "lane_match_conf": (a,),
        "remaining_time_s": (1,),
        "candidate_lane_ptr": (a, c, p),
        "candidate_lane_mask": (a, c, p),
        "candidate_valid": (a, c),
        "candidate_status": (a, c),
        "candidate_progress_m": (a, c),
        "candidate_unknown_count": (a,),
        "candidate_zone_event_s": (a, c, z, 2),
        "candidate_zone_event_mask": (a, c, z),
        "candidate_zone_status": (a, c, z),
        # Signed distance along the public candidate route from the current
        # vehicle center to the footprint entry/clear boundaries. Kept separate
        # from CV timing so stationary actors retain route-order geometry.
        "candidate_zone_s_m": (a, c, z, 2),
        "candidate_zone_path_mask": (a, c, z),
        # Sparse-by-value candidate-route pair flags. A nonzero value means a
        # public typed topology relation was audited but has no mapped event
        # zone; zero does not certify safety or absence of all interactions.
        "candidate_pair_topology_status": (a, c, a, c),
    }


def make_observation_space(max_actors: int, history_samples: int, max_candidates: int,
                           max_path_lanes: int, zone_count: int, deadline_seconds: float) -> spaces.Dict:
    shapes = observation_shapes(max_actors, history_samples, max_candidates, max_path_lanes, zone_count)
    result: dict[str, spaces.Space] = {}
    boolean_keys = {"history_valid", "actor_valid", "candidate_lane_mask", "candidate_valid",
                    "candidate_zone_event_mask", "candidate_zone_path_mask"}
    uint8_keys = {"candidate_pair_topology_status"}
    integer_keys = {"actor_lane_ptr", "candidate_lane_ptr", "candidate_status", "candidate_unknown_count",
                    "candidate_zone_status"}
    for key, shape in shapes.items():
        if key in boolean_keys:
            result[key] = spaces.Box(0, 1, shape=shape, dtype=np.bool_)
        elif key in uint8_keys:
            result[key] = spaces.Box(0, 3, shape=shape, dtype=np.uint8)
        elif key in integer_keys:
            low = -1 if key in {"actor_lane_ptr", "candidate_lane_ptr", "candidate_unknown_count"} else 0
            result[key] = spaces.Box(low, np.iinfo(np.int32).max, shape=shape, dtype=np.int32)
        elif key == "remaining_time_s":
            result[key] = spaces.Box(0.0, float(deadline_seconds), shape=shape, dtype=np.float32)
        else:
            result[key] = spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
    return spaces.Dict(result)


def empty_observation(max_actors: int, history_samples: int, max_candidates: int,
                      max_path_lanes: int, zone_count: int, deadline_seconds: float) -> dict[str, np.ndarray]:
    shapes = observation_shapes(max_actors, history_samples, max_candidates, max_path_lanes, zone_count)
    bool_keys = {"history_valid", "actor_valid", "candidate_lane_mask", "candidate_valid",
                 "candidate_zone_event_mask", "candidate_zone_path_mask"}
    uint8_keys = {"candidate_pair_topology_status"}
    int_keys = {"actor_lane_ptr", "candidate_lane_ptr", "candidate_status", "candidate_unknown_count", "candidate_zone_status"}
    out: dict[str, np.ndarray] = {}
    for key, shape in shapes.items():
        if key in bool_keys:
            out[key] = np.zeros(shape, dtype=np.bool_)
        elif key in uint8_keys:
            out[key] = np.zeros(shape, dtype=np.uint8)
        elif key in int_keys:
            out[key] = np.full(shape, -1 if key in {"actor_lane_ptr", "candidate_lane_ptr"} else UNKNOWN,
                               dtype=np.int32)
        else:
            out[key] = np.zeros(shape, dtype=np.float32)
    out["candidate_unknown_count"].fill(0)
    out["remaining_time_s"][0] = np.float32(deadline_seconds)
    return out


def validate_observation(observation: dict[str, Any], *, max_actors: int, history_samples: int,
                         max_candidates: int, max_path_lanes: int, zone_count: int,
                         deadline_seconds: float) -> None:
    expected = observation_shapes(max_actors, history_samples, max_candidates, max_path_lanes, zone_count)
    if set(observation) != set(expected):
        raise ValueError(f"observation keys differ: missing={set(expected)-set(observation)}, extra={set(observation)-set(expected)}")
    bool_keys = {"history_valid", "actor_valid", "candidate_lane_mask", "candidate_valid",
                 "candidate_zone_event_mask", "candidate_zone_path_mask"}
    uint8_keys = {"candidate_pair_topology_status"}
    int_keys = {"actor_lane_ptr", "candidate_lane_ptr", "candidate_status", "candidate_unknown_count", "candidate_zone_status"}
    for key, shape in expected.items():
        value = observation[key]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"{key} must be ndarray with shape {shape}")
        expected_dtype = np.dtype(np.bool_ if key in bool_keys else
                                  np.uint8 if key in uint8_keys else
                                  np.int32 if key in int_keys else np.float32)
        if value.dtype != expected_dtype:
            raise ValueError(f"{key} dtype must be {expected_dtype}, got {value.dtype}")
        if key in uint8_keys and np.any(value > 3):
            raise ValueError(f"{key} contains an unknown topology status bit")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    remaining = observation["remaining_time_s"]
    if np.any(remaining < 0) or np.any(remaining > deadline_seconds):
        raise ValueError("remaining_time_s is outside the task deadline")
