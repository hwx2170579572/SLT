from __future__ import annotations

from pathlib import Path
import sys


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.schema import empty_observation, make_observation_space  # noqa: E402


def test_candidate_coverage_unknown_sentinel_is_inside_declared_space() -> None:
    space = make_observation_space(
        max_actors=2,
        history_samples=3,
        max_candidates=4,
        max_path_lanes=16,
        zone_count=2,
        deadline_seconds=60.0,
    )
    observation = empty_observation(
        max_actors=2,
        history_samples=3,
        max_candidates=4,
        max_path_lanes=16,
        zone_count=2,
        deadline_seconds=60.0,
    )
    observation["candidate_unknown_count"][0] = -1

    assert space.contains(observation)

