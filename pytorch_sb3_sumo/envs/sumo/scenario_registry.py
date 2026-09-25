"""Static metadata for the SUMO equivalents of the released scenarios.

The original project hard-codes the episode limits in ``tools/test.py`` and
uses a different waypoint tensor for CARLA.  The registry keeps those public
contracts explicit so the simulator adapter does not need scenario-specific
conditionals scattered throughout the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_SUMO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SumoScenarioSpec:
    """Description of one scenario and its observation contract."""

    name: str
    max_episode_steps: int
    network_name: str = "intersection"
    map_paths_per_actor: int = 2
    map_feature_dim: int = 5
    include_pedestrians: bool = False
    ego_id: str = "ego"

    @property
    def config_path(self) -> Path:
        return _SUMO_ROOT / "scenarios" / self.name / "scenario.sumocfg"

    @property
    def network_path(self) -> Path:
        return (
            _SUMO_ROOT
            / "networks"
            / self.network_name
            / f"{self.network_name}.net.xml"
        )


SCENARIOS: dict[str, SumoScenarioSpec] = {
    "left_turn": SumoScenarioSpec("left_turn", max_episode_steps=400),
    "cross": SumoScenarioSpec(
        "cross", max_episode_steps=600, network_name="double_merge"
    ),
    # 4-way priority (yield) intersection; ego takes an unprotected left turn
    # south_in -> west_out while social traffic goes straight + right.
    "cross_left": SumoScenarioSpec("cross_left", max_episode_steps=600),
    # The legacy CARLA scene records three candidate polylines with x/y only.
    "carla": SumoScenarioSpec(
        "carla",
        max_episode_steps=300,
        map_paths_per_actor=3,
        map_feature_dim=2,
        include_pedestrians=True,
    ),
    "roundabout": SumoScenarioSpec(
        "roundabout", max_episode_steps=1000, network_name="roundabout"
    ),
    "roundabout_easy": SumoScenarioSpec(
        "roundabout_easy", max_episode_steps=400, network_name="roundabout"
    ),
    "roundabout_medium": SumoScenarioSpec(
        "roundabout_medium", max_episode_steps=600, network_name="roundabout"
    ),
}


def get_scenario_spec(name: str) -> SumoScenarioSpec:
    """Return a validated scenario specification."""

    try:
        spec = SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown SUMO scenario {name!r}; choose one of: {choices}") from exc
    if not spec.config_path.is_file():
        raise FileNotFoundError(f"SUMO config is missing: {spec.config_path}")
    if not spec.network_path.is_file():
        raise FileNotFoundError(
            f"SUMO network is missing: {spec.network_path}. "
            "Run envs/sumo/build_networks.py first."
        )
    return spec


def available_scenarios() -> tuple[str, ...]:
    return tuple(SCENARIOS)
