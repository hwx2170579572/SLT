"""Released scenario assets and exact mission metadata for paper experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "original_scenarios_v1"


def _traffic_seed(path: Path) -> int:
    match = re.fullmatch(r"traffic_(\d+)\.rou\.xml", path.name)
    if match is None:
        raise ValueError(f"Unexpected released traffic filename: {path}")
    return int(match.group(1))


@dataclass(frozen=True)
class PaperScenarioSpec:
    name: str
    max_episode_steps: int
    ego_depart_time: float
    map_paths_per_actor: int = 2
    map_feature_dim: int = 5
    include_pedestrians: bool = False
    ego_id: str = "ego"
    released_assets: bool = True
    source_observation_contract: str = "smarts"
    waypoint_spacing: float = 1.0
    coordinate_offset: tuple[float, float] = (0.0, 0.0)
    # Static lane-graph padding capacity.  The released 4-way ``cross_left``
    # network has more conflict/merge typed edges than the small double-merge /
    # roundabout / T-junction assets, so it declares a larger edge capacity.
    # Defaults keep every pre-existing scenario byte-identical (64 / 256).
    topology_max_nodes: int = 64
    topology_max_edges: int = 256

    @property
    def asset_directory(self) -> Path:
        return ROOT / self.name

    @property
    def network_path(self) -> Path:
        return self.asset_directory / "map.net.xml"

    @property
    def ego_route_path(self) -> Path:
        return self.asset_directory / "ego.rou.xml"

    @property
    def traffic_paths(self) -> tuple[Path, ...]:
        # SMARTS v0.4.17 Scenario.discover_routes() uses Python's default
        # lexicographic filename ordering, not numeric ordering by seed.
        paths = tuple(
            sorted((self.asset_directory / "traffic").glob("traffic_*.rou.xml"))
        )
        if not paths:
            raise FileNotFoundError(
                f"No released traffic variants found in {self.asset_directory / 'traffic'}"
            )
        return paths


PAPER_SCENARIOS: dict[str, PaperScenarioSpec] = {
    # Episode limits follow the released tools/test.py source, as requested.
    "left_turn": PaperScenarioSpec("left_turn", 400, ego_depart_time=15.0),
    "cross": PaperScenarioSpec("cross", 600, ego_depart_time=5.0),
    "cross_left": PaperScenarioSpec(
        "cross_left", 600, ego_depart_time=15.0,
        topology_max_edges=512,
    ),
    "roundabout_easy": PaperScenarioSpec(
        "roundabout_easy", 400, ego_depart_time=30.0
    ),
    "roundabout_medium": PaperScenarioSpec(
        "roundabout_medium", 600, ego_depart_time=30.0
    ),
    "roundabout": PaperScenarioSpec("roundabout", 1000, ego_depart_time=30.0),
    "carla": PaperScenarioSpec(
        "carla",
        # carla_env.py evaluates ``count > 300`` before incrementing count,
        # so a zero-based episode can execute 302 calls to step().
        302,
        ego_depart_time=0.0,
        map_paths_per_actor=3,
        map_feature_dim=2,
        include_pedestrians=True,
        released_assets=False,
        source_observation_contract="carla",
        # CARLA builds waypoints every 0.5 m and then keeps every fifth point.
        waypoint_spacing=2.5,
        # netconvert translates negative Town-10 source coordinates into its
        # non-negative Cartesian network; TraCI observations must be shifted
        # back before they are paired with the released wp.npy/wp2.npy arrays.
        coordinate_offset=(95.0, 64.83),
    ),
}


def get_paper_scenario_spec(name: str) -> PaperScenarioSpec:
    try:
        specification = PAPER_SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(
            f"No released SMARTS asset mapping for {name!r}; "
            f"choose one of {tuple(PAPER_SCENARIOS)}"
        ) from exc
    for path in (specification.network_path, specification.ego_route_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    specification.traffic_paths
    return specification


def traffic_seed(path: Path) -> int:
    return _traffic_seed(path)
