"""Static metadata for the SUMO equivalents of the released scenarios.

The original project hard-codes the episode limits in ``tools/test.py`` and
uses a different waypoint tensor for CARLA.  The registry keeps those public
contracts explicit so the simulator adapter does not need scenario-specific
conditionals scattered throughout the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .random_intersection import RANDOM_INTERSECTION_SCENARIOS


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
    # False for paper-only scenarios whose base registry entry exists solely so
    # PaperSumoSceneEnv.__init__ (via super().__init__) can size the observation
    # space and validate file presence before overriding ``self.specification``
    # with the paper spec.  The legacy base SumoSceneEnv cannot run these: their
    # ego departs after the base env's fixed 50-step insertion window and their
    # real assets live under original_scenarios_v1/, not scenarios/.
    runnable_in_base_env: bool = True

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
    # south_in -> west_out while social traffic goes straight + right.  Paper-
    # only: the base entry mirrors the paper ego depart time (t=15) and is not
    # runnable by the legacy base SumoSceneEnv's 50-step insertion window.
    "cross_left": SumoScenarioSpec(
        "cross_left", max_episode_steps=600, runnable_in_base_env=False
    ),
    # cross_left variant: uncontrolled (unregulated) 4-way, 70 m lanes, social
    # left turns on every approach.  Paper-only (assets live under
    # original_scenarios_v1/).
    "cross_left_unreg": SumoScenarioSpec(
        "cross_left_unreg",
        max_episode_steps=600,
        network_name="cross_left_unreg",
        runnable_in_base_env=False,
    ),
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
    # New SUMO scenarios (this project).  The paper assets live under
    # original_scenarios_v1/ and are loaded by PaperSumoSceneEnv; these entries
    # exist only so the base SumoSceneEnv constructor can size the observation
    # space and validate file presence before the paper spec overrides them.
    "merge": SumoScenarioSpec(
        "merge", max_episode_steps=600, network_name="merge",
        runnable_in_base_env=False,
    ),
    "intersection": SumoScenarioSpec(
        "intersection", max_episode_steps=600, runnable_in_base_env=False
    ),
    # intersection 副本（depart 排序版）：base 入口仅为满足 SumoSceneEnv 构造
    # 时的观测空间/文件存在性检查；真实资产在 original_scenarios_v1/ 下，由
    # PaperSumoSceneEnv 加载。network_name 复用 "intersection"。
    "intersection_sorted": SumoScenarioSpec(
        "intersection_sorted", max_episode_steps=600, runnable_in_base_env=False
    ),
}


SCENARIOS.update({
    name: SumoScenarioSpec(name, max_episode_steps=600, runnable_in_base_env=False)
    for name in RANDOM_INTERSECTION_SCENARIOS
})


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


def base_runnable_scenarios() -> tuple[str, ...]:
    """Scenarios the legacy base ``SumoSceneEnv`` can actually run a full episode.

    Paper-only scenarios (``cross_left`` / ``merge`` / ``intersection``) are
    registered so ``PaperSumoSceneEnv`` can construct, but their real assets
    live under ``original_scenarios_v1/`` and their ego departs after the base
    env's insertion window, so they are excluded here.
    """
    return tuple(
        name for name, spec in SCENARIOS.items() if spec.runnable_in_base_env
    )
