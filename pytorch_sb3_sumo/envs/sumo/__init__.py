"""SUMO environments with the Scene-Rep observation and action contracts."""

from gymnasium.envs.registration import register, registry

from .scenario_registry import SCENARIOS, available_scenarios, get_scenario_spec


def _register() -> None:
    for scenario in SCENARIOS:
        env_id = f"SceneRepSUMO-{scenario}-v0"
        if env_id not in registry:
            register(
                id=env_id,
                entry_point="envs.sumo.sumo_env:SumoSceneEnv",
                kwargs={"scenario": scenario},
            )


_register()


def __getattr__(name: str):
    # Keep SUMO/TraCI imports lazy so package discovery works without SUMO.
    if name == "SumoSceneEnv":
        from .sumo_env import SumoSceneEnv

        return SumoSceneEnv
    if name == "LegacySumoAdapter":
        from .legacy import LegacySumoAdapter

        return LegacySumoAdapter
    raise AttributeError(name)


__all__ = [
    "SCENARIOS",
    "SumoSceneEnv",
    "LegacySumoAdapter",
    "available_scenarios",
    "get_scenario_spec",
]
