"""Independent Gymnasium environment adapter for the Scene-Event methods.

This module wraps the existing SMARTS/SUMO task without changing legacy files.
The old observation builder is replaced by a zero placeholder so it cannot
query hidden social routes; policy observations come only from the new raw-tick
collector and public lane graph.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

# Entry scripts run from fast-developer, while the existing SUMO package lives
# one directory above it. Make the new SceneEvent package self-bootstrapping;
# no caller-specific PYTHONPATH is required.
_PR_ROOT = Path(__file__).resolve().parents[2]
if str(_PR_ROOT) not in sys.path:
    sys.path.insert(0, str(_PR_ROOT))

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import IndependentV2FiveBySixEnvV1
from .collector import SceneEventCollector
from .map_cache import PublicMapCache
from .schema import make_observation_space


class SceneEventRawEnv(IndependentV2FiveBySixEnvV1):
    """Fresh environment subclass with raw-history capture and route-safe placeholder."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._scene_event_config: Any | None = None
        self._scene_event_map_cache: PublicMapCache | None = None
        self._scene_event_collector: SceneEventCollector | None = None
        self._scene_event_run_root: Path | None = None
        self._scene_event_evaluation = False
        super().__init__(*args, **kwargs)

    def configure_scene_event(self, config: Any, run_root: str | Path, *, evaluation: bool) -> None:
        config.validate()
        if getattr(self.specification, "source_observation_contract", None) != "smarts":
            raise ValueError("SceneEventEnv currently accepts only the SMARTS observation contract")
        if int(self.neighbors) + 1 != int(config.max_actors):
            raise ValueError("base environment actor capacity differs from ExperimentConfig")
        if int(self.action_repeat) != int(config.action_repeat):
            raise ValueError("base environment action_repeat differs from ExperimentConfig")
        deadline_raw_steps = int(round(float(config.deadline_seconds) / float(config.raw_dt)))
        if not math.isclose(deadline_raw_steps * float(config.raw_dt), float(config.deadline_seconds), abs_tol=1e-8):
            raise ValueError("deadline_seconds must be an integral number of raw ticks")
        if int(self.max_episode_steps) != deadline_raw_steps:
            raise ValueError(
                f"scenario limit ({self.max_episode_steps} raw ticks) and task deadline "
                f"({deadline_raw_steps}) differ"
            )
        self._scene_event_config = config
        self._scene_event_run_root = Path(run_root).resolve()
        self._scene_event_evaluation = bool(evaluation)
        self._scene_event_map_cache = PublicMapCache.from_spec(self._paper_specification)
        self._scene_event_collector = SceneEventCollector(config, self._scene_event_map_cache)
        self._scene_event_collector.set_ego_id(str(self.specification.ego_id))

    @property
    def public_map(self) -> dict[str, np.ndarray]:
        self._require_scene_event()
        assert self._scene_event_map_cache is not None
        return self._scene_event_map_cache.public_map

    @property
    def last_audit(self) -> dict[str, Any]:
        self._require_scene_event()
        assert self._scene_event_collector is not None
        return self._scene_event_collector.summary()

    @property
    def last_collection(self) -> dict[str, Any] | None:
        self._require_scene_event()
        assert self._scene_event_collector is not None
        return self._scene_event_collector.last_collection

    def _require_scene_event(self) -> None:
        if self._scene_event_collector is None or self._scene_event_map_cache is None:
            raise RuntimeError("Call configure_scene_event before reset/use")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._require_scene_event()
        assert self._scene_event_collector is not None
        self._scene_event_collector.begin_episode(seed)
        return super().reset(seed=seed, options=options)

    def _record_histories(self) -> None:
        super()._record_histories()
        if self._scene_event_collector is not None:
            self._scene_event_collector.capture_raw_tick(self)

    def _make_observation(self) -> dict[str, np.ndarray]:
        """Return a shape-safe placeholder; never invoke legacy `_actor_paths`.

        The deadline wrapper discards this base observation and returns the
        typed SceneObservation. Keeping a valid base-shaped value lets the
        existing simulator's reset/step lifecycle run unchanged.
        """
        if self.state_lstm_only or self.include_state_lstm:
            raise ValueError("SceneEventRawEnv requires trajectory/map base lifecycle only")
        actor_count = self.neighbors + 1
        path_count = actor_count * int(self.specification.map_paths_per_actor)
        return {
            "trajectory": np.zeros((actor_count, self.history_steps, self.state_dim), dtype=np.float32),
            "map": np.zeros((path_count, self.path_length, int(self.specification.map_feature_dim)), dtype=np.float32),
        }

    def scene_metadata(self) -> dict[str, Any]:
        self._require_scene_event()
        assert self._scene_event_map_cache is not None
        assert self._scene_event_config is not None
        assert self._paper_specification is not None
        selected_templates = self._partitioned_traffic_paths(self._paper_specification)
        selected_traffic = getattr(self, "_selected_traffic_path", None)
        overlay = getattr(self, "_selected_high_density_overlay", None)
        assert self._scene_event_collector is not None
        topology_status_contract = self._scene_event_collector.summary()[
            "candidate_pair_topology_status_contract"
        ]
        return {
            "protocol": self._scene_event_config.protocol,
            "config_sha256": self._scene_event_config.digest(),
            "method": self._scene_event_config.method,
            "scenario": self.scenario,
            "phase": "eval" if self._scene_event_evaluation else "train",
            "raw_dt_s": float(self._scene_event_config.raw_dt),
            "action_repeat": int(self.action_repeat),
            "deadline_seconds": float(self._scene_event_config.deadline_seconds),
            "deadline_raw_steps": int(round(self._scene_event_config.deadline_seconds / self._scene_event_config.raw_dt)),
            "history_contract": {"max_actors": self._scene_event_config.max_actors,
                                 "history_samples": self._scene_event_config.history_samples,
                                 "radius_m": self._scene_event_config.observation_radius,
                                 "world_coordinates": True},
            "coordinate_contract": "SUMO world Cartesian; center state; velocity=(speed*sin(theta),speed*cos(theta)); theta clockwise from north",
            "policy_social_future_routes_read": False,
            "process_wide_route_api_calls": "not_instrumented; simulator-side route management is outside the policy observation collector",
            "policy_builder": "SceneEventCollector public successor graph; legacy _make_observation is a zero placeholder",
            "candidate_contract": {"max_candidates": 4, "max_path_lanes": 16,
                                   "lane_changes_as_paths": False,
                                   "unmodeled_or_truncated_coverage": "candidate_unknown_count=-1"},
            "candidate_pair_topology_status_contract": topology_status_contract,
            "prediction_horizon_s": float(self._scene_event_config.prediction_seconds),
            "public_map": {
                "network_path": self._scene_event_map_cache.network_path,
                "network_sha256": self._scene_event_map_cache.network_sha256,
                "ego_route_path": self._scene_event_map_cache.ego_route_path,
                "fingerprint": self._scene_event_map_cache.fingerprint,
                **self._scene_event_map_cache.build_metadata,
            },
            "traffic_templates": [
                {"path": str(path.resolve()), "sha256": _sha256(Path(path))}
                for path in selected_templates
            ],
            "selected_traffic_path": str(Path(selected_traffic).resolve()) if selected_traffic else None,
            "selected_traffic_sha256": _sha256(Path(selected_traffic)) if selected_traffic and Path(selected_traffic).is_file() else None,
            "overlay_path": str(Path(overlay).resolve()) if overlay else None,
            "overlay_manifest": getattr(self, "_selected_high_density_manifest", None),
        }


class SceneEventDeadlineWrapper(gym.Wrapper):
    """Expose SceneObservation and make the 60-second task deadline terminal."""

    def __init__(self, env: SceneEventRawEnv, config: Any):
        super().__init__(env)
        self.config = config
        self._raw_limit = int(round(float(config.deadline_seconds) / float(config.raw_dt)))
        self._task_raw_steps = 0
        self._configured_repeat = int(config.action_repeat)
        self.observation_space = make_observation_space(
            int(config.max_actors), int(config.history_samples), 4, 16,
            len(env.public_map["zone_valid"]), float(config.deadline_seconds),
        )
        self.action_space = env.action_space

    @property
    def public_map(self) -> dict[str, np.ndarray]:
        return self.env.public_map

    @property
    def last_audit(self) -> dict[str, Any]:
        return self.env.last_audit

    @property
    def last_collection(self) -> dict[str, Any] | None:
        return self.env.last_collection

    def scene_metadata(self) -> dict[str, Any]:
        return self.env.scene_metadata()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _placeholder, info = self.env.reset(seed=seed, options=options)
        self._task_raw_steps = 0
        obs = self.env._scene_event_collector.observation(float(self.config.deadline_seconds))
        augmented = self._augment_step_info(info, obs)
        augmented["terminated"] = False
        augmented["truncated"] = False
        return obs, augmented

    def step(self, action: np.ndarray):
        remaining = self._raw_limit - self._task_raw_steps
        if remaining <= 0:
            raise RuntimeError("step() called after the finite task deadline")
        original_repeat = int(self.env.action_repeat)
        self.env.action_repeat = min(self._configured_repeat, remaining)
        try:
            _placeholder, reward, terminated, truncated, info = self.env.step(action)
        finally:
            self.env.action_repeat = original_repeat
        raw = int(info.get("raw_steps_executed", 0))
        self._task_raw_steps += raw
        max_time = bool(info.get("max_time"))
        if max_time:
            # SMARTS legacy episodes label max_time as truncated. This independent
            # finite-horizon task treats its known deadline as a true terminal.
            terminated, truncated = True, False
        remaining_seconds = max(0.0, float(self.config.deadline_seconds) - self._task_raw_steps * float(self.config.raw_dt))
        obs = self.env._scene_event_collector.observation(remaining_seconds)
        augmented = self._augment_step_info(info, obs)
        augmented["terminated"] = bool(terminated)
        augmented["truncated"] = bool(truncated)
        if max_time:
            augmented["legacy_source_terminal_for_bootstrap"] = info.get("source_terminal_for_bootstrap")
            augmented["source_terminal_for_bootstrap"] = True
            augmented["TimeLimit.truncated"] = False
        return obs, float(reward), bool(terminated), bool(truncated), augmented

    def _augment_step_info(self, info: dict[str, Any], obs: dict[str, np.ndarray]) -> dict[str, Any]:
        result = dict(info)
        result["raw_simulation_steps"] = int(getattr(self.env, "_raw_steps", self._task_raw_steps))
        result["task_raw_steps"] = int(self._task_raw_steps)
        result["decision_steps"] = int(getattr(self.env, "_decision_steps", 0))
        result["remaining_time_s"] = float(obs["remaining_time_s"][0])
        result["deadline_terminal"] = bool(result.get("max_time"))
        collection = self.env.last_collection or {}
        keys = collection.get("actor_keys", ())
        lane_ids = self.env._scene_event_map_cache.lane_ids
        lane_ptr = int(obs["actor_lane_ptr"][0]) if obs["actor_valid"][0] else -1
        lane_name = lane_ids[lane_ptr] if 0 <= lane_ptr < len(lane_ids) else None
        ego_speed = float(np.hypot(*obs["actor_history"][0, -1, 4:6])) if obs["actor_valid"][0] else None
        base_partner = getattr(self.env, "_last_geometric_collision_partner", None)
        result["scene_event_behavior"] = {
            "target_speed_mps": result.get("target_speed", result.get("requested_target_speed_mps")),
            "actual_speed_mps": ego_speed,
            "lane_id": lane_name,
            "lane_ptr": lane_ptr,
            "lane_change_applied": result.get("lane_change_applied"),
            "lane_command": result.get("lane_command"),
            "collision": bool(result.get("collision", False)),
            "raw_sumo_collision": bool(result.get("raw_sumo_collision", False)),
            "geometric_collision": bool(result.get("geometric_collision", False)),
            "geometric_obb_partner": dict(base_partner) if base_partner else None,
            "collision_partner_status": "geometric_obb_partner_only" if base_partner else
                ("sumo_partner_not_exposed_by_base_info" if result.get("raw_sumo_collision") else "none_recorded"),
            "actor_key_sidecar_count": sum(key is not None for key in keys),
        }
        return result


class SceneEventRewardAuditWrapper(gym.Wrapper):
    """Record the exact final reward returned after the v2 shaping wrapper."""

    def _scene_event_base(self) -> SceneEventRawEnv:
        return _find_scene_event_raw_env(self.env)

    @property
    def public_map(self) -> dict[str, np.ndarray]:
        return self._scene_event_base().public_map

    @property
    def last_audit(self) -> dict[str, Any]:
        return self._scene_event_base().last_audit

    @property
    def last_collection(self) -> dict[str, Any] | None:
        return self._scene_event_base().last_collection

    def scene_metadata(self) -> dict[str, Any]:
        return self._scene_event_base().scene_metadata()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["environment_step_reward_v2"] = float(reward)
        return obs, float(reward), terminated, truncated, info


def _find_scene_event_raw_env(env: Any) -> SceneEventRawEnv:
    """Resolve the raw SceneEvent env through wrappers without __getattr__.

    Gym wrappers do not consistently forward arbitrary project attributes.
    Follow explicit wrapper links (and use `unwrapped` only as a fallback) so
    diagnostics work when reward shaping is nested around the deadline env.
    """
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, SceneEventRawEnv) or (
            getattr(current, "_scene_event_collector", None) is not None
            and getattr(current, "_scene_event_map_cache", None) is not None
        ):
            return current
        child = getattr(current, "env", None)
        if child is None:
            unwrapped = getattr(current, "unwrapped", current)
            child = unwrapped if unwrapped is not current else None
        current = child
    raise RuntimeError("SceneEvent raw environment is not reachable through the wrapper chain")


def make_scene_env(config: Any, run_root: str | Path, evaluation: bool = False) -> gym.Env:
    """Construct the fixed-task SceneEvent environment; does not start SUMO."""
    config.validate()
    scenario = str(config.scenario)
    suffix = "_depart4p0"
    if scenario.endswith(suffix):
        scenario = scenario[:-len(suffix)]
    if scenario != "intersection_sorted":
        raise ValueError(f"unsupported SceneEvent scenario {config.scenario!r}")
    run_root = Path(run_root).resolve()
    overlay_root = run_root / "traffic_overlay"
    args = SimpleNamespace(
        scenario=scenario,
        history_steps=int(config.history_samples),
        neighbors=int(config.max_actors) - 1,
        path_length=16,
        action_repeat=int(config.action_repeat),
        discount=float(config.gamma),
        ego_control_profile="direct",
        episode_limit_profile="source",
        gui=False,
        evaluation_split="validation",
    )
    density = {
        # intersection_sorted_depart4p0 templates already define the fixed
        # traffic density; no extra high-density clones are introduced here.
        "vehicle_scale": 1.0,
        "pedestrian_scale": 1.0,
        "clone_depart_jitter_seconds": (0.0, 0.0),
    }
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    factory = _make_environment_factory(
        adapter="base", density=density, overlay_root=overlay_root,
        baseline_environment_class=SceneEventRawEnv,
    )
    raw_env = factory(args, evaluation=bool(evaluation))
    raw_env.configure_scene_event(config, run_root, evaluation=bool(evaluation))
    deadline_env = SceneEventDeadlineWrapper(raw_env, config)
    from reward_shaping_v2 import GeneralizedRewardShapingWrapper
    shaped_env = GeneralizedRewardShapingWrapper(deadline_env)
    return SceneEventRewardAuditWrapper(shaped_env)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
