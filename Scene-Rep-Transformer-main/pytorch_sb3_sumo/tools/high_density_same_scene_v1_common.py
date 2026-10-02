"""Shared contract helpers for the same-scene high-density comparison."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "high_density_same_scene_v1"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_high_density_same_scene_v1"

METHOD_ADAPTERS: dict[str, tuple[str, str]] = {
    "mst_slt": ("base", "scene_rep"),
    "temporal_graph": ("base", "temporal_graph"),
    "initial_full": ("base", "topo_scene"),
    "full_balanced": ("base", "topo_scene_balanced"),
    "v4_1": ("v4_1", "topo_v4_da_hybrid"),
    "v4_8": ("v4_8", "topo_v4_8_tie_only_replicated_calibration"),
}
TARGET_SCENARIOS = ("carla", "cross", "roundabout")
DEFAULT_RUN_ID_PREFIX = "hdv1"
_RUN_ID_PREFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{1,31}$")


def sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    protocol["_path"] = str(protocol_path)
    protocol["_sha256"] = sha256(protocol_path)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != "same-scene-high-density-comparison/v1":
        raise ValueError("unexpected high-density protocol schema")
    if protocol.get("no_fabrication") is not True:
        raise ValueError("high-density protocol must enforce no_fabrication")
    run_id_prefix = str(protocol.get("run_id_prefix", DEFAULT_RUN_ID_PREFIX))
    if _RUN_ID_PREFIX_PATTERN.fullmatch(run_id_prefix) is None:
        raise ValueError(
            "run_id_prefix must contain 2-32 lowercase letters, digits, or underscores"
        )
    matrix = protocol.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("protocol matrix is missing")
    method_order = tuple(matrix.get("method_order", ()))
    if method_order != tuple(METHOD_ADAPTERS):
        raise ValueError(
            f"method order drifted: {method_order!r} != {tuple(METHOD_ADAPTERS)!r}"
        )
    scenario_order = tuple(matrix.get("scenario_order", ()))
    if scenario_order != TARGET_SCENARIOS:
        raise ValueError(
            f"scenario order drifted: {scenario_order!r} != {TARGET_SCENARIOS!r}"
        )
    methods = protocol.get("methods")
    if not isinstance(methods, dict) or tuple(methods) != method_order:
        raise ValueError("protocol methods must exactly follow method_order")
    for method, (trainer, algorithm) in METHOD_ADAPTERS.items():
        payload = methods[method]
        if payload.get("trainer_adapter") != trainer:
            raise ValueError(f"trainer adapter drifted for {method}")
        if payload.get("cli_algorithm") != algorithm:
            raise ValueError(f"CLI algorithm drifted for {method}")

    density = protocol.get("density")
    if not isinstance(density, dict) or tuple(density) != TARGET_SCENARIOS:
        raise ValueError("density settings must cover exactly the target scenarios")
    for scenario in TARGET_SCENARIOS:
        setting = density[scenario]
        vehicle_scale = float(setting["vehicle_scale"])
        pedestrian_scale = float(setting["pedestrian_scale"])
        jitter = tuple(setting["clone_depart_jitter_seconds"])
        if not 1.0 < vehicle_scale <= 2.0:
            raise ValueError(f"invalid vehicle scale for {scenario}")
        if not 1.0 <= pedestrian_scale <= 2.0:
            raise ValueError(f"invalid pedestrian scale for {scenario}")
        if len(jitter) != 2 or not 0.0 <= float(jitter[0]) <= float(jitter[1]):
            raise ValueError(f"invalid clone jitter for {scenario}")

    profiles = protocol.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("protocol profiles are missing")
    for name, profile in profiles.items():
        if int(profile["raw_training_steps"]) <= 0:
            raise ValueError(f"profile {name} raw_training_steps must be positive")
        if int(profile["evaluation_episodes"]) <= 0:
            raise ValueError(f"profile {name} evaluation_episodes must be positive")
        seeds = profile.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"profile {name} seeds are missing")
        if len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError(f"profile {name} contains duplicate seeds")


def density_setting(protocol: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario not in TARGET_SCENARIOS:
        raise ValueError(f"unsupported target scenario {scenario!r}")
    return dict(protocol["density"][scenario])


def method_setting(protocol: dict[str, Any], method: str) -> dict[str, Any]:
    if method not in METHOD_ADAPTERS:
        raise ValueError(f"unsupported comparison method {method!r}")
    return dict(protocol["methods"][method])


def profile_setting(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    try:
        return dict(protocol["profiles"][profile])
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {tuple(protocol['profiles'])}"
        ) from exc


def protocol_run_id_prefix(protocol: dict[str, Any]) -> str:
    return str(protocol.get("run_id_prefix", DEFAULT_RUN_ID_PREFIX))


def job_id(
    profile: str,
    method: str,
    scenario: str,
    seed: int,
    *,
    run_id_prefix: str = DEFAULT_RUN_ID_PREFIX,
) -> str:
    return (
        f"{run_id_prefix}__{profile}__{method}__{scenario}__seed{int(seed)}"
    )


@dataclass(frozen=True)
class ComparisonJob:
    profile: str
    method: str
    scenario: str
    seed: int
    run_id: str


def build_jobs(protocol: dict[str, Any], profile: str) -> list[ComparisonJob]:
    settings = profile_setting(protocol, profile)
    run_id_prefix = protocol_run_id_prefix(protocol)
    return [
        ComparisonJob(
            profile=profile,
            method=method,
            scenario=scenario,
            seed=int(seed),
            run_id=job_id(
                profile,
                method,
                scenario,
                int(seed),
                run_id_prefix=run_id_prefix,
            ),
        )
        for method in protocol["matrix"]["method_order"]
        for scenario in protocol["matrix"]["scenario_order"]
        for seed in settings["seeds"]
    ]


__all__ = [
    "ComparisonJob",
    "DEFAULT_PROTOCOL_PATH",
    "DEFAULT_RESULT_ROOT",
    "METHOD_ADAPTERS",
    "DEFAULT_RUN_ID_PREFIX",
    "PROJECT_ROOT",
    "TARGET_SCENARIOS",
    "build_jobs",
    "density_setting",
    "job_id",
    "load_protocol",
    "method_setting",
    "profile_setting",
    "protocol_run_id_prefix",
    "sha256",
    "validate_protocol",
]
