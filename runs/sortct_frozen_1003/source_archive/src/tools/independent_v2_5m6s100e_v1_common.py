"""Frozen contract helpers for the independent-v2 5-method x 6-scene study."""

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
    / "independent_v2_five_methods_six_scenarios_100ep_v1"
    / "protocol.json"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results_iv2_5m6s100e_v1"

METHOD_ADAPTERS: dict[str, tuple[str, str]] = {
    "mst_slt": ("base", "scene_rep"),
    "temporal_graph": ("base", "temporal_graph"),
    "full_balanced": ("base", "topo_scene_balanced"),
    "v4_8": ("v4_8", "topo_v4_8_tie_only_replicated_calibration"),
    "v4_13": (
        "v4_13",
        "topo_v4_13_gradient_isolated_tempered_joint_support_prcr_full",
    ),
}
SCENARIOS = (
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
)
PARENT_METHODS = ("mst_slt", "temporal_graph", "full_balanced", "v4_8")
PARENT_SCENARIOS = ("cross", "roundabout", "carla")
REMOVED_METHODS = ("initial_full", "v4_1")
RUN_ID_PREFIX = "i5m6s100_v1"
RECEIPT_NAME = "i5m6s100_job_receipt.json"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{1,31}$")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_bound_file(
    protocol: dict[str, Any], path_key: str, hash_key: str, section: str
) -> Path:
    payload = protocol[section]
    path = (PROJECT_ROOT / str(payload[path_key])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    expected = str(payload[hash_key])
    if actual != expected:
        raise ValueError(
            f"{section}.{path_key} SHA-256 drifted: {actual} != {expected}"
        )
    return path


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != (
        "independent-v2-five-method-six-scenario-comparison/v1"
    ):
        raise ValueError("unexpected independent-v2 extension protocol schema")
    if protocol.get("no_fabrication") is not True:
        raise ValueError("protocol must enforce no_fabrication")
    if protocol.get("formal_test_accessed") is not False:
        raise ValueError("formal/test access must remain false")
    prefix = str(protocol.get("run_id_prefix", ""))
    if prefix != RUN_ID_PREFIX or _RUN_ID_PATTERN.fullmatch(prefix) is None:
        raise ValueError("run_id_prefix drifted")

    matrix = protocol.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("matrix is missing")
    methods = tuple(matrix.get("method_order", ()))
    scenarios = tuple(matrix.get("scenario_order", ()))
    if methods != tuple(METHOD_ADAPTERS):
        raise ValueError(f"method order drifted: {methods!r}")
    if any(method in methods for method in REMOVED_METHODS):
        raise ValueError("Initial Full and v4.1 must not enter this matrix")
    if scenarios != SCENARIOS:
        raise ValueError(f"scenario order drifted: {scenarios!r}")
    if int(matrix.get("method_scenario_cells", -1)) != 30:
        raise ValueError("matrix must contain exactly 30 method-scenario cells")
    if int(matrix.get("reused_parent_cells", -1)) != 12:
        raise ValueError("matrix must adopt exactly 12 parent cells")
    if int(matrix.get("fresh_cells", -1)) != 18:
        raise ValueError("matrix must execute exactly 18 fresh cells")
    if int(matrix.get("training_seed", -1)) != 0:
        raise ValueError("training seed must remain 0")
    episodes = int(matrix.get("evaluation_episodes_per_cell", -1))
    if episodes != 100 or int(matrix.get("total_evaluation_episodes", -1)) != 3000:
        raise ValueError("evaluation contract must be 30 x 100 = 3000 episodes")

    method_payload = protocol.get("methods")
    if not isinstance(method_payload, dict) or tuple(method_payload) != methods:
        raise ValueError("methods must exactly follow matrix.method_order")
    for method, (adapter, algorithm) in METHOD_ADAPTERS.items():
        if method_payload[method].get("trainer_adapter") != adapter:
            raise ValueError(f"trainer adapter drifted for {method}")
        if method_payload[method].get("cli_algorithm") != algorithm:
            raise ValueError(f"CLI algorithm drifted for {method}")

    scenario_payload = protocol.get("scenarios")
    if not isinstance(scenario_payload, dict) or tuple(scenario_payload) != scenarios:
        raise ValueError("scenarios must exactly follow matrix.scenario_order")
    for scenario in scenarios:
        setting = scenario_payload[scenario]
        vehicle_scale = float(setting["vehicle_scale"])
        pedestrian_scale = float(setting["pedestrian_scale"])
        jitter = tuple(float(value) for value in setting["clone_depart_jitter_seconds"])
        if not 1.0 < vehicle_scale <= 2.0:
            raise ValueError(f"{scenario} is not configured as increased traffic")
        if not 1.0 <= pedestrian_scale <= 2.0:
            raise ValueError(f"invalid pedestrian scale for {scenario}")
        if len(jitter) != 2 or not 0.0 <= jitter[0] <= jitter[1]:
            raise ValueError(f"invalid clone jitter for {scenario}")
    expected_new_scales = {
        "left_turn": 1.30,
        "roundabout_easy": 1.25,
        "roundabout_medium": 1.20,
    }
    for scenario, expected in expected_new_scales.items():
        if float(scenario_payload[scenario]["vehicle_scale"]) != expected:
            raise ValueError(f"calibrated density drifted for {scenario}")

    parent = protocol.get("parent_v2_reuse")
    if not isinstance(parent, dict) or parent.get("enabled") is not True:
        raise ValueError("parent-v2 adoption contract is missing")
    if parent.get("training_and_evaluation_rerun_forbidden") is not True:
        raise ValueError("parent-v2 reruns must be forbidden")
    if tuple(parent.get("methods", ())) != PARENT_METHODS:
        raise ValueError("parent method set drifted")
    if tuple(parent.get("scenarios", ())) != PARENT_SCENARIOS:
        raise ValueError("parent scenario set drifted")

    profiles = protocol.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"smoke", "comparison"}:
        raise ValueError("smoke/comparison profiles are required")
    for name, profile in profiles.items():
        if profile.get("seeds") != [0]:
            raise ValueError(f"{name} must contain training seed 0 only")
        if int(profile["raw_training_steps"]) <= 0:
            raise ValueError(f"{name} raw training steps must be positive")
        if int(profile["evaluation_episodes"]) <= 0:
            raise ValueError(f"{name} evaluation episodes must be positive")
    comparison = profiles["comparison"]
    if int(comparison["raw_training_steps"]) != 50_000:
        raise ValueError("comparison must remain 50k raw steps")
    if int(comparison["evaluation_episodes"]) != 100:
        raise ValueError("comparison must remain 100 evaluation episodes")
    if int(comparison["evaluation_seed_start"]) != 10_000:
        raise ValueError("comparison evaluation seeds must start at 10000")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    protocol = json.loads(source.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    protocol["_path"] = str(source)
    protocol["_sha256"] = sha256(source)
    return protocol


def method_setting(protocol: dict[str, Any], method: str) -> dict[str, Any]:
    if method not in METHOD_ADAPTERS:
        raise ValueError(f"unsupported method {method!r}")
    return dict(protocol["methods"][method])


def density_setting(protocol: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario {scenario!r}")
    return dict(protocol["scenarios"][scenario])


def profile_setting(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    try:
        return dict(protocol["profiles"][profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}") from exc


def is_adopted_cell(method: str, scenario: str) -> bool:
    return method in PARENT_METHODS and scenario in PARENT_SCENARIOS


def run_id(profile: str, method: str, scenario: str, seed: int) -> str:
    return f"{RUN_ID_PREFIX}__{profile}__{method}__{scenario}__seed{int(seed)}"


@dataclass(frozen=True)
class MatrixCell:
    profile: str
    method: str
    scenario: str
    seed: int
    run_id: str
    execution: str


def build_cells(protocol: dict[str, Any], profile: str) -> list[MatrixCell]:
    settings = profile_setting(protocol, profile)
    cells: list[MatrixCell] = []
    for method in protocol["matrix"]["method_order"]:
        for scenario in protocol["matrix"]["scenario_order"]:
            for seed in settings["seeds"]:
                adopted = profile == "comparison" and is_adopted_cell(method, scenario)
                cells.append(
                    MatrixCell(
                        profile=profile,
                        method=method,
                        scenario=scenario,
                        seed=int(seed),
                        run_id=run_id(profile, method, scenario, int(seed)),
                        execution="adopt_parent_v2" if adopted else "fresh",
                    )
                )
    return cells


def build_fresh_cells(protocol: dict[str, Any], profile: str) -> list[MatrixCell]:
    return [cell for cell in build_cells(protocol, profile) if cell.execution == "fresh"]


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "DEFAULT_RESULT_ROOT",
    "METHOD_ADAPTERS",
    "MatrixCell",
    "PARENT_METHODS",
    "PARENT_SCENARIOS",
    "PROJECT_ROOT",
    "RECEIPT_NAME",
    "REMOVED_METHODS",
    "RUN_ID_PREFIX",
    "SCENARIOS",
    "assert_bound_file",
    "build_cells",
    "build_fresh_cells",
    "density_setting",
    "is_adopted_cell",
    "load_protocol",
    "method_setting",
    "profile_setting",
    "run_id",
    "sha256",
    "validate_protocol",
]
