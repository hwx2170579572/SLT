"""Prepare/launch the sorted-scene route-aware Topo vs ST-RT-3slot pair.

Preview is the default and creates no files.  ``--start`` creates a fresh run
root, archives the exact source and sorted traffic inputs, then supervises the
existing D1 runner.  This file is intentionally separate from the DARRL queue
launcher because this experiment uses the fixed sorted 30-template pool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCRIPT_PATH = Path(__file__).resolve()
PROJECT = SCRIPT_PATH.parents[1]
WORKSPACE = PROJECT.parent.parent
DEFAULT_RUN_ROOT = WORKSPACE / "runs" / "sort2_1001"

SCENARIO = "intersection_sorted"
DEPART_SCALE = 4.0
METHODS = (
    "sac_mlp_d1_st_rt_topo_routeaware_v1",
    "sac_mlp_d1_st_rt_3slot",
)
RAW_STEPS = 100_000
CHECKPOINT_FREQUENCY_RAW = 10_000
EVAL_EPISODES = 100
EVAL_SEED_START = 10_000
EVAL_SPLIT_LABEL = "validation"
EVALUATION_RETURN_PROTOCOL = "environment_step_reward_v2"
RAW_RETURN_PROTOCOL = "info_undiscounted_reward_or_step_reward_fallback_v1"
REWARD_COMPONENT_PROTOCOL = "yield_v2_cumulative_reward_branches_v1"
REWARD_COMPONENT_KEYS = (
    "reward_success",
    "reward_collision",
    "reward_off_route",
    "reward_timeout",
    "reward_step_cost",
    "reward_progress",
)
SAFE_PATH_LIMIT = 240  # keep margin below legacy Windows MAX_PATH=260

SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
    "node_modules", "runs", "results", "outputs", "output", "logs",
    "checkpoints", "tensorboard", "tb", "overlays", "live",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write small launcher metadata atomically, with bounded Windows retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    delays = (0.05, 0.10, 0.20)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def load_runtime_modules():
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    import train_intersection_yield_v2 as base
    import train_intersection_yield_v2_d1 as d1
    from algos.sb3_torch import evaluation
    from envs.sumo.random_intersection import is_random_intersection_scenario

    return base, d1, evaluation, is_random_intersection_scenario


def _assert_method_config(base, d1, method: str) -> dict:
    if method not in d1.DISPATCH or method not in d1.D1_CONFIG:
        raise ValueError(f"D1 method is not registered: {method}")
    if d1.PARENT.get(method) != "sac_mlp":
        raise ValueError(f"{method} must inherit the pure SAC+MLP base")
    if d1._env_adapter_d1(method) != "base":
        raise ValueError(f"{method} must use the base SAC/MLP environment contract")

    cfg = d1.D1_CONFIG[method]
    if method == "sac_mlp_d1_st_rt_topo_routeaware_v1":
        expected = {
            "use_route": True,
            "use_topology": True,
            "use_slots": False,
            "use_route_reachability": True,
            "use_graph_slt": False,
            "use_sbs": False,
            "representation_coef": 0.0,
            "slot_balance_coef": 0.0,
        }
    elif method == "sac_mlp_d1_st_rt_3slot":
        expected = {
            "use_route": True,
            "use_topology": False,
            "use_slots": True,
            "use_incremental_slots": True,
            "use_graph_slt": False,
            "use_sbs": False,
            "representation_coef": 0.0,
            "slot_balance_coef": 0.0,
        }
    else:
        raise ValueError(f"No sorted-pair configuration assertion for {method}")

    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unexpected {method}.{key}: {cfg.get(key)!r}; expected {value!r}")
    return dict(cfg)


def validate_runtime_plan(run_root: Path) -> dict:
    base, d1, evaluation, is_random_profile = load_runtime_modules()

    # Run the same scenario routing/configuration code that the D1 child will
    # use; sorted must not inherit any randomized DARRL profile or flow settings.
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT_LABEL)
    if d1.NEW_SCENARIO != SCENARIO or is_random_profile(d1.NEW_SCENARIO):
        raise ValueError("sorted launch unexpectedly selected a randomized traffic profile")
    if d1.NEW_SOURCE_TRAFFIC != (
        PROJECT / "envs" / "sumo" / "original_scenarios_v1" / SCENARIO / "traffic"
    ):
        raise ValueError(f"unexpected traffic source: {d1.NEW_SOURCE_TRAFFIC}")

    expected_density = {
        "vehicle_scale": 1.0,
        "pedestrian_scale": 1.0,
        "clone_depart_jitter_seconds": (0.0, 0.0),
    }
    if base.DENSITY != expected_density:
        raise ValueError(f"unexpected density/overlay parameters: {base.DENSITY!r}")
    expected_reward = {
        "success_reward": 10.0,
        "collision_reward": -10.0,
        "off_route_reward": -10.0,
        "timeout_reward": -5.0,
        "progress_scale": 0.02,
        "step_cost": 0.01,
    }
    if base.REWARD != expected_reward:
        raise ValueError(f"unexpected training reward config: {base.REWARD!r}")
    protocol = getattr(evaluation, "EVALUATION_RETURN_PROTOCOL_VERSION", None)
    raw_protocol = getattr(evaluation, "RAW_RETURN_PROTOCOL_VERSION", None)
    component_protocol = getattr(evaluation, "REWARD_COMPONENT_PROTOCOL_VERSION", None)
    if (protocol, raw_protocol, component_protocol) != (
        EVALUATION_RETURN_PROTOCOL,
        RAW_RETURN_PROTOCOL,
        REWARD_COMPONENT_PROTOCOL,
    ):
        raise ValueError(
            "evaluation reward protocol is not the requested v2/raw/component protocol: "
            f"{protocol!r}, {raw_protocol!r}, {component_protocol!r}"
        )

    protocol_checks = {
        "training_seed": base.SEED == 0,
        "action_repeat": base.ACTION_REPEAT == 3,
        "learning_starts_raw_steps": base.LEARNING_STARTS == 5_000,
        "batch_size": base.BATCH_SIZE == 32,
        "learning_rate": math.isclose(base.LEARNING_RATE["sac_mlp"], 1e-4),
        "buffer_size": base.BUFFER_SIZE == 20_000,
        "discount": math.isclose(base.DISCOUNT, 0.99),
        "eval_episodes": base.EVAL_EPISODES_TOTAL == EVAL_EPISODES,
        "eval_seed_start": base.SEED_START["sac_mlp"] == EVAL_SEED_START,
        "training_workers": d1.TRAIN_WORKERS == 2,
    }
    failed = [name for name, passed in protocol_checks.items() if not passed]
    if failed:
        raise ValueError(f"base training protocol changed: {failed}")

    configs = {method: _assert_method_config(base, d1, method) for method in METHODS}
    asset_report = inspect_sorted_assets()
    anticipated = anticipated_paths(run_root, asset_report["source_files"])
    max_path = max((len(str(p)) for p in anticipated), default=0)
    too_long = [str(p) for p in anticipated if len(str(p)) > SAFE_PATH_LIMIT]
    if too_long:
        raise OSError(
            f"refusing launch: anticipated source/output path exceeds {SAFE_PATH_LIMIT} chars; "
            f"maximum={max_path}: {too_long[:3]}"
        )

    return {
        "scenario": SCENARIO,
        "scenario_profile_random": False,
        "depart_scale": DEPART_SCALE,
        "training_seed": base.SEED,
        "raw_steps_per_method": RAW_STEPS,
        "learning_starts_raw_steps": base.LEARNING_STARTS,
        "action_repeat": base.ACTION_REPEAT,
        "checkpoint_frequency_raw_steps": CHECKPOINT_FREQUENCY_RAW,
        "methods": list(METHODS),
        "method_configs": configs,
        "method_environment_contracts": {m: d1._env_adapter_d1(m) for m in METHODS},
        "workers": d1.TRAIN_WORKERS,
        "training_hyperparameters": {
            "batch_size": base.BATCH_SIZE,
            "learning_rate": base.LEARNING_RATE["sac_mlp"],
            "buffer_size": base.BUFFER_SIZE,
            "discount": base.DISCOUNT,
            "density": dict(base.DENSITY),
        },
        "training_reward_config": dict(base.REWARD),
        "evaluation": {
            "episodes": EVAL_EPISODES,
            "seed_start": EVAL_SEED_START,
            "seed_end_inclusive": EVAL_SEED_START + EVAL_EPISODES - 1,
            "traffic_split_label": EVAL_SPLIT_LABEL,
            "traffic_pool_protocol": "same complete effective 30-template pool for train and eval; no holdout",
            "deterministic": True,
            "checkpoint": "final_model.zip",
            "primary_return_protocol": protocol,
            "raw_return_protocol": raw_protocol,
            "reward_component_protocol": component_protocol,
            "raw_reward_retained_per_episode": True,
            "reward_component_keys": list(REWARD_COMPONENT_KEYS),
        },
        "traffic": asset_report,
        "path_check": {
            "limit_chars": SAFE_PATH_LIMIT,
            "maximum_anticipated_path_chars": max_path,
            "all_anticipated_paths_within_limit": True,
        },
        "run_root": str(run_root),
    }


def _route_counts(path: Path) -> Counter:
    root = ET.parse(path).getroot()
    counts: Counter = Counter()
    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        if route is None:
            route_id = vehicle.get("route")
            counts[f"route_id:{route_id}"] += 1
        else:
            counts[" ".join(route.get("edges", "").split())] += 1
    return counts


def inspect_sorted_assets() -> dict:
    scenario_root = PROJECT / "envs" / "sumo" / "original_scenarios_v1" / SCENARIO
    traffic_root = scenario_root / "traffic"
    templates = tuple(sorted(traffic_root.glob("traffic_*.rou.xml")))
    if len(templates) != 30:
        raise FileNotFoundError(f"expected 30 sorted traffic templates, found {len(templates)}")
    if not (scenario_root / ".depart_sorted").is_file():
        raise FileNotFoundError(f"sorted-scene marker is missing: {scenario_root / '.depart_sorted'}")
    for required in (scenario_root / "map.net.xml", scenario_root / "ego.rou.xml"):
        if not required.is_file():
            raise FileNotFoundError(f"sorted-scene asset is missing: {required}")

    expected_routes = Counter({"-E3 -E0": 200, "E0 E3": 150, "E2 E1": 240})
    template_rows = []
    for path in templates:
        root = ET.parse(path).getroot()
        vehicles = root.findall("vehicle")
        order = [
            (float(vehicle.get("depart", "0")), int(vehicle.get("departLane", "0")))
            for vehicle in vehicles
        ]
        if order != sorted(order):
            raise ValueError(f"traffic template is not sorted by depart/departLane: {path}")
        route_counts = _route_counts(path)
        if route_counts != expected_routes:
            raise ValueError(
                f"sorted template does not contain the three expected background flows: "
                f"{path}: {dict(route_counts)}"
            )
        template_rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "vehicle_count": len(vehicles),
            "route_counts": dict(sorted(route_counts.items())),
        })

    ego_root = ET.parse(scenario_root / "ego.rou.xml").getroot()
    ego = ego_root.find("vehicle")
    ego_route = ego_root.find("route")
    if ego is None or ego_route is None:
        raise ValueError("sorted ego route definition is incomplete")
    ego_info = {
        "route_edges": " ".join(ego_route.get("edges", "").split()),
        "depart": float(ego.get("depart", "nan")),
        "depart_lane": ego.get("departLane"),
        "arrival_lane": ego.get("arrivalLane"),
        "depart_speed": ego.get("departSpeed"),
    }
    if (ego_info["route_edges"], ego_info["depart"], ego_info["depart_lane"]) != (
        "-E1 -E0", 50.0, "2"
    ):
        raise ValueError(f"unexpected sorted ego task: {ego_info}")

    runtime_scenario = PROJECT / "envs" / "sumo" / "scenarios" / SCENARIO
    runtime_files = tuple(sorted(p for p in runtime_scenario.rglob("*") if p.is_file()))
    for name in ("scenario.sumocfg", "routes.rou.xml"):
        if not (runtime_scenario / name).is_file():
            raise FileNotFoundError(f"SUMO runtime asset is missing: {runtime_scenario / name}")
    config_root = ET.parse(runtime_scenario / "scenario.sumocfg").getroot()
    collision_nodes = config_root.findall(".//processing/collision.action")
    collision_action = collision_nodes[0].get("value") if collision_nodes else None

    source_files = [scenario_root / ".depart_sorted", scenario_root / "ego.rou.xml",
                    scenario_root / "map.net.xml", *templates, *runtime_files]
    network = PROJECT / "envs" / "sumo" / "networks" / "intersection" / "intersection.net.xml"
    if network.is_file():
        source_files.append(network)
    source_files = list(dict.fromkeys(source_files))
    return {
        "traffic_template_count": len(templates),
        "traffic_templates": template_rows,
        "source_files": [str(path) for path in source_files],
        "background_vehicle_counts_per_template": dict(sorted(expected_routes.items())),
        "ego_task": ego_info,
        "sumo_config_collision_action": collision_action,
        "terminal_outcome_protocol": "exclusive_terminal_v2",
        "traffic_schedule_protocol": "sorted_explicit_vehicle_departures_scaled_by_4.0",
        "train_eval_pool": "same complete effective 30-template pool; no holdout",
    }


def _source_candidates() -> list[Path]:
    candidates: list[Path] = []
    for directory, children, filenames in os.walk(PROJECT):
        current = Path(directory)
        children[:] = sorted(
            name for name in children
            if name not in SKIP_DIRECTORIES and "__intersection" not in name
            and not name.startswith("_visual_eval")
        )
        if "training_complete.json" in filenames or "evaluation_results.json" in filenames:
            children[:] = []
            continue
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix.lower() in SOURCE_SUFFIXES:
                candidates.append(path)
    return candidates


def anticipated_paths(run_root: Path, scenario_files: list[str]) -> list[Path]:
    code = _source_candidates()
    paths = [run_root / "source_archive" / "src" / p.relative_to(PROJECT) for p in code]
    paths.extend(run_root / "source_archive" / "src" / Path(p).relative_to(PROJECT)
                 for p in scenario_files)
    paths.extend(
        run_root / f"{method}__{SCENARIO}_depart4p0" / name
        for method in METHODS
        for name in (
            "final_model.zip",
            "training_complete.json",
            "evaluation_results.json",
            "diagnostics/eval/manifest.json",
        )
    )
    paths.extend(
        run_root / "_hd" / method / "ns_eval" / "intersection_sorted"
        / "traffic_29.rou.xml__hdx1_i5m6s100_v1.rou.xml.manifest.json"
        for method in METHODS
    )
    paths.extend(
        run_root / "_p4_lowdensity_s4p0__intersection_sorted" / f"traffic_{index:02d}.rou.xml"
        for index in range(30)
    )
    paths.extend((
        run_root / "source_archive" / "manifest.json",
        run_root / "suite_manifest.json",
        run_root / "sorted_pair_status.json",
        run_root / "paired_method_comparison.json",
        run_root / "evaluation_protocol_validation.json",
        run_root / "launcher_logs" / "sac_mlp_d1_st_rt_topo_routeaware_v1__train.log",
        run_root / "launcher_logs" / "sac_mlp_d1_st_rt_3slot__train.log",
    ))
    return paths


def package_versions() -> dict:
    result = {}
    for package in ("torch", "stable-baselines3", "gymnasium", "numpy", "sumolib", "traci"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _archive_file(source: Path, destination: Path, records: list[dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"archive input disappeared: {source}")
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied = sha256(destination)
    if before != copied or before != sha256(source):
        raise RuntimeError(f"source changed while archiving: {source}")
    archive_root = next(
        (parent for parent in destination.parents if parent.name == "source_archive"),
        None,
    )
    if archive_root is None:
        raise ValueError(f"destination is outside source_archive: {destination}")
    records.append({
        "source": str(source),
        "copy": str(destination.relative_to(archive_root)),
        "sha256": copied,
        "bytes": destination.stat().st_size,
    })


def archive_sources(run_root: Path, effective_traffic: tuple[Path, ...]) -> dict:
    archive_root = run_root / "source_archive"
    archive_root.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    for source in _source_candidates():
        _archive_file(source, archive_root / "src" / source.relative_to(PROJECT), records)

    assets = inspect_sorted_assets()
    for source_text in assets["source_files"]:
        source = Path(source_text)
        _archive_file(source, archive_root / "src" / source.relative_to(PROJECT), records)

    context_sources = (
        WORKSPACE / "AGENTS.md",
        PROJECT / "fast-developer" / "RESEARCH_CONTEXT.md",
        PROJECT / "fast-developer" / "analysis" / "full_mst_slt_implementation.md",
        PROJECT / "fast-developer" / "analysis" / "experiment_history.md",
    )
    for source in context_sources:
        if source.is_file():
            _archive_file(source, archive_root / "research" / source.name, records)

    effective_records = []
    expected_count = assets["traffic_template_count"]
    if len(effective_traffic) != expected_count:
        raise ValueError(
            f"effective traffic pool has {len(effective_traffic)} files, expected {expected_count}"
        )
    source_template_by_name = {
        Path(row["path"]).name: Path(row["path"])
        for row in assets["traffic_templates"]
    }
    for path in effective_traffic:
        source_template = source_template_by_name.get(path.name)
        if source_template is None:
            raise ValueError(f"effective traffic file has no matching sorted template: {path}")
        _verify_scaled_traffic(source_template, path)
        destination = archive_root / "effective_traffic" / path.name
        before = sha256(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied = sha256(destination)
        if copied != before or before != sha256(path):
            raise RuntimeError(f"effective traffic changed while archiving: {path}")
        effective_records.append({
            "source": str(path),
            "copy": str(destination.relative_to(archive_root)),
            "sha256": copied,
            "bytes": destination.stat().st_size,
        })

    manifest = {
        "created_at_utc": now_utc(),
        "source_project": str(PROJECT),
        "source_file_count": len(records),
        "files": records,
        "effective_traffic_pool": {
            "count": len(effective_records),
            "depart_scale": DEPART_SCALE,
            "files": effective_records,
        },
        "sorted_template_count": assets["traffic_template_count"],
        "sorted_template_hashes": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in assets["traffic_templates"]
        ],
    }
    write_json(archive_root / "manifest.json", manifest)
    return manifest


def _verify_scaled_traffic(source: Path, scaled: Path) -> None:
    source_root = ET.parse(source).getroot()
    scaled_root = ET.parse(scaled).getroot()
    source_vehicles = {v.get("id"): v for v in source_root.findall("vehicle")}
    scaled_vehicles = {v.get("id"): v for v in scaled_root.findall("vehicle")}
    if source_vehicles.keys() != scaled_vehicles.keys():
        raise ValueError(f"scaled traffic vehicle identities differ: {source} -> {scaled}")
    for vehicle_id, source_vehicle in source_vehicles.items():
        scaled_vehicle = scaled_vehicles[vehicle_id]
        expected_depart = float(source_vehicle.get("depart", "0")) * DEPART_SCALE
        actual_depart = float(scaled_vehicle.get("depart", "nan"))
        if not math.isclose(actual_depart, expected_depart, abs_tol=0.00051):
            raise ValueError(
                f"depart_scale mismatch for {vehicle_id}: {actual_depart} vs {expected_depart}"
            )
        for key, value in source_vehicle.attrib.items():
            if key != "depart" and scaled_vehicle.get(key) != value:
                raise ValueError(f"scaled vehicle attribute changed: {vehicle_id}.{key}")
        source_route = source_vehicle.find("route")
        scaled_route = scaled_vehicle.find("route")
        if (source_route is None) != (scaled_route is None):
            raise ValueError(f"scaled route missing for vehicle {vehicle_id}")
        if source_route is not None and source_route.attrib != scaled_route.attrib:
            raise ValueError(f"scaled route changed for vehicle {vehicle_id}")


def build_worker_command(run_root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT / "fast-developer" / "train_intersection_yield_v2_d1.py"),
        "--scenario", SCENARIO,
        "--depart-scale", str(DEPART_SCALE),
        "--eval-traffic-split", EVAL_SPLIT_LABEL,
        "--methods", ",".join(METHODS),
        "--max-steps", str(RAW_STEPS),
        "--run-root", str(run_root),
        "--checkpoint-frequency", str(CHECKPOINT_FREQUENCY_RAW),
        "--behavior-diagnostics",
    ]


def _run_dir(run_root: Path, method: str) -> Path:
    return run_root / f"{method}__{SCENARIO}_depart4p0"


def validate_completed_outputs(run_root: Path) -> dict:
    base, d1, _, _ = load_runtime_modules()
    rows = {}
    problems = []
    for method in METHODS:
        run_dir = _run_dir(run_root, method)
        required = {
            "arguments": run_dir / "arguments.json",
            "training_complete": run_dir / "training_complete.json",
            "final_model": run_dir / "final_model.zip",
            "evaluation": run_dir / "evaluation_results.json",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            problems.append(f"{method}: missing {missing}")
            rows[method] = {"status": "incomplete", "missing": missing}
            continue
        args = json.loads(required["arguments"].read_text(encoding="utf-8"))
        training = json.loads(required["training_complete"].read_text(encoding="utf-8"))
        result = json.loads(required["evaluation"].read_text(encoding="utf-8"))
        identity = result.get("identity") or {}
        records = result.get("episode_records") or []
        checkpoint_hash = sha256(required["final_model"])
        expected_seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_EPISODES))
        actual_seeds = [int(record.get("seed", -1)) for record in records]
        record_checks = {
            "episode_count": len(records) == EVAL_EPISODES,
            "eval_seeds": actual_seeds == expected_seeds,
            "evaluation_return_protocol": all(
                record.get("evaluation_return_protocol_version") == EVALUATION_RETURN_PROTOCOL
                for record in records
            ),
            "raw_return_retained": all(
                record.get("raw_episode_return") is not None
                and record.get("raw_return_source") == "info.undiscounted_reward"
                for record in records
            ),
            "shaped_reward_components": all(
                record.get("reward_component_protocol_version") == REWARD_COMPONENT_PROTOCOL
                and all(record.get(key) is not None for key in REWARD_COMPONENT_KEYS)
                for record in records
            ),
            "reward_reconciliation": all(
                record.get("reward_component_reconciliation_error") is not None
                and float(record["reward_component_reconciliation_error"]) <= 1e-5
                for record in records
            ),
        }
        checks = {
            "method": args.get("method") == method,
            "scenario": args.get("scenario") == SCENARIO,
            "parent": args.get("parent") == "sac_mlp",
            "env_contract": args.get("env_contract") == "base",
            "fresh_seed0": args.get("seed") == 0 and args.get("smoke") is False,
            "raw_budget": args.get("raw_budget") == RAW_STEPS
            and training.get("raw_steps") == RAW_STEPS,
            "warmup": args.get("learning_starts_raw_steps") == 5_000,
            "action_repeat": args.get("action_repeat") == 3,
            "cuda": args.get("device") == "cuda",
            "diagnostics": args.get("behavior_diagnostics") is True
            and training.get("behavior_diagnostics") is True,
            "training_checkpoint_hash": training.get("checkpoint_sha256") == checkpoint_hash,
            "evaluation_checkpoint_hash": identity.get("checkpoint_sha256") == checkpoint_hash,
            "evaluation_identity": identity.get("scenario") == SCENARIO
            and identity.get("eval_traffic_split") == EVAL_SPLIT_LABEL
            and identity.get("episodes") == EVAL_EPISODES
            and identity.get("smoke") is False,
            "evaluation_records": all(record_checks.values()),
            "checkpoint_frequency_raw_steps": (
                args.get("checkpoint_frequency") == CHECKPOINT_FREQUENCY_RAW
            ),
            "density_no_extra_sampling": args.get("density") == {
                "vehicle_scale": 1.0,
                "pedestrian_scale": 1.0,
                "clone_depart_jitter_seconds": [0.0, 0.0],
            },
            "training_hyperparameters": (
                args.get("batch_size") == 32
                and math.isclose(float(args.get("learning_rate", -1)), 1e-4)
                and args.get("buffer_size") == 20_000
                and math.isclose(float(args.get("discount", -1)), 0.99)
            ),
        }
        checks["expected_updates_95001"] = training.get("updates") == 95_001
        expected_cfg = _assert_method_config(base, d1, method)
        for field, value in expected_cfg.items():
            if field in args:
                checks[f"method_config_{field}"] = args.get(field) == value
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            problems.append(f"{method}: validation failures {failures}")
        shaped_returns = [float(record["episode_return"]) for record in records]
        raw_returns = [float(record["raw_episode_return"]) for record in records
                       if record.get("raw_episode_return") is not None]
        rows[method] = {
            "status": "valid" if not failures else "validation_failed",
            "checks": checks,
            "training_raw_steps": training.get("raw_steps"),
            "training_updates": training.get("updates"),
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_episodes": len(records),
            "evaluation_seed_min": min(actual_seeds) if actual_seeds else None,
            "evaluation_seed_max": max(actual_seeds) if actual_seeds else None,
            "evaluation_shaped_return_mean": (
                sum(shaped_returns) / len(shaped_returns) if shaped_returns else None
            ),
            "evaluation_raw_return_mean": (
                sum(raw_returns) / len(raw_returns) if raw_returns else None
            ),
            "reward_component_coverage": sum(
                1 for record in records
                if record.get("reward_component_protocol_version") == REWARD_COMPONENT_PROTOCOL
            ),
            "result_path": str(required["evaluation"]),
        }

    report = {
        "created_at_utc": now_utc(),
        "run_root": str(run_root),
        "expected_evaluation_return_protocol": EVALUATION_RETURN_PROTOCOL,
        "raw_reward_retained_per_episode": True,
        "methods": rows,
        "problems": problems,
        "status": "passed" if not problems else "failed",
    }
    write_json(run_root / "evaluation_protocol_validation.json", report)
    return report


def launch(run_root: Path, plan: dict) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    base, d1, _, _ = load_runtime_modules()
    d1._apply_patch(DEPART_SCALE, SCENARIO, EVAL_SPLIT_LABEL)
    d1.ensure_sorted_scenario()
    base.RESULT_ROOT = run_root
    effective_traffic = base._traffic_paths_for_scale(DEPART_SCALE)
    assets = inspect_sorted_assets()
    manifest_source = archive_sources(run_root, tuple(effective_traffic))

    command = build_worker_command(run_root)
    manifest = {
        "created_at_utc": now_utc(),
        "experiment_id": "sorted_routeaware_topo_vs_st_rt_3slot_fresh100k",
        "protocol_revision": "environment_step_reward_v2_sorted_pair_v1",
        "scenario": SCENARIO,
        "scenario_revision": "intersection_sorted_depart_sorted_templates",
        "traffic_protocol": "fixed explicit-vehicle schedule; depart timestamps multiplied by 4.0",
        "traffic_template_count": 30,
        "training_evaluation_traffic_pool": "same complete effective 30-template pool; no holdout",
        "training_seed": 0,
        "raw_steps_per_method": RAW_STEPS,
        "learning_starts_raw_steps": 5_000,
        "action_repeat": 3,
        "expected_updates": 95_001,
        "checkpoint_frequency_raw_steps": CHECKPOINT_FREQUENCY_RAW,
        "fresh_model": True,
        "resume": False,
        "smoke": False,
        "max_cuda_workers": 2,
        "cuda_visible_devices": "0",
        "method_workers_are_independent_processes": True,
        "methods": list(METHODS),
        "method_configs": plan["method_configs"],
        "method_environment_contracts": plan["method_environment_contracts"],
        "training_hyperparameters": plan["training_hyperparameters"],
        "training_reward_config": plan["training_reward_config"],
        "terminal_outcome_protocol": "exclusive_terminal_v2",
        "sumo_config_collision_action": assets["sumo_config_collision_action"],
        "evaluation": plan["evaluation"],
        "evaluation_reward_accounting": {
            "primary_episode_return": "sum of shaped reward returned by env.step",
            "raw_episode_return": "sum of info.undiscounted_reward, retained per episode",
            "reward_components": list(REWARD_COMPONENT_KEYS),
            "component_sum_reconciled_to_primary_return": True,
        },
        "traffic_assets": {
            "source_template_count": assets["traffic_template_count"],
            "source_templates": assets["traffic_templates"],
            "effective_scaled_pool": manifest_source["effective_traffic_pool"],
            "ego_task": assets["ego_task"],
        },
        "run_root": str(run_root),
        "command": command,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": package_versions(),
        "source_archive_manifest": str(run_root / "source_archive" / "manifest.json"),
        "source_archive_file_count": manifest_source["source_file_count"],
    }
    write_json(run_root / "suite_manifest.json", manifest)
    status_path = run_root / "sorted_pair_status.json"
    status = {
        "status": "starting",
        "created_at_utc": now_utc(),
        "launcher_pid": os.getpid(),
        "run_root": str(run_root),
        "methods": list(METHODS),
        "max_cuda_workers": 2,
        "command": command,
    }
    write_json(status_path, status)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    stdout_path = run_root / "supervisor.stdout.log"
    stderr_path = run_root / "supervisor.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        status.update(status="running", supervisor_pid=process.pid, started_at_utc=now_utc())
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        exit_code = process.wait()

    status.update(
        status="child_complete" if exit_code == 0 else "child_failed",
        child_exit_code=exit_code,
        child_finished_at_utc=now_utc(),
    )
    write_json(status_path, status)
    if exit_code != 0:
        return exit_code

    validation = validate_completed_outputs(run_root)
    status["evaluation_reward_validation"] = validation["status"]
    if validation["status"] != "passed":
        status.update(status="validation_failed", exit_code=2)
        write_json(status_path, status)
        return 2

    try:
        from paired_stage_comparison import compare_stages

        pair_result = compare_stages(
            _run_dir(run_root, METHODS[0]) / "evaluation_results.json",
            _run_dir(run_root, METHODS[1]) / "evaluation_results.json",
            run_root / "paired_method_comparison.json",
            label="routeaware_topo_vs_st_rt_3slot",
            expected_episodes=EVAL_EPISODES,
        )
        status["pairing_status"] = pair_result.get("status", "written")
    except Exception as error:
        status["pairing_status"] = "failed"
        status["pairing_error"] = repr(error)
    status.update(status="complete", exit_code=0, finished_at_utc=now_utc())
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--start",
        action="store_true",
        help="create a fresh root, archive exact inputs, and start the two-method queue",
    )
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    runs_root = (WORKSPACE / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        parser.error(f"--run-root must be a new child of {runs_root}")
    if run_root.exists():
        parser.error(f"refusing existing run root: {run_root}")

    plan = validate_runtime_plan(run_root)
    command = build_worker_command(run_root)
    if not args.start:
        compact_plan = dict(plan)
        compact_plan["traffic"] = {
            key: plan["traffic"][key]
            for key in (
                "traffic_template_count",
                "background_vehicle_counts_per_template",
                "ego_task",
                "sumo_config_collision_action",
                "terminal_outcome_protocol",
                "traffic_schedule_protocol",
                "train_eval_pool",
            )
        }
        compact_plan["traffic"]["template_sha256_by_name"] = {
            Path(row["path"]).name: row["sha256"]
            for row in plan["traffic"]["traffic_templates"]
        }
        print(json.dumps({
            "mode": "preview_only",
            "training_started": False,
            "run_root_exists": run_root.exists(),
            "command": command,
            "plan": compact_plan,
            "note": "No run files, SUMO environments, or training processes were created.",
        }, ensure_ascii=False, indent=2))
        return 0
    return launch(run_root, plan)


if __name__ == "__main__":
    raise SystemExit(main())
