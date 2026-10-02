"""Versioned, seeded random traffic on the existing paper intersection.

The scenarios share geometry, ego task and driver distributions.  The original
three rates use native SUMO probability flows.  The separately named step
variants pre-sample a Bernoulli request per lane and tick because their requested
rates exceed SUMO's native probability-flow range.  Splits use disjoint seeds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET


SUMO_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = SUMO_ROOT / "original_scenarios_v1"
PROTOCOL = "intersection_bernoulli_v1"
SOURCE_SCENARIO = "intersection_sorted"
_RATE_SCENARIOS = tuple(
    f"intersection_random_{level}_v1" for level in ("low", "medium", "high")
)
STEP_P05_SCENARIO = "intersection_random_medium_p05_v1"
STEP_SCHEDULE_PROTOCOL = "intersection_bernoulli_step_schedule_v1"
DARRL_STEP_PROBABILITIES = {
    "intersection_random_darrl_low_v1": 0.015,
    "intersection_random_darrl_medium_v1": 0.03,
    "intersection_random_darrl_high_v1": 0.05,
}
STEP_PROBABILITIES = {
    STEP_P05_SCENARIO: 0.5,
    "intersection_random_medium_p03_v1": 0.3,
    "intersection_random_medium_p02_v1": 0.2,
}
RANDOM_INTERSECTION_SCENARIOS = _RATE_SCENARIOS + tuple(STEP_PROBABILITIES) + tuple(DARRL_STEP_PROBABILITIES)
_MULTIPLIERS = dict(zip(_RATE_SCENARIOS, (1, 2, 3)))
# Preserve the source schedules' 200:150:240 directional demand ratio.
_ROUTES = (
    ("neg_E3_to_neg_E0", ("-E3", "-E0"), 200),
    ("E0_to_E3", ("E0", "E3"), 150),
    ("E2_to_E1", ("E2", "E1"), 240),
)
SEED_DOMAINS = {
    "train": (0, 1_000_000_000),
    "validation": (1_000_000_000, 500_000_000),
    "test": (1_500_000_000, 500_000_000),
}


def is_random_intersection_scenario(name: str) -> bool:
    return name in RANDOM_INTERSECTION_SCENARIOS


def random_intersection_seed(split: str, requested_seed: int) -> int:
    """Map a logical episode seed to a disjoint SUMO RNG seed domain."""
    if split not in SEED_DOMAINS:
        raise ValueError(f"Unknown traffic split {split!r}; choose {tuple(SEED_DOMAINS)}")
    if int(requested_seed) < 0:
        raise ValueError("Traffic episode seeds must be nonnegative")
    offset, width = SEED_DOMAINS[split]
    return offset + int(requested_seed) % width


def get_random_intersection_config(name: str) -> dict:
    if name in STEP_PROBABILITIES or name in DARRL_STEP_PROBABILITIES:
        # Preserve all existing versioned configurations byte-for-byte.  This
        # variant has a separate name and a different demand sampler because
        # SUMO's native per-second flow probability cannot exceed one.
        probability = (
            DARRL_STEP_PROBABILITIES if name in DARRL_STEP_PROBABILITIES else STEP_PROBABILITIES
        )[name]
        vehicles_per_hour = int(round(probability * 36000))
        total_vehicles_per_hour = vehicles_per_hour * len(_ROUTES)
        template_level = (
            name.removeprefix("intersection_random_darrl_").removesuffix("_v1")
            if name in DARRL_STEP_PROBABILITIES else "medium"
        )
        template_scenario = f"intersection_random_{template_level}_v1"
        config = get_random_intersection_config(template_scenario)
        config.update({
            "protocol": STEP_SCHEDULE_PROTOCOL,
            "scenario": name,
            "density_level": name.removeprefix("intersection_random_").removesuffix("_v1"),
            "sumo_probability_semantics": "not a native probability flow; an episode-seeded Bernoulli request is sampled independently for each inlet at each 0.1-second tick",
            "arrival_sampler": "sha256-seeded Python Random.random per route; sample once per tick; ordered explicit vehicle departures",
            "dynamic_episode_traffic": True,
            "demand_definition": f"user-requested p={probability:g} per 0.1-second step; {total_vehicles_per_hour} veh/h requested, realized insertion is capacity-limited",
            "calibration_status": "user-named medium variant; demand is not calibrated medium difficulty",
        })
        for flow in config["flows"]:
            flow.update({
                "vehicles_per_hour": vehicles_per_hour,
                "rate_per_second": vehicles_per_hour / 3600.0,
                "sumo_probability": None,
                "probability_per_step": probability,
                "mean_headway_seconds": 3600.0 / vehicles_per_hour,
            })
        config["total_vehicles_per_hour"] = total_vehicles_per_hour
        config["vehicles_per_hour_by_route"] = {flow["id"]: vehicles_per_hour for flow in config["flows"]}
        if name in DARRL_STEP_PROBABILITIES:
            config.update({
                "scenario_family": "darrl_lane1_v1",
                "configuration_revision": "darrl_r2_20260930",
                "previous_configuration_revision": "darrl_r1_20260930",
                "terminal_outcome_protocol": "exclusive_terminal_v2",
                "template_scenario": template_scenario,
                "density_level": name.removeprefix("intersection_random_darrl_").removesuffix("_v1"),
                "collision_action": "remove",
                "collision_detection_mode": "sumo_events_or_ego_geometry",
                "background_vtype_overrides": {"jmIgnoreFoeProb": "0"},
                "ego_vtype_overrides": {"minGap": "1", "jmIgnoreFoeProb": "0"},
                "calibration_status": "user-specified demand tiers; learned-policy difficulty not yet calibrated",
            })
            for flow in config["flows"]:
                flow.update({"depart_lane": "1", "arrival_lane": "1"})
        return config
    if name not in _MULTIPLIERS:
        raise ValueError(f"Not a random intersection scenario: {name!r}")
    multiplier = _MULTIPLIERS[name]
    flows = [
        {
            "id": route_id,
            "edges": list(edges),
            "vehicles_per_hour": base_rate * multiplier,
            "rate_per_second": base_rate * multiplier / 3600.0,
            "sumo_probability": base_rate * multiplier / 3600.0,
            "probability_per_step": base_rate * multiplier / 36000.0,
            "mean_headway_seconds": 3600.0 / (base_rate * multiplier),
            "depart_lane": "0",
            "depart_pos": "0",
            "depart_speed": "max",
        }
        for route_id, edges, base_rate in _ROUTES
    ]
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scenario": name,
        "density_level": name.removeprefix("intersection_random_").removesuffix("_v1"),
        "source_scenario": SOURCE_SCENARIO,
        "arrival_process": "independent Bernoulli requests per inlet lane per 0.1-second simulation step",
        "sumo_probability_semantics": "flow probability is per second; SUMO scales it by step length",
        "demand_definition": "scheduled background arrivals, not realized road occupancy",
        "total_vehicles_per_hour": sum(flow["vehicles_per_hour"] for flow in flows),
        "vehicles_per_hour_by_route": {flow["id"]: flow["vehicles_per_hour"] for flow in flows},
        "flows": flows,
        "flow_begin_seconds": 0.0,
        "flow_end_seconds": 130.0,
        "ego_depart_seconds": 50.0,
        "ego_route_edges": ["-E1", "-E0"],
        "ego_depart_lane": "2",
        "ego_depart_speed": "0",
        "step_length_seconds": 0.1,
        "max_episode_raw_steps": 600,
        "default_action_repeat": 3,
        "endless_traffic": False,
        "background_exit_behavior": "leave after the configured route ends; no recycling",
        "depart_scale": 1.0,
        "driver_distribution": "all source templates; route-conditional empirical vehicle-type weights",
        "traffic_seed_domains": {
            split: {"offset": offset, "width": width, "last_seed": offset + width - 1}
            for split, (offset, width) in SEED_DOMAINS.items()
        },
        "traffic_seed_mapping": "offset[split] + logical_episode_seed % width[split]",
        "default_evaluation_split": "validation",
        "default_evaluation_logical_seeds": {"start": 10000, "episodes": 100},
        "holdout_unit": "SUMO simulation RNG seed, including random arrival realization",
        "calibration_status": "predefined demand tiers; learned-policy difficulty not yet calibrated",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _source_distributions() -> tuple[list[ET.Element], dict, dict]:
    source = ASSET_ROOT / SOURCE_SCENARIO
    traffic_paths = tuple(sorted((source / "traffic").glob("traffic_*.rou.xml")))
    if len(traffic_paths) != 30:
        raise ValueError(f"Expected the frozen 30-template source pool, got {len(traffic_paths)}")
    types: list[ET.Element] = []
    weights = {edges: {} for _, edges, _ in _ROUTES}
    source_hashes = {}
    for index, path in enumerate(traffic_paths):
        source_hashes[path.name] = _sha256(path)
        root = ET.parse(path).getroot()
        id_map = {}
        for original in root.findall("vType"):
            new_type = copy.deepcopy(original)
            original_id = original.attrib["id"]
            new_id = f"source{index:02d}_{original_id}"
            new_type.set("id", new_id)
            id_map[original_id] = new_id
            types.append(new_type)
        for vehicle in root.findall("vehicle"):
            route = vehicle.find("route")
            if route is None:
                raise ValueError(f"Source vehicle lacks an explicit route: {path}")
            edges = tuple(route.attrib["edges"].split())
            if edges not in weights:
                raise ValueError(f"Unexpected source route {edges} in {path}")
            for field, expected in (("departLane", "0"), ("departPos", "0"), ("departSpeed", "max")):
                if vehicle.get(field) != expected:
                    raise ValueError(f"Source {field} differs from the frozen protocol: {path}")
            type_id = id_map[vehicle.attrib["type"]]
            weights[edges][type_id] = weights[edges].get(type_id, 0) + 1
    if len(types) != 120 or any(not value for value in weights.values()):
        raise ValueError("Unexpected source driver distribution")
    return types, weights, source_hashes


def _traffic_xml(config: dict, types: list[ET.Element], weights: dict) -> bytes:
    root = ET.Element("routes")
    root.append(ET.Comment("Independent seeded random arrivals; do not multiply depart times."))
    for vtype in types:
        vtype = copy.deepcopy(vtype)
        for key, value in config.get("background_vtype_overrides", {}).items():
            vtype.set(key, str(value))
        root.append(vtype)
    for flow in config["flows"]:
        route_weights = weights[tuple(flow["edges"])]
        ordered = sorted(route_weights)
        ET.SubElement(root, "vTypeDistribution", {
            "id": f"drivers_{flow['id']}",
            "vTypes": " ".join(ordered),
            # SUMO normalizes positive relative weights.
            "probabilities": " ".join(str(route_weights[key]) for key in ordered),
        })
        ET.SubElement(root, "route", {"id": f"route_{flow['id']}", "edges": " ".join(flow["edges"])})
    for flow in config["flows"]:
        if config.get("dynamic_episode_traffic"):
            continue
        ET.SubElement(root, "flow", {
            "id": f"background_{flow['id']}",
            "type": f"drivers_{flow['id']}",
            "route": f"route_{flow['id']}",
            "begin": "0",
            "end": "130",
            "probability": f"{flow['sumo_probability']:.15g}",
            "departLane": flow["depart_lane"],
            "departPos": flow["depart_pos"],
            "departSpeed": flow["depart_speed"],
            **({"arrivalLane": flow["arrival_lane"]} if "arrival_lane" in flow else {}),
        })
    return _xml_bytes(root)


def write_seeded_episode_traffic(
    scenario: str, actual_seed: int, traffic_path: Path, output_path: Path,
) -> dict:
    """Materialize the exact per-tick Bernoulli process before SUMO starts.

    Pre-sampling is distributionally equivalent to drawing during each tick,
    and keeps demand independent of policy actions.  SUMO retains requests
    whose insertion is blocked; no request is discarded by this generator.
    The static traffic file supplies the frozen route/type distributions.
    """
    config = get_random_intersection_config(scenario)
    if not config.get("dynamic_episode_traffic"):
        raise ValueError(f"{scenario} uses native SUMO probability flows")
    root = ET.parse(traffic_path).getroot()
    if root.findall("flow") or root.findall("vehicle"):
        raise ValueError("The seeded schedule template must contain definitions only")
    dt = float(config["step_length_seconds"])
    first_tick = round(float(config["flow_begin_seconds"]) / dt)
    last_tick = round(float(config["flow_end_seconds"]) / dt)
    requests = []
    departures_by_route = {}
    for route_index, flow in enumerate(config["flows"]):
        seed_material = f"{STEP_SCHEDULE_PROTOCOL}:{int(actual_seed)}:{flow['id']}:arrival"
        route_seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest(), "big")
        rng = random.Random(route_seed)
        departures = []
        for tick in range(first_tick, last_tick):
            if rng.random() < float(flow["probability_per_step"]):
                departure = round(tick * dt, 10)
                sequence = len(departures)
                departures.append(departure)
                requests.append((tick, route_index, sequence, flow))
        departures_by_route[flow["id"]] = departures
    for tick, _route_index, sequence, flow in sorted(requests, key=lambda item: item[:3]):
        ET.SubElement(root, "vehicle", {
            "id": f"background_{flow['id']}.{sequence}",
            "type": f"drivers_{flow['id']}",
            "route": f"route_{flow['id']}",
            "depart": f"{tick * dt:.1f}",
            "departLane": flow["depart_lane"],
            "departPos": flow["depart_pos"],
            "departSpeed": flow["depart_speed"],
            **({"arrivalLane": flow["arrival_lane"]} if "arrival_lane" in flow else {}),
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _xml_bytes(root)
    output_path.write_bytes(payload)
    return {
        "sampler": STEP_SCHEDULE_PROTOCOL,
        "actual_seed": int(actual_seed),
        "scheduled_vehicle_count": len(requests),
        "scheduled_count_by_route": {key: len(value) for key, value in departures_by_route.items()},
        "sha256": hashlib.sha256(payload).hexdigest(),
        "departure_times_by_route": departures_by_route,
    }


def _write_new_or_identical(path: Path, data: bytes, *, rebuild: bool = False) -> None:
    """Only explicit draft rebuilding may replace an existing new asset."""
    if path.is_file():
        if path.read_bytes() != data:
            if not rebuild:
                raise ValueError(f"Versioned scenario asset differs; use a new protocol version: {path}")
            path.write_bytes(data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _validate_existing_assets(name: str) -> dict | None:
    asset_dir = ASSET_ROOT / name
    manifest_path = asset_dir / "scenario_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("configuration") != get_random_intersection_config(name):
        raise ValueError(f"Scenario configuration changed without a version change: {name}")
    for relative, expected in manifest["asset_sha256"].items():
        path = asset_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Missing or modified versioned scenario asset: {path}")
    placeholder = SUMO_ROOT / "scenarios" / name / "scenario.sumocfg"
    if not placeholder.is_file():
        raise FileNotFoundError(placeholder)
    return manifest


def ensure_random_intersection_assets(name: str | None = None, *, rebuild: bool = False) -> dict:
    """Create only the new named scenarios, with a source and asset hash ledger.

    ``rebuild`` is for explicit edits during initial scenario setup. Training
    entry points must retain the default so experiment assets stay frozen.
    """
    names = RANDOM_INTERSECTION_SCENARIOS if name is None else (name,)
    for candidate in names:
        get_random_intersection_config(candidate)
    existing = {
        candidate: None if rebuild else _validate_existing_assets(candidate)
        for candidate in names
    }
    if all(value is not None for value in existing.values()):
        return existing
    source_dir = ASSET_ROOT / SOURCE_SCENARIO
    source_network = source_dir / "map.net.xml"
    source_ego = source_dir / "ego.rou.xml"
    ego_root = ET.parse(source_ego).getroot()
    ego = ego_root.find("vehicle")
    if ego is None or ego.get("depart") is None or not math.isclose(float(ego.get("depart")), 50.0):
        raise ValueError("The source ego release time does not match the frozen protocol")
    ego_route = ego.find("route")
    if ego_route is None and ego.get("route"):
        ego_route = next((
            route for route in ego_root.findall("route")
            if route.get("id") == ego.get("route")
        ), None)
    if ego_route is None or ego_route.get("edges", "").split() != ["-E1", "-E0"]:
        raise ValueError("The source ego route does not match the frozen protocol")
    if ego.get("departLane") != "2" or float(ego.get("departSpeed", "nan")) != 0.0:
        raise ValueError("The source ego departure settings do not match the frozen protocol")
    types, weights, traffic_hashes = _source_distributions()
    for candidate in names:
        if existing[candidate] is not None:
            continue
        config = get_random_intersection_config(candidate)
        directory = ASSET_ROOT / candidate
        ego_payload = source_ego.read_bytes()
        if config.get("ego_vtype_overrides"):
            derived_ego_root = copy.deepcopy(ego_root)
            ego_type_id = ego.attrib.get("type")
            ego_type = next((
                item for item in derived_ego_root.findall("vType")
                if item.get("id") == ego_type_id
            ), None)
            if ego_type is None:
                raise ValueError(f"Cannot override missing ego vType {ego_type_id!r}")
            for key, value in config["ego_vtype_overrides"].items():
                ego_type.set(key, str(value))
            ego_payload = _xml_bytes(derived_ego_root)
        data = {
            "map.net.xml": source_network.read_bytes(),
            "ego.rou.xml": ego_payload,
            "traffic/traffic_00000.rou.xml": _traffic_xml(config, types, weights),
        }
        for relative, content in data.items():
            _write_new_or_identical(directory / relative, content, rebuild=rebuild)
        # PaperSumoSceneEnv initially calls the base registry. This placeholder
        # is deliberately non-runnable there; the paper environment uses the
        # original road network above instead.
        placeholder = ET.Element("configuration")
        inputs = ET.SubElement(placeholder, "input")
        ET.SubElement(inputs, "net-file", {"value": "../../networks/intersection/intersection.net.xml"})
        ET.SubElement(inputs, "route-files", {"value": "routes.rou.xml"})
        base_directory = SUMO_ROOT / "scenarios" / candidate
        _write_new_or_identical(base_directory / "scenario.sumocfg", _xml_bytes(placeholder), rebuild=rebuild)
        _write_new_or_identical(base_directory / "routes.rou.xml", _xml_bytes(ET.Element("routes")), rebuild=rebuild)
        manifest = {
            "configuration": config,
            "source_sha256": {
                "map.net.xml": _sha256(source_network),
                "ego.rou.xml": _sha256(source_ego),
                "traffic": traffic_hashes,
            },
            "asset_sha256": {relative: hashlib.sha256(content).hexdigest() for relative, content in data.items()},
            "driver_types": len(types),
            "source_vehicle_counts_by_route": {
                " ".join(edges): sum(counts.values()) for edges, counts in weights.items()
            },
        }
        _write_new_or_identical(directory / "scenario_manifest.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), rebuild=rebuild)
        existing[candidate] = manifest
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=RANDOM_INTERSECTION_SCENARIOS)
    args = parser.parse_args()
    manifests = ensure_random_intersection_assets(args.scenario)
    print(json.dumps({name: value["configuration"] for name, value in manifests.items()}, indent=2))


if __name__ == "__main__":
    main()
