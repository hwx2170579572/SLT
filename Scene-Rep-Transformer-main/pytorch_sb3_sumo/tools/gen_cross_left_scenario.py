"""Generate the ``cross_left`` SUMO scenario: a 4-way cross unprotected-left-turn.

Mirrors the SMARTS ``gen_traffic`` idiom recorded in ``scenario_source.txt``
(flows + randomised TrafficActor params), but emits explicit ``<vehicle>``
elements deterministically (seeded per traffic file) so the released additive
high-density overlay's explicit-vehicle cloning path applies unchanged.

Geometry (from ``envs/sumo/networks/intersection/intersection.net.xml``):
  8 edges -- north/east/south/west ``_in``/``_out`` -- 2 driving lanes each
  (index 1 = right, index 2 = left; index 0 is a guessed sidewalk).  Junction
  ``C`` is ``type="priority"`` (unprotected / yield).

  Ego: ``south_in`` lane 2 -> ``west_out`` lane 2 (unprotected left turn).
  Social traffic: straight on all four approaches (both lanes) + right turns
  (right lane only).  NO social left turns: the ego is the sole left-turner,
  which makes the unprotected-left-turn the scenario's signature challenge.

Density: ~1000 veh/h total across 12 flows -> ~42 departures in the first
150 s (between left_turn's ~40 and cross's ~30, on the dense end), then the
1.4x additive overlay in the experiment protocol.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "envs" / "sumo" / "original_scenarios_v1" / "cross_left"
NETWORK_SRC = (
    ROOT / "envs" / "sumo" / "networks" / "intersection" / "intersection.net.xml"
)

NUM_FILES = 60
HORIZON_SECONDS = 3600.0
# (source_edge, target_edge, depart_lane, arrival_lane, vehs_per_hour)
_FLOWS = []
for _lane in (1, 2):
    _FLOWS.append(("north_in", "south_out", _lane, _lane, 100.0))  # straight N->S
    _FLOWS.append(("south_in", "north_out", _lane, _lane, 100.0))  # straight S->N
    _FLOWS.append(("east_in", "west_out", _lane, _lane, 100.0))   # straight E->W
    _FLOWS.append(("west_in", "east_out", _lane, _lane, 100.0))   # straight W->E
_FLOWS += [
    ("south_in", "east_out", 1, 1, 50.0),  # right S->E
    ("east_in", "north_out", 1, 1, 50.0),  # right E->N
    ("north_in", "west_out", 1, 1, 50.0),  # right N->W
    ("west_in", "south_out", 1, 1, 50.0),  # right W->S
]

EGO_ROUTE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="scene_rep_ego_type" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0" length="4.7" minGap="2.0" maxSpeed="10.0" color="0,0.35,1"/>
    <route id="scene_rep_ego_route" edges="south_in west_out"/>
    <vehicle id="ego" type="scene_rep_ego_type" route="scene_rep_ego_route" depart="15" departLane="2" departPos="40" departSpeed="0" arrivalLane="2" arrivalPos="max"/>
</routes>
"""

SCENARIO_SOURCE_TXT = '''"""SMARTS sstudio reference for the 4-way cross unprotected-left-turn.

This scenario reuses the 4-arm priority (yield) network at
envs/sumo/networks/intersection/intersection.net.xml and adds social traffic in
the same idiom as the released left_turn/cross scenarios (SMARTS v0.4.17,
Huawei's NeurIPS-2020 SMARTS): per-lane flows with randomised TrafficActor
speed/min-gap/imperfection/lane-change/junction parameters, expanded to
explicit vehicles.  The ego is the sole left-turner (south_in -> west_out).

The generator tools/gen_cross_left_scenario.py emits deterministic explicit
<vehicle> elements directly (seeded), equivalent to the SMARTS gen_traffic ->
duarouter pipeline, so the additive high-density overlay clones vehicles as it
does for the released scenarios.
"""
from pathlib import Path
import os

from smarts.sstudio import gen_scenario, gen_missions, gen_traffic
from smarts.sstudio.types import (
    Scenario, Traffic, Flow, Route, TrafficActor, Distribution,
    LaneChangingModel, JunctionModel, Mission,
)
import numpy as np

scenario = os.path.dirname(os.path.realpath(__file__))

# straight (2 lanes) + right-turn flows on all four arms; no social left turns.
straight = {
    "north": ("north_in", "south_out"),
    "south": ("south_in", "north_out"),
    "east": ("east_in", "west_out"),
    "west": ("west_in", "east_out"),
}
right_turn = {
    "south": ("south_in", "east_out"),
    "east": ("east_in", "north_out"),
    "north": ("north_in", "west_out"),
    "west": ("west_in", "south_out"),
}

for seed in np.random.choice(1000, 60, replace=False):
    actors = {}
    for i in range(4):
        car = TrafficActor(
            name=f"car_type_{i+1}",
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(
                speed_gain=np.random.uniform(1.0, 2.0),
                impatience=np.random.uniform(0, 1.0),
                cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(
                ignore_foe_prob=np.random.uniform(0, 1.0),
                impatience=np.random.uniform(0, 1.0)),
        )
        actors[car] = 0.25

    flows = []
    for (src, dst) in straight.values():
        flows += [Flow(route=Route(begin=(src, lane, "random"), end=(dst, lane, "random")),
                       rate=100, actors=actors) for lane in (1, 2)]
    for (src, dst) in right_turn.values():
        flows += [Flow(route=Route(begin=(src, 1, "random"), end=(dst, 1, "max")),
                       rate=50, actors=actors)]

    gen_traffic(scenario, Traffic(flows=flows), seed=seed, name=f"traffic_{seed}")

# ego: approach from the south arm and take an unprotected left turn to the west.
gen_missions(scenario=scenario, missions=[
    Mission(Route(begin=("south_in", 2, 40), end=("west_out", 2, "max")), start_time=15),
])
'''


def _format(value: float, decimals: int = 6) -> str:
    return f"{float(value):.{decimals}f}"


def _make_vtypes(rng: random.Random) -> list[dict[str, str]]:
    vtypes = []
    for i in range(4):
        vtypes.append(
            {
                "id": f"car_type_{i + 1}",
                "minGap": _format(rng.uniform(2.0, 4.0), 2),
                "maxSpeed": "13.89",
                "speedFactor": (
                    f"normc({_format(rng.uniform(0.6, 1.0), 2)},0.10,0.20,2.00)"
                ),
                "vClass": "passenger",
                "impatience": _format(rng.uniform(0.0, 1.0), 3),
                "lcCooperative": _format(rng.uniform(0.0, 1.0)),
                "lcSpeedGain": _format(rng.uniform(1.0, 2.0)),
                "lcImpatience": _format(rng.uniform(0.0, 1.0)),
                "jmIgnoreFoeProb": _format(rng.uniform(0.0, 1.0)),
                "accel": "2.6",
                "decel": "4.5",
                "sigma": _format(rng.uniform(0.3, 0.7)),
            }
        )
    return vtypes


def _generate_traffic_file(seed: int, destination: Path) -> int:
    rng = random.Random(seed)
    vtypes = _make_vtypes(rng)
    routes = ET.Element("routes")
    for vtype in vtypes:
        ET.SubElement(routes, "vType", vtype)

    vehicle_counter = 0
    departures_first_150 = 0
    for source, target, depart_lane, arrival_lane, vph in _FLOWS:
        rate_per_second = vph / 3600.0
        depart = rng.expovariate(rate_per_second)
        while depart < HORIZON_SECONDS:
            car_type = rng.choice(vtypes)
            vehicle = ET.SubElement(
                routes,
                "vehicle",
                {
                    "id": (
                        f"car_type_{car_type['id']}-flow-route-{source}-"
                        f"{depart_lane}-{target}-{arrival_lane}-{seed}-{vehicle_counter}"
                    ),
                    "type": car_type["id"],
                    "depart": _format(depart, 2),
                    "departLane": str(depart_lane),
                    "departPos": "random",
                    "departSpeed": "max",
                    "arrivalLane": str(arrival_lane),
                    "arrivalPos": "max",
                },
            )
            ET.SubElement(vehicle, "route", {"edges": f"{source} {target}"})
            vehicle_counter += 1
            if depart < 150.0:
                departures_first_150 += 1
            depart += rng.expovariate(rate_per_second)

    ET.indent(routes, space="    ")
    ET.ElementTree(routes).write(
        destination, encoding="utf-8", xml_declaration=True
    )
    return departures_first_150


def build() -> dict:
    traffic_dir = SCENARIO_DIR / "traffic"
    traffic_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(NETWORK_SRC, SCENARIO_DIR / "map.net.xml")
    (SCENARIO_DIR / "ego.rou.xml").write_text(EGO_ROUTE_XML, encoding="utf-8")
    (SCENARIO_DIR / "scenario_source.txt").write_text(
        SCENARIO_SOURCE_TXT, encoding="utf-8"
    )

    first_150 = []
    for seed in range(NUM_FILES):
        count = _generate_traffic_file(seed, traffic_dir / f"traffic_{seed}.rou.xml")
        first_150.append(count)
    summary = {
        "files": NUM_FILES,
        "first_150_seconds_mean": sum(first_150) / len(first_150),
        "first_150_seconds_min": min(first_150),
        "first_150_seconds_max": max(first_150),
        "total_veh_per_hour": sum(f[4] for f in _FLOWS),
    }
    (SCENARIO_DIR / "traffic" / "generation_manifest.json").write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = build()
    print(
        f"cross_left built: {summary['files']} traffic files, "
        f"first-150s departures mean={summary['first_150_seconds_mean']:.1f} "
        f"(min {summary['first_150_seconds_min']}, max {summary['first_150_seconds_max']}), "
        f"total demand {summary['total_veh_per_hour']} veh/h"
    )


if __name__ == "__main__":
    main()
