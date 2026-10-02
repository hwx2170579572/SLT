"""Generate the ``cross_left_unreg`` SUMO scenario.

A 4-way **uncontrolled** (unregulated) cross where the ego takes an unprotected
left turn, derived from ``cross_left`` but with three deliberate changes:

  1. Lane length is shortened 139.60 m -> 70 m (four arms).
  2. The centre junction is ``type="unregulated"`` (no right-of-way rules at
     all) instead of ``cross_left``'s ``priority``/yield junction.  This is the
     same junction model the project already uses for the DARRL
     ``intersection`` scenario.
  3. Every approach now carries all three turning behaviours -- straight, left,
     and right -- so social traffic *does* include left turners (``cross_left``
     deliberately had none; there the ego was the sole left turner).

Geometry (from ``envs/sumo/networks/cross_left_unreg/``):
  8 edges -- north/east/south/west ``_in``/``_out`` -- 2 driving lanes each
  (index 1 = right lane, index 2 = left lane; index 0 is a guessed sidewalk).
  Node ``C`` is at (0, 0); the four terminal nodes sit at +/- 80.4 m so each
  driving lane is exactly 70 m.

  Ego: ``south_in`` lane 2 -> ``west_out`` lane 2 (unprotected left turn),
  departing at t=15 from 20 m.

Density: 0.2 veh/s **per approach arm** (4 arms -> 0.8 veh/s = 2880 veh/h
total).  Each arm is split straight:left:right = 2:1:1, i.e. 16 flows of
0.05 veh/s (180 veh/h) each: 8 straight (2 lanes x 4 arms), 4 right (right
lane), 4 left (left lane).  This mirrors the DARRL ``intersection`` "0.2 veh/s
per flow" idiom, but with the total demand spread over the full 12 movements.
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "envs" / "sumo" / "original_scenarios_v1" / "cross_left_unreg"
NETWORK_SRC = (
    ROOT / "envs" / "sumo" / "networks" / "cross_left_unreg" / "cross_left_unreg.net.xml"
)
PLACEHOLDER_DIR = ROOT / "envs" / "sumo" / "scenarios" / "cross_left_unreg"

NUM_FILES = 60
HORIZON_SECONDS = 3600.0

# (source_edge, target_edge, depart_lane, arrival_lane, vehs_per_hour)
# 0.05 veh/s per flow = 180 veh/h; 0.2 veh/s per arm = straight(2 lanes)*0.05
# + left*0.05 + right*0.05.
_FLOWS: list[tuple[str, str, int, int, float]] = []
for _lane in (1, 2):
    _FLOWS.append(("north_in", "south_out", _lane, _lane, 180.0))  # straight N->S
    _FLOWS.append(("south_in", "north_out", _lane, _lane, 180.0))  # straight S->N
    _FLOWS.append(("east_in", "west_out", _lane, _lane, 180.0))   # straight E->W
    _FLOWS.append(("west_in", "east_out", _lane, _lane, 180.0))   # straight W->E
_FLOWS += [
    ("south_in", "east_out", 1, 1, 180.0),  # right S->E
    ("east_in", "north_out", 1, 1, 180.0),  # right E->N
    ("north_in", "west_out", 1, 1, 180.0),  # right N->W
    ("west_in", "south_out", 1, 1, 180.0),  # right W->S
    ("south_in", "west_out", 2, 2, 180.0),  # left S->W (ego's movement)
    ("east_in", "south_out", 2, 2, 180.0),  # left E->S
    ("north_in", "east_out", 2, 2, 180.0),  # left N->E
    ("west_in", "north_out", 2, 2, 180.0),  # left W->N
]

EGO_ROUTE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="scene_rep_ego_type" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0" length="4.7" minGap="2.0" maxSpeed="10.0" color="0,0.35,1"/>
    <route id="scene_rep_ego_route" edges="south_in west_out"/>
    <vehicle id="ego" type="scene_rep_ego_type" route="scene_rep_ego_route" depart="15" departLane="2" departPos="20" departSpeed="0" arrivalLane="2" arrivalPos="max"/>
</routes>
"""

SCENARIO_SOURCE_TXT = '''"""SMARTS sstudio reference for the 4-way uncontrolled unprotected-left-turn.

This scenario reuses the 4-arm network at
envs/sumo/networks/cross_left_unreg/ (unregulated junction, 70 m lanes) and
adds social traffic in the same idiom as the released left_turn/cross/cross_left
scenarios (SMARTS v0.4.17): per-lane flows with randomised TrafficActor
speed/min-gap/imperfection/lane-change/junction parameters, expanded to
explicit vehicles.  The ego takes an unprotected left turn (south_in ->
west_out).

Unlike cross_left (where the ego is the sole left turner), every approach here
carries straight, left, and right social flows, and the junction is
``unregulated`` (no right-of-way rules) rather than ``priority``.

The generator tools/gen_cross_left_unreg_scenario.py emits deterministic
explicit <vehicle> elements directly (seeded), equivalent to the SMARTS
gen_traffic -> duarouter pipeline, so the additive high-density overlay clones
vehicles as it does for the released scenarios.
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

# straight (2 lanes) + right-turn + left-turn flows on all four arms.
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
left_turn = {
    "south": ("south_in", "west_out"),
    "east": ("east_in", "south_out"),
    "north": ("north_in", "east_out"),
    "west": ("west_in", "north_out"),
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
                       rate=180, actors=actors) for lane in (1, 2)]
    for (src, dst) in right_turn.values():
        flows += [Flow(route=Route(begin=(src, 1, "random"), end=(dst, 1, "max")),
                       rate=180, actors=actors)]
    for (src, dst) in left_turn.values():
        flows += [Flow(route=Route(begin=(src, 2, "random"), end=(dst, 2, "max")),
                       rate=180, actors=actors)]

    gen_traffic(scenario, Traffic(flows=flows), seed=seed, name=f"traffic_{seed}")

# ego: approach from the south arm and take an unprotected left turn to the west.
gen_missions(scenario=scenario, missions=[
    Mission(Route(begin=("south_in", 2, 20), end=("west_out", 2, "max")), start_time=15),
])
'''


PLACEHOLDER_SUMOCFG = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="../../networks/cross_left_unreg/cross_left_unreg.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="6000"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
        <verbose value="false"/>
    </report>
</configuration>
"""

PLACEHOLDER_ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="ego_type" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0" length="4.7" minGap="2.0" maxSpeed="10.0" color="0,0.35,1"/>
    <vType id="traffic" accel="2.4" decel="4.5" sigma="0.4" length="4.7" minGap="2.5" maxSpeed="13.89" speedFactor="normc(0.9,0.1,0.7,1.1)"/>
    <route id="ego_route" edges="south_in west_out"/>
    <route id="north_south" edges="north_in south_out"/>
    <route id="south_north" edges="south_in north_out"/>
    <route id="east_west" edges="east_in west_out"/>
    <route id="west_east" edges="west_in east_out"/>
    <route id="south_east" edges="south_in east_out"/>
    <route id="east_north" edges="east_in north_out"/>
    <route id="north_west" edges="north_in west_out"/>
    <route id="west_south" edges="west_in south_out"/>
    <route id="south_west" edges="south_in west_out"/>
    <route id="east_south" edges="east_in south_out"/>
    <route id="north_east" edges="north_in east_out"/>
    <route id="west_north" edges="west_in north_out"/>
    <vehicle id="ego" type="ego_type" route="ego_route" depart="15" departLane="2" departPos="20" departSpeed="0"/>
    <flow id="north_south_flow" type="traffic" route="north_south" begin="0" end="6000" vehsPerHour="360" departLane="random" departSpeed="random"/>
    <flow id="south_north_flow" type="traffic" route="south_north" begin="0" end="6000" vehsPerHour="360" departLane="random" departSpeed="random"/>
    <flow id="east_west_flow" type="traffic" route="east_west" begin="0" end="6000" vehsPerHour="360" departLane="random" departSpeed="random"/>
    <flow id="west_east_flow" type="traffic" route="west_east" begin="0" end="6000" vehsPerHour="360" departLane="random" departSpeed="random"/>
    <flow id="south_east_flow" type="traffic" route="south_east" begin="0" end="6000" vehsPerHour="180" departLane="1" departSpeed="random"/>
    <flow id="east_north_flow" type="traffic" route="east_north" begin="0" end="6000" vehsPerHour="180" departLane="1" departSpeed="random"/>
    <flow id="north_west_flow" type="traffic" route="north_west" begin="0" end="6000" vehsPerHour="180" departLane="1" departSpeed="random"/>
    <flow id="west_south_flow" type="traffic" route="west_south" begin="0" end="6000" vehsPerHour="180" departLane="1" departSpeed="random"/>
    <flow id="south_west_flow" type="traffic" route="south_west" begin="0" end="6000" vehsPerHour="180" departLane="2" departSpeed="random"/>
    <flow id="east_south_flow" type="traffic" route="east_south" begin="0" end="6000" vehsPerHour="180" departLane="2" departSpeed="random"/>
    <flow id="north_east_flow" type="traffic" route="north_east" begin="0" end="6000" vehsPerHour="180" departLane="2" departSpeed="random"/>
    <flow id="west_north_flow" type="traffic" route="west_north" begin="0" end="6000" vehsPerHour="180" departLane="2" departSpeed="random"/>
</routes>
"""


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

    PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)
    (PLACEHOLDER_DIR / "scenario.sumocfg").write_text(
        PLACEHOLDER_SUMOCFG, encoding="utf-8"
    )
    (PLACEHOLDER_DIR / "routes.rou.xml").write_text(
        PLACEHOLDER_ROUTES, encoding="utf-8"
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
        "veh_per_second_total": sum(f[4] for f in _FLOWS) / 3600.0,
        "veh_per_second_per_arm": 0.2,
        "manoeuvre_split": "straight:left:right = 2:1:1 per arm",
        "lane_length_m": 70.0,
        "junction_type": "unregulated",
    }
    (SCENARIO_DIR / "traffic" / "generation_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = build()
    print(
        f"cross_left_unreg built: {summary['files']} traffic files, "
        f"first-150s departures mean={summary['first_150_seconds_mean']:.1f} "
        f"(min {summary['first_150_seconds_min']}, max {summary['first_150_seconds_max']}), "
        f"total demand {summary['total_veh_per_hour']} veh/h "
        f"({summary['veh_per_second_total']} veh/s total, "
        f"{summary['veh_per_second_per_arm']} veh/s per arm)"
    )


if __name__ == "__main__":
    main()
