"""Generate the two new paper scenarios: ``merge`` and ``intersection``.

``merge``        -- single-lane highway + 45-degree on-ramp merge (urban-scaled
                   adaptation of TorchGRL ``flow.networks.MergeNetwork``).  The
                   FLOW multi-agent benchmark is converted to single-ego control:
                   one ``ego`` drives the through highway route while all other
                   vehicles are human background traffic.

``intersection`` -- DARRL's 4-way unsignalised intersection
                   (``Data/Intersection.net.xml``).  The ego makes an unprotected
                   left turn (``-E1 -> -E0``, depart t=50, lane 2) exactly as the
                   DARRL ``Auto`` vehicle; the four DARRL ``<flow>`` elements are
                   discretised into the framework's per-``traffic_<seed>.rou.xml``
                   convention.

Both scenes reuse the framework's background vType template
(``actor-car_type_N``: maxSpeed 55.5, speedFactor normc, accel 2.6, decel 4.5)
so their *speed settings* are byte-compatible with ``cross`` / ``carla`` /
``cross_left``.  Edge speed limits are lowered to 15 m/s (merge edges are also
shortened to urban scale) so the *observed* background speed stays urban
(~10-15 m/s), matching the user's instruction that merge road lengths be
adapted for the framework's urban speed regime.

Outputs (idempotent; safe to re-run):
    envs/sumo/original_scenarios_v1/merge/{map.net.xml, ego.rou.xml,
        merge.nod.xml, merge.edg.xml, traffic/traffic_00..29.rou.xml}
    envs/sumo/original_scenarios_v1/intersection/{map.net.xml, ego.rou.xml,
        traffic/traffic_00..29.rou.xml}
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_ROOT = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1"
SUMO_BIN = Path(r"D:\Program Files (x86)\Eclipse\Sumo\bin")
NETCONVERT = SUMO_BIN / "netconvert.exe"
DARRL_NET = Path(r"D:\Program Files (x86)\paper\DARRL-main\Data\Intersection.net.xml")

N_TRAFFIC_VARIANTS = 30
N_CAR_TYPES = 4

EGO_VTYPE = (
    '    <vType id="scene_rep_ego_type" accel="2.6" decel="4.5" '
    'emergencyDecel="8.0" sigma="0" length="4.7" minGap="2.0" '
    'maxSpeed="15.0" color="0,0.35,1"/>'
)


def _vtype_line(type_idx: int, random_id: int, rng: random.Random) -> str:
    """One framework-identical ``actor-car_type_N`` vType (cross template)."""
    min_gap = round(rng.uniform(1.9, 3.6), 2)
    speed_mean = round(rng.uniform(0.55, 0.99), 2)
    impatience = round(rng.uniform(0.0, 0.9), 2)
    lc_coop = round(rng.uniform(0.4, 1.0), 6)
    lc_gain = round(rng.uniform(1.0, 2.0), 6)
    lc_imp = round(rng.uniform(0.0, 0.9), 6)
    jm_ignore = round(rng.uniform(0.0, 0.9), 6)
    sigma = round(rng.uniform(0.1, 0.7), 6)
    return (
        f'    <vType id="actor-car_type_{type_idx}--{random_id}" '
        f'minGap="{min_gap}" maxSpeed="55.50" '
        f'speedFactor="normc({speed_mean:.2f},0.10,0.20,2.00)" '
        f'vClass="passenger" impatience="{impatience}" '
        f'lcCooperative="{lc_coop}" lcSpeedGain="{lc_gain}" '
        f'lcImpatience="{lc_imp}" jmIgnoreFoeProb="{jm_ignore}" '
        f'accel="2.6" decel="4.5" sigma="{sigma}"/>'
    )


def _vehicle_line(
    type_idx: int,
    random_id: int,
    route_edges: str,
    depart: float,
    seed: int,
    seq: int,
) -> str:
    vehicle_id = f"car_type_{type_idx}-flow-{seed}-{seq}"
    route_id = f"route_{seed}_{seq}"
    return (
        f'    <vehicle id="{vehicle_id}" type="actor-car_type_{type_idx}--{random_id}" '
        f'depart="{depart:.2f}" departLane="0" departPos="0" departSpeed="max" '
        f'arrivalLane="0" arrivalPos="max">\n'
        f'        <route edges="{route_edges}"/>\n'
        f'    </vehicle>'
    )


def _write_traffic(
    scenario_dir: Path,
    routes: list[tuple[str, float]],
    *,
    base_seed: int,
) -> None:
    """Write ``traffic/traffic_<seed>.rou.xml`` variants for one scenario.

    ``routes`` is a list of ``(edge_sequence, depart_headway_seconds)`` pairs.
    Vehicles depart from t=0 up to 600 s with a small seeded jitter so each
    variant is a distinct deterministic realisation.
    """
    traffic_dir = scenario_dir / "traffic"
    traffic_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(N_TRAFFIC_VARIANTS):
        rng = random.Random(base_seed + seed * 97 + 1)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<routes>",
        ]
        vtype_ids: dict[int, int] = {}
        for t in range(1, N_CAR_TYPES + 1):
            vtype_ids[t] = rng.randint(-10**18, 10**18)
            lines.append(_vtype_line(t, vtype_ids[t], rng))

        seq = 0
        horizon = 600.0
        for route_edges, headway in routes:
            t = 0.0
            while t < horizon:
                jitter = rng.uniform(-headway * 0.4, headway * 0.4)
                depart = max(0.0, t + jitter)
                type_idx = 1 + (seq % N_CAR_TYPES)
                lines.append(
                    _vehicle_line(
                        type_idx, vtype_ids[type_idx], route_edges, depart, seed, seq
                    )
                )
                seq += 1
                t += headway

        lines.append("</routes>")
        path = traffic_dir / f"traffic_{seed:02d}.rou.xml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ego(scenario_dir: Path, route_edges: str, depart: float,
               depart_lane: int, arrival_lane: int, depart_pos: int) -> None:
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<routes>\n"
        f"{EGO_VTYPE}\n"
        f'    <route id="scene_rep_ego_route" edges="{route_edges}"/>\n'
        f'    <vehicle id="ego" type="scene_rep_ego_type" '
        f'route="scene_rep_ego_route" depart="{depart}" '
        f'departLane="{depart_lane}" departPos="{depart_pos}" '
        f'departSpeed="0" arrivalLane="{arrival_lane}" arrivalPos="max"/>\n'
        "</routes>\n"
    )
    (scenario_dir / "ego.rou.xml").write_text(content, encoding="utf-8")


def build_merge() -> None:
    scenario_dir = SCENARIOS_ROOT / "merge"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "traffic").mkdir(parents=True, exist_ok=True)

    # Urban-scaled FLOW MergeNetwork: highway 80 + 150 + 80 m, ramp 80 + 60 m at 45 deg.
    angle = math.pi / 4.0
    merge_len, ramp_inflow_len = 60.0, 80.0
    premerge, postmerge = 150.0, 80.0
    inflow_highway_len = 80.0
    center_x = inflow_highway_len + premerge

    nodes = [
        ("inflow_highway", -inflow_highway_len, 0.0),
        ("left", 0.0, 0.0),
        ("center", center_x, 0.0),
        ("right", center_x + postmerge, 0.0),
        (
            "inflow_merge",
            center_x - (merge_len + ramp_inflow_len) * math.cos(angle),
            -(merge_len + ramp_inflow_len) * math.sin(angle),
        ),
        (
            "bottom",
            center_x - merge_len * math.cos(angle),
            -merge_len * math.sin(angle),
        ),
    ]
    nod_lines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<nodes>"]
    for node_id, x, y in nodes:
        radius = ' radius="8"' if node_id == "center" else ""
        nod_lines.append(f'    <node id="{node_id}" x="{x:.2f}" y="{y:.2f}"{radius}/>')
    nod_lines.append("</nodes>")

    # Highway is the priority road; the on-ramp yields to through traffic.
    # priority=5/3 mirrors the existing double_merge network's main/ramp split
    # so netconvert marks left->center major and bottom->center minor (a real
    # merge, not the right-before-left default that inverts the right-of-way).
    edges = [
        ("inflow_highway", "inflow_highway", "left", "5"),
        ("left", "left", "center", "5"),
        ("center", "center", "right", "5"),
        ("inflow_merge", "inflow_merge", "bottom", "3"),
        ("bottom", "bottom", "center", "3"),
    ]
    edg_lines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<edges>"]
    for edge_id, fr, to, priority in edges:
        edg_lines.append(
            f'    <edge id="{edge_id}" from="{fr}" to="{to}" '
            f'numLanes="1" speed="15.0" priority="{priority}"/>'
        )
    edg_lines.append("</edges>")

    nod_path = scenario_dir / "merge.nod.xml"
    edg_path = scenario_dir / "merge.edg.xml"
    net_path = scenario_dir / "map.net.xml"
    nod_path.write_text("\n".join(nod_lines) + "\n", encoding="utf-8")
    edg_path.write_text("\n".join(edg_lines) + "\n", encoding="utf-8")

    subprocess.run(
        [
            str(NETCONVERT),
            "--node-files", str(nod_path),
            "--edge-files", str(edg_path),
            "--output-file", str(net_path),
            "--no-turnarounds", "true",
        ],
        check=True,
        capture_output=True,
    )

    _write_ego(
        scenario_dir,
        route_edges="inflow_highway left center",
        depart=5,
        depart_lane=0,
        arrival_lane=0,
        depart_pos=10,
    )
    # highway ~0.25 veh/s, on-ramp ~0.05 veh/s
    _write_traffic(
        scenario_dir,
        routes=[
            ("inflow_highway left center", 4.0),
            ("inflow_merge bottom center", 20.0),
        ],
        base_seed=710_000,
    )


def _shorten_intersection_arms(net: str, arm_length: float = 70.0) -> str:
    """Shorten DARRL's four approach arms to ``arm_length`` m of lane length.

    DARRL stores each arm's far end as a dead_end junction at +/-500 m
    (east/west, priority-20 main road) and -200 m (south, priority-1 side
    road); the lane ``length`` attribute is that junction distance minus the
    13.60 m junction radius.  Moving each arm end to ``arm_length + 13.60`` m
    from the centre makes every external lane exactly ``arm_length`` m long,
    while junction J1 (unregulated), its 14 internal edges, all 42 connections
    and the request/foes tables stay byte-identical -- DARRL's
    unprotected-left-turn semantics are preserved.
    """
    radius = 13.60
    endpoint = arm_length + radius
    pos, neg = f"{endpoint:.2f}", f"-{endpoint:.2f}"
    # Order matters: replace the signed -500.00 before the unsigned 500.00 so
    # the former is never matched by the latter.
    net = net.replace("-500.00", neg)
    net = net.replace("500.00", pos)
    net = net.replace("-200.00", neg)
    net = net.replace('length="486.40"', f'length="{arm_length:.2f}"')
    net = net.replace('length="186.40"', f'length="{arm_length:.2f}"')
    return net


def build_intersection() -> None:
    scenario_dir = SCENARIOS_ROOT / "intersection"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "traffic").mkdir(parents=True, exist_ok=True)

    net_path = scenario_dir / "map.net.xml"
    source = DARRL_NET.read_text(encoding="utf-8")
    # Lower straight-through edge speed 30 -> 15 m/s so background traffic stays
    # in the framework's urban speed regime, then shorten the four approach arms
    # to 70 m lane length (geometry only; junction J1 / connections untouched).
    net = source.replace('speed="30.00"', 'speed="15.00"')
    net = _shorten_intersection_arms(net)
    net_path.write_text(net, encoding="utf-8")

    _write_ego(
        scenario_dir,
        route_edges="-E1 -E0",
        depart=50,
        depart_lane=2,
        arrival_lane=1,
        depart_pos=0,
    )
    # DARRL f_1/f_2/f_3 discretised, then densified so the unprotected left turn
    # (-E1 -> -E0) must yield to (a) oncoming through traffic E2 -> E1 (the
    # left turn's primary foe), (b) the priority westbound stream -E3 -> -E0,
    # and (c) the priority eastbound stream E0 -> E3.  Headways keep per-stream
    # gaps large enough that a yielding policy can still find a window to turn.
    _write_traffic(
        scenario_dir,
        routes=[
            ("E2 E1", 2.5),    # oncoming through (ego's primary conflict)
            ("-E3 -E0", 3.0),  # priority westbound (ego must yield)
            ("E0 E3", 4.0),    # priority eastbound (secondary)
        ],
        base_seed=820_000,
    )


def main() -> None:
    build_merge()
    build_intersection()
    print("merge + intersection assets generated under", SCENARIOS_ROOT)


if __name__ == "__main__":
    main()
