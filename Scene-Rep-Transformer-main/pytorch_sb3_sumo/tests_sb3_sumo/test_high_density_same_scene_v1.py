from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from envs.sumo.high_density_env_v1 import (
    HighDensityPaperSumoSceneEnvV1,
    HighDensityPaperSumoSceneEnvV4V1,
    build_high_density_overlay,
    inject_overlay_route_file,
)
from tools.high_density_same_scene_v1_common import (
    DEFAULT_PROTOCOL_PATH,
    build_jobs,
    load_protocol,
)


def _write_source(path: Path, body: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + body,
        encoding="utf-8",
    )


def test_explicit_vehicle_overlay_is_additive_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "traffic_7.rou.xml"
    overlay = tmp_path / "traffic_7__hdx1p5_v1.rou.xml"
    _write_source(
        source,
        """<routes>
    <vType id="car" maxSpeed="10"/>
    <vehicle id="v0" type="car" depart="0"><route edges="a b"/></vehicle>
    <vehicle id="v1" type="car" depart="10"><route edges="b c"/></vehicle>
    <vehicle id="v2" type="car" depart="20"><route edges="c d"/></vehicle>
    <vehicle id="v3" type="car" depart="30"><route edges="d e"/></vehicle>
</routes>""",
    )
    source_before = source.read_bytes()
    first = build_high_density_overlay(
        source,
        overlay,
        scenario="cross",
        vehicle_scale=1.5,
        pedestrian_scale=1.0,
        clone_depart_jitter_seconds=(1.0, 2.0),
    )
    second = build_high_density_overlay(
        source,
        overlay,
        scenario="cross",
        vehicle_scale=1.5,
        pedestrian_scale=1.0,
        clone_depart_jitter_seconds=(1.0, 2.0),
    )

    assert source.read_bytes() == source_before
    assert first == second
    assert first["base_explicit_vehicles"] == 4
    assert first["additional_explicit_vehicles"] == 2
    vehicles = ET.parse(overlay).getroot().findall("vehicle")
    assert len(vehicles) == 2
    assert len({vehicle.attrib["id"] for vehicle in vehicles}) == 2
    assert all("__hdv1_" in vehicle.attrib["id"] for vehicle in vehicles)
    assert [float(vehicle.attrib["depart"]) for vehicle in vehicles] == sorted(
        float(vehicle.attrib["depart"]) for vehicle in vehicles
    )
    assert all(vehicle.find("route") is not None for vehicle in vehicles)
    assert Path(f"{overlay}.manifest.json").is_file()


def test_flow_and_person_flow_overlay_add_only_requested_rate(tmp_path: Path) -> None:
    source = tmp_path / "traffic_0.rou.xml"
    overlay = tmp_path / "traffic_0__hdx1p35_v1.rou.xml"
    _write_source(
        source,
        """<routes>
    <vType id="car" maxSpeed="10"/>
    <vType id="ped" vClass="pedestrian"/>
    <route id="r" edges="a b"/>
    <vehicle id="v0" type="car" route="r" depart="0"/>
    <vehicle id="v1" type="car" route="r" depart="1"/>
    <vehicle id="v2" type="car" route="r" depart="2"/>
    <flow id="cars" type="car" route="r" begin="2" end="100" vehsPerHour="360"/>
    <personFlow id="walkers" type="ped" begin="0" end="100" period="12">
        <walk edges="a b"/>
    </personFlow>
</routes>""",
    )
    manifest = build_high_density_overlay(
        source,
        overlay,
        scenario="carla",
        vehicle_scale=1.35,
        pedestrian_scale=1.35,
        clone_depart_jitter_seconds=(0.8, 1.8),
    )
    root = ET.parse(overlay).getroot()
    flow = root.find("flow")
    person_flow = root.find("personFlow")

    assert manifest["additional_explicit_vehicles"] == 1
    assert flow is not None
    assert float(flow.attrib["vehsPerHour"]) == pytest.approx(126.0)
    assert person_flow is not None
    assert float(person_flow.attrib["period"]) == pytest.approx(12.0 / 0.35)
    assert person_flow.find("walk") is not None


def test_command_injection_preserves_source_and_ego_order(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.rou.xml"
    command = [
        "sumo",
        "--route-files",
        "source.rou.xml,ego.rou.xml",
        "--seed",
        "7",
    ]
    output = inject_overlay_route_file(command, overlay)
    assert command[2] == "source.rou.xml,ego.rou.xml"
    assert output[2] == f"source.rou.xml,{overlay},ego.rou.xml"


def test_baseline_and_v4_use_identical_frozen_80_20_partitions() -> None:
    paths = tuple(Path(f"traffic_{index}.rou.xml") for index in range(10))
    specification = SimpleNamespace(name="cross", traffic_paths=paths)

    baseline = HighDensityPaperSumoSceneEnvV1.__new__(
        HighDensityPaperSumoSceneEnvV1
    )
    baseline._connection = None
    baseline._high_density_partition = "evaluation"
    v4 = HighDensityPaperSumoSceneEnvV4V1.__new__(HighDensityPaperSumoSceneEnvV4V1)
    v4._connection = None
    v4._high_density_partition = "evaluation"
    expected_evaluation = (paths[0], paths[5])
    assert baseline._partitioned_traffic_paths(specification) == expected_evaluation
    assert v4._partitioned_traffic_paths(specification) == expected_evaluation

    baseline._high_density_partition = "train"
    v4._high_density_partition = "train"
    expected_training = tuple(path for path in paths if path not in expected_evaluation)
    assert baseline._partitioned_traffic_paths(specification) == expected_training
    assert v4._partitioned_traffic_paths(specification) == expected_training


def test_protocol_freezes_requested_matrix_and_three_selected_iterations() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL_PATH)
    assert protocol["matrix"]["method_order"] == [
        "mst_slt",
        "temporal_graph",
        "initial_full",
        "full_balanced",
        "v4_1",
        "v4_8",
    ]
    assert protocol["matrix"]["scenario_order"] == [
        "carla",
        "cross",
        "roundabout",
    ]
    assert [row["method"] for row in protocol["iteration_selection"]["selected"]] == [
        "v4_8",
        "full_balanced",
        "v4_1",
    ]
    assert [
        protocol["density"][scenario]["vehicle_scale"]
        for scenario in protocol["matrix"]["scenario_order"]
    ] == [1.35, 1.5, 1.25]
    assert protocol["density"]["cross"]["clone_depart_jitter_seconds"][0] > 5.0
    assert len(build_jobs(protocol, "smoke")) == 18
    assert len(build_jobs(protocol, "comparison")) == 54
    assert len({job.run_id for job in build_jobs(protocol, "comparison")}) == 54


def test_overlay_rejects_drift_in_existing_manifest(tmp_path: Path) -> None:
    source = tmp_path / "traffic_1.rou.xml"
    overlay = tmp_path / "overlay.rou.xml"
    _write_source(
        source,
        """<routes>
    <vType id="car"/>
    <vehicle id="v0" type="car" depart="0"><route edges="a b"/></vehicle>
</routes>""",
    )
    build_high_density_overlay(
        source,
        overlay,
        scenario="roundabout",
        vehicle_scale=1.25,
        pedestrian_scale=1.0,
    )
    manifest_path = Path(f"{overlay}.manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["vehicle_scale"] = 1.5
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest drifted"):
        build_high_density_overlay(
            source,
            overlay,
            scenario="roundabout",
            vehicle_scale=1.25,
            pedestrian_scale=1.0,
        )
