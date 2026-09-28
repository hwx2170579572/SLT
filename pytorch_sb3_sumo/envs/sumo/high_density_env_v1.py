"""Same-map high-density traffic overlays for stress comparisons.

The released networks, ego routes, observations, actions, rewards, and episode
limits remain untouched.  Only additional social traffic demand is supplied as
an extra SUMO route file.  Keeping the overlay separate makes the stress setup
auditable and prevents changes to the frozen source assets.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from .paper_env import PaperSumoSceneEnv
from .paper_env_v4 import PaperSumoSceneEnvV4
from .paper_scenario_registry import PaperScenarioSpec


HIGH_DENSITY_ENV_VERSION = "same-scene-high-density-v1"
HIGH_DENSITY_PARTITIONS = ("all", "train", "evaluation")
# The overlay mechanism is map-agnostic: it clones only social-demand entries
# and leaves the selected network and ego route untouched.  The first study
# exercised three scenarios; the independent-v2 six-scenario extension uses
# the same audited mechanism for the remaining released scenarios as well.
SUPPORTED_HIGH_DENSITY_SCENARIOS = (
    "left_turn",
    "cross",
    "cross_left",
    "cross_left_unreg",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
    # New SUMO scenarios (this project).
    "merge",
    "intersection",
    # intersection 副本：traffic 按 depart 排序（三股车流都参与仿真）。
    "intersection_sorted",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_number(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _extra_count(base_count: int, scale: float) -> int:
    return int(math.floor(base_count * (float(scale) - 1.0) + 0.5))


def _selected_clone_indices(base_count: int, extra_count: int) -> list[int]:
    """Choose clone sources evenly without depending on Python hash state."""

    if base_count <= 0 or extra_count <= 0:
        return []
    return [
        min(base_count - 1, int(math.floor((index + 0.5) * base_count / extra_count)))
        for index in range(extra_count)
    ]


def _stable_jitter(
    element_id: str,
    clone_ordinal: int,
    jitter_range_seconds: tuple[float, float],
) -> float:
    low, high = (float(value) for value in jitter_range_seconds)
    if low < 0.0 or high < low:
        raise ValueError("clone jitter must satisfy 0 <= low <= high")
    if high == low:
        return low
    token = f"{element_id}|{clone_ordinal}|{HIGH_DENSITY_ENV_VERSION}"
    value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    fraction = value / float(0xFFFFFFFF)
    return low + fraction * (high - low)


def _clone_departure_element(
    source: ET.Element,
    *,
    clone_ordinal: int,
    jitter_range_seconds: tuple[float, float],
) -> ET.Element:
    clone = copy.deepcopy(source)
    source_id = source.attrib.get("id")
    if not source_id:
        raise ValueError(f"{source.tag} is missing an id")
    depart_text = source.attrib.get("depart")
    if depart_text is None:
        raise ValueError(f"{source.tag} {source_id!r} is missing depart")
    try:
        depart = float(depart_text)
    except ValueError as exc:
        raise ValueError(
            f"{source.tag} {source_id!r} has non-numeric depart={depart_text!r}"
        ) from exc
    jitter = _stable_jitter(source_id, clone_ordinal, jitter_range_seconds)
    clone.attrib["id"] = f"{source_id}__hdv1_{clone_ordinal:04d}"
    clone.attrib["depart"] = _format_number(depart + jitter)
    return clone


def _scaled_flow_overlay(
    source: ET.Element,
    *,
    extra_factor: float,
    clone_ordinal: int,
) -> ET.Element | None:
    if extra_factor <= 0.0:
        return None
    clone = copy.deepcopy(source)
    source_id = source.attrib.get("id")
    if not source_id:
        raise ValueError(f"{source.tag} is missing an id")
    clone.attrib["id"] = f"{source_id}__hdv1_extra_{clone_ordinal:04d}"

    if "vehsPerHour" in clone.attrib:
        base_rate = float(clone.attrib["vehsPerHour"])
        clone.attrib["vehsPerHour"] = _format_number(base_rate * extra_factor)
    elif "personsPerHour" in clone.attrib:
        base_rate = float(clone.attrib["personsPerHour"])
        clone.attrib["personsPerHour"] = _format_number(base_rate * extra_factor)
    elif "period" in clone.attrib:
        base_period = float(clone.attrib["period"])
        clone.attrib["period"] = _format_number(base_period / extra_factor)
    elif "probability" in clone.attrib:
        base_probability = float(clone.attrib["probability"])
        extra_probability = base_probability * extra_factor
        if extra_probability > 1.0:
            raise ValueError(
                f"flow {source_id!r} requires probability {extra_probability} > 1"
            )
        clone.attrib["probability"] = _format_number(extra_probability)
    elif "number" in clone.attrib:
        base_number = int(clone.attrib["number"])
        clone.attrib["number"] = str(_extra_count(base_number, 1.0 + extra_factor))
    else:
        raise ValueError(
            f"flow {source_id!r} has no supported demand attribute "
            "(vehsPerHour/personsPerHour/period/probability/number)"
        )
    return clone


def _manifest_path(overlay_path: Path) -> Path:
    return overlay_path.with_name(f"{overlay_path.name}.manifest.json")


def _load_matching_manifest(
    overlay_path: Path,
    *,
    source_sha256: str,
    vehicle_scale: float,
    pedestrian_scale: float,
    jitter_range_seconds: tuple[float, float],
) -> dict[str, Any] | None:
    manifest_path = _manifest_path(overlay_path)
    if not overlay_path.is_file() or not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "same-scene-high-density-overlay/v1",
        "source_sha256": source_sha256,
        "vehicle_scale": float(vehicle_scale),
        "pedestrian_scale": float(pedestrian_scale),
        "clone_depart_jitter_seconds": [
            float(jitter_range_seconds[0]),
            float(jitter_range_seconds[1]),
        ],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"existing high-density overlay manifest drifted at {key!r}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    if payload.get("overlay_sha256") != _sha256(overlay_path):
        raise ValueError(f"high-density overlay hash mismatch: {overlay_path}")
    return payload


def build_high_density_overlay(
    source_path: str | os.PathLike[str],
    overlay_path: str | os.PathLike[str],
    *,
    scenario: str,
    vehicle_scale: float,
    pedestrian_scale: float | None = None,
    clone_depart_jitter_seconds: tuple[float, float] = (1.0, 2.0),
) -> dict[str, Any]:
    """Create an additive route file that realizes the requested demand scale."""

    source = Path(source_path).resolve()
    destination = Path(overlay_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if scenario not in SUPPORTED_HIGH_DENSITY_SCENARIOS:
        raise ValueError(
            f"high-density scenario must be one of {SUPPORTED_HIGH_DENSITY_SCENARIOS}"
        )
    vehicle_scale = float(vehicle_scale)
    pedestrian_scale = (
        vehicle_scale if pedestrian_scale is None else float(pedestrian_scale)
    )
    # ``vehicle_scale == 1.0`` is a valid no-op: ``_extra_count(..., 1.0)`` and
    # the flow extra-factor both evaluate to zero, so the overlay is an empty
    # additive file and the scenario runs at its base demand.  This is used for
    # scenarios whose base traffic *already is* the experiment density (e.g.
    # ``cross_left_unreg`` at 0.2 veh/s per approach), where any overlay would
    # push an already-saturated uncontrolled junction into gridlock.
    if not 1.0 <= vehicle_scale <= 2.0:
        raise ValueError("vehicle_scale must be in [1, 2]")
    if not 1.0 <= pedestrian_scale <= 2.0:
        raise ValueError("pedestrian_scale must be in [1, 2]")
    jitter = tuple(float(value) for value in clone_depart_jitter_seconds)
    if len(jitter) != 2:
        raise ValueError("clone_depart_jitter_seconds must contain two values")

    source_hash = _sha256(source)
    existing = _load_matching_manifest(
        destination,
        source_sha256=source_hash,
        vehicle_scale=vehicle_scale,
        pedestrian_scale=pedestrian_scale,
        jitter_range_seconds=(jitter[0], jitter[1]),
    )
    if existing is not None:
        return existing

    root = ET.parse(source).getroot()
    if root.tag.split("}")[-1] != "routes":
        raise ValueError(f"expected SUMO <routes> root in {source}")

    vehicles = list(root.findall("vehicle"))
    persons = list(root.findall("person"))
    vehicle_flows = list(root.findall("flow"))
    person_flows = list(root.findall("personFlow"))
    extra_vehicle_count = _extra_count(len(vehicles), vehicle_scale)
    extra_person_count = _extra_count(len(persons), pedestrian_scale)

    overlay_root = ET.Element("routes")
    overlay_root.append(
        ET.Comment(
            " additive social-demand overlay; original traffic file must be loaded first "
        )
    )
    departures: list[tuple[float, ET.Element]] = []
    for ordinal, source_index in enumerate(
        _selected_clone_indices(len(vehicles), extra_vehicle_count)
    ):
        clone = _clone_departure_element(
            vehicles[source_index],
            clone_ordinal=ordinal,
            jitter_range_seconds=(jitter[0], jitter[1]),
        )
        departures.append((float(clone.attrib["depart"]), clone))
    for ordinal, source_index in enumerate(
        _selected_clone_indices(len(persons), extra_person_count)
    ):
        clone = _clone_departure_element(
            persons[source_index],
            clone_ordinal=ordinal,
            jitter_range_seconds=(jitter[0], jitter[1]),
        )
        departures.append((float(clone.attrib["depart"]), clone))
    for _, element in sorted(
        departures, key=lambda item: (item[0], item[1].attrib.get("id", ""))
    ):
        overlay_root.append(element)

    additional_vehicle_flows = 0
    for ordinal, flow in enumerate(vehicle_flows):
        clone = _scaled_flow_overlay(
            flow,
            extra_factor=vehicle_scale - 1.0,
            clone_ordinal=ordinal,
        )
        if clone is not None:
            overlay_root.append(clone)
            additional_vehicle_flows += 1

    additional_person_flows = 0
    for ordinal, flow in enumerate(person_flows):
        clone = _scaled_flow_overlay(
            flow,
            extra_factor=pedestrian_scale - 1.0,
            clone_ordinal=ordinal,
        )
        if clone is not None:
            overlay_root.append(clone)
            additional_person_flows += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(overlay_root, space="    ")
    # Keep the temporary basename short.  The project commonly lives below a
    # long Windows path and repeating the full destination basename can push
    # an otherwise valid overlay beyond the legacy MAX_PATH boundary.
    temporary = destination.with_name(f".o{os.getpid()}.tmp")
    ET.ElementTree(overlay_root).write(
        temporary, encoding="utf-8", xml_declaration=True
    )
    os.replace(temporary, destination)

    manifest: dict[str, Any] = {
        "schema_version": "same-scene-high-density-overlay/v1",
        "environment_version": HIGH_DENSITY_ENV_VERSION,
        "scenario": scenario,
        "source_path": str(source),
        "source_sha256": source_hash,
        "overlay_path": str(destination),
        "overlay_sha256": _sha256(destination),
        "vehicle_scale": vehicle_scale,
        "pedestrian_scale": pedestrian_scale,
        "clone_depart_jitter_seconds": [jitter[0], jitter[1]],
        "base_explicit_vehicles": len(vehicles),
        "additional_explicit_vehicles": extra_vehicle_count,
        "base_vehicle_flows": len(vehicle_flows),
        "additional_vehicle_flows": additional_vehicle_flows,
        "base_explicit_persons": len(persons),
        "additional_explicit_persons": extra_person_count,
        "base_person_flows": len(person_flows),
        "additional_person_flows": additional_person_flows,
        "original_assets_modified": False,
        "map_or_ego_route_changed": False,
    }
    manifest_path = _manifest_path(destination)
    temporary_manifest = manifest_path.with_name(f".m{os.getpid()}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def inject_overlay_route_file(command: Iterable[str], overlay_path: Path) -> list[str]:
    """Insert an additive traffic file between source traffic and the ego route."""

    output = list(command)
    try:
        index = output.index("--route-files") + 1
    except ValueError as exc:
        raise ValueError("SUMO command has no --route-files option") from exc
    if index >= len(output):
        raise ValueError("SUMO command has an empty --route-files option")
    route_files = output[index].split(",")
    if len(route_files) != 2:
        raise ValueError(
            "same-scene overlay expects source traffic and ego route files exactly"
        )
    output[index] = ",".join((route_files[0], str(overlay_path), route_files[1]))
    return output


class _HighDensityPaperEnvMixin:
    def __init__(
        self,
        *args: Any,
        scenario: str,
        high_density_vehicle_scale: float,
        high_density_overlay_root: str | os.PathLike[str],
        high_density_pedestrian_scale: float | None = None,
        high_density_partition: str = "all",
        high_density_clone_jitter_seconds: tuple[float, float] = (1.0, 2.0),
        **kwargs: Any,
    ) -> None:
        if scenario not in SUPPORTED_HIGH_DENSITY_SCENARIOS:
            raise ValueError(
                f"scenario must be one of {SUPPORTED_HIGH_DENSITY_SCENARIOS}"
            )
        if high_density_partition not in HIGH_DENSITY_PARTITIONS:
            raise ValueError(
                f"high_density_partition must be one of {HIGH_DENSITY_PARTITIONS}"
            )
        self._high_density_vehicle_scale = float(high_density_vehicle_scale)
        self._high_density_pedestrian_scale = float(
            high_density_vehicle_scale
            if high_density_pedestrian_scale is None
            else high_density_pedestrian_scale
        )
        self._high_density_overlay_root = Path(high_density_overlay_root).resolve()
        self._high_density_partition = str(high_density_partition)
        self._high_density_clone_jitter_seconds = tuple(
            float(value) for value in high_density_clone_jitter_seconds
        )
        self._selected_high_density_overlay: Path | None = None
        self._selected_high_density_manifest: dict[str, Any] | None = None

        # Both parent families accept ``all``.  The mixin installs one matched
        # 80/20 split after construction so baseline and v4 methods see the
        # exact same route-file partitions.
        kwargs["traffic_partition"] = "all"
        super().__init__(*args, scenario=scenario, **kwargs)
        self._traffic_partition = self._high_density_partition

    def _partitioned_traffic_paths(
        self, specification: PaperScenarioSpec
    ) -> tuple[Path, ...]:
        paths = specification.traffic_paths
        if self._high_density_partition == "all" or len(paths) < 2:
            return paths
        evaluation = tuple(
            path for index, path in enumerate(paths) if index % 5 == 0
        )
        selected = (
            evaluation
            if self._high_density_partition == "evaluation"
            else tuple(path for path in paths if path not in set(evaluation))
        )
        if not selected:
            raise RuntimeError(
                f"empty high-density {self._high_density_partition} partition "
                f"for {specification.name}"
            )
        return selected

    def _overlay_path(self, source_path: Path) -> Path:
        scale_token = _format_number(self._high_density_vehicle_scale).replace(
            ".", "p"
        )
        return (
            self._high_density_overlay_root
            / self.scenario
            / f"{source_path.stem}__hdx{scale_token}_v1.rou.xml"
        )

    def _sumo_command(self, seed: int) -> list[str]:
        command = super()._sumo_command(seed)
        source = self._selected_traffic_path
        if source is None:
            raise RuntimeError("parent environment did not select a traffic route file")
        overlay = self._overlay_path(source)
        manifest = build_high_density_overlay(
            source,
            overlay,
            scenario=self.scenario,
            vehicle_scale=self._high_density_vehicle_scale,
            pedestrian_scale=self._high_density_pedestrian_scale,
            clone_depart_jitter_seconds=self._high_density_clone_jitter_seconds,
        )
        self._selected_high_density_overlay = overlay
        self._selected_high_density_manifest = manifest
        return inject_overlay_route_file(command, overlay)

    def _info_dict(self, **extra: Any) -> dict[str, Any]:
        info = super()._info_dict(**extra)
        info.update(
            {
                "high_density_environment_version": HIGH_DENSITY_ENV_VERSION,
                "high_density_same_map": True,
                "high_density_vehicle_scale": self._high_density_vehicle_scale,
                "high_density_pedestrian_scale": self._high_density_pedestrian_scale,
                "high_density_partition": self._high_density_partition,
                "high_density_original_assets_modified": False,
            }
        )
        if self._selected_high_density_overlay is not None:
            info["high_density_overlay"] = self._selected_high_density_overlay.name
        if self._selected_high_density_manifest is not None:
            info["high_density_overlay_sha256"] = self._selected_high_density_manifest[
                "overlay_sha256"
            ]
            info["high_density_additional_explicit_vehicles"] = (
                self._selected_high_density_manifest[
                    "additional_explicit_vehicles"
                ]
            )
        return info


class HighDensityPaperSumoSceneEnvV1(
    _HighDensityPaperEnvMixin, PaperSumoSceneEnv
):
    """High-density view for continuous-action baseline/full methods."""


class HighDensityPaperSumoSceneEnvV4V1(
    _HighDensityPaperEnvMixin, PaperSumoSceneEnvV4
):
    """High-density view with the decision-aligned v4 observation contract."""


__all__ = [
    "HIGH_DENSITY_ENV_VERSION",
    "HIGH_DENSITY_PARTITIONS",
    "SUPPORTED_HIGH_DENSITY_SCENARIOS",
    "HighDensityPaperSumoSceneEnvV1",
    "HighDensityPaperSumoSceneEnvV4V1",
    "build_high_density_overlay",
    "inject_overlay_route_file",
]
