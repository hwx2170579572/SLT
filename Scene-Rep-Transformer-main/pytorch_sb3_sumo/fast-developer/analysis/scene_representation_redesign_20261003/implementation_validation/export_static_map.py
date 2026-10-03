from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


FD = Path(__file__).resolve().parents[3]
PR = FD.parent
if str(PR) not in sys.path:
    sys.path.insert(0, str(PR))
if str(FD) not in sys.path:
    sys.path.insert(0, str(FD))

from envs.sumo.paper_scenario_registry import get_paper_scenario_spec
from scene_event.collector import _enumerate_successor_paths
from scene_event.map_cache import PublicMapCache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path(__file__).resolve().parent
    specification = get_paper_scenario_spec("intersection_sorted")
    cache = PublicMapCache.from_spec(specification)
    vtype_rows = [vtype.attrib for route_path in specification.traffic_paths
                  for vtype in ET.parse(route_path).getroot().findall("vType")]
    explicit_length_count = sum("length" in row for row in vtype_rows)
    explicit_width_count = sum("width" in row for row in vtype_rows)
    explicit_both_count = sum("length" in row and "width" in row for row in vtype_rows)
    public_map = cache.public_map
    zone_vertices = public_map["zone_vertex_valid"].sum(axis=1)
    lane_count = len(cache.lane_ids)
    edge_count = public_map["edge_index"].shape[1]
    zone_count = len(public_map["zone_valid"])
    vertex_count = int(zone_vertices.max(initial=0))
    capacities = {"all_lanes": [], "ego_route_lanes": []}
    edge_index = public_map["edge_index"]
    edge_type = public_map["edge_type"]
    lateral_sources = {int(source) for (source, _target), rel in zip(edge_index.T, edge_type)
                       if int(rel) in (2, 3)}
    for lane_ptr in range(lane_count):
        for label, route_mask in (("all_lanes", None), ("ego_route_lanes", public_map["ego_route_lane_mask"])):
            if route_mask is not None and not bool(route_mask[lane_ptr]):
                continue
            paths, overflow = _enumerate_successor_paths(
                lane_ptr, cache.lane_successors, cache.lane_ids,
                max_path_lanes=16, max_enumerated=8192, route_mask=route_mask,
            )
            capacities[label].append({
                "lane": cache.lane_ids[lane_ptr],
                "path_count": len(paths),
                "incomplete_paths": sum(not complete for _path, complete in paths),
                "max_path_lanes": max((len(path) for path, _complete in paths), default=0),
                "overflow_unknown": bool(overflow),
                "has_unmodeled_lateral_alternative": bool(
                    {lane for path, _complete in paths for lane in path} & lateral_sources
                ),
            })
    npz_path = output / "intersection_sorted_public_map_corridor_v2.npz"
    json_path = output / "intersection_sorted_public_map_corridor_v2.json"
    np.savez_compressed(npz_path, **public_map)
    pair_audit = cache.build_metadata["zone_geometry"]["pair_relation_audit"]
    unsupported_critical = [
        item["pair_id"] for item in pair_audit
        if item["classification"] == "unsupported_public_corridor_geometry"
        and item["candidate_relevant"]
    ]
    no_zone_critical = [
        item["pair_id"] for item in pair_audit
        if not item["event_zone_created"] and item["candidate_relevant"]
    ]
    pair_audit_sha256 = hashlib.sha256(json.dumps(
        pair_audit, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    lane_index = {lane_id: index for index, lane_id in enumerate(cache.lane_ids)}
    unknown_classes = {
        "evidence_backed_foe_relation_without_corridor_overlap": "foe_without_zone",
        "evidence_backed_adjacent_exit_topology_only": "adjacent_exit_topology_only",
    }
    unknown_static_counts = {"foe_without_zone": 0, "adjacent_exit_topology_only": 0}
    endpoint_route_pair_counts = {"foe_without_zone": 0, "adjacent_exit_topology_only": 0}
    unknown_pair_ids = {"foe_without_zone": [], "adjacent_exit_topology_only": []}
    endpoint_route_pair_missing = []
    for item in pair_audit:
        kind = unknown_classes.get(item["classification"])
        if kind is None or item["event_zone_created"] or not item["candidate_relevant"]:
            continue
        unknown_static_counts[kind] += 1
        unknown_pair_ids[kind].append(item["pair_id"])
        left_ptr, right_ptr = (lane_index[lane_id] for lane_id in item["lane_pair"])
        left_paths, left_overflow = _enumerate_successor_paths(
            left_ptr, cache.lane_successors, cache.lane_ids,
            max_path_lanes=16, max_enumerated=8192, route_mask=None,
        )
        right_paths, right_overflow = _enumerate_successor_paths(
            right_ptr, cache.lane_successors, cache.lane_ids,
            max_path_lanes=16, max_enumerated=8192, route_mask=None,
        )
        matched_paths = sum(
            int(left_ptr in left_path and right_ptr in right_path)
            for left_path, _left_complete in left_paths
            for right_path, _right_complete in right_paths
        )
        endpoint_route_pair_counts[kind] += matched_paths
        if matched_paths == 0 or left_overflow or right_overflow:
            endpoint_route_pair_missing.append({
                "pair_id": item["pair_id"], "matched_candidate_path_pairs": matched_paths,
                "left_overflow": bool(left_overflow), "right_overflow": bool(right_overflow),
            })
    source_files = {
        "map_cache.py": Path(__file__).resolve().parents[3] / "scene_event" / "map_cache.py",
        "geometry.py": Path(__file__).resolve().parents[3] / "scene_event" / "geometry.py",
        "collector.py": Path(__file__).resolve().parents[3] / "scene_event" / "collector.py",
        "topology_graph.py": PR / "envs" / "sumo" / "topology_graph.py",
        "topology_graph_v2.py": PR / "envs" / "sumo" / "topology_graph_v2.py",
        "paper_scenario_registry.py": PR / "envs" / "sumo" / "paper_scenario_registry.py",
        "export_static_map.py": Path(__file__).resolve(),
    }
    report = {
        "geometry_audit_version": "corridor_v2_audit_v2",
        "scenario": "intersection_sorted_depart4p0",
        "network_path": cache.network_path,
        "ego_route_path": cache.ego_route_path,
        "network_sha256": cache.network_sha256,
        "public_map_fingerprint": cache.fingerprint,
        "L_lanes": lane_count,
        "E_typed_edges": edge_count,
        "Z_zones": zone_count,
        "Vmax_zone_vertices": vertex_count,
        "ego_route_lane_count": int(public_map["ego_route_lane_mask"].sum()),
        "zone_geometry": cache.build_metadata["zone_geometry"],
        "vehicle_dimensions_audit": {
            "traffic_template_count": len(specification.traffic_paths),
            "vtype_record_count": len(vtype_rows),
            "vtypes_with_explicit_length": explicit_length_count,
            "vtypes_with_explicit_width": explicit_width_count,
            "vtypes_with_both_explicit": explicit_both_count,
            "runtime_actor_dimension_source": "TraCI vehicle.getLength/getWidth per observed actor",
            "static_pairwise_footprint_clearance": "unknown; route-template vType declarations omit at least one required dimension",
            "scope": "event-zone OBB passage is computed per actor at runtime; this artifact does not certify pairwise vehicle collision clearance",
        },
        "typed_pair_coverage": {
            "pair_audit_schema": cache.build_metadata["zone_geometry"]["pair_relation_audit_schema"],
            "record_count": len(pair_audit),
            **cache.build_metadata["zone_geometry"]["pair_relation_audit_counts"],
            "unsupported_ego_route_reachable_pair_ids": unsupported_critical,
            "unsupported_candidate_relevant_pair_count": len(unsupported_critical),
            "no_event_zone_candidate_relevant_pair_ids": no_zone_critical,
            "no_event_zone_candidate_relevant_count": len(no_zone_critical),
            "candidate_relevance_definition": "both endpoint lanes have <=4 fully enumerated public successor paths; traffic-template participation was not used to filter legal topology candidates",
            "pair_audit_sha256": pair_audit_sha256,
            "sumo_foe_classification_scope": "exact areFoes request-index evidence is recorded when the topology edge came from the SUMO foe matrix; relation provenance is not a physical vehicle-contact oracle",
            "pair_records": pair_audit,
        },
        "candidate_pair_topology_status_contract": {
            "key": "candidate_pair_topology_status",
            "dtype": "uint8",
            "axes": ["actor_i", "candidate_i", "actor_j", "candidate_j"],
            "shape_rule": "[max_actors, max_candidates, max_actors, max_candidates]",
            "bit_codes": {
                "1": "foe_without_zone",
                "2": "adjacent_exit_topology_only",
            },
            "bit_code_semantics": {
                "1": "public topology relation only; event and vehicle footprint clearance are unknown",
                "2": "adjacent exit topology only; event and vehicle footprint clearance are unknown",
            },
            "map_fingerprint": cache.fingerprint,
            "pair_audit_sha256": pair_audit_sha256,
            "static_pair_counts": unknown_static_counts,
            "foe_without_zone_pair_ids": unknown_pair_ids["foe_without_zone"],
            "adjacent_exit_pair_ids": unknown_pair_ids["adjacent_exit_topology_only"],
            "static_endpoint_candidate_path_pair_counts": endpoint_route_pair_counts,
            "static_endpoint_candidate_path_pair_missing": endpoint_route_pair_missing,
            "unknown_relation_is_safety_label": False,
            "physical_clearance_known": False,
            "zero_semantics": "no audited topology-only pair matched these currently valid candidate routes; does not mean safe or interaction-free",
            "dynamic_route_pair_hits": 0,
            "static_only_no_actor_observations": True,
        },
        "candidate_route_coverage": {
            label: {
                "starting_lane_count": len(rows),
                "starts_with_more_than_four_paths": sum(row["path_count"] > 4 for row in rows),
                "starts_with_incomplete_prefix": sum(row["incomplete_paths"] > 0 for row in rows),
                "starts_with_enumeration_overflow": sum(row["overflow_unknown"] for row in rows),
                "starts_with_unmodeled_lateral_alternative": sum(
                    row["has_unmodeled_lateral_alternative"] for row in rows
                ),
                "maximum_enumerated_paths": max((row["path_count"] for row in rows), default=0),
            }
            for label, rows in capacities.items()
        },
        "candidate_routes_by_lane": capacities,
        "unmodeled_lateral_source_lane_count": len(lateral_sources),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256(npz_path),
        "geometry_source_sha256": {name: _sha256(path) for name, path in source_files.items()},
        "json_path": str(json_path.resolve()),
        "static_only_no_sumo_or_simulation": True,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "L_lanes", "E_typed_edges", "Z_zones", "Vmax_zone_vertices",
        "ego_route_lane_count", "candidate_route_coverage", "npz_path", "json_path",
    )}, indent=2))
    print(json.dumps({
        "typed_pair_class_counts": cache.build_metadata["zone_geometry"]["pair_relation_audit_counts"],
        "no_event_zone_relation_counts": cache.build_metadata["zone_geometry"]["unmapped_pairs_by_relation"],
        "npz_sha256": report["npz_sha256"],
        "pair_audit_sha256": pair_audit_sha256,
        "unsupported_candidate_relevant_pair_count": len(unsupported_critical),
        "no_event_zone_candidate_relevant_pair_count": len(no_zone_critical),
        "geometry_source_sha256": report["geometry_source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
