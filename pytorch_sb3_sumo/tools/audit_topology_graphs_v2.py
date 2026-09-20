"""Audit merge-aware topology graphs on all six paper scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS, get_paper_scenario_spec
from envs.sumo.topology_graph import build_topology_graph
from envs.sumo.topology_graph_v2 import (
    MERGE,
    RELATION_NAMES,
    build_topology_graph_v2,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed_edges(graph: Any) -> set[tuple[int, int, int]]:
    valid = np.asarray(graph.edge_mask, dtype=bool)
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in zip(
            np.asarray(graph.edge_index)[0, valid],
            np.asarray(graph.edge_index)[1, valid],
            np.asarray(graph.edge_type)[valid],
        )
    }


def audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for scenario in PAPER_SCENARIOS:
        specification = get_paper_scenario_spec(scenario)
        offset = tuple(specification.coordinate_offset)
        v1, v1_info = build_topology_graph(
            specification.network_path,
            coordinate_offset=offset,
            return_info=True,
        )
        v2, v2_info = build_topology_graph_v2(
            specification.network_path,
            coordinate_offset=offset,
            return_info=True,
        )
        repeated, repeated_info = build_topology_graph_v2(
            specification.network_path,
            coordinate_offset=offset,
            return_info=True,
        )

        geometry_preserved = bool(
            np.array_equal(v1.lane_points, v2.lane_points)
            and np.array_equal(v1.lane_attrs, v2.lane_attrs)
            and np.array_equal(v1.node_mask, v2.node_mask)
        )
        v1_edges = _typed_edges(v1)
        v2_edges = _typed_edges(v2)
        base_edges_preserved = v1_edges.issubset(v2_edges)
        merge_edges = {edge for edge in v2_edges if edge[2] == MERGE}
        merge_symmetric = all((target, source, MERGE) in merge_edges for source, target, _ in merge_edges)
        deterministic = bool(
            v2_info.fingerprint == repeated_info.fingerprint
            and np.array_equal(v2.edge_index, repeated.edge_index)
            and np.array_equal(v2.edge_type, repeated.edge_type)
            and np.array_equal(v2.edge_mask, repeated.edge_mask)
        )
        valid_nodes = int(v2_info.valid_node_count)
        possible_directed = max(valid_nodes * max(valid_nodes - 1, 0), 1)
        merge_density = len(merge_edges) / possible_directed
        # This is a corruption guard, not a scientific threshold: a static
        # merge parser should never make one quarter of every ordered lane pair
        # mutually related on the released maps.
        not_near_complete = merge_density < 0.25
        row = {
            "scenario": scenario,
            "network_path": str(specification.network_path.resolve()),
            "network_sha256": _sha256(specification.network_path),
            "coordinate_offset": list(offset),
            "nodes": valid_nodes,
            "v1_edges": int(v1_info.valid_edge_count),
            "v2_edges": int(v2_info.valid_edge_count),
            "relation_counts": v2_info.relation_counts,
            "merge_groups": {
                target: list(sources)
                for target, sources in sorted(v2_info.merge_groups.items())
            },
            "merge_density": merge_density,
            "geometry_preserved": geometry_preserved,
            "base_edges_preserved": base_edges_preserved,
            "merge_symmetric": merge_symmetric,
            "deterministic": deterministic,
            "within_capacity": bool(
                v2.valid_node_count <= len(v2.node_mask)
                and v2.valid_edge_count <= len(v2.edge_mask)
            ),
            "not_near_complete": not_near_complete,
            "v1_fingerprint": v1_info.fingerprint,
            "v2_fingerprint": v2_info.fingerprint,
        }
        if scenario == "cross" and not merge_edges:
            failures.append("cross has no MERGE relation")
        for check in (
            "geometry_preserved",
            "base_edges_preserved",
            "merge_symmetric",
            "deterministic",
            "within_capacity",
            "not_near_complete",
        ):
            if not row[check]:
                failures.append(f"{scenario}: {check} failed")
        rows.append(row)

    return {
        "schema_version": "topology-graph-v2-audit/v1",
        "passed": not failures,
        "relation_names": list(RELATION_NAMES),
        "scenario_count": len(rows),
        "failures": failures,
        "scenarios": rows,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    payload = audit()
    if args.output is not None:
        path = args.output.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
