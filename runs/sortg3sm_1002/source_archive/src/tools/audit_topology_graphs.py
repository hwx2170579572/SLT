"""Audit every released SUMO map against the L1 topology capacity contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.topology_graph import (
    MAX_TOPO_EDGES,
    MAX_TOPO_NODES,
    build_topology_graph,
)


def audit() -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for name, specification in PAPER_SCENARIOS.items():
        _, info = build_topology_graph(
            specification.network_path,
            coordinate_offset=specification.coordinate_offset,
            return_info=True,
        )
        scenarios[name] = info.to_dict()
    maximum_nodes = max(
        int(row["valid_node_count"]) for row in scenarios.values()  # type: ignore[index, union-attr]
    )
    maximum_edges = max(
        int(row["valid_edge_count"]) for row in scenarios.values()  # type: ignore[index, union-attr]
    )
    return {
        "contract_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "declared_capacity": {
            "max_nodes": MAX_TOPO_NODES,
            "max_edges": MAX_TOPO_EDGES,
        },
        "observed_maximum": {
            "nodes": maximum_nodes,
            "edges": maximum_edges,
        },
        "matches_l1_reference_scan": maximum_nodes == 44 and maximum_edges == 160,
        "all_within_capacity": (
            maximum_nodes <= MAX_TOPO_NODES and maximum_edges <= MAX_TOPO_EDGES
        ),
        "scenarios": scenarios,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result = audit()
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_within_capacity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
