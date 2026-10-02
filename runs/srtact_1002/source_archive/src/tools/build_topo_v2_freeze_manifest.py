"""Freeze v2 source files and the exact 60/20/20 traffic asset assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS


SOURCE_PATHS = (
    "envs/sumo/topology_graph.py",
    "algos/sb3_torch/topo_temporal_features.py",
    "algos/sb3_torch/sac.py",
    "configs/sb3_configs.py",
    "envs/sumo/topology_graph_v2.py",
    "envs/sumo/paper_env_v2.py",
    "algos/sb3_torch/topo_temporal_features_v2.py",
    "algos/sb3_torch/sac_v2.py",
    "configs/sb3_configs_v2.py",
    "tools/train_sb3_v2.py",
    "tools/train_paper_sb3_sumo_v2.py",
    "tools/paper_evaluation_contract_v2.py",
    "tools/action_diagnostics_v2.py",
    "tools/topo_v2_statistics.py",
    "tools/run_topo_v2_experiments.py",
    "tools/audit_topology_graphs_v2.py",
    "tools/build_topo_v2_freeze_manifest.py",
    "tools/benchmark_topo_v2_workers.py",
    "tools/capture_topo_v2_environment.py",
    "experiments/topo_scene_v2/experiment_contract.yaml",
    "tests_sb3_sumo/test_topology_graph_v2.py",
    "tests_sb3_sumo/test_topo_temporal_v2.py",
    "tests_sb3_sumo/test_paper_env_v2.py",
    "tests_sb3_sumo/test_action_diagnostics_v2.py",
    "tests_sb3_sumo/test_topo_v2_experiments.py",
    "tests_sb3_sumo/test_topo_v2_statistics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partition(index: int, count: int) -> str:
    if count < 3:
        return "shared_non_disjoint"
    residue = index % 5
    if residue == 0:
        return "validation"
    if residue == 1:
        return "test"
    return "train"


def build_manifest() -> dict[str, Any]:
    source_files = []
    for relative in SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    scenarios = []
    for name, specification in PAPER_SCENARIOS.items():
        traffic = []
        paths = specification.traffic_paths
        for index, path in enumerate(paths):
            traffic.append(
                {
                    "index": index,
                    "filename": path.name,
                    "partition": _partition(index, len(paths)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        counts: dict[str, int] = {}
        for row in traffic:
            partition = str(row["partition"])
            counts[partition] = counts.get(partition, 0) + 1
        scenarios.append(
            {
                "scenario": name,
                "released_assets": specification.released_assets,
                "network": {
                    "path": str(specification.network_path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(specification.network_path),
                },
                "ego_route": {
                    "path": str(specification.ego_route_path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(specification.ego_route_path),
                },
                "traffic_count": len(traffic),
                "partition_counts": counts,
                "traffic": traffic,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "topology-temporal-v2-freeze/v1",
        "project_root": str(PROJECT_ROOT),
        "partition_rule": {
            "validation": "lexicographic index modulo 5 == 0",
            "test": "lexicographic index modulo 5 == 1",
            "train": "lexicographic index modulo 5 in {2,3,4}",
            "small_asset_exception": "fewer than 3 files are shared and marked non-disjoint",
        },
        "source_files": source_files,
        "scenarios": scenarios,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "topo_v2" / "freeze_manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    payload = build_manifest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_content_sha256": payload["manifest_content_sha256"],
                "source_files": len(payload["source_files"]),
                "scenario_traffic_counts": {
                    row["scenario"]: row["traffic_count"]
                    for row in payload["scenarios"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
