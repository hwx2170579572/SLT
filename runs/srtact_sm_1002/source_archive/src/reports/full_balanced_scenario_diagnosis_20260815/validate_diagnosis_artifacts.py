"""Validate the Full+BalancedSlots diagnosis package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return digest


required = [
    ROOT / "TECHNICAL_REPORT.md",
    ROOT / "report.html",
    ROOT / "evidence_manifest.json",
    ROOT / "full_balanced_diagnosis_companion.ipynb",
    ROOT / "tables" / "full_vs_temporal_by_scene.csv",
    ROOT / "tables" / "full_vs_temporal_by_seed.csv",
    ROOT / "tables" / "scenario_validity_audit.csv",
    ROOT / "tables" / "topology_audit.csv",
    ROOT / "tables" / "selected_rollout_trace_summary.csv",
    ROOT / "figures" / "full_vs_temporal_scene_deltas.svg",
    ROOT / "figures" / "full_vs_temporal_scene_deltas.png",
    ROOT / "figures" / "seed_instability_roundabout_b_carla.svg",
    ROOT / "figures" / "seed_instability_roundabout_b_carla.png",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise AssertionError({"missing": missing})

with (ROOT / "tables" / "full_vs_temporal_by_scene.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    scene = list(csv.DictReader(handle))
with (ROOT / "tables" / "full_vs_temporal_by_seed.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    seeds = list(csv.DictReader(handle))
with (ROOT / "tables" / "scenario_validity_audit.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    audit = list(csv.DictReader(handle))
with (ROOT / "tables" / "topology_audit.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    topology = list(csv.DictReader(handle))

assert len(scene) == 6
assert len(seeds) == 18
assert len(audit) == 4
assert len(topology) == 6
roundabout_c = next(row for row in scene if row["scenario"] == "roundabout")
assert abs(float(roundabout_c["delta_success_rate"]) - 0.02) < 1e-12
cross = next(row for row in topology if row["scenario"] == "cross")
assert int(cross["conflict"]) == 0
carla = next(row for row in audit if row["scenario"] == "carla")
assert carla["traffic_partition_disjoint"] == "False"

notebook = nbformat.read(ROOT / "full_balanced_diagnosis_companion.ipynb", as_version=4)
errors = []
for index, cell in enumerate(notebook.cells):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append({"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")})
if errors:
    raise AssertionError({"notebook_errors": errors})

html_text = (ROOT / "report.html").read_text(encoding="utf-8")
for phrase in ("环岛 C", "CARLA", "conflict=0", "不替换现有主场景", "未修改研究方法"):
    if phrase not in html_text:
        raise AssertionError({"missing_html_phrase": phrase})

result = {
    "ok": True,
    "validated": {
        "scenario_rows": len(scene),
        "seed_rows": len(seeds),
        "scenario_audit_rows": len(audit),
        "topology_rows": len(topology),
        "notebook_cells": len(notebook.cells),
        "notebook_errors": len(errors),
    },
    "hashes": {path.name: sha256(path) for path in required},
    "research_method_modified": False,
}
(ROOT / "validation_report.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
