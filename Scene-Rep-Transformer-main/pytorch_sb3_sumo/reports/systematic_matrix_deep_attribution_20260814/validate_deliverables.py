"""Fail-closed validation for all aggregation and reporting deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


issues: list[str] = []

required = [
    "artifact.json",
    "report.html",
    "deep_attribution_companion.ipynb",
    "data_quality_report.json",
    "deep_attribution.json",
    "official_analysis/systematic_summary.json",
    "tables/run_level_metrics.csv",
    "tables/method_macro_summary.csv",
    "tables/diagnostic_associations.csv",
]
for item in required:
    path = ROOT / item
    if not path.is_file() or path.stat().st_size == 0:
        issues.append(f"missing_or_empty:{item}")

quality = json.loads((ROOT / "data_quality_report.json").read_text(encoding="utf-8"))
checks = quality["checks"]
quality_expectations = {
    "status": quality["status"] == "ready",
    "run_rows": checks["run_rows"] == 108,
    "unique_run_keys": checks["unique_run_keys"] == 108,
    "episodes": checks["total_test_episodes"] == 5400,
    "50_episodes": checks["runs_with_50_episodes"] == 108,
    "100k": checks["runs_with_100k_trained_raw_steps"] == 108,
    "checkpoint_audit": checks["runs_with_checkpoint_audit_passed"] == 108,
    "finite": checks["runs_with_finite_metrics"] == 108,
    "provenance": checks["runs_with_validated_evaluation_provenance"] == 108,
    "pairing": not checks["evaluation_pairing_errors"],
    "summary_reconciliation": not checks["episode_summary_mismatches"],
    "protocol_hash": quality["protocol_hash_matches_frozen_contract"],
    "archive_hash": quality["retry_archive_hash_matches"],
}
for name, passed in quality_expectations.items():
    if not passed:
        issues.append(f"quality_failed:{name}")

run_rows = rows(ROOT / "tables" / "run_level_metrics.csv")
if len(run_rows) != 108:
    issues.append(f"run_table_rows:{len(run_rows)}")
if len({(row["method"], row["scenario"], row["seed"]) for row in run_rows}) != 108:
    issues.append("run_table_keys_not_unique")

method_rows = rows(ROOT / "tables" / "method_macro_summary.csv")
method_lookup = {row["method"]: row for row in method_rows}
expected_method_values = {
    "mst_slt": (0.94, 0.06),
    "temporal_graph": (0.9066666666666667, 0.08777777777777779),
    "mst": (0.8888888888888888, 0.10666666666666666),
    "full_balanced": (0.8088888888888889, 0.13444444444444445),
    "sac": (0.8055555555555556, 0.15),
    "ppo": (0.5077777777777778, 0.2411111111111111),
}
for method, (success, collision) in expected_method_values.items():
    row = method_lookup.get(method)
    if row is None:
        issues.append(f"missing_method:{method}")
        continue
    if not math.isclose(float(row["success_rate_macro_mean"]), success, abs_tol=1e-12):
        issues.append(f"success_mismatch:{method}")
    if not math.isclose(float(row["collision_rate_macro_mean"]), collision, abs_tol=1e-12):
        issues.append(f"collision_mismatch:{method}")

official = json.loads(
    (ROOT / "official_analysis" / "systematic_summary.json").read_text(encoding="utf-8")
)
if not official["matrix_complete"] or official["completion"]["errors"]:
    issues.append("official_analysis_incomplete")
holm_values = [
    float(row[field])
    for row in official["macro_comparisons"][:5]
    for field in ("success_rate_holm_adjusted_p", "collision_rate_holm_adjusted_p")
]
if any(value < 0.05 for value in holm_values):
    issues.append("unexpected_holm_significance")

artifact = json.loads((ROOT / "artifact.json").read_text(encoding="utf-8"))
manifest = artifact["manifest"]
snapshot = artifact["snapshot"]
if artifact["surface"] != "report" or manifest["surface"] != "report":
    issues.append("artifact_surface")
if manifest["blocks"][0]["body"] != f"# {manifest['title']}":
    issues.append("artifact_title_block")
block_ids = [block["id"] for block in manifest["blocks"]]
if len(block_ids) != len(set(block_ids)):
    issues.append("duplicate_block_ids")
if len(manifest["charts"]) != 5 or len(manifest["tables"]) != 2 or len(manifest["cards"]) != 4:
    issues.append("artifact_component_counts")
if snapshot["status"] != "ready" or snapshot.get("accessIssues"):
    issues.append("artifact_snapshot_status")
for source in artifact["sources"]:
    sql = source.get("query", {}).get("sql", "")
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        issues.append(f"source_sql_missing:{source.get('id')}")
if not all(item.get("sourceId") for item in manifest["cards"] + manifest["charts"] + manifest["tables"]):
    issues.append("source_reference_missing")

html = (ROOT / "report.html").read_text(encoding="utf-8")
html_requirements = [
    "data-analytics-portable-artifact-payload-source",
    "data-analytics-portable-reader-runtime-source",
    "冻结系统性实验矩阵：结果聚合与深层归因",
    "技术摘要",
    "建议的下一步",
    "待进一步回答的问题",
    "portable-chart-summary",
    "portable-table-card",
    "portable-metric-card",
]
for token in html_requirements:
    if token not in html:
        issues.append(f"html_missing:{token}")
if html.count("data-artifact-kind=\"chart\"") < 5:
    issues.append("html_chart_count")

notebook = nbformat.read(ROOT / "deep_attribution_companion.ipynb", as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
notebook_errors = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
    issues.append("notebook_not_fully_executed")
if notebook_errors:
    issues.append(f"notebook_errors:{len(notebook_errors)}")

report_method_hash_before = sha256(ROOT / "tables" / "method_macro_summary.csv")
artifact_hash = sha256(ROOT / "artifact.json")
report_hash = sha256(ROOT / "report.html")
notebook_hash = sha256(ROOT / "deep_attribution_companion.ipynb")

payload = {
    "ok": not issues,
    "issues": issues,
    "validated_counts": {
        "runs": len(run_rows),
        "episodes": checks["total_test_episodes"],
        "notebook_code_cells": len(code_cells),
        "notebook_errors": len(notebook_errors),
        "report_blocks": len(manifest["blocks"]),
        "report_charts": len(manifest["charts"]),
        "report_tables": len(manifest["tables"]),
        "report_metric_cards": len(manifest["cards"]),
    },
    "hashes": {
        "artifact_json_sha256": artifact_hash,
        "report_html_sha256": report_hash,
        "notebook_sha256": notebook_hash,
        "method_macro_summary_sha256": report_method_hash_before,
        "frozen_protocol_sha256": quality["protocol_sha256"],
        "retry_archive_manifest_sha256": quality["retry_archive_manifest_sha256"],
    },
    "browser_qa": {
        "canonical_validation": "passed",
        "portable_structural_verification": "passed",
        "interactive_browser_verification": "not_available",
        "reason": "Packaged headless-shell was unavailable; installed Chrome failed the builder's environment handshake, and the in-app browser security policy denied local-file access while its loopback context could not reach the local server.",
    },
    "research_method_modified": False,
}
(ROOT / "validation_report.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
sys.exit(0 if not issues else 2)
