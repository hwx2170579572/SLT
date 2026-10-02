"""Machine-audit final topology-temporal claims, numbers, and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("scene_rep", "temporal_graph", "topo_scene", "topo_scene_balanced")
RESULT_METRICS = (
    "success_rate",
    "collision_rate",
    "mean_return",
    "successful_completion_time_seconds",
    "parameter_count",
    "train_ms_per_gradient_step",
    "inference_ms_per_action",
    "peak_gpu_memory_mb",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-summary", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/final_summary.json")
    parser.add_argument("--attribution", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/attribution.json")
    parser.add_argument("--iteration-ledger", type=Path, default=PROJECT_ROOT / "results_topo_scene/diagnosis/iteration_ledger.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/FINAL_REPORT.md")
    parser.add_argument("--figure-source", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/figures/dashboard_source_data.json")
    parser.add_argument("--figure-receipt", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/figures/dashboard_receipt.json")
    parser.add_argument("--code-manifest", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/code_manifest.json")
    parser.add_argument("--development-gate", type=Path, default=PROJECT_ROOT / "results_topo_scene/development/gate_decision.json")
    parser.add_argument("--ccfa", type=Path, default=PROJECT_ROOT / "ccfa.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results_topo_scene/final/integrity_audit.json")
    args = parser.parse_args()

    final = _read_json(args.final_summary.resolve())
    attribution = _read_json(args.attribution.resolve())
    ledger = _read_json(args.iteration_ledger.resolve())
    figure_source = _read_json(args.figure_source.resolve())
    figure_receipt = _read_json(args.figure_receipt.resolve())
    code_manifest = _read_json(args.code_manifest.resolve())
    gate = _read_json(args.development_gate.resolve())
    report = args.report.resolve().read_text(encoding="utf-8")
    ccfa = yaml.safe_load(args.ccfa.resolve().read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "detail": detail})

    check(
        "real_evidence_flags",
        final.get("computed_from_real_runs") is True
        and final.get("fabricated_values") is False
        and final.get("claim_boundary_recorded") is True
        and attribution.get("evidence_bound") is True
        and attribution.get("fabricated_values") is False
        and ledger.get("fabricated_values") is False,
        "Final summary, attribution, and ledger must all bind to real evidence.",
    )
    check(
        "decision_consistency",
        final.get("final_decision") == ledger.get("final_decision"),
        f"final={final.get('final_decision')}, ledger={ledger.get('final_decision')}",
    )
    check(
        "confirmation_gate_consistency",
        gate.get("claim_allowed") is False
        and final.get("confirmation", {}).get("completed") is False
        and ledger.get("confirmation_started") is False,
        "A failed development gate must not be represented as completed confirmation.",
    )

    attribution_numbers_ok = True
    attribution_diffs: list[str] = []
    for method in METHODS:
        for metric in RESULT_METRICS:
            left = final["method_results"][method][metric]
            right = attribution["method_results"][method][metric]
            if not _same_number(left, right):
                attribution_numbers_ok = False
                attribution_diffs.append(f"{method}.{metric}: {left!r} != {right!r}")
    check(
        "attribution_numeric_consistency",
        attribution_numbers_ok,
        "; ".join(attribution_diffs) if attribution_diffs else "All method metrics match final_summary.json.",
    )

    panel_rows = {
        (row["method"], row["metric"]): row["value"]
        for row in figure_source.get("rows", [])
        if row.get("panel") == "a"
    }
    figure_numbers_ok = True
    figure_diffs: list[str] = []
    for method in METHODS:
        for metric in ("success_rate", "collision_rate", "mean_return"):
            key = (method, metric)
            expected = final["method_results"][method][metric]
            if key not in panel_rows or not _same_number(expected, panel_rows[key]):
                figure_numbers_ok = False
                figure_diffs.append(f"{method}.{metric}: final={expected!r}, figure={panel_rows.get(key)!r}")
    check(
        "figure_numeric_consistency",
        figure_numbers_ok,
        "; ".join(figure_diffs) if figure_diffs else "Panel (a) values exactly match final_summary.json.",
    )

    claim_statuses = {row["claim_id"]: row["status"] for row in ccfa["research"]["claim_evidence"]}
    final_statuses = {claim_id: row["status"] for claim_id, row in final["claims"].items()}
    check(
        "ccfa_claim_status_consistency",
        claim_statuses == final_statuses,
        f"ccfa={claim_statuses}, final={final_statuses}",
    )

    report_numbers_ok = True
    missing_report_values: list[str] = []
    for method in METHODS:
        row = final["method_results"][method]
        expected_prefix = (
            f"| {row['label']} | {float(row['success_rate']):.3f} | "
            f"{float(row['collision_rate']):.3f} | {float(row['mean_return']):.3f} |"
        )
        if expected_prefix not in report:
            report_numbers_ok = False
            missing_report_values.append(expected_prefix)
    check(
        "report_numeric_presence",
        report_numbers_ok,
        f"Missing exact report rows: {missing_report_values}" if missing_report_values else "All final outcome rows exactly match final_summary.json.",
    )
    check(
        "report_no_placeholders",
        "TBD" not in report and "TODO" not in report and final["final_decision"] in report,
        "Final report must contain the decision and no unresolved result placeholders.",
    )

    receipt_ok = True
    receipt_diffs: list[str] = []
    for section in ("sources", "outputs"):
        for row in figure_receipt.get(section, []):
            path = PROJECT_ROOT / row["path"]
            actual = _sha256(path) if path.is_file() else None
            if actual != row["sha256"]:
                receipt_ok = False
                receipt_diffs.append(f"{row['path']}: expected={row['sha256']}, actual={actual}")
    check(
        "figure_receipt_hashes",
        receipt_ok,
        "; ".join(receipt_diffs) if receipt_diffs else "All figure source/output hashes match.",
    )

    manifest_ok = code_manifest.get("complete") is True
    manifest_diffs: list[str] = []
    for row in code_manifest.get("files", []):
        path = PROJECT_ROOT / row["path"]
        actual = _sha256(path) if path.is_file() else None
        if actual != row["sha256"] or (path.is_file() and path.stat().st_size != row["bytes"]):
            manifest_ok = False
            manifest_diffs.append(row["path"])
    check(
        "code_manifest_hashes",
        manifest_ok,
        f"Mismatched files: {manifest_diffs}" if manifest_diffs else "All implementation/test/contract hashes match.",
    )

    curve_path = PROJECT_ROOT / final["primary_learning_curve"]
    curve = _read_json(curve_path)
    check(
        "primary_curve_contract",
        curve.get("computed_from_real_rollouts") is True
        and curve.get("fabricated_values") is False
        and curve.get("fixed_checkpoint_curves_are_primary") is True
        and isinstance(curve.get("paired_protocol"), dict),
        "Primary curve must be explicit-seed, paired, and real.",
    )

    passed = all(row["passed"] for row in checks)
    payload = {
        "contract": "topo-scene.integrity-audit/v1",
        "mode": "claim-and-numeric-audit",
        "passed": passed,
        "severity": "none" if passed else "high",
        "checks": checks,
        "citation_audit": "not_applicable_no_citations_in_experiment_report",
        "no_invention_status": True,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
