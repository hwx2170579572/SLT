"""Prove read-only v4.9 -> v4.9.2 target-only development equivalence.

The proof reconstructs the new selector inputs from immutable train calibration
files.  It never reads validation traces for selection and never rewrites a
completed v4.9 run.  Validation summaries are used only after selection to
adopt the already completed pass when the exact deployed checkpoint, decoder,
and sealed parameter-state receipt are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import (
    SELECTOR_MODE,
    select_deployment,
    tied_top_pairs,
)
from tools.checkpoint_selector_v4_4 import sha256


CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_9_2.yaml"
PREREGISTRATION = ROOT / "results_topo_v4_9_2_dev" / "preregistration_receipt.json"
PARENT_SUMMARY = ROOT / "results_topo_v4_9_dev" / "development" / "summary.json"
PARENT_DECISION = ROOT / "results_topo_v4_9_dev" / "development" / "development_decision.json"
ATTRIBUTION = ROOT / "results_topo_v4_9_dev" / "development" / "attribution" / "complete_m1_m4_learned_model.json"
OUTPUT_ROOT = ROOT / "results_topo_v4_9_2_dev" / "development"
DEFAULT_REPORT = OUTPUT_ROOT / "target_only_equivalence.json"
DEFAULT_DECISION = OUTPUT_ROOT / "development_decision.json"

CONTRACT_SHA256 = "9e99845d3987a32216b213e34267c19fddaa2386880035498d60610615344a95"
PARENT_SUMMARY_SHA256 = "27dcdc5c1cf560f7364080ad4b5671887b870bb6b23ec343e6f7acfc4a6cfbe1"
PARENT_DECISION_SHA256 = "b0ffaf18245a484ec4567772ae80593887ba1eb7ec32f563a0a90063af9dbbf0"
ATTRIBUTION_SHA256 = "2211e0a01be87cde765e782299c246cfa3849de355ea6c1395447235876ea731"
EXPECTED_JOB_IDS = ("M1", "M2", "M3", "M4")


class EquivalenceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EquivalenceError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _tree_hashes(root: Path) -> dict[str, str]:
    _require(root.is_dir(), f"parent run directory missing: {root}")
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def validate_inputs() -> dict[str, Any]:
    _require(_sha256(CONTRACT) == CONTRACT_SHA256, "v4.9.2 contract hash drifted")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.9.2.strict-target-only-deployment-contract/v1",
        "unexpected v4.9.2 contract schema",
    )
    prereg = _load_json(PREREGISTRATION)
    _require(
        prereg.get("schema_version")
        == "topo-scene-v4.9.2.strict-target-only-preregistration/v1",
        "unexpected preregistration schema",
    )
    _require(prereg.get("created_before_v4_9_2_implementation") is True, "not preregistered")
    _require(prereg.get("experiment_contract_sha256") == CONTRACT_SHA256, "preregistration contract drifted")
    _require(prereg.get("formal_test_accessed") is False, "preregistration accessed formal")
    for path, expected, label in (
        (PARENT_SUMMARY, PARENT_SUMMARY_SHA256, "parent summary"),
        (PARENT_DECISION, PARENT_DECISION_SHA256, "parent decision"),
        (ATTRIBUTION, ATTRIBUTION_SHA256, "attribution"),
    ):
        _require(_sha256(path) == expected, f"{label} hash drifted")
    decision = _load_json(PARENT_DECISION)
    _require(decision.get("decision") == "pass", "parent development did not pass")
    _require(decision.get("formal_test_accessed") is False, "parent development accessed formal")
    return contract


def _candidate_from_calibration(
    root: Path, *, checkpoint_kind: str, secondary: bool
) -> dict[str, Any]:
    short = {"highest_training_success": "best", "exact_final": "final"}[
        checkpoint_kind
    ]
    block = "cal_secondary" if secondary else "cal"
    directory = root / "selector" / block / short / TARGET_DECODER
    detailed_path = directory / "detailed.json"
    actions_path = directory / "actions.json"
    detailed = _load_json(detailed_path)
    _require(detailed.get("computed_from_real_run") is True, "calibration not real")
    _require(detailed.get("fabricated_values") is False, "calibration fabricated")
    _require(detailed.get("formal_test_accessed") is False, "calibration accessed formal")
    _require(detailed.get("traffic_partition") == "train", "calibration is not train")
    _require(detailed.get("checkpoint_kind") == checkpoint_kind, "checkpoint kind drifted")
    _require(detailed.get("deployment_decoder") == TARGET_DECODER, "non-target calibration entered proof")
    _require(detailed.get("decoder_integrity_passed") is True, "decoder integrity failed")
    checkpoint_path = Path(detailed["checkpoint_path"])
    _require(checkpoint_path.is_file(), "source checkpoint is missing")
    _require(sha256(checkpoint_path) == detailed["checkpoint_sha256"], "source checkpoint hash drifted")
    return {
        "checkpoint_kind": checkpoint_kind,
        "deployment_decoder": TARGET_DECODER,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": detailed["checkpoint_sha256"],
        "traffic_partition": "train",
        "calibration_seed_start": int(detailed["calibration_seed_start"]),
        "summary": detailed["summary"],
        "episode_records": detailed["episode_records"],
        "calibration_result_sha256": _sha256(detailed_path),
        "calibration_action_diagnostics_sha256": _sha256(actions_path),
        "decoder_integrity": detailed["decoder_integrity"],
        "decoder_integrity_passed": True,
    }


def _run_proof(row: dict[str, Any]) -> dict[str, Any]:
    root = Path(row["run_directory"])
    original = _load_json(root / "selector" / "receipt.json")
    before = _tree_hashes(root)
    primary = [
        _candidate_from_calibration(root, checkpoint_kind=kind, secondary=False)
        for kind in ("highest_training_success", "exact_final")
    ]
    tied = tied_top_pairs(primary)
    secondary = None
    if len(tied) > 1:
        secondary = [
            _candidate_from_calibration(root, checkpoint_kind=kind, secondary=True)
            for kind, decoder in tied
            if decoder == TARGET_DECODER
        ]
    recomputed = select_deployment(primary, secondary, secondary_seed_offset=100)
    after = _tree_hashes(root)

    checks = {
        "parent_decoder_is_target_critic": original.get("selected_deployment_decoder") == TARGET_DECODER,
        "target_only_decoder_is_target_critic": recomputed.get("selected_deployment_decoder") == TARGET_DECODER,
        "selected_checkpoint_kind_matches": recomputed.get("selected_checkpoint_kind") == original.get("selected_checkpoint_kind"),
        "selected_source_checkpoint_sha256_matches": recomputed.get("selected_checkpoint_sha256") == original.get("selected_source_checkpoint_sha256"),
        "parent_source_and_selected_state_match": original.get("source_policy_parameter_state_sha256") == original.get("selected_model_parameter_state_sha256"),
        "parent_policy_parameter_state_preserved": original.get("policy_parameter_state_preserved") is True,
        "parent_policy_class_is_target_only": original.get("selected_model_policy_class") == "CollisionConstrainedTargetCriticSACPolicyV49",
        "target_only_candidate_count_is_two": recomputed.get("candidate_count") == 2,
        "fusion_candidate_absent": recomputed.get("fusion_candidate_present") is False,
        "actor_confidence_threshold_absent": recomputed.get("actor_confidence_threshold_present") is False,
        "selection_partition_train": recomputed.get("selection_partition") == "train",
        "validation_not_used_for_selection": recomputed.get("validation_used_for_selection") is False,
        "formal_not_used_for_selection": recomputed.get("formal_test_used_for_selection") is False,
        "completed_run_tree_unchanged": before == after,
    }
    return {
        "job_id": row["job_id"],
        "scenario": row["scenario"],
        "seed": int(row["seed"]),
        "parent_run_directory": str(root.resolve()),
        "parent_selector_receipt_sha256": _sha256(root / "selector" / "receipt.json"),
        "parent_selected_checkpoint_kind": original.get("selected_checkpoint_kind"),
        "parent_selected_deployment_decoder": original.get("selected_deployment_decoder"),
        "parent_selected_source_checkpoint_sha256": original.get("selected_source_checkpoint_sha256"),
        "parent_selected_model_parameter_state_sha256": original.get("selected_model_parameter_state_sha256"),
        "target_only_selector_mode": SELECTOR_MODE,
        "target_only_initial_tied_pairs": [f"{a}__{b}" for a, b in tied],
        "target_only_secondary_triggered": recomputed["secondary_calibration_triggered"],
        "target_only_selected_checkpoint_kind": recomputed["selected_checkpoint_kind"],
        "target_only_selected_deployment_decoder": recomputed["selected_deployment_decoder"],
        "target_only_selected_source_checkpoint_sha256": recomputed["selected_checkpoint_sha256"],
        "checks": checks,
        "equivalent": all(checks.values()),
        "parent_run_tree_hashes_before": before,
        "parent_run_tree_hashes_after": after,
    }


def prove(
    *, report_path: Path = DEFAULT_REPORT, decision_path: Path = DEFAULT_DECISION
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_inputs()
    summary = _load_json(PARENT_SUMMARY)
    _require(summary.get("complete") is True, "parent development matrix incomplete")
    rows = summary.get("per_run", [])
    _require([row.get("job_id") for row in rows] == list(EXPECTED_JOB_IDS), "parent job order drifted")
    proofs = [_run_proof(row) for row in rows]
    passed = all(row["equivalent"] for row in proofs)
    report = {
        "schema_version": "topo-scene-v4.9.2.target-only-development-equivalence/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_train_calibration": True,
        "fabricated_values": False,
        "read_only_parent_run_audit": True,
        "parent_validation_used_for_reselection": False,
        "formal_test_accessed": False,
        "experiment_contract_sha256": CONTRACT_SHA256,
        "parent_development_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_development_decision_sha256": PARENT_DECISION_SHA256,
        "complete_model_attribution_sha256": ATTRIBUTION_SHA256,
        "expected_cells": len(EXPECTED_JOB_IDS),
        "equivalent_cells": sum(row["equivalent"] for row in proofs),
        "all_conditions_passed": passed,
        "per_cell": proofs,
    }
    _write_json(report_path, report)
    decision = {
        "schema_version": "topo-scene-v4.9.2.development-adoption-decision/v1",
        "decision": "pass" if passed else "fail",
        "adoption_kind": "exact_target_only_protocol_equivalence",
        "fresh_v4_9_2_development_training_run": False,
        "scientific_model_changed_from_v4_9": False,
        "completed_parent_run_artifacts_modified": False,
        "validation_used_for_reselection": False,
        "formal_test_accessed": False,
        "experiment_contract_sha256": CONTRACT_SHA256,
        "target_only_equivalence_report": str(report_path.resolve()),
        "target_only_equivalence_report_sha256": _sha256(report_path),
        "all_four_parent_deployments_target_critic": all(
            row["parent_selected_deployment_decoder"] == TARGET_DECODER for row in proofs
        ),
        "all_four_target_only_checkpoints_match": all(
            row["checks"]["selected_checkpoint_kind_matches"] for row in proofs
        ),
        "all_four_source_checkpoint_hashes_match": all(
            row["checks"]["selected_source_checkpoint_sha256_matches"] for row in proofs
        ),
        "all_four_sealed_parameter_states_preserved": all(
            row["checks"]["parent_source_and_selected_state_match"] for row in proofs
        ),
        "promotion_unlocked": passed,
    }
    _write_json(decision_path, decision)
    return report, decision


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    root.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report, decision = prove(
        report_path=args.report.resolve(), decision_path=args.decision.resolve()
    )
    print(json.dumps({"report": report, "decision": decision}, ensure_ascii=False, indent=2))
    return 0 if decision["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
