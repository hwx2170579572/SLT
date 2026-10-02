"""Validate and freeze the strict target-only v4.9.2 promotion protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_9 import V49_CANDIDATE, V49_IMPLEMENTATION_IDS
from tools import run_topo_v4_9_2_experiments as runner
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER


ENGINEERING = ROOT / "results_topo_v4_9_2_dev" / "engineering"
AUDIT = ENGINEERING / "model_only_integrity_audit.json"
EXCEPTION = ENGINEERING / "historical_state_regression_exception.json"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
SMOKE_V1 = ENGINEERING / "smoke" / "model_e2e"
SMOKE_V2 = ENGINEERING / "smoke" / "model_e2e_r2"
TARGETED_XML = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_XML = ENGINEERING / "logs" / "full_regression.xml"
CURRENT_XML = ENGINEERING / "logs" / "full_regression_current_state.xml"

SCIENTIFIC_FILES = (
    "experiments/topo_scene_v4/experiment_contract_v4_9_2.yaml",
    "experiments/topo_scene_v4/V4_9_2_STRICT_TARGET_ONLY_PROTOCOL.md",
    "tools/checkpoint_decoder_selector_v4_9_2.py",
    "tools/prove_v4_9_2_target_only_equivalence.py",
    "tools/train_sb3_v4_9_2.py",
    "tools/train_paper_sb3_sumo_v4_9_2.py",
    "tools/run_topo_v4_9_2_experiments.py",
    "tools/run_all_v4_9_2_promotion.py",
    "tools/run_all_pending_promotions.py",
    "tools/freeze_v4_9_2.py",
    "tests_sb3_sumo/test_checkpoint_decoder_selector_v4_9_2.py",
    "tests_sb3_sumo/test_train_sb3_v4_9_2.py",
    "tests_sb3_sumo/test_run_topo_v4_9_2_experiments.py",
    "results_topo_v4_9_2_dev/preregistration_receipt.json",
    "results_topo_v4_9_2_dev/development/target_only_equivalence.json",
    "results_topo_v4_9_2_dev/development/development_decision.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    values = root.attrib
    if "tests" not in values:
        suite = root.find("testsuite")
        _require(suite is not None, f"JUnit testsuite missing: {path}")
        values = suite.attrib
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "tests": int(values.get("tests", 0)),
        "failures": int(values.get("failures", 0)),
        "errors": int(values.get("errors", 0)),
        "skipped": int(values.get("skipped", 0)),
        "time_seconds": float(values.get("time", 0.0)),
    }


def _source_function_audit(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    forbidden = (
        "kinematic_projection",
        "safety_projection",
        "ttc_threshold",
        "headway_threshold",
        "lane_change_veto",
        "action_override",
    )
    hits = [name for name in names if any(token in name.lower() for token in forbidden)]
    return {
        "path": relative,
        "sha256": _sha256(path),
        "defined_functions": names,
        "forbidden_operational_function_names": hits,
    }


def _smoke_job() -> runner.Job:
    return runner.Job(
        stage="promotion",
        job_id="S",
        kind="engineering_smoke",
        method="selected_v4_candidate",
        algorithm=V49_CANDIDATE,
        implementation_id=V49_IMPLEMENTATION_IDS[V49_CANDIDATE],
        scenario="cross",
        seed=118,
        raw_steps=96,
        calibration_episodes=2,
        calibration_seed_start=207000,
        evaluation_episodes=2,
        evaluation_split="validation",
        evaluation_seed_start=217000,
        role="engineering_smoke",
        protocol_tag="smoke",
    )


def validate_smoke() -> dict[str, Any]:
    first = _load_json(SMOKE_V1 / "selector" / "receipt.json")
    _require(first.get("selected_deployment_decoder") == TARGET_DECODER, "first smoke did not actually deploy target")
    _require(first.get("calibration_deterministic_lane_decoder") == "parent_control", "first smoke metadata trigger no longer reproduces")
    second = _load_json(SMOKE_V2 / "selector" / "receipt.json")
    _require(second.get("selector_mode") == runner.SELECTOR_MODE, "smoke selector mode drifted")
    _require(second.get("candidate_count") == 2, "smoke candidate count drifted")
    _require(second.get("deployment_decoder_candidates") == [TARGET_DECODER], "smoke decoder candidates drifted")
    _require(second.get("calibration_deterministic_lane_decoder") == TARGET_DECODER, "smoke calibration metadata drifted")
    _require(second.get("selected_deployment_decoder") == TARGET_DECODER, "smoke selected non-target decoder")
    _require(second.get("selected_model_policy_class") == "CollisionConstrainedTargetCriticSACPolicyV49", "smoke policy class drifted")
    _require(second.get("policy_parameter_state_preserved") is True, "smoke serialization changed tensors")
    for key in (
        "fusion_candidate_present",
        "actor_confidence_threshold_present",
        "external_kinematic_projection",
        "action_postprocessing_override",
    ):
        _require(second.get(key) is False, f"smoke enabled {key}")
    fusion_dirs = [path for path in (SMOKE_V2 / "selector").rglob("*") if path.name == "fusion_0_90"]
    _require(not fusion_dirs, "smoke created a fusion calibration directory")
    runner._validate_candidate(SMOKE_V2, _smoke_job())
    return {
        "engineering_only_not_counted_for_scientific_gate": True,
        "first_attempt": {
            "path": str(SMOKE_V1.resolve()),
            "selector_receipt_sha256": _sha256(SMOKE_V1 / "selector" / "receipt.json"),
            "accepted": False,
            "reason": "legacy_single_decoder_metadata_reported_parent_control",
            "run_preserved": True,
        },
        "corrected_fresh_attempt": {
            "path": str(SMOKE_V2.resolve()),
            "selector_receipt_sha256": _sha256(SMOKE_V2 / "selector" / "receipt.json"),
            "selected_model_sha256": _sha256(SMOKE_V2 / "selected_model.zip"),
            "action_diagnostics_sha256": _sha256(SMOKE_V2 / "action_diagnostics.json"),
            "collision_label_diagnostics_sha256": _sha256(SMOKE_V2 / "collision_label_diagnostics.json"),
            "candidate_artifact_validation": "passed",
            "fusion_calibration_directory_count": 0,
            "selected_decoder": TARGET_DECODER,
            "accepted": True,
        },
    }


def main() -> int:
    _require(_sha256(runner.DEFAULT_CONTRACT) == runner.CONTRACT_SHA256, "contract hash drifted")
    runner.validate_contract(runner.load_contract())
    runner.validate_preregistration()
    runner.validate_development_adoption()

    targeted = _junit(TARGETED_XML)
    full = _junit(FULL_XML)
    current = _junit(CURRENT_XML)
    _require(targeted == {**targeted, "tests": 14, "failures": 0, "errors": 0, "skipped": 0}, "targeted regression did not pass")
    _require(current["tests"] == 449 and current["failures"] == 0 and current["errors"] == 0, "current-state regression did not pass")
    _require(full["tests"] == 450 and full["failures"] == 3 and full["errors"] == 0, "diagnostic full regression drifted")

    exception = {
        "schema_version": "topo-scene-v4.9.2.historical-regression-exception/v1",
        "formal_test_accessed": False,
        "full_diagnostic_result": full,
        "environmental_path_length_failures": [
            "tests_sb3_sumo/test_paper_reproduction.py::test_compatible_predecessor_requires_hash_and_detailed_evidence",
            "tests_sb3_sumo/test_reevaluate_paper_run.py::test_adopt_valid_primary_evaluation_preserves_evidence",
        ],
        "short_basetemp_recheck": "2 passed in 0.27s",
        "non_hermetic_historical_state_test": "tests_sb3_sumo/test_topo_v4_8_2_experiments.py::test_v4_8_2_h1_preflight_reproduces_both_bugs_and_adopts_in_place",
        "historical_test_reason": "asserts v4.8.2 remains at H1 although later completed v4.8/v4.9 evidence intentionally supersedes that state",
        "current_state_result": current,
        "scientific_v4_9_2_tests_deselected": 0,
    }
    _write_json(EXCEPTION, exception)

    smoke = validate_smoke()
    source_audit = {
        relative: _source_function_audit(relative)
        for relative in (
            "tools/checkpoint_decoder_selector_v4_9_2.py",
            "tools/train_sb3_v4_9_2.py",
        )
    }
    _require(all(not row["forbidden_operational_function_names"] for row in source_audit.values()), "forbidden operational function was defined")
    audit = {
        "schema_version": "topo-scene-v4.9.2.model-only-integrity-audit/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "scientific_model_changed_from_v4_9": False,
        "parent_model_freeze_sha256": runner.PARENT_FREEZE_SHA256,
        "new_operational_source_ast_audit": source_audit,
        "runtime_smoke": smoke,
        "deployment_decoder_candidates": [TARGET_DECODER],
        "checkpoint_decoder_candidate_pairs": 2,
        "fusion_candidate_present": False,
        "actor_confidence_threshold_present": False,
        "external_kinematic_projection": False,
        "headway_or_ttc_threshold": False,
        "lane_change_veto": False,
        "manual_geometry_collision_label": False,
        "action_postprocessing_override": False,
        "learned_decision_equation": "minimum_reward_q_minus_lambda_maximum_collision_value",
        "collision_label_source": "observed_environment_info_collision",
    }
    _write_json(AUDIT, audit)

    files = {relative: _sha256(ROOT / relative) for relative in SCIENTIFIC_FILES}
    files[str(AUDIT.relative_to(ROOT)).replace("\\", "/")] = _sha256(AUDIT)
    files[str(EXCEPTION.relative_to(ROOT)).replace("\\", "/")] = _sha256(EXCEPTION)
    freeze = {
        "schema_version": "topo-scene-v4.9.2.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "promotion_started_before_freeze": False,
        "single_protocol_change": "restrict_candidate_deployment_to_learned_target_critic",
        "scientific_model_changed_from_v4_9": False,
        "training_changed_from_v4_9": False,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "parent_scientific_freeze_sha256": runner.PARENT_FREEZE_SHA256,
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "development_adoption_decision_sha256": _sha256(runner.DEFAULT_DEVELOPMENT_DECISION),
        "target_only_equivalence_sha256": _sha256(runner.DEFAULT_EQUIVALENCE),
        "model_only_integrity_audit_sha256": _sha256(AUDIT),
        "scientific_files": files,
        "scientific_content_sha256": _canonical_sha256(files),
        "engineering_evidence": {
            "targeted_regression": targeted,
            "full_diagnostic_regression": full,
            "current_state_regression": current,
            "historical_state_exception": {
                "path": str(EXCEPTION.resolve()),
                "sha256": _sha256(EXCEPTION),
            },
            "smoke": smoke,
        },
        "execution_policy": {
            "promotion": "attempt_all_then_summarize",
            "registered_future_promotions": "attempt_all_versions_and_all_jobs_then_summarize",
            "failure_cancels_remaining_jobs_or_versions": False,
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "conda_environment": "llm_pipeline",
        },
    }
    _write_json(FREEZE, freeze)
    receipt = {
        "schema_version": "topo-scene-v4.9.2.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "promotion_started_before_freeze": False,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": _sha256(FREEZE),
        "scientific_content_sha256": freeze["scientific_content_sha256"],
        "targeted_tests": targeted,
        "current_state_regression": current,
        "model_only_integrity_audit": {"path": str(AUDIT.resolve()), "sha256": _sha256(AUDIT)},
        "strict_target_only_smoke": smoke["corrected_fresh_attempt"],
        "promotion_execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "next_stage": "fresh_12_job_v4_9_2_promotion",
        "formal_test_locked": True,
    }
    _write_json(RECEIPT, receipt)
    print(json.dumps({"freeze": freeze, "receipt": receipt}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
