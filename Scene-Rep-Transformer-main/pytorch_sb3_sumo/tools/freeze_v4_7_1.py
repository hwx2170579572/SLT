"""Seal the v4.7.1 engineering-only selected-deployment binding patch."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_7_1_experiments as runner
from tools import run_topo_v4_7_experiments as parent


ENGINEERING = ROOT / "results_topo_v4_7_1_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
SMOKE = ENGINEERING / "smoke" / "s1"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
MIGRATION = ENGINEERING / "logs" / "parent_v4_7_migration.json"
QUARANTINE = ENGINEERING / "logs" / "rejected_attempt_quarantine.json"

PATCH_FILES = (
    "tools/train_sb3_v4_7_1.py",
    "tools/train_paper_sb3_sumo_v4_7_1.py",
    "tools/run_topo_v4_7_1_experiments.py",
    "tools/freeze_v4_7_1.py",
    "tests_sb3_sumo/test_v4_7_1_adapter.py",
    "tests_sb3_sumo/test_topo_v4_7_1_experiments.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().lower()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _verify_zip(path: Path) -> None:
    with ZipFile(path, "r") as archive:
        bad = archive.testzip()
    _require(bad is None, f"ZIP CRC failure: {path} ({bad})")


def _pytest_evidence(path: Path, *, minimum_tests: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    _require(tests >= minimum_tests, f"test count is below {minimum_tests}: {tests}")
    _require(failures == 0 and errors == 0, f"test failures/errors: {failures}/{errors}")
    return {
        **_source(path), "tests": tests, "failures": failures,
        "errors": errors, "skipped": skipped,
    }


def _parent_migration() -> dict[str, Any]:
    parent_path = runner.PARENT_V47_FREEZE
    _require(_sha256(parent_path) == runner.PARENT_V47_FREEZE_SHA256, "parent v4.7 freeze hash drifted")
    payload = parent.validate_implementation_freeze(parent_path)
    files = payload["scientific_files"]
    for relative, expected in files.items():
        path = ROOT / relative
        _require(path.is_file(), f"parent v4.7 file missing: {relative}")
        _require(_sha256(path) == expected, f"parent v4.7 file drifted: {relative}")
    value = {
        "schema_version": "topo-scene-v4.7.1.parent-v4.7-migration/v1",
        "parent_v4_7_files_unchanged": True,
        "checked_files": len(files),
        "parent_v4_7_implementation_freeze_sha256": _sha256(parent_path),
        "parent_v4_7_scientific_content_sha256": payload["scientific_content_sha256"],
        "formal_test_accessed": False,
    }
    _write(MIGRATION, value)
    return {**value, "source": _source(MIGRATION)}


def _preregistration_evidence() -> dict[str, Any]:
    receipt = _load(runner.DEFAULT_PATCH_PREREGISTRATION)
    _require(receipt.get("schema_version") == "topo-scene-v4.7.1.engineering-patch-preregistration/v1", "patch preregistration schema drifted")
    _require(receipt.get("preregistered_before_patch_implementation") is True, "patch was not preregistered")
    _require(receipt.get("preregistered_before_from_scratch_retry") is True, "retry was not preregistered")
    _require(receipt.get("scientific_protocol_changed") is False, "preregistration changed science")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("scientific_contract_sha256") == parent.contract_sha256(), "scientific contract hash drifted")
    _require(receipt.get("engineering_patch_contract_sha256") == runner.PATCH_CONTRACT_SHA256, "patch contract hash drifted")
    _require(receipt.get("engineering_patch_plan_sha256") == runner.PATCH_PLAN_SHA256, "patch plan hash drifted")
    planned = receipt.get("planned_files_absent_at_seal", {})
    _require(set(planned) == set(PATCH_FILES), "planned patch file set drifted")
    _require(all(value is True for value in planned.values()), "planned patch file absence drifted")
    return {**_source(runner.DEFAULT_PATCH_PREREGISTRATION), "planned_files_absent_at_seal": len(planned)}


def _rejected_attempt_quarantine() -> dict[str, Any]:
    root = ROOT / "results_topo_v4_7_dev" / "development" / "runs" / "G1__cand__cross__s6__pe97ef91a"
    detailed = root / "paper_evaluation_detailed.json"
    selector = root / "selector" / "receipt.json"
    _require(_sha256(detailed) == runner.REJECTED_DETAILED_SHA256, "rejected detailed artifact drifted")
    _require(_sha256(selector) == runner.REJECTED_SELECTOR_SHA256, "rejected selector artifact drifted")
    detailed_value = _load(detailed)
    _require("selected_deployment_decoder" not in detailed_value, "rejected detailed artifact was repaired in place")
    execution = _load(ROOT / "results_topo_v4_7_dev" / "development" / "last_execution.json")
    failures = execution.get("failures", [])
    _require(len(failures) == 1, "rejected attempt failure record drifted")
    _require(failures[0].get("acceptance_reason") == "detailed selected decoder mismatch", "rejected attempt reason drifted")
    value = {
        "schema_version": "topo-scene-v4.7.1.rejected-attempt-quarantine/v1",
        "run_directory": str(root.resolve()),
        "status": "engineering_rejected_not_gate_eligible",
        "validation_outcome_inspected_for_patch_design": False,
        "validation_outcome_gate_eligible": False,
        "checkpoint_eligible_for_retry": False,
        "detailed_sha256": _sha256(detailed),
        "selector_receipt_sha256": _sha256(selector),
        "acceptance_reason": failures[0]["acceptance_reason"],
        "formal_test_accessed": False,
    }
    _write(QUARANTINE, value)
    return {**value, "source": _source(QUARANTINE)}


def _smoke_evidence() -> dict[str, Any]:
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    selector = _load(SMOKE / "selector" / "receipt.json")
    training = _load(SMOKE / "training_diagnostics.json")
    returns = _load(SMOKE / "return_estimator_diagnostics.json")
    _require(detailed.get("engineering_patch_id") == runner.PATCH_ID, "smoke patch id drifted")
    _require(detailed.get("scientific_protocol_changed_by_engineering_patch") is False, "smoke claims scientific change")
    expected = {
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_source_checkpoint_sha256": selector["selected_checkpoint_sha256"],
        "selected_model_policy_class": selector["selected_model_policy_class"],
        "selected_model_parameter_state_sha256": selector["selected_model_parameter_state_sha256"],
    }
    for key, value in expected.items():
        _require(detailed.get(key) == value, f"smoke detailed binding drifted: {key}")
    sealed = datetime.fromisoformat(selector["sealed_at_utc"])
    constructed = datetime.fromisoformat(detailed["validation_environment_constructed_at_utc"])
    _require(sealed <= constructed, "smoke validation predates selector receipt")
    _require(training.get("return_estimator") == returns, "smoke return diagnostics disagree")
    _require(returns.get("n_step") == 16 and returns.get("horizon_correct_bootstrap") is True, "smoke scientific return estimator drifted")
    _require(int(returns.get("sampled_horizon_count", 0)) > 0, "smoke sampled no replay horizons")
    _verify_zip(SMOKE / "final_model.zip"); _verify_zip(SMOKE / "selected_model.zip")
    return {
        "directory": str(SMOKE.resolve()),
        "paper_evaluation_sha256": _sha256(SMOKE / "paper_evaluation_detailed.json"),
        "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
        "return_estimator_diagnostics_sha256": _sha256(SMOKE / "return_estimator_diagnostics.json"),
        "all_four_bindings_match": True,
        "receipt_predates_validation": True,
        "scientific_protocol_changed": False,
    }


def main() -> int:
    parent.validate_contract(parent.load_contract())
    runner.validate_patch_contract(runner.load_patch_contract())
    _require(_sha256(runner.DEFAULT_PATCH_CONTRACT) == runner.PATCH_CONTRACT_SHA256, "patch contract file hash drifted")
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=25)
    full = _pytest_evidence(FULL_LOG, minimum_tests=327)
    migration = _parent_migration()
    preregistration = _preregistration_evidence()
    quarantine = _rejected_attempt_quarantine()
    smoke = _smoke_evidence()
    patch_files = {}
    for relative in PATCH_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        patch_files[relative] = _sha256(path)
    parent_freeze = _load(runner.PARENT_V47_FREEZE)
    payload = {
        "schema_version": "topo-scene-v4.7.1.engineering-patch-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_retry_job": True,
        "formal_test_accessed": False,
        "patch_id": runner.PATCH_ID,
        "single_engineering_change": "bind_sealed_selected_deployment_fields_into_detailed_evaluation",
        "scientific_protocol_changed": False,
        "scientific_contract_sha256": parent.contract_sha256(),
        "parent_v4_7_implementation_freeze_sha256": _sha256(runner.PARENT_V47_FREEZE),
        "parent_v4_7_scientific_content_sha256": parent_freeze["scientific_content_sha256"],
        "patch_contract_sha256": _sha256(runner.DEFAULT_PATCH_CONTRACT),
        "patch_plan_sha256": runner.PATCH_PLAN_SHA256,
        "patch_preregistration_sha256": _sha256(runner.DEFAULT_PATCH_PREREGISTRATION),
        "patch_files": patch_files,
        "patch_content_sha256": _canonical_sha256(patch_files),
        "engineering_evidence": {
            "parent_migration": migration,
            "preregistration": preregistration,
            "rejected_attempt_quarantine": quarantine,
            "targeted_tests": targeted,
            "full_regression": full,
            "real_sumo_metadata_binding_smoke": smoke,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "conda_environment": "llm_pipeline",
        },
    }
    _write(FREEZE, payload)
    freeze_hash = _sha256(FREEZE)
    receipt = {
        "schema_version": "topo-scene-v4.7.1.engineering-patch-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "scientific_contract_sha256": payload["scientific_contract_sha256"],
        "engineering_patch_contract_sha256": payload["patch_contract_sha256"],
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "patch_content_sha256": payload["patch_content_sha256"],
        "targeted_tests": targeted,
        "full_regression": full,
        "smoke_paper_evaluation_sha256": smoke["paper_evaluation_sha256"],
        "rejected_attempt_gate_eligible": False,
    }
    _write(RECEIPT, receipt)
    print(json.dumps({
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "patch_content_sha256": payload["patch_content_sha256"],
        "engineering_receipt": str(RECEIPT.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
