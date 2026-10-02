"""Seal the v4.7.2 runner-context isolation patch."""

from __future__ import annotations

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

from tools import run_topo_v4_7_2_experiments as runner
from tools import run_topo_v4_7_1_experiments as parent_patch
from tools import run_topo_v4_7_experiments as scientific


ENGINEERING = ROOT / "results_topo_v4_7_2_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
PREFLIGHT = ENGINEERING / "logs" / "corrected_context_preflight.json"

PATCH_FILES = (
    "tools/run_topo_v4_7_2_experiments.py",
    "tools/freeze_v4_7_2.py",
    "tests_sb3_sumo/test_topo_v4_7_2_experiments.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def _pytest_evidence(path: Path, *, minimum_tests: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    _require(tests >= minimum_tests, f"test count is below {minimum_tests}: {tests}")
    _require(failures == 0 and errors == 0, f"test failures/errors: {failures}/{errors}")
    return {**_source(path), "tests": tests, "failures": failures, "errors": errors, "skipped": skipped}


def _preregistration_evidence() -> dict[str, Any]:
    receipt = _load(runner.DEFAULT_PREREGISTRATION)
    _require(receipt.get("schema_version") == "topo-scene-v4.7.2.runner-context-preregistration/v1", "preregistration schema drifted")
    _require(receipt.get("preregistered_before_runner_patch") is True, "runner patch was not preregistered")
    _require(receipt.get("preregistered_before_retry_training") is True, "retry was not preregistered")
    _require(receipt.get("scientific_job_started_under_v4_7_1") is False, "v4.7.1 scientific job unexpectedly started")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("patch_contract_sha256") == runner.PATCH_CONTRACT_SHA256, "patch contract hash drifted")
    _require(receipt.get("patch_plan_sha256") == runner.PATCH_PLAN_SHA256, "patch plan hash drifted")
    planned = receipt.get("planned_files_absent_at_seal", {})
    _require(set(planned) == set(PATCH_FILES), "planned patch files drifted")
    _require(all(value is True for value in planned.values()), "planned absence receipt drifted")
    return {**_source(runner.DEFAULT_PREREGISTRATION), "planned_files_absent_at_seal": len(planned)}


def _parent_evidence() -> dict[str, Any]:
    _require(_sha256(runner.PARENT_V471_FREEZE) == runner.PARENT_V471_FREEZE_SHA256, "parent v4.7.1 freeze hash drifted")
    v471 = parent_patch.validate_implementation_freeze(runner.PARENT_V471_FREEZE)
    v47 = scientific.validate_implementation_freeze(parent_patch.PARENT_V47_FREEZE)
    return {
        "parent_v4_7_1_freeze_sha256": _sha256(runner.PARENT_V471_FREEZE),
        "parent_v4_7_1_patch_content_sha256": v471["patch_content_sha256"],
        "parent_v4_7_freeze_sha256": _sha256(parent_patch.PARENT_V47_FREEZE),
        "parent_v4_7_scientific_content_sha256": v47["scientific_content_sha256"],
        "parents_revalidated": True,
        "formal_test_accessed": False,
    }


def _context_preflight() -> dict[str, Any]:
    value = runner.corrected_context_preflight()
    _require(value.get("parent_preregistration_identity_preserved") is True, "parent preregistration identity changed")
    _require(value.get("parent_v4_7_freeze_revalidated") is True, "parent v4.7 freeze was not revalidated")
    _require(value.get("next_job") == "G1", "preflight next job drifted")
    _require(value.get("trainer") == "train_paper_sb3_sumo_v4_7_1.py", "preflight trainer drifted")
    _require(value.get("formal_test_accessed") is False, "context preflight accessed formal test")
    _write(PREFLIGHT, {"schema_version": "topo-scene-v4.7.2.corrected-context-preflight/v1", **value})
    return {**value, "source": _source(PREFLIGHT)}


def _parent_smoke_evidence() -> dict[str, Any]:
    smoke = ROOT / "results_topo_v4_7_1_dev" / "engineering" / "smoke" / "s1"
    detailed = _load(smoke / "paper_evaluation_detailed.json")
    selector = _load(smoke / "selector" / "receipt.json")
    _require(detailed.get("engineering_patch_id") == parent_patch.PATCH_ID, "parent metadata smoke patch id drifted")
    _require(detailed.get("selected_deployment_decoder") == selector.get("selected_deployment_decoder"), "parent metadata smoke decoder binding drifted")
    _require(detailed.get("selected_source_checkpoint_sha256") == selector.get("selected_checkpoint_sha256"), "parent metadata smoke source binding drifted")
    _require(detailed.get("selected_model_policy_class") == selector.get("selected_model_policy_class"), "parent metadata smoke policy binding drifted")
    _require(detailed.get("selected_model_parameter_state_sha256") == selector.get("selected_model_parameter_state_sha256"), "parent metadata smoke tensor binding drifted")
    return {
        "directory": str(smoke.resolve()),
        "paper_evaluation_sha256": _sha256(smoke / "paper_evaluation_detailed.json"),
        "selector_receipt_sha256": _sha256(smoke / "selector" / "receipt.json"),
        "all_four_bindings_match": True,
        "training_logic_reused_unchanged": True,
    }


def main() -> int:
    scientific.validate_contract(scientific.load_contract())
    parent_patch.validate_patch_contract(parent_patch.load_patch_contract())
    runner.validate_patch_contract(runner.load_patch_contract())
    _require(_sha256(runner.DEFAULT_PATCH_CONTRACT) == runner.PATCH_CONTRACT_SHA256, "patch contract file hash drifted")
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=29)
    full = _pytest_evidence(FULL_LOG, minimum_tests=331)
    preregistration = _preregistration_evidence()
    parents = _parent_evidence()
    preflight = _context_preflight()
    smoke = _parent_smoke_evidence()
    files = {}
    for relative in PATCH_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        files[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.7.2.runner-context-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_retry_job": True,
        "formal_test_accessed": False,
        "patch_id": runner.PATCH_ID,
        "single_engineering_change": "preserve_parent_default_preregistration_inside_runner_context",
        "scientific_protocol_changed": False,
        "training_or_metadata_writer_changed": False,
        "scientific_contract_sha256": scientific.contract_sha256(),
        "parent_v4_7_1_freeze_sha256": _sha256(runner.PARENT_V471_FREEZE),
        "patch_contract_sha256": _sha256(runner.DEFAULT_PATCH_CONTRACT),
        "patch_plan_sha256": runner.PATCH_PLAN_SHA256,
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "patch_files": files,
        "patch_content_sha256": _canonical_sha256(files),
        "engineering_evidence": {
            "preregistration": preregistration,
            "parents": parents,
            "corrected_context_preflight": preflight,
            "reused_v4_7_1_real_sumo_smoke": smoke,
            "targeted_tests": targeted,
            "full_regression": full,
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
        "schema_version": "topo-scene-v4.7.2.runner-context-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "scientific_contract_sha256": payload["scientific_contract_sha256"],
        "patch_contract_sha256": payload["patch_contract_sha256"],
        "patch_content_sha256": payload["patch_content_sha256"],
        "targeted_tests": targeted,
        "full_regression": full,
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
