"""Seal the v4.8.3 control secondary-seed acceptance patch."""

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

from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_2_experiments as parent_patch
from tools import run_topo_v4_8_3_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_8_3_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
INITIAL_FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
SECOND_FULL_LOG = ENGINEERING / "logs" / "full_regression_pass.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression_final.xml"
REGRESSION_EXCEPTION = ENGINEERING / "full_regression_state_exception.json"
PREFLIGHT = ENGINEERING / "promotion_adoption_preflight.json"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"

PATCH_FILES = (
    "tools/run_topo_v4_8_3_experiments.py",
    "tools/freeze_v4_8_3.py",
    "tests_sb3_sumo/test_topo_v4_8_3_experiments.py",
    "experiments/topo_scene_v4/V4_8_3_CONTROL_SECONDARY_SEED_ACCEPTANCE_PATCH.md",
    "experiments/topo_scene_v4/experiment_contract_v4_8_3.yaml",
    "results_topo_v4_8_3_dev/preregistration_receipt.json",
    "results_topo_v4_8_3_dev/engineering/full_regression_state_exception.json",
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


def _source(path: Path) -> dict[str, Any]:
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
    _require(
        receipt.get("schema_version")
        == "topo-scene-v4.8.3.control-secondary-seed-acceptance-preregistration/v1",
        "preregistration schema drifted",
    )
    _require(receipt.get("created_before_patch_implementation") is True, "patch was not preregistered")
    _require(receipt.get("created_after_completed_promotion_acceptance_failure") is True, "promotion trigger missing")
    _require(receipt.get("scientific_protocol_changed") is False, "preregistration changed science")
    _require(receipt.get("completed_promotion_job_will_not_be_rerun") is True, "promotion rerun was not forbidden")
    _require(receipt.get("completed_promotion_artifacts_will_not_be_modified") is True, "promotion rewrite was not forbidden")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("parent_v4_8_freeze_sha256") == runner.PARENT_V48_FREEZE_SHA256, "preregistration parent science drifted")
    _require(receipt.get("parent_v4_8_2_runner_sha256") == runner.PARENT_V482_RUNNER_SHA256, "preregistration parent runner drifted")
    _require(receipt.get("parent_v4_8_2_freeze_sha256") == runner.PARENT_V482_FREEZE_SHA256, "preregistration parent freeze drifted")
    _require(receipt.get("trigger_job") == runner.PROMOTION_JOB_NAME, "preregistration trigger job drifted")
    _require(receipt.get("trigger_acceptance_reason") == "argument secondary_calibration_seed_start mismatch", "preregistration reason drifted")
    planned = receipt.get("planned_patch_files_absent_at_receipt", {})
    _require(set(planned) == {
        "tools/run_topo_v4_8_3_experiments.py",
        "tools/freeze_v4_8_3.py",
        "tests_sb3_sumo/test_topo_v4_8_3_experiments.py",
    }, "planned patch file set drifted")
    _require(all(value is True for value in planned.values()), "planned absence evidence drifted")
    return {**_source(runner.DEFAULT_PREREGISTRATION), "planned_files_absent_at_seal": len(planned)}


def _regression_exception_evidence() -> dict[str, Any]:
    record = _load(REGRESSION_EXCEPTION)
    _require(
        record.get("schema_version")
        == "topo-scene-v4.8.3.full-regression-state-exception/v1",
        "regression exception schema drifted",
    )
    _require(record.get("status") == "corrected_and_retested", "regression exception is unresolved")
    _require(record.get("scientific_protocol_changed") is False, "regression correction changed science")
    _require(record.get("formal_test_accessed") is False, "regression correction accessed formal test")
    _require(record.get("initial_junit_sha256") == _sha256(INITIAL_FULL_LOG), "initial regression log drifted")
    root = ET.parse(INITIAL_FULL_LOG).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    _require((tests, failures, errors) == (357, 1, 50), "initial regression counts drifted")
    xml_text = INITIAL_FULL_LOG.read_text(encoding="utf-8")
    _require("PermissionError" in xml_text, "initial tmp-path permission evidence missing")
    _require("patched development continuation drifted" in xml_text, "historical state failure evidence missing")
    _require(record.get("second_junit_sha256") == _sha256(SECOND_FULL_LOG), "second regression log drifted")
    second_root = ET.parse(SECOND_FULL_LOG).getroot()
    second_suites = [second_root] if second_root.tag == "testsuite" else list(second_root.findall("testsuite"))
    second_tests = sum(int(suite.attrib.get("tests", 0)) for suite in second_suites)
    second_failures = sum(int(suite.attrib.get("failures", 0)) for suite in second_suites)
    second_errors = sum(int(suite.attrib.get("errors", 0)) for suite in second_suites)
    _require((second_tests, second_failures, second_errors) == (357, 3, 0), "second regression counts drifted")
    _require("WinError 206" in SECOND_FULL_LOG.read_text(encoding="utf-8"), "path-length evidence missing")
    _require(record.get("tmp_path_permission_error_count") == 50, "permission error count drifted")
    _require(record.get("historical_state_assertion_failure_count") == 1, "state failure count drifted")
    _require(
        record.get("deselected_nodeid")
        == "tests_sb3_sumo/test_topo_v4_8_2_experiments.py::test_v4_8_2_h1_preflight_reproduces_both_bugs_and_adopts_in_place",
        "deselected node drifted",
    )
    _require(
        record.get("replacement_nodeid")
        == "tests_sb3_sumo/test_topo_v4_8_3_experiments.py::test_v4_8_3_completed_development_supersedes_h1_only_snapshot",
        "replacement node drifted",
    )
    _require(record.get("workspace_basetemp_required") is True, "workspace basetemp is not required")
    _require(record.get("executed_test_count_not_reduced") is True, "test count was reduced")
    _require(record.get("scientific_or_mechanism_test_removed") is False, "scientific test was removed")
    _require(record.get("final_junit_sha256") == _sha256(FULL_LOG), "final regression log drifted")
    _require(record.get("final_tests") == 357 and record.get("final_passed") == 357, "final regression counts drifted")
    _require(record.get("final_failures") == 0 and record.get("final_errors") == 0, "final regression did not pass")
    return {
        **_source(REGRESSION_EXCEPTION),
        "initial_junit": _source(INITIAL_FULL_LOG),
        "initial_tests": tests,
        "initial_failures": failures,
        "initial_errors": errors,
        "second_junit": _source(SECOND_FULL_LOG),
        "second_tests": second_tests,
        "second_failures": second_failures,
        "second_errors": second_errors,
        "deselected_nodeid": record["deselected_nodeid"],
        "replacement_nodeid": record["replacement_nodeid"],
        "final_junit": _source(FULL_LOG),
        "formal_test_accessed": False,
    }


def _trigger_evidence() -> dict[str, Any]:
    _require(_sha256(runner.PROMOTION_LAST_EXECUTION) == runner.PROMOTION_LAST_EXECUTION_TRIGGER_SHA256, "promotion trigger receipt drifted")
    execution = _load(runner.PROMOTION_LAST_EXECUTION)
    _require(execution.get("stage") == "promotion", "promotion trigger stage drifted")
    _require(execution.get("selected") == [runner.PROMOTION_JOB_NAME], "promotion trigger selection drifted")
    failures = execution.get("failures", [])
    _require(len(failures) == 1, "promotion trigger failure count drifted")
    _require(failures[0].get("job") == runner.PROMOTION_JOB_NAME, "promotion failure job drifted")
    _require(failures[0].get("returncode") == 0, "promotion subprocess did not succeed")
    _require(failures[0].get("acceptance_reason") == "argument secondary_calibration_seed_start mismatch", "promotion reason drifted")
    arguments = _load(runner.PROMOTION_RUN / "arguments.json")
    requested = arguments.get("requested_raw_steps", {})
    _require("secondary_calibration_seed_start" not in requested, "trigger control field is no longer omitted")
    _require(arguments.get("implementation_fidelity", {}).get("secondary_calibration_seed_offset") is None, "trigger fidelity offset drifted")
    selector = _load(runner.PROMOTION_RUN / "selector" / "receipt.json")
    _require(selector.get("selector_mode") == "checkpoint_only_parent_control", "trigger selector mode drifted")
    _require(not (runner.PROMOTION_RUN / "selector" / "cal_secondary").exists(), "trigger secondary directory exists")
    return {
        **_source(runner.PROMOTION_LAST_EXECUTION),
        "subprocess_returncode": 0,
        "acceptance_reason": failures[0]["acceptance_reason"],
        "requested_secondary_seed_field_present": False,
        "requested_secondary_seed_value": None,
        "formal_test_accessed": False,
    }


def _parent_evidence() -> dict[str, Any]:
    _require(_sha256(runner.PARENT_V48_FREEZE) == runner.PARENT_V48_FREEZE_SHA256, "parent v4.8 freeze drifted")
    _require(_sha256(ROOT / "tools" / "run_topo_v4_8_2_experiments.py") == runner.PARENT_V482_RUNNER_SHA256, "parent v4.8.2 runner drifted")
    _require(_sha256(parent_patch.DEFAULT_FREEZE) == runner.PARENT_V482_FREEZE_SHA256, "parent v4.8.2 freeze drifted")
    _require(_sha256(parent_patch.DEFAULT_ENGINEERING_RECEIPT) == runner.PARENT_V482_RECEIPT_SHA256, "parent v4.8.2 receipt drifted")
    parent_patch.validate_implementation_freeze(parent_patch.DEFAULT_FREEZE)
    parent = v48.validate_implementation_freeze(runner.PARENT_V48_FREEZE)
    return {
        **_source(parent_patch.DEFAULT_FREEZE),
        "parent_v4_8_freeze_sha256": runner.PARENT_V48_FREEZE_SHA256,
        "scientific_content_sha256": parent["scientific_content_sha256"],
        "parent_v4_8_2_runner_sha256": runner.PARENT_V482_RUNNER_SHA256,
        "formal_test_accessed": False,
    }


def main() -> int:
    v48.validate_contract(v48.load_contract())
    parent_patch.validate_patch_contract(parent_patch.load_patch_contract())
    runner.validate_patch_contract(runner.load_patch_contract())
    parent = _parent_evidence()
    preregistration = _preregistration_evidence()
    trigger = _trigger_evidence()
    regression_exception = _regression_exception_evidence()
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=6)
    full = _pytest_evidence(FULL_LOG, minimum_tests=357)
    _require(full["tests"] == 357, "corrected full regression test count drifted")
    before = runner._validate_promotion_artifact_hashes()
    preflight = runner.promotion_adoption_preflight()
    _write(PREFLIGHT, {"schema_version": "topo-scene-v4.8.3.promotion-adoption-preflight/v1", **preflight})
    after = runner._validate_promotion_artifact_hashes()
    _require(before == after == preflight["promotion_artifact_hashes"], "promotion run changed during patch validation")
    files: dict[str, str] = {}
    for relative in PATCH_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        files[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.8.3.control-secondary-seed-patch-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "single_engineering_change": "accept_control_null_secondary_seed_against_control_semantics",
        "acceptance_only": True,
        "control_only": True,
        "read_only_normalized_view": True,
        "candidate_acceptance_changed": False,
        "completed_promotion_job_rerun": False,
        "completed_promotion_artifacts_modified": False,
        "parent_v4_8_freeze_sha256": runner.PARENT_V48_FREEZE_SHA256,
        "parent_v4_8_2_runner_sha256": runner.PARENT_V482_RUNNER_SHA256,
        "parent_v4_8_2_freeze_sha256": runner.PARENT_V482_FREEZE_SHA256,
        "patch_contract_sha256": _sha256(runner.DEFAULT_PATCH_CONTRACT),
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "patch_files": files,
        "patch_content_sha256": _canonical_sha256(files),
        "promotion_artifact_hashes": before,
        "engineering_evidence": {
            "parent": parent,
            "preregistration": preregistration,
            "trigger": trigger,
            "initial_full_regression_and_state_exception": regression_exception,
            "promotion_adoption_preflight": {**preflight, "source": _source(PREFLIGHT)},
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
        "schema_version": "topo-scene-v4.8.3.control-secondary-seed-patch-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "parent_v4_8_freeze_sha256": runner.PARENT_V48_FREEZE_SHA256,
        "parent_v4_8_2_freeze_sha256": runner.PARENT_V482_FREEZE_SHA256,
        "patch_content_sha256": payload["patch_content_sha256"],
        "promotion_job_adopted_in_place": runner.PROMOTION_JOB_NAME,
        "promotion_job_rerun": False,
        "promotion_artifacts_modified": False,
        "next_promotion_job": runner.NEXT_PROMOTION_JOB,
        "targeted_tests": targeted,
        "full_regression": full,
        "full_regression_state_exception": regression_exception,
    }
    _write(RECEIPT, receipt)
    print(json.dumps({
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "patch_content_sha256": payload["patch_content_sha256"],
        "engineering_receipt": str(RECEIPT.resolve()),
        "promotion_job_adopted_in_place": runner.PROMOTION_JOB_NAME,
        "next_promotion_job": runner.NEXT_PROMOTION_JOB,
        "formal_test_accessed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
