"""Seal the v4.8.2 selector-gate interface engineering patch."""

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

from tools import run_topo_v4_8_2_experiments as runner
from tools import run_topo_v4_8_1_experiments as acceptance_patch
from tools import run_topo_v4_8_experiments as v48


ENGINEERING = ROOT / "results_topo_v4_8_2_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
PREFLIGHT = ENGINEERING / "h1_adoption_preflight.json"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"

PATCH_FILES = (
    "tools/run_topo_v4_8_1_experiments.py",
    "tools/run_topo_v4_8_2_experiments.py",
    "tools/freeze_v4_8_2.py",
    "tests_sb3_sumo/test_topo_v4_8_2_experiments.py",
    "experiments/topo_scene_v4/V4_8_1_RUNTIME_DIAGNOSTIC_ACCEPTANCE_PATCH.md",
    "experiments/topo_scene_v4/experiment_contract_v4_8_1.yaml",
    "results_topo_v4_8_1_dev/preregistration_receipt.json",
    "results_topo_v4_8_1_dev/engineering/preflight_failure.json",
    "experiments/topo_scene_v4/V4_8_2_SELECTOR_GATE_INTERFACE_PATCH.md",
    "experiments/topo_scene_v4/experiment_contract_v4_8_2.yaml",
    "results_topo_v4_8_2_dev/preregistration_receipt.json",
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
    _require(receipt.get("schema_version") == "topo-scene-v4.8.2.selector-gate-interface-preregistration/v1", "preregistration schema drifted")
    _require(receipt.get("created_before_patch_implementation") is True, "patch was not preregistered")
    _require(receipt.get("created_after_v4_8_1_preflight_failure") is True, "v4.8.1 trigger missing")
    _require(receipt.get("scientific_protocol_changed") is False, "preregistration changed science")
    _require(receipt.get("h1_will_not_be_rerun") is True, "H1 rerun was not forbidden")
    _require(receipt.get("h1_artifacts_will_not_be_modified") is True, "H1 rewrite was not forbidden")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("parent_v4_8_freeze_sha256") == runner.PARENT_V48_FREEZE_SHA256, "preregistration parent freeze drifted")
    _require(receipt.get("parent_v4_8_1_runner_sha256") == runner.V481_RUNNER_SHA256, "preregistration acceptance adapter drifted")
    planned = receipt.get("planned_patch_files_absent_at_receipt", {})
    _require(set(planned) == {
        "tools/run_topo_v4_8_2_experiments.py",
        "tools/freeze_v4_8_2.py",
        "tests_sb3_sumo/test_topo_v4_8_2_experiments.py",
    }, "planned patch file set drifted")
    _require(all(value is True for value in planned.values()), "planned absence evidence drifted")
    return {**_source(runner.DEFAULT_PREREGISTRATION), "planned_files_absent_at_seal": len(planned)}


def _v481_failure_evidence() -> dict[str, Any]:
    value = _load(runner.V481_FAILURE)
    _require(value.get("schema_version") == "topo-scene-v4.8.1.engineering-preflight-failure/v1", "v4.8.1 failure schema drifted")
    _require(value.get("status") == "failed", "v4.8.1 failure status drifted")
    _require(value.get("scientific_job_started_under_patch") is False, "v4.8.1 started science")
    _require(value.get("h1_rerun") is False and value.get("h1_artifacts_modified") is False, "v4.8.1 changed H1")
    _require(value.get("formal_test_accessed") is False, "v4.8.1 accessed formal test")
    _require(value.get("preflight", {}).get("patched_acceptance") is True, "v4.8.1 acceptance did not pass")
    _require(value.get("preflight", {}).get("failed_gate_checks") == ["selector_mode"], "v4.8.1 gate failure drifted")
    _require(value.get("patch_runner_sha256") == _sha256(ROOT / "tools" / "run_topo_v4_8_1_experiments.py"), "v4.8.1 runner hash drifted")
    return _source(runner.V481_FAILURE)


def _parent_evidence() -> dict[str, Any]:
    _require(_sha256(runner.PARENT_V48_FREEZE) == runner.PARENT_V48_FREEZE_SHA256, "parent v4.8 freeze drifted")
    parent = v48.validate_implementation_freeze(runner.PARENT_V48_FREEZE)
    _require(parent.get("scientific_content_sha256") == "0ef153763770384c9bacae780db0b6ebf19de25320a61aa976b11c3d5300aee6", "parent scientific content drifted")
    _require(_sha256(ROOT / "tools" / "run_topo_v4_8_1_experiments.py") == runner.V481_RUNNER_SHA256, "v4.8.1 acceptance adapter changed")
    return {
        **_source(runner.PARENT_V48_FREEZE),
        "scientific_content_sha256": parent["scientific_content_sha256"],
        "v4_8_1_runner_sha256": runner.V481_RUNNER_SHA256,
        "formal_test_accessed": False,
    }


def main() -> int:
    v48.validate_contract(v48.load_contract())
    acceptance_patch.validate_patch_contract(acceptance_patch.load_patch_contract())
    runner.validate_patch_contract(runner.load_patch_contract())
    parent = _parent_evidence()
    preregistration = _preregistration_evidence()
    failure = _v481_failure_evidence()
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=4)
    full = _pytest_evidence(FULL_LOG, minimum_tests=300)
    preflight = runner.h1_adoption_preflight()
    _write(PREFLIGHT, {"schema_version": "topo-scene-v4.8.2.h1-adoption-preflight/v1", **preflight})
    before = acceptance_patch._validate_h1_hashes()
    after = acceptance_patch._validate_h1_hashes()
    _require(before == after == preflight["h1_artifact_hashes"], "H1 changed during patch validation")
    files: dict[str, str] = {}
    for relative in PATCH_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        files[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.8.2.selector-gate-patch-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "scientific_job_started_under_v4_8_1": False,
        "h1_rerun": False,
        "h1_artifacts_modified": False,
        "single_engineering_change": "align_candidate_gate_selector_mode_with_frozen_v4_8_selector",
        "old_expected_selector_mode": "joint_checkpoint_decoder",
        "new_expected_selector_mode": "tie_only_replicated_joint_checkpoint_decoder",
        "parent_v4_8_freeze_sha256": runner.PARENT_V48_FREEZE_SHA256,
        "v4_8_1_runner_sha256": runner.V481_RUNNER_SHA256,
        "patch_contract_sha256": _sha256(runner.DEFAULT_PATCH_CONTRACT),
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "patch_files": files,
        "patch_content_sha256": _canonical_sha256(files),
        "h1_artifact_hashes": before,
        "engineering_evidence": {
            "parent": parent,
            "preregistration": preregistration,
            "v4_8_1_preflight_failure": failure,
            "h1_adoption_preflight": {**preflight, "source": _source(PREFLIGHT)},
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
        "schema_version": "topo-scene-v4.8.2.selector-gate-patch-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "scientific_protocol_changed": False,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "parent_v4_8_freeze_sha256": runner.PARENT_V48_FREEZE_SHA256,
        "patch_content_sha256": payload["patch_content_sha256"],
        "h1_adopted_in_place": True,
        "h1_rerun": False,
        "h1_artifacts_modified": False,
        "next_development_job": "H2",
        "targeted_tests": targeted,
        "full_regression": full,
    }
    _write(RECEIPT, receipt)
    print(json.dumps({
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "patch_content_sha256": payload["patch_content_sha256"],
        "engineering_receipt": str(RECEIPT.resolve()),
        "h1_adopted_in_place": True,
        "next_development_job": "H2",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
