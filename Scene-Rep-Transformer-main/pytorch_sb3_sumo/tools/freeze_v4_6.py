"""Validate v4.6 engineering evidence and seal its implementation manifest."""

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

from tools import run_topo_v4_6_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_6_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
SMOKE = ENGINEERING / "smoke" / "s1"
PREFLIGHT = ENGINEERING / "preflight"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
MIGRATION = ENGINEERING / "logs" / "migration.json"

SCIENTIFIC_FILES = (
    "envs/sumo/decision_alignment_v4.py",
    "envs/sumo/paper_env_v4.py",
    "algos/sb3_torch/hybrid_policy_v4.py",
    "algos/sb3_torch/hybrid_policy_v4_3.py",
    "algos/sb3_torch/hybrid_policy_v4_5.py",
    "algos/sb3_torch/sac_v4_2.py",
    "algos/sb3_torch/sac_v4_3.py",
    "algos/sb3_torch/sac_v4_5.py",
    "configs/sb3_configs_v4_3.py",
    "configs/sb3_configs_v4_4.py",
    "configs/sb3_configs_v4_5.py",
    "configs/sb3_configs_v4_6.py",
    "tools/action_diagnostics_v4_2.py",
    "tools/action_diagnostics_v4_3.py",
    "tools/action_diagnostics_v4_5.py",
    "tools/action_diagnostics_v4_6.py",
    "tools/checkpoint_selector_v4_4.py",
    "tools/checkpoint_selector_v4_5_1.py",
    "tools/checkpoint_decoder_selector_v4_6.py",
    "tools/train_sb3_v4_5.py",
    "tools/train_paper_sb3_sumo_v4_5.py",
    "tools/train_sb3_v4_6.py",
    "tools/train_paper_sb3_sumo_v4_6.py",
    "tools/run_topo_v4_6_experiments.py",
    "tools/freeze_v4_6.py",
    "tools/attribute_v4_5_e1_failure.py",
    "tests_sb3_sumo/test_checkpoint_decoder_selector_v4_6.py",
    "tests_sb3_sumo/test_v4_6_adapter.py",
    "tests_sb3_sumo/test_topo_v4_6_experiments.py",
    "experiments/topo_scene_v4/V4_6_JOINT_CHECKPOINT_DECODER_SELECTOR.md",
    "experiments/topo_scene_v4/experiment_contract_v4_6.yaml",
    "results_topo_v4_6_dev/preregistration_receipt.json",
    "results_topo_v4_5_dev/attribution/e1_failure/attribution_summary.json",
    "results_topo_v4_5_dev/attribution/e1_failure/deep_attribution.md",
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
        **_source(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _migration_evidence() -> dict[str, Any]:
    parent_freeze_path = ROOT / "results_topo_v4_5_dev" / "engineering" / "implementation_freeze.json"
    patch_freeze_path = ROOT / "results_topo_v4_5_dev" / "engineering" / "v4_5_1" / "implementation_freeze.json"
    _require(_sha256(parent_freeze_path) == runner.LINEAGE_HASHES["parent_implementation_freeze_sha256"], "parent freeze hash drifted")
    _require(_sha256(patch_freeze_path) == runner.LINEAGE_HASHES["parent_engineering_patch_freeze_sha256"], "parent patch freeze hash drifted")
    parent = _load(parent_freeze_path)
    patch = _load(patch_freeze_path)
    checked: dict[str, str] = {}
    for mapping in (parent["scientific_files"], patch["patch_files"]):
        for relative, expected in mapping.items():
            path = ROOT / relative
            _require(path.is_file(), f"parent frozen file missing: {relative}")
            _require(_sha256(path) == expected, f"parent frozen file drifted: {relative}")
            checked[relative] = expected
    value = {
        "schema_version": "topo-scene-v4.6.parent-migration/v1",
        "parent_v4_5_files_unchanged": True,
        "parent_v4_5_1_patch_files_unchanged": True,
        "checked_unique_files": len(checked),
        "parent_implementation_freeze_sha256": _sha256(parent_freeze_path),
        "parent_patch_freeze_sha256": _sha256(patch_freeze_path),
        "formal_test_accessed": False,
    }
    _write(MIGRATION, value)
    return {**value, "source": _source(MIGRATION)}


def _preflight_evidence() -> list[dict[str, Any]]:
    expected = {
        "candidate_cross": ("topo_v4_6_joint_checkpoint_decoder_selector", "cross"),
        "candidate_carla": ("topo_v4_6_joint_checkpoint_decoder_selector", "carla"),
        "candidate_roundabout_medium": ("topo_v4_6_joint_checkpoint_decoder_selector", "roundabout_medium"),
        "baseline_cross": ("temporal_graph_v1_control", "cross"),
    }
    output = []
    for directory, (algorithm, scenario) in expected.items():
        path = PREFLIGHT / directory / "check_result.json"
        value = _load(path)
        _require(value.get("status") == "ok", f"preflight failed: {directory}")
        _require(value.get("algorithm") == algorithm and value.get("scenario") == scenario, f"preflight identity drifted: {directory}")
        _require(value.get("action_finite") is True, f"preflight action is nonfinite: {directory}")
        _require(value.get("checkpoint_selector_constructed_validation_env") is False, f"preflight touched validation: {directory}")
        if algorithm.startswith("topo_v4_6"):
            _require(value.get("exact_hybrid_lane_code") is True, f"candidate preflight lane code drifted: {directory}")
            _require(value.get("checkpoint_decoder_candidate_pairs") == 4, f"candidate preflight selector drifted: {directory}")
        output.append({"directory": str(path.parent.resolve()), "algorithm": algorithm, "scenario": scenario, "check_result_sha256": _sha256(path)})
    return output


def _smoke_evidence() -> dict[str, Any]:
    arguments = _load(SMOKE / "arguments.json")
    training = _load(SMOKE / "training_diagnostics.json")
    selector = _load(SMOKE / "selector" / "receipt.json")
    actions = _load(SMOKE / "action_diagnostics.json")
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    _require(arguments.get("schema_version") == "topo-scene-v4.6.run-arguments/v1", "smoke argument schema drifted")
    _require(training.get("raw_steps") == 60 and int(training.get("learner_updates", 0)) > 0, "smoke training did not run")
    _require(selector.get("candidate_count") == 4, "smoke did not expose four deployment candidates")
    _require({row["deployment_decoder"] for row in selector["candidates"]} == {"target_critic", "fusion_0_90"}, "smoke decoder candidates drifted")
    _require(all(row.get("decoder_integrity_passed") is True for row in selector["candidates"]), "smoke calibration decoder integrity failed")
    _require(selector.get("training_best_missing_fallback_used") is True, "short smoke did not exercise exact-final fallback")
    _require(selector.get("selected_deployment_decoder") == "target_critic", "smoke deterministic tie decoder drifted")
    _require(selector.get("selected_model_policy_class") == "TargetCriticDecisionAlignedSACPolicyV43", "smoke selected policy class drifted")
    _require(selector.get("policy_parameter_state_preserved") is True, "smoke deployment materialization changed weights")
    _require(actions.get("selected_deployment_decoder") == "target_critic", "smoke validation decoder drifted")
    _require(actions.get("selected_decoder_exact_rule_match_rate") == 1.0, "smoke target rule mismatch")
    _require(actions.get("target_keep_tie_rule_match_rate") == 1.0, "smoke keep tie mismatch")
    sealed = datetime.fromisoformat(selector["sealed_at_utc"])
    constructed = datetime.fromisoformat(detailed["validation_environment_constructed_at_utc"])
    _require(sealed <= constructed, "smoke validation environment predates receipt")
    _verify_zip(SMOKE / "final_model.zip")
    _verify_zip(SMOKE / "selected_model.zip")
    return {
        "directory": str(SMOKE.resolve()),
        "training_diagnostics_sha256": _sha256(SMOKE / "training_diagnostics.json"),
        "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
        "selected_model_sha256": _sha256(SMOKE / "selected_model.zip"),
        "action_diagnostics_sha256": _sha256(SMOKE / "action_diagnostics.json"),
        "paper_evaluation_sha256": _sha256(SMOKE / "paper_evaluation_detailed.json"),
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "policy_parameter_state_preserved": True,
        "receipt_predates_validation": True,
    }


def main() -> int:
    contract = runner.validate_contract(runner.load_contract())
    del contract
    stage0 = runner._validate_stage0(runner.DEFAULT_STAGE0_RESULTS, runner.DEFAULT_STAGE0_ATTRIBUTION)
    attribution = runner._validate_failure_attribution(runner.DEFAULT_FAILURE_ATTRIBUTION, runner.DEFAULT_DEEP_ATTRIBUTION)
    prereg = _load(runner.DEFAULT_PREREGISTRATION)
    _require(prereg.get("status") == "sealed_before_implementation_and_fresh_development", "preregistration status drifted")
    _require(prereg.get("formal_test_accessed") is False, "preregistration accessed formal test")
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=14)
    full = _pytest_evidence(FULL_LOG, minimum_tests=280)
    migration = _migration_evidence()
    preflights = _preflight_evidence()
    smoke = _smoke_evidence()
    scientific = {}
    for relative in SCIENTIFIC_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"scientific file missing: {relative}")
        scientific[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.6.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_job": True,
        "formal_test_accessed": False,
        "single_scientific_change": "train_only_joint_checkpoint_decoder_selector",
        "training_model_reward_and_decoder_definitions_unchanged": True,
        "experiment_contract_sha256": _sha256(runner.DEFAULT_CONTRACT),
        **stage0,
        **attribution,
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "scientific_files": scientific,
        "scientific_content_sha256": _canonical_sha256(scientific),
        "engineering_evidence": {
            "migration": migration,
            "targeted_tests": targeted,
            "full_regression": full,
            "real_sumo_preflights": preflights,
            "joint_selector_training_smoke": smoke,
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
        "schema_version": "topo-scene-v4.6.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "scientific_content_sha256": payload["scientific_content_sha256"],
        "targeted_tests": targeted,
        "full_regression": full,
        "smoke_selector_receipt_sha256": smoke["selector_receipt_sha256"],
    }
    _write(RECEIPT, receipt)
    print(json.dumps({
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "scientific_content_sha256": payload["scientific_content_sha256"],
        "engineering_receipt": str(RECEIPT.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
