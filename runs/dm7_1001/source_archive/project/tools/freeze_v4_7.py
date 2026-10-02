"""Validate v4.7 engineering evidence and seal its implementation manifest."""

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

from configs.sb3_configs_v4_7 import (
    PARENT_CONTROL,
    TEMPORAL_GRAPH,
    V47_CANDIDATE,
)
from tools import run_topo_v4_7_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_7_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
SMOKE = ENGINEERING / "smoke" / "s1"
PREFLIGHT = ENGINEERING / "preflight"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
MIGRATION = ENGINEERING / "logs" / "migration.json"

SCIENTIFIC_FILES = (
    "algos/sb3_torch/replay_buffer_v4_7.py",
    "configs/sb3_configs_v4_7.py",
    "tools/train_sb3_v4_7.py",
    "tools/train_paper_sb3_sumo_v4_7.py",
    "tools/run_topo_v4_7_experiments.py",
    "tools/freeze_v4_7.py",
    "tests_sb3_sumo/test_replay_buffer_v4_7.py",
    "tests_sb3_sumo/test_v4_7_adapter.py",
    "tests_sb3_sumo/test_topo_v4_7_experiments.py",
    "experiments/topo_scene_v4/V4_7_HORIZON_CORRECT_16_STEP_CREDIT.md",
    "experiments/topo_scene_v4/experiment_contract_v4_7.yaml",
    "results_topo_v4_7_dev/preregistration_receipt.json",
    "results_topo_v4_6_dev/attribution/f1_failure/attribution_summary.json",
    "results_topo_v4_6_dev/attribution/f1_failure/deep_attribution.md",
    "tools/attribute_v4_6_f1_failure.py",
    "tools/replay_v4_6_pair_posthoc.py",
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
        **_source(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _migration_evidence() -> dict[str, Any]:
    parent_path = ROOT / "results_topo_v4_6_dev" / "engineering" / "implementation_freeze.json"
    _require(
        _sha256(parent_path)
        == runner.LINEAGE_HASHES["parent_implementation_freeze_sha256"],
        "parent freeze hash drifted",
    )
    parent = _load(parent_path)
    _require(
        parent.get("schema_version")
        == "topo-scene-v4.6.implementation-freeze/v1",
        "parent freeze schema drifted",
    )
    checked: dict[str, str] = {}
    for relative, expected in parent["scientific_files"].items():
        path = ROOT / relative
        _require(path.is_file(), f"parent frozen file missing: {relative}")
        _require(_sha256(path) == expected, f"parent frozen file drifted: {relative}")
        checked[relative] = expected
    value = {
        "schema_version": "topo-scene-v4.7.parent-migration/v1",
        "parent_v4_6_files_unchanged": True,
        "checked_unique_files": len(checked),
        "parent_implementation_freeze_sha256": _sha256(parent_path),
        "formal_test_accessed": False,
    }
    _write(MIGRATION, value)
    return {**value, "source": _source(MIGRATION)}


def _preregistration_evidence() -> dict[str, Any]:
    receipt = _load(runner.DEFAULT_PREREGISTRATION)
    _require(
        receipt.get("schema_version")
        == "topo-scene-v4.7.preregistration-receipt/v1",
        "preregistration schema drifted",
    )
    _require(receipt.get("preregistered_before_implementation") is True, "implementation was not preregistered")
    _require(receipt.get("preregistered_before_fresh_development") is True, "development was not preregistered")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("experiment_contract_sha256") == _sha256(runner.DEFAULT_CONTRACT), "preregistration contract hash drifted")
    planned = receipt.get("planned_files_absent_at_seal", {})
    _require(set(planned) == set(SCIENTIFIC_FILES[:9]), "preregistered implementation file set drifted")
    _require(all(value is True for value in planned.values()), "planned file absence receipt drifted")
    return {**_source(runner.DEFAULT_PREREGISTRATION), "planned_files_absent_at_seal": len(planned)}


def _preflight_evidence() -> list[dict[str, Any]]:
    expected = {
        "candidate_cross": (V47_CANDIDATE, "cross", 16, True, 4),
        "candidate_carla": (V47_CANDIDATE, "carla", 16, True, 4),
        "candidate_roundabout_medium": (V47_CANDIDATE, "roundabout_medium", 16, True, 4),
        "parent_cross": (PARENT_CONTROL, "cross", 4, False, 4),
        "temporal_cross": (TEMPORAL_GRAPH, "cross", 4, False, 2),
    }
    output = []
    for directory, (algorithm, scenario, n_step, changed, pairs) in expected.items():
        path = PREFLIGHT / directory / "check_result.json"
        value = _load(path)
        _require(value.get("status") == "ok", f"preflight failed: {directory}")
        _require(value.get("algorithm") == algorithm and value.get("scenario") == scenario, f"preflight identity drifted: {directory}")
        _require(value.get("action_finite") is True, f"preflight action is nonfinite: {directory}")
        _require(value.get("checkpoint_selector_constructed_validation_env") is False, f"preflight touched validation: {directory}")
        diagnostics = value.get("return_estimator", {})
        _require(diagnostics.get("n_step") == n_step, f"preflight n-step drifted: {directory}")
        _require(diagnostics.get("return_estimator_changed") is changed, f"preflight return role drifted: {directory}")
        if algorithm != TEMPORAL_GRAPH:
            _require(value.get("exact_hybrid_lane_code") is True, f"joint preflight lane code drifted: {directory}")
            _require(value.get("checkpoint_decoder_candidate_pairs") == pairs, f"joint preflight selector drifted: {directory}")
        output.append(
            {
                "directory": str(path.parent.resolve()),
                "algorithm": algorithm,
                "scenario": scenario,
                "n_step": n_step,
                "return_estimator_changed": changed,
                "check_result_sha256": _sha256(path),
            }
        )
    return output


def _smoke_evidence() -> dict[str, Any]:
    arguments = _load(SMOKE / "arguments.json")
    training = _load(SMOKE / "training_diagnostics.json")
    returns = _load(SMOKE / "return_estimator_diagnostics.json")
    selector = _load(SMOKE / "selector" / "receipt.json")
    actions = _load(SMOKE / "action_diagnostics.json")
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    _require(arguments.get("schema_version") == "topo-scene-v4.7.run-arguments/v1", "smoke argument schema drifted")
    _require(training.get("raw_steps") == 120 and int(training.get("learner_updates", 0)) > 0, "smoke training did not run")
    _require(training.get("return_estimator") == returns, "smoke return artifacts disagree")
    _require(returns.get("n_step") == 16 and returns.get("horizon_correct_bootstrap") is True, "smoke return estimator drifted")
    _require(returns.get("bootstrap_discount") == "gamma_power_actual_horizon", "smoke bootstrap discount drifted")
    _require(int(returns.get("sampled_horizon_count", 0)) > 0, "smoke sampled no horizons")
    _require(1.0 <= float(returns["mean_sampled_actual_horizon"]) <= 16.0, "smoke mean horizon invalid")
    _require(0.0 <= float(returns["short_horizon_sample_rate"]) <= 1.0, "smoke short-horizon rate invalid")
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
    _verify_zip(SMOKE / "final_model.zip"); _verify_zip(SMOKE / "selected_model.zip")
    return {
        "directory": str(SMOKE.resolve()),
        "training_diagnostics_sha256": _sha256(SMOKE / "training_diagnostics.json"),
        "return_estimator_diagnostics_sha256": _sha256(SMOKE / "return_estimator_diagnostics.json"),
        "sampled_horizon_count": returns["sampled_horizon_count"],
        "mean_sampled_actual_horizon": returns["mean_sampled_actual_horizon"],
        "short_horizon_sample_rate": returns["short_horizon_sample_rate"],
        "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
        "selected_model_sha256": _sha256(SMOKE / "selected_model.zip"),
        "action_diagnostics_sha256": _sha256(SMOKE / "action_diagnostics.json"),
        "paper_evaluation_sha256": _sha256(SMOKE / "paper_evaluation_detailed.json"),
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "policy_parameter_state_preserved": True,
        "receipt_predates_validation": True,
    }


def main() -> int:
    runner.validate_contract(runner.load_contract())
    stage0 = runner.parent._validate_stage0(
        runner.DEFAULT_STAGE0_RESULTS, runner.DEFAULT_STAGE0_ATTRIBUTION
    )
    attribution = runner._validate_failure_attribution(
        runner.DEFAULT_FAILURE_ATTRIBUTION, runner.DEFAULT_DEEP_ATTRIBUTION
    )
    preregistration = _preregistration_evidence()
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=18)
    full = _pytest_evidence(FULL_LOG, minimum_tests=315)
    migration = _migration_evidence()
    preflights = _preflight_evidence()
    smoke = _smoke_evidence()
    scientific = {}
    for relative in SCIENTIFIC_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"scientific file missing: {relative}")
        scientific[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.7.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_job": True,
        "formal_test_accessed": False,
        "single_scientific_change": "horizon_correct_16_step_terminal_credit",
        "reward_network_entropy_selector_and_decoder_unchanged": True,
        "candidate_n_step": 16,
        "candidate_bootstrap_discount": "gamma_power_actual_horizon",
        "parent_control_n_step": 4,
        "parent_control_bootstrap_discount": "single_gamma_source_equivalent",
        "experiment_contract_sha256": _sha256(runner.DEFAULT_CONTRACT),
        **stage0,
        **attribution,
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "scientific_files": scientific,
        "scientific_content_sha256": _canonical_sha256(scientific),
        "engineering_evidence": {
            "migration": migration,
            "preregistration": preregistration,
            "targeted_tests": targeted,
            "full_regression": full,
            "real_sumo_preflights": preflights,
            "horizon_correct_training_smoke": smoke,
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
        "schema_version": "topo-scene-v4.7.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "scientific_content_sha256": payload["scientific_content_sha256"],
        "targeted_tests": targeted,
        "full_regression": full,
        "smoke_return_estimator_diagnostics_sha256": smoke[
            "return_estimator_diagnostics_sha256"
        ],
        "smoke_selector_receipt_sha256": smoke["selector_receipt_sha256"],
    }
    _write(RECEIPT, receipt)
    print(
        json.dumps(
            {
                "implementation_freeze": str(FREEZE.resolve()),
                "implementation_freeze_sha256": freeze_hash,
                "scientific_content_sha256": payload["scientific_content_sha256"],
                "engineering_receipt": str(RECEIPT.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
