"""Validate v4.8 engineering evidence and seal its immutable implementation."""

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

from configs.sb3_configs_v4_8 import (
    TEMPORAL_GRAPH,
    V48_CANDIDATE,
    V48_IMPLEMENTATION_IDS,
)
from tools import run_topo_v4_8_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_8_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression.xml"
SMOKE = ENGINEERING / "smoke" / "s5"
PREFLIGHT = ENGINEERING / "preflight"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
ATTEMPTS = ENGINEERING / "engineering_attempts.json"
PARENT_FREEZE = ROOT / "results_topo_v4_7_2_dev" / "engineering" / "implementation_freeze.json"

SCIENTIFIC_FILES = (
    "configs/sb3_configs_v4_8.py",
    "tools/checkpoint_decoder_selector_v4_8.py",
    "tools/train_sb3_v4_8.py",
    "tools/train_paper_sb3_sumo_v4_8.py",
    "tools/run_topo_v4_8_experiments.py",
    "tools/freeze_v4_8.py",
    "tests_sb3_sumo/test_checkpoint_decoder_selector_v4_8.py",
    "tests_sb3_sumo/test_v4_8_adapter.py",
    "tests_sb3_sumo/test_topo_v4_8_experiments.py",
    "experiments/topo_scene_v4/V4_8_TIE_ONLY_REPLICATED_TRAIN_CALIBRATION.md",
    "experiments/topo_scene_v4/experiment_contract_v4_8.yaml",
    "results_topo_v4_8_dev/preregistration_receipt.json",
    "results_topo_v4_7_2_dev/attribution/g3_failure/attribution_summary.json",
    "results_topo_v4_7_2_dev/attribution/g3_failure/deep_attribution.md",
    "tools/attribute_v4_7_g3_failure.py",
    "tools/replay_v4_7_g3_posthoc.py",
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


def _preregistration_evidence() -> dict[str, Any]:
    receipt = _load(runner.DEFAULT_PREREGISTRATION)
    _require(receipt.get("schema_version") == "topo-scene-v4.8.preregistration-receipt/v1", "preregistration schema drifted")
    _require(receipt.get("created_before_implementation") is True, "implementation was not preregistered")
    _require(receipt.get("created_before_fresh_development") is True, "development was not preregistered")
    _require(receipt.get("formal_test_locked") is True, "formal test was not locked")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal test")
    _require(receipt.get("posthoc_results_counted_for_gate") is False, "post-hoc results counted for gate")
    _require(receipt.get("experiment_contract_sha256") == _sha256(runner.DEFAULT_CONTRACT), "preregistration contract hash drifted")
    _require(receipt.get("failure_attribution_sha256") == runner.LINEAGE_HASHES["failure_attribution_sha256"], "preregistration attribution drifted")
    _require(receipt.get("planned_development_cells") == ["H1", "H2", "H3", "H4"], "development plan drifted")
    return {**_source(runner.DEFAULT_PREREGISTRATION), "created_before_implementation": True}


def _parent_evidence() -> dict[str, Any]:
    _require(_sha256(PARENT_FREEZE) == runner.LINEAGE_HASHES["parent_implementation_freeze_sha256"], "parent freeze hash drifted")
    parent = _load(PARENT_FREEZE)
    _require(parent.get("schema_version") == "topo-scene-v4.7.2.runner-context-freeze/v1", "parent freeze schema drifted")
    for relative, expected in parent["patch_files"].items():
        path = ROOT / relative
        _require(path.is_file() and _sha256(path) == expected, f"parent patch drifted: {relative}")
    return {
        **_source(PARENT_FREEZE),
        "parent_patch_content_sha256": parent["patch_content_sha256"],
        "formal_test_accessed": False,
    }


def _preflight_evidence() -> list[dict[str, Any]]:
    expected = {
        "candidate_cross_v3": (V48_CANDIDATE, "cross", 16, 4),
        "candidate_carla_v3": (V48_CANDIDATE, "carla", 16, 4),
        "candidate_roundabout_medium_v3": (V48_CANDIDATE, "roundabout_medium", 16, 4),
        "temporal_cross_v3": (TEMPORAL_GRAPH, "cross", 4, 2),
    }
    evidence = []
    for directory, (algorithm, scenario, n_step, pairs) in expected.items():
        path = PREFLIGHT / directory / "check_result.json"
        value = _load(path)
        _require(value.get("status") == "ok", f"preflight failed: {directory}")
        _require(value.get("algorithm") == algorithm, f"preflight algorithm drifted: {directory}")
        _require(value.get("scenario") == scenario, f"preflight scenario drifted: {directory}")
        _require(value.get("action_finite") is True, f"preflight action is nonfinite: {directory}")
        _require(value.get("checkpoint_selector_constructed_validation_env") is False, f"preflight touched validation: {directory}")
        returns = value.get("return_estimator", {})
        _require(returns.get("n_step") == n_step, f"preflight n-step drifted: {directory}")
        if algorithm == V48_CANDIDATE:
            _require(value.get("exact_hybrid_lane_code") is True, f"candidate lane code drifted: {directory}")
            _require(value.get("checkpoint_decoder_candidate_pairs") == pairs, f"candidate pair count drifted: {directory}")
            _require(value.get("selector_mode") == "tie_only_replicated_joint_checkpoint_decoder", f"candidate selector mode drifted: {directory}")
        evidence.append({"directory": str(path.parent.resolve()), "algorithm": algorithm, "scenario": scenario, "check_result_sha256": _sha256(path)})
    return evidence


def _smoke_evidence() -> dict[str, Any]:
    arguments = _load(SMOKE / "arguments.json")
    training = _load(SMOKE / "training_diagnostics.json")
    returns = _load(SMOKE / "return_estimator_diagnostics.json")
    selector = _load(SMOKE / "selector" / "receipt.json")
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    actions = _load(SMOKE / "action_diagnostics.json")
    _require(arguments.get("schema_version") == "topo-scene-v4.8.run-arguments/v1", "smoke argument schema drifted")
    _require(arguments["requested_raw_steps"].get("learning_starts") == 60, "smoke warmup drifted")
    _require(training.get("raw_steps") == 120 and int(training.get("learner_updates", 0)) > 0, "smoke training did not run")
    _require(training.get("return_estimator") == returns, "smoke return artifacts disagree")
    _require(returns.get("n_step") == 16 and returns.get("horizon_correct_bootstrap") is True, "smoke return estimator drifted")
    _require(returns.get("bootstrap_discount") == "gamma_power_actual_horizon", "smoke bootstrap drifted")
    _require(int(returns.get("sampled_horizon_count", 0)) > 0, "smoke sampled no horizons")
    _require(selector.get("schema_version") == "topo-scene-v4.8.tie-only-replicated-selector/v1", "smoke selector schema drifted")
    _require(selector.get("candidate_count") == 4, "smoke primary matrix drifted")
    _require(selector.get("initial_empirical_tied_top_count") == 4, "smoke did not exercise tie branch")
    _require(selector.get("secondary_calibration_triggered") is True, "smoke did not trigger secondary calibration")
    _require(selector.get("secondary_calibration_seed_start") == 97_100, "smoke secondary seed drifted")
    _require(len(selector.get("secondary_candidates", [])) == 4, "smoke secondary matrix drifted")
    _require(selector.get("secondary_candidate_scope") == "empirical_tied_top_only", "smoke secondary scope drifted")
    _require(selector.get("validation_used_for_selection") is False, "smoke selector used validation")
    job = runner.Job(
        stage="development",
        job_id="SMOKE",
        kind="engineering_smoke",
        method="selected_v4_candidate",
        algorithm=V48_CANDIDATE,
        implementation_id=V48_IMPLEMENTATION_IDS[V48_CANDIDATE],
        scenario="cross",
        seed=108,
        raw_steps=120,
        calibration_episodes=2,
        calibration_seed_start=97_000,
        evaluation_episodes=2,
        evaluation_split="validation",
        evaluation_seed_start=98_100,
        role="engineering_only",
        protocol_tag="smoke",
    )
    runner._validate_selector_evidence(SMOKE, job)
    _require(detailed.get("selected_deployment_decoder") == selector.get("selected_deployment_decoder"), "smoke decoder binding drifted")
    _require(detailed.get("selected_source_checkpoint_sha256") == selector.get("selected_source_checkpoint_sha256"), "smoke source binding drifted")
    _require(detailed.get("selected_model_policy_class") == selector.get("selected_model_policy_class"), "smoke policy binding drifted")
    _require(detailed.get("selected_model_parameter_state_sha256") == selector.get("selected_model_parameter_state_sha256"), "smoke tensor binding drifted")
    sealed = datetime.fromisoformat(selector["sealed_at_utc"])
    constructed = datetime.fromisoformat(detailed["validation_environment_constructed_at_utc"])
    _require(sealed <= constructed, "smoke validation environment predates receipt")
    _require(actions.get("selected_decoder_exact_rule_match_rate") == 1.0, "smoke selected decoder rule mismatch")
    _verify_zip(SMOKE / "final_model.zip")
    _verify_zip(SMOKE / "selected_model.zip")
    return {
        "directory": str(SMOKE.resolve()),
        "training_diagnostics_sha256": _sha256(SMOKE / "training_diagnostics.json"),
        "return_estimator_diagnostics_sha256": _sha256(SMOKE / "return_estimator_diagnostics.json"),
        "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
        "paper_evaluation_sha256": _sha256(SMOKE / "paper_evaluation_detailed.json"),
        "selected_model_sha256": _sha256(SMOKE / "selected_model.zip"),
        "secondary_branch_exercised": True,
        "receipt_predates_validation": True,
        "all_selected_deployment_bindings_match": True,
    }


def _write_attempt_ledger() -> dict[str, Any]:
    payload = {
        "schema_version": "topo-scene-v4.8.engineering-attempts/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "attempts": [
            {
                "id": "s1",
                "directory": str((ENGINEERING / "smoke" / "s1").resolve()),
                "status": "failed_before_completion",
                "reason": "engineering_only_learning_starts_0_preceded_first_finalized_16_step_sequence",
                "scientific_gate_input": False,
            },
            {
                "id": "s2",
                "directory": str((ENGINEERING / "smoke" / "s2").resolve()),
                "status": "completed_but_superseded",
                "reason": "acceptance_detected_generic_selector_schema_overwrite",
                "selector_receipt_sha256": _sha256(ENGINEERING / "smoke" / "s2" / "selector" / "receipt.json"),
                "scientific_gate_input": False,
            },
            {
                "id": "s3",
                "directory": str((ENGINEERING / "smoke" / "s3").resolve()),
                "status": "completed_but_superseded",
                "reason": "acceptance_detected_parent_schema_in_embedded_return_diagnostics",
                "selector_receipt_sha256": _sha256(ENGINEERING / "smoke" / "s3" / "selector" / "receipt.json"),
                "scientific_gate_input": False,
            },
            {
                "id": "s4",
                "directory": str((ENGINEERING / "smoke" / "s4").resolve()),
                "status": "completed_but_superseded",
                "reason": "post_version_diagnostic_assignment_retained_parent_schema",
                "selector_receipt_sha256": _sha256(ENGINEERING / "smoke" / "s4" / "selector" / "receipt.json"),
                "scientific_gate_input": False,
            },
            {
                "id": "s5",
                "directory": str(SMOKE.resolve()),
                "status": "canonical_pass",
                "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
                "scientific_gate_input": False,
            },
        ],
    }
    _write(ATTEMPTS, payload)
    return {**_source(ATTEMPTS), "attempts": len(payload["attempts"])}


def main() -> int:
    runner.validate_contract(runner.load_contract())
    stage0 = runner.scientific.parent._validate_stage0(
        runner.DEFAULT_STAGE0_RESULTS, runner.DEFAULT_STAGE0_ATTRIBUTION
    )
    attribution = runner._validate_failure_attribution(
        runner.DEFAULT_FAILURE_ATTRIBUTION, runner.DEFAULT_DEEP_ATTRIBUTION
    )
    parent = _parent_evidence()
    preregistration = _preregistration_evidence()
    targeted = _pytest_evidence(TARGETED_LOG, minimum_tests=16)
    full = _pytest_evidence(FULL_LOG, minimum_tests=300)
    preflights = _preflight_evidence()
    smoke = _smoke_evidence()
    attempts = _write_attempt_ledger()
    files: dict[str, str] = {}
    for relative in SCIENTIFIC_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"scientific file missing: {relative}")
        files[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.8.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_job": True,
        "formal_test_accessed": False,
        "single_scientific_change": "tie_only_replicated_train_calibration",
        "training_return_reward_network_entropy_and_decoders_unchanged": True,
        "secondary_calibration_trigger": "empirical_tied_top_count_greater_than_one",
        "secondary_calibration_seed_offset": 100,
        "secondary_candidate_scope": "empirical_tied_top_only",
        "non_top_candidate_reentry_forbidden": True,
        "experiment_contract_sha256": _sha256(runner.DEFAULT_CONTRACT),
        **stage0,
        **attribution,
        "preregistration_receipt_sha256": _sha256(runner.DEFAULT_PREREGISTRATION),
        "parent_implementation_freeze_sha256": _sha256(PARENT_FREEZE),
        "scientific_files": files,
        "scientific_content_sha256": _canonical_sha256(files),
        "engineering_evidence": {
            "parent": parent,
            "preregistration": preregistration,
            "targeted_tests": targeted,
            "full_regression": full,
            "real_sumo_preflights": preflights,
            "real_sumo_secondary_branch_smoke": smoke,
            "engineering_attempts": attempts,
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
        "schema_version": "topo-scene-v4.8.engineering-receipt/v1",
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
        "smoke_secondary_branch_exercised": True,
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
