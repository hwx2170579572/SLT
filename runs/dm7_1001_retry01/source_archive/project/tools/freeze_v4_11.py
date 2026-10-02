"""Validate and freeze the v4.11 model-only PRCR development protocol."""

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

from configs.sb3_configs_v4_11 import (
    TEMPORAL_GRAPH,
    V411_FULL,
    V411_IMPLEMENTATION_IDS,
)
from tools import run_topo_v4_11_experiments as runner
from tools.audit_v4_11_no_kinematic_projection import build_audit
from tools.checkpoint_decoder_selector_v4_6 import PARENT_DECODER, TARGET_DECODER


ENGINEERING = ROOT / "results_topo_v4_11_dev" / "engineering"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
HISTORICAL_EXCEPTION = ENGINEERING / "historical_regression_exception.json"
AUDIT = ENGINEERING / "no_kinematic_projection_audit.json"
CANDIDATE_SMOKE = (
    ENGINEERING / "smoke_runs" / "E0c__prcr_full__left_turn__s1"
)
CONTROL_SMOKE = (
    ENGINEERING / "smoke_runs" / "E1c__temporal_graph__left_turn__s1"
)
TARGETED_XML = ENGINEERING / "logs" / "targeted_tests.xml"
CURRENT_XML = ENGINEERING / "logs" / "current_state_regression.xml"
HISTORICAL_XML = ENGINEERING / "logs" / "historical_state_diagnostic.xml"

SCIENTIFIC_FILES = (
    "experiments/topo_scene_v4/experiment_contract_v4_11.yaml",
    "experiments/topo_scene_v4/V4_11_PROPER_CALIBRATED_RANKED_RISK_MODEL.md",
    "experiments/topo_scene_v4/V4_11_NO_KINEMATIC_PROJECTION_AUDIT_20260830.md",
    "algos/sb3_torch/hybrid_policy_v4.py",
    "algos/sb3_torch/hybrid_policy_v4_9_model.py",
    "algos/sb3_torch/hybrid_policy_v4_10_model.py",
    "algos/sb3_torch/hybrid_policy_v4_11_model.py",
    "algos/sb3_torch/replay_buffer_v4_9.py",
    "algos/sb3_torch/sac_v4_9_model.py",
    "algos/sb3_torch/sac_v4_11_model.py",
    "algos/sb3_torch/topo_temporal_features_v2.py",
    "configs/sb3_configs_v4_9.py",
    "configs/sb3_configs_v4_11.py",
    "envs/sumo/decision_alignment_v4.py",
    "envs/sumo/paper_env_v4.py",
    "envs/sumo/sumo_env.py",
    "tools/checkpoint_decoder_selector_v4_6.py",
    "tools/checkpoint_decoder_selector_v4_9_2.py",
    "tools/checkpoint_selector_v4_4.py",
    "tools/train_sb3_v4_7.py",
    "tools/train_sb3_v4_8.py",
    "tools/train_sb3_v4_9.py",
    "tools/train_sb3_v4_11.py",
    "tools/train_paper_sb3_sumo_v4_5.py",
    "tools/train_paper_sb3_sumo_v4_11.py",
    "tools/action_diagnostics_v4_11_model.py",
    "tools/run_topo_v4_11_experiments.py",
    "tools/run_all_v4_11_pipeline.py",
    "tools/run_all_pending_promotions_v4_11.py",
    "tools/audit_v4_11_no_kinematic_projection.py",
    "tools/attribute_v4_10_development_failure.py",
    "tools/freeze_v4_11.py",
    "tests_sb3_sumo/test_proper_ranked_risk_v4_11.py",
    "tests_sb3_sumo/test_prcr_model_v4_11.py",
    "tests_sb3_sumo/test_train_sb3_v4_11.py",
    "tests_sb3_sumo/test_action_diagnostics_v4_11.py",
    "tests_sb3_sumo/test_audit_v4_11_no_kinematic_projection.py",
    "tests_sb3_sumo/test_run_topo_v4_11_experiments.py",
    "tests_sb3_sumo/test_run_all_v4_11_pipeline.py",
    "tests_sb3_sumo/test_attribute_v4_10_development_failure.py",
    "results_topo_v4_11_dev/preregistration_receipt.json",
    "results_topo_v4_11_dev/engineering/no_kinematic_projection_audit.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    _require(bool(suites), f"JUnit testsuite missing: {path}")
    failed_cases: list[str] = []
    for suite in suites:
        for case in suite.findall("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed_cases.append(
                    f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
                )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
        "failed_cases": sorted(failed_cases),
    }


def _smoke_job(*, candidate: bool) -> runner.Job:
    algorithm = V411_FULL if candidate else TEMPORAL_GRAPH
    method = "prcr_full" if candidate else "temporal_graph"
    return runner.Job(
        stage="engineering",
        job_id="E0c" if candidate else "E1c",
        kind="engineering_smoke",
        method=method,
        algorithm=algorithm,
        implementation_id=V411_IMPLEMENTATION_IDS[algorithm],
        scenario="left_turn",
        seed=1,
        raw_steps=96,
        calibration_episodes=1,
        calibration_seed_start=942_000,
        evaluation_episodes=1,
        evaluation_split="validation",
        evaluation_seed_start=943_000,
        role="engineering_only_not_scientific_evidence",
        protocol_tag="engineering",
    )


def _common_smoke_validation(root: Path, job: runner.Job) -> dict[str, Any]:
    for relative in runner.FULL_RUN_REQUIRED_FILES:
        _require(
            (root / relative).is_file(),
            f"smoke artifact missing: {root}/{relative}",
        )
    arguments = _load_json(root / "arguments.json")
    _require(
        arguments.get("schema_version") == "topo-scene-v4.11.run-arguments/v1",
        "smoke argument schema drifted",
    )
    requested = arguments.get("requested_raw_steps", {})
    expected = {
        "scenario": "left_turn",
        "algo": job.algorithm,
        "max_steps": 96,
        "seed": 1,
        "learning_starts": 48,
        "buffer_size": 128,
        "batch_size": 2,
        "action_repeat": 3,
        "ego_control_profile": "direct",
        "traffic_protocol": "frozen_60_20_20",
        "episode_limit_profile": "source",
        "evaluation_split": "validation",
        "evaluation_seed_start": 943_000,
        "calibration_seed_start": 942_000,
        "calibration_episodes": 1,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "stage0_results_sha256": runner.STAGE0_RESULTS_SHA256,
        "attribution_sha256": runner.ATTRIBUTION_SHA256,
        "implementation_freeze_sha256": "engineering_smoke_before_freeze",
    }
    for key, expected_value in expected.items():
        _require(
            requested.get(key) == expected_value,
            f"smoke argument {key} drifted",
        )
    _require(arguments.get("formal_unlock") is None, "smoke accessed formal unlock")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    _require(detailed.get("evaluation_split") == "validation", "smoke split drifted")
    _require(detailed.get("formal_test_accessed") is False, "smoke accessed formal")
    _require(detailed.get("trained_raw_steps") == 96, "smoke training budget drifted")
    _require(
        detailed.get("summary", {}).get("episodes") == 1,
        "smoke episode count drifted",
    )
    return {
        "path": str(root.resolve()),
        "arguments_sha256": _sha256(root / "arguments.json"),
        "selected_model_sha256": _sha256(root / "selected_model.zip"),
        "selector_receipt_sha256": _sha256(root / "selector" / "receipt.json"),
        "action_diagnostics_sha256": _sha256(root / "action_diagnostics.json"),
        "paper_evaluation_detailed_sha256": _sha256(
            root / "paper_evaluation_detailed.json"
        ),
        "engineering_only_not_counted_for_scientific_gate": True,
        "accepted": True,
    }


def validate_smokes() -> dict[str, Any]:
    candidate_job = _smoke_job(candidate=True)
    candidate = _common_smoke_validation(CANDIDATE_SMOKE, candidate_job)
    runner._validate_candidate(CANDIDATE_SMOKE, candidate_job)
    candidate_selector = _load_json(CANDIDATE_SMOKE / "selector" / "receipt.json")
    _require(
        candidate_selector.get("selected_deployment_decoder") == TARGET_DECODER,
        "candidate smoke selected non-target decoder",
    )
    _require(
        candidate_selector.get("selected_model_policy_class")
        == "ProperCalibratedRankedRiskSACPolicyV411",
        "candidate smoke policy class drifted",
    )
    training = _load_json(CANDIDATE_SMOKE / "training_diagnostics.json")
    statistics = training.get("statistics", {})
    risk_diagnostics = {
        name: statistics.get(name, {}).get("last")
        for name in (
            "train/collision_soft_bce_loss",
            "train/collision_pairwise_rank_loss",
            "train/collision_temporal_consistency_loss",
            "risk/collision_probability_absolute_error",
        )
    }
    risk_diagnostics["risk/collision_pairwise_active_pair_rate.maximum"] = (
        statistics.get("risk/collision_pairwise_active_pair_rate", {}).get(
            "maximum"
        )
    )
    _require(
        all(isinstance(value, (int, float)) for value in risk_diagnostics.values()),
        "candidate risk diagnostics missing",
    )

    control_job = _smoke_job(candidate=False)
    control = _common_smoke_validation(CONTROL_SMOKE, control_job)
    runner._validate_control(CONTROL_SMOKE, control_job)
    control_selector = _load_json(CONTROL_SMOKE / "selector" / "receipt.json")
    _require(
        control_selector.get("selected_deployment_decoder") == PARENT_DECODER,
        "control smoke decoder drifted",
    )
    _require(
        _load_json(CONTROL_SMOKE / "method_metadata.json").get("network_changed")
        is False,
        "TemporalGraph control network changed",
    )
    return {
        "candidate": candidate,
        "temporal_graph_control": control,
        "paired_environment_settings": True,
        "candidate_selected_decoder": TARGET_DECODER,
        "control_selected_decoder": PARENT_DECODER,
        "candidate_risk_training_diagnostics": risk_diagnostics,
    }


def validate_no_projection_audit() -> dict[str, Any]:
    stored = _load_json(AUDIT)
    _require(stored.get("integrity_passed") is True, "stored projection audit failed")
    _require(stored.get("runtime_required") is True, "runtime audit was optional")
    runtime = stored.get("engineering_runtime", {})
    _require(runtime.get("available") is True, "runtime audit evidence missing")
    for key in (
        "all_exact_model_action_rates_are_one",
        "direct_speed_command_preserved_for_all_traced_actions",
        "normalized_speed_parameterization_exact_for_all_traced_actions",
        "lane_code_parameterization_exact_for_all_traced_actions",
        "environment_domain_clip_is_identity_for_all_traced_model_actions",
    ):
        _require(runtime.get(key) is True, f"runtime projection check failed: {key}")
    recomputed = build_audit(CANDIDATE_SMOKE, require_runtime=True)
    _require(
        recomputed.get("integrity_passed") is True,
        "recomputed projection audit failed",
    )
    _require(
        stored.get("source_boundary", {}).get("source_sha256")
        == recomputed.get("source_boundary", {}).get("source_sha256"),
        "projection-audited source changed",
    )
    return {
        "path": str(AUDIT.resolve()),
        "sha256": _sha256(AUDIT),
        "integrity_passed": True,
        "runtime_decision_records": runtime["decision_records"],
        "exact_model_action_rates": runtime["exact_rates"],
        "direct_speed_identity": True,
        "lane_code_identity": True,
        "external_kinematic_projection": False,
        "inference_rule_mechanism": False,
        "semantic_tie_override": False,
    }


def main() -> int:
    _require(
        Path(sys.executable).resolve() == runner.PYTHON_EXECUTABLE.resolve(),
        "freeze must run inside llm_pipeline",
    )
    _require(
        _sha256(runner.DEFAULT_CONTRACT) == runner.CONTRACT_SHA256,
        "experiment contract hash drifted",
    )
    runner.validate_contract(runner.load_contract())
    runner.validate_preregistration()
    _require(
        not (ROOT / "results_topo_v4_11_dev" / "development").exists(),
        "scientific development appears to have started before freeze",
    )
    _require(
        not (ROOT / "results_topo_v4_11_promotion").exists(),
        "promotion appears to have started before freeze",
    )

    targeted = _junit(TARGETED_XML)
    current = _junit(CURRENT_XML)
    historical = _junit(HISTORICAL_XML)
    _require(
        targeted["tests"] == 42
        and targeted["failures"] == 0
        and targeted["errors"] == 0,
        "targeted v4.11 regression did not pass",
    )
    _require(
        current["tests"] == 564
        and current["failures"] == 0
        and current["errors"] == 0
        and current["skipped"] == 0,
        "current-state regression did not pass",
    )
    expected_historical = [
        "tests_sb3_sumo.test_topo_v4_8_2_experiments::"
        "test_v4_8_2_h1_preflight_reproduces_both_bugs_and_adopts_in_place"
    ]
    _require(
        historical["tests"] == 1
        and historical["failures"] == 1
        and historical["errors"] == 0
        and historical["failed_cases"] == expected_historical,
        "historical-state diagnostic drifted",
    )

    exception = {
        "schema_version": "topo-scene-v4.11.historical-regression-exception/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "current_state_regression": current,
        "historical_state_diagnostic": historical,
        "historical_state_test": expected_historical[0],
        "reason": (
            "the diagnostic requires v4.8.2 to remain at H1 with next_job H2, "
            "while later completed v4.8/v4.9/v4.9.2/v4.10 evidence intentionally "
            "advanced that immutable historical state"
        ),
        "excluded_from_current_state_command": True,
        "v4_11_targeted_tests_deselected": 0,
        "unknown_failures_waived": 0,
    }
    _write_json(HISTORICAL_EXCEPTION, exception)

    smokes = validate_smokes()
    projection_audit = validate_no_projection_audit()
    files = {}
    for relative in SCIENTIFIC_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"scientific file missing: {relative}")
        files[relative] = _sha256(path)
    exception_relative = str(HISTORICAL_EXCEPTION.relative_to(ROOT)).replace(
        "\\", "/"
    )
    files[exception_relative] = _sha256(HISTORICAL_EXCEPTION)
    freeze = {
        "schema_version": "topo-scene-v4.11.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "development_started_before_freeze": False,
        "promotion_started_before_freeze": False,
        "model_change": "proper_calibrated_ranked_risk_training_model_only",
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "preregistration_receipt_sha256": _sha256(
            runner.DEFAULT_PREREGISTRATION
        ),
        "v4_10_failed_development_decision_sha256": (
            runner.STAGE0_RESULTS_SHA256
        ),
        "v4_10_complete_attribution_sha256": runner.ATTRIBUTION_SHA256,
        "no_kinematic_projection_audit_sha256": _sha256(AUDIT),
        "scientific_files": files,
        "scientific_content_sha256": _canonical_sha256(files),
        "model_integrity": {
            "shared_learned_risk_encoder": True,
            "proper_soft_target_bernoulli_nll": True,
            "continuous_target_difference_pairwise_ranking": True,
            "hard_target_threshold": False,
            "fixed_ranking_margin": False,
            "detached_ema_temporal_consistency": True,
            "learned_speed_components_per_lane": 3,
            "replay_supported_proposals": True,
            "learned_twin_uncertainty": True,
            "optimizer_parameter_sets_disjoint": True,
            "external_kinematic_projection": False,
            "fixed_speed_grid_at_inference": False,
            "ttc_or_headway_threshold": False,
            "lane_change_veto": False,
            "actor_confidence_gate": False,
            "semantic_tie_preference": False,
            "safety_shield_or_rule_fallback": False,
            "post_decoder_action_rewrite": False,
            "collision_label_source": "observed_environment_info_collision",
        },
        "engineering_evidence": {
            "targeted_regression": targeted,
            "current_state_regression": current,
            "historical_state_diagnostic": historical,
            "historical_regression_exception": {
                "path": str(HISTORICAL_EXCEPTION.resolve()),
                "sha256": _sha256(HISTORICAL_EXCEPTION),
            },
            "real_sumo_smokes": smokes,
            "no_kinematic_projection_audit": projection_audit,
        },
        "execution_policy": {
            "entered_stage": "attempt_all_then_summarize",
            "failure_cancels_remaining_jobs_in_entered_stage": False,
            "downstream_stage_requires_upstream_gate": True,
            "formal_requires_complete_promotion_pass": True,
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "conda_environment": "llm_pipeline",
        },
    }
    _write_json(FREEZE, freeze)
    receipt = {
        "schema_version": "topo-scene-v4.11.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "development_started_before_freeze": False,
        "promotion_started_before_freeze": False,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": _sha256(FREEZE),
        "scientific_content_sha256": freeze["scientific_content_sha256"],
        "targeted_tests": targeted,
        "current_state_regression": current,
        "candidate_smoke": smokes["candidate"],
        "temporal_graph_control_smoke": smokes["temporal_graph_control"],
        "candidate_risk_training_diagnostics": smokes[
            "candidate_risk_training_diagnostics"
        ],
        "no_kinematic_projection_audit": projection_audit,
        "development_execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "next_stage": "fresh_four_job_v4_11_development",
        "formal_test_locked": True,
    }
    _write_json(RECEIPT, receipt)

    runner.validate_implementation_freeze(FREEZE)
    runner.validate_engineering_receipt(RECEIPT)
    print(
        json.dumps(
            {
                "freeze": freeze,
                "freeze_sha256": _sha256(FREEZE),
                "receipt": receipt,
                "receipt_sha256": _sha256(RECEIPT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
