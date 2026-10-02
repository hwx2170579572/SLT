"""Seal the v4.9 learned collision-value model before fresh development."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_9_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_9_dev" / "engineering"
TARGETED_LOG = ENGINEERING / "logs" / "targeted_tests.xml"
FULL_LOG = ENGINEERING / "logs" / "full_regression_short_tmp.xml"
CURRENT_STATE_LOG = ENGINEERING / "logs" / "full_regression_current_state.xml"
LONG_BASETEMP_LOG = ENGINEERING / "logs" / "full_regression.xml"
STATE_EXCEPTION = ENGINEERING / "historical_state_regression_exception.json"
MODEL_AUDIT = ENGINEERING / "model_integrity_audit.json"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"

SCIENTIFIC_FILES = (
    "algos/sb3_torch/replay_buffer_v4_9.py",
    "algos/sb3_torch/hybrid_policy_v4_9_model.py",
    "algos/sb3_torch/sac_v4_9_model.py",
    "configs/sb3_configs_v4_9.py",
    "tools/action_diagnostics_v4_9_model.py",
    "tools/train_sb3_v4_9.py",
    "tools/train_paper_sb3_sumo_v4_9.py",
    "tools/run_topo_v4_9_experiments.py",
    "tools/run_all_v4_9_promotion.py",
    "tools/freeze_v4_9.py",
    "tests_sb3_sumo/test_learned_collision_model_v4_9.py",
    "tests_sb3_sumo/test_v4_9_adapter.py",
    "tests_sb3_sumo/test_topo_v4_9_experiments.py",
    "experiments/topo_scene_v4/V4_9_LEARNED_COLLISION_VALUE_MODEL.md",
    "experiments/topo_scene_v4/experiment_contract_v4_9.yaml",
    "results_topo_v4_9_dev/preregistration_receipt.json",
    "results_topo_v4_9_dev/engineering/historical_state_regression_exception.json",
    "results_topo_v4_9_dev/engineering/model_integrity_audit.json",
)

CORE_MODEL_FILES = (
    "algos/sb3_torch/replay_buffer_v4_9.py",
    "algos/sb3_torch/hybrid_policy_v4_9_model.py",
    "algos/sb3_torch/sac_v4_9_model.py",
    "configs/sb3_configs_v4_9.py",
)

BANNED_OPERATIONAL_FRAGMENTS = (
    "headway",
    "time_to_collision",
    "unsafe_distance",
    "lane_change_veto",
    "kinematic_projection",
    "project_action",
    "safety_override",
    "unsafe_target",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().lower()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        **_source(path),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }


def _validate_preregistration() -> dict[str, Any]:
    receipt = _load(runner.DEFAULT_PREREGISTRATION)
    _require(
        receipt.get("schema_version")
        == "topo-scene-v4.9.learned-collision-value-development-preregistration/v1",
        "preregistration schema drifted",
    )
    _require(
        receipt.get("created_after_initial_implementation_and_engineering_validation")
        is True,
        "preregistration timing is not disclosed",
    )
    _require(
        receipt.get("created_before_fresh_development_runs") is True,
        "fresh development was not preregistered",
    )
    _require(
        receipt.get("not_retroactive_scientific_preregistration") is True,
        "preregistration overclaims timing",
    )
    _require(
        receipt.get("experiment_contract_sha256") == runner.contract_sha256(),
        "preregistration contract drifted",
    )
    boundary = receipt.get("model_boundary", {})
    for key in (
        "external_kinematic_projection",
        "manual_geometry_threshold",
        "lane_change_veto",
        "action_postprocessing_override",
    ):
        _require(boundary.get(key) is False, f"preregistration enabled {key}")
    development = receipt.get("development_execution", {})
    _require(
        development.get("policy") == "attempt_all_then_summarize",
        "development batch policy drifted",
    )
    _require(
        development.get("failure_cancels_remaining_jobs") is False,
        "development fail-fast was re-enabled",
    )
    promotion = receipt.get("promotion_execution", {})
    _require(
        promotion.get("policy") == "attempt_all_then_summarize",
        "promotion batch policy drifted",
    )
    _require(
        promotion.get("failure_cancels_remaining_jobs") is False,
        "promotion fail-fast was re-enabled",
    )
    formal = receipt.get("formal_test", {})
    _require(formal.get("locked") is True, "formal test was not locked")
    _require(formal.get("accessed") is False, "formal test was accessed")
    return _source(runner.DEFAULT_PREREGISTRATION)


def _validate_regression_evidence() -> dict[str, Any]:
    targeted = _junit(TARGETED_LOG)
    full = _junit(FULL_LOG)
    current = _junit(CURRENT_STATE_LOG)
    long_path = _junit(LONG_BASETEMP_LOG)
    _require(
        (targeted["tests"], targeted["failures"], targeted["errors"])
        == (22, 0, 0),
        "targeted regression drifted",
    )
    _require(
        (full["tests"], full["failures"], full["errors"])
        == (414, 1, 0),
        "full regression counts drifted",
    )
    full_text = FULL_LOG.read_text(encoding="utf-8")
    nodeid = (
        "test_v4_8_2_h1_preflight_reproduces_both_bugs_and_adopts_in_place"
    )
    _require(nodeid in full_text, "historical-state nodeid missing")
    _require(
        "patched development continuation drifted" in full_text,
        "historical-state failure message missing",
    )
    _require(
        (current["tests"], current["failures"], current["errors"])
        == (413, 0, 0),
        "current-state regression did not pass",
    )
    _require(
        (long_path["tests"], long_path["failures"], long_path["errors"])
        == (414, 3, 0),
        "long-basetemp diagnostic counts drifted",
    )
    _require(
        "WinError 206" in LONG_BASETEMP_LOG.read_text(encoding="utf-8"),
        "Windows path-length evidence missing",
    )
    exception = _load(STATE_EXCEPTION)
    _require(
        exception.get("schema_version")
        == "topo-scene-v4.9.historical-state-regression-exception/v1",
        "state exception schema drifted",
    )
    _require(
        exception.get("status") == "attributed_external_artifact_state_only",
        "state exception unresolved",
    )
    _require(
        exception["full_regression"]["sha256"] == full["sha256"],
        "state exception full log drifted",
    )
    _require(
        exception["current_state_regression"]["sha256"] == current["sha256"],
        "state exception current-state log drifted",
    )
    _require(exception.get("waives_v4_9_failure") is False, "v4.9 failure waived")
    _require(exception.get("formal_test_accessed") is False, "state audit accessed formal")
    return {
        "targeted": targeted,
        "full_with_historical_snapshot_assertion": full,
        "current_state": current,
        "long_basetemp_diagnostic": long_path,
        "exception": _source(STATE_EXCEPTION),
    }


def _validate_no_rule_model() -> dict[str, Any]:
    findings: dict[str, Any] = {}
    for relative in CORE_MODEL_FILES:
        source = (ROOT / relative).read_text(encoding="utf-8").lower()
        found = [term for term in BANNED_OPERATIONAL_FRAGMENTS if term in source]
        _require(not found, f"rule/projection token found in {relative}: {found}")
        findings[relative] = {
            "sha256": _sha256(ROOT / relative),
            "banned_operational_fragments_found": found,
        }
    replay = (ROOT / CORE_MODEL_FILES[0]).read_text(encoding="utf-8")
    policy = (ROOT / CORE_MODEL_FILES[1]).read_text(encoding="utf-8")
    learner = (ROOT / CORE_MODEL_FILES[2]).read_text(encoding="utf-8")
    _require('info["collision"]' in replay, "observed collision event label missing")
    _require("collision_critic_target" in policy, "target collision critic missing")
    _require("minimum_reward_q" in policy, "reward model score missing")
    _require("maximum_collision_value" in policy, "collision model score missing")
    _require("target_collision_values" in learner, "collision Bellman target missing")
    _require("actor_loss.backward()" in learner, "actor risk gradient path missing")
    audit = _load(MODEL_AUDIT)
    _require(
        audit.get("finding")
        == "model_only_safety_learning_no_external_kinematic_projection",
        "model audit finding drifted",
    )
    _require(audit.get("post_decoder_safety_projection") is False, "projection enabled")
    checks = audit.get("checks", {})
    required_true = (
        "label_is_observed_collision_event",
        "label_is_independent_of_reward_sign_or_normalization",
        "n_step_collision_label_stops_at_first_episode_boundary",
        "n_step_bootstrap_uses_gamma_power_actual_horizon",
        "online_and_target_collision_critics_are_parameter_independent",
        "twin_collision_values_use_bounded_sigmoid_outputs",
        "twin_collision_values_use_pessimistic_maximum",
        "actor_receives_gradient_through_expected_learned_collision_value",
        "reward_critic_objective_is_unchanged",
        "deterministic_score_is_learned_reward_minus_learned_risk",
        "learned_collision_modules_survive_save_load",
    )
    required_false = (
        "external_action_rewrite_added",
        "headway_or_time_to_collision_threshold_added",
        "lane_change_safety_veto_added",
        "manual_geometry_unsafe_label_added",
        "future_or_oracle_collision_label_used",
    )
    for key in required_true:
        _require(checks.get(key) is True, f"model integrity check failed: {key}")
    for key in required_false:
        _require(checks.get(key) is False, f"forbidden mechanism enabled: {key}")
    _require(audit.get("formal_test_accessed") is False, "model audit accessed formal")
    return {
        "audit": _source(MODEL_AUDIT),
        "operational_fragment_scan": findings,
        "external_kinematic_projection": False,
        "post_decoder_action_override": False,
    }


def _finite_stat(payload: dict[str, Any], name: str) -> bool:
    value = payload.get("statistics", {}).get(name, {}).get("last")
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_smoke_and_preflights() -> dict[str, Any]:
    smoke = ENGINEERING / "smoke" / "model_e2e"
    labels_path = smoke / "collision_label_diagnostics.json"
    training_path = smoke / "training_diagnostics.json"
    actions_path = smoke / "action_diagnostics.json"
    selector_path = smoke / "selector" / "receipt.json"
    checkpoint_path = smoke / "checkpoint_audit.json"
    labels = _load(labels_path)
    training = _load(training_path)
    actions = _load(actions_path)
    selector = _load(selector_path)
    checkpoint = _load(checkpoint_path)
    _require(labels.get("source") == "observed_environment_info_collision", "label source drifted")
    _require(labels.get("future_or_oracle_labels_used") is False, "oracle label used")
    _require(int(labels.get("sampled_label_count", 0)) > 0, "smoke sampled no labels")
    _require(int(training.get("learner_updates", 0)) > 0, "smoke made no updates")
    for name in (
        "train/collision_critic_loss",
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/policy_expected_collision_cost",
    ):
        _require(_finite_stat(training, name), f"non-finite smoke statistic: {name}")
    _require(actions.get("computed_from_real_rollout") is True, "smoke rollout is not real")
    _require(actions.get("risk_adjusted_score_equation_match_rate") == 1.0, "risk equation drifted")
    _require(actions.get("selected_action_mask_feasible_rate") == 1.0, "invalid action selected")
    _require(actions.get("learned_collision_critic_present") is True, "online risk critic missing")
    _require(actions.get("learned_collision_critic_target_present") is True, "target risk critic missing")
    _require(actions.get("inference_safety_rule_added") is False, "inference rule added")
    legacy = actions.get("legacy_reward_only_decoder_counterfactual", {})
    _require(
        float(legacy.get("selected_decoder_exact_rule_match_rate", 1.0)) < 1.0,
        "learned risk did not affect any smoke decision",
    )
    _require(selector.get("policy_parameter_state_preserved") is True, "selector changed parameters")
    _require(
        all(bool(row.get("zip_crc_ok")) for row in checkpoint.get("checkpoints", [])),
        "checkpoint archive failed CRC audit",
    )
    preflights: dict[str, Any] = {}
    for name in (
        "candidate_cross_risk_model",
        "candidate_roundabout_medium_risk_model",
        "candidate_carla_risk_model",
    ):
        path = ENGINEERING / "preflight" / name / "check_result.json"
        value = _load(path)
        metadata = value.get("method_metadata", {})
        _require(value.get("status") == "ok", f"preflight failed: {name}")
        _require(metadata.get("learned_collision_critic") is True, f"risk model missing: {name}")
        _require(metadata.get("inference_safety_rule_added") is False, f"rule added: {name}")
        _require(value.get("action_finite") is True, f"non-finite action: {name}")
        _require(value.get("exact_hybrid_lane_code") is True, f"invalid lane code: {name}")
        preflights[name] = _source(path)
    return {
        "engineering_only_not_counted_for_scientific_gate": True,
        "collision_labels": _source(labels_path),
        "training": _source(training_path),
        "actions": _source(actions_path),
        "selector": _source(selector_path),
        "checkpoint": _source(checkpoint_path),
        "preflights": preflights,
    }


def main() -> int:
    contract = runner.validate_contract(runner.load_contract())
    runner._validate_failure_attribution(
        runner.DEFAULT_FAILURE_ATTRIBUTION, runner.DEFAULT_DEEP_ATTRIBUTION
    )
    _require(contract["formal_test"]["accessed"] is False, "contract accessed formal")
    formal_root = ROOT / "results_topo_v4_9_formal"
    _require(
        not formal_root.exists() or not any(formal_root.rglob("*")),
        "formal result directory is not untouched",
    )
    development_root = ROOT / "results_topo_v4_9_dev" / "development"
    _require(
        not development_root.exists() or not any(development_root.rglob("runs")),
        "fresh development began before freeze",
    )

    preregistration = _validate_preregistration()
    regression = _validate_regression_evidence()
    no_rule = _validate_no_rule_model()
    runtime = _validate_smoke_and_preflights()

    files: dict[str, str] = {}
    for relative in SCIENTIFIC_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"scientific file missing: {relative}")
        files[relative] = _sha256(path)

    payload = {
        "schema_version": "topo-scene-v4.9.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_test_accessed": False,
        "fresh_development_started_before_freeze": False,
        "single_change": "learned_twin_collision_value_model",
        "external_kinematic_projection": False,
        "manual_safety_rule_added": False,
        "experiment_contract_sha256": runner.contract_sha256(),
        "design_record_sha256": _sha256(runner.DEFAULT_DESIGN),
        "failure_attribution_sha256": runner.LINEAGE_HASHES[
            "failure_attribution_sha256"
        ],
        "deep_attribution_sha256": runner.LINEAGE_HASHES[
            "deep_attribution_sha256"
        ],
        "preregistration_receipt_sha256": preregistration["sha256"],
        "scientific_files": files,
        "scientific_content_sha256": _canonical_sha256(files),
        "model_integrity": no_rule,
        "engineering_evidence": {
            "regression": regression,
            "smoke_and_preflights": runtime,
        },
        "execution_policy": {
            "development": "attempt_all_then_summarize",
            "promotion": "attempt_all_then_summarize",
            "failure_cancels_remaining_jobs": False,
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
        "schema_version": "topo-scene-v4.9.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_test_accessed": False,
        "fresh_development_started_before_freeze": False,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "scientific_content_sha256": payload["scientific_content_sha256"],
        "external_kinematic_projection": False,
        "manual_safety_rule_added": False,
        "targeted_tests": regression["targeted"],
        "full_regression_with_historical_snapshot_assertion": regression[
            "full_with_historical_snapshot_assertion"
        ],
        "current_state_regression": regression["current_state"],
        "historical_state_exception": regression["exception"],
        "next_stage": "fresh_development_all_four_jobs",
        "promotion_locked": True,
        "formal_test_locked": True,
    }
    _write(RECEIPT, receipt)
    print(
        json.dumps(
            {
                "implementation_freeze": str(FREEZE.resolve()),
                "implementation_freeze_sha256": freeze_hash,
                "scientific_content_sha256": payload["scientific_content_sha256"],
                "engineering_receipt": str(RECEIPT.resolve()),
                "external_kinematic_projection": False,
                "manual_safety_rule_added": False,
                "next_stage": receipt["next_stage"],
                "formal_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
