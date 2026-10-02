"""Run the hash-bound strict target-only v4.9.2 experiment protocol.

Every selected promotion job is attempted before aggregation.  Formal testing
stays locked unless the complete 12-job promotion matrix passes every frozen
relative and model-integrity gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_9 import (
    TEMPORAL_GRAPH,
    V49_CANDIDATE,
    V49_FORMAL_ALGORITHMS,
    V49_IMPLEMENTATION_IDS,
)
from tools import run_topo_v4_9_experiments as parent
from tools.checkpoint_decoder_selector_v4_6 import PARENT_DECODER, TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE


PYTHON_EXECUTABLE = Path("D:/Programs/Anaconda/envs/llm_pipeline/python.exe")
DEFAULT_CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_9_2.yaml"
PARENT_CONTRACT = ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_9.yaml"
DEFAULT_PREREGISTRATION = ROOT / "results_topo_v4_9_2_dev" / "preregistration_receipt.json"
DEFAULT_DEVELOPMENT_DECISION = ROOT / "results_topo_v4_9_2_dev" / "development" / "development_decision.json"
DEFAULT_EQUIVALENCE = ROOT / "results_topo_v4_9_2_dev" / "development" / "target_only_equivalence.json"
DEFAULT_ENGINEERING_ROOT = ROOT / "results_topo_v4_9_2_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = ROOT / "results_topo_v4_9_2_promotion" / "promotion_gate.json"

CONTRACT_SHA256 = "9e99845d3987a32216b213e34267c19fddaa2386880035498d60610615344a95"
PARENT_CONTRACT_SHA256 = "4a8f39b734357efda228e1981c0a3d1bf956dba2cdd6af0d49cfafa8b7aae16d"
PARENT_FREEZE_SHA256 = "766ff1a0b4856e5f9a07afb43ac7122fdc7d9a63a3ec76e810cdf947eea22a5b"
PARENT_SUMMARY_SHA256 = "27dcdc5c1cf560f7364080ad4b5671887b870bb6b23ec343e6f7acfc4a6cfbe1"
PARENT_DECISION_SHA256 = "b0ffaf18245a484ec4567772ae80593887ba1eb7ec32f563a0a90063af9dbbf0"
COMPLETE_ATTRIBUTION_SHA256 = "2211e0a01be87cde765e782299c246cfa3849de355ea6c1395447235876ea731"
STAGE0_RESULTS_SHA256 = "7fe77bb2d20e2e67005bdc8743ff4b84762f0ae75fdb8cfe43d2e4922906e1d1"
STAGE0_ATTRIBUTION_SHA256 = "6c0873cb6a50337859eaea7fa28ff0db76dbd977d5d680159bd219916ccd0402"
FAILURE_ATTRIBUTION_SHA256 = "80e2d132e507bb8bcdf4d306f1842e7f71a0dee744921efe853bb2bbccba3ea6"
DEEP_ATTRIBUTION_SHA256 = "314a1a32771b2966eb807aefaeb904310b9e719a5de0d22b4d3314fa88b0acfb"

SCENARIOS = parent.SCENARIOS
STAGES = ("promotion", "formal")
METHOD_ALGORITHMS = {
    "temporal_graph": TEMPORAL_GRAPH,
    "selected_v4_candidate": V49_CANDIDATE,
}
METHOD_CODES = {"temporal_graph": "tg", "selected_v4_candidate": "cand"}
SCENARIO_CODES = parent.SCENARIO_CODES
Job = parent.Job
FULL_RUN_REQUIRED_FILES = (*parent.FULL_RUN_REQUIRED_FILES,)


class V492ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V492ProtocolError(message)


def _sha256(path: Path) -> str:
    return parent._sha256(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.9.2 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.9.2.strict-target-only-deployment-contract/v1",
        "unexpected v4.9.2 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication drifted")
    _require(contract.get("formal_test_locked") is True, "formal lock drifted")
    lineage = contract.get("lineage", {})
    expected_lineage = {
        "parent_scientific_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_scientific_freeze_sha256": PARENT_FREEZE_SHA256,
        "completed_development_summary_sha256": PARENT_SUMMARY_SHA256,
        "completed_development_decision_sha256": PARENT_DECISION_SHA256,
        "complete_model_attribution_sha256": COMPLETE_ATTRIBUTION_SHA256,
        "formal_test_accessed": False,
    }
    for key, expected in expected_lineage.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")
    change = contract.get("single_protocol_change", {})
    _require(change.get("name") == "restrict_candidate_deployment_to_learned_target_critic", "protocol change drifted")
    _require(change.get("deployment_decoder_candidates") == [TARGET_DECODER], "target-only decoder drifted")
    _require(int(change.get("checkpoint_decoder_candidate_pairs", 0)) == 2, "candidate pair count drifted")
    _require(change.get("selector") == SELECTOR_MODE, "selector mode drifted")
    for key in (
        "fusion_0_90_candidate_present",
        "actor_confidence_threshold_present",
        "external_kinematic_projection",
        "headway_or_ttc_threshold",
        "lane_change_veto",
        "manual_geometry_collision_label",
        "action_postprocessing_override",
    ):
        _require(change.get(key) is False, f"forbidden mechanism enabled: {key}")
    _require(change.get("scientific_model_changed_from_v4_9") is False, "model change drifted")
    promotion = contract.get("promotion", {})
    _require(promotion.get("execution_policy") == "attempt_all_then_summarize", "promotion execution policy drifted")
    _require(promotion.get("failure_does_not_cancel_remaining_jobs") is True, "promotion fail-fast enabled")
    _require(promotion.get("methods") == ["temporal_graph", "selected_v4_candidate"], "promotion methods drifted")
    _require(promotion.get("scenarios") == ["cross", "roundabout_medium", "carla"], "promotion scenarios drifted")
    _require(promotion.get("seeds") == [20, 21], "promotion seeds drifted")
    _require(int(promotion.get("jobs", 0)) == 12, "promotion job count drifted")
    formal = contract.get("formal_test", {})
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal state drifted")
    _require(len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"]) == 120, "formal matrix drifted")
    return contract


def validate_preregistration() -> dict[str, Any]:
    receipt = _load_json(DEFAULT_PREREGISTRATION)
    _require(receipt.get("schema_version") == "topo-scene-v4.9.2.strict-target-only-preregistration/v1", "preregistration schema drifted")
    _require(receipt.get("created_before_v4_9_2_implementation") is True, "implementation was not preregistered")
    _require(receipt.get("created_before_v4_9_2_promotion") is True, "promotion was not preregistered")
    _require(receipt.get("experiment_contract_sha256") == CONTRACT_SHA256, "preregistration contract drifted")
    _require(receipt.get("promotion_execution_policy") == "attempt_all_then_summarize", "preregistration execution drifted")
    _require(receipt.get("promotion_failure_does_not_cancel_remaining_jobs") is True, "preregistration fail-fast drifted")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal")
    return receipt


def validate_development_adoption() -> dict[str, Any]:
    decision = _load_json(DEFAULT_DEVELOPMENT_DECISION)
    report = _load_json(DEFAULT_EQUIVALENCE)
    _require(decision.get("schema_version") == "topo-scene-v4.9.2.development-adoption-decision/v1", "development decision schema drifted")
    _require(decision.get("decision") == "pass", "development adoption did not pass")
    _require(decision.get("promotion_unlocked") is True, "development did not unlock promotion")
    _require(decision.get("formal_test_accessed") is False, "development accessed formal")
    _require(report.get("all_conditions_passed") is True, "equivalence proof failed")
    _require(int(report.get("equivalent_cells", 0)) == 4, "equivalence matrix incomplete")
    _require(decision.get("target_only_equivalence_report_sha256") == _sha256(DEFAULT_EQUIVALENCE), "equivalence report hash drifted")
    return decision


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _require(payload.get("schema_version") == "topo-scene-v4.9.2.implementation-freeze/v1", "freeze schema drifted")
    _require(payload.get("experiment_contract_sha256") == CONTRACT_SHA256, "freeze contract drifted")
    _require(payload.get("formal_test_accessed") is False, "freeze accessed formal")
    files = payload.get("scientific_files", {})
    _require(isinstance(files, dict) and files, "freeze files missing")
    for relative, expected in files.items():
        source = ROOT / relative
        _require(source.is_file(), f"frozen file missing: {relative}")
        _require(_sha256(source) == expected, f"frozen file drifted: {relative}")
    return payload


def validate_engineering_receipt(path: Path = DEFAULT_ENGINEERING_RECEIPT) -> dict[str, Any]:
    receipt = _load_json(path)
    _require(receipt.get("status") == "passed", "engineering gate did not pass")
    _require(receipt.get("formal_test_accessed") is False, "engineering accessed formal")
    _require(receipt.get("implementation_freeze_sha256") == _sha256(DEFAULT_FREEZE), "engineering freeze mismatch")
    return receipt


def protocol_hashes() -> dict[str, str]:
    validate_contract(load_contract())
    validate_preregistration()
    validate_development_adoption()
    validate_implementation_freeze()
    validate_engineering_receipt()
    return {
        "experiment_contract_sha256": CONTRACT_SHA256,
        "stage0_results_sha256": STAGE0_RESULTS_SHA256,
        "stage0_attribution_sha256": STAGE0_ATTRIBUTION_SHA256,
        "failure_attribution_sha256": FAILURE_ATTRIBUTION_SHA256,
        "deep_attribution_sha256": DEEP_ATTRIBUTION_SHA256,
        "complete_model_attribution_sha256": COMPLETE_ATTRIBUTION_SHA256,
        "preregistration_receipt_sha256": _sha256(DEFAULT_PREREGISTRATION),
        "development_adoption_decision_sha256": _sha256(DEFAULT_DEVELOPMENT_DECISION),
        "target_only_equivalence_sha256": _sha256(DEFAULT_EQUIVALENCE),
        "implementation_freeze_sha256": _sha256(DEFAULT_FREEZE),
    }


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.9.2 stage {stage!r}")
    settings = contract["promotion" if stage == "promotion" else "formal_test"]
    tag = digest[:8]
    jobs: list[Job] = []
    for method in settings["methods"]:
        algorithm = METHOD_ALGORITHMS[method]
        for scenario in settings["scenarios"]:
            for seed_value in settings["seeds"]:
                seed = int(seed_value)
                if stage == "promotion":
                    offset = seed - 20
                    calibration_seed = 170_000 + 1_000 * offset
                    evaluation_seed = 270_000 + 1_000 * offset
                else:
                    calibration_seed = 310_000 + 1_000 * seed
                    evaluation_seed = 510_000 + 1_000 * seed
                jobs.append(
                    Job(
                        stage=stage,
                        job_id="P" if stage == "promotion" else "F",
                        kind="fresh_training_with_strict_target_only_model_selection",
                        method=str(method),
                        algorithm=algorithm,
                        implementation_id=V49_IMPLEMENTATION_IDS[algorithm],
                        scenario=str(scenario),
                        seed=seed,
                        raw_steps=int(settings["raw_training_steps"]),
                        calibration_episodes=12,
                        calibration_seed_start=calibration_seed,
                        evaluation_episodes=int(settings["evaluation_episodes"]),
                        evaluation_split=str(settings["evaluation_split"]),
                        evaluation_seed_start=evaluation_seed,
                        role=("fresh_paired_promotion" if stage == "promotion" else "untouched_formal_test"),
                        protocol_tag=tag,
                    )
                )
    expected = 12 if stage == "promotion" else 120
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} job")
    return jobs


def stage_root(stage: str) -> Path:
    return {
        "promotion": ROOT / "results_topo_v4_9_2_promotion",
        "formal": ROOT / "results_topo_v4_9_2_formal",
    }[stage]


def run_directory(job: Job) -> Path:
    return stage_root(job.stage) / "runs" / job.name


def command_for(job: Job, hashes: dict[str, str], *, device: str) -> list[str]:
    command = [
        str(PYTHON_EXECUTABLE),
        str(ROOT / "tools" / "train_paper_sb3_sumo_v4_9_2.py"),
        "--scenario", job.scenario,
        "--algo", job.algorithm,
        "--max-steps", str(job.raw_steps),
        "--learning-starts", "5000",
        "--checkpoint-freq", str(job.raw_steps),
        "--eval-freq", "0",
        "--eval-episodes", str(job.evaluation_episodes),
        "--evaluation-split", job.evaluation_split,
        "--evaluation-seed-start", str(job.evaluation_seed_start),
        "--calibration-episodes", str(job.calibration_episodes),
        "--calibration-seed-start", str(job.calibration_seed_start),
        "--seed", str(job.seed),
        "--device", device,
        "--batch-size", "32",
        "--learning-rate", "0.0001",
        "--discount", "0.99",
        "--buffer-size", "20000",
        "--action-repeat", "3",
        "--traffic-protocol", "frozen_60_20_20",
        "--episode-limit-profile", "source",
        "--output-dir", str((stage_root(job.stage) / "runs").resolve()),
        "--model-name", job.name,
        "--experiment-contract-sha256", hashes["experiment_contract_sha256"],
        "--stage0-results-sha256", hashes["stage0_results_sha256"],
        "--attribution-sha256", hashes["failure_attribution_sha256"],
        "--implementation-freeze-sha256", hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(["--formal-unlock-receipt", str(DEFAULT_PROMOTION_GATE.resolve())])
    return command


def _calibration_path(
    root: Path,
    candidate: dict[str, Any],
    *,
    secondary: bool,
    fallback_used: bool,
) -> Path:
    kind = candidate["checkpoint_kind"]
    if fallback_used and kind == "highest_training_success":
        kind = "exact_final"
    short = {"highest_training_success": "best", "exact_final": "final"}[kind]
    block = "cal_secondary" if secondary else "cal"
    return root / "selector" / block / short / candidate["deployment_decoder"]


def _validate_selector_calibrations(root: Path, selector: dict[str, Any], *, candidate: bool) -> None:
    expected_decoder = TARGET_DECODER if candidate else PARENT_DECODER
    expected_count = 2
    _require(selector.get("selection_partition") == "train", "selector did not use train")
    _require(selector.get("validation_used_for_selection") is False, "validation entered selection")
    _require(selector.get("formal_test_used_for_selection") is False, "formal entered selection")
    _require(int(selector.get("candidate_count", 0)) == expected_count, "selector candidate count drifted")
    rows = selector.get("candidates", [])
    _require(len(rows) == expected_count, "selector candidate receipt matrix incomplete")
    _require({row.get("checkpoint_kind") for row in rows} == {"highest_training_success", "exact_final"}, "checkpoint candidates drifted")
    _require(all(row.get("deployment_decoder") == expected_decoder for row in rows), "unexpected decoder entered selector")
    for secondary, candidates in ((False, rows), (True, selector.get("secondary_candidates", []))):
        for row in candidates:
            directory = _calibration_path(
                root,
                row,
                secondary=secondary,
                fallback_used=selector.get("training_best_missing_fallback_used")
                is True,
            )
            _require(_sha256(directory / "detailed.json") == row.get("calibration_result_sha256"), "calibration detailed hash drifted")
            _require(_sha256(directory / "actions.json") == row.get("calibration_action_diagnostics_sha256"), "calibration action hash drifted")
            detailed = _load_json(directory / "detailed.json")
            _require(detailed.get("traffic_partition") == "train", "calibration partition drifted")
            _require(detailed.get("formal_test_accessed") is False, "calibration accessed formal")
            _require(detailed.get("decoder_integrity_passed") is True, "calibration decoder integrity failed")
    if candidate:
        _require(selector.get("selector_mode") == SELECTOR_MODE, "target-only selector mode drifted")
        _require(selector.get("selected_deployment_decoder") == TARGET_DECODER, "candidate did not select target_critic")
        _require(
            selector.get("calibration_deterministic_lane_decoder")
            == TARGET_DECODER,
            "calibration decoder metadata is not target_critic",
        )
        _require(selector.get("deployment_decoder_candidates") == [TARGET_DECODER], "decoder candidate list drifted")
        for key in ("fusion_candidate_present", "actor_confidence_threshold_present", "external_kinematic_projection", "action_postprocessing_override"):
            _require(selector.get(key) is False, f"selector enabled forbidden {key}")
        _require(not any(path.name == "fusion_0_90" for path in (root / "selector").rglob("*")), "fusion calibration directory exists")
    else:
        _require(selector.get("selected_deployment_decoder") == PARENT_DECODER, "control decoder drifted")
    selected_path = Path(selector["selected_source_checkpoint_path"])
    _require(selected_path.is_file(), "selected source checkpoint missing")
    _require(_sha256(selected_path) == selector.get("selected_source_checkpoint_sha256"), "selected source checkpoint hash drifted")
    selected_model = Path(selector["selected_model_path"])
    _require(selected_model.is_file(), "selected model missing")
    _require(_sha256(selected_model) == selector.get("selected_model_sha256"), "selected model hash drifted")
    _require(selector.get("source_policy_parameter_state_sha256") == selector.get("selected_model_parameter_state_sha256"), "selected policy tensors changed")
    _require(selector.get("policy_parameter_state_preserved") is True, "state preservation flag failed")
    _require(selector.get("selector_receipt_precedes_validation_environment") is True, "selector was not sealed before validation")


def _validate_arguments(root: Path, job: Job, hashes: dict[str, str]) -> dict[str, Any]:
    arguments = _load_json(root / "arguments.json")
    _require(arguments.get("schema_version") == "topo-scene-v4.9.2.run-arguments/v1", "argument schema drifted")
    requested = arguments.get("requested_raw_steps", {})
    expected = {
        "scenario": job.scenario,
        "algo": job.algorithm,
        "max_steps": job.raw_steps,
        "seed": job.seed,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "discount": 0.99,
        "learning_starts": 5000,
        "buffer_size": 20000,
        "action_repeat": 3,
        "traffic_protocol": "frozen_60_20_20",
        "episode_limit_profile": "source",
        "eval_episodes": job.evaluation_episodes,
        "evaluation_split": job.evaluation_split,
        "evaluation_seed_start": job.evaluation_seed_start,
        "calibration_seed_start": job.calibration_seed_start,
        "calibration_episodes": job.calibration_episodes,
        "experiment_contract_sha256": hashes["experiment_contract_sha256"],
        "stage0_results_sha256": hashes["stage0_results_sha256"],
        "attribution_sha256": hashes["failure_attribution_sha256"],
        "implementation_freeze_sha256": hashes["implementation_freeze_sha256"],
    }
    for key, value in expected.items():
        _require(requested.get(key) == value, f"argument {key} drifted")
    _require(arguments.get("experiment_contract_sha256") == hashes["experiment_contract_sha256"], "top-level contract hash drifted")
    _require(arguments.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "top-level freeze hash drifted")
    fidelity = arguments.get("implementation_fidelity", {})
    _require(fidelity.get("implementation_id") == job.implementation_id, "implementation id drifted")
    _require(fidelity.get("v4_9_2_isolated_files") is True, "v4.9.2 wrapper not recorded")
    _require(fidelity.get("external_kinematic_projection") is False, "kinematic projection enabled")
    _require(fidelity.get("action_postprocessing_override") is False, "action override enabled")
    if job.algorithm == V49_CANDIDATE:
        _require(fidelity.get("deployment_decoder_candidates") == [TARGET_DECODER], "candidate decoder metadata drifted")
        _require(fidelity.get("checkpoint_decoder_candidate_pairs") == 2, "candidate pair metadata drifted")
        _require(fidelity.get("fusion_candidate_present") is False, "fusion candidate metadata enabled")
        _require(fidelity.get("actor_confidence_threshold_present") is False, "confidence threshold metadata enabled")
        _require("actor_non_keep_confidence_threshold" not in fidelity, "fixed confidence threshold remains")
        _require(fidelity.get("scientific_model_changed_from_v4_9") is False, "model changed from v4.9")
    return arguments


def _validate_candidate(root: Path, job: Job) -> None:
    metadata = _load_json(root / "method_metadata.json")
    actions = _load_json(root / "action_diagnostics.json")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    training = _load_json(root / "training_diagnostics.json")
    labels = _load_json(root / "collision_label_diagnostics.json")
    selector = _load_json(root / "selector" / "receipt.json")
    _require(metadata.get("deployment_decoder_candidates") == [TARGET_DECODER], "method metadata is not target-only")
    _require(metadata.get("checkpoint_decoder_candidate_pairs") == 2, "method pair count drifted")
    _require(metadata.get("selector_mode") == SELECTOR_MODE, "method selector drifted")
    _require(metadata.get("fusion_candidate_present") is False, "method permits fusion")
    _require(metadata.get("actor_confidence_threshold_present") is False, "method permits confidence threshold")
    _require("actor_non_keep_confidence_threshold" not in metadata, "method contains fixed confidence threshold")
    for source in (metadata, actions, detailed, selector):
        _require(source.get("external_kinematic_projection") is False, "kinematic projection present")
        _require(source.get("action_postprocessing_override") is False, "action rewrite present")
    required_actions = {
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "collision_risk_coef": 1.0,
        "inference_safety_rule_added": False,
        "model_decision_equation": "minimum_reward_q_minus_lambda_maximum_collision_value",
        "selected_decoder_exact_rule_match_rate": 1.0,
        "selected_decoder_exact_action_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "risk_adjusted_score_equation_match_rate": 1.0,
        "exact_risk_adjusted_model_argmax_rate": 1.0,
        "risk_adjusted_keep_tie_match_rate": 1.0,
        "fusion_candidate_present": False,
        "actor_confidence_threshold_present": False,
    }
    for key, expected in required_actions.items():
        _require(actions.get(key) == expected, f"action evidence {key} drifted")
    _require(actions.get("actor_non_keep_confidence_threshold") is None, "action diagnostics used confidence threshold")
    _require(detailed.get("selector_receipt_sha256") == _sha256(root / "selector" / "receipt.json"), "evaluation selector binding drifted")
    _require(int(detailed.get("summary", {}).get("episodes", 0)) == job.evaluation_episodes, "evaluation episode count drifted")
    _require(labels.get("source") == "observed_environment_info_collision", "collision label source drifted")
    _require(labels.get("future_or_oracle_labels_used") is False, "oracle collision labels used")
    _require(labels.get("reward_return_changed") is False, "collision labels changed reward")
    _require(int(labels.get("sampled_label_count", 0)) > 0, "no collision labels sampled")
    rate = labels.get("sampled_positive_label_rate")
    _require(rate is not None and 0.0 <= float(rate) <= 1.0, "collision label rate invalid")
    for name in (
        "train/collision_critic_loss",
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/collision_cost_replay_prediction",
        "risk/policy_expected_collision_cost",
        "risk/collision_risk_actor_penalty",
    ):
        value = training.get("statistics", {}).get(name, {}).get("last")
        _require(value is not None and math.isfinite(float(value)), f"invalid training statistic {name}")
    _validate_selector_calibrations(root, selector, candidate=True)


def _validate_control(root: Path) -> None:
    actions = _load_json(root / "action_diagnostics.json")
    selector = _load_json(root / "selector" / "receipt.json")
    _require(actions.get("learned_collision_critic_present") is False, "control gained collision critic")
    _require(actions.get("collision_risk_coef") is None, "control gained collision risk coefficient")
    _require(actions.get("inference_safety_rule_added") is False, "control gained safety rule")
    _require(not (root / "collision_label_diagnostics.json").exists(), "control emitted collision learner diagnostics")
    _validate_selector_calibrations(root, selector, candidate=False)


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        _require(root.is_dir(), "run directory missing")
        for relative in FULL_RUN_REQUIRED_FILES:
            _require((root / relative).is_file(), f"required artifact missing: {relative}")
        _validate_arguments(root, job, hashes)
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        _require(detailed.get("schema_version") == "topo-scene-v4.9.2.detailed-evaluation/v1", "detailed schema drifted")
        _require(detailed.get("formal_test_accessed") is (job.stage == "formal"), "formal access marker drifted")
        if job.algorithm == V49_CANDIDATE:
            _require((root / "collision_label_diagnostics.json").is_file(), "collision diagnostics missing")
            _validate_candidate(root, job)
        else:
            _validate_control(root)
    except (OSError, KeyError, TypeError, ValueError, V492ProtocolError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


@contextmanager
def _parent_row_view() -> Iterator[None]:
    original = parent.run_directory
    parent.run_directory = run_directory
    try:
        yield
    finally:
        parent.run_directory = original


def _run_row(job: Job) -> dict[str, Any]:
    with _parent_row_view():
        row = parent._run_row(job)
    root = run_directory(job)
    selector = _load_json(root / "selector" / "receipt.json")
    row.update(
        {
            "strict_target_only_protocol": job.algorithm == V49_CANDIDATE,
            "fusion_candidate_present": selector.get("fusion_candidate_present", False),
            "actor_confidence_threshold_present": selector.get("actor_confidence_threshold_present", False),
            "external_kinematic_projection": selector.get("external_kinematic_projection", False),
            "action_postprocessing_override": selector.get("action_postprocessing_override", False),
        }
    )
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return parent.aggregate_rows(rows)


def summarize_stage(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str]) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job) for job in jobs if accepted_run(job, hashes)]
    payload = {
        "schema_version": "topo-scene-v4.9.2.stage-results/v1",
        "stage": stage,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "formal_test_accessed": stage == "formal" and bool(rows),
        **hashes,
        "accepted_runs": len(rows),
        "expected_runs": len(jobs),
        "complete": len(rows) == len(jobs),
        "per_run": rows,
        "aggregate": aggregate_rows(rows),
    }
    root = stage_root(stage)
    _write_json(root / "summary.json", payload)
    _write_json(root / "per_run.json", rows)
    if rows:
        flat = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        with (root / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=flat)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in flat})
    return payload


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def _candidate_gate(row: dict[str, Any], parent_contract: dict[str, Any]) -> dict[str, Any]:
    gate = parent._candidate_gate(row, parent_contract)
    strict = {
        "selected_decoder_target_critic": row.get("selected_deployment_decoder") == TARGET_DECODER,
        "strict_target_only_protocol": row.get("strict_target_only_protocol") is True,
        "fusion_candidate_absent": row.get("fusion_candidate_present") is False,
        "actor_confidence_threshold_absent": row.get("actor_confidence_threshold_present") is False,
        "external_kinematic_projection_absent": row.get("external_kinematic_projection") is False,
        "action_postprocessing_override_absent": row.get("action_postprocessing_override") is False,
    }
    gate["checks"].update(strict)
    gate["model_integrity_passed"] = gate["model_integrity_passed"] and all(strict.values())
    gate["passed"] = gate["passed"] and all(strict.values())
    return gate


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return parent._paired_deltas(rows)


def compute_promotion_gate(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    parent_contract = parent.validate_contract(parent.load_contract(PARENT_CONTRACT))
    rows = summary["per_run"]
    aggregate = {(row["method"], row["scenario"]): row for row in summary["aggregate"]}
    scenarios = contract["promotion"]["scenarios"]
    success = {scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"] - aggregate[("temporal_graph", scenario)]["success_rate"] for scenario in scenarios}
    collision = {scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"] - aggregate[("temporal_graph", scenario)]["collision_rate"] for scenario in scenarios}
    paired = _paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {
        f"{row['scenario']}__seed{row['seed']}": _candidate_gate(row, parent_contract)
        for row in rows if row["method"] == "selected_v4_candidate"
    }
    checks = {
        "candidate_development_guards_each_seed": all(gate["passed"] for gate in candidate_gates.values()),
        "all_candidate_off_route_zero": all(aggregate[("selected_v4_candidate", scenario)]["off_route_rate"] <= 0.0 for scenario in scenarios),
        "per_scenario_collision_noninferiority": all(delta <= float(rules["per_scenario_collision_noninferiority_margin"]) for delta in collision.values()),
        "worst_paired_seed_success_noninferiority": min(row["success_rate_delta"] for row in paired) >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success.values()) >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": max(success.values()) >= float(rules["positive_effect_success_gain"]) or min(collision.values()) <= -float(rules["positive_effect_collision_reduction"]),
        "train_only_target_only_selection_applied": all(row["selected_checkpoint_kind"] in ("highest_training_success", "exact_final") and row["selected_deployment_decoder"] == TARGET_DECODER and row.get("fusion_candidate_present") is False for row in rows if row["method"] == "selected_v4_candidate"),
    }
    receipt = {
        "schema_version": "topo-scene-v4.9.2.promotion-gate/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if all(checks.values()) else "fail",
        **hashes,
        "promotion_results": str((stage_root("promotion") / "summary.json").resolve()),
        "promotion_results_sha256": _sha256(stage_root("promotion") / "summary.json"),
        "formal_algorithms": list(V49_FORMAL_ALGORITHMS),
        "checks": checks,
        "candidate_per_seed_gates": candidate_gates,
        "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision,
        "worst_paired_seed_success_delta": min(row["success_rate_delta"] for row in paired),
        "macro_success_delta": _mean(success.values()),
        "paired_seed_deltas": paired,
        "formal_test_unlocked": all(checks.values()),
        "formal_test_accessed": False,
    }
    _write_json(DEFAULT_PROMOTION_GATE, receipt)
    return receipt


def compute_formal_decision(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    receipt = {
        "schema_version": "topo-scene-v4.9.2.formal-decision/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        **hashes,
        "formal_test_accessed": True,
        "summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "status": "complete_pending_statistical_audit",
    }
    _write_json(stage_root("formal") / "formal_decision.json", receipt)
    return receipt


def _assert_stage_can_run(hashes: dict[str, str], stage: str) -> None:
    validate_implementation_freeze()
    validate_engineering_receipt()
    validate_development_adoption()
    if stage == "formal":
        receipt = _load_json(DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        for key, value in hashes.items():
            _require(receipt.get(key) == value, f"formal lock {key} mismatch")


def write_plan(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, device: str) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    payload = {
        "schema_version": "topo-scene-v4.9.2.stage-plan/v1",
        "stage": stage,
        "device": device,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        **hashes,
        "jobs": [{**asdict(job), "name": job.name, "run_directory": str(run_directory(job).resolve()), "accepted": accepted_run(job, hashes), "command": command_for(job, hashes, device=device)} for job in jobs],
    }
    _write_json(stage_root(stage) / "stage_plan.json", payload)
    return payload


def _select_jobs(jobs: list[Job], selector: str | None) -> list[Job]:
    if selector in (None, "all"):
        return jobs
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [job for job in jobs if job.name in requested]
    _require(len(selected) == len(requested), f"unknown or ambiguous job selector: {selector}")
    return selected


def execute_jobs(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, selector: str | None, device: str, workers: int, dry_run: bool) -> int:
    _assert_stage_can_run(hashes, stage)
    _require(workers == 1, "v4.9.2 freezes one sequential GPU worker")
    jobs = jobs_for_stage(contract, digest, stage)
    selected = _select_jobs(jobs, selector)
    plan = write_plan(contract, digest, stage, hashes, device=device)
    if dry_run:
        print(json.dumps({"selected": [job.name for job in selected], "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    log_root = stage_root(stage) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for job in selected:
        if accepted_run(job, hashes):
            attempts.append({"job": job.name, "status": "already_accepted"})
            continue
        directory = run_directory(job)
        if directory.exists():
            _, reason = accepted_run_reason(job, hashes)
            attempts.append({"job": job.name, "status": "immutable_incomplete_run", "acceptance_reason": reason})
            continue
        log_path = log_root / f"{job.name}.log"
        print(json.dumps({"job": job.name, "status": "running", "log": str(log_path)}, ensure_ascii=False), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command_for(job, hashes, device=device), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
        accepted, reason = accepted_run_reason(job, hashes)
        status = "accepted" if completed.returncode == 0 and accepted else "failed"
        attempt = {"job": job.name, "status": status, "returncode": completed.returncode, "acceptance_reason": reason, "log": str(log_path)}
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)
    execution = {
        "schema_version": "topo-scene-v4.9.2.execution/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        **hashes,
        "selected": [job.name for job in selected],
        "attempts": attempts,
        "accepted_after_batch": sum(accepted_run(job, hashes) for job in jobs),
        "expected": len(jobs),
        "formal_stage_launched": stage == "formal",
    }
    _write_json(stage_root(stage) / "last_execution.json", execution)
    summarize_stage(contract, digest, stage, hashes)
    failures = [row for row in attempts if row["status"] not in {"accepted", "already_accepted"}]
    return 1 if failures else 0


def status_payload(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    stages = {}
    for stage in STAGES:
        jobs = jobs_for_stage(contract, digest, stage)
        accepted = sum(accepted_run(job, hashes) for job in jobs)
        stages[stage] = {"accepted": accepted, "expected": len(jobs), "complete": accepted == len(jobs)}
    promotion = _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    return {
        "schema_version": "topo-scene-v4.9.2.status/v1",
        **hashes,
        "development_adoption_decision": "pass",
        "stages": stages,
        "formal_test_unlocked": bool(promotion and promotion.get("decision") == "pass"),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    root.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    plan = sub.add_parser("plan")
    plan.add_argument("--stage", choices=STAGES, required=True)
    plan.add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--job", default="all")
    run.add_argument("--device", default="cuda")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--stage", choices=STAGES, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--stage", choices=STAGES, required=True)
    sub.add_parser("status")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract_path = args.contract.resolve()
    _require(_sha256(contract_path) == CONTRACT_SHA256, "contract bytes drifted")
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    if args.command == "validate":
        hashes = protocol_hashes()
        value = {"status": "valid", **hashes, "formal_test_accessed": False}
    else:
        hashes = protocol_hashes()
        if args.command == "plan":
            value = write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
        elif args.command == "summarize":
            value = summarize_stage(contract, digest, args.stage, hashes)
        elif args.command == "gate":
            value = compute_promotion_gate(contract, digest, hashes) if args.stage == "promotion" else compute_formal_decision(contract, digest, hashes)
        else:
            value = status_payload(contract, digest, hashes)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
