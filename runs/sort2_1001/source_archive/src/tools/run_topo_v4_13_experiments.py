"""Run the hash-bound v4.13 experiment protocol without stage fail-fast."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sb3_configs_v4_13 import (
    TEMPORAL_GRAPH,
    V413_FORMAL_ALGORITHMS,
    V413_FULL,
    V413_IMPLEMENTATION_IDS,
    V413_ISOLATED_UNTEMPERED,
    V413_NO_CROSS_AUGMENTATION,
    V413_UNISOLATED_TEMPERED,
)
from tools.checkpoint_decoder_selector_v4_6 import (
    PARENT_DECODER,
    TARGET_DECODER,
)
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE


PYTHON_EXECUTABLE = Path("D:/Programs/Anaconda/envs/llm_pipeline/python.exe")
DEFAULT_CONTRACT = (
    ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_13.yaml"
)
DEFAULT_PREREGISTRATION = (
    ROOT / "results_topo_v4_13_dev" / "preregistration_receipt.json"
)
DEFAULT_ENGINEERING_ROOT = ROOT / "results_topo_v4_13_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_DEVELOPMENT_DECISION = (
    ROOT / "results_topo_v4_13_dev" / "development" / "development_decision.json"
)
DEFAULT_ABLATION_DECISION = (
    ROOT / "results_topo_v4_13_dev" / "ablation" / "ablation_decision.json"
)
DEFAULT_PROMOTION_GATE = ROOT / "results_topo_v4_13_promotion" / "promotion_gate.json"
PARENT_DEVELOPMENT_SUMMARY = (
    ROOT / "results_topo_v4_12_dev" / "development" / "summary.json"
)
PARENT_DEVELOPMENT_DECISION = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "development_decision.json"
)
PARENT_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
    / "complete_attribution.json"
)
PARENT_FREEZE = (
    ROOT / "results_topo_v4_12_dev" / "engineering" / "implementation_freeze.json"
)

CONTRACT_SHA256 = "64705b16ec83b62b2dd9113d9c21bc4aa70033bc9402ef7d5cca69447a24a2fd"
PARENT_DEVELOPMENT_SUMMARY_SHA256 = "3a39ef71c61b94500c5c9d6a58fd6ddf812da0720769d4489e07205b1ef283a6"
PARENT_DEVELOPMENT_DECISION_SHA256 = "26910e81abb8590cf297a0962e9957953baae0c7a3fa9b6adb1e71ef2e329181"
PARENT_ATTRIBUTION_SHA256 = "40c32791653d52756fb3e5e7c4a86d8b04be5b24bae6f32dbad71b5e15eff0b2"
PARENT_FREEZE_SHA256 = "bb8ea7dd83e424d125d4fbb1b7dd9d0837208defc7e9c37665dbb5c83397e0fa"

STAGES = ("development", "ablation", "promotion", "formal")
METHOD_ALGORITHMS = {
    "temporal_graph": TEMPORAL_GRAPH,
    "gits_prcr_full": V413_FULL,
    "unisolated_tempered": V413_UNISOLATED_TEMPERED,
    "isolated_untempered": V413_ISOLATED_UNTEMPERED,
    "isolated_tempered_no_cross_augmentation": V413_NO_CROSS_AUGMENTATION,
}
METHOD_CODES = {
    "temporal_graph": "tg",
    "gits_prcr_full": "gits",
    "unisolated_tempered": "uit",
    "isolated_untempered": "giu",
    "isolated_tempered_no_cross_augmentation": "gna",
}
SCENARIO_CODES = {
    "left_turn": "lt",
    "cross": "x",
    "roundabout_easy": "rae",
    "roundabout_medium": "ram",
    "roundabout": "ra",
    "carla": "ca",
}
REPORT_ROOTS = {
    "development": ROOT / "results_topo_v4_13_dev" / "development",
    "ablation": ROOT / "results_topo_v4_13_dev" / "ablation",
    "promotion": ROOT / "results_topo_v4_13_promotion",
    "formal": ROOT / "results_topo_v4_13_formal",
}
SHORT_RUN_ROOTS = {
    "development": ROOT / "r413" / "d",
    "ablation": ROOT / "r413" / "a",
    "promotion": ROOT / "r413" / "p",
    "formal": ROOT / "r413" / "f",
}
MAX_IMMUTABLE_ATTEMPTS = 4
NUMERIC_EPSILON = 1e-12
COMMON_REQUIRED_FILES = (
    "arguments.json",
    "method_metadata.json",
    "training_diagnostics.json",
    "return_estimator_diagnostics.json",
    "checkpoint_audit.json",
    "final_model.zip",
    "selected_model.zip",
    "performance_profile.json",
    "final_evaluation.json",
    "action_diagnostics.json",
    "action_diagnostics_decisions.jsonl",
    "paper_evaluation_detailed.json",
    "selector/receipt.json",
)
FORBIDDEN_RUNTIME_FIELDS = (
    "inference_safety_rule_added",
    "external_kinematic_projection",
    "kinematic_safety_projection",
    "traffic_risk_in_lane_mask",
    "ttc_or_headway_threshold",
    "lane_change_veto",
    "actor_confidence_gate",
    "action_postprocessing_override",
    "scenario_conditioned_inference_rule",
)


class V413ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V413ProtocolError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.13 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.13.gradient-isolated-tempered-joint-support-contract/v1",
        "unexpected v4.13 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication drifted")
    _require(contract.get("formal_test_locked") is True, "formal lock drifted")
    lineage = contract.get("lineage", {})
    expected_lineage = {
        "parent_contract_sha256": "3f978ef94df4e601abcbc55d0c7ba56fe817f09848809a3a009dfe950e444c3f",
        "parent_implementation_freeze_sha256": PARENT_FREEZE_SHA256,
        "complete_parent_development_summary_sha256": PARENT_DEVELOPMENT_SUMMARY_SHA256,
        "failed_parent_development_decision_sha256": PARENT_DEVELOPMENT_DECISION_SHA256,
        "complete_parent_attribution_sha256": PARENT_ATTRIBUTION_SHA256,
        "lane_prior_zero_matrix_sha256": "135e2cb8b8cf4c259c72d6dfb5aabfe1e028d4f5f0abb83c0d8e3176c3d0db94",
        "lane_prior_0_10_sensitivity_sha256": "b161c649a83eb08c19af1aa655cba131a45f15b7faf7a293600ada1396f08592",
        "lane_speed_gradient_diagnostic_sha256": "a951fb777c8af87d4cbd1c498b9ae6ae7956d1305cb56cd1e44d20dd82d2b553",
        "formal_test_accessed": False,
    }
    for key, expected in expected_lineage.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")
    model = contract.get("model_change", {})
    lane = model.get("lane_categorical_nll", {})
    speed = model.get("conditional_speed_mixture_nll", {})
    _require(model.get("same_lane_head_parameters") is True, "lane head drifted")
    _require(model.get("new_inference_head") is False, "new inference head enabled")
    _require(model.get("inference_score_equation_changed") is False, "inference score drifted")
    _require(lane.get("scale_inside_joint_nll") == 0.25, "lane scale drifted")
    _require(lane.get("gradient_targets") == ["deployed_lane_head"], "lane gradient target drifted")
    _require("actor_latent_trunk" in lane.get("zero_gradient_targets", []), "latent isolation missing")
    _require(speed.get("unchanged") is True, "speed support changed")
    _require(model.get("replay_support_outer_coefficient") == 0.05, "outer support drifted")
    _require(model.get("lane_log_prior_coefficient") == 0.05, "lane prior drifted")
    for key in (
        "kinematic_safety_projection",
        "ttc_or_headway_threshold",
        "geometry_unsafe_label",
        "lane_change_veto",
        "actor_confidence_gate",
        "safety_shield_or_rule_fallback",
        "semantic_tie_override",
        "post_decoder_action_rewrite",
        "scenario_conditioned_inference_rule",
    ):
        _require(contract.get("forbidden", {}).get(key) is True, f"forbidden.{key} missing")
    development = contract["development"]
    _require(development.get("execution_policy") == "attempt_all_then_summarize", "development fail-fast drifted")
    _require(development.get("failure_does_not_cancel_remaining_jobs") is True, "development cancellation enabled")
    _require(len(development.get("ordered_primary_cells", [])) == 4, "development matrix drifted")
    _require(len(development.get("ordered_ablation_cells_after_primary_pass", [])) == 3, "ablation matrix drifted")
    promotion = contract["promotion"]
    _require(promotion.get("methods") == ["temporal_graph", "gits_prcr_full"], "promotion methods drifted")
    _require(promotion.get("scenarios") == ["cross", "roundabout_medium", "carla"], "promotion scenarios drifted")
    _require(promotion.get("seeds") == [20, 21], "promotion seeds drifted")
    _require(int(promotion.get("jobs")) == 12, "promotion count drifted")
    formal = contract["formal_test"]
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal state drifted")
    _require(len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"]) == 120, "formal matrix drifted")
    return contract


def validate_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    value = _load_json(path)
    _require(value.get("schema_version") == "topo-scene-v4.13.preregistration-receipt/v1", "preregistration schema drifted")
    _require(value.get("contract", {}).get("sha256") == CONTRACT_SHA256, "preregistration contract drifted")
    _require(value.get("formal_test_accessed") is False, "preregistration accessed formal")
    _require(value.get("failure_does_not_cancel_remaining_jobs_in_stage") is True, "fail-fast preregistered")
    return value


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    value = _load_json(path)
    _require(value.get("schema_version") == "topo-scene-v4.13.implementation-freeze/v1", "freeze schema drifted")
    _require(value.get("status") == "passed_and_frozen", "implementation is not frozen")
    _require(value.get("experiment_contract_sha256") == CONTRACT_SHA256, "freeze contract drifted")
    _require(value.get("formal_test_accessed") is False, "freeze accessed formal")
    _require(value.get("no_kinematic_projection_audit_passed") is True, "projection audit failed")
    _require(value.get("gradient_isolation_gate_passed") is True, "gradient gate failed")
    _require(value.get("real_sumo_smoke_passed") is True, "SUMO smoke failed")
    return value


def validate_engineering_receipt(path: Path = DEFAULT_ENGINEERING_RECEIPT) -> dict[str, Any]:
    value = _load_json(path)
    _require(value.get("schema_version") == "topo-scene-v4.13.engineering-receipt/v1", "engineering receipt schema drifted")
    _require(value.get("status") == "passed", "engineering receipt failed")
    _require(value.get("implementation_freeze_sha256") == _sha256(DEFAULT_FREEZE), "engineering receipt freeze drifted")
    _require(value.get("formal_test_accessed") is False, "engineering accessed formal")
    return value


def validate_development_decision() -> dict[str, Any]:
    value = _load_json(DEFAULT_DEVELOPMENT_DECISION)
    _require(value.get("decision") == "pass", "development gate did not pass")
    _require(value.get("formal_test_accessed") is False, "development accessed formal")
    return value


def validate_ablation_decision() -> dict[str, Any]:
    value = _load_json(DEFAULT_ABLATION_DECISION)
    _require(value.get("decision") == "complete", "ablation is incomplete")
    _require(value.get("formal_test_accessed") is False, "ablation accessed formal")
    return value


def protocol_hashes() -> dict[str, str]:
    expected = {
        DEFAULT_CONTRACT: CONTRACT_SHA256,
        PARENT_DEVELOPMENT_SUMMARY: PARENT_DEVELOPMENT_SUMMARY_SHA256,
        PARENT_DEVELOPMENT_DECISION: PARENT_DEVELOPMENT_DECISION_SHA256,
        PARENT_ATTRIBUTION: PARENT_ATTRIBUTION_SHA256,
        PARENT_FREEZE: PARENT_FREEZE_SHA256,
    }
    for path, digest in expected.items():
        _require(_sha256(path) == digest, f"frozen lineage bytes drifted: {path}")
    validate_contract(load_contract())
    validate_preregistration()
    validate_implementation_freeze()
    validate_engineering_receipt()
    return {
        "experiment_contract_sha256": CONTRACT_SHA256,
        "stage0_results_sha256": PARENT_DEVELOPMENT_SUMMARY_SHA256,
        "attribution_sha256": PARENT_ATTRIBUTION_SHA256,
        "preregistration_receipt_sha256": _sha256(DEFAULT_PREREGISTRATION),
        "implementation_freeze_sha256": _sha256(DEFAULT_FREEZE),
        "engineering_receipt_sha256": _sha256(DEFAULT_ENGINEERING_RECEIPT),
    }


@dataclass(frozen=True)
class Job:
    stage: str
    job_id: str
    kind: str
    method: str
    algorithm: str
    implementation_id: str
    scenario: str
    seed: int
    raw_steps: int
    calibration_episodes: int
    calibration_seed_start: int
    evaluation_episodes: int
    evaluation_split: str
    evaluation_seed_start: int
    role: str
    protocol_tag: str

    @property
    def name(self) -> str:
        prefix = self.job_id if self.stage in {"development", "ablation"} else self.stage[0].upper()
        return f"{prefix}__{METHOD_CODES[self.method]}__{SCENARIO_CODES[self.scenario]}__s{self.seed}__p{self.protocol_tag}"


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.13 stage {stage!r}")
    tag = digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        roles = {"D1": "cross_primary", "D2": "cross_replication", "D3": "roundabout_recovery", "D4": "carla_retention"}
        for row in settings["ordered_primary_cells"]:
            method = str(row["method"])
            algorithm = METHOD_ALGORITHMS[method]
            identifier = str(row["id"])
            jobs.append(Job(stage, identifier, "fresh_primary_development", method, algorithm, V413_IMPLEMENTATION_IDS[algorithm], str(row["scenario"]), int(row["seed"]), int(row["raw_steps"]), 12, int(row["calibration_seed_start"]), int(settings["evaluation_episodes"]), str(settings["evaluation_split"]), int(row["evaluation_seed_start"]), roles[identifier], tag))
    elif stage == "ablation":
        settings = contract["development"]
        calibration = {"A1": 153000, "A2": 153000, "A3": 151000}
        for row in settings["ordered_ablation_cells_after_primary_pass"]:
            method = str(row["method"])
            algorithm = METHOD_ALGORITHMS[method]
            identifier = str(row["id"])
            jobs.append(Job(stage, identifier, "required_mechanism_ablation", method, algorithm, V413_IMPLEMENTATION_IDS[algorithm], str(row["scenario"]), int(row["seed"]), int(row["raw_steps"]), 12, calibration[identifier], int(settings["evaluation_episodes"]), str(settings["evaluation_split"]), int(row["evaluation_seed_start"]), "mechanism_ablation_not_used_for_selection", tag))
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        for method_value in settings["methods"]:
            method = str(method_value)
            algorithm = METHOD_ALGORITHMS[method]
            for scenario in settings["scenarios"]:
                for seed_value in settings["seeds"]:
                    seed = int(seed_value)
                    if stage == "promotion":
                        offset = seed - 20
                        calibration_seed = 170000 + 1000 * offset
                        evaluation_seed = 270000 + 1000 * offset
                    else:
                        calibration_seed = 310000 + 1000 * seed
                        evaluation_seed = 510000 + 1000 * seed
                    jobs.append(Job(stage, "P" if stage == "promotion" else "F", "fresh_paired_model_training", method, algorithm, V413_IMPLEMENTATION_IDS[algorithm], str(scenario), seed, int(settings["raw_training_steps"]), 12, calibration_seed, int(settings["evaluation_episodes"]), str(settings["evaluation_split"]), evaluation_seed, "fresh_paired_promotion" if stage == "promotion" else "untouched_formal_test", tag))
    expected = {"development": 4, "ablation": 3, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} job")
    return jobs


def stage_root(stage: str) -> Path:
    return REPORT_ROOTS[stage]


def run_directory(job: Job, attempt: int = 0) -> Path:
    suffix = "" if attempt == 0 else f"__r{attempt}"
    return SHORT_RUN_ROOTS[job.stage] / f"{job.name}{suffix}"


def attempt_directories(job: Job) -> list[Path]:
    return [run_directory(job, attempt) for attempt in range(MAX_IMMUTABLE_ATTEMPTS)]


def command_for(job: Job, hashes: dict[str, str], *, device: str, attempt: int = 0) -> list[str]:
    directory = run_directory(job, attempt)
    command = [
        str(PYTHON_EXECUTABLE), str(ROOT / "tools" / "train_paper_sb3_sumo_v4_13.py"),
        "--scenario", job.scenario, "--algo", job.algorithm,
        "--max-steps", str(job.raw_steps), "--learning-starts", "5000",
        "--checkpoint-freq", str(job.raw_steps), "--eval-freq", "0",
        "--eval-episodes", str(job.evaluation_episodes),
        "--evaluation-split", job.evaluation_split,
        "--evaluation-seed-start", str(job.evaluation_seed_start),
        "--calibration-episodes", str(job.calibration_episodes),
        "--calibration-seed-start", str(job.calibration_seed_start),
        "--seed", str(job.seed), "--device", device, "--batch-size", "32",
        "--learning-rate", "0.0001", "--discount", "0.99",
        "--buffer-size", "20000", "--action-repeat", "3",
        "--ego-control-profile", "direct", "--traffic-protocol", "frozen_60_20_20",
        "--episode-limit-profile", "source", "--output-dir", str(directory.parent.resolve()),
        "--model-name", directory.name,
        "--experiment-contract-sha256", hashes["experiment_contract_sha256"],
        "--stage0-results-sha256", hashes["stage0_results_sha256"],
        "--attribution-sha256", hashes["attribution_sha256"],
        "--implementation-freeze-sha256", hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(["--formal-unlock-receipt", str(DEFAULT_PROMOTION_GATE.resolve())])
    return command


def _all_false(surface: dict[str, Any]) -> bool:
    return all(surface.get(key) is False for key in FORBIDDEN_RUNTIME_FIELDS if key in surface)


def _finite_stat(training: dict[str, Any], name: str) -> bool:
    row = training.get("statistics", {}).get(name, {})
    value = row.get("last")
    return int(row.get("count", 0)) > 0 and value is not None and math.isfinite(float(value))


def _variant_expected(algorithm: str, scenario: str) -> tuple[bool, float, bool, bool]:
    isolated, lane_scale, cross_aug = {
        V413_FULL: (True, 0.25, True),
        V413_UNISOLATED_TEMPERED: (False, 0.25, True),
        V413_ISOLATED_UNTEMPERED: (True, 1.0, True),
        V413_NO_CROSS_AUGMENTATION: (True, 0.25, False),
    }[algorithm]
    return isolated, lane_scale, cross_aug, scenario != "cross" or cross_aug


def _validate_selector(root: Path, *, candidate: bool) -> dict[str, Any]:
    value = _load_json(root / "selector" / "receipt.json")
    expected_decoder = TARGET_DECODER if candidate else PARENT_DECODER
    _require(value.get("selected_deployment_decoder") == expected_decoder, "selector decoder drifted")
    _require(value.get("validation_used_for_selection") is False, "selector used validation")
    _require(value.get("formal_test_used_for_selection") is False, "selector used formal")
    _require(value.get("selector_receipt_precedes_validation_environment") is True, "selector was not sealed")
    _require(value.get("policy_parameter_state_preserved") is True, "selector changed policy parameters")
    _require(value.get("source_policy_parameter_state_sha256") == value.get("selected_model_parameter_state_sha256"), "selector parameter hash drifted")
    _require(_sha256(root / "selected_model.zip") == value.get("selected_model_sha256"), "selected model hash drifted")
    if candidate:
        _require(value.get("schema_version") == "topo-scene-v4.13.target-only-tie-replicated-selector/v1", "candidate selector schema drifted")
        _require(value.get("selector_mode") == SELECTOR_MODE, "selector mode drifted")
        _require(value.get("deployment_decoder_candidates") == [TARGET_DECODER], "non-target decoder entered selector")
        _require(value.get("fusion_candidate_present") is False, "fusion candidate entered selector")
        _require(value.get("actor_confidence_threshold_present") is False, "confidence threshold entered selector")
    return value


def _validate_run(root: Path, job: Job, hashes: dict[str, str]) -> None:
    _require(root.is_dir(), "run directory missing")
    for relative in COMMON_REQUIRED_FILES:
        _require((root / relative).is_file(), f"required artifact missing: {relative}")
    arguments = _load_json(root / "arguments.json")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    metadata = _load_json(root / "method_metadata.json")
    actions = _load_json(root / "action_diagnostics.json")
    training = _load_json(root / "training_diagnostics.json")
    _require(arguments.get("schema_version") == "topo-scene-v4.13.run-arguments/v1", "arguments schema drifted")
    requested = arguments.get("requested_raw_steps", {})
    expected_requested = {
        "scenario": job.scenario, "algo": job.algorithm, "max_steps": job.raw_steps,
        "seed": job.seed, "batch_size": 32, "learning_rate": 0.0001,
        "discount": 0.99, "learning_starts": 5000, "buffer_size": 20000,
        "action_repeat": 3, "ego_control_profile": "direct",
        "traffic_protocol": "frozen_60_20_20", "eval_episodes": job.evaluation_episodes,
        "evaluation_split": job.evaluation_split, "evaluation_seed_start": job.evaluation_seed_start,
        "calibration_seed_start": job.calibration_seed_start, "calibration_episodes": job.calibration_episodes,
        "experiment_contract_sha256": hashes["experiment_contract_sha256"],
        "stage0_results_sha256": hashes["stage0_results_sha256"],
        "attribution_sha256": hashes["attribution_sha256"],
        "implementation_freeze_sha256": hashes["implementation_freeze_sha256"],
    }
    for key, expected in expected_requested.items():
        _require(requested.get(key) == expected, f"argument {key} drifted")
    fidelity = arguments.get("implementation_fidelity", {})
    _require(fidelity.get("implementation_id") == job.implementation_id, "argument implementation id drifted")
    _require(_all_false(fidelity), "argument admits rule/projection")
    _require(detailed.get("schema_version") == "topo-scene-v4.13.detailed-evaluation/v1", "detailed schema drifted")
    _require(detailed.get("algorithm") == job.algorithm and detailed.get("scenario") == job.scenario, "detailed identity drifted")
    _require(detailed.get("evaluation_split") == job.evaluation_split, "evaluation split drifted")
    _require(detailed.get("formal_test_accessed") is (job.stage == "formal"), "formal marker drifted")
    _require(int(detailed.get("trained_raw_steps", 0)) == job.raw_steps, "training budget drifted")
    _require(int(detailed.get("summary", {}).get("episodes", 0)) == job.evaluation_episodes, "evaluation episodes drifted")
    provenance = detailed.get("evaluation_provenance", {})
    _require(provenance.get("validated") is True and provenance.get("traffic_partition") == job.evaluation_split, "evaluation provenance drifted")
    _require(provenance.get("evaluation_seed_start") == job.evaluation_seed_start, "evaluation seed drifted")
    _require(_all_false(metadata) and _all_false(actions) and _all_false(training) and _all_false(detailed), "runtime admits rule/projection")
    candidate = job.algorithm != TEMPORAL_GRAPH
    selector = _validate_selector(root, candidate=candidate)
    _require(detailed.get("selector_receipt_sha256") == _sha256(root / "selector" / "receipt.json"), "evaluation selector binding drifted")
    if not candidate:
        _require(metadata.get("implementation_id") == job.implementation_id, "control implementation drifted")
        _require(metadata.get("network_changed") is False, "control network changed")
        _require(actions.get("learned_collision_critic_present") is False, "control gained collision critic")
        _require(not (root / "collision_label_diagnostics.json").exists(), "control emitted collision labels")
        return
    _require((root / "collision_label_diagnostics.json").is_file(), "candidate collision labels missing")
    labels = _load_json(root / "collision_label_diagnostics.json")
    isolated, lane_scale, cross_aug, effective_aug = _variant_expected(job.algorithm, job.scenario)
    target = "deployed_lane_head_only" if isolated else "deployed_lane_head_and_actor_latent_trunk"
    expected_metadata = {
        "implementation_id": job.implementation_id,
        "shared_online_scene_encoder": True, "actor_encoder_detach": True,
        "lane_support_gradient_isolated": isolated, "lane_support_gradient_target": target,
        "speed_components_per_lane": 3,
        "replay_support_objective": "gradient_isolated_masked_lane_categorical_nll_plus_conditional_speed_mixture_nll",
        "replay_support_coef": 0.05, "lane_support_scale": lane_scale,
        "lane_prior_coef": 0.05, "component_prior_coef": 0.05,
        "cross_rotation_augmentation": cross_aug,
        "effective_random_rotation_augmentation": effective_aug,
        "collision_risk_coef": 1.0,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": 0.25, "collision_temporal_consistency_coef": 0.05,
        "deployment_decoder_candidates": [TARGET_DECODER],
        "fusion_candidate_present": False, "actor_confidence_threshold_present": False,
        "inference_score_equation_changed_from_v4_12": False,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"method metadata {key} drifted")
    exact_keys = (
        "selected_decoder_exact_model_match_rate", "selected_decoder_exact_action_match_rate",
        "selected_action_mask_feasible_rate", "exact_joint_support_model_argmax_rate",
        "joint_support_score_equation_match_rate", "selected_speed_exact_proposal_match_rate",
        "collision_value_bounded_rate", "learned_proposal_source_rate", "no_action_rewrite_rate",
        "no_semantic_tie_override_rate", "no_actor_confidence_threshold_rate",
    )
    for key in exact_keys:
        _require(actions.get(key) == 1.0, f"action diagnostic {key} failed")
    _require(int(actions.get("selected_decoder_records", 0)) > 0, "no action records")
    _require(actions.get("joint_replay_action_support_present") is True, "joint support missing")
    _require(actions.get("lane_support_gradient_isolated") is isolated, "gradient isolation runtime drifted")
    _require(actions.get("lane_support_gradient_target") == target, "gradient target runtime drifted")
    _require(actions.get("same_deployed_lane_head_for_support") is True, "support head split")
    _require(actions.get("new_inference_head") is False, "new inference head present")
    _require(actions.get("lane_support_scale") == lane_scale and actions.get("lane_prior_coef") == 0.05, "action support coefficients drifted")
    _require(actions.get("effective_random_rotation_augmentation") is effective_aug, "augmentation runtime drifted")
    _require(actions.get("optimizer_ownership", {}).get("overlap_count") == 0, "optimizer owners overlap")
    for name in (
        "train/actor_loss", "train/critic_loss", "train/collision_critic_loss",
        "train/collision_soft_bce_loss", "train/collision_pairwise_rank_loss",
        "train/collision_temporal_consistency_loss", "train/representation_loss",
        "support/replay_joint_action_nll", "support/replay_joint_action_nll_recomputed",
        "support/replay_lane_categorical_nll", "support/replay_conditional_speed_mixture_nll",
        "risk/policy_expected_collision_cost", "risk/learned_uncertainty",
    ):
        _require(_finite_stat(training, name), f"training statistic {name} invalid")
    _require(training.get("lane_support_gradient_isolated") is isolated, "training gradient isolation drifted")
    _require(training.get("lane_support_scale") == lane_scale and training.get("lane_prior_coef") == 0.05, "training coefficients drifted")
    _require(labels.get("source") == "observed_environment_info_collision", "collision label source drifted")
    _require(labels.get("future_or_oracle_labels_used") is False, "oracle labels used")
    _require(int(labels.get("sampled_label_count", 0)) > 0, "no collision labels sampled")
    _require(detailed.get("scientific_version") == "v4.13_gradient_isolated_tempered_joint_support_prcr", "scientific version drifted")
    _require(detailed.get("lane_support_gradient_isolated") is isolated, "detailed gradient isolation drifted")
    _require(detailed.get("selected_model_policy_class") == "GradientIsolatedTemperedJointSupportPolicyV413", "selected policy class drifted")
    _require(selector.get("selected_model_policy_class") == "GradientIsolatedTemperedJointSupportPolicyV413", "selector policy class drifted")


def accepted_run_location(job: Job, hashes: dict[str, str]) -> Path | None:
    for root in attempt_directories(job):
        try:
            _validate_run(root, job, hashes)
        except (OSError, KeyError, TypeError, ValueError, V413ProtocolError, json.JSONDecodeError):
            continue
        return root
    return None


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    reasons = []
    for root in attempt_directories(job):
        if not root.exists():
            continue
        try:
            _validate_run(root, job, hashes)
            return True, f"accepted:{root}"
        except (OSError, KeyError, TypeError, ValueError, V413ProtocolError, json.JSONDecodeError) as exc:
            reasons.append(f"{root.name}: {exc}")
    return False, "; ".join(reasons) if reasons else "run directory missing"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_location(job, hashes) is not None


def _run_row(job: Job, hashes: dict[str, str]) -> dict[str, Any]:
    root = accepted_run_location(job, hashes)
    _require(root is not None, f"job is not accepted: {job.name}")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    summary = detailed["summary"]
    selector = _load_json(root / "selector" / "receipt.json")
    actions = _load_json(root / "action_diagnostics.json")
    row = {
        "stage": job.stage, "job_id": job.job_id, "job": job.name,
        "evidence_directory": str(root.resolve()), "kind": job.kind, "role": job.role,
        "method": job.method, "algorithm": job.algorithm,
        "implementation_id": job.implementation_id, "scenario": job.scenario,
        "seed": job.seed, "raw_steps": job.raw_steps,
        "evaluation_episodes": int(summary["episodes"]),
        **{key: float(summary[key]) for key in ("success_rate", "collision_rate", "off_route_rate", "timeout_rate", "mean_return", "mean_decision_steps", "mean_raw_steps")},
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_model_sha256": _sha256(root / "selected_model.zip"),
        "selector_receipt_sha256": _sha256(root / "selector" / "receipt.json"),
        "action_diagnostics_sha256": _sha256(root / "action_diagnostics.json"),
        **{key: actions.get(key, False) for key in FORBIDDEN_RUNTIME_FIELDS},
        "selected_action_mask_feasible_rate": actions.get("selected_action_mask_feasible_rate"),
    }
    if job.algorithm != TEMPORAL_GRAPH:
        labels = _load_json(root / "collision_label_diagnostics.json")
        training = _load_json(root / "training_diagnostics.json")
        statistics = training.get("statistics", {})
        stat = lambda name, field="last": statistics.get(name, {}).get(field)
        row.update({
            "shared_risk_encoder_present": actions.get("shared_risk_encoder_present"),
            "joint_replay_action_support_present": actions.get("joint_replay_action_support_present"),
            "lane_support_gradient_isolated": actions.get("lane_support_gradient_isolated"),
            "lane_support_gradient_target": actions.get("lane_support_gradient_target"),
            "same_deployed_lane_head_for_support": actions.get("same_deployed_lane_head_for_support"),
            "exact_joint_support_model_argmax_rate": actions.get("exact_joint_support_model_argmax_rate"),
            "joint_support_score_equation_match_rate": actions.get("joint_support_score_equation_match_rate"),
            "learned_proposal_source_rate": actions.get("learned_proposal_source_rate"),
            "no_action_rewrite_rate": actions.get("no_action_rewrite_rate"),
            "no_actor_confidence_threshold_rate": actions.get("no_actor_confidence_threshold_rate"),
            "optimizer_overlap_count": actions.get("optimizer_ownership", {}).get("overlap_count"),
            "lane_support_scale": actions.get("lane_support_scale"),
            "lane_prior_coef": actions.get("lane_prior_coef"),
            "effective_random_rotation_augmentation": actions.get("effective_random_rotation_augmentation"),
            "sampled_collision_label_count": labels.get("sampled_label_count"),
            "collision_return_loss": training.get("collision_return_loss"),
            "collision_pairwise_rank_coef": training.get("collision_pairwise_rank_coef"),
            "collision_temporal_consistency_coef": training.get("collision_temporal_consistency_coef"),
            "replay_lane_categorical_nll": stat("support/replay_lane_categorical_nll"),
            "replay_conditional_speed_mixture_nll": stat("support/replay_conditional_speed_mixture_nll"),
            "replay_joint_action_nll": stat("support/replay_joint_action_nll"),
        })
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["scenario"])), []).append(row)
    metrics = ("success_rate", "collision_rate", "off_route_rate", "timeout_rate", "mean_return", "mean_decision_steps", "mean_raw_steps")
    output = []
    for (method, scenario), items in sorted(groups.items()):
        episodes = sum(int(row["evaluation_episodes"]) for row in items)
        output.append({"method": method, "scenario": scenario, "runs": len(items), "episodes": episodes, "seeds": sorted(int(row["seed"]) for row in items), **{key: sum(float(row[key]) * int(row["evaluation_episodes"]) for row in items) / episodes for key in metrics}})
    return output


def summarize_stage(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str]) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job, hashes) for job in jobs if accepted_run(job, hashes)]
    payload = {
        "schema_version": "topo-scene-v4.13.stage-results/v1", "stage": stage,
        "computed_from_real_runs": True, "fabricated_values": False,
        "formal_test_accessed": stage == "formal" and bool(rows),
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True, **hashes,
        "accepted_runs": len(rows), "expected_runs": len(jobs),
        "complete": len(rows) == len(jobs), "per_run": rows,
        "aggregate": aggregate_rows(rows),
    }
    root = stage_root(stage)
    _write_json(root / "summary.json", payload)
    _write_json(root / "per_run.json", rows)
    if rows:
        keys = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        root.mkdir(parents=True, exist_ok=True)
        with (root / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return payload


def _candidate_integrity(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "joint_support_present": row.get("joint_replay_action_support_present") is True,
        "exact_joint_model_argmax": row.get("exact_joint_support_model_argmax_rate") == 1.0,
        "joint_score_equation": row.get("joint_support_score_equation_match_rate") == 1.0,
        "learned_proposal_source": row.get("learned_proposal_source_rate") == 1.0,
        "no_action_rewrite": row.get("no_action_rewrite_rate") == 1.0,
        "no_confidence_threshold": row.get("no_actor_confidence_threshold_rate") == 1.0,
        "optimizer_owners_disjoint": row.get("optimizer_overlap_count") == 0,
        "no_rule_or_projection": all(row.get(key) is False for key in FORBIDDEN_RUNTIME_FIELDS),
        "feasible_lane_mask": row.get("selected_action_mask_feasible_rate") == 1.0,
        "proper_collision_loss": row.get("collision_return_loss") == "soft_target_binary_cross_entropy_with_logits",
        "rank_coefficient": row.get("collision_pairwise_rank_coef") == 0.25,
        "consistency_coefficient": row.get("collision_temporal_consistency_coef") == 0.05,
        "joint_nll_finite": isinstance(row.get("replay_joint_action_nll"), (int, float)) and math.isfinite(float(row["replay_joint_action_nll"])),
        "lane_nll_finite": isinstance(row.get("replay_lane_categorical_nll"), (int, float)) and math.isfinite(float(row["replay_lane_categorical_nll"])),
        "same_deployed_lane_head": row.get("same_deployed_lane_head_for_support") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _metric_gate(row: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, bool]:
    return {
        "success_rate": float(row["success_rate"]) + NUMERIC_EPSILON >= float(thresholds["success_rate_min"]),
        "collision_rate": float(row["collision_rate"]) <= float(thresholds["collision_rate_max"]) + NUMERIC_EPSILON,
        "off_route_rate": float(row["off_route_rate"]) <= float(thresholds["off_route_rate_max"]) + NUMERIC_EPSILON,
        "timeout_rate": float(row["timeout_rate"]) <= float(thresholds["timeout_rate_max"]) + NUMERIC_EPSILON,
    }


def _promotion_guard(scenario: str) -> dict[str, float]:
    return {
        "cross": {"success_rate_min": 0.80, "collision_rate_max": 0.20, "off_route_rate_max": 0.0, "timeout_rate_max": 0.10},
        "roundabout_medium": {"success_rate_min": 0.85, "collision_rate_max": 0.15, "off_route_rate_max": 0.0, "timeout_rate_max": 0.10},
        "carla": {"success_rate_min": 0.95, "collision_rate_max": 0.05, "off_route_rate_max": 0.0, "timeout_rate_max": 0.05},
    }[scenario]


def compute_development_gate(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "development", hashes)
    rows = summary["per_run"]
    references = contract["development"]["matched_parent_v4_12_reference"]
    guards = contract["development"]["exact_parent_guards"]
    per_run = {}
    for row in rows:
        gate = _candidate_integrity(row)
        gate["checks"].update(_metric_gate(row, contract["development"]["per_run_gates"][row["job_id"]]))
        reference = references[row["job_id"]]
        success_delta = float(row["success_rate"]) - float(reference["success_rate"])
        collision_delta = float(row["collision_rate"]) - float(reference["collision_rate"])
        gate["parent_deltas"] = {"success_rate": success_delta, "collision_rate": collision_delta}
        if row["job_id"] == "D3":
            gate["checks"]["D3_parent_improvement"] = (
                success_delta + NUMERIC_EPSILON >= float(guards["D3_success_delta_min"])
                or collision_delta <= float(guards["D3_collision_delta_max"]) + NUMERIC_EPSILON
            )
        else:
            gate["checks"]["retained_parent_success_guard"] = success_delta + NUMERIC_EPSILON >= float(guards["per_cell_success_delta_min"])
            gate["checks"]["retained_parent_collision_guard"] = collision_delta <= float(guards["per_cell_collision_delta_max"]) + NUMERIC_EPSILON
        gate["passed"] = all(gate["checks"].values())
        per_run[row["job"]] = gate
    cross = [row for row in rows if row["scenario"] == "cross"]
    cross_success = sum(float(row["success_rate"]) for row in cross) / len(cross) if cross else float("nan")
    cross_collision = sum(float(row["collision_rate"]) for row in cross) / len(cross) if cross else float("nan")
    checks = {
        "complete_matrix": summary["complete"] is True,
        "all_per_run_gates_and_parent_guards": len(per_run) == 4 and all(value["passed"] for value in per_run.values()),
        "D3_improves_parent": any(row["job_id"] == "D3" and per_run[row["job"]]["checks"].get("D3_parent_improvement") is True for row in rows),
        "cross_aggregate_success": len(cross) == 2 and cross_success + NUMERIC_EPSILON >= float(guards["cross_aggregate_success_rate_min"]),
        "cross_aggregate_collision": len(cross) == 2 and cross_collision <= float(guards["cross_aggregate_collision_rate_max"]) + NUMERIC_EPSILON,
        "formal_test_untouched": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "topo-scene-v4.13.development-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True, "fabricated_values": False,
        "decision": "pass" if passed else "fail", "ablation_unlocked": passed,
        "promotion_unlocked": False, "formal_test_accessed": False, **hashes,
        "checks": checks, "per_run_gates": per_run,
        "cross_aggregate": {"success_rate": cross_success if cross else None, "collision_rate": cross_collision if cross else None},
        "development_summary_sha256": _sha256(stage_root("development") / "summary.json"),
        "next_stage": "required_ablation" if passed else "complete_attribution_and_new_version",
    }
    _write_json(DEFAULT_DEVELOPMENT_DECISION, value)
    return value


def compute_ablation_decision(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    validate_development_decision()
    summary = summarize_stage(contract, digest, "ablation", hashes)
    integrity = {row["job"]: _candidate_integrity(row) for row in summary["per_run"]}
    complete = summary["complete"] is True and len(integrity) == 3 and all(value["passed"] for value in integrity.values())
    development = _load_json(stage_root("development") / "summary.json")
    primary = {row["job_id"]: row for row in development["per_run"]}
    comparisons = {}
    for row in summary["per_run"]:
        reference = primary["D1" if row["job_id"] == "A3" else "D3"]
        comparisons[row["job_id"]] = {
            "reference_primary_job_id": reference["job_id"],
            "success_rate_delta": float(row["success_rate"]) - float(reference["success_rate"]),
            "collision_rate_delta": float(row["collision_rate"]) - float(reference["collision_rate"]),
        }
    value = {
        "schema_version": "topo-scene-v4.13.ablation-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True, "fabricated_values": False,
        "decision": "complete" if complete else "incomplete",
        "promotion_unlocked": complete, "formal_test_accessed": False, **hashes,
        "accepted_runs": summary["accepted_runs"], "expected_runs": summary["expected_runs"],
        "integrity": integrity, "mechanism_comparisons": comparisons,
        "ablation_summary_sha256": _sha256(stage_root("ablation") / "summary.json"),
        "next_stage": "fresh_promotion" if complete else "repair_ablation_evidence",
    }
    _write_json(DEFAULT_ABLATION_DECISION, value)
    return value


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["method"], row["scenario"], int(row["seed"])): row for row in rows}
    output = []
    for scenario, seed in sorted({(row["scenario"], int(row["seed"])) for row in rows}):
        control = indexed[("temporal_graph", scenario, seed)]
        candidate = indexed[("gits_prcr_full", scenario, seed)]
        output.append({"scenario": scenario, "seed": seed, **{f"{metric}_delta": float(candidate[metric]) - float(control[metric]) for metric in ("success_rate", "collision_rate", "off_route_rate", "timeout_rate", "mean_return")}})
    return output


def compute_promotion_gate(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    validate_development_decision()
    validate_ablation_decision()
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {(row["method"], row["scenario"]): row for row in summary["aggregate"]}
    scenarios = contract["promotion"]["scenarios"]
    success = {scenario: aggregate[("gits_prcr_full", scenario)]["success_rate"] - aggregate[("temporal_graph", scenario)]["success_rate"] for scenario in scenarios}
    collision = {scenario: aggregate[("gits_prcr_full", scenario)]["collision_rate"] - aggregate[("temporal_graph", scenario)]["collision_rate"] for scenario in scenarios}
    paired = _paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {}
    for row in rows:
        if row["method"] != "gits_prcr_full":
            continue
        gate = _candidate_integrity(row)
        gate["checks"].update(_metric_gate(row, _promotion_guard(row["scenario"])))
        gate["passed"] = all(gate["checks"].values())
        candidate_gates[f"{row['scenario']}__seed{row['seed']}"] = gate
    checks = {
        "candidate_development_guards_each_seed": all(row["passed"] for row in candidate_gates.values()),
        "all_candidate_off_route_zero": all(aggregate[("gits_prcr_full", scenario)]["off_route_rate"] <= NUMERIC_EPSILON for scenario in scenarios),
        "per_scenario_success_noninferiority": all(delta + NUMERIC_EPSILON >= -float(rules["per_scenario_success_noninferiority_margin"]) for delta in success.values()),
        "per_scenario_collision_noninferiority": all(delta <= float(rules["per_scenario_collision_noninferiority_margin"]) + NUMERIC_EPSILON for delta in collision.values()),
        "worst_paired_seed_success_noninferiority": min(row["success_rate_delta"] for row in paired) + NUMERIC_EPSILON >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success.values()) + NUMERIC_EPSILON >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": max(success.values()) + NUMERIC_EPSILON >= float(rules["positive_effect_success_gain"]) or min(collision.values()) <= -float(rules["positive_effect_collision_reduction"]) + NUMERIC_EPSILON,
        "learned_target_only_selection": all(row["selected_checkpoint_kind"] in ("highest_training_success", "exact_final") and row["selected_deployment_decoder"] == TARGET_DECODER for row in rows if row["method"] == "gits_prcr_full"),
    }
    passed = all(checks.values())
    value = {
        "schema_version": "topo-scene-v4.13.promotion-gate/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True, "fabricated_values": False,
        "decision": "pass" if passed else "fail", **hashes,
        "promotion_results": str((stage_root("promotion") / "summary.json").resolve()),
        "promotion_results_sha256": _sha256(stage_root("promotion") / "summary.json"),
        "formal_algorithms": list(V413_FORMAL_ALGORITHMS), "checks": checks,
        "candidate_per_seed_gates": candidate_gates,
        "per_scenario_success_delta": success, "per_scenario_collision_delta": collision,
        "worst_paired_seed_success_delta": min(row["success_rate_delta"] for row in paired),
        "macro_success_delta": _mean(success.values()), "paired_seed_deltas": paired,
        "formal_test_unlocked": passed, "formal_test_accessed": False,
        "next_stage": "untouched_formal_test" if passed else "complete_attribution_and_new_version",
    }
    _write_json(DEFAULT_PROMOTION_GATE, value)
    return value


def compute_formal_decision(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    paired = _paired_deltas(summary["per_run"])
    by_scenario = {}
    for scenario in contract["formal_test"]["scenarios"]:
        cells = [row for row in paired if row["scenario"] == scenario]
        by_scenario[scenario] = {metric: _mean(row[metric] for row in cells) for metric in ("success_rate_delta", "collision_rate_delta", "off_route_rate_delta", "timeout_rate_delta", "mean_return_delta")}
    acceptance = contract["formal_test"]["final_acceptance"]
    checks = {
        "every_scenario_success_noninferior": all(row["success_rate_delta"] + NUMERIC_EPSILON >= float(acceptance["every_scenario_success_delta_min"]) for row in by_scenario.values()),
        "every_scenario_collision_noninferior": all(row["collision_rate_delta"] <= float(acceptance["every_scenario_collision_delta_max"]) + NUMERIC_EPSILON for row in by_scenario.values()),
        "positive_effect": max(row["success_rate_delta"] for row in by_scenario.values()) + NUMERIC_EPSILON >= float(acceptance["positive_effect_any_success_delta_min"]) or min(row["collision_rate_delta"] for row in by_scenario.values()) <= float(acceptance["positive_effect_any_collision_delta_max"]) + NUMERIC_EPSILON,
    }
    value = {
        "schema_version": "topo-scene-v4.13.formal-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True, "fabricated_values": False, **hashes,
        "formal_test_accessed": True, "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks, "per_scenario_paired_mean_deltas": by_scenario,
        "paired_seed_deltas": paired,
        "summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "statistical_analysis_status": "pending_preregistered_hierarchical_bootstrap_and_holm",
    }
    _write_json(stage_root("formal") / "formal_decision.json", value)
    return value


def _assert_stage_can_run(hashes: dict[str, str], stage: str) -> None:
    validate_implementation_freeze()
    validate_engineering_receipt()
    if stage in {"ablation", "promotion", "formal"}:
        validate_development_decision()
    if stage in {"promotion", "formal"}:
        validate_ablation_decision()
    if stage == "formal":
        value = _load_json(DEFAULT_PROMOTION_GATE)
        _require(value.get("decision") == "pass", "formal test locked: promotion failed")
        for key in ("experiment_contract_sha256", "implementation_freeze_sha256"):
            _require(value.get(key) == hashes[key], f"formal lock {key} mismatch")


def write_plan(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, device: str) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    value = {
        "schema_version": "topo-scene-v4.13.stage-plan/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "device": device, "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        "immutable_recovery_attempts": MAX_IMMUTABLE_ATTEMPTS, **hashes,
        "jobs": [{**asdict(job), "name": job.name, "canonical_run_directory": str(run_directory(job).resolve()), "accepted_evidence_directory": str(accepted_run_location(job, hashes).resolve()) if accepted_run_location(job, hashes) else None, "command": command_for(job, hashes, device=device)} for job in jobs],
    }
    _write_json(stage_root(stage) / "stage_plan.json", value)
    return value


def _select_jobs(jobs: list[Job], selector: str | None) -> list[Job]:
    if selector in (None, "all"):
        return jobs
    requested = {item.strip() for item in str(selector).split(",") if item.strip()}
    selected = [job for job in jobs if job.name in requested or job.job_id in requested]
    _require(len(selected) == len(requested), f"unknown or ambiguous job selector: {selector}")
    return selected


def _next_attempt(job: Job) -> int | None:
    for attempt, directory in enumerate(attempt_directories(job)):
        if not directory.exists():
            return attempt
    return None


def execute_jobs(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, selector: str | None, device: str, workers: int, dry_run: bool) -> int:
    _assert_stage_can_run(hashes, stage)
    _require(workers == 1, "v4.13 freezes one sequential worker")
    jobs = jobs_for_stage(contract, digest, stage)
    selected = _select_jobs(jobs, selector)
    plan = write_plan(contract, digest, stage, hashes, device=device)
    if dry_run:
        print(json.dumps({"selected": [job.name for job in selected], "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    log_root = stage_root(stage) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for job in selected:
        existing = accepted_run_location(job, hashes)
        if existing is not None:
            attempts.append({"job": job.name, "status": "already_accepted", "evidence_directory": str(existing.resolve())})
            continue
        attempt = _next_attempt(job)
        if attempt is None:
            accepted, reason = accepted_run_reason(job, hashes)
            attempts.append({"job": job.name, "status": "immutable_attempts_exhausted", "acceptance_reason": reason})
            continue
        directory = run_directory(job, attempt)
        log_path = log_root / f"{directory.name}.log"
        print(json.dumps({"job": job.name, "status": "running", "attempt": attempt, "directory": str(directory), "log": str(log_path)}, ensure_ascii=False), flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command_for(job, hashes, device=device, attempt=attempt), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
            try:
                _validate_run(directory, job, hashes)
                accepted, reason = True, "accepted"
            except Exception as exc:
                accepted, reason = False, f"{type(exc).__name__}: {exc}"
            status = "accepted" if completed.returncode == 0 and accepted else "failed"
            row = {"job": job.name, "status": status, "attempt": attempt, "returncode": completed.returncode, "acceptance_reason": reason, "directory": str(directory.resolve()), "log": str(log_path.resolve())}
        except Exception as exc:
            row = {"job": job.name, "status": "launcher_exception", "attempt": attempt, "returncode": None, "acceptance_reason": f"{type(exc).__name__}: {exc}", "directory": str(directory.resolve()), "log": str(log_path.resolve())}
        attempts.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    execution = {
        "schema_version": "topo-scene-v4.13.execution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True, "incomplete_attempts_preserved": True,
        **hashes, "selected": [job.name for job in selected], "attempts": attempts,
        "accepted_after_batch": sum(accepted_run(job, hashes) for job in jobs),
        "expected": len(jobs), "formal_stage_launched": stage == "formal",
    }
    _write_json(stage_root(stage) / "last_execution.json", execution)
    summarize_stage(contract, digest, stage, hashes)
    failed = [row for row in attempts if row["status"] not in {"accepted", "already_accepted"}]
    return 1 if failed else 0


def status_payload(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    stages = {}
    for stage in STAGES:
        jobs = jobs_for_stage(contract, digest, stage)
        accepted = sum(accepted_run(job, hashes) for job in jobs)
        stages[stage] = {"accepted": accepted, "expected": len(jobs), "complete": accepted == len(jobs)}
    development = _load_json(DEFAULT_DEVELOPMENT_DECISION) if DEFAULT_DEVELOPMENT_DECISION.is_file() else None
    ablation = _load_json(DEFAULT_ABLATION_DECISION) if DEFAULT_ABLATION_DECISION.is_file() else None
    promotion = _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    return {
        "schema_version": "topo-scene-v4.13.status/v1", **hashes,
        "stages": stages,
        "development_decision": development.get("decision") if development else None,
        "ablation_decision": ablation.get("decision") if ablation else None,
        "promotion_decision": promotion.get("decision") if promotion else None,
        "formal_test_unlocked": bool(promotion and promotion.get("decision") == "pass"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    sub = value.add_subparsers(dest="command", required=True)
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
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract_path = args.contract.resolve()
    _require(_sha256(contract_path) == CONTRACT_SHA256, "contract bytes drifted")
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    hashes = protocol_hashes()
    if args.command == "validate":
        output = {"status": "valid", **hashes, "formal_test_accessed": False}
    elif args.command == "plan":
        _assert_stage_can_run(hashes, args.stage)
        output = write_plan(contract, digest, args.stage, hashes, device=args.device)
    elif args.command == "run":
        return execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
    elif args.command == "summarize":
        output = summarize_stage(contract, digest, args.stage, hashes)
    elif args.command == "gate":
        output = {"development": compute_development_gate, "ablation": compute_ablation_decision, "promotion": compute_promotion_gate, "formal": compute_formal_decision}[args.stage](contract, digest, hashes)
    else:
        output = status_payload(contract, digest, hashes)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
