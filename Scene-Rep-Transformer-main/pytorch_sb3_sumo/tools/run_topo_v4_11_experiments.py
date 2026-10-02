"""Run the hash-bound v4.11 PRCR experiment protocol.

Every selected job in a stage is attempted before a batch result is returned.
Development, ablation, promotion, and formal access are separate gates.  The
formal partition remains inaccessible until a complete promotion matrix passes.
"""

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

from configs.sb3_configs_v4_11 import (
    TEMPORAL_GRAPH,
    V411_FORMAL_ALGORITHMS,
    V411_FULL,
    V411_IMPLEMENTATION_IDS,
    V411_RANK_NO_CONSISTENCY,
    V411_SOFT_BCE_ONLY,
)
from tools.checkpoint_decoder_selector_v4_6 import PARENT_DECODER, TARGET_DECODER
from tools.checkpoint_decoder_selector_v4_9_2 import SELECTOR_MODE


PYTHON_EXECUTABLE = Path("D:/Programs/Anaconda/envs/llm_pipeline/python.exe")
DEFAULT_CONTRACT = (
    ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_11.yaml"
)
DEFAULT_PREREGISTRATION = ROOT / "results_topo_v4_11_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = ROOT / "results_topo_v4_11_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_DEVELOPMENT_DECISION = (
    ROOT / "results_topo_v4_11_dev" / "development" / "development_decision.json"
)
DEFAULT_ABLATION_DECISION = (
    ROOT / "results_topo_v4_11_dev" / "ablation" / "ablation_decision.json"
)
DEFAULT_PROMOTION_GATE = ROOT / "results_topo_v4_11_promotion" / "promotion_gate.json"
DEFAULT_STAGE0_RESULTS = (
    ROOT
    / "results_topo_v4_10_dev"
    / "development"
    / "development_decision.json"
)
DEFAULT_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_10_dev"
    / "development"
    / "attribution"
    / "d1_d2_cross_instability"
    / "attribution.json"
)

CONTRACT_SHA256 = "c71e98e709874a5b2a7c2a9f4a558c2dea7e7c1054688757615c8ffff886f880"
STAGE0_RESULTS_SHA256 = "14c77f6d286fac0e06d075342223291d032abec48a03c02417befbff3a00caf1"
ATTRIBUTION_SHA256 = "0fddccc26580224a4f1d300905d3399a03ba87d66dc3d4ac7ba5fa07e7d10243"

STAGES = ("development", "ablation", "promotion", "formal")
METHOD_ALGORITHMS = {
    "temporal_graph": TEMPORAL_GRAPH,
    "prcr_full": V411_FULL,
    "soft_bce_only": V411_SOFT_BCE_ONLY,
    "ranked_no_consistency": V411_RANK_NO_CONSISTENCY,
}
METHOD_CODES = {
    "temporal_graph": "tg",
    "prcr_full": "prcr",
    "soft_bce_only": "bce",
    "ranked_no_consistency": "ranknc",
}
SCENARIO_CODES = {
    "left_turn": "lt",
    "cross": "cross",
    "roundabout_easy": "rae",
    "roundabout_medium": "ram",
    "roundabout": "ra",
    "carla": "carla",
}
FULL_RUN_REQUIRED_FILES = (
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


class V411ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V411ProtocolError(message)


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.11 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.11.proper-calibrated-ranked-risk-contract/v1",
        "unexpected v4.11 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication drifted")
    _require(contract.get("formal_test_locked") is True, "formal lock drifted")
    lineage = contract.get("lineage", {})
    expected_lineage = {
        "parent_contract_sha256": "3cdb4f3b23207812ec66d78e535e7db4d0fb0ed14d01c040b961783f054225a3",
        "completed_development_summary_sha256": "f4ad92c75459695ac3c673cd1899474d891690e67a3ee01a9b7e614b741ef295",
        "failed_development_decision_sha256": STAGE0_RESULTS_SHA256,
        "completed_pipeline_report_sha256": "8d0cd5e47af1875dcce1fcce2cfb078cc126645e3fcd6d54cdfcf584e46c25f4",
        "development_attribution_sha256": ATTRIBUTION_SHA256,
        "formal_test_accessed": False,
    }
    for key, expected in expected_lineage.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")

    model = contract.get("model_change", {})
    expected_model = {
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_risk_coefficient": 1.0,
        "deterministic_model_score_changed_from_v4_10": False,
        "actor_objective_changed_from_v4_10": False,
        "proposal_model_changed_from_v4_10": False,
    }
    for key, expected in expected_model.items():
        _require(model.get(key) == expected, f"model_change.{key} drifted")
    ranking = model.get("collision_pairwise_ranking", {})
    expected_ranking = {
        "coefficient": 0.25,
        "pair_weight": "absolute_soft_td_target_difference",
        "normalization": "sum_of_pair_weights",
        "hard_target_threshold": False,
        "fixed_ranking_margin": False,
    }
    for key, expected in expected_ranking.items():
        _require(ranking.get(key) == expected, f"model_change.ranking.{key} drifted")
    consistency = model.get("collision_temporal_consistency", {})
    expected_consistency = {
        "coefficient": 0.05,
        "teacher": "exponential_moving_average_target_collision_critic",
        "student": "online_collision_critic",
        "same_observation_action_pairs": True,
        "loss": "probability_mean_squared_error",
        "teacher_gradient": False,
    }
    for key, expected in expected_consistency.items():
        _require(
            consistency.get(key) == expected,
            f"model_change.consistency.{key} drifted",
        )

    forbidden = contract.get("forbidden", {})
    for key in (
        "kinematic_safety_projection",
        "fixed_speed_search_at_inference",
        "ttc_or_headway_threshold",
        "geometry_unsafe_label",
        "lane_change_veto",
        "actor_confidence_gate",
        "safety_shield_or_rule_fallback",
        "semantic_tie_override",
        "post_decoder_action_rewrite",
        "validation_or_formal_outcome_as_model_input",
    ):
        _require(forbidden.get(key) is True, f"forbidden declaration missing: {key}")

    development = contract.get("development", {})
    _require(
        development.get("execution_policy") == "attempt_all_then_summarize",
        "development execution policy drifted",
    )
    _require(
        development.get("failure_does_not_cancel_remaining_jobs") is True,
        "development fail-fast enabled",
    )
    _require(len(development.get("ordered_cells", [])) == 4, "development matrix drifted")
    _require(
        len(development.get("required_ablation_cells_after_primary_pass", [])) == 2,
        "ablation matrix drifted",
    )
    _require(
        [row.get("method") for row in development.get("ordered_cells", [])]
        == ["prcr_full"] * 4,
        "development method matrix drifted",
    )
    promotion = contract.get("promotion", {})
    _require(
        promotion.get("execution_policy") == "attempt_all_then_summarize",
        "promotion execution policy drifted",
    )
    _require(
        promotion.get("failure_does_not_cancel_remaining_jobs") is True,
        "promotion fail-fast enabled",
    )
    _require(promotion.get("methods") == ["temporal_graph", "prcr_full"], "promotion methods drifted")
    _require(promotion.get("scenarios") == ["cross", "roundabout_medium", "carla"], "promotion scenarios drifted")
    _require(promotion.get("seeds") == [20, 21], "promotion seeds drifted")
    _require(int(promotion.get("jobs", 0)) == 12, "promotion job count drifted")
    formal = contract.get("formal_test", {})
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal state drifted")
    _require(
        len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"])
        == 120,
        "formal matrix drifted",
    )
    return contract


def validate_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    value = _load_json(path)
    _require(
        value.get("schema_version") == "topo-scene-v4.11.preregistration-receipt/v1",
        "preregistration schema drifted",
    )
    _require(value.get("experiment_contract_sha256") == CONTRACT_SHA256, "preregistration contract drifted")
    _require(value.get("contract_preceded_scientific_training") is True, "training was not preregistered")
    _require(value.get("formal_test_accessed") is False, "preregistration accessed formal")
    _require(value.get("execution_policy") == "attempt_all_then_summarize", "execution policy drifted")
    return value


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    value = _load_json(path)
    _require(
        value.get("schema_version") == "topo-scene-v4.11.implementation-freeze/v1",
        "freeze schema drifted",
    )
    _require(value.get("experiment_contract_sha256") == CONTRACT_SHA256, "freeze contract drifted")
    _require(value.get("formal_test_accessed") is False, "freeze accessed formal")
    files = value.get("scientific_files", {})
    _require(isinstance(files, dict) and files, "freeze file manifest missing")
    for relative, expected in files.items():
        path = ROOT / relative
        _require(path.is_file(), f"frozen file missing: {relative}")
        _require(_sha256(path) == expected, f"frozen file drifted: {relative}")
    return value


def validate_engineering_receipt(
    path: Path = DEFAULT_ENGINEERING_RECEIPT,
) -> dict[str, Any]:
    value = _load_json(path)
    _require(value.get("status") == "passed", "engineering gate did not pass")
    _require(value.get("formal_test_accessed") is False, "engineering accessed formal")
    _require(
        value.get("implementation_freeze_sha256") == _sha256(DEFAULT_FREEZE),
        "engineering freeze binding drifted",
    )
    return value


def validate_development_decision() -> dict[str, Any]:
    value = _load_json(DEFAULT_DEVELOPMENT_DECISION)
    _require(value.get("decision") == "pass", "development gate did not pass")
    _require(value.get("ablation_unlocked") is True, "development did not unlock ablation")
    _require(value.get("formal_test_accessed") is False, "development accessed formal")
    _require(value.get("experiment_contract_sha256") == CONTRACT_SHA256, "development contract drifted")
    _require(value.get("implementation_freeze_sha256") == _sha256(DEFAULT_FREEZE), "development freeze drifted")
    return value


def validate_ablation_decision() -> dict[str, Any]:
    value = _load_json(DEFAULT_ABLATION_DECISION)
    _require(value.get("decision") == "complete", "required ablations are incomplete")
    _require(value.get("promotion_unlocked") is True, "ablation did not unlock promotion")
    _require(value.get("accepted_runs") == 2, "ablation matrix drifted")
    _require(value.get("formal_test_accessed") is False, "ablation accessed formal")
    return value


def protocol_hashes() -> dict[str, str]:
    _require(_sha256(DEFAULT_CONTRACT) == CONTRACT_SHA256, "contract bytes drifted")
    _require(_sha256(DEFAULT_STAGE0_RESULTS) == STAGE0_RESULTS_SHA256, "stage-0 evidence drifted")
    _require(_sha256(DEFAULT_ATTRIBUTION) == ATTRIBUTION_SHA256, "attribution evidence drifted")
    validate_contract(load_contract())
    validate_preregistration()
    validate_implementation_freeze()
    validate_engineering_receipt()
    return {
        "experiment_contract_sha256": CONTRACT_SHA256,
        "stage0_results_sha256": STAGE0_RESULTS_SHA256,
        "attribution_sha256": ATTRIBUTION_SHA256,
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
        return (
            f"{prefix}__{METHOD_CODES[self.method]}__{SCENARIO_CODES[self.scenario]}"
            f"__s{self.seed}__p{self.protocol_tag}"
        )


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.11 stage {stage!r}")
    tag = digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        for row in settings["ordered_cells"]:
            method = str(row["method"])
            algorithm = METHOD_ALGORITHMS[method]
            jobs.append(
                Job(
                    stage=stage,
                    job_id=str(row["id"]),
                    kind="fresh_primary_development",
                    method=method,
                    algorithm=algorithm,
                    implementation_id=V411_IMPLEMENTATION_IDS[algorithm],
                    scenario=str(row["scenario"]),
                    seed=int(row["seed"]),
                    raw_steps=int(row["raw_steps"]),
                    calibration_episodes=12,
                    calibration_seed_start=int(row["calibration_seed_start"]),
                    evaluation_episodes=int(settings["evaluation_episodes"]),
                    evaluation_split=str(settings["evaluation_split"]),
                    evaluation_seed_start=int(row["evaluation_seed_start"]),
                    role=str(
                        row.get(
                            "role",
                            {
                                "D1": "primary_safety",
                                "D2": "seed_replication",
                                "D3": "regression_guard",
                                "D4": "retained_gain_guard",
                            }[str(row["id"])],
                        )
                    ),
                    protocol_tag=tag,
                )
            )
    elif stage == "ablation":
        settings = contract["development"]
        for row in settings["required_ablation_cells_after_primary_pass"]:
            method = str(row["method"])
            algorithm = METHOD_ALGORITHMS[method]
            jobs.append(
                Job(
                    stage=stage,
                    job_id=str(row["id"]),
                    kind="required_mechanism_ablation",
                    method=method,
                    algorithm=algorithm,
                    implementation_id=V411_IMPLEMENTATION_IDS[algorithm],
                    scenario=str(row["scenario"]),
                    seed=int(row["seed"]),
                    raw_steps=int(row["raw_steps"]),
                    calibration_episodes=12,
                    calibration_seed_start=131_000,
                    evaluation_episodes=int(row["evaluation_episodes"]),
                    evaluation_split=str(settings["evaluation_split"]),
                    evaluation_seed_start=141_000,
                    role="mechanism_ablation_not_used_for_selection",
                    protocol_tag=tag,
                )
            )
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        for method in settings["methods"]:
            algorithm = METHOD_ALGORITHMS[str(method)]
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
                            kind="fresh_paired_model_training",
                            method=str(method),
                            algorithm=algorithm,
                            implementation_id=V411_IMPLEMENTATION_IDS[algorithm],
                            scenario=str(scenario),
                            seed=seed,
                            raw_steps=int(settings["raw_training_steps"]),
                            calibration_episodes=12,
                            calibration_seed_start=calibration_seed,
                            evaluation_episodes=int(settings["evaluation_episodes"]),
                            evaluation_split=str(settings["evaluation_split"]),
                            evaluation_seed_start=evaluation_seed,
                            role=(
                                "fresh_paired_promotion"
                                if stage == "promotion"
                                else "untouched_formal_test"
                            ),
                            protocol_tag=tag,
                        )
                    )
    expected = {"development": 4, "ablation": 2, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} job")
    return jobs


def stage_root(stage: str) -> Path:
    return {
        "development": ROOT / "results_topo_v4_11_dev" / "development",
        "ablation": ROOT / "results_topo_v4_11_dev" / "ablation",
        "promotion": ROOT / "results_topo_v4_11_promotion",
        "formal": ROOT / "results_topo_v4_11_formal",
    }[stage]


def run_directory(job: Job) -> Path:
    return stage_root(job.stage) / "runs" / job.name


def command_for(job: Job, hashes: dict[str, str], *, device: str) -> list[str]:
    command = [
        str(PYTHON_EXECUTABLE),
        str(ROOT / "tools" / "train_paper_sb3_sumo_v4_11.py"),
        "--scenario",
        job.scenario,
        "--algo",
        job.algorithm,
        "--max-steps",
        str(job.raw_steps),
        "--learning-starts",
        "5000",
        "--checkpoint-freq",
        str(job.raw_steps),
        "--eval-freq",
        "0",
        "--eval-episodes",
        str(job.evaluation_episodes),
        "--evaluation-split",
        job.evaluation_split,
        "--evaluation-seed-start",
        str(job.evaluation_seed_start),
        "--calibration-episodes",
        str(job.calibration_episodes),
        "--calibration-seed-start",
        str(job.calibration_seed_start),
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--batch-size",
        "32",
        "--learning-rate",
        "0.0001",
        "--discount",
        "0.99",
        "--buffer-size",
        "20000",
        "--action-repeat",
        "3",
        "--ego-control-profile",
        "direct",
        "--traffic-protocol",
        "frozen_60_20_20",
        "--episode-limit-profile",
        "source",
        "--output-dir",
        str((stage_root(job.stage) / "runs").resolve()),
        "--model-name",
        job.name,
        "--experiment-contract-sha256",
        hashes["experiment_contract_sha256"],
        "--stage0-results-sha256",
        hashes["stage0_results_sha256"],
        "--attribution-sha256",
        hashes["attribution_sha256"],
        "--implementation-freeze-sha256",
        hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(
            ["--formal-unlock-receipt", str(DEFAULT_PROMOTION_GATE.resolve())]
        )
    return command


def _calibration_directory(
    root: Path,
    row: dict[str, Any],
    *,
    secondary: bool,
    fallback_used: bool,
) -> Path:
    kind = str(row["checkpoint_kind"])
    if fallback_used and kind == "highest_training_success":
        kind = "exact_final"
    short = {"highest_training_success": "best", "exact_final": "final"}[kind]
    block = "cal_secondary" if secondary else "cal"
    return root / "selector" / block / short / str(row["deployment_decoder"])


def _validate_selector(root: Path, *, candidate: bool) -> dict[str, Any]:
    value = _load_json(root / "selector" / "receipt.json")
    expected_decoder = TARGET_DECODER if candidate else PARENT_DECODER
    _require(value.get("selection_partition") == "train", "selector did not use train")
    _require(value.get("validation_used_for_selection") is False, "validation entered selection")
    _require(value.get("formal_test_used_for_selection") is False, "formal entered selection")
    _require(int(value.get("candidate_count", 0)) == 2, "selector candidate count drifted")
    rows = value.get("candidates", [])
    _require(len(rows) == 2, "selector candidate matrix incomplete")
    _require(
        {row.get("checkpoint_kind") for row in rows}
        == {"highest_training_success", "exact_final"},
        "checkpoint candidates drifted",
    )
    _require(
        all(row.get("deployment_decoder") == expected_decoder for row in rows),
        "unexpected deployment decoder entered selection",
    )
    fallback = value.get("training_best_missing_fallback_used") is True
    for secondary, candidates in (
        (False, rows),
        (True, value.get("secondary_candidates", [])),
    ):
        for row in candidates:
            directory = _calibration_directory(
                root, row, secondary=secondary, fallback_used=fallback
            )
            _require(directory.is_dir(), f"calibration directory missing: {directory}")
            _require(
                _sha256(directory / "detailed.json")
                == row.get("calibration_result_sha256"),
                "calibration detailed hash drifted",
            )
            _require(
                _sha256(directory / "actions.json")
                == row.get("calibration_action_diagnostics_sha256"),
                "calibration action hash drifted",
            )
            detailed = _load_json(directory / "detailed.json")
            _require(detailed.get("traffic_partition") == "train", "calibration partition drifted")
            _require(detailed.get("formal_test_accessed") is False, "calibration accessed formal")
            _require(detailed.get("decoder_integrity_passed") is True, "calibration integrity failed")
    if candidate:
        _require(value.get("selector_mode") == SELECTOR_MODE, "target-only selector mode drifted")
        _require(value.get("selected_deployment_decoder") == TARGET_DECODER, "candidate selected non-target decoder")
        _require(value.get("deployment_decoder_candidates") == [TARGET_DECODER], "candidate decoder list drifted")
        _require(value.get("calibration_deterministic_lane_decoder") == TARGET_DECODER, "calibration decoder metadata drifted")
        _require(value.get("schema_version") == "topo-scene-v4.11.target-only-tie-replicated-selector/v1", "target-only receipt schema drifted")
        for key in (
            "fusion_candidate_present",
            "actor_confidence_threshold_present",
            "external_kinematic_projection",
            "action_postprocessing_override",
        ):
            _require(value.get(key) is False, f"selector enabled forbidden {key}")
        _require(
            not any(path.name == "fusion_0_90" for path in (root / "selector").rglob("*")),
            "fusion calibration directory exists",
        )
    else:
        _require(value.get("selected_deployment_decoder") == PARENT_DECODER, "control decoder drifted")
    selected_source = Path(value["selected_source_checkpoint_path"])
    selected_model = Path(value["selected_model_path"])
    _require(selected_source.is_file(), "selected source checkpoint missing")
    _require(selected_model.is_file(), "selected model missing")
    _require(_sha256(selected_source) == value.get("selected_source_checkpoint_sha256"), "source checkpoint hash drifted")
    _require(_sha256(selected_model) == value.get("selected_model_sha256"), "selected model hash drifted")
    _require(value.get("source_policy_parameter_state_sha256") == value.get("selected_model_parameter_state_sha256"), "selected policy tensors changed")
    _require(value.get("policy_parameter_state_preserved") is True, "policy preservation flag failed")
    _require(value.get("selector_receipt_precedes_validation_environment") is True, "selector was not sealed before validation")
    return value


def _validate_arguments(root: Path, job: Job, hashes: dict[str, str]) -> dict[str, Any]:
    value = _load_json(root / "arguments.json")
    _require(value.get("schema_version") == "topo-scene-v4.11.run-arguments/v1", "argument schema drifted")
    requested = value.get("requested_raw_steps", {})
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
        "ego_control_profile": "direct",
        "traffic_protocol": "frozen_60_20_20",
        "episode_limit_profile": "source",
        "eval_episodes": job.evaluation_episodes,
        "evaluation_split": job.evaluation_split,
        "evaluation_seed_start": job.evaluation_seed_start,
        "calibration_seed_start": job.calibration_seed_start,
        "calibration_episodes": job.calibration_episodes,
        "experiment_contract_sha256": hashes["experiment_contract_sha256"],
        "stage0_results_sha256": hashes["stage0_results_sha256"],
        "attribution_sha256": hashes["attribution_sha256"],
        "implementation_freeze_sha256": hashes["implementation_freeze_sha256"],
    }
    for key, expected_value in expected.items():
        _require(requested.get(key) == expected_value, f"argument {key} drifted")
    fidelity = value.get("implementation_fidelity", {})
    _require(fidelity.get("implementation_id") == job.implementation_id, "implementation id drifted")
    _require("actor_non_keep_confidence_threshold" not in fidelity, "inherited confidence rule metadata remains")
    _require(fidelity.get("inference_safety_rule_added") is False, "safety rule enabled")
    _require(fidelity.get("external_kinematic_projection") is False, "kinematic projection enabled")
    _require(fidelity.get("kinematic_safety_projection") is False, "safety projection enabled")
    _require(fidelity.get("traffic_risk_in_lane_mask") is False, "traffic risk entered lane mask")
    _require(fidelity.get("action_postprocessing_override") is False, "action rewrite enabled")
    if job.algorithm != TEMPORAL_GRAPH:
        _require(fidelity.get("v4_11_isolated_files") is True, "v4.11 wrapper not recorded")
        _require(fidelity.get("shared_risk_encoder") is True, "shared risk encoder missing")
        _require(fidelity.get("learned_speed_mixture") is True, "learned proposals missing")
        _require(fidelity.get("fixed_speed_grid_at_inference") is False, "fixed speed grid enabled")
        _require(fidelity.get("optimizer_parameter_sets_disjoint") is True, "optimizer ownership metadata drifted")
        _require(fidelity.get("deployment_decoder_candidates") == [TARGET_DECODER], "candidate decoder metadata drifted")
        _require(
            fidelity.get("collision_return_loss")
            == "soft_target_binary_cross_entropy_with_logits",
            "proper collision loss metadata drifted",
        )
        _require(fidelity.get("collision_pairwise_hard_threshold") is False, "ranking threshold enabled")
        _require(fidelity.get("collision_pairwise_fixed_margin") is False, "fixed ranking margin enabled")
    return value


def _finite_last(training: dict[str, Any], name: str) -> None:
    row = training.get("statistics", {}).get(name, {})
    value = row.get("last")
    _require(int(row.get("count", 0)) > 0, f"training statistic {name} has no samples")
    _require(value is not None and math.isfinite(float(value)), f"training statistic {name} is non-finite")


def _variant_expected(
    algorithm: str,
) -> tuple[int, float, float, float, float, float]:
    return {
        V411_FULL: (3, 0.05, 0.25, 0.05, 0.25, 0.05),
        V411_SOFT_BCE_ONLY: (3, 0.05, 0.25, 0.05, 0.0, 0.0),
        V411_RANK_NO_CONSISTENCY: (3, 0.05, 0.25, 0.05, 0.25, 0.0),
    }[algorithm]


def _validate_candidate(root: Path, job: Job) -> None:
    metadata = _load_json(root / "method_metadata.json")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    actions = _load_json(root / "action_diagnostics.json")
    training = _load_json(root / "training_diagnostics.json")
    labels = _load_json(root / "collision_label_diagnostics.json")
    selector = _validate_selector(root, candidate=True)
    components, support, uncertainty, prior, rank, consistency = (
        _variant_expected(job.algorithm)
    )

    expected_metadata = {
        "implementation_id": job.implementation_id,
        "shared_online_scene_encoder": True,
        "collision_supervision_shapes_actor_representation": True,
        "actor_encoder_detach": True,
        "speed_components_per_lane": components,
        "fixed_speed_grid_at_inference": False,
        "replay_support_coef": support,
        "component_entropy_scale": 0.25,
        "twin_uncertainty_coef": uncertainty,
        "component_prior_coef": prior,
        "optimizer_parameter_sets_must_be_disjoint": True,
        "encoder_optimizer_owner_count": 1,
        "collision_label_source": "observed_environment_info_collision",
        "collision_label_future_or_oracle_used": False,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": rank,
        "collision_pairwise_target_weight": "absolute_soft_td_target_difference",
        "collision_pairwise_hard_threshold": False,
        "collision_pairwise_fixed_margin": False,
        "collision_temporal_consistency_coef": consistency,
        "collision_temporal_consistency_teacher": "ema_target_collision_critic",
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "ttc_or_headway_threshold": False,
        "lane_change_veto": False,
        "actor_confidence_gate": False,
        "action_postprocessing_override": False,
        "deployment_decoder_candidates": [TARGET_DECODER],
        "checkpoint_decoder_candidate_pairs": 2,
        "selector_mode": SELECTOR_MODE,
        "fusion_candidate_present": False,
        "actor_confidence_threshold_present": False,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"method metadata {key} drifted")
    _require("actor_non_keep_confidence_threshold" not in metadata, "method contains confidence rule")

    expected_actions = {
        "selected_deployment_decoder": TARGET_DECODER,
        "selected_decoder_exact_action_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "exact_supported_mixture_model_argmax_rate": 1.0,
        "supported_mixture_score_equation_match_rate": 1.0,
        "selected_speed_exact_proposal_match_rate": 1.0,
        "collision_value_bounded_rate": 1.0,
        "learned_proposal_source_rate": 1.0,
        "no_action_rewrite_rate": 1.0,
        "selected_decoder_exact_model_match_rate": 1.0,
        "no_semantic_tie_override_rate": 1.0,
        "learned_speed_components_per_lane": components,
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "shared_risk_encoder_present": True,
        "collision_return_loss": "soft_target_binary_cross_entropy_with_logits",
        "collision_pairwise_rank_coef": rank,
        "collision_pairwise_hard_threshold": False,
        "collision_pairwise_fixed_margin": False,
        "collision_temporal_consistency_coef": consistency,
        "fixed_speed_grid_at_inference": False,
        "inference_safety_rule_added": False,
        "external_kinematic_projection": False,
        "kinematic_safety_projection": False,
        "traffic_risk_in_lane_mask": False,
        "action_postprocessing_override": False,
    }
    for key, expected in expected_actions.items():
        _require(actions.get(key) == expected, f"action evidence {key} drifted")
    _require(int(actions.get("selected_decoder_records", 0)) > 0, "no model decisions recorded")
    ownership = actions.get("optimizer_ownership", {})
    _require(ownership.get("overlap_count") == 0, "optimizer parameter owners overlap")
    structure = ownership.get("policy_structure", {})
    _require(structure.get("overlap_count") == 0, "policy optimizer owners overlap")
    _require(structure.get("online_encoder_shared_by_identity") is True, "online encoder is not shared")
    _require(structure.get("target_encoder_shared_by_identity") is True, "target encoder is not shared")

    for name in (
        "train/actor_loss",
        "train/critic_loss",
        "train/collision_critic_loss",
        "train/collision_soft_bce_loss",
        "train/collision_pairwise_rank_loss",
        "train/collision_temporal_consistency_loss",
        "train/representation_loss",
        "support/replay_speed_mixture_nll",
        "support/component_entropy",
        "support/component_speed_spread",
        "support/proposal_boundary_rate",
        "risk/policy_expected_collision_cost",
        "risk/reward_twin_disagreement",
        "risk/collision_twin_disagreement",
        "risk/learned_uncertainty",
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/collision_cost_replay_prediction",
        "risk/collision_pairwise_active_pair_rate",
        "risk/collision_pairwise_mean_target_difference",
        "risk/collision_probability_absolute_error",
    ):
        _finite_last(training, name)
    training_ownership = training.get("optimizer_ownership", {})
    _require(training_ownership.get("overlap_count") == 0, "training optimizer owners overlap")
    _require(training.get("learned_speed_components_per_lane") == components, "training component count drifted")
    _require(training.get("fixed_speed_grid_at_inference") is False, "training admits fixed grid")
    _require(
        training.get("collision_return_loss")
        == "soft_target_binary_cross_entropy_with_logits",
        "training collision loss metadata drifted",
    )
    _require(training.get("collision_pairwise_rank_coef") == rank, "training ranking coefficient drifted")
    _require(
        training.get("collision_temporal_consistency_coef") == consistency,
        "training consistency coefficient drifted",
    )
    if rank > 0.0:
        active = training.get("statistics", {}).get(
            "risk/collision_pairwise_active_pair_rate", {}
        ).get("maximum")
        _require(active is not None and float(active) > 0.0, "risk ranking had no active soft-target pairs")

    _require(labels.get("source") == "observed_environment_info_collision", "collision labels drifted")
    _require(labels.get("future_or_oracle_labels_used") is False, "oracle collision labels used")
    _require(labels.get("reward_return_changed") is False, "collision labels changed reward")
    _require(int(labels.get("sampled_label_count", 0)) > 0, "no collision labels sampled")
    rate = labels.get("sampled_positive_label_rate")
    _require(rate is not None and 0.0 <= float(rate) <= 1.0, "collision label rate invalid")

    _require(detailed.get("scientific_version") == "v4.11_proper_calibrated_ranked_risk", "detailed scientific version drifted")
    _require(detailed.get("shared_risk_encoder") is True, "detailed shared encoder missing")
    _require(detailed.get("learned_speed_components_per_lane") == components, "detailed component count drifted")
    _require(detailed.get("fixed_speed_grid_at_inference") is False, "detailed fixed grid enabled")
    _require(detailed.get("external_kinematic_projection") is False, "detailed kinematic projection enabled")
    _require(detailed.get("kinematic_safety_projection") is False, "detailed safety projection enabled")
    _require(detailed.get("traffic_risk_in_lane_mask") is False, "detailed traffic-risk mask enabled")
    _require(detailed.get("collision_return_loss") == "soft_target_binary_cross_entropy_with_logits", "detailed collision loss drifted")
    _require(detailed.get("collision_pairwise_rank_coef") == rank, "detailed ranking coefficient drifted")
    _require(detailed.get("collision_temporal_consistency_coef") == consistency, "detailed consistency coefficient drifted")
    _require(detailed.get("action_postprocessing_override") is False, "detailed action rewrite enabled")
    _require(detailed.get("selector_receipt_sha256") == _sha256(root / "selector" / "receipt.json"), "evaluation selector binding drifted")
    _require(selector.get("selected_model_policy_class") == "ProperCalibratedRankedRiskSACPolicyV411", "selected policy class drifted")


def _validate_control(root: Path, job: Job) -> None:
    metadata = _load_json(root / "method_metadata.json")
    actions = _load_json(root / "action_diagnostics.json")
    _require(metadata.get("implementation_id") == job.implementation_id, "control implementation id drifted")
    _require(metadata.get("network_changed") is False, "control network changed")
    _require(actions.get("learned_collision_critic_present") is False, "control gained collision critic")
    _require(actions.get("inference_safety_rule_added") is False, "control gained a safety rule")
    _require(not (root / "collision_label_diagnostics.json").exists(), "control emitted collision learner labels")
    _validate_selector(root, candidate=False)


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        _require(root.is_dir(), "run directory missing")
        for relative in FULL_RUN_REQUIRED_FILES:
            _require((root / relative).is_file(), f"required artifact missing: {relative}")
        _validate_arguments(root, job, hashes)
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        _require(detailed.get("schema_version") == "topo-scene-v4.11.detailed-evaluation/v1", "detailed schema drifted")
        _require(detailed.get("algorithm") == job.algorithm, "detailed algorithm drifted")
        _require(detailed.get("scenario") == job.scenario, "detailed scenario drifted")
        _require(detailed.get("evaluation_split") == job.evaluation_split, "evaluation split drifted")
        _require(detailed.get("formal_test_accessed") is (job.stage == "formal"), "formal access marker drifted")
        _require(int(detailed.get("trained_raw_steps", 0)) == job.raw_steps, "raw training budget drifted")
        _require(int(detailed.get("summary", {}).get("episodes", 0)) == job.evaluation_episodes, "evaluation episode count drifted")
        provenance = detailed.get("evaluation_provenance", {})
        _require(provenance.get("validated") is True, "paper evaluation provenance failed")
        _require(provenance.get("traffic_partition") == job.evaluation_split, "provenance partition drifted")
        _require(provenance.get("evaluation_seed_start") == job.evaluation_seed_start, "provenance seed drifted")
        if job.algorithm == TEMPORAL_GRAPH:
            _validate_control(root, job)
        else:
            _require((root / "collision_label_diagnostics.json").is_file(), "collision diagnostics missing")
            _validate_candidate(root, job)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        V411ProtocolError,
        json.JSONDecodeError,
    ) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


def _run_row(job: Job) -> dict[str, Any]:
    root = run_directory(job)
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    summary = detailed["summary"]
    selector = _load_json(root / "selector" / "receipt.json")
    actions = _load_json(root / "action_diagnostics.json")
    row = {
        "stage": job.stage,
        "job_id": job.job_id,
        "job": job.name,
        "kind": job.kind,
        "role": job.role,
        "method": job.method,
        "algorithm": job.algorithm,
        "implementation_id": job.implementation_id,
        "scenario": job.scenario,
        "seed": job.seed,
        "raw_steps": job.raw_steps,
        "evaluation_episodes": int(summary["episodes"]),
        "success_rate": float(summary["success_rate"]),
        "collision_rate": float(summary["collision_rate"]),
        "off_route_rate": float(summary["off_route_rate"]),
        "timeout_rate": float(summary["timeout_rate"]),
        "mean_return": float(summary["mean_return"]),
        "mean_decision_steps": float(summary["mean_decision_steps"]),
        "mean_raw_steps": float(summary["mean_raw_steps"]),
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_model_sha256": _sha256(root / "selected_model.zip"),
        "selector_receipt_sha256": _sha256(root / "selector" / "receipt.json"),
        "action_diagnostics_sha256": _sha256(root / "action_diagnostics.json"),
        "selected_action_mask_feasible_rate": actions.get("selected_action_mask_feasible_rate"),
        "inference_safety_rule_added": actions.get("inference_safety_rule_added", False),
        "external_kinematic_projection": actions.get("external_kinematic_projection", False),
        "kinematic_safety_projection": actions.get("kinematic_safety_projection", False),
        "traffic_risk_in_lane_mask": actions.get("traffic_risk_in_lane_mask", False),
        "action_postprocessing_override": actions.get("action_postprocessing_override", False),
    }
    if job.algorithm != TEMPORAL_GRAPH:
        labels = _load_json(root / "collision_label_diagnostics.json")
        training = _load_json(root / "training_diagnostics.json")
        statistics = training.get("statistics", {})

        def statistic(name: str, field: str = "last") -> Any:
            return statistics.get(name, {}).get(field)

        row.update(
            {
                "shared_risk_encoder_present": actions.get("shared_risk_encoder_present"),
                "learned_speed_components_per_lane": actions.get("learned_speed_components_per_lane"),
                "exact_supported_mixture_model_argmax_rate": actions.get("exact_supported_mixture_model_argmax_rate"),
                "supported_mixture_score_equation_match_rate": actions.get("supported_mixture_score_equation_match_rate"),
                "learned_proposal_source_rate": actions.get("learned_proposal_source_rate"),
                "no_action_rewrite_rate": actions.get("no_action_rewrite_rate"),
                "optimizer_overlap_count": actions.get("optimizer_ownership", {}).get("overlap_count"),
                "sampled_collision_label_count": labels.get("sampled_label_count"),
                "sampled_positive_collision_label_rate": labels.get("sampled_positive_label_rate"),
                "collision_label_diagnostics_sha256": _sha256(root / "collision_label_diagnostics.json"),
                "collision_return_loss": training.get("collision_return_loss"),
                "collision_pairwise_rank_coef": training.get("collision_pairwise_rank_coef"),
                "collision_temporal_consistency_coef": training.get(
                    "collision_temporal_consistency_coef"
                ),
                "collision_soft_bce_loss": statistic(
                    "train/collision_soft_bce_loss"
                ),
                "collision_pairwise_rank_loss": statistic(
                    "train/collision_pairwise_rank_loss"
                ),
                "collision_pairwise_active_pair_rate_maximum": statistic(
                    "risk/collision_pairwise_active_pair_rate", "maximum"
                ),
                "collision_temporal_consistency_loss": statistic(
                    "train/collision_temporal_consistency_loss"
                ),
                "collision_probability_absolute_error": statistic(
                    "risk/collision_probability_absolute_error"
                ),
            }
        )
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["scenario"])), []).append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "mean_decision_steps",
        "mean_raw_steps",
    )
    for (method, scenario), items in sorted(groups.items()):
        episodes = sum(int(row["evaluation_episodes"]) for row in items)
        aggregate = {
            key: sum(float(row[key]) * int(row["evaluation_episodes"]) for row in items) / episodes
            for key in metrics
        }
        output.append(
            {
                "method": method,
                "scenario": scenario,
                "runs": len(items),
                "episodes": episodes,
                "seeds": sorted(int(row["seed"]) for row in items),
                **aggregate,
            }
        )
    return output


def summarize_stage(
    contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str]
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job) for job in jobs if accepted_run(job, hashes)]
    payload = {
        "schema_version": "topo-scene-v4.11.stage-results/v1",
        "stage": stage,
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "formal_test_accessed": stage == "formal" and bool(rows),
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
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
        keys = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        root.mkdir(parents=True, exist_ok=True)
        with (root / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return payload


def _candidate_gate(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["development"]["per_run_gates"][row["scenario"]]

    def finite(name: str) -> bool:
        value = row.get(name)
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    checks = {
        "success_rate": float(row["success_rate"]) >= float(thresholds["success_rate_min"]),
        "collision_rate": float(row["collision_rate"]) <= float(thresholds["collision_rate_max"]),
        "off_route_rate": float(row["off_route_rate"]) <= float(thresholds["off_route_rate_max"]),
        "timeout_rate": float(row["timeout_rate"]) <= float(thresholds["timeout_rate_max"]),
        "shared_risk_encoder": row.get("shared_risk_encoder_present") is True,
        "exact_model_argmax": row.get("exact_supported_mixture_model_argmax_rate") == 1.0,
        "score_equation": row.get("supported_mixture_score_equation_match_rate") == 1.0,
        "learned_proposal_source": row.get("learned_proposal_source_rate") == 1.0,
        "no_action_rewrite": row.get("no_action_rewrite_rate") == 1.0,
        "optimizer_owners_disjoint": row.get("optimizer_overlap_count") == 0,
        "no_inference_safety_rule": row.get("inference_safety_rule_added") is False,
        "no_kinematic_projection": row.get("external_kinematic_projection") is False,
        "no_kinematic_safety_projection": row.get("kinematic_safety_projection") is False,
        "no_traffic_risk_lane_mask": row.get("traffic_risk_in_lane_mask") is False,
        "no_action_postprocessing": row.get("action_postprocessing_override") is False,
        "feasible_lane_mask": row.get("selected_action_mask_feasible_rate") == 1.0,
        "proper_soft_bernoulli_loss": row.get("collision_return_loss")
        == "soft_target_binary_cross_entropy_with_logits",
        "preregistered_pairwise_rank_coef": row.get("collision_pairwise_rank_coef")
        == 0.25,
        "preregistered_temporal_consistency_coef": row.get(
            "collision_temporal_consistency_coef"
        )
        == 0.05,
        "collision_soft_bce_finite": finite("collision_soft_bce_loss"),
        "collision_pairwise_rank_loss_finite": finite(
            "collision_pairwise_rank_loss"
        ),
        "collision_pairwise_active": finite(
            "collision_pairwise_active_pair_rate_maximum"
        )
        and float(row["collision_pairwise_active_pair_rate_maximum"]) > 0.0,
        "collision_temporal_consistency_finite": finite(
            "collision_temporal_consistency_loss"
        ),
        "collision_probability_error_finite": finite(
            "collision_probability_absolute_error"
        ),
    }
    outcome_keys = {"success_rate", "collision_rate", "off_route_rate", "timeout_rate"}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "outcome_passed": all(checks[key] for key in outcome_keys),
        "model_integrity_passed": all(
            value for key, value in checks.items() if key not in outcome_keys
        ),
    }


def compute_development_gate(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "development", hashes)
    rows = summary["per_run"]
    per_run = {row["job_id"]: _candidate_gate(row, contract) for row in rows}
    cross = [row for row in rows if row["scenario"] == "cross"]
    cross_success = sum(row["success_rate"] for row in cross) / len(cross) if cross else float("nan")
    cross_collision = sum(row["collision_rate"] for row in cross) / len(cross) if cross else float("nan")
    aggregate_rule = contract["development"]["aggregate_cross_gate"]
    checks = {
        "complete_matrix": summary["complete"] is True,
        "all_per_run_gates": len(per_run) == 4 and all(row["passed"] for row in per_run.values()),
        "cross_success": len(cross) == 2 and cross_success >= float(aggregate_rule["success_rate_min"]),
        "cross_collision": len(cross) == 2 and cross_collision <= float(aggregate_rule["collision_rate_max"]),
        "positive_cross_cell": any(
            row["success_rate"] >= float(aggregate_rule["positive_cell_success_rate_min"])
            and row["collision_rate"] <= float(aggregate_rule["positive_cell_collision_rate_max"])
            for row in cross
        ),
        "formal_test_untouched": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "topo-scene-v4.11.development-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if passed else "fail",
        "ablation_unlocked": passed,
        "promotion_unlocked": False,
        "formal_test_accessed": False,
        **hashes,
        "development_summary": str((stage_root("development") / "summary.json").resolve()),
        "development_summary_sha256": _sha256(stage_root("development") / "summary.json"),
        "checks": checks,
        "per_run_gates": per_run,
        "cross_aggregate": {
            "cells": len(cross),
            "success_rate": cross_success if cross else None,
            "collision_rate": cross_collision if cross else None,
        },
        "next_stage": "required_ablation" if passed else "complete_attribution_and_new_version",
    }
    _write_json(DEFAULT_DEVELOPMENT_DECISION, value)
    return value


def compute_ablation_decision(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    validate_development_decision()
    summary = summarize_stage(contract, digest, "ablation", hashes)
    jobs = jobs_for_stage(contract, digest, "ablation")
    integrity = {
        job.name: accepted_run(job, hashes)
        for job in jobs
    }
    complete = summary["complete"] is True and all(integrity.values())
    value = {
        "schema_version": "topo-scene-v4.11.ablation-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "complete" if complete else "incomplete",
        "promotion_unlocked": complete,
        "formal_test_accessed": False,
        **hashes,
        "accepted_runs": summary["accepted_runs"],
        "expected_runs": summary["expected_runs"],
        "integrity": integrity,
        "ablation_summary_sha256": _sha256(stage_root("ablation") / "summary.json"),
        "next_stage": "fresh_promotion" if complete else "repair_ablation_evidence",
    }
    _write_json(DEFAULT_ABLATION_DECISION, value)
    return value


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["method"], row["scenario"], int(row["seed"])): row for row in rows
    }
    output = []
    keys = sorted({(row["scenario"], int(row["seed"])) for row in rows})
    for scenario, seed in keys:
        control = indexed[("temporal_graph", scenario, seed)]
        candidate = indexed[("prcr_full", scenario, seed)]
        output.append(
            {
                "scenario": scenario,
                "seed": seed,
                "success_rate_delta": candidate["success_rate"] - control["success_rate"],
                "collision_rate_delta": candidate["collision_rate"] - control["collision_rate"],
                "off_route_rate_delta": candidate["off_route_rate"] - control["off_route_rate"],
                "timeout_rate_delta": candidate["timeout_rate"] - control["timeout_rate"],
                "mean_return_delta": candidate["mean_return"] - control["mean_return"],
            }
        )
    return output


def compute_promotion_gate(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    validate_development_decision()
    validate_ablation_decision()
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {(row["method"], row["scenario"]): row for row in summary["aggregate"]}
    scenarios = contract["promotion"]["scenarios"]
    success = {
        scenario: aggregate[("prcr_full", scenario)]["success_rate"]
        - aggregate[("temporal_graph", scenario)]["success_rate"]
        for scenario in scenarios
    }
    collision = {
        scenario: aggregate[("prcr_full", scenario)]["collision_rate"]
        - aggregate[("temporal_graph", scenario)]["collision_rate"]
        for scenario in scenarios
    }
    paired = _paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {
        f"{row['scenario']}__seed{row['seed']}": _candidate_gate(row, contract)
        for row in rows
        if row["method"] == "prcr_full"
    }
    checks = {
        "candidate_development_guards_each_seed": all(
            row["passed"] for row in candidate_gates.values()
        ),
        "all_candidate_off_route_zero": all(
            aggregate[("prcr_full", scenario)]["off_route_rate"] <= 0.0
            for scenario in scenarios
        ),
        "per_scenario_collision_noninferiority": all(
            delta <= float(rules["per_scenario_collision_noninferiority_margin"])
            for delta in collision.values()
        ),
        "worst_paired_seed_success_noninferiority": min(
            row["success_rate_delta"] for row in paired
        )
        >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success.values())
        >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": max(success.values())
        >= float(rules["positive_effect_success_gain"])
        or min(collision.values())
        <= -float(rules["positive_effect_collision_reduction"]),
        "learned_target_only_selection": all(
            row["selected_checkpoint_kind"]
            in ("highest_training_success", "exact_final")
            and row["selected_deployment_decoder"] == TARGET_DECODER
            for row in rows
            if row["method"] == "prcr_full"
        ),
    }
    passed = all(checks.values())
    value = {
        "schema_version": "topo-scene-v4.11.promotion-gate/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if passed else "fail",
        **hashes,
        "promotion_results": str((stage_root("promotion") / "summary.json").resolve()),
        "promotion_results_sha256": _sha256(stage_root("promotion") / "summary.json"),
        "formal_algorithms": list(V411_FORMAL_ALGORITHMS),
        "checks": checks,
        "candidate_per_seed_gates": candidate_gates,
        "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision,
        "worst_paired_seed_success_delta": min(
            row["success_rate_delta"] for row in paired
        ),
        "macro_success_delta": _mean(success.values()),
        "paired_seed_deltas": paired,
        "formal_test_unlocked": passed,
        "formal_test_accessed": False,
        "next_stage": "untouched_formal_test" if passed else "complete_attribution_and_new_version",
    }
    _write_json(DEFAULT_PROMOTION_GATE, value)
    return value


def compute_formal_decision(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    value = {
        "schema_version": "topo-scene-v4.11.formal-decision/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "computed_from_real_runs": True,
        "fabricated_values": False,
        **hashes,
        "formal_test_accessed": True,
        "summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "status": "complete_pending_preregistered_statistical_analysis",
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


def write_plan(
    contract: dict[str, Any],
    digest: str,
    stage: str,
    hashes: dict[str, str],
    *,
    device: str,
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    value = {
        "schema_version": "topo-scene-v4.11.stage-plan/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "device": device,
        "execution_policy": "attempt_all_then_summarize",
        "failure_does_not_cancel_remaining_jobs": True,
        **hashes,
        "jobs": [
            {
                **asdict(job),
                "name": job.name,
                "run_directory": str(run_directory(job).resolve()),
                "accepted": accepted_run(job, hashes),
                "command": command_for(job, hashes, device=device),
            }
            for job in jobs
        ],
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


def execute_jobs(
    contract: dict[str, Any],
    digest: str,
    stage: str,
    hashes: dict[str, str],
    *,
    selector: str | None,
    device: str,
    workers: int,
    dry_run: bool,
) -> int:
    _assert_stage_can_run(hashes, stage)
    _require(workers == 1, "v4.11 freezes one sequential GPU worker")
    jobs = jobs_for_stage(contract, digest, stage)
    selected = _select_jobs(jobs, selector)
    plan = write_plan(contract, digest, stage, hashes, device=device)
    if dry_run:
        print(
            json.dumps(
                {"selected": [job.name for job in selected], "plan": plan},
                ensure_ascii=False,
                indent=2,
            )
        )
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
            attempts.append(
                {
                    "job": job.name,
                    "status": "immutable_incomplete_run",
                    "acceptance_reason": reason,
                }
            )
            continue
        log_path = log_root / f"{job.name}.log"
        print(
            json.dumps(
                {"job": job.name, "status": "running", "log": str(log_path)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command_for(job, hashes, device=device),
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            accepted, reason = accepted_run_reason(job, hashes)
            status = "accepted" if completed.returncode == 0 and accepted else "failed"
            attempt = {
                "job": job.name,
                "status": status,
                "returncode": completed.returncode,
                "acceptance_reason": reason,
                "log": str(log_path.resolve()),
            }
        except Exception as exc:
            attempt = {
                "job": job.name,
                "status": "launcher_exception",
                "returncode": None,
                "acceptance_reason": f"{type(exc).__name__}: {exc}",
                "log": str(log_path.resolve()),
            }
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)
    execution = {
        "schema_version": "topo-scene-v4.11.execution/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    failed = [
        row
        for row in attempts
        if row["status"] not in {"accepted", "already_accepted"}
    ]
    return 1 if failed else 0


def status_payload(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    stages = {}
    for stage in STAGES:
        jobs = jobs_for_stage(contract, digest, stage)
        accepted = sum(accepted_run(job, hashes) for job in jobs)
        stages[stage] = {
            "accepted": accepted,
            "expected": len(jobs),
            "complete": accepted == len(jobs),
        }
    development = (
        _load_json(DEFAULT_DEVELOPMENT_DECISION)
        if DEFAULT_DEVELOPMENT_DECISION.is_file()
        else None
    )
    ablation = (
        _load_json(DEFAULT_ABLATION_DECISION)
        if DEFAULT_ABLATION_DECISION.is_file()
        else None
    )
    promotion = (
        _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    )
    return {
        "schema_version": "topo-scene-v4.11.status/v1",
        **hashes,
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
        return execute_jobs(
            contract,
            digest,
            args.stage,
            hashes,
            selector=args.job,
            device=args.device,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    elif args.command == "summarize":
        output = summarize_stage(contract, digest, args.stage, hashes)
    elif args.command == "gate":
        output = {
            "development": compute_development_gate,
            "ablation": compute_ablation_decision,
            "promotion": compute_promotion_gate,
            "formal": compute_formal_decision,
        }[args.stage](contract, digest, hashes)
    else:
        output = status_payload(contract, digest, hashes)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
