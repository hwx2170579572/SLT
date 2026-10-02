"""Run the hash-bound v4.9 learned collision-value experiment protocol.

Every selected job is attempted before a batch result is returned.  Scientific
gates are computed only from complete, accepted matrices, and formal runs stay
locked behind a passing promotion receipt.
"""

from __future__ import annotations

import argparse
import copy
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_9 import (
    TEMPORAL_GRAPH,
    V49_CANDIDATE,
    V49_FORMAL_ALGORITHMS,
    V49_IMPLEMENTATION_IDS,
)
from tools import run_topo_v4_8_experiments as parent
from tools.checkpoint_decoder_selector_v4_6 import (
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
)


scientific = parent.scientific
PYTHON_EXECUTABLE = Path("D:/Programs/Anaconda/envs/llm_pipeline/python.exe")
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_9.yaml"
)
DEFAULT_DESIGN = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "V4_9_LEARNED_COLLISION_VALUE_MODEL.md"
)
DEFAULT_STAGE0_RESULTS = parent.DEFAULT_STAGE0_RESULTS
DEFAULT_STAGE0_ATTRIBUTION = parent.DEFAULT_STAGE0_ATTRIBUTION
DEFAULT_FAILURE_ATTRIBUTION = (
    PROJECT_ROOT
    / "results_topo_v4_8_promotion"
    / "attribution"
    / "promotion_failure_v4_8.json"
)
DEFAULT_DEEP_ATTRIBUTION = DEFAULT_FAILURE_ATTRIBUTION.with_name(
    "PROMOTION_FAILURE_V4_8.md"
)
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "results_topo_v4_9_dev" / "preregistration_receipt.json"
)
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_9_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = (
    PROJECT_ROOT / "results_topo_v4_9_promotion" / "promotion_gate.json"
)

LINEAGE_HASHES = {
    "parent_contract_sha256": "42f54135e4d105563b0ad2113147bfef7811c80664793419538a18e9d8c08e51",
    "parent_scientific_freeze_sha256": "5827ca63caf5cfd187ac5ad0bdc2fd0fbb0fb00c324459007cf24620f0b57aa6",
    "parent_acceptance_patch_contract_sha256": "00ba144b723f86a047a78022aab4228cc4caf5fe566543b9c7dc273e5aaa19d4",
    "parent_acceptance_patch_freeze_sha256": "be1653a5b20322cbcd376537a90445c5087811b69d24c2b990df2306a43a2528",
    "parent_promotion_gate_sha256": "581d09ffecef472ea0581d76041198247b3cf05849b50e1d3b11e0cc7f80de32",
    "failure_attribution_sha256": "80e2d132e507bb8bcdf4d306f1842e7f71a0dee744921efe853bb2bbccba3ea6",
    "deep_attribution_sha256": "314a1a32771b2966eb807aefaeb904310b9e719a5de0d22b4d3314fa88b0acfb",
    "attribution_tool_sha256": "5ede54634bc9c2a91f07653e04eb8d5c23f4bcfe69becd2855eff61977ea66d6",
}

SCENARIOS = scientific.SCENARIOS
STAGES = scientific.STAGES
METHOD_ALGORITHMS = {
    "temporal_graph": TEMPORAL_GRAPH,
    "selected_v4_candidate": V49_CANDIDATE,
}
METHOD_CODES = {"temporal_graph": "tg", "selected_v4_candidate": "cand"}
SCENARIO_CODES = scientific.SCENARIO_CODES
FULL_RUN_REQUIRED_FILES = parent.FULL_RUN_REQUIRED_FILES
Job = parent.Job


class V49ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V49ProtocolError(message)


def _sha256(path: Path) -> str:
    return scientific._sha256(path)


def _canonical_sha256(value: Any) -> str:
    return scientific._canonical_sha256(value)


def _load_json(path: Path) -> dict[str, Any]:
    return scientific._load_json(path)


def _version_tree(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        schema = output.get("schema_version")
        if isinstance(schema, str):
            for old in ("topo-scene-v4.7.", "topo-scene-v4.8."):
                if schema.startswith(old):
                    output["schema_version"] = schema.replace(
                        old, "topo-scene-v4.9.", 1
                    )
                    break
        output = {key: _version_tree(item) for key, item in output.items()}
    elif isinstance(output, list):
        output = [_version_tree(item) for item in output]
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_version_tree(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.9 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.9.learned-collision-value-model-contract/v1",
        "unexpected v4.9 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication drifted")
    _require(contract.get("formal_test_locked") is True, "formal must start locked")
    lineage = contract.get("lineage", {})
    for key, expected in LINEAGE_HASHES.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")
    _require(lineage.get("formal_test_accessed") is False, "lineage accessed formal")

    change = contract.get("single_change", {})
    _require(
        change.get("name") == "learned_twin_collision_value_model",
        "single change drifted",
    )
    supervision = change.get("supervision", {})
    _require(
        supervision.get("source") == "observed_environment_info_collision",
        "collision supervision source drifted",
    )
    for key in (
        "future_or_oracle_labels_used",
        "manual_geometry_labels_used",
        "reward_normalization_used_for_labels",
    ):
        _require(supervision.get(key) is False, f"supervision.{key} drifted")
    _require(change.get("collision_risk_coef") == 1.0, "risk coefficient drifted")
    for key in (
        "external_kinematic_projection",
        "headway_threshold",
        "lane_change_veto",
        "action_postprocessing_override",
    ):
        _require(change.get(key) is False, f"rule mechanism enabled: {key}")

    development = contract.get("development", {})
    _require(
        development.get("execution_policy") == "attempt_all_then_summarize",
        "development execution policy drifted",
    )
    _require(
        development.get("failure_does_not_cancel_remaining_jobs") is True,
        "development failure cancellation drifted",
    )
    expected_cells = [
        ("M1", "cross", 10, 131_000, 131_100, 141_000),
        ("M2", "cross", 11, 132_000, 132_100, 142_000),
        ("M3", "roundabout_medium", 12, 133_000, 133_100, 143_000),
        ("M4", "carla", 13, 134_000, 134_100, 144_000),
    ]
    observed_cells = [
        (
            str(row["id"]),
            str(row["scenario"]),
            int(row["seed"]),
            int(row["calibration_seed_start"]),
            int(row["secondary_calibration_seed_start"]),
            int(row["evaluation_seed_start"]),
        )
        for row in development.get("ordered_cells", [])
    ]
    _require(observed_cells == expected_cells, "development matrix drifted")
    _require(int(development["evaluation_episodes"]) == 20, "development eval drifted")
    _require(
        all(int(row["raw_steps"]) == 20_000 for row in development["ordered_cells"]),
        "development budget drifted",
    )

    promotion = contract.get("promotion", {})
    _require(
        promotion.get("execution_policy") == "attempt_all_then_summarize",
        "promotion execution policy drifted",
    )
    _require(promotion.get("seeds") == [20, 21], "promotion seeds drifted")
    _require(
        len(promotion["methods"])
        * len(promotion["scenarios"])
        * len(promotion["seeds"])
        == 12,
        "promotion matrix drifted",
    )
    formal = contract.get("formal_test", {})
    _require(formal.get("locked") is True, "formal lock drifted")
    _require(formal.get("accessed") is False, "formal accessed in contract")
    _require(
        len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"])
        == 120,
        "formal matrix drifted",
    )
    _require(
        formal["methods"] == ["temporal_graph", "selected_v4_candidate"],
        "formal pair drifted",
    )
    return contract


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return _sha256(path.resolve())


def _validate_failure_attribution(summary: Path, deep: Path) -> dict[str, str]:
    payload = _load_json(summary)
    _require(payload.get("computed_from_real_runs") is True, "attribution is not real")
    _require(payload.get("fabricated_values") is False, "attribution is fabricated")
    _require(payload.get("post_hoc_diagnostic_only") is True, "attribution scope drifted")
    _require(payload.get("counted_for_promotion_gate") is False, "posthoc entered gate")
    _require(payload.get("formal_test_accessed") is False, "attribution accessed formal")
    _require(_sha256(summary) == LINEAGE_HASHES["failure_attribution_sha256"], "attribution hash drifted")
    _require(_sha256(deep) == LINEAGE_HASHES["deep_attribution_sha256"], "deep attribution hash drifted")
    return {
        "failure_attribution_sha256": _sha256(summary),
        "deep_attribution_sha256": _sha256(deep),
    }


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = _load_json(path.resolve())
    _require(
        payload.get("schema_version") == "topo-scene-v4.9.implementation-freeze/v1",
        "invalid v4.9 freeze",
    )
    _require(payload.get("formal_test_accessed") is False, "freeze accessed formal")
    _require(
        payload.get("experiment_contract_sha256") == contract_sha256(),
        "freeze contract hash drifted",
    )
    _require(
        payload.get("failure_attribution_sha256")
        == LINEAGE_HASHES["failure_attribution_sha256"],
        "freeze attribution drifted",
    )
    _require(
        payload.get("preregistration_receipt_sha256")
        == _sha256(DEFAULT_PREREGISTRATION),
        "freeze preregistration drifted",
    )
    files = payload.get("scientific_files")
    _require(isinstance(files, dict) and files, "freeze scientific files missing")
    for relative, expected in files.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"frozen source missing: {relative}")
        _require(_sha256(source) == expected, f"frozen source drifted: {relative}")
    _require(
        _canonical_sha256(files) == payload.get("scientific_content_sha256"),
        "freeze content digest invalid",
    )
    return payload


def protocol_hashes(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    stage0_results: Path = DEFAULT_STAGE0_RESULTS,
    stage0_attribution: Path = DEFAULT_STAGE0_ATTRIBUTION,
    failure_attribution: Path = DEFAULT_FAILURE_ATTRIBUTION,
    deep_attribution: Path = DEFAULT_DEEP_ATTRIBUTION,
    freeze_path: Path = DEFAULT_FREEZE,
) -> dict[str, str]:
    validate_contract(load_contract(contract_path.resolve()))
    hashes = {
        "experiment_contract_sha256": _sha256(contract_path.resolve()),
        **scientific.parent._validate_stage0(
            stage0_results.resolve(), stage0_attribution.resolve()
        ),
        **_validate_failure_attribution(
            failure_attribution.resolve(), deep_attribution.resolve()
        ),
        "preregistration_receipt_sha256": _sha256(DEFAULT_PREREGISTRATION),
    }
    validate_implementation_freeze(freeze_path.resolve())
    hashes["implementation_freeze_sha256"] = _sha256(freeze_path.resolve())
    return hashes


def jobs_for_stage(
    contract: dict[str, Any], digest: str, stage: str
) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.9 stage {stage!r}")
    tag = digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        for specification in settings["ordered_cells"]:
            method = str(specification["method"])
            algorithm = METHOD_ALGORITHMS[method]
            jobs.append(
                Job(
                    stage=stage,
                    job_id=str(specification["id"]),
                    kind=str(specification["kind"]),
                    method=method,
                    algorithm=algorithm,
                    implementation_id=V49_IMPLEMENTATION_IDS[algorithm],
                    scenario=str(specification["scenario"]),
                    seed=int(specification["seed"]),
                    raw_steps=int(specification["raw_steps"]),
                    calibration_episodes=int(
                        settings["primary_calibration_episodes_per_pair"]
                    ),
                    calibration_seed_start=int(
                        specification["calibration_seed_start"]
                    ),
                    evaluation_episodes=int(settings["evaluation_episodes"]),
                    evaluation_split=str(settings["evaluation_split"]),
                    evaluation_seed_start=int(
                        specification["evaluation_seed_start"]
                    ),
                    role=str(specification["role"]),
                    protocol_tag=tag,
                )
            )
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
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
                            kind="fresh_training_with_train_only_model_selection",
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
                            role=(
                                "fresh_paired_promotion"
                                if stage == "promotion"
                                else "untouched_formal_test"
                            ),
                            protocol_tag=tag,
                        )
                    )
    expected = {"development": 4, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} job")
    return jobs


def stage_root(stage: str) -> Path:
    return {
        "development": PROJECT_ROOT / "results_topo_v4_9_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_9_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_9_formal",
    }[stage]


def run_directory(job: Job) -> Path:
    return stage_root(job.stage) / "runs" / job.name


def command_for(
    job: Job,
    hashes: dict[str, str],
    *,
    device: str,
    promotion_gate: Path = DEFAULT_PROMOTION_GATE,
) -> list[str]:
    command = [
        str(PYTHON_EXECUTABLE),
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_9.py"),
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
        hashes["failure_attribution_sha256"],
        "--implementation-freeze-sha256",
        hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(
            ["--formal-unlock-receipt", str(promotion_gate.resolve())]
        )
    return command


def _replace_parent_policy_names(value: Any) -> Any:
    names = {
        "CollisionConstrainedTargetCriticSACPolicyV49": (
            "TargetCriticDecisionAlignedSACPolicyV43"
        ),
        "CollisionConstrainedFusionSACPolicyV49": "ConfidentActorFusionSACPolicyV45",
    }
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_parent_policy_names(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_parent_policy_names(item) for item in value]
    return value


def _parent_schema_view(value: Any) -> Any:
    output = copy.deepcopy(value)
    if isinstance(output, dict):
        schema = output.get("schema_version")
        if isinstance(schema, str) and schema.startswith("topo-scene-v4.9."):
            output["schema_version"] = schema.replace(
                "topo-scene-v4.9.", "topo-scene-v4.8.", 1
            )
        output = {key: _parent_schema_view(item) for key, item in output.items()}
    elif isinstance(output, list):
        output = [_parent_schema_view(item) for item in output]
    return output


@contextmanager
def _parent_acceptance_view(job: Job) -> Iterator[None]:
    """Expose only audited v4.9 deltas as their v4.8 structural equivalents.

    This read-only view lets the mature parent validator recheck every unchanged
    artifact contract.  The real v4.9 fields are then validated separately.
    No file is rewritten.
    """

    original = {
        "candidate": parent.V48_CANDIDATE,
        "ids": parent.V48_IMPLEMENTATION_IDS,
        "run_directory": parent.run_directory,
        "load_json": parent._load_json,
        "load_trace": parent._load_trace,
        "scientific_run_directory": scientific.run_directory,
    }

    def compatibility_json(path: Path) -> dict[str, Any]:
        value = _replace_parent_policy_names(_parent_schema_view(_load_json(path)))
        if path.name == "arguments.json":
            requested = value.get("requested_raw_steps", {})
            if job.algorithm == TEMPORAL_GRAPH:
                requested["secondary_calibration_seed_start"] = (
                    job.calibration_seed_start + 100
                )
            else:
                fidelity = value.get("implementation_fidelity", {})
                fidelity["single_change"] = "tie_only_replicated_train_calibration"
                fidelity["training_changed_from_v4_7"] = False
                fidelity["decoder_changed"] = False
        if job.algorithm == V49_CANDIDATE:
            if path.name == "return_estimator_diagnostics.json":
                value["return_estimator_changed_from_v4_7"] = False
            elif path.name == "training_diagnostics.json":
                value.get("return_estimator", {})[
                    "return_estimator_changed_from_v4_7"
                ] = False
        if path.name == "action_diagnostics.json" and job.algorithm == V49_CANDIDATE:
            value["exact_target_critic_argmax_rate"] = 1.0
            value["target_keep_tie_rule_match_rate"] = 1.0
            value["exact_fusion_rule_match_rate"] = 1.0
            value["target_fallback_rule_match_rate"] = 1.0
        return value

    def compatibility_trace(path: Path) -> list[dict[str, Any]]:
        rows = scientific._load_trace(path)
        output = copy.deepcopy(rows)
        for row in output:
            for key in ("target_critic_decoder", "fusion_decoder"):
                decoder = row.get(key)
                if isinstance(decoder, dict) and "risk_adjusted_target_score" in decoder:
                    decoder["minimum_target_twin_q"] = decoder[
                        "risk_adjusted_target_score"
                    ]
        return output

    parent.V48_CANDIDATE = V49_CANDIDATE
    parent.V48_IMPLEMENTATION_IDS = V49_IMPLEMENTATION_IDS
    parent.run_directory = run_directory
    parent._load_json = compatibility_json
    parent._load_trace = compatibility_trace
    scientific.run_directory = run_directory
    try:
        yield
    finally:
        parent.V48_CANDIDATE = original["candidate"]
        parent.V48_IMPLEMENTATION_IDS = original["ids"]
        parent.run_directory = original["run_directory"]
        parent._load_json = original["load_json"]
        parent._load_trace = original["load_trace"]
        scientific.run_directory = original["scientific_run_directory"]


def _strict_candidate_evidence(root: Path, job: Job) -> dict[str, Any]:
    arguments = _load_json(root / "arguments.json")
    metadata = _load_json(root / "method_metadata.json")
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    actions = _load_json(root / "action_diagnostics.json")
    training = _load_json(root / "training_diagnostics.json")
    selector = _load_json(root / "selector" / "receipt.json")
    labels = _load_json(root / "collision_label_diagnostics.json")

    _require(
        arguments.get("schema_version") == "topo-scene-v4.9.run-arguments/v1",
        "v4.9 argument schema drifted",
    )
    fidelity = arguments.get("implementation_fidelity", {})
    expected = {
        "implementation_id": job.implementation_id,
        "single_change": "learned_twin_collision_value_actor_constraint",
        "network_changed": True,
        "training_changed_from_v4_8": True,
        "return_estimator_changed_from_v4_8": False,
        "learned_collision_critic": True,
        "collision_risk_coef": 1.0,
        "inference_safety_rule_added": False,
        "selector_changed_from_v4_8": False,
        "decoder_changed_from_v4_8": True,
        "decoder_change_kind": "learned_reward_minus_collision_value_scoring",
    }
    for key, value in expected.items():
        _require(fidelity.get(key) == value, f"v4.9 fidelity {key} drifted")
    _require(metadata.get("learned_collision_critic") is True, "metadata lost collision model")
    _require(metadata.get("collision_label_source") == "observed_environment_info_collision", "metadata label source drifted")
    _require(metadata.get("inference_safety_rule_added") is False, "metadata admits inference rule")
    _require(detailed.get("learned_collision_critic") is True, "detailed lost collision model")
    _require(detailed.get("inference_safety_rule_added") is False, "detailed admits rule")
    _require(
        actions.get("schema_version") == "topo-scene-v4.9.action-diagnostics/v1",
        "v4.9 action schema drifted",
    )
    for key, value in {
        "learned_collision_critic_present": True,
        "learned_collision_critic_target_present": True,
        "collision_risk_coef": 1.0,
        "inference_safety_rule_added": False,
        "model_decision_equation": "minimum_reward_q_minus_lambda_maximum_collision_value",
        "selected_decoder_exact_rule_match_rate": 1.0,
        "selected_decoder_exact_action_match_rate": 1.0,
        "selected_action_mask_feasible_rate": 1.0,
        "risk_adjusted_score_equation_match_rate": 1.0,
    }.items():
        _require(actions.get(key) == value, f"v4.9 action evidence {key} drifted")
    decoder = selector.get("selected_deployment_decoder")
    expected_policy = {
        TARGET_DECODER: "CollisionConstrainedTargetCriticSACPolicyV49",
        FUSION_DECODER: "CollisionConstrainedFusionSACPolicyV49",
    }.get(decoder)
    _require(expected_policy is not None, "candidate selected unsupported decoder")
    _require(
        selector.get("selected_model_policy_class") == expected_policy,
        "selected learned policy class drifted",
    )
    if decoder == TARGET_DECODER:
        _require(actions.get("exact_risk_adjusted_model_argmax_rate") == 1.0, "risk target argmax drifted")
        _require(actions.get("risk_adjusted_keep_tie_match_rate") == 1.0, "risk keep tie drifted")
    else:
        _require(actions.get("exact_risk_adjusted_model_fusion_rate") == 1.0, "risk fusion drifted")
        _require(actions.get("actor_override_predicate_valid_rate") == 1.0, "actor predicate drifted")
        _require(actions.get("risk_adjusted_target_fallback_match_rate") == 1.0, "risk fallback drifted")

    _require(
        labels.get("schema_version")
        == "topo-scene-v4.9.collision-label-diagnostics/v1",
        "collision label schema drifted",
    )
    _require(labels.get("source") == "observed_environment_info_collision", "collision label source drifted")
    _require(labels.get("future_or_oracle_labels_used") is False, "oracle label used")
    _require(labels.get("reward_return_changed") is False, "reward return changed")
    _require(int(labels.get("sampled_label_count", 0)) > 0, "no collision labels sampled")
    _require(int(labels.get("stored_transition_count_including_source_duplicates", 0)) > 0, "no collision-labelled transitions stored")
    positive_rate = labels.get("sampled_positive_label_rate")
    _require(positive_rate is not None and 0.0 <= float(positive_rate) <= 1.0, "collision positive rate invalid")
    statistics = training.get("statistics", {})
    for name in (
        "train/collision_critic_loss",
        "risk/collision_cost_label",
        "risk/collision_cost_target",
        "risk/collision_cost_replay_prediction",
        "risk/policy_expected_collision_cost",
        "risk/collision_risk_actor_penalty",
    ):
        value = statistics.get(name, {}).get("last")
        _require(value is not None and math.isfinite(float(value)), f"missing/invalid {name}")
    return labels


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        with _parent_acceptance_view(job):
            accepted, reason = parent.accepted_run_reason(job, hashes)
        _require(accepted, f"parent structural acceptance failed: {reason}")
        arguments = _load_json(root / "arguments.json")
        actions = _load_json(root / "action_diagnostics.json")
        if job.algorithm == V49_CANDIDATE:
            _require((root / "collision_label_diagnostics.json").is_file(), "collision diagnostics missing")
            _strict_candidate_evidence(root, job)
        else:
            fidelity = arguments.get("implementation_fidelity", {})
            _require(fidelity.get("implementation_id") == job.implementation_id, "control implementation drifted")
            _require(fidelity.get("network_changed") is False, "control network changed")
            _require(fidelity.get("inference_safety_rule_added") is False, "control added rule")
            _require(actions.get("learned_collision_critic_present") is False, "control gained collision critic")
            _require(actions.get("collision_risk_coef") is None, "control gained risk coefficient")
            _require(not (root / "collision_label_diagnostics.json").exists(), "control emitted collision learner diagnostics")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        V49ProtocolError,
        json.JSONDecodeError,
    ) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


@contextmanager
def _row_path_view() -> Iterator[None]:
    original_parent_run = parent.run_directory
    original_scientific_run = scientific.run_directory
    parent.run_directory = run_directory
    scientific.run_directory = run_directory
    try:
        yield
    finally:
        parent.run_directory = original_parent_run
        scientific.run_directory = original_scientific_run


def _run_row(job: Job) -> dict[str, Any]:
    with _row_path_view():
        row = parent._run_row(job)
    root = run_directory(job)
    actions = _load_json(root / "action_diagnostics.json")
    row.update(
        {
            "learned_collision_critic_present": actions.get(
                "learned_collision_critic_present"
            ),
            "risk_adjusted_score_equation_match_rate": actions.get(
                "risk_adjusted_score_equation_match_rate"
            ),
            "inference_safety_rule_added": actions.get(
                "inference_safety_rule_added"
            ),
        }
    )
    if job.algorithm == V49_CANDIDATE:
        labels = _load_json(root / "collision_label_diagnostics.json")
        row.update(
            {
                "stored_collision_event_count": labels[
                    "stored_collision_event_count_including_source_duplicates"
                ],
                "sampled_collision_label_count": labels["sampled_label_count"],
                "sampled_positive_collision_label_rate": labels[
                    "sampled_positive_label_rate"
                ],
                "collision_label_diagnostics_sha256": _sha256(
                    root / "collision_label_diagnostics.json"
                ),
            }
        )
    return row


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return scientific.aggregate_rows(rows)


def summarize_stage(
    contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str]
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job) for job in jobs if accepted_run(job, hashes)]
    payload = {
        "schema_version": "topo-scene-v4.9.stage-results/v1",
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
        flat_keys = [
            key for key, value in rows[0].items() if not isinstance(value, (dict, list))
        ]
        with (root / "per_run.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in flat_keys})
    return payload


def _candidate_gate(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["development"]["per_run_gates"][row["scenario"]]
    checks = {
        "success_rate": float(row["success_rate"]) >= float(thresholds["success_rate_min"]),
        "collision_rate": float(row["collision_rate"]) <= float(thresholds["collision_rate_max"]),
        "off_route_rate": float(row["off_route_rate"]) <= float(thresholds["off_route_rate_max"]),
        "timeout_rate": float(row["timeout_rate"]) <= float(thresholds["timeout_rate_max"]),
        "learned_collision_critic_present": row.get("learned_collision_critic_present") is True,
        "risk_adjusted_score_equation": row.get("risk_adjusted_score_equation_match_rate") == 1.0,
        "no_inference_safety_rule": row.get("inference_safety_rule_added") is False,
        "selected_action_mask_feasible": row.get("selected_action_mask_feasible_rate") == 1.0,
    }
    if thresholds.get("observed_collision_supervision_required") is True:
        checks["observed_collision_supervision"] = int(
            row.get("stored_collision_event_count", 0)
        ) > 0 and float(row.get("sampled_positive_collision_label_rate") or 0.0) > 0.0
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "outcome_passed": all(
            checks[key]
            for key in ("success_rate", "collision_rate", "off_route_rate", "timeout_rate")
        ),
        "model_integrity_passed": all(
            value
            for key, value in checks.items()
            if key not in {"success_rate", "collision_rate", "off_route_rate", "timeout_rate"}
        ),
    }


def development_status(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, "development")
    states: dict[str, Any] = {}
    accepted_rows: list[dict[str, Any]] = []
    for job in jobs:
        accepted, reason = accepted_run_reason(job, hashes)
        entry: dict[str, Any] = {
            "job": job.name,
            "accepted": accepted,
            "reason": reason,
        }
        if accepted:
            row = _run_row(job)
            gate = _candidate_gate(row, contract)
            entry.update({"row": row, "gate": gate})
            accepted_rows.append(row)
        states[job.job_id] = entry

    all_accepted = len(accepted_rows) == len(jobs)
    per_run_pass = all(
        entry.get("gate", {}).get("passed", False) for entry in states.values()
    )
    aggregate_gate: dict[str, Any] | None = None
    if all_accepted:
        cross = [row for row in accepted_rows if row["scenario"] == "cross"]
        rules = contract["development"]["aggregate_cross_gate"]
        aggregate_checks = {
            "cross_success_rate": _mean(row["success_rate"] for row in cross)
            >= float(rules["success_rate_min"]),
            "cross_collision_rate": _mean(row["collision_rate"] for row in cross)
            <= float(rules["collision_rate_max"]),
            "positive_cross_cell": any(
                float(row["success_rate"])
                >= float(rules["positive_cell_success_rate_min"])
                and float(row["collision_rate"])
                <= float(rules["positive_cell_collision_rate_max"])
                for row in cross
            ),
        }
        aggregate_gate = {
            "passed": all(aggregate_checks.values()),
            "checks": aggregate_checks,
            "cross_success_rate": _mean(row["success_rate"] for row in cross),
            "cross_collision_rate": _mean(row["collision_rate"] for row in cross),
        }
    if not all_accepted:
        decision = "incomplete"
    elif per_run_pass and aggregate_gate and aggregate_gate["passed"]:
        decision = "pass"
    else:
        decision = "fail"
    return {
        "schema_version": "topo-scene-v4.9.development-status/v1",
        **hashes,
        "jobs": states,
        "execution_policy": "attempt_all_then_summarize",
        "all_jobs_accepted": all_accepted,
        "per_run_gates_passed": per_run_pass,
        "aggregate_cross_gate": aggregate_gate,
        "complete": decision in {"pass", "fail"},
        "decision": decision,
        "next_job": next(
            (job.job_id for job in jobs if not states[job.job_id]["accepted"]),
            None,
        ),
        "formal_test_unlocked": False,
        "formal_test_accessed": False,
    }


def write_development_decision(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    value = development_status(contract, digest, hashes)
    _write_json(stage_root("development") / "development_decision.json", value)
    return value


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["method"], row["scenario"], int(row["seed"])): row for row in rows
    }
    output = []
    for scenario in sorted({row["scenario"] for row in rows}):
        seeds = sorted(
            {int(row["seed"]) for row in rows if row["scenario"] == scenario}
        )
        for seed in seeds:
            candidate = indexed[("selected_v4_candidate", scenario, seed)]
            baseline = indexed[("temporal_graph", scenario, seed)]
            output.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "success_rate_delta": float(candidate["success_rate"])
                    - float(baseline["success_rate"]),
                    "collision_rate_delta": float(candidate["collision_rate"])
                    - float(baseline["collision_rate"]),
                }
            )
    return output


def compute_promotion_gate(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {
        (row["method"], row["scenario"]): row for row in summary["aggregate"]
    }
    scenarios = contract["promotion"]["scenarios"]
    success = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"]
        - aggregate[("temporal_graph", scenario)]["success_rate"]
        for scenario in scenarios
    }
    collision = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"]
        - aggregate[("temporal_graph", scenario)]["collision_rate"]
        for scenario in scenarios
    }
    paired = _paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {
        f"{row['scenario']}__seed{row['seed']}": _candidate_gate(row, contract)
        for row in rows
        if row["method"] == "selected_v4_candidate"
    }
    checks = {
        "candidate_development_guards_each_seed": all(
            gate["passed"] for gate in candidate_gates.values()
        ),
        "all_candidate_off_route_zero": all(
            aggregate[("selected_v4_candidate", scenario)]["off_route_rate"] <= 0.0
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
        "train_only_selection_applied": all(
            row["selected_checkpoint_kind"]
            in ("highest_training_success", "exact_final")
            for row in rows
        ),
    }
    receipt = {
        "schema_version": "topo-scene-v4.9.promotion-gate/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if all(checks.values()) else "fail",
        **hashes,
        "promotion_results": str(
            (stage_root("promotion") / "summary.json").resolve()
        ),
        "promotion_results_sha256": _sha256(
            stage_root("promotion") / "summary.json"
        ),
        "formal_algorithms": list(V49_FORMAL_ALGORITHMS),
        "checks": checks,
        "candidate_per_seed_gates": candidate_gates,
        "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision,
        "worst_paired_seed_success_delta": min(
            row["success_rate_delta"] for row in paired
        ),
        "macro_success_delta": _mean(success.values()),
        "paired_seed_deltas": paired,
        "formal_test_unlocked": all(checks.values()),
        "formal_test_accessed": False,
    }
    _write_json(DEFAULT_PROMOTION_GATE, receipt)
    return receipt


def compute_formal_decision(
    contract: dict[str, Any], digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    promotion = _load_json(DEFAULT_PROMOTION_GATE)
    _require(promotion.get("decision") == "pass", "formal test remains locked")
    summary = summarize_stage(contract, digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {
        (row["method"], row["scenario"]): row for row in summary["aggregate"]
    }
    success = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"]
        - aggregate[("temporal_graph", scenario)]["success_rate"]
        for scenario in SCENARIOS
    }
    collision = {
        scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"]
        - aggregate[("temporal_graph", scenario)]["collision_rate"]
        for scenario in SCENARIOS
    }
    acceptance = contract["formal_test"]["final_acceptance"]
    checks = {
        "every_scenario_success_noninferiority": all(
            delta >= float(acceptance["every_scenario_success_delta_min"])
            for delta in success.values()
        ),
        "every_scenario_collision_noninferiority": all(
            delta <= float(acceptance["every_scenario_collision_delta_max"])
            for delta in collision.values()
        ),
        "positive_effect": max(success.values())
        >= float(acceptance["positive_effect_any_success_delta_min"])
        or min(collision.values())
        <= float(acceptance["positive_effect_any_collision_delta_max"]),
    }
    formal = contract["formal_test"]
    statistics = scientific.paired_hierarchical_bootstrap(
        rows,
        candidate_method="selected_v4_candidate",
        baseline_method="temporal_graph",
        scenarios=SCENARIOS,
        resamples=int(formal["hierarchical_bootstrap_resamples"]),
        seed=int(formal["bootstrap_seed"]),
    )
    statistics["schema_version"] = (
        "topo-scene-v4.9.paired-hierarchical-bootstrap/v1"
    )
    stats_path = stage_root("formal") / "paired_hierarchical_bootstrap.json"
    _write_json(stats_path, statistics)
    decision = {
        "schema_version": "topo-scene-v4.9.formal-decision/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if all(checks.values()) else "fail",
        **hashes,
        "promotion_gate_sha256": _sha256(DEFAULT_PROMOTION_GATE),
        "formal_summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "statistics_sha256": _sha256(stats_path),
        "checks": checks,
        "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision,
        "paired_seed_deltas": _paired_deltas(rows),
        "formal_test_accessed": True,
    }
    _write_json(stage_root("formal") / "formal_decision.json", decision)
    return decision


def _assert_stage_can_run(hashes: dict[str, str], stage: str) -> None:
    validate_implementation_freeze()
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing engineering receipt")
    engineering = _load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(engineering.get("status") == "passed", "engineering gate did not pass")
    _require(
        engineering.get("implementation_freeze_sha256")
        == hashes["implementation_freeze_sha256"],
        "engineering freeze mismatch",
    )
    if stage == "promotion":
        decision = _load_json(
            stage_root("development") / "development_decision.json"
        )
        _require(decision.get("decision") == "pass", "development gate did not pass")
        _require(
            decision.get("implementation_freeze_sha256")
            == hashes["implementation_freeze_sha256"],
            "development freeze mismatch",
        )
    if stage == "formal":
        receipt = _load_json(DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        _require(
            receipt.get("formal_algorithms") == list(V49_FORMAL_ALGORITHMS),
            "formal method pair drifted",
        )
        for key, value in hashes.items():
            if key != "stage0_attribution_sha256":
                _require(receipt.get(key) == value, f"formal lock {key} mismatch")


def write_plan(
    contract: dict[str, Any],
    digest: str,
    stage: str,
    hashes: dict[str, str],
    *,
    device: str,
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    payload = {
        "schema_version": "topo-scene-v4.9.stage-plan/v1",
        "stage": stage,
        "device": device,
        "execution_policy": "attempt_all_then_summarize",
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
    _write_json(stage_root(stage) / "stage_plan.json", payload)
    return payload


def _select_jobs(jobs: list[Job], selector: str | None) -> list[Job]:
    if selector in (None, "all"):
        return jobs
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [
        job for job in jobs if job.job_id in requested or job.name in requested
    ]
    _require(
        len(selected) == len(requested),
        f"unknown or ambiguous job selector: {selector}",
    )
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
    _require(workers == 1, "v4.9 freezes one sequential GPU worker")
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
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command_for(job, hashes, device=device),
                cwd=PROJECT_ROOT,
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
            "log": str(log_path),
        }
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)

    execution = {
        "schema_version": "topo-scene-v4.9.execution/v1",
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
    if stage == "development":
        write_development_decision(contract, digest, hashes)
    failures = [row for row in attempts if row["status"] not in {"accepted", "already_accepted"}]
    return 1 if failures else 0


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
    development = development_status(contract, digest, hashes)
    promotion = (
        _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    )
    return {
        "schema_version": "topo-scene-v4.9.status/v1",
        **hashes,
        "stages": stages,
        "development_decision": development["decision"],
        "next_development_job": development["next_job"],
        "formal_test_unlocked": bool(
            promotion and promotion.get("decision") == "pass"
        ),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    root.add_argument("--stage0-results", type=Path, default=DEFAULT_STAGE0_RESULTS)
    root.add_argument(
        "--stage0-attribution", type=Path, default=DEFAULT_STAGE0_ATTRIBUTION
    )
    root.add_argument(
        "--failure-attribution", type=Path, default=DEFAULT_FAILURE_ATTRIBUTION
    )
    root.add_argument("--deep-attribution", type=Path, default=DEFAULT_DEEP_ATTRIBUTION)
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
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    if args.command == "validate":
        values = {
            **scientific.parent._validate_stage0(
                args.stage0_results.resolve(), args.stage0_attribution.resolve()
            ),
            **_validate_failure_attribution(
                args.failure_attribution.resolve(), args.deep_attribution.resolve()
            ),
        }
        print(
            json.dumps(
                {
                    "status": "valid",
                    "experiment_contract_sha256": digest,
                    **values,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    hashes = protocol_hashes(
        contract_path=contract_path,
        stage0_results=args.stage0_results.resolve(),
        stage0_attribution=args.stage0_attribution.resolve(),
        failure_attribution=args.failure_attribution.resolve(),
        deep_attribution=args.deep_attribution.resolve(),
        freeze_path=args.freeze.resolve(),
    )
    if args.command == "plan":
        value = write_plan(contract, digest, args.stage, hashes, device=args.device)
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
        value = summarize_stage(contract, digest, args.stage, hashes)
    elif args.command == "gate":
        if args.stage == "development":
            value = write_development_decision(contract, digest, hashes)
        elif args.stage == "promotion":
            value = compute_promotion_gate(contract, digest, hashes)
        else:
            value = compute_formal_decision(contract, digest, hashes)
    else:
        value = status_payload(contract, digest, hashes)
    print(json.dumps(_version_tree(value), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
