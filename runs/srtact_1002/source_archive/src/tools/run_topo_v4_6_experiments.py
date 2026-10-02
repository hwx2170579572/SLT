"""Hash-bound v4.6 development, promotion, and formal experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_6 import V46_ALGORITHMS, V46_IMPLEMENTATION_IDS
from tools.action_diagnostics_v4_5 import fusion_decoder_metrics
from tools.action_diagnostics_v4_6 import _target_integrity
from tools.checkpoint_decoder_selector_v4_6 import (
    CANDIDATE_DECODERS,
    FUSION_DECODER,
    PARENT_DECODER,
    TARGET_DECODER,
    select_deployment,
)
from tools.checkpoint_selector_v4_5_1 import terminal_event_evidence
from tools.topo_v2_statistics import paired_hierarchical_bootstrap


DEFAULT_CONTRACT = PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_6.yaml"
DEFAULT_STAGE0_RESULTS = PROJECT_ROOT / "results_topo_v4_dev" / "stage_0_offline_diagnosis" / "probe_results.json"
DEFAULT_STAGE0_ATTRIBUTION = PROJECT_ROOT / "results_topo_v4_dev" / "stage_0_offline_diagnosis" / "deep_attribution.md"
DEFAULT_FAILURE_ATTRIBUTION = PROJECT_ROOT / "results_topo_v4_5_dev" / "attribution" / "e1_failure" / "attribution_summary.json"
DEFAULT_DEEP_ATTRIBUTION = PROJECT_ROOT / "results_topo_v4_5_dev" / "attribution" / "e1_failure" / "deep_attribution.md"
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_6_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_6_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = PROJECT_ROOT / "results_topo_v4_6_promotion" / "promotion_gate.json"

LINEAGE_HASHES = {
    "parent_contract_sha256": "3c1f7f8bb2f2967f7559907386b58e67760c53dfa8e46bf581fc4824d186bb7c",
    "parent_implementation_freeze_sha256": "9ab0ccc71b33110ac526f5a9f89e8f504d0f3fc0ddafd6407c6ff36dc2414666",
    "parent_engineering_patch_freeze_sha256": "2dc362f52b80c03162d3b4875f4c667ebc976af5fd128b5c9680a0c15cea3cea",
    "parent_development_decision_sha256": "3e6e2d2c403db0784b7a2c67a8205bce47fc8e7cd7765f265a0039584540a0c3",
    "failure_attribution_sha256": "c32edd285d5a4a701b312bfcd57449969b2f0bf3367d50c2464f7373f09d925a",
    "deep_attribution_sha256": "070a6a911f46a466c520644da780dc14249a9335e9e418d505ffdaff87c292cf",
    "attribution_tool_sha256": "63b8ac7b739e42a30b7f4f3df9a5242c8bd5708fc08d429c0c8fcbeddc5d61fc",
}

SCENARIOS = ("left_turn", "cross", "roundabout_easy", "roundabout_medium", "roundabout", "carla")
STAGES = ("development", "promotion", "formal")
METHOD_ALGORITHMS = {
    "temporal_graph": "temporal_graph_v1_control",
    "selected_v4_candidate": "topo_v4_6_joint_checkpoint_decoder_selector",
}
METHOD_CODES = {"temporal_graph": "tg", "selected_v4_candidate": "cand"}
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


class V46ProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V46ProtocolError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _verify_zip(path: Path) -> None:
    _require(path.is_file(), f"missing checkpoint: {path}")
    with ZipFile(path, "r") as archive:
        bad = archive.testzip()
    _require(bad is None, f"ZIP CRC failure in {path}: {bad}")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.6 contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version") == "topo-scene-v4.6.joint-checkpoint-decoder-selector-contract/v1",
        "unexpected v4.6 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test must start locked")
    lineage = contract.get("lineage", {})
    for key, expected in LINEAGE_HASHES.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")
    _require(lineage.get("formal_test_accessed") is False, "lineage accessed formal test")
    change = contract.get("single_change", {})
    expected_change = {
        "name": "train_only_joint_checkpoint_decoder_selector",
        "checkpoint_candidates": ["highest_training_success", "exact_final"],
        "decoder_candidates": [TARGET_DECODER, FUSION_DECODER],
        "candidate_pairs": 4,
        "actor_non_keep_confidence_threshold": 0.90,
        "scenario_name_branching": False,
        "calibration_partition": "train",
        "calibration_episodes_per_pair": 12,
        "paired_calibration": True,
        "deployment_receipt_must_predate_validation": True,
        "training_changed": False,
        "stochastic_actor_changed": False,
        "reward_changed": False,
        "encoder_changed": False,
        "critic_training_changed": False,
        "entropy_changed": False,
        "decoder_definitions_changed": False,
        "fusion_threshold_changed": False,
    }
    for key, expected in expected_change.items():
        _require(change.get(key) == expected, f"single_change.{key} drifted")
    expected_order = [
        "maximize_success_count",
        "minimize_collision_count",
        "minimize_off_route_count",
        "minimize_timeout_count",
        "maximize_mean_return",
        "prefer_exact_final_on_complete_outcome_tie",
        "prefer_target_critic_on_complete_checkpoint_decoder_tie",
    ]
    _require(change.get("selection_order") == expected_order, "selection order drifted")
    cells = contract.get("development", {}).get("ordered_cells", [])
    expected_cells = [
        ("F1", "cross", 6, 72_000, 63_000),
        ("F2", "carla", 5, 73_000, 64_000),
        ("F3", "cross", 7, 74_000, 65_000),
        ("F4", "roundabout_medium", 2, 75_000, 66_000),
    ]
    observed = [
        (str(row["id"]), str(row["scenario"]), int(row["seed"]), int(row["calibration_seed_start"]), int(row["evaluation_seed_start"]))
        for row in cells
    ]
    _require(observed == expected_cells, "development cell matrix drifted")
    _require(all(int(row["raw_steps"]) == 20_000 for row in cells), "development raw steps drifted")
    development = contract["development"]
    _require(development.get("historical_or_post_hoc_gate_inputs_forbidden") is True, "post-hoc inputs are not forbidden")
    _require(int(development["evaluation_episodes"]) == 12, "development episodes drifted")
    _require(int(development["calibration_episodes_per_pair"]) == 12, "calibration episodes drifted")
    promotion = contract["promotion"]
    _require(len(promotion["methods"]) * len(promotion["scenarios"]) * len(promotion["seeds"]) == 12, "promotion matrix drifted")
    formal = contract["formal_test"]
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal lock drifted")
    _require(len(formal["methods"]) * len(formal["scenarios"]) * len(formal["seeds"]) == 120, "formal matrix drifted")
    return contract


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return _sha256(path.resolve())


def _validate_stage0(results: Path, attribution: Path) -> dict[str, str]:
    payload = _load_json(results)
    _require(payload.get("computed_from_real_rollout") is True, "stage-0 is not real")
    _require(payload.get("fabricated_values") is False, "stage-0 contains fabricated values")
    _require(_sha256(results) == "7fe77bb2d20e2e67005bdc8743ff4b84762f0ae75fdb8cfe43d2e4922906e1d1", "stage-0 hash drifted")
    _require(_sha256(attribution) == "6c0873cb6a50337859eaea7fa28ff0db76dbd977d5d680159bd219916ccd0402", "stage-0 attribution hash drifted")
    return {"stage0_results_sha256": _sha256(results), "stage0_attribution_sha256": _sha256(attribution)}


def _validate_failure_attribution(summary: Path, deep: Path) -> dict[str, str]:
    payload = _load_json(summary)
    _require(payload.get("schema_version") == "topo-scene-v4.5.e1-failure-attribution/v1", "failure attribution schema drifted")
    _require(payload.get("post_hoc_diagnostic_only") is True, "attribution is not post-hoc")
    _require(payload.get("counted_for_development_gate") is False, "attribution counted toward a gate")
    _require(payload.get("formal_test_accessed") is False, "attribution accessed formal test")
    recommendation = payload.get("recommended_single_change", {})
    _require(recommendation.get("version") == "v4.6", "attribution version drifted")
    _require(recommendation.get("name") == "train_only_joint_checkpoint_decoder_selector", "attribution recommendation drifted")
    _require(_sha256(summary) == LINEAGE_HASHES["failure_attribution_sha256"], "failure attribution hash drifted")
    _require(_sha256(deep) == LINEAGE_HASHES["deep_attribution_sha256"], "deep attribution hash drifted")
    return {"failure_attribution_sha256": _sha256(summary), "deep_attribution_sha256": _sha256(deep)}


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = _load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.6.implementation-freeze/v1", "invalid v4.6 freeze")
    _require(payload.get("formal_test_accessed") is False, "freeze accessed formal test")
    _require(payload.get("experiment_contract_sha256") == contract_sha256(), "freeze contract hash drifted")
    _require(payload.get("failure_attribution_sha256") == LINEAGE_HASHES["failure_attribution_sha256"], "freeze attribution hash drifted")
    _require(payload.get("preregistration_receipt_sha256") == _sha256(DEFAULT_PREREGISTRATION), "freeze preregistration hash drifted")
    scientific = payload.get("scientific_files")
    _require(isinstance(scientific, dict), "freeze scientific file map missing")
    for relative, expected in scientific.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"frozen source missing: {relative}")
        _require(_sha256(source) == expected, f"frozen source drifted: {relative}")
    _require(_canonical_sha256(scientific) == payload.get("scientific_content_sha256"), "freeze content digest invalid")
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
        **_validate_stage0(stage0_results.resolve(), stage0_attribution.resolve()),
        **_validate_failure_attribution(failure_attribution.resolve(), deep_attribution.resolve()),
        "preregistration_receipt_sha256": _sha256(DEFAULT_PREREGISTRATION),
    }
    validate_implementation_freeze(freeze_path.resolve())
    hashes["implementation_freeze_sha256"] = _sha256(freeze_path.resolve())
    return hashes


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
        prefix = self.job_id if self.stage == "development" else self.stage[:1].upper()
        return f"{prefix}__{METHOD_CODES[self.method]}__{SCENARIO_CODES[self.scenario]}__s{self.seed}__p{self.protocol_tag}"


def jobs_for_stage(contract: dict[str, Any], contract_digest: str, stage: str) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4.6 stage {stage!r}")
    tag = contract_digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        for specification in settings["ordered_cells"]:
            algorithm = METHOD_ALGORITHMS["selected_v4_candidate"]
            jobs.append(Job(
                stage=stage, job_id=str(specification["id"]), kind=str(specification["kind"]),
                method="selected_v4_candidate", algorithm=algorithm,
                implementation_id=V46_IMPLEMENTATION_IDS[algorithm], scenario=str(specification["scenario"]),
                seed=int(specification["seed"]), raw_steps=int(specification["raw_steps"]),
                calibration_episodes=int(settings["calibration_episodes_per_pair"]),
                calibration_seed_start=int(specification["calibration_seed_start"]),
                evaluation_episodes=int(settings["evaluation_episodes"]), evaluation_split=str(settings["evaluation_split"]),
                evaluation_seed_start=int(specification["evaluation_seed_start"]),
                role=str(specification.get("role", specification["kind"])), protocol_tag=tag,
            ))
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        calibration_base = 80_000 if stage == "promotion" else 100_000
        evaluation_base = 90_000 if stage == "promotion" else 200_000
        for declared_method in settings["methods"]:
            method = "temporal_graph" if declared_method == "temporal_graph" else "selected_v4_candidate"
            algorithm = METHOD_ALGORITHMS[method]
            for scenario in settings["scenarios"]:
                for seed in settings["seeds"]:
                    jobs.append(Job(
                        stage=stage, job_id="P" if stage == "promotion" else "F",
                        kind="fresh_training_with_train_only_deployment_selection", method=method,
                        algorithm=algorithm, implementation_id=V46_IMPLEMENTATION_IDS[algorithm],
                        scenario=str(scenario), seed=int(seed), raw_steps=int(settings["raw_training_steps"]),
                        calibration_episodes=int(settings["unique_calibration_traffic_episodes"]),
                        calibration_seed_start=calibration_base + int(seed) * 1_000,
                        evaluation_episodes=int(settings["evaluation_episodes"]), evaluation_split=str(settings["evaluation_split"]),
                        evaluation_seed_start=evaluation_base + int(seed) * 1_000,
                        role="paired_promotion" if stage == "promotion" else "untouched_formal_test", protocol_tag=tag,
                    ))
    expected = {"development": 4, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    _require(len({job.name for job in jobs}) == expected, f"duplicate {stage} run identity")
    return jobs


def stage_root(stage: str) -> Path:
    return {
        "development": PROJECT_ROOT / "results_topo_v4_6_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_6_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_6_formal",
    }[stage]


def run_directory(job: Job) -> Path:
    return stage_root(job.stage) / "runs" / job.name


def command_for(job: Job, hashes: dict[str, str], *, device: str, promotion_gate: Path = DEFAULT_PROMOTION_GATE) -> list[str]:
    command = [
        sys.executable, str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_6.py"),
        "--scenario", job.scenario, "--algo", job.algorithm, "--max-steps", str(job.raw_steps),
        "--learning-starts", "5000", "--checkpoint-freq", str(job.raw_steps), "--eval-freq", "0",
        "--eval-episodes", str(job.evaluation_episodes), "--evaluation-split", job.evaluation_split,
        "--evaluation-seed-start", str(job.evaluation_seed_start), "--calibration-episodes", str(job.calibration_episodes),
        "--calibration-seed-start", str(job.calibration_seed_start), "--seed", str(job.seed), "--device", device,
        "--batch-size", "32", "--learning-rate", "0.0001", "--discount", "0.99", "--buffer-size", "20000",
        "--action-repeat", "3", "--traffic-protocol", "frozen_60_20_20", "--episode-limit-profile", "source",
        "--output-dir", str((stage_root(job.stage) / "runs").resolve()), "--model-name", job.name,
        "--experiment-contract-sha256", hashes["experiment_contract_sha256"],
        "--stage0-results-sha256", hashes["stage0_results_sha256"],
        "--attribution-sha256", hashes["failure_attribution_sha256"],
        "--implementation-freeze-sha256", hashes["implementation_freeze_sha256"],
    ]
    if job.stage == "formal":
        command.extend(["--formal-unlock-receipt", str(promotion_gate.resolve())])
    return command


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), "action trace is empty or invalid")
    return rows


def _validate_selector_evidence(root: Path, job: Job) -> dict[str, Any]:
    receipt_path = root / "selector" / "receipt.json"
    receipt = _load_json(receipt_path)
    _require(receipt.get("schema_version") == "topo-scene-v4.6.checkpoint-decoder-selector-receipt/v1", "selector receipt schema drifted")
    _require(receipt.get("selection_partition") == "train", "selector partition drifted")
    _require(receipt.get("paired_calibration") is True, "selector calibration is not paired")
    _require(receipt.get("calibration_episodes") == job.calibration_episodes, "selector episode count drifted")
    _require(receipt.get("calibration_seed_start") == job.calibration_seed_start, "selector seed block drifted")
    _require(receipt.get("validation_used_for_selection") is False, "selector used validation")
    _require(receipt.get("formal_test_used_for_selection") is False, "selector used formal test")
    _require(receipt.get("selector_receipt_precedes_validation_environment") is True, "selector ordering flag is false")
    expected_decoders = set(CANDIDATE_DECODERS) if job.method == "selected_v4_candidate" else {PARENT_DECODER}
    expected_count = 4 if job.method == "selected_v4_candidate" else 2
    candidates = receipt.get("candidates", [])
    _require(len(candidates) == expected_count, "selector candidate count drifted")
    _require({row["deployment_decoder"] for row in candidates} == expected_decoders, "selector decoder set drifted")
    _require({row["checkpoint_kind"] for row in candidates} == {"highest_training_success", "exact_final"}, "selector checkpoint set drifted")
    signatures = []
    recompute = []
    fallback = bool(receipt.get("training_best_missing_fallback_used"))
    for row in candidates:
        kind = str(row["checkpoint_kind"])
        decoder = str(row["deployment_decoder"])
        short = "final" if fallback and kind == "highest_training_success" else {"highest_training_success": "best", "exact_final": "final"}[kind]
        pair_dir = root / "selector" / "cal" / short / decoder
        evaluation = _load_json(pair_dir / "evaluation.json")
        detailed = _load_json(pair_dir / "detailed.json")
        actions = _load_json(pair_dir / "actions.json")
        _require(evaluation == row["summary"], f"calibration summary drifted: {kind} x {decoder}")
        observed_signature = [
            {
                "episode": int(episode["episode"]),
                "seed": int(episode["seed"]),
                "traffic_variant": episode.get("traffic_variant"),
            }
            for episode in detailed["episode_records"]
        ]
        _require(
            observed_signature == row["episode_pair_signature"],
            f"calibration episode signature drifted: {kind} x {decoder}",
        )
        _require(_sha256(pair_dir / "detailed.json") == row["calibration_result_sha256"], "calibration detail hash drifted")
        _require(_sha256(pair_dir / "actions.json") == row["calibration_action_diagnostics_sha256"], "calibration action hash drifted")
        _require(actions.get("selected_deployment_decoder") == decoder, "calibration decoder trace drifted")
        _require(row.get("decoder_integrity_passed") is True, "calibration decoder integrity failed")
        terminal_event_evidence(evaluation, detailed["episode_records"])
        signature = [(int(item["episode"]), int(item["seed"]), item.get("traffic_variant")) for item in detailed["episode_records"]]
        signatures.append(signature)
        recompute.append({
            "checkpoint_kind": kind, "deployment_decoder": decoder,
            "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": row["checkpoint_sha256"],
            "traffic_partition": "train", "calibration_seed_start": job.calibration_seed_start,
            "summary": evaluation, "episode_records": detailed["episode_records"],
            "calibration_result_sha256": row["calibration_result_sha256"],
            "calibration_action_diagnostics_sha256": row["calibration_action_diagnostics_sha256"],
            "decoder_integrity": row["decoder_integrity"], "decoder_integrity_passed": True,
        })
    _require(all(signature == signatures[0] for signature in signatures), "calibration pairs are not traffic-paired")
    selected = select_deployment(recompute)
    _require(selected["selected_checkpoint_kind"] == receipt.get("selected_checkpoint_kind"), "selected checkpoint does not recompute")
    _require(selected["selected_deployment_decoder"] == receipt.get("selected_deployment_decoder"), "selected decoder does not recompute")
    source = Path(receipt["selected_source_checkpoint_path"])
    selected_model = root / "selected_model.zip"
    _verify_zip(source)
    _verify_zip(selected_model)
    _require(_sha256(source) == receipt.get("selected_source_checkpoint_sha256") == receipt.get("selected_checkpoint_sha256"), "selected source hash drifted")
    _require(_sha256(selected_model) == receipt.get("selected_model_sha256"), "selected model hash drifted")
    _require(receipt.get("policy_parameter_state_preserved") is True, "selected deployment changed tensors")
    _require(receipt.get("source_policy_parameter_state_sha256") == receipt.get("selected_model_parameter_state_sha256"), "selected tensor digest drifted")
    expected_policy = {
        TARGET_DECODER: "TargetCriticDecisionAlignedSACPolicyV43",
        FUSION_DECODER: "ConfidentActorFusionSACPolicyV45",
    }.get(receipt["selected_deployment_decoder"])
    if expected_policy:
        _require(receipt.get("selected_model_policy_class") == expected_policy, "selected policy class drifted")
    return receipt


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        _require(root.is_dir(), "run directory is absent")
        for filename in FULL_RUN_REQUIRED_FILES:
            _require((root / filename).is_file(), f"missing {filename}")
        arguments = _load_json(root / "arguments.json")
        metadata = _load_json(root / "method_metadata.json")
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        final_eval = _load_json(root / "final_evaluation.json")
        actions = _load_json(root / "action_diagnostics.json")
        training = _load_json(root / "training_diagnostics.json")
        checkpoints = _load_json(root / "checkpoint_audit.json")
        performance = _load_json(root / "performance_profile.json")
        selector = _validate_selector_evidence(root, job)
        for key in ("experiment_contract_sha256", "stage0_results_sha256", "attribution_sha256", "implementation_freeze_sha256"):
            source_key = "failure_attribution_sha256" if key == "attribution_sha256" else key
            _require(arguments.get(key) == hashes[source_key], f"arguments {key} mismatch")
            _require(detailed.get(key) == hashes[source_key], f"detailed {key} mismatch")
        _require(selector.get("experiment_contract_sha256") == hashes["experiment_contract_sha256"], "selector contract hash mismatch")
        _require(selector.get("attribution_sha256") == hashes["failure_attribution_sha256"], "selector attribution hash mismatch")
        _require(selector.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "selector freeze hash mismatch")
        requested = arguments["requested_raw_steps"]
        exact = {
            "algo": job.algorithm, "scenario": job.scenario, "seed": job.seed,
            "max_steps": job.raw_steps, "learning_starts": 5000, "batch_size": 32,
            "learning_rate": 0.0001, "discount": 0.99, "buffer_size": 20_000,
            "action_repeat": 3, "traffic_protocol": "frozen_60_20_20",
            "episode_limit_profile": "source", "evaluation_split": job.evaluation_split,
            "evaluation_seed_start": job.evaluation_seed_start, "eval_episodes": job.evaluation_episodes,
            "calibration_seed_start": job.calibration_seed_start, "calibration_episodes": job.calibration_episodes,
            "eval_freq": 0, "checkpoint_freq": job.raw_steps,
        }
        for key, expected in exact.items():
            observed = requested.get(key)
            _require(abs(float(observed) - expected) <= 1e-12 if isinstance(expected, float) else observed == expected, f"argument {key} mismatch")
        fidelity = arguments.get("implementation_fidelity", {})
        _require(fidelity.get("implementation_id") == job.implementation_id, "implementation id mismatch")
        _require(fidelity.get("single_change") == "train_only_joint_checkpoint_decoder_selector", "single change mismatch")
        _require(metadata.get("implementation_id") == job.implementation_id, "metadata id mismatch")
        _require(metadata.get("checkpoint_calibration_partition") == "train", "metadata selector partition drifted")
        _require(metadata.get("validation_used_for_checkpoint_selection") is False, "metadata admits validation selection")
        _require(detailed.get("schema_version") == "topo-scene-v4.6.detailed-evaluation/v1", "detailed schema drifted")
        _require(detailed.get("implementation_id") == job.implementation_id, "detailed implementation id mismatch")
        _require(detailed.get("algorithm") == job.algorithm and detailed.get("scenario") == job.scenario, "detailed method identity mismatch")
        _require(detailed.get("evaluation_split") == job.evaluation_split and detailed.get("evaluation_seed_start") == job.evaluation_seed_start, "evaluation partition/seed drifted")
        _require(detailed.get("trained_raw_steps") == job.raw_steps == detailed.get("collected_training_raw_steps"), "raw training clock drifted")
        _require(detailed.get("model_sha256") == selector.get("selected_model_sha256"), "detailed selected model hash mismatch")
        _require(detailed.get("selected_checkpoint_kind") == selector.get("selected_checkpoint_kind"), "detailed selected checkpoint mismatch")
        _require(detailed.get("selected_deployment_decoder") == selector.get("selected_deployment_decoder"), "detailed selected decoder mismatch")
        _require(detailed.get("selector_receipt_sha256") == _sha256(root / "selector" / "receipt.json"), "selector receipt binding mismatch")
        sealed = datetime.fromisoformat(str(detailed["selector_receipt_sealed_at_utc"]))
        constructed = datetime.fromisoformat(str(detailed["validation_environment_constructed_at_utc"]))
        _require(sealed <= constructed, "validation environment predates selector receipt")
        _require(final_eval == detailed["summary"], "final and detailed summaries differ")
        _require(final_eval.get("episodes") == job.evaluation_episodes, "final episode count mismatch")
        provenance = detailed.get("evaluation_provenance", {})
        _require(provenance.get("validated") is True and provenance.get("traffic_partition") == job.evaluation_split, "paper evaluation provenance drifted")
        _require(provenance.get("model_environment_spaces_match") is True, "space contract mismatch")
        _require(actions.get("schema_version") == "topo-scene-v4.6.action-diagnostics/v1", "action diagnostic schema drifted")
        _require(actions.get("aligned_with_paper_evaluation") is True, "action trace is not aligned")
        _require(actions.get("evaluation_seed_start") == job.evaluation_seed_start and actions.get("episodes") == job.evaluation_episodes, "action trace seed/episodes drifted")
        trace_path = root / "action_diagnostics_decisions.jsonl"
        _require(actions.get("trace", {}).get("sha256") == _sha256(trace_path), "action trace hash mismatch")
        decoder = selector["selected_deployment_decoder"]
        _require(actions.get("selected_deployment_decoder") == decoder, "action decoder differs from selector")
        trace_rows = _load_trace(trace_path)
        if job.method == "selected_v4_candidate":
            _require(actions.get("hybrid_exact_lateral_code_rate") == 1.0, "candidate emitted non-exact lane code")
            _require(actions.get("selected_decoder_records", 0) > 0, "selected decoder trace is empty")
            _require(actions.get("selected_decoder_exact_rule_match_rate") == 1.0, "selected decoder rule mismatch")
            _require(actions.get("selected_decoder_exact_action_match_rate") == 1.0, "selected decoder action mismatch")
            _require(actions.get("selected_action_mask_feasible_rate") == 1.0, "selected decoder chose infeasible action")
            if decoder == TARGET_DECODER:
                recomputed = _target_integrity(trace_rows)
                _require(recomputed["selected_decoder_exact_rule_match_rate"] == 1.0, "target rule does not recompute")
                _require(actions.get("exact_target_critic_argmax_rate") == 1.0, "target argmax drifted")
                _require(actions.get("target_keep_tie_rule_match_rate") == 1.0, "target keep tie drifted")
            elif decoder == FUSION_DECODER:
                recomputed = fusion_decoder_metrics(trace_rows)
                _require(recomputed["exact_fusion_rule_match_rate"] == 1.0, "fusion rule does not recompute")
                _require(actions.get("actor_non_keep_confidence_threshold") == 0.90, "fusion threshold drifted")
                _require(actions.get("actor_override_predicate_valid_rate") == 1.0, "fusion override predicate drifted")
                _require(actions.get("target_fallback_rule_match_rate") == 1.0, "fusion fallback drifted")
            else:
                raise V46ProtocolError("candidate selected an unsupported decoder")
            entropy = training.get("statistics", {}).get("hybrid/effective_lane_entropy_coefficient", {})
            _require(entropy.get("last") == 0.0, "effective lane entropy coefficient is nonzero")
        else:
            _require(decoder == PARENT_DECODER, "TemporalGraph decoder drifted")
        _require(training.get("raw_steps") == job.raw_steps and int(training.get("learner_updates", 0)) > 0, "training diagnostics drifted")
        _require(checkpoints.get("expected_checkpoint_count") == 1, "checkpoint count mismatch")
        checkpoint_rows = checkpoints.get("checkpoints", [])
        _require(len(checkpoint_rows) == 1 and checkpoint_rows[0].get("raw_step") == job.raw_steps and checkpoint_rows[0].get("zip_crc_ok") is True, "checkpoint audit drifted")
        _require(performance.get("raw_steps") == job.raw_steps, "performance clock drifted")
        _verify_zip(root / "final_model.zip")
        _verify_zip(root / "selected_model.zip")
        _require(all(_finite_tree(value) for value in (detailed, actions, training, performance)), "non-finite run artifact")
    except (OSError, KeyError, TypeError, ValueError, V46ProtocolError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


def _run_row(job: Job) -> dict[str, Any]:
    root = run_directory(job)
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    actions = _load_json(root / "action_diagnostics.json")
    performance = _load_json(root / "performance_profile.json")
    training = _load_json(root / "training_diagnostics.json")
    selector = _load_json(root / "selector" / "receipt.json")
    summary = detailed["summary"]
    return {
        "job_id": job.job_id, "kind": job.kind, "run_name": job.name,
        "run_directory": str(root.resolve()), "stage": job.stage, "role": job.role,
        "method": job.method, "algorithm": job.algorithm, "implementation_id": job.implementation_id,
        "scenario": job.scenario, "seed": job.seed, "raw_steps": job.raw_steps,
        "calibration_seed_start": job.calibration_seed_start, "calibration_episodes": job.calibration_episodes,
        "evaluation_seed_start": job.evaluation_seed_start, "evaluation_episodes": job.evaluation_episodes,
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selected_deployment_decoder": selector["selected_deployment_decoder"],
        "selected_source_checkpoint_sha256": selector["selected_source_checkpoint_sha256"],
        "selected_model_sha256": selector["selected_model_sha256"],
        "selector_receipt_sha256": _sha256(root / "selector" / "receipt.json"),
        "selector_candidate_count": selector["candidate_count"],
        "selector_mode": selector["selector_mode"],
        "selector_receipt_precedes_validation": True,
        "calibration_candidate_outcomes": {
            f"{row['checkpoint_kind']}__{row['deployment_decoder']}": row["summary"] for row in selector["candidates"]
        },
        **summary,
        "lane_command_negative_rate": actions["lane_command_rates"]["negative"],
        "lane_command_keep_rate": actions["lane_command_rates"]["keep"],
        "lane_command_positive_rate": actions["lane_command_rates"]["positive"],
        "lane_change_applied_rate": actions["lane_change_applied_rate"],
        "route_action_window_match_rate": actions["route_action_window_match_rate"],
        "hybrid_exact_lateral_code_rate": actions.get("hybrid_exact_lateral_code_rate"),
        "selected_decoder_records": actions.get("selected_decoder_records"),
        "selected_decoder_exact_rule_match_rate": actions.get("selected_decoder_exact_rule_match_rate"),
        "selected_decoder_exact_action_match_rate": actions.get("selected_decoder_exact_action_match_rate"),
        "selected_action_mask_feasible_rate": actions.get("selected_action_mask_feasible_rate"),
        "exact_target_critic_argmax_rate": actions.get("exact_target_critic_argmax_rate"),
        "target_keep_tie_rule_match_rate": actions.get("target_keep_tie_rule_match_rate"),
        "actor_non_keep_confidence_threshold": actions.get("actor_non_keep_confidence_threshold"),
        "actor_override_predicate_valid_rate": actions.get("actor_override_predicate_valid_rate"),
        "target_fallback_rule_match_rate": actions.get("target_fallback_rule_match_rate"),
        "learner_updates": training["learner_updates"],
        "training_wall_seconds": performance["end_to_end_training_wall_seconds"],
        "mean_inference_ms": performance["inference"]["mean_milliseconds_per_action"],
        "peak_gpu_memory_mb": performance["peak_gpu_memory_mb"],
        "model_plus_representation_parameters": performance["parameters"]["model_plus_representation"],
        "episode_records": detailed["episode_records"],
        "paper_evaluation_sha256": _sha256(root / "paper_evaluation_detailed.json"),
        "action_diagnostics_sha256": _sha256(root / "action_diagnostics.json"),
        "final_checkpoint_sha256": _sha256(root / "final_model.zip"),
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = ("success_rate", "collision_rate", "off_route_rate", "timeout_rate", "mean_return", "mean_decision_steps", "mean_raw_steps", "training_wall_seconds", "mean_inference_ms")
    groups = sorted({(row["method"], row["scenario"]) for row in rows})
    for method, scenario in groups:
        selected = [row for row in rows if row["method"] == method and row["scenario"] == scenario]
        output.append({
            "method": method, "scenario": scenario, "runs": len(selected),
            **{key: _mean(row[key] for row in selected) for key in keys},
            "selected_checkpoint_kinds": {kind: sum(row["selected_checkpoint_kind"] == kind for row in selected) for kind in ("highest_training_success", "exact_final")},
            "selected_decoders": {decoder: sum(row["selected_deployment_decoder"] == decoder for row in selected) for decoder in (*CANDIDATE_DECODERS, PARENT_DECODER)},
        })
    return output


def summarize_stage(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str]) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    rows = [_run_row(job) for job in jobs if accepted_run(job, hashes)]
    payload = {
        "schema_version": "topo-scene-v4.6.stage-results/v1", "stage": stage,
        "computed_from_real_runs": True, "fabricated_values": False, **hashes,
        "accepted_runs": len(rows), "expected_runs": len(jobs), "complete": len(rows) == len(jobs),
        "per_run": rows, "aggregate": aggregate_rows(rows),
    }
    root = stage_root(stage)
    _write_json(root / "summary.json", payload)
    _write_json(root / "per_run.json", rows)
    if rows:
        flat_keys = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        with (root / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in flat_keys})
    return payload


def _metric_gate(row: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, bool]:
    return {
        "success_rate": float(row["success_rate"]) >= float(thresholds["success_rate_min"]),
        "collision_rate": float(row["collision_rate"]) <= float(thresholds["collision_rate_max"]),
        "off_route_rate": float(row["off_route_rate"]) <= float(thresholds["off_route_rate_max"]),
        "timeout_rate": float(row["timeout_rate"]) <= float(thresholds["timeout_rate_max"]),
    }


def _candidate_gate(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["development"]["carla_outcome_gate" if row["scenario"] == "carla" else "non_carla_guard"]
    checks = _metric_gate(row, thresholds)
    mechanism = {
        "selected_decoder_records": int(row.get("selected_decoder_records") or 0) > 0,
        "selected_decoder_exact_rule_match_rate": row.get("selected_decoder_exact_rule_match_rate") == 1.0,
        "selected_decoder_exact_action_match_rate": row.get("selected_decoder_exact_action_match_rate") == 1.0,
        "selected_action_mask_feasible_rate": row.get("selected_action_mask_feasible_rate") == 1.0,
        "hybrid_exact_lateral_code_rate": row.get("hybrid_exact_lateral_code_rate") == 1.0,
    }
    if row.get("selected_deployment_decoder") == TARGET_DECODER:
        mechanism.update({
            "exact_target_critic_argmax_rate": row.get("exact_target_critic_argmax_rate") == 1.0,
            "target_keep_tie_rule_match_rate": row.get("target_keep_tie_rule_match_rate") == 1.0,
        })
    elif row.get("selected_deployment_decoder") == FUSION_DECODER:
        mechanism.update({
            "fusion_threshold": row.get("actor_non_keep_confidence_threshold") == 0.90,
            "actor_override_predicate_valid_rate": row.get("actor_override_predicate_valid_rate") == 1.0,
            "target_fallback_rule_match_rate": row.get("target_fallback_rule_match_rate") == 1.0,
        })
    else:
        mechanism["supported_selected_decoder"] = False
    selector = {
        "calibration_episode_count": row.get("calibration_episodes") == 12,
        "candidate_pair_count": row.get("selector_candidate_count") == 4,
        "selector_mode": row.get("selector_mode") == "joint_checkpoint_decoder",
        "selected_decoder": row.get("selected_deployment_decoder") in CANDIDATE_DECODERS,
        "receipt_predates_validation": row.get("selector_receipt_precedes_validation") is True,
    }
    return {
        "passed": all(checks.values()) and all(mechanism.values()) and all(selector.values()),
        "checks": {**checks, **mechanism, **selector},
        "outcome_passed": all(checks.values()),
        "mechanism_passed": all(mechanism.values()),
        "selector_passed": all(selector.values()),
    }


def development_status(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, "development")
    states: dict[str, Any] = {}
    next_job = None
    stopped_after = None
    decision = "incomplete"
    for job in jobs:
        accepted, reason = accepted_run_reason(job, hashes)
        entry: dict[str, Any] = {"job": job.name, "accepted": accepted, "reason": reason}
        if not accepted:
            states[job.job_id] = entry
            if stopped_after is None and next_job is None:
                next_job = job.job_id
            continue
        row = _run_row(job)
        gate = _candidate_gate(row, contract)
        entry.update({"row": row, "gate": gate})
        states[job.job_id] = entry
        if not gate["passed"] and stopped_after is None:
            stopped_after = job.job_id
            next_job = None
            decision = "fail"
    if stopped_after is None:
        accepted_ids = [job.job_id for job in jobs if states[job.job_id]["accepted"]]
        passed_ids = [job.job_id for job in jobs if states[job.job_id].get("gate", {}).get("passed")]
        if len(passed_ids) == len(jobs):
            decision = "pass"
            next_job = None
        else:
            next_job = next((job.job_id for job in jobs if job.job_id not in accepted_ids), None)
    prerequisites: dict[str, bool] = {}
    prior_pass = True
    for job in jobs:
        prerequisites[job.job_id] = prior_pass
        prior_pass = prior_pass and states[job.job_id].get("gate", {}).get("passed", False)
    return {
        "schema_version": "topo-scene-v4.6.development-status/v1", **hashes,
        "jobs": states, "prerequisites": prerequisites, "next_job": next_job,
        "stopped_after": stopped_after, "complete": decision == "pass", "decision": decision,
        "formal_test_unlocked": False,
    }


def write_development_decision(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    value = development_status(contract, digest, hashes)
    _write_json(stage_root("development") / "development_decision.json", value)
    return value


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["method"], row["scenario"], int(row["seed"])): row for row in rows}
    output = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if row["scenario"] == scenario}):
            candidate = indexed[("selected_v4_candidate", scenario, seed)]
            baseline = indexed[("temporal_graph", scenario, seed)]
            output.append({
                "scenario": scenario, "seed": seed,
                "success_rate_delta": float(candidate["success_rate"]) - float(baseline["success_rate"]),
                "collision_rate_delta": float(candidate["collision_rate"]) - float(baseline["collision_rate"]),
            })
    return output


def compute_promotion_gate(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    summary = summarize_stage(contract, digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {(row["method"], row["scenario"]): row for row in summary["aggregate"]}
    scenarios = contract["promotion"]["scenarios"]
    success = {scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"] - aggregate[("temporal_graph", scenario)]["success_rate"] for scenario in scenarios}
    collision = {scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"] - aggregate[("temporal_graph", scenario)]["collision_rate"] for scenario in scenarios}
    paired = _paired_deltas(rows)
    rules = contract["promotion"]["thresholds"]
    candidate_gates = {f"{row['scenario']}__seed{row['seed']}": _candidate_gate(row, contract) for row in rows if row["method"] == "selected_v4_candidate"}
    checks = {
        "candidate_development_guards_each_seed": all(gate["passed"] for gate in candidate_gates.values()),
        "all_candidate_off_route_zero": all(aggregate[("selected_v4_candidate", scenario)]["off_route_rate"] <= 0.0 for scenario in scenarios),
        "per_scenario_collision_noninferiority": all(delta <= float(rules["per_scenario_collision_noninferiority_margin"]) for delta in collision.values()),
        "worst_paired_seed_success_noninferiority": min(row["success_rate_delta"] for row in paired) >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success.values()) >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": max(success.values()) >= float(rules["positive_effect_success_gain"]) or min(collision.values()) <= -float(rules["positive_effect_collision_reduction"]),
        "train_only_selection_applied": all(row["selected_checkpoint_kind"] in ("highest_training_success", "exact_final") for row in rows),
    }
    receipt = {
        "schema_version": "topo-scene-v4.6.promotion-gate/v1", "computed_from_real_runs": True,
        "fabricated_values": False, "decision": "pass" if all(checks.values()) else "fail", **hashes,
        "promotion_results": str((stage_root("promotion") / "summary.json").resolve()),
        "promotion_results_sha256": _sha256(stage_root("promotion") / "summary.json"),
        "formal_algorithms": list(V46_ALGORITHMS), "checks": checks,
        "candidate_per_seed_gates": candidate_gates, "per_scenario_success_delta": success,
        "per_scenario_collision_delta": collision, "worst_paired_seed_success_delta": min(row["success_rate_delta"] for row in paired),
        "macro_success_delta": _mean(success.values()), "paired_seed_deltas": paired,
        "formal_test_unlocked": all(checks.values()),
    }
    _write_json(DEFAULT_PROMOTION_GATE, receipt)
    return receipt


def compute_formal_decision(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    promotion = _load_json(DEFAULT_PROMOTION_GATE)
    _require(promotion.get("decision") == "pass", "formal test remains locked")
    summary = summarize_stage(contract, digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {(row["method"], row["scenario"]): row for row in summary["aggregate"]}
    success = {scenario: aggregate[("selected_v4_candidate", scenario)]["success_rate"] - aggregate[("temporal_graph", scenario)]["success_rate"] for scenario in SCENARIOS}
    collision = {scenario: aggregate[("selected_v4_candidate", scenario)]["collision_rate"] - aggregate[("temporal_graph", scenario)]["collision_rate"] for scenario in SCENARIOS}
    acceptance = contract["formal_test"]["final_acceptance"]
    checks = {
        "every_scenario_success_noninferiority": all(delta >= float(acceptance["every_scenario_success_delta_min"]) for delta in success.values()),
        "every_scenario_collision_noninferiority": all(delta <= float(acceptance["every_scenario_collision_delta_max"]) for delta in collision.values()),
        "positive_effect": max(success.values()) >= float(acceptance["positive_effect_any_success_delta_min"]) or min(collision.values()) <= float(acceptance["positive_effect_any_collision_delta_max"]),
        "train_only_selection_applied": all(row["selected_checkpoint_kind"] in ("highest_training_success", "exact_final") for row in rows),
    }
    formal = contract["formal_test"]
    statistics = paired_hierarchical_bootstrap(rows, candidate_method="selected_v4_candidate", baseline_method="temporal_graph", scenarios=SCENARIOS, resamples=int(formal["hierarchical_bootstrap_resamples"]), seed=int(formal["bootstrap_seed"]))
    statistics["schema_version"] = "topo-scene-v4.6.paired-hierarchical-bootstrap/v1"
    stats_path = stage_root("formal") / "paired_hierarchical_bootstrap.json"
    _write_json(stats_path, statistics)
    decision = {
        "schema_version": "topo-scene-v4.6.formal-decision/v1", "computed_from_real_runs": True,
        "fabricated_values": False, "decision": "pass" if all(checks.values()) else "fail", **hashes,
        "promotion_gate_sha256": _sha256(DEFAULT_PROMOTION_GATE),
        "formal_summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "statistics_sha256": _sha256(stats_path), "checks": checks,
        "per_scenario_success_delta": success, "per_scenario_collision_delta": collision,
        "paired_seed_deltas": _paired_deltas(rows), "formal_test_accessed": True,
    }
    _write_json(stage_root("formal") / "formal_decision.json", decision)
    return decision


def _assert_stage_can_run(hashes: dict[str, str], stage: str) -> None:
    validate_implementation_freeze()
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing engineering receipt")
    engineering = _load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(engineering.get("status") == "passed", "engineering gate did not pass")
    _require(engineering.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "engineering freeze mismatch")
    if stage == "promotion":
        decision = _load_json(stage_root("development") / "development_decision.json")
        _require(decision.get("decision") == "pass", "development gate did not pass")
        _require(decision.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "development freeze mismatch")
    if stage == "formal":
        receipt = _load_json(DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        _require(receipt.get("formal_algorithms") == list(V46_ALGORITHMS), "formal method pair drifted")
        for key, value in hashes.items():
            if key != "stage0_attribution_sha256":
                _require(receipt.get(key) == value, f"formal test locked: {key} mismatch")


def write_plan(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, device: str) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, digest, stage)
    payload = {
        "schema_version": "topo-scene-v4.6.stage-plan/v1", "stage": stage, "device": device, **hashes,
        "jobs": [{**asdict(job), "name": job.name, "run_directory": str(run_directory(job).resolve()), "accepted": accepted_run(job, hashes), "command": command_for(job, hashes, device=device)} for job in jobs],
    }
    _write_json(stage_root(stage) / "stage_plan.json", payload)
    return payload


def _select_jobs(jobs: list[Job], selector: str | None, *, development_next: str | None) -> list[Job]:
    if selector in (None, "all"):
        return jobs
    if selector == "next":
        _require(development_next is not None, "no eligible next development job")
        return [next(job for job in jobs if job.job_id == development_next)]
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [job for job in jobs if job.job_id in requested or job.name in requested]
    _require(len(selected) == len(requested), f"unknown or ambiguous job selector: {selector}")
    return selected


def execute_jobs(contract: dict[str, Any], digest: str, stage: str, hashes: dict[str, str], *, selector: str | None, device: str, workers: int, dry_run: bool) -> int:
    _assert_stage_can_run(hashes, stage)
    jobs = jobs_for_stage(contract, digest, stage)
    next_job = development_status(contract, digest, hashes)["next_job"] if stage == "development" else None
    if stage == "development" and selector in (None, "next"):
        selector = "next"
    selected = _select_jobs(jobs, selector, development_next=next_job)
    if stage == "development":
        _require(len(selected) == 1 and selected[0].job_id == next_job, f"development protocol requires {next_job}")
    plan = write_plan(contract, digest, stage, hashes, device=device)
    if dry_run:
        print(json.dumps({"selected": [job.name for job in selected], "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    _require(workers > 0 and (stage != "development" or workers == 1), "invalid worker count")
    pending = []
    for job in selected:
        if accepted_run(job, hashes):
            continue
        directory = run_directory(job)
        if directory.exists():
            _, reason = accepted_run_reason(job, hashes)
            raise FileExistsError(f"refusing to overwrite incomplete immutable run {directory}: {reason}")
        pending.append(job)
    log_root = stage_root(stage) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []

    def run_one(job: Job):
        log_path = log_root / f"{job.name}.log"
        print(json.dumps({"job": job.name, "status": "running", "log": str(log_path)}, ensure_ascii=False), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command_for(job, hashes, device=device), cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
        accepted, reason = accepted_run_reason(job, hashes)
        if completed.returncode != 0 or not accepted:
            value = {"job": job.name, "returncode": completed.returncode, "acceptance_reason": reason, "log": str(log_path)}
            print(json.dumps({**value, "status": "failed"}, ensure_ascii=False), flush=True)
            return value
        print(json.dumps({"job": job.name, "status": "accepted"}, ensure_ascii=False), flush=True)
        return None

    if workers == 1:
        for job in pending:
            failure = run_one(job)
            if failure:
                failures.append(failure)
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, job) for job in pending]
            for future in as_completed(futures):
                failure = future.result()
                if failure:
                    failures.append(failure)
    execution = {
        "schema_version": "topo-scene-v4.6.execution/v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, **hashes, "selected": [job.name for job in selected],
        "attempted": [job.name for job in pending], "workers": workers, "failures": failures,
    }
    _write_json(stage_root(stage) / "last_execution.json", execution)
    summarize_stage(contract, digest, stage, hashes)
    if stage == "development":
        write_development_decision(contract, digest, hashes)
    return 1 if failures else 0


def status_payload(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    stages = {}
    for stage in STAGES:
        jobs = jobs_for_stage(contract, digest, stage)
        accepted = sum(accepted_run(job, hashes) for job in jobs)
        stages[stage] = {"accepted": accepted, "expected": len(jobs), "complete": accepted == len(jobs)}
    development = development_status(contract, digest, hashes)
    promotion = _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    return {
        "schema_version": "topo-scene-v4.6.status/v1", **hashes, "stages": stages,
        "development_decision": development["decision"], "next_development_job": development["next_job"],
        "formal_test_unlocked": bool(promotion and promotion.get("decision") == "pass"),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    root.add_argument("--stage0-results", type=Path, default=DEFAULT_STAGE0_RESULTS)
    root.add_argument("--stage0-attribution", type=Path, default=DEFAULT_STAGE0_ATTRIBUTION)
    root.add_argument("--failure-attribution", type=Path, default=DEFAULT_FAILURE_ATTRIBUTION)
    root.add_argument("--deep-attribution", type=Path, default=DEFAULT_DEEP_ATTRIBUTION)
    root.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    plan = sub.add_parser("plan"); plan.add_argument("--stage", choices=STAGES, required=True); plan.add_argument("--device", default="cuda")
    run = sub.add_parser("run"); run.add_argument("--stage", choices=STAGES, required=True); run.add_argument("--job", default=None); run.add_argument("--device", default="cuda"); run.add_argument("--workers", type=int, default=1); run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize"); summary.add_argument("--stage", choices=STAGES, required=True)
    gate = sub.add_parser("gate"); gate.add_argument("--stage", choices=STAGES, required=True)
    sub.add_parser("status")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    if args.command == "validate":
        values = {**_validate_stage0(args.stage0_results.resolve(), args.stage0_attribution.resolve()), **_validate_failure_attribution(args.failure_attribution.resolve(), args.deep_attribution.resolve())}
        print(json.dumps({"status": "valid", "experiment_contract_sha256": digest, **values}, ensure_ascii=False, indent=2))
        return 0
    hashes = protocol_hashes(contract_path=contract_path, stage0_results=args.stage0_results.resolve(), stage0_attribution=args.stage0_attribution.resolve(), failure_attribution=args.failure_attribution.resolve(), deep_attribution=args.deep_attribution.resolve(), freeze_path=args.freeze.resolve())
    if args.command == "plan":
        print(json.dumps(write_plan(contract, digest, args.stage, hashes, device=args.device), ensure_ascii=False, indent=2)); return 0
    if args.command == "run":
        return execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
    if args.command == "summarize":
        print(json.dumps(summarize_stage(contract, digest, args.stage, hashes), ensure_ascii=False, indent=2)); return 0
    if args.command == "gate":
        if args.stage == "development": value = write_development_decision(contract, digest, hashes)
        elif args.stage == "promotion": value = compute_promotion_gate(contract, digest, hashes)
        else: value = compute_formal_decision(contract, digest, hashes)
        print(json.dumps(value, ensure_ascii=False, indent=2)); return 0
    print(json.dumps(status_payload(contract, digest, hashes), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
