"""Hash-bound v4.3 target-critic development, promotion, and formal runner.

The runner keeps the untouched test split inaccessible until the complete
validation promotion matrix passes its preregistered gate.  It also treats an
existing but incomplete run directory as immutable evidence and refuses to
overwrite it.
"""

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

from configs.sb3_configs_v4_3 import V43_IMPLEMENTATION_IDS
from tools.action_diagnostics_v4_3 import _target_event_metrics
from tools.topo_v2_statistics import paired_hierarchical_bootstrap


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_3.yaml"
)
DEFAULT_STAGE0_RESULTS = (
    PROJECT_ROOT
    / "results_topo_v4_dev"
    / "stage_0_offline_diagnosis"
    / "probe_results.json"
)
DEFAULT_STAGE0_ATTRIBUTION = (
    PROJECT_ROOT
    / "results_topo_v4_dev"
    / "stage_0_offline_diagnosis"
    / "deep_attribution.md"
)
DEFAULT_FAILURE_ATTRIBUTION = (
    PROJECT_ROOT
    / "results_topo_v4_2_r1_dev"
    / "attribution"
    / "c1_failure"
    / "attribution_summary.json"
)
DEFAULT_REPLAY_SOURCES = {
    "R1": (
        PROJECT_ROOT
        / "results_topo_v4_2_r1_dev"
        / "development"
        / "runs"
        / "C1__cand__carla__s0__p331cfbcd"
    ),
    "R2": (
        PROJECT_ROOT
        / "results_topo_v4_2_dev"
        / "development"
        / "runs"
        / "F1__cand__carla__s1__p91f2cf40"
    ),
}
DEFAULT_REPLAY_MODEL_SHA256 = {
    "R1": "0fbb5acf3cac4a0acfbef5b9871216304c03bd96ad43d75d40c41500f36dc44a",
    "R2": "8ccf0e2f11c66a2451740d4671d934187f3992e3143898d5492c15877149fb7a",
}
DEFAULT_PARENT_DECISION = (
    PROJECT_ROOT
    / "results_topo_v4_2_r1_dev"
    / "development"
    / "development_decision.json"
)
DEFAULT_PARENT_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_2_r1.yaml"
)
DEFAULT_PARENT_FREEZE = (
    PROJECT_ROOT / "results_topo_v4_2_r1_dev" / "engineering" / "implementation_freeze.json"
)
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_3_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = PROJECT_ROOT / "results_topo_v4_3_promotion" / "promotion_gate.json"

SCENARIOS = (
    "left_turn",
    "cross",
    "roundabout_easy",
    "roundabout_medium",
    "roundabout",
    "carla",
)
STAGES = ("development", "promotion", "formal")
METHOD_ALGORITHMS = {
    "temporal_graph": "temporal_graph_v1_control",
    "selected_v4_candidate": "topo_v4_3_target_critic_decoder",
}
METHOD_CODES = {
    "temporal_graph": "tg",
    "selected_v4_candidate": "cand",
}
SCENARIO_CODES = {
    "left_turn": "lt",
    "cross": "cross",
    "roundabout_easy": "rae",
    "roundabout_medium": "ram",
    "roundabout": "ra",
    "carla": "carla",
}
SCIENTIFIC_IMPLEMENTATION_FILES = (
    "envs/sumo/decision_alignment_v4.py",
    "envs/sumo/paper_env_v4.py",
    "algos/sb3_torch/hybrid_policy_v4.py",
    "algos/sb3_torch/hybrid_policy_v4_3.py",
    "algos/sb3_torch/sac_v4_2.py",
    "algos/sb3_torch/sac_v4_3.py",
    "configs/sb3_configs_v4_3.py",
    "tools/action_diagnostics_v4_2.py",
    "tools/action_diagnostics_v4_3.py",
    "tools/train_sb3_v4_3.py",
    "tools/train_paper_sb3_sumo_v4_3.py",
    "tools/replay_v4_3_checkpoint.py",
    "tools/run_topo_v4_3_experiments.py",
    "tests_sb3_sumo/test_target_critic_decoder_v4_3.py",
    "tests_sb3_sumo/test_topo_v4_3_experiments.py",
    "experiments/topo_scene_v4/V4_3_TARGET_CRITIC_DECODER.md",
    "experiments/topo_scene_v4/experiment_contract_v4_3.yaml",
)
FULL_RUN_REQUIRED_FILES = (
    "arguments.json",
    "method_metadata.json",
    "training_diagnostics.json",
    "checkpoint_audit.json",
    "final_model.zip",
    "performance_profile.json",
    "final_evaluation.json",
    "action_diagnostics.json",
    "action_diagnostics_decisions.jsonl",
    "paper_evaluation_detailed.json",
)
REPLAY_REQUIRED_FILES = (
    "replay_arguments.json",
    "source_checkpoint_receipt.json",
    "method_metadata.json",
    "final_evaluation.json",
    "action_diagnostics.json",
    "action_diagnostics_decisions.jsonl",
    "paper_evaluation_detailed.json",
)


class V43ProtocolError(ValueError):
    """Raised when a v4.3 evidence or preregistration invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V43ProtocolError(message)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


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
    return hashlib.sha256(encoded).hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4 experiment contract must be a mapping")
    return value


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.3.target-critic-contract/v1",
        "unexpected v4.3 contract schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test must start locked")
    lineage = contract.get("lineage", {})
    expected_lineage = {
        "parent_contract_sha256": "331cfbcdd380656093abe1931bc7fd77ab297a5113853946f3cc40c133535d5d",
        "parent_implementation_freeze_sha256": "78982b246a728c4c153bb7003e62b4e1156436830840dd3d953f618e44fc20fe",
        "parent_development_decision_sha256": "a150dace99dfdcc0e2d81c9361138952cecca4556a34621ce0b2f1feaf54d61b",
        "failure_attribution_sha256": "d1b84cad36af100c6e446c4730e8893fe09c06f1333bbea2020aa0cfe0a62e5e",
        "formal_test_accessed": False,
    }
    for key, expected in expected_lineage.items():
        _require(lineage.get(key) == expected, f"lineage.{key} drifted")

    change = contract.get("single_change", {})
    _require(
        change.get("name") == "deterministic_target_critic_lane_decoder",
        "single change identity drifted",
    )
    _require(
        change.get("deterministic_lane_rule")
        == "argmax_feasible_min_target_twin_q",
        "deterministic decoder drifted",
    )
    _require(
        change.get("conditional_speed_source") == "actor_deterministic_per_lane",
        "conditional speed source drifted",
    )
    for key in (
        "stochastic_training_policy_changed",
        "checkpoint_selection_changed",
        "reward_changed",
        "encoder_changed",
        "critic_training_changed",
        "entropy_changed",
    ):
        _require(change.get(key) is False, f"single_change.{key} drifted")

    controls = contract.get("fixed_runtime", {})
    expected_controls = {
        "conda_environment": "llm_pipeline",
        "device": "cuda",
        "batch_size": 32,
        "learning_rate": 0.0001,
        "discount": 0.99,
        "replay_buffer_capacity": 20_000,
        "n_step": 4,
        "learning_starts_raw_steps": 5_000,
        "action_repeat": 3,
        "traffic_protocol": "frozen_60_20_20",
        "training_partition": "train",
        "development_partition": "validation",
        "formal_partition": "test",
        "episode_limit_profile": "source",
        "actor_encoder_detach": True,
        "slot_balance_coef": 0.01,
        "lane_entropy_scale": 0.0,
        "speed_target_entropy": -1.0,
        "final_checkpoint_rule": "exactly_at_requested_raw_steps",
    }
    for key, expected in expected_controls.items():
        _require(controls.get(key) == expected, f"fixed_runtime.{key} drifted")

    development = contract.get("development", {})
    _require(development.get("evaluation_episodes") == 12, "development eval drifted")
    _require(development.get("evaluation_split") == "validation", "development split drifted")
    ordered = development.get("ordered_cells")
    _require(isinstance(ordered, list) and len(ordered) == 5, "expected five development cells")
    _require([item.get("id") for item in ordered] == ["R1", "R2", "T1", "T2", "T3"], "development order drifted")
    _require(
        [item.get("kind") for item in ordered]
        == [
            "frozen_checkpoint_replay",
            "frozen_checkpoint_replay",
            "fresh_training",
            "fresh_training",
            "fresh_training",
        ],
        "development kinds drifted",
    )
    _require([item.get("scenario") for item in ordered] == ["carla", "carla", "carla", "cross", "cross"], "development scenarios drifted")
    _require([item.get("evaluation_seed_start") for item in ordered] == [46_000, 46_000, 47_000, 48_000, 49_000], "development traffic blocks drifted")
    _require([item.get("training_seed", item.get("seed")) for item in ordered] == [0, 1, 2, 0, 1], "development training seeds drifted")
    _require([item.get("raw_steps") for item in ordered[2:]] == [20_000, 20_000, 20_000], "development training clocks drifted")
    event_gate = development.get("carla_target_decoder_gate", {})
    _require(int(event_gate.get("episodes_with_non_keep_route_window_min")) == 1, "target event count gate drifted")
    _require(float(event_gate.get("route_event_applied_match_episode_rate_min")) == 0.50, "target applied gate drifted")
    _require(float(event_gate.get("applied_match_positive_target_q_margin_episode_rate_min")) == 0.50, "target margin gate drifted")
    _require(float(event_gate.get("exact_target_critic_argmax_rate")) == 1.0, "target argmax gate drifted")
    _require(development.get("no_gate_compensation") is True, "development gates must be noncompensatory")

    promotion = contract.get("promotion", {})
    _require(promotion.get("methods") == ["temporal_graph", "selected_v4_candidate"], "promotion methods drifted")
    _require(promotion.get("scenarios") == ["cross", "roundabout_medium", "carla"], "promotion scenarios drifted")
    _require(promotion.get("seeds") == [0, 1], "promotion seeds drifted")
    _require(promotion.get("raw_training_steps") == 50_000, "promotion steps drifted")
    _require(promotion.get("evaluation_episodes") == 30, "promotion eval drifted")
    _require(promotion.get("evaluation_split") == "validation", "promotion split drifted")
    _require(promotion.get("jobs") == 12, "promotion cardinality drifted")
    _require(
        promotion.get("evaluation_seed_start_formula")
        == "50000 + 1000 * training_seed",
        "promotion seed formula drifted",
    )

    formal = contract.get("formal_test", {})
    _require(formal.get("locked") is True and formal.get("accessed") is False, "formal declaration drifted")
    _require(formal.get("methods") == ["temporal_graph", "selected_v4_candidate"], "formal methods drifted")
    _require(tuple(formal.get("scenarios", ())) == SCENARIOS, "formal scenarios drifted")
    _require(formal.get("seeds") == list(range(10)), "formal seeds drifted")
    _require(formal.get("raw_training_steps") == 100_000, "formal steps drifted")
    _require(formal.get("evaluation_episodes") == 50, "formal eval drifted")
    _require(formal.get("evaluation_split") == "test", "formal split drifted")
    _require(formal.get("jobs") == 120, "formal cardinality drifted")
    _require(formal.get("hierarchical_bootstrap_resamples") == 10_000, "bootstrap drifted")
    _require(formal.get("bootstrap_seed") == 20_260_817, "bootstrap seed drifted")
    return contract


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return _sha256(path.resolve())


def _validate_stage0(path: Path, attribution_path: Path) -> dict[str, str]:
    _require(path.is_file(), f"missing stage-0 results: {path}")
    _require(attribution_path.is_file(), f"missing stage-0 attribution: {attribution_path}")
    result = _load_json(path)
    decision = result.get("decision")
    selected = (
        decision.get("selected_implementation_branch")
        if isinstance(decision, dict)
        else decision
    )
    _require(selected == "action_head_only", "stage 0 did not select action_head_only")
    coverage = result.get("coverage_gate", {})
    dataset = result.get("dataset", {})
    successful = coverage.get("successful_episodes", dataset.get("successful_episodes"))
    _require(int(successful) >= 4, "stage-0 successful episode coverage is insufficient")
    return {
        "stage0_results_sha256": _sha256(path),
        "stage0_attribution_sha256": _sha256(attribution_path),
    }


def _validate_failure_attribution(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"missing v4.3 failure attribution: {path}")
    payload = _load_json(path)
    _require(
        payload.get("schema_version")
        == "topo-scene-v4.2.r1.c1-failure-attribution/v1",
        "failure attribution schema drifted",
    )
    _require(payload.get("computed_from_real_runs") is True, "failure attribution is not run-backed")
    _require(payload.get("fabricated_values") is False, "failure attribution contains fabricated values")
    _require(payload.get("post_hoc_diagnostic_only") is True, "failure attribution status drifted")
    _require(payload.get("formal_test_accessed") is False, "failure attribution accessed formal test")
    _require(
        payload.get("failed_job") == "C1__cand__carla__s0__p331cfbcd",
        "failure attribution job identity drifted",
    )
    selected = payload.get("selected_next_hypothesis", {})
    _require(
        selected.get("version") == "v4_3_target_critic_lane_decoder",
        "selected hypothesis drifted",
    )
    _require(selected.get("training_changed") is False, "attribution changed training")
    _require(
        selected.get("requires_fresh_confirmation") is True,
        "fresh confirmation requirement drifted",
    )
    lineage = payload.get("sha256", {})
    _require(
        lineage.get("v4_2_r1_contract")
        == "331cfbcdd380656093abe1931bc7fd77ab297a5113853946f3cc40c133535d5d",
        "failure attribution parent contract drifted",
    )
    _require(
        lineage.get("v4_2_r1_implementation_freeze")
        == "78982b246a728c4c153bb7003e62b4e1156436830840dd3d953f618e44fc20fe",
        "failure attribution parent freeze drifted",
    )
    _require(
        lineage.get("v4_2_r1_development_decision")
        == "a150dace99dfdcc0e2d81c9361138952cecca4556a34621ce0b2f1feaf54d61b",
        "failure attribution parent decision drifted",
    )
    for replay_id, key in (("R1", "seed0_final_model"), ("R2", "seed1_final_model")):
        _require(
            lineage.get(key) == DEFAULT_REPLAY_MODEL_SHA256[replay_id],
            f"{replay_id} source hash drifted",
        )
    _require(
        _sha256(path)
        == "d1b84cad36af100c6e446c4730e8893fe09c06f1333bbea2020aa0cfe0a62e5e",
        "failure attribution file hash drifted",
    )
    return {"failure_attribution_sha256": _sha256(path)}


def _verify_zip(path: Path) -> None:
    with ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        _require(bad_member is None, f"ZIP CRC failure in {path}: {bad_member}")


def build_implementation_freeze(
    *,
    contract_path: Path,
    stage0_results: Path,
    stage0_attribution: Path,
    failure_attribution: Path,
    migration_log: Path,
    regression_log: Path,
    targeted_log: Path,
    smoke_dir: Path,
    replay_smoke_dir: Path,
    check_dirs: Iterable[Path],
    output: Path = DEFAULT_FREEZE,
) -> dict[str, Any]:
    """Freeze tested scientific files and immutable engineering evidence."""

    contract_path = contract_path.resolve()
    contract = validate_contract(load_contract(contract_path))
    del contract
    stage_hashes = {
        **_validate_stage0(stage0_results.resolve(), stage0_attribution.resolve()),
        **_validate_failure_attribution(failure_attribution.resolve()),
    }
    parent_evidence = {
        "parent_contract": (
            DEFAULT_PARENT_CONTRACT,
            "331cfbcdd380656093abe1931bc7fd77ab297a5113853946f3cc40c133535d5d",
        ),
        "parent_implementation_freeze": (
            DEFAULT_PARENT_FREEZE,
            "78982b246a728c4c153bb7003e62b4e1156436830840dd3d953f618e44fc20fe",
        ),
        "parent_development_decision": (
            DEFAULT_PARENT_DECISION,
            "a150dace99dfdcc0e2d81c9361138952cecca4556a34621ce0b2f1feaf54d61b",
        ),
    }
    lineage_receipts: dict[str, dict[str, str]] = {}
    for name, (path, expected) in parent_evidence.items():
        _require(path.is_file(), f"missing parent evidence: {path}")
        observed = _sha256(path)
        _require(observed == expected, f"{name} hash drifted")
        lineage_receipts[name] = {"path": str(path.resolve()), "sha256": observed}
    for replay_id, source_run in DEFAULT_REPLAY_SOURCES.items():
        source_model = source_run / "final_model.zip"
        _require(source_model.is_file(), f"missing {replay_id} source checkpoint")
        _verify_zip(source_model)
        observed = _sha256(source_model)
        _require(
            observed == DEFAULT_REPLAY_MODEL_SHA256[replay_id],
            f"{replay_id} source checkpoint hash drifted",
        )
        lineage_receipts[f"{replay_id.lower()}_source_model"] = {
            "path": str(source_model.resolve()),
            "sha256": observed,
        }
    evidence_paths = {
        "migration_log": migration_log.resolve(),
        "full_regression_log": regression_log.resolve(),
        "targeted_test_log": targeted_log.resolve(),
    }
    for name, path in evidence_paths.items():
        _require(path.is_file(), f"missing engineering evidence {name}: {path}")
    migration_text = evidence_paths["migration_log"].read_text(encoding="utf-8", errors="replace")
    regression_text = evidence_paths["full_regression_log"].read_text(encoding="utf-8", errors="replace")
    targeted_text = evidence_paths["targeted_test_log"].read_text(encoding="utf-8", errors="replace")
    try:
        migration_payload = json.loads(migration_text)
    except json.JSONDecodeError:
        migration_payload = None
    migration_passed = bool(
        isinstance(migration_payload, dict)
        and migration_payload.get("ok") is True
        and migration_payload.get("baseline_files") == 111
        and migration_payload.get("unchanged_files") == 111
        and migration_payload.get("missing") == []
        and migration_payload.get("changed") == []
    )
    _require(
        migration_passed or "111/111" in migration_text,
        "migration invariant did not report 111/111 unchanged",
    )
    _require("failed" not in regression_text.lower(), "full regression contains failures")
    _require("passed" in regression_text.lower(), "full regression has no passing summary")
    _require("failed" not in targeted_text.lower(), "targeted tests contain failures")
    _require("passed" in targeted_text.lower(), "targeted tests have no passing summary")

    smoke_dir = smoke_dir.resolve()
    _require(smoke_dir.is_dir(), f"missing end-to-end smoke directory: {smoke_dir}")
    for filename in FULL_RUN_REQUIRED_FILES:
        _require((smoke_dir / filename).is_file(), f"smoke artifact missing: {filename}")
    smoke_detailed = _load_json(smoke_dir / "paper_evaluation_detailed.json")
    smoke_training = _load_json(smoke_dir / "training_diagnostics.json")
    smoke_actions = _load_json(smoke_dir / "action_diagnostics.json")
    _require(smoke_detailed.get("evaluation_split") == "validation", "smoke used wrong split")
    _require(int(smoke_training.get("learner_updates", 0)) > 0, "smoke performed no updates")
    _require(smoke_actions.get("hybrid_exact_lateral_code_rate") == 1.0, "smoke emitted non-exact lane codes")
    _require(smoke_actions.get("exact_target_critic_argmax_rate") == 1.0, "smoke target decoder disagreed with emitted actions")
    _require(
        smoke_actions.get("deterministic_lane_decoder")
        == "argmax_feasible_min_target_twin_q",
        "smoke did not exercise the v4.3 target decoder",
    )
    _verify_zip(smoke_dir / "final_model.zip")

    replay_smoke_dir = replay_smoke_dir.resolve()
    _require(
        replay_smoke_dir.is_dir(),
        f"missing checkpoint replay smoke directory: {replay_smoke_dir}",
    )
    for filename in REPLAY_REQUIRED_FILES:
        _require(
            (replay_smoke_dir / filename).is_file(),
            f"replay smoke artifact missing: {filename}",
        )
    replay_arguments = _load_json(replay_smoke_dir / "replay_arguments.json")
    replay_source = _load_json(
        replay_smoke_dir / "source_checkpoint_receipt.json"
    )
    replay_detailed = _load_json(
        replay_smoke_dir / "paper_evaluation_detailed.json"
    )
    replay_actions = _load_json(replay_smoke_dir / "action_diagnostics.json")
    _require(
        replay_arguments.get("evaluation_split") == "validation",
        "replay smoke used wrong split",
    )
    _require(
        replay_source.get("source_model_sha256")
        == DEFAULT_REPLAY_MODEL_SHA256["R1"],
        "replay smoke used an unbound source checkpoint",
    )
    _require(
        replay_detailed.get("source_checkpoint_receipt_sha256")
        == _sha256(replay_smoke_dir / "source_checkpoint_receipt.json"),
        "replay smoke source receipt is not bound",
    )
    _require(
        replay_actions.get("exact_target_critic_argmax_rate") == 1.0,
        "replay smoke target decoder disagreed with emitted actions",
    )
    _require(
        replay_actions.get("trace", {}).get("sha256")
        == _sha256(replay_smoke_dir / "action_diagnostics_decisions.jsonl"),
        "replay smoke trace hash drifted",
    )

    checks: list[dict[str, Any]] = []
    observed_algorithms: set[str] = set()
    observed_candidate_scenarios: set[str] = set()
    for directory in check_dirs:
        directory = directory.resolve()
        result_path = directory / "check_result.json"
        _require(result_path.is_file(), f"missing check result: {result_path}")
        result = _load_json(result_path)
        _require(result.get("status") == "ok" and result.get("action_finite") is True, f"failed preflight: {directory}")
        algorithm = str(result.get("algorithm"))
        scenario = str(result.get("scenario"))
        observed_algorithms.add(algorithm)
        if algorithm == "topo_v4_3_target_critic_decoder":
            observed_candidate_scenarios.add(scenario)
            _require(result.get("exact_hybrid_lane_code") is True, "candidate preflight lane code is not exact")
        checks.append(
            {
                "directory": str(directory),
                "algorithm": algorithm,
                "scenario": scenario,
                "check_result_sha256": _sha256(result_path),
            }
        )
    _require(observed_algorithms == set(V43_IMPLEMENTATION_IDS), "preflight did not cover all v4.3 algorithms")
    _require({"carla", "cross"} <= observed_candidate_scenarios, "candidate preflight must cover CARLA and Cross")

    scientific_files: dict[str, str] = {}
    for relative in SCIENTIFIC_IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"scientific implementation file is missing: {path}")
        scientific_files[relative] = _sha256(path)
    content_sha = _canonical_sha256(scientific_files)
    payload = {
        "schema_version": "topo-scene-v4.3.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_after_first_scientific_job": True,
        "experiment_contract_sha256": _sha256(contract_path),
        **stage_hashes,
        "scientific_files": scientific_files,
        "scientific_content_sha256": content_sha,
        "engineering_evidence": {
            "lineage": lineage_receipts,
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in evidence_paths.items()
            },
            "end_to_end_smoke": {
                "path": str(smoke_dir),
                "paper_evaluation_sha256": _sha256(smoke_dir / "paper_evaluation_detailed.json"),
                "training_diagnostics_sha256": _sha256(smoke_dir / "training_diagnostics.json"),
                "action_diagnostics_sha256": _sha256(smoke_dir / "action_diagnostics.json"),
                "final_model_sha256": _sha256(smoke_dir / "final_model.zip"),
            },
            "checkpoint_replay_smoke": {
                "path": str(replay_smoke_dir),
                "source_checkpoint_receipt_sha256": _sha256(
                    replay_smoke_dir / "source_checkpoint_receipt.json"
                ),
                "paper_evaluation_sha256": _sha256(
                    replay_smoke_dir / "paper_evaluation_detailed.json"
                ),
                "action_diagnostics_sha256": _sha256(
                    replay_smoke_dir / "action_diagnostics.json"
                ),
            },
            "preflights": checks,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "conda_environment": "llm_pipeline",
        },
    }
    output = output.resolve()
    if output.exists():
        existing = _load_json(output)
        for key in (
            "schema_version",
            "experiment_contract_sha256",
            "stage0_results_sha256",
            "stage0_attribution_sha256",
            "failure_attribution_sha256",
            "scientific_files",
            "scientific_content_sha256",
            "engineering_evidence",
        ):
            _require(existing.get(key) == payload.get(key), f"existing implementation freeze drifted at {key}")
        return existing
    _write_json(output, payload)
    receipt = {
        "schema_version": "topo-scene-v4.3.engineering-receipt/v1",
        "status": "passed",
        "formal_test_accessed": False,
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "stage0_results_sha256": payload["stage0_results_sha256"],
        "failure_attribution_sha256": payload["failure_attribution_sha256"],
        "implementation_freeze": str(output),
        "implementation_freeze_sha256": _sha256(output),
        "scientific_content_sha256": content_sha,
        "checks": {
            "migration_invariant": "111/111",
            "targeted_tests": "passed",
            "full_regression": "passed",
            "real_sumo_preflights": "passed",
            "end_to_end_training_smoke": "passed",
            "checkpoint_replay_smoke": "passed",
        },
    }
    _write_json(DEFAULT_ENGINEERING_RECEIPT, receipt)
    return payload


def validate_implementation_freeze(
    path: Path = DEFAULT_FREEZE,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    stage0_results: Path = DEFAULT_STAGE0_RESULTS,
    failure_attribution: Path = DEFAULT_FAILURE_ATTRIBUTION,
) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing implementation freeze: {path}")
    payload = _load_json(path)
    _require(payload.get("schema_version") == "topo-scene-v4.3.implementation-freeze/v1", "invalid implementation freeze")
    _require(payload.get("experiment_contract_sha256") == _sha256(contract_path.resolve()), "freeze contract hash drifted")
    _require(payload.get("stage0_results_sha256") == _sha256(stage0_results.resolve()), "freeze stage-0 hash drifted")
    _require(payload.get("failure_attribution_sha256") == _sha256(failure_attribution.resolve()), "freeze failure attribution hash drifted")
    scientific_files = payload.get("scientific_files")
    _require(isinstance(scientific_files, dict), "freeze has no scientific file map")
    for relative, expected in scientific_files.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"frozen source missing: {source}")
        _require(_sha256(source) == expected, f"frozen source drifted: {relative}")
    _require(_canonical_sha256(scientific_files) == payload.get("scientific_content_sha256"), "freeze content digest is invalid")
    return payload


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
    evaluation_episodes: int
    evaluation_split: str
    evaluation_seed_start: int
    role: str
    protocol_tag: str

    @property
    def name(self) -> str:
        prefix = self.job_id if self.stage == "development" else self.stage[:1].upper()
        return (
            f"{prefix}__{METHOD_CODES[self.method]}__{SCENARIO_CODES[self.scenario]}__"
            f"s{self.seed}__p{self.protocol_tag}"
        )


def jobs_for_stage(
    contract: dict[str, Any], contract_digest: str, stage: str
) -> list[Job]:
    _require(stage in STAGES, f"unsupported v4 stage {stage!r}")
    tag = contract_digest[:8]
    jobs: list[Job] = []
    if stage == "development":
        settings = contract["development"]
        for specification in settings["ordered_cells"]:
            job_id = str(specification["id"])
            kind = str(specification["kind"])
            method = "selected_v4_candidate"
            scenario = str(specification["scenario"])
            seed = int(specification.get("training_seed", specification.get("seed")))
            role = str(specification.get("role", kind))
            algorithm = METHOD_ALGORITHMS[method]
            jobs.append(
                Job(
                    stage=stage,
                    job_id=job_id,
                    kind=kind,
                    method=method,
                    algorithm=algorithm,
                    implementation_id=V43_IMPLEMENTATION_IDS[algorithm],
                    scenario=scenario,
                    seed=seed,
                    raw_steps=int(specification.get("raw_steps", 0)),
                    evaluation_episodes=int(settings["evaluation_episodes"]),
                    evaluation_split=str(settings["evaluation_split"]),
                    evaluation_seed_start=int(specification["evaluation_seed_start"]),
                    role=role,
                    protocol_tag=tag,
                )
            )
    else:
        settings = contract["promotion" if stage == "promotion" else "formal_test"]
        methods = [
            "temporal_graph" if value == "temporal_graph" else "selected_v4_candidate"
            for value in settings["methods"]
        ]
        base_seed = 50_000 if stage == "promotion" else 100_000
        for method in methods:
            algorithm = METHOD_ALGORITHMS[method]
            for scenario in settings["scenarios"]:
                for seed in settings["seeds"]:
                    jobs.append(
                        Job(
                            stage=stage,
                            job_id=("P" if stage == "promotion" else "F"),
                            kind="fresh_training",
                            method=method,
                            algorithm=algorithm,
                            implementation_id=V43_IMPLEMENTATION_IDS[algorithm],
                            scenario=str(scenario),
                            seed=int(seed),
                            raw_steps=int(settings["raw_training_steps"]),
                            evaluation_episodes=int(settings["evaluation_episodes"]),
                            evaluation_split=str(settings["evaluation_split"]),
                            evaluation_seed_start=base_seed + int(seed) * 1_000,
                            role="paired_promotion" if stage == "promotion" else "untouched_formal_test",
                            protocol_tag=tag,
                        )
                    )
    names = [job.name for job in jobs]
    _require(len(names) == len(set(names)), f"duplicate run identity in {stage}")
    expected = {"development": 5, "promotion": 12, "formal": 120}[stage]
    _require(len(jobs) == expected, f"{stage} must contain {expected} jobs")
    return jobs


def stage_root(stage: str) -> Path:
    roots = {
        "development": PROJECT_ROOT / "results_topo_v4_3_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_3_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_3_formal",
    }
    return roots[stage]


def run_directory(job: Job) -> Path:
    collection = (
        "replays"
        if job.stage == "development" and job.kind == "frozen_checkpoint_replay"
        else "runs"
    )
    return stage_root(job.stage) / collection / job.name


def protocol_hashes(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    stage0_results: Path = DEFAULT_STAGE0_RESULTS,
    stage0_attribution: Path = DEFAULT_STAGE0_ATTRIBUTION,
    failure_attribution: Path = DEFAULT_FAILURE_ATTRIBUTION,
    freeze_path: Path = DEFAULT_FREEZE,
) -> dict[str, str]:
    validate_contract(load_contract(contract_path.resolve()))
    hashes = {
        "experiment_contract_sha256": _sha256(contract_path.resolve()),
        **_validate_stage0(stage0_results.resolve(), stage0_attribution.resolve()),
        **_validate_failure_attribution(failure_attribution.resolve()),
    }
    validate_implementation_freeze(
        freeze_path.resolve(),
        contract_path=contract_path.resolve(),
        stage0_results=stage0_results.resolve(),
        failure_attribution=failure_attribution.resolve(),
    )
    hashes["implementation_freeze_sha256"] = _sha256(freeze_path.resolve())
    return hashes


def command_for(
    job: Job,
    hashes: dict[str, str],
    *,
    device: str,
    promotion_gate: Path = DEFAULT_PROMOTION_GATE,
) -> list[str]:
    if job.kind == "frozen_checkpoint_replay":
        _require(job.job_id in DEFAULT_REPLAY_SOURCES, "unknown replay source")
        return [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "replay_v4_3_checkpoint.py"),
            "--source-run-dir",
            str(DEFAULT_REPLAY_SOURCES[job.job_id].resolve()),
            "--source-model-sha256",
            DEFAULT_REPLAY_MODEL_SHA256[job.job_id],
            "--source-training-seed",
            str(job.seed),
            "--scenario",
            job.scenario,
            "--evaluation-seed-start",
            str(job.evaluation_seed_start),
            "--evaluation-episodes",
            str(job.evaluation_episodes),
            "--action-repeat",
            "3",
            "--discount",
            "0.99",
            "--device",
            device,
            "--output-dir",
            str(run_directory(job).resolve()),
            "--experiment-contract-sha256",
            hashes["experiment_contract_sha256"],
            "--attribution-sha256",
            hashes["failure_attribution_sha256"],
            "--implementation-freeze-sha256",
            hashes["implementation_freeze_sha256"],
        ]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_3.py"),
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
        command.extend(["--formal-unlock-receipt", str(promotion_gate.resolve())])
    return command


def accepted_replay_reason(
    job: Job, hashes: dict[str, str]
) -> tuple[bool, str]:
    root = run_directory(job)
    try:
        _require(job.kind == "frozen_checkpoint_replay", "job is not a replay")
        _require(root.is_dir(), "replay directory is absent")
        for filename in REPLAY_REQUIRED_FILES:
            _require((root / filename).is_file(), f"missing {filename}")
        arguments = _load_json(root / "replay_arguments.json")
        source = _load_json(root / "source_checkpoint_receipt.json")
        metadata = _load_json(root / "method_metadata.json")
        detailed = _load_json(root / "paper_evaluation_detailed.json")
        final_eval = _load_json(root / "final_evaluation.json")
        actions = _load_json(root / "action_diagnostics.json")
        exact = {
            "implementation_id": job.implementation_id,
            "source_training_seed": job.seed,
            "scenario": job.scenario,
            "evaluation_split": job.evaluation_split,
            "evaluation_seed_start": job.evaluation_seed_start,
            "evaluation_episodes": job.evaluation_episodes,
            "traffic_partition": "validation",
            "traffic_protocol": "frozen_60_20_20",
            "episode_limit_profile": "source",
            "action_repeat": 3,
            "discount": 0.99,
            "experiment_contract_sha256": hashes["experiment_contract_sha256"],
            "attribution_sha256": hashes["failure_attribution_sha256"],
            "implementation_freeze_sha256": hashes[
                "implementation_freeze_sha256"
            ],
        }
        for key, expected in exact.items():
            observed = arguments.get(key)
            if isinstance(expected, float):
                _require(
                    abs(float(observed) - expected) <= 1e-12,
                    f"replay argument {key} mismatch",
                )
            else:
                _require(observed == expected, f"replay argument {key} mismatch")
        expected_source = DEFAULT_REPLAY_SOURCES[job.job_id] / "final_model.zip"
        _require(
            Path(source.get("source_model", "")).resolve()
            == expected_source.resolve(),
            "replay source path mismatch",
        )
        _require(
            source.get("source_model_sha256")
            == DEFAULT_REPLAY_MODEL_SHA256[job.job_id],
            "replay source checkpoint hash mismatch",
        )
        _require(expected_source.is_file(), "replay source checkpoint disappeared")
        _require(
            _sha256(expected_source) == DEFAULT_REPLAY_MODEL_SHA256[job.job_id],
            "replay source checkpoint drifted",
        )
        _verify_zip(expected_source)
        _require(
            detailed.get("source_checkpoint_receipt_sha256")
            == _sha256(root / "source_checkpoint_receipt.json"),
            "source receipt binding mismatch",
        )
        for key in (
            "experiment_contract_sha256",
            "attribution_sha256",
            "implementation_freeze_sha256",
        ):
            expected_key = (
                "failure_attribution_sha256"
                if key == "attribution_sha256"
                else key
            )
            _require(
                detailed.get(key) == hashes[expected_key],
                f"replay detailed {key} mismatch",
            )
        _require(metadata.get("implementation_id") == job.implementation_id, "replay metadata id mismatch")
        _require(metadata.get("weights_changed") is False, "replay changed checkpoint weights")
        _require(metadata.get("stochastic_training_policy_changed") is False, "replay changed stochastic policy")
        _require(detailed.get("implementation_id") == job.implementation_id, "replay implementation id mismatch")
        _require(detailed.get("algorithm") == job.algorithm, "replay algorithm mismatch")
        _require(detailed.get("scenario") == job.scenario, "replay scenario mismatch")
        _require(detailed.get("source_training_seed") == job.seed, "replay source seed mismatch")
        _require(detailed.get("evaluation_seed_start") == job.evaluation_seed_start, "replay traffic block mismatch")
        _require(detailed.get("evaluation_split") == "validation", "replay used non-validation split")
        _require(final_eval.get("episodes") == job.evaluation_episodes, "replay episode count mismatch")
        _require(detailed.get("summary", {}).get("episodes") == job.evaluation_episodes, "replay detailed episode count mismatch")
        provenance = detailed.get("evaluation_provenance", {})
        _require(provenance.get("validated") is True, "replay paper contract not validated")
        _require(provenance.get("traffic_partition") == "validation", "replay traffic partition mismatch")
        _require(provenance.get("model_environment_spaces_match") is True, "replay space contract mismatch")
        _require(actions.get("schema_version") == "topo-scene-v4.3.action-diagnostics/v1", "replay action diagnostics schema mismatch")
        _require(actions.get("aligned_with_paper_evaluation") is True, "replay action trace not aligned")
        _require(actions.get("evaluation_seed_start") == job.evaluation_seed_start, "replay trace seed mismatch")
        _require(actions.get("episodes") == job.evaluation_episodes, "replay trace episodes mismatch")
        _require(actions.get("hybrid_exact_lateral_code_rate") == 1.0, "replay emitted non-exact lane code")
        _require(actions.get("exact_target_critic_argmax_rate") == 1.0, "replay action disagreed with target decoder")
        _require(
            actions.get("deterministic_lane_decoder")
            == "argmax_feasible_min_target_twin_q",
            "replay decoder identity mismatch",
        )
        trace = actions.get("trace", {})
        _require(
            trace.get("sha256")
            == _sha256(root / "action_diagnostics_decisions.jsonl"),
            "replay trace hash mismatch",
        )
        _require(
            int(actions.get("target_decoder_records", -1))
            == int(trace.get("records", -2)),
            "replay decoder/trace cardinality mismatch",
        )
        _require(_finite_tree(detailed), "non-finite replay evaluation")
        _require(_finite_tree(actions), "non-finite replay diagnostics")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        V43ProtocolError,
        json.JSONDecodeError,
    ) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    if job.kind == "frozen_checkpoint_replay":
        return accepted_replay_reason(job, hashes)
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
        for key in (
            "experiment_contract_sha256",
            "stage0_results_sha256",
            "attribution_sha256",
            "implementation_freeze_sha256",
        ):
            hash_key = "failure_attribution_sha256" if key == "attribution_sha256" else key
            _require(arguments.get(key) == hashes[hash_key], f"arguments {key} mismatch")
            _require(detailed.get(key) == hashes[hash_key], f"detailed {key} mismatch")
        requested = arguments.get("requested_raw_steps", {})
        exact = {
            "algo": job.algorithm,
            "scenario": job.scenario,
            "seed": job.seed,
            "max_steps": job.raw_steps,
            "learning_starts": 5000,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "discount": 0.99,
            "buffer_size": 20_000,
            "action_repeat": 3,
            "traffic_protocol": "frozen_60_20_20",
            "episode_limit_profile": "source",
            "evaluation_split": job.evaluation_split,
            "evaluation_seed_start": job.evaluation_seed_start,
            "eval_episodes": job.evaluation_episodes,
            "eval_freq": 0,
            "checkpoint_freq": job.raw_steps,
        }
        for key, expected in exact.items():
            observed = requested.get(key)
            if isinstance(expected, float):
                _require(abs(float(observed) - expected) <= 1e-12, f"argument {key} mismatch")
            else:
                _require(observed == expected, f"argument {key} mismatch")
        fidelity = arguments.get("implementation_fidelity", {})
        _require(fidelity.get("implementation_id") == job.implementation_id, "implementation id mismatch")
        _require(metadata.get("implementation_id") == job.implementation_id, "metadata id mismatch")
        _require(detailed.get("implementation_id") == job.implementation_id, "evaluation id mismatch")
        _require(detailed.get("algorithm") == job.algorithm, "evaluation algorithm mismatch")
        _require(detailed.get("scenario") == job.scenario, "evaluation scenario mismatch")
        _require(detailed.get("evaluation_split") == job.evaluation_split, "evaluation split mismatch")
        _require(detailed.get("evaluation_seed_start") == job.evaluation_seed_start, "evaluation seed block mismatch")
        _require(detailed.get("trained_raw_steps") == job.raw_steps, "trained raw-step count mismatch")
        _require(detailed.get("collected_training_raw_steps") == job.raw_steps, "collected raw-step count mismatch")
        _require(final_eval.get("episodes") == job.evaluation_episodes, "final episode count mismatch")
        _require(detailed.get("summary", {}).get("episodes") == job.evaluation_episodes, "detailed episode count mismatch")
        provenance = detailed.get("evaluation_provenance", {})
        _require(provenance.get("validated") is True, "paper evaluation contract not validated")
        _require(provenance.get("traffic_partition") == job.evaluation_split, "traffic partition mismatch")
        _require(provenance.get("model_environment_spaces_match") is True, "space contract mismatch")
        _require(actions.get("aligned_with_paper_evaluation") is True, "action trace not aligned")
        _require(actions.get("evaluation_seed_start") == job.evaluation_seed_start, "action trace seed mismatch")
        _require(actions.get("episodes") == job.evaluation_episodes, "action trace episode mismatch")
        trace = actions.get("trace", {})
        _require(trace.get("sha256") == _sha256(root / "action_diagnostics_decisions.jsonl"), "action trace hash mismatch")
        if job.algorithm == "topo_v4_3_target_critic_decoder":
            _require(actions.get("hybrid_exact_lateral_code_rate") == 1.0, "candidate emitted non-exact lane code")
            _require(
                actions.get("schema_version")
                == "topo-scene-v4.3.action-diagnostics/v1",
                "candidate action diagnostics are not v4.3",
            )
            entropy = arguments.get("effective_method_hyperparameters", {})
            _require(entropy.get("lane_entropy_scale") == 0.0, "lane entropy scale drifted")
            _require(entropy.get("speed_target_entropy") == -1.0, "speed target entropy drifted")
            _require(metadata.get("lane_entropy_scale") == 0.0, "metadata lane entropy scale drifted")
            _require(metadata.get("speed_target_entropy") == -1.0, "metadata speed target entropy drifted")
            _require(
                metadata.get("deterministic_lane_decoder")
                == "argmax_feasible_min_target_twin_q",
                "metadata target decoder drifted",
            )
            _require(actions.get("exact_target_critic_argmax_rate") == 1.0, "candidate action disagreed with target decoder")
            entropy_stat = training.get("statistics", {}).get(
                "hybrid/effective_lane_entropy_coefficient", {}
            )
            _require(entropy_stat.get("last") == 0.0, "effective lane entropy coefficient is nonzero")
        _require(training.get("raw_steps") == job.raw_steps, "training diagnostics step mismatch")
        _require(int(training.get("learner_updates", 0)) > 0, "run performed no learner updates")
        _require(checkpoints.get("expected_checkpoint_count") == 1, "checkpoint count mismatch")
        checkpoint_rows = checkpoints.get("checkpoints", [])
        _require(len(checkpoint_rows) == 1, "checkpoint audit row count mismatch")
        _require(checkpoint_rows[0].get("raw_step") == job.raw_steps, "checkpoint raw clock mismatch")
        _require(checkpoint_rows[0].get("zip_crc_ok") is True, "checkpoint CRC not verified")
        _require(performance.get("raw_steps") == job.raw_steps, "performance step mismatch")
        _verify_zip(root / "final_model.zip")
        _require(_finite_tree(detailed), "non-finite detailed evaluation")
        _require(_finite_tree(actions), "non-finite action diagnostics")
        _require(_finite_tree(training), "non-finite training diagnostics")
        _require(_finite_tree(performance), "non-finite performance profile")
    except (OSError, KeyError, TypeError, ValueError, V43ProtocolError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "accepted"


def accepted_run(job: Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason(job, hashes)[0]


def target_decoder_metrics_from_path(path: Path) -> dict[str, Any]:
    """Recompute the v4.3 mechanism evidence directly from the bound trace."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"trace line {line_number} is not an object")
            records.append(value)
    _require(bool(records), "target decoder trace is empty")
    _require(
        all("target_critic_decoder" in row for row in records),
        "target decoder trace is incomplete",
    )
    return _target_event_metrics(records)


def _run_row(job: Job) -> dict[str, Any]:
    root = run_directory(job)
    detailed = _load_json(root / "paper_evaluation_detailed.json")
    actions = _load_json(root / "action_diagnostics.json")
    is_replay = job.kind == "frozen_checkpoint_replay"
    performance = (
        None if is_replay else _load_json(root / "performance_profile.json")
    )
    training = None if is_replay else _load_json(root / "training_diagnostics.json")
    summary = detailed["summary"]
    speed = actions.get("actual_speed_mps", {})
    target_events = (
        target_decoder_metrics_from_path(
            root / "action_diagnostics_decisions.jsonl"
        )
        if job.algorithm == "topo_v4_3_target_critic_decoder"
        else {
            "target_decoder_records": 0,
            "exact_target_critic_argmax_rate": None,
            "target_route_event_episode_count": 0,
            "target_route_event_applied_match_episode_rate": None,
            "applied_match_positive_target_q_margin_episode_rate": None,
            "target_route_minus_keep_q_margin": {
                "count": 0,
                "mean": None,
                "minimum": None,
                "maximum": None,
            },
            "target_route_event_per_episode": [],
        }
    )
    if job.algorithm == "topo_v4_3_target_critic_decoder":
        for key in (
            "target_decoder_records",
            "exact_target_critic_argmax_rate",
            "target_route_event_episode_count",
            "target_route_event_applied_match_episode_rate",
            "applied_match_positive_target_q_margin_episode_rate",
        ):
            _require(
                actions.get(key) == target_events.get(key),
                f"stored target decoder metric drifted: {key}",
            )
    target_margin = target_events["target_route_minus_keep_q_margin"]
    source_model_sha = (
        _load_json(root / "source_checkpoint_receipt.json")[
            "source_model_sha256"
        ]
        if is_replay
        else _sha256(root / "final_model.zip")
    )
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "run_name": job.name,
        "run_directory": str(root.resolve()),
        "stage": job.stage,
        "role": job.role,
        "method": job.method,
        "algorithm": job.algorithm,
        "implementation_id": job.implementation_id,
        "scenario": job.scenario,
        "seed": job.seed,
        "raw_steps": job.raw_steps,
        "evaluation_seed_start": job.evaluation_seed_start,
        "evaluation_episodes": job.evaluation_episodes,
        "success_rate": float(summary["success_rate"]),
        "collision_rate": float(summary["collision_rate"]),
        "off_route_rate": float(summary["off_route_rate"]),
        "timeout_rate": float(summary["timeout_rate"]),
        "mean_return": float(summary["mean_return"]),
        "mean_decision_steps": float(summary["mean_decision_steps"]),
        "mean_raw_steps": float(summary["mean_raw_steps"]),
        "lane_command_negative_rate": float(actions["lane_command_rates"]["negative"]),
        "lane_command_keep_rate": float(actions["lane_command_rates"]["keep"]),
        "lane_command_positive_rate": float(actions["lane_command_rates"]["positive"]),
        "lane_change_applied_rate": float(actions["lane_change_applied_rate"]),
        "predicted_lane_action_feasible_rate": float(actions["predicted_lane_action_feasible_rate"]),
        "valid_route_intent_match_rate": actions.get("valid_route_intent_match_rate"),
        "route_action_window_samples": int(actions.get("route_action_window_samples", 0)),
        "route_action_window_match_rate": actions.get("route_action_window_match_rate"),
        "route_action_window_applied_match_rate": actions.get("route_action_window_applied_match_rate"),
        "target_decoder_records": int(target_events["target_decoder_records"]),
        "exact_target_critic_argmax_rate": target_events[
            "exact_target_critic_argmax_rate"
        ],
        "target_route_event_episode_count": int(
            target_events["target_route_event_episode_count"]
        ),
        "target_route_event_applied_match_episode_rate": target_events[
            "target_route_event_applied_match_episode_rate"
        ],
        "applied_match_positive_target_q_margin_episode_rate": target_events[
            "applied_match_positive_target_q_margin_episode_rate"
        ],
        "target_route_minus_keep_q_margin_count": int(
            target_margin.get("count", 0)
        ),
        "target_route_minus_keep_q_margin_mean": target_margin.get("mean"),
        "target_route_minus_keep_q_margin_minimum": target_margin.get("minimum"),
        "target_route_minus_keep_q_margin_maximum": target_margin.get("maximum"),
        "target_route_event_per_episode": target_events[
            "target_route_event_per_episode"
        ],
        "hybrid_exact_lateral_code_rate": actions.get("hybrid_exact_lateral_code_rate"),
        "mean_speed_mps": speed.get("mean"),
        "learner_updates": 0 if training is None else int(training["learner_updates"]),
        "training_wall_seconds": (
            None
            if performance is None
            else float(performance["end_to_end_training_wall_seconds"])
        ),
        "evaluation_wall_seconds": detailed.get("evaluation_wall_seconds"),
        "mean_inference_ms": (
            None
            if performance is None
            else float(performance["inference"]["mean_milliseconds_per_action"])
        ),
        "peak_gpu_memory_mb": (
            None if performance is None else performance.get("peak_gpu_memory_mb")
        ),
        "model_plus_representation_parameters": (
            None
            if performance is None
            else int(performance["parameters"]["model_plus_representation"])
        ),
        "episode_records": detailed["episode_records"],
        "paper_evaluation_sha256": _sha256(root / "paper_evaluation_detailed.json"),
        "action_diagnostics_sha256": _sha256(root / "action_diagnostics.json"),
        "final_model_sha256": source_model_sha,
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    _require(bool(items), "cannot average an empty sequence")
    return sum(items) / len(items)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
        "mean_return",
        "lane_command_keep_rate",
        "lane_change_applied_rate",
        "exact_target_critic_argmax_rate",
        "target_route_event_applied_match_episode_rate",
        "applied_match_positive_target_q_margin_episode_rate",
        "target_route_minus_keep_q_margin_mean",
        "target_route_minus_keep_q_margin_minimum",
        "mean_speed_mps",
        "training_wall_seconds",
        "mean_inference_ms",
    )
    keys = sorted({(row["method"], row["scenario"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method, scenario in keys:
        selected = [row for row in rows if row["method"] == method and row["scenario"] == scenario]
        item: dict[str, Any] = {
            "method": method,
            "scenario": scenario,
            "seeds": sorted(int(row["seed"]) for row in selected),
            "runs": len(selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            item[metric] = _mean(values) if values else None
        route_values = [
            float(row["route_action_window_match_rate"])
            for row in selected
            if row.get("route_action_window_match_rate") is not None
            and int(row.get("route_action_window_samples", 0)) > 0
        ]
        item["route_action_window_match_rate"] = _mean(route_values) if route_values else None
        parameter_values = [
            int(row["model_plus_representation_parameters"])
            for row in selected
            if row.get("model_plus_representation_parameters") is not None
        ]
        item["model_plus_representation_parameters"] = (
            max(parameter_values) if parameter_values else None
        )
        output.append(item)
    return output


def summarize_stage(
    contract: dict[str, Any], contract_digest: str, stage: str, hashes: dict[str, str]
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, contract_digest, stage)
    accepted: list[Job] = []
    rejected: list[dict[str, str]] = []
    for job in jobs:
        ok, reason = accepted_run_reason(job, hashes)
        if ok:
            accepted.append(job)
        else:
            rejected.append({"job": job.name, "reason": reason})
    rows = [_run_row(job) for job in accepted]
    payload = {
        "schema_version": "topo-scene-v4.3.stage-results/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "stage": stage,
        **hashes,
        "expected_jobs": len(jobs),
        "accepted_jobs": len(accepted),
        "complete": len(accepted) == len(jobs),
        "rejected_or_missing": rejected,
        "per_run": rows,
        "aggregate": aggregate_rows(rows) if rows else [],
    }
    root = stage_root(stage)
    _write_json(root / "summary.json", payload)
    if rows:
        scalar_keys = [
            key for key, value in rows[0].items() if not isinstance(value, (dict, list))
        ]
        with (root / "per_run.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return payload


def _metric_gate(row: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "success_rate": float(row["success_rate"]) >= float(thresholds["success_rate_min"]),
        "collision_rate": float(row["collision_rate"]) <= float(thresholds["collision_rate_max"]),
        "off_route_rate": float(row["off_route_rate"]) <= float(thresholds["off_route_rate_max"]),
        "timeout_rate": float(row["timeout_rate"]) <= float(thresholds["timeout_rate_max"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _carla_target_gate(
    row: dict[str, Any], development: dict[str, Any]
) -> dict[str, Any]:
    outcome = _metric_gate(row, development["carla_outcome_gate"])
    thresholds = development["carla_target_decoder_gate"]
    event_count = int(row["target_route_event_episode_count"])
    applied_rate = row.get("target_route_event_applied_match_episode_rate")
    positive_rate = row.get(
        "applied_match_positive_target_q_margin_episode_rate"
    )
    exact_rate = row.get("exact_target_critic_argmax_rate")
    mechanism_checks = {
        "episodes_with_non_keep_route_window": event_count
        >= int(thresholds["episodes_with_non_keep_route_window_min"]),
        "route_event_applied_match_episode_rate": applied_rate is not None
        and float(applied_rate)
        >= float(thresholds["route_event_applied_match_episode_rate_min"]),
        "applied_match_positive_target_q_margin_episode_rate": positive_rate
        is not None
        and float(positive_rate)
        >= float(
            thresholds[
                "applied_match_positive_target_q_margin_episode_rate_min"
            ]
        ),
        "exact_target_critic_argmax_rate": exact_rate is not None
        and float(exact_rate)
        == float(thresholds["exact_target_critic_argmax_rate"]),
    }
    checks = {**outcome["checks"], **mechanism_checks}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "outcome_passed": outcome["passed"],
        "mechanism_passed": all(mechanism_checks.values()),
    }


def development_status(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, contract_digest, "development")
    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        ok, reason = accepted_run_reason(job, hashes)
        item: dict[str, Any] = {"job": job.name, "accepted": ok, "reason": reason}
        if ok:
            row = _run_row(job)
            item["row"] = row
            if job.scenario == "carla":
                item["gate"] = _carla_target_gate(row, contract["development"])
            else:
                item["gate"] = _metric_gate(row, contract["development"]["cross_guard"])
        by_id[job.job_id] = item
    prerequisites = {
        "R1": True,
        "R2": bool(by_id["R1"].get("gate", {}).get("passed")),
        "T1": bool(by_id["R2"].get("gate", {}).get("passed")),
        "T2": bool(by_id["T1"].get("gate", {}).get("passed")),
        "T3": bool(by_id["T2"].get("gate", {}).get("passed")),
    }
    order = ("R1", "R2", "T1", "T2", "T3")
    next_job = next(
        (
            job_id
            for job_id in order
            if prerequisites[job_id] and not by_id[job_id]["accepted"]
        ),
        None,
    )
    stopped_after = next(
        (
            job_id
            for job_id in order
            if by_id[job_id].get("accepted")
            and not by_id[job_id].get("gate", {}).get("passed", False)
        ),
        None,
    )
    complete = all(item["accepted"] for item in by_id.values())
    passed = complete and all(
        by_id[job_id].get("gate", {}).get("passed", False) for job_id in order
    )
    return {
        "schema_version": "topo-scene-v4.3.development-status/v1",
        **hashes,
        "jobs": by_id,
        "prerequisites": prerequisites,
        "next_job": next_job,
        "stopped_after": stopped_after,
        "complete": complete,
        "decision": "pass" if passed else ("fail" if stopped_after else "incomplete"),
        "parent_control": {
            "implementation": "topo_v4_2_factorized_entropy",
            "attribution_sha256": hashes["failure_attribution_sha256"],
            "interpretation": (
                "Post-hoc v4.2 decoder replays proposed v4.3 only; R1/R2 use "
                "fresh validation traffic, and T1 supplies fresh-training confirmation."
            ),
        },
    }


def write_development_decision(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    status = development_status(contract, contract_digest, hashes)
    _write_json(stage_root("development") / "development_decision.json", status)
    return status


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], row["scenario"], int(row["seed"])): row for row in rows
    }
    output: list[dict[str, Any]] = []
    scenarios = sorted({row["scenario"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for scenario in scenarios:
        for seed in seeds:
            candidate = by_key.get(("selected_v4_candidate", scenario, seed))
            baseline = by_key.get(("temporal_graph", scenario, seed))
            if candidate is None or baseline is None:
                continue
            output.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    **{
                        f"{metric}_delta": float(candidate[metric]) - float(baseline[metric])
                        for metric in ("success_rate", "collision_rate", "off_route_rate", "timeout_rate", "mean_return")
                    },
                }
            )
    return output


def compute_promotion_gate(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    summary = summarize_stage(contract, contract_digest, "promotion", hashes)
    _require(summary["complete"] is True, "promotion matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {
        (row["method"], row["scenario"]): row for row in summary["aggregate"]
    }
    paired = _paired_deltas(rows)
    scenarios = contract["promotion"]["scenarios"]
    success_deltas = {
        scenario: float(aggregate[("selected_v4_candidate", scenario)]["success_rate"])
        - float(aggregate[("temporal_graph", scenario)]["success_rate"])
        for scenario in scenarios
    }
    collision_deltas = {
        scenario: float(aggregate[("selected_v4_candidate", scenario)]["collision_rate"])
        - float(aggregate[("temporal_graph", scenario)]["collision_rate"])
        for scenario in scenarios
    }
    rules = contract["promotion"]["thresholds"]
    candidate_carla = aggregate[("selected_v4_candidate", "carla")]
    candidate_cross = aggregate[("selected_v4_candidate", "cross")]
    carla_run_gates = {
        str(row["seed"]): _carla_target_gate(row, contract["development"])
        for row in rows
        if row["method"] == "selected_v4_candidate" and row["scenario"] == "carla"
    }
    cross_run_gates = {
        str(row["seed"]): _metric_gate(row, contract["development"]["cross_guard"])
        for row in rows
        if row["method"] == "selected_v4_candidate" and row["scenario"] == "cross"
    }
    checks = {
        "carla_success_action_and_margin_each_seed": bool(carla_run_gates)
        and all(gate["passed"] for gate in carla_run_gates.values()),
        "cross_regression_guard_each_seed": bool(cross_run_gates)
        and all(gate["passed"] for gate in cross_run_gates.values()),
        "all_candidate_off_route_zero": all(
            float(aggregate[("selected_v4_candidate", scenario)]["off_route_rate"]) <= 0.0
            for scenario in scenarios
        ),
        "per_scenario_collision_noninferiority": all(
            delta <= float(rules["per_scenario_collision_noninferiority_margin"])
            for delta in collision_deltas.values()
        ),
        "worst_paired_seed_success_noninferiority": min(
            float(row["success_rate_delta"]) for row in paired
        )
        >= -float(rules["worst_paired_seed_success_margin"]),
        "macro_success_noninferiority": _mean(success_deltas.values())
        >= -float(rules["macro_success_noninferiority_margin"]),
        "positive_effect": (
            max(success_deltas.values()) >= float(rules["positive_effect_success_gain"])
            or min(collision_deltas.values())
            <= -float(rules["positive_effect_collision_reduction"])
        ),
    }
    candidate_parameters = max(
        int(row["model_plus_representation_parameters"])
        for row in rows
        if row["method"] == "selected_v4_candidate"
    )
    baseline_parameters = max(
        int(row["model_plus_representation_parameters"])
        for row in rows
        if row["method"] == "temporal_graph"
    )
    candidate_inference = _mean(
        row["mean_inference_ms"] for row in rows if row["method"] == "selected_v4_candidate"
    )
    baseline_inference = _mean(
        row["mean_inference_ms"] for row in rows if row["method"] == "temporal_graph"
    )
    decision = "pass" if all(checks.values()) else "fail"
    summary_path = stage_root("promotion") / "summary.json"
    receipt = {
        "schema_version": "topo-scene-v4.3.promotion-gate/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": decision,
        **hashes,
        "promotion_results": str(summary_path.resolve()),
        "promotion_results_sha256": _sha256(summary_path),
        "formal_algorithms": [
            "temporal_graph_v1_control",
            "topo_v4_3_target_critic_decoder",
        ],
        "checks": checks,
        "carla_per_seed_gates": carla_run_gates,
        "cross_per_seed_gates": cross_run_gates,
        "per_scenario_success_delta": success_deltas,
        "per_scenario_collision_delta": collision_deltas,
        "worst_paired_seed_success_delta": min(float(row["success_rate_delta"]) for row in paired),
        "macro_success_delta": _mean(success_deltas.values()),
        "paired_seed_deltas": paired,
        "compute_cost": {
            "candidate_parameters": candidate_parameters,
            "baseline_parameters": baseline_parameters,
            "parameter_ratio": candidate_parameters / max(baseline_parameters, 1),
            "candidate_mean_inference_ms": candidate_inference,
            "baseline_mean_inference_ms": baseline_inference,
            "inference_ratio": candidate_inference / max(baseline_inference, 1e-12),
            "non_compensatory": True,
        },
        "formal_test_unlocked": decision == "pass",
    }
    _write_json(DEFAULT_PROMOTION_GATE, receipt)
    return receipt


def compute_formal_decision(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    gate = _load_json(DEFAULT_PROMOTION_GATE)
    _require(gate.get("decision") == "pass", "formal test remains locked")
    summary = summarize_stage(contract, contract_digest, "formal", hashes)
    _require(summary["complete"] is True, "formal matrix is incomplete")
    rows = summary["per_run"]
    aggregate = {
        (row["method"], row["scenario"]): row for row in summary["aggregate"]
    }
    success_deltas = {
        scenario: float(aggregate[("selected_v4_candidate", scenario)]["success_rate"])
        - float(aggregate[("temporal_graph", scenario)]["success_rate"])
        for scenario in SCENARIOS
    }
    collision_deltas = {
        scenario: float(aggregate[("selected_v4_candidate", scenario)]["collision_rate"])
        - float(aggregate[("temporal_graph", scenario)]["collision_rate"])
        for scenario in SCENARIOS
    }
    formal = contract["formal_test"]
    acceptance = formal["final_acceptance"]
    checks = {
        "every_scenario_success_noninferiority": all(
            delta >= float(acceptance["every_scenario_success_delta_min"])
            for delta in success_deltas.values()
        ),
        "every_scenario_collision_noninferiority": all(
            delta <= float(acceptance["every_scenario_collision_delta_max"])
            for delta in collision_deltas.values()
        ),
        "positive_effect": (
            max(success_deltas.values()) >= float(acceptance["positive_effect_any_success_delta_min"])
            or min(collision_deltas.values())
            <= float(acceptance["positive_effect_any_collision_delta_max"])
        ),
    }
    statistics = paired_hierarchical_bootstrap(
        rows,
        candidate_method="selected_v4_candidate",
        baseline_method="temporal_graph",
        scenarios=SCENARIOS,
        resamples=int(formal["hierarchical_bootstrap_resamples"]),
        seed=int(formal["bootstrap_seed"]),
    )
    statistics["schema_version"] = "topo-scene-v4.3.paired-hierarchical-bootstrap/v1"
    statistics_path = stage_root("formal") / "paired_hierarchical_bootstrap.json"
    _write_json(statistics_path, statistics)
    decision = {
        "schema_version": "topo-scene-v4.3.formal-decision/v1",
        "computed_from_real_runs": True,
        "fabricated_values": False,
        "decision": "pass" if all(checks.values()) else "fail",
        **hashes,
        "promotion_gate_sha256": _sha256(DEFAULT_PROMOTION_GATE),
        "formal_summary_sha256": _sha256(stage_root("formal") / "summary.json"),
        "statistics_sha256": _sha256(statistics_path),
        "checks": checks,
        "per_scenario_success_delta": success_deltas,
        "per_scenario_collision_delta": collision_deltas,
        "paired_seed_deltas": _paired_deltas(rows),
        "formal_test_accessed": True,
    }
    _write_json(stage_root("formal") / "formal_decision.json", decision)
    return decision


def _assert_stage_can_run(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str], stage: str
) -> None:
    validate_implementation_freeze()
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing engineering receipt")
    engineering = _load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(engineering.get("status") == "passed", "engineering gate did not pass")
    _require(engineering.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "engineering receipt freeze mismatch")
    if stage == "promotion":
        path = stage_root("development") / "development_decision.json"
        _require(path.is_file(), "development gate is incomplete")
        decision = _load_json(path)
        _require(decision.get("decision") == "pass", "development gate did not pass")
        _require(decision.get("implementation_freeze_sha256") == hashes["implementation_freeze_sha256"], "development gate freeze mismatch")
    if stage == "formal":
        _require(DEFAULT_PROMOTION_GATE.is_file(), "formal test locked: promotion receipt missing")
        receipt = _load_json(DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        for key, value in hashes.items():
            if key in ("stage0_attribution_sha256",):
                continue
            _require(receipt.get(key) == value, f"formal test locked: {key} mismatch")


def _select_jobs(
    jobs: list[Job], selector: str | None, *, development_next: str | None
) -> list[Job]:
    if selector in (None, "all"):
        return jobs
    if selector == "next":
        _require(development_next is not None, "no eligible next development job")
        return [next(job for job in jobs if job.job_id == development_next)]
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [job for job in jobs if job.job_id in requested or job.name in requested]
    _require(len(selected) == len(requested), f"unknown or ambiguous job selector: {selector}")
    return selected


def write_plan(
    contract: dict[str, Any], contract_digest: str, stage: str, hashes: dict[str, str], *, device: str
) -> dict[str, Any]:
    jobs = jobs_for_stage(contract, contract_digest, stage)
    payload = {
        "schema_version": "topo-scene-v4.3.stage-plan/v1",
        "stage": stage,
        "device": device,
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


def execute_jobs(
    contract: dict[str, Any],
    contract_digest: str,
    stage: str,
    hashes: dict[str, str],
    *,
    selector: str | None,
    device: str,
    workers: int,
    dry_run: bool,
) -> int:
    _assert_stage_can_run(contract, contract_digest, hashes, stage)
    jobs = jobs_for_stage(contract, contract_digest, stage)
    next_job = None
    if stage == "development":
        status = development_status(contract, contract_digest, hashes)
        next_job = status["next_job"]
        if selector in (None, "next"):
            selector = "next"
    selected = _select_jobs(jobs, selector, development_next=next_job)
    if stage == "development":
        _require(len(selected) == 1, "development jobs must execute sequentially")
        _require(selected[0].job_id == next_job, f"development protocol requires {next_job}")
    plan = write_plan(contract, contract_digest, stage, hashes, device=device)
    if dry_run:
        print(json.dumps({"selected": [job.name for job in selected], "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    _require(workers > 0, "workers must be positive")
    pending: list[Job] = []
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
    failures: list[dict[str, Any]] = []

    def run_one(job: Job) -> dict[str, Any] | None:
        log_path = log_root / f"{job.name}.log"
        command = command_for(job, hashes, device=device)
        print(json.dumps({"job": job.name, "status": "running", "log": str(log_path)}, ensure_ascii=False), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        accepted, reason = accepted_run_reason(job, hashes)
        if completed.returncode != 0 or not accepted:
            failure = {
                "job": job.name,
                "returncode": completed.returncode,
                "acceptance_reason": reason,
                "log": str(log_path),
            }
            print(json.dumps({**failure, "status": "failed"}, ensure_ascii=False), flush=True)
            return failure
        print(json.dumps({"job": job.name, "status": "accepted"}, ensure_ascii=False), flush=True)
        return None

    if workers == 1:
        for job in pending:
            failure = run_one(job)
            if failure is not None:
                failures.append(failure)
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, job) for job in pending]
            for future in as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
    execution = {
        "schema_version": "topo-scene-v4.3.execution/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **hashes,
        "selected": [job.name for job in selected],
        "attempted": [job.name for job in pending],
        "workers": workers,
        "failures": failures,
    }
    _write_json(stage_root(stage) / "last_execution.json", execution)
    summarize_stage(contract, contract_digest, stage, hashes)
    if stage == "development":
        write_development_decision(contract, contract_digest, hashes)
    return 1 if failures else 0


def status_payload(
    contract: dict[str, Any], contract_digest: str, hashes: dict[str, str]
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in STAGES:
        jobs = jobs_for_stage(contract, contract_digest, stage)
        accepted = sum(accepted_run(job, hashes) for job in jobs)
        stages[stage] = {"accepted": accepted, "expected": len(jobs), "complete": accepted == len(jobs)}
    development = development_status(contract, contract_digest, hashes)
    promotion = _load_json(DEFAULT_PROMOTION_GATE) if DEFAULT_PROMOTION_GATE.is_file() else None
    return {
        "schema_version": "topo-scene-v4.3.status/v1",
        **hashes,
        "stages": stages,
        "development_decision": development["decision"],
        "next_development_job": development["next_job"],
        "formal_test_unlocked": bool(promotion and promotion.get("decision") == "pass"),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    root.add_argument("--stage0-results", type=Path, default=DEFAULT_STAGE0_RESULTS)
    root.add_argument("--stage0-attribution", type=Path, default=DEFAULT_STAGE0_ATTRIBUTION)
    root.add_argument("--failure-attribution", type=Path, default=DEFAULT_FAILURE_ATTRIBUTION)
    root.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    subparsers = root.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--migration-log", type=Path, required=True)
    freeze.add_argument("--regression-log", type=Path, required=True)
    freeze.add_argument("--targeted-log", type=Path, required=True)
    freeze.add_argument("--smoke-dir", type=Path, required=True)
    freeze.add_argument("--replay-smoke-dir", type=Path, required=True)
    freeze.add_argument("--check-dir", type=Path, action="append", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--stage", choices=STAGES, required=True)
    plan.add_argument("--device", default="cuda")
    run = subparsers.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--job", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--stage", choices=STAGES, required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--stage", choices=STAGES, required=True)
    subparsers.add_parser("status")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    contract_path = args.contract.resolve()
    contract = validate_contract(load_contract(contract_path))
    digest = _sha256(contract_path)
    if args.command == "freeze":
        payload = build_implementation_freeze(
            contract_path=contract_path,
            stage0_results=args.stage0_results.resolve(),
            stage0_attribution=args.stage0_attribution.resolve(),
            failure_attribution=args.failure_attribution.resolve(),
            migration_log=args.migration_log,
            regression_log=args.regression_log,
            targeted_log=args.targeted_log,
            smoke_dir=args.smoke_dir,
            replay_smoke_dir=args.replay_smoke_dir,
            check_dirs=args.check_dir,
            output=args.freeze,
        )
        print(json.dumps({"freeze": str(args.freeze.resolve()), "scientific_content_sha256": payload["scientific_content_sha256"], "implementation_freeze_sha256": _sha256(args.freeze.resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        stage_hashes = {
            **_validate_stage0(
                args.stage0_results.resolve(), args.stage0_attribution.resolve()
            ),
            **_validate_failure_attribution(args.failure_attribution.resolve()),
        }
        print(json.dumps({"status": "valid", "experiment_contract_sha256": digest, **stage_hashes}, ensure_ascii=False, indent=2))
        return 0
    hashes = protocol_hashes(
        contract_path=contract_path,
        stage0_results=args.stage0_results.resolve(),
        stage0_attribution=args.stage0_attribution.resolve(),
        failure_attribution=args.failure_attribution.resolve(),
        freeze_path=args.freeze.resolve(),
    )
    if args.command == "plan":
        print(json.dumps(write_plan(contract, digest, args.stage, hashes, device=args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
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
    if args.command == "summarize":
        print(json.dumps(summarize_stage(contract, digest, args.stage, hashes), ensure_ascii=False, indent=2))
        return 0
    if args.command == "gate":
        if args.stage == "development":
            payload = write_development_decision(contract, digest, hashes)
        elif args.stage == "promotion":
            payload = compute_promotion_gate(contract, digest, hashes)
        else:
            payload = compute_formal_decision(contract, digest, hashes)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(status_payload(contract, digest, hashes), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())




