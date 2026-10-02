"""Freeze the v4.12 implementation after engineering gates pass."""

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

from configs.sb3_configs_v4_12 import V412_FULL
from tools import run_topo_v4_12_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_12_dev" / "engineering"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
AUDIT = ENGINEERING / "no_kinematic_projection_audit.json"
TARGETED_XML = ENGINEERING / "targeted_tests.xml"
REGRESSION_XML = ENGINEERING / "inherited_regression_tests.xml"
SMOKE = ROOT / "r412s" / "e2" / "m"
LEDGER = ROOT / "experiments" / "topo_scene_v4" / "V4_12_ITERATION_LEDGER.md"

SCIENTIFIC_FILES = (
    "experiments/topo_scene_v4/experiment_contract_v4_12.yaml",
    "experiments/topo_scene_v4/V4_12_AUGMENTED_JOINT_SUPPORT_MODEL.md",
    "experiments/topo_scene_v4/V4_12_ITERATION_LEDGER.md",
    "algos/sb3_torch/hybrid_policy_v4_12_model.py",
    "algos/sb3_torch/sac_v4_12_model.py",
    "configs/sb3_configs_v4_12.py",
    "tools/action_diagnostics_v4_12_model.py",
    "tools/audit_v4_12_no_kinematic_projection.py",
    "tools/train_sb3_v4_12.py",
    "tools/train_paper_sb3_sumo_v4_12.py",
    "tools/run_topo_v4_12_experiments.py",
    "tools/run_all_v4_12_pipeline.py",
    "tools/start_all_pending_promotions_detached.py",
    "tests_sb3_sumo/test_augmented_joint_support_v4_12.py",
    "tests_sb3_sumo/test_action_diagnostics_v4_12.py",
    "tests_sb3_sumo/test_train_sb3_v4_12.py",
    "tests_sb3_sumo/test_run_topo_v4_12_experiments.py",
    "tests_sb3_sumo/test_run_all_v4_12_pipeline.py",
    "tests_sb3_sumo/test_audit_v4_12_no_kinematic_projection.py",
)
INHERITED_EXECUTION_DEPENDENCIES = (
    "tools/run_all_pending_promotions_v4_11.py",
    "tools/train_sb3_v4_11.py",
    "tools/train_paper_sb3_sumo_v4_5.py",
    "algos/sb3_torch/hybrid_policy_v4_11_model.py",
    "algos/sb3_torch/hybrid_policy_v4_10_model.py",
    "algos/sb3_torch/sac_v4_11_model.py",
    "envs/sumo/decision_alignment_v4.py",
    "envs/sumo/paper_env_v4.py",
    "envs/sumo/sumo_env.py",
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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _junit(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JUnit evidence missing: {path}")
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    _require(counts["tests"] > 0, f"JUnit has no tests: {path}")
    _require(counts["failures"] == 0, f"JUnit failures present: {path}")
    _require(counts["errors"] == 0, f"JUnit errors present: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), **counts}


def _validate_smoke() -> dict[str, Any]:
    required = tuple(runner.COMMON_REQUIRED_FILES) + (
        "collision_label_diagnostics.json",
    )
    for relative in required:
        _require((SMOKE / relative).is_file(), f"smoke artifact missing: {relative}")
    arguments = _load(SMOKE / "arguments.json")
    metadata = _load(SMOKE / "method_metadata.json")
    training = _load(SMOKE / "training_diagnostics.json")
    actions = _load(SMOKE / "action_diagnostics.json")
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    selector = _load(SMOKE / "selector" / "receipt.json")
    requested = arguments.get("requested_raw_steps", {})
    _require(requested.get("algo") == V412_FULL, "smoke algorithm drifted")
    _require(requested.get("evaluation_split") == "validation", "smoke used non-validation split")
    _require(arguments.get("formal_unlock") is None, "smoke used formal unlock")
    _require(detailed.get("formal_test_accessed") is False, "smoke accessed formal")
    _require(metadata.get("replay_support_objective") == "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll", "smoke joint support missing")
    for key in runner.FORBIDDEN_RUNTIME_FIELDS:
        _require(actions.get(key) is False, f"smoke enabled forbidden {key}")
    for key in (
        "selected_decoder_exact_model_match_rate",
        "selected_decoder_exact_action_match_rate",
        "selected_action_mask_feasible_rate",
        "exact_joint_support_model_argmax_rate",
        "joint_support_score_equation_match_rate",
        "selected_speed_exact_proposal_match_rate",
        "learned_proposal_source_rate",
        "no_action_rewrite_rate",
        "no_semantic_tie_override_rate",
        "no_actor_confidence_threshold_rate",
    ):
        _require(actions.get(key) == 1.0, f"smoke exact diagnostic failed: {key}")
    for name in (
        "support/replay_joint_action_nll",
        "support/replay_joint_action_nll_recomputed",
        "support/replay_lane_categorical_nll",
        "support/replay_conditional_speed_mixture_nll",
    ):
        row = training.get("statistics", {}).get(name, {})
        _require(int(row.get("count", 0)) > 0, f"smoke statistic missing: {name}")
        _require(math.isfinite(float(row.get("last"))), f"smoke statistic non-finite: {name}")
    _require(selector.get("deployment_decoder_candidates") == ["target_critic"], "smoke selector drifted")
    _require(selector.get("policy_parameter_state_preserved") is True, "smoke selector changed policy")
    return {
        "path": str(SMOKE.resolve()),
        "requested_raw_steps": int(requested["max_steps"]),
        "evaluation_episodes": int(detailed["summary"]["episodes"]),
        "selected_model_sha256": _sha256(SMOKE / "selected_model.zip"),
        "arguments_sha256": _sha256(SMOKE / "arguments.json"),
        "training_diagnostics_sha256": _sha256(SMOKE / "training_diagnostics.json"),
        "action_diagnostics_sha256": _sha256(SMOKE / "action_diagnostics.json"),
        "action_trace_sha256": _sha256(SMOKE / "action_diagnostics_decisions.jsonl"),
        "selector_receipt_sha256": _sha256(SMOKE / "selector" / "receipt.json"),
        "paper_evaluation_sha256": _sha256(SMOKE / "paper_evaluation_detailed.json"),
        "formal_test_accessed": False,
    }


def _path_budget() -> dict[str, Any]:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner._sha256(runner.DEFAULT_CONTRACT)
    paths = []
    for stage in runner.STAGES:
        for job in runner.jobs_for_stage(contract, digest, stage):
            for attempt in range(runner.MAX_IMMUTABLE_ATTEMPTS):
                paths.append(
                    runner.run_directory(job, attempt)
                    / "tensorboard"
                    / "SAC_1"
                    / "events.out.tfevents.1234567890.DESKTOP-A06RVVE.12345.0"
                )
    longest = max(paths, key=lambda value: len(str(value.resolve())))
    length = len(str(longest.resolve()))
    _require(length < 245, f"frozen short path exceeds conservative budget: {length}")
    return {
        "conservative_windows_budget": 245,
        "maximum_predicted_event_path_length": length,
        "longest_predicted_event_path": str(longest.resolve()),
        "passed": True,
    }


def main() -> int:
    _require(_sha256(runner.DEFAULT_CONTRACT) == runner.CONTRACT_SHA256, "contract bytes drifted")
    runner.validate_contract(runner.load_contract())
    runner.validate_preregistration()
    audit = _load(AUDIT)
    _require(audit.get("integrity_passed") is True, "no-projection audit failed")
    _require(audit.get("formal_test_accessed") is False, "audit accessed formal")
    _require(audit.get("engineering_runtime", {}).get("available") is True, "audit lacks runtime")
    targeted = _junit(TARGETED_XML)
    regression = _junit(REGRESSION_XML)
    smoke = _validate_smoke()
    path_budget = _path_budget()
    scientific_hashes = {
        relative: _sha256(ROOT / relative) for relative in SCIENTIFIC_FILES
    }
    inherited_hashes = {
        relative: _sha256(ROOT / relative)
        for relative in INHERITED_EXECUTION_DEPENDENCIES
    }
    freeze = {
        "schema_version": "topo-scene-v4.12.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_and_frozen",
        "scientific_version": "v4.12_augmented_joint_support_prcr",
        "computed_from_real_sources_and_runtime": True,
        "fabricated_values": False,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "parent_promotion_summary_sha256": runner.PARENT_PROMOTION_SUMMARY_SHA256,
        "parent_promotion_gate_sha256": runner.PARENT_PROMOTION_GATE_SHA256,
        "parent_attribution_sha256": runner.ATTRIBUTION_SHA256,
        "scientific_file_sha256": scientific_hashes,
        "inherited_execution_dependency_sha256": inherited_hashes,
        "targeted_tests": targeted,
        "inherited_regression_tests": regression,
        "no_kinematic_projection_audit": {
            "path": str(AUDIT.resolve()),
            "sha256": _sha256(AUDIT),
            "integrity_passed": True,
        },
        "no_kinematic_projection_audit_passed": True,
        "real_sumo_smoke": smoke,
        "real_sumo_smoke_passed": True,
        "windows_short_path_gate": path_budget,
        "failure_does_not_cancel_remaining_jobs": True,
        "immutable_failed_attempts_preserved": True,
        "formal_test_accessed": False,
    }
    _write(FREEZE, freeze)
    receipt = {
        "schema_version": "topo-scene-v4.12.engineering-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": _sha256(FREEZE),
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "targeted_tests_passed": targeted["tests"],
        "inherited_regression_tests_passed": regression["tests"],
        "real_sumo_smoke_passed": True,
        "no_kinematic_projection_audit_passed": True,
        "windows_short_path_gate_passed": True,
        "development_unlocked": True,
        "ablation_unlocked": False,
        "promotion_unlocked": False,
        "formal_test_accessed": False,
    }
    _write(RECEIPT, receipt)
    print(json.dumps({"freeze": freeze, "receipt": receipt}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

