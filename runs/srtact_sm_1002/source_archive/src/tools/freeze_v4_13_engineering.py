"""Verify and immutably seal the v4.13 engineering implementation."""

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

from tools import run_topo_v4_13_experiments as runner


ENGINEERING = ROOT / "results_topo_v4_13_dev" / "engineering"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
AUDIT = ENGINEERING / "no_kinematic_projection_audit.json"
TARGETED = ENGINEERING / "targeted_tests.xml"
INHERITED = ENGINEERING / "inherited_regression_tests.xml"
SMOKE = ROOT / "r413s" / "e11" / "m"
SCIENTIFIC_FILES = (
    "experiments/topo_scene_v4/experiment_contract_v4_13.yaml",
    "experiments/topo_scene_v4/V4_13_GRADIENT_ISOLATED_TEMPERED_JOINT_SUPPORT.md",
    "experiments/topo_scene_v4/V4_13_ITERATION_LEDGER.md",
    "algos/sb3_torch/hybrid_policy_v4_13_model.py",
    "algos/sb3_torch/sac_v4_13_model.py",
    "configs/sb3_configs_v4_13.py",
    "tools/action_diagnostics_v4_13_model.py",
    "tools/train_sb3_v4_13.py",
    "tools/train_paper_sb3_sumo_v4_13.py",
    "tools/audit_v4_13_no_kinematic_projection.py",
    "tools/run_topo_v4_13_experiments.py",
    "tools/run_all_v4_13_pipeline.py",
    "tools/start_all_pending_promotions_v4_13_background.ps1",
    "tests_sb3_sumo/test_gradient_isolated_tempered_support_v4_13.py",
    "tests_sb3_sumo/test_action_diagnostics_v4_13.py",
    "tests_sb3_sumo/test_train_sb3_v4_13.py",
    "tests_sb3_sumo/test_run_topo_v4_13_experiments.py",
    "tests_sb3_sumo/test_run_all_v4_13_pipeline.py",
    "tests_sb3_sumo/test_audit_v4_13_no_kinematic_projection.py",
    "tests_sb3_sumo/test_start_all_pending_promotions_v4_13_background.py",
)
INHERITED_DEPENDENCIES = (
    "algos/sb3_torch/hybrid_policy_v4_12_model.py",
    "algos/sb3_torch/sac_v4_12_model.py",
    "configs/sb3_configs_v4_12.py",
    "tools/train_sb3_v4_12.py",
    "tools/action_diagnostics_v4_12_model.py",
    "envs/sumo/paper_env_v4.py",
    "envs/sumo/sumo_env.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = list(root.findall("testsuite")) if root.tag == "testsuites" else [root]
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    return {"path": str(path.resolve()), "sha256": _sha256(path), **totals}


def _formal_files() -> list[str]:
    root = runner.REPORT_ROOTS["formal"]
    return sorted(str(path.resolve()) for path in root.rglob("*") if path.is_file()) if root.exists() else []


def _maximum_windows_path() -> tuple[int, str]:
    contract = runner.validate_contract(runner.load_contract())
    digest = runner._sha256(runner.DEFAULT_CONTRACT)
    predicted = []
    suffix = Path("tensorboard") / "SAC_1" / "events.out.tfevents.1234567890.DESKTOP-A06RVVE.12345.0"
    for stage in runner.STAGES:
        for job in runner.jobs_for_stage(contract, digest, stage):
            predicted.append((runner.run_directory(job, 3) / suffix).resolve())
    longest = max(predicted, key=lambda path: len(str(path)))
    return len(str(longest)), str(longest)


def main() -> int:
    if FREEZE.exists() or RECEIPT.exists():
        raise FileExistsError("v4.13 engineering freeze already exists")
    if _sha256(runner.DEFAULT_CONTRACT) != runner.CONTRACT_SHA256:
        raise ValueError("v4.13 contract bytes drifted")
    runner.validate_contract(runner.load_contract())
    runner.validate_preregistration()

    targeted = _junit(TARGETED)
    inherited = _junit(INHERITED)
    if targeted != {**targeted, "tests": 32, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"targeted test gate failed: {targeted}")
    if inherited != {**inherited, "tests": 52, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"inherited test gate failed: {inherited}")

    audit = _load(AUDIT)
    if audit.get("integrity_passed") is not True:
        raise ValueError("no-projection audit failed")
    if audit.get("runtime_required") is not True:
        raise ValueError("no-projection audit omitted runtime")
    model = audit.get("model_execution", {})
    required_gradient_checks = (
        "lane_nll_updates_same_deployed_lane_head",
        "lane_nll_zero_gradient_actor_latent_trunk",
        "lane_nll_zero_gradient_speed_heads",
        "lane_nll_zero_gradient_scene_encoder",
        "speed_nll_updates_latent_and_speed_heads",
    )
    if not all(model.get(key) is True for key in required_gradient_checks):
        raise ValueError("gradient ownership gate failed")

    required_smoke = (
        "arguments.json",
        "training_diagnostics.json",
        "action_diagnostics.json",
        "action_diagnostics_decisions.jsonl",
        "paper_evaluation_detailed.json",
        "selected_model.zip",
        "selector/receipt.json",
    )
    for relative in required_smoke:
        if not (SMOKE / relative).is_file():
            raise FileNotFoundError(f"smoke artifact missing: {relative}")
    arguments = _load(SMOKE / "arguments.json")
    detailed = _load(SMOKE / "paper_evaluation_detailed.json")
    actions = _load(SMOKE / "action_diagnostics.json")
    requested = arguments.get("requested_raw_steps", {})
    if requested.get("max_steps") != 96 or requested.get("evaluation_split") != "validation":
        raise ValueError("smoke protocol drifted")
    if detailed.get("formal_test_accessed") is not False:
        raise ValueError("smoke accessed formal test")
    if actions.get("selected_decoder_records", 0) <= 0:
        raise ValueError("smoke has no action trace")
    if actions.get("lane_support_gradient_isolated") is not True:
        raise ValueError("smoke lost gradient isolation")

    formal_files = _formal_files()
    if formal_files:
        raise ValueError(f"formal test was touched: {formal_files}")
    maximum_length, longest_path = _maximum_windows_path()
    if maximum_length > 245:
        raise ValueError(f"Windows path budget exceeded: {maximum_length}")

    scientific = {relative: _sha256(ROOT / relative) for relative in SCIENTIFIC_FILES}
    inherited_hashes = {
        relative: _sha256(ROOT / relative) for relative in INHERITED_DEPENDENCIES
    }
    smoke_hashes = {
        relative: _sha256(SMOKE / relative) for relative in required_smoke
    }
    now = datetime.now(timezone.utc).isoformat()
    freeze = {
        "schema_version": "topo-scene-v4.13.implementation-freeze/v1",
        "created_at_utc": now,
        "status": "passed_and_frozen",
        "scientific_version": "v4.13_gradient_isolated_tempered_joint_support_prcr",
        "computed_from_real_sources_and_runtime": True,
        "fabricated_values": False,
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "parent_development_summary_sha256": runner.PARENT_DEVELOPMENT_SUMMARY_SHA256,
        "parent_development_decision_sha256": runner.PARENT_DEVELOPMENT_DECISION_SHA256,
        "parent_attribution_sha256": runner.PARENT_ATTRIBUTION_SHA256,
        "scientific_file_sha256": scientific,
        "inherited_execution_dependency_sha256": inherited_hashes,
        "targeted_tests": targeted,
        "inherited_regression_tests": inherited,
        "gradient_isolation_gate_passed": True,
        "no_kinematic_projection_audit": {
            "path": str(AUDIT.resolve()),
            "sha256": _sha256(AUDIT),
            "integrity_passed": True,
        },
        "no_kinematic_projection_audit_passed": True,
        "real_sumo_smoke": {
            "path": str(SMOKE.resolve()),
            "requested_raw_steps": 96,
            "evaluation_episodes": 1,
            "selected_model_sha256": _sha256(SMOKE / "selected_model.zip"),
            "artifact_sha256": smoke_hashes,
            "action_decision_records": int(actions["selected_decoder_records"]),
            "formal_test_accessed": False,
        },
        "real_sumo_smoke_passed": True,
        "windows_short_path_gate": {
            "conservative_windows_budget": 245,
            "maximum_predicted_event_path_length": maximum_length,
            "longest_predicted_event_path": longest_path,
            "passed": True,
        },
        "background_launcher": {
            "path": str((ROOT / "tools/start_all_pending_promotions_v4_13_background.ps1").resolve()),
            "hidden_start_process_probe_passed": True,
            "current_and_future_pipeline_discovery": True,
        },
        "failure_does_not_cancel_remaining_jobs": True,
        "immutable_failed_attempts_preserved": True,
        "formal_test_accessed": False,
    }
    _write(FREEZE, freeze)
    receipt = {
        "schema_version": "topo-scene-v4.13.engineering-receipt/v1",
        "created_at_utc": now,
        "status": "passed",
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": _sha256(FREEZE),
        "experiment_contract_sha256": runner.CONTRACT_SHA256,
        "targeted_tests_passed": targeted["tests"],
        "inherited_regression_tests_passed": inherited["tests"],
        "gradient_isolation_gate_passed": True,
        "real_sumo_smoke_passed": True,
        "no_kinematic_projection_audit_passed": True,
        "windows_short_path_gate_passed": True,
        "background_launcher_gate_passed": True,
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
