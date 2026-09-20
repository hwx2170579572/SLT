"""Validate and freeze the v4.5.1 selector compatibility patch."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.checkpoint_selector_v4_4 import sha256


BASE_CONTRACT = PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_5.yaml"
BASE_FREEZE = PROJECT_ROOT / "results_topo_v4_5_dev" / "engineering" / "implementation_freeze.json"
PREREG = PROJECT_ROOT / "results_topo_v4_5_dev" / "engineering" / "v4_5_1" / "preregistration_receipt.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results_topo_v4_5_dev" / "engineering" / "v4_5_1" / "implementation_freeze.json"
PATCH_FILES = (
    "experiments/topo_scene_v4/V4_5_1_OVERLAPPING_TERMINAL_FLAGS_PATCH.md",
    "experiments/topo_scene_v4/engineering_amendment_v4_5_1.yaml",
    "results_topo_v4_5_dev/engineering/v4_5_1/preregistration_receipt.json",
    "tools/checkpoint_selector_v4_5_1.py",
    "tools/train_paper_sb3_sumo_v4_5_1.py",
    "tools/run_topo_v4_5_1_experiments.py",
    "tools/resume_topo_v4_5_1_e1.py",
    "tools/freeze_v4_5_1_patch.py",
    "tests_sb3_sumo/test_checkpoint_selector_v4_5_1.py",
)
EXPECTED_BASE_CONTRACT = "3c1f7f8bb2f2967f7559907386b58e67760c53dfa8e46bf581fc4824d186bb7c"
EXPECTED_BASE_FREEZE = "9ab0ccc71b33110ac526f5a9f89e8f504d0f3fc0ddafd6407c6ff36dc2414666"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def freeze(*, targeted_log: Path, regression_log: Path, output: Path) -> dict[str, Any]:
    _require(sha256(BASE_CONTRACT) == EXPECTED_BASE_CONTRACT, "base contract drifted")
    _require(sha256(BASE_FREEZE) == EXPECTED_BASE_FREEZE, "base freeze drifted")
    base = _load(BASE_FREEZE)
    for relative, expected in base.get("scientific_files", {}).items():
        path = PROJECT_ROOT / relative
        _require(path.is_file() and sha256(path) == expected, f"frozen v4.5 file drifted: {relative}")

    prereg = _load(PREREG)
    _require(prereg.get("status") == "sealed_before_implementation_and_before_validation", "patch was not preregistered")
    for key in ("amendment", "design_note"):
        binding = prereg[key]
        path = PROJECT_ROOT / binding["path"]
        _require(sha256(path) == binding["sha256"], f"preregistration {key} drifted")

    e1 = PROJECT_ROOT / "results_topo_v4_5_dev" / "development" / "runs" / "E1__cand__cross__s4__p3c1f7f8b"
    for relative in prereg["validation_artifacts_absent_at_seal"]:
        _require(not (e1 / relative).exists(), f"validation artifact exists before patch freeze: {relative}")

    for log in (targeted_log, regression_log):
        text = log.read_text(encoding="utf-8", errors="replace").lower()
        _require("passed" in text and "failed" not in text, f"test log did not pass: {log}")

    patch_hashes = {}
    for relative in PATCH_FILES:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"missing patch file: {relative}")
        patch_hashes[relative] = sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.5.1.implementation-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_e1_validation",
        "formal_test_accessed": False,
        "validation_environment_constructed": False,
        "base_contract_sha256": EXPECTED_BASE_CONTRACT,
        "base_implementation_freeze_sha256": EXPECTED_BASE_FREEZE,
        "single_engineering_change": "allow_integral_source_terminal_event_flags_to_overlap_in_checkpoint_selection",
        "model_training_decoder_evaluation_and_gates_unchanged": True,
        "patch_files": patch_hashes,
        "targeted_test_log": {"path": str(targeted_log.resolve()), "sha256": sha256(targeted_log)},
        "full_regression_log": {"path": str(regression_log.resolve()), "sha256": sha256(regression_log)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--targeted-log", type=Path, required=True)
    value.add_argument("--regression-log", type=Path, required=True)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = freeze(targeted_log=args.targeted_log, regression_log=args.regression_log, output=args.output)
    print(json.dumps({"output": str(args.output.resolve()), "patch_file_count": len(payload["patch_files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
