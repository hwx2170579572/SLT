"""Seal the v4.9.1 read-only selector-metadata acceptance patch."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_9_1_experiments as patch


ENGINEERING = ROOT / "results_topo_v4_9_1_dev" / "engineering"
TARGETED = ENGINEERING / "logs" / "targeted_tests.xml"
FULL = ENGINEERING / "logs" / "full_regression_current_state.xml"
FREEZE = ENGINEERING / "implementation_freeze.json"
RECEIPT = ENGINEERING / "engineering_receipt.json"
INTERRUPTION_ROOT = (
    ROOT
    / "results_topo_v4_9_dev"
    / "development"
    / "interrupted_attempts"
    / "M3__cand__ram__s12__p4a8f39b7__attempt1"
)

PATCH_FILES = (
    "tools/run_topo_v4_9_1_experiments.py",
    "tools/run_all_v4_9_1_promotion.py",
    "tools/freeze_v4_9_1.py",
    "tests_sb3_sumo/test_topo_v4_9_1_experiments.py",
    "experiments/topo_scene_v4/experiment_contract_v4_9_1.yaml",
    "results_topo_v4_9_1_dev/preregistration_receipt.json",
    "results_topo_v4_9_dev/development/interrupted_attempts/M3__cand__ram__s12__p4a8f39b7__attempt1/interruption_receipt.json",
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


def _junit(path: Path, *, tests: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {
        **_source(path),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    _require(result["tests"] == tests, f"JUnit test count drifted: {path}")
    _require(result["failures"] == 0, f"JUnit failures: {path}")
    _require(result["errors"] == 0, f"JUnit errors: {path}")
    return result


def _archive_manifest(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    extended = "\\\\?\\" + str(resolved)
    prefix = extended.rstrip("\\") + "\\"
    rows = []
    for directory, _, names in os.walk(extended):
        for name in names:
            path = os.path.join(directory, name)
            # The pre-move PowerShell receipt hashed native Windows relative
            # paths. Preserve that exact serializer for byte-for-byte replay.
            relative = path[len(prefix) :]
            with open(path, "rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest().lower()
            rows.append((relative, digest, os.path.getsize(path)))
    # PowerShell Sort-Object is case-insensitive by default.
    rows.sort(key=lambda row: row[0].casefold())
    lines = "\n".join(f"{relative}\t{digest}\t{size}" for relative, digest, size in rows)
    return {
        "file_count": len(rows),
        "bytes": sum(row[2] for row in rows),
        "sorted_relative_path_sha256_bytes_manifest_sha256": hashlib.sha256(
            lines.encode("utf-8")
        ).hexdigest().lower(),
    }


def _validate_interruption_archive() -> dict[str, Any]:
    receipt_path = INTERRUPTION_ROOT / "interruption_receipt.json"
    run = INTERRUPTION_ROOT / "run"
    launcher = INTERRUPTION_ROOT / "launcher.log"
    manifest = INTERRUPTION_ROOT / "pre_move_manifest.tsv"
    _require(_sha256(receipt_path) == patch.INTERRUPTION_RECEIPT_SHA256, "interruption receipt drifted")
    receipt = _load(receipt_path)
    _require(receipt.get("formal_test_accessed") is False, "interruption accessed formal")
    _require(receipt["observation"]["accepted_scientific_run"] is False, "interrupted run was accepted")
    _require(receipt["recovery_policy"]["delete_or_overwrite_interrupted_attempt"] is False, "interrupted attempt may be deleted")
    _require(receipt["recovery_policy"]["count_interrupted_attempt_for_gate"] is False, "interrupted attempt entered gate")
    observed = _archive_manifest(run)
    expected = receipt["preserved_tree"]
    for key in ("file_count", "bytes"):
        _require(observed[key] == expected[key], f"interrupted archive {key} drifted")
    _require(
        _sha256(manifest)
        == expected["sorted_relative_path_sha256_bytes_manifest_sha256"],
        "preserved pre-move manifest drifted",
    )
    lines = manifest.read_text(encoding="utf-8").splitlines()
    _require(len(lines) == expected["file_count"], "pre-move manifest row count drifted")
    extended_root = "\\\\?\\" + str(run.resolve())
    for line in lines:
        relative, digest, size_text = line.split("\t")
        target = os.path.join(extended_root, *relative.split("/"))
        _require(os.path.isfile(target), f"archived file missing: {relative}")
        _require(os.path.getsize(target) == int(size_text), f"archived file size drifted: {relative}")
        with open(target, "rb") as stream:
            observed_digest = hashlib.sha256(stream.read()).hexdigest().lower()
        _require(observed_digest == digest, f"archived file hash drifted: {relative}")
    observed["sorted_relative_path_sha256_bytes_manifest_sha256"] = _sha256(
        manifest
    )
    _require(_sha256(launcher) == receipt["launcher_log"]["sha256"], "archived launcher log drifted")
    _require(
        not (
            ROOT
            / "results_topo_v4_9_dev"
            / "development"
            / "runs"
            / "M3__cand__ram__s12__p4a8f39b7"
        ).exists(),
        "canonical M3 retry path is not empty before seal",
    )
    return {
        "receipt": _source(receipt_path),
        "run_archive": observed,
        "pre_move_manifest": _source(manifest),
        "launcher_log": _source(launcher),
        "counted_for_gate": False,
    }


def main() -> int:
    patch.validate_patch_contract()
    preregistration = patch.validate_preregistration()
    _require(
        preregistration.get("contract_syntax_correction_after_implementation_start")
        is True,
        "contract syntax correction disclosure missing",
    )
    _require(
        preregistration.get("contract_syntax_correction_scientific_change")
        is False,
        "contract syntax correction changed science",
    )
    parent_freeze = patch.parent.validate_implementation_freeze()
    _require(_sha256(patch.parent.DEFAULT_FREEZE) == patch.PARENT_FREEZE_SHA256, "parent science freeze drifted")
    _require(_sha256(patch.parent.DEFAULT_ENGINEERING_RECEIPT) == patch.PARENT_RECEIPT_SHA256, "parent receipt drifted")
    _require(_sha256(ROOT / "tools" / "run_topo_v4_9_experiments.py") == patch.PARENT_RUNNER_SHA256, "parent runner changed")

    targeted = _junit(TARGETED, tests=8)
    full = _junit(FULL, tests=432)
    archive = _validate_interruption_archive()
    preflight = patch.acceptance_preflight(write=True)
    _require(preflight["completed_run_artifacts_modified"] is False, "preflight modified completed runs")
    _require(preflight["completed_jobs_rerun"] is False, "preflight reran completed jobs")
    _require(
        preflight["completed_run_artifact_hashes_before"]
        == preflight["completed_run_artifact_hashes_after"],
        "completed run hashes changed",
    )

    files: dict[str, str] = {}
    for relative in PATCH_FILES:
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        files[relative] = _sha256(path)
    payload = {
        "schema_version": "topo-scene-v4.9.1.read-only-selector-metadata-patch-freeze/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_protocol_changed": False,
        "model_changed": False,
        "training_changed": False,
        "selection_changed": False,
        "evaluation_changed": False,
        "gate_thresholds_changed": False,
        "formal_test_accessed": False,
        "single_engineering_change": "derive_parent_selector_metadata_from_existing_v4_9_evidence_in_read_only_view",
        "read_only_normalized_view": True,
        "completed_run_artifacts_modified": False,
        "completed_jobs_rerun": False,
        "parent_scientific_contract_sha256": patch.PARENT_CONTRACT_SHA256,
        "parent_scientific_freeze_sha256": patch.PARENT_FREEZE_SHA256,
        "parent_scientific_content_sha256": parent_freeze["scientific_content_sha256"],
        "parent_engineering_receipt_sha256": patch.PARENT_RECEIPT_SHA256,
        "parent_runner_sha256": patch.PARENT_RUNNER_SHA256,
        "patch_contract_sha256": patch.PATCH_CONTRACT_SHA256,
        "preregistration_receipt_sha256": _sha256(patch.PREREGISTRATION),
        "patch_files": files,
        "patch_content_sha256": _canonical_sha256(files),
        "engineering_evidence": {
            "targeted_tests": targeted,
            "full_current_state_regression": full,
            "acceptance_preflight": _source(patch.PREFLIGHT),
            "interrupted_attempt_archive": archive,
        },
        "recovery": {
            "accepted_completed_jobs": ["M1", "M2"],
            "next_jobs": ["M3", "M4"],
            "execution_policy": "attempt_all_then_summarize",
            "failure_does_not_cancel_remaining_jobs": True,
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
        "schema_version": "topo-scene-v4.9.1.read-only-selector-metadata-patch-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scientific_protocol_changed": False,
        "formal_test_accessed": False,
        "implementation_freeze": str(FREEZE.resolve()),
        "implementation_freeze_sha256": freeze_hash,
        "parent_scientific_freeze_sha256": patch.PARENT_FREEZE_SHA256,
        "patch_content_sha256": payload["patch_content_sha256"],
        "targeted_tests": targeted,
        "full_current_state_regression": full,
        "acceptance_preflight": _source(patch.PREFLIGHT),
        "interrupted_attempt_archive": archive,
        "completed_jobs_adopted_without_rerun": ["M1", "M2"],
        "next_jobs": ["M3", "M4"],
        "promotion_automation": "tools/run_all_v4_9_1_promotion.py",
        "formal_test_locked": True,
    }
    _write(RECEIPT, receipt)
    print(
        json.dumps(
            {
                "implementation_freeze": str(FREEZE.resolve()),
                "implementation_freeze_sha256": freeze_hash,
                "patch_content_sha256": payload["patch_content_sha256"],
                "engineering_receipt": str(RECEIPT.resolve()),
                "completed_jobs_adopted_without_rerun": ["M1", "M2"],
                "next_jobs": ["M3", "M4"],
                "formal_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
