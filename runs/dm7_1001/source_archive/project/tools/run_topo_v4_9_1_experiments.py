"""Apply the v4.9.1 read-only selector-metadata acceptance patch.

The scientific v4.9 contract, implementation freeze, model, training,
selection, evaluation, and gates remain unchanged.  This wrapper only exposes
two redundant parent-validator metadata fields when existing selector and
validation receipts prove their values.  No run artifact is rewritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_topo_v4_9_experiments as parent


PATCH_CONTRACT = (
    ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_9_1.yaml"
)
PREREGISTRATION = ROOT / "results_topo_v4_9_1_dev" / "preregistration_receipt.json"
ENGINEERING_ROOT = ROOT / "results_topo_v4_9_1_dev" / "engineering"
PREFLIGHT = ENGINEERING_ROOT / "acceptance_preflight.json"
PATCH_FREEZE = ENGINEERING_ROOT / "implementation_freeze.json"
ENGINEERING_RECEIPT = ENGINEERING_ROOT / "engineering_receipt.json"

PATCH_CONTRACT_SHA256 = (
    "c2f75d8be80c1001ccde76b36cdbfaadca098f681555c63974b0686b73c1cdd1"
)
PARENT_CONTRACT_SHA256 = (
    "4a8f39b734357efda228e1981c0a3d1bf956dba2cdd6af0d49cfafa8b7aae16d"
)
PARENT_FREEZE_SHA256 = (
    "766ff1a0b4856e5f9a07afb43ac7122fdc7d9a63a3ec76e810cdf947eea22a5b"
)
PARENT_RECEIPT_SHA256 = (
    "aeb77f12885c01082d665b8992420c21ea1aa2a0108ab66c2dc16b23720838cc"
)
PARENT_RUNNER_SHA256 = (
    "11bdc61834018e4e55027e76dbc836b82ba143ee660c39dc8096f6e0d97a5241"
)
INTERRUPTION_RECEIPT_SHA256 = (
    "6e7cae3383c30b77e5acdf479cf515de502380b0fa081a37c255875878c59859"
)
TRIGGER_JOB_IDS = ("M1", "M2")
TRIGGER_REASON = (
    "parent structural acceptance failed: metadata selector partition drifted"
)

_PARENT_ACCEPTANCE_VIEW = parent._parent_acceptance_view


class V491PatchError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V491PatchError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_patch_contract(path: Path = PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.9.1 patch contract must be a mapping")
    return value


def validate_patch_contract(
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = load_patch_contract() if contract is None else contract
    _require(
        value.get("schema_version")
        == "topo-scene-v4.9.1.read-only-selector-metadata-acceptance-patch/v1",
        "unexpected v4.9.1 patch schema",
    )
    for key in (
        "scientific_protocol_changed",
        "model_changed",
        "training_changed",
        "selection_changed",
        "evaluation_changed",
        "gate_thresholds_changed",
    ):
        _require(value.get(key) is False, f"patch changed science: {key}")
    _require(value.get("formal_test_locked") is True, "formal lock drifted")
    lineage = value.get("parent", {})
    expected = {
        "scientific_contract_sha256": PARENT_CONTRACT_SHA256,
        "scientific_freeze_sha256": PARENT_FREEZE_SHA256,
        "engineering_receipt_sha256": PARENT_RECEIPT_SHA256,
        "runner_sha256": PARENT_RUNNER_SHA256,
        "formal_test_accessed": False,
    }
    for key, observed in expected.items():
        _require(lineage.get(key) == observed, f"parent {key} drifted")
    change = value.get("single_engineering_change", {})
    _require(
        change.get("name")
        == "derive_parent_selector_metadata_from_existing_v4_9_evidence_in_read_only_view",
        "engineering change drifted",
    )
    _require(change.get("writes_completed_run_artifacts") is False, "patch writes runs")
    _require(change.get("bypasses_strict_v4_9_model_checks") is False, "strict checks bypassed")
    _require(change.get("changes_candidate_outcome_gate") is False, "outcome gate changed")
    recovery = value.get("recovery", {})
    _require(recovery.get("preserve_interrupted_attempt") is True, "M3 evidence not preserved")
    _require(recovery.get("rerun_M1") is False, "M1 rerun enabled")
    _require(recovery.get("rerun_M2") is False, "M2 rerun enabled")
    _require(
        recovery.get("development_execution_policy")
        == "attempt_all_then_summarize",
        "development batch policy drifted",
    )
    promotion = value.get("promotion", {})
    _require(promotion.get("jobs") == 12, "promotion matrix drifted")
    _require(
        promotion.get("failure_does_not_cancel_remaining_jobs") is True,
        "promotion fail-fast re-enabled",
    )
    _require(value.get("formal_test", {}).get("accessed") is False, "formal accessed")
    return value


def validate_preregistration() -> dict[str, Any]:
    receipt = _load_json(PREREGISTRATION)
    _require(
        receipt.get("schema_version")
        == "topo-scene-v4.9.1.read-only-selector-metadata-patch-preregistration/v1",
        "preregistration schema drifted",
    )
    _require(receipt.get("created_before_patch_implementation") is True, "patch was not preregistered")
    _require(receipt.get("created_after_real_run_trigger") is True, "real trigger missing")
    _require(receipt.get("scientific_protocol_changed") is False, "preregistration changed science")
    _require(receipt.get("completed_M1_or_M2_will_be_rerun") is False, "completed jobs may rerun")
    _require(receipt.get("completed_run_artifacts_will_be_modified") is False, "completed runs may change")
    _require(receipt.get("interrupted_M3_attempt_will_be_preserved") is True, "M3 attempt not preserved")
    _require(receipt.get("patch_contract_sha256") == PATCH_CONTRACT_SHA256, "patch contract hash drifted")
    _require(receipt.get("parent_scientific_freeze_sha256") == PARENT_FREEZE_SHA256, "parent freeze drifted")
    _require(receipt.get("parent_runner_sha256") == PARENT_RUNNER_SHA256, "parent runner drifted")
    _require(receipt.get("trigger_acceptance_reason") == TRIGGER_REASON, "trigger reason drifted")
    _require(receipt.get("formal_test_accessed") is False, "preregistration accessed formal")
    return receipt


def _derived_parent_metadata(
    selector: dict[str, Any], detailed: dict[str, Any]
) -> dict[str, Any]:
    """Derive only facts already proved by immutable selector/eval receipts."""

    _require(
        selector.get("selection_partition") == "train",
        "cannot derive train selector partition",
    )
    _require(
        detailed.get("selector_receipt_precedes_validation_environment") is True,
        "cannot derive validation exclusion without temporal receipt evidence",
    )
    _require(
        detailed.get("selector_receipt_sha256") is not None,
        "detailed selector receipt binding is missing",
    )
    return {
        "checkpoint_calibration_partition": "train",
        "validation_used_for_checkpoint_selection": False,
    }


@contextmanager
def patched_parent_acceptance_view(job: parent.Job) -> Iterator[None]:
    """Add the two proved metadata fields to the parent's in-memory view."""

    with _PARENT_ACCEPTANCE_VIEW(job):
        original_loader = parent.parent._load_json

        def compatibility_loader(path: Path) -> dict[str, Any]:
            value = original_loader(path)
            if job.algorithm != parent.V49_CANDIDATE or path.name != "method_metadata.json":
                return value
            root = path.parent
            selector = parent._load_json(root / "selector" / "receipt.json")
            detailed = parent._load_json(root / "paper_evaluation_detailed.json")
            derived = _derived_parent_metadata(selector, detailed)
            output = copy.deepcopy(value)
            for key, expected in derived.items():
                observed = output.get(key)
                _require(
                    observed in (None, expected),
                    f"real metadata conflicts with derived {key}",
                )
                output[key] = expected
            return output

        parent.parent._load_json = compatibility_loader
        try:
            yield
        finally:
            parent.parent._load_json = original_loader


@contextmanager
def patched_acceptance() -> Iterator[None]:
    original = parent._parent_acceptance_view
    parent._parent_acceptance_view = patched_parent_acceptance_view
    try:
        yield
    finally:
        parent._parent_acceptance_view = original


def _tree_hashes(root: Path) -> dict[str, str]:
    _require(root.is_dir(), f"run directory missing: {root}")
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _trigger_jobs() -> tuple[dict[str, Any], str, dict[str, str], list[parent.Job]]:
    contract_path = parent.DEFAULT_CONTRACT.resolve()
    contract = parent.validate_contract(parent.load_contract(contract_path))
    digest = parent._sha256(contract_path)
    hashes = parent.protocol_hashes(contract_path=contract_path)
    jobs = [
        job
        for job in parent.jobs_for_stage(contract, digest, "development")
        if job.job_id in TRIGGER_JOB_IDS
    ]
    _require([job.job_id for job in jobs] == list(TRIGGER_JOB_IDS), "trigger jobs drifted")
    return contract, digest, hashes, jobs


def acceptance_preflight(*, write: bool = True) -> dict[str, Any]:
    validate_patch_contract()
    validate_preregistration()
    _require(_sha256(PATCH_CONTRACT) == PATCH_CONTRACT_SHA256, "patch contract bytes drifted")
    _require(_sha256(parent.DEFAULT_CONTRACT) == PARENT_CONTRACT_SHA256, "parent contract drifted")
    _require(_sha256(parent.DEFAULT_FREEZE) == PARENT_FREEZE_SHA256, "parent freeze drifted")
    _require(_sha256(parent.DEFAULT_ENGINEERING_RECEIPT) == PARENT_RECEIPT_SHA256, "parent receipt drifted")
    _require(_sha256(ROOT / "tools" / "run_topo_v4_9_experiments.py") == PARENT_RUNNER_SHA256, "parent runner drifted")
    interruption = (
        ROOT
        / "results_topo_v4_9_dev"
        / "development"
        / "interrupted_attempts"
        / "M3__cand__ram__s12__p4a8f39b7__attempt1"
        / "interruption_receipt.json"
    )
    _require(_sha256(interruption) == INTERRUPTION_RECEIPT_SHA256, "M3 interruption receipt drifted")

    _, _, hashes, jobs = _trigger_jobs()
    roots = {job.job_id: parent.run_directory(job) for job in jobs}
    before = {job_id: _tree_hashes(root) for job_id, root in roots.items()}
    original = {
        job.job_id: {
            "accepted": result[0],
            "reason": result[1],
        }
        for job in jobs
        for result in [parent.accepted_run_reason(job, hashes)]
    }
    _require(
        all(
            not row["accepted"] and row["reason"] == TRIGGER_REASON
            for row in original.values()
        ),
        "original acceptance trigger no longer reproduces",
    )
    with patched_acceptance():
        patched = {
            job.job_id: {
                "accepted": result[0],
                "reason": result[1],
            }
            for job in jobs
            for result in [parent.accepted_run_reason(job, hashes)]
        }
    _require(
        all(row == {"accepted": True, "reason": "accepted"} for row in patched.values()),
        f"patched acceptance did not adopt trigger jobs: {patched}",
    )
    after = {job_id: _tree_hashes(root) for job_id, root in roots.items()}
    _require(before == after, "completed run artifacts changed during preflight")
    value = {
        "schema_version": "topo-scene-v4.9.1.acceptance-preflight/v1",
        "scientific_protocol_changed": False,
        "read_only_normalized_view": True,
        "derived_fields": {
            "checkpoint_calibration_partition": "train",
            "validation_used_for_checkpoint_selection": False,
        },
        "original": original,
        "patched": patched,
        "completed_run_artifact_hashes_before": before,
        "completed_run_artifact_hashes_after": after,
        "completed_run_artifacts_modified": False,
        "completed_jobs_rerun": False,
        "interrupted_M3_attempt_preserved": True,
        "formal_test_accessed": False,
    }
    if write:
        _write_json(PREFLIGHT, value)
    return value


def validate_patch_freeze(path: Path = PATCH_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _require(
        payload.get("schema_version")
        == "topo-scene-v4.9.1.read-only-selector-metadata-patch-freeze/v1",
        "invalid v4.9.1 patch freeze",
    )
    _require(payload.get("scientific_protocol_changed") is False, "patch freeze changed science")
    _require(payload.get("formal_test_accessed") is False, "patch freeze accessed formal")
    _require(payload.get("patch_contract_sha256") == PATCH_CONTRACT_SHA256, "patch freeze contract drifted")
    _require(payload.get("parent_scientific_freeze_sha256") == PARENT_FREEZE_SHA256, "patch freeze parent drifted")
    files = payload.get("patch_files", {})
    _require(isinstance(files, dict) and files, "patch file hashes missing")
    for relative, expected in files.items():
        path = ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        _require(_sha256(path) == expected, f"patch file drifted: {relative}")
    return payload


def validate_engineering_receipt(path: Path = ENGINEERING_RECEIPT) -> dict[str, Any]:
    receipt = _load_json(path)
    _require(receipt.get("status") == "passed", "v4.9.1 engineering did not pass")
    _require(receipt.get("formal_test_accessed") is False, "engineering accessed formal")
    _require(
        receipt.get("implementation_freeze_sha256") == _sha256(PATCH_FREEZE),
        "v4.9.1 engineering freeze mismatch",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["preflight"]:
        value = acceptance_preflight(write=True)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if arguments == ["validate-patch"]:
        validate_patch_contract()
        validate_preregistration()
        freeze = validate_patch_freeze()
        validate_engineering_receipt()
        print(
            json.dumps(
                {
                    "status": "valid",
                    "patch_contract_sha256": PATCH_CONTRACT_SHA256,
                    "parent_scientific_freeze_sha256": PARENT_FREEZE_SHA256,
                    "patch_freeze_sha256": _sha256(PATCH_FREEZE),
                    "patch_content_sha256": freeze.get("patch_content_sha256"),
                    "formal_test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    validate_patch_contract()
    validate_preregistration()
    validate_patch_freeze()
    validate_engineering_receipt()
    with patched_acceptance():
        return parent.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
