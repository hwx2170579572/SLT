"""Apply the v4.8.3 control secondary-seed acceptance patch."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_8 import (
    TEMPORAL_GRAPH,
    V48_CANDIDATE,
    V48_FORMAL_ALGORITHMS,
)
from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_2_experiments as parent_patch


PATCH_ID = "v4_8_3_accept_control_null_secondary_seed"
DEFAULT_PATCH_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_8_3.yaml"
)
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_8_3_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_8_3_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
PARENT_V48_FREEZE = v48.DEFAULT_FREEZE
PARENT_V48_FREEZE_SHA256 = parent_patch.PARENT_V48_FREEZE_SHA256
PARENT_V482_RUNNER_SHA256 = "3a51451ec894e566d0c7899d56c48e1d253213f87abbb62881a8d4e41f03d951"
PARENT_V482_FREEZE_SHA256 = "c39081844a2c8e28d9a3eeff62493f6d91402c674842a3fd52b60af938507bda"
PARENT_V482_RECEIPT_SHA256 = "11fe1516b6de2cef6aa7c1cee6b5c9b1827e4c1869ea15159739f33048ccae8e"
PROMOTION_JOB_NAME = "P__tg__cross__s0__p42f54135"
NEXT_PROMOTION_JOB = "P__tg__cross__s1__p42f54135"
PROMOTION_RUN = PROJECT_ROOT / "results_topo_v4_8_promotion" / "runs" / PROMOTION_JOB_NAME
PROMOTION_LAST_EXECUTION = PROJECT_ROOT / "results_topo_v4_8_promotion" / "last_execution.json"
PROMOTION_LAST_EXECUTION_TRIGGER_SHA256 = "1bcd2d401785ded6af8ce828f9ef194effd777f6f9543b12ae99f5f3d498b2d4"
PROMOTION_RUN_HASHES = {
    "arguments.json": "04c16c74b2e8179ee2666aabf466a3918374b93033f944e54aab94e36fc9789b",
    "method_metadata.json": "7503faadbf64074eb58a57d5df3db034722095fea703af66aaf89c6f05d7a576",
    "training_diagnostics.json": "14de6e8cef512a6d119c3676ed4cf540aab716287cd5b37719a935b97f53e9dd",
    "checkpoint_audit.json": "4c28e7ea20e3ff2506804f6385481ec37854cfc500c23d45788477ce5a4a78a9",
    "final_model.zip": "40939519be78ec7a1f2936c175733c3fbca495e223528a9ad647e399a493c62c",
    "selected_model.zip": "3fd39feffe4e63ed6cd92185e3e3bbe3aceb0bd88700a3b0449d76ce67e876bb",
    "performance_profile.json": "33f5907ba0bfd827e3a971730534e9c0bb3050158dc7a3ac3eab9608a7c0d39c",
    "final_evaluation.json": "b65d1f5494547e42c010ae7dca48c1329976b46e8a208a7faa4df99337f283c8",
    "action_diagnostics.json": "e2742e31dd30a72a465f8c42ad1c3e30be1612550e265ddf151a7ca65a8486aa",
    "action_diagnostics_decisions.jsonl": "0daa54ab0076024562f8ee0d12ca19c495495ef579623ea153f5c80c261a1438",
    "paper_evaluation_detailed.json": "0f14c62f6ccfe100dde254acafef7a6b54c9a8aff3dea2e19266766af02bdcc8",
    "selector/receipt.json": "4e698f7d173e3d5158dd4fc93e259e84dd184db2d856c88f58669784fa93115b",
    "return_estimator_diagnostics.json": "2c0e82ac392038672ec792c4c03fa826de7992407af59faaabd08a8616e1a440",
}
STAGES = v48.STAGES
_FROZEN_ACCEPTED_RUN_REASON = v48.accepted_run_reason


class V483PatchError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V483PatchError(message)


def load_patch_contract(path: Path = DEFAULT_PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.8.3 patch contract must be a mapping")
    return value


def validate_patch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.8.3.control-secondary-seed-acceptance-engineering-patch/v1",
        "unexpected v4.8.3 patch schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test is not locked")
    _require(contract.get("formal_test_accessed") is False, "patch accessed formal test")
    parent = contract["parent_scientific_protocol"]
    _require(parent.get("contract_sha256") == v48.contract_sha256(), "parent contract drifted")
    _require(parent.get("implementation_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "parent freeze drifted")
    _require(parent.get("unchanged") is True, "parent science changed")
    engineering = contract["parent_engineering_patch"]
    _require(engineering.get("runner_sha256") == PARENT_V482_RUNNER_SHA256, "v4.8.2 runner drifted")
    _require(engineering.get("implementation_freeze_sha256") == PARENT_V482_FREEZE_SHA256, "v4.8.2 freeze drifted")
    _require(engineering.get("engineering_receipt_sha256") == PARENT_V482_RECEIPT_SHA256, "v4.8.2 receipt drifted")
    _require(engineering.get("reused_unchanged") is True, "v4.8.2 is not frozen")
    _require(engineering.get("development_decision") == "pass", "development gate trigger drifted")
    trigger = contract["trigger"]
    _require(trigger.get("stage") == "promotion", "trigger stage drifted")
    _require(trigger.get("job") == PROMOTION_JOB_NAME, "trigger job drifted")
    _require(trigger.get("algorithm") == TEMPORAL_GRAPH, "trigger algorithm drifted")
    _require(trigger.get("subprocess_returncode") == 0, "promotion subprocess failed")
    _require(
        trigger.get("frozen_v4_8_2_acceptance_reason")
        == "argument secondary_calibration_seed_start mismatch",
        "trigger reason drifted",
    )
    _require(trigger.get("arguments_sha256") == PROMOTION_RUN_HASHES["arguments.json"], "trigger arguments drifted")
    _require(trigger.get("selector_receipt_sha256") == PROMOTION_RUN_HASHES["selector/receipt.json"], "trigger selector drifted")
    _require(trigger.get("paper_evaluation_detailed_sha256") == PROMOTION_RUN_HASHES["paper_evaluation_detailed.json"], "trigger evaluation drifted")
    _require("requested_secondary_calibration_seed_start" in trigger, "trigger null field missing")
    _require(trigger.get("requested_secondary_calibration_seed_start") is None, "trigger secondary seed is non-null")
    _require(trigger.get("requested_secondary_calibration_seed_start_field_present") is False, "trigger field-presence drifted")
    _require(trigger.get("fidelity_secondary_calibration_seed_offset") is None, "control fidelity offset drifted")
    _require(trigger.get("selector_mode") == "checkpoint_only_parent_control", "control selector mode drifted")
    _require(trigger.get("selector_secondary_calibration_environment_exists") is False, "control secondary environment exists")
    _require(trigger.get("formal_test_accessed") is False, "trigger accessed formal test")
    change = contract["single_engineering_change"]
    _require(change.get("name") == "accept_control_null_secondary_seed_against_control_semantics", "patch identity drifted")
    _require(change.get("acceptance_only") is True, "patch is not acceptance-only")
    _require(change.get("control_only") is True, "patch is not control-only")
    _require("control_expected_value" in change and change.get("control_expected_value") is None, "control expectation drifted")
    _require(change.get("control_omitted_field_allowed") is True, "omitted control field is not allowed")
    _require(change.get("candidate_expected_value") == "calibration_seed_start_plus_100", "candidate expectation drifted")
    _require(change.get("candidate_acceptance_changed") is False, "candidate acceptance changed")
    _require(change.get("delegated_frozen_checks_changed") is False, "frozen checks changed")
    _require(change.get("read_only_normalized_view") is True, "adapter is not read-only")
    _require(change.get("source_json_rewritten") is False, "adapter rewrites JSON")
    for key in (
        "training_changed",
        "return_estimator_changed",
        "selector_changed",
        "decoder_changed",
        "result_artifacts_changed",
        "scientific_gates_changed",
        "thresholds_changed",
        "seeds_or_traffic_changed",
        "result_root_changed",
    ):
        _require(change.get(key) is False, f"single_engineering_change.{key} drifted")
    _require(change.get("completed_promotion_job_rerun_forbidden") is True, "promotion rerun is not forbidden")
    _require(change.get("completed_promotion_artifact_rewrite_forbidden") is True, "promotion rewrite is not forbidden")
    verification = contract["verification"]
    for key in (
        "reproduce_original_rejection",
        "same_artifacts_patched_acceptance",
        "all_13_required_artifacts_hash_guarded",
        "reject_control_non_null_mutation",
        "reject_candidate_null_mutation",
        "preserve_existing_candidate_acceptance",
        "targeted_tests_required",
        "full_regression_required",
    ):
        _require(verification.get(key) is True, f"verification.{key} is not required")
    continuation = contract["continuation"]
    _require(continuation.get("next_job_after_adoption") == NEXT_PROMOTION_JOB, "next promotion job drifted")
    _require(continuation.get("promotion_order_unchanged") is True, "promotion order changed")
    _require(continuation.get("baseline_first_order_unchanged") is True, "baseline-first order changed")
    _require(continuation.get("scientific_implementation_hash_remains_parent_v4_8") is True, "scientific hash would change")
    _require(continuation.get("formal_test_locked") is True, "formal test was unlocked")
    return contract


def _validate_promotion_artifact_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    _require(set(PROMOTION_RUN_HASHES) == set(v48.FULL_RUN_REQUIRED_FILES), "required promotion file set drifted")
    for relative, expected in PROMOTION_RUN_HASHES.items():
        path = PROMOTION_RUN / relative
        _require(path.is_file(), f"promotion artifact missing: {relative}")
        digest = v48._sha256(path)
        _require(digest == expected, f"promotion artifact changed: {relative}")
        observed[relative] = digest
    return observed


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = v48._load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.8.3.control-secondary-seed-patch-freeze/v1", "invalid v4.8.3 freeze")
    _require(payload.get("formal_test_accessed") is False, "patch freeze accessed formal test")
    _require(payload.get("scientific_protocol_changed") is False, "patch freeze changed science")
    _require(payload.get("parent_v4_8_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "patch parent science drifted")
    _require(payload.get("parent_v4_8_2_runner_sha256") == PARENT_V482_RUNNER_SHA256, "parent runner drifted")
    _require(payload.get("parent_v4_8_2_freeze_sha256") == PARENT_V482_FREEZE_SHA256, "parent patch freeze drifted")
    _require(payload.get("patch_contract_sha256") == v48._sha256(DEFAULT_PATCH_CONTRACT), "patch contract hash drifted")
    _require(payload.get("preregistration_receipt_sha256") == v48._sha256(DEFAULT_PREREGISTRATION), "patch preregistration drifted")
    files = payload.get("patch_files")
    _require(isinstance(files, dict), "patch file map missing")
    for relative, expected in files.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"patch source missing: {relative}")
        _require(v48._sha256(source) == expected, f"patch source drifted: {relative}")
    _require(v48._canonical_sha256(files) == payload.get("patch_content_sha256"), "patch content digest invalid")
    parent_patch.validate_implementation_freeze(parent_patch.DEFAULT_FREEZE)
    _validate_promotion_artifact_hashes()
    return payload


def _validate_control_secondary_seed_semantics(
    root: Path,
    job: v48.Job,
    arguments: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    _require(job.algorithm == TEMPORAL_GRAPH, "control adapter received a non-control job")
    requested = arguments.get("requested_raw_steps")
    _require(isinstance(requested, dict), "requested_raw_steps is not a mapping")
    _require(
        requested.get("secondary_calibration_seed_start") is None,
        "control secondary_calibration_seed_start must be null",
    )
    fidelity = arguments.get("implementation_fidelity")
    _require(isinstance(fidelity, dict), "implementation_fidelity is not a mapping")
    _require(
        fidelity.get("secondary_calibration_seed_offset") is None,
        "control secondary_calibration_seed_offset must be null",
    )
    _require(
        fidelity.get("selector_mode") == "checkpoint_only_parent_control",
        "control fidelity selector mode drifted",
    )
    _require(
        receipt.get("schema_version")
        == "topo-scene-v4.8.checkpoint-decoder-selector-receipt/v1",
        "control selector schema drifted",
    )
    _require(receipt.get("selector_mode") == "checkpoint_only_parent_control", "control selector mode drifted")
    _require(not bool(receipt.get("secondary_calibration_triggered", False)), "control triggered secondary calibration")
    _require(receipt.get("secondary_calibration_seed_start") is None, "control selector secondary seed is non-null")
    _require(receipt.get("secondary_calibration_seed_offset") is None, "control selector secondary offset is non-null")
    _require(receipt.get("secondary_candidates") in (None, []), "control selector has secondary candidates")
    _require(not (root / "selector" / "cal_secondary").exists(), "control secondary calibration directory exists")


@contextmanager
def _normalized_control_argument_view(job: v48.Job) -> Iterator[None]:
    original_loader = v48._load_json
    target = (v48.run_directory(job) / "arguments.json").resolve()

    def load_with_normalized_control_seed(path: Path) -> dict[str, Any]:
        value = original_loader(path)
        if Path(path).resolve() != target:
            return value
        normalized = copy.deepcopy(value)
        requested = normalized.get("requested_raw_steps")
        _require(isinstance(requested, dict), "requested_raw_steps is not a mapping")
        _require(requested.get("secondary_calibration_seed_start") is None, "control seed changed during normalization")
        requested["secondary_calibration_seed_start"] = job.calibration_seed_start + 100
        return normalized

    try:
        v48._load_json = load_with_normalized_control_seed
        yield
    finally:
        v48._load_json = original_loader


@contextmanager
def _temporary_secondary_seed_view(job: v48.Job, value: int | None) -> Iterator[None]:
    """Expose an in-memory argument mutation for engineering guard tests only."""

    original_loader = v48._load_json
    target = (v48.run_directory(job) / "arguments.json").resolve()

    def load_with_override(path: Path) -> dict[str, Any]:
        payload = original_loader(path)
        if Path(path).resolve() != target:
            return payload
        overridden = copy.deepcopy(payload)
        requested = overridden.get("requested_raw_steps")
        _require(isinstance(requested, dict), "requested_raw_steps is not a mapping")
        requested["secondary_calibration_seed_start"] = value
        return overridden

    try:
        v48._load_json = load_with_override
        yield
    finally:
        v48._load_json = original_loader


def accepted_run_reason_v4_8_3(job: v48.Job, hashes: dict[str, str]) -> tuple[bool, str]:
    if job.algorithm != TEMPORAL_GRAPH:
        return _FROZEN_ACCEPTED_RUN_REASON(job, hashes)
    root = v48.run_directory(job)
    try:
        arguments = v48._load_json(root / "arguments.json")
        receipt = v48._load_json(root / "selector" / "receipt.json")
        _validate_control_secondary_seed_semantics(root, job, arguments, receipt)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    original_accepted, original_reason = _FROZEN_ACCEPTED_RUN_REASON(job, hashes)
    if original_accepted:
        return False, "frozen control acceptance trigger unexpectedly passed"
    if original_reason != "argument secondary_calibration_seed_start mismatch":
        return False, original_reason
    try:
        with _normalized_control_argument_view(job):
            return _FROZEN_ACCEPTED_RUN_REASON(job, hashes)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)


def accepted_run_v4_8_3(job: v48.Job, hashes: dict[str, str]) -> bool:
    return accepted_run_reason_v4_8_3(job, hashes)[0]


def _assert_stage_can_run_v4_8_3(hashes: dict[str, str], stage: str) -> None:
    v48.validate_implementation_freeze(PARENT_V48_FREEZE)
    parent_patch.validate_implementation_freeze(parent_patch.DEFAULT_FREEZE)
    validate_implementation_freeze(DEFAULT_FREEZE)
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing v4.8.3 engineering receipt")
    patch_receipt = v48._load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(patch_receipt.get("status") == "passed", "v4.8.3 engineering gate did not pass")
    _require(patch_receipt.get("implementation_freeze_sha256") == v48._sha256(DEFAULT_FREEZE), "v4.8.3 engineering freeze mismatch")
    parent_receipt = v48._load_json(parent_patch.DEFAULT_ENGINEERING_RECEIPT)
    _require(parent_receipt.get("status") == "passed", "parent v4.8.2 engineering gate did not pass")
    _require(v48._sha256(parent_patch.DEFAULT_ENGINEERING_RECEIPT) == PARENT_V482_RECEIPT_SHA256, "parent v4.8.2 receipt drifted")
    _require(hashes.get("implementation_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "scientific hash is not frozen v4.8")
    if stage == "promotion":
        decision = v48._load_json(v48.stage_root("development") / "development_decision.json")
        _require(decision.get("decision") == "pass", "development gate did not pass")
        _require(decision.get("implementation_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "development scientific freeze mismatch")
    if stage == "formal":
        receipt = v48._load_json(v48.DEFAULT_PROMOTION_GATE)
        _require(receipt.get("decision") == "pass", "formal test locked: promotion failed")
        _require(receipt.get("formal_algorithms") == list(V48_FORMAL_ALGORITHMS), "formal pair drifted")
        for key, value in hashes.items():
            if key != "stage0_attribution_sha256":
                _require(receipt.get(key) == value, f"formal test locked: {key} mismatch")


@contextmanager
def _patched_v4_8_3_interfaces() -> Iterator[None]:
    with parent_patch._patched_v4_8_2_interfaces():
        originals = {
            "accepted_run_reason": v48.accepted_run_reason,
            "accepted_run": v48.accepted_run,
            "_assert_stage_can_run": v48._assert_stage_can_run,
        }
        try:
            v48.accepted_run_reason = accepted_run_reason_v4_8_3
            v48.accepted_run = accepted_run_v4_8_3
            v48._assert_stage_can_run = _assert_stage_can_run_v4_8_3
            yield
        finally:
            for name, value in originals.items():
                setattr(v48, name, value)


def promotion_adoption_preflight() -> dict[str, Any]:
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    hashes = v48.protocol_hashes(freeze_path=PARENT_V48_FREEZE)
    promotion_jobs = v48.jobs_for_stage(contract, digest, "promotion")
    job = next(item for item in promotion_jobs if item.name == PROMOTION_JOB_NAME)
    candidate_job = v48.jobs_for_stage(contract, digest, "development")[0]
    _require(candidate_job.algorithm == V48_CANDIDATE, "candidate guard job drifted")
    before = _validate_promotion_artifact_hashes()
    with parent_patch._patched_v4_8_2_interfaces():
        original_accepted, original_reason = v48.accepted_run_reason(job, hashes)
    _require(original_accepted is False, "v4.8.2 unexpectedly accepts the promotion run")
    _require(original_reason == "argument secondary_calibration_seed_start mismatch", "v4.8.2 rejection reason drifted")
    with _patched_v4_8_3_interfaces():
        patched_accepted, patched_reason = v48.accepted_run_reason(job, hashes)
        candidate_accepted, candidate_reason = v48.accepted_run_reason(candidate_job, hashes)
        with _temporary_secondary_seed_view(job, job.calibration_seed_start + 100):
            corrupt_control_accepted, corrupt_control_reason = v48.accepted_run_reason(job, hashes)
        with _temporary_secondary_seed_view(candidate_job, None):
            corrupt_candidate_accepted, corrupt_candidate_reason = v48.accepted_run_reason(candidate_job, hashes)
    _require(patched_accepted is True and patched_reason == "accepted", f"patched promotion acceptance failed: {patched_reason}")
    _require(candidate_accepted is True and candidate_reason == "accepted", f"existing candidate acceptance changed: {candidate_reason}")
    _require(corrupt_control_accepted is False, "non-null control mutation was accepted")
    _require(corrupt_control_reason == "control secondary_calibration_seed_start must be null", "control mutation guard reason drifted")
    _require(corrupt_candidate_accepted is False, "null candidate mutation was accepted")
    _require(corrupt_candidate_reason == "argument secondary_calibration_seed_start mismatch", "candidate mutation guard reason drifted")
    after = _validate_promotion_artifact_hashes()
    _require(before == after, "promotion artifacts changed during adoption preflight")
    evaluation = v48._load_json(PROMOTION_RUN / "paper_evaluation_detailed.json")
    summary = evaluation["summary"]
    return {
        "frozen_v4_8_2_rejection_reproduced": True,
        "frozen_v4_8_2_reason": original_reason,
        "patched_runner_accepts_same_artifacts": True,
        "patched_runner_reason": patched_reason,
        "delegated_frozen_validator_after_single_read_only_normalization": True,
        "required_artifact_count": len(before),
        "promotion_artifact_hashes": before,
        "promotion_artifacts_modified": False,
        "promotion_job_rerun": False,
        "existing_candidate_acceptance_preserved": True,
        "control_non_null_mutation_rejected": True,
        "control_mutation_reason": corrupt_control_reason,
        "candidate_null_mutation_rejected": True,
        "candidate_mutation_reason": corrupt_candidate_reason,
        "validation_episodes": summary["episodes"],
        "validation_success_rate": summary["success_rate"],
        "validation_collision_rate": summary["collision_rate"],
        "validation_off_route_rate": summary["off_route_rate"],
        "validation_timeout_rate": summary["timeout_rate"],
        "next_promotion_job": NEXT_PROMOTION_JOB,
        "formal_test_accessed": False,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--patch-contract", type=Path, default=DEFAULT_PATCH_CONTRACT)
    root.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    plan = sub.add_parser("plan")
    plan.add_argument("--stage", choices=STAGES, required=True)
    plan.add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--job", default=None)
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
    validate_patch_contract(load_patch_contract(args.patch_contract.resolve()))
    v48.validate_contract(v48.load_contract())
    parent_patch.validate_patch_contract(parent_patch.load_patch_contract())
    parent_patch.validate_implementation_freeze(parent_patch.DEFAULT_FREEZE)
    if args.command == "validate":
        print(json.dumps({
            "status": "valid",
            "patch_id": PATCH_ID,
            "patch_contract_sha256": v48._sha256(args.patch_contract.resolve()),
            "parent_v4_8_freeze_sha256": v48._sha256(PARENT_V48_FREEZE),
            "parent_v4_8_2_freeze_sha256": v48._sha256(parent_patch.DEFAULT_FREEZE),
            "promotion_adoption_preflight": promotion_adoption_preflight(),
        }, ensure_ascii=False, indent=2))
        return 0
    validate_implementation_freeze(args.freeze.resolve())
    hashes = v48.protocol_hashes(freeze_path=PARENT_V48_FREEZE)
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    with (
        _patched_v4_8_3_interfaces(),
        v48._patched_scientific_runner(),
        parent_patch._patched_scientific_gate(),
    ):
        if args.command == "plan":
            value = v48.scientific.write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return v48.scientific.execute_jobs(
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
            value = v48.scientific.summarize_stage(contract, digest, args.stage, hashes)
        elif args.command == "gate":
            if args.stage == "development":
                value = v48.scientific.write_development_decision(contract, digest, hashes)
            elif args.stage == "promotion":
                value = v48.scientific.compute_promotion_gate(contract, digest, hashes)
            else:
                value = v48.scientific.compute_formal_decision(contract, digest, hashes)
        else:
            value = v48.scientific.status_payload(contract, digest, hashes)
    if isinstance(value, dict):
        value["engineering_patch_id"] = PATCH_ID
        value["engineering_patch_freeze_sha256"] = v48._sha256(args.freeze.resolve())
    print(json.dumps(v48._version_tree(value), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

