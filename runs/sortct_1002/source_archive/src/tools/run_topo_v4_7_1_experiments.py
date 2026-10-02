"""Run the frozen v4.7 protocol through the v4.7.1 engineering patch."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import run_topo_v4_7_experiments as frozen


PATCH_ID = "v4_7_1_bind_sealed_selected_deployment_fields"
DEFAULT_SCIENTIFIC_CONTRACT = frozen.DEFAULT_CONTRACT
DEFAULT_PATCH_CONTRACT = PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_7_1.yaml"
DEFAULT_PATCH_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_7_1_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_7_1_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = PROJECT_ROOT / "results_topo_v4_7_1_promotion" / "promotion_gate.json"
PARENT_V47_FREEZE = PROJECT_ROOT / "results_topo_v4_7_dev" / "engineering" / "implementation_freeze.json"

PATCH_CONTRACT_SHA256 = "6e27c933130f3a33a62eb294fabab7f06af0851703faa2b1419087db0191aa8a"
PATCH_PLAN_SHA256 = "2fe008eeb9dfbea5c9c4308ab39de79e576eac76056569f9bfd57677b6fbde57"
PARENT_V47_FREEZE_SHA256 = "eb4526f1c9b197c2dc8259af8f3b7e2eea9c6b0b22de05054f54957fbf0082b0"
REJECTED_DETAILED_SHA256 = "6ef859c5737aa91d47523759095f5bb0bc37838f6983efc3a3555cbc2ad3364c"
REJECTED_SELECTOR_SHA256 = "ffdfa9540f5a35d20f9e5f43819e5336b2a52a903a40b53db5ea7fe2e50379c8"

STAGES = frozen.STAGES
Job = frozen.Job
_FROZEN_COMMAND_FOR = frozen.command_for
_FROZEN_VALIDATE_IMPLEMENTATION_FREEZE = frozen.validate_implementation_freeze


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise frozen.V47ProtocolError(message)


def load_patch_contract(path: Path = DEFAULT_PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.7.1 patch contract must be a mapping")
    return value


def validate_patch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.7.1.metadata-binding-engineering-patch/v1",
        "unexpected v4.7.1 patch schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test must be locked")
    _require(contract.get("formal_test_accessed") is False, "patch accessed formal test")
    scientific = contract["scientific_protocol"]
    _require(scientific.get("unchanged") is True, "scientific protocol changed")
    _require(
        scientific.get("contract_sha256") == frozen.contract_sha256(),
        "scientific contract hash drifted",
    )
    _require(
        scientific.get("implementation_freeze_sha256")
        == PARENT_V47_FREEZE_SHA256,
        "parent v4.7 freeze hash drifted",
    )
    for key in (
        "method_changed",
        "return_estimator_changed",
        "reward_changed",
        "network_changed",
        "optimizer_changed",
        "selector_changed",
        "decoder_changed",
        "seeds_or_traffic_changed",
        "gates_changed",
    ):
        _require(scientific.get(key) is False, f"scientific_protocol.{key} drifted")
    rejected = contract["rejected_attempt"]
    _require(rejected.get("status") == "engineering_rejected_not_gate_eligible", "rejected attempt status drifted")
    _require(rejected.get("validation_outcome_inspected_for_patch_design") is False, "rejected outcome was inspected")
    _require(rejected.get("validation_outcome_gate_eligible") is False, "rejected outcome became gate eligible")
    _require(rejected.get("detailed_sha256") == REJECTED_DETAILED_SHA256, "rejected detailed hash drifted")
    _require(rejected.get("selector_receipt_sha256") == REJECTED_SELECTOR_SHA256, "rejected selector hash drifted")
    change = contract["single_engineering_change"]
    _require(change.get("name") == "bind_sealed_selected_deployment_fields_into_detailed_evaluation", "patch change drifted")
    _require(change.get("source_of_truth") == "selector_receipt_sealed_before_validation", "binding source drifted")
    _require(
        change.get("fields")
        == [
            "selected_deployment_decoder",
            "selected_source_checkpoint_sha256",
            "selected_model_policy_class",
            "selected_model_parameter_state_sha256",
        ],
        "binding fields drifted",
    )
    retry = contract["retry"]
    _require(retry.get("from_scratch") is True, "retry is not from scratch")
    _require(retry.get("allowed_attempts") == 1, "retry count drifted")
    for key in (
        "reuse_parent_checkpoint",
        "reuse_parent_replay",
        "reuse_parent_calibration",
        "reuse_parent_validation",
    ):
        _require(retry.get(key) is False, f"retry.{key} drifted")
    _require(retry.get("ordered_cells_unchanged") == ["G1", "G2", "G3", "G4", "G5"], "cell order drifted")
    return contract


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = frozen._load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.7.1.engineering-patch-freeze/v1", "invalid v4.7.1 freeze")
    _require(payload.get("formal_test_accessed") is False, "patch freeze accessed formal test")
    _require(payload.get("scientific_contract_sha256") == frozen.contract_sha256(), "patch freeze scientific contract drifted")
    _require(payload.get("parent_v4_7_implementation_freeze_sha256") == PARENT_V47_FREEZE_SHA256, "patch parent freeze drifted")
    _require(payload.get("patch_contract_sha256") == PATCH_CONTRACT_SHA256, "patch contract hash drifted")
    _require(payload.get("patch_preregistration_sha256") == frozen._sha256(DEFAULT_PATCH_PREREGISTRATION), "patch preregistration hash drifted")
    files = payload.get("patch_files")
    _require(isinstance(files, dict), "patch file map missing")
    for relative, expected in files.items():
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"patch file missing: {relative}")
        _require(frozen._sha256(path) == expected, f"patch file drifted: {relative}")
    _require(frozen._canonical_sha256(files) == payload.get("patch_content_sha256"), "patch content digest invalid")
    parent = _FROZEN_VALIDATE_IMPLEMENTATION_FREEZE(PARENT_V47_FREEZE)
    _require(parent.get("scientific_content_sha256") == payload.get("parent_v4_7_scientific_content_sha256"), "parent scientific content drifted")
    return payload


def protocol_hashes(*, freeze_path: Path = DEFAULT_FREEZE) -> dict[str, str]:
    frozen.validate_contract(frozen.load_contract(DEFAULT_SCIENTIFIC_CONTRACT))
    validate_patch_contract(load_patch_contract())
    _require(frozen._sha256(DEFAULT_PATCH_CONTRACT) == PATCH_CONTRACT_SHA256, "patch contract file hash drifted")
    hashes = {
        "experiment_contract_sha256": frozen._sha256(DEFAULT_SCIENTIFIC_CONTRACT),
        **frozen.parent._validate_stage0(
            frozen.DEFAULT_STAGE0_RESULTS, frozen.DEFAULT_STAGE0_ATTRIBUTION
        ),
        **frozen._validate_failure_attribution(
            frozen.DEFAULT_FAILURE_ATTRIBUTION, frozen.DEFAULT_DEEP_ATTRIBUTION
        ),
        "preregistration_receipt_sha256": frozen._sha256(
            DEFAULT_PATCH_PREREGISTRATION
        ),
        "engineering_patch_contract_sha256": PATCH_CONTRACT_SHA256,
    }
    validate_implementation_freeze(freeze_path.resolve())
    hashes["implementation_freeze_sha256"] = frozen._sha256(freeze_path.resolve())
    return hashes


def stage_root(stage: str) -> Path:
    return {
        "development": PROJECT_ROOT / "results_topo_v4_7_1_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_7_1_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_7_1_formal",
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
    command = _FROZEN_COMMAND_FOR(
        job, hashes, device=device, promotion_gate=promotion_gate
    )
    command[1] = str(PROJECT_ROOT / "tools" / "train_paper_sb3_sumo_v4_7_1.py")
    output_index = command.index("--output-dir") + 1
    command[output_index] = str((stage_root(job.stage) / "runs").resolve())
    if job.stage == "formal":
        receipt_index = command.index("--formal-unlock-receipt") + 1
        command[receipt_index] = str(promotion_gate.resolve())
    return command


@contextmanager
def _patched_frozen_runner() -> Iterator[None]:
    replacements = {
        "DEFAULT_ENGINEERING_ROOT": DEFAULT_ENGINEERING_ROOT,
        "DEFAULT_FREEZE": DEFAULT_FREEZE,
        "DEFAULT_ENGINEERING_RECEIPT": DEFAULT_ENGINEERING_RECEIPT,
        "DEFAULT_PROMOTION_GATE": DEFAULT_PROMOTION_GATE,
        "DEFAULT_PREREGISTRATION": DEFAULT_PATCH_PREREGISTRATION,
        "stage_root": stage_root,
        "run_directory": run_directory,
        "command_for": command_for,
        "validate_implementation_freeze": validate_implementation_freeze,
    }
    originals = {name: getattr(frozen, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(frozen, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(frozen, name, value)


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    return frozen.jobs_for_stage(contract, digest, stage)


def accepted_run_reason(job: Job, hashes: dict[str, str]) -> tuple[bool, str]:
    with _patched_frozen_runner():
        return frozen.accepted_run_reason(job, hashes)


def development_status(contract: dict[str, Any], digest: str, hashes: dict[str, str]) -> dict[str, Any]:
    with _patched_frozen_runner():
        value = frozen.development_status(contract, digest, hashes)
    value["engineering_patch_id"] = PATCH_ID
    return value


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
    with _patched_frozen_runner():
        return frozen.execute_jobs(
            contract,
            digest,
            stage,
            hashes,
            selector=selector,
            device=device,
            workers=workers,
            dry_run=dry_run,
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
    contract = frozen.validate_contract(frozen.load_contract(DEFAULT_SCIENTIFIC_CONTRACT))
    patch_contract = validate_patch_contract(load_patch_contract())
    del patch_contract
    digest = frozen.contract_sha256(DEFAULT_SCIENTIFIC_CONTRACT)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "valid",
                    "scientific_contract_sha256": digest,
                    "engineering_patch_contract_sha256": frozen._sha256(
                        DEFAULT_PATCH_CONTRACT
                    ),
                    "parent_v4_7_implementation_freeze_sha256": frozen._sha256(
                        PARENT_V47_FREEZE
                    ),
                    "rejected_attempt_gate_eligible": False,
                    "formal_test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    hashes = protocol_hashes(freeze_path=args.freeze)
    with _patched_frozen_runner():
        if args.command == "plan":
            value = frozen.write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return frozen.execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
        elif args.command == "summarize":
            value = frozen.summarize_stage(contract, digest, args.stage, hashes)
        elif args.command == "gate":
            if args.stage == "development": value = frozen.write_development_decision(contract, digest, hashes)
            elif args.stage == "promotion": value = frozen.compute_promotion_gate(contract, digest, hashes)
            else: value = frozen.compute_formal_decision(contract, digest, hashes)
        else:
            value = frozen.status_payload(contract, digest, hashes)
    if isinstance(value, dict):
        value["engineering_patch_id"] = PATCH_ID
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
