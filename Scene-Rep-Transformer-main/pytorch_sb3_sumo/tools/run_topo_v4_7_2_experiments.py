"""Run v4.7 with both frozen engineering patches and isolated parent context."""

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

from tools import run_topo_v4_7_1_experiments as parent_patch
from tools import run_topo_v4_7_experiments as scientific


PATCH_ID = "v4_7_2_preserve_parent_preregistration_context"
DEFAULT_PATCH_CONTRACT = PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_7_2.yaml"
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_7_2_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_7_2_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
DEFAULT_PROMOTION_GATE = PROJECT_ROOT / "results_topo_v4_7_2_promotion" / "promotion_gate.json"
PARENT_V471_FREEZE = PROJECT_ROOT / "results_topo_v4_7_1_dev" / "engineering" / "implementation_freeze.json"

PATCH_CONTRACT_SHA256 = "672f258ea9715313e9111f0454366582d52f99fd7a1a95757ccbe7a687763e6c"
PATCH_PLAN_SHA256 = "b77a390efecf5ca08e8e5c48a6bae8b1dc5c8a149308f66a805c39acd283fd82"
PARENT_V471_FREEZE_SHA256 = "979080cc6a76936f572cc79d730ce23202803be976f201ab61810967b1b93359"
STAGES = scientific.STAGES
Job = scientific.Job


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise scientific.V47ProtocolError(message)


def load_patch_contract(path: Path = DEFAULT_PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.7.2 patch contract must be a mapping")
    return value


def validate_patch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.7.2.runner-context-engineering-patch/v1",
        "unexpected v4.7.2 patch schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test is not locked")
    _require(contract.get("formal_test_accessed") is False, "patch accessed formal test")
    science = contract["scientific_protocol"]
    _require(science.get("unchanged") is True, "scientific protocol changed")
    _require(science.get("contract_sha256") == scientific.contract_sha256(), "scientific contract drifted")
    _require(science.get("v4_7_implementation_freeze_sha256") == parent_patch.PARENT_V47_FREEZE_SHA256, "v4.7 freeze drifted")
    for key in ("method_changed", "return_estimator_changed", "seeds_or_traffic_changed", "gates_changed"):
        _require(science.get(key) is False, f"scientific_protocol.{key} drifted")
    parent = contract["parent_engineering_patch"]
    _require(parent.get("implementation_freeze_sha256") == PARENT_V471_FREEZE_SHA256, "v4.7.1 freeze drifted")
    _require(parent.get("training_adapter_changed") is False, "training adapter change drifted")
    _require(parent.get("metadata_binding_writer_changed") is False, "metadata writer change drifted")
    trigger = contract["trigger"]
    _require(trigger.get("scientific_job_started") is False, "v4.7.1 scientific job unexpectedly started")
    _require(trigger.get("formal_test_accessed") is False, "trigger accessed formal test")
    change = contract["single_engineering_change"]
    _require(change.get("name") == "preserve_parent_default_preregistration_inside_runner_context", "runner patch drifted")
    _require(change.get("removed_context_override") == "DEFAULT_PREREGISTRATION", "removed override drifted")
    _require(change.get("command_or_training_logic_changed") is False, "runner patch changed training logic")
    retry = contract["retry"]
    _require(retry.get("from_scratch") is True and retry.get("allowed_attempts") == 1, "retry protocol drifted")
    _require(retry.get("training_adapter") == "tools/train_paper_sb3_sumo_v4_7_1.py", "training adapter drifted")
    _require(retry.get("ordered_cells_unchanged") == ["G1", "G2", "G3", "G4", "G5"], "cell order drifted")
    return contract


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = scientific._load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.7.2.runner-context-freeze/v1", "invalid v4.7.2 freeze")
    _require(payload.get("formal_test_accessed") is False, "freeze accessed formal test")
    _require(payload.get("scientific_contract_sha256") == scientific.contract_sha256(), "freeze scientific contract drifted")
    _require(payload.get("parent_v4_7_1_freeze_sha256") == PARENT_V471_FREEZE_SHA256, "freeze parent patch drifted")
    _require(payload.get("patch_contract_sha256") == PATCH_CONTRACT_SHA256, "freeze patch contract drifted")
    _require(payload.get("preregistration_receipt_sha256") == scientific._sha256(DEFAULT_PREREGISTRATION), "freeze preregistration drifted")
    files = payload.get("patch_files")
    _require(isinstance(files, dict), "v4.7.2 patch files missing")
    for relative, expected in files.items():
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"v4.7.2 patch file missing: {relative}")
        _require(scientific._sha256(path) == expected, f"v4.7.2 patch file drifted: {relative}")
    _require(scientific._canonical_sha256(files) == payload.get("patch_content_sha256"), "patch content digest invalid")
    parent_patch.validate_implementation_freeze(PARENT_V471_FREEZE)
    return payload


def protocol_hashes(*, freeze_path: Path = DEFAULT_FREEZE) -> dict[str, str]:
    scientific.validate_contract(scientific.load_contract())
    parent_patch.validate_patch_contract(parent_patch.load_patch_contract())
    validate_patch_contract(load_patch_contract())
    hashes = {
        "experiment_contract_sha256": scientific.contract_sha256(),
        **scientific.parent._validate_stage0(
            scientific.DEFAULT_STAGE0_RESULTS, scientific.DEFAULT_STAGE0_ATTRIBUTION
        ),
        **scientific._validate_failure_attribution(
            scientific.DEFAULT_FAILURE_ATTRIBUTION,
            scientific.DEFAULT_DEEP_ATTRIBUTION,
        ),
        "preregistration_receipt_sha256": scientific._sha256(
            DEFAULT_PREREGISTRATION
        ),
        "engineering_patch_contract_sha256": PATCH_CONTRACT_SHA256,
    }
    validate_implementation_freeze(freeze_path.resolve())
    hashes["implementation_freeze_sha256"] = scientific._sha256(
        freeze_path.resolve()
    )
    return hashes


def stage_root(stage: str) -> Path:
    return {
        "development": PROJECT_ROOT / "results_topo_v4_7_2_dev" / "development",
        "promotion": PROJECT_ROOT / "results_topo_v4_7_2_promotion",
        "formal": PROJECT_ROOT / "results_topo_v4_7_2_formal",
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
    command = parent_patch.command_for(
        job, hashes, device=device, promotion_gate=promotion_gate
    )
    command[command.index("--output-dir") + 1] = str(
        (stage_root(job.stage) / "runs").resolve()
    )
    if job.stage == "formal":
        command[command.index("--formal-unlock-receipt") + 1] = str(
            promotion_gate.resolve()
        )
    return command


@contextmanager
def _patched_scientific_runner() -> Iterator[None]:
    """Patch orchestration paths without shadowing the parent preregistration."""

    replacements = {
        "DEFAULT_ENGINEERING_ROOT": DEFAULT_ENGINEERING_ROOT,
        "DEFAULT_FREEZE": DEFAULT_FREEZE,
        "DEFAULT_ENGINEERING_RECEIPT": DEFAULT_ENGINEERING_RECEIPT,
        "DEFAULT_PROMOTION_GATE": DEFAULT_PROMOTION_GATE,
        "stage_root": stage_root,
        "run_directory": run_directory,
        "command_for": command_for,
        "validate_implementation_freeze": validate_implementation_freeze,
    }
    originals = {name: getattr(scientific, name) for name in replacements}
    preregistration_identity = scientific.DEFAULT_PREREGISTRATION
    try:
        for name, value in replacements.items():
            setattr(scientific, name, value)
        _require(
            scientific.DEFAULT_PREREGISTRATION is preregistration_identity,
            "parent preregistration was shadowed",
        )
        yield
    finally:
        for name, value in originals.items():
            setattr(scientific, name, value)


def jobs_for_stage(contract: dict[str, Any], digest: str, stage: str) -> list[Job]:
    return scientific.jobs_for_stage(contract, digest, stage)


def corrected_context_preflight() -> dict[str, Any]:
    before = scientific.DEFAULT_PREREGISTRATION
    with _patched_scientific_runner():
        _require(scientific.DEFAULT_PREREGISTRATION is before, "parent preregistration identity changed")
        parent = parent_patch._FROZEN_VALIDATE_IMPLEMENTATION_FREEZE(
            parent_patch.PARENT_V47_FREEZE
        )
        contract = scientific.validate_contract(scientific.load_contract())
        digest = scientific.contract_sha256()
        job = scientific.jobs_for_stage(contract, digest, "development")[0]
        command = command_for(
            job,
            {
                "experiment_contract_sha256": "a" * 64,
                "stage0_results_sha256": "b" * 64,
                "failure_attribution_sha256": "c" * 64,
                "implementation_freeze_sha256": "d" * 64,
            },
            device="cuda",
        )
    return {
        "parent_preregistration_identity_preserved": True,
        "parent_v4_7_freeze_revalidated": parent["schema_version"]
        == "topo-scene-v4.7.implementation-freeze/v1",
        "next_job": job.job_id,
        "trainer": Path(command[1]).name,
        "output_root": str(stage_root("development").resolve()),
        "formal_test_accessed": False,
    }


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
    contract = scientific.validate_contract(scientific.load_contract())
    validate_patch_contract(load_patch_contract())
    digest = scientific.contract_sha256()
    if args.command == "validate":
        print(json.dumps({
            "status": "valid",
            "scientific_contract_sha256": digest,
            "patch_contract_sha256": scientific._sha256(DEFAULT_PATCH_CONTRACT),
            "parent_v4_7_1_freeze_sha256": scientific._sha256(PARENT_V471_FREEZE),
            "context_preflight": corrected_context_preflight(),
        }, ensure_ascii=False, indent=2)); return 0
    hashes = protocol_hashes(freeze_path=args.freeze)
    with _patched_scientific_runner():
        if args.command == "plan":
            value = scientific.write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return scientific.execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
        elif args.command == "summarize":
            value = scientific.summarize_stage(contract, digest, args.stage, hashes)
        elif args.command == "gate":
            if args.stage == "development": value = scientific.write_development_decision(contract, digest, hashes)
            elif args.stage == "promotion": value = scientific.compute_promotion_gate(contract, digest, hashes)
            else: value = scientific.compute_formal_decision(contract, digest, hashes)
        else:
            value = scientific.status_payload(contract, digest, hashes)
    if isinstance(value, dict):
        value["engineering_patch_id"] = PATCH_ID
    print(json.dumps(value, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
