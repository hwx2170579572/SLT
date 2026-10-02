"""Apply the v4.8.2 selector-gate interface patch to frozen v4.8 science."""

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

from configs.sb3_configs_v4_8 import V48_CANDIDATE, V48_FORMAL_ALGORITHMS
from tools import run_topo_v4_8_experiments as v48
from tools import run_topo_v4_8_1_experiments as acceptance_patch
from tools.checkpoint_decoder_selector_v4_6 import CANDIDATE_DECODERS
from tools.checkpoint_decoder_selector_v4_8 import SELECTOR_MODE


PATCH_ID = "v4_8_2_align_candidate_gate_selector_mode"
DEFAULT_PATCH_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_8_2.yaml"
)
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_8_2_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_8_2_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
V481_FAILURE = PROJECT_ROOT / "results_topo_v4_8_1_dev" / "engineering" / "preflight_failure.json"
PARENT_V48_FREEZE = v48.DEFAULT_FREEZE
PARENT_V48_FREEZE_SHA256 = acceptance_patch.PARENT_V48_FREEZE_SHA256
V481_RUNNER_SHA256 = "ead5e5bb65c95464f5978e27686c3f3f89dd2a7bcc342186faaf9f6bc0a33795"
STAGES = v48.STAGES
_FROZEN_CANDIDATE_GATE = v48.scientific._candidate_gate


class V482PatchError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V482PatchError(message)


def load_patch_contract(path: Path = DEFAULT_PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.8.2 patch contract must be a mapping")
    return value


def validate_patch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.8.2.selector-gate-interface-engineering-patch/v1",
        "unexpected v4.8.2 patch schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test is not locked")
    _require(contract.get("formal_test_accessed") is False, "patch accessed formal test")
    parent = contract["parent_scientific_protocol"]
    _require(parent.get("contract_sha256") == v48.contract_sha256(), "parent contract drifted")
    _require(parent.get("implementation_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "parent freeze drifted")
    _require(parent.get("unchanged") is True, "parent science changed")
    prior = contract["parent_acceptance_patch"]
    _require(prior.get("runner_sha256") == V481_RUNNER_SHA256, "v4.8.1 runner drifted")
    _require(prior.get("acceptance_adapter_reused_unchanged") is True, "acceptance adapter is not frozen")
    _require(prior.get("overall_engineering_preflight_passed") is False, "v4.8.1 failure trigger drifted")
    trigger = contract["trigger"]
    _require(trigger.get("only_failed_check") == "selector_mode", "trigger is not isolated")
    _require(trigger.get("observed_selector_mode") == SELECTOR_MODE, "observed selector mode drifted")
    _require(trigger.get("inherited_expected_selector_mode") == "joint_checkpoint_decoder", "inherited selector mode drifted")
    _require(trigger.get("h1_outcome_gate_metrics_passed") is True, "H1 outcomes did not pass")
    _require(trigger.get("h1_mechanism_checks_passed") is True, "H1 mechanism did not pass")
    _require(trigger.get("h1_other_selector_checks_passed") is True, "other selector checks did not pass")
    _require(trigger.get("formal_test_accessed") is False, "trigger accessed formal test")
    change = contract["single_engineering_change"]
    _require(change.get("name") == "align_candidate_gate_selector_mode_with_frozen_v4_8_selector", "patch identity drifted")
    _require(change.get("old_expected_mode") == "joint_checkpoint_decoder", "old mode drifted")
    _require(change.get("new_expected_mode") == SELECTOR_MODE, "new mode drifted")
    _require(change.get("candidate_only") is True, "patch is not candidate-only")
    _require(change.get("acceptance_adapter_changed") is False, "acceptance adapter changed")
    for key in (
        "training_changed",
        "return_estimator_changed",
        "selector_changed",
        "decoder_changed",
        "result_artifacts_changed",
        "thresholds_changed",
        "seeds_or_traffic_changed",
        "result_root_changed",
    ):
        _require(change.get(key) is False, f"single_engineering_change.{key} drifted")
    _require(change.get("h1_rerun_forbidden") is True, "H1 rerun is not forbidden")
    _require(change.get("h1_artifact_rewrite_forbidden") is True, "H1 rewrite is not forbidden")
    continuation = contract["continuation"]
    _require(continuation.get("next_job_after_adoption") == "H2", "next job drifted")
    _require(continuation.get("development_order_unchanged") == ["H1", "H2", "H3", "H4"], "development order drifted")
    _require(continuation.get("scientific_implementation_hash_remains_parent_v4_8") is True, "scientific hash would change")
    return contract


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = v48._load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.8.2.selector-gate-patch-freeze/v1", "invalid v4.8.2 freeze")
    _require(payload.get("formal_test_accessed") is False, "patch freeze accessed formal test")
    _require(payload.get("scientific_protocol_changed") is False, "patch freeze changed science")
    _require(payload.get("parent_v4_8_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "patch parent freeze drifted")
    _require(payload.get("v4_8_1_runner_sha256") == V481_RUNNER_SHA256, "frozen acceptance adapter drifted")
    _require(payload.get("patch_contract_sha256") == v48._sha256(DEFAULT_PATCH_CONTRACT), "patch contract hash drifted")
    _require(payload.get("preregistration_receipt_sha256") == v48._sha256(DEFAULT_PREREGISTRATION), "patch preregistration drifted")
    files = payload.get("patch_files")
    _require(isinstance(files, dict), "patch file map missing")
    for relative, expected in files.items():
        source = PROJECT_ROOT / relative
        _require(source.is_file(), f"patch source missing: {relative}")
        _require(v48._sha256(source) == expected, f"patch source drifted: {relative}")
    _require(v48._canonical_sha256(files) == payload.get("patch_content_sha256"), "patch content digest invalid")
    v48.validate_implementation_freeze(PARENT_V48_FREEZE)
    acceptance_patch._validate_h1_hashes()
    return payload


def _candidate_gate_v4_8_2(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    gate = copy.deepcopy(_FROZEN_CANDIDATE_GATE(row, contract))
    _require(row.get("algorithm") == V48_CANDIDATE, "v4.8.2 gate adapter received a non-candidate row")
    gate["checks"]["selector_mode"] = row.get("selector_mode") == SELECTOR_MODE
    selector_keys = (
        "calibration_episode_count",
        "candidate_pair_count",
        "selector_mode",
        "selected_decoder",
        "receipt_predates_validation",
    )
    gate["selector_passed"] = all(gate["checks"].get(key) is True for key in selector_keys)
    gate["passed"] = bool(
        gate.get("outcome_passed")
        and gate.get("mechanism_passed")
        and gate.get("selector_passed")
    )
    gate["engineering_patch_id"] = PATCH_ID
    gate["only_changed_check"] = "selector_mode"
    gate["expected_selector_mode"] = SELECTOR_MODE
    return gate


def _assert_stage_can_run_v4_8_2(hashes: dict[str, str], stage: str) -> None:
    v48.validate_implementation_freeze(PARENT_V48_FREEZE)
    validate_implementation_freeze(DEFAULT_FREEZE)
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing v4.8.2 engineering receipt")
    patch_receipt = v48._load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(patch_receipt.get("status") == "passed", "v4.8.2 engineering gate did not pass")
    _require(patch_receipt.get("implementation_freeze_sha256") == v48._sha256(DEFAULT_FREEZE), "v4.8.2 engineering freeze mismatch")
    parent_receipt = v48._load_json(v48.DEFAULT_ENGINEERING_RECEIPT)
    _require(parent_receipt.get("status") == "passed", "parent v4.8 engineering gate did not pass")
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
def _patched_v4_8_2_interfaces() -> Iterator[None]:
    with acceptance_patch._patched_acceptance():
        original_assert = v48._assert_stage_can_run
        try:
            v48._assert_stage_can_run = _assert_stage_can_run_v4_8_2
            yield
        finally:
            v48._assert_stage_can_run = original_assert


@contextmanager
def _patched_scientific_gate() -> Iterator[None]:
    original = v48.scientific._candidate_gate
    try:
        v48.scientific._candidate_gate = _candidate_gate_v4_8_2
        yield
    finally:
        v48.scientific._candidate_gate = original


def h1_adoption_preflight() -> dict[str, Any]:
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    hashes = v48.protocol_hashes(freeze_path=PARENT_V48_FREEZE)
    job = v48.jobs_for_stage(contract, digest, "development")[0]
    original_accepted, original_reason = v48.accepted_run_reason(job, hashes)
    with acceptance_patch._patched_acceptance():
        accepted_481, reason_481 = v48.accepted_run_reason(job, hashes)
    _require(original_accepted is False and original_reason == "v4.8 changed return estimator", "original rejection did not reproduce")
    _require(accepted_481 is True and reason_481 == "accepted", "v4.8.1 acceptance did not reproduce")
    with _patched_v4_8_2_interfaces(), v48._patched_scientific_runner():
        row = v48.scientific._run_row(job)
        inherited_gate = _FROZEN_CANDIDATE_GATE(row, contract)
        patched_gate = _candidate_gate_v4_8_2(row, contract)
        with _patched_scientific_gate():
            status = v48.scientific.development_status(contract, digest, hashes)
    failed = [key for key, passed in inherited_gate["checks"].items() if not passed]
    _require(failed == ["selector_mode"], f"inherited gate failure is not isolated: {failed}")
    _require(inherited_gate.get("outcome_passed") is True, "H1 outcome gate failed")
    _require(inherited_gate.get("mechanism_passed") is True, "H1 mechanism gate failed")
    _require(patched_gate.get("passed") is True, "patched H1 gate failed")
    _require(status.get("decision") == "incomplete" and status.get("next_job") == "H2", "patched development continuation drifted")
    return {
        "original_v4_8_acceptance_rejection_reproduced": True,
        "original_reason": original_reason,
        "v4_8_1_acceptance_passed": True,
        "h1_artifact_hashes": acceptance_patch._validate_h1_hashes(),
        "inherited_gate_failed_checks": failed,
        "h1_outcome_gate_passed": inherited_gate["outcome_passed"],
        "h1_mechanism_gate_passed": inherited_gate["mechanism_passed"],
        "h1_other_selector_checks_passed": all(
            inherited_gate["checks"][key]
            for key in (
                "calibration_episode_count",
                "candidate_pair_count",
                "selected_decoder",
                "receipt_predates_validation",
            )
        ),
        "patched_selector_mode": row["selector_mode"],
        "patched_gate_passed": True,
        "next_development_job": status["next_job"],
        "h1_rerun": False,
        "h1_artifacts_modified": False,
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
    if args.command == "validate":
        print(json.dumps({
            "status": "valid",
            "patch_id": PATCH_ID,
            "patch_contract_sha256": v48._sha256(args.patch_contract.resolve()),
            "parent_v4_8_freeze_sha256": v48._sha256(PARENT_V48_FREEZE),
            "h1_adoption_preflight": h1_adoption_preflight(),
        }, ensure_ascii=False, indent=2))
        return 0
    validate_implementation_freeze(args.freeze.resolve())
    hashes = v48.protocol_hashes(freeze_path=PARENT_V48_FREEZE)
    contract = v48.validate_contract(v48.load_contract())
    digest = v48.contract_sha256()
    with _patched_v4_8_2_interfaces(), v48._patched_scientific_runner(), _patched_scientific_gate():
        if args.command == "plan":
            value = v48.scientific.write_plan(contract, digest, args.stage, hashes, device=args.device)
        elif args.command == "run":
            return v48.scientific.execute_jobs(contract, digest, args.stage, hashes, selector=args.job, device=args.device, workers=args.workers, dry_run=args.dry_run)
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
