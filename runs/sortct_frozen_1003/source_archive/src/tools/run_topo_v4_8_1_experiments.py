"""Apply the v4.8.1 acceptance-only patch to the frozen v4.8 runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sb3_configs_v4_8 import TEMPORAL_GRAPH, V48_CANDIDATE, V48_FORMAL_ALGORITHMS
from tools import run_topo_v4_8_experiments as v48


PATCH_ID = "v4_8_1_accept_emitted_runtime_diagnostic_interface"
DEFAULT_PATCH_CONTRACT = (
    PROJECT_ROOT / "experiments" / "topo_scene_v4" / "experiment_contract_v4_8_1.yaml"
)
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "results_topo_v4_8_1_dev" / "preregistration_receipt.json"
DEFAULT_ENGINEERING_ROOT = PROJECT_ROOT / "results_topo_v4_8_1_dev" / "engineering"
DEFAULT_FREEZE = DEFAULT_ENGINEERING_ROOT / "implementation_freeze.json"
DEFAULT_ENGINEERING_RECEIPT = DEFAULT_ENGINEERING_ROOT / "engineering_receipt.json"
PARENT_V48_FREEZE = v48.DEFAULT_FREEZE
PARENT_V48_FREEZE_SHA256 = "5827ca63caf5cfd187ac5ad0bdc2fd0fbb0fb00c324459007cf24620f0b57aa6"
H1_RUN = (
    PROJECT_ROOT
    / "results_topo_v4_8_dev"
    / "development"
    / "runs"
    / "H1__cand__cross__s8__p42f54135"
)
H1_HASHES = {
    "return_estimator_diagnostics.json": "2b656b852228d00970d2bd6fe4063614094dccf514d17c58afb1ab4d0b9ee08c",
    "training_diagnostics.json": "9a61b247331042552f112610f83225a77831ab2b68f22bfa77e203b5f8bcadb4",
    "paper_evaluation_detailed.json": "1b22046c621a87851f4af0d8dc8e527af19183e3539f7a7c0890cfb71bc1ef71",
    "selector/receipt.json": "40afb02053af79f89819c863145918bae67d0083660c5a71e7c00cf9ff5e5091",
}
STAGES = v48.STAGES


class V481PatchError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V481PatchError(message)


def load_patch_contract(path: Path = DEFAULT_PATCH_CONTRACT) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "v4.8.1 patch contract must be a mapping")
    return value


def validate_patch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(
        contract.get("schema_version")
        == "topo-scene-v4.8.1.runtime-diagnostic-acceptance-engineering-patch/v1",
        "unexpected v4.8.1 patch schema",
    )
    _require(contract.get("no_fabrication") is True, "no_fabrication must be true")
    _require(contract.get("formal_test_locked") is True, "formal test is not locked")
    _require(contract.get("formal_test_accessed") is False, "patch accessed formal test")
    parent = contract["parent_scientific_protocol"]
    _require(parent.get("contract_sha256") == v48.contract_sha256(), "parent contract drifted")
    _require(parent.get("implementation_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "parent freeze drifted")
    _require(parent.get("unchanged") is True, "parent science changed")
    trigger = contract["trigger"]
    _require(trigger.get("subprocess_returncode") == 0, "H1 scientific process failed")
    _require(trigger.get("frozen_runner_acceptance_reason") == "v4.8 changed return estimator", "trigger reason drifted")
    _require(trigger.get("formal_test_accessed") is False, "trigger accessed formal test")
    for relative, expected in H1_HASHES.items():
        key = relative.replace("/", "_").replace(".", "_") + "_sha256"
        explicit = {
            "return_estimator_diagnostics.json": "return_estimator_diagnostics_sha256",
            "training_diagnostics.json": "training_diagnostics_sha256",
            "paper_evaluation_detailed.json": "paper_evaluation_detailed_sha256",
            "selector/receipt.json": "selector_receipt_sha256",
        }[relative]
        _require(trigger.get(explicit) == expected, f"trigger {key} drifted")
    observed = contract["observed_interface"]
    _require(observed.get("runtime_diagnostic_has_return_estimator_changed") is True, "legacy field missing")
    _require(observed.get("runtime_diagnostic_return_estimator_changed") is True, "legacy reference drifted")
    _require(observed.get("runtime_diagnostic_has_return_estimator_changed_from_v4_7") is False, "invented runtime field unexpectedly exists")
    _require(observed.get("run_fidelity_return_estimator_changed_from_v4_7") is False, "run fidelity drifted")
    _require(observed.get("candidate_n_step") == 16, "candidate n-step drifted")
    _require(observed.get("candidate_bootstrap_discount") == "gamma_power_actual_horizon", "candidate bootstrap drifted")
    change = contract["single_engineering_change"]
    _require(change.get("name") == "accept_runtime_diagnostic_against_its_emitted_legacy_reference_field", "patch identity drifted")
    _require(change.get("acceptance_only") is True, "patch is not acceptance-only")
    for key in (
        "training_changed",
        "return_estimator_changed",
        "selector_changed",
        "decoder_changed",
        "result_artifacts_changed",
        "scientific_gates_changed",
        "seeds_or_traffic_changed",
        "result_root_changed",
    ):
        _require(change.get(key) is False, f"single_engineering_change.{key} drifted")
    _require(change.get("h1_rerun_forbidden") is True, "H1 rerun is not forbidden")
    _require(change.get("h1_artifact_rewrite_forbidden") is True, "H1 rewrite is not forbidden")
    continuation = contract["continuation"]
    _require(continuation.get("development_order_unchanged") == ["H1", "H2", "H3", "H4"], "development order drifted")
    _require(continuation.get("next_job_after_adoption") == "H2", "next job drifted")
    _require(continuation.get("scientific_implementation_hash_remains_parent_v4_8") is True, "scientific hash would change")
    return contract


def _validate_h1_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in H1_HASHES.items():
        path = H1_RUN / relative
        _require(path.is_file(), f"H1 artifact missing: {relative}")
        digest = v48._sha256(path)
        _require(digest == expected, f"H1 artifact changed: {relative}")
        observed[relative] = digest
    return observed


def validate_implementation_freeze(path: Path = DEFAULT_FREEZE) -> dict[str, Any]:
    payload = v48._load_json(path.resolve())
    _require(payload.get("schema_version") == "topo-scene-v4.8.1.acceptance-patch-freeze/v1", "invalid v4.8.1 freeze")
    _require(payload.get("formal_test_accessed") is False, "patch freeze accessed formal test")
    _require(payload.get("scientific_protocol_changed") is False, "patch freeze changed science")
    _require(payload.get("parent_v4_8_freeze_sha256") == PARENT_V48_FREEZE_SHA256, "patch parent freeze drifted")
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
    _validate_h1_hashes()
    return payload


def _validate_return_estimator_v4_8_1(
    root: Path, job: v48.Job, training: dict[str, Any]
) -> dict[str, Any]:
    diagnostics = v48._load_json(root / "return_estimator_diagnostics.json")
    _require(training.get("return_estimator") == diagnostics, "return diagnostics are not bound to training")
    _require(diagnostics.get("schema_version") == "topo-scene-v4.8.return-estimator-diagnostics/v1", "return schema drifted")
    _require(diagnostics.get("algorithm") == job.algorithm, "return algorithm drifted")
    _require(diagnostics.get("implementation_id") == job.implementation_id, "return implementation drifted")
    _require(abs(float(diagnostics.get("gamma")) - 0.99) <= 1e-12, "return gamma drifted")
    _require(diagnostics.get("return_estimator_changed_from_v4_7") in (None, False), "runtime diagnostic explicitly changed from v4.7")
    if job.algorithm == V48_CANDIDATE:
        _require(diagnostics.get("return_estimator_changed") is True, "candidate legacy return role drifted")
        _require(diagnostics.get("n_step") == 16, "candidate n-step drifted")
        _require(diagnostics.get("horizon_correct_bootstrap") is True, "candidate bootstrap is not horizon-correct")
        _require(diagnostics.get("bootstrap_discount") == "gamma_power_actual_horizon", "candidate bootstrap discount drifted")
        _require(int(diagnostics.get("sampled_horizon_count", 0)) > 0, "candidate sampled no replay horizons")
        mean_horizon = float(diagnostics["mean_sampled_actual_horizon"])
        short_rate = float(diagnostics["short_horizon_sample_rate"])
        _require(math.isfinite(mean_horizon) and 1.0 <= mean_horizon <= 16.0, "candidate mean horizon invalid")
        _require(math.isfinite(short_rate) and 0.0 <= short_rate <= 1.0, "candidate short-horizon rate invalid")
    elif job.algorithm == TEMPORAL_GRAPH:
        _require(diagnostics.get("return_estimator_changed") is False, "control return role drifted")
        _require(diagnostics.get("n_step") == 4, "control n-step drifted")
        _require(diagnostics.get("horizon_correct_bootstrap") is False, "control bootstrap drifted")
        _require(diagnostics.get("bootstrap_discount") == "single_gamma_source_equivalent", "control bootstrap discount drifted")
    else:
        raise V481PatchError(f"unsupported algorithm {job.algorithm!r}")
    return diagnostics


def _assert_stage_can_run_v4_8_1(hashes: dict[str, str], stage: str) -> None:
    v48.validate_implementation_freeze(PARENT_V48_FREEZE)
    validate_implementation_freeze(DEFAULT_FREEZE)
    _require(DEFAULT_ENGINEERING_RECEIPT.is_file(), "missing v4.8.1 engineering receipt")
    patch_receipt = v48._load_json(DEFAULT_ENGINEERING_RECEIPT)
    _require(patch_receipt.get("status") == "passed", "v4.8.1 engineering gate did not pass")
    _require(patch_receipt.get("implementation_freeze_sha256") == v48._sha256(DEFAULT_FREEZE), "v4.8.1 engineering freeze mismatch")
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
def _patched_acceptance() -> Iterator[None]:
    originals = {
        "_validate_return_estimator": v48._validate_return_estimator,
        "_assert_stage_can_run": v48._assert_stage_can_run,
    }
    try:
        v48._validate_return_estimator = _validate_return_estimator_v4_8_1
        v48._assert_stage_can_run = _assert_stage_can_run_v4_8_1
        yield
    finally:
        for name, value in originals.items():
            setattr(v48, name, value)


def h1_adoption_preflight() -> dict[str, Any]:
    contract = v48.validate_contract(v48.load_contract())
    hashes = v48.protocol_hashes(freeze_path=PARENT_V48_FREEZE)
    job = v48.jobs_for_stage(contract, v48.contract_sha256(), "development")[0]
    original_accepted, original_reason = v48.accepted_run_reason(job, hashes)
    with _patched_acceptance():
        patched_accepted, patched_reason = v48.accepted_run_reason(job, hashes)
    _require(original_accepted is False, "frozen runner unexpectedly accepts H1")
    _require(original_reason == "v4.8 changed return estimator", "frozen rejection reason drifted")
    _require(patched_accepted is True and patched_reason == "accepted", f"patched H1 acceptance failed: {patched_reason}")
    detailed = v48._load_json(H1_RUN / "paper_evaluation_detailed.json")
    gate = v48.scientific._candidate_gate(
        {
            **detailed["summary"],
            "algorithm": V48_CANDIDATE,
            "return_n_step": 16,
            "return_bootstrap_discount": "gamma_power_actual_horizon",
            "hybrid_exact_lateral_code_rate": 1.0,
            "selected_decoder_records": 1,
            "selected_decoder_exact_rule_match_rate": 1.0,
            "selected_decoder_exact_action_match_rate": 1.0,
            "selected_action_mask_feasible_rate": 1.0,
        },
        contract,
    )
    _require(gate.get("passed") is True, "H1 outcome gate did not pass")
    return {
        "frozen_runner_reproduced_rejection": True,
        "frozen_runner_reason": original_reason,
        "patched_runner_accepts_same_artifacts": True,
        "patched_runner_reason": patched_reason,
        "h1_artifact_hashes": _validate_h1_hashes(),
        "h1_outcome_gate_passed": True,
        "h1_success_rate": detailed["summary"]["success_rate"],
        "h1_collision_rate": detailed["summary"]["collision_rate"],
        "h1_off_route_rate": detailed["summary"]["off_route_rate"],
        "h1_timeout_rate": detailed["summary"]["timeout_rate"],
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
    with _patched_acceptance(), v48._patched_scientific_runner():
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
