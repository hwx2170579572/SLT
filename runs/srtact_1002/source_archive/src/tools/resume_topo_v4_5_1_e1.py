"""Resume the preregistered E1 run after the v4.5 selector boundary failure.

This entry point never trains and never recalibrates.  It verifies the frozen E1
inputs, seals the v4.5.1 selector receipt, and only then constructs the original
v4.5 validation environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
from configs.sb3_configs_v4_5 import V45_IMPLEMENTATION_IDS
from tools.action_diagnostics_v4_5 import evaluate_with_action_diagnostics_v4_5
from tools.checkpoint_selector_v4_5_1 import select_checkpoint, sha256
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.paper_evaluation_contract_v2 import (
    build_evaluation_provenance_v2,
    callable_name,
)
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5
from tools.train_sb3_v4_5 import (
    _seal_selector_then_construct_evaluation_env,
    _write_json,
)


E1_NAME = "E1__cand__cross__s4__p3c1f7f8b"
BASE_CONTRACT_SHA256 = (
    "3c1f7f8bb2f2967f7559907386b58e67760c53dfa8e46bf581fc4824d186bb7c"
)
BASE_FREEZE_SHA256 = (
    "9ab0ccc71b33110ac526f5a9f89e8f504d0f3fc0ddafd6407c6ff36dc2414666"
)
PREREG_SHA256 = ""
EXPECTED_INPUT_HASHES = {
    "arguments.json": (
        "68058cf9f8ef42da160cdf8610b8b230245025d5ff88d5f7e2d53cdcce97e509"
    ),
    "best_training_success_model.zip": (
        "5f324d6f68a46eea527983b589d89960ba72fe40a9970521403d12665be125d6"
    ),
    "final_model.zip": (
        "2d9afa0420fd78e6dec7b60f8b5fb53f5139ccdbb63affe8bd3a29fcc30b7519"
    ),
    "selector/cal/best/detailed.json": (
        "748a504f01db7fe8b07b144f7e0e115825ab57812176b153cf5f51ee40332710"
    ),
    "selector/cal/final/detailed.json": (
        "e729ff915e5cc8a423959706a0635fa81af76e09982796726da1fabc3058a21c"
    ),
}
FORBIDDEN_PREVALIDATION_ARTIFACTS = (
    "selector/receipt.json",
    "selected_model.zip",
    "paper_evaluation_detailed.json",
    "final_evaluation.json",
    "action_diagnostics.json",
    "action_diagnostics_decisions.jsonl",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_preregistered_inputs(run_dir: Path) -> dict[str, Any]:
    _require(run_dir.name == E1_NAME, f"recovery is restricted to {E1_NAME}")
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = run_dir / relative
        _require(path.is_file(), f"missing preregistered E1 input: {relative}")
        _require(sha256(path) == expected, f"preregistered E1 hash drift: {relative}")
    for relative in FORBIDDEN_PREVALIDATION_ARTIFACTS:
        _require(
            not (run_dir / relative).exists(),
            f"validation artifact already exists; refusing first recovery: {relative}",
        )

    arguments = _load_json(run_dir / "arguments.json")
    _require(
        arguments.get("experiment_contract_sha256") == BASE_CONTRACT_SHA256,
        "E1 base contract hash drifted",
    )
    _require(
        arguments.get("implementation_freeze_sha256") == BASE_FREEZE_SHA256,
        "E1 base implementation freeze hash drifted",
    )
    requested = arguments.get("requested_raw_steps", {})
    exact = {
        "algo": "topo_v4_5_confident_actor_fusion",
        "scenario": "cross",
        "seed": 4,
        "max_steps": 20_000,
        "evaluation_split": "validation",
        "evaluation_seed_start": 59_000,
        "eval_episodes": 12,
        "calibration_seed_start": 68_000,
        "calibration_episodes": 12,
    }
    for key, expected in exact.items():
        _require(requested.get(key) == expected, f"E1 requested {key} drifted")

    training = _load_json(run_dir / "training_diagnostics.json")
    _require(training.get("raw_steps") == 20_000, "E1 training budget is incomplete")
    _require(int(training.get("learner_updates", 0)) > 0, "E1 has no learner updates")
    checkpoint_audit = _load_json(run_dir / "checkpoint_audit.json")
    rows = checkpoint_audit.get("checkpoints", [])
    _require(
        checkpoint_audit.get("expected_checkpoint_count") == 1
        and len(rows) == 1
        and rows[0].get("raw_step") == 20_000
        and rows[0].get("zip_crc_ok") is True,
        "E1 exact checkpoint audit failed",
    )
    return arguments


def _candidate(run_dir: Path, kind: str) -> dict[str, Any]:
    short = {"highest_training_success": "best", "exact_final": "final"}[kind]
    checkpoint = run_dir / (
        "best_training_success_model.zip"
        if kind == "highest_training_success"
        else "final_model.zip"
    )
    calibration = run_dir / "selector" / "cal" / short
    detailed_path = calibration / "detailed.json"
    evaluation_path = calibration / "evaluation.json"
    actions_path = calibration / "actions.json"
    trace_path = calibration / "decisions.jsonl"
    for path in (detailed_path, evaluation_path, actions_path, trace_path):
        _require(path.is_file(), f"missing immutable calibration artifact: {path}")
    detailed = _load_json(detailed_path)
    summary = _load_json(evaluation_path)
    actions = _load_json(actions_path)
    _require(detailed.get("summary") == summary, f"{kind} summary drifted")
    _require(detailed.get("traffic_partition") == "train", f"{kind} is not train-only")
    _require(detailed.get("formal_test_accessed") is False, f"{kind} accessed formal test")
    _require(detailed.get("calibration_seed_start") == 68_000, f"{kind} seed block drifted")
    _require(detailed.get("calibration_episodes") == 12, f"{kind} episode count drifted")
    _require(detailed.get("checkpoint_sha256") == sha256(checkpoint), f"{kind} checkpoint binding drifted")
    _require(
        actions.get("deterministic_lane_decoder")
        == "actor_confident_non_keep_else_target_critic",
        f"{kind} calibration decoder drifted",
    )
    _require(
        actions.get("trace", {}).get("sha256") == sha256(trace_path),
        f"{kind} calibration action trace drifted",
    )
    return {
        "checkpoint_kind": kind,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "traffic_partition": "train",
        "calibration_seed_start": 68_000,
        "summary": summary,
        "episode_records": detailed["episode_records"],
        "calibration_result_sha256": sha256(detailed_path),
        "calibration_action_diagnostics_sha256": sha256(actions_path),
        "deterministic_lane_decoder": actions["deterministic_lane_decoder"],
    }


def _namespace(arguments: dict[str, Any], device: str) -> Namespace:
    requested = dict(arguments["requested_raw_steps"])
    requested["device"] = device
    requested["output_dir"] = Path(requested["output_dir"])
    formal = requested.get("formal_unlock_receipt")
    requested["formal_unlock_receipt"] = Path(formal) if formal else None
    return Namespace(**requested)


def resume(run_dir: Path, *, device: str, patch_freeze: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    patch_freeze = patch_freeze.resolve()
    arguments = _verify_preregistered_inputs(run_dir)
    _require(patch_freeze.is_file(), f"missing v4.5.1 implementation freeze: {patch_freeze}")
    patch_freeze_payload = _load_json(patch_freeze)
    _require(
        patch_freeze_payload.get("status") == "frozen_before_e1_validation",
        "v4.5.1 patch is not frozen for validation",
    )
    prereg = PROJECT_ROOT / "results_topo_v4_5_dev" / "engineering" / "v4_5_1" / "preregistration_receipt.json"
    _require(prereg.is_file(), "missing v4.5.1 preregistration receipt")

    candidates = [
        _candidate(run_dir, "highest_training_success"),
        _candidate(run_dir, "exact_final"),
    ]
    selector = select_checkpoint(candidates)
    decoders = {str(row["deterministic_lane_decoder"]) for row in candidates}
    _require(len(decoders) == 1, "calibration candidates used different decoders")
    selector.update(
        {
            "calibration_deterministic_lane_decoder": next(iter(decoders)),
            "calibration_uses_method_own_frozen_decoder": True,
            "engineering_amendment_path": str(
                (PROJECT_ROOT / "experiments" / "topo_scene_v4" / "engineering_amendment_v4_5_1.yaml").resolve()
            ),
            "engineering_amendment_sha256": sha256(
                PROJECT_ROOT / "experiments" / "topo_scene_v4" / "engineering_amendment_v4_5_1.yaml"
            ),
            "engineering_patch_freeze_path": str(patch_freeze),
            "engineering_patch_freeze_sha256": sha256(patch_freeze),
            "preregistration_receipt_sha256": sha256(prereg),
            "reused_training_and_calibration": True,
            "validation_artifacts_absent_before_recovery": True,
        }
    )
    recovery_audit = {
        "schema_version": "topo-scene-v4.5.1.e1-recovery-audit/v1",
        "run": E1_NAME,
        "formal_test_accessed": False,
        "validation_environment_constructed": False,
        "verified_input_hashes": EXPECTED_INPUT_HASHES,
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
        "terminal_event_semantics": selector["terminal_event_semantics"],
        "engineering_patch_freeze_sha256": sha256(patch_freeze),
    }
    audit_path = run_dir / "selector" / "v4_5_1_recovery_audit.json"
    _write_json(audit_path, recovery_audit)
    selector["recovery_audit_sha256"] = sha256(audit_path)

    args = _namespace(arguments, device)
    (
        selected_model,
        selector_receipt,
        selector_receipt_sha,
        validation_env,
        validation_constructed_at,
    ) = _seal_selector_then_construct_evaluation_env(
        selector=selector,
        run_dir=run_dir,
        args=args,
        env_factory=_make_paper_env_v4_5,
    )
    try:
        model = ConfidentActorFusionSACV45.load(
            selected_model, env=validation_env, device=device
        )
        space_contract = validate_model_environment_spaces(model, validation_env)
        report, action_diagnostics = evaluate_with_action_diagnostics_v4_5(
            model,
            validation_env,
            episodes=args.eval_episodes,
            seed=args.evaluation_seed_start,
            trace_path=run_dir / "action_diagnostics_decisions.jsonl",
        )
        provenance = build_evaluation_provenance_v2(
            model=model,
            env=validation_env,
            report=report,
            algorithm=args.algo,
            scenario=args.scenario,
            traffic_protocol=args.traffic_protocol,
            evaluation_split=args.evaluation_split,
            episode_limit_profile=args.episode_limit_profile,
            evaluation_seed_start=args.evaluation_seed_start,
            environment_factory=callable_name(_make_paper_env_v4_5),
            space_contract=space_contract,
        )
    finally:
        validation_env.close()

    metadata = _load_json(run_dir / "method_metadata.json")
    training = _load_json(run_dir / "training_diagnostics.json")
    collected_raw_steps = int(training["raw_steps"])
    _write_json(run_dir / "final_evaluation.json", report.summary.to_dict())
    _write_json(run_dir / "action_diagnostics.json", action_diagnostics)
    detailed = {
        "schema_version": "topo-scene-v4.5.detailed-evaluation/v1",
        "model": str(selected_model.resolve()),
        "model_sha256": sha256(selected_model),
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selector_receipt": str(selector_receipt.resolve()),
        "selector_receipt_sha256": selector_receipt_sha,
        "selector_receipt_precedes_validation_environment": True,
        "selector_receipt_sealed_at_utc": selector["sealed_at_utc"],
        "validation_environment_constructed_at_utc": validation_constructed_at,
        "algorithm": args.algo,
        "implementation_id": V45_IMPLEMENTATION_IDS[args.algo],
        "method_metadata": metadata,
        "scenario": args.scenario,
        "evaluation_seed_start": args.evaluation_seed_start,
        "trained_raw_steps": collected_raw_steps,
        "collected_training_raw_steps": collected_raw_steps,
        "selected_model_learner_timesteps": int(model.num_timesteps),
        "experiment_contract_sha256": args.experiment_contract_sha256,
        "stage0_results_sha256": args.stage0_results_sha256,
        "attribution_sha256": args.attribution_sha256,
        "implementation_freeze_sha256": args.implementation_freeze_sha256,
        "evaluation_split": args.evaluation_split,
        "evaluation_provenance": provenance,
        "completion_time_population_std_ddof": 0,
        "engineering_patch": "v4.5.1_overlapping_terminal_event_flags",
        "engineering_patch_freeze_sha256": sha256(patch_freeze),
        **report.to_dict(),
    }
    _write_json(run_dir / "paper_evaluation_detailed.json", detailed)
    return {
        "status": "completed",
        "run": E1_NAME,
        "selected_checkpoint_kind": selector["selected_checkpoint_kind"],
        "selected_model_sha256": sha256(selected_model),
        "selector_receipt_sha256": selector_receipt_sha,
        "engineering_patch_freeze_sha256": sha256(patch_freeze),
        **report.summary.to_dict(),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument(
        "--run-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_topo_v4_5_dev"
            / "development"
            / "runs"
            / E1_NAME
        ),
    )
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--patch-freeze",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_topo_v4_5_dev"
            / "engineering"
            / "v4_5_1"
            / "implementation_freeze.json"
        ),
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = resume(args.run_dir, device=args.device, patch_freeze=args.patch_freeze)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
