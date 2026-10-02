"""Evaluate a frozen v4.12 checkpoint after a continuous lane-prior ablation.

The intervention changes only ``policy.lane_prior_coef``.  It does not retrain
the model, inspect future traffic, add a threshold, veto an action, project
kinematics, or rewrite an environment action.  The default run uses the exact
v4.12 D3 validation seeds so its closed-loop outcome can be compared directly
with the immutable source rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.sac_v4_12_model import AugmentedJointSupportPRCRSACV412
from tools.action_diagnostics_v4_12_model import (
    decoder_integrity_passed_v4_12,
    evaluate_with_action_diagnostics_v4_12_model,
    load_model_for_deployment_v4_12,
)
from tools.checkpoint_decoder_selector_v4_6 import TARGET_DECODER
from tools.paper_evaluation_contract import validate_model_environment_spaces
from tools.paper_evaluation_contract_v2 import (
    build_evaluation_provenance_v2,
    callable_name,
)
from tools.train_paper_sb3_sumo_v4_5 import _make_paper_env_v4_5


DEFAULT_SOURCE_RUN = ROOT / "r412" / "d" / "D3__ajs__ram__s12__p3f978ef9"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results_topo_v4_12_dev"
    / "development"
    / "attribution"
    / "d3_roundabout_regression"
    / "closed_loop_lane_prior_0"
)
FORBIDDEN_DIAGNOSTIC_FIELDS = (
    "inference_safety_rule_added",
    "external_kinematic_projection",
    "kinematic_safety_projection",
    "traffic_risk_in_lane_mask",
    "actor_confidence_gate",
    "action_postprocessing_override",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parameter_state_sha256(model: Any) -> str:
    """Hash tensor state without serializing or mutating the model."""

    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def apply_lane_prior_intervention(
    model: Any,
    coefficient: float,
    *,
    allow_increase: bool = False,
) -> dict[str, Any]:
    _require(np.isfinite(coefficient), "lane-prior coefficient must be finite")
    _require(coefficient >= 0.0, "lane-prior coefficient must be non-negative")
    policy = model.policy
    _require(hasattr(policy, "lane_prior_coef"), "policy has no lane-prior model")
    original = float(policy.lane_prior_coef)
    if allow_increase:
        _require(
            coefficient <= 0.10,
            "sensitivity intervention is preregistered only through 0.10",
        )
    else:
        _require(
            coefficient <= original,
            "attribution ablation may not increase the frozen lane-prior coefficient",
        )
    before = parameter_state_sha256(model)
    policy.lane_prior_coef = float(coefficient)
    after = parameter_state_sha256(model)
    _require(before == after, "lane-prior intervention changed learned tensor state")
    return {
        "kind": (
            "continuous_learned_model_coefficient_sensitivity"
            if allow_increase
            else "continuous_learned_model_coefficient_ablation"
        ),
        "attribute": "policy.lane_prior_coef",
        "original_value": original,
        "intervened_value": float(policy.lane_prior_coef),
        "parameter_state_sha256_before": before,
        "parameter_state_sha256_after": after,
        "learned_tensor_state_unchanged": True,
        "training_performed": False,
        "increase_explicitly_enabled": bool(allow_increase),
        "scenario_conditioned": False,
        "threshold_or_veto": False,
        "action_rewrite": False,
        "kinematic_projection": False,
    }


def _summary_delta(
    candidate: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, float]:
    keys = (
        "mean_return",
        "std_return",
        "mean_decision_steps",
        "mean_raw_steps",
        "success_rate",
        "collision_rate",
        "off_route_rate",
        "timeout_rate",
    )
    return {
        key: float(candidate[key]) - float(source[key])
        for key in keys
        if candidate.get(key) is not None and source.get(key) is not None
    }


def _source_contract(source_run: Path) -> dict[str, Any]:
    paths = {
        "arguments": source_run / "arguments.json",
        "paper_evaluation": source_run / "paper_evaluation_detailed.json",
        "action_diagnostics": source_run / "action_diagnostics.json",
        "trace": source_run / "action_diagnostics_decisions.jsonl",
        "selected_model": source_run / "selected_model.zip",
        "selector_receipt": source_run / "selector" / "receipt.json",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing source {name}: {path}")
    arguments = _load_json(paths["arguments"])
    paper = _load_json(paths["paper_evaluation"])
    actions = _load_json(paths["action_diagnostics"])
    _require(paper["evaluation_split"] == "validation", "source is not validation")
    _require(paper["formal_test_accessed"] is False, "formal test was accessed")
    _require(
        _sha256(paths["selected_model"]) == paper["model_sha256"],
        "selected checkpoint hash drifted",
    )
    _require(
        _sha256(paths["trace"]) == actions["trace"]["sha256"],
        "source action trace hash drifted",
    )
    requested = arguments["requested_raw_steps"]
    _require(requested["algo"] == "topo_v4_12_augmented_joint_support_prcr_full", "source method drift")
    _require(
        requested["scenario"] in {"cross", "roundabout_medium", "carla"},
        "source scenario is outside the v4.12 development matrix",
    )
    _require(
        int(requested["evaluation_seed_start"]) == int(paper["evaluation_seed_start"]),
        "source evaluation seed drifted",
    )
    return {
        "paths": paths,
        "arguments": arguments,
        "paper": paper,
        "actions": actions,
    }


def evaluate(
    *,
    source_run: Path,
    output_dir: Path,
    lane_prior_coef: float,
    device: str,
    episodes: int | None,
    evaluation_seed_start: int | None,
    allow_increase: bool = False,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"immutable output already exists: {output_dir}")
    source = _source_contract(source_run)
    requested = dict(source["arguments"]["requested_raw_steps"])
    scenario = str(requested["scenario"])
    source_episodes = int(requested["eval_episodes"])
    source_seed = int(requested["evaluation_seed_start"])
    episodes = source_episodes if episodes is None else int(episodes)
    evaluation_seed_start = (
        source_seed
        if evaluation_seed_start is None
        else int(evaluation_seed_start)
    )
    _require(episodes == source_episodes, "ablation must use source episode count")
    _require(
        evaluation_seed_start == source_seed,
        "ablation must use exact source validation seeds",
    )
    _require(requested["evaluation_split"] == "validation", "formal split forbidden")
    requested["gui"] = False
    args = SimpleNamespace(**requested)
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "action_diagnostics_decisions.jsonl"
    env = _make_paper_env_v4_5(args, evaluation=True)
    try:
        model = load_model_for_deployment_v4_12(
            AugmentedJointSupportPRCRSACV412,
            source["paths"]["selected_model"],
            decoder=TARGET_DECODER,
            env=env,
            device=device,
        )
        space_contract = validate_model_environment_spaces(model, env)
        intervention = apply_lane_prior_intervention(
            model, lane_prior_coef, allow_increase=allow_increase
        )
        report, diagnostics = evaluate_with_action_diagnostics_v4_12_model(
            model,
            env,
            deployment_decoder=TARGET_DECODER,
            episodes=episodes,
            seed=evaluation_seed_start,
            trace_path=trace_path,
        )
        provenance = build_evaluation_provenance_v2(
            model=model,
            env=env,
            report=report,
            algorithm="topo_v4_12_frozen_lane_prior_ablation",
            scenario=scenario,
            traffic_protocol=requested["traffic_protocol"],
            evaluation_split="validation",
            episode_limit_profile=requested["episode_limit_profile"],
            evaluation_seed_start=evaluation_seed_start,
            environment_factory=callable_name(_make_paper_env_v4_5),
            space_contract=space_contract,
        )
    finally:
        env.close()
    _require(decoder_integrity_passed_v4_12(diagnostics), "decoder integrity failed")
    for field in FORBIDDEN_DIAGNOSTIC_FIELDS:
        _require(diagnostics.get(field) is False, f"forbidden mechanism: {field}")
    _require(diagnostics["lane_prior_coef"] == lane_prior_coef, "coefficient drift")
    _require(diagnostics["no_action_rewrite_rate"] == 1.0, "action was rewritten")
    summary = report.summary.to_dict()
    payload = {
        "schema_version": "topo-scene-v4.12.frozen-lane-prior-ablation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": (
            "closed_loop_frozen_checkpoint_continuous_model_sensitivity"
            if allow_increase
            else "closed_loop_frozen_checkpoint_continuous_model_ablation"
        ),
        "computed_from_real_rollout": True,
        "fabricated_values": False,
        "formal_test_accessed": False,
        "evaluation_split": "validation",
        "scenario": scenario,
        "same_checkpoint_as_source": True,
        "same_environment_seed_contract_as_source": True,
        "source_run": _relative_or_absolute(source_run),
        "source_summary": source["paper"]["summary"],
        "ablation_summary": summary,
        "ablation_minus_source": _summary_delta(summary, source["paper"]["summary"]),
        "intervention": intervention,
        "evaluation_seed_start": evaluation_seed_start,
        "episodes": episodes,
        "evaluation_provenance": provenance,
        "report": report.to_dict(),
        "decoder_integrity_passed": True,
        "forbidden_mechanisms_used": {
            field: diagnostics[field] for field in FORBIDDEN_DIAGNOSTIC_FIELDS
        },
        "source_artifacts": {
            name: {
                "path": _relative_or_absolute(path),
                "sha256": _sha256(path),
            }
            for name, path in source["paths"].items()
        },
        "output_trace": {
            "path": _relative_or_absolute(trace_path),
            "sha256": _sha256(trace_path),
            "records": diagnostics["decision_records"],
        },
    }
    _write_json(output_dir / "action_diagnostics.json", diagnostics)
    _write_json(output_dir / "closed_loop_evaluation.json", payload)
    receipt = {
        "schema_version": "topo-scene-v4.12.frozen-lane-prior-ablation-receipt/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_path": _relative_or_absolute(
            output_dir / "closed_loop_evaluation.json"
        ),
        "evaluation_sha256": _sha256(output_dir / "closed_loop_evaluation.json"),
        "action_diagnostics_sha256": _sha256(
            output_dir / "action_diagnostics.json"
        ),
        "trace_sha256": _sha256(trace_path),
        "complete": True,
        "formal_test_accessed": False,
    }
    _write_json(output_dir / "receipt.json", receipt)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    value.add_argument("--lane-prior-coef", type=float, default=0.0)
    value.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    value.add_argument("--episodes", type=int, default=None)
    value.add_argument("--evaluation-seed-start", type=int, default=None)
    value.add_argument("--allow-increase", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = evaluate(
        source_run=args.source_run,
        output_dir=args.output_dir,
        lane_prior_coef=args.lane_prior_coef,
        device=args.device,
        episodes=args.episodes,
        evaluation_seed_start=args.evaluation_seed_start,
        allow_increase=args.allow_increase,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "source_summary": payload["source_summary"],
                "ablation_summary": payload["ablation_summary"],
                "ablation_minus_source": payload["ablation_minus_source"],
                "decoder_integrity_passed": payload["decoder_integrity_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "apply_lane_prior_intervention",
    "evaluate",
    "parameter_state_sha256",
]
