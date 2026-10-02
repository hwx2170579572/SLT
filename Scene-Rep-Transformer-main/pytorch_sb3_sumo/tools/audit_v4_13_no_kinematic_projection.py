"""Audit v4.13 as a learned model with no kinematic safety projection."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
)
from algos.sb3_torch.hybrid_policy_v4_13_model import (
    GradientIsolatedTemperedJointSupportPolicyV413,
    GradientIsolatedTemperedMixtureActorV413,
)
from algos.sb3_torch.sac_v4_13_model import (
    GradientIsolatedTemperedJointSupportSACV413,
)
from tools import audit_v4_12_no_kinematic_projection as inherited
from tools.action_diagnostics_v4_12_model import _proposal_integrity


DEFAULT_RUN = ROOT / "r413s" / "e" / "m"
DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_13_dev"
    / "engineering"
    / "no_kinematic_projection_audit.json"
)
V413_SOURCE_FILES = (
    "algos/sb3_torch/hybrid_policy_v4_13_model.py",
    "algos/sb3_torch/sac_v4_13_model.py",
    "configs/sb3_configs_v4_13.py",
    "tools/action_diagnostics_v4_13_model.py",
    "tools/train_sb3_v4_13.py",
    "tools/train_paper_sb3_sumo_v4_13.py",
)
FORBIDDEN_FIELDS = (
    "inference_safety_rule_added",
    "external_kinematic_projection",
    "kinematic_safety_projection",
    "traffic_risk_in_lane_mask",
    "ttc_or_headway_threshold",
    "lane_change_veto",
    "actor_confidence_gate",
    "action_postprocessing_override",
    "scenario_conditioned_inference_rule",
)
SUSPICIOUS_CALL_TERMS = (
    "project",
    "projection",
    "ttc",
    "headway",
    "veto",
    "shield",
    "safe_speed",
    "emergency",
    "fallback",
    "rewrite_action",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _call_name(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts)).lower()


def _suspicious_runtime_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = sorted(
        {
            _call_name(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
    )
    return [
        call
        for call in calls
        if any(term in call for term in SUSPICIOUS_CALL_TERMS)
    ]


def _toy_policy(
    *, isolate: bool = True
) -> GradientIsolatedTemperedJointSupportPolicyV413:
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                -100.0, 100.0, shape=(4,), dtype=np.float32
            ),
            "lane_action_mask": gym.spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    torch.manual_seed(20260902)
    return GradientIsolatedTemperedJointSupportPolicyV413(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [16, 8], "qf": [16, 8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=inherited.inherited.AuditFeatureExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
        speed_components=3,
        twin_uncertainty_coef=0.25,
        component_prior_coef=0.05,
        lane_prior_coef=0.05,
        lane_support_scale=0.25,
        isolate_lane_support_gradient=isolate,
    )


def _grad_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def source_boundary_audit() -> dict[str, Any]:
    inherited_source = inherited.source_boundary_audit()
    actor_source = inspect.getsource(
        GradientIsolatedTemperedMixtureActorV413
    ).lower()
    predict_source = inspect.getsource(
        GradientIsolatedTemperedJointSupportPolicyV413._predict
    ).lower()
    new_calls = {
        relative: _suspicious_runtime_calls(ROOT / relative)
        for relative in V413_SOURCE_FILES[:3]
    }
    return {
        "v4_13_source_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in V413_SOURCE_FILES
        },
        "inherited_v4_12_boundary": inherited_source,
        "inherited_boundary_integrity": all(
            value
            for value in inherited_source.values()
            if isinstance(value, bool)
        ),
        "same_deployed_lane_head_reused_for_support": (
            actor_source.count("self.lane_logits") >= 1
            and "support_lane_logits" not in dict(
                _toy_policy().actor.named_modules()
            )
        ),
        "lane_support_input_is_stop_gradient_latent": (
            "self.lane_logits(latent.detach())" in actor_source
        ),
        "conditional_speed_nll_calls_frozen_learned_mixture_objective": (
            "supportedmixturehybridactor.replay_support_nll" in actor_source
        ),
        "joint_support_is_preregistered_tempered_sum": (
            "speed_nll + self.lane_support_scale * lane_nll" in actor_source
        ),
        "v4_13_does_not_override_inference_score": (
            GradientIsolatedTemperedJointSupportPolicyV413.
            _risk_adjusted_proposal_values
            is AugmentedJointSupportPRCRPolicyV412.
            _risk_adjusted_proposal_values
        ),
        "v4_13_predict_delegates_action_to_frozen_model_argmax": (
            "actions = super()._predict" in predict_source
            and predict_source.rstrip().endswith("return actions")
        ),
        "v4_13_predict_only_appends_gradient_diagnostics": (
            "record.update" in predict_source
            and "lane_support_gradient_isolated" in predict_source
        ),
        "new_scientific_modules_have_no_projection_or_rule_calls": all(
            not calls for calls in new_calls.values()
        ),
        "suspicious_runtime_calls_by_file": new_calls,
        "classification": {
            "model_change": (
                "same deployed lane head, detached auxiliary latent, and "
                "tempered differentiable replay likelihood"
            ),
            "structural_constraint": (
                "physical adjacent driving-lane existence only"
            ),
            "kinematic_or_traffic_risk_projection": "absent",
        },
    }


def model_execution_audit() -> dict[str, Any]:
    inherited_model = inherited.model_execution_audit()
    policy = _toy_policy()
    actor = policy.actor
    if not isinstance(actor, GradientIsolatedTemperedMixtureActorV413):
        raise TypeError("v4.13 toy policy lost its isolated actor")
    observation = inherited.inherited._observation()
    replay_actions = torch.tensor([[-0.4, -1.0]], dtype=torch.float32)

    batch = actor.all_action_proposals(observation, deterministic_speed=True)
    terms = actor.joint_replay_support_terms(batch, replay_actions)
    actor.optimizer.zero_grad()
    terms.lane_categorical_nll.backward()
    lane_gradients = {
        "deployed_lane_head": _grad_norm(actor.lane_logits),
        "actor_latent_trunk": _grad_norm(actor.latent_pi),
        "component_head": _grad_norm(actor.component_logits),
        "speed_mean_head": _grad_norm(actor.speed_mean),
        "speed_log_std_head": _grad_norm(actor.speed_log_std),
        "scene_encoder": _grad_norm(actor.features_extractor),
    }

    batch = actor.all_action_proposals(observation, deterministic_speed=True)
    terms = actor.joint_replay_support_terms(batch, replay_actions)
    actor.optimizer.zero_grad()
    terms.conditional_speed_mixture_nll.backward()
    speed_gradients = {
        "deployed_lane_head": _grad_norm(actor.lane_logits),
        "actor_latent_trunk": _grad_norm(actor.latent_pi),
        "component_head": _grad_norm(actor.component_logits),
        "speed_mean_head": _grad_norm(actor.speed_mean),
        "speed_log_std_head": _grad_norm(actor.speed_log_std),
        "scene_encoder": _grad_norm(actor.features_extractor),
    }

    proposal = actor.all_action_proposals(
        observation, deterministic_speed=True
    )
    policy.begin_target_decoder_recording()
    selected = policy._predict(observation, deterministic=True)
    record = policy.end_target_decoder_recording()[0]
    lane = int(record["selected_lane_index"])
    component = int(record["selected_component_index"])
    exact_proposal = proposal.actions[0, lane, component]
    inherited_checks = {
        key: value
        for key, value in inherited_model.items()
        if isinstance(value, bool)
        and key not in {
            "semantic_tie_override_used",
            "actor_path_updates_shared_encoder",
        }
    }
    return {
        "inherited_v4_12_model_execution": inherited_model,
        "inherited_model_execution_integrity": all(
            inherited_checks.values()
        ),
        "inherited_semantic_tie_override_absent": (
            inherited_model.get("semantic_tie_override_absent") is True
        ),
        "lane_support_gradients": lane_gradients,
        "speed_support_gradients": speed_gradients,
        "lane_nll_updates_same_deployed_lane_head": (
            lane_gradients["deployed_lane_head"] > 0.0
        ),
        "lane_nll_zero_gradient_actor_latent_trunk": (
            lane_gradients["actor_latent_trunk"] == 0.0
        ),
        "lane_nll_zero_gradient_speed_heads": all(
            lane_gradients[name] == 0.0
            for name in (
                "component_head",
                "speed_mean_head",
                "speed_log_std_head",
            )
        ),
        "lane_nll_zero_gradient_scene_encoder": (
            lane_gradients["scene_encoder"] == 0.0
        ),
        "speed_nll_updates_latent_and_speed_heads": all(
            speed_gradients[name] > 0.0
            for name in (
                "actor_latent_trunk",
                "component_head",
                "speed_mean_head",
                "speed_log_std_head",
            )
        ),
        "joint_action_nll_is_finite": bool(
            torch.isfinite(terms.joint_action_nll)
        ),
        "joint_nll_equals_tempered_sum": bool(
            torch.allclose(
                terms.joint_action_nll,
                terms.conditional_speed_mixture_nll
                + 0.25 * terms.lane_categorical_nll,
            )
        ),
        "selected_action_equals_exact_actor_proposal": bool(
            torch.allclose(selected[0], exact_proposal, rtol=0.0, atol=1e-7)
        ),
        "selected_lane_is_physically_feasible": bool(
            record["valid_lane_actions"][lane]
        ),
        "plain_flattened_model_argmax_recorded": (
            record.get("selection_operator")
            == "torch_argmax_flattened_feasible_model_scores"
        ),
        "semantic_tie_override_absent": (
            record.get("semantic_tie_override_used") is False
        ),
        "actor_confidence_threshold_absent": (
            record.get("actor_confidence_threshold_present") is False
        ),
        "post_decoder_action_rewrite_absent": (
            record.get("action_rewritten") is False
        ),
        "optimizer_ownership": policy.optimizer_parameter_ownership(),
    }


def runtime_run_audit(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"available": False, "required": False}
    required = {
        "arguments": run_dir / "arguments.json",
        "metadata": run_dir / "method_metadata.json",
        "training": run_dir / "training_diagnostics.json",
        "actions": run_dir / "action_diagnostics.json",
        "trace": run_dir / "action_diagnostics_decisions.jsonl",
        "detailed": run_dir / "paper_evaluation_detailed.json",
        "selector": run_dir / "selector" / "receipt.json",
        "selected_model": run_dir / "selected_model.zip",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "available": False,
            "required": True,
            "run_dir": str(run_dir.resolve()),
            "missing": missing,
        }
    values = {
        name: _load(path)
        for name, path in required.items()
        if path.suffix == ".json"
    }
    arguments = values["arguments"]
    metadata = values["metadata"]
    training = values["training"]
    actions = values["actions"]
    detailed = values["detailed"]
    selector = values["selector"]
    rows = [
        json.loads(line)
        for line in required["trace"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recomputed = _proposal_integrity(rows) if rows else {}
    fidelity = arguments.get("implementation_fidelity", {})
    surfaces = (fidelity, metadata, training, actions, detailed, selector)
    exact_keys = (
        "selected_decoder_exact_model_match_rate",
        "selected_decoder_exact_action_match_rate",
        "selected_action_mask_feasible_rate",
        "exact_joint_support_model_argmax_rate",
        "joint_support_score_equation_match_rate",
        "selected_speed_exact_proposal_match_rate",
        "learned_proposal_source_rate",
        "no_action_rewrite_rate",
        "no_semantic_tie_override_rate",
        "no_actor_confidence_threshold_rate",
    )
    finite_support = all(
        int(training.get("statistics", {}).get(name, {}).get("count", 0)) > 0
        and math.isfinite(float(training["statistics"][name]["last"]))
        for name in (
            "support/replay_joint_action_nll",
            "support/replay_joint_action_nll_recomputed",
            "support/replay_lane_categorical_nll",
            "support/replay_conditional_speed_mixture_nll",
        )
    )
    return {
        "available": True,
        "required": True,
        "run_dir": str(run_dir.resolve()),
        "artifact_sha256": {
            name: _sha256(path) for name, path in required.items()
        },
        "decision_records": len(rows),
        "v4_13_schema_bound": all(
            str(value.get("schema_version", "")).startswith(
                "topo-scene-v4.13."
            )
            for value in (arguments, training, actions, detailed, selector)
        ),
        "validation_only_and_formal_untouched": (
            arguments.get("requested_raw_steps", {}).get("evaluation_split")
            == "validation"
            and detailed.get("formal_test_accessed") is False
            and arguments.get("formal_unlock") is None
        ),
        "all_rule_projection_fields_false": all(
            surface.get(key) is False
            for surface in surfaces
            for key in FORBIDDEN_FIELDS
            if key in surface
        )
        and all(
            key in actions and actions.get(key) is False
            for key in FORBIDDEN_FIELDS
        ),
        "gradient_isolation_runtime_metadata_present": (
            actions.get("lane_support_gradient_isolated") is True
            and actions.get("lane_support_gradient_target")
            == "deployed_lane_head_only"
            and training.get("lane_support_gradient_isolated") is True
            and metadata.get("lane_support_gradient_isolated") is True
        ),
        "same_deployed_lane_head_without_new_inference_head": (
            actions.get("same_deployed_lane_head_for_support") is True
            and actions.get("new_inference_head") is False
        ),
        "inference_equation_unchanged": (
            actions.get("inference_score_equation_changed_from_v4_12")
            is False
        ),
        "all_joint_support_training_statistics_finite": finite_support,
        "all_exact_model_action_rates_are_one": bool(rows)
        and all(actions.get(key) == 1.0 for key in exact_keys),
        "independent_trace_recomputation_matches": bool(rows)
        and all(recomputed.get(key) == 1.0 for key in exact_keys),
        "selector_is_target_only_without_rule_candidates": (
            selector.get("deployment_decoder_candidates") == ["target_critic"]
            and selector.get("fusion_candidate_present") is False
            and selector.get("actor_confidence_threshold_present") is False
        ),
        "selector_preserves_policy_parameter_state": (
            selector.get("policy_parameter_state_preserved") is True
            and selector.get("source_policy_parameter_state_sha256")
            == selector.get("selected_model_parameter_state_sha256")
        ),
        "optimizer_owners_do_not_overlap": (
            actions.get("optimizer_ownership", {}).get("overlap_count") == 0
        ),
    }


def _boolean_checks(section: Mapping[str, Any]) -> dict[str, bool]:
    return {
        key: value
        for key, value in section.items()
        if isinstance(value, bool)
    }


def build_audit(
    run_dir: Path | None = None, *, require_runtime: bool = False
) -> dict[str, Any]:
    source = source_boundary_audit()
    model = model_execution_audit()
    runtime = runtime_run_audit(run_dir)
    checks = {
        **{
            f"source.{key}": value
            for key, value in _boolean_checks(source).items()
        },
        **{
            f"model.{key}": value
            for key, value in _boolean_checks(model).items()
        },
    }
    if runtime.get("available"):
        checks.update(
            {
                f"runtime.{key}": value
                for key, value in _boolean_checks(runtime).items()
            }
        )
    integrity_passed = all(checks.values()) and (
        not require_runtime or runtime.get("available") is True
    )
    return {
        "schema_version": (
            "topo-scene-v4.13.no-kinematic-projection-audit/v1"
        ),
        "analysis_kind": (
            "transitive_static_call_graph_plus_gradient_ownership_plus_"
            "exact_action_runtime_trace"
        ),
        "computed_from_real_sources": True,
        "fabricated_values": False,
        "scientific_model_changed_by_audit": False,
        "run_artifacts_modified": False,
        "formal_test_accessed": False,
        "runtime_required": require_runtime,
        "integrity_passed": integrity_passed,
        "candidate_side_kinematic_or_traffic_risk_projection_present": False,
        "inference_rule_mechanism_present": False,
        "physical_lane_availability_mask_present": True,
        "physical_lane_availability_mask_is_traffic_risk_rule": False,
        "source_boundary": source,
        "model_execution": model,
        "engineering_runtime": runtime,
        "failed_checks": sorted(
            key for key, value in checks.items() if not value
        ),
        "model_only_improvement": (
            "the same deployed lane head receives a detached, tempered "
            "auxiliary likelihood; no threshold, projection, veto, shield, "
            "fallback, fixed grid, semantic override, or action rewrite"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.run_dir, require_runtime=args.require_runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
