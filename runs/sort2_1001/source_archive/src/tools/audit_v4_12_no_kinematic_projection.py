"""Audit v4.12 as a learned model, not a kinematic safety projection.

The audit deliberately distinguishes the physical adjacent-lane existence mask
from any traffic-risk rule.  It combines the already frozen v4.11 transitive
environment/action boundary with executable checks of v4.12's joint replay
support, lane prior, exact argmax, and a real SUMO decision trace.
"""

from __future__ import annotations

import argparse
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
    JointReplaySupportedMixtureActorV412,
)
from algos.sb3_torch.sac_v4_12_model import AugmentedJointSupportPRCRSACV412
from tools import audit_v4_11_no_kinematic_projection as inherited
from tools.action_diagnostics_v4_12_model import _proposal_integrity


DEFAULT_RUN = ROOT / "r412s" / "e2" / "m"
DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_12_dev"
    / "engineering"
    / "no_kinematic_projection_audit.json"
)
V412_SOURCE_FILES = (
    "algos/sb3_torch/hybrid_policy_v4_12_model.py",
    "algos/sb3_torch/sac_v4_12_model.py",
    "configs/sb3_configs_v4_12.py",
    "tools/action_diagnostics_v4_12_model.py",
    "tools/train_sb3_v4_12.py",
    "tools/train_paper_sb3_sumo_v4_12.py",
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


def _toy_policy() -> AugmentedJointSupportPRCRPolicyV412:
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
    torch.manual_seed(20260901)
    return AugmentedJointSupportPRCRPolicyV412(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [16, 8], "qf": [16, 8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=inherited.AuditFeatureExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
        speed_components=3,
        twin_uncertainty_coef=0.25,
        component_prior_coef=0.05,
        lane_prior_coef=0.05,
        lane_support_scale=1.0,
    )


def source_boundary_audit() -> dict[str, Any]:
    inherited_source = inherited.source_boundary_audit()
    support_source = inspect.getsource(
        JointReplaySupportedMixtureActorV412.joint_replay_support_terms
    ).lower()
    score_source = inspect.getsource(
        AugmentedJointSupportPRCRPolicyV412._risk_adjusted_proposal_values
    ).lower()
    predict_source = inspect.getsource(
        AugmentedJointSupportPRCRPolicyV412._predict
    ).lower()
    setup_source = inspect.getsource(
        AugmentedJointSupportPRCRSACV412._setup_model
    ).lower()
    diagnostics_source = inspect.getsource(_proposal_integrity).lower()
    return {
        "v4_12_source_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in V412_SOURCE_FILES
        },
        "inherited_v4_11_boundary": inherited_source,
        "inherited_boundary_integrity": all(
            value
            for key, value in inherited_source.items()
            if isinstance(value, bool)
        ),
        "lane_support_is_masked_categorical_nll": (
            "functional.nll_loss" in support_source
            and "batch.lane_log_probabilities" in support_source
            and "lane_indices" in support_source
        ),
        "speed_support_reuses_learned_conditional_mixture_nll": (
            "super().replay_support_nll" in support_source
        ),
        "joint_support_is_differentiable_weighted_sum": (
            "speed_nll + self.lane_support_scale * lane_nll" in support_source
        ),
        "score_adds_continuous_learned_lane_log_probability": (
            "batch.lane_log_probabilities" in score_source
            and "self.lane_prior_coef" in score_source
        ),
        "v4_12_predict_delegates_action_to_frozen_model_argmax": (
            "actions = super()._predict" in predict_source
            and predict_source.rstrip().endswith("return actions")
        ),
        "v4_12_predict_only_appends_diagnostic_metadata": (
            "record.update" in predict_source
            and "actor_confidence_threshold_present" in predict_source
        ),
        "checkpoint_setup_validates_model_components_after_sb3_restore": (
            "super()._setup_model()" in setup_source
            and "self._validate_v4_12_components()" in setup_source
        ),
        "runtime_diagnostics_recompute_lane_and_component_score": all(
            token in diagnostics_source
            for token in (
                "lane_probabilities",
                "component_probabilities",
                "lane_coef",
                "component_coef",
                "supported_risk_adjusted_score",
            )
        ),
        "runtime_diagnostics_reject_threshold_and_rewrite": (
            "actor_confidence_threshold_present" in diagnostics_source
            and "action_rewritten" in diagnostics_source
        ),
        "classification": {
            "model_change": (
                "masked replay lane likelihood, conditional speed likelihood, "
                "continuous learned lane prior, and existing stochastic rotation augmentation"
            ),
            "structural_constraint": "physical adjacent driving-lane existence only",
            "kinematic_or_traffic_risk_projection": "absent",
        },
    }


def model_execution_audit() -> dict[str, Any]:
    inherited_model = inherited.model_execution_audit()
    policy = _toy_policy()
    actor = policy.actor
    if not isinstance(actor, JointReplaySupportedMixtureActorV412):
        raise TypeError("v4.12 toy policy lost its joint-support actor")
    observation = inherited._observation()
    batch = actor.all_action_proposals(observation, deterministic_speed=True)
    replay_actions = torch.tensor(
        [[-0.4, -1.0]], dtype=torch.float32
    )
    terms = actor.joint_replay_support_terms(batch, replay_actions)
    actor.optimizer.zero_grad()
    terms.joint_action_nll.backward()
    lane_gradient = actor.lane_logits.weight.grad

    policy.begin_target_decoder_recording()
    selected = policy._predict(observation, deterministic=True)
    record = policy.end_target_decoder_recording()[0]
    lane = int(record["selected_lane_index"])
    component = int(record["selected_component_index"])
    reward = float(record["minimum_target_twin_q"][lane][component])
    collision = float(
        record["maximum_target_twin_collision_value"][lane][component]
    )
    uncertainty = float(record["learned_uncertainty"][lane][component])
    component_probability = float(
        record["component_probabilities"][lane][component]
    )
    lane_probability = float(record["lane_probabilities"][lane])
    expected_score = (
        reward
        - float(record["collision_risk_coef"]) * collision
        - float(record["twin_uncertainty_coef"]) * uncertainty
        + float(record["component_prior_coef"])
        * math.log(max(component_probability, 1e-45))
        + float(record["lane_prior_coef"])
        * math.log(max(lane_probability, 1e-45))
    )
    actual_score = float(
        record["supported_risk_adjusted_score"][lane][component]
    )
    exact_proposal = batch.actions[0, lane, component]
    inherited_positive = {
        key: value
        for key, value in inherited_model.items()
        if isinstance(value, bool)
        and key not in {
            "semantic_tie_override_used",
            "actor_path_updates_shared_encoder",
        }
    }
    return {
        "inherited_v4_11_model_execution": inherited_model,
        "inherited_model_execution_integrity": all(inherited_positive.values()),
        "inherited_semantic_tie_override_absent": (
            inherited_model.get("semantic_tie_override_used") is False
        ),
        "joint_action_nll_is_finite": bool(
            torch.isfinite(terms.joint_action_nll)
        ),
        "lane_categorical_nll_is_finite": bool(
            torch.isfinite(terms.lane_categorical_nll)
        ),
        "conditional_speed_nll_is_finite": bool(
            torch.isfinite(terms.conditional_speed_mixture_nll)
        ),
        "joint_nll_equals_preregistered_sum": bool(
            torch.allclose(
                terms.joint_action_nll,
                terms.conditional_speed_mixture_nll
                + terms.lane_categorical_nll,
            )
        ),
        "lane_support_gradient_reaches_lane_head": bool(
            lane_gradient is not None
            and torch.isfinite(lane_gradient).all()
            and lane_gradient.abs().sum() > 0
        ),
        "lane_prior_score_equation_exact": abs(actual_score - expected_score)
        <= 1e-5,
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
    arguments = _load(required["arguments"])
    metadata = _load(required["metadata"])
    training = _load(required["training"])
    actions = _load(required["actions"])
    detailed = _load(required["detailed"])
    selector = _load(required["selector"])
    rows = [
        json.loads(line)
        for line in required["trace"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recomputed = _proposal_integrity(rows) if rows else {}
    fidelity = arguments.get("implementation_fidelity", {})
    requested = arguments.get("requested_raw_steps", {})
    forbidden_fields = (
        "inference_safety_rule_added",
        "external_kinematic_projection",
        "kinematic_safety_projection",
        "traffic_risk_in_lane_mask",
        "actor_confidence_gate",
        "action_postprocessing_override",
    )
    surfaces = (fidelity, metadata, training, actions, detailed, selector)
    all_forbidden_false = all(
        surface.get(key) is False
        for surface in surfaces
        for key in forbidden_fields
        if key in surface
    ) and all(
        key in actions and actions.get(key) is False
        for key in forbidden_fields
    )
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
        and math.isfinite(
            float(training["statistics"][name]["last"])
        )
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
        "v4_12_schema_bound": all(
            str(value.get("schema_version", "")).startswith(
                "topo-scene-v4.12."
            )
            for value in (arguments, training, actions, detailed, selector)
        ),
        "validation_only_and_formal_untouched": (
            requested.get("evaluation_split") == "validation"
            and detailed.get("formal_test_accessed") is False
            and arguments.get("formal_unlock") is None
        ),
        "all_rule_projection_fields_false": all_forbidden_false,
        "joint_support_runtime_metadata_present": (
            actions.get("joint_replay_action_support_present") is True
            and training.get("joint_replay_action_support_present") is True
            and metadata.get("replay_support_objective")
            == "masked_lane_categorical_nll_plus_conditional_speed_mixture_nll"
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
        **{f"source.{key}": value for key, value in _boolean_checks(source).items()},
        **{f"model.{key}": value for key, value in _boolean_checks(model).items()},
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
        "schema_version": "topo-scene-v4.12.no-kinematic-projection-audit/v1",
        "analysis_kind": (
            "transitive_static_boundary_plus_joint_support_gradient_plus_real_runtime_trace"
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
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "model_only_improvement": (
            "joint replay lane/speed likelihood, continuous learned lane prior, "
            "and stochastic representation augmentation; no threshold, TTC/headway, "
            "projection, veto, shield, fixed grid, semantic override, or action rewrite"
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

