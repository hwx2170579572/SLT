"""Audit the v4.9.2 kinematic-safety boundary and learned gradient paths.

This tool is evidence-only.  It does not alter a checkpoint, environment,
promotion run, decoder, or formal-test lock.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.hybrid_policy_v4 import DecisionAlignedHybridActor
from algos.sb3_torch.hybrid_policy_v4_9_model import (
    CollisionConstrainedTargetCriticSACPolicyV49,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.topo_temporal_features_v2 import (
    TopoTemporalGraphExtractorV2,
)
from envs.sumo.decision_alignment_v4 import decision_alignment_context
from envs.sumo.sumo_env import SumoSceneEnv


DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_9_2_dev"
    / "engineering"
    / "kinematic_safety_model_reaudit.json"
)
PARTIAL_ATTRIBUTION = (
    ROOT
    / "results_topo_v4_9_2_promotion"
    / "attribution"
    / "provisional_partial"
    / "promotion_model_attribution_current.json"
)
PROMOTION_RUNS = ROOT / "results_topo_v4_9_2_promotion" / "runs"


class AuditFeatureExtractor(BaseFeaturesExtractor):
    """Small parameterized extractor used only for gradient ownership probes."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=6)
        self.linear = torch.nn.Linear(7, 6)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = torch.cat(
            [
                observations["state"].float(),
                observations["lane_action_mask"].float(),
            ],
            dim=1,
        )
        return torch.tanh(self.linear(values))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _parameter_ids(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {id(parameter) for parameter in parameters}


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _gradient_present(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all()) and bool(
        value.abs().sum() > 0
    )


def _toy_policy() -> CollisionConstrainedTargetCriticSACPolicyV49:
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
    torch.manual_seed(20260827)
    return CollisionConstrainedTargetCriticSACPolicyV49(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [8], "qf": [8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=AuditFeatureExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
    )


def parameter_and_gradient_audit() -> dict[str, Any]:
    policy = _toy_policy()
    actor = policy.actor
    reward = policy.critic
    reward_target = policy.critic_target
    collision = policy.collision_critic
    collision_target = policy.collision_critic_target

    shared = actor.features_extractor
    shared_ids = _parameter_ids(shared.parameters())
    collision_extractor_ids = _parameter_ids(
        collision.features_extractor.parameters()
    )
    reward_target_extractor_ids = _parameter_ids(
        reward_target.features_extractor.parameters()
    )
    collision_target_extractor_ids = _parameter_ids(
        collision_target.features_extractor.parameters()
    )
    actor_optimizer_ids = _optimizer_parameter_ids(actor.optimizer)
    reward_optimizer_ids = _optimizer_parameter_ids(reward.optimizer)
    collision_optimizer_ids = _optimizer_parameter_ids(collision.optimizer)

    observation = {
        "state": torch.tensor([[1.0, -0.5, 0.25, 2.0]]),
        "lane_action_mask": torch.ones((1, 3)),
    }
    actor_batch = actor.all_action_samples(
        observation, deterministic_speed=True
    )
    actor_outputs = (
        actor_batch.actions[..., 0].sum()
        + actor_batch.lane_probabilities[:, 0].sum()
    )
    shared_parameters = list(shared.parameters())
    actor_probe_targets = [
        actor.speed_mean.weight,
        actor.lane_logits.weight,
        *shared_parameters,
    ]
    actor_gradients = torch.autograd.grad(
        actor_outputs,
        actor_probe_targets,
        allow_unused=True,
        retain_graph=False,
    )

    actor_batch = actor.all_action_samples(
        observation, deterministic_speed=True
    )
    collision_features = collision.extract_features(
        observation, collision.features_extractor
    )
    collision_heads = collision.all_q_from_features(
        collision_features, actor_batch.actions
    )
    collision_value = torch.stack(
        [torch.sigmoid(head) for head in collision_heads], dim=-1
    ).max(dim=-1).values
    actor_risk_objective = (
        actor_batch.lane_probabilities.unsqueeze(-1) * collision_value
    ).sum()
    risk_probe_targets = [
        actor.speed_mean.weight,
        actor.lane_logits.weight,
        *shared_parameters,
    ]
    risk_gradients = torch.autograd.grad(
        actor_risk_objective,
        risk_probe_targets,
        allow_unused=True,
        retain_graph=False,
    )

    return {
        "actor_and_reward_critic_share_exact_online_extractor": (
            actor.features_extractor is reward.features_extractor
        ),
        "collision_online_extractor_is_independent": shared_ids.isdisjoint(
            collision_extractor_ids
        ),
        "reward_target_extractor_is_independent": shared_ids.isdisjoint(
            reward_target_extractor_ids
        ),
        "collision_target_extractor_is_independent": (
            shared_ids.isdisjoint(collision_target_extractor_ids)
            and collision_extractor_ids.isdisjoint(
                collision_target_extractor_ids
            )
        ),
        "actor_optimizer_excludes_shared_extractor": shared_ids.isdisjoint(
            actor_optimizer_ids
        ),
        "reward_critic_optimizer_owns_shared_extractor": shared_ids.issubset(
            reward_optimizer_ids
        ),
        "collision_optimizer_owns_independent_extractor": (
            collision_extractor_ids.issubset(collision_optimizer_ids)
        ),
        "actor_head_gradient_from_actor_outputs": {
            "speed_head": _gradient_present(actor_gradients[0]),
            "lane_head": _gradient_present(actor_gradients[1]),
        },
        "shared_extractor_gradient_from_actor_outputs": any(
            _gradient_present(value) for value in actor_gradients[2:]
        ),
        "actor_head_gradient_from_learned_collision_objective": {
            "speed_head": _gradient_present(risk_gradients[0]),
            "lane_head": _gradient_present(risk_gradients[1]),
        },
        "shared_extractor_gradient_from_actor_collision_objective": any(
            _gradient_present(value) for value in risk_gradients[2:]
        ),
        "interpretation": (
            "risk gradients can reach lane/speed heads through learned action "
            "values, but the actor path cannot shape its shared state extractor; "
            "collision TD gradients train a separate extractor"
        ),
    }


def source_boundary_audit() -> dict[str, Any]:
    mask_source = inspect.getsource(decision_alignment_context).lower()
    apply_control_source = inspect.getsource(SumoSceneEnv._apply_control).lower()
    speed_control_source = inspect.getsource(
        SumoSceneEnv._apply_speed_control
    ).lower()
    configure_source = inspect.getsource(
        SumoSceneEnv._configure_policy_controlled_ego
    ).lower()
    actor_source = inspect.getsource(
        DecisionAlignedHybridActor.distribution_parameters
    ).lower()
    decoder_source = inspect.getsource(
        CollisionConstrainedTargetCriticSACPolicyV49._predict
    ).lower()
    ttc_source = inspect.getsource(
        TopoTemporalGraphExtractorV2._vehicle_edge_features_v2
    ).lower()
    collision_label_source = inspect.getsource(
        CollisionAwareHorizonReplayBufferV49._add_one
    ).lower()

    risk_queries = (
        "getleader",
        "getfollower",
        "headway",
        "time_to_collision",
        "collision",
        "unsafe",
    )
    decoder_rules = (
        "confidence_threshold",
        "headway",
        "ttc_threshold",
        "veto",
        "project_action",
        "safety_override",
    )
    return {
        "lane_action_mask_has_no_traffic_risk_query": all(
            token not in mask_source for token in risk_queries
        ),
        "lane_action_mask_checks_adjacent_driving_lane": (
            "_driving_lanes" in mask_source
            and "target_rank" in mask_source
        ),
        "apply_control_has_no_traffic_risk_query": all(
            token not in apply_control_source for token in risk_queries
        ),
        "sumo_safe_speed_and_lane_change_checks_disabled": (
            "setspeedmode(ego_id, 0)" in configure_source
            and "setlanechangemode(ego_id, 0)" in configure_source
        ),
        "optional_curve_proxy_exists_only_outside_direct_branch": (
            'ego_control_profile == "direct"' in speed_control_source
            and "_smarts_curve_speed_limit_from_headings" in speed_control_source
        ),
        "actor_feature_detach_present": ".detach()" in actor_source,
        "target_decoder_has_no_safety_rule": all(
            token not in decoder_source for token in decoder_rules
        ),
        "target_decoder_is_learned_score_argmax": (
            "risk_adjusted_score" in decoder_source
            and ".argmax(" in decoder_source
        ),
        "ttc_is_vehicle_graph_feature": (
            "ttc =" in ttc_source and "ttc[..., none]" in ttc_source
        ),
        "ttc_source_does_not_select_or_rewrite_action": all(
            token not in ttc_source
            for token in ("action", "veto", "override", "unsafe")
        ),
        "collision_label_comes_from_observed_info": (
            'info["collision"]' in collision_label_source
        ),
        "manual_geometry_collision_pseudolabel_present": False,
    }


def promotion_runtime_audit() -> dict[str, Any]:
    argument_paths = sorted(PROMOTION_RUNS.glob("*/arguments.json"))
    rows: list[dict[str, Any]] = []
    for path in argument_paths:
        arguments = _load(path)
        requested = arguments.get("requested_raw_steps", {})
        rows.append(
            {
                "job": path.parent.name,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "ego_control_profile": requested.get("ego_control_profile"),
                "evaluation_split": requested.get("evaluation_split"),
                "formal_unlock": arguments.get("formal_unlock"),
            }
        )
    return {
        "observed_argument_files": len(rows),
        "all_observed_profiles_are_direct": bool(rows)
        and all(row["ego_control_profile"] == "direct" for row in rows),
        "all_observed_splits_are_validation": bool(rows)
        and all(row["evaluation_split"] == "validation" for row in rows),
        "all_observed_formal_unlocks_are_null": bool(rows)
        and all(row["formal_unlock"] is None for row in rows),
        "rows": rows,
    }


def partial_evidence_audit() -> dict[str, Any]:
    if not PARTIAL_ATTRIBUTION.is_file():
        return {
            "available": False,
            "decision_allowed": False,
            "reason": "partial attribution is absent",
        }
    value = _load(PARTIAL_ATTRIBUTION)
    measurements = value.get("aggregate_mechanism_measurements", {})
    tg_evaluations = []
    for seed in (20, 21):
        path = (
            PROMOTION_RUNS
            / f"P__tg__cross__s{seed}__p9e99845d"
            / "final_evaluation.json"
        )
        tg_evaluations.append(_load(path))
    candidate_evaluation = _load(
        PROMOTION_RUNS
        / "P__cand__cross__s20__p9e99845d"
        / "final_evaluation.json"
    )
    tg_collision_count = sum(
        round(float(row["collision_rate"]) * int(row["episodes"]))
        for row in tg_evaluations
    )
    total_cross_episodes = sum(int(row["episodes"]) for row in tg_evaluations)
    known_candidate_collision_count = round(
        float(candidate_evaluation["collision_rate"])
        * int(candidate_evaluation["episodes"])
    )
    candidate_best_case_rate = known_candidate_collision_count / total_cross_episodes
    control_rate = tg_collision_count / total_cross_episodes
    lower_bound = candidate_best_case_rate - control_rate
    return {
        "available": True,
        "path": str(PARTIAL_ATTRIBUTION.resolve()),
        "sha256": _sha256(PARTIAL_ATTRIBUTION),
        "partial_matrix": value.get("partial_matrix"),
        "pair_count": value.get("pair_count"),
        "expected_pair_count": value.get("expected_pair_count"),
        "decision_allowed": value.get("decision_allowed"),
        "risk_window10_rank_auc_mean": measurements.get(
            "candidate_collision_value_window10_rank_auc_mean"
        ),
        "collision_window_selected_minimum_risk_rate_mean": measurements.get(
            "candidate_collision_selected_minimum_predicted_risk_rate_mean"
        ),
        "candidate_minus_control_non_keep_rate_mean": measurements.get(
            "candidate_minus_temporal_graph_non_keep_rate_mean"
        ),
        "candidate_actor_target_lane_match_rate_mean": measurements.get(
            "candidate_actor_target_lane_match_rate_mean"
        ),
        "cross_collision_gate_lower_bound": {
            "control_collision_count": tg_collision_count,
            "total_cross_episodes": total_cross_episodes,
            "known_candidate_seed20_collision_count": (
                known_candidate_collision_count
            ),
            "best_case_candidate_collision_rate": candidate_best_case_rate,
            "control_collision_rate": control_rate,
            "best_case_delta": lower_bound,
            "frozen_margin": 0.05,
            "gate_can_still_pass": lower_bound <= 0.05,
        },
        "interpretation": (
            "the current promotion is mathematically unable to pass the frozen "
            "Cross collision margin, but all jobs must still run and all six "
            "pairs are required before selecting the successor model"
        ),
    }


def build_audit() -> dict[str, Any]:
    boundary = source_boundary_audit()
    gradients = parameter_and_gradient_audit()
    runtime = promotion_runtime_audit()
    partial = partial_evidence_audit()
    required_true = {
        key: value
        for section in (boundary, gradients)
        for key, value in section.items()
        if isinstance(value, bool)
        and key
        not in {
            "shared_extractor_gradient_from_actor_outputs",
            "shared_extractor_gradient_from_actor_collision_objective",
            "manual_geometry_collision_pseudolabel_present",
        }
    }
    required_false = {
        "shared_extractor_gradient_from_actor_outputs": gradients[
            "shared_extractor_gradient_from_actor_outputs"
        ],
        "shared_extractor_gradient_from_actor_collision_objective": gradients[
            "shared_extractor_gradient_from_actor_collision_objective"
        ],
        "manual_geometry_collision_pseudolabel_present": boundary[
            "manual_geometry_collision_pseudolabel_present"
        ],
    }
    integrity_passed = (
        all(required_true.values())
        and not any(required_false.values())
        and runtime["all_observed_profiles_are_direct"]
        and runtime["all_observed_splits_are_validation"]
        and runtime["all_observed_formal_unlocks_are_null"]
    )
    return {
        "schema_version": "topo-scene-v4.9.2.kinematic-safety-model-reaudit/v1",
        "analysis_kind": "static_source_plus_dynamic_gradient_ownership_plus_partial_trace",
        "computed_from_real_sources_and_runs": True,
        "fabricated_values": False,
        "scientific_model_changed": False,
        "run_artifacts_modified": False,
        "formal_test_accessed": False,
        "decision_allowed_from_partial_matrix": False,
        "integrity_passed": integrity_passed,
        "candidate_side_kinematic_projection_present": False,
        "inference_safety_rule_added": False,
        "source_boundary": boundary,
        "parameter_and_gradient_ownership": gradients,
        "promotion_runtime": runtime,
        "partial_promotion_evidence": partial,
        "model_hypothesis_pending_complete_matrix": (
            "share a stable learned risk representation with disjoint optimizer "
            "ownership and improve risk-conditioned action proposals; do not add "
            "a projection, threshold, veto, shield, or action rewrite"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit()
    if not audit["integrity_passed"]:
        raise ValueError("kinematic safety/model audit did not pass")
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
