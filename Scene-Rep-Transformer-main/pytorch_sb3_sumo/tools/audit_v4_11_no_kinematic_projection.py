"""Audit that v4.11 improves the learned model without a safety projection.

The audit separates three concepts that are easy to conflate:

1. learned action selection (actor proposals and learned value scores),
2. physical action-space feasibility (an adjacent driving lane must exist), and
3. kinematic/risk projection (threshold, veto, shield, or action rewrite).

Only (1) and the pre-existing structural constraint (2) are permitted.  The
tool is evidence-only and never changes a checkpoint or experiment result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.sb3_torch.hybrid_policy_v4_11_model import (
    ProperCalibratedRankedRiskSACPolicyV411,
    SupportedMixtureHybridActor,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_11_model import (
    ProperCalibratedRankedRiskSACV411,
    continuous_pairwise_ranking_loss,
    soft_target_bernoulli_loss,
    temporal_risk_consistency_loss,
)
from envs.sumo.decision_alignment_v4 import LANE_COMMANDS, decision_alignment_context
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from envs.sumo.sumo_env import SumoSceneEnv


DEFAULT_RUN = (
    ROOT
    / "results_topo_v4_11_dev"
    / "engineering"
    / "smoke_runs"
    / "E0c__prcr_full__left_turn__s1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results_topo_v4_11_dev"
    / "engineering"
    / "no_kinematic_projection_audit.json"
)
SOURCE_FILES = (
    # v4.11 deliberately inherits the v4.10 decoder/actor implementation. Keep
    # the complete transitive action-selection boundary in the audit, rather
    # than hashing only the thin v4.11 identity class.
    "algos/sb3_torch/hybrid_policy_v4_11_model.py",
    "algos/sb3_torch/hybrid_policy_v4_10_model.py",
    "algos/sb3_torch/hybrid_policy_v4_9_model.py",
    "algos/sb3_torch/hybrid_policy_v4.py",
    "algos/sb3_torch/sac_v4_11_model.py",
    "algos/sb3_torch/sac_v4_9_model.py",
    "algos/sb3_torch/replay_buffer_v4_9.py",
    "configs/sb3_configs_v4_11.py",
    "envs/sumo/decision_alignment_v4.py",
    "envs/sumo/paper_env_v4.py",
    "envs/sumo/sumo_env.py",
    "tools/action_diagnostics_v4_11_model.py",
    "tools/checkpoint_decoder_selector_v4_6.py",
    "tools/checkpoint_decoder_selector_v4_9_2.py",
    "tools/run_topo_v4_11_experiments.py",
    "tools/train_sb3_v4_11.py",
    "tools/train_paper_sb3_sumo_v4_11.py",
    "tools/train_paper_sb3_sumo_v4_5.py",
)


class AuditFeatureExtractor(BaseFeaturesExtractor):
    """Small trainable extractor used for executable gradient probes."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=8)
        self.projection = torch.nn.Linear(7, 8)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = torch.cat(
            [observations["state"].float(), observations["lane_action_mask"].float()],
            dim=1,
        )
        return torch.tanh(self.projection(values))


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


def _toy_policy() -> ProperCalibratedRankedRiskSACPolicyV411:
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-100.0, 100.0, shape=(4,), dtype=np.float32),
            "lane_action_mask": gym.spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    torch.manual_seed(20260829)
    return ProperCalibratedRankedRiskSACPolicyV411(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [16, 8], "qf": [16, 8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=AuditFeatureExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
        speed_components=3,
        twin_uncertainty_coef=0.25,
        component_prior_coef=0.05,
    )


def _observation() -> dict[str, torch.Tensor]:
    return {
        "state": torch.tensor([[1.0, -0.4, 0.3, 1.7]], dtype=torch.float32),
        "lane_action_mask": torch.ones((1, 3), dtype=torch.float32),
    }


def source_boundary_audit() -> dict[str, Any]:
    predict_source = inspect.getsource(
        ProperCalibratedRankedRiskSACPolicyV411._predict
    ).lower()
    score_source = inspect.getsource(
        ProperCalibratedRankedRiskSACPolicyV411._risk_adjusted_proposal_values
    ).lower()
    actor_source = inspect.getsource(SupportedMixtureHybridActor).lower()
    learner_source = inspect.getsource(
        ProperCalibratedRankedRiskSACV411.train
    ).lower()
    bce_source = inspect.getsource(soft_target_bernoulli_loss).lower()
    ranking_source = inspect.getsource(
        continuous_pairwise_ranking_loss
    ).lower()
    consistency_source = inspect.getsource(
        temporal_risk_consistency_loss
    ).lower()
    replay_source = inspect.getsource(
        CollisionAwareHorizonReplayBufferV49._add_one
    ).lower()
    paper_step_source = inspect.getsource(PaperSumoSceneEnvV4.step).lower()
    environment_step_source = inspect.getsource(SumoSceneEnv.step).lower()
    adapt_source = inspect.getsource(SumoSceneEnv.adapt_action).lower()
    apply_control_source = inspect.getsource(SumoSceneEnv._apply_control).lower()
    speed_control_source = inspect.getsource(SumoSceneEnv._apply_speed_control).lower()
    configure_source = inspect.getsource(
        SumoSceneEnv._configure_policy_controlled_ego
    ).lower()
    mask_source = inspect.getsource(decision_alignment_context).lower()

    traffic_risk_queries = (
        "getleader",
        "getfollower",
        "headway",
        "time_to_collision",
        "ttc_threshold",
        "collision",
        "unsafe",
    )
    rule_calls = (
        "project_action",
        "safety_projection",
        "safety_shield",
        "rule_fallback",
        "lane_change_veto",
        "confidence_threshold",
        "unsafe_target",
        "postprocess_action",
    )
    source_hashes = {
        relative: _sha256(ROOT / relative) for relative in SOURCE_FILES
    }
    return {
        "source_sha256": source_hashes,
        "transitive_inherited_policy_sources_hashed": all(
            relative in source_hashes
            for relative in (
                "algos/sb3_torch/hybrid_policy_v4_11_model.py",
                "algos/sb3_torch/hybrid_policy_v4_10_model.py",
                "algos/sb3_torch/hybrid_policy_v4_9_model.py",
                "algos/sb3_torch/hybrid_policy_v4.py",
            )
        ),
        "resolved_decoder_owner_is_frozen_v4_10_model": (
            ProperCalibratedRankedRiskSACPolicyV411._predict.__qualname__.split(
                ".", 1
            )[0]
            == "SharedRiskSupportedMixtureSACPolicyV410"
        ),
        "resolved_score_owner_is_frozen_v4_10_model": (
            ProperCalibratedRankedRiskSACPolicyV411
            ._risk_adjusted_proposal_values.__qualname__.split(".", 1)[0]
            == "SharedRiskSupportedMixtureSACPolicyV410"
        ),
        "learned_proposals_are_actor_head_outputs": all(
            token in actor_source
            for token in (
                "self.component_logits(latent)",
                "self.speed_mean(latent)",
                "self._sample_squashed_speeds",
            )
        ),
        "decoder_uses_learned_reward_collision_uncertainty_score": all(
            token in score_source
            for token in (
                "minimum_reward_q",
                "maximum_collision_value",
                "learned_uncertainty",
                "component_log_probabilities",
            )
        ),
        "decoder_masks_only_physical_lane_feasibility": (
            "feasible = batch.action_mask" in predict_source
            and "masked_fill(~feasible, -th.inf)" in predict_source
        ),
        "decoder_is_plain_flattened_argmax": (
            "flattened = masked.reshape" in predict_source
            and "selected_flat = flattened.argmax(dim=1)" in predict_source
        ),
        "decoder_returns_exact_selected_actor_proposal": (
            "selected_actions = actor._select_proposals" in predict_source
            and predict_source.rstrip().endswith("return selected_actions")
        ),
        "decoder_has_no_keep_tie_semantic_override": (
            "keep_command_index" not in predict_source
            and "keep_was_exact_tied_maximum" not in predict_source
            and '"semantic_tie_override_used": false' in predict_source
        ),
        "decoder_has_no_rule_projection_call": all(
            token not in predict_source for token in rule_calls
        ),
        "decoder_has_no_fixed_speed_grid": (
            "linspace" not in predict_source
            and "fixed_speed_candidates" not in predict_source
            and '"fixed_speed_grid_used": false' in predict_source
        ),
        "collision_label_is_observed_environment_event": (
            'info["collision"]' in replay_source
        ),
        "learner_uses_joint_learned_reward_collision_objectives": (
            "collision_critic_loss" in learner_source
            and "critic_loss" in learner_source
            and "encoder_optimizer.step()" in learner_source
        ),
        "collision_loss_is_proper_soft_bernoulli_nll": (
            "binary_cross_entropy_with_logits" in bce_source
            and "target" in bce_source
        ),
        "collision_ranking_is_continuous_without_fixed_margin": (
            "target_difference.abs()" in ranking_source
            and "functional.softplus" in ranking_source
            and "fixed_margin" not in ranking_source
        ),
        "ema_risk_teacher_is_detached": (
            ".detach()" in consistency_source
            and "mse_loss" in consistency_source
            and "teacher_current_collision_logits" in learner_source
            and "with th.no_grad()" in learner_source
        ),
        "paper_environment_passes_policy_action_unchanged": (
            "super().step(action)" in paper_step_source
            and "clip(action" not in paper_step_source
        ),
        "environment_only_clips_to_declared_action_domain": (
            "np.clip(action_array, -1.0, 1.0)" in environment_step_source
            and "target_speed, lane_command = self.adapt_action(action_array)"
            in environment_step_source
        ),
        "environment_action_adapter_is_parameterization_not_risk_query": (
            "* 5.0" in adapt_source
            and all(token not in adapt_source for token in traffic_risk_queries)
        ),
        "direct_speed_profile_applies_requested_speed_exactly": (
            'ego_control_profile == "direct"' in speed_control_source
            and "effective = requested" in speed_control_source
        ),
        "optional_dynamics_proxy_is_outside_direct_branch": (
            'ego_control_profile == "direct"' in speed_control_source
            and "_smarts_curve_speed_limit_from_headings" in speed_control_source
        ),
        "lane_mask_has_no_traffic_risk_query": all(
            token not in mask_source for token in traffic_risk_queries
        ),
        "lane_mask_only_checks_adjacent_driving_lane": (
            "_driving_lanes" in mask_source and "target_rank" in mask_source
        ),
        "lane_application_has_no_traffic_risk_query": all(
            token not in apply_control_source for token in traffic_risk_queries
        ),
        "sumo_safe_speed_and_lane_veto_disabled": (
            "setspeedmode(ego_id, 0)" in configure_source
            and "setlanechangemode(ego_id, 0)" in configure_source
        ),
        "classification": {
            "learned_model_selection": (
                "learned actor proposals scored by learned twin reward, collision, "
                "and uncertainty estimates"
            ),
            "structural_feasibility_constraint": (
                "mask only removes lane commands for which no adjacent driving lane exists"
            ),
            "action_parameterization": (
                "normalized speed is mapped linearly to [0,10] m/s and exact lane codes "
                "map to {-1,0,+1}"
            ),
            "kinematic_or_traffic_risk_projection": "absent",
        },
    }


def model_execution_audit() -> dict[str, Any]:
    policy = _toy_policy()
    observation = _observation()
    actor = policy.actor
    if not isinstance(actor, SupportedMixtureHybridActor):
        raise TypeError("v4.11 toy policy lost its learned mixture actor")

    proposals = actor.all_action_proposals(
        observation, deterministic_speed=True
    )
    policy.begin_target_decoder_recording()
    selected = policy._predict(observation, deterministic=True)
    record = policy.end_target_decoder_recording()[0]
    scores = record["supported_risk_adjusted_score"]
    flattened_scores = [
        float(score)
        for lane in scores
        if lane is not None
        for score in lane
    ]
    expected_flat = int(np.argmax(np.asarray(flattened_scores)))
    expected_lane = expected_flat // actor.speed_components
    expected_component = expected_flat % actor.speed_components
    exact_proposal = proposals.actions[
        0, int(record["selected_lane_index"]), int(record["selected_component_index"])
    ]

    original_score_method = policy._risk_adjusted_proposal_values

    def tied_scores(self, current_observation, batch):
        del self, current_observation
        zeros = torch.zeros(
            batch.actions.shape[:3],
            dtype=batch.actions.dtype,
            device=batch.actions.device,
        )
        return zeros, zeros, zeros, zeros, zeros, zeros

    policy._risk_adjusted_proposal_values = MethodType(tied_scores, policy)
    try:
        policy.begin_target_decoder_recording()
        tied_action = policy._predict(observation, deterministic=True)
        tied_record = policy.end_target_decoder_recording()[0]
    finally:
        policy._risk_adjusted_proposal_values = original_score_method

    encoder_parameters = list(policy.critic.features_extractor.parameters())
    action = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
    reward_value = sum(value.sum() for value in policy.critic(observation, action))
    reward_gradients = torch.autograd.grad(
        reward_value, encoder_parameters, allow_unused=True
    )
    collision_value = sum(
        value.sum() for value in policy.collision_critic(observation, action)
    )
    collision_gradients = torch.autograd.grad(
        collision_value, encoder_parameters, allow_unused=True
    )
    actor_batch = actor.all_action_proposals(
        observation, deterministic_speed=True
    )
    actor_output = (
        actor_batch.actions[..., 0].sum()
        + actor_batch.lane_probabilities.sum()
        + actor_batch.component_probabilities.sum()
    )
    actor_targets = [
        actor.speed_mean.weight,
        actor.component_logits.weight,
        actor.lane_logits.weight,
        *encoder_parameters,
    ]
    actor_gradients = torch.autograd.grad(
        actor_output, actor_targets, allow_unused=True
    )

    ownership = policy.optimizer_parameter_ownership()
    encoder_ids = _parameter_ids(encoder_parameters)
    encoder_optimizer_ids = _optimizer_parameter_ids(policy.encoder_optimizer)
    actor_optimizer_ids = _optimizer_parameter_ids(actor.optimizer)
    reward_optimizer_ids = _optimizer_parameter_ids(policy.critic.optimizer)
    collision_optimizer_ids = _optimizer_parameter_ids(
        policy.collision_critic.optimizer
    )

    selected_numpy = selected.detach().cpu().numpy()[0]
    clipped = np.clip(selected_numpy, -1.0, 1.0)
    target_speed, lane_command = SumoSceneEnv.adapt_action(clipped)

    class VehicleControlProbe:
        def __init__(self) -> None:
            self.speed_commands: list[tuple[str, float]] = []
            self.speed_modes: list[tuple[str, int]] = []
            self.lane_modes: list[tuple[str, int]] = []

        def setSpeed(self, ego_id: str, speed: float) -> None:
            self.speed_commands.append((ego_id, float(speed)))

        def setSpeedMode(self, ego_id: str, mode: int) -> None:
            self.speed_modes.append((ego_id, int(mode)))

        def setLaneChangeMode(self, ego_id: str, mode: int) -> None:
            self.lane_modes.append((ego_id, int(mode)))

    vehicle_probe = VehicleControlProbe()
    direct_env = object.__new__(SumoSceneEnv)
    direct_env.ego_control_profile = "direct"
    direct_env._connection = type(
        "ConnectionProbe", (), {"vehicle": vehicle_probe}
    )()
    direct_env.specification = type(
        "SpecificationProbe",
        (),
        {"source_observation_contract": "carla", "ego_id": "ego-probe"},
    )()
    dynamics_queries = 0

    def fail_if_dynamics_proxy_is_queried(self) -> np.ndarray:
        del self
        nonlocal dynamics_queries
        dynamics_queries += 1
        raise AssertionError("direct control queried the optional dynamics proxy")

    direct_env._control_path_headings = MethodType(
        fail_if_dynamics_proxy_is_queried, direct_env
    )
    requested_probe_speed = 6.375
    effective_probe_speed = direct_env._apply_speed_control(requested_probe_speed)
    direct_env._configure_policy_controlled_ego("ego-probe")
    return {
        "learned_component_tensor_shape": list(proposals.actions.shape),
        "actual_score_argmax_matches_record": (
            int(record["selected_lane_index"]) == expected_lane
            and int(record["selected_component_index"]) == expected_component
        ),
        "selected_action_equals_exact_actor_proposal": bool(
            torch.allclose(selected[0], exact_proposal, rtol=0.0, atol=1e-7)
        ),
        "selected_action_inside_declared_domain": bool(
            np.isfinite(selected_numpy).all()
            and (selected_numpy >= -1.0).all()
            and (selected_numpy <= 1.0).all()
        ),
        "environment_domain_clip_is_identity_for_model_action": bool(
            np.array_equal(selected_numpy, clipped)
        ),
        "environment_lane_parameterization_preserves_exact_model_code": (
            int(lane_command) == int(round(float(selected_numpy[1])))
            and float(selected_numpy[1]) in (-1.0, 0.0, 1.0)
        ),
        "environment_speed_parameterization_matches_linear_contract": bool(
            abs(target_speed - (float(selected_numpy[0]) + 1.0) * 5.0) <= 1e-6
        ),
        "direct_control_execution_preserves_requested_speed_exactly": (
            effective_probe_speed == requested_probe_speed
            and vehicle_probe.speed_commands
            == [("ego-probe", requested_probe_speed)]
        ),
        "direct_control_execution_never_queries_dynamics_proxy": (
            dynamics_queries == 0
        ),
        "sumo_internal_safe_modes_disabled_by_execution": (
            vehicle_probe.speed_modes == [("ego-probe", 0)]
            and vehicle_probe.lane_modes == [("ego-probe", 0)]
        ),
        "exact_tie_uses_first_flattened_feasible_argmax": (
            int(tied_record["selected_lane_index"]) == 0
            and int(tied_record["selected_component_index"]) == 0
            and int(tied_record["selected_lane"]) == LANE_COMMANDS[0]
            and bool(torch.allclose(tied_action[0], proposals.actions[0, 0, 0]))
        ),
        "semantic_tie_override_used": bool(
            tied_record.get("semantic_tie_override_used")
        ),
        "reward_objective_reaches_shared_encoder": any(
            _gradient_present(value) for value in reward_gradients
        ),
        "collision_objective_reaches_shared_encoder": any(
            _gradient_present(value) for value in collision_gradients
        ),
        "actor_heads_receive_actor_gradient": {
            "speed_mean": _gradient_present(actor_gradients[0]),
            "component_logits": _gradient_present(actor_gradients[1]),
            "lane_logits": _gradient_present(actor_gradients[2]),
        },
        "actor_path_updates_shared_encoder": any(
            _gradient_present(value) for value in actor_gradients[3:]
        ),
        "encoder_has_single_optimizer_owner": (
            encoder_ids == encoder_optimizer_ids
            and encoder_ids.isdisjoint(actor_optimizer_ids)
            and encoder_ids.isdisjoint(reward_optimizer_ids)
            and encoder_ids.isdisjoint(collision_optimizer_ids)
        ),
        "optimizer_ownership": ownership,
    }


def runtime_run_audit(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"available": False, "required": False}
    required = {
        "arguments": run_dir / "arguments.json",
        "method_metadata": run_dir / "method_metadata.json",
        "actions": run_dir / "action_diagnostics.json",
        "trace": run_dir / "action_diagnostics_decisions.jsonl",
        "detailed": run_dir / "paper_evaluation_detailed.json",
        "selector": run_dir / "selector" / "receipt.json",
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
    metadata = _load(required["method_metadata"])
    actions = _load(required["actions"])
    detailed = _load(required["detailed"])
    selector = _load(required["selector"])
    rows = [
        json.loads(line)
        for line in required["trace"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requested = arguments.get("requested_raw_steps", {})
    fidelity = arguments.get("implementation_fidelity", {})
    action_domain_identity = True
    trace_has_new_operator = True
    direct_speed_identity = True
    normalized_speed_parameterization_identity = True
    lane_parameterization_identity = True
    for row in rows:
        normalized = np.asarray(
            [row.get("action_longitudinal"), row.get("action_lateral")],
            dtype=np.float64,
        )
        action_domain_identity &= bool(
            np.isfinite(normalized).all()
            and np.array_equal(normalized, np.clip(normalized, -1.0, 1.0))
        )
        target_speed = row.get("target_speed_mps")
        effective_speed = row.get("effective_target_speed_mps")
        longitudinal = row.get("action_longitudinal")
        lateral = row.get("action_lateral")
        lane_command = row.get("lane_command")
        direct_speed_identity &= bool(
            isinstance(target_speed, (int, float))
            and isinstance(effective_speed, (int, float))
            and np.isfinite(target_speed)
            and np.isfinite(effective_speed)
            and abs(float(target_speed) - float(effective_speed)) <= 1e-6
        )
        normalized_speed_parameterization_identity &= bool(
            isinstance(target_speed, (int, float))
            and isinstance(longitudinal, (int, float))
            and abs(float(target_speed) - (float(longitudinal) + 1.0) * 5.0)
            <= 1e-6
        )
        lane_parameterization_identity &= bool(
            isinstance(lateral, (int, float))
            and float(lateral) in (-1.0, 0.0, 1.0)
            and isinstance(lane_command, int)
            and int(lane_command) == int(float(lateral))
        )
        decoder = row.get("target_critic_decoder", {})
        trace_has_new_operator &= (
            decoder.get("selection_operator")
            == "torch_argmax_flattened_feasible_model_scores"
            and decoder.get("semantic_tie_override_used") is False
            and decoder.get("action_rewritten") is False
            and decoder.get("fixed_speed_grid_used") is False
        )
    exact_rates = {
        key: actions.get(key)
        for key in (
            "selected_decoder_exact_model_match_rate",
            "selected_decoder_exact_action_match_rate",
            "selected_action_mask_feasible_rate",
            "exact_supported_mixture_model_argmax_rate",
            "supported_mixture_score_equation_match_rate",
            "selected_speed_exact_proposal_match_rate",
            "learned_proposal_source_rate",
            "no_action_rewrite_rate",
            "no_semantic_tie_override_rate",
        )
    }
    return {
        "available": True,
        "required": True,
        "run_dir": str(run_dir.resolve()),
        "run_artifact_sha256": {
            name: _sha256(path) for name, path in required.items()
        },
        "decision_records": len(rows),
        "ego_control_profile_is_explicit_direct": (
            requested.get("ego_control_profile") == "direct"
        ),
        "validation_only_and_formal_locked": (
            requested.get("evaluation_split") == "validation"
            and arguments.get("formal_unlock") is None
            and detailed.get("formal_test_accessed") is False
        ),
        "metadata_has_no_rule_or_projection": all(
            value is False
            for value in (
                fidelity.get("inference_safety_rule_added"),
                fidelity.get("external_kinematic_projection"),
                fidelity.get("action_postprocessing_override"),
                metadata.get("inference_safety_rule_added"),
                metadata.get("external_kinematic_projection"),
                metadata.get("action_postprocessing_override"),
                actions.get("inference_safety_rule_added"),
                actions.get("external_kinematic_projection"),
                actions.get("action_postprocessing_override"),
            )
        ),
        "legacy_confidence_rule_metadata_absent": (
            "actor_non_keep_confidence_threshold" not in fidelity
            and "actor_non_keep_confidence_threshold" not in metadata
        ),
        "fixed_speed_grid_absent": (
            fidelity.get("fixed_speed_grid_at_inference") is False
            and metadata.get("fixed_speed_grid_at_inference") is False
            and actions.get("fixed_speed_grid_at_inference") is False
        ),
        "selector_is_target_only_and_no_fusion": (
            selector.get("deployment_decoder_candidates") == ["target_critic"]
            and selector.get("candidate_count") == 2
            and selector.get("fusion_candidate_present") is False
            and selector.get("actor_confidence_threshold_present") is False
        ),
        "policy_parameter_state_preserved_by_selector": (
            selector.get("policy_parameter_state_preserved") is True
            and selector.get("source_policy_parameter_state_sha256")
            == selector.get("selected_model_parameter_state_sha256")
        ),
        "all_exact_model_action_rates_are_one": bool(exact_rates)
        and all(value == 1.0 for value in exact_rates.values()),
        "exact_rates": exact_rates,
        "trace_uses_new_plain_argmax_operator": bool(rows)
        and trace_has_new_operator,
        "environment_domain_clip_is_identity_for_all_traced_model_actions": (
            bool(rows) and action_domain_identity
        ),
        "direct_speed_command_preserved_for_all_traced_actions": (
            bool(rows) and direct_speed_identity
        ),
        "normalized_speed_parameterization_exact_for_all_traced_actions": (
            bool(rows) and normalized_speed_parameterization_identity
        ),
        "lane_code_parameterization_exact_for_all_traced_actions": (
            bool(rows) and lane_parameterization_identity
        ),
        "optimizer_owners_do_not_overlap": (
            actions.get("optimizer_ownership", {}).get("overlap_count") == 0
        ),
    }


def _required_boolean_values(section: Mapping[str, Any]) -> dict[str, bool]:
    return {
        key: value
        for key, value in section.items()
        if isinstance(value, bool)
        and key not in {"semantic_tie_override_used", "actor_path_updates_shared_encoder"}
    }


def build_audit(
    run_dir: Path | None = None, *, require_runtime: bool = False
) -> dict[str, Any]:
    source = source_boundary_audit()
    model = model_execution_audit()
    runtime = runtime_run_audit(run_dir)
    positive_checks = {
        **{f"source.{key}": value for key, value in _required_boolean_values(source).items()},
        **{f"model.{key}": value for key, value in _required_boolean_values(model).items()},
    }
    expected_false = {
        "model.semantic_tie_override_used": model["semantic_tie_override_used"],
        "model.actor_path_updates_shared_encoder": model[
            "actor_path_updates_shared_encoder"
        ],
    }
    if runtime.get("available"):
        positive_checks.update(
            {
                f"runtime.{key}": value
                for key, value in _required_boolean_values(runtime).items()
            }
        )
    integrity_passed = (
        all(positive_checks.values())
        and not any(expected_false.values())
        and (not require_runtime or runtime.get("available") is True)
    )
    return {
        "schema_version": "topo-scene-v4.11.no-kinematic-projection-audit/v1",
        "analysis_kind": "static_call_chain_plus_dynamic_model_gradient_and_runtime_trace",
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
        "failed_positive_checks": sorted(
            key for key, value in positive_checks.items() if not value
        ),
        "failed_expected_false_checks": sorted(
            key for key, value in expected_false.items() if value
        ),
        "model_only_improvement": (
            "proper soft-target Bernoulli risk learning, continuous pairwise risk "
            "ordering, and detached EMA temporal consistency on the inherited "
            "learned proposal/value model; no threshold, fixed grid, projection, "
            "veto, shield, tie preference, or post-decoder action rewrite"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="omit the real-run requirement; intended only for pre-smoke checks",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = None if args.source_only else args.run_dir
    audit = build_audit(run_dir, require_runtime=not args.source_only)
    if not audit["integrity_passed"]:
        raise ValueError(
            "v4.11 no-kinematic-projection audit failed: "
            f"{audit['failed_positive_checks']} "
            f"{audit['failed_expected_false_checks']}"
        )
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
