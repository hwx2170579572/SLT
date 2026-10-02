from __future__ import annotations

import argparse
import json
import types
from typing import Any

import numpy as np
import torch

from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.hybrid_policy_v4_3 import (
    TargetCriticDecisionAlignedSACPolicyV43,
)
from algos.sb3_torch.sac_v4_2 import FactorizedEntropyHybridSACV42
from algos.sb3_torch.sac_v4_3 import TargetCriticLaneDecoderSACV43
from configs.sb3_configs_v4_2 import make_model_v4_2
from configs.sb3_configs_v4_3 import V43_IMPLEMENTATION_IDS, make_model_v4_3
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.action_diagnostics_v4_3 import _target_event_metrics
from tools.train_sb3_v4_3 import _method_hyperparameters, _validate_formal_unlock


def _carla_env() -> PaperSumoSceneEnvV4:
    return PaperSumoSceneEnvV4(
        scenario="carla",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )


def _model_and_tensor_observation(mask: list[float]):
    env = _carla_env()
    observation, _ = env.reset(seed=30_000)
    observation = {key: np.array(value, copy=True) for key, value in observation.items()}
    observation["lane_action_mask"] = np.asarray(mask, dtype=np.float32)
    model = make_model_v4_3(
        "topo_v4_3_target_critic_decoder",
        env,
        scenario="carla",
        learning_starts=0,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    tensor_observation, _ = model.policy.obs_to_tensor(observation)
    return env, model, observation, tensor_observation


def _install_target_q(model: Any, first: list[float], second: list[float]) -> None:
    def fixed_q(_critic, features, actions):
        del features
        dtype = actions.dtype
        device = actions.device
        return (
            torch.tensor(first, dtype=dtype, device=device).view(1, 3, 1),
            torch.tensor(second, dtype=dtype, device=device).view(1, 3, 1),
        )

    model.critic_target.all_q_from_features = types.MethodType(
        fixed_q, model.critic_target
    )


def test_v4_3_registry_and_metadata_freeze_only_decoder_change() -> None:
    env, model, _, _ = _model_and_tensor_observation([1.0, 1.0, 1.0])
    try:
        assert isinstance(model, TargetCriticLaneDecoderSACV43)
        assert isinstance(model.policy, TargetCriticDecisionAlignedSACPolicyV43)
        metadata = model.v4_method_metadata
        assert metadata["implementation_id"] == V43_IMPLEMENTATION_IDS[
            "topo_v4_3_target_critic_decoder"
        ]
        assert metadata["single_change"] == "deterministic_target_critic_lane_decoder"
        assert metadata["deterministic_lane_decoder"] == (
            "argmax_feasible_min_target_twin_q"
        )
        assert metadata["stochastic_lane_decoder"] == (
            "masked_categorical_actor_sampling"
        )
        assert metadata["reward_unchanged"] is True
        assert model.lane_entropy_scale == 0.0
        assert model.target_entropy == -1.0
        hyperparameters = _method_hyperparameters(
            "topo_v4_3_target_critic_decoder"
        )
        assert hyperparameters["deterministic_lane_decoder"] == (
            "argmax_feasible_min_target_twin_q"
        )
    finally:
        env.close()


def test_stochastic_predict_delegates_to_actor_without_querying_target_critic() -> None:
    env, model, _, tensor_observation = _model_and_tensor_observation(
        [1.0, 1.0, 1.0]
    )
    try:
        expected = torch.tensor([[0.25, -1.0]], dtype=torch.float32)

        def actor_forward(_actor, observation, deterministic=False):
            del observation
            assert deterministic is False
            return expected

        def forbidden(*_args, **_kwargs):
            raise AssertionError("stochastic inference must not query target critic")

        model.actor.forward = types.MethodType(actor_forward, model.actor)
        model.critic_target.extract_features = forbidden
        actual = model.policy._predict(tensor_observation, deterministic=False)
        assert actual is expected
    finally:
        env.close()


def test_deterministic_decoder_uses_conservative_target_q_mask_and_lane_speed() -> None:
    env, model, _, tensor_observation = _model_and_tensor_observation(
        [1.0, 1.0, 0.0]
    )
    try:
        # BasePolicy.predict() applies this deployment-mode transition before
        # invoking _predict(); mirror it here while inspecting tensor outputs.
        model.policy.set_training_mode(False)
        with source_evaluation_augmentation(model), torch.no_grad():
            candidates = model.actor.all_action_samples(
                tensor_observation, deterministic_speed=True
            ).actions
            # min twin-Q is [2, 1, 50]; lane +1 is infeasible despite its high Q.
            _install_target_q(model, [3.0, 1.0, 100.0], [2.0, 4.0, 50.0])
            model.policy.begin_target_decoder_recording()
            actual = model.policy._predict(tensor_observation, deterministic=True)
            records = model.policy.end_target_decoder_recording()
        torch.testing.assert_close(actual, candidates[:, 0, :])
        assert float(actual[0, 1]) == -1.0
        assert records[0]["selected_lane"] == -1
        assert records[0]["minimum_target_twin_q"] == [2.0, 1.0, None]
        assert records[0]["selected_q_minus_keep_q"] == 1.0
    finally:
        env.close()


def test_exact_target_q_tie_prefers_keep() -> None:
    env, model, _, tensor_observation = _model_and_tensor_observation(
        [1.0, 1.0, 1.0]
    )
    try:
        _install_target_q(model, [2.0, 2.0, 0.0], [3.0, 2.0, 1.0])
        with torch.no_grad():
            actual = model.policy._predict(tensor_observation, deterministic=True)
        assert float(actual[0, 1]) == 0.0
    finally:
        env.close()


def test_target_event_metrics_are_episode_event_based_and_margin_bound() -> None:
    rows = [
        {
            "episode": 0,
            "lane_command": -1,
            "pre_action_route_intent": -1,
            "pre_action_route_intent_valid": True,
            "route_intent_match": True,
            "lane_change_applied": True,
            "target_critic_decoder": {
                "selected_lane": -1,
                "minimum_target_twin_q": [2.0, 1.0, 0.0],
            },
        },
        {
            "episode": 0,
            "lane_command": 0,
            "pre_action_route_intent": -1,
            "pre_action_route_intent_valid": True,
            "route_intent_match": False,
            "lane_change_applied": False,
            "target_critic_decoder": {
                "selected_lane": 0,
                "minimum_target_twin_q": [0.0, 1.0, 0.0],
            },
        },
        {
            "episode": 1,
            "lane_command": 0,
            "pre_action_route_intent": 0,
            "pre_action_route_intent_valid": True,
            "route_intent_match": True,
            "lane_change_applied": False,
            "target_critic_decoder": {
                "selected_lane": 0,
                "minimum_target_twin_q": [0.0, 1.0, 0.0],
            },
        },
    ]
    metrics = _target_event_metrics(rows)
    assert metrics["exact_target_critic_argmax_rate"] == 1.0
    assert metrics["target_route_event_episode_count"] == 1
    assert metrics["target_route_event_applied_match_episode_rate"] == 1.0
    assert (
        metrics["applied_match_positive_target_q_margin_episode_rate"] == 1.0
    )
    assert metrics["target_route_minus_keep_q_margin"]["count"] == 2


def test_v4_3_save_load_and_parent_checkpoint_policy_migration(tmp_path: Any) -> None:
    env, model, observation, _ = _model_and_tensor_observation([1.0, 1.0, 1.0])
    try:
        with source_evaluation_augmentation(model):
            expected, _ = model.predict(observation, deterministic=True)
        checkpoint = tmp_path / "v4_3"
        model.save(checkpoint)
        restored = TargetCriticLaneDecoderSACV43.load(
            checkpoint, env=env, device="cpu"
        )
        assert isinstance(restored.policy, TargetCriticDecisionAlignedSACPolicyV43)
        with source_evaluation_augmentation(restored):
            actual, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)

        parent = make_model_v4_2(
            "topo_v4_2_factorized_entropy",
            env,
            scenario="carla",
            learning_starts=0,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        parent_checkpoint = tmp_path / "v4_2_parent"
        parent.save(parent_checkpoint)
        migrated = TargetCriticLaneDecoderSACV43.load(
            parent_checkpoint,
            env=env,
            device="cpu",
            custom_objects={
                "policy_class": TargetCriticDecisionAlignedSACPolicyV43,
            },
        )
        assert isinstance(migrated, TargetCriticLaneDecoderSACV43)
        assert isinstance(migrated.policy, TargetCriticDecisionAlignedSACPolicyV43)
        migrated_action, _ = migrated.predict(observation, deterministic=True)
        assert np.isfinite(migrated_action).all()
        assert float(migrated_action[1]) in (-1.0, 0.0, 1.0)
    finally:
        env.close()


def test_v4_3_formal_unlock_is_hash_bound(tmp_path: Any) -> None:
    arguments = argparse.Namespace(
        evaluation_split="test",
        algo="topo_v4_3_target_critic_decoder",
        experiment_contract_sha256="a" * 64,
        implementation_freeze_sha256="b" * 64,
        formal_unlock_receipt=None,
    )
    try:
        _validate_formal_unlock(arguments)
    except ValueError as exc:
        assert "missing promotion gate receipt" in str(exc)
    else:
        raise AssertionError("formal test must reject a missing promotion receipt")

    receipt = tmp_path / "promotion_gate_v4_3.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.3.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "a" * 64,
                "implementation_freeze_sha256": "b" * 64,
                "promotion_results_sha256": "c" * 64,
                "formal_algorithms": [
                    "temporal_graph_v1_control",
                    "topo_v4_3_target_critic_decoder",
                ],
            }
        ),
        encoding="utf-8",
    )
    arguments.formal_unlock_receipt = receipt
    validated = _validate_formal_unlock(arguments)
    assert validated is not None
    assert validated["path"] == str(receipt.resolve())
