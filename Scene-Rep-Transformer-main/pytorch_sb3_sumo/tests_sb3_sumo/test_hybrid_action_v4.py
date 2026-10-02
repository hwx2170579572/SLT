from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from algos.sb3_torch.callbacks import RawStepControlCallback
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.hybrid_policy_v4 import (
    DecisionAlignedHybridActor,
    HybridActionEmbeddedCritic,
    hybrid_action_features,
)
from algos.sb3_torch.sac_v4 import DecisionAlignedHybridSACV4
from configs.sb3_configs_v4 import V4_IMPLEMENTATION_IDS, make_model_v4
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.train_sb3_v4 import _method_hyperparameters, _validate_formal_unlock


def _carla_env() -> PaperSumoSceneEnvV4:
    return PaperSumoSceneEnvV4(
        scenario="carla",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )


def test_v4_observation_exposes_mask_but_not_route_label() -> None:
    env = _carla_env()
    try:
        observation, _ = env.reset(seed=30000)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(observation["lane_action_mask"], [0.0, 1.0, 0.0])
        assert "route_intent" not in observation
        _, _, _, _, info = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
        assert info["pre_action_route_intent_valid"] is True
        assert info["pre_action_route_intent"] == 0
    finally:
        env.close()


def test_hybrid_action_features_match_environment_thresholds() -> None:
    actions = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, -1.0 / 3.0],
            [0.5, 0.0],
            [1.0, 1.0 / 3.0],
            [0.25, 1.0],
        ],
        dtype=torch.float32,
    )
    encoded = hybrid_action_features(actions)
    torch.testing.assert_close(encoded[:, 0], actions[:, 0])
    torch.testing.assert_close(
        encoded[:, 1:],
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_hybrid_actor_masks_invalid_lanes_and_enumerates_exact_codes() -> None:
    env = _carla_env()
    try:
        observation, _ = env.reset(seed=30000)
        model = make_model_v4(
            "topo_v4_da_hybrid",
            env,
            scenario="carla",
            learning_starts=0,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model.actor, DecisionAlignedHybridActor)
        observation = {key: np.array(value, copy=True) for key, value in observation.items()}
        observation["lane_action_mask"] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        tensor_observation, _ = model.policy.obs_to_tensor(observation)
        with source_evaluation_augmentation(model), torch.no_grad():
            batch = model.actor.all_action_samples(
                tensor_observation, deterministic_speed=True
            )
            deterministic_action, _ = model.predict(observation, deterministic=True)
        assert batch.actions.shape == (1, 3, 2)
        torch.testing.assert_close(
            batch.actions[0, :, 1], torch.tensor([-1.0, 0.0, 1.0])
        )
        assert batch.lane_probabilities[0, 2].item() == 0.0
        assert deterministic_action[1] in (-1.0, 0.0)
        assert np.isfinite(deterministic_action).all()
    finally:
        env.close()


def test_exact_lane_expectation_backpropagates_to_logits() -> None:
    env = _carla_env()
    try:
        observation, _ = env.reset(seed=30000)
        model = make_model_v4(
            "topo_v4_da_hybrid",
            env,
            scenario="carla",
            learning_starts=0,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        observation = {key: np.array(value, copy=True) for key, value in observation.items()}
        observation["lane_action_mask"] = np.ones(3, dtype=np.float32)
        tensor_observation, _ = model.policy.obs_to_tensor(observation)
        batch = model.actor.all_action_samples(
            tensor_observation, deterministic_speed=False
        )
        lane_cost = torch.tensor([[[-1.0], [0.0], [2.0]]])
        loss = (batch.lane_probabilities.unsqueeze(-1) * lane_cost).sum()
        model.actor.optimizer.zero_grad()
        loss.backward()
        gradient = model.actor.lane_logits.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0
    finally:
        env.close()


def test_v4_single_update_diagnostics_save_and_load(tmp_path: Any) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4(
            "topo_v4_da_hybrid",
            env,
            scenario="left_turn",
            learning_starts=16,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model, DecisionAlignedHybridSACV4)
        assert isinstance(model.critic, HybridActionEmbeddedCritic)
        model.learn(
            total_timesteps=16,
            callback=RawStepControlCallback(raw_step_budget=16),
        )
        assert model._n_updates == 1
        diagnostics = model.training_diagnostics()
        for name in (
            "train/soft_slot_balance_loss",
            "hybrid/lane_entropy",
            "hybrid/valid_lane_actions",
            "hybrid/lane_probability_negative",
            "hybrid/lane_probability_keep",
            "hybrid/lane_probability_positive",
        ):
            assert name in diagnostics
            assert np.isfinite(diagnostics[name]["last"])

        observation, _ = env.reset(seed=4)
        with source_evaluation_augmentation(model):
            expected_action, _ = model.predict(observation, deterministic=True)
        assert expected_action[1] in (-1.0, 0.0, 1.0)
        checkpoint = tmp_path / "topo_v4_da_hybrid_smoke"
        model.save(checkpoint)
        restored = DecisionAlignedHybridSACV4.load(checkpoint, env=env, device="cpu")
        with source_evaluation_augmentation(restored):
            actual_action, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
    finally:
        env.close()


def test_continuous_key_ablation_reuses_soft_v2_encoder_only() -> None:
    env = _carla_env()
    try:
        model = make_model_v4(
            "topo_v4_continuous_ablation",
            env,
            scenario="carla",
            learning_starts=0,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        metadata = model.v4_method_metadata
        assert metadata["implementation_id"] == V4_IMPLEMENTATION_IDS[
            "topo_v4_continuous_ablation"
        ]
        assert metadata["single_removed_mechanism"] == "hybrid_action_interface"
        assert metadata["soft_slot_balance_coef"] == 0.01
        assert metadata["lane_action_mask_observation"] is False
        hyperparameters = _method_hyperparameters("topo_v4_continuous_ablation")
        assert hyperparameters["slot_balance_coef"] == 0.01
        assert hyperparameters["lane_action_mask"] is False
        assert hyperparameters["critic_lane_one_hot"] is False
    finally:
        env.close()


def test_formal_unlock_is_receipt_and_hash_bound(tmp_path: Any) -> None:
    arguments = argparse.Namespace(
        evaluation_split="test",
        algo="topo_v4_da_hybrid",
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

    receipt = tmp_path / "promotion_gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "a" * 64,
                "implementation_freeze_sha256": "b" * 64,
                "promotion_results_sha256": "c" * 64,
                "formal_algorithms": [
                    "temporal_graph_v1_control",
                    "topo_v4_da_hybrid",
                ],
            }
        ),
        encoding="utf-8",
    )
    arguments.formal_unlock_receipt = receipt
    validated = _validate_formal_unlock(arguments)
    assert validated is not None
    assert validated["path"] == str(receipt.resolve())
    assert len(validated["sha256"]) == 64
