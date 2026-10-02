from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from algos.sb3_torch.callbacks import RawStepControlCallback
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.sac_v4_2 import FactorizedEntropyHybridSACV42
from configs.sb3_configs_v4_2 import (
    V42_IMPLEMENTATION_IDS,
    make_model_v4_2,
)
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.action_diagnostics_v4_2 import _route_margins, _summary
from tools.train_sb3_v4_2 import _method_hyperparameters, _validate_formal_unlock


def _carla_env() -> PaperSumoSceneEnvV4:
    return PaperSumoSceneEnvV4(
        scenario="carla",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )


def test_v4_2_registry_freezes_only_factorized_entropy() -> None:
    env = _carla_env()
    try:
        model = make_model_v4_2(
            "topo_v4_2_factorized_entropy",
            env,
            scenario="carla",
            learning_starts=0,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model, FactorizedEntropyHybridSACV42)
        assert model.lane_entropy_scale == 0.0
        assert model.target_entropy == -1.0
        metadata = model.v4_method_metadata
        assert metadata["implementation_id"] == V42_IMPLEMENTATION_IDS[
            "topo_v4_2_factorized_entropy"
        ]
        assert metadata["single_change"] == "factorized_hybrid_entropy"
        assert metadata["lane_entropy_scale"] == 0.0
        assert metadata["speed_target_entropy"] == -1.0
        assert metadata["reward_unchanged"] is True
        assert metadata["actor_encoder_detach"] is True
        assert metadata["route_intent_observation"] is False
        hyperparameters = _method_hyperparameters("topo_v4_2_factorized_entropy")
        assert hyperparameters["lane_entropy_scale"] == 0.0
        assert hyperparameters["speed_target_entropy"] == -1.0
    finally:
        env.close()


def test_v4_2_entropy_objective_excludes_lane_log_probability() -> None:
    env = _carla_env()
    try:
        observation, _ = env.reset(seed=30000)
        model = make_model_v4_2(
            "topo_v4_2_factorized_entropy",
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
        with torch.no_grad():
            batch = model.actor.all_action_samples(
                tensor_observation, deterministic_speed=False
            )
            actual = model.entropy_log_probabilities(batch)
            expected = batch.speed_log_probabilities
            expected_average = (
                batch.lane_probabilities * batch.speed_log_probabilities
            ).sum(dim=1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            model.expected_entropy_log_probability(batch), expected_average
        )
        assert not torch.allclose(actual, batch.joint_log_probabilities)
    finally:
        env.close()


def test_v4_2_single_update_diagnostics_and_save_load(tmp_path: Any) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_2(
            "topo_v4_2_factorized_entropy",
            env,
            scenario="left_turn",
            learning_starts=16,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        model.learn(
            total_timesteps=16,
            callback=RawStepControlCallback(raw_step_budget=16),
        )
        assert model._n_updates == 1
        diagnostics = model.training_diagnostics()
        for name in (
            "hybrid/lane_entropy",
            "hybrid/expected_speed_log_probability",
            "hybrid/effective_lane_entropy_coefficient",
        ):
            assert name in diagnostics
            assert np.isfinite(diagnostics[name]["last"])
        assert diagnostics["hybrid/effective_lane_entropy_coefficient"]["last"] == 0.0

        observation, _ = env.reset(seed=4)
        with source_evaluation_augmentation(model):
            expected_action, _ = model.predict(observation, deterministic=True)
        checkpoint = tmp_path / "topo_v4_2_factorized_entropy_smoke"
        model.save(checkpoint)
        restored = FactorizedEntropyHybridSACV42.load(
            checkpoint, env=env, device="cpu"
        )
        with source_evaluation_augmentation(restored):
            actual_action, _ = restored.predict(observation, deterministic=True)
        assert restored.lane_entropy_scale == 0.0
        assert restored.target_entropy == -1.0
        np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
    finally:
        env.close()


def test_route_margin_summary_handles_empty_and_signed_values() -> None:
    empty = _summary([])
    assert empty["count"] == 0
    assert empty["mean"] is None
    values = _summary([-0.1, 0.2, 0.5])
    assert values["count"] == 3
    assert values["minimum"] == -0.1
    assert values["maximum"] == 0.5
    assert abs(float(values["mean"]) - 0.2) < 1e-12


def test_route_margin_trace_allows_nonhybrid_baseline_but_not_partial_candidate() -> None:
    baseline = [
        {
            "pre_action_route_intent": -1,
            "pre_action_route_intent_valid": True,
            "hybrid_policy": None,
        }
    ]
    assert _route_margins(baseline) == []
    candidate = [
        {
            "pre_action_route_intent": -1,
            "pre_action_route_intent_valid": True,
            "hybrid_policy": {"lane_probabilities": [0.7, 0.3, 0.0]},
        },
        {
            "pre_action_route_intent": -1,
            "pre_action_route_intent_valid": True,
            "hybrid_policy": None,
        },
    ]
    try:
        _route_margins(candidate)
    except ValueError as exc:
        assert "missing hybrid-policy" in str(exc)
    else:
        raise AssertionError("partial candidate traces must fail")


def test_v4_2_formal_unlock_is_hash_bound(tmp_path: Any) -> None:
    arguments = argparse.Namespace(
        evaluation_split="test",
        algo="topo_v4_2_factorized_entropy",
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

    receipt = tmp_path / "promotion_gate_v4_2.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "topo-scene-v4.2.promotion-gate/v1",
                "decision": "pass",
                "experiment_contract_sha256": "a" * 64,
                "implementation_freeze_sha256": "b" * 64,
                "promotion_results_sha256": "c" * 64,
                "formal_algorithms": [
                    "temporal_graph_v1_control",
                    "topo_v4_2_factorized_entropy",
                ],
            }
        ),
        encoding="utf-8",
    )
    arguments.formal_unlock_receipt = receipt
    validated = _validate_formal_unlock(arguments)
    assert validated is not None
    assert validated["path"] == str(receipt.resolve())
