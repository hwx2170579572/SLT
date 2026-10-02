from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import CombinedExtractor

from algos.sb3_torch.callbacks import RawStepControlCallback
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.hybrid_policy_v4_9_model import (
    CollisionConstrainedFusionSACPolicyV49,
    CollisionConstrainedTargetCriticSACPolicyV49,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_9_model import CollisionConstrainedFusionSACV49
from configs.sb3_configs_v4_9 import V49_CANDIDATE, make_model_v4_9
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4
from tools.action_diagnostics_v4_9_model import load_model_for_deployment_v4_9
from tools import action_diagnostics_v4_9_model as action_diagnostics
from tools.checkpoint_decoder_selector_v4_6 import FUSION_DECODER


def _spaces():
    observation = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                -100.0, 100.0, shape=(4,), dtype=np.float32
            ),
            "lane_action_mask": gym.spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float32
            ),
        }
    )
    action = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return observation, action


def _add(
    buffer: CollisionAwareHorizonReplayBufferV49,
    step: int,
    reward: float,
    done: bool,
    *,
    collision: bool = False,
) -> None:
    observation = {
        "state": np.full((1, 4), step, dtype=np.float32),
        "lane_action_mask": np.ones((1, 3), dtype=np.float32),
    }
    next_observation = {
        "state": np.full((1, 4), step + 1, dtype=np.float32),
        "lane_action_mask": np.ones((1, 3), dtype=np.float32),
    }
    buffer.add(
        observation,
        next_observation,
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([reward], dtype=np.float32),
        np.asarray([done]),
        [{"TimeLimit.truncated": False, "collision": collision}],
    )


def test_collision_replay_label_uses_discounted_negative_terminal_reward() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    _add(buffer, 0, 0.0, False)
    _add(buffer, 1, 0.0, False)
    _add(buffer, 2, -1.0, True, collision=True)
    sample = buffer._get_samples(np.asarray([0]))
    assert sample.rewards.item() == pytest.approx(-(0.9**2))
    assert sample.collision_costs.item() == pytest.approx(0.9**2)
    assert sample.dones.item() == 1.0
    diagnostics = buffer.collision_label_diagnostics()
    assert diagnostics["source"] == "observed_environment_info_collision"
    assert diagnostics["future_or_oracle_labels_used"] is False
    assert diagnostics["reward_return_changed"] is False


def test_collision_label_is_independent_of_reward_normalization_or_sign() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    _add(buffer, 0, 0.0, False)
    _add(buffer, 1, 0.0, True, collision=True)
    sample = buffer._get_samples(np.asarray([0]))
    assert sample.rewards.item() == 0.0
    assert sample.collision_costs.item() == pytest.approx(0.9)


def test_collision_label_requires_observed_environment_event() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    observation = {
        "state": np.zeros((1, 4), dtype=np.float32),
        "lane_action_mask": np.ones((1, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match=r"info\['collision'\]"):
        buffer.add(
            observation,
            observation,
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            [{"TimeLimit.truncated": False}],
        )


def test_success_return_has_zero_collision_label() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    _add(buffer, 0, 0.0, False)
    _add(buffer, 1, 1.0, True)
    sample = buffer._get_samples(np.asarray([0]))
    assert sample.rewards.item() == pytest.approx(0.9)
    assert sample.collision_costs.item() == 0.0


def test_source_terminal_duplication_keeps_collision_labels_aligned() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
        duplicate_episode_end_transition=True,
    )
    _add(buffer, 0, 0.0, False)
    _add(buffer, 1, -1.0, True, collision=True)

    assert buffer.pos == 3
    assert buffer.collision_events[:3, 0].tolist() == [0.0, 1.0, 1.0]
    assert buffer.stored_transition_count == 3
    assert buffer.stored_collision_event_count == 2
    assert buffer._get_samples(np.asarray([0])).collision_costs.item() == pytest.approx(
        0.9
    )
    assert buffer._get_samples(np.asarray([1])).collision_costs.item() == 1.0
    assert buffer._get_samples(np.asarray([2])).collision_costs.item() == 1.0


def test_timeout_boundary_without_collision_does_not_create_safety_label() -> None:
    observation_space, action_space = _spaces()
    buffer = CollisionAwareHorizonReplayBufferV49(
        16,
        observation_space,
        action_space,
        device="cpu",
        n_steps=4,
        gamma=0.9,
    )
    observation = {
        "state": np.zeros((1, 4), dtype=np.float32),
        "lane_action_mask": np.ones((1, 3), dtype=np.float32),
    }
    next_observation = {
        "state": np.ones((1, 4), dtype=np.float32),
        "lane_action_mask": np.ones((1, 3), dtype=np.float32),
    }
    buffer.add(
        observation,
        next_observation,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=bool),
        [{"TimeLimit.truncated": True, "collision": False}],
    )
    sample = buffer._get_samples(np.asarray([0]))
    assert sample.collision_costs.item() == 0.0
    assert sample.dones.item() == 0.0
    assert buffer.collision_label_diagnostics()[
        "stored_collision_event_count_including_source_duplicates"
    ] == 0


def _policy(policy_class=CollisionConstrainedFusionSACPolicyV49):
    observation_space, action_space = _spaces()
    return policy_class(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [8], "qf": [8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=CombinedExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
    )


def test_policy_builds_independent_online_and_target_collision_critics() -> None:
    policy = _policy()
    assert policy.learned_collision_critic is True
    online_ids = {id(parameter) for parameter in policy.collision_critic.parameters()}
    target_ids = {
        id(parameter) for parameter in policy.collision_critic_target.parameters()
    }
    reward_ids = {id(parameter) for parameter in policy.critic.parameters()}
    assert online_ids.isdisjoint(target_ids)
    assert online_ids.isdisjoint(reward_ids)
    for online, target in zip(
        policy.collision_critic.parameters(),
        policy.collision_critic_target.parameters(),
    ):
        assert torch.equal(online, target)
    state_keys = policy.state_dict()
    assert any(key.startswith("collision_critic.") for key in state_keys)
    assert any(key.startswith("collision_critic_target.") for key in state_keys)


def test_both_frozen_deployment_decoders_retain_collision_model() -> None:
    fusion = _policy(CollisionConstrainedFusionSACPolicyV49)
    target = _policy(CollisionConstrainedTargetCriticSACPolicyV49)
    assert fusion.deterministic_lane_decoder == (
        "actor_confident_non_keep_else_learned_risk_adjusted_critic"
    )
    assert target.deterministic_lane_decoder == (
        "argmax_feasible_min_reward_q_minus_max_collision_value"
    )
    assert hasattr(fusion, "collision_critic")
    assert hasattr(target, "collision_critic")


def test_collision_twin_aggregation_uses_bounded_pessimistic_maximum() -> None:
    heads = (torch.tensor([[0.0], [2.0]]), torch.tensor([[1.0], [-2.0]]))
    values = CollisionConstrainedFusionSACV49._collision_probabilities(heads)
    assert values.shape == (2, 1)
    assert values[:, 0].tolist() == pytest.approx(
        [torch.sigmoid(torch.tensor(1.0)).item(), torch.sigmoid(torch.tensor(2.0)).item()]
    )
    assert bool(((values >= 0.0) & (values <= 1.0)).all())


def test_deterministic_model_score_is_auditable_without_geometry_rule() -> None:
    target = _policy(CollisionConstrainedTargetCriticSACPolicyV49)
    observation = {
        "state": torch.zeros((1, 4)),
        "lane_action_mask": torch.ones((1, 3)),
    }
    target.begin_target_decoder_recording()
    selected = target._predict(observation, deterministic=True)
    record = target.end_target_decoder_recording()[0]
    integrity = action_diagnostics._risk_adjusted_target_integrity(
        [
            {
                "target_critic_decoder": record,
                "lane_command": float(selected[0, 1].detach().cpu()),
            }
        ]
    )
    assert integrity["exact_risk_adjusted_model_argmax_rate"] == 1.0
    assert integrity["risk_adjusted_score_equation_match_rate"] == 1.0
    assert all(
        0.0 <= value <= 1.0
        for value in record["maximum_target_twin_collision_value"]
    )


def test_model_source_contains_no_geometry_or_action_override_rule() -> None:
    source = __import__("inspect").getsource(CollisionConstrainedFusionSACV49)
    assert "collision_costs" in source
    assert "collision_risk_coef" in source
    assert "headway" not in source
    assert "lane_change_veto" not in source
    assert "unsafe_target" not in source


def test_real_model_gradient_update_and_deployment_round_trip(tmp_path) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_9(
            V49_CANDIDATE,
            env,
            scenario="left_turn",
            learning_starts=48,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        model.learn(
            total_timesteps=96,
            callback=RawStepControlCallback(raw_step_budget=96),
        )
        assert model._n_updates > 0
        diagnostics = model.training_diagnostics()
        for name in (
            "train/collision_critic_loss",
            "risk/collision_cost_label",
            "risk/collision_cost_target",
            "risk/policy_expected_collision_cost",
        ):
            assert name in diagnostics
            assert np.isfinite(diagnostics[name]["last"])

        observation, _ = env.reset(seed=4)
        with source_evaluation_augmentation(model):
            expected_action, _ = model.predict(observation, deterministic=True)
        checkpoint = tmp_path / "topo_v4_9_model_smoke"
        model.save(checkpoint)
        restored = load_model_for_deployment_v4_9(
            CollisionConstrainedFusionSACV49,
            checkpoint.with_suffix(".zip"),
            decoder=FUSION_DECODER,
            env=env,
            device="cpu",
        )
        with source_evaluation_augmentation(restored):
            actual_action, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_allclose(expected_action, actual_action, rtol=1e-6, atol=1e-6)
        assert restored.collision_risk_coef == 1.0
    finally:
        env.close()
