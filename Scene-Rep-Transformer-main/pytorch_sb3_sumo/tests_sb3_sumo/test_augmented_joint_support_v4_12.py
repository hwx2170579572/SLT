from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algos.sb3_torch.hybrid_policy_v4_12_model import (
    AugmentedJointSupportPRCRPolicyV412,
    JointReplaySupportedMixtureActorV412,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_12_model import (
    AugmentedJointSupportPRCRSACV412,
)
from configs.sb3_configs_v4_12 import (
    V412_AUGMENTATION_ONLY,
    V412_FULL,
    V412_NO_CROSS_AUGMENTATION,
    make_model_v4_12,
)
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4


def _spaces():
    return (
        gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    -10.0, 10.0, shape=(4,), dtype=np.float32
                ),
                "lane_action_mask": gym.spaces.Box(
                    0.0, 1.0, shape=(3,), dtype=np.float32
                ),
            }
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )


class _Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=8)
        self.projection = torch.nn.Linear(7, 8)

    def forward(self, observations):
        values = torch.cat(
            [observations["state"], observations["lane_action_mask"]], dim=1
        )
        return torch.relu(self.projection(values))


class _Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space, self.action_space = _spaces()
        self.steps = 0
        self.episode = -1

    def _obs(self):
        return {
            "state": np.asarray(
                [self.steps / 5.0, self.episode % 2, 0.0, 1.0],
                dtype=np.float32,
            ),
            "lane_action_mask": np.ones(3, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode += 1
        return self._obs(), {}

    def step(self, action):
        del action
        self.steps += 1
        terminated = self.steps >= 5
        collision = bool(terminated and self.episode % 2 == 0)
        return self._obs(), 0.0, terminated, False, {
            "collision": collision,
            "raw_simulation_steps": 1,
            "undiscounted_reward": 0.0,
        }


def _small_model(env: gym.Env) -> AugmentedJointSupportPRCRSACV412:
    return AugmentedJointSupportPRCRSACV412(
        AugmentedJointSupportPRCRPolicyV412,
        env,
        learning_rate=1e-3,
        buffer_size=128,
        learning_starts=2,
        batch_size=8,
        tau=0.01,
        gamma=0.95,
        train_freq=(1, "step"),
        gradient_steps=1,
        n_steps=2,
        replay_buffer_class=CollisionAwareHorizonReplayBufferV49,
        replay_buffer_kwargs={
            "n_steps": 2,
            "gamma": 0.95,
            "duplicate_episode_end_transition": True,
            "source_action_repeat": 1,
        },
        ent_coef="auto_0.2",
        target_entropy=-1.0,
        policy_kwargs={
            "net_arch": {"pi": [16, 8], "qf": [16, 8]},
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": _Extractor,
            "normalize_images": False,
            "optimizer_class": torch.optim.Adam,
            "n_critics": 2,
            "action_embedding_dim": 4,
            "speed_components": 3,
            "twin_uncertainty_coef": 0.25,
            "component_prior_coef": 0.05,
            "lane_prior_coef": 0.05,
            "lane_support_scale": 1.0,
        },
        representation_coef=0.0,
        structured_representation=False,
        action_embedding_dim=4,
        raw_learning_starts=0,
        lane_entropy_scale=0.0,
        collision_risk_coef=1.0,
        replay_support_coef=0.05,
        component_entropy_scale=0.25,
        twin_uncertainty_coef=0.25,
        collision_pairwise_rank_coef=0.25,
        collision_temporal_consistency_coef=0.05,
        seed=7,
        device="cpu",
        verbose=0,
    )


def test_joint_support_adds_masked_lane_nll_and_gradients() -> None:
    env = _Env()
    model = _small_model(env)
    actor = model.actor
    assert isinstance(actor, JointReplaySupportedMixtureActorV412)
    observations = {
        "state": torch.zeros((3, 4)),
        "lane_action_mask": torch.ones((3, 3)),
    }
    batch = actor.all_action_proposals(observations, deterministic_speed=True)
    replay_actions = torch.tensor([[0.0, -1.0], [0.2, 0.0], [-0.2, 1.0]])
    terms = actor.joint_replay_support_terms(batch, replay_actions)
    assert torch.isfinite(terms.joint_action_nll)
    assert terms.joint_action_nll.detach().item() == pytest.approx(
        (
            terms.conditional_speed_mixture_nll
            + terms.lane_categorical_nll
        ).item()
    )
    actor.optimizer.zero_grad()
    terms.joint_action_nll.backward()
    assert actor.lane_logits.weight.grad is not None
    assert bool(torch.isfinite(actor.lane_logits.weight.grad).all())


def test_deterministic_score_contains_continuous_lane_log_prior() -> None:
    env = _Env()
    model = _small_model(env)
    observation, _ = env.reset(seed=3)
    model.policy.begin_target_decoder_recording()
    action, _ = model.predict(observation, deterministic=True)
    record = model.policy.end_target_decoder_recording()[0]
    lane = int(record["selected_lane_index"])
    component = int(record["selected_component_index"])
    probability = float(record["component_probabilities"][lane][component])
    lane_probability = float(record["lane_probabilities"][lane])
    reward = float(record["minimum_target_twin_q"][lane][component])
    collision = float(
        record["maximum_target_twin_collision_value"][lane][component]
    )
    uncertainty = float(record["learned_uncertainty"][lane][component])
    expected = (
        reward
        - float(record["collision_risk_coef"]) * collision
        - float(record["twin_uncertainty_coef"]) * uncertainty
        + float(record["component_prior_coef"]) * math.log(probability)
        + float(record["lane_prior_coef"]) * math.log(lane_probability)
    )
    score = float(record["supported_risk_adjusted_score"][lane][component])
    assert score == pytest.approx(expected, abs=1e-5)
    assert record["joint_lane_component_support_score"] is True
    assert record["actor_confidence_threshold_present"] is False
    assert record["action_rewritten"] is False
    assert action[1] == pytest.approx(record["selected_lane"])


def test_small_model_trains_and_records_all_support_terms() -> None:
    env = _Env()
    model = _small_model(env)
    model.learn(total_timesteps=40)
    diagnostics = model.training_diagnostics()
    for name in (
        "support/replay_joint_action_nll",
        "support/replay_joint_action_nll_recomputed",
        "support/replay_lane_categorical_nll",
        "support/replay_conditional_speed_mixture_nll",
    ):
        assert name in diagnostics
        assert np.isfinite(diagnostics[name]["last"])
    assert model.optimizer_ownership_audit()["overlap_count"] == 0


def test_checkpoint_round_trip_restores_isolated_v4_12_components(
    tmp_path: Path,
) -> None:
    env = _Env()
    model = _small_model(env)
    model.learn(total_timesteps=8)
    checkpoint = tmp_path / "v4_12_round_trip.zip"
    model.save(checkpoint)

    restored = AugmentedJointSupportPRCRSACV412.load(
        checkpoint,
        env=env,
        device="cpu",
        custom_objects={"policy_class": AugmentedJointSupportPRCRPolicyV412},
    )

    assert isinstance(restored.policy, AugmentedJointSupportPRCRPolicyV412)
    assert isinstance(restored.actor, JointReplaySupportedMixtureActorV412)
    assert restored.optimizer_ownership_audit()["overlap_count"] == 0


@pytest.mark.parametrize(
    ("algorithm", "cross_augmentation", "lane_scale", "lane_prior"),
    [
        (V412_FULL, True, 1.0, 0.05),
        (V412_NO_CROSS_AUGMENTATION, False, 1.0, 0.05),
        (V412_AUGMENTATION_ONLY, True, 0.0, 0.0),
    ],
)
def test_registry_builds_preregistered_v4_12_variants(
    algorithm: str,
    cross_augmentation: bool,
    lane_scale: float,
    lane_prior: float,
) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="cross",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_12(
            algorithm,
            env,
            scenario="cross",
            learning_starts=48,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model, AugmentedJointSupportPRCRSACV412)
        assert isinstance(model.actor, JointReplaySupportedMixtureActorV412)
        assert model.actor.lane_support_scale == pytest.approx(lane_scale)
        assert model.policy.lane_prior_coef == pytest.approx(lane_prior)
        assert (
            model.critic.features_extractor.random_augmentation
            is cross_augmentation
        )
        metadata = model.v4_method_metadata
        assert metadata["kinematic_safety_projection"] is False
        assert metadata["traffic_risk_in_lane_mask"] is False
        assert metadata["actor_confidence_gate"] is False
        assert metadata["action_postprocessing_override"] is False
    finally:
        env.close()


def test_policy_declares_model_only_boundary() -> None:
    assert AugmentedJointSupportPRCRPolicyV412.inference_safety_rule_added is False
    assert AugmentedJointSupportPRCRPolicyV412.external_kinematic_projection is False
    assert AugmentedJointSupportPRCRPolicyV412.action_postprocessing_override is False
