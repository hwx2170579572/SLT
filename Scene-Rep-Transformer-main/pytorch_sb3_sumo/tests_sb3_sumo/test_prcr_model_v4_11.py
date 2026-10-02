from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algos.sb3_torch.hybrid_policy_v4_11_model import (
    ProperCalibratedRankedRiskSACPolicyV411,
    SupportedMixtureHybridActor,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_11_model import (
    ProperCalibratedRankedRiskSACV411,
)
from configs.sb3_configs_v4_11 import (
    V411_FULL,
    V411_RANK_NO_CONSISTENCY,
    V411_SOFT_BCE_ONLY,
    make_model_v4_11,
)
from envs.sumo.paper_env_v4 import PaperSumoSceneEnvV4


def _spaces():
    observation = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                -10.0, 10.0, shape=(4,), dtype=np.float32
            ),
            "lane_action_mask": gym.spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float32
            ),
        }
    )
    action = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return observation, action


class _TrainableExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=8)
        self.projection = torch.nn.Linear(7, 8)

    def forward(self, observations):
        values = torch.cat(
            [observations["state"], observations["lane_action_mask"]], dim=1
        )
        return torch.relu(self.projection(values))


class _AlternatingCollisionEnv(gym.Env):
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


def _small_model(env: gym.Env) -> ProperCalibratedRankedRiskSACV411:
    return ProperCalibratedRankedRiskSACV411(
        ProperCalibratedRankedRiskSACPolicyV411,
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
            "features_extractor_class": _TrainableExtractor,
            "normalize_images": False,
            "optimizer_class": torch.optim.Adam,
            "n_critics": 2,
            "action_embedding_dim": 4,
            "speed_components": 3,
            "twin_uncertainty_coef": 0.25,
            "component_prior_coef": 0.05,
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


def test_small_model_trains_risk_losses_and_round_trips(tmp_path) -> None:
    env = _AlternatingCollisionEnv()
    model = _small_model(env)
    model.learn(total_timesteps=40)
    assert model._n_updates > 0
    assert model.optimizer_ownership_audit()["overlap_count"] == 0
    diagnostics = model.training_diagnostics()
    for name in (
        "train/collision_critic_loss",
        "train/collision_soft_bce_loss",
        "train/collision_pairwise_rank_loss",
        "train/collision_temporal_consistency_loss",
        "risk/collision_pairwise_active_pair_rate",
        "risk/collision_probability_absolute_error",
    ):
        assert name in diagnostics
        assert np.isfinite(diagnostics[name]["last"])
    assert diagnostics["risk/collision_pairwise_active_pair_rate"]["maximum"] > 0.0
    observation, _ = env.reset(seed=3)
    expected, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / "v411"
    model.save(checkpoint)
    restored = ProperCalibratedRankedRiskSACV411.load(
        checkpoint.with_suffix(".zip"), env=env, device="cpu"
    )
    actual, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
    assert restored.collision_pairwise_rank_coef == pytest.approx(0.25)
    assert restored.collision_temporal_consistency_coef == pytest.approx(0.05)


def test_deterministic_action_is_exact_learned_score_argmax() -> None:
    env = _AlternatingCollisionEnv()
    model = _small_model(env)
    observation, _ = env.reset(seed=4)
    model.policy.begin_target_decoder_recording()
    action, _ = model.predict(observation, deterministic=True)
    record = model.policy.end_target_decoder_recording()[0]
    scores = record["supported_risk_adjusted_score"]
    maximum = max(
        float(score)
        for lane in scores
        if lane is not None
        for score in lane
    )
    selected_score = scores[record["selected_lane_index"]][
        record["selected_component_index"]
    ]
    selected_speed = record["learned_speed_proposals_normalized"][
        record["selected_lane_index"]
    ][record["selected_component_index"]]
    assert selected_score == pytest.approx(maximum)
    assert action[0] == pytest.approx(selected_speed)
    assert action[1] == pytest.approx(record["selected_lane"])
    assert record["fixed_speed_grid_used"] is False
    assert record["action_rewritten"] is False


@pytest.mark.parametrize(
    ("algorithm", "rank", "consistency"),
    [
        (V411_FULL, 0.25, 0.05),
        (V411_SOFT_BCE_ONLY, 0.0, 0.0),
        (V411_RANK_NO_CONSISTENCY, 0.25, 0.0),
    ],
)
def test_registry_builds_frozen_model_variants(
    algorithm: str, rank: float, consistency: float
) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_11(
            algorithm,
            env,
            scenario="left_turn",
            learning_starts=48,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model, ProperCalibratedRankedRiskSACV411)
        assert isinstance(model.actor, SupportedMixtureHybridActor)
        assert model.actor.speed_components == 3
        assert model.collision_pairwise_rank_coef == pytest.approx(rank)
        assert model.collision_temporal_consistency_coef == pytest.approx(
            consistency
        )
        assert model.optimizer_ownership_audit()["overlap_count"] == 0
        metadata = model.v4_method_metadata
        assert metadata["collision_return_loss"] == (
            "soft_target_binary_cross_entropy_with_logits"
        )
        assert metadata["collision_pairwise_hard_threshold"] is False
        assert metadata["collision_pairwise_fixed_margin"] is False
        assert metadata["inference_safety_rule_added"] is False
        assert metadata["external_kinematic_projection"] is False
        assert metadata["kinematic_safety_projection"] is False
        assert metadata["lane_change_veto"] is False
        assert metadata["traffic_risk_in_lane_mask"] is False
        assert metadata["action_postprocessing_override"] is False
    finally:
        env.close()
