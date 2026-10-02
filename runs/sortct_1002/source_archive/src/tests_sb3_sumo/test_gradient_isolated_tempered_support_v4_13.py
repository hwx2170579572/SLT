from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algos.sb3_torch.hybrid_policy_v4_13_model import (
    GradientIsolatedSupportActionBatch,
    GradientIsolatedTemperedJointSupportPolicyV413,
    GradientIsolatedTemperedMixtureActorV413,
)
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_13_model import (
    GradientIsolatedTemperedJointSupportSACV413,
)
from configs.sb3_configs_v4_13 import (
    V413_FULL,
    V413_ISOLATED_UNTEMPERED,
    V413_NO_CROSS_AUGMENTATION,
    V413_UNISOLATED_TEMPERED,
    make_model_v4_13,
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


def _small_model(
    env: gym.Env,
    *,
    lane_support_scale: float = 0.25,
    isolate_lane_support_gradient: bool = True,
) -> GradientIsolatedTemperedJointSupportSACV413:
    return GradientIsolatedTemperedJointSupportSACV413(
        GradientIsolatedTemperedJointSupportPolicyV413,
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
            "lane_support_scale": lane_support_scale,
            "isolate_lane_support_gradient": isolate_lane_support_gradient,
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


def _grad_norm(module: torch.nn.Module) -> float:
    squared = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def _support_terms(model: GradientIsolatedTemperedJointSupportSACV413):
    actor = model.actor
    observations = {
        "state": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [-0.2, 0.1, 0.4, -0.3],
                [0.5, -0.4, 0.3, -0.2],
            ],
            dtype=torch.float32,
        ),
        "lane_action_mask": torch.ones((4, 3)),
    }
    replay_actions = torch.tensor(
        [[0.0, -1.0], [0.2, 0.0], [-0.2, 1.0], [0.4, -1.0]]
    )
    batch = actor.all_action_proposals(observations, deterministic_speed=True)
    return actor, batch, actor.joint_replay_support_terms(batch, replay_actions)


def test_lane_support_gradient_reaches_same_lane_head_only() -> None:
    model = _small_model(_Env())
    actor, batch, terms = _support_terms(model)
    assert isinstance(batch, GradientIsolatedSupportActionBatch)
    assert actor.isolate_lane_support_gradient is True
    assert actor.lane_support_scale == pytest.approx(0.25)
    assert batch.support_lane_log_probabilities.grad_fn is not None

    actor.optimizer.zero_grad()
    terms.lane_categorical_nll.backward()
    assert _grad_norm(actor.lane_logits) > 0.0
    assert _grad_norm(actor.latent_pi) == 0.0
    assert _grad_norm(actor.component_logits) == 0.0
    assert _grad_norm(actor.speed_mean) == 0.0
    assert _grad_norm(actor.speed_log_std) == 0.0
    assert _grad_norm(actor.features_extractor) == 0.0


def test_speed_support_still_trains_latent_and_speed_heads() -> None:
    model = _small_model(_Env())
    actor, _, terms = _support_terms(model)
    actor.optimizer.zero_grad()
    terms.conditional_speed_mixture_nll.backward()
    assert _grad_norm(actor.latent_pi) > 0.0
    assert _grad_norm(actor.component_logits) > 0.0
    assert _grad_norm(actor.speed_mean) > 0.0
    assert _grad_norm(actor.speed_log_std) > 0.0
    assert _grad_norm(actor.lane_logits) == 0.0
    assert _grad_norm(actor.features_extractor) == 0.0


def test_unisolated_ablation_restores_lane_to_latent_gradient() -> None:
    model = _small_model(_Env(), isolate_lane_support_gradient=False)
    actor, _, terms = _support_terms(model)
    actor.optimizer.zero_grad()
    terms.lane_categorical_nll.backward()
    assert _grad_norm(actor.lane_logits) > 0.0
    assert _grad_norm(actor.latent_pi) > 0.0
    assert _grad_norm(actor.component_logits) == 0.0


def test_joint_support_equation_uses_preregistered_tempering() -> None:
    model = _small_model(_Env())
    _, _, terms = _support_terms(model)
    expected = (
        terms.conditional_speed_mixture_nll
        + 0.25 * terms.lane_categorical_nll
    )
    assert terms.joint_action_nll.detach().item() == pytest.approx(
        expected.detach().item()
    )


def test_deterministic_inference_equation_is_unchanged_and_rule_free() -> None:
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
    assert float(record["supported_risk_adjusted_score"][lane][component]) == (
        pytest.approx(expected, abs=1e-5)
    )
    assert record["lane_support_gradient_isolated"] is True
    assert record["lane_support_gradient_target"] == "deployed_lane_head_only"
    assert record["actor_confidence_threshold_present"] is False
    assert record["action_rewritten"] is False
    assert action[1] == pytest.approx(record["selected_lane"])


def test_small_model_trains_and_checkpoint_round_trip(tmp_path: Path) -> None:
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

    checkpoint = tmp_path / "v4_13_round_trip.zip"
    model.save(checkpoint)
    restored = GradientIsolatedTemperedJointSupportSACV413.load(
        checkpoint,
        env=env,
        device="cpu",
        custom_objects={
            "policy_class": GradientIsolatedTemperedJointSupportPolicyV413
        },
    )
    assert isinstance(
        restored.policy, GradientIsolatedTemperedJointSupportPolicyV413
    )
    assert isinstance(
        restored.actor, GradientIsolatedTemperedMixtureActorV413
    )
    assert restored.actor.isolate_lane_support_gradient is True
    assert restored.actor.lane_support_scale == pytest.approx(0.25)
    assert restored.optimizer_ownership_audit()["overlap_count"] == 0


@pytest.mark.parametrize(
    ("algorithm", "isolated", "scale", "cross_augmentation"),
    [
        (V413_FULL, True, 0.25, True),
        (V413_UNISOLATED_TEMPERED, False, 0.25, True),
        (V413_ISOLATED_UNTEMPERED, True, 1.0, True),
        (V413_NO_CROSS_AUGMENTATION, True, 0.25, False),
    ],
)
def test_registry_builds_preregistered_real_sumo_variants(
    algorithm: str,
    isolated: bool,
    scale: float,
    cross_augmentation: bool,
) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="cross",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_13(
            algorithm,
            env,
            scenario="cross",
            learning_starts=48,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(
            model, GradientIsolatedTemperedJointSupportSACV413
        )
        assert isinstance(
            model.actor, GradientIsolatedTemperedMixtureActorV413
        )
        assert model.actor.isolate_lane_support_gradient is isolated
        assert model.actor.lane_support_scale == pytest.approx(scale)
        assert model.policy.lane_prior_coef == pytest.approx(0.05)
        assert (
            model.critic.features_extractor.random_augmentation
            is cross_augmentation
        )
        metadata = model.v4_method_metadata
        assert metadata["kinematic_safety_projection"] is False
        assert metadata["traffic_risk_in_lane_mask"] is False
        assert metadata["ttc_or_headway_threshold"] is False
        assert metadata["lane_change_veto"] is False
        assert metadata["actor_confidence_gate"] is False
        assert metadata["action_postprocessing_override"] is False
    finally:
        env.close()


def test_policy_declares_model_only_boundary() -> None:
    policy = GradientIsolatedTemperedJointSupportPolicyV413
    assert policy.inference_safety_rule_added is False
    assert policy.external_kinematic_projection is False
    assert policy.action_postprocessing_override is False
