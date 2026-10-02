from __future__ import annotations

import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algos.sb3_torch.hybrid_policy_v4_10_model import (
    SharedRiskSupportedMixtureSACPolicyV410,
    SupportedMixtureHybridActor,
)
from algos.sb3_torch.callbacks import RawStepControlCallback
from algos.sb3_torch.evaluation import source_evaluation_augmentation
from algos.sb3_torch.replay_buffer_v4_9 import (
    CollisionAwareHorizonReplayBufferV49,
)
from algos.sb3_torch.sac_v4_10_model import (
    SharedRiskSupportedMixtureSACV410,
)
from configs.sb3_configs_v4_10 import (
    V410_FULL,
    V410_MIXTURE_NO_SUPPORT,
    V410_SHARED_SINGLE,
    make_model_v4_10,
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


def _policy(speed_components: int = 3):
    observation_space, action_space = _spaces()
    return SharedRiskSupportedMixtureSACPolicyV410(
        observation_space,
        action_space,
        lambda _: 1e-3,
        net_arch={"pi": [16, 8], "qf": [16, 8]},
        activation_fn=torch.nn.ReLU,
        features_extractor_class=_TrainableExtractor,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        n_critics=2,
        action_embedding_dim=4,
        speed_components=speed_components,
        twin_uncertainty_coef=0.25,
        component_prior_coef=0.05,
    )


def _observation(batch: int = 2):
    mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])[:batch]
    return {
        "state": torch.zeros((batch, 4)),
        "lane_action_mask": mask,
    }


def test_actor_enumerates_learned_components_and_exact_masses() -> None:
    policy = _policy()
    actor = policy.actor
    assert isinstance(actor, SupportedMixtureHybridActor)
    batch = actor.all_action_proposals(_observation(), deterministic_speed=True)
    assert batch.actions.shape == (2, 3, 3, 2)
    assert batch.component_probabilities.shape == (2, 3, 3)
    torch.testing.assert_close(
        batch.component_probabilities.sum(dim=2), torch.ones((2, 3))
    )
    torch.testing.assert_close(
        batch.lane_probabilities.sum(dim=1), torch.ones(2)
    )
    assert batch.lane_probabilities[1, 0].item() == pytest.approx(0.0)
    assert batch.lane_probabilities[1, 2].item() == pytest.approx(0.0)
    assert bool((batch.actions[..., 0].abs() < 1.0).all())
    expected_lanes = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1)
    torch.testing.assert_close(
        batch.actions[..., 1], expected_lanes.expand(2, 3, 3)
    )


def test_replay_support_likelihood_updates_actor_heads_not_encoder() -> None:
    policy = _policy()
    actor = policy.actor
    assert isinstance(actor, SupportedMixtureHybridActor)
    batch = actor.all_action_proposals(_observation(), deterministic_speed=False)
    replay = torch.tensor([[0.2, -1.0], [-0.4, 0.0]])
    loss = actor.replay_support_nll(batch, replay)
    assert torch.isfinite(loss)
    loss.backward()
    assert actor.speed_mean.weight.grad is not None
    assert actor.component_logits.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in actor.features_extractor.parameters()
    )


def test_shared_risk_encoder_and_optimizer_ownership_are_exact() -> None:
    policy = _policy()
    ownership = policy.optimizer_parameter_ownership()
    assert ownership["online_encoder_shared_by_identity"] is True
    assert ownership["target_encoder_shared_by_identity"] is True
    assert ownership["overlap_count"] == 0
    encoder_ids = {id(parameter) for parameter in policy.critic.features_extractor.parameters()}
    collision_ids = {id(parameter) for parameter in policy.collision_critic.parameters()}
    assert encoder_ids <= collision_ids
    collision_optimizer_ids = {
        id(parameter)
        for group in policy.collision_critic.optimizer.param_groups
        for parameter in group["params"]
    }
    assert encoder_ids.isdisjoint(collision_optimizer_ids)


def test_deterministic_decoder_selects_exact_learned_score_argmax() -> None:
    policy = _policy()
    observation = _observation(batch=1)
    policy.begin_target_decoder_recording()
    action = policy._predict(observation, deterministic=True)
    record = policy.end_target_decoder_recording()[0]
    scores = record["supported_risk_adjusted_score"]
    candidates = []
    for lane_index, lane in enumerate(scores):
        if lane is None:
            continue
        for component_index, score in enumerate(lane):
            candidates.append((float(score), lane_index, component_index))
    maximum = max(item[0] for item in candidates)
    selected_score = float(
        scores[record["selected_lane_index"]][record["selected_component_index"]]
    )
    assert selected_score == pytest.approx(maximum)
    expected_speed = record["learned_speed_proposals_normalized"][
        record["selected_lane_index"]
    ][record["selected_component_index"]]
    assert action[0, 0].item() == pytest.approx(expected_speed)
    assert action[0, 1].item() == pytest.approx(record["selected_lane"])
    assert record["candidate_source"] == "learned_actor_component_means"
    assert record["fixed_speed_grid_used"] is False
    assert record["action_rewritten"] is False


class _CollisionInfoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space, self.action_space = _spaces()
        self.steps = 0

    def _obs(self):
        return {
            "state": np.full(4, self.steps / 10.0, dtype=np.float32),
            "lane_action_mask": np.ones(3, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        del action
        self.steps += 1
        terminated = self.steps >= 5
        return self._obs(), 0.0, terminated, False, {
            "collision": False,
            "raw_simulation_steps": 1,
            "undiscounted_reward": 0.0,
        }


def _small_model(env: gym.Env) -> SharedRiskSupportedMixtureSACV410:
    return SharedRiskSupportedMixtureSACV410(
        SharedRiskSupportedMixtureSACPolicyV410,
        env,
        learning_rate=1e-3,
        buffer_size=64,
        learning_starts=2,
        batch_size=2,
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
        seed=7,
        device="cpu",
        verbose=0,
    )


def test_gradient_update_has_disjoint_owners_and_round_trips(tmp_path) -> None:
    env = _CollisionInfoEnv()
    model = _small_model(env)
    model.learn(total_timesteps=16)
    assert model._n_updates > 0
    assert model.optimizer_ownership_audit()["overlap_count"] == 0
    diagnostics = model.training_diagnostics()
    for name in (
        "train/collision_critic_loss",
        "support/replay_speed_mixture_nll",
        "support/component_speed_spread",
        "risk/learned_uncertainty",
    ):
        assert name in diagnostics
        assert np.isfinite(diagnostics[name]["last"])
    observation, _ = env.reset(seed=3)
    expected, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / "v410"
    model.save(checkpoint)
    restored = SharedRiskSupportedMixtureSACV410.load(
        checkpoint.with_suffix(".zip"), env=env, device="cpu"
    )
    actual, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
    assert restored.optimizer_ownership_audit()["overlap_count"] == 0


def test_model_source_contains_no_kinematic_rule_or_action_override() -> None:
    source = inspect.getsource(SharedRiskSupportedMixtureSACV410)
    policy_source = inspect.getsource(SharedRiskSupportedMixtureSACPolicyV410)
    assert "replay_support_nll" in source
    assert "learned_uncertainty" in source
    for forbidden in (
        "time_to_collision",
        "headway",
        "kinematic_projection",
        "lane_change_veto",
        "unsafe_target",
        "postprocess_action",
    ):
        assert forbidden not in source.lower()
        assert forbidden not in policy_source.lower()


@pytest.mark.parametrize(
    ("algorithm", "components", "support", "uncertainty"),
    [
        (V410_FULL, 3, 0.05, 0.25),
        (V410_SHARED_SINGLE, 1, 0.05, 0.25),
        (V410_MIXTURE_NO_SUPPORT, 3, 0.0, 0.0),
    ],
)
def test_real_topology_registry_builds_preregistered_variants(
    algorithm: str,
    components: int,
    support: float,
    uncertainty: float,
) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_10(
            algorithm,
            env,
            scenario="left_turn",
            learning_starts=48,
            buffer_size=64,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        assert isinstance(model, SharedRiskSupportedMixtureSACV410)
        assert isinstance(model.actor, SupportedMixtureHybridActor)
        assert model.actor.speed_components == components
        assert model.replay_support_coef == support
        assert model.twin_uncertainty_coef == uncertainty
        audit = model.optimizer_ownership_audit()
        assert audit["overlap_count"] == 0
        assert audit["policy_structure"][
            "online_encoder_shared_by_identity"
        ] is True
        assert audit["policy_structure"][
            "target_encoder_shared_by_identity"
        ] is True
        assert model.v4_method_metadata["inference_safety_rule_added"] is False
        assert model.v4_method_metadata["fixed_speed_grid_at_inference"] is False
    finally:
        env.close()


def test_real_sumo_gradient_serialization_and_finite_diagnostics(tmp_path) -> None:
    env = PaperSumoSceneEnvV4(
        scenario="left_turn",
        action_repeat=3,
        reward_discount=0.99,
        traffic_partition="validation",
        episode_limit_profile="source",
    )
    try:
        model = make_model_v4_10(
            V410_FULL,
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
            "train/critic_loss",
            "train/collision_critic_loss",
            "train/representation_loss",
            "support/replay_speed_mixture_nll",
            "support/component_speed_spread",
            "risk/learned_uncertainty",
        ):
            assert name in diagnostics
            assert np.isfinite(diagnostics[name]["last"])
        observation, _ = env.reset(seed=4)
        model.policy.begin_target_decoder_recording()
        with source_evaluation_augmentation(model):
            expected, _ = model.predict(observation, deterministic=True)
        records = model.policy.end_target_decoder_recording()
        assert len(records) == 1
        assert records[0]["fixed_speed_grid_used"] is False
        checkpoint = tmp_path / "v410_real"
        model.save(checkpoint)
        restored = SharedRiskSupportedMixtureSACV410.load(
            checkpoint.with_suffix(".zip"), env=env, device="cpu"
        )
        with source_evaluation_augmentation(restored):
            actual, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
        assert restored.optimizer_ownership_audit()["overlap_count"] == 0
    finally:
        env.close()
