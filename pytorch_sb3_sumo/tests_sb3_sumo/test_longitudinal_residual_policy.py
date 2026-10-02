from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from algos.sb3_torch.policy_shadow_probes import capture_policy_snapshot

from algos.sb3_torch.longitudinal_residual_policy import (
    BoundedLongitudinalResidualActor,
    LongitudinalResidualContinuousCritic,
    LongitudinalResidualSceneRepSACPolicy,
    summarize_conflict_timing,
)
from algos.sb3_torch.policies import ActionEmbeddedContinuousCritic, DetachedSceneActor, SceneRepSACPolicy


class _TinyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=4):
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        return observations["base"].float().reshape(-1, self.features_dim)


class _ActorOnlyPolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor


class _ActorOnlyModel:
    def __init__(self, actor):
        self.policy = _ActorOnlyPolicy(actor)

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        mean, _, _ = self.policy.actor.get_action_dist_params(observation)
        return torch.tanh(mean).detach().cpu().numpy(), None


def _spaces():
    observation_space = gym.spaces.Dict(
        {
            "base": gym.spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32),
            "conflict_timing": gym.spaces.Box(-2.0, 2.0, shape=(4, 20), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return observation_space, action_space


def _obs(batch=2, *, valid=True):
    conflict = torch.zeros(batch, 4, 20, dtype=torch.float32)
    if valid:
        conflict[:, 1, 0] = 1.0
        conflict[:, 1, 1:7] = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.5, 0.9])
        conflict[:, 1, 7:15] = torch.tensor([0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 0.2, 0.3])
        conflict[:, 1, 15:] = torch.tensor([0.8, 0.6, 0.0, 0.0, 0.5])
    return {
        "base": torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=torch.float32).repeat(batch, 1),
        "conflict_timing": conflict,
    }


def _actor_kwargs(observation_space, action_space):
    return dict(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[8],
        features_extractor=_TinyExtractor(observation_space, 4),
        features_dim=4,
        activation_fn=torch.nn.ReLU,
        use_sde=False,
        log_std_init=-3.0,
        use_expln=False,
        clip_mean=2.0,
    )


def test_summary_uses_only_finite_valid_social_rows_and_unknown_is_unsupported():
    observation = _obs(valid=True)
    summary, supported, count = summarize_conflict_timing(observation)
    assert summary.shape == (2, 41)
    assert supported.tolist() == [True, True]
    assert count.tolist() == [1, 1]
    assert torch.all(summary[:, -1] > 0)

    invalid = _obs(valid=False)
    summary, supported, count = summarize_conflict_timing(invalid)
    assert supported.tolist() == [False, False]
    assert count.tolist() == [0, 0]
    assert torch.equal(summary, torch.zeros_like(summary))

    nonfinite = _obs(valid=True)
    nonfinite["conflict_timing"][:, 1, 3] = float("nan")
    _, supported, count = summarize_conflict_timing(nonfinite)
    assert supported.tolist() == [False, False]
    assert count.tolist() == [0, 0]


def test_actor_zero_init_and_longres_off_change_only_speed_mean():
    observation_space, action_space = _spaces()
    torch.manual_seed(17)
    baseline = DetachedSceneActor(**_actor_kwargs(observation_space, action_space))
    actor = BoundedLongitudinalResidualActor(
        **_actor_kwargs(observation_space, action_space),
        longitudinal_residual_bound=0.2,
    )
    actor.load_state_dict(baseline.state_dict(), strict=False)
    observation = _obs(valid=True)

    base_mean, base_log_std, _ = baseline.get_action_dist_params(observation)
    zero_mean, zero_log_std, _ = actor.get_action_dist_params(observation)
    assert torch.equal(zero_mean, base_mean)
    assert torch.equal(zero_log_std, base_log_std)

    with torch.no_grad():
        actor.longitudinal_residual_head.bias.fill_(0.1)
    active_mean, active_log_std, _ = actor.get_action_dist_params(observation)
    assert torch.all(active_mean[:, 0] > base_mean[:, 0])
    assert torch.equal(active_mean[:, 1], base_mean[:, 1])
    assert torch.equal(active_log_std, base_log_std)
    assert torch.all((active_mean[:, 0] - base_mean[:, 0]).abs() <= 0.2)

    with actor.longitudinal_residual_probe("longitudinal_residual_off"):
        off_mean, off_log_std, _ = actor.get_action_dist_params(observation)
    assert torch.equal(off_mean, base_mean)
    assert torch.equal(off_log_std, base_log_std)
    assert actor._longitudinal_residual_force_off is False
    assert actor.longitudinal_diagnostic_values()["longres_actor_active"] == 1.0

    unsupported = _obs(valid=False)
    unsupported_mean, _, _ = actor.get_action_dist_params(unsupported)
    assert torch.equal(unsupported_mean, base_mean)
    metrics = actor.longitudinal_diagnostic_values()
    assert metrics["longres_actor_context_valid_fraction"] == 0.0
    assert metrics["longres_actor_invalid_reason"] == "no_valid_conflict_relations"


def test_shadow_snapshot_uses_final_distribution_mean_and_restores_longres_cache():
    observation_space, action_space = _spaces()
    actor = BoundedLongitudinalResidualActor(
        **_actor_kwargs(observation_space, action_space),
        longitudinal_residual_bound=0.2,
    )
    with torch.no_grad():
        actor.longitudinal_residual_head.weight.fill_(0.0)
        actor.longitudinal_residual_head.bias.fill_(0.15)
    observation = _obs(valid=True)
    model = _ActorOnlyModel(actor)

    with torch.no_grad():
        expected_active = actor.get_action_dist_params(observation)[0].detach().clone()
        latent = actor.latent_pi(actor.extract_features(observation, actor.features_extractor).detach())
        expected_mu_layer = actor.mu(latent).detach().clone()
    active_cache = actor.policy_shadow_mean_snapshot().clone()
    # Leave deliberately stale actor-side diagnostic state from a different
    # unsupported batch; the snapshot must report the actual probed state while
    # restoring this pre-probe cache afterwards.
    with torch.no_grad():
        actor.get_action_dist_params(_obs(valid=False))
    stale_cache = actor.policy_shadow_mean_snapshot().clone()
    assert actor.longitudinal_diagnostic_values()["longres_actor_context_valid_fraction"] == 0.0
    active_snapshot = capture_policy_snapshot(model, observation)
    assert active_snapshot["mean_pre_tanh"] == pytest.approx(
        expected_active[0].tolist(), abs=1e-7
    )
    assert active_snapshot["mu_layer_output_pre_tanh"] == pytest.approx(
        expected_mu_layer[0].tolist(), abs=1e-7
    )
    assert active_snapshot["action_normalized_mean"] == pytest.approx(
        torch.tanh(expected_active)[0].tolist(), abs=1e-7
    )
    assert active_snapshot["actor_diagnostic_values"][
        "longres_actor_context_valid_fraction"
    ] == 1.0
    torch.testing.assert_close(actor.policy_shadow_mean_snapshot(), stale_cache)
    assert actor.longitudinal_diagnostic_values()["longres_actor_context_valid_fraction"] == 0.0

    with actor.longitudinal_residual_probe("longitudinal_residual_off"):
        off_snapshot = capture_policy_snapshot(model, observation)
        assert actor._longitudinal_residual_force_off is True
    assert actor._longitudinal_residual_force_off is False
    torch.testing.assert_close(actor.policy_shadow_mean_snapshot(), stale_cache)
    torch.testing.assert_close(active_cache, expected_active)
    assert actor.longitudinal_diagnostic_values()["longres_actor_context_valid_fraction"] == 0.0
    assert off_snapshot["actor_diagnostic_values"]["longres_actor_active"] == 0.0
    assert off_snapshot["actor_diagnostic_values"][
        "longres_actor_context_valid_fraction"
    ] == 1.0
    assert off_snapshot["mean_pre_tanh"] == pytest.approx(
        expected_mu_layer[0].tolist(), abs=1e-7
    )
    deltas = [
        off - active
        for active, off in zip(
            active_snapshot["mean_pre_tanh"], off_snapshot["mean_pre_tanh"]
        )
    ]
    assert deltas[1] == pytest.approx(0.0, abs=1e-7)
    assert abs(deltas[0]) > 0.0


def test_shadow_snapshot_without_provider_keeps_legacy_mu_path():
    observation_space, action_space = _spaces()
    actor = DetachedSceneActor(**_actor_kwargs(observation_space, action_space))
    observation = _obs(valid=True)
    model = _ActorOnlyModel(actor)
    snapshot = capture_policy_snapshot(model, observation)
    assert "mu_layer_output_pre_tanh" not in snapshot
    assert snapshot["valid"] is True
    assert snapshot["mean_pre_tanh"] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_shadow_snapshot_uses_cuda_final_mean_and_json_safe_values():
    observation_space, action_space = _spaces()
    actor = BoundedLongitudinalResidualActor(
        **_actor_kwargs(observation_space, action_space),
        longitudinal_residual_bound=0.2,
    ).cuda()
    observation = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in _obs(valid=True).items()
    }
    model = _ActorOnlyModel(actor)
    with torch.no_grad():
        actor.longitudinal_residual_head.weight.zero_()
        actor.longitudinal_residual_head.bias.fill_(0.15)
        expected_mean = actor.get_action_dist_params(observation)[0].detach().cpu().numpy()[0]
    expected_cache = actor.policy_shadow_mean_snapshot().detach().clone()

    snapshot = capture_policy_snapshot(model, observation)

    assert snapshot["valid"] is True
    assert snapshot["mean_source"] == "actor_get_action_dist_params_return"
    assert np.allclose(snapshot["mean_pre_tanh"], expected_mean, atol=1e-7)
    assert np.allclose(snapshot["model_action"], np.tanh(expected_mean), atol=1e-7)
    assert np.allclose(snapshot["action_normalized_mean"], np.tanh(expected_mean), atol=1e-7)
    assert snapshot["actor_diagnostic_values"]["longres_actor_context_valid_fraction"] == 1.0
    torch.testing.assert_close(actor.policy_shadow_mean_snapshot(), expected_cache)


def test_critic_zero_init_preserves_base_q_and_private_branch_has_grad_update():
    observation_space, action_space = _spaces()
    extractor_base = _TinyExtractor(observation_space, 4)
    base = ActionEmbeddedContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[8],
        features_extractor=extractor_base,
        features_dim=4,
        activation_fn=torch.nn.ReLU,
        normalize_images=False,
        n_critics=2,
        share_features_extractor=False,
        action_embedding_dim=4,
    )
    extended = LongitudinalResidualContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[8],
        features_extractor=_TinyExtractor(observation_space, 4),
        features_dim=4,
        activation_fn=torch.nn.ReLU,
        normalize_images=False,
        n_critics=2,
        share_features_extractor=False,
        action_embedding_dim=4,
    )
    extended.load_state_dict(base.state_dict(), strict=False)
    observation = _obs(valid=True)
    actions = torch.tensor([[0.1, -0.2], [-0.3, 0.4]], dtype=torch.float32)
    base_values = base(observation, actions)
    extended_values = extended(observation, actions)
    for expected, actual in zip(base_values, extended_values):
        assert torch.equal(actual, expected)

    optimizer = torch.optim.Adam(extended.parameters(), lr=1e-2)
    optimizer.zero_grad()
    sum(value.sum() for value in extended(observation, actions)).backward()
    final_bias = extended.conflict_q_residuals[0][-1].bias
    assert final_bias.grad is not None and final_bias.grad.abs().sum() > 0
    before = final_bias.detach().clone()
    optimizer.step()
    assert not torch.equal(final_bias.detach(), before)
    metrics = extended.longitudinal_diagnostic_values()
    assert metrics["longres_critic_context_valid_fraction"] == 1.0

    # The private Q residual must remain differentiable with respect to the
    # proposed action, not only with respect to its own parameters.
    with torch.no_grad():
        extended.action_encoders[0][0].weight.fill_(0.1)
        extended.action_encoders[0][0].bias.fill_(0.5)
        extended.conflict_context_encoders[0][0].weight.fill_(0.1)
        extended.conflict_context_encoders[0][0].bias.fill_(0.5)
        extended.conflict_q_residuals[0][0].weight.fill_(0.1)
        extended.conflict_q_residuals[0][0].bias.fill_(0.5)
        extended.conflict_q_residuals[0][-1].weight.fill_(0.1)
    proposed = actions.detach().clone().requires_grad_(True)
    summary, supported, _ = summarize_conflict_timing(observation)
    context_features = extended.conflict_context_encoders[0](summary)
    action_features = extended.action_encoders[0](proposed)
    residual_q = extended.conflict_q_residuals[0](
        torch.cat([context_features, action_features], dim=1)
    )
    action_gradient = torch.autograd.grad(residual_q.sum(), proposed)[0]
    assert torch.isfinite(action_gradient).all()
    assert action_gradient.abs().sum() > 0


def test_policy_rng_optimizer_ownership_target_copy_and_update_hooks():
    observation_space, action_space = _spaces()
    common = dict(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [8], "qf": [8]},
        features_extractor_class=_TinyExtractor,
        features_extractor_kwargs={"features_dim": 4},
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={},
        action_embedding_dim=4,
    )
    torch.manual_seed(33)
    SceneRepSACPolicy(**common)
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(33)
    policy = LongitudinalResidualSceneRepSACPolicy(
        **common, longitudinal_residual_bound=0.2
    )
    assert torch.equal(torch.get_rng_state(), base_rng)

    metadata = policy.longitudinal_diagnostic_metadata()
    assert metadata["actor_changes_pre_tanh_speed_mean_only"] is True
    assert metadata["actor_log_std_unchanged"] is True
    assert metadata["shared_scene_encoder_receives_conflict_timing"] is False
    assert metadata["optimizer_groups"]["actor"]["longres_actor"]["optimizer_coverage"] == 1.0
    assert metadata["optimizer_groups"]["critic"]["longres_critic"]["optimizer_coverage"] == 1.0
    assert all(
        torch.equal(value, policy.critic_target.state_dict()[name])
        for name, value in policy.critic.state_dict().items()
    )

    # SB3's ordinary critic Polyak update must include the added private Q
    # branch because online and target critics have identical named structure.
    online = policy.critic.conflict_q_residuals[0][-1].bias
    target = policy.critic_target.conflict_q_residuals[0][-1].bias
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(1.0)
    polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau=0.25)
    torch.testing.assert_close(target, target_before + 0.25 * (online - target_before))

    policy.configure_longitudinal_diagnostics(True)
    observation = _obs(valid=True)
    policy.actor.optimizer.zero_grad()
    mean, _, _ = policy.actor.get_action_dist_params(observation)
    mean[:, 0].sum().backward()
    policy.actor.optimizer.step()
    actor_events = policy.pop_longitudinal_diagnostic_events()
    assert len(actor_events) == 1
    assert actor_events[0]["source"] == "longitudinal_residual_actor_optimizer_update"
    assert actor_events[0]["metrics"]["encoder_grad/longres_actor/gradient_nonzero_elements"] > 0
    assert actor_events[0]["metrics"]["encoder_update/longres_actor/changed_elements"] > 0

    policy.critic.optimizer.zero_grad()
    q_values = policy.critic(observation, torch.zeros(2, 2))
    sum(value.sum() for value in q_values).backward()
    policy.critic.optimizer.step()
    critic_events = policy.pop_longitudinal_diagnostic_events()
    assert len(critic_events) == 1
    assert critic_events[0]["source"] == "longitudinal_residual_critic_optimizer_update"
    assert critic_events[0]["metrics"]["encoder_grad/longres_critic/gradient_nonzero_elements"] > 0
    assert critic_events[0]["metrics"]["encoder_update/longres_critic/changed_elements"] > 0


def test_d1_registry_adds_private_ct_observation_without_enabling_shared_ct_encoder():
    project_root = Path(__file__).resolve().parents[1]
    fast_developer = project_root / "fast-developer"
    sys.path.insert(0, str(fast_developer))
    try:
        d1 = importlib.import_module("train_intersection_yield_v2_d1")
        method = "sac_mlp_d1_st_rt_longres_v1"
        assert d1.PARENT[method] == "sac_mlp"
        config = d1.D1_CONFIG[method]
        assert config["use_route"] is True
        assert config["use_route_conflict_timing"] is False
        assert config["use_longitudinal_residual"] is True
        assert config["longitudinal_residual_bound"] == 0.2

        calls = []

        class _StubConflictWrapper:
            def __init__(self, env, **kwargs):
                calls.append(kwargs)
                self.env = env

        fake_module = ModuleType("envs.sumo.conflict_timing_observation")
        fake_module.RouteConflictTimingObservationWrapper = _StubConflictWrapper
        base_env = object()
        with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            result = d1._wrap_method_env(base_env, method, Path("run"), "train")
        assert isinstance(result, _StubConflictWrapper)
        assert calls == [{"diagnostics_directory": Path("run") / "diagnostics" / "train", "phase": "train"}]
    finally:
        try:
            sys.path.remove(str(fast_developer))
        except ValueError:
            pass
        sys.modules.pop("train_intersection_yield_v2_d1", None)


def test_longres_full_sac_checkpoint_roundtrip_restores_class_bound_branches_and_values(
    tmp_path, monkeypatch
):
    # Reuse the repository's no-SUMO dictionary environment; append the CT
    # observation that the real PaperEnv wrapper normally adds at reset.
    from algos.sb3_torch.longitudinal_residual_policy import (
        LongitudinalResidualSceneRepSACPolicy as LongresPolicy,
    )
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from tests_sb3_sumo.test_d1_method_checkpoint_roundtrip import _PolicyOnlyDictEnv

    fast_developer = str(Path(__file__).resolve().parents[1] / "fast-developer")
    monkeypatch.syspath_prepend(fast_developer)
    d1 = importlib.import_module("train_intersection_yield_v2_d1")
    method = "sac_mlp_d1_st_rt_longres_v1"
    monkeypatch.setattr(d1.base, "BUFFER_SIZE", 8)
    monkeypatch.setattr(d1.base, "BATCH_SIZE", 2)
    env = _PolicyOnlyDictEnv()
    env.observation_space.spaces["conflict_timing"] = gym.spaces.Box(
        -2.0, 2.0, shape=(6, 20), dtype=np.float32
    )
    ct = np.zeros((6, 20), dtype=np.float32)
    ct[1, 0] = 1.0
    ct[1, 1:7] = np.asarray([0.2, 0.4, 0.6, 0.8, 0.5, 0.9], dtype=np.float32)
    ct[1, 7:15] = np.asarray([0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 0.2, 0.3], dtype=np.float32)
    ct[1, 15:] = np.asarray([0.8, 0.6, 0.0, 0.0, 0.5], dtype=np.float32)
    env._observation["conflict_timing"] = ct

    model = d1._build_model_d1(method, env, learning_starts=0, device="cpu")
    assert isinstance(model, SceneRepresentationSAC)
    assert model.policy_class is LongresPolicy
    policy = model.policy
    assert isinstance(policy, LongresPolicy)
    assert policy.longitudinal_residual_bound == 0.2
    assert policy.actor.longitudinal_residual_bound == 0.2

    # Make both added branches nonzero so the roundtrip proves their weights,
    # not merely their zero-initialized presence, are restored.
    with torch.no_grad():
        policy.actor.longitudinal_residual_head.weight.fill_(0.025)
        policy.actor.longitudinal_residual_head.bias.fill_(0.12)
        for branch in policy.critic.conflict_q_residuals:
            branch[-1].weight.fill_(0.075)
            branch[-1].bias.fill_(0.04)

    observation, _ = env.reset(seed=2026)
    observation_tensor, _ = policy.obs_to_tensor(observation)
    proposed_action = torch.tensor([[0.15, -0.25]], dtype=torch.float32)
    with torch.no_grad():
        expected_mean = policy.actor.get_action_dist_params(observation_tensor)[0]
        expected_q = policy.critic(observation_tensor, proposed_action)
    checkpoint = tmp_path / "longres_sac_checkpoint.zip"
    model.save(checkpoint)

    restored = SceneRepresentationSAC.load(checkpoint, env=env, device="cpu")
    assert restored.policy_class is LongresPolicy
    assert isinstance(restored.policy, LongresPolicy)
    assert restored.policy.longitudinal_residual_bound == 0.2
    assert torch.equal(
        restored.policy.actor.longitudinal_residual_head.weight,
        policy.actor.longitudinal_residual_head.weight,
    )
    for restored_branch, original_branch in zip(
        restored.policy.critic.conflict_q_residuals,
        policy.critic.conflict_q_residuals,
    ):
        assert torch.equal(restored_branch[-1].weight, original_branch[-1].weight)
        assert torch.equal(restored_branch[-1].bias, original_branch[-1].bias)
    restored_observation, _ = env.reset(seed=2026)
    restored_tensor, _ = restored.policy.obs_to_tensor(restored_observation)
    with torch.no_grad():
        actual_mean = restored.policy.actor.get_action_dist_params(restored_tensor)[0]
        actual_q = restored.policy.critic(restored_tensor, proposed_action)
    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-6, atol=1e-6)
    assert len(actual_q) == len(expected_q)
    for actual, expected in zip(actual_q, expected_q):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    sys.modules.pop("train_intersection_yield_v2_d1", None)
