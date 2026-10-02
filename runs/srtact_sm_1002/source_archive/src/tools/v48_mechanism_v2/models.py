"""Load frozen final models on CPU; actor dispatch never uses a decoder."""
from __future__ import annotations

import hashlib
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, checkpoint_path, describe_file


def model_path(scene, name):
    if name in ("mst_slt", "v4_8_lr_half"):
        return checkpoint_path(scene, name)
    if name not in ("late_decay", "batch64"):
        raise ValueError(name)
    path = ROOT / "r48s1/train" / f"{scene}__{name}" / "final_model.zip"
    receipt = json.loads(path.with_name("training_complete.json").read_text(encoding="utf-8-sig"))
    if receipt["smoke"] or describe_file(path)["sha256"] != receipt["checkpoint_sha256"]:
        raise ValueError("Stable-model completion receipt mismatch")
    return path


def tensor_digest(model):
    result = hashlib.sha256()
    for prefix, module in (("policy", model.policy), ("representation", model.representation)):
        if module is not None:
            for name, value in sorted(module.state_dict().items()):
                result.update((prefix + "." + name).encode())
                result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def actor_prediction(policy, observation, deterministic=False):
    return policy.actor(observation, deterministic=deterministic)


def load_model(scene, name):
    from algos.sb3_torch.sac import SceneRepresentationSAC
    from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
    from tools.v48_stability_v1.model import StabilitySAC
    cls = SceneRepresentationSAC if name == "mst_slt" else (
        ConfidentActorFusionSACV45 if name == "v4_8_lr_half" else StabilitySAC)
    # Replay is unused for inference. Avoid allocating the saved 20k capacity.
    model = cls.load(model_path(scene, name), env=None, device="cpu", buffer_size=32)
    before = tensor_digest(model)
    model.policy._predict = types.MethodType(actor_prediction, model.policy)
    model.policy.set_training_mode(False)
    if tensor_digest(model) != before:
        raise AssertionError("Actor dispatch changed a learned tensor")
    return model


def filter_observation(model, observation):
    return {k: observation[k] for k in model.observation_space.spaces}


def observation_tensors(model, observation):
    return model.policy.obs_to_tensor(filter_observation(model, observation))[0]


def checkpoint_audit(scene, name, real_observations):
    from algos.sb3_torch.evaluation import source_evaluation_augmentation
    path = model_path(scene, name)
    model = load_model(scene, name)
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read("data"))
    before = tensor_digest(model)
    keys = ("gamma", "tau", "batch_size", "learning_rate", "representation_learning_rate",
            "representation_coef", "slot_balance_coef", "structured_representation",
            "representation_online_target_encoder", "target_update_interval", "target_entropy",
            "buffer_size", "learning_starts", "_n_updates", "_raw_steps_seen", "num_timesteps",
            "replay_buffer_kwargs", "max_grad_norm", "seed")
    config = {key: saved.get(key) for key in keys if key in saved}
    online = model.critic.features_extractor
    target = model.critic_target.features_extractor
    feature_parameters = list(online.parameters())
    feature_ids = {id(p) for p in feature_parameters}
    optimizers = {"actor": model.actor.optimizer, "critic": model.critic.optimizer,
                  "representation": model.representation_optimizer}
    membership = {name_: len(feature_ids & {id(p) for g in opt.param_groups for p in g["params"]})
                  for name_, opt in optimizers.items() if opt is not None}
    obs = {key: np.array(value[:4], copy=True) for key, value in real_observations.items()}
    synthetic_mask = "lane_action_mask" in model.observation_space.spaces and "lane_action_mask" not in obs
    if synthetic_mask:
        obs["lane_action_mask"] = np.ones((4, 3), dtype=np.float32)
    tensor_obs = observation_tensors(model, obs)
    with source_evaluation_augmentation(model):
        with torch.no_grad():
            features = online(tensor_obs)
        assert features.shape == (4, 128) and torch.isfinite(features).all()
        outputs = model.actor(tensor_obs, deterministic=True)
        actor_grads = torch.autograd.grad(outputs.sum(), feature_parameters, allow_unused=True)
        assert all(g is None or not bool(g.abs().any()) for g in actor_grads)
        original_forward = model.critic.forward
        original_target_forward = model.critic_target.forward
        def forbidden(*args, **kwargs):
            raise AssertionError("Direct actor prediction called a critic")
        try:
            model.critic.forward = forbidden
            model.critic_target.forward = forbidden
            prediction, _ = model.predict(obs, deterministic=True)
            expected = model.policy.unscale_action(outputs.detach().numpy()) if model.policy.squash_output else np.clip(
                outputs.detach().numpy(), model.action_space.low, model.action_space.high)
            np.testing.assert_allclose(prediction, expected, atol=1e-6, rtol=1e-6)
        finally:
            model.critic.forward = original_forward
            model.critic_target.forward = original_target_forward
        online_leaf = features.detach().clone().requires_grad_(True)
        target_leaf = features.detach().clone().requires_grad_(True)
        actions = torch.zeros((len(features), *model.action_space.shape))
        losses = model.representation(online_leaf, actions, target_leaf)
        loss = losses.total if hasattr(losses, "total") else losses
        g_online, g_target = torch.autograd.grad(loss, (online_leaf, target_leaf), allow_unused=True)
        assert g_target is None and g_online is not None and torch.isfinite(g_online).all()
    assert tensor_digest(model) == before
    assert next(model.policy.parameters()).device.type == "cpu"
    assert model.actor.features_extractor is online and online is not target
    assert membership["actor"] == 0 and membership["critic"] > 0 and membership["representation"] > 0
    return dict(scene=scene, model=name, checkpoint=describe_file(path), saved_config=config,
                policy_class=type(model.policy).__name__, encoder_class=type(online).__name__,
                auxiliary_class=type(model.representation).__name__,
                encoder_parameters=sum(p.numel() for p in feature_parameters),
                auxiliary_parameters=sum(p.numel() for p in model.representation.parameters()),
                policy_unique_parameters=sum(p.numel() for p in model.policy.parameters()),
                observation_shapes={k: list(s.shape) for k, s in model.observation_space.spaces.items()},
                action_low=model.action_space.low.tolist(), action_high=model.action_space.high.tolist(),
                extractor_optimizer_parameter_tensor_counts=membership,
                actor_gradient_to_encoder_zero=True, target_auxiliary_gradient_stopped=True,
                actor_predict_without_critic_verified=True, tensors_unchanged=True, tensor_sha256=before,
                shared_actor_critic_encoder=True, separate_target_encoder=True,
                synthetic_all_legal_mask_for_structural_smoke_only=synthetic_mask,
                inference_only_replay_capacity_override=32, feature_shape=list(features.shape), status="pass")
