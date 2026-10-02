"""Optimization-only adapter. No network, loss, reward or decoder changes."""
from __future__ import annotations

import json
import math
import types
from pathlib import Path

import torch

from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
from tools.phase2_model_factory_v1 import make_phase2_model, optimizer_rates, verify_optimizer_settings
from .common import CANDIDATES, model_state_sha


def learning_rate(config, raw_step):
    weight = min(1., max(0., (raw_step - config['decay_start']) /
                            (config['decay_end'] - config['decay_start'])))
    return config['learning_rate'] + weight * (config['final_learning_rate'] - config['learning_rate'])


class StabilitySAC(ConfidentActorFusionSACV45):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['stability_trace_path']

    def _update_learning_rate(self, optimizers):
        config = getattr(self, 'stability_config', None)
        if config is None:
            return super()._update_learning_rate(optimizers)
        raw_step = self.raw_learning_starts + self._n_updates
        rate = learning_rate(config, raw_step)
        self.logger.record('train/learning_rate', rate)
        # The original representation optimizer has an explicit rate and is
        # therefore excluded from SB3's schedule. Update it explicitly as well.
        extra = [self.representation_optimizer, getattr(self.policy, 'encoder_optimizer', None)]
        for optimizer in [*optimizers, *extra]:
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group['lr'] = rate
        self.representation_learning_rate = rate

    def train(self, gradient_steps, batch_size=64):
        config = getattr(self, 'stability_config', None)
        if config is None:
            return super().train(gradient_steps, batch_size)
        if self._pending_raw_gradient_steps is not None:
            gradient_steps = self._pending_raw_gradient_steps
            self._pending_raw_gradient_steps = None
        if gradient_steps <= 0:
            return
        # All models use target_update_interval=1. Splitting scheduled calls
        # preserves replay contents and update order, and gives an exact rate
        # for each raw-step update. Constant candidates keep the original calls.
        scheduled = config['learning_rate'] != config['final_learning_rate']
        if scheduled and self.target_update_interval != 1:
            raise ValueError('Scheduled adapter requires target update interval 1')
        for count in ([1] * gradient_steps if scheduled else [gradient_steps]):
            super().train(count, batch_size)
            values = {k: float(v) for k, v in self.logger.name_to_value.items()
                      if k.startswith(('train/', 'hybrid/')) and isinstance(v, (float, int))}
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError('Non-finite training diagnostic')
            # No extra random draws. Every train call is checked; a compact
            # trajectory is persisted about every 100 updates, plus the first.
            previous = getattr(self, 'stability_last_trace', -100)
            if self._n_updates - previous >= 100 or self._n_updates == 1:
                self.stability_last_trace = self._n_updates
                self.audit_parameters()
                path = getattr(self, 'stability_trace_path', None)
                if path is not None:
                    row = dict(raw_step=self.raw_learning_starts + self._n_updates - 1,
                               updates=self._n_updates, values=values,
                               optimizer_rates=optimizer_rates(self), replay_size=self.replay_buffer.size(),
                               replay_insertions=self.replay_buffer._next_insertion_id)
                    with Path(path).open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row, allow_nan=False) + '\n')

    def audit_parameters(self):
        for name, tensor in self.policy.state_dict().items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Non-finite model tensor: {name}')
        if self.log_ent_coef is not None and not torch.isfinite(self.log_ent_coef).all():
            raise FloatingPointError('Non-finite entropy coefficient')


def make_model(env, scene, candidate, *, device='cuda', smoke=False):
    config = dict(CANDIDATES[candidate])
    if smoke:
        config.update(decay_start=60, decay_end=96, buffer_size=128)
    model = make_phase2_model('v4_8', env, learning_rate=config['learning_rate'], tau=config['tau'],
                             scenario=scene, batch_size=config['batch_size'], learning_starts=48 if smoke else 5000,
                             buffer_size=config['buffer_size'], action_repeat=3,
                             seed=930100 if smoke else 0, device=device, verbose=0)
    before = model_state_sha(model)
    model.__class__ = StabilitySAC
    model.stability_config = config
    if model_state_sha(model) != before:
        raise AssertionError('Attaching diagnostics changed learned parameters')
    verify_optimizer_settings(model, learning_rate=config['learning_rate'], tau=config['tau'])
    return model


def actor_prediction(policy, observation, deterministic=False):
    """Bypass the saved policy's target-critic/fusion deployment decoder."""
    return policy.actor(observation, deterministic=deterministic)


def use_actor_only(model):
    before = model_state_sha(model)
    model.policy._predict = types.MethodType(actor_prediction, model.policy)
    if model_state_sha(model) != before:
        raise ValueError('Actor-only deployment changed learned tensors')
    return model


def load_model(path, scene, env, device):
    del scene
    # Load the stored exact-final policy without replacing its policy class or
    # invoking any checkpoint/decoder selector. Only prediction dispatch changes.
    return use_actor_only(StabilitySAC.load(path, env=env, device=device))
