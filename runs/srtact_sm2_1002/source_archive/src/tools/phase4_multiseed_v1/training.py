"""Aligned 50k-step training for the three methods across training seeds 0/1/2.

The environment, optimizer settings, budget and checkpoint cadence are the ones
the frozen phase-2 runner used.  Only the training seed varies, which is exactly
what the multi-seed confirmation needs.  A completed cell is never retrained:
its ``training_complete.json`` hash is re-checked and returned, and an
interrupted directory raises instead of being silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from algos.sb3_torch import RawStepControlCallback
from .common import (OUT, ROOT, PROTOCOL, model_state_sha, read, relative, seal, sha,
                     validate_frozen_sources, write)
from .config import (CHECKPOINT_FREQUENCY, METHODS, REUSED_SEED_ZERO_METHODS, SMOKE_BUDGET,
                     SMOKE_CHECKPOINT_FREQUENCY, SMOKE_WARMUP, TRAINING_BUDGET, TRAINING_WARMUP)
from .model import assert_candidate, make_model
from .traffic import requested_arguments


def check_preflight():
    data = read(OUT / 'preflight.json')
    if not data['passed'] or data['manifest_sha256'] != sha(OUT / 'manifest.json'):
        raise ValueError('Missing/stale phase-4 preflight')
    for path, expected in data['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Phase-4 implementation changed after preflight: {path}')
    validate_frozen_sources()


def training_environment(job, source, namespace, seed):
    """Frozen environment factory plus the seed-specific environment arguments."""
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    requested = dict(requested_arguments(source))
    requested.update(seed=seed, gui=False, evaluation_split='validation')
    env_args = argparse.Namespace(**requested)
    density = read(PROTOCOL)['scenarios'][source['scenario']]
    adapter = 'base' if job['method'] == 'mst_slt' else job['method']
    factory = _make_environment_factory(adapter=adapter, density=density,
                                        overlay_root=OUT / 'ov' / namespace)
    return factory, env_args


def _restore(job, source, checkpoint, evaluation_env, device):
    """Deployment-side reload, one adapter per method (identical to phase 2)."""
    if job['method'] == 'mst_slt':
        from algos.sb3_torch.sac import SceneRepresentationSAC
        return SceneRepresentationSAC.load(checkpoint, env=evaluation_env, device=device)
    if job['method'] == 'v4_8':
        from algos.sb3_torch.sac_v4_5 import ConfidentActorFusionSACV45
        from tools.action_diagnostics_v4_6 import load_model_for_deployment
        return load_model_for_deployment(ConfidentActorFusionSACV45, checkpoint,
                                        decoder=source['decoder'], env=evaluation_env, device=device)
    from algos.sb3_torch.sac_v4_13_model import GradientIsolatedTemperedJointSupportSACV413
    from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13
    return load_model_for_deployment_v4_13(GradientIsolatedTemperedJointSupportSACV413, checkpoint,
                                          decoder=source['decoder'], env=evaluation_env, device=device)


def train(job_id, device='cuda', smoke=False):
    from tools.phase2_model_factory_v1 import verify_optimizer_settings
    torch.set_num_threads(1)
    manifest = read(OUT / 'manifest.json')
    job = next((j for j in manifest['training_jobs'] if j['id'] == job_id), None)
    if job is None:
        raise ValueError(f'Not a scheduled phase-4 training cell: {job_id}')
    if job['seed'] == 0 and job['method'] in REUSED_SEED_ZERO_METHODS:
        raise ValueError('Seed-0 cells with sealed evidence are reused, never retrained')
    if not smoke:
        check_preflight()
    source = manifest['sources'][job_id]
    assert_candidate(job['method'], job['candidate'])
    settings = METHODS[job['method']]
    output = OUT / ('smoke_train' if smoke else 'train') / job_id
    if (output / 'training_complete.json').exists():
        complete = read(output / 'training_complete.json')
        if sha(output / 'final_model.zip') != complete['checkpoint_sha256']:
            raise ValueError('Completed model hash drift')
        return relative(output / 'training_complete.json')
    if output.exists():
        raise ValueError(f'Incomplete existing training requires recovery, never silent overwrite: {output}')
    output.mkdir(parents=True, exist_ok=False)

    budget, warmup = (SMOKE_BUDGET, SMOKE_WARMUP) if smoke else (TRAINING_BUDGET, TRAINING_WARMUP)
    frequency = SMOKE_CHECKPOINT_FREQUENCY if smoke else CHECKPOINT_FREQUENCY
    namespace = ('sm_' if smoke else 'tr_') + hashlib.sha256(job_id.encode()).hexdigest()[:12]
    factory, env_args = training_environment(job, source, namespace, job['seed'])
    started = time.time()
    env = evaluation_env = model = None
    checks = []
    write(output / 'status.json', {'status': 'starting', 'pid': os.getpid(), 'smoke': smoke})

    class Audit(BaseCallback):
        def _on_step(self):
            verify_optimizer_settings(self.model, learning_rate=settings['learning_rate'],
                                      tau=settings['tau'])
            if self.n_calls % 100 == 0:
                write(output / 'progress.json', {'raw_steps': self.model._raw_steps_seen,
                                                 'learner_updates': self.model._n_updates,
                                                 'updated_at': time.time()})
            return True

        def _on_training_end(self):
            checks.append(verify_optimizer_settings(self.model, learning_rate=settings['learning_rate'],
                                                    tau=settings['tau']))

    try:
        env = Monitor(factory(env_args), filename=str(output / 'train_monitor.csv'),
                      info_keywords=('raw_simulation_steps', 'is_success', 'collision',
                                     'off_route', 'max_time'))
        model = make_model(env, job['method'], job['candidate'], job['scenario'], job['seed'],
                           device, smoke)
        seal(output / 'arguments.json', {
            **job, 'raw_budget': budget, 'warmup': warmup, 'smoke': smoke,
            'learning_rate': settings['learning_rate'], 'tau': settings['tau'],
            'decoder': source['decoder'], 'manifest_sha256': sha(OUT / 'manifest.json'),
            'environment_protocol_sha256': sha(PROTOCOL),
            'model_factory_sha256': sha(ROOT / 'tools/phase2_model_factory_v1.py'),
            'trainer_sha256': sha(Path(__file__)),
            'inherited_environment_arguments': vars(env_args),
            'initial_parameter_sha256': model_state_sha(model),
        })
        model.learn(total_timesteps=budget, callback=CallbackList([
            RawStepControlCallback(raw_step_budget=budget, checkpoint_frequency=frequency,
                                   checkpoint_path=output / 'checkpoints', checkpoint_prefix='ckpt'),
            Audit(),
        ]))
        if model._raw_steps_seen != budget or model._n_updates != budget - warmup + 1:
            raise ValueError('Raw budget/update accounting failed')
        verify_optimizer_settings(model, learning_rate=settings['learning_rate'], tau=settings['tau'])
        final = output / 'final_model.zip'
        model.save(final)
        evaluation_env = factory(env_args, evaluation=True)
        restored = _restore(job, source, final, evaluation_env, device)
        if restored._n_updates != model._n_updates or restored._raw_steps_seen != model._raw_steps_seen:
            raise ValueError('Restored model accounting differs')
        for key, tensor in model.policy.state_dict().items():
            if not torch.equal(tensor.cpu(), restored.policy.state_dict()[key].cpu()):
                raise ValueError(f'Saved model tensor roundtrip differs: {key}')
        after = verify_optimizer_settings(restored, learning_rate=settings['learning_rate'],
                                          tau=settings['tau'])
        from tools.paper_evaluation_contract import validate_model_environment_spaces
        validate_model_environment_spaces(restored, evaluation_env)
        write(output / 'training_diagnostics.json', model.training_diagnostics())
        completion = {'checkpoint_sha256': sha(final), 'raw_steps': budget,
                      'learner_updates': model._n_updates, 'method': job['method'],
                      'candidate': job['candidate'], 'training_seed': job['seed'], 'smoke': smoke,
                      'optimizer_settings': after, 'optimizer_checks': len(checks),
                      'policy_tensor_roundtrip_equal': True, 'wall_seconds': time.time() - started,
                      'checkpoints': {relative(p): sha(p) for p in sorted((output / 'checkpoints').glob('*.zip'))}}
        seal(output / 'training_complete.json', completion)
        write(output / 'status.json', {'status': 'completed', **completion})
        return relative(output / 'training_complete.json')
    except BaseException as exc:
        write(output / 'status.json', {'status': 'failed', 'error': repr(exc), 'smoke': smoke,
                                      'raw_steps': getattr(model, '_raw_steps_seen', 0),
                                      'wall_seconds': time.time() - started})
        raise
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        if env is not None:
            env.close()
