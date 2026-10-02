"""Only the twelve missing seed-0 mechanism controls may train here."""
from __future__ import annotations

import os
import time

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from algos.sb3_torch import RawStepControlCallback
from .common import ROOT, OUT, model_state_sha, read, relative, seal, sha, validate_frozen_sources, write
from .model import assert_variant, load_model, make_model
from .traffic import make_env


def check_preflight():
    data = read(OUT / 'preflight.json')
    if not data['passed'] or data['manifest_sha256'] != sha(OUT / 'manifest.json'):
        raise ValueError('Missing/stale phase3 preflight')
    for path, expected in data['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Phase3 implementation changed after preflight: {path}')
    validate_frozen_sources()


def train(job_id, device='cuda', smoke=False):
    torch.set_num_threads(1)
    manifest = read(OUT / 'manifest.json')
    job = next(j for j in manifest['training_jobs'] if j['id'] == job_id)
    if job['variant'] == 'g1_s025' or job['seed'] != 0:
        raise ValueError('Repeated full training or deferred training seed')
    if not smoke:
        check_preflight()
    output = OUT / ('smoke_train' if smoke else 'train') / job_id
    if (output / 'training_complete.json').exists():
        complete = read(output / 'training_complete.json')
        if sha(output / 'final_model.zip') != complete['checkpoint_sha256']:
            raise ValueError('Completed model hash drift')
        return relative(output / 'training_complete.json')
    if output.exists():
        raise ValueError(f'Incomplete existing training requires recovery, never silent overwrite: {output}')
    output.mkdir(parents=True, exist_ok=False)
    parent = manifest['sources'][job['source']]
    budget, warmup = (72, 48) if smoke else (50000, 5000)
    namespace = ('sm_' if smoke else 'tr_') + job_id
    env = Monitor(make_env(parent, 'legacy', namespace, training=True),
                  filename=str(output / 'train_monitor.csv'),
                  info_keywords=('raw_simulation_steps', 'is_success', 'collision', 'off_route', 'max_time'))
    started = time.time()
    model = None
    write(output / 'status.json', {'status': 'starting', 'pid': os.getpid(), 'smoke': smoke})
    try:
        model = make_model(env, job['scenario'], job['variant'], device, smoke)
        from tools.phase2_model_factory_v1 import verify_optimizer_settings
        verify_optimizer_settings(model, learning_rate=1e-4, tau=.0025)
        seal(output / 'arguments.json', {
            **job, 'raw_budget': budget, 'phase3_variant': job['variant'],
            'method': 'v4_13', 'candidate': {'learning_rate': 1e-4, 'tau': .0025},
            'smoke': smoke, 'manifest_sha256': sha(OUT / 'manifest.json'),
            'inherited_environment_arguments': __import__('tools.phase3_mechanism_v1.traffic', fromlist=['requested_arguments']).requested_arguments(parent),
            'intervention': model.phase3_mechanism_metadata,
            'initial_parameter_sha256': model_state_sha(model),
        })

        class Audit(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    assert_variant(self.model, job['variant'])
                    verify_optimizer_settings(self.model, learning_rate=1e-4, tau=.0025)
                    write(output / 'progress.json', {'raw_steps': self.model._raw_steps_seen,
                                                    'updates': self.model._n_updates,
                                                    'updated_at': time.time()})
                return True

        model.learn(total_timesteps=budget, callback=CallbackList([
            RawStepControlCallback(raw_step_budget=budget,
                                   checkpoint_frequency=36 if smoke else 10000,
                                   checkpoint_path=output / 'checkpoints', checkpoint_prefix='ckpt'),
            Audit(),
        ]))
        if model._raw_steps_seen != budget or model._n_updates != budget - warmup + 1:
            raise ValueError('Raw budget/update accounting failed')
        assert_variant(model, job['variant'])
        final = output / 'final_model.zip'
        model.save(final)
        restored = load_model(parent, final, env, device)
        assert_variant(restored, job['variant'])
        if model_state_sha(model) != model_state_sha(restored):
            raise ValueError('Saved model tensor roundtrip differs')
        verify_optimizer_settings(restored, learning_rate=1e-4, tau=.0025)
        diagnostics = model.training_diagnostics()
        diagnostics['phase3_mechanism_metadata'] = model.phase3_mechanism_metadata
        write(output / 'training_diagnostics.json', diagnostics)
        completion = {'checkpoint_sha256': sha(final), 'raw_steps': budget,
                      'updates': model._n_updates, 'variant': job['variant'], 'smoke': smoke,
                      'intervention': model.phase3_mechanism_metadata,
                      'tensor_roundtrip_equal': True, 'wall_seconds': time.time() - started,
                      'checkpoints': {relative(p): sha(p) for p in (output / 'checkpoints').glob('*.zip')}}
        seal(output / 'training_complete.json', completion)
        write(output / 'status.json', {'status': 'completed', **completion})
        return relative(output / 'training_complete.json')
    except BaseException as exc:
        write(output / 'status.json', {'status': 'failed', 'error': repr(exc), 'smoke': smoke,
                                      'raw_steps': getattr(model, '_raw_steps_seen', 0),
                                      'wall_seconds': time.time() - started})
        raise
    finally:
        env.close()
