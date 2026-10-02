from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np
import torch

from .common import (ROOT, OUT, CANDIDATES, read, write, seal, sha, digest, relative, cell, plan,
                     train_path, source, resolved_model, make_env, check_preflight, model_state_sha, lock)
from .model import make_model, load_model, learning_rate
from .driving import DrivingTrace, aggregate_driving


def train(scene, candidate, device='cuda', smoke=False):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from algos.sb3_torch import RawStepControlCallback
    from tools.phase2_model_factory_v1 import verify_optimizer_settings
    if CANDIDATES[candidate]['reuse']:
        return resolved_model(scene, candidate)
    output = train_path(scene, candidate, smoke)
    if (output / 'training_complete.json').exists():
        return resolved_model(scene, candidate, smoke)
    if output.exists():
        raise RuntimeError(f'Partial training retained, no silent restart: {output}')
    output.mkdir(parents=True, exist_ok=False)
    started, env, model = time.time(), None, None
    write(output / 'status.json', dict(status='starting', pid=os.getpid(), smoke=smoke))
    try:
        env = Monitor(make_env(scene, ('sm' if smoke else 'tr') + digest(cell(scene, candidate))[:10], training=True),
                      filename=str(output / 'train_monitor.csv'),
                      info_keywords=('raw_simulation_steps', 'is_success', 'collision', 'off_route', 'max_time'))
        model = make_model(env, scene, candidate, device=device, smoke=smoke)
        model.stability_trace_path = str(output / 'optimization_trace.jsonl')
        budget, warmup = (96, 48) if smoke else (50000, 5000)
        seal(output / 'arguments.json', dict(scene=scene, candidate=candidate, settings=model.stability_config,
             seed=930100 if smoke else 0, raw_budget=budget, warmup=warmup, smoke=smoke,
             manifest_sha256=sha(OUT / 'manifest.json'), initial_tensor_sha256=model_state_sha(model),
             environment_source=source(scene, 'lr_half')['arguments']))

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % 100 == 0:
                    write(output / 'progress.json', dict(raw_steps=self.model._raw_steps_seen,
                          updates=self.model._n_updates, updated_at=time.time()))
                return True

        model.learn(total_timesteps=budget, callback=CallbackList([
            RawStepControlCallback(raw_step_budget=budget, checkpoint_frequency=48 if smoke else 10000,
                                   checkpoint_path=output / 'checkpoints', checkpoint_prefix='ckpt'), Progress()]))
        if model._raw_steps_seen != budget or model._n_updates != budget - warmup + 1:
            raise AssertionError('Raw-step/update budget mismatch')
        rate = learning_rate(model.stability_config, budget)
        verify_optimizer_settings(model, learning_rate=rate, tau=CANDIDATES[candidate]['tau'])
        model.audit_parameters()
        final = output / 'final_model.zip'
        model.save(final)
        restored = load_model(final, scene, env, device)
        if model_state_sha(restored) != model_state_sha(model):
            raise AssertionError('Final model roundtrip changes tensors')
        verify_optimizer_settings(restored, learning_rate=rate, tau=CANDIDATES[candidate]['tau'])
        write(output / 'training_diagnostics.json', model.training_diagnostics())
        complete = dict(smoke=smoke, checkpoint_sha256=sha(final), raw_steps=budget, updates=model._n_updates,
                        manifest_sha256=sha(OUT / 'manifest.json'), tensor_roundtrip_equal=True,
                        finite_parameter_check=True, numerical_divergence_detected=False,
                        replay_size=model.replay_buffer.size(), replay_insertions=model.replay_buffer._next_insertion_id,
                        optimizer_settings=verify_optimizer_settings(model, learning_rate=rate, tau=CANDIDATES[candidate]['tau']),
                        train_monitor_sha256=sha(output / 'train_monitor.csv'),
                        optimization_trace_sha256=sha(output / 'optimization_trace.jsonl'),
                        wall_seconds=time.time() - started)
        seal(output / 'training_complete.json', complete)
        write(output / 'status.json', dict(status='completed', **complete))
        return final
    except BaseException as exc:
        write(output / 'status.json', dict(status='failed', error=repr(exc), smoke=smoke,
              numerical_divergence_detected=isinstance(exc, FloatingPointError),
              raw_steps=getattr(model, '_raw_steps_seen', None), wall_seconds=time.time() - started))
        raise
    finally:
        if env is not None:
            env.close()


def eval_path(scene, candidate, smoke=False):
    return OUT / ('smoke/eval' if smoke else 'eval') / cell(scene, candidate)


def evaluate(scene, candidate, device='cuda', smoke=False):
    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from tools.phase1_checkpoint_diagnostics import summarize, validate_report
    from tools.paper_evaluation_contract import validate_model_environment_spaces
    checkpoint = resolved_model(scene, candidate, smoke)
    output = eval_path(scene, candidate, smoke)
    count, first = (1, 930200) if smoke else (100, 420000)
    identity = dict(checkpoint_sha256=sha(checkpoint), scene=scene, candidate=candidate,
                    deployment=source(scene, candidate)['deployment'], episodes=count, seed_start=first,
                    manifest_sha256=sha(OUT / 'manifest.json'), smoke=smoke,
                    telemetry_code_sha256=sha(ROOT / 'tools/v48_stability_v1/driving.py'),
                    evaluation_code_sha256=sha(__file__))
    output.mkdir(parents=True, exist_ok=True)
    seal(output / 'identity.json', identity)
    if (output / 'result.json').exists():
        result = read(output / 'result.json')
        if result['identity'] != identity:
            raise ValueError('Evaluation identity mismatch')
        validate_report(result, count, first)
        return output / 'result.json'
    env, started = None, time.time()
    try:
        env = make_env(scene, ('se' if smoke else 'ev') + digest(cell(scene, candidate))[:10])
        model = load_model(checkpoint, scene, env, device)
        expected_steps = 96 if smoke and not CANDIDATES[candidate]['reuse'] else 50000
        if model._raw_steps_seen != expected_steps:
            raise ValueError('Evaluation is not using the exact-final model')
        validate_model_environment_spaces(model, env)
        original_tensor_hash = model_state_sha(model)
        wrapped, episodes = DrivingTrace(env), []
        for index in range(count):
            path = output / f'e{index:03d}.json'
            if path.exists():
                item = read(path)
                if item['identity_sha256'] != digest(identity) or sha(output / item['trace']) != item['trace_sha256']:
                    raise ValueError('Resumed episode identity/trace changed')
                episodes.append(item)
                continue
            episode_seed = first + index
            np.random.seed(episode_seed + 600000)
            torch.manual_seed(episode_seed + 600000)
            env._traffic_episode_index, env._traffic_roll = index, None
            detailed = evaluate_model_detailed(model, wrapped, episodes=1, seed=episode_seed,
                      deterministic=True, sumo_step_seconds=.1, policy_action_hold=1).to_dict()
            validate_report(detailed, 1, episode_seed)
            record = detailed['episode_records'][0]
            record['episode'] = index
            driving = wrapped.metrics(record['raw_steps'])
            traffic = wrapped.traffic_identity()
            if traffic['seed'] != episode_seed:
                raise ValueError('SUMO seed mismatch')
            trace = output / f'e{index:03d}.json.gz'
            temp = trace.with_suffix('.tmp')
            with gzip.open(temp, 'wt', encoding='utf-8') as f:
                json.dump(dict(initial=wrapped.initial, samples=wrapped.samples, actions=wrapped.actions), f, allow_nan=False)
            temp.replace(trace)
            item = dict(identity_sha256=digest(identity), record=record, driving=driving,
                        traffic=traffic, trace=trace.name, trace_sha256=sha(trace))
            seal(path, item)
            episodes.append(item)
            write(output / 'progress.json', dict(episodes=len(episodes), total=count, updated_at=time.time()))
        if model_state_sha(model) != original_tensor_hash:
            raise ValueError('Evaluation changed learned tensors')
        result = dict(**summarize([e['record'] for e in episodes]), identity=identity,
                      driving=aggregate_driving(episodes), tensor_state_unchanged=True,
                      episode_files={p.name: sha(p) for p in sorted(output.glob('e[0-9][0-9][0-9].json'))},
                      wall_seconds=time.time() - started,
                      evidence_scope='Development validation on seed-0-trained models, not final test or training-seed uncertainty')
        validate_report(result, count, first)
        seal(output / 'result.json', result)
        return output / 'result.json'
    finally:
        if env is not None:
            env.close()


def run_cell(scene, candidate, *, device='cuda', smoke=False):
    torch.set_num_threads(1)
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This screen is configured for GPU workers only')
    if not smoke:
        check_preflight()
    name = cell(scene, candidate)
    with lock(OUT / 'locks' / (('sm_' if smoke else '') + name + '.json')):
        train(scene, candidate, device=device, smoke=smoke)
        return evaluate(scene, candidate, device=device, smoke=smoke)
