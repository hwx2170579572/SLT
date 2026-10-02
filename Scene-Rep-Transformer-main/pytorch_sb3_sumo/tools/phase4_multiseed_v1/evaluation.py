"""Resumable paired evaluations with content-addressed reuse and sealed prior evidence.

Two reuse paths exist and both are explicitly declared, never silent:

1. Content-addressed cache: an identical (model, arguments, decoder, mode) tuple
   already evaluated under ``r4m1/eval`` returns the sealed result, and a
   partially finished evaluation resumes only its missing episodes.
2. Bound phase-3 evidence: seed-0 cells whose training, unified selection and
   scoring were already sealed by the frozen phase-3 run carry an explicit
   sha256 binding.  The prior artefact is re-verified and re-wrapped as this
   phase's own receipt; no simulator step is repeated.
"""
from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from algos.sb3_torch.evaluation import evaluate_model_detailed
from tools.phase1_checkpoint_diagnostics import summarize, validate_report
from .common import ROOT, OUT, digest, exclusive_lock, model_state_sha, read, relative, seal, sha, write
from .config import STEPS
from .model import load_model
from .traffic import MODES, make_env


def apply_decoder(model, decoder):
    before = model_state_sha(model)
    if decoder != 'native':
        raise ValueError(f'Phase-4 scoring uses the frozen native decoder, not {decoder!r}')
    if model_state_sha(model) != before:
        raise ValueError('Inference intervention changed learned tensors')
    return {'decoder': decoder, 'learned_tensor_state_sha256': before,
            'no_training': True, 'no_action_postprocessing': True}


class TraceEnv(gym.Wrapper):
    """Record per-decision behaviour without touching the learned policy."""

    def __init__(self, env, model):
        super().__init__(env)
        self.model = model
        self.last_obs = None
        self.actions = []
        self.collisions = []
        self.commanded_speeds = []

    def reset(self, **kwargs):
        self.actions, self.collisions, self.commanded_speeds = [], [], []
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        self.collisions.append(bool(info.get('collision', False)))
        self.commanded_speeds.append(float(self.env.unwrapped._last_effective_target_speed))
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


def resolved_checkpoint(source, step):
    cp = dict(source['checkpoints'][str(step)])
    path = ROOT / cp['path']
    actual_sha = sha(path)
    if cp['sha256'] is not None and actual_sha != cp['sha256']:
        raise ValueError(f'Model hash drift: {path}')
    cp['sha256'] = actual_sha
    return cp


def bound_evidence(source, step, mode, decoder):
    """Explicitly declared phase-3 evidence for this exact cell, or ``None``."""
    binding = source.get('reused_evaluations', {}).get(f'{mode}::{step}::{decoder}')
    if binding is None:
        return None
    path = ROOT / binding['path']
    if sha(path) != binding['sha256']:
        raise ValueError(f'Bound prior evidence changed: {path}')
    return read(path)


def _identity(source, cp, mode, decoder, first, count, smoke):
    return {'checkpoint_sha256': cp['sha256'], 'method': source['method'],
            'scenario': source['scenario'], 'native_decoder': source['decoder'],
            'mode': mode, 'decoder': decoder, 'seed_start': first, 'episodes': count,
            'environment_arguments_sha256': source['arguments_sha256'],
            'protocol_sha256': sha(OUT / 'manifest.json'),
            'training_seed': source['training_seed'], 'smoke': smoke}


def evaluate(source, step, mode, decoder='native', device='cuda', smoke=False):
    if mode not in MODES:
        raise ValueError(f'Unsupported phase-4 mode: {mode}')
    cp = resolved_checkpoint(source, step)
    key = digest({'model': cp['sha256'], 'arguments': source['arguments_sha256'],
                  'native_decoder': source['decoder'], 'mode': mode, 'decoder': decoder,
                  'seed': source['training_seed'], 'smoke': smoke})[:20]
    with exclusive_lock(OUT / 'evaluation_locks' / f'{key}.lock'):
        return _evaluate(source, step, mode, decoder, device, smoke)


def _evaluate(source, step, mode, decoder='native', device='cuda', smoke=False):
    torch.set_num_threads(1)
    cp = resolved_checkpoint(source, step)
    count = MODES[mode][1]
    first = MODES[mode][0]
    if smoke:
        count = 1
        first = 930000  # Engineering smoke never consumes sealed scientific seeds.
    # A reused cell may only serve sealed evidence.  Anything else is outside the
    # protocol and fails loudly rather than quietly re-running the simulator.
    prior = None
    if not smoke and decoder == 'native' and source.get('reused_evaluations') is not None:
        prior = bound_evidence(source, step, mode, decoder)
        if prior is None:
            raise ValueError(f'Reused cell requested an unsealed evaluation: '
                             f'{source["id"]} {mode} {step} {decoder}')
    identity = _identity(source, cp, mode, decoder, first, count, smoke)
    key = digest(identity)[:20]
    directory = OUT / ('smoke_eval' if smoke else 'eval') / key
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if result['identity'] != identity:
            raise ValueError('Evaluation cache identity mismatch')
        validate_report(result, count, first)
        return relative(result_path)
    seal(directory / 'identity.json', identity)

    if prior is not None:
        validate_report(prior, count, first)
        if (prior['identity']['checkpoint_sha256'] != cp['sha256']
                or prior['identity']['mode'] != mode or prior['identity']['seed_start'] != first
                or prior['identity']['episodes'] != count
                or prior['identity']['method'] != source['method']
                or prior['identity']['scenario'] != source['scenario']):
            raise ValueError('Bound prior evidence does not match this cell')
        seal(result_path, {**summarize(prior['episode_records']), 'identity': identity,
                           'origin': 'reused_sealed_prior_evidence',
                           'source': source['reused_evaluations'][f'{mode}::{step}::{decoder}']['path'],
                           'prior_identity': prior['identity'],
                           'checkpoint': cp, 'source_id': source['id'],
                           'tensor_state_unchanged': True, 'wall_seconds': 0.})
        return relative(result_path)

    env = make_env(source, mode, key)
    model = None
    started = time.time()
    try:
        model = load_model(source, cp['path'], env, device)
        from tools.paper_evaluation_contract import validate_model_environment_spaces
        validate_model_environment_spaces(model, env)
        if int(model._raw_steps_seen) != cp['raw_steps']:
            raise ValueError('Loaded model raw-step mismatch')
        binding = apply_decoder(model, decoder)
        wrapped = TraceEnv(env, model)
        records = []
        for offset in range(count):
            path = directory / f'episode_{offset:03d}.json'
            if path.exists():
                record = read(path)
                if record['identity_sha256'] != digest(identity):
                    raise ValueError('Episode belongs to another evaluation')
                records.append(record['record'])
                continue
            episode_seed = first + offset
            torch.manual_seed(episode_seed + 600000)
            np.random.seed(episode_seed + 600000)
            detailed = evaluate_model_detailed(model, wrapped, episodes=1,
                                               seed=episode_seed, deterministic=True,
                                               sumo_step_seconds=.1, policy_action_hold=1).to_dict()
            validate_report(detailed, 1, episode_seed)
            record = detailed['episode_records'][0]
            record['episode'] = offset
            actions = np.stack(wrapped.actions)
            behavior = {'decisions': len(actions),
                        'normalized_action_mean': actions.mean(axis=0).tolist(),
                        'mean_effective_target_speed': float(np.mean(wrapped.commanded_speeds)),
                        'fraction_effective_target_speed_below_0_1_mps': float(np.mean(np.asarray(wrapped.commanded_speeds) < .1))}
            seal(path, {'identity_sha256': digest(identity), 'record': record, 'behavior': behavior,
                        'traffic': getattr(env, 'phase4_traffic_receipt', None)})
            records.append(record)
            write(directory / 'progress.json', {'episodes': len(records), 'total': count,
                                                'updated_at': time.time(), 'source': source['id']})
        if model_state_sha(model) != binding['learned_tensor_state_sha256']:
            raise ValueError('Evaluation mutated learned state')
        result = {**summarize(records), 'identity': identity, 'origin': 'new_real_rollouts',
                  'source_id': source['id'], 'checkpoint': cp, 'intervention': binding,
                  'tensor_state_unchanged': True, 'wall_seconds': time.time() - started}
        validate_report(result, count, first)
        seal(result_path, result)
        return relative(result_path)
    finally:
        env.close()


def selection_key(row):
    """Success desc, collision asc, timeout asc, later checkpoint."""
    summary = row['summary']
    return (summary['success_rate'], -summary['collision_rate'], -summary['timeout_rate'],
            row['checkpoint']['raw_steps'])


def select(source, evaluations):
    rows = [read(ROOT / path) for path in evaluations]
    steps = sorted(row['checkpoint']['raw_steps'] for row in rows)
    if steps != list(STEPS):
        raise ValueError('Common selection requires exactly five fixed checkpoints')
    for row in rows:
        if (row['identity']['mode'] != 'selection' or row['identity']['smoke']
                or row['identity']['episodes'] != 100 or row['source_id'] != source['id']
                or row['identity']['method'] != source['method']
                or row['identity']['scenario'] != source['scenario']
                or row['identity']['training_seed'] != source['training_seed']):
            raise ValueError('Invalid selection evidence')
        validate_report(row, 100, MODES['selection'][0])
    winner = max(rows, key=selection_key)
    result = {'source': source['id'], 'selected_checkpoint': winner['checkpoint'],
              'selected_step': winner['checkpoint']['raw_steps'],
              'candidate_results': [{'path': path, 'sha256': sha(ROOT / path)} for path in evaluations],
              'ranking': 'success desc, collision asc, timeout asc, later checkpoint',
              'scoring_outcomes_used': False}
    path = OUT / 'selection' / f'{source["id"]}.json'
    if path.exists():
        old = read(path)
        if {k: v for k, v in old.items() if k != 'sealed_at_unix'} != result:
            raise ValueError('Sealed checkpoint choice changed')
        return old
    result['sealed_at_unix'] = time.time()
    seal(path, result)
    return result
