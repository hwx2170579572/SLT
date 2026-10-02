"""Resumable paired evaluations with content-addressed reuse and trace records."""
from __future__ import annotations

import json
import time
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from algos.sb3_torch.evaluation import evaluate_model_detailed
from tools.phase1_checkpoint_diagnostics import summarize, validate_report
from .common import ROOT, OUT, digest, exclusive_lock, model_state_sha, read, relative, seal, sha, write
from .model import load_model
from .traffic import MODES, make_env


def apply_decoder(model, decoder):
    before = model_state_sha(model)
    policy = model.policy
    if decoder in ('actor_deterministic', 'actor_stochastic'):
        deterministic_actor = decoder == 'actor_deterministic'

        def actor_prediction(self, observation, deterministic=False):
            return self.actor(observation, deterministic=deterministic_actor)

        policy._predict = types.MethodType(actor_prediction, policy)
    elif decoder == 'no_risk_score':
        policy.collision_risk_coef = 0.
        original = policy._risk_adjusted_proposal_values

        def no_collision_terms(self, observation, batch):
            values = original(observation, batch)
            # Also remove collision disagreement; retain reward-Q disagreement.
            score = values[-1] + self.twin_uncertainty_coef * values[3]
            return (*values[:-1], score)

        policy._risk_adjusted_proposal_values = types.MethodType(no_collision_terms, policy)
    elif decoder == 'no_support_prior':
        policy.lane_prior_coef = 0.
        policy.component_prior_coef = 0.
    elif decoder == 'target_only':
        from algos.sb3_torch.hybrid_policy_v4_3 import TargetCriticDecisionAlignedSACPolicyV43
        policy._predict = types.MethodType(TargetCriticDecisionAlignedSACPolicyV43._predict, policy)
    elif decoder != 'native':
        raise ValueError(f'Unsupported diagnostic decoder: {decoder}')
    if model_state_sha(model) != before:
        raise ValueError('Inference intervention changed learned tensors')
    return {'decoder': decoder, 'learned_tensor_state_sha256': before,
            'no_training': True, 'no_action_postprocessing': True}


def discounted_collision_returns(events, gamma):
    """Per stored decision transition, matching the frozen replay gamma powers."""
    result = np.empty(len(events), dtype=np.float64)
    continuation = 0.
    for i in range(len(events) - 1, -1, -1):
        continuation = min(1., float(events[i]) + gamma * continuation)
        result[i] = continuation
    return result


class TraceEnv(gym.Wrapper):
    def __init__(self, env, model, bank=False):
        super().__init__(env)
        self.model = model
        self.bank = bank
        self.last_obs = None
        self.observations = []
        self.actions = []
        self.collisions = []
        self.risks = []
        self.commanded_speeds = []

    def reset(self, **kwargs):
        self.observations, self.actions, self.collisions, self.risks, self.commanded_speeds = [], [], [], [], []
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        if self.bank:
            self.observations.append({k: np.asarray(v).copy() for k, v in self.last_obs.items()})
            with torch.no_grad():
                obs, _ = self.model.policy.obs_to_tensor(self.last_obs)
                a = torch.as_tensor(np.asarray(action).reshape(1, -1), device=self.model.device, dtype=torch.float32)
                heads = self.model.policy.collision_critic_target(obs, a)
                risks = [float(torch.sigmoid(head).item()) for head in heads]
                self.risks.append(risks)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        self.collisions.append(bool(info.get('collision', False)))
        self.commanded_speeds.append(float(self.env.unwrapped._last_effective_target_speed))
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


def resolved_checkpoint(source, step):
    if step == 'historical':
        cp = dict(source['historical_selected'])
    else:
        cp = dict(source['checkpoints'][str(step)])
    path = ROOT / cp['path']
    actual_sha = sha(path)
    if cp['sha256'] is not None and actual_sha != cp['sha256']:
        raise ValueError(f'Model hash drift: {path}')
    cp['sha256'] = actual_sha
    return cp


def evaluate(source, step, mode, decoder='native', device='cuda', smoke=False):
    cp = resolved_checkpoint(source, step)
    key = digest({'model':cp['sha256'],'arguments':source['arguments_sha256'],
                  'native_decoder':source['decoder'],'mode':mode,'decoder':decoder,'smoke':smoke})[:20]
    with exclusive_lock(OUT / 'evaluation_locks' / f'{key}.lock'):
        return _evaluate(source,step,mode,decoder,device,smoke)


def _evaluate(source, step, mode, decoder='native', device='cuda', smoke=False):
    torch.set_num_threads(1)
    cp = resolved_checkpoint(source, step)
    count = (32 if mode == 'mechanism' else 100)
    first = 10000 if mode in ('legacy', 'legacy_train') else MODES[mode][0]
    if smoke:
        count = 1
        first = 930000  # Engineering smoke never consumes sealed scientific seeds.
    identity = {'checkpoint_sha256': cp['sha256'], 'method': source['method'],
                'scenario': source['scenario'], 'native_decoder': source['decoder'],
                'mode': mode, 'decoder': decoder, 'seed_start': first, 'episodes': count,
                'environment_arguments_sha256': source['arguments_sha256'],
                'protocol_sha256': sha(OUT / 'manifest.json'), 'smoke': smoke}
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
    if not smoke and mode == 'legacy' and decoder == 'native' and str(step) == '50000' and source.get('legacy_evaluation'):
        existing = ROOT / source['legacy_evaluation']
        if sha(existing) != source['legacy_evaluation_sha256']:
            raise ValueError('Existing evaluation changed')
        prior = read(existing)
        validate_report(prior, 100, 10000)
        if prior.get('checkpoint_sha256') != cp['sha256']:
            raise ValueError('Legacy evaluation bound to a different checkpoint')
        seal(result_path, {**summarize(prior['episode_records']), 'identity': identity,
                           'origin': 'reused_completed_evaluation', 'source': relative(existing),
                           'checkpoint': cp, 'source_id': source['id']})
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
        wrapped = TraceEnv(env, model, bank=mode == 'mechanism')
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
            if mode in ('legacy', 'legacy_train'):
                env._traffic_episode_index = offset
                env._traffic_roll = None
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
            trace = {'identity_sha256': digest(identity), 'record': record, 'behavior': behavior,
                     'traffic': getattr(env, 'phase3_traffic_receipt', None)}
            if mode == 'legacy' and source.get('legacy_evaluation'):
                reference = read(ROOT / source['legacy_evaluation'])['episode_records'][offset]
                if record['traffic_variant'] != reference['traffic_variant']:
                    raise ValueError('Legacy paired traffic schedule differs')
            if mode == 'mechanism':
                bank = {k: np.stack([o[k] for o in wrapped.observations]) for k in wrapped.observations[0]}
                bank.update(_actions=actions, _risk_heads=np.asarray(wrapped.risks),
                            _collision_events=np.asarray(wrapped.collisions),
                            _collision_return=discounted_collision_returns(wrapped.collisions, model.gamma),
                            _episode_seed=np.full(len(actions), episode_seed, dtype=np.int64))
                bank_path = directory / f'bank_{offset:03d}.npz'
                np.savez_compressed(bank_path, **bank)
                trace['bank'] = {'path': relative(bank_path), 'sha256': sha(bank_path),
                                 'risk_target_unit': 'stored_decision_transition',
                                 'gamma': float(model.gamma),
                                 'risk_outcome_censored': bool(record['timeout']),
                                 'collection_policy': decoder}
            seal(path, trace)
            records.append(record)
            write(directory / 'progress.json', {'episodes': len(records), 'total': count,
                                                'updated_at': time.time(), 'source': source['id']})
        after = model_state_sha(model)
        if after != binding['learned_tensor_state_sha256']:
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
    summary = row['summary']
    return (summary['success_rate'], -summary['collision_rate'], -summary['timeout_rate'],
            row['checkpoint']['raw_steps'])


def select(source, evaluations):
    rows = [read(ROOT / path) for path in evaluations]
    steps = sorted(row['checkpoint']['raw_steps'] for row in rows)
    if steps != [10000, 20000, 30000, 40000, 50000]:
        raise ValueError('Common selection requires exactly five fixed checkpoints')
    for row in rows:
        if (row['identity']['mode'] != 'selection' or row['identity']['smoke']
                or row['identity']['episodes'] != 100 or row['source_id'] != source['id']):
            raise ValueError('Invalid selection evidence')
        validate_report(row, 100, 310000)
    winner = max(rows, key=selection_key)
    result = {'source': source['id'], 'selected_checkpoint': winner['checkpoint'],
              'selected_step': winner['checkpoint']['raw_steps'],
              'candidate_results': [{'path': path, 'sha256': sha(ROOT / path)} for path in evaluations],
              'ranking': 'success desc, collision asc, timeout asc, later checkpoint',
              'scoring_outcomes_used': False}
    path = OUT / 'selection' / f'{source["id"]}.json'
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k != 'sealed_at_unix'} != result:
            raise ValueError('Sealed checkpoint choice changed')
        return old
    result['sealed_at_unix'] = time.time()
    seal(path, result)
    return result
