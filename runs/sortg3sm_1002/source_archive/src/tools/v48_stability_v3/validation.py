"""Engineering-only validation, using seeds outside the scientific screen."""
from __future__ import annotations

import time
import importlib.util
import sys

import numpy as np
import torch

from .common import ROOT, OUT, SCENES, CANDIDATES, DEPLOYMENT, read, sha, write, tracked_sources, plan, make_env, model_state_sha, resolved_model


def parity_checks():
    from stable_baselines3.common.monitor import Monitor
    from algos.sb3_torch import RawStepControlCallback
    from algos.sb3_torch.evaluation import evaluate_model_detailed
    from tools.phase2_model_factory_v1 import make_phase2_model
    from .model import make_model, load_model
    from .driving import DrivingTrace
    torch.set_num_threads(1)
    hashes = {}
    # Direct comparison with the unchanged training implementation, same RNG,
    # buffers, traffic, update ordering and optimizer rates.
    legacy_path = OUT / 'evidence/legacy_sb3_configs_v4_5.py'
    spec = importlib.util.spec_from_file_location('v48_stability_verified_legacy_constructor', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    from configs.sb3_configs_v4_7 import _replace_empty_replay_buffer
    def _seed_env(env):
        # gymnasium 1.x dropped Env.seed(); fix the SUMO --seed draw so the
        # comparison is not polluted by a random simulation seed before training.
        inner = env.env
        while hasattr(inner, 'env'):
            inner = inner.env
        inner._np_random = np.random.default_rng(930100)

    def _smoke(scene, *, adapted, tag):
        env = Monitor(make_env(scene, tag, training=True))
        try:
            _seed_env(env)
            if adapted:
                model = make_model(env, scene, 'lr_half', device='cuda', smoke=True)
            else:
                # v4.8 training = frozen v4.5 learner + frozen v4.7 16-step
                # replay. v4.6/v4.8 add only selector metadata, unused here.
                model = legacy.make_model_v4_5('topo_v4_5_confident_actor_fusion', env,
                        learning_rate=5e-5, scenario=scene, batch_size=32, learning_starts=48,
                        buffer_size=128, action_repeat=3, seed=930100, device='cuda', verbose=0)
                _replace_empty_replay_buffer(model, discount=.99, action_repeat=3)
            initial = model_state_sha(model)
            settings = {k: getattr(model, k) for k in ('tau', 'gamma', 'n_steps', 'batch_size',
                        'target_entropy', 'lane_entropy_scale', 'max_grad_norm', 'representation_learning_rate')}
            model.learn(total_timesteps=96, callback=RawStepControlCallback(raw_step_budget=96))
            return dict(initial=initial, final=model_state_sha(model), updates=model._n_updates,
                        raw_steps=model._raw_steps_seen, settings=settings)
        finally:
            env.close()

    # Deterministic fields prove model construction, hyperparameters and update
    # schedule match exactly. Learned weights (``final``) are byte-exact only
    # where the frozen scenario training is itself deterministic.
    deterministic_fields = ('initial', 'settings', 'updates', 'raw_steps')
    parity_report = {}
    for scene in SCENES:
        hashes[scene] = [_smoke(scene, adapted=False, tag='parity' + scene + '_legacy'),
                         _smoke(scene, adapted=True, tag='parity' + scene + '_adapted')]
        for field in deterministic_fields:
            if hashes[scene][0][field] != hashes[scene][1][field]:
                raise ValueError('Constant adapter differs from the verified historical training path: '
                                 + scene + ' (' + field + ')')
        weights_match = hashes[scene][0]['final'] == hashes[scene][1]['final']
        parity_report[scene] = dict(deterministic_fields_match=True, final_weights_match=weights_match)
        if scene == 'cross' and not weights_match:
            raise ValueError('Constant adapter differs from the verified historical training path: cross (weights)')
    # CARLA's frozen path applies random rotation augmentation plus an online
    # target encoder, so its replay contents and learned weights are inherently
    # non-deterministic across runs. Prove that divergence is a property of the
    # historical implementation (not the adapter) by showing the unchanged path
    # disagrees with itself under identical seeds.
    carla_self = [_smoke('carla', adapted=False, tag='paritycarla_self_' + tag) for tag in ('a', 'b')]
    if carla_self[0]['final'] == carla_self[1]['final']:
        raise ValueError('CARLA historical training is deterministic; adapter weight mismatch would be a real defect')
    parity_report['carla_legacy_self_consistency'] = dict(
        final_weights_match=carla_self[0]['final'] == carla_self[1]['final'],
        note='historical carla path diverges under identical seeds; final weights are not byte-comparable')
    if 'tools.run_latent_probes' in sys.modules:
        raise ValueError('Unrelated changed probe tool was imported into training')
    results = {}
    for scene in SCENES:
        outcomes, actions = [], []
        for instrumented in (False, True):
            env = make_env(scene, 'invariance' + scene + str(int(instrumented)))
            try:
                model = load_model(resolved_model(scene, 'lr_half'), scene, env, 'cuda')
                before = model_state_sha(model)
                # Direct actor inference must not consult online or target Q.
                def forbidden(*args, **kwargs):
                    raise AssertionError('Actor-only inference consulted a critic')
                model.critic.forward = forbidden
                model.critic_target.forward = forbidden
                model.critic.all_q_from_features = forbidden
                model.critic_target.all_q_from_features = forbidden
                wrapped = DrivingTrace(env) if instrumented else env
                np.random.seed(930301)
                torch.manual_seed(930301)
                env._traffic_episode_index, env._traffic_roll = 0, None
                record = evaluate_model_detailed(model, wrapped, episodes=1, seed=930300,
                          deterministic=True, sumo_step_seconds=.1, policy_action_hold=1).to_dict()['episode_records'][0]
                if model_state_sha(model) != before:
                    raise ValueError('Evaluation altered tensors')
                if instrumented:
                    stats = wrapped.metrics(record['raw_steps'])
                    if stats['raw_step_coverage'] < .95 or stats['action_mask_coverage'] != 1.:
                        raise ValueError('Unexpectedly incomplete telemetry')
                    results[scene] = dict(record=record, telemetry=stats, critic_calls_forbidden=True)
                outcomes.append(record)
            finally:
                env.close()
        if outcomes[0] != outcomes[1]:
            raise ValueError('Read-only telemetry changes deterministic episode outcomes')
    result = dict(passed=True, training_parity=hashes, parity_report=parity_report,
                  telemetry_invariance=results, source_sha256=tracked_sources(), checked_at=time.time())
    write(OUT / 'smoke/parity.json', result)
    return result


def seal_preflight():
    sources = tracked_sources()
    test = read(OUT / 'tests.json')
    parity = read(OUT / 'smoke/parity.json')
    smoke = read(OUT / 'smoke/controller.json')
    for artifact in (test, parity, smoke):
        if not artifact['passed'] or artifact['source_sha256'] != sources:
            raise ValueError('Tests/smokes failed or refer to a different code version')
    for scene in SCENES:
        for candidate in CANDIDATES:
            result = read(OUT / 'smoke/eval' / f'{scene}__{candidate}' / 'result.json')
            if not result['identity']['smoke'] or result['identity']['deployment'] != DEPLOYMENT:
                raise ValueError('Smoke deployment differs')
    payload = dict(passed=True, manifest_sha256=sha(OUT / 'manifest.json'), source_sha256=sources,
                   test_receipt_sha256=sha(OUT / 'tests.json'), parity_receipt_sha256=sha(OUT / 'smoke/parity.json'),
                   smoke_receipt_sha256=sha(OUT / 'smoke/controller.json'), two_gpu_worker_smoke=True,
                   formal_training_started=False, prepared_at=time.time())
    write(OUT / 'preflight.json', payload)
    return payload
