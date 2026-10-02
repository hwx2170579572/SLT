from __future__ import annotations

import zipfile
import json

import numpy as np

from .common import ROOT, OUT, SCENES, CANDIDATES, PROTOCOL, DEPLOYMENT, read, sha, seal, write, relative, cell
from .curves import analyze_rows, read_monitor, draw_curves


def prepare():
    sources = {}
    for scene in SCENES:
        for candidate, settings in CANDIDATES.items():
            spec = dict(scene=scene, candidate=candidate, seed=0, reuse=bool(settings['reuse']),
                        deployment=DEPLOYMENT)
            if spec['reuse']:
                directory = ROOT / 'results_phase2_runtime_v2/screen' / f'v4_8__{scene}__{candidate}__seed0'
                for key, name in (('checkpoint', 'final_model.zip'), ('training_log', 'train_monitor.csv'),
                                  ('arguments', 'arguments.json'), ('training_complete', 'training_complete.json')):
                    spec[key] = relative(directory / name)
                    spec[key + '_sha256'] = sha(directory / name)
                a, done = read(directory / 'arguments.json'), read(directory / 'training_complete.json')
                if a['smoke'] or a['seed'] != 0 or a['raw_budget'] != 50000:
                    raise ValueError('Incompatible existing cell')
                spec['historical_decoder_metadata_not_used'] = a['decoder']
                if done['checkpoint_sha256'] != spec['checkpoint_sha256'] or done['learner_updates'] != 45001:
                    raise ValueError('Incomplete/changed existing training')
                for key in ('learning_rate', 'tau'):
                    if a['candidate'][key] != settings[key]:
                        raise ValueError('Candidate identity collision')
                for key in ('batch_size', 'buffer_size'):
                    if a[key] != settings[key]:
                        raise ValueError('Replay/batch differs')
            sources[cell(scene, candidate)] = spec
    manifest = dict(schema='v48-stability/v1', training_seeds=[0], scenes=list(SCENES),
                    candidates=CANDIDATES, sources=sources, new_training_cells=8, reused_training_cells=2,
                    raw_budget=50000, learning_starts=5000, gradient_updates=45001, action_repeat=3,
                    max_gpu_workers=2, cpu_workers=0, auto_launch=False, primary_checkpoint='exact_final_50000',
                    deployment=DEPLOYMENT, checkpoint_selector=False, critic_deployment_decoder=False,
                    evaluation=dict(episodes=100, seed_start=420000, deterministic=True,
                                    physical_partition='evaluation', purpose='development parameter selection',
                                    formal_test=False, shared_traffic_templates_in_carla=True),
                    curve=dict(episode_window=20, zero_padding=True, initial_value=0, ema_old_weight=.999,
                               update_unit='every raw simulation step', display_stride=100,
                               retrospective_current_episode_backfill=True, average_candidates=False),
                    gates=dict(eval_success_tolerance=.03, eval_collision_tolerance=.03, eval_timeout_tolerance=.03,
                               train_tail_success_tolerance=.03, train_tail_reward_tolerance=.06,
                               train_success_auc_tolerance=.02, max_drawdown_tolerance=.02,
                               max_oscillation_relative_tolerance=.10, min_tail_std_relative_improvement=.10),
                    environment_protocol=relative(PROTOCOL), environment_protocol_sha256=sha(PROTOCOL),
                    variance_scope='single-run fluctuation only; no multi-training-seed claim',
                    efficiency_is_not_a_selection_gate=True)
    seal(OUT / 'manifest.json', manifest)
    evidence()
    return manifest


def evidence():
    manifest = read(ROOT / 'results_phase2_training_curves_v2_episode20_ema999/source_manifest.json')
    old_values = read(ROOT / 'results_phase2_training_curves_v2_episode20_ema999/curve_values.json')
    results, maximum_error, points = {}, 0., 0
    for scene in SCENES:
        plots = {}
        for candidate in ('lr_half', 'lr_quarter', 'control', 'tau_half', 'tau_double'):
            key = f'v4_8__{scene}__{candidate}__seed0'
            spec = manifest[key]
            path = ROOT / spec['training_log']
            if sha(path) != spec['training_log_sha256']:
                raise ValueError('Prior curve source changed')
            stats, curves = analyze_rows(read_monitor(path))
            for metric, a in curves.items():
                prior = np.asarray(old_values[key][metric])
                if prior.shape != a.shape:
                    raise ValueError('Reference curve shape mismatch')
                maximum_error = max(maximum_error, float(np.max(np.abs(a - prior))))
                points += len(a)
            stats['source'] = spec
            if candidate != 'control':
                directory = path.parent
                stats['aggregate_training_diagnostics'] = read(directory / 'training_diagnostics.json')
                stats['arguments'] = read(directory / 'arguments.json')
                with zipfile.ZipFile(directory / 'final_model.zip') as z:
                    data = json.loads(z.read('data'))
                stats['stored_decision_timesteps'] = data['num_timesteps']
                stats['replay_insertion_upper_bound'] = data['num_timesteps'] + stats['complete_episodes']
                stats['replay_capacity'] = data['buffer_size']
            results[f'{scene}__{candidate}'] = stats
            plots[candidate] = curves
        draw_curves(plots, {'lr_half': 'Reference: LR 5e-5, tau .005', 'lr_quarter': 'Existing: LR 2.5e-5, tau .005',
                           'control': 'Historical LR 1e-4 (code equivalence unresolved)',
                           'tau_half': 'Existing LR 1e-4, tau .0025', 'tau_double': 'Existing LR 1e-4, tau .01'},
                    OUT / 'evidence', scene, scene.upper() + ' | existing evidence')
    if maximum_error >= 1e-12:
        raise ValueError('Reference EMA transform differs')
    write(OUT / 'evidence/curve_diagnostics.json', results)
    write(OUT / 'evidence/reference_validation.json', dict(points=points, maximum_absolute_error=maximum_error,
          reference=relative(ROOT / 'results_phase2_training_curves_v2_episode20_ema999/curve_values.json')))
    return results
