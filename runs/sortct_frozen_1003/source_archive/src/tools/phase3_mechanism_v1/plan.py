"""Seal the diagnostic matrix before new scientific outcomes are observed."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .common import (ROOT, OUT, PLAN_DIR, PROTOCOL, SCENES, MECHANISM_SCENES,
                     STEPS, digest, read, relative, seal, sha, validate_frozen_sources)
from .inventory import scan
from .model import VARIANTS


def checkpoint(path, steps=None):
    path = ROOT / path
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
        actual = int(data['_raw_steps_seen'])
    if steps is not None and actual != steps:
        raise ValueError(f'Checkpoint budget mismatch: {path}: {actual} != {steps}')
    return {'path': relative(path), 'sha256': sha(path), 'raw_steps': actual}


def prepare():
    validate_frozen_sources()
    inventory = scan()
    phase1 = read(ROOT / 'results_phase1_checkpoint_diagnostics_v1/plan.json')
    sources = {}
    for method, candidate in (('mst_slt', 'control'), ('v4_8', 'control'),
                              ('v4_13', 'control'), ('v4_8', 'lr_half'), ('v4_13', 'tau_half')):
        for scene in SCENES:
            original = next(s for s in phase1['sources'] if s['method'] == method and s['scenario'] == scene)
            if candidate == 'control':
                run = ROOT / original['run']
                reused = ROOT / f'results_phase1_checkpoint_diagnostics_v1/jobs/{method}__{scene}__exact_final/result.json'
            else:
                run = ROOT / f'results_phase2_runtime_v2/screen/{method}__{scene}__{candidate}__seed0'
                reused = run / 'evaluation.json'
                if read(run / 'status.json')['status'] != 'completed':
                    raise ValueError(f'Incomplete source: {run}')
            key = f'{method}__{scene}__{candidate}'
            points = {}
            for step in STEPS:
                paths = [run / 'final_model.zip'] if step == 50000 else list((run / 'checkpoints').glob(f'*_raw_{step}_steps.zip'))
                if len(paths) != 1:
                    raise ValueError(f'Missing/nonunique checkpoint: {key}: {step}')
                points[str(step)] = checkpoint(paths[0], step)
            source = {
                'id': key, 'method': method, 'candidate': candidate, 'scenario': scene,
                'decoder': original['decoder'], 'arguments': relative(run / 'arguments.json'),
                'arguments_sha256': sha(run / 'arguments.json'), 'checkpoints': points,
                'legacy_evaluation': relative(reused), 'legacy_evaluation_sha256': sha(reused),
                'training_seed': 0, 'new_training': False,
            }
            if candidate == 'control':
                detailed = read(run / 'paper_evaluation_detailed.json')
                source['historical_selected'] = checkpoint(run / Path(detailed['model']).name)
                source['historical_selected_kind'] = detailed.get('selected_checkpoint_kind', 'exact_final')
            sources[key] = source

    new_training = []
    reuse_notes = []
    for scene in MECHANISM_SCENES:
        full_key = f'v4_13__{scene}__tau_half'
        for variant in VARIANTS:
            if variant == 'g1_s025':
                reuse_notes.append({'variant': variant, 'scenario': scene, 'source': full_key,
                                    'reason': 'Matched seed-0 50k tau=.0025 full model already complete'})
                continue
            key = f'{variant}__{scene}'
            candidates = [r for r in inventory['rows'] if r['final_model_exists']
                          and r['phase3_variant'] == variant and r['scenario'] == scene
                          and r['seed'] == 0 and r['raw_budget'] == 50000]
            if candidates:
                # The exact output remains in the original sealed plan on resume.
                if not (OUT / 'manifest.json').exists():
                    raise ValueError(f'Existing matching phase3 training needs explicit reuse binding: {key}')
            parent = sources[full_key]
            output = f'r3m1/train/{key}'
            sources[key] = {
                'id': key, 'method': 'v4_13', 'candidate': variant, 'scenario': scene,
                'variant': variant, 'decoder': 'target_critic',
                'arguments': parent['arguments'], 'arguments_sha256': parent['arguments_sha256'],
                'checkpoints': {str(step): {'path': f'{output}/final_model.zip' if step == 50000
                                           else f'{output}/checkpoints/ckpt_raw_{step}_steps.zip',
                                           'raw_steps': step, 'sha256': None} for step in STEPS},
                'training_seed': 0, 'new_training': True, 'parent': full_key,
            }
            new_training.append({'id': key, 'variant': variant, 'source': full_key,
                                 'scenario': scene, 'raw_budget': 50000, 'seed': 0,
                                 'learning_rate': 1e-4, 'tau': .0025})
    plan = {
        'schema': 'phase3-mechanism-v1', 'original_frozen_protocol_sha256': sha(PROTOCOL),
        'scope': 'Selection, factorial ablation, decoder diagnosis, gradient/risk and traffic robustness; all development evidence.',
        'multiseed_confirmation_deferred_by_user': True,
        'new_training_seeds': [0], 'gpu_workers': 3, 'cpu_workers': 0,
        'efficiency_veto': False, 'timeout_remains_task_failure': True,
        'sources': sources, 'training_jobs': new_training, 'reuse_notes': reuse_notes,
        'selection': {'steps': list(STEPS), 'episodes_per_checkpoint': 100,
                      'seed_start': 310000, 'order': ['success_desc', 'collision_asc', 'timeout_asc', 'later_checkpoint'],
                      'historical_best_not_in_common_candidate_set': True},
        'score': {'episodes': 100, 'seed_start': 320000,
                  'weights': ['selected', 'exact_final', 'historical_selected_if_available'],
                  'seal_selection_before_scoring': True},
        'robustness': {'modes': ['robust_lower', 'robust_higher'], 'episodes': 100,
                       'vehicle_multipliers_relative_to_frozen_density': [.9, 1.1],
                       'methods': ['mst_slt/control', 'v4_8/lr_half', 'v4_13/tau_half']},
        'mechanism': {'scenes': list(MECHANISM_SCENES), 'variants': VARIANTS,
                      'bank_episodes': 32, 'bank_seed_start': 350000, 'batch_size': 32,
                      'maximum_bank_samples': 4096, 'checkpoint_steps': [10000, 30000, 50000],
                      'collection_policy': 'full_tau_half_stochastic_actor',
                      'bank_is_original_training_replay': False},
        'decoder_interventions': ['actor_deterministic', 'actor_stochastic', 'no_risk_score', 'no_support_prior'],
        'traffic': {'new_departure_jitter_seconds': [-.75, .75],
                    'split_unit': 'new social traffic realization and simulation seed',
                    'carla_single_template_shared_with_training': True,
                    'no_new_map_generalization_claim': True, 'formal_test_accessed': False},
        'scientific_caveats': ['single training seed', 'historical code equivalence not fully established',
                                'new traffic realizations retain source route templates',
                                'risk critic is discounted collision-return estimator, not certified probability'],
    }
    if (OUT / 'manifest.json').exists():
        old = read(OUT / 'manifest.json')
        if old != plan:
            raise ValueError('Existing phase3 manifest changed; create a new version instead')
    seal(OUT / 'manifest.json', plan)
    seal(PLAN_DIR / 'protocol.json', plan)
    return plan


def prepare_diagnostic_sources():
    sources = {}
    manifest = read(OUT / 'manifest.json')
    for candidate in ('tau_half', 'tau_double'):
        key = f'v4_8__carla__{candidate}'
        run = ROOT / f'results_phase2_runtime_v2/screen/{key}__seed0'
        if read(run / 'status.json')['status'] != 'completed':
            raise ValueError('CARLA diagnostic source incomplete')
        parent = manifest['sources']['v4_8__carla__control']
        sources[key] = {
            'id': key, 'method': 'v4_8', 'candidate': candidate, 'scenario': 'carla',
            'decoder': parent['decoder'], 'arguments': relative(run / 'arguments.json'),
            'arguments_sha256': sha(run / 'arguments.json'),
            'checkpoints': {'50000': checkpoint(run / 'final_model.zip', 50000)},
            'legacy_evaluation': relative(run / 'evaluation.json'),
            'legacy_evaluation_sha256': sha(run / 'evaluation.json'),
            'training_seed': 0, 'new_training': False, 'diagnostic_only': True,
        }
    value = {'schema': 'phase3-diagnostic-source-extension-v1', 'sources': sources,
             'reason': 'Previously observed CARLA train/deployment failures; inference diagnosis only',
             'selection_candidate_extension': False, 'additional_training': False}
    seal(OUT / 'diagnostic_sources.json', value)
    return value
