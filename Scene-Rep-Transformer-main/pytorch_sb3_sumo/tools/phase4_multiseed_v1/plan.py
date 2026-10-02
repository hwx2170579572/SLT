"""Seal the 54-cell multi-seed matrix and bind the reusable seed-0 evidence.

The plan is written before any new training outcome is observed.  Cells whose
seed-0 result was already produced by the frozen phase-2 screen run and sealed
by the frozen phase-3 run are bound here by sha256, so the controller dispatches
them straight to the evaluation cache without retraining.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .common import (OUT, PHASE2_RUNTIME, PHASE3_OUT, PLAN_DIR, PROTOCOL, ROOT, TUNING_CONTRACT,
                     digest, read, relative, seal, sha, validate_frozen_sources)
from .config import (METHODS, SCENES, SELECTION, SCORE, STEPS, TRAINING_SEEDS, cell_id,
                     is_reused, phase3_cell_id)
from .traffic import requested_arguments


def checkpoint(path, steps=None):
    path = ROOT / path
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
        actual = int(data['_raw_steps_seen'])
    if steps is not None and actual != steps:
        raise ValueError(f'Checkpoint budget mismatch: {path}: {actual} != {steps}')
    return {'path': relative(path), 'sha256': sha(path), 'raw_steps': actual}


def phase1_source(method, scene):
    plan = read(ROOT / 'results_phase1_checkpoint_diagnostics_v1/plan.json')
    return next(s for s in plan['sources'] if s['method'] == method and s['scenario'] == scene)


def aligned_environment_arguments(method, scene, seed):
    """Phase-1 environment arguments plus the two frozen overrides, seed-resolved.

    The phase-2 screen run recorded this exact dict for seed 0; the equivalence is
    asserted below before any reuse is claimed.
    """
    original = phase1_source(method, scene)
    requested = dict(requested_arguments({'arguments': relative(ROOT / original['run'] / 'arguments.json'),
                                          'scenario': scene}))
    return {**requested, 'seed': seed, 'gui': False, 'evaluation_split': 'validation'}


def assert_environment_equivalence(method, scene):
    """The reused seed-0 run must have used exactly the arguments we now declare."""
    candidate = METHODS[method]['candidate']
    recorded = read(PHASE2_RUNTIME / 'screen' / f'{method}__{scene}__{candidate}__seed0' / 'arguments.json')
    if recorded['inherited_environment_arguments'] != aligned_environment_arguments(method, scene, 0):
        raise ValueError(f'Reused run used different environment arguments: {method}/{scene}')
    if recorded['candidate']['id'] != candidate:
        raise ValueError(f'Reused run used a different candidate: {method}/{scene}')
    settings = METHODS[method]
    if (recorded['candidate']['learning_rate'] != settings['learning_rate']
            or recorded['candidate']['tau'] != settings['tau']):
        raise ValueError(f'Reused run used different optimizer settings: {method}/{scene}')
    return True


def phase3_evaluation_binding(phase3_id, mode, job_step):
    """Locate the sealed phase-3 evaluation receipt for one mode/step of a cell."""
    matches = []
    for receipt_path in sorted((PHASE3_OUT / 'jobs').glob('*.json')):
        receipt = read(receipt_path)
        job = receipt['job']
        if (job['kind'] == 'eval' and job['source'] == phase3_id and job['mode'] == mode
                and str(job['step']) == str(job_step)):
            if sha(ROOT / receipt['artifact']) != receipt['artifact_sha256']:
                raise ValueError(f'Sealed phase-3 evidence changed: {receipt["artifact"]}')
            matches.append({'path': receipt['artifact'], 'sha256': receipt['artifact_sha256'],
                            'job': job['id']})
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one sealed {mode} evaluation: {phase3_id}: {job_step}')
    return matches[0]


def bind_seed_zero_evidence(method, scene):
    """Every selection checkpoint and both scored models, bound by sha256."""
    phase3_id = phase3_cell_id(method, scene)
    run = PHASE2_RUNTIME / 'screen' / f'{method}__{scene}__{METHODS[method]["candidate"]}__seed0'
    checkpoints = {str(step): checkpoint(run / 'final_model.zip' if step == 50000
                                         else _only((run / 'checkpoints').glob(f'*_raw_{step}_steps.zip')), step)
                   for step in STEPS}
    selection = read(PHASE3_OUT / 'selection' / f'{phase3_id}.json')
    selected_step = int(selection['selected_step'])
    if selected_step not in STEPS:
        raise ValueError(f'Sealed selection picked an out-of-protocol checkpoint: {phase3_id}')
    bindings = {}
    for step in STEPS:
        bindings[f'selection::{step}::native'] = phase3_evaluation_binding(phase3_id, 'selection', step)
    bindings[f'score::{selected_step}::native'] = phase3_evaluation_binding(phase3_id, 'score', 'selected')
    exact_final = phase3_evaluation_binding(phase3_id, 'score', 50000)
    existing = bindings.get('score::50000::native')
    if existing is not None and existing['sha256'] != exact_final['sha256']:
        raise ValueError(f'Selected and exact-final evidence disagree for step 50000: {phase3_id}')
    bindings['score::50000::native'] = exact_final
    for key, binding in bindings.items():
        mode, step, _ = key.split('::')
        prior = read(ROOT / binding['path'])
        if prior['identity']['checkpoint_sha256'] != checkpoints[step]['sha256']:
            raise ValueError(f'Bound evidence is not this checkpoint: {phase3_id}: {key}')
        expected_count, expected_start = (100, 310000) if mode == 'selection' else (100, 320000)
        if prior['identity']['episodes'] != expected_count or prior['identity']['seed_start'] != expected_start:
            raise ValueError(f'Bound evidence uses a different protocol: {phase3_id}: {key}')
    return {'checkpoints': checkpoints, 'bindings': bindings, 'selected_step': selected_step,
            'run': relative(run),
            'phase3_selection': relative(PHASE3_OUT / 'selection' / f'{phase3_id}.json')}


def _only(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one checkpoint file: {paths}')
    return paths[0]


def prepare():
    validate_frozen_sources()
    contract = read(TUNING_CONTRACT)
    for method, settings in METHODS.items():
        candidate = next(c for c in contract['candidates'] if c['id'] == settings['candidate'])
        if (candidate['learning_rate'] != settings['learning_rate'] or candidate['tau'] != settings['tau']):
            raise ValueError(f'Optimizer settings drifted from the frozen tuning contract: {method}')

    sources, training_jobs, reuse_notes = {}, [], []
    for method, scene, seed in [(m, s, k) for m in METHODS for s in SCENES for k in TRAINING_SEEDS]:
        key = cell_id(method, scene, seed)
        original = phase1_source(method, scene)
        settings = METHODS[method]
        if method == 'v4_13' and original['decoder'] != 'target_critic':
            raise ValueError(f'v4_13 deployment rule changed: {scene}')
        arguments = aligned_environment_arguments(method, scene, seed)
        if is_reused(method, seed):
            assert_environment_equivalence(method, scene)
            bound = bind_seed_zero_evidence(method, scene)
            if bound['selected_step'] != read(PHASE3_OUT / 'selection' / f'{phase3_cell_id(method, scene)}.json')['selected_step']:
                raise ValueError('Sealed selection outcome drifted')
            sources[key] = {
                'id': key, 'method': method, 'candidate': settings['candidate'], 'scenario': scene,
                'decoder': original['decoder'], 'training_seed': seed, 'new_training': False,
                'training_origin': 'reused_sealed_seed_zero',
                'source_run': bound['run'],
                'arguments': relative(ROOT / original['run'] / 'arguments.json'),
                'arguments_sha256': sha(ROOT / original['run'] / 'arguments.json'),
                'checkpoints': bound['checkpoints'],
                'reused_evaluations': bound['bindings'],
                'reused_selection_receipt': bound['phase3_selection'],
                'environment_arguments': arguments,
            }
            reuse_notes.append({'cell': key, 'kind': 'training_and_evaluation',
                                'reason': 'Seed-0 training, unified selection and dual-model scoring '
                                          'were completed and sealed by the frozen phase-2/phase-3 runs',
                                'source_run': bound['run']})
        else:
            output = f'r4m1/train/{key}'
            sources[key] = {
                'id': key, 'method': method, 'candidate': settings['candidate'], 'scenario': scene,
                'decoder': original['decoder'], 'training_seed': seed, 'new_training': True,
                'training_origin': 'new_training_this_phase',
                'arguments': relative(ROOT / original['run'] / 'arguments.json'),
                'arguments_sha256': sha(ROOT / original['run'] / 'arguments.json'),
                'checkpoints': {str(step): {'path': f'{output}/final_model.zip' if step == 50000
                                           else f'{output}/checkpoints/ckpt_raw_{step}_steps.zip',
                                           'raw_steps': step, 'sha256': None} for step in STEPS},
                'environment_arguments': arguments,
            }
            training_jobs.append({'id': key, 'method': method, 'candidate': settings['candidate'],
                                  'scenario': scene, 'seed': seed,
                                  'raw_budget': 50000, 'learning_rate': settings['learning_rate'],
                                  'tau': settings['tau']})
            if seed == 0:
                reuse_notes.append({'cell': key, 'kind': 'retrained_for_alignment',
                                    'reason': 'Historical seed-0 MST+SLT models come from two different '
                                              'trainer lineages; the aligned trainer retrains them'})

    plan = {
        'schema': 'phase4-multiseed-v1',
        'scope': 'Three methods by six high-density scenarios by three training seeds, each with '
                 'unified checkpoint selection and dual-model (selected, exact_final) scoring. '
                 'Development evidence only.',
        'frozen_protocol_sha256': sha(PROTOCOL), 'tuning_contract_sha256': sha(TUNING_CONTRACT),
        'matrix': {'methods': list(METHODS), 'scenarios': list(SCENES), 'training_seeds': list(TRAINING_SEEDS),
                   'cells': len(sources), 'new_training_cells': len(training_jobs),
                   'reused_seed_zero_cells': len(sources) - len(training_jobs)},
        'methods': METHODS, 'gpu_workers': 3, 'cpu_workers': 0,
        'source_registry_sha256': digest({k: {'path': v['arguments'], 'decoder': v['decoder'],
                                              'seed': v['training_seed']} for k, v in sources.items()}),
        'sources': sources, 'training_jobs': training_jobs, 'reuse_notes': reuse_notes,
        'selection': SELECTION, 'score': SCORE,
        'traffic': {'new_departure_jitter_seconds': [-.75, .75],
                    'split_unit': 'new social traffic realization and simulation seed',
                    'shared_across_training_seeds': True,
                    'carla_single_template_shared_with_training': True,
                    'no_new_map_generalization_claim': True, 'formal_test_accessed': False},
        'evaluation': {'selection_seed_start': SELECTION['seed_start'],
                       'score_seed_start': SCORE['seed_start'],
                       'identical_across_training_seeds': True,
                       'timeout_remains_task_failure': True, 'efficiency_veto': False},
        'scientific_caveats': ['development evidence only', 'no formal-test access',
                               'risk critic is a discounted collision-return estimator, not a certified probability',
                               'reused traffic realizations retain the source route templates'],
    }
    path = OUT / 'manifest.json'
    if path.exists():
        old = read(path)
        if old != plan:
            raise ValueError('Existing phase-4 manifest changed; create a new version instead')
    seal(path, plan)
    seal(PLAN_DIR / 'protocol.json', plan)
    return plan
