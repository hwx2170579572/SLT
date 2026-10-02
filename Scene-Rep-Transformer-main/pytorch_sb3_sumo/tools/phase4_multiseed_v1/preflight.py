"""Require targeted tests and real CUDA/SUMO smoke before scientific execution.

The smoke set is chosen to exercise exactly what is new in this phase: training
at non-zero seeds for all three methods, and the selection/score evaluation
paths for the reused seed-0 evidence.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

from .common import OUT, PLAN_DIR, ROOT, read, relative, seal, sha, validate_frozen_sources

SMOKE_TRAIN = ('mst_slt__left_turn__control__seed1',
               'v4_8__left_turn__lr_half__seed1',
               'v4_13__roundabout_medium__tau_half__seed2')
SMOKE_EVAL = {'selection': 'v4_8__left_turn__lr_half__seed0',
              'score': 'v4_13__left_turn__tau_half__seed0'}
MINIMUM_TESTS = 14


def prepare_preflight():
    previous = validate_frozen_sources()
    junit = OUT / 'tests_junit.xml'
    tree = ET.parse(junit).getroot()
    cases = tree.findall('.//testcase')
    if len(cases) < MINIMUM_TESTS or tree.findall('.//failure') or tree.findall('.//error'):
        raise ValueError('Targeted tests incomplete or failed')

    smoke = []
    for name in SMOKE_TRAIN:
        path = OUT / 'smoke_train' / name / 'training_complete.json'
        value = read(path)
        if not value['smoke'] or value['raw_steps'] != 72 or not value['policy_tensor_roundtrip_equal']:
            raise ValueError(f'Training smoke failed: {name}')
        if value['learner_updates'] != 72 - 48 + 1:
            raise ValueError(f'Smoke update accounting differs: {name}')
        if sha(path.parent / 'final_model.zip') != value['checkpoint_sha256']:
            raise ValueError(f'Smoke weights changed: {name}')
        smoke.append(path)

    seen_modes = {}
    for path in (OUT / 'smoke_eval').glob('*/result.json'):
        value = read(path)
        if not value['identity']['smoke'] or not value['tensor_state_unchanged']:
            raise ValueError('Invalid evaluation smoke')
        smoke.append(path)
        seen_modes.setdefault(value['identity']['mode'], set()).add(value['source_id'])
    for mode, expected in SMOKE_EVAL.items():
        if expected not in seen_modes.get(mode, set()):
            raise ValueError(f'Missing {mode} evaluation smoke for {expected}')

    source_paths = list((ROOT / 'tools/phase4_multiseed_v1').glob('*.py'))
    source_paths += [ROOT / 'tests_sb3_sumo/test_phase4_multiseed_v1.py',
                     PLAN_DIR / 'SCOPE.md', PLAN_DIR / 'protocol.json']
    value = {'schema': 'phase4-preflight-v1', 'passed': True, 'sealed_at': time.time(),
             'manifest_sha256': sha(OUT / 'manifest.json'),
             'source_sha256': {relative(p): sha(p) for p in source_paths},
             'prior_frozen_source_verification': previous,
             'targeted_tests': len(cases),
             'junit': {'path': relative(junit), 'sha256': sha(junit)},
             'smoke_artifacts': {relative(p): sha(p) for p in smoke},
             'smoke_training_cells': list(SMOKE_TRAIN), 'smoke_evaluation_modes': sorted(seen_modes),
             'gpu_workers': 3, 'cpu_workers': 0, 'multi_training_seed_confirmation': True}
    path = OUT / 'preflight.json'
    if path.exists():
        value['sealed_at'] = read(path)['sealed_at']
    seal(path, value)
    return value


if __name__ == '__main__':
    import json
    result = prepare_preflight()
    print(json.dumps({'passed': result['passed'], 'tests': result['targeted_tests'],
                      'source_files': len(result['source_sha256'])}))
