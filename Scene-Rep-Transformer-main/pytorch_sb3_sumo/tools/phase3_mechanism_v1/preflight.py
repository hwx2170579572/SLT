"""Require targeted tests and real CUDA/SUMO smoke before scientific execution."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

from .common import ROOT, OUT, PLAN_DIR, read, relative, seal, sha, validate_frozen_sources


def prepare_preflight():
    previous = validate_frozen_sources()
    inventory = read(OUT/'inventory.json')
    if inventory['errors']:
        raise ValueError('Inventory still has scan errors')
    junit = OUT/'tests_junit.xml'
    tree = ET.parse(junit).getroot()
    cases = tree.findall('.//testcase')
    if len(cases) < 13 or tree.findall('.//failure') or tree.findall('.//error'):
        raise ValueError('Targeted tests incomplete or failed')
    smoke = []
    for name in ('g0_s025__cross','g1_s1__roundabout_medium','g0_s1_lowall__carla'):
        path = OUT/'smoke_train'/name/'training_complete.json'
        value = read(path)
        if not value['smoke'] or value['raw_steps'] != 72 or not value['tensor_roundtrip_equal']:
            raise ValueError('Training smoke failed')
        if sha(path.parent/'final_model.zip') != value['checkpoint_sha256']:
            raise ValueError('Smoke weights changed')
        smoke.append(path)
    cross = {}
    for path in (OUT/'smoke_eval').glob('*/result.json'):
        value = read(path)
        if not value['identity']['smoke'] or not value['tensor_state_unchanged']:
            raise ValueError('Invalid evaluation smoke')
        if value['identity']['scenario']=='cross' and value['identity']['mode']=='selection':
            ep = read(path.parent/'episode_000.json')
            cross[value['identity']['method']] = [f['sha256'] for f in ep['traffic']['files']]
        smoke.append(path)
    if cross.get('mst_slt') != cross.get('v4_13') or not cross.get('mst_slt'):
        raise ValueError('Baseline/v4 fresh traffic pairing smoke missing')
    gradients = list((OUT/'smoke_gradients').glob('*.json'))
    if not gradients:
        raise ValueError('Real gradient diagnostic smoke missing')
    for path in gradients:
        if not read(path)['tensor_state_unchanged']:
            raise ValueError('Gradient smoke changed model')
        smoke.append(path)
    source_paths = list((ROOT/'tools/phase3_mechanism_v1').glob('*.py'))
    source_paths += [ROOT/'tests_sb3_sumo/test_phase3_mechanism_v1.py',
                     PLAN_DIR/'PLAN.md', PLAN_DIR/'protocol.json', OUT/'diagnostic_sources.json']
    value = {'schema':'phase3-preflight-v1','passed':True,'sealed_at':time.time(),
             'manifest_sha256':sha(OUT/'manifest.json'),
             'source_sha256':{relative(p):sha(p) for p in source_paths},
             'prior_frozen_source_verification':previous,
             'targeted_tests':len(cases),'junit':{'path':relative(junit),'sha256':sha(junit)},
             'smoke_artifacts':{relative(p):sha(p) for p in smoke},
             'fresh_traffic_pairing_verified':True,'inventory_errors':0,
             'gpu_workers':3,'cpu_workers':0,'multiseed_confirmation_deferred':True}
    path = OUT/'preflight.json'
    if path.exists():
        old = read(path)
        value['sealed_at'] = old['sealed_at']
    seal(path,value)
    return value


if __name__=='__main__':
    import json
    value = prepare_preflight()
    print(json.dumps({'passed':value['passed'],'tests':value['targeted_tests'],
                      'source_files':len(value['source_sha256'])}))
