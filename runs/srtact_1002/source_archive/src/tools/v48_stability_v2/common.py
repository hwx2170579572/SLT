from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from tools.phase3_mechanism_v1.common import read, sha, digest, model_state_sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'r48s2'
DOC = ROOT / 'experiments/v48_stability_v2'
SCENES = ('cross', 'carla')
PROTOCOL = ROOT / 'experiments/independent_v2_five_methods_six_scenarios_100ep_v1/protocol.json'
BASE = dict(learning_rate=5e-5, final_learning_rate=5e-5, decay_start=20000,
            decay_end=50000, tau=.005, batch_size=32, buffer_size=20000)
CANDIDATES = {
    'lr_half': dict(BASE, label='Reference: LR 5e-5, tau .005', reuse='lr_half'),
    'tau_slow_decay': dict(BASE, tau=.0025, final_learning_rate=2.5e-5,
                           label='Tau .0025 + LR decay to 2.5e-5', reuse=None),
    'tau_slow_decay_highfloor': dict(BASE, tau=.0025, final_learning_rate=3.75e-5,
                           label='Tau .0025 + LR decay to 3.75e-5', reuse=None),
    'tau_slower': dict(BASE, tau=.001,
                       label='Tau .001 at LR 5e-5', reuse=None),
    'tau_slow_batch64': dict(BASE, tau=.0025, batch_size=64,
                             label='Tau .0025 + Batch 64', reuse=None),
}
DEPLOYMENT = 'exact_final_actor_deterministic'
# These three changes predate this task (2026-09-19). Full-method compatibility
# is checked against the byte-identical recovered old constructor in parity.
EXISTING_SOURCE_CHANGES = {
    'configs/sb3_configs_v4_5.py': ('f45f73135872cc88add56722dc7c11890194c7db9ad696613cd14113eb612a74',
                                 'a8fe90d96295d8d622ed31bca381d3500963319265607b8d204283d0a4216032'),
    'configs/sb3_configs_v4_8.py': ('0e6aaf2852fc3fced150fa62ff44ff60f842506da060cfc3fae1dd764fd769e0',
                                 'f7770ed7ef34b7845c1f60d4718c7e9d03004b181c4cf5b5258d54e5a0e06940'),
    'tools/run_latent_probes.py': ('eb82ec01cb0c2d1382119fce897b9c03377e5557fc30f809c8c3731d144093aa',
                                 '976b7e436f01f8db6c575820b3b19a96a5864a6a65d6a5491973a68caccc8cdc'),
}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def seal(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        write(path, value)


@contextmanager
def lock(path):
    """Fail fast on a live owner; recover only a demonstrably dead PID."""
    import psutil
    path = Path(path)
    path.resolve().relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = read(path)
        try:
            alive = abs(psutil.Process(owner['pid']).create_time() - owner['create_time']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError(f'Already running (PID {owner["pid"]}): {path}')
        path.unlink()
    with path.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time()), f)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def plan():
    return read(OUT / 'manifest.json')


def cell(scene, candidate):
    if scene not in SCENES or candidate not in CANDIDATES:
        raise ValueError('Outside the frozen Cross/CARLA seed-0 screen')
    return f'{scene}__{candidate}'


def train_path(scene, candidate, smoke=False):
    return OUT / ('smoke/train' if smoke else 'train') / cell(scene, candidate)


def source(scene, candidate):
    return plan()['sources'][cell(scene, candidate)]


def resolved_model(scene, candidate, smoke=False):
    spec = source(scene, candidate)
    if spec['reuse']:
        p = ROOT / spec['checkpoint']
        if sha(p) != spec['checkpoint_sha256']:
            raise ValueError('Existing model changed')
    else:
        directory = train_path(scene, candidate, smoke)
        receipt = read(directory / 'training_complete.json')
        if receipt['smoke'] != smoke or receipt['manifest_sha256'] != sha(OUT / 'manifest.json'):
            raise ValueError('Training receipt/protocol mismatch')
        p = directory / 'final_model.zip'
        if sha(p) != receipt['checkpoint_sha256']:
            raise ValueError('Completed model changed')
    return p


def make_env(scene, namespace, *, training=False):
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    spec = source(scene, 'lr_half')
    args = read(ROOT / spec['arguments'])['inherited_environment_arguments'].copy()
    args.update(seed=0, gui=False, evaluation_split='validation')
    factory = _make_environment_factory(adapter='v4_8', density=read(PROTOCOL)['scenarios'][scene],
                                        overlay_root=OUT / 'ov' / namespace)
    return factory(argparse.Namespace(**args), evaluation=not training)


def check_preflight():
    receipt = read(OUT / 'preflight.json')
    if not receipt['passed'] or receipt['manifest_sha256'] != sha(OUT / 'manifest.json'):
        raise ValueError('Missing/stale preflight; run preparation tests and seal first')
    for name, expected in receipt['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Code/protocol changed since validation: {name}')
    for spec in plan()['sources'].values():
        for key in ('checkpoint', 'training_log', 'arguments', 'training_complete'):
            if spec.get(key) and sha(ROOT / spec[key]) != spec[key + '_sha256']:
                raise ValueError(f'Reuse source changed: {spec[key]}')


def tracked_sources():
    result = dict(read(ROOT / 'results_phase2_runtime_v2/preflight_v2.json')['source_sha256'])
    phase3 = read(ROOT / 'r3m1/preflight.json')
    result.update(phase3['source_sha256'])
    for name, expected in result.items():
        actual = sha(ROOT / name)
        if actual != expected:
            bound = EXISTING_SOURCE_CHANGES.get(name.replace('\\', '/'))
            if bound != (expected, actual):
                raise ValueError(f'Unreviewed frozen implementation change: {name}')
            result[name] = actual
    legacy = OUT / 'evidence/legacy_sb3_configs_v4_5.py'
    if sha(legacy) != EXISTING_SOURCE_CHANGES['configs/sb3_configs_v4_5.py'][0]:
        raise ValueError('Recovered historical constructor is not byte-identical')
    paths = list((ROOT / 'tools/v48_stability_v2').glob('*.py')) + list(DOC.glob('*'))
    paths += [ROOT / 'tests_sb3_sumo/test_v48_stability_v2.py',
              ROOT / 'run_v48_stability_v2.ps1', PROTOCOL,
              ROOT / 'tools/report_phase2_curves_reference_style_v2.py',
              ROOT / 'visual-composer/plot_training_curves_v5_episode20_ema999.py',
              ROOT / 'tools/phase3_mechanism_v1/common.py', legacy]
    result.update({relative(p): sha(p) for p in paths if p.is_file()})
    return result
