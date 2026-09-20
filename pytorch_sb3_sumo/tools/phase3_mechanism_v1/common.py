from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'r3m1'
PLAN_DIR = ROOT / 'experiments/phase3_mechanism_v1'
SCENES = ('left_turn', 'cross', 'roundabout_easy', 'roundabout_medium', 'roundabout', 'carla')
MECHANISM_SCENES = ('cross', 'roundabout_medium', 'carla')
STEPS = (10000, 20000, 30000, 40000, 50000)
PROTOCOL = ROOT / 'experiments/independent_v2_five_methods_six_scenarios_100ep_v1/protocol.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def seal(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable artifact differs: {path}')
        return
    write(path, value)


def relative(path):
    text = str(path)
    if text.startswith('\\\\?\\'):
        text = text[4:]
    return Path(text).resolve().relative_to(ROOT).as_posix()


def validate_frozen_sources():
    receipt = read(ROOT / 'results_phase2_runtime_v2/preflight_v2.json')
    for path, expected in receipt['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen prior source changed: {path}')
    return {'source_files': len(receipt['source_sha256']), 'passed': True}


def model_state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def source_registry():
    sources = dict(read(OUT / 'manifest.json')['sources'])
    path = OUT / 'diagnostic_sources.json'
    if path.exists():
        sources.update(read(path)['sources'])
    return sources


@contextmanager
def exclusive_lock(path):
    import psutil
    path = Path(path)
    path.resolve().relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump({'pid':os.getpid(),'create_time':psutil.Process().create_time()},handle)
            break
        except FileExistsError:
            try:
                owner = read(path)
                alive = psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['create_time']) < 1
                if not alive:
                    path.unlink(missing_ok=True)
                    continue
            except (ValueError, KeyError, FileNotFoundError, psutil.NoSuchProcess):
                pass
            time.sleep(1)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
