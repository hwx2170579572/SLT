"""Inventory completed prior models, including Windows long paths."""
from __future__ import annotations

import os
from pathlib import Path

from .common import ROOT, OUT, read, relative, sha, write


def scan():
    roots = sorted(p for p in ROOT.iterdir() if p.is_dir() and
                   (p.name.startswith('results') or p.name in ('r413', 'r412', 'r411', 'r3m1')))
    rows, errors = [], []
    scanned = 0
    for root in roots:
        path = '\\\\?\\' + str(root) if os.name == 'nt' else str(root)
        for directory, children, files in os.walk(path, onerror=lambda e: errors.append(str(e))):
            children[:] = [n for n in children if not n.startswith(('pytest', 'tmp', 'overlays', 'tb_'))
                           and n not in ('ov', 'tb', 'checkpoints', '__pycache__', '.git', 'traffic')]
            if 'arguments.json' not in files:
                continue
            scanned += 1
            p = Path(directory) / 'arguments.json'
            try:
                data = read(p)
                requested = data.get('requested_raw_steps', data.get('inherited_environment_arguments', data))
                label = str(requested.get('algo', data.get('method', '')))
                if not ('v4_13' in label or data.get('phase3_variant')):
                    continue
                row = {
                    'arguments': relative(p), 'arguments_sha256': sha(p), 'algorithm': label,
                    'seed': requested.get('seed', data.get('seed')),
                    'scenario': requested.get('scenario', data.get('scenario')),
                    'raw_budget': data.get('raw_budget', requested.get('max_steps')),
                    'phase3_variant': data.get('phase3_variant'),
                    'final_model_exists': (p.parent / 'final_model.zip').exists(),
                    'candidate': data.get('candidate'),
                }
                if row['final_model_exists']:
                    row['final_model_sha256'] = sha(p.parent / 'final_model.zip')
                rows.append(row)
            except (OSError, ValueError, TypeError) as exc:
                errors.append({'path': relative(p), 'error': repr(exc)})
    result = {'scanned_arguments': scanned, 'rows': rows, 'errors': errors,
              'scan_roots': [relative(p) for p in roots]}
    write(OUT / 'inventory.json', result)
    if errors:
        raise ValueError(f'Inventory incomplete: {len(errors)} errors')
    return result
