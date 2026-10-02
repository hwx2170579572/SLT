"""Start the phase-4 controller in a detached, hidden Windows process.

The controller must outlive the shell that starts it, so it is launched with
DETACHED_PROCESS + CREATE_NO_WINDOW and its own log files, and a launch record
is written next to the run directory for auditing.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

from .common import OUT, ROOT, relative, seal, sha


def main():
    if os.name != 'nt':
        raise RuntimeError('this detached launcher is intentionally Windows-only')
    if not (OUT / 'preflight.json').exists():
        raise RuntimeError('Seal the phase-4 preflight before launching')
    logs = OUT / 'launcher_logs'
    logs.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    stdout_path = logs / f'controller_{timestamp}.stdout.log'
    stderr_path = logs / f'controller_{timestamp}.stderr.log'
    command = [sys.executable, '-m', 'tools.phase4_multiseed_v1.cli', 'run']
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    creation_flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                      | subprocess.CREATE_NO_WINDOW)
    with stdout_path.open('wb') as stdout_handle, stderr_path.open('wb') as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stdout_handle, stderr=stderr_handle, close_fds=True,
                                   creationflags=creation_flags)
    record = {'schema': 'phase4-multiseed-launcher-v1',
              'started_at_local': dt.datetime.now().astimezone().isoformat(),
              'pid': int(process.pid), 'command': command, 'working_directory': str(ROOT),
              'stdout_log': relative(stdout_path), 'stderr_log': relative(stderr_path),
              'preflight_sha256': sha(OUT / 'preflight.json'),
              'manifest_sha256': sha(OUT / 'manifest.json'),
              'gpu_workers': 3, 'hidden_window': True, 'detached_process': True}
    seal(OUT / 'launch_record.json', record)
    return record


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
