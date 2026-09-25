"""Prepare, validate or manually launch the two-GPU-process stability screen."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import time

from .common import ROOT, OUT, SCENES, CANDIDATES, read, write, sha, cell, lock, check_preflight, tracked_sources


def controller(smoke=False):
    if not smoke:
        check_preflight()
    with lock(OUT / ('smoke/controller.lock' if smoke else 'controller.lock')):
        import torch
        import psutil
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; this plan has no CPU fallback')
        torch.empty(1, device='cuda')
        free, total = torch.cuda.mem_get_info()
        torch.cuda.empty_cache()
        print(f'2 GPU worker processes on {torch.cuda.get_device_name(0)}; free {free / 2**30:.2f} GiB', flush=True)
        jobs = [(scene, candidate) for candidate in CANDIDATES for scene in SCENES]
        start = time.time()
        output = OUT / ('smoke' if smoke else '')
        (output / 'logs').mkdir(parents=True, exist_ok=True)
        running, done = {}, {}
        stop_requested = threading.Event()

        def execute(job):
            if stop_requested.is_set():
                return dict(exit_code=-1, log=None, cancelled=True)
            scene, candidate = job
            name = cell(scene, candidate)
            command = [sys.executable, '-m', 'tools.v48_stability_v3.cli', 'cell', '--scene', scene, '--candidate', candidate]
            if smoke:
                command.append('--smoke')
            stamp = time.strftime('%Y%m%d_%H%M%S')
            log = output / 'logs' / f'{name}_{stamp}.log'
            with log.open('w', encoding='utf-8') as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                running[name] = process
                returncode = process.wait()
                running.pop(name, None)
            return dict(exit_code=returncode, log=str(log), completed_at=time.time())

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        futures = {pool.submit(execute, job): cell(*job) for job in jobs}
        try:
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                done[name] = future.result()
                print(name, done[name]['exit_code'], flush=True)
                write(output / 'controller_progress.json', dict(done=done, total=len(jobs), updated_at=time.time()))
        except BaseException:
            stop_requested.set()
            for future in futures:
                future.cancel()
            # Stop only descendants launched by this controller. Preserve all
            # partial training files and completed evaluation episodes.
            for process in list(running.values()):
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        receipt = dict(passed=all(v['exit_code'] == 0 for v in done.values()), jobs=done, workers=2, smoke=smoke,
                       source_sha256=tracked_sources(), wall_seconds=time.time() - start)
        write(output / 'controller.json', receipt)
        if not smoke:
            from .report import build_report
            build_report()
        if not receipt['passed']:
            raise RuntimeError('Some jobs failed; inspect logs. Partial training was not restarted.')
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'test', 'parity', 'smoke', 'seal', 'check', 'run', 'cell', 'report', 'status'))
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--candidate', choices=tuple(CANDIDATES))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        from .plan import prepare
        result = prepare()
        print(f"Prepared: {result['new_training_cells']} new training cells + "
              f"{result['reused_training_cells']} reused models. No formal jobs started.")
    elif args.command == 'test':
        output = OUT / 'tests.log'
        OUT.mkdir(exist_ok=True)
        # System Temp\pytest-of-<user> and .pytest_cache can end up permission-locked
        # on Windows after SUMO subprocesses; isolate both under a local writable dir.
        with output.open('w', encoding='utf-8') as handle:
            code = subprocess.call([sys.executable, '-m', 'pytest', 'tests_sb3_sumo/test_v48_stability_v3.py', '-q',
                                    '-p', 'no:cacheprovider', '--basetemp', str(OUT / 'pytest_basetemp'),
                                    '--junitxml=' + str(OUT / 'tests.xml')], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        write(OUT / 'tests.json', dict(passed=code == 0, source_sha256=tracked_sources(), log_sha256=sha(output)))
        print(output.read_text(encoding='utf-8', errors='replace'))
        if code:
            raise SystemExit(code)
    elif args.command == 'parity':
        from .validation import parity_checks
        parity_checks()
        print('Training equivalence and actor-only telemetry invariance passed.')
    elif args.command in ('smoke', 'run'):
        controller(smoke=args.command == 'smoke')
    elif args.command == 'seal':
        from .validation import seal_preflight
        seal_preflight()
        print('Preflight sealed. Formal experiments await the user manual launch.')
    elif args.command == 'check':
        check_preflight()
        print('Preflight valid; no jobs launched.')
    elif args.command == 'cell':
        if not args.scene or not args.candidate:
            parser.error('cell requires --scene and --candidate')
        from .jobs import run_cell
        print(run_cell(args.scene, args.candidate, smoke=args.smoke))
    elif args.command == 'report':
        from .report import build_report
        result = build_report()
        print('Report complete:', result['complete'])
    elif args.command == 'status':
        for name in ('controller_progress.json', 'controller.json'):
            path = OUT / name
            print(name, read(path) if path.exists() else 'not started')


if __name__ == '__main__':
    main()
