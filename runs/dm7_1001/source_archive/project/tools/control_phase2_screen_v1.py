"""Bounded phase-2 screen; preserve failures and never silently restart training."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.phase1_checkpoint_diagnostics import read, write, sha, validate_report
OUT = ROOT/'results_phase2_runtime_v1'
CONTRACT = ROOT/'results_phase2_diagnosis_20260908/tuning_contract.json'


def jobs(contract):
    # Round-robin candidates/methods; every entered job runs, no outcome-based pruning.
    return [dict(method=m, scenario=s, candidate=c['id'], seed=seed,
                 id=f"{m}__{s}__{c['id']}__seed{seed}")
            for c in contract['candidates'] for s in contract['scenarios']
            for m in contract['methods'] for seed in contract['screen_training_seeds']]


def validate_finished(directory):
    status = read(directory/'status.json')
    if status['status']!='completed' or status['smoke']:
        raise ValueError('Not a completed screen cell')
    d = read(directory/'evaluation.json'); validate_report(d,100,10000)
    if d['checkpoint_sha256'] != sha(directory/'final_model.zip'):
        raise ValueError('Final checkpoint changed')
    if status['raw_steps']!=50000 or status['learner_updates']!=45001:
        raise ValueError('Budget mismatch')


def run_job(job, device):
    directory = OUT/'screen'/job['id']
    if directory.exists():
        try:
            validate_finished(directory)
            return dict(id=job['id'], status='completed_reused')
        except Exception as exc:
            return dict(id=job['id'],status='needs_recovery',error=str(exc))
    logs = OUT/'logs'; logs.mkdir(exist_ok=True)
    argv = [sys.executable,str(ROOT/'tools/run_phase2_cell_v1.py'), '--method',job['method'],
            '--scenario',job['scenario'],'--candidate',job['candidate'],'--seed',str(job['seed']), '--device',device]
    with (logs/(job['id']+'.stdout.log')).open('w') as stdout, (logs/(job['id']+'.stderr.log')).open('w') as stderr:
        process = subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr)
        write(logs/(job['id']+'.process.json'),dict(pid=process.pid,started_at=time.time(),argv=argv))
        code = process.wait()
    if code:
        return dict(id=job['id'],status='failed',returncode=code)
    try:
        validate_finished(directory)
    except Exception as exc:
        return dict(id=job['id'],status='invalid_result',error=str(exc))
    return dict(id=job['id'],status='completed')


def main(args):
    import psutil
    OUT.mkdir(exist_ok=True)
    lock = OUT/'controller.lock'
    if lock.exists():
        old = read(lock)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['create_time'])<1:
            raise RuntimeError('Controller is still alive')
        # An orphan worker must finish or be handled explicitly before a restart.
        for file in (OUT/'logs').glob('*.process.json'):
            d = read(file)
            if psutil.pid_exists(d['pid']):
                raise RuntimeError(f'Possible orphan worker {d["pid"]}; inspect before restarting')
        lock.unlink()
    with lock.open('x') as f:
        import json
        json.dump(dict(pid=os.getpid(),create_time=psutil.Process().create_time()),f)
    try:
        contract = read(CONTRACT); manifest = jobs(contract)
        assert len(manifest)==contract['screen_max_cells']==120
        preflight = read(OUT/'preflight_v1.json')
        assert preflight['passed'] and preflight['contract_sha256']==sha(CONTRACT)
        write(OUT/'screen_manifest.json',dict(jobs=manifest,contract_sha256=sha(CONTRACT),
            preflight_sha256=sha(OUT/'preflight_v1.json'),control_reuse=False,
            note='All controls freshly trained; historical code identity insufficient for reuse'))
        results=[]
        write(OUT/'controller_status.json',dict(status='running',pid=os.getpid(),workers=args.workers,
            completed=0,expected=len(manifest),results=[],updated_at=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_job,j,args.device) for j in manifest]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                write(OUT/'controller_status.json',dict(status='running',pid=os.getpid(),workers=args.workers,
                    completed=len(results),expected=len(manifest),results=results,updated_at=time.time()))
        ok=all(r['status'] in ('completed','completed_reused') for r in results)
        write(OUT/'controller_status.json',dict(status='completed' if ok else 'needs_attention',
            completed=len(results),expected=len(manifest),results=results,updated_at=time.time()))
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=2)
    parser.add_argument('--device',default='cuda')
    main(parser.parse_args())
