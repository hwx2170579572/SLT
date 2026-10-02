"""Dispatch only inventory-confirmed missing variants, using two CUDA slots plus two CPU slots."""
import sys
import subprocess
import time
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase1_checkpoint_diagnostics import read,write,sha
from tools.control_phase2_screen_v2 import validate_finished
OUT=ROOT/'results_phase2_reuse_v3'
RUNTIME=ROOT/'results_phase2_runtime_v2'


def missing_jobs(inventory):
    if inventory['scan_errors']:raise ValueError('Resolve inventory errors before new experiments')
    jobs=inventory['proposed_missing_candidates']
    if any(j['candidate']=='control' or j['method'] not in ('v4_8','v4_13') for j in jobs):raise ValueError('Repeated controls or out-of-scope method in missing queue')
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate jobs')
    return jobs


def live_workers():
    import psutil
    result=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if any(str(ROOT/'tools/run_phase2_cell_v2.py').lower()==x.lower() for x in cmd):
                values={flag:cmd[cmd.index('--'+flag)+1] for flag in ('method','scenario','candidate','seed')}
                jobid=f"{values['method']}__{values['scenario']}__{values['candidate']}__seed{values['seed']}"
                result.append(dict(pid=p.pid,create_time=p.create_time(),jobid=jobid,device=cmd[cmd.index('--device')+1] if '--device' in cmd else 'cuda'))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def available_device(workers):
    if sum(w['device'] != 'cpu' for w in workers) < 2:
        return 'cuda'
    if sum(w['device'] == 'cpu' for w in workers) < 2:
        return 'cpu'
    return None


def adopt_running(pending, workers):
    ids={j['id'] for j in pending}
    adopted={w['jobid']:w for w in workers if w['jobid'] in ids}
    return [j for j in pending if j['id'] not in adopted],adopted


def main():
    import psutil,json
    lock=OUT/'controller_mixed_v5.lock'
    # Exclusive ownership. Stale locks are left for explicit inspection.
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),create_time=psutil.Process().create_time()),f)
    handles={};results=[];adopted={}
    try:
        inv=read(OUT/'inventory.json');pending=list(missing_jobs(inv))
        write(OUT/'execution_manifest.json',dict(jobs=pending,historical_controls=inv['controls'],inventory_sha256=sha(OUT/'inventory.json'),rule='No repeated control runs; historical evidence kept separately attributed'))
        if not (RUNTIME/'pause_new_jobs.json').exists():raise ValueError('Old all-control queue must remain paused')
        pending,adopted=adopt_running(pending,live_workers())
        while pending or handles or adopted:
            for name,w in list(adopted.items()):
                if psutil.pid_exists(w['pid']) and abs(psutil.Process(w['pid']).create_time()-w['create_time'])<1:continue
                try:validate_finished(RUNTIME/'screen'/name);status='completed_adopted';error=None
                except Exception as exc:status='needs_attention';error=str(exc)
                results.append(dict(id=name,status=status,error=error));del adopted[name]
            for name,(process,stdout,stderr) in list(handles.items()):
                code=process.poll()
                if code is None:continue
                stdout.close();stderr.close()
                try:
                    if code:raise RuntimeError(f'worker exit {code}')
                    validate_finished(RUNTIME/'screen'/name);status='completed';error=None
                except Exception as exc:status='needs_attention';error=str(exc)
                results.append(dict(id=name,status=status,error=error));del handles[name]
            workers=live_workers()
            while pending and available_device(workers) is not None and not (OUT/'pause_new_jobs.json').exists():
                device=available_device(workers)
                j=pending.pop(0);directory=RUNTIME/'screen'/j['id']
                if directory.exists():
                    try:validate_finished(directory);status='completed_reused'
                    except Exception:status='needs_recovery_no_automatic_retraining'
                    results.append(dict(id=j['id'],status=status));continue
                logs=OUT/'logs';logs.mkdir(exist_ok=True)
                stdout=(logs/(j['id']+'.stdout.log')).open('w');stderr=(logs/(j['id']+'.stderr.log')).open('w')
                argv=[sys.executable,str(ROOT/'tools/run_phase2_cell_v2.py'),'--method',j['method'],'--scenario',j['scenario'],'--candidate',j['candidate'],'--seed','0','--device',device]
                process=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr)
                write(RUNTIME/'logs'/(j['id']+'.process.json'),dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time(),argv=argv,dispatcher='mixed_v5',device=device))
                handles[j['id']]=(process,stdout,stderr);workers.append(dict(pid=process.pid,device=device))
            write(OUT/'controller_mixed_v5_status.json',dict(status='running' if pending or handles or adopted else 'finished',pid=os.getpid(),
                pending=len(pending),active=list(handles),adopted_running=list(adopted),all_live_workers=workers,results=results,updated_at=time.time()))
            if pending or handles or adopted:time.sleep(10)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
