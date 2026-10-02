"""Dependency-aware three-GPU queue, with reuse receipts and resumable evals."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .common import (ROOT, OUT, SCENES, MECHANISM_SCENES, STEPS, digest, read,
                     relative, seal, sha, source_registry, write)
from .model import VARIANTS


class AdoptedProcess:
    def __init__(self, pid, create_time):
        self.pid, self.create_time = pid, create_time

    def poll(self):
        try:
            if psutil.pid_exists(self.pid) and abs(psutil.Process(self.pid).create_time()-self.create_time)<1:
                return None
        except psutil.NoSuchProcess:
            pass
        return 0  # Artifact validation determines success for an adopted worker.


def build_jobs(manifest, sources):
    jobs = {}

    def add(kind, source, *, step=None, mode=None, decoder=None, dependencies=(), priority=50):
        spec = {'kind':kind, 'source':source, 'step':step, 'mode':mode, 'decoder':decoder,
                'dependencies':list(dependencies), 'priority':priority}
        key = kind + '_' + digest(spec)[:16]
        jobs[key] = {'id':key, **spec}
        return key

    trains = {j['id']:add('train',j['id'],priority=25) for j in manifest['training_jobs']}
    banks = {}
    for scene in MECHANISM_SCENES:
        source = f'v4_13__{scene}__tau_half'
        banks[scene] = add('eval', source, step=50000, mode='mechanism', decoder='actor_stochastic', priority=5)
    for key, source in sources.items():
        dependency = [trains[key]] if source['new_training'] else []
        add('eval', key, step=50000, mode='legacy', decoder='native', dependencies=dependency, priority=8)
        if source['new_training']:
            add('eval', key, step=50000, mode='score', decoder='native', dependencies=dependency, priority=32)
            continue
        if source.get('diagnostic_only'):
            continue
        validations = [add('eval', key, step=step, mode='selection', decoder='native', priority=50)
                       for step in STEPS]
        selection = add('select', key, dependencies=validations, priority=12)
        add('eval', key, step='selected', mode='score', decoder='native', dependencies=[selection], priority=20)
        add('eval', key, step=50000, mode='score', decoder='native', dependencies=[selection], priority=21)
        if source.get('historical_selected'):
            historical = 50000 if source['historical_selected_kind'] == 'exact_final' else 'historical'
            add('eval', key, step=historical, mode='score', decoder='native', dependencies=[selection], priority=22)
        if (source['method'],source['candidate']) in (('mst_slt','control'),('v4_8','lr_half'),('v4_13','tau_half')):
            for mode in ('robust_lower','robust_higher'):
                add('eval', key, step='selected', mode=mode, decoder='native', dependencies=[selection], priority=60)

    for scene in MECHANISM_SCENES:
        keys = [f'v4_13__{scene}__tau_half',f'v4_8__{scene}__lr_half']
        if scene == 'carla':
            keys.extend(['v4_8__carla__tau_half','v4_8__carla__tau_double'])
        for key in keys:
            source = sources[key]
            decoders = ['actor_deterministic','actor_stochastic']
            if source['method'] == 'v4_13':
                decoders += ['no_risk_score','no_support_prior']
            elif source['decoder'] != 'target_critic':
                decoders.append('target_only')
            for decoder in decoders:
                add('eval',key,step=50000,mode='legacy',decoder=decoder,priority=10)
            for decoder in ('native','actor_stochastic'):
                add('eval',key,step=50000,mode='legacy_train',decoder=decoder,priority=15)
        for variant in VARIANTS:
            key = f'v4_13__{scene}__tau_half' if variant == 'g1_s025' else f'{variant}__{scene}'
            deps = [banks[scene]] + ([trains[key]] if key in trains else [])
            for step in (10000,30000,50000):
                add('gradient',key,step=step,dependencies=deps,priority=35)
    return jobs


def command_for(job, sources, completed):
    command = [sys.executable,'-m','tools.phase3_mechanism_v1.cli',job['kind'],'--source',job['source']]
    if job['kind'] == 'train':
        return command
    step = job['step']
    if step == 'selected':
        step = read(OUT / 'selection' / f'{job["source"]}.json')['selected_step']
    command += ['--step',str(step)]
    if job['kind'] == 'gradient':
        bank = completed[job['dependencies'][0]]['artifact']
        command += ['--bank',bank]
    else:
        command += ['--mode',job['mode'],'--decoder',job['decoder']]
    return command


def run():
    from .training import check_preflight
    check_preflight()
    sources = source_registry()
    manifest = read(OUT / 'manifest.json')
    jobs = build_jobs(manifest, sources)
    seal(OUT / 'queue_manifest.json', jobs)
    lock = OUT / 'controller.lock'
    if lock.exists():
        old = read(lock)
        if psutil.pid_exists(old['pid']):
            try:
                if abs(psutil.Process(old['pid']).create_time() - old['create_time']) < 1:
                    raise RuntimeError('Another phase3 controller is running')
            except psutil.NoSuchProcess:
                pass
        lock.replace(OUT / f'controller_stale_{int(time.time())}.json')
    seal(lock, {'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    handles, completed, failed, attempts = {}, {}, {}, {}
    started = time.time()
    try:
        for key, job in jobs.items():
            receipt_path = OUT / 'jobs' / f'{key}.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt['job'] != job or sha(ROOT / receipt['artifact']) != receipt['artifact_sha256']:
                    raise ValueError(f'Completed job evidence changed: {key}')
                completed[key] = receipt
            process_path = OUT / 'processes' / f'{key}.json'
            if key not in completed and process_path.exists():
                process = read(process_path)
                if process['job'] != job:
                    raise ValueError('Existing worker belongs to a different job specification')
                adopted = AdoptedProcess(process['pid'],process['create_time'])
                if adopted.poll() is None:
                    handles[key] = (adopted,None,None)
                    attempts[key] = process['attempt']
        while len(completed) + len(failed) < len(jobs):
            for key,(process,out,err) in list(handles.items()):
                code = process.poll()
                if code is None:
                    continue
                if out is not None:
                    out.close()
                if err is not None:
                    err.close()
                job = jobs[key]
                try:
                    if code:
                        raise RuntimeError(f'Worker exited {code}')
                    lines = (OUT/'logs'/f'{key}.stdout.log').read_text(encoding='utf-8-sig').splitlines()
                    artifact = json.loads(next(line for line in reversed(lines) if line.strip()))
                    if not isinstance(artifact,str) or not (ROOT/artifact).is_file():
                        raise ValueError('Worker did not return an existing artifact')
                    receipt = {'job':job,'artifact':artifact,'artifact_sha256':sha(ROOT/artifact),
                               'completed_at':time.time(),'attempts':attempts[key]}
                    seal(OUT/'jobs'/f'{key}.json',receipt)
                    completed[key] = receipt
                except Exception as exc:
                    if attempts[key] < 2 and job['kind'] in ('eval','gradient'):
                        # Evaluation resumes only missing episodes; completed episodes are immutable.
                        write(OUT/'retries'/f'{key}_{attempts[key]}.json',{'error':repr(exc),'time':time.time()})
                    else:
                        failed[key] = {'job':job,'error':repr(exc)}
                        write(OUT/'failures'/f'{key}.json',failed[key])
                del handles[key]

            for key,job in jobs.items():
                if key in completed or key in handles or key in failed:
                    continue
                bad = [dep for dep in job['dependencies'] if dep in failed]
                if bad:
                    failed[key] = {'job':job,'blocked_by':bad}

            ready = [job for key,job in jobs.items() if key not in completed and key not in handles and key not in failed
                     and all(dep in completed for dep in job['dependencies'])]
            for job in list(ready):
                if job['kind'] != 'select':
                    continue
                from .evaluation import select
                select(sources[job['source']], [completed[dep]['artifact'] for dep in job['dependencies']])
                artifact = f'r3m1/selection/{job["source"]}.json'
                receipt = {'job':job,'artifact':artifact,'artifact_sha256':sha(ROOT/artifact),'completed_at':time.time()}
                seal(OUT/'jobs'/f'{job["id"]}.json',receipt)
                completed[job['id']] = receipt
                ready.remove(job)

            paused = (OUT/'pause_new_jobs.json').exists()
            while ready and len(handles) < 3 and not paused:
                running_train = sum(jobs[key]['kind'] == 'train' for key in handles)
                scientific_other_ready = any(job['kind'] != 'train' for job in ready)
                eligible = [job for job in ready if job['kind'] != 'train' or running_train < (2 if scientific_other_ready else 3)]
                if not eligible:
                    break
                # Keep training progressing alongside shorter diagnostic jobs.
                training_ready = [job for job in eligible if job['kind'] == 'train']
                job = min(training_ready or eligible, key=lambda j:(j['priority'],j['id']))
                ready.remove(job)
                key = job['id']
                attempts[key] = attempts.get(key,0) + 1
                logs = OUT/'logs'; logs.mkdir(exist_ok=True)
                stdout = (logs/f'{key}.stdout.log').open('a',encoding='utf-8')
                stderr = (logs/f'{key}.stderr.log').open('a',encoding='utf-8')
                command = command_for(job,sources,completed)
                env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
                process = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=env,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handles[key] = (process,stdout,stderr)
                write(OUT/'processes'/f'{key}.json',{'pid':process.pid,'create_time':psutil.Process(process.pid).create_time(),
                                                   'command':command,'job':job,'attempt':attempts[key]})
            status = {'status':'running','pid':os.getpid(),'expected':len(jobs),'completed':len(completed),
                      'failed_or_blocked':len(failed),'active':[{'pid':v[0].pid,**jobs[k]} for k,v in handles.items()],
                      'updated_at':time.time(),'started_at':started,'paused_new_jobs':paused,
                      'by_kind':{kind:{'total':sum(j['kind']==kind for j in jobs.values()),
                                       'completed':sum(jobs[k]['kind']==kind for k in completed)}
                                 for kind in ('train','eval','gradient','select')}}
            write(OUT/'status.json',status)
            if not handles and not ready and len(completed)+len(failed)<len(jobs):
                # New scoring jobs become eligible after synchronous selection; loop once more.
                continue
            if len(completed)+len(failed)<len(jobs):
                time.sleep(5)
        final = {'status':'completed' if not failed else 'needs_attention','expected':len(jobs),
                 'completed':len(completed),'failed_or_blocked':failed,'updated_at':time.time()}
        write(OUT/'status.json',final)
        from .analyze import analyze
        analyze()
        return final
    finally:
        lock.unlink(missing_ok=True)
